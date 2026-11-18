import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps
from scipy.ndimage import find_objects

from data import letterbox, normalize, overlay_regions
from post import instances_from_logits

IMAGE_TYPES = {'.jpg', '.jpeg', '.png'}


class Predictor:
    def __init__(self, weights, variant, threads):
        self.weights = Path(weights)
        self.manifest = json.loads((self.weights / 'manifest.json').read_text())
        self.variant = self.manifest['variants'][variant]
        self.classes = self.manifest['classes']
        self.size = self.variant['size']
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        cv2.setNumThreads(1)
        self.segmenter = ort.InferenceSession(str(self.weights / self.variant['file']), options,
                                              providers=['CPUExecutionProvider'])
        self.classifier = ort.InferenceSession(str(self.weights / self.manifest['classifier']), options,
                                               providers=['CPUExecutionProvider'])

    def prep_crop(self, image, mask):
        settings = self.manifest['classifier_preprocessing']
        size = self.manifest['classifier_size']
        if settings['mode'] == 'day1':
            rgb = np.asarray(Image.fromarray(image).resize((size, size), Image.Resampling.BILINEAR),
                             dtype=np.float32) / 255
            mean = np.asarray(settings['mean'], np.float32)
            std = np.asarray(settings['std'], np.float32)
            return np.ascontiguousarray(((rgb - mean) / std).transpose(2, 0, 1))
        patch = image.copy()
        patch[~mask] = 114
        return normalize(letterbox(patch, size)[0])

    def predict(self, image):
        labels, classes, _ = self.predict_instances(image)
        return labels, classes

    def predict_instances(self, image):
        canvas, box = letterbox(image, self.size)
        logits = self.segmenter.run(None, {'image': normalize(canvas)[None]})[0][0]
        labels = instances_from_logits(logits, box, image.shape[:2], self.variant.get('watershed', self.manifest['watershed']))
        classes = []
        regions = [(i, box, labels[box] == i) for i, box in enumerate(find_objects(labels), 1) if box is not None]
        for start in range(0, len(regions), 16):
            crops = [self.prep_crop(image[box], mask) for _, box, mask in regions[start:start+16]]
            scores = self.classifier.run(None, {'image': np.stack(crops)})[0]
            classes.extend(self.classes[i] for i in scores.argmax(1))
        return labels, classes, regions


def polygon(mask, offset=(0, 0)):
    pixels = mask.astype(np.uint8)
    contours, _ = cv2.findContours(pixels, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 1:
        raise ValueError('Expected one exterior contour')

    points = contours[0].reshape(-1, 2)
    if len(points) < 3 or cv2.contourArea(contours[0]) == 0:
        raise ValueError('Expected a polygon with at least three distinct points and nonzero area')

    filled = np.zeros_like(pixels)
    cv2.drawContours(filled, contours, 0, 1, cv2.FILLED)
    if not np.array_equal(filled, pixels):
        raise ValueError('Polygon does not reproduce the instance mask')
    return (points + offset).tolist()


def exterior(mask):
    count, components, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count == 1:
        return None
    largest = 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())
    contours, _ = cv2.findContours(np.uint8(components == largest), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if len(contours) != 1 or cv2.contourArea(contours[0]) == 0:
        return None
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, 0, 1, cv2.FILLED)
    return filled


def repair_mask(mask, forbidden):
    pixels = mask.astype(np.uint8)
    filled = exterior(pixels)
    while filled is not None:
        conflicts = filled.astype(bool) & forbidden
        count = np.count_nonzero(conflicts)
        if count == 0:
            return filled.astype(bool)
        y, x = np.argwhere(conflicts)[0]
        # Open a hole containing another instance with a narrow cut to the exterior.
        slices = (np.s_[y, :x+1], np.s_[y, x:], np.s_[:y+1, x], np.s_[y:, x])
        best = best_rank = None
        for cut in slices:
            trial = pixels.copy()
            trial[cut] = 0
            candidate = exterior(trial)
            if candidate is not None:
                overlap = np.count_nonzero(candidate & forbidden)
                rank = (overlap, -np.count_nonzero(candidate & mask))
                if overlap < count and (best_rank is None or rank < best_rank):
                    best, best_rank = (trial, candidate), rank
        if best is None:
            return None
        pixels, filled = best
    return None


def submission_instances(labels, classes, regions, image_name):
    output = np.zeros_like(labels)
    objects, kept_regions = [], []
    for (old_id, (ys, xs), mask), category in zip(regions, classes, strict=True):
        original = labels[ys, xs]
        target = output[ys, xs]
        offset = (xs.start, ys.start)
        try:
            points = polygon(mask, offset)
        except ValueError:
            forbidden = ((original != 0) & (original != old_id)) | (target != 0)
            repaired = repair_mask(mask, forbidden)
            if repaired is None:
                print(f'{image_name}, instance {old_id}: omitted; no valid single polygon without overlap',
                      file=sys.stderr)
                continue
            added = np.count_nonzero(repaired & ~mask)
            removed = np.count_nonzero(mask & ~repaired)
            print(f'{image_name}, instance {old_id}: mask repaired (+{added}, -{removed} pixels)',
                  file=sys.stderr)
            mask = repaired
            points = polygon(mask, offset)
        new_id = len(objects) + 1
        target[mask] = new_id
        objects.append({'id': new_id, 'class': category, 'polygon': points})
        kept_regions.append((new_id, (ys, xs), mask))
    return output, objects, kept_regions


def write_image(path, pixels):
    ok, encoded = cv2.imencode(path.suffix, pixels)
    if not ok:
        raise OSError(f'Cannot encode {path}')
    encoded.tofile(path)


def run_one(predictor, path, output, vis_max_side=1280):
    with Image.open(path) as original:
        image = np.asarray(ImageOps.exif_transpose(original).convert('RGB'))
    labels, classes, regions = predictor.predict_instances(image)
    labels, objects, regions = submission_instances(labels, classes, regions, path.name)
    write_image(output / 'seg_mask' / f'{path.stem}.png', labels)
    h, w = labels.shape
    if 0 < vis_max_side < max(h, w):
        scale = vis_max_side / max(h, w)
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        preview_labels = cv2.resize(labels, size, interpolation=cv2.INTER_NEAREST)
        regions = [(i, box, preview_labels[box] == i)
                   for i, box in enumerate(find_objects(preview_labels), 1) if box is not None]
    visible = ((i, f"{i} {objects[i-1]['class']}", box, mask) for i, box, mask in regions)
    write_image(output / 'vis' / f'{path.stem}.png', overlay_regions(image, visible))
    return {'image_name': path.name, 'objects': objects}


def main(args):
    source, output = args.input.resolve(), args.out.resolve()
    if output == source or output in source.parents or source in output.parents:
        raise SystemExit('Use an output directory separate from the input directory')
    if args.threads < 1 or args.warmup < 0:
        raise SystemExit('Threads must be positive and warmup must be nonnegative')
    vis_max_side = getattr(args, 'vis_max_side', 1280)
    if vis_max_side < 0:
        raise SystemExit('Preview side must be nonnegative')
    files = sorted(p for p in args.input.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_TYPES)
    if not files:
        raise SystemExit('No supported images found')
    if len({p.stem.casefold() for p in files}) != len(files):
        raise SystemExit('Input image stems must be unique')
    for folder in ('seg_mask', 'vis'):
        (args.out / folder).mkdir(parents=True, exist_ok=True)
    predictor = Predictor(args.weights, args.variant, args.threads)
    for i in range(args.warmup):
        run_one(predictor, files[i % len(files)], args.out, vis_max_side)
    rows, seconds = [], []
    for path in files:
        start = time.perf_counter()
        row = run_one(predictor, path, args.out, vis_max_side)
        # Include per-image JSON serialization in the timed path.
        serialized = json.dumps(row, ensure_ascii=False, separators=(',', ':'))
        rows.append(serialized)
        seconds.append(time.perf_counter() - start)
    start = time.perf_counter()
    (args.out / 'predictions.json').write_text('[' + ','.join(rows) + ']', encoding='utf-8')
    json_write_per_image = (time.perf_counter() - start) / len(files)
    model_paths = [args.weights / predictor.variant['file'], args.weights / predictor.manifest['classifier']]
    stats = {'images': len(files), 'variant': args.variant, 'threads': args.threads, 'warmup': args.warmup,
             'median_seconds': statistics.median(seconds) + json_write_per_image,
             'vis_max_side': vis_max_side,
             'model_bytes': {p.name: p.stat().st_size for p in model_paths}}
    (args.out / 'benchmark.json').write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--weights', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--variant', default='fp32_768')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=0)
    parser.add_argument('--vis-max-side', type=int, default=1280, help='Limit overlay size only (0 for full resolution)')
    main(parser.parse_args())
