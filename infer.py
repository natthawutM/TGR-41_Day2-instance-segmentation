import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from scipy.ndimage import find_objects

from data import letterbox, normalize, overlay_regions, read_rgb
from post import instances_from_logits

IMAGE_TYPES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


class Predictor:
    def __init__(self, weights, variant, threads):
        self.weights = Path(weights)
        self.manifest = json.loads((self.weights / 'manifest.json').read_text())
        self.variant = self.manifest['variants'][variant]
        self.watershed = self.variant.get('watershed', self.manifest['watershed'])
        self.tta_passes = self.variant.get('tta_passes', 1)
        if self.tta_passes not in (1, 2, 4):
            raise ValueError('tta_passes must be 1, 2, or 4')
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
        model_input = normalize(canvas)[None]
        logits = self.segmenter.run(None, {'image': model_input})[0][0]
        passes = self.tta_passes
        if passes >= 2:
            flipped_input = np.ascontiguousarray(model_input[..., ::-1])
            flipped_logits = self.segmenter.run(None, {'image': flipped_input})[0][0][..., ::-1]
            logits = logits + flipped_logits
        if passes == 4:
            vertical_input = np.ascontiguousarray(model_input[..., ::-1, :])
            both_input = np.ascontiguousarray(model_input[..., ::-1, ::-1])
            vertical = self.segmenter.run(None, {'image': vertical_input})[0][0][..., ::-1, :]
            both = self.segmenter.run(None, {'image': both_input})[0][0][..., ::-1, ::-1]
            logits = logits + (vertical + both)
        if passes > 1:
            logits /= passes
        labels = instances_from_logits(logits, box, image.shape[:2], self.watershed)
        classes = []
        regions = [(i, box, labels[box] == i) for i, box in enumerate(find_objects(labels), 1) if box is not None]
        for start in range(0, len(regions), 16):
            crops = [self.prep_crop(image[box], mask) for _, box, mask in regions[start:start+16]]
            scores = self.classifier.run(None, {'image': np.stack(crops)})[0]
            classes.extend(self.classes[i] for i in scores.argmax(1))
        return labels, classes, regions


def polygon(mask, offset=(0, 0)):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    points = cv2.approxPolyDP(contour, .5, True).reshape(-1, 2)
    if len(points) < 3:
        x, y, w, h = cv2.boundingRect(contour)
        points = np.array([[x, y], [x+w-1, y], [x+w-1, y+h-1], [x, y+h-1]])
    return (points + offset).tolist()


def write_image(path, pixels):
    ok, encoded = cv2.imencode(path.suffix, pixels)
    if not ok:
        raise OSError(f'Cannot encode {path}')
    encoded.tofile(path)


def run_one(predictor, path, output, vis_max_side=1280):
    image = read_rgb(path)
    labels, classes, regions = predictor.predict_instances(image)
    objects = []
    for (instance_id, (ys, xs), mask), category in zip(regions, classes):
        objects.append({'id': instance_id, 'class': category, 'polygon': polygon(mask, (xs.start, ys.start))})
    write_image(output / 'seg_mask' / f'{path.stem}.png', labels)
    h, w = labels.shape
    if 0 < vis_max_side < max(h, w):
        scale = vis_max_side / max(h, w)
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        preview_labels = cv2.resize(labels, size, interpolation=cv2.INTER_NEAREST)
        regions = [(i, box, preview_labels[box] == i)
                   for i, box in enumerate(find_objects(preview_labels), 1) if box is not None]
    visible = ((i, f'{i} {classes[i-1]}', box, mask) for i, box, mask in regions)
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
    parser.add_argument('--variant', default='fp32_512')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=0)
    parser.add_argument('--vis-max-side', type=int, default=1280, help='Limit overlay size only (0 for full resolution)')
    main(parser.parse_args())
