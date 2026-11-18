import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from data import crop, letterbox, normalize, overlay, read_rgb
from post import instances_from_logits

IMAGE_TYPES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


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
            patch = crop(image, mask, masked=False)
            rgb = np.asarray(Image.fromarray(patch).resize((size, size), Image.Resampling.BILINEAR),
                             dtype=np.float32) / 255
            mean = np.asarray(settings['mean'], np.float32)
            std = np.asarray(settings['std'], np.float32)
            return np.ascontiguousarray(((rgb - mean) / std).transpose(2, 0, 1))
        return normalize(letterbox(crop(image, mask), size)[0])

    def predict(self, image):
        canvas, box = letterbox(image, self.size)
        logits = self.segmenter.run(None, {'image': normalize(canvas)[None]})[0][0]
        labels = instances_from_logits(logits, box, image.shape[:2], self.manifest['watershed'])
        classes = []
        if n := int(labels.max()):
            for start in range(1, n+1, 16):
                crops = [self.prep_crop(image, labels == i) for i in range(start, min(n+1, start+16))]
                scores = self.classifier.run(None, {'image': np.stack(crops)})[0]
                classes.extend(self.classes[i] for i in scores.argmax(1))
        return labels, classes


def polygon(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    points = cv2.approxPolyDP(contour, .5, True).reshape(-1, 2)
    if len(points) < 3:
        x, y, w, h = cv2.boundingRect(contour)
        points = np.array([[x, y], [x+w-1, y], [x+w-1, y+h-1], [x, y+h-1]])
    return points.tolist()


def write_image(path, pixels):
    ok, encoded = cv2.imencode(path.suffix, pixels)
    if not ok:
        raise OSError(f'Cannot encode {path}')
    encoded.tofile(path)


def run_one(predictor, path, output):
    image = read_rgb(path)
    labels, classes = predictor.predict(image)
    objects, visible = [], []
    for instance_id, category in enumerate(classes, 1):
        mask = labels == instance_id
        objects.append({'id': instance_id, 'class': category, 'polygon': polygon(mask)})
        visible.append((instance_id, f'{instance_id} {category}', mask))
    write_image(output / 'seg_mask' / f'{path.stem}.png', labels)
    write_image(output / 'vis' / f'{path.stem}.png', overlay(image, visible))
    return {'image_name': path.name, 'objects': objects}


def main(args):
    source, output = args.input.resolve(), args.out.resolve()
    if output == source or output in source.parents or source in output.parents:
        raise SystemExit('Use an output directory separate from the input directory')
    if args.threads < 1 or args.warmup < 0:
        raise SystemExit('Threads must be positive and warmup must be nonnegative')
    files = sorted(p for p in args.input.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_TYPES)
    if not files:
        raise SystemExit('No supported images found')
    if len({p.stem.casefold() for p in files}) != len(files):
        raise SystemExit('Input image stems must be unique')
    for folder in ('seg_mask', 'vis'):
        (args.out / folder).mkdir(parents=True, exist_ok=True)
    predictor = Predictor(args.weights, args.variant, args.threads)
    for i in range(args.warmup):
        run_one(predictor, files[i % len(files)], args.out)
    rows, seconds = [], []
    for path in files:
        start = time.perf_counter()
        row = run_one(predictor, path, args.out)
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
    main(parser.parse_args())
