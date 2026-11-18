import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import find_objects
from scipy.optimize import linear_sum_assignment

from data import CLASSES, Coco, letterbox, normalize, overlay
from infer import Predictor, polygon, submission_instances, write_image
from post import instances_from_logits, match


def prediction_map(folder):
    rows = json.loads((folder / 'predictions.json').read_text(encoding='utf-8'))
    result = {r['image_name']: r['objects'] for r in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate image names in predictions.json')
    return result


def diagnostics(labels, gt):
    overlaps = {}
    hits = {}
    for aid, _, mask in gt:
        if not mask.any():
            continue
        covered = np.bincount(labels[mask], minlength=int(labels.max())+1)
        ids = np.flatnonzero(covered[1:] / mask.sum() >= .20) + 1
        overlaps[aid] = ids.tolist()
        for pid in ids:
            hits.setdefault(int(pid), []).append(aid)
    return {'merged': {pid: a for pid, a in hits.items() if len(a) > 1},
            'split': {aid: ids for aid, ids in overlaps.items() if len(ids) > 1}}


def score(source, labels, classes, gt):
    metrics = match(labels, gt)
    matched = [p for p in metrics['pairs'] if p[3] >= .5]
    small = [i for i, c, m in gt if m.any() and source.names[c] in {'broken', 'husk'} and m.mean() <= .01]
    ious = {aid: iou for aid, _, _, iou in metrics['pairs']}
    missed = [i for i, _, m in gt if m.any() and ious.get(i, 0.) < .5]
    return {**metrics, **diagnostics(labels, gt), 'missed': missed,
            'class_pairs': len(matched) if classes is not None else 0,
            'class_correct': sum(classes[p-1] == source.names[c] for _, c, p, _ in matched) if classes is not None else 0,
            'small_n': len(small), 'small_detected': sum(ious.get(i, 0.) >= .5 for i in small),
            'small_iou_sum': sum(ious.get(i, 0.) for i in small)}


def f1_counts(source, labels, classes, gt):
    masks = [(source.names[c], m) for _, c, m in gt if m.any()]
    n = len(classes)
    areas = np.bincount(labels.ravel(), minlength=n+1)[1:]
    eligible = np.zeros((len(masks), n), dtype=bool)
    same_class = np.zeros_like(eligible)
    for i, (category, mask) in enumerate(masks):
        overlap = np.bincount(labels[mask], minlength=n+1)[1:]
        eligible[i] = overlap >= .95 * (areas + np.count_nonzero(mask) - overlap)
        same_class[i] = eligible[i] & (np.asarray(classes) == category)
    results = {}
    for name, matrix in (('class_agnostic', eligible), ('class_aware', same_class)):
        r, c = linear_sum_assignment(matrix, maximize=True)
        tp = int(matrix[r, c].sum())
        results[name] = {'tp': tp, 'fp': n - tp, 'fn': len(masks) - tp}
    return results


def exportable_labels(labels):
    regions = [(i, box, labels[box] == i) for i, box in enumerate(find_objects(labels), 1) if box is not None]
    return submission_instances(labels, [''] * len(regions), regions, 'Parity check')[0]


def summarize(rows):
    class_n = sum(r['class_pairs'] for r in rows)
    small_n = sum(r['small_n'] for r in rows)
    small_detected = sum(r['small_detected'] for r in rows)
    metrics = ('instance_iou', 'counting', 'count_error')
    pile = [r for r in rows if '_pile_' in r['file']]
    return {**{k: float(np.mean([r[k] for r in rows])) for k in metrics},
            'images': len(rows), 'pile_images': len(pile),
            'pile_metrics': {k: float(np.mean([r[k] for r in pile])) for k in metrics} if pile else None,
            'class_pairs_iou_ge_0.5': class_n,
            'class_accuracy': sum(r['class_correct'] for r in rows) / class_n if class_n else None,
            'small_broken_husk_gt': small_n, 'small_broken_husk_matched_iou_ge_0.5': small_detected,
            'small_broken_husk_matched_fraction': small_detected / small_n if small_n else None,
            'small_broken_husk_iou': sum(r['small_iou_sum'] for r in rows) / small_n if small_n else None}


def verify_outputs(source, args):
    objects = prediction_map(args.predictions)
    if set(objects) != {i['file_name'] for i in source.images}:
        raise ValueError('Prediction filenames differ from Valid')
    parity = args.weights is not None
    if parity:
        import torch
        from model import Segmenter

        torch.set_num_threads(args.threads)
        state = torch.load(args.weights / 'best.pt', map_location='cpu')
        config = state['config']
        predictor = Predictor(args.weights, 'fp32_768', args.threads)
        model = Segmenter(widths=config['widths']).eval()
        model.load_state_dict(state['model'])
        parity_iou, parity_counts, logit_diffs, identical_masks = [], [], [], []
    rows = []
    for idx, item in enumerate(source.images):
        image, gt = source.load(idx)
        name = item['file_name']
        path = args.predictions / 'seg_mask' / f'{Path(name).stem}.png'
        labels = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
        h, w = image.shape[:2]
        if labels.shape != (h, w) or labels.dtype != np.uint16:
            raise ValueError(f'Invalid mask shape or format: {name}')
        items = objects[name]
        ids = [obj['id'] for obj in items]
        if list(np.unique(labels[labels > 0])) != ids or ids != list(range(1, len(ids)+1)):
            raise ValueError(f'PNG and JSON instance IDs differ: {name}')
        for obj in items:
            points = obj['polygon']
            if (obj['class'] not in CLASSES or points != polygon(labels == obj['id']) or len(points) < 3
                    or any(not (0 <= x < w and 0 <= y < h) for x, y in points)):
                raise ValueError(f'Invalid object in {name}: {obj}')
        classes = [o['class'] for o in items]
        rows.append({'file': name, **score(source, labels, classes, gt),
                     'f1_iou_0.95_counts': f1_counts(source, labels, classes, gt)})
        if parity:
            canvas, box = letterbox(image, predictor.size)
            x = normalize(canvas)[None]
            with torch.inference_mode():
                pt = model(torch.from_numpy(x)).numpy()[0]
            ox = predictor.segmenter.run(None, {'image': x})[0][0]
            logit_diffs.append(float(np.max(np.abs(pt - ox))))
            watershed = predictor.manifest['watershed']
            expected = exportable_labels(instances_from_logits(pt, box, (h, w), watershed))
            restored = exportable_labels(instances_from_logits(ox, box, (h, w), watershed))
            if not np.array_equal(restored, labels):
                raise ValueError(f'FP32 CLI output differs from parity run: {name}')
            instances = [(i, 0, expected == i) for i in range(1, int(expected.max())+1)]
            parity_iou.append(match(restored, instances)['instance_iou'] if instances or restored.any() else 1.)
            parity_counts.append(int(restored.max()) == int(expected.max()))
            identical_masks.append(bool(np.array_equal(expected, restored)))
    report = {'summary': summarize(rows), 'rows': rows}
    results = {}
    for mode in ('class_agnostic', 'class_aware'):
        totals = {k: sum(r['f1_iou_0.95_counts'][mode][k] for r in rows) for k in ('tp', 'fp', 'fn')}
        numerator = 2 * totals['tp']
        denominator = numerator + totals['fp'] + totals['fn']
        results[mode] = {**totals, 'f1': numerator / denominator if denominator else 1.}
    report['summary']['f1_iou_0.95'] = results
    if parity:
        report['pytorch_vs_fp32'] = {'max_absolute_logit_difference': max(logit_diffs),
                                     'mean_instance_iou': float(np.mean(parity_iou)),
                                     'same_count_images': sum(parity_counts),
                                     'identical_id_png_images': sum(identical_masks)}
    return report


def compare(source, checkpoints, output):
    import torch
    from model import Segmenter

    torch.set_num_threads(4)
    result = {}
    settings = None
    for checkpoint in checkpoints:
        state = torch.load(checkpoint, map_location='cpu')
        conf = state['config']
        if settings is not None and settings != (conf['size'], conf['watershed']):
            raise ValueError('A/B must use identical input size and watershed')
        settings = (conf['size'], conf['watershed'])
        net = Segmenter(widths=conf['widths']).eval()
        net.load_state_dict(state['model'])
        rows, examples = [], {}
        for idx, item in enumerate(source.images):
            image, gt = source.load(idx)
            canvas, box = letterbox(image, conf['size'])
            with torch.inference_mode():
                logits = net(torch.from_numpy(normalize(canvas)[None])).numpy()[0]
            labels = instances_from_logits(logits, box, image.shape[:2], conf['watershed'])
            row = {'file': item['file_name'], **score(source, labels, None, gt)}
            rows.append(row)
            for issue in ('merged', 'split', 'missed'):
                if row[issue] and issue not in examples:
                    left = overlay(image, [(i, i, m) for i, _, m in gt])
                    right = overlay(image, [(i, i, labels == i) for i in range(1, int(labels.max())+1)])
                    path = output / f'{checkpoint.parent.name}_{issue}.png'
                    write_image(path, np.hstack((left, right)))
                    examples[issue] = {'image': row['file'], 'overlay': path.name, 'ids': row[issue]}
        result[checkpoint.parent.name] = {'summary': summarize(rows), 'examples': examples, 'rows': rows}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--weights', type=Path, help='Check FP32 prediction parity with PyTorch checkpoint')
    parser.add_argument('--compare', nargs='+', type=Path)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    coco = Coco(args.data, 'valid')
    if bool(args.compare) == bool(args.predictions):
        parser.error('Select --predictions or --compare')
    report = compare(coco, args.compare, args.out.parent) if args.compare else verify_outputs(coco, args)
    args.out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    summary = {k: v['summary'] for k, v in report.items()} if args.compare else report['summary']
    print(json.dumps(summary), flush=True)
