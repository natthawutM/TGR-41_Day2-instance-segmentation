import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.segmentation import watershed

DEFAULTS = {'foreground': .5, 'boundary': .5, 'min_area': 12, 'min_marker': 8}


def instances_from_logits(logits, box, shape, settings, debug=False):
    y, x, h, w = box
    fg_prob, boundary = 1 / (1 + np.exp(-np.clip(logits[:, y:y+h, x:x+w], -40, 40)))
    fg = (fg_prob >= settings['foreground']).astype(np.uint8)
    count, components, stats_component, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    allowed = stats_component[:, cv2.CC_STAT_AREA] >= settings['min_area']
    allowed[0] = False
    fg = allowed[components]
    dist = cv2.distanceTransform(np.pad(fg.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    core = (fg & (boundary < settings['boundary'])).astype(np.uint8)
    _, markers, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= settings['min_marker']
    keep[0] = False
    markers[~keep[markers]] = 0
    separation = settings.get('marker_separation', 0)
    if separation:
        # Retain the larger seed when two seed centroids in the same object are close.
        retained = {}
        for marker in sorted(np.flatnonzero(keep), key=lambda i: (-stats[i, cv2.CC_STAT_AREA], i)):
            yy, xx = np.where(markers == marker)
            region = int(components[yy[0], xx[0]])
            center = np.array([yy.mean(), xx.mean()])
            if any(np.linalg.norm(center - prior) < separation for prior in retained.get(region, [])):
                markers[yy, xx] = 0
            else:
                retained.setdefault(region, []).append(center)
    next_id = int(markers.max()) + 1
    seeded = set(np.unique(components[markers > 0]))
    for label in range(1, count):
        if allowed[label] and label not in seeded:
            peak = np.argmax(np.where(components == label, dist, -1))
            markers.flat[peak] = next_id
            next_id += 1
    prominence = settings.get('peak_prominence', 0)
    if prominence:
        from skimage.morphology import h_maxima

        for label in np.flatnonzero(allowed):
            x, y, cw, ch, _ = stats_component[label]
            region = np.s_[y:y+ch, x:x+cw]
            in_component = components[region] == label
            local_markers = markers[region]
            existing = np.unique(local_markers[in_component & (local_markers > 0)])
            if len(existing) != 1:
                continue
            distance = np.where(in_component, dist[region], 0)
            if distance.max() <= prominence:
                continue
            peaks = h_maxima(np.pad(distance, 1), prominence)[1:-1, 1:-1]
            n, _, _, points = cv2.connectedComponentsWithStats(peaks.astype(np.uint8))
            centers = []
            for px, py in points[1:n]:
                iy, ix = round(py), round(px)
                if in_component[iy, ix] and distance[iy, ix] >= .50 * distance.max():
                    centers.append((iy, ix))
            if len(centers) < 2:
                continue
            centers.sort(key=lambda p: float(distance[p]), reverse=True)
            first = np.array(centers[0])
            second = next((p for p in centers[1:] if np.linalg.norm(first-p) >=
                           settings.get('peak_separation', 16)), None)
            if second is None:
                continue
            local_markers[in_component] = 0
            local_markers[tuple(first)] = existing[0]
            local_markers[second] = next_id
            next_id += 1
    elevation = boundary - dist / max(float(dist.max()), 1)
    labels = watershed(elevation, markers, mask=fg, connectivity=2)
    final_min_area = settings.get('final_min_area', 1)
    if final_min_area > 1:
        # Merge tiny watershed regions into adjacent seeds by reflooding the same foreground.
        sizes = np.bincount(labels.ravel(), minlength=int(markers.max()) + 1)
        filtered = markers.copy()
        filtered[sizes[markers] < final_min_area] = 0
        filtered_fg = fg & np.isin(components, np.unique(components[filtered > 0]))
        labels = watershed(elevation, filtered, mask=filtered_fg, connectivity=2)
        markers = filtered
    restored = cv2.resize(labels.astype(np.float32), (shape[1], shape[0]),
                          interpolation=cv2.INTER_NEAREST).astype(np.int32)
    present = np.unique(restored[restored > 0])
    if len(present) > np.iinfo(np.uint16).max:
        raise ValueError('Too many instances for a 16-bit ID mask')
    lookup = np.zeros(int(restored.max()) + 1, np.uint16)
    lookup[present] = np.arange(1, len(present) + 1, dtype=np.uint16)
    output = lookup[restored]
    if debug:
        return output, {'foreground_probability': fg_prob, 'foreground': fg,
                        'boundary': boundary, 'markers': markers,
                        'foreground_pixels': int(fg.sum()), 'assigned_pixels': int((labels > 0).sum())}
    return output


def match(labels, instances):
    gt = [(i, c, m) for i, c, m in instances if m.any()]
    n = int(labels.max())
    areas = np.bincount(labels.ravel(), minlength=n+1)[1:]
    matrix = np.zeros((len(gt), n))
    for row, (_, _, mask) in enumerate(gt):
        overlap = np.bincount(labels[mask], minlength=n+1)[1:]
        matrix[row] = overlap / (areas + mask.sum() - overlap)
    rows, cols = linear_sum_assignment(matrix, maximize=True)
    pairs = [(gt[r][0], gt[r][1], int(c+1), float(matrix[r, c]))
             for r, c in zip(rows, cols) if matrix[r, c] > 0]
    return {'instance_iou': sum(p[3] for p in pairs) / max(n, len(gt)) if n or gt else 1.,
            'counting': min(n, len(gt)) / max(n, len(gt)) if n or gt else 1.,
            'count_error': abs(n - len(gt)), 'predicted': n, 'gt': len(gt),
            'pairs': pairs}
