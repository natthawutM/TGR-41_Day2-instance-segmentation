import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

MEAN = np.array([.485, .456, .406], np.float32)
STD = np.array([.229, .224, .225], np.float32)
CLASSES = {'black', 'broken', 'good', 'husk', 'immature', 'parchment', 'shell'}


def read_rgb(path):
    image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError(f'Cannot read {path}')
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def decode(seg, h, w):
    from pycocotools import mask as coco_mask

    if isinstance(seg, list):
        polygons = [p for p in seg if len(p) >= 6 and len(p) % 2 == 0]
        if not polygons:
            return np.zeros((h, w), bool)
        rle = coco_mask.merge(coco_mask.frPyObjects(polygons, h, w))
    else:
        if tuple(seg['size']) != (h, w):
            raise ValueError('RLE and image dimensions differ')
        rle = coco_mask.frPyObjects(seg, h, w) if isinstance(seg['counts'], list) else seg
    return coco_mask.decode(rle).astype(bool)


class Coco:
    def __init__(self, root, split):
        self.folder = Path(root) / split
        doc = json.loads((self.folder / '_annotations.coco.json').read_text(encoding='utf-8'))
        self.images = doc['images']
        self.annotations = doc['annotations']
        self.names = {c['id']: c['name'] for c in doc['categories']}
        self.classes = [self.names[k] for k in sorted(self.names) if self.names[k] in CLASSES]
        self.by_image = defaultdict(list)
        for ann in self.annotations:
            self.by_image[ann['image_id']].append(ann)

    def load(self, index):
        info = self.images[index]
        image = read_rgb(self.folder / info['file_name'])
        h, w = image.shape[:2]
        if (w, h) != (info['width'], info['height']):
            raise ValueError(f"Unexpected image dimensions for {info['file_name']}")
        instances = [(a['id'], a['category_id'], decode(a['segmentation'], h, w))
                     for a in self.by_image[info['id']]]
        return image, instances


def letterbox(image, size):
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    y, x = (size - nh) // 2, (size - nw) // 2
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[y:y+nh, x:x+nw] = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas, (y, x, nh, nw)


def normalize(image):
    return np.ascontiguousarray(((image.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1))


def targets(instances, box, size):
    y, x, h, w = box
    target = np.zeros((2, size, size), np.uint8)
    valid = np.zeros((1, size, size), np.float32)
    valid[:, y:y+h, x:x+w] = 1
    kernel = np.ones((3, 3), np.uint8)
    for _, _, mask in instances:
        m = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        edge = m - cv2.erode(m, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        target[:, y:y+h, x:x+w] |= np.stack((m, edge))
    return target.astype(np.float32), valid


def crop(image, mask, masked=True):
    y, x = np.where(mask)
    result = image[y.min():y.max()+1, x.min():x.max()+1].copy()
    if masked:
        result[~mask[y.min():y.max()+1, x.min():x.max()+1]] = 114
    return result


def overlay(image, instances):
    full_image = (slice(None), slice(None))
    return overlay_regions(image, ((i, label, full_image, mask) for i, label, mask in instances))


def overlay_regions(image, instances):
    result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    captions = []
    for instance_id, label, region, mask in instances:
        if not mask.any():
            continue
        color = tuple(int(c) for c in np.random.default_rng(instance_id).integers(50, 255, 3))
        patch = result[region]
        patch[mask] = (patch[mask] * .65 + np.asarray(color) * .35).astype(np.uint8)
        oy, ox = (part.start or 0 for part in region)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                                       offset=(ox, oy))
        cv2.drawContours(result, contours, -1, color, 1)
        y, x = np.where(mask)
        pos = (int(x.mean()) + ox, int(y.mean()) + oy)
        captions.append((str(label), pos, color))
    placed = []
    for label, (x, y), color in captions:
        width, height = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .42, 1)[0]
        tx = min(x, max(0, image.shape[1]-width-2))
        choices = [int(np.clip(y+dy, height+1, max(height+1, image.shape[0]-2)))
                   for dy in (0, -14, 14, -28, 28, -42, 42)]
        ty = next((ty for ty in choices if all(abs(ty-py) > 12 or tx > px+pw or tx+width < px
                                              for px, py, pw in placed)), choices[0])
        placed.append((tx, ty, width))
        if (tx, ty) != (x, y):
            cv2.line(result, (x, y), (tx, ty), color, 1, cv2.LINE_AA)
        cv2.putText(result, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(result, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 255), 1, cv2.LINE_AA)
    return result
