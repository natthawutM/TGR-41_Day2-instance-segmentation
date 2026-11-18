import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from data import CLASSES, Coco, crop, letterbox, normalize, overlay, targets
from model import Classifier, Segmenter, copy_pruned_decoder, loss_fn
from post import DEFAULTS, instances_from_logits, match


class TrainingSet(Dataset):
    def __init__(self, source, config, preview):
        self.source, self.config, self.preview = source, config, preview
        self.epoch = 0
        self.scenes = None
        if sum(i['width'] * i['height'] * (3 + len(source.by_image[i['id']]))
               for i in source.images) <= 384 * 1024**2:
            self.scenes = [source.load(i) for i in range(len(source.images))]
        if config['copy_paste']:
            colors = []
            for idx in range(len(source.images)):
                image, instances = self.scene(idx)
                foreground = np.zeros(image.shape[:2], bool)
                for _, _, mask in instances:
                    foreground |= mask
                pixels = image[~foreground] if not foreground.all() else image.reshape(-1, 3)
                colors.append(np.median(pixels, axis=0))
            colors = np.asarray(colors)
            distances = np.linalg.norm(colors[:, None] - colors[None, :], axis=-1)
            self.neighbors = np.argsort(distances, axis=1)[:, :min(8, len(colors))]

    def __len__(self):
        return len(self.source.images)

    def scene(self, index):
        return self.scenes[index] if self.scenes is not None else self.source.load(index)

    def paste(self, index, image, instances, rng):
        if not self.config['copy_paste'] or rng.random() >= self.config['copy_paste']:
            return image, instances
        candidates = [i for i in self.neighbors[index] if i != index]
        if not candidates or not instances:
            return image, instances
        donor_image, donor_instances = self.scene(rng.choice(candidates))
        donors = [(cat, m) for _, cat, m in donor_instances if m.any()]
        if not donors:
            return image, instances
        category, donor = donors[rng.integers(len(donors))]
        ys, xs = np.where(donor)
        ay, ax = np.where(instances[rng.integers(len(instances))][2])
        h, w = image.shape[:2]
        ph, pw = int(np.ptp(ys))+1, int(np.ptp(xs))+1
        if ph > h or pw > w:
            return image, instances
        # Set the pasted center near an existing bean so edges touch or overlap.
        side = rng.choice((-1, 1))
        offset = (np.ptp(ax) + pw) * rng.uniform(.33, .52)
        top = int(np.clip(ay.mean() - ph / 2 + rng.uniform(-.2, .2) * ph, 0, h-ph))
        left = int(np.clip(ax.mean() - pw / 2 + side * offset, 0, w-pw))
        dy, dx = ys - ys.min() + top, xs - xs.min() + left
        pasted = np.zeros((h, w), bool)
        pasted[dy, dx] = True
        result = image.copy()
        result[dy, dx] = donor_image[ys, xs]
        fresh = max(i for i, _, _ in instances) + 1
        visible = [(i, c, m & ~pasted) for i, c, m in instances]
        visible = [(i, c, m) for i, c, m in visible if m.any()]
        visible.append((fresh, category, pasted))
        if self.preview is not None and len(list(self.preview.glob('paste_*.png'))) < 6:
            cv2.imwrite(str(self.preview / f'paste_{index}.png'),
                        overlay(result, [(i, f'{i} {self.source.names[c]}', m) for i, c, m in visible]))
        return result, visible

    def __getitem__(self, index):
        rng = np.random.default_rng([self.config['seed'], self.epoch, index])
        image, instances = self.scene(index)
        instances = [(i, c, m) for i, c, m in instances if m.any()]
        paste_rng = np.random.default_rng([self.config['seed'], self.epoch, index, 1])
        image, instances = self.paste(index, image, instances, paste_rng)
        rotation = rng.integers(4)
        flip_x, flip_y = rng.random(2) < .5

        def transform(a):
            a = np.rot90(a, rotation)
            return np.ascontiguousarray(a[::(-1 if flip_y else 1), ::(-1 if flip_x else 1)])

        image = np.clip(transform(image).astype(np.float32) * rng.uniform(.90, 1.10)
                        + rng.uniform(-10, 10), 0, 255).astype(np.uint8)
        instances = [(i, c, transform(m)) for i, c, m in instances]
        size = self.config['size']
        image, box = letterbox(image, size)
        target, valid = targets(instances, box, size)
        if self.preview is not None and not (self.preview / 'targets.png').exists():
            rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            panels = [cv2.cvtColor((channel * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR) for channel in target]
            blended = rgb.copy()
            blended[target[0] > 0] = (.65 * rgb[target[0] > 0] + .35 * np.array([30, 220, 40])).astype(np.uint8)
            blended[target[1] > 0] = (30, 30, 240)
            cv2.imwrite(str(self.preview / 'targets.png'), np.hstack((rgb, blended, *panels)))
        return normalize(image), target, valid


def evaluate(source, config, predict):
    rows = []
    for index, item in enumerate(source.images):
        image, gt = source.load(index)
        canvas, box = letterbox(image, config['size'])
        logits = predict(normalize(canvas)[None])[0]
        labels = instances_from_logits(logits, box, image.shape[:2], config['watershed'])
        result = match(labels, gt)
        pair_iou = {aid: iou for aid, _, _, iou in result['pairs']}
        small = [pair_iou.get(i, 0.) for i, c, m in gt if m.any() and
                 source.names[c] in {'broken', 'husk'} and m.mean() <= .01]
        rows.append({'file': item['file_name'], **result, 'small_broken_husk_iou': small,
                     'pile_candidate': 'pile' in item['file_name'].lower()})
    return summarize(rows)


def write_metrics_csv(path, rows, epoch=None):
    fields = ['filename', 'gt_count', 'pred_count', 'signed_count_error',
              'absolute_count_error', 'instance_iou', 'counting']
    append = epoch is not None and epoch > 0 and path.exists()
    with path.open('a' if append else 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        if not append:
            writer.writerow((['epoch'] if epoch is not None else []) + fields)
        for row in rows:
            values = [row['file'], row['gt'], row['predicted'], row['predicted'] - row['gt'],
                      row['count_error'], row['instance_iou'], row['counting']]
            writer.writerow(([epoch] if epoch is not None else []) + values)


def summarize(rows):
    keys = ('instance_iou', 'counting', 'count_error')
    piles = [r for r in rows if r['pile_candidate']]
    small = [iou for r in rows for iou in r['small_broken_husk_iou']]
    return {**{k: float(np.mean([r[k] for r in rows])) for k in keys},
            'pile_candidates': len(piles),
            'pile_metrics': {k: float(np.mean([r[k] for r in piles])) for k in keys} if piles else None,
            'small_broken_husk_n': len(small), 'small_broken_husk_iou': float(np.mean(small)) if small else None,
            'images': rows}


def save_weights(path, state):
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def train(args):
    config = json.loads(args.config.read_text())
    config.setdefault('watershed', DEFAULTS)
    config.setdefault('widths', [96, 64, 32, 24])
    readmes = list(args.data.glob('README.roboflow*'))
    stretched = any('stretch' in p.read_text(encoding='utf-8-sig').lower() for p in readmes)
    if stretched and not (args.smoke or args.allow_stretch):
        raise SystemExit('This export uses Stretch. Supply a non-Stretch export or pass --allow-stretch.')
    if args.smoke:
        config.update(epochs=1, steps_per_epoch=8)
    args.out.mkdir(parents=True, exist_ok=True)
    config['smoke'] = args.smoke
    config['data'] = str(args.data.resolve())
    config['stretch_export'] = stretched
    config['amp'] = config.get('amp', True)
    torch.manual_seed(config['seed'])
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device {device}: {torch.cuda.get_device_name() if device.type == "cuda" else "CPU"}', flush=True)
    train_coco, valid_coco = (Coco(args.data, s) for s in ('train', 'valid'))
    if set(train_coco.classes) != CLASSES or train_coco.names != valid_coco.names:
        raise ValueError('Expected matching category mapping across splits and all seven classes')
    dataset = TrainingSet(train_coco, config, args.out)
    if args.preview_only or (config['copy_paste'] and not args.resume):
        for index in range(len(dataset)):
            dataset[index]
            if not config['copy_paste'] or len(list(args.out.glob('paste_*.png'))) >= 6:
                break
    if args.preview_only:
        print(f'Preview directory: {args.out}', flush=True)
        return
    if args.resume and args.init:
        raise ValueError('Choose either resume or decoder initialization')
    net = Segmenter(pretrained=not (args.resume or args.init), widths=config['widths'])
    if args.init:
        initial = torch.load(args.init, map_location='cpu')
        if initial['config']['smoke'] != args.smoke:
            raise ValueError('Use checkpoints from the same training mode')
        config['initialized_from'] = str(args.init.resolve())
        copy_pruned_decoder(net, initial)
    net.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=config['lr'])
    amp = device.type == 'cuda' and config['amp']
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    start, best = 0, -1.
    if args.resume:
        state = torch.load(args.out / 'last.pt', map_location=device)
        prior_config = state['config'].copy()
        prior_config.setdefault('stretch_export', stretched)
        prior_config.setdefault('amp', True)
        init_source = prior_config.pop('initialized_from', None)
        if prior_config != config:
            raise ValueError('Resume requires the original training configuration')
        if init_source is not None:
            config['initialized_from'] = init_source
        net.load_state_dict(state['model'])
        opt.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        start, best = state['epoch'] + 1, state['best']
    (args.out / 'config.json').write_text(json.dumps(config, indent=2))
    for epoch in range(start, config['epochs']):
        net.train()
        dataset.epoch = epoch
        shuffle = torch.Generator().manual_seed(config['seed'] + epoch)
        loader = DataLoader(dataset, config['batch'], shuffle=True, generator=shuffle)
        losses = []
        for step, (x, y, valid) in enumerate(loader, 1):
            x, y, valid = x.to(device), y.to(device), valid.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=amp):
                loss = loss_fn(net(x), y, valid)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss')
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
            if step >= config.get('steps_per_epoch', len(loader)):
                break
        net.eval()
        with torch.inference_mode():
            scores = evaluate(valid_coco, config, lambda x: net(torch.from_numpy(x).to(device)).cpu().numpy())
        score = scores['instance_iou'] + scores['counting']
        improved = score > best
        best = max(best, score)
        row = {'epoch': epoch, 'steps': len(losses), 'loss': float(np.mean(losses)),
               **{k: v for k, v in scores.items() if k != 'images'}}
        with (args.out / 'training.jsonl').open('a' if epoch else 'w') as log:
            log.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
        write_metrics_csv(args.out / 'valid_epochs.csv', scores['images'], epoch)
        state = {'model': net.state_dict(), 'optimizer': opt.state_dict(), 'scaler': scaler.state_dict(),
                 'config': config, 'epoch': epoch, 'best': best}
        if improved:
            save_weights(args.out / 'best.pt', state)
            (args.out / 'valid.json').write_text(json.dumps(scores, indent=2))
            write_metrics_csv(args.out / 'valid_best.csv', scores['images'])
        save_weights(args.out / 'last.pt', state)


def train_classifier(args):
    if not args.smoke and any('stretch' in p.read_text(encoding='utf-8-sig').lower()
                              for p in args.data.glob('README.roboflow*')):
        raise SystemExit('This export uses Stretch. Supply the non-Stretch export or use --smoke.')
    torch.manual_seed(41)
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    source = Coco(args.data, 'train')
    if set(source.classes) != CLASSES:
        raise ValueError('Expected all seven classes in training mapping')
    crops, labels = [], []
    for index in range(len(source.images)):
        image, instances = source.load(index)
        for _, category, mask in instances:
            if mask.any():
                crops.append(normalize(letterbox(crop(image, mask), 96)[0]))
                labels.append(source.classes.index(source.names[category]))
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(np.stack(crops)), torch.tensor(labels))
    model = Classifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    weights = torch.from_numpy((len(labels) / (7 * np.maximum(np.bincount(labels, minlength=7), 1))) ** .5)
    loss_fn_cls = torch.nn.CrossEntropyLoss(weight=weights.float().to(device))
    args.out.mkdir(parents=True, exist_ok=True)
    epochs = 3 if args.smoke else args.epochs
    logs = []
    for epoch in range(epochs):
        losses = []
        for x, y in DataLoader(dataset, batch_size=32, shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn_cls(model(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        logs.append({'epoch': epoch, 'loss': float(np.mean(losses))})
    (args.out / 'classifier_training.json').write_text(json.dumps(logs, indent=2))
    save_weights(args.out / 'classifier.pt', {'model': model.state_dict(), 'classes': source.classes,
                                              'size': 96, 'smoke': args.smoke})
    print(logs, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--config', type=Path, default=Path('configs/A.json'))
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--allow-stretch', action='store_true')
    parser.add_argument('--preview-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--init', type=Path, help='Initialize a narrower decoder from a checkpoint')
    parser.add_argument('--classifier', action='store_true')
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()
    (train_classifier if args.classifier else train)(args)
