import argparse
import json
import shutil
from pathlib import Path

import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from torchvision.models import mobilenet_v3_small

from data import CLASSES, Coco, letterbox, normalize, read_rgb
from model import Classifier, Segmenter


class Calibration(CalibrationDataReader):
    def __init__(self, root, size):
        coco = Coco(root, 'train')
        self.images = iter(sorted(coco.folder / row['file_name'] for row in coco.images))
        self.size = size

    def get_next(self):
        path = next(self.images, None)
        if path is not None:
            image, _ = letterbox(read_rgb(path), self.size)
            return {'image': normalize(image)[None]}
        return None


def export_model(model, path, size, dynamic=False):
    axes = {'image': {0: 'instances'}, 'logits': {0: 'instances'}} if dynamic else None
    torch.onnx.export(model.eval(), (torch.zeros(1, 3, size, size),), path, dynamo=False,
                      opset_version=17, input_names=['image'], output_names=['logits'], dynamic_axes=axes)


def export(args):
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=True)
    state = torch.load(args.checkpoint, map_location='cpu')
    classifier_state = torch.load(args.classifier, map_location='cpu')
    config = state['config']
    seg = Segmenter(widths=config['widths'])
    seg.load_state_dict(state['model'])
    if set(classifier_state['classes']) != CLASSES or len(classifier_state['classes']) != 7:
        raise ValueError('Expected seven classifier classes in checkpoint order')
    if classifier_state.get('architecture') == 'mobilenet_v3_small':
        classifier = mobilenet_v3_small(weights=None)
        classifier.classifier[3] = torch.nn.Linear(classifier.classifier[3].in_features, 7)
        classifier.load_state_dict(classifier_state['model_state_dict'])
        size = classifier_state['input_size']
        preprocessing = {'mode': 'day1', 'mean': classifier_state['mean'], 'std': classifier_state['std']}
    else:
        if config['smoke'] != classifier_state['smoke']:
            raise ValueError('Use checkpoint and classifier from the same training mode')
        classifier = Classifier()
        classifier.load_state_dict(classifier_state['model'])
        size = classifier_state['size']
        preprocessing = {'mode': 'letterbox'}
    if args.checkpoint.resolve() != (args.out / 'best.pt').resolve():
        shutil.copyfile(args.checkpoint, args.out / 'best.pt')
    export_model(classifier, args.out / 'classifier.onnx', size, dynamic=True)
    watershed_config = json.loads(args.watershed.read_text()) if args.watershed else config['watershed']
    manifest = {'classes': classifier_state['classes'], 'classifier_size': size,
                'classifier_preprocessing': preprocessing,
                'classifier': 'classifier.onnx', 'watershed': watershed_config,
                'smoke': config['smoke'], 'variants': {}}
    for size in dict.fromkeys(args.sizes):
        fp32 = args.out / f'seg_fp32_{size}.onnx'
        export_model(seg, fp32, size)
        manifest['variants'][f'fp32_{size}'] = {'file': fp32.name, 'size': size}
        if args.int8:
            int8 = args.out / f'seg_int8_{size}.onnx'
            quantize_static(fp32, int8, Calibration(args.data, size), quant_format=QuantFormat.QDQ,
                            activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                            per_channel=True, op_types_to_quantize=['Conv'])
            manifest['variants'][f'int8_{size}'] = {'file': int8.name, 'size': size}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--classifier', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--sizes', nargs='+', type=int, choices=[512, 384, 320], default=[512])
    parser.add_argument('--int8', action='store_true')
    parser.add_argument('--watershed', type=Path, help='Watershed JSON to embed in the deployment manifest')
    export(parser.parse_args())
