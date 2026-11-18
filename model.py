import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


def block(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Segmenter(nn.Module):
    def __init__(self, pretrained=False, widths=(96, 64, 32, 24)):
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.encoder = mobilenet_v2(weights=weights).features[:18]
        self.indices = (1, 3, 6, 13, 17)
        skip_channels = (96, 32, 24, 16)
        self.decoder = nn.ModuleList(
            block(prev + skip, width)
            for prev, skip, width in zip((320, *widths), skip_channels, widths))
        self.head = nn.Conv2d(widths[-1], 2, 1)

    def forward(self, x):
        skips = []
        for index, layer in enumerate(self.encoder):
            x = layer(x)
            if index in self.indices:
                skips.append(x)
        for skip, layer in zip(reversed(skips[:-1]), self.decoder):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = layer(torch.cat((x, skip), dim=1))
        return F.interpolate(self.head(x), scale_factor=2, mode='bilinear', align_corners=False)


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(block(3, 24, 2), block(24, 48, 2), block(48, 64, 2),
                                    block(64, 96, 2), nn.AdaptiveAvgPool2d(1),
                                    nn.Flatten(), nn.Linear(96, 7))

    def forward(self, x):
        return self.layers(x)


def copy_pruned_decoder(model, checkpoint):
    original = Segmenter(widths=checkpoint['config']['widths'])
    original.load_state_dict(checkpoint['model'])
    model.encoder.load_state_dict(original.encoder.state_dict())
    selected = torch.arange(320)
    original_previous_width = 320
    with torch.no_grad():
        for old, new in zip(original.decoder, model.decoder):
            old_conv, old_bn = old[:2]
            new_conv, new_bn = new[:2]
            if new_conv.out_channels > old_conv.out_channels:
                raise ValueError('Decoder pruning cannot add channels')
            inputs = torch.cat((selected, torch.arange(original_previous_width, old_conv.in_channels)))
            importance = old_conv.weight.abs().sum((1, 2, 3))
            selected = importance.topk(new_conv.out_channels).indices.sort().values
            new_conv.weight.copy_(old_conv.weight[selected][:, inputs])
            for name in ('weight', 'bias', 'running_mean', 'running_var'):
                getattr(new_bn, name).copy_(getattr(old_bn, name)[selected])
            new_bn.num_batches_tracked.copy_(old_bn.num_batches_tracked)
            original_previous_width = old_conv.out_channels
        model.head.weight.copy_(original.head.weight[:, selected])
        model.head.bias.copy_(original.head.bias)


def loss_fn(logits, targets, valid):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    probs = logits.float().sigmoid() * valid
    target = targets * valid
    intersection = (probs * target).sum((0, 2, 3))
    dice = (2 * intersection + 1) / (probs.sum((0, 2, 3)) + target.sum((0, 2, 3)) + 1)
    return (bce * valid).sum() / (2 * valid.sum()) + (1 - dice).mean()
