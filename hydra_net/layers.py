# AUTOGENERATED! DO NOT EDIT! File to edit: 02_layers.ipynb (unless otherwise specified).

__all__ = ['ConvLayer', 'CRPBlock', 'UnetBlock']

# Cell
import torch
import torch.nn as nn
import torch.nn.functional as F

# Cell
class ConvLayer(nn.Sequential):
    "A simple Conv+BN+Relu"
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1, groups: int = 1, stride: int = 1, activation: bool = True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2, groups=groups, bias=False),
                  nn.BatchNorm2d(out_channels)] + ([nn.ReLU6(inplace=True)] if activation else [])
        super().__init__(*layers)

# Cell
# Reference: https://github.com/DrSleep/refinenet-pytorch/blob/master/utils/layer_factory.py

# class CRPBlock(nn.Module):

#     def __init__(self, in_channels, out_channels, num_stages, groups=False):
#         super().__init__()
#         for i in range(num_stages):
#             setattr(
#                 self, '{}_{}'.format(i+1, 'outvar_dimred'),
#                 nn.Conv2d(
#                     in_channels if (i == 0) else out_channels,
#                     out_channels, kernel_size=1, stride=1,
#                     padding=0, bias=False,
#                     groups=in_channels if groups else 1
#                 )
#             )
#         self.num_stages = num_stages
#         self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

#     def forward(self, x):
#         out = x
#         for i in range(self.num_stages):
#             out = self.maxpool(out)
#             out = getattr(
#                 self, '{}_{}'.format(i + 1, 'outvar_dimred')
#             )(out)
#             x = out + x
#         return x

class CRPBlock(nn.Module):
    "A bunch of convs and a maxpool with a tricky forward"
    def __init__(self, in_channels, out_channels, num_stages=1, use_groups=False):
        super().__init__()
        groups = in_channels if use_groups else 1
        convs = [nn.Conv2d(in_channels if (i == 0) else out_channels, out_channels, kernel_size=1, bias=False, groups=groups) for i in range(num_stages)]
        self.convs = nn.ModuleList(convs)
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        "y = x + f(x) + f(f(x)) + f(f(f(x)))..."
        out = x
        for conv in self.convs:
            out = conv(self.pool(out))
            x = out + x
        return x

# Cell
class UnetBlock(nn.Module):

    def __init__(self, in_up, in_side, out_channels, kernel_size=1, num_stages=4, use_groups=False):
        super().__init__()
        self.conv_up   = ConvLayer(in_up, out_channels, kernel_size)
        self.conv_side = ConvLayer(in_side, out_channels, kernel_size)
        self.crp = CRPBlock(out_channels, out_channels, num_stages=num_stages, use_groups=use_groups)

    def forward(self, side_input, up_input):
        up_input = self.conv_up(up_input)
        side_input = self.conv_side(side_input)
        if up_input.shape[-2:] != side_input.shape[-2:]:
            up_input = F.interpolate(up_input, size=side_input.shape[-2:], mode="bilinear", align_corners=False)
        out = self.crp(F.relu(up_input + side_input))
        return out