import timm
import torch
import torch.nn as nn
from torch.nn import functional as F

from .layers import ConvLayer, UnetBlock


class DynamicUnet(nn.Module):
    """A Unet that take almost any backbone from timm"""

    def __init__(self, backbone: str = "mobilenetv2_100", dim: int = 256):
        super().__init__()
        self.encoder = timm.create_model(backbone, pretrained=True, features_only=True)

        # passing dummy tensor to get sizes
        dummy_tensor = torch.rand([1, 3, 64, 64])
        features = self.encoder(dummy_tensor)
        ch_sizes = [list(f.shape)[1] for f in features][::-1]
        self.upsample_blocks = []

        self.mid_conv = ConvLayer(ch_sizes[0], dim, 3)

        for i, ch_size in enumerate(ch_sizes[1:]):
            self.upsample_blocks.append(
                UnetBlock(
                    dim,
                    ch_size,
                    out_channels=dim,
                    use_groups=(i == (len(features) - 2)),
                )
            )

    def forward(self, x):
        input_shape = x.shape

        # features reversed in order
        features = self.encoder(x)[::-1]

        # put last feature on dim of the model
        x = self.mid_conv(features[0])

        # upsample blocks with shortcurts from the sides
        for f, ublock in zip(features[1:], self.upsample_blocks):
            x = ublock(f, x)
        x = F.interpolate(
            x, size=input_shape[-2:], mode="bilinear", align_corners=False
        )
        return x
