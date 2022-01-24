import torch.nn as nn

from .dynamic_unet import DynamicUnet


class HydraNet(nn.Module):
    def __init__(
        self,
        backbone: str = "mobilenetv2_100",
        hidden_dim: int = 256,
        num_classes: int = 21,
    ):
        super().__init__()
        self.backbone = DynamicUnet(backbone, dim=hidden_dim)
        self.segmentation_head = nn.Conv2d(hidden_dim, num_classes, 1, bias=False)

    def forward(self, x):
        backbone_out = self.backbone(x)
        return self.segmentation_head(backbone_out)
