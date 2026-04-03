"""
backbone.py  –  ResNet-101 + Feature Pyramid Network (FPN)
Paper §3  Network Architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet101
from torchvision.models.resnet import ResNet101_Weights


# ---------------------------------------------------------------------------
# Bottleneck (standard; used only if you want a pure-scratch build)
# We re-use torchvision's ResNet-101 weights for the backbone.
# ---------------------------------------------------------------------------

class FPN(nn.Module):
    """
    Feature Pyramid Network.
    Lin et al. CVPR 2017.  Referenced in §3 Network Architecture.

    Inputs  : C2, C3, C4, C5  –  stride 4, 8, 16, 32 feature maps from ResNet
    Outputs : P2, P3, P4, P5  –  256-d lateral + top-down fusion
              P6               –  max-pool of P5 (used by RPN only)
    """

    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.output_convs  = nn.ModuleList()

        for in_ch in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_ch, out_channels, 1))
            self.output_convs.append(
                nn.Conv2d(out_channels, out_channels, 3, padding=1)
            )

        # Init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, features):
        # features: [C2, C3, C4, C5]  (low-res → high-res when reversed)
        laterals = [l(f) for l, f in zip(self.lateral_convs, features)]

        # Top-down pathway
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i + 1], size=laterals[i].shape[-2:], mode="nearest"
            )

        outs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]
        # P6 for RPN (not used in RoI head)
        outs.append(F.max_pool2d(outs[-1], 1, stride=2, padding=0))
        return outs   # [P2, P3, P4, P5, P6]


class ResNet101FPN(nn.Module):
    """
    ResNet-101 trunk (pretrained ImageNet) + FPN head.
    Returns five feature maps: P2 … P6.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
        base    = resnet101(weights=weights)

        # ---- Strip classifier & avgpool ----
        self.layer0 = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1   # C2  stride 4   256-d
        self.layer2 = base.layer2   # C3  stride 8   512-d
        self.layer3 = base.layer3   # C4  stride 16  1024-d
        self.layer4 = base.layer4   # C5  stride 32  2048-d

        self.fpn = FPN(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=256,
        )

        self.out_channels = 256

    def forward(self, x):
        x  = self.layer0(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return self.fpn([c2, c3, c4, c5])   # [P2, P3, P4, P5, P6]