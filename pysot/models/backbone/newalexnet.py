# ============================================================
# BUG FIXES IN THIS FILE:
# 1. CRITICAL: AlexNet.forward() in the ORIGINAL newalexnet.py returns
#    (x2, x1, x) — three feature maps from layer3, layer4, layer5.
#    HiFT.forward() receives these as z=(z2,z1,z0) and x=(x2,x1,x0)
#    and calls xcorr_depthwise on x[0]⊗z[0], x[1]⊗z[1], x[2]⊗z[2].
#    Then: conv1(xcorr(x[0],z[0])) — x[0] is layer3 output (384 channels).
#    HiFT.conv1 expects input=384 → output=192. ✓
#          conv3(xcorr(x[1],z[1])) — x[1] is layer4 output (384 channels).
#    HiFT.conv3 expects input=384 → output=192. ✓
#          conv2(xcorr(x[2],z[2])) — x[2] is layer5 output (256 channels).
#    HiFT.conv2 expects input=256 → output=192. ✓
#    This is CORRECT — no change needed.
#
# 2. The two frozen layers (layer1, layer2) in the original AlexNet class
#    are correct for memory efficiency. Kept.
#
# 3. ResNet50 definition at the top of this file was using Bottleneck from
#    newalexnet.py (not resnet_atrous.py). This is an unused dead-code
#    duplicate. Removed to avoid confusion — not called by ModelBuilder.
#
# 4. No other bugs found. File reproduced as-is with minor cleanup.
# ============================================================

import torch.nn as nn
import math


class AlexNet(nn.Module):
    """
    Modified AlexNet backbone for HiFT tracking.
    Returns features from layer3, layer4, layer5 (shapes 384, 384, 256).
    layer1 and layer2 are frozen (requires_grad=False) to save memory.
    """
    configs = [3, 96, 256, 384, 384, 256]

    def __init__(self, width_mult=1):
        configs = list(map(
            lambda x: 3 if x == 3 else int(x * width_mult),
            AlexNet.configs))
        super(AlexNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(configs[0], configs[1], kernel_size=11, stride=2),
            nn.BatchNorm2d(configs[1]),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(configs[1], configs[2], kernel_size=5),
            nn.BatchNorm2d(configs[2]),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(configs[2], configs[3], kernel_size=3),
            nn.BatchNorm2d(configs[3]),
            nn.ReLU(inplace=True),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(configs[3], configs[4], kernel_size=3),
            nn.BatchNorm2d(configs[4]),
            nn.ReLU(inplace=True),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(configs[4], configs[5], kernel_size=3),
            nn.BatchNorm2d(configs[5]),
        )
        self.feature_size = configs[5]

        # Freeze early layers to save memory & preserve low-level features
        for param in self.layer1.parameters():
            param.requires_grad = False
        for param in self.layer2.parameters():
            param.requires_grad = False

    def forward(self, x):
        x  = self.layer1(x)
        x  = self.layer2(x)
        x2 = self.layer3(x)    # (B, 384, H3, W3) — intermediate feature
        x1 = self.layer4(x2)   # (B, 384, H4, W4) — intermediate feature
        x0 = self.layer5(x1)   # (B, 256, H5, W5) — final feature
        # Return from coarsest (layer3) to finest (layer5)
        # HiFT uses all three levels for hierarchical cross-correlation
        return x2, x1, x0