import torch.nn as nn


class _ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        # Zero-init last conv so each block starts as identity (x + 0)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))


class ESPCNR(nn.Module):
    """
    ESPCN with Residual blocks — evolved architecture for larger NPU budgets.

    Architecture:
        Conv(1→d, 5x5) + ReLU
        N x ResidualBlock(d)  [Conv→ReLU→Conv + skip]
        Conv(d→scale², 3x3) + PixelShuffle(scale)

    ~600K params at d=64, num_blocks=8.
    Operates on the Y channel only (YCbCr pipeline).
    """

    def __init__(self, scale=2, d=64, num_blocks=8):
        super().__init__()
        self.scale = scale

        self.entry = nn.Sequential(
            nn.Conv2d(1, d, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )
        self.residuals = nn.Sequential(*[_ResidualBlock(d) for _ in range(num_blocks)])
        self.exit = nn.Sequential(
            nn.Conv2d(d, scale ** 2, kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        feat = self.entry(x)
        feat = self.residuals(feat)
        return self.exit(feat).clamp(0, 1)


class ESPCN(nn.Module):
    """
    ESPCN — Efficient Sub-Pixel CNN (Shi et al., CVPR 2016).

    Architecture:
        Conv(1→64, 5x5) + Tanh
        Conv(64→32, 3x3) + Tanh
        Conv(32→scale², 3x3) + PixelShuffle(scale)

    NPU-friendly: tiny parameter count (~24K at scale=2).
    Operates on the Y channel only (YCbCr pipeline).
    """

    def __init__(self, scale=2, d=64, s=32, m=4):
        # d/s/m kept for CLI compatibility — ESPCN uses fixed 64/32 channels
        super().__init__()
        self.scale = scale

        self.first_part = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        self.last_part = nn.Sequential(
            nn.Conv2d(32, scale ** 2, kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.001)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.first_part(x)
        x = self.last_part(x)
        return x.clamp(0, 1)
