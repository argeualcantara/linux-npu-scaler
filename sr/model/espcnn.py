import torch.nn as nn


class FSRCNN(nn.Module):
    """
    ESPCN — Efficient Sub-Pixel CNN (Shi et al., CVPR 2016).
    Named FSRCNN for API compatibility but implements ESPCN architecture.

    Architecture (matches yjn870/ESPCN-pytorch x2 pretrained weights exactly):
        Conv(1→64, 5x5) + Tanh
        Conv(64→32, 3x3) + Tanh
        Conv(32→scale², 3x3) + PixelShuffle(scale)

    Why ESPCN instead of FSRCNN:
        - Pretrained x2 weights available (yjn870/ESPCN-pytorch) — exact match
        - Same PixelShuffle upsampler, same Y-channel pipeline
        - Smaller and faster than FSRCNN for comparable quality
        - NPU-friendly: tiny parameter count (~24K at scale=2)
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
