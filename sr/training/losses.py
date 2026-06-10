import torch
import torch.nn as nn


class PerceptualLoss(nn.Module):
    """
    Compares intermediate VGG16 feature maps instead of raw pixels.
    Preserves edges and textures much better than L1 alone.
    """

    def __init__(self, device):
        super().__init__()
        try:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.DEFAULT).features[:16].to(device)
        except Exception:
            from torchvision.models import vgg16
            vgg = vgg16(pretrained=True).features[:16].to(device)

        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.criterion = nn.L1Loss()

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def forward(self, pred, target):
        pred_n   = (pred   - self.mean) / self.std
        target_n = (target - self.mean) / self.std
        return self.criterion(self.vgg(pred_n), self.vgg(target_n))
