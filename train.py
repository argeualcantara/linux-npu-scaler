"""
train.py — Trains a lightweight FSRCNN model for 2x super-resolution
and exports it to ONNX.

HR source: 1440p native game captures (no upscaler active)
LR generated: 720p (2x downscale, on-the-fly during training)
Inference target: 720p → 1440p, or 540p → 1080p (same scale ratio)

Usage:
    python train.py --data_dir ./dataset --epochs 50 --output model.onnx
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image


# ─────────────────────────────────────────────
# Model architecture: lightweight FSRCNN
# Input:  LR frame (low resolution)
# Output: HR frame (high resolution, 2x)
# ─────────────────────────────────────────────
class FSRCNN(nn.Module):
    """
    FSRCNN adapted for 2x upscaling.
    Small enough to run in real-time on the XDNA 2 NPU.

    Architecture:
        Feature extraction → Shrinking → Mapping → Expanding → Sub-pixel conv
    """
    def __init__(self, scale=2, d=56, s=12, m=4):
        super().__init__()
        self.scale = scale

        # Feature extraction
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(3, d, kernel_size=5, padding=2),
            nn.PReLU(d),
        )

        # Shrinking
        self.shrinking = nn.Sequential(
            nn.Conv2d(d, s, kernel_size=1),
            nn.PReLU(s),
        )

        # Mapping (m layers)
        mapping_layers = []
        for _ in range(m):
            mapping_layers += [
                nn.Conv2d(s, s, kernel_size=3, padding=1),
                nn.PReLU(s),
            ]
        self.mapping = nn.Sequential(*mapping_layers)

        # Expanding
        self.expanding = nn.Sequential(
            nn.Conv2d(s, d, kernel_size=1),
            nn.PReLU(d),
        )

        # Sub-pixel convolution (PixelShuffle) for learned upscaling
        self.subpixel = nn.Sequential(
            nn.Conv2d(d, 3 * (scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
            nn.Sigmoid(),  # clamp output to [0, 1]
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.shrinking(x)
        x = self.mapping(x)
        x = self.expanding(x)
        x = self.subpixel(x)
        return x


# ─────────────────────────────────────────────
# Dataset: on-the-fly LR/HR pair generation
# ─────────────────────────────────────────────
class SRDataset(Dataset):
    """
    Reads native 1440p HR images from data_dir and generates LR/HR pairs
    by degrading them on-the-fly during training.

    Expected structure:
        data_dir/
            frame_000001.png   (1440p native, no upscaler)
            frame_000002.png
            ...

    Degradation pipeline:
        1440p HR → random crop → 2x bicubic downsample → 720p LR
    """

    EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}

    # 1440p: images must be at least this tall to yield valid HR patches
    MIN_SOURCE_HEIGHT = 1440

    def __init__(self, data_dir, patch_size=64, scale=2, augment=True):
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size  # LR patch size
        self.scale = scale
        self.augment = augment
        self.hr_size = patch_size * scale  # HR patch size (128px for scale=2, patch=64)

        self.image_paths = [
            p for p in self.data_dir.rglob('*')
            if p.suffix.lower() in self.EXTENSIONS
        ]

        if len(self.image_paths) == 0:
            raise FileNotFoundError(
                f"No images found in {data_dir}. "
                "Accepted formats: PNG, JPG, JPEG, WEBP, BMP"
            )

        print(f"Dataset: {len(self.image_paths)} images found in {data_dir}")
        print(f"HR patch size: {self.hr_size}px | LR patch size: {self.patch_size}px | Scale: {self.scale}x")

    def __len__(self):
        # Multiple patches per image to make better use of each 1440p frame
        return len(self.image_paths) * 8

    def _degrade(self, hr_patch):
        """
        Generates a LR patch from an HR patch.
        Simulates what happens when a game renders at lower resolution.

        HR (e.g. 128x128 from a 1440p capture)
            → bicubic downsample to LR (64x64)
            → optional Gaussian noise (simulates capture/compression artifacts)
        """
        w, h = hr_patch.size
        lr_w, lr_h = w // self.scale, h // self.scale

        # Primary downsample
        lr = hr_patch.resize((lr_w, lr_h), Image.BICUBIC)

        # Mild Gaussian noise (simulates encoder noise, capture artifacts)
        if random.random() < 0.5:
            lr_np = np.array(lr, dtype=np.float32)
            noise = np.random.normal(0, random.uniform(0, 5), lr_np.shape)
            lr_np = np.clip(lr_np + noise, 0, 255).astype(np.uint8)
            lr = Image.fromarray(lr_np)

        return lr

    def __getitem__(self, idx):
        path = self.image_paths[idx % len(self.image_paths)]

        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            # Corrupted image: pick a random fallback
            img = Image.open(random.choice(self.image_paths)).convert('RGB')

        # Ensure the image is large enough for an HR patch.
        # 1440p captures are 2560x1440 — well above the minimum.
        # Images smaller than 1440p in the dataset will be upscaled,
        # which slightly reduces quality. Keep your dataset native 1440p.
        min_size = self.hr_size + 1
        if img.width < min_size or img.height < min_size:
            img = img.resize(
                (max(img.width, min_size), max(img.height, min_size)),
                Image.BICUBIC
            )

        # Random HR crop
        i = random.randint(0, img.height - self.hr_size)
        j = random.randint(0, img.width - self.hr_size)
        hr_patch = img.crop((j, i, j + self.hr_size, i + self.hr_size))

        # Data augmentation (preserves HR quality, increases variety)
        if self.augment:
            if random.random() < 0.5:
                hr_patch = TF.hflip(hr_patch)
            if random.random() < 0.3:
                angle = random.choice([90, 180, 270])
                hr_patch = TF.rotate(hr_patch, angle)

        # Generate LR via degradation pipeline
        lr_patch = self._degrade(hr_patch)

        # Convert to float tensor [0, 1]
        to_tensor = transforms.ToTensor()
        return to_tensor(lr_patch), to_tensor(hr_patch)


# ─────────────────────────────────────────────
# Perceptual Loss (VGG features)
# ─────────────────────────────────────────────
class PerceptualLoss(nn.Module):
    """
    Compares intermediate VGG16 feature maps instead of raw pixels.
    Preserves edges and textures much better than L1 alone.
    Particularly important for 1440p source material where fine
    texture detail should be accurately reconstructed.
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

        # ImageNet normalization (VGG was trained with these stats)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        pred_n   = (pred   - self.mean) / self.std
        target_n = (target - self.mean) / self.std
        return self.criterion(self.vgg(pred_n), self.vgg(target_n))


# ─────────────────────────────────────────────
# ONNX export
# ─────────────────────────────────────────────
def export_onnx(model, output_path, scale=2, device='cpu'):
    """
    Exports the trained model to ONNX with dynamic spatial dimensions.
    Dynamic axes allow the model to accept any resolution at inference time,
    not just the patch size used during training.

    Training pair:  720p LR  → 1440p HR  (2x, native 1440p captures)
    Inference use:  720p  → 1440p  (same ratio, full frames)
                    540p  → 1080p  (same ratio, Ally X use case)
    """
    model.eval()
    model_cpu = model.to('cpu')

    # Dummy input: batch=1, 3 channels, 720p (upscaled to 1440p)
    dummy = torch.randn(1, 3, 720, 1280)

    torch.onnx.export(
        model_cpu,
        dummy,
        output_path,
        opset_version=17,
        input_names=['lr_frame'],
        output_names=['sr_frame'],
        dynamic_axes={
            # Accept any height/width at inference time
            'lr_frame': {0: 'batch', 2: 'height', 3: 'width'},
            'sr_frame': {0: 'batch', 2: 'height', 3: 'width'},
        },
        export_params=True,
        do_constant_folding=True,  # folds constants at compile time
    )
    print(f"\nModel exported to: {output_path}")

    # Validate the exported ONNX graph
    try:
        import onnx
        m = onnx.load(output_path)
        onnx.checker.check_model(m)
        print("ONNX validation: OK")
    except ImportError:
        print("(install 'onnx' to validate: pip install onnx)")
    except Exception as e:
        print(f"ONNX validation warning: {e}")


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────
def train(args):
    # Device detection (ROCm exposes GPU through the CUDA interface)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("No GPU detected, falling back to CPU (slower)")

    # Model
    model = FSRCNN(scale=args.scale).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Dataset and DataLoader
    dataset = SRDataset(args.data_dir, patch_size=args.patch_size, scale=args.scale)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        drop_last=True,
    )

    # Loss functions
    l1_loss = nn.L1Loss()
    perceptual = PerceptualLoss(device) if not args.no_perceptual else None

    # Optimizer with cosine LR schedule
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (lr, hr) in enumerate(loader):
            lr = lr.to(device)
            hr = hr.to(device)

            optimizer.zero_grad()
            pred = model(lr)

            # Combined loss: L1 + Perceptual
            loss = l1_loss(pred, hr)
            if perceptual is not None:
                loss = loss + 0.1 * perceptual(pred, hr)

            loss.backward()
            # Gradient clipping: prevents exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 50 == 0:
                print(
                    f"Epoch {epoch}/{args.epochs} | "
                    f"Batch {batch_idx}/{len(loader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        lr_current = optimizer.param_groups[0]['lr']
        print(f"\n→ Epoch {epoch} done | Avg loss: {avg_loss:.4f} | LR: {lr_current:.6f}\n")

        # Save checkpoint on improvement
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = Path(args.output).with_suffix('.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'scale': args.scale,
            }, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path} (loss: {best_loss:.4f})")

    # Export to ONNX after training completes
    export_onnx(model, args.output, scale=args.scale, device=device)
    print("\nTraining complete!")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Train SR model and export to ONNX')
    parser.add_argument('--data_dir',      default='./dataset',  help='Folder with 1440p HR images')
    parser.add_argument('--output',        default='model.onnx', help='ONNX output path')
    parser.add_argument('--epochs',        type=int, default=50)
    parser.add_argument('--batch_size',    type=int, default=32)
    parser.add_argument('--patch_size',    type=int, default=64,  help='LR patch size in pixels')
    parser.add_argument('--scale',         type=int, default=2,   help='Upscale factor')
    parser.add_argument('--lr',            type=float, default=1e-3)
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--no_perceptual', action='store_true',   help='L1 loss only (faster)')
    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
