"""
train.py — Trains a lightweight ESPCN model for 2x super-resolution
and exports it to ONNX.

HR source: 1440p native game captures (no upscaler active)
LR generated: 720p (2x downscale, on-the-fly during training)
Inference target: 720p → 1440p, or 540p → 1080p (same scale ratio)

Usage:
    # Train from scratch
    python train.py --data_dir ./dataset --epochs 100 --output model.onnx

    # Fine-tune from a pretrained checkpoint (recommended — much faster)
    python train.py --data_dir ./dataset --pretrained fsrcnn_div2k.ckpt --epochs 30

    # Resume an interrupted training run
    python train.py --data_dir ./dataset --resume model.pt --epochs 100

    # Export only (no training — convert checkpoint to ONNX)
    python train.py --pretrained checkpoint.ckpt --output model.onnx --export_only --d 64 --s 16
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger(__name__)

from sr.model import ESPCN
from sr.data import SRDataset
from sr.training import PerceptualLoss, GradientLoss, compute_ssim, compute_psnr_tensor
from sr.training import load_pretrained, export_onnx


def train(args):
    # ── Device ────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device('cuda')
        log.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        log.warning("No GPU detected, falling back to CPU (slower)")

    # ── Mixed precision scaler ─────────────────────────────────────────────
    use_amp = (device.type == 'cuda') and not args.no_amp
    scaler  = torch.amp.GradScaler('cuda', enabled=use_amp)
    if use_amp:
        log.info("Mixed precision (AMP) enabled")

    # ── Model ──────────────────────────────────────────────────────────────
    model = ESPCN(scale=args.scale, d=args.d, s=args.s, m=args.m).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model parameters: {total_params:,}")

    # ── Load pretrained weights (optional) ────────────────────────────────
    start_epoch = 1
    if args.pretrained:
        log.info(f"Loading pretrained weights from: {args.pretrained}")
        model = load_pretrained(model, args.pretrained, device)

    # ── Export only: skip training entirely ───────────────────────────────
    if args.export_only:
        log.info("--export_only: skipping training, exporting directly to ONNX.")
        export_onnx(model, args.output, scale=args.scale)
        log.info(f"Done. ONNX model saved to: {args.output}")
        return

    # ── Resume training (optional) ────────────────────────────────────────
    optimizer_state = None
    scheduler_state = None
    if args.resume:
        log.info(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer_state = ckpt.get('optimizer_state_dict')
        scheduler_state = ckpt.get('scheduler_state_dict')
        start_epoch     = ckpt.get('epoch', 0) + 1
        log.info(f"Resuming from epoch {start_epoch}")

    # ── Dataset ────────────────────────────────────────────────────────────
    dataset = SRDataset(
        args.data_dir,
        patch_size=args.patch_size,
        scale=args.scale,
        cache=args.cache_images,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(args.num_workers > 0),
        drop_last=True,
    )

    # ── Loss functions ─────────────────────────────────────────────────────
    l1_loss    = nn.L1Loss()
    perceptual = PerceptualLoss(device) if not args.no_perceptual else None
    grad_loss  = GradientLoss() if not args.no_gradient else None

    # ── Optimizer ──────────────────────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)

    total = args.epochs
    milestones = [
        int(total * 0.40),
        int(total * 0.60),
        int(total * 0.80),
        int(total * 0.90),
    ]
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.5
    )
    if scheduler_state:
        scheduler.load_state_dict(scheduler_state)

    log.info(f"LR milestones (epochs): {milestones}")

    # ── Validation subset ─────────────────────────────────────────────────
    val_dataset = SRDataset(args.data_dir, patch_size=args.patch_size, scale=args.scale, augment=False)
    val_loader  = DataLoader(val_dataset, batch_size=16, shuffle=True,
                             num_workers=2, drop_last=False)

    # ── Training loop ──────────────────────────────────────────────────────
    best_psnr = 0.0
    checkpoint_path = Path(args.output).with_suffix('.pt')

    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (lr, hr) in enumerate(loader):
            lr = lr.to(device)
            hr = hr.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=use_amp):
                pred = model(lr)
                loss = l1_loss(pred, hr)
                if grad_loss is not None:
                    loss = loss + args.gradient_weight * grad_loss(pred, hr)
                if perceptual is not None:
                    loss = loss + args.perceptual_weight * perceptual(pred, hr)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            if batch_idx % 50 == 0:
                log.info(
                    f"Epoch {epoch}/{start_epoch + args.epochs - 1} | "
                    f"Batch {batch_idx}/{len(loader)} | "
                    f"Loss: {loss.item():.4f} | "
                    f"LR: {optimizer.param_groups[0]['lr']:.2e}"
                )

        scheduler.step()
        avg_loss = epoch_loss / len(loader)

        if epoch % args.val_freq == 0 or epoch == start_epoch + args.epochs - 1:
            model.eval()
            val_psnr_list, val_ssim_list = [], []
            with torch.no_grad():
                for lr_v, hr_v in val_loader:
                    lr_v, hr_v = lr_v.to(device), hr_v.to(device)
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        pred_v = model(lr_v)
                    val_psnr_list.append(compute_psnr_tensor(pred_v, hr_v))
                    val_ssim_list.append(compute_ssim(pred_v, hr_v))
                    if len(val_psnr_list) >= 20:
                        break

            avg_psnr = np.mean(val_psnr_list)
            avg_ssim = np.mean(val_ssim_list)
            log.info(
                f"Epoch {epoch} | Loss: {avg_loss:.4f} | "
                f"PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f}"
            )

            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                torch.save({
                    'epoch': epoch,
                    'model_state_dict':     model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss':  avg_loss,
                    'psnr':  avg_psnr,
                    'ssim':  avg_ssim,
                    'scale': args.scale,
                    'd': args.d, 's': args.s, 'm': args.m,
                }, checkpoint_path)
                log.info(f"Best checkpoint saved: {checkpoint_path} (PSNR: {best_psnr:.2f} dB)")
        else:
            log.info(f"Epoch {epoch} | Loss: {avg_loss:.4f}")

    # ── Export to ONNX ────────────────────────────────────────────────────
    if checkpoint_path.exists():
        best_ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt['model_state_dict'])
        log.info(f"Exporting best checkpoint (PSNR: {best_ckpt['psnr']:.2f} dB) to ONNX...")

    export_onnx(model, args.output, scale=args.scale)
    log.info("Training complete!")
    log.info(f"Best PSNR:   {best_psnr:.2f} dB")
    log.info(f"Checkpoint:  {checkpoint_path}")
    log.info(f"ONNX model:  {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description='Train ESPCN super-resolution model and export to ONNX',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Paths
    parser.add_argument('--data_dir',    default='./dataset',  help='Folder with 1440p HR images')
    parser.add_argument('--output',      default='model.onnx', help='ONNX output path')
    parser.add_argument('--pretrained',  default=None,
                        help='Path to pretrained .ckpt or .pt weights for fine-tuning')
    parser.add_argument('--resume',      default=None,
                        help='Path to checkpoint to resume an interrupted training run')

    # Training
    parser.add_argument('--epochs',      type=int,   default=50)
    parser.add_argument('--batch_size',  type=int,   default=32)
    parser.add_argument('--patch_size',  type=int,   default=64,  help='LR patch size in pixels')
    parser.add_argument('--scale',       type=int,   default=2,   help='Upscale factor')
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int,   default=4)
    parser.add_argument('--val_freq',    type=int,   default=5,   help='Run validation every N epochs')

    # Model architecture
    parser.add_argument('--d', type=int, default=56, help='Feature map depth')
    parser.add_argument('--s', type=int, default=12, help='Shrinking channels')
    parser.add_argument('--m', type=int, default=4,  help='Mapping layers')

    # Flags
    parser.add_argument('--no_perceptual',      action='store_true', help='Disable perceptual (VGG16) loss')
    parser.add_argument('--no_gradient',        action='store_true', help='Disable gradient edge loss')
    parser.add_argument('--perceptual_weight',  type=float, default=0.1, help='Weight for perceptual loss term')
    parser.add_argument('--gradient_weight',    type=float, default=0.1, help='Weight for gradient edge loss term')
    parser.add_argument('--no_amp',        action='store_true', help='Disable mixed precision training')
    parser.add_argument('--cache_images',  action='store_true', help='Pre-load entire dataset into RAM')
    parser.add_argument('--export_only',   action='store_true',
                        help='Load pretrained weights and export to ONNX, skip training')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
