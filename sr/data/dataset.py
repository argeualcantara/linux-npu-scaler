import random
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import functional as TF
from PIL import Image
from torch.utils.data import Dataset


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

    def __init__(self, data_dir, patch_size=64, scale=2, augment=True, cache=False):
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.scale = scale
        self.augment = augment
        self.hr_size = patch_size * scale
        self.cache = cache
        self._image_cache = {}

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
        if cache:
            print("Image caching enabled — pre-loading dataset into RAM...")
            self._preload_cache()

    def _preload_cache(self):
        for path in self.image_paths:
            try:
                self._image_cache[str(path)] = Image.open(path).convert('RGB')
            except Exception:
                pass
        print(f"Cached {len(self._image_cache)} images in RAM.")

    def _load_image(self, path):
        key = str(path)
        if self.cache and key in self._image_cache:
            return self._image_cache[key].copy()
        return Image.open(path).convert('RGB')

    def __len__(self):
        # 8 random patches per image per epoch
        return len(self.image_paths) * 8

    def _degrade(self, hr_patch):
        w, h = hr_patch.size
        lr_w, lr_h = w // self.scale, h // self.scale
        lr = hr_patch.resize((lr_w, lr_h), Image.BICUBIC)
        if random.random() < 0.5:
            lr_np = np.array(lr, dtype=np.float32)
            noise = np.random.normal(0, random.uniform(0, 5), lr_np.shape)
            lr_np = np.clip(lr_np + noise, 0, 255).astype(np.uint8)
            lr = Image.fromarray(lr_np)
        return lr

    def __getitem__(self, idx):
        path = self.image_paths[idx % len(self.image_paths)]
        try:
            img = self._load_image(path)
        except Exception:
            img = self._load_image(random.choice(self.image_paths))

        min_size = self.hr_size + 1
        if img.width < min_size or img.height < min_size:
            img = img.resize(
                (max(img.width, min_size), max(img.height, min_size)),
                Image.BICUBIC
            )

        i = random.randint(0, img.height - self.hr_size)
        j = random.randint(0, img.width - self.hr_size)
        hr_patch = img.crop((j, i, j + self.hr_size, i + self.hr_size))

        if self.augment:
            if random.random() < 0.5:
                hr_patch = TF.hflip(hr_patch)
            if random.random() < 0.3:
                angle = random.choice([90, 180, 270])
                hr_patch = TF.rotate(hr_patch, angle)

        lr_patch = self._degrade(hr_patch)

        def to_y_tensor(img):
            y = img.convert('YCbCr').split()[0]
            arr = np.array(y, dtype=np.float32) / 255.0
            return torch.from_numpy(arr).unsqueeze(0)

        return to_y_tensor(lr_patch), to_y_tensor(hr_patch)
