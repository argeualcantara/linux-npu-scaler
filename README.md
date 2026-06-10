# SR-Linux — NPU Super-Resolution for Linux
### End-to-end pipeline: capture → train → ONNX → inference

---

## Resolution strategy

| Stage | Resolution | Notes |
|---|---|---|
| Dataset (HR source) | **1440p native** | No upscaler active |
| Generated LR | **720p** | 2x bicubic downsample, on-the-fly |
| Inference — training machine | 720p → 1440p | Full test |
| Inference — Ally X | 540p → 1080p | Same 2x ratio, fits the device |

Capture at native 1440p with **all upscalers disabled** (FSR, DLSS, XeSS, NIS,
Gamescope SR). The model learns from real pixel detail — upscaled source frames
undermine training quality.

---

## Project structure

```
sr_project/
├── train.py            # Trains the model, exports to ONNX
├── upscale.py          # Inference: image in → upscaled image out
├── capture_dataset.py  # Captures native 1440p frames from games
├── dataset/            # Your HR images go here
└── model.onnx          # Generated after training
```

---

## 0. Installation

```bash
# Create isolated environment
python -m venv venv
source venv/bin/activate

# PyTorch with ROCm (RX 9060 XT)
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.3

# Inference and utility dependencies
pip install onnx onnxruntime numpy Pillow scipy

# Optional: live screen capture
pip install mss

# Verify GPU is detected (ROCm exposes itself through the CUDA interface)
python -c "import torch; print('GPU OK:', torch.cuda.get_device_name(0))"
```

---

## 1. Build the dataset (1440p HR images)

### Option A: Extract from a recorded video

Record the game at **native 1440p** with OBS (no post-process filters, no sharpening).
Recommended OBS settings: 2560×1440, lossless or high-bitrate H.264/AV1.

```bash
python capture_dataset.py video \
  --source recording_1440p.mp4 \
  --output ./dataset \
  --every 30          # 1 frame every 30 (avoids redundant frames at 30fps video)
```

### Option B: Live screen capture while playing

```bash
python capture_dataset.py screen \
  --output ./dataset \
  --duration 300 \    # 5 minutes
  --fps 2.0
```

Switch to your game when prompted. Move the camera, explore different environments.
Scene variety = better generalisation.

Checklist before capturing:
- [ ] Game resolution: 2560×1440
- [ ] FSR / DLSS / XeSS: **OFF**
- [ ] Gamescope upscaling: **OFF**
- [ ] In-game sharpening filter: **OFF**

### Remove near-duplicate frames (loading screens, static menus)

```bash
python capture_dataset.py dedup --dir ./dataset
```

**Minimum goal:** 5,000 images from at least 5 different games.  
**Good goal:** 50,000+ images from 10–15 games with varied art styles.

---

## 2. Train the model

### Quick test run (~30 min on RX 9060 XT)

```bash
python train.py \
  --data_dir ./dataset \
  --output model.onnx \
  --epochs 10 \
  --batch_size 32 \
  --no_perceptual     # L1 only = faster, lower quality
```

### Full recommended training (~8–12h)

```bash
python train.py \
  --data_dir ./dataset \
  --output model.onnx \
  --epochs 100 \
  --batch_size 32 \
  --patch_size 64 \   # LR patch size; HR patch = 128px (2x)
  --scale 2 \
  --lr 0.001
```

Checkpoints (`.pt`) are saved automatically whenever loss improves.
You can interrupt and resume training if needed.

**What happens during training:**
1. Load a native 1440p HR image from the dataset
2. Take a random 128×128 patch
3. Downsample it 2x to 64×64 → LR patch
4. Pass LR through the model → SR prediction
5. Compute loss between SR and the original 128×128 HR patch
6. Update model weights
7. Repeat for N epochs

---

## 3. Test with an image

### Basic test

```bash
# Use any 720p image as input (e.g. a frame from your dataset downscaled)
python upscale.py \
  --input screenshot_720p.png \
  --model model.onnx
# → output: screenshot_720p_sr.png  (1440p)
```

### With side-by-side comparison

```bash
python upscale.py \
  --input screenshot_720p.png \
  --model model.onnx \
  --compare
# → saves screenshot_720p_sr_compare.png
# Left: bicubic resize (baseline) | Right: model output
```

### With PSNR measurement (requires native HR reference)

```bash
python upscale.py \
  --input screenshot_720p.png \
  --model model.onnx \
  --hr_ref screenshot_1440p_native.png \
  --compare
# → prints PSNR in dB  (>30 dB = acceptable, >35 dB = good)
```

### Tile mode (for large images that don't fit in memory)

```bash
python upscale.py \
  --input screenshot_1440p.png \
  --model model.onnx \
  --tile 256 \
  --overlap 16
```

---

## 4. What to look for in the output

Compare the model output against the bicubic baseline:

- **Edges** — sharper, or blurry/over-sharpened?
- **HUD / text** — readable? No ringing artifacts?
- **Flat areas** (sky, walls) — clean, or spurious patterns?
- **Temporal stability** (if testing on a sequence) — do details flicker between frames?

If the result looks blurry → model needs more epochs or more training data.  
If the result has grid-like artifacts → patch size or architecture may need tuning.  
If edges ring or glow → perceptual loss weight may be too high.

---

## 5. Next steps after validating the model

### INT8 quantization for the XDNA 2 NPU (Ally X)

```bash
pip install quark

python -c "
from quark.onnx import ModelQuantizer
q = ModelQuantizer('model.onnx', 'model_int8.onnx')
q.quantize()
"
```

INT8 quantization trades a small amount of accuracy for ~4x faster NPU inference.
Benchmark both and pick the threshold you are comfortable with.

### Gamescope integration

The next step is adding an NPU backend to Gamescope that intercepts frames
between the game and the display, calls `upscale_full()` via ONNX Runtime
with the VitisAI Execution Provider, and flips the SR frame to the screen.

---

## References

- [FSRCNN paper](https://arxiv.org/abs/1608.00367)
- [AMD Ryzen AI docs](https://ryzenai.docs.amd.com)
- [ONNX Runtime docs](https://onnxruntime.ai/docs/)
- [Gamescope source](https://github.com/ValveSoftware/gamescope)
- [AMD XDNA driver](https://github.com/amd/xdna-driver)
