# npu-scope — NPU Super-Resolution for Linux
### End-to-end pipeline: capture → train → ONNX → inference → Gamescope

---

## What this is

An open-source Linux equivalent of Windows Auto SR, targeting the AMD XDNA 2 NPU
inside the ROG Ally X (and any Ryzen AI 300 / Strix Point device). The model runs
as a Gamescope upscaling backend: the game renders at 540p, the NPU upscales to
1080p, the display sees 1080p — with zero GPU overhead.

Current status: training + inference pipeline complete. Gamescope integration in progress.

---

## Resolution strategy

| Stage | Resolution | Notes |
|---|---|---|
| Dataset (HR source) | **1440p native** | No upscaler active during capture |
| Generated LR (training) | **720p** | 2x bicubic downsample, on-the-fly |
| Inference — training machine | 720p → 1440p | Full quality test |
| Inference — Ally X (target) | 540p → 1080p | Same 2x ratio |

Capture at native 1440p with **all upscalers disabled** (FSR, DLSS, XeSS, NIS,
Gamescope SR). The model learns from real pixel detail — upscaled source frames
undermine training quality.

---

## Project structure

```
npu-scope/
├── train.py            # Train the model, export to ONNX
├── upscale.py          # Inference: image in → upscaled image out
├── capture_dataset.py  # Capture native 1440p frames from games
├── dataset/            # Your HR images go here (git-ignored)
├── model.onnx          # Generated after training (versioned on Hugging Face)
└── requirements.txt    # Python dependencies
```

---

## 0. Prerequisites

- Python 3.10 or newer
- An AMD GPU with ROCm support **or** any CPU (for testing)
- `ffmpeg` installed (for video frame extraction)
- Git

---

## 1. Installation

### 1a. Clone the repo and create a virtual environment

```bash
git clone https://github.com/your-username/npu-scope
cd npu-scope

python -m venv venv
source venv/bin/activate
# Your prompt will change to (venv) — all pip commands below run inside it
```

### 1b. Install PyTorch

**If you have an AMD GPU (RX 9060 XT or similar — ROCm):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.3
```

**If you have an NVIDIA GPU (CUDA):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**CPU only (for testing, no GPU required):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 1c. Install remaining dependencies

```bash
pip install onnx onnxruntime numpy Pillow scipy
pip install mss          # optional: for live screen capture
```

### 1d. Verify your GPU is detected

```bash
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not detected (CPU mode)')"
```

### 1e. Save your dependencies

```bash
pip freeze > requirements.txt
# Anyone cloning the repo later can run: pip install -r requirements.txt
```

---

## 2. Get a pretrained model (recommended starting point)

Instead of training from scratch, start from a pretrained FSRCNN checkpoint and
fine-tune on your gaming dataset. This gives much better results in fewer epochs.

```bash
# Download the DIV2K-pretrained checkpoint from suxrobGM/fsrcnn
# (trained on high-resolution photography — good starting point for gaming)
curl -L https://github.com/suxrobGM/fsrcnn/raw/main/pretrained/fsrcnn_div2k.ckpt \
     -o fsrcnn_div2k.ckpt
```

Or clone the full repo and copy the file:
```bash
git clone https://github.com/suxrobGM/fsrcnn /tmp/fsrcnn_ref
cp /tmp/fsrcnn_ref/pretrained/fsrcnn_div2k.ckpt ./
```

### Export Only

``` bash
python train.py \
  --pretrained pretrained/fsrcnn_div2k.ckpt \
  --output model.onnx \
  --export_only \
  --d 64 --s 16
```

---

## 3. Build the dataset (1440p HR images)

### Option A: Extract from a recorded video (recommended)

Record the game at **native 1440p** with OBS. Settings: 2560×1440, high-bitrate
H.264 or AV1, **no post-process filters, no sharpening**.

Checklist before recording:
- [ ] Game resolution: **2560×1440**
- [ ] FSR / DLSS / XeSS / NIS: **OFF**
- [ ] Gamescope upscaling: **OFF**
- [ ] In-game sharpening: **OFF**

```bash
python capture_dataset.py video \
  --source recording_1440p.mp4 \
  --output ./dataset \
  --every 30
# Extracts 1 frame every 30 frames (~1 fps at 30fps video)
# Target: 5,000+ frames from 5+ different games
```

### Option B: Live screen capture while playing

```bash
python capture_dataset.py screen \
  --output ./dataset \
  --duration 300 \
  --fps 2.0
# Switch to your game when prompted. Play for 5 minutes.
```

### Remove near-duplicate frames

```bash
python capture_dataset.py dedup --dir ./dataset
# Removes loading screens, static menus, near-identical frames
```

**Minimum goal:** 5,000 images from 5+ games.
**Good goal:** 50,000+ images from 10–15 games with varied art styles.

---

## 4. Train the model

### Option A: Fine-tune from pretrained checkpoint (recommended)

Much faster and better quality than training from scratch. Uses the DIV2K
checkpoint as the starting point and fine-tunes on your gaming dataset.

```bash
python train.py \
  --data_dir ./dataset \
  --pretrained fsrcnn_div2k.ckpt \
  --output model.onnx \
  --epochs 30 \
  --batch_size 32 \
  --lr 0.0001
# Lower LR for fine-tuning (10x less than scratch training)
# ~2–4h on RX 9060 XT
```

### Option B: Train from scratch

```bash
python train.py \
  --data_dir ./dataset \
  --output model.onnx \
  --epochs 100 \
  --batch_size 32 \
  --patch_size 64 \
  --scale 2 \
  --lr 0.001
# ~8–12h on RX 9060 XT
```

### Option C: Quick test run (validate the pipeline works, ~20 min)

```bash
python train.py \
  --data_dir ./dataset \
  --output model.onnx \
  --epochs 5 \
  --batch_size 32 \
  --no_perceptual
# L1 loss only = faster. Use this just to confirm everything runs.
```

### Resume an interrupted run

```bash
python train.py \
  --data_dir ./dataset \
  --resume model.pt \
  --output model.onnx \
  --epochs 100
# Restores model weights, optimizer state, and LR scheduler exactly where you left off
```

### Training flags reference

| Flag | Default | Description |
|---|---|---|
| `--data_dir` | `./dataset` | Folder with 1440p HR images |
| `--output` | `model.onnx` | ONNX export path |
| `--pretrained` | — | Pretrained `.ckpt` for fine-tuning |
| `--resume` | — | Checkpoint to resume from |
| `--epochs` | `50` | Number of training epochs |
| `--batch_size` | `32` | Patches per batch |
| `--patch_size` | `64` | LR patch size (HR = 128px at 2x) |
| `--lr` | `0.001` | Learning rate |
| `--val_freq` | `5` | Validate every N epochs |
| `--no_perceptual` | off | L1 loss only (faster) |
| `--no_amp` | off | Disable mixed precision |
| `--cache_images` | off | Pre-load dataset into RAM |
| `--d` / `--s` / `--m` | 56/12/4 | Model architecture parameters |

**What you'll see during training:**
```
Epoch 1/30 | Batch 0/250 | Loss: 0.0842 | LR: 1.00e-04
→ Epoch 5 | Loss: 0.0431 | PSNR: 28.14 dB | SSIM: 0.8203
  ✓ Best checkpoint saved: model.pt (PSNR: 28.14 dB)
```

PSNR above 30 dB = model is working well. Above 35 dB = excellent.

---

## 5. Test with an image

### Basic test (540p input → 1080p output, auto-detected)

```bash
python upscale.py \
  --input test_540p.png \
  --model model.onnx
# → saves screenshot_540p_sr.png at 1080p
# No --target_res needed: 540 × 2 = 1080 automatically
```

### With side-by-side comparison (model vs bicubic baseline)

```bash
python upscale.py \
  --input screenshot_540p.png \
  --model model.onnx \
  --compare
# → saves screenshot_540p_sr.png and screenshot_540p_sr_compare.png
# Left half: bicubic resize | Right half: model output
```

### Explicit target resolution

```bash
# Force 1080p output from any input size
python upscale.py --input frame.png --model model.onnx --target_res 1920x1080

# Force 1440p output
python upscale.py --input frame.png --model model.onnx --target_res 2560x1440

# Force 4K output (model runs 2x; input is pre-scaled to 1920x1080 first)
python upscale.py --input frame.png --model model.onnx --target_res 3840x2160
```

### With PSNR/SSIM measurement (requires a native HR reference)

```bash
python upscale.py \
  --input screenshot_720p.png \
  --model model.onnx \
  --hr_ref screenshot_1440p_native.png \
  --compare
# → prints PSNR in dB (>30 = acceptable, >35 = good)
```

### Tile mode (for large images or low-memory systems)

```bash
python upscale.py \
  --input screenshot_1440p.png \
  --model model.onnx \
  --tile 256 \
  --overlap 16
```

### What to look for

| Good sign | Problem |
|---|---|
| Sharper edges than bicubic | Blurry → train more epochs |
| Readable HUD text | Ringing/glow → lower perceptual weight |
| Clean flat areas (sky, walls) | Grid artifacts → tune patch size |
| No flickering between frames | Temporal flicker → add temporal input |

---

## 6. INT8 quantization for the XDNA 2 NPU

Once you're happy with the FP32 ONNX model, quantize it for NPU deployment.
INT8 is ~4x faster on the NPU's integer accelerators.

```bash
pip install quark

python -c "
from quark.onnx import ModelQuantizer
q = ModelQuantizer('model.onnx', 'model_int8.onnx')
q.quantize()
print('Done: model_int8.onnx')
"
```

Benchmark both models with `upscale.py` and compare inference time vs visual quality.

---

## 7. Versioning the ONNX model

The model file is too large for Git. Use one of:

**Hugging Face (recommended for public release):**
```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli upload your-username/npu-scope model.onnx
```

**GitHub Releases (simplest):**
Upload `model.onnx` as a binary attachment to each GitHub Release.
Tags like `v0.1-alpha`, `v1.0` give natural versioning.

**Git LFS (stays in GitHub):**
```bash
git lfs install
git lfs track "*.onnx" "*.pt" "*.ckpt"
git add .gitattributes
```

---

## 8. Next: Gamescope integration

The Gamescope backend is the next milestone. The architecture:

```
Game renders at 540p
    → Gamescope intercepts frame (rendervulkan.cpp)
    → [npu-scope backend] GPU→CPU copy
    → ONNX Runtime + VitisAI EP → XDNA 2 NPU
    → NPU outputs 1080p
    → CPU→GPU copy
    → Gamescope flips 1080p to display
```

Target latency budget at 60fps (16ms total):
- GPU→CPU copy: ~1-2ms
- NPU inference (INT8): ~3-5ms
- CPU→GPU copy: ~1-2ms
- Total overhead: ~5-9ms ✓

---

## References

- [FSRCNN paper](https://arxiv.org/abs/1608.00367)
- [suxrobGM/fsrcnn](https://github.com/suxrobGM/fsrcnn) — pretrained checkpoint source
- [AMD Ryzen AI docs](https://ryzenai.docs.amd.com)
- [ONNX Runtime docs](https://onnxruntime.ai/docs/)
- [Gamescope source](https://github.com/ValveSoftware/gamescope)
- [AMD XDNA driver](https://github.com/amd/xdna-driver)
- [amd/RyzenAI-SW SR example](https://github.com/amd/RyzenAI-SW)
