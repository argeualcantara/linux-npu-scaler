"""
upscale.py — Runs a frame through the ONNX model and saves the upscaled result.

Basic usage:
    python upscale.py --input screenshot_720p.png --model model.onnx
    # → output: screenshot_720p_sr.png  (1440p, 2x model scale)

Explicit target resolution:
    python upscale.py --input screenshot.png --model model.onnx --target_res 1920x1080
    python upscale.py --input screenshot.png --model model.onnx --target_res 2560x1440
    python upscale.py --input screenshot.png --model model.onnx --target_res 3840x2160

    The input is pre-scaled to (target / model_scale) before inference, so the
    model output lands exactly on the requested resolution. Aspect ratio is always
    preserved; if the target ratio differs from the input, the output is padded.

With tile mode (for images that don't fit in memory at once):
    python upscale.py --input screenshot_1440p.png --model model.onnx --tile 256

Resolution expectations:
    Training source:    1440p native (no upscaler active)
    Training LR:        720p  (2x bicubic downsample, generated on-the-fly)
    Inference (PC):     720p  → 1440p  (model native)
    Inference (Ally X): 540p  → 1080p  (same 2x ratio)
    Custom target:      any   → --target_res WxH
"""

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image


def parse_target_res(target_res: str) -> tuple[int, int]:
    """
    Parses a resolution string into (width, height).

    Accepted formats:
        "1920x1080"   → (1920, 1080)
        "2560x1440"   → (2560, 1440)
        "3840x2160"   → (3840, 2160)

    Raises ValueError on invalid input.
    """
    try:
        w, h = target_res.lower().split('x')
        return int(w), int(h)
    except Exception:
        raise ValueError(
            f"Invalid --target_res '{target_res}'. "
            "Expected format: WIDTHxHEIGHT, e.g. 1920x1080"
        )


def prepare_input_for_target(
    image: Image.Image,
    target_w: int,
    target_h: int,
    model_scale: int,
) -> tuple[Image.Image, tuple[int, int], tuple[int, int, int, int]]:
    """
    Pre-scales the input image so that after the model's fixed upscale
    the output lands exactly on (target_w, target_h).

    Strategy:
      1. Compute the LR size the model needs: (target_w / scale, target_h / scale)
      2. Fit the input inside that LR box preserving aspect ratio (letterbox/pillarbox)
      3. Pad to exact LR size with black bars if the aspect ratios differ
      4. After inference, crop the black bars from the SR output

    Returns:
        prepared:    the pre-scaled + padded LR image ready for inference
        lr_size:     (lr_w, lr_h) — the LR dimensions passed to the model
        crop_box:    (left, top, right, bottom) in SR coordinates to crop
                     black padding after inference; (0, 0, target_w, target_h)
                     when no padding was needed
    """
    lr_w = target_w // model_scale
    lr_h = target_h // model_scale

    src_w, src_h = image.size
    scale_x = lr_w / src_w
    scale_y = lr_h / src_h
    fit_scale = min(scale_x, scale_y)  # preserve aspect ratio

    fit_w = round(src_w * fit_scale)
    fit_h = round(src_h * fit_scale)

    # High-quality resize to the fitting dimensions
    resized = image.resize((fit_w, fit_h), Image.LANCZOS)

    # Pad to exact LR size (black bars on the sides that don't fit)
    pad_left = (lr_w - fit_w) // 2
    pad_top  = (lr_h - fit_h) // 2

    if pad_left == 0 and pad_top == 0 and fit_w == lr_w and fit_h == lr_h:
        # No padding needed — aspect ratios match exactly
        prepared = resized
        crop_box = (0, 0, target_w, target_h)
    else:
        prepared = Image.new('RGB', (lr_w, lr_h), (0, 0, 0))
        prepared.paste(resized, (pad_left, pad_top))

        # Crop box in SR space to remove the padding after inference
        crop_box = (
            pad_left  * model_scale,
            pad_top   * model_scale,
            (pad_left + fit_w) * model_scale,
            (pad_top  + fit_h) * model_scale,
        )

    return prepared, (lr_w, lr_h), crop_box


def detect_model_scale(session) -> int:
    """
    Runs a tiny dummy input through the model to detect its upscale factor.
    Works for any scale (2x, 3x, 4x) without needing it hardcoded.
    """
    input_name = session.get_inputs()[0].name
    dummy = np.zeros((1, 3, 64, 64), dtype=np.float32)
    out = session.run(None, {input_name: dummy})[0]
    return out.shape[2] // 64


def load_model(model_path: str):
    """
    Loads the ONNX model and selects the best available execution provider.

    Priority order:
      1. ROCMExecutionProvider  — RX 9060 XT (training machine)
      2. CUDAExecutionProvider  — NVIDIA GPU
      3. CPUExecutionProvider   — universal fallback

    On the Ally X with Ryzen AI SDK installed, VitisAIExecutionProvider
    would be added here to dispatch to the XDNA 2 NPU.
    """
    import onnxruntime as ort

    providers_priority = [
        'ROCMExecutionProvider',
        'CUDAExecutionProvider',
        'CPUExecutionProvider',
    ]

    available = ort.get_available_providers()
    providers = [p for p in providers_priority if p in available]

    if not providers:
        providers = ['CPUExecutionProvider']

    print(f"Available ONNX providers: {available}")
    print(f"Using: {providers[0]}")

    session = ort.InferenceSession(model_path, providers=providers)

    inp = session.get_inputs()[0]
    out = session.get_outputs()[0]
    print(f"Input:  {inp.name} | shape: {inp.shape} | dtype: {inp.type}")
    print(f"Output: {out.name} | shape: {out.shape} | dtype: {out.type}")

    return session


def preprocess(image: Image.Image) -> np.ndarray:
    """
    PIL Image → float32 numpy array normalized to [0, 1].
    Output shape: (1, 3, H, W) — NCHW format expected by the model.
    """
    img = image.convert('RGB')
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[np.newaxis, ...]  # HWC → CHW → NCHW
    return arr


def postprocess(output: np.ndarray) -> Image.Image:
    """
    NCHW float32 numpy array [0, 1] → PIL Image.
    """
    arr = output[0]               # drop batch dim: (3, H, W)
    arr = arr.transpose(1, 2, 0)  # CHW → HWC
    arr = np.clip(arr, 0, 1)
    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr)


def upscale_full(session, image: Image.Image) -> Image.Image:
    """
    Upscales the entire image in a single inference pass.

    Works well up to 1440p input (→ 2880p output with scale=2).
    For larger inputs use upscale_tiled() to avoid OOM.

    Typical use: 720p → 1440p on the training machine.
    Ally X use:  540p → 1080p using the same model.
    """
    input_name = session.get_inputs()[0].name
    arr = preprocess(image)

    t0 = time.perf_counter()
    output = session.run(None, {input_name: arr})[0]
    elapsed = (time.perf_counter() - t0) * 1000

    print(f"Inference time: {elapsed:.1f} ms")
    return postprocess(output)


def upscale_tiled(session, image: Image.Image, tile_size: int = 256, overlap: int = 16) -> Image.Image:
    """
    Upscales by processing overlapping tiles and blending them together.
    Use this when the full image doesn't fit in GPU/NPU memory.

    The overlap + Gaussian weighting eliminates visible seams between tiles.

    Args:
        tile_size: LR tile size in pixels (default 256)
        overlap:   Overlap between adjacent tiles in LR pixels (default 16)
                   Larger overlap = smoother seams, more compute
    """
    input_name = session.get_inputs()[0].name

    # Detect scale factor from a dummy run
    dummy = np.zeros((1, 3, tile_size, tile_size), dtype=np.float32)
    dummy_out = session.run(None, {input_name: dummy})[0]
    scale = dummy_out.shape[2] // tile_size
    print(f"Scale factor detected: {scale}x")

    w, h = image.size
    out_w, out_h = w * scale, h * scale

    output_canvas = np.zeros((out_h, out_w, 3), dtype=np.float32)
    weight_canvas = np.zeros((out_h, out_w, 1), dtype=np.float32)

    step = tile_size - overlap
    tiles_processed = 0

    for y in range(0, h, step):
        for x in range(0, w, step):
            # Clamp tile coordinates to image boundaries
            x1 = min(x, w - tile_size)
            y1 = min(y, h - tile_size)
            x2 = x1 + tile_size
            y2 = y1 + tile_size

            tile_lr = image.crop((x1, y1, x2, y2))
            arr = preprocess(tile_lr)
            tile_sr = session.run(None, {input_name: arr})[0]
            tile_sr = tile_sr[0].transpose(1, 2, 0)  # NCHW → HWC

            # SR canvas coordinates
            ox1, oy1 = x1 * scale, y1 * scale
            ox2, oy2 = x2 * scale, y2 * scale

            # Gaussian weight: center of tile has full weight, edges fade to 0.
            # This blends overlapping tiles smoothly with no visible seam.
            tile_h, tile_w = tile_sr.shape[:2]
            weight = _gaussian_weight(tile_h, tile_w)

            output_canvas[oy1:oy2, ox1:ox2] += tile_sr * weight
            weight_canvas[oy1:oy2, ox1:ox2] += weight
            tiles_processed += 1

    print(f"Tiles processed: {tiles_processed}")

    output_canvas /= np.maximum(weight_canvas, 1e-8)
    output_canvas = np.clip(output_canvas, 0, 1)
    output_canvas = (output_canvas * 255).astype(np.uint8)

    return Image.fromarray(output_canvas)


def _gaussian_weight(h: int, w: int) -> np.ndarray:
    """
    2D Gaussian weight mask. Center pixel weight ≈ 1, edges ≈ 0.
    Eliminates visible seams when blending adjacent tiles.
    """
    from scipy.ndimage import gaussian_filter
    weight = np.zeros((h, w, 1), dtype=np.float32)
    weight[h // 4: 3 * h // 4, w // 4: 3 * w // 4, 0] = 1.0
    weight = gaussian_filter(weight[:, :, 0], sigma=h // 4)[:, :, np.newaxis]
    weight = weight / weight.max()
    return weight


def compare_side_by_side(original: Image.Image, upscaled: Image.Image, output_path: str):
    """
    Saves a side-by-side comparison image:
      Left:  original bicubic-upscaled to SR size (baseline)
      Right: model output

    Use this to evaluate whether the model adds real detail
    over a simple bicubic resize.
    """
    target_w, target_h = upscaled.size
    original_resized = original.resize((target_w, target_h), Image.BICUBIC)

    comparison = Image.new('RGB', (target_w * 2, target_h))
    comparison.paste(original_resized, (0, 0))
    comparison.paste(upscaled, (target_w, 0))

    try:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(comparison)
        draw.text((10, 10), "Bicubic (baseline)", fill=(255, 255, 0))
        draw.text((target_w + 10, 10), "ONNX model", fill=(0, 255, 0))
    except Exception:
        pass

    comparison.save(output_path)
    print(f"Comparison saved: {output_path}")


def compute_psnr(hr: Image.Image, sr: Image.Image) -> float:
    """
    PSNR (Peak Signal-to-Noise Ratio) in dB.
    Higher is better. >30 dB is acceptable, >35 dB is good for SR.
    Requires a native HR reference image (not bicubic-upscaled).
    """
    hr_arr = np.array(hr.convert('RGB'), dtype=np.float32)
    sr_arr = np.array(sr.resize(hr.size, Image.BICUBIC).convert('RGB'), dtype=np.float32)
    mse = np.mean((hr_arr - sr_arr) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def main():
    parser = argparse.ArgumentParser(description='Upscale an image using the ONNX SR model')
    parser.add_argument('--input',      required=True,         help='Input image path (LR)')
    parser.add_argument('--model',      default='model.onnx',  help='ONNX model file')
    parser.add_argument('--output',     default=None,          help='Output path (default: input_sr.png)')
    parser.add_argument('--target_res', default=None,
                        help='Desired output resolution as WxH, e.g. 1920x1080 or 2560x1440. '
                             'If omitted, output is input_size * model_scale (e.g. 720p → 1440p).')
    parser.add_argument('--tile',       type=int, default=0,   help='Tile size in px (0 = no tiling)')
    parser.add_argument('--overlap',    type=int, default=16,  help='Tile overlap in px')
    parser.add_argument('--compare',    action='store_true',   help='Save side-by-side comparison')
    parser.add_argument('--hr_ref',     default=None,          help='Native HR reference for PSNR')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else \
        input_path.with_name(input_path.stem + '_sr.png')

    print(f"\n{'='*50}")
    print(f"Input:      {input_path}")
    print(f"Model:      {args.model}")
    print(f"Output:     {output_path}")
    if args.target_res:
        print(f"Target res: {args.target_res}")
    print(f"{'='*50}\n")

    image = Image.open(input_path).convert('RGB')
    print(f"Input resolution: {image.width}x{image.height}")

    session = load_model(args.model)
    model_scale = detect_model_scale(session)
    print(f"Model scale: {model_scale}x")

    # ── Target resolution pre-processing ──────────────────────────────────
    crop_box = None
    inference_image = image  # default: pass the image as-is

    if args.target_res:
        target_w, target_h = parse_target_res(args.target_res)

        # Validate: target must be larger than input
        if target_w < image.width or target_h < image.height:
            print(
                f"Warning: target {target_w}x{target_h} is smaller than "
                f"input {image.width}x{image.height}. "
                "Proceeding anyway — output will be downscaled after inference."
            )

        # Validate: target must be divisible by model scale
        if target_w % model_scale != 0 or target_h % model_scale != 0:
            # Round up to the nearest multiple and warn
            target_w = ((target_w + model_scale - 1) // model_scale) * model_scale
            target_h = ((target_h + model_scale - 1) // model_scale) * model_scale
            print(
                f"Warning: target resolution adjusted to {target_w}x{target_h} "
                f"(must be divisible by model scale {model_scale}x)."
            )

        inference_image, lr_size, crop_box = prepare_input_for_target(
            image, target_w, target_h, model_scale
        )
        print(
            f"Pre-scaled input to: {inference_image.width}x{inference_image.height} "
            f"(LR) → {target_w}x{target_h} (SR target)"
        )
    else:
        native_w = image.width  * model_scale
        native_h = image.height * model_scale
        print(f"No target resolution set — native output: {native_w}x{native_h}")

    # ── Inference ──────────────────────────────────────────────────────────
    t_total = time.perf_counter()
    if args.tile > 0:
        print(f"\nTile mode (tile_size={args.tile}, overlap={args.overlap})")
        result = upscale_tiled(session, inference_image, tile_size=args.tile, overlap=args.overlap)
    else:
        print("\nFull image mode")
        result = upscale_full(session, inference_image)

    total_ms = (time.perf_counter() - t_total) * 1000

    # ── Post-processing: crop padding if target_res was used ───────────────
    if crop_box is not None:
        result = result.crop(crop_box)
        print(f"Cropped padding: {crop_box}")

    print(f"Output resolution: {result.width}x{result.height}")
    print(f"Total time: {total_ms:.1f} ms")

    result.save(output_path)
    print(f"\nSaved: {output_path}")

    if args.compare:
        compare_path = output_path.with_name(output_path.stem + '_compare.png')
        compare_side_by_side(image, result, str(compare_path))

    if args.hr_ref:
        hr = Image.open(args.hr_ref).convert('RGB')
        psnr = compute_psnr(hr, result)
        print(f"\nPSNR vs HR reference: {psnr:.2f} dB")
        if psnr > 35:
            print("  → Excellent")
        elif psnr > 30:
            print("  → Good")
        else:
            print("  → Below target — model needs more training or data")


if __name__ == '__main__':
    main()
