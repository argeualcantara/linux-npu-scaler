import time

import numpy as np
from PIL import Image

from .ycbcr import split_ycbcr, merge_ycbcr


# ─────────────────────────────────────────────
# Zoo model (fixed input, tile-and-stitch)
# ─────────────────────────────────────────────

def upscale_zoo(session, info: dict, image: Image.Image) -> Image.Image:
    """
    ONNX Model Zoo super-resolution-10: fixed 224x224 input → 672x672 output (3x).

    Tiles with overlap context: each tile is run with `overlap` extra pixels on
    each side so the CNN has full receptive-field context. Only the inner region
    (no border) is written to the output — no blending needed, no seams.
    """
    input_name = info['input_name']
    fixed_w, fixed_h = info['fixed_size']   # 224, 224
    scale = info['scale']                    # 3
    overlap = 32                             # LR context pixels on each side
    inner_w = fixed_w - 2 * overlap         # 160
    inner_h = fixed_h - 2 * overlap         # 160

    y, cb, cr = split_ycbcr(image)
    y_arr = np.array(y, dtype=np.float32) / 255.0
    src_h, src_w = y_arr.shape
    y_canvas = np.zeros((src_h * scale, src_w * scale), dtype=np.float32)

    tiles = 0
    elapsed = 0.0

    for iy in range(0, src_h, inner_h):
        for ix in range(0, src_w, inner_w):
            ix1 = max(0, ix - overlap);  ctx_x = ix - ix1
            iy1 = max(0, iy - overlap);  ctx_y = iy - iy1
            ix2 = min(src_w, ix + inner_w + overlap)
            iy2 = min(src_h, iy + inner_h + overlap)

            tile = y_arr[iy1:iy2, ix1:ix2]
            th, tw = tile.shape

            if th < fixed_h or tw < fixed_w:
                tile_input = np.zeros((fixed_h, fixed_w), dtype=np.float32)
                tile_input[:th, :tw] = tile
            else:
                tile_input = tile

            t0 = time.perf_counter()
            inp_arr = tile_input[np.newaxis, np.newaxis, ...]
            tile_sr = session.run(None, {input_name: inp_arr})[0][0, 0]
            elapsed = (time.perf_counter() - t0) * 1000

            sr_x1 = ctx_x * scale
            sr_y1 = ctx_y * scale
            iw_sr = min(inner_w, src_w - ix, tw - ctx_x) * scale
            ih_sr = min(inner_h, src_h - iy, th - ctx_y) * scale

            y_canvas[iy * scale: iy * scale + ih_sr,
                     ix * scale: ix * scale + iw_sr] = \
                tile_sr[sr_y1: sr_y1 + ih_sr, sr_x1: sr_x1 + iw_sr]
            tiles += 1

    print(f"Tiles processed: {tiles} ({elapsed:.0f} ms last tile)")
    y_sr_arr = np.clip(y_canvas, 0, 1)
    y_sr = Image.fromarray((y_sr_arr * 255).astype(np.uint8), mode='L')
    return merge_ycbcr(y_sr, cb, cr)


# ─────────────────────────────────────────────
# Custom model (dynamic input)
# ─────────────────────────────────────────────

def upscale_custom(session, info: dict, image: Image.Image) -> Image.Image:
    """
    Our trained ESPCN: dynamic input size, 2x scale.
    Full image in a single pass — fast and no tiling artifacts.
    """
    input_name = info['input_name']
    y, cb, cr = split_ycbcr(image)
    y_arr = np.array(y, dtype=np.float32) / 255.0
    inp_arr = y_arr[np.newaxis, np.newaxis, ...]

    t0 = time.perf_counter()
    y_sr_arr = session.run(None, {input_name: inp_arr})[0][0, 0]
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"Inference time: {elapsed:.1f} ms")

    y_sr_arr = np.clip(y_sr_arr, 0, 1)
    y_sr = Image.fromarray((y_sr_arr * 255).astype(np.uint8), mode='L')
    return merge_ycbcr(y_sr, cb, cr)


def upscale_custom_tiled(session, info: dict, image: Image.Image,
                         tile_size: int = 256, overlap: int = 64) -> Image.Image:
    """
    Tiled upscale for large inputs. Each tile is inferred with `overlap` pixels
    of extra context on each side; only the inner region (no border) is written
    to the output. No blending — each output pixel comes from one tile center
    where the CNN had full receptive-field context.
    """
    input_name = info['input_name']
    scale = info['scale']
    inner = tile_size - 2 * overlap

    y, cb, cr = split_ycbcr(image)
    y_arr = np.array(y, dtype=np.float32) / 255.0
    src_h, src_w = y_arr.shape
    y_canvas = np.zeros((src_h * scale, src_w * scale), dtype=np.float32)

    tiles = 0
    for iy in range(0, src_h, inner):
        for ix in range(0, src_w, inner):
            ix1 = max(0, ix - overlap);  ctx_x = ix - ix1
            iy1 = max(0, iy - overlap);  ctx_y = iy - iy1
            ix2 = min(src_w, ix + inner + overlap)
            iy2 = min(src_h, iy + inner + overlap)

            tile = y_arr[iy1:iy2, ix1:ix2]
            tile_sr = session.run(None, {input_name: tile[np.newaxis, np.newaxis, ...]})[0][0, 0]

            sr_x1 = ctx_x * scale
            sr_y1 = ctx_y * scale
            iw_sr = min(inner, src_w - ix) * scale
            ih_sr = min(inner, src_h - iy) * scale

            y_canvas[iy * scale: iy * scale + ih_sr,
                     ix * scale: ix * scale + iw_sr] = \
                tile_sr[sr_y1: sr_y1 + ih_sr, sr_x1: sr_x1 + iw_sr]
            tiles += 1

    print(f"Tiles processed: {tiles}")
    y_sr_arr = np.clip(y_canvas, 0, 1)
    y_sr = Image.fromarray((y_sr_arr * 255).astype(np.uint8), mode='L')
    return merge_ycbcr(y_sr, cb, cr)


# ─────────────────────────────────────────────
# Resolution helpers
# ─────────────────────────────────────────────

def parse_target_res(s: str) -> tuple:
    try:
        w, h = s.lower().split('x')
        return int(w), int(h)
    except Exception:
        raise ValueError(f"Invalid --target_res '{s}'. Expected WxH e.g. 1920x1080")


def prepare_input_for_target(image, target_w, target_h, scale):
    lr_w, lr_h = target_w // scale, target_h // scale
    src_w, src_h = image.size
    fit = min(lr_w / src_w, lr_h / src_h)
    fw, fh = round(src_w * fit), round(src_h * fit)
    resized = image.resize((fw, fh), Image.LANCZOS)
    pl, pt = (lr_w - fw) // 2, (lr_h - fh) // 2
    if pl == 0 and pt == 0 and fw == lr_w and fh == lr_h:
        return resized, (0, 0, target_w, target_h)
    padded = Image.new('RGB', (lr_w, lr_h), (0, 0, 0))
    padded.paste(resized, (pl, pt))
    crop = (pl * scale, pt * scale, (pl + fw) * scale, (pt + fh) * scale)
    return padded, crop


# ─────────────────────────────────────────────
# Comparison and metrics
# ─────────────────────────────────────────────

def compare_side_by_side(original: Image.Image, upscaled: Image.Image, path: str):
    tw, th = upscaled.size
    baseline = original.resize((tw, th), Image.BICUBIC)
    comp = Image.new('RGB', (tw * 2, th))
    comp.paste(baseline, (0, 0))
    comp.paste(upscaled, (tw, 0))
    try:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(comp)
        draw.text((10, 10), "Bicubic (baseline)", fill=(255, 255, 0))
        draw.text((tw + 10, 10), "ONNX SR model", fill=(0, 255, 0))
    except Exception:
        pass
    comp.save(path)
    print(f"Comparison saved: {path}")


def compute_psnr(hr: Image.Image, sr: Image.Image) -> float:
    hr_arr = np.array(hr.convert('RGB'), dtype=np.float32)
    sr_arr = np.array(sr.resize(hr.size, Image.BICUBIC).convert('RGB'), dtype=np.float32)
    mse = np.mean((hr_arr - sr_arr) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def difference_image(upscaled: Image.Image, path: str, amplify: float = 5.0):
    """
    Saves a visual diff between the SR model output and a bicubic baseline
    at the same resolution.

    Positive diff (model sharper): green channel
    Negative diff (model darker/softer): red channel
    Amplified by `amplify` so subtle differences are visible.
    """
    tw, th = upscaled.size
    bicubic = upscaled.resize((tw // 2, th // 2), Image.BICUBIC) \
                       .resize((tw, th), Image.BICUBIC)

    sr_arr  = np.array(upscaled.convert('RGB'), dtype=np.float32)
    bic_arr = np.array(bicubic.convert('RGB'),  dtype=np.float32)

    diff = (sr_arr - bic_arr) * amplify

    r = np.clip(-diff, 0, 255).mean(axis=2)
    g = np.clip( diff, 0, 255).mean(axis=2)
    b = np.zeros_like(r)

    vis = np.stack([r, g, b], axis=2).astype(np.uint8)
    Image.fromarray(vis).save(path)
    print(f"Difference image saved: {path}")
    print(f"  Green = model sharper than bicubic | Red = model softer than bicubic | Amplify: {amplify}x")
