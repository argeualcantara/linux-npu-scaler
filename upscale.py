"""
upscale.py — Image upscaling via ONNX SR model (YCbCr pipeline).

Supports two model modes:
  - zoo   : ONNX Model Zoo super-resolution-10.onnx (fixed 224→672, 3x scale)
             Output is downsampled to 2x after inference (e.g. 540p → 1080p).
  - custom: Our trained FSRCNN model (dynamic input, 2x scale)

The mode is detected automatically from the model's input shape.

Usage:
    # Zoo model (540p → 1080p)
    python upscale.py --input test_540.png --model super_resolution.onnx

    # Our trained model
    python upscale.py --input screenshot_540p.png --model model.onnx

    # Side-by-side comparison
    python upscale.py --input screenshot.png --model super_resolution.onnx --compare

    # Explicit target resolution (custom model only)
    python upscale.py --input screenshot.png --model model.onnx --target_res 1920x1080
"""

import argparse
import time
from pathlib import Path

from PIL import Image

from sr.inference import (
    load_model, get_model_info,
    upscale_zoo, upscale_custom, upscale_custom_tiled,
    parse_target_res, prepare_input_for_target,
    compare_side_by_side, compute_psnr,
)


def main():
    parser = argparse.ArgumentParser(
        description='Upscale an image using an ONNX SR model (YCbCr pipeline)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--input',      required=True,         help='Input image path')
    parser.add_argument('--model',      default='model.onnx',  help='ONNX model file')
    parser.add_argument('--output',     default=None,          help='Output path (default: input_sr.png)')
    parser.add_argument('--target_res', default=None,
                        help='Force output resolution WxH — custom model only (e.g. 1920x1080)')
    parser.add_argument('--tile',       type=int, default=0,
                        help='Tile size in px for custom model (0 = full image)')
    parser.add_argument('--overlap',    type=int, default=64,  help='Tile overlap in px')
    parser.add_argument('--compare',    action='store_true',   help='Save side-by-side comparison')
    parser.add_argument('--hr_ref',     default=None,          help='Native HR reference for PSNR')
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output) if args.output else \
        input_path.with_name(input_path.stem + '_sr.png')

    print(f"\n{'='*50}")
    print(f"Input:  {input_path}")
    print(f"Model:  {args.model}")
    print(f"Output: {output_path}")
    print(f"{'='*50}\n")

    image = Image.open(input_path).convert('RGB')
    print(f"Input resolution: {image.width}x{image.height}")

    session = load_model(args.model)
    info    = get_model_info(session)
    print(f"Model type: {'Zoo (fixed input)' if info['is_zoo'] else 'Custom (dynamic)'} | "
          f"Scale: {info['scale']}x | "
          f"Fixed size: {info['fixed_size']}")

    t_total = time.perf_counter()

    if info['is_zoo']:
        if args.target_res:
            print("Note: --target_res is ignored for zoo model (fixed 3x scale)")
        print(f"\nRunning zoo model (tile-and-stitch {info['fixed_size'][0]}px tiles)")
        result = upscale_zoo(session, info, image)
        target_w, target_h = image.width * 2, image.height * 2
        print(f"Downsampling {result.width}x{result.height} → {target_w}x{target_h}")
        result = result.resize((target_w, target_h), Image.LANCZOS)

    else:
        inference_image = image
        crop_box = None

        if args.target_res:
            tw, th = parse_target_res(args.target_res)
            if tw % info['scale'] != 0 or th % info['scale'] != 0:
                tw = ((tw + info['scale'] - 1) // info['scale']) * info['scale']
                th = ((th + info['scale'] - 1) // info['scale']) * info['scale']
                print(f"Target adjusted to {tw}x{th}")
            inference_image, crop_box = prepare_input_for_target(
                image, tw, th, info['scale'])
            print(f"Pre-scaled to {inference_image.width}x{inference_image.height} → target {tw}x{th}")
        else:
            print(f"Native output: {image.width * info['scale']}x{image.height * info['scale']}")

        if args.tile > 0:
            print(f"\nTile mode (tile={args.tile}, overlap={args.overlap})")
            result = upscale_custom_tiled(session, info, inference_image,
                                          tile_size=args.tile, overlap=args.overlap)
        else:
            print("\nFull image mode")
            result = upscale_custom(session, info, inference_image)

        if crop_box:
            result = result.crop(crop_box)

    total_ms = (time.perf_counter() - t_total) * 1000
    print(f"\nOutput resolution: {result.width}x{result.height}")
    print(f"Total time: {total_ms:.1f} ms")

    result.save(output_path)
    print(f"Saved: {output_path}")

    if args.compare:
        compare_path = output_path.with_name(output_path.stem + '_compare.png')
        compare_side_by_side(image, result, str(compare_path))

    if args.hr_ref:
        hr   = Image.open(args.hr_ref).convert('RGB')
        psnr = compute_psnr(hr, result)
        print(f"\nPSNR vs HR reference: {psnr:.2f} dB")
        print("  → Excellent" if psnr > 35 else "  → Good" if psnr > 30 else "  → Below target")


if __name__ == '__main__':
    main()
