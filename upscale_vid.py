"""
upscale_vid.py — Video super-resolution via ONNX SR model.

Loads the model once and streams frames through ffmpeg pipes — no temp files,
no PNG compression overhead. Frames travel as raw RGB24 bytes between
ffmpeg and Python, processed by the YCbCr SR pipeline on GPU.

Self-contained: does not import from sr/.

Usage:
    python upscale_vid.py --input gameplay.mp4 --model model_perceptual.onnx
    python upscale_vid.py --input gameplay.mp4 --model model_perceptual.onnx --sharpen
    python upscale_vid.py --input gameplay.mp4 --model model_perceptual.onnx --sharpen --sharpen_percent 200
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageFilter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger(__name__)


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(model_path: str):
    providers_priority = ['ROCMExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
    available = ort.get_available_providers()
    providers = [p for p in providers_priority if p in available] or ['CPUExecutionProvider']
    log.info(f"ONNX provider: {providers[0]}")
    return ort.InferenceSession(model_path, providers=providers)


def get_model_info(session) -> dict:
    inp = session.get_inputs()[0]
    dummy = np.zeros((1, 1, 64, 64), dtype=np.float32)
    dummy_out = session.run(None, {inp.name: dummy})[0]
    scale = dummy_out.shape[2] // 64
    return {'input_name': inp.name, 'scale': scale}


# ── YCbCr pipeline ────────────────────────────────────────────────────────────

def upscale_frame(session, info: dict, image: Image.Image) -> Image.Image:
    ycbcr = image.convert('YCbCr')
    y, cb, cr = ycbcr.split()

    y_arr = np.array(y, dtype=np.float32) / 255.0
    y_sr_arr = session.run(None, {info['input_name']: y_arr[np.newaxis, np.newaxis]})[0][0, 0]
    y_sr = Image.fromarray((np.clip(y_sr_arr, 0, 1) * 255).astype(np.uint8), mode='L')

    target_size = y_sr.size
    cb_up = cb.resize(target_size, Image.BICUBIC)
    cr_up = cr.resize(target_size, Image.BICUBIC)
    return Image.merge('YCbCr', [y_sr, cb_up, cr_up]).convert('RGB')


# ── Sharpener (Y-channel only — remove this block to disable) ─────────────────

def sharpen_frame(image: Image.Image, radius: float, percent: int, threshold: int) -> Image.Image:
    y, cb, cr = image.convert('YCbCr').split()
    y_sharp = y.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))
    return Image.merge('YCbCr', (y_sharp, cb, cr)).convert('RGB')

# ─────────────────────────────────────────────────────────────────────────────


# ── Video probe ───────────────────────────────────────────────────────────────

def probe_video(path: str) -> dict:
    cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json',
           '-show_streams', path]
    data = json.loads(subprocess.check_output(cmd))

    video  = next(s for s in data['streams'] if s['codec_type'] == 'video')
    fps_n, fps_d = video['r_frame_rate'].split('/')
    fps = float(fps_n) / float(fps_d)

    has_audio = any(s['codec_type'] == 'audio' for s in data['streams'])

    return {
        'width':     video['width'],
        'height':    video['height'],
        'fps':       fps,
        'frames':    int(video.get('nb_frames', 0)),
        'has_audio': has_audio,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Video super-resolution via ONNX SR model (ffmpeg pipe, no temp files)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--input',  required=True,        help='Input video path')
    parser.add_argument('--model',  default='model.onnx', help='ONNX model file')
    parser.add_argument('--output', default=None,         help='Output video path (default: input_sr.mp4)')

    # ── Sharpen (remove this block + sharpen_frame() to disable) ─────────────
    parser.add_argument('--sharpen',           action='store_true', help='Apply unsharp mask after SR')
    parser.add_argument('--sharpen_radius',    type=float, default=1.5)
    parser.add_argument('--sharpen_percent',   type=int,   default=150)
    parser.add_argument('--sharpen_threshold', type=int,   default=3)
    # ─────────────────────────────────────────────────────────────────────────

    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = args.output or str(input_path.with_name(input_path.stem + '_sr.mp4'))

    if not input_path.exists():
        log.error(f"Input not found: {input_path}")
        sys.exit(1)

    log.info(f"Input:  {input_path}")
    log.info(f"Model:  {args.model}")
    log.info(f"Output: {output_path}")

    vid   = probe_video(str(input_path))
    in_w  = vid['width']
    in_h  = vid['height']
    fps   = vid['fps']
    total = vid['frames']
    log.info(f"Video: {in_w}x{in_h} @ {fps:.2f} fps | frames: {total or '?'}")

    session    = load_model(args.model)
    model_info = get_model_info(session)
    scale      = model_info['scale']
    out_w      = in_w * scale
    out_h      = in_h * scale
    log.info(f"Scale: {scale}x → output {out_w}x{out_h}")
    if args.sharpen:
        log.info(f"Sharpener: radius={args.sharpen_radius} percent={args.sharpen_percent} threshold={args.sharpen_threshold}")

    in_frame_bytes  = in_w  * in_h  * 3
    out_frame_bytes = out_w * out_h * 3

    # ── ffmpeg reader: decode video → raw RGB24 bytes on stdout ───────────────
    reader = subprocess.Popen([
        'ffmpeg', '-i', str(input_path),
        '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-loglevel', 'error', 'pipe:1',
    ], stdout=subprocess.PIPE)

    # ── ffmpeg writer: raw RGB24 bytes on stdin → encoded video + original audio
    writer_cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{out_w}x{out_h}', '-r', str(fps),
        '-loglevel', 'error',
        '-i', 'pipe:0',
    ]
    if vid['has_audio']:
        writer_cmd += ['-i', str(input_path), '-map', '0:v', '-map', '1:a?', '-c:a', 'copy']
    writer_cmd += ['-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', '-shortest', output_path]

    writer = subprocess.Popen(writer_cmd, stdin=subprocess.PIPE)

    # ── Frame loop ────────────────────────────────────────────────────────────
    log.info("Processing frames...")
    frame_idx = 0
    t_start   = time.perf_counter()

    try:
        while True:
            raw = reader.stdout.read(in_frame_bytes)
            if len(raw) < in_frame_bytes:
                break

            image  = Image.frombuffer('RGB', (in_w, in_h), raw)
            result = upscale_frame(session, model_info, image)

            # ── Sharpen (remove this block to disable) ────────────────────
            if args.sharpen:
                result = sharpen_frame(result, args.sharpen_radius,
                                       args.sharpen_percent, args.sharpen_threshold)
            # ─────────────────────────────────────────────────────────────

            writer.stdin.write(result.tobytes())
            frame_idx += 1

            if frame_idx % 30 == 0:
                elapsed  = time.perf_counter() - t_start
                fps_proc = frame_idx / elapsed
                eta      = (total - frame_idx) / fps_proc if (total and fps_proc > 0) else 0
                log.info(f"Frame {frame_idx}/{total or '?'} | {fps_proc:.1f} fps | ETA: {eta:.0f}s")

    finally:
        writer.stdin.close()
        reader.stdout.close()
        reader.wait()
        writer.wait()

    log.info(f"Done: {output_path} ({frame_idx} frames)")


if __name__ == '__main__':
    main()
