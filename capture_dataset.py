"""
capture_dataset.py — Captures native 1440p frames from games for use as HR training data.

IMPORTANT: Always capture at native 1440p with NO upscaler active (no FSR, no DLSS,
no XeSS). The model learns from real 1440p detail — upscaled frames defeat the purpose.

From a recorded video:
    python capture_dataset.py video --source recording_1440p.mp4 --output ./dataset

From live screen capture while playing:
    python capture_dataset.py screen --output ./dataset --duration 300

Remove near-duplicate frames:
    python capture_dataset.py dedup --dir ./dataset
"""

import argparse
import time
from pathlib import Path


def extract_from_video(source: str, output_dir: Path, every_n: int, min_height: int):
    """
    Extracts frames from a video file using ffmpeg.
    Takes 1 frame every N frames to avoid redundancy.

    The video should be a native 1440p recording with no upscaler active.
    OBS recommended settings: 2560x1440, lossless or high-bitrate H.264/AV1.

    Args:
        source:     Path to the video file
        output_dir: Where to save extracted frames
        every_n:    Extract 1 frame every N (default 30 = ~1fps at 30fps video)
        min_height: Minimum frame height to keep (default 1440)
    """
    import subprocess

    output_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = str(output_dir / 'frame_%06d.png')

    # ffmpeg filter:
    #   select=not(mod(n,N))  → pick 1 frame every N
    #   scale=-1:1440         → ensure height is exactly 1440px (keeps aspect ratio)
    #   flags=lanczos         → high-quality downscale if needed
    cmd = [
        'ffmpeg', '-i', source,
        '-vf', f'select=not(mod(n\\,{every_n})),scale=-1:{min_height}:flags=lanczos',
        '-vsync', 'vfr',
        '-q:v', '1',        # near-lossless PNG quality
        output_pattern,
        '-y'
    ]

    print(f"Extracting frames from: {source}")
    print(f"Target height: {min_height}px | 1 frame every {every_n}")
    print(f"Command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    frames = list(output_dir.glob('frame_*.png'))
    print(f"Frames extracted: {len(frames)}")
    return frames


def capture_screen(output_dir: Path, duration: int, fps: float, min_height: int):
    """
    Captures the screen in real-time while you play.

    Before running:
      - Set the game to native 1440p resolution
      - Disable ALL upscalers (FSR, DLSS, XeSS, NIS, Gamescope SR)
      - Disable post-process sharpening filters
      - Use the highest quality preset you can maintain at 1440p

    Args:
        output_dir: Where to save captured frames
        duration:   How many seconds to capture
        fps:        Capture rate (2.0 FPS is enough; avoid redundancy)
        min_height: Minimum frame height; frames smaller than this are upscaled
    """
    try:
        import mss
        from PIL import Image
    except ImportError:
        print("Install required packages: pip install mss Pillow")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    interval = 1.0 / fps
    frame_count = 0
    start = time.time()

    print(f"Capturing screen for {duration}s at {fps} FPS...")
    print("Switch to your game now! (native 1440p, no upscaler)")
    time.sleep(3)

    with mss.mss() as sct:
        monitor = sct.monitors[1]  # primary monitor
        while time.time() - start < duration:
            t0 = time.time()

            screenshot = sct.grab(monitor)
            img = Image.frombytes('RGB', screenshot.size, screenshot.bgra, 'raw', 'BGRX')

            # Skip very dark frames (loading screens, black fades, etc.)
            brightness = sum(img.convert('L').getdata()) / (img.width * img.height)
            if brightness > 20:
                # Upscale if below minimum height (should not happen on 1440p monitor)
                if img.height < min_height:
                    ratio = min_height / img.height
                    img = img.resize(
                        (int(img.width * ratio), min_height),
                        Image.LANCZOS
                    )

                path = output_dir / f'screen_{frame_count:06d}.png'
                img.save(path)
                frame_count += 1

                if frame_count % 50 == 0:
                    elapsed = time.time() - start
                    print(f"  {frame_count} frames captured ({elapsed:.0f}s / {duration}s)")

            sleep_time = interval - (time.time() - t0)
            if sleep_time > 0:
                time.sleep(sleep_time)

    print(f"\nCapture complete: {frame_count} frames saved to {output_dir}")


def deduplicate(dataset_dir: Path, threshold: float = 0.95):
    """
    Removes near-duplicate frames (static menus, loading screens, paused scenes).
    Compares color histograms between consecutive frames.

    Args:
        dataset_dir: Folder containing captured frames
        threshold:   Similarity threshold (0–1). Higher = more aggressive removal.
                     0.95 removes frames that are >95% similar to the previous one.
    """
    from PIL import Image

    paths = sorted(dataset_dir.glob('*.png')) + sorted(dataset_dir.glob('*.jpg'))
    if len(paths) < 2:
        print("Not enough frames to deduplicate.")
        return

    print(f"Deduplicating {len(paths)} frames (threshold={threshold})...")
    removed = 0
    prev_hist = None

    for path in paths:
        img = Image.open(path).convert('RGB').resize((64, 64))
        hist = img.histogram()
        hist_norm = [h / sum(hist) for h in hist]

        if prev_hist is not None:
            # Cosine similarity between histograms
            similarity = sum(a * b for a, b in zip(prev_hist, hist_norm))
            similarity /= (
                sum(a * a for a in prev_hist) ** 0.5 *
                sum(b * b for b in hist_norm) ** 0.5 + 1e-8
            )

            if similarity > threshold:
                path.unlink()
                removed += 1
                continue

        prev_hist = hist_norm

    print(f"Removed {removed} duplicate frames. Remaining: {len(paths) - removed}")


def main():
    parser = argparse.ArgumentParser(
        description='Capture native 1440p game frames for SR training dataset'
    )
    subparsers = parser.add_subparsers(dest='command')

    # video subcommand
    video_p = subparsers.add_parser('video', help='Extract frames from a recorded video')
    video_p.add_argument('--source',     required=True,           help='Video file path')
    video_p.add_argument('--output',     default='./dataset',     help='Output folder')
    video_p.add_argument('--every',      type=int,   default=30,  help='1 frame every N frames')
    video_p.add_argument('--min_height', type=int,   default=1440, help='Minimum frame height (px)')

    # screen subcommand
    screen_p = subparsers.add_parser('screen', help='Capture screen live while playing')
    screen_p.add_argument('--output',     default='./dataset',     help='Output folder')
    screen_p.add_argument('--duration',   type=int,   default=300, help='Capture duration in seconds')
    screen_p.add_argument('--fps',        type=float, default=2.0, help='Capture rate (FPS)')
    screen_p.add_argument('--min_height', type=int,   default=1440, help='Minimum frame height (px)')

    # dedup subcommand
    dedup_p = subparsers.add_parser('dedup', help='Remove near-duplicate frames')
    dedup_p.add_argument('--dir',       required=True,             help='Dataset folder to clean')
    dedup_p.add_argument('--threshold', type=float, default=0.95,  help='Similarity threshold (0–1)')

    args = parser.parse_args()

    if args.command == 'video':
        extract_from_video(args.source, Path(args.output), args.every, args.min_height)
    elif args.command == 'screen':
        capture_screen(Path(args.output), args.duration, args.fps, args.min_height)
    elif args.command == 'dedup':
        deduplicate(Path(args.dir), args.threshold)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
