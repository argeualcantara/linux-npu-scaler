import time
from pathlib import Path


def extract_from_video(source: str, output_dir: Path, every_n: int, min_height: int):
    """
    Extracts frames from a video file using ffmpeg.
    Takes 1 frame every N frames to avoid redundancy.

    The video should be a native 1440p recording with no upscaler active.
    OBS recommended settings: 2560x1440, lossless or high-bitrate H.264/AV1.
    """
    import subprocess

    output_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(output_dir.glob('frame_*.png'))
    start_number = len(existing) + 1
    output_pattern = str(output_dir / 'frame_%06d.png')

    cmd = [
        'ffmpeg', '-i', source,
        '-vf', f'select=not(mod(n\\,{every_n})),scale=-1:{min_height}:flags=lanczos',
        '-vsync', 'vfr',
        '-q:v', '1',
        '-start_number', str(start_number),
        output_pattern,
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
    """
    try:
        import mss
        from PIL import Image
    except ImportError:
        print("Install required packages: pip install mss Pillow")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    interval = 1.0 / fps
    frame_count = len(list(output_dir.glob('screen_*.png')))
    start = time.time()

    print(f"Capturing screen for {duration}s at {fps} FPS...")
    print("Switch to your game now! (native 1440p, no upscaler)")
    time.sleep(3)

    with mss.mss() as sct:
        monitor = sct.monitors[1]
        while time.time() - start < duration:
            t0 = time.time()

            screenshot = sct.grab(monitor)
            img = Image.frombytes('RGB', screenshot.size, screenshot.bgra, 'raw', 'BGRX')

            brightness = sum(img.convert('L').getdata()) / (img.width * img.height)
            if brightness > 20:
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
