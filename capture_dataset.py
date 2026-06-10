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
from pathlib import Path

from sr.data.capture import extract_from_video, capture_screen, deduplicate


def main():
    parser = argparse.ArgumentParser(
        description='Capture native 1440p game frames for SR training dataset'
    )
    subparsers = parser.add_subparsers(dest='command')

    video_p = subparsers.add_parser('video', help='Extract frames from a recorded video')
    video_p.add_argument('--source',     required=True,            help='Video file path')
    video_p.add_argument('--output',     default='./dataset',      help='Output folder')
    video_p.add_argument('--every',      type=int,   default=30,   help='1 frame every N frames')
    video_p.add_argument('--min_height', type=int,   default=1440, help='Minimum frame height (px)')

    screen_p = subparsers.add_parser('screen', help='Capture screen live while playing')
    screen_p.add_argument('--output',     default='./dataset',     help='Output folder')
    screen_p.add_argument('--duration',   type=int,   default=300, help='Capture duration in seconds')
    screen_p.add_argument('--fps',        type=float, default=2.0, help='Capture rate (FPS)')
    screen_p.add_argument('--min_height', type=int,   default=1440, help='Minimum frame height (px)')

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
