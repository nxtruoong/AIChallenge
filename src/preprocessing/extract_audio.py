#!/usr/bin/env python3
"""
Audio extraction script matching DAKE script pattern.
Extracts audio from video files (.mp4) in input folders using ffmpeg,
saving output .mp3 files into a specified output directory.
"""

import argparse
import subprocess
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
import sys
import time

# Locate FFmpeg binary
DEFAULT_FFMPEG_PATH = Path(r"C:\Users\MODERN15\AppData\Local\Programs\Softdeluxe\Free Download Manager\ffmpeg.exe")

def get_ffmpeg_path():
    if DEFAULT_FFMPEG_PATH.exists():
        return str(DEFAULT_FFMPEG_PATH)
    return "ffmpeg"  # fallback to PATH

def process_video(video_path, out_dir, ffmpeg_bin):
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_audio_path = out_dir / f"{video_path.stem}.mp3"

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "libmp3lame",
        "-ab", "128k",
        "-loglevel", "error",
        str(out_audio_path)
    ]

    start_time = time.time()
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    elapsed = time.time() - start_time

    if res.returncode != 0:
        return video_path.name, False, f"FFmpeg error: {res.stderr}"

    out_size_mb = out_audio_path.stat().st_size / (1024 * 1024)
    return video_path.name, True, f"{out_audio_path.name} ({out_size_mb:.2f} MB in {elapsed:.1f}s)"

def main():
    ap = argparse.ArgumentParser(description="Extract audio from videos using ffmpeg (matching DAKE format)")
    ap.add_argument("video_dirs", nargs="*", default=None, help="Folders containing .mp4 videos. If omitted, scans data/raw/.")
    ap.add_argument("--audio-out-dir", default="data/processed/DAKE_output/extracted_audios", help="Output directory for extracted audio files")
    ap.add_argument("--workers", type=int, default=None, help="Parallel workers. Default: min(6, cpu_count())")
    args = ap.parse_args()

    ffmpeg_bin = get_ffmpeg_path()
    print(f"Using FFmpeg binary: {ffmpeg_bin}")

    vdirs = args.video_dirs
    if not vdirs:
        raw_root = Path("data/raw")
        if raw_root.exists():
            vdirs = [str(p) for p in raw_root.iterdir() if p.is_dir() and any(p.glob("*.mp4"))]
            if not vdirs and any(raw_root.glob("*.mp4")):
                vdirs = [str(raw_root)]
        else:
            vdirs = ["data/raw"]

    videos = []
    for vdir in vdirs:
        found = sorted(Path(vdir).glob("*.mp4"))
        print(f"Found {len(found)} videos in {vdir}")
        videos.extend(found)

    if not videos:
        print("No .mp4 files found in any of the specified directories.")
        sys.exit(1)

    workers = args.workers or min(6, cpu_count())
    print(f"Processing {len(videos)} videos using {workers} parallel workers...")
    print(f"Target output directory: {args.audio_out_dir}")

    worker_fn = partial(
        process_video,
        out_dir=args.audio_out_dir,
        ffmpeg_bin=ffmpeg_bin
    )

    success_count = 0
    fail_count = 0
    total = len(videos)

    start_total = time.time()
    with Pool(processes=workers) as pool:
        for idx, (name, ok, msg) in enumerate(pool.imap_unordered(worker_fn, videos), 1):
            if ok:
                success_count += 1
                print(f"[{idx}/{total}] Extracted: {name} -> {msg}")
            else:
                fail_count += 1
                print(f"[{idx}/{total}] ERROR processing {name}: {msg}")

    total_time = time.time() - start_total
    print(f"\nFinished in {total_time:.1f}s. Total: {total}, Success: {success_count}, Failed: {fail_count}")

if __name__ == "__main__":
    main()
