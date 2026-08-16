#!/usr/bin/env python3
"""
Shot boundary detection for DAKE-extracted keyframes.

Since DAKE already selected keyframes at high-steepness (dynamic) points,
the steepness *between* consecutive DAKE keyframes is almost universally
high (median ~0.92) — making raw steepness thresholding useless for shot
segmentation on this data.

Instead, this uses a **temporal gap** strategy:
  1. Compute the time gap between consecutive keyframes (from pts_time).
  2. A shot boundary is placed wherever the gap exceeds a threshold.
  3. The threshold is adaptive per video: median_gap × multiplier.

This works because DAKE naturally clusters keyframes more densely within
dynamic scenes/shots and produces larger temporal gaps across shot
boundaries (scene cuts, fade-outs, etc.).

Additionally, shots are capped at a maximum duration to ensure the
downstream captioning model gets manageable context windows.

Output: per-video JSON in DAKE_output/shot_boundaries/
Format: [
  {
    "shot_id": 0,
    "keyframes": [
      {"n": 1, "pts_time": 0.28, "fps": 25.0, "frame_idx": 7, "image": "0001.jpg"},
      ...
    ],
    "start_time": 0.28,
    "end_time": 5.12,
    "n_keyframes": 5
  },
  ...
]

Usage:
    python detect_shots_dake.py                       # full run
    python detect_shots_dake.py --test                # single-video smoke test
    python detect_shots_dake.py --gap-multiplier 3.0  # fewer, longer shots
    python detect_shots_dake.py --max-shot-sec 30     # cap shot duration
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm


def load_keyframe_csv(csv_path: Path) -> list[dict]:
    """Load the DAKE keyframe CSV (n, pts_time, fps, frame_idx)."""
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "n": int(row["n"]),
                "pts_time": float(row["pts_time"]),
                "fps": float(row["fps"]),
                "frame_idx": int(row["frame_idx"]),
                "image": f"{int(row['n']):04d}.jpg",
            })
    return rows


def detect_shot_boundaries(
    keyframes: list[dict],
    gap_multiplier: float,
    min_gap_sec: float,
    max_shot_sec: float,
) -> list[int]:
    """Find shot boundary indices using adaptive temporal gap detection.

    A boundary is placed where the time gap between consecutive keyframes
    exceeds max(median_gap * gap_multiplier, min_gap_sec). Additionally,
    any shot exceeding max_shot_sec is force-split.

    Returns a list of keyframe indices where each new shot starts.
    """
    if len(keyframes) < 2:
        return [0]

    times = np.array([kf["pts_time"] for kf in keyframes])
    gaps = np.diff(times)

    # Adaptive threshold: use median gap × multiplier, but no less than min_gap_sec
    median_gap = float(np.median(gaps)) if len(gaps) > 0 else 1.0
    threshold = max(median_gap * gap_multiplier, min_gap_sec)

    boundaries = [0]
    current_shot_start_time = times[0]

    for i in range(len(gaps)):
        time_since_shot_start = times[i + 1] - current_shot_start_time
        gap_exceeds = gaps[i] >= threshold
        shot_too_long = time_since_shot_start >= max_shot_sec

        if gap_exceeds or shot_too_long:
            boundaries.append(i + 1)
            current_shot_start_time = times[i + 1]

    return boundaries


def group_into_shots(
    keyframes: list[dict],
    boundaries: list[int],
) -> list[dict]:
    """Group keyframes into shots based on boundary indices."""
    shots = []
    for shot_idx, start_kf in enumerate(boundaries):
        end_kf = boundaries[shot_idx + 1] if shot_idx + 1 < len(boundaries) else len(keyframes)
        shot_keyframes = keyframes[start_kf:end_kf]
        if not shot_keyframes:
            continue

        shots.append({
            "shot_id": shot_idx,
            "keyframes": shot_keyframes,
            "start_time": shot_keyframes[0]["pts_time"],
            "end_time": shot_keyframes[-1]["pts_time"],
            "n_keyframes": len(shot_keyframes),
        })

    return shots


def process_video(
    video_name: str,
    csv_dir: Path,
    output_dir: Path,
    gap_multiplier: float,
    min_gap_sec: float,
    max_shot_sec: float,
) -> dict:
    """Process one video: detect boundaries and group into shots."""
    output_path = output_dir / f"{video_name}.json"
    if output_path.exists():
        return {"video": video_name, "status": "skipped"}

    csv_path = csv_dir / f"{video_name}.csv"
    if not csv_path.exists():
        return {"video": video_name, "status": "error", "msg": f"no CSV: {csv_path}"}

    keyframes = load_keyframe_csv(csv_path)
    if not keyframes:
        return {"video": video_name, "status": "error", "msg": "empty CSV"}

    boundaries = detect_shot_boundaries(
        keyframes, gap_multiplier, min_gap_sec, max_shot_sec
    )
    shots = group_into_shots(keyframes, boundaries)

    # Atomic write
    tmp_path = output_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(shots, f, ensure_ascii=False, indent=2)
    tmp_path.rename(output_path)

    return {
        "video": video_name,
        "status": "done",
        "keyframes": len(keyframes),
        "shots": len(shots),
        "avg_kf_per_shot": round(len(keyframes) / len(shots), 1) if shots else 0,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Shot boundary detection via temporal gap analysis on DAKE keyframes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--csv-dir", default="DAKE_output/extracted_keyframe_csvs",
                    help="Directory of DAKE keyframe CSVs")
    ap.add_argument("--output-dir", default="DAKE_output/shot_boundaries",
                    help="Output directory for shot boundary JSONs")
    ap.add_argument("--gap-multiplier", type=float, default=2.5,
                    help="Boundary if gap >= median_gap * this multiplier. "
                         "Lower = more shots, higher = fewer shots.")
    ap.add_argument("--min-gap-sec", type=float, default=2.0,
                    help="Minimum gap (seconds) to ever split on, regardless of median. "
                         "Prevents over-splitting on very dense keyframe sequences.")
    ap.add_argument("--max-shot-sec", type=float, default=30.0,
                    help="Force-split shots longer than this (seconds). "
                         "Keeps context windows manageable for captioning.")
    ap.add_argument("--test", action="store_true",
                    help="Process only 1 video (smoke test)")

    args = ap.parse_args()

    csv_dir = Path(args.csv_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_names = sorted(p.stem for p in csv_dir.glob("*.csv"))
    if not video_names:
        print(f"No CSV files found in {csv_dir}")
        sys.exit(1)

    print(f"Found {len(video_names)} videos")

    if args.test:
        video_names = video_names[:1]
        print("TEST MODE: processing only 1 video")

    todo = [v for v in video_names if not (output_dir / f"{v}.json").exists()]
    print(f"Already completed: {len(video_names) - len(todo)}, remaining: {len(todo)}")

    if not todo:
        print("All videos already processed. Nothing to do.")
        return

    print(f"Gap multiplier: {args.gap_multiplier}, Min gap: {args.min_gap_sec}s, "
          f"Max shot: {args.max_shot_sec}s\n")

    done = 0
    errors = 0
    total_shots = 0
    total_keyframes = 0

    for video_name in tqdm(todo, desc="Detecting shots", unit="video"):
        result = process_video(
            video_name, csv_dir, output_dir,
            args.gap_multiplier, args.min_gap_sec, args.max_shot_sec,
        )
        if result["status"] == "done":
            done += 1
            total_shots += result["shots"]
            total_keyframes += result["keyframes"]
            tqdm.write(
                f"  {result['video']}: {result['keyframes']} keyframes -> "
                f"{result['shots']} shots ({result['avg_kf_per_shot']} kf/shot)"
            )
        elif result["status"] == "error":
            errors += 1
            tqdm.write(f"  ERROR {result['video']}: {result['msg']}")

    print(f"\nDone: {done}, Errors: {errors}")
    print(f"Total: {total_keyframes} keyframes -> {total_shots} shots")
    if total_shots > 0:
        print(f"Average: {total_keyframes / total_shots:.1f} keyframes per shot")


if __name__ == "__main__":
    main()
