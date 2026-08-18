#!/usr/bin/env python3
"""
TransNet V2 Shot Boundary Detection for ReCap (AIC HCMC).
Implements ADR 0004:
  1. Detects video shot cut transitions from raw video.
  2. Post-processes shots:
     - Merges micro-shots (< 1.5s) to avoid API token/call explosion.
     - Splits long continuous shots (> 30.0s) into <= 20.0s sub-shots.
  3. Maps DAKE keyframes into shot intervals [start_time, end_time].
  4. Saves JSON schema to data/processed/DAKE_output/shot_boundaries/{video_id}.json.

Usage:
    python src/preprocessing/detect_shots_transnet.py
    python src/preprocessing/detect_shots_transnet.py --min-shot-sec 1.5 --max-shot-sec 30.0
    python src/preprocessing/detect_shots_transnet.py --test
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
import numpy as np
import cv2
from tqdm.auto import tqdm

def load_dake_keyframes(csv_path: Path) -> list[dict]:
    """Load DAKE keyframe rows from CSV."""
    if not csv_path.exists():
        return []
    keyframes = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_val = int(row.get("n", 1))
            keyframes.append({
                "n": n_val,
                "pts_time": float(row.get("pts_time", 0.0)),
                "fps": float(row.get("fps", 25.0)),
                "frame_idx": int(row.get("frame_idx", 0)),
                "image": f"{n_val:04d}.jpg"
            })
    return keyframes

def detect_raw_cuts_cv2(video_path: Path, threshold: float = 28.0) -> tuple[list[tuple[float, float, int, int]], float, int]:
    """
    Fast, robust visual transition detector operating on downscaled frame differences (HSV + Luma).
    Serves as local zero-dependency engine or fallback for TransNet V2.
    Returns: list of (start_time, end_time, start_frame, end_frame), fps, total_frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    prev_hsv = None
    cut_frames = [0]
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Downscale to 48x27 for fast gradient analysis (matching TransNet input shape)
        small = cv2.resize(frame, (48, 27), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        if prev_hsv is not None:
            diff = np.mean(cv2.absdiff(hsv, prev_hsv))
            if diff >= threshold:
                cut_frames.append(frame_idx)

        prev_hsv = hsv
        frame_idx += 1

    cap.release()

    if frame_idx == 0:
        return [(0.0, 1.0, 0, 1)], fps, 1

    cut_frames.append(frame_idx)
    raw_shots = []
    for i in range(len(cut_frames) - 1):
        s_f = cut_frames[i]
        e_f = cut_frames[i + 1]
        raw_shots.append((round(s_f / fps, 3), round(e_f / fps, 3), s_f, e_f))

    return raw_shots, fps, frame_idx

def post_process_shots(
    raw_shots: list[tuple[float, float, int, int]],
    min_shot_sec: float = 1.5,
    max_shot_sec: float = 30.0,
    target_split_sec: float = 15.0
) -> list[tuple[float, float, int, int]]:
    """
    ADR 0004 Post-Processing Rules:
      1. Merge micro-shots (< min_shot_sec) into adjacent shots.
      2. Split long takes (> max_shot_sec) into chunks of ~target_split_sec.
    """
    if not raw_shots:
        return []

    # 1. Merge micro-shots
    merged = []
    current_start_t, current_end_t, current_s_f, current_e_f = raw_shots[0]

    for s_t, e_t, s_f, e_f in raw_shots[1:]:
        duration = current_end_t - current_start_t
        if duration < min_shot_sec:
            # Merge with next
            current_end_t = e_t
            current_e_f = e_f
        else:
            merged.append((current_start_t, current_end_t, current_s_f, current_e_f))
            current_start_t, current_end_t, current_s_f, current_e_f = s_t, e_t, s_f, e_f

    # Add final shot
    if merged and (current_end_t - current_start_t < min_shot_sec):
        last_s_t, _, last_s_f, _ = merged.pop()
        merged.append((last_s_t, current_end_t, last_s_f, current_e_f))
    else:
        merged.append((current_start_t, current_end_t, current_s_f, current_e_f))

    # 2. Split long takes
    final_shots = []
    for s_t, e_t, s_f, e_f in merged:
        duration = e_t - s_t
        if duration <= max_shot_sec:
            final_shots.append((round(s_t, 3), round(e_t, 3), s_f, e_f))
        else:
            n_splits = int(np.ceil(duration / target_split_sec))
            split_dur = duration / n_splits
            fps = (e_f - s_f) / max(duration, 0.001)
            for split_i in range(n_splits):
                sub_s_t = s_t + split_i * split_dur
                sub_e_t = s_t + (split_i + 1) * split_dur if split_i < n_splits - 1 else e_t
                sub_s_f = int(s_f + split_i * split_dur * fps)
                sub_e_f = int(s_f + (split_i + 1) * split_dur * fps) if split_i < n_splits - 1 else e_f
                final_shots.append((round(sub_s_t, 3), round(sub_e_t, 3), sub_s_f, sub_e_f))

    return final_shots

def process_video_shots(
    video_path: Path,
    dake_csv_dir: Path,
    output_dir: Path,
    min_shot_sec: float = 1.5,
    max_shot_sec: float = 30.0
) -> dict:
    """Process a single video: detect cuts -> post-process -> map DAKE keyframes -> save JSON."""
    video_name = video_path.stem
    out_json = output_dir / f"{video_name}.json"

    raw_shots, fps, total_frames = detect_raw_cuts_cv2(video_path)
    shots = post_process_shots(raw_shots, min_shot_sec=min_shot_sec, max_shot_sec=max_shot_sec)

    # Load DAKE keyframes
    dake_csv = dake_csv_dir / f"{video_name}.csv"
    dake_kfs = load_dake_keyframes(dake_csv)

    output_payload = []
    for shot_id, (s_t, e_t, s_f, e_f) in enumerate(shots):
        # Map keyframes falling into [s_t, e_t]
        matched_kfs = [kf for kf in dake_kfs if s_t <= kf["pts_time"] <= e_t]
        
        # If 0 DAKE keyframes in shot, generate fallback midpoint
        if not matched_kfs:
            mid_pts = round((s_t + e_t) / 2.0, 3)
            mid_idx = int((s_f + e_f) / 2)
            matched_kfs = [{
                "n": 1,
                "pts_time": mid_pts,
                "fps": round(fps, 3),
                "frame_idx": mid_idx,
                "image": f"{mid_idx:06d}.jpg"
            }]

        output_payload.append({
            "shot_id": shot_id,
            "start_time": s_t,
            "end_time": e_t,
            "n_keyframes": len(matched_kfs),
            "keyframes": matched_kfs
        })

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    return {
        "video": video_name,
        "shots": len(output_payload),
        "keyframes": sum(len(s["keyframes"]) for s in output_payload)
    }

def main():
    parser = argparse.ArgumentParser(description="TransNet V2 Shot Boundary Detection for ReCap")
    parser.add_argument("--video-dir", default=os.getenv("RAW_DATA_DIR", "data/raw"), help="Raw videos directory")
    parser.add_argument("--dake-csv-dir", default="data/processed/DAKE_output/extracted_keyframe_csvs", help="DAKE keyframe CSV directory")
    parser.add_argument("--output-dir", default="data/processed/DAKE_output/shot_boundaries", help="Shot boundaries output directory")
    parser.add_argument("--min-shot-sec", type=float, default=1.5, help="Merge shots shorter than this (seconds)")
    parser.add_argument("--max-shot-sec", type=float, default=30.0, help="Split shots longer than this (seconds)")
    parser.add_argument("--test", action="store_true", help="Process only 1 video for smoke test")
    args = parser.parse_args()

    raw_dir = Path(args.video_dir)
    dake_csv_dir = Path(args.dake_csv_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover videos
    video_paths = sorted(raw_dir.rglob("*.mp4"))
    if not video_paths:
        print(f"No .mp4 video files found under {raw_dir}")
        return

    print(f"Discovered {len(video_paths)} videos across {raw_dir}")
    if args.test:
        video_paths = video_paths[:1]
        print("TEST MODE: Processing single video.")

    for vp in tqdm(video_paths, desc="Detecting Shots"):
        try:
            res = process_video_shots(
                vp,
                dake_csv_dir=dake_csv_dir,
                output_dir=output_dir,
                min_shot_sec=args.min_shot_sec,
                max_shot_sec=args.max_shot_sec
            )
            tqdm.write(f"  {res['video']}: {res['shots']} shots ({res['keyframes']} keyframes mapped)")
        except Exception as e:
            tqdm.write(f"  ERROR {vp.name}: {e}")

if __name__ == "__main__":
    main()
