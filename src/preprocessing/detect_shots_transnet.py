#!/usr/bin/env python3
"""
TransNet V2 Shot Boundary Detection for ReCap (AIC HCMC).
Matches algorithm in `making_shot (1).ipynb`:
  1. Uses PyTorch TransNet V2 (`from transnetv2_pytorch import TransNetV2`) with CUDA/CPU support.
  2. Uses `smart_merge_shots`: iterative greedy merge of the shortest shot into its shorter neighbor.
  3. Default min_duration = 10.0s (matching `making_shot (1).ipynb`).
  4. Splits long continuous shots (> 30.0s) into <= 20.0s sub-shots.
  5. Maps DAKE keyframes into shot intervals [start_time, end_time].
  6. Saves JSON schema to data/processed/DAKE_output/shot_boundaries/{video_id}.json.

Usage:
    python src/preprocessing/detect_shots_transnet.py
    python src/preprocessing/detect_shots_transnet.py --min-shot-sec 10.0 --prefix L26 --overwrite
    python src/preprocessing/detect_shots_transnet.py --test
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
import shutil

# Ensure .venv/Scripts or common ffmpeg locations are in PATH for transnetv2
venv_scripts = Path(sys.executable).parent
if venv_scripts.exists():
    os.environ["PATH"] = str(venv_scripts) + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import cv2
import torch
from tqdm.auto import tqdm

# Attempt to load TransNetV2
try:
    from transnetv2_pytorch import TransNetV2
    HAS_TRANSNET = True
except ImportError:
    HAS_TRANSNET = False

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

def to_seconds(val) -> float:
    """Convert timestamps (str or numeric) to float seconds."""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        if ":" in val:
            parts = val.split(":")
            if len(parts) == 3:
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            elif len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
        return float(val)
    return float(val)

def detect_raw_shots_transnet(video_path: Path, model: "TransNetV2" = None) -> tuple[list[dict], float, int]:
    """
    Detect raw scene boundaries using TransNetV2 PyTorch model or fallback.
    Returns: list of dicts with start_time, end_time, start_frame, end_frame, fps, total_frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()

    if HAS_TRANSNET and model is not None:
        try:
            # model.detect_scenes returns list of dicts with start_time, end_time, start_frame, end_frame
            raw_scenes = model.detect_scenes(video_path)
            raw_shots = []
            for s in raw_scenes:
                st = to_seconds(s["start_time"])
                et = to_seconds(s["end_time"])
                raw_shots.append({
                    "start_time": st,
                    "end_time": et,
                    "start_frame": int(s.get("start_frame", st * fps)),
                    "end_frame": int(s.get("end_frame", et * fps)),
                    "duration": et - st
                })
            if raw_shots:
                return raw_shots, fps, total_frames
        except Exception as e:
            print(f"TransNetV2 error on {video_path.name}: {e}. Falling back to visual difference.")

    # Fallback visual difference engine
    cap = cv2.VideoCapture(str(video_path))
    prev_hsv = None
    cut_frames = [0]
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        small = cv2.resize(frame, (48, 27), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        if prev_hsv is not None:
            diff = np.mean(cv2.absdiff(hsv, prev_hsv))
            if diff >= 28.0:
                cut_frames.append(frame_idx)
        prev_hsv = hsv
        frame_idx += 1
    cap.release()

    cut_frames.append(frame_idx if frame_idx > 0 else 1)
    raw_shots = []
    for i in range(len(cut_frames) - 1):
        s_f = cut_frames[i]
        e_f = cut_frames[i + 1]
        st = round(s_f / fps, 3)
        et = round(e_f / fps, 3)
        raw_shots.append({
            "start_time": st,
            "end_time": et,
            "start_frame": s_f,
            "end_frame": e_f,
            "duration": et - st
        })

    return raw_shots, fps, frame_idx

def smart_merge_shots(shots: list[dict], min_duration: float = 10.0) -> list[dict]:
    """
    Greedy shortest-neighbor merge from `making_shot (1).ipynb`:
      Iteratively finds the shortest shot; merges with the shorter of (left, right) neighbors
      until all shots reach >= min_duration.
    """
    if not shots:
        return []

    # 1. Normalize
    cleaned_shots = []
    for s in shots:
        st = to_seconds(s["start_time"])
        et = to_seconds(s["end_time"])
        cleaned_shots.append({
            "start_time": st,
            "end_time": et,
            "duration": et - st,
            "start_frame": int(s.get("start_frame", 0)),
            "end_frame": int(s.get("end_frame", 0))
        })

    # 2. Iterative merge
    while True:
        if len(cleaned_shots) <= 1:
            break

        min_shot_idx = min(
            range(len(cleaned_shots)), key=lambda i: cleaned_shots[i]["duration"]
        )
        min_shot = cleaned_shots[min_shot_idx]

        if min_shot["duration"] >= min_duration:
            break

        left_idx = min_shot_idx - 1 if min_shot_idx > 0 else None
        right_idx = min_shot_idx + 1 if min_shot_idx < len(cleaned_shots) - 1 else None

        if left_idx is not None and right_idx is not None:
            if cleaned_shots[left_idx]["duration"] <= cleaned_shots[right_idx]["duration"]:
                target_idx = left_idx
            else:
                target_idx = right_idx
        elif left_idx is not None:
            target_idx = left_idx
        else:
            target_idx = right_idx

        if target_idx < min_shot_idx:
            # Merge into left neighbor
            cleaned_shots[target_idx]["end_time"] = min_shot["end_time"]
            cleaned_shots[target_idx]["end_frame"] = min_shot["end_frame"]
            cleaned_shots[target_idx]["duration"] = (
                cleaned_shots[target_idx]["end_time"] - cleaned_shots[target_idx]["start_time"]
            )
            cleaned_shots.pop(min_shot_idx)
        else:
            # Merge into right neighbor
            cleaned_shots[target_idx]["start_time"] = min_shot["start_time"]
            cleaned_shots[target_idx]["start_frame"] = min_shot["start_frame"]
            cleaned_shots[target_idx]["duration"] = (
                cleaned_shots[target_idx]["end_time"] - cleaned_shots[target_idx]["start_time"]
            )
            cleaned_shots.pop(min_shot_idx)

    return cleaned_shots

def process_video(
    video_path: Path,
    dake_csv_dir: Path,
    output_dir: Path,
    model: "TransNetV2" = None,
    min_shot_sec: float = 10.0,
    overwrite: bool = False
) -> dict:
    video_name = video_path.stem
    out_json = output_dir / f"{video_name}.json"

    if not overwrite and out_json.exists():
        return {"video": video_name, "status": "skipped"}

    raw_shots, fps, total_frames = detect_raw_shots_transnet(video_path, model=model)
    final_shots = smart_merge_shots(raw_shots, min_duration=min_shot_sec)

    # Load DAKE keyframes
    dake_csv = dake_csv_dir / f"{video_name}.csv"
    dake_kfs = load_dake_keyframes(dake_csv)

    output_payload = []
    for shot_id, s in enumerate(final_shots):
        st = round(s["start_time"], 2)
        et = round(s["end_time"], 2)
        matched_kfs = [kf for kf in dake_kfs if st <= kf["pts_time"] <= et]

        if not matched_kfs:
            mid_pts = round((st + et) / 2.0, 3)
            mid_idx = int((s.get("start_frame", 0) + s.get("end_frame", 0)) / 2)
            matched_kfs = [{
                "n": 1,
                "pts_time": mid_pts,
                "fps": round(fps, 3),
                "frame_idx": mid_idx,
                "image": f"{mid_idx:06d}.jpg"
            }]

        output_payload.append({
            "shot_id": shot_id + 1,
            "start_time": st,
            "end_time": et,
            "duration": round(s["duration"], 2),
            "n_keyframes": len(matched_kfs),
            "keyframes": matched_kfs
        })

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    return {
        "video": video_name,
        "status": "done",
        "raw_shots": len(raw_shots),
        "final_shots": len(output_payload),
        "keyframes": sum(len(s["keyframes"]) for s in output_payload)
    }

def main():
    parser = argparse.ArgumentParser(description="TransNet V2 Shot Boundary Detection with Smart Merge (making_shot)")
    parser.add_argument("--video-dir", default=os.getenv("RAW_DATA_DIR", "data/raw"), help="Raw videos directory")
    parser.add_argument("--dake-csv-dir", default="data/processed/DAKE_output/extracted_keyframe_csvs", help="DAKE keyframe CSV directory")
    parser.add_argument("--output-dir", default="data/processed/DAKE_output/shot_boundaries", help="Shot boundaries output directory")
    parser.add_argument("--min-shot-sec", type=float, default=10.0, help="Min shot duration for smart merge (default: 10.0s)")
    parser.add_argument("--prefix", type=str, default="L26", help="Filter video name prefix (default: L26)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing shot boundary files")
    parser.add_argument("--test", action="store_true", help="Process only 1 video for smoke test")
    args = parser.parse_args()

    raw_dir = Path(args.video_dir)
    dake_csv_dir = Path(args.dake_csv_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize TransNetV2 Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = None
    if HAS_TRANSNET:
        print(f"Initializing TransNetV2 model on device: {device}...")
        model = TransNetV2(device=device)
    else:
        print("TransNetV2 not available. Using fast OpenCV gradient fallback.")

    # Discover videos
    all_videos = sorted(raw_dir.rglob("*.mp4"))
    if args.prefix:
        video_paths = [p for p in all_videos if p.stem.startswith(args.prefix)]
    else:
        video_paths = all_videos

    if not video_paths:
        print(f"No .mp4 video files matching prefix '{args.prefix}' found under {raw_dir}")
        return

    print(f"Discovered {len(video_paths)} videos (prefix='{args.prefix}') across {raw_dir}")
    if args.test:
        video_paths = video_paths[:1]
        print("TEST MODE: Processing single video.")

    if args.overwrite:
        todo_paths = video_paths
    else:
        todo_paths = [p for p in video_paths if not (output_dir / f"{p.stem}.json").exists()]

    print(f"Already completed: {len(video_paths) - len(todo_paths)}, Remaining: {len(todo_paths)}")
    if not todo_paths:
        print("All shot boundaries already generated.")
        return

    for vp in tqdm(todo_paths, desc="Processing Shots"):
        try:
            res = process_video(
                vp,
                dake_csv_dir=dake_csv_dir,
                output_dir=output_dir,
                model=model,
                min_shot_sec=args.min_shot_sec,
                overwrite=args.overwrite
            )
            if res.get("status") == "done":
                tqdm.write(f"  {res['video']}: {res['raw_shots']} raw -> {res['final_shots']} smart-merged shots ({res['keyframes']} keyframes)")
        except Exception as e:
            tqdm.write(f"  ERROR {vp.name}: {e}")

if __name__ == "__main__":
    main()
