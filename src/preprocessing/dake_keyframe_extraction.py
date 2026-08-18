#!/usr/bin/env python3
"""
DAKE - Dynamic-Aware Keyframe Extraction
Implements Algorithm 1 from "U-CESE: Unified Clip-based Event Search Engine
for AI Challenge HCMC 2025" (Le et al.), Sec. 4.1.

Setting used (per paper Sec. 5.1): rho = 0.02, local window = 3 frames,
plus the minimum-density guarantee (>=1 keyframe every delta = 2*fps frames)
that the paper adopts alongside rho=0.02 for near-perfect AutoShot recall.

Optimized for: MSI Modern 15 B7M -- Ryzen 7 7730U (8C/16T), 16GB RAM, no dGPU.
"""

import argparse
import csv
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Pass 1: sequential decode, JPEG-encode in memory, keep only the byte size
# ---------------------------------------------------------------------------

def compute_jpeg_sizes(video_path, jpeg_quality=90, resize_width=None):
    """
    Decode every frame once and record its JPEG-encoded size. Frame pixel
    data is discarded immediately after encoding -- only an int per frame is
    kept, so a 30k-frame video costs ~240KB of RAM here, not tens of GB.

    resize_width: if set, frames are downscaled before size scoring. This is
    NOT specified in the paper -- it's a speed optimization that assumes
    JPEG-size steepness patterns are roughly preserved under a fixed
    downscale factor. Full-resolution frames are still what gets saved in
    pass 2. Leave as None to match the paper exactly; use e.g. 640 for a
    large speedup on 1080p/4K source video.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    sizes = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if resize_width:
            h, w = frame.shape[:2]
            scale = resize_width / w
            frame = cv2.resize(
                frame, (resize_width, int(h * scale)), interpolation=cv2.INTER_AREA
            )
        ok, enc = cv2.imencode(".jpg", frame, encode_params)
        sizes.append(len(enc) if ok else 0)

    cap.release()
    return sizes, fps


# ---------------------------------------------------------------------------
# Algorithm 1: Dynamic-aware Keyframe Selection
# ---------------------------------------------------------------------------

def dake_select(sizes, rho=0.02, window=3):
    """
    Direct implementation of Algorithm 1, using the confirmed steepness
    equation:

        S(i,j) = d / sqrt((j-i)^2 + d^2),   where d = 100 * |s_j - s_i| / s_max

    This is an arctan-style slope measure (rise = 100*|delta size|/s_max,
    run = frame distance j-i): S in [0,1), approaching 1 for a sharp size
    jump between adjacent frames and shrinking as either the size change
    shrinks or the two frames get further apart in time. That temporal
    term is why nearby jumps outweigh distant ones of the same magnitude.
    """
    n = len(sizes)
    if n < 2:
        return list(range(n))

    sizes = np.asarray(sizes, dtype=np.float64)
    s_max = sizes.max()
    if s_max == 0:
        return []

    steepness = np.zeros(n)
    for i in range(n - 1):
        j_end = min(n, i + window + 1)
        js = np.arange(i + 1, j_end)
        d = 100.0 * np.abs(sizes[i] - sizes[js]) / s_max
        run = (js - i).astype(np.float64)
        steepness[i] = (d / np.sqrt(run ** 2 + d ** 2)).mean()

    order = np.argsort(-steepness)  # descending, matches Algorithm 1 line 13
    k = int(np.floor(rho * n))       # Algorithm 1 line 14
    return sorted(order[:k].tolist())


def enforce_min_density(keyframes, n_frames, fps, delta_mult=2.0):
    """
    Sec. 5.1: "we adopt rho=0.02 and additionally enforce that at least one
    keyframe is included within every delta=2*fps-frame window."

    The paper states the constraint but not the fill rule for gaps. This
    fills gaps at fixed delta-frame intervals (cheap, deterministic). If you
    have the paper's actual fill rule, swap it in here.
    """
    delta = max(1, int(delta_mult * fps))
    kf = sorted(set(keyframes))
    if not kf:
        return list(range(0, n_frames, delta))

    filled = [kf[0]]
    for idx in kf[1:]:
        while idx - filled[-1] > delta:
            filled.append(filled[-1] + delta)
        filled.append(idx)
    if n_frames - 1 - filled[-1] > delta:
        filled.append(filled[-1] + delta)

    return sorted(f for f in set(filled) if 0 <= f < n_frames)


# ---------------------------------------------------------------------------
# Pass 2: extract only the selected frames (single sequential read, stops
# once the last target index has been reached)
# ---------------------------------------------------------------------------

def save_keyframes(video_path, keyframe_indices, out_dir, csv_out_dir, fps):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_out_dir = Path(csv_out_dir)
    csv_out_dir.mkdir(parents=True, exist_ok=True)
    target = sorted(keyframe_indices)

    ti = 0
    idx = 0
    rows = []

    while ti < len(target):
        ret, frame = cap.read()
        if not ret:
            break
        if idx == target[ti]:
            fname = f"{ti + 1:04d}.jpg"
            cv2.imwrite(str(out_dir / fname), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            rows.append((ti + 1, round(idx / fps, 3), round(fps, 3), idx))
            ti += 1
        idx += 1
    cap.release()

    csv_path = csv_out_dir / f"{video_path.stem}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "pts_time", "fps", "frame_idx"])
        w.writerows(rows)

    return len(rows)


# ---------------------------------------------------------------------------
# Per-video worker
# ---------------------------------------------------------------------------

def process_video(video_path, image_out_root, csv_out_root, rho, jpeg_quality, resize_width, enforce_density):
    cv2.setNumThreads(1)  # avoid oversubscription: parallelism is at the process level
    video_path = Path(video_path)
    out_dir = Path(image_out_root) / video_path.stem

    sizes, fps = compute_jpeg_sizes(video_path, jpeg_quality=jpeg_quality, resize_width=resize_width)
    kf = dake_select(sizes, rho=rho)
    if enforce_density:
        kf = enforce_min_density(kf, len(sizes), fps)

    n_saved = save_keyframes(video_path, kf, out_dir, csv_out_root, fps)
    return video_path.name, len(sizes), n_saved


# ---------------------------------------------------------------------------
# CLI / batch driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="DAKE keyframe extraction (U-CESE, AIC HCMC 2025)")
    ap.add_argument("video_dirs", nargs="*", default=None, help="Folders of .mp4 videos. If omitted, scans data/raw/.")
    ap.add_argument("--image-out-dir", default="data/processed/DAKE_output/extracted_keyframe_images", help="Output root for extracted keyframe images")
    ap.add_argument("--csv-out-dir", default="data/processed/DAKE_output/extracted_keyframe_csvs", help="Output root for keyframe CSV files")
    ap.add_argument("--rho", type=float, default=0.02, help="Keyframe ratio (paper: 0.02)")
    ap.add_argument("--jpeg-quality", type=int, default=90)
    ap.add_argument(
        "--resize-width", type=int, default=None,
        help="Downscale width for JPEG-size SCORING only (pass 1 speedup). "
             "Saved keyframes are always full resolution. Not in the paper -- optional.",
    )
    ap.add_argument(
        "--no-density-guarantee", action="store_true",
        help="Disable the delta=2*fps minimum-density fill from Sec. 5.1",
    )
    ap.add_argument(
        "--workers", type=int, default=None,
        help="Videos processed in parallel. Default: min(6, cpu_count()).",
    )
    args = ap.parse_args()

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
        return

    # Ryzen 7 7730U: 8C/16T. cv2 JPEG encode/decode is CPU-bound and each
    # process holds only scalar sizes (pass 1) or one frame at a time
    # (pass 2), so RAM is not the bottleneck -- process-level parallelism
    # across videos scales close to linearly with cores. 6 workers is used
    # as a default rather than 8 to leave the OS/UI responsive on a laptop.
    workers = args.workers or min(6, cpu_count())

    worker_fn = partial(
        process_video,
        image_out_root=args.image_out_dir,
        csv_out_root=args.csv_out_dir,
        rho=args.rho,
        jpeg_quality=args.jpeg_quality,
        resize_width=args.resize_width,
        enforce_density=not args.no_density_guarantee,
    )

    with Pool(processes=workers) as pool:
        for name, n_frames, n_kf in pool.imap_unordered(worker_fn, videos):
            ratio = n_kf / n_frames if n_frames else 0.0
            print(f"{name}: {n_frames} frames -> {n_kf} keyframes ({ratio:.3%})")


if __name__ == "__main__":
    main()
