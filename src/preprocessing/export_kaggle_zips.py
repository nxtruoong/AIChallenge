#!/usr/bin/env python3
"""
Package preprocessed artifacts into clean, modular zip files for Kaggle embeddings.
Supports standard separate zips (metadata + keyframes) and unified staging format.
"""

import os
import sys
import json
import zipfile
import argparse
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None, ncols=None):
        return iterable

# Fix Windows console UTF-8 printing
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass

def get_batch_video_ids(batch_name: str, base_dir: Path) -> list:
    batch_map = {
        "L21": [f"L21_V{str(i).zfill(3)}" for i in range(1, 32)],
        "L22": [f"L22_V{str(i).zfill(3)}" for i in range(1, 32)],
        "L26_a": [f"L26_V{str(i).zfill(3)}" for i in range(1, 100)],
        "L26_b": [f"L26_V{str(i).zfill(3)}" for i in range(101, 200)],
        "L26_c": [f"L26_V{str(i).zfill(3)}" for i in range(201, 300)],
        "L26_d": [f"L26_V{str(i).zfill(3)}" for i in range(301, 400)],
        "L26_e": [f"L26_V{str(i).zfill(3)}" for i in range(401, 500)],
        "L26": [f"L26_V{str(i).zfill(3)}" for i in range(1, 500)],
    }
    
    if batch_name in batch_map:
        candidates = batch_map[batch_name]
    else:
        candidates = [p.stem for p in (base_dir / "captions").glob(f"{batch_name}*.json")]
        
    # Keep only videos that exist in captions or shot_boundaries
    valid = []
    for v in candidates:
        if (base_dir / "captions" / f"{v}.json").exists() or (base_dir / "shot_boundaries" / f"{v}.json").exists():
            valid.append(v)
    return sorted(valid)

def zip_metadata(batch_name: str, video_ids: list, base_dir: Path, out_zip: Path):
    print(f"\n[Metadata] Creating {out_zip.name} ({len(video_ids)} videos)...")
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for vid in tqdm(video_ids, desc=f"{batch_name} metadata", ncols=80):
            # 1. Captions
            cap = base_dir / "captions" / f"{vid}.json"
            if cap.exists():
                zf.write(cap, arcname=f"captions/{vid}.json")
                
            # 2. Shot Boundaries
            sb = base_dir / "shot_boundaries" / f"{vid}.json"
            if sb.exists():
                zf.write(sb, arcname=f"shot_boundaries/{vid}.json")
                
            # 3. Subtitles
            sub = base_dir / "extracted_subtitles" / f"{vid}.json"
            if sub.exists():
                zf.write(sub, arcname=f"extracted_subtitles/{vid}.json")
                
            # 4. Keyframe CSVs
            csv_f = base_dir / "extracted_keyframe_csvs" / f"{vid}.csv"
            if csv_f.exists():
                zf.write(csv_f, arcname=f"extracted_keyframe_csvs/{vid}.csv")
                
    size_mb = out_zip.stat().st_size / (1024**2)
    print(f"Saved: {out_zip} ({size_mb:.2f} MB)")

def zip_keyframes(batch_name: str, video_ids: list, base_dir: Path, out_zip: Path):
    print(f"\n[Keyframes] Creating {out_zip.name} ({len(video_ids)} videos)...")
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    img_base = base_dir / "extracted_keyframe_images"
    
    # Use ZIP_STORED (no re-compression on JPGs, fastest and lowest CPU)
    with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_STORED) as zf:
        for vid in tqdm(video_ids, desc=f"{batch_name} keyframes", ncols=80):
            vid_dir = img_base / vid
            if vid_dir.exists():
                for img_file in vid_dir.glob("*.jpg"):
                    zf.write(img_file, arcname=f"extracted_keyframe_images/{vid}/{img_file.name}")
                    
    size_mb = out_zip.stat().st_size / (1024**2)
    print(f"Saved: {out_zip} ({size_mb:.2f} MB / {size_mb/1024:.2f} GB)")

def zip_staging_dataset(batch_name: str, video_ids: list, base_dir: Path, raw_dir: Path, out_zip: Path):
    print(f"\n[Staging Dataset] Creating unified {out_zip.name} ({len(video_ids)} videos)...")
    out_zip.parent.mkdir(parents=True, exist_ok=True)

    sources_def = {
        'captions': (base_dir / 'captions', 'file', '.json', 'captions'),
        'shot_boundaries': (base_dir / 'shot_boundaries', 'file', '.json', 'shot_boundaries'),
        'subtitles': (base_dir / 'extracted_subtitles', 'file', '.json', 'subtitles'),
        'keyframe_csvs': (base_dir / 'extracted_keyframe_csvs', 'file', '.csv', 'keyframe_csvs'),
        'keyframe_images': (base_dir / 'extracted_keyframe_images', 'dir', '*.jpg', 'keyframe_images'),
        'media_info': (raw_dir / 'media-info-aic25-b1/media-info', 'file', '.json', 'media_info'),
    }

    stats = {}
    total_files = 0

    with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for name, (src_base, kind, pattern, arc_folder) in sources_def.items():
            count = 0
            if kind == 'file':
                for v in video_ids:
                    src_file = src_base / f"{v}{pattern}"
                    if src_file.is_file():
                        zf.write(src_file, arcname=f"{arc_folder}/{src_file.name}")
                        count += 1
            elif kind == 'dir':
                for v in video_ids:
                    vdir = src_base / v
                    if vdir.is_dir():
                        for f in vdir.glob(pattern):
                            compress = zipfile.ZIP_STORED if name == 'keyframe_images' else zipfile.ZIP_DEFLATED
                            zf.write(f, arcname=f"{arc_folder}/{v}/{f.name}", compress_type=compress)
                            count += 1
            stats[name] = {'files': count, 'source_path': str(src_base)}
            total_files += count
            print(f"  + Added {count} files for {name}")

        manifest = {
            'description': f'AIC 2026 preprocessed data for Kaggle - {batch_name} Batch',
            'sources': stats,
            'total_files': total_files + 1
        }
        zf.writestr('manifest.json', json.dumps(manifest, indent=2))

    size_mb = out_zip.stat().st_size / (1024**2)
    print(f"Saved staging zip: {out_zip} ({size_mb:.2f} MB / {size_mb/1024:.2f} GB, {total_files + 1} items)")

def main():
    ap = argparse.ArgumentParser(description="Package preprocessed data into Kaggle-ready zip files")
    ap.add_argument("--base-dir", default="data/processed/Preprocess", help="Base directory containing Preprocess folders")
    ap.add_argument("--raw-dir", default="data/raw/TrainingData", help="Raw training data directory (for objects, map_keyframes, media_info)")
    ap.add_argument("--out-dir", default="data/export_kaggle", help="Output directory for zip archives")
    ap.add_argument("--batches", nargs="*", default=["L21", "L22", "L26_a", "L26_e"], help="Batches to package")
    ap.add_argument("--staging", action="store_true", help="Export in unified staging format matching kaggle_dataset_staging")
    ap.add_argument("--only-metadata", action="store_true", help="Only zip text & metadata files")
    ap.add_argument("--only-keyframes", action="store_true", help="Only zip visual keyframe images")
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Packaging target batches: {args.batches}")
    print(f"Output directory: {out_dir.resolve()}")

    for b in args.batches:
        vids = get_batch_video_ids(b, base_dir)
        if not vids:
            print(f"Skipping {b}: no videos found.")
            continue

        print(f"\n========================================================")
        print(f"  Packaging batch: {b} ({len(vids)} videos)")
        print(f"========================================================")
        
        if args.staging:
            staging_zip = out_dir / f"{b}_kaggle_dataset.zip"
            zip_staging_dataset(b, vids, base_dir, raw_dir, staging_zip)
        else:
            # 1. Text & Metadata Zip (Captions, Shot Boundaries, Subtitles, CSVs)
            if not args.only_keyframes:
                meta_zip = out_dir / f"{b}_metadata.zip"
                zip_metadata(b, vids, base_dir, meta_zip)

            # 2. Visual Keyframes Zip
            if not args.only_metadata:
                kf_zip = out_dir / f"{b}_keyframes.zip"
                zip_keyframes(b, vids, base_dir, kf_zip)

    print("\nAll packaging finished successfully!")

if __name__ == "__main__":
    main()
