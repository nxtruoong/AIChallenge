#!/usr/bin/env python3
"""
Prepare and package all preprocessed data for Kaggle Dataset upload.

Assembles a staging directory with:
  - DAKE keyframe images (per-video folders)
  - DAKE keyframe CSVs
  - Subtitle JSONs
  - Caption JSONs
  - Shot boundary JSONs
  - Organizer's object detection data
  - Organizer's map-keyframes CSVs
  - Organizer's media-info JSONs

Optionally creates a zip file for upload via kaggle CLI or web UI.

Usage:
    python prepare_kaggle_dataset.py                       # just report stats
    python prepare_kaggle_dataset.py --zip                 # create zip
    python prepare_kaggle_dataset.py --staging-dir kaggle_upload  # custom dir
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from tqdm.auto import tqdm


def count_files(directory: Path, pattern: str = "*") -> int:
    """Count files matching a pattern in a directory."""
    if not directory.is_dir():
        return 0
    return sum(1 for _ in directory.rglob(pattern) if _.is_file())


def dir_size_mb(directory: Path) -> float:
    """Calculate total size of a directory in MB."""
    if not directory.is_dir():
        return 0.0
    total = sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def copy_tree(src: Path, dst: Path, desc: str = ""):
    """Copy directory tree with progress bar."""
    if not src.is_dir():
        print(f"  WARNING: source not found: {src}")
        return 0

    files = [f for f in src.rglob("*") if f.is_file()]
    
    import re
    filtered_files = []
    for f in files:
        match = re.search(r'L26_V(\d{3})', str(f))
        if match:
            vid = int(match.group(1))
            if 100 <= vid <= 399:
                filtered_files.append(f)
    files = filtered_files

    dst.mkdir(parents=True, exist_ok=True)

    copied = 0
    for f in tqdm(files, desc=f"  Copying {desc}", unit="file", leave=False):
        rel = f.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)
        copied += 1

    return copied


def main():
    ap = argparse.ArgumentParser(
        description="Prepare Kaggle dataset from preprocessed data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--staging-dir", default="kaggle_dataset_staging",
                    help="Staging directory to assemble files into")
    ap.add_argument("--zip", action="store_true",
                    help="Create a zip file from the staging directory")
    ap.add_argument("--zip-name", default="aic2026_preprocessed",
                    help="Name for the output zip file (without .zip)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Only report statistics, don't copy anything")

    args = ap.parse_args()

    # Define source directories
    sources = {
        "keyframe_images": Path("data/processed/DAKE_output/extracted_keyframe_images"),
        "keyframe_csvs": Path("data/processed/DAKE_output/extracted_keyframe_csvs"),
        "subtitles": Path("data/processed/DAKE_output/extracted_subtitles"),
        "captions": Path("data/processed/DAKE_output/captions"),
        "shot_boundaries": Path("data/processed/DAKE_output/shot_boundaries"),
        "objects": Path("data/raw/TrainingData/objects-aic25-b1/objects"),
        "map_keyframes": Path("data/raw/TrainingData/map-keyframes-aic25-b1/map-keyframes"),
        "media_info": Path("data/raw/TrainingData/media-info-aic25-b1/media-info"),
    }

    # Report statistics
    print("=" * 60)
    print("DATA INVENTORY")
    print("=" * 60)

    total_files = 0
    total_size_mb = 0

    for name, path in sources.items():
        n = count_files(path)
        size = dir_size_mb(path)
        status = "OK" if n > 0 else "MISSING"
        print(f"  {name:20s}: {n:6d} files, {size:8.1f} MB  [{status}]")
        total_files += n
        total_size_mb += size

    print(f"  {'TOTAL':20s}: {total_files:6d} files, {total_size_mb:8.1f} MB")
    print("=" * 60)

    # Check for critical missing data
    missing = [name for name, path in sources.items()
               if not path.is_dir() or count_files(path) == 0]
    if missing:
        print(f"\nWARNING: The following sources are missing or empty:")
        for m in missing:
            print(f"  - {m} ({sources[m]})")
        print("\nMissing data will be skipped. The pipeline can still work")
        print("but some features (e.g., captioning, object search) may be limited.\n")

    if args.dry_run:
        print("DRY RUN: No files copied.")
        return

    # Assemble staging directory
    staging = Path(args.staging_dir)
    if staging.exists():
        print(f"\nStaging directory {staging} already exists.")
        print("Delete it first or choose a different name.")
        sys.exit(1)

    print(f"\nAssembling staging directory: {staging}/")

    copied_total = 0
    for name, src_path in sources.items():
        if not src_path.is_dir() or count_files(src_path) == 0:
            print(f"  Skipping {name} (missing)")
            continue

        dst_path = staging / name
        n = copy_tree(src_path, dst_path, desc=name)
        copied_total += n
        print(f"  {name}: {n} files copied")

    # Write a manifest file
    manifest = {
        "description": "AIC 2026 preprocessed data for Kaggle",
        "sources": {
            name: {
                "files": count_files(staging / name),
                "source_path": str(path),
            }
            for name, path in sources.items()
            if (staging / name).is_dir()
        },
        "total_files": copied_total,
    }

    manifest_path = staging / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\nManifest written to {manifest_path}")

    staged_size = dir_size_mb(staging)
    print(f"Staging directory: {copied_total} files, {staged_size:.1f} MB")

    # Optionally create zip
    if args.zip:
        zip_path = Path(f"{args.zip_name}")
        print(f"\nCreating zip archive: {zip_path}.zip ...")
        shutil.make_archive(str(zip_path), "zip", str(staging))
        zip_size = Path(f"{zip_path}.zip").stat().st_size / (1024 * 1024)
        print(f"Created {zip_path}.zip ({zip_size:.1f} MB)")

    print(f"\n{'=' * 60}")
    print("NEXT STEPS:")
    print("1. Upload to Kaggle as a Dataset:")
    print(f"   - Web UI: kaggle.com/datasets > New Dataset > upload {args.staging_dir}/")
    print(f"   - CLI:    kaggle datasets create -p {args.staging_dir}/")
    print("2. In your Kaggle notebook, the data will be at:")
    print("   /kaggle/input/your-dataset-name/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
