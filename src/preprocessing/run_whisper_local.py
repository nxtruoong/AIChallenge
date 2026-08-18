#!/usr/bin/env python3
"""
Local ASR: faster-whisper large-v3-turbo on CPU with int8 quantization.
Optimized for Ryzen 7 7730U (8C/16T, 16GB RAM).

Key optimizations over the naive version:
  1. BatchedInferencePipeline — batches VAD-segmented chunks for ~2x decode speedup
  2. beam_size=1 — ~3-5x faster than default beam_size=5, minimal quality loss
  3. cpu_threads + num_workers — multi-worker model allows ThreadPoolExecutor
     to process 2 files concurrently, each using 4 CPU threads (4×2 = 8 cores)
  4. condition_on_previous_text=False — skips re-encoding context, faster
  5. Concurrent I/O — ThreadPoolExecutor overlaps file loading with transcription
  6. Per-file ETA reporting — shows speed (audio seconds / wall seconds)

Resume-safe: skips videos whose output JSON already exists.

Usage:
    python run_whisper_local.py                    # full run, all defaults
    python run_whisper_local.py --test             # single-video smoke test
    python run_whisper_local.py --workers 1        # single-threaded (safer)
    python run_whisper_local.py --beam-size 3      # trade speed for quality
    python run_whisper_local.py --batch-size 16    # larger decode batches
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from faster_whisper import WhisperModel, BatchedInferencePipeline
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Transcription worker
# ---------------------------------------------------------------------------

def transcribe_one(
    pipeline: BatchedInferencePipeline,
    audio_path: Path,
    output_dir: Path,
    beam_size: int,
    batch_size: int,
) -> dict:
    """Transcribe a single audio file. Returns a result dict."""
    output_path = output_dir / f"{audio_path.stem}.json"

    if output_path.exists():
        return {"file": audio_path.name, "status": "skipped"}

    t0 = time.perf_counter()

    segments, info = pipeline.transcribe(
        str(audio_path),
        language="vi",
        initial_prompt="Món ngon mỗi ngày, Ajinomoto, Aji-ngon, hạt nêm, bột ngọt, mã QR, hạnh nhân, xôi, bông cải, tầm bóp, chiên, nướng, lẩu, xào, kho",
        beam_size=beam_size,
        batch_size=batch_size,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=300,   # merge pauses < 300ms
            speech_pad_ms=200,             # pad speech segments by 200ms
        ),
        condition_on_previous_text=False,  # faster, avoids hallucination loops
        word_timestamps=False,             # we don't need word-level timing
        without_timestamps=False,          # keep segment timestamps
    )

    subtitles = []
    for seg in segments:
        text = seg.text.strip()
        if text:  # skip empty segments
            subtitles.append({
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": text,
            })

    # Atomic write: temp file then rename (crash-safe)
    tmp_path = output_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(subtitles, f, ensure_ascii=False, indent=2)
    tmp_path.rename(output_path)

    elapsed = time.perf_counter() - t0
    audio_duration = info.duration if info.duration else 0
    rtf = elapsed / audio_duration if audio_duration > 0 else float("inf")

    return {
        "file": audio_path.name,
        "status": "done",
        "segments": len(subtitles),
        "audio_sec": round(audio_duration, 1),
        "wall_sec": round(elapsed, 1),
        "rtf": round(rtf, 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Fast local Whisper ASR for AIC 2026 (optimized for Ryzen 7 7730U)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model settings
    ap.add_argument("--model", default="large-v3-turbo",
                    help="Whisper model name")
    ap.add_argument("--device", default="cpu",
                    help="Device: cpu or cuda")
    ap.add_argument("--compute-type", default="int8",
                    help="Quantization: int8, float16, float32")

    # Performance tuning
    ap.add_argument("--cpu-threads", type=int, default=4,
                    help="CTranslate2 threads per worker (default: 4, "
                         "total = cpu-threads × workers)")
    ap.add_argument("--workers", type=int, default=2,
                    help="Concurrent transcription workers. 2 workers × 4 "
                         "threads = 8 cores. Set to 1 for single-threaded.")
    ap.add_argument("--beam-size", type=int, default=1,
                    help="Beam search width (1=greedy ~3-5x faster, 5=default)")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="BatchedInferencePipeline batch size for VAD segments")

    # I/O
    ap.add_argument("--audio-dir", default="data/processed/DAKE_output/extracted_audios",
                    help="Directory containing .mp3 audio files")
    ap.add_argument("--output-dir", default="data/processed/DAKE_output/extracted_subtitles",
                    help="Directory for output subtitle JSONs")
    ap.add_argument("--test", action="store_true",
                    help="Process only 1 file (smoke test)")

    args = ap.parse_args()

    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Discover files ---
    audio_paths = sorted(audio_dir.glob("*.mp3"))
    if not audio_paths:
        print(f"No .mp3 files found in {audio_dir}")
        sys.exit(1)

    print(f"Found {len(audio_paths)} audio files in {audio_dir}")

    if args.test:
        audio_paths = audio_paths[:1]
        print("TEST MODE: processing only 1 file")

    # Filter to only remaining files
    todo = [p for p in audio_paths if not (output_dir / f"{p.stem}.json").exists()]
    already_done = len(audio_paths) - len(todo)
    print(f"Already completed: {already_done}, remaining: {len(todo)}")

    if not todo:
        print("All files already transcribed. Nothing to do.")
        return

    # --- Load model ---
    total_threads = args.cpu_threads * args.workers
    physical_cores = os.cpu_count() or 8
    print(f"\nConfig: model={args.model}, beam_size={args.beam_size}, "
          f"batch_size={args.batch_size}")
    print(f"Threading: {args.workers} workers × {args.cpu_threads} threads "
          f"= {total_threads} total (machine has {physical_cores} logical cores)")

    if total_threads > physical_cores:
        print(f"  WARNING: total threads ({total_threads}) exceeds logical "
              f"cores ({physical_cores}). Consider reducing --workers or "
              f"--cpu-threads to avoid contention.")

    print(f"\nLoading model: {args.model} (device={args.device}, "
          f"compute={args.compute_type}) ...")
    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
        cpu_threads=args.cpu_threads,
        num_workers=args.workers,
    )
    pipeline = BatchedInferencePipeline(model=model)
    print("Model loaded.\n")

    # --- Transcribe ---
    transcribed = 0
    total_audio_sec = 0
    start_wall = time.perf_counter()

    if args.workers <= 1:
        # Single-threaded: simpler, guaranteed thread-safe
        for audio_path in tqdm(todo, desc="Transcribing", unit="file"):
            result = transcribe_one(
                pipeline, audio_path, output_dir,
                args.beam_size, args.batch_size,
            )
            if result["status"] == "done":
                transcribed += 1
                total_audio_sec += result["audio_sec"]
                tqdm.write(
                    f"  {result['file']}: {result['segments']} segments, "
                    f"{result['audio_sec']}s audio in {result['wall_sec']}s "
                    f"(RTF={result['rtf']})"
                )
    else:
        # Multi-threaded: 2+ files concurrently via CTranslate2 num_workers
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    transcribe_one, pipeline, p, output_dir,
                    args.beam_size, args.batch_size,
                ): p
                for p in todo
            }
            pbar = tqdm(total=len(todo), desc="Transcribing", unit="file")
            for future in as_completed(futures):
                result = future.result()
                pbar.update(1)
                if result["status"] == "done":
                    transcribed += 1
                    total_audio_sec += result["audio_sec"]
                    pbar.write(
                        f"  {result['file']}: {result['segments']} segs, "
                        f"{result['audio_sec']}s audio in {result['wall_sec']}s "
                        f"(RTF={result['rtf']})"
                    )
            pbar.close()

    total_wall = time.perf_counter() - start_wall
    overall_rtf = total_wall / total_audio_sec if total_audio_sec > 0 else 0

    print(f"\n{'='*60}")
    print(f"DONE — {transcribed} files transcribed in {total_wall / 60:.1f} minutes")
    print(f"Total audio: {total_audio_sec / 60:.1f} min, "
          f"wall time: {total_wall / 60:.1f} min")
    print(f"Overall RTF: {overall_rtf:.2f}x "
          f"({'faster' if overall_rtf < 1 else 'slower'} than real-time)")
    print(f"Total subtitle files on disk: "
          f"{len(list(output_dir.glob('*.json')))}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
