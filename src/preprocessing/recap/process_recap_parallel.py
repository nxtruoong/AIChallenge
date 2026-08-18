import os
import json
import sys
import traceback
import concurrent.futures

try:
    from .recap_api import call_recap_api
except ImportError:
    from recap_api import call_recap_api

def get_subtitle_for_shot(shot_start, shot_end, subtitles):
    shot_subs = []
    for sub in subtitles:
        if sub['start'] <= shot_end and sub['end'] >= shot_start:
            shot_subs.append(sub['text'])
    return " ".join(shot_subs)

def process_video(video_id, base_dir, raw_dir, out_dir, overwrite=False):
    out_path = os.path.join(out_dir, f"{video_id}.json")
    
    shots_path = os.path.join(base_dir, "DAKE_output", "shot_boundaries", f"{video_id}.json")
    if not os.path.exists(shots_path):
        print(f"No shot boundaries for {video_id}. Skipping.", flush=True)
        return False
        
    with open(shots_path, 'r', encoding='utf-8') as f:
        shots = json.load(f)
        
    if not overwrite and os.path.exists(out_path):
        try:
            with open(out_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            if len(existing) == len(shots):
                print(f"Skipping {video_id}, all {len(shots)} shots already processed.", flush=True)
                return True
        except Exception:
            pass

    print(f"\n========== Processing {video_id} ({len(shots)} shots) ==========", flush=True)
    
    # Auto-find media info JSON in raw_dir subfolders
    media_info_path = os.path.join(raw_dir, "TrainingData", "media-info-aic25-b1", "media-info", f"{video_id}.json")
    if not os.path.exists(media_info_path):
        media_info_path = os.path.join(raw_dir, "media-info-aic25-b1", "media-info", f"{video_id}.json")
    if not os.path.exists(media_info_path):
        candidates = list(Path(raw_dir).glob(f"**/{video_id}.json"))
        if candidates:
            media_info_path = str(candidates[0])
            
    video_info = ""
    if os.path.exists(media_info_path):
        with open(media_info_path, 'r', encoding='utf-8') as f:
            media_info = json.load(f)
        video_info = f"Title: {media_info.get('title', '')}\nDescription: {media_info.get('description', '')}"
        
    subs_path = os.path.join(base_dir, "DAKE_output", "extracted_subtitles", f"{video_id}.json")
    subtitles = []
    if os.path.exists(subs_path):
        with open(subs_path, 'r', encoding='utf-8') as f:
            subtitles = json.load(f)
        
    img_dir = os.path.join(base_dir, "DAKE_output", "extracted_keyframe_images", video_id)
    
    # Check for partial progress to resume
    previous_memory = "None"
    output_captions = []
    start_shot_idx = 0
    
    if not overwrite and os.path.exists(out_path):
        try:
            with open(out_path, 'r', encoding='utf-8') as f:
                output_captions = json.load(f)
            if output_captions:
                start_shot_idx = len(output_captions)
                previous_memory = output_captions[-1].get("memory", "None")
                print(f"[{video_id}] Resuming from shot {start_shot_idx + 1}/{len(shots)}...", flush=True)
        except Exception:
            output_captions = []
            start_shot_idx = 0
            
    error_occurred = False
    
    for i in range(start_shot_idx, len(shots)):
        shot = shots[i]
        print(f"[{video_id}] Processing shot {i+1}/{len(shots)}", flush=True)
        shot_start = shot['start_time']
        shot_end = shot['end_time']
        
        shot_subtitle = get_subtitle_for_shot(shot_start, shot_end, subtitles)
        
        keyframes_paths = [os.path.join(img_dir, kf['image']) for kf in shot['keyframes']]
        if len(keyframes_paths) > 0:
            mid_idx = len(keyframes_paths) // 2
            keyframes_paths = [keyframes_paths[mid_idx]]
        
        try:
            result = call_recap_api(video_info, keyframes_paths, shot_subtitle, previous_memory, template_name="prompt_template.txt")
            
            caption = result.get('caption', '')
            previous_memory = result.get('memory', '')
            
            output_captions.append({
                "shot_id": shot["shot_id"],
                "start_time": shot_start,
                "end_time": shot_end,
                "caption": caption,
                "memory": previous_memory
            })
            
            # Incremental auto-save after every single shot
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(output_captions, f, ensure_ascii=False, indent=2)
            
        except Exception as e:
            safe_e = str(e).encode('ascii', 'backslashreplace').decode('ascii')
            print(f"[{video_id}] Error processing shot {i}: {safe_e}", flush=True)
            traceback.print_exc()
            error_occurred = True
            break
            
    if error_occurred:
        print(f"[{video_id}] Stopped partially at shot {len(output_captions)}/{len(shots)}.", flush=True)
        return False
        
    print(f"[{video_id}] Done! All {len(output_captions)} shots saved to {out_path}", flush=True)
    return True

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Parallel ReCap captioning with recurrent memory")
    parser.add_argument("--base-dir", type=str, default=os.getenv("PROCESSED_DATA_DIR", "data/processed"), help="Processed data directory containing DAKE_output")
    parser.add_argument("--raw-dir", type=str, default=os.getenv("RAW_DATA_DIR", "data/raw"), help="Raw data directory containing media-info")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for captions")
    parser.add_argument("--video-ids", nargs="*", default=None, help="List of video IDs (e.g. L26_V245)")
    parser.add_argument("--batch-range", nargs=2, type=int, default=None, help="Start and end index (e.g. 101 200 for L26_V101..L26_V200)")
    parser.add_argument("--batch-prefix", type=str, default="L26_V", help="Video prefix (e.g. L26_V)")
    parser.add_argument("--workers", type=int, default=5, help="Number of concurrent worker threads")
    parser.add_argument("--model", type=str, default=None, help="LLM/LVLM model slug (default: xiaomi/mimo-v2.5)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing caption files")
    args = parser.parse_args()

    base_dir = args.base_dir
    raw_dir = args.raw_dir
    out_dir = args.out_dir or os.path.join(base_dir, "DAKE_output", "captions")
    os.makedirs(out_dir, exist_ok=True)
    
    if args.video_ids:
        video_ids = args.video_ids
    elif args.batch_range:
        video_ids = [f"{args.batch_prefix}{str(i).zfill(3)}" for i in range(args.batch_range[0], args.batch_range[1] + 1)]
    else:
        # Auto-discover from shot_boundaries directory
        shots_dir = os.path.join(base_dir, "DAKE_output", "shot_boundaries")
        if os.path.exists(shots_dir):
            prefix = args.batch_prefix if args.batch_prefix else ""
            video_ids = sorted([os.path.splitext(f)[0] for f in os.listdir(shots_dir) if f.startswith(prefix) and f.endswith(".json")])
        else:
            video_ids = ["L26_V245"]
            
    print(f"Total videos to process: {len(video_ids)}")
    print(f"Workers: {args.workers} | Model: {args.model or os.getenv('RECAP_MODEL', 'xiaomi/mimo-v2.5')}")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_video, vid, base_dir, raw_dir, out_dir, args.overwrite): vid for vid in video_ids}
        
        for future in concurrent.futures.as_completed(futures):
            vid = futures[future]
            try:
                success = future.result()
            except Exception as exc:
                print(f"{vid} generated an exception: {exc}")

if __name__ == "__main__":
    main()
