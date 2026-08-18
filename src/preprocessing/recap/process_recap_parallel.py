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

def process_video(video_id, base_dir, raw_dir, out_dir):
    out_path = os.path.join(out_dir, f"{video_id}.json")
    if os.path.exists(out_path):
        print(f"Skipping {video_id}, already processed.")
        return True
        
    print(f"\n========== Processing {video_id} ==========")
    
    media_info_path = os.path.join(raw_dir, "media-info-aic25-b1", "media-info", f"{video_id}.json")
    video_info = ""
    if os.path.exists(media_info_path):
        with open(media_info_path, 'r', encoding='utf-8') as f:
            media_info = json.load(f)
        video_info = f"Title: {media_info.get('title', '')}\nDescription: {media_info.get('description', '')}"
    
    shots_path = os.path.join(base_dir, "DAKE_output", "shot_boundaries", f"{video_id}.json")
    if not os.path.exists(shots_path):
        print(f"No shot boundaries for {video_id}. Skipping.")
        return False
        
    with open(shots_path, 'r', encoding='utf-8') as f:
        shots = json.load(f)
        
    subs_path = os.path.join(base_dir, "DAKE_output", "extracted_subtitles", f"{video_id}.json")
    subtitles = []
    if os.path.exists(subs_path):
        with open(subs_path, 'r', encoding='utf-8') as f:
            subtitles = json.load(f)
        
    img_dir = os.path.join(base_dir, "DAKE_output", "extracted_keyframe_images", video_id)
    
    previous_memory = "None"
    output_captions = []
    
    error_occurred = False
    
    for i, shot in enumerate(shots):
        print(f"[{video_id}] Processing shot {i+1}/{len(shots)}")
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
            
        except Exception as e:
            safe_e = str(e).encode('ascii', 'backslashreplace').decode('ascii')
            print(f"[{video_id}] Error processing shot {i}: {safe_e}")
            traceback.print_exc()
            error_occurred = True
            break
            
    if error_occurred:
        print(f"[{video_id}] Failed to process completely.")
        return False
        
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output_captions, f, ensure_ascii=False, indent=2)
    print(f"[{video_id}] Done! Results saved to {out_path}")
    return True

def main():
    base_dir = r"c:\Users\MODERN15\Downloads\AIHCM\data\processed"
    raw_dir = r"c:\Users\MODERN15\Downloads\AIHCM\data\raw\TrainingData"
    out_dir = os.path.join(base_dir, "DAKE_output", "captions")
    os.makedirs(out_dir, exist_ok=True)
    
    missing_nums = [245]
    video_ids = [f"L26_V{str(i).zfill(3)}" for i in missing_nums]
    
    # Process 5 videos concurrently
    max_workers = 5
    print(f"Starting parallel processing with {max_workers} workers...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_video, vid, base_dir, raw_dir, out_dir): vid for vid in video_ids}
        
        for future in concurrent.futures.as_completed(futures):
            vid = futures[future]
            try:
                success = future.result()
            except Exception as exc:
                print(f"{vid} generated an exception: {exc}")

if __name__ == "__main__":
    main()
