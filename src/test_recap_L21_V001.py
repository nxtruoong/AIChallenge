import os
import json
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.recap_api import call_recap_api

def get_subtitle_for_shot(shot_start, shot_end, subtitles):
    shot_subs = []
    for sub in subtitles:
        # Check if subtitle overlaps with the shot
        if sub['start'] <= shot_end and sub['end'] >= shot_start:
            shot_subs.append(sub['text'])
    return " ".join(shot_subs)

def main():
    video_id = "L21_V001"
    base_dir = r"c:\Users\MODERN15\Downloads\AIHCM\data\processed"
    
    # 1. Load Media Info (optional, might not exist for L21 so we handle it)
    media_info_path = os.path.join(base_dir, "kaggle_dataset_staging", "media_info", f"{video_id}.json")
    video_info = ""
    if os.path.exists(media_info_path):
        with open(media_info_path, 'r', encoding='utf-8') as f:
            media_info = json.load(f)
        video_info = f"Title: {media_info.get('title', '')}\nDescription: {media_info.get('description', '')}"
    
    # 2. Load Shot boundaries
    shots_path = os.path.join(base_dir, "DAKE_output", "shot_boundaries", f"{video_id}.json")
    with open(shots_path, 'r', encoding='utf-8') as f:
        shots = json.load(f)
        
    # 3. Load Subtitles
    subs_path = os.path.join(base_dir, "DAKE_output", "extracted_subtitles", f"{video_id}.json")
    subtitles = []
    if os.path.exists(subs_path):
        with open(subs_path, 'r', encoding='utf-8') as f:
            subtitles = json.load(f)
        
    # Image directory
    img_dir = os.path.join(base_dir, "DAKE_output", "extracted_keyframe_images", video_id)
    
    previous_memory = "None"
    
    output_captions = []
    
    for i, shot in enumerate(shots):
        print(f"Processing shot {i+1}/{len(shots)}")
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
            
            try:
                print(f"--- Shot {i} ---")
                print(f"Caption: {caption.encode('utf-8').decode('cp1252', 'ignore')}")
                print(f"Memory: {previous_memory.encode('utf-8').decode('cp1252', 'ignore')}\n")
            except Exception:
                pass
            
        except Exception as e:
            print(f"Error processing shot {i}: {e}")
            
    # Save output
    out_dir = os.path.join(base_dir, "DAKE_output", "captions")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{video_id}_recap_test.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output_captions, f, ensure_ascii=False, indent=2)
    print(f"Done! Results saved to {out_path}")

if __name__ == "__main__":
    main()
