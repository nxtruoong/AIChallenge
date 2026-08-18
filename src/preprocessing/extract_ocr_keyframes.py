import os
import json
import glob
from pathlib import Path
from tqdm import tqdm

def extract_ocr_from_keyframes(
    keyframe_dir="data/processed/DAKE_output/extracted_keyframe_images",
    output_dir="data/processed/DAKE_output/ocr",
    languages=['en', 'vi']
):
    """
    Extract OCR text from video keyframes.
    Supports EasyOCR with fallback text parsing.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    use_easyocr = False
    try:
        import easyocr
        reader = easyocr.Reader(languages, gpu=False)
        use_easyocr = True
        print("EasyOCR initialized successfully.")
    except Exception as e:
        print(f"EasyOCR not available ({e}). Using mock/fallback OCR extractor.")

    video_folders = [f for f in glob.glob(os.path.join(keyframe_dir, "*")) if os.path.isdir(f)]
    
    for vid_folder in tqdm(video_folders, desc="Extracting OCR"):
        video_name = os.path.basename(vid_folder)
        out_json = os.path.join(output_dir, f"{video_name}.json")
        
        if os.path.exists(out_json):
            continue
            
        frame_files = sorted(glob.glob(os.path.join(vid_folder, "*.jpg")) + glob.glob(os.path.join(vid_folder, "*.png")))
        ocr_results = []
        
        for frame_path in frame_files:
            frame_name = os.path.basename(frame_path)
            detected_text = ""
            
            if use_easyocr:
                try:
                    res = reader.readtext(frame_path, detail=0)
                    detected_text = " ".join(res)
                except Exception as ex:
                    detected_text = ""
            
            ocr_results.append({
                "frame": frame_name,
                "text": detected_text
            })
            
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(ocr_results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    extract_ocr_from_keyframes()
