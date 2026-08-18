import json
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/search')))
from test_search import SearchEngine
from fastapi.responses import HTMLResponse
import os

app = FastAPI()

# Mount the static files for the UI
app.mount("/static", StaticFiles(directory="api/static"), name="static")

# Ensure raw directory exists and mount it
os.makedirs("data/raw", exist_ok=True)
app.mount("/videos", StaticFiles(directory="data/raw"), name="videos")

# Initialize search engine lazily or eagerly
engine = None

def get_engine():
    global engine
    if engine is None:
        engine = SearchEngine()
    return engine

class SearchRequest(BaseModel):
    query: str

@app.get("/")
def read_root():
    with open("api/static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)

def find_video_rel_path(video_name: str) -> str:
    raw_path = Path("data/raw")
    # Check direct match
    if (raw_path / f"{video_name}.mp4").exists():
        return f"/videos/{video_name}.mp4"
    # Search all subdirectories
    for mp4_file in raw_path.rglob(f"{video_name}.mp4"):
        rel = mp4_file.relative_to(raw_path).as_posix()
        return f"/videos/{rel}"
    return None

@app.post("/api/search")
def search(req: SearchRequest):
    search_engine = get_engine()
    results = search_engine.search_clips([req.query], return_separate=True)
    
    def format_clips(clips):
        formatted_clips = []
        for clip in clips:
            video_name = clip["video_name"]
            video_url = find_video_rel_path(video_name)
            
            formatted_clips.append({
                "video_name": video_name,
                "start_time": clip["start_time"],
                "end_time": clip["end_time"],
                "video_url": video_url,
                "description": clip.get("description", "")
            })
        return formatted_clips
        
    return {
        "results": format_clips(results["fused"]),
        "visual": format_clips(results["visual"]),
        "text": format_clips(results["text"])
    }

@app.get("/api/evaluate_results")
def get_eval_results():
    try:
        with open("data/eval/evaluation_results.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"error": "Evaluation results not found. Run evaluation first."}
