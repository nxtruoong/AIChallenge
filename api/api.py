import json
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

# Ensure video directories exist to prevent StaticFiles from crashing
os.makedirs("data/raw/Videos_L26_b", exist_ok=True)
os.makedirs("data/raw/Videos_L26_c", exist_ok=True)
os.makedirs("data/raw/Videos_L26_d", exist_ok=True)

# Mount the video directories
app.mount("/videos/L26_b", StaticFiles(directory="data/raw/Videos_L26_b"), name="videos_b")
app.mount("/videos/L26_c", StaticFiles(directory="data/raw/Videos_L26_c"), name="videos_c")
app.mount("/videos/L26_d", StaticFiles(directory="data/raw/Videos_L26_d"), name="videos_d")

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

@app.post("/api/search")
def search(req: SearchRequest):
    search_engine = get_engine()
    results = search_engine.search_clips([req.query], return_separate=True)
    
    def format_clips(clips):
        formatted_clips = []
        for clip in clips:
            video_name = clip["video_name"]
            folder = None
            if os.path.exists(f"data/raw/Videos_L26_b/{video_name}.mp4"):
                folder = "L26_b"
            elif os.path.exists(f"data/raw/Videos_L26_c/{video_name}.mp4"):
                folder = "L26_c"
            elif os.path.exists(f"data/raw/Videos_L26_d/{video_name}.mp4"):
                folder = "L26_d"
                
            video_url = f"/videos/{folder}/{video_name}.mp4" if folder else None
            
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
