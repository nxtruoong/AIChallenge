import json
import argparse
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../search')))
from test_search import SearchEngine

# Fix Windows console unicode print error
sys.stdout.reconfigure(encoding='utf-8')

def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries for testing")
    parser.add_argument("--input", type=str, default="query.json", help="Path to input queries json")
    parser.add_argument("--output", type=str, default="data/eval/evaluation_results.json", help="Path to output results json")
    args = parser.parse_args()

    engine = SearchEngine()
    
    with open(args.input, "r", encoding="utf-8") as f:
        queries_data = json.load(f)
        
    if args.limit:
        queries_data = queries_data[:args.limit]
        
    total_queries = len(queries_data)
    hits_at_1 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    vqa_12frame_hits = 0
    
    fps = 25.0
    frame_tolerance_sec = 12.0 / fps  # 0.48 seconds
    
    detailed_results = []
    
    print(f"\nEvaluating {total_queries} queries (12-frame tolerance = {frame_tolerance_sec:.2f}s)...")
    
    for i, item in enumerate(tqdm(queries_data)):
        q_text = item["query"]
        true_video = item["answer"]["video_name"]
        if "pts_time" in item["answer"]:
            true_pts = float(item["answer"]["pts_time"])
        else:
            true_pts = (float(item["answer"]["start_time"]) + float(item["answer"]["end_time"])) / 2.0
        
        clips = engine.search_clips([q_text])
        
        # Check matches
        match_rank = -1
        vqa_exact_match = False
        for rank, clip in enumerate(clips):
            vid = clip["video_name"]
            start = clip["start_time"]
            end = clip["end_time"]
            
            # General shot match tolerance
            if vid == true_video and (start - 5.0 <= true_pts <= end + 5.0):
                if match_rank == -1:
                    match_rank = rank + 1
            
            # VQA 12-frame strict precision check (~0.48s window around frame points)
            for frame_info in clip.get("frames", []):
                fr_pts = frame_info.get("pts_time", (start + end) / 2.0)
                if vid == true_video and abs(fr_pts - true_pts) <= frame_tolerance_sec:
                    vqa_exact_match = True
                    break
                
        if match_rank != -1:
            if match_rank <= 1:
                hits_at_1 += 1
            if match_rank <= 5:
                hits_at_5 += 1
            if match_rank <= 10:
                hits_at_10 += 1

        if vqa_exact_match:
            vqa_12frame_hits += 1
                
        detailed_results.append({
            "query": q_text,
            "true_video": true_video,
            "true_pts": true_pts,
            "match_rank": match_rank,
            "vqa_exact_match": vqa_exact_match,
            "top_10_clips": clips[:10]
        })
                
    print("\n--- Evaluation Results ---")
    print(f"Total Queries: {total_queries}")
    print(f"Recall@1           : {hits_at_1 / total_queries * 100:.2f}%")
    print(f"Recall@5           : {hits_at_5 / total_queries * 100:.2f}%")
    print(f"Recall@10          : {hits_at_10 / total_queries * 100:.2f}%")
    print(f"VQA 12-Frame Prec. : {vqa_12frame_hits / total_queries * 100:.2f}%")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": {
                "total_queries": total_queries,
                "recall_at_1": hits_at_1 / total_queries * 100,
                "recall_at_5": hits_at_5 / total_queries * 100,
                "recall_at_10": hits_at_10 / total_queries * 100,
                "vqa_12frame_precision": vqa_12frame_hits / total_queries * 100
            },
            "results": detailed_results
        }, f, ensure_ascii=False, indent=2)
    print(f"Saved detailed results to {args.output}")


if __name__ == "__main__":
    evaluate()
