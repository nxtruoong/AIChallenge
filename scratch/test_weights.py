import sys
import os
import json
import torch

sys.path.append(os.path.abspath("src/search"))
from test_search import SearchEngine, reciprocal_rank_fusion

sys.stdout.reconfigure(encoding='utf-8')

with open("data/eval/test_queries.json", "r", encoding="utf-8") as f:
    queries_data = json.load(f)[:77]

engine = SearchEngine()
engine.reranker = None

configs = [
    {"name": "BGE=3.0, SigLIP=1.0, BM25=2.0", "weights": [3.0, 1.0, 2.0]},
    {"name": "BGE=3.0, SigLIP=1.0, BM25=3.0", "weights": [3.0, 1.0, 3.0]},
    {"name": "BGE=3.0, SigLIP=1.5, BM25=2.5", "weights": [3.0, 1.5, 2.5]},
    {"name": "BGE=3.0, SigLIP=1.5, BM25=3.5", "weights": [3.0, 1.5, 3.5]},
    {"name": "BGE=3.0, SigLIP=2.0, BM25=3.0", "weights": [3.0, 2.0, 3.0]},
]

fps = 25.0
frame_tolerance_sec = 12.0 / fps

# Pre-compute all query results once for lightning-fast weight tuning
query_cached_results = []
for item in queries_data:
    q_text = item["query"]
    true_video = item["answer"]["video_name"]
    true_pts = float(item["answer"]["pts_time"]) if "pts_time" in item["answer"] else (float(item["answer"]["start_time"]) + float(item["answer"]["end_time"])) / 2.0

    q_txt_emb = engine.text_model.encode([q_text], normalize_embeddings=True)[0].tolist()
    txt_res = engine.milvus.search(
        collection_name="video_shots", data=[q_txt_emb], anns_field="text_vector", limit=100,
        output_fields=["video_name", "frame_idx", "pts_time"]
    )[0]
    res_text = [{"video_name": h["entity"]["video_name"], "frame_idx": h["entity"]["frame_idx"], "pts_time": h["entity"]["pts_time"], "score": h["distance"], "query_id": 0} for h in txt_res]
    
    with torch.no_grad():
        inputs = engine.siglip_processor(text=[q_text], return_tensors="pt", padding=True, truncation=True, max_length=64).to(engine.device)
        q_vis = engine.siglip_model.get_text_features(**inputs)
        q_vis_tensor = getattr(q_vis, 'pooler_output', q_vis[0])
        q_vis_tensor = q_vis_tensor / q_vis_tensor.norm(dim=-1, keepdim=True)
        q_vis_emb = q_vis_tensor.cpu().numpy().tolist()[0]
    vis_res = engine.milvus.search(
        collection_name="video_shots", data=[q_vis_emb], anns_field="visual_vector", limit=100,
        output_fields=["video_name", "frame_idx", "pts_time"]
    )[0]
    res_vis = [{"video_name": h["entity"]["video_name"], "frame_idx": h["entity"]["frame_idx"], "pts_time": h["entity"]["pts_time"], "score": h["distance"], "query_id": 0} for h in vis_res]
    
    es_resp = engine.es.search(
        index="video_shots",
        query={"multi_match": {"query": q_text, "fields": ["text^2", "ocr_text^3"], "fuzziness": "AUTO"}},
        size=100
    )
    res_bm25 = [{"video_name": h["_source"].get("video_name", h["_source"]["id"].rsplit("_", 1)[0]), "frame_idx": h["_source"].get("frame_idx", 0), "pts_time": h["_source"].get("pts_time", 0.0), "score": h["_score"]} for h in es_resp['hits']['hits']]

    query_cached_results.append({
        "q_text": q_text, "true_video": true_video, "true_pts": true_pts,
        "res_text": res_text, "res_vis": res_vis, "res_bm25": res_bm25
    })

print(f"Pre-cached search results for {len(query_cached_results)} queries. Testing weights instantly:")

for cfg in configs:
    hits_1, hits_5, hits_10, vqa_hits = 0, 0, 0, 0
    for item in query_cached_results:
        fused = reciprocal_rank_fusion([item["res_text"], item["res_vis"], item["res_bm25"]], weights=cfg["weights"])
        clips = engine._group_and_nms(fused, 20.0, 'fused_score')
        
        match_rank = -1
        vqa_match = False
        for rank, clip in enumerate(clips[:10]):
            vid = clip["video_name"]
            start, end = clip["start_time"], clip["end_time"]
            if vid == item["true_video"] and (start - 5.0 <= item["true_pts"] <= end + 5.0):
                if match_rank == -1:
                    match_rank = rank + 1
            for fr in clip.get("frames", []):
                fr_pts = fr.get("pts_time", (start + end)/2.0)
                if vid == item["true_video"] and abs(fr_pts - item["true_pts"]) <= frame_tolerance_sec:
                    vqa_match = True
                    break
        if match_rank != -1:
            if match_rank <= 1: hits_1 += 1
            if match_rank <= 5: hits_5 += 1
            if match_rank <= 10: hits_10 += 1
        if vqa_match:
            vqa_hits += 1
            
    total = len(query_cached_results)
    print(f"\n--- {cfg['name']} ---")
    print(f"Recall@1 : {hits_1/total*100:.2f}% ({hits_1}/{total})")
    print(f"Recall@5 : {hits_5/total*100:.2f}% ({hits_5}/{total})")
    print(f"Recall@10: {hits_10/total*100:.2f}% ({hits_10}/{total})")
    print(f"VQA Prec.: {vqa_hits/total*100:.2f}% ({vqa_hits}/{total})")
