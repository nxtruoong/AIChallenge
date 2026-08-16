import sys
import os
import json
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath("src/search"))
from test_search import SearchEngine, reciprocal_rank_fusion

sys.stdout.reconfigure(encoding='utf-8')

with open("data/eval/test_queries.json", "r", encoding="utf-8") as f:
    queries_data = json.load(f)[:77]

engine = SearchEngine()

configs = [
    {"name": "No-Rerank, BGE=3.0, SigLIP=0.0, BM25=1.0", "weights": [3.0, 0.0, 1.0], "rerank": False},
    {"name": "No-Rerank, BGE=3.0, SigLIP=0.5, BM25=0.5", "weights": [3.0, 0.5, 0.5], "rerank": False},
    {"name": "No-Rerank, BGE=3.0, SigLIP=0.5, BM25=1.5", "weights": [3.0, 0.5, 1.5], "rerank": False},
    {"name": "No-Rerank, BGE=3.0, SigLIP=1.0, BM25=2.0", "weights": [3.0, 1.0, 2.0], "rerank": False},
    {"name": "With-Rerank (blend 0.3), BGE=3.0, SigLIP=0.5, BM25=1.0", "weights": [3.0, 0.5, 1.0], "rerank": True, "blend": 0.3},
]

fps = 25.0
frame_tolerance_sec = 12.0 / fps

for cfg in configs:
    hits_1 = 0
    hits_5 = 0
    hits_10 = 0
    vqa_hits = 0
    
    # Save original reranker
    orig_reranker = engine.reranker
    if not cfg.get("rerank", False):
        engine.reranker = None
    else:
        engine.reranker = orig_reranker
        
    for item in queries_data:
        q_text = item["query"]
        true_video = item["answer"]["video_name"]
        if "pts_time" in item["answer"]:
            true_pts = float(item["answer"]["pts_time"])
        else:
            true_pts = (float(item["answer"]["start_time"]) + float(item["answer"]["end_time"])) / 2.0
            
        # Custom search with cfg weights
        all_retrieved = []
        # We can call engine.search_clips directly if we adjust weights or do custom search
        # Let's run query
        q_txt_emb = engine.text_model.encode([q_text], normalize_embeddings=True)[0].tolist()
        txt_res = engine.milvus.search(
            collection_name="video_shots", data=[q_txt_emb], anns_field="text_vector", limit=100,
            output_fields=["video_name", "frame_idx", "pts_time"]
        )[0]
        res_text = [{"video_name": h["entity"]["video_name"], "frame_idx": h["entity"]["frame_idx"], "pts_time": h["entity"]["pts_time"], "score": h["distance"], "query_id": 0} for h in txt_res]
        
        # SigLIP visual
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
        
        # BM25 ES
        es_resp = engine.es.search(
            index="video_shots",
            query={"multi_match": {"query": q_text, "fields": ["text^2", "ocr_text^3"], "fuzziness": "AUTO"}},
            size=100
        )
        res_bm25 = [{"video_name": h["_source"].get("video_name", h["_source"]["id"].rsplit("_", 1)[0]), "frame_idx": h["_source"].get("frame_idx", 0), "pts_time": h["_source"].get("pts_time", 0.0), "score": h["_score"]} for h in es_resp['hits']['hits']]
        
        fused = reciprocal_rank_fusion([res_text, res_vis, res_bm25], weights=cfg["weights"])
        clips = engine._group_and_nms(fused, 20.0, 'fused_score')
        
        if cfg.get("rerank", False) and orig_reranker and len(clips) > 0:
            pairs = []
            for clip in clips:
                vid = clip['video_name']
                shot_text = ""
                if hasattr(engine, 'shot_boundaries') and vid in engine.shot_boundaries:
                    for s_id, (s_time, e_time) in engine.shot_boundaries[vid].items():
                        if abs(s_time - clip['start_time']) < 1.0:
                            shot_text = engine.shot_captions.get(vid, {}).get(s_id, "")
                            break
                if not shot_text:
                    shot_text = f"Video {vid} moment {clip['start_time']:.1f}s"
                pairs.append([q_text, shot_text])
            scores = orig_reranker.predict(pairs)
            blend_w = cfg.get("blend", 0.5)
            for idx, score in enumerate(scores):
                # Blend normalized RRF score + reranker score
                clips[idx]['clip_score'] = (1.0 - blend_w) * clips[idx]['clip_score'] + blend_w * float(score)
            clips.sort(key=lambda x: x['clip_score'], reverse=True)
            
        match_rank = -1
        vqa_match = False
        for rank, clip in enumerate(clips[:10]):
            vid = clip["video_name"]
            start, end = clip["start_time"], clip["end_time"]
            if vid == true_video and (start - 5.0 <= true_pts <= end + 5.0):
                if match_rank == -1:
                    match_rank = rank + 1
            for fr in clip.get("frames", []):
                fr_pts = fr.get("pts_time", (start + end)/2.0)
                if vid == true_video and abs(fr_pts - true_pts) <= frame_tolerance_sec:
                    vqa_match = True
                    break
        if match_rank != -1:
            if match_rank <= 1: hits_1 += 1
            if match_rank <= 5: hits_5 += 1
            if match_rank <= 10: hits_10 += 1
        if vqa_match:
            vqa_hits += 1
            
    total = len(queries_data)
    print(f"\n--- Result: {cfg['name']} ---")
    print(f"Recall@1 : {hits_1/total*100:.2f}% ({hits_1}/{total})")
    print(f"Recall@5 : {hits_5/total*100:.2f}% ({hits_5}/{total})")
    print(f"Recall@10: {hits_10/total*100:.2f}% ({hits_10}/{total})")
    print(f"VQA Prec.: {vqa_hits/total*100:.2f}% ({vqa_hits}/{total})")
