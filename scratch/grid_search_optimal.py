import sys
import os
import json
import torch
from collections import defaultdict

sys.path.append(os.path.abspath("src/search"))
from test_search import SearchEngine, reciprocal_rank_fusion

sys.stdout.reconfigure(encoding='utf-8')

with open("query.json", "r", encoding="utf-8") as f:
    queries_data = json.load(f)

engine = SearchEngine()

fps = 25.0
frame_tolerance_sec = 12.0 / fps

print("Pre-fetching all individual stream search results for 77 queries...")
stream_results = []

for item in queries_data:
    q_text = item["query"]
    true_video = item["answer"]["video_name"]
    if "pts_time" in item["answer"]:
        true_pts = float(item["answer"]["pts_time"])
    else:
        true_pts = (float(item["answer"]["start_time"]) + float(item["answer"]["end_time"])) / 2.0
        
    # 1. Text vector (BGE-m3)
    q_txt_emb = engine.text_model.encode([q_text], normalize_embeddings=True)[0].tolist()
    txt_res = engine.milvus.search(
        collection_name="video_shots", data=[q_txt_emb], anns_field="text_vector", limit=100,
        output_fields=["video_name", "frame_idx", "pts_time"]
    )[0]
    res_text = [{"video_name": h["entity"]["video_name"], "frame_idx": h["entity"]["frame_idx"], "pts_time": h["entity"]["pts_time"], "score": h["distance"], "query_id": 0} for h in txt_res]
    
    # 2. SigLIP visual (first 200 chars)
    with torch.no_grad():
        inputs = engine.siglip_processor(text=[q_text[:200]], return_tensors="pt", padding=True, truncation=True, max_length=64).to(engine.device)
        q_vis = engine.siglip_model.get_text_features(**inputs)
        q_vis_tensor = getattr(q_vis, 'pooler_output', q_vis[0])
        q_vis_tensor = q_vis_tensor / q_vis_tensor.norm(dim=-1, keepdim=True)
        q_vis_emb = q_vis_tensor.cpu().numpy().tolist()[0]
    vis_res = engine.milvus.search(
        collection_name="video_shots", data=[q_vis_emb], anns_field="visual_vector", limit=100,
        output_fields=["video_name", "frame_idx", "pts_time"]
    )[0]
    res_vis = [{"video_name": h["entity"]["video_name"], "frame_idx": h["entity"]["frame_idx"], "pts_time": h["entity"]["pts_time"], "score": h["distance"], "query_id": 0} for h in vis_res]
    
    # 3. BM25
    es_resp = engine.es.search(
        index="video_shots",
        query={"multi_match": {"query": q_text, "fields": ["text^2", "ocr_text^3"], "fuzziness": "AUTO"}},
        size=100
    )
    res_bm25 = [{"video_name": h["_source"].get("video_name", h["_source"]["id"].rsplit("_", 1)[0]), "frame_idx": h["_source"].get("frame_idx", 0), "pts_time": h["_source"].get("pts_time", 0.0), "score": h["_score"]} for h in es_resp['hits']['hits']]
    
    stream_results.append({
        'query': q_text,
        'true_video': true_video,
        'true_pts': true_pts,
        'res_text': res_text,
        'res_vis': res_vis,
        'res_bm25': res_bm25
    })

total = len(stream_results)
print(f"Pre-fetched {total} queries. Testing grid...")

configs = [
    ("Direct Fusion: Text 3.0, Vis 2.5, BM25 0.5 (No Rerank)", [3.0, 2.5, 0.5], False, 0.0),
    ("Direct Fusion: Text 2.0, Vis 4.0, BM25 0.5 (No Rerank)", [2.0, 4.0, 0.5], False, 0.0),
    ("Direct Fusion: Text 3.5, Vis 3.5, BM25 0.2 (No Rerank)", [3.5, 3.5, 0.2], False, 0.0),
    ("Direct Fusion: Text 2.0, Vis 5.0, BM25 0.2 (No Rerank)", [2.0, 5.0, 0.2], False, 0.0),
    ("Reranked Blend 0.2: Text 3.0, Vis 2.5, BM25 0.5", [3.0, 2.5, 0.5], True, 0.2),
    ("Reranked Blend 0.5: Text 3.0, Vis 2.5, BM25 0.5", [3.0, 2.5, 0.5], True, 0.5),
]

for name, w, use_rerank, blend_w in configs:
    h1, h5, h10, vqa = 0, 0, 0, 0
    for sr in stream_results:
        fused = reciprocal_rank_fusion([sr['res_text'], sr['res_vis'], sr['res_bm25']], weights=w)
        candidate_clips = engine._group_and_nms(fused, 20.0, 'fused_score', top_n=25 if use_rerank else 10)
        
        if use_rerank and len(candidate_clips) > 0:
            query_text = sr['query'][:200]
            pairs = [[query_text, clip.get('shot_text', '')[:200]] for clip in candidate_clips]
            scores = engine.reranker.predict(pairs, batch_size=32)
            max_rrf = max(c['clip_score'] for c in candidate_clips)
            min_rrf = min(c['clip_score'] for c in candidate_clips)
            rrf_range = max_rrf - min_rrf if max_rrf > min_rrf else 1.0
            for idx, score in enumerate(scores):
                norm_rrf = (candidate_clips[idx]['clip_score'] - min_rrf) / rrf_range
                candidate_clips[idx]['clip_score'] = (1.0 - blend_w) * norm_rrf + blend_w * float(score)
            candidate_clips.sort(key=lambda x: (x['unique_queries'], x['clip_score']), reverse=True)
            
        clips = candidate_clips[:10]
        
        match_rank = -1
        vqa_match = False
        for rank, clip in enumerate(clips):
            vid = clip["video_name"]
            start, end = clip["start_time"], clip["end_time"]
            if vid == sr['true_video'] and (start - 5.0 <= sr['true_pts'] <= end + 5.0):
                if match_rank == -1: match_rank = rank + 1
            for fr in clip.get("frames", []):
                fr_pts = fr.get("pts_time", (start + end)/2.0)
                if vid == sr['true_video'] and abs(fr_pts - sr['true_pts']) <= frame_tolerance_sec:
                    vqa_match = True
                    break
        if match_rank != -1:
            if match_rank <= 1: h1 += 1
            if match_rank <= 5: h5 += 1
            if match_rank <= 10: h10 += 1
        if vqa_match: vqa += 1
        
    print(f"\n--- {name} ---")
    print(f"Recall@1 : {h1/total*100:.2f}% ({h1:2d}/{total})")
    print(f"Recall@5 : {h5/total*100:.2f}% ({h5:2d}/{total})")
    print(f"Recall@10: {h10/total*100:.2f}% ({h10:2d}/{total})")
    print(f"VQA Prec.: {vqa/total*100:.2f}% ({vqa:2d}/{total})")
