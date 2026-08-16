import sys
import os
import json
import torch
import time
from tqdm import tqdm
from collections import defaultdict

sys.path.append(os.path.abspath("src/search"))
from test_search import SearchEngine, reciprocal_rank_fusion

sys.stdout.reconfigure(encoding='utf-8')

with open("query.json", "r", encoding="utf-8") as f:
    queries_data = json.load(f)

engine = SearchEngine()

fps = 25.0
frame_tolerance_sec = 12.0 / fps

engine.reranker.max_length = 128

print(f"\nEvaluating real English test queries from query.json ({len(queries_data)} queries)...")
query_candidates = []

start_t0 = time.time()
for item in tqdm(queries_data, desc="Stage 1 + Rerank"):
    q_text = item["query"]
    true_video = item["answer"]["video_name"]
    if "pts_time" in item["answer"]:
        true_pts = float(item["answer"]["pts_time"])
    else:
        true_pts = (float(item["answer"]["start_time"]) + float(item["answer"]["end_time"])) / 2.0
        
    # Text vector
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
    
    # BM25
    es_resp = engine.es.search(
        index="video_shots",
        query={"multi_match": {"query": q_text, "fields": ["text^2", "ocr_text^3"], "fuzziness": "AUTO"}},
        size=100
    )
    res_bm25 = [{"video_name": h["_source"].get("video_name", h["_source"]["id"].rsplit("_", 1)[0]), "frame_idx": h["_source"].get("frame_idx", 0), "pts_time": h["_source"].get("pts_time", 0.0), "score": h["_score"]} for h in es_resp['hits']['hits']]
    
    fused = reciprocal_rank_fusion([res_text, res_vis, res_bm25], weights=[3.0, 1.5, 3.5])
    
    # Group into candidate clips (up to 30 candidate clips)
    sg = defaultdict(list)
    for frame in fused:
        vid = frame['video_name']
        pts_time = frame['pts_time']
        assigned_shot_id = -1
        if hasattr(engine, 'shot_boundaries') and vid in engine.shot_boundaries:
            for s_id, (s_time, e_time) in engine.shot_boundaries[vid].items():
                if s_time <= pts_time <= e_time:
                    assigned_shot_id = s_id
                    break
        group_key = f"shot_{assigned_shot_id}" if assigned_shot_id != -1 else f"frame_{frame['frame_idx']}"
        sg[(vid, group_key)].append(frame)
        
    candidate_clips = []
    for (vid, group_key), frames in sg.items():
        score = sum(fr.get('fused_score', 0) for fr in frames)
        start_time, end_time = 0.0, 0.0
        shot_text = ""
        if group_key.startswith("shot_"):
            shot_id = int(group_key.replace("shot_", ""))
            start_time, end_time = engine.shot_boundaries[vid][shot_id]
            shot_text = engine.shot_captions.get(vid, {}).get(shot_id, "")
        else:
            start_time = frames[0]['pts_time']
            end_time = start_time + 20.0
        if not shot_text:
            shot_text = f"Video {vid} moment from {start_time:.1f}s to {end_time:.1f}s"
            
        candidate_clips.append({
            'video_name': vid,
            'start_time': start_time,
            'end_time': end_time,
            'clip_score': score,
            'shot_text': shot_text[:300],
            'frames': frames
        })
    candidate_clips.sort(key=lambda x: x['clip_score'], reverse=True)
    candidate_clips = candidate_clips[:30]
    
    # Predict reranker scores
    pairs = [[q_text, c['shot_text']] for c in candidate_clips]
    if len(pairs) > 0:
        rerank_scores = engine.reranker.predict(pairs, batch_size=32)
        for idx, s in enumerate(rerank_scores):
            candidate_clips[idx]['rerank_score'] = float(s)
    
    query_candidates.append({
        'query': q_text,
        'true_video': true_video,
        'true_pts': true_pts,
        'candidates': candidate_clips
    })

total_time = time.time() - start_t0
print(f"\nCompleted in {total_time:.2f}s ({total_time/len(query_candidates):.3f}s/query)")

total = len(query_candidates)

# Baseline without reranking
hits_1, hits_5, hits_10, vqa_hits = 0, 0, 0, 0
for qc in query_candidates:
    top10 = qc['candidates'][:10]
    match_rank = -1
    vqa_match = False
    for rank, clip in enumerate(top10):
        vid = clip["video_name"]
        start, end = clip["start_time"], clip["end_time"]
        if vid == qc['true_video'] and (start - 5.0 <= qc['true_pts'] <= end + 5.0):
            if match_rank == -1: match_rank = rank + 1
        for fr in clip.get("frames", []):
            fr_pts = fr.get("pts_time", (start + end)/2.0)
            if vid == qc['true_video'] and abs(fr_pts - qc['true_pts']) <= frame_tolerance_sec:
                vqa_match = True
                break
    if match_rank != -1:
        if match_rank <= 1: hits_1 += 1
        if match_rank <= 5: hits_5 += 1
        if match_rank <= 10: hits_10 += 1
    if vqa_match: vqa_hits += 1

print("\n=== Baseline (query.json: Stage-1 RRF Only, No Rerank) ===")
print(f"Recall@1 : {hits_1/total*100:.2f}% ({hits_1}/{total})")
print(f"Recall@5 : {hits_5/total*100:.2f}% ({hits_5}/{total})")
print(f"Recall@10: {hits_10/total*100:.2f}% ({hits_10}/{total})")
print(f"VQA Prec.: {vqa_hits/total*100:.2f}% ({vqa_hits}/{total})")

pool_sizes = [10, 15, 20, 25, 30]
blend_weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]

print("\n=== Stage-2 Reranking Grid on query.json ===")
best_r1, best_cfg = 0, ""
for pool_size in pool_sizes:
    for blend_w in blend_weights:
        h1, h5, h10, vqa = 0, 0, 0, 0
        for qc in query_candidates:
            pool = [dict(c) for c in qc['candidates'][:pool_size]]
            if len(pool) == 0:
                continue
            
            max_rrf = max(c['clip_score'] for c in pool)
            min_rrf = min(c['clip_score'] for c in pool)
            rrf_range = max_rrf - min_rrf if max_rrf > min_rrf else 1.0
            
            for c in pool:
                norm_rrf = (c['clip_score'] - min_rrf) / rrf_range
                c['final_score'] = (1.0 - blend_w) * norm_rrf + blend_w * c.get('rerank_score', 0.0)
                
            pool.sort(key=lambda x: x['final_score'], reverse=True)
            top10 = pool[:10]
            
            match_rank = -1
            vqa_match = False
            for rank, clip in enumerate(top10):
                vid = clip["video_name"]
                start, end = clip["start_time"], clip["end_time"]
                if vid == qc['true_video'] and (start - 5.0 <= qc['true_pts'] <= end + 5.0):
                    if match_rank == -1: match_rank = rank + 1
                for fr in clip.get("frames", []):
                    fr_pts = fr.get("pts_time", (start + end)/2.0)
                    if vid == qc['true_video'] and abs(fr_pts - qc['true_pts']) <= frame_tolerance_sec:
                        vqa_match = True
                        break
            if match_rank != -1:
                if match_rank <= 1: h1 += 1
                if match_rank <= 5: h5 += 1
                if match_rank <= 10: h10 += 1
            if vqa_match: vqa += 1
            
        r1_pct = h1 / total * 100
        if r1_pct > best_r1:
            best_r1 = r1_pct
            best_cfg = f"Pool={pool_size:2d}, Blend={blend_w:.1f}"
        print(f"Pool={pool_size:2d} | Blend={blend_w:.1f} -> R@1: {h1/total*100:.2f}% ({h1:2d}) | R@5: {h5/total*100:.2f}% ({h5:2d}) | R@10: {h10/total*100:.2f}% ({h10:2d}) | VQA: {vqa/total*100:.2f}% ({vqa:2d})")

print(f"\nBest Config on query.json: {best_cfg} -> Recall@1: {best_r1:.2f}%")
