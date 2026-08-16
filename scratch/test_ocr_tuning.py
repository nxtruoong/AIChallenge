import sys
import os
import json
import re
import torch
from collections import defaultdict

sys.path.append(os.path.abspath("src/search"))
from test_search import SearchEngine, reciprocal_rank_fusion

sys.stdout.reconfigure(encoding='utf-8')

FOOD_SYNONYMS = {
    "prawn": ["tôm", "tôm sú", "tôm thẻ"],
    "shrimp": ["tôm", "tép", "tôm khô"],
    "lobster": ["tôm hùm", "tôm"],
    "crab": ["cua", "thịt cua", "gạch cua"],
    "fish": ["cá", "thịt cá", "phi lê cá"],
    "beef": ["thịt bò", "bò", "thăn bò"],
    "pork": ["thịt heo", "thịt lợn", "sườn heo", "thịt ba chỉ"],
    "chicken": ["thịt gà", "gà", "ức gà", "đùi gà"],
    "meatball": ["thịt viên", "chả viên", "mọc"],
    "egg": ["trứng", "lòng đỏ", "lòng trắng"],
    "mayonnaise": ["ajimayo", "xốt mayonnaise", "mayonnaise"],
    "mustard": ["mù tạt", "yellow mustard"],
    "yogurt": ["sữa chua", "yaourt"],
    "milk": ["sữa tươi", "sữa đặc"],
    "sesame": ["mè", "vừng", "mè rang"],
    "chili": ["ớt", "ớt hiểm", "ớt sừng"],
    "chilli": ["ớt", "ớt hiểm", "sa tế"],
    "bell pepper": ["ớt chuông", "ớt đà lạt"],
    "pepper": ["tiêu", "tiêu xay", "hạt tiêu"],
    "garlic": ["tỏi", "tỏi băm", "tỏi phi"],
    "onion": ["hành tây", "hành tím", "củ hành"],
    "green onion": ["hành lá", "đầu hành"],
    "scallion": ["hành lá", "đầu hành lá"],
    "ginger": ["gừng", "gừng băm", "gừng lát"],
    "tomato": ["cà chua", "sốt cà chua"],
    "potato": ["khoai tây"],
    "carrot": ["cà rốt"],
    "pumpkin": ["bí đỏ", "bí ngô"],
    "rice": ["cơm", "gạo", "xôi", "nếp"],
    "sticky rice": ["gạo nếp", "xôi nếp", "xôi"],
    "noodle": ["mì", "bún", "phở", "hủ tiếu"],
    "bread": ["bánh mì", "sandwich"],
    "pancake": ["bánh rán", "bánh xèo", "bánh pancake"],
    "banh ran": ["bánh rán", "bột bánh rán"],
    "oil": ["dầu ăn", "dầu mè", "dầu hào", "dầu đậu phộng"],
    "sugar": ["đường", "đường cát"],
    "salt": ["muối", "muối hạt"],
    "sauce": ["nước sốt", "nước chấm", "xốt", "gia vị"],
    "soy sauce": ["nước tương", "xì dầu"],
    "fish sauce": ["nước mắm"],
    "vinegar": ["giấm", "giấm gạo"],
    "ajinomoto": ["ajinomoto", "bột ngọt", "ajingon", "hạt nêm"],
    "seasoning": ["hạt nêm", "gia vị", "bột ngọt"],
    "salad": ["salad", "gỏi", "rau trộn"],
    "soup": ["canh", "súp", "nước dùng", "nước lèo"],
    "broth": ["nước dùng", "nước hầm", "nước lèo"],
    "mango": ["xoài", "xoài chín", "xoài cát"],
    "peanut": ["đậu phộng", "lạc", "đậu phộng rang"],
    "bean": ["đậu", "đậu que", "đậu rồng", "đậu cô ve"],
    "winged bean": ["đậu rồng"],
    "cucumber": ["dưa leo", "dưa chuột"],
    "lemon": ["chanh", "nước cốt chanh"],
    "lime": ["chanh", "chanh tươi"],
    "orange": ["cam", "vỏ cam", "nước cam"],
    "coconut": ["nước dừa", "nước cốt dừa"],
    "vegetable": ["rau", "rau củ", "rau thơm"]
}

def extract_food_terms(text):
    text_lower = text.lower()
    found_en = []
    found_vn = []
    
    # Sort keys by length descending to match multi-word phrases first
    for en_key in sorted(FOOD_SYNONYMS.keys(), key=lambda x: -len(x)):
        pattern = r'\b' + re.escape(en_key) + r's?\b'
        if re.search(pattern, text_lower):
            found_en.append(en_key)
            found_vn.extend(FOOD_SYNONYMS[en_key])
            
    return list(set(found_en)), list(set(found_vn))

with open("query.json", "r", encoding="utf-8") as f:
    queries_data = json.load(f)

engine = SearchEngine()
fps = 25.0
frame_tolerance_sec = 12.0 / fps

print("Pre-fetching baseline search streams for 77 queries...")
stream_results = []
for item in queries_data:
    q_text = item["query"]
    true_video = item["answer"]["video_name"]
    if "pts_time" in item["answer"]:
        true_pts = float(item["answer"]["pts_time"])
    else:
        true_pts = (float(item["answer"]["start_time"]) + float(item["answer"]["end_time"])) / 2.0
        
    q_txt_emb = engine.text_model.encode([q_text], normalize_embeddings=True)[0].tolist()
    txt_res = engine.milvus.search(
        collection_name="video_shots", data=[q_txt_emb], anns_field="text_vector", limit=100,
        output_fields=["video_name", "frame_idx", "pts_time"]
    )[0]
    res_text = [{"video_name": h["entity"]["video_name"], "frame_idx": h["entity"]["frame_idx"], "pts_time": h["entity"]["pts_time"], "score": h["distance"], "query_id": 0} for h in txt_res]
    
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
    
    # 1. Standard BM25
    es_std = engine.es.search(
        index="video_shots",
        query={"multi_match": {"query": q_text, "fields": ["text^2", "ocr_text^3"], "fuzziness": "AUTO"}},
        size=100
    )
    res_bm25_std = [{"video_name": h["_source"].get("video_name", h["_source"]["id"].rsplit("_", 1)[0]), "frame_idx": h["_source"].get("frame_idx", 0), "pts_time": h["_source"].get("pts_time", 0.0), "score": h["_score"]} for h in es_std['hits']['hits']]
    
    # 2. Tuned OCR Compound Query
    en_kws, vn_kws = extract_food_terms(q_text)
    en_str = " ".join(en_kws) if en_kws else q_text[:100]
    vn_str = " ".join(vn_kws) if vn_kws else ""
    
    should_clauses = [
        {"multi_match": {"query": q_text, "fields": ["text^2", "ocr_text^1"], "fuzziness": "AUTO", "boost": 1.0}}
    ]
    if en_kws:
        should_clauses.append({
            "multi_match": {"query": en_str, "fields": ["text^3"], "boost": 2.5}
        })
    if vn_kws:
        should_clauses.append({
            "match": {"ocr_text": {"query": vn_str, "boost": 4.0, "minimum_should_match": "40%"}}
        })
        
    es_tuned = engine.es.search(
        index="video_shots",
        query={"bool": {"should": should_clauses}},
        size=100
    )
    res_bm25_tuned = [{"video_name": h["_source"].get("video_name", h["_source"]["id"].rsplit("_", 1)[0]), "frame_idx": h["_source"].get("frame_idx", 0), "pts_time": h["_source"].get("pts_time", 0.0), "score": h["_score"]} for h in es_tuned['hits']['hits']]
    
    stream_results.append({
        'query': q_text,
        'true_video': true_video,
        'true_pts': true_pts,
        'res_text': res_text,
        'res_vis': res_vis,
        'res_bm25_std': res_bm25_std,
        'res_bm25_tuned': res_bm25_tuned
    })

total = len(stream_results)
print(f"Pre-fetched {total} queries. Comparing OCR Stream Tuning...")

eval_modes = [
    ("Standard BM25 + Temporal Expansion", 'res_bm25_std', [3.0, 2.5, 0.5]),
    ("Tuned OCR Compound Query + Temporal Expansion (w=0.5)", 'res_bm25_tuned', [3.0, 2.5, 0.5]),
    ("Tuned OCR Compound Query + Temporal Expansion (w=1.0)", 'res_bm25_tuned', [3.0, 2.5, 1.0]),
    ("Tuned OCR Compound Query + Temporal Expansion (w=1.5)", 'res_bm25_tuned', [3.0, 2.5, 1.5]),
]

for name, bm25_key, w in eval_modes:
    h1, h5, h10, vqa = 0, 0, 0, 0
    for sr in stream_results:
        fused = reciprocal_rank_fusion([sr['res_text'], sr['res_vis'], sr[bm25_key]], weights=w)
        clips = engine._group_and_nms(fused, 20.0, 'fused_score', top_n=10, window_k=1)
        
        match_rank = -1
        vqa_match = False
        for rank, clip in enumerate(clips):
            vid = clip["video_name"]
            start, end = clip["start_time"], clip["end_time"]
            if vid == sr['true_video'] and (start - 2.0 <= sr['true_pts'] <= end + 2.0):
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
