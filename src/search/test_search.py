import os
import sys
import json
import glob
import torch
import open_clip
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
from elasticsearch import Elasticsearch
from collections import defaultdict

# Fix Windows console unicode print error
sys.stdout.reconfigure(encoding='utf-8')

import re

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
    for en_key in sorted(FOOD_SYNONYMS.keys(), key=lambda x: -len(x)):
        pattern = r'\b' + re.escape(en_key) + r's?\b'
        if re.search(pattern, text_lower):
            found_en.append(en_key)
            found_vn.extend(FOOD_SYNONYMS[en_key])
    return list(set(found_en)), list(set(found_vn))

def reciprocal_rank_fusion(results_lists, weights=None, k=60):
    if weights is None:
        weights = [1.0] * len(results_lists)
        
    rrf_scores = {}
    item_map = {}
    for res_list, weight in zip(results_lists, weights):
        for rank, item in enumerate(res_list):
            key = f"{item['video_name']}_{item['frame_idx']}"
            if key not in item_map:
                item_map[key] = item
                rrf_scores[key] = 0.0
            rrf_scores[key] += weight * (1.0 / (k + rank + 1))
    
    sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    fused_results = []
    for k_val in sorted_keys:
        item = item_map[k_val]
        item['fused_score'] = rrf_scores[k_val]
        fused_results.append(item)
    return fused_results

class SearchEngine:
    def __init__(self):
        print("Loading shot boundaries & captions...")
        self.shot_boundaries = {}
        self.shot_captions = {}
        caption_dir = "kaggle_dataset_staging/captions"
        for json_path in glob.glob(os.path.join(caption_dir, "*.json")):
            video_name = os.path.basename(json_path).replace(".json", "")
            self.shot_boundaries[video_name] = {}
            self.shot_captions[video_name] = {}
            with open(json_path, 'r', encoding='utf-8') as f:
                try:
                    shots = json.load(f)
                    for shot in shots:
                        s_id = shot['shot_id']
                        self.shot_boundaries[video_name][s_id] = (shot['start_time'], shot['end_time'])
                        cap = shot.get('caption', '')
                        mem = shot.get('memory', '')
                        self.shot_captions[video_name][s_id] = f"{cap} {mem}".strip()
                except Exception as e:
                    print(f"Error loading {json_path}: {e}")

        print("Connecting to Milvus...")
        self.milvus = MilvusClient(uri="http://localhost:19530")
        try:
            self.milvus.load_collection("video_shots")
        except Exception as e:
            print(f"Notice loading collection: {e}")
        
        print("Connecting to Elasticsearch...")
        self.es = Elasticsearch("http://localhost:9200")
        
        print("Loading models...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. SigLIP-SO400M / MobileCLIP for visual vector search
        self.use_siglip = False
        try:
            from transformers import AutoProcessor, AutoModel
            siglip_model_id = "google/siglip-so400m-patch14-384"
            self.siglip_processor = AutoProcessor.from_pretrained(siglip_model_id)
            self.siglip_model = AutoModel.from_pretrained(siglip_model_id).to(self.device).eval()
            self.use_siglip = True
            print(f"Loaded SigLIP model ({siglip_model_id}) for visual vector search.")
        except Exception as e:
            print(f"SigLIP not loaded ({e}). Falling back to MobileCLIP2-S4.")
            self.clip_name = "MobileCLIP2-S4"
            self.clip_path = "model/mobileclip2_s4.pt"
            self.clip_model, _, _ = open_clip.create_model_and_transforms(self.clip_name, pretrained=self.clip_path)
            self.tokenizer = open_clip.get_tokenizer(self.clip_name)
            self.clip_model = self.clip_model.to(self.device).eval()
        
        # 2. BGE-m3 for text vector search
        self.text_model = SentenceTransformer('BAAI/bge-m3').to(self.device)

        # 3. Cross-Encoder for Stage 2 Candidate Reranking
        self.reranker = None
        try:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder('BAAI/bge-reranker-base', max_length=128, device=str(self.device))
            print("Loaded Stage 2 BGE Cross-Encoder reranker.")
        except Exception as e:
            print(f"Cross-Encoder not initialized ({e}). Using stage-1 rank fusion.")


    def distill_query(self, query: str):
        import requests
        api_key = os.getenv("OPENROUTER_DISTILL_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise Exception("OPENROUTER_API_KEY not found in environment.")

        prompt = (
            "You are an expert video retrieval assistant. "
            "Given a long descriptive query about a video scene, extract and distill it into targeted search fields in JSON format:\n"
            "1. 'visual_query': A short, visual-only English phrase (<10 words) describing key visible elements, actions, and objects for CLIP/SigLIP.\n"
            "2. 'text_query': A concise English summary (1-2 sentences) of the scene including relevant synonyms for semantic vector search.\n"
            "3. 'keywords': Key English entity nouns, ingredients, objects, and actions space-separated for BM25 search.\n"
            "4. 'vietnamese_keywords': Important entities, ingredients, food names, or actions translated into Vietnamese for OCR/subtitle matching.\n\n"
            f"Input query: {query}\n\n"
            "Respond ONLY with valid JSON in raw text (no markdown formatting)."
        )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        models_to_try = [
            "google/gemini-2.0-flash-001",
            "google/gemini-flash-1.5",
            "google/gemini-2.0-flash-exp:free",
            "meta-llama/llama-3.3-70b-instruct"
        ]
        
        last_error = None
        for model_name in models_to_try:
            data = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}]
            }
            try:
                resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=8)
                if resp.status_code == 200:
                    res_json = resp.json()
                    content = res_json['choices'][0]['message']['content'].strip()
                    if content.startswith("```json"):
                        content = content.replace("```json", "").replace("```", "").strip()
                    elif content.startswith("```"):
                        content = content.replace("```", "").strip()
                    parsed = json.loads(content)
                    return parsed
                else:
                    last_error = f"Status {resp.status_code}: {resp.text}"
            except Exception as e:
                last_error = e

        print(f"Error calling OpenRouter Gemini API: {last_error}")
        raise Exception(f"Query Distillation failed: {last_error}")

    def _group_and_nms(self, all_retrieved_frames, max_clip_duration, score_key, top_n=10, window_k=None):
        if window_k is None:
            window_k = int(os.getenv("TEMPORAL_WINDOW_K", "1"))

        shot_groups = defaultdict(list)
        for frame in all_retrieved_frames:
            vid = frame['video_name']
            pts_time = frame['pts_time']
            
            assigned_shot_id = -1
            if hasattr(self, 'shot_boundaries') and vid in self.shot_boundaries:
                for s_id, (s_time, e_time) in self.shot_boundaries[vid].items():
                    if s_time <= pts_time <= e_time:
                        assigned_shot_id = s_id
                        break
                        
            group_key = f"shot_{assigned_shot_id}" if assigned_shot_id != -1 else f"frame_{frame['frame_idx']}"
            shot_groups[(vid, group_key)].append(frame)
            
        candidate_clips = []
        for (vid, group_key), frames in shot_groups.items():
            query_scores = {}
            for fr in frames:
                qid = fr.get('query_id', 0)
                score = fr.get(score_key, fr.get('score', 0))
                if qid not in query_scores or score > query_scores[qid]:
                    query_scores[qid] = score
            unique_queries = len(query_scores)
            clip_score = sum(query_scores.values())
            
            start_time, end_time = 0.0, 0.0
            shot_text = ""
            if group_key.startswith("shot_"):
                shot_id = int(group_key.replace("shot_", ""))
                all_shots = self.shot_boundaries.get(vid, {})
                s_id_min = max(0, shot_id - window_k)
                s_id_max = min(max(all_shots.keys(), default=shot_id), shot_id + window_k)
                
                start_time = all_shots.get(s_id_min, all_shots.get(shot_id, (0.0, 0.0)))[0]
                end_time = all_shots.get(s_id_max, all_shots.get(shot_id, (0.0, 0.0)))[1]
                
                # Contextual caption stitching across temporal window
                captions = [self.shot_captions.get(vid, {}).get(sid, "") for sid in range(s_id_min, s_id_max + 1) if self.shot_captions.get(vid, {}).get(sid)]
                shot_text = " ".join(captions)
            else:
                start_time = frames[0]['pts_time']
                end_time = start_time + max_clip_duration
                
            if not shot_text:
                shot_text = f"Video {vid} moment from {start_time:.1f}s to {end_time:.1f}s"
                
            candidate_clips.append({
                'video_name': vid,
                'start_time': start_time,
                'end_time': end_time,
                'unique_queries': unique_queries,
                'clip_score': clip_score,
                'covered_queries': list(query_scores.keys()),
                'frames': frames,
                'shot_text': shot_text
            })
                
        candidate_clips.sort(key=lambda x: (x['unique_queries'], x['clip_score']), reverse=True)
        if top_n is not None:
            return candidate_clips[:top_n]
        return candidate_clips

    def search_clips(self, queries, top_k=100, max_clip_duration=20.0, return_separate=False, use_distill=None):
        all_retrieved_frames = []
        all_vis_frames = []
        all_text_frames = []

        if use_distill is None:
            use_distill = os.getenv("USE_DISTILL", "0") == "1"

        for q_idx, query in enumerate(queries):
            if use_distill:
                try:
                    distilled = self.distill_query(query)
                    vis_query = distilled.get("visual_query", query[:200])
                    txt_query = distilled.get("text_query", query)
                    kws = distilled.get("keywords", "")
                    vn_kws = distilled.get("vietnamese_keywords", "")
                    bm25_query = f"{kws} {vn_kws} {vis_query}".strip() or query
                except Exception as e:
                    print(f"Distillation warning ({e}). Using raw query.")
                    vis_query = query[:200]
                    txt_query = query
                    bm25_query = query
            else:
                vis_query = query[:200]
                txt_query = query
                bm25_query = query

            # 1. Visual Search (Milvus) using SigLIP / MobileCLIP Text Encoder
            with torch.no_grad():
                if self.use_siglip:
                    inputs = self.siglip_processor(text=[vis_query], return_tensors="pt", padding=True, truncation=True, max_length=64).to(self.device)
                    q_vis_emb = self.siglip_model.get_text_features(**inputs)
                    if not isinstance(q_vis_emb, torch.Tensor):
                        if hasattr(q_vis_emb, 'pooler_output') and q_vis_emb.pooler_output is not None:
                            q_vis_emb = q_vis_emb.pooler_output
                        else:
                            q_vis_emb = q_vis_emb[0]
                    q_vis_emb /= q_vis_emb.norm(dim=-1, keepdim=True)
                    q_vis_emb = q_vis_emb.cpu().numpy().tolist()[0]
                else:
                    tokens = self.tokenizer([vis_query]).to(self.device)
                    q_vis_emb = self.clip_model.encode_text(tokens)
                    q_vis_emb /= q_vis_emb.norm(dim=-1, keepdim=True)
                    q_vis_emb = q_vis_emb.cpu().numpy().tolist()[0]

            vis_results = self.milvus.search(
                collection_name="video_shots",
                data=[q_vis_emb],
                anns_field="visual_vector",
                limit=top_k,
                output_fields=["video_name", "frame_idx", "pts_time"]
            )[0]
            
            res_vis = []
            for hit in vis_results:
                item = {
                    "video_name": hit["entity"]["video_name"],
                    "frame_idx": hit["entity"]["frame_idx"],
                    "pts_time": hit["entity"]["pts_time"],
                    "score": hit["distance"],
                    "query_id": q_idx
                }
                res_vis.append(item)
                all_vis_frames.append(item)

            # 2. Text Vector Search (Milvus) using BGE-m3
            q_txt_emb = self.text_model.encode([txt_query], normalize_embeddings=True)[0].tolist()
            txt_results = self.milvus.search(
                collection_name="video_shots",
                data=[q_txt_emb],
                anns_field="text_vector",
                limit=top_k,
                output_fields=["video_name", "frame_idx", "pts_time"]
            )[0]

            res_text = []
            for hit in txt_results:
                item = {
                    "video_name": hit["entity"]["video_name"],
                    "frame_idx": hit["entity"]["frame_idx"],
                    "pts_time": hit["entity"]["pts_time"],
                    "score": hit["distance"],
                    "query_id": q_idx
                }
                res_text.append(item)
                all_text_frames.append(item)

            # 3. Text Keyword Search (Elasticsearch) with Tuned OCR Compound Query
            en_kws, vn_kws = extract_food_terms(query)
            en_str = " ".join(en_kws) if en_kws else query[:100]
            vn_str = " ".join(vn_kws) if vn_kws else ""
            
            should_clauses = [
                {"multi_match": {"query": bm25_query, "fields": ["text^2", "ocr_text^1"], "fuzziness": "AUTO", "boost": 1.0}}
            ]
            if en_kws:
                should_clauses.append({
                    "multi_match": {"query": en_str, "fields": ["text^3"], "boost": 2.5}
                })
            if vn_kws:
                should_clauses.append({
                    "match": {"ocr_text": {"query": vn_str, "boost": 4.0, "minimum_should_match": "40%"}}
                })
                
            es_resp = self.es.search(
                index="video_shots",
                query={"bool": {"should": should_clauses}},
                size=top_k
            )
            
            res_bm25 = []
            for hit in es_resp['hits']['hits']:
                doc = hit['_source']
                res_bm25.append({
                    "video_name": doc.get('video_name', doc['id'].rsplit('_', 1)[0]),
                    "frame_idx": doc.get('frame_idx', 0),
                    "pts_time": doc.get('pts_time', 0.0),
                    "score": hit['_score']
                })

            # Optimal calibrated fusion for long descriptive English queries
            vis_weight = 2.5 if self.use_siglip else 1.0
            bm25_weight = float(os.getenv("BM25_WEIGHT", "1.0"))
            fused = reciprocal_rank_fusion([res_text, res_vis, res_bm25], weights=[3.0, vis_weight, bm25_weight])
            for item in fused:
                item['query_id'] = q_idx
                all_retrieved_frames.append(item)
        
        candidate_clips = self._group_and_nms(all_retrieved_frames, max_clip_duration, 'fused_score', top_n=25)

        # Stage 2 Cross-Encoder Reranking
        use_reranker = os.getenv("USE_CROSS_ENCODER", "0") == "1"
        if use_reranker and self.reranker and len(candidate_clips) > 0 and len(queries) > 0:
            try:
                query_text = " ".join(queries)[:200]
                pairs = [[query_text, clip.get('shot_text', '')[:200]] for clip in candidate_clips]
                    
                scores = self.reranker.predict(pairs, batch_size=32)
                
                max_rrf = max(c['clip_score'] for c in candidate_clips)
                min_rrf = min(c['clip_score'] for c in candidate_clips)
                rrf_range = max_rrf - min_rrf if max_rrf > min_rrf else 1.0

                blend_w = float(os.getenv("RERANK_BLEND_WEIGHT", "0.2"))
                for idx, score in enumerate(scores):
                    norm_rrf = (candidate_clips[idx]['clip_score'] - min_rrf) / rrf_range
                    candidate_clips[idx]['clip_score'] = (1.0 - blend_w) * norm_rrf + blend_w * float(score)

                candidate_clips.sort(key=lambda x: (x['unique_queries'], x['clip_score']), reverse=True)
            except Exception as e:
                print(f"Reranking warning: {e}")

        fused_clips = candidate_clips[:10]

        if return_separate:
            return {
                "fused": fused_clips,
                "visual": self._group_and_nms(all_vis_frames, max_clip_duration, 'score', top_n=10),
                "text": self._group_and_nms(all_text_frames, max_clip_duration, 'score', top_n=10)
            }
        return fused_clips


if __name__ == "__main__":
    test_queries = [
        "Má heo nướng táo",
        "Đầu bếp cắt xoài"
    ]
    engine = SearchEngine()
    clips = engine.search_clips(test_queries)
    
    print("\nTop 10 Candidate Clips:")
    for i, clip in enumerate(clips):
        print(f"{i+1}. Video: {clip['video_name']} | Time: {clip['start_time']:.1f}s - {clip['end_time']:.1f}s")
        print(f"   Unique Queries: {clip['unique_queries']} | Score: {clip['clip_score']:.4f}")
        print(f"   Covered Queries: {clip['covered_queries']}")
