import os
import sys
import json
import glob
import torch
import numpy as np
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv(override=True)
sys.stdout.reconfigure(encoding='utf-8')

from pymilvus import MilvusClient
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

try:
    from .query_distiller import QueryDistiller, DistilledQueryPayload
except ImportError:
    from query_distiller import QueryDistiller, DistilledQueryPayload

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
    
    fused_results = []
    for key, score in rrf_scores.items():
        item = item_map[key].copy()
        item['fused_score'] = score
        fused_results.append(item)
        
    fused_results.sort(key=lambda x: x['fused_score'], reverse=True)
    return fused_results


class SearchEngine:
    """
    Deep Multi-Modal Video Retrieval Engine.
    Encapsulates:
      - SigLIP-SO400M Visual Vector Search (Milvus)
      - BGE-M3 Dense Text Vector Search (Milvus)
      - Elasticsearch BM25 Keyword & OCR Text Search
    def __init__(self, milvus_uri: str = "http://localhost:19530", es_uri: str = "http://localhost:9200"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.distiller = QueryDistiller()

        # Load video metadata & captions
        self.shot_boundaries = {}
        self.shot_captions = {}
        self._load_metadata()

        # Database connections
        print("Connecting to Milvus...")
        self.milvus = MilvusClient(uri=milvus_uri)
        try:
            self.milvus.load_collection("video_shots")
        except Exception as e:
            print(f"Notice loading collection: {e}")

        print("Connecting to Elasticsearch...")
        self.es = Elasticsearch(es_uri)

        # Encoders
        print("Loading models...")
        self.use_siglip = False
        try:
            from transformers import AutoProcessor, AutoModel
            siglip_id = "google/siglip-so400m-patch14-384"
            self.siglip_processor = AutoProcessor.from_pretrained(siglip_id)
            self.siglip_model = AutoModel.from_pretrained(siglip_id).to(self.device).eval()
            self.use_siglip = True
            print(f"Loaded SigLIP model ({siglip_id})")
        except Exception as e:
            print(f"SigLIP not loaded ({e}). Falling back to MobileCLIP2-S4.")
            import open_clip
            self.clip_name = "MobileCLIP2-S4"
            self.clip_path = "model/mobileclip2_s4.pt"
            self.clip_model, _, _ = open_clip.create_model_and_transforms(self.clip_name, pretrained=self.clip_path)
            self.tokenizer = open_clip.get_tokenizer(self.clip_name)
            self.clip_model = self.clip_model.to(self.device).eval()

        self.text_model = SentenceTransformer('BAAI/bge-m3').to(self.device)

    def _load_metadata(self):
        base_dir = os.getenv("PROCESSED_DATA_DIR", "data/processed")
        
        # 1. Load shot boundaries
        shot_files = glob.glob(os.path.join(base_dir, "DAKE_output/shot_boundaries/*.json")) or glob.glob("kaggle_dataset_staging/shot_boundaries/*.json")
        for sf in shot_files:
            try:
                vid = os.path.splitext(os.path.basename(sf))[0]
                with open(sf, "r", encoding="utf-8") as f:
                    shots_data = json.load(f)
                    self.shot_boundaries[vid] = {}
                    if isinstance(shots_data, list):
                        for s in shots_data:
                            sid = s.get("shot_id", 0)
                            self.shot_boundaries[vid][sid] = (s.get("start_time", 0.0), s.get("end_time", 0.0))
            except Exception:
                pass

        # 2. Load captions
        cap_files = glob.glob(os.path.join(base_dir, "DAKE_output/captions/*.json")) or glob.glob("kaggle_dataset_staging/captions/*.json")
        for cf in cap_files:
            try:
                vid = os.path.splitext(os.path.basename(cf))[0]
                with open(cf, "r", encoding="utf-8") as f:
                    caps_data = json.load(f)
                    self.shot_captions[vid] = {}
                    if isinstance(caps_data, list):
                        for c in caps_data:
                            sid = c.get("shot_id", 0)
                            self.shot_captions[vid][sid] = c.get("caption", "")
            except Exception:
                pass

    def distill_query(self, query: str):
        """Compatibility wrapper for query distillation."""
        payload = self.distiller.distill(query)
        return {
            "visual_query": payload.visual_query,
            "text_query": payload.text_query,
            "keywords": " ".join(payload.keywords),
            "vietnamese_keywords": " ".join(payload.vietnamese_keywords)
        }

    def _search_visual(self, vis_query: str, top_k: int = 100, q_idx: int = 0) -> list[dict]:
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
        
        return [{
            "video_name": hit["entity"]["video_name"],
            "frame_idx": hit["entity"]["frame_idx"],
            "pts_time": hit["entity"]["pts_time"],
            "score": hit["distance"],
            "query_id": q_idx
        } for hit in vis_results]

    def _search_text(self, txt_query: str, top_k: int = 100, q_idx: int = 0) -> list[dict]:
        q_txt_emb = self.text_model.encode([txt_query], normalize_embeddings=True)[0].tolist()
        txt_results = self.milvus.search(
            collection_name="video_shots",
            data=[q_txt_emb],
            anns_field="text_vector",
            limit=top_k,
            output_fields=["video_name", "frame_idx", "pts_time"]
        )[0]

        return [{
            "video_name": hit["entity"]["video_name"],
            "frame_idx": hit["entity"]["frame_idx"],
            "pts_time": hit["entity"]["pts_time"],
            "score": hit["distance"],
            "query_id": q_idx
        } for hit in txt_results]

    def _search_bm25(self, bm25_query: str, en_kws: list, vn_kws: list, top_k: int = 100) -> list[dict]:
        en_str = " ".join(en_kws) if en_kws else bm25_query[:100]
        vn_str = " ".join(vn_kws) if vn_kws else ""
        
        should_clauses = [
            {"multi_match": {"query": bm25_query, "fields": ["text^2", "ocr_text^1"], "fuzziness": "AUTO", "boost": 1.0}}
        ]
        if en_kws:
            should_clauses.append({"multi_match": {"query": en_str, "fields": ["text^3"], "boost": 2.5}})
        if vn_kws:
            should_clauses.append({"match": {"ocr_text": {"query": vn_str, "boost": 4.0, "minimum_should_match": "40%"}}})
            
        es_resp = self.es.search(index="video_shots", query={"bool": {"should": should_clauses}}, size=top_k)
        
        return [{
            "video_name": hit['_source'].get('video_name', hit['_source']['id'].rsplit('_', 1)[0]),
            "frame_idx": hit['_source'].get('frame_idx', 0),
            "pts_time": hit['_source'].get('pts_time', 0.0),
            "score": hit['_score']
        } for hit in es_resp['hits']['hits']]

    def _group_and_nms(self, all_retrieved_frames, max_clip_duration, score_key, top_n=10, window_k=None):
        if window_k is None:
            window_k = int(os.getenv("TEMPORAL_WINDOW_K", "1"))

        shot_groups = defaultdict(list)
        for frame in all_retrieved_frames:
            vid = frame['video_name']
            pts_time = frame['pts_time']
            
            assigned_shot_id = -1
            if vid in self.shot_boundaries:
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
        """
        Public Search Interface:
        Accepts single query string, list of queries, or >> sequence query.
        """
        if isinstance(queries, str):
            queries = [queries]

        expanded_queries = []
        for q in queries:
            if ">>" in q:
                parts = [p.strip() for p in q.split(">>") if p.strip()]
                expanded_queries.extend(parts)
            else:
                expanded_queries.append(q)
        queries = expanded_queries

        if use_distill is None:
            use_distill = os.getenv("USE_DISTILL", "0") == "1"

        all_retrieved_frames = []
        all_vis_frames = []
        all_text_frames = []
        full_query = " ".join(queries)

        for q_idx, query in enumerate(queries):
            if use_distill:
                payload = self.distiller.distill(query, full_context_query=full_query)
                vis_q = payload.visual_query
                txt_q = payload.text_query
                bm25_q = payload.bm25_query
                en_kws = payload.keywords
                vn_kws = payload.vietnamese_keywords
            else:
                en_kws, vn_kws = self.distiller.extract_terms(query)
                vis_q = query[:200]
                txt_q = full_query if len(queries) > 1 else query
                bm25_q = full_query if len(queries) > 1 else query

            res_vis = self._search_visual(vis_q, top_k=top_k, q_idx=q_idx)
            all_vis_frames.extend(res_vis)

            res_text = self._search_text(txt_q, top_k=top_k, q_idx=q_idx)
            all_text_frames.extend(res_text)

            res_bm25 = self._search_bm25(bm25_q, en_kws, vn_kws, top_k=top_k)

            vis_weight = 2.5 if self.use_siglip else 1.0
            bm25_weight = float(os.getenv("BM25_WEIGHT", "1.0"))
            fused = reciprocal_rank_fusion([res_text, res_vis, res_bm25], weights=[3.0, vis_weight, bm25_weight])
            for item in fused:
                item['query_id'] = q_idx
                all_retrieved_frames.append(item)

        fused_clips = self._group_and_nms(all_retrieved_frames, max_clip_duration, 'fused_score', top_n=10)

        if return_separate:
            return {
                "fused": fused_clips,
                "visual": self._group_and_nms(all_vis_frames, max_clip_duration, 'score', top_n=10),
                "text": self._group_and_nms(all_text_frames, max_clip_duration, 'score', top_n=10)
            }
        return fused_clips


if __name__ == "__main__":
    test_queries = [
        "The chef begins by placing a pot of water on the stove, preparing it for boiling. He then adds a whole lobster, green onions, and garlic."
    ]
    engine = SearchEngine()
    clips = engine.search_clips(test_queries, use_distill=True)
    
    print("\nTop Candidate Clips:")
    for i, clip in enumerate(clips[:5]):
        print(f"{i+1}. Video: {clip['video_name']} | Time: {clip['start_time']:.1f}s - {clip['end_time']:.1f}s | Score: {clip['clip_score']:.4f}")
