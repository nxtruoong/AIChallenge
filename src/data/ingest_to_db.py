import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from pymilvus import MilvusClient, DataType
from elasticsearch import Elasticsearch, helpers

# --- Config ---
KAGGLE_OUTPUT_DIR = Path(os.getenv("KAGGLE_OUTPUT_DIR", "d:/AIHCM/data/processed/kaggle_output_0002"))
MILVUS_URI = "http://localhost:19530"
ES_HOST = "http://localhost:9200"

COLLECTION_NAME = "video_shots"
VISUAL_DIM = 1152   # SigLIP-SO400M
TEXT_DIM = 1024     # BGE-m3

def setup_milvus(visual_dim, text_dim):
    print(f"Connecting to Milvus at {MILVUS_URI}...")
    client = MilvusClient(uri=MILVUS_URI)
    
    if client.has_collection(collection_name=COLLECTION_NAME):
        print(f"Collection {COLLECTION_NAME} already exists. Dropping it...")
        client.drop_collection(collection_name=COLLECTION_NAME)
        
    schema = client.create_schema(auto_id=False, description="Video shots multi-modal vectors")
    schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=255)
    schema.add_field(field_name="video_name", datatype=DataType.VARCHAR, max_length=255)
    schema.add_field(field_name="frame_idx", datatype=DataType.INT64)
    schema.add_field(field_name="pts_time", datatype=DataType.DOUBLE)
    schema.add_field(field_name="visual_vector", datatype=DataType.FLOAT_VECTOR, dim=visual_dim)
    schema.add_field(field_name="text_vector", datatype=DataType.FLOAT_VECTOR, dim=text_dim)
    
    print("Creating Milvus collection & indexes...")
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="visual_vector", metric_type="IP", index_type="HNSW", params={"M": 16, "efConstruction": 200})
    index_params.add_index(field_name="text_vector", metric_type="IP", index_type="HNSW", params={"M": 16, "efConstruction": 200})
    
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params
    )
    
    client.load_collection(collection_name=COLLECTION_NAME)
    print("Milvus setup complete.")
    return client

def setup_elasticsearch():
    print(f"Connecting to Elasticsearch at {ES_HOST}...")
    es = Elasticsearch(ES_HOST)
    
    if es.indices.exists(index=COLLECTION_NAME):
        print(f"Index {COLLECTION_NAME} already exists. Deleting it...")
        es.indices.delete(index=COLLECTION_NAME)
        
    mapping = {
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "video_name": {"type": "keyword"},
                "video_title": {"type": "text", "analyzer": "standard"},
                "video_description": {"type": "text", "analyzer": "standard"},
                "video_keywords": {"type": "text", "analyzer": "standard"},
                "text": {"type": "text", "analyzer": "standard"},
                "ocr_text": {"type": "text", "analyzer": "standard"},
                "frame_idx": {"type": "integer"},
                "pts_time": {"type": "float"}
            }
        }
    }
    
    es.indices.create(index=COLLECTION_NAME, body=mapping)
    print("Elasticsearch setup complete.")
    return es

def main():
    print("Loading data assets...")
    out_dir = KAGGLE_OUTPUT_DIR
    if not (out_dir / "metadata_df.pkl").exists():
        fallback_dirs = [
            Path("data/processed/kaggle_output_0002"),
            Path("data/processed/kaggle_output"),
            Path("kaggle_output")
        ]
        for fb in fallback_dirs:
            if (fb / "metadata_df.pkl").exists():
                out_dir = fb
                break

    print(f"Loading assets from {out_dir}...")
    df = pd.read_pickle(out_dir / "metadata_df.pkl")
    vis_emb = np.load(out_dir / "visual_embeddings.npy")
    txt_emb = np.load(out_dir / "text_embeddings.npy")
    
    with open(out_dir / "bm25_corpus.pkl", "rb") as f:
        bm25_data = pickle.load(f)
        corpus = bm25_data["corpus"] if isinstance(bm25_data, dict) and "corpus" in bm25_data else bm25_data
        
    assert len(df) == len(vis_emb) == len(txt_emb) == len(corpus), f"Data lengths do not match! {len(df)} vs {len(vis_emb)} vs {len(txt_emb)} vs {len(corpus)}"
    
    visual_dim = vis_emb.shape[1]
    text_dim = txt_emb.shape[1]
    print(f"Detected dimensions -> Visual: {visual_dim} (SigLIP-SO400M), Text: {text_dim} (BGE-m3 Multimodal)")
    
    client = setup_milvus(visual_dim, text_dim)
    es = setup_elasticsearch()
    
    print("Preparing bulk insert...")
    ids = []
    for _, row in df.iterrows():
        n_val = row['n'] if 'n' in row else row['frame_idx']
        ids.append(f"{row['video_name']}_{n_val:06d}")
        
    video_names = df['video_name'].tolist()
    frame_idxs = df['frame_idx'].tolist()
    pts_times = df['pts_time'].tolist()
    ocr_texts = df['ocr_text'].tolist() if 'ocr_text' in df.columns else [""] * len(df)
    video_titles = df['video_title'].tolist() if 'video_title' in df.columns else [""] * len(df)
    video_descriptions = df['video_description'].tolist() if 'video_description' in df.columns else [""] * len(df)
    video_keywords = df['video_keywords'].tolist() if 'video_keywords' in df.columns else [""] * len(df)
    
    print("Inserting into Milvus...")
    batch_size = 1000
    for i in tqdm(range(0, len(ids), batch_size), desc="Milvus Insert"):
        data = []
        for j in range(i, min(i + batch_size, len(ids))):
            data.append({
                "id": ids[j],
                "video_name": video_names[j],
                "frame_idx": frame_idxs[j],
                "pts_time": pts_times[j],
                "visual_vector": vis_emb[j].tolist(),
                "text_vector": txt_emb[j].tolist()
            })
        client.insert(collection_name=COLLECTION_NAME, data=data)
    
    res = client.query(collection_name=COLLECTION_NAME, filter="id != ''", output_fields=["count(*)"])
    print(f"Milvus entities count: {res}")
    
    print("Inserting into Elasticsearch...")
    es_actions = []
    for i in tqdm(range(len(ids)), desc="ES Prepare"):
        item_text = corpus[i]
        doc_text = " ".join(item_text) if isinstance(item_text, (list, tuple)) else str(item_text)
        ocr_val = str(ocr_texts[i]) if ocr_texts[i] else ""
        if ocr_val and ocr_val not in doc_text:
            doc_text = f"{doc_text} {ocr_val}"
            
        es_actions.append({
            "_index": COLLECTION_NAME,
            "_id": ids[i],
            "_source": {
                "id": ids[i],
                "video_name": video_names[i],
                "video_title": str(video_titles[i]) if video_titles[i] else "",
                "video_description": str(video_descriptions[i]) if video_descriptions[i] else "",
                "video_keywords": str(video_keywords[i]) if video_keywords[i] else "",
                "text": doc_text,
                "ocr_text": ocr_val,
                "frame_idx": frame_idxs[i],
                "pts_time": pts_times[i]
            }
        })
        
    helpers.bulk(es, es_actions, chunk_size=1000)
    es.indices.refresh(index=COLLECTION_NAME)
    
    count_res = es.count(index=COLLECTION_NAME)
    print(f"Elasticsearch entities count: {count_res['count']}")
    print("Ingestion complete!")

if __name__ == "__main__":
    main()
