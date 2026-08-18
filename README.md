# AIHCM — Multi-Modal Video Retrieval Pipeline & Workspace Guide

## 1. Workspace Directory Structure

```
AIHCM/
├── api/                                    # FastAPI Backend & Web UI
│   ├── api.py                              # Search REST API endpoints & video static mounts
│   ├── prompt_template.txt                 # ReCap prompt with recurrent memory schema
│   └── static/                             # Search UI assets (HTML, CSS, JS)
│       ├── index.html
│       ├── styles.css
│       └── app.js
├── data/                                   # Data Store (Git-ignored)
│   ├── raw/                                # Raw video files & media-info JSONs
│   │   ├── Videos_L26_b/
│   │   ├── Videos_L26_c/
│   │   └── media-info-aic25-b1/
│   └── processed/                          # Pipeline outputs
│       ├── DAKE_output/                    # Extracted keyframes, audio, subtitles, shots, captions
│       │   ├── extracted_keyframe_images/
│       │   ├── extracted_keyframe_csvs/
│       │   ├── extracted_subtitles/
│       │   ├── shot_boundaries/
│       │   ├── ocr/
│       │   └── captions/
│       └── kaggle_output_0002/             # Downloaded vector embeddings & metadata DataFrames
├── notebooks/                              # Kaggle GPU Indexing Notebooks
│   └── kaggle_generate_vectors.ipynb       # SigLIP-SO400M + BGE-m3 + EasyOCR vector builder
├── src/                                    # Main Pipeline Source Code
│   ├── preprocessing/                      # Preprocessing & Video Ingestion
│   │   ├── dake_keyframe_extraction.py     # Dynamic-aware keyframe extraction (U-CESE)
│   │   ├── extract_audio.py                # Video audio track extractor (.mp3)
│   │   ├── run_whisper_local.py            # Local faster-whisper ASR transcription
│   │   ├── detect_shots_dake.py            # Adaptive temporal gap shot boundary detection
│   │   ├── extract_ocr_keyframes.py        # EasyOCR text extractor from keyframes
│   │   └── recap/                          # Recurrent Video Captioning (ReCap)
│   │       ├── recap_api.py                # OpenRouter API client (MiMo v2.5 / Gemini)
│   │       ├── process_recap_parallel.py   # Multi-worker parallel batch caption generator
│   │       └── test_recap_L21_V001.py      # Single-video verification driver
│   ├── data/                               # Database Ingestion
│   │   └── ingest_to_db.py                 # Milvus (HNSW vectors) & Elasticsearch (BM25) bulk loader
│   ├── search/                             # Search Engine Core
│   │   ├── __init__.py                     # Package export
│   │   ├── test_search.py                  # Multi-Modal SearchEngine (SigLIP + BGE + BM25 + RRF)
│   │   └── query_distiller.py              # LLM Query Distillation (Gemini via OpenRouter)
│   └── eval/                               # Evaluation & Benchmarking
│       └── evaluate.py                     # Strict ground truth benchmark evaluator (R@1, R@5, R@10)
├── docker-compose.yml                      # Milvus (Standalone) + Elasticsearch container orchestration
├── query.json                              # Ground-truth evaluation queries & timestamps
├── requirements.txt                        # Python dependencies
├── workspace_guide.md                      # Detailed technical guide
└── README.md                               # Project documentation
```

---

## 2. Environment Setup
- Dependencies in `requirements.txt`
- Install: `pip install -r requirements.txt`
- Create `.env` file in root with API keys:
  ```ini
  # ReCap Captioning (MiMo v2.5 via OpenRouter)
  OPENROUTER_API_KEY=sk-or-v1-...
  RECAP_MODEL=xiaomi/mimo-v2.5

  # Query Distillation (Gemini via OpenRouter / Google API)
  OPENROUTER_DISTILL_API_KEY=sk-or-v1-...
  DISTILL_MODEL=google/gemini-2.5-flash
  GEMINI_API_KEY=AIza...

  # Databases
  ES_URL=http://localhost:9200
  MILVUS_URI=http://localhost:19530
  ```
- Place videos in `data/raw/` (e.g. `Videos_L26_b`, `Videos_L26_c`, `Videos_L26_d`, `media-info-aic25-b1`)

---

## 3. Data Pipeline Workflow
All preprocessing code is located in `src/preprocessing/`.

### 3.1 Keyframe Extraction (DAKE)
- Script: `src/preprocessing/dake_keyframe_extraction.py`
- Extracts dynamic-aware keyframes (U-CESE Algorithm 1) into `data/processed/DAKE_output/extracted_keyframe_images/`

### 3.2 Audio Extraction & Transcription (ASR)
- Audio: `src/preprocessing/extract_audio.py` (extracts `.mp3`)
- Subtitles: `src/preprocessing/run_whisper_local.py` (uses `faster-whisper` `large-v3-turbo` with int8 quantization)
- Outputs subtitle JSONs into `data/processed/DAKE_output/extracted_subtitles/`

### 3.3 Shot Boundary Detection
- Script: `src/preprocessing/detect_shots_dake.py`
- Groups keyframes into temporal shot units with metadata (`start_time`, `end_time`, keyframe indices)
- **Important Note on DAKE Keyframe Density**: DAKE guarantees $\ge 1$ keyframe every 2.0s (`delta = 2*fps`). To prevent shots from falling back to the 30s timeout ceiling, set `--min-gap-sec 0.6` and `--gap-multiplier 1.8`:
  ```powershell
  python src/preprocessing/detect_shots_dake.py --min-gap-sec 0.6 --gap-multiplier 1.8 --max-shot-sec 15.0
  ```
- Outputs JSON files into `data/processed/DAKE_output/shot_boundaries/`

### 3.4 OCR Keyframe Extraction
- Script: `src/preprocessing/extract_ocr_keyframes.py`
- Runs EasyOCR across keyframes to extract screen text strings (`ocr_text`)

### 3.5 ReCap Video Captioning (with Recurrent Memory)
- **API Client**: `src/preprocessing/recap/recap_api.py` (OpenRouter API client for `xiaomi/mimo-v2.5`, image downscaling to 512px, exponential backoff)
- **Prompt Template**: `api/prompt_template.txt` (Structured recurrent memory $M_t$ with tag pruning and detailed shot description)
- **Single-Video Smoke Test**:
  ```powershell
  python src/preprocessing/recap/test_recap_L21_V001.py --video-id L21_V001 --base-dir data/processed
  ```
- **Parallel Batch Processing**:
  ```powershell
  # Batch process videos with 5 workers
  python src/preprocessing/recap/process_recap_parallel.py --batch-range 101 200 --batch-prefix L26_V --workers 5
  ```
- Outputs saved to `data/processed/DAKE_output/captions/{video_id}.json`

---

## 4. Vector Generation & Database Ingestion

### 4.1 Vector Extraction (Kaggle GPU)
- Notebook: `notebooks/kaggle_generate_vectors.ipynb`
- Extracts:
  - **Visual vectors**: SigLIP-SO400M (`1152-d`)
  - **Dense text vectors**: BGE-m3 (`1024-d`) over concatenated captions, transcripts & video metadata
  - **Metadata DataFrame**: `metadata_df.pkl` (contains timestamps, frame indices, OCR strings)
  - **BM25 Corpus**: `bm25_corpus.pkl`
- Save downloaded outputs to `data/processed/kaggle_output_0002/`

### 4.2 Database Ingestion (Milvus + Elasticsearch)
- Ingestion Script: `src/data/ingest_to_db.py`
- **Milvus Collection (`video_shots`)**:
  - `visual_vector` (1152-d, HNSW index, Metric: IP)
  - `text_vector` (1024-d, HNSW index, Metric: IP)
- **Elasticsearch Index (`video_shots`)**:
  - Standard text field: `text` (captions + transcript)
  - Standard text field: `ocr_text` (extracted on-screen text)

---

## 5. Search Engine Architecture
- Core Engine: `src/search/test_search.py`
- **Modules**:
  - `QueryDistiller` (`src/search/query_distiller.py`): Uses Gemini / OpenRouter API to split raw queries into visual subqueries, dense text query, English BM25 keywords, and Vietnamese OCR keywords.
  - `Reciprocal Rank Fusion (RRF)`: Combines Visual dense hits (SigLIP in Milvus), Text dense hits (BGE-m3 in Milvus), and BM25 text/OCR hits (Elasticsearch).
  - `Non-Maximum Suppression (NMS)`: Groups retrieved frame hits into contiguous video clips with window thresholding.

---

## 6. API Server & Web UI
- Server: `api/api.py` (FastAPI)
- UI: `api/static/`
- Run server:
  ```powershell
  uvicorn api.api:app --host 0.0.0.0 --port 8000
  ```

---

## 7. Benchmark Evaluation
- Benchmark Evaluator: `src/eval/evaluate.py`
- Ground Truth: `query.json`
- Run evaluation:
  ```powershell
  # Baseline Search
  python src/eval/evaluate.py --input query.json

  # With LLM Query Distillation ON
  python src/eval/evaluate.py --input query.json --use_distill
  ```

---

## 8. Complete Execution Pipeline

```powershell
# 1. Start Vector DB & Search Engine
docker-compose up -d

# 2. Extract Keyframes & Audio
python src/preprocessing/dake_keyframe_extraction.py
python src/preprocessing/extract_audio.py
python src/preprocessing/run_whisper_local.py
python src/preprocessing/detect_shots_dake.py --min-gap-sec 0.6 --gap-multiplier 1.8 --max-shot-sec 15.0
python src/preprocessing/extract_ocr_keyframes.py

# 3. Generate ReCap Captions (MiMo v2.5)
python src/preprocessing/recap/process_recap_parallel.py --base-dir data/processed --workers 5

# 4. Ingest vectors & metadata into Milvus & Elasticsearch
python src/data/ingest_to_db.py

# 5. Launch FastAPI & Search Interface
uvicorn api.api:app --host 0.0.0.0 --port 8000
```
