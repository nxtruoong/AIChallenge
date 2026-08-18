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
│   │   ├── detect_shots_transnet.py        # TransNet V2 Shot Boundary Detection (ADR 0004)
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
  # ReCap Captioning (apimaster.ai / OpenAI-compatible / OpenRouter)
  RECAP_API_KEY=sk-...
  RECAP_API_BASE=https://api.apimaster.ai/v1/chat/completions
  RECAP_MODEL=gpt-5.4-mini

  # Query Distillation (Gemini via OpenRouter / Google API)
  OPENROUTER_DISTILL_API_KEY=sk-or-v1-...
  DISTILL_MODEL=google/gemini-2.5-flash
  GEMINI_API_KEY=AIza...

  # Databases
  ES_URL=http://localhost:9200
  MILVUS_URI=http://localhost:19530
  ```
- Place videos in `data/raw/` (e.g. `Videos_L26_b`, `Videos_L26_c`, `Videos_L26_d`, and `TrainingData/media-info-aic25-b1/media-info/`)

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

### 3.3 Shot Boundary Detection (TransNet V2 + Smart Merge)

Shot segmentation uses **TransNet V2** (PyTorch 3D-CNN) with **`smart_merge_shots`** (greedy shortest-neighbor iterative merge, $T_{min} = 10.0$s) to generate natural shot boundaries and map DAKE keyframes into shot intervals.

#### Method A: Local Multi-Core CPU Execution
```powershell
# Run with 4 CPU worker processes on L26 videos
python src/preprocessing/detect_shots_transnet.py --prefix L26 --workers 4 --min-shot-sec 10.0 --overwrite

# Smoke test on a single video
python src/preprocessing/detect_shots_transnet.py --test
```

#### Method B: Kaggle GPU Cloud Execution (~5-7s / video)
1. **Accelerator**: Select **GPU T4 x 2** (Settings $\to$ Accelerator $\to$ **GPU T4 x 2** for fastest CUDA batch processing).
2. **Kaggle Dataset Input Structure**:
   ```
   /kaggle/input/
   └── <dataset-name>/               # e.g., l21-01-31 or l26-100-399
       └── Videos_<batch>/           # e.g., Videos_L21_a or Videos_L26_b
           ├── L21_V001.mp4
           ├── L21_V002.mp4
           └── ...
   ```
3. Open [`notebooks/kaggle_transnet_shots.ipynb`](notebooks/kaggle_transnet_shots.ipynb) on Kaggle.
4. Set `START_VIDEO_ID = None` (for all videos) or e.g. `"L26_V128"` to resume from a specific ID.
5. Run the notebook to generate and download `shot_boundaries.zip`.
6. Extract JSON files into `data/processed/DAKE_output/shot_boundaries/`.

- Outputs: `data/processed/DAKE_output/shot_boundaries/{video_id}.json`

### 3.4 OCR Keyframe Extraction
- Script: `src/preprocessing/extract_ocr_keyframes.py`
- Runs EasyOCR across keyframes to extract screen text strings (`ocr_text`)

### 3.5 ReCap Video Captioning (with Recurrent Memory)
- **API Client**: `src/preprocessing/recap/recap_api.py` (Supports `gpt-5.4-mini` on `apimaster.ai` or `xiaomi/mimo-v2.5` on OpenRouter with exponential backoff)
- **Multi-Modal Inputs Fed Per Shot**:
  1. **Video Info**: Title & Description auto-discovered from `data/raw/**/media-info/{video_id}.json`.
  2. **Subtitle ($S_t$)**: Audio transcript matching current shot interval `[start_time, end_time]`.
  3. **Keyframes ($K_t$)**: Midpoint DAKE image representing visual scene ($512\times 512$ JPEG).
  4. **Previous Memory ($M_{t-1}$)**: Recurrent context carry-over with dynamic entity tagging and pruning.
- **Prompt Template**: `api/prompt_template.txt` (Structured recurrent memory $M_t$ with dynamic tag pruning and strict English captions with Vietnamese entity retention)
- **Parallel Multi-Worker Batch Processing**:
  ```powershell
  # Run parallel caption generation across all L26 shot files with 10 workers
  python src/preprocessing/recap/process_recap_parallel.py --base-dir data/processed --workers 10
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
python src/preprocessing/detect_shots_transnet.py --min-shot-sec 1.5 --max-shot-sec 30.0
python src/preprocessing/extract_ocr_keyframes.py

# 3. Generate ReCap Captions (MiMo v2.5)
python src/preprocessing/recap/process_recap_parallel.py --base-dir data/processed --workers 5

# 4. Ingest vectors & metadata into Milvus & Elasticsearch
python src/data/ingest_to_db.py

# 5. Launch FastAPI & Search Interface
uvicorn api.api:app --host 0.0.0.0 --port 8000
```
