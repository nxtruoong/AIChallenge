# AIChallenge - Multimodal Video Retrieval & VQA System

A high-performance, two-stage hybrid multimodal video search and Visual Question Answering (VQA) system designed for the Ho Chi Minh City AI Challenge.

---

## 🌟 Overview & Architecture

The system provides end-to-end capabilities for large-scale video archive indexing, multimodal search, temporal keyframe localization, and visual question answering.

```
                  ┌────────────────────────────────────────────────────────┐
                  │                 Interactive Web UI                     │
                  │   Query Distillation • Video Scrubbing • Gemini VQA    │
                  └──────────────────────────┬─────────────────────────────┘
                                             │ REST API
                  ┌──────────────────────────▼─────────────────────────────┐
                  │                FastAPI / Search Backend                │
                  └──────────────┬───────────────────────────┬─────────────┘
                                 │                           │
                   Stage 1: Multi-Vector Search      Stage 2: Neural Reranking
          ┌──────────────────────┼──────────────────────┐    ┌─────────────────────┐
          │                      │                      │    │  BGE Cross-Encoder  │
┌─────────▼───────────┐┌─────────▼──────────┐┌──────────▼────┴────┐ (BAAI/bge-      │
│ MobileCLIP2/SigLIP  ││ Text Embedding     ││ BM25 Sparse Index  │  reranker-base)   │
│ Visual Embeddings   ││ Semantic Captions  ││ OCR + ASR Subtitles│                   │
└─────────────────────┘└────────────────────┘└────────────────────┴─────────────────────┘
                                 │
                    Reciprocal Rank Fusion (RRF)
```

### Key Features
1. **Two-Stage Hybrid Search Pipeline**:
   - **Stage 1 (Retrieval & Fusion)**: Weighted Reciprocal Rank Fusion combining MobileCLIP2/SigLIP visual similarity, dense caption embeddings, and BM25 sparse matching on OCR and ASR speech transcripts.
   - **Stage 2 (Reranking & Expansion)**: BGE Cross-Encoder reranker (`BAAI/bge-reranker-base`) with temporal window expansion (neighboring keyframe contextual boosting).
2. **LLM Query Distillation**: Automatically parses complex natural language queries into visual descriptors, semantic summaries, and entity/Vietnamese keywords for targeted multi-field retrieval.
3. **Visual Question Answering (VQA)**: Built-in integration with Google Gemini Multimodal API to answer detailed queries on retrieved keyframes and video segments.
4. **Interactive Web Interface**: Real-time video playback, keyframe thumbnail grid, temporal scrubbing, dynamic weight adjustment, and query analysis.

---

## 📁 Repository Structure

```
.
├── api/
│   ├── api.py                    # FastAPI server exposing search & VQA endpoints
│   ├── prompt_template.txt       # System prompt for multimodal video captioning
│   └── static/                   # Web frontend assets (HTML, CSS, JavaScript)
│       ├── app.js
│       ├── index.html
│       └── styles.css
├── src/
│   ├── data/                     # Data loading and batch iteration utilities
│   ├── eval/                     # Evaluation metrics and retrieval evaluation tools
│   ├── preprocessing/            # Keyframe extraction, OCR, and Whisper ASR scripts
│   ├── search/                   # Hybrid retrieval engine and test search scripts
│   ├── utils/                    # Helper utilities and notebook update scripts
│   ├── process_recap_L26.py      # Video recap generation pipeline
│   ├── process_recap_L26_parallel.py # Parallelized video recap processor
│   ├── recap_api.py              # Multimodal captioning API client
│   └── test_latency.py           # API latency benchmarking script
├── notebooks/                    # Kaggle and local development Jupyter notebooks
│   ├── kaggle_notebook_a_preprocessing.ipynb
│   ├── kaggle_notebook_b_search.ipynb
│   ├── kaggle_generate_vectors.ipynb
│   └── whisper_template.ipynb
├── scratch/                      # Benchmark, fusion tuning, and grid-search scripts
├── docker-compose.yml            # Dockerized deployment configuration
├── requirements.txt              # Project Python dependencies
└── .env.example                  # Environment variables template
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10 or higher
- NVIDIA GPU with CUDA support (recommended for embedding extraction and reranking)
- Git & Git LFS (if handling model weights)

### 2. Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/nxtruoong/AIChallenge.git
   cd AIChallenge
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### 3. Environment Configuration

Copy the example environment configuration:
```bash
cp .env.example .env
```
Edit `.env` with your API keys:
```ini
# Google Gemini API Keys (for VQA & multimodal captioning)
GEMINI_API_KEY=your_gemini_api_key_here

# OpenRouter API Key (for LLM query distillation)
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Optional: Elasticsearch endpoint if used
ES_URL=http://localhost:9200
ES_API_KEY=your_elasticsearch_api_key_here
```

---

## 🛠️ Usage

### 1. Running the Web Application & API
Start the backend server:
```bash
python api/api.py
```
Open your browser and navigate to:
```
http://localhost:8000
```

### 2. Running Feature Extraction & Preprocessing
To extract keyframes, OCR, ASR audio transcripts, and visual embeddings:
```bash
# Process keyframes and extract embeddings
python src/preprocessing/extract_features.py --video_dir /path/to/videos --output_dir /path/to/output

# Run parallelized recap/caption generation
python src/process_recap_L26_parallel.py
```

### 3. Testing Hybrid Search CLI
```bash
python src/search/test_search.py --query "người đàn ông mặc áo xanh lái xe máy"
```

### 4. Running Benchmarks & Parameter Tuning
Evaluate fusion weights and retrieval accuracy:
```bash
python scratch/benchmark_fusion.py
python scratch/grid_search_optimal.py
```

---

## 🐳 Docker Deployment

To build and run the entire stack using Docker Compose:
```bash
docker-compose up --build
```

---

## 📄 License
This project is licensed under the MIT License.
