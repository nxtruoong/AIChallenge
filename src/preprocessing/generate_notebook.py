import json
import os

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Generate SigLIP-SO400M, BGE-m3, and EasyOCR Vectors for AIHCM Database\n",
                "\n",
                "This notebook extracts visual features with SigLIP-SO400M (1152-d), text embeddings with BGE-m3 (1024-d), and OCR text using EasyOCR."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!pip install transformers sentence-transformers easyocr pandas numpy tqdm torch torchvision Pillow -q"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import glob\n",
                "import json\n",
                "import pickle\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import torch\n",
                "from PIL import Image\n",
                "from transformers import AutoProcessor, AutoModel\n",
                "from sentence_transformers import SentenceTransformer\n",
                "import easyocr\n",
                "from tqdm import tqdm\n",
                "\n",
                "DATASET_PATH = '/kaggle/input/datasets/justavnesedude/aic2026-preprocessed/kaggle_dataset_staging'\n",
                "OUTPUT_PATH = '/kaggle/working'\n",
                "\n",
                "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
                "print('Using device:', device)\n",
                "\n",
                "if not os.path.exists(DATASET_PATH):\n",
                "    matching = glob.glob('/kaggle/input/**/kaggle_dataset_staging', recursive=True)\n",
                "    if matching:\n",
                "        DATASET_PATH = matching[0]\n",
                "print('Dataset path verified:', DATASET_PATH)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Load SigLIP-SO400M for Keyframe Visual Features\n",
                "print('Loading SigLIP-SO400M...')\n",
                "model_name = 'google/siglip-so400m-patch14-384'\n",
                "processor = AutoProcessor.from_pretrained(model_name)\n",
                "model_vis = AutoModel.from_pretrained(model_name).to(device).eval()\n",
                "\n",
                "image_files = sorted(glob.glob(f'{DATASET_PATH}/keyframe_images/**/*.jpg', recursive=True) + glob.glob(f'{DATASET_PATH}/keyframe_images/**/*.png', recursive=True))\n",
                "print(f'Found {len(image_files)} images in {DATASET_PATH}/keyframe_images.')\n",
                "\n",
                "vis_records = []\n",
                "vis_embeddings = []\n",
                "\n",
                "# Fast SigLIP GPU Extraction (Batch size 64)\n",
                "batch_size = 64\n",
                "for i in tqdm(range(0, len(image_files), batch_size), desc='Extracting SigLIP Visual Vectors'):\n",
                "    batch_paths = image_files[i:i+batch_size]\n",
                "    images = []\n",
                "    for path in batch_paths:\n",
                "        try:\n",
                "            images.append(Image.open(path).convert('RGB'))\n",
                "        except Exception:\n",
                "            images.append(Image.new('RGB', (384, 384)))\n",
                "\n",
                "    with torch.no_grad():\n",
                "        inputs = processor(images=images, return_tensors='pt', padding=True).to(device)\n",
                "        image_features = model_vis.get_image_features(**inputs)\n",
                "        if not isinstance(image_features, torch.Tensor):\n",
                "            image_features = getattr(image_features, 'pooler_output', image_features[0])\n",
                "        image_features = image_features / image_features.norm(dim=-1, keepdim=True)\n",
                "        vis_embeddings.extend(image_features.cpu().numpy())\n",
                "\n",
                "    for path in batch_paths:\n",
                "        parts = path.split(os.sep)\n",
                "        video_name = parts[-2]\n",
                "        frame_id = parts[-1].replace('.jpg', '').replace('.png', '')\n",
                "        vis_records.append({'video_name': video_name, 'frame_idx': int(frame_id), 'pts_time': float(frame_id)/25.0})\n",
                "\n",
                "# Fast EasyOCR Text Extraction (Sampled 1 frame per 5 keyframes to avoid redundancy)\n",
                "print('Extracting OCR text (sampled 1 per 5 frames for 10x speedup)...')\n",
                "ocr_reader = easyocr.Reader(['en', 'vi'], gpu=torch.cuda.is_available())\n",
                "ocr_texts = [''] * len(image_files)\n",
                "sampled_indices = list(range(0, len(image_files), 5))\n",
                "\n",
                "for idx in tqdm(sampled_indices, desc='OCR Sampling'):\n",
                "    try:\n",
                "        img_ocr = Image.open(image_files[idx]).convert('RGB').resize((480, 270))\n",
                "        res = ocr_reader.readtext(np.array(img_ocr), detail=0)\n",
                "        ocr_texts[idx] = ' '.join(res)\n",
                "    except Exception:\n",
                "        pass\n",
                "\n",
                "np.save(os.path.join(OUTPUT_PATH, 'visual_embeddings.npy'), np.array(vis_embeddings, dtype=np.float32))\n",
                "df_meta = pd.DataFrame(vis_records)\n",
                "df_meta['ocr_text'] = ocr_texts\n",
                "df_meta.to_pickle(os.path.join(OUTPUT_PATH, 'metadata_df.pkl'))\n",
                "print('Visual embeddings and OCR metadata saved successfully.')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 2. Load BGE-m3 for Captions & OCR Text\n"
,
                "print('Loading BGE-m3...')\n",
                "model_text = SentenceTransformer('BAAI/bge-m3').to(device)\n",
                "\n",
                "corpus = []\n",
                "for idx, row in df_meta.iterrows():\n",
                "    text_combined = f\"{row.get('ocr_text', '')} {row.get('video_name', '')}\"\n",
                "    corpus.append([text_combined])\n",
                "\n",
                "texts = [' '.join(c) for c in corpus]\n",
                "text_embeddings = model_text.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)\n",
                "np.save(os.path.join(OUTPUT_PATH, 'text_embeddings.npy'), text_embeddings)\n",
                "\n",
                "with open(os.path.join(OUTPUT_PATH, 'bm25_corpus.pkl'), 'wb') as f:\n",
                "    pickle.dump({'corpus': corpus}, f)\n",
                "\n",
                "print('Text embeddings and BM25 corpus saved successfully.')"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

os.makedirs('d:/AIHCM/notebooks', exist_ok=True)
with open('d:/AIHCM/notebooks/kaggle_generate_vectors.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print("Kaggle vector generation notebook created at d:/AIHCM/notebooks/kaggle_generate_vectors.ipynb.")

