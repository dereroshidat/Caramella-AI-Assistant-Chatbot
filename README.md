# Caramella-AI-Assitant-Chatbot
Developing an AI Assistant Chatbot system that leverages AI-based work manual analysis and generating response based on the user's natural language request. This chatbot must be capable of operating in a 4GB edge Environment.
AI Assistant Chatbot optimized for 4GB edge devices using llama.cpp, Qwen2.5-1.5B Q5_K_M, and ChromaDB.

## Features
- Lightweight LLM (Q5_K_M quantization) with GPU offloading
- Multilingual embeddings (KO/EN) and fast retrieval (ChromaDB)
- Edge-friendly latency control (concise vs comprehensive)
- React frontend + FastAPI backend

## Architecture
- Backend: FastAPI + `llama.cpp` via `llama-cpp-python`
- Retrieval: ChromaDB (disk-backed), multilingual MPNET embeddings
- Frontend: React + Vite

## Requirements
- Python 3.10+
- Node.js 18+
- Optional GPU: NVIDIA CUDA (Jetson/Desktop) or CPU-only

## Quick Start

```bash
# 1) Create env and install deps
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_rag.txt

# 2) Configure environment
cp .env.example .env
# (edit paths if needed)

# 3) Prepare sample data and build a small ChromaDB
bash scripts/build_chromadb.sh

# 4) Start backend
python api/main.py

# 5) Start frontend in another terminal
cd frontend
npm install
npm run dev
```

## Configuration
- Main config: `fast_rag_config.py`
- Environment: `.env` (copy from `.env.example`)
- Key params:
  - `FAST_MAX_TOKENS=80` (concise, ~6s)
  - `RAG_GPU_LAYERS=28` (GPU offload)
  - `CHROMADB_PATH` (disk-backed vector DB)

## Evaluation
Run a small evaluation against `SAMPLE_DATA` or provided tiny JSONL sets.

```bash
python evaluate_rag_quality.py --dataset SAMPLE_DATA
```
Metrics reported include BERTScore, latency breakdown, and throughput.

## Deployment
- Docker: `Dockerfile` + `docker-compose.yml` (optional)
- Edge devices:
  - Jetson Xavier: build `llama.cpp` with CUDA (`LLAMA_CUBLAS=ON`)
  - ARM CPU-only: set `LLM_GPU_LAYERS=0`, keep `MAX_TOKENS=60–80`
- Tunnels for demo: Cloudflare Tunnel or Ngrok

## Limitations
- CPU-only devices have higher latency (15–30s for concise answers)
- Coral TPU does not accelerate transformer generation
- English performance lower than Korean with current corpus

## Folder Guide
- `api/`: FastAPI backend
- `frontend/`: React UI
- `SAMPLE_DATA/`: small public-safe dataset
- `scripts/`: helper scripts (build DB, run services)

## Citation
If you use this repo, please cite the project or link back to the GitHub page.
