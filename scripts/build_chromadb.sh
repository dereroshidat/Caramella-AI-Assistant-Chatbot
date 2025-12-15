#!/usr/bin/env bash
set -euo pipefail

# Build a tiny ChromaDB from SAMPLE_DATA
# Requires: Python env with requirements installed

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
DATA_FILE="$ROOT_DIR/SAMPLE_DATA/docs_ko_en.jsonl"
DB_PATH="$ROOT_DIR/data/chromadb"

mkdir -p "$ROOT_DIR/data"

python - <<'PY'
import os, json
from chromadb import Client
from chromadb.config import Settings

root = os.path.dirname(os.path.dirname(__file__))
DATA_FILE = os.path.join(root, 'SAMPLE_DATA', 'docs_ko_en.jsonl')
DB_PATH = os.path.join(root, 'data', 'chromadb')

client = Client(Settings(chroma_db_impl='duckdb+parquet', persist_directory=DB_PATH))
col = client.get_or_create_collection('caramella_paragraphs')

ids, docs, metas = [], [], []
with open(DATA_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        rec = json.loads(line)
        ids.append(rec['id'])
        docs.append(rec['text'])
        metas.append({'lang': rec['lang'], 'category': rec['category'], 'source': rec['source']})

col.add(ids=ids, documents=docs, metadatas=metas)
client.persist()
print('Built ChromaDB at', DB_PATH, 'with', len(ids), 'docs')
PY

echo "ChromaDB ready at $DB_PATH"
