#!/usr/bin/env python3
"""
ingest_caramella.py

Build a fresh local Chroma vector database from the preprocessed paragraph corpus.

Targets:
  - Input: JSONL file with lines shaped {"id":..., "text":..., "metadata":{...}}
  - Output: Persistent Chroma DB at --db-path (default ./caramella_vector_db)
  - Collection name: --collection (default caramella_paragraphs)

Features:
  - Batched embeddings (configurable --batch-size)
  - Resume support via processed_ids log (skip already ingested ids)
  - Periodic progress + persistence (--persist-interval)
  - Graceful retry on transient embedding/DB errors
  - Lightweight (no LangChain dependency)

Usage Examples:
  Dry run first 5k paragraphs:
    python ingest_caramella.py --limit 5000

  Full ingestion in background while chatbot runs:
    nohup python ingest_caramella.py > ingest_caramella.log 2>&1 &

  Resume after interruption:
    python ingest_caramella.py --resume

Query test while ingest running (same process or separate):
    python - <<'EOF'
    import chromadb
    client = chromadb.PersistentClient(path='./caramella_vector_db')
    col = client.get_collection('caramella_paragraphs')
    q = 'query: 브런치 콘서트 단체 예매 방법'
    res = col.query(query_texts=[q], n_results=3)
    for t,m in zip(res['documents'][0], res['metadatas'][0]):
        print('\n---\n', t[:400])
        print('meta:', m)
    EOF

Note: Paragraph texts already include the E5 prefix 'passage:' from preprocessing.
"""

import os
import sys
import time
import argparse
import traceback
from typing import List, Dict

import chromadb

try:
    import orjson as jsonlib
except ImportError:  # fallback standard json
    import json as jsonlib

import torch
from transformers import AutoTokenizer, AutoModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='preprocessed_corpus/paragraphs.jsonl', help='Path to paragraphs JSONL')
    p.add_argument('--db-path', default='./caramella_vector_db', help='Persistent Chroma directory')
    p.add_argument('--collection', default='caramella_paragraphs', help='Collection name')
    p.add_argument('--batch-size', type=int, default=128, help='Embedding batch size')
    p.add_argument('--progress-interval', type=int, default=5000, help='Print progress after this many new vectors')
    p.add_argument('--limit', type=int, default=0, help='Optional cap on number of lines ingested')
    p.add_argument('--resume', action='store_true', help='Skip ids already logged in processed_ids.log')
    p.add_argument('--max-errors', type=int, default=20, help='Abort if errors exceed threshold')
    return p.parse_args()


class E5Embedding:
    def __init__(self, model_name: str = 'intfloat/multilingual-e5-base', batch_size: int = 128):
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    @torch.no_grad()
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        out_vectors: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
            enc = {k: v.to(self.device) for k,v in enc.items()}
            model_out = self.model(**enc)
            # mean pool last hidden state with attention mask
            attn = enc['attention_mask'].unsqueeze(-1)
            hidden = model_out.last_hidden_state * attn
            summed = hidden.sum(dim=1)
            counts = attn.sum(dim=1)
            embeddings = summed / counts
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            out_vectors.extend(embeddings.cpu().tolist())
        return out_vectors


def load_processed_ids(log_path: str) -> set:
    if not os.path.exists(log_path):
        return set()
    with open(log_path, 'r', encoding='utf-8') as f:
        return {ln.strip() for ln in f if ln.strip()}


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.db_path, exist_ok=True)
    client = chromadb.PersistentClient(path=args.db_path)
    collection = client.get_or_create_collection(name=args.collection, metadata={"hnsw:space": "cosine"})

    processed_log = os.path.join(args.db_path, 'processed_ids.log')
    processed_ids = load_processed_ids(processed_log) if args.resume else set()

    embedder = E5Embedding(batch_size=args.batch_size)
    print(f"Embedding model loaded (HF transformers). batch_size={args.batch_size} device={embedder.device}")

    added = 0
    errors = 0
    batch_texts: List[str] = []
    batch_ids: List[str] = []
    batch_metas: List[Dict] = []
    last_progress = 0
    t0 = time.time()

    def sanitize_meta(d: Dict) -> Dict:
        for k,v in list(d.items()):
            if isinstance(v, (list, dict)):
                try:
                    d[k] = jsonlib.dumps(v, ensure_ascii=False)
                except Exception:
                    d[k] = str(v)
        return d

    def flush():
        nonlocal added, batch_texts, batch_ids, batch_metas, last_progress
        if not batch_texts:
            return
        try:
            embeddings = embedder.embed_batch(batch_texts)
            collection.add(documents=batch_texts, ids=batch_ids, metadatas=batch_metas, embeddings=embeddings)
            added += len(batch_texts)
        except Exception as e:
            print(f"[flush-error] {e}")
        batch_texts = []
        batch_ids = []
        batch_metas = []
        if added - last_progress >= args.progress_interval:
            last_progress = added
            print(f"[progress] total_added={added} elapsed={time.time()-t0:.1f}s collection_count={collection.count()}")

    with open(args.input, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            if args.limit and line_idx >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = jsonlib.loads(line)
            except Exception:
                errors += 1
                if errors > args.max_errors:
                    print("Too many JSON parse errors; aborting.")
                    break
                continue
            pid = obj.get('id') or f'line-{line_idx}'
            if pid in processed_ids:
                continue
            text = obj.get('text', '').strip()
            if not text:
                continue
            meta = obj.get('metadata', {})
            meta['corpus_level'] = 'paragraph'
            meta['ingest_source'] = 'preprocessed_corpus'
            meta = sanitize_meta(meta)
            batch_texts.append(text)
            batch_ids.append(pid)
            batch_metas.append(meta)
            processed_ids.add(pid)
            # log immediately for resume safety
            with open(processed_log, 'a', encoding='utf-8') as lg:
                lg.write(pid + '\n')
            if len(batch_texts) >= args.batch_size:
                flush()
            if added and added % 2000 == 0 and len(batch_texts) == 0:
                print(f"[progress] added={added} line_idx={line_idx} elapsed={time.time()-t0:.1f}s")
    flush()

    elapsed = time.time() - t0
    count = collection.count()
    print(f"Done. vectors_in_collection={count} added_this_run={added} errors={errors} elapsed={elapsed:.1f}s")
    print("You can now point your RAG runtime to --db-path and --collection while this grows.")


if __name__ == '__main__':
    main()
