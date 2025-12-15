#!/usr/bin/env python3
"""
Ingest documents into CaramellaVectordb with proper E5 embeddings.
Uses 768-dim sentence-transformers model to match existing collection.
"""

import json
import sys
from pathlib import Path
import chromadb
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import time

class SentenceTransformerEmbedding:
    """768-dim sentence transformer embeddings."""
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2", batch_size=64):
        self.model_name = model_name
        self.batch_size = batch_size
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
    
    def embed_batch(self, texts):
        """Embed a batch of texts."""
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings.tolist()

def process_jsonl(input_file, embedder, collection, batch_size=64):
    """Process JSONL file and add to collection."""
    
    processed = 0
    errors = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        batch_texts = []
        batch_ids = []
        batch_metas = []
        
        for line in f:
            try:
                doc = json.loads(line.strip())
                doc_id = doc.get('id')
                text = doc.get('text', '')
                metadata = doc.get('metadata', {})
                
                # Ensure passage prefix
                if not text.startswith('passage: '):
                    text = 'passage: ' + text
                
                batch_texts.append(text)
                batch_ids.append(doc_id)
                batch_metas.append(metadata)
                
                # Process batch
                if len(batch_texts) >= batch_size:
                    embeddings = embedder.embed_batch(batch_texts)
                    collection.add(
                        documents=batch_texts,
                        ids=batch_ids,
                        metadatas=batch_metas,
                        embeddings=embeddings
                    )
                    processed += len(batch_texts)
                    batch_texts = []
                    batch_ids = []
                    batch_metas = []
                    
                    if processed % 100 == 0:
                        print(f"[progress] total_processed={processed}", file=sys.stderr)
                        
            except Exception as e:
                print(f"Error on line: {e}", file=sys.stderr)
                errors += 1
        
        # Process remaining batch
        if batch_texts:
            embeddings = embedder.embed_batch(batch_texts)
            collection.add(
                documents=batch_texts,
                ids=batch_ids,
                metadatas=batch_metas,
                embeddings=embeddings
            )
            processed += len(batch_texts)
    
    return processed, errors

def main():
    if len(sys.argv) < 2:
        print("Usage: python ingest_768.py <input.jsonl>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    batch_size = 64
    db_path = "caramella_vector_db"
    
    # Initialize embedder (768-dim)
    print("Loading 768-dim embeddings...", file=sys.stderr)
    embedder = SentenceTransformerEmbedding(batch_size=batch_size)
    
    # Connect to ChromaDB
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection("caramella_paragraphs")
    
    initial_count = collection.count()
    print(f"Initial collection size: {initial_count}", file=sys.stderr)
    
    # Process file
    start = time.time()
    processed, errors = process_jsonl(input_file, embedder, collection, batch_size)
    elapsed = time.time() - start
    
    final_count = collection.count()
    added = final_count - initial_count
    
    print(f"Done. vectors_in_collection={final_count} added_this_run={added} errors={errors} elapsed={elapsed:.1f}s")

if __name__ == '__main__':
    main()
