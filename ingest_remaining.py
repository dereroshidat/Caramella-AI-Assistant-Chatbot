#!/usr/bin/env python3
"""
Ingest remaining datasets with correct 768-dim embeddings.
- Audio transcripts (33,542)
- AllJsons conversations (121,508)
"""

import json
import sys
import os
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import time

class Embedder:
    """768-dim sentence transformer embeddings."""
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2", batch_size=64):
        self.model_name = model_name
        self.batch_size = batch_size
        print(f"Loading {model_name}...", file=sys.stderr)
        self.model = SentenceTransformer(model_name)
        self.device = 'cuda' if next(self.model.parameters()).is_cuda else 'cpu'
        print(f"Model loaded on {self.device}", file=sys.stderr)
    
    def embed_batch(self, texts):
        """Embed a batch of texts."""
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings.tolist()

def ensure_passage_prefix(text):
    """Add 'passage: ' prefix if not present."""
    if not text.startswith('passage: '):
        return 'passage: ' + text
    return text

def process_audio_jsonl(file_path, embedder, collection, batch_size=64):
    """Process audio JSONL file."""
    print(f"\n📚 Processing audio transcripts: {file_path}", file=sys.stderr)
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}", file=sys.stderr)
        return 0, 0
    
    processed = 0
    errors = 0
    batch_texts = []
    batch_ids = []
    batch_metas = []
    
    # Count lines
    with open(file_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, total=total_lines, desc="  audio")):
            try:
                doc = json.loads(line.strip())
                doc_id = doc.get('id')
                text = doc.get('text', '')
                metadata = doc.get('metadata', {})
                
                if not doc_id or not text:
                    errors += 1
                    continue
                
                metadata['dataset'] = 'audio_transcripts'
                text = ensure_passage_prefix(text)
                
                batch_texts.append(text)
                batch_ids.append(doc_id)
                batch_metas.append(metadata)
                
                if len(batch_texts) >= batch_size:
                    try:
                        embeddings = embedder.embed_batch(batch_texts)
                        collection.add(
                            documents=batch_texts,
                            ids=batch_ids,
                            metadatas=batch_metas,
                            embeddings=embeddings
                        )
                        processed += len(batch_texts)
                    except Exception as e:
                        print(f"\n❌ Batch error: {e}", file=sys.stderr)
                        errors += len(batch_texts)
                    
                    batch_texts = []
                    batch_ids = []
                    batch_metas = []
                        
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"\n⚠️  Line {line_num+1}: {e}", file=sys.stderr)
        
        # Final batch
        if batch_texts:
            try:
                embeddings = embedder.embed_batch(batch_texts)
                collection.add(
                    documents=batch_texts,
                    ids=batch_ids,
                    metadatas=batch_metas,
                    embeddings=embeddings
                )
                processed += len(batch_texts)
            except Exception as e:
                print(f"\n❌ Final batch error: {e}", file=sys.stderr)
                errors += len(batch_texts)
    
    print(f"  ✅ Audio: {processed:,} docs, {errors} errors", file=sys.stderr)
    return processed, errors

def process_alljsons_directory(dir_path, embedder, collection, batch_size=64):
    """Process AllJsons directory."""
    print(f"\n📚 Processing AllJsons: {dir_path}", file=sys.stderr)
    
    if not os.path.exists(dir_path):
        print(f"❌ Directory not found: {dir_path}", file=sys.stderr)
        return 0, 0
    
    json_files = list(Path(dir_path).glob("*.json"))
    total_files = len(json_files)
    print(f"  Found {total_files:,} JSON files", file=sys.stderr)
    
    processed = 0
    errors = 0
    batch_texts = []
    batch_ids = []
    batch_metas = []
    
    for json_file in tqdm(json_files, desc="  alljsons"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            conversations = data if isinstance(data, list) else [data]
            
            for conv in conversations:
                content = conv.get('consulting_content', '')
                if not content:
                    continue
                
                source_id = conv.get('source_id', 'unknown')
                file_id = json_file.stem
                doc_id = f"alljsons::{file_id}::{source_id}"
                
                metadata = {
                    'dataset': 'alljsons',
                    'source': conv.get('source', ''),
                    'source_id': source_id,
                    'consulting_date': conv.get('consulting_date', ''),
                    'consulting_category': conv.get('consulting_category', ''),
                    'file_id': file_id
                }
                
                text = ensure_passage_prefix(content)
                
                batch_texts.append(text)
                batch_ids.append(doc_id)
                batch_metas.append(metadata)
                
                if len(batch_texts) >= batch_size:
                    try:
                        embeddings = embedder.embed_batch(batch_texts)
                        collection.add(
                            documents=batch_texts,
                            ids=batch_ids,
                            metadatas=batch_metas,
                            embeddings=embeddings
                        )
                        processed += len(batch_texts)
                    except Exception as e:
                        print(f"\n❌ Batch error: {e}", file=sys.stderr)
                        errors += len(batch_texts)
                    
                    batch_texts = []
                    batch_ids = []
                    batch_metas = []
                        
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"\n⚠️  {json_file.name}: {e}", file=sys.stderr)
    
    if batch_texts:
        try:
            embeddings = embedder.embed_batch(batch_texts)
            collection.add(
                documents=batch_texts,
                ids=batch_ids,
                metadatas=batch_metas,
                embeddings=embeddings
            )
            processed += len(batch_texts)
        except Exception as e:
            print(f"\n❌ Final batch error: {e}", file=sys.stderr)
            errors += len(batch_texts)
    
    print(f"  ✅ AllJsons: {processed:,} docs, {errors} errors", file=sys.stderr)
    return processed, errors

def main():
    print("=" * 70, file=sys.stderr)
    print("  Ingest Remaining Datasets", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    
    db_path = "/mnt/d/Roshidat_Msc_Project/AI_Project/AI_Project/CleanInferenceRAG/caramella_vector_db"
    batch_size = 64
    
    embedder = Embedder(batch_size=batch_size)
    
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection("caramella_paragraphs")
    
    initial_count = collection.count()
    print(f"\nInitial: {initial_count:,} documents", file=sys.stderr)
    
    start_time = time.time()
    total_processed = 0
    total_errors = 0
    
    # Audio transcripts
    audio_path = "/mnt/d/Roshidat_Msc_Project/AI_Project/AI_Project/CleanInferenceRAG/retrieval_ready_audio.jsonl"
    processed, errors = process_audio_jsonl(audio_path, embedder, collection, batch_size)
    total_processed += processed
    total_errors += errors
    
    # AllJsons
    alljsons_path = "/mnt/d/Roshidat_Msc_Project/AI_Project/Mytrainingdataset/AllJsons"
    processed, errors = process_alljsons_directory(alljsons_path, embedder, collection, batch_size)
    total_processed += processed
    total_errors += errors
    
    elapsed = time.time() - start_time
    final_count = collection.count()
    added = final_count - initial_count
    
    print("\n" + "=" * 70, file=sys.stderr)
    print("  ✅ COMPLETE", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print(f"  Initial:    {initial_count:,}", file=sys.stderr)
    print(f"  Final:      {final_count:,}", file=sys.stderr)
    print(f"  Added:      {added:,}", file=sys.stderr)
    print(f"  Processed:  {total_processed:,}", file=sys.stderr)
    print(f"  Errors:     {total_errors}", file=sys.stderr)
    print(f"  Time:       {elapsed/60:.1f} minutes ({elapsed/3600:.1f} hours)", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

if __name__ == '__main__':
    main()
