#!/usr/bin/env python3
"""
Comprehensive ingestion script for all datasets.
Ingests:
1. Newretieval_ready.jsonl (585 diving safety documents)
2. retrieval_ready_audio.jsonl (33,542 documents)
3. AllJsons/*.json (121,508 customer service conversations)

Uses sentence-transformers/paraphrase-multilingual-mpnet-base-v2 (768-dim)
to match the existing collection.
"""

import json
import sys
import os
from pathlib import Path
import chromadb
import torch
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
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
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

def process_jsonl_file(file_path, embedder, collection, batch_size=64, dataset_name="unknown"):
    """Process a JSONL file and add to collection."""
    print(f"\n📚 Processing {dataset_name}: {file_path}", file=sys.stderr)
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}", file=sys.stderr)
        return 0, 0
    
    processed = 0
    errors = 0
    batch_texts = []
    batch_ids = []
    batch_metas = []
    
    # Count lines for progress bar
    with open(file_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, total=total_lines, desc=f"  {dataset_name}")):
            try:
                doc = json.loads(line.strip())
                doc_id = doc.get('id')
                text = doc.get('text', '')
                metadata = doc.get('metadata', {})
                
                if not doc_id or not text:
                    errors += 1
                    continue
                
                # Add dataset source to metadata
                metadata['dataset'] = dataset_name
                
                # Ensure passage prefix
                text = ensure_passage_prefix(text)
                
                batch_texts.append(text)
                batch_ids.append(doc_id)
                batch_metas.append(metadata)
                
                # Process batch
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
                if errors <= 5:  # Only print first 5 errors
                    print(f"\n⚠️  Line {line_num+1} error: {e}", file=sys.stderr)
        
        # Process remaining batch
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
    
    print(f"  ✅ {dataset_name}: {processed:,} docs added, {errors} errors", file=sys.stderr)
    return processed, errors

def process_alljsons_directory(dir_path, embedder, collection, batch_size=64):
    """Process AllJsons directory - convert conversations to JSONL format."""
    print(f"\n📚 Processing AllJsons directory: {dir_path}", file=sys.stderr)
    
    if not os.path.exists(dir_path):
        print(f"❌ Directory not found: {dir_path}", file=sys.stderr)
        return 0, 0
    
    # Get all JSON files
    json_files = list(Path(dir_path).glob("*.json"))
    total_files = len(json_files)
    print(f"  Found {total_files:,} JSON files", file=sys.stderr)
    
    processed = 0
    errors = 0
    batch_texts = []
    batch_ids = []
    batch_metas = []
    
    for json_file in tqdm(json_files, desc="  AllJsons"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle both list and single object formats
            if isinstance(data, list):
                conversations = data
            else:
                conversations = [data]
            
            for conv in conversations:
                # Extract conversation content
                content = conv.get('consulting_content', '')
                if not content:
                    continue
                
                # Create unique ID from filename and source_id
                source_id = conv.get('source_id', 'unknown')
                file_id = json_file.stem  # filename without extension
                doc_id = f"alljsons::{file_id}::{source_id}"
                
                # Create metadata
                metadata = {
                    'dataset': 'alljsons',
                    'source': conv.get('source', ''),
                    'source_id': source_id,
                    'consulting_date': conv.get('consulting_date', ''),
                    'consulting_category': conv.get('consulting_category', ''),
                    'file_id': file_id
                }
                
                # Add passage prefix
                text = ensure_passage_prefix(content)
                
                batch_texts.append(text)
                batch_ids.append(doc_id)
                batch_metas.append(metadata)
                
                # Process batch
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
                print(f"\n⚠️  File {json_file.name} error: {e}", file=sys.stderr)
    
    # Process remaining batch
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
    
    print(f"  ✅ AllJsons: {processed:,} docs added, {errors} errors", file=sys.stderr)
    return processed, errors

def main():
    print("=" * 70, file=sys.stderr)
    print("  Comprehensive Dataset Ingestion", file=sys.stderr)
    print("  Model: sentence-transformers/paraphrase-multilingual-mpnet-base-v2", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    
    # Configuration
    db_path = "/mnt/d/Roshidat_Msc_Project/AI_Project/AI_Project/CleanInferenceRAG/caramella_vector_db"
    batch_size = 64
    
    datasets = [
        ("Newretieval_ready_with_prefix.jsonl", "diving_safety"),
        ("/mnt/d/Roshidat_Msc_Project/AI_Project/AI_Project/CleanInferenceRAG/retrieval_ready_audio.jsonl", "audio_transcripts"),
    ]
    
    # Initialize embedder
    embedder = Embedder(batch_size=batch_size)
    
    # Connect to ChromaDB
    print(f"\n🗄️  Connecting to ChromaDB: {db_path}", file=sys.stderr)
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection("caramella_paragraphs")
    
    initial_count = collection.count()
    print(f"  Initial collection size: {initial_count:,} documents", file=sys.stderr)
    
    # Process datasets
    start_time = time.time()
    total_processed = 0
    total_errors = 0
    
    # Process JSONL files
    for file_path, dataset_name in datasets:
        processed, errors = process_jsonl_file(file_path, embedder, collection, batch_size, dataset_name)
        total_processed += processed
        total_errors += errors
    
    # Process AllJsons directory
    alljsons_path = "/mnt/d/Roshidat_Msc_Project/AI_Project/Mytrainingdataset/AllJsons"
    processed, errors = process_alljsons_directory(alljsons_path, embedder, collection, batch_size)
    total_processed += processed
    total_errors += errors
    
    elapsed = time.time() - start_time
    final_count = collection.count()
    added = final_count - initial_count
    
    # Summary
    print("\n" + "=" * 70, file=sys.stderr)
    print("  ✅ INGESTION COMPLETE", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print(f"  Initial documents:  {initial_count:,}", file=sys.stderr)
    print(f"  Final documents:    {final_count:,}", file=sys.stderr)
    print(f"  Added this run:     {added:,}", file=sys.stderr)
    print(f"  Processed:          {total_processed:,}", file=sys.stderr)
    print(f"  Errors:             {total_errors}", file=sys.stderr)
    print(f"  Elapsed time:       {elapsed/60:.1f} minutes", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

if __name__ == '__main__':
    main()
