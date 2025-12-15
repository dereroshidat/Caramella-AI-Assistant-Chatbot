#!/usr/bin/env python3
"""
Offline preprocessing pipeline that DOES NOT touch ChromaDB.
- Reads existing JSONL sources (docs/images/audio)
- Cleans, deduplicates, classifies (taxonomy-based tags)
- Produces three outputs: paragraphs, documents, qa_pairs

Inputs (default):
  - ../retrieval_ready_rechunked.jsonl         (docs/images, longer chunks)
  - ./retrieval_ready_audio.jsonl              (audio transcripts)
  - ./NewInferenceextracted_images/Newretrieval_ready.jsonl (extractions)
  - ./domain_taxonomy.json                     (taxonomy for tagging)

Outputs (under ./preprocessed_corpus):
  - paragraphs.jsonl
  - documents.jsonl
  - qa_pairs.jsonl

Usage examples:
  python preprocess_pipeline.py --limit 500
  python preprocess_pipeline.py --all
"""
import orjson
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, Iterable, List, Tuple
import argparse
import re

from text_utils import safe_clean, normalized_hash, split_paragraphs

BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "preprocessed_corpus"

# Default known sources (kept for compatibility). Additional sources can be added
# via --source-dir. We also auto-include the company's AllJsons directory if present.
DEFAULT_SOURCES = [
    {"path": BASE.parent / "retrieval_ready_rechunked.jsonl", "prefix": "doc", "desc": "Rechunked PDFs"},
    {"path": BASE / "retrieval_ready_audio.jsonl", "prefix": "audio", "desc": "Audio transcripts"},
    {"path": BASE / "NewInferenceextracted_images" / "Newretrieval_ready.jsonl", "prefix": "img", "desc": "Image/PDF extractions"},
]

# Well-known external dataset directory to auto-discover
COMPANY_ALLJSONS = Path("/mnt/d/Roshidat_Msc_Project/AI_Project/Company dataset/Mytrainingdataset/AllJsons")
TRANSCRIPTS_DIR = BASE.parent / "audio_outputs" / "audio_outputs" / "transcripts"

TAXONOMY_PATH = BASE / "domain_taxonomy.json"


def load_taxonomy(path: Path) -> Dict[str, List[str]]:
    """Load taxonomy and normalize to {category: [keywords,...]} mapping.
    Accepts raw mapping or objects with a top-level 'categories' list of
    items like {"name": str, "keywords": [str,...]} or {"terms": [...]}.
    """
    if not path.exists():
        return {}
    try:
        import json
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return {}

    # Already in desired form
    if isinstance(raw, dict) and all(isinstance(v, list) for v in raw.values()):
        return {k: [t for t in v if isinstance(t, str) and t.strip()] for k, v in raw.items()}

    # Try to extract from common schema with 'categories'
    if isinstance(raw, dict) and isinstance(raw.get("categories"), list):
        out: Dict[str, List[str]] = {}
        for item in raw["categories"]:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("id") or item.get("label")
            terms = item.get("keywords") or item.get("terms") or []
            if name and isinstance(terms, list):
                out[str(name)] = [t for t in terms if isinstance(t, str) and t.strip()]
        return out

    return {}


def classify_text(text: str, taxonomy: Dict[str, List[str]]) -> List[str]:
    # Simple keyword membership per category; case-insensitive for latin; best-effort for CJK
    tags: List[str] = []
    if not taxonomy:
        return tags
    t = text.lower()
    for category, keywords in taxonomy.items():
        if not isinstance(keywords, list):
            continue
        for kw in keywords:
            if isinstance(kw, str) and kw:
                k = kw.lower()
                if k in t:
                    tags.append(category)
                    break
    return tags


def iter_records(path: Path, limit: int = 0) -> Iterable[Dict[str, Any]]:
    """Iterate records from .jsonl or .json files.
    - .jsonl: one JSON per line
    - .json: list of objects, or single object, or dict with 'data' list
    """
    if not path.exists():
        return
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                if limit and i > limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = orjson.loads(line)
                    if isinstance(obj, dict):
                        yield obj
                except Exception:
                    continue
    elif suffix == ".json":
        try:
            import json
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            return
        if isinstance(obj, list):
            for i, item in enumerate(obj, 1):
                if limit and i > limit:
                    break
                if isinstance(item, dict):
                    yield item
        elif isinstance(obj, dict):
            data = obj.get("data")
            if isinstance(data, list):
                for i, item in enumerate(data, 1):
                    if limit and i > limit:
                        break
                    if isinstance(item, dict):
                        yield item
            else:
                # Single object
                yield obj
    else:
        return


def build_documents(sources: List[Dict[str, Any]], taxonomy: Dict[str, List[str]], limit: int = 0) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    # Aggregate chunks by file_id (when available) to create document-level items
    grouped: Dict[str, Dict[str, Any]] = {}
    stats = defaultdict(int)

    for src in sources:
        path = src["path"]
        prefix = src["prefix"]
        for obj in iter_records(path, limit):
            stats[f"read_{prefix}"] += 1
            
            # Extract text based on schema
            text = None
            file_id = obj.get("id")
            
            # Standard retrieval_ready schema
            if "text" in obj:
                text = obj.get("text", "")
            # Company dataset schema (AllJsons)
            elif "consulting_content" in obj:
                text = obj.get("consulting_content", "")
                file_id = f"{obj.get('source_id', 'unknown')}_{obj.get('consulting_date', '')}"
                # Also collect instruction data if present
                instructions = obj.get("instructions", [])
                if isinstance(instructions, list) and len(instructions) > 0:
                    for inst_group in instructions:
                        if isinstance(inst_group, dict) and "data" in inst_group:
                            for inst in inst_group.get("data", []):
                                if isinstance(inst, dict):
                                    inp = inst.get("input", "")
                                    out = inst.get("output", "")
                                    if inp or out:
                                        text += f"\n\nInstruction: {inst.get('instruction', '')}\nInput: {inp}\nOutput: {out}"
            
            if not text or not text.strip():
                continue
            
            text = text if text.startswith("passage:") else f"passage: {text}"
            cleaned = safe_clean(text)
            
            if not file_id:
                file_id = (obj.get("metadata", {}) or {}).get("file_id")
            if not file_id:
                file_id = str(hash(text[:100]))
            
            key = f"{prefix}:{file_id}"
            if key not in grouped:
                grouped[key] = {
                    "id": f"doc::{key}",
                    "text": cleaned,
                    "metadata": {
                        "source_type": prefix,
                        "file_id": file_id,
                        "tags": [],
                        "source": obj.get("source", ""),
                        "category": obj.get("consulting_category", obj.get("category", "")),
                    },
                }
            else:
                grouped[key]["text"] += "\n\n" + cleaned

    # Tagging and prepare output
    documents = []
    seen_hashes = set()
    for item in grouped.values():
        tags = classify_text(item["text"], taxonomy)
        item["metadata"]["tags"] = tags
        h = normalized_hash(item["text"])
        if h in seen_hashes:
            stats["doc_duplicates"] += 1
            continue
        seen_hashes.add(h)
        documents.append(item)

    stats["documents_built"] = len(documents)
    return documents, stats


def build_paragraphs(documents: List[Dict[str, Any]], taxonomy: Dict[str, List[str]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    stats = defaultdict(int)
    out: List[Dict[str, Any]] = []
    seen = set()

    for doc in documents:
        doc_id = doc["id"]
        source_type = doc["metadata"].get("source_type", "unknown")
        for idx, para in enumerate(split_paragraphs(doc["text"])):
            para_clean = safe_clean(para)
            h = normalized_hash(para_clean)
            if h in seen:
                stats["para_duplicates"] += 1
                continue
            seen.add(h)
            out.append({
                "id": f"para::{doc_id}::p{idx}",
                "text": para_clean if para_clean.startswith("passage:") else f"passage: {para_clean}",
                "metadata": {"source_type": source_type, "parent": doc_id, "tags": classify_text(para_clean, taxonomy)},
            })

    stats["paragraphs_built"] = len(out)
    return out, stats


_QA_USER_RE = re.compile(r"^(?:user|customer|고객)\s*[:：]\s*", re.I)
_QA_AGENT_RE = re.compile(r"^(?:agent|assistant|상담원)\s*[:：]\s*", re.I)


def build_qa_pairs(sources: List[Dict[str, Any]], taxonomy: Dict[str, List[str]], limit: int = 0) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    # Heuristic turn-based QA extraction for audio transcripts
    stats = defaultdict(int)
    out: List[Dict[str, Any]] = []
    seen = set()

    def _pairs_from_text(text: str) -> List[Tuple[str, str]]:
        """Extract QA pairs from conversation text using sentence-level patterns.
        Treats sentences with '?' as questions, following sentences as answers."""
        # Split into sentences (rough Korean/English sentence splitting)
        sentences = []
        for chunk in text.split('.'):
            chunk = chunk.strip()
            if chunk:
                # Further split on Korean sentence endings and question marks
                for part in chunk.replace('?', '?|').replace('!', '!|').split('|'):
                    part = part.strip()
                    if part and len(part) > 3:
                        sentences.append(part)
        
        pairs: List[Tuple[str, str]] = []
        i = 0
        while i < len(sentences):
            sent = sentences[i]
            # Check if this is a question
            if '?' in sent or sent.endswith('요') and len(sent) < 100:
                question = sent
                # Collect answer from next few sentences until we hit another question
                answer_parts = []
                j = i + 1
                while j < len(sentences) and j < i + 5:  # Look ahead max 4 sentences
                    next_sent = sentences[j]
                    if '?' in next_sent:  # Stop at next question
                        break
                    answer_parts.append(next_sent)
                    j += 1
                
                if answer_parts:
                    answer = ' '.join(answer_parts)
                    # Validate minimum lengths
                    if len(question) >= 10 and len(answer) >= 15:
                        pairs.append((question, answer))
                    i = j  # Skip processed sentences
                    continue
            i += 1
        
        return pairs

    def _pairs_from_segments(segments: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
        # Concatenate all segments into a single text, then extract turn-based QA
        # This avoids fragmentation issues from short segments
        if not segments:
            return []
        full_text = " ".join(str(seg.get("text", "")).strip() for seg in segments if seg.get("text", "").strip())
        if not full_text:
            return []
        # Use existing text-based extraction which handles longer context better
        return _pairs_from_text(full_text)

    for src in sources:
        if src["prefix"] != "audio":
            continue
        for obj in iter_records(src["path"], limit):
            text = obj.get("text", "")
            segments = obj.get("segments") if isinstance(obj, dict) else None
            pairs: List[Tuple[str, str]] = []
            if segments and isinstance(segments, list) and segments:
                pairs = _pairs_from_segments(segments)
                stats["segment_sources"] += 1
            elif text:
                pairs = _pairs_from_text(text)
                stats["line_sources"] += 1
            else:
                continue
            for i, (q, a) in enumerate(pairs):
                q_clean = safe_clean(q)
                a_clean = safe_clean(a)
                # Filter out pairs that are too short for meaningful retrieval
                # Minimum 20 chars for question, 30 chars for answer
                if len(q_clean) < 20 or len(a_clean) < 30:
                    stats["too_short_skipped"] += 1
                    continue
                text_pair = f"Q: {q_clean}\nA: {a_clean}"
                h = normalized_hash(text_pair)
                if h in seen:
                    stats["qa_duplicates"] += 1
                    continue
                seen.add(h)
                out.append({
                    "id": f"qa::{obj.get('file', obj.get('id','unknown'))}::pair{i}",
                    "text": f"question: {q_clean}\nanswer: {a_clean}",
                    "metadata": {"source_type": "audio", "tags": classify_text(text_pair, taxonomy)},
                })

    stats["qa_pairs_built"] = len(out)
    return out, stats


def write_jsonl(path: Path, items: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(orjson.dumps(obj).decode("utf-8") + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="Limit lines per source for dry-run")
    ap.add_argument("--all", action="store_true", help="Ignore limit and process all lines")
    ap.add_argument("--source-dir", type=str, default="", help="Additional directory with JSON/JSONL files to include")
    ap.add_argument("--source-prefix", type=str, default="company", help="Prefix for records from --source-dir")
    ap.add_argument("--include-transcripts", action="store_true", help="Include raw audio transcript JSON files for segment-based QA extraction")
    args = ap.parse_args()

    limit = 0 if args.all else args.limit

    print("Loading taxonomy...")
    taxonomy = load_taxonomy(TAXONOMY_PATH)
    print(f"Taxonomy categories: {list(taxonomy.keys()) if taxonomy else 'None'}")

    # Build source list
    sources: List[Dict[str, Any]] = []
    # Auto-include the company AllJsons directory if present
    if COMPANY_ALLJSONS.exists() and COMPANY_ALLJSONS.is_dir():
        for p in sorted(COMPANY_ALLJSONS.rglob("*")):
            if p.suffix.lower() in (".jsonl", ".json") and p.is_file():
                sources.append({"path": p, "prefix": "company", "desc": "Company AllJsons"})

    # Optional inclusion of raw transcript JSONs (many files) – gated by flag
    if args.include_transcripts and TRANSCRIPTS_DIR.exists() and TRANSCRIPTS_DIR.is_dir():
        added = 0
        for p in sorted(TRANSCRIPTS_DIR.glob("*.json")):
            sources.append({"path": p, "prefix": "audio", "desc": "Raw audio transcript"})
            added += 1
        print(f"Included {added} transcript JSON files for segment QA extraction.")

    # Include user-provided source directory
    if args.source_dir:
        extra_dir = Path(args.source_dir)
        if extra_dir.exists() and extra_dir.is_dir():
            for p in sorted(extra_dir.rglob("*")):
                if p.suffix.lower() in (".jsonl", ".json") and p.is_file():
                    sources.append({"path": p, "prefix": args.source_prefix, "desc": f"Extra {args.source_prefix}"})

    # Fallback to defaults if nothing else found
    if not sources:
        sources = DEFAULT_SOURCES

    print("\nSources included:")
    for s in sources[:10]:
        print(f" - [{s['prefix']}] {s['path']}")
    if len(sources) > 10:
        print(f" ... and {len(sources)-10} more")

    print("\nBuilding document-level items...")
    documents, s1 = build_documents(sources, taxonomy, limit)
    print(f" - Read counts: { {k:v for k,v in s1.items() if k.startswith('read_')} }")
    print(f" - Duplicates skipped (docs): {s1.get('doc_duplicates',0)}")
    print(f" - Documents built: {s1.get('documents_built',0)}")

    print("\nBuilding paragraph-level items...")
    paragraphs, s2 = build_paragraphs(documents, taxonomy)
    print(f" - Duplicates skipped (paras): {s2.get('para_duplicates',0)}")
    print(f" - Paragraphs built: {s2.get('paragraphs_built',0)}")

    print("\nBuilding QA pairs from audio...")
    qa_pairs, s3 = build_qa_pairs(sources, taxonomy, limit)
    print(f" - Duplicates skipped (qa): {s3.get('qa_duplicates',0)}")
    print(f" - QA pairs built: {s3.get('qa_pairs_built',0)}")

    # Write outputs
    print("\nWriting outputs (preprocessed_corpus)...")
    write_jsonl(OUT_DIR / "documents.jsonl", documents)
    write_jsonl(OUT_DIR / "paragraphs.jsonl", paragraphs)
    write_jsonl(OUT_DIR / "qa_pairs.jsonl", qa_pairs)

    print("\n✅ Preprocessing complete (no DB changes).")
    print(f"   documents:  {len(documents):,} -> {OUT_DIR/'documents.jsonl'}")
    print(f"   paragraphs: {len(paragraphs):,} -> {OUT_DIR/'paragraphs.jsonl'}")
    print(f"   qa_pairs:   {len(qa_pairs):,} -> {OUT_DIR/'qa_pairs.jsonl'}")


if __name__ == "__main__":
    main()
