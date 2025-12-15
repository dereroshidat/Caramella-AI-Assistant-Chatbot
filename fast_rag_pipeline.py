#!/usr/bin/env python3


import time
import re
import psutil
import os
from typing import List, Dict, Tuple
from functools import lru_cache

try:
    from langdetect import detect
except Exception:
    detect = None

import chromadb
import torch
from transformers import AutoTokenizer, AutoModel
from llama_cpp import Llama

from fast_rag_config import FastRAGConfig



# Language detection

def detect_language(text: str) -> str:
    """Return 'ko' for Korean, else 'en'."""
    if not text:
        return "en"

    # Try langdetect first
    if detect:
        try:
            lang = detect(text)
            if lang.startswith("ko"):
                return "ko"
            return "en"
        except Exception:
            pass

    # Regex fallback
    if re.search(r"[가-힣]", text):
        return "ko"

    return "en"


# Performance logging

class PerformanceMetrics:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.total_queries = 0
        self.total_latency_ms = 0.0

    def before(self, query: str, prompt_tokens: int):
        m = self.process.memory_info()
        return {
            "memory_mb": m.rss / 1024 / 1024,
            "prompt_tokens": prompt_tokens,
        }

    def after(self, answer: str, gen_time: float, answer_tokens: int):
        m = self.process.memory_info()
        return {
            "memory_mb": m.rss / 1024 / 1024,
            "generation_ms": gen_time * 1000,
            "answer_tokens": answer_tokens,
            "tps": answer_tokens / gen_time if gen_time > 0 else 0
        }

    def log(self, before: dict, after: dict):
        print("\n⏱️ Performance Metrics")
        print(f"   Memory before: {before['memory_mb']:.1f} MB")
        print(f"   Memory after:  {after['memory_mb']:.1f} MB")
        print(f"   Prompt tokens: {before['prompt_tokens']}")
        print(f"   Answer tokens: {after['answer_tokens']}")
        print(f"   Generation:    {after['generation_ms']:.0f} ms")
        print(f"   Throughput:    {after['tps']:.1f} tokens/sec")



# Main RAG Pipeline

class FastRAGPipeline:
    def __init__(self, db_path=None, collection_name=None, model_path=None,
                 config=None, verbose=False):

        self.config = config or FastRAGConfig()
        self.verbose = verbose
        self.metrics = PerformanceMetrics()

        db_path = db_path or self.config.DB_PATH
        collection_name = collection_name or self.config.COLLECTION_NAME
        model_path = model_path or self.config.LLM_MODEL_PATH

        print("🚀 Initializing Fast RAG Pipeline...")
        if verbose:
            self.config.print_config()

        self._init_vector_db(db_path, collection_name)
        self._init_embeddings()
        self._init_llm(model_path)

        print("✅ Fast RAG Pipeline Ready!\n")

    
    # Vector DB
    
    def _init_vector_db(self, db_path, collection_name):
        start = time.time()
        print(f"📚 Loading ChromaDB: {db_path}/{collection_name}")

        self.chroma = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma.get_collection(collection_name)

        count = self.collection.count()
        print(f"   ✅ {count:,} documents loaded in {(time.time() - start) * 1000:.0f} ms")


    # Embeddings
   
    def _init_embeddings(self):
        start = time.time()
        print(f"🔤 Loading embeddings: {self.config.EMBED_MODEL}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.EMBED_MODEL)
        self.embed_model = AutoModel.from_pretrained(self.config.EMBED_MODEL)

        self.device = torch.device(self.config.EMBED_DEVICE)
        self.embed_model.to(self.device)
        self.embed_model.eval()

        for p in self.embed_model.parameters():
            p.requires_grad = False

        print(f"   ✅ Ready on {self.device} in {(time.time() - start) * 1000:.0f} ms")

   
    # LLM
 
    def _init_llm(self, model_path):
        start = time.time()
        print(f"🤖 Loading LLM: {model_path}")

        self.llm = Llama(
            model_path=model_path,
            n_ctx=self.config.LLM_CONTEXT_SIZE,
            n_threads=self.config.LLM_THREADS,
            n_gpu_layers=self.config.LLM_GPU_LAYERS,
            n_batch=self.config.LLM_BATCH_SIZE,
            use_mlock=self.config.LLM_USE_MLOCK,
            use_mmap=self.config.LLM_USE_MMAP,
            verbose=False
        )

        print(f"   ✅ Model loaded in {(time.time() - start) * 1000:.0f} ms")

    # Embedding
   
    @torch.no_grad()
    @lru_cache(maxsize=1000)
    def embed_query(self, q: str) -> tuple:
        if not q.startswith("query: "):
            q = f"query: {q}"

        enc = self.tokenizer([q], padding=True, truncation=True, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}

        out = self.embed_model(**enc)
        attn = enc["attention_mask"].unsqueeze(-1)
        hidden = out.last_hidden_state * attn
        summed = hidden.sum(1)
        count = attn.sum(1)
        emb = summed / count
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)

        return tuple(emb[0].cpu().tolist())

 
    # Retrieval

    def retrieve(self, query: str, top_k=None):
        start = time.time()
        top_k = top_k or self.config.TOP_K

        qemb = list(self.embed_query(query))

        res = self.collection.query(
            query_embeddings=[qemb],
            n_results=top_k
        )

        docs = []
        for d, m, dist in zip(
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0]
        ):
            score = 1 - dist
            if score >= self.config.MIN_SIMILARITY:
                docs.append({"text": d, "score": score, "meta": m})

        if not docs:
            docs = [{"text": d, "score": 0, "meta": m}
                    for d, m in zip(res["documents"][0], res["metadatas"][0])]

        return docs, (time.time() - start)

 
    # Prompt creation (BILINGUAL)

    # Prompt creation (bilingual, grounded)

    def _build_prompt(self, query: str, docs: List[Dict]) -> str:
        q_lang = detect_language(query)

        context_parts = [
            d["text"][: self.config.MAX_CONTEXT_CHARS]
            for d in docs[: self.config.MAX_CONTEXT_DOCS]
        ]
        context = "\n\n".join(context_parts).strip() or "No relevant information."

        # Balanced: Handle bilingual context (Korean docs with English questions)
        rules_en = (
            "You are a helpful assistant. Answer based ONLY on the CONTEXT below.\n\n"
            "RULES:\n"
            "- Answer in English (same language as the QUESTION)\n"
            "- Use ONLY information from CONTEXT; do NOT add external knowledge\n"
            "- CONTEXT may be in Korean or other languages - extract and translate the key facts to English\n"
            "- If CONTEXT is completely unrelated to QUESTION topic, say: Information not found\n"
            "- If CONTEXT has partial/related info, provide that answer; do NOT guess missing parts\n"
            "- If multiple questions, answer ONLY the first main question\n"
            "- Direct, factual, 2-5 sentences"
        )

        # Balanced Korean prompt - handle bilingual context
        rules_ko = (
            "당신은 유용한 도우미입니다. 아래 맥락(CONTEXT)을 바탕으로 질문에 답하세요.\n\n"
            "규칙:\n"
            "- 한국어로 답변 (질문과 같은 언어)\n"
            "- 맥락의 정보만 사용; 절대 외부 지식 추가 금지\n"
            "- 맥락이 영어나 다른 언어일 수 있음 - 핵심 내용을 추출하여 한국어로 답변\n"
            "- 맥락이 질문 주제와 완전히 무관하면: 정보를 찾을 수 없습니다\n"
            "- 맥락에 부분/관련 정보가 있으면 그것으로 답변; 빠진 부분 추측 금지\n"
            "- 여러 질문이 섞여 있으면 첫 번째 주요 질문만 답변\n"
            "- 사실적이고 명확하게, 2-5문장"
        )

        if q_lang == "ko":
            prompt = f"""{rules_ko}

맥락:
{context}

질문: {query}

답변:"""
        else:
            prompt = f"""{rules_en}

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

        # Token safety
        tokens = self.llm.tokenize(prompt.encode("utf-8"))
        max_prompt_tokens = self.config.LLM_CONTEXT_SIZE - self.config.MAX_TOKENS - 64
        if len(tokens) > max_prompt_tokens:
            short = docs[0]["text"][:150] if docs else ""
            if q_lang == "ko":
                prompt = f"""{rules_ko}

[CONTEXT]
{short}

[QUESTION]
{query}

[ANSWER]"""
            else:
                prompt = f"""{rules_en}

[CONTEXT]
{short}

[QUESTION]
{query}

[ANSWER]"""

        return prompt.strip()

  
    # Generation

    def generate(self, query: str, docs: List[Dict]) -> Tuple[str, float]:
        start = time.time()
        prompt = self._build_prompt(query, docs)
        q_lang = detect_language(query)

        prompt_tokens = len(self.llm.tokenize(prompt.encode("utf-8")))
        before = self.metrics.before(query, prompt_tokens)

        try:
            out = self.llm(
                prompt,
                max_tokens=self.config.MAX_TOKENS,
                temperature=self.config.TEMPERATURE,
                top_p=self.config.TOP_P,
                repeat_penalty=1.1,
                echo=False,
                stop=["[QUESTION]", "[CONTEXT]", "\n\n"],
            )
            raw = out["choices"][0]["text"].strip()
        except Exception:
            raw = ""

        # Basic fallback on backend error
        if not raw:
            answer = "Information not found."
            gen_time = time.time() - start
        else:
            answer = raw

            # Remove typical meta phrases anywhere in the string
            meta_phrases = [
                "context does not provide",
                "passage does not provide",
                "it mentions that",
                "it can be inferred",
                "based on the context",
                "according to the passage",
                "from the context",
                "from this context",
            ]
            lower = answer.lower()
            if any(p in lower for p in meta_phrases):
                # Try to keep the part after the first full stop
                parts = re.split(r"[.!?]", answer, maxsplit=1)
                if len(parts) > 1 and parts[1].strip():
                    answer = parts[1].strip()
                    lower = answer.lower()

            # If it still contains meta phrases or is too short, use the explicit fallback
            lower = answer.lower()
            if any(p in lower for p in meta_phrases) or len(answer) < 5:
                if q_lang == "ko":
                    answer = "정보를 찾을 수 없습니다."
                else:
                    answer = "Information not found."

            # Final cleanup
            answer = re.sub(r"\s+", " ", answer).strip()

        gen_time = time.time() - start
        answer_tokens = len(self.llm.tokenize(answer.encode("utf-8")))
        after = self.metrics.after(answer, gen_time, answer_tokens)
        if self.verbose:
            self.metrics.log(before, after)

        return answer, gen_time

    
    # Query
    
    def query(self, query: str, top_k=None, timeout=None):
        start = time.time()
        timeout = timeout or self.config.REQUEST_TIMEOUT

        docs, rt = self.retrieve(query, top_k)
        if time.time() - start > timeout:
            return {"answer": self.config.FALLBACK_ANSWER, "status": "timeout"}

        answer, gt = self.generate(query, docs)

        total_ms = (time.time() - start) * 1000
        
        # Track stats
        self.metrics.total_queries += 1
        self.metrics.total_latency_ms += total_ms

        return {
            "answer": answer,
            "sources": docs,
            "latency": {
                "retrieval_ms": round(rt * 1000, 1),
                "generation_ms": round(gt * 1000, 1),
                "total_ms": round(total_ms, 1)
            },
            "status": "success"
        }

    def get_stats(self):
        """Get pipeline statistics for health check."""
        avg_latency = self.metrics.total_latency_ms / max(self.metrics.total_queries, 1)
        return {
            "total_queries": self.metrics.total_queries,
            "avg_total_ms": avg_latency,
               "config": {"profile": self.config.__class__.__name__}
        }



# CLI

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fast RAG Pipeline")
    parser.add_argument("--query", "-q", type=str)
    parser.add_argument("--profile", choices=["ultra_fast", "balanced", "quality", "gpu"],
                        default="balanced")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    from fast_rag_config import DeploymentProfiles
    getattr(DeploymentProfiles, args.profile)()

    pipeline = FastRAGPipeline(verbose=args.verbose)

    if args.query:
        result = pipeline.query(args.query)
        print("\nAnswer:", result["answer"])
        print("Latency:", result["latency"])


if __name__ == "__main__":
    main()
