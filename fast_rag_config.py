#!/usr/bin/env python3


import os


class FastRAGConfig:
  
    # VECTOR DATABASE SETTINGS
    

    DB_PATH = os.getenv(
        "RAG_DB_PATH",
        "/mnt/d/Roshidat_Msc_Project/AI_Project/AI_Project/CleanInferenceRAG/caramella_vector_db"
    )

    COLLECTION_NAME = os.getenv(
        "RAG_COLLECTION",
        "caramella_paragraphs"
    )

    # EMBEDDING MODEL (optimized for speed)

    # MUST match what was used during ingestion (768-dim model)
    EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    EMBED_DEVICE = "cpu"             
    EMBED_BATCH_SIZE = 1

    # Retrieval parameters
    TOP_K = 4                         
    INITIAL_RETRIEVAL_K = TOP_K
    USE_RERANKING = False             
    MIN_SIMILARITY = 0.5

 
    # CONTEXT SIZE (CRITICAL FOR SPEED)
    

    MAX_CONTEXT_DOCS = 3              # allow one more snippet for completeness
    MAX_CONTEXT_CHARS = 600           # allow a bit more text per doc

    USE_MULTI_COLLECTION = False

   
    # GENERATION SETTINGS
    

    LLM_MODEL_PATH = os.getenv(
        "RAG_MODEL_PATH",
        r"/mnt/d/Roshidat_Msc_Project/AI_Project/AI_Project/CleanInferenceRAG/models/qwen2.5-1.5b-instruct-q5_k_m.gguf"
    )

    LLM_CONTEXT_SIZE = 2048

    # Performance tuning for Qwen2.5-1.5B Q5_K_M on 4GB edge device
    LLM_THREADS = int(os.getenv("RAG_THREADS", "8"))  
    LLM_GPU_LAYERS = int(os.getenv("RAG_GPU_LAYERS", "28"))  
    LLM_BATCH_SIZE = 512  

    # Memory mapping
    LLM_USE_MLOCK = True
    LLM_USE_MMAP = True

    # Generation parameters - Optimized for Qwen2.5-1.5B Q5_K_M quality + speed balance
    MAX_TOKENS = int(os.getenv("FAST_MAX_TOKENS", "80"))  
    TEMPERATURE = float(os.getenv("FAST_TEMPERATURE", "0.2"))  
    TOP_P = 0.80  
    REPEAT_PENALTY = 1.1  

 
    # TRANSLATION DISABLED FOR SPEED
 

    TRANSLATION_ENABLED = False
    TRANSLATE_CONTEXT_TO_ENGLISH = False
    TRANSLATE_ANSWER_TO_QUERY_LANG = False
    TRANSLATION_MAX_CHARS = 2000
    TRANSLATION_MODEL_KO_EN = "Helsinki-NLP/opus-mt-ko-en"
    TRANSLATION_MODEL_EN_KO = "Helsinki-NLP/opus-mt-en-ko"

  
    # PERFORMANCE SETTINGS


    REQUEST_TIMEOUT = 10.0            
    ENABLE_CACHING = True
    CACHE_TTL = 3600

    # Logging
    LOG_LATENCY = True
    WARN_THRESHOLD_MS = 8000          

    # Source annotation
    DISABLE_SOURCE_ANNOTATION = False 

    
    # FALLBACK STRATEGY
  

    ENABLE_FALLBACK = True
    FALLBACK_ANSWER = (
        "I could not retrieve the information quickly enough. "
        "Please try rephrasing your question."
    )

    # CONFIG SUMMARY


    @classmethod
    def get_config_summary(cls) -> dict:
        return {
            "retrieval": {
                "top_k": cls.TOP_K,
                "reranking": cls.USE_RERANKING,
                "multi_collection": cls.USE_MULTI_COLLECTION,
                "min_similarity": cls.MIN_SIMILARITY,
                "max_context_docs": cls.MAX_CONTEXT_DOCS,
                "max_context_chars": cls.MAX_CONTEXT_CHARS,
            },
            "generation": {
                "max_tokens": cls.MAX_TOKENS,
                "temperature": cls.TEMPERATURE,
                "top_p": cls.TOP_P,
                "threads": cls.LLM_THREADS,
                "gpu_layers": cls.LLM_GPU_LAYERS,
                "batch_size": cls.LLM_BATCH_SIZE,
            },
            "performance": {
                "target_latency_ms": 3000,
                "timeout_ms": cls.REQUEST_TIMEOUT * 1000,
                "caching": cls.ENABLE_CACHING,
            },
            "translation": {
                "enabled": cls.TRANSLATION_ENABLED,
                "translate_context_to_english": cls.TRANSLATE_CONTEXT_TO_ENGLISH,
                "translate_answer_to_query_lang": cls.TRANSLATE_ANSWER_TO_QUERY_LANG,
            },
        }

    @classmethod
    def print_config(cls):
        import json
        print("=" * 70)
        print("FAST RAG CONFIGURATION (1–3 second target latency)")
        print("=" * 70)
        print(json.dumps(cls.get_config_summary(), indent=2))
        print("=" * 70)


# DEPLOYMENT PROFILES


class DeploymentProfiles:
    """Preset profiles for deployment environments."""

    @staticmethod
    def ultra_fast():
        """Sub-1.5 second mode (smallest context)."""
        os.environ["FAST_MAX_TOKENS"] = "50"
        os.environ["FAST_TEMPERATURE"] = "0.4"
        os.environ["RAG_THREADS"] = "8"

    @staticmethod
    def balanced():
        """1–3 second mode (recommended)."""
        os.environ["FAST_MAX_TOKENS"] = "80"
        os.environ["FAST_TEMPERATURE"] = "0.3"
        os.environ["RAG_THREADS"] = "6"

    @staticmethod
    def quality():
        """High quality, <= 5 seconds latency."""
        os.environ["FAST_MAX_TOKENS"] = "120"
        os.environ["FAST_TEMPERATURE"] = "0.25"
        os.environ["RAG_THREADS"] = "6"

    @staticmethod
    def gpu_optimized():
        """GPU-accelerated mode."""
        os.environ["FAST_MAX_TOKENS"] = "80"
        os.environ["FAST_TEMPERATURE"] = "0.3"
        os.environ["RAG_THREADS"] = "4"
        os.environ["RAG_GPU_LAYERS"] = "32"
