#!/usr/bin/env python3
"""
Comprehensive RAG Evaluation with ROUGE, BLEU, BERTScore
Evaluates both generation quality and retrieval performance
"""

import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
import numpy as np

# Evaluation metrics
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False

try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False


@dataclass
class EvaluationMetrics:
    """Complete evaluation metrics for a single query."""
    query: str
    reference: str
    prediction: str
    
    # Generation quality
    rouge_1_f1: float = 0.0
    rouge_2_f1: float = 0.0
    rouge_l_f1: float = 0.0
    bleu_score: float = 0.0
    bert_score_f1: float = 0.0
    token_f1: float = 0.0
    exact_match: bool = False
    
    # Performance
    latency_ms: float = 0.0
    retrieved_docs: int = 0
    
    # Retrieval (if ground truth available)
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0


class RAGEvaluator:
    """Comprehensive RAG system evaluator with multiple metrics."""
    
    def __init__(self, rag_pipeline, verbose: bool = False):
        self.pipeline = rag_pipeline
        self.verbose = verbose
        
        # Initialize scorers
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
            )
        else:
            self.rouge_scorer = None
            print("⚠️  ROUGE not available. Install: pip install rouge-score")
        
        if BLEU_AVAILABLE:
            self.bleu_smoothing = SmoothingFunction().method1
        else:
            print("⚠️  BLEU not available. Install: pip install nltk")
    
    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization for Korean/English."""
        import re
        text = text.lower().strip()
        tokens = re.findall(r'\w+', text, re.UNICODE)
        return tokens
    
    def calculate_rouge(self, reference: str, prediction: str) -> Dict:
        """Calculate ROUGE scores."""
        if not self.rouge_scorer:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        
        scores = self.rouge_scorer.score(reference, prediction)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    
    def calculate_bleu(self, reference: str, prediction: str) -> float:
        """Calculate BLEU score."""
        if not BLEU_AVAILABLE:
            return 0.0
        
        ref_tokens = [self.tokenize(reference)]
        pred_tokens = self.tokenize(prediction)
        
        if not pred_tokens or not ref_tokens[0]:
            return 0.0
        
        try:
            return sentence_bleu(ref_tokens, pred_tokens, 
                               smoothing_function=self.bleu_smoothing)
        except:
            return 0.0
    
    def calculate_bert_score(self, reference: str, prediction: str) -> float:
        """Calculate BERTScore F1."""
        if not BERTSCORE_AVAILABLE:
            return 0.0
        
        try:
            P, R, F1 = bert_score([prediction], [reference], 
                                 lang='ko', verbose=False)
            return F1.mean().item()
        except:
            return 0.0
    
    def calculate_token_f1(self, reference: str, prediction: str) -> float:
        """Calculate token-level F1."""
        ref_tokens = set(self.tokenize(reference))
        pred_tokens = set(self.tokenize(prediction))
        
        if not ref_tokens or not pred_tokens:
            return 0.0
        
        common = ref_tokens & pred_tokens
        if not common:
            return 0.0
        
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens)
        
        return 2 * precision * recall / (precision + recall)
    
    def evaluate_single(self, query: str, reference: str, **kwargs) -> EvaluationMetrics:
        """Evaluate a single query-answer pair."""
        start = time.time()
        
        try:
            # Only pass supported arguments to pipeline.query
            result = self.pipeline.query(query)
            latency = (time.time() - start) * 1000
            
            prediction = result.get('answer', '')
            sources = result.get('sources', [])
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return EvaluationMetrics(
                query=query, reference=reference, prediction="",
                latency_ms=(time.time() - start) * 1000
            )
        
        # Calculate metrics
        rouge = self.calculate_rouge(reference, prediction)
        bleu = self.calculate_bleu(reference, prediction)
        bert_f1 = self.calculate_bert_score(reference, prediction)
        token_f1 = self.calculate_token_f1(reference, prediction)
        exact = reference.lower().strip() == prediction.lower().strip()
        
        metrics = EvaluationMetrics(
            query=query,
            reference=reference,
            prediction=prediction,
            rouge_1_f1=rouge['rouge1'],
            rouge_2_f1=rouge['rouge2'],
            rouge_l_f1=rouge['rougeL'],
            bleu_score=bleu,
            bert_score_f1=bert_f1,
            token_f1=token_f1,
            exact_match=exact,
            latency_ms=latency,
            retrieved_docs=len(sources)
        )

        # Optional: retrieval metrics if ground-truth doc IDs provided
        # Expect caller to pass 'relevant_doc_ids' in kwargs when available
        gt_ids = kwargs.get('relevant_doc_ids') or []
        if gt_ids and sources:
            try:
                # Extract candidate IDs from source metadata
                cand_ids = []
                for s in sources:
                    meta = s.get('metadata', {})
                    doc_id = meta.get('doc_id') or meta.get('id') or meta.get('source_id')
                    if not doc_id:
                        # Attempt to reconstruct id from filename + chunk
                        fname = meta.get('source') or meta.get('file') or meta.get('audio_file')
                        chunk = meta.get('chunk_id') or meta.get('chunk')
                        if fname and chunk is not None:
                            doc_id = f"{fname}::chunk-{chunk}"
                    if doc_id:
                        cand_ids.append(str(doc_id))

                k = max(1, len(cand_ids))
                # Precision@K and Recall@K
                true_pos = len(set(cand_ids) & set(gt_ids))
                precision_k = true_pos / k if k else 0.0
                recall_k = true_pos / len(gt_ids) if gt_ids else 0.0

                # MRR: rank of first relevant in the candidate list
                mrr_val = 0.0
                for idx, cid in enumerate(cand_ids, start=1):
                    if cid in gt_ids:
                        mrr_val = 1.0 / idx
                        break

                metrics.precision_at_k = precision_k
                metrics.recall_at_k = recall_k
                metrics.mrr = mrr_val
            except Exception:
                pass
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Q: {query}")
            print(f"R: {reference[:80]}...")
            print(f"P: {prediction[:80]}...")
            extra = []
            if gt_ids:
                extra.append(f"P@K: {metrics.precision_at_k:.3f}")
                extra.append(f"R@K: {metrics.recall_at_k:.3f}")
                extra.append(f"MRR: {metrics.mrr:.3f}")
            extra_str = " | ".join(extra)
            print(f"ROUGE-L: {rouge['rougeL']:.3f} | BLEU: {bleu:.3f} | BERTScore: {bert_f1:.3f} | Latency: {latency:.0f}ms" + (f" | {extra_str}" if extra_str else ""))
        
        return metrics
    
    def evaluate_dataset(self, dataset_path: str, **kwargs) -> Tuple[List[EvaluationMetrics], Dict]:
        """Evaluate on a dataset from JSONL file."""
        dataset = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    dataset.append(json.loads(line))
        
        print(f"\nEvaluating on {len(dataset)} examples...")
        
        results = []
        for i, item in enumerate(dataset, 1):
            print(f"[{i}/{len(dataset)}] {item['query'][:50]}...")
            
            metrics = self.evaluate_single(
                query=item['query'],
                reference=item['reference_answer'],
                relevant_doc_ids=item.get('relevant_doc_ids', []),
                **kwargs
            )
            results.append(metrics)
        
        # Aggregate results
        aggregate = {
            'total': len(results),
            'avg_rouge_1': np.mean([r.rouge_1_f1 for r in results]),
            'avg_rouge_2': np.mean([r.rouge_2_f1 for r in results]),
            'avg_rouge_l': np.mean([r.rouge_l_f1 for r in results]),
            'avg_bleu': np.mean([r.bleu_score for r in results]),
            'avg_bert_score': np.mean([r.bert_score_f1 for r in results]),
            'avg_token_f1': np.mean([r.token_f1 for r in results]),
            'exact_match_rate': sum(r.exact_match for r in results) / len(results),
            'avg_latency_ms': np.mean([r.latency_ms for r in results]),
            'queries_under_3s': sum(r.latency_ms < 3000 for r in results),
            # Retrieval aggregates
            'avg_precision_at_k': np.mean([r.precision_at_k for r in results]),
            'avg_recall_at_k': np.mean([r.recall_at_k for r in results]),
            'avg_mrr': np.mean([r.mrr for r in results]),
        }
        
        # Print summary
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total queries: {aggregate['total']}")
        print(f"\nGeneration Metrics:")
        print(f"  ROUGE-1 F1:    {aggregate['avg_rouge_1']:.4f}")
        print(f"  ROUGE-2 F1:    {aggregate['avg_rouge_2']:.4f}")
        print(f"  ROUGE-L F1:    {aggregate['avg_rouge_l']:.4f}")
        print(f"  BLEU:          {aggregate['avg_bleu']:.4f}")
        print(f"  BERTScore F1:  {aggregate['avg_bert_score']:.4f}")
        print(f"  Token F1:      {aggregate['avg_token_f1']:.4f}")
        print(f"  Exact Match:   {aggregate['exact_match_rate']:.4f}")
        print(f"\nPerformance:")
        print(f"  Avg latency:   {aggregate['avg_latency_ms']:.0f}ms")
        print(f"  Queries <3s:   {aggregate['queries_under_3s']}/{aggregate['total']}")
        print(f"\nRetrieval Metrics:")
        print(f"  Precision@K:   {aggregate['avg_precision_at_k']:.4f}")
        print(f"  Recall@K:      {aggregate['avg_recall_at_k']:.4f}")
        print(f"  MRR:           {aggregate['avg_mrr']:.4f}")
        print(f"{'='*60}\n")
        
        return results, aggregate


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG system")
    parser.add_argument('--dataset', required=True, help='Evaluation dataset (JSONL)')
    parser.add_argument('--output', default='eval_results.json', help='Output file')
    parser.add_argument('--pipeline', default='fast', choices=['fast', 'baseline'])
    parser.add_argument('--verbose', '-v', action='store_true')
    
    args = parser.parse_args()
    
    # Load pipeline
    if args.pipeline == 'fast':
        from fast_rag_pipeline import FastRAGPipeline
        pipeline = FastRAGPipeline(verbose=False)
    else:
        from rag_phi2_integration import EdgeRAGPipeline
        pipeline = EdgeRAGPipeline(verbose=False)
    
    # Evaluate
    evaluator = RAGEvaluator(pipeline, verbose=args.verbose)
    results, aggregate = evaluator.evaluate_dataset(args.dataset)
    
    # Save results
    output = {
        'aggregate': aggregate,
        'individual': [asdict(r) for r in results]
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Results saved to {args.output}")


if __name__ == "__main__":
    main()
