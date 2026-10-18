"""
Offline Evaluation Suite for ConstructRec Recommendation System.
Computes industry-standard ranking metrics:
- Precision@K
- Recall@K
- MAP@K (Mean Average Precision)
- NDCG@K (Normalized Discounted Cumulative Gain)
- Latency Benchmark (P50, P95, P99 ms)
"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class RecommenderEvaluator:
    def __init__(self, hybrid_engine):
        self.engine = hybrid_engine
        
    def _dcg_at_k(self, relevance_list, k):
        """Computes Discounted Cumulative Gain at K."""
        relevance = np.asarray(relevance_list, dtype=float)[:k]
        if relevance.size == 0:
            return 0.0
        # Formula: sum((2^rel - 1) / log2(idx + 2))
        discounts = np.log2(np.arange(2, relevance.size + 2))
        return float(np.sum((2 ** relevance - 1) / discounts))

    def _ndcg_at_k(self, relevance_list, k):
        """Computes Normalized Discounted Cumulative Gain at K."""
        dcg = self._dcg_at_k(relevance_list, k)
        ideal_relevance = sorted(relevance_list, reverse=True)
        idcg = self._dcg_at_k(ideal_relevance, k)
        if idcg == 0.0:
            return 0.0
        return float(dcg / idcg)

    def evaluate_model(self, test_evaluations_df, k=5, relevance_threshold=3.8):
        """
        Runs batch offline evaluation on held-out project evaluations.
        A subcontractor interaction is considered 'relevant' if rating >= relevance_threshold.
        """
        # Group true positive relevant subcontractors for each project in test set
        grouped = test_evaluations_df.groupby("project_id")
        
        precisions = []
        recalls = []
        ndcgs = []
        average_precisions = []
        latencies = []
        
        dim_prjs_dict = self.engine.dim_prjs.set_index("project_id").to_dict(orient="index")
        
        for project_id, group in grouped:
            if project_id not in dim_prjs_dict:
                continue
                
            project_info = dim_prjs_dict[project_id]
            project_info["project_id"] = project_id
            
            # Ground truth relevant subcontractors (rating >= threshold)
            relevant_subs = set(group[group["rating"] >= relevance_threshold]["subcontractor_id"])
            if len(relevant_subs) == 0:
                continue
                
            # Time the recommendation inference
            start_t = time.perf_counter()
            rec_result = self.engine.recommend(project_info, top_k=k, apply_safety_filter=False)
            latency_ms = (time.perf_counter() - start_t) * 1000.0
            latencies.append(latency_ms)
            
            recommended_subs = [r["subcontractor_id"] for r in rec_result["recommendations"]]
            
            # 1. Precision@K
            hits = [1 if sub_id in relevant_subs else 0 for sub_id in recommended_subs]
            precision = sum(hits) / float(k)
            precisions.append(precision)
            
            # 2. Recall@K
            recall = sum(hits) / float(len(relevant_subs))
            recalls.append(recall)
            
            # 3. NDCG@K
            # Binary relevance 1 or 0
            ndcg = self._ndcg_at_k(hits, k)
            ndcgs.append(ndcg)
            
            # 4. Average Precision (AP@K)
            running_hits = 0
            score_sum = 0.0
            for rank_idx, hit in enumerate(hits, start=1):
                if hit:
                    running_hits += 1
                    score_sum += running_hits / float(rank_idx)
            ap = score_sum / min(len(relevant_subs), k) if min(len(relevant_subs), k) > 0 else 0.0
            average_precisions.append(ap)
            
        metrics = {
            "evaluated_queries_count": len(precisions),
            f"Precision@{k}": round(float(np.mean(precisions)), 4) if precisions else 0.0,
            f"Recall@{k}": round(float(np.mean(recalls)), 4) if recalls else 0.0,
            f"MAP@{k}": round(float(np.mean(average_precisions)), 4) if average_precisions else 0.0,
            f"NDCG@{k}": round(float(np.mean(ndcgs)), 4) if ndcgs else 0.0,
            "Latency_P50_ms": round(float(np.percentile(latencies, 50)), 2) if latencies else 0.0,
            "Latency_P95_ms": round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0,
            "Latency_P99_ms": round(float(np.percentile(latencies, 99)), 2) if latencies else 0.0,
        }
        return metrics

if __name__ == "__main__":
    from src.pipeline import MedallionPipeline
    from src.recommender.hybrid_engine import HybridRecommenderEngine
    
    pipeline = MedallionPipeline()
    dim_subs, dim_prjs, fct_evls, features_mart = pipeline.run_full_pipeline()
    
    # Train/Test Split on evaluations (80/20)
    train_evls, test_evls = train_test_split(fct_evls, test_size=0.20, random_state=42)
    
    engine = HybridRecommenderEngine()
    engine.fit(dim_subs, dim_prjs, train_evls, features_mart)
    
    evaluator = RecommenderEvaluator(engine)
    results = evaluator.evaluate_model(test_evls, k=5)
    
    print("\n" + "=" * 50)
    print("Offline Model Evaluation Metrics Summary (Top-5)")
    print("=" * 50)
    for metric, val in results.items():
        print(f" - {metric}: {val}")
    print("=" * 50)
