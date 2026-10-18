"""
Hybrid Recommendation Engine & Enterprise Business Rule Re-ranker.
Fuses Content-Based and Collaborative Filtering predictions, enforces DPR/vConstruct
construction safety constraints (EMR thresholds), handles cold starts, and outputs Explainable AI diagnostics.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from config import RECOMMENDER_CONFIG
from src.recommender.content_based import ContentBasedRecommender
from src.recommender.collaborative import CollaborativeRecommender

class HybridRecommenderEngine:
    def __init__(self, content_weight=RECOMMENDER_CONFIG["content_weight"],
                 collab_weight=RECOMMENDER_CONFIG["collaborative_weight"]):
        self.content_weight = content_weight
        self.collab_weight = collab_weight
        self.content_engine = ContentBasedRecommender()
        self.collab_engine = CollaborativeRecommender()
        self.dim_subs = None
        self.dim_prjs = None
        self.features_mart = None
        self.is_ready = False
        
    def fit(self, dim_subs, dim_prjs, fct_evls, features_mart):
        """Fits both underlying recommenders on Gold analytical data."""
        self.dim_subs = dim_subs.copy()
        self.dim_prjs = dim_prjs.copy()
        self.features_mart = features_mart.copy()
        
        print("[HybridEngine] Training Content-Based TF-IDF models...")
        self.content_engine.fit(self.dim_subs)
        
        print("[HybridEngine] Training Collaborative SVD Matrix Factorization models...")
        self.collab_engine.fit(fct_evls, self.dim_subs, self.dim_prjs)
        
        self.is_ready = True
        print("[HybridEngine] Hybrid Engine successfully trained & initialized!")
        return self

    def _generate_explanation(self, sub_row, project_dict, pred_rating):
        """Generates clear explainable rationale for recommendation."""
        reasons = []
        # Trade & Specialty match
        if sub_row["trade"].lower() == project_dict["required_trade"].lower():
            reasons.append(f"Direct match for '{sub_row['trade']}' package")
        if "BIM-LOD400" in str(sub_row.get("certifications_str", "")):
            reasons.append("VDC/BIM Level 400 Certified (Critical for prefabrication)")
        if "ASHE-Healthcare" in str(sub_row.get("certifications_str", "")):
            reasons.append("ASHE Healthcare Certified for infection control")
            
        # Safety EMR
        emr = sub_row.get("safety_emr", 1.0)
        if emr <= 0.85:
            reasons.append(f"Superior safety record (EMR {emr:.2f} < 0.85)")
        else:
            reasons.append(f"Safety compliant (EMR {emr:.2f})")
            
        # Location
        if sub_row["primary_location"] == project_dict["location"]:
            reasons.append(f"Local workforce in {sub_row['primary_location']}")
            
        # Performance
        reasons.append(f"Predicted job performance score: {pred_rating:.1f}/5.0")
        return " | ".join(reasons)

    def recommend(self, project_dict, top_k=RECOMMENDER_CONFIG["default_top_k"], apply_safety_filter=True):
        """
        Generates Top-K recommended subcontractors for a project query.
        Applies business safety rules and returns explainability metadata.
        """
        if not self.is_ready:
            raise RuntimeError("HybridRecommenderEngine must be fitted before running inference.")
            
        all_subs_list = self.dim_subs["subcontractor_id"].tolist()
        
        # 1. Compute Content-Based scores
        df_content = self.content_engine.score_all_for_project(project_dict)
        
        # 2. Compute Collaborative SVD scores
        project_id = project_dict.get("project_id", "NEW_PROJECT_QUERY")
        df_collab = self.collab_engine.predict_for_project(project_id, all_subs_list)
        
        # 3. Merge scores with Subcontractor Dimensions and Features Mart
        merged = pd.merge(self.dim_subs, df_content, on="subcontractor_id")
        merged = pd.merge(merged, df_collab, on="subcontractor_id")
        merged = pd.merge(merged, self.features_mart[["subcontractor_id", "rating_count", "bayesian_rating"]], on="subcontractor_id")
        
        # 4. Handle Cold-Start Adaptively
        # If a subcontractor has fewer than 3 reviews, shift weight towards Content + Bayesian Prior
        def compute_adaptive_hybrid(row):
            count = row["rating_count"]
            if count < RECOMMENDER_CONFIG["min_ratings_threshold"]:
                # High uncertainty in collaborative latent factor -> rely 80% on content + 20% on prior
                c_wt = 0.80
                k_wt = 0.20
                prior_norm = (row["bayesian_rating"] - 1.0) / 4.0
                return (c_wt * row["content_score"]) + (k_wt * prior_norm)
            else:
                return (self.content_weight * row["content_score"]) + (self.collab_weight * row["collaborative_score"])
                
        merged["hybrid_score"] = merged.apply(compute_adaptive_hybrid, axis=1)
        
        # 5. Business Rule Hard Filters (DPR Construction Requirements)
        # Filter 1: Safety EMR Constraint
        strict_safety = project_dict.get("strict_safety_mandate", False)
        max_emr = 0.90 if strict_safety else RECOMMENDER_CONFIG["max_allowed_emr"]
        
        if apply_safety_filter:
            eligible = merged[merged["safety_emr"] <= max_emr].copy()
            # Must match required trade
            eligible = eligible[eligible["trade"].str.lower() == project_dict["required_trade"].strip().lower()]
        else:
            eligible = merged.copy()
            
        # If filtered set is empty, gracefully relax trade constraint to prevent hard crash
        if len(eligible) == 0:
            eligible = merged[merged["safety_emr"] <= max_emr].copy()
            
        # Sort by hybrid score descending
        eligible = eligible.sort_values(by="hybrid_score", ascending=False).reset_index(drop=True)
        top_candidates = eligible.head(top_k).copy()
        
        # 6. Format Final Response with Explainability
        recommendations = []
        for rank, (_, row) in enumerate(top_candidates.iterrows(), start=1):
            explanation = self._generate_explanation(row, project_dict, row["predicted_rating_5star"])
            recommendations.append({
                "rank": rank,
                "subcontractor_id": row["subcontractor_id"],
                "company_name": row["company_name"],
                "trade": row["trade"],
                "primary_location": row["primary_location"],
                "safety_emr": float(row["safety_emr"]),
                "safety_tier": row["safety_tier"],
                "bonding_capacity_usd": int(row["bonding_capacity_usd"]),
                "experience_years": int(row["experience_years"]),
                "certifications": row["certifications_str"].split(", ") if row["certifications_str"] else [],
                "scores": {
                    "hybrid_match_pct": round(float(row["hybrid_score"]) * 100, 1),
                    "content_alignment_score": round(float(row["content_score"]), 3),
                    "predicted_rating_5star": float(row["predicted_rating_5star"]),
                    "bayesian_adjusted_rating": float(row["bayesian_rating"])
                },
                "is_cold_start": bool(row["rating_count"] < RECOMMENDER_CONFIG["min_ratings_threshold"]),
                "explanation": explanation
            })
            
        return {
            "query_project": {
                "project_id": project_dict.get("project_id", "CUSTOM_NEW_PROJECT"),
                "project_name": project_dict.get("project_name", "Prospective Construction Project"),
                "sector": project_dict.get("sector"),
                "location": project_dict.get("location"),
                "required_trade": project_dict.get("required_trade"),
                "budget_usd": project_dict.get("budget_usd"),
                "strict_safety_mandate": strict_safety
            },
            "top_k": top_k,
            "total_evaluated_contractors": len(merged),
            "compliant_candidates_count": len(eligible),
            "recommendations": recommendations
        }
