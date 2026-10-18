"""
Unit Tests for Recommendation Engine (Content-Based, SVD Collaborative & Hybrid Fusion).
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from src.pipeline import MedallionPipeline
from src.recommender.content_based import ContentBasedRecommender
from src.recommender.collaborative import CollaborativeRecommender
from src.recommender.hybrid_engine import HybridRecommenderEngine
from src.recommender.evaluation import RecommenderEvaluator

@pytest.fixture(scope="module")
def prepared_engine():
    pipeline = MedallionPipeline()
    dim_subs, dim_prjs, fct_evls, features_mart = pipeline.run_full_pipeline()
    engine = HybridRecommenderEngine()
    engine.fit(dim_subs, dim_prjs, fct_evls, features_mart)
    return engine, fct_evls

def test_content_based_scoring(prepared_engine):
    engine, _ = prepared_engine
    sample_query = {
        "required_trade": "HVAC & Mechanical",
        "sector": "Healthcare (Hospitals & Clinics)",
        "location": "Austin, TX",
        "budget_usd": 15_000_000,
        "bim_lod_required": "LOD 400"
    }
    df_scores = engine.content_engine.score_all_for_project(sample_query)
    assert len(df_scores) == len(engine.dim_subs)
    assert "content_score" in df_scores.columns
    assert (df_scores["content_score"] >= 0.0).all() and (df_scores["content_score"] <= 1.0).all()

def test_collaborative_svd_prediction(prepared_engine):
    engine, _ = prepared_engine
    sub_ids = engine.dim_subs["subcontractor_id"].tolist()
    preds = engine.collab_engine.predict_for_project("PRJ-0001", sub_ids)
    assert len(preds) == len(sub_ids)
    assert "collaborative_score" in preds.columns
    assert (preds["collaborative_score"] >= 0.0).all() and (preds["collaborative_score"] <= 1.0).all()

def test_hybrid_recommendation_and_safety_filtering(prepared_engine):
    engine, _ = prepared_engine
    sample_project = {
        "project_name": "NextGen Biopharma Facility",
        "sector": "Biopharmaceutical & Lab Facilities",
        "location": "Austin, TX",
        "required_trade": "Cleanroom MEP & Process Piping",
        "budget_usd": 30_000_000,
        "strict_safety_mandate": True
    }
    result = engine.recommend(sample_project, top_k=5, apply_safety_filter=True)
    assert "recommendations" in result
    recs = result["recommendations"]
    assert len(recs) <= 5
    
    # Strict safety check: all recommended contractors must have EMR <= 0.90
    for rec in recs:
        assert rec["safety_emr"] <= 0.90, f"Contractor {rec['subcontractor_id']} exceeded strict safety threshold!"
        assert "scores" in rec
        assert "explanation" in rec

def test_offline_evaluator(prepared_engine):
    engine, fct_evls = prepared_engine
    evaluator = RecommenderEvaluator(engine)
    metrics = evaluator.evaluate_model(fct_evls.tail(100), k=5)
    assert "Precision@5" in metrics
    assert "NDCG@5" in metrics
    assert metrics["NDCG@5"] > 0.35, "NDCG@5 must be above baseline threshold"
