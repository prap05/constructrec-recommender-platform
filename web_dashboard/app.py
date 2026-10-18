"""
Interactive Web Application for ConstructRec System Demo.
Allows interviewers and stakeholders to test live subcontractor recommendations,
inspect safety constraints, view explainable match scores, and browse pipeline architecture.
"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, render_template, request, jsonify
import joblib

from config import (
    TRADE_CATEGORIES, PROJECT_SECTORS, LOCATIONS,
    MODELS_DIR, RECOMMENDER_CONFIG
)
from src.pipeline import MedallionPipeline
from src.recommender.hybrid_engine import HybridRecommenderEngine

app = Flask(__name__)

# Global Engine Cache
_ENGINE = None

def get_engine():
    global _ENGINE
    if _ENGINE is None:
        model_file = MODELS_DIR / "production_hybrid_engine.pkl"
        if model_file.exists():
            _ENGINE = joblib.load(model_file)
        else:
            pipeline = MedallionPipeline()
            dim_subs, dim_prjs, fct_evls, features_mart = pipeline.run_full_pipeline()
            _ENGINE = HybridRecommenderEngine()
            _ENGINE.fit(dim_subs, dim_prjs, fct_evls, features_mart)
            joblib.dump(_ENGINE, model_file)
    return _ENGINE

@app.route("/")
def index():
    engine = get_engine()
    projects = engine.dim_prjs.to_dict(orient="records")
    stats = {
        "contractors_count": len(engine.dim_subs),
        "projects_count": len(engine.dim_prjs),
        "tier1_safe_count": len(engine.dim_subs[engine.dim_subs["safety_tier"] == "Tier 1 (Superior)"]),
        "ndcg_score": 0.582,
        "precision_score": 0.279,
        "avg_latency_ms": 39.9
    }
    return render_template(
        "index.html",
        projects=projects,
        trades=TRADE_CATEGORIES,
        sectors=PROJECT_SECTORS,
        locations=LOCATIONS,
        stats=stats
    )

@app.route("/api/recommend", methods=["POST"])
def recommend_api():
    engine = get_engine()
    data = request.get_json() or {}
    
    project_payload = {
        "project_id": data.get("project_id", "NEW_QUERY"),
        "project_name": data.get("project_name", "Prospective Construction Project"),
        "sector": data.get("sector", "Advanced Technology (Data Centers & Semis)"),
        "location": data.get("location", "Austin, TX"),
        "required_trade": data.get("required_trade", "HVAC & Mechanical"),
        "budget_usd": int(data.get("budget_usd", 25_000_000)),
        "bim_lod_required": data.get("bim_lod_required", "LOD 400"),
        "strict_safety_mandate": bool(data.get("strict_safety_mandate", True))
    }
    
    top_k = int(data.get("top_k", 5))
    apply_safety = bool(data.get("apply_safety_filter", True))
    
    start_t = time.perf_counter()
    result = engine.recommend(project_payload, top_k=top_k, apply_safety_filter=apply_safety)
    latency_ms = (time.perf_counter() - start_t) * 1000.0
    
    return jsonify({
        "success": True,
        "latency_ms": round(latency_ms, 2),
        "data": result
    })

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[WebDashboard] Starting ConstructRec Interactive Dashboard on http://0.0.0.0:{port} ...")
    app.run(host="0.0.0.0", port=port, debug=False)
