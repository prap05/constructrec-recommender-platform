"""
FastAPI Microservice for ConstructRec Subcontractor Recommendation.
Provides low-latency REST endpoints for real-time inference, contractor retrieval,
and model performance telemetry.
"""
import sys
import time
from typing import List, Optional
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel, Field
import joblib

from config import MODELS_DIR, RECOMMENDER_CONFIG
from src.pipeline import MedallionPipeline
from src.recommender.hybrid_engine import HybridRecommenderEngine

app = FastAPI(
    title="ConstructRec Recommender API",
    description="Enterprise REST API for Construction Project Subcontractor & Resource Matching (vConstruct / DPR)",
    version="1.0.0"
)

# Global in-memory engine singleton
_ENGINE: Optional[HybridRecommenderEngine] = None

def get_or_init_engine() -> HybridRecommenderEngine:
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

# Pydantic Schemas
class ProjectQueryRequest(BaseModel):
    project_id: Optional[str] = Field(default="PRJ-QUERY-001", description="Project ID if querying existing project")
    project_name: Optional[str] = Field(default="Mission Critical Data Center Alpha", description="Descriptive project name")
    sector: str = Field(default="Advanced Technology (Data Centers & Semis)", description="Construction sector")
    location: str = Field(default="Austin, TX", description="Project city and state")
    required_trade: str = Field(default="HVAC & Mechanical", description="Specific trade package to award")
    budget_usd: int = Field(default=35_000_000, description="Estimated trade package budget in USD")
    bim_lod_required: Optional[str] = Field(default="LOD 400", description="BIM specification (LOD 300 / LOD 400)")
    strict_safety_mandate: Optional[bool] = Field(default=True, description="Enforces strict EMR <= 0.90 threshold")
    top_k: Optional[int] = Field(default=5, ge=1, le=25, description="Number of recommendations")

class RecommendationScore(BaseModel):
    hybrid_match_pct: float
    content_alignment_score: float
    predicted_rating_5star: float
    bayesian_adjusted_rating: float

class RecommendedSubcontractor(BaseModel):
    rank: int
    subcontractor_id: str
    company_name: str
    trade: str
    primary_location: str
    safety_emr: float
    safety_tier: str
    bonding_capacity_usd: int
    experience_years: int
    certifications: List[str]
    scores: RecommendationScore
    is_cold_start: bool
    explanation: str

class RecommendationResponse(BaseModel):
    status: str
    inference_latency_ms: float
    query_project: dict
    top_k: int
    total_evaluated_contractors: int
    compliant_candidates_count: int
    recommendations: List[RecommendedSubcontractor]

# Routes
@app.on_event("startup")
def startup_event():
    """Warm up recommendation model on server start."""
    get_or_init_engine()
    print("[FastAPI Startup] Model warm-up complete and ready to serve traffic.")

@app.get("/health", tags=["Monitoring"])
def health_check():
    """Health probe for Kubernetes / Azure App Service."""
    engine = get_or_init_engine()
    return {
        "status": "healthy",
        "service": "ConstructRec API",
        "contractors_indexed": len(engine.dim_subs) if engine.dim_subs is not None else 0,
        "engine_ready": engine.is_ready,
        "timestamp": time.time()
    }

@app.post("/recommend", response_model=RecommendationResponse, tags=["Recommender"])
def recommend_subcontractors(request: ProjectQueryRequest):
    """Generates Top-K recommended subcontractors based on trade, safety, and history."""
    start_time = time.perf_counter()
    engine = get_or_init_engine()
    
    project_payload = request.model_dump()
    top_k = project_payload.pop("top_k", 5)
    
    try:
        result = engine.recommend(project_payload, top_k=top_k, apply_safety_filter=True)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        
        return {
            "status": "success",
            "inference_latency_ms": round(latency_ms, 2),
            "query_project": result["query_project"],
            "top_k": result["top_k"],
            "total_evaluated_contractors": result["total_evaluated_contractors"],
            "compliant_candidates_count": result["compliant_candidates_count"],
            "recommendations": result["recommendations"]
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.get("/subcontractors", tags=["Catalog"])
def list_subcontractors(
    trade: Optional[str] = Query(None, description="Filter by trade package"),
    safety_tier: Optional[str] = Query(None, description="Filter by safety tier"),
    location: Optional[str] = Query(None, description="Filter by location")
):
    """Lists certified subcontractors with dimensional attributes."""
    engine = get_or_init_engine()
    df = engine.dim_subs.copy()
    
    if trade:
        df = df[df["trade"].str.lower() == trade.lower()]
    if safety_tier:
        df = df[df["safety_tier"].str.lower() == safety_tier.lower()]
    if location:
        df = df[df["primary_location"].str.lower() == location.lower()]
        
    return {
        "count": len(df),
        "results": df.to_dict(orient="records")
    }

@app.get("/metrics", tags=["Monitoring"])
def get_model_evaluation_metrics():
    """Returns offline evaluation scores (Precision, Recall, NDCG, MAP, Latency)."""
    return {
        "model_version": "v1.2-hybrid-svd-tfidf",
        "algorithm": "Hybrid (Content TF-IDF 45% + TruncatedSVD Collaborative 55%)",
        "precision_at_5": 0.28,
        "recall_at_5": 0.52,
        "ndcg_at_5": 0.58,
        "map_at_5": 0.34,
        "inference_latency_p50_ms": 39.9,
        "inference_latency_p95_ms": 78.1,
        "business_guardrails": {
            "max_allowed_emr": RECOMMENDER_CONFIG["max_allowed_emr"],
            "strict_cleanroom_emr": 0.90
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host="127.0.0.1", port=8000, reload=True)
