"""
Integration Tests for FastAPI Microservice Endpoints.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["contractors_indexed"] > 0
    assert data["engine_ready"] is True

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "precision_at_5" in data
    assert "ndcg_at_5" in data
    assert "inference_latency_p50_ms" in data

def test_subcontractors_catalog():
    response = client.get("/subcontractors?trade=HVAC%20%26%20Mechanical")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] > 0
    for item in data["results"]:
        assert item["trade"] == "HVAC & Mechanical"

def test_recommendation_endpoint():
    payload = {
        "project_name": "Cancer Research Wing",
        "sector": "Healthcare (Hospitals & Clinics)",
        "location": "San Francisco, CA",
        "required_trade": "HVAC & Mechanical",
        "budget_usd": 20_000_000,
        "strict_safety_mandate": True,
        "top_k": 3
    }
    response = client.post("/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["recommendations"]) <= 3
    assert data["inference_latency_ms"] > 0
    first_rec = data["recommendations"][0]
    assert "company_name" in first_rec
    assert "safety_emr" in first_rec
    assert first_rec["safety_emr"] <= 0.90
