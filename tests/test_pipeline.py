"""
Unit & Integration Tests for Medallion Data Engineering Pipeline.
Validates Bronze Ingestion, Silver Cleansing, and Gold Star Schema / Feature Store Integrity.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import pandas as pd
from src.pipeline import MedallionPipeline
from config import RECOMMENDER_CONFIG

@pytest.fixture(scope="module")
def pipeline_data():
    pipeline = MedallionPipeline()
    dim_subs, dim_prjs, fct_evls, features_mart = pipeline.run_full_pipeline()
    return {
        "dim_subs": dim_subs,
        "dim_prjs": dim_prjs,
        "fct_evls": fct_evls,
        "features_mart": features_mart
    }

def test_dim_subcontractors_integrity(pipeline_data):
    dim_subs = pipeline_data["dim_subs"]
    assert len(dim_subs) > 0, "dim_subcontractors must not be empty"
    assert dim_subs["subcontractor_id"].is_unique, "subcontractor_id must be strictly unique"
    assert not dim_subs["safety_emr"].isnull().any(), "safety_emr cannot contain NULL values"
    assert dim_subs["safety_tier"].isin(["Tier 1 (Superior)", "Tier 2 (Standard)", "Tier 3 (High Risk)"]).all()

def test_dim_projects_integrity(pipeline_data):
    dim_prjs = pipeline_data["dim_prjs"]
    assert len(dim_prjs) > 0, "dim_projects must not be empty"
    assert dim_prjs["project_id"].is_unique, "project_id must be strictly unique"
    assert (dim_prjs["budget_usd"] > 0).all(), "budget_usd must be positive"

def test_fct_evaluations_referential_integrity(pipeline_data):
    dim_subs = pipeline_data["dim_subs"]
    dim_prjs = pipeline_data["dim_prjs"]
    fct_evls = pipeline_data["fct_evls"]
    
    assert len(fct_evls) > 0, "fct_evaluations must not be empty"
    assert fct_evls["evaluation_id"].is_unique, "evaluation_id must be unique"
    
    # Check valid foreign keys
    assert fct_evls["subcontractor_id"].isin(dim_subs["subcontractor_id"]).all(), "Orphaned subcontractor_id in fct_evaluations!"
    assert fct_evls["project_id"].isin(dim_prjs["project_id"]).all(), "Orphaned project_id in fct_evaluations!"
    
    # Rating bounds
    assert (fct_evls["rating"] >= 1.0).all() and (fct_evls["rating"] <= 5.0).all()

def test_features_recommender_mart_bayesian_ratings(pipeline_data):
    features_mart = pipeline_data["features_mart"]
    assert len(features_mart) == len(pipeline_data["dim_subs"]), "Every subcontractor must have a feature mart record"
    assert "bayesian_rating" in features_mart.columns
    assert (features_mart["bayesian_rating"] >= 1.0).all() and (features_mart["bayesian_rating"] <= 5.0).all()
