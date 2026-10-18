"""
Configuration Module for ConstructRec System.
Centralizes paths, database connections, model hyperparameters, and business rules.
"""
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
POWERBI_EXPORT_DIR = DATA_DIR / "powerbi_export"
MODELS_DIR = BASE_DIR / "models_store"

# Database Configuration (DuckDB for local OLAP / SQLite fallback)
DB_PATH = PROCESSED_DATA_DIR / "constructrec_dwh.duckdb"
SQLITE_DB_PATH = PROCESSED_DATA_DIR / "constructrec_dwh.db"

# Ensure directories exist
for folder in [RAW_DATA_DIR, PROCESSED_DATA_DIR, POWERBI_EXPORT_DIR, MODELS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Recommender Hyperparameters & Business Rules
RECOMMENDER_CONFIG = {
    # Hybrid weighting
    "content_weight": 0.45,
    "collaborative_weight": 0.55,
    
    # SVD Collaborative Filtering
    "svd_n_components": 12,
    "svd_random_state": 42,
    
    # Construction Business Hard & Soft Rules
    "max_allowed_emr": 1.05,       # Safety Experience Modification Rate threshold (industry standard <= 1.0)
    "min_quality_score": 3.0,      # Minimum historical score out of 5.0
    "default_top_k": 5,            # Number of recommendations returned
    
    # Cold-Start Strategy
    "min_ratings_threshold": 3,    # Below this, fallback to Content-Based + Bayesian Prior
    "bayesian_prior_mean": 3.8,    # Global prior mean for contractor rating
    "bayesian_prior_weight": 5     # Pseudo-counts weight
}

# Trade Categories in Commercial Construction (DPR / vConstruct Domains)
TRADE_CATEGORIES = [
    "HVAC & Mechanical",
    "Electrical & Low Voltage",
    "Structural Steel & Precast",
    "Concrete & Foundations",
    "Cleanroom MEP & Process Piping",
    "Drywall & Framing",
    "Fire Protection & Life Safety",
    "Building Automation & Controls",
    "Roofing & Waterproofing",
    "Plumbing & Industrial Piping"
]

PROJECT_SECTORS = [
    "Healthcare (Hospitals & Clinics)",
    "Advanced Technology (Data Centers & Semis)",
    "Biopharmaceutical & Lab Facilities",
    "Commercial High-Rise & Corporate Campus",
    "Higher Education & Research Centers"
]

LOCATIONS = [
    "Austin, TX",
    "San Francisco, CA",
    "Phoenix, AZ",
    "Raleigh-Durham, NC",
    "Seattle, WA",
    "Denver, CO"
]
