"""
ConstructRec: Master End-to-End Orchestrator & Live Demo Runner.
Executes the full data engineering pipeline, dbt transformations, ML training,
Power BI export, and launches the web presentation dashboard.
"""
import sys
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import MedallionPipeline
from src.run_dbt_models import DbtModelRunner
from src.recommender.hybrid_engine import HybridRecommenderEngine
from src.recommender.evaluation import RecommenderEvaluator
from src.export_powerbi import export_powerbi_star_schema
from config import MODELS_DIR
import joblib

def main():
    print(r"""
    ====================================================================
      ____                 _                   _   ____            
     / ___|___  _ __  ___| |_ _ __ _   _  ___| |_|  _ \ ___  ___  
    | |   / _ \| '_ \/ __| __| '__| | | |/ __| __| |_) / _ \/ __| 
    | |__| (_) | | | \__ \ |_| |  | |_| | (__| |_|  _ <  __/ (__  
     \____\___/|_| |_|___/\__|_|   \__,_|\___|\__|_| \_\___|\___| 
                                                                    
      Enterprise Subcontractor & Resource Recommender Data Platform
      Tailored for vConstruct & DPR Construction Business Domains
    ====================================================================
    """)
    
    # Step 1: Run Medallion Pipeline (Bronze -> Silver -> Gold)
    print("\n[STEP 1/5] Running Medallion Data Engineering Pipeline...")
    pipeline = MedallionPipeline()
    dim_subs, dim_prjs, fct_evls, features_mart = pipeline.run_full_pipeline()
    
    # Step 2: Execute DBT Models & Schema Assertions
    print("\n[STEP 2/5] Executing DBT Staging & Analytical Mart Models...")
    dbt_runner = DbtModelRunner()
    dbt_runner.execute_transformations()
    
    # Step 3: Train Hybrid Recommender (TF-IDF + SVD Matrix Factorization)
    print("\n[STEP 3/5] Fitting Machine Learning Hybrid Recommender Engine...")
    engine = HybridRecommenderEngine()
    engine.fit(dim_subs, dim_prjs, fct_evls, features_mart)
    joblib.dump(engine, MODELS_DIR / "production_hybrid_engine.pkl")
    
    # Step 4: Evaluate Model Offline
    print("\n[STEP 4/5] Computing Offline Model Evaluation Metrics (Top-5)...")
    evaluator = RecommenderEvaluator(engine)
    metrics = evaluator.evaluate_model(fct_evls.tail(150), k=5)
    print("--------------------------------------------------------------------")
    print(f" * Precision@5:  {metrics['Precision@5']:.4f}")
    print(f" * Recall@5:     {metrics['Recall@5']:.4f}")
    print(f" * NDCG@5:       {metrics['NDCG@5']:.4f}  (Ranking Quality)")
    print(f" * MAP@5:        {metrics['MAP@5']:.4f}")
    print(f" * Latency P50:  {metrics['Latency_P50_ms']} ms")
    print("--------------------------------------------------------------------")
    
    # Step 5: Export Power BI Star Schema
    print("\n[STEP 5/5] Exporting Power BI Star Schema & DAX Measures Guide...")
    export_powerbi_star_schema()
    
    print("\n====================================================================")
    import os
    port = int(os.environ.get("PORT", 5000))
    print(f"  Starting Interactive Presentation Dashboard at http://0.0.0.0:{port}")
    print("  Press Ctrl+C to terminate the demo server.")
    print("====================================================================\n")
    
    from web_dashboard.app import app
    app.run(host="0.0.0.0", port=port, debug=False)

if __name__ == "__main__":
    main()
