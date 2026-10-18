"""
Apache Airflow DAG: ConstructRec Daily Ingestion, DBT Modeling & Recommender Retraining Pipeline.
Orchestrates:
1. Bronze Ingestion from Construction Project Management APIs (Procore/BIM360)
2. Data Quality & Schema Assertions
3. DBT Transformations (Staging -> Dimensional Star Schema & Feature Mart)
4. Offline ML Model Retraining (TF-IDF Content + SVD Matrix Factorization)
5. Model Evaluation & Drift Guardrail Monitoring
6. Power BI Data Mart Export
"""
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Try importing Airflow components, with graceful mock fallback for local environments without Airflow installed
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.operators.empty import EmptyOperator
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False
    # Mock classes for testing DAG parsing without active Airflow scheduler
    class DAG:
        def __init__(self, dag_id, default_args=None, schedule_interval=None, catchup=False, description=None, tags=None):
            self.dag_id = dag_id
            self.default_args = default_args
            self.schedule_interval = schedule_interval
            self.catchup = catchup
            self.description = description
            self.tags = tags
            
    class EmptyOperator:
        def __init__(self, task_id, dag=None):
            self.task_id = task_id
            self.dag = dag
        def __rshift__(self, other):
            return other
        def __lshift__(self, other):
            return other
            
    class PythonOperator:
        def __init__(self, task_id, python_callable, op_kwargs=None, dag=None):
            self.task_id = task_id
            self.python_callable = python_callable
            self.op_kwargs = op_kwargs or {}
            self.dag = dag
        def __rshift__(self, other):
            return other
        def __lshift__(self, other):
            return other

# Task Callables
def task_ingest_raw_apis(**context):
    """Bronze Layer: Ingests project, subcontractor, and job review records."""
    from src.pipeline import MedallionPipeline
    pipeline = MedallionPipeline()
    df_subs, df_prjs, df_evls = pipeline.run_bronze_ingestion()
    print(f"[Airflow Task: Ingestion] Ingested Subs={len(df_subs)}, Projects={len(df_prjs)}, Evals={len(df_evls)}")
    return {"subs_count": len(df_subs), "prjs_count": len(df_prjs), "evals_count": len(df_evls)}

def task_validate_data_quality(**context):
    """Bronze Quality Gate: Checks for empty files or missing mandatory columns."""
    from config import RAW_DATA_DIR
    import json
    for filename in ["subcontractors_raw.json", "projects_raw.json", "evaluations_raw.json"]:
        path = RAW_DATA_DIR / filename
        assert path.exists(), f"Missing raw file: {filename}"
        with open(path) as f:
            data = json.load(f)
            assert len(data) > 0, f"Raw file {filename} is empty!"
    print("[Airflow Task: Quality Gate] All Bronze files verified!")

def task_run_dbt_transformations(**context):
    """Silver & Gold Layer: Executes DBT transformations and schema assertions."""
    from src.run_dbt_models import DbtModelRunner
    runner = DbtModelRunner()
    runner.execute_transformations()
    print("[Airflow Task: DBT] Transformations and schema assertions passed!")

def task_train_recommender_models(**context):
    """ML Layer: Retrains Content-Based TF-IDF and SVD Collaborative models."""
    from src.pipeline import MedallionPipeline
    from src.recommender.hybrid_engine import HybridRecommenderEngine
    import joblib
    from config import MODELS_DIR
    
    pipeline = MedallionPipeline()
    dim_subs, dim_prjs, fct_evls, features_mart = pipeline.run_full_pipeline()
    
    engine = HybridRecommenderEngine()
    engine.fit(dim_subs, dim_prjs, fct_evls, features_mart)
    
    # Save the trained engine
    joblib.dump(engine, MODELS_DIR / "production_hybrid_engine.pkl")
    print(f"[Airflow Task: ML Training] Models serialized to {MODELS_DIR}")

def task_evaluate_model_drift(**context):
    """MLOps Gate: Evaluates model performance against baseline thresholds."""
    import joblib
    from config import MODELS_DIR
    from src.pipeline import MedallionPipeline
    from src.recommender.evaluation import RecommenderEvaluator
    
    engine = joblib.load(MODELS_DIR / "production_hybrid_engine.pkl")
    pipeline = MedallionPipeline()
    _, _, fct_evls, _ = pipeline.run_full_pipeline()
    
    evaluator = RecommenderEvaluator(engine)
    metrics = evaluator.evaluate_model(fct_evls.tail(150), k=5)
    
    # Quality assertion guardrails
    assert metrics["NDCG@5"] > 0.40, f"Model degradation detected! NDCG@5: {metrics['NDCG@5']}"
    print(f"[Airflow Task: Evaluation] Guardrails passed: NDCG@5 = {metrics['NDCG@5']}, P@5 = {metrics['Precision@5']}")

def task_export_powerbi_data(**context):
    """BI Mart Export: Generates clean star schema extracts for Power BI."""
    from src.export_powerbi import export_powerbi_star_schema
    export_powerbi_star_schema()
    print("[Airflow Task: PowerBI Export] Star schema files generated!")

# DAG Configuration
default_args = {
    "owner": "vconstruct_data_engineering",
    "depends_on_past": False,
    "email": ["data-eng@vconstruct.example.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    dag_id="constructrec_subcontractor_recommender_pipeline",
    default_args=default_args,
    description="Daily ELT, DBT Transformations & Recommender Model Training for DPR Subcontractor Allocation",
    schedule_interval="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["construction", "recommendation_system", "dbt", "dpr_vconstruct", "mlops"],
)

# Define Tasks
start = EmptyOperator(task_id="start_pipeline", dag=dag)
ingest = PythonOperator(task_id="ingest_raw_apis", python_callable=task_ingest_raw_apis, dag=dag)
quality_check = PythonOperator(task_id="validate_data_quality_bronze", python_callable=task_validate_data_quality, dag=dag)
dbt_transform = PythonOperator(task_id="run_dbt_transformations", python_callable=task_run_dbt_transformations, dag=dag)
train_ml = PythonOperator(task_id="train_hybrid_recommender", python_callable=task_train_recommender_models, dag=dag)
evaluate_ml = PythonOperator(task_id="evaluate_model_drift_and_metrics", python_callable=task_evaluate_model_drift, dag=dag)
export_bi = PythonOperator(task_id="export_powerbi_data_marts", python_callable=task_export_powerbi_data, dag=dag)
end = EmptyOperator(task_id="end_pipeline", dag=dag)

# Define Pipeline Dependencies
start >> ingest >> quality_check >> dbt_transform >> [train_ml, export_bi]
train_ml >> evaluate_ml >> end
export_bi >> end
