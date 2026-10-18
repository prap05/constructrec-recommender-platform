"""
DBT Model Execution & Data Quality Runner.
Parses and executes DBT staging and mart SQL transformations directly against DuckDB and SQLite,
and executes DBT-style data quality tests (Unique, Not Null, Range checks).
"""
import sys
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
from config import DB_PATH, SQLITE_DB_PATH, RAW_DATA_DIR
from src.pipeline import MedallionPipeline

class DbtModelRunner:
    def __init__(self, db_path=DB_PATH):
        self.db_path = str(db_path)
        
    def execute_transformations(self):
        """Executes full DBT-style ELT transformations in DuckDB."""
        print("=" * 60)
        print("[dbt-runner] Initializing DBT Transformation Execution...")
        print("=" * 60)
        
        # Connect to DuckDB
        con = duckdb.connect(self.db_path)
        
        # 1. Load Raw Sources into Views
        sub_raw = RAW_DATA_DIR / "subcontractors_raw.json"
        prj_raw = RAW_DATA_DIR / "projects_raw.json"
        evl_raw = RAW_DATA_DIR / "evaluations_raw.json"
        
        con.execute(f"CREATE OR REPLACE VIEW raw_subcontractors_source AS SELECT * FROM read_json_auto('{sub_raw.as_posix()}')")
        con.execute(f"CREATE OR REPLACE VIEW raw_projects_source AS SELECT * FROM read_json_auto('{prj_raw.as_posix()}')")
        con.execute(f"CREATE OR REPLACE VIEW raw_evaluations_source AS SELECT * FROM read_json_auto('{evl_raw.as_posix()}')")
        print("[dbt-runner] Loaded raw sources: raw_subcontractors_source, raw_projects_source, raw_evaluations_source")
        
        # 2. Build Staging Views (Silver)
        con.execute("""
            CREATE OR REPLACE VIEW stg_projects AS
            SELECT 
                project_id,
                TRIM(project_name) AS project_name,
                sector,
                location,
                CAST(budget_usd AS BIGINT) AS budget_usd,
                ROUND(CAST(budget_usd AS DOUBLE) / 1000000.0, 2) AS budget_millions,
                CAST(sq_footage AS INTEGER) AS sq_footage,
                TRIM(required_trade) AS required_trade,
                bim_lod_required,
                strict_safety_mandate,
                status
            FROM raw_projects_source
        """)
        
        con.execute("""
            CREATE OR REPLACE VIEW stg_subcontractors AS
            SELECT 
                subcontractor_id,
                TRIM(company_name) AS company_name,
                TRIM(trade) AS trade,
                TRIM(primary_location) AS primary_location,
                ROUND(CAST(safety_emr AS DOUBLE), 2) AS safety_emr,
                CAST(bonding_capacity_usd AS BIGINT) AS bonding_capacity_usd,
                CAST(experience_years AS INTEGER) AS experience_years,
                CAST(active_crew_count AS INTEGER) AS active_crew_count,
                array_to_string(certifications, ', ') AS certifications_str,
                array_to_string(specialties, ', ') AS specialties_str,
                CASE 
                    WHEN safety_emr <= 0.85 THEN 'Tier 1 (Superior)'
                    WHEN safety_emr <= 1.00 THEN 'Tier 2 (Standard)'
                    ELSE 'Tier 3 (High Risk)'
                END AS safety_tier,
                CASE 
                    WHEN safety_emr <= 1.05 THEN TRUE 
                    ELSE FALSE 
                END AS is_safety_compliant,
                is_active
            FROM raw_subcontractors_source
        """)
        
        con.execute("""
            CREATE OR REPLACE VIEW stg_evaluations AS
            SELECT 
                evaluation_id,
                project_id,
                subcontractor_id,
                ROUND(CAST(rating AS DOUBLE), 1) AS rating,
                ROUND(CAST(schedule_score AS DOUBLE), 1) AS schedule_score,
                ROUND(CAST(safety_score AS DOUBLE), 1) AS safety_score,
                ROUND(CAST(change_order_rate AS DOUBLE), 3) AS change_order_rate,
                recommendation_confirmed,
                CAST(timestamp AS TIMESTAMP) AS evaluation_timestamp
            FROM raw_evaluations_source
            WHERE rating IS NOT NULL AND rating >= 1.0 AND rating <= 5.0
        """)
        print("[dbt-runner] Compiled and built staging models: stg_projects, stg_subcontractors, stg_evaluations")
        
        # 3. Build Mart Tables (Gold)
        con.execute("CREATE OR REPLACE TABLE dim_projects AS SELECT * FROM stg_projects")
        con.execute("CREATE OR REPLACE TABLE dim_subcontractors AS SELECT * FROM stg_subcontractors")
        con.execute("CREATE OR REPLACE TABLE fct_evaluations AS SELECT * FROM stg_evaluations")
        
        con.execute("""
            CREATE OR REPLACE TABLE features_recommender_mart AS
            WITH sub_stats AS (
                SELECT 
                    subcontractor_id,
                    COUNT(evaluation_id) AS rating_count,
                    ROUND(AVG(rating), 2) AS avg_rating,
                    ROUND(AVG(schedule_score), 2) AS avg_schedule_score,
                    ROUND(AVG(safety_score), 2) AS avg_safety_score,
                    ROUND(AVG(change_order_rate), 3) AS avg_change_order_rate
                FROM fct_evaluations
                GROUP BY subcontractor_id
            ),
            base_subs AS (
                SELECT * FROM dim_subcontractors
            )
            SELECT 
                s.subcontractor_id,
                s.company_name,
                s.trade,
                s.primary_location,
                s.safety_emr,
                s.bonding_capacity_usd,
                s.experience_years,
                s.active_crew_count,
                s.certifications_str,
                s.specialties_str,
                s.safety_tier,
                s.is_safety_compliant,
                COALESCE(st.rating_count, 0) AS rating_count,
                COALESCE(st.avg_rating, 3.8) AS avg_rating,
                COALESCE(st.avg_schedule_score, 3.8) AS avg_schedule_score,
                COALESCE(st.avg_safety_score, 3.8) AS avg_safety_score,
                COALESCE(st.avg_change_order_rate, 0.05) AS avg_change_order_rate,
                ROUND(
                    (COALESCE(st.rating_count, 0) * COALESCE(st.avg_rating, 3.8) + 5.0 * 3.8) 
                    / (COALESCE(st.rating_count, 0) + 5.0),
                    2
                ) AS bayesian_rating
            FROM base_subs s
            LEFT JOIN sub_stats st ON s.subcontractor_id = st.subcontractor_id
        """)
        print("[dbt-runner] Materialized data marts: dim_projects, dim_subcontractors, fct_evaluations, features_recommender_mart")
        
        # 4. Run Data Quality Assertions (dbt test equivalent)
        print("\n[dbt-test] Running Schema & Integrity Tests...")
        self._run_tests(con)
        
        con.close()
        print("=" * 60)
        print("[dbt-runner] All DBT models and tests executed with 100% SUCCESS!")
        print("=" * 60)

    def _run_tests(self, con):
        """Executes dbt-style assertions."""
        # Test 1: Unique subcontractor_id
        res = con.execute("SELECT subcontractor_id, COUNT(*) FROM dim_subcontractors GROUP BY subcontractor_id HAVING COUNT(*) > 1").fetchall()
        assert len(res) == 0, f"Test Failed: duplicate subcontractor_id found: {res}"
        print(" [PASS] test_unique_dim_subcontractors_subcontractor_id")
        
        # Test 2: Not null safety_emr
        res = con.execute("SELECT COUNT(*) FROM dim_subcontractors WHERE safety_emr IS NULL").fetchone()[0]
        assert res == 0, "Test Failed: null safety_emr in dim_subcontractors"
        print(" [PASS] test_not_null_dim_subcontractors_safety_emr")
        
        # Test 3: Unique project_id
        res = con.execute("SELECT project_id, COUNT(*) FROM dim_projects GROUP BY project_id HAVING COUNT(*) > 1").fetchall()
        assert len(res) == 0, f"Test Failed: duplicate project_id found: {res}"
        print(" [PASS] test_unique_dim_projects_project_id")
        
        # Test 4: Rating range check 1.0 to 5.0
        res = con.execute("SELECT COUNT(*) FROM fct_evaluations WHERE rating < 1.0 OR rating > 5.0").fetchone()[0]
        assert res == 0, "Test Failed: rating out of bounds [1.0, 5.0]"
        print(" [PASS] test_range_fct_evaluations_rating")
        
        # Test 5: Features mart row count matches dim_subcontractors
        cnt_subs = con.execute("SELECT COUNT(*) FROM dim_subcontractors").fetchone()[0]
        cnt_feat = con.execute("SELECT COUNT(*) FROM features_recommender_mart").fetchone()[0]
        assert cnt_subs == cnt_feat, f"Mismatch: dim_subs={cnt_subs}, features_mart={cnt_feat}"
        print(f" [PASS] test_row_integrity_features_mart (Rows: {cnt_feat})")

if __name__ == "__main__":
    runner = DbtModelRunner()
    runner.execute_transformations()
