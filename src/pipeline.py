"""
Medallion Architecture ETL Pipeline for ConstructRec.
Transforms raw enterprise JSON data (Bronze) into cleaned relational tables (Silver)
and analytical feature marts (Gold) using DuckDB and SQLite.
"""
import json
import sqlite3
import sys
from pathlib import Path

# Add project root to path for standalone execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import duckdb

from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, DB_PATH, SQLITE_DB_PATH,
    RECOMMENDER_CONFIG
)
from src.data_generator import generate_all_raw_data

class MedallionPipeline:
    def __init__(self, db_path=DB_PATH, sqlite_path=SQLITE_DB_PATH):
        self.db_path = str(db_path)
        self.sqlite_path = str(sqlite_path)
        
    def run_bronze_ingestion(self):
        """
        Bronze Layer: Ingests raw JSON dumps simulating API calls / Azure Data Factory payloads.
        Returns raw DataFrames.
        """
        sub_file = RAW_DATA_DIR / "subcontractors_raw.json"
        prj_file = RAW_DATA_DIR / "projects_raw.json"
        evl_file = RAW_DATA_DIR / "evaluations_raw.json"
        
        if not (sub_file.exists() and prj_file.exists() and evl_file.exists()):
            generate_all_raw_data()
            
        with open(sub_file, "r") as f:
            raw_subs = json.load(f)
        with open(prj_file, "r") as f:
            raw_prjs = json.load(f)
        with open(evl_file, "r") as f:
            raw_evls = json.load(f)
            
        df_subs_bronze = pd.DataFrame(raw_subs)
        df_prjs_bronze = pd.DataFrame(raw_prjs)
        df_evls_bronze = pd.DataFrame(raw_evls)
        
        print(f"[Pipeline: Bronze] Ingested raw records: Subs={len(df_subs_bronze)}, Projects={len(df_prjs_bronze)}, Evals={len(df_evls_bronze)}")
        return df_subs_bronze, df_prjs_bronze, df_evls_bronze

    def run_silver_transformation(self, df_subs_raw, df_prjs_raw, df_evls_raw):
        """
        Silver Layer: Data cleansing, deduplication, type casting, schema conformance,
        and business indicator derivation (e.g. Safety EMR flags).
        """
        # 1. Clean Subcontractors
        df_subs = df_subs_raw.copy().drop_duplicates(subset=["subcontractor_id"])
        df_subs["certifications_str"] = df_subs["certifications"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
        df_subs["specialties_str"] = df_subs["specialties"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
        df_subs["safety_tier"] = pd.cut(
            df_subs["safety_emr"],
            bins=[0.0, 0.85, 1.00, 2.0],
            labels=["Tier 1 (Superior)", "Tier 2 (Standard)", "Tier 3 (High Risk)"]
        ).astype(str)
        df_subs["is_safety_compliant"] = df_subs["safety_emr"] <= RECOMMENDER_CONFIG["max_allowed_emr"]
        
        # 2. Clean Projects
        df_prjs = df_prjs_raw.copy().drop_duplicates(subset=["project_id"])
        df_prjs["budget_millions"] = (df_prjs["budget_usd"] / 1_000_000).round(2)
        
        # 3. Clean Evaluations
        df_evls = df_evls_raw.copy().drop_duplicates(subset=["evaluation_id"])
        df_evls["rating"] = pd.to_numeric(df_evls["rating"], errors="coerce").fillna(3.0)
        df_evls["safety_score"] = pd.to_numeric(df_evls["safety_score"], errors="coerce").fillna(3.0)
        df_evls["schedule_score"] = pd.to_numeric(df_evls["schedule_score"], errors="coerce").fillna(3.0)
        df_evls["timestamp"] = pd.to_datetime(df_evls["timestamp"])
        
        # Join check to filter orphaned evaluations
        valid_subs = set(df_subs["subcontractor_id"])
        valid_prjs = set(df_prjs["project_id"])
        df_evls = df_evls[df_evls["subcontractor_id"].isin(valid_subs) & df_evls["project_id"].isin(valid_prjs)]
        
        print(f"[Pipeline: Silver] Cleansed & validated: Subs={len(df_subs)}, Projects={len(df_prjs)}, Evals={len(df_evls)}")
        return df_subs, df_prjs, df_evls

    def run_gold_modeling(self, df_subs_silver, df_prjs_silver, df_evls_silver):
        """
        Gold Layer: Dimensional modeling (Star Schema) and analytical Feature Store.
        Computes Bayesian average rating for cold-start mitigation and contractor reputation.
        """
        # Dim Tables
        dim_subcontractors = df_subs_silver[[
            "subcontractor_id", "company_name", "trade", "primary_location", 
            "safety_emr", "bonding_capacity_usd", "experience_years", 
            "active_crew_count", "certifications_str", "specialties_str", 
            "safety_tier", "is_safety_compliant", "is_active"
        ]].copy()
        
        dim_projects = df_prjs_silver[[
            "project_id", "project_name", "sector", "location", 
            "budget_usd", "budget_millions", "sq_footage", 
            "required_trade", "bim_lod_required", "strict_safety_mandate", "status"
        ]].copy()
        
        fct_evaluations = df_evls_silver[[
            "evaluation_id", "project_id", "subcontractor_id", 
            "rating", "schedule_score", "safety_score", 
            "change_order_rate", "recommendation_confirmed", "timestamp"
        ]].copy()
        
        # Feature Mart: Aggregated Subcontractor Metrics for ML Features
        prior_mean = RECOMMENDER_CONFIG["bayesian_prior_mean"]
        prior_weight = RECOMMENDER_CONFIG["bayesian_prior_weight"]
        
        stats = fct_evaluations.groupby("subcontractor_id").agg(
            rating_count=("rating", "count"),
            avg_rating=("rating", "mean"),
            avg_schedule_score=("schedule_score", "mean"),
            avg_safety_score=("safety_score", "mean"),
            avg_change_order_rate=("change_order_rate", "mean")
        ).reset_index()
        
        # Merge stats onto dim_subcontractors
        features_mart = pd.merge(dim_subcontractors, stats, on="subcontractor_id", how="left")
        features_mart["rating_count"] = features_mart["rating_count"].fillna(0).astype(int)
        features_mart["avg_rating"] = features_mart["avg_rating"].fillna(prior_mean).round(2)
        features_mart["avg_schedule_score"] = features_mart["avg_schedule_score"].fillna(prior_mean).round(2)
        features_mart["avg_safety_score"] = features_mart["avg_safety_score"].fillna(prior_mean).round(2)
        features_mart["avg_change_order_rate"] = features_mart["avg_change_order_rate"].fillna(0.05).round(3)
        
        # Compute Bayesian Adjusted Rating: (v * R + m * C) / (v + m)
        v = features_mart["rating_count"]
        R = features_mart["avg_rating"]
        features_mart["bayesian_rating"] = ((v * R + prior_weight * prior_mean) / (v + prior_weight)).round(2)
        
        print(f"[Pipeline: Gold] Star schema & feature mart constructed. Features Mart shape: {features_mart.shape}")
        return dim_subcontractors, dim_projects, fct_evaluations, features_mart

    def persist_to_warehouses(self, dim_subs, dim_prjs, fct_evls, features_mart):
        """
        Saves Gold & Silver tables to DuckDB (OLAP) and SQLite (universal portability),
        plus columnar Parquet files.
        """
        # 1. Parquet and CSV Analytical Exports
        try:
            dim_subs.to_parquet(PROCESSED_DATA_DIR / "dim_subcontractors.parquet", index=False)
            dim_prjs.to_parquet(PROCESSED_DATA_DIR / "dim_projects.parquet", index=False)
            fct_evls.to_parquet(PROCESSED_DATA_DIR / "fct_evaluations.parquet", index=False)
            features_mart.to_parquet(PROCESSED_DATA_DIR / "features_recommender_mart.parquet", index=False)
            print("[Pipeline: Storage] Parquet files exported to data/processed/")
        except Exception as e:
            print(f"[Pipeline: Storage Parquet Note] {e}")
            
        dim_subs.to_csv(PROCESSED_DATA_DIR / "dim_subcontractors.csv", index=False)
        dim_prjs.to_csv(PROCESSED_DATA_DIR / "dim_projects.csv", index=False)
        fct_evls.to_csv(PROCESSED_DATA_DIR / "fct_evaluations.csv", index=False)
        features_mart.to_csv(PROCESSED_DATA_DIR / "features_recommender_mart.csv", index=False)
        print("[Pipeline: Storage] CSV files exported to data/processed/")
        
        # 2. DuckDB DWH
        try:
            con = duckdb.connect(self.db_path)
            con.execute("CREATE OR REPLACE TABLE dim_subcontractors AS SELECT * FROM dim_subs")
            con.execute("CREATE OR REPLACE TABLE dim_projects AS SELECT * FROM dim_prjs")
            con.execute("CREATE OR REPLACE TABLE fct_evaluations AS SELECT * FROM fct_evls")
            con.execute("CREATE OR REPLACE TABLE features_recommender_mart AS SELECT * FROM features_mart")
            con.close()
            print(f"[Pipeline: Warehouse] Successfully stored tables in DuckDB at: {self.db_path}")
        except Exception as e:
            print(f"[Pipeline: Warehouse DuckDB Note] {e}")
            
        # 3. SQLite DWH (Always available & portable)
        sqlite_con = sqlite3.connect(self.sqlite_path)
        # Convert timestamp to string for sqlite compatibility
        fct_evls_sqlite = fct_evls.copy()
        fct_evls_sqlite["timestamp"] = fct_evls_sqlite["timestamp"].astype(str)
        
        dim_subs.to_sql("dim_subcontractors", sqlite_con, if_exists="replace", index=False)
        dim_prjs.to_sql("dim_projects", sqlite_con, if_exists="replace", index=False)
        fct_evls_sqlite.to_sql("fct_evaluations", sqlite_con, if_exists="replace", index=False)
        features_mart.to_sql("features_recommender_mart", sqlite_con, if_exists="replace", index=False)
        sqlite_con.close()
        print(f"[Pipeline: Warehouse] Successfully stored tables in SQLite at: {self.sqlite_path}")

    def run_full_pipeline(self):
        """Runs the complete Medallion pipeline end-to-end."""
        print("=" * 60)
        print("Starting ConstructRec Medallion Pipeline Execution")
        print("=" * 60)
        df_subs_raw, df_prjs_raw, df_evls_raw = self.run_bronze_ingestion()
        df_subs_silver, df_prjs_silver, df_evls_silver = self.run_silver_transformation(
            df_subs_raw, df_prjs_raw, df_evls_raw
        )
        dim_subs, dim_prjs, fct_evls, features_mart = self.run_gold_modeling(
            df_subs_silver, df_prjs_silver, df_evls_silver
        )
        self.persist_to_warehouses(dim_subs, dim_prjs, fct_evls, features_mart)
        print("=" * 60)
        print("Pipeline Execution Completed Successfully!")
        print("=" * 60)
        return dim_subs, dim_prjs, fct_evls, features_mart

if __name__ == "__main__":
    pipeline = MedallionPipeline()
    pipeline.run_full_pipeline()
