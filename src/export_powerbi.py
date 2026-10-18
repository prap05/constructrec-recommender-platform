"""
Power BI Integration & Star Schema Data Exporter.
Extracts Gold layer tables from DuckDB/SQLite, creates clean dimensional CSVs for Power BI,
and generates ready-to-use DAX formula definitions.
"""
import sys
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from config import POWERBI_EXPORT_DIR, SQLITE_DB_PATH, DB_PATH
from src.pipeline import MedallionPipeline

def export_powerbi_star_schema():
    """Generates clean Star Schema tables and DAX documentation for Power BI Desktop."""
    POWERBI_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Read from local SQLite/DuckDB
    con = sqlite3.connect(SQLITE_DB_PATH)
    
    # Check if tables exist, else trigger pipeline
    tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    if "dim_subcontractors" not in tables:
        con.close()
        pipeline = MedallionPipeline()
        pipeline.run_full_pipeline()
        con = sqlite3.connect(SQLITE_DB_PATH)
        
    dim_subs = pd.read_sql_query("SELECT * FROM dim_subcontractors", con)
    dim_prjs = pd.read_sql_query("SELECT * FROM dim_projects", con)
    fct_evls = pd.read_sql_query("SELECT * FROM fct_evaluations", con)
    features = pd.read_sql_query("SELECT * FROM features_recommender_mart", con)
    con.close()
    
    # Export Star Schema CSV files
    dim_subs.to_csv(POWERBI_EXPORT_DIR / "dim_subcontractors.csv", index=False)
    dim_prjs.to_csv(POWERBI_EXPORT_DIR / "dim_projects.csv", index=False)
    fct_evls.to_csv(POWERBI_EXPORT_DIR / "fct_project_evaluations.csv", index=False)
    features.to_csv(POWERBI_EXPORT_DIR / "features_recommender_mart.csv", index=False)
    
    # Write Power BI DAX & Data Modeling Documentation
    dax_guide_path = POWERBI_EXPORT_DIR / "POWERBI_MODEL_GUIDE.md"
    with open(dax_guide_path, "w") as f:
        f.write("""# ConstructRec: Power BI Reporting & DAX Guide

## 1. Star Schema Relationships
In Power BI Desktop Model View, create the following 1-to-Many relationships:
- `dim_subcontractors[subcontractor_id]` (1) ---> `fct_project_evaluations[subcontractor_id]` (*)
- `dim_projects[project_id]` (1) -------------> `fct_project_evaluations[project_id]` (*)
- `dim_subcontractors[subcontractor_id]` (1) ---> `features_recommender_mart[subcontractor_id]` (1)

---

## 2. Recommended DAX Measures for Dashboard

### Metric 1: Average Subcontractor Safety EMR
```dax
Avg Contractor Safety EMR = 
AVERAGE(dim_subcontractors[safety_emr])
```
*Goal: EMR <= 0.85 indicates industry-leading safety.*

### Metric 2: High Risk Subcontractor Count
```dax
High Risk Subcontractor Count = 
CALCULATE(
    COUNTROWS(dim_subcontractors),
    dim_subcontractors[safety_tier] = "Tier 3 (High Risk)"
)
```

### Metric 3: Subcontractor On-Time Schedule Rate %
```dax
Schedule Adherence % = 
DIVIDE(
    AVERAGE(fct_project_evaluations[schedule_score]),
    5.0,
    0
) * 100
```

### Metric 4: Average Change Order Rate %
```dax
Avg Change Order Rate % = 
AVERAGE(fct_project_evaluations[change_order_rate]) * 100
```

### Metric 5: Recommendation Confirmation Rate %
```dax
AI Rec Confirmation Rate % = 
DIVIDE(
    CALCULATE(COUNTROWS(fct_project_evaluations), fct_project_evaluations[recommendation_confirmed] = TRUE()),
    COUNTROWS(fct_project_evaluations),
    0
) * 100
```

---

## 3. Power BI Visuals to Build in Interview Demo
1. **Safety Card**: Average EMR & Tier 1 vs Tier 3 breakdown.
2. **Scatter Plot**: Subcontractor Experience (Years) vs Bayesian Rating, sized by Bonding Capacity.
3. **Map Visual**: Primary Location distribution of active certified contractors.
4. **Decomposition Tree**: Trade Package -> Location -> Quality Score breakdown.
""")
        
    print(f"[PowerBI Export] Exported 4 CSVs and DAX Guide to: {POWERBI_EXPORT_DIR}")

if __name__ == "__main__":
    export_powerbi_star_schema()
