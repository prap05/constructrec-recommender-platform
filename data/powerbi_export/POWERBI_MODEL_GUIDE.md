# ConstructRec: Power BI Reporting & DAX Guide

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
