-- Dimension table: Projects (Gold Layer)
SELECT 
    project_id,
    project_name,
    sector,
    location,
    budget_usd,
    budget_millions,
    sq_footage,
    required_trade,
    bim_lod_required,
    strict_safety_mandate,
    status
FROM {{ ref('stg_projects') }}
