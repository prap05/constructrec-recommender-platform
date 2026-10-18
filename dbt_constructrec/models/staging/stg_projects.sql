-- Staging model for Projects (Silver layer transformation)
WITH raw_projects AS (
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
)
SELECT * FROM raw_projects
