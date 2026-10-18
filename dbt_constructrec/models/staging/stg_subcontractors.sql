-- Staging model for Subcontractors (Silver layer transformation)
WITH raw_subs AS (
    SELECT 
        subcontractor_id,
        TRIM(company_name) AS company_name,
        TRIM(trade) AS trade,
        TRIM(primary_location) AS primary_location,
        ROUND(CAST(safety_emr AS DOUBLE), 2) AS safety_emr,
        CAST(bonding_capacity_usd AS BIGINT) AS bonding_capacity_usd,
        CAST(experience_years AS INTEGER) AS experience_years,
        CAST(active_crew_count AS INTEGER) AS active_crew_count,
        certifications_str,
        specialties_str,
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
)
SELECT * FROM raw_subs
