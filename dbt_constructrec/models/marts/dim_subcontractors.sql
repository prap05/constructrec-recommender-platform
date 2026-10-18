-- Dimension table: Subcontractors (Gold Layer)
SELECT 
    subcontractor_id,
    company_name,
    trade,
    primary_location,
    safety_emr,
    bonding_capacity_usd,
    experience_years,
    active_crew_count,
    certifications_str,
    specialties_str,
    safety_tier,
    is_safety_compliant,
    is_active
FROM {{ ref('stg_subcontractors') }}
