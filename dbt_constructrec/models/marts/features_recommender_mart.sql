-- Feature Store Mart for Recommender System (Gold Layer)
-- Computes aggregated contractor performance metrics and Bayesian adjusted rating
WITH sub_stats AS (
    SELECT 
        subcontractor_id,
        COUNT(evaluation_id) AS rating_count,
        ROUND(AVG(rating), 2) AS avg_rating,
        ROUND(AVG(schedule_score), 2) AS avg_schedule_score,
        ROUND(AVG(safety_score), 2) AS avg_safety_score,
        ROUND(AVG(change_order_rate), 3) AS avg_change_order_rate
    FROM {{ ref('fct_evaluations') }}
    GROUP BY subcontractor_id
),
base_subs AS (
    SELECT * FROM {{ ref('dim_subcontractors') }}
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
    -- Bayesian Adjusted Rating: (v * R + m * C) / (v + m) where m=5, C=3.8
    ROUND(
        (COALESCE(st.rating_count, 0) * COALESCE(st.avg_rating, 3.8) + 5.0 * 3.8) 
        / (COALESCE(st.rating_count, 0) + 5.0),
        2
    ) AS bayesian_rating
FROM base_subs s
LEFT JOIN sub_stats st ON s.subcontractor_id = st.subcontractor_id
