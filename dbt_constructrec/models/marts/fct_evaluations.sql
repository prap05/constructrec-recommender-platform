-- Fact table: Project Evaluations & Ratings (Gold Layer)
SELECT 
    evaluation_id,
    project_id,
    subcontractor_id,
    rating,
    schedule_score,
    safety_score,
    change_order_rate,
    recommendation_confirmed,
    evaluation_timestamp
FROM {{ ref('stg_evaluations') }}
