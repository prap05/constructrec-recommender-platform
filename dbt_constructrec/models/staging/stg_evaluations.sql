-- Staging model for Historical Job Evaluations (Silver layer transformation)
WITH raw_evaluations AS (
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
)
SELECT * FROM raw_evaluations
