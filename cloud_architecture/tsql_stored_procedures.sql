-- ==============================================================================
-- Microsoft SQL Server / T-SQL Production Scripts for vConstruct
-- Stored Procedures, Partitioning & Columnstore Analytical Indexing
-- ==============================================================================

USE DPR_Construction_DWH;
GO

-- 1. Create Non-Clustered Columnstore Index for High-Speed Aggregations
IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'NCCI_FctProjectEvaluations')
BEGIN
    CREATE NONCLUSTERED COLUMNSTORE INDEX NCCI_FctProjectEvaluations
    ON dbo.fct_project_evaluations (
        evaluation_id, project_id, subcontractor_id, rating, schedule_score, safety_score, change_order_rate
    );
END
GO

-- 2. Stored Procedure: sp_GetTopContractorsForBid
-- Employs T-SQL CTEs, window ranking, and parameter safety checks
CREATE OR ALTER PROCEDURE dbo.sp_GetTopContractorsForBid
    @RequiredTrade          NVARCHAR(100),
    @Location               NVARCHAR(100),
    @BudgetUSD              DECIMAL(18, 2),
    @StrictSafetyMandate    BIT = 0,
    @TopK                   INT = 5
AS
BEGIN
    SET NOCOUNT ON;

    DECLARE @MaxAllowedEMR DECIMAL(4, 2) = CASE WHEN @StrictSafetyMandate = 1 THEN 0.90 ELSE 1.05 END;

    WITH ContractorMetrics AS (
        SELECT 
            s.subcontractor_id,
            s.company_name,
            s.trade,
            s.primary_location,
            s.safety_emr,
            s.safety_tier,
            s.bonding_capacity_usd,
            s.experience_years,
            s.certifications_str,
            ISNULL(fm.rating_count, 0) AS rating_count,
            ISNULL(fm.avg_rating, 3.8) AS avg_rating,
            ISNULL(fm.bayesian_rating, 3.8) AS bayesian_rating,
            -- Proximity match weight
            CASE WHEN s.primary_location = @Location THEN 1.0 ELSE 0.0 END AS is_local,
            -- Financial solvency check
            CASE WHEN s.bonding_capacity_usd >= (@BudgetUSD * 0.5) THEN 1.0 ELSE 0.0 END AS has_bonding_capacity
        FROM dbo.dim_subcontractors s
        LEFT JOIN dbo.features_recommender_mart fm ON s.subcontractor_id = fm.subcontractor_id
        WHERE s.is_active = 1
          AND s.safety_emr <= @MaxAllowedEMR
          AND LOWER(s.trade) = LOWER(@RequiredTrade)
    ),
    ScoredContractors AS (
        SELECT 
            subcontractor_id,
            company_name,
            trade,
            primary_location,
            safety_emr,
            safety_tier,
            bonding_capacity_usd,
            experience_years,
            certifications_str,
            rating_count,
            avg_rating,
            bayesian_rating,
            -- Composite Rule-Based Score in T-SQL
            ROUND(
                (0.50 * (bayesian_rating / 5.0)) + 
                (0.20 * (1.20 - safety_emr)) + 
                (0.15 * is_local) + 
                (0.15 * has_bonding_capacity),
                3
            ) AS composite_fit_score
        FROM ContractorMetrics
    )
    SELECT TOP (@TopK)
        ROW_NUMBER() OVER (ORDER BY composite_fit_score DESC, bayesian_rating DESC) AS recommendation_rank,
        subcontractor_id,
        company_name,
        trade,
        primary_location,
        safety_emr,
        safety_tier,
        bonding_capacity_usd,
        experience_years,
        certifications_str,
        rating_count,
        bayesian_rating,
        composite_fit_score
    FROM ScoredContractors
    ORDER BY composite_fit_score DESC, bayesian_rating DESC;
END;
GO
