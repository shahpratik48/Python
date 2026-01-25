-- Insight Flow Tracking Query
-- Tracks ODM release details with exclusion status and go-live dates

WITH max_exclusion_dates AS (
    -- Get the maximum last_upd_dte for each insight_type
    SELECT 
        insight_type,
        MAX(last_upd_dte) AS max_last_upd_dte
    FROM sandbox_prj_smart_insights.odm_exclusion_insight_type
    GROUP BY insight_type
),
latest_exclusion_records AS (
    -- Get the latest record for each insight_type based on max_last_upd_dte
    SELECT 
        b.*
    FROM sandbox_prj_smart_insights.odm_exclusion_insight_type b
    INNER JOIN max_exclusion_dates m
        ON b.insight_type = m.insight_type
        AND b.last_upd_dte = m.max_last_upd_dte
)
SELECT 
    -- All columns from table a (odm_release_details)
    a.*,
    
    -- All columns from table b (odm_exclusion_insight_type) with alias to avoid conflicts
    b.insight_type AS exclusion_insight_type,
    b.last_upd_dte AS exclusion_last_upd_dte,
    b.is_curr AS exclusion_is_curr,
    
    -- Go_live column logic
    CASE 
        -- If insight_type exists in exclusion table with max date and is_curr = 1
        WHEN b.insight_type IS NOT NULL AND b.is_curr = 1 THEN 'No'
        -- If insight_type exists in exclusion table with max date and is_curr = 0
        WHEN b.insight_type IS NOT NULL AND b.is_curr = 0 THEN 'Yes'
        -- If insight_type is not in exclusion table (null)
        WHEN b.insight_type IS NULL THEN 'Yes'
        ELSE 'Yes'
    END AS go_live,
    
    -- Go_live_date column logic
    CASE 
        -- If insight_type exists in exclusion table with max date and is_curr = 1
        WHEN b.insight_type IS NOT NULL AND b.is_curr = 1 THEN NULL
        -- If insight_type exists in exclusion table with max date and is_curr = 0
        WHEN b.insight_type IS NOT NULL AND b.is_curr = 0 THEN b.last_upd_dte
        -- If insight_type is not in exclusion table (null)
        WHEN b.insight_type IS NULL THEN a.prod_release_date
        ELSE a.prod_release_date
    END AS go_live_date

FROM sandbox_prj_smart_insights.odm_release_details a

LEFT JOIN latest_exclusion_records b
    ON a.rule_name = b.insight_type

WHERE 
    a.rule_name IS NOT NULL 
    AND a.rule_name != ''
    AND a.change_type = 'added'

ORDER BY 
    a.iteration_end_date DESC,
    a.issue_id,
    a.rule_name;
