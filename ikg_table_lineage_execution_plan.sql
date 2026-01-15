-- IKG table lineage execution plan (Greenplum/Postgres).
-- Input values:
--   source_tables: comma-separated list of source tables (schema.table or table).
--   target_table: optional target table (schema.table or table). Leave blank to traverse to the end.
--
-- Replace the values in the params CTE before running.
WITH RECURSIVE
params AS (
    SELECT
        'schema.table1,table2'::text AS source_tables,
        ''::text AS target_table
),
source_list AS (
    SELECT
        trim(source_raw) AS source_raw
    FROM params,
        regexp_split_to_table(params.source_tables, '\s*,\s*') AS source_raw
),
source_inputs AS (
    SELECT
        NULLIF(split_part(source_raw, '.', 1), '') AS source_schema,
        CASE
            WHEN source_raw LIKE '%.%' THEN split_part(source_raw, '.', 2)
            ELSE source_raw
        END AS source_table
    FROM source_list
    WHERE source_raw <> ''
),
edges AS (
    SELECT
        lower(source_schema) AS source_schema,
        lower(source_table) AS source_table,
        lower(target_table) AS target_table,
        filename,
        filepath,
        process
    FROM sandbox_prj_smart_insights.ikg_table_lineage_metadata_auto_refresh
    WHERE source_table IS NOT NULL
),
start_edges AS (
    SELECT e.*
    FROM edges e
    JOIN source_inputs s
        ON e.source_table = lower(s.source_table)
        AND (s.source_schema IS NULL OR e.source_schema = lower(s.source_schema))
),
walk AS (
    SELECT
        1 AS depth,
        e.*,
        ARRAY[e.target_table] AS target_path
    FROM start_edges e
    UNION ALL
    SELECT
        w.depth + 1,
        e.*,
        w.target_path || e.target_table
    FROM walk w
    JOIN edges e
        ON e.source_table = w.target_table
    WHERE NOT (e.target_table = ANY(w.target_path))
),
target_input AS (
    SELECT NULLIF(lower(trim((SELECT target_table FROM params))), '') AS target_raw
),
target_param AS (
    SELECT
        CASE
            WHEN target_raw LIKE '%.%' THEN split_part(target_raw, '.', 2)
            ELSE target_raw
        END AS target_table
    FROM target_input
),
target_paths AS (
    SELECT DISTINCT w.target_path
    FROM walk w
    JOIN target_param t
        ON t.target_table IS NOT NULL
        AND w.target_table = t.target_table
),
filtered_walk AS (
    SELECT w.*
    FROM walk w
    JOIN target_param t
        ON t.target_table IS NULL
    UNION ALL
    SELECT w.*
    FROM walk w
    JOIN target_paths p
        ON p.target_path[1:array_length(w.target_path, 1)] = w.target_path
),
execution_plan AS (
    SELECT
        min(depth) AS run_order,
        target_table,
        min(process) AS process,
        min(filename) AS filename,
        min(filepath) AS filepath
    FROM filtered_walk
    GROUP BY target_table
)
SELECT
    run_order AS step_index,
    process,
    filename AS script,
    filepath,
    target_table
FROM execution_plan
ORDER BY run_order, process, filename, target_table;
