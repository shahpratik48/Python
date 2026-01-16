-- IKG table lineage execution plan (Greenplum/Postgres).
-- Input values:
--   source_tables: comma-separated list of source tables (table only).
--   target_table: optional target table (schema.table or table). Leave blank to traverse to the end.
--
-- Ordering logic:
--   step_index = 1 => targets with no dependencies on other targets in scope.
--   step_index increases by dependency depth (topological level).
--
-- Replace the values in the params CTE before running.
WITH RECURSIVE
params AS (
    SELECT
        'table1,table2'::text AS source_tables,
        ''::text AS target_table
),
source_list AS (
    SELECT
        trim(source_raw) AS source_raw
    FROM params,
        regexp_split_to_table(params.source_tables, '\s*,\s*') AS source_raw
),
source_inputs AS (
    SELECT lower(source_raw) AS source_table
    FROM source_list
    WHERE source_raw <> ''
),
edges AS (
    SELECT
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
        ON e.source_table = s.source_table
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
target_nodes AS (
    SELECT DISTINCT target_table
    FROM filtered_walk
),
dependency_edges AS (
    SELECT
        w.source_table,
        w.target_table
    FROM filtered_walk w
    JOIN target_nodes src
        ON w.source_table = src.target_table
    JOIN target_nodes tgt
        ON w.target_table = tgt.target_table
),
root_targets AS (
    SELECT n.target_table
    FROM target_nodes n
    LEFT JOIN dependency_edges d
        ON d.target_table = n.target_table
    WHERE d.target_table IS NULL
),
levels AS (
    SELECT
        r.target_table,
        1 AS level,
        ARRAY[r.target_table] AS path
    FROM root_targets r
    UNION ALL
    SELECT
        d.target_table,
        l.level + 1,
        l.path || d.target_table
    FROM levels l
    JOIN dependency_edges d
        ON d.source_table = l.target_table
    WHERE NOT (d.target_table = ANY(l.path))
),
execution_plan AS (
    SELECT
        max(level) AS run_order,
        l.target_table,
        min(process) AS process,
        min(filename) AS filename,
        min(filepath) AS filepath
    FROM levels l
    JOIN filtered_walk w
        ON w.target_table = l.target_table
    GROUP BY l.target_table
)
SELECT
    run_order AS step_index,
    process,
    filename AS script,
    filepath,
    target_table
FROM execution_plan
ORDER BY run_order, process, filename, target_table;
