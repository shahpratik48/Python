-- IKG table lineage execution plan (Greenplum/Postgres).
-- Input values:
--   source_tables: comma-separated list of source tables (table only).
--   target_table: optional target table (schema.table or table). Leave blank to traverse to the end.
--
-- Ordering logic:
--   step_index = 1 => targets with no dependencies on other targets in scope.
--   step_index increases by dependency depth (topological level).
--   Depth is capped to avoid cycles (increase if needed).
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
edges_raw AS (
    SELECT
        lower(source_table) AS source_table,
        lower(target_table) AS target_table,
        filename,
        filepath,
        process
    FROM sandbox_prj_smart_insights.ikg_table_lineage_metadata_auto_refresh
    WHERE source_table IS NOT NULL
),
edges_distinct AS (
    SELECT DISTINCT
        source_table,
        target_table
    FROM edges_raw
),
target_info AS (
    SELECT
        target_table,
        min(process) AS process,
        min(filename) AS filename,
        min(filepath) AS filepath
    FROM edges_raw
    GROUP BY target_table
),
forward_edges AS (
    SELECT
        e.source_table,
        e.target_table
    FROM edges_distinct e
    JOIN source_inputs s
        ON e.source_table = s.source_table
    UNION
    SELECT
        e.source_table,
        e.target_table
    FROM edges_distinct e
    JOIN forward_edges f
        ON e.source_table = f.target_table
),
forward_targets AS (
    SELECT DISTINCT target_table
    FROM forward_edges
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
target_flag AS (
    SELECT
        target_table,
        (target_table IS NOT NULL) AS has_target
    FROM target_param
),
backward_nodes AS (
    SELECT
        target_table AS node
    FROM target_param
    WHERE target_table IS NOT NULL
    UNION
    SELECT
        e.source_table AS node
    FROM edges_distinct e
    JOIN backward_nodes b
        ON e.target_table = b.node
),
scoped_targets AS (
    SELECT f.target_table
    FROM forward_targets f
    CROSS JOIN target_flag tf
    WHERE (NOT tf.has_target)
        OR (tf.has_target AND f.target_table IN (SELECT node FROM backward_nodes))
),
target_nodes AS (
    SELECT DISTINCT target_table
    FROM scoped_targets
),
dependency_edges AS (
    SELECT
        w.source_table,
        w.target_table
    FROM edges_distinct w
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
        1 AS level
    FROM root_targets r
    UNION ALL
    SELECT
        d.target_table,
        l.level + 1
    FROM levels l
    JOIN dependency_edges d
        ON d.source_table = l.target_table
    WHERE l.level < 100
),
execution_plan AS (
    SELECT
        max(level) AS run_order,
        l.target_table,
        min(t.process) AS process,
        min(t.filename) AS filename,
        min(t.filepath) AS filepath
    FROM levels l
    JOIN target_info t
        ON t.target_table = l.target_table
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
