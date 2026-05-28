from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.models import Variable
import pandas as pd

connect = Variable.get("GP_Dash_connect")
pg_hook  = PostgresHook.get_hook(connect)
ikg_schema_name = Variable.get("IKG_DASHBOARD_SCHEMA")


# ─────────────────────────────────────────────────────────────────────────────
# Existing functions (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def get_metadata() -> pd.DataFrame:
    sql = f"select * from {ikg_schema_name}.ikg_dictionary_metadata_auto_refresh;"
    return pg_hook.get_pandas_df(sql)


def get_metadata_rules() -> pd.DataFrame:
    sql = f"""select r.*, a.time_horizon_in_weeks, a.insight_eval_type, a.metric_1
            from {ikg_schema_name}.odm_rule_metadata_auto_refresh r
            left join {ikg_schema_name}.attribution_insights_metrics_metadata_ikg a
            on r.insight_type = a.insight_type;"""
    return pg_hook.get_pandas_df(sql)


def get_business_descriptions() -> pd.DataFrame:
    sql = f"select * from {ikg_schema_name}.ikg_attribute_description;"
    return pg_hook.get_pandas_df(sql)


def get_profile_tables() -> pd.DataFrame:
    sql = (
        f"select distinct target_table "
        f"from {ikg_schema_name}.ikg_table_lineage_metadata_auto_refresh "
        f"where target_table like '%_profile_curr_ikg';"
    )
    return pg_hook.get_pandas_df(sql)


def get_profile_table_lineage(profile_table) -> pd.DataFrame:
    sql = f"select * from {ikg_schema_name}.ikg_lineage_{profile_table} order by order_index desc;"
    return pg_hook.get_pandas_df(sql)


def get_insight_details() -> pd.DataFrame:
    sql = (
        f"select insight_type, target_type, profile_table, insight_type_label, "
        f"description, sample_narrative "
        f"from {ikg_schema_name}.insights_dashboard_details;"
    )
    return pg_hook.get_pandas_df(sql)


def get_narrative_columns_for_insight(insight_type: str) -> list:
    """
    Returns distinct column_names from ikg_dictionary_metadata_auto_refresh
    where nlg_insight_types contains the given insight_type and is_nlg = 'Y'.
    """
    safe = str(insight_type).replace("'", "''")
    sql = f"""
        SELECT DISTINCT column_name
        FROM {ikg_schema_name}.ikg_dictionary_metadata_auto_refresh m
        WHERE m.nlg_insight_types ILIKE '%{safe}%'
          AND is_nlg = 'Y'
        ORDER BY column_name;
    """
    try:
        df = pg_hook.get_pandas_df(sql)
        if df.empty:
            return []
        return df["column_name"].dropna().astype(str).tolist()
    except Exception:
        return []


def get_sample_narrative_for_insight(insight_type: str) -> str | None:
    """
    Returns the sample_narrative from insights_dashboard_details
    for the given insight_type, or None if not found.
    """
    safe = str(insight_type).replace("'", "''")
    sql = f"""
        SELECT sample_narrative
        FROM {ikg_schema_name}.insights_dashboard_details
        WHERE insight_type = '{safe}'
        LIMIT 1;
    """
    try:
        df = pg_hook.get_pandas_df(sql)
        if df.empty or df["sample_narrative"].iloc[0] is None:
            return None
        val = str(df["sample_narrative"].iloc[0]).strip()
        return val if val and val.lower() not in ("none", "null", "") else None
    except Exception:
        return None


def get_metadata_transformed() -> pd.DataFrame:
    df_metadata     = get_metadata()
    df2_descriptions = get_business_descriptions()
    df = pd.merge(df_metadata, df2_descriptions, on=["table_name", "column_name"], how="left")
    df.rename(columns={
        "is_cid": "Contains CID", "column_name": "Column Name",
        "table_name": "Table Name", "data_type": "Data Type",
        "is_odm": "ODM Rule Impact", "is_nlg": "Narrative Impact",
        "description": "Description", "business_attribute_name": "Business Attribute Name",
    }, inplace=True)
    df["ODM Rule Impact"] = df["ODM Rule Impact"].map({"Y": "Yes", "N": "No"})
    df["Narrative Impact"] = df["Narrative Impact"].map({"Y": "Yes", "N": "No"})
    df["Contains CID"]    = df["Contains CID"].map({"Y": "Yes", "": "No"})
    df["odm_insight_types"] = df["odm_insight_types"].fillna("None")
    df["nlg_insight_types"] = df["nlg_insight_types"].fillna("None")
    df["nlg_target_type"]   = df["nlg_target_type"].fillna(df["odm_target_type"])
    df["nlg_target_type"]   = df["nlg_target_type"].fillna("None")
    df["odm_target_type"]   = df["odm_target_type"].fillna("None")
    return df


def get_rules_transformed() -> pd.DataFrame:
    df = get_metadata_rules()
    for c in ['metric_name', 'metric_type']:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.strip("'").str.strip('"')
    df['is_active'] = df['is_active'].map({'Y': 'Yes', 'N': 'No'})
    df["time_horizon_in_weeks"] = df["time_horizon_in_weeks"].fillna("None").apply(
        lambda x: str(int(x)) if isinstance(x, (int, float)) and not pd.isna(x) else "None"
    )
    vary_cols  = [c for c in ['rule_column', 'metric_1'] if c in df.columns]
    group_cols = [c for c in df.columns if c not in vary_cols]
    agg_dict   = {c: lambda s: ', '.join(sorted(set(s.dropna().astype(str)))) for c in vary_cols}
    return df.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()


# ─────────────────────────────────────────────────────────────────────────────
# Column lineage  (used by column_lineage_tab.py)
# ─────────────────────────────────────────────────────────────────────────────

def get_all_target_columns() -> pd.DataFrame:
    sql = f"""
        SELECT DISTINCT target_column
        FROM {ikg_schema_name}.ikg_column_lineage_master_auto_refresh
        WHERE target_column IS NOT NULL AND TRIM(target_column) <> ''
        ORDER BY target_column;
    """
    return pg_hook.get_pandas_df(sql)


def get_tables_for_column(target_column: str) -> pd.DataFrame:
    safe = str(target_column).replace("'", "''")
    sql = f"""
        SELECT DISTINCT target_table
        FROM {ikg_schema_name}.ikg_column_lineage_master_auto_refresh
        WHERE target_column = '{safe}'
          AND target_table IS NOT NULL AND TRIM(target_table) <> ''
        ORDER BY target_table;
    """
    return pg_hook.get_pandas_df(sql)


def get_column_lineage_upstream(target_column: str, target_table: str) -> pd.DataFrame:
    """Single WITH RECURSIVE — required by Greenplum."""
    sc = str(target_column).replace("'", "''")
    st = str(target_table).replace("'", "''")
    sql = f"""
        WITH RECURSIVE upstream AS (
            SELECT target_table, target_schema, target_column,
                   source_table, source_schema, source_column,
                   logic, process, path, web_url,
                   sub_target_table, sub_target_schema, sql_process,
                   0 AS depth
            FROM {ikg_schema_name}.ikg_column_lineage_master_auto_refresh
            WHERE target_table  = '{st}'
              AND target_column = '{sc}'
            UNION ALL
            SELECT cl.target_table, cl.target_schema, cl.target_column,
                   cl.source_table, cl.source_schema, cl.source_column,
                   cl.logic, cl.process, cl.path, cl.web_url,
                   cl.sub_target_table, cl.sub_target_schema, cl.sql_process,
                   u.depth + 1
            FROM {ikg_schema_name}.ikg_column_lineage_master_auto_refresh cl
            JOIN upstream u
              ON cl.target_table  = u.source_table
             AND cl.target_column = u.source_column
            WHERE u.depth < 6
        )
        SELECT DISTINCT * FROM upstream;
    """
    return pg_hook.get_pandas_df(sql)


# ─────────────────────────────────────────────────────────────────────────────
# Insight lineage  (used by insight_lineage_tab.py)
# ─────────────────────────────────────────────────────────────────────────────

def get_all_insight_types() -> pd.DataFrame:
    sql = f"""
        SELECT DISTINCT insight_type
        FROM {ikg_schema_name}.odm_rule_metadata_auto_refresh
        WHERE insight_type IS NOT NULL AND TRIM(insight_type) <> ''
        ORDER BY insight_type;
    """
    return pg_hook.get_pandas_df(sql)


def get_insight_rule_meta(insight_type: str) -> pd.DataFrame:
    safe = str(insight_type).replace("'", "''")
    sql = f"""
        SELECT *
        FROM {ikg_schema_name}.odm_rule_metadata_auto_refresh
        WHERE insight_type = '{safe}';
    """
    return pg_hook.get_pandas_df(sql)


def get_insight_column_lineage(insight_type: str) -> pd.DataFrame:
    """Single WITH RECURSIVE — required by Greenplum."""
    safe = str(insight_type).replace("'", "''")
    sql = f"""
        WITH RECURSIVE upstream AS (
            SELECT cl.target_table, cl.target_schema, cl.target_column,
                   cl.source_table, cl.source_schema, cl.source_column,
                   cl.logic, cl.process, cl.path, cl.web_url,
                   cl.sub_target_table, cl.sub_target_schema, cl.sql_process,
                   0 AS depth
            FROM {ikg_schema_name}.ikg_column_lineage_master_auto_refresh cl
            WHERE cl.target_column IN (
                SELECT DISTINCT rule_column
                FROM {ikg_schema_name}.odm_rule_metadata_auto_refresh
                WHERE insight_type = '{safe}'
                  AND rule_column IS NOT NULL
                  AND TRIM(rule_column) <> ''
            )
            UNION ALL
            SELECT cl.target_table, cl.target_schema, cl.target_column,
                   cl.source_table, cl.source_schema, cl.source_column,
                   cl.logic, cl.process, cl.path, cl.web_url,
                   cl.sub_target_table, cl.sub_target_schema, cl.sql_process,
                   u.depth + 1
            FROM {ikg_schema_name}.ikg_column_lineage_master_auto_refresh cl
            JOIN upstream u
              ON cl.target_table  = u.source_table
             AND cl.target_column = u.source_column
            WHERE u.depth < 5
        )
        SELECT DISTINCT
            u.*,
            CASE WHEN u.target_column IN (
                SELECT DISTINCT rule_column
                FROM {ikg_schema_name}.odm_rule_metadata_auto_refresh
                WHERE insight_type = '{safe}'
            ) THEN 'Y' ELSE 'N' END AS is_rule_col
        FROM upstream u;
    """
    return pg_hook.get_pandas_df(sql)


# ─────────────────────────────────────────────────────────────────────────────
# Table lineage  (used by table_lineage_er_tab.py)
# ─────────────────────────────────────────────────────────────────────────────

def is_profile_table(table_name: str) -> bool:
    """Returns True if the table name follows the *_profile_curr_ikg pattern."""
    return str(table_name).lower().strip().endswith("_profile_curr_ikg")


def get_all_lineage_tables() -> pd.DataFrame:
    """
    Returns all distinct table names from the master lineage table —
    both target tables and source tables — for use in a searchable dropdown.
    """
    sql = f"""
        SELECT DISTINCT table_name FROM (
            SELECT TRIM(target_table) AS table_name
            FROM {ikg_schema_name}.ikg_table_lineage_metadata_auto_refresh
            WHERE target_table IS NOT NULL AND TRIM(target_table) <> ''
            UNION
            SELECT TRIM(source_table)
            FROM {ikg_schema_name}.ikg_table_lineage_metadata_auto_refresh
            WHERE source_table IS NOT NULL AND TRIM(source_table) <> ''
        ) t
        ORDER BY table_name;
    """
    return pg_hook.get_pandas_df(sql)


def get_table_lineage_for_profile(table_name: str) -> pd.DataFrame:
    """
    For profile tables (*_profile_curr_ikg) query the pre-built specific lineage
    table: ikg_lineage_<table_name>.
    Columns: order_index, root_profile_table, target_table, source_schema,
             source_table, process, filename, filepath.
    """
    safe_name = str(table_name).lower().strip().replace("'", "''")
    lineage_table = f"ikg_lineage_{safe_name}"
    sql = f"""
        SELECT order_index, root_profile_table, target_table,
               source_schema, source_table, process, filename, filepath
        FROM {ikg_schema_name}.{lineage_table}
        WHERE source_table IS NOT NULL AND TRIM(source_table) <> ''
        ORDER BY order_index ASC;
    """
    return pg_hook.get_pandas_df(sql)


def get_table_lineage_upstream(table_name: str) -> pd.DataFrame:
    """
    Upstream lineage for any table: walks backwards through
    ikg_table_lineage_metadata_auto_refresh using a single WITH RECURSIVE.
    Returns all rows where the given table (or its ancestors) is the target.
    """
    safe = str(table_name).replace("'", "''")
    sql = f"""
        WITH RECURSIVE upstream AS (
            SELECT target_table, source_table, source_schema,
                   process, filename, filepath, 0 AS depth
            FROM {ikg_schema_name}.ikg_table_lineage_metadata_auto_refresh
            WHERE LOWER(TRIM(target_table)) = LOWER(TRIM('{safe}'))
              AND source_table IS NOT NULL AND TRIM(source_table) <> ''
            UNION ALL
            SELECT m.target_table, m.source_table, m.source_schema,
                   m.process, m.filename, m.filepath, u.depth + 1
            FROM {ikg_schema_name}.ikg_table_lineage_metadata_auto_refresh m
            JOIN upstream u
              ON LOWER(TRIM(m.target_table)) = LOWER(TRIM(u.source_table))
            WHERE u.depth < 6
              AND m.source_table IS NOT NULL AND TRIM(m.source_table) <> ''
        )
        SELECT DISTINCT * FROM upstream;
    """
    return pg_hook.get_pandas_df(sql)


def get_table_lineage_downstream(table_name: str) -> pd.DataFrame:
    """
    Downstream lineage for any table: walks forward through
    ikg_table_lineage_metadata_auto_refresh using a single WITH RECURSIVE.
    Returns all rows where the given table (or its descendants) is the source.
    """
    safe = str(table_name).replace("'", "''")
    sql = f"""
        WITH RECURSIVE downstream AS (
            SELECT target_table, source_table, source_schema,
                   process, filename, filepath, 0 AS depth
            FROM {ikg_schema_name}.ikg_table_lineage_metadata_auto_refresh
            WHERE LOWER(TRIM(source_table)) = LOWER(TRIM('{safe}'))
              AND target_table IS NOT NULL AND TRIM(target_table) <> ''
            UNION ALL
            SELECT m.target_table, m.source_table, m.source_schema,
                   m.process, m.filename, m.filepath, d.depth + 1
            FROM {ikg_schema_name}.ikg_table_lineage_metadata_auto_refresh m
            JOIN downstream d
              ON LOWER(TRIM(m.source_table)) = LOWER(TRIM(d.target_table))
            WHERE d.depth < 4
              AND m.target_table IS NOT NULL AND TRIM(m.target_table) <> ''
        )
        SELECT DISTINCT * FROM downstream;
    """
    return pg_hook.get_pandas_df(sql)
