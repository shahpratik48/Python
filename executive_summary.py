# dags/ikg/sprint_report/scripts/executive_summary.py
# -*- coding: utf-8 -*-
"""Executive Summary – shared logic for the Dash UI and the monthly DAG.

Environment detection
─────────────────────
The module uses the same two-phase detection pattern as the other scripts
in this folder:

  1. _HAS_AIRFLOW  – True when the Airflow package is importable.
  2. is_running_in_airflow()  – True only when _HAS_AIRFLOW AND the
     AIRFLOW_CTX_DAG_ID env var is set (Airflow injects this at run-time).

This distinction matters when a developer has Airflow installed locally:
_HAS_AIRFLOW would be True, but is_running_in_airflow() returns False so
credentials are still read from env vars / prompts rather than Airflow
Connections / Variables.

  Production / DAG  (is_running_in_airflow() = True)
    GP         →  PostgresHook  via  Variable.get("GP_Dash_connect")
    Schema     →  Variable.get("IKG_DASHBOARD_SCHEMA")
    OpenAI     →  Connection.get_connection_from_secrets("STAAT-DS-OPENAI-LLM")
    Run month  →  datetime.now()  (no user input)

  Local / Jupyter  (is_running_in_airflow() = False)
    GP         →  psycopg2 using env vars or getpass prompt:
                    GP_HOST  GP_PORT  GP_DB  GP_USER  GP_PASSWORD  GP_SCHEMA
    OpenAI     →  env vars:  OPENAI_API_KEY   OPENAI_BASE_URL
    Run month  →  user input (handled in notebook / __main__ block)

DAG entry point
───────────────
generate_executive_summary(**kwargs) is the callable registered in the DAG.
It resolves the current month automatically and saves the output to
  <this_script_dir>/output/executive_summary_YYYY-MM.txt

Dash layout
───────────
layout() is only wired when Dash / styles are importable (production UI).
It is never called from the DAG or the notebook.
"""

from __future__ import annotations

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from openai import OpenAI

from config import (
    GREENPLUM_HOST,
    GREENPLUM_PORT,
    GREENPLUM_DB,
    GREENPLUM_USER,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Airflow detection  (mirrors swat_issues_report.py convention)
# ---------------------------------------------------------------------------

try:
    from airflow.models import Connection, Variable                       # type: ignore
    from airflow.configuration import conf                                # type: ignore
    _HAS_AIRFLOW = True
except Exception:
    Variable   = None   # type: ignore
    Connection = None   # type: ignore
    _HAS_AIRFLOW = False


def is_running_in_airflow() -> bool:
    """True only when Airflow is installed AND we are inside a DAG run."""
    return _HAS_AIRFLOW and bool(os.environ.get("AIRFLOW_CTX_DAG_ID"))


# Dash / styles only exist inside the dashboard server.
try:
    from dash import dcc, html
    import dash_bootstrap_components as dbc
    from styles import release_dashboard_styles
    _DASH_AVAILABLE = True
except ImportError:
    _DASH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

MODEL_NAME  = "gpt-4.1"
MAX_TOKENS  = 12000
TEMPERATURE = 0.1

POSTGRES_CONN_ID_VAR    = "GP_Dash_connect"
IKG_SCHEMA_VAR          = "IKG_DASHBOARD_SCHEMA"
OPENAI_CONN_ID          = "STAAT-DS-OPENAI-LLM"
DEFAULT_OPENAI_BASE_URL = "https://cirruspl-staat-ste-dev-ai.openai.azure.com/openai/v1/"
DEFAULT_GP_SCHEMA       = "core_ikg"

_client: Optional[OpenAI] = None


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def _get_openai_client() -> OpenAI:
    """Return the shared OpenAI client, initialised lazily.

    Production : Airflow Connection  STAAT-DS-OPENAI-LLM
    Local      : env vars  OPENAI_API_KEY / OPENAI_BASE_URL
    """
    global _client
    if _client is not None:
        return _client
    if is_running_in_airflow():
        conn     = Connection.get_connection_from_secrets(OPENAI_CONN_ID)
        api_key  = conn.password
        base_url = conn.host
    else:
        api_key  = os.environ["OPENAI_API_KEY"]
        base_url = os.environ.get("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL)
    _client = OpenAI(api_key=api_key, base_url=base_url)
    return _client


def _get_gp_conn(allow_prompt: bool = True):
    """Return a live psycopg2 connection to Greenplum.

    Production : PostgresHook via Airflow Variable  GP_Dash_connect
    Local      : env vars  GP_HOST / GP_PORT / GP_DB / GP_USER / GP_PASSWORD
                 (falls back to getpass when allow_prompt=True)
    """
    if is_running_in_airflow():
        from airflow.providers.postgres.hooks.postgres import PostgresHook  # type: ignore
        conn_id = Variable.get(POSTGRES_CONN_ID_VAR)
        return PostgresHook(postgres_conn_id=conn_id).get_conn()

    import psycopg2
    host     = os.environ.get("GP_HOST", GREENPLUM_HOST)
    port     = int(os.environ.get("GP_PORT", GREENPLUM_PORT))
    dbname   = os.environ.get("GP_DB", GREENPLUM_DB)
    user     = os.environ.get("GP_USER", GREENPLUM_USER)
    password = os.environ.get("GP_PASSWORD")
    if not password:
        if allow_prompt:
            import getpass
            password = getpass.getpass("Enter Greenplum password: ")
        else:
            raise RuntimeError("GP_PASSWORD env var is required for non-interactive runs.")
    return psycopg2.connect(
        host=host, port=port, dbname=dbname, user=user, password=password,
    )


def _get_schema() -> str:
    """Return the IKG schema name.

    Production : Airflow Variable  IKG_DASHBOARD_SCHEMA
    Local      : env var  GP_SCHEMA  (default "core_ikg")
    """
    if is_running_in_airflow():
        return Variable.get(IKG_SCHEMA_VAR)
    return os.environ.get("GP_SCHEMA", DEFAULT_GP_SCHEMA)


# ---------------------------------------------------------------------------
# SQL helpers
# ---------------------------------------------------------------------------

def _build_month_query(month_value: str, schema: str) -> str:
    """SQL for the latest batch per iteration whose prod_release_date is in *month_value*.

    1. Identify every iteration_end_date in STAAT with prod_release_date in the month.
    2. For each iteration take MAX(batch) from both STAAT and ODM.
    3. LEFT JOIN ODM onto STAAT so stories with no ODM rows are still returned.
    """
    staat = f"{schema}.staat_insight_release"
    odm   = f"{schema}.odm_release_details"
    return f"""
    WITH target_iterations AS (
        SELECT
            iteration_end_date,
            MAX(batch) AS max_batch
        FROM {staat}
        WHERE TO_CHAR(prod_release_date::date, 'YYYY-MM') = '{month_value}'
        GROUP BY iteration_end_date
    ),
    staat_latest AS (
        SELECT s.*
        FROM {staat} s
        INNER JOIN target_iterations ti
            ON  s.iteration_end_date = ti.iteration_end_date
            AND s.batch              = ti.max_batch
    ),
    odm_latest AS (
        SELECT o.*
        FROM {odm} o
        INNER JOIN target_iterations ti
            ON  o.iteration_end_date = ti.iteration_end_date
            AND o.batch              = ti.max_batch
    )
    SELECT
        s.id_x,
        s.title,
        s.labels,
        s.issue_summary,
        s.state,
        s.weight,
        s.prod_release_date,
        s.iteration_end_date,
        s.iteration_start_date,
        s.batch,
        o.rule_name,
        o.target_type,
        o.change_type
    FROM staat_latest s
    LEFT JOIN odm_latest o
        ON  s.id_x               = o.issue_id
        AND s.iteration_end_date = o.iteration_end_date
    ORDER BY s.iteration_end_date, s.id_x
    """


def _build_reactivated_query(insight_types: list, schema: str) -> str:
    """SQL for story details of reactivated insight types.

    - Latest batch per iteration_end_date from ODM for the given rule_names.
    - LEFT JOIN staat_insight_release on issue_id = id_x, iteration_end_date, batch.
    """
    staat     = f"{schema}.staat_insight_release"
    odm       = f"{schema}.odm_release_details"
    in_clause = ", ".join(f"'{t}'" for t in insight_types)
    return f"""
    WITH odm_ranked AS (
        SELECT
            o.*,
            MAX(o.batch) OVER (PARTITION BY o.iteration_end_date) AS max_batch
        FROM {odm} o
        WHERE o.rule_name IN ({in_clause})
    ),
    odm_latest AS (
        SELECT * FROM odm_ranked WHERE batch = max_batch
    )
    SELECT
        o.issue_id          AS id_x,
        o.iteration_end_date,
        o.batch,
        o.rule_name,
        o.target_type,
        o.change_type,
        s.title,
        s.issue_summary,
        s.labels,
        s.state,
        s.weight,
        s.prod_release_date,
        s.iteration_start_date
    FROM odm_latest o
    LEFT JOIN {staat} s
        ON  o.issue_id           = s.id_x
        AND o.iteration_end_date = s.iteration_end_date
        AND o.batch              = s.batch
    ORDER BY o.rule_name, o.iteration_end_date, o.issue_id
    """


# ---------------------------------------------------------------------------
# Reactivated insight helpers
# ---------------------------------------------------------------------------

def get_reactivated_insights(
    schema: str,
    conn,
    year: int,
    month: int,
) -> pd.DataFrame:
    """Find insight types reactivated in *year*/*month* and return their story details.

    Step 1 – Query odm_exclusion_insight_type for rows where is_curr=0
              (was excluded; now re-enabled) and last_upd_dte is in the given month.
    Step 2 – For each reactivated insight_type (= rule_name in ODM), fetch
              the latest-batch rows from odm_release_details and join
              staat_insight_release for title + description.
    Multiple iterations / stories for the same rule are bundled in the
    returned DataFrame (group by rule_name to process them together).
    Returns an empty DataFrame if no reactivated types are found.
    """
    sql_reactivated = f"""
        SELECT DISTINCT insight_type
        FROM {schema}.odm_exclusion_insight_type
        WHERE EXTRACT(YEAR  FROM last_upd_dte::date) = {year}
          AND EXTRACT(MONTH FROM last_upd_dte::date) = {month}
          AND is_curr = 0
    """
    try:
        reactivated_df = pd.read_sql_query(sql_reactivated, conn)
    except Exception as exc:
        logger.warning("[reactivated] exclusion table query failed: %s", exc)
        return pd.DataFrame()

    if reactivated_df.empty:
        logger.info("[reactivated] No reactivated insight types found for %s-%02d.", year, month)
        return pd.DataFrame()

    insight_types = reactivated_df["insight_type"].dropna().str.strip().tolist()
    logger.info("[reactivated] Found %d reactivated type(s): %s", len(insight_types), insight_types)

    try:
        details_df = pd.read_sql_query(
            _build_reactivated_query(insight_types, schema), conn
        )
        details_df["change_type"] = details_df["change_type"].replace({"added": "new"})
        return details_df
    except Exception as exc:
        logger.warning("[reactivated] story detail query failed: %s", exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Public data helpers
# ---------------------------------------------------------------------------

def get_exec_summary_data(
    month_value: str,
    allow_prompt: bool = True,
) -> tuple:
    """Query GP and return (current_df, next_month_df, reactivated_df).

    Parameters
    ----------
    month_value   : 'YYYY-MM'  e.g. '2026-05'
    allow_prompt  : when False (DAG runs), missing credentials raise an error
                    instead of prompting interactively.

    Returns
    -------
    current_df     Latest-batch STAAT+ODM rows for the selected month.
    next_month_df  Latest-batch rows for the following month, or None.
    reactivated_df Story details for insight types reactivated this month.

    Note for insights_release_dashboard.py callback
    ────────────────────────────────────────────────
    Unpack as:
        df_month, df_next, df_react = get_exec_summary_data(selected_month)
    """
    schema  = _get_schema()
    conn    = _get_gp_conn(allow_prompt=allow_prompt)
    year, month_num = (int(x) for x in month_value.split("-"))

    current_df = pd.read_sql_query(_build_month_query(month_value, schema), conn)
    current_df["change_type"] = current_df["change_type"].replace({"added": "new"})

    reactivated_df = get_reactivated_insights(schema, conn, year, month_num)

    try:
        next_period = str(pd.Period(month_value, "M") + 1)
        next_df     = pd.read_sql_query(_build_month_query(next_period, schema), conn)
        next_df["change_type"] = next_df["change_type"].replace({"added": "new"})
        next_month_df: Optional[pd.DataFrame] = next_df if not next_df.empty else None
    except Exception:
        next_month_df = None

    conn.close()
    return current_df, next_month_df, reactivated_df


def get_month_options_from_db() -> list:
    """Return sorted dropdown options queried from GP (used by the Dash refresh callback)."""
    schema = _get_schema()
    conn   = _get_gp_conn()
    sql    = f"""
        SELECT DISTINCT TO_CHAR(prod_release_date::date, 'YYYY-MM') AS month_val
        FROM {schema}.staat_insight_release
        WHERE prod_release_date IS NOT NULL
        ORDER BY month_val DESC
    """
    df = pd.read_sql_query(sql, conn)
    conn.close()
    options = []
    for val in df["month_val"]:
        try:
            dt = pd.Period(val, "M").to_timestamp()
            options.append({"label": dt.strftime("%B %Y"), "value": val})
        except Exception:
            pass
    return options


# ---------------------------------------------------------------------------
# Legacy helpers (kept for backward compatibility with Dash callbacks)
# ---------------------------------------------------------------------------

def extract_month_options(df: pd.DataFrame) -> list:
    """Return sorted options from a pre-loaded DataFrame."""
    if df is None or df.empty or "prod_release_date" not in df.columns:
        return []
    dates = pd.to_datetime(df["prod_release_date"], errors="coerce").dropna()
    if dates.empty:
        return []
    months_sorted = sorted(dates.dt.to_period("M").unique(), reverse=True)
    return [
        {"label": p.to_timestamp().strftime("%B %Y"), "value": str(p)}
        for p in months_sorted
    ]


def filter_by_month(df: pd.DataFrame, month_value: str) -> pd.DataFrame:
    """Filter df by prod_release_date month (kept for compatibility)."""
    if df is None or df.empty or "prod_release_date" not in df.columns:
        return pd.DataFrame()
    dates = pd.to_datetime(df["prod_release_date"], errors="coerce")
    return df.loc[dates.dt.to_period("M").astype(str) == month_value]


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def _has_label(labels_value, target: str) -> bool:
    """True if *target* is one of the comma-separated labels (case-insensitive)."""
    if not labels_value or not isinstance(labels_value, str):
        return False
    return target.lower() in [lbl.strip().lower() for lbl in labels_value.split(",")]


def _filter_by_label(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Return rows whose labels column contains *label* exactly."""
    if df is None or df.empty or "labels" not in df.columns:
        return pd.DataFrame()
    return df.loc[df["labels"].apply(lambda v: _has_label(v, label))]


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def _build_reactivated_text(
    reactivated_df: pd.DataFrame,
    skip_rules: set = None,
) -> str:
    """Format reactivated-insight rows into a prompt-ready string.

    Stories for the same rule_name are bundled under one header.
    Any rule_name present in *skip_rules* is omitted so each insight
    appears only once across the full prompt.
    """
    if reactivated_df is None or reactivated_df.empty:
        return "No reactivated insights found for this period."

    skip     = skip_rules or set()
    lines: list = []
    rule_num = 0

    for rule_name, group in reactivated_df.groupby("rule_name", sort=True):
        if rule_name in skip:
            continue   # already listed in the New Insights section
        rule_num += 1
        lines.append(f"Reactivated Insight #{rule_num}: {rule_name}")
        seen_ids: set = set()
        story_num = 0
        for _, row in group.iterrows():
            id_x = row.get("id_x", "")
            if id_x and id_x not in seen_ids:
                seen_ids.add(id_x)
                story_num += 1
                lines.append(f"  Story #{story_num}:")
                lines.append(f"    Title:   {row.get('title', 'N/A')}")
                lines.append(f"    Summary: {row.get('issue_summary', 'N/A')}")
        lines.append("")

    return "\n".join(lines).strip() if lines else "No reactivated insights found for this period."


def _build_prompt(
    issues_df: pd.DataFrame,
    next_month_df: Optional[pd.DataFrame] = None,
    reactivated_df: Optional[pd.DataFrame] = None,
) -> str:
    """Build the LLM prompt.

    Label conventions
    -----------------
    "Top Feature" label  →  Section 2
    "New Insight"  label  →  Section 3  (also deduped against reactivated)
    Reactivated insights  →  appended to Section 3 data
    All stories           →  Section 1 context
    """

    # ── Section 1: all stories ────────────────────────────────────────────────
    issues_lines: list = []
    seen_ids: set = set()
    for _, row in issues_df.iterrows():
        id_x = row.get("id_x", "")
        if id_x and id_x not in seen_ids:
            seen_ids.add(id_x)
            issues_lines.append(
                f"Story #{len(issues_lines)+1}:\n"
                f"Title:   {row.get('title','N/A')}\n"
                f"Summary: {row.get('issue_summary','N/A')}\n"
                f"State:   {row.get('state','N/A')}\n"
                f"Labels:  {row.get('labels','N/A')}\n"
                f"Weight:  {row.get('weight','N/A')}"
            )
    issues_text = "\n\n".join(issues_lines[:20]) if issues_lines else "No issues data available."

    # ── Section 2: Top Feature ────────────────────────────────────────────────
    top_df    = _filter_by_label(issues_df, "Top Feature")
    top_lines: list = []
    seen_top: set = set()
    for _, row in top_df.iterrows():
        id_x = row.get("id_x", "")
        if id_x and id_x not in seen_top:
            seen_top.add(id_x)
            top_lines.append(
                f"Top Feature #{len(top_lines)+1}:\n"
                f"Title:   {row.get('title','N/A')}\n"
                f"Summary: {row.get('issue_summary','N/A')}\n"
                f"Labels:  {row.get('labels','N/A')}\n"
                f"Rule:    {row.get('rule_name','N/A')}\n"
                f"Change:  {row.get('change_type','N/A')}"
            )
    top_feature_text = (
        "\n\n".join(top_lines[:5]) if top_lines
        else "No issue labelled 'Top Feature' found for this period."
    )

    # ── Section 3: New Insights (label-based) ─────────────────────────────────
    ni_df    = _filter_by_label(issues_df, "New Insight")
    ni_lines: list = []
    seen_rules: set = set()   # carried into reactivated block for dedup
    for _, row in ni_df.iterrows():
        rule = row.get("rule_name", "")
        if rule and rule not in seen_rules:
            seen_rules.add(rule)
            ni_lines.append(
                f"New Insight #{len(ni_lines)+1}:\n"
                f"Rule Name:   {rule}\n"
                f"Target Type: {row.get('target_type','N/A')}\n"
                f"Change Type: {row.get('change_type','N/A')}\n"
                f"Title:       {row.get('title','N/A')}\n"
                f"Summary:     {row.get('issue_summary','N/A')}"
            )
    insights_text = (
        "\n\n".join(ni_lines[:15]) if ni_lines
        else "No issues labelled 'New Insight' found for this period."
    )

    # ── Reactivated insights (deduped against seen_rules) ─────────────────────
    reactivated_text = _build_reactivated_text(reactivated_df, skip_rules=seen_rules)

    # ── Section 6: Coming Soon – next month New Insight titles only ───────────
    next_lines: list = []
    if next_month_df is not None and not next_month_df.empty:
        next_ni_df  = _filter_by_label(next_month_df, "New Insight")
        next_seen: set = set()
        for _, row in next_ni_df.iterrows():
            rule  = row.get("rule_name", "")
            title = row.get("title", "")
            key   = rule or title
            if key and key not in next_seen:
                next_seen.add(key)
                next_lines.append(f"- {row.get('title', rule)}")
    next_month_text = (
        "\n".join(next_lines[:10]) if next_lines
        else "No next-month data found \u2013 provide directional focus areas based on current trends."
    )

    prompt = f"""
    You are an expert product manager and technical writer. Based on the following GitLab issues and
    new insights data, generate a comprehensive executive summary report.

    === GITLAB ISSUES DATA (all stories for this month) ===
    {issues_text}

    === TOP FEATURE DATA (issues labelled "Top Feature") ===
    {top_feature_text}

    === NEW INSIGHTS DATA (issues labelled "New Insight") ===
    {insights_text}

    === REACTIVATED INSIGHTS (insight types re-enabled this month) ===
    These insights were previously excluded but have been re-activated this month.
    Include them as additional items in the New Insights section (Section 3), clearly noting they are reactivated.
    {reactivated_text}

    === REQUIRED OUTPUT FORMAT ===
    Please analyse the data and provide a structured report. Use "<Question>: <answer>" style:

    1. Executive Summary:
    - What materially changed this month: <answer>
    - Why it matters to the business: <answer>

    2. Top Feature (or Insight) of the Month:
    - Insight name: <answer>
    - Brief description / example of narrative: <answer>
    - Main benefit / value: <answer>

    3. New Insights (ranked by importance/impact, including reactivated insights):
    For each new or reactivated insight:
    - Insight name: <answer>
    - Brief description / example of narrative: <answer>
    - Main benefit / value: <answer>
    - Status: New | Reactivated

    4. Process Improvements & Optimization:
    - Name / brief description: <answer>
    - How did we do it? (If AI was used, altered process, etc.): <answer>
    - Benefits: <answer>

    5. Platform Maintenance & Stability:
    - Maintenance, fixes, or technical improvements: <answer>
    - Why this matters (risk reduction, performance, cost control): <answer>

    6. New Insights \u2013 Coming Soon:
    Use the following next-month data (if available):
    {next_month_text}
    - Focus areas (2-4 items max): list Insight Title only, limited to "New Insight" labelled items

    === INSTRUCTIONS ===
    - Be concise and business-focused
    - Rank insights by business impact and value
    - Use clear, non-technical language where possible
    - Focus on outcomes and benefits, not just features
    - If certain sections have no relevant data, state "No significant changes in this area"
    - Ensure all answers are data-driven based on the provided information
    - Section 1 Executive Summary: 2 sentences per question
    - Section 2 Top Feature: pick the single most impactful item from the "Top Feature" labelled issues
    - Section 3 New Insights: list each insight on a separate line; include reactivated insights at the end, marked as "Reactivated". Bundle multiple stories for the same reactivated rule under one entry
    - Section 6 New Insights \u2013 Coming Soon: simple list of Insight Titles only. If no next-month data, provide directional focus areas based on current trends
    - Do NOT include instruction notes in the final output
    """
    return prompt.strip()


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def generate_summary(
    issues_df: pd.DataFrame,
    next_month_df: Optional[pd.DataFrame] = None,
    reactivated_df: Optional[pd.DataFrame] = None,
) -> str:
    """Call the LLM and return the executive summary text."""
    client   = _get_openai_client()
    prompt   = _build_prompt(issues_df, next_month_df=next_month_df, reactivated_df=reactivated_df)
    response = client.chat.completions.create(
        model       =MODEL_NAME,
        messages    =[{"role": "user", "content": prompt}],
        max_tokens  =MAX_TOKENS,
        temperature =TEMPERATURE,
    )
    return response.choices[0].message.content.strip()


def build_save_content(summary_text: str, month_label: str) -> str:
    """Format the summary for saving to a .txt file."""
    header = (
        f"Executive Summary Report \u2013 {month_label}\n"
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        + "=" * 80 + "\n\n"
    )
    return header + summary_text


# ---------------------------------------------------------------------------
# DAG entry point
# ---------------------------------------------------------------------------

def generate_executive_summary(**kwargs) -> str:
    """Generate and save the executive summary for the current month.

    This is the callable registered in the monthly DAG (PythonOperator).
    Month is derived from datetime.now() \u2013 no user input required.
    Output is saved to  <script_dir>/output/executive_summary_YYYY-MM.txt.
    """
    now         = datetime.now()
    month_value = now.strftime("%Y-%m")
    month_label = now.strftime("%B %Y")

    logger.info("[exec_summary] Generating executive summary for %s ...", month_label)

    df_current, df_next, df_reactivated = get_exec_summary_data(
        month_value, allow_prompt=False
    )

    if df_current.empty and (df_reactivated is None or df_reactivated.empty):
        msg = f"[exec_summary] No data found for {month_label}. Skipping."
        logger.warning(msg)
        return msg

    logger.info("[exec_summary] Current month rows : %d", len(df_current))
    logger.info(
        "[exec_summary] Reactivated rules  : %d",
        0 if df_reactivated is None or df_reactivated.empty
        else df_reactivated["rule_name"].nunique(),
    )
    logger.info("[exec_summary] Next month rows    : %d", 0 if df_next is None else len(df_next))

    summary_text = generate_summary(
        df_current,
        next_month_df  =df_next,
        reactivated_df =df_reactivated if (df_reactivated is not None and not df_reactivated.empty) else None,
    )
    content = build_save_content(summary_text, month_label)

    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"executive_summary_{month_value}.txt"

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(content)

    logger.info("[exec_summary] Saved \u2192 %s", out_path)
    return summary_text


# ---------------------------------------------------------------------------
# Dash layout (production UI only)
# ---------------------------------------------------------------------------

def layout(df: pd.DataFrame):
    """Return the dcc.Tab component for the Executive Summary tab.

    Only callable when Dash and styles are installed (production dashboard).
    Never called from the DAG or the notebook.
    """
    if not _DASH_AVAILABLE:
        raise RuntimeError(
            "Dash is not installed in this environment. "
            "layout() can only be called from the dashboard server."
        )

    try:
        month_options = get_month_options_from_db()
    except Exception:
        month_options = extract_month_options(df)

    default_month = month_options[0]["value"] if month_options else None

    return dcc.Tab(
        label="Executive Summary",
        value="tab-executive-summary",
        children=html.Div(
            [
                dcc.Store(id="exec-summary-text-store", data=""),

                html.H3(
                    "Executive Summary Generator",
                    style=release_dashboard_styles["section_header_style"],
                ),

                # Toolbar
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span(
                                    "Controls",
                                    style=release_dashboard_styles["panel_section_label_style"],
                                ),
                                html.Div(
                                    [
                                        html.Button(
                                            "Generate Executive Summary",
                                            id="exec-summary-generate-btn",
                                            n_clicks=0,
                                            style=release_dashboard_styles["btn_primary_style"],
                                        ),
                                        html.Div(style=release_dashboard_styles["btn_divider_style"]),
                                        html.Button(
                                            "Save as TXT",
                                            id="exec-summary-save-btn",
                                            n_clicks=0,
                                            style=release_dashboard_styles["btn_export_style"],
                                            disabled=True,
                                        ),
                                    ],
                                    style=release_dashboard_styles["btn_group_style"],
                                ),
                                html.Div(
                                    [
                                        html.Span(
                                            "Month",
                                            style=release_dashboard_styles["dropdown_label_style"],
                                        ),
                                        dcc.Dropdown(
                                            id="exec-summary-month-filter",
                                            options=month_options,
                                            value=default_month,
                                            placeholder="Select month\u2026",
                                            clearable=True,
                                            style=release_dashboard_styles["dropdown_style"],
                                        ),
                                    ],
                                    style=release_dashboard_styles["dropdown_container_style"],
                                ),
                            ],
                            style=release_dashboard_styles["panel_left_style"],
                        ),
                    ],
                    style=release_dashboard_styles["toolbar_style"],
                ),

                # Output card
                html.Div(
                    [
                        dcc.Download(id="exec-summary-download"),
                        dcc.Loading(
                            id="exec-summary-loading",
                            type="default",
                            children=html.Div(
                                id="exec-summary-output",
                                children="Select a month and click 'Generate Executive Summary' to begin.",
                                style={
                                    **release_dashboard_styles.get("table_style", {}),
                                    "padding"     : "20px",
                                    "background"  : "#ffffff",
                                    "borderRadius": "8px",
                                    "minHeight"   : "220px",
                                    "whiteSpace"  : "pre-wrap",
                                    "fontSize"    : "14px",
                                    "lineHeight"  : "1.6",
                                },
                            ),
                        ),
                    ],
                    style={"marginTop": "18px"},
                ),

                # Toast
                dbc.Toast(
                    id="exec-summary-save-toast",
                    header="Executive Summary",
                    children="",
                    is_open=False,
                    duration=4000,
                    icon="success",
                    dismissable=True,
                    style={"position": "fixed", "top": 16, "right": 16, "zIndex": 9999},
                ),
            ],
            style=release_dashboard_styles["table_container_style"],
        ),
    )


if __name__ == "__main__":
    # Local quick-test: prompt for month, then generate.
    import getpass
    month_raw = input("Month to summarise (e.g. '2026-05' or 'May 2026'): ").strip()
    try:
        MONTH = str(pd.Period(month_raw, "M"))
    except Exception:
        MONTH = datetime.strptime(
            month_raw.replace("-", " ").replace("/", " "), "%B %Y"
        ).strftime("%Y-%m")
    label = pd.Period(MONTH, "M").to_timestamp().strftime("%B %Y")
    df_c, df_n, df_r = get_exec_summary_data(MONTH, allow_prompt=True)
    text = generate_summary(df_c, next_month_df=df_n, reactivated_df=df_r if not df_r.empty else None)
    print("\n" + "=" * 80)
    print(f"Executive Summary \u2013 {label}")
    print("=" * 80 + "\n")
    print(text)
