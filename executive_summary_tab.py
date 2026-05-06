"""Executive Summary tab – pick a month (from prod_release_date) and generate an
LLM-powered executive summary, with an option to save it as a .txt file.

Environment detection
─────────────────────
The module detects its runtime context at import time:

  Production (Airflow / Dash)
    Airflow packages are importable.  GP connection → PostgresHook via
    Variable "GP_Dash_connect".  OpenAI credentials → Airflow Connection
    "STAAT-DS-OPENAI-LLM".  Schema → Variable "IKG_DASHBOARD_SCHEMA".

  Local (Jupyter / script)
    Airflow is not installed.  All credentials are read from environment
    variables, which the notebook sets (via getpass) before importing this
    module:

      GP_HOST, GP_PORT, GP_DB, GP_USER, GP_PASSWORD, GP_SCHEMA
      OPENAI_API_KEY, OPENAI_BASE_URL

The data/LLM functions behave identically in both environments.
The Dash layout helpers are only wired when Dash is available.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import pandas as pd
from openai import OpenAI


# ── Environment detection ─────────────────────────────────────────────────────

try:
    from airflow.models import Connection, Variable          # type: ignore
    from airflow.providers.postgres.hooks.postgres import PostgresHook  # type: ignore
    _AIRFLOW_AVAILABLE = True
except ImportError:
    _AIRFLOW_AVAILABLE = False

# Dash / styles only needed inside the dashboard server.
try:
    from dash import dcc, html
    import dash_bootstrap_components as dbc
    from styles import release_dashboard_styles
    _DASH_AVAILABLE = True
except ImportError:
    _DASH_AVAILABLE = False


# ── OpenAI / Azure configuration ──────────────────────────────────────────────

MODEL_NAME  = "gpt-4.1"
MAX_TOKENS  = 12000
TEMPERATURE = 0.1

_client: Optional[OpenAI] = None


def _get_openai_client() -> OpenAI:
    """Return the shared OpenAI client, initialising it on first call.

    Production : reads credentials from Airflow Connection "STAAT-DS-OPENAI-LLM".
    Local      : reads OPENAI_API_KEY and OPENAI_BASE_URL from env vars.
    """
    global _client
    if _client is not None:
        return _client

    if _AIRFLOW_AVAILABLE:
        conn     = Connection.get_connection_from_secrets("STAAT-DS-OPENAI-LLM")
        api_key  = conn.password
        base_url = conn.host
    else:
        api_key  = os.environ["OPENAI_API_KEY"]
        base_url = os.environ.get(
            "OPENAI_BASE_URL",
            "https://cirruspl-staat-ste-dev-ai.openai.azure.com/openai/v1/",
        )

    _client = OpenAI(api_key=api_key, base_url=base_url)
    return _client


# ── GP connection ─────────────────────────────────────────────────────────────

def _get_gp_conn():
    """Return a live database connection.

    Production : PostgresHook via Airflow Variable "GP_Dash_connect".
    Local      : psycopg2 direct connection using GP_* env vars.
    """
    if _AIRFLOW_AVAILABLE:
        connect = Variable.get("GP_Dash_connect")
        return PostgresHook(postgres_conn_id=connect).get_conn()

    import psycopg2  # available locally; not required in the Airflow image
    return psycopg2.connect(
        host    =os.environ["GP_HOST"],
        port    =int(os.environ.get("GP_PORT", 5432)),
        dbname  =os.environ["GP_DB"],
        user    =os.environ["GP_USER"],
        password=os.environ["GP_PASSWORD"],
    )


def _get_schema() -> str:
    """Return the IKG schema name.

    Production : Airflow Variable "IKG_DASHBOARD_SCHEMA".
    Local      : env var GP_SCHEMA (default "core_ikg").
    """
    if _AIRFLOW_AVAILABLE:
        return Variable.get("IKG_DASHBOARD_SCHEMA")
    return os.environ.get("GP_SCHEMA", "core_ikg")


# ── SQL helpers ───────────────────────────────────────────────────────────────

def _build_month_query(month_value: str, schema: str) -> str:
    """Return SQL that selects the latest-batch rows from STAAT + ODM for *month_value*.

    Strategy
    --------
    1. Identify every ``iteration_end_date`` in STAAT whose ``prod_release_date``
       falls in the target month (e.g. ``'2026-05'``).
    2. For each such iteration take ``MAX(batch)`` rows from both STAAT and ODM.
    3. LEFT JOIN ODM onto STAAT so stories without ODM rows are still included.
    """
    staat_table = f"{schema}.staat_insight_release"
    odm_table   = f"{schema}.odm_release_details"

    return f"""
    WITH target_iterations AS (
        SELECT
            iteration_end_date,
            MAX(batch) AS max_batch
        FROM {staat_table}
        WHERE TO_CHAR(prod_release_date::date, 'YYYY-MM') = '{month_value}'
        GROUP BY iteration_end_date
    ),
    staat_latest AS (
        SELECT s.*
        FROM {staat_table} s
        INNER JOIN target_iterations ti
            ON  s.iteration_end_date = ti.iteration_end_date
            AND s.batch              = ti.max_batch
    ),
    odm_latest AS (
        SELECT o.*
        FROM {odm_table} o
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


# ── Public data-access helpers ────────────────────────────────────────────────

def get_exec_summary_data(
    month_value: str,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Query GP and return ``(current_month_df, next_month_df)``.

    Parameters
    ----------
    month_value : str
        Period string in ``'YYYY-MM'`` format (e.g. ``'2026-05'``).

    Returns
    -------
    current_month_df : pd.DataFrame
        Latest-batch rows for the selected month.
    next_month_df : Optional[pd.DataFrame]
        Latest-batch rows for the following month, or ``None`` if no data.
    """
    schema = _get_schema()
    conn   = _get_gp_conn()

    current_df = pd.read_sql_query(_build_month_query(month_value, schema), conn)
    current_df["change_type"] = current_df["change_type"].replace({"added": "new"})

    try:
        next_period = str(pd.Period(month_value, "M") + 1)
        next_df     = pd.read_sql_query(_build_month_query(next_period, schema), conn)
        next_df["change_type"] = next_df["change_type"].replace({"added": "new"})
        next_month_df: Optional[pd.DataFrame] = next_df if not next_df.empty else None
    except Exception:
        next_month_df = None

    conn.close()
    return current_df, next_month_df


def get_month_options_from_db() -> list[dict]:
    """Return sorted dropdown options queried directly from GP."""
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


# ── Legacy helpers (kept for backward compatibility) ──────────────────────────

def extract_month_options(df: pd.DataFrame) -> list[dict]:
    """Return sorted dropdown options from a pre-loaded DataFrame."""
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
    """Filter *df* by prod_release_date month (kept for compatibility)."""
    if df is None or df.empty or "prod_release_date" not in df.columns:
        return pd.DataFrame()
    dates = pd.to_datetime(df["prod_release_date"], errors="coerce")
    return df.loc[dates.dt.to_period("M").astype(str) == month_value]


# ── Label helpers ─────────────────────────────────────────────────────────────

def _has_label(labels_value, target: str) -> bool:
    """Return True if *target* is one of the comma-separated labels (case-insensitive)."""
    if not labels_value or not isinstance(labels_value, str):
        return False
    return target.lower() in [lbl.strip().lower() for lbl in labels_value.split(",")]


def _filter_by_label(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Return rows whose labels column contains *label* exactly."""
    if df is None or df.empty or "labels" not in df.columns:
        return pd.DataFrame()
    return df.loc[df["labels"].apply(lambda v: _has_label(v, label))]


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(
    issues_df: pd.DataFrame,
    next_month_df: Optional[pd.DataFrame] = None,
) -> str:
    """Build the LLM prompt from the filtered insight-release data.

    Label conventions
    -----------------
    ``"Top Feature"`` label → Section 2 (Top Feature of the Month).
    ``"New Insight"`` label → Section 3 (New Insights ranked by impact).
    All stories feed Section 1 context regardless of label.
    """

    # ── Section 1: all stories ────────────────────────────────────────────────
    issues_lines: list[str] = []
    seen_ids: set = set()

    for _, row in issues_df.iterrows():
        id_x = row.get("id_x", "")
        if id_x and id_x not in seen_ids:
            seen_ids.add(id_x)
            issues_lines.append(
                f"Story #{len(issues_lines) + 1}:\n"
                f"Title:   {row.get('title', 'N/A')}\n"
                f"Summary: {row.get('issue_summary', 'N/A')}\n"
                f"State:   {row.get('state', 'N/A')}\n"
                f"Labels:  {row.get('labels', 'N/A')}\n"
                f"Weight:  {row.get('weight', 'N/A')}"
            )

    issues_text = "\n\n".join(issues_lines[:20]) if issues_lines else "No issues data available."

    # ── Section 2: Top Feature ────────────────────────────────────────────────
    top_feature_df    = _filter_by_label(issues_df, "Top Feature")
    top_feature_lines: list[str] = []
    seen_top: set = set()

    for _, row in top_feature_df.iterrows():
        id_x = row.get("id_x", "")
        if id_x and id_x not in seen_top:
            seen_top.add(id_x)
            top_feature_lines.append(
                f"Top Feature #{len(top_feature_lines) + 1}:\n"
                f"Title:   {row.get('title', 'N/A')}\n"
                f"Summary: {row.get('issue_summary', 'N/A')}\n"
                f"Labels:  {row.get('labels', 'N/A')}\n"
                f"Rule:    {row.get('rule_name', 'N/A')}\n"
                f"Change:  {row.get('change_type', 'N/A')}"
            )

    top_feature_text = (
        "\n\n".join(top_feature_lines[:5])
        if top_feature_lines
        else "No issue labelled 'Top Feature' found for this period."
    )

    # ── Section 3: New Insights ───────────────────────────────────────────────
    new_insight_df    = _filter_by_label(issues_df, "New Insight")
    new_insights_lines: list[str] = []
    seen_rules: set = set()

    for _, row in new_insight_df.iterrows():
        rule = row.get("rule_name", "")
        if rule and rule not in seen_rules:
            seen_rules.add(rule)
            new_insights_lines.append(
                f"New Insight #{len(new_insights_lines) + 1}:\n"
                f"Rule Name:   {rule}\n"
                f"Target Type: {row.get('target_type', 'N/A')}\n"
                f"Change Type: {row.get('change_type', 'N/A')}\n"
                f"Title:       {row.get('title', 'N/A')}\n"
                f"Summary:     {row.get('issue_summary', 'N/A')}"
            )

    insights_text = (
        "\n\n".join(new_insights_lines[:15])
        if new_insights_lines
        else "No issues labelled 'New Insight' found for this period."
    )

    # ── Section 6: Looking Ahead ──────────────────────────────────────────────
    next_month_lines: list[str] = []
    if next_month_df is not None and not next_month_df.empty:
        next_seen_rules: set = set()
        for _, row in next_month_df.iterrows():
            rule = row.get("rule_name", "")
            if rule and rule not in next_seen_rules:
                next_seen_rules.add(rule)
                next_month_lines.append(
                    f"- Rule Name:   {rule}\n"
                    f"  Target Type: {row.get('target_type', 'N/A')}\n"
                    f"  Change Type: {row.get('change_type', 'N/A')}\n"
                    f"  Title:       {row.get('title', 'N/A')}\n"
                    f"  Labels:      {row.get('labels', 'N/A')}"
                )

    next_month_text = (
        "\n".join(next_month_lines[:10])
        if next_month_lines
        else "No data found for the next month – provide directional focus areas based on current trends."
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

    === REQUIRED OUTPUT FORMAT ===
    Please analyse the data and provide a structured report. Use "<Question>: <answer>" style:

    1. Executive Summary:
    - What materially changed this month: <answer>
    - Why it matters to the business: <answer>

    2. Top Feature (or Insight) of the Month:
    - Insight name: <answer>

    3. New Insights (ranked by importance/impact):
    For each new insight:
    - Insight name: <answer>

    4. Process Improvements & Optimization:
    - Name / brief description: <answer>
    - How did we do it? (If AI was used, altered process, etc.): <answer>
    - Benefits: <answer>

    5. Platform Maintenance & Stability:
    - Maintenance, fixes, or technical improvements: <answer>
    - Why this matters (risk reduction, performance, cost control): <answer>

    6. Looking Ahead (Next Month):
    Use the following NEXT MONTH data (if available) to ground your answer:
    {next_month_text}
    - Focus areas (2-4 items max, focus on new insights coming next month): <answer>

    === INSTRUCTIONS ===
    - Be concise and business-focused
    - Rank insights by business impact and value
    - Use clear, non-technical language where possible
    - Focus on outcomes and benefits, not just features
    - If certain sections have no relevant data, state "No significant changes in this area"
    - Ensure all answers are data-driven based on the provided information
    - Section 1 Executive Summary: 2 sentences per question
    - Section 2 Top Feature: pick the single most impactful item from the "Top Feature" labelled issues
    - Section 3 New Insights: list each individual insight on a separate line, drawn from "New Insight" labelled issues
    - Section 6 Looking Ahead: if no next-month data is available, provide directional focus areas based on current trends
    - Do NOT include instruction notes in the final output
    """
    return prompt.strip()


# ── LLM call ──────────────────────────────────────────────────────────────────

def generate_summary(
    issues_df: pd.DataFrame,
    next_month_df: Optional[pd.DataFrame] = None,
) -> str:
    """Call the LLM and return the executive summary text."""
    client   = _get_openai_client()
    prompt   = _build_prompt(issues_df, next_month_df=next_month_df)
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
        f"Executive Summary Report – {month_label}\n"
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        + "=" * 80
        + "\n\n"
    )
    return header + summary_text


# ── Dash layout (production only) ─────────────────────────────────────────────
# layout() is only called from insights_release_dashboard.py when Dash is
# running inside the server.  It is never called from the notebook.

def layout(df: pd.DataFrame):
    """Return the ``dcc.Tab`` component for the Executive Summary tab."""
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
                                        html.Div(
                                            style=release_dashboard_styles["btn_divider_style"]
                                        ),
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
                                            placeholder="Select month…",
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
                                children=(
                                    "Select a month and click "
                                    "'Generate Executive Summary' to begin."
                                ),
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
                    style={
                        "position": "fixed",
                        "top"     : 16,
                        "right"   : 16,
                        "zIndex"  : 9999,
                    },
                ),
            ],
            style=release_dashboard_styles["table_container_style"],
        ),
    )
