"""Executive Summary tab – allows the user to pick a month (from prod_release_date)
and generate an LLM-powered executive summary, with an option to save it as a .txt file.

Key changes vs. original
─────────────────────────
• Data is fetched directly from GP at generate-time (no store filtering).
  Use get_exec_summary_data(month_value) to obtain (current_df, next_month_df).
• "Top Feature" issues  → identified by the label  "Top Feature"  (not change_type).
• "New Insight" issues  → identified by the label  "New Insight"  (not change_type).
• Latest batch is resolved per-iteration (PARTITION BY iteration_end_date) so that
  multiple batches per iteration are handled correctly.
• Month dropdown is populated dynamically from GP on tab load / refresh.

Callback wiring note (insights_release_dashboard.py)
──────────────────────────────────────────────────────
Replace the existing generate_exec_summary callback body with:

    df_month, df_next = get_exec_summary_data(selected_month)
    if df_month.empty:
        msg = f"No data found for the selected month ({selected_month})."
        return msg, "", True
    try:
        summary = generate_summary(df_month, next_month_df=df_next)
    except Exception as exc:
        return f"Error generating summary: {exc}", "", True
    return summary, summary, False

The refresh_exec_summary_data callback should call get_month_options_from_db()
instead of extract_month_options(latest).
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from dash import dcc, html
import dash_bootstrap_components as dbc

from openai import OpenAI
from styles import release_dashboard_styles

# Airflow / GP imports (production only – kept identical to original)
from airflow.models import Connection, Variable  # type: ignore
from airflow.providers.postgres.hooks.postgres import PostgresHook  # type: ignore


# ── OpenAI / Azure configuration ─────────────────────────────────────────────
# All connection variables kept the same as the original file.

MODEL_NAME = "gpt-4.1"
MAX_TOKENS = 12000
TEMPERATURE = 0.1

_client: OpenAI | None = None


def _get_openai_client() -> OpenAI:
    """Lazily initialise and return the shared OpenAI client."""
    global _client
    openai_conn = Connection.get_connection_from_secrets("STAAT-DS-OPENAI-LLM")
    OPENAI_API_KEY = openai_conn.password
    OPENAI_BASE_URL = openai_conn.host
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    return _client


# ── GP connection ─────────────────────────────────────────────────────────────

def _get_gp_conn():
    """Return a live GP connection via the Airflow PostgresHook."""
    connect = Variable.get("GP_Dash_connect")
    return PostgresHook(postgres_conn_id=connect).get_conn()


def _get_schema() -> str:
    return Variable.get("IKG_DASHBOARD_SCHEMA")


# ── SQL helpers ───────────────────────────────────────────────────────────────

def _build_month_query(month_value: str, schema: str) -> str:
    """
    Return a SQL query that fetches the latest batch per iteration whose
    prod_release_date falls in *month_value* (e.g. '2025-01').

    Strategy
    --------
    1. Find every iteration_end_date where at least one row in STAAT has a
       prod_release_date in the target month.
    2. For each such iteration take only the MAX(batch) rows from both STAAT
       and ODM, then LEFT JOIN ODM onto STAAT.
    """
    staat_table = f"{schema}.staat_insight_release"
    odm_table   = f"{schema}.odm_release_details"

    return f"""
    WITH target_iterations AS (
        -- Iterations that have a prod_release_date in the selected month
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
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Query GP and return ``(current_month_df, next_month_df)``.

    Parameters
    ----------
    month_value : str
        Period string in ``'YYYY-MM'`` format (e.g. ``'2026-05'``).

    Returns
    -------
    current_month_df : pd.DataFrame
        Rows for the selected month (latest batch per iteration).
    next_month_df : pd.DataFrame | None
        Rows for the following month, or ``None`` if no data exists.
    """
    schema = _get_schema()
    conn   = _get_gp_conn()

    current_df = pd.read_sql_query(_build_month_query(month_value, schema), conn)
    current_df["change_type"] = current_df["change_type"].replace({"added": "new"})

    # Derive next month
    try:
        next_period = str(pd.Period(month_value, "M") + 1)
        next_df = pd.read_sql_query(_build_month_query(next_period, schema), conn)
        next_df["change_type"] = next_df["change_type"].replace({"added": "new"})
        next_month_df: pd.DataFrame | None = next_df if not next_df.empty else None
    except Exception:
        next_month_df = None

    conn.close()
    return current_df, next_month_df


def get_month_options_from_db() -> list[dict]:
    """
    Return sorted dropdown options derived from STAAT prod_release_date values.
    Call this from the refresh_exec_summary_data callback instead of
    extract_month_options().
    """
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


# ── Legacy helper (kept for backward compatibility with refresh callback) ─────

def extract_month_options(df: pd.DataFrame) -> list[dict]:
    """Return sorted dropdown options from a pre-loaded DataFrame.

    Prefer ``get_month_options_from_db()`` for fresh data.
    """
    if df is None or df.empty or "prod_release_date" not in df.columns:
        return []
    dates = pd.to_datetime(df["prod_release_date"], errors="coerce").dropna()
    if dates.empty:
        return []
    months        = dates.dt.to_period("M").unique()
    months_sorted = sorted(months, reverse=True)
    return [
        {"label": p.to_timestamp().strftime("%B %Y"), "value": str(p)}
        for p in months_sorted
    ]


def filter_by_month(df: pd.DataFrame, month_value: str) -> pd.DataFrame:
    """Filter *df* by prod_release_date month (kept for compatibility)."""
    if df is None or df.empty or "prod_release_date" not in df.columns:
        return pd.DataFrame()
    dates = pd.to_datetime(df["prod_release_date"], errors="coerce")
    mask  = dates.dt.to_period("M").astype(str) == month_value
    return df.loc[mask]


# ── Label helpers ─────────────────────────────────────────────────────────────

def _has_label(labels_value, target: str) -> bool:
    """Return True if *target* appears as one of the comma-separated labels."""
    if not labels_value or not isinstance(labels_value, str):
        return False
    parts = [lbl.strip().lower() for lbl in labels_value.split(",")]
    return target.lower() in parts


def _filter_by_label(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Return rows whose *labels* column contains *label* (case-insensitive)."""
    if df is None or df.empty or "labels" not in df.columns:
        return pd.DataFrame()
    mask = df["labels"].apply(lambda v: _has_label(v, label))
    return df.loc[mask]


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(issues_df: pd.DataFrame, next_month_df: pd.DataFrame | None = None) -> str:
    """Build the LLM prompt from the filtered insight-release data.

    Label conventions
    -----------------
    * ``"Top Feature"`` label  → Section 2 (Top Feature of the Month).
    * ``"New Insight"`` label  → Section 3 (New Insights ranked by impact).
    All other issues feed Section 1 (Executive Summary context).
    """

    # ── Section 1: All story/issue context ───────────────────────────────────
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

    # ── Section 2: Top Feature (label = "Top Feature") ────────────────────────
    top_feature_df   = _filter_by_label(issues_df, "Top Feature")
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

    # ── Section 3: New Insights (label = "New Insight") ───────────────────────
    new_insight_df   = _filter_by_label(issues_df, "New Insight")
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

    # ── Section 6: Looking Ahead – next-month data ────────────────────────────
    next_month_lines: list[str] = []
    if next_month_df is not None and not next_month_df.empty:
        next_seen_rules: set = set()
        for _, row in next_month_df.iterrows():
            rule   = row.get("rule_name", "")
            change = row.get("change_type", "")
            if rule and rule not in next_seen_rules:
                next_seen_rules.add(rule)
                next_month_lines.append(
                    f"- Rule Name:   {rule}\n"
                    f"  Target Type: {row.get('target_type', 'N/A')}\n"
                    f"  Change Type: {change}\n"
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
    - Do NOT include the instruction notes in the final output
    """
    return prompt.strip()


# ── LLM call ─────────────────────────────────────────────────────────────────

def generate_summary(issues_df: pd.DataFrame, next_month_df: pd.DataFrame | None = None) -> str:
    """Call the LLM and return the executive summary text."""
    client = _get_openai_client()
    prompt = _build_prompt(issues_df, next_month_df=next_month_df)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    return response.choices[0].message.content.strip()


def build_save_content(summary_text: str, month_label: str) -> str:
    """Format the summary for saving to a text file."""
    header = (
        f"Executive Summary Report – {month_label}\n"
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        + "=" * 80
        + "\n\n"
    )
    return header + summary_text


# ── Layout ────────────────────────────────────────────────────────────────────

def layout(df: pd.DataFrame):
    """Return the ``dcc.Tab`` component for the Executive Summary tab.

    ``df`` is accepted for backward compatibility but month options are now
    driven from the DB via ``get_month_options_from_db()``.
    """
    try:
        month_options = get_month_options_from_db()
    except Exception:
        # Fallback to in-memory DataFrame if DB is unavailable at import time
        month_options = extract_month_options(df)

    default_month = month_options[0]["value"] if month_options else None

    return dcc.Tab(
        label="Executive Summary",
        value="tab-executive-summary",
        children=html.Div(
            [
                # Hidden stores
                dcc.Store(id="exec-summary-text-store", data=""),

                html.H3(
                    "Executive Summary Generator",
                    style=release_dashboard_styles["section_header_style"],
                ),

                # Toolbar – controls panel
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
                                    "Select a month and click 'Generate Executive Summary' to begin."
                                ),
                                style={
                                    **release_dashboard_styles.get("table_style", {}),
                                    "padding": "20px",
                                    "background": "#ffffff",
                                    "borderRadius": "8px",
                                    "minHeight": "220px",
                                    "whiteSpace": "pre-wrap",
                                    "fontSize": "14px",
                                    "lineHeight": "1.6",
                                },
                            ),
                        ),
                    ],
                    style={"marginTop": "18px"},
                ),

                # Toast for save feedback
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
                        "top": 16,
                        "right": 16,
                        "zIndex": 9999,
                    },
                ),
            ],
            style=release_dashboard_styles["table_container_style"],
        ),
    )
