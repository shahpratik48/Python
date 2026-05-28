# dags/ikg/sprint_report/scripts/executive_summary.py
# -*- coding: utf-8 -*-
"""Executive Summary – shared logic for the Dash UI and the monthly DAG.

Environment detection
─────────────────────
The module uses the same two-phase detection pattern as the other scripts
in this folder (swat_issues_report.py, release_gitlab_report.py):

  1. _HAS_AIRFLOW  – True when the Airflow package is importable.
  2. is_running_in_airflow()  – True only when _HAS_AIRFLOW AND
     AIRFLOW_CTX_DAG_ID env var is set (injected by Airflow at run-time).

  Production / DAG  (is_running_in_airflow() = True)
    GP         →  PostgresHook  via  Variable.get("GP_Dash_connect")
    Schema     →  Variable.get("IKG_DASHBOARD_SCHEMA")
    OpenAI     →  Connection.get_connection_from_secrets("STAAT-DS-OPENAI-LLM")
    Run month  →  datetime.now()
    Email to   →  Variable "staat_monthly_executive_summary_email" (UAT/Prod)
               →  LOCAL_EMAIL_RECIPIENTS list (Dev/local Airflow)

  Local / Jupyter  (is_running_in_airflow() = False)
    GP         →  psycopg2 via GP_* env vars or getpass
    OpenAI     →  env vars OPENAI_API_KEY / OPENAI_BASE_URL
    Run month  →  user input
    Email      →  not sent (summary saved to file only)

Email
─────
  Subject: "Executive Summary – <Month Year> – <Environment>"
  Body   : richly styled HTML; the full summary is in the email body
           (no attachments).
  Sending: airflow.utils.email.send_email (only when in Airflow).
"""

from __future__ import annotations

import os
import re
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
# Airflow detection  (mirrors swat_issues_report.py / release_gitlab_report.py)
# ---------------------------------------------------------------------------

try:
    from airflow.models import Connection, Variable                       # type: ignore
    from airflow.configuration import conf                                # type: ignore
    _HAS_AIRFLOW = True
except Exception:
    Variable   = None   # type: ignore
    Connection = None   # type: ignore
    conf       = None   # type: ignore
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
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME  = "gpt-4.1"
MAX_TOKENS  = 12000
TEMPERATURE = 0.1

POSTGRES_CONN_ID_VAR    = "GP_Dash_connect"
IKG_SCHEMA_VAR          = "IKG_DASHBOARD_SCHEMA"
OPENAI_CONN_ID          = "STAAT-DS-OPENAI-LLM"
DEFAULT_OPENAI_BASE_URL = "https://cirruspl-staat-ste-dev-ai.openai.azure.com/openai/v1/"
DEFAULT_GP_SCHEMA       = "core_ikg"

# Email
EMAIL_RECIPIENTS_VAR  = "staat_monthly_executive_summary_email"
LOCAL_EMAIL_RECIPIENTS = ["pratik.shah.2@ubs.com"]
EMAIL_FROM_ADDR        = "staat-insights@ubs.com"

_client: Optional[OpenAI] = None


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def _get_openai_client() -> OpenAI:
    """Production: Airflow Connection.  Local: env vars."""
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
    """Production: PostgresHook.  Local: psycopg2 + env vars / getpass."""
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
    return psycopg2.connect(host=host, port=port, dbname=dbname, user=user, password=password)


def _get_schema() -> str:
    if is_running_in_airflow():
        return Variable.get(IKG_SCHEMA_VAR)
    return os.environ.get("GP_SCHEMA", DEFAULT_GP_SCHEMA)


# ---------------------------------------------------------------------------
# Environment & email helpers
# ---------------------------------------------------------------------------

def _get_environment() -> str:
    """Return 'Dev', 'UAT', or 'Prod' by comparing Airflow webserver URL.
    Follows the same pattern as swat_issues_report.py and release_gitlab_report.py.
    """
    if not is_running_in_airflow():
        return "Dev"
    try:
        base_url = conf.get("webserver", "base_url")
        if base_url == Variable.get("AIRFLOW_UAT_URL", default_var=""):
            return "Dev"   # UAT Airflow instance → Dev label (matches existing convention)
        if base_url == Variable.get("AIRFLOW_PROD_URL", default_var=""):
            return "Prod"
    except Exception as exc:
        logger.warning("Could not determine environment: %s", exc)
    return "Dev"


def get_email_recipients() -> list:
    """Return the list of email recipients.

    Dev / local Airflow  →  LOCAL_EMAIL_RECIPIENTS  (pratik.shah.2@ubs.com only)
    UAT / Prod           →  Airflow Variable  staat_monthly_executive_summary_email
    Not in Airflow       →  empty list  (email not sent outside Airflow)
    """
    if not is_running_in_airflow():
        return []
    env = _get_environment()
    if env != "Prod":          # Dev (UAT-mapped) → local list only
        return LOCAL_EMAIL_RECIPIENTS
    try:
        raw        = Variable.get(EMAIL_RECIPIENTS_VAR, default_var="")
        recipients = [r.strip() for r in raw.split(",") if r.strip()]
        return recipients if recipients else LOCAL_EMAIL_RECIPIENTS
    except Exception as exc:
        logger.warning("Could not read %s: %s", EMAIL_RECIPIENTS_VAR, exc)
        return LOCAL_EMAIL_RECIPIENTS


# ---------------------------------------------------------------------------
# SQL helpers
# ---------------------------------------------------------------------------

def _build_month_query(month_value: str, schema: str) -> str:
    staat = f"{schema}.staat_insight_release"
    odm   = f"{schema}.odm_release_details"
    return f"""
    WITH target_iterations AS (
        SELECT iteration_end_date, MAX(batch) AS max_batch
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
        s.id_x, s.title, s.labels, s.issue_summary, s.state, s.weight,
        s.prod_release_date, s.iteration_end_date, s.iteration_start_date, s.batch,
        o.rule_name, o.target_type, o.change_type
    FROM staat_latest s
    LEFT JOIN odm_latest o
        ON  s.id_x               = o.issue_id
        AND s.iteration_end_date = o.iteration_end_date
    ORDER BY s.iteration_end_date, s.id_x
    """


def _build_reactivated_query(insight_types: list, schema: str) -> str:
    staat     = f"{schema}.staat_insight_release"
    odm       = f"{schema}.odm_release_details"
    in_clause = ", ".join(f"'{t}'" for t in insight_types)
    return f"""
    WITH odm_ranked AS (
        SELECT o.*,
               MAX(o.batch) OVER (PARTITION BY o.iteration_end_date) AS max_batch
        FROM {odm} o
        WHERE o.rule_name IN ({in_clause})
    ),
    odm_latest AS (SELECT * FROM odm_ranked WHERE batch = max_batch)
    SELECT
        o.issue_id AS id_x, o.iteration_end_date, o.batch,
        o.rule_name, o.target_type, o.change_type,
        s.title, s.issue_summary, s.labels, s.state, s.weight,
        s.prod_release_date, s.iteration_start_date
    FROM odm_latest o
    LEFT JOIN {staat} s
        ON  o.issue_id           = s.id_x
        AND o.iteration_end_date = s.iteration_end_date
        AND o.batch              = s.batch
    ORDER BY o.rule_name, o.iteration_end_date, o.issue_id
    """


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def get_reactivated_insights(schema: str, conn, year: int, month: int) -> pd.DataFrame:
    sql = f"""
        SELECT DISTINCT insight_type
        FROM {schema}.odm_exclusion_insight_type
        WHERE EXTRACT(YEAR  FROM last_upd_dte::date) = {year}
          AND EXTRACT(MONTH FROM last_upd_dte::date) = {month}
          AND is_curr = 0
    """
    try:
        df = pd.read_sql_query(sql, conn)
    except Exception as exc:
        logger.warning("[reactivated] exclusion table query failed: %s", exc)
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    insight_types = df["insight_type"].dropna().str.strip().tolist()
    logger.info("[reactivated] Found %d reactivated type(s): %s", len(insight_types), insight_types)
    try:
        details = pd.read_sql_query(_build_reactivated_query(insight_types, schema), conn)
        details["change_type"] = details["change_type"].replace({"added": "new"})
        return details
    except Exception as exc:
        logger.warning("[reactivated] story detail query failed: %s", exc)
        return pd.DataFrame()


def get_exec_summary_data(month_value: str, allow_prompt: bool = True) -> tuple:
    """Return (current_df, next_month_df, reactivated_df)."""
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
    schema = _get_schema()
    conn   = _get_gp_conn()
    df = pd.read_sql_query(
        f"""SELECT DISTINCT TO_CHAR(prod_release_date::date, 'YYYY-MM') AS month_val
            FROM {schema}.staat_insight_release
            WHERE prod_release_date IS NOT NULL
            ORDER BY month_val DESC""",
        conn,
    )
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
# Legacy helpers (Dash compatibility)
# ---------------------------------------------------------------------------

def extract_month_options(df: pd.DataFrame) -> list:
    if df is None or df.empty or "prod_release_date" not in df.columns:
        return []
    dates = pd.to_datetime(df["prod_release_date"], errors="coerce").dropna()
    if dates.empty:
        return []
    return [
        {"label": p.to_timestamp().strftime("%B %Y"), "value": str(p)}
        for p in sorted(dates.dt.to_period("M").unique(), reverse=True)
    ]


def filter_by_month(df: pd.DataFrame, month_value: str) -> pd.DataFrame:
    if df is None or df.empty or "prod_release_date" not in df.columns:
        return pd.DataFrame()
    dates = pd.to_datetime(df["prod_release_date"], errors="coerce")
    return df.loc[dates.dt.to_period("M").astype(str) == month_value]


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def _has_label(labels_value, target: str) -> bool:
    if not labels_value or not isinstance(labels_value, str):
        return False
    return target.lower() in [lbl.strip().lower() for lbl in labels_value.split(",")]


def _filter_by_label(df: pd.DataFrame, label: str) -> pd.DataFrame:
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
    if reactivated_df is None or reactivated_df.empty:
        return "No reactivated insights found for this period."
    skip  = skip_rules or set()
    lines: list = []
    rule_num = 0
    for rule_name, group in reactivated_df.groupby("rule_name", sort=True):
        if rule_name in skip:
            continue
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
    # Section 1 – all stories
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

    # Section 2 – Top Feature
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

    # Section 3 – New Insights
    ni_df    = _filter_by_label(issues_df, "New Insight")
    ni_lines: list = []
    seen_rules: set = set()
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

    # Reactivated (deduped)
    reactivated_text = _build_reactivated_text(reactivated_df, skip_rules=seen_rules)

    # Section 6 – Coming Soon
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
        else "No next-month data \u2013 provide directional focus areas based on current trends."
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
    - Brief description: <answer>
    - Main benefit / value: <answer>

    3. New Insights (ranked by importance/impact, including reactivated insights):
    For each new or reactivated insight:
    - Insight name: <answer>
    - Brief description: <answer>
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
    - Always include a single space after every colon (:) in your response
    - Be concise and business-focused
    - Rank insights by business impact and value
    - Use clear, non-technical language where possible
    - Focus on outcomes and benefits, not just features
    - If certain sections have no relevant data, state "No significant changes in this area"
    - Ensure all answers are data-driven based on the provided information
    - Section 1 Executive Summary: 2 sentences per question
    - Section 2 Top Feature: pick the single most impactful item from the "Top Feature" labelled issues
    - Section 3 New Insights: list each insight on a separate line; include reactivated insights at the end, marked as "Reactivated". Bundle multiple stories for the same rule under one entry
    - Section 6 New Insights \u2013 Coming Soon: simple list of Insight Titles only. If no next-month data, provide directional focus areas
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
    client   = _get_openai_client()
    prompt   = _build_prompt(issues_df, next_month_df=next_month_df, reactivated_df=reactivated_df)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    return response.choices[0].message.content.strip()


def build_save_content(summary_text: str, month_label: str) -> str:
    header = (
        f"Executive Summary Report \u2013 {month_label}\n"
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        + "=" * 80 + "\n\n"
    )
    return header + summary_text


# ---------------------------------------------------------------------------
# Email – HTML formatter
# ---------------------------------------------------------------------------

# One accent colour per section number (1-indexed)
_SECTION_COLORS = [
    "#002060",  # 1  Executive Summary   – dark navy
    "#0072CE",  # 2  Top Feature         – UBS blue
    "#006341",  # 3  New Insights        – green
    "#C55A11",  # 4  Process Improvement – burnt orange
    "#7B2D8B",  # 5  Stability           – purple
    "#156082",  # 6  Coming Soon         – teal
]


def _format_summary_as_html(summary_text: str) -> str:
    """Convert the plain-text LLM summary into styled inner HTML.

    Handles:
    • Numbered section headers  (N. Title  or  N. Title:)
    • Bullet key-value pairs    (- Key: Value)
    • Plain bullets             (- text)
    • Status badges             (Status: New | Reactivated)
    • Prose / sub-headings
    """
    lines   = summary_text.strip().split("\n")
    parts: list = []
    in_ul      = False
    in_section = False
    cur_color  = _SECTION_COLORS[0]

    def _close_ul() -> None:
        nonlocal in_ul
        if in_ul:
            parts.append("</ul>")
            in_ul = False

    def _close_section() -> None:
        nonlocal in_section
        if in_section:
            _close_ul()
            parts.append("  </div>")   # inner padding div
            parts.append("</div>")     # section card
            in_section = False

    def _open_ul() -> None:
        nonlocal in_ul
        if not in_ul:
            parts.append(
                '<ul style="margin:8px 0 4px 0;padding:0;list-style:none;">'
            )
            in_ul = True

    def _status_badge(value: str) -> str:
        v_lower = value.lower()
        if "reactivated" in v_lower:
            bg, fg = "#fff3e0", "#c55a11"
        elif "new" in v_lower:
            bg, fg = "#e8f5e9", "#2e7d32"
        else:
            bg, fg = "#e3f2fd", "#0072ce"
        return (
            f'<span style="background:{bg};color:{fg};padding:2px 10px;'
            f'border-radius:12px;font-size:11px;font-weight:700;'
            f'letter-spacing:0.5px;border:1px solid {fg}40;">{value}</span>'
        )

    for line in lines:
        s = line.rstrip()

        # ── Empty line ─────────────────────────────────────────────────────
        if not s:
            _close_ul()
            continue

        # ── Section header:  "N. Title"  or  "N. Title:" ──────────────────
        m = re.match(r"^(\d+)\.\s+(.+)", s)
        if m:
            _close_section()
            num   = int(m.group(1))
            title = m.group(2).rstrip(":").strip()
            color = _SECTION_COLORS[(num - 1) % len(_SECTION_COLORS)]
            cur_color = color
            parts.append(
                f'<div style="margin-bottom:20px;border-left:4px solid {color};'
                f'background:#ffffff;border-radius:0 8px 8px 0;'
                f'box-shadow:0 1px 4px rgba(0,0,0,0.06);">'
            )
            parts.append(
                f'  <div style="background:linear-gradient(to right,{color}14,transparent);'
                f'padding:11px 20px;border-bottom:1px solid {color}22;">'
            )
            parts.append(
                f'    <h2 style="margin:0;font-size:15px;font-weight:700;'
                f'color:{color};line-height:1.3;">'
                f'<span style="background:{color};color:#fff;border-radius:50%;'
                f'display:inline-flex;align-items:center;justify-content:center;'
                f'width:24px;height:24px;font-size:12px;font-weight:800;'
                f'margin-right:10px;flex-shrink:0;">{num}</span>'
                f'{title}</h2>'
            )
            parts.append("  </div>")
            parts.append('  <div style="padding:12px 20px 8px 20px;">')
            in_section = True
            continue

        # ── Bullet line ─────────────────────────────────────────────────────
        bm = re.match(r"^[-\u2022\u25b8]\s+(.+)", s)
        if bm:
            content = bm.group(1).strip()
            _open_ul()
            kv = re.match(r"^([^:]+?):\s*(.+)", content)
            if kv:
                key = kv.group(1).strip()
                val = kv.group(2).strip()
                if key.lower() == "status":
                    val_html = _status_badge(val)
                else:
                    val_html = f'<span style="color:#444;line-height:1.5;">{val}</span>'
                parts.append(
                    f'<li style="padding:6px 0;border-bottom:1px solid #f4f4f4;'
                    f'font-size:13.5px;display:flex;align-items:flex-start;gap:4px;">'
                    f'<strong style="color:#222;min-width:190px;flex-shrink:0;'
                    f'padding-right:8px;">{key}:</strong>'
                    f'{val_html}</li>'
                )
            else:
                parts.append(
                    f'<li style="padding:6px 0;font-size:13.5px;color:#333;'
                    f'line-height:1.5;display:flex;align-items:flex-start;">'
                    f'<span style="color:{cur_color};margin-right:8px;font-size:11px;'
                    f'padding-top:3px;">&#9658;</span>'
                    f'<span>{content}</span></li>'
                )
            continue

        # ── Prose / sub-heading ─────────────────────────────────────────────
        _close_ul()
        parts.append(
            f'<p style="margin:6px 0 4px 0;font-size:13px;color:#666;'
            f'font-style:italic;">{s}</p>'
        )

    _close_section()
    return "\n".join(parts)


def build_email_html(summary_text: str, month_label: str, environment: str) -> str:
    """Return a complete styled HTML email body."""
    env_badge_colors = {
        "Prod": ("#c41e3a", "#ffffff"),
        "UAT" : ("#e07b00", "#ffffff"),
        "Dev" : ("#28a745", "#ffffff"),
    }
    badge_bg, badge_fg = env_badge_colors.get(environment, ("#003087", "#ffffff"))
    generated_at = datetime.now().strftime("%B %d, %Y at %H:%M")
    summary_html = _format_summary_as_html(summary_text)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Executive Summary \u2013 {month_label}</title>
</head>
<body style="margin:0;padding:0;background-color:#eef1f6;
     font-family:'Helvetica Neue',Helvetica,Arial,sans-serif;">

<table width="100%" cellpadding="0" cellspacing="0" border="0"
       style="background:#eef1f6;">
  <tr><td align="center" style="padding:28px 12px;">

    <table width="660" cellpadding="0" cellspacing="0" border="0"
           style="max-width:660px;width:100%;background:#ffffff;
                  border-radius:10px;overflow:hidden;
                  box-shadow:0 4px 18px rgba(0,0,0,0.12);">

      <!-- ═══════════════════ HEADER ═══════════════════ -->
      <tr>
        <td style="background:linear-gradient(135deg,#001a4d 0%,#002e6e 55%,#00529b 100%);
                   padding:30px 36px 26px 36px;">
          <table width="100%" cellpadding="0" cellspacing="0" border="0">
            <tr>
              <td>
                <div style="font-size:10.5px;font-weight:600;color:#7eb3e0;
                            letter-spacing:2px;text-transform:uppercase;
                            margin-bottom:8px;">
                  STAAT Insights &nbsp;|&nbsp; Monthly Report
                </div>
                <h1 style="margin:0 0 6px 0;font-size:26px;font-weight:800;
                           color:#ffffff;letter-spacing:-0.5px;line-height:1.2;">
                  Executive Summary
                </h1>
                <div style="font-size:16px;color:#a8cef0;font-weight:400;
                            margin-top:4px;">
                  {month_label}
                </div>
              </td>
              <td style="text-align:right;vertical-align:top;padding-left:16px;">
                <span style="background:{badge_bg};color:{badge_fg};
                             padding:5px 14px;border-radius:20px;
                             font-size:11px;font-weight:700;
                             letter-spacing:1px;text-transform:uppercase;
                             white-space:nowrap;">
                  {environment}
                </span>
              </td>
            </tr>
          </table>
        </td>
      </tr>

      <!-- ═══════════════════ INTRO ═══════════════════ -->
      <tr>
        <td style="background:#f7f9fc;padding:16px 36px;
                   border-bottom:1px solid #e4e9f0;">
          <p style="margin:0;font-size:13.5px;color:#555;line-height:1.5;">
            Hi Team,<br>
            Please find below the <strong style="color:#002e6e;">
            {month_label} Executive Summary</strong> for the IKG Insights
            Release. This report is auto-generated by the STAAT Insights
            platform.
          </p>
        </td>
      </tr>

      <!-- ═══════════════════ SUMMARY BODY ═══════════════════ -->
      <tr>
        <td style="padding:28px 36px 20px 36px;">
          {summary_html}
        </td>
      </tr>

      <!-- ═══════════════════ DIVIDER ═══════════════════ -->
      <tr>
        <td style="padding:0 36px;">
          <div style="height:2px;
                      background:linear-gradient(to right,#002e6e,#7eb3e0,#eef1f6);">
          </div>
        </td>
      </tr>

      <!-- ═══════════════════ FOOTER ═══════════════════ -->
      <tr>
        <td style="padding:20px 36px 24px 36px;background:#f7f9fc;">
          <table width="100%" cellpadding="0" cellspacing="0" border="0">
            <tr>
              <td style="vertical-align:top;">
                <div style="font-size:13px;font-weight:700;color:#002e6e;
                            margin-bottom:3px;">
                  STAAT Insights Team
                </div>
                <div style="font-size:12px;color:#8a9bb5;line-height:1.6;">
                  Generated on {generated_at}<br>
                  Environment:&nbsp;
                  <span style="color:{badge_bg};font-weight:600;">
                    {environment}
                  </span>
                </div>
              </td>
              <td style="text-align:right;vertical-align:top;
                         font-size:11px;color:#b0bec5;line-height:1.6;">
                This is an automated report.<br>
                Please do not reply to this email.
              </td>
            </tr>
          </table>
        </td>
      </tr>

    </table><!-- /main card -->
  </td></tr>
</table><!-- /outer table -->

</body>
</html>"""


# ---------------------------------------------------------------------------
# Email – send helper  (mirrors swat_issues_report._send_email_with_attachments)
# ---------------------------------------------------------------------------

def _send_exec_summary_email(
    recipients: list,
    subject: str,
    html_body: str,
) -> None:
    """Send the executive summary email via Airflow's email utility.

    Only called when is_running_in_airflow() is True.
    """
    if not recipients:
        logger.warning("No recipients provided for email, skipping send.")
        return
    try:
        from airflow.utils.email import send_email  # type: ignore
        send_email(to=recipients, subject=subject, html_content=html_body, files=[])
        logger.info("Email sent to: %s", ", ".join(recipients))
    except Exception as exc:
        logger.error("Failed to send email via airflow.utils.email.send_email: %s", exc)


# ---------------------------------------------------------------------------
# DAG entry point
# ---------------------------------------------------------------------------

def generate_executive_summary(**kwargs) -> str:
    """Generate the executive summary for the current month, save it, and email it.

    Callable for the monthly DAG PythonOperator.
    Month is derived from datetime.now() – no user input required.
    """
    now         = datetime.now()
    month_value = now.strftime("%Y-%m")
    month_label = now.strftime("%B %Y")
    environment = _get_environment()

    logger.info("[exec_summary] Generating executive summary for %s (%s)…",
                month_label, environment)

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
    logger.info("[exec_summary] Next month rows    : %d",
                0 if df_next is None else len(df_next))

    summary_text = generate_summary(
        df_current,
        next_month_df  =df_next,
        reactivated_df =df_reactivated if (df_reactivated is not None
                                            and not df_reactivated.empty) else None,
    )

    # ── Save to file ──────────────────────────────────────────────────────
    content    = build_save_content(summary_text, month_label)
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path   = output_dir / f"executive_summary_{month_value}.txt"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(content)
    logger.info("[exec_summary] Saved \u2192 %s", out_path)

    # ── Send email (Airflow only) ─────────────────────────────────────────
    if is_running_in_airflow():
        timestamp  = now.strftime("%Y%m%d_%H%M%S")
        subject    = (
            f"Executive Summary \u2013 {month_label} \u2013 {environment}"
        )
        html_body  = build_email_html(summary_text, month_label, environment)
        recipients = get_email_recipients()
        logger.info("[exec_summary] Sending email (%s) to: %s",
                    environment, ", ".join(recipients))
        _send_exec_summary_email(recipients, subject, html_body)

    return summary_text


# ---------------------------------------------------------------------------
# Dash layout (production UI only)
# ---------------------------------------------------------------------------

def layout(df: pd.DataFrame):
    if not _DASH_AVAILABLE:
        raise RuntimeError(
            "Dash is not installed. layout() can only be called from the dashboard server."
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
                html.H3("Executive Summary Generator",
                        style=release_dashboard_styles["section_header_style"]),
                html.Div([
                    html.Div([
                        html.Span("Controls",
                                  style=release_dashboard_styles["panel_section_label_style"]),
                        html.Div([
                            html.Button("Generate Executive Summary",
                                        id="exec-summary-generate-btn", n_clicks=0,
                                        style=release_dashboard_styles["btn_primary_style"]),
                            html.Div(style=release_dashboard_styles["btn_divider_style"]),
                            html.Button("Save as TXT",
                                        id="exec-summary-save-btn", n_clicks=0,
                                        style=release_dashboard_styles["btn_export_style"],
                                        disabled=True),
                        ], style=release_dashboard_styles["btn_group_style"]),
                        html.Div([
                            html.Span("Month",
                                      style=release_dashboard_styles["dropdown_label_style"]),
                            dcc.Dropdown(id="exec-summary-month-filter",
                                         options=month_options, value=default_month,
                                         placeholder="Select month\u2026", clearable=True,
                                         style=release_dashboard_styles["dropdown_style"]),
                        ], style=release_dashboard_styles["dropdown_container_style"]),
                    ], style=release_dashboard_styles["panel_left_style"]),
                ], style=release_dashboard_styles["toolbar_style"]),
                html.Div([
                    dcc.Download(id="exec-summary-download"),
                    dcc.Loading(id="exec-summary-loading", type="default",
                                children=html.Div(
                                    id="exec-summary-output",
                                    children="Select a month and click 'Generate Executive Summary' to begin.",
                                    style={**release_dashboard_styles.get("table_style", {}),
                                           "padding": "20px", "background": "#ffffff",
                                           "borderRadius": "8px", "minHeight": "220px",
                                           "whiteSpace": "pre-wrap", "fontSize": "14px",
                                           "lineHeight": "1.6"})),
                ], style={"marginTop": "18px"}),
                dbc.Toast(id="exec-summary-save-toast", header="Executive Summary",
                          children="", is_open=False, duration=4000, icon="success",
                          dismissable=True,
                          style={"position": "fixed", "top": 16, "right": 16, "zIndex": 9999}),
            ],
            style=release_dashboard_styles["table_container_style"],
        ),
    )


if __name__ == "__main__":
    import getpass as _gp
    _raw = input("Month to summarise (e.g. '2026-05' or 'May 2026'): ").strip()
    try:
        _MONTH = str(pd.Period(_raw, "M"))
    except Exception:
        _MONTH = datetime.strptime(
            _raw.replace("-", " ").replace("/", " "), "%B %Y"
        ).strftime("%Y-%m")
    _label   = pd.Period(_MONTH, "M").to_timestamp().strftime("%B %Y")
    _dfc, _dfn, _dfr = get_exec_summary_data(_MONTH, allow_prompt=True)
    _text = generate_summary(
        _dfc,
        next_month_df=_dfn,
        reactivated_df=_dfr if not _dfr.empty else None,
    )
    print("\n" + "=" * 80)
    print(f"Executive Summary \u2013 {_label}")
    print("=" * 80 + "\n")
    print(_text)
    # Save HTML preview locally for inspection
    _html = build_email_html(_text, _label, "Dev")
    _out  = Path(f"exec_summary_preview_{_MONTH}.html")
    _out.write_text(_html, encoding="utf-8")
    print(f"\nHTML preview saved \u2192 {_out}")
