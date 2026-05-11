"""
Generate LLM summaries for GitLab issues already stored in Greenplum.
- Reads rows from sandbox_prj_smart_insights.staat_insight_release where issue_summary is empty.
- Calls an OpenAI GPT-4.x model using title + issue_description.
- Writes the summary back into issue_summary.

Prereqs:
- Set env OPENAI_API_KEY for the model.
- Set env GREENPLUM_PASSWORD for the DB.
- Install deps: pip install openai psycopg2-binary
"""
from __future__ import annotations

import os
import logging
import textwrap
from typing import Optional
import getpass

import psycopg2
from psycopg2.extras import DictCursor
from openai import OpenAI
import os
import logging

from config import (
    GREENPLUM_HOST,
    GREENPLUM_PORT,
    GREENPLUM_DB,
    GREENPLUM_USER,
    is_iteration_ending_this_week,
)

# Airflow detection (do not import Variable at module import time unguarded)
try:
    from airflow.models import Variable  # type: ignore
    _HAS_AIRFLOW = True
except Exception:
    Variable = None  # type: ignore
    _HAS_AIRFLOW = False


def is_running_in_airflow() -> bool:
    """Return True when executing inside an Airflow task context."""
    return _HAS_AIRFLOW and bool(os.environ.get("AIRFLOW_CTX_DAG_ID"))


def get_postgres_conn_id() -> str:
    """Return the Airflow Postgres connection id when running in Airflow.

    Falls back to raising when not running in Airflow.
    """
    if not is_running_in_airflow():
        raise RuntimeError("Not running in Airflow; Postgres conn id unavailable")
    return Variable.get("IKG_POSTGRES_CONN_ID")


def get_greenplum_credentials() -> Optional[dict]:
    """Resolve Greenplum credentials from Airflow Postgres connection when in Airflow.

    Returns a dict with keys: host, port, db, user, password. Returns None when
    not running in Airflow or when resolution fails.
    """
    if not is_running_in_airflow():
        return None
    try:
        from airflow.providers.postgres.hooks.postgres import PostgresHook  # type: ignore
        conn_id = get_postgres_conn_id()
        if not conn_id:
            return None
        hook = PostgresHook(postgres_conn_id=conn_id)
        conn = hook.get_connection(conn_id)
        return {
            "host": conn.host,
            "port": int(conn.port) if conn.port else 5432,
            "db": conn.schema,
            "user": conn.login,
            "password": conn.password,
        }
    except Exception as e:
        logger.warning(f"Could not resolve Greenplum credentials from Airflow connection: {e}")
        return None

# Greenplum connection details (aligned with the notebook)
GREENPLUM_SCHEMA = "sandbox_prj_smart_insights"
OUTPUT_TABLE1 = "staat_insight_release"

# Model configuration
MODEL_NAME = "gpt-4.1"  # GPT-4.x family; adjust if you prefer a different GPT-4 model
MAX_TOKENS = 15000
TEMPERATURE = 0.1

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_db_connection() -> psycopg2.extensions.connection:
    """Create a Greenplum connection using env GREENPLUM_PASSWORD."""
    # When running inside Airflow prefer resolving credentials from the
    # Airflow Postgres connection (PostgresHook). Locally we keep the
    # existing behavior (hardcoded/password from env as before).
    creds = get_greenplum_credentials()
    if creds:
        conn = psycopg2.connect(
            host=creds.get("host") or GREENPLUM_HOST,
            port=creds.get("port") or GREENPLUM_PORT,
            dbname=creds.get("db") or GREENPLUM_DB,
            user=creds.get("user") or GREENPLUM_USER,
            password=creds.get("password"),
        )
        return conn

    # Local behavior: prompt the user
    password = getpass.getpass("Enter Greenplum database password: ")
    if not password:
        raise RuntimeError("Greenplum password is required for local runs")

    conn = psycopg2.connect(
        host=GREENPLUM_HOST,
        port=GREENPLUM_PORT,
        dbname=GREENPLUM_DB,
        user=GREENPLUM_USER,
        password=password,
    )
    return conn


def fetch_iteration_end_date_for_latest_batch(cur: DictCursor, iteration: str) -> str | None:
    """Return the iteration_end_date of the latest batch for a given iteration value.

    Used by the iteration-week guard to decide whether a Previous-iteration
    summarize run should be skipped.

    Parameters
    ----------
    cur : DictCursor
    iteration : str  e.g. 'Previous' or 'Current'

    Returns
    -------
    str | None
        The iteration_end_date as 'YYYY-MM-DD', or None if no rows found.
    """
    query = f"""
        WITH latest_batch AS (
            SELECT COALESCE(MAX(batch), 0) AS batch
            FROM {GREENPLUM_SCHEMA}.{OUTPUT_TABLE1}
            WHERE LOWER(iteration) = LOWER(%s)
        )
        SELECT CAST(s.iteration_end_date AS VARCHAR) AS iteration_end_date
        FROM {GREENPLUM_SCHEMA}.{OUTPUT_TABLE1} s
        CROSS JOIN latest_batch b
        WHERE COALESCE(s.batch, 0) = b.batch
          AND LOWER(s.iteration) = LOWER(%s)
        LIMIT 1
    """
    cur.execute(query, (iteration, iteration))
    row = cur.fetchone()
    if row:
        return str(row["iteration_end_date"]).strip()
    return None


def fetch_issues_missing_summary(cur: DictCursor) -> list[dict]:
    """Fetch issues from the latest batch that need summaries."""
    query = f"""
        WITH latest_batch AS (
            SELECT COALESCE(MAX(batch), 0) AS batch
            FROM {GREENPLUM_SCHEMA}.{OUTPUT_TABLE1}
        )
        SELECT s.id_x, s.title, s.issue_description, s.batch
        FROM {GREENPLUM_SCHEMA}.{OUTPUT_TABLE1} s
        CROSS JOIN latest_batch b
        WHERE COALESCE(s.issue_summary, '') = ''
          AND COALESCE(s.batch, 0) = b.batch
        ORDER BY s.id_x
    """
    cur.execute(query)
    rows = cur.fetchall()
    return rows


def build_prompt(title: str, description: str) -> str:
    """Create a concise prompt for summarization."""
    description = description or ""
    wrapped_desc = textwrap.shorten(description, width=2000, placeholder=" …")
    prompt = f"""
You are an expert release manager. Summarize the issue for a status report.
- Keep it under ~80 words.
- Focus on problem, impact, and outcome/next step if present.
- Do not invent details.

Title: {title}
Description:
{wrapped_desc}
"""
    return prompt.strip()


def generate_summary(client: OpenAI, title: str, description: str) -> str:
    """Call the OpenAI chat model to produce a summary."""
    prompt = build_prompt(title, description)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    return response.choices[0].message.content.strip()


def update_issue_summary(cur: DictCursor, issue_id: str, batch: int, summary: str) -> None:
    """Update issue_summary for a given issue id_x and batch to avoid cross-batch edits."""
    update_sql = f"""
        UPDATE {GREENPLUM_SCHEMA}.{OUTPUT_TABLE1}
        SET issue_summary = %s
        WHERE id_x = %s AND COALESCE(batch, 0) = %s
    """
    cur.execute(update_sql, (summary, issue_id, batch))


    


def main():
    # ------------------------------------------------------------------
    # Iteration-week guard
    # ------------------------------------------------------------------
    # When this task runs as part of a "Previous" iteration DAG on a
    # Tuesday or Wednesday that is NOT the iteration-ending week, we
    # must not process any new summaries.  Doing so would touch the new
    # batch that was (wrongly) inserted for the Previous iteration,
    # reinforcing the override.  Because generate_release_reports already
    # skips the DB insert on such runs, there will be no new rows to
    # summarize anyway — but we add a matching guard here for safety and
    # to produce an explicit log message.
    #
    # We detect "Previous" DAGs by inspecting the Airflow DAG ID
    # (AIRFLOW_CTX_DAG_ID env var).  If the word "previous" appears in
    # the DAG ID we treat this as a Previous-iteration run and apply the
    # guard.  When running outside Airflow (local / notebook) the env var
    # is absent and the guard is not applied.
    #
    # The iteration_end_date used for the check is read from the DB
    # (latest batch for iteration='Previous') so this script does not
    # need any extra arguments from the DAG.
    dag_id = os.environ.get("AIRFLOW_CTX_DAG_ID", "")
    if "previous" in dag_id.lower():
        logger.info("Detected Previous-iteration DAG (dag_id=%s). "
                    "Applying iteration-week guard …", dag_id)
        try:
            with get_db_connection() as _guard_conn:
                with _guard_conn.cursor(cursor_factory=DictCursor) as _guard_cur:
                    prev_end_date = fetch_iteration_end_date_for_latest_batch(
                        _guard_cur, "Previous"
                    )
            if prev_end_date and not is_iteration_ending_this_week(prev_end_date):
                logger.info("=" * 60)
                logger.info(
                    "⏭️  SKIPPING summarize_issues: iteration='Previous' and the "
                    "iteration (end date=%s) is NOT ending in the current ISO week. "
                    "Skipping to avoid processing summaries for a batch that should "
                    "not have been inserted.", prev_end_date
                )
                logger.info("=" * 60)
                return  # exit without doing any work
            else:
                logger.info(
                    "Iteration-week guard: iteration_end_date=%s is in the current "
                    "ISO week (or could not be determined). Proceeding with summarization.",
                    prev_end_date,
                )
        except Exception as guard_exc:
            # If the guard check itself fails, log a warning but do NOT
            # block the run — fail-safe behaviour.
            logger.warning(
                "Could not evaluate iteration-week guard (will proceed): %s", guard_exc
            )


    # Get OpenAI credentials: prefer Airflow Connection when running in Airflow,
    # otherwise use environment variables or hardcoded defaults (local behavior)
    if is_running_in_airflow():
        try:
            from airflow.models import Connection  
            openai_conn = Connection.get_connection_from_secrets("STAAT-DS-OPENAI-LLM")
            api_key = openai_conn.password
            base_url = openai_conn.host
            if not api_key:
                raise RuntimeError("OpenAI API key not found in Airflow connection 'STAAT-DS-OPENAI-LLM'")
            logger.info("Using OpenAI credentials from Airflow connection 'STAAT-DS-OPENAI-LLM'")
        except Exception as e:
            logger.error(f"Failed to retrieve OpenAI connection from Airflow: {e}")
            raise RuntimeError("Could not retrieve OpenAI credentials from Airflow") from e
    else:
        # Local behavior: use environment variable or hardcoded fallback
        api_key = os.environ.get("OPENAI_API_KEY", '')
        base_url = 'https://cirruspl-staat-ste-dev-ai.openai.azure.com/'
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY env var is required for local runs")

    client = OpenAI(api_key=api_key, base_url=base_url)

    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            rows = fetch_issues_missing_summary(cur)
            if not rows:
                logger.info("No issues found with empty issue_summary. Nothing to do.")
                return

            logger.info("Found %d issues missing summaries", len(rows))

            for row in rows:
                issue_id = str(row["id_x"])
                batch = int(row.get("batch") or 0)
                title = row.get("title") or ""
                description = row.get("issue_description") or ""

                try:
                    summary = generate_summary(client, title, description)
                    update_issue_summary(cur, issue_id, batch, summary)
                    logger.info("Updated issue %s (batch %s)", issue_id, batch)
                except Exception as exc:
                    logger.error("Failed to summarize issue %s: %s", issue_id, exc)
                    conn.rollback()
                else:
                    conn.commit()

    logger.info("All available summaries applied.")


if __name__ == "__main__":
    main()
