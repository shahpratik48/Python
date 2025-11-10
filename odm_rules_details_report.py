#!/usr/bin/env python3
"""
Generate the ODM rules details Excel report from ODM SQL assets hosted in GitLab
and upload the resulting data into sandbox_prj_smart_insights.odm_rules_details.
"""

from __future__ import annotations

import base64
import datetime as dt
import getpass
import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import gitlab  # type: ignore
import pandas as pd
import psycopg2  # type: ignore
from psycopg2.extras import execute_values  # type: ignore
import sqlparse  # type: ignore
from sqlglot import exp, parse_one  # type: ignore
from sqlglot.errors import ParseError  # type: ignore

GITLAB_URL = "https://devcloud.ubs.net"
ODM_PROJECT_PATH = (
    "ubs/gwma/smart-technology-and-analytics/staat-data-science/"
    "staat-ds-genesis/genesis-platform/odm-dags"
)
BRANCH = "odm-master"
SQL_PATH = "dags/odm/script/sql"

IGNORE_FILES = {"aa_nlg_review.sql"}
IGNORE_FOLDERS = {"odm_process"}

REPORT_COLUMNS: Sequence[str] = (
    "target_type",
    "insight_type",
    "profile_table",
    "target_key",
    "metric_name",
    "metric_type",
    "metric_val",
    "run_id",
    "insight_date_time",
    "insight_category",
    "insight_att_name",
    "insight_att_val",
    "profile_column",
    "logic",
)

PLACEHOLDER_PATTERN = re.compile(r"\{\{\s*([^\{\}]+?)\s*\}\}")


@dataclass
class ParsedRow:
    data: Dict[str, str]
    source_statement: str
    source_file: str


def mask_placeholders(sql_text: str) -> Tuple[str, Dict[str, str]]:
    """Replace Jinja placeholders with SQL-safe tokens and keep a reverse map."""
    mapping: Dict[str, str] = {}

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        token = f"__JINJA_{len(mapping)}__"
        mapping[token] = match.group(0)
        return token

    masked_sql = PLACEHOLDER_PATTERN.sub(_replace, sql_text)
    return masked_sql, mapping


def unmask_placeholders(text: str, mapping: Dict[str, str]) -> str:
    """Restore original Jinja placeholders in the provided text."""
    result = text
    for token, placeholder in mapping.items():
        result = result.replace(token, placeholder)
    return result


def collect_sql_file_paths(project: gitlab.v4.objects.Project, sql_root: str, ref: str) -> List[str]:
    """Recursively list SQL files under the provided path using the GitLab tree API."""
    stack = [sql_root]
    sql_files: List[str] = []

    while stack:
        current_path = stack.pop()
        for node in project.repository_tree(path=current_path, ref=ref, all=True):
            node_path: str = node["path"]
            node_type: str = node["type"]

            lower_parts = {part.lower() for part in node_path.split("/")}
            if any(folder in lower_parts for folder in IGNORE_FOLDERS):
                continue

            if node_type == "tree":
                stack.append(node_path)
            elif node_type == "blob" and node_path.lower().endswith(".sql"):
                if os.path.basename(node_path) in IGNORE_FILES:
                    continue
                sql_files.append(node_path)

    return sorted(sql_files)


def fetch_sql(project: gitlab.v4.objects.Project, path: str, ref: str) -> str:
    """Download and decode a SQL file from GitLab."""
    file_obj = project.files.get(file_path=path, ref=ref)
    return base64.b64decode(file_obj.content).decode("utf-8", errors="replace")


def parse_insert_statement(statement: str, source_file: str) -> List[ParsedRow]:
    """Parse INSERT statements that load into the ODM table and extract column mappings."""
    masked_sql, placeholder_map = mask_placeholders(statement)

    try:
        expression = parse_one(masked_sql, read="postgres")
    except ParseError as exc:
        logging.warning("Skipping statement in %s due to parse error: %s", source_file, exc)
        return []

    if not isinstance(expression, exp.Insert):
        return []

    target_table = unmask_placeholders(expression.this.sql(dialect="postgres"), placeholder_map)
    if "{{params.ODM_TABLE}}" not in target_table:
        return []

    columns_expr = expression.args.get("columns")
    if not columns_expr:
        logging.warning("Insert without explicit column list in %s; skipping.", source_file)
        return []

    target_columns = [
        unmask_placeholders(col.sql(dialect="postgres"), placeholder_map).strip('"').strip()
        for col in columns_expr
    ]

    payload = expression.args.get("expression")
    if payload is None:
        logging.warning("Insert without payload in %s; skipping.", source_file)
        return []

    logic_sql = unmask_placeholders(payload.sql(dialect="postgres"), placeholder_map)
    full_statement = unmask_placeholders(expression.sql(dialect="postgres"), placeholder_map)

    rows: List[ParsedRow] = []

    if isinstance(payload, exp.Select):
        row_values = [
            unmask_placeholders(item.sql(dialect="postgres"), placeholder_map).strip()
            for item in payload.expressions
        ]
        rows.extend(
            build_rows_from_values(
                target_columns,
                [row_values],
                logic_sql,
                full_statement,
                source_file,
            )
        )
    elif isinstance(payload, exp.Values):
        value_rows: List[List[str]] = []
        for tup in payload.expressions:
            value_rows.append(
                [
                    unmask_placeholders(item.sql(dialect="postgres"), placeholder_map).strip()
                    for item in tup.expressions
                ]
            )
        rows.extend(
            build_rows_from_values(
                target_columns,
                value_rows,
                logic_sql,
                full_statement,
                source_file,
            )
        )
    else:
        logging.warning(
            "Unsupported INSERT payload (%s) in %s; statement skipped.",
            type(payload).__name__,
            source_file,
        )

    return rows


def build_rows_from_values(
    target_columns: Sequence[str],
    value_rows: Sequence[Sequence[str]],
    logic_sql: str,
    full_statement: str,
    source_file: str,
) -> List[ParsedRow]:
    """Align column names and row values, trimming and normalizing for the report."""
    normalized_columns = [col.lower() for col in target_columns]
    missing = [col for col in REPORT_COLUMNS if col not in normalized_columns and col != "logic"]
    if missing:
        logging.debug(
            "Columns missing from INSERT target in %s: %s",
            source_file,
            ", ".join(missing),
        )

    parsed_rows: List[ParsedRow] = []

    for values in value_rows:
        row_map = {
            col.lower(): (values[idx] if idx < len(values) else "")
            for idx, col in enumerate(target_columns)
        }

        report_data = {column: row_map.get(column, "") for column in REPORT_COLUMNS}
        report_data["logic"] = f"-- Source: {source_file}\n{logic_sql.strip()}"
        parsed_rows.append(
            ParsedRow(
                data=report_data,
                source_statement=full_statement,
                source_file=source_file,
            )
        )

    return parsed_rows


def extract_report_rows(sql_text: str, source_file: str) -> List[ParsedRow]:
    """Split a SQL file into statements and parse relevant INSERT statements."""
    rows: List[ParsedRow] = []
    for statement in sqlparse.split(sql_text):
        cleaned_statement = statement.strip()
        if not cleaned_statement:
            continue
        rows.extend(parse_insert_statement(cleaned_statement, source_file))
    return rows


def create_report_dataframe(rows: Iterable[ParsedRow]) -> pd.DataFrame:
    """Build the final DataFrame in the desired column order."""
    data = [row.data for row in rows]
    return pd.DataFrame(data, columns=REPORT_COLUMNS)


def write_excel_report(df: pd.DataFrame, directory: str = ".") -> str:
    """Persist the DataFrame to an Excel file and return the absolute path."""
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"odm_rules_details_{timestamp}.xlsx"
    output_path = os.path.abspath(os.path.join(directory, filename))

    logging.info("Writing Excel report to %s", output_path)
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="odm_rules_details")

    return output_path


def upload_dataframe_to_db(df: pd.DataFrame, db_password: str) -> None:
    """Drop and recreate the destination table, then load the DataFrame contents."""
    db_config = {
        "host": "greenulum-rdso.zur_swissbank.com",
        "port": 5432,
        "dbname": "gprdsp",
        "user": "ds_rdsp_dev",
        "password": db_password,
    }

    table_fqn = "sandbox_prj_smart_insights.odm_rules_details"
    owner_role = "erd_gpdb_prj_smart_insights"
    reader_role = "erd_gpdb_prj_smart_insights_ro"

    logging.info("Connecting to GPDB cluster to refresh %s", table_fqn)

    with psycopg2.connect(**db_config) as conn:
        conn.autocommit = False
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {table_fqn};")

            column_definitions = ", ".join(f'"{column}" TEXT' if column != "logic" else '"logic" TEXT'
                                           for column in REPORT_COLUMNS)
            cur.execute(f"CREATE TABLE {table_fqn} ({column_definitions});")

            if not df.empty:
                columns_clause = ", ".join(f'"{column}"' for column in REPORT_COLUMNS)
                values = [tuple(row[column] for column in REPORT_COLUMNS) for _, row in df.iterrows()]
                execute_values(
                    cur,
                    f"INSERT INTO {table_fqn} ({columns_clause}) VALUES %s",
                    values,
                )

            cur.execute(f"ALTER TABLE {table_fqn} OWNER TO {owner_role};")
            cur.execute(f"GRANT SELECT ON {table_fqn} TO {reader_role};")

        conn.commit()
        logging.info("Table %s refreshed successfully.", table_fqn)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    private_token = getpass.getpass("Enter your private token: ")
    gl = gitlab.Gitlab(GITLAB_URL, private_token=private_token)

    logging.info("Fetching ODM project metadata...")
    project = gl.projects.get(ODM_PROJECT_PATH)

    logging.info("Collecting SQL file paths under %s...", SQL_PATH)
    sql_files = collect_sql_file_paths(project, SQL_PATH, BRANCH)
    logging.info("Found %d SQL files to inspect.", len(sql_files))

    all_rows: List[ParsedRow] = []
    for path in sql_files:
        sql_content = fetch_sql(project, path, BRANCH)
        rows = extract_report_rows(sql_content, path)
        if rows:
            logging.info("Extracted %d row(s) from %s", len(rows), path)
            all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError("No ODM rule rows were parsed from the SQL sources.")

    df = create_report_dataframe(all_rows)
    excel_path = write_excel_report(df)
    logging.info("Report contains %d rows.", len(df))

    db_password = getpass.getpass("Enter Password for DB User: ")
    upload_dataframe_to_db(df, db_password)

    logging.info("Process completed. Excel report saved at: %s", excel_path)


if __name__ == "__main__":
    main()
