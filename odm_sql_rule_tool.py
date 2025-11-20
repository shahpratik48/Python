#!/usr/bin/env python3
"""
Automation utility for managing ODM SQL rules in GitLab.

Features
--------
* Authenticates with GitLab using a private token captured via getpass.
* Scans every SQL file under `dags/odm/script/sql` on `odm-master`.
* Pulls rule metadata for requested insight types from Postgres.
* Displays logic/rule_column pairs in a friendly table and exports them to Excel.
* Prompts for a rule_column update, creates a feature branch, and writes a
  modified SQL copy (e.g. `si_acc_bookshift_model_modified_1.sql`) with the
  requested value change inside the INSERT ... WHERE clause.
* Saves the modified SQL both to GitLab (new branch) and to the local workspace
  for immediate inspection.

The script is interactive and is intended to be executed from a secure, network
enabled environment that can reach both GitLab and the target Postgres cluster.
"""

from __future__ import annotations

import base64
import datetime as dt
import getpass
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Optional, Tuple

import gitlab
import pandas as pd
import psycopg2
from gitlab import exceptions as gl_exceptions
from psycopg2.extras import RealDictCursor

GITLAB_URL = "https://devcloud.ubs.net"
ODM_PROJECT_PATH = (
    "ubs/gwma/smart-technology-and-analytics/staat-data-science/"
    "staat-ds-genesis/genesis-platform/odm-dags"
)
BASE_BRANCH = "odm-master"
SQL_PATH = "dags/odm/script/sql"
BRANCH_PREFIX = "odm-branch-created-by-script-"
LOCAL_SQL_EXPORT_DIR = Path("generated_sql")

DB_CONFIG = {
    "host": "greenplum-rdsp.zur.swissbank.com",
    "port": 5432,
    "dbname": "gprdsp",
    "user": "ds_rdsp_dev",
    "schema": "sandbox_prj_smart_insights",
}
RULE_METADATA_TABLE = "sandbox_prj_smart_insights.odm_rule_metadata_auto_refresh"
TARGET_INSERT_KEYS = ("ilparams.ikg_schempoi", "{{params.ikg_schema}}")


def prompt_private_token() -> str:
    token = getpass.getpass("Enter your private token: ").strip()
    if not token:
        raise ValueError("GitLab private token is required.")
    return token


def prompt_db_password() -> str:
    password = getpass.getpass("Enter Password for DB User: ").strip()
    if not password:
        raise ValueError("Database password is required.")
    return password


def prompt_insight_types() -> List[str]:
    raw = input("Enter insight_type(s) separated by comma: ").strip()
    if not raw:
        raise ValueError("At least one insight_type is required.")
    insight_types = [item.strip() for item in raw.split(",") if item.strip()]
    if not insight_types:
        raise ValueError("Failed to parse any insight_type values.")
    return insight_types


def connect_gitlab(token: str) -> gitlab.Gitlab:
    gl = gitlab.Gitlab(GITLAB_URL, private_token=token)
    gl.auth()
    return gl


def load_project(gl_client: gitlab.Gitlab):
    return gl_client.projects.get(ODM_PROJECT_PATH)


def build_sql_index(
    project, ref: str = BASE_BRANCH, path: str = SQL_PATH
) -> Dict[str, List[str]]:
    """
    Build a lookup of insight_type -> list of matching SQL paths.
    """
    sql_index: Dict[str, List[str]] = {}
    tree = project.repository_tree(
        path=path, ref=ref, recursive=True, all=True
    )
    for node in tree:
        if node["type"] != "blob" or not node["path"].lower().endswith(".sql"):
            continue
        filename = PurePosixPath(node["path"]).stem
        sql_index.setdefault(filename, []).append(node["path"])
    return sql_index


def connect_db(password: str):
    conn = psycopg2.connect(
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"],
        dbname=DB_CONFIG["dbname"],
        user=DB_CONFIG["user"],
        password=password,
    )
    schema = DB_CONFIG.get("schema")
    if schema:
        with conn.cursor() as cur:
            cur.execute(f"SET search_path TO {schema}")
    return conn


def fetch_rule_metadata(conn, insight_types: Iterable[str]) -> pd.DataFrame:
    placeholders = ",".join(["%s"] * len(insight_types))
    query = (
        f"SELECT * FROM {RULE_METADATA_TABLE} "
        f"WHERE insight_type IN ({placeholders}) "
        f"ORDER BY insight_type, rule_column"
    )
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(query, tuple(insight_types))
        rows = cur.fetchall()
    return pd.DataFrame(rows)


def export_metadata(df: pd.DataFrame) -> Optional[Path]:
    if df.empty:
        print("No metadata to export.")
        return None
    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_file = Path(f"odm_rules_details_{timestamp}.xlsx")
    df.to_excel(output_file, index=False)
    print(f"Metadata exported to {output_file.resolve()}")
    return output_file


def display_metadata(df: pd.DataFrame) -> None:
    if df.empty:
        print("No rows returned for the provided insight_type list.")
        return
    for insight_type, group in df.groupby("insight_type"):
        print(f"\nInsight Type: {insight_type}")
        cols_to_show = [
            col for col in group.columns if col in {"rule_column", "logic"}
        ]
        if not cols_to_show:
            cols_to_show = list(group.columns)
        print(group[cols_to_show].to_string(index=False))


def prompt_rule_change(df: pd.DataFrame) -> Tuple[str, str, str]:
    if df.empty:
        raise ValueError(
            "Cannot proceed with rule change because no metadata rows were returned."
        )
    if "insight_type" not in df.columns or "rule_column" not in df.columns:
        raise ValueError(
            "Result set must contain 'insight_type' and 'rule_column' columns."
        )
    change_target = input(
        'Enter "rule_column to be changed" in format insight_type.rule_column: '
    ).strip()
    if "." not in change_target:
        raise ValueError("Input must be in format insight_type.rule_column")
    insight_type, rule_column = [part.strip() for part in change_target.split(".", 1)]
    if insight_type not in set(df["insight_type"]):
        raise ValueError(
            f'Insight type "{insight_type}" not found in metadata results.'
        )
    if rule_column not in set(
        df.loc[df["insight_type"] == insight_type, "rule_column"]
    ):
        raise ValueError(
            f'Rule column "{rule_column}" not associated with insight type "{insight_type}".'
        )
    new_value = input('Enter "value to be changed": ').strip()
    if not new_value:
        raise ValueError("A replacement value is required.")
    return (insight_type, rule_column, new_value)


def ensure_feature_branch(project, base_branch: str = BASE_BRANCH) -> str:
    counter = 1
    while True:
        branch_name = f"{BRANCH_PREFIX}{counter}"
        try:
            project.branches.get(branch_name)
            counter += 1
            continue
        except gl_exceptions.GitlabGetError as err:
            if err.response_code == 404:
                project.branches.create(
                    {"branch": branch_name, "ref": base_branch}
                )
                print(f"Created branch {branch_name} from {base_branch}.")
                return branch_name
            raise


def get_file_content(project, file_path: str, ref: str) -> str:
    file_obj = project.files.get(file_path=file_path, ref=ref)
    decoded_bytes = base64.b64decode(file_obj.content)
    return decoded_bytes.decode("utf-8")


def determine_modified_filename(
    project, branch: str, original_path: str, insight_type: str
) -> str:
    directory = str(PurePosixPath(original_path).parent)
    index = 1
    while True:
        candidate_name = f"{insight_type}_modified_{index}.sql"
        candidate_path = (
            candidate_name if directory == "." else f"{directory}/{candidate_name}"
        )
        try:
            project.files.get(file_path=candidate_path, ref=branch)
            index += 1
            continue
        except gl_exceptions.GitlabGetError as err:
            if err.response_code == 404:
                return candidate_path
            raise


def format_sql_literal(value: str) -> str:
    if re.fullmatch(r"-?\d+(\.\d+)?", value):
        return value
    if value.upper() in {"TRUE", "FALSE", "NULL"}:
        return value.upper()
    if (value.startswith("'") and value.endswith("'")) or (
        value.startswith('"') and value.endswith('"')
    ):
        return value
    return f"'{value}'"


def _find_statement_end(sql_text: str, start_idx: int) -> int:
    in_single = False
    in_double = False
    i = start_idx
    while i < len(sql_text):
        ch = sql_text[i]
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == ";" and not in_single and not in_double:
            return i + 1
        i += 1
    return len(sql_text)


def locate_insert_block(sql_text: str) -> Tuple[str, int, int]:
    lowered = sql_text.lower()
    search_start = 0
    while True:
        idx = lowered.find("insert into", search_start)
        if idx == -1:
            raise ValueError(
                "Could not find INSERT INTO statement targeting ODM table."
            )
        end_idx = _find_statement_end(sql_text, idx)
        statement = sql_text[idx:end_idx]
        normalized = re.sub(r"\s+", "", statement.lower())
        if any(key in normalized for key in TARGET_INSERT_KEYS) and "{{params.odm_table}}" in normalized:
            return statement, idx, end_idx
        search_start = end_idx


def apply_rule_change(sql_text: str, rule_column: str, new_value: str) -> str:
    insert_block, start_idx, end_idx = locate_insert_block(sql_text)

    column_pattern = re.compile(
        rf"({re.escape(rule_column)}\s*(?:=|<>|>=|<=|>|<)\s*)([^\s)]+)",
        re.IGNORECASE,
    )
    formatted_value = format_sql_literal(new_value)
    updated_block, replacements = column_pattern.subn(
        rf"\1{formatted_value}", insert_block, count=1
    )
    if replacements == 0:
        raise ValueError(
            f"Rule column {rule_column} not found inside the INSERT statement."
        )
    return sql_text[:start_idx] + updated_block + sql_text[end_idx:]


def write_gitlab_file(
    project,
    branch: str,
    file_path: str,
    content: str,
    commit_message: str,
) -> None:
    project.files.create(
        {
            "file_path": file_path,
            "branch": branch,
            "content": content,
            "commit_message": commit_message,
        }
    )
    print(f"Wrote {file_path} to branch {branch}.")


def export_local_sql(file_name: str, content: str) -> Path:
    LOCAL_SQL_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    export_path = LOCAL_SQL_EXPORT_DIR / Path(file_name).name
    export_path.write_text(content, encoding="utf-8")
    print(f"Local copy saved to {export_path.resolve()}")
    return export_path


def main():
    print("GitLab Setup ---")
    try:
        token = prompt_private_token()
        gl_client = connect_gitlab(token)
        project = load_project(gl_client)

        sql_index = build_sql_index(project, ref=BASE_BRANCH, path=SQL_PATH)
        print(f"Discovered {sum(len(v) for v in sql_index.values())} SQL files.")

        insight_types = prompt_insight_types()
        password = prompt_db_password()

        with connect_db(password) as conn:
            metadata_df = fetch_rule_metadata(conn, insight_types)

        display_metadata(metadata_df)
        export_metadata(metadata_df)

        insight_type, rule_column, new_value = prompt_rule_change(metadata_df)

        matching_paths = sql_index.get(insight_type)
        if not matching_paths:
            raise ValueError(
                f"No SQL file found for insight_type {insight_type} under {SQL_PATH}."
            )
        if len(matching_paths) > 1:
            print(
                f"Multiple SQL files found for {insight_type}. "
                f"Choosing the first match: {matching_paths[0]}"
            )
        original_path = matching_paths[0]

        feature_branch = ensure_feature_branch(project, base_branch=BASE_BRANCH)
        sql_text = get_file_content(project, original_path, ref=feature_branch)
        updated_sql = apply_rule_change(sql_text, rule_column, new_value)

        new_file_path = determine_modified_filename(
            project, branch=feature_branch, original_path=original_path, insight_type=insight_type
        )
        commit_message = (
            f"[Automated] Adjust {rule_column} for {insight_type} to {new_value}"
        )
        write_gitlab_file(
            project,
            branch=feature_branch,
            file_path=new_file_path,
            content=updated_sql,
            commit_message=commit_message,
        )
        export_local_sql(new_file_path, updated_sql)

        print(
            "\nDone. Review the new branch in GitLab, validate the SQL, and create a merge request when ready."
        )
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
