#!/usr/bin/env python3
"""
Automation utility for managing ODM SQL rules in GitLab.

Features
--------
* Authenticates with GitLab using a private token captured via getpass.
* Scans every SQL file under `dags/odm/script/sql` on `odm-master`.
* Pulls rule metadata for requested insight types from Postgres.
* Displays logic/rule_column pairs in a friendly table and exports them to Excel.
* Prompts for which insight_type should change, captures a replacement WHERE
  clause fragment, and updates the corresponding SQL file in a freshly-created
  branch.
* Saves the modified SQL both to GitLab (overwriting the existing file in the
  new branch) and to the local workspace for immediate inspection.

The script is interactive and should be executed from a secure, network-enabled
environment that can reach both GitLab and the target Postgres cluster.
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
SCHEMA_TEMPLATE_PATTERN = re.compile(
    r"\{\{\s*params\.ikg_schema.*?\}\}", re.IGNORECASE | re.DOTALL
)
TABLE_TEMPLATE_PATTERN = re.compile(
    r"\{\{\s*params\.odm_table.*?\}\}", re.IGNORECASE | re.DOTALL
)
CLAUSE_BOUNDARY_PATTERN = re.compile(
    r"\b(group\s+by|order\s+by|limit|offset|union|intersect|except|having)\b",
    re.IGNORECASE,
)


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


def prompt_target_insight_type(df: pd.DataFrame) -> str:
    if df.empty or "insight_type" not in df.columns:
        raise ValueError("Cannot select an insight_type because no metadata was returned.")
    available = sorted(df["insight_type"].unique())
    print("\nAvailable insight types:")
    for item in available:
        print(f" - {item}")
    selection = input("Which insight_type should be changed? ").strip()
    if not selection:
        raise ValueError("You must specify an insight_type to change.")
    if selection not in available:
        raise ValueError(f'Insight type "{selection}" is not present in the metadata results.')
    return selection


def prompt_where_clause_replacement() -> str:
    new_clause = input('What to change? Provide the replacement for the WHERE clause: ').strip()
    if not new_clause:
        raise ValueError("A replacement WHERE clause is required.")
    lowered = new_clause.lower()
    if lowered.startswith("where "):
        new_clause = new_clause.split(" ", 1)[1].strip()
    return new_clause


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


def _is_target_insert(statement: str) -> bool:
    lowered = statement.lower()
    if "ilparams.ikg_schempoi" in lowered and TABLE_TEMPLATE_PATTERN.search(statement):
        return True
    return bool(
        SCHEMA_TEMPLATE_PATTERN.search(statement)
        and TABLE_TEMPLATE_PATTERN.search(statement)
    )


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
        if _is_target_insert(statement):
            return statement, idx, end_idx
        search_start = end_idx


def apply_where_change(sql_text: str, new_where_clause: str) -> str:
    insert_block, start_idx, end_idx = locate_insert_block(sql_text)
    where_match = re.search(r"\bwhere\b", insert_block, re.IGNORECASE)
    if not where_match:
        raise ValueError(
            "No WHERE clause found inside the INSERT statement targeting the ODM table."
        )

    clause_start = where_match.end()
    remainder = insert_block[clause_start:]
    boundary_match = CLAUSE_BOUNDARY_PATTERN.search(remainder)
    if boundary_match:
        clause_end = clause_start + boundary_match.start()
    else:
        semicolon_idx = insert_block.find(";", clause_start)
        clause_end = semicolon_idx if semicolon_idx != -1 else len(insert_block)

    sanitized_clause = new_where_clause.strip()
    replacement = f" WHERE {sanitized_clause} "
    updated_block = (
        insert_block[: where_match.start()] + replacement + insert_block[clause_end:]
    )
    return sql_text[:start_idx] + updated_block + sql_text[end_idx:]


def update_gitlab_file(
    project,
    branch: str,
    file_path: str,
    content: str,
    commit_message: str,
) -> None:
    file_obj = project.files.get(file_path=file_path, ref=branch)
    file_obj.content = content
    file_obj.save(branch=branch, commit_message=commit_message)
    print(f"Updated {file_path} on branch {branch}.")


def export_local_sql(file_name: str, content: str) -> Path:
    LOCAL_SQL_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    relative_path = Path(file_name)
    export_path = (LOCAL_SQL_EXPORT_DIR / relative_path).resolve()
    export_path.parent.mkdir(parents=True, exist_ok=True)
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

        target_insight = prompt_target_insight_type(metadata_df)
        new_where_clause = prompt_where_clause_replacement()

        matching_paths = sql_index.get(target_insight)
        if not matching_paths:
            raise ValueError(
                f"No SQL file found for insight_type {target_insight} under {SQL_PATH}."
            )
        if len(matching_paths) > 1:
            print(
                f"Multiple SQL files found for {target_insight}. "
                f"Choosing the first match: {matching_paths[0]}"
            )
        original_path = matching_paths[0]

        feature_branch = ensure_feature_branch(project, base_branch=BASE_BRANCH)
        sql_text = get_file_content(project, original_path, ref=feature_branch)
        updated_sql = apply_where_change(sql_text, new_where_clause)

        commit_message = f"[Automated] Update WHERE clause for {target_insight}"
        update_gitlab_file(
            project,
            branch=feature_branch,
            file_path=original_path,
            content=updated_sql,
            commit_message=commit_message,
        )
        export_local_sql(original_path, updated_sql)

        print(
            "\nDone. Review the new branch in GitLab, validate the SQL, and create a merge request when ready."
        )
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
