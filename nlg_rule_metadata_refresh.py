#!/usr/bin/env python3
"""
Generate the NLG rule metadata report and refresh sandbox_prj_smart_insights.nlg_rule_metadata_auto_refresh.

Steps
-----
1. Authenticate to GitLab with a personal access token.
2. Collect ODM profile metadata from `dags/nlg/src/config/odm_profile_map/profile_tbl.yaml`.
3. Walk every YAML/YML/JSON rule file under `dags/nlg/src/rules`, extracting rule tags that use the `{value}`
   placeholder syntax.
4. Full-outer join both datasets on `target_type`, enforce manual overrides, and persist the results to
   an XLSX report.
5. Connect to Greenplum (psycopg2) to replace `sandbox_prj_smart_insights.nlg_rule_metadata_auto_refresh`
   with the freshly generated records, updating ownership and grants.
"""

from __future__ import annotations

import base64
import getpass
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gitlab
import pandas as pd
import psycopg2
import yaml
from psycopg2.extras import execute_values

# --------------------------------------------------------------------------------------
# GitLab repo configuration
# --------------------------------------------------------------------------------------
GITLAB_URL = "https://devcloud.ubs.net"
NLG_PROJECT_PATH = (
    "ubs/gwma/smart-technology-and-analytics/staat-data-science/"
    "staat-ds-genesis/genesis-platform/nlg-dags"
)
BRANCH = "nlg-master"
PATH_PROFILE_MAP = "dags/nlg/src/config/odm_profile_map"
PROFILE_FILE_NAME = "profile_tbl.yaml"
PATH_RULES = "dags/nlg/src/rules"

# --------------------------------------------------------------------------------------
# Output configuration
# --------------------------------------------------------------------------------------
OUTPUT_TEMPLATE = "nlg_rules_metadata_{timestamp}.xlsx"
TIMESTAMP_FMT = "%Y-%m-%d_%H%M%S"

# --------------------------------------------------------------------------------------
# Database configuration
# --------------------------------------------------------------------------------------
DB_CONFIG = {
    "host": "greenplum-rdsp.zur.swissbank.com",
    "port": 5432,
    "dbname": "gprdsp",
    "user": "ds_rdsp_dev",
    "schema": "sandbox_prj_smart_insights",
    "table": "nlg_rule_metadata_auto_refresh",
    "owner": "erd_gpdb_prj_smart_insights",
    "read_role": "erd_gpdb_prj_smart_insights_ro",
}

# --------------------------------------------------------------------------------------
# Constants/utilities
# --------------------------------------------------------------------------------------
RULE_FILE_EXTS = {".yaml", ".yml", ".json"}
RULE_TAG_PATTERN = re.compile(r"\{([^{}]+)\}")
OVERRIDES = {
    "fl_recruitment": ("fl_rec_profile_curr_ikg", "fl_rec_profile_key"),
    "fl_ubs_fa": ("fl_fa_profile_curr_ikg", "f1_fa_profile_key"),
}


def prompt_token(prompt: str) -> str:
    token = getpass.getpass(prompt)
    if not token:
        raise ValueError("A non-empty value is required for authentication.")
    return token


def gitlab_project(token: str) -> gitlab.v4.objects.Project:
    gl = gitlab.Gitlab(GITLAB_URL, private_token=token)
    gl.auth()
    return gl.projects.get(NLG_PROJECT_PATH)


def fetch_file_text(project: gitlab.v4.objects.Project, file_path: str) -> str:
    file_obj = project.files.get(file_path=file_path, ref=BRANCH)
    return base64.b64decode(file_obj.content).decode("utf-8")


def normalize_target_type(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def parse_profile_entries(data: Any) -> List[Dict[str, Optional[str]]]:
    """
    Attempt to normalize profile metadata into a list of {target_type, profile_table, joining_key}.
    Handles dict-of-dict and list-of-dict layouts.
    """
    rows: List[Dict[str, Optional[str]]] = []

    def extract(entry: Dict[str, Any], fallback_target: Optional[str] = None) -> None:
        lowered = {str(k).lower(): v for k, v in entry.items()}
        target = normalize_target_type(
            lowered.get("target_type") or lowered.get("targettype") or fallback_target
        )
        if target is None:
            return
        profile_table = (
            lowered.get("profile_table")
            or lowered.get("profiletbl")
            or lowered.get("profiletblname")
        )
        joining_key = lowered.get("joining_key") or lowered.get("joiningkey") or lowered.get(
            "join_key"
        )
        rows.append(
            {
                "target_type": target,
                "profile_table": profile_table.strip() if isinstance(profile_table, str) else profile_table,
                "joining_key": joining_key.strip() if isinstance(joining_key, str) else joining_key,
            }
        )

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                extract(value, fallback_target=key)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        extract(item, fallback_target=key)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                extract(item)
    else:
        raise ValueError("Unsupported profile_tbl.yaml structure.")

    return rows


def build_profile_dataframe(project: gitlab.v4.objects.Project) -> pd.DataFrame:
    profile_path = f"{PATH_PROFILE_MAP}/{PROFILE_FILE_NAME}"
    content = fetch_file_text(project, profile_path)
    yaml_payload = yaml.safe_load(content)
    rows = parse_profile_entries(yaml_payload)
    if not rows:
        raise ValueError(f"No profile metadata extracted from {profile_path}.")
    df = pd.DataFrame(rows)
    df["target_type"] = df["target_type"].str.strip().str.lower()
    return df


def gather_rule_files(project: gitlab.v4.objects.Project) -> List[Dict[str, Any]]:
    tree = project.repository_tree(path=PATH_RULES, ref=BRANCH, recursive=True, get_all=True)
    return [
        node
        for node in tree
        if node.get("type") == "blob" and Path(node["path"]).suffix.lower() in RULE_FILE_EXTS
    ]


def safe_yaml_load(text: str) -> Any:
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse YAML: {exc}") from exc


def safe_json_load(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON: {exc}") from exc


def parse_rule_document(text: str, suffix: str) -> Any:
    if suffix in (".yaml", ".yml"):
        return safe_yaml_load(text)
    if suffix == ".json":
        return safe_json_load(text)
    raise ValueError(f"Unsupported file extension: {suffix}")


def extract_rule_tags(node: Any) -> List[Tuple[str, str]]:
    matches: List[Tuple[str, str]] = []

    def handle_pair(key: str, value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    for match in RULE_TAG_PATTERN.findall(item):
                        matches.append((key, match))
        elif isinstance(value, str):
            for match in RULE_TAG_PATTERN.findall(value):
                matches.append((key, match))

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(key, str):
                    handle_pair(key, value)
                if isinstance(value, (dict, list)):
                    walk(value)
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, (dict, list)):
                    walk(item)

    walk(node)
    return matches


def build_rules_dataframe(project: gitlab.v4.objects.Project, run_ts: pd.Timestamp) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    files = gather_rule_files(project)
    for node in files:
        path = node["path"]
        relative = path[len(PATH_RULES) + 1 :]
        if "/" not in relative:
            continue
        parts = relative.split("/", 1)
        target_type = normalize_target_type(parts[0]) if len(parts) > 0 else None
        if not target_type:
            continue
        file_name = Path(path).name
        insight_type = Path(path).stem
        content = fetch_file_text(project, path)
        document = parse_rule_document(content, Path(path).suffix.lower())
        for rule_tag, rule_value in extract_rule_tags(document):
            records.append(
                {
                    "target_type": target_type.lower(),
                    "insight_type": insight_type,
                    "rule_tag": rule_tag,
                    "rule_tag_value": rule_value,
                    "filepath": path,
                    "filename": file_name,
                    "current_date_time": run_ts,
                }
            )

    if records:
        return pd.DataFrame(records)

    # Ensure downstream merge still works by returning an empty frame with expected columns.
    return pd.DataFrame(
        columns=[
            "target_type",
            "insight_type",
            "rule_tag",
            "rule_tag_value",
            "filepath",
            "filename",
            "current_date_time",
        ]
    )


def merge_datasets(profile_df: pd.DataFrame, rules_df: pd.DataFrame) -> pd.DataFrame:
    merged = profile_df.merge(rules_df, on="target_type", how="outer")
    return merged


def apply_overrides(df: pd.DataFrame) -> None:
    if df.empty:
        return
    for target, (profile_tbl, join_key) in OVERRIDES.items():
        mask = df["target_type"].str.lower() == target
        df.loc[mask, "profile_table"] = profile_tbl
        df.loc[mask, "joining_key"] = join_key


def materialize_xlsx(df: pd.DataFrame, timestamp: str) -> Path:
    output_file = OUTPUT_TEMPLATE.format(timestamp=timestamp)
    output_path = Path(output_file).resolve()
    df.to_excel(output_path, index=False)
    return output_path


def refresh_greenplum_table(df: pd.DataFrame, db_password: str) -> None:
    df = df.copy()
    df["current_date_time"] = pd.to_datetime(df["current_date_time"]).dt.to_pydatetime()
    schema = DB_CONFIG["schema"]
    table = DB_CONFIG["table"]
    columns = [
        "target_type",
        "profile_table",
        "joining_key",
        "insight_type",
        "rule_tag",
        "rule_tag_value",
        "filepath",
        "filename",
        "current_date_time",
    ]
    payload = list(df[columns].where(pd.notnull(df), None).itertuples(index=False, name=None))
    connect_kwargs = {k: v for k, v in DB_CONFIG.items() if k in {"host", "port", "dbname", "user"}}
    connect_kwargs["password"] = db_password

    ddl = f"DROP TABLE IF EXISTS {schema}.{table};"
    create = f"""
        CREATE TABLE {schema}.{table} (
            target_type TEXT,
            profile_table TEXT,
            joining_key TEXT,
            insight_type TEXT,
            rule_tag TEXT,
            rule_tag_value TEXT,
            filepath TEXT,
            filename TEXT,
            current_date_time TIMESTAMP
        );
    """
    owner = f"ALTER TABLE {schema}.{table} OWNER TO {DB_CONFIG['owner']};"
    grant = f"GRANT SELECT ON {schema}.{table} TO {DB_CONFIG['read_role']};"

    with psycopg2.connect(**connect_kwargs) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(ddl)
            cur.execute(create)
            if payload:
                insert_sql = f"INSERT INTO {schema}.{table} ({', '.join(columns)}) VALUES %s"
                execute_values(cur, insert_sql, payload)
            cur.execute(owner)
            cur.execute(grant)


def main() -> None:
    try:
        token = prompt_token("Enter your private token: ")
        project = gitlab_project(token)

        run_ts = pd.Timestamp.utcnow()
        timestamp_str = run_ts.strftime(TIMESTAMP_FMT)

        profile_df = build_profile_dataframe(project)
        rules_df = build_rules_dataframe(project, run_ts)

        merged_df = merge_datasets(profile_df, rules_df)
        if "current_date_time" not in merged_df:
            merged_df["current_date_time"] = run_ts
        merged_df["current_date_time"] = merged_df["current_date_time"].fillna(run_ts)

        apply_overrides(merged_df)
        output_path = materialize_xlsx(merged_df, timestamp_str)

        print(f"Wrote report: {output_path}")

        db_password = prompt_token("Enter Password for DB User: ")
        refresh_greenplum_table(merged_df, db_password)

        print(
            f"Table {DB_CONFIG['schema']}.{DB_CONFIG['table']} refreshed successfully "
            f"({len(merged_df)} rows)."
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
