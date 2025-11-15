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
import logging
import getpass
import json
import re
import sys
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import gitlab
import pandas as pd
import psycopg2
import yaml
from pandas.api.types import is_datetime64tz_dtype
from psycopg2.extras import execute_values

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

# --------------------------------------------------------------------------------------
# GitLab repo configuration
# --------------------------------------------------------------------------------------
GITLAB_URL = "https://devcloud.ubs.net"
NLG_PROJECT_PATH = (
    "ubs/gwma/smart-technology-and-analytics/staat-data-science/"
    "staat-ds-genesis/genesis-platform/nlg-dags"
)
BRANCH = "nlg-master"
PROFILE_FILE_PATH = "dags/nlg/src/config/odm_profile_map/profile_tbl.yaml"
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
CURRENT_TS_COLUMN = "current_timestamp"
DEDUP_COLUMNS = [
    "target_type",
    "profile_table",
    "joining_key",
    "insight_type",
    "rule_tag",
    "rule_tag_value",
    "filepath",
    "filename",
]
OVERRIDES = {
    "fl_recruitment": ("fl_rec_profile_curr_ikg", "fl_rec_profile_key"),
    "fl_ubs_fa": ("fl_fa_profile_curr_ikg", "f1_fa_profile_key"),
}

logger = logging.getLogger(__name__)


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


def _first_matching_key(entry: Dict[str, Any], keywords: List[str]) -> Optional[Any]:
    if isinstance(entry, dict):
        for key, value in entry.items():
            lower = str(key).lower()
            if all(keyword in lower for keyword in keywords):
                return value
    return None


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
            or lowered.get("profile_table_name")
            or _first_matching_key(entry, ["profile", "table"])
        )
        joining_key = (
            lowered.get("joining_key")
            or lowered.get("joiningkey")
            or lowered.get("join_key")
            or _first_matching_key(entry, ["key"])
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
    profile_path = PROFILE_FILE_PATH
    content = fetch_file_text(project, profile_path)
    yaml_payload = yaml.safe_load(content)
    rows = parse_profile_entries(yaml_payload)
    if not rows:
        raise ValueError(f"No profile metadata extracted from {profile_path}.")
    df = pd.DataFrame(rows)
    df["target_type"] = df["target_type"].str.strip().str.lower()
    logger.info("Loaded %s profile rows from %s", len(df), profile_path)
    return df


def gather_rule_files(project: gitlab.v4.objects.Project) -> List[Dict[str, Any]]:
    tree = project.repository_tree(path=PATH_RULES, ref=BRANCH, recursive=True, get_all=True)
    files = [
        node
        for node in tree
        if node.get("type") == "blob" and Path(node["path"]).suffix.lower() in RULE_FILE_EXTS
    ]
    logger.info("Discovered %s candidate rule files under %s", len(files), PATH_RULES)
    return files


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
    seen_values: Dict[Tuple[str, str, str], set] = defaultdict(set)
    logger.info("Scanning %s rule files for tag definitions", len(files))
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
            target_key = target_type.lower()
            dedupe_key = (target_key, insight_type, rule_tag)
            if rule_value in seen_values[dedupe_key]:
                continue
            seen_values[dedupe_key].add(rule_value)
            records.append(
                {
                    "target_type": target_key,
                    "insight_type": insight_type,
                    "rule_tag": rule_tag,
                    "rule_tag_value": rule_value,
                    "filepath": path,
                    "filename": file_name,
                    CURRENT_TS_COLUMN: run_ts,
                }
            )

    if records:
        df = pd.DataFrame(records)
        logger.info("Extracted %s rule tag rows", len(df))
        return df

    # Ensure downstream merge still works by returning an empty frame with expected columns.
    logger.info("No rule tag rows extracted; returning empty DataFrame")
    return pd.DataFrame(
        columns=[
            "target_type",
            "insight_type",
            "rule_tag",
            "rule_tag_value",
            "filepath",
            "filename",
            CURRENT_TS_COLUMN,
        ]
    )


def merge_datasets(profile_df: pd.DataFrame, rules_df: pd.DataFrame) -> pd.DataFrame:
    merged = profile_df.merge(rules_df, on="target_type", how="outer")
    logger.info("Merged dataset contains %s rows", len(merged))
    return merged


def apply_overrides(df: pd.DataFrame) -> None:
    if df.empty:
        return
    for target, (profile_tbl, join_key) in OVERRIDES.items():
        mask = df["target_type"].str.lower() == target
        df.loc[mask, "profile_table"] = profile_tbl
        df.loc[mask, "joining_key"] = join_key
        if mask.any():
            logger.info(
                "Applied override for target_type '%s' (%s rows)",
                target,
                int(mask.sum()),
            )


def materialize_xlsx(df: pd.DataFrame, timestamp: str) -> Path:
    output_file = OUTPUT_TEMPLATE.format(timestamp=timestamp)
    output_path = Path(output_file).resolve()
    logger.info("Writing XLSX report to %s", output_path)
    df.to_excel(output_path, index=False)
    return output_path


def refresh_greenplum_table(df: pd.DataFrame, db_password: str) -> None:
    df = df.copy()
    df[CURRENT_TS_COLUMN] = pd.to_datetime(df[CURRENT_TS_COLUMN], errors="coerce")
    if is_datetime64tz_dtype(df[CURRENT_TS_COLUMN]):
        df[CURRENT_TS_COLUMN] = df[CURRENT_TS_COLUMN].dt.tz_localize(None)
    df[CURRENT_TS_COLUMN] = df[CURRENT_TS_COLUMN].fillna(pd.Timestamp.now()).dt.to_pydatetime()
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
        CURRENT_TS_COLUMN,
    ]
    payload = list(df[columns].where(pd.notnull(df), None).itertuples(index=False, name=None))
    connect_kwargs = {k: v for k, v in DB_CONFIG.items() if k in {"host", "port", "dbname", "user"}}
    connect_kwargs["password"] = db_password

    logger.info(
        "Refreshing %s.%s with %s records",
        schema,
        table,
        len(payload),
    )
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
            {CURRENT_TS_COLUMN} TIMESTAMP
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
    logger.info("Greenplum table %s.%s refreshed successfully", schema, table)


def main() -> None:
    try:
        logger.info("Starting NLG rule metadata refresh run")
        token = prompt_token("Enter your private token: ")
        logger.info("Authenticating to GitLab project %s", NLG_PROJECT_PATH)
        project = gitlab_project(token)
        logger.info("Authenticated to GitLab project %s", NLG_PROJECT_PATH)

        run_ts = pd.Timestamp.now()
        timestamp_str = run_ts.strftime(TIMESTAMP_FMT)

        profile_df = build_profile_dataframe(project)
        rules_df = build_rules_dataframe(project, run_ts)

        merged_df = merge_datasets(profile_df, rules_df)
        if CURRENT_TS_COLUMN not in merged_df:
            merged_df[CURRENT_TS_COLUMN] = run_ts
        merged_df[CURRENT_TS_COLUMN] = pd.to_datetime(
            merged_df[CURRENT_TS_COLUMN], errors="coerce"
        ).fillna(run_ts)
        if is_datetime64tz_dtype(merged_df[CURRENT_TS_COLUMN]):
            merged_df[CURRENT_TS_COLUMN] = (
                merged_df[CURRENT_TS_COLUMN].dt.tz_localize(None)
            )

        profile_table_map = (
            profile_df.dropna(subset=["profile_table", "target_type"])
            .drop_duplicates(subset=["target_type"], keep="last")
            .set_index("target_type")["profile_table"]
            .to_dict()
        )
        joining_key_map = (
            profile_df.dropna(subset=["joining_key", "target_type"])
            .drop_duplicates(subset=["target_type"], keep="last")
            .set_index("target_type")["joining_key"]
            .to_dict()
        )
        merged_df["profile_table"] = merged_df["profile_table"].fillna(
            merged_df["target_type"].map(profile_table_map)
        )
        merged_df["joining_key"] = merged_df["joining_key"].fillna(
            merged_df["target_type"].map(joining_key_map)
        )

        apply_overrides(merged_df)

        before_dedupe = len(merged_df)
        merged_df = merged_df.drop_duplicates(subset=DEDUP_COLUMNS).reset_index(drop=True)
        if len(merged_df) != before_dedupe:
            logger.info(
                "Removed %s duplicate rows (kept %s)",
                before_dedupe - len(merged_df),
                len(merged_df),
            )
        output_path = materialize_xlsx(merged_df, timestamp_str)
        logger.info("Report available at %s", output_path)

        db_password = prompt_token("Enter Password for DB User: ")
        logger.info(
            "Refreshing Greenplum table using user '%s' on host '%s'",
            DB_CONFIG["user"],
            DB_CONFIG["host"],
        )
        refresh_greenplum_table(merged_df, db_password)

        logger.info(
            "Run finished successfully with %s merged rows",
            len(merged_df),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Run failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
