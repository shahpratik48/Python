"""
Generate a CID profile columns report by parsing YAML configs stored in GitLab.

Steps performed:
1. Authenticate to GitLab and download all YAML files under
   `dags/nlg/src/config/input_data_config_dummy` as well as the profile map YAML
   at `dags/nlg/src/config/odm_profile_map/profile_tbl.yaml` on the specified branch.
2. Parse each YAML document looking for expressions that contain MD5 logic while
   ignoring HTTP/HTTPS URL fragments when determining profile columns. Extract the
   associated target_type, tag, logic, and the profile columns referenced inside each
   MD5 expression (normalizing away table aliases), and enrich each row with profile
   table metadata when available, leaving those fields blank when no mapping exists.
3. Persist the extracted metadata to an XLSX report file named with the current
   timestamp.
4. Load the XLSX content into the Greenplum table
   `sandbox_prj_smart_insights.cid_profile_columns_auto_task`, replacing any
   existing data.

The output columns are:
    target_type, profile_table, joining_key, tag, cid_profile_column, logic,
    current_timestamp, filepath, filename
"""

from __future__ import annotations

import base64
import datetime
import getpass
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import gitlab  # type: ignore
import pandas as pd
import psycopg2  # type: ignore
import yaml
from psycopg2 import sql  # type: ignore
from psycopg2.extras import execute_values  # type: ignore


GITLAB_URL = "https://devcloud.ubs.net"
GENESYS_GROUP_PATH = "ubs/gwma/smart-technology-and-analytics/staat-data-science/staat-ds-genesis/genesis-platform"
NLG_PROJECT_PATH = (
    "ubs/gwma/smart-technology-and-analytics/staat-data-science/staat-ds-genesis/genesis-platform/nlg-dags"
)
BRANCH = "nlg-master"
CONFIG_BASE_PATH = "dags/nlg/src/config/input_data_config_dummy"
PROFILE_MAP_BASE_PATH = "dags/nlg/src/config/odm_profile_map"
PROFILE_MAP_FILE_PATH = f"{PROFILE_MAP_BASE_PATH}/profile_tbl.yaml"

OUTPUT_TIMESTAMP_FORMAT = "%Y-%m-%d_%H%M%S"
OUTPUT_FILE_TEMPLATE = "cid_profile_columns_{timestamp}.xlsx"

TARGET_SCHEMA = "sandbox_prj_smart_insights"
TARGET_TABLE = "cid_profile_columns_auto_task"
TARGET_TABLE_FQN = f"{TARGET_SCHEMA}.{TARGET_TABLE}"

OUTPUT_COLUMNS = [
    "target_type",
    "profile_table",
    "joining_key",
    "tag",
    "cid_profile_column",
    "logic",
    "current_timestamp",
    "filepath",
    "filename",
]


SQL_KEYWORDS = {
    "ABS",
    "ADD",
    "AND",
    "ANY",
    "ARRAY",
    "AS",
    "ASC",
    "AVG",
    "BETWEEN",
    "BTRIM",
    "BY",
    "CASE",
    "CAST",
    "COALESCE",
    "COLLATE",
    "CONCAT",
    "CONCAT_WS",
    "COUNT",
    "DATE",
    "DESC",
    "DISTINCT",
    "ELSE",
    "END",
    "EXISTS",
    "FALSE",
    "FILTER",
    "FIRST_VALUE",
    "GREATEST",
    "GROUP",
    "HAVING",
    "INITCAP",
    "INNER",
    "JOIN",
    "LAG",
    "LAST_VALUE",
    "LEAD",
    "LEFT",
    "LENGTH",
    "LIKE",
    "LOWER",
    "LTRIM",
    "MAX",
    "MD5",
    "MIN",
    "NOT",
    "NULL",
    "NULLIF",
    "ON",
    "OR",
    "ORDER",
    "OUTER",
    "OVER",
    "POSITION",
    "POWER",
    "REGEXP_REPLACE",
    "REPLACE",
    "RIGHT",
    "ROW_NUMBER",
    "RTRIM",
    "SELECT",
    "SPLIT_PART",
    "SUBSTR",
    "SUBSTRING",
    "SUM",
    "THEN",
    "TO_CHAR",
    "TO_DATE",
    "TRIM",
    "TRUE",
    "UPPER",
    "USING",
    "WHEN",
    "WHERE",
    "WITH",
}


ALIAS_COLUMN_PATTERN = re.compile(
    r'(?:"?([A-Za-z_][\w$]*)"?\.)"?([A-Za-z_][\w$]*)"?', re.IGNORECASE
)
BARE_IDENTIFIER_PATTERN = re.compile(r'"?([A-Za-z_][\w$]*)"?')
STRING_LITERAL_PATTERN = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")


def is_url_token(value: str) -> bool:
    lowered = value.lower()
    return lowered in {"http", "https"} or lowered.startswith("http://") or lowered.startswith("https://")


def normalize_column_name(name: str) -> str:
    if name is None:
        return ""
    cleaned = name.strip().strip('"').strip("'")
    if "." in cleaned:
        cleaned = cleaned.split(".")[-1]
    return cleaned.strip('"').strip("'")


@dataclass
class ProfileInfo:
    target_type: str
    profile_table: str | None = None
    joining_key: str | None = None


@dataclass
class Row:
    target_type: str
    tag: str
    cid_profile_column: str
    logic: str
    current_timestamp: datetime.datetime
    filepath: str
    filename: str
    profile_table: str | None = None
    joining_key: str | None = None


def main() -> None:
    private_token = getpass.getpass("Enter your GitLab private token: ")

    gl = gitlab.Gitlab(GITLAB_URL, private_token=private_token)
    project = gl.projects.get(NLG_PROJECT_PATH)

    print(f"Connected to project {project.path_with_namespace} on branch {BRANCH!r}")

    file_paths = fetch_yaml_file_paths(project, CONFIG_BASE_PATH, BRANCH)
    print(f"Found {len(file_paths)} YAML files to process.")

    profile_tree = project.repository_tree(
        path=PROFILE_MAP_BASE_PATH,
        ref=BRANCH,
        get_all=True,
        recursive=True,
    )
    print(f"Discovered {len(profile_tree)} items under {PROFILE_MAP_BASE_PATH}.")

    profile_map = load_profile_table_mapping(project, PROFILE_MAP_FILE_PATH, BRANCH)
    print(f"Loaded profile table details for {len(profile_map)} target types.")

    now = datetime.datetime.now()
    timestamp_label = now.strftime(OUTPUT_TIMESTAMP_FORMAT)
    output_filename = OUTPUT_FILE_TEMPLATE.format(timestamp=timestamp_label)
    output_path = Path.cwd() / output_filename

    records: List[Row] = []
    for file_path in file_paths:
        yaml_text = load_file_from_gitlab(project, file_path, BRANCH)
        target_type = Path(file_path).stem
        for document in yaml.safe_load_all(yaml_text):
            if document is None:
                continue
            extracted = extract_md5_rows(
                document=document,
                target_type=target_type,
                filepath=file_path,
                current_timestamp=now,
                profile_info_map=profile_map,
            )
            records.extend(extracted)

    if not records:
        print("No MD5-based profile columns were found in any YAML file.")
        return

    md5_df = pd.DataFrame([asdict(row) for row in records])
    for column in OUTPUT_COLUMNS:
        if column not in md5_df.columns:
            md5_df[column] = None
    md5_df["target_type_lower"] = md5_df["target_type"].str.lower()

    profile_meta_df = build_profile_metadata_dataframe(profile_map)

    if not profile_meta_df.empty:
        md5_df = md5_df.merge(profile_meta_df, how="left", on="target_type_lower")
        md5_df["target_type"] = md5_df["target_type"].combine_first(md5_df.pop("target_type_profile"))
        md5_df["profile_table"] = md5_df["profile_table"].combine_first(md5_df.pop("profile_table_profile"))
        md5_df["joining_key"] = md5_df["joining_key"].combine_first(md5_df.pop("joining_key_profile"))

        md5_keys = set(md5_df["target_type_lower"].dropna())
        missing_meta = profile_meta_df[
            ~profile_meta_df["target_type_lower"].isin(md5_keys)
        ]
        if not missing_meta.empty:
            append_df = pd.DataFrame(
                {
                    "target_type": missing_meta["target_type_profile"],
                    "tag": None,
                    "cid_profile_column": None,
                    "logic": None,
                    "current_timestamp": now,
                    "filepath": PROFILE_MAP_FILE_PATH,
                    "filename": Path(PROFILE_MAP_FILE_PATH).name,
                    "profile_table": missing_meta["profile_table_profile"],
                    "joining_key": missing_meta["joining_key_profile"],
                    "target_type_lower": missing_meta["target_type_lower"],
                }
            )
            md5_df = pd.concat([md5_df, append_df], ignore_index=True)

    md5_df.drop(columns=["target_type_lower"], inplace=True)
    md5_df = md5_df.where(pd.notna(md5_df), None)
    md5_df = md5_df[OUTPUT_COLUMNS]
    md5_df.sort_values(["target_type", "tag", "cid_profile_column", "logic"], inplace=True)
    md5_df.drop_duplicates(subset=["target_type", "tag", "cid_profile_column"], keep="first", inplace=True)

    print(f"Writing {len(md5_df)} rows to {output_path}")
    md5_df.to_excel(output_path, index=False)

    password = getpass.getpass("Enter password for DB user ds_rdsp_dev: ")
    load_dataframe_to_greenplum(md5_df, password)
    print(f"Data successfully loaded into {TARGET_TABLE_FQN}.")


def fetch_yaml_file_paths(project: Any, base_path: str, ref: str) -> List[str]:
    tree = project.repository_tree(
        path=base_path,
        ref=ref,
        get_all=True,
        recursive=True,
    )
    paths = [
        item["path"]
        for item in tree
        if item.get("type") == "blob" and item["path"].lower().endswith((".yml", ".yaml"))
    ]
    return sorted(paths)


def load_file_from_gitlab(project: Any, file_path: str, ref: str) -> str:
    remote_file = project.files.get(file_path=file_path, ref=ref)
    content_bytes = base64.b64decode(remote_file.content)
    return content_bytes.decode("utf-8")


def load_profile_table_mapping(project: Any, file_path: str, ref: str) -> Dict[str, ProfileInfo]:
    mapping: Dict[str, ProfileInfo] = {}
    try:
        yaml_text = load_file_from_gitlab(project, file_path, ref)
    except Exception as exc:  # noqa: BLE001
        print(f"Unable to load profile map file {file_path}: {exc}")
        return mapping

    for document in yaml.safe_load_all(yaml_text):
        if document is None:
            continue
        for entry in collect_profile_table_entries(document):
            target_type_raw = entry.get("target_type")
            if target_type_raw is None:
                continue
            target_type = str(target_type_raw).strip()
            if not target_type:
                continue
            key = target_type.lower()
            info = mapping.setdefault(key, ProfileInfo(target_type=target_type))
            profile_table_val = entry.get("profile_table")
            joining_key_val = entry.get("joining_key")
            if profile_table_val:
                info.profile_table = str(profile_table_val).strip()
            if joining_key_val:
                info.joining_key = str(joining_key_val).strip()

    return mapping


def build_profile_metadata_dataframe(profile_map: Dict[str, ProfileInfo]) -> pd.DataFrame:
    if not profile_map:
        return pd.DataFrame(
            columns=[
                "target_type_lower",
                "target_type_profile",
                "profile_table_profile",
                "joining_key_profile",
            ]
        )

    records = []
    for key, info in profile_map.items():
        records.append(
            {
                "target_type_lower": key,
                "target_type_profile": info.target_type,
                "profile_table_profile": info.profile_table,
                "joining_key_profile": info.joining_key,
            }
        )

    return pd.DataFrame.from_records(records)


def collect_profile_table_entries(node: Any) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []

    def traverse(current: Any) -> None:
        if isinstance(current, dict):
            normalized = {normalize_key(str(k)): k for k in current.keys()}
            if "targettype" in normalized:
                target_type_key = normalized["targettype"]
                target_type_value = current.get(target_type_key)

                profile_table_value = None
                for candidate in (
                    "profiletable",
                    "profiletablename",
                    "profiletable_name",
                    "profiletbl",
                ):
                    if candidate in normalized:
                        profile_table_value = current.get(normalized[candidate])
                        if profile_table_value is not None:
                            break

                joining_key_value = None
                for candidate in (
                    "joiningkey",
                    "joiningkeys",
                    "joining_key",
                    "joinkey",
                ):
                    if candidate in normalized:
                        joining_key_value = current.get(normalized[candidate])
                        if joining_key_value is not None:
                            break

                if target_type_value is not None:
                    entries.append(
                        {
                            "target_type": target_type_value,
                            "profile_table": profile_table_value,
                            "joining_key": joining_key_value,
                        }
                    )

            for value in current.values():
                traverse(value)
        elif isinstance(current, list):
            for item in current:
                traverse(item)

    traverse(node)
    return entries


def normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.lower())


def extract_md5_rows(
    document: Any,
    target_type: str,
    filepath: str,
    current_timestamp: datetime.datetime,
    profile_info_map: Dict[str, ProfileInfo],
) -> List[Row]:
    rows: List[Row] = []
    info = profile_info_map.get(target_type.lower())
    profile_table = info.profile_table if info else None
    joining_key = info.joining_key if info else None

    def walker(node: Any, parent_keys: Sequence[str]) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walker(value, [*parent_keys, str(key)])
        elif isinstance(node, list):
            for idx, item in enumerate(node):
                walker(item, [*parent_keys, f"[{idx}]"])
        elif isinstance(node, str) and "MD5" in node.upper():
            tag = determine_tag(parent_keys)
            logic = node.strip()
            for column in extract_profile_columns(logic):
                rows.append(
                    Row(
                        target_type=target_type,
                        tag=tag,
                        cid_profile_column=column,
                        logic=logic,
                        current_timestamp=current_timestamp,
                        filepath=filepath,
                        filename=os.path.basename(filepath),
                        profile_table=profile_table,
                        joining_key=joining_key,
                    )
                )

    walker(document, [])
    return rows


def determine_tag(parent_keys: Sequence[str]) -> str:
    for key in reversed(parent_keys):
        if not key.startswith("["):
            return key
    return parent_keys[-1] if parent_keys else "unknown"


def extract_profile_columns(logic: str) -> List[str]:
    columns: List[str] = []
    seen: set[str] = set()
    for md5_argument in extract_md5_arguments(logic):
        local_found = False
        alias_cols = extract_alias_columns(md5_argument)
        for col in alias_cols:
            normalized = normalize_column_name(col)
            if not normalized or normalized in seen:
                continue
            if is_url_token(normalized):
                continue
            columns.append(normalized)
            seen.add(normalized)
            local_found = True

        stripped_argument = ALIAS_COLUMN_PATTERN.sub(lambda m: m.group(2), md5_argument)
        without_strings = STRING_LITERAL_PATTERN.sub(" ", stripped_argument)
        for ident in BARE_IDENTIFIER_PATTERN.findall(without_strings):
            if not ident:
                continue
            normalized = normalize_column_name(ident)
            if not normalized:
                continue
            upper_ident = normalized.upper()
            if upper_ident in SQL_KEYWORDS:
                continue
            if normalized.isdigit():
                continue
            if is_url_token(normalized):
                continue
            if normalized not in seen:
                columns.append(normalized)
                seen.add(normalized)
                local_found = True

        if not local_found:
            cleaned_expr = STRING_LITERAL_PATTERN.sub(" ", md5_argument)
            fallback_added = False
            for ident in BARE_IDENTIFIER_PATTERN.findall(cleaned_expr):
                if not ident:
                    continue
                normalized = normalize_column_name(ident)
                if not normalized or normalized in seen:
                    continue
                upper_ident = normalized.upper()
                if upper_ident in SQL_KEYWORDS or normalized.isdigit() or is_url_token(normalized):
                    continue
                columns.append(normalized)
                seen.add(normalized)
                fallback_added = True

            if not fallback_added:
                cleaned = normalize_column_name(" ".join(cleaned_expr.split()))
                if cleaned and cleaned not in seen and not is_url_token(cleaned):
                    columns.append(cleaned)
                    seen.add(cleaned)

    return columns


def extract_md5_arguments(logic: str) -> List[str]:
    arguments: List[str] = []
    upper_logic = logic.upper()
    search_start = 0
    marker = "MD5("

    while True:
        idx = upper_logic.find(marker, search_start)
        if idx == -1:
            break
        idx += len(marker)
        depth = 1
        end = idx
        while end < len(logic) and depth > 0:
            char = logic[end]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            end += 1
        argument = logic[idx : end - 1]
        arguments.append(argument)
        search_start = end

    return arguments


def extract_alias_columns(md5_argument: str) -> List[str]:
    columns: List[str] = []
    for _, column in ALIAS_COLUMN_PATTERN.findall(md5_argument):
        columns.append(column)
    return columns


def load_dataframe_to_greenplum(df: pd.DataFrame, password: str) -> None:
    db_config = {
        "host": "greenplum-rdsp.zur.swissbank.com",
        "port": "5432",
        "dbname": "gprdsp",
        "user": "ds_rdsp_dev",
        "password": password,
    }

    with psycopg2.connect(**db_config) as conn, conn.cursor() as cur:
        drop_statement = sql.SQL("DROP TABLE IF EXISTS {schema}.{table}").format(
            schema=sql.Identifier(TARGET_SCHEMA),
            table=sql.Identifier(TARGET_TABLE),
        )
        cur.execute(drop_statement)

        column_definitions = []
        for column in OUTPUT_COLUMNS:
            data_type = sql.SQL("TIMESTAMP") if column == "current_timestamp" else sql.SQL("TEXT")
            column_definitions.append(
                sql.SQL("{} {}").format(sql.Identifier(column), data_type)
            )
        create_statement = sql.SQL(
            "CREATE TABLE {schema}.{table} (\n                {columns}\n            )"
        ).format(
            schema=sql.Identifier(TARGET_SCHEMA),
            table=sql.Identifier(TARGET_TABLE),
            columns=sql.SQL(",\n                ").join(column_definitions),
        )
        cur.execute(create_statement)

        rows = list(df[OUTPUT_COLUMNS].itertuples(index=False, name=None))
        insert_columns = [sql.Identifier(col) for col in OUTPUT_COLUMNS]
        insert_statement = sql.SQL(
            "INSERT INTO {schema}.{table} ({columns}) VALUES %s"
        ).format(
            schema=sql.Identifier(TARGET_SCHEMA),
            table=sql.Identifier(TARGET_TABLE),
            columns=sql.SQL(", ").join(insert_columns),
        )
        insert_sql = insert_statement.as_string(cur)
        execute_values(cur, insert_sql, rows)

        alter_owner = sql.SQL("ALTER TABLE {schema}.{table} OWNER TO {owner}").format(
            schema=sql.Identifier(TARGET_SCHEMA),
            table=sql.Identifier(TARGET_TABLE),
            owner=sql.Identifier("erd_gpdb_prj_smart_insights"),
        )
        cur.execute(alter_owner)

        grant_select = sql.SQL(
            "GRANT SELECT ON {schema}.{table} TO {role}"
        ).format(
            schema=sql.Identifier(TARGET_SCHEMA),
            table=sql.Identifier(TARGET_TABLE),
            role=sql.Identifier("erd_gpdb_prj_smart_insights_ro"),
        )
        cur.execute(grant_select)


if __name__ == "__main__":
    main()
