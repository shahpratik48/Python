#!/usr/bin/env python3
"""
Generate the NLG profile columns report and publish it to Greenplum.
Required packages: python-gitlab, pandas, psycopg2-binary, PyYAML
"""

import base64
import datetime
import getpass
import io
import json
import re
from pathlib import PurePosixPath

import gitlab
import pandas as pd
import psycopg2
from psycopg2 import sql
import yaml

GITLAB_URL = "https://devcloud.ubs.net"
NLG_PROJECT_PATH = (
    "ubs/gwma/smart-technology-and-analytics/"
    "staat-data-science/staat-ds-genesis/genesis-platform/nlg-dags"
)
BRANCH = "nlg-master"

PATH_PROFILE_MAP = "dags/nlg/src/config/odm_profile_map/profile_tbl.yml"
PATH_INPUT_CONFIG = "dags/nlg/src/config/input_data_config"
PATH_SQL_MAPPING = "dags/nlg/src/sql_column_mapping"

TARGET_SCHEMA = "sandbox_prj_smart_insights"
TARGET_TABLE = "nlg_profile_columns_auto_task"

SQL_OWNER = "erd_gpdb_prj_smart_insights"
SQL_READ_ROLE = "erd_gpdb_prj_smart_insights_ro"

COLUMN_PATTERN = re.compile(r'\bb\.(?:"([^"]+)"|([A-Za-z0-9_]+))', re.IGNORECASE)
PROFILE_TABLE_KEYS = [
    "profile_table",
    "profile_tbl",
    "profile_table_name",
    "profile_tbl_name",
]
JOIN_KEY_KEYS = ["joining_key", "join_key", "joining_keys", "join_keys"]


def fetch_file_content(project, file_path, ref):
    gl_file = project.files.get(file_path=file_path, ref=ref)
    return base64.b64decode(gl_file.content)


def load_structured_file(project, file_path, ref):
    raw_bytes = fetch_file_content(project, file_path, ref)
    text = raw_bytes.decode("utf-8")
    if file_path.lower().endswith(".json"):
        return json.loads(text)
    return yaml.safe_load(text)


def walk_repository(project, path, ref):
    for item in project.repository_tree(path=path, ref=ref, all=True):
        full_path = f"{path}/{item['name']}" if path else item["name"]
        if item["type"] == "tree":
            yield from walk_repository(project, full_path, ref)
        elif item["type"] == "blob":
            yield full_path


def first_non_empty(container, keys):
    for key in keys:
        value = container.get(key) if isinstance(container, dict) else None
        if value not in (None, ""):
            return value
    return None


def serialise_value(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def deduplicate(rows, key_fields):
    seen = set()
    deduped = []
    for row in rows:
        key = tuple(row.get(field) for field in key_fields)
        if key not in seen:
            seen.add(key)
            deduped.append(row)
    return deduped


def search_profile_map(node, fallback_target=None):
    rows = []
    if isinstance(node, dict):
        current_target = node.get("target_type") or fallback_target
        profile_table = first_non_empty(node, PROFILE_TABLE_KEYS)
        joining_key = first_non_empty(node, JOIN_KEY_KEYS)
        if profile_table or joining_key:
            row = {
                "target_type": serialise_value(current_target)
                or serialise_value(fallback_target),
                "profile_table": serialise_value(profile_table),
                "joining_key": serialise_value(joining_key),
            }
            if row["target_type"]:
                rows.append(row)
        for key, value in node.items():
            new_fallback = node.get("target_type") or key or fallback_target
            rows.extend(search_profile_map(value, fallback_target=new_fallback))
    elif isinstance(node, list):
        for item in node:
            rows.extend(search_profile_map(item, fallback_target=fallback_target))
    return rows


def collect_profile_map(project, ref, file_path):
    content = load_structured_file(project, file_path, ref)
    if content is None:
        return []
    rows = search_profile_map(content)
    return deduplicate(rows, ["target_type", "profile_table", "joining_key"])


def find_b_columns(expression):
    matches = COLUMN_PATTERN.findall(expression)
    columns = []
    seen = set()
    for quoted, unquoted in matches:
        column = quoted or unquoted
        if column:
            key = column.lower()
            if key not in seen:
                seen.add(key)
                columns.append(column)
    return columns


def extract_logic_entries(node, prefix=None):
    entries = []
    if isinstance(node, dict):
        for key, value in node.items():
            new_prefix = f"{prefix}.{key}" if prefix else str(key)
            entries.extend(extract_logic_entries(value, new_prefix))
    elif isinstance(node, list):
        for idx, item in enumerate(node):
            new_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            entries.extend(extract_logic_entries(item, new_prefix))
    else:
        if isinstance(node, str):
            logic = node.strip()
            if not logic:
                return entries
            columns = find_b_columns(logic)
            if columns:
                entries.append((prefix, logic, columns))
    return entries


def derive_tag_name(tag_path):
    if not tag_path:
        return None
    last_part = tag_path.split(".")[-1]
    last_part = re.sub(r"\[\d+\]", "", last_part)
    return last_part


def should_skip_tag(tag_path):
    if not tag_path:
        return True
    parts = [re.sub(r"\[\d+\]", "", segment) for segment in tag_path.split(".")]
    return any(part.lower() in {"target_type"} for part in parts if part)


def find_explicit_target(node):
    if isinstance(node, dict):
        value = node.get("target_type")
        if isinstance(value, str) and value:
            return value
        for child in node.values():
            result = find_explicit_target(child)
            if result:
                return result
    elif isinstance(node, list):
        for item in node:
            result = find_explicit_target(item)
            if result:
                return result
    return None


def collect_input_config_rows(project, ref, base_path):
    rows = []
    for file_path in walk_repository(project, base_path, ref):
        if not file_path.lower().endswith((".yml", ".yaml")):
            continue
        data = load_structured_file(project, file_path, ref)
        if data is None:
            continue
        target_type = find_explicit_target(data) or PurePosixPath(file_path).stem
        for tag_path, logic, columns in extract_logic_entries(data):
            if should_skip_tag(tag_path):
                continue
            tag = derive_tag_name(tag_path)
            for column in columns:
                rows.append(
                    {
                        "target_type": serialise_value(target_type),
                        "config_tag": tag,
                        "config_tag_path": tag_path,
                        "config_profile_column": column,
                        "config_logic": logic,
                        "config_filename": PurePosixPath(file_path).name,
                        "config_file_path": file_path,
                    }
                )
    return rows


def collect_sql_mapping_rows(project, ref, base_path):
    rows = []
    base_parts = PurePosixPath(base_path).parts
    for file_path in walk_repository(project, base_path, ref):
        if not file_path.lower().endswith((".yml", ".yaml", ".json")):
            continue
        path_obj = PurePosixPath(file_path)
        if len(path_obj.parts) <= len(base_parts):
            continue
        target_type = path_obj.parts[len(base_parts)]
        data = load_structured_file(project, file_path, ref)
        if data is None:
            continue
        explicit_target = find_explicit_target(data)
        if explicit_target:
            target_type = explicit_target
        insight_type = path_obj.stem
        for tag_path, logic, columns in extract_logic_entries(data):
            if should_skip_tag(tag_path):
                continue
            tag = derive_tag_name(tag_path)
            for column in columns:
                rows.append(
                    {
                        "target_type": serialise_value(target_type),
                        "insight_type": insight_type,
                        "sql_tag": tag,
                        "sql_tag_path": tag_path,
                        "sql_profile_column": column,
                        "sql_logic": logic,
                        "filename_under_sql_column_mapping": path_obj.name,
                        "file_path_under_sql_column_mapping": file_path,
                    }
                )
    return rows


def ensure_dataframe(rows, columns):
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=columns)
    return df


def build_report_dataframe(project, ref):
    profile_map_rows = collect_profile_map(project, ref, PATH_PROFILE_MAP)
    input_config_rows = collect_input_config_rows(project, ref, PATH_INPUT_CONFIG)
    sql_mapping_rows = collect_sql_mapping_rows(project, ref, PATH_SQL_MAPPING)

    df_profile = ensure_dataframe(
        profile_map_rows, ["target_type", "profile_table", "joining_key"]
    )
    df_config = ensure_dataframe(
        input_config_rows,
        [
            "target_type",
            "config_tag",
            "config_tag_path",
            "config_profile_column",
            "config_logic",
            "config_filename",
            "config_file_path",
        ],
    )
    df_sql = ensure_dataframe(
        sql_mapping_rows,
        [
            "target_type",
            "insight_type",
            "sql_tag",
            "sql_tag_path",
            "sql_profile_column",
            "sql_logic",
            "filename_under_sql_column_mapping",
            "file_path_under_sql_column_mapping",
        ],
    )

    merged = df_profile.merge(df_config, on="target_type", how="outer")
    merged = merged.merge(df_sql, on="target_type", how="outer")

    report_timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    merged["current_timestamp"] = report_timestamp

    preferred_order = [
        "target_type",
        "profile_table",
        "joining_key",
        "config_tag",
        "config_tag_path",
        "config_profile_column",
        "config_logic",
        "config_filename",
        "config_file_path",
        "insight_type",
        "sql_tag",
        "sql_tag_path",
        "sql_profile_column",
        "sql_logic",
        "filename_under_sql_column_mapping",
        "file_path_under_sql_column_mapping",
        "current_timestamp",
    ]
    ordered_cols = [col for col in preferred_order if col in merged.columns]
    remaining_cols = [col for col in merged.columns if col not in ordered_cols]
    merged = merged[ordered_cols + remaining_cols]

    return merged


def load_dataframe_to_greenplum(df, db_config, schema, table_name):
    df_for_db = df.fillna("").astype(str)

    conn_params = {
        "host": db_config["host"],
        "port": db_config["port"],
        "dbname": db_config["dbname"],
        "user": db_config["user"],
        "password": db_config["password"],
    }

    with psycopg2.connect(**conn_params) as conn:
        conn.autocommit = True

        with conn.cursor() as cur:
            drop_sql = sql.SQL("DROP TABLE IF EXISTS {}.{}").format(
                sql.Identifier(schema), sql.Identifier(table_name)
            )
            cur.execute(drop_sql)

            columns_ddl = sql.SQL(", ").join(
                sql.SQL("{} TEXT").format(sql.Identifier(column))
                for column in df_for_db.columns
            )
            create_sql = sql.SQL(
                "CREATE TABLE {}.{} ({}) DISTRIBUTED RANDOMLY"
            ).format(sql.Identifier(schema), sql.Identifier(table_name), columns_ddl)
            cur.execute(create_sql)

        buffer = io.StringIO()
        df_for_db.to_csv(buffer, index=False)
        buffer.seek(0)

        with conn.cursor() as cur:
            copy_sql = sql.SQL("COPY {}.{} ({}) FROM STDIN WITH CSV HEADER").format(
                sql.Identifier(schema),
                sql.Identifier(table_name),
                sql.SQL(", ").join(sql.Identifier(col) for col in df_for_db.columns),
            )
            cur.copy_expert(copy_sql.as_string(cur), buffer)

        with conn.cursor() as cur:
            alter_sql = sql.SQL("ALTER TABLE {}.{} OWNER TO {}").format(
                sql.Identifier(schema),
                sql.Identifier(table_name),
                sql.Identifier(SQL_OWNER),
            )
            cur.execute(alter_sql)

            grant_sql = sql.SQL("GRANT SELECT ON {}.{} TO {}").format(
                sql.Identifier(schema),
                sql.Identifier(table_name),
                sql.Identifier(SQL_READ_ROLE),
            )
            cur.execute(grant_sql)


def main():
    private_token = getpass.getpass("Enter your private token: ").strip()
    if not private_token:
        raise ValueError("GitLab private token is required.")

    gl = gitlab.Gitlab(GITLAB_URL, private_token=private_token)
    project = gl.projects.get(NLG_PROJECT_PATH)

    report_df = build_report_dataframe(project, BRANCH)

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    output_file = f"nlg_profile_columns_{timestamp}.xlsx"
    report_df.to_excel(output_file, index=False)
    print(f"Report saved to {output_file}")

    db_password = getpass.getpass("Enter Password for DB User: ").strip()
    if not db_password:
        raise ValueError("Database password is required.")

    db_config = {
        "host": "greenplum-rdsp.zur.swissbank.com",
        "port": "5432",
        "dbname": "gprdsp",
        "user": "ds_rdsp_dev",
        "password": db_password,
    }

    load_dataframe_to_greenplum(
        report_df, db_config, TARGET_SCHEMA, TARGET_TABLE
    )
    print(
        f"Table {TARGET_SCHEMA}.{TARGET_TABLE} refreshed and privileges applied."
    )


if __name__ == "__main__":
    main()
