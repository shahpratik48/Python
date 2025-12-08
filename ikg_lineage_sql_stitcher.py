import base64
import datetime
import getpass
import logging
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import gitlab
import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values

try:
    import sqlparse
except ImportError:  # pragma: no cover - optional dependency
    sqlparse = None


LOG_LEVEL = os.environ.get("IKG_LINEAGE_LOG_LEVEL", "DEBUG")
GITLAB_URL = "https://devcloud.ubs.net"
GENESTS_GROUP_PATH = "ubs/gwma/smart-technology-and-analytics/staat-data-science/staat-ds-genesis/genesis-platform"
IKG_PROJECT_PATH = f"{GENESTS_GROUP_PATH}/ikg-dags"
BRANCH = "ikg-master"
SQL_PATH = "dags/ikg/scripts/sql"
TARGET_SCHEMA = "sandbox_prj_smart_insights"
METADATA_TABLE = "ikg_table_lineage_metadata_auto_refresh"
RULE_METADATA_TABLE = "odm_rule_metadata_auto_refresh"
LINEAGE_TEMP_TABLE = "ikg_table_lineage_auto_refresh_temp"
CORE_WMA_METADATA_TABLE = "core_wma_shared_table_list_metadata"
OUTPUT_SUFFIX = "_new.sql"
MODIFIED_SUFFIX = "_modified.sql"
PROFILE_MODIFIED_SUFFIX = "_modified_profile.sql"
EXCLUDE_FOLDER = "ikg_new fa_shhp_map"
PROFILE_DATE_TOKENS = ("IKG_PROFILE_DATE", "IKG_PREV_PROFILE_DATE")
LINEAGE_OUTPUT_PREFIX = "ikg_table_lineage"
LINEAGE_OUTPUT_SUFFIX = "temp"
TABLE_SUFFIX = "_temp_auto"
TEMP_TABLE_SCRIPT_NAME = "core_wma_shared_temp_table_script.sql"
BASE_DB_CONFIG = {
    "host": "greenplum-rdsp.zur.swissbank.com",
    "port": "5432",
    "dbname": "gprdsp",
    "user": "ds_rdsp_dev",
}


@dataclass(frozen=True)
class LineageEntry:
    order_index: int
    source_table: str
    process: Optional[str]
    filename: Optional[str]
    filepath: Optional[str]
    target_table: Optional[str]
    root_profile_table: Optional[str]


@dataclass
class MetadataRow:
    filename: str
    filepath: str
    process: str
    target_table: str
    source_schema: Optional[str]
    source_table: Optional[str]


@dataclass
class DependencyRow:
    order_index: int
    root_profile_table: str
    target_table: str
    source_schema: Optional[str]
    source_table: Optional[str]
    process: str
    filename: str
    filepath: str
    level: int
    current_timestamp: datetime.datetime


@dataclass
class TempTablePlan:
    script_path: Optional[Path]
    script_text: str
    table_map: Dict[str, str]
    timestamp_token: Optional[str]


class GitLabSQLFetcher:
    RETRY_ATTEMPTS = 3
    RETRY_BASE_DELAY = 2.0

    def __init__(self, private_token: str, exclude_folders: Sequence[str]) -> None:
        self._gl = gitlab.Gitlab(GITLAB_URL, private_token=private_token)
        self._project = self._gl.projects.get(IKG_PROJECT_PATH)
        self._exclude = {folder.lower() for folder in exclude_folders}
        self._path_map = self._build_path_map()

    def _build_path_map(self) -> Dict[str, List[str]]:
        path_map: Dict[str, List[str]] = {}
        tree = self._project.repository_tree(
            path=SQL_PATH, ref=BRANCH, recursive=True, get_all=True
        )
        for node in tree:
            if node.get("type") != "blob":
                continue
            path = node.get("path", "")
            if not path.lower().endswith(".sql"):
                continue
            if self._is_excluded(path):
                continue
            key = Path(path).name.lower()
            path_map.setdefault(key, []).append(path)
        return path_map

    def _is_excluded(self, path: str) -> bool:
        parts = {part.lower() for part in Path(path).parts}
        return any(part in self._exclude for part in parts)

    def fetch_sql(self, file_path: str) -> str:
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.RETRY_ATTEMPTS + 1):
            try:
                file_obj = self._project.files.get(file_path=file_path, ref=BRANCH)
                return base64.b64decode(file_obj.content).decode("utf-8", errors="replace")
            except gitlab.exceptions.GitlabGetError as exc:
                status = getattr(exc, "response_code", None)
                if status and int(status) >= 500 and attempt < self.RETRY_ATTEMPTS:
                    wait_seconds = self.RETRY_BASE_DELAY * attempt
                    logging.warning(
                        "GitLab 500 while fetching %s (attempt %d/%d). Retrying in %.1fs",
                        file_path,
                        attempt,
                        self.RETRY_ATTEMPTS,
                        wait_seconds,
                    )
                    time.sleep(wait_seconds)
                    last_exc = exc
                    continue
                raise
        if last_exc:
            raise last_exc
        raise RuntimeError(f"Failed to fetch {file_path}")

    def resolve_path(self, filename: str, process_hint: Optional[str] = None) -> Optional[str]:
        candidates = self._path_map.get(filename.lower())
        if not candidates:
            return None
        if process_hint:
            normalized = process_hint.lower()
            for candidate in candidates:
                if normalized in candidate.lower():
                    return candidate
        return candidates[0]

    def iter_sql_paths(self) -> Iterable[str]:
        return self._path_map.keys()


def build_db_config(password: str) -> Dict[str, str]:
    config = dict(BASE_DB_CONFIG)
    config["password"] = password
    return config


def parse_insight_values(raw: str) -> List[str]:
    if not raw:
        return []
    parts = [part.strip() for part in re.split(r",", raw)]
    return [part for part in parts if part]


def qualified_table(table_name: str) -> sql.SQL:
    return sql.SQL("{}.{}").format(sql.Identifier(TARGET_SCHEMA), sql.Identifier(table_name))


def qualified_table_custom(schema: str, table_name: str) -> sql.SQL:
    return sql.SQL("{}.{}").format(sql.Identifier(schema), sql.Identifier(table_name))


def fetch_profile_tables(
    conn: psycopg2.extensions.connection, insight_types: Sequence[str]
) -> List[str]:
    if not insight_types:
        return []
    query = sql.SQL(
        """
        SELECT DISTINCT profile_table
        FROM {}
        WHERE lower(trim(coalesce(insight_type, ''))) = ANY(%s)
        ORDER BY profile_table
        """
    ).format(qualified_table(RULE_METADATA_TABLE))
    lowered = [value.lower() for value in insight_types]
    with conn.cursor() as cur:
        cur.execute(query, (lowered,))
        rows = [row[0] for row in cur.fetchall() if row and row[0]]
    seen: Set[str] = set()
    ordered: List[str] = []
    for value in rows:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(value)
    return ordered


def fetch_metadata_rows(conn: psycopg2.extensions.connection) -> List[MetadataRow]:
    query = sql.SQL(
        """
        SELECT filename, filepath, process, target_table, source_schema, source_table
        FROM {}
        """
    ).format(qualified_table(METADATA_TABLE))
    with conn.cursor() as cur:
        cur.execute(query)
        return [
            MetadataRow(
                filename=row[0],
                filepath=row[1],
                process=row[2],
                target_table=row[3],
                source_schema=row[4],
                source_table=row[5],
            )
            for row in cur.fetchall()
        ]


def build_adjacency(metadata_rows: Sequence[MetadataRow]) -> Dict[str, List[MetadataRow]]:
    adjacency: Dict[str, List[MetadataRow]] = defaultdict(list)
    for row in metadata_rows:
        if not row.target_table:
            continue
        adjacency[row.target_table.lower()].append(row)
    return adjacency


def build_lineage_rows(
    profile_tables: Sequence[str],
    adjacency: Dict[str, List[MetadataRow]],
    run_timestamp: datetime.datetime,
) -> List[DependencyRow]:
    results: List[DependencyRow] = []
    order_index = 1

    def dfs(
        current_target: str,
        root_target: str,
        level: int,
        path: Set[str],
        edge_seen: Set[Tuple[str, str, str]],
    ) -> None:
        nonlocal order_index
        if not current_target:
            return
        current_lower = current_target.lower()
        if current_lower in path:
            logging.debug("Cycle detected at %s under root %s", current_target, root_target)
            return
        path.add(current_lower)
        rows = adjacency.get(current_lower, [])
        for row in rows:
            source = row.source_table
            if not source:
                continue
            source_lower = source.lower()
            edge_key = (root_target.lower(), row.target_table.lower(), source_lower)
            if edge_key in edge_seen:
                continue
            if source_lower in path:
                logging.debug(
                    "Cycle detected: %s -> %s for root %s", row.target_table, source, root_target
                )
                continue
            dfs(source, root_target, level + 1, path, edge_seen)
            edge_seen.add(edge_key)
            results.append(
                DependencyRow(
                    order_index=order_index,
                    root_profile_table=root_target,
                    target_table=row.target_table,
                    source_schema=row.source_schema,
                    source_table=source,
                    process=row.process,
                    filename=row.filename,
                    filepath=row.filepath,
                    level=level,
                    current_timestamp=run_timestamp,
                )
            )
            order_index += 1
        path.remove(current_lower)

    for root in profile_tables:
        edge_seen: Set[Tuple[str, str, str]] = set()
        dfs(root, root, 1, set(), edge_seen)
        template_rows = adjacency.get(root.lower(), [])
        template = template_rows[0] if template_rows else None
        results.append(
            DependencyRow(
                order_index=order_index,
                root_profile_table=root,
                target_table=root,
                source_schema=None,
                source_table=None,
                process=template.process if template else "",
                filename=template.filename if template else "",
                filepath=template.filepath if template else "",
                level=0,
                current_timestamp=run_timestamp,
            )
        )
        order_index += 1

    return _dedupe_blank_schema_sources(results)


def _dedupe_blank_schema_sources(rows: Sequence[DependencyRow]) -> List[DependencyRow]:
    filtered: List[DependencyRow] = []
    seen: Set[Tuple[str, str, str]] = set()
    for row in rows:
        schema = (row.source_schema or "").strip()
        source = row.source_table or ""
        target = row.target_table or ""
        if not schema and source:
            key = (row.root_profile_table.lower(), target.lower(), source.lower())
            if key in seen:
                continue
            seen.add(key)
        filtered.append(row)
    return filtered


def rows_to_dataframe(rows: Sequence[DependencyRow]) -> pd.DataFrame:
    records = [
        {
            "order_index": row.order_index,
            "root_profile_table": row.root_profile_table,
            "target_table": row.target_table,
            "source_schema": row.source_schema,
            "source_table": row.source_table,
            "process": row.process,
            "filename": row.filename,
            "filepath": row.filepath,
            "level": row.level,
            "current_timestamp": row.current_timestamp,
        }
        for row in rows
    ]
    return pd.DataFrame(records)


def write_temp_excel(
    df: pd.DataFrame,
    run_timestamp: datetime.datetime,
) -> str:
    timestamp_str = run_timestamp.strftime("%Y%m%d%H%M%S")
    output_file = f"{LINEAGE_OUTPUT_PREFIX}_{timestamp_str}_{LINEAGE_OUTPUT_SUFFIX}.xlsx"
    df.to_excel(output_file, index=False)
    logging.info("Wrote %s", output_file)
    return output_file


def refresh_temp_table(
    conn: psycopg2.extensions.connection,
    rows: Sequence[DependencyRow],
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("DROP TABLE IF EXISTS {}").format(qualified_table(LINEAGE_TEMP_TABLE))
        )
        cur.execute(
            sql.SQL(
                """
                CREATE TABLE {} (
                    order_index INTEGER,
                    root_profile_table TEXT,
                    target_table TEXT,
                    source_schema TEXT,
                    source_table TEXT,
                    process TEXT,
                    filename TEXT,
                    filepath TEXT,
                    level INTEGER,
                    "current_timestamp" TIMESTAMP
                )
                """
            ).format(qualified_table(LINEAGE_TEMP_TABLE))
        )
        values = [
            (
                row.order_index,
                row.root_profile_table,
                row.target_table,
                row.source_schema,
                row.source_table,
                row.process,
                row.filename,
                row.filepath,
                row.level,
                row.current_timestamp,
            )
            for row in rows
        ]
        if values:
            execute_values(
                cur,
                sql.SQL(
                    'INSERT INTO {} (order_index, root_profile_table, target_table, source_schema, source_table, process, filename, filepath, level, "current_timestamp") VALUES %s'
                ).format(qualified_table(LINEAGE_TEMP_TABLE)),
                values,
            )
        cur.execute(
            sql.SQL("ALTER TABLE {} OWNER TO {}").format(
                qualified_table(LINEAGE_TEMP_TABLE),
                sql.Identifier("erd_gpdb_prj_smart_insights"),
            )
        )
        cur.execute(
            sql.SQL("GRANT SELECT ON {} TO {}").format(
                qualified_table(LINEAGE_TEMP_TABLE),
                sql.Identifier("erd_gpdb_prj_smart_insights_ro"),
            )
        )
    conn.commit()


def build_dependency_artifacts(
    insight_types: Sequence[str],
    db_config: Dict[str, str],
    run_timestamp: Optional[datetime.datetime] = None,
) -> Tuple[List[DependencyRow], List[str], Optional[str]]:
    timestamp = run_timestamp or datetime.datetime.utcnow()
    with psycopg2.connect(**db_config) as conn:
        profile_tables = fetch_profile_tables(conn, insight_types)
        if not profile_tables:
            logging.warning("No profile tables found for %s", insight_types)
            return [], [], None
        metadata_rows = fetch_metadata_rows(conn)
    adjacency = build_adjacency(metadata_rows)
    lineage_rows = build_lineage_rows(profile_tables, adjacency, timestamp)
    df = rows_to_dataframe(lineage_rows)
    excel_path = write_temp_excel(df, timestamp)
    with psycopg2.connect(**db_config) as conn:
        refresh_temp_table(conn, lineage_rows)
    logging.info(
        "Refreshed %s.%s with %d rows.",
        TARGET_SCHEMA,
        LINEAGE_TEMP_TABLE,
        len(lineage_rows),
    )
    return lineage_rows, profile_tables, excel_path


def fetch_core_metadata_table_names(
    conn: psycopg2.extensions.connection,
    sandbox_schema: str,
) -> List[str]:
    query = sql.SQL(
        """
        SELECT DISTINCT table_name
        FROM {}
        WHERE coalesce(table_name, '') <> ''
        ORDER BY table_name
        """
    ).format(qualified_table_custom(sandbox_schema, CORE_WMA_METADATA_TABLE))
    with conn.cursor() as cur:
        cur.execute(query)
        return [row[0] for row in cur.fetchall() if row and row[0]]


def filter_tables_with_household_column(
    conn: psycopg2.extensions.connection,
    candidate_tables: Sequence[str],
) -> List[str]:
    if not candidate_tables:
        return []
    query = """
        SELECT DISTINCT c.relname
        FROM pg_catalog.pg_class c
        JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
        JOIN pg_catalog.pg_attribute a ON a.attrelid = c.oid
        WHERE n.nspname = 'core_wma_shared'
          AND lower(a.attname) = 'acc_mhh_n'
          AND a.attnum > 0
          AND NOT a.attisdropped
          AND c.relkind = ANY(%s)
          AND c.relname = ANY(%s)
    """
    with conn.cursor() as cur:
        cur.execute(query, (["r", "m", "v"], list(candidate_tables)))
        rows = [row[0] for row in cur.fetchall() if row and row[0]]
    ordered: List[str] = []
    seen: Set[str] = set()
    for name in candidate_tables:
        if name in rows and name.lower() not in seen:
            ordered.append(name)
            seen.add(name.lower())
    return ordered


def render_core_wma_temp_script(
    tables: Sequence[str],
    sandbox_schema: str,
    acc_mhh_n_value: str,
    timestamp_token: str,
) -> Tuple[str, Dict[str, str]]:
    if not tables:
        return "", {}
    lines: List[str] = [
        "-- Auto-generated script to materialize core_wma_shared tables filtered by acc_mhh_n.",
        f"-- Generated at {datetime.datetime.utcnow().isoformat()}",
        "",
    ]
    sanitized_value = acc_mhh_n_value.replace("'", "''")
    table_map: Dict[str, str] = {}
    for table in tables:
        temp_table = f"{table}_temp_{timestamp_token}"
        table_map[table] = temp_table
        lines.append(f"-- Source table: core_wma_shared.{table}")
        lines.append(f"DROP TABLE IF EXISTS {sandbox_schema}.{temp_table};")
        lines.append(
            f"CREATE TABLE {sandbox_schema}.{temp_table} AS\n"
            f"SELECT *\n"
            f"FROM core_wma_shared.{table}\n"
            f"WHERE acc_mhh_n = '{sanitized_value}'\n"
            f"DISTRIBUTED BY (acc_mhh_n);"
        )
        lines.append("")
    return "\n".join(lines).strip() + "\n", table_map


def prepare_temp_table_plan(
    db_config: Dict[str, str],
    sandbox_schema: str,
    acc_mhh_n_value: Optional[str],
) -> TempTablePlan:
    if not acc_mhh_n_value:
        logging.info("No acc_mhh_n provided; skipping vendor temp table preparation.")
        return TempTablePlan(None, "", {}, None)
    with psycopg2.connect(**db_config) as conn:
        candidate_tables = fetch_core_metadata_table_names(conn, sandbox_schema)
        eligible_tables = filter_tables_with_household_column(conn, candidate_tables)
    if not eligible_tables:
        logging.warning(
            "No tables with acc_mhh_n column found in %s.%s.",
            sandbox_schema,
            CORE_WMA_METADATA_TABLE,
        )
        return TempTablePlan(None, "", {}, None)
    timestamp_token = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S%f")[:-3]
    script_text, table_map = render_core_wma_temp_script(
        eligible_tables, sandbox_schema, acc_mhh_n_value, timestamp_token
    )
    script_path = Path(TEMP_TABLE_SCRIPT_NAME)
    script_path.write_text(script_text, encoding="utf-8")
    logging.info("Wrote temp table script to %s", script_path)
    if script_text.strip():
        run_sql_script(script_text, db_config)
        logging.info(
            "Created %d core_wma_shared temp tables filtered by acc_mhh_n.",
            len(table_map),
        )
    return TempTablePlan(script_path, script_text, table_map, timestamp_token)


def fetch_lineage_entries(
    conn: psycopg2.extensions.connection,
) -> List[LineageEntry]:
    query = sql.SQL(
        """
        SELECT
            order_index,
            root_profile_table,
            target_table,
            source_table,
            process,
            filename,
            filepath
        FROM {}
        WHERE source_table IS NOT NULL
          AND (source_schema ILIKE '%ikg%')
        ORDER BY order_index ASC
        """
    ).format(qualified_table(LINEAGE_TEMP_TABLE))
    with conn.cursor() as cur:
        cur.execute(query)
        rows = []
        for (
            order_index,
            root_profile_table,
            target_table,
            source_table,
            process,
            filename,
            filepath,
        ) in cur.fetchall():
            if not source_table:
                continue
            rows.append(
                LineageEntry(
                    order_index=order_index,
                    source_table=source_table,
                    process=process,
                    filename=filename,
                    filepath=filepath,
                    target_table=target_table,
                    root_profile_table=root_profile_table,
                )
            )
    return rows


def fetch_profile_scripts(
    fetcher: "GitLabSQLFetcher", profile_tables: Sequence[str]
) -> List[Tuple[str, str]]:
    scripts: List[Tuple[str, str]] = []
    seen: Set[str] = set()
    for table in profile_tables:
        if not table:
            continue
        filename = f"{table}.sql"
        lower = filename.lower()
        if lower in seen:
            continue
        path = fetcher.resolve_path(filename, None)
        if not path:
            logging.warning(
                "Unable to locate profile table script %s in GitLab repository.", filename
            )
            continue
        try:
            content = fetcher.fetch_sql(path)
        except gitlab.exceptions.GitlabGetError as exc:  # pragma: no cover - external call
            logging.warning("Failed to fetch %s: %s", path, exc)
            continue
        scripts.append((path, content))
        seen.add(lower)
    return scripts


def split_sql_statements(script: str) -> List[str]:
    if sqlparse:
        return [stmt.strip() for stmt in sqlparse.split(script) if stmt.strip()]
    statements: List[str] = []
    current: List[str] = []
    in_single = False
    in_double = False
    prev_char = ""
    for char in script:
        current.append(char)
        if char == "'" and not in_double and prev_char != "\\":
            in_single = not in_single
        elif char == '"' and not in_single and prev_char != "\\":
            in_double = not in_double
        if char == ";" and not in_single and not in_double:
            statement = "".join(current).strip()
            if statement:
                statements.append(statement.rstrip(";").strip())
            current = []
        prev_char = char
    trailing = "".join(current).strip()
    if trailing:
        statements.append(trailing.rstrip(";").strip())
    return statements


def run_sql_script(script_text: str, db_config: Dict[str, str]) -> None:
    statements = split_sql_statements(script_text)
    if not statements:
        logging.warning("No SQL statements to execute.")
        return
    with psycopg2.connect(**db_config) as conn:
        conn.autocommit = False
        with conn.cursor() as cur:
            for idx, statement in enumerate(statements, start=1):
                if not statement:
                    continue
                logging.debug("Executing statement #%d", idx)
                try:
                    cur.execute(statement)
                except Exception as exc:
                    conn.rollback()
                    logging.error("Error executing statement #%d: %s", idx, exc)
                    logging.debug("Failed statement text:\n%s", statement)
                    raise
        conn.commit()
    logging.info("Executed %d SQL statements successfully.", len(statements))


def preview_final_table(
    db_config: Dict[str, str],
    profile_tables: Sequence[str],
    default_schema: str = "core_ikg",
) -> None:
    if not profile_tables:
        logging.info("No profile table available for preview.")
        return
    target = profile_tables[-1]
    if "." in target:
        schema, table = target.split(".", 1)
    else:
        schema, table = default_schema, target
    final_table = f"{table}{TABLE_SUFFIX}"
    query = sql.SQL("SELECT * FROM {}.{} LIMIT 5").format(
        sql.Identifier(schema), sql.Identifier(final_table)
    )
    with psycopg2.connect(**db_config) as conn, conn.cursor() as cur:
        try:
            cur.execute(query)
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description]
        except Exception as exc:
            logging.warning(
                "Unable to preview %s.%s: %s", schema, final_table, exc
            )
            return
    logging.info("Top 5 rows from %s.%s:", schema, final_table)
    for row in rows:
        logging.info(dict(zip(columns, row)))


def _sanitize_token(value: Optional[str], fallback: str) -> str:
    if not value:
        return fallback
    token = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip())
    token = token.strip("_")
    return token.lower() if token else fallback


def sanitize_filename(insight_type: str, profile_table: Optional[str]) -> str:
    profile_part = _sanitize_token(profile_table, "profile_table")
    insight_part = _sanitize_token(insight_type, "insight")
    return f"{profile_part}_{insight_part}{OUTPUT_SUFFIX}"


def stitch_sql(
    entries: Sequence[LineageEntry],
    fetcher: GitLabSQLFetcher,
) -> List[Tuple[str, str]]:
    stitched: List[Tuple[str, str]] = []
    seen_files: set[str] = set()
    for entry in entries:
        filename = f"{entry.source_table}.sql"
        if filename.lower() in seen_files:
            continue
        path = fetcher.resolve_path(filename, entry.process)
        if not path:
            logging.warning("Unable to locate %s in GitLab repository.", filename)
            continue
        try:
            content = fetcher.fetch_sql(path)
        except gitlab.exceptions.GitlabGetError as exc:  # pragma: no cover - network call
            logging.warning("Failed to fetch %s: %s", path, exc)
            continue
        stitched.append((path, content))
        seen_files.add(filename.lower())
    return stitched


PLACEHOLDER_PATTERNS = [
    (re.compile(r"{{\s*params\.IKG_SCHEMA\s*}}", re.IGNORECASE), "core_ikg"),
    (re.compile(r"{{\s*params\.EDW_VIEW_INPUT_SCHEMA\s*}}", re.IGNORECASE), "core_wma_shared"),
    (re.compile(r"{{\s*params\.EDW_INPUT_SCHEMA\s*}}", re.IGNORECASE), "core_wma_shared"),
    (re.compile(r"{{\s*params\.EDW_ETL_SCHEMA\s*}}", re.IGNORECASE), "core_etl"),
    (re.compile(r"{{\s*params\.IKG_CLIP_SCHEMA\s*}}", re.IGNORECASE), "core_in_shared"),
    (re.compile(r"{{\s*params\.IKG_PRE_PROD_SCHEMA\s*}}", re.IGNORECASE), "sandbox_ikg_pre_prd"),
    (re.compile(r"{{\s*params\.IKG_VENDOR_SCHEMA\s*}}", re.IGNORECASE), "core_wma_shared"),
    (re.compile(r"{{\s*params\.IKG_WEALTHX_SCHEMA\s*}}", re.IGNORECASE), "sandbox_prj_smart_relationship"),
    (re.compile(r"{{\s*params\.MODEL_SCHEMA\s*}}", re.IGNORECASE), "core_model"),
    (
        re.compile(r"[\"']?\{\{\s*params\.IKG_TABLE_OWNER_GROUP\s*\}\}[\"']?", re.IGNORECASE),
        "erd_gpdb_prj_smart_insights",
    ),
    (
        re.compile(r"[\"']?\{\{\s*params\.IKG_TABLE_READER_GROUP\s*\}\}[\"']?", re.IGNORECASE),
        "erd_gpdb_prj_smart_insights_ro",
    ),
]


def apply_placeholder_replacements(text: str, profile_date: str) -> str:
    result = text
    for pattern, replacement in PLACEHOLDER_PATTERNS:
        result = pattern.sub(replacement, result)
    for token in PROFILE_DATE_TOKENS:
        partition_pattern = re.compile(
            r"(?P<prefix>[sS])\s*\{\{\s*params\." + token + r"\s*\}\}",
            re.IGNORECASE,
        )
        result = partition_pattern.sub(
            lambda m: f"{m.group('prefix')}{profile_date}", result
        )
        profile_pattern = re.compile(
            r"(?:'\s*)?\{\{\s*params\." + token + r"\s*\}\}(?:\s*')?",
            re.IGNORECASE,
        )
        result = profile_pattern.sub(f"'{profile_date}'", result)
    return result


SCHEMA_SWAP_PATTERN = (
    r"(?:core_wma_shared|{{\s*params\.IKG_VENDOR_SCHEMA\s*}}|"
    r"{{\s*params\.EDW_VIEW_INPUT_SCHEMA\s*}}|{{\s*params\.EDW_INPUT_SCHEMA\s*}})"
)


def apply_temp_schema_swaps(
    text: str, temp_table_map: Dict[str, str], sandbox_schema: str
) -> str:
    if not temp_table_map:
        return text
    result = text
    for source_table, temp_table in temp_table_map.items():
        pattern = re.compile(
            rf"{SCHEMA_SWAP_PATTERN}\s*\.\s*(?P<quote>\"?){re.escape(source_table)}(?P=quote)",
            re.IGNORECASE,
        )
        replacement = f"{sandbox_schema}.{temp_table}"
        result = pattern.sub(replacement, result)
    return result


def find_tables_for_suffix(text: str) -> Set[str]:
    pattern = re.compile(
        r"(?i)create\s+(?:global\s+temp\s+|temp\s+)?table\s+(?!if\s+not\s+exists)(?P<name>(?:[a-z0-9_]+\.)?[a-z0-9_]+)",
    )
    tables: Set[str] = set()
    for match in pattern.finditer(text):
        name = match.group("name")
        if not name:
            continue
        tables.add(name)
    return tables


def derive_forced_table_tokens(table_names: Sequence[str]) -> Set[str]:
    tokens: Set[str] = set()
    for name in table_names:
        if not name:
            continue
        tokens.add(name)
        plain = name.split(".", 1)[-1]
        tokens.add(plain)
    return {token for token in tokens if token}


def enforce_profile_table_suffix(script_text: str, profile_table: str) -> str:
    if not profile_table:
        return script_text
    forced = derive_forced_table_tokens([profile_table])
    tables = find_tables_for_suffix(script_text)
    return apply_table_suffixes(script_text, tables, forced_tables=forced)


def apply_table_suffixes(
    text: str, tables: Set[str], forced_tables: Optional[Set[str]] = None
) -> str:
    names: Set[str] = set(tables)
    if forced_tables:
        names.update(forced_tables)
    sorted_tables = sorted(names, key=len, reverse=True)
    result = text
    for name in sorted_tables:
        escaped = re.escape(name)
        pattern = re.compile(rf"(?<![\w$]){escaped}(?![\w$])", re.IGNORECASE)
        result = pattern.sub(lambda m: append_temp_identifier(m.group(0)), result)
    return result


def append_temp_identifier(identifier: str) -> str:
    if "." in identifier:
        schema, table = identifier.rsplit(".", 1)
        return f"{schema}.{append_temp_suffix(table)}"
    return append_temp_suffix(identifier)


def append_temp_suffix(table_name: str) -> str:
    if table_name.lower().endswith(TABLE_SUFFIX):
        return table_name
    if table_name.endswith('"'):
        base = table_name[:-1]
        return f'{base}{TABLE_SUFFIX}"'
    return f"{table_name}{TABLE_SUFFIX}"


def render_stitched_text(stitched_sql: Sequence[Tuple[str, str]]) -> str:
    lines: List[str] = []
    for path, content in stitched_sql:
        lines.append(f"-- Source: {path}")
        lines.append(content.rstrip())
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_output_file(
    insight_type: str, stitched_text: str, profile_table: Optional[str]
) -> str:
    output_name = sanitize_filename(insight_type, profile_table)
    Path(output_name).write_text(stitched_text, encoding="utf-8")
    logging.info("Wrote stitched SQL to %s", output_name)
    return output_name


def write_modified_file(
    insight_type: str,
    stitched_text: str,
    profile_date: str,
    forced_tables: Optional[Set[str]] = None,
    temp_table_map: Optional[Dict[str, str]] = None,
    sandbox_schema: str = TARGET_SCHEMA,
    profile_table: Optional[str] = None,
) -> Tuple[str, str]:
    modified_text = apply_placeholder_replacements(stitched_text, profile_date)
    if temp_table_map:
        modified_text = apply_temp_schema_swaps(modified_text, temp_table_map, sandbox_schema)
    tables_to_suffix = find_tables_for_suffix(modified_text)
    modified_text = apply_table_suffixes(modified_text, tables_to_suffix, forced_tables)
    output_name = sanitize_filename(insight_type, profile_table).replace(
        OUTPUT_SUFFIX, MODIFIED_SUFFIX
    )
    Path(output_name).write_text(modified_text, encoding="utf-8")
    logging.info("Wrote modified SQL to %s", output_name)
    return output_name, modified_text


def build_profile_only_text(
    profile_script: str,
    profile_date: str,
    forced_tables: Optional[Set[str]] = None,
    temp_table_map: Optional[Dict[str, str]] = None,
    sandbox_schema: str = TARGET_SCHEMA,
) -> str:
    text = apply_placeholder_replacements(profile_script, profile_date)
    if temp_table_map:
        text = apply_temp_schema_swaps(text, temp_table_map, sandbox_schema)
    tables_to_suffix = find_tables_for_suffix(text)
    return apply_table_suffixes(text, tables_to_suffix, forced_tables)


def write_profile_only_file(
    insight_type: str, profile_text: str, profile_table: Optional[str]
) -> str:
    base_name = sanitize_filename(insight_type, profile_table)
    if base_name.endswith(OUTPUT_SUFFIX):
        profile_name = base_name.replace(OUTPUT_SUFFIX, PROFILE_MODIFIED_SUFFIX)
    else:
        profile_name = f"{base_name}_{PROFILE_MODIFIED_SUFFIX}"
    Path(profile_name).write_text(profile_text, encoding="utf-8")
    logging.info("Wrote profile-only SQL to %s", profile_name)
    return profile_name


def extract_created_tables(script_text: str) -> List[str]:
    names: List[str] = []
    seen: Set[str] = set()
    pattern = re.compile(
        r"(?i)create\s+(?:global\s+temp\s+|temp\s+)?table\s+(?!if\s+not\s+exists)(?P<name>(?:[a-z0-9_]+\.)?[a-z0-9_]+)"
    )
    for match in pattern.finditer(script_text):
        name = match.group("name")
        if not name:
            continue
        normalized = name.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        names.append(name)
    return names


def display_table_counts(
    tables: Sequence[str],
    db_config: Dict[str, str],
) -> None:
    if not tables:
        logging.info("No created tables detected for count preview.")
        return
    with psycopg2.connect(**db_config) as conn, conn.cursor() as cur:
        for name in tables:
            if "." in name:
                schema, table = name.split(".", 1)
            else:
                schema, table = TARGET_SCHEMA, name
            query = sql.SQL("SELECT COUNT(*) FROM {}.{}").format(
                sql.Identifier(schema),
                sql.Identifier(table),
            )
            try:
                cur.execute(query)
                count = cur.fetchone()[0]
                logging.info("Row count for %s.%s: %s", schema, table, count)
            except Exception as exc:  # noqa: BLE001
                logging.warning("Unable to count %s.%s: %s", schema, table, exc)


def run_pipeline() -> None:
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.DEBUG),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    insight_raw = input("Enter insight_type (comma-separated allowed): ").strip()
    if not insight_raw:
        raise ValueError("insight_type is required.")
    insight_values = parse_insight_values(insight_raw)
    if not insight_values:
        raise ValueError("At least one insight_type value is required.")

    profile_date = input("Enter profile date (YYYYMMDD): ").strip()
    if not profile_date:
        raise ValueError("profile date is required.")
    acc_mhh_n_value = input("Enter acc_mhh_n (optional, press Enter to skip): ").strip()
    sandbox_schema = (
        input(f"Enter sandbox schema for temp tables [{TARGET_SCHEMA}]: ").strip()
        or TARGET_SCHEMA
    )

    db_password = getpass.getpass("Enter Password for DB User: ")
    db_config = build_db_config(db_password)

    run_timestamp = datetime.datetime.utcnow()
    _, profile_tables, excel_path = build_dependency_artifacts(
        insight_values, db_config, run_timestamp=run_timestamp
    )
    if not profile_tables:
        logging.warning("Stopping because no profile tables were detected.")
        return
    logging.info("Dependency Excel generated at %s", excel_path)
    forced_table_tokens = derive_forced_table_tokens(profile_tables)

    temp_plan = prepare_temp_table_plan(db_config, sandbox_schema, acc_mhh_n_value)
    temp_table_map = temp_plan.table_map

    with psycopg2.connect(**db_config) as conn:
        entries = fetch_lineage_entries(conn)

    if not entries:
        logging.warning(
            "No lineage entries found in %s.%s", TARGET_SCHEMA, LINEAGE_TEMP_TABLE
        )
        return

    private_token = getpass.getpass("Enter your private token: ")
    exclude_folders = [folder for folder in EXCLUDE_FOLDER.split() if folder]
    fetcher = GitLabSQLFetcher(private_token=private_token, exclude_folders=exclude_folders)

    stitched_sql = stitch_sql(entries, fetcher)
    if not stitched_sql:
        logging.warning("No SQL files were stitched; please verify repository contents.")
        return
    profile_scripts = fetch_profile_scripts(fetcher, profile_tables)
    if profile_scripts:
        processed_profiles: List[Tuple[str, str]] = []
        for (path, content), table_name in zip(profile_scripts, profile_tables):
            processed_profiles.append((path, enforce_profile_table_suffix(content, table_name)))
        stitched_sql.extend(processed_profiles)
    elif profile_tables:
        logging.warning("Profile table scripts were not appended; none were retrieved.")

    profile_only_text: Optional[str] = None
    profile_only_path: Optional[str] = None
    if profile_scripts:
        profile_only_text = build_profile_only_text(
            profile_scripts[-1][1],
            profile_date,
            forced_table_tokens,
            temp_table_map,
            sandbox_schema,
        )
        profile_only_path = write_profile_only_file(
            insight_raw, profile_only_text, primary_profile_table
        )

    primary_profile_table = profile_tables[-1] if profile_tables else None
    stitched_text = render_stitched_text(stitched_sql)
    original_path = write_output_file(insight_raw, stitched_text, primary_profile_table)
    modified_path, modified_text = write_modified_file(
        insight_raw,
        stitched_text,
        profile_date,
        forced_table_tokens,
        temp_table_map,
        sandbox_schema,
        primary_profile_table,
    )
    created_tables = sorted(find_tables_for_suffix(modified_text))
    logging.info(
        "Created stitched artifacts for %s: %s, %s",
        insight_raw,
        original_path,
        modified_path,
    )

    try:
        run_sql_script(modified_text, db_config)
        display_table_counts(created_tables, db_config)
        preview_final_table(db_config, profile_tables)
    except Exception as exc:  # pragma: no cover - requires DB
        logging.error("Execution of %s failed: %s", modified_path, exc)

    if profile_only_text and profile_only_path:
        try:
            run_sql_script(profile_only_text, db_config)
            preview_final_table(db_config, profile_tables)
        except Exception as exc:  # pragma: no cover - requires DB
            logging.error("Execution of %s failed: %s", profile_only_path, exc)


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
