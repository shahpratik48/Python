import base64
import getpass
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import gitlab
import psycopg2
from psycopg2 import sql

try:
    import sqlparse
except ImportError:
    sqlparse = None


LOG_LEVEL = os.environ.get("IKG_LINEAGE_LOG_LEVEL", "DEBUG")
GITLAB_URL = "https://devcloud.ubs.net"
GENESTS_GROUP_PATH = "ubs/gwma/smart-technology-and-analytics/staat-data-science/staat-ds-genesis/genesis-platform"
IKG_PROJECT_PATH = f"{GENESTS_GROUP_PATH}/ikg-dags"
BRANCH = "ikg-master"
SQL_PATH = "dags/ikg/scripts/sql"
TARGET_SCHEMA = "sandbox_prj_smart_insights"
LINEAGE_TEMP_TABLE = "ikg_table_lineage_auto_refresh_temp"
RULE_METADATA_TABLE = "odm_rule_metadata_auto_refresh"
OUTPUT_SUFFIX = "_new.sql"
MODIFIED_SUFFIX = "_modified.sql"
PROFILE_MODIFIED_SUFFIX = "_modified_profile.sql"
EXCLUDE_FOLDER = "ikg_new fa_shhp_map"
PROFILE_DATE_TOKENS = ("IKG_PROFILE_DATE", "IKG_PREV_PROFILE_DATE")


@dataclass(frozen=True)
class LineageEntry:
    order_index: int
    source_table: str
    process: Optional[str]


class GitLabSQLFetcher:
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
        file_obj = self._project.files.get(file_path=file_path, ref=BRANCH)
        return base64.b64decode(file_obj.content).decode("utf-8", errors="replace")

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


def fetch_lineage_entries(
    conn: psycopg2.extensions.connection,
) -> List[LineageEntry]:
    query = sql.SQL(
        """
        SELECT order_index, source_table, process
        FROM {}.{}
        WHERE source_schema ILIKE '%ikg%'
        ORDER BY order_index ASC
        """
    ).format(sql.Identifier(TARGET_SCHEMA), sql.Identifier(LINEAGE_TEMP_TABLE))
    with conn.cursor() as cur:
        cur.execute(query)
        rows = []
        for order_index, source_table, process in cur.fetchall():
            if not source_table:
                continue
            rows.append(
                LineageEntry(
                    order_index=order_index,
                    source_table=source_table,
                    process=process,
                )
            )
    return rows


def parse_insight_values(raw: str) -> List[str]:
    if not raw:
        return []
    parts = [part.strip() for part in re.split(r",", raw)]
    return [part for part in parts if part]


def fetch_profile_tables(
    conn: psycopg2.extensions.connection, insight_types: Sequence[str]
) -> List[str]:
    if not insight_types:
        return []
    query = sql.SQL(
        """
        SELECT DISTINCT profile_table
        FROM {}.{}
        WHERE lower(trim(coalesce(insight_type, ''))) = ANY(%s)
        ORDER BY profile_table
        """
    ).format(sql.Identifier(TARGET_SCHEMA), sql.Identifier(RULE_METADATA_TABLE))
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
        except gitlab.exceptions.GitlabGetError as exc:
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
    final_table = f"{table}_temp"
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


def sanitize_filename(insight_type: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", insight_type.strip())
    safe = safe.strip("_")
    if not safe:
        safe = "insight"
    return f"{safe.lower()}{OUTPUT_SUFFIX}"


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
        except gitlab.exceptions.GitlabGetError as exc:
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
        re.compile(r"{{\s*params\.IKG_TABLE_OWNER_GROUP\s*}}", re.IGNORECASE),
        "erd_gpdb_prj_smart_insights",
    ),
    (
        re.compile(r"{{\s*params\.IKG_TABLE_READER_GROUP\s*}}", re.IGNORECASE),
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


def apply_table_suffixes(text: str, tables: Set[str]) -> str:
    sorted_tables = sorted(tables, key=len, reverse=True)
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
    if table_name.endswith('"'):
        base = table_name[:-1]
        return f'{base}_temp"'
    return f"{table_name}_temp"


def render_stitched_text(stitched_sql: Sequence[Tuple[str, str]]) -> str:
    lines: List[str] = []
    for path, content in stitched_sql:
        lines.append(f"-- Source: {path}")
        lines.append(content.rstrip())
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_output_file(insight_type: str, stitched_text: str) -> str:
    output_name = sanitize_filename(insight_type)
    Path(output_name).write_text(stitched_text, encoding="utf-8")
    logging.info("Wrote stitched SQL to %s", output_name)
    return output_name


def write_modified_file(
    insight_type: str,
    stitched_text: str,
    profile_date: str,
) -> Tuple[str, str]:
    modified_text = apply_placeholder_replacements(stitched_text, profile_date)
    tables_to_suffix = find_tables_for_suffix(modified_text)
    modified_text = apply_table_suffixes(modified_text, tables_to_suffix)
    output_name = sanitize_filename(insight_type).replace(OUTPUT_SUFFIX, MODIFIED_SUFFIX)
    Path(output_name).write_text(modified_text, encoding="utf-8")
    logging.info("Wrote modified SQL to %s", output_name)
    return output_name, modified_text


def build_profile_only_text(profile_script: str, profile_date: str) -> str:
    text = apply_placeholder_replacements(profile_script, profile_date)
    tables_to_suffix = find_tables_for_suffix(text)
    return apply_table_suffixes(text, tables_to_suffix)


def write_profile_only_file(insight_type: str, profile_text: str) -> str:
    base_name = sanitize_filename(insight_type)
    if base_name.endswith(OUTPUT_SUFFIX):
        profile_name = base_name.replace(OUTPUT_SUFFIX, PROFILE_MODIFIED_SUFFIX)
    else:
        profile_name = f"{base_name}_{PROFILE_MODIFIED_SUFFIX}"
    Path(profile_name).write_text(profile_text, encoding="utf-8")
    logging.info("Wrote profile-only SQL to %s", profile_name)
    return profile_name


def main() -> None:
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.DEBUG),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    insight_raw = input("Enter insight_type: ").strip()
    if not insight_raw:
        raise ValueError("insight_type is required.")
    insight_values = parse_insight_values(insight_raw)
    if not insight_values:
        raise ValueError("At least one insight_type value is required.")
    profile_date = input("Enter profile date (YYYYMMDD): ").strip()
    if not profile_date:
        raise ValueError("profile date is required.")
    db_password = getpass.getpass("Enter Password for DB User: ")

    db_config = {
        "host": "greenplum-rdsp.zur.swissbank.com",
        "port": "5432",
        "dbname": "gprdsp",
        "user": "ds_rdsp_dev",
        "password": db_password,
    }

    with psycopg2.connect(**db_config) as conn:
        entries = fetch_lineage_entries(conn)
        profile_tables = fetch_profile_tables(conn, insight_values)

    if not entries:
        logging.warning("No lineage entries found in %s.%s", TARGET_SCHEMA, LINEAGE_TEMP_TABLE)
        return
    if not profile_tables:
        logging.warning(
            "No profile tables found in %s.%s for the provided insight types.",
            TARGET_SCHEMA,
            RULE_METADATA_TABLE,
        )

    private_token = getpass.getpass("Enter your private token: ")
    exclude_folders = [folder for folder in EXCLUDE_FOLDER.split() if folder]
    fetcher = GitLabSQLFetcher(private_token=private_token, exclude_folders=exclude_folders)

    stitched_sql = stitch_sql(entries, fetcher)
    if not stitched_sql:
        logging.warning("No SQL files were stitched; please verify repository contents.")
        return
    profile_scripts = fetch_profile_scripts(fetcher, profile_tables)
    if profile_scripts:
        stitched_sql.extend(profile_scripts)
    elif profile_tables:
        logging.warning("Profile table scripts were not appended; none were retrieved.")
    profile_only_text = None
    profile_only_path = None
    if profile_scripts:
        profile_only_text = build_profile_only_text(
            profile_scripts[-1][1], profile_date
        )
        profile_only_path = write_profile_only_file(insight_raw, profile_only_text)

    stitched_text = render_stitched_text(stitched_sql)
    original_path = write_output_file(insight_raw, stitched_text)
    modified_path, modified_text = write_modified_file(
        insight_raw, stitched_text, profile_date
    )
    logging.info(
        "Created original and modified SQL files for %s: %s, %s",
        insight_raw,
        original_path,
        modified_path,
    )

    if profile_only_text:
        try:
            run_sql_script(profile_only_text, db_config)
        except Exception:
            logging.error(
                "Execution of %s failed.", profile_only_path or "profile-only script"
            )
            raise
        preview_final_table(db_config, profile_tables)
    else:
        logging.warning(
            "No profile-only script available to execute; skipping run and preview."
        )


if __name__ == "__main__":
    main()
