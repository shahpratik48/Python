import base64
import datetime
import getpass
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import gitlab
import psycopg2
from psycopg2 import sql


LOG_LEVEL = os.environ.get("IKG_LINEAGE_LOG_LEVEL", "DEBUG")
GITLAB_URL = "https://devcloud.ubs.net"
GENESTS_GROUP_PATH = "ubs/gwma/smart-technology-and-analytics/staat-data-science/staat-ds-genesis/genesis-platform"
IKG_PROJECT_PATH = f"{GENESTS_GROUP_PATH}/ikg-dags"
BRANCH = "ikg-master"
SQL_PATH = "dags/ikg/scripts/sql"
TARGET_SCHEMA = "sandbox_prj_smart_insights"
LINEAGE_TEMP_TABLE = "ikg_table_lineage_auto_refresh_temp"
OUTPUT_SUFFIX = "_new.sql"
EXCLUDE_FOLDER = "ikg_new fa_shhp_map"


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


def write_output_file(
    insight_type: str,
    stitched_sql: Sequence[Tuple[str, str]],
) -> str:
    output_name = sanitize_filename(insight_type)
    lines: List[str] = []
    for path, content in stitched_sql:
        lines.append(f"-- Source: {path}")
        lines.append(content.rstrip())
        lines.append("")  # blank line between scripts
    Path(output_name).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    logging.info("Wrote stitched SQL to %s", output_name)
    return output_name


def main() -> None:
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.DEBUG),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    insight_type = input("Enter insight_type: ").strip()
    if not insight_type:
        raise ValueError("insight_type is required.")
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

    if not entries:
        logging.warning("No lineage entries found in %s.%s", TARGET_SCHEMA, LINEAGE_TEMP_TABLE)
        return

    private_token = getpass.getpass("Enter your private token: ")
    exclude_folders = [folder for folder in EXCLUDE_FOLDER.split() if folder]
    fetcher = GitLabSQLFetcher(private_token=private_token, exclude_folders=exclude_folders)

    stitched_sql = stitch_sql(entries, fetcher)
    if not stitched_sql:
        logging.warning("No SQL files were stitched; please verify repository contents.")
        return

    write_output_file(insight_type, stitched_sql)


if __name__ == "__main__":
    main()
