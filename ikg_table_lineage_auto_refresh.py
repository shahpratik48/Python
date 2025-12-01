import base64
import datetime
import getpass
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Match, Optional, Sequence, Set, Tuple

import gitlab
import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
import sqlglot
from sqlglot import exp


LOG_LEVEL = os.environ.get("IKG_LINEAGE_LOG_LEVEL", "DEBUG")
GITLAB_URL = "https://devcloud.ubs.net"
GENESTS_GROUP_PATH = "ubs/gwma/smart-technology-and-analytics/staat-data-science/staat-ds-genesis/genesis-platform"
IKG_PROJECT_PATH = f"{GENESTS_GROUP_PATH}/ikg-dags"
BRANCH = "ikg-master"
SQL_PATH = "dags/ikg/scripts/sql"
EXCLUDE_FOLDER = "ikg_new fa_shhp_map"
OUTPUT_PREFIX = "ikg_table_lineage"
TARGET_SCHEMA = "sandbox_prj_smart_insights"
TARGET_TABLE = "ikg_table_lineage_auto_refresh"
TARGET_OWNER = "erd_gpdbprj_smart_insights"
TARGET_READER = "erd_gpdb_prj_smart_insights_ro"


@dataclass(frozen=True)
class LineageRow:
    filename: str
    filepath: str
    target_table: str
    source_schema: Optional[str]
    source_table: Optional[str]
    current_timestamp: datetime.datetime


class GitLabSQLFetcher:
    def __init__(self, private_token: str, exclude_folders: Sequence[str]) -> None:
        self._gl = gitlab.Gitlab(GITLAB_URL, private_token=private_token)
        self._project = self._gl.projects.get(IKG_PROJECT_PATH)
        self._exclude_folders = {folder.lower() for folder in exclude_folders}

    def iter_sql_paths(self) -> Iterable[str]:
        tree = self._project.repository_tree(
            path=SQL_PATH, ref=BRANCH, recursive=True, all=True
        )
        for node in tree:
            if node.get("type") != "blob":
                continue
            path = node.get("path", "")
            if not path.lower().endswith(".sql"):
                continue
            if self._is_excluded(path):
                logging.debug("Skipping excluded path: %s", path)
                continue
            yield path

    def fetch_sql(self, file_path: str) -> str:
        f = self._project.files.get(file_path=file_path, ref=BRANCH)
        decoded = base64.b64decode(f.content).decode("utf-8", errors="replace")
        return decoded

    def _is_excluded(self, path: str) -> bool:
        parts = {part.lower() for part in Path(path).parts}
        return any(part in self._exclude_folders for part in parts)


class SQLParser:
    TEMPLATE_PATTERN = re.compile(r"{{\s*([^{}]+?)\s*}}")

    def extract_tables(self, sql_text: str) -> Set[Tuple[Optional[str], str]]:
        """Return a set of (schema, table) pairs referenced inside the SQL text."""
        cleaned = self._remove_sql_comments(sql_text)
        sanitized, placeholders = self._replace_templates(cleaned)
        logging.debug("Sanitized SQL length: %d", len(sanitized))
        try:
            parsed = sqlglot.parse(sanitized, read="postgres")
            return self._collect_tables(parsed, placeholders)
        except sqlglot.errors.ParseError as exc:
            logging.warning("sqlglot failed to parse SQL (%s). Falling back to regex.", exc)
            return self._regex_fallback(sanitized, placeholders)

    def _remove_sql_comments(self, sql_text: str) -> str:
        no_block = re.sub(r"/\*.*?\*/", "", sql_text, flags=re.S)
        no_inline = re.sub(r"--.*?$", "", no_block, flags=re.M)
        return no_inline

    def _replace_templates(self, sql_text: str) -> Tuple[str, Dict[str, str]]:
        placeholders: Dict[str, str] = {}

        def repl(match: Match[str]) -> str:
            inner = match.group(1).strip()
            # Replace Jinja-style placeholders with SQL-safe tokens and remember originals.
            token = f"TEMPLATE_TOKEN_{len(placeholders)}"
            placeholders[token] = f"{{{{{inner}}}}}"
            return token

        sanitized = self.TEMPLATE_PATTERN.sub(repl, sql_text)
        return sanitized, placeholders

    def _collect_tables(
        self, statements: List[exp.Expression], placeholders: Dict[str, str]
    ) -> Set[Tuple[Optional[str], str]]:
        tables: Set[Tuple[Optional[str], str]] = set()
        placeholder_lookup = {k.lower(): v for k, v in placeholders.items()}

        for statement in statements:
            cte_names = {
                (cte.alias or "").lower() for cte in statement.find_all(exp.CTE)
            }
            for table in statement.find_all(exp.Table):
                raw_name = table.name
                if not raw_name:
                    continue
                table_name = placeholder_lookup.get(raw_name.lower(), raw_name)
                candidate = table_name.lower()
                if candidate in cte_names and not table.db:
                    continue
                schema = table.db
                if schema:
                    restored = placeholder_lookup.get(schema.lower())
                    schema_name = restored or schema
                else:
                    schema_name = None
                tables.add((schema_name, table_name))
        return tables

    def _regex_fallback(
        self, sanitized_sql: str, placeholders: Dict[str, str]
    ) -> Set[Tuple[Optional[str], str]]:
        placeholder_lookup = {k.lower(): v for k, v in placeholders.items()}
        pattern = re.compile(
            r"""(?ix)
            (?:from|join|into|update|table|truncate|delete\s+from)\s+
            (?:
                (?P<schema>[a-z0-9_]+)\.(?P<table>[a-z0-9_]+) |
                (?P<table_only>[a-z0-9_]+)
            )
            """,
        )
        tables: Set[Tuple[Optional[str], str]] = set()
        for match in pattern.finditer(sanitized_sql):
            schema = match.group("schema")
            table_name = match.group("table") or match.group("table_only")
            if not table_name:
                continue
            schema_name: Optional[str]
            if schema:
                schema_name = placeholder_lookup.get(schema.lower()) or schema
            else:
                schema_name = None
            restored_table = placeholder_lookup.get(table_name.lower()) or table_name
            tables.add((schema_name, restored_table))
        return tables


class LineageBuilder:
    def __init__(self, fetcher: GitLabSQLFetcher, parser: SQLParser) -> None:
        self._fetcher = fetcher
        self._parser = parser

    def build(self) -> List[LineageRow]:
        timestamp = datetime.datetime.utcnow()
        rows: List[LineageRow] = []
        for file_path in self._fetcher.iter_sql_paths():
            logging.info("Processing %s", file_path)
            sql_text = self._fetcher.fetch_sql(file_path)
            tables = self._parser.extract_tables(sql_text)
            if not tables:
                tables = {(None, None)}
            filename = Path(file_path).name
            target_table = Path(filename).stem
            for schema, table in tables:
                row = LineageRow(
                    filename=filename,
                    filepath=file_path,
                    target_table=target_table,
                    source_schema=schema,
                    source_table=table,
                    current_timestamp=timestamp,
                )
                rows.append(row)
        return rows


class DatabaseUploader:
    def __init__(self, db_config: Dict[str, str]) -> None:
        self._db_config = db_config

    def refresh_table(self, rows: Sequence[LineageRow]) -> None:
        logging.info("Loading %d rows into %s.%s", len(rows), TARGET_SCHEMA, TARGET_TABLE)
        with psycopg2.connect(**self._db_config) as conn:
            conn.autocommit = False
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("DROP TABLE IF EXISTS {}.{}").format(
                        sql.Identifier(TARGET_SCHEMA), sql.Identifier(TARGET_TABLE)
                    )
                )
                cur.execute(
                    sql.SQL(
                        """
                        CREATE TABLE {}.{} (
                            filename TEXT,
                            filepath TEXT,
                            target_table TEXT,
                            source_schema TEXT,
                            source_table TEXT,
                            current_timestamp TIMESTAMP
                        )
                        """
                    ).format(sql.Identifier(TARGET_SCHEMA), sql.Identifier(TARGET_TABLE))
                )
                values = [
                    (
                        row.filename,
                        row.filepath,
                        row.target_table,
                        row.source_schema,
                        row.source_table,
                        row.current_timestamp,
                    )
                    for row in rows
                ]
                if values:
                    execute_values(
                        cur,
                        sql.SQL(
                            "INSERT INTO {}.{} (filename, filepath, target_table, source_schema, source_table, current_timestamp) VALUES %s"
                        ).format(
                            sql.Identifier(TARGET_SCHEMA), sql.Identifier(TARGET_TABLE)
                        ),
                        values,
                    )
                cur.execute(
                    sql.SQL(
                        "ALTER TABLE {}.{} OWNER TO {}"
                    ).format(
                        sql.Identifier(TARGET_SCHEMA),
                        sql.Identifier(TARGET_TABLE),
                        sql.Identifier(TARGET_OWNER),
                    )
                )
                cur.execute(
                    sql.SQL(
                        "GRANT SELECT ON {}.{} TO {}"
                    ).format(
                        sql.Identifier(TARGET_SCHEMA),
                        sql.Identifier(TARGET_TABLE),
                        sql.Identifier(TARGET_READER),
                    )
                )
            conn.commit()


def rows_to_dataframe(rows: Sequence[LineageRow]) -> pd.DataFrame:
    data = [
        {
            "filename": row.filename,
            "filepath": row.filepath,
            "target_table": row.target_table,
            "source_schema": row.source_schema,
            "source_table": row.source_table,
            "current_timestamp": row.current_timestamp,
        }
        for row in rows
    ]
    return pd.DataFrame(data)


def write_to_excel(df: pd.DataFrame, run_timestamp: datetime.datetime) -> str:
    timestamp_str = run_timestamp.strftime("%Y%m%d%H%M%S")
    output_file = f"{OUTPUT_PREFIX}_{timestamp_str}.xlsx"
    df.to_excel(output_file, index=False)
    logging.info("Wrote %s", output_file)
    return output_file


def main() -> None:
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.DEBUG),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    private_token = getpass.getpass("Enter your private token: ")
    db_password = getpass.getpass("Enter Password for DB User: ")

    exclude_folders = [folder for folder in EXCLUDE_FOLDER.split() if folder]
    fetcher = GitLabSQLFetcher(private_token=private_token, exclude_folders=exclude_folders)
    parser = SQLParser()
    builder = LineageBuilder(fetcher, parser)
    rows = builder.build()

    df = rows_to_dataframe(rows)
    now = datetime.datetime.utcnow()
    write_to_excel(df, now)

    db_config = {
        "host": "greenplum-rdsp.zur.swissbank.com",
        "port": "5432",
        "dbname": "gprdsp",
        "user": "ds_rdsp_dev",
        "password": db_password,
    }
    uploader = DatabaseUploader(db_config)
    uploader.refresh_table(rows)


if __name__ == "__main__":
    main()
