import base64
import datetime
import getpass
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Match, Optional, Sequence, Set, Tuple

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
OUTPUT_PREFIX = "ikg_table_lineage_metadata"
TARGET_SCHEMA = "sandbox_prj_smart_insights"
TARGET_TABLE = "ikg_table_lineage_metadata_auto_refresh"
TARGET_OWNER = "erd_gpdb_prj_smart_insights"
TARGET_READER = "erd_gpdb_prj_smart_insights_ro"
SQL_PATH_PARTS = Path(SQL_PATH).parts
LINEAGE_COLUMNS = [
    "filename",
    "filepath",
    "process",
    "target_table",
    "source_schema",
    "source_table",
    "current_timestamp",
]


@dataclass(frozen=True)
class LineageRow:
    filename: str
    filepath: str
    process: str
    target_table: str
    source_schema: Optional[str]
    source_table: Optional[str]
    current_timestamp: datetime.datetime


def derive_process(file_path: str) -> str:
    parts = Path(file_path).parts
    prefix_len = len(SQL_PATH_PARTS)
    if parts[:prefix_len] == SQL_PATH_PARTS and len(parts) > prefix_len:
        return parts[prefix_len]
    return ""


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
    SOURCE_CONTEXTS: Tuple[type, ...] = (
        exp.From,
        exp.Join,
        exp.Subquery,
        exp.Select,
        exp.With,
        exp.Union,
        exp.Where,
        exp.Group,
        exp.Having,
        exp.Order,
        exp.SetOperation,
    )

    def extract_tables(self, sql_text: str) -> Set[Tuple[Optional[str], str]]:
        """Return a set of (schema, table) pairs referenced inside the SQL text."""
        cleaned = self._remove_sql_comments(sql_text)
        normalized = self._strip_vendor_specific(cleaned)
        sanitized, placeholders = self._replace_templates(normalized)
        sanitized_no_do, do_blocks = self._strip_do_blocks(sanitized)
        logging.debug("Sanitized SQL length: %d", len(sanitized_no_do))
        tables = self._parse_sql_to_tables(sanitized_no_do, placeholders)
        for block in do_blocks:
            tables |= self._parse_do_block(block, placeholders)
        return tables

    def _remove_sql_comments(self, sql_text: str) -> str:
        no_block = re.sub(r"/\*.*?\*/", "", sql_text, flags=re.S)
        no_inline = re.sub(r"--.*?$", "", no_block, flags=re.M)
        return no_inline

    def _strip_do_blocks(self, sql_text: str) -> Tuple[str, List[str]]:
        blocks: List[str] = []
        pattern = re.compile(
            r"do\s+\$\$(.*?)\$\$\s*(?:language\s+\w+)?\s*;",
            flags=re.I | re.S,
        )

        def repl(match: Match) -> str:
            blocks.append(match.group(1))
            return ""

        stripped = pattern.sub(repl, sql_text)
        return stripped, blocks

    def _strip_vendor_specific(self, sql_text: str) -> str:
        patterns = [
            r"\bdistributed\s+by\s*\([^;]+?\)",
            r"\bdistributed\s+replicated",
            r"\bon\s+commit\s+preserve\s+rows",
            r"\bwith\s*\(.*?appendonly.*?\)",
            r"\bencode\s*'.*?'",
            r"\borganization\s*\([^)]*\)",
            r"\bpartition\s+by\s+range\s+\([^)]*\)",
        ]
        cleaned = sql_text
        for pattern in patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.I | re.S)
        return cleaned

    def _replace_templates(self, sql_text: str) -> Tuple[str, Dict[str, str]]:
        placeholders: Dict[str, str] = {}

        def repl(match: Match) -> str:
            inner = re.sub(r"\s+", "", match.group(1))
            # Replace Jinja-style placeholders with SQL-safe tokens and remember originals.
            token = f"TEMPLATE_TOKEN_{len(placeholders)}"
            placeholders[token] = f"{{{{{inner}}}}}"
            return token

        sanitized = self.TEMPLATE_PATTERN.sub(repl, sql_text)
        return sanitized, placeholders

    def _parse_sql_to_tables(
        self, sql_text: str, placeholders: Dict[str, str]
    ) -> Set[Tuple[Optional[str], str]]:
        if not sql_text.strip():
            return set()
        try:
            parsed = sqlglot.parse(sql_text, read="postgres", error_level="ignore")
        except sqlglot.errors.ParseError as exc:
            message = str(exc).lower()
            if "alter table" in message:
                logging.debug("sqlglot parse error (alter table): %s", exc)
            else:
                logging.warning("sqlglot failed to parse SQL (%s). Falling back to regex.", exc)
            return self._regex_fallback(sql_text, placeholders)
        if not parsed:
            return set()
        if all(isinstance(statement, exp.Command) for statement in parsed):
            logging.debug("sqlglot returned Command nodes only; using regex fallback.")
            return self._regex_fallback(sql_text, placeholders)
        return self._collect_tables(parsed, placeholders)

    def _collect_tables(
        self, statements: List[exp.Expression], placeholders: Dict[str, str]
    ) -> Set[Tuple[Optional[str], str]]:
        tables: Set[Tuple[Optional[str], str]] = set()
        placeholder_lookup = {k.lower(): v for k, v in placeholders.items()}

        for statement in statements:
            if statement is None:
                continue
            if isinstance(statement, (exp.Alter, exp.Grant)):
                logging.debug("Skipping %s statement", statement.__class__.__name__)
                continue
            if isinstance(statement, exp.Command):
                this_obj = statement.this
                if isinstance(this_obj, exp.Expression):
                    command_text = this_obj.sql().upper()
                else:
                    command_text = str(this_obj or "").upper()
                if command_text.startswith("ALTER") or command_text.startswith("GRANT"):
                    logging.debug("Skipping command statement: %s", command_text[:20])
                    continue
            cte_names = {
                (cte.alias_or_name or "").lower() for cte in statement.find_all(exp.CTE)
            }
            for table in statement.find_all(exp.Table):
                if not self._is_source_context(table):
                    continue
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
        self, sanitized_sql: str, placeholders: Dict[str, str], depth: int = 0
    ) -> Set[Tuple[Optional[str], str]]:
        if depth > 5:
            logging.debug("Regex fallback max depth reached.")
            return set()
        placeholder_lookup = {k.lower(): v for k, v in placeholders.items()}
        pattern = re.compile(
            r"""(?ix)
            (?:from|inner\s+join|left\s+(?:outer\s+)?join|right\s+(?:outer\s+)?join|full\s+(?:outer\s+)?join|cross\s+join|join)\s+
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
        subquery_pattern = re.compile(
            r"""(?is)from\s+\((?P<inner>select.+?)\)\s+[a-z0-9_]+""",
        )
        for submatch in subquery_pattern.finditer(sanitized_sql):
            inner_sql = submatch.group("inner")
            tables |= self._regex_fallback(inner_sql, placeholders, depth + 1)
        return tables

    def _is_source_context(self, table: exp.Table) -> bool:
        return any(table.find_ancestor(ctx) is not None for ctx in self.SOURCE_CONTEXTS)

    def _parse_do_block(
        self, block_text: str, placeholders: Dict[str, str]
    ) -> Set[Tuple[Optional[str], str]]:
        prepared = self._prepare_do_block(block_text)
        tables = self._parse_sql_to_tables(prepared, placeholders)
        if tables:
            return tables
        return self._regex_fallback(prepared, placeholders)

    def _prepare_do_block(self, block_text: str) -> str:
        block = self._remove_raise_statements(block_text)
        block = re.sub(r"\blanguage\s+\w+\s*;?", "", block, flags=re.I)
        block = re.sub(r"^\s*begin\b", "", block, flags=re.I)
        block = re.sub(r"\bend\s*;?\s*$", "", block, flags=re.I)
        block = re.sub(r"\bif\b.+?\bthen\b", "", block, flags=re.I | re.S)
        block = re.sub(r"\belse\b", "", block, flags=re.I)
        block = re.sub(r"\bend\s+if\b", "", block, flags=re.I)
        return block

    def _remove_raise_statements(self, sql_text: str) -> str:
        return re.sub(r"\braise\s+(?:exception|error).*?;", "", sql_text, flags=re.I | re.S)


class LineageBuilder:
    def __init__(self, fetcher: GitLabSQLFetcher, parser: SQLParser) -> None:
        self._fetcher = fetcher
        self._parser = parser

    def build(
        self,
        progress_callback: Optional[Callable[[Sequence[LineageRow]], None]] = None,
        run_timestamp: Optional[datetime.datetime] = None,
    ) -> List[LineageRow]:
        timestamp = run_timestamp or datetime.datetime.utcnow()
        rows: List[LineageRow] = []
        for file_path in self._fetcher.iter_sql_paths():
            logging.info("Processing %s", file_path)
            try:
                sql_text = self._fetcher.fetch_sql(file_path)
                tables = self._parser.extract_tables(sql_text)
                if not tables:
                    tables = {(None, None)}
                filename = Path(file_path).name
                target_table = Path(filename).stem
                process = derive_process(file_path)
                for schema, table in tables:
                    row = LineageRow(
                        filename=filename,
                        filepath=file_path,
                        process=process,
                        target_table=target_table,
                        source_schema=schema,
                        source_table=table,
                        current_timestamp=timestamp,
                    )
                    rows.append(row)
            except Exception as exc:  # noqa: BLE001
                logging.exception("Failed to process %s: %s", file_path, exc)
            finally:
                if progress_callback:
                    progress_callback(self._filter_self_references(rows))
        return self._filter_self_references(rows)

    @staticmethod
    def _filter_self_references(rows: Sequence[LineageRow]) -> List[LineageRow]:
        return [
            row
            for row in rows
            if not (row.source_table and row.source_table.lower() == row.target_table.lower())
        ]


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
                            process TEXT,
                            target_table TEXT,
                            source_schema TEXT,
                            source_table TEXT,
                            "current_timestamp" TIMESTAMP
                        )
                        """
                    ).format(sql.Identifier(TARGET_SCHEMA), sql.Identifier(TARGET_TABLE))
                )
                values = [
                    (
                        row.filename,
                        row.filepath,
                        row.process,
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
                            'INSERT INTO {}.{} (filename, filepath, process, target_table, source_schema, source_table, "current_timestamp") VALUES %s'
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
            "process": row.process,
            "target_table": row.target_table,
            "source_schema": row.source_schema,
            "source_table": row.source_table,
            "current_timestamp": row.current_timestamp,
        }
        for row in rows
    ]
    return pd.DataFrame(data, columns=LINEAGE_COLUMNS)


def write_to_excel(
    df: pd.DataFrame,
    run_timestamp: Optional[datetime.datetime] = None,
    output_path: Optional[str] = None,
) -> str:
    if output_path is None:
        if run_timestamp is None:
            raise ValueError("run_timestamp must be provided when output_path is None.")
        timestamp_str = run_timestamp.strftime("%Y%m%d%H%M%S")
        output_path = f"{OUTPUT_PREFIX}_{timestamp_str}.xlsx"
    df.to_excel(output_path, index=False)
    logging.info("Wrote %s", output_path)
    return output_path


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
    run_timestamp = datetime.datetime.utcnow()
    output_file = f"{OUTPUT_PREFIX}_{run_timestamp.strftime('%Y%m%d%H%M%S')}.xlsx"
    # Initialize the Excel file with headers so it exists even if parsing fails early.
    write_to_excel(rows_to_dataframe([]), output_path=output_file)

    def flush_excel(current_rows: Sequence[LineageRow]) -> None:
        df_snapshot = rows_to_dataframe(current_rows)
        write_to_excel(df_snapshot, output_path=output_file)

    rows = builder.build(progress_callback=flush_excel, run_timestamp=run_timestamp)
    logging.info("Captured %d lineage rows", len(rows))

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
