"""
IKG metadata extraction utility.

This script connects to GitLab, downloads SQL assets, parses them with sqlglot,
produces lineage metadata, writes the result to Excel, and loads it into
Greenplum as sandbox_prj_smart_insights.ikg_metadata_auto_refresh.
"""

from __future__ import annotations

import argparse
import base64
import getpass
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import gitlab
import pandas as pd
import psycopg2
from gitlab import exceptions as gl_exceptions
from psycopg2.extras import execute_values
from sqlglot import exp, parse_one
from sqlglot.errors import ParseError, SqlglotError
from sqlglot.lineage import lineage
from sqlglot.tokens import TokenType, Tokenizer

LOGGER = logging.getLogger("ikg-metadata")

TEMPLATE_PATTERN = re.compile(r"\{\{\s*([^}]+?)\s*\}\}")
DISTRIBUTED_BY_PATTERN = re.compile(r"DISTRIBUTED\s+BY\s*\([^;]*\)", re.IGNORECASE)
DO_BLOCK_PATTERN = re.compile(
    r"DO\s+\$(?P<tag>[A-Za-z0-9_]*)\$(.*?)\$(?P=tag)\$\s*;",
    re.IGNORECASE | re.DOTALL,
)
SKIP_PREFIXES = ("ALTER", "GRANT")


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def get_identifier_name(identifier: Optional[object]) -> Optional[str]:
    if identifier is None:
        return None
    if hasattr(identifier, "name"):
        return identifier.name  # type: ignore[attr-defined]
    return str(identifier)


def get_table_name(table_expr: Optional[exp.Table]) -> Optional[str]:
    if not isinstance(table_expr, exp.Table):
        return None
    return get_identifier_name(table_expr.this)


def get_table_schema(table_expr: Optional[exp.Table]) -> Optional[str]:
    if not isinstance(table_expr, exp.Table) or not table_expr.db:
        return None
    return get_identifier_name(table_expr.db)


@dataclass
class Settings:
    gitlab_url: str = os.getenv("GITLAB_URL", "https://devcloud.ubs.net")
    project_id: str = os.getenv("PROJECT_ID", "")
    branch: str = os.getenv("BRANCH", "main")
    sql_folder: str = os.getenv("SQL_FOLDER", "dags/ikg/scripts/sql")
    exclude_folders: List[str] = field(
        default_factory=lambda: os.getenv("EXCLUDE_FOLDER", "").split()
    )
    private_token: Optional[str] = os.getenv("PRIVATE_TOKEN")
    output_dir: Path = Path(os.getenv("OUTPUT_DIR") or Path.cwd())
    output_file_template: str = os.getenv(
        "OUTPUT_FILE", "ikg_metadata_<date>_<timestamp>.xlsx"
    )
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    greenplum_host: str = os.getenv("GREENPLUM_HOST", "localhost")
    greenplum_port: int = int(os.getenv("GREENPLUM_PORT", "5432"))
    greenplum_db: str = os.getenv("GREENPLUM_DB", "")
    greenplum_user: str = os.getenv("GREENPLUM_USER", "")
    greenplum_password: Optional[str] = os.getenv("GREENPLUM_PASSWORD")
    greenplum_schemas: List[str] = field(
        default_factory=lambda: [
            schema.strip()
            for schema in os.getenv(
                "GREENPLUM_SCHEMA", "core_wma_shared,core_model,core_ikg,core_etl"
            ).split(",")
            if schema.strip()
        ]
    )
    target_table: str = os.getenv(
        "GREENPLUM_TARGET_TABLE",
        "sandbox_prj_smart_insights.ikg_metadata_auto_refresh",
    )
    table_owner: str = os.getenv(
        "GREENPLUM_TABLE_OWNER", "erd_gpdbprj_smart_insights"
    )
    table_reader_role: str = os.getenv(
        "GREENPLUM_TABLE_READER_ROLE", "erd_gpdb_prj_smart_insights_ro"
    )
    files_to_parse: str = os.getenv("FILES_TO_PARSE", "ALL")

    @property
    def output_path(self) -> Path:
        today = _now_utc()
        date_str = today.strftime("%Y%m%d")
        time_str = today.strftime("%H%M%S")
        filename = self.output_file_template.strip()
        if "<date>" in filename or "<timestamp>" in filename:
            filename = (
                filename.replace("<date>", date_str)
                .replace("<timestamp>", time_str)
                .strip()
            )
        else:
            path_obj = Path(filename)
            suffix = path_obj.suffix or ".xlsx"
            stem = path_obj.stem or "ikg_metadata"
            filename = f"{stem}_{date_str}_{time_str}{suffix}"
        path = (self.output_dir / filename).resolve()
        if path.suffix.lower() != ".xlsx":
            path = path.with_suffix(".xlsx")
        return path

    @classmethod
    def from_args(cls) -> "Settings":
        parser = argparse.ArgumentParser(
            description="Extract IKG metadata from GitLab SQL assets."
        )
        parser.add_argument("--project-id", help="GitLab project path or numeric id")
        parser.add_argument("--branch", help="Git branch/tag to read from")
        parser.add_argument("--sql-folder", help="Folder containing SQL files")
        parser.add_argument("--exclude", nargs="*", help="Subfolders to skip")
        parser.add_argument("--output-dir", help="Directory for Excel output")
        parser.add_argument(
            "--log-level",
            default=os.getenv("LOG_LEVEL", "INFO"),
            help="Logging level (DEBUG, INFO, ...)",
        )
        parser.add_argument("--prompt-secrets", action="store_true")
        args = parser.parse_args()

        settings = cls()
        if args.project_id:
            settings.project_id = args.project_id
        if args.branch:
            settings.branch = args.branch
        if args.sql_folder:
            settings.sql_folder = args.sql_folder
        if args.exclude:
            settings.exclude_folders = args.exclude
        if args.output_dir:
            settings.output_dir = Path(args.output_dir)
        if args.log_level:
            settings.log_level = args.log_level
        if settings.project_id == "":
            parser.error("PROJECT_ID must be provided via env or --project-id")

        if args.prompt_secrets:
            if not settings.private_token:
                settings.private_token = getpass.getpass("GitLab Private Token: ")
            if not settings.greenplum_password:
                settings.greenplum_password = getpass.getpass("Greenplum password: ")
        else:
            if not settings.private_token:
                raise SystemExit(
                    "Missing PRIVATE_TOKEN. Set env var or run with --prompt-secrets."
                )
            if not settings.greenplum_password:
                raise SystemExit(
                    "Missing GREENPLUM_PASSWORD. Set env var or run with --prompt-secrets."
                )

        if not settings.greenplum_db or not settings.greenplum_user:
            raise SystemExit(
                "GREENPLUM_DB and GREENPLUM_USER must be configured via env."
            )

        settings.exclude_folders = [folder.strip() for folder in settings.exclude_folders if folder.strip()]
        return settings


class TemplateNormalizer:
    """Replace Jinja-style templates with SQL-safe tokens and restore later."""

    def __init__(self) -> None:
        self._map: Dict[str, str] = {}

    def normalize(self, text: str) -> str:
        def repl(match: re.Match[str]) -> str:
            key = re.sub(r"\s+", "", match.group(1))
            placeholder = f"__TPL_{len(self._map)}__"
            self._map[placeholder] = f"{{{{{key}}}}}"
            return placeholder

        return TEMPLATE_PATTERN.sub(repl, text)

    def restore(self, text: str) -> str:
        restored = text
        for placeholder, original in self._map.items():
            restored = restored.replace(placeholder, original)
        return restored

    def restore_identifier(self, identifier: Optional[str]) -> Optional[str]:
        if identifier is None:
            return None
        return self._map.get(identifier, identifier)


def split_statements(sql_text: str) -> List[str]:
    """Split SQL text into statements using sqlglot tokenizer."""
    tokens = Tokenizer().tokenize(sql_text)
    statements: List[str] = []
    statement_start = 0
    for token in tokens:
        if token.token_type == TokenType.SEMICOLON:
            fragment = sql_text[statement_start : token.start].strip()
            if fragment:
                statements.append(fragment)
            statement_start = token.end + 1
    tail = sql_text[statement_start:].strip()
    if tail:
        statements.append(tail)
    return statements


def strip_greenplum_specifics(sql_text: str) -> str:
    """Remove constructs sqlglot cannot parse (e.g., DISTRIBUTED BY)."""
    return DISTRIBUTED_BY_PATTERN.sub("", sql_text)


def remove_do_blocks(sql_text: str) -> str:
    """Remove DO $$ ... $$ blocks which are not relevant for metadata extraction."""
    return DO_BLOCK_PATTERN.sub("", sql_text)


@dataclass
class SqlFile:
    path: str
    filename: str
    process: str
    content: str


class ErrorLogger:
    def __init__(self, output_dir: Path) -> None:
        self.path = (output_dir / "error.txt").resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def log_error(self, message: str) -> None:
        self._write("ERROR", message)

    def log_warning(self, message: str) -> None:
        self._write("WARNING", message)

    def _write(self, level: str, message: str) -> None:
        timestamp = _now_utc().isoformat()
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{timestamp}] [{level}] {message}\n")


class GitLabSqlFetcher:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = gitlab.Gitlab(
            settings.gitlab_url, private_token=settings.private_token
        )
        self._project = None

    @property
    def project(self):
        if self._project is None:
            self._project = self._client.projects.get(self.settings.project_id)
        return self._project

    def fetch_sql_files(self) -> List[SqlFile]:
        LOGGER.info("Fetching SQL file list from %s@%s", self.settings.project_id, self.settings.branch)
        tree = self.project.repository_tree(
            path=self.settings.sql_folder,
            ref=self.settings.branch,
            recursive=True,
            all=True,
        )
        sql_files: List[SqlFile] = []
        for node in tree:
            if node.get("type") != "blob":
                continue
            path = node["path"]
            if not path.lower().endswith(".sql"):
                continue
            if any(excl and f"/{excl}/" in f"/{path}/" for excl in self.settings.exclude_folders):
                LOGGER.debug("Skipping %s (excluded folder)", path)
                continue
            try:
                file_obj = self.project.files.get(
                    file_path=path, ref=self.settings.branch
                )
            except gl_exceptions.GitlabGetError as exc:
                LOGGER.warning("Failed to fetch %s: %s", path, exc)
                continue
            content = base64.b64decode(file_obj.content).decode("utf-8", errors="ignore")
            relative = Path(path).relative_to(self.settings.sql_folder)
            process = relative.parts[0] if len(relative.parts) > 1 else ""
            sql_files.append(
                SqlFile(
                    path=path,
                    filename=Path(path).name,
                    process=process,
                    content=content,
                )
            )
        LOGGER.info("Fetched %d SQL files", len(sql_files))
        return sql_files


@dataclass
class TableMetadata:
    table_name: str
    schemas: Dict[str, set]

    @property
    def columns(self) -> set:
        combined: set = set()
        for cols in self.schemas.values():
            combined.update(cols)
        return combined


class GreenplumMetadataResolver:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._conn: Optional[psycopg2.extensions.connection] = None
        self._table_cache: Dict[str, TableMetadata] = {}

    def _connection(self) -> psycopg2.extensions.connection:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(
                host=self.settings.greenplum_host,
                port=self.settings.greenplum_port,
                dbname=self.settings.greenplum_db,
                user=self.settings.greenplum_user,
                password=self.settings.greenplum_password,
            )
            self._conn.autocommit = False
        return self._conn

    def ensure_tables(self, table_names: Iterable[str]) -> None:
        missing = [
            name.lower()
            for name in table_names
            if name and name.lower() not in self._table_cache
        ]
        if not missing:
            return
        LOGGER.debug("Loading metadata for tables: %s", ", ".join(sorted(set(missing))))
        query = """
            SELECT table_schema, table_name, column_name
            FROM information_schema.columns
            WHERE table_schema = ANY(%s)
              AND table_name = ANY(%s)
        """
        conn = self._connection()
        with conn.cursor() as cur:
            cur.execute(query, (self.settings.greenplum_schemas, list(set(missing))))
            rows = cur.fetchall()
        grouped: Dict[str, Dict[str, set]] = {}
        for schema, table_name, column_name in rows:
            grouped.setdefault(table_name.lower(), {}).setdefault(schema, set()).add(
                column_name.lower()
            )
        for name in missing:
            metadata = TableMetadata(table_name=name, schemas=grouped.get(name, {}))
            self._table_cache[name] = metadata

    def get_metadata(self, table_name: str) -> TableMetadata:
        self.ensure_tables([table_name])
        return self._table_cache.get(table_name.lower(), TableMetadata(table_name, {}))

    def build_schema_dict(
        self, table_refs: Sequence["TableRef"]
    ) -> Dict[str, Dict[str, Dict[str, str]]]:
        schema_dict: Dict[str, Dict[str, Dict[str, str]]] = {}
        for ref in table_refs:
            metadata = self.get_metadata(ref.name)
            if not metadata.columns:
                continue
            columns_map = {col: "text" for col in metadata.columns}
            schema_key = ref.schema_token or "default"
            schema_dict.setdefault(schema_key, {})[ref.name] = columns_map
            if not ref.schema_token:
                continue
            if ref.schema_token not in metadata.schemas:
                for schema_name in metadata.schemas:
                    schema_dict.setdefault(schema_name, {})[ref.name] = columns_map
        return schema_dict

    def close(self) -> None:
        if self._conn and not self._conn.closed:
            self._conn.close()

    def reset_table(self, dataframe: pd.DataFrame) -> None:
        conn = self._connection()
        columns_sql = """
            filename TEXT,
            filepath TEXT,
            process TEXT,
            target_table TEXT,
            source_schema TEXT,
            source_table TEXT,
            source_column TEXT,
            column_alias TEXT,
            logic TEXT,
            sql_operation TEXT
        """
        drop_sql = f"DROP TABLE IF EXISTS {self.settings.target_table};"
        create_sql = (
            f"CREATE TABLE {self.settings.target_table} ({columns_sql}) DISTRIBUTED RANDOMLY;"
        )
        alter_sql = f"ALTER TABLE {self.settings.target_table} OWNER TO {self.settings.table_owner};"
        grant_sql = f"GRANT SELECT ON {self.settings.target_table} TO {self.settings.table_reader_role};"
        insert_sql = f"""
            INSERT INTO {self.settings.target_table} (
                filename, filepath, process, target_table,
                source_schema, source_table, source_column,
                column_alias, logic, sql_operation
            ) VALUES %s
        """
        records = dataframe.fillna("").to_records(index=False)
        values = [tuple(row) for row in records]
        LOGGER.info("Loading %d rows into %s", len(values), self.settings.target_table)
        with conn.cursor() as cur:
            cur.execute(drop_sql)
            cur.execute(create_sql)
            if values:
                execute_values(cur, insert_sql, values, page_size=1000)
            cur.execute(alter_sql)
            cur.execute(grant_sql)
        conn.commit()


@dataclass
class TableRef:
    alias: str
    name: str
    schema_token: Optional[str]
    display_schema: Optional[str]


def collect_table_refs(expression: exp.Expression, normalizer: TemplateNormalizer) -> List[TableRef]:
    refs: List[TableRef] = []
    for table in expression.find_all(exp.Table):
        alias = table.alias_or_name
        if not alias:
            continue
        table_name_value = get_identifier_name(table.this)
        if not table_name_value:
            continue
        schema_token = get_identifier_name(table.db) if table.db else None
        refs.append(
            TableRef(
                alias=alias.lower(),
                name=table_name_value,
                schema_token=schema_token,
                display_schema=normalizer.restore_identifier(schema_token),
            )
        )
    return refs


def iter_lineage_leaves(node, visited: Optional[set] = None) -> Iterator:
    visited = visited or set()
    node_id = id(node)
    if node_id in visited:
        return
    visited.add(node_id)
    downstream = getattr(node, "downstream", None)
    if not downstream:
        yield node
        return
    for child in downstream:
        yield from iter_lineage_leaves(child, visited)


class SqlMetadataExtractor:
    def __init__(
        self,
        metadata_resolver: GreenplumMetadataResolver,
        settings: Settings,
    ) -> None:
        self.metadata_resolver = metadata_resolver
        self.settings = settings

    def process_file(
        self,
        sql_file: SqlFile,
        run_timestamp: datetime,
        error_logger: Optional[ErrorLogger] = None,
    ) -> List[Dict[str, object]]:
        normalizer = TemplateNormalizer()
        normalized = normalizer.normalize(sql_file.content)
        cleaned = remove_do_blocks(normalized)
        statements = split_statements(cleaned)
        rows: List[Dict[str, object]] = []
        sources_map: Dict[str, exp.Expression] = {}
        drop_cache: Dict[str, str] = {}
        for raw_statement in statements:
            stripped = raw_statement.lstrip()
            if not stripped:
                continue
            keyword = stripped.split(None, 1)[0].upper()
            if keyword.startswith(SKIP_PREFIXES):
                LOGGER.debug("Skipping %s statement in %s", keyword, sql_file.path)
                continue
            sanitized_statement = strip_greenplum_specifics(raw_statement)
            if not sanitized_statement.strip():
                continue
            try:
                expr = parse_one(sanitized_statement, read="postgres")
            except (ParseError, SqlglotError) as exc:
                message = f"{sql_file.path}: {exc}"
                LOGGER.error("Failed to parse %s: %s", sql_file.path, exc)
                if error_logger is not None:
                    error_logger.log_error(message)
                continue
            logic_text = normalizer.restore(raw_statement)
            if isinstance(expr, exp.Drop) and isinstance(expr.this, exp.Table):
                schema_name = (get_table_schema(expr.this) or "").lower()
                table_id = get_table_name(expr.this) or ""
                table_key = f"{schema_name}.{table_id.lower()}"
                drop_cache[table_key] = logic_text
                continue
            if isinstance(expr, exp.Create) and isinstance(expr.this, exp.Table):
                table_schema_name = get_table_schema(expr.this)
                created_table_name = get_table_name(expr.this)
                if not created_table_name:
                    continue
                table_key = f"{(table_schema_name or '').lower()}.{created_table_name.lower()}"
                drop_sql = drop_cache.pop(table_key, "")
                combined_logic = f"{drop_sql}\n{logic_text}".strip()
                rows.extend(
                    self._handle_select_target(
                        sql_file,
                        expr.expression,
                        created_table_name,
                        combined_logic,
                        run_timestamp,
                        normalizer,
                        error_logger,
                        sources_map,
                    )
                )
                if isinstance(expr.expression, exp.Expression):
                    sources_map[created_table_name.lower()] = expr.expression
            elif isinstance(expr, exp.Insert):
                target_table_expr: Optional[exp.Table] = None
                if isinstance(expr.this, exp.Schema) and isinstance(expr.this.this, exp.Table):
                    target_table_expr = expr.this.this
                elif isinstance(expr.this, exp.Table):
                    target_table_expr = expr.this
                target_table_name = get_table_name(target_table_expr) or "unknown"
                rows.extend(
                    self._handle_select_target(
                        sql_file,
                        expr.expression,
                        target_table_name,
                        logic_text,
                        run_timestamp,
                        normalizer,
                        error_logger,
                        sources_map,
                    )
                )
        return rows

    def _handle_select_target(
        self,
        sql_file: SqlFile,
        query: Optional[exp.Expression],
        target_table: str,
        logic_text: str,
        run_timestamp: datetime,
        normalizer: TemplateNormalizer,
        error_logger: Optional[ErrorLogger],
        sources_map: Dict[str, exp.Expression],
    ) -> List[Dict[str, object]]:
        if not isinstance(query, exp.Expression):
            LOGGER.debug("Skipping %s (no SELECT body)", target_table)
            return []
        table_refs = collect_table_refs(query, normalizer)
        self.metadata_resolver.ensure_tables([ref.name for ref in table_refs])
        schema_dict = self.metadata_resolver.build_schema_dict(table_refs)
        alias_map = {ref.alias: ref for ref in table_refs}
        row_accumulator: Dict[
            Tuple[str, str, str, str, str, str, str, str],
            Dict[str, object],
        ] = {}

        def base_row(
            logic: str,
            column_alias: str = "",
            source_schema: str = "",
            source_table: str = "",
            source_column: str = "",
        ) -> Dict[str, object]:
            return {
                "filename": sql_file.filename,
                "filepath": sql_file.path,
                "process": sql_file.process,
                "target_table": target_table,
                "source_schema": source_schema,
                "source_table": source_table,
                "source_column": source_column,
                "column_alias": column_alias,
                "logic": logic,
            }

        def add_row(row: Dict[str, object], operation: Optional[str] = None) -> bool:
            key = (
                row["filename"],
                row["filepath"],
                row["process"],
                row["target_table"],
                row["source_schema"] or "",
                row["source_table"] or "",
                row["source_column"] or "",
                row["column_alias"] or "",
            )
            entry = row_accumulator.get(key)
            created = False
            if not entry:
                entry = {**row, "sql_operations": []}
                row_accumulator[key] = entry
                created = True
            else:
                incoming_logic = row.get("logic")
                if incoming_logic:
                    if not entry.get("logic"):
                        entry["logic"] = incoming_logic
                    elif operation and operation.upper() == "SELECT":
                        entry["logic"] = incoming_logic
            if operation:
                op = operation.upper()
                if op not in entry["sql_operations"]:
                    entry["sql_operations"].append(op)
            return created

        add_row(base_row(logic_text))
        select_expr = None
        for candidate in query.walk():
            if isinstance(candidate, exp.Select):
                select_expr = candidate
                break
        if not select_expr:
            message = f"No SELECT found for target table {target_table} in {sql_file.path}"
            LOGGER.warning(message)
            if error_logger:
                error_logger.log_warning(message)
            return self._finalize_rows(row_accumulator)
        sources_override = {
            name: source for name, source in sources_map.items()
        }
        for projection in select_expr.expressions:
            logic_sql = normalizer.restore(projection.sql(dialect="postgres"))
            column_alias = projection.alias_or_name or ""
            if self._is_star_projection(projection):
                self._append_star_rows(
                    projection,
                    logic_sql,
                    alias_map,
                    add_row,
                    base_row,
                    sources_map,
                )
                continue
            if self._is_static_projection(projection):
                add_row(
                    base_row(
                        logic=logic_sql,
                        column_alias=column_alias,
                    ),
                    "SELECT",
                )
                continue
            added_in_lineage = False
            try:
                lineage_node = lineage(
                    column_alias,
                    query,
                    schema=schema_dict,
                    sources=sources_override,
                    dialect="postgres",
                    trim_selects=False,
                )
            except SqlglotError as exc:
                message = f"Lineage failed for {target_table}.{column_alias} in {sql_file.path}: {exc}"
                LOGGER.warning(message)
                if error_logger:
                    error_logger.log_warning(message)
                added_in_lineage |= add_row(
                    base_row(
                        logic=logic_sql,
                        column_alias=column_alias,
                    ),
                    "SELECT",
                )
                self._collect_projection_columns(
                    projection,
                    alias_map,
                    add_row,
                    base_row,
                    logic_sql,
                    column_alias,
                )
                continue
            for leaf in iter_lineage_leaves(lineage_node):
                table_part, column_part = self._split_leaf_name(leaf.name)
                table_ref = alias_map.get(table_part.lower()) if table_part else None
                source_schema = (
                    table_ref.display_schema
                    if table_ref and table_ref.display_schema
                    else ""
                )
                source_table = table_ref.name if table_ref else table_part
                if not table_ref:
                    inferred = self._infer_table_from_column(column_part, alias_map)
                    if inferred:
                        table_ref = inferred
                        source_table = inferred.name
                        source_schema = inferred.display_schema or source_schema
                alias_value = column_alias if column_alias and column_alias != column_part else ""
                added_in_lineage |= add_row(
                    base_row(
                        logic=logic_sql,
                        column_alias=alias_value,
                        source_schema=source_schema,
                        source_table=source_table,
                        source_column=column_part,
                    ),
                    "SELECT",
                )
            if not added_in_lineage:
                self._collect_projection_columns(
                    projection,
                    alias_map,
                    add_row,
                    base_row,
                    logic_sql,
                    column_alias,
                )

        self._collect_clause_columns(
            select_expr.args.get("where"),
            "WHERE",
            alias_map,
            normalizer,
            add_row,
            base_row,
        )
        self._collect_clause_columns(
            select_expr.args.get("having"),
            "HAVING",
            alias_map,
            normalizer,
            add_row,
            base_row,
        )
        for join in select_expr.args.get("joins") or []:
            operation = self._join_operation_label(join)
            self._collect_clause_columns(
                join.args.get("on"),
                operation,
                alias_map,
                normalizer,
                add_row,
                base_row,
            )
            using_clause = join.args.get("using")
            if using_clause:
                self._collect_using_columns(
                    using_clause,
                    operation,
                    alias_map,
                    add_row,
                    base_row,
                    normalizer,
                )

        for subquery in query.find_all(exp.Subquery):
            sub_alias_map = {
                ref.alias: ref for ref in collect_table_refs(subquery, normalizer)
            }
            self._collect_clause_columns(
                subquery,
                "SUB QUERY",
                sub_alias_map,
                normalizer,
                add_row,
                base_row,
                column_alias="",
            )

        return self._finalize_rows(row_accumulator)

    def _split_leaf_name(self, name: str) -> Tuple[str, str]:
        if "." in name:
            alias, column = name.split(".", 1)
            return alias, column
        return "", name

    def _infer_table_from_column(
        self, column: str, alias_map: Dict[str, TableRef]
    ) -> Optional[TableRef]:
        matches = []
        for ref in alias_map.values():
            metadata = self.metadata_resolver.get_metadata(ref.name)
            if column.lower() in metadata.columns:
                matches.append(ref)
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            LOGGER.debug(
                "Ambiguous column %s across tables %s",
                column,
                ", ".join(ref.name for ref in matches),
            )
        return None

    def _is_star_projection(self, projection: exp.Expression) -> bool:
        if isinstance(projection, exp.Column) and getattr(projection, "is_star", False):
            return True
        return isinstance(projection, exp.Star)

    def _get_star_table_alias(self, projection: exp.Expression) -> Optional[str]:
        if isinstance(projection, exp.Column) and getattr(projection, "is_star", False):
            return projection.table
        if isinstance(projection, exp.Star):
            identifier = projection.args.get("this")
            if isinstance(identifier, exp.Identifier):
                return identifier.name
            if isinstance(identifier, str):
                return identifier
        return None

    def _is_static_projection(self, projection: exp.Expression) -> bool:
        return not any(projection.find_all(exp.Column))

    def _append_star_rows(
        self,
        projection: exp.Expression,
        logic_sql: str,
        alias_map: Dict[str, TableRef],
        add_row: Callable[[Dict[str, object], Optional[str]], None],
        base_row_fn: Callable[..., Dict[str, object]],
        sources_map: Dict[str, exp.Expression],
    ) -> None:
        table_alias = self._get_star_table_alias(projection)
        target_refs: List[TableRef] = []
        if table_alias:
            ref = alias_map.get(table_alias.lower())
            if ref:
                target_refs.append(ref)
        else:
            target_refs.extend(alias_map.values())

        for ref in target_refs:
            for column_name, source_schema in self._get_columns_for_table_ref(ref, sources_map):
                add_row(
                    base_row_fn(
                        logic=logic_sql,
                        source_schema=source_schema or (ref.display_schema or ""),
                        source_table=ref.name,
                        source_column=column_name,
                    ),
                    "SELECT",
                )

    def _get_columns_for_table_ref(
        self, table_ref: TableRef, sources_map: Dict[str, exp.Expression]
    ) -> List[Tuple[str, str]]:
        if not table_ref:
            return []
        columns = self._columns_from_sources_map(table_ref.name, sources_map)
        if columns:
            return [(column, table_ref.display_schema or "") for column in columns]
        metadata = self.metadata_resolver.get_metadata(table_ref.name)
        if metadata.columns:
            schema = (
                table_ref.display_schema
                or next(iter(metadata.schemas.keys()), "")
            )
            return [(column, schema) for column in sorted(metadata.columns)]
        return []

    def _columns_from_sources_map(
        self, table_name: str, sources_map: Dict[str, exp.Expression]
    ) -> List[str]:
        expression = sources_map.get(table_name.lower())
        if not isinstance(expression, exp.Select):
            return []
        names: List[str] = []
        for item in expression.expressions:
            if self._is_star_projection(item):
                continue
            alias = item.alias_or_name
            if alias:
                names.append(alias)
                continue
            column = next(item.find_all(exp.Column), None)
            if column:
                names.append(column.name)
        return names

    def _collect_projection_columns(
        self,
        projection: exp.Expression,
        alias_map: Dict[str, TableRef],
        add_row: Callable[[Dict[str, object], Optional[str]], None],
        base_row_fn: Callable[..., Dict[str, object]],
        logic_sql: str,
        column_alias: str,
    ) -> None:
        columns = self._collect_columns_from_expression(projection)
        for column in columns:
            source_schema, source_table = self._resolve_column_source(column, alias_map)
            column_name = column.name
            alias_value = column_alias if column_alias and column_alias != column_name else ""
            if not source_table:
                continue
            add_row(
                base_row_fn(
                    logic=logic_sql,
                    column_alias=alias_value,
                    source_schema=source_schema,
                    source_table=source_table,
                    source_column=column_name,
                ),
                "SELECT",
            )

    def _collect_clause_columns(
        self,
        expression: Optional[exp.Expression],
        operation: str,
        alias_map: Dict[str, TableRef],
        normalizer: TemplateNormalizer,
        add_row: Callable[[Dict[str, object], Optional[str]], None],
        base_row_fn: Callable[..., Dict[str, object]],
        column_alias: str = "",
    ) -> None:
        if not expression:
            return
        logic_sql = normalizer.restore(expression.sql(dialect="postgres"))
        for column in self._collect_columns_from_expression(expression):
            source_schema, source_table = self._resolve_column_source(column, alias_map)
            if not source_table:
                continue
            add_row(
                base_row_fn(
                    logic=logic_sql,
                    column_alias=column_alias,
                    source_schema=source_schema,
                    source_table=source_table,
                    source_column=column.name,
                ),
                operation,
            )

    def _collect_using_columns(
        self,
        using_clause: exp.Expression,
        operation: str,
        alias_map: Dict[str, TableRef],
        add_row: Callable[[Dict[str, object], Optional[str]], None],
        base_row_fn: Callable[..., Dict[str, object]],
        normalizer: TemplateNormalizer,
    ) -> None:
        if not using_clause:
            return
        logic_sql = normalizer.restore(using_clause.sql(dialect="postgres"))
        identifiers = [
            identifier.name
            for identifier in getattr(using_clause, "expressions", []) or []
            if isinstance(identifier, exp.Identifier)
        ]
        for column_name in identifiers:
            table_ref = self._infer_table_from_column(column_name, alias_map)
            source_schema = table_ref.display_schema if table_ref and table_ref.display_schema else ""
            source_table = table_ref.name if table_ref else ""
            if not source_table:
                continue
            add_row(
                base_row_fn(
                    logic=logic_sql,
                    source_schema=source_schema,
                    source_table=source_table,
                    source_column=column_name,
                ),
                operation,
            )

    def _collect_columns_from_expression(
        self, expression: exp.Expression
    ) -> List[exp.Column]:
        columns: List[exp.Column] = []
        seen: set = set()
        for column in expression.find_all(exp.Column):
            identifier = (column.table or "", column.name)
            if identifier in seen:
                continue
            seen.add(identifier)
            columns.append(column)
        return columns

    def _resolve_column_source(
        self, column: exp.Column, alias_map: Dict[str, TableRef]
    ) -> Tuple[str, str]:
        table_alias = column.table
        if table_alias:
            table_ref = alias_map.get(table_alias.lower())
            if table_ref:
                return table_ref.display_schema or "", table_ref.name
            return "", table_alias
        inferred = self._infer_table_from_column(column.name, alias_map)
        if inferred:
            return inferred.display_schema or "", inferred.name
        return "", ""

    def _join_operation_label(self, join: exp.Join) -> str:
        kind = (join.args.get("kind") or "").upper()
        if not kind:
            return "JOIN"
        if "CROSS" in kind:
            return "CROSS JOIN"
        if "FULL" in kind and "OUTER" in kind:
            return "FULL OUTER JOIN"
        if "FULL" in kind:
            return "FULL JOIN"
        if "LEFT" in kind and "OUTER" in kind:
            return "LEFT OUTER JOIN"
        if "RIGHT" in kind and "OUTER" in kind:
            return "RIGHT OUTER JOIN"
        if "LEFT" in kind:
            return "LEFT JOIN"
        if "RIGHT" in kind:
            return "RIGHT JOIN"
        if "INNER" in kind:
            return "INNER JOIN"
        return f"{kind} JOIN"

    def _finalize_rows(
        self,
        row_accumulator: Dict[
            Tuple[str, str, str, str, str, str, str, str],
            Dict[str, object],
        ],
    ) -> List[Dict[str, object]]:
        finalized: List[Dict[str, object]] = []
        for entry in row_accumulator.values():
            operations = entry.pop("sql_operations", [])
            entry["sql_operation"] = ", ".join(operations)
            finalized.append(entry)
        return finalized


def write_excel(dataframe: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() != ".xlsx":
        path = path.with_suffix(".xlsx")
    safe_df = _prepare_dataframe_for_excel(dataframe)
    LOGGER.info("Writing Excel output to %s using openpyxl", path)
    safe_df.to_excel(path, index=False, engine="openpyxl")


def _prepare_dataframe_for_excel(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe
    safe_df = dataframe.copy()
    datetime_cols = safe_df.select_dtypes(include=["datetimetz"]).columns
    for col in datetime_cols:
        safe_df[col] = safe_df[col].dt.tz_convert("UTC").dt.tz_localize(None)
    object_cols = [col for col in safe_df.columns if safe_df[col].dtype == "object"]
    for col in object_cols:
        safe_df[col] = safe_df[col].apply(_strip_timezone_from_value)
    return safe_df


def _strip_timezone_from_value(value):
    if isinstance(value, datetime) and value.tzinfo:
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def _parse_files_limit(value: str) -> Optional[int]:
    if not value:
        return None
    cleaned = value.strip()
    if not cleaned or cleaned.upper() == "ALL":
        return None
    try:
        limit = int(cleaned)
        if limit > 0:
            return limit
    except ValueError:
        LOGGER.warning("Invalid files_to_parse value '%s'. Processing all files.", value)
    return None


def run_pipeline(settings: Settings) -> Path:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    LOGGER.info("Starting metadata extraction")
    fetcher = GitLabSqlFetcher(settings)
    sql_files = fetcher.fetch_sql_files()
    resolver = GreenplumMetadataResolver(settings)
    extractor = SqlMetadataExtractor(resolver, settings)
    error_logger = ErrorLogger(settings.output_dir)
    run_ts = _now_utc()
    output_path = settings.output_path
    all_rows: List[Dict[str, object]] = []
    max_files = _parse_files_limit(settings.files_to_parse)
    processed_files = 0
    for sql_file in sql_files:
        LOGGER.info("Parsing %s", sql_file.path)
        try:
            rows = extractor.process_file(sql_file, run_ts, error_logger=error_logger)
        except Exception as exc:  # noqa: BLE001
            message = f"{sql_file.path}: {exc}"
            LOGGER.exception("Unhandled error while parsing %s", sql_file.path)
            error_logger.log_error(message)
            continue
        all_rows.extend(rows)
        dataframe = pd.DataFrame(all_rows)
        write_excel(dataframe, output_path)
        processed_files += 1
        if max_files is not None and processed_files >= max_files:
            LOGGER.info(
                "Reached files_to_parse limit (%d). Stopping further processing.",
                max_files,
            )
            break
    if not all_rows:
        LOGGER.warning("No metadata rows generated.")
    dataframe = pd.DataFrame(all_rows)
    write_excel(dataframe, output_path)
    resolver.reset_table(dataframe)
    resolver.close()
    LOGGER.info("Pipeline complete")
    return output_path


def main() -> None:
    settings = Settings.from_args()
    output_path = run_pipeline(settings)
    print(f"Metadata exported to {output_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted by user")
        sys.exit(1)
