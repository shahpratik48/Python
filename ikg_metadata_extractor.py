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
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import gitlab
import pandas as pd
import psycopg2
from gitlab import exceptions as gl_exceptions
from psycopg2.extras import execute_values
from sqlglot import exp, parse
from sqlglot.errors import ParseError, SqlglotError
from sqlglot.lineage import lineage
from sqlglot.tokens import TokenType, Tokenizer

LOGGER = logging.getLogger("ikg-metadata")

TEMPLATE_PATTERN = re.compile(r"\{\{\s*([^}]+?)\s*\}\}")
DISTRIBUTED_BY_PATTERN = re.compile(r"DISTRIBUTED\s+BY\s*\([^;]*\)", re.IGNORECASE)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


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
        "OUTPUT_FILE", "ikg_metadata_<date>_<timestamp>.xls"
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

    @property
    def output_path(self) -> Path:
        today = _now_utc()
        filename = (
            self.output_file_template.replace("<date>", today.strftime("%Y%m%d"))
            .replace("<timestamp>", today.strftime("%H%M%S"))
            .strip()
        )
        return (self.output_dir / filename).resolve()

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


@dataclass
class SqlFile:
    path: str
    filename: str
    process: str
    content: str


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
            current_timestamp TIMESTAMPTZ
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
                column_alias, logic, current_timestamp
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
        schema_token = table.db.name if table.db else None
        refs.append(
            TableRef(
                alias=alias.lower(),
                name=table.this.name,
                schema_token=schema_token,
                display_schema=normalizer.restore_identifier(schema_token),
            )
        )
    return refs


def iter_lineage_leaves(node) -> Iterator:
    if not getattr(node, "downstream", None):
        yield node
        return
    for child in node.downstream:
        yield from iter_lineage_leaves(child)


class SqlMetadataExtractor:
    def __init__(
        self,
        metadata_resolver: GreenplumMetadataResolver,
        settings: Settings,
    ) -> None:
        self.metadata_resolver = metadata_resolver
        self.settings = settings
        self.sources_map: Dict[str, exp.Expression] = {}

    def process_file(self, sql_file: SqlFile, run_timestamp: datetime) -> List[Dict[str, object]]:
        normalizer = TemplateNormalizer()
        normalized = normalizer.normalize(sql_file.content)
        statements = split_statements(normalized)
        parse_ready = strip_greenplum_specifics(normalized)
        try:
            expressions = parse(parse_ready, read="postgres")
        except ParseError as exc:
            LOGGER.error("Failed to parse %s: %s", sql_file.path, exc)
            return []
        if len(expressions) != len(statements):
            LOGGER.warning(
                "Statement count mismatch in %s (parsed=%d vs split=%d)",
                sql_file.path,
                len(expressions),
                len(statements),
            )
        rows: List[Dict[str, object]] = []
        drop_cache: Dict[str, str] = {}
        for idx, expr in enumerate(expressions):
            logic_text = normalizer.restore(
                statements[idx] if idx < len(statements) else expr.sql(dialect="postgres")
            )
            if isinstance(expr, exp.Drop) and isinstance(expr.this, exp.Table):
                schema_key = expr.this.db.name.lower() if expr.this.db else ""
                table_key = f"{schema_key}.{expr.this.this.name.lower()}"
                drop_cache[table_key] = logic_text
                continue
            if isinstance(expr, exp.Create) and isinstance(expr.this, exp.Table):
                table_schema = expr.this.db.name if expr.this.db else None
                table_name = expr.this.this.name
                table_key = f"{(table_schema or '').lower()}.{table_name.lower()}"
                drop_sql = drop_cache.pop(table_key, "")
                combined_logic = f"{drop_sql}\n{logic_text}".strip()
                rows.extend(
                    self._handle_select_target(
                        sql_file,
                        expr.expression,
                        table_name,
                        combined_logic,
                        run_timestamp,
                        normalizer,
                    )
                )
                if isinstance(expr.expression, exp.Expression):
                    self.sources_map[table_name.lower()] = expr.expression
            elif isinstance(expr, exp.Insert):
                target_table = expr.this.this.this.name if isinstance(expr.this, exp.Schema) else "unknown"
                rows.extend(
                    self._handle_select_target(
                        sql_file,
                        expr.expression,
                        target_table,
                        logic_text,
                        run_timestamp,
                        normalizer,
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
    ) -> List[Dict[str, object]]:
        if not isinstance(query, exp.Expression):
            LOGGER.debug("Skipping %s (no SELECT body)", target_table)
            return []
        table_refs = collect_table_refs(query, normalizer)
        self.metadata_resolver.ensure_tables([ref.name for ref in table_refs])
        schema_dict = self.metadata_resolver.build_schema_dict(table_refs)
        alias_map = {ref.alias: ref for ref in table_refs}
        seen: set = set()
        rows: List[Dict[str, object]] = [
            {
                "filename": sql_file.filename,
                "filepath": sql_file.path,
                "process": sql_file.process,
                "target_table": target_table,
                "source_schema": "",
                "source_table": "",
                "source_column": "",
                "column_alias": "",
                "logic": logic_text,
                "current_timestamp": run_timestamp,
            }
        ]
        select_expr = None
        for candidate in query.walk():
            if isinstance(candidate, exp.Select):
                select_expr = candidate
                break
        if not select_expr:
            LOGGER.warning("No SELECT found for %s in %s", target_table, sql_file.path)
            return rows
        sources_override = {
            name: source
            for name, source in self.sources_map.items()
        }
        for projection in select_expr.expressions:
            column_alias = projection.alias_or_name
            if not column_alias:
                continue
            logic_sql = self.normalizer.restore(projection.sql(dialect="postgres"))
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
                LOGGER.warning(
                    "Lineage failed for %s.%s (%s): %s",
                    target_table,
                    column_alias,
                    sql_file.path,
                    exc,
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
                row_key = (
                    target_table.lower(),
                    logic_sql,
                    source_schema or "",
                    source_table or "",
                    column_part or "",
                    column_alias or "",
                )
                if row_key in seen:
                    continue
                seen.add(row_key)
                rows.append(
                    {
                        "filename": sql_file.filename,
                        "filepath": sql_file.path,
                        "process": sql_file.process,
                        "target_table": target_table,
                        "source_schema": source_schema,
                        "source_table": source_table,
                        "source_column": column_part,
                        "column_alias": column_alias if column_alias != column_part else "",
                        "logic": logic_sql,
                        "current_timestamp": run_timestamp,
                    }
                )
        return rows

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


def write_excel(dataframe: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Writing Excel output to %s", path)
    dataframe.to_excel(path, index=False, engine="xlwt")


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
    run_ts = _now_utc()
    all_rows: List[Dict[str, object]] = []
    for sql_file in sql_files:
        LOGGER.info("Parsing %s", sql_file.path)
        rows = extractor.process_file(sql_file, run_ts)
        all_rows.extend(rows)
    if not all_rows:
        LOGGER.warning("No metadata rows generated.")
    dataframe = pd.DataFrame(all_rows)
    output_path = settings.output_path
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
