#!/usr/bin/env python3
"""
IKG SQL lineage extractor.

Features
--------
* Connects to the UBS GitLab instance and downloads SQL files from a project/branch.
* Recursively walks the configured SQL folder (respecting optional exclude folders).
* Accepts a start table and follows downstream SQL files (based on table/file naming) to
  build table/column lineage using sqlglot with CTE support.
* Handles templated schema placeholders such as ``{{params.IKG_SCHEMA}}`` by normalizing
  them for parsing yet restoring the placeholders in the final outputs.
* Resolves missing table qualifiers for columns by consulting Greenplum metadata.
* Produces XLS, JSON, GEXF, and indented tree exports that capture target/source
  schema-table-column lineage with the transformation logic and SQL file path.

Usage
-----
```
python ikg_sql_lineage.py run \
    --gitlab-url https://devcloud.ubs.net \
    --project-id ubs/gwma/.../ikg-dags \
    --branch ikg-master \
    --sql-folder dags/ikg/scripts/sql \
    --exclude-folder "ikg_new fa_shhp_map" \
    --token <PRIVATE_TOKEN> \
    --start-table account_profile_curr_ikg \
    --greenplum-host greenplum-rdsp.zur.swissbank.com \
    --greenplum-db gprdsp \
    --greenplum-user ds_rdsp_dev \
    --greenplum-password <PASSWORD> \
    --greenplum-schema "core_wma_shared,core_model,core_ikg,core_etl"
```
"""

from __future__ import annotations

import base64
import datetime as dt
import json
import logging
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import gitlab
import networkx as nx
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from sqlglot import expressions as exp, parse
from sqlglot.optimizer import qualify
from tqdm import tqdm
import typer


LOGGER = logging.getLogger("ikg_sql_lineage")
APP = typer.Typer(no_args_is_help=True, add_completion=False)


PLACEHOLDER_PATTERN = re.compile(r"\{\{\s*([^}]+)\s*\}\}")
LINE_COMMENT_PATTERN = re.compile(r"--.*?$", re.MULTILINE)
BLOCK_COMMENT_PATTERN = re.compile(r"/\*.*?\*/", re.DOTALL)


def setup_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def strip_sql_comments(sql_text: str) -> str:
    without_block = BLOCK_COMMENT_PATTERN.sub(" ", sql_text)
    return LINE_COMMENT_PATTERN.sub(" ", without_block)


class PlaceholderNormalizer:
    """Replaces templated placeholders with SQL-safe tokens and can restore them later."""

    def __init__(self) -> None:
        self._mapping: Dict[str, str] = {}

    @staticmethod
    def _sanitize(value: str) -> str:
        sanitized = re.sub(r"[^0-9a-zA-Z_]+", "_", value.strip())
        return sanitized.upper().strip("_") or "PLACEHOLDER"

    def normalize(self, sql_text: str) -> str:
        def _replace(match: re.Match[str]) -> str:
            raw = match.group(0)
            inner = match.group(1)
            token = f"__PH_{len(self._mapping)}_{self._sanitize(inner)}__"
            self._mapping[token] = raw
            return token

        return PLACEHOLDER_PATTERN.sub(_replace, sql_text)

    def restore(self, text: str) -> str:
        restored = text
        for token, raw in self._mapping.items():
            restored = restored.replace(token, raw)
        return restored

    def restore_identifier(self, identifier: Optional[str]) -> Optional[str]:
        if identifier is None:
            return None
        return self.restore(identifier)


@dataclass
class TableRef:
    schema: Optional[str]
    table: Optional[str]
    alias: Optional[str]
    is_cte: bool = False
    is_temp: bool = False

    def label(self) -> str:
        schema_part = self.schema or ""
        base = self.table or ""
        return f"{schema_part}.{base}" if schema_part else base


@dataclass
class LineageRecord:
    target_schema: Optional[str]
    target_table: str
    target_column: str
    source_schema: Optional[str]
    source_table: Optional[str]
    source_column: Optional[str]
    logic: str
    sql_file: str


class GreenplumMetadata:
    """Lazy metadata fetcher that resolves column ownership per table."""

    def __init__(
        self,
        host: Optional[str],
        port: Optional[int],
        database: Optional[str],
        user: Optional[str],
        password: Optional[str],
        schemas: Sequence[str],
    ) -> None:
        self.host = host
        self.port = port or 5432
        self.database = database
        self.user = user
        self.password = password
        self.schemas = [schema.strip() for schema in schemas if schema.strip()]
        self._conn: Optional[psycopg2.extensions.connection] = None
        self._cache: Dict[Tuple[Optional[str], str], Set[str]] = {}

    @property
    def enabled(self) -> bool:
        return all([self.host, self.database, self.user, self.password]) and bool(self.schemas)

    def _connect(self) -> Optional[psycopg2.extensions.connection]:
        if not self.enabled:
            return None
        if self._conn and not self._conn.closed:
            return self._conn
        try:
            self._conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                dbname=self.database,
                user=self.user,
                password=self.password,
            )
            return self._conn
        except Exception as exc:
            LOGGER.warning("Failed to connect to Greenplum metadata: %s", exc)
            return None

    def ensure_connection(self) -> None:
        if not self.enabled or not self._connect():
            raise RuntimeError(
                "Unable to establish a connection to Greenplum metadata. Verify credentials and network access."
            )

    def close(self) -> None:
        if self._conn and not self._conn.closed:
            self._conn.close()
        self._conn = None

    def _fetch_columns(self, schema: Optional[str], table: str) -> Set[str]:
        key = (schema.lower() if schema else None, table.lower())
        if key in self._cache:
            return self._cache[key]

        conn = self._connect()
        if not conn:
            self._cache[key] = set()
            return self._cache[key]

        query: str
        params: Tuple
        if schema:
            query = """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = %s
                  AND table_name = %s
            """
            params = (schema, table)
        else:
            query = """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = ANY(%s)
                  AND table_name = %s
            """
            params = (self.schemas, table)

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)
                columns = {row["column_name"].lower() for row in cursor.fetchall()}
                self._cache[key] = columns
        except Exception as exc:
            LOGGER.warning(
                "Failed to query information_schema for %s.%s: %s",
                schema,
                table,
                exc,
            )
            self._cache[key] = set()

        return self._cache[key]

    def _schema_candidates(self, schema_name: Optional[str]) -> List[Optional[str]]:
        if schema_name and schema_name.startswith("{{"):
            return [None]
        if schema_name:
            return [schema_name]
        return list(self.schemas)

    def column_exists(
        self,
        schema: Optional[str],
        table: Optional[str],
        column: Optional[str],
    ) -> bool:
        if not self.enabled or not table or not column:
            return False
        column_lower = column.lower()
        for schema_candidate in self._schema_candidates(schema):
            columns = self._fetch_columns(schema_candidate, table)
            if column_lower in columns:
                return True
        return False

    def resolve(
        self,
        column: Optional[str],
        candidates: Sequence[TableRef],
        preferred: Optional[TableRef] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Return (schema, table) for the column across candidate tables."""
        if not self.enabled or not column:
            if preferred:
                return preferred.schema, preferred.table
            return None, None

        column_lower = column.lower()
        ordered: List[TableRef] = []
        if preferred:
            ordered.append(preferred)
        ordered.extend([t for t in candidates if t is not preferred])

        for table_ref in ordered:
            if table_ref.is_cte or table_ref.is_temp or not table_ref.table:
                continue
            for schema_candidate in self._schema_candidates(table_ref.schema):
                columns = self._fetch_columns(schema_candidate, table_ref.table)
                if column_lower in columns:
                    return schema_candidate or table_ref.schema, table_ref.table

        if preferred:
            return preferred.schema, preferred.table
        return None, None


class GitLabSQLFetcher:
    """Downloads SQL files from GitLab and indexes them by table name."""

    def __init__(
        self,
        gitlab_url: str,
        private_token: str,
        project_id: str,
        branch: str,
        sql_folder: str,
        exclude_folders: Sequence[str],
    ) -> None:
        self.gitlab_url = gitlab_url
        self.private_token = private_token
        self.project_id = project_id
        self.branch = branch
        self.sql_folder = sql_folder.strip("/")
        self.exclude_folders = {folder.strip("/").lower() for folder in exclude_folders if folder}
        self._gl = gitlab.Gitlab(self.gitlab_url, private_token=self.private_token)
        self._project = self._gl.projects.get(self.project_id)

    def _should_skip(self, path: str) -> bool:
        lowered = path.lower()
        return any(excluded in lowered for excluded in self.exclude_folders)

    def download(self, dest_dir: Path) -> "SQLRepository":
        dest_dir.mkdir(parents=True, exist_ok=True)
        tree = self._project.repository_tree(
            path=self.sql_folder,
            ref=self.branch,
            recursive=True,
            all=True,
        )
        sql_files = [item for item in tree if item["type"] == "blob" and item["path"].lower().endswith(".sql")]
        repository = SQLRepository(dest_dir, self.sql_folder)
        for item in tqdm(sql_files, desc="Downloading SQL files"):
            rel_path = item["path"]
            if self._should_skip(rel_path):
                continue
            file_obj = self._project.files.get(file_path=rel_path, ref=self.branch)
            decoded = base64.b64decode(file_obj.content).decode("utf-8", errors="ignore")
            repository.register(rel_path, decoded)
        return repository


class SQLRepository:
    """Local cache of SQL files keyed by table name."""

    def __init__(self, root_dir: Path, remote_root: str) -> None:
        self.root_dir = root_dir
        self.remote_root = remote_root
        self.table_to_paths: Dict[str, List[Path]] = defaultdict(list)
        self.path_to_text: Dict[Path, str] = {}

    def register(self, relative_path: str, content: str) -> None:
        path = self.root_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        self.path_to_text[path] = content
        table_name = path.stem.lower()
        self.table_to_paths[table_name].append(path)

    def get_paths_for_table(self, table: str) -> List[Path]:
        return self.table_to_paths.get(table.lower(), [])

    def get_text(self, path: Path) -> Optional[str]:
        return self.path_to_text.get(path)

    def find_preferred_path(self, table: str, preferred_subfolder: Optional[str] = None) -> Optional[Path]:
        candidates = self.get_paths_for_table(table)
        if not candidates:
            return None
        if preferred_subfolder:
            for candidate in candidates:
                if preferred_subfolder in str(candidate):
                    return candidate
        return candidates[0]


class SQLLineageParser:
    """Parses SQL files and produces column-level lineage records."""

    def __init__(
        self,
        repository: SQLRepository,
        metadata: GreenplumMetadata,
        start_table: str,
        preferred_start_subfolder: str = "ikg_create_profiles",
        dialect: str = "postgres",
    ) -> None:
        self.repository = repository
        self.metadata = metadata
        self.start_table = start_table
        self.preferred_start_subfolder = preferred_start_subfolder
        self.dialect = dialect
        self.records: List[LineageRecord] = []
        self._processed_tables: Set[str] = set()
        self._table_dependencies: Dict[str, Set[str]] = defaultdict(set)
        self._table_display_names: Dict[str, str] = {}
        self._remember_display_name(start_table, None)

    @property
    def dependencies(self) -> Dict[str, Set[str]]:
        return self._table_dependencies

    @property
    def display_names(self) -> Dict[str, str]:
        return self._table_display_names.copy()

    def extract(self) -> List[LineageRecord]:
        queue: deque[str] = deque([self.start_table.lower()])
        queued: Set[str] = {self.start_table.lower()}
        while queue:
            table = queue.popleft()
            if table in self._processed_tables:
                continue
            LOGGER.debug("Processing table %s", table)
            sql_path = self._resolve_table_path(table)
            if not sql_path:
                LOGGER.info("Table %s treated as original source (no SQL file).", table)
                self._processed_tables.add(table)
                continue
            file_records, downstream_tables = self._parse_file(sql_path, table)
            self.records.extend(file_records)
            self._processed_tables.add(table)
            for downstream in downstream_tables:
                dep_key = downstream.lower()
                if dep_key not in self._processed_tables and dep_key not in queued:
                    queue.append(dep_key)
                    queued.add(dep_key)
        return self.records

    def _resolve_table_path(self, table: str) -> Optional[Path]:
        preferred = self.repository.find_preferred_path(table, self.preferred_start_subfolder if table == self.start_table.lower() else None)
        if preferred:
            return preferred
        candidates = self.repository.get_paths_for_table(table)
        if candidates:
            return candidates[0]
        return None

    def _parse_file(self, path: Path, logical_table: str) -> Tuple[List[LineageRecord], Set[str]]:
        text = self.repository.get_text(path)
        if text is None:
            LOGGER.warning("Missing content for %s", path)
            return [], set()
        cleaner = strip_sql_comments
        cleaned = cleaner(text)
        normalizer = PlaceholderNormalizer()
        normalized = normalizer.normalize(cleaned)
        try:
            statements = parse(normalized, read=self.dialect)
        except Exception as exc:
            LOGGER.error("Failed to parse %s: %s", path, exc)
            return [], set()

        file_records: List[LineageRecord] = []
        downstream_tables: Set[str] = set()
        for statement in statements:
            if isinstance(statement, exp.Create) and isinstance(statement.this, exp.Table):
                target_table = statement.this
                query = statement.args.get("expression")
                if isinstance(query, exp.Select):
                    recs, deps = self._process_select(
                        query,
                        target_table,
                        path,
                        normalizer,
                    )
                    file_records.extend(recs)
                    downstream_tables.update(deps)
                elif isinstance(query, exp.Subqueryable):
                    for select in query.find_all(exp.Select):
                        recs, deps = self._process_select(
                            select,
                            target_table,
                            path,
                            normalizer,
                        )
                        file_records.extend(recs)
                        downstream_tables.update(deps)
            elif isinstance(statement, exp.Insert):
                target_table = statement.this
                query = statement.args.get("expression")
                if isinstance(query, exp.Select):
                    recs, deps = self._process_select(
                        query,
                        target_table,
                        path,
                        normalizer,
                    )
                    file_records.extend(recs)
                    downstream_tables.update(deps)
            elif isinstance(statement, exp.Update):
                target_table = statement.this
                select_like = statement.args.get("expressions")
                if select_like and isinstance(select_like[0], exp.Select):
                    recs, deps = self._process_select(
                        select_like[0],
                        target_table,
                        path,
                        normalizer,
                    )
                    file_records.extend(recs)
                    downstream_tables.update(deps)
            # Additional statement types (Delete, Merge, etc.) can be added as required.

        if not file_records:
            LOGGER.warning("No lineage records extracted from %s", path)

        return file_records, downstream_tables

    def _process_select(
        self,
        select_expr: exp.Select,
        target_table_expr: exp.Table,
        sql_path: Path,
        normalizer: PlaceholderNormalizer,
    ) -> Tuple[List[LineageRecord], Set[str]]:
        select_copy = select_expr.copy()
        try:
            qualified = qualify.qualify(
                select_copy,
                dialect=self.dialect,
                schema=None,
                validate_qualify_columns=False,
                identify=False,
            )
        except Exception as exc:
            LOGGER.warning("Qualification failed for %s: %s", sql_path, exc)
            qualified = select_copy

        cte_names = self._collect_cte_names(qualified)
        tables = self._collect_tables(qualified, cte_names, normalizer)
        queue_tables = {
            table.table
            for table in tables
            if not table.is_cte and not table.is_temp and table.table
        }
        all_source_tables = {table.table for table in tables if table.table}

        target_schema = normalizer.restore_identifier(target_table_expr.db)
        target_table_name = self._extract_table_name(target_table_expr, normalizer)
        self._remember_display_name(target_table_name, target_schema)

        nested_records, nested_dependencies = self._process_nested_structures(
            qualified,
            sql_path,
            normalizer,
        )
        logic_records: List[LineageRecord] = list(nested_records)
        all_source_tables.update(nested_dependencies)
        queue_tables.update({dep for dep in nested_dependencies if dep})

        for projection in qualified.expressions:
            logic_sql = normalizer.restore(
                projection.this.sql(dialect=self.dialect)
                if hasattr(projection, "this")
                else projection.sql(dialect=self.dialect)
            )
            source_refs = self._extract_sources(projection, tables, normalizer)
            deduped_refs: List[Tuple[Optional[str], Optional[str], Optional[str]]] = []
            seen_refs: Set[Tuple[Optional[str], Optional[str], Optional[str]]] = set()
            for ref in source_refs:
                key = (ref[0], ref[1], ref[2])
                if key in seen_refs:
                    continue
                seen_refs.add(key)
                deduped_refs.append(ref)
            source_refs = deduped_refs

            alias = projection.alias_or_name or projection.name
            alias = normalizer.restore(alias) if alias else alias
            if not alias and source_refs and source_refs[0][2]:
                alias = source_refs[0][2]
            if not alias:
                alias = logic_sql

            if not source_refs:
                record = LineageRecord(
                    target_schema=target_schema,
                    target_table=target_table_name or sql_path.stem,
                    target_column=alias,
                    source_schema=None,
                    source_table=None,
                    source_column=None,
                    logic=logic_sql,
                    sql_file=str(sql_path),
                )
                logic_records.append(record)
            else:
                for source_schema, source_table, source_column in source_refs:
                    record = LineageRecord(
                        target_schema=target_schema,
                        target_table=target_table_name or sql_path.stem,
                        target_column=alias,
                        source_schema=source_schema,
                        source_table=source_table,
                        source_column=source_column,
                        logic=logic_sql,
                        sql_file=str(sql_path),
                    )
                    logic_records.append(record)
        self._register_dependencies(
            target_table_name or sql_path.stem,
            target_schema,
            all_source_tables,
        )
        return logic_records, queue_tables

    @staticmethod
    def _collect_cte_names(select_expr: exp.Select) -> Set[str]:
        cte_clause = select_expr.args.get("with") or select_expr.args.get("with_")
        if not cte_clause:
            return set()
        names = {cte.alias_or_name.lower() for cte in cte_clause.expressions}
        return names

    def _collect_tables(
        self,
        select_expr: exp.Select,
        cte_names: Set[str],
        normalizer: PlaceholderNormalizer,
    ) -> List[TableRef]:
        tables: List[TableRef] = []
        for table in select_expr.find_all(exp.Table):
            base_name = normalizer.restore_identifier(table.name or table.alias_or_name)
            schema = normalizer.restore_identifier(table.db)
            alias = normalizer.restore_identifier(table.alias)
            is_cte = base_name.lower() in cte_names if base_name else False
            table_ref = TableRef(
                schema=schema,
                table=base_name or table.alias_or_name,
                alias=alias,
                is_cte=is_cte,
            )
            tables.append(table_ref)
            self._remember_display_name(table_ref.table, table_ref.schema)
        tables.extend(self._collect_subquery_refs(select_expr, normalizer))
        return tables

    def _collect_subquery_refs(
        self,
        select_expr: exp.Select,
        normalizer: PlaceholderNormalizer,
    ) -> List[TableRef]:
        refs: List[TableRef] = []
        seen_aliases: Set[str] = set()
        seen_tables: Set[str] = set()
        for subquery in select_expr.find_all(exp.Subquery):
            alias = normalizer.restore_identifier(subquery.alias)
            base_table = None
            source_table = None
            if isinstance(subquery.this, exp.Select):
                first_table = next(subquery.this.find_all(exp.Table), None)
                if first_table:
                    base_table = normalizer.restore_identifier(first_table.name or first_table.alias_or_name)
                    source_table = base_table
            table_name = alias or base_table
            if not table_name:
                continue
            key = table_name.lower()
            if alias and alias.lower() in seen_aliases:
                continue
            if not alias and key in seen_tables:
                continue
            if alias:
                seen_aliases.add(alias.lower())
            else:
                seen_tables.add(key)
            table_ref = TableRef(
                schema=None,
                table=table_name,
                alias=alias,
                is_cte=False,
                is_temp=True,
            )
            refs.append(table_ref)
            self._remember_display_name(table_name, None)
        return refs

    def _extract_sources(
        self,
        projection: exp.Expression,
        tables: Sequence[TableRef],
        normalizer: PlaceholderNormalizer,
    ) -> List[Tuple[Optional[str], Optional[str], Optional[str]]]:
        sources: List[Tuple[Optional[str], Optional[str], Optional[str]]] = []
        columns = list(projection.find_all(exp.Column))
        if not columns:
            return []
        for column in columns:
            column_name = normalizer.restore_identifier(column.name)
            table_name = normalizer.restore_identifier(column.table)
            preferred_table = self._match_table_ref(table_name, tables) if table_name else None
            schema_name, table_output = self.metadata.resolve(
                column_name,
                tables,
                preferred=preferred_table,
            )
            if not table_output and preferred_table:
                schema_name = preferred_table.schema
                table_output = preferred_table.table
            if not table_output:
                real_candidates = [
                    t for t in tables if not t.is_cte and not t.is_temp and t.table
                ]
                if len(real_candidates) == 1:
                    schema_name = real_candidates[0].schema
                    table_output = real_candidates[0].table
            if not table_output:
                table_output = table_name or "__UNKNOWN__"
            sources.append((schema_name, table_output, column_name))
        return sources

    @staticmethod
    def _match_table_ref(table_alias: str, tables: Sequence[TableRef]) -> Optional[TableRef]:
        lowered = table_alias.lower()
        for table_ref in tables:
            if table_ref.alias and table_ref.alias.lower() == lowered:
                return table_ref
            if table_ref.table and table_ref.table.lower() == lowered:
                return table_ref
        return None

    def _process_nested_structures(
        self,
        select_expr: exp.Select,
        sql_path: Path,
        normalizer: PlaceholderNormalizer,
    ) -> Tuple[List[LineageRecord], Set[str]]:
        records: List[LineageRecord] = []
        dependencies: Set[str] = set()
        processed: Set[str] = set()

        cte_clause = select_expr.args.get("with") or select_expr.args.get("with_")
        if cte_clause:
            for cte in cte_clause.expressions:
                alias = normalizer.restore_identifier(cte.alias_or_name)
                if not alias:
                    continue
                key = alias.lower()
                if key in processed:
                    continue
                processed.add(key)
                inner = cte.this
                if isinstance(inner, exp.Select):
                    table_expr = self._make_table_expr(alias)
                    nested_records, nested_deps = self._process_select(inner, table_expr, sql_path, normalizer)
                    records.extend(nested_records)
                    dependencies.update(nested_deps)

        for subquery in select_expr.find_all(exp.Subquery):
            alias = normalizer.restore_identifier(subquery.alias)
            if not alias:
                continue
            key = alias.lower()
            if key in processed:
                continue
            processed.add(key)
            if isinstance(subquery.this, exp.Select):
                table_expr = self._make_table_expr(alias)
                nested_records, nested_deps = self._process_select(subquery.this, table_expr, sql_path, normalizer)
                records.extend(nested_records)
                dependencies.update(nested_deps)

        return records, dependencies

    @staticmethod
    def _make_table_expr(name: str) -> exp.Table:
        return exp.to_table(name)

    def _extract_table_name(
        self,
        table_expr: exp.Table,
        normalizer: PlaceholderNormalizer,
    ) -> Optional[str]:

        if not table_expr:
            return None
        alias = table_expr.alias_or_name
        if alias:
            return normalizer.restore_identifier(alias)
        if table_expr.name:
            return normalizer.restore_identifier(table_expr.name)
        if table_expr.this:
            return normalizer.restore_identifier(str(table_expr.this))
        return None

    def _register_dependencies(
        self,
        target_table: Optional[str],
        target_schema: Optional[str],
        downstream_tables: Set[str],
    ) -> None:
        if not target_table:
            return
        key = target_table.lower()
        self._remember_display_name(target_table, target_schema)
        for dep in downstream_tables:
            if not dep:
                continue
            dep_key = dep.lower()
            self._table_dependencies[key].add(dep_key)

    def _remember_display_name(self, table: Optional[str], schema: Optional[str]) -> None:
        if not table:
            return
        key = table.lower()
        label = self._format_display_name(schema, table)
        self._table_display_names.setdefault(key, label)

    @staticmethod
    def _format_display_name(schema: Optional[str], table: str) -> str:
        if schema:
            return f"{schema}.{table}"
        return table


class LineageExporter:
    def __init__(
        self,
        records: Sequence[LineageRecord],
        output_dir: Path,
        start_table: str,
        dependencies: Optional[Dict[str, Set[str]]] = None,
        display_names: Optional[Dict[str, str]] = None,
    ) -> None:
        self.records = records
        self.output_dir = output_dir
        self.start_table = start_table
        self.dependencies = {k: set(v) for k, v in (dependencies or {}).items()}
        self.display_names = (display_names or {}).copy()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        now = dt.datetime.now()
        self.timestamp = now.strftime("%Y%m%d_%H%M%S")

    def export(self) -> Dict[str, Path]:
        base_name = f"ikg_lineage_{self.start_table}_{self.timestamp}"
        xls_path = self.output_dir / f"{base_name}.xls"
        json_path = self.output_dir / f"{base_name}.json"
        gexf_path = self.output_dir / f"{base_name}.gexf"
        tree_path = self.output_dir / f"{base_name}_tree.txt"
        self._to_excel(xls_path)
        self._to_json(json_path)
        self._to_gexf(gexf_path)
        self._to_tree(tree_path)
        return {"xls": xls_path, "json": json_path, "gexf": gexf_path, "tree": tree_path}

    def _to_excel(self, path: Path) -> None:
        df = pd.DataFrame(
            [
                {
                    "target_schema": record.target_schema,
                    "target_table": record.target_table,
                    "target_column": record.target_column,
                    "source_schema": record.source_schema,
                    "source_table": record.source_table,
                    "source_column": record.source_column,
                    "logic": record.logic,
                    "sql_file": record.sql_file,
                }
                for record in self.records
            ]
        )
        with pd.ExcelWriter(str(path), engine="xlwt") as writer:
            df.to_excel(writer, index=False, sheet_name="column_lineage")

    def _to_json(self, path: Path) -> None:
        payload = [
            {
                "target_schema": record.target_schema,
                "target_table": record.target_table,
                "target_column": record.target_column,
                "source_schema": record.source_schema,
                "source_table": record.source_table,
                "source_column": record.source_column,
                "logic": record.logic,
                "sql_file": record.sql_file,
            }
            for record in self.records
        ]
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _to_gexf(self, path: Path) -> None:
        graph = nx.MultiDiGraph()
        for record in self.records:
            target_node = (
                f"{record.target_schema or ''}.{record.target_table}".strip(".")
                if record.target_table
                else record.target_schema
            )
            if not target_node:
                target_node = record.target_table or "__TARGET__"
            source_node = None
            if record.source_table:
                source_node = f"{record.source_schema or ''}.{record.source_table}".strip(".") or record.source_table
            graph.add_node(
                target_node,
                schema=(record.target_schema or ""),
                table=(record.target_table or ""),
            )
            if source_node:
                graph.add_node(
                    source_node,
                    schema=(record.source_schema or ""),
                    table=(record.source_table or ""),
                )
                graph.add_edge(
                    source_node,
                    target_node,
                    source_column=(record.source_column or ""),
                    target_column=(record.target_column or ""),
                    logic=(record.logic or ""),
                    sql_file=(record.sql_file or ""),
                )
        nx.write_gexf(graph, str(path))

    def _to_tree(self, path: Path) -> None:
        lines: List[str] = []
        visited: Set[str] = set()

        def dfs(node_key: str, depth: int) -> None:
            label = self.display_names.get(node_key, node_key)
            prefix = "  " * depth + f"- {label}"
            if node_key in visited:
                lines.append(f"{prefix} (revisited)")
                return
            lines.append(prefix)
            visited.add(node_key)
            for child in sorted(self.dependencies.get(node_key, [])):
                dfs(child, depth + 1)

        start_key = self.start_table.lower()
        dfs(start_key, 0)
        path.write_text("\n".join(lines), encoding="utf-8")


def _split_exclude_folders(exclude: Optional[str]) -> List[str]:
    if not exclude:
        return []
    parts = re.split(r"[,\s]+", exclude.strip())
    return [part for part in parts if part]


def _split_schemas(schemas: Optional[str]) -> List[str]:
    if not schemas:
        return []
    return [schema.strip() for schema in schemas.split(",") if schema.strip()]


@APP.command()
def run(
    gitlab_url: str = typer.Option(..., envvar="GITLAB_URL"),
    project_id: str = typer.Option(..., envvar="PROJECT_ID"),
    branch: str = typer.Option("ikg-master", envvar="BRANCH"),
    sql_folder: str = typer.Option(..., envvar="SQL_FOLDER"),
    exclude_folder: Optional[str] = typer.Option("", envvar="EXCLUDE_FOLDER"),
    token: str = typer.Option(
        ...,
        envvar="TOKEN",
        prompt="GitLab private token",
        hide_input=True,
    ),
    start_table: str = typer.Option(..., envvar="START_TABLE", prompt=True),
    output_dir: Path = typer.Option(Path.cwd(), envvar="OUTPUT_DIR"),
    log_level: str = typer.Option("INFO", envvar="LOG_LEVEL"),
    greenplum_host: Optional[str] = typer.Option(None, envvar="GREENPLUM_HOST"),
    greenplum_port: Optional[int] = typer.Option(5432, envvar="GREENPLUM_PORT"),
    greenplum_db: Optional[str] = typer.Option(None, envvar="GREENPLUM_DB"),
    greenplum_user: Optional[str] = typer.Option(None, envvar="GREENPLUM_USER"),
    greenplum_password: Optional[str] = typer.Option(
        None,
        envvar="GREENPLUM_PASSWORD",
        prompt="Greenplum password (press enter to skip)",
        hide_input=True,
    ),
    greenplum_schema: Optional[str] = typer.Option(None, envvar="GREENPLUM_SCHEMA"),
    preferred_start_subfolder: str = typer.Option("ikg_create_profiles"),
) -> None:
    """Entry point for the SQL lineage extraction workflow."""

    setup_logging(log_level)
    LOGGER.info("Starting lineage extraction for table '%s'", start_table)
    exclude_folders = _split_exclude_folders(exclude_folder)
    schemas = _split_schemas(greenplum_schema)
    greenplum_password = greenplum_password or None
    metadata = GreenplumMetadata(
        host=greenplum_host,
        port=greenplum_port,
        database=greenplum_db,
        user=greenplum_user,
        password=greenplum_password,
        schemas=schemas,
    )
    if not metadata.enabled:
        raise typer.BadParameter(
            "Greenplum credentials (host, db, user, password, and schema list) are required for lineage disambiguation."
        )
    try:
        metadata.ensure_connection()
    except RuntimeError as exc:
        raise typer.BadParameter(str(exc))

    cache_dir = Path.cwd() / ".ikg_sql_cache"
    fetcher = GitLabSQLFetcher(
        gitlab_url=gitlab_url,
        private_token=token,
        project_id=project_id,
        branch=branch,
        sql_folder=sql_folder,
        exclude_folders=exclude_folders,
    )
    repository = fetcher.download(cache_dir)

    parser = SQLLineageParser(
        repository=repository,
        metadata=metadata,
        start_table=start_table.lower(),
        preferred_start_subfolder=preferred_start_subfolder,
    )
    records = parser.extract()
    if not records:
        LOGGER.warning("No lineage records generated.")
    exporter = LineageExporter(
        records,
        output_dir,
        start_table,
        parser.dependencies,
        parser.display_names,
    )
    outputs = exporter.export()
    LOGGER.info("Exported %d lineage rows", len(records))
    LOGGER.info("Lineage exports created:")
    for fmt, path in outputs.items():
        LOGGER.info("  %s -> %s", fmt.upper(), path)
    metadata.close()


def main() -> None:
    APP()


if __name__ == "__main__":
    main()
