#!/usr/bin/env python3
"""
IKG SQL Lineage Extraction Tool
================================

This tool connects to a GitLab repository, downloads SQL files from a specified branch
and directory, parses them with sqlglot to derive column-level lineage, and exports
the lineage to multiple formats (Excel, JSON, graph files, PyVis HTML, and optionally
Neo4j). It supports recursive discovery of SQL dependencies starting from a user-supplied
root SQL file, CTE parsing, alias handling, and Greenplum metadata lookups for unmapped
columns.

Key features:
    * GitLab API integration (python-gitlab)
    * Recursive SQL discovery with exclusion support
    * Column-level lineage extraction via sqlglot
    * CTE and subquery lineage propagation
    * Optional metadata resolution via Greenplum (psycopg2)
    * Outputs: Excel, JSON, NetworkX graph (GEXF), PyVis interactive HTML, Neo4j push

Usage:
    python ikg_lineage.py --project-id <PROJECT_ID> --token <TOKEN> --start-table <TABLE>

Environment variables can also be used; run with --help for full options.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

try:
    import gitlab
except ImportError as err:  # pragma: no cover - dependency check
    raise SystemExit(
        "python-gitlab is required. Install with `pip install python-gitlab`."
    ) from err

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = None

try:
    import networkx as nx
except ImportError:  # pragma: no cover - optional dependency
    nx = None

try:
    from pyvis.network import Network as PyVisNetwork
except ImportError:  # pragma: no cover - optional dependency
    PyVisNetwork = None

try:
    from neo4j import GraphDatabase
except ImportError:  # pragma: no cover - optional dependency
    GraphDatabase = None

try:
    import psycopg2
    from psycopg2.extras import DictCursor
except ImportError:  # pragma: no cover - optional dependency
    psycopg2 = None
    DictCursor = None

from sqlglot import exp, parse
from sqlglot.errors import ParseError

LOGGER = logging.getLogger("ikg_lineage")


# ---------------------------------------------------------------------------
# Data classes for configuration and lineage entities
# ---------------------------------------------------------------------------


@dataclass
class GitLabConfig:
    url: str
    private_token: str
    project_id: str
    branch: str = "ikg-master"
    sql_folder: str = "dags/ikg/scripts/sql"
    exclude_folder: Optional[str] = "ikg_new_fa_shhp_map"


@dataclass
class GreenplumConfig:
    host: Optional[str] = None
    port: int = 5432
    database: Optional[str] = None
    user: Optional[str] = None
    password: Optional[str] = None
    default_schema: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return all(
            [
                self.host,
                self.database,
                self.user,
                self.password,
            ]
        )


@dataclass
class Neo4jConfig:
    uri: Optional[str] = None
    user: Optional[str] = None
    password: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return all([self.uri, self.user, self.password])


@dataclass
class LineageRecord:
    target_schema: Optional[str]
    target_table: str
    target_column: str
    source_schema: Optional[str]
    source_table: str
    source_column: str
    logic: str
    sql_file: str
    note: Optional[str] = None


@dataclass
class TableReference:
    name: str
    schema: Optional[str]
    alias: Optional[str]
    expression: Optional[exp.Expression] = None
    is_cte: bool = False
    is_subquery: bool = False

    @property
    def identifier(self) -> str:
        if self.schema:
            return f"{self.schema}.{self.name}"
        return self.name

    @property
    def alias_or_name(self) -> str:
        return self.alias or self.name


# ---------------------------------------------------------------------------
# GitLab integration
# ---------------------------------------------------------------------------


class GitLabSQLFetcher:
    """Fetches SQL file contents from GitLab."""

    def __init__(self, config: GitLabConfig) -> None:
        self.config = config
        self._project = None

    @property
    def project(self):
        if self._project is None:
            LOGGER.debug("Connecting to GitLab project %s", self.config.project_id)
            gl = gitlab.Gitlab(self.config.url, private_token=self.config.private_token)
            self._project = gl.projects.get(self.config.project_id)
        return self._project

    def fetch_sql_files(self) -> Dict[str, str]:
        """Return mapping of file_path -> SQL text for all SQL files under the configured folder."""
        LOGGER.info(
            "Fetching SQL files from project %s branch %s",
            self.config.project_id,
            self.config.branch,
        )
        sql_files: Dict[str, str] = {}
        self._walk_folder(self.config.sql_folder, sql_files)
        LOGGER.info("Fetched %d SQL files", len(sql_files))
        return sql_files

    def _walk_folder(self, folder: str, accumulator: Dict[str, str]) -> None:
        tree_items = self.project.repository_tree(
            path=folder,
            ref=self.config.branch,
            all=True,
            recursive=False,
        )
        for item in tree_items:
            item_path = item["path"]
            if (
                self.config.exclude_folder
                and item_path.startswith(self.config.exclude_folder)
            ):
                LOGGER.debug("Skipping excluded path %s", item_path)
                continue
            if item["type"] == "tree":
                self._walk_folder(item_path, accumulator)
                continue
            if item["type"] != "blob":
                continue
            if not item_path.lower().endswith(".sql"):
                continue
            accumulator[item_path] = self._download_file(item_path)

    def _download_file(self, file_path: str) -> str:
        LOGGER.debug("Downloading %s", file_path)
        file_obj = self.project.files.get(file_path=file_path, ref=self.config.branch)
        content = base64.b64decode(file_obj.content).decode("utf-8", errors="ignore")
        return content


# ---------------------------------------------------------------------------
# Greenplum metadata lookup
# ---------------------------------------------------------------------------


class GreenplumMetadataClient:
    """Resolves table/column metadata using a Greenplum (PostgreSQL) catalog."""

    def __init__(self, config: GreenplumConfig) -> None:
        self.config = config
        self._connection = None
        if self.config.enabled and psycopg2 is None:
            LOGGER.warning(
                "psycopg2 is not installed. Greenplum metadata lookups are disabled."
            )

    def _ensure_connection(self):
        if not self.config.enabled or psycopg2 is None:
            return None
        if self._connection is None or self._connection.closed != 0:
            LOGGER.debug("Opening Greenplum connection to %s", self.config.host)
            self._connection = psycopg2.connect(
                host=self.config.host,
                port=self.config.port,
                dbname=self.config.database,
                user=self.config.user,
                password=self.config.password,
                cursor_factory=DictCursor if DictCursor else None,
            )
        return self._connection

    def resolve_column(
        self,
        column_name: str,
        candidate_tables: Optional[Iterable[str]] = None,
        schema: Optional[str] = None,
    ) -> Optional[Tuple[str, str, str]]:
        """
        Attempt to resolve a column to (schema, table, column) via metadata.
        Returns None if resolution fails or metadata is unavailable.
        """
        conn = self._ensure_connection()
        if conn is None:
            return None

        tables_filter = ""
        params: List[str] = []
        if candidate_tables:
            placeholders = ", ".join(["%s"] * len(list(candidate_tables)))
            tables_filter = f"AND table_name IN ({placeholders})"
            params.extend(candidate_tables)
        if schema or self.config.default_schema:
            params.append(schema or self.config.default_schema or "")
            schema_clause = "AND table_schema = %s"
        else:
            schema_clause = ""

        query = f"""
            SELECT table_schema, table_name, column_name
            FROM information_schema.columns
            WHERE column_name = %s
            {tables_filter}
            {schema_clause}
            ORDER BY table_schema, table_name
            LIMIT 1
        """
        params.insert(0, column_name)

        try:
            with conn.cursor() as cur:
                cur.execute(query, params)
                row = cur.fetchone()
                if row:
                    LOGGER.debug(
                        "Resolved column %s -> %s.%s",
                        column_name,
                        row["table_name"] if isinstance(row, dict) else row[1],
                        row["column_name"] if isinstance(row, dict) else row[2],
                    )
                    if isinstance(row, dict):
                        return row["table_schema"], row["table_name"], row["column_name"]
                    return row[0], row[1], row[2]
        except Exception as err:  # pragma: no cover - defensive
            LOGGER.warning("Metadata lookup failed for %s: %s", column_name, err)
        return None

    def close(self) -> None:
        if self._connection and self._connection.closed == 0:
            self._connection.close()
            self._connection = None


# ---------------------------------------------------------------------------
# Lineage parsing
# ---------------------------------------------------------------------------


class SQLLineageExtractor:
    """Parses SQL using sqlglot and extracts column-level lineage."""

    def __init__(
        self,
        dialect: str = "postgres",
        metadata_client: Optional[GreenplumMetadataClient] = None,
    ) -> None:
        self.dialect = dialect
        self.metadata_client = metadata_client
        self._cte_cache: Dict[str, List[LineageRecord]] = {}

    # Public API ---------------------------------------------------------

    def extract_file_lineage(
        self,
        sql_text: str,
        file_path: str,
        default_target_table: str,
        default_target_schema: Optional[str] = None,
    ) -> Tuple[List[LineageRecord], Set[str]]:
        records: List[LineageRecord] = []
        referenced_tables: Set[str] = set()

        try:
            statements = parse(sql_text, read=self.dialect)
        except ParseError as err:
            LOGGER.error("Failed to parse %s: %s", file_path, err)
            return records, referenced_tables

        cte_definitions: Dict[str, exp.Expression] = {}
        for statement in statements:
            cte_definitions.update(self._collect_ctes(statement))

        for statement in statements:
            stmt_records, stmt_tables = self._process_statement(
                statement=statement,
                file_path=file_path,
                default_target_table=default_target_table,
                default_target_schema=default_target_schema,
                cte_definitions=cte_definitions,
                ancestors=set(),
            )
            records.extend(stmt_records)
            referenced_tables.update(stmt_tables)
        return records, referenced_tables

    # Internal helpers ---------------------------------------------------

    def _process_statement(
        self,
        statement: exp.Expression,
        file_path: str,
        default_target_table: str,
        default_target_schema: Optional[str],
        cte_definitions: Dict[str, exp.Expression],
        ancestors: Set[str],
    ) -> Tuple[List[LineageRecord], Set[str]]:
        if isinstance(statement, exp.Insert):
            return self._process_insert_statement(
                statement,
                file_path,
                default_target_table,
                default_target_schema,
                cte_definitions,
                ancestors,
            )
        if isinstance(statement, exp.Create):
            return self._process_create_statement(
                statement,
                file_path,
                default_target_table,
                default_target_schema,
                cte_definitions,
                ancestors,
            )
        if isinstance(statement, exp.Update):
            return self._process_update_statement(
                statement,
                file_path,
                default_target_table,
                default_target_schema,
                cte_definitions,
                ancestors,
            )
        if isinstance(statement, exp.Delete):
            # DELETE does not produce target columns but we can capture dependencies
            return self._process_delete_statement(
                statement,
                file_path,
                default_target_table,
                default_target_schema,
                cte_definitions,
            )

        if isinstance(statement, exp.Select):
            return self._process_select_block(
                statement,
                file_path=file_path,
                target_table=default_target_table,
                target_schema=default_target_schema,
                provided_target_columns=None,
                cte_definitions=cte_definitions,
                ancestors=ancestors,
            )

        if isinstance(statement, (exp.Union, exp.Except, exp.Intersect)):
            records: List[LineageRecord] = []
            referenced: Set[str] = set()
            for select_part in self._flatten_set_operation(statement):
                part_records, part_tables = self._process_select_block(
                    select_part,
                    file_path=file_path,
                    target_table=default_target_table,
                    target_schema=default_target_schema,
                    provided_target_columns=None,
                    cte_definitions=cte_definitions,
                    ancestors=ancestors,
                )
                records.extend(part_records)
                referenced.update(part_tables)
            return records, referenced

        LOGGER.debug(
            "Encountered unsupported statement type %s in %s",
            type(statement).__name__,
            file_path,
        )
        return [], set()

    def _process_insert_statement(
        self,
        statement: exp.Insert,
        file_path: str,
        default_target_table: str,
        default_target_schema: Optional[str],
        cte_definitions: Dict[str, exp.Expression],
        ancestors: Set[str],
    ) -> Tuple[List[LineageRecord], Set[str]]:
        target_schema, target_table = self._table_from_expression(
            statement.this,
            default_target_table,
            default_target_schema,
        )
        columns = self._extract_identifier_list(statement.args.get("columns"))
        expression = statement.args.get("expression")
        if expression is None:
            LOGGER.warning("INSERT without expression in %s", file_path)
            return [], set()
        if isinstance(expression, exp.Values):
            return self._process_values_block(
                expression,
                file_path=file_path,
                target_table=target_table,
                target_schema=target_schema,
                target_columns=columns,
            )

        records, referenced_tables = self._process_select_block(
            expression,
            file_path=file_path,
            target_table=target_table,
            target_schema=target_schema,
            provided_target_columns=columns or None,
            cte_definitions=cte_definitions,
            ancestors=ancestors,
        )
        return records, referenced_tables

    def _process_create_statement(
        self,
        statement: exp.Create,
        file_path: str,
        default_target_table: str,
        default_target_schema: Optional[str],
        cte_definitions: Dict[str, exp.Expression],
        ancestors: Set[str],
    ) -> Tuple[List[LineageRecord], Set[str]]:
        target_schema, target_table = self._table_from_expression(
            statement.this,
            default_target_table,
            default_target_schema,
        )
        expression = statement.args.get("expression")
        if expression is None:
            LOGGER.debug("CREATE without expression in %s", file_path)
            return [], set()
        if isinstance(expression, exp.Values):
            return self._process_values_block(
                expression,
                file_path=file_path,
                target_table=target_table,
                target_schema=target_schema,
                target_columns=None,
            )
        records, referenced_tables = self._process_select_block(
            expression,
            file_path=file_path,
            target_table=target_table,
            target_schema=target_schema,
            provided_target_columns=self._extract_identifier_list(
                statement.args.get("columns")
            ),
            cte_definitions=cte_definitions,
            ancestors=ancestors,
        )
        return records, referenced_tables

    def _process_update_statement(
        self,
        statement: exp.Update,
        file_path: str,
        default_target_table: str,
        default_target_schema: Optional[str],
        cte_definitions: Dict[str, exp.Expression],
        ancestors: Set[str],
    ) -> Tuple[List[LineageRecord], Set[str]]:
        target_schema, target_table = self._table_from_expression(
            statement.this,
            default_target_table,
            default_target_schema,
        )
        alias_map = self._collect_table_references(
            select_like=statement,
            cte_definitions=cte_definitions,
            ancestors=ancestors,
        )
        records: List[LineageRecord] = []
        referenced_tables = {
            ref.identifier for ref in alias_map.values() if not ref.is_cte
        }
        assignments = statement.args.get("expressions") or []
        for assignment in assignments:
            if isinstance(assignment, exp.EQ):
                target_col = self._column_name(assignment.left)
                logic = assignment.right.sql(dialect=self.dialect)
                sources = self._extract_source_columns(
                    assignment.right,
                    alias_map=alias_map,
                    cte_definitions=cte_definitions,
                    ancestors=ancestors,
                )
                if not sources:
                    resolved = self._fallback_metadata_resolution(
                        column_name=target_col,
                        candidate_aliases=alias_map.keys(),
                    )
                    sources = [resolved] if resolved else []
                for source in sources:
                    records.append(
                        LineageRecord(
                            target_schema=target_schema,
                            target_table=target_table,
                            target_column=target_col,
                            source_schema=source[0],
                            source_table=source[1],
                            source_column=source[2],
                            logic=logic,
                            sql_file=file_path,
                        )
                    )
        return records, referenced_tables

    def _process_delete_statement(
        self,
        statement: exp.Delete,
        file_path: str,
        default_target_table: str,
        default_target_schema: Optional[str],
        cte_definitions: Dict[str, exp.Expression],
    ) -> Tuple[List[LineageRecord], Set[str]]:
        target_schema, target_table = self._table_from_expression(
            statement.this,
            default_target_table,
            default_target_schema,
        )
        alias_map = self._collect_table_references(
            select_like=statement,
            cte_definitions=cte_definitions,
            ancestors=set(),
        )
        referenced_tables = {
            ref.identifier for ref in alias_map.values() if not ref.is_cte
        }
        logic = statement.args.get("where").sql(dialect=self.dialect) if statement.args.get("where") else "DELETE"
        records = [
            LineageRecord(
                target_schema=target_schema,
                target_table=target_table,
                target_column="*",
                source_schema=target_schema,
                source_table=target_table,
                source_column="*",
                logic=logic,
                sql_file=file_path,
                note="DELETE statement (no explicit column lineage)",
            )
        ]
        return records, referenced_tables

    def _process_values_block(
        self,
        values_exp: exp.Values,
        file_path: str,
        target_table: str,
        target_schema: Optional[str],
        target_columns: Optional[List[str]],
    ) -> Tuple[List[LineageRecord], Set[str]]:
        records: List[LineageRecord] = []
        provided_cols = target_columns or [
            f"col_{idx + 1}" for idx, _ in enumerate(values_exp.expressions)
        ]
        for row in values_exp.expressions:
            for idx, value in enumerate(row.expressions):
                target_col = (
                    provided_cols[idx] if idx < len(provided_cols) else f"col_{idx+1}"
                )
                records.append(
                    LineageRecord(
                        target_schema=target_schema,
                        target_table=target_table,
                        target_column=target_col,
                        source_schema=None,
                        source_table="VALUES",
                        source_column=value.sql(dialect=self.dialect),
                        logic=value.sql(dialect=self.dialect),
                        sql_file=file_path,
                    )
                )
        return records, set()

    def _process_select_block(
        self,
        select_like: exp.Expression,
        file_path: str,
        target_table: str,
        target_schema: Optional[str],
        provided_target_columns: Optional[List[str]],
        cte_definitions: Dict[str, exp.Expression],
        ancestors: Set[str],
    ) -> Tuple[List[LineageRecord], Set[str]]:
        records: List[LineageRecord] = []
        referenced_tables: Set[str] = set()

        for select_expr in self._flatten_select(select_like):
            alias_map = self._collect_table_references(
                select_expr,
                cte_definitions=cte_definitions,
                ancestors=ancestors,
            )
            referenced_tables.update(
                ref.identifier for ref in alias_map.values() if not ref.is_cte
            )
            projections = list(select_expr.expressions or [])
            target_columns = self._resolve_target_columns(
                projections, provided_target_columns
            )

            for idx, projection in enumerate(projections):
                target_column = (
                    target_columns[idx] if idx < len(target_columns) else f"col_{idx+1}"
                )
                expression = projection.this if isinstance(projection, exp.Alias) else projection
                logic_sql = expression.sql(dialect=self.dialect)
                sources = self._extract_source_columns(
                    expression,
                    alias_map=alias_map,
                    cte_definitions=cte_definitions,
                    ancestors=ancestors,
                )
                if not sources:
                    resolved = self._fallback_metadata_resolution(
                        column_name=target_column,
                        candidate_aliases=alias_map.keys(),
                        schema_hint=target_schema,
                    )
                    sources = [resolved] if resolved else []
                for source in sources:
                    records.append(
                        LineageRecord(
                            target_schema=target_schema,
                            target_table=target_table,
                            target_column=target_column,
                            source_schema=source[0],
                            source_table=source[1],
                            source_column=source[2],
                            logic=logic_sql,
                            sql_file=file_path,
                        )
                    )
        return records, referenced_tables

    def _collect_ctes(self, statement: exp.Expression) -> Dict[str, exp.Expression]:
        cte_definitions: Dict[str, exp.Expression] = {}
        with_clause = statement.args.get("with")
        if not with_clause:
            return cte_definitions
        for cte in with_clause.expressions:
            alias = cte.alias
            if not alias:
                continue
            name = alias.this.name if isinstance(alias.this, exp.Identifier) else alias.this.sql()
            cte_definitions[name] = cte.this
        return cte_definitions

    def _collect_table_references(
        self,
        select_like: exp.Expression,
        cte_definitions: Dict[str, exp.Expression],
        ancestors: Set[str],
    ) -> Dict[str, TableReference]:
        alias_map: Dict[str, TableReference] = {}

        def register(ref: TableReference) -> None:
            alias_map[ref.alias_or_name] = ref

        from_clause = select_like.args.get("from")
        if isinstance(from_clause, exp.From):
            for source in from_clause.expressions or []:
                register(self._table_reference_from_source(source, cte_definitions, ancestors))

        for join in select_like.args.get("joins") or []:
            register(
                self._table_reference_from_source(
                    join.this,
                    cte_definitions,
                    ancestors,
                )
            )
        return alias_map

    def _table_reference_from_source(
        self,
        source: exp.Expression,
        cte_definitions: Dict[str, exp.Expression],
        ancestors: Set[str],
    ) -> TableReference:
        if isinstance(source, exp.Subquery):
            alias = self._alias_name(source)
            return TableReference(
                name=alias or "subquery",
                schema=None,
                alias=alias,
                expression=source.this,
                is_subquery=True,
            )
        if isinstance(source, exp.Table):
            schema, table = self._schema_table_from_table_exp(source)
            alias = self._alias_name(source)
            is_cte = table in cte_definitions
            if is_cte and table not in self._cte_cache:
                if table in ancestors:
                    LOGGER.warning("Detected recursive CTE %s, skipping deeper lineage", table)
                else:
                    cte_expr = cte_definitions[table]
                    cte_columns = self._extract_identifier_list(
                        getattr(cte_expr, "args", {}).get("columns")
                    )
                    self._cte_cache[table] = self._process_select_block(
                        cte_expr,
                        file_path=f"[cte:{table}]",
                        target_table=table,
                        target_schema=schema,
                        provided_target_columns=cte_columns or None,
                        cte_definitions=cte_definitions,
                        ancestors=ancestors | {table},
                    )[0]
            return TableReference(
                name=table,
                schema=schema,
                alias=alias,
                is_cte=is_cte,
            )
        if isinstance(source, exp.TableFunction):
            alias = self._alias_name(source)
            func_name = source.this.sql() if source.this else "table_function"
            return TableReference(
                name=func_name,
                schema=None,
                alias=alias or func_name,
                expression=source,
            )
        if isinstance(source, exp.Generator):
            alias = self._alias_name(source)
            generator_name = source.this.sql() if source.this else "generator"
            return TableReference(
                name=generator_name,
                schema=None,
                alias=alias or generator_name,
                expression=source,
            )
        alias = self._alias_name(source)
        return TableReference(
            name=alias or source.sql(dialect=self.dialect),
            schema=None,
            alias=alias,
            expression=source,
        )

    def _extract_source_columns(
        self,
        expression: exp.Expression,
        alias_map: Dict[str, TableReference],
        cte_definitions: Dict[str, exp.Expression],
        ancestors: Set[str],
    ) -> List[Tuple[Optional[str], str, str]]:
        sources: List[Tuple[Optional[str], str, str]] = []
        for column in expression.find_all(exp.Column):
            column_name = column.name
            table_alias = column.table
            if table_alias and table_alias in alias_map:
                table_ref = alias_map[table_alias]
                if table_ref.is_cte:
                    cte_records = self._cte_cache.get(table_ref.name, [])
                    filtered = [
                        (r.source_schema, r.source_table, r.source_column)
                        for r in cte_records
                        if r.target_column == column_name
                    ]
                    if filtered:
                        sources.extend(filtered)
                        continue
                if table_ref.is_subquery and table_ref.expression:
                    subquery_records, _ = self._process_select_block(
                        table_ref.expression,
                        file_path=f"[subquery:{table_ref.alias_or_name}]",
                        target_table=table_ref.alias_or_name,
                        target_schema=table_ref.schema,
                        provided_target_columns=None,
                        cte_definitions=cte_definitions,
                        ancestors=ancestors | {table_ref.alias_or_name},
                    )
                    filtered = [
                        (r.source_schema, r.source_table, r.source_column)
                        for r in subquery_records
                        if r.target_column == column_name
                    ]
                    if filtered:
                        sources.extend(filtered)
                        continue
                sources.append((table_ref.schema, table_ref.name, column_name))
                continue

            if table_alias and table_alias not in alias_map:
                # Possibly schema.table syntax
                schema, table = self._split_schema_table(table_alias)
                sources.append((schema, table, column_name))
                continue

            if not table_alias and len(alias_map) == 1:
                single_ref = next(iter(alias_map.values()))
                sources.append((single_ref.schema, single_ref.name, column_name))
                continue

            resolved = self._fallback_metadata_resolution(
                column_name=column_name,
                candidate_aliases=alias_map.keys(),
            )
            if resolved:
                sources.append(resolved)
            else:
                sources.append((None, table_alias or "UNKNOWN", column_name))
        return sources

    def _fallback_metadata_resolution(
        self,
        column_name: str,
        candidate_aliases: Iterable[str],
        schema_hint: Optional[str] = None,
    ) -> Optional[Tuple[Optional[str], str, str]]:
        if not self.metadata_client:
            return None
        candidate_tables = [alias for alias in candidate_aliases if alias]
        resolved = self.metadata_client.resolve_column(
            column_name=column_name,
            candidate_tables=candidate_tables,
            schema=schema_hint,
        )
        if resolved:
            return resolved
        return None

    def _resolve_target_columns(
        self,
        projections: List[exp.Expression],
        provided_target_columns: Optional[List[str]],
    ) -> List[str]:
        if provided_target_columns:
            return provided_target_columns
        results: List[str] = []
        for idx, projection in enumerate(projections):
            if isinstance(projection, exp.Alias):
                alias = self._alias_name(projection)
                results.append(alias or f"col_{idx+1}")
            elif isinstance(projection, exp.Column):
                results.append(projection.name)
            else:
                results.append(f"col_{idx+1}")
        return results

    # Utility helpers ----------------------------------------------------

    def _flatten_select(self, expression: exp.Expression) -> List[exp.Select]:
        if isinstance(expression, exp.Select):
            return [expression]
        if isinstance(expression, (exp.Subquery, exp.CTE)):
            return self._flatten_select(expression.this)
        if isinstance(expression, (exp.Union, exp.Except, exp.Intersect)):
            selects: List[exp.Select] = []
            selects.extend(self._flatten_select(expression.this))
            selects.extend(self._flatten_select(expression.expression))
            return selects
        if hasattr(expression, "this"):
            return self._flatten_select(expression.this)
        return []

    def _flatten_set_operation(self, expression: exp.Expression) -> List[exp.Select]:
        selects: List[exp.Select] = []
        selects.extend(self._flatten_select(expression.this))
        selects.extend(self._flatten_select(expression.expression))
        return selects

    def _schema_table_from_table_exp(
        self, table_exp: exp.Table
    ) -> Tuple[Optional[str], str]:
        schema = table_exp.args.get("db")
        schema_name = (
            schema.this if isinstance(schema, exp.Identifier) else schema.sql()
        ) if schema else None
        table_name = table_exp.name
        return schema_name, table_name

    def _table_from_expression(
        self,
        table_exp: Optional[exp.Expression],
        default_table: str,
        default_schema: Optional[str],
    ) -> Tuple[Optional[str], str]:
        if isinstance(table_exp, exp.Table):
            schema, table = self._schema_table_from_table_exp(table_exp)
            return schema or default_schema, table or default_table
        return default_schema, default_table

    def _alias_name(self, expression: exp.Expression) -> Optional[str]:
        alias = expression.args.get("alias")
        if isinstance(alias, exp.TableAlias):
            identifier = alias.this
            if isinstance(identifier, exp.Identifier):
                return identifier.name
        if isinstance(expression, exp.Alias):
            if isinstance(expression.alias, exp.Identifier):
                return expression.alias.name
        return None

    def _column_name(self, expression: exp.Expression) -> str:
        if isinstance(expression, exp.Column):
            return expression.name
        if isinstance(expression, exp.Identifier):
            return expression.name
        return expression.sql(dialect=self.dialect)

    def _extract_identifier_list(
        self, node: Optional[Iterable[exp.Expression]]
    ) -> List[str]:
        if not node:
            return []
        results: List[str] = []
        for item in node:
            if isinstance(item, exp.Identifier):
                results.append(item.name)
            elif isinstance(item, exp.Column):
                results.append(item.name)
            else:
                results.append(item.sql(dialect=self.dialect))
        return results

    def _split_schema_table(self, identifier: str) -> Tuple[Optional[str], str]:
        parts = identifier.split(".", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return None, identifier


# ---------------------------------------------------------------------------
# Output management
# ---------------------------------------------------------------------------


class OutputManager:
    def __init__(self, output_dir: Path, timestamp: Optional[str] = None) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = timestamp or dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    def write_excel(self, records: List[LineageRecord]) -> Optional[Path]:
        if pd is None:
            LOGGER.warning("pandas not installed; skipping Excel export.")
            return None
        data = [
            {
                "target_schema": r.target_schema,
                "target_table": r.target_table,
                "target_column": r.target_column,
                "source_schema": r.source_schema,
                "source_table": r.source_table,
                "source_column": r.source_column,
                "logic": r.logic,
                "sql_file": r.sql_file,
                "note": r.note,
            }
            for r in records
        ]
        df = pd.DataFrame(data)
        path = self.output_dir / f"ikg_column_lineage_output_{self.timestamp}.xlsx"
        df.to_excel(path, index=False)
        LOGGER.info("Excel lineage exported to %s", path)
        return path

    def write_json(self, records: List[LineageRecord]) -> Path:
        data = [
            {
                "target_schema": r.target_schema,
                "target_table": r.target_table,
                "target_column": r.target_column,
                "source_schema": r.source_schema,
                "source_table": r.source_table,
                "source_column": r.source_column,
                "logic": r.logic,
                "sql_file": r.sql_file,
                "note": r.note,
            }
            for r in records
        ]
        path = self.output_dir / f"ikg_column_lineage_output_{self.timestamp}.json"
        with path.open("w", encoding="utf-8") as json_file:
            json.dump(data, json_file, indent=2)
        LOGGER.info("JSON lineage exported to %s", path)
        return path

    def write_graph(self, records: List[LineageRecord]) -> Optional[Path]:
        if nx is None:
            LOGGER.warning("networkx not installed; skipping graph export.")
            return None
        graph = nx.DiGraph()
        for record in records:
            target_node = (
                f"{record.target_schema or 'public'}.{record.target_table}.{record.target_column}"
            )
            source_node = (
                f"{record.source_schema or 'public'}.{record.source_table}.{record.source_column}"
            )
            graph.add_node(
                target_node,
                label=target_node,
                type="target",
                table=record.target_table,
            )
            graph.add_node(
                source_node,
                label=source_node,
                type="source",
                table=record.source_table,
            )
            graph.add_edge(
                source_node,
                target_node,
                logic=record.logic,
                sql_file=record.sql_file,
            )
        path = self.output_dir / f"ikg_column_lineage_output_{self.timestamp}.gexf"
        nx.write_gexf(graph, path)
        LOGGER.info("Graph GEXF exported to %s", path)
        return path

    def write_pyvis(self, records: List[LineageRecord]) -> Optional[Path]:
        if PyVisNetwork is None:
            LOGGER.warning("pyvis not installed; skipping interactive visualization.")
            return None
        network = PyVisNetwork(height="750px", width="100%", directed=True)
        nodes = set()
        for record in records:
            target_node = (
                f"{record.target_schema or 'public'}.{record.target_table}.{record.target_column}"
            )
            source_node = (
                f"{record.source_schema or 'public'}.{record.source_table}.{record.source_column}"
            )
            if target_node not in nodes:
                network.add_node(
                    target_node,
                    label=target_node,
                    title=f"Target column\nLogic: {record.logic}",
                    color="#1f78b4",
                )
                nodes.add(target_node)
            if source_node not in nodes:
                network.add_node(
                    source_node,
                    label=source_node,
                    title=f"Source column\nLogic: {record.logic}",
                    color="#33a02c",
                )
                nodes.add(source_node)
            network.add_edge(
                source_node,
                target_node,
                title=record.logic,
                arrows="to",
            )
        path = self.output_dir / f"ikg_column_lineage_output_{self.timestamp}.html"
        network.show(str(path))
        LOGGER.info("PyVis HTML exported to %s", path)
        return path


class Neo4jExporter:
    """Push lineage graph into Neo4j if credentials are supplied."""

    def __init__(self, config: Neo4jConfig) -> None:
        self.config = config
        if config.enabled and GraphDatabase is None:
            LOGGER.warning("neo4j driver is not installed; Neo4j export skipped.")
        self._driver = None

    def _ensure_driver(self):
        if not self.config.enabled or GraphDatabase is None:
            return None
        if self._driver is None:
            LOGGER.info("Connecting to Neo4j at %s", self.config.uri)
            self._driver = GraphDatabase.driver(
                self.config.uri,
                auth=(self.config.user, self.config.password),
            )
        return self._driver

    def export(self, records: List[LineageRecord]) -> None:
        driver = self._ensure_driver()
        if not driver:
            return

        create_nodes_query = """
        MERGE (t:Table {name: $target_table, schema: coalesce($target_schema, 'public')})
        MERGE (s:Table {name: $source_table, schema: coalesce($source_schema, 'public')})
        MERGE (tc:Column {name: $target_column})-[:BELONGS_TO]->(t)
        MERGE (sc:Column {name: $source_column})-[:BELONGS_TO]->(s)
        MERGE (sc)-[r:DERIVES]->(tc)
        SET r.logic = $logic, r.sql_file = $sql_file
        """

        with driver.session() as session:
            for record in records:
                session.run(
                    create_nodes_query,
                    target_table=record.target_table,
                    target_schema=record.target_schema,
                    target_column=record.target_column,
                    source_table=record.source_table,
                    source_schema=record.source_schema,
                    source_column=record.source_column,
                    logic=record.logic,
                    sql_file=record.sql_file,
                )
        LOGGER.info("Neo4j export completed.")

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


class LineageService:
    def __init__(
        self,
        gitlab_config: GitLabConfig,
        extractor: SQLLineageExtractor,
        metadata_client: Optional[GreenplumMetadataClient],
        neo4j_exporter: Optional[Neo4jExporter],
        output_manager: OutputManager,
    ) -> None:
        self.gitlab_config = gitlab_config
        self.extractor = extractor
        self.metadata_client = metadata_client
        self.neo4j_exporter = neo4j_exporter
        self.output_manager = output_manager
        self.fetcher = GitLabSQLFetcher(gitlab_config)

    def run(self, start_table: str) -> Dict[str, Path]:
        sql_files = self.fetcher.fetch_sql_files()
        if not sql_files:
            raise RuntimeError("No SQL files were discovered in the GitLab project.")

        table_to_paths: Dict[str, List[str]] = {}
        for path in sql_files.keys():
            table_name = Path(path).stem
            table_to_paths.setdefault(table_name.lower(), []).append(path)

        if start_table.lower() not in table_to_paths:
            raise ValueError(
                f"Start table {start_table} not found. Available tables: {list(table_to_paths)[:5]} ..."
            )

        visited_files: Set[str] = set()
        visited_tables: Set[str] = set()
        records: List[LineageRecord] = []
        queue: List[str] = table_to_paths[start_table.lower()].copy()

        while queue:
            file_path = queue.pop(0)
            if file_path in visited_files:
                continue
            visited_files.add(file_path)
            sql_text = sql_files[file_path]
            default_target_table = Path(file_path).stem
            default_target_schema = None
            LOGGER.info("Parsing lineage for %s", file_path)
            file_records, referenced_tables = self.extractor.extract_file_lineage(
                sql_text=sql_text,
                file_path=file_path,
                default_target_table=default_target_table,
                default_target_schema=default_target_schema,
            )
            records.extend(file_records)
            for table in referenced_tables:
                table_key = table.split(".")[-1].lower()
                if table_key in visited_tables:
                    continue
                visited_tables.add(table_key)
                next_paths = table_to_paths.get(table_key)
                if next_paths:
                    queue.extend(next_paths)

        if not records:
            raise RuntimeError("No lineage records were generated.")

        outputs: Dict[str, Path] = {}
        excel_path = self.output_manager.write_excel(records)
        if excel_path:
            outputs["excel"] = excel_path
        outputs["json"] = self.output_manager.write_json(records)
        graph_path = self.output_manager.write_graph(records)
        if graph_path:
            outputs["graph"] = graph_path
        pyvis_path = self.output_manager.write_pyvis(records)
        if pyvis_path:
            outputs["pyvis"] = pyvis_path

        if self.neo4j_exporter:
            self.neo4j_exporter.export(records)

        if self.metadata_client:
            self.metadata_client.close()
        if self.neo4j_exporter:
            self.neo4j_exporter.close()

        return outputs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IKG SQL Lineage Extraction Tool")
    parser.add_argument("--gitlab-url", default=os.getenv("GITLAB_URL", "https://gitlab.com"))
    parser.add_argument("--project-id", default=os.getenv("GITLAB_PROJECT_ID"), required=False)
    parser.add_argument("--token", default=os.getenv("GITLAB_PRIVATE_TOKEN"), required=False)
    parser.add_argument("--branch", default=os.getenv("GITLAB_BRANCH", "ikg-master"))
    parser.add_argument("--sql-folder", default=os.getenv("SQL_FOLDER", "dags/ikg/scripts/sql"))
    parser.add_argument("--exclude-folder", default=os.getenv("EXCLUDE_FOLDER", "ikg_new_fa_shhp_map"))
    parser.add_argument("--start-table", help="Root SQL/table name to start lineage tracing", required=True)
    parser.add_argument("--dialect", default=os.getenv("SQL_DIALECT", "postgres"))
    parser.add_argument("--output-dir", default=os.getenv("OUTPUT_DIR", "./lineage_output"))
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))

    # Greenplum metadata options
    parser.add_argument("--greenplum-host", default=os.getenv("GREENPLUM_HOST"))
    parser.add_argument("--greenplum-port", type=int, default=int(os.getenv("GREENPLUM_PORT", "5432")))
    parser.add_argument("--greenplum-db", default=os.getenv("GREENPLUM_DB"))
    parser.add_argument("--greenplum-user", default=os.getenv("GREENPLUM_USER"))
    parser.add_argument("--greenplum-password", default=os.getenv("GREENPLUM_PASSWORD"))
    parser.add_argument("--greenplum-schema", default=os.getenv("GREENPLUM_SCHEMA"))

    # Neo4j options
    parser.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI"))
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER"))
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD"))

    return parser


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main(args: Optional[List[str]] = None) -> int:
    parser = build_argument_parser()
    parsed = parser.parse_args(args=args)

    if not parsed.project_id:
        parser.error("GitLab project ID must be provided via --project-id or GITLAB_PROJECT_ID env var.")
    if not parsed.token:
        parser.error("GitLab private token must be provided via --token or GITLAB_PRIVATE_TOKEN env var.")

    configure_logging(parsed.log_level)

    gitlab_config = GitLabConfig(
        url=parsed.gitlab_url,
        private_token=parsed.token,
        project_id=parsed.project_id,
        branch=parsed.branch,
        sql_folder=parsed.sql_folder,
        exclude_folder=parsed.exclude_folder,
    )

    greenplum_config = GreenplumConfig(
        host=parsed.greenplum_host,
        port=parsed.greenplum_port,
        database=parsed.greenplum_db,
        user=parsed.greenplum_user,
        password=parsed.greenplum_password,
        default_schema=parsed.greenplum_schema,
    )

    metadata_client = GreenplumMetadataClient(greenplum_config) if greenplum_config.enabled else None

    extractor = SQLLineageExtractor(
        dialect=parsed.dialect,
        metadata_client=metadata_client,
    )

    neo4j_config = Neo4jConfig(
        uri=parsed.neo4j_uri,
        user=parsed.neo4j_user,
        password=parsed.neo4j_password,
    )
    neo4j_exporter = Neo4jExporter(neo4j_config) if neo4j_config.enabled else None

    output_manager = OutputManager(Path(parsed.output_dir))

    service = LineageService(
        gitlab_config=gitlab_config,
        extractor=extractor,
        metadata_client=metadata_client,
        neo4j_exporter=neo4j_exporter,
        output_manager=output_manager,
    )

    outputs = service.run(start_table=parsed.start_table)
    LOGGER.info("Lineage extraction completed. Outputs: %s", outputs)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
