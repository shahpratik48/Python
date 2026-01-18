import argparse
import base64
import datetime
import getpass
import logging
import os
import re
import sys
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
OUTPUT_PREFIX = "ikg_column_lineage_master_auto_refresh"
TARGET_SCHEMA = "sandbox_prj_smart_insights"
TARGET_TABLE = "ikg_column_lineage_master_auto_refresh"
TARGET_OWNER = "erd_gpdb_prj_smart_insights"
TARGET_READER = "erd_gpdb_prj_smart_insights_ro"
GITLAB_TOKEN_ENV = "IKG_GITLAB_TOKEN"
DB_PASSWORD_ENV = "IKG_DB_PASSWORD"
SQL_PATH_PARTS = Path(SQL_PATH).parts
LINEAGE_COLUMNS = [
    "filename",
    "filepath",
    "process",
    "target_table",
    "sub_target_schema",
    "sub_target_table",
    "target_column",
    "source_schema",
    "source_table",
    "source_column",
    "logic",
    "sql_process",
    "current_timestamp",
]


@dataclass(frozen=True)
class ColumnLineageRow:
    filename: str
    filepath: str
    process: str
    target_table: str
    sub_target_schema: Optional[str]
    sub_target_table: Optional[str]
    target_column: Optional[str]
    source_schema: Optional[str]
    source_table: Optional[str]
    source_column: Optional[str]
    logic: Optional[str]
    sql_process: Optional[str]
    current_timestamp: datetime.datetime


@dataclass(frozen=True)
class TargetTable:
    schema: Optional[str]
    table: Optional[str]
    is_temp: bool = False


@dataclass(frozen=True)
class ResolvedColumn:
    schema: Optional[str]
    table: Optional[str]
    column: Optional[str]


@dataclass
class TableRef:
    schema: Optional[str]
    table: Optional[str]
    alias: Optional[str]
    is_cte: bool = False
    is_subquery: bool = False
    columns: Optional[Dict[str, List[ResolvedColumn]]] = None

    def keys(self) -> List[str]:
        keys: List[str] = []
        if self.alias:
            keys.append(self.alias.lower())
        if self.table:
            keys.append(self.table.lower())
        return keys


@dataclass(frozen=True)
class ColumnRecord:
    target_column: Optional[str]
    source_schema: Optional[str]
    source_table: Optional[str]
    source_column: Optional[str]
    logic: Optional[str]
    sql_process: Optional[str]


@dataclass(frozen=True)
class StatementLineage:
    targets: List[TargetTable]
    records: List[ColumnRecord]


def derive_process(file_path: str) -> str:
    parts = Path(file_path).parts
    prefix_len = len(SQL_PATH_PARTS)
    if parts[:prefix_len] == SQL_PATH_PARTS and len(parts) > prefix_len:
        return parts[prefix_len]
    return ""


class TemplateNormalizer:
    TEMPLATE_PATTERN = re.compile(r"{{\s*([^{}]+?)\s*}}")
    ENV_PATTERN = re.compile(r"\$\{[^}]+\}")

    def __init__(self) -> None:
        self._mapping: Dict[str, str] = {}

    def normalize(self, sql_text: str) -> str:
        def template_replace(match: Match) -> str:
            token = f"TEMPLATE_TOKEN_{len(self._mapping)}"
            self._mapping[token] = match.group(0)
            return token

        def env_replace(match: Match) -> str:
            token = f"ENV_TOKEN_{len(self._mapping)}"
            self._mapping[token] = match.group(0)
            return token

        sanitized = self.TEMPLATE_PATTERN.sub(template_replace, sql_text)
        sanitized = self.ENV_PATTERN.sub(env_replace, sanitized)
        return sanitized

    def restore(self, text: Optional[str]) -> Optional[str]:
        if text is None:
            return None
        restored = text
        for token, raw in self._mapping.items():
            restored = restored.replace(token, raw)
        return restored

    def restore_identifier(self, identifier: Optional[str]) -> Optional[str]:
        return self.restore(identifier)

    def is_placeholder(self, identifier: Optional[str]) -> bool:
        if not identifier:
            return False
        return any(token in identifier for token in self._mapping.keys())


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


class MetadataResolver:
    def __init__(self, db_config: Dict[str, str]) -> None:
        self._db_config = db_config
        self._conn: Optional[psycopg2.extensions.connection] = None
        self._cache: Dict[Tuple[Optional[str], str], Set[str]] = {}

    def close(self) -> None:
        if self._conn and not self._conn.closed:
            self._conn.close()
        self._conn = None

    def resolve_candidates(
        self,
        column: str,
        candidates: Sequence[TableRef],
        normalizer: TemplateNormalizer,
    ) -> List[TableRef]:
        if not column:
            return []
        matches: List[TableRef] = []
        for table_ref in candidates:
            if table_ref.is_cte or table_ref.is_subquery or not table_ref.table:
                continue
            schema = table_ref.schema
            if normalizer.is_placeholder(schema) or normalizer.is_placeholder(table_ref.table):
                continue
            if self._column_exists(schema, table_ref.table, column):
                matches.append(table_ref)
        return matches

    def column_exists(
        self, schema: Optional[str], table: Optional[str], column: Optional[str]
    ) -> bool:
        if not table or not column:
            return False
        return self._column_exists(schema, table, column)

    def _column_exists(self, schema: Optional[str], table: str, column: str) -> bool:
        cache_key = (schema.lower() if schema else None, table.lower())
        if cache_key in self._cache:
            return column.lower() in self._cache[cache_key]
        columns = self._fetch_columns(schema, table)
        self._cache[cache_key] = columns
        return column.lower() in columns

    def _fetch_columns(self, schema: Optional[str], table: str) -> Set[str]:
        conn = self._connect()
        if not conn:
            return set()
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
                WHERE table_name = %s
            """
            params = (table,)
        try:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                return {row[0].lower() for row in cursor.fetchall()}
        except Exception as exc:  # noqa: BLE001
            logging.warning(
                "Failed metadata lookup for %s.%s: %s",
                schema,
                table,
                exc,
            )
            return set()

    def _connect(self) -> Optional[psycopg2.extensions.connection]:
        if self._conn and not self._conn.closed:
            return self._conn
        try:
            self._conn = psycopg2.connect(**self._db_config)
            return self._conn
        except Exception as exc:  # noqa: BLE001
            logging.warning("Metadata connection unavailable: %s", exc)
            return None


class SQLColumnParser:
    def __init__(self, metadata_resolver: MetadataResolver) -> None:
        self._metadata_resolver = metadata_resolver

    def extract_records(self, sql_text: str) -> List[StatementLineage]:
        cleaned = self._remove_sql_comments(sql_text)
        cleaned = self._strip_template_blocks(cleaned)
        normalized = self._strip_vendor_specific(cleaned)
        normalizer = TemplateNormalizer()
        sanitized = normalizer.normalize(normalized)
        sanitized_no_do, _ = self._strip_do_blocks(sanitized)
        statements = self._parse_sql(sanitized_no_do)
        if not statements:
            regex_records = self._regex_fallback(sanitized_no_do, normalizer)
            regex_targets = self._regex_targets(sanitized_no_do, normalizer)
            if not regex_targets:
                regex_targets = [TargetTable(schema=None, table=None)]
            if not regex_records:
                regex_records = [
                    ColumnRecord(
                        target_column=None,
                        source_schema=None,
                        source_table=None,
                        source_column=None,
                        logic=None,
                        sql_process=None,
                    )
                ]
            return [StatementLineage(targets=regex_targets, records=regex_records)]
        statement_results: List[StatementLineage] = []
        for statement in statements:
            statement_targets = self._extract_targets(statement)
            restored_targets = [
                TargetTable(
                    schema=normalizer.restore_identifier(target.schema),
                    table=normalizer.restore_identifier(target.table),
                    is_temp=target.is_temp,
                )
                for target in statement_targets
            ]
            statement_records = self._extract_statement_records(statement, normalizer)
            if not restored_targets:
                restored_targets = [TargetTable(schema=None, table=None)]
            if not statement_records:
                statement_records = [
                    ColumnRecord(
                        target_column=None,
                        source_schema=None,
                        source_table=None,
                        source_column=None,
                        logic=None,
                        sql_process=None,
                    )
                ]
            statement_results.append(
                StatementLineage(
                    targets=restored_targets,
                    records=self._deduplicate_records(statement_records),
                )
            )
        return statement_results

    def _parse_sql(self, sql_text: str) -> List[exp.Expression]:
        if not sql_text.strip():
            return []
        try:
            parsed = sqlglot.parse(sql_text, read="postgres", error_level="ignore")
        except sqlglot.errors.ParseError as exc:
            logging.warning("sqlglot failed to parse SQL: %s", exc)
            return []
        return [statement for statement in parsed if statement is not None]

    def _remove_sql_comments(self, sql_text: str) -> str:
        no_block = re.sub(r"/\*.*?\*/", "", sql_text, flags=re.S)
        no_inline = re.sub(r"--.*?$", "", no_block, flags=re.M)
        return no_inline

    def _strip_template_blocks(self, sql_text: str) -> str:
        patterns = [
            r"\{\#.*?\#\}",
            r"\{\%.*?\%\}",
        ]
        cleaned = sql_text
        for pattern in patterns:
            cleaned = re.sub(pattern, " ", cleaned, flags=re.S)
        return cleaned

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

    def _extract_targets(self, statement: exp.Expression) -> List[TargetTable]:
        targets: List[TargetTable] = []
        if isinstance(statement, exp.Create):
            table_expr = statement.this
            target = self._target_from_table(table_expr, statement)
            if target:
                targets.append(target)
        elif isinstance(statement, exp.Insert):
            table_expr = statement.this
            target = self._target_from_table(table_expr, statement)
            if target:
                targets.append(target)
        elif isinstance(statement, exp.Select) and statement.args.get("into"):
            into_expr = statement.args.get("into")
            target = self._target_from_table(into_expr, statement)
            if target:
                targets.append(target)
        return targets

    def _target_from_table(
        self, table_expr: Optional[exp.Expression], statement: exp.Expression
    ) -> Optional[TargetTable]:
        if isinstance(table_expr, exp.Table):
            schema = table_expr.db
            table = table_expr.name
            is_temp = bool(statement.args.get("temporary") or statement.args.get("temp"))
            kind = statement.args.get("kind")
            if isinstance(kind, str) and "temp" in kind.lower():
                is_temp = True
            return TargetTable(schema=schema, table=table, is_temp=is_temp)
        if isinstance(table_expr, exp.Identifier):
            return TargetTable(schema=None, table=table_expr.name, is_temp=False)
        return None

    def _extract_statement_records(
        self, statement: exp.Expression, normalizer: TemplateNormalizer
    ) -> List[ColumnRecord]:
        target_columns_override = self._extract_insert_columns(statement)
        query = self._extract_statement_query(statement)
        if query is None:
            return self._extract_create_definition_records(statement, normalizer)
        return self._extract_query_records(query, normalizer, target_columns_override)

    def _extract_statement_query(
        self, statement: exp.Expression
    ) -> Optional[exp.Expression]:
        if isinstance(statement, exp.Create):
            return (
                statement.args.get("expression")
                or statement.args.get("query")
                or statement.args.get("select")
            )
        if isinstance(statement, exp.Insert):
            return statement.args.get("expression")
        if isinstance(statement, exp.Select) and statement.args.get("into"):
            return statement
        return None

    def _extract_insert_columns(self, statement: exp.Expression) -> Optional[List[exp.Expression]]:
        if isinstance(statement, exp.Insert):
            columns = statement.args.get("columns")
            if columns:
                return list(columns)
        return None

    def _extract_create_definition_records(
        self, statement: exp.Expression, normalizer: TemplateNormalizer
    ) -> List[ColumnRecord]:
        records: List[ColumnRecord] = []
        if isinstance(statement, exp.Create):
            for coldef in statement.find_all(exp.ColumnDef):
                target_column = coldef.name
                records.append(
                    ColumnRecord(
                        target_column=target_column,
                        source_schema=None,
                        source_table=None,
                        source_column=None,
                        logic=normalizer.restore(coldef.sql(dialect="postgres")),
                        sql_process="create",
                    )
                )
        if not records:
            records.append(
                ColumnRecord(
                    target_column=None,
                    source_schema=None,
                    source_table=None,
                    source_column=None,
                    logic=None,
                    sql_process="create",
                )
            )
        return records

    def _extract_query_records(
        self,
        query: exp.Expression,
        normalizer: TemplateNormalizer,
        target_columns_override: Optional[List[exp.Expression]] = None,
    ) -> List[ColumnRecord]:
        visited: Set[int] = set()
        records: List[ColumnRecord] = []
        self._extract_query_records_recursive(
            query=query,
            normalizer=normalizer,
            target_columns_override=target_columns_override,
            visited=visited,
            include_select=True,
            in_cte=False,
            records=records,
        )
        return self._deduplicate_records(records)

    def _extract_query_records_recursive(
        self,
        query: exp.Expression,
        normalizer: TemplateNormalizer,
        target_columns_override: Optional[List[exp.Expression]],
        visited: Set[int],
        include_select: bool,
        in_cte: bool,
        records: List[ColumnRecord],
    ) -> None:
        if query is None or id(query) in visited:
            return
        visited.add(id(query))

        if isinstance(query, exp.Subquery):
            self._extract_query_records_recursive(
                query.this,
                normalizer,
                target_columns_override,
                visited,
                False,
                in_cte,
                records,
            )
            return

        if isinstance(query, exp.SetOperation):
            self._extract_query_records_recursive(
                query.this,
                normalizer,
                target_columns_override,
                visited,
                include_select,
                in_cte,
                records,
            )
            self._extract_query_records_recursive(
                query.expression,
                normalizer,
                target_columns_override,
                visited,
                include_select,
                in_cte,
                records,
            )
            return

        if not isinstance(query, exp.Select):
            return

        context = self._build_context(query, normalizer)
        if include_select:
            select_records = self._extract_select_records(
                query, context, normalizer, target_columns_override
            )
            records.extend(select_records)
        records.extend(self._extract_clause_records(query, context, normalizer, in_cte))

        with_clause = query.args.get("with")
        if with_clause is not None:
            for cte in with_clause.expressions:
                self._extract_query_records_recursive(
                    cte.this, normalizer, None, visited, False, True, records
                )

        for subquery in query.find_all(exp.Subquery):
            self._extract_query_records_recursive(
                subquery.this, normalizer, None, visited, False, in_cte, records
            )

    def _extract_select_records(
        self,
        query: exp.Select,
        context: "QueryContext",
        normalizer: TemplateNormalizer,
        target_columns_override: Optional[List[exp.Expression]] = None,
    ) -> List[ColumnRecord]:
        records: List[ColumnRecord] = []
        projections = list(query.expressions)
        for index, projection in enumerate(projections):
            if self._is_star_projection(projection):
                records.extend(
                    self._expand_star_projection(projection, context, normalizer)
                )
                continue
            target_column = self._output_column_name(
                projection, target_columns_override, index
            )
            logic = normalizer.restore(projection.sql(dialect="postgres"))
            sources = self._resolve_expression_sources(
                projection, query, context, normalizer
            )
            if not sources:
                records.append(
                    ColumnRecord(
                        target_column=target_column,
                        source_schema=None,
                        source_table=None,
                        source_column=None,
                        logic=logic,
                        sql_process="select",
                    )
                )
            else:
                for resolved in sources:
                    records.append(
                        ColumnRecord(
                            target_column=target_column,
                            source_schema=normalizer.restore_identifier(resolved.schema),
                            source_table=normalizer.restore_identifier(resolved.table),
                            source_column=normalizer.restore_identifier(resolved.column),
                            logic=logic,
                            sql_process="select",
                        )
                    )
        return records

    def _extract_clause_records(
        self,
        query: exp.Select,
        context: "QueryContext",
        normalizer: TemplateNormalizer,
        in_cte: bool,
    ) -> List[ColumnRecord]:
        records: List[ColumnRecord] = []
        suffix = "-with" if in_cte else ""
        where_clause = query.args.get("where")
        if where_clause is not None:
            where_logic = f"where {where_clause.this.sql(dialect='postgres')}"
            records.extend(
                self._records_for_expression(
                    where_clause.this,
                    query,
                    context,
                    normalizer,
                    f"where{suffix}",
                    include_target=False,
                    logic_override=normalizer.restore(where_logic),
                )
            )

        for join in query.args.get("joins") or []:
            on_clause = join.args.get("on")
            if on_clause is None and join.args.get("using") is None:
                continue
            join_expression = on_clause or join.args.get("using")
            join_logic = self._join_logic(query, join, normalizer, in_cte)
            records.extend(
                self._records_for_expression(
                    join_expression,
                    query,
                    context,
                    normalizer,
                    f"join{suffix}",
                    include_target=False,
                    logic_override=join_logic,
                )
            )

        having_clause = query.args.get("having")
        if having_clause is not None:
            having_logic = f"having {having_clause.this.sql(dialect='postgres')}"
            records.extend(
                self._records_for_expression(
                    having_clause.this,
                    query,
                    context,
                    normalizer,
                    f"having{suffix}",
                    include_target=False,
                    logic_override=normalizer.restore(having_logic),
                )
            )

        return records

    def _records_for_expression(
        self,
        expression: exp.Expression,
        query: exp.Select,
        context: "QueryContext",
        normalizer: TemplateNormalizer,
        sql_process: str,
        include_target: bool,
        logic_override: Optional[str] = None,
    ) -> List[ColumnRecord]:
        records: List[ColumnRecord] = []
        if expression is None:
            return records
        logic = logic_override or normalizer.restore(expression.sql(dialect="postgres"))
        sources = self._resolve_expression_sources(expression, query, context, normalizer)
        for resolved in sources:
            target_column = (
                normalizer.restore_identifier(resolved.column) if include_target else None
            )
            records.append(
                ColumnRecord(
                    target_column=target_column,
                    source_schema=normalizer.restore_identifier(resolved.schema),
                    source_table=normalizer.restore_identifier(resolved.table),
                    source_column=normalizer.restore_identifier(resolved.column),
                    logic=logic,
                    sql_process=sql_process,
                )
            )
        return records

    def _join_logic(
        self,
        query: exp.Select,
        join: exp.Join,
        normalizer: TemplateNormalizer,
        in_cte: bool,
    ) -> str:
        on_clause = join.args.get("on")
        if in_cte:
            if on_clause is not None:
                return normalizer.restore(f"on {on_clause.sql(dialect='postgres')}")
            return normalizer.restore(join.sql(dialect="postgres"))
        from_clause = query.args.get("from")
        left_expr = None
        if from_clause and from_clause.expressions:
            left_expr = from_clause.expressions[0]
        join_sql = join.sql(dialect="postgres")
        if left_expr is None:
            return normalizer.restore(join_sql)
        left_sql = left_expr.sql(dialect="postgres")
        return normalizer.restore(f"{left_sql} {join_sql}")

    def _resolve_expression_sources(
        self,
        expression: exp.Expression,
        query: exp.Select,
        context: "QueryContext",
        normalizer: TemplateNormalizer,
    ) -> List[ResolvedColumn]:
        sources: List[ResolvedColumn] = []
        for column in expression.find_all(exp.Column):
            if self._is_in_subquery(column, query):
                continue
            sources.extend(self._resolve_column(column, context, normalizer))
        for subquery in expression.find_all(exp.Subquery):
            if subquery.this is query:
                continue
            output_map = self._build_output_map(subquery.this, normalizer)
            for resolved_list in output_map.values():
                sources.extend(resolved_list)
        unique: Dict[Tuple[Optional[str], Optional[str], Optional[str]], ResolvedColumn] = {}
        for source in sources:
            key = (source.schema, source.table, source.column)
            unique[key] = source
        return list(unique.values())

    def _resolve_column(
        self,
        column: exp.Column,
        context: "QueryContext",
        normalizer: TemplateNormalizer,
    ) -> List[ResolvedColumn]:
        column_name = column.name
        if column.table:
            table_ref = context.table_refs.get(column.table.lower())
            if table_ref:
                return self._resolve_table_ref_column(
                    table_ref, column_name, normalizer
                )
            if self._metadata_resolver.column_exists(None, column.table, column_name):
                return [ResolvedColumn(schema=None, table=column.table, column=column_name)]
            return [ResolvedColumn(schema=None, table=None, column=column_name)]
        if len(context.base_tables) == 1:
            base = context.base_tables[0]
            return [ResolvedColumn(schema=base.schema, table=base.table, column=column_name)]

        cte_matches = self._resolve_cte_subquery_column(context, column_name)
        if cte_matches:
            return cte_matches

        candidates = self._metadata_resolver.resolve_candidates(
            column_name, context.base_tables, normalizer
        )
        if candidates:
            return [
                ResolvedColumn(
                    schema=ref.schema,
                    table=ref.table,
                    column=column_name,
                )
                for ref in candidates
            ]

        if context.base_tables:
            return [
                ResolvedColumn(schema=ref.schema, table=ref.table, column=column_name)
                for ref in context.base_tables
            ]
        return [ResolvedColumn(schema=None, table=None, column=column_name)]

    def _resolve_cte_subquery_column(
        self, context: "QueryContext", column_name: str
    ) -> List[ResolvedColumn]:
        matches: List[ResolvedColumn] = []
        for table_ref in context.table_refs.values():
            if not (table_ref.is_cte or table_ref.is_subquery):
                continue
            if not table_ref.columns:
                continue
            key = column_name.lower()
            if key in table_ref.columns:
                matches.extend(table_ref.columns[key])
        return matches

    def _resolve_table_ref_column(
        self,
        table_ref: TableRef,
        column_name: str,
        normalizer: TemplateNormalizer,
    ) -> List[ResolvedColumn]:
        if table_ref.is_cte or table_ref.is_subquery:
            if table_ref.columns:
                mapped = table_ref.columns.get(column_name.lower())
                if mapped:
                    return mapped
            return [ResolvedColumn(schema=None, table=None, column=column_name)]
        return [ResolvedColumn(schema=table_ref.schema, table=table_ref.table, column=column_name)]

    def _build_context(
        self,
        query: exp.Select,
        normalizer: TemplateNormalizer,
        visited: Optional[Set[int]] = None,
    ) -> "QueryContext":
        if visited is None:
            visited = set()
        cte_maps: Dict[str, Dict[str, List[ResolvedColumn]]] = {}
        with_clause = query.args.get("with")
        if with_clause is not None:
            for cte in with_clause.expressions:
                cte_name = (cte.alias_or_name or "").lower()
                if not cte_name:
                    continue
                output_map = self._build_output_map(cte.this, normalizer, visited)
                cte_maps[cte_name] = output_map

        table_refs: Dict[str, TableRef] = {}
        base_tables: List[TableRef] = []
        for source in self._iter_source_expressions(query):
            table_ref = self._table_ref_from_expression(
                source, normalizer, cte_maps, visited
            )
            if table_ref is None:
                continue
            for key in table_ref.keys():
                table_refs[key] = table_ref
            if not table_ref.is_cte and not table_ref.is_subquery:
                base_tables.append(table_ref)
        return QueryContext(
            table_refs=table_refs,
            base_tables=base_tables,
            cte_maps=cte_maps,
        )

    def _iter_source_expressions(self, query: exp.Select) -> Iterable[exp.Expression]:
        from_clause = query.args.get("from")
        if from_clause is not None:
            for expr in from_clause.expressions:
                yield expr
        for join in query.args.get("joins") or []:
            source = join.this
            if isinstance(source, exp.Lateral):
                source = source.this
            if source is not None:
                yield source

    def _table_ref_from_expression(
        self,
        expr: exp.Expression,
        normalizer: TemplateNormalizer,
        cte_maps: Dict[str, Dict[str, List[ResolvedColumn]]],
        visited: Optional[Set[int]] = None,
    ) -> Optional[TableRef]:
        if isinstance(expr, exp.Table):
            schema = expr.db
            table = expr.name
            alias = expr.alias_or_name
            is_cte = table.lower() in cte_maps if table else False
            columns = cte_maps.get(table.lower()) if is_cte and table else None
            return TableRef(
                schema=schema,
                table=table,
                alias=alias,
                is_cte=is_cte,
                columns=columns,
            )
        if isinstance(expr, exp.Subquery):
            alias = expr.alias_or_name
            output_map = self._build_output_map(expr.this, normalizer, visited)
            return TableRef(
                schema=None,
                table=alias,
                alias=alias,
                is_subquery=True,
                columns=output_map,
            )
        if isinstance(expr, exp.Select):
            output_map = self._build_output_map(expr, normalizer, visited)
            return TableRef(
                schema=None,
                table=None,
                alias=None,
                is_subquery=True,
                columns=output_map,
            )
        return None

    def _build_output_map(
        self,
        query: exp.Expression,
        normalizer: TemplateNormalizer,
        visited: Optional[Set[int]] = None,
    ) -> Dict[str, List[ResolvedColumn]]:
        if query is None:
            return {}
        if visited is None:
            visited = set()
        if id(query) in visited:
            return {}
        visited.add(id(query))
        output_list = self._build_output_list(query, normalizer, visited)
        output_map: Dict[str, List[ResolvedColumn]] = {}
        for output in output_list:
            if not output[0]:
                continue
            output_map.setdefault(output[0].lower(), [])
            for resolved in output[1]:
                output_map[output[0].lower()].append(resolved)
        return output_map

    def _build_output_list(
        self,
        query: exp.Expression,
        normalizer: TemplateNormalizer,
        visited: Set[int],
    ) -> List[Tuple[str, List[ResolvedColumn]]]:
        if isinstance(query, exp.Subquery):
            return self._build_output_list(query.this, normalizer, visited)
        if isinstance(query, exp.SetOperation):
            left_outputs = self._build_output_list(query.this, normalizer, visited)
            right_outputs = self._build_output_list(query.expression, normalizer, visited)
            combined: List[Tuple[str, List[ResolvedColumn]]] = []
            for idx, left in enumerate(left_outputs):
                right_sources = right_outputs[idx][1] if idx < len(right_outputs) else []
                combined.append((left[0], left[1] + right_sources))
            return combined
        if not isinstance(query, exp.Select):
            return []

        context = self._build_context(query, normalizer, visited)
        outputs: List[Tuple[str, List[ResolvedColumn]]] = []
        for index, projection in enumerate(query.expressions):
            if self._is_star_projection(projection):
                for star_output in self._expand_star_projection(
                    projection, context, normalizer, map_only=True
                ):
                    outputs.append((star_output.target_column or "", [ResolvedColumn(
                        star_output.source_schema, star_output.source_table, star_output.source_column
                    )]))
                continue
            target_name = self._output_column_name(projection, None, index)
            sources = self._resolve_expression_sources(
                projection, query, context, normalizer
            )
            outputs.append((target_name or "", sources))
        return outputs

    def _output_column_name(
        self,
        projection: exp.Expression,
        target_columns_override: Optional[List[exp.Expression]],
        index: int,
    ) -> Optional[str]:
        if target_columns_override and index < len(target_columns_override):
            override = target_columns_override[index]
            if isinstance(override, exp.Column):
                return override.name
            if isinstance(override, exp.Identifier):
                return override.name
            return override.sql(dialect="postgres")
        if isinstance(projection, exp.Alias):
            return projection.alias
        if projection.alias_or_name:
            return projection.alias_or_name
        if isinstance(projection, exp.Column):
            return projection.name
        return projection.sql(dialect="postgres")

    def _is_star_projection(self, projection: exp.Expression) -> bool:
        if isinstance(projection, exp.Star):
            return True
        if isinstance(projection, exp.Column) and projection.name == "*":
            return True
        return False

    def _expand_star_projection(
        self,
        projection: exp.Expression,
        context: "QueryContext",
        normalizer: TemplateNormalizer,
        map_only: bool = False,
    ) -> List[ColumnRecord]:
        qualifier = None
        projection_sql = projection.sql(dialect="postgres")
        if "." in projection_sql:
            qualifier = projection_sql.split(".", 1)[0].strip('"')

        table_refs: List[TableRef] = []
        if qualifier:
            ref = context.table_refs.get(qualifier.lower())
            if ref:
                table_refs = [ref]
        elif context.base_tables:
            table_refs = context.base_tables
        else:
            seen_ids: Set[int] = set()
            for ref in context.table_refs.values():
                if id(ref) in seen_ids:
                    continue
                seen_ids.add(id(ref))
                table_refs.append(ref)

        expanded_records: List[ColumnRecord] = []
        for table_ref in table_refs:
            if not table_ref.table:
                continue
            if table_ref.is_cte or table_ref.is_subquery:
                if not table_ref.columns:
                    continue
                for column_name, resolved_list in table_ref.columns.items():
                    for resolved in resolved_list:
                        expanded_records.append(
                            ColumnRecord(
                                target_column=column_name,
                                source_schema=normalizer.restore_identifier(resolved.schema),
                                source_table=normalizer.restore_identifier(resolved.table),
                                source_column=normalizer.restore_identifier(resolved.column),
                                logic=normalizer.restore(projection_sql),
                                sql_process="select",
                            )
                        )
                continue

            columns = self._metadata_columns(table_ref, normalizer)
            if not columns:
                expanded_records.append(
                    ColumnRecord(
                        target_column="*",
                        source_schema=normalizer.restore_identifier(table_ref.schema),
                        source_table=normalizer.restore_identifier(table_ref.table),
                        source_column="*",
                        logic=normalizer.restore(projection_sql),
                        sql_process="select",
                    )
                )
                continue
            for column_name in columns:
                expanded_records.append(
                    ColumnRecord(
                        target_column=column_name,
                        source_schema=normalizer.restore_identifier(table_ref.schema),
                        source_table=normalizer.restore_identifier(table_ref.table),
                        source_column=column_name,
                        logic=normalizer.restore(projection_sql),
                        sql_process="select",
                    )
                )
        return expanded_records

    def _metadata_columns(
        self, table_ref: TableRef, normalizer: TemplateNormalizer
    ) -> List[str]:
        if not table_ref.table:
            return []
        schema = table_ref.schema
        if normalizer.is_placeholder(schema) or normalizer.is_placeholder(table_ref.table):
            return []
        cache_key = (schema.lower() if schema else None, table_ref.table.lower())
        columns = self._metadata_resolver._cache.get(cache_key)
        if columns is None:
            columns = self._metadata_resolver._fetch_columns(schema, table_ref.table)
            self._metadata_resolver._cache[cache_key] = columns
        return sorted(columns)

    def _is_in_subquery(self, column: exp.Column, query: exp.Select) -> bool:
        ancestor = column.find_ancestor(exp.Subquery)
        if ancestor is None:
            return False
        return ancestor.this is not query

    def _regex_fallback(
        self, sanitized_sql: str, normalizer: TemplateNormalizer
    ) -> List[ColumnRecord]:
        pattern = re.compile(
            r"""(?ix)
            (?:select)\s+(?P<columns>.+?)\s+from\s+(?P<table>[a-z0-9_.]+)
            """,
        )
        records: List[ColumnRecord] = []
        for match in pattern.finditer(sanitized_sql):
            table_name = match.group("table")
            columns = [col.strip() for col in match.group("columns").split(",")]
            for col in columns:
                column_name = col.split()[-1] if " " in col else col
                records.append(
                    ColumnRecord(
                        target_column=normalizer.restore(column_name),
                        source_schema=None,
                        source_table=normalizer.restore(table_name),
                        source_column=normalizer.restore(column_name),
                        logic=normalizer.restore(col),
                        sql_process="select",
                    )
                )
        return records

    def _regex_targets(
        self, sanitized_sql: str, normalizer: TemplateNormalizer
    ) -> List[TargetTable]:
        targets: List[TargetTable] = []
        pattern = re.compile(
            r"""(?ix)
            (?:create\s+(?:temporary|temp)?\s*table|insert\s+into)\s+
            (?:if\s+not\s+exists\s+)?(?:(?P<schema>[a-z0-9_]+)\.)?(?P<table>[a-z0-9_]+)
            """,
        )
        seen: Set[Tuple[Optional[str], Optional[str]]] = set()
        for match in pattern.finditer(sanitized_sql):
            schema = normalizer.restore(match.group("schema"))
            table = normalizer.restore(match.group("table"))
            key = (schema, table)
            if key in seen:
                continue
            seen.add(key)
            targets.append(TargetTable(schema=schema, table=table))
        return targets

    def _deduplicate_records(
        self, records: Sequence[ColumnRecord]
    ) -> List[ColumnRecord]:
        unique: Dict[
            Tuple[
                Optional[str],
                Optional[str],
                Optional[str],
                Optional[str],
                Optional[str],
                Optional[str],
            ],
            ColumnRecord,
        ] = {}
        for record in records:
            key = (
                record.target_column,
                record.source_schema,
                record.source_table,
                record.source_column,
                record.logic,
                record.sql_process,
            )
            unique[key] = record
        return list(unique.values())


@dataclass
class QueryContext:
    table_refs: Dict[str, TableRef]
    base_tables: List[TableRef]
    cte_maps: Dict[str, Dict[str, List[ResolvedColumn]]]


class LineageBuilder:
    def __init__(self, fetcher: GitLabSQLFetcher, parser: SQLColumnParser) -> None:
        self._fetcher = fetcher
        self._parser = parser

    def build(
        self,
        progress_callback: Optional[Callable[[Sequence[ColumnLineageRow]], None]] = None,
        run_timestamp: Optional[datetime.datetime] = None,
    ) -> List[ColumnLineageRow]:
        timestamp = run_timestamp or datetime.datetime.utcnow()
        rows: List[ColumnLineageRow] = []
        for file_path in self._fetcher.iter_sql_paths():
            logging.info("Processing %s", file_path)
            try:
                sql_text = self._fetcher.fetch_sql(file_path)
                statement_results = self._parser.extract_records(sql_text)
                filename = Path(file_path).name
                target_table = Path(filename).stem
                process = derive_process(file_path)
                for result in statement_results:
                    for record in result.records:
                        is_clause = (
                            record.sql_process in {"join", "where", "having"}
                            or (record.sql_process or "").endswith("-with")
                        )
                        targets = (
                            [TargetTable(schema=None, table=None)]
                            if is_clause
                            else result.targets
                        )
                        for target in targets:
                            rows.append(
                                ColumnLineageRow(
                                    filename=filename,
                                    filepath=file_path,
                                    process=process,
                                    target_table=None if is_clause else target_table,
                                    sub_target_schema=target.schema,
                                    sub_target_table=target.table,
                                    target_column=None if is_clause else record.target_column,
                                    source_schema=record.source_schema,
                                    source_table=record.source_table,
                                    source_column=record.source_column,
                                    logic=record.logic,
                                    sql_process=record.sql_process,
                                    current_timestamp=timestamp,
                                )
                            )
            except Exception as exc:  # noqa: BLE001
                logging.exception("Failed to process %s: %s", file_path, exc)
            finally:
                if progress_callback:
                    progress_callback(rows)
        return rows


class DatabaseUploader:
    def __init__(self, db_config: Dict[str, str]) -> None:
        self._db_config = db_config

    def refresh_table(self, rows: Sequence[ColumnLineageRow]) -> None:
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
                            sub_target_schema TEXT,
                            sub_target_table TEXT,
                            target_column TEXT,
                            source_schema TEXT,
                            source_table TEXT,
                            source_column TEXT,
                            logic TEXT,
                            sql_process TEXT,
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
                        row.sub_target_schema,
                        row.sub_target_table,
                        row.target_column,
                        row.source_schema,
                        row.source_table,
                        row.source_column,
                        row.logic,
                        row.sql_process,
                        row.current_timestamp,
                    )
                    for row in rows
                ]
                if values:
                    execute_values(
                        cur,
                        sql.SQL(
                            'INSERT INTO {}.{} (filename, filepath, process, target_table, sub_target_schema, sub_target_table, target_column, source_schema, source_table, source_column, logic, sql_process, "current_timestamp") VALUES %s'
                        ).format(
                            sql.Identifier(TARGET_SCHEMA), sql.Identifier(TARGET_TABLE)
                        ),
                        values,
                    )
                cur.execute(
                    sql.SQL("ALTER TABLE {}.{} OWNER TO {}").format(
                        sql.Identifier(TARGET_SCHEMA),
                        sql.Identifier(TARGET_TABLE),
                        sql.Identifier(TARGET_OWNER),
                    )
                )
                cur.execute(
                    sql.SQL("GRANT SELECT ON {}.{} TO {}").format(
                        sql.Identifier(TARGET_SCHEMA),
                        sql.Identifier(TARGET_TABLE),
                        sql.Identifier(TARGET_READER),
                    )
                )
            conn.commit()


def rows_to_dataframe(rows: Sequence[ColumnLineageRow]) -> pd.DataFrame:
    data = [
        {
            "filename": row.filename,
            "filepath": row.filepath,
            "process": row.process,
            "target_table": row.target_table,
            "sub_target_schema": row.sub_target_schema,
            "sub_target_table": row.sub_target_table,
            "target_column": row.target_column,
            "source_schema": row.source_schema,
            "source_table": row.source_table,
            "source_column": row.source_column,
            "logic": row.logic,
            "sql_process": row.sql_process,
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


def _read_secret(
    label: str,
    env_key: str,
    arg_value: Optional[str],
) -> str:
    if arg_value:
        return arg_value
    env_value = os.environ.get(env_key)
    if env_value:
        return env_value
    if sys.stdin.isatty():
        return getpass.getpass(label)
    try:
        return input(label)
    except EOFError as exc:
        raise RuntimeError(
            f"Missing {env_key}. Provide CLI args or set the env var."
        ) from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IKG Column Lineage Auto Refresh")
    parser.add_argument(
        "--gitlab-token",
        dest="gitlab_token",
        help=f"GitLab token (or set {GITLAB_TOKEN_ENV}).",
    )
    parser.add_argument(
        "--db-password",
        dest="db_password",
        help=f"DB password (or set {DB_PASSWORD_ENV}).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.DEBUG),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    private_token = _read_secret(
        f"Enter your private token (or set {GITLAB_TOKEN_ENV}): ",
        GITLAB_TOKEN_ENV,
        args.gitlab_token,
    )
    db_password = _read_secret(
        f"Enter Password for DB User (or set {DB_PASSWORD_ENV}): ",
        DB_PASSWORD_ENV,
        args.db_password,
    )

    exclude_folders = [folder for folder in EXCLUDE_FOLDER.split() if folder]
    fetcher = GitLabSQLFetcher(private_token=private_token, exclude_folders=exclude_folders)
    db_config = {
        "host": "greenplum-rdsp.zur.swissbank.com",
        "port": "5432",
        "dbname": "gprdsp",
        "user": "ds_rdsp_dev",
        "password": db_password,
    }
    metadata_resolver = MetadataResolver(db_config)
    parser = SQLColumnParser(metadata_resolver)
    builder = LineageBuilder(fetcher, parser)
    run_timestamp = datetime.datetime.utcnow()
    output_file = f"{OUTPUT_PREFIX}_{run_timestamp.strftime('%Y%m%d%H%M%S')}.xlsx"
    write_to_excel(rows_to_dataframe([]), output_path=output_file)

    def flush_excel(current_rows: Sequence[ColumnLineageRow]) -> None:
        df_snapshot = rows_to_dataframe(current_rows)
        write_to_excel(df_snapshot, output_path=output_file)

    rows = builder.build(progress_callback=flush_excel, run_timestamp=run_timestamp)
    logging.info("Captured %d column lineage rows", len(rows))

    uploader = DatabaseUploader(db_config)
    uploader.refresh_table(rows)
    metadata_resolver.close()


if __name__ == "__main__":
    main()
