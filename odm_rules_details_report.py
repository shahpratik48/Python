#!/usr/bin/env python3
"""
Generate the ODM rules details Excel report from ODM SQL assets hosted in GitLab
and upload the resulting data into sandbox_prj_smart_insights.odm_rules_details.
"""

from __future__ import annotations

import base64
import datetime as dt
import getpass
import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import gitlab  # type: ignore
import pandas as pd
import psycopg2  # type: ignore
from psycopg2.extras import execute_values  # type: ignore
import sqlparse  # type: ignore
from sqlglot import exp, parse_one  # type: ignore
from sqlglot.errors import ParseError  # type: ignore

GITLAB_URL = "https://devcloud.ubs.net"
ODM_PROJECT_PATH = (
    "ubs/gwma/smart-technology-and-analytics/staat-data-science/"
    "staat-ds-genesis/genesis-platform/odm-dags"
)
BRANCH = "odm-master"
SQL_PATH = "dags/odm/script/sql"

IGNORE_FILES = {"aa_nlg_review.sql"}
IGNORE_FOLDERS = {"odm_process"}

REPORT_COLUMNS: Sequence[str] = (
    "filename",
    "target_type",
    "insight_type",
    "profile_table",
    "target_key",
    "metric_name",
    "metric_type",
    "metric_val",
    "run_id",
    "insight_date_time",
    "insight_category",
    "insight_att_name",
    "insight_att_val",
    "rule_column",
    "logic",
    "file_path",
    "dependency",
    "is_active",
    "generated_at",
)

PLACEHOLDER_PATTERN = re.compile(r"\{\{\s*([^\{\}]+?)\s*\}\}")


@dataclass
class ParsedRow:
    data: Dict[str, str]
    source_statement: str
    source_file: str


def mask_placeholders(sql_text: str) -> Tuple[str, Dict[str, str]]:
    """Replace Jinja placeholders with SQL-safe tokens and keep a reverse map."""
    mapping: Dict[str, str] = {}

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        token = f"__JINJA_{len(mapping)}__"
        mapping[token] = match.group(0)
        return token

    masked_sql = PLACEHOLDER_PATTERN.sub(_replace, sql_text)
    return masked_sql, mapping


def unmask_placeholders(text: str, mapping: Dict[str, str]) -> str:
    """Restore original Jinja placeholders in the provided text."""
    result = text
    for token, placeholder in mapping.items():
        result = result.replace(token, placeholder)
    return result


def normalize_value(column_name: str, value: str) -> str:
    """Apply column-specific normalization rules."""
    text = (value or "").strip()
    if column_name in {"insight_type", "target_type"}:
        if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
            text = text[1:-1].strip()
    return text


def normalize_report_column_name(column_name: str) -> str:
    """Normalize report column names to match expected output schema."""
    normalized = (column_name or "").strip().lower()
    if normalized == "profile_column":
        return "rule_column"
    return normalized


def strip_quotes(value: str) -> str:
    """Remove single/double quotes surrounding a literal value."""
    text = (value or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1]
    return text.strip()


def normalize_profile_table_value(value: str) -> str:
    """Remove schema parameter prefix from profile table references."""
    if not value:
        return value
    cleaned = value.strip()
    cleaned = re.sub(
        r"^\{\{\s*params\.IKG_SCHEMA\s*\}\}\.",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return strip_quotes(cleaned)


def normalize_filter_text(value: str) -> str:
    """Normalize text for comparison in exclusion filters."""
    return strip_quotes((value or "")).strip().lower()


EXCLUDED_PROFILE_TABLES = {
    normalize_filter_text(normalize_profile_table_value("f1_rec_profile_curr_ikg AS fl"))
}
EXCLUDED_TARGET_KEYS = {
    normalize_filter_text("cast(ip_cli_i as text)")
}


def extract_insert_columns_from_text(sql_text: str) -> List[str]:
    """Extract the column list from an INSERT statement using simple bracket matching."""
    lowered = sql_text.lower()
    select_idx = lowered.find("select")
    if select_idx == -1:
        return []

    insert_segment = sql_text[:select_idx]
    open_idx = insert_segment.find("(")
    if open_idx == -1:
        return []

    depth = 0
    close_idx = -1
    for pos in range(open_idx, len(sql_text)):
        char = sql_text[pos]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                close_idx = pos
                break

    if close_idx == -1:
        return []

    column_section = sql_text[open_idx + 1 : close_idx]
    columns = [
        col.strip().strip('"').strip()
        for col in column_section.split(",")
        if col.strip()
    ]
    return columns


def collect_sql_file_paths(project: gitlab.v4.objects.Project, sql_root: str, ref: str) -> List[str]:
    """Recursively list SQL files under the provided path using the GitLab tree API."""
    stack = [sql_root]
    sql_files: List[str] = []

    while stack:
        current_path = stack.pop()
        for node in project.repository_tree(path=current_path, ref=ref, all=True):
            node_path: str = node["path"]
            node_type: str = node["type"]

            lower_parts = {part.lower() for part in node_path.split("/")}
            if any(folder in lower_parts for folder in IGNORE_FOLDERS):
                continue

            if node_type == "tree":
                stack.append(node_path)
            elif node_type == "blob" and node_path.lower().endswith(".sql"):
                if os.path.basename(node_path) in IGNORE_FILES:
                    continue
                sql_files.append(node_path)

    return sorted(sql_files)


def fetch_sql(project: gitlab.v4.objects.Project, path: str, ref: str) -> str:
    """Download and decode a SQL file from GitLab."""
    file_obj = project.files.get(file_path=path, ref=ref)
    return base64.b64decode(file_obj.content).decode("utf-8", errors="replace")


def parse_insert_statement(statement: str, source_file: str) -> List[ParsedRow]:
    """Parse INSERT statements that load into the ODM table and extract column mappings."""
    masked_sql, placeholder_map = mask_placeholders(statement)

    try:
        expression = parse_one(masked_sql, read="postgres")
    except ParseError as exc:
        logging.warning("Skipping statement in %s due to parse error: %s", source_file, exc)
        return []

    if not isinstance(expression, exp.Insert):
        return []

    table_expr = expression.this
    if isinstance(table_expr, exp.Schema) and table_expr.this is not None:
        table_sql = table_expr.this.sql(dialect="postgres")
    else:
        table_sql = expression.this.sql(dialect="postgres")
    target_table = unmask_placeholders(table_sql, placeholder_map)
    target_table_normalized = target_table.replace(" ", "").lower()
    if "{{params.odm_table}}" not in target_table_normalized:
        return []

    columns_expr = expression.args.get("columns")
    target_columns: List[str] = []

    column_items = None
    if columns_expr:
        column_items = columns_expr.expressions if hasattr(columns_expr, "expressions") else columns_expr
    elif isinstance(table_expr, exp.Schema) and table_expr.expressions:
        column_items = table_expr.expressions

    if column_items:
        target_columns = [
            unmask_placeholders(col.sql(dialect="postgres"), placeholder_map).strip('"').strip()
            for col in column_items
        ]
    else:
        # Fallback to textual extraction when sqlglot does not return a column tuple.
        target_columns = extract_insert_columns_from_text(
            unmask_placeholders(statement, placeholder_map)
        )
        if not target_columns:
            logging.warning("Insert without explicit column list in %s; skipping.", source_file)
            return []

    payload = expression.args.get("expression")
    if payload is None:
        fallback_rows = fallback_build_rows_from_select(
            masked_sql,
            placeholder_map,
            target_columns,
            full_statement,
            source_file,
        )
        if fallback_rows:
            return fallback_rows
        logging.warning("Insert without payload in %s; skipping.", source_file)
        return []

    full_statement = unmask_placeholders(expression.sql(dialect="postgres"), placeholder_map)

    rows: List[ParsedRow] = []

    base_ctes: Dict[str, exp.Expression] = {}
    insert_with = expression.args.get("with")
    if isinstance(insert_with, exp.With):
        for cte in insert_with.expressions:
            cte_name = (cte.alias or cte.alias_or_name or "").lower()
            if cte_name:
                base_ctes[cte_name] = cte.this

    if isinstance(payload, exp.With):
        for cte in payload.expressions:
            cte_name = (cte.alias or cte.alias_or_name or "").lower()
            if cte_name:
                base_ctes[cte_name] = cte.this
        payload = payload.this
    if isinstance(payload, exp.Paren):
        payload = payload.this

    select_statements = list(iter_select_statements(payload, base_ctes=base_ctes))
    if select_statements:
        for select_stmt, cte_scope in select_statements:
            rows.extend(
                build_rows_from_select(
                    target_columns,
                    select_stmt,
                    placeholder_map,
                    full_statement,
                    source_file,
                    cte_scope,
                )
            )
    elif isinstance(payload, exp.Values):
        value_rows: List[List[str]] = []
        for tup in payload.expressions:
            value_rows.append(
                [
                    unmask_placeholders(item.sql(dialect="postgres"), placeholder_map).strip()
                    for item in tup.expressions
                ]
            )
        rows.extend(
            build_rows_from_values(
                target_columns,
                value_rows,
                "",
                full_statement,
                source_file,
            )
        )
    else:
        logging.warning(
            "Unsupported INSERT payload (%s) in %s; statement skipped.",
            type(payload).__name__,
            source_file,
        )

    if not rows:
        rows = fallback_build_rows_from_select(
            masked_sql,
            placeholder_map,
            target_columns,
            full_statement,
            source_file,
        )

    return rows


def build_rows_from_values(
    target_columns: Sequence[str],
    value_rows: Sequence[Sequence[str]],
    logic_sql: str,
    full_statement: str,
    source_file: str,
) -> List[ParsedRow]:
    """Align column names and row values, trimming and normalizing for the report."""
    normalized_columns = [normalize_report_column_name(col) for col in target_columns]
    missing = [
        col for col in REPORT_COLUMNS
        if col not in normalized_columns and col not in {"logic", "file_path"}
    ]
    if missing:
        logging.debug(
            "Columns missing from INSERT target in %s: %s",
            source_file,
            ", ".join(missing),
        )

    parsed_rows: List[ParsedRow] = []

    for values in value_rows:
        row_map: Dict[str, str] = {}
        for idx, col in enumerate(target_columns):
            normalized_col = normalize_report_column_name(col)
            raw_value = values[idx] if idx < len(values) else ""
            row_map[normalized_col] = normalize_value(normalized_col, raw_value)

        report_data = {column: row_map.get(column, "") for column in REPORT_COLUMNS}
        report_data["logic"] = logic_sql.strip()
        report_data["file_path"] = source_file
        report_data["dependency"] = ""
        report_data["filename"] = os.path.basename(source_file)
        report_data["profile_table"] = normalize_profile_table_value(report_data.get("profile_table", ""))
        report_data["is_active"] = ""
        report_data["generated_at"] = ""
        parsed_rows.append(
            ParsedRow(
                data=report_data,
                source_statement=full_statement,
                source_file=source_file,
            )
        )

    return parsed_rows


def iter_select_statements(
    expression: exp.Expression | None,
    base_ctes: Dict[str, exp.Expression] | None = None,
) -> List[Tuple[exp.Select, Dict[str, exp.Expression]]]:
    """Yield SELECT statements and their visible CTE map, flattening set operations."""
    if expression is None:
        return []

    initial_ctes = dict(base_ctes or {})
    stack: List[Tuple[exp.Expression, Dict[str, exp.Expression]]] = [(expression, initial_ctes)]
    results: List[Tuple[exp.Select, Dict[str, exp.Expression]]] = []

    while stack:
        node, cte_scope = stack.pop()
        if isinstance(node, exp.Paren):
            stack.append((node.this, dict(cte_scope)))
            continue
        if isinstance(node, exp.With):
            updated_scope = dict(cte_scope)
            for cte in node.expressions:
                cte_name = (cte.alias or cte.alias_or_name or "").lower()
                if not cte_name:
                    continue
                updated_scope[cte_name] = cte.this
            stack.append((node.this, updated_scope))
            continue
        if isinstance(node, exp.Subquery):
            stack.append((node.this, dict(cte_scope)))
            continue
        if isinstance(node, (exp.Order, exp.Limit)):
            stack.append((node.this, dict(cte_scope)))
            continue
        if isinstance(node, exp.Select):
            scope_for_select = dict(cte_scope)
            with_clause = node.args.get("with")
            if isinstance(with_clause, exp.With):
                for cte in with_clause.expressions:
                    cte_name = (cte.alias or cte.alias_or_name or "").lower()
                    if not cte_name:
                        continue
                    scope_for_select[cte_name] = cte.this
            results.append((node, scope_for_select))
            continue
        if isinstance(node, exp.SetOperation):
            stack.append((node.right, dict(cte_scope)))
            stack.append((node.left, dict(cte_scope)))
            continue

    return results


def expand_select_expressions(
    select_expression: exp.Select,
    placeholder_map: Dict[str, str],
    cte_map: Dict[str, exp.Expression],
    column_cache: Dict[str, List[exp.Expression]],
    visited_ctes: Set[str],
) -> List[exp.Expression]:
    """Expand SELECT list, replacing star expressions with concrete expressions when possible."""
    expanded: List[exp.Expression] = []

    for expression in select_expression.expressions:
        replacement = _expand_select_expression_item(
            expression,
            select_expression,
            placeholder_map,
            cte_map,
            column_cache,
            visited_ctes,
        )
        if not replacement:
            expanded.append(expression)
        else:
            expanded.extend(replacement)

    return expanded


def _expand_select_expression_item(
    expression: exp.Expression,
    select_expression: exp.Select,
    placeholder_map: Dict[str, str],
    cte_map: Dict[str, exp.Expression],
    column_cache: Dict[str, List[exp.Expression]],
    visited_ctes: Set[str],
) -> List[exp.Expression]:
    if isinstance(expression, exp.Star):
        resolved = _resolve_star_expressions(
            select_expression,
            qualifier=None,
            placeholder_map=placeholder_map,
            cte_map=cte_map,
            column_cache=column_cache,
            visited_ctes=visited_ctes,
        )
        return resolved

    if isinstance(expression, exp.Column) and getattr(expression, "is_star", False):
        table_ref = expression.table
        if isinstance(table_ref, exp.Expression):
            qualifier = table_ref.sql(dialect="postgres")
        else:
            qualifier = str(table_ref) if table_ref is not None else None
        resolved = _resolve_star_expressions(
            select_expression,
            qualifier=qualifier.lower() if qualifier else None,
            placeholder_map=placeholder_map,
            cte_map=cte_map,
            column_cache=column_cache,
            visited_ctes=visited_ctes,
        )
        return resolved

    return []


def _resolve_star_expressions(
    select_expression: exp.Select,
    qualifier: str | None,
    placeholder_map: Dict[str, str],
    cte_map: Dict[str, exp.Expression],
    column_cache: Dict[str, List[exp.Expression]],
    visited_ctes: Set[str],
) -> List[exp.Expression]:
    """Resolve star projections to explicit expressions when sources are CTEs or subqueries."""
    resolved: List[exp.Expression] = []

    sources: List[exp.Expression] = []
    from_expr = select_expression.args.get("from")
    if isinstance(from_expr, exp.From) and from_expr.this is not None:
        sources.append(from_expr.this)
    for join in select_expression.args.get("joins") or []:
        sources.append(join.this)

    qualifier_lower = qualifier.lower() if qualifier else None

    for source in sources:
        alias = (source.alias or source.alias_or_name or "").strip()
        alias_lower = alias.lower()
        if qualifier_lower and alias_lower != qualifier_lower:
            continue

        expressions = _resolve_source_output_expressions(
            source,
            alias_lower,
            placeholder_map,
            cte_map,
            column_cache,
            visited_ctes,
        )
        if expressions:
            resolved.extend(expressions)
        if qualifier_lower:
            break

    return resolved


def _resolve_source_output_expressions(
    source_expression: exp.Expression,
    alias_lower: str,
    placeholder_map: Dict[str, str],
    cte_map: Dict[str, exp.Expression],
    column_cache: Dict[str, List[exp.Expression]],
    visited_ctes: Set[str],
) -> List[exp.Expression]:
    """Resolve the output expressions for a source referenced by a star projection."""
    if alias_lower in column_cache:
        return [expr.copy() for expr in column_cache[alias_lower]]

    resolved: List[exp.Expression] = []

    if alias_lower and alias_lower in cte_map:
        if alias_lower in visited_ctes:
            return []
        visited_ctes.add(alias_lower)
        cte_expression = cte_map[alias_lower]
        for inner_select, inner_cte_scope in iter_select_statements(cte_expression, base_ctes=cte_map):
            combined_scope = dict(cte_map)
            combined_scope.update(inner_cte_scope)
            resolved = expand_select_expressions(
                inner_select,
                placeholder_map,
                combined_scope,
                column_cache,
                visited_ctes,
            )
            if resolved:
                break
        visited_ctes.remove(alias_lower)
    elif isinstance(source_expression, exp.Subquery):
        subquery_expression = source_expression.this
        for inner_select, inner_cte_scope in iter_select_statements(subquery_expression, base_ctes=cte_map):
            combined_scope = dict(cte_map)
            combined_scope.update(inner_cte_scope)
            resolved = expand_select_expressions(
                inner_select,
                placeholder_map,
                combined_scope,
                column_cache,
                visited_ctes,
            )
            if resolved:
                break

    if alias_lower and resolved:
        column_cache[alias_lower] = [expr.copy() for expr in resolved]

    return [expr.copy() for expr in resolved]


def build_rows_from_select(
    target_columns: Sequence[str],
    select_expression: exp.Select,
    placeholder_map: Dict[str, str],
    full_statement: str,
    source_file: str,
    cte_map: Dict[str, exp.Expression],
) -> List[ParsedRow]:
    """Extract report rows from an INSERT ... SELECT statement."""
    normalized_columns = [normalize_report_column_name(col) for col in target_columns]
    select_items = expand_select_expressions(
        select_expression,
        placeholder_map,
        cte_map,
        column_cache={},
        visited_ctes=set(),
    )

    profile_table = extract_profile_table(select_expression, placeholder_map)
    where_logic, profile_columns = extract_where_context(select_expression, placeholder_map)

    alias_map: Dict[str, List[exp.Expression]] = {}
    for expr in select_items:
        raw_key = (expr.alias_or_name or "").replace('"', "").strip()
        key = normalize_report_column_name(raw_key)
        if key:
            value_expr = expr.this if isinstance(expr, exp.Alias) else expr
            alias_map.setdefault(key, []).append(value_expr)

    select_mappings: List[Tuple[str, exp.Expression | None]] = []
    for idx, column_name in enumerate(normalized_columns):
        expr: exp.Expression | None = None
        if column_name in alias_map and alias_map[column_name]:
            expr = alias_map[column_name].pop(0)
        elif idx < len(select_items):
            candidate = select_items[idx]
            expr = candidate.this if isinstance(candidate, exp.Alias) else candidate
        select_mappings.append((column_name, expr))

    dependency_values = extract_dependency_values(select_expression, placeholder_map, cte_map, expected_column_names=normalized_columns)
    dependency_text = ", ".join(dependency_values)

    base_defaults: Dict[str, str] = {column: "" for column in REPORT_COLUMNS}
    base_defaults["profile_table"] = profile_table
    base_defaults["file_path"] = source_file
    base_defaults["dependency"] = dependency_text
    base_defaults["filename"] = os.path.basename(source_file)
    base_defaults["is_active"] = ""
    base_defaults["generated_at"] = ""

    parsed_rows: List[ParsedRow] = []

    target_rule_columns = profile_columns or [""]
    value_combinations = extract_value_combinations(select_expression, placeholder_map)

    if len(select_items) < len(target_columns):
        logging.warning(
            "Column/SELECT mismatch in %s: %d columns but %d select expressions.",
            source_file,
            len(target_columns),
            len(select_items),
        )

    for combination in value_combinations:
        row_base = base_defaults.copy()
        for column_name, expr in select_mappings:
            raw_value = evaluate_select_expression(expr, combination, placeholder_map)
            row_base[column_name] = normalize_value(column_name, raw_value)

        for rule_column in target_rule_columns:
            row_data = row_base.copy()
            row_data["rule_column"] = rule_column
            row_data["logic"] = where_logic
            insight_key = normalize_insight_type_key(row_data.get("insight_type", ""))
            filtered_dependencies = [
                value
                for value in dependency_values
                if normalize_insight_type_key(value) != insight_key and value
            ]
            row_data["dependency"] = ", ".join(filtered_dependencies)
            row_data["profile_table"] = normalize_profile_table_value(row_data.get("profile_table", ""))
            report_data = {column: row_data.get(column, "") for column in REPORT_COLUMNS}
            parsed_rows.append(
                ParsedRow(
                    data=report_data,
                    source_statement=full_statement,
                    source_file=source_file,
                )
            )

    return parsed_rows


def extract_profile_table(select_expression: exp.Select, placeholder_map: Dict[str, str]) -> str:
    """Return the first table reference found in the SELECT statement."""
    for table in select_expression.find_all(exp.Table):
        table_name = unmask_placeholders(table.sql(dialect="postgres"), placeholder_map)
        return normalize_profile_table_value(table_name)
    return ""


def extract_where_context(
    select_expression: exp.Select,
    placeholder_map: Dict[str, str],
) -> Tuple[str, List[str]]:
    """Return the textual WHERE clause and the ordered unique column references."""
    where_clause = ""
    profile_columns: List[str] = []
    seen: Set[str] = set()

    where_exp = select_expression.args.get("where")
    if not where_exp:
        return where_clause, profile_columns

    where_clause = unmask_placeholders(
        where_exp.this.sql(dialect="postgres"),
        placeholder_map,
    ).strip()

    for column in where_exp.this.find_all(exp.Column):
        if column_in_odm_table_subquery(column, select_expression, placeholder_map):
            continue
        column_sql = unmask_placeholders(column.sql(dialect="postgres"), placeholder_map).strip()
        if not column_sql or column_sql.lower() == "null":
            continue
        if column_sql not in seen:
            seen.add(column_sql)
            profile_columns.append(column_sql)

    return where_clause, profile_columns


def extract_dependency_values(
    select_expression: exp.Select,
    placeholder_map: Dict[str, str],
    cte_map: Dict[str, exp.Expression],
    visited_ctes: Set[str] | None = None,
    expected_column_names: Sequence[str] | None = None,
) -> List[str]:
    """Collect unique insight_type literals from SELECT context, WHERE clauses, and referenced CTEs."""
    seen: Set[str] = set()
    ordered: List[str] = []

    def add_value(raw_value: str) -> None:
        literal = strip_quotes(unmask_placeholders(raw_value, placeholder_map))
        if literal and literal not in seen:
            seen.add(literal)
            ordered.append(literal)

    def is_insight_type_reference(expr: exp.Expression | None) -> bool:
        if expr is None:
            return False
        if isinstance(expr, exp.Column):
            column_sql = unmask_placeholders(expr.sql(dialect="postgres"), placeholder_map)
            normalized = column_sql.replace('"', "").strip().lower()
            return normalized.endswith(".insight_type") or normalized == "insight_type"
        return False

    # Gather dependency values from VALUES-based cross joins.
    for combination in extract_value_combinations(select_expression, placeholder_map):
        for key, raw in combination.items():
            column_ref = key.replace('"', "").strip().lower().split(".")[-1]
            if column_ref == "insight_type":
                add_value(raw)

    # Gather from SELECT list aliases.
    for idx, expression in enumerate(select_expression.expressions):
        alias_raw = (expression.alias_or_name or "").replace('"', "").strip()
        alias_or_name = alias_raw.lower()
        if isinstance(expression, exp.Literal):
            literal_text = strip_quotes(unmask_placeholders(expression.sql(dialect="postgres"), placeholder_map))
            if alias_raw.lower() == literal_text.lower():
                alias_or_name = ""
        if not alias_or_name and expected_column_names and idx < len(expected_column_names):
            alias_or_name = expected_column_names[idx]
        expr_node = expression.this if isinstance(expression, exp.Alias) else expression
        if alias_or_name == "insight_type" and isinstance(expr_node, exp.Literal):
            add_value(expr_node.sql(dialect="postgres"))

    # Gather from WHERE clause filters.
    where_exp = select_expression.args.get("where")
    if where_exp:
        for condition in where_exp.this.walk():
            if isinstance(condition, exp.EQ):
                left = condition.left
                right = condition.right
                if is_insight_type_reference(left) and isinstance(right, exp.Literal):
                    add_value(right.sql(dialect="postgres"))
                elif is_insight_type_reference(right) and isinstance(left, exp.Literal):
                    add_value(left.sql(dialect="postgres"))
            elif isinstance(condition, exp.In):
                if is_insight_type_reference(condition.this):
                    for option in condition.expressions:
                        if isinstance(option, exp.Literal):
                            add_value(option.sql(dialect="postgres"))

    # Explore referenced CTEs when joins target WITH clauses.
    if cte_map:
        visited = visited_ctes or set()
        referenced_ctes: Set[str] = set()
        for table in select_expression.find_all(exp.Table):
            table_name = (table.name or "").lower()
            if table_name in cte_map:
                referenced_ctes.add(table_name)

        for cte_name in referenced_ctes:
            if cte_name in visited or cte_name not in cte_map:
                continue
            visited.add(cte_name)
            cte_expression = cte_map[cte_name]
            reference_names: Sequence[str] | None = None
            for inner_select, inner_cte_map in iter_select_statements(cte_expression, base_ctes=cte_map):
                combined_map = dict(cte_map)
                combined_map.update(inner_cte_map)
                if reference_names is None:
                    reference_names = _infer_select_column_names(
                        inner_select,
                        placeholder_map,
                        combined_map,
                    )
                inner_values = extract_dependency_values(
                    inner_select,
                    placeholder_map,
                    combined_map,
                    visited,
                    expected_column_names=reference_names,
                )
                for value in inner_values:
                    if value not in seen:
                        seen.add(value)
                        ordered.append(value)

    return ordered


def _infer_select_column_names(
    select_expression: exp.Select,
    placeholder_map: Dict[str, str],
    cte_map: Dict[str, exp.Expression],
) -> List[str]:
    """Infer output column names for a SELECT statement."""
    expanded = expand_select_expressions(
        select_expression,
        placeholder_map,
        cte_map,
        column_cache={},
        visited_ctes=set(),
    )

    names: List[str] = []
    for idx, expression in enumerate(expanded):
        alias_or_name = (expression.alias_or_name or "").replace('"', "").strip()
        if not alias_or_name and isinstance(expression, exp.Column):
            alias_or_name = (expression.alias_or_name or expression.output_name or "").replace('"', "").strip()
        if not alias_or_name:
            alias_or_name = f"column_{idx}"
        names.append(alias_or_name.lower())
    return names


def normalize_insight_type_key(value: str) -> str:
    return strip_quotes(value).strip().lower()


def extract_value_combinations(
    select_expression: exp.Select,
    placeholder_map: Dict[str, str],
) -> List[Dict[str, str]]:
    """Return cartesian combinations of VALUES-based cross joins."""
    value_tables: List[Tuple[Set[str], List[str], List[List[str]]]] = []

    for values_expr in select_expression.find_all(exp.Values):
        base_expr = values_expr
        if isinstance(base_expr, exp.Paren):
            base_expr = base_expr.this

        table_alias = values_expr.args.get("alias")
        if not table_alias:
            continue

        alias_sql_name = table_alias.this.sql(dialect="postgres").strip()
        if not alias_sql_name:
            continue
        alias_variants = {
            alias_sql_name.lower(),
            alias_sql_name.replace('"', "").lower(),
        }

        alias_columns_expr = table_alias.args.get("columns")
        if alias_columns_expr is None:
            continue

        if isinstance(alias_columns_expr, list):
            alias_column_nodes = alias_columns_expr
        elif hasattr(alias_columns_expr, "expressions"):
            alias_column_nodes = list(alias_columns_expr.expressions)
        else:
            alias_column_nodes = [alias_columns_expr]
        alias_columns: List[str] = []
        for col in alias_column_nodes:
            column_sql = unmask_placeholders(col.sql(dialect="postgres"), placeholder_map).strip()
            if not column_sql:
                continue
            alias_columns.append(column_sql)

        rows: List[List[str]] = []
        for tuple_expr in base_expr.expressions:
            row_values: List[str] = []
            for value_expr in tuple_expr.expressions:
                value_text = unmask_placeholders(
                    value_expr.sql(dialect="postgres"),
                    placeholder_map,
                ).strip()
                row_values.append(value_text)
            rows.append(row_values)

        if alias_columns and rows:
            value_tables.append((alias_variants, alias_columns, rows))

    if not value_tables:
        return [dict()]

    combinations: List[Dict[str, str]] = [dict()]
    for alias_variants, columns, rows in value_tables:
        new_combinations: List[Dict[str, str]] = []
        for combo in combinations:
            for row in rows:
                updated = combo.copy()
                for idx, column_name in enumerate(columns):
                    value = row[idx] if idx < len(row) else ""
                    column_clean = column_name.replace('"', "")
                    column_variants = {
                        column_name.lower(),
                        column_clean.lower(),
                        f'"{column_clean}"'.lower(),
                    }
                    for alias_variant in alias_variants:
                        if alias_variant:
                            updated[f"{alias_variant}.{column_name}".lower()] = value
                            updated[f"{alias_variant}.{column_clean}".lower()] = value
                            updated[f"{alias_variant}.\"{column_clean}\"".lower()] = value
                    for column_variant in column_variants:
                        updated[column_variant] = value
                new_combinations.append(updated)
        combinations = new_combinations or combinations

    return combinations or [dict()]


def evaluate_select_expression(
    expr: exp.Expression | None,
    combination: Dict[str, str],
    placeholder_map: Dict[str, str],
) -> str:
    """Evaluate a SELECT expression using known VALUES combinations."""
    if expr is None:
        return ""

    if isinstance(expr, exp.Alias):
        expr = expr.this

    expr_sql = unmask_placeholders(expr.sql(dialect="postgres"), placeholder_map).strip()
    lookup_key = expr_sql.lower()

    if lookup_key in combination:
        return combination[lookup_key]

    if "." in lookup_key:
        _, column_only = lookup_key.rsplit(".", 1)
        if column_only in combination:
            return combination[column_only]

    if isinstance(expr, exp.Literal):
        return expr_sql

    return expr_sql


def column_in_odm_table_subquery(
    column: exp.Column,
    root_select: exp.Select,
    placeholder_map: Dict[str, str],
) -> bool:
    """Determine if the column resides in a subquery that targets the ODM table."""
    current: exp.Expression | None = column
    while current is not None:
        if current is root_select:
            return False
        if isinstance(current, exp.Select):
            for table in current.find_all(exp.Table):
                table_sql = unmask_placeholders(
                    table.sql(dialect="postgres"),
                    placeholder_map,
                )
                normalized = table_sql.replace(" ", "").lower()
                if "{{params.odm_table}}" in normalized:
                    return True
        current = current.parent
    return False


def fallback_build_rows_from_select(
    masked_statement: str,
    placeholder_map: Dict[str, str],
    target_columns: Sequence[str],
    full_statement: str,
    source_file: str,
) -> List[ParsedRow]:
    """Fallback parser for INSERT statements when sqlglot fails on payload."""
    if not target_columns:
        target_columns = extract_insert_columns_from_text(
            unmask_placeholders(masked_statement, placeholder_map)
        )
        if not target_columns:
            return []

    lower_masked = masked_statement.lower()
    select_pos = lower_masked.find("select")
    if select_pos == -1:
        return []

    select_sql_masked = masked_statement[select_pos:].strip().rstrip(";")

    try:
        select_expression = parse_one(select_sql_masked, read="postgres")
    except ParseError as exc:
        logging.warning("Fallback parse error in %s: %s", source_file, exc)
        return []

    base_ctes: Dict[str, exp.Expression] = {}
    if isinstance(select_expression, exp.With):
        for cte in select_expression.expressions:
            cte_name = (cte.alias or cte.alias_or_name or "").lower()
            if cte_name:
                base_ctes[cte_name] = cte.this
        select_expression = select_expression.this

    select_statements = list(iter_select_statements(select_expression, base_ctes=base_ctes))
    if not select_statements:
        logging.warning(
            "Fallback parser produced %s instead of SELECT in %s; skipping.",
            type(select_expression).__name__,
            source_file,
        )
        return []

    fallback_rows: List[ParsedRow] = []
    for stmt, cte_scope in select_statements:
        fallback_rows.extend(
            build_rows_from_select(
                target_columns,
                stmt,
                placeholder_map,
                full_statement,
                source_file,
                cte_scope,
            )
        )

    return fallback_rows


def extract_report_rows(sql_text: str, source_file: str) -> List[ParsedRow]:
    """Split a SQL file into statements and parse relevant INSERT statements."""
    rows: List[ParsedRow] = []
    for statement in sqlparse.split(sql_text):
        cleaned_statement = statement.strip()
        if not cleaned_statement:
            continue
        rows.extend(parse_insert_statement(cleaned_statement, source_file))
    return rows


def create_report_dataframe(rows: Iterable[ParsedRow]) -> pd.DataFrame:
    """Build the final DataFrame in the desired column order."""
    data = [row.data for row in rows]
    return pd.DataFrame(data, columns=REPORT_COLUMNS)


def filter_excluded_records(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows that match configured exclusion criteria."""
    profile_table_normalized = df["profile_table"].fillna("").map(normalize_filter_text)
    target_key_normalized = df["target_key"].fillna("").map(normalize_filter_text)

    exclusion_mask = profile_table_normalized.isin(EXCLUDED_PROFILE_TABLES) | target_key_normalized.isin(EXCLUDED_TARGET_KEYS)

    excluded_count = int(exclusion_mask.sum())
    if excluded_count:
        logging.info("Excluding %d row(s) from report/output.", excluded_count)

    return df.loc[~exclusion_mask].reset_index(drop=True)


def write_excel_report(df: pd.DataFrame, directory: str = ".") -> str:
    """Persist the DataFrame to an Excel file and return the absolute path."""
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"odm_rules_details_{timestamp}.xlsx"
    output_path = os.path.abspath(os.path.join(directory, filename))

    logging.info("Writing Excel report to %s", output_path)
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="odm_rules_details")

    return output_path


def upload_dataframe_to_db(df: pd.DataFrame, db_password: str) -> None:
    """Drop and recreate the destination table, then load the DataFrame contents."""
    db_config = {
        "host": "greenplum-rdsp.zur.swissbank.com",
        "port": 5432,
        "dbname": "gprdsp",
        "user": "ds_rdsp_dev",
        "password": db_password,
    }

    table_fqn = "sandbox_prj_smart_insights.odm_rule_metadata_auto_refresh"
    owner_role = "erd_gpdb_prj_smart_insights"
    reader_role = "erd_gpdb_prj_smart_insights_ro"

    logging.info("Connecting to GPDB cluster to refresh %s", table_fqn)

    with psycopg2.connect(**db_config) as conn:
        conn.autocommit = False
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {table_fqn};")

            column_definitions = ", ".join(f'"{column}" TEXT' for column in REPORT_COLUMNS)
            cur.execute(f"CREATE TABLE {table_fqn} ({column_definitions});")

            if not df.empty:
                columns_clause = ", ".join(f'"{column}"' for column in REPORT_COLUMNS)
                values = [tuple(row[column] for column in REPORT_COLUMNS) for _, row in df.iterrows()]
                execute_values(
                    cur,
                    f"INSERT INTO {table_fqn} ({columns_clause}) VALUES %s",
                    values,
                )

            cur.execute(f"ALTER TABLE {table_fqn} OWNER TO {owner_role};")
            cur.execute(f"GRANT SELECT ON {table_fqn} TO {reader_role};")

        conn.commit()
        logging.info("Table %s refreshed successfully.", table_fqn)


def fetch_exclusion_insight_types(db_password: str) -> Set[str]:
    """Fetch insight types that should be marked as inactive."""
    db_config = {
        "host": "greenplum-rdsp.zur.swissbank.com",
        "port": 5432,
        "dbname": "gprdsp",
        "user": "ds_rdsp_dev",
        "password": db_password,
    }

    query = """
        SELECT insight_type
        FROM core_ikg.odm_exclusion_insight_type
        WHERE is_curr = 1
    """

    with psycopg2.connect(**db_config) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            results = cur.fetchall()

    return {normalize_insight_type_key(row[0]) for row in results if row and row[0]}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    private_token = getpass.getpass("Enter your private token: ")
    gl = gitlab.Gitlab(GITLAB_URL, private_token=private_token)

    logging.info("Fetching ODM project metadata...")
    project = gl.projects.get(ODM_PROJECT_PATH)

    logging.info("Collecting SQL file paths under %s...", SQL_PATH)
    sql_files = collect_sql_file_paths(project, SQL_PATH, BRANCH)
    logging.info("Found %d SQL files to inspect.", len(sql_files))

    all_rows: List[ParsedRow] = []
    for path in sql_files:
        sql_content = fetch_sql(project, path, BRANCH)
        rows = extract_report_rows(sql_content, path)
        if rows:
            logging.info("Extracted %d row(s) from %s", len(rows), path)
            all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError("No ODM rule rows were parsed from the SQL sources.")

    df = create_report_dataframe(all_rows)

    generated_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")
    df["generated_at"] = generated_at
    df["filename"] = df["filename"].fillna("").apply(lambda v: os.path.basename(v) if isinstance(v, str) else "")
    df["profile_table"] = df["profile_table"].fillna("").apply(normalize_profile_table_value)
    df = filter_excluded_records(df)

    db_password = getpass.getpass("Enter Password for DB User: ")
    exclusion_types = fetch_exclusion_insight_types(db_password)
    logging.info("Fetched %d exclusion insight_type record(s) for is_active flag.", len(exclusion_types))

    def resolve_is_active(value: str) -> str:
        key = normalize_insight_type_key(value)
        return "N" if key and key in exclusion_types else "Y"

    df["is_active"] = df["insight_type"].fillna("").apply(resolve_is_active)

    excel_path = write_excel_report(df)
    logging.info("Report contains %d rows.", len(df))

    upload_dataframe_to_db(df, db_password)

    logging.info("Process completed. Excel report saved at: %s", excel_path)


if __name__ == "__main__":
    main()
