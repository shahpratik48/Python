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
OUTPUT_PREFIX = "ikg_column_lineage_master"
TARGET_SCHEMA = "sandbox_prj_smart_insights"
TARGET_TABLE = "ikg_column_lineage_master_auto_refresh"
TARGET_OWNER = "erd_gpdb_prj_smart_insights"
TARGET_READER = "erd_gpdb_prj_smart_insights_ro"
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


def derive_process(file_path: str) -> str:
    """Derive process from the last subfolder before the SQL file."""
    parts = Path(file_path).parts
    # Get the parent directory of the file (last folder before the .sql file)
    if len(parts) >= 2:
        # parts[-1] is the filename, parts[-2] is the last folder
        return parts[-2]
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


class ColumnLineageParser:
    """Advanced SQL parser for extracting column-level lineage."""
    
    TEMPLATE_PATTERN = re.compile(r"{{\s*([^{}]+?)\s*}}")
    
    def __init__(self):
        self.cte_definitions = {}  # Store CTE definitions for resolution
        self.table_aliases = {}  # Store table aliases
        
    def extract_column_lineage(self, sql_text: str) -> List[Dict]:
        """Extract column lineage from SQL text."""
        # Store original SQL BEFORE any cleaning
        self.original_sql = sql_text
        
        cleaned = self._remove_sql_comments(sql_text)
        # Store cleaned SQL (with comments removed but vendor-specific syntax intact)
        self.cleaned_sql = cleaned
        
        # Now do vendor-specific stripping only for parsing
        normalized = self._strip_vendor_specific(cleaned)
        sanitized, placeholders = self._replace_templates(normalized)
        sanitized_no_do, do_blocks = self._strip_do_blocks(sanitized)
        
        lineage_records = []
        
        # Parse main SQL
        try:
            parsed = sqlglot.parse(sanitized_no_do, read="postgres", error_level="ignore")
            for statement in parsed:
                if statement:
                    statement_records = self._process_statement(statement, placeholders)
                    
                    # Replace logic for CREATE statements with original SQL
                    for record in statement_records:
                        if record.get("_needs_original_sql"):
                            record["logic"] = self._extract_create_statement_sql(
                                record.get("sub_target_table"),
                                self.cleaned_sql  # Use cleaned SQL (has DISTRIBUTED BY)
                            )
                            del record["_needs_original_sql"]
                    
                    lineage_records.extend(statement_records)
        except Exception as e:
            logging.warning(f"Failed to parse SQL: {e}")
            
        # Parse DO blocks
        for block in do_blocks:
            try:
                block_prepared = self._prepare_do_block(block)
                parsed_block = sqlglot.parse(block_prepared, read="postgres", error_level="ignore")
                for statement in parsed_block:
                    if statement:
                        lineage_records.extend(self._process_statement(statement, placeholders))
            except Exception as e:
                logging.debug(f"Failed to parse DO block: {e}")
                
        return lineage_records
    
    def _extract_create_statement_sql(self, table_name: str, original_sql: str) -> str:
        """Extract the original CREATE statement SQL for a specific table."""
        if not table_name:
            return original_sql
        
        # Find the CREATE statement for this table in the original SQL
        # Look for patterns like: CREATE TEMP TABLE table_name ... ; or CREATE TABLE ...
        import re
        
        # Pattern to match CREATE [TEMP/TEMPORARY] TABLE table_name ... up to the semicolon or next CREATE
        pattern = rf'(?i)((?:DROP\s+TABLE\s+IF\s+EXISTS\s+{re.escape(table_name)}\s*;?\s*)?' \
                  rf'CREATE\s+(?:TEMP(?:ORARY)?\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?{re.escape(table_name)}\s+' \
                  rf'(?:.*?)(?:;|\Z))'
        
        match = re.search(pattern, original_sql, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # If specific table not found, return the whole CREATE statement
        create_match = re.search(r'(?i)CREATE\s+(?:.*?)(?:;|\Z)', original_sql, re.DOTALL)
        if create_match:
            return create_match.group(0).strip()
        
        return original_sql
    
    def _process_statement(self, statement: exp.Expression, placeholders: Dict[str, str]) -> List[Dict]:
        """Process a single SQL statement to extract column lineage."""
        records = []
        
        # Reset for each statement
        self.cte_definitions = {}
        self.table_aliases = {}
        
        # Handle CREATE TABLE / CREATE TEMP TABLE statements
        if isinstance(statement, exp.Create):
            records.extend(self._process_create_statement(statement, placeholders))
        
        # Handle SELECT statements
        elif isinstance(statement, exp.Select):
            records.extend(self._process_select_statement(statement, placeholders, None))
            
        # Handle INSERT statements
        elif isinstance(statement, exp.Insert):
            records.extend(self._process_insert_statement(statement, placeholders))
            
        return records
    
    def _process_create_statement(self, create_node: exp.Create, placeholders: Dict[str, str]) -> List[Dict]:
        """Process CREATE TABLE statement."""
        records = []
        
        # Get target table info
        target_table = None
        target_schema = None
        
        if create_node.this:
            table_expr = create_node.this
            if isinstance(table_expr, exp.Schema):
                table_expr = table_expr.this
            
            if isinstance(table_expr, exp.Table):
                target_table = table_expr.name
                if table_expr.db:
                    target_schema = self._resolve_template(table_expr.db, placeholders)
        
        # Get the SELECT part
        select_expr = create_node.expression
        
        # Add a record for the whole table creation
        # CRITICAL: Use the ORIGINAL SQL text, not the sqlglot-generated version
        # We need to extract this from the original source
        # For now, we'll store the create node and extract original later
        records.append({
            "sub_target_schema": None,
            "sub_target_table": target_table,
            "target_column": None,
            "source_schema": None,
            "source_table": None,
            "source_column": None,
            "logic": None,  # Will be filled with original SQL by caller
            "sql_process": "create",
            "_needs_original_sql": True  # Flag to replace with original
        })
        
        if select_expr:
            # Process the main SELECT (CTEs will be processed inside _process_select_for_target)
            if isinstance(select_expr, exp.Select):
                column_records = self._process_select_for_target(
                    select_expr, 
                    target_table, 
                    target_schema,
                    placeholders
                )
                records.extend(column_records)
        
        return records
    
    def _process_select_for_target(
        self, 
        select_node: exp.Select, 
        target_table: Optional[str],
        target_schema: Optional[str],
        placeholders: Dict[str, str]
    ) -> List[Dict]:
        """Process SELECT statement for a target table."""
        records = []
        
        # CRITICAL: Process CTEs FIRST before building table aliases
        with_node = select_node.args.get("with")
        if with_node:
            self._extract_cte_definitions(with_node, placeholders)
            
            # IMPORTANT: Also process joins/where/having from WITHIN each CTE
            cte_internal_records = self._process_cte_internals(
                with_node, 
                target_table, 
                target_schema, 
                placeholders
            )
            records.extend(cte_internal_records)
        
        # Build table alias map (now CTEs are already defined)
        self._build_table_aliases(select_node, placeholders)
        
        # Process each column in SELECT clause
        for projection in select_node.expressions:
            column_info = self._extract_column_info(projection, placeholders)
            
            if column_info:
                # Resolve source table for each source column
                for col_data in column_info:
                    source_records = self._resolve_source_column(
                        col_data, 
                        select_node, 
                        placeholders
                    )
                    
                    # Get the actual source logic from CTE if the column references a CTE
                    original_logic = col_data.get("logic")
                    source_col_expr = col_data.get("source_column_expr")
                    
                    if source_col_expr:
                        table_ref = source_col_expr.get("table")
                        column_name = source_col_expr.get("column")
                        
                        # Get actual source logic from CTE
                        cte_source_logic = self._get_source_logic_detail(
                            table_ref,
                            column_name,
                            select_node,
                            placeholders
                        )
                        
                        if cte_source_logic:
                            # Use the CTE source logic instead
                            original_logic = cte_source_logic
                    
                    for src_rec in source_records:
                        records.append({
                            "sub_target_schema": target_schema,
                            "sub_target_table": target_table,
                            "target_column": col_data.get("target_column"),
                            "source_schema": src_rec.get("source_schema"),
                            "source_table": src_rec.get("source_table"),
                            "source_column": src_rec.get("source_column"),
                            "logic": original_logic,
                            "sql_process": "select"
                        })
        
        # Process JOIN clauses - now with proper target table and column tracking
        join_records = self._process_joins(select_node, placeholders, target_table, target_schema)
        records.extend(join_records)
        
        # Process WHERE clause
        where_records = self._process_where(select_node, placeholders, target_table, target_schema)
        records.extend(where_records)
        
        # Process HAVING clause
        having_records = self._process_having(select_node, placeholders, target_table, target_schema)
        records.extend(having_records)
        
        return records
    
    def _process_cte_internals(
        self,
        with_node: exp.With,
        target_table: Optional[str],
        target_schema: Optional[str],
        placeholders: Dict[str, str]
    ) -> List[Dict]:
        """Process joins/where/having from within CTE definitions."""
        records = []
        
        for cte in with_node.expressions:
            if isinstance(cte, exp.CTE):
                cte_query = cte.this
                
                if isinstance(cte_query, exp.Select):
                    # Save current state
                    saved_aliases = self.table_aliases.copy()
                    saved_ctes = self.cte_definitions.copy()
                    
                    # Process nested CTEs if any
                    nested_with = cte_query.args.get("with")
                    if nested_with:
                        self._extract_cte_definitions(nested_with, placeholders)
                    
                    # Build table aliases for this CTE
                    self._build_table_aliases(cte_query, placeholders)
                    
                    # Process joins/where/having within this CTE
                    cte_joins = self._process_joins(cte_query, placeholders, target_table, target_schema)
                    records.extend(cte_joins)
                    
                    cte_where = self._process_where(cte_query, placeholders, target_table, target_schema)
                    records.extend(cte_where)
                    
                    cte_having = self._process_having(cte_query, placeholders, target_table, target_schema)
                    records.extend(cte_having)
                    
                    # Restore state
                    self.table_aliases = saved_aliases
                    self.cte_definitions = saved_ctes
        
        return records
    
    def _extract_column_info(self, projection: exp.Expression, placeholders: Dict[str, str]) -> List[Dict]:
        """Extract column information from a projection."""
        results = []
        
        # Get the alias (target column name)
        target_column = None
        if isinstance(projection, exp.Alias):
            target_column = projection.alias
            source_expr = projection.this
        else:
            source_expr = projection
            # If no alias, use the column name itself
            if isinstance(source_expr, exp.Column):
                target_column = source_expr.name
            elif isinstance(source_expr, exp.Star):
                # Handle SELECT *
                return []
        
        # Get the logic
        logic = projection.sql(dialect="postgres")
        
        # Extract source columns from the expression
        source_columns = self._extract_source_columns_from_expr(source_expr)
        
        if source_columns:
            for src_col in source_columns:
                results.append({
                    "target_column": target_column,
                    "source_column_expr": src_col,
                    "logic": logic
                })
        else:
            # No source columns (e.g., constant)
            results.append({
                "target_column": target_column,
                "source_column_expr": None,
                "logic": logic
            })
        
        return results
    
    def _extract_source_columns_from_expr(self, expr: exp.Expression) -> List[Dict]:
        """Extract all source columns from an expression."""
        columns = []
        
        for node in expr.walk():
            if isinstance(node, exp.Column):
                # Get the table reference from the column
                table_ref = None
                if hasattr(node, "table") and node.table:
                    table_ref = node.table
                # Also check for the 'this' attribute which sqlglot uses for table references
                elif hasattr(node, "this") and isinstance(node.this, exp.Identifier):
                    # The table might be in the parent
                    pass
                
                # Extract table from the column's SQL if not found
                if not table_ref:
                    col_sql = node.sql(dialect="postgres")
                    if "." in col_sql:
                        parts = col_sql.split(".")
                        if len(parts) == 2:
                            table_ref = parts[0].strip()
                
                col_info = {
                    "column": node.name,
                    "table": table_ref
                }
                columns.append(col_info)
        
        return columns
    
    def _resolve_source_column(
        self, 
        column_data: Dict, 
        select_node: exp.Select,
        placeholders: Dict[str, str]
    ) -> List[Dict]:
        """Resolve the actual source table and schema for a column."""
        results = []
        
        source_col_expr = column_data.get("source_column_expr")
        
        if not source_col_expr:
            # No source column (constant or expression without columns)
            return [{
                "source_schema": None,
                "source_table": None,
                "source_column": None
            }]
        
        table_ref = source_col_expr.get("table")
        column_name = source_col_expr.get("column")
        
        # Resolve the table reference
        if table_ref:
            # CRITICAL FIX: Check table_aliases first (this includes both CTEs and real tables)
            if table_ref in self.table_aliases:
                table_info = self.table_aliases[table_ref]
                
                # Check if this is a CTE reference
                if table_info.get("is_cte", False):
                    # This is a CTE - resolve from the CTE definition
                    cte_name = table_info.get("table")
                    cte_results = self._resolve_from_cte(cte_name, column_name, placeholders)
                    results.extend(cte_results)
                else:
                    # This is a real table
                    results.append({
                        "source_schema": table_info.get("schema"),
                        "source_table": table_info.get("table"),
                        "source_column": column_name
                    })
            # If not in table_aliases, check if it's directly a CTE name
            elif table_ref in self.cte_definitions:
                # Direct CTE reference (unlikely but handle it)
                cte_results = self._resolve_from_cte(table_ref, column_name, placeholders)
                results.extend(cte_results)
            else:
                # Direct table reference (not aliased, not CTE)
                results.append({
                    "source_schema": None,
                    "source_table": table_ref,
                    "source_column": column_name
                })
        else:
            # No table reference - need to infer from FROM clause
            from_tables = self._get_from_tables(select_node, placeholders)
            
            if len(from_tables) == 1:
                # Only one table, so it must be from there
                table_info = list(from_tables.values())[0]
                
                # Check if it's a CTE
                if table_info.get("is_cte", False):
                    cte_name = table_info.get("table")
                    cte_results = self._resolve_from_cte(cte_name, column_name, placeholders)
                    results.extend(cte_results)
                else:
                    results.append({
                        "source_schema": table_info.get("schema"),
                        "source_table": table_info.get("table"),
                        "source_column": column_name
                    })
            else:
                # Multiple tables - would need INFORMATION_SCHEMA lookup
                # For now, return all possibilities
                for table_info in from_tables.values():
                    if table_info.get("is_cte", False):
                        cte_name = table_info.get("table")
                        cte_results = self._resolve_from_cte(cte_name, column_name, placeholders)
                        results.extend(cte_results)
                    else:
                        results.append({
                            "source_schema": table_info.get("schema"),
                            "source_table": table_info.get("table"),
                            "source_column": column_name
                        })
        
        return results if results else [{
            "source_schema": None,
            "source_table": None,
            "source_column": column_name
        }]
    
    def _resolve_from_cte(self, cte_name: str, column_name: str, placeholders: Dict[str, str]) -> List[Dict]:
        """Resolve column from a CTE definition."""
        results = []
        
        cte_select = self.cte_definitions.get(cte_name)
        if not cte_select:
            return results
        
        # Build table aliases for this CTE's SELECT
        saved_aliases = self.table_aliases.copy()
        self._build_table_aliases(cte_select, placeholders)
        
        # Find the column in CTE's SELECT clause
        for projection in cte_select.expressions:
            proj_alias = None
            proj_expr = projection
            
            if isinstance(projection, exp.Alias):
                proj_alias = projection.alias
                proj_expr = projection.this
            elif isinstance(proj_expr, exp.Column):
                proj_alias = proj_expr.name
            
            # Check if this projection matches our target column
            if proj_alias == column_name:
                # Extract source columns from this projection
                source_cols = self._extract_source_columns_from_expr(proj_expr)
                
                if not source_cols:
                    # This is a constant or expression without columns
                    results.append({
                        "source_schema": None,
                        "source_table": None,
                        "source_column": None
                    })
                else:
                    # Recursively resolve each source column
                    for src_col in source_cols:
                        col_data = {
                            "source_column_expr": src_col,
                            "target_column": column_name
                        }
                        resolved = self._resolve_source_column(col_data, cte_select, placeholders)
                        results.extend(resolved)
                break
        
        # Restore the original table aliases
        self.table_aliases = saved_aliases
        
        return results
    
    def _trace_logic_through_ctes(
        self,
        table_ref: Optional[str],
        column_name: str,
        original_logic: str,
        select_node: exp.Select,
        placeholders: Dict[str, str]
    ) -> str:
        """Trace the logic for a column back through CTEs to get the original source logic."""
        if not table_ref:
            return original_logic
        
        # Check if this table reference is a CTE alias
        if table_ref in self.table_aliases:
            table_info = self.table_aliases[table_ref]
            
            if table_info.get("is_cte", False):
                # This is a CTE - get the original logic from the CTE definition
                cte_name = table_info.get("table")
                cte_select = self.cte_definitions.get(cte_name)
                
                if cte_select:
                    # Find the column in the CTE's SELECT clause
                    for projection in cte_select.expressions:
                        proj_alias = None
                        proj_expr = projection
                        
                        if isinstance(projection, exp.Alias):
                            proj_alias = projection.alias
                            proj_expr = projection.this
                        elif isinstance(proj_expr, exp.Column):
                            proj_alias = proj_expr.name
                        
                        if proj_alias == column_name:
                            # Get the original logic from the CTE
                            cte_logic = projection.sql(dialect="postgres")
                            
                            # Check if we need to trace further through nested CTEs
                            source_cols = self._extract_source_columns_from_expr(proj_expr)
                            if source_cols and len(source_cols) == 1:
                                src_col = source_cols[0]
                                src_table = src_col.get("table")
                                src_column = src_col.get("column")
                                
                                # Recursively trace if this is also from a CTE
                                return self._trace_logic_through_ctes(
                                    src_table,
                                    src_column,
                                    cte_logic,
                                    cte_select,
                                    placeholders
                                )
                            
                            return cte_logic
        
        return original_logic
    
    def _get_source_logic_detail(
        self,
        table_ref: Optional[str],
        column_name: str,
        select_node: exp.Select,
        placeholders: Dict[str, str]
    ) -> Optional[str]:
        """Get the source logic detail for a column, particularly from CTEs."""
        if not table_ref:
            return None
        
        # Check if this table reference is a CTE alias
        if table_ref in self.table_aliases:
            table_info = self.table_aliases[table_ref]
            
            if table_info.get("is_cte", False):
                # This is a CTE - get the original logic from the CTE definition
                cte_name = table_info.get("table")
                cte_select = self.cte_definitions.get(cte_name)
                
                if cte_select:
                    # Find the column in the CTE's SELECT clause
                    for projection in cte_select.expressions:
                        proj_alias = None
                        
                        if isinstance(projection, exp.Alias):
                            proj_alias = projection.alias
                        elif isinstance(projection.this, exp.Column) if hasattr(projection, 'this') else isinstance(projection, exp.Column):
                            proj_alias = projection.this.name if hasattr(projection, 'this') else projection.name
                        
                        if proj_alias == column_name:
                            # Return the original logic from the CTE
                            return projection.sql(dialect="postgres")
        
        return None
    
    def _build_table_aliases(self, select_node: exp.Select, placeholders: Dict[str, str]):
        """Build a map of table aliases to actual tables."""
        self.table_aliases = {}
        
        # Process FROM clause
        from_expr = select_node.args.get("from")
        if from_expr:
            self._extract_table_from_source(from_expr.this, placeholders)
        
        # Process JOINs
        joins = select_node.args.get("joins", [])
        for join in joins:
            self._extract_table_from_source(join.this, placeholders)
    
    def _extract_table_from_source(self, source: exp.Expression, placeholders: Dict[str, str]):
        """Extract table information from a FROM/JOIN source."""
        if isinstance(source, exp.Table):
            table_name = source.name
            schema_name = None
            if source.db:
                schema_name = self._resolve_template(source.db, placeholders)
            
            # Get the alias - this is the key used in column references
            alias = source.alias if hasattr(source, "alias") and source.alias else table_name
            
            # CRITICAL FIX: Check if table_name is a CTE
            if table_name in self.cte_definitions:
                # This is a CTE reference - mark it as a CTE, not a real table
                # Don't add to table_aliases as a real table
                # The alias maps to the CTE name
                self.table_aliases[alias] = {
                    "table": table_name,
                    "schema": None,
                    "is_cte": True
                }
            else:
                # This is a real table
                self.table_aliases[alias] = {
                    "table": table_name,
                    "schema": schema_name,
                    "is_cte": False
                }
        elif isinstance(source, exp.Subquery):
            # Handle subqueries
            if source.alias:
                # This is a derived table, not tracked as real table
                pass
    
    def _get_from_tables(self, select_node: exp.Select, placeholders: Dict[str, str]) -> Dict[str, Dict]:
        """Get all tables from FROM and JOIN clauses."""
        tables = {}
        
        # FROM clause
        from_expr = select_node.args.get("from")
        if from_expr and isinstance(from_expr.this, exp.Table):
            table = from_expr.this
            alias = table.alias if hasattr(table, "alias") and table.alias else table.name
            
            # Check if this is a CTE
            if table.name in self.cte_definitions:
                tables[alias] = {
                    "table": table.name,
                    "schema": None,
                    "is_cte": True
                }
            else:
                schema = None
                if table.db:
                    schema = self._resolve_template(table.db, placeholders)
                
                tables[alias] = {
                    "table": table.name,
                    "schema": schema,
                    "is_cte": False
                }
        
        # JOINs
        joins = select_node.args.get("joins", [])
        for join in joins:
            if isinstance(join.this, exp.Table):
                table = join.this
                alias = table.alias if hasattr(table, "alias") and table.alias else table.name
                
                # Check if this is a CTE
                if table.name in self.cte_definitions:
                    tables[alias] = {
                        "table": table.name,
                        "schema": None,
                        "is_cte": True
                    }
                else:
                    schema = None
                    if table.db:
                        schema = self._resolve_template(table.db, placeholders)
                    
                    tables[alias] = {
                        "table": table.name,
                        "schema": schema,
                        "is_cte": False
                    }
        
        return tables
    
    def _extract_cte_definitions(self, with_node: exp.With, placeholders: Dict[str, str]):
        """Extract all CTE definitions and process their internal joins/where/having."""
        for cte in with_node.expressions:
            if isinstance(cte, exp.CTE):
                cte_name = cte.alias
                cte_query = cte.this
                
                if isinstance(cte_query, exp.Select):
                    # Store the CTE definition
                    self.cte_definitions[cte_name] = cte_query
                    
                    # Recursively process nested CTEs
                    nested_with = cte_query.args.get("with")
                    if nested_with:
                        self._extract_cte_definitions(nested_with, placeholders)
    
    def _process_joins(
        self, 
        select_node: exp.Select, 
        placeholders: Dict[str, str],
        target_table: Optional[str],
        target_schema: Optional[str]
    ) -> List[Dict]:
        """Process JOIN clauses to extract column lineage with proper target column tracking."""
        records = []
        
        joins = select_node.args.get("joins", [])
        for join in joins:
            on_condition = join.args.get("on")
            if on_condition:
                # Get the actual join condition (e.g., "ON a.x = b.x")
                join_condition_sql = on_condition.sql(dialect="postgres")
                
                # Extract columns from ON condition
                columns = self._extract_source_columns_from_expr(on_condition)
                
                # Check if this is from a CTE
                sql_process = "join"
                if select_node.args.get("with"):
                    sql_process = "join-with"
                
                # Process each column in the join condition
                for col_info in columns:
                    col_data = {"source_column_expr": col_info}
                    resolved = self._resolve_source_column(col_data, select_node, placeholders)
                    
                    # Get the target column name from the join condition
                    target_col_name = col_info.get("column")
                    table_alias = col_info.get("table")
                    
                    # Build the logic: "ON a.x = b.x (source: actual_table.actual_column)"
                    for src_rec in resolved:
                        # Get the original source logic from CTE if applicable
                        source_logic_detail = self._get_source_logic_detail(
                            table_alias,
                            target_col_name,
                            select_node,
                            placeholders
                        )
                        
                        if source_logic_detail:
                            # Format: "ON a.x = b.x (a.x = actual_source_logic)"
                            ref_expr = f"{table_alias}.{target_col_name}" if table_alias else target_col_name
                            logic = f"ON {join_condition_sql} ({ref_expr} = {source_logic_detail})"
                        else:
                            logic = f"ON {join_condition_sql}"
                        
                        records.append({
                            "sub_target_schema": target_schema,
                            "sub_target_table": target_table,
                            "target_column": target_col_name,
                            "source_schema": src_rec.get("source_schema"),
                            "source_table": src_rec.get("source_table"),
                            "source_column": src_rec.get("source_column"),
                            "logic": logic,
                            "sql_process": sql_process
                        })
        
        return records
    
    def _process_where(
        self, 
        select_node: exp.Select, 
        placeholders: Dict[str, str],
        target_table: Optional[str],
        target_schema: Optional[str]
    ) -> List[Dict]:
        """Process WHERE clause to extract column lineage with proper target tracking."""
        records = []
        
        where_expr = select_node.args.get("where")
        if where_expr:
            where_condition_sql = where_expr.sql(dialect="postgres")
            columns = self._extract_source_columns_from_expr(where_expr.this)
            
            # Check if this is from a CTE
            sql_process = "where"
            if select_node.args.get("with"):
                sql_process = "where-with"
            
            for col_info in columns:
                col_data = {"source_column_expr": col_info}
                resolved = self._resolve_source_column(col_data, select_node, placeholders)
                
                # Get the target column name
                target_col_name = col_info.get("column")
                table_alias = col_info.get("table")
                
                # Get actual source logic from CTE if applicable
                source_logic_detail = self._get_source_logic_detail(
                    table_alias,
                    target_col_name,
                    select_node,
                    placeholders
                )
                
                # Use the CTE source logic if available, otherwise use the WHERE condition
                if source_logic_detail:
                    logic = source_logic_detail
                else:
                    logic = where_condition_sql
                
                for src_rec in resolved:
                    records.append({
                        "sub_target_schema": target_schema,
                        "sub_target_table": target_table,
                        "target_column": target_col_name,
                        "source_schema": src_rec.get("source_schema"),
                        "source_table": src_rec.get("source_table"),
                        "source_column": src_rec.get("source_column"),
                        "logic": logic,
                        "sql_process": sql_process
                    })
        
        return records
    
    def _process_having(
        self, 
        select_node: exp.Select, 
        placeholders: Dict[str, str],
        target_table: Optional[str],
        target_schema: Optional[str]
    ) -> List[Dict]:
        """Process HAVING clause to extract column lineage with proper target tracking."""
        records = []
        
        having_expr = select_node.args.get("having")
        if having_expr:
            having_condition_sql = having_expr.sql(dialect="postgres")
            columns = self._extract_source_columns_from_expr(having_expr.this)
            
            # Check if this is from a CTE
            sql_process = "having"
            if select_node.args.get("with"):
                sql_process = "having-with"
            
            for col_info in columns:
                col_data = {"source_column_expr": col_info}
                resolved = self._resolve_source_column(col_data, select_node, placeholders)
                
                # Get the target column name
                target_col_name = col_info.get("column")
                table_alias = col_info.get("table")
                
                # Get actual source logic from CTE if applicable
                source_logic_detail = self._get_source_logic_detail(
                    table_alias,
                    target_col_name,
                    select_node,
                    placeholders
                )
                
                # Use the CTE source logic if available, otherwise use the HAVING condition
                if source_logic_detail:
                    logic = source_logic_detail
                else:
                    logic = having_condition_sql
                
                for src_rec in resolved:
                    records.append({
                        "sub_target_schema": target_schema,
                        "sub_target_table": target_table,
                        "target_column": target_col_name,
                        "source_schema": src_rec.get("source_schema"),
                        "source_table": src_rec.get("source_table"),
                        "source_column": src_rec.get("source_column"),
                        "logic": logic,
                        "sql_process": sql_process
                    })
        
        return records
    
    def _process_select_statement(
        self, 
        select_node: exp.Select, 
        placeholders: Dict[str, str],
        target_table: Optional[str]
    ) -> List[Dict]:
        """Process standalone SELECT statement."""
        # For standalone SELECT, we don't have a target table
        return self._process_select_for_target(select_node, None, None, placeholders)
    
    def _process_insert_statement(self, insert_node: exp.Insert, placeholders: Dict[str, str]) -> List[Dict]:
        """Process INSERT statement."""
        records = []
        
        # Get target table
        target_table = None
        target_schema = None
        if insert_node.this and isinstance(insert_node.this, exp.Table):
            target_table = insert_node.this.name
            if insert_node.this.db:
                target_schema = self._resolve_template(insert_node.this.db, placeholders)
        
        # Process the SELECT part
        select_expr = insert_node.expression
        if select_expr and isinstance(select_expr, exp.Select):
            records.extend(self._process_select_for_target(
                select_expr, target_table, target_schema, placeholders
            ))
        
        return records
    
    def _resolve_template(self, template_str: str, placeholders: Dict[str, str]) -> str:
        """Resolve a template placeholder to its original value."""
        if template_str in placeholders:
            return placeholders[template_str]
        return template_str
    
    # Utility methods from original parser
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
            token = f"TEMPLATE_TOKEN_{len(placeholders)}"
            placeholders[token] = f"{{{{{inner}}}}}"
            return token

        sanitized = self.TEMPLATE_PATTERN.sub(repl, sql_text)
        return sanitized, placeholders
    
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


class ColumnLineageBuilder:
    def __init__(self, fetcher: GitLabSQLFetcher, parser: ColumnLineageParser) -> None:
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
                lineage_records = self._parser.extract_column_lineage(sql_text)
                
                filename = Path(file_path).name
                target_table = Path(filename).stem
                process = derive_process(file_path)
                
                for record in lineage_records:
                    row = ColumnLineageRow(
                        filename=filename,
                        filepath=file_path,
                        process=process,
                        target_table=target_table,
                        sub_target_schema=record.get("sub_target_schema"),
                        sub_target_table=record.get("sub_target_table"),
                        target_column=record.get("target_column"),
                        source_schema=record.get("source_schema"),
                        source_table=record.get("source_table"),
                        source_column=record.get("source_column"),
                        logic=record.get("logic"),
                        sql_process=record.get("sql_process"),
                        current_timestamp=timestamp,
                    )
                    rows.append(row)
                    
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
        output_path = f"{OUTPUT_PREFIX}_auto_refresh_{timestamp_str}.xlsx"
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
    parser = ColumnLineageParser()
    builder = ColumnLineageBuilder(fetcher, parser)
    run_timestamp = datetime.datetime.utcnow()
    output_file = f"{OUTPUT_PREFIX}_auto_refresh_{run_timestamp.strftime('%Y%m%d%H%M%S')}.xlsx"
    # Initialize the Excel file with headers
    write_to_excel(rows_to_dataframe([]), output_path=output_file)

    def flush_excel(current_rows: Sequence[ColumnLineageRow]) -> None:
        df_snapshot = rows_to_dataframe(current_rows)
        write_to_excel(df_snapshot, output_path=output_file)

    rows = builder.build(progress_callback=flush_excel, run_timestamp=run_timestamp)
    logging.info("Captured %d column lineage rows", len(rows))

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
