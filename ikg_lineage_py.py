"""
IKG Column Lineage Master Auto Refresh
Updated with CTE (Common Table Expression) Support

This script extracts column-level lineage from SQL queries including:
- Simple SELECT statements
- Complex queries with JOINs
- Common Table Expressions (WITH clauses)
- CASE statements and functions
"""

import re
import sqlparse
from sqlparse.sql import IdentifierList, Identifier, Where, Parenthesis, Function
from sqlparse.tokens import Keyword, DML
from typing import Dict, List, Tuple, Set, Optional
import pandas as pd
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CTEColumnLineageParser:
    """
    Enhanced parser to handle CTE (Common Table Expressions) in SQL queries
    for column lineage tracking.
    """
    
    def __init__(self, schema_placeholder="{{params.IKG_SCHEMA}}"):
        self.schema_placeholder = schema_placeholder
        self.cte_definitions = {}
        self.cte_columns = {}
        self.table_aliases = {}
        
    def parse_sql_with_cte(self, sql_text: str, target_schema: str = None, 
                          target_table: str = None) -> List[Dict]:
        """
        Main method to parse SQL with CTEs and extract lineage
        
        Args:
            sql_text: SQL query text
            target_schema: Target schema name
            target_table: Target table name (will be auto-detected if None)
        
        Returns:
            List of dictionaries with column lineage information
        """
        try:
            # Clean and parse SQL
            sql_text = self._clean_sql(sql_text)
            
            # Extract target table if not provided
            if not target_table:
                target_table = self._extract_target_table(sql_text)
            
            # Check if SQL contains CTE
            if self._has_cte(sql_text):
                logger.info(f"Processing CTE query for table: {target_table}")
                return self._parse_cte_query(sql_text, target_schema, target_table)
            else:
                logger.info(f"Processing simple query for table: {target_table}")
                return self._parse_simple_query(sql_text, target_schema, target_table)
                
        except Exception as e:
            logger.error(f"Error parsing SQL: {str(e)}")
            return []
    
    def _clean_sql(self, sql: str) -> str:
        """Clean SQL text by removing comments"""
        # Remove single-line comments
        sql = re.sub(r'--[^\n]*', '', sql)
        # Remove multi-line comments
        sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
        return sql.strip()
    
    def _has_cte(self, sql: str) -> bool:
        """Check if SQL contains WITH clause (CTE)"""
        return bool(re.search(r'\bWITH\b', sql, re.IGNORECASE))
    
    def _extract_target_table(self, sql: str) -> str:
        """Extract target table name from CREATE TEMP TABLE or INSERT"""
        # Match: CREATE TEMP TABLE table_name
        match = re.search(r'CREATE\s+(?:TEMP\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)', 
                         sql, re.IGNORECASE)
        if match:
            return match.group(1)
        
        # Match: INSERT INTO table_name
        match = re.search(r'INSERT\s+INTO\s+(\w+)', sql, re.IGNORECASE)
        if match:
            return match.group(1)
        
        return "unknown_table"
    
    def _parse_cte_query(self, sql: str, target_schema: str, 
                        target_table: str) -> List[Dict]:
        """Parse SQL query with CTEs"""
        try:
            # Split CTEs and main query
            cte_section, main_query = self._split_cte_and_main(sql)
            
            # Parse each CTE definition
            ctes = self._parse_cte_definitions(cte_section)
            
            # Parse main query
            columns = self._parse_main_query(main_query, ctes, target_schema, target_table)
            
            return columns
            
        except Exception as e:
            logger.error(f"Error parsing CTE query: {str(e)}")
            return []
    
    def _split_cte_and_main(self, sql: str) -> Tuple[str, str]:
        """
        Split SQL into CTE section and main query
        Returns: (cte_section, main_query)
        """
        with_match = re.search(r'\bWITH\b', sql, re.IGNORECASE)
        if not with_match:
            return "", sql
        
        with_start = with_match.start()
        after_with = sql[with_start + 4:].strip()
        
        # Find main SELECT by tracking parentheses depth
        paren_depth = 0
        main_select_pos = -1
        
        i = 0
        while i < len(after_with):
            char = after_with[i]
            
            if char == '(':
                paren_depth += 1
            elif char == ')':
                paren_depth -= 1
            
            # At depth 0, find SELECT keyword
            if paren_depth == 0:
                if re.match(r'\bSELECT\b', after_with[i:], re.IGNORECASE):
                    main_select_pos = i
                    break
            
            i += 1
        
        if main_select_pos == -1:
            # Fallback: find last SELECT
            matches = list(re.finditer(r'\bSELECT\b', after_with, re.IGNORECASE))
            if matches:
                main_select_pos = matches[-1].start()
        
        cte_section = after_with[:main_select_pos].strip() if main_select_pos > 0 else ""
        main_query = after_with[main_select_pos:].strip() if main_select_pos >= 0 else after_with
        
        return cte_section, main_query
    
    def _parse_cte_definitions(self, cte_section: str) -> Dict:
        """
        Parse CTE definitions to extract column lineage within each CTE
        
        Returns:
            Dict[cte_name] = {
                'columns': {column_name: source_info},
                'source_tables': [table_info]
            }
        """
        ctes = {}
        
        # Split by commas at parentheses depth 0
        cte_defs = self._split_cte_definitions(cte_section)
        
        for cte_def in cte_defs:
            cte_name, cte_info = self._parse_single_cte(cte_def)
            if cte_name:
                ctes[cte_name] = cte_info
                logger.debug(f"Parsed CTE: {cte_name} with {len(cte_info['columns'])} columns")
        
        return ctes
    
    def _split_cte_definitions(self, cte_section: str) -> List[str]:
        """Split multiple CTE definitions"""
        cte_defs = []
        current_def = ""
        paren_depth = 0
        
        for char in cte_section:
            if char == '(':
                paren_depth += 1
            elif char == ')':
                paren_depth -= 1
            elif char == ',' and paren_depth == 0:
                if current_def.strip():
                    cte_defs.append(current_def.strip())
                current_def = ""
                continue
            
            current_def += char
        
        if current_def.strip():
            cte_defs.append(current_def.strip())
        
        return cte_defs
    
    def _parse_single_cte(self, cte_def: str) -> Tuple[Optional[str], Optional[Dict]]:
        """Parse a single CTE definition"""
        # Extract CTE name
        match = re.match(r'(\w+)\s+[Aa][Ss]\s*\(', cte_def)
        if not match:
            return None, None
        
        cte_name = match.group(1)
        
        # Extract SELECT query inside AS (...)
        as_pos = match.end() - 1
        query = self._extract_balanced_parentheses(cte_def[as_pos:])
        
        # Parse the SELECT query
        cte_info = self._parse_cte_select(query)
        
        return cte_name, cte_info
    
    def _extract_balanced_parentheses(self, text: str) -> str:
        """Extract content within balanced parentheses"""
        if not text.startswith('('):
            return text
        
        depth = 0
        for i, char in enumerate(text):
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
                if depth == 0:
                    return text[1:i]
        
        return text[1:-1] if text.startswith('(') and text.endswith(')') else text
    
    def _parse_cte_select(self, select_query: str) -> Dict:
        """Parse SELECT query within a CTE"""
        # Extract source tables
        source_tables = self._extract_source_tables(select_query)
        
        # Extract columns
        columns = self._extract_select_columns(select_query, source_tables)
        
        return {
            'columns': columns,
            'source_tables': source_tables
        }
    
    def _extract_source_tables(self, select_query: str) -> List[Dict]:
        """Extract source tables from FROM and JOIN clauses"""
        tables = []
        
        # Find FROM clause
        from_match = re.search(
            r'\bFROM\b(.*?)(?:\bWHERE\b|\bGROUP BY\b|\bORDER BY\b|\bDISTRIBUTED BY\b|$)',
            select_query, re.IGNORECASE | re.DOTALL
        )
        
        if not from_match:
            return tables
        
        from_clause = from_match.group(1)
        
        # Pattern to match schema.table alias or table alias
        # Handles {{params.IKG_SCHEMA}}.table_name alias
        pattern = r'(?:(\{\{[^}]+\}\}|[\w]+)\.)?([\w]+)\s+(\w+)'
        
        matches = re.finditer(pattern, from_clause, re.IGNORECASE)
        
        for match in matches:
            groups = match.groups()
            if len(groups) == 3:
                schema, table, alias = groups
                # Skip SQL keywords
                if table.upper() in ('LEFT', 'RIGHT', 'INNER', 'OUTER', 'JOIN', 'ON', 'USING'):
                    continue
                    
                tables.append({
                    'schema': schema if schema else None,
                    'table': table,
                    'alias': alias
                })
        
        return tables
    
    def _extract_select_columns(self, select_query: str, 
                               source_tables: List[Dict]) -> Dict:
        """Extract column mappings from SELECT clause"""
        columns = {}
        
        # Find SELECT clause
        select_match = re.search(
            r'\bSELECT\b(.*?)\bFROM\b',
            select_query, re.IGNORECASE | re.DOTALL
        )
        
        if not select_match:
            return columns
        
        select_clause = select_match.group(1).strip()
        
        # Handle DISTINCT
        select_clause = re.sub(r'^\s*DISTINCT\s+', '', select_clause, flags=re.IGNORECASE)
        
        # Split columns
        column_exprs = self._split_select_columns(select_clause)
        
        for col_expr in column_exprs:
            col_expr = col_expr.strip()
            if not col_expr:
                continue
            
            col_info = self._parse_column_expression(col_expr, source_tables)
            if col_info:
                columns[col_info['target_column']] = col_info
        
        return columns
    
    def _split_select_columns(self, select_clause: str) -> List[str]:
        """Split SELECT columns respecting CASE/function parentheses"""
        columns = []
        current = ""
        paren_depth = 0
        case_depth = 0
        
        i = 0
        while i < len(select_clause):
            # Check for CASE keyword
            if select_clause[i:i+4].upper() == 'CASE':
                case_depth += 1
                current += select_clause[i:i+4]
                i += 4
                continue
            elif select_clause[i:i+3].upper() == 'END':
                if case_depth > 0:
                    case_depth -= 1
                current += select_clause[i:i+3]
                i += 3
                continue
            
            char = select_clause[i]
            
            if char == '(':
                paren_depth += 1
            elif char == ')':
                paren_depth -= 1
            elif char == ',' and paren_depth == 0 and case_depth == 0:
                columns.append(current.strip())
                current = ""
                i += 1
                continue
            
            current += char
            i += 1
        
        if current.strip():
            columns.append(current.strip())
        
        return columns
    
    def _parse_column_expression(self, col_expr: str, 
                                 source_tables: List[Dict]) -> Optional[Dict]:
        """Parse a single column expression"""
        # Check for AS alias
        as_match = re.search(r'\s+[Aa][Ss]\s+(\w+)\s*$', col_expr)
        
        if as_match:
            target_column = as_match.group(1)
            source_expr = col_expr[:as_match.start()].strip()
        else:
            # No alias - extract column name
            simple_match = re.match(r'(?:(\w+)\.)?(\w+)$', col_expr.strip())
            if simple_match:
                target_column = simple_match.group(2)
                source_expr = col_expr.strip()
            else:
                # Complex expression without alias
                target_column = f"expr_{abs(hash(col_expr)) % 10000}"
                source_expr = col_expr.strip()
        
        # Parse source expression
        if re.search(r'\bCASE\b', source_expr, re.IGNORECASE):
            source_info = self._parse_case_expression(source_expr, source_tables)
        elif '(' in source_expr:
            source_info = self._parse_function_expression(source_expr, source_tables)
        else:
            source_info = self._parse_simple_column(source_expr, source_tables)
        
        if source_info:
            source_info['target_column'] = target_column
        
        return source_info
    
    def _parse_simple_column(self, col_ref: str, 
                            source_tables: List[Dict]) -> Optional[Dict]:
        """Parse simple column reference"""
        match = re.match(r'(?:(\w+)\.)?(\w+)', col_ref.strip())
        
        if not match:
            return None
        
        alias = match.group(1)
        column = match.group(2)
        
        # Find source table
        if alias:
            for table in source_tables:
                if table['alias'] == alias:
                    return {
                        'source_table': table['table'],
                        'source_schema': table['schema'],
                        'source_column': column,
                        'source_alias': alias
                    }
        
        # No alias - use first table
        if source_tables:
            return {
                'source_table': source_tables[0]['table'],
                'source_schema': source_tables[0]['schema'],
                'source_column': column,
                'source_alias': source_tables[0]['alias']
            }
        
        return {
            'source_table': None,
            'source_schema': None,
            'source_column': column,
            'source_alias': None
        }
    
    def _parse_case_expression(self, case_expr: str, 
                              source_tables: List[Dict]) -> Dict:
        """Extract columns used in CASE expression"""
        # Extract all column references
        column_refs = re.findall(r'(\w+)\.(\w+)', case_expr)
        
        if column_refs:
            alias, column = column_refs[0]
            for table in source_tables:
                if table['alias'] == alias:
                    return {
                        'source_table': table['table'],
                        'source_schema': table['schema'],
                        'source_column': column,
                        'source_alias': alias,
                        'is_case': True
                    }
        
        return {
            'source_table': None,
            'source_schema': None,
            'source_column': 'CASE_EXPRESSION',
            'source_alias': None,
            'is_case': True
        }
    
    def _parse_function_expression(self, func_expr: str, 
                                   source_tables: List[Dict]) -> Dict:
        """Parse function expression"""
        # Extract column references inside function
        column_refs = re.findall(r'(\w+)\.(\w+)', func_expr)
        
        if column_refs:
            alias, column = column_refs[0]
            for table in source_tables:
                if table['alias'] == alias:
                    return {
                        'source_table': table['table'],
                        'source_schema': table['schema'],
                        'source_column': column,
                        'source_alias': alias,
                        'is_function': True
                    }
        
        return {
            'source_table': None,
            'source_schema': None,
            'source_column': func_expr,
            'source_alias': None,
            'is_function': True
        }
    
    def _parse_main_query(self, main_query: str, ctes: Dict, 
                         target_schema: str, target_table: str) -> List[Dict]:
        """Parse main SELECT query that uses CTEs"""
        results = []
        
        # Extract SELECT columns
        select_match = re.search(
            r'\bSELECT\b(.*?)\bFROM\b',
            main_query, re.IGNORECASE | re.DOTALL
        )
        
        if not select_match:
            return results
        
        select_clause = select_match.group(1).strip()
        
        # Extract FROM clause to identify CTE references
        from_match = re.search(
            r'\bFROM\b(.*?)(?:\bWHERE\b|\bGROUP BY\b|\bLEFT JOIN\b|\bINNER JOIN\b|\bDISTRIBUTED BY\b|$)',
            main_query, re.IGNORECASE | re.DOTALL
        )
        
        from_clause = from_match.group(1) if from_match else ""
        
        # Find CTE references
        cte_refs = self._extract_cte_references(from_clause, ctes)
        
        # Parse SELECT columns
        column_exprs = self._split_select_columns(select_clause)
        
        for col_expr in column_exprs:
            col_expr = col_expr.strip()
            if not col_expr:
                continue
            
            col_info = self._parse_main_query_column(
                col_expr, cte_refs, ctes, target_schema, target_table
            )
            if col_info:
                results.append(col_info)
        
        return results
    
    def _extract_cte_references(self, from_clause: str, ctes: Dict) -> List[Dict]:
        """Extract CTE references from FROM clause"""
        cte_refs = []
        
        for cte_name in ctes.keys():
            pattern = rf'\b{cte_name}\b\s+(\w+)'
            matches = re.finditer(pattern, from_clause, re.IGNORECASE)
            for match in matches:
                alias = match.group(1)
                cte_refs.append({
                    'cte_name': cte_name,
                    'alias': alias
                })
        
        return cte_refs
    
    def _parse_main_query_column(self, col_expr: str, cte_refs: List[Dict],
                                ctes: Dict, target_schema: str, 
                                target_table: str) -> Optional[Dict]:
        """Parse column from main query and trace to CTE source"""
        # Extract target column name
        as_match = re.search(r'\s+[Aa][Ss]\s+(\w+)\s*$', col_expr)
        
        if as_match:
            target_column = as_match.group(1)
            source_expr = col_expr[:as_match.start()].strip()
        else:
            simple_match = re.match(r'(?:(\w+)\.)?(\w+)$', col_expr.strip())
            if simple_match:
                target_column = simple_match.group(2)
                source_expr = col_expr.strip()
            else:
                return None
        
        # Parse source expression (e.g., hh.household_plus)
        match = re.match(r'(\w+)\.(\w+)', source_expr.strip())
        
        if not match:
            return None
        
        alias = match.group(1)
        column = match.group(2)
        
        # Find which CTE this alias refers to
        cte_name = None
        for ref in cte_refs:
            if ref['alias'] == alias:
                cte_name = ref['cte_name']
                break
        
        if not cte_name or cte_name not in ctes:
            return None
        
        # Trace column through CTE
        cte_info = ctes[cte_name]
        if column in cte_info['columns']:
            source_info = cte_info['columns'][column]
            
            # Determine sql_process type
            sql_process = self._determine_sql_process(source_info, cte_name, alias, column)
            
            return {
                'sub_target_schema': target_schema,
                'sub_target_table': target_table,
                'target_column': target_column,
                'source_schema': source_info['source_schema'],
                'source_table': source_info['source_table'],
                'source_column': source_info['source_column'],
                'logic': self._build_logic_description(source_info, cte_name, alias, column),
                'sql_process': sql_process,
                'comments': f"Traced through CTE {cte_name}"
            }
        
        return None
    
    def _determine_sql_process(self, source_info: Dict, cte_name: str,
                               alias: str, column: str) -> str:
        """Determine the SQL process type"""
        if source_info.get('is_case'):
            return 'select'
        elif source_info.get('is_function'):
            return 'select'
        elif source_info.get('source_table'):
            # Check if it's a join based on CTE usage
            if cte_name and alias:
                return 'join' if 'JOIN' in cte_name.upper() else 'select'
            return 'select'
        return 'select'
    
    def _build_logic_description(self, source_info: Dict, cte_name: str,
                                 alias: str, column: str) -> str:
        """Build human-readable logic description"""
        if source_info.get('is_case'):
            return f"{alias}.{column} (from CTE {cte_name}, CASE expression)"
        elif source_info.get('is_function'):
            return f"{alias}.{column} (from CTE {cte_name}, function)"
        else:
            src_col = source_info.get('source_column', column)
            src_alias = source_info.get('source_alias', '')
            if src_alias:
                return f"{alias}.{column} (from CTE {cte_name}: {src_alias}.{src_col})"
            else:
                return f"{alias}.{column} (from CTE {cte_name})"
    
    def _parse_simple_query(self, sql: str, target_schema: str,
                           target_table: str) -> List[Dict]:
        """Parse queries without CTEs"""
        results = []
        
        try:
            # Extract source tables
            source_tables = self._extract_source_tables(sql)
            
            # Extract SELECT clause
            select_match = re.search(
                r'\bSELECT\b(.*?)\bFROM\b',
                sql, re.IGNORECASE | re.DOTALL
            )
            
            if not select_match:
                return results
            
            select_clause = select_match.group(1).strip()
            column_exprs = self._split_select_columns(select_clause)
            
            for col_expr in column_exprs:
                col_expr = col_expr.strip()
                if not col_expr:
                    continue
                
                col_info = self._parse_column_expression(col_expr, source_tables)
                if col_info:
                    results.append({
                        'sub_target_schema': target_schema,
                        'sub_target_table': target_table,
                        'target_column': col_info['target_column'],
                        'source_schema': col_info.get('source_schema'),
                        'source_table': col_info.get('source_table'),
                        'source_column': col_info.get('source_column'),
                        'logic': col_info.get('source_column', ''),
                        'sql_process': 'select',
                        'comments': 'Simple query (no CTE)'
                    })
            
        except Exception as e:
            logger.error(f"Error parsing simple query: {str(e)}")
        
        return results


def process_sql_file(file_path: str, schema_name: str = None) -> pd.DataFrame:
    """
    Process a SQL file and extract column lineage
    
    Args:
        file_path: Path to SQL file
        schema_name: Schema name (optional)
    
    Returns:
        DataFrame with column lineage information
    """
    try:
        with open(file_path, 'r') as f:
            sql_content = f.read()
        
        parser = CTEColumnLineageParser()
        lineage_data = parser.parse_sql_with_cte(sql_content, schema_name)
        
        if not lineage_data:
            logger.warning(f"No lineage data extracted from {file_path}")
            return pd.DataFrame()
        
        df = pd.DataFrame(lineage_data)
        logger.info(f"Extracted {len(df)} column mappings from {file_path}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        return pd.DataFrame()


def process_sql_directory(directory_path: str, schema_name: str = None,
                         output_file: str = None) -> pd.DataFrame:
    """
    Process all SQL files in a directory
    
    Args:
        directory_path: Path to directory containing SQL files
        schema_name: Schema name (optional)
        output_file: Output CSV file path (optional)
    
    Returns:
        Combined DataFrame with all column lineage
    """
    import os
    
    all_lineage = []
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.sql'):
            file_path = os.path.join(directory_path, filename)
            logger.info(f"Processing {filename}...")
            
            df = process_sql_file(file_path, schema_name)
            if not df.empty:
                all_lineage.append(df)
    
    if not all_lineage:
        logger.warning("No lineage data extracted from any files")
        return pd.DataFrame()
    
    combined_df = pd.concat(all_lineage, ignore_index=True)
    
    if output_file:
        combined_df.to_csv(output_file, index=False)
        logger.info(f"Saved combined lineage to {output_file}")
    
    return combined_df


# Main execution
if __name__ == "__main__":
    # Example usage
    example_sql = """
    DROP TABLE IF EXISTS fee_waiver_hh_2m_4m;
    CREATE TEMP TABLE fee_waiver_hh_2m_4m AS 
    WITH hh_Filtered AS (
    SELECT
    b.acc_mhh_n AS household_plus, 
    b.mhh_assets_curr AS shh_assets_curr,
    b.max_marketing_nh_assets_curr AS mh_assets_curr_max,
    CASE WHEN e.acc_mhh_n IS NULL THEN 'N'
    ELSE 'Y'
    END AS is_employee_hh
    FROM {{params.IKG_SCHEMA}}.base_feature_shhp_ikg b
    LEFT JOIN {{params.IKG_SCHEMA}}.employee_household_ikg e
    ON b.acc_mhh_n = e.acc_mhh_n
    ),
    acc_mapping as (
    SELECT DISTINCT acc_n, acc_mhh_n AS household_plus, acc_i
    FROM {{params.IKG_SCHEMA}}.master_ids_curr_ikg
    )
    SELECT
    acc.acc_n,
    hh.household_plus,
    acc.acc_i,
    hh.shh_assets_curr,
    hh.mh_assets_curr_max,
    hh.is_employee_hh
    FROM hh_Filtered hh
    LEFT JOIN acc_mapping acc
    ON hh.household_plus = acc.household_plus
    DISTRIBUTED BY (acc_n);
    """
    
    parser = CTEColumnLineageParser()
    results = parser.parse_sql_with_cte(example_sql, "{{params.IKG_SCHEMA}}")
    
    # Convert to DataFrame and display
    df = pd.DataFrame(results)
    print("\n" + "="*100)
    print("COLUMN LINEAGE RESULTS")
    print("="*100)
    print(df.to_string(index=False))
    