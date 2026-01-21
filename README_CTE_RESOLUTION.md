# IKG Column Lineage Master - CTE Alias Resolution

## Key Improvements

### Problem Solved
When SQL queries use CTEs (Common Table Expressions) with aliases in FROM/JOIN clauses, the parser now correctly resolves these aliases back to their original source tables.

**Example:**
```sql
WITH hh_Filtered AS (
    SELECT b.acc_mhh_n AS household_plus
    FROM schema1.base_feature_shhp_ikg b
),
acc_mapping AS (
    SELECT acc_n, acc_mhh_n AS household_plus
    FROM schema1.master_ids_curr_ikg
)
SELECT 
    hh.household_plus,  -- alias 'hh' refers to CTE 'hh_Filtered'
    acc.acc_n           -- alias 'acc' refers to CTE 'acc_mapping'
FROM hh_Filtered hh
LEFT JOIN acc_mapping acc
ON hh.household_plus = acc.household_plus
```

### Before Fix
- Source table would be incorrectly identified as `hh` or `acc` (the aliases)
- No resolution to the actual source tables inside the CTEs

### After Fix
- `hh.household_plus` correctly resolves to:
  - source_table: `base_feature_shhp_ikg`
  - source_column: `acc_mhh_n`
  - source_schema: `schema1`

- `acc.acc_n` correctly resolves to:
  - source_table: `master_ids_curr_ikg`
  - source_column: `acc_n`
  - source_schema: `schema1`

## How It Works

### 1. CTE Definition Extraction
```python
def _extract_cte_definitions(self, with_node, placeholders):
    """Extract all CTE definitions first"""
    # Stores CTE name -> SELECT statement mapping
    # e.g., 'hh_Filtered' -> SELECT ...
```

### 2. Table Alias Mapping with CTE Detection
```python
def _build_table_aliases(self, select_node, placeholders):
    """Build map of aliases to tables, marking CTEs"""
    # Maps alias to table info with CTE flag
    # e.g., 'hh' -> {'table': 'hh_Filtered', 'is_cte': True}
    # e.g., 'b' -> {'table': 'base_feature_shhp_ikg', 'is_cte': False}
```

### 3. Recursive CTE Resolution
```python
def _resolve_source_column(self, column_data, select_node, placeholders):
    """Resolve column through CTE chain to actual source"""
    # When encountering 'hh.household_plus':
    # 1. Check table_aliases: 'hh' -> 'hh_Filtered' (CTE)
    # 2. Call _resolve_from_cte('hh_Filtered', 'household_plus')
    # 3. Look in CTE definition for 'household_plus'
    # 4. Find it maps to 'b.acc_mhh_n'
    # 5. Recursively resolve 'b' -> 'base_feature_shhp_ikg'
    # 6. Return actual source
```

### 4. Multi-Level CTE Support
The parser supports nested CTEs and multiple levels of resolution:
```sql
WITH level1 AS (
    SELECT col1 FROM real_table
),
level2 AS (
    SELECT col1 AS col2 FROM level1
)
SELECT col2 FROM level2
```
Correctly resolves: `col2` -> `col1` -> `real_table.col1`

## Output Schema

### Column Lineage Table: `sandbox_prj_smart_insights.ikg_column_lineage_master_auto_refresh`

| Column | Description |
|--------|-------------|
| filename | SQL script filename |
| filepath | Full path in GitLab |
| process | Process folder (from SQL_PATH structure) |
| target_table | Main table being created (filename without .sql) |
| sub_target_schema | Schema of the specific table created in script |
| sub_target_table | Specific table created (including temp tables) |
| target_column | Column name in the target table |
| source_schema | Schema of the actual source table |
| source_table | Actual source table name (NOT alias, NOT CTE name) |
| source_column | Column name in the source table |
| logic | SQL expression for the column |
| sql_process | Type of SQL operation |
| current_timestamp | When lineage was extracted |

### SQL Process Types

- **create**: Whole table CREATE statement
- **select**: Column mappings from SELECT clause
- **join**: Columns used in JOIN conditions
- **join-with**: JOIN conditions inside queries with CTEs
- **where**: Columns used in WHERE clause
- **where-with**: WHERE conditions inside queries with CTEs
- **having**: Columns used in HAVING clause
- **having-with**: HAVING conditions inside queries with CTEs

## Usage

### Python Script
```bash
python ikg_column_lineage_master_auto_refresh.py
```

### Jupyter Notebook
Open `ikg_column_lineage_master_auto_refresh.ipynb` and run all cells.

Both will:
1. Fetch all SQL files from GitLab (branch: ikg-master)
2. Parse each file for column-level lineage
3. Resolve all CTEs to actual source tables
4. Generate Excel report
5. Upload to Greenplum database

## Configuration

Edit these constants in the script:
- `GITLAB_URL`: GitLab instance URL
- `IKG_PROJECT_PATH`: Project path
- `BRANCH`: Git branch to process
- `SQL_PATH`: Path to SQL scripts
- `EXCLUDE_FOLDER`: Folders to skip
- `TARGET_SCHEMA`: Output schema
- `TARGET_TABLE`: Output table name

## Requirements

- Python 3.7+
- python-gitlab
- sqlglot
- pandas
- openpyxl
- psycopg2

## Advanced Features

### Template Parameter Support
Handles Jinja-style templates:
```sql
FROM {{params.IKG_SCHEMA}}.table_name
```
Preserves the template in source_schema.

### Vendor-Specific SQL Stripping
Removes Greenplum-specific syntax before parsing:
- DISTRIBUTED BY clauses
- WITH (appendonly) options
- ENCODE specifications
- etc.

### DO Block Processing
Parses procedural code in DO $$ ... $$ blocks.

### Complex Expression Handling
Extracts columns from:
- CASE WHEN expressions
- Aggregate functions
- Window functions
- Subqueries
- Complex conditions

## Troubleshooting

### Issue: CTE still showing as source table
- Check that the CTE is defined in a WITH clause before it's used
- Verify the alias mapping is correct
- Check logs for parsing errors

### Issue: Missing columns
- Some columns may be from constants (logic will show the constant)
- Columns from * expansions are not tracked (need explicit column names)

### Issue: Multiple source tables for one column
- When column name is ambiguous and appears in multiple tables
- Parser returns all possibilities
- Can be resolved with INFORMATION_SCHEMA lookup (future enhancement)
