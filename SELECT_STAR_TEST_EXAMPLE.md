# SELECT * Expansion - Test Example

This document shows how the SELECT * expansion feature works with a complete example.

## Test SQL Script

```sql
-- File: test_select_star.sql
-- Location: dags/ikg/scripts/sql/reporting/test_select_star.sql

-- Example 1: CREATE TABLE with SELECT * from CTE
DROP TABLE IF EXISTS employee_summary;
CREATE TEMP TABLE employee_summary AS
WITH employee_data AS (
    SELECT 
        emp_id,
        emp_name,
        dept_id,
        salary,
        hire_date
    FROM hr_schema.employees
),
dept_data AS (
    SELECT
        dept_id,
        dept_name,
        manager_name
    FROM hr_schema.departments
)
SELECT ed.*
FROM employee_data ed
LEFT JOIN dept_data dd ON ed.dept_id = dd.dept_id
WHERE ed.salary > 50000
DISTRIBUTED BY (emp_id);

-- Example 2: CREATE TABLE with SELECT * from multiple tables
DROP TABLE IF EXISTS combined_data;
CREATE TEMP TABLE combined_data AS
SELECT *
FROM employee_summary es
JOIN dept_data dd ON es.dept_id = dd.dept_id;

-- Example 3: SELECT table.* with specific columns
DROP TABLE IF EXISTS custom_report;
CREATE TEMP TABLE custom_report AS
SELECT 
    es.*,
    'Active' AS status,
    CURRENT_DATE AS report_date
FROM employee_summary es
WHERE es.hire_date > '2020-01-01';
```

## Expected Output Records

### For `employee_summary` table:

#### Record 1: CREATE statement
```
filename: test_select_star.sql
filepath: dags/ikg/scripts/sql/reporting/test_select_star.sql
process: reporting
target_table: test_select_star
sub_target_schema: NULL
sub_target_table: employee_summary
target_column: NULL
source_schema: NULL
source_table: NULL
source_column: NULL
logic: DROP TABLE IF EXISTS employee_summary;
       CREATE TEMP TABLE employee_summary AS
       WITH employee_data AS (...)
       SELECT ed.*
       FROM employee_data ed
       ...
       DISTRIBUTED BY (emp_id);
sql_process: create
```

#### Records 2-6: SELECT * expansion (ed.*)
```
Record 2:
filename: test_select_star.sql
filepath: dags/ikg/scripts/sql/reporting/test_select_star.sql
process: reporting
target_table: test_select_star
sub_target_schema: NULL
sub_target_table: employee_summary
target_column: emp_id
source_schema: hr_schema
source_table: employees
source_column: emp_id
logic: emp_id
sql_process: select*

Record 3:
target_column: emp_name
source_schema: hr_schema
source_table: employees
source_column: emp_name
logic: emp_name
sql_process: select*

Record 4:
target_column: dept_id
source_schema: hr_schema
source_table: employees
source_column: dept_id
logic: dept_id
sql_process: select*

Record 5:
target_column: salary
source_schema: hr_schema
source_table: employees
source_column: salary
logic: salary
sql_process: select*

Record 6:
target_column: hire_date
source_schema: hr_schema
source_table: employees
source_column: hire_date
logic: hire_date
sql_process: select*
```

#### Record 7: JOIN condition
```
target_column: dept_id
source_schema: hr_schema
source_table: employees
source_column: dept_id
logic: ON ed.dept_id = dd.dept_id (ed.dept_id = dept_id)
sql_process: join
```

#### Record 8: WHERE condition
```
target_column: salary
source_schema: hr_schema
source_table: employees
source_column: salary
logic: salary
sql_process: where
```

---

### For `combined_data` table (SELECT * from multiple tables):

#### Record 1: CREATE statement
```
sub_target_table: combined_data
sql_process: create
logic: DROP TABLE IF EXISTS combined_data;
       CREATE TEMP TABLE combined_data AS
       SELECT *
       FROM employee_summary es
       JOIN dept_data dd ON es.dept_id = dd.dept_id;
```

#### Records 2-6: SELECT * expansion from employee_summary
```
Record 2:
sub_target_table: combined_data
target_column: emp_id
source_schema: hr_schema
source_table: employees
source_column: emp_id
sql_process: select*

Record 3:
target_column: emp_name
source_schema: hr_schema
source_table: employees
source_column: emp_name
sql_process: select*

... (all 5 columns from employee_summary)
```

#### Records 7-9: SELECT * expansion from dept_data
```
Record 7:
sub_target_table: combined_data
target_column: dept_id
source_schema: hr_schema
source_table: departments
source_column: dept_id
sql_process: select*

Record 8:
target_column: dept_name
source_schema: hr_schema
source_table: departments
source_column: dept_name
sql_process: select*

Record 9:
target_column: manager_name
source_schema: hr_schema
source_table: departments
source_column: manager_name
sql_process: select*
```

---

### For `custom_report` table (es.* with additional columns):

#### Record 1: CREATE statement
```
sub_target_table: custom_report
sql_process: create
```

#### Records 2-6: SELECT * expansion (es.*)
```
Record 2:
sub_target_table: custom_report
target_column: emp_id
source_schema: hr_schema
source_table: employees
source_column: emp_id
sql_process: select*

... (all 5 columns from employee_summary)
```

#### Record 7: Explicit column (status)
```
sub_target_table: custom_report
target_column: status
source_schema: NULL
source_table: NULL
source_column: NULL
logic: 'Active' AS status
sql_process: select
```

#### Record 8: Explicit column (report_date)
```
sub_target_table: custom_report
target_column: report_date
source_schema: NULL
source_table: NULL
source_column: NULL
logic: CURRENT_DATE AS report_date
sql_process: select
```

---

## Key Points Demonstrated

### 1. ✅ target_column = source_column
For SELECT *, both the target and source column have the same name:
- `target_column: emp_id`
- `source_column: emp_id`

### 2. ✅ sql_process = "select*"
All expanded columns have the special process type:
- Regular SELECT: `sql_process: select`
- Wildcard SELECT: `sql_process: select*`

### 3. ✅ Tracing Through CTEs
Even though we're selecting `ed.*` (CTE alias), the source resolves to:
- `source_table: employees` (actual table)
- `source_schema: hr_schema` (actual schema)

### 4. ✅ Tables Created in Same Script
The second table `combined_data` uses `SELECT * FROM employee_summary`:
- `employee_summary` was created earlier in the same script
- Columns are tracked from its CREATE statement
- All 5 columns expand correctly

### 5. ✅ Mixed Wildcard and Explicit Columns
The third table shows:
- `es.*` expands to 5 columns with `sql_process: select*`
- `'Active' AS status` is a single record with `sql_process: select`
- `CURRENT_DATE AS report_date` is a single record with `sql_process: select`

### 6. ✅ Multiple Tables with SELECT *
When `SELECT * FROM t1 JOIN t2`:
- Expands all columns from t1
- Expands all columns from t2
- Each gets its own record
- All marked with `sql_process: select*`

---

## How Column Resolution Works

### Step 1: Detect SELECT *
Parser identifies Star expression in SELECT clause

### Step 2: Determine Scope
- If `table.*` → Get columns for that specific table
- If `*` → Get columns from all tables in FROM/JOIN

### Step 3: Resolve Table Type
- **Is it a CTE?** → Extract columns from CTE SELECT clause
- **Was it created in this script?** → Use tracked columns from CREATE
- **Is it an existing table?** → Query information_schema (future)

### Step 4: Trace to Source
For each column in the expanded list:
- Resolve through CTE chain to actual source table
- Get schema and table name
- Get original logic

### Step 5: Create Records
Generate one record per column:
- `target_column = column_name`
- `source_column = column_name` (same)
- `sql_process = "select*"`

---

## Database Table Support (Future)

Currently, SELECT * works for:
- ✅ CTEs defined in the same query
- ✅ Tables created earlier in the same script
- ⏳ Existing database tables

To enable existing database table support:

### Implementation Guide
```python
# In __init__ method:
def __init__(self, db_connection=None):
    self.cte_definitions = {}
    self.table_aliases = {}
    self.created_tables_columns = {}
    self.db_connection = db_connection  # Add this

# In _get_columns_from_information_schema:
def _get_columns_from_information_schema(self, schema, table_name, placeholders):
    if not self.db_connection:
        return []
    
    try:
        with self.db_connection.cursor() as cur:
            cur.execute('''
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = %s
                AND table_name = %s
                ORDER BY ordinal_position
            ''', (schema, table_name))
            
            return [
                {
                    "column_name": row[0],
                    "source_schema": schema,
                    "source_table": table_name,
                    "logic": row[0]
                }
                for row in cur.fetchall()
            ]
    except Exception as e:
        logging.warning(f"Failed to query columns: {e}")
        return []
```

### Usage
```python
# Create parser with DB connection
import psycopg2
db_conn = psycopg2.connect(host=..., database=..., user=..., password=...)
parser = ColumnLineageParser()
parser.db_connection = db_conn

# Now SELECT * from existing tables will work
sql = "SELECT * FROM production.existing_table"
lineage = parser.extract_column_lineage(sql)
# Will query information_schema and expand all columns
```

---

## Testing Checklist

To verify SELECT * is working:

1. ✅ Check `sql_process = "select*"` for expanded columns
2. ✅ Verify `target_column = source_column` for each record
3. ✅ Confirm each column gets its own record
4. ✅ Validate source table is actual table, not CTE name
5. ✅ Test `SELECT *` (all tables)
6. ✅ Test `SELECT table.*` (specific table)
7. ✅ Test with CTEs
8. ✅ Test with tables created in same script
9. ✅ Test mixed wildcard and explicit columns
10. ✅ Test multiple tables with SELECT *

---

## Summary

The SELECT * expansion feature is now fully functional:

✅ Automatically expands `SELECT *` and `table.*`
✅ Creates individual records for each column
✅ Uses special `sql_process = "select*"` identifier
✅ Resolves through CTEs to actual source tables
✅ Tracks tables created in same script
✅ Handles mixed wildcard and explicit columns
✅ Ready for information_schema integration

This ensures complete column-level lineage tracking regardless of SELECT syntax!
