# SELECT * Expansion Feature

## Overview
The column lineage parser now automatically expands `SELECT *` and `SELECT table.*` patterns into individual column records, providing complete column-level lineage even when wildcard selectors are used.

---

## Feature Details

### What Gets Expanded

#### 1. SELECT * (All Tables)
```sql
SELECT *
FROM table1 t1
JOIN table2 t2 ON t1.id = t2.id
```
**Expands to:** All columns from both `table1` and `table2`

#### 2. SELECT table.* (Specific Table)
```sql
SELECT t1.*, t2.specific_column
FROM table1 t1
JOIN table2 t2 ON t1.id = t2.id
```
**Expands to:** All columns from `table1` plus `specific_column` from `table2`

#### 3. SELECT alias.*
```sql
WITH cte AS (
    SELECT col1, col2, col3 FROM source_table
)
SELECT c.*
FROM cte c
```
**Expands to:** `col1`, `col2`, `col3` from `source_table`

---

## How It Works

### Step 1: Detection
When processing SELECT projections, the parser detects `Star` expressions:
- `*` → All tables
- `table.*` or `alias.*` → Specific table/alias

### Step 2: Column Resolution
The parser resolves columns in this priority order:

#### A. From CTEs (Common Table Expressions)
```sql
WITH my_cte AS (
    SELECT col1, col2, col3 FROM source_table
)
SELECT c.*  -- Expands to col1, col2, col3
FROM my_cte c
```
**Resolution:** Extracts column list from the CTE's SELECT clause

#### B. From Tables Created in Same Script
```sql
-- Earlier in the same script:
CREATE TABLE temp_table AS
SELECT col1, col2, col3 FROM source_table;

-- Later in the same script:
SELECT t.*  -- Expands to col1, col2, col3
FROM temp_table t
```
**Resolution:** Uses columns tracked from the CREATE statement in the same script

#### C. From information_schema (Future Enhancement)
For existing database tables, columns would be queried from `information_schema.columns`:
```sql
SELECT * FROM existing_production_table
```
**Resolution:** Would query database metadata (requires DB connection - placeholder for now)

### Step 3: Record Creation
For each expanded column, a record is created with:
- `sql_process = "select*"` (note: not "select")
- `target_column = column_name`
- `source_column = column_name` (same as target)
- `source_table = actual_source_table`
- `source_schema = actual_source_schema`
- `logic = column_definition` (from CTE or CREATE)

---

## Examples

### Example 1: CTE with SELECT *

**SQL:**
```sql
CREATE TABLE output_table AS
WITH employee_data AS (
    SELECT 
        emp_id,
        emp_name,
        dept_id,
        salary
    FROM hr.employees
)
SELECT ed.*
FROM employee_data ed
WHERE ed.salary > 50000;
```

**Output Records (SELECT * expansion):**

| target_column | source_table | source_column | logic | sql_process |
|---------------|--------------|---------------|-------|-------------|
| emp_id | employees | emp_id | emp_id | select* |
| emp_name | employees | emp_name | emp_name | select* |
| dept_id | employees | dept_id | dept_id | select* |
| salary | employees | salary | salary | select* |

✅ All 4 columns expanded from the wildcard

---

### Example 2: Multiple Tables with SELECT *

**SQL:**
```sql
WITH orders AS (
    SELECT order_id, customer_id, order_date FROM sales.orders
),
customers AS (
    SELECT customer_id, customer_name FROM sales.customers
)
SELECT *
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id;
```

**Output Records (SELECT * expansion):**

| target_column | source_table | source_column | logic | sql_process |
|---------------|--------------|---------------|-------|-------------|
| order_id | orders | order_id | order_id | select* |
| customer_id | orders | customer_id | customer_id | select* |
| order_date | orders | order_date | order_date | select* |
| customer_id | customers | customer_id | customer_id | select* |
| customer_name | customers | customer_name | customer_name | select* |

✅ Columns from both tables expanded

---

### Example 3: Specific Table Wildcard

**SQL:**
```sql
WITH product_info AS (
    SELECT 
        product_id,
        product_name,
        category,
        price
    FROM inventory.products
)
SELECT 
    p.*,
    100 AS discount_amount
FROM product_info p
WHERE p.price > 100;
```

**Output Records:**

| target_column | source_table | source_column | logic | sql_process |
|---------------|--------------|---------------|-------|-------------|
| product_id | products | product_id | product_id | select* |
| product_name | products | product_name | product_name | select* |
| category | products | category | category | select* |
| price | products | price | price | select* |
| discount_amount | NULL | NULL | 100 | select |

✅ `p.*` expanded to 4 columns, constant gets regular `select` process

---

### Example 4: Nested CTE with SELECT *

**SQL:**
```sql
CREATE TABLE summary AS
WITH base_data AS (
    SELECT 
        region,
        sales_amount,
        cost_amount
    FROM finance.transactions
),
calculated AS (
    SELECT 
        region,
        sales_amount,
        cost_amount,
        sales_amount - cost_amount AS profit
    FROM base_data
)
SELECT c.*
FROM calculated c
WHERE c.profit > 10000;
```

**Output Records (SELECT * expansion):**

| target_column | source_table | source_column | logic | sql_process |
|---------------|--------------|---------------|-------|-------------|
| region | transactions | region | region | select* |
| sales_amount | transactions | sales_amount | sales_amount | select* |
| cost_amount | transactions | cost_amount | cost_amount | select* |
| profit | NULL | NULL | sales_amount - cost_amount AS profit | select* |

✅ Traces through nested CTEs to original source

---

## Key Methods

### 1. `_extract_column_info()`
**Enhanced to detect Star expressions:**
```python
if isinstance(projection, exp.Star):
    return [{
        "is_star": True,
        "table_qualifier": projection.table,  # None for *, or table name for table.*
        ...
    }]
```

### 2. `_expand_star_projection()`
**Main expansion logic:**
```python
def _expand_star_projection(star_data, select_node, target_table, ...):
    if table_qualifier:
        # Get columns for specific table
        columns = _get_columns_for_table(table_qualifier, ...)
    else:
        # Get columns for all tables
        columns = _get_columns_for_all_tables(...)
    
    # Create record for each column with sql_process="select*"
    return records
```

### 3. `_get_columns_for_table()`
**Resolves columns based on table type:**
```python
def _get_columns_for_table(table_ref, ...):
    if is_cte:
        return _extract_columns_from_select(cte_definition)
    elif table_ref in created_tables_columns:
        return created_tables_columns[table_ref]
    else:
        return _get_columns_from_information_schema(...)  # Future
```

### 4. `_extract_columns_from_select()`
**Extracts column list from SELECT statement:**
```python
def _extract_columns_from_select(select_node, ...):
    for projection in select_node.expressions:
        # Extract column name and source
        # Handle aliases, direct columns, expressions
        columns.append({
            "column_name": ...,
            "source_schema": ...,
            "source_table": ...,
            "logic": ...
        })
    return columns
```

### 5. `created_tables_columns` Dictionary
**Tracks columns of tables created in current script:**
```python
# When processing CREATE TABLE:
self.created_tables_columns[table_name] = extracted_columns

# When expanding SELECT *:
if table_name in self.created_tables_columns:
    use_these_columns = self.created_tables_columns[table_name]
```

---

## SQL Process Type: "select*"

Records created from SELECT * expansion have a special process type:

- **sql_process = "select*"** (with asterisk)
- Distinguished from regular **"select"** records
- Indicates the column was included via wildcard expansion

### Why This Matters

Users can differentiate between:
1. Explicitly selected columns: `SELECT col1, col2`
   - sql_process = "select"
2. Wildcard expanded columns: `SELECT *`
   - sql_process = "select*"

This is useful for:
- Understanding which columns were explicitly chosen vs. implicitly included
- Identifying scripts that might break if table structure changes
- Finding opportunities to make SELECT statements more explicit

---

## Benefits

### 1. Complete Lineage Coverage
Even queries using `SELECT *` now have full column-level lineage tracking.

### 2. Change Impact Analysis
Identify all downstream queries affected when a table's structure changes.

### 3. Explicit Column Mapping
See exactly which columns flow through wildcard selects, even through multiple CTE layers.

### 4. Documentation
Auto-generate documentation showing all columns that pass through each transformation.

---

## Future Enhancements

### information_schema Integration
Currently, the parser tracks columns from:
- ✅ CTEs in the same query
- ✅ Tables created in the same script
- ⏳ Existing database tables (requires DB connection)

**Future Implementation:**
```python
def _get_columns_from_information_schema(schema, table_name, ...):
    """Query database for column list."""
    query = f"""
        SELECT column_name 
        FROM information_schema.columns
        WHERE table_schema = '{schema}'
        AND table_name = '{table_name}'
        ORDER BY ordinal_position
    """
    # Execute query and return columns
```

This would enable complete expansion even for existing production tables.

---

## Limitations

### Current Limitations

1. **No DB Connection**
   - Cannot expand `SELECT *` for existing database tables
   - Only works for CTEs and tables created in same script

2. **Column Order**
   - Columns are returned in the order they appear in the CTE/CREATE
   - May not match actual database column order

3. **Dynamic Columns**
   - Cannot handle columns added via dynamic SQL
   - Cannot handle columns from UNNEST or table functions

### Workarounds

For existing tables without DB connection:
- Manually specify columns instead of using `*`
- Or, add DB connection to enable information_schema lookup

---

## Testing

### Test Case 1: Basic CTE with *
```sql
WITH t AS (SELECT a, b FROM src)
SELECT * FROM t
```
**Expected:** 2 records (a, b) with sql_process="select*"

### Test Case 2: Multiple Tables with *
```sql
SELECT * FROM t1 JOIN t2 ON t1.id = t2.id
```
**Expected:** All columns from both tables

### Test Case 3: Mixed Explicit and Wildcard
```sql
SELECT t1.*, t2.specific_col FROM t1 JOIN t2
```
**Expected:** All t1 columns (select*) + specific_col (select)

### Test Case 4: Nested CTEs with *
```sql
WITH t1 AS (...), t2 AS (SELECT * FROM t1)
SELECT * FROM t2
```
**Expected:** Traces back to original source through both CTEs

---

## Summary

The SELECT * expansion feature provides:
- ✅ Complete column-level lineage for wildcard selects
- ✅ Automatic expansion from CTEs and created tables
- ✅ Special "select*" process type for identification
- ✅ Full tracing through nested CTEs
- ⏳ Future support for information_schema lookup

This ensures comprehensive lineage tracking regardless of SELECT syntax!
