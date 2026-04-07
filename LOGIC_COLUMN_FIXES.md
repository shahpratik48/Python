# Logic Column Fixes - Complete Summary

## Overview
Fixed the `logic` column to properly preserve original SQL and trace through CTEs to show actual source logic.

---

## Fix 1: CREATE Statement Logic ✓

### Problem
- Logic was using sqlglot-generated SQL which changed:
  - `TEMP` → `TEMPORARY`
  - Schema names → Template tokens
  - Removed `DISTRIBUTED BY` clauses
  - Changed formatting

### Solution
Extract the exact CREATE statement from the original SQL text before any processing.

### Implementation
```python
def _extract_create_statement_sql(self, table_name: str, original_sql: str) -> str:
    """Extract the original CREATE statement SQL for a specific table."""
    # Find the exact CREATE statement in original SQL using regex
    # Pattern matches: DROP TABLE IF EXISTS ... ; CREATE TEMP TABLE name ... ;
    # Returns the exact text as written in the script
```

### Example
**Original SQL:**
```sql
DROP TABLE IF EXISTS fee_waiver_hh_2m_4m;
CREATE TEMP TABLE fee_waiver_hh_2m_4m AS 
WITH hh_Filtered AS (...)
SELECT ... 
DISTRIBUTED BY (acc_n);
```

**Logic stored:**
```
DROP TABLE IF EXISTS fee_waiver_hh_2m_4m;
CREATE TEMP TABLE fee_waiver_hh_2m_4m AS 
WITH hh_Filtered AS (...)
SELECT ... 
DISTRIBUTED BY (acc_n);
```

✅ **Preserves:** TEMP keyword, schema names, DISTRIBUTED BY clause, exact formatting

---

## Fix 2: JOIN Logic with Source Details ✓

### Problem
- Logic only showed join condition like `hh.household_plus = acc.household_plus`
- Didn't show the actual source logic from CTEs

### Solution
Format: `ON a.x = b.x (a.x = actual_source_from_cte)`

### Implementation
```python
# For each column in JOIN condition:
# 1. Get the join condition: "ON hh.household_plus = acc.household_plus"
# 2. Get source logic from CTE for the column
# 3. Format: "ON {condition} ({alias.col} = {cte_source_logic})"
```

### Example
**SQL:**
```sql
WITH hh_Filtered AS (
    SELECT b.acc_mhh_n AS household_plus
    FROM base_feature_shhp_ikg b
),
acc_mapping AS (
    SELECT acc_mhh_n AS household_plus
    FROM master_ids_curr_ikg
)
SELECT *
FROM hh_Filtered hh
LEFT JOIN acc_mapping acc
ON hh.household_plus = acc.household_plus
```

**Logic stored for hh.household_plus:**
```
ON hh.household_plus = acc.household_plus (hh.household_plus = b.acc_mhh_n AS household_plus)
```

**Logic stored for acc.household_plus:**
```
ON hh.household_plus = acc.household_plus (acc.household_plus = acc_mhh_n AS household_plus)
```

✅ **Shows:** 
- Actual join condition
- Source logic from CTE in brackets
- Clear mapping from alias to source

---

## Fix 3: SELECT Column Logic from CTE ✓

### Problem
- Logic showed the SELECT expression like `hh.household_plus`
- Didn't trace back to the actual source logic inside the CTE

### Solution
Look up the column definition inside the CTE and use that as the logic.

### Implementation
```python
def _get_source_logic_detail(table_ref, column_name, select_node, placeholders):
    """Get the source logic detail for a column from CTEs."""
    # 1. Check if table_ref is a CTE alias
    # 2. Find the CTE definition
    # 3. Find the column in CTE's SELECT clause
    # 4. Return the column's definition logic
```

### Example
**SQL:**
```sql
WITH hh_Filtered AS (
    SELECT 
        b.acc_mhh_n AS household_plus,
        b.mhh_assets_curr AS shh_assets_curr,
        CASE WHEN e.acc_mhh_n IS NULL THEN 'N' ELSE 'Y' END AS is_employee_hh
    FROM base_feature_shhp_ikg b
    LEFT JOIN employee_household_ikg e ON b.acc_mhh_n = e.acc_mhh_n
)
SELECT 
    hh.household_plus,
    hh.shh_assets_curr,
    hh.is_employee_hh
FROM hh_Filtered hh
```

**Logic stored:**

| target_column | logic |
|---------------|-------|
| household_plus | `b.acc_mhh_n AS household_plus` |
| shh_assets_curr | `b.mhh_assets_curr AS shh_assets_curr` |
| is_employee_hh | `CASE WHEN e.acc_mhh_n IS NULL THEN 'N' ELSE 'Y' END AS is_employee_hh` |

✅ **Shows:** Actual source expression from inside the CTE, not the reference

---

## Fix 4: WHERE/HAVING Logic from CTE ✓

### Problem
- Logic showed WHERE/HAVING condition with CTE alias like `WHERE hh.status = 'active'`
- Didn't trace to the actual source column definition

### Solution
Look up the column in the CTE and use its source logic.

### Example
**SQL:**
```sql
WITH hh_Filtered AS (
    SELECT 
        b.acc_mhh_n AS household_plus,
        b.status_code AS status
    FROM base_feature_shhp_ikg b
)
SELECT *
FROM hh_Filtered hh
WHERE hh.status = 'active'
```

**Logic stored for WHERE:**
```
b.status_code AS status
```

✅ **Shows:** The actual source column definition from the CTE

---

## Complete Example Output

### Input SQL:
```sql
DROP TABLE IF EXISTS fee_waiver_hh_2m_4m;
CREATE TEMP TABLE fee_waiver_hh_2m_4m AS 
WITH hh_Filtered AS (
    SELECT
        b.acc_mhh_n AS household_plus, 
        b.mhh_assets_curr AS shh_assets_curr
    FROM {{params.IKG_SCHEMA}}.base_feature_shhp_ikg b
    LEFT JOIN {{params.IKG_SCHEMA}}.employee_household_ikg e
    ON b.acc_mhh_n = e.acc_mhh_n
    WHERE b.status = 'active'
),
acc_mapping AS (
    SELECT DISTINCT acc_n, acc_mhh_n AS household_plus
    FROM {{params.IKG_SCHEMA}}.master_ids_curr_ikg
)
SELECT
    acc.acc_n,
    hh.household_plus,
    hh.shh_assets_curr
FROM hh_Filtered hh
LEFT JOIN acc_mapping acc
ON hh.household_plus = acc.household_plus
WHERE hh.shh_assets_curr > 1000
DISTRIBUTED BY (acc_n);
```

### Output Records:

#### Record 1: CREATE
```
sql_process: create
logic: DROP TABLE IF EXISTS fee_waiver_hh_2m_4m;
       CREATE TEMP TABLE fee_waiver_hh_2m_4m AS 
       WITH hh_Filtered AS (...)
       ...
       DISTRIBUTED BY (acc_n);
```
✅ Exact original SQL including TEMP and DISTRIBUTED BY

#### Record 2: SELECT - household_plus
```
sql_process: select
target_column: household_plus
source_table: base_feature_shhp_ikg
source_column: acc_mhh_n
logic: b.acc_mhh_n AS household_plus
```
✅ Source logic from CTE definition

#### Record 3: SELECT - shh_assets_curr
```
sql_process: select
target_column: shh_assets_curr
source_table: base_feature_shhp_ikg
source_column: mhh_assets_curr
logic: b.mhh_assets_curr AS shh_assets_curr
```
✅ Source logic from CTE definition

#### Record 4: SELECT - acc_n
```
sql_process: select
target_column: acc_n
source_table: master_ids_curr_ikg
source_column: acc_n
logic: acc_n
```
✅ Direct column reference (not aliased in CTE)

#### Record 5: JOIN-WITH (from hh_Filtered CTE)
```
sql_process: join-with
target_column: acc_mhh_n
source_table: base_feature_shhp_ikg
source_column: acc_mhh_n
logic: ON b.acc_mhh_n = e.acc_mhh_n (b.acc_mhh_n = b.acc_mhh_n AS household_plus)
```
✅ Join condition with source logic in brackets

#### Record 6: JOIN-WITH (from hh_Filtered CTE)
```
sql_process: join-with
target_column: acc_mhh_n
source_table: employee_household_ikg
source_column: acc_mhh_n
logic: ON b.acc_mhh_n = e.acc_mhh_n (e.acc_mhh_n = e.acc_mhh_n)
```
✅ Join condition with source logic in brackets

#### Record 7: WHERE-WITH (from hh_Filtered CTE)
```
sql_process: where-with
target_column: status
source_table: base_feature_shhp_ikg
source_column: status
logic: b.status
```
✅ Source logic from CTE

#### Record 8: JOIN (main query)
```
sql_process: join
target_column: household_plus
source_table: base_feature_shhp_ikg
source_column: acc_mhh_n
logic: ON hh.household_plus = acc.household_plus (hh.household_plus = b.acc_mhh_n AS household_plus)
```
✅ Join with traced CTE source logic

#### Record 9: WHERE (main query)
```
sql_process: where
target_column: shh_assets_curr
source_table: base_feature_shhp_ikg
source_column: mhh_assets_curr
logic: b.mhh_assets_curr AS shh_assets_curr
```
✅ Source logic from CTE

---

## Key Methods Added/Modified

### 1. `_extract_create_statement_sql()`
- Extracts exact CREATE statement from original SQL
- Uses regex to find CREATE TABLE block
- Preserves all original formatting and keywords

### 2. `_get_source_logic_detail()`
- Looks up column definition in CTE
- Returns the actual source expression
- Used by SELECT, JOIN, WHERE, HAVING processing

### 3. Modified `extract_column_lineage()`
- Stores original SQL before any processing
- Stores cleaned SQL (with DISTRIBUTED BY intact)
- Replaces CREATE logic with extracted original

### 4. Modified JOIN/WHERE/HAVING processing
- Gets source logic from CTE using `_get_source_logic_detail()`
- Formats JOIN with source details in brackets
- Uses CTE source logic for WHERE/HAVING

---

## Validation Checklist

✅ CREATE logic: Original SQL with TEMP, DISTRIBUTED BY, exact formatting
✅ SELECT logic: Source expression from CTE (e.g., `b.col AS alias`)
✅ JOIN logic: Condition with source in brackets (e.g., `ON a.x = b.x (a.x = source)`)
✅ WHERE logic: Source expression from CTE
✅ HAVING logic: Source expression from CTE
✅ Template parameters: Preserved in CREATE logic ({{params.IKG_SCHEMA}})
✅ Nested CTEs: Logic traced through multiple levels
✅ Non-CTE columns: Direct logic without CTE lookup

---

## Usage

The logic column now provides complete traceability:
1. **CREATE**: See the exact SQL as written in the script
2. **SELECT**: See the actual source expression (not just the reference)
3. **JOIN**: See both the join condition and source expressions
4. **WHERE/HAVING**: See the actual source expression being filtered

This enables complete understanding of data lineage from source to target!
