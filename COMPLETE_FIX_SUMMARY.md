# IKG Column Lineage - Complete Fix Summary

## Issues Fixed

### Issue 1: Process Derivation ✓ FIXED
**Problem:** Process was being derived from the SQL_PATH prefix instead of the last subfolder.

**Example:**
- Filepath: `ikg-dags/new/xyz.sql`
- Old behavior: process = first folder after SQL_PATH
- New behavior: process = `new` (last subfolder before the SQL file)

**Fix:**
```python
def derive_process(file_path: str) -> str:
    """Derive process from the last subfolder before the SQL file."""
    parts = Path(file_path).parts
    if len(parts) >= 2:
        return parts[-2]  # Last folder before the filename
    return ""
```

---

### Issue 2: sub_target_table Population ✓ FIXED
**Problem:** sub_target_table was blank/null for join, where, and having operations.

**Fix:** All processing methods now receive and populate `target_table` and `target_schema`:
- `_process_joins(select_node, placeholders, target_table, target_schema)`
- `_process_where(select_node, placeholders, target_table, target_schema)`
- `_process_having(select_node, placeholders, target_table, target_schema)`

**Result:** Every record now has sub_target_table populated with the table it belongs to.

---

### Issue 3: target_column Population ✓ FIXED
**Problem:** target_column was NULL/blank for join, where, and having operations.

**Solution:** 
1. **For JOIN conditions:** Extract the actual column name from the join predicate
   ```sql
   ON hh.household_plus = acc.household_plus
   ```
   - target_column = `household_plus` (for both hh and acc references)

2. **For WHERE conditions:** Extract column names from the where clause
   ```sql
   WHERE hh.status = 'active'
   ```
   - target_column = `status`

3. **For HAVING conditions:** Extract column names from having clause
   ```sql
   HAVING COUNT(hh.id) > 10
   ```
   - target_column = `id`

**Implementation:**
```python
# In _process_joins, _process_where, _process_having:
for col_info in columns:
    target_col_name = col_info.get("column")  # Extract column name
    # ...
    records.append({
        "target_column": target_col_name,  # Now populated
        # ...
    })
```

---

### Issue 4: Logic Tracing Through CTEs ✓ FIXED
**Problem:** Logic was not traced back through CTE definitions to show the original source logic.

**Example:**
```sql
WITH hh_Filtered AS (
    SELECT b.acc_mhh_n AS household_plus 
    FROM base_feature_shhp_ikg b
)
SELECT hh.household_plus 
FROM hh_Filtered hh
WHERE hh.household_plus IS NOT NULL
```

**Old behavior:**
- logic = `hh.household_plus IS NOT NULL`

**New behavior:**
- logic = `b.acc_mhh_n AS household_plus` (traced back to original)

**Fix:** New method `_trace_logic_through_ctes()`:
```python
def _trace_logic_through_ctes(
    self,
    table_ref: Optional[str],
    column_name: str,
    original_logic: str,
    select_node: exp.Select,
    placeholders: Dict[str, str]
) -> str:
    """Trace the logic for a column back through CTEs."""
    # 1. Check if table_ref is a CTE alias
    # 2. Find the CTE definition
    # 3. Find the column in the CTE's SELECT
    # 4. Get the original logic from the CTE
    # 5. Recursively trace through nested CTEs if needed
    # 6. Return the ultimate source logic
```

---

### Issue 5: Processing Joins/Where/Having Inside CTEs ✓ FIXED
**Problem:** Joins, where, and having clauses inside CTE definitions were not being processed.

**Example:**
```sql
WITH hh_Filtered AS (
    SELECT b.acc_mhh_n AS household_plus
    FROM base_feature_shhp_ikg b
    LEFT JOIN employee_household_ikg e
    ON b.acc_mhh_n = e.acc_mhh_n  -- This join was not captured
    WHERE b.status = 'active'      -- This where was not captured
)
SELECT * FROM hh_Filtered
```

**Fix:** New method `_process_cte_internals()`:
```python
def _process_cte_internals(
    self,
    with_node: exp.With,
    target_table: Optional[str],
    target_schema: Optional[str],
    placeholders: Dict[str, str]
) -> List[Dict]:
    """Process joins/where/having from within CTE definitions."""
    # For each CTE in the WITH clause:
    # 1. Process its JOIN conditions
    # 2. Process its WHERE conditions
    # 3. Process its HAVING conditions
    # 4. Handle nested CTEs recursively
```

This method is called in `_process_select_for_target()` right after extracting CTE definitions.

---

## Complete Example

### Input SQL:
```sql
DROP TABLE IF EXISTS fee_waiver_hh_2m_4m;
CREATE TEMP TABLE fee_waiver_hh_2m_4m AS 
WITH hh_Filtered AS (
    SELECT
        b.acc_mhh_n AS household_plus, 
        b.mhh_assets_curr AS shh_assets_curr
    FROM schema1.base_feature_shhp_ikg b
    LEFT JOIN schema1.employee_household_ikg e
    ON b.acc_mhh_n = e.acc_mhh_n
    WHERE b.status = 'active'
),
acc_mapping AS (
    SELECT DISTINCT acc_n, acc_mhh_n AS household_plus
    FROM schema1.master_ids_curr_ikg
)
SELECT
    acc.acc_n,
    hh.household_plus,
    hh.shh_assets_curr
FROM hh_Filtered hh
LEFT JOIN acc_mapping acc
ON hh.household_plus = acc.household_plus
WHERE hh.shh_assets_curr > 1000;
```

### Output Records (Sample):

#### Record 1: CREATE statement
```
sub_target_table: fee_waiver_hh_2m_4m
target_column: NULL
sql_process: create
logic: [full CREATE statement]
```

#### Record 2: SELECT column
```
sub_target_table: fee_waiver_hh_2m_4m
target_column: household_plus
source_table: base_feature_shhp_ikg
source_column: acc_mhh_n
logic: b.acc_mhh_n AS household_plus
sql_process: select
```

#### Record 3: JOIN from CTE (hh_Filtered)
```
sub_target_table: fee_waiver_hh_2m_4m
target_column: acc_mhh_n
source_table: base_feature_shhp_ikg
source_column: acc_mhh_n
logic: b.acc_mhh_n
sql_process: join-with
```

#### Record 4: JOIN from CTE (hh_Filtered)
```
sub_target_table: fee_waiver_hh_2m_4m
target_column: acc_mhh_n
source_table: employee_household_ikg
source_column: acc_mhh_n
logic: e.acc_mhh_n
sql_process: join-with
```

#### Record 5: WHERE from CTE (hh_Filtered)
```
sub_target_table: fee_waiver_hh_2m_4m
target_column: status
source_table: base_feature_shhp_ikg
source_column: status
logic: b.status = 'active'
sql_process: where-with
```

#### Record 6: JOIN from main query
```
sub_target_table: fee_waiver_hh_2m_4m
target_column: household_plus
source_table: base_feature_shhp_ikg
source_column: acc_mhh_n
logic: b.acc_mhh_n AS household_plus
sql_process: join
```

#### Record 7: WHERE from main query
```
sub_target_table: fee_waiver_hh_2m_4m
target_column: shh_assets_curr
source_table: base_feature_shhp_ikg
source_column: mhh_assets_curr
logic: b.mhh_assets_curr AS shh_assets_curr
sql_process: where
```

---

## Key Architecture Changes

### 1. Method Signature Updates
All clause processing methods now include target information:
```python
# OLD
_process_joins(select_node, placeholders)
_process_where(select_node, placeholders)
_process_having(select_node, placeholders)

# NEW
_process_joins(select_node, placeholders, target_table, target_schema)
_process_where(select_node, placeholders, target_table, target_schema)
_process_having(select_node, placeholders, target_table, target_schema)
```

### 2. New Helper Methods
- `_trace_logic_through_ctes()`: Recursively traces logic through CTE chain
- `_process_cte_internals()`: Processes joins/where/having from within CTEs

### 3. Enhanced CTE Resolution
The CTE resolution now:
1. Maps aliases to CTE names
2. Marks CTE references with `is_cte: True`
3. Recursively resolves through CTE definitions
4. Traces logic back to original source
5. Processes internal CTE structures

---

## Usage

### Python Script
```bash
python ikg_column_lineage_master_auto_refresh.py
```

### Jupyter Notebook
Open and run: `ikg_column_lineage_master_auto_refresh.ipynb`

Both will generate:
1. Excel file with timestamp
2. Database table: `sandbox_prj_smart_insights.ikg_column_lineage_master_auto_refresh`

---

## Validation

To verify the fixes are working:

1. **Check process field**: Should be the last folder name before .sql file
2. **Check sub_target_table**: Should never be NULL (except for create statement whole table record)
3. **Check target_column**: Should be populated for all records except sql_process='create'
4. **Check logic**: Should show original source logic, not CTE alias references
5. **Check CTE processing**: JOIN/WHERE/HAVING inside CTEs should appear in output

---

## Notes

- The parser handles nested CTEs (CTEs within CTEs)
- Multiple levels of CTE resolution are supported
- Template parameters like `{{params.IKG_SCHEMA}}` are preserved
- All vendor-specific SQL (DISTRIBUTED BY, etc.) is cleaned before parsing
- Error handling prevents single file failures from stopping the entire process
