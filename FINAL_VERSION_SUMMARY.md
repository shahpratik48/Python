# GitLab Issues Fetcher - FINAL VERSION Summary

## ✅ All Requirements Implemented

### 1. Labels Column After Title ✓
**Location**: Column 5 (right after title)

**Column Order:**
1. project
2. issue_id
3. issue_iid
4. title
5. **labels** ← NEW POSITION!
6. description
7. state
8. ...

**Benefits:**
- Easier to scan and filter by labels
- More logical grouping with title
- Consistent with typical issue tracking UIs

### 2. Linked Project Name ✓
**Column**: `linked_project_name`

**How It Works:**
- Fetches project names for all linked issues
- Uses caching to avoid repeated API calls
- Handles multiple linked projects (comma-separated)

**Example Data:**
```
linked_project_id: "123, 456, 789"
linked_project_name: "Project A, Project B, Project C"
```

**Performance:**
- Project names cached after first fetch
- Parallel fetching when needed
- Fallback to "Project_ID" if fetch fails

### 3. Logging Functionality ✓
**Log File**: `gitlab_fetch_YYYYMMDD_HHMMSS.log`

**What's Logged:**
- Process start/end times
- Each major step (fetch, extract, insert)
- API calls and responses
- Database operations
- Errors and warnings
- Performance metrics

**Log Levels:**
- **INFO**: Normal operations, progress updates
- **WARNING**: Non-critical issues (e.g., missing project name)
- **ERROR**: Critical failures
- **DEBUG**: Detailed troubleshooting info

**Log Output:**
- Console (real-time feedback)
- File (permanent record)

**Example Log:**
```
2026-01-25 10:30:15 - INFO - Starting GitLab Issues Fetch Process
2026-01-25 10:30:20 - INFO - Fetching project: staat-ds-insights-home
2026-01-25 10:30:25 - INFO - Found: IKG Project (ID: 12345)
2026-01-25 10:30:30 - INFO - Page 1: fetched 100 issues (Total: 100)
2026-01-25 10:32:15 - INFO - Fetched 450 issues from IKG Project
2026-01-25 10:32:20 - INFO - Fetching links for 450 issues (20 threads)
2026-01-25 10:32:45 - INFO - All 450 links fetched
2026-01-25 10:33:00 - INFO - Inserted 450 rows into sandbox_prj_smart_insights.ikg_issue_details
2026-01-25 10:35:00 - INFO - Process completed successfully in 285.5 seconds
```

### 4. Archive Tables: CREATE IF NOT EXISTS ✓

**Behavior:**
- **First run**: Creates archive tables
- **Subsequent runs**: Uses existing archive tables
- **Data**: Appends new data each run (cumulative history)

**SQL Generated:**
```sql
CREATE TABLE IF NOT EXISTS sandbox_prj_smart_insights.ikg_issue_details_archive (
    project TEXT,
    issue_id BIGINT,
    ...
) DISTRIBUTED RANDOMLY
```

**Benefits:**
- No error if table exists
- Preserves historical data
- Automatic versioning by current_date_time

### 5. Main Tables: DROP and CREATE Every Time ✓

**Behavior:**
- **Every run**: Drops and recreates main tables
- **Data**: Fresh snapshot of current state
- **No history**: Only latest data

**SQL Generated:**
```sql
DROP TABLE IF EXISTS sandbox_prj_smart_insights.ikg_issue_details;

CREATE TABLE sandbox_prj_smart_insights.ikg_issue_details (
    project TEXT,
    issue_id BIGINT,
    ...
) DISTRIBUTED RANDOMLY
```

**Benefits:**
- Always clean data
- No duplicate issues
- Schema changes automatically applied

## 📊 Complete Column List (in order)

1. project
2. issue_id
3. issue_iid
4. title
5. **labels** ← After title
6. description
7. state
8. web_url
9. link_id
10. link_issue_id
11. link_issue_iid
12. link_type
13. link_url
14. link_issue_title
15. linked_project_id
16. **linked_project_name** ← NEW!
17. author
18. author_username
19. created_by_id
20. assignee
21. assignee_ids
22. issue_created_date
23. created_at
24. updated_at
25. closed_at
26. due_date
27. start_date
28. current_date_time
29. milestone
30. iteration
31. epic
32. epic_iid
33. weight
34. parent_iid
35. has_tasks
36. task_completion_status
37. participants
38. upvotes
39. downvotes
40. user_notes_count
41. merge_requests_count
42. time_estimate_hours
43. time_spent_hours
44. confidential
45. discussion_locked
46. issue_type
47. severity
48. health_status

**Total: 48 columns**

## 🚀 Performance Features (Retained)

All optimization features from previous version:
- ✅ Parallel link fetching (20 threads)
- ✅ COPY command inserts (10,000+ rows/sec)
- ✅ Efficient Excel creation
- ✅ Connection pooling
- ✅ Progress tracking

## 📝 Log File Analysis

### Log File Contains:

**Startup:**
- Configuration details
- Authentication status
- Project information

**Progress:**
- Pages fetched (real-time)
- Links fetched (progress updates every 100)
- Data extraction progress
- Database operations

**Performance:**
- Timing for each major step
- Row counts
- Insert speeds

**Errors:**
- API failures
- Database errors
- Network timeouts

### Useful Log Queries:

**Check total time:**
```bash
grep "Process completed" gitlab_fetch_*.log
```

**Find errors:**
```bash
grep "ERROR" gitlab_fetch_*.log
```

**Count issues fetched:**
```bash
grep "Fetched.*issues from" gitlab_fetch_*.log
```

## 🗄️ Database Table Strategy

### Main Tables (Fresh Snapshot)
```
sandbox_prj_smart_insights.ikg_issue_details
sandbox_prj_smart_insights.swat_issue_details
```
- **Purpose**: Current state
- **Strategy**: Drop & Create
- **Use Case**: Daily reporting, dashboards
- **Data**: Latest snapshot only

### Archive Tables (Historical Record)
```
sandbox_prj_smart_insights.ikg_issue_details_archive
sandbox_prj_smart_insights.swat_issue_details_archive
```
- **Purpose**: Historical tracking
- **Strategy**: Create IF NOT EXISTS, then INSERT
- **Use Case**: Trend analysis, auditing
- **Data**: Cumulative (all runs)

### Querying Archive for History

**See issue evolution:**
```sql
SELECT 
    issue_iid,
    title,
    state,
    current_date_time as snapshot_date
FROM sandbox_prj_smart_insights.ikg_issue_details_archive
WHERE issue_iid = 123
ORDER BY current_date_time DESC;
```

**Track label changes:**
```sql
SELECT 
    issue_iid,
    labels,
    current_date_time
FROM sandbox_prj_smart_insights.ikg_issue_details_archive
WHERE issue_iid = 123
ORDER BY current_date_time;
```

## 📋 Usage Examples

### Run the Script

```bash
python gitlab_issues_final.py
```

**Output:**
```
Log file: gitlab_fetch_20260125_103015.log
✓ GitLab authentication configured
✓ Found project: IKG Project (ID: 12345)

Fetching issues from IKG Project...
  Page 1: 100 issues (Total: 100)
  Page 2: 200 issues (Total: 200)
  ...
✓ Fetched 450 issues
Fetching links in parallel (using 20 threads)...
✓ Links fetched
✓ IKG: 450 issues extracted

...

✓ Excel file saved: gitlab_issues_ikg_swat.xlsx
✓ Connected to Greenplum database: gprdsp

Creating main tables (drop and create)...
  Dropped sandbox_prj_smart_insights.ikg_issue_details (if existed)
  ✓ ikg_issue_details
  ...

Creating archive tables (if not exists)...
  ✓ ikg_issue_details_archive
  ...

Inserting data into main tables (COPY command)...
  ✓ ikg_issue_details: 450 rows
  ...

⏱️  Total execution time: 285.50 seconds (4.76 minutes)
📝 Log file: gitlab_fetch_20260125_103015.log

✓ Process completed successfully!
```

### Check Logs

```bash
# View latest log
tail -f gitlab_fetch_*.log

# Search for errors
grep ERROR gitlab_fetch_*.log

# Check performance
grep "execution time" gitlab_fetch_*.log
```

### Verify Data

```sql
-- Check main table
SELECT COUNT(*) FROM sandbox_prj_smart_insights.ikg_issue_details;

-- Check archive growth
SELECT 
    DATE(current_date_time) as run_date,
    COUNT(*) as issues_count
FROM sandbox_prj_smart_insights.ikg_issue_details_archive
GROUP BY DATE(current_date_time)
ORDER BY run_date DESC;

-- Verify new columns
SELECT 
    issue_iid,
    title,
    labels,
    linked_project_name
FROM sandbox_prj_smart_insights.ikg_issue_details
WHERE labels IS NOT NULL
LIMIT 5;
```

## 🎯 Quick Reference

| Feature | Status | Implementation |
|---------|--------|----------------|
| Labels after title | ✅ | Column 5 |
| Linked project name | ✅ | New column with API fetch |
| Logging | ✅ | File + console |
| Archive: IF NOT EXISTS | ✅ | CREATE IF NOT EXISTS |
| Main: DROP & CREATE | ✅ | DROP TABLE IF EXISTS |
| Parallel processing | ✅ | 20 threads |
| COPY inserts | ✅ | 10k+ rows/sec |

## 📦 Files Delivered

1. **gitlab_issues_final.py** - Python script with all features
2. **gitlab_issues_final.ipynb** - Jupyter notebook version
3. **requirements.txt** - Dependencies (unchanged)

## ✨ Summary

All 5 requirements implemented:
1. ✅ Labels column repositioned after title
2. ✅ Linked project name column added
3. ✅ Comprehensive logging to file and console
4. ✅ Archive tables use CREATE IF NOT EXISTS
5. ✅ Main tables DROP and CREATE every time

Plus all performance optimizations retained!
