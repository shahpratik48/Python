# GitLab Issues to Greenplum - ONE ROW PER ISSUE

## 🎯 Key Feature: Consolidated Data Structure

**ONE ROW PER ISSUE_ID** - Multiple values are comma-separated

This solution ensures clean, normalized data with no duplicate issue rows.

## 📊 Data Structure

### How It Works

**Before (Multiple Rows Per Issue):**
```
issue_id | title        | link_type     | link_issue_iid
---------|--------------|---------------|---------------
123      | Fix bug      | blocks        | 124
123      | Fix bug      | relates_to    | 125
123      | Fix bug      | duplicates    | 126
```
❌ **3 rows for same issue** - difficult to analyze

**After (One Row Per Issue):**
```
issue_id | title        | link_type                      | link_issue_iid
---------|--------------|--------------------------------|---------------
123      | Fix bug      | blocks, relates_to, duplicates | 124, 125, 126
```
✅ **1 row per issue** - clean and easy to query

### Benefits

1. **No Duplicate Issues** - Each issue_id appears exactly once
2. **Easy Counting** - Row count = Issue count
3. **Simple Queries** - No need for GROUP BY or DISTINCT
4. **Comma-Separated Links** - All link data preserved in single row
5. **Database Efficiency** - Fewer rows, better performance

## 🔍 Link Columns with Comma Separation

All link-related columns store multiple values as comma-separated strings:

- `link_id` - "1234, 1235, 1236"
- `link_issue_id` - "5678, 5679, 5680"  
- `link_issue_iid` - "42, 43, 44"
- `link_type` - "blocks, relates_to, is_blocked_by"
- `link_url` - "https://..., https://..., https://..."
- `link_issue_title` - "Title 1, Title 2, Title 3"
- `linked_project_id` - "100, 101, 102"

**Example:**
```python
# Issue with 3 links
{
    'issue_id': 12345,
    'title': 'Implement feature X',
    'link_type': 'blocks, relates_to, duplicates',
    'link_issue_iid': '101, 102, 103',
    'link_issue_title': 'Bug fix A, Task B, Issue C'
}
```

## 📦 Features

### Data Extraction
- Fetches **ALL issues** from IKG and SWAT projects
- **All states** (open, closed, locked, etc.)
- **All iterations**
- **One row per issue_id**
- Links consolidated with comma-separated values

### Excel Output
- Single file: `gitlab_issues_ikg_swat.xlsx`
- **2 worksheets**: IKG and SWAT
- Professional formatting
- One row per issue in each sheet

### Greenplum Database

**Main Tables** (recreated each run):
- `sandbox_prj_smart_insights.ikg_issue_details`
- `sandbox_prj_smart_insights.swat_issue_details`

**Archive Tables** (cumulative history):
- `sandbox_prj_smart_insights.ikg_issue_details_archive`
- `sandbox_prj_smart_insights.swat_issue_details_archive`

## 📋 Prerequisites

### GitLab
- Personal Access Token with `api` and `read_api` scopes
- Access to both projects

### Greenplum
- Database: `gprdsp`
- Schema: `sandbox_prj_smart_insights`
- User: `ds_rdsp_dev`

## 🚀 Installation

```bash
pip install -r requirements.txt
```

## 💻 Usage

### Python Script
```bash
python gitlab_issues_to_greenplum_v2.py
```

### Jupyter Notebook
```bash
jupyter notebook gitlab_issues_to_greenplum_v2.ipynb
```

## 📊 Data Fields (40+ columns)

### Core Fields
- `project` - Project identifier
- `issue_id` - Global unique ID (primary)
- `issue_iid` - Project-specific ID
- `title`, `description`, `state`, `web_url`

### Link Columns (Comma-Separated)
- `link_id` - All link IDs
- `link_issue_id` - All linked issue global IDs
- `link_issue_iid` - All linked issue IIDs
- `link_type` - All link types (blocks, relates_to, etc.)
- `link_url` - All link URLs
- `link_issue_title` - All linked issue titles
- `linked_project_id` - All linked project IDs

### Other Fields (Also Comma-Separated When Multiple)
- `assignee` - All assignees
- `assignee_ids` - All assignee IDs
- `labels` - All labels
- `participants` - All participants

### Single-Value Fields
- Dates: `issue_created_date`, `created_at`, `updated_at`, `closed_at`, `due_date`, `start_date`, `current_date_time`
- Organization: `milestone`, `iteration`, `epic`, `epic_iid`, `weight`
- Metrics: `upvotes`, `downvotes`, `user_notes_count`, `merge_requests_count`
- Time: `time_estimate_hours`, `time_spent_hours`
- Metadata: `confidential`, `issue_type`, `severity`, `health_status`

## 🔍 Query Examples

### Simple Issue Count
```sql
-- Count total issues (one row = one issue)
SELECT COUNT(*) FROM sandbox_prj_smart_insights.ikg_issue_details;
```

### Issues with Links
```sql
-- Find issues that have any links
SELECT 
    issue_iid,
    title,
    link_type,
    link_issue_iid
FROM sandbox_prj_smart_insights.ikg_issue_details
WHERE link_id IS NOT NULL;
```

### Parse Comma-Separated Links
```sql
-- Split comma-separated link types (PostgreSQL)
SELECT 
    issue_iid,
    title,
    unnest(string_to_array(link_type, ', ')) as individual_link_type
FROM sandbox_prj_smart_insights.ikg_issue_details
WHERE link_type IS NOT NULL;
```

### Count Links Per Issue
```sql
-- Count how many links each issue has
SELECT 
    issue_iid,
    title,
    array_length(string_to_array(link_id, ', '), 1) as num_links
FROM sandbox_prj_smart_insights.ikg_issue_details
WHERE link_id IS NOT NULL
ORDER BY num_links DESC;
```

### Issues by State
```sql
-- Group issues by state (simple because one row per issue)
SELECT 
    state,
    COUNT(*) as issue_count
FROM sandbox_prj_smart_insights.ikg_issue_details
GROUP BY state;
```

### Issues Blocking Other Issues
```sql
-- Find issues that block others
SELECT 
    issue_iid,
    title,
    link_type,
    link_issue_iid as blocked_issues
FROM sandbox_prj_smart_insights.ikg_issue_details
WHERE link_type LIKE '%blocks%';
```

### Cross-Project Links
```sql
-- Find issues linked to other projects
SELECT 
    project,
    issue_iid,
    title,
    linked_project_id as other_projects
FROM sandbox_prj_smart_insights.ikg_issue_details
WHERE linked_project_id IS NOT NULL;
```

## 🔧 Working with Comma-Separated Data

### In SQL (PostgreSQL/Greenplum)

**Split into rows:**
```sql
SELECT 
    issue_iid,
    unnest(string_to_array(link_type, ', ')) as link_type
FROM sandbox_prj_smart_insights.ikg_issue_details
WHERE link_type IS NOT NULL;
```

**Count values:**
```sql
SELECT 
    issue_iid,
    array_length(string_to_array(link_id, ', '), 1) as link_count
FROM sandbox_prj_smart_insights.ikg_issue_details;
```

**Check if contains value:**
```sql
SELECT * 
FROM sandbox_prj_smart_insights.ikg_issue_details
WHERE link_type LIKE '%blocks%';
```

### In Python/Pandas

**Split into list:**
```python
import pandas as pd

df = pd.read_sql_query(
    "SELECT * FROM sandbox_prj_smart_insights.ikg_issue_details",
    conn
)

# Split link types into list
df['link_types_list'] = df['link_type'].str.split(', ')

# Explode to one row per link
df_exploded = df.explode('link_types_list')
```

**Count links:**
```python
# Count number of links per issue
df['num_links'] = df['link_id'].str.count(',') + 1
df['num_links'] = df['num_links'].fillna(0)
```

### In Excel

**Split into columns:**
```
=TEXTSPLIT(A2, ", ")
```

**Count values:**
```
=LEN(A2)-LEN(SUBSTITUTE(A2,",",""))+1
```

## 📈 Data Verification

### Verify One Row Per Issue
```sql
-- Should return 0 duplicates
SELECT issue_id, COUNT(*) 
FROM sandbox_prj_smart_insights.ikg_issue_details
GROUP BY issue_id
HAVING COUNT(*) > 1;
```

### Row Count = Issue Count
```sql
-- These should match
SELECT COUNT(*) as row_count FROM sandbox_prj_smart_insights.ikg_issue_details;
SELECT COUNT(DISTINCT issue_id) as unique_issues FROM sandbox_prj_smart_insights.ikg_issue_details;
```

## 🔄 Data Flow

1. **Fetch** from GitLab API
2. **Collect** all links per issue
3. **Join** link values with comma separator
4. **Create** single row per issue
5. **Save** to Excel (one row per issue in each sheet)
6. **Load** to Greenplum (one row per issue in each table)

## 🆚 Comparison: Old vs New Approach

| Aspect | Old (Multi-Row) | New (One-Row) |
|--------|-----------------|---------------|
| Rows per issue | 1 to N | Always 1 |
| Issue count | Need DISTINCT | Direct count |
| Link data | Separate rows | Comma-separated |
| Queries | Complex (GROUP BY) | Simple (WHERE) |
| Excel clarity | Confusing duplicates | Clean, one per issue |
| Database size | Larger | Smaller |
| Performance | Slower | Faster |

## 🐛 Troubleshooting

### Parsing Comma-Separated Values

**Problem:** Need individual link values
**Solution:** Use string split functions

```python
# Python
links = row['link_type'].split(', ')

# SQL
unnest(string_to_array(link_type, ', '))
```

### Counting Links

**Problem:** Count how many links
**Solution:** Count commas + 1

```python
# Python
num_links = value.count(',') + 1 if pd.notna(value) else 0

# SQL
array_length(string_to_array(link_id, ', '), 1)
```

### Searching Within Links

**Problem:** Find if specific link type exists
**Solution:** Use LIKE or contains

```sql
-- SQL
WHERE link_type LIKE '%blocks%'

-- Python
df[df['link_type'].str.contains('blocks', na=False)]
```

## 📝 Important Notes

1. **One Row = One Issue**: Guaranteed unique issue_id per row
2. **Comma Separator**: Multiple values joined with ", " (comma + space)
3. **NULL vs Empty**: No links = NULL, not empty string
4. **Consistent Order**: Link arrays maintain consistent position
   - If 3 links: link_id[0] corresponds to link_type[0], link_issue_iid[0], etc.
5. **Excel Benefits**: Easy sorting, filtering, pivot tables
6. **Database Benefits**: Simpler queries, better performance

## 🎯 Best Practices

### Querying
- Use `IS NOT NULL` to find issues with links
- Use `string_to_array()` to split values in SQL
- Use `.str.split()` to split values in pandas
- Use LIKE for partial matching within comma-separated values

### Analysis
- Count rows directly = count issues
- No need for GROUP BY or DISTINCT on issue_id
- Parse comma-separated when you need individual link details
- Use archive tables for historical trend analysis

### Data Quality
- Verify issue_id uniqueness after each run
- Check for orphaned links (links to non-existent issues)
- Monitor archive table growth
- Validate comma separation consistency

## 📞 Support

For issues:
1. Verify one row per issue_id
2. Check comma-separated format
3. Review SQL parsing functions
4. Consult documentation

## 📄 License

Internal use only.
