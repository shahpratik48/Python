#!/usr/bin/env python3
"""
GitLab Release Report Generator - Unified Version
Generates a single Excel file with:
  - Sheet 1: Issues, Branches, and Merge Requests Report
  - Sheet 2: ODM Release Details with file changes and SQL parsing
"""

import gitlab
import pandas as pd
import numpy as np
import getpass
import requests
import re
from datetime import datetime, timedelta
from pathlib import Path
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill
import logging
import psycopg2
from psycopg2 import sql
from sqlalchemy import create_engine
import psycopg2
from psycopg2 import sql

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
GITLAB_URL = 'https://devcloud.ubs.net'
GROUP_PATH = 'ubs/gwma/smart-technology-and-analytics/staat-data-science/staat-ds-insights-cl/commons'
PROJECT_PATH = 'ubs/gwma/smart-technology-and-analytics/staat-data-science/staat-ds-insights-cl/commons/staat-ds-insights-home'
IKG_PROJECT_PATH = 'ubs/gwma/smart-technology-and-analytics/staat-data-science/staat-ds-genesis/genesis-platform/ikg-dags'
NLG_PROJECT_PATH = 'ubs/gwma/smart-technology-and-analytics/staat-data-science/staat-ds-genesis/genesis-platform/nlg-dags'
ODM_PROJECT_PATH = 'ubs/gwma/smart-technology-and-analytics/staat-data-science/staat-ds-genesis/genesis-platform/odm-dags'
DEFAULT_BASE_BRANCH = 'odm-master'

# Greenplum Configuration
GREENPLUM_HOST = 'greenplum-rdsp.zur.swissbank.com'
GREENPLUM_PORT = 5432
GREENPLUM_DB = 'gprdsp'
GREENPLUM_USER = 'ds_rdsp_dev'
GREENPLUM_SCHEMA = 'sandbox_prj_smart_insights'
OUTPUT_TABLE1 = 'staat_insight_release'
OUTPUT_TABLE2 = 'odm_release_details'


def get_linked_issues(project_id, issue_iid, headers):
    """Get linked issues for a specific issue"""
    url = f"{GITLAB_URL}/api/v4/projects/{requests.utils.quote(str(project_id), safe='')}/issues/{issue_iid}/links"
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json()
        return []
    except Exception as e:
        logger.error(f"Error fetching linked issues for issue {issue_iid}: {e}")
        return []


def extract_digits(text):
    """Extract first digit sequence from text"""
    if pd.isna(text):
        return None
    text_str = str(text).strip()
    if text_str.upper() == '':
        return None
    match = re.search(r'\d+', text_str)
    return match.group() if match else None


def clean_labels(labels):
    """Remove status:: prefix (case insensitive) and @ symbol from labels"""
    if pd.isna(labels) or labels == '':
        return ''
    label_list = [l.strip() for l in str(labels).split(',')]
    cleaned = []
    for l in label_list:
        # Skip labels starting with 'status::' (case insensitive)
        if l.lower().startswith('status::'):
            continue
        # Remove @ symbol from the beginning
        if l.startswith('@'):
            l = l[1:].strip()
        if l:  # Only add non-empty labels
            cleaned.append(l)
    return ', '.join(cleaned)


def save_to_greenplum(df, table_name, schema, password):
    """Save DataFrame to Greenplum database"""
    logger.info(f"Connecting to Greenplum database...")
    
    try:
        # Create SQLAlchemy engine
        connection_string = f"postgresql://{GREENPLUM_USER}:{password}@{GREENPLUM_HOST}:{GREENPLUM_PORT}/{GREENPLUM_DB}"
        engine = create_engine(connection_string)
        
        # Clean column names for database (replace spaces and special characters)
        df_clean = df.copy()
        df_clean.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in df_clean.columns]
        
        # Save to database
        logger.info(f"Saving data to {schema}.{table_name}...")
        df_clean.to_sql(
            name=table_name,
            con=engine,
            schema=schema,
            if_exists='append',  # Append data to existing table
            index=False,
            method='multi'
        )
        
        logger.info(f"✅ Successfully saved {len(df_clean)} rows to {schema}.{table_name}")
        engine.dispose()
        return True
        
    except Exception as e:
        logger.error(f"❌ Error saving to Greenplum: {e}")
        return False


def infer_change_type(diff):
    """Infer change type from diff object"""
    if diff.get("new_file"):
        return "added"
    if diff.get("deleted_file"):
        return "deleted"
    if diff.get("renamed_file"):
        return "renamed"
    return "modified"


def extract_target_type_from_sql_content(content):
    """Extract target_type value from SQL INSERT statement"""
    try:
        # Find INSERT INTO statement with target_type column
        insert_pattern = r"insert\s+into\s+\{\{params\.IKG_Schema\}\}\.\{\{params\.odm_table\}\}\s*\(([^)]+)\)"
        insert_match = re.search(insert_pattern, content, re.IGNORECASE | re.DOTALL)
        
        if not insert_match:
            return ""
        
        columns = [c.strip() for c in insert_match.group(1).split(',')]
        target_idx = next((i for i, c in enumerate(columns) if 'target_type' in c.lower()), -1)
        
        if target_idx < 0:
            return ""
        
        # Find SELECT statement
        select_pattern = r"select\s+(.*?)(?:from|$)"
        select_match = re.search(select_pattern, content, re.IGNORECASE | re.DOTALL)
        
        if not select_match:
            return ""
        
        # Parse values - handle quoted strings
        values_str = select_match.group(1)
        values = []
        current_value = ""
        in_quotes = False
        quote_char = None
        
        for char in values_str:
            if char in ("'", '"') and (not in_quotes or char == quote_char):
                if not in_quotes:
                    in_quotes = True
                    quote_char = char
                else:
                    in_quotes = False
                    quote_char = None
                current_value += char
            elif char == ',' and not in_quotes:
                values.append(current_value.strip())
                current_value = ""
            else:
                current_value += char
        
        if current_value:
            values.append(current_value.strip())
        
        if target_idx < len(values):
            target_value = values[target_idx].strip("'\"")
            return target_value
        
        return ""
    except Exception as e:
        logger.error(f"Error extracting target_type: {e}")
        return ""


def collect_odm_release_details(odm_project, base_branch, df_final):
    """Collect detailed ODM release information including file changes and SQL parsing"""
    logger.info("Collecting ODM release details...")
    
    odm_issues = df_final[df_final['odm_branch_name'].str.strip() != ''].copy()
    
    if len(odm_issues) == 0:
        logger.info("No ODM branches found, skipping ODM release details")
        return pd.DataFrame()
    
    odm_details_list = []
    
    for idx, issue_row in odm_issues.iterrows():
        issue_id = issue_row['id_x']
        odm_branches = [b.strip() for b in str(issue_row['odm_branch_name']).split(',') if b.strip()]
        
        for branch_name in odm_branches:
            logger.info(f"Processing ODM branch: {branch_name} for issue {issue_id}")
            
            try:
                # Get branch object
                branch_obj = odm_project.branches.get(branch_name)
                commit = branch_obj.commit
                
                branch_meta = {
                    'branch_head_sha': commit['id'],
                    'branch_head_short_sha': commit['short_id'],
                    'branch_head_title': commit['title'],
                    'branch_head_author_email': commit['author_email'],
                    'branch_head_committed_at': commit['committed_date'],
                    'branch_head_created_at': commit['created_at'],
                }
                
                # Get file changes using compare
                try:
                    comparison = odm_project.repository_compare(
                        base_branch,
                        branch_name,
                        straight=False
                    )
                    diffs = comparison.get("diffs", []) if comparison else []
                except Exception as e:
                    logger.warning(f"Compare failed for {branch_name}, trying merge request: {e}")
                    diffs = []
                
                # If no diffs from compare, try merge request
                if not diffs:
                    try:
                        mrs = odm_project.mergerequests.list(
                            source_branch=branch_name,
                            state="merged",
                            order_by="updated_at",
                            sort="desc",
                            per_page=1
                        )
                        if mrs:
                            mr = odm_project.mergerequests.get(mrs[0].iid)
                            mr_details = mr.changes()
                            diffs = mr_details.get("changes", [])
                    except Exception as e:
                        logger.warning(f"Merge request lookup failed for {branch_name}: {e}")
                
                # Process each diff
                if diffs:
                    for diff in diffs:
                        file_path = diff.get("new_path") or diff.get("old_path")
                        if not file_path:
                            continue
                        
                        file_name = Path(file_path).name
                        # rule_name only for .sql files, otherwise blank
                        rule_name = Path(file_path).stem if file_path.endswith('.sql') else ''
                        change_type = infer_change_type(diff)
                        
                        # Extract target_type for SQL files
                        target_type = ""
                        if file_path.endswith('.sql') and change_type != 'deleted':
                            try:
                                file_obj = odm_project.files.get(file_path=file_path, ref=branch_name)
                                content = file_obj.decode().decode('utf-8')
                                target_type = extract_target_type_from_sql_content(content)
                            except Exception as e:
                                logger.warning(f"Could not read SQL file {file_path}: {e}")
                        
                        odm_details_list.append({
                            'iteration_end_date': issue_row.get('iteration_end_date', ''),
                            'issue_id': issue_id,
                            'issue_title': issue_row['title'],
                            'issue_state': issue_row['state'],
                            'issue_weight': issue_row['weight'],
                            'issue_labels': clean_labels(issue_row['labels']),
                            'issue_epic': issue_row['epic'],
                            'swat': issue_row.get('swat', ''),
                            'branch_name': branch_name,
                            'file_name': file_name,
                            'rule_name': rule_name,
                            'target_type': target_type,
                            'change_type': change_type,
                            'prod_release_date': issue_row.get('prod_release_date', ''),
                            'linked_issue_id': issue_row.get('linked_issue_id', ''),
                            'linked_project_id': issue_row.get('linked_project_id', ''),
                            'linked_issue_title': issue_row.get('linked_issue_title', ''),
                            'link_type': issue_row.get('link_type', ''),
                            'link_url': issue_row.get('link_url', ''),
                            'iteration_start_date': issue_row.get('iteration_start_date', ''),
                            'file_path': diff.get("new_path", ""),
                            'old_path': diff.get("old_path", ""),
                            'new_path': diff.get("new_path", ""),
                            **branch_meta
                        })
                else:
                    logger.warning(f"No file changes found for branch {branch_name}")
                    
            except Exception as e:
                logger.error(f"Error processing branch {branch_name}: {e}")
                continue
    
    if odm_details_list:
        df_odm_details = pd.DataFrame(odm_details_list)
        logger.info(f"Collected {len(df_odm_details)} ODM file change records")
        return df_odm_details
    
    return pd.DataFrame()


def main():
    logger.info("="*60)
    logger.info("GitLab Release Report Generator - Unified Version")
    logger.info("="*60)
    
    start_time = datetime.now()
    logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get credentials
    PRIVATE_TOKEN = getpass.getpass("Enter your private token: ")
    iteration = input("Current or Previous Iteration: ").strip()
    if iteration == '':
        iteration = 'current'
    logger.info(f"Selected iteration: {iteration}")
    
    # Connect to GitLab
    logger.info("Connecting to GitLab...")
    gl = gitlab.Gitlab(GITLAB_URL, private_token=PRIVATE_TOKEN)
    gl.auth()
    headers = {"PRIVATE-TOKEN": PRIVATE_TOKEN}
    user = gl.user
    logger.info(f"Connected as: {user.username}")
    
    # Get group and iterations
    logger.info(f"Fetching group: {GROUP_PATH}")
    group = gl.groups.get(GROUP_PATH)
    iterations = group.iterations.list(state='current')
    iterations_prev = group.iterations.list(state='closed', get_all=True)
    last_element = iterations_prev[-1]
    
    if not iterations:
        raise Exception("No current iteration found.")
    
    current_iteration_id = iterations[0].id
    current_iteration_title = iterations[0].title
    previous_iteration_id = last_element.id
    
    logger.info(f"Current iteration: {current_iteration_title} (ID: {current_iteration_id})")
    logger.info(f"Previous iteration ID: {previous_iteration_id}")
    
    # Select iteration
    project = gl.projects.get(PROJECT_PATH)
    if iteration == 'Previous':
        iteration_id = previous_iteration_id
        iteration_obj = last_element
    else:
        iteration_id = current_iteration_id
        iteration_obj = iterations[0]
    
    # Get iteration details for dates
    iteration_start_date = iteration_obj.start_date
    iteration_end_date = iteration_obj.due_date
    logger.info(f"Iteration dates: {iteration_start_date} to {iteration_end_date}")
    
    # Get issues
    logger.info("Fetching issues...")
    filtered_issues = project.issues.list(iteration_id=iteration_id, get_all=True)
    logger.info(f"Found {len(filtered_issues)} issues")
    
    # Collect issue data with linked issues
    logger.info("Processing issues and linked issues...")
    rows = []
    for issue in filtered_issues:
        linked_issues = get_linked_issues(PROJECT_PATH, issue.iid, headers)
        if linked_issues:
            for link in linked_issues:
                rows.append({
                    'id': issue.iid,
                    'title': issue.title,
                    'linked_issue_id': link.get('iid'),
                    'linked_project_id': link.get('project_id'),
                    'linked_issue_title': link.get('title'),
                    'link_type': link.get('link_type'),
                    'link_url': link.get('web_url', ''),
                })
        else:
            rows.append({
                'id': issue.iid,
                'title': issue.title,
                'linked_issue_id': None,
                'linked_project_id': None,
                'linked_issue_title': None,
                'link_type': None,
                'link_url': None,
            })
    
    df_linked_issues = pd.DataFrame(rows)
    
    # Convert to DataFrame with issue details
    logger.info("Creating issues dataframe...")
    data = []
    for issue in filtered_issues:
        labels_str = ', '.join(issue.labels) if isinstance(issue.labels, list) else ''
        data.append({
            'id': issue.iid,
            'title': issue.title,
            'state': issue.state,
            'weight': issue.weight if hasattr(issue, 'weight') else None,
            'labels': labels_str,
            'epic': issue.epic['title'] if hasattr(issue, 'epic') and issue.epic else None,
        })
    
    df_issues = pd.DataFrame(data)
    df_issues = df_issues.sort_values(by='id')
    
    # Clean labels
    df_issues['labels'] = df_issues['labels'].apply(clean_labels)
    
    # Convert id to string
    df_issues['id'] = df_issues['id'].astype(str)
    df_linked_issues['id'] = df_linked_issues['id'].astype(str)
    
    # Merge issues with linked issues
    df_issues = pd.merge(df_issues, df_linked_issues, how='left', on='id')
    
    # Get branches from IKG, NLG, ODM projects
    logger.info("Fetching branches from IKG, NLG, ODM projects...")
    ikg_project = gl.projects.get(IKG_PROJECT_PATH)
    nlg_project = gl.projects.get(NLG_PROJECT_PATH)
    odm_project = gl.projects.get(ODM_PROJECT_PATH)
    
    ikg_branches = ikg_project.branches.list(all=True)
    nlg_branches = nlg_project.branches.list(all=True)
    odm_branches = odm_project.branches.list(all=True)
    
    logger.info(f"Found {len(ikg_branches)} IKG, {len(nlg_branches)} NLG, {len(odm_branches)} ODM branches")
    
    # Convert branches to dataframes
    ikg_data = [{'name': branch.name} for branch in ikg_branches]
    df_ikg_branches = pd.DataFrame(ikg_data)
    
    nlg_data = [{'name': branch.name} for branch in nlg_branches]
    df_nlg_branches = pd.DataFrame(nlg_data)
    
    odm_data = [{'name': branch.name} for branch in odm_branches]
    df_odm_branches = pd.DataFrame(odm_data)
    
    # Combine all branches
    df_branches = pd.concat([df_ikg_branches, df_nlg_branches, df_odm_branches], ignore_index=True)
    
    # Extract issue ID from branch names
    logger.info("Extracting issue IDs from branch names...")
    df_branches['id'] = df_branches['name'].apply(extract_digits)
    df_branches['id'] = df_branches['id'].astype(str)
    
    # Get merge requests
    logger.info("Fetching merge requests...")
    target_ikg_branch = ['ikg-master']
    target_nlg_branch = ['nlg-master']
    target_odm_branch = ['odm-master']
    
    filtered_ikg_mrs = []
    for branch in target_ikg_branch:
        mrs = ikg_project.mergerequests.list(target_branch=branch, all=True)
        filtered_ikg_mrs.extend(mrs)
    
    filtered_nlg_mrs = []
    for branch in target_nlg_branch:
        mrs = nlg_project.mergerequests.list(target_branch=branch, all=True)
        filtered_nlg_mrs.extend(mrs)
    
    filtered_odm_mrs = []
    for branch in target_odm_branch:
        mrs = odm_project.mergerequests.list(target_branch=branch, all=True)
        filtered_odm_mrs.extend(mrs)
    
    logger.info(f"Found {len(filtered_ikg_mrs)} IKG, {len(filtered_nlg_mrs)} NLG, {len(filtered_odm_mrs)} ODM MRs")
    
    # Convert MRs to dataframes
    ikg_mr_data = [{'source_branch': mr.source_branch, 'state': mr.state, 'merged_at': mr.merged_at} for mr in filtered_ikg_mrs]
    df_ikg_merge_requests = pd.DataFrame(ikg_mr_data)
    
    nlg_mr_data = [{'source_branch': mr.source_branch, 'state': mr.state, 'merged_at': mr.merged_at} for mr in filtered_nlg_mrs]
    df_nlg_merge_requests = pd.DataFrame(nlg_mr_data)
    
    odm_mr_data = [{'source_branch': mr.source_branch, 'state': mr.state, 'merged_at': mr.merged_at} for mr in filtered_odm_mrs]
    df_odm_merge_requests = pd.DataFrame(odm_mr_data)
    
    # Combine merge requests
    df_merge_requests = pd.concat([df_ikg_merge_requests, df_nlg_merge_requests, df_odm_merge_requests], ignore_index=True)
    df_merge_requests['id'] = df_merge_requests['source_branch'].apply(extract_digits)
    
    # Merge with issues and branches
    logger.info("Merging issues, branches, and merge requests...")
    df_issues_branches = pd.merge(df_issues, df_branches, how='left', on='id')
    df_merge_requests['name'] = df_merge_requests['source_branch'].astype(str)
    df_issues_branches_merge_requests = pd.merge(df_issues_branches, df_merge_requests, how='left', on='name')
    
    # Categorize branches
    logger.info("Categorizing branches...")
    df_issues_branches_merge_requests['ikg_branch_name'] = ''
    df_issues_branches_merge_requests['nlg_branch_name'] = ''
    df_issues_branches_merge_requests['odm_branch_name'] = ''
    df_issues_branches_merge_requests['ikg_merged'] = ''
    df_issues_branches_merge_requests['nlg_merged'] = ''
    df_issues_branches_merge_requests['odm_merged'] = ''
    
    # Update branch names
    df_issues_branches_merge_requests['ikg_branch_name'] = np.where(
        df_issues_branches_merge_requests['name'].str.contains('ikg', case=False, na=False),
        df_issues_branches_merge_requests['name'],
        ''
    )
    
    df_issues_branches_merge_requests['nlg_branch_name'] = np.where(
        df_issues_branches_merge_requests['name'].str.contains('nlg', case=False, na=False),
        df_issues_branches_merge_requests['name'],
        ''
    )
    
    df_issues_branches_merge_requests['odm_branch_name'] = np.where(
        df_issues_branches_merge_requests['name'].str.contains('odm', case=False, na=False),
        df_issues_branches_merge_requests['name'],
        ''
    )
    
    # Set merged status
    choices = ['', 'Yes', 'No']
    
    # IKG merged
    df_issues_branches_merge_requests['ikg_branch_name'] = df_issues_branches_merge_requests['ikg_branch_name'].astype(str)
    conditions = [
        (df_issues_branches_merge_requests['ikg_branch_name'].str.strip() == ''),
        ((df_issues_branches_merge_requests['ikg_branch_name'].str.strip() != '') & (df_issues_branches_merge_requests['state_y'] == 'merged')),
        ((df_issues_branches_merge_requests['ikg_branch_name'].str.strip() != '') & (df_issues_branches_merge_requests['state_y'] != 'merged'))
    ]
    df_issues_branches_merge_requests['ikg_merged'] = np.select(conditions, choices, default='')
    
    # NLG merged
    df_issues_branches_merge_requests['nlg_branch_name'] = df_issues_branches_merge_requests['nlg_branch_name'].astype(str)
    conditions = [
        (df_issues_branches_merge_requests['nlg_branch_name'].str.strip() == ''),
        ((df_issues_branches_merge_requests['nlg_branch_name'].str.strip() != '') & (df_issues_branches_merge_requests['state_y'] == 'merged')),
        ((df_issues_branches_merge_requests['nlg_branch_name'].str.strip() != '') & (df_issues_branches_merge_requests['state_y'] != 'merged'))
    ]
    df_issues_branches_merge_requests['nlg_merged'] = np.select(conditions, choices, default='')
    
    # ODM merged
    df_issues_branches_merge_requests['odm_branch_name'] = df_issues_branches_merge_requests['odm_branch_name'].astype(str)
    conditions = [
        (df_issues_branches_merge_requests['odm_branch_name'].str.strip() == ''),
        ((df_issues_branches_merge_requests['odm_branch_name'].str.strip() != '') & (df_issues_branches_merge_requests['state_y'] == 'merged')),
        ((df_issues_branches_merge_requests['odm_branch_name'].str.strip() != '') & (df_issues_branches_merge_requests['state_y'] != 'merged'))
    ]
    df_issues_branches_merge_requests['odm_merged'] = np.select(conditions, choices, default='')
    
    # Add CID and SWAT columns
    df_issues_branches_merge_requests['cid'] = np.where(
        df_issues_branches_merge_requests['labels'].str.contains('cid', case=False, na=False),
        'Yes', ''
    )
    
    df_issues_branches_merge_requests['swat'] = np.where(
        df_issues_branches_merge_requests['labels'].str.contains('swat', case=False, na=False),
        'Yes', ''
    )
    
    # Add iteration column
    df_issues_branches_merge_requests['iteration'] = iteration
    
    # Filter closed state
    df_issues_branches_merge_requests = df_issues_branches_merge_requests[
        df_issues_branches_merge_requests['state_y'] != 'closed'
    ]
    
    # Group by issue ID
    logger.info("Grouping by issue ID and aggregating data...")
    
    # Create a list to hold expanded rows with individual branch merge status
    expanded_rows = []
    
    for issue_id, group in df_issues_branches_merge_requests.groupby('id_x'):
        # Get basic issue info
        issue_info = {
            'id_x': issue_id,
            'title': group['title_x'].iloc[0],
            'state': group['state_x'].iloc[0],
            'weight': group['weight'].iloc[0],
            'labels': group['labels'].iloc[0],
            'epic': group['epic'].iloc[0],
            'cid': group['cid'].iloc[0],
            'swat': group['swat'].iloc[0],
            'iteration': group['iteration'].iloc[0],
        }
        
        # Collect all branches (regardless of category)
        all_branches = []
        ikg_branches_list = []
        nlg_branches_list = []
        odm_branches_list = []
        ikg_merged_list = []
        nlg_merged_list = []
        odm_merged_list = []
        
        for _, row in group.iterrows():
            branch = row.get('name', '')
            if pd.notna(branch) and str(branch).strip():
                all_branches.append(str(branch).strip())
                
                # Categorize by type
                if row['ikg_branch_name']:
                    ikg_branches_list.append(str(row['ikg_branch_name']).strip())
                    ikg_merged_list.append(row['ikg_merged'])
                
                if row['nlg_branch_name']:
                    nlg_branches_list.append(str(row['nlg_branch_name']).strip())
                    nlg_merged_list.append(row['nlg_merged'])
                
                if row['odm_branch_name']:
                    odm_branches_list.append(str(row['odm_branch_name']).strip())
                    odm_merged_list.append(row['odm_merged'])
        
        # Remove duplicates while preserving order
        all_branches = list(dict.fromkeys(all_branches))
        ikg_branches_list = list(dict.fromkeys(ikg_branches_list))
        nlg_branches_list = list(dict.fromkeys(nlg_branches_list))
        odm_branches_list = list(dict.fromkeys(odm_branches_list))
        
        # Handle merged status for multiple branches of same type
        ikg_merged_final = ', '.join(ikg_merged_list) if ikg_merged_list else ''
        nlg_merged_final = ', '.join(nlg_merged_list) if nlg_merged_list else ''
        odm_merged_final = ', '.join(odm_merged_list) if odm_merged_list else ''
        
        issue_info.update({
            'branch_name': ', '.join(all_branches),
            'ikg_branch_name': ', '.join(ikg_branches_list),
            'nlg_branch_name': ', '.join(nlg_branches_list),
            'odm_branch_name': ', '.join(odm_branches_list),
            'ikg_merged': ikg_merged_final,
            'nlg_merged': nlg_merged_final,
            'odm_merged': odm_merged_final,
        })
        
        # Aggregate linked issues
        linked_issue_ids = []
        linked_project_ids = []
        linked_issue_titles = []
        link_types = []
        link_urls = []
        
        for _, row in group.iterrows():
            if pd.notna(row.get('linked_issue_id')) and str(row.get('linked_issue_id')).strip():
                linked_issue_ids.append(str(row['linked_issue_id']).strip())
            if pd.notna(row.get('linked_project_id')) and str(row.get('linked_project_id')).strip():
                linked_project_ids.append(str(row['linked_project_id']).strip())
            if pd.notna(row.get('linked_issue_title')) and str(row.get('linked_issue_title')).strip():
                linked_issue_titles.append(str(row['linked_issue_title']).strip())
            if pd.notna(row.get('link_type')) and str(row.get('link_type')).strip():
                link_types.append(str(row['link_type']).strip())
            if pd.notna(row.get('link_url')) and str(row.get('link_url')).strip():
                link_urls.append(str(row['link_url']).strip())
        
        # Remove duplicates while preserving order
        linked_issue_ids = list(dict.fromkeys(linked_issue_ids))
        linked_project_ids = list(dict.fromkeys(linked_project_ids))
        linked_issue_titles = list(dict.fromkeys(linked_issue_titles))
        link_types = list(dict.fromkeys(link_types))
        link_urls = list(dict.fromkeys(link_urls))
        
        issue_info.update({
            'linked_issue_id': ', '.join(linked_issue_ids),
            'linked_project_id': ', '.join(linked_project_ids),
            'linked_issue_title': ', '.join(linked_issue_titles),
            'link_type': ', '.join(link_types),
            'link_url': ', '.join(link_urls),
        })
        
        expanded_rows.append(issue_info)
    
    df_final = pd.DataFrame(expanded_rows)
    
    # Convert linked_issue_id and linked_project_id to integers (remove decimals)
    def convert_to_int_list(val):
        if pd.isna(val) or val == '':
            return ''
        parts = [p.strip() for p in str(val).split(',') if p.strip()]
        int_parts = []
        for p in parts:
            try:
                # Convert to int to remove decimal
                int_parts.append(str(int(float(p))))
            except (ValueError, TypeError):
                int_parts.append(p)
        return ', '.join(int_parts)
    
    df_final['linked_issue_id'] = df_final['linked_issue_id'].apply(convert_to_int_list)
    df_final['linked_project_id'] = df_final['linked_project_id'].apply(convert_to_int_list)
    
    # Calculate iteration dates
    iteration_end = datetime.strptime(iteration_end_date, '%Y-%m-%d')
    preprod_release = iteration_end + timedelta(days=1)
    prod_release = iteration_end + timedelta(days=13)
    
    df_final['iteration_start_date'] = iteration_start_date
    df_final['iteration_end_date'] = iteration_end_date
    df_final['preprod_release_date'] = preprod_release.strftime('%Y-%m-%d')
    df_final['prod_release_date'] = prod_release.strftime('%Y-%m-%d')
    
    # Reorder columns
    col_order = [
        'iteration', 'state', 'iteration_end_date', 'title', 'id_x', 'weight', 'labels', 'epic',
        'swat', 'preprod_release_date', 'prod_release_date', 
        'ikg_merged', 'ikg_branch_name', 'nlg_merged', 'nlg_branch_name',
        'odm_merged', 'odm_branch_name', 'cid', 
        'linked_issue_id', 'linked_project_id', 'linked_issue_title', 'link_type', 'link_url',
        'iteration_start_date', 'branch_name'
    ]
    df_final = df_final[col_order]
    
    # Collect ODM release details
    df_odm_details = collect_odm_release_details(odm_project, DEFAULT_BASE_BRANCH, df_final)
    
    # Export to Excel
    logger.info("Exporting to Excel...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'issues_branches_merge_requests_{timestamp}.xlsx'
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df_final.to_excel(writer, sheet_name='Issues_Report', index=False)
        if not df_odm_details.empty:
            df_odm_details.to_excel(writer, sheet_name='odm_release_details', index=False)
    
    # Apply styling to Issues_Report sheet
    logger.info("Applying styling...")
    wb = load_workbook(output_file)
    ws = wb['Issues_Report']
    
    red_fill = PatternFill(start_color='FF0000', end_color='FF0000', fill_type='solid')
    
    for idx, row in df_final.iterrows():
        excel_row = idx + 2
        
        ikg_branches = [b.strip() for b in str(row['ikg_branch_name']).split(',') if b.strip()]
        nlg_branches = [b.strip() for b in str(row['nlg_branch_name']).split(',') if b.strip()]
        odm_branches = [b.strip() for b in str(row['odm_branch_name']).split(',') if b.strip()]
        
        all_branches = ikg_branches + nlg_branches + odm_branches
        uncategorized = False
        for branch in all_branches:
            branch_lower = branch.lower()
            if 'ikg' not in branch_lower and 'nlg' not in branch_lower and 'odm' not in branch_lower:
                uncategorized = True
                break
        
        if uncategorized:
            for cell in ws[excel_row]:
                cell.fill = red_fill
        else:
            if len(ikg_branches) > 1:
                ws.cell(row=excel_row, column=col_order.index('ikg_branch_name')+1).fill = red_fill
            if len(nlg_branches) > 1:
                ws.cell(row=excel_row, column=col_order.index('nlg_branch_name')+1).fill = red_fill
            if len(odm_branches) > 1:
                ws.cell(row=excel_row, column=col_order.index('odm_branch_name')+1).fill = red_fill
    
    wb.save(output_file)
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    logger.info("="*60)
    logger.info(f"✅ Report generated successfully: {output_file}")
    logger.info(f"📊 Sheet 1: Issues_Report with {len(df_final)} issues")
    logger.info(f"📊 Sheet 2: odm_release_details with {len(df_odm_details)} file changes")
    logger.info(f"⏱️  Total duration: {duration}")
    logger.info("="*60)
    
    # Save to Greenplum database
    save_to_db = input("\nSave data to Greenplum database? (yes/no): ").strip().lower()
    if save_to_db in ['yes', 'y']:
        logger.info("\n" + "="*60)
        logger.info("Saving data to Greenplum database...")
        logger.info("="*60)
        
        GREENPLUM_PASSWORD = getpass.getpass("Enter Greenplum database password: ")
        
        # Save Issues Report
        logger.info(f"\n📊 Saving Issues_Report to {GREENPLUM_SCHEMA}.{OUTPUT_TABLE1}...")
        success1 = save_to_greenplum(df_final, OUTPUT_TABLE1, GREENPLUM_SCHEMA, GREENPLUM_PASSWORD)
        
        # Save ODM Release Details
        if not df_odm_details.empty:
            logger.info(f"\n📊 Saving odm_release_details to {GREENPLUM_SCHEMA}.{OUTPUT_TABLE2}...")
            success2 = save_to_greenplum(df_odm_details, OUTPUT_TABLE2, GREENPLUM_SCHEMA, GREENPLUM_PASSWORD)
        else:
            logger.info("\n⚠️  No ODM details to save")
            success2 = True
        
        if success1 and success2:
            logger.info("\n" + "="*60)
            logger.info("✅ All data successfully saved to Greenplum!")
            logger.info("="*60)
        else:
            logger.warning("\n⚠️  Some data failed to save to Greenplum")
    else:
        logger.info("\n⏭️  Skipping database save")
    
    return output_file


if __name__ == "__main__":
    main()
