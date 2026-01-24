#!/usr/bin/env python3
"""
GitLab Release Report Generator
Generates Excel reports for IKG, NLG, and ODM branches linked to iteration issues
"""

import gitlab
import pandas as pd
import numpy as np
import getpass
import requests
import re
from datetime import datetime, timedelta
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
GITLAB_URL = 'https://devcloud.ubs.net'
GROUP_PATH = 'ubs/gwma/smart-technology-and-analytics/staat-data-science/staat-ds-insights-ci/commons'
PROJECT_PATH = 'ubs/gwma/smart-technology-and-analytics/staat-data-science/staat-ds-insights-home'
GENESIS_GROUP_PATH = 'ubs/gwma/smart-technology-and-analytics/staat-data-science/staat-ds-insights-ci/commons/staat-ds-insights-home/genesis/genesis-platform'
IKG_PROJECT_PATH = 'ubs/gwma/smart-technology-and-analytics/staat-data-science/staat-ds-genesis/genesis-platform/ikg-dags'
NLG_PROJECT_PATH = 'ubs/gwma/smart-technology-and-analytics/staat-data-science/staat-ds-genesis/genesis-platform/nlg-dags'
ODM_PROJECT_PATH = 'ubs/gwma/smart-technology-and-analytics/staat-data-science/staat-ds-genesis/genesis-platform/odm-dags'

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
    """Remove status:: prefix from labels"""
    if pd.isna(labels) or labels == '':
        return ''
    label_list = [l.strip() for l in str(labels).split(',')]
    cleaned = [l for l in label_list if not l.startswith('status::')]
    return ', '.join(cleaned)

def fetch_branch_commits(project, branch_name):
    """Fetch commit details for a specific branch"""
    try:
        branch = project.branches.get(branch_name)
        commit = branch.commit
        return {
            'branch_head_sha': commit['id'],
            'branch_head_short_sha': commit['short_id'],
            'branch_head_title': commit['title'],
            'branch_head_author_email': commit['author_email'],
            'branch_head_commited_at': commit['committed_date'],
            'branch_head_created_at': commit['created_at']
        }
    except Exception as e:
        logger.error(f"Error fetching commit for branch {branch_name}: {e}")
        return {}

def fetch_branch_file_changes(project, branch_name):
    """Fetch file changes for a specific branch"""
    try:
        branch = project.branches.get(branch_name)
        commit = project.commits.get(branch.commit['id'])
        diffs = commit.diff()
        
        file_changes = []
        for diff in diffs:
            file_changes.append({
                'file_name': diff['new_path'],
                'change_type': 'added' if diff['new_file'] else ('deleted' if diff['deleted_file'] else 'modified')
            })
        return file_changes
    except Exception as e:
        logger.error(f"Error fetching file changes for branch {branch_name}: {e}")
        return []

def extract_target_type_from_sql(project, branch_name, file_path):
    """Extract target_type from SQL file in specific branch"""
    try:
        file_content = project.files.get(file_path=file_path, ref=branch_name)
        content = file_content.decode().decode('utf-8')
        
        # Parse SQL to find target_type value in insert statement
        pattern = r"insert\s+into\s+\{\{params\.IKG_Schema\}\}\.\{\{params\.odm_table\}\}\s*\([^)]*target_type[^)]*\)\s*select\s+(?:[^,]*,\s*)*'([^']*)'.*target_type"
        match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
        if match:
            # Find the position of target_type in column list
            insert_match = re.search(r"insert\s+into[^(]*\(([^)]*)\)", content, re.IGNORECASE)
            if insert_match:
                columns = [c.strip() for c in insert_match.group(1).split(',')]
                target_idx = next((i for i, c in enumerate(columns) if 'target_type' in c.lower()), -1)
                
                if target_idx >= 0:
                    select_match = re.search(r"select\s+(.*?)(?:from|$)", content, re.IGNORECASE | re.DOTALL)
                    if select_match:
                        values = [v.strip() for v in select_match.group(1).split(',')]
                        if target_idx < len(values):
                            target_value = values[target_idx].strip("'\"")
                            return target_value
        return ''
    except Exception as e:
        logger.error(f"Error extracting target_type from {file_path} in branch {branch_name}: {e}")
        return ''

def main():
    logger.info("Starting GitLab Release Report Generation")
    
    # Get credentials
    PRIVATE_TOKEN = getpass.getpass("Enter your private token: ")
    iteration = input("Current or Previous Iteration: ").strip()
    if iteration == '':
        iteration = 'current'
    logger.info(f"Selected iteration: {iteration}")
    
    start_time = datetime.now()
    logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
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
                })
        else:
            rows.append({
                'id': issue.iid,
                'title': issue.title,
                'linked_issue_id': None,
                'linked_project_id': None,
                'linked_issue_title': None,
                'link_type': None,
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
            'total_time_spent': issue.time_stats()['total_time_spent'] / 28800 if issue.time_stats else None,
            'author': issue.author['username'] if issue.author else None,
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
    
    logger.info(f"Found {len(ikg_branches)} IKG branches, {len(nlg_branches)} NLG branches, {len(odm_branches)} ODM branches")
    
    # Convert branches to dataframes
    ikg_data = []
    for branch in ikg_branches:
        ikg_data.append({
            'name': branch.name,
        })
    df_ikg_branches = pd.DataFrame(ikg_data)
    
    nlg_data = []
    for branch in nlg_branches:
        nlg_data.append({
            'name': branch.name,
        })
    df_nlg_branches = pd.DataFrame(nlg_data)
    
    odm_data = []
    for branch in odm_branches:
        odm_data.append({
            'name': branch.name,
        })
    df_odm_branches = pd.DataFrame(odm_data)
    
    # Combine all branches
    df_branches = pd.concat([df_ikg_branches, df_nlg_branches, df_odm_branches], ignore_index=True)
    
    # Extract issue ID from branch names
    logger.info("Extracting issue IDs from branch names...")
    df_branches['id'] = df_branches['name'].apply(extract_digits)
    
    # Convert to string
    df_branches['id'] = df_branches['id'].astype(str)
    
    # Merge issues with branches
    df_issues_branches = pd.merge(df_issues, df_branches, how='left', on='id')
    df_issues_branches = df_issues_branches.sort_values(by='id')
    
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
    
    logger.info(f"Found {len(filtered_ikg_mrs)} IKG MRs, {len(filtered_nlg_mrs)} NLG MRs, {len(filtered_odm_mrs)} ODM MRs")
    
    # Convert MRs to dataframes
    ikg_mr_data = []
    for mr in filtered_ikg_mrs:
        approvals = mr.approvals.get()
        approved_by_users = [user['user']['name'] for user in approvals.approved_by]
        ikg_mr_data.append({
            'source_branch': mr.source_branch,
            'state': mr.state,
            'merged_at': mr.merged_at,
        })
    df_ikg_merge_requests = pd.DataFrame(ikg_mr_data)
    
    nlg_mr_data = []
    for mr in filtered_nlg_mrs:
        approvals = mr.approvals.get()
        approved_by_users = [user['user']['name'] for user in approvals.approved_by]
        nlg_mr_data.append({
            'source_branch': mr.source_branch,
            'state': mr.state,
            'merged_at': mr.merged_at,
        })
    df_nlg_merge_requests = pd.DataFrame(nlg_mr_data)
    
    odm_mr_data = []
    for mr in filtered_odm_mrs:
        approvals = mr.approvals.get()
        approved_by_users = [user['user']['name'] for user in approvals.approved_by]
        odm_mr_data.append({
            'source_branch': mr.source_branch,
            'state': mr.state,
            'merged_at': mr.merged_at,
        })
    df_odm_merge_requests = pd.DataFrame(odm_mr_data)
    
    # Combine merge requests
    df_merge_requests = pd.concat([df_ikg_merge_requests, df_nlg_merge_requests, df_odm_merge_requests], ignore_index=True)
    df_merge_requests['id'] = df_merge_requests['source_branch'].apply(extract_digits)
    
    # Merge with issues and branches
    logger.info("Merging issues, branches, and merge requests...")
    df_issues_branches = pd.merge(df_issues, df_branches, how='left', on='id')
    df_merge_requests['name'] = df_merge_requests['source_branch'].astype(str)
    df_issues_branches_merge_requests = pd.merge(df_issues_branches, df_merge_requests, how='left', on='name')
    
    df_issues_mr = pd.merge(df_issues, df_merge_requests, how='left', on='id')
    df_issues_mr = df_issues_mr[~df_issues_mr['source_branch'].isin(df_branches['name'])]
    
    # Filter only merged records
    df_issues_mr = df_issues_mr[
        (df_issues_mr['source_branch'].notna()) &
        (df_issues_mr['source_branch'].astype(str).str.strip() != '') &
        (df_issues_mr['state_y'] == 'merged')
    ]
    
    # Categorize branches
    logger.info("Categorizing branches...")
    df_issues_branches_merge_requests['ikg_branch_name'] = ''
    df_issues_branches_merge_requests['nlg_branch_name'] = ''
    df_issues_branches_merge_requests['odm_branch_name'] = ''
    df_issues_branches_merge_requests['ikg_merged'] = ''
    df_issues_branches_merge_requests['nlg_merged'] = ''
    df_issues_branches_merge_requests['odm_merged'] = ''
    
    # Update IKG branch name
    df_issues_branches_merge_requests['ikg_branch_name'] = np.where(
        df_issues_branches_merge_requests['name'].str.contains('ikg', case=False, na=False),
        df_issues_branches_merge_requests['name'],
        df_issues_branches_merge_requests['ikg_branch_name']
    )
    
    # Update NLG branch name
    df_issues_branches_merge_requests['nlg_branch_name'] = np.where(
        df_issues_branches_merge_requests['name'].str.contains('nlg', case=False, na=False),
        df_issues_branches_merge_requests['name'],
        df_issues_branches_merge_requests['nlg_branch_name']
    )
    
    # Update ODM branch name
    df_issues_branches_merge_requests['odm_branch_name'] = np.where(
        df_issues_branches_merge_requests['name'].str.contains('odm', case=False, na=False),
        df_issues_branches_merge_requests['name'],
        df_issues_branches_merge_requests['odm_branch_name']
    )
    
    # IKG merged status
    df_issues_branches_merge_requests['ikg_branch_name'] = df_issues_branches_merge_requests['ikg_branch_name'].astype(str)
    conditions = [
        (df_issues_branches_merge_requests['ikg_branch_name'].str.strip() == ''),
        ((df_issues_branches_merge_requests['ikg_branch_name'].str.strip() != '') & (df_issues_branches_merge_requests['state_y'] == 'merged')),
        ((df_issues_branches_merge_requests['ikg_branch_name'].str.strip() != '') & (df_issues_branches_merge_requests['state_y'] != 'merged'))
    ]
    choices = ['', 'Yes', 'No']
    df_issues_branches_merge_requests['ikg_merged'] = np.select(conditions, choices, default='')
    
    # NLG merged status
    df_issues_branches_merge_requests['nlg_branch_name'] = df_issues_branches_merge_requests['nlg_branch_name'].astype(str)
    conditions = [
        (df_issues_branches_merge_requests['nlg_branch_name'].str.strip() == ''),
        ((df_issues_branches_merge_requests['nlg_branch_name'].str.strip() != '') & (df_issues_branches_merge_requests['state_y'] == 'merged')),
        ((df_issues_branches_merge_requests['nlg_branch_name'].str.strip() != '') & (df_issues_branches_merge_requests['state_y'] != 'merged'))
    ]
    df_issues_branches_merge_requests['nlg_merged'] = np.select(conditions, choices, default='')
    
    # ODM merged status
    df_issues_branches_merge_requests['odm_branch_name'] = df_issues_branches_merge_requests['odm_branch_name'].astype(str)
    conditions = [
        (df_issues_branches_merge_requests['odm_branch_name'].str.strip() == ''),
        ((df_issues_branches_merge_requests['odm_branch_name'].str.strip() != '') & (df_issues_branches_merge_requests['state_y'] == 'merged')),
        ((df_issues_branches_merge_requests['odm_branch_name'].str.strip() != '') & (df_issues_branches_merge_requests['state_y'] != 'merged'))
    ]
    df_issues_branches_merge_requests['odm_merged'] = np.select(conditions, choices, default='')
    
    # Replace blanks with empty string
    df_issues_branches_merge_requests['ikg_branch_name'] = df_issues_branches_merge_requests['ikg_branch_name'].replace(r'^\s*$', '', regex=True).fillna('')
    df_issues_branches_merge_requests['nlg_branch_name'] = df_issues_branches_merge_requests['nlg_branch_name'].replace(r'^\s*$', '', regex=True).fillna('')
    df_issues_branches_merge_requests['odm_branch_name'] = df_issues_branches_merge_requests['odm_branch_name'].replace(r'^\s*$', '', regex=True).fillna('')
    
    # Add CID and SWAT columns
    df_issues_branches_merge_requests['cid'] = ''
    df_issues_branches_merge_requests['cid'] = np.where(
        df_issues_branches_merge_requests['labels'].str.contains('cid', case=False, na=False),
        'Yes',
        df_issues_branches_merge_requests['cid']
    )
    
    df_issues_branches_merge_requests['swat'] = ''
    df_issues_branches_merge_requests['swat'] = np.where(
        df_issues_branches_merge_requests['labels'].str.contains('swat', case=False, na=False),
        'Yes',
        df_issues_branches_merge_requests['swat']
    )
    
    # Add iteration column
    df_issues_branches_merge_requests['iteration'] = iteration
    
    # Filter closed state
    df_issues_branches_merge_requests = df_issues_branches_merge_requests[df_issues_branches_merge_requests['state_y'] != 'closed']
    
    # Group by issue ID and concatenate multiple values
    logger.info("Grouping by issue ID and aggregating data...")
    agg_dict = {
        'title_x': 'first',
        'state_x': 'first',
        'weight': 'first',
        'labels': 'first',
        'epic': 'first',
        'cid': 'first',
        'swat': 'first',
        'iteration': 'first',
        'ikg_merged': lambda x: ', '.join(filter(None, x.unique())),
        'ikg_branch_name': lambda x: ', '.join(filter(None, x.unique())),
        'nlg_merged': lambda x: ', '.join(filter(None, x.unique())),
        'nlg_branch_name': lambda x: ', '.join(filter(None, x.unique())),
        'odm_merged': lambda x: ', '.join(filter(None, x.unique())),
        'odm_branch_name': lambda x: ', '.join(filter(None, x.unique())),
        'linked_issue_id': lambda x: ', '.join(filter(None, [str(v) for v in x.unique() if pd.notna(v)])),
        'linked_project_id': lambda x: ', '.join(filter(None, [str(v) for v in x.unique() if pd.notna(v)])),
        'linked_issue_title': lambda x: ', '.join(filter(None, x.unique())),
        'link_type': lambda x: ', '.join(filter(None, x.unique())),
    }
    
    df_final = df_issues_branches_merge_requests.groupby('id_x', as_index=False).agg(agg_dict)
    
    # Rename columns back to original names
    df_final = df_final.rename(columns={'title_x': 'title', 'state_x': 'state'})
    
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
        'iteration', 'id_x', 'title', 'state', 'weight', 'labels', 'epic',
        'cid', 'swat', 'ikg_merged', 'ikg_branch_name', 'nlg_merged', 'nlg_branch_name',
        'odm_merged', 'odm_branch_name', 'linked_issue_id', 'linked_project_id',
        'linked_issue_title', 'link_type', 'iteration_start_date', 'iteration_end_date',
        'preprod_release_date', 'prod_release_date'
    ]
    df_final = df_final[col_order]
    
    # Export to Excel with styling
    logger.info("Exporting to Excel...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'issues_branches_merge_requests_{timestamp}.xlsx'
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df_final.to_excel(writer, sheet_name='Issues_Report', index=False)
    
    # Apply styling
    logger.info("Applying styling...")
    wb = load_workbook(output_file)
    ws = wb['Issues_Report']
    
    red_fill = PatternFill(start_color='FF0000', end_color='FF0000', fill_type='solid')
    
    # Check for branches that can't be categorized
    for idx, row in df_final.iterrows():
        excel_row = idx + 2  # +2 because of header and 0-indexing
        
        # Get all branch names for this issue
        ikg_branches = [b.strip() for b in str(row['ikg_branch_name']).split(',') if b.strip()]
        nlg_branches = [b.strip() for b in str(row['nlg_branch_name']).split(',') if b.strip()]
        odm_branches = [b.strip() for b in str(row['odm_branch_name']).split(',') if b.strip()]
        
        # Check if any branch can't be categorized (doesn't contain ikg, nlg, or odm)
        all_branches = ikg_branches + nlg_branches + odm_branches
        uncategorized = False
        for branch in all_branches:
            branch_lower = branch.lower()
            if 'ikg' not in branch_lower and 'nlg' not in branch_lower and 'odm' not in branch_lower:
                uncategorized = True
                break
        
        # Color entire row red if uncategorized
        if uncategorized:
            for cell in ws[excel_row]:
                cell.fill = red_fill
        else:
            # Check for multiple branches of same type
            if len(ikg_branches) > 1:
                ws.cell(row=excel_row, column=col_order.index('ikg_branch_name')+1).fill = red_fill
            if len(nlg_branches) > 1:
                ws.cell(row=excel_row, column=col_order.index('nlg_branch_name')+1).fill = red_fill
            if len(odm_branches) > 1:
                ws.cell(row=excel_row, column=col_order.index('odm_branch_name')+1).fill = red_fill
    
    # Create ODM Release Details sheet
    logger.info("Creating ODM Release Details sheet...")
    odm_issues = df_final[df_final['odm_branch_name'].str.strip() != ''].copy()
    
    if len(odm_issues) > 0:
        odm_details_list = []
        
        for _, issue_row in odm_issues.iterrows():
            issue_id = issue_row['id_x']
            odm_branches = [b.strip() for b in str(issue_row['odm_branch_name']).split(',') if b.strip()]
            
            for branch_name in odm_branches:
                logger.info(f"Processing ODM branch {branch_name} for issue {issue_id}...")
                
                # Get branch commit details
                commit_details = fetch_branch_commits(odm_project, branch_name)
                
                # Get file changes
                file_changes = fetch_branch_file_changes(odm_project, branch_name)
                
                for file_change in file_changes:
                    file_name = file_change['file_name']
                    change_type = file_change['change_type']
                    
                    # Extract rule name (remove extension)
                    rule_name = file_name.rsplit('.', 1)[0] if '.' in file_name else file_name
                    
                    # Extract target_type if it's a SQL file
                    target_type = ''
                    if file_name.endswith('.sql'):
                        target_type = extract_target_type_from_sql(odm_project, branch_name, file_name)
                    
                    odm_details_list.append({
                        'issue_id': issue_id,
                        'issue_title': issue_row['title'],
                        'status': issue_row['state'],
                        'weight': issue_row['weight'],
                        'labels': issue_row['labels'],
                        'epic': issue_row['epic'],
                        'odm_branch_name': branch_name,
                        'file_name': file_name,
                        'rule_name': rule_name,
                        'target_type': target_type,
                        'change_type': change_type,
                        **commit_details
                    })
        
        df_odm_details = pd.DataFrame(odm_details_list)
        
        # Add to workbook
        with pd.ExcelWriter(output_file, engine='openpyxl', mode='a') as writer:
            df_odm_details.to_excel(writer, sheet_name='odm_release_details', index=False)
    
    wb.save(output_file)
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    logger.info(f"Report generated successfully: {output_file}")
    logger.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total duration: {duration}")

if __name__ == "__main__":
    main()
