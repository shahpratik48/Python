"""
GitLab Issues Fetcher with Issue Links and Greenplum Integration
Fetches issues from IKG and SWAT projects with comprehensive linkage details
Links are consolidated per issue with comma-separated values
"""

import requests
import pandas as pd
import getpass
from typing import List, Dict, Any, Tuple
from datetime import datetime
import psycopg2
from psycopg2 import sql
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils.dataframe import dataframe_to_rows
import warnings
warnings.filterwarnings('ignore')


class GitLabIssuesFetcher:
    """Fetch all issues from GitLab projects with comprehensive linkage details"""
    
    def __init__(self, gitlab_url: str, ikg_project_path: str, swat_project_path: str):
        """Initialize GitLab Issues Fetcher"""
        self.gitlab_url = gitlab_url.rstrip('/')
        self.ikg_project_path = ikg_project_path
        self.swat_project_path = swat_project_path
        self.token = None
        self.headers = None
        
    def authenticate(self):
        """Get authentication token from user"""
        self.token = getpass.getpass("Enter your GitLab Personal Access Token: ")
        self.headers = {
            'PRIVATE-TOKEN': self.token,
            'Content-Type': 'application/json'
        }
        print("✓ GitLab authentication configured")
        
    def get_project_id(self, project_path: str) -> Tuple[str, str]:
        """Get project ID and name from project path"""
        encoded_path = requests.utils.quote(project_path, safe='')
        url = f"{self.gitlab_url}/api/v4/projects/{encoded_path}"
        
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        project_data = response.json()
        
        project_id = project_data['id']
        project_name = project_data['name']
        
        print(f"✓ Found project: {project_name} (ID: {project_id})")
        return project_id, project_name
    
    def fetch_issue_links(self, project_id: str, issue_iid: int) -> List[Dict[str, Any]]:
        """Fetch all links for a specific issue"""
        url = f"{self.gitlab_url}/api/v4/projects/{project_id}/issues/{issue_iid}/links"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except:
            return []
    
    def fetch_all_issues(self, project_id: str, project_name: str) -> List[Dict[str, Any]]:
        """Fetch all issues from the project with linkage details"""
        all_issues = []
        page = 1
        per_page = 100
        
        url = f"{self.gitlab_url}/api/v4/projects/{project_id}/issues"
        
        print(f"\nFetching issues from {project_name}...")
        
        while True:
            params = {
                'per_page': per_page,
                'page': page,
                'state': 'all',
                'scope': 'all',
                'with_labels_details': True
            }
            
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            issues = response.json()
            
            if not issues:
                break
            
            # Fetch links for each issue
            for issue in issues:
                issue['_links_data'] = self.fetch_issue_links(project_id, issue['iid'])
            
            all_issues.extend(issues)
            print(f"  Page {page}: {len(issues)} issues (Total: {len(all_issues)})")
            
            if len(issues) < per_page:
                break
            
            page += 1
        
        print(f"✓ Total issues fetched: {len(all_issues)}")
        return all_issues
    
    def extract_issue_data(self, issues: List[Dict[str, Any]], project_identifier: str) -> pd.DataFrame:
        """
        Extract comprehensive parameters from issues with consolidated linkage details.
        One row per issue_id with comma-separated values for links.
        """
        extracted_data = []
        current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        for issue in issues:
            # Base issue data
            assignees = [a.get('name', '') for a in issue.get('assignees', [])]
            assignee_str = ', '.join(assignees) if assignees else None
            
            labels = issue.get('labels', [])
            label_str = ', '.join(labels) if labels else None
            
            iteration = issue.get('iteration', {}).get('title', '') if issue.get('iteration') else None
            
            epic = issue.get('epic', {}).get('title', '') if issue.get('epic') else None
            epic_iid = issue.get('epic', {}).get('iid', '') if issue.get('epic') else None
            
            participants = [p.get('name', '') for p in issue.get('participants', [])]
            participants_str = ', '.join(participants) if participants else None
            
            milestone = issue.get('milestone', {}).get('title', '') if issue.get('milestone') else None
            
            time_estimate = issue.get('time_stats', {}).get('time_estimate')
            time_spent = issue.get('time_stats', {}).get('total_time_spent')
            
            task_completion = None
            if issue.get('task_completion_status'):
                completed = issue['task_completion_status'].get('completed_count', 0)
                total = issue['task_completion_status'].get('count', 0)
                task_completion = f"{completed}/{total}"
            
            # Consolidate link data - collect all links into comma-separated strings
            links_data = issue.get('_links_data', [])
            
            link_ids = []
            link_issue_ids = []
            link_issue_iids = []
            link_types = []
            link_urls = []
            link_issue_titles = []
            linked_project_ids = []
            
            for link in links_data:
                if link.get('id'):
                    link_ids.append(str(link.get('id', '')))
                if link.get('issue_link_id'):
                    link_issue_ids.append(str(link.get('issue_link_id', '')))
                if link.get('iid'):
                    link_issue_iids.append(str(link.get('iid', '')))
                if link.get('link_type'):
                    link_types.append(str(link.get('link_type', '')))
                if link.get('web_url'):
                    link_urls.append(str(link.get('web_url', '')))
                if link.get('title'):
                    link_issue_titles.append(str(link.get('title', '')))
                if link.get('project_id'):
                    linked_project_ids.append(str(link.get('project_id', '')))
            
            # Create single row per issue with comma-separated link values
            issue_data = {
                'project': project_identifier,
                'issue_id': issue.get('id'),
                'issue_iid': issue.get('iid'),
                'title': issue.get('title'),
                'description': issue.get('description'),
                'state': issue.get('state'),
                'web_url': issue.get('web_url', ''),
                
                # Linkage columns - comma-separated
                'link_id': ', '.join(link_ids) if link_ids else None,
                'link_issue_id': ', '.join(link_issue_ids) if link_issue_ids else None,
                'link_issue_iid': ', '.join(link_issue_iids) if link_issue_iids else None,
                'link_type': ', '.join(link_types) if link_types else None,
                'link_url': ', '.join(link_urls) if link_urls else None,
                'link_issue_title': ', '.join(link_issue_titles) if link_issue_titles else None,
                'linked_project_id': ', '.join(linked_project_ids) if linked_project_ids else None,
                
                # Assignment and ownership
                'author': issue.get('author', {}).get('name'),
                'author_username': issue.get('author', {}).get('username'),
                'created_by_id': issue.get('author', {}).get('id'),
                'assignee': assignee_str,
                'assignee_ids': ', '.join([str(a.get('id', '')) for a in issue.get('assignees', [])]),
                
                # Dates
                'issue_created_date': issue.get('created_at'),
                'created_at': issue.get('created_at'),
                'updated_at': issue.get('updated_at'),
                'closed_at': issue.get('closed_at'),
                'due_date': issue.get('due_date'),
                'start_date': issue.get('start_date'),
                'current_date_time': current_datetime,
                
                # Organization
                'labels': label_str,
                'milestone': milestone,
                'iteration': iteration,
                'epic': epic,
                'epic_iid': epic_iid,
                'weight': issue.get('weight'),
                
                # Relationships
                'parent_iid': None,
                'has_tasks': issue.get('has_tasks'),
                'task_completion_status': task_completion,
                
                # Engagement
                'participants': participants_str,
                'upvotes': issue.get('upvotes'),
                'downvotes': issue.get('downvotes'),
                'user_notes_count': issue.get('user_notes_count'),
                'merge_requests_count': issue.get('merge_requests_count'),
                
                # Time tracking
                'time_estimate_hours': time_estimate / 3600 if time_estimate else None,
                'time_spent_hours': time_spent / 3600 if time_spent else None,
                
                # Metadata
                'confidential': issue.get('confidential'),
                'discussion_locked': issue.get('discussion_locked'),
                'issue_type': issue.get('issue_type'),
                'severity': issue.get('severity'),
                'health_status': issue.get('health_status'),
            }
            
            extracted_data.append(issue_data)
        
        df = pd.DataFrame(extracted_data)
        return df


class GreenplumLoader:
    """Load data into Greenplum database"""
    
    def __init__(self, host: str, port: int, database: str, user: str, schema: str):
        """Initialize Greenplum connection parameters"""
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.schema = schema
        self.password = None
        self.conn = None
        
    def get_password(self):
        """Get database password from user"""
        self.password = getpass.getpass("Enter Greenplum database password: ")
        print("✓ Database password configured")
        
    def connect(self):
        """Establish connection to Greenplum"""
        try:
            self.conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
            self.conn.autocommit = False
            print(f"✓ Connected to Greenplum database: {self.database}")
        except Exception as e:
            print(f"✗ Error connecting to database: {e}")
            raise
    
    def create_table(self, table_name: str, df: pd.DataFrame, drop_if_exists: bool = True):
        """Create table in Greenplum"""
        cursor = self.conn.cursor()
        
        try:
            # Drop table if exists and flag is set
            if drop_if_exists:
                drop_query = sql.SQL("DROP TABLE IF EXISTS {}.{}").format(
                    sql.Identifier(self.schema),
                    sql.Identifier(table_name)
                )
                cursor.execute(drop_query)
                print(f"✓ Dropped table {self.schema}.{table_name} (if existed)")
            
            # Generate CREATE TABLE statement
            columns = []
            for col, dtype in df.dtypes.items():
                if dtype == 'object':
                    col_type = 'TEXT'
                elif dtype == 'int64':
                    col_type = 'BIGINT'
                elif dtype == 'float64':
                    col_type = 'DOUBLE PRECISION'
                elif dtype == 'bool':
                    col_type = 'BOOLEAN'
                elif dtype == 'datetime64[ns]':
                    col_type = 'TIMESTAMP'
                else:
                    col_type = 'TEXT'
                
                columns.append(f"{col} {col_type}")
            
            columns_str = ', '.join(columns)
            create_query = f"""
                CREATE TABLE {self.schema}.{table_name} (
                    {columns_str}
                )
                DISTRIBUTED RANDOMLY
            """
            
            cursor.execute(create_query)
            self.conn.commit()
            print(f"✓ Created table {self.schema}.{table_name}")
            
        except Exception as e:
            self.conn.rollback()
            print(f"✗ Error creating table: {e}")
            raise
        finally:
            cursor.close()
    
    def insert_data(self, table_name: str, df: pd.DataFrame):
        """Insert data into Greenplum table"""
        cursor = self.conn.cursor()
        
        try:
            # Prepare insert statement
            columns = df.columns.tolist()
            placeholders = ', '.join(['%s'] * len(columns))
            columns_str = ', '.join(columns)
            
            insert_query = f"""
                INSERT INTO {self.schema}.{table_name} ({columns_str})
                VALUES ({placeholders})
            """
            
            # Convert DataFrame to list of tuples
            data = [tuple(x) for x in df.to_numpy()]
            
            # Execute batch insert
            cursor.executemany(insert_query, data)
            self.conn.commit()
            
            print(f"✓ Inserted {len(df)} rows into {self.schema}.{table_name}")
            
        except Exception as e:
            self.conn.rollback()
            print(f"✗ Error inserting data: {e}")
            raise
        finally:
            cursor.close()
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("✓ Database connection closed")


def create_excel_with_sheets(ikg_df: pd.DataFrame, swat_df: pd.DataFrame, filename: str):
    """Create Excel file with two worksheets"""
    print("\nCreating Excel file with multiple sheets...")
    
    wb = Workbook()
    wb.remove(wb.active)  # Remove default sheet
    
    # Create IKG sheet
    ikg_sheet = wb.create_sheet('IKG')
    for r in dataframe_to_rows(ikg_df, index=False, header=True):
        ikg_sheet.append(r)
    
    # Format IKG header
    for cell in ikg_sheet[1]:
        cell.font = Font(bold=True, color='FFFFFF')
        cell.fill = PatternFill(start_color='366092', end_color='366092', fill_type='solid')
        cell.alignment = Alignment(horizontal='center', vertical='center')
    
    # Auto-adjust column widths for IKG
    for column in ikg_sheet.columns:
        max_length = 0
        column_letter = column[0].column_letter
        for cell in column:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        adjusted_width = min(max_length + 2, 50)
        ikg_sheet.column_dimensions[column_letter].width = adjusted_width
    
    # Create SWAT sheet
    swat_sheet = wb.create_sheet('SWAT')
    for r in dataframe_to_rows(swat_df, index=False, header=True):
        swat_sheet.append(r)
    
    # Format SWAT header
    for cell in swat_sheet[1]:
        cell.font = Font(bold=True, color='FFFFFF')
        cell.fill = PatternFill(start_color='366092', end_color='366092', fill_type='solid')
        cell.alignment = Alignment(horizontal='center', vertical='center')
    
    # Auto-adjust column widths for SWAT
    for column in swat_sheet.columns:
        max_length = 0
        column_letter = column[0].column_letter
        for cell in column:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        adjusted_width = min(max_length + 2, 50)
        swat_sheet.column_dimensions[column_letter].width = adjusted_width
    
    wb.save(filename)
    print(f"✓ Excel file saved: {filename}")


def main():
    """Main execution function"""
    
    print("=" * 80)
    print("GitLab Issues Fetcher with Greenplum Integration")
    print("One row per issue_id with comma-separated link values")
    print("=" * 80)
    
    # Configuration
    GITLAB_URL = 'https://devcloud.ubs.net'
    IKG_PROJECT_PATH = 'ubs/gwma/smart-technology-and-analytics/staat-data-science/staat-ds-insights-cl/commons/staat-ds-insights-home'
    SWAT_PROJECT_PATH = 'ubs/gwma/smart-technology-and-analytics/staat-data-science/staat-ds-insights-cl/commons/staat-ds-insights-cl-home'
    
    GREENPLUM_HOST = 'greenplum-rdsp.zur.swissbank.com'
    GREENPLUM_PORT = 5432
    GREENPLUM_DB = 'gprdsp'
    GREENPLUM_USER = 'ds_rdsp_dev'
    GREENPLUM_SCHEMA = 'sandbox_prj_smart_insights'
    
    OUTPUT_TABLE1 = 'ikg_issue_details'
    OUTPUT_TABLE2 = 'swat_issue_details'
    ARCHIVE_TABLE1 = 'ikg_issue_details_archive'
    ARCHIVE_TABLE2 = 'swat_issue_details_archive'
    
    # Step 1: Authenticate with GitLab
    fetcher = GitLabIssuesFetcher(GITLAB_URL, IKG_PROJECT_PATH, SWAT_PROJECT_PATH)
    fetcher.authenticate()
    
    # Step 2: Fetch IKG issues
    ikg_project_id, ikg_project_name = fetcher.get_project_id(IKG_PROJECT_PATH)
    ikg_issues = fetcher.fetch_all_issues(ikg_project_id, ikg_project_name)
    ikg_df = fetcher.extract_issue_data(ikg_issues, 'staat-ds-insights-home')
    print(f"\n✓ IKG data extracted: {len(ikg_df)} rows (one per issue)")
    
    # Step 3: Fetch SWAT issues
    swat_project_id, swat_project_name = fetcher.get_project_id(SWAT_PROJECT_PATH)
    swat_issues = fetcher.fetch_all_issues(swat_project_id, swat_project_name)
    swat_df = fetcher.extract_issue_data(swat_issues, 'staat-ds-insights-cl-home')
    print(f"✓ SWAT data extracted: {len(swat_df)} rows (one per issue)")
    
    # Step 4: Create Excel file
    excel_filename = 'gitlab_issues_ikg_swat.xlsx'
    create_excel_with_sheets(ikg_df, swat_df, excel_filename)
    
    # Step 5: Connect to Greenplum
    print("\n" + "=" * 80)
    print("GREENPLUM DATABASE OPERATIONS")
    print("=" * 80)
    
    loader = GreenplumLoader(GREENPLUM_HOST, GREENPLUM_PORT, GREENPLUM_DB, 
                            GREENPLUM_USER, GREENPLUM_SCHEMA)
    loader.get_password()
    loader.connect()
    
    # Step 6: Create/recreate main tables (drop and create)
    print("\nCreating main tables...")
    loader.create_table(OUTPUT_TABLE1, ikg_df, drop_if_exists=True)
    loader.create_table(OUTPUT_TABLE2, swat_df, drop_if_exists=True)
    
    # Step 7: Create archive tables (only if they don't exist)
    print("\nCreating archive tables (if not exist)...")
    loader.create_table(ARCHIVE_TABLE1, ikg_df, drop_if_exists=False)
    loader.create_table(ARCHIVE_TABLE2, swat_df, drop_if_exists=False)
    
    # Step 8: Insert data into main tables
    print("\nInserting data into main tables...")
    loader.insert_data(OUTPUT_TABLE1, ikg_df)
    loader.insert_data(OUTPUT_TABLE2, swat_df)
    
    # Step 9: Insert data into archive tables
    print("\nInserting data into archive tables...")
    loader.insert_data(ARCHIVE_TABLE1, ikg_df)
    loader.insert_data(ARCHIVE_TABLE2, swat_df)
    
    # Step 10: Close connection
    loader.close()
    
    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"IKG Issues: {len(ikg_df)} rows (one per issue)")
    print(f"SWAT Issues: {len(swat_df)} rows (one per issue)")
    print(f"Excel file: {excel_filename}")
    print(f"\nGreenplum Tables Created/Updated:")
    print(f"  - {GREENPLUM_SCHEMA}.{OUTPUT_TABLE1}")
    print(f"  - {GREENPLUM_SCHEMA}.{OUTPUT_TABLE2}")
    print(f"  - {GREENPLUM_SCHEMA}.{ARCHIVE_TABLE1}")
    print(f"  - {GREENPLUM_SCHEMA}.{ARCHIVE_TABLE2}")
    print("\n✓ Process completed successfully!")
    print("\nNote: Link columns contain comma-separated values for multiple links")


if __name__ == "__main__":
    main()
