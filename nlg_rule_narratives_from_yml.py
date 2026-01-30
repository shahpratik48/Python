#!/usr/bin/env python3
"""
NLG Rules Parser
Connects to GitLab, parses YAML/JSON rule files, and outputs to Excel and Greenplum.
"""

import re
import json
import yaml
import getpass
import psycopg2
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Set
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, Alignment
import gitlab

# --------------------------------------------------------------------------------------
# GitLab configuration
# --------------------------------------------------------------------------------------
GITLAB_URL = "https://devcloud.ubs.net"
NLG_PROJECT_PATH = (
    "ubs/gwma/smart-technology-and-analytics/staat-data-science/"
    "staat-ds-genesis/genesis-platform/nlg-dags"
)
BRANCH = "develop"
PATH_RULES = "dags/nlg/src/rules"

# --------------------------------------------------------------------------------------
# Output configuration
# --------------------------------------------------------------------------------------
OUTPUT_TEMPLATE = "nlg_rules_narratives_{timestamp}.xlsx"
TIMESTAMP_FMT = "%Y-%m-%d_%H%M%S"

# --------------------------------------------------------------------------------------
# Database configuration
# --------------------------------------------------------------------------------------
DB_CONFIG = {
    "host": "greenplum-rdsp.zur.swissbank.com",
    "port": 5432,
    "dbname": "gprdsp",
    "user": "ds_rdsp_dev",
    "schema": "sandbox_prj_smart_insights",
    "table": "nlg_rules_narratives_auto_refresh",
    "owner": "erd_gpdb_prj_smart_insights",
    "read_role": "erd_gpdb_prj_smart_insights_ro",
}

# --------------------------------------------------------------------------------------
# Constants/utilities
# --------------------------------------------------------------------------------------
RULE_FILE_EXTS = {".yaml", ".yml", ".json"}
RULE_TAG_PATTERN = re.compile(r"\{([^{}]+)\}")

# Rule tags to parse
RULE_TAGS = {
    "rule_narrative",
    "rule_narrative_title",
    "rule_narrative_single",
    "rule_narrative_footer",
    "rule_narrative_item",
    "disclaimer",
}


def extract_rule_tags_from_value(value: Any) -> Set[str]:
    """Extract rule tag values like {account_name1} from text."""
    if isinstance(value, str):
        return set(RULE_TAG_PATTERN.findall(value))
    return set()


def parse_yaml_json_content(content: str, file_ext: str) -> Dict[str, Any]:
    """Parse YAML or JSON content."""
    try:
        if file_ext in {".yaml", ".yml"}:
            return yaml.safe_load(content) or {}
        elif file_ext == ".json":
            return json.loads(content) or {}
    except Exception as e:
        print(f"Error parsing content: {e}")
        return {}
    return {}


def extract_rules_recursive(
    data: Any,
    target_type: str,
    insight_type: str,
    filepath: str,
    filename: str,
    current_timestamp: str,
    records: List[Dict],
    parent_key: str = ""
):
    """Recursively extract rule tags and values from nested structures."""
    if isinstance(data, dict):
        for key, value in data.items():
            current_key = f"{parent_key}.{key}" if parent_key else key
            
            # Check if this key is a rule tag we're interested in
            if key in RULE_TAGS:
                # Extract rule tag values from the value
                rule_tag_values = extract_rule_tags_from_value(value)
                
                # Convert value to string for storage
                if isinstance(value, (list, dict)):
                    rule_value = json.dumps(value) if isinstance(value, dict) else "\n".join(str(v) for v in value)
                else:
                    rule_value = str(value) if value is not None else ""
                
                # Create record for each rule_tag_value found
                if rule_tag_values:
                    for tag_value in rule_tag_values:
                        records.append({
                            "target_type": target_type,
                            "insight_type": insight_type,
                            "rule_tag": key,
                            "rule_tag_value": tag_value,
                            "rule_value": rule_value,
                            "filepath": filepath,
                            "filename": filename,
                            "current_timestamp": current_timestamp,
                        })
                else:
                    # No tag values found, but still record the rule tag
                    records.append({
                        "target_type": target_type,
                        "insight_type": insight_type,
                        "rule_tag": key,
                        "rule_tag_value": "",
                        "rule_value": rule_value,
                        "filepath": filepath,
                        "filename": filename,
                        "current_timestamp": current_timestamp,
                    })
            
            # Also check if value contains any of our rule tags
            if isinstance(value, str):
                tag_values = extract_rule_tags_from_value(value)
                if tag_values and key not in RULE_TAGS:
                    # This is a key-value pair where key is a tag and value contains {tag_value}
                    for tag_value in tag_values:
                        records.append({
                            "target_type": target_type,
                            "insight_type": insight_type,
                            "rule_tag": key,
                            "rule_tag_value": tag_value,
                            "rule_value": value,
                            "filepath": filepath,
                            "filename": filename,
                            "current_timestamp": current_timestamp,
                        })
            
            # Recurse into nested structures
            extract_rules_recursive(
                value, target_type, insight_type, filepath, filename,
                current_timestamp, records, current_key
            )
    
    elif isinstance(data, list):
        for item in data:
            extract_rules_recursive(
                item, target_type, insight_type, filepath, filename,
                current_timestamp, records, parent_key
            )


def get_gitlab_files(gl_project, branch: str, base_path: str) -> List[Dict]:
    """Recursively get all YAML/JSON files from GitLab repository."""
    files_data = []
    
    def traverse_directory(path: str, target_type: str = ""):
        try:
            items = gl_project.repository_tree(path=path, ref=branch, all=True)
            
            for item in items:
                item_path = item['path']
                item_name = item['name']
                
                if item['type'] == 'tree':
                    # This is a directory
                    # Determine target_type from folder structure
                    if path == base_path:
                        # Direct subfolder of rules = target_type
                        new_target_type = item_name
                    else:
                        new_target_type = target_type
                    
                    traverse_directory(item_path, new_target_type)
                
                elif item['type'] == 'blob':
                    # This is a file
                    file_ext = Path(item_name).suffix.lower()
                    
                    if file_ext in RULE_FILE_EXTS:
                        # Extract insight_type from filename (remove extension)
                        insight_type = Path(item_name).stem
                        
                        files_data.append({
                            'path': item_path,
                            'name': item_name,
                            'target_type': target_type,
                            'insight_type': insight_type,
                            'extension': file_ext,
                        })
        
        except gitlab.exceptions.GitlabGetError as e:
            print(f"Error accessing {path}: {e}")
    
    traverse_directory(base_path)
    return files_data


def main():
    """Main execution function."""
    print("=" * 80)
    print("NLG Rules Parser")
    print("=" * 80)
    
    # Get GitLab token
    private_token = getpass.getpass("Enter the git token: ")
    
    # Generate timestamp
    current_timestamp = datetime.now().strftime(TIMESTAMP_FMT)
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Output filename
    output_filename = OUTPUT_TEMPLATE.format(timestamp=current_timestamp)
    output_path = f"/home/claude/{output_filename}"
    
    print(f"\n[1/5] Connecting to GitLab: {GITLAB_URL}")
    try:
        gl = gitlab.Gitlab(GITLAB_URL, private_token=private_token)
        gl.auth()
        print("✓ Connected to GitLab successfully")
    except Exception as e:
        print(f"✗ Failed to connect to GitLab: {e}")
        return
    
    print(f"\n[2/5] Accessing project: {NLG_PROJECT_PATH}")
    try:
        project = gl.projects.get(NLG_PROJECT_PATH)
        print(f"✓ Project accessed: {project.name}")
    except Exception as e:
        print(f"✗ Failed to access project: {e}")
        return
    
    print(f"\n[3/5] Discovering files in: {PATH_RULES} (branch: {BRANCH})")
    files = get_gitlab_files(project, BRANCH, PATH_RULES)
    print(f"✓ Found {len(files)} YAML/JSON files")
    
    # Parse all files and extract rules
    print(f"\n[4/5] Parsing files and extracting rules...")
    all_records = []
    
    for idx, file_info in enumerate(files, 1):
        try:
            file_content = project.files.get(file_path=file_info['path'], ref=BRANCH)
            content = file_content.decode().decode('utf-8')
            
            parsed_data = parse_yaml_json_content(content, file_info['extension'])
            
            extract_rules_recursive(
                data=parsed_data,
                target_type=file_info['target_type'],
                insight_type=file_info['insight_type'],
                filepath=file_info['path'],
                filename=file_info['name'],
                current_timestamp=current_datetime,
                records=all_records
            )
            
            if idx % 10 == 0:
                print(f"  Processed {idx}/{len(files)} files...")
        
        except Exception as e:
            print(f"  ✗ Error processing {file_info['path']}: {e}")
    
    print(f"✓ Extracted {len(all_records)} rule records")
    
    # Create DataFrame
    df = pd.DataFrame(all_records)
    
    # Reorder columns
    column_order = [
        "target_type",
        "insight_type",
        "rule_tag",
        "rule_tag_value",
        "rule_value",
        "filepath",
        "filename",
        "current_timestamp",
    ]
    df = df[column_order]
    
    # Create Excel file
    print(f"\n[5/5] Creating Excel file: {output_filename}")
    df.to_excel(output_path, index=False, sheet_name="NLG Rules")
    
    # Format Excel
    wb = load_workbook(output_path)
    ws = wb.active
    
    # Format headers
    header_font = Font(bold=True, size=11, name='Arial')
    header_alignment = Alignment(horizontal='center', vertical='center')
    
    for cell in ws[1]:
        cell.font = header_font
        cell.alignment = header_alignment
    
    # Auto-adjust column widths
    for column in ws.columns:
        max_length = 0
        column_letter = column[0].column_letter
        for cell in column:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        adjusted_width = min(max_length + 2, 50)
        ws.column_dimensions[column_letter].width = adjusted_width
    
    wb.save(output_path)
    print(f"✓ Excel file created: {output_path}")
    
    # Database operations
    print(f"\n[6/6] Uploading to Greenplum database...")
    try:
        conn = psycopg2.connect(
            host=DB_CONFIG["host"],
            port=DB_CONFIG["port"],
            dbname=DB_CONFIG["dbname"],
            user=DB_CONFIG["user"],
            password=getpass.getpass("Enter Greenplum password: ")
        )
        cursor = conn.cursor()
        
        schema = DB_CONFIG["schema"]
        table = DB_CONFIG["table"]
        owner = DB_CONFIG["owner"]
        read_role = DB_CONFIG["read_role"]
        
        # Drop existing table
        print(f"  Dropping table {schema}.{table} if exists...")
        cursor.execute(f"DROP TABLE IF EXISTS {schema}.{table}")
        
        # Create new table
        print(f"  Creating table {schema}.{table}...")
        create_table_sql = f"""
        CREATE TABLE {schema}.{table} (
            target_type TEXT,
            insight_type TEXT,
            rule_tag TEXT,
            rule_tag_value TEXT,
            rule_value TEXT,
            filepath TEXT,
            filename TEXT,
            current_timestamp TIMESTAMP
        ) DISTRIBUTED RANDOMLY
        """
        cursor.execute(create_table_sql)
        
        # Insert data
        print(f"  Inserting {len(df)} records...")
        for _, row in df.iterrows():
            insert_sql = f"""
            INSERT INTO {schema}.{table} 
            (target_type, insight_type, rule_tag, rule_tag_value, rule_value, 
             filepath, filename, current_timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(insert_sql, tuple(row))
        
        # Set owner and permissions
        print(f"  Setting permissions...")
        cursor.execute(f"ALTER TABLE {schema}.{table} OWNER TO {owner}")
        cursor.execute(f"GRANT SELECT ON {schema}.{table} TO {read_role}")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"✓ Data uploaded to {schema}.{table}")
    
    except Exception as e:
        print(f"✗ Database error: {e}")
        if 'conn' in locals():
            conn.rollback()
    
    print("\n" + "=" * 80)
    print("Processing complete!")
    print(f"Excel file: {output_path}")
    print(f"Database table: {DB_CONFIG['schema']}.{DB_CONFIG['table']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
