import os
import pandas as pd
import psycopg2
from psycopg2 import sql
import getpass
from datetime import datetime
from io import StringIO

# Try to import tkinter, but don't fail if not available
try:
    import tkinter as tk
    from tkinter import filedialog
    TKINTER_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    TKINTER_AVAILABLE = False
    print("Note: tkinter not available. File browser will not be available.")


def browse_file():
    """
    Open file dialog to browse and select CSV or XLSX file
    Returns filepath and filename
    """
    if not TKINTER_AVAILABLE:
        raise RuntimeError("tkinter is not available. Please use manual file path input.")
    
    try:
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        root.attributes('-topmost', True)  # Bring dialog to front
        
        # Open file dialog with filter for CSV and XLSX files
        file_path = filedialog.askopenfilename(
            title="Select a CSV or XLSX file",
            filetypes=[
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx"),
                ("Excel files (old)", "*.xls"),
                ("All supported files", "*.csv *.xlsx *.xls"),
                ("All files", "*.*")
            ]
        )
        
        root.destroy()
        
        if not file_path:
            raise ValueError("No file selected")
        
        # Validate file extension
        file_extension = file_path.lower().split('.')[-1]
        if file_extension not in ['csv', 'xlsx', 'xls']:
            raise ValueError(f"Invalid file type. Please select a CSV or XLSX file. Selected: {file_extension}")
        
        # Extract filepath and filename
        filepath = os.path.dirname(file_path)
        filename = os.path.basename(file_path)
        
        return filepath, filename
    
    except tk.TclError as e:
        if "no display" in str(e).lower() or "couldn't connect" in str(e).lower():
            raise RuntimeError("No display available. File browser requires a graphical environment. Please use manual file path input.")
        else:
            raise


def get_db_connection(config):
    """
    Create database connection
    """
    try:
        conn = psycopg2.connect(
            host=config['host'],
            port=config['port'],
            dbname=config['dbname'],
            user=config['user'],
            password=config['password']
        )
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        raise


def read_file(filepath, filename):
    """
    Read CSV or XLSX file into pandas DataFrame
    """
    full_path = os.path.join(filepath, filename)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"File not found: {full_path}")
    
    file_extension = filename.lower().split('.')[-1]
    
    try:
        if file_extension == 'csv':
            df = pd.read_csv(full_path)
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(full_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}. Use CSV or XLSX.")
        
        print(f"File loaded successfully. Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        return df
    
    except Exception as e:
        print(f"Error reading file: {e}")
        raise


def clean_column_name(col_name):
    """
    Clean column names for database compatibility
    """
    # Convert to string and lowercase
    col_name = str(col_name).lower()
    # Replace spaces and special characters with underscore
    col_name = col_name.replace(' ', '_').replace('-', '_').replace('.', '_')
    # Remove special characters
    col_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in col_name)
    # Remove multiple consecutive underscores
    while '__' in col_name:
        col_name = col_name.replace('__', '_')
    # Remove leading/trailing underscores
    col_name = col_name.strip('_')
    # Ensure it doesn't start with a number
    if col_name and col_name[0].isdigit():
        col_name = 'col_' + col_name
    
    return col_name


def infer_column_type(series):
    """
    Infer PostgreSQL/Greenplum column type from pandas series
    """
    # Check for null values
    if series.isnull().all():
        return 'TEXT'
    
    # Try to infer type from non-null values
    dtype = series.dtype
    
    if pd.api.types.is_integer_dtype(dtype):
        max_val = series.max()
        min_val = series.min()
        if min_val >= -32768 and max_val <= 32767:
            return 'SMALLINT'
        elif min_val >= -2147483648 and max_val <= 2147483647:
            return 'INTEGER'
        else:
            return 'BIGINT'
    
    elif pd.api.types.is_float_dtype(dtype):
        return 'DOUBLE PRECISION'
    
    elif pd.api.types.is_bool_dtype(dtype):
        return 'BOOLEAN'
    
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return 'TIMESTAMP'
    
    else:
        # For text, determine appropriate length
        max_length = series.astype(str).str.len().max()
        if pd.isna(max_length) or max_length == 0:
            return 'TEXT'
        elif max_length <= 255:
            return f'VARCHAR({min(int(max_length * 1.5), 255)})'
        else:
            return 'TEXT'


def drop_table(conn, schema, table_name):
    """
    Drop table if exists
    """
    try:
        cursor = conn.cursor()
        drop_query = sql.SQL("DROP TABLE IF EXISTS {}.{} CASCADE").format(
            sql.Identifier(schema),
            sql.Identifier(table_name)
        )
        cursor.execute(drop_query)
        conn.commit()
        print(f"Table {schema}.{table_name} dropped (if existed)")
        cursor.close()
    except Exception as e:
        print(f"Error dropping table: {e}")
        conn.rollback()
        raise


def create_table(conn, schema, table_name, df):
    """
    Create table based on DataFrame structure
    """
    try:
        cursor = conn.cursor()
        
        # Clean column names
        original_columns = df.columns.tolist()
        cleaned_columns = [clean_column_name(col) for col in original_columns]
        
        # Create mapping for renaming
        column_mapping = dict(zip(original_columns, cleaned_columns))
        df.rename(columns=column_mapping, inplace=True)
        
        # Generate column definitions
        column_definitions = []
        for col in df.columns:
            col_type = infer_column_type(df[col])
            column_definitions.append(f"{col} {col_type}")
        
        # Add current_date_time column
        column_definitions.append("current_date_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        
        columns_sql = ", ".join(column_definitions)
        
        create_query = f"""
            CREATE TABLE {schema}.{table_name} (
                {columns_sql}
            )
        """
        
        cursor.execute(create_query)
        conn.commit()
        print(f"Table {schema}.{table_name} created successfully")
        print(f"Columns created: {', '.join(df.columns)}, current_date_time")
        cursor.close()
        
        return df
    
    except Exception as e:
        print(f"Error creating table: {e}")
        conn.rollback()
        raise


def load_data(conn, schema, table_name, df):
    """
    Load data into Greenplum table using COPY command for better performance
    """
    try:
        cursor = conn.cursor()
        
        # Get current timestamp
        current_timestamp = datetime.now()
        
        # Add current_date_time column to DataFrame
        df['current_date_time'] = current_timestamp
        
        # Convert DataFrame to CSV string
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False, header=False, na_rep='NULL')
        csv_buffer.seek(0)
        
        # Prepare COPY command
        columns = ', '.join([clean_column_name(col) for col in df.columns])
        copy_query = f"""
            COPY {schema}.{table_name} ({columns})
            FROM STDIN
            WITH (FORMAT CSV, NULL 'NULL')
        """
        
        # Execute COPY
        cursor.copy_expert(copy_query, csv_buffer)
        conn.commit()
        
        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {schema}.{table_name}")
        row_count = cursor.fetchone()[0]
        
        print(f"Successfully loaded {row_count} rows into {schema}.{table_name}")
        cursor.close()
        
    except Exception as e:
        print(f"Error loading data: {e}")
        conn.rollback()
        raise


def upload_to_greenplum(filepath, filename, output_tablename, db_config):
    """
    Main function to upload file to Greenplum
    
    Parameters:
    -----------
    filepath : str
        Path to the directory containing the file
    filename : str
        Name of the file (CSV or XLSX)
    output_tablename : str
        Name of the output table in Greenplum
    db_config : dict
        Database configuration dictionary
    """
    conn = None
    
    try:
        # Step 1: Read the file
        print(f"\n{'='*60}")
        print(f"Starting upload process for: {filename}")
        print(f"{'='*60}\n")
        
        df = read_file(filepath, filename)
        
        # Step 2: Connect to database
        print("\nConnecting to Greenplum database...")
        conn = get_db_connection(db_config)
        print("Connected successfully!")
        
        schema = db_config['schema']
        
        # Step 3: Drop existing table
        print(f"\nDropping table if exists: {schema}.{output_tablename}")
        drop_table(conn, schema, output_tablename)
        
        # Step 4: Create new table
        print(f"\nCreating table: {schema}.{output_tablename}")
        df = create_table(conn, schema, output_tablename, df)
        
        # Step 5: Load data
        print(f"\nLoading data into {schema}.{output_tablename}...")
        load_data(conn, schema, output_tablename, df)
        
        print(f"\n{'='*60}")
        print("Upload completed successfully!")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\n{'!'*60}")
        print(f"Upload failed: {e}")
        print(f"{'!'*60}\n")
        raise
    
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")


# Main execution
if __name__ == "__main__":
    print("\n" + "="*60)
    print("GREENPLUM FILE UPLOADER")
    print("="*60 + "\n")
    
    # Check if file browser is available
    if TKINTER_AVAILABLE:
        use_browser = input("Do you want to browse for file? (yes/no) [default: no]: ").strip().lower()
    else:
        print("Note: File browser is not available in this environment.")
        use_browser = 'no'
    
    if use_browser in ['yes', 'y'] and TKINTER_AVAILABLE:
        print("\nOpening file browser...")
        try:
            filepath, filename = browse_file()
            print(f"\nSelected file: {filename}")
            print(f"File path: {filepath}")
        except RuntimeError as e:
            print(f"\n{e}")
            print("Falling back to manual input...\n")
            use_browser = 'no'
        except Exception as e:
            print(f"Error: {e}")
            print("Falling back to manual input...\n")
            use_browser = 'no'
    
    if use_browser not in ['yes', 'y'] or not TKINTER_AVAILABLE:
        # Manual input
        filepath = input("Enter file path (or full path with filename): ").strip()
        
        # Check if user provided full path or just directory
        if os.path.isfile(filepath):
            # Full path provided
            full_path = filepath
            filepath = os.path.dirname(full_path)
            filename = os.path.basename(full_path)
        else:
            # Directory provided, ask for filename
            filename = input("Enter file name (CSV or XLSX): ").strip()
        
        # Validate file extension
        file_extension = filename.lower().split('.')[-1]
        if file_extension not in ['csv', 'xlsx', 'xls']:
            print(f"Error: Invalid file type '{file_extension}'. Please use CSV or XLSX files.")
            exit(1)
        
        # Verify file exists
        full_file_path = os.path.join(filepath, filename) if filepath else filename
        if not os.path.exists(full_file_path):
            print(f"Error: File not found: {full_file_path}")
            exit(1)
    
    # Get output table name
    output_tablename = input("\nEnter output table name: ").strip()
    
    # Get password securely
    password = getpass.getpass(f"\nEnter Password for DB User: ")
    
    # Database configuration
    DB_CONFIG = {
        'host': 'greenplum-rdsp.zur.swissbank.com',
        'port': '5432',
        'dbname': 'gprdsp',
        'user': 'ds_rdsp_dev',
        'password': password,
        'schema': 'sandbox_prj_smart_insights'
    }
    
    # Execute upload
    upload_to_greenplum(filepath, filename, output_tablename, DB_CONFIG)
