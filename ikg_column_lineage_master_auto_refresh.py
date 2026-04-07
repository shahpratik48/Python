"""
IKG Column Lineage Master Auto Refresh
=======================================
Parses all SQL files from the IKG GitLab repository (develop branch),
extracts column-level lineage, and writes results to:
  - Greenplum: <schema>.ikg_column_lineage_master_auto_refresh  (DROP/RECREATE)
  - Excel:     ikg_column_lineage_master_auto_refresh_<YYYYMMDD_HHMMSS>.xlsx
"""

import os
import re
import getpass
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Set

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GITLAB_URL         = "https://devcloud.ubs.net"
IKG_PROJECT_PATH   = "ubs/gwma/smart-technology-and-analytics/staat-data-science/staat-ds-genesis/genesis-platform/ikg-dags"
DEFAULT_BASE_BRANCH = "develop"
SQL_FOLDER_IN_REPO  = "dags/ikg/scripts/sql"

GREENPLUM_HOST = "greenplum-rdsp.zur.swissbank.com"
GREENPLUM_PORT = 5432
GREENPLUM_DB   = "gprdsp"
GREENPLUM_USER = "ds_rdsp_dev"
GREENPLUM_SCHEMA: Optional[str] = None

OUTPUT_TABLE     = "ikg_column_lineage_master_auto_refresh"
GITLAB_TOKEN_VAR = "GENESIS_DDLC_IKG_GIT_SECRET"
POSTGRES_CONN_ID_VAR = "IKG_POSTGRES_CONN_ID"

# Folders to skip entirely (req 18)
EXCLUDED_PROCESSES = {
    "current", "ikg_pre_validation_new", "ikg_create_profiles_history",
    "ikg_postvalidation", "ikg_drop_backup_tables", "ikg_datasets_validation",
}

# Cache for information_schema column lookups: {(schema, table): [col1, col2, ...]}
_INFO_SCHEMA_CACHE: Dict[Tuple[str, str], List[str]] = {}
# Global DB engine for information_schema lookups (set during run)
_DB_ENGINE = None

# ── Schema parameter → actual schema name mapping ─────────────────────────────
# These are the local defaults. When running in Airflow the actual values come
# from Airflow Variables / props. Override below for local runs.
JINJA_SCHEMA_MAP: Dict[str, str] = {
    "IKG_SCHEMA":           "core_ikg",
    "EDW_INPUT_SCHEMA":     "core_wma_shared",
    "EDW_VIEW_INPUT_SCHEMA":"core_wma_shared",
    "IKG_VENDOR_SCHEMA":    "core_wma_shared",
    "MODEL_SCHEMA":         "core_model",
    "NLG_SCHEMA":           "core_nlg",
    "IKG_WEALTHX_SCHEMA":   "sandbox_prj_smart_relationship",
}

def _resolve_schema(jinja_param_name: str) -> str:
    """
    Resolve a Jinja parameter name like 'IKG_SCHEMA' to an actual schema.
    In Airflow, attempts to read from Airflow Variables/props first.
    Falls back to JINJA_SCHEMA_MAP for local runs.
    """
    if is_running_in_airflow():
        try:
            from ikg.scripts.python import props as ikg_props  # type: ignore
            airflow_vars = ikg_props.get_airflow_variables()
            val = airflow_vars.get(jinja_param_name)
            if val:
                return val
        except Exception:
            pass
        try:
            if Variable is not None:
                val = Variable.get(jinja_param_name, default_var=None)
                if val:
                    return val
        except Exception:
            pass
    return JINJA_SCHEMA_MAP.get(jinja_param_name, jinja_param_name)


def _resolve_schema_label(schema_label: str) -> str:
    """
    Convert a schema label (either a real schema name or a Jinja param name
    like 'IKG_SCHEMA') to an actual schema name for information_schema queries.
    """
    if not schema_label:
        return ''
    # Already a real schema name (contains no uppercase-only identifier pattern)
    if schema_label in JINJA_SCHEMA_MAP:
        return _resolve_schema(schema_label)
    # Try stripping {{params.X}} to get X
    jm = re.search(r'params\.([^}]+)', schema_label)
    if jm:
        return _resolve_schema(jm.group(1).strip())
    return schema_label

COLUMN_ORDER = [
    "file", "path", "target_table", "target_schema", "process",
    "current_date_time", "sub_target_table", "sub_target_schema",
    "target_column", "source_table", "source_schema", "source_column",
    "logic", "sql_process",
]

# SQL reserved words that can NEVER be column / table names
SQL_KEYWORDS: Set[str] = {
    "SELECT","FROM","WHERE","AND","OR","NOT","IN","IS","NULL","CASE","WHEN",
    "THEN","ELSE","END","AS","ON","JOIN","LEFT","RIGHT","INNER","OUTER","FULL",
    "CROSS","GROUP","BY","ORDER","HAVING","LIMIT","OFFSET","DISTINCT","ALL",
    "UNION","INTERSECT","EXCEPT","WITH","RECURSIVE","INSERT","INTO","UPDATE",
    "SET","DELETE","CREATE","TABLE","TEMP","TEMPORARY","DROP","IF","EXISTS",
    "PARTITION","OVER","TRUE","FALSE","LIKE","ILIKE","BETWEEN","ASC","DESC",
    "DISTRIBUTED","GRANT","ALTER","OWNER","TO","INDEX","BIGINT","INTEGER",
    "VARCHAR","CHARACTER","VARYING","NUMERIC","DOUBLE","PRECISION","BOOLEAN",
    "TEXT","SERIAL","APPENDONLY","COMPRESSLEVEL","USING","VALUES","RETURNING",
    "FILTER","NULLS","LAST","FIRST","ROWS","RANGE","PRECEDING","FOLLOWING",
    "UNBOUNDED","CURRENT","ROW","WINDOW","WITHIN","GROUPS","EXCLUDE","TIES",
    "NATURAL","LATERAL","STRAIGHT","NO","CYCLE","RECURSIVE","MATERIALIZED",
    "VIEW","TRUNCATE","VACUUM","ANALYZE","REINDEX","REFRESH","EXPLAIN",
    "PERFORM","RAISE","NOTICE","EXCEPTION","BEGIN","COMMIT","ROLLBACK",
    "SAVEPOINT","RELEASE","PRIMARY","KEY","UNIQUE","REFERENCES","CHECK",
    "DEFAULT","GENERATED","ALWAYS","STORED","VIRTUAL","IDENTITY","SEQUENCE",
    "OWNED","INCREMENT","START","MINVALUE","MAXVALUE","CACHE","INHERIT",
    "TABLESPACE","FILLFACTOR","RELOPTIONS","WITHOUT","OIDS","INHERITS",
    "INCLUDING","EXCLUDING","DEFAULTS","INDEXES","CONSTRAINTS","STORAGE",
    "COMMENTS","STATISTICS","NOTHING","COLUMN","TYPE","RENAME","ADD","DISABLE",
    "ENABLE","REPLICA","TRIGGER","RULE","PROCEDURE","FUNCTION","AGGREGATE",
    "OPERATOR","CLASS","FAMILY","DOMAIN","CONVERSION","ENCODING","SCHEMA",
    "DATABASE","ROLE","USER","PUBLICATION","SUBSCRIPTION","SERVER","EXTENSION",
    "POLICY","TRANSFORM","CAST","FOREIGN","DATA","WRAPPER","IMPORT","EXPORT",
    "COPY","LOCK","NOTIFY","LISTEN","UNLISTEN","LOAD","RESET","SHOW","DISCARD",
    "DEALLOCATE","FETCH","MOVE","CLOSE","DECLARE","OPEN","RETURN","NEXT",
    "PRIOR","ABSOLUTE","RELATIVE","FORWARD","BACKWARD","SCROLL","HOLD",
    "BINARY","INSENSITIVE","ASENSITIVE","SENSITIVE","READ","WRITE",
    "REPEATABLE","UNCOMMITTED","COMMITTED","SERIALIZABLE","SNAPSHOT",
    "INTERVAL","EPOCH","TIMEZONE","DOW","DOY","HOUR","MINUTE","SECOND",
    "MILLISECOND","MICROSECOND","QUARTER","WEEK","MONTH","YEAR","DECADE",
    "CENTURY","MILLENNIUM","INFINITY","NAN","DEFERRABLE","INITIALLY",
    "DEFERRED","IMMEDIATE","CONSTRAINT","AT","TIME","ZONE","SIMILAR","ESCAPE",
    "COLLATE","ARRAY","SPECIFIC","CALLED","INPUT","LANGUAGE","IMMUTABLE",
    "STABLE","VOLATILE","STRICT","SECURITY","DEFINER","INVOKER","LOOP",
    "EXECUTE","BOTH","LEADING","TRAILING","OVERLAY","PLACING","POSITION",
    "FOR","ONLY","ENUM","RANGE","MULTIRANGE","GENERATED","IMPLICIT","EXPLICIT",
}

# SQL built-in functions — these appear in expressions but are NOT columns
SQL_FUNCTIONS: Set[str] = {
    "COALESCE","NULLIF","GREATEST","LEAST","NVL","IIF",
    "COUNT","SUM","AVG","AVERAGE","MAX","MIN","STDDEV","VARIANCE",
    "ROW_NUMBER","RANK","DENSE_RANK","PERCENT_RANK","CUME_DIST","NTILE",
    "LAG","LEAD","FIRST_VALUE","LAST_VALUE","NTH_VALUE",
    "CAST","CONVERT","TRY_CAST",
    "EXTRACT","DATE_PART","DATE_TRUNC","NOW","CURRENT_DATE",
    "CURRENT_TIMESTAMP","CURRENT_TIME","LOCALTIMESTAMP","LOCALTIME",
    "TO_DATE","TO_TIMESTAMP","TO_CHAR","TO_NUMBER","TO_HEX",
    "DATEADD","DATEDIFF","MONTHS_BETWEEN","ADD_MONTHS","NEXT_DAY","LAST_DAY",
    "AGE","MAKE_DATE","MAKE_INTERVAL","MAKE_TIME","MAKE_TIMESTAMP",
    "ISFINITE","CLOCK_TIMESTAMP","STATEMENT_TIMESTAMP","TRANSACTION_TIMESTAMP",
    "UPPER","LOWER","INITCAP","LENGTH","CHAR_LENGTH","OCTET_LENGTH",
    "TRIM","LTRIM","RTRIM","BTRIM","LPAD","RPAD",
    "SUBSTRING","SUBSTR","LEFT","RIGHT","MID",
    "REPLACE","REGEXP_REPLACE","REGEXP_MATCH","REGEXP_MATCHES","REGEXP_SPLIT_TO_ARRAY",
    "CONCAT","CONCAT_WS","FORMAT","SPLIT_PART","STRING_TO_ARRAY",
    "ARRAY_TO_STRING","STRING_AGG","ARRAY_AGG","LISTAGG","ARRAY_LENGTH",
    "CARDINALITY","ARRAY_UPPER","ARRAY_LOWER",
    "POSITION","STRPOS","CHARINDEX",
    "REVERSE","REPEAT","QUOTE_LITERAL","QUOTE_IDENT","CHR","ASCII",
    "MD5","SHA256","ENCODE","DECODE",
    "ROUND","FLOOR","CEIL","CEILING","TRUNC","ABS","SIGN","MOD","POWER",
    "SQRT","LOG","LN","EXP","RANDOM","SETSEED","DIV",
    "GREATEST","LEAST",
    "UNNEST","GENERATE_SERIES","GENERATE_SUBSCRIPTS",
    "JSONB_AGG","JSON_AGG","JSONB_BUILD_OBJECT","JSON_BUILD_OBJECT",
    "JSONB_OBJECT_KEYS","JSON_OBJECT_KEYS","JSONB_EACH","JSON_EACH",
    "JSONB_EXTRACT_PATH","JSON_EXTRACT_PATH","JSONB_ARRAY_ELEMENTS",
    "ROW","COMPOSITE",
    "LEVENSHTEIN","METAPHONE","SOUNDEX","SIMILARITY","WORD_SIMILARITY",
    "DIFFERENCE","DMETAPHONE","DMETAPHONE_ALT",
    "BOOL_AND","BOOL_OR","EVERY","BIT_AND","BIT_OR","XOR",
    "PERCENTILE_CONT","PERCENTILE_DISC","MODE",
    "REGR_SLOPE","REGR_INTERCEPT","REGR_R2","CORR","COVAR_POP","COVAR_SAMP",
    "ST_ASTEXT","ST_GEOMFROMTEXT","ST_DISTANCE","ST_CONTAINS",
    "ISNULL","IFNULL","ZAP","IFFULL",
    # JSON functions
    "ROW_TO_JSON","TO_JSON","JSON_BUILD_ARRAY","JSONB_BUILD_ARRAY",
    "JSON_TYPEOF","JSONB_TYPEOF","JSON_STRIP_NULLS","JSONB_STRIP_NULLS",
    "JSON_POPULATE_RECORD","JSONB_POPULATE_RECORD","JSON_TO_RECORD","JSONB_TO_RECORD",
    # Misc
    "MD5","SHA256","ENCODE","DECODE","PG_TYPEOF","OID",
    "GENERATE_SERIES","GENERATE_SUBSCRIPTS","UNNEST",
    "WIDTH_BUCKET","SETSEED","RANDOM",
}


# ===========================================================================
#  PRE-COMPILED REGEX — compiled once at import time for speed
# ===========================================================================
_RC_BLOCK_CMT   = re.compile(r'/\*.*?\*/', re.DOTALL)
_RC_LINE_CMT    = re.compile(r'--[^\n]*')
_RC_WS          = re.compile(r'[ \t]+')
_RC_JINJA       = re.compile(r'\{\{[^}]+\}\}')
_RC_JINJA_SCH   = re.compile(r'(\{\{[^}]+\}\})\.(.*)')
_RC_CAST        = re.compile(r'::\s*\w[\w\s,()]*$')
_RC_SELECT      = re.compile(r'\bSELECT\b\s*', re.IGNORECASE)
_RC_DISTINCT_SK = re.compile(r'(?:DISTINCT\s*(?:ON\s*\((?:[^()]*|\([^()]*\))*\)\s*)?|ALL\s+)', re.IGNORECASE)
_RC_DISTINCT_ON = re.compile(r'^DISTINCT\s+ON\s*\(', re.IGNORECASE)
_RC_FROM_KW     = re.compile(r'\bFROM\b', re.IGNORECASE)
_RC_AS_KW       = re.compile(r'\bAS\b', re.IGNORECASE)
_RC_OVER        = re.compile(r'\bOVER\s*\(', re.IGNORECASE)
_RC_FILTER_W    = re.compile(r'\bFILTER\s*\(\s*WHERE\s+', re.IGNORECASE)
_RC_FILTER      = re.compile(r'\bFILTER\s*\(', re.IGNORECASE)
_RC_PART_BY     = re.compile(r'\bPARTITION\s+BY\b', re.IGNORECASE)
_RC_ORDER_BY    = re.compile(r'\bORDER\s+BY\b', re.IGNORECASE)
_RC_PART_STOP   = re.compile(r'\b(PARTITION\s+BY|ORDER\s+BY|ROWS|RANGE|GROUPS|EXCLUDE|FILTER)\b', re.IGNORECASE)
_RC_WHERE       = re.compile(r'\bWHERE\b', re.IGNORECASE)
_RC_HAVING      = re.compile(r'\bHAVING\b', re.IGNORECASE)
_RC_WITH        = re.compile(r'\bWITH\b\s*', re.IGNORECASE)
_RC_CTE_HEAD    = re.compile(r'(\w+)\s+AS\s*\(', re.IGNORECASE)
_RC_TBLREF      = re.compile(
    r'\b(FROM|JOIN)\s+((?:\{\{[^}]+\}\}|\w+)(?:\.(?:\{\{[^}]+\}\}|\w+))?)'
    r'(?:\s+(?:AS\s+)?(?!(LEFT|RIGHT|FULL|INNER|CROSS|OUTER|JOIN|WHERE|ON|GROUP|ORDER|HAVING|LIMIT|SET|DISTRIBUTED|UNION|INTERSECT|EXCEPT)\b)(\w+)(?!\s*\())?',
    re.IGNORECASE)
_RC_JOIN_FULL   = re.compile(
    r'((?:LEFT|RIGHT|FULL|INNER|CROSS)?\s*(?:OUTER\s+)?JOIN)\s+'
    r'((?:\{\{[^}]+\}\}|\w+)(?:\.(?:\{\{[^}]+\}\}|\w+))?)'
    r'(?:\s+(?:AS\s+)?(\w+))?'
    r'(?:\s+ON\s+(.*?))?'
    r'(?=\s*(?:LEFT|RIGHT|FULL|INNER|CROSS|WHERE|GROUP|HAVING|ORDER|LIMIT|DISTRIBUTED|UNION|;|$))',
    re.IGNORECASE | re.DOTALL)
_RC_ALIAS_COL   = re.compile(r'\b(\w+)\s*\.\s*("(?:[^"]+)"|\w+)')
_RC_ALIAS_DOT   = re.compile(r'\b(\w+)\.(\w+)\b')
_RC_WORDS       = re.compile(r'\b([a-zA-Z_]\w*)\b')
_RC_NUMS        = re.compile(r'\b\d+\.?\d*\b')
_RC_SQ          = re.compile(r"'[^']*'")
_RC_DQ          = re.compile(r'"[^"]*"')
_RC_AS_TRAIL    = re.compile(r'\bAS\s+', re.IGNORECASE)
_RC_COUNT_STAR  = re.compile(r'^(COUNT\s*\(\s*(?:DISTINCT\s+)?\*?\s*\))', re.IGNORECASE)
_RC_CREATE_CTA  = re.compile(
    r'CREATE\s+(?:TEMP(?:ORARY)?\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?'
    r'((?:\{\{[^}]+\}\}|\w+)(?:\.(?:\w+|\{\{[^}]+\}\}))?)'
    r'\s+AS\s*', re.IGNORECASE)
_RC_CREATE_TEMP = re.compile(r'\bTEMP(?:ORARY)?\b', re.IGNORECASE)
_RC_INSERT      = re.compile(
    r'INSERT\s+INTO\s+((?:\{\{[^}]+\}\}|\w+)(?:\.(?:\w+|\{\{[^}]+\}\}))?)',
    re.IGNORECASE)
_RC_SUBQ_ALIAS  = re.compile(r'^\((.+)\)\s*(?:AS\s+)?(\w+)\s*$', re.IGNORECASE | re.DOTALL)
_RC_FROM_SUBQ   = re.compile(
    r'\bFROM\s*\((.+?)\)\s*(?:AS\s+)?(\w+)\s*(?:;|$|\bWHERE\b|\bJOIN\b)',
    re.IGNORECASE | re.DOTALL)
_RC_STAR        = re.compile(r'^(\w+\.)?\*$')
_RC_STAR_PFX    = re.compile(r'^(\w+)\.\*$')
_RC_DQ_TRAIL    = re.compile(r'\s("(?:[^"]+)")\s*$')
_RC_BARE        = re.compile(r'^(\w+)(?:::\w+)?$')
_RC_DOTCOL      = re.compile(r'^(\w+)\.(\w+)(?:::\w+)?$')
_RC_ALIAS_DQ    = re.compile(r'^(\w+)\."([^"]+)"(?:::\w+)?$')
_RC_ALIAS_SONLY = re.compile(r'^\w+\.\w+(?:::\w[\w\s,()]*)?$')
_RC_ALIAS_DQONLY= re.compile(r'^\w+\."[^"]+"(?:::\w+)?$')

# ---------------------------------------------------------------------------
# Airflow helpers
# ---------------------------------------------------------------------------
try:
    from airflow.models import Variable  # type: ignore
    _HAS_AIRFLOW = True
except Exception:
    Variable = None  # type: ignore
    _HAS_AIRFLOW = False


def is_running_in_airflow() -> bool:
    return _HAS_AIRFLOW and bool(os.environ.get("AIRFLOW_CTX_DAG_ID"))


def get_private_token(allow_prompt: bool = True) -> str:
    if is_running_in_airflow():
        return Variable.get(GITLAB_TOKEN_VAR)
    token = os.environ.get(GITLAB_TOKEN_VAR)
    if token:
        return token
    if allow_prompt:
        return getpass.getpass("Enter your GitLab private token: ")
    raise RuntimeError("GitLab token not found.")


def ensure_greenplum_schema() -> str:
    global GREENPLUM_SCHEMA
    if GREENPLUM_SCHEMA:
        return GREENPLUM_SCHEMA
    if is_running_in_airflow():
        try:
            from ikg.scripts.python import props as ikg_props  # type: ignore
            GREENPLUM_SCHEMA = ikg_props.get_airflow_variables().get("IKG_SCHEMA", "sandbox_prj_smart_insights")
        except Exception:
            GREENPLUM_SCHEMA = "sandbox_prj_smart_insights"
    else:
        GREENPLUM_SCHEMA = "sandbox_prj_smart_insights"
    return GREENPLUM_SCHEMA


def get_greenplum_credentials() -> Optional[dict]:
    if not is_running_in_airflow():
        return None
    try:
        from airflow.providers.postgres.hooks.postgres import PostgresHook  # type: ignore
        conn_id = Variable.get(POSTGRES_CONN_ID_VAR)
        hook = PostgresHook(postgres_conn_id=conn_id)
        conn = hook.get_connection(conn_id)
        return {"host": conn.host, "port": int(conn.port or GREENPLUM_PORT),
                "db": conn.schema or GREENPLUM_DB,
                "user": conn.login or GREENPLUM_USER, "password": conn.password}
    except Exception as e:
        logger.warning(f"Could not resolve Greenplum credentials: {e}")
        return None


# ===========================================================================
#  TEXT UTILITIES
# ===========================================================================

def _strip_comments(sql: str) -> str:
    sql = _RC_BLOCK_CMT.sub(' ', sql)
    sql = _RC_LINE_CMT.sub(' ', sql)
    return sql


def _normalize(sql: str) -> str:
    sql = _strip_comments(sql)
    sql = sql.replace('\r\n', '\n').replace('\r', '\n')
    return _RC_WS.sub(' ', sql).strip()


def _jinja_label(token: str) -> str:
    """{{params.IKG_SCHEMA}} → 'IKG_SCHEMA'"""
    m = re.search(r'params\.([^}]+)', token)
    return m.group(1).strip() if m else token.strip()


def _parse_table_token(raw: str) -> Tuple[str, str]:
    """
    '{{params.X}}.tablename'  → ('X', 'tablename')
    'schema.tablename'        → ('schema', 'tablename')
    'tablename'               → ('', 'tablename')
    """
    raw = raw.strip().rstrip(';').strip()
    jm = re.match(r'(\{\{[^}]+\}\})\.(.*)', raw)
    if jm:
        return _jinja_label(jm.group(1)), jm.group(2).strip()
    if '.' in raw:
        p = raw.rsplit('.', 1)
        schema = p[0].strip()
        if re.match(r'\{\{', schema):
            return _jinja_label(schema), p[1].strip()
        return schema, p[1].strip()
    return '', raw.strip()


def _find_paren_end(text: str, start: int) -> int:
    """Return index of ')' closing '(' at start."""
    depth, i, n = 0, start, len(text)
    while i < n:
        c = text[i]
        if c == "'":
            i += 1
            while i < n and text[i] != "'":
                if text[i] == '\\': i += 1
                i += 1
        elif c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return n - 1


def _split_comma(text: str) -> List[str]:
    """Split by top-level commas (not inside parentheses/quotes)."""
    parts, cur, depth = [], [], 0
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == "'":
            cur.append(c); i += 1
            while i < n and text[i] != "'":
                if text[i] == '\\': cur.append(text[i]); i += 1
                cur.append(text[i]); i += 1
            if i < n: cur.append(text[i])
        elif c == '"':
            cur.append(c); i += 1
            while i < n and text[i] != '"':
                cur.append(text[i]); i += 1
            if i < n: cur.append(text[i])
        elif c == '(':
            depth += 1; cur.append(c)
        elif c == ')':
            depth -= 1; cur.append(c)
        elif c == ',' and depth == 0:
            parts.append(''.join(cur).strip()); cur = []
        else:
            cur.append(c)
        i += 1
    if cur:
        parts.append(''.join(cur).strip())
    return parts


def _split_stmts(sql: str) -> List[str]:
    """Split on top-level semicolons. Slice-based for speed."""
    result, start, depth, i, n = [], 0, 0, 0, len(sql)
    while i < n:
        c = sql[i]
        if c == "'":
            i += 1
            while i < n:
                if sql[i] == "'": break
                if sql[i] == '\\': i += 1
                i += 1
        elif c == '(': depth += 1
        elif c == ')':
            if depth > 0: depth -= 1
        elif c == ';' and depth == 0:
            chunk = sql[start:i].strip()
            if chunk: result.append(chunk)
            start = i + 1
        i += 1
    chunk = sql[start:].strip()
    if chunk: result.append(chunk)
    return result

# ── Speed: module-level constants (prevents rebuilding on every call) ──────────
_LITERAL_PATTERNS = [
    re.compile(r"^'[^']*'(?::\:\w[\w\s]*)?$"),
    re.compile(r"^\{\{[^}]+\}\}(?::\:\w[\w\s]*)?$"),
    re.compile(r"^-?\d+(\.\d+)?(?::\:\w[\w\s]*)?$"),
    re.compile(r"^(true|false|null)(?::\:\w[\w\s]*)?$", re.IGNORECASE),
]
_SYSTEM_VALUE_TOKENS = {
    'CURRENT_DATE', 'CURRENT_TIMESTAMP', 'CURRENT_TIME',
    'LOCALTIMESTAMP', 'LOCALTIME', 'NOW', 'CLOCK_TIMESTAMP',
    'TRANSACTION_TIMESTAMP', 'STATEMENT_TIMESTAMP',
}
_HARD_KW_EXPR = frozenset(SQL_KEYWORDS | {
    'NEW','OLD','ARRAY',
    'YEAR','MONTH','DAY','HOUR','MINUTE','SECOND','MILLISECOND','MICROSECOND',
    'WEEK','DOW','DOY','EPOCH','DECADE','CENTURY','MILLENNIUM',
    'TIMEZONE','TIMEZONE_HOUR','TIMEZONE_MINUTE','INTERVAL','AT','TIME','ZONE',
})
_HARD_KW_OVER = frozenset({
    'SELECT','FROM','WHERE','AND','OR','NOT','IN','IS','CASE','WHEN','THEN',
    'ELSE','END','AS','ON','JOIN','LEFT','RIGHT','INNER','OUTER','FULL','CROSS',
    'GROUP','BY','ORDER','HAVING','LIMIT','DISTINCT','ALL','UNION','NULLS',
    'LAST','FIRST','ASC','DESC','NULL','TRUE','FALSE','ROWS','RANGE','GROUPS',
    'PARTITION','OVER','FILTER','BETWEEN','LIKE','ILIKE','UNBOUNDED',
    'PRECEDING','FOLLOWING','CURRENT','ROW','WINDOW','WITHIN','EXCLUDE',
})
_NEVER_ALIAS_SET = frozenset({
    'FROM','BY','PARTITION','ORDER','ROWS','RANGE','PRECEDING','FOLLOWING',
    'UNBOUNDED','BETWEEN','AND','OR','NOT','WHEN','THEN',
    'ON','USING','INTO','SET','WHERE','HAVING','IS','IN','LIKE','ILIKE','WITHIN','GROUPS',
})
_SYSONLY_KW = frozenset({
    'SELECT','FROM','WHERE','AND','OR','NOT','IN','IS','AS','ON',
    'JOIN','GROUP','BY','ORDER','HAVING','LIMIT','CASE','WHEN','THEN',
    'ELSE','END','NULL','TRUE','FALSE','BETWEEN','LIKE','ILIKE',
    'PARTITION','OVER','FILTER','INTERVAL','EXTRACT','EPOCH','DOW',
    'CONCAT','SUBSTRING','TEXT','INT','INTEGER','NUMERIC',
    'BIGINT','VARCHAR','TIMESTAMP','DAYS','COALESCE',
})
_P_SQ           = re.compile(r"'[^']*'")
_P_DQ           = re.compile(r'"[^"]*"')
_P_JINJA        = re.compile(r'\{\{[^}]+\}\}')
_P_CAST         = re.compile(r'::\s*\w+')
_P_NUM          = re.compile(r'\b\d+\.?\d*\b')
_P_DOT_ANY      = re.compile(r'\b\w+\.\w+\b')
_P_ALIAS_DOTCOL = re.compile(r'\b(\w+)\s*\.\s*("(?:[^"]+)"|\w+)')
_P_LONE_DQ      = re.compile(r'(?<!\w)"([^"]+)"')
_P_PART_BY      = re.compile(r'\bPARTITION\s+BY\b', re.IGNORECASE)
_P_ORDER_BY     = re.compile(r'\bORDER\s+BY\b', re.IGNORECASE)
_P_PART_STOP    = re.compile(r'\b(PARTITION\s+BY|ORDER\s+BY|ROWS|RANGE|GROUPS|EXCLUDE|FILTER)\b', re.IGNORECASE)
_P_FROM_M       = re.compile(r'\bFROM\b', re.IGNORECASE)
_P_WORDS        = re.compile(r'\b([a-zA-Z_]\w*)\b')
_P_KW_CACHE: dict = {}

# Pre-warm keyword position cache — eliminates re.compile overhead at runtime
for _kw in ('GROUP BY', 'HAVING', 'ORDER BY', 'LIMIT', 'DISTRIBUTED',
            'UNION', 'INTERSECT', 'EXCEPT', 'UNION ALL', 'WHERE', 'FROM'):
    _w0 = _kw.split()[0]
    _P_KW_CACHE[_kw.upper()] = (
        _w0[0].upper(), _w0[0].lower(),
        re.compile(r'\b' + re.escape(_w0) + r'\b', re.IGNORECASE),
        re.compile(r'\b' + _kw.replace(' ', r'\s+') + r'\b', re.IGNORECASE),
    )
del _kw, _w0


def _is_system_value(expr: str) -> bool:
    """Return True if expr is a system pseudo-column (CURRENT_DATE, NOW(), etc.)."""
    e = _strip_cast(expr).strip().upper()
    # bare: CURRENT_DATE
    if e in _SYSTEM_VALUE_TOKENS:
        return True
    # function call: NOW() or NOW(...)
    fn_m = re.match(r'^(\w+)\s*\(', e)
    if fn_m and fn_m.group(1) in _SYSTEM_VALUE_TOKENS:
        return True
    return False


def _is_literal(expr: str) -> bool:
    """Return True if expr is a plain literal value (string, number, bool, null, jinja param)."""
    e = expr.strip()
    for pat in _LITERAL_PATTERNS:
        if pat.match(e):
            return True
    return False


def _extract_literal_src(expr: str) -> str:
    """Extract the readable source label from a literal expression."""
    e = re.sub(r'::\s*\w[\w\s,()]*$', '', expr.strip()).strip()
    jm = re.match(r'\{\{([^}]+)\}\}', e)
    if jm:
        return '{{' + jm.group(1) + '}}'
    if e.startswith("'") and e.endswith("'"):
        return e[1:-1]
    return e


def _strip_cast(expr: str) -> str:
    """Remove trailing ::typename (possibly with spaces) from expression."""
    return _RC_CAST.sub('', expr).strip()


def _dequote(name: str) -> str:
    """Strip surrounding double-quotes from a quoted identifier: \"col\" → col."""
    name = name.strip()
    if name.startswith('"') and name.endswith('"'):
        return name[1:-1]
    return name


def _is_double_quoted_ident(token: str) -> bool:
    """Return True if token is a double-quoted identifier like \"RepCRD\"."""
    t = token.strip()
    return t.startswith('"') and t.endswith('"') and len(t) > 2


# ===========================================================================
#  EXPRESSION PARSER
#  Returns: (target_col, [(source_table_alias, source_col)], logic, sql_process)
# ===========================================================================

def _is_function_name(word: str) -> bool:
    return word.upper() in SQL_FUNCTIONS


def _extract_col_refs_from_expr(logic: str) -> List[Tuple[str, str]]:
    """Extract (alias, col) refs using pre-compiled patterns and module-level sets."""
    refs: List[Tuple[str, str]] = []
    for m in _P_ALIAS_DOTCOL.finditer(logic):
        pfx = m.group(1)
        if pfx.upper() in SQL_FUNCTIONS or pfx.upper() in _HARD_KW_EXPR or pfx[0].isdigit():
            continue
        col_raw = m.group(2)
        refs.append((pfx.lower(), col_raw[1:-1] if col_raw.startswith('"') else col_raw))
    if refs:
        return refs
    for m in _P_LONE_DQ.finditer(logic):
        refs.append(('', m.group(1)))
    if refs:
        return refs
    cl = _P_SQ.sub(' ', logic)
    cl = _P_DQ.sub(' __Q__ ', cl)
    cl = _P_JINJA.sub(' __J__ ', cl)
    cl = _P_CAST.sub(' ', cl)
    cl = _P_NUM.sub(' ', cl)
    for m in _P_WORDS.finditer(cl):
        w = m.group(1)
        if w.upper() not in _HARD_KW_EXPR and w.upper() not in SQL_FUNCTIONS and w not in ('__J__', '__Q__'):
            refs.append(('', w))
    return refs
def _parse_over_clause(logic: str) -> List[Tuple[str, str]]:
    """Extract col refs from OVER/FILTER clauses. Pre-compiled patterns."""
    refs: List[Tuple[str, str]] = []
    over_m = _RC_OVER.search(logic)
    if over_m:
        ep = _find_paren_end(logic, over_m.end() - 1)
        inner = logic[over_m.end():ep]
        for pat in (_P_PART_BY, _P_ORDER_BY):
            km = pat.search(inner)
            if km:
                rest = inner[km.end():]
                stop = _P_PART_STOP.search(rest)
                clause = rest[:stop.start()] if stop else rest
                seen: set = set()
                for m in _RC_ALIAS_DOT.finditer(clause):
                    p, c = m.group(1), m.group(2)
                    if p.upper() not in _HARD_KW_OVER and p.upper() not in SQL_FUNCTIONS:
                        refs.append((p.lower(), c)); seen.add(c)
                cl2 = _P_DOT_ANY.sub(' ', clause)
                cl2 = _P_SQ.sub(' ', cl2); cl2 = _P_NUM.sub(' ', cl2)
                for m in _P_WORDS.finditer(cl2):
                    w = m.group(1)
                    if w.upper() not in _HARD_KW_OVER and w.upper() not in SQL_FUNCTIONS and w not in seen:
                        refs.append(('', w)); seen.add(w)
    filter_m = _RC_FILTER_W.search(logic)
    if filter_m:
        fp = logic.index('(', filter_m.start())
        ep2 = _find_paren_end(logic, fp)
        inner2 = logic[fp+1:ep2]
        seen2: set = set()
        for m in _RC_ALIAS_DOT.finditer(inner2):
            p, c = m.group(1), m.group(2)
            if p.upper() not in _HARD_KW_OVER and p.upper() not in SQL_FUNCTIONS:
                refs.append((p.lower(), c)); seen2.add(c)
        cl3 = _P_DOT_ANY.sub(' ', inner2)
        cl3 = _P_SQ.sub(' ', cl3); cl3 = _P_NUM.sub(' ', cl3)
        for m in _P_WORDS.finditer(cl3):
            w = m.group(1)
            if w.upper() not in _HARD_KW_OVER and w.upper() not in SQL_FUNCTIONS and w not in seen2:
                refs.append(('', w)); seen2.add(w)
    return refs
def _determine_sql_process(logic: str, refs: List[Tuple[str, str]]) -> str:
    """
    Decide sql_process:
      'select'       — normal column reference
      'select-value' — literal / computed value (no real column source)
    """
    if not refs:
        return 'select-value'
    # If every ref is a literal/value, return select-value
    if all(alias == '' and col == '' for alias, col in refs):
        return 'select-value'
    return 'select'


# ===========================================================================
#  TABLE ALIAS EXTRACTION
# ===========================================================================

_TBL_TOK = r'((?:\{\{[^}]+\}\}|\w+)(?:\.(?:\{\{[^}]+\}\}|\w+))?)'


def _extract_aliases(query: str) -> Dict[str, Tuple[str, str]]:
    """
    Scan FROM / JOIN for table references, including inline subqueries.
    JOIN (SELECT ... FROM real_table) alias  ->  alias -> real_table
    Returns {alias_lower: (schema, tablename)}.
    """
    aliases: Dict[str, Tuple[str, str]] = {}

    # Register named real tables from FROM/JOIN
    for m in _RC_TBLREF.finditer(query):
        tbl_raw = m.group(2).strip()
        alias_raw = (m.group(4) or '').strip()  # group 4 after negative lookahead fix
        if tbl_raw.upper() in SQL_KEYWORDS or tbl_raw.upper() in SQL_FUNCTIONS:
            continue
        if tbl_raw.startswith('('):
            continue
        schema, tbl = _parse_table_token(tbl_raw)
        tbl_lower = tbl.lower()
        alias = alias_raw.lower() if alias_raw and alias_raw.upper() not in SQL_KEYWORDS else tbl_lower
        aliases[alias] = (schema, tbl)
        aliases[tbl_lower] = (schema, tbl)

    # Register inline subquery aliases WITHOUT building col maps (avoids recursion)
    # Just map alias -> first real table inside the subquery for source resolution
    pat = re.compile(
        r'\b(?:FROM|(?:LEFT|RIGHT|FULL|INNER|CROSS)?\s*(?:OUTER\s+)?JOIN)\s*\(',
        re.IGNORECASE
    )
    i = 0
    n = len(query)
    while i < n:
        m = pat.search(query, i)
        if not m:
            break
        try:
            paren_start = query.index('(', m.start())
        except ValueError:
            break
        paren_end = _find_paren_end(query, paren_start)
        inner_sql = query[paren_start+1:paren_end].strip()
        if re.search(r'\bSELECT\b', inner_sql, re.IGNORECASE):
            after = query[paren_end+1:].lstrip()
            alias_m = re.match(r'(?:AS\s+)?(\w+)', after, re.IGNORECASE)
            if alias_m:
                alias = alias_m.group(1).lower()
                if alias.upper() not in SQL_KEYWORDS and alias not in aliases:
                    # Find first real table in the inner SELECT (lightweight scan)
                    inner_tbls = []
                    for tm in _RC_TBLREF.finditer(inner_sql):
                        tr = tm.group(2).strip()
                        if tr and not tr.startswith('(') and tr.upper() not in SQL_KEYWORDS:
                            s, t = _parse_table_token(tr)
                            if t:
                                inner_tbls.append((s, t))
                                break
                    if inner_tbls:
                        aliases[alias] = inner_tbls[0]
        i = paren_end + 1

    return aliases


def _from_tables(query: str) -> List[Tuple[str, str]]:
    """Return unique (schema, table) from FROM/JOIN clauses."""
    seen, result = set(), []
    for s, t in _extract_aliases(query).values():
        if t and (s, t) not in seen:
            seen.add((s, t)); result.append((s, t))
    return result


# ===========================================================================
#  CTE PARSING
# ===========================================================================

def _extract_ctes(sql: str) -> Tuple[Dict[str, str], str]:
    """Extract WITH CTEs. Returns ({name: body}, remainder)."""
    ctes: Dict[str, str] = {}
    text = sql.lstrip()
    m = re.match(r'\bWITH\b\s*', text, re.IGNORECASE)
    if not m:
        return ctes, sql
    tail = text[m.end():]
    while True:
        nm = re.match(r'(\w+)\s+AS\s*\(', tail, re.IGNORECASE)
        if not nm:
            break
        cte_name = nm.group(1).lower()
        open_p = nm.end() - 1
        close_p = _find_paren_end(tail, open_p)
        ctes[cte_name] = tail[open_p+1:close_p].strip()
        tail = tail[close_p+1:].lstrip()
        if tail.startswith(','):
            tail = tail[1:].lstrip()
        else:
            break
    return ctes, tail


# ===========================================================================
#  SELECT LIST EXTRACTION
# ===========================================================================

def _extract_select_list(query: str) -> str:
    """Return column list between SELECT [DISTINCT [ON (...)]] and top-level FROM."""
    sel_m = _RC_SELECT.search(query)
    if not sel_m:
        return ''
    start = sel_m.end()
    # Skip DISTINCT ON (...), DISTINCT, or ALL
    rest = query[start:]
    skip_m = _RC_DISTINCT_SK.match(rest)
    if skip_m:
        start += skip_m.end()
    depth, i, n = 0, start, len(query)
    while i < n:
        c = query[i]
        if c == "'":
            i += 1
            while i < n and query[i] != "'":
                if query[i] == '\\': i += 1
                i += 1
        elif c == '"':
            i += 1
            while i < n and query[i] != '"': i += 1
        elif c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
            if depth < 0:
                return query[start:i].strip()
        elif depth == 0:
            if _P_FROM_M.match(query, i) and (i == 0 or not query[i-1].isalpha()):
                return query[start:i].strip()
        i += 1
    return query[start:].strip()


# ===========================================================================
#  CORE EXPRESSION ANALYSIS
# ===========================================================================

def _analyse_expr(
    expr: str,
    aliases: Dict[str, Tuple[str, str]],
    all_ctes: Dict[str, str],
    cte_col_maps: Dict[str, Dict[str, List[Tuple[str, str, str]]]],
    all_from_tables: List[Tuple[str, str]],
) -> List[Dict[str, Any]]:
    """
    Parse one SELECT expression. Returns list of partial row dicts.
    """
    expr = expr.strip()
    if not expr:
        return []

    # ── Skip DISTINCT ON (...) ─────────────────────────────────────────────
    if _RC_DISTINCT_ON.match(expr):
        return []

    # ── Subquery expression: (SELECT ...) alias ───────────────────────────
    if expr.startswith('('):
        subq_m = re.match(r'^\((.+)\)\s*(?:AS\s+)?(\w+)\s*$', expr, re.IGNORECASE | re.DOTALL)
        if subq_m:
            inner_sql = subq_m.group(1).strip()
            sub_alias = subq_m.group(2)
            if _RC_SELECT.search(inner_sql):
                inner_sel = _extract_select_list(inner_sql)
                inner_from_tables = _from_tables(inner_sql)
                inner_aliases = _extract_aliases(inner_sql)
                sub_rows = []
                if inner_sel:
                    for inner_expr in _split_comma(inner_sel):
                        inner_expr = inner_expr.strip()
                        if not inner_expr or inner_expr == '*':
                            continue
                        _, inner_logic = _split_alias(inner_expr)
                        inner_refs = _extract_col_refs_from_expr(inner_logic or inner_expr)
                        for r_alias, r_col in inner_refs:
                            res_list = _resolve_col_ref(
                                r_alias, r_col, inner_aliases, all_ctes, cte_col_maps, inner_from_tables
                            )
                            for res_schema, res_tbl, res_col in res_list:
                                if not res_tbl and inner_from_tables:
                                    res_schema2, res_tbl2 = _find_column_in_tables(res_col, inner_from_tables)
                                    if res_tbl2:
                                        res_schema, res_tbl = res_schema2, res_tbl2
                                sub_rows.append(dict(
                                    target_column=sub_alias,
                                    source_table=res_tbl, source_schema=res_schema,
                                    source_column=res_col, logic=expr, sql_process='select',
                                ))
                if sub_rows:
                    return sub_rows
                return [dict(target_column=sub_alias, source_table='', source_schema='',
                             source_column='', logic=expr, sql_process='select')]

    # ── Step 1: split alias ────────────────────────────────────────────────
    target_col, logic = _split_alias(expr)

    if not target_col and _is_double_quoted_ident(expr.strip()):
        col_name = _dequote(expr.strip())
        return [dict(target_column=col_name, source_table='', source_schema='',
                     source_column=col_name, logic=expr, sql_process='select')]

    if not target_col:
        dqcol_m = re.match(r'^(\w+)\."([^"]+)"(?:::\w+)?$', expr.strip())
        if dqcol_m:
            pfx, col = dqcol_m.group(1), dqcol_m.group(2)
            schema_r, tbl_r = aliases.get(pfx.lower(), ('', pfx))
            if schema_r == '__CTE__': schema_r = ''
            return [dict(target_column=col, source_table=tbl_r, source_schema=schema_r,
                         source_column=col, logic=expr, sql_process='select')]

    if not target_col:
        dm = re.match(r'^(\w+)\.(\w+)(?:::\w+)?$', (logic or expr).strip())
        if dm:
            target_col = dm.group(2)
        elif re.match(r'^(\w+)(?:::\w+)?$', (logic or expr).strip()):
            target_col = re.match(r'^(\w+)', (logic or expr).strip()).group(1)
        if not target_col:
            target_col = ''

    use_logic = logic if logic else expr

    # ── Step 2: literal / system-value / select-value detection ───────────
    logic_stripped = _strip_cast(use_logic)

    # (a) Plain string / number / null / jinja param → select-value
    if _is_literal(logic_stripped):
        src = _extract_literal_src(logic_stripped)
        return [dict(target_column=target_col, source_table='', source_schema='',
                     source_column=src, logic=expr, sql_process='select-value')]

    # (b) System pseudo-columns: CURRENT_DATE, CURRENT_TIMESTAMP, NOW() etc.
    #     These appear anywhere in an expression referencing only system state.
    #     If the ENTIRE logic (stripped) is a system value → select-value.
    #     If system value appears INSIDE a larger expression (concat, extract),
    #     the system pseudo-column itself becomes the source_column.
    if _is_system_value(logic_stripped):
        # Whole expression is a system function — no real table source
        src_col = _strip_cast(logic_stripped).upper()
        # Normalise: CURRENT_DATE or NOW() → keep as-is for readability
        src_col = re.sub(r'\(\s*\)', '()', src_col)
        return [dict(target_column=target_col or src_col.lower(),
                     source_table='', source_schema='', source_column=src_col,
                     logic=expr, sql_process='select-value')]

    # (c) Expression uses ONLY system pseudo-cols / pure functions → select-value
    logic_no_literals = _P_SQ.sub(' ', use_logic)
    logic_no_literals = _P_NUM.sub(' ', logic_no_literals)
    logic_no_literals = _P_CAST.sub(' ', logic_no_literals)
    logic_no_literals = _P_JINJA.sub(' ', logic_no_literals)
    word_tokens = _P_WORDS.findall(logic_no_literals)
    _SYSTEM_ONLY_KW = _SYSONLY_KW | _SYSTEM_VALUE_TOKENS | SQL_FUNCTIONS
    real_word_tokens = [w for w in word_tokens if w.upper() not in _SYSTEM_ONLY_KW]
    sys_val_tokens = [w.upper() for w in word_tokens if w.upper() in _SYSTEM_VALUE_TOKENS]

    if not real_word_tokens:
        if sys_val_tokens:
            # Only system pseudo-columns (CURRENT_DATE, NOW, etc.) — no real table column
            return [dict(target_column=target_col, source_table='', source_schema='',
                         source_column=sys_val_tokens[0],
                         logic=expr, sql_process='select-value')]
        elif word_tokens:
            # No real column tokens found with strict filter. Try looser filter:
            # Remove only hard control-flow keywords + known SQL functions.
            # This catches real col names that share names with datetime fields
            # like 'date', 'year', 'quarter' — and hidden cols like 'range' inside
            # split_part(range, '-', 1).
            _LOOSE_HARD = {
                'SELECT','FROM','WHERE','AND','OR','NOT','IN','IS','AS','ON',
                'JOIN','GROUP','BY','ORDER','HAVING','LIMIT','CASE','WHEN','THEN',
                'ELSE','END','NULL','TRUE','FALSE','BETWEEN','LIKE','ILIKE',
                'PARTITION','OVER','FILTER','INTERVAL','EXTRACT','EPOCH','DOW',
                'CONCAT','SUBSTRING','TEXT','INT','INTEGER','NUMERIC',
                'BIGINT','VARCHAR','TIMESTAMP','DAYS','COALESCE',
            }
            loose_real = [w for w in word_tokens
                          if w.upper() not in _LOOSE_HARD
                          and w.upper() not in SQL_FUNCTIONS
                          and w.upper() not in _SYSTEM_VALUE_TOKENS]
            if loose_real:
                # Found hidden real column refs → inject them as refs and fall through
                # to normal resolution (Step 5+)
                pass  # word_tokens handled via body_refs below
            else:
                # Pure function, truly no real col refs (COUNT(*), row_number() etc.)
                src_col = use_logic.strip()[:120]
                return [dict(target_column=target_col, source_table='', source_schema='',
                             source_column=src_col, logic=expr, sql_process='select-value')]

    # (d) COUNT(*) / COUNT(DISTINCT *) without OVER → select-value
    count_star_m = re.match(r'^(COUNT\s*\(\s*(?:DISTINCT\s+)?\*?\s*\))',
                            use_logic.strip(), re.IGNORECASE)
    if count_star_m and not re.search(r'\bOVER\s*\(', use_logic, re.IGNORECASE):
        return [dict(target_column=target_col, source_table='', source_schema='',
                     source_column=count_star_m.group(1),
                     logic=expr, sql_process='select-value')]

    # (e) Double-quoted identifier as logic: "col" 
    # (d) Expression references ONLY system pseudo-columns
    if _is_double_quoted_ident(use_logic.strip()):
        col_name = _dequote(use_logic.strip())
        return [dict(target_column=target_col or col_name, source_table='', source_schema='',
                     source_column=col_name, logic=expr, sql_process='select')]

    # ── Step 2c: infer target_col for bare-word logic ─────────────────────
    if not target_col and use_logic:
        bm = re.match(r'^(\w+)(?:::\w+)?$', use_logic.strip())
        if bm:
            target_col = bm.group(1)

    # ── Step 3: window function detection ─────────────────────────────────
    has_over = bool(_RC_OVER.search(use_logic))

    main_logic = use_logic
    if has_over:
        ov_m = re.search(r'\bOVER\s*\(', main_logic, re.IGNORECASE)
        if ov_m:
            ov_start = main_logic.index('(', ov_m.start())
            ov_end = _find_paren_end(main_logic, ov_start)
            main_logic = main_logic[:ov_m.start()] + main_logic[ov_end+1:]

    fl_m = re.search(r'\bFILTER\s*\(', main_logic, re.IGNORECASE)
    if fl_m:
        fl_start = main_logic.index('(', fl_m.start())
        fl_end = _find_paren_end(main_logic, fl_start)
        main_logic = main_logic[:fl_m.start()] + main_logic[fl_end+1:]

    body_refs = _extract_col_refs_from_expr(main_logic)
    window_refs = _parse_over_clause(use_logic)

    seen_refs: Set[Tuple[str, str]] = set()
    unique_refs: List[Tuple[str, str]] = []
    for r in body_refs + window_refs:
        if r not in seen_refs:
            seen_refs.add(r); unique_refs.append(r)

    # ── Step 4: no refs → select-value ────────────────────────────────────
    if not unique_refs:
        # row_number() over() or other pure function with no column input
        src_col = ''
        if has_over:
            # Extract the aggregate function call itself as source
            fn_m = re.match(r'^(\w+\s*\(.*?\))', use_logic.strip(), re.DOTALL)
            src_col = fn_m.group(1)[:80] if fn_m else use_logic[:80]
        return [dict(target_column=target_col, source_table='', source_schema='',
                     source_column=src_col, logic=expr, sql_process='select-value')]

    # ── Step 5: infer target_col from first ref if still blank ────────────
    if not target_col and unique_refs:
        _, first_col = unique_refs[0]
        target_col = first_col

    # ── Step 6: resolve each ref, fall back to information_schema ─────────
    rows = []
    for alias_pfx, col in unique_refs:
        resolved = _resolve_col_ref(
            alias_pfx, col, aliases, all_ctes, cte_col_maps, all_from_tables
        )
        for res_schema, res_tbl, res_col in resolved:
            # If no table resolved and we have a DB, try information_schema
            if not res_tbl and all_from_tables:
                is2, it2 = _find_column_in_tables(res_col, all_from_tables)
                if it2:
                    res_schema, res_tbl = is2, it2
            rows.append(dict(
                target_column=target_col,
                source_table=res_tbl, source_schema=res_schema, source_column=res_col,
                logic=expr, sql_process='select',
            ))

    return rows if rows else [dict(
        target_column=target_col, source_table='', source_schema='', source_column='',
        logic=expr, sql_process='select',
    )]


def _split_alias(expr: str) -> Tuple[str, str]:
    """
    Split 'logic [AS] alias' into (alias, logic).
    Handles:
      b.col AS col2                  → ('col2', 'b.col')
      'Adobe'::text AS "source"      → ('source', "'Adobe'::text")
      1::bigint AS "Custom Field 1"  → ('Custom Field 1', '1::bigint')
      "quarter" as quarter           → ('quarter', '"quarter"')
      "quarter" quarter              → ('quarter', '"quarter"')  implicit
      "Company ID"                   → ('Company ID', '"Company ID"')  self-ref
      curr."RepCRD"                  → ('RepCRD', 'curr."RepCRD"')  no alias; self-ref
      b.comp_srch_lst::text array    → ('array', 'b.comp_srch_lst::text')
      row_number() over() as index   → ('index', 'row_number() over()')
      CASE...END already_engaged     → ('already_engaged', 'CASE...END')
      optimus_first_name first_name  → ('first_name', 'optimus_first_name')
      null sub_parameter             → ('sub_parameter', 'null')
    """
    expr = expr.strip()
    if not expr:
        return '', ''

    # ── Explicit AS at top level ───────────────────────────────────────────
    as_result = _find_top_level_as(expr)
    if as_result is not None:
        as_start, as_end = as_result
        raw_alias = expr[as_end:].strip()
        logic = expr[:as_start].strip()
        # Alias may itself be double-quoted: AS "source"
        alias = _dequote(raw_alias) if _is_double_quoted_ident(raw_alias) else raw_alias
        if alias and alias.upper() not in {'SELECT','FROM','WHERE','AND','OR','ON',
                                            'JOIN','LEFT','RIGHT','INNER','OUTER',
                                            'FULL','CROSS','GROUP','HAVING','ORDER'}:
            return alias, logic

    # ── No explicit AS — try implicit trailing alias ──────────────────────
    alias, logic = _find_implicit_alias(expr)
    return alias, logic


def _find_top_level_as(expr: str):
    """Return (start, end) of top-level AS keyword, or None."""
    depth = 0
    i = 0
    n = len(expr)
    last_as = None
    while i < n:
        c = expr[i]
        if c == "'":
            i += 1
            while i < n and expr[i] != "'":
                if expr[i] == '\\': i += 1
                i += 1
        elif c == '"':
            i += 1
            while i < n and expr[i] != '"': i += 1
        elif c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        elif depth == 0 and (c == 'A' or c == 'a'):
            if i+2 < n and (expr[i+1] == 'S' or expr[i+1] == 's') and not (expr[i+2].isalnum() or expr[i+2] == '_'):
                b = expr[i-1] if i > 0 else ' '
                if not (b.isalnum() or b == '_'):
                    last_as = (i, i + 3)
        i += 1
    return last_as


def _find_implicit_alias(expr: str) -> Tuple[str, str]:
    """
    Detect implicit alias — the last top-level token that looks like an alias.
    Returns (alias, logic) or ('', expr).
    """
    # ── Simple alias.col: NEVER split these — whole expr is logic ─────────
    # e.g. 'a.col1', 'b.acc_mhh_n', 's.profile_date::text'
    if re.match(r'^\w+\.\w+(?:::\w[\w\s,()]*)?$', expr.strip()):
        return '', expr
    if re.match(r'^\w+\."[^"]+"(?:::\w+)?$', expr.strip()):
        return '', expr

    # ── Trailing double-quoted alias: expr "Quoted Alias" ─────────────────
    dq_trail = re.search(r'\s("(?:[^"]+)")\s*$', expr)
    if dq_trail:
        alias = _dequote(dq_trail.group(1))
        logic = expr[:dq_trail.start()].strip()
        if logic:
            return alias, logic

    # ── Special case: (paren_expr)[::type] alias_word ─────────────────────
    if expr.lstrip().startswith('('):
        stripped = expr.lstrip()
        close = _find_paren_end(stripped, 0)
        after_paren = stripped[close+1:].strip()
        after_paren = re.sub(r'^::\s*\w[\w\s]*', '', after_paren).strip()
        trail_m = re.match(r'^([a-zA-Z_]\w*)\s*$', after_paren)
        _NEVER = _NEVER_ALIAS_SET
        if trail_m and trail_m.group(1).upper() not in _NEVER:
            alias_word = trail_m.group(1)
            logic_part = stripped[:close+1].strip()
            try:
                cast_part = stripped[close+1:stripped.index(alias_word, close+1)].strip()
            except ValueError:
                cast_part = ''
            if cast_part:
                logic_part = logic_part + cast_part
            return alias_word, logic_part

    tokens = _top_level_tokens(expr)

    if len(tokens) == 1 and _is_double_quoted_ident(tokens[0]):
        return _dequote(tokens[0]), expr

    if len(tokens) < 2:
        return '', expr

    last_tok = tokens[-1]
    if not re.match(r'^[a-zA-Z_]\w*$', last_tok):
        return '', expr

    _NEVER = {
        'FROM','BY','PARTITION','ORDER','ROWS','RANGE','PRECEDING','FOLLOWING',
        'UNBOUNDED','BETWEEN','AND','OR','NOT','WHEN','THEN',
        'ON','USING','INTO','SET','WHERE','HAVING','IS','IN',
        'LIKE','ILIKE','WITHIN','GROUPS',
    }
    if last_tok.upper() in _NEVER:
        return '', expr

    prev_tok = tokens[-2] if len(tokens) >= 2 else ''
    if prev_tok.upper() in _NEVER:
        return '', expr

    pos = _find_last_top_level_word_pos(expr, last_tok)
    if pos is None:
        return '', expr

    logic = expr[:pos].strip()
    if not logic:
        return '', expr
    if logic.upper() in {'SELECT','FROM','WHERE','AND','OR','ON',
                         'JOIN','LEFT','RIGHT','INNER','GROUP','HAVING','ORDER'}:
        return '', expr

    return last_tok, logic
def _top_level_tokens(expr: str) -> List[str]:
    """Extract all top-level tokens from expr: words, numbers, quoted strings."""
    tokens = []
    depth, i, n = 0, 0, len(expr)
    while i < n:
        c = expr[i]
        if c == "'":
            if depth == 0:
                start = i; i += 1
                while i < n and expr[i] != "'":
                    if expr[i] == '\\': i += 1
                    i += 1
                tokens.append(expr[start:i+1] if i < n else expr[start:])
            else:
                i += 1
                while i < n and expr[i] != "'": i += 1
        elif c == '"':
            if depth == 0:
                start = i; i += 1
                while i < n and expr[i] != '"': i += 1
                tokens.append(expr[start:i+1] if i < n else expr[start:])
            else:
                i += 1
                while i < n and expr[i] != '"': i += 1
        elif c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        elif depth == 0 and (c.isalpha() or c == '_'):
            # Word token
            start = i
            while i < n and (expr[i].isalnum() or expr[i] == '_'):
                i += 1
            tokens.append(expr[start:i])
            continue
        elif depth == 0 and (c.isdigit() or (c == '-' and i + 1 < n and expr[i+1].isdigit())):
            # Numeric token
            start = i
            if c == '-': i += 1
            while i < n and (expr[i].isdigit() or expr[i] == '.'):
                i += 1
            tokens.append(expr[start:i])
            continue
        i += 1
    return tokens


def _find_last_top_level_word_pos(expr: str, word: str) -> Optional[int]:
    """Return start position of last occurrence of `word` at top-level depth."""
    depth, i, n, last_pos = 0, 0, len(expr), None
    while i < n:
        c = expr[i]
        if c == "'":
            i += 1
            while i < n and expr[i] != "'":
                if expr[i] == '\\': i += 1
                i += 1
        elif c == '"':
            i += 1
            while i < n and expr[i] != '"': i += 1
        elif c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        elif depth == 0:
            if expr[i:i+len(word)] == word:
                # Check word boundaries
                before = expr[i-1] if i > 0 else ' '
                after = expr[i+len(word)] if i+len(word) < n else ' '
                if not (before.isalnum() or before == '_') and not (after.isalnum() or after == '_'):
                    last_pos = i
        i += 1
    return last_pos


# ===========================================================================
#  CTE COLUMN MAP
# ===========================================================================

_BUILD_CTE_VISITED: set = set()  # tracks CTE names currently being built (cycle guard)

def _build_cte_col_map(
    cte_name: str, cte_body: str, all_ctes: Dict[str, str]
) -> Dict[str, List[Tuple[str, str, str]]]:
    """
    Returns {output_col_lower: [(schema, real_table, real_col), ...]}.
    Uses a visited-set to prevent mutual recursion cycles.
    """
    if cte_name in _BUILD_CTE_VISITED:
        return {}  # cycle detected — return empty rather than recurse
    _BUILD_CTE_VISITED.add(cte_name)
    try:
        return _build_cte_col_map_inner(cte_name, cte_body, all_ctes)
    finally:
        _BUILD_CTE_VISITED.discard(cte_name)


def _build_cte_col_map_inner(
    cte_name: str, cte_body: str, all_ctes: Dict[str, str]
) -> Dict[str, List[Tuple[str, str, str]]]:
    col_map: Dict[str, List[Tuple[str, str, str]]] = {}

    inner_ctes, inner_body = _extract_ctes(cte_body)
    merged = {**all_ctes, **inner_ctes}

    aliases = _extract_aliases(cte_body)
    for cn in merged:
        aliases[cn] = ('__CTE__', cn)

    sel = _extract_select_list(inner_body)
    if not sel:
        return col_map

    all_tbls = _from_tables(cte_body)

    for expr in _split_comma(sel):
        expr = expr.strip()
        if not expr:
            continue
        tgt, logic = _split_alias(expr)
        if not tgt:
            dm = re.match(r'^(?:\w+\.)?(\w+)(?:::\w+)?$', logic.strip())
            if dm:
                tgt = dm.group(1)
        if not tgt or tgt == '*':
            continue

        refs = _extract_col_refs_from_expr(logic)
        resolved: List[Tuple[str, str, str]] = []

        for alias_pfx, col in refs:
            resolved.extend(_resolve_col_ref(alias_pfx, col, aliases, merged, {}, all_tbls))

        col_map[tgt.lower()] = resolved if resolved else [('', '', logic[:80])]

    return col_map


# ===========================================================================
#  RESOLVE COLUMN REFERENCE
# ===========================================================================

def _resolve_col_ref(
    alias: str, col: str,
    aliases: Dict[str, Tuple[str, str]],
    all_ctes: Dict[str, str],
    cte_col_maps: Dict[str, Dict[str, List[Tuple[str, str, str]]]],
    all_from_tables: List[Tuple[str, str]],
) -> List[Tuple[str, str, str]]:
    """
    Return [(schema, table, col), ...].
    Handles CTE lookup, inline subquery virtual CTEs, and single-table inference.
    """
    if alias and (alias.upper() in SQL_KEYWORDS or alias.upper() in SQL_FUNCTIONS):
        alias = ''

    if alias:
        schema, tbl = aliases.get(alias, ('', alias))
        if schema == '__CTE__':
            cte_map = cte_col_maps.get(tbl, {})
            r = cte_map.get(col.lower())
            if r:
                return r
            if tbl in all_ctes:
                try:
                    inline_map = _build_cte_col_map(tbl, all_ctes[tbl], all_ctes)
                    r2 = inline_map.get(col.lower())
                    if r2:
                        return r2
                except RecursionError:
                    pass
            return [('', tbl, col)]
        return [(schema, tbl, col)]
    else:
        # No alias — try virtual CTE col map FIRST before single-table shortcut.
        # When the outer SELECT references a bare column (e.g. 'assets') and the only
        # FROM source is an inline subquery alias, look up through its col map so we
        # get the inner source column (e.g. 'assets_sd') not the outer alias name.
        cte_sources_early = [(s, t) for t, (s, _) in aliases.items()
                              if s == '__CTE__' and t in cte_col_maps]
        seen_e: set = set()
        unique_ctes_early = []
        for s, t in cte_sources_early:
            if t not in seen_e:
                seen_e.add(t); unique_ctes_early.append((s, t))
        if unique_ctes_early:
            for _, cte_name in unique_ctes_early:
                cte_map = cte_col_maps.get(cte_name, {})
                r = cte_map.get(col.lower())
                if r:
                    return r

        # No alias — single real-table inference
        real_tbls = [(s, t) for s, t in all_from_tables if t.lower() not in all_ctes]
        if len(real_tbls) == 1:
            return [(real_tbls[0][0], real_tbls[0][1], col)]

        # No real tables — check if there's a single CTE/virtual-CTE in scope
        # (e.g. outer SELECT FROM (SELECT ...) k — all columns come from k)
        cte_sources = [(s, t) for t, (s, _) in aliases.items()
                       if s == '__CTE__' and t in cte_col_maps]
        # Deduplicate by CTE name
        seen_cte: set = set()
        unique_ctes = []
        for s, t in cte_sources:
            if t not in seen_cte:
                seen_cte.add(t); unique_ctes.append((s, t))

        if len(unique_ctes) == 1:
            cte_name = unique_ctes[0][1]
            cte_map = cte_col_maps.get(cte_name, {})
            r = cte_map.get(col.lower())
            if r:
                return r
            if cte_name in all_ctes:
                inline_map = _build_cte_col_map(cte_name, all_ctes[cte_name], all_ctes)
                r2 = inline_map.get(col.lower())
                if r2:
                    return r2

        # Try all CTEs in scope for column lookup
        for _, cte_name in unique_ctes:
            cte_map = cte_col_maps.get(cte_name, {})
            r = cte_map.get(col.lower())
            if r:
                return r

        return [('', '', col)]


# ===========================================================================
#  STATEMENT CLASSIFIER
# ===========================================================================

def _classify(stmt: str) -> str:
    s = stmt.lstrip().upper()
    if re.match(r'CREATE\s+(?:TEMP(?:ORARY)?\s+)?TABLE', s):
        if re.search(r'\bAS\s*(?:\bWITH\b|\bSELECT\b|\()', stmt, re.IGNORECASE):
            return 'CTA'   # CREATE TABLE AS
        return 'CTD'       # CREATE TABLE DDL
    if re.match(r'INSERT\s+INTO', s):
        return 'INSERT'
    return 'OTHER'



# ===========================================================================
#  DDL COLUMN EXTRACTION  (for CREATE TABLE col_name type, ... patterns)
# ===========================================================================

def _extract_ddl_columns(stmt: str) -> List[str]:
    """
    Extract ordered column names from a CREATE TABLE DDL statement.
    Returns list of column names in definition order, empty if not a plain DDL.
    """
    s = re.sub(r'\{\{[^}]+\}\}', '__JINJA__', stmt)
    m = re.search(
        r'CREATE\s+(?:TEMP(?:ORARY)?\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?'
        r'(?:\w+\.)?\w+\s*\(',
        s, re.IGNORECASE
    )
    if not m:
        return []
    paren_start = m.end() - 1
    paren_end   = _find_paren_end(s, paren_start)
    if paren_end < 0:
        return []
    body = s[paren_start + 1 : paren_end]
    cols: List[str] = []
    depth = 0
    current: List[str] = []
    _SKIP_KW = {'PRIMARY', 'UNIQUE', 'CHECK', 'FOREIGN', 'CONSTRAINT',
                'PARTITION', 'SUBPARTITION', 'DEFAULT'}
    for ch in body:
        if ch == '(':
            depth += 1; current.append(ch)
        elif ch == ')':
            depth -= 1; current.append(ch)
        elif ch == ',' and depth == 0:
            col_def = ''.join(current).strip(); current = []
            if not col_def: continue
            toks = col_def.split()
            if not toks or toks[0].upper() in _SKIP_KW: continue
            cn = toks[0].strip('"').strip('`').strip()
            if cn and re.match(r'^[A-Za-z_]\w*$', cn):
                cols.append(cn.lower())
        else:
            current.append(ch)
    if current:
        col_def = ''.join(current).strip()
        toks = col_def.split()
        if toks and toks[0].upper() not in _SKIP_KW:
            cn = toks[0].strip('"').strip('`').strip()
            if cn and re.match(r'^[A-Za-z_]\w*$', cn):
                cols.append(cn.lower())
    return cols

# ===========================================================================
#  FINAL TABLE DETECTION
# ===========================================================================

def _find_final_table(stmts: List[str], file_stem: str) -> Tuple[str, str]:
    creates, inserts = [], []
    for stmt in stmts:
        kind = _classify(stmt)
        is_temp = bool(re.search(r'\bTEMP(?:ORARY)?\b', stmt[:80], re.IGNORECASE))
        m = re.search(r'(?:CREATE\s+(?:TEMP(?:ORARY)?\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?|INSERT\s+INTO\s+)' + _TBL_TOK,
                      stmt, re.IGNORECASE)
        if not m:
            continue
        schema, tbl = _parse_table_token(m.group(1))
        if kind in ('CTA', 'CTD'):
            creates.append((is_temp, schema, tbl))
        elif kind == 'INSERT':
            inserts.append((schema, tbl))

    for it, s, t in creates:
        if not it and t.lower() == file_stem.lower():
            return s, t
    for s, t in inserts:
        if t.lower() == file_stem.lower():
            return s, t
    non_temp = [(s, t) for it, s, t in creates if not it]
    if non_temp:
        return non_temp[-1]
    if inserts:
        return inserts[-1]
    if creates:
        return creates[-1][1], creates[-1][2]
    return '', file_stem


# ===========================================================================
#  WHERE / HAVING / JOIN EXTRACTION
# ===========================================================================

def _find_top_level_where(query: str) -> int:
    """Find top-level WHERE (not inside parens/quotes). Returns index or -1."""
    depth = 0
    i = 0
    n = len(query)
    while i < n:
        c = query[i]
        if c == "'":
            i += 1
            while i < n and query[i] != "'":
                if query[i] == '\\': i += 1
                i += 1
        elif c == '"':
            i += 1
            while i < n and query[i] != '"': i += 1
        elif c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        elif depth == 0 and (c == 'W' or c == 'w'):
            if _RC_WHERE.match(query, i):
                b = query[i-1] if i > 0 else ' '
                a = query[i+5] if i+5 < n else ' '
                if not (b.isalnum() or b == '_') and not (a.isalnum() or a == '_'):
                    return i
        i += 1
    return -1


def _find_top_level_kw_pos(query: str, kw: str) -> int:
    """Find top-level keyword. Compiled patterns cached in _P_KW_CACHE."""
    kw_u = kw.upper()
    if kw_u not in _P_KW_CACHE:
        w0 = kw.split()[0]
        _P_KW_CACHE[kw_u] = (
            w0[0].upper(), w0[0].lower(),
            re.compile(r'\b' + re.escape(w0) + r'\b', re.IGNORECASE),
            re.compile(r'\b' + kw.replace(' ', r'\s+') + r'\b', re.IGNORECASE),
        )
    fc_u, fc_l, fp, fkp = _P_KW_CACHE[kw_u]
    depth, i, n = 0, 0, len(query)
    while i < n:
        c = query[i]
        if c == "'":
            i += 1
            while i < n:
                if query[i] == "'": break
                if query[i] == '\\': i += 1
                i += 1
        elif c == '"':
            i += 1
            while i < n and query[i] != '"': i += 1
        elif c == '(': depth += 1
        elif c == ')': depth -= 1
        elif depth == 0 and (c == fc_u or c == fc_l):
            if fp.match(query, i) and fkp.match(query, i):
                b = query[i-1] if i > 0 else ' '
                if not (b.isalnum() or b == '_'):
                    return i
        i += 1
    return -1
def _where_text(query: str) -> str:
    pos = _find_top_level_where(query)
    if pos == -1:
        return ''
    start = pos + 5  # len("WHERE")
    # Find end of WHERE clause at top level
    stop_kws = ['GROUP BY', 'HAVING', 'ORDER BY', 'LIMIT', 'DISTRIBUTED', 'UNION', 'INTERSECT', 'EXCEPT']
    stop_pos = len(query)
    for kw in stop_kws:
        p = _find_top_level_kw_pos(query[start:], kw)
        if p != -1 and start + p < stop_pos:
            stop_pos = start + p
    body = query[start:stop_pos].strip()
    return ('WHERE ' + body) if body else ''


def _having_text(query: str) -> str:
    pos = _find_top_level_kw_pos(query, 'HAVING')
    if pos == -1:
        return ''
    start = pos + 6  # len("HAVING")
    stop_kws = ['ORDER BY', 'LIMIT', 'DISTRIBUTED', 'UNION', 'INTERSECT', 'EXCEPT']
    stop_pos = len(query)
    for kw in stop_kws:
        p = _find_top_level_kw_pos(query[start:], kw)
        if p != -1 and start + p < stop_pos:
            stop_pos = start + p
    body = query[start:stop_pos].strip()
    return ('HAVING ' + body) if body else ''



def _extract_joins(query: str) -> List[Tuple[str, str, str, str, str]]:
    """Returns [(keyword, tbl_raw, alias, on_clause, full_text), ...]."""
    pat = re.compile(
        r'((?:LEFT|RIGHT|FULL|INNER|CROSS)?\s*(?:OUTER\s+)?JOIN)\s+'
        r'(' + _TBL_TOK[1:-1] + r')'
        r'(?:\s+(?:AS\s+)?(\w+))?'
        r'(?:\s+ON\s+(.*?))?'
        r'(?=\s*(?:LEFT|RIGHT|FULL|INNER|CROSS|WHERE|GROUP|HAVING|ORDER|LIMIT|DISTRIBUTED|UNION|;|$))',
        re.IGNORECASE | re.DOTALL
    )
    results = []
    for m in pat.finditer(query):
        results.append((
            m.group(1).strip(), m.group(2).strip(),
            (m.group(3) or '').strip(), (m.group(4) or '').strip(),
            m.group(0).strip()
        ))
    return results


def _clause_col_rows(clause_text: str, sql_proc: str,
                     aliases: Dict[str, Tuple[str, str]],
                     sub_tbl: str, sub_schema: str,
                     final_tbl: str, final_schema: str,
                     file: str, path: str, process: str, now_ts: datetime
                     ) -> List[Dict[str, Any]]:
    """Extract column-level rows from a WHERE / HAVING / ON clause.
    Emits one row per alias.col reference with resolved source_table/schema.
    Skips numeric literals, keywords, function names, and bare alias names.
    Also emits rows for plain (unaliased) column references in WHERE/HAVING.
    """
    rows = []
    seen = set()

    # ── 1. Prefixed: alias.col ─────────────────────────────────────────────
    for m in _RC_ALIAS_DOT.finditer(clause_text):
        pfx, col = m.group(1).lower(), m.group(2)
        if pfx.upper() in SQL_KEYWORDS or pfx.upper() in SQL_FUNCTIONS:
            continue
        # Skip numeric-only column names
        if re.match(r'^\d+$', col):
            continue
        if col.upper() in SQL_KEYWORDS or col.upper() in SQL_FUNCTIONS:
            continue
        schema, tbl = aliases.get(pfx, ('', pfx))
        if schema == '__CTE__':
            schema = ''
        if not tbl or tbl.upper() in SQL_KEYWORDS:
            continue
        key = (tbl, col)
        if key in seen:
            continue
        seen.add(key)
        rows.append(_make_row(
            file, path, final_tbl, final_schema, process, now_ts,
            sub_tbl, sub_schema, '', tbl, schema, col,
            clause_text[:500], sql_proc,
        ))

    # ── 2. Plain (unaliased) column references ─────────────────────────────
    # Remove string literals, numbers, jinja params, and aliased refs first
    cleaned = _P_SQ.sub(' ', clause_text)
    cleaned = _P_JINJA.sub(' ', cleaned)
    cleaned = _P_CAST.sub(' ', cleaned)
    cleaned = _P_DOT_ANY.sub(' ', cleaned)
    cleaned = _P_NUM.sub(' ', cleaned)

    # Collect real tables (non-CTE)
    real_tbls = [(s, t) for t, (s, _t) in aliases.items()
                 if s not in ('__CTE__', '') and t == _t]

    for m in _RC_WORDS.finditer(cleaned):
        w = m.group(1)
        _HARD_KW = {
            'SELECT','FROM','WHERE','AND','OR','NOT','IN','IS','CASE','WHEN',
            'THEN','ELSE','END','AS','ON','JOIN','LEFT','RIGHT','INNER','OUTER',
            'FULL','CROSS','GROUP','BY','ORDER','HAVING','LIMIT','OFFSET',
            'DISTINCT','ALL','UNION','INTERSECT','EXCEPT','WITH','NULL',
            'TRUE','FALSE','BETWEEN','LIKE','ILIKE','FILTER','OVER','PARTITION',
            'NULLS','LAST','FIRST','ASC','DESC','DISTRIBUTED','ARRAY',
        }
        if w.upper() in _HARD_KW or w.upper() in SQL_FUNCTIONS:
            continue
        # Skip if it's a known alias (table alias names are not column names)
        if w.lower() in aliases and aliases[w.lower()][0] != '__CTE__':
            continue
        key = ('', w)
        if key in seen:
            continue
        seen.add(key)
        # Single-table inference
        if len(real_tbls) == 1:
            tbl_s, tbl_t = real_tbls[0]
        else:
            tbl_s, tbl_t = '', ''
        rows.append(_make_row(
            file, path, final_tbl, final_schema, process, now_ts,
            sub_tbl, sub_schema, '', tbl_t, tbl_s, w,
            clause_text[:500], sql_proc,
        ))

    return rows


def _make_row(file, path, target_table, target_schema, process, now_ts,
              sub_tbl, sub_schema, target_col, src_tbl, src_schema, src_col,
              logic, sql_process) -> Dict[str, Any]:
    return {
        'file': file, 'path': path,
        'target_table': target_table, 'target_schema': target_schema,
        'process': process, 'current_date_time': now_ts,
        'sub_target_table': sub_tbl, 'sub_target_schema': sub_schema,
        'target_column': target_col,
        'source_table': src_tbl, 'source_schema': src_schema, 'source_column': src_col,
        'logic': logic, 'sql_process': sql_process,
    }


# ===========================================================================
#  MAIN PER-FILE EXTRACTOR
# ===========================================================================

def _get_table_columns(schema: str, table: str) -> List[str]:
    """
    Fetch column names for a table from information_schema.
    Resolves Jinja schema params (IKG_SCHEMA -> core_ikg) automatically.
    """
    actual_schema = _resolve_schema_label(schema) if schema else ''
    key = (actual_schema.lower(), table.lower())
    if key in _INFO_SCHEMA_CACHE:
        return _INFO_SCHEMA_CACHE[key]
    if _DB_ENGINE is None:
        _INFO_SCHEMA_CACHE[key] = []
        return []
    try:
        import pandas as _pd
        if actual_schema:
            q = (f"SELECT column_name FROM information_schema.columns "
                 f"WHERE table_schema = '{actual_schema}' AND table_name = '{table}' "
                 f"ORDER BY ordinal_position")
        else:
            q = (f"SELECT column_name FROM information_schema.columns "
                 f"WHERE table_name = '{table}' ORDER BY ordinal_position LIMIT 200")
        result = _pd.read_sql(q, _DB_ENGINE)
        cols = result['column_name'].tolist()
        _INFO_SCHEMA_CACHE[key] = cols
        return cols
    except Exception as e:
        logger.debug(f"info_schema lookup failed for {actual_schema}.{table}: {e}")
        _INFO_SCHEMA_CACHE[key] = []
        return []


def _find_column_in_tables(col_name: str,
                           candidate_tables: List[Tuple[str, str]]) -> Tuple[str, str]:
    """
    Query information_schema to find which table among candidates owns col_name.
    Returns (schema, table) of the first match, or ('', '') if not found / no DB.
    """
    if not candidate_tables or _DB_ENGINE is None:
        return '', ''
    try:
        import pandas as _pd
        tbl_list = list({t for s, t in candidate_tables
                         if t and t.upper() not in {'SELECT','FROM','WHERE','JOIN','ON'}})
        if not tbl_list:
            return '', ''
        tbl_in = ', '.join(f"'{t}'" for t in tbl_list)
        q = (f"SELECT table_schema AS source_schema, table_name AS source_table "
             f"FROM information_schema.columns "
             f"WHERE column_name = '{col_name}' AND table_name IN ({tbl_in}) "
             f"LIMIT 1")
        result = _pd.read_sql(q, _DB_ENGINE)
        if not result.empty:
            return result['source_schema'].iloc[0], result['source_table'].iloc[0]
    except Exception as e:
        logger.debug(f"Column lookup failed for {col_name}: {e}")
    return '', ''


def _split_union_branches(sql: str) -> List[str]:
    """
    Split a SELECT (or CTE body) on top-level UNION [ALL] / INTERSECT / EXCEPT.
    Returns list of individual SELECT branch strings.
    """
    branches = []
    _UNION_PAT = re.compile(
        r'\bUNION(?:\s+ALL)?\b|\bINTERSECT\b|\bEXCEPT\b',
        re.IGNORECASE
    )
    depth = 0
    i = 0
    n = len(sql)
    start = 0
    while i < n:
        c = sql[i]
        if c == "'":
            i += 1
            while i < n:
                if sql[i] == "'": break
                if sql[i] == '\\': i += 1
                i += 1
        elif c == '"':
            i += 1
            while i < n and sql[i] != '"': i += 1
        elif c == '(':
            depth += 1
        elif c == ')':
            if depth > 0: depth -= 1
        elif depth == 0:
            m = _UNION_PAT.match(sql, i)
            if m:
                b = sql[i-1] if i > 0 else ' '
                if not (b.isalnum() or b == '_'):
                    chunk = sql[start:i].strip()
                    if chunk:
                        branches.append(chunk)
                    start = m.end()
                    i = m.end()
                    continue
        i += 1
    chunk = sql[start:].strip()
    if chunk:
        branches.append(chunk)
    return branches if len(branches) > 1 else [sql]


def _extract_inline_subquery_aliases(query: str) -> Dict[str, Tuple[str, str, str]]:
    """
    Find inline subqueries in JOIN/FROM clauses:
      JOIN (SELECT ... FROM schema.real_table) alias
    Returns {alias_lower: (schema, real_table, 'subquery')}
    so _extract_aliases can register them correctly.
    """
    result: Dict[str, Tuple[str, str, str]] = {}
    # Pattern: FROM/JOIN followed by ( ... ) alias
    pat = re.compile(
        r'\b(?:FROM|(?:LEFT|RIGHT|FULL|INNER|CROSS)?\s*(?:OUTER\s+)?JOIN)\s*\(',
        re.IGNORECASE
    )
    i = 0
    n = len(query)
    while i < n:
        m = pat.search(query, i)
        if not m:
            break
        # Find the matching closing paren
        paren_start = query.index('(', m.start())
        paren_end = _find_paren_end(query, paren_start)
        inner_sql = query[paren_start+1:paren_end].strip()
        # Get alias after the closing paren
        after = query[paren_end+1:].lstrip()
        alias_m = re.match(r'(?:AS\s+)?(\w+)', after, re.IGNORECASE)
        if alias_m:
            alias = alias_m.group(1).lower()
            # Extract all real tables from the inner subquery
            inner_from = _from_tables_raw(inner_sql)
            for schema, tbl in inner_from:
                result[alias] = (schema, tbl, 'subquery')
                break  # take first real table as the representative
        i = paren_end + 1
    return result


def _from_tables_raw(query: str) -> List[Tuple[str, str]]:
    """Like _from_tables but skips registering aliases — direct FROM/JOIN table list."""
    seen, result = set(), []
    for m in _RC_TBLREF.finditer(query):
        tbl_raw = m.group(2).strip()
        if tbl_raw.upper() in SQL_KEYWORDS or tbl_raw.upper() in SQL_FUNCTIONS:
            continue
        if tbl_raw.startswith('('):
            continue
        schema, tbl = _parse_table_token(tbl_raw)
        key = (schema, tbl)
        if key not in seen and tbl:
            seen.add(key)
            result.append((schema, tbl))
    return result


def _expand_inline_subqueries(
    query: str,
    virtual_ctes: dict,
    virtual_cte_col_maps: dict,
    all_ctes: dict,
    _depth: int = 0,
) -> None:
    """
    Find FROM (SELECT ...) alias and JOIN (SELECT ...) alias patterns.
    Register each as a virtual CTE so outer columns can resolve through it.
    Mutates virtual_ctes and virtual_cte_col_maps in-place.
    """
    if _depth > 2:   # guard against infinite recursion on deeply nested subqueries
        return
    pat = re.compile(
        r'\b(?:FROM|(?:LEFT|RIGHT|FULL|INNER|CROSS)?\s*(?:OUTER\s+)?JOIN)\s*\(',
        re.IGNORECASE
    )
    i = 0
    n = len(query)
    while i < n:
        m = pat.search(query, i)
        if not m:
            break
        paren_start = query.index('(', m.start())
        paren_end = _find_paren_end(query, paren_start)
        inner_sql = query[paren_start+1:paren_end].strip()
        if not re.search(r'\bSELECT\b', inner_sql, re.IGNORECASE):
            i = paren_end + 1
            continue
        after = query[paren_end+1:].lstrip()
        alias_m = re.match(r'(?:AS\s+)?(\w+)', after, re.IGNORECASE)
        if alias_m:
            alias = alias_m.group(1).lower()
            if alias.upper() not in SQL_KEYWORDS and alias not in virtual_ctes:
                virtual_ctes[alias] = inner_sql
                merged = {**all_ctes, **virtual_ctes}
                try:
                    virtual_cte_col_maps[alias] = _build_cte_col_map(alias, inner_sql, merged)
                except RecursionError:
                    virtual_cte_col_maps[alias] = {}
        i = paren_end + 1


def _parse_file_worker(args: tuple):
    """Top-level worker for multiprocessing — must be picklable."""
    from pathlib import Path as _P2
    fpath, content = args
    try:
        rows = extract_lineage_from_sql(content, _P2(fpath).name, fpath, _P2(fpath).parent.name)
        return rows, None
    except Exception as e:
        return [], {'file': fpath, 'error': str(e)}


def parse_lineage_parallel(file_dict: dict, n_workers: int = 0) -> tuple:
    """
    Fast parallel lineage parser for Jupyter notebooks.
    Drop-in replacement for the slow serial loop.

    Usage in notebook::

        all_rows, parse_errors = parse_lineage_parallel(file_dict)
    """
    import concurrent.futures as _cf
    import multiprocessing as _mp

    items = list(file_dict.items())
    n = len(items)
    if n_workers <= 0:
        n_workers = min(_mp.cpu_count(), 8)

    all_rows: list = []
    parse_errors: list = []

    try:
        chunksize = max(1, n // (n_workers * 4))
        with _cf.ProcessPoolExecutor(max_workers=n_workers) as pool:
            for done, (rows, err) in enumerate(pool.map(_parse_file_worker, items, chunksize=chunksize), 1):
                all_rows.extend(rows)
                if err: parse_errors.append(err)
                if done % 100 == 0: print(f'  {done}/{n}...', end='\r')
    except Exception as exc:
        logger.warning(f"Parallel failed ({exc}), running serially...")
        all_rows, parse_errors = [], []
        for item in items:
            rows, err = _parse_file_worker(item)
            all_rows.extend(rows)
            if err: parse_errors.append(err)

    print(f'  Done: {n} files parsed, {len(all_rows)} records, {len(parse_errors)} errors.      ')
    return all_rows, parse_errors


def extract_lineage_from_sql(
    sql_content: str,
    file_name_with_ext: str,
    file_path: str,
    process: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    now_ts = datetime.now()
    file_stem = Path(file_name_with_ext).stem

    normalized = _normalize(sql_content)
    stmts = _split_stmts(normalized)

    final_schema, final_table = _find_final_table(stmts, file_stem)

    def make(sub_tbl='', sub_schema='', target_col='', src_tbl='', src_schema='',
             src_col='', logic='', sql_proc='select') -> Dict[str, Any]:
        return _make_row(file_name_with_ext, file_path, final_table, final_schema,
                        process, now_ts, sub_tbl, sub_schema, target_col,
                        src_tbl, src_schema, src_col, logic, sql_proc)

    # ── Pre-scan: collect DDL column definitions for each named table ──────
    # When INSERT INTO has no explicit column list, use these as target_column
    _ddl_cols_map: Dict[str, List[str]] = {}   # lower(table_name) -> [col, ...]
    for _stmt in stmts:
        if _classify(_stmt) == 'CTD':
            _m = re.search(
                r'CREATE\s+(?:TEMP(?:ORARY)?\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?' + _TBL_TOK,
                _stmt, re.IGNORECASE
            )
            if _m:
                _, _tbl = _parse_table_token(_m.group(1))
                _cols = _extract_ddl_columns(_stmt)
                if _tbl and _cols:
                    _ddl_cols_map[_tbl.lower()] = _cols

    for stmt in stmts:
        kind = _classify(stmt)
        if kind not in ('CTA', 'INSERT'):
            continue

        # ── Identify sub_target and get select body ───────────────────
        _insert_target_cols: List[str] = []   # populated below for INSERT kind
        if kind == 'CTA':
            m = re.search(
                r'CREATE\s+(?:TEMP(?:ORARY)?\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?' + _TBL_TOK + r'\s+AS\s*',
                stmt, re.IGNORECASE
            )
            if not m:
                continue
            sub_schema, sub_tbl = _parse_table_token(m.group(1))
            select_body = stmt[m.end():]

        else:  # INSERT
            m = re.search(r'INSERT\s+INTO\s+' + _TBL_TOK, stmt, re.IGNORECASE)
            if not m:
                continue
            sub_schema, sub_tbl = _parse_table_token(m.group(1))
            select_body = stmt[m.end():].lstrip()
            # Check for explicit column list: INSERT INTO tbl (col1, col2, ...)
            _insert_explicit_cols: List[str] = []
            if select_body.startswith('('):
                ce = _find_paren_end(select_body, 0)
                _col_list_candidate = select_body[1:ce].strip()
                # It's a column list if it contains only identifiers and commas (no SELECT)
                if not re.search(r'\bSELECT\b', _col_list_candidate, re.IGNORECASE):
                    _insert_explicit_cols = [
                        c.strip().strip('"').strip() for c in _col_list_candidate.split(',')
                        if c.strip()
                    ]
                select_body = select_body[ce+1:].lstrip()
            # Determine effective target column list for this INSERT:
            # priority: explicit col list > DDL col list > derive from SELECT aliases
            _insert_target_cols: List[str] = (
                _insert_explicit_cols if _insert_explicit_cols
                else _ddl_cols_map.get(sub_tbl.lower(), [])
            )

        # ── CTEs from select body ─────────────────────────────────────
        ctes, outer_sel = _extract_ctes(select_body)

        outer_aliases = _extract_aliases(outer_sel)
        for cn in ctes:
            outer_aliases[cn] = ('__CTE__', cn)

        all_ctes = dict(ctes)
        cte_col_maps: Dict[str, Dict[str, List[Tuple[str, str, str]]]] = {}
        for cn, cb in ctes.items():
            cte_col_maps[cn] = _build_cte_col_map(cn, cb, all_ctes)

        all_from_tables = _from_tables(outer_sel)

        # ── Emit CTE sub-target rows ──────────────────────────────────
        for cte_name, cte_body in ctes.items():
            _emit_cte_rows(
                cte_name, cte_body, all_ctes,
                file_name_with_ext, file_path, final_table, final_schema,
                process, now_ts, rows
            )

        # ── SELECT list of outer query (handles UNION ALL) ───────────────
        outer_branches = _split_union_branches(outer_sel)
        for outer_branch in outer_branches:
            branch_aliases_outer = _extract_aliases(outer_branch)
            for cn in all_ctes:
                branch_aliases_outer[cn] = ('__CTE__', cn)
            branch_tbls_outer = _from_tables_raw(outer_branch) or all_from_tables


            # Register inline subqueries  FROM/JOIN (SELECT ...) alias  as virtual CTEs
            virtual_ctes: dict = {}
            virtual_cte_col_maps: dict = {}
            _expand_inline_subqueries(outer_branch, virtual_ctes, virtual_cte_col_maps, all_ctes, _depth=0)
            merged_ctes = {**all_ctes, **virtual_ctes}
            merged_col_maps = {**cte_col_maps, **virtual_cte_col_maps}
            for vn in virtual_ctes:
                branch_aliases_outer[vn] = ('__CTE__', vn)
            # Emit lineage rows for each virtual CTE (inner subquery)
            for vn, vbody in virtual_ctes.items():
                _emit_cte_rows(vn, vbody, merged_ctes,
                               file_name_with_ext, file_path, final_table, final_schema,
                               process, now_ts, rows)

            sel_list = _extract_select_list(outer_branch)
            if sel_list:
                _sel_exprs = [e.strip() for e in _split_comma(sel_list) if e.strip()]
                _expr_idx  = 0   # position counter for DDL column mapping (Fix 1)
                for expr in _sel_exprs:
                    # ── Handle SELECT * or a.* ─────────────────────────────
                    if re.match(r'^(\w+\.)?\*$', expr.strip()):
                        star_pfx_m = re.match(r'^(\w+)\.\*$', expr.strip())
                        if star_pfx_m:
                            pfx = star_pfx_m.group(1).lower()
                            s_schema, s_tbl = branch_aliases_outer.get(pfx, ('', pfx))
                            if s_schema == '__CTE__': s_schema = ''
                            source_tbls = [(s_schema, s_tbl)] if s_tbl else []
                        else:
                            source_tbls = [(s, t) for s, t in branch_tbls_outer
                                           if t and t.lower() not in all_ctes]
                            s_schema, s_tbl = source_tbls[0] if len(source_tbls) == 1 else ('', '')

                        from_subq_m = re.search(
                            r'\bFROM\s*\((.+?)\)\s*(?:AS\s+)?(\w+)\s*(?:;|$|\bWHERE\b|\bJOIN\b)',
                            outer_branch, re.IGNORECASE | re.DOTALL
                        )
                        if from_subq_m and not source_tbls:
                            inner_sql = from_subq_m.group(1).strip()
                            inner_from_tables = _from_tables(inner_sql)
                            inner_aliases = _extract_aliases(inner_sql)
                            inner_ctes2, inner_body2 = _extract_ctes(inner_sql)
                            merged2 = {**all_ctes, **inner_ctes2}
                            for cn in merged2: inner_aliases[cn] = ('__CTE__', cn)
                            inner_sel2 = _extract_select_list(inner_body2)
                            if inner_sel2:
                                for ie in _split_comma(inner_sel2):
                                    ie = ie.strip()
                                    if not ie: continue
                                    i_alias, i_logic = _split_alias(ie)
                                    bm = re.match(r'^(?:\w+\.)?(\w+)(?:::\w+)?$', (i_logic or ie).strip())
                                    i_tgt = i_alias or (bm.group(1) if bm else '')
                                    i_refs = _extract_col_refs_from_expr(i_logic or ie)
                                    if i_refs:
                                        for ra, rc in i_refs:
                                            rl = _resolve_col_ref(ra, rc, inner_aliases, merged2, {}, inner_from_tables)
                                            for rs, rt, rcol in rl:
                                                if not rt and inner_from_tables:
                                                    rs2, rt2 = _find_column_in_tables(rcol, inner_from_tables)
                                                    if rt2: rs, rt = rs2, rt2
                                                rows.append(make(sub_tbl, sub_schema, i_tgt, rt, rs, rcol, ie, 'select*'))
                                    else:
                                        rows.append(make(sub_tbl, sub_schema, i_tgt, '', '', i_tgt, ie, 'select*'))
                            inner_wh = _where_text(inner_body2)
                            if inner_wh:
                                rows.append(make(sub_tbl, sub_schema, '', '', '', '', inner_wh[:2000], 'where-subquery'))
                                rows.extend(_clause_col_rows(inner_wh, 'where-subquery', inner_aliases,
                                                             sub_tbl, sub_schema, final_table, final_schema,
                                                             file_name_with_ext, file_path, process, now_ts))
                            continue

                        # Case B: expand via information_schema
                        expanded = False
                        for ts_schema, ts_tbl in (source_tbls if source_tbls else [(s_schema, s_tbl)]):
                            cols_from_db = _get_table_columns(ts_schema, ts_tbl) if ts_tbl else []
                            if cols_from_db:
                                for col in cols_from_db:
                                    rows.append(make(sub_tbl, sub_schema, col, ts_tbl,
                                                     _resolve_schema_label(ts_schema), col, col, 'select*'))
                                expanded = True
                        if not expanded:
                            rows.append(make(sub_tbl, sub_schema, '*', s_tbl,
                                             _resolve_schema_label(s_schema), '*', expr, 'select*'))
                        continue

                    partial_rows = _analyse_expr(
                        expr, branch_aliases_outer, merged_ctes, merged_col_maps, branch_tbls_outer
                    )
                    for pr in partial_rows:
                        # ── Fix 1: override target_column from DDL col list ──────────
                        # When INSERT INTO has no explicit column list, use the CREATE
                        # TABLE DDL column names positionally as target_column.
                        tgt_col = pr['target_column']
                        if kind == 'INSERT' and _insert_target_cols:
                            if _expr_idx < len(_insert_target_cols):
                                tgt_col = _insert_target_cols[_expr_idx]

                        # ── Fix 2: trace through inline subquery col map ─────────────
                        # If the source resolves to an inline subquery alias, look up
                        # the real source_column and source_table from the inner select.
                        # The inner map may itself reference a table alias — resolve that
                        # one more level via branch_aliases_outer.
                        src_tbl    = pr['source_table']
                        src_schema = pr['source_schema']
                        src_col    = pr['source_column']
                        if src_tbl and src_tbl.lower() in virtual_cte_col_maps:
                            inner_map = virtual_cte_col_maps[src_tbl.lower()]
                            # Try lookup by source_col first, then by target_col
                            resolved  = inner_map.get(src_col.lower()) or inner_map.get(tgt_col.lower())
                            if resolved:
                                for r_schema, r_tbl, r_col in resolved:
                                    if not r_tbl:
                                        continue
                                    rl = r_tbl.lower()
                                    # If r_tbl is another virtual CTE alias, recurse one level
                                    if rl in virtual_cte_col_maps:
                                        inner2 = virtual_cte_col_maps[rl]
                                        resolved2 = inner2.get(r_col.lower())
                                        if resolved2:
                                            for r2_schema, r2_tbl, r2_col in resolved2:
                                                if r2_tbl and r2_tbl.lower() not in virtual_cte_col_maps:
                                                    src_schema = r2_schema
                                                    src_tbl    = r2_tbl
                                                    src_col    = r2_col
                                                    break
                                    elif rl in branch_aliases_outer:
                                        # r_tbl is a FROM-clause alias — dereference it
                                        alias_schema, alias_real_tbl = branch_aliases_outer[rl]
                                        if alias_schema != '__CTE__' and alias_real_tbl:
                                            src_schema = alias_schema
                                            src_tbl    = alias_real_tbl
                                            src_col    = r_col
                                        break
                                    else:
                                        # r_tbl is an actual table name
                                        src_schema = r_schema
                                        src_tbl    = r_tbl
                                        src_col    = r_col
                                        break

                        rows.append(make(
                            sub_tbl, sub_schema,
                            tgt_col, src_tbl, src_schema,
                            src_col, pr['logic'], pr['sql_process'],
                        ))
                    _expr_idx += 1   # advance after all partial_rows for this expression

            # ── WHERE ─────────────────────────────────────────────────
            wh = _where_text(outer_branch)
            if wh:
                rows.append(make(sub_tbl, sub_schema, '', '', '', '', wh[:2000], 'where'))
                rows.extend(_clause_col_rows(wh, 'where', branch_aliases_outer, sub_tbl, sub_schema,
                                             final_table, final_schema,
                                             file_name_with_ext, file_path, process, now_ts))

            # ── HAVING ────────────────────────────────────────────────
            hv = _having_text(outer_branch)
            if hv:
                rows.append(make(sub_tbl, sub_schema, '', '', '', '', hv[:2000], 'having'))
                rows.extend(_clause_col_rows(hv, 'having', branch_aliases_outer, sub_tbl, sub_schema,
                                             final_table, final_schema,
                                             file_name_with_ext, file_path, process, now_ts))

            # ── JOINs ─────────────────────────────────────────────────
            for jkw, tbl_raw, j_alias, on_clause, full_text in _extract_joins(outer_branch):
                j_schema, j_tbl = _parse_table_token(tbl_raw)
                rows.append(make(sub_tbl, sub_schema, '', j_tbl, j_schema, '', full_text[:2000], 'join'))
                if on_clause:
                    rows.extend(_clause_col_rows('ON ' + on_clause, 'join', branch_aliases_outer,
                                                 sub_tbl, sub_schema, final_table, final_schema,
                                                 file_name_with_ext, file_path, process, now_ts))
    return rows


def _emit_cte_rows(
    cte_name: str, cte_body: str, all_ctes: Dict[str, str],
    file: str, path: str, final_tbl: str, final_schema: str,
    process: str, now_ts: datetime, rows: List[Dict[str, Any]]
):
    """Parse one CTE body and emit rows with sub_target_table = cte_name."""
    inner_ctes, inner_body = _extract_ctes(cte_body)
    merged = {**all_ctes, **inner_ctes}

    aliases = _extract_aliases(cte_body)
    for cn in merged:
        aliases[cn] = ('__CTE__', cn)

    cte_col_maps_inner: Dict[str, Dict] = {
        cn: _build_cte_col_map(cn, cb, merged) for cn, cb in inner_ctes.items()
    }
    all_tbls = _from_tables(cte_body)

    # Handle UNION ALL / UNION — parse each branch independently
    branches = _split_union_branches(inner_body)

    for branch_sql in branches:
        branch_aliases = _extract_aliases(branch_sql)
        for cn in merged:
            branch_aliases[cn] = ('__CTE__', cn)
        branch_tbls = _from_tables_raw(branch_sql)
        if not branch_tbls:
            branch_tbls = all_tbls

        # Expand inline FROM (SELECT ...) subqueries as virtual CTEs
        v_ctes: dict = {}
        v_cte_maps: dict = {}
        _expand_inline_subqueries(branch_sql, v_ctes, v_cte_maps, merged, _depth=1)
        merged_v = {**merged, **v_ctes}
        cte_col_maps_inner_v = {**cte_col_maps_inner, **v_cte_maps}
        for vn in v_ctes:
            branch_aliases[vn] = ('__CTE__', vn)
        for vn, vbody in v_ctes.items():
            _emit_cte_rows(vn, vbody, merged_v,
                           file, path, final_tbl, final_schema, process, now_ts, rows)

        sel = _extract_select_list(branch_sql)
        if not sel:
            continue
        for expr in _split_comma(sel):
            expr = expr.strip()
            if not expr:
                continue
            if re.match(r'^(\w+\.)?\*$', expr.strip()):
                    star_pfx_m = re.match(r'^(\w+)\.\*$', expr.strip())
                    if star_pfx_m:
                        pfx = star_pfx_m.group(1).lower()
                        s_schema, s_tbl = branch_aliases.get(pfx, ('', pfx))
                        if s_schema == '__CTE__': s_schema = ''
                    elif len(branch_tbls) == 1:
                        s_schema, s_tbl = branch_tbls[0]
                    else:
                        s_schema, s_tbl = '', ''
                    cols_from_db = _get_table_columns(s_schema, s_tbl) if s_tbl else []
                    if cols_from_db:
                        for col in cols_from_db:
                            rows.append(_make_row(file, path, final_tbl, final_schema, process, now_ts,
                                                  cte_name, '', col, s_tbl, s_schema, col, expr, 'select*'))
                    else:
                        rows.append(_make_row(file, path, final_tbl, final_schema, process, now_ts,
                                              cte_name, '', '*', s_tbl, s_schema, '*', expr, 'select*'))
                    continue
            partial = _analyse_expr(expr, branch_aliases, merged_v, cte_col_maps_inner_v, branch_tbls)
            for pr in partial:
                rows.append(_make_row(file, path, final_tbl, final_schema, process, now_ts,
                                      cte_name, '',
                                      pr['target_column'], pr['source_table'], pr['source_schema'],
                                      pr['source_column'], pr['logic'], pr['sql_process']))

        wh = _where_text(branch_sql)
        if wh:
            rows.append(_make_row(file, path, final_tbl, final_schema, process, now_ts,
                                  cte_name, '', '', '', '', '', wh[:2000], 'where'))
            rows.extend(_clause_col_rows(wh, 'where', branch_aliases, cte_name, '',
                                         final_tbl, final_schema, file, path, process, now_ts))


# ===========================================================================
#  FILE LOADERS
# ===========================================================================

def _fetch_sql_files_from_gitlab(token: str) -> Dict[str, str]:
    import urllib.request, urllib.parse, json

    base = GITLAB_URL.rstrip('/')
    proj = urllib.parse.quote(IKG_PROJECT_PATH, safe='')
    hdrs = {"PRIVATE-TOKEN": token}
    all_items: List[dict] = []
    url: Optional[str] = (
        f"{base}/api/v4/projects/{proj}/repository/tree"
        f"?path={urllib.parse.quote(SQL_FOLDER_IN_REPO)}"
        f"&ref={DEFAULT_BASE_BRANCH}&recursive=true&per_page=1000"
    )
    while url:
        req = urllib.request.Request(url, headers=hdrs)
        with urllib.request.urlopen(req, timeout=30) as resp:
            all_items.extend(json.loads(resp.read().decode()))
            link = resp.headers.get('Link', '')
            url = None
            if 'rel="next"' in link:
                for part in link.split(','):
                    if 'rel="next"' in part:
                        url = part.split(';')[0].strip().strip('<>')

    sql_files = [x for x in all_items if x['type'] == 'blob' and x['path'].endswith('.sql')]
    logger.info(f"Found {len(sql_files)} SQL files on GitLab.")

    file_contents: Dict[str, str] = {}
    for item in sql_files:
        fpath = item['path']
        process = Path(fpath).parent.name
        if process in EXCLUDED_PROCESSES:
            continue
        raw_url = (f"{base}/api/v4/projects/{proj}/repository/files"
                   f"/{urllib.parse.quote(fpath, safe='')}/raw?ref={DEFAULT_BASE_BRANCH}")
        try:
            req = urllib.request.Request(raw_url, headers=hdrs)
            with urllib.request.urlopen(req, timeout=30) as resp:
                file_contents[fpath] = resp.read().decode('utf-8', errors='replace')
        except Exception as e:
            logger.warning(f"Could not fetch {fpath}: {e}")
    logger.info(f"Fetched {len(file_contents)} SQL files.")
    return file_contents


def _load_local_sql_files(sql_root: str) -> Dict[str, str]:
    files: Dict[str, str] = {}
    root = Path(sql_root)
    for p in root.rglob('*.sql'):
        process = p.parent.name
        if process in EXCLUDED_PROCESSES:
            continue
        try:
            try:
                rel = str(p.relative_to(root.parent.parent.parent))
            except ValueError:
                rel = str(p)
            files[rel] = p.read_text(encoding='utf-8', errors='replace')
        except Exception as e:
            logger.warning(f"Could not read {p}: {e}")
    logger.info(f"Loaded {len(files)} SQL files (after exclusions) from '{sql_root}'.")
    return files


# ===========================================================================
#  GREENPLUM WRITER
# ===========================================================================

def save_to_greenplum(df: pd.DataFrame, schema: str, password: str,
                      host=GREENPLUM_HOST, port=GREENPLUM_PORT,
                      db=GREENPLUM_DB, user=GREENPLUM_USER) -> bool:
    try:
        from sqlalchemy import create_engine
        from psycopg2 import sql as pgsql
    except ImportError:
        logger.error("sqlalchemy/psycopg2 not installed."); return False

    engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{db}")
    try:
        raw = engine.raw_connection()
        with raw.cursor() as cur:
            cur.execute(pgsql.SQL("DROP TABLE IF EXISTS {}.{}").format(
                pgsql.Identifier(schema), pgsql.Identifier(OUTPUT_TABLE)))
        raw.commit(); raw.close()
    except Exception as e:
        logger.warning(f"Drop: {e}")

    df.to_sql(OUTPUT_TABLE, engine, schema=schema, if_exists='replace', index=False, method='multi')

    try:
        raw = engine.raw_connection()
        with raw.cursor() as cur:
            cur.execute(pgsql.SQL("ALTER TABLE {}.{} OWNER TO erd_gpdb_prj_smart_insights").format(
                pgsql.Identifier(schema), pgsql.Identifier(OUTPUT_TABLE)))
            cur.execute(pgsql.SQL("GRANT SELECT ON {}.{} TO erd_gpdb_prj_smart_insights_ro").format(
                pgsql.Identifier(schema), pgsql.Identifier(OUTPUT_TABLE)))
        raw.commit(); raw.close()
    except Exception as e:
        logger.warning(f"Grant: {e}")

    engine.dispose()
    logger.info(f"✅ Saved {len(df)} rows to {schema}.{OUTPUT_TABLE}")
    return True


# ===========================================================================
#  EXCEL WRITER
# ===========================================================================

def save_to_excel(df: pd.DataFrame, output_path: str):
    logger.info(f"Writing Excel → {output_path}")
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Column_Lineage', index=False)

    wb = load_workbook(output_path)
    ws = wb['Column_Lineage']
    hdr_fill = PatternFill(start_color='1F4E79', end_color='1F4E79', fill_type='solid')
    hdr_font = Font(name='Arial', bold=True, color='FFFFFF', size=10)
    for cell in ws[1]:
        cell.fill = hdr_fill; cell.font = hdr_font
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)

    light = PatternFill(start_color='DCE6F1', end_color='DCE6F1', fill_type='solid')
    dfont = Font(name='Arial', size=9)
    for ri, row in enumerate(ws.iter_rows(min_row=2, max_row=ws.max_row), 2):
        for cell in row:
            cell.font = dfont
            if ri % 2 == 0: cell.fill = light

    for col_letter, width in zip('ABCDEFGHIJKLMN', [38,60,38,30,28,22,38,28,38,35,38,35,80,15]):
        ws.column_dimensions[col_letter].width = width

    ws.freeze_panes = 'A2'
    ws.auto_filter.ref = ws.dimensions
    wb.save(output_path)
    logger.info(f"✅ Excel saved: {output_path}")


# ===========================================================================
#  MAIN
# ===========================================================================

def run(private_token=None, pg_password=None, use_local_sql=False,
        local_sql_path=None, allow_prompt=True, **kwargs):
    global _DB_ENGINE
    logger.info("=" * 60)
    logger.info("IKG Column Lineage Master Auto Refresh")
    logger.info("=" * 60)

    schema = ensure_greenplum_schema()

    # Resolve DB password early so information_schema lookups work during parsing
    if pg_password is None and is_running_in_airflow():
        creds = get_greenplum_credentials()
        if creds:
            pg_password = creds.get('password')
            try:
                from sqlalchemy import create_engine as _ce
                _DB_ENGINE = _ce(
                    f"postgresql://{creds.get('user', GREENPLUM_USER)}:{pg_password}"
                    f"@{creds.get('host', GREENPLUM_HOST)}:{creds.get('port', GREENPLUM_PORT)}"
                    f"/{creds.get('db', GREENPLUM_DB)}"
                )
                logger.info("DB engine ready for information_schema lookups.")
            except Exception as e:
                logger.warning(f"Could not create DB engine for info_schema: {e}")
    elif pg_password:
        try:
            from sqlalchemy import create_engine as _ce
            _DB_ENGINE = _ce(
                f"postgresql://{GREENPLUM_USER}:{pg_password}"
                f"@{GREENPLUM_HOST}:{GREENPLUM_PORT}/{GREENPLUM_DB}"
            )
            logger.info("DB engine ready for information_schema lookups.")
        except Exception as e:
            logger.warning(f"Could not create DB engine: {e}")

    if use_local_sql:
        sql_path = local_sql_path or str(Path(__file__).parent / 'dags/ikg/scripts/sql')
        file_dict = _load_local_sql_files(sql_path)
    else:
        if private_token is None:
            private_token = get_private_token(allow_prompt=allow_prompt)
        file_dict = _fetch_sql_files_from_gitlab(private_token)

    def _parse_file(args):
        fpath, sql_content = args
        try:
            rows = extract_lineage_from_sql(
                sql_content, Path(fpath).name, fpath, Path(fpath).parent.name)
            return rows, None
        except Exception as e:
            return [], (fpath, str(e))

    all_rows, errors = [], []
    items = list(file_dict.items())
    try:
        import concurrent.futures as _cf
        import multiprocessing as _mp
        n_workers = min(_mp.cpu_count(), 8)
        logger.info(f"Parsing {len(items)} files using {n_workers} CPU workers...")
        with _cf.ProcessPoolExecutor(max_workers=n_workers) as pool:
            for i, (rows, err) in enumerate(pool.map(_parse_file, items, chunksize=20)):
                all_rows.extend(rows)
                if err:
                    errors.append(err)
                if (i + 1) % 200 == 0:
                    logger.info(f"  {i+1}/{len(items)} files parsed...")
    except Exception as e:
        logger.warning(f"Parallel parse failed ({e}), falling back to serial...")
        all_rows, errors = [], []
        for fpath, content in items:
            rows, err = _parse_file((fpath, content))
            all_rows.extend(rows)
            if err: errors.append(err)

    logger.info(f"Total records: {len(all_rows)}, Errors: {len(errors)}")
    if errors:
        for f, e in errors[:10]:
            logger.warning(f"  {f}: {e}")

    if not all_rows:
        logger.warning("No records found."); return None

    df = pd.DataFrame(all_rows, columns=COLUMN_ORDER).drop_duplicates()
    df['current_date_time'] = pd.to_datetime(df['current_date_time'])

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = f"ikg_column_lineage_master_auto_refresh_{ts}.xlsx"
    save_to_excel(df, out)

    if pg_password or is_running_in_airflow():
        if not pg_password:
            creds = get_greenplum_credentials()
            if creds: pg_password = creds.get('password')
        if pg_password:
            save_to_greenplum(df, schema, pg_password)
    elif allow_prompt:
        if input("Save to Greenplum? (yes/no): ").strip().lower() in ('yes', 'y'):
            pg_password = getpass.getpass("Password: ")
            save_to_greenplum(df, schema, pg_password)

    logger.info(f"✅ Done → {out}")
    return out


def generate_column_lineage(private_token=None, **kwargs):
    if private_token is None and is_running_in_airflow():
        private_token = get_private_token(allow_prompt=False)
    return run(private_token=private_token, allow_prompt=False)


if __name__ == "__main__":
    run(use_local_sql=True,
        local_sql_path=str(Path(__file__).parent / "dags/ikg/scripts/sql"),
        allow_prompt=True)
