# GitLab Projects Export  ── v5
#
# What changed from v4
# ─────────────────────
# HANG FIXES
#   • GITLAB_TIMEOUT        : every python-gitlab API call is wrapped with a
#                             requests Session that enforces (connect=10s, read=30s).
#                             A hung HTTP request will now raise instead of blocking forever.
#   • CONCURRENCY_SEMAPHORE : hard cap on simultaneous open GitLab connections so we never
#                             trigger rate-limiting / connection-pool exhaustion.
#   • Per-project timeout   : each Phase-2 future.result() has a 120 s wall-clock timeout.
#                             Projects that still hang are logged and skipped automatically.
#   • repository_tree       : get_all=True replaced with manual page iteration so a single
#                             giant tree cannot block indefinitely.
#   • Live progress line    : prints "P1: 45/200 done …" every 10 completions so you can
#                             see the script is still moving.
#
# USER GATE
#   • After Phase 1 completes the script prints a summary (total projects, how many have
#     package.json) and asks: "Proceed to import-statement scan? (yes/no)"
#     If the user types anything other than yes/y the script writes the Phase-1 results
#     to Greenplum and exits cleanly.
#
# GREENPLUM OUTPUT
#   • Replaces XLSX/JSON output with psycopg2 writes to Greenplum.
#   • Table : sandbox_prj_smart_insights.gwma_ui_projects
#   • DROP + CREATE on every run (idempotent).
#   • ALTER OWNER and GRANT SELECT applied after creation.
#   • Rows inserted in batches of DB_BATCH_SIZE using execute_values for speed.

import getpass
import json
import logging
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeout
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import gitlab
import psycopg2
import psycopg2.extras
import pandas as pd
import requests
from requests.adapters import HTTPAdapter

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger('gitlab_export')
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('gitlab').setLevel(logging.WARNING)

logger.info('=' * 70)
logger.info('GitLab Projects Export  v5  — started')
logger.info('=' * 70)

# ── Configuration ─────────────────────────────────────────────────────────────
GITLAB_URL    = 'https://devcloud.ubs.net'
GROUP_PATH    = 'ubs/gwma'

PER_PAGE      = 100          # GitLab max page size
MAX_WORKERS   = 10           # concurrent project threads (keep low to avoid rate-limit)
FILE_WORKERS  = 5            # concurrent file-fetch threads per project (keep low)
MAX_CONNECTIONS = 10         # hard cap on simultaneous GitLab HTTP connections
GITLAB_TIMEOUT  = (10, 30)   # (connect_timeout_s, read_timeout_s) per request
FUTURE_TIMEOUT  = 120        # seconds before a stuck project future is abandoned
PROGRESS_EVERY  = 10         # print a progress line every N completions

UPDATED_SINCE = datetime(2025, 1, 1, tzinfo=timezone.utc)

SOURCE_EXTENSIONS = ('.jsx', '.tsx', '.js', '.ts')

# Greenplum
DB_SCHEMA     = 'sandbox_prj_smart_insights'
DB_TABLE      = 'gwma_ui_projects'
DB_BATCH_SIZE = 500          # rows per INSERT batch

# ── Import regex — ALL ES-module import forms ─────────────────────────────────
ANY_IMPORT_RE = re.compile(
    r"^[ \t]*"
    r"import"
    r"(?:"
        r"\s+(?:"
            r"[\w\$_][\w\$_]*(?:\s*,\s*(?:\*\s+as\s+[\w\$_]+|\{[^}]*\}))?"
            r"|\*\s+as\s+[\w\$_][\w\$_]*"
            r"|\{[^}]*\}"
        r")"
        r"\s+from"
    r")?"
    r"\s*['\"]([^'\"]+)['\"]"
    r"\s*;?[^\n]*",
    re.MULTILINE,
)

# ── Thread-safe counters ──────────────────────────────────────────────────────
_stats_lock = threading.Lock()
run_stats: Dict[str, int] = {
    'total_in_group'          : 0,
    'recent_projects'         : 0,
    'skipped_too_old'         : 0,
    'has_package_json'        : 0,
    'skipped_no_package_json' : 0,
    'p1_timeout'              : 0,
    'source_files_fetched'    : 0,
    'source_files_empty'      : 0,
    'source_files_error'      : 0,
    'import_statements_total' : 0,
    'p2_timeout'              : 0,
    'output_rows'             : 0,
}

def _inc(key: str, n: int = 1) -> None:
    with _stats_lock:
        run_stats[key] += n

# ── Semaphore — hard cap on concurrent GitLab connections ─────────────────────
_gitlab_sem = threading.Semaphore(MAX_CONNECTIONS)

# ── GitLab client with per-request timeouts ───────────────────────────────────
logger.info('AUTH | requesting GitLab private token …')
private_token = getpass.getpass('Enter your GitLab private token: ')

# Build a requests Session that enforces read + connect timeouts
_session = requests.Session()
_adapter = HTTPAdapter(
    max_retries=1,
    pool_connections=MAX_CONNECTIONS,
    pool_maxsize=MAX_CONNECTIONS,
)
_session.mount('https://', _adapter)
_session.mount('http://',  _adapter)

# Monkey-patch a timeout into every request made by the session
_orig_request = _session.request
def _timed_request(method, url, **kwargs):
    kwargs.setdefault('timeout', GITLAB_TIMEOUT)
    return _orig_request(method, url, **kwargs)
_session.request = _timed_request

client = gitlab.Gitlab(GITLAB_URL, private_token=private_token, session=_session)
logger.info('AUTH | client ready (timeout=%s) for %s', GITLAB_TIMEOUT, GITLAB_URL)

# ── DB credentials ────────────────────────────────────────────────────────────
logger.info('DB | requesting Greenplum password …')
db_password = getpass.getpass('Enter Password for DB User: ')
DB_CONFIG = {
    'host'    : 'greenplum-rdsp.zur.swissbank.com',
    'port'    : '5432',
    'dbname'  : 'gprdsp',
    'user'    : 'ds_rdsp_dev',
    'password': db_password,
}
logger.info('DB | credentials stored (host=%s  db=%s  user=%s)',
            DB_CONFIG['host'], DB_CONFIG['dbname'], DB_CONFIG['user'])


# ── Utility helpers ───────────────────────────────────────────────────────────

def _parse_dt(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace('Z', '+00:00'))
    except Exception:
        return None


def is_recent(proj_ref: Any) -> bool:
    dt = _parse_dt(getattr(proj_ref, 'updated_at', None) or '')
    return True if dt is None else dt >= UPDATED_SINCE


def extract_team_name(web_url: str) -> str:
    try:
        segs = web_url.rstrip('/').split('//', 1)[-1].split('/')[1:]
        return segs[-2] if len(segs) >= 2 else ''
    except Exception:
        return ''


def _raw_file(project: Any, path: str, ref: str) -> Optional[str]:
    """Fetch one file as UTF-8; returns None on error or timeout."""
    with _gitlab_sem:
        try:
            raw = project.files.raw(file_path=path, ref=ref)
            return raw.decode('utf-8', errors='replace')
        except Exception as exc:
            logger.debug('FILE | err | %s | %s', path, exc)
            return None


def _repo_tree_pages(project: Any, ref: str) -> List[Dict]:
    """
    Manually page through repository_tree to avoid the all=True hang.
    Each page request is individually bounded by GITLAB_TIMEOUT.
    Returns a flat list of all tree items.
    """
    items = []
    page  = 1
    while True:
        with _gitlab_sem:
            try:
                batch = project.repository_tree(
                    ref=ref, recursive=True, per_page=PER_PAGE,
                    page=page, get_all=False,
                )
            except Exception as exc:
                logger.warning('TREE | page %d error | %s | %s',
                               page, project.path_with_namespace, exc)
                break
        if not batch:
            break
        items.extend(batch)
        if len(batch) < PER_PAGE:
            break    # last page
        page += 1
    return items


def _derive_library(module_path: str) -> str:
    s = module_path.lower()
    found = []
    if s.startswith('@uwr/'):
        found.append('uwr')
    if s.startswith('@ubs.websdk/'):
        found.append('websdk')
    return ', '.join(found)


# ── Phase 1: check package.json (one API call per project) ───────────────────

def check_package_json(proj_ref: Any) -> Optional[Dict[str, Any]]:
    """
    Phase-1 worker.  One raw-file fetch only — no full project.get().
    Returns metadata dict if package.json exists, else None.
    """
    ns  = proj_ref.path_with_namespace
    ref = getattr(proj_ref, 'default_branch', None) or 'main'

    with _gitlab_sem:
        try:
            raw     = proj_ref.files.raw(file_path='package.json', ref=ref)
            content = raw.decode('utf-8', errors='replace')
        except Exception:
            logger.debug('P1 | NO  pkg.json | %s', ns)
            _inc('skipped_no_package_json')
            return None

    try:
        deps: Dict[str, str] = json.loads(content).get('dependencies', {})
    except json.JSONDecodeError as exc:
        logger.warning('P1 | bad JSON | %s | %s', ns, exc)
        deps = {}

    dependency_str = ', '.join(sorted(deps.keys())) if deps else '-'
    uwr_deps       = [d for d in deps if d.startswith('@uwr/')]
    int_ext        = 'internal' if uwr_deps else 'external'

    logger.info('P1 | YES pkg.json | %-60s | %s | deps=%d',
                ns, int_ext, len(deps))
    _inc('has_package_json')

    return {
        'proj_ref'  : proj_ref,
        'ref'       : ref,
        'int_ext'   : int_ext,
        'dependency': dependency_str,
        # Phase-1 summary row fields (used if user stops after Phase 1)
        'name'               : getattr(proj_ref, 'name', ''),
        'path'               : getattr(proj_ref, 'path', ''),
        'path_with_namespace': ns,
        'web_url'            : getattr(proj_ref, 'web_url', ''),
        'visibility'         : getattr(proj_ref, 'visibility', ''),
        'created_at'         : getattr(proj_ref, 'created_at', ''),
        'updated_at'         : getattr(proj_ref, 'updated_at', ''),
        'default_branch'     : ref,
    }


# ── Phase 2: file scan + import extraction ────────────────────────────────────

def scan_and_build_rows(phase1: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Phase-2 worker.
    Fetches full project, pages the tree manually, reads source files in parallel.
    Returns one row per import statement (or one summary row if none found).
    """
    proj_ref   = phase1['proj_ref']
    ref        = phase1['ref']
    int_ext    = phase1['int_ext']
    dependency = phase1['dependency']
    ns         = proj_ref.path_with_namespace

    t0 = time.perf_counter()
    logger.info('P2 | START | %s', ns)

    # Full project object (needed for namespace dict and accurate web_url)
    with _gitlab_sem:
        try:
            project = client.projects.get(proj_ref.id)
        except Exception as exc:
            logger.error('P2 | project.get FAILED | %s | %s', ns, exc)
            return []

    namespace = project.namespace or {}
    web_url   = project.web_url
    team_name = extract_team_name(web_url)

    # Walk tree — manual pagination, each page has its own timeout
    all_items = _repo_tree_pages(project, ref)
    src_items = [
        i for i in all_items
        if i.get('type') == 'blob' and i['path'].endswith(SOURCE_EXTENSIONS)
    ]
    logger.info('P2 | tree=%d items  src=%d files | %s', len(all_items), len(src_items), ns)

    base = {
        'project_id'         : project.id,
        'name'               : project.name,
        'path'               : project.path,
        'path_with_namespace': project.path_with_namespace,
        'group_path'         : namespace.get('full_path'),
        'web_url'            : web_url,
        'description'        : project.description,
        'visibility'         : project.visibility,
        'archived'           : project.archived,
        'created_at'         : project.created_at,
        'last_activity_at'   : project.last_activity_at,
        'updated_at'         : project.updated_at,
        'default_branch'     : ref,
        'forks_count'        : getattr(project, 'forks_count', None),
        'star_count'         : getattr(project, 'star_count', None),
        'open_issues_count'  : getattr(project, 'open_issues_count', None),
        'team_name'          : team_name,
        'package_json'       : 'Yes',
        'int_ext'            : int_ext,
        'dependency'         : dependency,
        'component'          : '',
    }

    if not src_items:
        logger.info('P2 | no src files → 1 summary row | %s', ns)
        return [{**base, 'library': '', 'import_filename': '',
                 'import_file_url': '', 'import_statement': ''}]

    # Fetch all source files in parallel (bounded by FILE_WORKERS and _gitlab_sem)
    import_rows: List[Dict[str, Any]] = []
    local_fetched = local_empty = local_error = local_stmts = 0

    def process_file(item) -> Optional[List[Dict[str, Any]]]:
        nonlocal local_fetched, local_empty, local_error, local_stmts
        path    = item['path']
        content = _raw_file(project, path, ref)

        if content is None:
            local_error += 1
            return None
        if not content.strip():
            local_empty += 1
            return None

        local_fetched += 1
        file_rows = []
        for m in ANY_IMPORT_RE.finditer(content):
            stmt       = m.group(0).strip()
            module_str = m.group(1)
            library    = _derive_library(module_str)
            logger.debug('  IMP | %s | %s', path, stmt[:80])
            file_rows.append({
                **base,
                'library'         : library,
                'import_filename' : path.split('/')[-1],
                'import_file_url' : f'{web_url}/-/blob/{ref}/{path}',
                'import_statement': stmt,
            })
            local_stmts += 1
        return file_rows or None

    with ThreadPoolExecutor(max_workers=FILE_WORKERS) as pool:
        for result in pool.map(process_file, src_items):
            if result:
                import_rows.extend(result)

    elapsed = time.perf_counter() - t0
    logger.info('P2 | DONE %.1fs | ok=%d empty=%d err=%d imports=%d | %s',
                elapsed, local_fetched, local_empty, local_error, len(import_rows), ns)

    _inc('source_files_fetched', local_fetched)
    _inc('source_files_empty',   local_empty)
    _inc('source_files_error',   local_error)
    _inc('import_statements_total', local_stmts)

    if not import_rows:
        return [{**base, 'library': '', 'import_filename': '',
                 'import_file_url': '', 'import_statement': ''}]

    _inc('output_rows', len(import_rows))
    return import_rows


# ── Greenplum helpers ─────────────────────────────────────────────────────────

# Full column list in the exact order used for CREATE TABLE and INSERT
_COLUMNS = [
    ('project_id',          'BIGINT'),
    ('name',                'TEXT'),
    ('path',                'TEXT'),
    ('path_with_namespace', 'TEXT'),
    ('group_path',          'TEXT'),
    ('web_url',             'TEXT'),
    ('description',         'TEXT'),
    ('visibility',          'TEXT'),
    ('archived',            'BOOLEAN'),
    ('created_at',          'TIMESTAMP'),
    ('last_activity_at',    'TIMESTAMP'),
    ('updated_at',          'TIMESTAMP'),
    ('default_branch',      'TEXT'),
    ('forks_count',         'INTEGER'),
    ('star_count',          'INTEGER'),
    ('open_issues_count',   'INTEGER'),
    ('team_name',           'TEXT'),
    ('package_json',        'TEXT'),
    ('int_ext',             'TEXT'),
    ('dependency',          'TEXT'),
    ('component',           'TEXT'),
    ('library',             'TEXT'),
    ('import_filename',     'TEXT'),
    ('import_file_url',     'TEXT'),
    ('import_statement',    'TEXT'),
]
_COL_NAMES  = [c[0] for c in _COLUMNS]
_FQTABLE    = f'{DB_SCHEMA}.{DB_TABLE}'


def _get_conn():
    """Open and return a new psycopg2 connection."""
    cfg = {k: v for k, v in DB_CONFIG.items()}   # shallow copy
    return psycopg2.connect(**cfg)


def db_setup_table(conn) -> None:
    """DROP (if exists) + CREATE table + ALTER OWNER + GRANT."""
    col_defs = ',\n    '.join(f'{col} {dtype}' for col, dtype in _COLUMNS)
    ddl_drop   = f'DROP TABLE IF EXISTS {_FQTABLE};'
    ddl_create = (
        f'CREATE TABLE {_FQTABLE} (\n    {col_defs}\n)'
        f' DISTRIBUTED BY (project_id);'
    )
    ddl_owner  = (
        f'ALTER TABLE {_FQTABLE} '
        f'OWNER TO erd_gpdb_prj_smart_insights;'
    )
    ddl_grant  = (
        f'GRANT SELECT ON {_FQTABLE} '
        f'TO erd_gpdb_prj_smart_insights_ro;'
    )

    with conn.cursor() as cur:
        logger.info('DB | DROP TABLE IF EXISTS %s', _FQTABLE)
        cur.execute(ddl_drop)
        logger.info('DB | CREATE TABLE %s', _FQTABLE)
        cur.execute(ddl_create)
        logger.info('DB | ALTER OWNER')
        cur.execute(ddl_owner)
        logger.info('DB | GRANT SELECT')
        cur.execute(ddl_grant)
    conn.commit()
    logger.info('DB | table ready: %s', _FQTABLE)


def _coerce_val(val: Any) -> Any:
    """Normalise a single value so psycopg2 can safely bind it."""
    if val is None:
        return None
    if isinstance(val, float) and val != val:        # NaN -> NULL
        return None
    if isinstance(val, str) and val.strip() == '':
        return None
    # archived arrives as Python bool or the string 'false'/'true' from proj_ref
    if isinstance(val, str) and val.lower() in ('true', 'false'):
        return val.lower() == 'true'
    return val


def _coerce_row(row: Dict[str, Any]) -> tuple:
    """Convert a row dict to a tuple in _COL_NAMES order, safe for psycopg2."""
    return tuple(_coerce_val(row.get(col)) for col in _COL_NAMES)


def db_insert_rows(conn, rows: List[Dict[str, Any]]) -> None:
    """
    Batch-insert rows using execute_values.
    template= specifies the per-row shape; %s in the SQL is the VALUES placeholder.
    """
    if not rows:
        logger.warning('DB | no rows to insert')
        return

    col_list         = ', '.join(_COL_NAMES)
    col_placeholders = '(' + ', '.join(['%s'] * len(_COL_NAMES)) + ')'
    sql              = f'INSERT INTO {_FQTABLE} ({col_list}) VALUES %s'

    tuples   = [_coerce_row(r) for r in rows]
    total    = len(tuples)
    inserted = 0

    with conn.cursor() as cur:
        for start in range(0, total, DB_BATCH_SIZE):
            batch = tuples[start : start + DB_BATCH_SIZE]
            psycopg2.extras.execute_values(cur, sql, batch, template=col_placeholders)
            inserted += len(batch)
            logger.info('DB | inserted %d / %d rows', inserted, total)
    conn.commit()
    logger.info('DB | ✅ %d rows committed to %s', total, _FQTABLE)


def db_update_imports(conn, rows: List[Dict[str, Any]]) -> None:
    """
    Replace the Phase-1 summary rows for each project with enriched Phase-2 rows.
    Strategy: DELETE existing rows for these project_ids, then INSERT new rows.
    This is simpler and more reliable than UPDATE on Greenplum.
    """
    if not rows:
        logger.warning('DB | no import rows to update')
        return

    project_ids = list({_coerce_val(r.get('project_id')) for r in rows
                        if r.get('project_id') is not None})
    logger.info('DB | replacing rows for %d project(s) with import data …', len(project_ids))

    col_list         = ', '.join(_COL_NAMES)
    col_placeholders = '(' + ', '.join(['%s'] * len(_COL_NAMES)) + ')'
    sql_delete       = f'DELETE FROM {_FQTABLE} WHERE project_id = ANY(%s)'
    sql_insert       = f'INSERT INTO {_FQTABLE} ({col_list}) VALUES %s'

    tuples   = [_coerce_row(r) for r in rows]
    total    = len(tuples)
    inserted = 0

    with conn.cursor() as cur:
        cur.execute(sql_delete, (project_ids,))
        logger.info('DB | deleted %d Phase-1 summary row(s) for these projects', cur.rowcount)

        for start in range(0, total, DB_BATCH_SIZE):
            batch = tuples[start : start + DB_BATCH_SIZE]
            psycopg2.extras.execute_values(cur, sql_insert, batch, template=col_placeholders)
            inserted += len(batch)
            logger.info('DB | inserted %d / %d enriched rows', inserted, total)

    conn.commit()
    logger.info('DB | ✅ %d import rows committed to %s', total, _FQTABLE)


def db_write_phase1(rows: List[Dict[str, Any]]) -> None:
    """Connect → DROP/CREATE table → insert Phase-1 rows → close."""
    logger.info('DB | connecting to %s …', DB_CONFIG['host'])
    conn = _get_conn()
    try:
        db_setup_table(conn)
        db_insert_rows(conn, rows)
    except Exception as exc:
        logger.error('DB | Phase-1 write FAILED: %s', exc)
        conn.rollback()
        raise
    finally:
        conn.close()
        logger.info('DB | connection closed')


def db_write_phase2(rows: List[Dict[str, Any]]) -> None:
    """Connect → replace Phase-1 summary rows with enriched Phase-2 rows → close."""
    logger.info('DB | connecting to %s …', DB_CONFIG['host'])
    conn = _get_conn()
    try:
        db_update_imports(conn, rows)
    except Exception as exc:
        logger.error('DB | Phase-2 write FAILED: %s', exc)
        conn.rollback()
        raise
    finally:
        conn.close()
        logger.info('DB | connection closed')


# ── Phase-1-only summary row builder ─────────────────────────────────────────

def phase1_to_row(p1: Dict[str, Any]) -> Dict[str, Any]:
    """Build a minimal output row from Phase-1 metadata (no import data)."""
    return {
        'project_id'         : getattr(p1['proj_ref'], 'id', None),
        'name'               : p1['name'],
        'path'               : p1['path'],
        'path_with_namespace': p1['path_with_namespace'],
        'group_path'         : None,
        'web_url'            : p1['web_url'],
        'description'        : None,
        'visibility'         : p1['visibility'],
        'archived'           : None,
        'created_at'         : p1['created_at'] or None,
        'last_activity_at'   : None,
        'updated_at'         : p1['updated_at'] or None,
        'default_branch'     : p1['default_branch'],
        'forks_count'        : None,
        'star_count'         : None,
        'open_issues_count'  : None,
        'team_name'          : extract_team_name(p1['web_url']),
        'package_json'       : 'Yes',
        'int_ext'            : p1['int_ext'],
        'dependency'         : p1['dependency'],
        'component'          : '',
        'library'            : '',
        'import_filename'    : '',
        'import_file_url'    : '',
        'import_statement'   : '',
    }


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

logger.info('=' * 70)
logger.info("EXPORT | fetching project list for group '%s' …", GROUP_PATH)
t_total = time.perf_counter()

group     = client.groups.get(GROUP_PATH)
proj_refs = group.projects.list(include_subgroups=True, all=True, per_page=PER_PAGE)
total_all = len(proj_refs)
run_stats['total_in_group'] = total_all
logger.info('EXPORT | %d projects in group', total_all)

# ── Date filter (zero API calls) ──────────────────────────────────────────────
recent_refs = [p for p in proj_refs if is_recent(p)]
old_count   = total_all - len(recent_refs)
run_stats['recent_projects'] = len(recent_refs)
run_stats['skipped_too_old'] = old_count
logger.info('DATE   | recent=%d  too_old=%d (cutoff=%s)',
            len(recent_refs), old_count, UPDATED_SINCE.date())


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 1 — parallel package.json check
# ══════════════════════════════════════════════════════════════════════════════
logger.info('─' * 70)
logger.info('PHASE 1 | %d projects to check …', len(recent_refs))
t_p1 = time.perf_counter()

phase1_results : List[Dict[str, Any]] = []
p1_done = 0
p1_total = len(recent_refs)

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
    future_to_ref = {pool.submit(check_package_json, ref): ref for ref in recent_refs}

    for future in as_completed(future_to_ref):
        p1_done += 1
        ref = future_to_ref[future]

        # Per-future timeout — won't hang forever
        try:
            result = future.result(timeout=FUTURE_TIMEOUT)
        except FutureTimeout:
            logger.warning('P1 | TIMEOUT (%ds) | %s', FUTURE_TIMEOUT, ref.path_with_namespace)
            _inc('p1_timeout')
            result = None
        except Exception as exc:
            logger.error('P1 | EXCEPTION | %s | %s', ref.path_with_namespace, exc)
            result = None

        if result:
            phase1_results.append(result)

        # Live progress every PROGRESS_EVERY completions
        if p1_done % PROGRESS_EVERY == 0 or p1_done == p1_total:
            pct = 100 * p1_done // p1_total
            print(f'\r  P1: {p1_done}/{p1_total} ({pct}%)  '
                  f'found={len(phase1_results)}  '
                  f'no_pkg={run_stats["skipped_no_package_json"]}  '
                  f'timeout={run_stats["p1_timeout"]}   ',
                  end='', flush=True)

print()   # newline after progress line
t_p1_elapsed = time.perf_counter() - t_p1
logger.info('PHASE 1 | done in %.1fs | has_pkg=%d  no_pkg=%d  timeout=%d',
            t_p1_elapsed,
            run_stats['has_package_json'],
            run_stats['skipped_no_package_json'],
            run_stats['p1_timeout'])


# ── User gate after Phase 1 ───────────────────────────────────────────────────
print()
print('═' * 70)
print('  PHASE 1 COMPLETE')
print('═' * 70)
print(f'  Total projects in group             : {total_all}')
print(f'  Updated since {UPDATED_SINCE.date()}           : {len(recent_refs)}')
print(f'    └─ too old (skipped)              : {old_count}')
print(f'  Have package.json                   : {run_stats["has_package_json"]}')
print(f'    └─ no package.json (skipped)      : {run_stats["skipped_no_package_json"]}')
print(f'    └─ timed out                      : {run_stats["p1_timeout"]}')
print(f'  Phase 1 elapsed                     : {t_p1_elapsed:.1f}s')
print('═' * 70)
print()

_proceed = input(
    f'  Proceed to import-statement scan for {run_stats["has_package_json"]} projects? (yes/no): '
).strip().lower()

# ── Always write Phase-1 rows to DB first ────────────────────────────────────
logger.info('USER | writing Phase-1 project rows to Greenplum before asking about Phase 2 …')
p1_rows = [phase1_to_row(p) for p in phase1_results]
db_write_phase1(p1_rows)
print(f'\n✅  {len(p1_rows)} project rows written to {_FQTABLE}')

if _proceed not in ('yes', 'y'):
    logger.info('USER | chose to stop after Phase 1 — done.')
    sys.exit(0)

logger.info('USER | proceeding to Phase 2 (import scan) …')


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 2 — parallel file scan
# ══════════════════════════════════════════════════════════════════════════════
logger.info('─' * 70)
logger.info('PHASE 2 | scanning %d projects …', len(phase1_results))
t_p2 = time.perf_counter()

all_rows : List[Dict[str, Any]] = []
p2_done  = 0
p2_total = len(phase1_results)

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
    future_to_p1 = {pool.submit(scan_and_build_rows, p1): p1 for p1 in phase1_results}

    for future in as_completed(future_to_p1):
        p2_done += 1
        p1 = future_to_p1[future]
        ns = p1['path_with_namespace']

        try:
            rows = future.result(timeout=FUTURE_TIMEOUT)
        except FutureTimeout:
            logger.warning('P2 | TIMEOUT (%ds) | %s', FUTURE_TIMEOUT, ns)
            _inc('p2_timeout')
            rows = None
        except Exception as exc:
            logger.error('P2 | EXCEPTION | %s | %s', ns, exc)
            rows = None

        if rows:
            all_rows.extend(rows)
            logger.info('P2 | %d/%d ✓ +%d rows | %s', p2_done, p2_total, len(rows), ns)

        # Live progress
        if p2_done % PROGRESS_EVERY == 0 or p2_done == p2_total:
            pct = 100 * p2_done // p2_total
            print(f'\r  P2: {p2_done}/{p2_total} ({pct}%)  '
                  f'rows={len(all_rows)}  '
                  f'imports={run_stats["import_statements_total"]}  '
                  f'timeout={run_stats["p2_timeout"]}   ',
                  end='', flush=True)

print()
t_p2_elapsed = time.perf_counter() - t_p2
t_elapsed    = time.perf_counter() - t_total
included_projects = len({r['project_id'] for r in all_rows if r.get('project_id')})
run_stats['output_rows'] = len(all_rows)

logger.info('PHASE 2 | done in %.1fs', t_p2_elapsed)


# ── Final summary ─────────────────────────────────────────────────────────────
summary_lines = [
    ('Total projects in group',                             run_stats['total_in_group']),
    (f'Updated since {UPDATED_SINCE.date()}',              run_stats['recent_projects']),
    ('  └─ too old (skipped)',                             run_stats['skipped_too_old']),
    ('',                                                    ''),
    ('Have package.json',                                   run_stats['has_package_json']),
    ('  └─ no package.json (skipped)',                     run_stats['skipped_no_package_json']),
    ('  └─ Phase-1 timeout',                               run_stats['p1_timeout']),
    ('',                                                    ''),
    ('Source files fetched',                                run_stats['source_files_fetched']),
    ('Source files empty',                                  run_stats['source_files_empty']),
    ('Source files error',                                  run_stats['source_files_error']),
    ('Import statements captured',                          run_stats['import_statements_total']),
    ('  └─ Phase-2 timeout (projects)',                    run_stats['p2_timeout']),
    ('',                                                    ''),
    ('Distinct projects in output',                         included_projects),
    ('Total output rows',                                   run_stats['output_rows']),
    ('Rows with import statement',
        sum(1 for r in all_rows if r.get('import_statement'))),
    ('Phase 1 elapsed (s)',                                 f'{t_p1_elapsed:.1f}'),
    ('Phase 2 elapsed (s)',                                 f'{t_p2_elapsed:.1f}'),
    ('Total elapsed (s)',                                   f'{t_elapsed:.1f}'),
]

print()
print('═' * 70)
print('  FINAL SUMMARY')
print('═' * 70)
for label, value in summary_lines:
    if label == '':
        print()
    else:
        print(f'  {label:<52} {value}')
print('═' * 70)

for label, value in summary_lines:
    if label:
        logger.info('SUMMARY | %-52s %s', label, value)


# ── Write to Greenplum ────────────────────────────────────────────────────────
if not all_rows:
    logger.warning('DB | no import rows collected')
    print('\n⚠️  No import rows collected — Phase-1 data remains in table.')
else:
    db_write_phase2(all_rows)
    print(f'\n✅  {len(all_rows)} enriched rows written to {_FQTABLE}')

logger.info('=' * 70)
logger.info('GitLab Projects Export  v5  — complete (%.1fs total)', t_elapsed)
logger.info('=' * 70)
