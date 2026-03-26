# GitLab Projects Export  ── v5
#
# PIPELINE
# ────────
# Phase 1  Find every project updated since Jan-2025 that has package.json.
#          Fetch full project object to get ALL metadata (id, description, forks, etc.).
#          Insert one row per project into Greenplum table gwma_ui_projects.
#          Print summary.  Ask user: continue to import scan?
#
# Phase 2  (only if user says yes)
#          Walk each project's file tree, extract every import statement from
#          .js/.ts/.jsx/.tsx files.  One row per import statement.
#          ADD import columns to the table, then replace Phase-1 rows with
#          enriched Phase-2 rows (DELETE + INSERT per project).
#
# TABLE COLUMNS (Phase 1)
# ───────────────────────
#   project_id, name, path, path_with_namespace, group_path, web_url,
#   description, visibility, archived, created_at, last_activity_at,
#   updated_at, default_branch, forks_count, star_count, open_issues_count,
#   team_name, package_json
#
# Additional columns added in Phase 2:
#   library, import_filename, import_file_url, import_statement

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
logger.info('GitLab Projects Export  v5')
logger.info('=' * 70)

# ── Configuration ─────────────────────────────────────────────────────────────
GITLAB_URL      = 'https://devcloud.ubs.net'
GROUP_PATH      = 'ubs/gwma'

PER_PAGE        = 100         # GitLab max page size
MAX_WORKERS     = 10          # parallel project threads
FILE_WORKERS    = 5           # parallel file threads inside one project
MAX_CONNECTIONS = 10          # hard cap on open GitLab HTTP connections
GITLAB_TIMEOUT  = (10, 30)    # (connect_s, read_s) per HTTP request
FUTURE_TIMEOUT  = 120         # wall-clock seconds before a future is abandoned
PROGRESS_EVERY  = 5           # print progress line every N completions

UPDATED_SINCE     = datetime(2025, 1, 1, tzinfo=timezone.utc)
SOURCE_EXTENSIONS = ('.jsx', '.tsx', '.js', '.ts')

# Greenplum
DB_SCHEMA     = 'sandbox_prj_smart_insights'
DB_TABLE      = 'gwma_ui_projects'
DB_FQTABLE    = f'{DB_SCHEMA}.{DB_TABLE}'
DB_BATCH_SIZE = 200

# Phase-1 columns (exactly what was requested, in order)
P1_COLUMNS = [
    ('project_id',          'BIGINT'),
    ('name',                'TEXT'),
    ('path',                'TEXT'),
    ('path_with_namespace', 'TEXT'),
    ('group_path',          'TEXT'),
    ('web_url',             'TEXT'),
    ('description',         'TEXT'),
    ('visibility',          'TEXT'),
    ('archived',            'BOOLEAN'),
    ('created_at',          'TIMESTAMP WITH TIME ZONE'),
    ('last_activity_at',    'TIMESTAMP WITH TIME ZONE'),
    ('updated_at',          'TIMESTAMP WITH TIME ZONE'),
    ('default_branch',      'TEXT'),
    ('forks_count',         'INTEGER'),
    ('star_count',          'INTEGER'),
    ('open_issues_count',   'INTEGER'),
    ('team_name',           'TEXT'),
    ('package_json',        'TEXT'),
]

# Phase-2 columns added via ALTER TABLE
P2_EXTRA_COLUMNS = [
    ('library',           'TEXT'),
    ('import_filename',   'TEXT'),
    ('import_file_url',   'TEXT'),
    ('import_statement',  'TEXT'),
]

P1_COL_NAMES  = [c[0] for c in P1_COLUMNS]
P2_COL_NAMES  = [c[0] for c in P2_EXTRA_COLUMNS]
ALL_COL_NAMES = P1_COL_NAMES + P2_COL_NAMES

# ── Import regex — all ES-module import forms ─────────────────────────────────
ANY_IMPORT_RE = re.compile(
    r"^[ \t]*import"
    r"(?:\s+(?:"
        r"[\w\$_][\w\$_]*(?:\s*,\s*(?:\*\s+as\s+[\w\$_]+|\{[^}]*\}))?"
        r"|\*\s+as\s+[\w\$_][\w\$_]*"
        r"|\{[^}]*\}"
    r")\s+from)?"
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
    'p1_errors'               : 0,
    'p1_timeouts'             : 0,
    'source_files_fetched'    : 0,
    'source_files_empty'      : 0,
    'source_files_error'      : 0,
    'import_rows_total'       : 0,
    'p2_errors'               : 0,
    'p2_timeouts'             : 0,
}

def _inc(key: str, n: int = 1) -> None:
    with _stats_lock:
        run_stats[key] += n

_gitlab_sem = threading.Semaphore(MAX_CONNECTIONS)


# ── Credentials ───────────────────────────────────────────────────────────────
logger.info('AUTH | requesting GitLab private token ...')
private_token = getpass.getpass('Enter your GitLab private token: ')

_session = requests.Session()
_adapter = HTTPAdapter(max_retries=1,
                       pool_connections=MAX_CONNECTIONS,
                       pool_maxsize=MAX_CONNECTIONS)
_session.mount('https://', _adapter)
_session.mount('http://',  _adapter)
_orig_request = _session.request
def _timed_request(method, url, **kwargs):
    kwargs.setdefault('timeout', GITLAB_TIMEOUT)
    return _orig_request(method, url, **kwargs)
_session.request = _timed_request

client = gitlab.Gitlab(GITLAB_URL, private_token=private_token, session=_session)
logger.info('AUTH | GitLab client ready')

logger.info('DB   | requesting Greenplum password ...')
db_password = getpass.getpass('Enter Password for DB User: ')
DB_CONFIG = {
    'host'    : 'greenplum-rdsp.zur.swissbank.com',
    'port'    : 5432,
    'dbname'  : 'gprdsp',
    'user'    : 'ds_rdsp_dev',
    'password': db_password,
}
logger.info('DB   | credentials stored  host=%s  db=%s  user=%s',
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
    with _gitlab_sem:
        try:
            return project.files.raw(file_path=path, ref=ref).decode('utf-8', errors='replace')
        except Exception as exc:
            logger.debug('FILE | err | %s | %s', path, exc)
            return None


def _repo_tree_pages(project: Any, ref: str) -> List[Dict]:
    items, page = [], 1
    while True:
        with _gitlab_sem:
            try:
                batch = project.repository_tree(
                    ref=ref, recursive=True,
                    per_page=PER_PAGE, page=page, get_all=False,
                )
            except Exception as exc:
                logger.warning('TREE | page=%d error | %s | %s',
                               page, project.path_with_namespace, exc)
                break
        if not batch:
            break
        items.extend(batch)
        if len(batch) < PER_PAGE:
            break
        page += 1
    return items


def _coerce(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, float) and val != val:
        return None
    if isinstance(val, str) and val.strip() == '':
        return None
    return val


def _derive_library(module_path: str) -> str:
    s = module_path.lower()
    found = []
    if s.startswith('@uwr/'):
        found.append('uwr')
    if s.startswith('@ubs.websdk/'):
        found.append('websdk')
    return ', '.join(found)


# ── Phase 1 worker ────────────────────────────────────────────────────────────

def fetch_project_with_pkg(proj_ref: Any) -> Optional[Dict[str, Any]]:
    """
    1. Fetch full project object (for description, forks, stars, namespace, etc.)
    2. Check whether package.json exists.
    3. Return a complete row dict if package.json exists, else None.
    """
    ns  = proj_ref.path_with_namespace
    ref = getattr(proj_ref, 'default_branch', None) or 'main'

    # Full project object — gives us all metadata
    with _gitlab_sem:
        try:
            project = client.projects.get(proj_ref.id)
        except Exception as exc:
            logger.error('P1 | project.get FAILED | %s | %s', ns, exc)
            _inc('p1_errors')
            return None

    ref  = project.default_branch or 'main'
    ns_d = project.namespace or {}

    # Check package.json
    with _gitlab_sem:
        try:
            project.files.raw(file_path='package.json', ref=ref)
            has_pkg = True
        except Exception:
            has_pkg = False

    if not has_pkg:
        logger.debug('P1 | NO  pkg.json | %s', project.path_with_namespace)
        _inc('skipped_no_package_json')
        return None

    logger.info('P1 | YES pkg.json | %s', project.path_with_namespace)
    _inc('has_package_json')

    return {
        # Internal keys for Phase 2 (not DB columns)
        '_project': project,
        '_ref'    : ref,
        # DB columns
        'project_id'         : int(project.id),
        'name'               : project.name,
        'path'               : project.path,
        'path_with_namespace': project.path_with_namespace,
        'group_path'         : ns_d.get('full_path'),
        'web_url'            : project.web_url,
        'description'        : project.description,
        'visibility'         : project.visibility,
        'archived'           : bool(project.archived),
        'created_at'         : project.created_at,
        'last_activity_at'   : project.last_activity_at,
        'updated_at'         : project.updated_at,
        'default_branch'     : ref,
        'forks_count'        : int(project.forks_count)       if project.forks_count       is not None else None,
        'star_count'         : int(project.star_count)        if project.star_count        is not None else None,
        'open_issues_count'  : int(project.open_issues_count) if project.open_issues_count is not None else None,
        'team_name'          : extract_team_name(project.web_url),
        'package_json'       : 'Yes',
    }


# ── Phase 2 worker ────────────────────────────────────────────────────────────

def scan_imports(p1_row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Walk project tree, fetch source files in parallel, extract every import statement.
    Returns one enriched row dict per import (or one row with blank import columns).
    """
    project = p1_row['_project']
    ref     = p1_row['_ref']
    ns      = project.path_with_namespace
    web_url = project.web_url
    t0      = time.perf_counter()

    logger.info('P2 | START | %s', ns)

    all_items = _repo_tree_pages(project, ref)
    src_items = [i for i in all_items
                 if i.get('type') == 'blob' and i['path'].endswith(SOURCE_EXTENSIONS)]
    logger.info('P2 | tree=%d  src=%d | %s', len(all_items), len(src_items), ns)

    # Base: all P1 DB columns (no internal _ keys)
    base = {k: p1_row[k] for k in P1_COL_NAMES}

    if not src_items:
        return [{**base, 'library': '', 'import_filename': '',
                 'import_file_url': '', 'import_statement': ''}]

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
        rows = []
        for m in ANY_IMPORT_RE.finditer(content):
            stmt    = m.group(0).strip()
            module  = m.group(1)
            rows.append({
                **base,
                'library'         : _derive_library(module),
                'import_filename' : path.split('/')[-1],
                'import_file_url' : f'{web_url}/-/blob/{ref}/{path}',
                'import_statement': stmt,
            })
            local_stmts += 1
        return rows or None

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
    _inc('import_rows_total',    len(import_rows) if import_rows else 1)

    if not import_rows:
        return [{**base, 'library': '', 'import_filename': '',
                 'import_file_url': '', 'import_statement': ''}]
    return import_rows


# ── Greenplum helpers ─────────────────────────────────────────────────────────

def _get_conn():
    return psycopg2.connect(
        host    = DB_CONFIG['host'],
        port    = DB_CONFIG['port'],
        dbname  = DB_CONFIG['dbname'],
        user    = DB_CONFIG['user'],
        password= DB_CONFIG['password'],
    )


def db_test_connection() -> bool:
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute('SELECT 1')
        conn.close()
        logger.info('DB   | connection test OK')
        return True
    except Exception as exc:
        logger.error('DB   | connection test FAILED: %s', exc)
        return False


def db_setup_phase1(conn) -> None:
    col_defs = ',\n    '.join(f'"{col}" {dtype}' for col, dtype in P1_COLUMNS)
    with conn.cursor() as cur:
        logger.info('DB   | DROP TABLE IF EXISTS %s', DB_FQTABLE)
        cur.execute(f'DROP TABLE IF EXISTS {DB_FQTABLE};')

        sql_create = (
            f'CREATE TABLE {DB_FQTABLE} (\n    {col_defs}\n)'
            f' DISTRIBUTED BY (project_id);'
        )
        logger.info('DB   | CREATE TABLE %s', DB_FQTABLE)
        logger.debug('DDL  | %s', sql_create)
        cur.execute(sql_create)

        logger.info('DB   | ALTER OWNER TO erd_gpdb_prj_smart_insights')
        cur.execute(f'ALTER TABLE {DB_FQTABLE} OWNER TO erd_gpdb_prj_smart_insights;')

        logger.info('DB   | GRANT SELECT TO erd_gpdb_prj_smart_insights_ro')
        cur.execute(f'GRANT SELECT ON {DB_FQTABLE} TO erd_gpdb_prj_smart_insights_ro;')
    conn.commit()
    logger.info('DB   | table created and permissions set')


def db_add_import_columns(conn) -> None:
    with conn.cursor() as cur:
        for col, dtype in P2_EXTRA_COLUMNS:
            try:
                cur.execute(f'ALTER TABLE {DB_FQTABLE} ADD COLUMN "{col}" {dtype};')
                conn.commit()
                logger.info('DB   | added column %s', col)
            except psycopg2.errors.DuplicateColumn:
                conn.rollback()
                logger.debug('DB   | column %s already exists', col)
            except Exception as exc:
                conn.rollback()
                logger.warning('DB   | could not add column %s: %s', col, exc)


def _p1_tuple(row: Dict[str, Any]) -> tuple:
    return tuple(_coerce(row.get(col)) for col in P1_COL_NAMES)


def _all_tuple(row: Dict[str, Any]) -> tuple:
    return tuple(_coerce(row.get(col)) for col in ALL_COL_NAMES)


def db_insert_p1(conn, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        logger.warning('DB   | no rows to insert')
        return

    col_list = ', '.join(f'"{c}"' for c in P1_COL_NAMES)
    template = '(' + ', '.join(['%s'] * len(P1_COL_NAMES)) + ')'
    sql      = f'INSERT INTO {DB_FQTABLE} ({col_list}) VALUES %s'
    tuples   = [_p1_tuple(r) for r in rows]

    logger.info('DB   | INSERT %d rows  table=%s', len(tuples), DB_FQTABLE)
    logger.info('DB   | sample row[0]: %s', dict(zip(P1_COL_NAMES, tuples[0])) if tuples else '(none)')

    inserted = 0
    with conn.cursor() as cur:
        for start in range(0, len(tuples), DB_BATCH_SIZE):
            batch = tuples[start : start + DB_BATCH_SIZE]
            psycopg2.extras.execute_values(cur, sql, batch, template=template)
            inserted += len(batch)
            logger.info('DB   | committed %d / %d rows', inserted, len(tuples))
    conn.commit()
    logger.info('DB   | INSERT complete — %d rows', inserted)


def db_replace_with_imports(conn, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        logger.warning('DB   | no import rows to write')
        return

    project_ids = list({_coerce(r.get('project_id')) for r in rows
                        if r.get('project_id') is not None})
    logger.info('DB   | replacing rows for %d project(s)', len(project_ids))

    col_list = ', '.join(f'"{c}"' for c in ALL_COL_NAMES)
    template = '(' + ', '.join(['%s'] * len(ALL_COL_NAMES)) + ')'
    sql_del  = f'DELETE FROM {DB_FQTABLE} WHERE project_id = ANY(%s)'
    sql_ins  = f'INSERT INTO {DB_FQTABLE} ({col_list}) VALUES %s'
    tuples   = [_all_tuple(r) for r in rows]

    inserted = 0
    with conn.cursor() as cur:
        cur.execute(sql_del, (project_ids,))
        logger.info('DB   | deleted %d Phase-1 row(s)', cur.rowcount)
        for start in range(0, len(tuples), DB_BATCH_SIZE):
            batch = tuples[start : start + DB_BATCH_SIZE]
            psycopg2.extras.execute_values(cur, sql_ins, batch, template=template)
            inserted += len(batch)
            logger.info('DB   | inserted %d / %d enriched rows', inserted, len(tuples))
    conn.commit()
    logger.info('DB   | Phase-2 INSERT complete — %d rows', inserted)


# ── Main pipeline ─────────────────────────────────────────────────────────────

# Test DB before wasting time on GitLab
logger.info('DB   | testing connection ...')
if not db_test_connection():
    print('\nERROR: Cannot connect to Greenplum. Check credentials and network.')
    sys.exit(1)

# Fetch full project list
logger.info('=' * 70)
logger.info("EXPORT | fetching project list for group '%s' ...", GROUP_PATH)
t_total = time.perf_counter()

group     = client.groups.get(GROUP_PATH)
proj_refs = group.projects.list(include_subgroups=True, all=True, per_page=PER_PAGE)
total_all = len(proj_refs)
run_stats['total_in_group'] = total_all
logger.info('EXPORT | %d projects in group', total_all)

# Date filter (zero API calls)
recent_refs = [p for p in proj_refs if is_recent(p)]
old_count   = total_all - len(recent_refs)
run_stats['recent_projects'] = len(recent_refs)
run_stats['skipped_too_old'] = old_count
logger.info('DATE   | recent=%d  too_old=%d  cutoff=%s',
            len(recent_refs), old_count, UPDATED_SINCE.date())

# =========================================================================
#  PHASE 1 — fetch metadata + check package.json — parallel
# =========================================================================
logger.info('-' * 70)
logger.info('PHASE 1 | processing %d recent projects ...', len(recent_refs))
t_p1 = time.perf_counter()

p1_rows : List[Dict[str, Any]] = []
p1_done  = 0
p1_total = len(recent_refs)

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
    future_map = {pool.submit(fetch_project_with_pkg, ref): ref for ref in recent_refs}
    for future in as_completed(future_map):
        p1_done += 1
        ref = future_map[future]
        try:
            row = future.result(timeout=FUTURE_TIMEOUT)
        except FutureTimeout:
            logger.warning('P1 | TIMEOUT | %s', ref.path_with_namespace)
            _inc('p1_timeouts')
            row = None
        except Exception as exc:
            logger.error('P1 | ERROR | %s | %s', ref.path_with_namespace, exc)
            _inc('p1_errors')
            row = None

        if row:
            p1_rows.append(row)

        if p1_done % PROGRESS_EVERY == 0 or p1_done == p1_total:
            pct = 100 * p1_done // p1_total
            print(f'\r  P1: {p1_done}/{p1_total} ({pct}%) '
                  f'| has_pkg={len(p1_rows)} '
                  f'no_pkg={run_stats["skipped_no_package_json"]} '
                  f'err={run_stats["p1_errors"]} '
                  f'timeout={run_stats["p1_timeouts"]}   ',
                  end='', flush=True)
print()

t_p1_elapsed = time.perf_counter() - t_p1
logger.info('PHASE 1 | done %.1fs | rows_to_insert=%d', t_p1_elapsed, len(p1_rows))

# Insert Phase-1 rows into Greenplum
print()
print('-' * 70)
print(f'  Inserting {len(p1_rows)} project rows into Greenplum ...')
try:
    conn = _get_conn()
    db_setup_phase1(conn)
    db_insert_p1(conn, p1_rows)
    conn.close()
    print(f'  OK  {len(p1_rows)} rows written to {DB_FQTABLE}')
except Exception as exc:
    logger.error('DB   | Phase-1 write FAILED: %s', exc, exc_info=True)
    print(f'\n  ERROR: Greenplum write failed: {exc}')
    sys.exit(1)

# Phase-1 summary + user gate
print()
print('=' * 70)
print('  PHASE 1 COMPLETE')
print('=' * 70)
print(f'  Total projects in group             : {total_all}')
print(f'  Updated since {UPDATED_SINCE.date()}           : {len(recent_refs)}')
print(f'    -- too old (skipped)              : {old_count}')
print(f'  Have package.json  -> inserted to DB: {len(p1_rows)}')
print(f'    -- no package.json (skipped)      : {run_stats["skipped_no_package_json"]}')
print(f'    -- errors / timeouts              : {run_stats["p1_errors"] + run_stats["p1_timeouts"]}')
print(f'  Phase 1 elapsed                     : {t_p1_elapsed:.1f}s')
print('=' * 70)
print()

_proceed = input(
    f'  Proceed to import-statement scan for {len(p1_rows)} projects? (yes/no): '
).strip().lower()

if _proceed not in ('yes', 'y'):
    t_elapsed = time.perf_counter() - t_total
    logger.info('USER | stopped after Phase 1')
    print(f'\n  Total elapsed: {t_elapsed:.1f}s')
    print(f'  Project data is in {DB_FQTABLE}')
    sys.exit(0)

logger.info('USER | proceeding to Phase 2')

# =========================================================================
#  PHASE 2 — file tree scan + import extraction — parallel
# =========================================================================
logger.info('-' * 70)
logger.info('PHASE 2 | scanning %d projects for import statements ...', len(p1_rows))
t_p2 = time.perf_counter()

p2_rows : List[Dict[str, Any]] = []
p2_done  = 0
p2_total = len(p1_rows)

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
    future_map = {pool.submit(scan_imports, row): row for row in p1_rows}
    for future in as_completed(future_map):
        p2_done += 1
        row = future_map[future]
        ns  = row.get('path_with_namespace', '?')
        try:
            result_rows = future.result(timeout=FUTURE_TIMEOUT)
        except FutureTimeout:
            logger.warning('P2 | TIMEOUT | %s', ns)
            _inc('p2_timeouts')
            result_rows = None
        except Exception as exc:
            logger.error('P2 | ERROR | %s | %s', ns, exc)
            _inc('p2_errors')
            result_rows = None

        if result_rows:
            p2_rows.extend(result_rows)
            logger.info('P2 | %d/%d +%d rows | %s', p2_done, p2_total, len(result_rows), ns)

        if p2_done % PROGRESS_EVERY == 0 or p2_done == p2_total:
            pct = 100 * p2_done // p2_total
            print(f'\r  P2: {p2_done}/{p2_total} ({pct}%) '
                  f'| rows={len(p2_rows)} '
                  f'imports={run_stats["import_rows_total"]} '
                  f'err={run_stats["p2_errors"]} '
                  f'timeout={run_stats["p2_timeouts"]}   ',
                  end='', flush=True)
print()

t_p2_elapsed = time.perf_counter() - t_p2
logger.info('PHASE 2 | done %.1fs | %d import rows', t_p2_elapsed, len(p2_rows))

# Write Phase-2 rows to Greenplum
print()
print('-' * 70)
print(f'  Writing {len(p2_rows)} enriched rows to Greenplum ...')
try:
    conn = _get_conn()
    db_add_import_columns(conn)
    db_replace_with_imports(conn, p2_rows)
    conn.close()
    print(f'  OK  {len(p2_rows)} rows written to {DB_FQTABLE}')
except Exception as exc:
    logger.error('DB   | Phase-2 write FAILED: %s', exc, exc_info=True)
    print(f'\n  ERROR: Greenplum Phase-2 write failed: {exc}')
    print('  Phase-1 data is still intact in the table.')

# Final summary
t_elapsed = time.perf_counter() - t_total
print()
print('=' * 70)
print('  FINAL SUMMARY')
print('=' * 70)
print(f'  Total projects in group             : {total_all}')
print(f'  Updated since {UPDATED_SINCE.date()}           : {len(recent_refs)}')
print(f'    -- too old (skipped)              : {old_count}')
print(f'  Have package.json                   : {len(p1_rows)}')
print(f'    -- no package.json (skipped)      : {run_stats["skipped_no_package_json"]}')
print()
print(f'  Source files fetched                : {run_stats["source_files_fetched"]}')
print(f'  Import statements captured          : {run_stats["import_rows_total"]}')
print(f'  Phase-2 output rows                 : {len(p2_rows)}')
print()
print(f'  Phase 1 elapsed (s)                 : {t_p1_elapsed:.1f}')
print(f'  Phase 2 elapsed (s)                 : {t_p2_elapsed:.1f}')
print(f'  Total elapsed (s)                   : {t_elapsed:.1f}')
print('=' * 70)

logger.info('=' * 70)
logger.info('GitLab Projects Export  v5  -- complete (%.1fs)', t_elapsed)
logger.info('=' * 70)
