# GitLab Projects Export  ── v5
#
# THREE TABLES
# ────────────
#   gwma_ui_projects       (Phase 1)  One row per project that has package.json
#   gwma_ui_dependencies   (Phase 2)  One row per dependency per project
#   gwma_ui_imports        (Phase 2)  One row per import statement per file per project
#
# FLOW
# ────
#   [A] Ask: run Phase 1? (yes = scan GitLab + recreate projects table)
#            (no  = skip, assume table already populated, go straight to Phase 2 prompt)
#   [B] If yes → scan all projects updated since Jan-2025, find those with package.json,
#       insert into gwma_ui_projects (DROP+CREATE every time).
#   [C] Ask: run Phase 2?
#            (yes = parse package.json for deps + scan source files for imports)
#            (no  = exit)
#   [D] If yes → load project list from gwma_ui_projects, process each project:
#         • parse package.json  → insert rows into gwma_ui_dependencies
#         • scan .js/.ts/.jsx/.tsx files → insert rows into gwma_ui_imports
#
# gwma_ui_projects columns
#   project_id, name, path, path_with_namespace, group_path, web_url, description,
#   visibility, archived, created_at, last_activity_at, updated_at, default_branch,
#   forks_count, star_count, open_issues_count, team_name, package_json
#
# gwma_ui_dependencies columns
#   project_id, name, path, dependency
#   (dependency = one package name per row; join to projects on project_id)
#
# gwma_ui_imports columns
#   project_id, name, path, import_filename, import_file_url, import_statement
#   (one row per import statement; join to projects on project_id)

import getpass
import json
import logging
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeout
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

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

PER_PAGE        = 100
MAX_WORKERS     = 10          # parallel project threads
FILE_WORKERS    = 5           # parallel file threads within one project
MAX_CONNECTIONS = 10          # hard cap on concurrent GitLab HTTP connections
GITLAB_TIMEOUT  = (10, 30)    # (connect_s, read_s) per request
FUTURE_TIMEOUT  = 120         # seconds before abandoning a stuck future
PROGRESS_EVERY  = 5

UPDATED_SINCE     = datetime(2025, 1, 1, tzinfo=timezone.utc)
SOURCE_EXTENSIONS = ('.jsx', '.tsx', '.js', '.ts')

# Greenplum
DB_SCHEMA = 'sandbox_prj_smart_insights'
DB_OWNER  = 'erd_gpdb_prj_smart_insights'
DB_GRANT  = 'erd_gpdb_prj_smart_insights_ro'
DB_BATCH  = 200

# Table names
TBL_PROJECTS = f'{DB_SCHEMA}.gwma_ui_projects'
TBL_DEPS     = f'{DB_SCHEMA}.gwma_ui_dependencies'
TBL_IMPORTS  = f'{DB_SCHEMA}.gwma_ui_imports'

# ── Column definitions ────────────────────────────────────────────────────────

PROJECTS_COLS: List[Tuple[str, str]] = [
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

DEPS_COLS: List[Tuple[str, str]] = [
    ('project_id',  'BIGINT'),
    ('name',        'TEXT'),
    ('path',        'TEXT'),
    ('dependency',  'TEXT'),   # one row per package name
]

IMPORTS_COLS: List[Tuple[str, str]] = [
    ('project_id',         'BIGINT'),
    ('name',               'TEXT'),
    ('path',               'TEXT'),
    ('import_filename',    'TEXT'),
    ('import_file_url',    'TEXT'),
    ('import_statement',   'TEXT'),
]

PROJECTS_COL_NAMES = [c[0] for c in PROJECTS_COLS]
DEPS_COL_NAMES     = [c[0] for c in DEPS_COLS]
IMPORTS_COL_NAMES  = [c[0] for c in IMPORTS_COLS]

# ── Import regex — every ES-module import form ────────────────────────────────
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
    'total_in_group'           : 0,
    'recent_projects'          : 0,
    'skipped_too_old'          : 0,
    'has_package_json'         : 0,
    'skipped_no_package_json'  : 0,
    'p1_errors'                : 0,
    'p1_timeouts'              : 0,
    'dep_rows_total'           : 0,
    'source_files_fetched'     : 0,
    'source_files_empty'       : 0,
    'source_files_error'       : 0,
    'import_rows_total'        : 0,
    'p2_errors'                : 0,
    'p2_timeouts'              : 0,
}

def _inc(key: str, n: int = 1) -> None:
    with _stats_lock:
        run_stats[key] += n

_gitlab_sem = threading.Semaphore(MAX_CONNECTIONS)


# ── Credentials ───────────────────────────────────────────────────────────────
logger.info('AUTH | requesting GitLab private token ...')
private_token = getpass.getpass('Enter your GitLab private token: ')

class _TimeoutSession(requests.Session):
    """requests.Session that injects a default timeout on every request."""
    def request(self, method, url, **kwargs):
        kwargs.setdefault('timeout', GITLAB_TIMEOUT)
        return super().request(method, url, **kwargs)

_session = _TimeoutSession()
_adapter  = HTTPAdapter(max_retries=1,
                        pool_connections=MAX_CONNECTIONS,
                        pool_maxsize=MAX_CONNECTIONS)
_session.mount('https://', _adapter)
_session.mount('http://',  _adapter)

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

def _ask(question: str) -> bool:
    """Print question and return True if user answers yes/y."""
    ans = input(f'\n  {question} (yes/no): ').strip().lower()
    return ans in ('yes', 'y')


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
    """Manual pagination — avoids all=True hanging on large repos."""
    items, page = [], 1
    while True:
        with _gitlab_sem:
            try:
                batch = project.repository_tree(
                    ref=ref, recursive=True,
                    per_page=PER_PAGE, page=page, get_all=False,
                )
            except Exception as exc:
                logger.warning('TREE | page=%d err | %s | %s',
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
    """Safe-coerce a value for psycopg2 binding."""
    if val is None:
        return None
    if isinstance(val, float) and val != val:      # NaN → NULL
        return None
    if isinstance(val, str) and val.strip() == '':
        return None
    return val


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
        logger.error('DB   | connection FAILED: %s', exc)
        return False


def _create_table(conn, fqtable: str, cols: List[Tuple[str, str]],
                  distributed_by: str = 'project_id') -> None:
    """DROP IF EXISTS → CREATE → OWNER → GRANT."""
    col_defs = ',\n    '.join(f'"{col}" {dtype}' for col, dtype in cols)
    with conn.cursor() as cur:
        logger.info('DB   | DROP TABLE IF EXISTS %s', fqtable)
        cur.execute(f'DROP TABLE IF EXISTS {fqtable};')
        sql = (f'CREATE TABLE {fqtable} (\n    {col_defs}\n)'
               f' DISTRIBUTED BY ({distributed_by});')
        logger.info('DB   | CREATE TABLE %s', fqtable)
        logger.debug('DDL  | %s', sql)
        cur.execute(sql)
        cur.execute(f'ALTER TABLE {fqtable} OWNER TO {DB_OWNER};')
        cur.execute(f'GRANT SELECT ON {fqtable} TO {DB_GRANT};')
    conn.commit()
    logger.info('DB   | table ready: %s', fqtable)


def _bulk_insert(conn, fqtable: str, col_names: List[str],
                 rows: List[Dict[str, Any]]) -> int:
    """
    Batch-insert rows into fqtable.
    Each row dict is converted to a tuple aligned with col_names.
    Returns number of rows inserted.
    """
    if not rows:
        logger.warning('DB   | no rows to insert into %s', fqtable)
        return 0

    col_list = ', '.join(f'"{c}"' for c in col_names)
    template = '(' + ', '.join(['%s'] * len(col_names)) + ')'
    sql      = f'INSERT INTO {fqtable} ({col_list}) VALUES %s'
    tuples   = [tuple(_coerce(r.get(c)) for c in col_names) for r in rows]

    # Log first row so we can verify column alignment
    logger.info('DB   | INSERT %d rows into %s', len(tuples), fqtable)
    logger.info('DB   | sample[0]: %s', dict(zip(col_names, tuples[0])))

    inserted = 0
    with conn.cursor() as cur:
        for start in range(0, len(tuples), DB_BATCH):
            batch = tuples[start : start + DB_BATCH]
            psycopg2.extras.execute_values(cur, sql, batch, template=template)
            inserted += len(batch)
            logger.info('DB   | %s : committed %d / %d', fqtable, inserted, len(tuples))
    conn.commit()
    logger.info('DB   | INSERT complete: %d rows in %s', inserted, fqtable)
    return inserted


def db_load_projects() -> List[Dict[str, Any]]:
    """
    Load all rows from gwma_ui_projects.
    Returns list of dicts with keys: project_id, name, path, path_with_namespace,
    web_url, default_branch  (all we need for Phase 2).
    """
    logger.info('DB   | loading project list from %s ...', TBL_PROJECTS)
    conn = _get_conn()
    rows = []
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(f'SELECT project_id, name, path, path_with_namespace, '
                    f'web_url, default_branch '
                    f'FROM {TBL_PROJECTS} ORDER BY project_id;')
        rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    logger.info('DB   | loaded %d projects', len(rows))
    return rows


# ── Phase 1 worker ────────────────────────────────────────────────────────────

def fetch_project_with_pkg(proj_ref: Any) -> Optional[Dict[str, Any]]:
    """
    1. Fetch full project object (description, forks, stars, open_issues, namespace).
    2. Attempt to fetch package.json on the default branch.
    3. Return a row dict if package.json exists, else None.
    """
    ns  = proj_ref.path_with_namespace

    with _gitlab_sem:
        try:
            project = client.projects.get(proj_ref.id)
        except Exception as exc:
            logger.error('P1 | project.get FAILED | %s | %s', ns, exc)
            _inc('p1_errors')
            return None

    ref  = project.default_branch or 'main'
    ns_d = project.namespace or {}

    with _gitlab_sem:
        try:
            project.files.raw(file_path='package.json', ref=ref)
            has_pkg = True
        except Exception:
            has_pkg = False

    if not has_pkg:
        logger.debug('P1 | NO  pkg | %s', project.path_with_namespace)
        _inc('skipped_no_package_json')
        return None

    logger.info('P1 | YES pkg | %s', project.path_with_namespace)
    _inc('has_package_json')

    return {
        # Internal (used by Phase 2 GitLab calls — not written to DB directly)
        '_project': project,
        '_ref'    : ref,
        # DB columns for gwma_ui_projects
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

def scan_project(row: Dict[str, Any]) -> Tuple[List[Dict], List[Dict]]:
    """
    Given a project row (from DB or Phase-1 memory), fetch package.json for
    dependencies and scan all source files for import statements.

    Returns:
      dep_rows    – list of {project_id, name, path, dependency}
                    one row per package in dependencies{}
      import_rows – list of {project_id, name, path, import_filename,
                              import_file_url, import_statement}
                    one row per import statement in any source file
    """
    project_id = row['project_id']
    name       = row['name']
    path       = row['path']
    web_url    = row['web_url']
    ref        = row['default_branch'] or 'main'
    ns         = row.get('path_with_namespace', path)

    # We may have a live project object from Phase 1, or need to fetch it from DB path
    project = row.get('_project')
    if project is None:
        with _gitlab_sem:
            try:
                project = client.projects.get(project_id)
            except Exception as exc:
                logger.error('P2 | project.get FAILED | %s | %s', ns, exc)
                _inc('p2_errors')
                return [], []

    logger.info('P2 | START | %s', ns)
    t0 = time.perf_counter()

    dep_base    = {'project_id': project_id, 'name': name, 'path': path}
    import_base = {'project_id': project_id, 'name': name, 'path': path}

    # ── Parse package.json for dependencies ──────────────────────────────────
    dep_rows: List[Dict] = []
    pkg_content = _raw_file(project, 'package.json', ref)
    if pkg_content:
        try:
            pkg_json = json.loads(pkg_content)
            # Collect both "dependencies" and "devDependencies" if present
            all_deps: Dict[str, str] = {}
            all_deps.update(pkg_json.get('dependencies', {}))
            all_deps.update(pkg_json.get('devDependencies', {}))
            for dep_name in sorted(all_deps.keys()):
                dep_rows.append({**dep_base, 'dependency': dep_name})
            logger.info('P2 | deps=%d | %s', len(dep_rows), ns)
        except json.JSONDecodeError as exc:
            logger.warning('P2 | bad package.json | %s | %s', ns, exc)
    else:
        logger.warning('P2 | could not read package.json | %s', ns)

    _inc('dep_rows_total', len(dep_rows))

    # ── Walk tree and scan source files ───────────────────────────────────────
    all_items = _repo_tree_pages(project, ref)
    src_items = [i for i in all_items
                 if i.get('type') == 'blob' and i['path'].endswith(SOURCE_EXTENSIONS)]
    logger.info('P2 | tree=%d  src=%d | %s', len(all_items), len(src_items), ns)

    import_rows: List[Dict] = []
    local_fetched = local_empty = local_error = local_stmts = 0

    def process_file(item: Dict) -> Optional[List[Dict]]:
        nonlocal local_fetched, local_empty, local_error, local_stmts
        fpath   = item['path']
        content = _raw_file(project, fpath, ref)
        if content is None:
            local_error += 1
            return None
        if not content.strip():
            local_empty += 1
            return None
        local_fetched += 1
        file_rows = []
        for m in ANY_IMPORT_RE.finditer(content):
            stmt = m.group(0).strip()
            file_rows.append({
                **import_base,
                'import_filename'  : fpath.split('/')[-1],
                'import_file_url'  : f'{web_url}/-/blob/{ref}/{fpath}',
                'import_statement' : stmt,
            })
            local_stmts += 1
        return file_rows or None

    with ThreadPoolExecutor(max_workers=FILE_WORKERS) as pool:
        for result in pool.map(process_file, src_items):
            if result:
                import_rows.extend(result)

    elapsed = time.perf_counter() - t0
    logger.info('P2 | DONE %.1fs | files: ok=%d empty=%d err=%d | imports=%d | %s',
                elapsed, local_fetched, local_empty, local_error, len(import_rows), ns)

    _inc('source_files_fetched', local_fetched)
    _inc('source_files_empty',   local_empty)
    _inc('source_files_error',   local_error)
    _inc('import_rows_total',    len(import_rows))

    return dep_rows, import_rows


# =============================================================================
#  MAIN PIPELINE
# =============================================================================

# ── DB connectivity check (always first) ──────────────────────────────────────
logger.info('DB   | testing connection ...')
if not db_test_connection():
    print('\n  ERROR: Cannot connect to Greenplum. Check credentials and network.')
    sys.exit(1)

t_total = time.perf_counter()

# =============================================================================
#  PHASE 1 GATE
# =============================================================================
print()
print('=' * 70)
print('  PHASE 1 — Identify UI projects (find package.json)')
print('=' * 70)
run_phase1 = _ask('Run Phase 1? Scans GitLab and recreates gwma_ui_projects table')

p1_rows: List[Dict[str, Any]] = []

if run_phase1:
    # ── Fetch project list from GitLab ────────────────────────────────────────
    logger.info('EXPORT | fetching project list for group %s ...', GROUP_PATH)
    t_p1 = time.perf_counter()

    group     = client.groups.get(GROUP_PATH)
    proj_refs = group.projects.list(include_subgroups=True, all=True, per_page=PER_PAGE)
    total_all = len(proj_refs)
    run_stats['total_in_group'] = total_all
    logger.info('EXPORT | %d projects in group', total_all)

    # Date filter — zero API calls
    recent_refs = [p for p in proj_refs if is_recent(p)]
    old_count   = total_all - len(recent_refs)
    run_stats['recent_projects'] = len(recent_refs)
    run_stats['skipped_too_old'] = old_count
    logger.info('DATE   | recent=%d  too_old=%d  cutoff=%s',
                len(recent_refs), old_count, UPDATED_SINCE.date())

    print(f'\n  Total projects   : {total_all}')
    print(f'  Recent (>= {UPDATED_SINCE.date()}) : {len(recent_refs)}')
    print(f'  Too old (skip)   : {old_count}')

    # ── Parallel Phase-1 processing ───────────────────────────────────────────
    print(f'\n  Checking package.json for {len(recent_refs)} projects ...\n')
    p1_done = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        fmap = {pool.submit(fetch_project_with_pkg, ref): ref for ref in recent_refs}
        for future in as_completed(fmap):
            p1_done += 1
            ref = fmap[future]
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
            if p1_done % PROGRESS_EVERY == 0 or p1_done == len(recent_refs):
                pct = 100 * p1_done // len(recent_refs)
                print(f'\r  P1: {p1_done}/{len(recent_refs)} ({pct}%) '
                      f'| has_pkg={len(p1_rows)} '
                      f'no_pkg={run_stats["skipped_no_package_json"]} '
                      f'err={run_stats["p1_errors"]} '
                      f'timeout={run_stats["p1_timeouts"]}   ',
                      end='', flush=True)
    print()

    t_p1_elapsed = time.perf_counter() - t_p1
    logger.info('PHASE 1 | done %.1fs | %d rows to insert', t_p1_elapsed, len(p1_rows))

    # ── Insert into gwma_ui_projects ──────────────────────────────────────────
    print(f'\n  Writing {len(p1_rows)} rows to {TBL_PROJECTS} ...')
    try:
        conn = _get_conn()
        _create_table(conn, TBL_PROJECTS, PROJECTS_COLS, 'project_id')
        inserted = _bulk_insert(conn, TBL_PROJECTS, PROJECTS_COL_NAMES, p1_rows)
        conn.close()
        print(f'  OK  {inserted} rows in {TBL_PROJECTS}')
    except Exception as exc:
        logger.error('DB | Phase-1 write FAILED: %s', exc, exc_info=True)
        print(f'\n  ERROR: {exc}')
        sys.exit(1)

    print()
    print('  PHASE 1 SUMMARY')
    print(f'  Projects with package.json : {len(p1_rows)}')
    print(f'  No package.json (skipped)  : {run_stats["skipped_no_package_json"]}')
    print(f'  Errors / timeouts          : {run_stats["p1_errors"] + run_stats["p1_timeouts"]}')
    print(f'  Elapsed                    : {t_p1_elapsed:.1f}s')

else:
    print('\n  Skipping Phase 1 — will use existing data in gwma_ui_projects.')
    t_p1_elapsed = 0.0

# =============================================================================
#  PHASE 2 GATE
# =============================================================================
print()
print('=' * 70)
print('  PHASE 2 — Parse dependencies & import statements')
print('=' * 70)
run_phase2 = _ask('Run Phase 2? Parses package.json deps + scans source files for imports')

if not run_phase2:
    t_elapsed = time.perf_counter() - t_total
    print(f'\n  Exiting. Total elapsed: {t_elapsed:.1f}s')
    sys.exit(0)

# ── Load project list for Phase 2 ─────────────────────────────────────────────
# If Phase 1 ran, we use its in-memory rows (already have _project objects).
# If Phase 1 was skipped, load from the DB.
if p1_rows:
    phase2_projects = p1_rows
    logger.info('P2 | using %d in-memory Phase-1 rows', len(phase2_projects))
else:
    try:
        db_rows = db_load_projects()
    except Exception as exc:
        logger.error('DB | failed to load projects: %s', exc, exc_info=True)
        print(f'\n  ERROR loading projects from DB: {exc}')
        sys.exit(1)

    if not db_rows:
        print(f'\n  ERROR: {TBL_PROJECTS} is empty. Run Phase 1 first.')
        sys.exit(1)

    # Convert DB rows to the dict format scan_project() expects.
    # No _project object — scan_project() will call client.projects.get() itself.
    phase2_projects = [
        {
            'project_id'         : int(r['project_id']),
            'name'               : r['name'],
            'path'               : r['path'],
            'path_with_namespace': r.get('path_with_namespace', r['path']),
            'web_url'            : r['web_url'],
            'default_branch'     : r['default_branch'] or 'main',
            '_project'           : None,   # will be fetched in scan_project()
        }
        for r in db_rows
    ]
    logger.info('P2 | loaded %d projects from DB', len(phase2_projects))

print(f'\n  Scanning {len(phase2_projects)} projects for dependencies and imports ...\n')

# ── Create Phase-2 tables ─────────────────────────────────────────────────────
try:
    conn = _get_conn()
    _create_table(conn, TBL_DEPS,    DEPS_COLS,    'project_id')
    _create_table(conn, TBL_IMPORTS, IMPORTS_COLS, 'project_id')
    conn.close()
except Exception as exc:
    logger.error('DB | Phase-2 table creation FAILED: %s', exc, exc_info=True)
    print(f'\n  ERROR creating Phase-2 tables: {exc}')
    sys.exit(1)

# ── Parallel Phase-2 processing ───────────────────────────────────────────────
t_p2 = time.perf_counter()

all_dep_rows    : List[Dict] = []
all_import_rows : List[Dict] = []
p2_done = 0

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
    fmap = {pool.submit(scan_project, row): row for row in phase2_projects}
    for future in as_completed(fmap):
        p2_done += 1
        row = fmap[future]
        ns  = row.get('path_with_namespace', row.get('path', '?'))
        try:
            dep_rows, import_rows = future.result(timeout=FUTURE_TIMEOUT)
        except FutureTimeout:
            logger.warning('P2 | TIMEOUT | %s', ns)
            _inc('p2_timeouts')
            dep_rows, import_rows = [], []
        except Exception as exc:
            logger.error('P2 | ERROR | %s | %s', ns, exc)
            _inc('p2_errors')
            dep_rows, import_rows = [], []

        all_dep_rows.extend(dep_rows)
        all_import_rows.extend(import_rows)

        if p2_done % PROGRESS_EVERY == 0 or p2_done == len(phase2_projects):
            pct = 100 * p2_done // len(phase2_projects)
            print(f'\r  P2: {p2_done}/{len(phase2_projects)} ({pct}%) '
                  f'| deps={len(all_dep_rows)} '
                  f'imports={len(all_import_rows)} '
                  f'err={run_stats["p2_errors"]} '
                  f'timeout={run_stats["p2_timeouts"]}   ',
                  end='', flush=True)
print()

t_p2_elapsed = time.perf_counter() - t_p2
logger.info('PHASE 2 | done %.1fs | deps=%d imports=%d',
            t_p2_elapsed, len(all_dep_rows), len(all_import_rows))

# ── Write Phase-2 results to Greenplum ───────────────────────────────────────
print(f'\n  Writing {len(all_dep_rows)} dependency rows to {TBL_DEPS} ...')
try:
    conn = _get_conn()
    dep_inserted = _bulk_insert(conn, TBL_DEPS, DEPS_COL_NAMES, all_dep_rows)
    conn.close()
    print(f'  OK  {dep_inserted} rows in {TBL_DEPS}')
except Exception as exc:
    logger.error('DB | deps write FAILED: %s', exc, exc_info=True)
    print(f'  ERROR: {exc}')

print(f'\n  Writing {len(all_import_rows)} import rows to {TBL_IMPORTS} ...')
try:
    conn = _get_conn()
    imp_inserted = _bulk_insert(conn, TBL_IMPORTS, IMPORTS_COL_NAMES, all_import_rows)
    conn.close()
    print(f'  OK  {imp_inserted} rows in {TBL_IMPORTS}')
except Exception as exc:
    logger.error('DB | imports write FAILED: %s', exc, exc_info=True)
    print(f'  ERROR: {exc}')

# ── Final summary ─────────────────────────────────────────────────────────────
t_elapsed = time.perf_counter() - t_total
print()
print('=' * 70)
print('  FINAL SUMMARY')
print('=' * 70)
if run_phase1:
    print(f'  gwma_ui_projects rows       : {len(p1_rows)}')
print(f'  gwma_ui_dependencies rows   : {len(all_dep_rows)}')
print(f'  gwma_ui_imports rows        : {len(all_import_rows)}')
print()
print(f'  Source files fetched        : {run_stats["source_files_fetched"]}')
print(f'  Source files errors         : {run_stats["source_files_error"]}')
print(f'  Phase 2 errors / timeouts   : {run_stats["p2_errors"]} / {run_stats["p2_timeouts"]}')
print()
print(f'  Phase 1 elapsed (s)         : {t_p1_elapsed:.1f}')
print(f'  Phase 2 elapsed (s)         : {t_p2_elapsed:.1f}')
print(f'  Total elapsed (s)           : {t_elapsed:.1f}')
print('=' * 70)
print()
print(f'  JOIN TABLES ON project_id:')
print(f'    {TBL_PROJECTS}')
print(f'    {TBL_DEPS}')
print(f'    {TBL_IMPORTS}')
print()

logger.info('=' * 70)
logger.info('GitLab Projects Export  v5 -- complete (%.1fs)', t_elapsed)
logger.info('=' * 70)
