# GitLab Projects Export  ── v5
#
# TABLES  (never dropped — INSERT only)
# ───────
#   gwma_ui_projects       Phase 1  one row per project with package.json
#   gwma_ui_dependencies   Phase 2  one row per dependency per project
#   gwma_ui_imports        Phase 2  one row per import statement per file
#
# COLUMN MIGRATION
# ────────────────
#   batch (INTEGER) and inserted_at (TIMESTAMP WITH TIME ZONE) are added to
#   all three tables at startup if they do not already exist.
#   Uses information_schema — safe for Greenplum / older PostgreSQL that
#   does not support ALTER TABLE ADD COLUMN IF NOT EXISTS.
#
# BATCH BEHAVIOUR
# ───────────────
#   • Tables are never dropped.  Each run appends a new batch.
#   • batch number = max(batch) + 1 for each table independently.
#   • inserted_at = UTC timestamp of this run (same value for the whole run).
#
# PHASE 2 STREAMING INSERT
# ────────────────────────
#   • Projects are scanned in parallel.  As soon as INSERT_EVERY=5 projects finish,
#     their rows are flushed to the DB immediately — no waiting for all projects.
#   • If the process hangs or dies, all previously flushed rows are safe.
#
# PROGRESS EXPORT (XLSX)
# ──────────────────────
#   After Phase 2, an Excel file is written:
#     phase2_progress_<timestamp>.xlsx
#   Columns: project_id | name | path | deps_inserted | imports_inserted
#   Values:  Yes / No   (Yes = that table received rows for this project this run)
#
# NEW COLUMNS — gwma_ui_projects
# ──────────────────────────────
#   division  TEXT  — 1st ancestor group below ubs/gwma root
#                     (e.g. "Global Wealth Management Americas")
#   stream    TEXT  — 2nd ancestor group
#                     (e.g. "Digital Client and Marketing Platforms")
#   crew      TEXT  — 3rd ancestor group
#                     (e.g. "Client Onboarding")
#   Display names (not URL slugs) are fetched via GitLab Groups API and cached.
#
# PHASE 3 — gwma_ui_imports enrichment
# ─────────────────────────────────────
#   Adds import_what (TEXT) and import_from (TEXT) columns to gwma_ui_imports.
#   Parses every import_statement row:
#     "import React, { useState } from 'react'"
#       → import_what = "React, { useState }"
#       → import_from = "react"
#   Side-effect imports ("import './style.css'") → import_what = NULL, import_from = "./style.css"
#   Runs in UPDATE batches; safe to re-run (skips already-enriched rows).
#
# SAMPLING
# ────────
#   SAMPLE_MODE = True   asks for a single project_id, writes to *_sample tables.
#   SAMPLE_MODE = False  normal full run.

import getpass
import json
import logging
import os
import re
import sys
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeout
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import gitlab
import openpyxl
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
MAX_WORKERS     = 15          # parallel project scan threads
FILE_WORKERS    = 8           # parallel file-fetch threads within one project
MAX_CONNECTIONS = 20          # hard cap on concurrent GitLab HTTP connections
GITLAB_TIMEOUT  = (10, 30)    # (connect_s, read_s) per HTTP request
FUTURE_TIMEOUT  = 120         # seconds before abandoning a stuck project future
INSERT_EVERY    = 5           # flush to DB after this many projects complete

UPDATED_SINCE = datetime(2025, 1, 1, tzinfo=timezone.utc)

# Greenplum
DB_SCHEMA = 'sandbox_prj_smart_insights'
DB_OWNER  = 'erd_gpdb_prj_smart_insights'
DB_GRANT  = 'erd_gpdb_prj_smart_insights_ro'
DB_BATCH  = 500               # rows per execute_values call

# Sampling
SAMPLE_MODE  = False
_TBL_SUFFIX  = '_sample' if SAMPLE_MODE else ''
TBL_PROJECTS = f'{DB_SCHEMA}.gwma_ui_projects{_TBL_SUFFIX}'
TBL_DEPS     = f'{DB_SCHEMA}.gwma_ui_dependencies{_TBL_SUFFIX}'
TBL_IMPORTS  = f'{DB_SCHEMA}.gwma_ui_imports{_TBL_SUFFIX}'

# ── Column definitions ────────────────────────────────────────────────────────
# batch + inserted_at are appended to every table automatically

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
    # Hierarchy columns derived from GitLab group ancestors
    # Example breadcrumb: UBS Agile / Global Wealth Mgmt Americas / Digital Client... / Client Onboarding / ...
    #   division = 1st ancestor group after ubs/gwma   (e.g. "Global Wealth Management Americas")
    #   stream   = 2nd ancestor group                  (e.g. "Digital Client and Marketing Platforms")
    #   crew     = 3rd ancestor group                  (e.g. "Client Onboarding")
    ('division',            'TEXT'),
    ('stream',              'TEXT'),
    ('crew',                'TEXT'),
    ('batch',               'INTEGER'),
    ('inserted_at',         'TIMESTAMP WITH TIME ZONE'),
]

DEPS_COLS: List[Tuple[str, str]] = [
    ('project_id',  'BIGINT'),
    ('name',        'TEXT'),
    ('path',        'TEXT'),
    ('tag',         'TEXT'),
    ('dependency',  'TEXT'),
    ('version',     'TEXT'),
    ('batch',       'INTEGER'),
    ('inserted_at', 'TIMESTAMP WITH TIME ZONE'),
]

IMPORTS_COLS: List[Tuple[str, str]] = [
    ('project_id',        'BIGINT'),
    ('name',              'TEXT'),
    ('path',              'TEXT'),
    ('import_filename',   'TEXT'),
    ('import_file_url',   'TEXT'),
    ('import_statement',  'TEXT'),
    ('batch',             'INTEGER'),
    ('inserted_at',       'TIMESTAMP WITH TIME ZONE'),
]

PROJECTS_COL_NAMES = [c[0] for c in PROJECTS_COLS]
DEPS_COL_NAMES     = [c[0] for c in DEPS_COLS]
IMPORTS_COL_NAMES  = [c[0] for c in IMPORTS_COLS]

# ── Import regex ──────────────────────────────────────────────────────────────
ANY_IMPORT_RE = re.compile(
    r"^[^\S\n]*import"
    r"(?:[^\S\n]+(?:"
        r"[\w\$_][\w\$_]*"
        r"(?:[^\S\n]*,[^\S\n]*"
          r"(?:\*[^\S\n]+as[^\S\n]+[\w\$_]+|\{[^\}\n]*\})"
        r")?"
        r"|[^\S\n]*\*[^\S\n]+as[^\S\n]+[\w\$_]+"
        r"|\{[^\}\n]*\}"
    r")[^\S\n]+from)?"
    r"[^\S\n]*['\"]([^'\"]+)['\"]"
    r"[^\S\n]*;?[^\n]*",
    re.MULTILINE,
)

# ── Thread-safe counters and progress tracking ────────────────────────────────
_stats_lock = threading.Lock()
run_stats: Dict[str, int] = {
    'total_in_group': 0, 'recent_projects': 0, 'skipped_too_old': 0,
    'has_package_json': 0, 'skipped_no_package_json': 0,
    'p1_errors': 0, 'p1_timeouts': 0,
    'dep_rows_total': 0, 'source_files_fetched': 0,
    'source_files_empty': 0, 'source_files_error': 0,
    'import_rows_total': 0, 'p2_errors': 0, 'p2_timeouts': 0,
    'deps_inserted': 0, 'imports_inserted': 0,
}

def _inc(key: str, n: int = 1) -> None:
    with _stats_lock:
        run_stats[key] += n

# Progress tracker: project_id → {name, path, deps_rows, imports_rows, ...}
# Pre-registered with zero counts so every scanned project appears in XLSX.
_progress_lock = threading.Lock()
_progress: Dict[int, Dict[str, Any]] = {}  # keyed by project_id

def _register_project(project_id: int, name: str, path: str) -> None:
    """Register a project with zero counts before scanning starts."""
    with _progress_lock:
        if project_id not in _progress:
            _progress[project_id] = {
                'name': name, 'path': path,
                'deps_inserted': False,    'deps_rows': 0,
                'imports_inserted': False, 'imports_rows': 0,
            }

def _record_progress(project_id: int, name: str, path: str,
                     deps_rows: int = 0, imports_rows: int = 0) -> None:
    """Accumulate per-project row counts and flip done flags."""
    with _progress_lock:
        if project_id not in _progress:
            _progress[project_id] = {
                'name': name, 'path': path,
                'deps_inserted': False,    'deps_rows': 0,
                'imports_inserted': False, 'imports_rows': 0,
            }
        p = _progress[project_id]
        if deps_rows > 0:
            p['deps_inserted'] = True
            p['deps_rows']    += deps_rows
        if imports_rows > 0:
            p['imports_inserted'] = True
            p['imports_rows']    += imports_rows

_gitlab_sem = threading.Semaphore(MAX_CONNECTIONS)


# ── Credentials ───────────────────────────────────────────────────────────────
logger.info('AUTH | requesting GitLab private token ...')
private_token = getpass.getpass('Enter your GitLab private token: ')

class _TimeoutSession(requests.Session):
    """requests.Session subclass — injects timeout on every request."""
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


# Cache of group_id → ancestor name list so we don't re-fetch for every project
_group_ancestor_cache: Dict[int, List[str]] = {}
_group_cache_lock = threading.Lock()

def _get_ancestor_names(namespace: Dict[str, Any]) -> List[str]:
    """
    Return the list of ancestor group *display names* (not slugs) for a project,
    ordered from top-level down.

    GitLab stores the full slug path in namespace['full_path'], e.g.:
      ubs/gwma/global-wealth-mgmt/digital-client/client-onboarding/overdrive/assisted-mock-ui

    We strip the fixed GROUP_PATH prefix ('ubs/gwma'), split the remainder,
    then resolve each slug to its display name by fetching the group object.
    Results are cached by namespace id to avoid repeated API calls.

    Returns a list that may have 0-N entries depending on nesting depth.
    """
    ns_id   = namespace.get('id')
    ns_path = namespace.get('full_path', '')

    # Try cache first
    with _group_cache_lock:
        if ns_id and ns_id in _group_ancestor_cache:
            return _group_ancestor_cache[ns_id]

    # Strip the root GROUP_PATH prefix from the full path
    prefix  = GROUP_PATH.rstrip('/')            # e.g. 'ubs/gwma'
    if ns_path.startswith(prefix + '/'):
        rel = ns_path[len(prefix) + 1:]         # everything after 'ubs/gwma/'
    elif ns_path == prefix:
        rel = ''
    else:
        rel = ns_path                            # unexpected structure — use as-is

    slugs = [s for s in rel.split('/') if s]    # intermediate group slugs

    # Resolve each slug to its display name by fetching its group object
    ancestor_names: List[str] = []
    current_path = prefix
    for slug in slugs:
        current_path = f'{current_path}/{slug}'
        with _gitlab_sem:
            try:
                grp = client.groups.get(current_path)
                ancestor_names.append(grp.name)
            except Exception:
                # Fallback: use the slug itself (converted to title case)
                ancestor_names.append(slug.replace('-', ' ').title())

    # Store in cache
    with _group_cache_lock:
        if ns_id:
            _group_ancestor_cache[ns_id] = ancestor_names

    return ancestor_names


def extract_division_stream_crew(namespace: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Return (division, stream, crew) from the group ancestor name list.
      division = ancestor[0]  (1st group below ubs/gwma root)
      stream   = ancestor[1]  (2nd group)
      crew     = ancestor[2]  (3rd group)
    Any missing level is returned as None.
    """
    names = _get_ancestor_names(namespace)
    division = names[0] if len(names) > 0 else None
    stream   = names[1] if len(names) > 1 else None
    crew     = names[2] if len(names) > 2 else None
    return division, stream, crew

def _raw_file(project: Any, path: str, ref: str) -> Optional[str]:
    """Fetch file as UTF-8, normalising all line endings to \\n."""
    with _gitlab_sem:
        try:
            raw = project.files.raw(file_path=path, ref=ref).decode('utf-8', errors='replace')
            return raw.replace('\r\n', '\n').replace('\r', '\n')
        except Exception as exc:
            logger.debug('FILE | err | %s | %s', path, exc)
            return None

def _repo_tree_pages(project: Any, ref: str) -> List[Dict]:
    """Manual tree pagination — avoids all=True hanging on large repos."""
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
    if val is None:
        return None
    if isinstance(val, float) and val != val:
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

def _column_exists(conn, fqtable: str, column: str) -> bool:
    """Return True if column already exists in the table (Greenplum-safe check)."""
    schema, table = fqtable.split('.', 1) if '.' in fqtable else (None, fqtable)
    with conn.cursor() as cur:
        if schema:
            cur.execute(
                "SELECT COUNT(*) FROM information_schema.columns "
                "WHERE table_schema = %s AND table_name = %s AND column_name = %s;",
                (schema, table, column)
            )
        else:
            cur.execute(
                "SELECT COUNT(*) FROM information_schema.columns "
                "WHERE table_name = %s AND column_name = %s;",
                (table, column)
            )
        return cur.fetchone()[0] > 0


def db_add_columns_if_missing(conn, fqtable: str) -> None:
    """
    One-time migration: adds missing columns to an existing table.

    Columns added to ALL tables if absent:
      batch (INTEGER), inserted_at (TIMESTAMP WITH TIME ZONE)

    Columns added only to gwma_ui_projects if absent:
      division (TEXT), stream (TEXT), crew (TEXT)

    Uses information_schema — compatible with Greenplum / older PostgreSQL
    (no ALTER TABLE ADD COLUMN IF NOT EXISTS needed).
    Safe to call on every run; no-ops when columns already exist.
    """
    # Columns for every table
    cols_all = [
        ('batch',       'INTEGER'),
        ('inserted_at', 'TIMESTAMP WITH TIME ZONE'),
    ]
    # Extra columns only for the projects table
    cols_projects_extra = [
        ('division', 'TEXT'),
        ('stream',   'TEXT'),
        ('crew',     'TEXT'),
    ]

    target_cols = cols_all[:]
    # Check if this is the projects table (handles both _sample and normal)
    if fqtable.endswith('gwma_ui_projects') or fqtable.endswith('gwma_ui_projects_sample'):
        target_cols += cols_projects_extra

    for col, dtype in target_cols:
        if _column_exists(conn, fqtable, col):
            logger.info('MIGRATE | column "%s" already exists in %s — skipped', col, fqtable)
        else:
            try:
                with conn.cursor() as cur:
                    cur.execute(f'ALTER TABLE {fqtable} ADD COLUMN "{col}" {dtype};')
                conn.commit()
                logger.info('MIGRATE | added column "%s" %s to %s', col, dtype, fqtable)
            except Exception as exc:
                conn.rollback()
                logger.error('MIGRATE | could not add "%s" to %s: %s', col, fqtable, exc)


def _ensure_table(conn, fqtable: str, cols: List[Tuple[str, str]],
                  distributed_by: str = 'project_id') -> None:
    """
    CREATE TABLE IF NOT EXISTS, then run the batch/inserted_at migration.
    Never drops or truncates the table.
    """
    col_defs = ',\n    '.join(f'"{col}" {dtype}' for col, dtype in cols)
    sql_create = (
        f'CREATE TABLE IF NOT EXISTS {fqtable} (\n    {col_defs}\n)'
        f' DISTRIBUTED BY ({distributed_by});'
    )
    with conn.cursor() as cur:
        cur.execute(sql_create)
        try:
            cur.execute(f'ALTER TABLE {fqtable} OWNER TO {DB_OWNER};')
            cur.execute(f'GRANT SELECT ON {fqtable} TO {DB_GRANT};')
        except Exception:
            pass
    conn.commit()
    # Add batch + inserted_at if missing (safe for pre-existing tables)
    db_add_columns_if_missing(conn, fqtable)
    logger.info('DB   | table ready: %s', fqtable)

def _get_next_batch(conn, fqtable: str) -> int:
    """Return max(batch)+1 from the table, or 1 if the table is empty."""
    try:
        with conn.cursor() as cur:
            cur.execute(f'SELECT COALESCE(MAX("batch"), 0) FROM {fqtable};')
            max_batch = cur.fetchone()[0]
        return int(max_batch) + 1
    except Exception as exc:
        logger.warning('DB   | could not read max batch from %s: %s — using 1', fqtable, exc)
        return 1

def _bulk_insert(conn, fqtable: str, col_names: List[str],
                 rows: List[Dict[str, Any]]) -> int:
    """INSERT rows, return count inserted."""
    if not rows:
        return 0
    col_list = ', '.join(f'"{c}"' for c in col_names)
    template = '(' + ', '.join(['%s'] * len(col_names)) + ')'
    sql      = f'INSERT INTO {fqtable} ({col_list}) VALUES %s'
    tuples   = [tuple(_coerce(r.get(c)) for c in col_names) for r in rows]
    inserted = 0
    with conn.cursor() as cur:
        for start in range(0, len(tuples), DB_BATCH):
            chunk = tuples[start : start + DB_BATCH]
            psycopg2.extras.execute_values(cur, sql, chunk, template=template)
            inserted += len(chunk)
    conn.commit()
    logger.info('DB   | inserted %d rows into %s', inserted, fqtable)
    return inserted

def db_load_projects() -> List[Dict[str, Any]]:
    """Load project list from TBL_PROJECTS for Phase 2."""
    logger.info('DB   | loading project list from %s ...', TBL_PROJECTS)
    conn = _get_conn()
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(
            f'SELECT DISTINCT ON (project_id) '
            f'project_id, name, path, path_with_namespace, web_url, default_branch '
            f'FROM {TBL_PROJECTS} ORDER BY project_id, batch DESC;'
        )
        rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    logger.info('DB   | loaded %d projects', len(rows))
    return rows


# ── Streaming flush helper ────────────────────────────────────────────────────

class _StreamingFlusher:
    """
    Accumulates (dep_rows, import_rows) from completed projects.
    When INSERT_EVERY projects have accumulated, flushes them to the DB immediately.
    Thread-safe.  Also handles the final flush for leftover rows.
    """
    def __init__(self, batch_deps: int, batch_imports: int, inserted_at: datetime):
        self._lock         = threading.Lock()
        self._pending_deps : List[Dict] = []
        self._pending_imps : List[Dict] = []
        self._batch_deps   = batch_deps
        self._batch_imports= batch_imports
        self._inserted_at  = inserted_at
        self._project_count = 0
        self.total_deps    = 0
        self.total_imports = 0

    def add(self, dep_rows: List[Dict], import_rows: List[Dict],
            project_id: int, name: str, path: str) -> None:
        """Add rows for one completed project. Flushes every INSERT_EVERY projects."""
        with self._lock:
            self._pending_deps.extend(dep_rows)
            self._pending_imps.extend(import_rows)
            self._project_count += 1
            should_flush = (self._project_count % INSERT_EVERY == 0)

        if should_flush:
            self._flush()

    def _flush(self) -> None:
        with self._lock:
            deps_to_write = self._pending_deps[:]
            imps_to_write = self._pending_imps[:]
            self._pending_deps.clear()
            self._pending_imps.clear()

        if not deps_to_write and not imps_to_write:
            return

        ts = self._inserted_at.isoformat()
        for r in deps_to_write:
            r['batch']       = self._batch_deps
            r['inserted_at'] = ts
        for r in imps_to_write:
            r['batch']       = self._batch_imports
            r['inserted_at'] = ts

        # Write deps and record per-project row counts
        if deps_to_write:
            try:
                conn = _get_conn()
                n = _bulk_insert(conn, TBL_DEPS, DEPS_COL_NAMES, deps_to_write)
                conn.close()
                with self._lock:
                    self.total_deps += n
                _inc('deps_inserted', n)
                from collections import Counter
                pid_counts = Counter(r['project_id'] for r in deps_to_write)
                for pid, cnt in pid_counts.items():
                    sample = next(r for r in deps_to_write if r['project_id'] == pid)
                    _record_progress(pid, sample['name'], sample['path'],
                                     deps_rows=cnt, imports_rows=0)
                logger.info('FLUSH | deps: %d rows for %d projects committed', n, len(pid_counts))
            except Exception as exc:
                logger.error('FLUSH | deps write FAILED: %s', exc, exc_info=True)

        # Write imports and record per-project row counts
        if imps_to_write:
            try:
                conn = _get_conn()
                n = _bulk_insert(conn, TBL_IMPORTS, IMPORTS_COL_NAMES, imps_to_write)
                conn.close()
                with self._lock:
                    self.total_imports += n
                _inc('imports_inserted', n)
                from collections import Counter
                pid_counts = Counter(r['project_id'] for r in imps_to_write)
                for pid, cnt in pid_counts.items():
                    sample = next(r for r in imps_to_write if r['project_id'] == pid)
                    _record_progress(pid, sample['name'], sample['path'],
                                     deps_rows=0, imports_rows=cnt)
                logger.info('FLUSH | imports: %d rows for %d projects committed', n, len(pid_counts))
            except Exception as exc:
                logger.error('FLUSH | imports write FAILED: %s', exc, exc_info=True)

    def final_flush(self) -> None:
        """Flush any remaining rows after the parallel scan finishes."""
        logger.info('FLUSH | final flush ...')
        self._flush()


# ── Phase 1 worker ────────────────────────────────────────────────────────────

def fetch_project_with_pkg(proj_ref: Any) -> Optional[Dict[str, Any]]:
    ns = proj_ref.path_with_namespace
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
        _inc('skipped_no_package_json')
        return None

    logger.info('P1 | YES pkg | %s', project.path_with_namespace)
    _inc('has_package_json')

    division, stream, crew = extract_division_stream_crew(ns_d)
    logger.info('P1 | hierarchy | div=%s  stream=%s  crew=%s | %s',
                division, stream, crew, project.path_with_namespace)

    return {
        '_project': project, '_ref': ref,
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
        'division'           : division,
        'stream'             : stream,
        'crew'               : crew,
    }


# ── Phase 2 worker ────────────────────────────────────────────────────────────

def scan_project(row: Dict[str, Any]) -> Tuple[List[Dict], List[Dict]]:
    """
    Returns (dep_rows, import_rows).
    Rows do NOT yet have batch/inserted_at — the flusher stamps those.
    """
    project_id = row['project_id']
    name       = row['name']
    path       = row['path']
    web_url    = row['web_url']
    ref        = row.get('default_branch') or row.get('_ref') or 'main'
    ns         = row.get('path_with_namespace', path)

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
    t0       = time.perf_counter()
    dep_base = {'project_id': project_id, 'name': name, 'path': path}

    # ── Dependencies ─────────────────────────────────────────────────────────
    dep_rows: List[Dict] = []
    pkg = _raw_file(project, 'package.json', ref)
    if pkg:
        try:
            pj = json.loads(pkg)
            for tag in ('dependencies', 'devDependencies'):
                section = pj.get(tag, {})
                if not isinstance(section, dict):
                    continue
                for dep_name in sorted(section.keys()):
                    dep_rows.append({**dep_base,
                                     'tag': tag, 'dependency': dep_name,
                                     'version': section[dep_name]})
        except json.JSONDecodeError as exc:
            logger.warning('P2 | bad package.json | %s | %s', ns, exc)
    else:
        logger.warning('P2 | could not read package.json | %s', ns)
    _inc('dep_rows_total', len(dep_rows))

    # ── Import statements ─────────────────────────────────────────────────────
    all_items = _repo_tree_pages(project, ref)
    blobs     = [i for i in all_items if i.get('type') == 'blob']
    logger.info('P2 | tree=%d blobs=%d | %s', len(all_items), len(blobs), ns)

    import_rows: List[Dict]  = []
    lf = le = lerr = lstmts = 0
    imp_base = {'project_id': project_id, 'name': name, 'path': path}

    def process_file(item: Dict) -> Optional[List[Dict]]:
        nonlocal lf, le, lerr, lstmts
        fpath   = item['path']
        content = _raw_file(project, fpath, ref)
        if content is None:
            lerr += 1
            return None
        if not content.strip():
            le += 1
            return None
        lf += 1
        file_rows = []
        for m in ANY_IMPORT_RE.finditer(content):
            file_rows.append({
                **imp_base,
                'import_filename' : fpath.split('/')[-1],
                'import_file_url' : f'{web_url}/-/blob/{ref}/{fpath}',
                'import_statement': m.group(0).strip(),
            })
            lstmts += 1
        return file_rows or None

    with ThreadPoolExecutor(max_workers=FILE_WORKERS) as pool:
        for result in pool.map(process_file, blobs):
            if result:
                import_rows.extend(result)

    elapsed = time.perf_counter() - t0
    logger.info('P2 | DONE %.1fs | ok=%d empty=%d err=%d deps=%d imports=%d | %s',
                elapsed, lf, le, lerr, len(dep_rows), len(import_rows), ns)

    _inc('source_files_fetched', lf)
    _inc('source_files_empty',   le)
    _inc('source_files_error',   lerr)
    _inc('import_rows_total',    len(import_rows))
    return dep_rows, import_rows


# ── Progress XLSX export ──────────────────────────────────────────────────────

def write_progress_xlsx(progress: Dict[int, Dict], filepath: str) -> None:
    """
    Write per-project insert status to an Excel file.
    Every scanned project appears — zero-row projects included.

    Columns:
      project_id | name | path
      imports_table_insert_done     (Yes / No)
      imports_rows_inserted         (count, 0 if none)
      dependencies_table_insert_done (Yes / No)
      dependencies_rows_inserted    (count, 0 if none)
    """
    wb  = openpyxl.Workbook()
    ws  = wb.active
    ws.title = 'Phase2 Progress'

    # Styles
    bold      = openpyxl.styles.Font(bold=True, color='FFFFFF')
    hdr_fill  = openpyxl.styles.PatternFill('solid', fgColor='2E75B6')
    yes_fill  = openpyxl.styles.PatternFill('solid', fgColor='C6EFCE')
    no_fill   = openpyxl.styles.PatternFill('solid', fgColor='FFCCCC')
    zero_font = openpyxl.styles.Font(color='888888')
    center    = openpyxl.styles.Alignment(horizontal='center')
    thin      = openpyxl.styles.Side(style='thin', color='D0D0D0')
    border    = openpyxl.styles.Border(left=thin, right=thin, top=thin, bottom=thin)

    headers    = ['project_id', 'name', 'path',
                  'imports_table_insert_done', 'imports_rows_inserted',
                  'dependencies_table_insert_done', 'dependencies_rows_inserted']
    col_widths = [14, 40, 60, 30, 24, 34, 28]

    # Header row
    for ci, (h, w) in enumerate(zip(headers, col_widths), 1):
        cell = ws.cell(row=1, column=ci, value=h)
        cell.font = bold; cell.fill = hdr_fill
        cell.alignment = center; cell.border = border
        ws.column_dimensions[cell.column_letter].width = w

    ws.freeze_panes = 'A2'   # freeze header

    # Data rows — sorted by project_id, zero-row projects always included
    for ri, (pid, info) in enumerate(sorted(progress.items()), start=2):
        imp_done  = info.get('imports_inserted', False)
        dep_done  = info.get('deps_inserted',    False)
        imp_count = info.get('imports_rows',     0)
        dep_count = info.get('deps_rows',        0)
        values = [pid, info.get('name',''), info.get('path',''),
                  'Yes' if imp_done else 'No', imp_count,
                  'Yes' if dep_done else 'No', dep_count]
        for ci, val in enumerate(values, 1):
            cell = ws.cell(row=ri, column=ci, value=val)
            cell.border = border
            if ci == 4:    # imports_table_insert_done
                cell.fill = yes_fill if imp_done else no_fill
                cell.alignment = center
            elif ci == 6:  # dependencies_table_insert_done
                cell.fill = yes_fill if dep_done else no_fill
                cell.alignment = center
            elif ci in (5, 7):  # count columns
                cell.alignment = center
                if val == 0:
                    cell.font = zero_font

    # Totals row
    total_row = len(progress) + 2
    ws.cell(row=total_row, column=1, value='TOTAL').font = openpyxl.styles.Font(bold=True)
    ws.cell(row=total_row, column=5,
            value=sum(v.get('imports_rows', 0) for v in progress.values())
            ).font = openpyxl.styles.Font(bold=True)
    ws.cell(row=total_row, column=7,
            value=sum(v.get('deps_rows', 0) for v in progress.values())
            ).font = openpyxl.styles.Font(bold=True)

    wb.save(filepath)
    logger.info('XLSX | written: %s  (%d projects)', filepath, len(progress))



# =============================================================================
#  MAIN PIPELINE
# =============================================================================

logger.info('DB   | testing connection ...')
if not db_test_connection():
    print('\n  ERROR: Cannot connect to Greenplum.')
    sys.exit(1)

# ── One-time migration: add batch + inserted_at to all 3 tables if missing ────
# This runs every startup but is a no-op when columns already exist.
# Safe for Greenplum (no ADD COLUMN IF NOT EXISTS — uses information_schema check).
print()
print('  Checking / migrating table columns ...')
try:
    _mig_conn = _get_conn()
    for _tbl in [TBL_PROJECTS, TBL_DEPS, TBL_IMPORTS]:
        try:
            db_add_columns_if_missing(_mig_conn, _tbl)
        except Exception as _e:
            # Table may not exist yet — that is fine, _ensure_table will create it
            logger.debug('MIGRATE | %s not yet created: %s', _tbl, _e)
    _mig_conn.close()
    print('  Column check complete.')
except Exception as _mig_exc:
    logger.error('MIGRATE | migration check failed: %s', _mig_exc)
    print(f'  WARNING: migration check failed: {_mig_exc}')

t_total      = time.perf_counter()
run_ts       = datetime.now(timezone.utc)   # single timestamp for this whole run
run_ts_str   = run_ts.strftime('%Y%m%d_%H%M%S')

# =============================================================================
#  SAMPLE MODE
# =============================================================================
if SAMPLE_MODE:
    print()
    print('=' * 70)
    print('  SAMPLE MODE — single project test')
    print(f'  Tables: {TBL_PROJECTS}  |  {TBL_DEPS}  |  {TBL_IMPORTS}')
    print('=' * 70)

    while True:
        _pid_str = input('\n  Enter project_id to sample: ').strip()
        try:
            SAMPLE_PROJECT_ID = int(_pid_str)
            break
        except ValueError:
            print('  Please enter a numeric project_id.')

    logger.info('SAMPLE | fetching project %d ...', SAMPLE_PROJECT_ID)
    with _gitlab_sem:
        try:
            _sp = client.projects.get(SAMPLE_PROJECT_ID)
        except Exception as _exc:
            print(f'\n  ERROR: Could not fetch project {SAMPLE_PROJECT_ID}: {_exc}')
            sys.exit(1)

    _ref  = _sp.default_branch or 'main'
    _ns_d = _sp.namespace or {}
    _s_p1row = {
        '_project': _sp, '_ref': _ref,
        'project_id': int(_sp.id), 'name': _sp.name, 'path': _sp.path,
        'path_with_namespace': _sp.path_with_namespace,
        'group_path': _ns_d.get('full_path'), 'web_url': _sp.web_url,
        'description': _sp.description, 'visibility': _sp.visibility,
        'archived': bool(_sp.archived),
        'created_at': _sp.created_at, 'last_activity_at': _sp.last_activity_at,
        'updated_at': _sp.updated_at, 'default_branch': _ref,
        'forks_count'      : int(_sp.forks_count)       if _sp.forks_count       is not None else None,
        'star_count'       : int(_sp.star_count)        if _sp.star_count        is not None else None,
        'open_issues_count': int(_sp.open_issues_count) if _sp.open_issues_count is not None else None,
        'team_name': extract_team_name(_sp.web_url), 'package_json': 'Yes',
    }

    conn = _get_conn()
    _ensure_table(conn, TBL_PROJECTS, PROJECTS_COLS)
    batch_p = _get_next_batch(conn, TBL_PROJECTS)
    _s_p1row['batch']       = batch_p
    _s_p1row['inserted_at'] = run_ts.isoformat()
    _bulk_insert(conn, TBL_PROJECTS, PROJECTS_COL_NAMES, [_s_p1row])
    conn.close()
    print(f'  OK  1 row in {TBL_PROJECTS} (batch={batch_p})')

    _s_dep_rows, _s_import_rows = scan_project(_s_p1row)

    conn = _get_conn()
    _ensure_table(conn, TBL_DEPS, DEPS_COLS)
    batch_d = _get_next_batch(conn, TBL_DEPS)
    for r in _s_dep_rows:
        r['batch'] = batch_d; r['inserted_at'] = run_ts.isoformat()
    _bulk_insert(conn, TBL_DEPS, DEPS_COL_NAMES, _s_dep_rows)
    conn.close()

    conn = _get_conn()
    _ensure_table(conn, TBL_IMPORTS, IMPORTS_COLS)
    batch_i = _get_next_batch(conn, TBL_IMPORTS)
    for r in _s_import_rows:
        r['batch'] = batch_i; r['inserted_at'] = run_ts.isoformat()
    _bulk_insert(conn, TBL_IMPORTS, IMPORTS_COL_NAMES, _s_import_rows)
    conn.close()

    print(f'  OK  {len(_s_dep_rows)} rows in {TBL_DEPS} (batch={batch_d})')
    print(f'  OK  {len(_s_import_rows)} rows in {TBL_IMPORTS} (batch={batch_i})')

    _register_project(int(_sp.id), _sp.name, _sp.path)
    _record_progress(int(_sp.id), _sp.name, _sp.path,
                     deps_rows=len(_s_dep_rows),
                     imports_rows=len(_s_import_rows))
    xlsx_path = f'phase2_progress_{run_ts_str}.xlsx'
    write_progress_xlsx(_progress, xlsx_path)

    t_elapsed = time.perf_counter() - t_total
    print()
    print('=' * 70)
    print('  SAMPLE RUN COMPLETE')
    print('=' * 70)
    print(f'  Project   : {_sp.path_with_namespace}')
    print(f'  Batch P   : {batch_p}  |  Batch D: {batch_d}  |  Batch I: {batch_i}')
    print(f'  Deps rows : {len(_s_dep_rows)}')
    print(f'  Imp rows  : {len(_s_import_rows)}')
    print(f'  Progress  : {xlsx_path}')
    print(f'  Elapsed   : {t_elapsed:.1f}s')
    print('=' * 70)
    sys.exit(0)

# =============================================================================
#  PHASE 1 GATE
# =============================================================================
print()
print('=' * 70)
print('  PHASE 1 — Identify UI projects (find package.json)')
print('=' * 70)
run_phase1 = _ask('Run Phase 1? Scans GitLab and appends to gwma_ui_projects table')

p1_rows: List[Dict[str, Any]] = []
t_p1_elapsed = 0.0

if run_phase1:
    logger.info('EXPORT | fetching project list for group %s ...', GROUP_PATH)
    t_p1 = time.perf_counter()

    group     = client.groups.get(GROUP_PATH)
    proj_refs = group.projects.list(include_subgroups=True, all=True, per_page=PER_PAGE)
    total_all = len(proj_refs)
    run_stats['total_in_group'] = total_all

    recent_refs = [p for p in proj_refs if is_recent(p)]
    old_count   = total_all - len(recent_refs)
    run_stats['recent_projects'] = len(recent_refs)
    run_stats['skipped_too_old'] = old_count
    logger.info('DATE   | recent=%d  too_old=%d  cutoff=%s',
                len(recent_refs), old_count, UPDATED_SINCE.date())
    print(f'\n  Total: {total_all}  Recent: {len(recent_refs)}  Too old: {old_count}')
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
            if p1_done % 10 == 0 or p1_done == len(recent_refs):
                pct = 100 * p1_done // len(recent_refs)
                print(f'\r  P1: {p1_done}/{len(recent_refs)} ({pct}%) '
                      f'| has_pkg={len(p1_rows)} '
                      f'no_pkg={run_stats["skipped_no_package_json"]}   ',
                      end='', flush=True)
    print()

    t_p1_elapsed = time.perf_counter() - t_p1
    logger.info('PHASE 1 | done %.1fs | %d rows', t_p1_elapsed, len(p1_rows))

    # Ensure table exists, get next batch, stamp rows, insert
    conn = _get_conn()
    _ensure_table(conn, TBL_PROJECTS, PROJECTS_COLS)
    batch_projects = _get_next_batch(conn, TBL_PROJECTS)
    conn.close()
    logger.info('DB   | projects batch = %d', batch_projects)

    for r in p1_rows:
        r['batch']       = batch_projects
        r['inserted_at'] = run_ts.isoformat()

    print(f'\n  Writing {len(p1_rows)} rows to {TBL_PROJECTS} (batch={batch_projects}) ...')
    try:
        conn = _get_conn()
        n = _bulk_insert(conn, TBL_PROJECTS, PROJECTS_COL_NAMES, p1_rows)
        conn.close()
        print(f'  OK  {n} rows in {TBL_PROJECTS}')
    except Exception as exc:
        logger.error('DB | Phase-1 write FAILED: %s', exc, exc_info=True)
        print(f'\n  ERROR: {exc}')
        sys.exit(1)

    print()
    print(f'  Total projects in group               : {total_all}')
    print(f'  Modified since {UPDATED_SINCE.date()}        : {len(recent_refs)}')
    print(f'    └─ Too old (skipped)                : {old_count}')
    print(f'  UI projects (have package.json)       : {len(p1_rows)}')
    print(f'    └─ No package.json (skipped)        : {run_stats["skipped_no_package_json"]}')
    print(f'    └─ Errors / timeouts                : {run_stats["p1_errors"] + run_stats["p1_timeouts"]}')
    print(f'  Phase 1 elapsed (s)                   : {t_p1_elapsed:.1f}')
    print(f'  Batch                                 : {batch_projects}')

else:
    print('\n  Skipping Phase 1 — reading counts from existing DB table ...')
    # Read total-project and UI-project counts from what's already in the DB
    try:
        _cnt_conn = _get_conn()
        with _cnt_conn.cursor() as _cur:
            _cur.execute(f'SELECT COUNT(DISTINCT project_id) FROM {TBL_PROJECTS};')
            _db_ui_count = _cur.fetchone()[0]
        _cnt_conn.close()
        print(f'  UI projects in {TBL_PROJECTS}: {_db_ui_count}')
        logger.info('P1 SKIP | UI projects in DB: %d', _db_ui_count)
    except Exception as _ce:
        logger.warning('P1 SKIP | could not count DB projects: %s', _ce)
        print(f'  (Could not read count from DB: {_ce})')

# =============================================================================
#  PHASE 2 GATE
# =============================================================================
print()
print('=' * 70)
print('  PHASE 2 — Parse dependencies & import statements')
print('=' * 70)
run_phase2 = _ask('Run Phase 2? Parses package.json + scans files for imports')

if not run_phase2:
    print('\n  Skipping Phase 2 — continuing to Phase 3.')
    # Declare sentinel variables so the final summary block doesn't crash
    flusher       = type('_F', (), {'total_deps': 0, 'total_imports': 0})()
    batch_deps    = 0
    batch_imports = 0
    t_p2_elapsed  = 0.0

# ── Load project list (only if Phase 2 is running) ───────────────────────────
if run_phase2:
    if p1_rows:
        phase2_projects = p1_rows
        logger.info('P2 | using %d in-memory Phase-1 rows', len(phase2_projects))
    else:
        try:
            db_rows = db_load_projects()
        except Exception as exc:
            logger.error('DB | failed to load projects: %s', exc, exc_info=True)
            print(f'\n  ERROR: {exc}')
            sys.exit(1)
        if not db_rows:
            print(f'\n  ERROR: {TBL_PROJECTS} is empty. Run Phase 1 first.')
            sys.exit(1)
        phase2_projects = [
            {'project_id': int(r['project_id']), 'name': r['name'], 'path': r['path'],
             'path_with_namespace': r.get('path_with_namespace', r['path']),
             'web_url': r['web_url'], 'default_branch': r['default_branch'] or 'main',
             '_project': None}
            for r in db_rows
        ]
        logger.info('P2 | loaded %d projects from DB', len(phase2_projects))

    # ── Ensure Phase-2 tables exist and get batch numbers ────────────────────
    conn = _get_conn()
    _ensure_table(conn, TBL_DEPS,    DEPS_COLS)
    _ensure_table(conn, TBL_IMPORTS, IMPORTS_COLS)
    batch_deps    = _get_next_batch(conn, TBL_DEPS)
    batch_imports = _get_next_batch(conn, TBL_IMPORTS)
    conn.close()
    logger.info('DB   | deps batch=%d  imports batch=%d', batch_deps, batch_imports)

    print(f'\n  Scanning {len(phase2_projects)} projects ...')
    print(f'  Deps batch    : {batch_deps}')
    print(f'  Imports batch : {batch_imports}')
    print(f'  Flush every   : {INSERT_EVERY} projects\n')

    # Pre-register every project with zero counts
    for _proj in phase2_projects:
        _register_project(_proj['project_id'], _proj['name'], _proj['path'])

    # ── Parallel Phase-2 with streaming flush ────────────────────────────────
    t_p2     = time.perf_counter()
    flusher  = _StreamingFlusher(batch_deps, batch_imports, run_ts)
    p2_done  = 0
    p2_total = len(phase2_projects)

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

            flusher.add(dep_rows, import_rows,
                        row['project_id'], row['name'], row['path'])

            if p2_done % 5 == 0 or p2_done == p2_total:
                pct = 100 * p2_done // p2_total
                print(f'\r  P2: {p2_done}/{p2_total} ({pct}%) '
                      f'| deps_rows={flusher.total_deps} '
                      f'imp_rows={flusher.total_imports} '
                      f'err={run_stats["p2_errors"]} '
                      f'timeout={run_stats["p2_timeouts"]}   ',
                      end='', flush=True)

    print()
    flusher.final_flush()   # flush any leftover rows
    t_p2_elapsed = time.perf_counter() - t_total - t_p1_elapsed

    # ── Write progress XLSX ──────────────────────────────────────────────────
    xlsx_path = f'phase2_progress_{run_ts_str}.xlsx'
    write_progress_xlsx(_progress, xlsx_path)
    print(f'\n  Progress file : {xlsx_path}')

# ── Final summary ─────────────────────────────────────────────────────────────
t_elapsed = time.perf_counter() - t_total
print()
print('=' * 70)
print('  FINAL SUMMARY')
print('=' * 70)
if run_phase1:
    print(f'  Total projects in group               : {total_all}')
    print(f'  Modified since {UPDATED_SINCE.date()}        : {len(recent_refs)}')
    print(f'  UI projects (have package.json)       : {len(p1_rows)}')
    print(f'  gwma_ui_projects rows (this run)      : {len(p1_rows)}  (batch={batch_projects})')
if run_phase2:
    print(f'  gwma_ui_dependencies rows             : {flusher.total_deps}  (batch={batch_deps})')
    print(f'  gwma_ui_imports rows                  : {flusher.total_imports}  (batch={batch_imports})')
print(f'  inserted_at                           : {run_ts.isoformat()}')
print()
if run_phase2:
    projects_done    = sum(1 for v in _progress.values() if v['deps_inserted'] and v['imports_inserted'])
    projects_deps_ok = sum(1 for v in _progress.values() if v['deps_inserted'])
    projects_imps_ok = sum(1 for v in _progress.values() if v['imports_inserted'])
    print(f'  Projects: both tables done            : {projects_done}')
    print(f'  Projects: deps table done             : {projects_deps_ok}')
    print(f'  Projects: imports table done          : {projects_imps_ok}')
    print()
    print(f'  Source files fetched                  : {run_stats["source_files_fetched"]}')
    print(f'  P2 errors / timeouts                  : {run_stats["p2_errors"]} / {run_stats["p2_timeouts"]}')
print(f'  Phase 1 elapsed (s)                   : {t_p1_elapsed:.1f}')
if run_phase2:
    print(f'  Phase 2 elapsed (s)                   : {t_p2_elapsed:.1f}')
print(f'  Total elapsed (s)                     : {t_elapsed:.1f}')
print('=' * 70)
print()
print(f'  JOIN TABLES ON project_id:')
print(f'    {TBL_PROJECTS}')
print(f'    {TBL_DEPS}')
print(f'    {TBL_IMPORTS}')
if run_phase2 and _progress:
    print(f'  Progress XLSX: {xlsx_path}')

# =============================================================================
#  PHASE 3 — Enrich gwma_ui_imports with import_what + import_from columns
# =============================================================================
# This runs independently of Phase 1 and Phase 2.
# It adds two columns to the imports table (if they do not already exist) and
# populates them by parsing the existing import_statement values in-place:
#
#   import_statement : "import React, { useState } from 'react'"
#   import_what      : "React, { useState }"        (everything between 'import' and 'from')
#   import_from      : "react"                       (the quoted module specifier)
#
# For side-effect imports ("import './styles.css'") import_what is NULL and
# import_from is the quoted path.
#
# The UPDATE runs in batches of PHASE3_BATCH rows for safety.

PHASE3_BATCH = 1000   # rows per UPDATE batch

def _parse_import_statement(stmt: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse a single import statement string into (import_what, import_from).

    Handles all ES-module forms:
      import React from 'react'                  → ('React', 'react')
      import { useState } from 'react'           → ('{ useState }', 'react')
      import React, { useState } from 'react'    → ('React, { useState }', 'react')
      import * as Icons from '@uwr/icons'        → ('* as Icons', '@uwr/icons')
      import './styles.css'                      → (None, './styles.css')
    """
    stmt = stmt.strip()
    # Match: import <what> from '<from>'
    m_from = re.match(
        r'^import\s+(.+?)\s+from\s+[\'\"]([^\'\"]+)[\'\"](\s*;?)?$',
        stmt, re.DOTALL
    )
    if m_from:
        what = m_from.group(1).strip()
        frm  = m_from.group(2).strip()
        return what, frm
    # Match: import '<from>'  (side-effect)
    m_side = re.match(
        r'^import\s+[\'\"]([^\'\"]+)[\'\"](\s*;?)?$',
        stmt
    )
    if m_side:
        return None, m_side.group(1).strip()
    # Fallback: could not parse
    return None, None


def run_phase3_enrich_imports() -> None:
    """
    Phase 3: add import_what + import_from columns to gwma_ui_imports and
    populate them from the existing import_statement column.

    Steps:
      1. Add columns if missing (via db_add_columns_if_missing variant).
      2. SELECT rows where import_what IS NULL (unpopulated).
      3. Parse each import_statement in Python.
      4. UPDATE in batches of PHASE3_BATCH.
    """
    print()
    print('=' * 70)
    print('  PHASE 3 — Enrich imports table: import_what + import_from')
    print('=' * 70)
    logger.info('PHASE 3 | start | table=%s', TBL_IMPORTS)

    conn = _get_conn()

    # Step 1: ensure columns exist
    for col, dtype in [('import_what', 'TEXT'), ('import_from', 'TEXT')]:
        if _column_exists(conn, TBL_IMPORTS, col):
            logger.info('PHASE 3 | column "%s" already exists', col)
        else:
            try:
                with conn.cursor() as cur:
                    cur.execute(f'ALTER TABLE {TBL_IMPORTS} ADD COLUMN "{col}" {dtype};')
                conn.commit()
                logger.info('PHASE 3 | added column "%s" %s', col, dtype)
                print(f'  Added column: {col} ({dtype})')
            except Exception as exc:
                conn.rollback()
                logger.error('PHASE 3 | could not add column "%s": %s', col, exc)
                print(f'  ERROR adding column {col}: {exc}')
                conn.close()
                return

    # Step 2: count rows needing enrichment
    with conn.cursor() as cur:
        cur.execute(
            f'SELECT COUNT(*) FROM {TBL_IMPORTS} WHERE import_what IS NULL AND import_from IS NULL;'
        )
        total_pending = cur.fetchone()[0]

    if total_pending == 0:
        print('  All rows already enriched — nothing to do.')
        logger.info('PHASE 3 | all rows already enriched')
        conn.close()
        return

    print(f'  Rows to enrich: {total_pending}')
    logger.info('PHASE 3 | rows to enrich: %d', total_pending)

    # Step 3 + 4: fetch in batches, parse, UPDATE row-by-row
    # We do NOT use ctid-based UPDATE FROM VALUES — Greenplum distributed tables
    # can produce "multiple updates to a row by the same query" errors when the
    # same ctid appears across segments.
    # Instead we: (a) add a SERIAL surrogate key column, (b) do one UPDATE per
    # row keyed on that surrogate, or fall back to parsing + re-inserting.
    # Simplest Greenplum-safe approach: UPDATE with a unique indexed column.
    # We use (project_id, import_statement) as the match key — not ctid.
    # Since the same (project_id, import_statement) may appear multiple times
    # (same import in multiple files), we add an auto-increment helper column
    # _enrich_id for uniqueness, then drop it after enrichment.

    updated_total = 0

    # Add a temporary serial column to guarantee unique row identity
    _has_tmp_col = False
    if not _column_exists(conn, TBL_IMPORTS, '_enrich_id'):
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f'ALTER TABLE {TBL_IMPORTS} ADD COLUMN _enrich_id BIGSERIAL;'
                )
            conn.commit()
            _has_tmp_col = True
            logger.info('PHASE 3 | added temporary _enrich_id column')
        except Exception as exc:
            conn.rollback()
            logger.warning('PHASE 3 | could not add _enrich_id, falling back to statement match: %s', exc)
    else:
        _has_tmp_col = True   # already exists from a previous interrupted run

    if _has_tmp_col:
        # Fast path: unique key = _enrich_id
        offset = 0
        while True:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute(
                    f'SELECT _enrich_id, import_statement FROM {TBL_IMPORTS} '
                    f'WHERE import_what IS NULL AND import_from IS NULL '
                    f'ORDER BY _enrich_id LIMIT %s OFFSET %s;',
                    (PHASE3_BATCH, offset)
                )
                batch = cur.fetchall()
            if not batch:
                break

            # Parse in Python
            updates = []
            for r in batch:
                what, frm = _parse_import_statement(r['import_statement'] or '')
                updates.append((what, frm, int(r['_enrich_id'])))

            # UPDATE keyed on _enrich_id — each _enrich_id is unique so no
            # "multiple updates to same row" problem
            with conn.cursor() as cur:
                cur.executemany(
                    f'UPDATE {TBL_IMPORTS} '
                    f'SET import_what = %s, import_from = %s '
                    f'WHERE _enrich_id = %s;',
                    updates
                )
            conn.commit()

            updated_total += len(batch)
            pct = 100 * updated_total // total_pending if total_pending else 100
            print(f'\r  Enriched: {updated_total}/{total_pending} ({pct}%)   ',
                  end='', flush=True)
            logger.info('PHASE 3 | enriched %d / %d', updated_total, total_pending)

            if len(batch) < PHASE3_BATCH:
                break
            offset += PHASE3_BATCH

        # Drop the temporary helper column
        try:
            with conn.cursor() as cur:
                cur.execute(f'ALTER TABLE {TBL_IMPORTS} DROP COLUMN _enrich_id;')
            conn.commit()
            logger.info('PHASE 3 | dropped temporary _enrich_id column')
        except Exception as exc:
            conn.rollback()
            logger.warning('PHASE 3 | could not drop _enrich_id: %s', exc)

    else:
        # Fallback: match on import_statement text (may update multiple rows with
        # same statement, but that is correct — they all get the same parsed values)
        offset = 0
        while True:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute(
                    f'SELECT DISTINCT import_statement FROM {TBL_IMPORTS} '
                    f'WHERE import_what IS NULL AND import_from IS NULL '
                    f'LIMIT %s OFFSET %s;',
                    (PHASE3_BATCH, offset)
                )
                batch = cur.fetchall()
            if not batch:
                break

            updates = []
            for r in batch:
                what, frm = _parse_import_statement(r['import_statement'] or '')
                updates.append((what, frm, r['import_statement']))

            with conn.cursor() as cur:
                cur.executemany(
                    f'UPDATE {TBL_IMPORTS} '
                    f'SET import_what = %s, import_from = %s '
                    f'WHERE import_statement = %s '
                    f'  AND import_what IS NULL AND import_from IS NULL;',
                    updates
                )
            conn.commit()

            updated_total += len(batch)
            pct = 100 * updated_total // total_pending if total_pending else 100
            print(f'\r  Enriched: {updated_total}/{total_pending} ({pct}%)   ',
                  end='', flush=True)
            logger.info('PHASE 3 | enriched %d / %d', updated_total, total_pending)

            if len(batch) < PHASE3_BATCH:
                break
            offset += PHASE3_BATCH

    print()
    conn.close()
    print(f'  Phase 3 complete — {updated_total} rows enriched in {TBL_IMPORTS}')
    logger.info('PHASE 3 | done — %d rows enriched', updated_total)


# ── Ask user whether to run Phase 3 ──────────────────────────────────────────
print()
print('=' * 70)
print('  PHASE 3 — Parse import_statement into import_what + import_from')
print('=' * 70)
print(f'  Table: {TBL_IMPORTS}')
print('  Adds columns import_what and import_from (one-time if missing).')
print('  Populates unpopulated rows by parsing import_statement.')

_run_p3 = _ask('Run Phase 3? (enriches import_what + import_from in imports table)')
if _run_p3:
    run_phase3_enrich_imports()
else:
    print('  Skipping Phase 3.')

logger.info('=' * 70)
logger.info('GitLab Projects Export  v5 -- complete (%.1fs)', t_elapsed)
logger.info('=' * 70)
