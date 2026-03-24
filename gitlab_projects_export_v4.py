# GitLab Projects Export  ── v4
#
# Performance & correctness improvements over v3:
#
#  SPEED
#  ─────
#  • Phase 1 (lightweight)  : list all group projects, date-filter on proj_ref attributes alone
#    (zero extra API calls).  Then check package.json for every survivor IN PARALLEL across
#    MAX_WORKERS threads — one file fetch each, nothing more.
#  • Phase 2 (file scan)    : only projects that passed Phase 1 enter the file scan.
#    Within each project the full tree is fetched once; then all source files are fetched
#    in parallel with FILE_WORKERS threads.
#  • Both phases run concurrently via ThreadPoolExecutor + as_completed, so network I/O
#    never blocks on a single slow project.
#  • repository_tree uses per_page=100 (GitLab max) with all=True to minimise pagination.
#  • Full project object (client.projects.get) is only fetched in Phase 2, after we already
#    know package.json exists — cutting the API call count dramatically.
#
#  CORRECTNESS — import capture
#  ─────────────────────────────
#  Previous regex only matched  import { Named } from '...'
#  New ANY_IMPORT_RE matches ALL standard ES-module import forms:
#    import DefaultExport           from '...'
#    import * as Ns                 from '...'
#    import DefaultExport, { Named} from '...'
#    import { Named }               from '...'
#    import '...'                   (side-effect)
#  Every import line in a file is now captured as its own row, not just @uwr/@ubs.websdk ones.
#  The 'library' column is derived from the from-path so filtering is still easy.

import getpass
import json
import logging
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import gitlab
import pandas as pd

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger('gitlab_export')
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('gitlab').setLevel(logging.WARNING)

logger.info('=' * 70)
logger.info('GitLab Projects Export  v4  — started')
logger.info('=' * 70)

# ── Configuration ─────────────────────────────────────────────────────────────
GITLAB_URL   = 'https://devcloud.ubs.net'
GROUP_PATH   = 'ubs/gwma'
OUTPUT_XLSX  = 'gitlab_projects_export.xlsx'
OUTPUT_JSON  = 'gitlab_projects_export.json'
SAMPLE_XLSX  = 'sample_gitlab_projects_export.xlsx'
SAMPLE_JSON  = 'sample_gitlab_projects_export.json'

PER_PAGE     = 100   # GitLab max page size
MAX_WORKERS  = 30    # parallel threads across projects  (raise if network allows)
FILE_WORKERS = 15    # parallel file-fetch threads per project

# Only projects whose updated_at >= UPDATED_SINCE are considered
UPDATED_SINCE = datetime(2025, 1, 1, tzinfo=timezone.utc)

ENABLE_SAMPLE_MODE = False

SOURCE_EXTENSIONS = ('.jsx', '.tsx', '.js', '.ts')

# ── Import regex — captures ALL ES-module import statement forms ──────────────
#
#   Supported forms (any combination of default / namespace / named / side-effect):
#     import 'module'
#     import Default from 'module'
#     import * as Ns from 'module'
#     import Default, { A, B } from 'module'
#     import { A, B } from 'module'
#     import React, { useState } from 'react'          ← previously missed
#     import BatchTable from './BatchTable'             ← previously missed
#
#   The capture group 1 holds the full statement (stripped).
#   The capture group 2 holds the module specifier (the string inside quotes).
#
ANY_IMPORT_RE = re.compile(
    r"^[ \t]*"
    r"import"
    r"(?:"
        r"\s+(?:"
            # default + optional named  OR  namespace  OR  named-only
            r"[\w\$_][\w\$_]*(?:\s*,\s*(?:\*\s+as\s+[\w\$_]+|\{[^}]*\}))?"  # Default or Default,{...}
            r"|\*\s+as\s+[\w\$_][\w\$_]*"                                     # * as Ns
            r"|\{[^}]*\}"                                                       # { Named }
        r")"
        r"\s+from"
    r")?"
    r"\s*['\"]([^'\"]+)['\"]"      # <── module specifier (group 1)
    r"\s*;?[^\n]*",
    re.MULTILINE,
)

logger.info('Config | GitLab URL    : %s', GITLAB_URL)
logger.info('Config | Group path    : %s', GROUP_PATH)
logger.info('Config | Updated since : %s', UPDATED_SINCE.date())
logger.info('Config | Workers       : %d project / %d file', MAX_WORKERS, FILE_WORKERS)
logger.info('Config | Source exts   : %s', ', '.join(SOURCE_EXTENSIONS))
logger.info('Config | Sample mode   : %s', ENABLE_SAMPLE_MODE)

# ── Thread-safe run-wide counters ─────────────────────────────────────────────
_stats_lock = threading.Lock()
run_stats: Dict[str, int] = {
    'total_in_group'           : 0,
    'recent_projects'          : 0,   # updated_at >= UPDATED_SINCE
    'skipped_too_old'          : 0,
    'has_package_json'         : 0,
    'skipped_no_package_json'  : 0,
    'source_files_fetched'     : 0,
    'source_files_empty'       : 0,
    'source_files_error'       : 0,
    'import_statements_total'  : 0,
    'output_rows'              : 0,
}

def _inc(key: str, n: int = 1) -> None:
    with _stats_lock:
        run_stats[key] += n

# ── Auth ──────────────────────────────────────────────────────────────────────
logger.info('-' * 70)
logger.info('AUTH | requesting GitLab private token …')
private_token = getpass.getpass('Enter your GitLab private token: ')
client = gitlab.Gitlab(GITLAB_URL, private_token=private_token)
logger.info('AUTH | client ready for %s', GITLAB_URL)


# ── Utility helpers ───────────────────────────────────────────────────────────

def _parse_dt(s: str) -> Optional[datetime]:
    """Parse a GitLab ISO-8601 timestamp to an aware datetime, or None."""
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace('Z', '+00:00'))
    except Exception:
        return None


def is_recent(proj_ref: Any) -> bool:
    """Return True when proj_ref.updated_at >= UPDATED_SINCE."""
    dt = _parse_dt(getattr(proj_ref, 'updated_at', None) or '')
    if dt is None:
        return True   # keep if unparseable
    return dt >= UPDATED_SINCE


def extract_team_name(web_url: str) -> str:
    try:
        segs = web_url.rstrip('/').split('//', 1)[-1].split('/')[1:]
        return segs[-2] if len(segs) >= 2 else ''
    except Exception:
        return ''


def _raw_file(project: Any, path: str, ref: str) -> Optional[str]:
    """Fetch one file as UTF-8 text.  Returns None on any error."""
    try:
        raw = project.files.raw(file_path=path, ref=ref)
        content = raw.decode('utf-8', errors='replace')
        logger.debug('  FILE | ok  %d chars | %s', len(content), path)
        return content
    except Exception as exc:
        logger.debug('  FILE | err | %s | %s', path, exc)
        return None


# ── Phase 1 helpers ───────────────────────────────────────────────────────────

def check_package_json(proj_ref: Any) -> Optional[Dict[str, Any]]:
    """
    Lightweight Phase-1 worker.

    Fetches package.json via the raw-file endpoint (one API call).
    Returns a dict with extracted metadata if the file exists, else None.
    Uses proj_ref (the lightweight listing object) — no full project.get() call.
    """
    ns  = proj_ref.path_with_namespace
    ref = getattr(proj_ref, 'default_branch', None) or 'main'

    logger.debug('P1 | checking package.json | %s [%s]', ns, ref)

    # Use the files API directly on the lightweight object
    try:
        raw     = proj_ref.files.raw(file_path='package.json', ref=ref)
        content = raw.decode('utf-8', errors='replace')
    except Exception:
        logger.info('P1 | NO  package.json | %s', ns)
        _inc('skipped_no_package_json')
        return None

    logger.debug('P1 | found package.json (%d chars) | %s', len(content), ns)

    try:
        pkg  = json.loads(content)
        deps: Dict[str, str] = pkg.get('dependencies', {})
    except json.JSONDecodeError as exc:
        logger.warning('P1 | bad JSON in package.json | %s | %s', ns, exc)
        deps = {}

    dep_count      = len(deps)
    dependency_str = ', '.join(sorted(deps.keys())) if deps else '-'
    uwr_deps       = [d for d in deps if d.startswith('@uwr/')]
    int_ext        = 'internal' if uwr_deps else 'external'

    logger.info('P1 | YES package.json | %s | int_ext=%s | deps=%d | uwr=%s',
                ns, int_ext, dep_count, uwr_deps or 'none')
    _inc('has_package_json')

    return {
        'proj_ref'   : proj_ref,
        'ref'        : ref,
        'int_ext'    : int_ext,
        'dependency' : dependency_str,
    }


# ── Phase 2 helpers ───────────────────────────────────────────────────────────

def _derive_library(module_path: str) -> str:
    """Classify the module specifier into 'uwr', 'websdk', both, or ''."""
    s = module_path.lower()
    found = []
    if s.startswith('@uwr/'):
        found.append('uwr')
    if s.startswith('@ubs.websdk/'):
        found.append('websdk')
    return ', '.join(found)


def scan_and_build_rows(phase1: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Phase-2 worker.

    Fetches the full project object (one API call), walks the source tree,
    fetches all source files in parallel, extracts EVERY import statement
    from each file, and returns one output row per import statement.

    If no import statements exist in any file, returns one summary row
    with blank import columns.
    """
    proj_ref   = phase1['proj_ref']
    ref        = phase1['ref']
    int_ext    = phase1['int_ext']
    dependency = phase1['dependency']
    ns         = proj_ref.path_with_namespace

    t0 = time.perf_counter()
    logger.info('P2 | START | %s [ref=%s]', ns, ref)

    # Fetch full project object (needed for web_url, namespace details, etc.)
    project   = client.projects.get(proj_ref.id)
    namespace = project.namespace or {}
    web_url   = project.web_url
    team_name = extract_team_name(web_url)

    logger.debug('P2 | fetched full project | id=%s  team=%s  visibility=%s  archived=%s',
                 project.id, team_name or '(none)', project.visibility, project.archived)

    # ── Walk the repo tree ────────────────────────────────────────────────────
    try:
        all_items = project.repository_tree(ref=ref, recursive=True, all=True, per_page=PER_PAGE)
    except Exception as exc:
        logger.error('P2 | tree fetch FAILED | %s | %s', ns, exc)
        return []

    src_items = [
        i for i in all_items
        if i.get('type') == 'blob' and i['path'].endswith(SOURCE_EXTENSIONS)
    ]
    logger.info('P2 | tree: %d items, %d source files | %s', len(all_items), len(src_items), ns)

    for item in src_items:
        logger.debug('P2 | source: %s', item['path'])

    # ── Base row template (metadata shared across all rows for this project) ──
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
        logger.info('P2 | no source files — 1 summary row | %s', ns)
        return [{**base, 'library': '', 'import_filename': '',
                 'import_file_url': '', 'import_statement': ''}]

    # ── Fetch every source file in parallel and extract import statements ─────
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
            logger.debug('P2 | FILE empty | %s', path)
            return None

        local_fetched += 1
        file_rows = []

        for m in ANY_IMPORT_RE.finditer(content):
            stmt       = m.group(0).strip()
            module_str = m.group(1)                 # the quoted module path
            library    = _derive_library(module_str)
            preview    = stmt[:100] + ('…' if len(stmt) > 100 else '')
            logger.info('  IMPORT | %s | %s', path, preview)

            file_rows.append({
                **base,
                'library'          : library,
                'import_filename'  : path.split('/')[-1],
                'import_file_url'  : f'{web_url}/-/blob/{ref}/{path}',
                'import_statement' : stmt,
            })
            local_stmts += 1

        if file_rows:
            logger.debug('P2 | FILE %d import(s) | %s', len(file_rows), path)
        else:
            logger.debug('P2 | FILE no imports   | %s', path)

        return file_rows or None

    with ThreadPoolExecutor(max_workers=FILE_WORKERS) as pool:
        for result in pool.map(process_file, src_items):
            if result:
                import_rows.extend(result)

    elapsed = time.perf_counter() - t0
    logger.info('P2 | DONE %.1fs | files: fetched=%d empty=%d err=%d | imports=%d | %s',
                elapsed, local_fetched, local_empty, local_error, len(import_rows), ns)

    _inc('source_files_fetched', local_fetched)
    _inc('source_files_empty',   local_empty)
    _inc('source_files_error',   local_error)
    _inc('import_statements_total', local_stmts)

    if not import_rows:
        logger.info('P2 | no import statements found — 1 summary row | %s', ns)
        return [{**base, 'library': '', 'import_filename': '',
                 'import_file_url': '', 'import_statement': ''}]

    _inc('output_rows', len(import_rows))
    return import_rows


# ── DataFrame & I/O helpers ───────────────────────────────────────────────────

def build_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    logger.debug('DF | building from %d rows', len(rows))
    df = pd.DataFrame(rows)
    for col in ('created_at', 'last_activity_at', 'updated_at'):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce', utc=True).dt.tz_localize(None)
    df.sort_values(['group_path', 'name', 'import_filename', 'import_statement'],
                   inplace=True, ignore_index=True)
    return df


def write_xlsx(df: pd.DataFrame, path: str) -> None:
    logger.info('XLSX | writing %d rows → %s', len(df), path)
    t0 = time.perf_counter()
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='projects')
        ws = writer.sheets['projects']
        for col_cells in ws.columns:
            w = max((len(str(c.value)) if c.value is not None else 0) for c in col_cells)
            ws.column_dimensions[col_cells[0].column_letter].width = min(w + 4, 80)
    logger.info('XLSX | done %.1fs → %s', time.perf_counter() - t0, path)


def write_json(rows: List[Dict[str, Any]], path: str) -> None:
    logger.info('JSON | writing %d rows → %s', len(rows), path)
    t0 = time.perf_counter()
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2, default=str, separators=(',', ': '))
    logger.info('JSON | done %.1fs → %s', time.perf_counter() - t0, path)


logger.info('All helpers defined.')


# ── Sample mode ───────────────────────────────────────────────────────────────

sample_rows: List[Dict[str, Any]] = []

if ENABLE_SAMPLE_MODE:
    print('\n' + '=' * 60)
    print('SAMPLE MODE — test individual projects before full run')
    print('=' * 60)
    logger.info('SAMPLE | entering sample mode')

    while True:
        project_name = input('\nEnter project name to sample: ').strip()
        if not project_name:
            print('No name entered — skipping.')
        else:
            logger.info('SAMPLE | searching for %r', project_name)
            try:
                grp       = client.groups.get(GROUP_PATH)
                candidates = grp.projects.list(
                    search=project_name, include_subgroups=True, all=True, per_page=50
                )
                logger.info('SAMPLE | %d candidate(s) returned', len(candidates))
                match = next(
                    (p for p in candidates if p.name.lower() == project_name.lower()),
                    candidates[0] if candidates else None,
                )
            except Exception as exc:
                logger.error('SAMPLE | search failed: %s', exc)
                match = None

            if not match:
                print(f'⚠️  No project found matching "{project_name}".')
            else:
                logger.info('SAMPLE | matched: %s', match.path_with_namespace)
                if not is_recent(match):
                    print(f'⚠️  Project too old (updated_at < {UPDATED_SINCE.date()}): '
                          f'{match.path_with_namespace}')
                else:
                    p1 = check_package_json(match)
                    if p1 is None:
                        print(f'⚠️  No package.json: {match.path_with_namespace}')
                    else:
                        rows_out = scan_and_build_rows(p1)
                        sample_rows.extend(rows_out)
                        dep_preview = rows_out[0]['dependency']
                        print(f'\n✅  {rows_out[0]["path_with_namespace"]}')
                        print(f'    int_ext      : {rows_out[0]["int_ext"]}')
                        print(f'    dependency   : {dep_preview[:80]}'
                              f'{"…" if len(dep_preview) > 80 else ""}')
                        print(f'    import rows  : {sum(1 for r in rows_out if r["import_statement"])}')
                        df_s = build_dataframe(sample_rows)
                        write_xlsx(df_s, SAMPLE_XLSX)
                        write_json(sample_rows, SAMPLE_JSON)
                        print(f'    Sample files updated: {SAMPLE_XLSX} | {SAMPLE_JSON}')
                        logger.info('SAMPLE | files updated')

        if input('\nRun another sample? (yes/no): ').strip().lower() not in ('yes', 'y'):
            break

    logger.info('SAMPLE | done — %d row(s)', len(sample_rows))
    print(f'\nSample mode done — {len(sample_rows)} row(s).')
    if input('Proceed with full export? (yes/no): ').strip().lower() not in ('yes', 'y'):
        raise SystemExit('Stopped after sample mode.')
else:
    logger.info('Sample mode disabled.')


# ═══════════════════════════════════════════════════════════════════════════════
#  FULL EXPORT — two-phase pipeline
#
#  Phase 1  (fast, parallel)  : date-filter on proj_ref attrs → check package.json
#  Phase 2  (parallel)        : tree walk + file scan for Phase-1 survivors only
# ═══════════════════════════════════════════════════════════════════════════════

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

logger.info('EXPORT | %d recent (updated >= %s), %d too old — skipped',
            len(recent_refs), UPDATED_SINCE.date(), old_count)

# ── Phase 1: parallel package.json checks ────────────────────────────────────
logger.info('─' * 70)
logger.info('PHASE 1 | checking package.json for %d recent projects …', len(recent_refs))
t_p1 = time.perf_counter()

phase1_results: List[Dict[str, Any]] = []
p1_done = 0

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
    future_to_ref = {pool.submit(check_package_json, ref): ref for ref in recent_refs}
    for future in as_completed(future_to_ref):
        p1_done += 1
        ref = future_to_ref[future]
        try:
            result = future.result()
            if result:
                phase1_results.append(result)
                logger.debug('P1 | %d/%d ✓ has pkg.json | %s',
                             p1_done, len(recent_refs), ref.path_with_namespace)
            else:
                logger.debug('P1 | %d/%d – no pkg.json  | %s',
                             p1_done, len(recent_refs), ref.path_with_namespace)
        except Exception as exc:
            logger.error('P1 | %d/%d ✗ EXCEPTION | %s | %s',
                         p1_done, len(recent_refs), ref.path_with_namespace, exc, exc_info=True)

logger.info('PHASE 1 | done in %.1fs | %d/%d projects have package.json',
            time.perf_counter() - t_p1, len(phase1_results), len(recent_refs))

# ── Phase 2: parallel file scan for package.json survivors ───────────────────
logger.info('─' * 70)
logger.info('PHASE 2 | scanning source files for %d projects …', len(phase1_results))
t_p2 = time.perf_counter()

all_rows: List[Dict[str, Any]] = []
p2_done = 0

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
    future_to_p1 = {pool.submit(scan_and_build_rows, p1): p1 for p1 in phase1_results}
    for future in as_completed(future_to_p1):
        p2_done += 1
        p1 = future_to_p1[future]
        ns = p1['proj_ref'].path_with_namespace
        try:
            rows = future.result()
            if rows:
                all_rows.extend(rows)
                logger.info('P2 | %d/%d ✓ %d row(s) | %s',
                            p2_done, len(phase1_results), len(rows), ns)
        except Exception as exc:
            logger.error('P2 | %d/%d ✗ EXCEPTION | %s | %s',
                         p2_done, len(phase1_results), ns, exc, exc_info=True)

logger.info('PHASE 2 | done in %.1fs', time.perf_counter() - t_p2)

t_elapsed = time.perf_counter() - t_total
included_projects = len({r['project_id'] for r in all_rows})
run_stats['output_rows'] = len(all_rows)


# ── Summary ───────────────────────────────────────────────────────────────────

df = build_dataframe(all_rows)

summary_lines = [
    ('Total projects in group',                               run_stats['total_in_group']),
    (f'Projects updated since {UPDATED_SINCE.date()}',       run_stats['recent_projects']),
    (f'  └─ skipped (too old)',                              run_stats['skipped_too_old']),
    ('',                                                      ''),
    ('Projects with package.json (Phase 1 survivors)',        run_stats['has_package_json']),
    ('  └─ skipped (no package.json, but recently updated)', run_stats['skipped_no_package_json']),
    ('',                                                      ''),
    ('Source files fetched successfully',                     run_stats['source_files_fetched']),
    ('Source files empty',                                    run_stats['source_files_empty']),
    ('Source files fetch error',                              run_stats['source_files_error']),
    ('Import statements captured',                            run_stats['import_statements_total']),
    ('',                                                      ''),
    ('Distinct projects in output',                           included_projects),
    ('Total output rows (one per import statement)',          run_stats['output_rows']),
    ('Rows with import statement',                            int((df['import_statement'] != '').sum())),
    ('Total elapsed time (s)',                                f'{t_elapsed:.1f}'),
]

logger.info('═' * 70)
logger.info('SUMMARY')
for label, value in summary_lines:
    if label == '':
        logger.info('  │')
    else:
        logger.info('  │  %-55s %s', label, value)
logger.info('═' * 70)

if 'library' in df.columns:
    lib_counts = df[df['library'] != '']['library'].value_counts()
    if not lib_counts.empty:
        logger.info('LIBRARY BREAKDOWN (import rows):')
        for lib, cnt in lib_counts.items():
            logger.info('  │  %-20s %d', lib, cnt)
        logger.info('─' * 70)

print(f'\n{"=" * 70}')
print('  RUN SUMMARY')
print(f'{"=" * 70}')
for label, value in summary_lines:
    if label == '':
        print()
    else:
        print(f'  {label:<55} {value}')
print(f'{"=" * 70}\n')

df[['name', 'updated_at', 'team_name', 'package_json', 'int_ext',
    'dependency', 'library', 'import_filename', 'import_statement']].head(20)


# ── Write output ──────────────────────────────────────────────────────────────
if not all_rows:
    logger.warning('OUTPUT | no rows to write')
else:
    write_xlsx(df, OUTPUT_XLSX)
    write_json(all_rows, OUTPUT_JSON)
    logger.info('OUTPUT | ✅ XLSX → %s', OUTPUT_XLSX)
    logger.info('OUTPUT | ✅ JSON → %s', OUTPUT_JSON)
    print(f'✅ XLSX → {OUTPUT_XLSX}')
    print(f'✅ JSON → {OUTPUT_JSON}')

logger.info('=' * 70)
logger.info('GitLab Projects Export  v4  — complete (%.1fs)', t_elapsed)
logger.info('=' * 70)
