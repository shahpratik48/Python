# GitLab Projects Export  ── v3
#
# Changes vs v2:
#   • Verbose logging at every stage: project lifecycle, file fetch,
#     import match, skip reasons, timing, and run-wide counters
#   • Per-project elapsed time logged
#   • Per-file fetch status (fetched / empty / error)
#   • Per-import match logged with truncated statement preview
#   • Section banners in logs to make long runs easy to scan
#   • run_stats dict collects counters updated throughout the run;
#     full summary table printed at the end

import getpass
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import gitlab
import pandas as pd

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger('gitlab_export')

# Silence noisy third-party loggers so our messages stay readable
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('gitlab').setLevel(logging.WARNING)

logger.info('=' * 70)
logger.info('GitLab Projects Export  v3  — started')
logger.info('=' * 70)

# ── Configuration ─────────────────────────────────────────────────────────────
GITLAB_URL   = 'https://devcloud.ubs.net'
GROUP_PATH   = 'ubs/gwma'
OUTPUT_XLSX  = 'gitlab_projects_export.xlsx'
OUTPUT_JSON  = 'gitlab_projects_export.json'
SAMPLE_XLSX  = 'sample_gitlab_projects_export.xlsx'
SAMPLE_JSON  = 'sample_gitlab_projects_export.json'
PER_PAGE     = 100
MAX_WORKERS  = 20   # parallel project threads
FILE_WORKERS = 10   # parallel file-fetch threads per project

UPDATED_SINCE     = datetime(2025, 1, 1, tzinfo=timezone.utc)
ENABLE_SAMPLE_MODE = False

SOURCE_EXTENSIONS = ('.jsx', '.tsx', '.js', '.ts')

IMPORT_RE = re.compile(
    r"^[ \t]*import\s+\{[^}]+\}(?:\s*,\s*\{[^}]+\})*\s+from\s+"
    r"['\"](@ubs\.websdk/[^'\"]+|@uwr[^'\"]+)['\"]\s*;?[^\n]*",
    re.MULTILINE | re.IGNORECASE,
)

logger.info('Config | GitLab URL    : %s', GITLAB_URL)
logger.info('Config | Group path    : %s', GROUP_PATH)
logger.info('Config | Updated since : %s', UPDATED_SINCE.date())
logger.info('Config | Max workers   : %d project threads, %d file threads', MAX_WORKERS, FILE_WORKERS)
logger.info('Config | Source exts   : %s', ', '.join(SOURCE_EXTENSIONS))
logger.info('Config | Sample mode   : %s', ENABLE_SAMPLE_MODE)
logger.info('Config | Output XLSX   : %s', OUTPUT_XLSX)
logger.info('Config | Output JSON   : %s', OUTPUT_JSON)

# ── Run-wide counters (thread-safe reads; appended to from futures) ────────────
run_stats: Dict[str, int] = {
    'total_projects_in_group'  : 0,
    'skipped_too_old'          : 0,
    'skipped_no_package_json'  : 0,
    'processed_projects'       : 0,
    'source_files_scanned'     : 0,
    'source_files_empty'       : 0,
    'source_files_fetch_error' : 0,
    'imports_found'            : 0,
    'output_rows'              : 0,
}

# ── Auth ──────────────────────────────────────────────────────────────────────
logger.info('-' * 70)
logger.info('AUTH | Requesting GitLab private token …')
private_token = getpass.getpass('Enter your GitLab private token: ')
client = gitlab.Gitlab(GITLAB_URL, private_token=private_token)
logger.info('AUTH | Client initialised for %s', GITLAB_URL)


# ── Helper utilities ──────────────────────────────────────────────────────────

def updated_since_cutoff(updated_at_str: str) -> bool:
    try:
        dt = datetime.fromisoformat(updated_at_str.replace('Z', '+00:00'))
        return dt >= UPDATED_SINCE
    except Exception:
        logger.debug('DATE | Could not parse updated_at=%r — keeping project', updated_at_str)
        return True


def extract_team_name(web_url: str) -> str:
    try:
        segs = web_url.rstrip('/').split('//', 1)[-1].split('/')[1:]
        if len(segs) >= 2:
            return segs[-2]
    except Exception:
        pass
    return ''


def _raw_file(project: Any, path: str, ref: str) -> Optional[str]:
    """Fetch a single file from the repo and return its text, or None on error."""
    try:
        raw = project.files.raw(file_path=path, ref=ref)
        content = raw.decode('utf-8', errors='replace')
        logger.debug('  FILE | fetched  | %d chars | %s @ %s', len(content), path, ref)
        return content
    except Exception as exc:
        logger.debug('  FILE | error    | %s | %s', path, exc)
        return None


# ── package.json ──────────────────────────────────────────────────────────────

def get_package_json_info(project: Any, ref: str) -> Tuple[str, str, str]:
    """
    Returns (has_pkg_json, int_ext, dependency_string).
      has_pkg_json : 'Yes' | 'No'
      int_ext      : 'internal' | 'external' | '-'
      dependency   : sorted comma-separated dep names, or '-'
    """
    ns = project.path_with_namespace
    logger.debug('  PKG  | fetching package.json | %s', ns)

    content = _raw_file(project, 'package.json', ref)
    if content is None:
        logger.info('  PKG  | NOT FOUND          | %s', ns)
        return 'No', '-', '-'

    logger.debug('  PKG  | found (%d chars)     | %s', len(content), ns)

    try:
        pkg  = json.loads(content)
        deps: Dict[str, str] = pkg.get('dependencies', {})
    except json.JSONDecodeError as exc:
        logger.warning('  PKG  | JSON parse error    | %s | %s', ns, exc)
        return 'Yes', 'external', '-'

    dep_count      = len(deps)
    dependency_str = ', '.join(sorted(deps.keys())) if deps else '-'
    logger.debug('  PKG  | %d dependencies     | %s', dep_count, ns)

    uwr = [d for d in deps if d.startswith('@uwr/')]
    if uwr:
        logger.info('  PKG  | INTERNAL (%d @uwr deps) | %s | uwr deps: %s',
                    len(uwr), ns, ', '.join(uwr))
        return 'Yes', 'internal', dependency_str

    logger.info('  PKG  | external (%d deps)  | %s', dep_count, ns)
    return 'Yes', 'external', dependency_str


# ── File tree scan ────────────────────────────────────────────────────────────

def scan_imports(project: Any, ref: str) -> List[Tuple[str, str, str]]:
    """
    Walk the full repo tree, read every .js/.ts/.jsx/.tsx file, and return
    one (filename, blob_url, import_statement) tuple per matched import.
    """
    ns = project.path_with_namespace
    logger.info('  SCAN | starting tree walk   | %s  [ref=%s]', ns, ref)
    t0 = time.perf_counter()

    try:
        all_items = project.repository_tree(ref=ref, recursive=True, all=True, per_page=100)
    except Exception as exc:
        logger.error('  SCAN | tree fetch FAILED  | %s | %s', ns, exc)
        return []

    total_items = len(all_items)
    src = [i for i in all_items
           if i.get('type') == 'blob' and i['path'].endswith(SOURCE_EXTENSIONS)]

    logger.info('  SCAN | tree items: %d total, %d source files | %s',
                total_items, len(src), ns)

    if not src:
        logger.info('  SCAN | no source files found | %s', ns)
        return []

    # Log each source file path at DEBUG
    for item in src:
        logger.debug('  SCAN | source file: %s', item['path'])

    results: List[Tuple[str, str, str]] = []
    local_fetched = local_empty = local_error = local_match = 0

    def scan_one(item) -> Optional[List[Tuple[str, str, str]]]:
        nonlocal local_fetched, local_empty, local_error, local_match
        path    = item['path']
        content = _raw_file(project, path, ref)

        if content is None:
            local_error += 1
            logger.debug('  FILE | fetch error        | %s', path)
            return None
        if not content.strip():
            local_empty += 1
            logger.debug('  FILE | empty              | %s', path)
            return None

        local_fetched += 1
        matches = []
        for m in IMPORT_RE.finditer(content):
            stmt = m.group(0).strip()
            tup  = (
                path.split('/')[-1],
                f"{project.web_url}/-/blob/{ref}/{path}",
                stmt,
            )
            matches.append(tup)
            local_match += 1
            # Truncate long statements for readability in the log
            preview = stmt[:120] + ('…' if len(stmt) > 120 else '')
            logger.info('  MATCH| %s | %s', path, preview)

        if matches:
            logger.debug('  FILE | %d import(s) found  | %s', len(matches), path)
        else:
            logger.debug('  FILE | no imports          | %s', path)

        return matches or None

    with ThreadPoolExecutor(max_workers=FILE_WORKERS) as pool:
        for found in pool.map(scan_one, src):
            if found:
                results.extend(found)

    elapsed = time.perf_counter() - t0
    logger.info(
        '  SCAN | done in %.1fs | fetched=%d  empty=%d  error=%d  matches=%d | %s',
        elapsed, local_fetched, local_empty, local_error, len(results), ns,
    )

    # Update global stats (GIL makes += safe here for ints)
    run_stats['source_files_scanned']     += local_fetched
    run_stats['source_files_empty']       += local_empty
    run_stats['source_files_fetch_error'] += local_error
    run_stats['imports_found']            += len(results)

    return results


def derive_library(import_statement: str) -> str:
    if not import_statement:
        return ''
    s = import_statement.lower()
    found = []
    if '@uwr/' in s:
        found.append('uwr')
    if '@ubs.websdk/' in s:
        found.append('websdk')
    return ', '.join(found)


# ── Core per-project processor ────────────────────────────────────────────────

def process_one(proj_ref: Any, idx: int, total: int) -> Tuple[Optional[List[Dict[str, Any]]], bool]:
    """
    Returns (rows_or_None, was_skipped_no_pkg).

    Processing stages logged explicitly:
      [1] Date filter     — skip if updated_at < UPDATED_SINCE
      [2] Full project    — fetch via API
      [3] package.json    — fetch; skip scan if absent
      [4] File scan       — walk tree and match imports
      [5] Row building    — one row per import (or one summary row)
    """
    t_proj = time.perf_counter()
    ns     = getattr(proj_ref, 'path_with_namespace', str(proj_ref.id))

    logger.info('─' * 70)
    logger.info('PROJ [%d/%d] | START | %s', idx, total, ns)

    # ── [1] Date filter ───────────────────────────────────────────────────────
    updated_at_str = getattr(proj_ref, 'updated_at', None) or ''
    logger.debug('PROJ [%d/%d] | updated_at=%s', idx, total, updated_at_str[:10] or 'n/a')

    if updated_at_str and not updated_since_cutoff(updated_at_str):
        logger.info('PROJ [%d/%d] | SKIP — too old (updated=%s, cutoff=%s) | %s',
                    idx, total, updated_at_str[:10], UPDATED_SINCE.date(), ns)
        run_stats['skipped_too_old'] += 1
        return None, False

    logger.debug('PROJ [%d/%d] | passes date filter', idx, total)

    # ── [2] Fetch full project object ─────────────────────────────────────────
    logger.debug('PROJ [%d/%d] | fetching full project object …', idx, total)
    project   = client.projects.get(proj_ref.id)
    namespace = project.namespace or {}
    web_url   = project.web_url
    ref       = project.default_branch or 'main'
    team_name = extract_team_name(web_url)

    logger.info('PROJ [%d/%d] | fetched | id=%s  branch=%s  team=%s  visibility=%s  archived=%s',
                idx, total,
                project.id, ref, team_name or '(none)',
                project.visibility, project.archived)
    logger.debug('PROJ [%d/%d] | url=%s', idx, total, web_url)
    logger.debug('PROJ [%d/%d] | description=%s',
                 idx, total, (project.description or '')[:80] or '(none)')
    logger.debug('PROJ [%d/%d] | created=%s  last_activity=%s',
                 idx, total,
                 (project.created_at or '')[:10],
                 (project.last_activity_at or '')[:10])
    logger.debug('PROJ [%d/%d] | forks=%s  stars=%s  open_issues=%s',
                 idx, total,
                 getattr(project, 'forks_count', '?'),
                 getattr(project, 'star_count', '?'),
                 getattr(project, 'open_issues_count', '?'))

    # ── [3] package.json ──────────────────────────────────────────────────────
    pkg_json, int_ext, dependency = get_package_json_info(project, ref)

    if pkg_json == 'No':
        logger.info('PROJ [%d/%d] | SKIP — no package.json | %s', idx, total, ns)
        run_stats['skipped_no_package_json'] += 1
        return None, True

    logger.info('PROJ [%d/%d] | package.json present | int_ext=%s | dep_count=%d | %s',
                idx, total, int_ext,
                len(dependency.split(',')) if dependency != '-' else 0,
                ns)

    # ── [4] File scan ─────────────────────────────────────────────────────────
    imports = scan_imports(project, ref)

    # ── [5] Build output rows ─────────────────────────────────────────────────
    base = {
        'project_id':          project.id,
        'name':                project.name,
        'path':                project.path,
        'path_with_namespace': project.path_with_namespace,
        'group_path':          namespace.get('full_path'),
        'web_url':             web_url,
        'description':         project.description,
        'visibility':          project.visibility,
        'archived':            project.archived,
        'created_at':          project.created_at,
        'last_activity_at':    project.last_activity_at,
        'updated_at':          project.updated_at,
        'default_branch':      ref,
        'forks_count':         getattr(project, 'forks_count', None),
        'star_count':          getattr(project, 'star_count', None),
        'open_issues_count':   getattr(project, 'open_issues_count', None),
        'team_name':           team_name,
        'package_json':        pkg_json,
        'int_ext':             int_ext,
        'dependency':          dependency,
        'component':           '',
    }

    if not imports:
        logger.info('PROJ [%d/%d] | 0 matching imports → 1 summary row | %s', idx, total, ns)
        row = {**base,
               'library':          '',
               'import_filename':  '',
               'import_file_url':  '',
               'import_statement': ''}
        rows = [row]
    else:
        rows = []
        for filename, file_url, stmt in imports:
            lib = derive_library(stmt)
            row = {**base,
                   'library':          lib,
                   'import_filename':  filename,
                   'import_file_url':  file_url,
                   'import_statement': stmt}
            rows.append(row)
        logger.info('PROJ [%d/%d] | %d import row(s) built | %s', idx, total, len(rows), ns)

    elapsed = time.perf_counter() - t_proj
    run_stats['processed_projects'] += 1
    run_stats['output_rows']        += len(rows)

    logger.info('PROJ [%d/%d] | DONE in %.1fs | %d row(s) | %s',
                idx, total, elapsed, len(rows), ns)

    return rows, False


# ── DataFrame & output helpers ────────────────────────────────────────────────

def build_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    logger.debug('DF | building dataframe from %d rows …', len(rows))
    df = pd.DataFrame(rows)
    for col in ('created_at', 'last_activity_at', 'updated_at'):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce', utc=True).dt.tz_localize(None)
    df.sort_values(['group_path', 'name', 'import_statement'], inplace=True, ignore_index=True)
    logger.debug('DF | sorted by group_path / name / import_statement')
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
    logger.info('XLSX | done in %.1fs → %s', time.perf_counter() - t0, path)


def write_json(rows: List[Dict[str, Any]], path: str) -> None:
    logger.info('JSON | writing %d rows → %s', len(rows), path)
    t0 = time.perf_counter()
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2, default=str, separators=(',', ': '))
    logger.info('JSON | done in %.1fs → %s', time.perf_counter() - t0, path)


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
            logger.info('SAMPLE | searching for project: %r', project_name)
            try:
                group     = client.groups.get(GROUP_PATH)
                proj_refs = group.projects.list(
                    search=project_name, include_subgroups=True, all=True, per_page=50
                )
                logger.info('SAMPLE | search returned %d candidate(s)', len(proj_refs))
                match = next(
                    (p for p in proj_refs if p.name.lower() == project_name.lower()),
                    proj_refs[0] if proj_refs else None,
                )
            except Exception as exc:
                logger.error('SAMPLE | search failed: %s', exc)
                match = None

            if not match:
                print(f'⚠️  No project found matching "{project_name}".')
                logger.warning('SAMPLE | no match for %r', project_name)
            else:
                logger.info('SAMPLE | matched: %s', match.path_with_namespace)
                rows_out, skipped_no_pkg = process_one(match, 1, 1)
                if rows_out:
                    sample_rows.extend(rows_out)
                    print(f'\n✅  {rows_out[0]["path_with_namespace"]}')
                    print(f'    package_json : {rows_out[0]["package_json"]}')
                    print(f'    int_ext      : {rows_out[0]["int_ext"]}')
                    dep_preview = rows_out[0]['dependency']
                    print(f'    dependency   : {dep_preview[:80]}{"…" if len(dep_preview) > 80 else ""}')
                    n_imp = sum(1 for r in rows_out if r['import_statement'])
                    print(f'    import rows  : {n_imp}')
                    df_s = build_dataframe(sample_rows)
                    write_xlsx(df_s, SAMPLE_XLSX)
                    write_json(sample_rows, SAMPLE_JSON)
                    print(f'    Sample files updated: {SAMPLE_XLSX} | {SAMPLE_JSON}')
                    logger.info('SAMPLE | sample files updated')
                else:
                    reason = 'no package.json' if skipped_no_pkg else 'old updated_at or other filter'
                    print(f'⚠️  Project filtered out ({reason}): {match.path_with_namespace}')
                    logger.warning('SAMPLE | project filtered (%s): %s', reason, match.path_with_namespace)

        again = input('\nRun another sample? (yes/no): ').strip().lower()
        if again not in ('yes', 'y'):
            break

    logger.info('SAMPLE | done — %d row(s) collected', len(sample_rows))
    print(f'\nSample mode done — {len(sample_rows)} row(s) in sample files.')
    cont = input('Proceed with full export? (yes/no): ').strip().lower()
    if cont not in ('yes', 'y'):
        logger.info('SAMPLE | user chose to stop — exiting')
        print('Stopping here.')
        raise SystemExit(0)
else:
    logger.info('Sample mode disabled — proceeding directly to full export.')


# ── Full export ───────────────────────────────────────────────────────────────

logger.info('=' * 70)
logger.info("EXPORT | fetching project list for group '%s' …", GROUP_PATH)
t_export_start = time.perf_counter()

group     = client.groups.get(GROUP_PATH)
proj_refs = group.projects.list(include_subgroups=True, all=True, per_page=PER_PAGE)
total     = len(proj_refs)
run_stats['total_projects_in_group'] = total

logger.info('EXPORT | %d projects found in group', total)
logger.info('EXPORT | launching %d worker threads …', MAX_WORKERS)

all_rows: List[Dict[str, Any]] = []
done = 0

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
    future_map = {
        pool.submit(process_one, ref, idx, total): ref
        for idx, ref in enumerate(proj_refs, start=1)
    }
    for future in as_completed(future_map):
        done += 1
        ref = future_map[future]
        try:
            rows_out, skipped_no_pkg = future.result()
            if rows_out:
                all_rows.extend(rows_out)
                logger.debug('PROGRESS | %d/%d complete | +%d rows | %s',
                             done, total, len(rows_out), ref.path_with_namespace)
            elif skipped_no_pkg:
                logger.debug('PROGRESS | %d/%d skipped (no pkg.json) | %s',
                             done, total, ref.path_with_namespace)
            else:
                logger.debug('PROGRESS | %d/%d skipped (date filter) | %s',
                             done, total, ref.path_with_namespace)
        except Exception as exc:
            logger.error('PROGRESS | %d/%d EXCEPTION | %s | %s',
                         done, total, ref.path_with_namespace, exc, exc_info=True)

t_export_elapsed = time.perf_counter() - t_export_start
included_projects = len({r['project_id'] for r in all_rows})

logger.info('=' * 70)
logger.info('EXPORT | finished in %.1fs', t_export_elapsed)


# ── Final summary ─────────────────────────────────────────────────────────────

df = build_dataframe(all_rows)

summary_lines = [
    ('Total projects in group',                          run_stats['total_projects_in_group']),
    (f'Skipped — not updated since {UPDATED_SINCE.date()}', run_stats['skipped_too_old']),
    ('Skipped — no package.json (but recently updated)', run_stats['skipped_no_package_json']),
    ('Projects processed (reached file scan stage)',     run_stats['processed_projects']),
    ('',                                                 ''),
    ('Source files fetched successfully',                run_stats['source_files_scanned']),
    ('Source files empty (skipped)',                     run_stats['source_files_empty']),
    ('Source files fetch error',                         run_stats['source_files_fetch_error']),
    ('Import statements matched',                        run_stats['imports_found']),
    ('',                                                 ''),
    ('Distinct projects in output',                      included_projects),
    ('Total output rows',                                run_stats['output_rows']),
    ('Rows with import statement',                       int((df['import_statement'] != '').sum())),
    ('Elapsed time (seconds)',                           f'{t_export_elapsed:.1f}'),
]

logger.info('─' * 70)
logger.info('SUMMARY')
for label, value in summary_lines:
    if label == '':
        logger.info('  │')
    else:
        logger.info('  │  %-52s %s', label, value)
logger.info('─' * 70)

if 'library' in df.columns:
    lib_counts = df[df['library'] != '']['library'].value_counts()
    if not lib_counts.empty:
        logger.info('LIBRARY BREAKDOWN (by import row):')
        for lib, count in lib_counts.items():
            logger.info('  │  %-20s %d', lib, count)
        logger.info('─' * 70)

# Pretty-print the same summary to stdout for notebook / interactive use
print(f'\n{"=" * 70}')
print('  RUN SUMMARY')
print(f'{"=" * 70}')
for label, value in summary_lines:
    if label == '':
        print()
    else:
        print(f'  {label:<52} {value}')
print(f'{"=" * 70}\n')

df[['name', 'updated_at', 'team_name', 'package_json', 'int_ext',
    'dependency', 'library', 'import_filename', 'import_statement']].head(10)


# ── Write output ──────────────────────────────────────────────────────────────

if not all_rows:
    logger.warning('OUTPUT | no rows to write — output files not created')
else:
    write_xlsx(df, OUTPUT_XLSX)
    write_json(all_rows, OUTPUT_JSON)
    logger.info('OUTPUT | ✅ XLSX → %s', OUTPUT_XLSX)
    logger.info('OUTPUT | ✅ JSON → %s', OUTPUT_JSON)
    print(f'✅ XLSX → {OUTPUT_XLSX}')
    print(f'✅ JSON → {OUTPUT_JSON}')

logger.info('=' * 70)
logger.info('GitLab Projects Export  v3  — complete')
logger.info('=' * 70)
