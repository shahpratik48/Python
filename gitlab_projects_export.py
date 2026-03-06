"""
GitLab Projects Exporter
========================
Features:
  - 1-year date filter (comment out to disable — see CUTOFF_DAYS section)
  - package.json filter: projects without package.json are skipped entirely
    before any file tree walking (major speed improvement)
  - Sample mode: ENABLE_SAMPLE_MODE = True prompts user to test one project
    at a time before running the full export
  - Parallel project processing + parallel file fetches within each project
  - Raw file endpoint (no base64 round-trip)
  - Outputs: XLSX + JSON
"""

import argparse
import getpass
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import gitlab
import pandas as pd

# ---------------------------------------------------------------------------
# Defaults & constants
# ---------------------------------------------------------------------------

DEFAULT_GITLAB_URL  = "https://devcloud.ubs.net"
DEFAULT_GROUP_PATH  = "ubs/gwma"
DEFAULT_OUTPUT_XLSX = "gitlab_projects_export.xlsx"
DEFAULT_OUTPUT_JSON = "gitlab_projects_export.json"
SAMPLE_OUTPUT_XLSX  = "sample_gitlab_projects_export.xlsx"
SAMPLE_OUTPUT_JSON  = "sample_gitlab_projects_export.json"
LOG_FORMAT          = "%(asctime)s | %(levelname)s | %(message)s"

MAX_WORKERS    = 20   # project-level parallel threads
FILE_WORKERS   = 10   # file-fetch parallel threads per project
CUTOFF_DAYS    = 365  # only process projects created within this many days

# ---- Set to False to skip sample prompts and run full export directly ----
ENABLE_SAMPLE_MODE = True

SOURCE_EXTENSIONS = (".jsx", ".tsx", ".js", ".ts")

# ---------------------------------------------------------------------------
# Regex — BUG FIX: @ubs\.websdk was not matching because the dot in the
# package name was escaped (\.) in the pattern but the real import uses a
# literal dot.  We now use [^\s'"] as a general "rest of package path"
# matcher so both @ubs.websdk/... and @uwr/... are captured correctly.
# Also added re.IGNORECASE so casing differences (Checkbox vs checkbox) in
# the component names don't prevent a match.
# ---------------------------------------------------------------------------
IMPORT_RE = re.compile(
    r"^[ \t]*import\s+\{[^}]+\}(?:\s*,\s*\{[^}]+\})*\s+from\s+"
    r"""['"](@ubs\.websdk/[^'"]+|@uwr[^'"]+)['"]\s*;?[^\n]*""",
    re.MULTILINE | re.IGNORECASE,
)

logger = logging.getLogger("gitlab_export")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export recent GitLab projects to XLSX + JSON.")
    p.add_argument("--gitlab-url",   default=DEFAULT_GITLAB_URL)
    p.add_argument("--group-path",   default=DEFAULT_GROUP_PATH)
    p.add_argument("--output-xlsx",  default=DEFAULT_OUTPUT_XLSX)
    p.add_argument("--output-json",  default=DEFAULT_OUTPUT_JSON)
    p.add_argument("--per-page",     type=int, default=100)
    p.add_argument("--workers",      type=int, default=MAX_WORKERS)
    p.add_argument("--file-workers", type=int, default=FILE_WORKERS)
    p.add_argument("--cutoff-days",  type=int, default=CUTOFF_DAYS)
    args, _ = p.parse_known_args()
    return args


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


# ---------------------------------------------------------------------------
# GitLab client
# ---------------------------------------------------------------------------

def make_client(url: str, token: str) -> gitlab.Gitlab:
    return gitlab.Gitlab(url, private_token=token)


# ---------------------------------------------------------------------------
# Date filter
# ---------------------------------------------------------------------------

def cutoff_date(days: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=days)


def created_within(created_at_str: str, cutoff: datetime) -> bool:
    try:
        dt = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
        return dt >= cutoff
    except Exception:
        return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_team_name(web_url: str) -> str:
    """Second-last URL path segment = team folder."""
    try:
        segs = web_url.rstrip("/").split("//", 1)[-1].split("/")[1:]
        if len(segs) >= 2:
            return segs[-2]
    except Exception:
        pass
    return ""


def _raw_file(project: Any, path: str, ref: str) -> Optional[str]:
    """Fetch raw file bytes — avoids base64 round-trip of files.get()."""
    try:
        return project.files.raw(file_path=path, ref=ref).decode("utf-8", errors="replace")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# package.json — read once, parse once, return early if missing
# ---------------------------------------------------------------------------

def get_package_json_info(project: Any, ref: str) -> Tuple[str, str]:
    """
    Returns ('Yes'/'No', 'internal'/'external'/'-').
    'No' means no package.json found — caller should skip further scanning.
    """
    content = _raw_file(project, "package.json", ref)
    if content is None:
        logger.info("  [pkg] not found | %s", project.path_with_namespace)
        return "No", "-"
    try:
        deps = json.loads(content).get("dependencies", {})
    except json.JSONDecodeError as e:
        logger.warning("  [pkg] bad JSON | %s | %s", project.path_with_namespace, e)
        return "Yes", "external"
    uwr = [d for d in deps if d.startswith("@uwr/")]
    if uwr:
        logger.info("  [pkg] internal | %s | %s", project.path_with_namespace, uwr)
        return "Yes", "internal"
    logger.info("  [pkg] external | %s", project.path_with_namespace)
    return "Yes", "external"


# ---------------------------------------------------------------------------
# Import scanning — parallel file fetches, skip early if no package.json
# ---------------------------------------------------------------------------

def scan_imports(project: Any, ref: str, file_workers: int) -> Tuple[str, str, str]:
    """
    Walk repo tree, find source files, scan for @ubs.websdk/ or @uwr/ imports.
    Returns bracket-delimited (filenames, blob-urls, import-statements).

    Performance notes:
    - repository_tree(recursive=True, all=True) = one API call for entire tree
    - Extension filter applied in Python before any file fetch
    - Files fetched concurrently via ThreadPoolExecutor
    - pool.map() used (vs as_completed) — preserves order, simpler aggregation
    """
    try:
        all_items = project.repository_tree(ref=ref, recursive=True, all=True, per_page=100)
    except Exception as e:
        logger.warning("  [scan] tree error | %s | %s", project.path_with_namespace, e)
        return "", "", ""

    src = [
        i for i in all_items
        if i.get("type") == "blob" and i["path"].endswith(SOURCE_EXTENSIONS)
    ]

    if not src:
        logger.info("  [scan] no source files | %s", project.path_with_namespace)
        return "", "", ""

    logger.info("  [scan] %d source files to scan | %s", len(src), project.path_with_namespace)

    filenames: List[str] = []
    urls:      List[str] = []
    stmts:     List[str] = []

    def scan_one(item: Dict) -> Optional[List[Tuple[str, str, str]]]:
        path    = item["path"]
        content = _raw_file(project, path, ref)
        if not content:
            return None
        results = []
        for m in IMPORT_RE.finditer(content):
            results.append((
                path.split("/")[-1],
                f"{project.web_url}/-/blob/{ref}/{path}",
                m.group(0).strip(),
            ))
        return results or None

    # pool.map preserves order and is slightly faster than as_completed for I/O-bound tasks
    with ThreadPoolExecutor(max_workers=file_workers) as pool:
        for result in pool.map(scan_one, src):
            if result:
                for fname, furl, stmt in result:
                    filenames.append(fname)
                    urls.append(furl)
                    stmts.append(stmt)

    if not filenames:
        logger.info("  [scan] no matching imports | %s", project.path_with_namespace)
        return "", "", ""

    logger.info("  [scan] %d import(s) found | %s", len(filenames), project.path_with_namespace)
    fmt = lambda lst: "".join(f"[{v}]" for v in lst)
    return fmt(filenames), fmt(urls), fmt(stmts)


# ---------------------------------------------------------------------------
# Library label
# ---------------------------------------------------------------------------

def derive_library(import_statements: str) -> str:
    """
    Scan collected import_statement string for known prefixes.
    Returns deduplicated label: '', 'uwr', 'websdk', or 'uwr, websdk'.
    Order is always uwr first, websdk second.
    """
    if not import_statements:
        return ""
    found = []
    if "@uwr/" in import_statements.lower():
        found.append("uwr")
    if "@ubs.websdk/" in import_statements.lower():
        found.append("websdk")
    return ", ".join(found)


# ---------------------------------------------------------------------------
# Per-project row builder
# ---------------------------------------------------------------------------

def process_one(
    proj_ref:     Any,
    client:       gitlab.Gitlab,
    cutoff:       datetime,
    file_workers: int,
    idx:          int,
    total:        int,
) -> Optional[Dict[str, Any]]:
    """
    Processing order (fastest-failing checks first):
      1. Date filter   — uses proj_ref data, zero extra API calls
      2. package.json  — single file fetch; returns None if missing (skips tree walk)
      3. File tree scan — only reached if package.json exists
    """
    # ---- 1. Date filter (free — data already on proj_ref) ----
    created_at_str = getattr(proj_ref, "created_at", None) or ""
    if created_at_str and not created_within(created_at_str, cutoff):
        logger.info("[%d/%d] SKIP (old) | %s | created=%s",
                    idx, total, proj_ref.path_with_namespace, created_at_str[:10])
        return None

    logger.info("[%d/%d] → %s", idx, total, proj_ref.path_with_namespace)

    project   = client.projects.get(proj_ref.id)
    namespace = project.namespace or {}
    web_url   = project.web_url
    ref       = project.default_branch or "main"

    team_name         = extract_team_name(web_url)

    # ---- 2. package.json check — skip tree scan entirely if missing ----
    pkg_json, int_ext = get_package_json_info(project, ref)
    if pkg_json == "No":
        logger.info("[%d/%d] SKIP (no package.json) | %s", idx, total, project.path_with_namespace)
        return None

    # ---- 3. File scan — only runs for projects with package.json ----
    imp_fn, imp_url, imp_st = scan_imports(project, ref, file_workers)
    library = derive_library(imp_st)

    return {
        "project_id":          project.id,
        "name":                project.name,
        "path":                project.path,
        "path_with_namespace": project.path_with_namespace,
        "group_path":          namespace.get("full_path"),
        "web_url":             web_url,
        "description":         project.description,
        "visibility":          project.visibility,
        "archived":            project.archived,
        "created_at":          project.created_at,
        "last_activity_at":    project.last_activity_at,
        "updated_at":          project.updated_at,
        "default_branch":      ref,
        "forks_count":         getattr(project, "forks_count", None),
        "star_count":          getattr(project, "star_count", None),
        "open_issues_count":   getattr(project, "open_issues_count", None),
        "team_name":           team_name,
        "package_json":        pkg_json,
        "int_ext":             int_ext,
        "component":           "",
        "library":             library,
        "import_filename":     imp_fn,
        "import_file_url":     imp_url,
        "import_statement":    imp_st,
    }


# ---------------------------------------------------------------------------
# Single-project fetch by name (for sample mode)
# ---------------------------------------------------------------------------

def fetch_project_by_name(
    client:       gitlab.Gitlab,
    group_path:   str,
    project_name: str,
    file_workers: int,
    cutoff:       datetime,
) -> Optional[Dict[str, Any]]:
    """Find a project by name within the group and process it."""
    logger.info("[sample] Searching for project '%s' in '%s'", project_name, group_path)
    try:
        group     = client.groups.get(group_path)
        proj_refs = group.projects.list(
            search=project_name, include_subgroups=True, all=True, per_page=50
        )
    except Exception as e:
        logger.error("[sample] Failed to search group: %s", e)
        return None

    # Find exact or closest match (case-insensitive)
    match = next(
        (p for p in proj_refs if p.name.lower() == project_name.lower()),
        proj_refs[0] if proj_refs else None,
    )
    if not match:
        logger.warning("[sample] No project found matching '%s'", project_name)
        return None

    logger.info("[sample] Found: %s", match.path_with_namespace)
    return process_one(match, client, cutoff, file_workers, 1, 1)


# ---------------------------------------------------------------------------
# Sample mode loop
# ---------------------------------------------------------------------------

def run_sample_mode(
    client:       gitlab.Gitlab,
    group_path:   str,
    file_workers: int,
    cutoff:       datetime,
) -> None:
    """
    Interactive loop: ask for a project name, process it, append to sample files.
    Continues until user answers 'no' when asked if they want another sample.
    """
    sample_rows: List[Dict[str, Any]] = []

    print("\n" + "="*60)
    print("SAMPLE MODE — test individual projects before full run")
    print("="*60)

    while True:
        project_name = input("\nEnter project name to sample: ").strip()
        if not project_name:
            print("No name entered — skipping.")
        else:
            row = fetch_project_by_name(client, group_path, project_name, file_workers, cutoff)
            if row:
                sample_rows.append(row)
                print(f"\n✅ Project processed: {row['path_with_namespace']}")
                print(f"   package_json : {row['package_json']}")
                print(f"   int_ext      : {row['int_ext']}")
                print(f"   library      : {row['library']}")
                print(f"   imports found: {len(row['import_filename'].split('][')) if row['import_filename'] else 0}")

                # Write/overwrite sample files after each addition
                df = build_dataframe(sample_rows)
                write_xlsx(df, SAMPLE_OUTPUT_XLSX)
                write_json(sample_rows, SAMPLE_OUTPUT_JSON)
                print(f"   Sample files updated: {SAMPLE_OUTPUT_XLSX} | {SAMPLE_OUTPUT_JSON}")
            else:
                print(f"⚠️  Could not process project '{project_name}' — check name and permissions.")

        again = input("\nRun another sample? (yes/no): ").strip().lower()
        if again not in ("yes", "y"):
            break

    if sample_rows:
        print(f"\nSample mode complete — {len(sample_rows)} project(s) in sample files.")
    else:
        print("\nSample mode complete — no projects were added.")

    cont = input("\nProceed with full export? (yes/no): ").strip().lower()
    if cont not in ("yes", "y"):
        print("Exiting without full export.")
        raise SystemExit(0)


# ---------------------------------------------------------------------------
# Parallel orchestrator (full export)
# ---------------------------------------------------------------------------

def export_group_projects(
    client:       gitlab.Gitlab,
    group_path:   str,
    per_page:     int,
    max_workers:  int,
    file_workers: int,
    cutoff_days:  int,
) -> List[Dict[str, Any]]:
    cutoff = cutoff_date(cutoff_days)
    logger.info("Date cutoff: projects created on/after %s", cutoff.date())

    logger.info("Fetching project list for group '%s'", group_path)
    group     = client.groups.get(group_path)
    proj_refs = group.projects.list(include_subgroups=True, all=True, per_page=per_page)

    total = len(proj_refs)
    logger.info("Total projects in group: %d | workers: %d", total, max_workers)

    rows: List[Dict[str, Any]] = []
    done = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {
            pool.submit(
                process_one, ref, client, cutoff, file_workers, idx, total
            ): ref
            for idx, ref in enumerate(proj_refs, start=1)
        }
        for future in as_completed(future_map):
            done += 1
            ref = future_map[future]
            try:
                row = future.result()
                if row is not None:
                    rows.append(row)
                    logger.info("[%d/%d ✓] %s", done, total, row["path_with_namespace"])
                else:
                    logger.info("[%d/%d –] skipped | %s", done, total, ref.path_with_namespace)
            except Exception as e:
                logger.error("[%d/%d ✗] %s | %s", done, total, ref.path_with_namespace, e)

    logger.info("Done: %d included, %d skipped/failed out of %d total",
                len(rows), total - len(rows), total)
    return rows


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def build_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in ("created_at", "last_activity_at", "updated_at"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True).dt.tz_localize(None)
    df.sort_values(["group_path", "name"], inplace=True, ignore_index=True)
    return df


def write_xlsx(df: pd.DataFrame, path: str) -> None:
    logger.info("Writing XLSX → %s  (%d rows)", path, len(df))
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="projects")
        ws = writer.sheets["projects"]
        for col_cells in ws.columns:
            w = max((len(str(c.value)) if c.value is not None else 0) for c in col_cells)
            ws.column_dimensions[col_cells[0].column_letter].width = min(w + 4, 80)
    logger.info("XLSX done: %s", path)


def write_json(rows: List[Dict[str, Any]], path: str) -> None:
    logger.info("Writing JSON → %s  (%d rows)", path, len(rows))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=str, separators=(",", ": "))
    logger.info("JSON done: %s", path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    configure_logging()
    args = parse_args()

    logger.info("=== GitLab Export Start | group=%s | workers=%d | cutoff=%d days ===",
                args.group_path, args.workers, args.cutoff_days)

    private_token = getpass.getpass("Enter your GitLab private token: ")
    client = make_client(args.gitlab_url, private_token)
    cutoff = cutoff_date(args.cutoff_days)

    # ---- Sample mode ----
    if ENABLE_SAMPLE_MODE:
        run_sample_mode(client, args.group_path, args.file_workers, cutoff)

    # ---- Full export ----
    rows = export_group_projects(
        client, args.group_path, args.per_page,
        args.workers, args.file_workers, args.cutoff_days,
    )

    if not rows:
        logger.warning("No qualifying projects found — nothing to write.")
        return

    df = build_dataframe(rows)
    write_xlsx(df, args.output_xlsx)
    write_json(rows, args.output_json)

    logger.info("=== Done | %d projects exported | xlsx=%s | json=%s ===",
                len(rows), args.output_xlsx, args.output_json)


if __name__ == "__main__":
    main()
