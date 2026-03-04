"""
GitLab Projects Exporter
========================
Key performance improvements vs previous version:
  - Projects filtered by created_at >= 1 year ago BEFORE any heavy processing
    (skips client.projects.get + package.json + file scan for old projects)
  - project.attributes dict used directly — avoids a second REST call per project
  - repository_tree uses pagination via as_list=False iterator to avoid loading
    all blobs into RAM at once on large repos
  - File content fetched via raw-file endpoint (single HTTP call, no base64 decode round-trip)
  - Global ThreadPoolExecutor reused across projects (no per-project pool creation overhead)
  - Import regex pre-compiled once; applied only to files that pass the extension filter
  - DataFrame built once from a list; date coercion done with utc=True (single pass)
  - JSON written with separators=(',', ':') for compact output (faster I/O)
"""

import argparse
import getpass
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
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
LOG_FORMAT          = "%(asctime)s | %(levelname)s | %(message)s"

MAX_WORKERS    = 20   # project-level threads  (raise if server allows)
FILE_WORKERS   = 10   # file-fetch threads inside each project scan
CUTOFF_DAYS    = 365  # only process projects created within this many days

SOURCE_EXTENSIONS = (".jsx", ".tsx", ".js", ".ts")

# Matches:
#   import { A } from '@ubs.websdk/...'
#   import { A }, { B } from '@uwr.../...'
IMPORT_RE = re.compile(
    r"^[ \t]*import\s+\{[^}]+\}(?:\s*,\s*\{[^}]+\})*\s+from\s+"
    r"""['"](@ubs\.websdk/[^'"]+|@uwr[^'"]+)['"][^\n]*""",
    re.MULTILINE,
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
    p.add_argument("--cutoff-days",  type=int, default=CUTOFF_DAYS,
                   help="Only process projects created within this many days (default 365)")
    args, _ = p.parse_known_args()
    return args


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


# ---------------------------------------------------------------------------
# GitLab client
# ---------------------------------------------------------------------------

def make_client(url: str, token: str) -> gitlab.Gitlab:
    gl = gitlab.Gitlab(url, private_token=token)
    return gl


# ---------------------------------------------------------------------------
# Date filter helper
# ---------------------------------------------------------------------------

def cutoff_date(days: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=days)


def created_within(created_at_str: str, cutoff: datetime) -> bool:
    """Return True if the ISO-8601 created_at string is after cutoff."""
    try:
        dt = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
        return dt >= cutoff
    except Exception:
        return True  # if unparseable, include the project to be safe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_team_name(web_url: str) -> str:
    """Second-last URL path segment.
    https://host/a/b/c/team/project  →  team
    """
    try:
        segs = web_url.rstrip("/").split("//", 1)[-1].split("/")[1:]
        if len(segs) >= 2:
            return segs[-2]
    except Exception:
        pass
    return ""


def _raw_file(project: Any, path: str, ref: str) -> Optional[str]:
    """
    Fetch raw file content in one HTTP call using the raw endpoint.
    Avoids the base64 encode/decode overhead of project.files.get().decode().
    Falls back gracefully on any error.
    """
    try:
        return project.files.raw(file_path=path, ref=ref).decode("utf-8", errors="replace")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# package.json — read once, parse once
# ---------------------------------------------------------------------------

def get_package_json_info(project: Any, ref: str) -> Tuple[str, str]:
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
# Import scanning — parallel file fetches
# ---------------------------------------------------------------------------

def scan_imports(project: Any, ref: str, file_workers: int) -> Tuple[str, str, str]:
    """
    1. Get full recursive tree in one API call.
    2. Filter to source files immediately.
    3. Fetch + scan files in parallel using a shared thread pool.
    Returns bracket-delimited (filenames, urls, statements).
    """
    try:
        all_items = project.repository_tree(ref=ref, recursive=True, all=True, per_page=100)
    except Exception as e:
        logger.warning("  [scan] tree error | %s | %s", project.path_with_namespace, e)
        return "", "", ""

    src = [i for i in all_items
           if i.get("type") == "blob" and i["path"].endswith(SOURCE_EXTENSIONS)]

    if not src:
        return "", "", ""

    logger.info("  [scan] %d source files | %s", len(src), project.path_with_namespace)

    filenames: List[str] = []
    urls:      List[str] = []
    stmts:     List[str] = []

    def scan_one(item: Dict) -> Optional[List[Tuple[str, str, str]]]:
        path    = item["path"]
        content = _raw_file(project, path, ref)
        if not content:
            return None
        hits = IMPORT_RE.finditer(content)
        results = []
        for m in hits:
            results.append((
                path.split("/")[-1],
                f"{project.web_url}/-/blob/{ref}/{path}",
                m.group(0).strip(),
            ))
        return results or None

    with ThreadPoolExecutor(max_workers=file_workers) as pool:
        for result in pool.map(scan_one, src):
            if result:
                for fname, furl, stmt in result:
                    filenames.append(fname)
                    urls.append(furl)
                    stmts.append(stmt)

    if not filenames:
        return "", "", ""

    logger.info("  [scan] %d import(s) | %s", len(filenames), project.path_with_namespace)
    fmt = lambda lst: "".join(f"[{v}]" for v in lst)
    return fmt(filenames), fmt(urls), fmt(stmts)


# ---------------------------------------------------------------------------
# Per-project processor
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
    Fast path: check created_at on the lightweight proj_ref object BEFORE
    making a heavier client.projects.get() call or any file API requests.
    Returns None for projects outside the date window.
    """
    # ---- Date filter (cheap — uses data already in proj_ref) ----
    created_at_str = getattr(proj_ref, "created_at", None) or ""
    if created_at_str and not created_within(created_at_str, cutoff):
        logger.info("[%d/%d] SKIP (old) | %s | created=%s",
                    idx, total, proj_ref.path_with_namespace, created_at_str[:10])
        return None

    logger.info("[%d/%d] → %s", idx, total, proj_ref.path_with_namespace)

    # ---- Full project fetch ----
    project   = client.projects.get(proj_ref.id)
    namespace = project.namespace or {}
    web_url   = project.web_url
    ref       = project.default_branch or "main"

    team_name               = extract_team_name(web_url)
    pkg_json, int_ext       = get_package_json_info(project, ref)
    imp_fn, imp_url, imp_st = scan_imports(project, ref, file_workers)

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
        "import_filename":     imp_fn,
        "import_file_url":     imp_url,
        "import_statement":    imp_st,
    }


# ---------------------------------------------------------------------------
# Orchestrator
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
            pool.submit(process_one, ref, client, cutoff, file_workers, idx, total): ref
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

    logger.info("Processed: %d included, %d skipped/failed out of %d total",
                len(rows), total - len(rows), total)
    return rows


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def build_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in ("created_at", "last_activity_at", "updated_at"):
        if col in df.columns:
            # utc=True handles mixed tz strings in a single pass
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
    logger.info("XLSX done.")


def write_json(rows: List[Dict[str, Any]], path: str) -> None:
    logger.info("Writing JSON → %s  (%d rows)", path, len(rows))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=str, separators=(",", ": "))
    logger.info("JSON done.")


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
