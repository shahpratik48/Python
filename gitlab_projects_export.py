import argparse
import getpass
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import gitlab
import pandas as pd

DEFAULT_GITLAB_URL = "https://devcloud.ubs.net"
DEFAULT_GROUP_PATH = "ubs/gwma"
DEFAULT_OUTPUT_XLSX = "gitlab_projects_export.xlsx"
DEFAULT_OUTPUT_JSON = "gitlab_projects_export.json"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
MAX_WORKERS = 10   # parallel project threads
FILE_WORKERS = 5   # parallel file-fetch threads per project

# Matches: import { ... } from '@ubs.websdk/...' or '@uwr.../...'
IMPORT_PATTERN = re.compile(
    r"""^[ \t]*import\s+\{[^}]+\}(?:\s*,\s*\{[^}]+\})*\s+from\s+['"](@ubs\.websdk/[^'"]+|@uwr[^'"]+)['"].*$""",
    re.MULTILINE,
)
SOURCE_EXTENSIONS = (".jsx", ".tsx", ".js", ".ts")

logger = logging.getLogger("gitlab_export")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export GitLab group projects with attributes to XLSX + JSON."
    )
    parser.add_argument("--gitlab-url", default=DEFAULT_GITLAB_URL)
    parser.add_argument("--group-path", default=DEFAULT_GROUP_PATH)
    parser.add_argument("--output-xlsx", default=DEFAULT_OUTPUT_XLSX)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--per-page", type=int, default=100)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    args, _ = parser.parse_known_args()
    return args


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


# ---------------------------------------------------------------------------
# GitLab client
# ---------------------------------------------------------------------------

def get_gitlab_client(url: str, private_token: str) -> gitlab.Gitlab:
    return gitlab.Gitlab(url, private_token=private_token)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_team_name(web_url: str) -> str:
    """Second-last URL path segment = team folder.
    e.g. https://host/ubs/gwm/xxx/yyy/zzz/projectname  ->  zzz
    """
    try:
        segments = web_url.rstrip("/").split("//", 1)[-1].split("/")[1:]
        if len(segments) >= 2:
            return segments[-2]
    except Exception:
        pass
    return ""


def get_package_json_info(project: Any, ref: str) -> Tuple[str, str]:
    """
    Fetch package.json and check dependencies for @uwr/ prefixed packages.
    Returns ('Yes'/'No', 'internal'/'external'/'-').
    """
    try:
        raw = project.files.get(file_path="package.json", ref=ref).decode()
        content = raw if isinstance(raw, str) else raw.decode("utf-8")
        pkg = json.loads(content)
        deps = pkg.get("dependencies", {})
        uwr_deps = [d for d in deps if d.startswith("@uwr/")]
        if uwr_deps:
            logger.info("  [pkg.json] internal | %s | %s", project.path_with_namespace, uwr_deps)
            return "Yes", "internal"
        logger.info("  [pkg.json] external (no @uwr/ deps) | %s", project.path_with_namespace)
        return "Yes", "external"
    except gitlab.exceptions.GitlabGetError:
        logger.info("  [pkg.json] not found | %s", project.path_with_namespace)
        return "No", "-"
    except json.JSONDecodeError as e:
        logger.warning("  [pkg.json] parse error | %s | %s", project.path_with_namespace, e)
        return "Yes", "external"
    except Exception as e:
        logger.warning("  [pkg.json] error | %s | %s", project.path_with_namespace, e)
        return "No", "-"


# ---------------------------------------------------------------------------
# Import scanning
# ---------------------------------------------------------------------------

def _fetch_content(project: Any, path: str, ref: str) -> Optional[str]:
    try:
        raw = project.files.get(file_path=path, ref=ref).decode()
        return raw if isinstance(raw, str) else raw.decode("utf-8", errors="replace")
    except Exception:
        return None


def scan_imports(project: Any, ref: str) -> Tuple[str, str, str]:
    """
    Recursively walk repo tree, scan .js/.ts/.jsx/.tsx files for qualifying imports.
    Returns bracket-delimited (import_filename, import_file_url, import_statement).
    Each import line produces one bracketed entry — multiple imports per file or
    across files are all captured in one record per project.
    """
    try:
        items = project.repository_tree(ref=ref, recursive=True, all=True, per_page=100)
    except Exception as e:
        logger.warning("  [scan] tree error | %s | %s", project.path_with_namespace, e)
        return "", "", ""

    source_files = [
        item for item in items
        if item.get("type") == "blob"
        and item["path"].endswith(SOURCE_EXTENSIONS)
    ]
    logger.info("  [scan] %d source files | %s", len(source_files), project.path_with_namespace)

    filenames: List[str] = []
    file_urls: List[str] = []
    statements: List[str] = []

    def scan_one(item: Dict) -> Optional[List[Tuple[str, str, str]]]:
        path = item["path"]
        content = _fetch_content(project, path, ref)
        if not content:
            return None
        matches = list(IMPORT_PATTERN.finditer(content))
        if not matches:
            return None
        fname = path.split("/")[-1]
        furl = f"{project.web_url}/-/blob/{ref}/{path}"
        return [(fname, furl, m.group(0).strip()) for m in matches]

    with ThreadPoolExecutor(max_workers=FILE_WORKERS) as pool:
        futures = [pool.submit(scan_one, item) for item in source_files]
        for future in as_completed(futures):
            result = future.result()
            if result:
                for fname, furl, stmt in result:
                    filenames.append(fname)
                    file_urls.append(furl)
                    statements.append(stmt)

    if not filenames:
        logger.info("  [scan] no matching imports | %s", project.path_with_namespace)
        return "", "", ""

    logger.info("  [scan] %d import(s) found | %s", len(filenames), project.path_with_namespace)
    fmt = lambda lst: "".join(f"[{v}]" for v in lst)
    return fmt(filenames), fmt(file_urls), fmt(statements)


# ---------------------------------------------------------------------------
# Per-project row builder
# ---------------------------------------------------------------------------

def process_project(project: Any) -> Dict[str, Any]:
    namespace = project.namespace or {}
    web_url = project.web_url
    ref = project.default_branch or "main"

    team_name                           = extract_team_name(web_url)
    package_json, int_ext               = get_package_json_info(project, ref)
    import_filename, import_file_url, import_statement = scan_imports(project, ref)

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
        "package_json":        package_json,
        "int_ext":             int_ext,
        "component":           "",
        "import_filename":     import_filename,
        "import_file_url":     import_file_url,
        "import_statement":    import_statement,
    }


# ---------------------------------------------------------------------------
# Parallel project export
# ---------------------------------------------------------------------------

def export_group_projects(
    client: gitlab.Gitlab,
    group_path: str,
    per_page: int,
    max_workers: int,
) -> List[Dict[str, Any]]:
    logger.info("Fetching group '%s'", group_path)
    group = client.groups.get(group_path)
    proj_refs = group.projects.list(include_subgroups=True, all=True, per_page=per_page)

    total = len(proj_refs)
    logger.info("Found %d projects — processing with %d workers", total, max_workers)

    rows: List[Dict[str, Any]] = []
    done = 0

    def fetch_and_process(proj_ref: Any, idx: int) -> Dict[str, Any]:
        logger.info("[%d/%d] → %s", idx, total, proj_ref.path_with_namespace)
        project = client.projects.get(proj_ref.id)
        return process_project(project)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {
            pool.submit(fetch_and_process, ref, idx): ref
            for idx, ref in enumerate(proj_refs, start=1)
        }
        for future in as_completed(future_map):
            done += 1
            ref = future_map[future]
            try:
                row = future.result()
                rows.append(row)
                logger.info("[%d/%d ✓] %s", done, total, row["path_with_namespace"])
            except Exception as e:
                logger.error("[%d/%d ✗] %s | %s", done, total, ref.path_with_namespace, e)

    logger.info("Done: %d/%d projects processed", len(rows), total)
    return rows


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _build_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in ("created_at", "last_activity_at", "updated_at"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            if pd.api.types.is_datetime64tz_dtype(df[col]):
                df[col] = df[col].dt.tz_localize(None)
    df.sort_values(by=["group_path", "name"], inplace=True, ignore_index=True)
    return df


def write_xlsx(df: pd.DataFrame, path: str) -> None:
    logger.info("Writing XLSX → %s", path)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="projects")
        ws = writer.sheets["projects"]
        for col_cells in ws.columns:
            max_len = max((len(str(c.value)) if c.value is not None else 0) for c in col_cells)
            ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 4, 80)
    logger.info("XLSX done: %s", path)


def write_json(rows: List[Dict[str, Any]], path: str) -> None:
    logger.info("Writing JSON → %s", path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=str)
    logger.info("JSON done: %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    configure_logging()
    args = parse_args()
    logger.info("Export start | group='%s' | workers=%d", args.group_path, args.workers)

    private_token = getpass.getpass("Enter your GitLab private token: ")
    client = get_gitlab_client(args.gitlab_url, private_token)

    rows = export_group_projects(client, args.group_path, args.per_page, args.workers)
    if not rows:
        logger.warning("No projects found — exiting.")
        return

    df = _build_dataframe(rows)
    write_xlsx(df, args.output_xlsx)
    write_json(rows, args.output_json)
    logger.info("All done | %d projects | %s | %s", len(rows), args.output_xlsx, args.output_json)


if __name__ == "__main__":
    main()
