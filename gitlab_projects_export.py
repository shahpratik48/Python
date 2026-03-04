import argparse
import getpass
import logging
from typing import List, Dict, Any, Tuple

import gitlab
import pandas as pd

DEFAULT_GITLAB_URL = "https://devcloud.ubs.net"
DEFAULT_GROUP_PATH = "ubs/gwma"
DEFAULT_OUTPUT = "gitlab_projects_export.xlsx"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"

logger = logging.getLogger("gitlab_export")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export GitLab group projects with attributes to XLSX."
    )
    parser.add_argument("--gitlab-url", default=DEFAULT_GITLAB_URL)
    parser.add_argument("--group-path", default=DEFAULT_GROUP_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--per-page", type=int, default=100)
    args, _ = parser.parse_known_args()
    return args


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def get_gitlab_client(url: str, private_token: str) -> gitlab.Gitlab:
    return gitlab.Gitlab(url, private_token=private_token)


def extract_team_name(web_url: str) -> str:
    """
    Extract the second-last path segment (folder) from the project URL as team_name.
    e.g. https://devcloud.ubs.net/ubs/gwm/xxx/yyy/zzz/projectname -> zzz
    """
    try:
        path = web_url.rstrip("/").split("//", 1)[-1]  # strip scheme
        parts = path.split("/")
        # parts[0] = host, parts[1:] = path segments
        # Last segment is the project name, second-to-last is team_name
        segments = parts[1:]  # exclude hostname
        if len(segments) >= 2:
            return segments[-2]
    except Exception:
        pass
    return ""


def get_package_json_info(project: Any) -> Tuple[str, str]:
    """
    Check if package.json exists in the project's default branch.
    Looks at the 'dependencies' object — if any key starts with '@uwr/', int_ext = internal, else external.
    Returns (package_json: 'Yes'/'No', int_ext: 'internal'/'external'/'-')
    """
    import json

    ref = project.default_branch or "main"
    try:
        file_obj = project.files.get(file_path="package.json", ref=ref)
        content = file_obj.decode().decode("utf-8")

        try:
            pkg = json.loads(content)
        except json.JSONDecodeError as je:
            logger.warning("  [package.json] JSON PARSE ERROR | %s | %s", project.path_with_namespace, je)
            return "Yes", "external"

        dependencies = pkg.get("dependencies", {})
        uwr_deps = [dep for dep in dependencies if dep.startswith("@uwr/")]

        if uwr_deps:
            logger.info("  [package.json] FOUND | %s | internal | @uwr/ deps: %s", project.path_with_namespace, uwr_deps)
            return "Yes", "internal"
        else:
            logger.info("  [package.json] FOUND | %s | external | no @uwr/ deps found", project.path_with_namespace)
            return "Yes", "external"

    except gitlab.exceptions.GitlabGetError:
        logger.info("  [package.json] NOT FOUND | %s", project.path_with_namespace)
        return "No", "-"
    except Exception as e:
        logger.warning("  [package.json] ERROR | %s | %s", project.path_with_namespace, e)
        return "No", "-"


def project_to_row(project: Any) -> Dict[str, Any]:
    namespace = project.namespace or {}
    web_url = project.web_url

    team_name = extract_team_name(web_url)
    package_json, int_ext = get_package_json_info(project)

    return {
        "project_id": project.id,
        "name": project.name,
        "path": project.path,
        "path_with_namespace": project.path_with_namespace,
        "group_path": namespace.get("full_path"),
        "web_url": web_url,
        "description": project.description,
        "visibility": project.visibility,
        "archived": project.archived,
        "created_at": project.created_at,
        "last_activity_at": project.last_activity_at,
        "updated_at": project.updated_at,
        "default_branch": getattr(project, "default_branch", None),
        "forks_count": getattr(project, "forks_count", None),
        "star_count": getattr(project, "star_count", None),
        "open_issues_count": getattr(project, "open_issues_count", None),
        # New columns
        "team_name": team_name,
        "package_json": package_json,
        "int_ext": int_ext,
        "component": "",
    }


def export_group_projects(
    client: gitlab.Gitlab, group_path: str, per_page: int
) -> List[Dict[str, Any]]:
    logger.info("Fetching group '%s'", group_path)
    group = client.groups.get(group_path)
    projects = group.projects.list(include_subgroups=True, all=True, per_page=per_page)

    total = len(projects)
    logger.info("Found %s project references — starting detail fetch", total)

    rows: List[Dict[str, Any]] = []
    for idx, proj_ref in enumerate(projects, start=1):
        logger.info("[%d/%d] Processing: %s", idx, total, proj_ref.path_with_namespace)
        try:
            project = client.projects.get(proj_ref.id)
            rows.append(project_to_row(project))
        except Exception as e:
            logger.error("[%d/%d] FAILED to fetch project %s: %s", idx, total, proj_ref.id, e)

    logger.info("Resolved %s/%s projects successfully", len(rows), total)
    return rows


def write_xlsx(rows: List[Dict[str, Any]], output_path: str) -> None:
    logger.info("Writing %s rows to '%s'", len(rows), output_path)
    df = pd.DataFrame(rows)

    date_columns = ["created_at", "last_activity_at", "updated_at"]
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            if pd.api.types.is_datetime64tz_dtype(df[col]):
                df[col] = df[col].dt.tz_localize(None)

    df.sort_values(by=["group_path", "name"], inplace=True, ignore_index=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="projects")

        ws = writer.sheets["projects"]

        # Auto-fit column widths
        for col_cells in ws.columns:
            max_len = max((len(str(c.value)) if c.value is not None else 0) for c in col_cells)
            ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 4, 60)


def main() -> None:
    configure_logging()
    args = parse_args()
    logger.info("Starting export for group '%s'", args.group_path)
    private_token = getpass.getpass("Enter your private token: ")

    client = get_gitlab_client(args.gitlab_url, private_token)
    rows = export_group_projects(client, args.group_path, args.per_page)

    if not rows:
        logger.warning("No projects found.")
        return

    write_xlsx(rows, args.output)
    logger.info(
        "Exported %s projects from '%s' to '%s'.",
        len(rows),
        args.group_path,
        args.output,
    )


if __name__ == "__main__":
    main()
