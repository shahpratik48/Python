#!/usr/bin/env python3
"""
Generate an Excel report that links STAAT issues to ODM branches and enumerates
every file that changed on those branches relative to `odm-master`.

The script:
1. Resolves the current or previous iteration for the insights group.
2. Pulls all issues tied to that iteration.
3. Finds ODM branches whose names contain the issue ID.
4. For each branch, compares it against `odm-master` (or another base) and
   records every changed file (optionally filtered by a path fragment).
5. Falls back to the latest merged merge request if the branch no longer
   diverges from the base (e.g., already merged).
6. Writes the aggregated results to `odm_rules_release_list_<timestamp>.xlsx`
   alongside an `issues_export.csv` for traceability.

Prerequisites:
    pip install python-gitlab pandas openpyxl
"""
from __future__ import annotations

import argparse
import datetime as dt
import getpass
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import gitlab
import pandas as pd

# --- Defaults (override via CLI flags if needed) ---------------------------------
DEFAULT_GITLAB_URL = "https://devcloud.ubs.net"
DEFAULT_ODM_PROJECT_PATH = (
    "ubs/gwma/smart-technology-and-analytics/staat-data-science/"
    "staat-ds-genesis/genesis-platform/odm-dags"
)
DEFAULT_GROUP_PATH = (
    "ubs/gwma/smart-technology-and-analytics/staat-data-science/"
    "staat-ds-insights-cl/commons"
)
DEFAULT_ISSUES_PROJECT_PATH = (
    "ubs/gwma/smart-technology-and-analytics/staat-data-science/"
    "staat-ds-insights-cl/commons/staat-ds-insights-home"
)
DEFAULT_BASE_BRANCH = "odm-master"
DEFAULT_PATH_FILTER = "dags/odm/script/sql"
DEFAULT_TOKEN_ENV = "GITLAB_PRIVATE_TOKEN"

# ------------------------------------------------------------------------------- #


def running_inside_ipykernel() -> bool:
    """Detect whether the script is running inside an IPython kernel."""
    launcher = Path(sys.argv[0]).name if sys.argv else ""
    if "ipykernel_launcher" in launcher:
        return True
    return any("ipykernel_launcher" in arg for arg in sys.argv[1:])


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Produce an Excel export of all files changed per ODM branch."
    )
    parser.add_argument(
        "--iteration",
        choices=("Current", "Previous"),
        help="Iteration to report on (default: prompt).",
    )
    parser.add_argument(
        "--gitlab-url",
        default=DEFAULT_GITLAB_URL,
        help=f"GitLab base URL (default: {DEFAULT_GITLAB_URL}).",
    )
    parser.add_argument(
        "--odm-project-path",
        default=DEFAULT_ODM_PROJECT_PATH,
        help="Path of the ODM project that hosts branches (group/project).",
    )
    parser.add_argument(
        "--group-path",
        default=DEFAULT_GROUP_PATH,
        help="Group path used to resolve iterations.",
    )
    parser.add_argument(
        "--issues-project-path",
        default=DEFAULT_ISSUES_PROJECT_PATH,
        help="Project path containing the STAAT issues.",
    )
    parser.add_argument(
        "--base-branch",
        default=DEFAULT_BASE_BRANCH,
        help="Branch to compare against (default: odm-master).",
    )
    parser.add_argument(
        "--path-filter",
        default=DEFAULT_PATH_FILTER,
        help=(
            "Substring that must appear in the file path to include it. "
            "Provide an empty string to disable filtering."
        ),
    )
    parser.add_argument(
        "--token-env",
        default=DEFAULT_TOKEN_ENV,
        help=(
            "Environment variable that stores the GitLab private token. "
            "If missing, a prompt will be shown."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory where the Excel and CSV outputs will be written.",
    )
    parser.add_argument(
        "--max-branches",
        type=int,
        help="Process at most N branches (useful for dry runs or debugging).",
    )
    parser.add_argument(
        "--straight-compare",
        action="store_true",
        help=(
            "Use a straight diff (two-dot) when comparing branches. By default a "
            "merge-base diff is used so only files changed on the branch are reported."
        ),
    )
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        if running_inside_ipykernel():
            print(
                f"Ignoring unrecognized arguments from IPython: {' '.join(unknown)}",
                file=sys.stderr,
            )
        else:
            parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    return args


def resolve_private_token(env_var: str) -> str:
    token = os.environ.get(env_var)
    if token:
        return token.strip()
    token = getpass.getpass("Enter your private token: ").strip()
    if not token:
        raise SystemExit("A GitLab private token is required.")
    return token


def prompt_iteration(default: str = "Current") -> str:
    try:
        value = input("Current or Previous Iteration: ").strip()
    except EOFError:
        value = ""
    value = value or default
    normalized = value.lower()
    if normalized.startswith("p"):
        result = "Previous"
    else:
        result = "Current"
    print(f"Iteration: {result}")
    return result


def get_iteration_ids(group, iteration_choice: str) -> Tuple[int, str]:
    iterations = group.iterations.list(state="current")
    if not iterations:
        raise RuntimeError("No current iteration found for the group.")
    current_iter = iterations[0]

    closed_iters = group.iterations.list(state="closed", get_all=True) or []
    if closed_iters:
        closed_iters = sorted(
            closed_iters,
            key=lambda it: it.due_date or it.start_date or "",
        )
        previous_iter = closed_iters[-1]
    else:
        previous_iter = current_iter

    if iteration_choice == "Previous":
        return previous_iter.id, previous_iter.title
    return current_iter.id, current_iter.title


def issue_to_dict(issue) -> Dict[str, Optional[str]]:
    labels = issue.labels or []
    epic_title = None
    epic_attr = getattr(issue, "epic", None)
    if isinstance(epic_attr, dict):
        epic_title = epic_attr.get("title")
    return {
        "issue_id": str(issue.iid),
        "issue_title": issue.title,
        "issue_state": issue.state,
        "issue_weight": issue.weight,
        "issue_labels": ", ".join(labels),
        "issue_epic": epic_title,
        "issue_web_url": getattr(issue, "web_url", None),
    }


def build_issues_dataframe(issues: Iterable) -> pd.DataFrame:
    rows = [issue_to_dict(issue) for issue in issues]
    if not rows:
        return pd.DataFrame(
            columns=[
                "issue_id",
                "issue_title",
                "issue_state",
                "issue_weight",
                "issue_labels",
                "issue_epic",
                "issue_web_url",
            ]
        )
    df = pd.DataFrame(rows)
    df = df.sort_values(by="issue_id").reset_index(drop=True)
    return df


def extract_digits(text: Optional[str]) -> Optional[str]:
    if not isinstance(text, str):
        return None
    match = re.search(r"\d+", text)
    return match.group(0) if match else None


def build_branch_dataframe(branches: List[gitlab.v4.objects.Branch]) -> pd.DataFrame:
    names = [branch.name for branch in branches]
    df = pd.DataFrame({"Name": names})
    df["issue_id"] = df["Name"].apply(extract_digits)
    df = df.dropna(subset=["issue_id"]).reset_index(drop=True)
    df["issue_id"] = df["issue_id"].astype(str)
    return df


def infer_change_type(diff: dict) -> str:
    if diff.get("new_file"):
        return "added"
    if diff.get("deleted_file"):
        return "deleted"
    if diff.get("renamed_file"):
        return "renamed"
    return "modified"


def should_include(diff: dict, filter_fragment: Optional[str]) -> bool:
    if not filter_fragment:
        return True
    fragment = filter_fragment.strip()
    if not fragment:
        return True
    old_path = diff.get("old_path") or ""
    new_path = diff.get("new_path") or ""
    return fragment in old_path or fragment in new_path


def gather_compare_diffs(
    project,
    base_branch: str,
    branch_name: str,
    straight_compare: bool,
) -> Optional[dict]:
    try:
        comparison = project.repository_compare(
            base_branch,
            branch_name,
            straight=straight_compare,
        )
        return comparison
    except gitlab.GitlabGetError as exc:
        print(f"    ⚠️  Compare failed for {branch_name}: {exc}")
        return None


def gather_merged_mr_changes(project, branch_name: str):
    try:
        mrs = project.mergerequests.list(
            source_branch=branch_name,
            state="merged",
            order_by="updated_at",
            sort="desc",
            per_page=1,
        )
    except gitlab.GitlabListError as exc:
        print(f"    ⚠️  Merge request lookup failed for {branch_name}: {exc}")
        return None, None

    if not mrs:
        return None, None
    mr = project.mergerequests.get(mrs[0].iid)
    mr_details = mr.changes()
    return mr, mr_details


def branch_commit_meta(branch) -> Dict[str, Optional[str]]:
    commit = getattr(branch, "commit", None) or {}
    return {
        "branch_head_sha": commit.get("id"),
        "branch_head_short_sha": commit.get("short_id"),
        "branch_head_title": commit.get("title"),
        "branch_head_message": commit.get("message"),
        "branch_head_author": commit.get("author_name"),
        "branch_head_author_email": commit.get("author_email"),
        "branch_head_committed_at": commit.get("committed_date"),
        "branch_head_created_at": commit.get("created_at"),
    }


def record_from_diff(
    issue_row: Dict[str, str],
    branch_name: str,
    branch_meta: Dict[str, Optional[str]],
    base_branch: str,
    diff: dict,
    source: str,
    change_timestamp: Optional[str],
    mr_meta: Optional[dict] = None,
) -> Dict[str, Optional[str]]:
    file_path = diff.get("new_path") or diff.get("old_path")
    file_name = Path(file_path).name if file_path else None
    record = {
        **issue_row,
        "branch_name": branch_name,
        "base_branch": base_branch,
        "file_name": file_name,
        "file_path": diff.get("new_path"),
        "old_path": diff.get("old_path"),
        "new_path": diff.get("new_path"),
        "change_type": infer_change_type(diff),
        "change_source": source,
        "change_timestamp": change_timestamp,
    }
    record.update(branch_meta)
    if source == "merge_request" and mr_meta:
        record.update(mr_meta)
    return record


def collect_branch_changes(
    odm_project,
    base_branch: str,
    df_issue_branches: pd.DataFrame,
    branch_lookup: Dict[str, gitlab.v4.objects.Branch],
    filter_fragment: Optional[str],
    straight_compare: bool,
    max_branches: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    records: List[Dict[str, Optional[str]]] = []
    errors: List[str] = []
    rows = df_issue_branches.to_dict("records")
    if max_branches is not None:
        rows = rows[:max_branches]

    for idx, row in enumerate(rows, start=1):
        branch_name = row["Name"]
        print(f"[{idx}/{len(rows)}] Processing branch: {branch_name}")
        branch_obj = branch_lookup.get(branch_name)
        branch_meta = {
            "branch_head_sha": None,
            "branch_head_short_sha": None,
            "branch_head_title": None,
            "branch_head_message": None,
            "branch_head_author": None,
            "branch_head_author_email": None,
            "branch_head_committed_at": None,
            "branch_head_created_at": None,
            "branch_web_url": getattr(branch_obj, "web_url", None)
            if branch_obj
            else None,
        }

        if branch_obj is None:
            try:
                branch_obj = odm_project.branches.get(branch_name)
                branch_lookup[branch_name] = branch_obj
            except gitlab.GitlabGetError as exc:
                msg = f"Branch {branch_name} not found: {exc}"
                print(f"    ⚠️  {msg}")
                errors.append(msg)
                continue

        branch_meta.update(branch_commit_meta(branch_obj))

        comparison = gather_compare_diffs(
            odm_project,
            base_branch,
            branch_name,
            straight_compare=straight_compare,
        )
        diffs = comparison.get("diffs", []) if comparison else []

        filtered_diffs = [
            diff for diff in diffs if should_include(diff, filter_fragment)
        ]

        if filtered_diffs:
            for diff in filtered_diffs:
                records.append(
                    record_from_diff(
                        issue_row=row,
                        branch_name=branch_name,
                        branch_meta=branch_meta,
                        base_branch=base_branch,
                        diff=diff,
                        source="compare",
                        change_timestamp=branch_meta["branch_head_committed_at"],
                    )
                )
            continue

        # Fallback: merged MR diff
        mr, mr_details = gather_merged_mr_changes(odm_project, branch_name)
        if not mr_details:
            msg = (
                f"No file changes found for branch {branch_name} via compare or "
                "merged merge requests."
            )
            print(f"    ℹ️  {msg}")
            errors.append(msg)
            continue

        mr_meta = {
            "merge_request_iid": mr.iid,
            "merge_request_title": mr.title,
            "merge_request_state": mr.state,
            "merge_request_web_url": mr.web_url,
            "merge_commit_sha": mr.merge_commit_sha,
            "merge_request_target_branch": mr.target_branch,
            "merge_request_source_branch": mr.source_branch,
            "merge_request_merged_at": mr.merged_at,
        }

        for diff in mr_details.get("changes", []):
            if not should_include(diff, filter_fragment):
                continue
            records.append(
                record_from_diff(
                    issue_row=row,
                    branch_name=branch_name,
                    branch_meta=branch_meta,
                    base_branch=base_branch,
                    diff=diff,
                    source="merge_request",
                    change_timestamp=mr_meta["merge_request_merged_at"],
                    mr_meta=mr_meta,
                )
            )

    df_records = pd.DataFrame.from_records(records)
    if not df_records.empty:
        df_records = df_records.sort_values(
            ["branch_name", "file_path", "change_type"]
        ).reset_index(drop=True)
    return df_records, errors


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    token = resolve_private_token(args.token_env)
    iteration_choice = args.iteration or prompt_iteration()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    excel_path = output_dir / f"odm_rules_release_list_{timestamp}.xlsx"
    issues_csv_path = output_dir / "issues_export.csv"

    gl = gitlab.Gitlab(args.gitlab_url, private_token=token)
    odm_project = gl.projects.get(args.odm_project_path)
    group = gl.groups.get(args.group_path)
    issues_project = gl.projects.get(args.issues_project_path)

    iteration_id, iteration_title = get_iteration_ids(group, iteration_choice)
    print(f"Using iteration '{iteration_title}' (id={iteration_id}).")

    issues = issues_project.issues.list(iteration_id=iteration_id, get_all=True)
    df_issues = build_issues_dataframe(issues)
    if df_issues.empty:
        print("No issues found for the selected iteration.")
        df_issue_branches = pd.DataFrame(
            columns=[
                "issue_id",
                "issue_title",
                "issue_state",
                "issue_weight",
                "issue_labels",
                "issue_epic",
                "issue_web_url",
                "Name",
            ]
        )
    else:
        odm_branches = odm_project.branches.list(get_all=True)
        df_branches = build_branch_dataframe(odm_branches)
        df_issue_branches = pd.merge(
            df_issues,
            df_branches,
            how="left",
            left_on="issue_id",
            right_on="issue_id",
        )
        df_issue_branches = df_issue_branches.dropna(subset=["Name"]).reset_index(
            drop=True
        )

    df_issue_branches.to_csv(issues_csv_path, index=False)
    print(f"Wrote branch/issue mapping to {issues_csv_path}")

    if df_issue_branches.empty:
        print("No branches matched the issues; skipping change collection.")
        return 0

    odm_branch_lookup = {
        branch.name: branch for branch in odm_project.branches.list(get_all=True)
    }

    path_filter = args.path_filter if args.path_filter else None
    df_changes, errors = collect_branch_changes(
        odm_project=odm_project,
        base_branch=args.base_branch,
        df_issue_branches=df_issue_branches,
        branch_lookup=odm_branch_lookup,
        filter_fragment=path_filter,
        straight_compare=args.straight_compare,
        max_branches=args.max_branches,
    )

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_issue_branches.to_excel(writer, index=False, sheet_name="issues_branches")
        if df_changes.empty:
            empty_cols = [
                "issue_id",
                "issue_title",
                "branch_name",
                "change_type",
                "file_path",
            ]
            pd.DataFrame(columns=empty_cols).to_excel(
                writer, index=False, sheet_name="branch_changes"
            )
        else:
            df_changes.to_excel(writer, index=False, sheet_name="branch_changes")

    print(f"Exported results to {excel_path}")
    if errors:
        print("\nWarnings encountered:")
        for msg in errors:
            print(f" - {msg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
