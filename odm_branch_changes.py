#!/usr/bin/env python3
"""
Export the list of files that differ between ODM feature branches and `odm-master`.

The script expects a CSV or Excel file that contains a column with the branch
names (default column name: "Name"). For each branch, it calls the GitLab
compare API and records each changed file together with the change type
("added", "deleted", "renamed", "modified"). The aggregated result is written
to an Excel workbook named `odm_rules_details_<timestamp>.xlsx` by default.
"""
from __future__ import annotations

import argparse
import datetime as dt
import getpass
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import gitlab
import pandas as pd


DEFAULT_GITLAB_URL = "https://devcloud.ubs.net"
DEFAULT_PROJECT_PATH = (
    "ubs/gwma/smart-technology-and-analytics/staat-data-science/"
    "staat-ds-genesis/genesis-platform/ikg-dags"
)
DEFAULT_BASE_BRANCH = "odm-master"
DEFAULT_FILTER_PATH = "dags/odm/script/sql"
DEFAULT_BRANCH_COLUMN = "Name"
DEFAULT_TOKEN_ENV_VAR = "GITLAB_PRIVATE_TOKEN"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export per-branch ODM file changes to an Excel workbook."
    )
    parser.add_argument(
        "--input-file",
        required=True,
        help="Path to CSV/XLSX file containing a column with ODM branch names.",
    )
    parser.add_argument(
        "--branch-column",
        default=DEFAULT_BRANCH_COLUMN,
        help=f"Column name that stores branch names (default: {DEFAULT_BRANCH_COLUMN!r}).",
    )
    parser.add_argument(
        "--gitlab-url",
        default=DEFAULT_GITLAB_URL,
        help=f"GitLab base URL (default: {DEFAULT_GITLAB_URL}).",
    )
    parser.add_argument(
        "--project-path",
        default=DEFAULT_PROJECT_PATH,
        help="Full GitLab project path (group/project).",
    )
    parser.add_argument(
        "--base-branch",
        default=DEFAULT_BASE_BRANCH,
        help=f"Branch to compare against (default: {DEFAULT_BASE_BRANCH}).",
    )
    parser.add_argument(
        "--filter-path",
        default=DEFAULT_FILTER_PATH,
        help=(
            "Optional substring filter for file paths. Set to an empty string to disable "
            f"(default: {DEFAULT_FILTER_PATH})."
        ),
    )
    parser.add_argument(
        "--token-env-var",
        default=DEFAULT_TOKEN_ENV_VAR,
        help=(
            "Name of the environment variable that stores the GitLab private token. "
            f"If unset, you will be prompted (default: {DEFAULT_TOKEN_ENV_VAR})."
        ),
    )
    parser.add_argument(
        "--output-file",
        help=(
            "Destination XLSX path. Defaults to ./odm_rules_details_<timestamp>.xlsx "
            "in the working directory."
        ),
    )
    parser.add_argument(
        "--include-base-branch",
        action="store_true",
        help="Include the base branch itself in the results (normally skipped).",
    )
    parser.add_argument(
        "--max-branches",
        type=int,
        help="Optional limit on the number of branches processed (useful for testing).",
    )
    return parser.parse_args(argv)


def resolve_private_token(env_var: str) -> str:
    token = os.environ.get(env_var)
    if token:
        return token
    token = getpass.getpass("Enter your private token: ").strip()
    if not token:
        raise SystemExit("A private token is required to call the GitLab API.")
    return token


def load_branch_names(path: Path, column: str) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Branch list file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(
            f"Unsupported file extension '{path.suffix}'. Provide a CSV or Excel file."
        )

    if column not in df.columns:
        raise KeyError(
            f"Column '{column}' does not exist in {path}. "
            f"Available columns: {', '.join(df.columns)}"
        )

    branches = (
        df[column]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .unique()
    )
    return [branch for branch in branches if branch]


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


def collect_branch_changes(
    project: gitlab.v4.objects.Project,
    base_branch: str,
    branches: Iterable[str],
    filter_fragment: Optional[str],
) -> Tuple[pd.DataFrame, List[Tuple[str, Exception]]]:
    records = []
    errors: List[Tuple[str, Exception]] = []

    for idx, branch in enumerate(branches, start=1):
        print(f"[{idx}] Processing {branch} ...")
        try:
            comparison = project.repository_compare(
                base_branch, branch, straight=True
            )
        except Exception as exc:  # gitlab.GitlabGetError or others
            print(f"    ⚠️  Skipping {branch}: {exc}")
            errors.append((branch, exc))
            continue

        for diff in comparison.get("diffs", []):
            if not should_include(diff, filter_fragment):
                continue
            records.append(
                {
                    "branch_name": branch,
                    "base_branch": base_branch,
                    "change_type": infer_change_type(diff),
                    "old_path": diff.get("old_path"),
                    "new_path": diff.get("new_path"),
                }
            )

    df = pd.DataFrame.from_records(records)
    if not df.empty:
        df = df.sort_values(["branch_name", "change_type", "new_path"], ignore_index=True)
    return df, errors


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    token = resolve_private_token(args.token_env_var)
    input_path = Path(args.input_file).expanduser().resolve()
    output_path = (
        Path(args.output_file).expanduser().resolve()
        if args.output_file
        else Path.cwd()
        / f"odm_rules_details_{dt.datetime.now().strftime('%Y-%m-%d_%H%M%S')}.xlsx"
    )

    branches = load_branch_names(input_path, args.branch_column)
    if not args.include_base_branch:
        branches = [b for b in branches if b != args.base_branch]

    if args.max_branches:
        branches = branches[: args.max_branches]

    if not branches:
        raise SystemExit("No branches to process after filtering.")

    print(f"Loaded {len(branches)} branch names from {input_path}.")
    print(f"Connecting to {args.gitlab_url} ...")

    gl = gitlab.Gitlab(args.gitlab_url, private_token=token)
    project = gl.projects.get(args.project_path)

    filter_fragment = args.filter_path if args.filter_path else None
    df_changes, errors = collect_branch_changes(
        project,
        args.base_branch,
        branches,
        filter_fragment=filter_fragment,
    )

    if df_changes.empty:
        print("No file differences detected. Excel output skipped.")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_changes.to_excel(output_path, index=False)
        print(
            f"Wrote {len(df_changes)} change rows "
            f"for {df_changes['branch_name'].nunique()} branches -> {output_path}"
        )

    if errors:
        print("\nBranches with errors:")
        for branch, exc in errors:
            print(f" - {branch}: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
