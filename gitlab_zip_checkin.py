"""
gitlab_zip_checkin.py

Takes a zip file, unzips it, and checks in all folders / subfolders / files into a
GitLab branch under TARGET_FOLDER (see config.py), keeping the folder structure of
the zip.

- Branch is created from BASE_BRANCH if it does not exist yet.
- Files that already exist in the branch are replaced with the version from the zip.
- Files that do not exist yet are created.
- Nothing is deleted.

Usage (command line):
    python gitlab_zip_checkin.py                      # prompts for everything
    python gitlab_zip_checkin.py c:\\folder\\test.zip  # prompts for token + branch

Usage (notebook / import):
    from gitlab_zip_checkin import run
    run()                                             # prompts for everything
    run(zip_path=r"c:\\folder\\test.zip", branch="feature/x", token="glpat-...")

Dependencies: requests  (pip install requests)
"""

import base64
import getpass
import os
import sys
import tempfile
import zipfile
from urllib.parse import quote

import requests

import config


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class GitLab:
    """Minimal GitLab REST client (v4) using a personal / project access token."""

    def __init__(self, url, token, project_path, verify=True):
        self.base = url.rstrip("/") + "/api/v4"
        self.project = quote(project_path, safe="")
        self.session = requests.Session()
        self.session.headers.update({"PRIVATE-TOKEN": token})
        self.session.verify = verify

    # -- low level ----------------------------------------------------------
    def _url(self, path):
        return f"{self.base}/projects/{self.project}{path}"

    def _check(self, resp, ok=(200, 201)):
        if resp.status_code not in ok:
            raise RuntimeError(
                f"GitLab API error {resp.status_code} on {resp.request.method} "
                f"{resp.url}\n{resp.text[:1000]}"
            )
        return resp

    # -- project / branches -------------------------------------------------
    def project_info(self):
        return self._check(self.session.get(self._url(""))).json()

    def branch_exists(self, branch):
        r = self.session.get(self._url(f"/repository/branches/{quote(branch, safe='')}"))
        if r.status_code == 404:
            return False
        self._check(r)
        return True

    def create_branch(self, branch, ref):
        return self._check(
            self.session.post(
                self._url("/repository/branches"),
                params={"branch": branch, "ref": ref},
            )
        ).json()

    # -- tree ---------------------------------------------------------------
    def list_files(self, branch, folder):
        """Return the set of file paths (blobs) that exist under `folder` in `branch`."""
        files = set()
        page = 1
        while True:
            r = self.session.get(
                self._url("/repository/tree"),
                params={
                    "ref": branch,
                    "path": folder,
                    "recursive": "true",
                    "per_page": 100,
                    "page": page,
                },
            )
            if r.status_code == 404:  # folder does not exist yet
                return files
            self._check(r)
            for item in r.json():
                if item["type"] == "blob":
                    files.add(item["path"])
            next_page = r.headers.get("X-Next-Page")
            if not next_page:
                break
            page = int(next_page)
        return files

    # -- commits ------------------------------------------------------------
    def commit(self, branch, message, actions):
        payload = {"branch": branch, "commit_message": message, "actions": actions}
        return self._check(
            self.session.post(self._url("/repository/commits"), json=payload)
        ).json()


def _is_ignored(rel_path):
    norm = rel_path.replace("\\", "/")
    for pat in config.IGNORE_PATTERNS:
        if pat.endswith("/"):
            if norm.startswith(pat) or f"/{pat}" in norm:
                return True
        elif norm.endswith(pat) or os.path.basename(norm) == pat:
            return True
    return False


def _is_prohibited(rel_path):
    """True if the file extension is blocked by the project's GitLab push rule."""
    ext = os.path.splitext(rel_path)[1].lower()
    return ext in {e.lower() for e in getattr(config, "PROHIBITED_EXTENSIONS", [])}


def unzip(zip_path, dest_dir):
    """Extract ONLY the outer zip (zip files inside it are left packed) and return
    (files, inner_zips, prohibited):
    files      - list of (relative_path, absolute_path) to check in normally
    inner_zips - list of (relative_path, absolute_path) of .zip files found inside,
                 uploaded as-is in a separate commit (see UPLOAD_INNER_ZIPS)
    prohibited - list of relative paths skipped because of PROHIBITED_EXTENSIONS"""
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    files, inner_zips, prohibited = [], [], []
    for root, _dirs, names in os.walk(dest_dir):
        for name in names:
            abs_path = os.path.join(root, name)
            rel_path = os.path.relpath(abs_path, dest_dir).replace(os.sep, "/")
            if _is_ignored(rel_path):
                continue
            if rel_path.lower().endswith(".zip"):
                inner_zips.append((rel_path, abs_path))
                continue
            if _is_prohibited(rel_path):
                prohibited.append(rel_path)
                continue
            files.append((rel_path, abs_path))
    files.sort()
    inner_zips.sort()
    prohibited.sort()
    return files, inner_zips, prohibited


def _file_action(action, repo_path, abs_path):
    with open(abs_path, "rb") as fh:
        raw = fh.read()
    try:
        text = raw.decode("utf-8")
        if "\x00" in text:
            raise UnicodeDecodeError("utf-8", b"", 0, 1, "binary")
        return {"action": action, "file_path": repo_path, "content": text}
    except UnicodeDecodeError:
        return {
            "action": action,
            "file_path": repo_path,
            "content": base64.b64encode(raw).decode("ascii"),
            "encoding": "base64",
        }


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def run(zip_path=None, branch=None, token=None, dry_run=False):
    # ---- inputs -----------------------------------------------------------
    if not zip_path:
        zip_path = input("Path to zip file (e.g. c:\\folder\\test.zip): ").strip().strip('"')
    zip_path = os.path.expanduser(zip_path)
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"Zip file not found: {zip_path}")
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"Not a valid zip file: {zip_path}")

    if not token:
        token = getpass.getpass("GitLab token (input hidden): ").strip()
    if not token:
        raise ValueError("A GitLab token is required.")

    if not branch:
        branch = input(f"Branch name to check in to (created from '{config.BASE_BRANCH}' if missing): ").strip()
    if not branch:
        raise ValueError("A branch name is required.")

    gl = GitLab(config.GITLAB_URL, token, config.PROJECT_PATH, verify=config.VERIFY_SSL)

    # ---- sanity: project + branch ------------------------------------------
    info = gl.project_info()
    print(f"Project : {info['path_with_namespace']} (id {info['id']})")

    if gl.branch_exists(branch):
        print(f"Branch  : '{branch}' already exists - will commit on top of it.")
    else:
        if not gl.branch_exists(config.BASE_BRANCH):
            raise RuntimeError(f"Base branch '{config.BASE_BRANCH}' does not exist.")
        if dry_run:
            print(f"Branch  : '{branch}' would be created from '{config.BASE_BRANCH}' (dry run).")
        else:
            gl.create_branch(branch, config.BASE_BRANCH)
            print(f"Branch  : '{branch}' created from '{config.BASE_BRANCH}'.")

    # ---- unzip --------------------------------------------------------------
    target = config.TARGET_FOLDER.strip("/")
    with tempfile.TemporaryDirectory(prefix="zip_checkin_") as tmp:
        files, inner_zips, prohibited = unzip(zip_path, tmp)
        if prohibited:
            print(f"\nWARNING: {len(prohibited)} file(s) skipped - extension prohibited by the GitLab push rule:")
            for rel in prohibited:
                print(f"  [skip  ] {target}/{rel}")
            print()
        upload_inner = getattr(config, "UPLOAD_INNER_ZIPS", True)
        if inner_zips:
            if upload_inner:
                print(f"Inner zip file(s) found: {len(inner_zips)} - kept packed, uploaded as-is in a separate commit:")
            else:
                print(f"Inner zip file(s) found: {len(inner_zips)} - skipped (UPLOAD_INNER_ZIPS = False):")
            for rel, _ in inner_zips:
                print(f"  [zip   ] {target}/{rel}")
            print()
        if not files and not (inner_zips and upload_inner):
            print("Zip contains no files to check in.")
            return
        print(f"Unzipped: {len(files)} file(s) from {os.path.basename(zip_path)}")

        # ---- existing files in branch -------------------------------------
        existing = gl.list_files(branch, target) if gl.branch_exists(branch) else set()

        actions, created, updated = [], 0, 0
        for rel_path, abs_path in files:
            repo_path = f"{target}/{rel_path}"
            if repo_path in existing:
                action = "update"
                updated += 1
            else:
                action = "create"
                created += 1
            actions.append(_file_action(action, repo_path, abs_path))
            print(f"  [{action:6}] {repo_path}")

        print(f"\nSummary : {created} new, {updated} replaced -> {len(actions)} total"
              + (f", {len(prohibited)} skipped" if prohibited else ""))

        if dry_run:
            print("Dry run - nothing committed.")
            return

        # ---- commit in chunks ----------------------------------------------
        size = max(1, config.ACTIONS_PER_COMMIT)
        chunks = [actions[i : i + size] for i in range(0, len(actions), size)]
        base_msg = f"{config.COMMIT_MESSAGE}: {os.path.basename(zip_path)}"
        for n, chunk in enumerate(chunks, start=1):
            msg = base_msg if len(chunks) == 1 else f"{base_msg} (part {n}/{len(chunks)})"
            try:
                result = gl.commit(branch, msg, chunk)
            except RuntimeError as e:
                if "prohibited by the pattern" in str(e):
                    raise RuntimeError(
                        "GitLab rejected the commit because a file name is blocked by the project's "
                        "push rule. Add that file's extension to PROHIBITED_EXTENSIONS in config.py "
                        "and re-run.\n" + str(e)
                    ) from None
                raise
            print(f"Commit {n}/{len(chunks)}: {result['short_id']}  ({len(chunk)} file(s))")

        # ---- inner zip files: separate commit, uploaded packed ---------------
        if inner_zips and upload_inner:
            # refresh the file list: the commits above may have created folders
            existing = gl.list_files(branch, target)
            zip_actions = []
            for rel_path, abs_path in inner_zips:
                repo_path = f"{target}/{rel_path}"
                action = "update" if repo_path in existing else "create"
                zip_actions.append(_file_action(action, repo_path, abs_path))
                print(f"  [{action:6}] {repo_path}  (zip, uploaded as-is)")
            try:
                result = gl.commit(branch, f"{base_msg} (inner zip files)", zip_actions)
                print(f"Commit zip : {result['short_id']}  ({len(zip_actions)} zip file(s))")
            except RuntimeError as e:
                if "prohibited by the pattern" in str(e):
                    print(
                        "\nWARNING: GitLab's push rule on this project rejects .zip files, so the inner "
                        "zip file(s) above were NOT uploaded. All other files are already checked in.\n"
                        "Ask the project maintainers to allow .zip in the push rule, or set "
                        "UPLOAD_INNER_ZIPS = False in config.py to silence this.\n" + str(e)
                    )
                else:
                    raise

    web = info.get("web_url", "")
    if web:
        print(f"\nDone. Branch: {web}/-/tree/{quote(branch, safe='/')}/{target}")
    else:
        print("\nDone.")


if __name__ == "__main__":
    zip_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run(zip_path=zip_arg)
