# gitlab_zip_checkin

Unzips a zip file and checks in all folders/subfolders/files into a GitLab branch
under `dags/ikg`, keeping the zip's folder structure.

## Files
- `config.py` – GitLab URL, project path, base branch (`ikg-master`), target folder (`dags/ikg`), SSL, ignore patterns
- `gitlab_zip_checkin.py` – the script (run from command line or import `run()`)
- `gitlab_zip_checkin.ipynb` – same code as a notebook (config + code + run cell)

## Run
    pip install requests
    python gitlab_zip_checkin.py c:\folder\test.zip
You will be prompted for your GitLab token (hidden) and the branch name.

## Behaviour
- Branch missing -> created from `ikg-master`; branch present -> commits on top of it
- File already in branch -> replaced; not in branch -> created; nothing deleted
- Text files sent as UTF-8, binary files as base64
- `__MACOSX/`, `.DS_Store`, `Thumbs.db` skipped
- Commits are chunked (100 files per commit by default)
- `run(..., dry_run=True)` shows the create/update plan without committing

## Notes
- Token needs `api` scope (or `write_repository`) with Developer role on the project.
- Behind an internal CA, set `VERIFY_SSL` in `config.py` to the CA bundle path.
