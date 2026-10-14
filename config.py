"""
Configuration for gitlab_zip_checkin.

Everything that is NOT secret or run-specific lives here.
The GitLab token, the zip path and the target branch are asked from the user at run time.
"""

# GitLab server
GITLAB_URL = "https://devcloud.ubs.net"

# Project path (namespace/.../project) - it is URL-encoded automatically
PROJECT_PATH = (
    "ubs/gwma/smart-technology-and-analytics/staat-data-science/"
    "staat-ds-genesis/genesis-platform/ikg-dags"
)

# Branch the new branch is created from (only used when the target branch does not exist yet)
BASE_BRANCH = "ikg-master"

# Folder in the repo under which the unzipped content is checked in
TARGET_FOLDER = "dags/ikg"

# Commit message used for the check-in (the zip name is appended)
COMMIT_MESSAGE = "Check-in from zip"

# How many file actions to send per commit (GitLab limits request size; 100 is safe)
ACTIONS_PER_COMMIT = 100

# SSL verification for the GitLab API. Set to a CA bundle path (str) if you use
# an internal CA, or False to disable verification (not recommended).
VERIFY_SSL = True

# Files / folders inside the zip that should never be checked in
IGNORE_PATTERNS = ["__MACOSX/", ".DS_Store", "Thumbs.db"]
