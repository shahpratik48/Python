"""
Configuration for gitlab_dump (Tool 2).

Secrets (GitLab token) are asked from the user at run time, never stored here.
"""

GITLAB_URL = "https://devcloud.ubs.net"

_BASE = "ubs/gwma/smart-technology-and-analytics/staat-data-science/staat-ds-genesis/genesis-platform/"

IKG_PROJECT_PATH = _BASE + "ikg-dags"
NLG_PROJECT_PATH = _BASE + "nlg-dags"
ODM_PROJECT_PATH = _BASE + "odm-dags"

# Branch to read from in every project
BRANCH = "main"

# Projects to export: key -> (project path, output file prefix)
# Output files: <prefix>.txt, <prefix>_masked.txt, <prefix>_restored.txt
PROJECTS = {
    "ikg": (IKG_PROJECT_PATH, "i"),
    "odm": (ODM_PROJECT_PATH, "o"),
    "nlg": (NLG_PROJECT_PATH, "n"),
}

# Folder where the txt files are written / read. "." = same folder as the script / notebook.
OUTPUT_DIR = "."

# Mask shift: each letter is replaced by the letter this many positions later in the
# alphabet, wrapping around and keeping case. 1 -> a->b, z->a, A->B, Z->A.
# Unmask uses the same value in reverse. Must be 1..25 (0 would mean no masking).
MASK_SHIFT = 1

# Suffixes for the masked and restored files
MASKED_SUFFIX = "_masked"
RESTORED_SUFFIX = "_restored"

# Step 4 (rebuild): folder where the real files are recreated from the restored txt files.
# Inside it one folder per project is created: ikg/, odm/, nlg/ (the PROJECTS keys),
# and under each the original repo folder structure, e.g. rebuilt/ikg/dags/ikg/test/test.sql
REBUILD_DIR = "rebuilt"

# After rebuild, zip each project folder to REBUILD_DIR/ikg.zip, odm.zip, nlg.zip (next to the folders).
ZIP_REBUILT = True

# Separator line between files in the txt output
SEPARATOR = "-" * 127

# SSL verification for the GitLab API: True, False, or path to an internal CA bundle
VERIFY_SSL = True

# Text encoding used for all txt files
ENCODING = "utf-8"
