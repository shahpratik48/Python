"""
gitlab_dump.py  (Tool 2)

Three features, each can be run or skipped independently:

  (1) export  - connect to GitLab, read EVERY file (all folders / subfolders) from the
                `main` branch of the ikg / odm / nlg projects and write one txt per
                project: i.txt, o.txt, n.txt
  (2) mask    - create i_masked.txt, o_masked.txt, n_masked.txt from the txt files
                (every letter shifted by MASK_SHIFT from config, default 1:
                a->b ... z->a, A->B ... Z->A, everything else unchanged)
  (3) unmask  - read i_masked.txt / o_masked.txt / n_masked.txt from the same folder
                and restore the originals as i_restored.txt / o_restored.txt / n_restored.txt
                (set restore_to_original=True to overwrite i.txt / o.txt / n.txt instead)
  (4) rebuild - read the restored txt files and recreate every real file, with its
                original name and content, in the original folder structure under
                REBUILD_DIR/<ikg|odm|nlg>/...   e.g. rebuilt/ikg/dags/ikg/test/test.sql

Usage (command line):
    python gitlab_dump.py                       # asks which steps to run
    python gitlab_dump.py export mask           # run only these steps
    python gitlab_dump.py unmask
    python gitlab_dump.py rebuild

Usage (notebook / import):
    from gitlab_dump import run
    run()                                       # asks which steps to run
    run(steps=["export", "mask"])
    run(steps=["unmask"], restore_to_original=True)
    run(steps=["rebuild"])                      # from *_restored.txt
    run(steps=["rebuild"], rebuild_source="original")   # from i.txt / o.txt / n.txt

Dependencies: requests   (pip install requests)
"""

import getpass
import os
import sys
from urllib.parse import quote

import requests

import config


# =========================================================================== #
# GitLab client
# =========================================================================== #
class GitLab:
    def __init__(self, url, token, verify=True):
        self.base = url.rstrip("/") + "/api/v4"
        self.session = requests.Session()
        self.session.headers.update({"PRIVATE-TOKEN": token})
        self.session.verify = verify

    def _get(self, url, **params):
        r = self.session.get(url, params=params)
        if r.status_code != 200:
            raise RuntimeError(f"GitLab API error {r.status_code} on {r.url}\n{r.text[:500]}")
        return r

    def project(self, project_path):
        pid = quote(project_path, safe="")
        return self._get(f"{self.base}/projects/{pid}").json()

    def list_all_files(self, project_path, ref):
        """All blob paths in the branch, every folder and subfolder, with pagination."""
        pid = quote(project_path, safe="")
        paths, page = [], 1
        while True:
            r = self._get(
                f"{self.base}/projects/{pid}/repository/tree",
                ref=ref, recursive="true", per_page=100, page=page,
            )
            for item in r.json():
                if item["type"] == "blob":
                    paths.append(item["path"])
            nxt = r.headers.get("X-Next-Page")
            if not nxt:
                break
            page = int(nxt)
        return sorted(paths)

    def file_bytes(self, project_path, file_path, ref):
        pid = quote(project_path, safe="")
        fp = quote(file_path, safe="")
        return self._get(f"{self.base}/projects/{pid}/repository/files/{fp}/raw", ref=ref).content


# =========================================================================== #
# Helpers
# =========================================================================== #
def _out_path(prefix, suffix=""):
    return os.path.join(config.OUTPUT_DIR, f"{prefix}{suffix}.txt")


def _decode(raw):
    """Return file text, or a placeholder if the file is binary."""
    try:
        text = raw.decode("utf-8")
        if "\x00" in text:
            raise UnicodeDecodeError("utf-8", b"", 0, 1, "binary")
        return text
    except UnicodeDecodeError:
        return f"<<binary file, {len(raw)} bytes - content not included>>\n"


def _entry(file_name, full_path, code):
    """
    One block of the txt output in the required format.
    `code` is written EXACTLY as read from GitLab: nothing is added, stripped,
    re-indented, re-cased or re-encoded. Line endings (LF / CRLF), tabs, trailing
    spaces, a missing final newline and non-ASCII characters are all kept.
    The single "\n" before the separator belongs to the format, not to the file,
    and parse_dump removes exactly that one again.
    """
    return (
        f"****{file_name}******\n"
        f"****{full_path}******\n"
        f"\n"
        f"{code}"
        f"\n"
        f"{config.SEPARATOR}\n"
        f"\n"
    )


def shift_text(text, shift):
    """Shift every ASCII letter by `shift` positions (wrapping); keep case; leave everything else."""
    out = []
    for ch in text:
        if "a" <= ch <= "z":
            out.append(chr((ord(ch) - 97 + shift) % 26 + 97))
        elif "A" <= ch <= "Z":
            out.append(chr((ord(ch) - 65 + shift) % 26 + 65))
        else:
            out.append(ch)
    return "".join(out)


def _mask_shift():
    shift = int(config.MASK_SHIFT)
    if not 1 <= shift <= 25:
        raise ValueError(f"config.MASK_SHIFT must be between 1 and 25, got {shift}")
    return shift


def mask_text(text):
    return shift_text(text, _mask_shift())


def unmask_text(text):
    return shift_text(text, -_mask_shift())


def _read(path):
    with open(path, "r", encoding=config.ENCODING, newline="") as fh:
        return fh.read()


def _write(path, text):
    with open(path, "w", encoding=config.ENCODING, newline="") as fh:
        fh.write(text)


# =========================================================================== #
# Feature 1: export
# =========================================================================== #
def export_projects(token=None, projects=None):
    if not token:
        token = getpass.getpass("GitLab token (input hidden): ").strip()
    if not token:
        raise ValueError("A GitLab token is required for export.")
    projects = projects or config.PROJECTS
    gl = GitLab(config.GITLAB_URL, token, verify=config.VERIFY_SSL)

    for key, (project_path, prefix) in projects.items():
        info = gl.project(project_path)
        project_name = info["path"]  # e.g. "ikg-dags"
        print(f"[{key}] {info['path_with_namespace']} @ {config.BRANCH}")

        paths = gl.list_all_files(project_path, config.BRANCH)
        print(f"      {len(paths)} file(s) found")

        blocks = []
        for i, p in enumerate(paths, start=1):
            code = _decode(gl.file_bytes(project_path, p, config.BRANCH))
            blocks.append(_entry(os.path.basename(p), f"{project_name}/{p}", code))
            if i % 50 == 0 or i == len(paths):
                print(f"      fetched {i}/{len(paths)}")

        out = _out_path(prefix)
        _write(out, "".join(blocks))
        print(f"      -> {out}\n")


# =========================================================================== #
# Feature 2: mask
# =========================================================================== #
def mask_files(projects=None):
    projects = projects or config.PROJECTS
    for key, (_path, prefix) in projects.items():
        src = _out_path(prefix)
        dst = _out_path(prefix, config.MASKED_SUFFIX)
        if not os.path.isfile(src):
            print(f"[{key}] skip - {src} not found")
            continue
        _write(dst, mask_text(_read(src)))
        print(f"[{key}] {src} -> {dst}")


# =========================================================================== #
# Feature 3: unmask
# =========================================================================== #
def unmask_files(projects=None, restore_to_original=False):
    projects = projects or config.PROJECTS
    for key, (_path, prefix) in projects.items():
        src = _out_path(prefix, config.MASKED_SUFFIX)
        dst = _out_path(prefix, "" if restore_to_original else config.RESTORED_SUFFIX)
        if not os.path.isfile(src):
            print(f"[{key}] skip - {src} not found")
            continue
        _write(dst, unmask_text(_read(src)))
        print(f"[{key}] {src} -> {dst}")


# =========================================================================== #
# Feature 4: rebuild real files from the txt dump
# =========================================================================== #
BINARY_PLACEHOLDER = "<<binary file, "


def parse_dump(text):
    """
    Parse a txt dump produced by export (or restored by unmask) back into
    a list of (file_name, full_path, code). The code is returned unchanged.
    """
    entries = []
    token = "\n" + config.SEPARATOR + "\n\n"
    header = "****"

    def _is_block_start(piece):
        """True only for a genuine '****name******' / '****path******' header pair
        (the path must end with the name), so file content that happens to contain
        separator lines or ****-lines is never mistaken for a new block."""
        lines = piece.split("\n", 3)
        if len(lines) < 3 or lines[2] != "":
            return False
        a, b = lines[0], lines[1]
        if not (a.startswith(header) and a.endswith("******") and b.startswith(header) and b.endswith("******")):
            return False
        name, path = a[4:-6], b[4:-6]
        return bool(name) and (path == name or path.endswith("/" + name))

    if text.endswith(token):
        text = text[: -len(token)]  # every block ends with exactly one separator
    pieces = text.split(token)
    # A separator inside a file's own content splits that file in two; such a
    # piece is not a real block start, so glue it back together with the
    # separator that the split removed.
    merged = []
    for piece in pieces:
        if merged and not _is_block_start(piece):
            merged[-1] += token + piece
        else:
            merged.append(piece)
    for block in merged:
        if block == "":
            continue
        lines = block.split("\n", 3)
        if len(lines) < 3 or not (lines[0].startswith(header) and lines[1].startswith(header)):
            raise ValueError(f"Unexpected block in dump (first line: {lines[0][:80]!r})")
        if lines[2] != "":
            raise ValueError(f"Expected blank line after headers for {lines[0]!r}")
        name = lines[0][4:-6] if lines[0].endswith("******") else lines[0].strip("*")
        path = lines[1][4:-6] if lines[1].endswith("******") else lines[1].strip("*")
        code = lines[3] if len(lines) == 4 else ""
        entries.append((name, path, code))
    return entries


def rebuild_files(projects=None, source="restored"):
    """
    source: "restored" -> read i_restored.txt / o_restored.txt / n_restored.txt
            "original" -> read i.txt / o.txt / n.txt
    Writes REBUILD_DIR/<ikg|odm|nlg>/<path inside repo>, creating folders as needed.
    The leading project folder in the dump path (e.g. "ikg-dags/") is replaced by the
    project key folder (ikg/, odm/, nlg/).
    """
    projects = projects or config.PROJECTS
    suffix = config.RESTORED_SUFFIX if source == "restored" else ""
    root = os.path.join(config.OUTPUT_DIR, config.REBUILD_DIR)

    for key, (_path, prefix) in projects.items():
        src = _out_path(prefix, suffix)
        if not os.path.isfile(src):
            print(f"[{key}] skip - {src} not found")
            continue
        entries = parse_dump(_read(src))
        written = skipped = 0
        for name, full_path, code in entries:
            rel = full_path.split("/", 1)[1] if "/" in full_path else full_path
            rel = rel.replace("\\", "/").lstrip("/")
            if ".." in rel.split("/"):
                raise ValueError(f"Refusing unsafe path in dump: {full_path}")
            dest = os.path.join(root, key, *rel.split("/"))
            if code.startswith(BINARY_PLACEHOLDER):
                skipped += 1
                continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            _write(dest, code)  # exact content, no newline translation (newline="")
            written += 1
        msg = f"[{key}] {src} -> {os.path.join(root, key)}  ({written} file(s) written"
        msg += f", {skipped} binary placeholder(s) skipped)" if skipped else ")"
        print(msg)


# =========================================================================== #
# Runner
# =========================================================================== #
STEPS = ("export", "mask", "unmask", "rebuild")


def _ask_steps():
    chosen = []
    print("Select steps to run (y/n):")
    for s in STEPS:
        ans = input(f"  {s:7}? [y/N] ").strip().lower()
        if ans in ("y", "yes"):
            chosen.append(s)
    return chosen


def run(steps=None, token=None, restore_to_original=False, rebuild_source="restored"):
    """
    steps: list of any of "export", "mask", "unmask", "rebuild"
           (order is fixed: export -> mask -> unmask -> rebuild). None = ask interactively.
    token: GitLab token (only needed for "export"; asked if missing).
    restore_to_original: unmask writes i.txt/o.txt/n.txt instead of i_restored.txt/...
    rebuild_source: "restored" (default) rebuilds from *_restored.txt, "original" from i/o/n.txt
    """
    if steps is None:
        steps = _ask_steps()
    steps = [s.lower() for s in steps]
    bad = [s for s in steps if s not in STEPS]
    if bad:
        raise ValueError(f"Unknown step(s) {bad}; valid: {list(STEPS)}")
    if not steps:
        print("Nothing selected - exiting.")
        return

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    if "export" in steps:
        print("\n=== (1) export from GitLab ===")
        export_projects(token=token)
    if "mask" in steps:
        print(f"\n=== (2) mask (shift={_mask_shift()}) ===")
        mask_files()
    if "unmask" in steps:
        print(f"\n=== (3) unmask (shift={_mask_shift()}) ===")
        unmask_files(restore_to_original=restore_to_original)
    if "rebuild" in steps:
        print("\n=== (4) rebuild files from txt ===")
        if restore_to_original and rebuild_source == "restored":
            rebuild_source = "original"   # unmask wrote into i/o/n.txt, so read those
        rebuild_files(source=rebuild_source)
    print("\nDone.")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a.lower() in STEPS] or None
    run(steps=args)
