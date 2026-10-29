"""
gitlab_dump.py  (Tool 2)

Three features, each can be run or skipped independently:

  (1) export  - connect to GitLab, read EVERY file (all folders / subfolders) from the
                `main` branch of the ikg / odm / nlg projects and write one txt per
                project: i.txt, o.txt, n.txt
  (2) mask    - mask i.txt / o.txt / n.txt (every letter shifted by MASK_SHIFT from
                config, default 1: a->b ... z->a, A->B ... Z->A, everything else
                unchanged) and write the result into MASKED_DIR split into numbered
                parts each below MAX_MASKED_FILE_BYTES (~5 MB):
                masked/i_1_masked.txt, masked/i_2_masked.txt, ...
  (3) unmask  - read all parts for each project from MASKED_DIR (i_1_masked.txt,
                i1_masked.txt, i-1-masked.txt ... are all accepted), join them in
                numeric order 1,2,3..., unmask and write UNMASKED_DIR/i_restored.txt
                (set restore_to_original=True to overwrite i.txt / o.txt / n.txt instead)
  (4) rebuild - read the restored txt files and recreate every real file, with its
                original name and content, in the original folder structure under
                REBUILD_DIR/<ikg|odm|nlg>/...   e.g. rebuilt/ikg/dags/ikg/test/test.sql
                then zips each project folder to REBUILD_DIR/ikg.zip, odm.zip, nlg.zip
                (ZIP_REBUILT in config)
  (5) verify  - compare i.txt / o.txt / n.txt with i_restored.txt / o_restored.txt /
                n_restored.txt byte by byte, and (if present) every rebuilt file with
                the content in the original txt. Reports the first difference found.

Usage (command line):
    python gitlab_dump.py                       # asks which steps to run
    python gitlab_dump.py export mask           # run only these steps
    python gitlab_dump.py unmask
    python gitlab_dump.py rebuild
    python gitlab_dump.py verify

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
import re
import shutil
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


def _masked_dir():
    return os.path.join(config.OUTPUT_DIR, config.MASKED_DIR)


def _unmasked_dir():
    return os.path.join(config.OUTPUT_DIR, config.UNMASKED_DIR)


def _masked_part_path(prefix, n):
    return os.path.join(_masked_dir(), f"{prefix}_{n}{config.MASKED_SUFFIX}.txt")


def _restored_path(prefix):
    return os.path.join(_unmasked_dir(), f"{prefix}{config.RESTORED_SUFFIX}.txt")


def split_text(text, max_bytes):
    """
    Split `text` into chunks whose UTF-8 size is <= max_bytes, cutting only at
    line ends ("\n") so lines are never broken across parts. A single line longer
    than max_bytes is cut at character boundaries. "".join(chunks) == text.
    """
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    chunks, cur, cur_size = [], [], 0
    pieces = text.split("\n")
    for i, piece in enumerate(pieces):
        line = piece + "\n" if i < len(pieces) - 1 else piece
        if not line:
            continue
        size = len(line.encode(config.ENCODING))
        if size > max_bytes:  # over-long line: flush, then cut it by characters
            if cur:
                chunks.append("".join(cur)); cur, cur_size = [], 0
            start = 0
            while start < len(line):
                end = start
                acc = 0
                while end < len(line):
                    b = len(line[end].encode(config.ENCODING))
                    if acc + b > max_bytes:
                        break
                    acc += b; end += 1
                chunks.append(line[start:end]); start = end
            continue
        if cur_size + size > max_bytes:
            chunks.append("".join(cur)); cur, cur_size = [], 0
        cur.append(line); cur_size += size
    if cur:
        chunks.append("".join(cur))
    return chunks or [""]


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
def _part_pattern(prefix):
    """Regex matching i_1_masked.txt, i1_masked.txt, i-1-masked.txt, I_01_MASKED.TXT ..."""
    suf = re.escape(config.MASKED_SUFFIX.strip("_-"))
    return re.compile(rf"^{re.escape(prefix)}[_-]?(\d+)[_-]?{suf}\.txt$", re.IGNORECASE)


def mask_files(projects=None):
    projects = projects or config.PROJECTS
    out_dir = _masked_dir()
    os.makedirs(out_dir, exist_ok=True)
    limit = int(config.MAX_MASKED_FILE_BYTES)

    for key, (_path, prefix) in projects.items():
        src = _out_path(prefix)
        if not os.path.isfile(src):
            print(f"[{key}] skip - {src} not found")
            continue

        # remove parts from a previous run so stale numbers can't be picked up later
        pat = _part_pattern(prefix)
        for old in os.listdir(out_dir):
            if pat.match(old):
                os.remove(os.path.join(out_dir, old))

        text = _read(src)
        masked = mask_text(text)
        parts = split_text(masked, limit)
        total_bytes = len(masked.encode(config.ENCODING))
        total_lines = masked.count("\n") + (0 if masked.endswith("\n") or not masked else 1)
        print(f"[{key}] {src}: {total_bytes:,} bytes, {total_lines:,} lines -> {len(parts)} part(s) in {out_dir}")
        for n, part in enumerate(parts, start=1):
            dst = _masked_part_path(prefix, n)
            _write(dst, part)
            size = len(part.encode(config.ENCODING))
            lines = part.count("\n") + (0 if part.endswith("\n") or not part else 1)
            assert size <= limit, f"{dst} is {size} bytes > limit {limit}"
            print(f"      {os.path.basename(dst)}: {size:,} bytes ({size / 1024 / 1024:.2f} MB), {lines:,} lines")


# =========================================================================== #
# Feature 3: unmask
# =========================================================================== #
def find_masked_parts(prefix, folder=None):
    """
    Return [(number, path), ...] of all masked parts for `prefix` in MASKED_DIR,
    sorted by number. Also accepts a single un-numbered <prefix>_masked.txt as part 1.
    Raises if numbers are duplicated or have gaps (e.g. 1,2,4).
    """
    folder = folder or _masked_dir()
    if not os.path.isdir(folder):
        return []
    pat = _part_pattern(prefix)
    single = f"{prefix}{config.MASKED_SUFFIX}.txt".lower()
    found = {}
    for name in os.listdir(folder):
        m = pat.match(name)
        n = int(m.group(1)) if m else (1 if name.lower() == single else None)
        if n is None:
            continue
        if n in found:
            raise ValueError(f"Duplicate part number {n} for '{prefix}': {found[n]} and {name}")
        found[n] = name
    if not found:
        return []
    nums = sorted(found)
    expected = list(range(nums[0], nums[-1] + 1))
    if nums != expected:
        missing = sorted(set(expected) - set(nums))
        raise ValueError(f"Missing part number(s) {missing} for '{prefix}' in {folder}; found {nums}")
    if nums[0] not in (0, 1):
        raise ValueError(f"Parts for '{prefix}' start at {nums[0]}, expected 1")
    return [(n, os.path.join(folder, found[n])) for n in nums]


def unmask_files(projects=None, restore_to_original=False):
    projects = projects or config.PROJECTS
    os.makedirs(_unmasked_dir(), exist_ok=True)
    for key, (_path, prefix) in projects.items():
        parts = find_masked_parts(prefix)
        if not parts:
            print(f"[{key}] skip - no {prefix}_<n>{config.MASKED_SUFFIX}.txt files in {_masked_dir()}")
            continue
        joined = "".join(_read(path) for _n, path in parts)
        dst = _out_path(prefix) if restore_to_original else _restored_path(prefix)
        _write(dst, unmask_text(joined))
        names = ", ".join(os.path.basename(path) for _n, path in parts)
        print(f"[{key}] {len(parts)} part(s) [{names}] -> {dst}")


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
    Writes REBUILD_DIR/<ikg|odm|nlg>/<path inside repo>, creating folders as needed,
    and REBUILD_DIR/<ikg|odm|nlg>.zip next to each folder:
        rebuilt/
          ikg/        <- everything from ikg-dags (dags/..., README.md, ...)
          ikg.zip
          odm/  odm.zip
          nlg/  nlg.zip
    The leading project folder in the dump path (e.g. "ikg-dags/") is replaced by the
    project key folder (ikg/, odm/, nlg/).
    """
    projects = projects or config.PROJECTS
    root = os.path.join(config.OUTPUT_DIR, config.REBUILD_DIR)

    for key, (_path, prefix) in projects.items():
        src = _restored_path(prefix) if source == "restored" else _out_path(prefix)
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

        if getattr(config, "ZIP_REBUILT", True) and written:
            zip_path = _zip_folder(os.path.join(root, key), os.path.join(root, key))
            print(f"      zipped -> {zip_path}")


def _zip_folder(folder, zip_base):
    """Zip `folder` to `zip_base`.zip; the zip contains the folder itself as top level
    (e.g. ikg.zip -> ikg/dags/ikg/...). An existing zip is replaced."""
    if os.path.exists(zip_base + ".zip"):
        os.remove(zip_base + ".zip")
    return shutil.make_archive(
        zip_base, "zip",
        root_dir=os.path.dirname(os.path.abspath(folder)),
        base_dir=os.path.basename(folder),
    )


# =========================================================================== #
# Feature 5: verify original vs restored (and rebuilt)
# =========================================================================== #
def _first_diff(a, b):
    """Describe the first difference between two byte strings (line number + position)."""
    n = min(len(a), len(b))
    i = next((k for k in range(n) if a[k] != b[k]), n)
    if i == n and len(a) == len(b):
        return None
    line = a[:i].count(b"\n") + 1
    col = i - (a[:i].rfind(b"\n") + 1) + 1
    if i == n:
        return f"line {line}: one side is longer ({len(a)} vs {len(b)} bytes)"
    return f"line {line}, column {col}: {a[i:i+20]!r} vs {b[i:i+20]!r}"


def _read_bytes(path):
    with open(path, "rb") as fh:
        return fh.read()


def verify_files(projects=None):
    """
    1. <prefix>.txt  vs  <prefix>_restored.txt   - must be byte-identical
    2. every file under REBUILD_DIR/<key>/      vs its content in <prefix>.txt - must be byte-identical
    Returns True when everything matches.
    """
    projects = projects or config.PROJECTS
    root = os.path.join(config.OUTPUT_DIR, config.REBUILD_DIR)
    all_ok = True

    for key, (_path, prefix) in projects.items():
        orig = _out_path(prefix)
        rest = _restored_path(prefix)
        if not os.path.isfile(orig):
            print(f"[{key}] skip - {orig} not found")
            continue
        orig_bytes = _read_bytes(orig)

        # 1. txt vs restored txt
        if os.path.isfile(rest):
            diff = _first_diff(orig_bytes, _read_bytes(rest))
            if diff:
                all_ok = False
                print(f"[{key}] MISMATCH  {os.path.basename(orig)} vs {os.path.basename(rest)}: {diff}")
            else:
                print(f"[{key}] OK        {os.path.basename(orig)} == {os.path.basename(rest)}  ({len(orig_bytes):,} bytes)")
        else:
            print(f"[{key}] skip      {rest} not found")

        # 2. rebuilt files vs content inside the original txt
        rebuilt_dir = os.path.join(root, key)
        if not os.path.isdir(rebuilt_dir):
            continue
        entries = parse_dump(orig_bytes.decode(config.ENCODING))
        checked = mismatched = missing = 0
        for name, full_path, code in entries:
            if code.startswith(BINARY_PLACEHOLDER):
                continue
            rel = full_path.split("/", 1)[1] if "/" in full_path else full_path
            dest = os.path.join(rebuilt_dir, *rel.split("/"))
            if not os.path.isfile(dest):
                missing += 1
                all_ok = False
                print(f"[{key}]   MISSING  {rel}")
                continue
            checked += 1
            diff = _first_diff(code.encode(config.ENCODING), _read_bytes(dest))
            if diff:
                mismatched += 1
                all_ok = False
                print(f"[{key}]   MISMATCH {rel}: {diff}")
        status = "OK" if not (mismatched or missing) else "FAILED"
        print(f"[{key}] {status:9} rebuilt files: {checked} compared, {mismatched} mismatched, {missing} missing")

    print("\nVERIFY RESULT:", "all files identical" if all_ok else "DIFFERENCES FOUND (see above)")
    return all_ok


# =========================================================================== #
# Runner
# =========================================================================== #
STEPS = ("export", "mask", "unmask", "rebuild", "verify")


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
    steps: list of any of "export", "mask", "unmask", "rebuild", "verify"
           (order is fixed: export -> mask -> unmask -> rebuild -> verify). None = ask interactively.
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
    if "verify" in steps:
        print("\n=== (5) verify original vs restored ===")
        verify_files()
    print("\nDone.")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a.lower() in STEPS] or None
    run(steps=args)
