"""
Loads source files either from a local directory (offline / notebook-with-
checkout mode) or directly from a GitLab project via the REST API (no git
clone needed -- works well from inside an Airflow worker).

Performance notes (this is the module that made "build_hierarchy" slow):
  - GitLabSourceRepo used to fetch every single file with its own HTTP
    request (~2,000+ round-trips for nlg-src alone). It now downloads the
    whole subtree in ONE request via GitLab's `repository/archive`
    endpoint and reads everything from the in-memory tarball. That's a
    ~1000x drop in HTTP calls.
  - If the archive endpoint isn't reachable (older GitLab, permissions,
    proxy blocking non-API endpoints, etc.) it falls back to the old
    tree+raw-file approach, but now fetches files in parallel
    (ThreadPoolExecutor) instead of one at a time.
  - An optional on-disk cache (`cache_dir`) avoids re-downloading anything
    at all on repeat runs against the same ref, until the ref's commit SHA
    changes.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import posixpath
import tarfile
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class SourceFile:
    path: str
    content: str


class SourceRepo:
    def list_files(self, extensions: Optional[List[str]] = None) -> List[str]:
        raise NotImplementedError

    def read_file(self, path: str) -> str:
        raise NotImplementedError

    def read_all(self, extensions: Optional[List[str]] = None) -> Dict[str, str]:
        """Default (slow) path: read one file at a time. Subclasses override
        this with a bulk/parallel implementation when they can do better."""
        return {p: self.read_file(p) for p in self.list_files(extensions)}


class LocalSourceRepo(SourceRepo):
    """Reads from a directory already present on disk (e.g. an extracted
    GitLab archive or a `git clone`). Disk I/O is already fast; read_all()
    still parallelizes it for very large trees (network-mounted dirs etc.)."""

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self._files: Optional[List[str]] = None

    def _walk(self) -> List[str]:
        if self._files is not None:
            return self._files
        out = []
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__", ".ipynb_checkpoints")]
            for fn in filenames:
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, self.root_dir).replace(os.sep, "/")
                out.append(rel)
        self._files = sorted(out)
        return self._files

    def list_files(self, extensions: Optional[List[str]] = None) -> List[str]:
        files = self._walk()
        if extensions:
            exts = tuple(extensions)
            files = [f for f in files if f.endswith(exts)]
        return files

    def read_file(self, path: str) -> str:
        full = os.path.join(self.root_dir, path)
        with open(full, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()

    def read_all(self, extensions: Optional[List[str]] = None) -> Dict[str, str]:
        files = self.list_files(extensions)
        out: Dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=32) as ex:
            futs = {ex.submit(self.read_file, p): p for p in files}
            for fut in as_completed(futs):
                out[futs[fut]] = fut.result()
        return out


class GitLabSourceRepo(SourceRepo):
    """Reads a subtree of a GitLab project via the REST API using a
    personal / project access token. No local clone required.

    Primary strategy: one `repository/archive` download for the whole
    subtree. Falls back to tree-listing + parallel raw-file GETs only if
    the archive endpoint fails.
    """

    def __init__(self, base_url: str, project_path: str, token: str,
                 ref: str = "develop", subpath: str = "",
                 cache_dir: Optional[str] = None, max_workers: int = 24):
        import requests  # local import so the package works w/o `requests` in pure-local mode
        self._requests = requests
        self.base_url = base_url.rstrip("/")
        self.project_path = project_path
        self.token = token
        self.ref = ref
        self.subpath = subpath.strip("/")
        self.cache_dir = cache_dir
        self.max_workers = max_workers
        self._project_id: Optional[int] = None
        self._files: Optional[List[str]] = None
        self._contents: Optional[Dict[str, str]] = None
        self._path_map: Dict[str, str] = {}

    def _headers(self):
        return {"PRIVATE-TOKEN": self.token}

    def _session(self):
        s = self._requests.Session()
        s.headers.update(self._headers())
        return s

    def _project_id_lookup(self) -> int:
        if self._project_id is not None:
            return self._project_id
        encoded = urllib.parse.quote(self.project_path, safe="")
        url = f"{self.base_url}/api/v4/projects/{encoded}"
        r = self._requests.get(url, headers=self._headers(), timeout=30)
        r.raise_for_status()
        self._project_id = r.json()["id"]
        return self._project_id

    # ---------------------------------------------------------- caching
    def _cache_key(self) -> Optional[str]:
        if not self.cache_dir:
            return None
        raw = f"{self.project_path}::{self.ref}::{self.subpath}"
        return hashlib.sha1(raw.encode()).hexdigest()

    def _cache_paths(self):
        key = self._cache_key()
        if not key:
            return None, None
        os.makedirs(self.cache_dir, exist_ok=True)
        return (os.path.join(self.cache_dir, f"{key}.meta.json"),
                os.path.join(self.cache_dir, f"{key}.contents.json"))

    def _load_cache(self) -> bool:
        meta_path, contents_path = self._cache_paths()
        if not meta_path or not os.path.exists(meta_path) or not os.path.exists(contents_path):
            return False
        try:
            with open(meta_path) as fh:
                meta = json.load(fh)
            sha = self._current_commit_sha()
            if meta.get("commit_sha") != sha:
                return False  # stale -- ref has moved, must re-download
            with open(contents_path) as fh:
                self._contents = json.load(fh)
            self._files = sorted(self._contents.keys())
            return True
        except Exception:
            return False

    def _save_cache(self):
        meta_path, contents_path = self._cache_paths()
        if not meta_path:
            return
        try:
            with open(contents_path, "w") as fh:
                json.dump(self._contents, fh)
            with open(meta_path, "w") as fh:
                json.dump({"commit_sha": self._current_commit_sha()}, fh)
        except Exception:
            pass  # cache is a pure optimization; never fail the run over it

    def _current_commit_sha(self) -> Optional[str]:
        try:
            pid = self._project_id_lookup()
            url = f"{self.base_url}/api/v4/projects/{pid}/repository/commits/{self.ref}"
            r = self._requests.get(url, headers=self._headers(), timeout=20)
            r.raise_for_status()
            return r.json().get("id")
        except Exception:
            return None

    # ---------------------------------------------------------- bulk download
    def _download_archive(self) -> bool:
        """Single-request bulk download via GitLab's tarball archive
        endpoint. Returns True on success."""
        try:
            pid = self._project_id_lookup()
            url = f"{self.base_url}/api/v4/projects/{pid}/repository/archive.tar.gz"
            params = {"sha": self.ref}
            if self.subpath:
                params["path"] = self.subpath  # supported on modern GitLab; harmless if ignored
            r = self._requests.get(url, headers=self._headers(), params=params, timeout=180, stream=True)
            r.raise_for_status()
            buf = io.BytesIO(r.content)
            contents: Dict[str, str] = {}
            with tarfile.open(fileobj=buf, mode="r:gz") as tar:
                for member in tar.getmembers():
                    if not member.isfile():
                        continue
                    # member.name looks like "<project>-<ref>-<sha>/<subpath>/rest/of/path"
                    parts = member.name.split("/", 1)
                    inner = parts[1] if len(parts) > 1 else parts[0]
                    if self.subpath and inner.startswith(self.subpath + "/"):
                        rel = inner[len(self.subpath) + 1:]
                    elif self.subpath:
                        continue  # `path` param wasn't honored server-side; skip anything outside our subtree
                    else:
                        rel = inner
                    if not rel:
                        continue
                    fh = tar.extractfile(member)
                    if fh is None:
                        continue
                    try:
                        contents[rel] = fh.read().decode("utf-8", errors="replace")
                    finally:
                        fh.close()
            if not contents:
                return False
            self._contents = contents
            self._files = sorted(contents.keys())
            return True
        except Exception:
            return False

    # ---------------------------------------------------------- fallback (parallel)
    def _walk_tree(self) -> List[str]:
        pid = self._project_id_lookup()
        files: List[str] = []
        page = 1
        while True:
            url = f"{self.base_url}/api/v4/projects/{pid}/repository/tree"
            params = {"path": self.subpath, "ref": self.ref, "recursive": "true",
                      "per_page": 100, "page": page}
            r = self._requests.get(url, headers=self._headers(), params=params, timeout=60)
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            for item in batch:
                if item.get("type") == "blob":
                    files.append(item["path"])
            if len(batch) < 100:
                break
            page += 1
        rel = [posixpath.relpath(f, self.subpath) if self.subpath else f for f in files]
        self._path_map = dict(zip(rel, files))
        return sorted(rel)

    def _fetch_one_raw(self, session, full_path: str) -> str:
        pid = self._project_id_lookup()
        encoded = urllib.parse.quote(full_path, safe="")
        url = f"{self.base_url}/api/v4/projects/{pid}/repository/files/{encoded}/raw"
        r = session.get(url, params={"ref": self.ref}, timeout=30)
        r.raise_for_status()
        return r.text

    def _download_parallel(self):
        files = self._walk_tree()
        self._files = files
        session = self._session()
        contents: Dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futs = {}
            for rel in files:
                full_path = self._path_map.get(rel, rel)
                futs[ex.submit(self._fetch_one_raw, session, full_path)] = rel
            for fut in as_completed(futs):
                rel = futs[fut]
                try:
                    contents[rel] = fut.result()
                except Exception as e:
                    contents[rel] = f"# ERROR fetching {rel}: {e}"
        self._contents = contents

    # ---------------------------------------------------------- public API
    def _ensure_loaded(self):
        if self._contents is not None:
            return
        if self._load_cache():
            return
        if self._download_archive():
            self._save_cache()
            return
        self._download_parallel()
        self._save_cache()

    def list_files(self, extensions: Optional[List[str]] = None) -> List[str]:
        self._ensure_loaded()
        files = self._files or []
        if extensions:
            exts = tuple(extensions)
            files = [f for f in files if f.endswith(exts)]
        return files

    def read_file(self, path: str) -> str:
        self._ensure_loaded()
        return self._contents[path]

    def read_all(self, extensions: Optional[List[str]] = None) -> Dict[str, str]:
        self._ensure_loaded()
        files = self.list_files(extensions)
        return {p: self._contents[p] for p in files}


def build_repos(settings) -> Dict[str, SourceRepo]:
    """Returns {'dags': SourceRepo, 'src': SourceRepo} using local dirs if
    configured, otherwise GitLab."""
    if settings.local_dags_dir and settings.local_src_dir:
        return {
            "dags": LocalSourceRepo(settings.local_dags_dir),
            "src": LocalSourceRepo(settings.local_src_dir),
        }
    if not settings.gitlab_token:
        raise RuntimeError(
            "No local_dags_dir/local_src_dir configured and no GitLab token available. "
            "Call settings.resolve_secrets() first, or set NLG_LOCAL_DAGS_DIR / NLG_LOCAL_SRC_DIR."
        )
    cache_dir = getattr(settings, "gitlab_cache_dir", None)
    return {
        "dags": GitLabSourceRepo(settings.gitlab_url, settings.nlg_project_path, settings.gitlab_token,
                                  ref=settings.default_base_branch, subpath=settings.nlg_dags_subpath,
                                  cache_dir=cache_dir),
        "src": GitLabSourceRepo(settings.gitlab_url, settings.nlg_project_path, settings.gitlab_token,
                                 ref=settings.default_base_branch, subpath=settings.nlg_src_subpath,
                                 cache_dir=cache_dir),
    }
