"""
Indexes the `nlg-src` (a.k.a. `nlg/src`) library so tasks discovered in the
DAG files can be resolved down to an actual method + file, and so
`target_type` values can be mapped to their config/rule/layout resource
files (yaml/sql/ini).
"""
from __future__ import annotations

import ast
import os
import posixpath
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .model import ResourceRef


@dataclass
class ClassInfo:
    name: str
    file: str
    methods: Set[str] = field(default_factory=set)


@dataclass
class FuncInfo:
    name: str
    file: str
    lineno: int
    calls: Set[str] = field(default_factory=set)   # names of functions/methods this function calls (best-effort)


class SrcIndex:
    def __init__(self, src_files: Dict[str, str]):
        """src_files: {relative_path: content} for everything under nlg-src (py/yml/yaml/sql/ini)."""
        self.raw_files = src_files
        self.classes: Dict[str, ClassInfo] = {}
        self.functions: Dict[str, FuncInfo] = {}
        self.module_by_dotted: Dict[str, str] = {}   # "nlg.src.foo.bar" -> file path
        self._resource_index: Dict[str, List[ResourceRef]] = {}
        self._build_python_index()
        self._build_module_path_index()
        self._build_resource_index()

    # ------------------------------------------------------------ python
    def _build_python_index(self):
        for path, content in self.raw_files.items():
            if not path.endswith(".py"):
                continue
            try:
                tree = ast.parse(content, filename=path)
            except SyntaxError:
                continue
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    ci = self.classes.setdefault(node.name, ClassInfo(name=node.name, file=path))
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            ci.methods.add(item.name)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    calls = set()
                    for sub in ast.walk(node):
                        if isinstance(sub, ast.Call):
                            f = sub.func
                            if isinstance(f, ast.Name):
                                calls.add(f.id)
                            elif isinstance(f, ast.Attribute):
                                calls.add(f.attr)
                    self.functions.setdefault(node.name, FuncInfo(name=node.name, file=path,
                                                                    lineno=node.lineno, calls=calls))

    def _build_module_path_index(self):
        # nlg-src root corresponds to "nlg.src" package; "foo/bar.py" -> "nlg.src.foo.bar"
        for path in self.raw_files:
            if not path.endswith(".py"):
                continue
            dotted = "nlg.src." + path[:-3].replace("/", ".")
            self.module_by_dotted[dotted] = path
        self.module_by_dotted_norm = {k.replace("nlg.src.src.", "nlg.src."): v
                                       for k, v in self.module_by_dotted.items()}

    def _build_resource_index(self):
        """One pass over every file, building {key -> [ResourceRef]} so that
        resources_for_target_type() becomes O(1) dict lookups per
        comma-separated target_type instead of an O(files) scan *per task*
        (this was the main hot spot on large repos -- tasks * files
        string comparisons)."""
        for path in self.raw_files:
            if not path.endswith((".yml", ".yaml", ".sql", ".ini")):
                continue
            base = posixpath.basename(path)
            parent_dir = posixpath.basename(posixpath.dirname(path))
            stem = base.rsplit(".", 1)[0]

            if path.endswith((".yml", ".yaml")):
                tag = "insight_type" if base.startswith("si_") else "input_data_config"
                ref = ResourceRef(kind="yaml", path=path, tag=tag)
            elif path.endswith(".sql"):
                ref = ResourceRef(kind="sql", path=path, tag="sql")
            else:
                ref = ResourceRef(kind="ini", path=path, tag="ini")

            keys = {parent_dir, stem}
            if stem.startswith("si_"):
                keys.add(stem[len("si_"):])
            for key in keys:
                self._resource_index.setdefault(key, []).append(ref)

    # ------------------------------------------------------------ lookups
    def find_class_method_file(self, class_name: str, method_name: str) -> Optional[str]:
        ci = self.classes.get(class_name)
        if ci and method_name in ci.methods:
            return ci.file
        return None

    def find_function_file(self, func_name: str) -> Optional[str]:
        fi = self.functions.get(func_name)
        return fi.file if fi else None

    # ------------------------------------------------------------ resources
    def resources_for_target_type(self, target_type: str) -> List[ResourceRef]:
        """target_type may be comma-separated, e.g. 'market_event,market_event_bank_im'."""
        seen_paths = set()
        out: List[ResourceRef] = []
        for tt in (t.strip() for t in target_type.split(",") if t.strip()):
            for ref in self._resource_index.get(tt, []):
                if ref.path not in seen_paths:
                    seen_paths.add(ref.path)
                    out.append(ref)
        return out

    def insight_types_for_target_type(self, target_type: str) -> List[str]:
        names = []
        for r in self.resources_for_target_type(target_type):
            if r.tag == "insight_type":
                names.append(posixpath.basename(r.path))
        return sorted(set(names))
