"""
Exports a built `Hierarchy` (see pipeline.build_hierarchy) to a single
.xlsx workbook with three sheets:

  1. DAG Hierarchy   -- DAG -> sub-DAG/TaskGroup -> nested TaskGroup(s) ->
                         task -> method/function -> file where it's
                         defined -> file(s) where it's called, with paths.
  2. YAML Insight Types -- every yaml/yml file found in nlg-src, its
                         inferred target_type, its insight_type (only for
                         si_*.yml files), its path, and which
                         DAG/TaskGroup/task use that target_type.
  3. Product Hierarchy -- same shape as sheet 1, but scoped to tasks that
                         carry an explicit `product` tag, grouped by product.

No formulas are used (this is a flat data export, not a model), so no
recalculation step is required -- just professional formatting (bold
frozen header row, Arial, sensible column widths, filters).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

from .model import DagNode, TaskGroupNode, TaskNode
from .pipeline import Hierarchy

HEADER_FONT = Font(name="Arial", bold=True, color="FFFFFF")
HEADER_FILL = PatternFill(start_color="305496", end_color="305496", fill_type="solid")
BODY_FONT = Font(name="Arial", size=10)
WRAP = Alignment(vertical="top", wrap_text=True)


def _style_sheet(ws, col_widths: List[int]):
    ws.freeze_panes = "A2"
    for cell in ws[1]:
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(vertical="center", horizontal="center", wrap_text=True)
    ws.auto_filter.ref = ws.dimensions
    for i, w in enumerate(col_widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.font = BODY_FONT
            cell.alignment = WRAP


def _iter_tasks_with_path(dag: DagNode):
    """Yields (group_path, task) where group_path is a list of
    (group_id, defined_in_function, source_file) from the DAG root down to
    (but not including) the task's own TaskGroup -- i.e. the full
    sub-DAG/TaskGroup/nested-TaskGroup chain."""
    def walk(tasks, subgroups, path):
        for t in tasks:
            yield path, t
        for g in subgroups:
            yield from walk(g.tasks, g.subgroups,
                             path + [(g.group_id, g.defined_in_function, g.source_file)])
    yield from walk(dag.tasks, dag.groups, [])


def _hierarchy_rows(hierarchy: Hierarchy):
    """One row per (task, method) -- almost always one method per task."""
    rows = []
    for dag in hierarchy.dags:
        for path, task in _iter_tasks_with_path(dag):
            top_group = path[0][0] if path else ""
            nested_path = ".".join(p[0] for p in path[1:]) if len(path) > 1 else ""
            group_files = "; ".join(sorted(set(p[2] for p in path))) if path else ""
            methods = task.methods or []
            if not methods:
                methods = [None]
            for m in methods:
                rows.append({
                    "DAG": dag.dag_id,
                    "DAG File Path": dag.source_file,
                    "Sub-DAG / TaskGroup": top_group,
                    "Nested TaskGroup Path": nested_path,
                    "TaskGroup Defined-In File(s)": group_files,
                    "Task": task.task_id,
                    "Full Task Path": task.full_task_id,
                    "Operator": task.operator,
                    "Task Defined-In File (DAG py)": task.source_file,
                    "Method / Function": m.name if m else "",
                    "Method Defined-In File": m.file if m else "",
                    "Method/Function Called From File(s)": "; ".join(m.called_from) if m and m.called_from else "",
                    "Target Type": task.target_type or "",
                    "Insight Types": ", ".join(task.insight_types) if task.insight_types else "",
                    "Product": task.product or "",
                    "Resolution": m.resolution if m else "unresolved",
                    "Notes": m.note if m else "",
                })
    return rows


def _write_hierarchy_sheet(wb: Workbook, title: str, rows: List[dict]):
    ws = wb.create_sheet(title)
    headers = ["DAG", "DAG File Path", "Sub-DAG / TaskGroup", "Nested TaskGroup Path",
               "TaskGroup Defined-In File(s)", "Task", "Full Task Path", "Operator",
               "Task Defined-In File (DAG py)", "Method / Function", "Method Defined-In File",
               "Method/Function Called From File(s)", "Target Type", "Insight Types", "Product",
               "Resolution", "Notes"]
    ws.append(headers)
    for r in rows:
        ws.append([r.get(h, "") for h in headers])
    widths = [26, 34, 26, 26, 40, 32, 40, 20, 40, 32, 40, 46, 22, 30, 20, 12, 40]
    _style_sheet(ws, widths)
    return ws


def _write_product_sheet(wb: Workbook, rows: List[dict]):
    ws = wb.create_sheet("Product Hierarchy")
    headers = ["Product", "DAG", "DAG File Path", "Sub-DAG / TaskGroup", "Nested TaskGroup Path",
               "Task", "Full Task Path", "Operator", "Task Defined-In File (DAG py)",
               "Method / Function", "Method Defined-In File", "Method/Function Called From File(s)",
               "Target Type", "Insight Types", "Resolution", "Notes"]
    ws.append(headers)
    product_rows = [r for r in rows if r.get("Product")]
    product_rows.sort(key=lambda r: (r["Product"], r["DAG"], r["Full Task Path"]))
    for r in product_rows:
        ws.append([
            r["Product"], r["DAG"], r["DAG File Path"], r["Sub-DAG / TaskGroup"], r["Nested TaskGroup Path"],
            r["Task"], r["Full Task Path"], r["Operator"], r["Task Defined-In File (DAG py)"],
            r["Method / Function"], r["Method Defined-In File"], r["Method/Function Called From File(s)"],
            r["Target Type"], r["Insight Types"], r["Resolution"], r["Notes"],
        ])
    widths = [20, 26, 34, 26, 26, 32, 40, 20, 40, 32, 40, 46, 22, 30, 12, 40]
    _style_sheet(ws, widths)
    return ws, len(product_rows)


def _target_type_task_map(hierarchy: Hierarchy) -> Dict[str, List[Tuple[str, str, str, str]]]:
    """target_type -> [(dag_id, top_taskgroup, task_id, full_task_id), ...]"""
    m: Dict[str, List[Tuple[str, str, str, str]]] = {}
    for dag in hierarchy.dags:
        for path, task in _iter_tasks_with_path(dag):
            if not task.target_type:
                continue
            top_group = path[0][0] if path else ""
            for tt in (x.strip() for x in task.target_type.split(",") if x.strip()):
                m.setdefault(tt, []).append((dag.dag_id, top_group, task.task_id, task.full_task_id))
    return m


def _write_yaml_sheet(wb: Workbook, hierarchy: Hierarchy):
    ws = wb.create_sheet("YAML Insight Types")
    headers = ["Target Type (inferred)", "YAML/YML File Name", "File Path", "Is si_ Insight File",
               "Insight Type", "DAG", "Sub-DAG / TaskGroup", "Task", "Full Task Path"]
    ws.append(headers)

    tt_map = _target_type_task_map(hierarchy)
    src_index = hierarchy.src_index
    yaml_paths = sorted(p for p in src_index.raw_files if p.endswith((".yml", ".yaml")))

    row_count = 0
    for path in yaml_paths:
        base = path.rsplit("/", 1)[-1]
        parent_dir = path.rsplit("/", 2)[-2] if "/" in path else ""
        stem = base.rsplit(".", 1)[0]
        is_insight = base.startswith("si_")
        # Insight Type = the file name itself (keeps the "si_" prefix and
        # extension), e.g. "si_hk_529_no.yml" -- not the stripped stem.
        insight_type = base if is_insight else ""

        candidate = _infer_target_type(path, parent_dir, stem, tt_map)

        mappings = tt_map.get(candidate, []) if candidate else []
        if not mappings:
            ws.append([candidate or "(unmapped)", base, path, "Y" if is_insight else "N",
                       insight_type, "", "", "", ""])
            row_count += 1
        else:
            for dag_id, top_group, task_id, full_task_id in sorted(set(mappings)):
                ws.append([candidate, base, path, "Y" if is_insight else "N",
                           insight_type, dag_id, top_group, task_id, full_task_id])
                row_count += 1

    widths = [24, 40, 60, 16, 30, 26, 26, 32, 40]
    _style_sheet(ws, widths)
    return ws, row_count


# Folders whose immediate child directory names *are* target_type values,
# regardless of how deeply the actual file is nested underneath (e.g.
# "sql_column_mapping/household/housekeeping_checklist_explorer/si_hk_529_no.yml"
# -> target_type = "household", not "housekeeping_checklist_explorer").
#
# NOTE: "insight_group_layout" is deliberately excluded -- its immediate
# subfolders (market_event, life_events, goals, fga, banking, ...) are
# layout/section categories, not target_type values, so applying the same
# rule there produces wrong answers. Anything under insight_group_layout/
# is reported as "(unmapped)" for now until a real mapping is defined.
TARGET_TYPE_ROOT_FOLDERS = ("rules", "sql_column_mapping", "group_logic", "ref_table_columns")


def _infer_target_type(path: str, parent_dir: str, stem: str, tt_map: Dict[str, list]) -> Optional[str]:
    if path == "insight_group_layout" or path.startswith("insight_group_layout/"):
        return None
    parts = path.split("/")
    for i, part in enumerate(parts[:-1]):   # exclude the filename itself
        if part in TARGET_TYPE_ROOT_FOLDERS and i + 1 < len(parts) - 0:
            return parts[i + 1]
    # fallback for layouts without one of the known root folders, e.g.
    # config/input_data_config/<target_type>.yml, where the file stem
    # itself is the target_type.
    if parent_dir in tt_map:
        return parent_dir
    if stem in tt_map:
        return stem
    return None


def _write_legend_sheet(wb: Workbook, stats: dict):
    ws = wb.create_sheet("README", 0)
    ws.column_dimensions["A"].width = 34
    ws.column_dimensions["B"].width = 110
    rows = [
        ("Workbook", "NLG DAG hierarchy, YAML/insight-type mapping, and product hierarchy -- "
                      "auto-generated by nlg_dag_agent from static analysis of nlg-dags + nlg-src."),
        ("", ""),
        ("Sheet: DAG Hierarchy", "One row per task (per resolved method, usually 1:1). Columns walk "
                                  "DAG -> Sub-DAG/TaskGroup -> Nested TaskGroup Path -> Task -> "
                                  "Method/Function -> file where it's DEFINED -> file(s) where it's "
                                  "CALLED FROM. 'Sub-task' in this codebase = a task nested under a "
                                  "nested TaskGroup; see 'Nested TaskGroup Path' for that chain."),
        ("Sheet: YAML Insight Types", "Every .yml/.yaml file found under nlg-src. 'Target Type (inferred)' "
                                       "is the path segment immediately after a rules/sql_column_mapping/"
                                       "group_logic/ref_table_columns folder (however deeply the file is "
                                       "nested underneath), falling back to the parent folder name or file "
                                       "stem for other layouts (e.g. config/input_data_config/<target_type>."
                                       "yml). Anything under insight_group_layout/ is reported as "
                                       "'(unmapped)' -- its subfolders are layout/section categories, not "
                                       "target_type values, so the same rule doesn't apply there (yet). "
                                       "Files starting with 'si_' get Insight Type = the file name itself "
                                       "(e.g. 'si_hk_529_no.yml'), per spec. Rows with no DAG/Task shown "
                                       "are files whose target_type isn't currently referenced by any task "
                                       "(still listed for completeness)."),
        ("Sheet: Product Hierarchy", "Same shape as 'DAG Hierarchy', restricted to tasks that carry an "
                                      "explicit `product` value (from op_kwargs), grouped by product. Not "
                                      "every task has a product -- most product tags come from NlgProduct-"
                                      "driven tasks (client briefings, alerts, JSON insights, etc.)."),
        ("", ""),
        ("Resolution column", "static = direct python_callable match; dispatch = resolved via the "
                               "eval(f\"{obj}.{action}()\") dynamic-dispatch idiom; llm = resolved by LLM "
                               "fallback (verify manually); unresolved = needs manual check. "
                               "Postgres-only tasks show Method/Function = N/A, Notes = "
                               "'Postgres operator with sql code'."),
        ("Generated stats", f"DAGs: {stats['dags']}  |  Tasks: {stats['tasks']}  |  "
                             f"Methods resolved: {stats['resolved']}/{stats['total_methods']}  |  "
                             f"YAML files indexed: {stats['yaml_files']}"),
    ]
    for r in rows:
        ws.append(r)
    for row in ws.iter_rows():
        for cell in row:
            cell.font = BODY_FONT
            cell.alignment = WRAP
    ws["A1"].font = Font(name="Arial", bold=True, size=13)
    for r in (3, 4, 5, 7):
        ws.cell(row=r, column=1).font = Font(name="Arial", bold=True)
    return ws


def export_xlsx(hierarchy: Hierarchy, output_path: str) -> str:
    wb = Workbook()
    wb.remove(wb.active)  # drop the default empty sheet

    rows = _hierarchy_rows(hierarchy)
    _write_hierarchy_sheet(wb, "DAG Hierarchy", rows)
    _, product_rows = _write_product_sheet(wb, rows)
    _, yaml_rows = _write_yaml_sheet(wb, hierarchy)

    total_methods = sum(1 for r in rows)
    resolved = sum(1 for r in rows if r["Resolution"] not in ("unresolved",))
    stats = {
        "dags": len(hierarchy.dags),
        "tasks": len({r["Full Task Path"] for r in rows}),
        "total_methods": total_methods,
        "resolved": resolved,
        "yaml_files": yaml_rows,
    }
    _write_legend_sheet(wb, stats)

    import os
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    wb.save(output_path)
    return output_path
