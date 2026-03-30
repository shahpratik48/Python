#!/usr/bin/env python3
"""
SQL Script Parser
=================
Parses a complex SQL script to extract table lineage, relationships (JOINs,
source derivations), and generates a rich interactive ER diagram (HTML).

Usage:
    python sql_parser.py
    → You will be prompted to enter the SQL filename.
      The file must be in the SAME FOLDER as this script.

Output:
    - <filename>_lineage.json        — full table lineage as JSON
    - <filename>_er_diagram.html     — interactive ER diagram (open in browser)
"""

import re
import sys
import json
import os
from pathlib import Path
from collections import defaultdict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_table_name(raw: str) -> str:
    """Strip {{params.*}} schema prefixes and lowercase the table name."""
    # Remove Jinja-style schema placeholders like {{params.IKG_SCHEMA}}.
    name = re.sub(r'\{\{[^}]+\}\}\.', '', raw)
    return name.strip().strip('"').lower()


def strip_comments(sql: str) -> str:
    """Remove -- line comments and /* ... */ block comments."""
    # Block comments (non-greedy, including newlines)
    sql = re.sub(r'/\*.*?\*/', ' ', sql, flags=re.DOTALL)
    # Line comments
    sql = re.sub(r'--[^\n]*', ' ', sql)
    return sql


def is_ignorable(stmt: str) -> bool:
    """
    Return True for statements we should skip:
      - OWNER TO …
      - GRANT … TO …
      - SET …
      - ANALYZE …
    """
    s = stmt.strip().upper()
    if re.match(r'^\s*(OWNER\s+TO|GRANT\s+\w|SET\s+\w|ANALYZE\s+\w|COMMENT\s+ON)', s):
        return True
    # Also skip ALTER … OWNER / GRANT patterns anywhere in stmt
    if re.match(r'^\s*ALTER\s+TABLE\s+\S+\s+OWNER\s+TO', s):
        return True
    if re.match(r'^\s*GRANT\s+', s):
        return True
    return False


# ---------------------------------------------------------------------------
# Core Extractor
# ---------------------------------------------------------------------------

_TABLE_PLACEHOLDER = r'(?:\{\{[^}]+\}\}\.)?(\w+)'

# Pattern to capture CREATE TABLE / CREATE TEMP TABLE / CREATE TABLE IF NOT EXISTS
_RE_CREATE = re.compile(
    r'\bCREATE\s+(?:TEMP(?:ORARY)?\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?' + _TABLE_PLACEHOLDER,
    re.IGNORECASE,
)

# Pattern for INSERT INTO
_RE_INSERT = re.compile(
    r'\bINSERT\s+INTO\s+' + _TABLE_PLACEHOLDER,
    re.IGNORECASE,
)

# Pattern for DROP TABLE
_RE_DROP = re.compile(
    r'\bDROP\s+TABLE\s+(?:IF\s+EXISTS\s+)?' + _TABLE_PLACEHOLDER,
    re.IGNORECASE,
)

# Pattern for FROM / JOIN source tables
_RE_FROM_JOIN = re.compile(
    r'\b(?:FROM|JOIN)\s+' + _TABLE_PLACEHOLDER + r'(?:\s+(?:AS\s+)?(\w+))?',
    re.IGNORECASE,
)

# WITH clause CTE names — we will collect these to avoid treating them as real tables
_RE_WITH_CTE = re.compile(
    r'\bWITH\b(.*?)(?=\bSELECT\b)',
    re.IGNORECASE | re.DOTALL,
)
_RE_CTE_NAME = re.compile(r'\b(\w+)\s+AS\s*\(', re.IGNORECASE)


def extract_join_conditions(sql_block: str, source_table: str, aliases: dict) -> list:
    """
    Try to capture ON conditions from JOINs to derive join columns.
    Returns a list of (left_col, right_col) strings.
    """
    conds = []
    for m in re.finditer(r'\bON\b\s+(.+?)(?=\b(?:LEFT|RIGHT|INNER|FULL|CROSS|JOIN|WHERE|GROUP|ORDER|HAVING|LIMIT|DISTRIBUTED|;)\b)',
                         sql_block, re.IGNORECASE | re.DOTALL):
        cond = m.group(1).strip()
        # Find alias.col = alias.col patterns
        for eq in re.finditer(r'(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)', cond):
            conds.append((f"{eq.group(1)}.{eq.group(2)}", f"{eq.group(3)}.{eq.group(4)}"))
    return conds


class SQLParser:
    def __init__(self, sql_text: str):
        self.raw_sql = sql_text
        self.clean_sql = strip_comments(sql_text)

        # Tables created/written in this script (IKG schema tables)
        self.created_tables: set = set()
        # External / reference tables (EDW views, model schema, etc.)
        self.external_tables: set = set()
        # Temp tables
        self.temp_tables: set = set()

        # Relationships: list of dicts
        # { "from": source_table, "to": target_table, "join_cols": [...], "rel_type": "JOIN"|"SOURCE" }
        self.relationships: list = []
        # Alias map: alias -> table_name
        self.alias_map: dict = {}

        # CTE names to exclude from "real" table tracking
        self.cte_names: set = set()

        # Table to its source tables (direct derivation)
        self.table_sources: dict = defaultdict(set)   # target -> {sources}

    def _collect_cte_names(self):
        for m in _RE_WITH_CTE.finditer(self.clean_sql):
            block = m.group(1)
            for nm in _RE_CTE_NAME.finditer(block):
                self.cte_names.add(nm.group(1).lower())

    def _is_external(self, raw_ref: str) -> bool:
        """Detect if this reference came from an external schema."""
        schema_pat = r'\{\{params\.(EDW_VIEW_INPUT_SCHEMA|EDW_INPUT_SCHEMA|MODEL_SCHEMA)[^}]*\}\}\.'
        return bool(re.search(schema_pat, raw_ref, re.IGNORECASE))

    def _register_table(self, name: str, is_temp: bool = False, is_external: bool = False):
        name = name.lower()
        if name in self.cte_names:
            return
        if is_external:
            self.external_tables.add(name)
        elif is_temp:
            self.temp_tables.add(name)
        else:
            self.created_tables.add(name)

    def parse(self):
        self._collect_cte_names()

        # Split on semicolons to get individual statements
        statements = self.clean_sql.split(';')

        for stmt in statements:
            stmt = stmt.strip()
            if not stmt:
                continue
            if is_ignorable(stmt):
                continue
            self._parse_statement(stmt)

        # Resolve alias map for relationships
        self._resolve_aliases()

    def _parse_statement(self, stmt: str):
        upper = stmt.upper().strip()

        # ---- Detect CREATE TABLE ----
        is_temp_stmt = bool(re.match(r'\s*CREATE\s+TEMP', stmt, re.IGNORECASE))
        create_m = _RE_CREATE.search(stmt)
        if create_m:
            target = normalize_table_name(create_m.group(0).split()[-1])
            self._register_table(target, is_temp=is_temp_stmt)
            # Now find all FROM/JOIN in same statement → those are sources
            self._extract_sources(stmt, target)
            return

        # ---- Detect INSERT INTO ----
        insert_m = _RE_INSERT.search(stmt)
        if insert_m:
            # Extract the raw match to check schema
            raw = insert_m.group(0)
            target = normalize_table_name(insert_m.group(1))
            is_ext = self._is_external(raw)
            self._register_table(target, is_external=is_ext)
            self._extract_sources(stmt, target)
            return

        # ---- DROP TABLE (just record, no sources) ----
        drop_m = _RE_DROP.search(stmt)
        if drop_m:
            return  # nothing useful

        # ---- Fallback: try to find FROM/JOIN (e.g., plain SELECT) ----
        # No target table — skip relationship tracking here

    def _extract_sources(self, stmt: str, target: str):
        """
        Find all FROM / JOIN table references in a statement and record
        them as sources for `target`.
        Also build alias map.
        """
        # Collect CTE names defined in this statement
        local_ctes: set = set()
        for m in _RE_WITH_CTE.finditer(stmt):
            block = m.group(1)
            for nm in _RE_CTE_NAME.finditer(block):
                local_ctes.add(nm.group(1).lower())

        # Build per-statement alias map first (table -> alias)
        stmt_table_to_alias = {}
        _skip = {
            'where','on','set','select','inner','left','right','full',
            'outer','cross','join','from','as','not','null','and','or',
            'in','by','group','order','having','limit','union','all',
            'distinct','with','values','distributed','using','for',
            'case','when','then','else','end','exists','between',
            'like','ilike','is','true','false','into','insert','update',
            'delete','create','drop','alter','table','temp','temporary',
            'partition','over','partition','window','rows','range',
            'unbounded','preceding','following','current','row',
            'natural','lateral','recursive','only','returning',
        }
        for m in _RE_FROM_JOIN.finditer(stmt):
            traw = normalize_table_name(m.group(1))
            al   = (m.group(2) or '').lower()
            if al and al not in _skip and al != traw:
                stmt_table_to_alias[traw] = al

        for m in _RE_FROM_JOIN.finditer(stmt):
            raw_full = m.group(0)
            table_name_raw = m.group(1)
            alias = m.group(2)

            table_name = normalize_table_name(table_name_raw)

            # Skip CTEs, keywords, and the target itself
            if table_name in local_ctes or table_name in self.cte_names:
                continue
            if table_name.upper() in ('SELECT', 'WITH', 'VALUES', 'LATERAL',
                                       'UNNEST', 'ROWS', 'ONLY'):
                continue
            if table_name == target:
                continue

            is_ext = self._is_external(raw_full)
            self._register_table(table_name, is_external=is_ext,
                                 is_temp=table_name.endswith('_temp_auto') or
                                          table_name.endswith('_tmp') or
                                          table_name.startswith('temp_'))

            if alias:
                self.alias_map[alias.lower()] = table_name

            self.table_sources[target].add(table_name)

            # Extract join conditions for this source→target link
            raw_join_cols = extract_join_conditions(stmt, table_name, self.alias_map)

            # Deduplicate raw tuples (keep full alias.col = alias.col form)
            seen_jk = set()
            dedup_jk = []
            for jc in raw_join_cols:
                if isinstance(jc, (list, tuple)) and len(jc) == 2:
                    key = (str(jc[0]).lower(), str(jc[1]).lower())
                else:
                    key = str(jc).lower()
                if key not in seen_jk:
                    seen_jk.add(key)
                    if isinstance(jc, tuple):
                        dedup_jk.append([jc[0], jc[1]])
                    else:
                        dedup_jk.append(str(jc))

            src_alias = stmt_table_to_alias.get(table_name, alias.lower() if alias else "")
            tgt_alias = stmt_table_to_alias.get(target, "")

            # Record relationship with raw join pairs
            self.relationships.append({
                "from": table_name,
                "to": target,
                "join_cols": dedup_jk,
                "alias": src_alias,
                "target_alias": tgt_alias,
                "rel_type": "JOIN/SOURCE",
                "is_external": is_ext,
            })

    def _resolve_aliases(self):
        """Replace alias names in relationships with actual table names."""
        resolved = []
        for rel in self.relationships:
            src = self.alias_map.get(rel["from"], rel["from"])
            tgt = self.alias_map.get(rel["to"], rel["to"])
            rel["from"] = src
            rel["to"] = tgt
            resolved.append(rel)
        self.relationships = resolved

    def summary(self) -> dict:
        """Return a structured summary."""
        # Deduplicate relationships by (from, to), merging join_cols
        merged = {}
        for r in self.relationships:
            key = (r["from"], r["to"])
            if key not in merged:
                merged[key] = {
                    "from": r["from"],
                    "to": r["to"],
                    "join_cols": list(r.get("join_cols", [])),
                    "alias": r.get("alias", ""),
                    "target_alias": r.get("target_alias", ""),
                    "rel_type": r.get("rel_type", ""),
                    "is_external": r.get("is_external", False),
                }
            else:
                # Merge join keys (handle list/tuple items)
                existing_keys = set(
                    tuple(jk) if isinstance(jk, list) else jk
                    for jk in merged[key]["join_cols"]
                )
                for jk in r.get("join_cols", []):
                    jk_key = tuple(jk) if isinstance(jk, list) else jk
                    if jk_key not in existing_keys:
                        merged[key]["join_cols"].append(jk)
                        existing_keys.add(jk_key)
                # Keep alias if not yet set
                if not merged[key]["alias"] and r.get("alias"):
                    merged[key]["alias"] = r["alias"]
        deduped = list(merged.values())

        return {
            "created_tables": sorted(self.created_tables),
            "temp_tables": sorted(self.temp_tables),
            "external_tables": sorted(self.external_tables),
            "relationships": deduped,
            "table_sources": {k: list(v) for k, v in self.table_sources.items()},
        }


import json

def build_er_html(summary: dict) -> str:
    import json as _json
    from collections import deque as _deque

    FINAL = 'account_profile_curr_ikg_temp_auto'

    rel_up = {}
    for r in summary['relationships']:
        rel_up.setdefault(r['to'], []).append(r)

    level_of = {}
    queue = _deque([(FINAL, 0)])
    dep_edges_raw = []
    edge_set = set()

    while queue:
        tbl, depth = queue.popleft()
        if tbl in level_of:
            continue
        level_of[tbl] = depth
        for r in rel_up.get(tbl, []):
            src = r['from']
            key = (src, tbl)
            if key not in edge_set:
                edge_set.add(key)
                dep_edges_raw.append(r)
            if src not in level_of:
                queue.append((src, depth + 1))

    created  = set(summary['created_tables'])
    temp_set = set(summary['temp_tables'])

    def node_type(n):
        if n == FINAL:    return 'final'
        if n in created:  return 'ikg'
        if n in temp_set: return 'tmp'
        return 'ext'

    nodes_js = [{'id': n, 'level': lv, 'type': node_type(n)} for n, lv in level_of.items()]

    # Build edges with cleaned alias and structured join keys
    ALIAS_BLACKLIST = {
        'where','on','set','select','inner','left','right','full','outer','cross',
        'join','from','as','not','null','and','or','in','by','group','order',
        'having','limit','union','all','distinct','with','values','distributed',
        'using','for','case','when','then','else','end','exists','between',
        'like','ilike','is','true','false','into','insert','update','delete',
        'create','drop','alter','table','temp','temporary','partition','over',
        'window','rows','range','unbounded','preceding','following','current',
        'row','natural','lateral','recursive','only','returning','returning',
        'inner','model_data_temp_auto',
    }

    def clean_alias(a):
        if not a:
            return ''
        a = a.strip().lower()
        if a in ALIAS_BLACKLIST:
            return ''
        # Must look like a real alias: letters/digits/underscore, not too long
        import re
        if not re.match(r'^[a-z_][a-z0-9_]{0,29}$', a):
            return ''
        return a

    edges_js = []
    for e in dep_edges_raw:
        src_alias = clean_alias(e.get('alias', ''))
        tgt_alias = clean_alias(e.get('target_alias', ''))

        # Build join key list: each item is {"left": "a.col", "right": "b.col"}
        # or {"col": "colname"} if both sides are the same column
        jk_list = []
        seen_jk = set()
        for jc in e.get('join_cols', []):
            if isinstance(jc, (list, tuple)) and len(jc) == 2:
                l, r2 = str(jc[0]), str(jc[1])
                key = (l.lower(), r2.lower())
                if key not in seen_jk:
                    seen_jk.add(key)
                    jk_list.append({'left': l, 'right': r2})
            elif isinstance(jc, str) and jc:
                key = jc.lower()
                if key not in seen_jk:
                    seen_jk.add(key)
                    jk_list.append({'col': jc})

        edges_js.append({
            'from':         e['from'],
            'to':           e['to'],
            'jk':           jk_list,
            'src_alias':    src_alias,
            'tgt_alias':    tgt_alias,
            'is_external':  e.get('is_external', False),
        })

    data_json = _json.dumps(
        {'nodes': nodes_js, 'edges': edges_js, 'final': FINAL},
        separators=(',', ':')
    )

    css = """
*, *::before, *::after { box-sizing:border-box; margin:0; padding:0; }
body { background:#0f1117; color:#e2e8f0;
  font-family:'Segoe UI',system-ui,sans-serif;
  overflow:hidden; height:100vh; display:flex; flex-direction:column; }

#header { background:linear-gradient(90deg,#1e3a5f,#0f2744);
  padding:11px 20px; display:flex; align-items:center; gap:14px;
  border-bottom:1px solid #1e2d40; flex-shrink:0; }
#header h1 { font-size:.95rem; font-weight:700; color:#63b3ed; }
.sub { font-size:.68rem; color:#3d5070; margin-top:1px; }
.sp { padding:3px 10px; border-radius:20px; font-size:.67rem; font-weight:600; }
.sp-g { background:#1a4731; color:#68d391; border:1px solid #2f855a44; }
.sp-b { background:#1a365d; color:#63b3ed; border:1px solid #2b6cb044; }
.sp-p { background:#322659; color:#b794f4; border:1px solid #6b46c144; }

#toolbar { background:#161b27; border-bottom:1px solid #1e2d40;
  padding:6px 14px; display:flex; align-items:center; gap:7px;
  flex-shrink:0; flex-wrap:wrap; }
.tbtn { background:#1e2535; color:#8a9ab0; border:1px solid #252f40;
  border-radius:5px; padding:4px 12px; font-size:.73rem; cursor:pointer; transition:all .12s; }
.tbtn:hover { background:#2a3548; color:#e2e8f0; }
.tbtn.on { background:#17304d; color:#63b3ed; border-color:#2b4a70; }
.tsep { width:1px; height:20px; background:#1e2d40; margin:0 3px; }
#srch { background:#1a2230; color:#e2e8f0; border:1px solid #252f40;
  border-radius:5px; padding:4px 11px; font-size:.73rem; width:190px; outline:none; }
#srch:focus { border-color:#3a6090; }
#srch::placeholder { color:#2d3a4a; }
#legend { display:flex; gap:12px; margin-left:auto; align-items:center; }
.leg { display:flex; align-items:center; gap:5px; font-size:.67rem; color:#4a5a6a; }
.leg-b { width:13px; height:9px; border-radius:2px; }

#wrap { flex:1; position:relative; overflow:hidden; display:flex; }
#cv-area { flex:1; position:relative; overflow:hidden; background:#0b0f18; }
#cv { position:absolute; top:0; left:0; display:block; cursor:grab; }

/* ── DETAIL PANEL ─────────────────────────────────────────────── */
#panel {
  width:430px; min-width:430px; background:#0d1525;
  border-left:1px solid #1a2535;
  display:none; flex-direction:column; overflow:hidden; flex-shrink:0;
}
#panel.open { display:flex; }

#ph {
  background:#0a1220; padding:12px 14px 10px;
  border-bottom:1px solid #1a2535; flex-shrink:0;
}
#ph-title { font-size:.9rem; font-weight:700; color:#7ec8f8;
  word-break:break-all; margin-bottom:4px; padding-right:22px; }
#ph-meta { font-size:.69rem; }
#ph-close { position:absolute; right:14px; top:14px;
  background:none; border:none; color:#3a4a5a; cursor:pointer;
  font-size:1.1rem; transition:color .1s; }
#ph-close:hover { color:#e2e8f0; }

#pb { flex:1; overflow-y:auto; }
#pb::-webkit-scrollbar { width:4px; }
#pb::-webkit-scrollbar-track { background:#0a1018; }
#pb::-webkit-scrollbar-thumb { background:#1e2d40; border-radius:3px; }

.sec { border-bottom:1px solid #111d2b; }
.sec-hdr {
  padding:9px 14px; font-size:.68rem; font-weight:700;
  text-transform:uppercase; letter-spacing:.8px;
  display:flex; align-items:center; gap:8px;
  cursor:pointer; user-select:none; transition:background .1s;
  color:#3a5a7a;
}
.sec-hdr:hover { background:#0f1c2d; }
.sec-hdr.src-hdr { color:#5a9060; }
.sec-hdr.dst-hdr { color:#8a6020; }
.sec-cnt { margin-left:auto; background:#111d2b; color:#4a7090;
  font-size:.63rem; padding:1px 8px; border-radius:10px; font-weight:700; }
.arr { font-size:.65rem; color:#2a3a4a; transition:transform .15s; }
.sec-hdr.collapsed .arr { transform:rotate(-90deg); }
.sec-body { display:block; }
.sec-hdr.collapsed + .sec-body { display:none; }

/* table row */
.trow { padding:8px 12px 8px 16px; border-bottom:1px solid #0d1828;
  cursor:pointer; transition:background .1s; }
.trow:hover { background:#0f1c2e; }
.trow:last-child { border-bottom:none; }
.trow-top { display:flex; align-items:flex-start; gap:8px; margin-bottom:5px; }
.tdot { width:8px; height:8px; border-radius:2px; flex-shrink:0; margin-top:3px; }
.tname { font-size:.77rem; font-weight:600; color:#c0d0e0; flex:1;
  word-break:break-all; line-height:1.4; }
.talias { font-size:.64rem; color:#2d5070; background:#091525;
  border:1px solid #152535; border-radius:4px;
  padding:1px 7px; font-family:'Consolas','Courier New',monospace;
  white-space:nowrap; flex-shrink:0; align-self:flex-start; }

/* join keys table */
.jk-table { width:100%; border-collapse:collapse;
  margin:0 0 4px 16px; width:calc(100% - 16px); }
.jk-table th { font-size:.61rem; color:#2a4a5a; font-weight:600;
  text-transform:uppercase; letter-spacing:.5px;
  padding:3px 8px 3px 0; border-bottom:1px solid #0f1e2e; }
.jk-table td { font-size:.68rem; font-family:'Consolas','Courier New',monospace;
  padding:3px 8px 3px 0; color:#8ab0c0; vertical-align:top;
  border-bottom:1px solid #0a1520; }
.jk-table tr:last-child td { border-bottom:none; }
.jk-src { color:#68d391; }
.jk-dst { color:#f6ad55; }
.jk-eq  { color:#2a3a4a; padding:0 4px; }
.no-jk  { font-size:.65rem; color:#1e2d3a; padding:3px 0 3px 16px;
  font-style:italic; }

#sb { background:#161b27; border-top:1px solid #1a2535;
  padding:3px 14px; font-size:.67rem; color:#2d3a4a;
  display:flex; gap:18px; flex-shrink:0; }
#sb b { color:#3d4a5a; }
"""

    body = """
<div id="header">
  <div>
    <div id="h-title">&#128202; account_profile_curr_ikg &mdash; Table Dependency ER Diagram</div>
    <div class="sub">Click any node to see source/destination tables with aliases and join keys</div>
  </div>
  <span class="sp sp-g" id="sp-n">-- nodes</span>
  <span class="sp sp-b" id="sp-e">-- edges</span>
  <span class="sp sp-p" id="sp-l">-- levels</span>
</div>
<div id="toolbar">
  <button class="tbtn" onclick="resetView()">&#8635; Reset</button>
  <button class="tbtn" onclick="zoomIn()">+ Zoom</button>
  <button class="tbtn" onclick="zoomOut()">&#8722; Zoom</button>
  <div class="tsep"></div>
  <input id="srch" placeholder="&#128269; Search table..." oninput="onSearch()">
  <button class="tbtn" onclick="clearSearch()">&#10005;</button>
  <div class="tsep"></div>
  <button class="tbtn on" id="btn-ikg" onclick="toggle('ikg')">IKG</button>
  <button class="tbtn on" id="btn-tmp" onclick="toggle('tmp')">Temp</button>
  <button class="tbtn on" id="btn-ext" onclick="toggle('ext')">External</button>
  <div class="tsep"></div>
  <button class="tbtn" onclick="fitLv(1)">Lv 1</button>
  <button class="tbtn" onclick="fitLv(2)">Lv 1-2</button>
  <button class="tbtn" onclick="fitLv(3)">Lv 1-3</button>
  <button class="tbtn" onclick="fitAll()">All</button>
  <div id="legend">
    <div class="leg"><div class="leg-b" style="background:#2ecc71;border:1px solid #27ae60"></div>Final</div>
    <div class="leg"><div class="leg-b" style="background:#1a5080;border:1px solid #2980b9"></div>IKG</div>
    <div class="leg"><div class="leg-b" style="background:#3d1a6e;border:1px solid #8e44ad"></div>Temp</div>
    <div class="leg"><div class="leg-b" style="background:#4a2800;border:1px solid #e67e22"></div>External</div>
  </div>
</div>
<div id="wrap">
  <div id="cv-area"><canvas id="cv"></canvas></div>
  <div id="panel">
    <div id="ph" style="position:relative;">
      <button id="ph-close" onclick="closePanel()">&#10005;</button>
      <div id="ph-title"></div>
      <div id="ph-meta"></div>
    </div>
    <div id="pb">
      <div class="sec">
        <div class="sec-hdr src-hdr" id="hdr-src" onclick="toggleSec('src')">
          &#8593;&nbsp;Source Tables&nbsp;<span style="color:#2a5a30;font-weight:400;font-size:.62rem;text-transform:none">(inputs to this table)</span>
          <span class="sec-cnt" id="cnt-src">0</span>
          <span class="arr">&#9660;</span>
        </div>
        <div class="sec-body" id="body-src"></div>
      </div>
      <div class="sec">
        <div class="sec-hdr dst-hdr collapsed" id="hdr-dst" onclick="toggleSec('dst')">
          &#8595;&nbsp;Destination Tables&nbsp;<span style="color:#5a3a10;font-weight:400;font-size:.62rem;text-transform:none">(tables that use this table)</span>
          <span class="sec-cnt" id="cnt-dst">0</span>
          <span class="arr">&#9660;</span>
        </div>
        <div class="sec-body" id="body-dst"></div>
      </div>
    </div>
  </div>
</div>
<div id="sb">
  <span>Zoom: <b id="sb-z">100%</b></span>
  <span>Visible: <b id="sb-v">0</b></span>
  <span>Scroll=zoom &bull; Drag=pan &bull; Click node for details</span>
</div>
"""

    js = "const RAW=" + data_json + ";\n" + r"""
const STYLE={
  final:{fill:'#0a2a18',stroke:'#2ecc71',text:'#90f0b0'},
  ikg:  {fill:'#0a1c30',stroke:'#2980b9',text:'#80b8e8'},
  tmp:  {fill:'#180a2e',stroke:'#8e44ad',text:'#c080e8'},
  ext:  {fill:'#261200',stroke:'#e67e22',text:'#f0a860'},
};
const TC={final:'#2ecc71',ikg:'#2980b9',tmp:'#8e44ad',ext:'#e67e22'};
const TL={final:'Final Output',ikg:'IKG Table',tmp:'Temp Table',ext:'External/EDW'};

const NM={},EO={},EI={};
RAW.nodes.forEach(function(n){
  n.w=Math.max(180,n.id.length*7.2+32); n.h=36; n.x=0; n.y=0;
  NM[n.id]=n; EO[n.id]=[]; EI[n.id]=[];
});
RAW.edges.forEach(function(e){
  if(NM[e.from]&&NM[e.to]){EO[e.from].push(e);EI[e.to].push(e);}
});

function layout(){
  var byLv={};
  RAW.nodes.forEach(function(n){if(!byLv[n.level])byLv[n.level]=[];byLv[n.level].push(n);});
  var maxLv=Math.max.apply(null,RAW.nodes.map(function(n){return n.level;}));
  var CW=260,RH=52,PX=60,PY=50;
  Object.keys(byLv).forEach(function(lv){
    var col=byLv[lv];
    col.sort(function(a,b){return(EO[b.id].length+EI[b.id].length)-(EO[a.id].length+EI[a.id].length);});
    col.forEach(function(n,i){n.x=PX+(maxLv-lv)*CW;n.y=PY+i*RH;});
  });
  for(var i=0;i<80;i++)forceStep();
}
function forceStep(){
  var f={};
  RAW.nodes.forEach(function(n){f[n.id]={x:0,y:0};});
  for(var i=0;i<RAW.nodes.length;i++){
    for(var j=i+1;j<RAW.nodes.length;j++){
      var a=RAW.nodes[i],b=RAW.nodes[j];
      var dx=b.x-a.x||.1,dy=b.y-a.y||.1,d=Math.sqrt(dx*dx+dy*dy)||1,rep=500/(d*d);
      f[a.id].x-=rep*dx/d;f[a.id].y-=rep*dy/d;f[b.id].x+=rep*dx/d;f[b.id].y+=rep*dy/d;
    }
  }
  RAW.edges.forEach(function(e){
    var a=NM[e.from],b=NM[e.to];if(!a||!b)return;
    var dx=b.x-a.x,dy=b.y-a.y,att=0.04;
    f[a.id].x+=att*dx;f[a.id].y+=att*dy;f[b.id].x-=att*dx;f[b.id].y-=att*dy;
  });
  RAW.nodes.forEach(function(n){n.x+=f[n.id].x*.8;n.y+=f[n.id].y*.8;});
}
layout();

var cvA=document.getElementById('cv-area'),cv=document.getElementById('cv'),ctx=cv.getContext('2d');
var tx=0,ty=0,sc=1,maxLv=99,showT={ikg:true,tmp:true,ext:true},srchQ='',hlId=null;

function isVis(n){
  if(n.level>maxLv)return false;
  if(n.type!=='final'&&!showT[n.type])return false;
  if(srchQ&&n.id.indexOf(srchQ)<0)return false;
  return true;
}

function render(){
  ctx.clearRect(0,0,cv.width,cv.height);
  ctx.save();ctx.translate(tx,ty);ctx.scale(sc,sc);
  drawGrid();
  RAW.edges.forEach(function(e){
    var a=NM[e.from],b=NM[e.to];
    if(!a||!b||!isVis(a)||!isVis(b))return;
    drawEdge(e,a,b);
  });
  RAW.nodes.forEach(function(n){if(isVis(n))drawNode(n);});
  ctx.restore();
  document.getElementById('sb-v').textContent=RAW.nodes.filter(isVis).length;
}

function drawGrid(){
  var gs=40,ox=-tx/sc,oy=-ty/sc,W=cv.width/sc,H=cv.height/sc;
  ctx.strokeStyle='#10192a';ctx.lineWidth=1;
  for(var x=Math.floor(ox/gs)*gs;x<ox+W;x+=gs){ctx.beginPath();ctx.moveTo(x,oy);ctx.lineTo(x,oy+H);ctx.stroke();}
  for(var y=Math.floor(oy/gs)*gs;y<oy+H;y+=gs){ctx.beginPath();ctx.moveTo(ox,y);ctx.lineTo(ox+W,y);ctx.stroke();}
}

function drawEdge(e,a,b){
  var hl=hlId&&(e.from===hlId||e.to===hlId);
  var x1=a.x+a.w,y1=a.y+a.h/2,x2=b.x,y2=b.y+b.h/2,cp=Math.abs(x2-x1)*.42;
  ctx.beginPath();ctx.moveTo(x1,y1);ctx.bezierCurveTo(x1+cp,y1,x2-cp,y2,x2,y2);
  ctx.strokeStyle=hl?(e.to===hlId?'#4adb80':'#f0a840'):(srchQ?'#151f2e':'#1e2d42');
  ctx.lineWidth=hl?2:1;ctx.globalAlpha=hl?1:.5;ctx.stroke();ctx.globalAlpha=1;
  if(hl){
    var col=e.to===hlId?'#4adb80':'#f0a840';
    var ang=Math.atan2(y2-(y1+(y2-y1)*.8),x2-(x1+(x2-x1)*.8)),as=7;
    ctx.beginPath();ctx.moveTo(x2,y2);
    ctx.lineTo(x2-as*Math.cos(ang-.4),y2-as*Math.sin(ang-.4));
    ctx.lineTo(x2-as*Math.cos(ang+.4),y2-as*Math.sin(ang+.4));
    ctx.closePath();ctx.fillStyle=col;ctx.fill();
  }
}

function drawNode(n){
  var s=STYLE[n.type],isFinal=n.id===RAW.final,isHl=hlId===n.id;
  var conn=hlId&&(EO[n.id].some(function(e){return e.to===hlId;})||EI[n.id].some(function(e){return e.from===hlId;}));
  var dim=hlId&&!isHl&&!conn;
  if(isHl||isFinal){ctx.shadowColor=s.stroke;ctx.shadowBlur=isHl?18:10;}
  ctx.globalAlpha=dim?.18:1;
  ctx.fillStyle=s.fill;rrect(n.x,n.y,n.w,n.h,6);ctx.fill();
  ctx.strokeStyle=isHl?'#ffffff':(srchQ&&n.id.indexOf(srchQ)>=0?'#f0d060':s.stroke);
  ctx.lineWidth=isHl||isFinal?2:1;rrect(n.x,n.y,n.w,n.h,6);ctx.stroke();
  ctx.shadowBlur=0;
  ctx.fillStyle=s.stroke;rrect(n.x,n.y,n.w,3,{tl:6,tr:6,bl:0,br:0});ctx.fill();
  ctx.fillStyle=dim?'#1e2a38':s.text;
  ctx.font=(isFinal?'bold ':'')+' 10.5px Segoe UI,system-ui,sans-serif';
  ctx.textAlign='left';ctx.textBaseline='middle';
  var lbl=n.id,mw=n.w-36;
  while(ctx.measureText(lbl).width>mw&&lbl.length>4)lbl=lbl.slice(0,-4)+'...';
  ctx.fillText(lbl,n.x+10,n.y+n.h/2+2);
  ctx.font='8px Segoe UI,system-ui,sans-serif';
  ctx.fillStyle=dim?'#1e2a38':s.stroke+'99';ctx.textAlign='right';
  ctx.fillText(isFinal?'OUTPUT':'L'+n.level,n.x+n.w-5,n.y+n.h/2+2);
  ctx.globalAlpha=1;
}

function rrect(x,y,w,h,r){
  var tl,tr,bl,br;
  if(typeof r==='number'){tl=tr=bl=br=r;}else{tl=r.tl||0;tr=r.tr||0;bl=r.bl||0;br=r.br||0;}
  ctx.beginPath();
  ctx.moveTo(x+tl,y);ctx.lineTo(x+w-tr,y);ctx.arcTo(x+w,y,x+w,y+tr,tr);
  ctx.lineTo(x+w,y+h-br);ctx.arcTo(x+w,y+h,x+w-br,y+h,br);
  ctx.lineTo(x+bl,y+h);ctx.arcTo(x,y+h,x,y+h-bl,bl);
  ctx.lineTo(x,y+tl);ctx.arcTo(x,y,x+tl,y,tl);
  ctx.closePath();
}

var panning=null;
cv.addEventListener('mousedown',function(e){panning={mx:e.clientX,my:e.clientY,tx:tx,ty:ty};});
cv.addEventListener('mousemove',function(e){
  if(!panning)return;
  tx=panning.tx+(e.clientX-panning.mx);ty=panning.ty+(e.clientY-panning.my);render();
});
cv.addEventListener('mouseup',function(e){
  var moved=panning&&(Math.abs(e.clientX-panning.mx)>4||Math.abs(e.clientY-panning.my)>4);
  panning=null;
  if(!moved){
    var p=s2w(e.offsetX,e.offsetY),hit=hitNode(p.x,p.y);
    if(hit){hlId=hit.id;openPanel(hit);}else{hlId=null;closePanel();}
    render();
  }
});
cv.addEventListener('mouseleave',function(){panning=null;});
cv.addEventListener('wheel',function(e){
  e.preventDefault();
  var d=e.deltaY>0?.88:1.14;
  tx=e.offsetX-d*(e.offsetX-tx);ty=e.offsetY-d*(e.offsetY-ty);
  sc=Math.max(.04,Math.min(6,sc*d));
  document.getElementById('sb-z').textContent=Math.round(sc*100)+'%';render();
},{passive:false});

function s2w(sx,sy){return{x:(sx-tx)/sc,y:(sy-ty)/sc};}
function hitNode(wx,wy){
  for(var i=RAW.nodes.length-1;i>=0;i--){
    var n=RAW.nodes[i];if(!isVis(n))continue;
    if(wx>=n.x&&wx<=n.x+n.w&&wy>=n.y&&wy<=n.y+n.h)return n;
  }
  return null;
}

// ── Panel ────────────────────────────────────────────────────────────────────
function openPanel(n){
  var p=document.getElementById('panel');p.classList.add('open');
  document.getElementById('ph-title').textContent=n.id;
  document.getElementById('ph-meta').textContent=
    TL[n.type]+' \u2022 Level '+n.level+
    ' \u2022 '+EI[n.id].length+' inputs \u2022 '+EO[n.id].length+' outputs';
  document.getElementById('ph-meta').style.color=TC[n.type]+'88';
  buildSec('src',EI[n.id],n.id,true);
  buildSec('dst',EO[n.id],n.id,false);
}

function buildSec(sec,edges,nodeId,isSource){
  document.getElementById('cnt-'+sec).textContent=edges.length;
  var body=document.getElementById('body-'+sec);
  body.innerHTML='';
  if(!edges.length){
    var em=document.createElement('div');
    em.style.cssText='padding:8px 14px;color:#1e2d3a;font-size:.72rem;font-style:italic;';
    em.textContent=isSource?'No source tables (base/external input)':'No downstream tables (leaf output)';
    body.appendChild(em);return;
  }

  edges.forEach(function(e){
    var otherId=isSource?e.from:e.to;
    var other=NM[otherId];
    var type=other?other.type:'ext';
    var col=TC[type]||'#718096';

    // Determine which alias belongs to which side
    var thisAlias=isSource?e.src_alias:e.tgt_alias;
    var otherAlias=isSource?e.src_alias:e.tgt_alias;
    // src_alias = alias of "from" table, tgt_alias = alias of "to" table
    var fromAlias=e.src_alias||'';
    var toAlias=e.tgt_alias||'';
    var displayAlias=isSource?fromAlias:toAlias;

    var row=document.createElement('div');row.className='trow';
    row.addEventListener('click',function(){
      var target=NM[otherId];
      if(target){hlId=otherId;openPanel(target);focusNode(otherId);render();}
    });

    // Top line
    var top=document.createElement('div');top.className='trow-top';
    var dot=document.createElement('div');dot.className='tdot';
    dot.style.background=col;dot.style.border='1px solid '+col+'66';
    var nm=document.createElement('div');nm.className='tname';nm.textContent=otherId;
    top.appendChild(dot);top.appendChild(nm);
    if(displayAlias){
      var ab=document.createElement('span');ab.className='talias';
      ab.textContent='alias: '+displayAlias;top.appendChild(ab);
    }
    row.appendChild(top);

    // Join keys table
    var jks=e.jk||[];
    if(jks.length){
      var tbl=document.createElement('table');tbl.className='jk-table';
      var thead=tbl.createTHead();var hr=thead.insertRow();
      var th1=document.createElement('th');th1.textContent='Source ('+( fromAlias||e.from.slice(0,12))+')';
      var th2=document.createElement('th');th2.style.cssText='width:18px;text-align:center;';th2.textContent='';
      var th3=document.createElement('th');th3.textContent='Destination ('+( toAlias||e.to.slice(0,12))+')';
      hr.appendChild(th1);hr.appendChild(th2);hr.appendChild(th3);
      var tbody=tbl.createTBody();
      jks.forEach(function(jk){
        var tr=tbody.insertRow();
        if(jk.left&&jk.right){
          var td1=tr.insertCell();td1.className='jk-src';td1.textContent=jk.left;
          var td2=tr.insertCell();td2.className='jk-eq';td2.textContent='=';
          var td3=tr.insertCell();td3.className='jk-dst';td3.textContent=jk.right;
        } else if(jk.col){
          var td1=tr.insertCell();td1.className='jk-src';td1.textContent=jk.col;td1.colSpan=3;
        }
      });
      row.appendChild(tbl);
    } else {
      var nj=document.createElement('div');nj.className='no-jk';
      nj.textContent='(no explicit join key \u2014 derived or subquery source)';
      row.appendChild(nj);
    }
    body.appendChild(row);
  });
}

function closePanel(){
  document.getElementById('panel').classList.remove('open');
  hlId=null;render();
}
function toggleSec(s){document.getElementById('hdr-'+s).classList.toggle('collapsed');}

function focusNode(id){
  var n=NM[id];if(!n)return;
  var r=cvA.getBoundingClientRect();
  tx=r.width/2-n.x*sc-(n.w/2)*sc;ty=r.height/2-n.y*sc-(n.h/2)*sc;
}

function resize(){var r=cvA.getBoundingClientRect();cv.width=r.width;cv.height=r.height;render();}
window.addEventListener('resize',resize);

function resetView(){hlId=null;closePanel();fitAll();}
function fitAll(){
  maxLv=99;
  ['ikg','tmp','ext'].forEach(function(t){showT[t]=true;document.getElementById('btn-'+t).classList.add('on');});
  fitVisible();
}
function fitLv(lv){maxLv=lv;fitVisible();}
function fitVisible(){
  var vis=RAW.nodes.filter(isVis);if(!vis.length)return;
  var x1=Math.min.apply(null,vis.map(function(n){return n.x;}));
  var x2=Math.max.apply(null,vis.map(function(n){return n.x+n.w;}));
  var y1=Math.min.apply(null,vis.map(function(n){return n.y;}));
  var y2=Math.max.apply(null,vis.map(function(n){return n.y+n.h;}));
  var pad=60;
  sc=Math.min((cv.width-pad*2)/(x2-x1||1),(cv.height-pad*2)/(y2-y1||1),1.4);
  tx=pad-x1*sc;ty=(cv.height-(y2-y1)*sc)/2-y1*sc;
  document.getElementById('sb-z').textContent=Math.round(sc*100)+'%';render();
}
function zoomIn(){applyZoom(1.2);}function zoomOut(){applyZoom(.83);}
function applyZoom(d){
  var cx=cv.width/2,cy=cv.height/2;tx=cx-d*(cx-tx);ty=cy-d*(cy-ty);
  sc=Math.max(.04,Math.min(6,sc*d));
  document.getElementById('sb-z').textContent=Math.round(sc*100)+'%';render();
}
function toggle(t){showT[t]=!showT[t];document.getElementById('btn-'+t).classList.toggle('on',showT[t]);render();}
function onSearch(){srchQ=document.getElementById('srch').value.toLowerCase().trim();render();}
function clearSearch(){srchQ='';document.getElementById('srch').value='';render();}

window.addEventListener('load',function(){
  resize();
  var mx=Math.max.apply(null,RAW.nodes.map(function(n){return n.level;}));
  document.getElementById('sp-n').textContent=RAW.nodes.length+' nodes';
  document.getElementById('sp-e').textContent=RAW.edges.length+' edges';
  document.getElementById('sp-l').textContent=(mx+1)+' levels';
  fitVisible();
});
"""

    return (
        "<!DOCTYPE html>\n<html lang='en'>\n<head>\n"
        "<meta charset='UTF-8'>\n"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>\n"
        "<title>account_profile_curr_ikg \u2014 ER Diagram</title>\n"
        "<style>\n" + css + "</style>\n</head>\n<body>\n"
        + body +
        "<script>\n" + js + "\n</script>\n</body>\n</html>\n"
    )





# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    # -----------------------------------------------------------------------
    # Resolve the folder where THIS script lives — SQL file must be there too
    # -----------------------------------------------------------------------
    script_dir = Path(__file__).resolve().parent

    print("=" * 60)
    print("  SQL Script Parser & ER Diagram Generator")
    print("=" * 60)
    print(f"\n  Script folder: {script_dir}")
    print("  Place your .sql file in the SAME folder as this script.\n")

    # List available .sql files to help the user
    sql_files = sorted(script_dir.glob("*.sql"))
    if sql_files:
        print("  SQL files found in this folder:")
        for f in sql_files:
            print(f"    • {f.name}")
    else:
        print("  ⚠  No .sql files detected in this folder yet.")
    print()

    # -----------------------------------------------------------------------
    # Ask user for the filename (loop until valid)
    # -----------------------------------------------------------------------
    while True:
        raw = input("  Enter SQL filename (e.g. my_script.sql): ").strip()
        if not raw:
            print("  ⚠  Filename cannot be empty. Please try again.\n")
            continue

        # Accept just the name or a relative/absolute path — always resolve
        # relative to the script's own directory first
        candidate = script_dir / raw
        if not candidate.exists():
            # Fallback: try as an absolute / cwd-relative path
            candidate = Path(raw).resolve()

        if candidate.exists() and candidate.is_file():
            sql_path = candidate
            print(f"\n  ✅ Found: {sql_path}\n")
            break
        else:
            print(f"  ❌ File not found: {script_dir / raw}")
            print("     Make sure the file is in the same folder as sql_parser.py\n")

    # -----------------------------------------------------------------------
    # Read & parse
    # -----------------------------------------------------------------------
    print(f"[*] Reading {sql_path.name} ({sql_path.stat().st_size:,} bytes) …")
    with open(sql_path, encoding="utf-8", errors="replace") as f:
        sql_text = f.read()

    print("[*] Parsing SQL — this may take a moment for large files …")
    parser = SQLParser(sql_text)
    parser.parse()
    summary = parser.summary()

    print("\n" + "=" * 60)
    print("  PARSE SUMMARY")
    print("=" * 60)
    print(f"  IKG tables created : {len(summary['created_tables'])}")
    print(f"  Temp tables        : {len(summary['temp_tables'])}")
    print(f"  External tables    : {len(summary['external_tables'])}")
    print(f"  Relationships      : {len(summary['relationships'])}")

    print("\n--- IKG Table Lineage (target ← sources) ---")
    for t in summary["created_tables"]:
        srcs = summary["table_sources"].get(t, [])
        print(f"  {t}")
        print(f"    ← {', '.join(sorted(srcs)) if srcs else '(no upstream sources)'}")

    # -----------------------------------------------------------------------
    # Write outputs to the SAME folder as the script
    # -----------------------------------------------------------------------
    stem = sql_path.stem

    json_out = script_dir / f"{stem}_lineage.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[✓] Lineage JSON  → {json_out}")

    html_out = script_dir / f"{stem}_er_diagram.html"
    html = build_er_html(summary)
    with open(html_out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[✓] ER Diagram    → {html_out}")
    print("\n  Open the HTML file in any modern browser to explore the diagram.")
    print("=" * 60)


if __name__ == "__main__":
    main()
