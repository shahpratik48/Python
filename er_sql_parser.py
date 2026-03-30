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
            join_cols = extract_join_conditions(stmt, table_name, self.alias_map)

            # Record relationship
            self.relationships.append({
                "from": table_name,
                "to": target,
                "join_cols": join_cols,
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
        # Deduplicate relationships by (from, to)
        seen = set()
        deduped = []
        for r in self.relationships:
            key = (r["from"], r["to"])
            if key not in seen:
                seen.add(key)
                deduped.append(r)

        return {
            "created_tables": sorted(self.created_tables),
            "temp_tables": sorted(self.temp_tables),
            "external_tables": sorted(self.external_tables),
            "relationships": deduped,
            "table_sources": {k: list(v) for k, v in self.table_sources.items()},
        }


# ---------------------------------------------------------------------------
# ER Diagram HTML Generator
# ---------------------------------------------------------------------------

ER_DIAGRAM_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SQL Table Lineage ER Diagram</title>
<style>
  :root {{
    --bg: #0d1117;
    --panel: #161b22;
    --border: #30363d;
    --text: #e6edf3;
    --muted: #8b949e;
    --accent1: #58a6ff;
    --accent2: #3fb950;
    --accent3: #d2a8ff;
    --accent4: #ffa657;
    --accent5: #ff7b72;
    --accent6: #79c0ff;
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Segoe UI', system-ui, sans-serif;
    overflow: hidden;
  }}

  #app {{
    display: flex;
    flex-direction: column;
    height: 100vh;
  }}

  /* ---- HEADER ---- */
  #header {{
    background: linear-gradient(135deg, #1a237e 0%, #0d47a1 40%, #006064 100%);
    padding: 14px 24px;
    display: flex;
    align-items: center;
    gap: 16px;
    box-shadow: 0 2px 12px rgba(0,0,0,.6);
    z-index: 10;
    flex-shrink: 0;
  }}
  #header h1 {{
    font-size: 1.25rem;
    font-weight: 700;
    letter-spacing: .5px;
  }}
  #header .subtitle {{
    font-size: .8rem;
    color: rgba(255,255,255,.6);
  }}
  .badge {{
    padding: 3px 10px;
    border-radius: 12px;
    font-size: .72rem;
    font-weight: 600;
    letter-spacing: .3px;
  }}
  .badge-blue  {{ background: rgba(88,166,255,.2); color: #58a6ff; border: 1px solid #58a6ff44; }}
  .badge-green {{ background: rgba(63,185,80,.2);  color: #3fb950; border: 1px solid #3fb95044; }}
  .badge-purple{{ background: rgba(210,168,255,.2);color: #d2a8ff; border: 1px solid #d2a8ff44; }}
  .badge-orange{{ background: rgba(255,166,87,.2); color: #ffa657; border: 1px solid #ffa65744; }}

  /* ---- TOOLBAR ---- */
  #toolbar {{
    background: var(--panel);
    border-bottom: 1px solid var(--border);
    padding: 8px 16px;
    display: flex;
    align-items: center;
    gap: 10px;
    flex-shrink: 0;
    flex-wrap: wrap;
  }}
  #toolbar button {{
    background: #21262d;
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 5px 14px;
    font-size: .8rem;
    cursor: pointer;
    transition: all .15s;
  }}
  #toolbar button:hover {{ background: #30363d; border-color: var(--accent1); }}
  #toolbar input[type=text] {{
    background: #21262d;
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 5px 12px;
    font-size: .8rem;
    width: 200px;
    outline: none;
  }}
  #toolbar input[type=text]:focus {{ border-color: var(--accent1); }}
  #toolbar label {{ font-size: .78rem; color: var(--muted); }}
  .sep {{ width: 1px; height: 24px; background: var(--border); margin: 0 4px; }}

  /* ---- LEGEND ---- */
  #legend {{
    display: flex; gap: 14px; margin-left: auto; flex-wrap: wrap;
  }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; font-size: .72rem; }}
  .legend-dot {{ width: 10px; height: 10px; border-radius: 50%; }}

  /* ---- MAIN AREA ---- */
  #main {{
    display: flex;
    flex: 1;
    overflow: hidden;
  }}

  /* ---- SIDEBAR ---- */
  #sidebar {{
    width: 280px;
    background: var(--panel);
    border-right: 1px solid var(--border);
    overflow-y: auto;
    flex-shrink: 0;
    padding: 12px;
  }}
  #sidebar h3 {{
    font-size: .78rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--muted);
    margin-bottom: 8px;
    margin-top: 12px;
  }}
  #sidebar h3:first-child {{ margin-top: 0; }}
  .table-list-item {{
    padding: 6px 10px;
    border-radius: 6px;
    cursor: pointer;
    font-size: .8rem;
    display: flex;
    align-items: center;
    gap: 8px;
    transition: background .1s;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .table-list-item:hover {{ background: #21262d; }}
  .table-list-item.active {{ background: #1f3958; border-left: 3px solid var(--accent1); }}
  .tbl-icon {{ width: 8px; height: 8px; border-radius: 2px; flex-shrink: 0; }}

  /* ---- CANVAS ---- */
  #canvas-wrapper {{
    flex: 1;
    position: relative;
    overflow: hidden;
  }}
  #canvas {{
    position: absolute;
    top: 0; left: 0;
    cursor: grab;
  }}
  #canvas:active {{ cursor: grabbing; }}

  /* ---- INFO PANEL ---- */
  #info-panel {{
    width: 300px;
    background: var(--panel);
    border-left: 1px solid var(--border);
    padding: 16px;
    overflow-y: auto;
    flex-shrink: 0;
    display: none;
  }}
  #info-panel.visible {{ display: block; }}
  #info-panel h2 {{ font-size: 1rem; margin-bottom: 8px; }}
  #info-panel .info-section {{ margin-top: 12px; }}
  #info-panel .info-section h4 {{
    font-size: .72rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--muted);
    margin-bottom: 6px;
  }}
  .rel-chip {{
    display: inline-flex; align-items: center; gap: 4px;
    background: #21262d;
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 3px 8px;
    font-size: .72rem;
    margin: 2px;
    color: var(--accent1);
    cursor: pointer;
  }}
  .rel-chip:hover {{ border-color: var(--accent1); }}

  #close-info {{
    float: right;
    background: none;
    border: none;
    color: var(--muted);
    cursor: pointer;
    font-size: 1.1rem;
  }}
  #close-info:hover {{ color: var(--text); }}

  /* ---- STATS BAR ---- */
  #stats-bar {{
    background: var(--panel);
    border-top: 1px solid var(--border);
    padding: 4px 16px;
    font-size: .72rem;
    color: var(--muted);
    display: flex;
    gap: 20px;
    flex-shrink: 0;
  }}
  #stats-bar span b {{ color: var(--text); }}
</style>
</head>
<body>
<div id="app">

<!-- HEADER -->
<div id="header">
  <div>
    <h1>📊 SQL Table Lineage &amp; ER Diagram</h1>
    <div class="subtitle">account_profile_curr_ikg pipeline · IKG Schema</div>
  </div>
  <span class="badge badge-blue" id="badge-created">-- IKG tables</span>
  <span class="badge badge-purple" id="badge-temp">-- Temp tables</span>
  <span class="badge badge-orange" id="badge-external">-- External tables</span>
  <span class="badge badge-green" id="badge-rels">-- Relationships</span>
</div>

<!-- TOOLBAR -->
<div id="toolbar">
  <button onclick="resetView()">⟳ Reset View</button>
  <button onclick="zoomIn()">＋ Zoom In</button>
  <button onclick="zoomOut()">－ Zoom Out</button>
  <div class="sep"></div>
  <label>Search:</label>
  <input type="text" id="search-box" placeholder="Filter tables…" oninput="filterTables()">
  <button onclick="clearSearch()">✕</button>
  <div class="sep"></div>
  <button onclick="toggleExternal()">Toggle External</button>
  <button onclick="toggleTemp()">Toggle Temp</button>
  <div class="sep"></div>
  <div id="legend">
    <div class="legend-item"><div class="legend-dot" style="background:#58a6ff"></div>IKG Table</div>
    <div class="legend-item"><div class="legend-dot" style="background:#d2a8ff"></div>Temp Table</div>
    <div class="legend-item"><div class="legend-dot" style="background:#ffa657"></div>External</div>
    <div class="legend-item"><div class="legend-dot" style="background:#3fb950"></div>Final Table</div>
  </div>
</div>

<!-- MAIN -->
<div id="main">
  <!-- SIDEBAR -->
  <div id="sidebar">
    <h3>IKG Tables</h3>
    <div id="list-created"></div>
    <h3>Temp Tables</h3>
    <div id="list-temp"></div>
    <h3>External Tables</h3>
    <div id="list-external"></div>
  </div>

  <!-- CANVAS -->
  <div id="canvas-wrapper">
    <canvas id="canvas"></canvas>
  </div>

  <!-- INFO PANEL -->
  <div id="info-panel">
    <button id="close-info" onclick="closeInfo()">✕</button>
    <h2 id="info-title"></h2>
    <div class="info-section">
      <h4>Type</h4>
      <div id="info-type"></div>
    </div>
    <div class="info-section">
      <h4>Upstream Sources (inputs)</h4>
      <div id="info-upstream"></div>
    </div>
    <div class="info-section">
      <h4>Downstream Targets (outputs)</h4>
      <div id="info-downstream"></div>
    </div>
    <div class="info-section">
      <h4>Join Columns</h4>
      <div id="info-joinkeys"></div>
    </div>
  </div>
</div>

<!-- STATS BAR -->
<div id="stats-bar">
  <span>Tables: <b id="stat-tables">0</b></span>
  <span>Relationships: <b id="stat-rels">0</b></span>
  <span>Zoom: <b id="stat-zoom">100%</b></span>
  <span>Drag to pan · Scroll to zoom · Click node for details</span>
</div>

</div><!-- #app -->

<script>
// =========================================================================
// DATA (injected by Python)
// =========================================================================
const DATA = {DATA_PLACEHOLDER};

// =========================================================================
// GRAPH STATE
// =========================================================================
const canvas  = document.getElementById('canvas');
const ctx     = canvas.getContext('2d');
const wrapper = document.getElementById('canvas-wrapper');

let transform = {{ x: 0, y: 0, scale: 1 }};
let nodes  = [];   // {{ id, label, x, y, w, h, type, color, textColor }}
let edges  = [];   // {{ from, to, joinCols, color }}
let dragging = null;
let panStart = null;
let selectedNode = null;
let showExternal = true;
let showTemp     = true;
let highlightedNodes = new Set();

// ---- Color palette ----
const COLORS = {{
  created:  {{ fill: '#0d2137', stroke: '#58a6ff', text: '#c9d1d9', hdr: '#58a6ff' }},
  temp:     {{ fill: '#1a0d2b', stroke: '#d2a8ff', text: '#c9d1d9', hdr: '#d2a8ff' }},
  external: {{ fill: '#1a1200', stroke: '#ffa657', text: '#c9d1d9', hdr: '#ffa657' }},
  final:    {{ fill: '#0a2119', stroke: '#3fb950', text: '#e6edf3', hdr: '#3fb950' }},
}};

// =========================================================================
// INIT
// =========================================================================
function init() {{
  buildGraph();
  populateSidebar();
  updateBadges();
  resizeCanvas();
  resetView();
  render();
  requestAnimationFrame(animLoop);
}}

function buildGraph() {{
  const allTables = new Set([
    ...DATA.created_tables,
    ...DATA.temp_tables,
    ...DATA.external_tables,
  ]);

  const finalTable = 'account_profile_curr_ikg_temp_auto';

  // Create nodes
  allTables.forEach(tbl => {{
    const type = DATA.created_tables.includes(tbl) ? (tbl === finalTable ? 'final' : 'created')
               : DATA.temp_tables.includes(tbl)    ? 'temp'
               : 'external';
    nodes.push({{
      id: tbl,
      label: tbl,
      x: Math.random() * 2400,
      y: Math.random() * 1800,
      w: Math.max(180, tbl.length * 7.2 + 24),
      h: 48,
      type,
      color: COLORS[type],
    }});
  }});

  // Create edges (deduplicate)
  const seen = new Set();
  DATA.relationships.forEach(rel => {{
    const key = rel.from + '→' + rel.to;
    if (seen.has(key) || !allTables.has(rel.from) || !allTables.has(rel.to)) return;
    seen.add(key);
    const isExt = DATA.external_tables.includes(rel.from);
    edges.push({{
      from: rel.from,
      to: rel.to,
      joinCols: rel.join_cols || [],
      color: isExt ? '#ffa65755' : '#58a6ff55',
    }});
  }});

  layoutGraph();
}}

// =========================================================================
// LAYOUT — Layered / force-directed hybrid
// =========================================================================
function layoutGraph() {{
  // Compute topological levels
  const inDegree = {{}};
  const outEdges = {{}};
  nodes.forEach(n => {{ inDegree[n.id] = 0; outEdges[n.id] = []; }});
  edges.forEach(e => {{
    if (inDegree[e.to] !== undefined) inDegree[e.to]++;
    if (outEdges[e.from]) outEdges[e.from].push(e.to);
  }});

  const levels = {{}};
  const queue = nodes.filter(n => inDegree[n.id] === 0).map(n => n.id);
  const processed = new Set();
  queue.forEach(id => {{ levels[id] = 0; }});

  while (queue.length) {{
    const cur = queue.shift();
    if (processed.has(cur)) continue;
    processed.add(cur);
    (outEdges[cur] || []).forEach(nxt => {{
      levels[nxt] = Math.max(levels[nxt] || 0, (levels[cur] || 0) + 1);
      queue.push(nxt);
    }});
  }}

  // Group by level
  const byLevel = {{}};
  nodes.forEach(n => {{
    const lv = levels[n.id] ?? 99;
    if (!byLevel[lv]) byLevel[lv] = [];
    byLevel[lv].push(n);
  }});

  const COL_W = 240, ROW_H = 80, MARGIN_X = 60, MARGIN_Y = 60;
  Object.keys(byLevel).sort((a,b) => +a - +b).forEach((lv, li) => {{
    const col = byLevel[lv];
    col.forEach((n, ri) => {{
      n.x = MARGIN_X + li * COL_W;
      n.y = MARGIN_Y + ri * ROW_H;
    }});
  }});

  // Run a few force iterations to spread nodes
  for (let iter = 0; iter < 60; iter++) {{
    forceStep();
  }}
}}

function forceStep() {{
  const repulsion = 800, attraction = 0.05, damping = 0.8;
  const forces = {{}};
  nodes.forEach(n => {{ forces[n.id] = {{ x: 0, y: 0 }}; }});

  // Repulsion between all pairs
  for (let i = 0; i < nodes.length; i++) {{
    for (let j = i+1; j < nodes.length; j++) {{
      const a = nodes[i], b = nodes[j];
      const dx = b.x - a.x || 0.1, dy = b.y - a.y || 0.1;
      const dist = Math.sqrt(dx*dx + dy*dy) || 1;
      const force = repulsion / (dist * dist);
      forces[a.id].x -= force * dx / dist;
      forces[a.id].y -= force * dy / dist;
      forces[b.id].x += force * dx / dist;
      forces[b.id].y += force * dy / dist;
    }}
  }}

  // Attraction along edges
  edges.forEach(e => {{
    const a = nodes.find(n => n.id === e.from);
    const b = nodes.find(n => n.id === e.to);
    if (!a || !b) return;
    const dx = b.x - a.x, dy = b.y - a.y;
    forces[a.id].x += attraction * dx;
    forces[a.id].y += attraction * dy;
    forces[b.id].x -= attraction * dx;
    forces[b.id].y -= attraction * dy;
  }});

  nodes.forEach(n => {{
    n.x += forces[n.id].x * damping;
    n.y += forces[n.id].y * damping;
  }});
}}

// =========================================================================
// RENDER
// =========================================================================
let _raf = 0;
function animLoop() {{
  render();
  _raf = requestAnimationFrame(animLoop);
}}

function render() {{
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  ctx.save();
  ctx.translate(transform.x, transform.y);
  ctx.scale(transform.scale, transform.scale);

  // Draw grid
  drawGrid();

  // Draw edges first
  const visibleIds = getVisibleIds();
  edges.forEach(e => {{
    if (!visibleIds.has(e.from) || !visibleIds.has(e.to)) return;
    drawEdge(e, visibleIds);
  }});

  // Draw nodes
  nodes.forEach(n => {{
    if (!visibleIds.has(n.id)) return;
    drawNode(n);
  }});

  ctx.restore();
}}

function drawGrid() {{
  const gridSize = 40;
  const w = canvas.width / transform.scale;
  const h = canvas.height / transform.scale;
  const ox = -transform.x / transform.scale;
  const oy = -transform.y / transform.scale;

  ctx.strokeStyle = '#1a2030';
  ctx.lineWidth = 0.5;
  const startX = Math.floor(ox / gridSize) * gridSize;
  const startY = Math.floor(oy / gridSize) * gridSize;
  for (let x = startX; x < ox + w; x += gridSize) {{
    ctx.beginPath(); ctx.moveTo(x, oy); ctx.lineTo(x, oy + h); ctx.stroke();
  }}
  for (let y = startY; y < oy + h; y += gridSize) {{
    ctx.beginPath(); ctx.moveTo(ox, y); ctx.lineTo(ox + w, y); ctx.stroke();
  }}
}}

function drawEdge(e, visibleIds) {{
  const from = nodes.find(n => n.id === e.from);
  const to   = nodes.find(n => n.id === e.to);
  if (!from || !to) return;

  const isHighlighted = selectedNode &&
    (selectedNode === e.from || selectedNode === e.to);

  const x1 = from.x + from.w, y1 = from.y + from.h / 2;
  const x2 = to.x,            y2 = to.y   + to.h   / 2;
  const cp1x = x1 + (x2 - x1) * 0.4;
  const cp2x = x1 + (x2 - x1) * 0.6;

  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.bezierCurveTo(cp1x, y1, cp2x, y2, x2, y2);
  ctx.strokeStyle = isHighlighted ? '#58a6ff' : e.color;
  ctx.lineWidth   = isHighlighted ? 2 : 1;
  ctx.stroke();

  // Arrow head
  const ang = Math.atan2(y2 - (y1 + (y2-y1)*0.8), x2 - (x1 + (x2-x1)*0.8));
  const aSize = isHighlighted ? 8 : 6;
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - aSize * Math.cos(ang - 0.4), y2 - aSize * Math.sin(ang - 0.4));
  ctx.lineTo(x2 - aSize * Math.cos(ang + 0.4), y2 - aSize * Math.sin(ang + 0.4));
  ctx.closePath();
  ctx.fillStyle = isHighlighted ? '#58a6ff' : e.color;
  ctx.fill();
}}

function drawNode(n) {{
  const isSelected = selectedNode === n.id;
  const isHighlighted = highlightedNodes.has(n.id);
  const c = n.color;

  // Shadow / glow
  if (isSelected || isHighlighted) {{
    ctx.shadowColor = c.stroke;
    ctx.shadowBlur  = 15;
  }}

  // Node background
  ctx.fillStyle = c.fill;
  roundRect(ctx, n.x, n.y, n.w, n.h, 8);
  ctx.fill();

  // Border
  ctx.strokeStyle = isSelected ? '#ffffff' : c.stroke;
  ctx.lineWidth   = isSelected ? 2.5 : 1.5;
  roundRect(ctx, n.x, n.y, n.w, n.h, 8);
  ctx.stroke();

  // Header accent line
  ctx.fillStyle = c.hdr;
  roundRect(ctx, n.x, n.y, n.w, 3, {{ tl: 8, tr: 8, bl: 0, br: 0 }});
  ctx.fill();

  ctx.shadowBlur = 0;

  // Label
  ctx.fillStyle = c.text;
  ctx.font = 'bold 11px "Segoe UI", system-ui, sans-serif';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';

  // Truncate label if too long
  let label = n.label;
  const maxW = n.w - 16;
  while (ctx.measureText(label).width > maxW && label.length > 6) {{
    label = label.slice(0, -4) + '…';
  }}
  ctx.fillText(label, n.x + 10, n.y + n.h / 2 + 2);

  // Type badge (small, right-aligned)
  const badge = n.type === 'final' ? 'FINAL' : n.type === 'temp' ? 'TMP' : n.type === 'external' ? 'EXT' : 'IKG';
  ctx.font = '8px "Segoe UI", system-ui, sans-serif';
  ctx.fillStyle = c.hdr + 'aa';
  ctx.textAlign = 'right';
  ctx.fillText(badge, n.x + n.w - 6, n.y + n.h / 2 + 2);
}}

function roundRect(ctx, x, y, w, h, r) {{
  if (typeof r === 'number') r = {{ tl: r, tr: r, bl: r, br: r }};
  ctx.beginPath();
  ctx.moveTo(x + r.tl, y);
  ctx.lineTo(x + w - r.tr, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r.tr);
  ctx.lineTo(x + w, y + h - r.br);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r.br, y + h);
  ctx.lineTo(x + r.bl, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r.bl);
  ctx.lineTo(x, y + r.tl);
  ctx.quadraticCurveTo(x, y, x + r.tl, y);
  ctx.closePath();
}}

// =========================================================================
// VISIBILITY FILTER
// =========================================================================
function getVisibleIds() {{
  const ids = new Set();
  nodes.forEach(n => {{
    if (n.type === 'external' && !showExternal) return;
    if (n.type === 'temp'     && !showTemp)     return;
    ids.add(n.id);
  }});
  return ids;
}}

// =========================================================================
// INTERACTION
// =========================================================================
function resizeCanvas() {{
  const rect = wrapper.getBoundingClientRect();
  canvas.width  = rect.width;
  canvas.height = rect.height;
}}

window.addEventListener('resize', () => {{ resizeCanvas(); }});

canvas.addEventListener('mousedown', e => {{
  const pos = screenToWorld(e.offsetX, e.offsetY);
  const hit = nodes.find(n => pos.x >= n.x && pos.x <= n.x + n.w &&
                               pos.y >= n.y && pos.y <= n.y + n.h);
  if (hit) {{
    dragging = {{ node: hit, ox: pos.x - hit.x, oy: pos.y - hit.y }};
    selectNode(hit.id);
  }} else {{
    panStart = {{ mx: e.clientX, my: e.clientY, tx: transform.x, ty: transform.y }};
  }}
}});

canvas.addEventListener('mousemove', e => {{
  if (dragging) {{
    const pos = screenToWorld(e.offsetX, e.offsetY);
    dragging.node.x = pos.x - dragging.ox;
    dragging.node.y = pos.y - dragging.oy;
  }} else if (panStart) {{
    transform.x = panStart.tx + (e.clientX - panStart.mx);
    transform.y = panStart.ty + (e.clientY - panStart.my);
  }}
}});

canvas.addEventListener('mouseup', () => {{ dragging = null; panStart = null; }});
canvas.addEventListener('mouseleave', () => {{ dragging = null; panStart = null; }});

canvas.addEventListener('wheel', e => {{
  e.preventDefault();
  const delta = e.deltaY > 0 ? 0.9 : 1.1;
  const mx = e.offsetX, my = e.offsetY;
  transform.x = mx - delta * (mx - transform.x);
  transform.y = my - delta * (my - transform.y);
  transform.scale *= delta;
  transform.scale = Math.max(0.08, Math.min(4, transform.scale));
  document.getElementById('stat-zoom').textContent =
    Math.round(transform.scale * 100) + '%';
}}, {{ passive: false }});

function screenToWorld(sx, sy) {{
  return {{
    x: (sx - transform.x) / transform.scale,
    y: (sy - transform.y) / transform.scale,
  }};
}}

// =========================================================================
// NODE SELECTION / INFO PANEL
// =========================================================================
function selectNode(id) {{
  selectedNode = id;
  const n = nodes.find(n => n.id === id);
  if (!n) return;

  // Highlight connected nodes
  highlightedNodes = new Set([id]);
  edges.forEach(e => {{
    if (e.from === id) highlightedNodes.add(e.to);
    if (e.to   === id) highlightedNodes.add(e.from);
  }});

  // Info panel
  document.getElementById('info-title').textContent = id;
  document.getElementById('info-type').textContent =
    n.type === 'final' ? '🏁 Final Output Table' :
    n.type === 'temp'  ? '⏳ Temporary Table' :
    n.type === 'external' ? '🌐 External / EDW Table' : '📦 IKG Table';

  const upstream = edges.filter(e => e.to === id).map(e => e.from);
  const downstream = edges.filter(e => e.from === id).map(e => e.to);
  const joinCols = edges
    .filter(e => e.from === id || e.to === id)
    .flatMap(e => e.joinCols.map(j => (Array.isArray(j) ? j.join(' = ') : j)));

  renderChips('info-upstream', upstream, 'from');
  renderChips('info-downstream', downstream, 'to');

  const jkDiv = document.getElementById('info-joinkeys');
  jkDiv.innerHTML = joinCols.length
    ? joinCols.slice(0,20).map(c =>
        `<code style="display:block;font-size:.7rem;padding:2px 0;color:#8b949e">${{c}}</code>`
      ).join('')
    : '<span style="color:var(--muted);font-size:.78rem">—</span>';

  document.getElementById('info-panel').classList.add('visible');

  // Highlight in sidebar
  document.querySelectorAll('.table-list-item').forEach(el => {{
    el.classList.toggle('active', el.dataset.id === id);
  }});
}}

function renderChips(containerId, ids, dir) {{
  const el = document.getElementById(containerId);
  if (!ids.length) {{
    el.innerHTML = '<span style="color:var(--muted);font-size:.78rem">—</span>';
    return;
  }}
  el.innerHTML = ids.map(id =>
    `<span class="rel-chip" onclick="selectNode('${{id}}')">${{id}}</span>`
  ).join('');
}}

function closeInfo() {{
  selectedNode = null;
  highlightedNodes.clear();
  document.getElementById('info-panel').classList.remove('visible');
  document.querySelectorAll('.table-list-item').forEach(el => el.classList.remove('active'));
}}

// =========================================================================
// SIDEBAR
// =========================================================================
function populateSidebar() {{
  fillList('list-created',  DATA.created_tables,  'created');
  fillList('list-temp',     DATA.temp_tables,      'temp');
  fillList('list-external', DATA.external_tables,  'external');
  document.getElementById('stat-tables').textContent =
    DATA.created_tables.length + DATA.temp_tables.length + DATA.external_tables.length;
  document.getElementById('stat-rels').textContent = DATA.relationships.length;
}}

function fillList(containerId, tables, type) {{
  const el = document.getElementById(containerId);
  const colorMap = {{ created: '#58a6ff', temp: '#d2a8ff', external: '#ffa657', final: '#3fb950' }};
  el.innerHTML = tables.map(t => {{
    const col = t === 'account_profile_curr_ikg_temp_auto' ? colorMap.final : colorMap[type];
    return `<div class="table-list-item" data-id="${{t}}" onclick="selectNode('${{t}}'); focusNode('${{t}}')">\
<div class="tbl-icon" style="background:${{col}}"></div>\
<span title="${{t}}">${{t}}</span></div>`;
  }}).join('');
}}

function focusNode(id) {{
  const n = nodes.find(n => n.id === id);
  if (!n) return;
  const cx = canvas.width / 2, cy = canvas.height / 2;
  transform.x = cx - n.x * transform.scale - (n.w / 2) * transform.scale;
  transform.y = cy - n.y * transform.scale - (n.h / 2) * transform.scale;
}}

// =========================================================================
// BADGES
// =========================================================================
function updateBadges() {{
  document.getElementById('badge-created').textContent  = DATA.created_tables.length  + ' IKG tables';
  document.getElementById('badge-temp').textContent     = DATA.temp_tables.length     + ' Temp tables';
  document.getElementById('badge-external').textContent = DATA.external_tables.length + ' External tables';
  document.getElementById('badge-rels').textContent     = DATA.relationships.length   + ' Relationships';
}}

// =========================================================================
// TOOLBAR ACTIONS
// =========================================================================
function resetView() {{
  // Center all nodes
  if (!nodes.length) return;
  const xs = nodes.map(n => n.x), ys = nodes.map(n => n.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs.map((x,i) => x + nodes[i].w));
  const minY = Math.min(...ys), maxY = Math.max(...ys.map((y,i) => y + nodes[i].h));
  const W = canvas.width, H = canvas.height;
  const scaleX = W / (maxX - minX + 80);
  const scaleY = H / (maxY - minY + 80);
  transform.scale = Math.min(scaleX, scaleY, 1);
  transform.x = (W - (maxX + minX) * transform.scale) / 2;
  transform.y = (H - (maxY + minY) * transform.scale) / 2;
  document.getElementById('stat-zoom').textContent = Math.round(transform.scale * 100) + '%';
}}

function zoomIn()  {{ applyZoom(1.2); }}
function zoomOut() {{ applyZoom(0.8); }}
function applyZoom(delta) {{
  const cx = canvas.width / 2, cy = canvas.height / 2;
  transform.x = cx - delta * (cx - transform.x);
  transform.y = cy - delta * (cy - transform.y);
  transform.scale = Math.max(0.08, Math.min(4, transform.scale * delta));
  document.getElementById('stat-zoom').textContent = Math.round(transform.scale * 100) + '%';
}}

function toggleExternal() {{
  showExternal = !showExternal;
  render();
}}
function toggleTemp() {{
  showTemp = !showTemp;
  render();
}}

function filterTables() {{
  const q = document.getElementById('search-box').value.toLowerCase();
  document.querySelectorAll('.table-list-item').forEach(el => {{
    el.style.display = el.dataset.id.includes(q) ? '' : 'none';
  }});
}}
function clearSearch() {{
  document.getElementById('search-box').value = '';
  filterTables();
}}

// =========================================================================
// START
// =========================================================================
window.addEventListener('load', init);
</script>
</body>
</html>
"""


def build_er_html(summary: dict) -> str:
    """Inject parsed data into the ER diagram HTML template."""
    js_data = {
        "created_tables": summary["created_tables"],
        "temp_tables": summary["temp_tables"],
        "external_tables": summary["external_tables"],
        "relationships": [
            {
                "from": r["from"],
                "to": r["to"],
                "join_cols": [
                    list(jc) if isinstance(jc, tuple) else jc
                    for jc in r.get("join_cols", [])
                ],
                "is_external": r.get("is_external", False),
            }
            for r in summary["relationships"]
        ],
    }
    data_str = json.dumps(js_data, indent=2)

    # Step 1: inject the JSON data block
    html = ER_DIAGRAM_TEMPLATE.replace("{DATA_PLACEHOLDER}", data_str)

    # Step 2: unescape {{ }} that Python uses to escape braces inside the
    # template string.  Those appear throughout the JS code and must become
    # real { } for the browser.  We protect the already-injected JSON data
    # block (which contains real braces) with sentinels so it is not touched.
    SENT_S = "<<<DATA_START>>>"
    SENT_E = "<<<DATA_END>>>"

    marker_start = "const DATA = "
    marker_end   = "\n};\n"

    ds = html.find(marker_start)
    de = html.find(marker_end, ds) + len(marker_end)

    protected = html[:ds] + SENT_S + html[ds:de] + SENT_E + html[de:]

    before, rest    = protected.split(SENT_S, 1)
    data_block, after = rest.split(SENT_E, 1)

    before = before.replace("{{", "{").replace("}}", "}")
    after  = after.replace("{{", "{").replace("}}", "}")

    return before + data_block + after


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
