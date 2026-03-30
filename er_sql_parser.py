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


import re


import re as _re


import re as _re

def extract_table_columns(sql_text: str) -> dict:
    """
    Parse SQL and extract output columns for every CREATE TABLE / INSERT INTO.
    Returns dict: table_name -> [col1, col2, ...]
    """
    clean = _re.sub(r'/\*.*?\*/', ' ', sql_text, flags=_re.DOTALL)
    clean = _re.sub(r'--[^\n]*', ' ', clean)
    result = {}

    # ── CREATE TABLE ... AS ────────────────────────────────────────────────
    create_pat = _re.compile(
        r'CREATE\s+(?:TEMP(?:ORARY)?\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?'
        r'(?:\{\{[^}]+\}\}\.)?(\w+)\s+AS\s*',
        _re.IGNORECASE
    )
    for m in create_pat.finditer(clean):
        tbl = m.group(1).lower()
        body = clean[m.end():]
        # Skip WITH CTE preamble to reach the main SELECT
        with_m = _re.match(r'\s*WITH\b', body, _re.IGNORECASE)
        if with_m:
            # Advance past all CTE definitions to the outermost SELECT
            depth, pos = 0, 0
            while pos < len(body):
                ch = body[pos]
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                elif depth == 0 and body[pos:pos+6].upper() == 'SELECT':
                    break
                pos += 1
            body = body[pos:]
        sel_m = _re.search(r'\bSELECT\b', body, _re.IGNORECASE)
        if not sel_m:
            continue
        cols = _select_cols(body[sel_m.start():])
        if cols:
            result[tbl] = cols

    # ── INSERT INTO tbl SELECT ... ─────────────────────────────────────────
    # Use a more robust pattern that handles whitespace/newlines between table name and SELECT
    insert_pat = _re.compile(
        r'INSERT\s+INTO\s+(?:\{\{[^}]+\}\}\.)?(\w+)\s*\n?\s*(SELECT\b)',
        _re.IGNORECASE | _re.DOTALL
    )
    for m in insert_pat.finditer(clean):
        tbl = m.group(1).lower()
        if tbl not in result:
            cols = _select_cols(clean[m.start(2):])
            if cols:
                result[tbl] = cols

    return result


def _select_cols(body: str) -> list:
    """Extract output column names from text starting with SELECT."""
    # Strip SELECT [DISTINCT|ALL]
    body = _re.sub(r'^\s*SELECT\s+(?:ALL\s+|DISTINCT\s+)?', '', body, flags=_re.IGNORECASE)

    # Find the FROM at depth 0 that ends the column list
    depth, from_pos = 0, len(body)
    upper = body.upper()
    i = 0
    while i < len(body):
        ch = body[i]
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth < 0:
                from_pos = i
                break
        elif depth == 0 and _re.match(r'\bFROM\b', upper[i:]):
            from_pos = i
            break
        i += 1

    col_section = body[:from_pos]

    # Split by top-level commas
    cols_raw = []
    depth, buf = 0, []
    for ch in col_section:
        if ch == '(':
            depth += 1; buf.append(ch)
        elif ch == ')':
            depth -= 1; buf.append(ch)
        elif ch == ',' and depth == 0:
            cols_raw.append(''.join(buf).strip()); buf = []
        else:
            buf.append(ch)
    if buf:
        cols_raw.append(''.join(buf).strip())

    SKIP = {
        'null','true','false','desc','asc','int','text','date','varchar',
        'numeric','bigint','boolean','timestamp','char','n','y',
        'profile_date',  # keep this – actually useful, remove from skip
    }
    SKIP = {
        'null','true','false','desc','asc','int','text','date','varchar',
        'numeric','bigint','boolean','timestamp','char',
    }

    result, seen = [], set()
    for col in cols_raw:
        col = col.strip()
        if not col:
            continue
        # Explicit AS alias — highest priority
        as_m = _re.search(r'\bAS\s+(\w+)\s*$', col, _re.IGNORECASE)
        if as_m:
            name = as_m.group(1).lower()
        else:
            # Last identifier after dot or whitespace: handles "alias.col_name"
            last = _re.search(r'(?:[.\s,(]|^)(\w+)\s*$', col)
            name = last.group(1).lower() if last else None

        if name and len(name) > 1 and name not in SKIP and name not in seen:
            seen.add(name)
            result.append(name)

    return result

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
    Extract JOIN ON conditions that specifically involve source_table.
    Returns only pairs where at least one alias resolves to source_table,
    preventing conditions from other JOINs bleeding into this edge.
    """
    import re as _r

    # Build reverse: table_name -> set of aliases
    tbl_to_aliases = {}
    for alias, tbl in aliases.items():
        tbl_to_aliases.setdefault(tbl, set()).add(alias)
    src_aliases = tbl_to_aliases.get(source_table, set()) | {source_table}

    results = []
    # Pattern: JOIN <table> [alias] ON <condition> until next clause/join
    join_on_pat = _r.compile(
        r'\bJOIN\s+(?:\{\{[^}]+\}\}\.)?(\w+)(?:\s+(?:AS\s+)?(\w+))?\s+ON\b(.+?)'
        r'(?=\b(?:LEFT|RIGHT|INNER|FULL|CROSS|OUTER|JOIN|WHERE|GROUP\s+BY|'
        r'ORDER\s+BY|HAVING|LIMIT|DISTRIBUTED|;)\b|$)',
        _r.IGNORECASE | _r.DOTALL
    )
    eq_pat = _r.compile(r'(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)')

    for m in join_on_pat.finditer(sql_block):
        on_clause = m.group(3)
        for eq in eq_pat.finditer(on_clause):
            la = eq.group(1).lower()
            lc = eq.group(2)
            ra = eq.group(3).lower()
            rc = eq.group(4)
            ltbl = aliases.get(la, la)
            rtbl = aliases.get(ra, ra)
            if ltbl == source_table or rtbl == source_table or la in src_aliases or ra in src_aliases:
                results.append((la + '.' + lc, ra + '.' + rc))

    seen, deduped = set(), []
    for pair in results:
        key = (pair[0].lower(), pair[1].lower())
        if key not in seen:
            seen.add(key)
            deduped.append(pair)
    return deduped



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
            "final_table": "",  # populated by auto_detect_final_table if empty
        }


import json

def auto_detect_final_table(summary: dict, sql_text: str = "") -> str:
    """
    Auto-detect the final output table using multiple signals:
    1. Last CREATE TABLE in the SQL script (most reliable for pipelines)
    2. Leaf node with the most upstream dependencies (fallback)
    """
    created = list(summary['created_tables'])  # already sorted alphabetically

    if not created:
        return ''

    # Signal 1: Find the LAST created table by position in SQL script
    if sql_text:
        import re
        # Find all CREATE TABLE statements and their positions
        pattern = re.compile(
            r'CREATE\s+(?:TEMP(?:ORARY)?\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:\{\{[^}]+\}\}\.)?(\w+)',
            re.IGNORECASE
        )
        positions = []
        for m in pattern.finditer(sql_text):
            tbl = m.group(1).lower()
            if tbl in summary['created_tables']:
                positions.append((m.start(), tbl))
        if positions:
            # The last CREATE TABLE that is a proper IKG table (not temp/internal)
            positions.sort(key=lambda x: x[0])
            # Try last non-temp, non-internal table
            for _, tbl in reversed(positions):
                if tbl in set(summary['created_tables']) and tbl not in set(summary['temp_tables']):
                    return tbl
            # Fallback to absolute last
            return positions[-1][1]

    # Signal 2: Leaf node (never used as source) with most upstream deps
    all_sources = set(r['from'] for r in summary['relationships'])
    all_targets = set(r['to']   for r in summary['relationships'])
    created_set = set(summary['created_tables'])

    leaf_tables = [t for t in created_set if t not in all_sources and t in all_targets]

    if not leaf_tables:
        # No leaf found — use the table with most incoming edges
        from collections import Counter
        in_deg = Counter(r['to'] for r in summary['relationships'])
        candidates = sorted(created_set, key=lambda t: -in_deg.get(t, 0))
        return candidates[0] if candidates else created[-1]

    if len(leaf_tables) == 1:
        return leaf_tables[0]

    # Multiple leaves — pick the one with most upstream nodes
    from collections import deque
    rel_up = {}
    for r in summary['relationships']:
        rel_up.setdefault(r['to'], []).append(r['from'])

    def count_upstream(root):
        visited, q = set(), deque([root])
        while q:
            t = q.popleft()
            if t in visited: continue
            visited.add(t)
            for s in rel_up.get(t, []): q.append(s)
        return len(visited)

    leaf_tables.sort(key=lambda t: -count_upstream(t))
    return leaf_tables[0]



def build_er_html(summary: dict) -> str:
    import json as _json
    import re as _re
    from collections import deque as _deque, defaultdict as _dd

    FINAL = summary.get('final_table') or auto_detect_final_table(
        summary, summary.get('_sql_text', ''))

    # ── Collect ALL tables from the script ────────────────────────────────────
    created_set  = set(summary['created_tables'])
    temp_set     = set(summary['temp_tables'])
    ext_set      = set(summary['external_tables'])

    # Collect CTE names to exclude (they are inline aliases, not real tables)
    cte_names = set()
    sql_text = summary.get('_sql_text', '')
    clean_sql = _re.sub(r'/\*.*?\*/', ' ', sql_text, flags=_re.DOTALL)
    clean_sql = _re.sub(r'--[^\n]*', ' ', clean_sql)
    for m in _re.finditer(r'\bWITH\b(.*?)(?=\bSELECT\b)', clean_sql,
                          _re.IGNORECASE | _re.DOTALL):
        for nm in _re.finditer(r'\b(\w+)\s+AS\s*\(', m.group(1), _re.IGNORECASE):
            cte_names.add(nm.group(1).lower())

    # All tables referenced anywhere (FROM/JOIN)
    skip_kw = {'select','where','on','set','lateral','only','rows','unnest',
                'values','null','true','false','current','row','all','distinct'}
    all_referenced = set()
    for m in _re.finditer(
            r'\b(?:FROM|JOIN)\s+(?:\{\{[^}]+\}\}\.)?(\w+)', clean_sql, _re.IGNORECASE):
        nm = m.group(1).lower()
        if nm not in skip_kw and nm not in cte_names:
            all_referenced.add(nm)

    all_tables = (created_set | temp_set | ext_set | all_referenced) - cte_names

    # ── Column enrichment ─────────────────────────────────────────────────────
    col_map = {}
    for extra_sql in summary.get('_extra_sql_texts', []):
        col_map.update(extract_table_columns(extra_sql))
    if sql_text:
        col_map.update(extract_table_columns(sql_text))

    # ── Deduplicated edge list ─────────────────────────────────────────────────
    ALIAS_BL = {
        'where','on','set','select','inner','left','right','full','outer','cross',
        'join','from','as','not','null','and','or','in','by','group','order',
        'having','limit','union','all','distinct','with','values','distributed',
        'using','for','case','when','then','else','end','exists','between',
        'like','ilike','is','true','false','into','insert','update','delete',
        'create','drop','alter','table','temp','temporary','partition','over',
        'window','rows','range','unbounded','preceding','following','current',
        'row','natural','lateral','recursive','only','returning',
    }
    def clean_alias(a):
        if not a: return ''
        a = a.strip().lower()
        if a in ALIAS_BL: return ''
        if not _re.match(r'^[a-z_][a-z0-9_]{0,29}$', a): return ''
        return a

    merged_edges = {}
    for r in summary['relationships']:
        src_t = r['from']
        tgt_t = r['to']
        if src_t not in all_tables or tgt_t not in all_tables:
            continue
        key = (src_t, tgt_t)
        if key not in merged_edges:
            merged_edges[key] = {
                'from':       src_t,
                'to':         tgt_t,
                'src_alias':  clean_alias(r.get('alias', '')),
                'tgt_alias':  clean_alias(r.get('target_alias', '')),
                'jk':         [],
                'is_external': r.get('is_external', False),
            }
        # Merge join keys
        existing = set(
            tuple(j) if isinstance(j, list) else j
            for j in merged_edges[key]['jk']
        )
        for jc in r.get('join_cols', []):
            if isinstance(jc, (list, tuple)) and len(jc) == 2:
                item = [str(jc[0]), str(jc[1])]
                k2 = tuple(item)
            else:
                item = str(jc)
                k2 = item
            if k2 not in existing:
                existing.add(k2)
                merged_edges[key]['jk'].append(
                    {'left': item[0], 'right': item[1]}
                    if isinstance(item, list) else {'col': item}
                )

    edges_list = list(merged_edges.values())

    # ── Topological levels for ALL tables ─────────────────────────────────────
    in_edges  = _dd(set)
    out_edges = _dd(set)
    for e in edges_list:
        in_edges[e['to']].add(e['from'])
        out_edges[e['from']].add(e['to'])

    in_deg = {t: len(in_edges[t]) for t in all_tables}
    queue  = _deque([t for t in all_tables if in_deg[t] == 0])
    level  = {t: 0 for t in queue}
    while queue:
        t = queue.popleft()
        for nxt in out_edges[t]:
            if nxt in all_tables:
                in_deg[nxt] = max(0, in_deg.get(nxt, 0) - 1)
                level[nxt]  = max(level.get(nxt, 0), level[t] + 1)
                if in_deg[nxt] == 0:
                    queue.append(nxt)
    for t in all_tables:
        if t not in level:
            level[t] = 0

    # ── Build node type ────────────────────────────────────────────────────────
    def node_type(n):
        if n == FINAL:        return 'final'
        if n in created_set:  return 'ikg'
        if n in temp_set:     return 'tmp'
        return 'ext'

    nodes_js = []
    for t in all_tables:
        nodes_js.append({
            'id':    t,
            'level': level.get(t, 0),
            'type':  node_type(t),
            'cols':  col_map.get(t, []),
        })

    edges_js = []
    for e in edges_list:
        edges_js.append({
            'from':        e['from'],
            'to':          e['to'],
            'jk':          e['jk'],
            'src_alias':   e['src_alias'],
            'tgt_alias':   e['tgt_alias'],
            'is_external': e['is_external'],
        })

    data_json = _json.dumps(
        {'nodes': nodes_js, 'edges': edges_js, 'final': FINAL},
        separators=(',', ':')
    )

    # ── CSS ───────────────────────────────────────────────────────────────────
    css = """
*, *::before, *::after { box-sizing:border-box; margin:0; padding:0; }
body { background:#090e18; color:#e2e8f0;
  font-family:'Segoe UI',system-ui,sans-serif;
  overflow:hidden; height:100vh; display:flex; flex-direction:column; }

#hdr { background:linear-gradient(90deg,#1a3356,#0d2040);
  padding:10px 18px; display:flex; align-items:center; gap:12px;
  border-bottom:1px solid #1a2840; flex-shrink:0; }
#hdr h1 { font-size:.92rem; font-weight:700; color:#5aa8e8; }
.sub { font-size:.67rem; color:#2d4060; margin-top:1px; }
.sp { padding:2px 9px; border-radius:20px; font-size:.65rem; font-weight:600; }
.sp-g { background:#0f3020; color:#4ec87a; border:1px solid #2a6040; }
.sp-b { background:#0f2045; color:#5aa8e8; border:1px solid #1e4080; }
.sp-p { background:#1e1040; color:#a070e8; border:1px solid #3a2070; }
.sp-o { background:#2a1500; color:#e09050; border:1px solid #603010; }

#tb { background:#0d1525; border-bottom:1px solid #1a2535;
  padding:5px 12px; display:flex; align-items:center; gap:6px;
  flex-shrink:0; flex-wrap:wrap; }
.btn { background:#111d30; color:#7a90a8; border:1px solid #1e2d40;
  border-radius:4px; padding:4px 11px; font-size:.71rem; cursor:pointer; transition:all .12s; }
.btn:hover { background:#1a2d44; color:#c8d8e8; }
.btn.on { background:#122040; color:#5aa8e8; border-color:#1e4070; }
.sep { width:1px; height:18px; background:#1a2535; margin:0 2px; }
#srch { background:#111d30; color:#e2e8f0; border:1px solid #1e2d40;
  border-radius:4px; padding:4px 10px; font-size:.71rem; width:175px; outline:none; }
#srch:focus { border-color:#2a5090; }
#srch::placeholder { color:#2a3848; }
#leg { display:flex; gap:10px; margin-left:auto; align-items:center; }
.li { display:flex; align-items:center; gap:4px; font-size:.65rem; color:#3a5068; }
.lb { width:12px; height:8px; border-radius:2px; }

#wrap { flex:1; position:relative; overflow:hidden; display:flex; }
#cva { flex:1; position:relative; overflow:hidden; background:#090e18; }
#cv { position:absolute; top:0; left:0; display:block; cursor:grab; }

#panel { width:420px; min-width:420px; background:#0b1422;
  border-left:1px solid #162030; display:none; flex-direction:column;
  overflow:hidden; flex-shrink:0; }
#panel.open { display:flex; }
#ph { background:#091020; padding:11px 13px 9px; border-bottom:1px solid #162030;
  flex-shrink:0; position:relative; }
#ph-title { font-size:.87rem; font-weight:700; color:#5aa8e8;
  word-break:break-all; margin-bottom:3px; padding-right:24px; }
#ph-meta { font-size:.67rem; }
#ph-x { position:absolute; right:12px; top:12px; background:none; border:none;
  color:#2a3848; cursor:pointer; font-size:1rem; transition:color .1s; }
#ph-x:hover { color:#e2e8f0; }
#pb { flex:1; overflow-y:auto; }
#pb::-webkit-scrollbar { width:4px; }
#pb::-webkit-scrollbar-track { background:#080f1c; }
#pb::-webkit-scrollbar-thumb { background:#1a2840; border-radius:3px; }

.sec { border-bottom:1px solid #0f1e30; }
.sechdr { padding:8px 13px; font-size:.66rem; font-weight:700; text-transform:uppercase;
  letter-spacing:.8px; display:flex; align-items:center; gap:7px;
  cursor:pointer; user-select:none; transition:background .1s; }
.sechdr:hover { background:#0d1a2a; }
.sechdr.cols-hdr { color:#3a6888; }
.sechdr.src-hdr  { color:#2a6840; }
.sechdr.dst-hdr  { color:#6a4010; }
.seccnt { margin-left:auto; background:#0f1e2e; color:#3a6888;
  font-size:.61rem; padding:1px 7px; border-radius:10px; font-weight:700; }
.arr { font-size:.62rem; color:#1e2d3e; transition:transform .15s; }
.sechdr.collapsed .arr { transform:rotate(-90deg); }
.secbody { display:block; }
.sechdr.collapsed + .secbody { display:none; }

.col-row { padding:4px 13px 4px 22px; display:flex; align-items:center; gap:8px;
  border-bottom:1px solid #0a1520; font-size:.73rem; }
.col-row:last-child { border-bottom:none; }
.col-idx { color:#1e3040; font-size:.6rem; min-width:18px; font-family:monospace; }
.col-name { color:#90b8d8; font-family:'Consolas','Courier New',monospace; }

.trow { padding:7px 11px 7px 15px; border-bottom:1px solid #0a1828;
  cursor:pointer; transition:background .1s; }
.trow:hover { background:#0d1c2e; }
.trow:last-child { border-bottom:none; }
.trow-top { display:flex; align-items:flex-start; gap:7px; margin-bottom:4px; }
.tdot { width:7px; height:7px; border-radius:2px; flex-shrink:0; margin-top:4px; }
.tname { font-size:.75rem; font-weight:600; color:#b0c8d8; flex:1; word-break:break-all; }
.talias { font-size:.62rem; color:#1e4060; background:#081522;
  border:1px solid #102030; border-radius:3px; padding:1px 6px;
  font-family:'Consolas','Courier New',monospace; white-space:nowrap; }
.jktbl { width:calc(100% - 14px); border-collapse:collapse; margin:2px 0 2px 14px; }
.jktbl th { font-size:.59rem; color:#1e4050; font-weight:700; text-transform:uppercase;
  letter-spacing:.4px; padding:2px 6px 2px 0; border-bottom:1px solid #0f1e2e; }
.jktbl td { font-size:.66rem; font-family:'Consolas','Courier New',monospace;
  padding:2px 6px 2px 0; vertical-align:top; border-bottom:1px solid #081520; }
.jktbl tr:last-child td { border-bottom:none; }
.jk-s { color:#4ec87a; } .jk-e { color:#1e2d3a; padding:0 2px; } .jk-d { color:#e8a040; }
.no-jk { font-size:.62rem; color:#162028; padding:2px 0 2px 14px; font-style:italic; }

#sb { background:#0d1525; border-top:1px solid #162030; padding:3px 13px;
  font-size:.65rem; color:#1e3040; display:flex; gap:16px; flex-shrink:0; }
#sb b { color:#2a4050; }
"""

    body = """
<div id="hdr">
  <div>
    <div id="h1">&#128202; """ + FINAL + """ &mdash; ER Diagram</div>
    <div class="sub">All tables, dependencies and relationships &bull; Drag boxes to reposition &bull; Click any table for details</div>
  </div>
  <span class="sp sp-b"  id="sp-ikg">-- IKG tables</span>
  <span class="sp sp-p"  id="sp-tmp">-- Temp tables</span>
  <span class="sp sp-o"  id="sp-ext">-- External tables</span>
  <span class="sp sp-g"  id="sp-rel">-- Relationships</span>
</div>
<div id="tb">
  <button class="btn" onclick="resetView()">&#8635; Reset</button>
  <button class="btn" onclick="zoomIn()">+ Zoom</button>
  <button class="btn" onclick="zoomOut()">&#8722; Zoom</button>
  <div class="sep"></div>
  <input id="srch" placeholder="&#128269; Search..." oninput="onSearch()">
  <button class="btn" onclick="clearSearch()">&#10005;</button>
  <div class="sep"></div>
  <button class="btn on" id="btn-ikg" onclick="toggle('ikg')">IKG</button>
  <button class="btn on" id="btn-tmp" onclick="toggle('tmp')">Temp</button>
  <button class="btn on" id="btn-ext" onclick="toggle('ext')">External</button>
  <div class="sep"></div>
  <button class="btn" onclick="fitLv(1)">Lv 1</button>
  <button class="btn" onclick="fitLv(3)">Lv 1-3</button>
  <button class="btn" onclick="fitLv(6)">Lv 1-6</button>
  <button class="btn" onclick="fitAll()">All</button>
  <div id="leg">
    <div class="li"><div class="lb" style="background:#2ecc71"></div>Final</div>
    <div class="li"><div class="lb" style="background:#1a5888"></div>IKG</div>
    <div class="li"><div class="lb" style="background:#3d1a6e"></div>Temp</div>
    <div class="li"><div class="lb" style="background:#5a2800"></div>External</div>
  </div>
</div>
<div id="wrap">
  <div id="cva"><canvas id="cv"></canvas></div>
  <div id="panel">
    <div id="ph">
      <button id="ph-x" onclick="closePanel()">&#10005;</button>
      <div id="ph-title"></div>
      <div id="ph-meta"></div>
    </div>
    <div id="pb">
      <div class="sec">
        <div class="sechdr cols-hdr" id="hdr-cols" onclick="toggleSec('cols')">
          &#128196;&nbsp;Columns
          <span class="seccnt" id="cnt-cols">0</span>
          <span class="arr">&#9660;</span>
        </div>
        <div class="secbody" id="body-cols"></div>
      </div>
      <div class="sec">
        <div class="sechdr src-hdr" id="hdr-src" onclick="toggleSec('src')">
          &#8593;&nbsp;Source Tables&nbsp;<em style="color:#1a4828;font-weight:400;font-size:.6rem;text-transform:none">(inputs)</em>
          <span class="seccnt" id="cnt-src">0</span>
          <span class="arr">&#9660;</span>
        </div>
        <div class="secbody" id="body-src"></div>
      </div>
      <div class="sec">
        <div class="sechdr dst-hdr collapsed" id="hdr-dst" onclick="toggleSec('dst')">
          &#8595;&nbsp;Destination Tables&nbsp;<em style="color:#482810;font-weight:400;font-size:.6rem;text-transform:none">(outputs)</em>
          <span class="seccnt" id="cnt-dst">0</span>
          <span class="arr">&#9660;</span>
        </div>
        <div class="secbody" id="body-dst"></div>
      </div>
    </div>
  </div>
</div>
<div id="sb">
  <span>Zoom: <b id="sb-z">100%</b></span>
  <span>Visible: <b id="sb-v">0</b></span>
  <span>Drag table to move &bull; Drag background to pan &bull; Scroll to zoom</span>
</div>
"""

    js = "const RAW=" + data_json + ";\n" + r"""
const ST = {
  final:{ fill:'#081e10', stroke:'#2ecc71', hdr:'#0a3018', text:'#7ae8a0', sub:'#2a6040' },
  ikg:  { fill:'#071525', stroke:'#2a78b8', hdr:'#0a1e38', text:'#6ab0d8', sub:'#1a4060' },
  tmp:  { fill:'#12082a', stroke:'#7840c8', hdr:'#180d38', text:'#a878e8', sub:'#3a1870' },
  ext:  { fill:'#1a0a00', stroke:'#c06020', hdr:'#240e00', text:'#e09050', sub:'#602810' },
};
const TC = { final:'#2ecc71', ikg:'#2a78b8', tmp:'#7840c8', ext:'#c06020' };
const TL = { final:'Final Output', ikg:'IKG Table', tmp:'Temp Table', ext:'External/EDW' };

const HDR_H=28, ROW_H=16, PREVIEW_N=4, FOOTER_H=10, NODE_W_MIN=190;

function nodeH(n){ var r=Math.min((n.cols||[]).length,PREVIEW_N); return HDR_H+(r>0?r*ROW_H+6:0)+FOOTER_H; }
function nodeW(n){ return Math.max(NODE_W_MIN, n.id.length*7.4+32); }

const NM={}, EO={}, EI={};
RAW.nodes.forEach(function(n){
  n.w=nodeW(n); n.h=nodeH(n); n.x=0; n.y=0;
  NM[n.id]=n; EO[n.id]=[]; EI[n.id]=[];
});
RAW.edges.forEach(function(e){
  if(NM[e.from]&&NM[e.to]){ EO[e.from].push(e); EI[e.to].push(e); }
});

// ── Layout: level-based columns, no overlaps ──────────────────────────────────
function layout(){
  var byLv={};
  RAW.nodes.forEach(function(n){
    if(!byLv[n.level]) byLv[n.level]=[];
    byLv[n.level].push(n);
  });
  var maxLv=0;
  RAW.nodes.forEach(function(n){if(n.level>maxLv)maxLv=n.level;});

  // Sort each column by connectivity desc
  Object.keys(byLv).forEach(function(lv){
    byLv[lv].sort(function(a,b){
      return (EO[b.id].length+EI[b.id].length)-(EO[a.id].length+EI[a.id].length);
    });
  });

  var COL_GAP=55, ROW_GAP=18, PAD_X=60, PAD_Y=50;

  // Max width per level
  var maxW={};
  Object.keys(byLv).forEach(function(lv){
    var mw=0; byLv[lv].forEach(function(n){if(n.w>mw)mw=n.w;}); maxW[lv]=mw;
  });

  // Build column x positions:
  // Final table has highest level → place it on the LEFT (x = small)
  // Source tables have level 0   → place them on the RIGHT (x = large)
  // So we sort levels DESCENDING: maxLevel first → x increases right
  var sortedLvs=Object.keys(byLv).map(Number).sort(function(a,b){return b-a;});
  var xCursor=PAD_X, colX={};
  sortedLvs.forEach(function(lv){
    colX[lv]=xCursor; xCursor+=maxW[lv]+COL_GAP;
  });

  // Assign y: pack nodes with no overlap
  Object.keys(byLv).forEach(function(lv){
    var yCursor=PAD_Y;
    byLv[lv].forEach(function(n){
      n.x=colX[lv]; n.y=yCursor; yCursor+=n.h+ROW_GAP;
    });
  });
}
layout();

var cva=document.getElementById('cva'), cv=document.getElementById('cv'), ctx=cv.getContext('2d');
var tx=0,ty=0,sc=1, maxLv=99, showT={ikg:true,tmp:true,ext:true}, srchQ='', hlId=null;

function isVis(n){
  if(n.level>maxLv) return false;
  if(n.type!=='final'&&!showT[n.type]) return false;
  if(srchQ&&n.id.indexOf(srchQ)<0) return false;
  return true;
}

function render(){
  ctx.clearRect(0,0,cv.width,cv.height);
  ctx.save(); ctx.translate(tx,ty); ctx.scale(sc,sc);
  drawGrid();
  RAW.edges.forEach(function(e){
    var a=NM[e.from],b=NM[e.to];
    if(!a||!b||!isVis(a)||!isVis(b)) return;
    drawEdge(e,a,b);
  });
  RAW.nodes.forEach(function(n){if(isVis(n))drawNode(n);});
  ctx.restore();
  document.getElementById('sb-v').textContent=RAW.nodes.filter(isVis).length;
}

function drawGrid(){
  var gs=40,ox=-tx/sc,oy=-ty/sc,W=cv.width/sc,H=cv.height/sc;
  ctx.strokeStyle='#0c1525'; ctx.lineWidth=1;
  for(var x=Math.floor(ox/gs)*gs;x<ox+W;x+=gs){ctx.beginPath();ctx.moveTo(x,oy);ctx.lineTo(x,oy+H);ctx.stroke();}
  for(var y=Math.floor(oy/gs)*gs;y<oy+H;y+=gs){ctx.beginPath();ctx.moveTo(ox,y);ctx.lineTo(ox+W,y);ctx.stroke();}
}

function drawEdge(e,a,b){
  var hl=hlId&&(e.from===hlId||e.to===hlId);
  var x1=a.x+a.w,y1=a.y+a.h/2,x2=b.x,y2=b.y+b.h/2,cp=Math.abs(x2-x1)*0.4;
  ctx.beginPath(); ctx.moveTo(x1,y1);
  ctx.bezierCurveTo(x1+cp,y1,x2-cp,y2,x2,y2);
  ctx.strokeStyle=hl?(e.to===hlId?'#3adc80':'#e89030'):(srchQ?'#111d2e':'#182840');
  ctx.lineWidth=hl?2:1; ctx.globalAlpha=hl?1:.45; ctx.stroke(); ctx.globalAlpha=1;
  if(hl){
    var col=e.to===hlId?'#3adc80':'#e89030';
    var ang=Math.atan2(y2-(y1+(y2-y1)*.8),x2-(x1+(x2-x1)*.8)),as=6;
    ctx.beginPath(); ctx.moveTo(x2,y2);
    ctx.lineTo(x2-as*Math.cos(ang-.4),y2-as*Math.sin(ang-.4));
    ctx.lineTo(x2-as*Math.cos(ang+.4),y2-as*Math.sin(ang+.4));
    ctx.closePath(); ctx.fillStyle=col; ctx.fill();
  }
}

function drawNode(n){
  var s=ST[n.type],isFinal=(n.id===RAW.final),isHl=(hlId===n.id);
  var conn=hlId&&(EO[n.id].some(function(e){return e.to===hlId;})||EI[n.id].some(function(e){return e.from===hlId;}));
  var dim=hlId&&!isHl&&!conn;
  if(isHl||isFinal){ctx.shadowColor=s.stroke;ctx.shadowBlur=isHl?16:8;}
  ctx.globalAlpha=dim?0.2:1;
  ctx.fillStyle=s.fill; rrect(n.x,n.y,n.w,n.h,7); ctx.fill();
  ctx.fillStyle=s.hdr; rrect(n.x,n.y,n.w,HDR_H,{tl:7,tr:7,bl:0,br:0}); ctx.fill();
  ctx.strokeStyle=isHl?'#fff':(srchQ&&n.id.indexOf(srchQ)>=0?'#e8d040':s.stroke);
  ctx.lineWidth=isHl||isFinal?2:1; rrect(n.x,n.y,n.w,n.h,7); ctx.stroke();
  ctx.shadowBlur=0;
  ctx.fillStyle=s.stroke; rrect(n.x,n.y,n.w,3,{tl:7,tr:7,bl:0,br:0}); ctx.fill();
  ctx.fillStyle=dim?'#1e2d3e':s.text;
  ctx.font=(isFinal?'bold ':'')+' 10.5px Segoe UI,system-ui,sans-serif';
  ctx.textAlign='left'; ctx.textBaseline='middle';
  var lbl=n.id,mw=n.w-52;
  while(ctx.measureText(lbl).width>mw&&lbl.length>4) lbl=lbl.slice(0,-4)+'...';
  ctx.fillText(lbl,n.x+10,n.y+HDR_H/2+1);
  ctx.font='8px Segoe UI,system-ui,sans-serif'; ctx.fillStyle=dim?'#1a2838':s.sub; ctx.textAlign='right';
  ctx.fillText(isFinal?'OUT':'L'+n.level,n.x+n.w-5,n.y+HDR_H/2+1);
  if((n.cols||[]).length>0&&!dim){
    var cols=n.cols.slice(0,PREVIEW_N);
    ctx.font='9px Consolas,Courier New,monospace'; ctx.textAlign='left';
    cols.forEach(function(col,i){
      var cy=n.y+HDR_H+5+i*ROW_H+ROW_H/2;
      ctx.fillStyle=s.sub; ctx.beginPath(); ctx.arc(n.x+14,cy,2,0,Math.PI*2); ctx.fill();
      ctx.fillStyle='#5a7888';
      var clbl=col; while(ctx.measureText(clbl).width>n.w-28&&clbl.length>4) clbl=clbl.slice(0,-3)+'..';
      ctx.fillText(clbl,n.x+21,cy+1);
    });
    if(n.cols.length>PREVIEW_N){
      ctx.fillStyle=s.sub; ctx.font='8px Segoe UI,system-ui,sans-serif';
      ctx.fillText('+'+(n.cols.length-PREVIEW_N)+' more',n.x+10,n.y+HDR_H+5+PREVIEW_N*ROW_H+2);
    }
  }
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

// Separate drag vs click properly
var panning=null, draggingNode=null, dragOffX=0, dragOffY=0;
var mouseDownX=0, mouseDownY=0, nodeDragMoved=false;

cv.addEventListener('mousedown',function(e){
  mouseDownX=e.clientX; mouseDownY=e.clientY; nodeDragMoved=false;
  var p=s2w(e.offsetX,e.offsetY), hit=hitNode(p.x,p.y);
  if(hit){ draggingNode=hit; dragOffX=p.x-hit.x; dragOffY=p.y-hit.y; }
  else    { panning={mx:e.clientX,my:e.clientY,tx:tx,ty:ty}; }
  cv.style.cursor='grabbing';
});
cv.addEventListener('mousemove',function(e){
  if(draggingNode){
    if(Math.abs(e.clientX-mouseDownX)>3||Math.abs(e.clientY-mouseDownY)>3) nodeDragMoved=true;
    var p=s2w(e.offsetX,e.offsetY);
    draggingNode.x=p.x-dragOffX; draggingNode.y=p.y-dragOffY; render();
  } else if(panning){
    tx=panning.tx+(e.clientX-panning.mx); ty=panning.ty+(e.clientY-panning.my); render();
  }
});
cv.addEventListener('mouseup',function(e){
  var hitN=draggingNode, wasPan=panning;
  var panDx=wasPan?Math.abs(e.clientX-panning.mx):0, panDy=wasPan?Math.abs(e.clientY-panning.my):0;
  var wasClick=hitN&&!nodeDragMoved, wasPanClick=wasPan&&panDx<4&&panDy<4;
  draggingNode=null; panning=null; nodeDragMoved=false; cv.style.cursor='grab';
  if(wasClick){ hlId=hitN.id; openPanel(hitN); render(); }
  else if(wasPanClick){ hlId=null; closePanel(); render(); }
});
cv.addEventListener('mouseleave',function(){ draggingNode=null; panning=null; cv.style.cursor='grab'; });
cv.addEventListener('wheel',function(e){
  e.preventDefault();
  var d=e.deltaY>0?.88:1.14;
  tx=e.offsetX-d*(e.offsetX-tx); ty=e.offsetY-d*(e.offsetY-ty);
  sc=Math.max(.04,Math.min(6,sc*d));
  document.getElementById('sb-z').textContent=Math.round(sc*100)+'%'; render();
},{passive:false});

function s2w(sx,sy){return{x:(sx-tx)/sc,y:(sy-ty)/sc};}
function hitNode(wx,wy){
  for(var i=RAW.nodes.length-1;i>=0;i--){
    var n=RAW.nodes[i]; if(!isVis(n)) continue;
    if(wx>=n.x&&wx<=n.x+n.w&&wy>=n.y&&wy<=n.y+n.h) return n;
  }
  return null;
}

// ── Panel ─────────────────────────────────────────────────────────────────────
function openPanel(n){
  document.getElementById('panel').classList.add('open');
  document.getElementById('ph-title').textContent=n.id;
  document.getElementById('ph-meta').textContent=
    TL[n.type]+' \u2022 Level '+n.level+
    ' \u2022 '+(n.cols||[]).length+' cols'+
    ' \u2022 '+EI[n.id].length+' inputs \u2022 '+EO[n.id].length+' outputs';
  document.getElementById('ph-meta').style.color=TC[n.type]+'88';
  buildCols(n);
  buildRelSec('src',EI[n.id],true);
  buildRelSec('dst',EO[n.id],false);
}

function buildCols(n){
  var cols=n.cols||[];
  document.getElementById('cnt-cols').textContent=cols.length;
  var body=document.getElementById('body-cols'); body.innerHTML='';
  if(!cols.length){
    var em=document.createElement('div');
    em.style.cssText='padding:7px 13px;color:#1a2e3e;font-size:.7rem;font-style:italic;';
    em.textContent='No columns found in SQL (external/base table or not in parsed files)';
    body.appendChild(em); return;
  }
  cols.forEach(function(col,i){
    var row=document.createElement('div'); row.className='col-row';
    var idx=document.createElement('span'); idx.className='col-idx'; idx.textContent=(i+1)+'.';
    var nm=document.createElement('span'); nm.className='col-name'; nm.textContent=col;
    row.appendChild(idx); row.appendChild(nm); body.appendChild(row);
  });
}

function buildRelSec(sec,edges,isSource){
  document.getElementById('cnt-'+sec).textContent=edges.length;
  var body=document.getElementById('body-'+sec); body.innerHTML='';
  if(!edges.length){
    var em=document.createElement('div');
    em.style.cssText='padding:7px 13px;color:#162030;font-size:.7rem;font-style:italic;';
    em.textContent=isSource?'No source tables (base/external input)':'No downstream tables (leaf)';
    body.appendChild(em); return;
  }
  edges.forEach(function(e){
    var otherId=isSource?e.from:e.to;
    var other=NM[otherId], type=other?other.type:'ext', col=TC[type]||'#718096';
    var fromAlias=e.src_alias||'', toAlias=e.tgt_alias||'';
    var displayAlias=isSource?fromAlias:toAlias;
    var row=document.createElement('div'); row.className='trow';
    row.addEventListener('click',function(){
      if(NM[otherId]){hlId=otherId;openPanel(NM[otherId]);focusNode(otherId);render();}
    });
    var top=document.createElement('div'); top.className='trow-top';
    var dot=document.createElement('div'); dot.className='tdot';
    dot.style.background=col; dot.style.border='1px solid '+col+'55';
    var nm=document.createElement('div'); nm.className='tname'; nm.textContent=otherId;
    top.appendChild(dot); top.appendChild(nm);
    if(displayAlias){
      var ab=document.createElement('span'); ab.className='talias';
      ab.textContent='alias: '+displayAlias; top.appendChild(ab);
    }
    row.appendChild(top);
    var jks=e.jk||[];
    if(jks.length){
      var tbl=document.createElement('table'); tbl.className='jktbl';
      var thead=tbl.createTHead(), hr=thead.insertRow();
      function th(t){var el=document.createElement('th');el.textContent=t;hr.appendChild(el);}
      th('Source ('+(fromAlias||e.from.slice(0,14))+')');th('');th('Dest ('+(toAlias||e.to.slice(0,14))+')');
      var tb2=tbl.createTBody();
      jks.forEach(function(jk){
        var tr=tb2.insertRow();
        if(jk.left&&jk.right){
          tr.insertCell().className='jk-s'; tr.cells[0].textContent=jk.left;
          tr.insertCell().className='jk-e'; tr.cells[1].textContent='=';
          tr.insertCell().className='jk-d'; tr.cells[2].textContent=jk.right;
        } else if(jk.col){
          var td=tr.insertCell(); td.className='jk-s'; td.textContent=jk.col; td.colSpan=3;
        }
      });
      row.appendChild(tbl);
    } else {
      var nj=document.createElement('div'); nj.className='no-jk';
      nj.textContent='(no explicit join key)'; row.appendChild(nj);
    }
    body.appendChild(row);
  });
}

function closePanel(){document.getElementById('panel').classList.remove('open');hlId=null;render();}
function toggleSec(s){document.getElementById('hdr-'+s).classList.toggle('collapsed');}
function focusNode(id){
  var n=NM[id];if(!n)return;
  var r=cva.getBoundingClientRect();
  tx=r.width/2-n.x*sc-(n.w/2)*sc; ty=r.height/2-n.y*sc-(n.h/2)*sc;
}

function resize(){var r=cva.getBoundingClientRect();cv.width=r.width;cv.height=r.height;render();}
window.addEventListener('resize',resize);

function resetView(){hlId=null;closePanel();fitAll();}
function fitAll(){
  maxLv=99;
  ['ikg','tmp','ext'].forEach(function(t){showT[t]=true;document.getElementById('btn-'+t).classList.add('on');});
  fitVisible();
}
function fitLv(lv){maxLv=lv;fitVisible();}
function fitVisible(){
  var vis=RAW.nodes.filter(isVis); if(!vis.length)return;
  var x1=Math.min.apply(null,vis.map(function(n){return n.x;}));
  var x2=Math.max.apply(null,vis.map(function(n){return n.x+n.w;}));
  var y1=Math.min.apply(null,vis.map(function(n){return n.y;}));
  var y2=Math.max.apply(null,vis.map(function(n){return n.y+n.h;}));
  var pad=50;
  sc=Math.min((cv.width-pad*2)/(x2-x1||1),(cv.height-pad*2)/(y2-y1||1),1.4);
  tx=pad-x1*sc; ty=(cv.height-(y2-y1)*sc)/2-y1*sc;
  document.getElementById('sb-z').textContent=Math.round(sc*100)+'%'; render();
}
function zoomIn(){applyZoom(1.2);} function zoomOut(){applyZoom(.83);}
function applyZoom(d){
  var cx=cv.width/2,cy=cv.height/2; tx=cx-d*(cx-tx); ty=cy-d*(cy-ty);
  sc=Math.max(.04,Math.min(6,sc*d));
  document.getElementById('sb-z').textContent=Math.round(sc*100)+'%'; render();
}
function toggle(t){showT[t]=!showT[t];document.getElementById('btn-'+t).classList.toggle('on',showT[t]);render();}
function onSearch(){srchQ=document.getElementById('srch').value.toLowerCase().trim();render();}
function clearSearch(){srchQ='';document.getElementById('srch').value='';render();}

window.addEventListener('load',function(){
  resize();
  var ikg=RAW.nodes.filter(function(n){return n.type==='ikg'||n.type==='final';}).length;
  var tmp=RAW.nodes.filter(function(n){return n.type==='tmp';}).length;
  var ext=RAW.nodes.filter(function(n){return n.type==='ext';}).length;
  document.getElementById('sp-ikg').textContent=ikg+' IKG tables';
  document.getElementById('sp-tmp').textContent=tmp+' Temp tables';
  document.getElementById('sp-ext').textContent=ext+' External tables';
  document.getElementById('sp-rel').textContent=RAW.edges.length+' Relationships';
  fitVisible();
});
"""

    return (
        "<!DOCTYPE html>\n<html lang='en'>\n<head>\n"
        "<meta charset='UTF-8'><meta name='viewport' content='width=device-width,initial-scale=1'>\n"
        "<title>"+FINAL+" \u2014 ER Diagram</title>\n"
        "<style>\n"+css+"</style>\n</head>\n<body>\n"
        +body+
        "<script>\n"+js+"\n</script>\n</body>\n</html>\n"
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
    summary['final_table'] = auto_detect_final_table(summary, sql_text)
    summary['_sql_text'] = sql_text

    # Scan same folder for additional SQL files to enrich column data
    extra_sqls = []
    for ext in ('*.sql', '*.txt'):
        for f2 in script_dir.glob(ext):
            if f2.resolve() != sql_path.resolve():
                try:
                    with open(f2, encoding='utf-8', errors='replace') as fh:
                        extra_sqls.append(fh.read())
                except Exception:
                    pass
    summary['_extra_sql_texts'] = extra_sqls
    if extra_sqls:
        print(f"  [+] Loaded {len(extra_sqls)} additional SQL file(s) for column enrichment")

    print("\n" + "=" * 60)
    print("  PARSE SUMMARY")
    print("=" * 60)
    print(f"  Final table        : {summary['final_table']}")
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
