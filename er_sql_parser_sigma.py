#!/usr/bin/env python3
"""
SQL Script Parser — Sigma.js Edition
=====================================
Parses a SQL pipeline script and generates an interactive ER diagram
powered by Sigma.js with a full details panel.

Usage:
    python er_sql_parser_sigma.py
    → Prompted for the SQL filename (must be in same folder as this script).

Output:
    - <filename>_sigma_lineage.json
    - <filename>_sigma_er_diagram.html
"""

import re
import sys
import json
import os
from pathlib import Path
from collections import defaultdict, deque


# ─────────────────────────────────────────────────────────────────────────────
# Column Extractor
# ─────────────────────────────────────────────────────────────────────────────

def extract_table_columns(sql_text: str) -> dict:
    clean = re.sub(r'/\*.*?\*/', ' ', sql_text, flags=re.DOTALL)
    clean = re.sub(r'--[^\n]*', ' ', clean)
    result = {}

    create_pat = re.compile(
        r'CREATE\s+(?:TEMP(?:ORARY)?\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?'
        r'(?:\{\{[^}]+\}\}\.)?(\w+)\s+AS\s*', re.IGNORECASE)
    for m in create_pat.finditer(clean):
        tbl  = m.group(1).lower()
        body = clean[m.end():]
        with_m = re.match(r'\s*WITH\b', body, re.IGNORECASE)
        if with_m:
            depth, pos = 0, 0
            while pos < len(body):
                ch = body[pos]
                if ch == '(':   depth += 1
                elif ch == ')': depth -= 1
                elif depth == 0 and body[pos:pos+6].upper() == 'SELECT':
                    break
                pos += 1
            body = body[pos:]
        sel_m = re.search(r'\bSELECT\b', body, re.IGNORECASE)
        if not sel_m:
            continue
        cols = _select_cols(body[sel_m.start():])
        if cols:
            result[tbl] = cols

    insert_pat = re.compile(
        r'INSERT\s+INTO\s+(?:\{\{[^}]+\}\}\.)?(\w+)\s*\n?\s*(SELECT\b)',
        re.IGNORECASE | re.DOTALL)
    for m in insert_pat.finditer(clean):
        tbl = m.group(1).lower()
        if tbl not in result:
            cols = _select_cols(clean[m.start(2):])
            if cols:
                result[tbl] = cols
    return result


def _select_cols(body: str) -> list:
    body = re.sub(r'^\s*SELECT\s+(?:ALL\s+|DISTINCT\s+)?', '', body, flags=re.IGNORECASE)
    depth, from_pos = 0, len(body)
    upper = body.upper()
    i = 0
    while i < len(body):
        ch = body[i]
        if ch == '(':   depth += 1
        elif ch == ')':
            depth -= 1
            if depth < 0:
                from_pos = i; break
        elif depth == 0 and re.match(r'\bFROM\b', upper[i:]):
            from_pos = i; break
        i += 1

    cols_raw, depth, buf = [], 0, []
    for ch in body[:from_pos]:
        if ch == '(':   depth += 1; buf.append(ch)
        elif ch == ')': depth -= 1; buf.append(ch)
        elif ch == ',' and depth == 0:
            cols_raw.append(''.join(buf).strip()); buf = []
        else:
            buf.append(ch)
    if buf:
        cols_raw.append(''.join(buf).strip())

    SKIP = {'null','true','false','desc','asc','int','text','date','varchar',
            'numeric','bigint','boolean','timestamp','char'}
    result, seen = [], set()
    for col in cols_raw:
        col = col.strip()
        if not col: continue
        as_m = re.search(r'\bAS\s+(\w+)\s*$', col, re.IGNORECASE)
        if as_m:
            name = as_m.group(1).lower()
        else:
            last = re.search(r'(?:[.\s,(]|^)(\w+)\s*$', col)
            name = last.group(1).lower() if last else None
        if name and len(name) > 1 and name not in SKIP and name not in seen:
            seen.add(name); result.append(name)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def normalize_table_name(raw: str) -> str:
    name = re.sub(r'\{\{[^}]+\}\}\.', '', raw)
    return name.strip().strip('"').lower()

def strip_comments(sql: str) -> str:
    sql = re.sub(r'/\*.*?\*/', ' ', sql, flags=re.DOTALL)
    sql = re.sub(r'--[^\n]*', ' ', sql)
    return sql

def is_ignorable(stmt: str) -> bool:
    s = stmt.strip().upper()
    if re.match(r'^\s*ALTER\s+TABLE\s+\S+\s+OWNER\s+TO', s): return True
    if re.match(r'^\s*GRANT\s+', s): return True
    return False

def extract_join_conditions(sql_block: str, source_table: str, aliases: dict) -> list:
    tbl_to_aliases = {}
    for alias, tbl in aliases.items():
        tbl_to_aliases.setdefault(tbl, set()).add(alias)
    src_aliases = tbl_to_aliases.get(source_table, set()) | {source_table}

    results = []
    join_on_pat = re.compile(
        r'\bJOIN\s+(?:\{\{[^}]+\}\}\.)?(\w+)(?:\s+(?:AS\s+)?(\w+))?\s+ON\b(.+?)'
        r'(?=\b(?:LEFT|RIGHT|INNER|FULL|CROSS|OUTER|JOIN|WHERE|GROUP\s+BY|'
        r'ORDER\s+BY|HAVING|LIMIT|DISTRIBUTED|;)\b|$)',
        re.IGNORECASE | re.DOTALL)
    eq_pat = re.compile(r'(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)')

    for m in join_on_pat.finditer(sql_block):
        on_clause = m.group(3)
        for eq in eq_pat.finditer(on_clause):
            la, lc, ra, rc = eq.group(1).lower(), eq.group(2), eq.group(3).lower(), eq.group(4)
            ltbl = aliases.get(la, la); rtbl = aliases.get(ra, ra)
            if ltbl == source_table or rtbl == source_table or la in src_aliases or ra in src_aliases:
                results.append((la + '.' + lc, ra + '.' + rc))

    seen, deduped = set(), []
    for pair in results:
        key = (pair[0].lower(), pair[1].lower())
        if key not in seen:
            seen.add(key); deduped.append(pair)
    return deduped


# ─────────────────────────────────────────────────────────────────────────────
# SQL Parser
# ─────────────────────────────────────────────────────────────────────────────

_TABLE_PLACEHOLDER = r'(?:\{\{[^}]+\}\}\.)?(\w+)'
_RE_CREATE   = re.compile(r'\bCREATE\s+(?:TEMP(?:ORARY)?\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?' + _TABLE_PLACEHOLDER, re.IGNORECASE)
_RE_INSERT   = re.compile(r'\bINSERT\s+INTO\s+' + _TABLE_PLACEHOLDER, re.IGNORECASE)
_RE_DROP     = re.compile(r'\bDROP\s+TABLE\s+(?:IF\s+EXISTS\s+)?' + _TABLE_PLACEHOLDER, re.IGNORECASE)
_RE_FROM_JOIN= re.compile(r'\b(?:FROM|JOIN)\s+' + _TABLE_PLACEHOLDER + r'(?:\s+(?:AS\s+)?(\w+))?', re.IGNORECASE)
_RE_WITH_CTE = re.compile(r'\bWITH\b(.*?)(?=\bSELECT\b)', re.IGNORECASE | re.DOTALL)
_RE_CTE_NAME = re.compile(r'\b(\w+)\s+AS\s*\(', re.IGNORECASE)


class SQLParser:
    def __init__(self, sql_text: str):
        self.raw_sql       = sql_text
        self.clean_sql     = strip_comments(sql_text)
        self.created_tables: set  = set()
        self.external_tables: set = set()
        self.temp_tables: set     = set()
        self.relationships: list  = []
        self.alias_map: dict      = {}
        self.cte_names: set       = set()
        self.table_sources: dict  = defaultdict(set)

    def _collect_cte_names(self):
        for m in _RE_WITH_CTE.finditer(self.clean_sql):
            for nm in _RE_CTE_NAME.finditer(m.group(1)):
                self.cte_names.add(nm.group(1).lower())

    def _is_external(self, raw_ref: str) -> bool:
        return bool(re.search(r'\{\{params\.(EDW_VIEW_INPUT_SCHEMA|EDW_INPUT_SCHEMA|MODEL_SCHEMA)[^}]*\}\}\.', raw_ref, re.IGNORECASE))

    def _register_table(self, name: str, is_temp=False, is_external=False):
        name = name.lower()
        if name in self.cte_names: return
        if is_external:    self.external_tables.add(name)
        elif is_temp:      self.temp_tables.add(name)
        else:              self.created_tables.add(name)

    def parse(self):
        self._collect_cte_names()
        for stmt in self.clean_sql.split(';'):
            stmt = stmt.strip()
            if not stmt or is_ignorable(stmt): continue
            self._parse_statement(stmt)
        self._resolve_aliases()

    def _parse_statement(self, stmt: str):
        is_temp_stmt = bool(re.match(r'\s*CREATE\s+TEMP', stmt, re.IGNORECASE))
        create_m = _RE_CREATE.search(stmt)
        if create_m:
            target = normalize_table_name(create_m.group(0).split()[-1])
            self._register_table(target, is_temp=is_temp_stmt)
            self._extract_sources(stmt, target); return
        insert_m = _RE_INSERT.search(stmt)
        if insert_m:
            raw = insert_m.group(0)
            target = normalize_table_name(insert_m.group(1))
            self._register_table(target, is_external=self._is_external(raw))
            self._extract_sources(stmt, target); return

    def _extract_sources(self, stmt: str, target: str):
        local_ctes: set = set()
        for m in _RE_WITH_CTE.finditer(stmt):
            for nm in _RE_CTE_NAME.finditer(m.group(1)):
                local_ctes.add(nm.group(1).lower())

        _skip = {
            'where','on','set','select','inner','left','right','full',
            'outer','cross','join','from','as','not','null','and','or',
            'in','by','group','order','having','limit','union','all',
            'distinct','with','values','distributed','using','for',
            'case','when','then','else','end','exists','between',
            'like','ilike','is','true','false','into','insert','update',
            'delete','create','drop','alter','table','temp','temporary',
            'partition','over','window','rows','range','unbounded',
            'preceding','following','current','row','natural','lateral',
            'recursive','only','returning',
        }
        stmt_table_to_alias = {}
        for m in _RE_FROM_JOIN.finditer(stmt):
            traw = normalize_table_name(m.group(1))
            al   = (m.group(2) or '').lower()
            if al and al not in _skip and al != traw:
                stmt_table_to_alias[traw] = al

        for m in _RE_FROM_JOIN.finditer(stmt):
            raw_full       = m.group(0)
            table_name_raw = m.group(1)
            alias          = m.group(2)
            table_name     = normalize_table_name(table_name_raw)

            if table_name in local_ctes or table_name in self.cte_names: continue
            if table_name.upper() in ('SELECT','WITH','VALUES','LATERAL','UNNEST','ROWS','ONLY'): continue
            if table_name == target: continue

            is_ext = self._is_external(raw_full)
            self._register_table(table_name, is_external=is_ext,
                is_temp=table_name.endswith('_temp_auto') or
                        table_name.endswith('_tmp') or
                        table_name.startswith('temp_'))
            if alias:
                self.alias_map[alias.lower()] = table_name
            self.table_sources[target].add(table_name)

            raw_jc = extract_join_conditions(stmt, table_name, self.alias_map)
            dedup_jk = []
            seen_jk  = set()
            for jc in raw_jc:
                key = (str(jc[0]).lower(), str(jc[1]).lower()) if isinstance(jc, tuple) else str(jc).lower()
                if key not in seen_jk:
                    seen_jk.add(key)
                    dedup_jk.append([jc[0], jc[1]] if isinstance(jc, tuple) else str(jc))

            src_alias = stmt_table_to_alias.get(table_name, alias.lower() if alias else '')
            tgt_alias = stmt_table_to_alias.get(target, '')

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
            def _ca(a):
                if not a: return ''
                a = a.strip().lower()
                if a in ALIAS_BL: return ''
                if not re.match(r'^[a-z_][a-z0-9_]{0,29}$', a): return ''
                return a

            self.relationships.append({
                'from': table_name, 'to': target,
                'join_cols': dedup_jk,
                'alias': _ca(src_alias),
                'target_alias': _ca(tgt_alias),
                'rel_type': 'JOIN/SOURCE',
                'is_external': is_ext,
            })

    def _resolve_aliases(self):
        merged = {}
        for r in self.relationships:
            key = (r['from'], r['to'])
            if key not in merged:
                merged[key] = dict(r)
                merged[key]['join_cols'] = list(r.get('join_cols', []))
            else:
                existing = set(tuple(jk) if isinstance(jk, list) else jk for jk in merged[key]['join_cols'])
                for jk in r.get('join_cols', []):
                    jk_key = tuple(jk) if isinstance(jk, list) else jk
                    if jk_key not in existing:
                        existing.add(jk_key)
                        merged[key]['join_cols'].append(jk)
                if not merged[key]['alias'] and r.get('alias'):
                    merged[key]['alias'] = r['alias']
        self.relationships = list(merged.values())

    def summary(self) -> dict:
        return {
            'created_tables':  sorted(self.created_tables),
            'temp_tables':     sorted(self.temp_tables),
            'external_tables': sorted(self.external_tables),
            'relationships':   self.relationships,
            'table_sources':   {k: list(v) for k, v in self.table_sources.items()},
            'final_table':     '',
        }


# ─────────────────────────────────────────────────────────────────────────────
# Auto-detect final table
# ─────────────────────────────────────────────────────────────────────────────

def auto_detect_final_table(summary: dict, sql_text: str = '') -> str:
    created = list(summary['created_tables'])
    if not created: return ''

    if sql_text:
        pattern = re.compile(
            r'CREATE\s+(?:TEMP(?:ORARY)?\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:\{\{[^}]+\}\}\.)?(\w+)',
            re.IGNORECASE)
        positions = []
        for m in pattern.finditer(sql_text):
            tbl = m.group(1).lower()
            if tbl in summary['created_tables']:
                positions.append((m.start(), tbl))
        if positions:
            positions.sort(key=lambda x: x[0])
            for _, tbl in reversed(positions):
                if tbl in set(summary['created_tables']) and tbl not in set(summary['temp_tables']):
                    return tbl
            return positions[-1][1]

    all_sources = set(r['from'] for r in summary['relationships'])
    all_targets = set(r['to']   for r in summary['relationships'])
    leaf_tables = [t for t in set(summary['created_tables']) if t not in all_sources and t in all_targets]
    if not leaf_tables:
        from collections import Counter
        in_deg = Counter(r['to'] for r in summary['relationships'])
        candidates = sorted(set(summary['created_tables']), key=lambda t: -in_deg.get(t, 0))
        return candidates[0] if candidates else (created[-1] if created else '')
    if len(leaf_tables) == 1: return leaf_tables[0]
    rel_up = {}
    for r in summary['relationships']:
        rel_up.setdefault(r['to'], []).append(r['from'])
    def count_up(root):
        vis, q = set(), deque([root])
        while q:
            t = q.popleft()
            if t in vis: continue
            vis.add(t)
            for s in rel_up.get(t, []): q.append(s)
        return len(vis)
    leaf_tables.sort(key=lambda t: -count_up(t))
    return leaf_tables[0]


# ─────────────────────────────────────────────────────────────────────────────
# Sigma.js HTML Generator
# ─────────────────────────────────────────────────────────────────────────────

def build_sigma_html(summary: dict) -> str:

    FINAL = summary.get('final_table') or auto_detect_final_table(
        summary, summary.get('_sql_text', ''))

    # ── Collect all tables ────────────────────────────────────────────────────
    created_set = set(summary['created_tables'])
    temp_set    = set(summary['temp_tables'])
    ext_set     = set(summary['external_tables'])

    cte_names = set()
    sql_text  = summary.get('_sql_text', '')
    clean_sql = re.sub(r'/\*.*?\*/', ' ', sql_text, flags=re.DOTALL)
    clean_sql = re.sub(r'--[^\n]*', ' ', clean_sql)
    for m in re.finditer(r'\bWITH\b(.*?)(?=\bSELECT\b)', clean_sql, re.IGNORECASE | re.DOTALL):
        for nm in re.finditer(r'\b(\w+)\s+AS\s*\(', m.group(1), re.IGNORECASE):
            cte_names.add(nm.group(1).lower())

    skip_kw = {'select','where','on','set','lateral','only','rows','unnest','values',
               'null','true','false','current','row','all','distinct'}
    all_ref = set()
    for m in re.finditer(r'\b(?:FROM|JOIN)\s+(?:\{\{[^}]+\}\}\.)?(\w+)', clean_sql, re.IGNORECASE):
        nm = m.group(1).lower()
        if nm not in skip_kw and nm not in cte_names:
            all_ref.add(nm)

    all_tables = (created_set | temp_set | ext_set | all_ref) - cte_names

    # ── Column enrichment ─────────────────────────────────────────────────────
    col_map = {}
    for extra_sql in summary.get('_extra_sql_texts', []):
        col_map.update(extract_table_columns(extra_sql))
    if sql_text:
        col_map.update(extract_table_columns(sql_text))

    # ── Edges ─────────────────────────────────────────────────────────────────
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
        if not re.match(r'^[a-z_][a-z0-9_]{0,29}$', a): return ''
        return a

    merged_edges = {}
    for r in summary['relationships']:
        src_t, tgt_t = r['from'], r['to']
        if src_t not in all_tables or tgt_t not in all_tables: continue
        key = (src_t, tgt_t)
        if key not in merged_edges:
            merged_edges[key] = {
                'from': src_t, 'to': tgt_t,
                'src_alias': clean_alias(r.get('alias', '')),
                'tgt_alias': clean_alias(r.get('target_alias', '')),
                'jk': [], 'is_external': r.get('is_external', False),
            }
        existing = set(tuple(j) if isinstance(j, list) else j for j in merged_edges[key]['jk'])
        for jc in r.get('join_cols', []):
            if isinstance(jc, (list, tuple)) and len(jc) == 2:
                item = [str(jc[0]), str(jc[1])]; k2 = tuple(item)
            else:
                item = str(jc); k2 = item
            if k2 not in existing:
                existing.add(k2)
                merged_edges[key]['jk'].append({'left': item[0], 'right': item[1]} if isinstance(item, list) else {'col': item})
    edges_list = list(merged_edges.values())

    # ── Topological levels ────────────────────────────────────────────────────
    in_edges_map  = defaultdict(set)
    out_edges_map = defaultdict(set)
    for e in edges_list:
        in_edges_map[e['to']].add(e['from'])
        out_edges_map[e['from']].add(e['to'])

    in_deg = {t: len(in_edges_map[t]) for t in all_tables}
    queue  = deque([t for t in all_tables if in_deg[t] == 0])
    level  = {t: 0 for t in queue}
    while queue:
        t = queue.popleft()
        for nxt in out_edges_map[t]:
            if nxt in all_tables:
                in_deg[nxt]  = max(0, in_deg.get(nxt, 0) - 1)
                level[nxt]   = max(level.get(nxt, 0), level[t] + 1)
                if in_deg[nxt] == 0:
                    queue.append(nxt)
    for t in all_tables:
        if t not in level: level[t] = 0

    def node_type(n):
        if n == FINAL:       return 'final'
        if n in created_set: return 'ikg'
        if n in temp_set:    return 'tmp'
        return 'ext'

    # ── Sigma graph data ──────────────────────────────────────────────────────
    # Layout: final table (max level) LEFT, sources (level 0) RIGHT
    # x = (max_level - node_level) * col_width
    max_level = max(level.values()) if level else 0

    # Group by level, sort within level by connectivity
    by_level = defaultdict(list)
    for t in all_tables:
        by_level[level[t]].append(t)

    # Sort within each level by connectivity (most connected first)
    for lv in by_level:
        by_level[lv].sort(key=lambda t: -(len(in_edges_map[t]) + len(out_edges_map[t])))

    # Assign positions
    COL_W = 300   # horizontal spacing between levels
    ROW_H = 80    # vertical spacing within a level
    PAD_X = 100
    PAD_Y = 60

    positions = {}
    for lv, tables in by_level.items():
        # Final (max_level) → x=PAD_X; level 0 → x = PAD_X + max_level*COL_W
        x = PAD_X + (max_level - lv) * COL_W
        for i, t in enumerate(tables):
            y = PAD_Y + i * ROW_H
            positions[t] = (x, y)

    # Build Sigma nodes
    sigma_nodes = []
    for t in all_tables:
        x, y = positions[t]
        ntype = node_type(t)
        color_map = {'final': '#2ecc71', 'ikg': '#2a78b8', 'tmp': '#7840c8', 'ext': '#c06020'}
        sigma_nodes.append({
            'id':    t,
            'label': t,
            'x':     x,
            'y':     y,
            'size':  5 if ntype == 'final' else 4 if ntype == 'ikg' else 3,
            'color': color_map[ntype],
            'type':  'circle',   # Sigma renderer type — must be built-in
            'ntype': ntype,      # our custom category field
            'level': level.get(t, 0),
            'cols':  col_map.get(t, []),
        })

    # Build Sigma edges
    sigma_edges = []
    for i, e in enumerate(edges_list):
        sigma_edges.append({
            'id':         f'e{i}',
            'source':     e['from'],
            'target':     e['to'],
            'src_alias':  e['src_alias'],
            'tgt_alias':  e['tgt_alias'],
            'jk':         e['jk'],
            'is_external': e['is_external'],
        })

    graph_data = {'nodes': sigma_nodes, 'edges': sigma_edges, 'final': FINAL}
    data_json  = json.dumps(graph_data, separators=(',', ':'))

    # Count for badges
    ikg_count = len([n for n in sigma_nodes if n['ntype'] in ('ikg','final')])
    tmp_count = len([n for n in sigma_nodes if n['ntype'] == 'tmp'])
    ext_count = len([n for n in sigma_nodes if n['ntype'] == 'ext'])
    rel_count = len(sigma_edges)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{FINAL} — ER Diagram</title>

<!-- Sigma.js v2 + Graphology -->
<script src="https://cdn.jsdelivr.net/npm/graphology@0.25.4/dist/graphology.umd.min.js"
        crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/npm/sigma@2.4.0/build/sigma.min.js"
        crossorigin="anonymous"></script>

<style>
:root {{
  --bg:       #090e18;
  --panel-bg: #0b1422;
  --tb-bg:    #0d1525;
  --border:   #162030;
  --text:     #e2e8f0;
  --muted:    #4a6070;
  --final:    #2ecc71;
  --ikg:      #2a78b8;
  --tmp:      #7840c8;
  --ext:      #c06020;
  --hl:       #facc15;
}}
*,*::before,*::after {{ box-sizing:border-box; margin:0; padding:0; }}
body {{ background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,sans-serif;
       overflow:hidden; height:100vh; display:flex; flex-direction:column; }}

/* HEADER */
#hdr {{ background:linear-gradient(90deg,#1a3356,#0d2040);
       padding:10px 18px; display:flex; align-items:center; gap:12px;
       border-bottom:1px solid var(--border); flex-shrink:0; }}
#hdr h1 {{ font-size:.92rem; font-weight:700; color:#5aa8e8; }}
.sub {{ font-size:.67rem; color:#2d4060; margin-top:1px; }}
.sp {{ padding:2px 10px; border-radius:20px; font-size:.65rem; font-weight:600; }}
.sp-b {{ background:#0f2045; color:#5aa8e8; border:1px solid #1e4080; }}
.sp-p {{ background:#1e1040; color:#a070e8; border:1px solid #3a2070; }}
.sp-o {{ background:#2a1500; color:#e09050; border:1px solid #603010; }}
.sp-g {{ background:#0f3020; color:#4ec87a; border:1px solid #2a6040; }}

/* TOOLBAR */
#tb {{ background:var(--tb-bg); border-bottom:1px solid var(--border);
      padding:5px 12px; display:flex; align-items:center; gap:6px;
      flex-shrink:0; flex-wrap:wrap; }}
.btn {{ background:#111d30; color:#7a90a8; border:1px solid #1e2d40;
       border-radius:4px; padding:4px 11px; font-size:.71rem; cursor:pointer; transition:all .12s; }}
.btn:hover {{ background:#1a2d44; color:#c8d8e8; }}
.btn.on {{ background:#122040; color:#5aa8e8; border-color:#1e4070; }}
.sep {{ width:1px; height:18px; background:#1a2535; margin:0 2px; }}
#srch {{ background:#111d30; color:var(--text); border:1px solid #1e2d40;
        border-radius:4px; padding:4px 10px; font-size:.71rem; width:175px; outline:none; }}
#srch:focus {{ border-color:#2a5090; }}
#srch::placeholder {{ color:#2a3848; }}
#leg {{ display:flex; gap:10px; margin-left:auto; align-items:center; }}
.li {{ display:flex; align-items:center; gap:4px; font-size:.65rem; color:var(--muted); }}
.lb {{ width:12px; height:8px; border-radius:2px; }}

/* MAIN */
#wrap {{ flex:1; display:flex; overflow:hidden; }}
#sigma-container {{ flex:1; position:relative; background:var(--bg); }}

/* PANEL */
#panel {{ width:420px; min-width:420px; background:var(--panel-bg);
         border-left:1px solid var(--border); display:none;
         flex-direction:column; overflow:hidden; flex-shrink:0; }}
#panel.open {{ display:flex; }}
#ph {{ background:#091020; padding:11px 13px 9px; border-bottom:1px solid var(--border);
      flex-shrink:0; position:relative; }}
#ph-title {{ font-size:.87rem; font-weight:700; color:#5aa8e8;
            word-break:break-all; margin-bottom:3px; padding-right:24px; }}
#ph-meta {{ font-size:.67rem; color:var(--muted); }}
#ph-x {{ position:absolute; right:12px; top:12px; background:none; border:none;
        color:#2a3848; cursor:pointer; font-size:1rem; transition:color .1s; }}
#ph-x:hover {{ color:var(--text); }}
#pb {{ flex:1; overflow-y:auto; }}
#pb::-webkit-scrollbar {{ width:4px; }}
#pb::-webkit-scrollbar-track {{ background:#080f1c; }}
#pb::-webkit-scrollbar-thumb {{ background:#1a2840; border-radius:3px; }}

.sec {{ border-bottom:1px solid #0f1e30; }}
.sechdr {{ padding:8px 13px; font-size:.66rem; font-weight:700; text-transform:uppercase;
          letter-spacing:.8px; display:flex; align-items:center; gap:7px;
          cursor:pointer; user-select:none; transition:background .1s; }}
.sechdr:hover {{ background:#0d1a2a; }}
.sechdr.c-cols {{ color:#3a6888; }}
.sechdr.c-src  {{ color:#2a6840; }}
.sechdr.c-dst  {{ color:#6a4010; }}
.seccnt {{ margin-left:auto; background:#0f1e2e; color:#3a6888;
          font-size:.61rem; padding:1px 7px; border-radius:10px; font-weight:700; }}
.arr {{ font-size:.62rem; color:#1e2d3e; transition:transform .15s; }}
.sechdr.collapsed .arr {{ transform:rotate(-90deg); }}
.secbody {{ display:block; }}
.sechdr.collapsed + .secbody {{ display:none; }}

.col-row {{ padding:4px 13px 4px 22px; display:flex; align-items:center; gap:8px;
           border-bottom:1px solid #0a1520; font-size:.73rem; }}
.col-row:last-child {{ border-bottom:none; }}
.col-idx {{ color:#1e3040; font-size:.6rem; min-width:18px; font-family:monospace; }}
.col-name {{ color:#90b8d8; font-family:'Consolas','Courier New',monospace; }}

.trow {{ padding:7px 11px 7px 15px; border-bottom:1px solid #0a1828;
        cursor:pointer; transition:background .1s; }}
.trow:hover {{ background:#0d1c2e; }}
.trow-top {{ display:flex; align-items:flex-start; gap:7px; margin-bottom:4px; }}
.tdot {{ width:7px; height:7px; border-radius:2px; flex-shrink:0; margin-top:4px; }}
.tname {{ font-size:.75rem; font-weight:600; color:#b0c8d8; flex:1; word-break:break-all; }}
.talias {{ font-size:.62rem; color:#1e4060; background:#081522; border:1px solid #102030;
          border-radius:3px; padding:1px 6px; font-family:'Consolas','Courier New',monospace; white-space:nowrap; }}
.jktbl {{ width:calc(100% - 14px); border-collapse:collapse; margin:2px 0 2px 14px; }}
.jktbl th {{ font-size:.59rem; color:#1e4050; font-weight:700; text-transform:uppercase;
            letter-spacing:.4px; padding:2px 6px 2px 0; border-bottom:1px solid #0f1e2e; }}
.jktbl td {{ font-size:.66rem; font-family:'Consolas','Courier New',monospace;
            padding:2px 6px 2px 0; vertical-align:top; border-bottom:1px solid #081520; }}
.jktbl tr:last-child td {{ border-bottom:none; }}
.jk-s {{ color:#4ec87a; }} .jk-e {{ color:#1e2d3a; padding:0 2px; }} .jk-d {{ color:#e8a040; }}
.no-jk {{ font-size:.62rem; color:#162028; padding:2px 0 2px 14px; font-style:italic; }}

#sb {{ background:var(--tb-bg); border-top:1px solid var(--border); padding:3px 13px;
      font-size:.65rem; color:#1e3040; display:flex; gap:16px; flex-shrink:0; }}
#sb b {{ color:#2a4050; }}
</style>
</head>
<body>

<div id="hdr">
  <div>
    <h1>&#128202; {FINAL} &mdash; ER Diagram</h1>
    <div class="sub">Powered by Sigma.js &bull; Click any node for details &bull; Scroll to zoom &bull; Drag to pan</div>
  </div>
  <span class="sp sp-b" id="sp-ikg">{ikg_count} IKG tables</span>
  <span class="sp sp-p" id="sp-tmp">{tmp_count} Temp tables</span>
  <span class="sp sp-o" id="sp-ext">{ext_count} External tables</span>
  <span class="sp sp-g" id="sp-rel">{rel_count} Relationships</span>
</div>

<div id="tb">
  <button class="btn" onclick="resetCamera()">&#8635; Reset</button>
  <button class="btn" onclick="zoomIn()">+ Zoom</button>
  <button class="btn" onclick="zoomOut()">&#8722; Zoom</button>
  <div class="sep"></div>
  <input id="srch" placeholder="&#128269; Search..." oninput="onSearch()">
  <button class="btn" onclick="clearSearch()">&#10005;</button>
  <div class="sep"></div>
  <button class="btn on" id="btn-ikg" onclick="toggleType('ikg')">IKG</button>
  <button class="btn on" id="btn-tmp" onclick="toggleType('tmp')">Temp</button>
  <button class="btn on" id="btn-ext" onclick="toggleType('ext')">External</button>
  <div id="leg">
    <div class="li"><div class="lb" style="background:var(--final)"></div>Final</div>
    <div class="li"><div class="lb" style="background:var(--ikg)"></div>IKG</div>
    <div class="li"><div class="lb" style="background:var(--tmp)"></div>Temp</div>
    <div class="li"><div class="lb" style="background:var(--ext)"></div>External</div>
  </div>
</div>

<div id="wrap">
  <div id="sigma-container"></div>
<div id="load-error" style="display:none;position:absolute;top:50%;left:50%;
  transform:translate(-50%,-50%);background:#1a2535;border:1px solid #2a78b8;
  border-radius:8px;padding:24px 32px;text-align:center;color:#90b8d8;font-size:.9rem;z-index:100;">
  <div style="font-size:1.2rem;margin-bottom:8px;">&#9888; Libraries failed to load</div>
  <div style="color:#4a6070;font-size:.78rem;">Requires internet connection to load Sigma.js from CDN.<br>
  Open this file in a browser with internet access.</div>
</div>
  <div id="panel">
    <div id="ph">
      <button id="ph-x" onclick="closePanel()">&#10005;</button>
      <div id="ph-title"></div>
      <div id="ph-meta"></div>
    </div>
    <div id="pb">
      <div class="sec">
        <div class="sechdr c-cols" id="hdr-cols" onclick="toggleSec('cols')">
          &#128196;&nbsp;Columns
          <span class="seccnt" id="cnt-cols">0</span>
          <span class="arr">&#9660;</span>
        </div>
        <div class="secbody" id="body-cols"></div>
      </div>
      <div class="sec">
        <div class="sechdr c-src" id="hdr-src" onclick="toggleSec('src')">
          &#8593;&nbsp;Source Tables <em style="color:#1a4828;font-weight:400;font-size:.6rem;text-transform:none">(inputs)</em>
          <span class="seccnt" id="cnt-src">0</span>
          <span class="arr">&#9660;</span>
        </div>
        <div class="secbody" id="body-src"></div>
      </div>
      <div class="sec">
        <div class="sechdr c-dst collapsed" id="hdr-dst" onclick="toggleSec('dst')">
          &#8595;&nbsp;Destination Tables <em style="color:#482810;font-weight:400;font-size:.6rem;text-transform:none">(outputs)</em>
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
  <span id="sb-msg">Scroll to zoom &bull; Drag to pan &bull; Click node for details</span>
</div>

<script>
const GRAPH_DATA = {data_json};

// ── Type colours ──────────────────────────────────────────────────────────────
const TC = {{ final:'#2ecc71', ikg:'#2a78b8', tmp:'#7840c8', ext:'#c06020' }};
const TL = {{ final:'Final Output', ikg:'IKG Table', tmp:'Temp Table', ext:'External/EDW' }};

// ── Build Graphology graph ────────────────────────────────────────────────────
// Check libraries loaded
if (typeof graphology === 'undefined' || typeof Sigma === 'undefined') {{
  document.getElementById('load-error').style.display = 'block';
  document.getElementById('sigma-container').style.display = 'none';
  throw new Error('Sigma.js or Graphology not loaded. Check internet connection.');
}}
const graph = new graphology.Graph({{ multi:false, type:'directed' }});

GRAPH_DATA.nodes.forEach(function(n) {{
  graph.addNode(n.id, {{
    label:   n.id,
    x:       n.x,
    y:       n.y,
    size:    n.size,
    color:   n.color,
    type:    'circle',   // Sigma renderer type — must be 'circle'
    ntype:   n.ntype,    // our custom category: final/ikg/tmp/ext
    level:   n.level,
    cols:    n.cols || [],
    hidden:  false,
  }});
}});

GRAPH_DATA.edges.forEach(function(e) {{
  try {{
    graph.addEdge(e.source, e.target, {{
      id:         e.id,
      size:       1,
      color:      '#1e3048',
      src_alias:  e.src_alias || '',
      tgt_alias:  e.tgt_alias || '',
      jk:         e.jk || [],
      is_external: e.is_external || false,
    }});
  }} catch(err) {{ /* skip duplicate edges */ }}
}});

// ── Sigma renderer ────────────────────────────────────────────────────────────
const container = document.getElementById('sigma-container');

// ── State — declared BEFORE Sigma so reducers can reference them ─────────────
let selectedNode     = null;
let highlightedNodes = new Set();
let highlightedEdges = new Set();
let hiddenTypes      = new Set();
let srchQ            = '';

const renderer = new Sigma(graph, container, {{
  renderEdgeLabels:      false,
  defaultEdgeType:       'arrow',
  defaultNodeType:       'circle',
  nodeReducer: function(node, data) {{
    const res = Object.assign({{}}, data);
    res.type = 'circle';  // Always force Sigma's renderer type
    if (highlightedNodes.size > 0) {{
      if (!highlightedNodes.has(node)) {{
        res.color   = '#1a2535';
        res.label   = undefined;
        res.zIndex  = 0;
      }} else {{
        res.size    = data.size * 1.4;
        res.zIndex  = 1;
      }}
    }}
    if (srchQ && !node.toLowerCase().includes(srchQ)) {{
      res.color  = '#1a2535';
      res.label  = undefined;
    }}
    if (hiddenTypes.has(data.ntype) && data.ntype !== 'final') {{
      res.hidden = true;
    }}
    return res;
  }},
  edgeReducer: function(edge, data) {{
    const res = Object.assign({{}}, data);
    const src = graph.source(edge);
    const tgt = graph.target(edge);
    if (highlightedNodes.size > 0) {{
      if (highlightedNodes.has(src) && highlightedNodes.has(tgt)) {{
        res.color  = highlightedEdges.has(edge) ?
          (tgt === selectedNode ? '#4ec87a' : '#e89030') : '#1e3048';
        res.size   = highlightedEdges.has(edge) ? 2 : 1;
        res.zIndex = 1;
      }} else {{
        res.color  = '#0d1828';
        res.size   = 0.5;
      }}
    }}
    return res;
  }},
  labelFont:                  'Segoe UI, system-ui, sans-serif',
  labelSize:                  11,
  labelColor:                 {{ color:'#8ab0c8' }},
  labelThreshold:             0,
  labelRenderedSizeThreshold: 0,
  minCameraRatio: 0.02,
  maxCameraRatio: 8,
}});

// ── State ─────────────────────────────────────────────────────────────────────
// ── Edge index ────────────────────────────────────────────────────────────────
const edgesFrom = {{}};  // nodeId -> [edgeObj]
const edgesTo   = {{}};  // nodeId -> [edgeObj]
GRAPH_DATA.edges.forEach(function(e) {{
  if (!edgesFrom[e.source]) edgesFrom[e.source] = [];
  if (!edgesTo[e.target])   edgesTo[e.target]   = [];
  edgesFrom[e.source].push(e);
  edgesTo[e.target].push(e);
}});

// ── Click handler ─────────────────────────────────────────────────────────────
renderer.on('clickNode', function(evt) {{
  selectNode(evt.node);
}});

renderer.on('clickStage', function() {{
  clearSelection();
  closePanel();
}});

renderer.on('doubleClickNode', function(evt) {{
  // Center camera on node
  const attrs = graph.getNodeAttributes(evt.node);
  renderer.getCamera().animate({{ x: attrs.x, y: attrs.y, ratio: 0.3 }}, {{ duration: 400 }});
}});

// Update zoom display on camera change
renderer.getCamera().on('updated', function() {{
  var ratio = renderer.getCamera().ratio;
  document.getElementById('sb-z').textContent = Math.round(100 / ratio) + '%';
}});

function selectNode(nodeId) {{
  selectedNode = nodeId;
  highlightedNodes = new Set([nodeId]);
  highlightedEdges = new Set();

  // Highlight all neighbours and their edges
  graph.forEachNeighbor(nodeId, function(neighbor) {{
    highlightedNodes.add(neighbor);
  }});

  // Collect highlighted edges
  try {{
    graph.forEachEdge(nodeId, function(edge) {{
      highlightedEdges.add(edge);
      const src = graph.source(edge);
      const tgt = graph.target(edge);
      highlightedNodes.add(src);
      highlightedNodes.add(tgt);
    }});
  }} catch(e) {{}}

  renderer.refresh();
  openPanel(nodeId);
}}

function clearSelection() {{
  selectedNode     = null;
  highlightedNodes = new Set();
  highlightedEdges = new Set();
  renderer.refresh();
}}

// ── Panel ─────────────────────────────────────────────────────────────────────
function openPanel(nodeId) {{
  const nodeData = GRAPH_DATA.nodes.find(function(n) {{ return n.id === nodeId; }});
  if (!nodeData) return;

  document.getElementById('panel').classList.add('open');
  document.getElementById('ph-title').textContent = nodeId;

  const inEdges  = edgesTo[nodeId]   || [];
  const outEdges = edgesFrom[nodeId] || [];

  document.getElementById('ph-meta').textContent =
    TL[nodeData.ntype] + ' \u2022 Level ' + nodeData.level +
    ' \u2022 ' + nodeData.cols.length + ' cols' +
    ' \u2022 ' + inEdges.length + ' inputs' +
    ' \u2022 ' + outEdges.length + ' outputs';
  document.getElementById('ph-meta').style.color = TC[nodeData.ntype] + '88';

  buildCols(nodeData.cols);
  buildRelSec('src', inEdges, true);
  buildRelSec('dst', outEdges, false);
}}

function buildCols(cols) {{
  document.getElementById('cnt-cols').textContent = cols.length;
  var body = document.getElementById('body-cols');
  body.innerHTML = '';
  if (!cols.length) {{
    var em = document.createElement('div');
    em.style.cssText = 'padding:7px 13px;color:#1a2e3e;font-size:.7rem;font-style:italic;';
    em.textContent = 'No columns found (external/base table or not in parsed files)';
    body.appendChild(em); return;
  }}
  cols.forEach(function(col, i) {{
    var row = document.createElement('div'); row.className = 'col-row';
    var idx = document.createElement('span'); idx.className = 'col-idx'; idx.textContent = (i+1) + '.';
    var nm  = document.createElement('span'); nm.className  = 'col-name';  nm.textContent  = col;
    row.appendChild(idx); row.appendChild(nm); body.appendChild(row);
  }});
}}

function buildRelSec(sec, edges, isSource) {{
  document.getElementById('cnt-' + sec).textContent = edges.length;
  var body = document.getElementById('body-' + sec);
  body.innerHTML = '';
  if (!edges.length) {{
    var em = document.createElement('div');
    em.style.cssText = 'padding:7px 13px;color:#162030;font-size:.7rem;font-style:italic;';
    em.textContent = isSource ? 'No source tables' : 'No downstream tables';
    body.appendChild(em); return;
  }}
  edges.forEach(function(e) {{
    var otherId = isSource ? e.source : e.target;
    var otherNode = GRAPH_DATA.nodes.find(function(n) {{ return n.id === otherId; }});
    var ntype = otherNode ? otherNode.ntype : 'ext';
    var col   = TC[ntype] || '#718096';
    var fromAlias = e.src_alias || '', toAlias = e.tgt_alias || '';
    var displayAlias = isSource ? fromAlias : toAlias;

    var row = document.createElement('div'); row.className = 'trow';
    row.addEventListener('click', function() {{
      selectNode(otherId);
      // Fly camera to node
      var attrs = graph.getNodeAttributes(otherId);
      if (attrs) renderer.getCamera().animate({{ x:attrs.x, y:attrs.y, ratio:0.4 }}, {{duration:400}});
    }});

    var top = document.createElement('div'); top.className = 'trow-top';
    var dot = document.createElement('div'); dot.className = 'tdot';
    dot.style.background = col; dot.style.border = '1px solid ' + col + '55';
    var nm = document.createElement('div'); nm.className = 'tname'; nm.textContent = otherId;
    top.appendChild(dot); top.appendChild(nm);
    if (displayAlias) {{
      var ab = document.createElement('span'); ab.className = 'talias';
      ab.textContent = 'alias: ' + displayAlias; top.appendChild(ab);
    }}
    row.appendChild(top);

    var jks = e.jk || [];
    if (jks.length) {{
      var tbl = document.createElement('table'); tbl.className = 'jktbl';
      var thead = tbl.createTHead(), hr = thead.insertRow();
      function th(t) {{ var el=document.createElement('th'); el.textContent=t; hr.appendChild(el); }}
      th('Source (' + (fromAlias || e.source.slice(0,14)) + ')');
      th('');
      th('Dest (' + (toAlias || e.target.slice(0,14)) + ')');
      var tb2 = tbl.createTBody();
      jks.forEach(function(jk) {{
        var tr = tb2.insertRow();
        if (jk.left && jk.right) {{
          tr.insertCell().className = 'jk-s'; tr.cells[0].textContent = jk.left;
          tr.insertCell().className = 'jk-e'; tr.cells[1].textContent = '=';
          tr.insertCell().className = 'jk-d'; tr.cells[2].textContent = jk.right;
        }} else if (jk.col) {{
          var td = tr.insertCell(); td.className = 'jk-s';
          td.textContent = jk.col; td.colSpan = 3;
        }}
      }});
      row.appendChild(tbl);
    }} else {{
      var nj = document.createElement('div'); nj.className = 'no-jk';
      nj.textContent = '(no explicit join key)'; row.appendChild(nj);
    }}
    body.appendChild(row);
  }});
}}

function closePanel() {{
  document.getElementById('panel').classList.remove('open');
  clearSelection();
}}

function toggleSec(s) {{
  document.getElementById('hdr-' + s).classList.toggle('collapsed');
}}

// ── Controls ──────────────────────────────────────────────────────────────────
function resetCamera() {{
  renderer.getCamera().animate(graphology.graphologyLibrary.layoutUtils.circlepack ?
    {{x:0.5, y:0.5, ratio:1}} : {{x:0.5, y:0.5, ratio:1}}, {{duration:400}});
  renderer.getCamera().animatedReset({{duration:600}});
}}

function zoomIn()  {{ renderer.getCamera().animatedZoom({{factor:1.4, duration:300}}); }}
function zoomOut() {{ renderer.getCamera().animatedUnzoom({{factor:1.4, duration:300}}); }}

function toggleType(t) {{
  var btn = document.getElementById('btn-' + t);
  if (hiddenTypes.has(t)) {{
    hiddenTypes.delete(t);
    btn.classList.add('on');
  }} else {{
    hiddenTypes.add(t);
    btn.classList.remove('on');
  }}
  renderer.refresh();
}}

function onSearch() {{
  srchQ = document.getElementById('srch').value.toLowerCase().trim();
  renderer.refresh();
}}

function clearSearch() {{
  srchQ = '';
  document.getElementById('srch').value = '';
  renderer.refresh();
}}

// ── Fit graph to screen on load ───────────────────────────────────────────────
window.addEventListener('load', function() {{
  renderer.getCamera().animatedReset({{duration:800}});
}});
</script>
</body>
</html>
"""
    return html


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    script_dir = Path(__file__).resolve().parent

    print("=" * 60)
    print("  SQL Parser — Sigma.js ER Diagram Generator")
    print("=" * 60)
    print(f"\n  Script folder: {script_dir}\n")

    sql_files = sorted(list(script_dir.glob("*.sql")) + list(script_dir.glob("*.txt")))
    if sql_files:
        print("  SQL files found:")
        for f in sql_files:
            print(f"    • {f.name}")
    else:
        print("  ⚠  No SQL/TXT files found in this folder.")
    print()

    while True:
        raw = input("  Enter SQL filename: ").strip()
        if not raw:
            print("  ⚠  Filename cannot be empty.\n"); continue
        candidate = script_dir / raw
        if not candidate.exists():
            candidate = Path(raw).resolve()
        if candidate.exists() and candidate.is_file():
            sql_path = candidate
            print(f"\n  ✅ Found: {sql_path}\n"); break
        else:
            print(f"  ❌ Not found: {script_dir / raw}\n")

    print(f"[*] Reading {sql_path.name} ({sql_path.stat().st_size:,} bytes)…")
    with open(sql_path, encoding="utf-8", errors="replace") as f:
        sql_text = f.read()

    # Load extra SQL files for column enrichment
    extra_sqls = []
    for ext in ("*.sql", "*.txt"):
        for f2 in script_dir.glob(ext):
            if f2.resolve() != sql_path.resolve():
                try:
                    with open(f2, encoding="utf-8", errors="replace") as fh:
                        extra_sqls.append(fh.read())
                    print(f"  [+] Extra SQL loaded: {f2.name}")
                except Exception:
                    pass

    print("[*] Parsing SQL…")
    parser = SQLParser(sql_text)
    parser.parse()
    summary = parser.summary()
    summary['final_table']     = auto_detect_final_table(summary, sql_text)
    summary['_sql_text']       = sql_text
    summary['_extra_sql_texts'] = extra_sqls

    print("\n" + "=" * 60)
    print("  PARSE SUMMARY")
    print("=" * 60)
    print(f"  Final table   : {summary['final_table']}")
    print(f"  IKG tables    : {len(summary['created_tables'])}")
    print(f"  Temp tables   : {len(summary['temp_tables'])}")
    print(f"  External      : {len(summary['external_tables'])}")
    print(f"  Relationships : {len(summary['relationships'])}")

    stem = sql_path.stem

    json_out = script_dir / f"{stem}_sigma_lineage.json"
    export   = {k: v for k, v in summary.items() if k not in ('_sql_text', '_extra_sql_texts')}
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2)
    print(f"\n[✓] Lineage JSON  → {json_out}")

    html_out = script_dir / f"{stem}_sigma_er_diagram.html"
    with open(html_out, "w", encoding="utf-8") as f:
        f.write(build_sigma_html(summary))
    print(f"[✓] ER Diagram    → {html_out}")
    print("\n  Open the HTML file in any modern browser.")
    print("=" * 60)


if __name__ == "__main__":
    main()
