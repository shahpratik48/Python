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

    col_map = {}
    for extra_sql in summary.get('_extra_sql_texts', []):
        col_map.update(extract_table_columns(extra_sql))
    if sql_text:
        col_map.update(extract_table_columns(sql_text))

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
        if a in ALIAS_BL or not re.match(r'^[a-z_][a-z0-9_]{0,29}$', a): return ''
        return a

    merged_edges = {}
    for r in summary['relationships']:
        src_t, tgt_t = r['from'], r['to']
        if src_t not in all_tables or tgt_t not in all_tables: continue
        key = (src_t, tgt_t)
        if key not in merged_edges:
            merged_edges[key] = {'from': src_t, 'to': tgt_t,
                'src_alias': clean_alias(r.get('alias','')),
                'tgt_alias': clean_alias(r.get('target_alias','')),
                'jk': [], 'is_external': r.get('is_external', False)}
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

    from collections import defaultdict as _dd, deque as _dq
    in_em = _dd(set); out_em = _dd(set)
    for e in edges_list:
        in_em[e['to']].add(e['from']); out_em[e['from']].add(e['to'])
    in_deg = {t: len(in_em[t]) for t in all_tables}
    queue  = _dq([t for t in all_tables if in_deg[t] == 0])
    level  = {t: 0 for t in queue}
    while queue:
        t = queue.popleft()
        for nxt in out_em[t]:
            if nxt in all_tables:
                in_deg[nxt] = max(0, in_deg.get(nxt,0) - 1)
                level[nxt]  = max(level.get(nxt,0), level[t]+1)
                if in_deg[nxt] == 0: queue.append(nxt)
    for t in all_tables:
        if t not in level: level[t] = 0

    def node_type(n):
        if n == FINAL: return 'final'
        if n in created_set: return 'ikg'
        if n in temp_set: return 'tmp'
        return 'ext'

    max_level = max(level.values()) if level else 0
    by_level  = _dd(list)
    for t in all_tables: by_level[level[t]].append(t)
    for lv in by_level:
        by_level[lv].sort(key=lambda t: -(len(in_em[t]) + len(out_em[t])))

    BOX_W = 220; HDR_H = 30; ROW_H = 17; PREV_N = 4; FOOT_H = 10
    COL_GAP = 90; ROW_GAP = 22; PAD_X = 70; PAD_Y = 60

    def box_h(cols):
        rows = min(len(cols), PREV_N)
        return HDR_H + (rows * ROW_H + 8 if rows else 0) + FOOT_H

    sorted_lvs = sorted(by_level.keys(), reverse=True)
    xCursor = PAD_X; colX = {}
    for lv in sorted_lvs:
        colX[lv] = xCursor; xCursor += BOX_W + COL_GAP

    nodes_data = []
    for lv in sorted_lvs:
        yCursor = PAD_Y
        for t in by_level[lv]:
            cols = col_map.get(t, []); bh = box_h(cols)
            nodes_data.append({'id': t, 'x': colX[lv], 'y': yCursor,
                'w': BOX_W, 'h': bh, 'ntype': node_type(t),
                'level': level.get(t,0), 'cols': cols})
            yCursor += bh + ROW_GAP

    ikg_count = len([n for n in nodes_data if n['ntype'] in ('ikg','final')])
    tmp_count = len([n for n in nodes_data if n['ntype'] == 'tmp'])
    ext_count = len([n for n in nodes_data if n['ntype'] == 'ext'])
    rel_count = len(edges_list)

    import json as _j
    nodes_json = _j.dumps(nodes_data, separators=(',',':'))
    edges_json = _j.dumps(edges_list, separators=(',',':'))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{FINAL} — ER Diagram</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/graphology@0.25.4/dist/graphology.umd.min.js" crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/npm/sigma@2.4.0/build/sigma.min.js" crossorigin="anonymous"></script>
<style>
:root{{
  --bg0:#f8fafc; --bg1:#f1f5f9; --bg2:#e2e8f0;
  --glass:#00000005; --glass2:#0000000a;
  --border:#e2e8f0; --border2:#cbd5e1;
  --final-c:#059669; --final-g:#047857;
  --ikg-c:#2563eb;   --ikg-g:#1d4ed8;
  --tmp-c:#7c3aed;   --tmp-g:#6d28d9;
  --ext-c:#d97706;   --ext-g:#b45309;
  --text:#0f172a; --text2:#475569; --text3:#94a3b8;
  --panel:#fffffff5;
  --accent:#2563eb;
}}
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0;}}
body{{
  background:var(--bg0);
  background-image:
    radial-gradient(ellipse 70% 50% at 15% 10%, #2563eb08 0%, transparent 60%),
    radial-gradient(ellipse 50% 40% at 85% 85%, #7c3aed06 0%, transparent 50%),
    radial-gradient(ellipse 60% 60% at 50% 50%, #05966904 0%, transparent 70%);
  color:var(--text);
  font-family:'Sora',system-ui,sans-serif;
  overflow:hidden; height:100vh; display:flex; flex-direction:column;
}}

/* ── HEADER ─────────────────────────────────────────────────── */
#hdr{{
  padding:13px 24px;
  background:linear-gradient(135deg, #1e3a5f 0%, #162d4a 50%, #1a3354 100%);
  border-bottom:2px solid #0ea5e944;
  box-shadow:0 2px 16px #00000022;
  display:flex; align-items:center; gap:16px; flex-shrink:0;
  position:relative; overflow:hidden;
}}
#hdr::before{{
  content:''; position:absolute; inset:0;
  background:linear-gradient(90deg, #0ea5e912 0%, #7c3aed0a 50%, transparent 100%);
  pointer-events:none;
}}
#hdr::after{{
  content:''; position:absolute; bottom:0; left:0; right:0; height:2px;
  background:linear-gradient(90deg, #2563eb, #7c3aed, #059669, transparent);
  opacity:.6;
}}
.hdr-icon{{font-size:1.4rem; filter:drop-shadow(0 0 10px #38bdf8cc);}}
.hdr-title{{font-size:1rem; font-weight:700; color:#f0f9ff; letter-spacing:-.3px;
  text-shadow:0 0 20px #0ea5e944;}}
.hdr-sub{{font-size:.65rem; color:#94a3b8; margin-top:2px; font-weight:300; letter-spacing:.1px;}}
.badges{{display:flex; gap:8px; margin-left:auto; flex-wrap:wrap;}}
.badge{{
  padding:3px 12px; border-radius:20px; font-size:.63rem; font-weight:600;
  letter-spacing:.3px; text-transform:uppercase;
  border:1px solid; backdrop-filter:blur(8px);
}}
.badge-ikg{{background:#1e40af22;color:#93c5fd;border-color:#3b82f644;}}
.badge-tmp{{background:#5b21b622;color:#c4b5fd;border-color:#8b5cf644;}}
.badge-ext{{background:#92400e22;color:#fcd34d;border-color:#f59e0b44;}}
.badge-rel{{background:#06402022;color:#6ee7b7;border-color:#10b98144;}}

/* ── TOOLBAR ────────────────────────────────────────────────── */
#tb{{
  background:linear-gradient(180deg, #f1f5f9 0%, #e8eef5 100%);
  border-bottom:1px solid #cbd5e1;
  box-shadow:0 1px 6px #0000000d;
  padding:5px 16px; display:flex; align-items:center; gap:5px;
  flex-shrink:0; flex-wrap:wrap;
}}
.btn{{
  background:#ffffff; color:#475569;
  border:1px solid #cbd5e1; border-radius:6px;
  padding:4px 12px; font-size:.7rem; cursor:pointer;
  font-family:'Sora',sans-serif; font-weight:500;
  transition:all .15s; letter-spacing:.2px;
  box-shadow:0 1px 3px #0000000f;
}}
.btn:hover{{background:#f8fafc;color:#1e293b;border-color:#94a3b8;box-shadow:0 2px 6px #0000001a;}}
.btn.on{{background:#dbeafe;color:#1d4ed8;border-color:#93c5fd;}}
.sep{{width:1px;height:18px;background:#cbd5e1;margin:0 3px;}}
#srch{{
  background:#ffffff; color:#1e293b;
  border:1px solid #cbd5e1; border-radius:6px;
  padding:4px 12px; font-size:.7rem; width:180px; outline:none;
  font-family:'Sora',sans-serif; transition:all .15s;
  box-shadow:0 1px 3px #0000000f;
}}
#srch:focus{{border-color:#2563eb;box-shadow:0 0 0 3px #2563eb18;}}
#srch::placeholder{{color:var(--text3);}}
#leg{{display:flex;gap:12px;margin-left:auto;align-items:center;}}
.li{{display:flex;align-items:center;gap:5px;font-size:.62rem;color:#64748b;font-weight:500;}}
.lb{{width:10px;height:10px;border-radius:3px;}}

/* ── MAIN AREA ──────────────────────────────────────────────── */
#wrap{{flex:1;display:flex;overflow:hidden;}}
#cv-wrap{{flex:1;position:relative;overflow:hidden;min-width:0;}}
#cv{{display:block;position:absolute;top:0;left:0;cursor:grab;}}
#cv:active{{cursor:grabbing;}}

/* ── PANEL ──────────────────────────────────────────────────── */
#panel{{
  width:420px;min-width:420px;
  background:var(--panel);
  backdrop-filter:blur(24px);
  border-left:1px solid var(--border2);
  box-shadow:-4px 0 24px #00000010;
  display:none;flex-direction:column;overflow:hidden;flex-shrink:0;
}}
#panel.open{{display:flex;}}
#ph{{
  padding:14px 16px 12px;
  border-bottom:1px solid var(--border2);
  flex-shrink:0;position:relative;
  background:linear-gradient(135deg,#2563eb06,transparent);
}}
#ph::after{{
  content:'';position:absolute;bottom:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,var(--accent)33,transparent);
}}
#ph-title{{
  font-size:.9rem;font-weight:700;color:var(--text);
  word-break:break-all;margin-bottom:4px;padding-right:28px;
  font-family:'JetBrains Mono',monospace;
  text-shadow:none;
}}
#ph-meta{{font-size:.65rem;color:var(--text3);font-weight:300;line-height:1.5;}}
#ph-x{{
  position:absolute;right:14px;top:14px;background:none;border:none;
  color:var(--text3);cursor:pointer;font-size:1.1rem;
  width:24px;height:24px;display:flex;align-items:center;justify-content:center;
  border-radius:4px;transition:all .15s;
}}
#ph-x:hover{{background:var(--bg2);color:var(--text);}}
#pb{{flex:1;overflow-y:auto;}}
#pb::-webkit-scrollbar{{width:3px;}}
#pb::-webkit-scrollbar-track{{background:transparent;}}
#pb::-webkit-scrollbar-thumb{{background:#cbd5e1;border-radius:2px;}}
#pb::-webkit-scrollbar-thumb:hover{{background:#94a3b8;}}

.sec{{border-bottom:1px solid var(--border);}}
.sechdr{{
  padding:9px 16px;font-size:.62rem;font-weight:600;
  text-transform:uppercase;letter-spacing:1px;
  display:flex;align-items:center;gap:8px;
  cursor:pointer;user-select:none;transition:background .12s;
  font-family:'Sora',sans-serif;
}}
.sechdr:hover{{background:#f8fafc;}}
.sechdr.c-cols{{color:#38bdf8;}}.sechdr.c-src{{color:#34d399;}}.sechdr.c-dst{{color:#fbbf24;}}
.seccnt{{
  margin-left:auto;background:var(--bg1);color:var(--text2);
  font-size:.58rem;padding:1px 8px;border-radius:10px;font-weight:700;
  border:1px solid var(--border2);
}}
.arr{{font-size:.6rem;color:var(--text3);transition:transform .15s;}}
.sechdr.collapsed .arr{{transform:rotate(-90deg);}}
.secbody{{display:block;}}.sechdr.collapsed+.secbody{{display:none;}}

.col-row{{
  padding:5px 16px 5px 24px;display:flex;align-items:center;gap:10px;
  border-bottom:1px solid #ffffff06;font-size:.72rem;
  transition:background .1s;
}}
.col-row:hover{{background:#f8fafc;}}
.col-row:last-child{{border-bottom:none;}}
.col-idx{{color:var(--text3);font-size:.58rem;min-width:20px;font-family:'JetBrains Mono',monospace;}}
.col-name{{color:#2563eb;font-family:'JetBrains Mono',monospace;font-size:.7rem;}}

.trow{{padding:8px 14px 8px 16px;border-bottom:1px solid #ffffff06;cursor:pointer;transition:background .12s;}}
.trow:hover{{background:#f8fafc;}}
.trow-top{{display:flex;align-items:flex-start;gap:8px;margin-bottom:5px;}}
.tdot{{width:8px;height:8px;border-radius:2px;flex-shrink:0;margin-top:4px;}}
.tname{{font-size:.73rem;font-weight:500;color:var(--text);flex:1;word-break:break-all;
  font-family:'JetBrains Mono',monospace;}}
.talias{{font-size:.6rem;color:var(--text2);background:var(--bg1);
  border:1px solid var(--border2);border-radius:4px;padding:1px 6px;
  font-family:'JetBrains Mono',monospace;white-space:nowrap;}}
.jktbl{{width:calc(100% - 16px);border-collapse:collapse;margin:3px 0 3px 16px;}}
.jktbl th{{font-size:.57rem;color:var(--text3);font-weight:600;text-transform:uppercase;
  letter-spacing:.5px;padding:2px 6px 2px 0;border-bottom:1px solid var(--border2);}}
.jktbl td{{font-size:.65rem;font-family:'JetBrains Mono',monospace;
  padding:3px 6px 3px 0;vertical-align:top;border-bottom:1px solid var(--border);}}
.jktbl tr:last-child td{{border-bottom:none;}}
.jk-s{{color:#059669;}}.jk-e{{color:var(--text3);padding:0 4px;}}.jk-d{{color:#d97706;}}
.no-jk{{font-size:.6rem;color:var(--text3);padding:3px 0 3px 16px;font-style:italic;}}

/* ── STATUS BAR ─────────────────────────────────────────────── */
#sb{{
  background:#ffffffcc;backdrop-filter:blur(8px);
  border-top:1px solid var(--border2);padding:4px 16px;
  font-size:.62rem;color:var(--text2);display:flex;gap:20px;
  flex-shrink:0;font-family:'JetBrains Mono',monospace;
  box-shadow:0 -1px 4px #00000008;
}}
#sb b{{color:var(--text);font-weight:600;}}
</style>
</head>
<body>

<div id="hdr">
  <span class="hdr-icon">⬡</span>
  <div>
    <div class="hdr-title">{FINAL}</div>
    <div class="hdr-sub">SQL Pipeline ER Diagram &nbsp;·&nbsp; Drag boxes to reposition &nbsp;·&nbsp; Click any box for details</div>
  </div>
  <div class="badges">
    <span class="badge badge-ikg">{ikg_count} IKG</span>
    <span class="badge badge-tmp">{tmp_count} Temp</span>
    <span class="badge badge-ext">{ext_count} External</span>
    <span class="badge badge-rel">{rel_count} Relations</span>
  </div>
</div>

<div id="tb">
  <button class="btn" onclick="resetView()">⟳ Reset</button>
  <button class="btn" onclick="doZoom(1.25)">＋ Zoom</button>
  <button class="btn" onclick="doZoom(0.8)">－ Zoom</button>
  <div class="sep"></div>
  <input id="srch" placeholder="⌕  Search tables..." oninput="onSearch()">
  <button class="btn" onclick="clearSearch()">✕</button>
  <div class="sep"></div>
  <button class="btn on" id="btn-ikg" onclick="toggleType('ikg')">IKG</button>
  <button class="btn on" id="btn-tmp" onclick="toggleType('tmp')">Temp</button>
  <button class="btn on" id="btn-ext" onclick="toggleType('ext')">External</button>
  <div id="leg">
    <div class="li"><div class="lb" style="background:#059669;box-shadow:0 0 4px #05966944;"></div>Final</div>
    <div class="li"><div class="lb" style="background:#2563eb;box-shadow:0 0 4px #2563eb44;"></div>IKG</div>
    <div class="li"><div class="lb" style="background:#7c3aed;box-shadow:0 0 4px #7c3aed44;"></div>Temp</div>
    <div class="li"><div class="lb" style="background:#d97706;box-shadow:0 0 4px #d9770644;"></div>Ext</div>
  </div>
</div>

<div id="wrap">
  <div id="cv-wrap">
    <canvas id="cv"></canvas>
  </div>
  <div id="panel">
    <div id="ph">
      <button id="ph-x" onclick="closePanel()">✕</button>
      <div id="ph-title"></div>
      <div id="ph-meta"></div>
    </div>
    <div id="pb">
      <div class="sec">
        <div class="sechdr c-cols" id="hdr-cols" onclick="toggleSec('cols')">
          ◈ Columns <span class="seccnt" id="cnt-cols">0</span><span class="arr">▾</span>
        </div><div class="secbody" id="body-cols"></div>
      </div>
      <div class="sec">
        <div class="sechdr c-src" id="hdr-src" onclick="toggleSec('src')">
          ↑ Source Tables <em style="font-style:normal;text-transform:none;font-size:.58rem;color:#1e3a2e;letter-spacing:0;">inputs</em>
          <span class="seccnt" id="cnt-src">0</span><span class="arr">▾</span>
        </div><div class="secbody" id="body-src"></div>
      </div>
      <div class="sec">
        <div class="sechdr c-dst collapsed" id="hdr-dst" onclick="toggleSec('dst')">
          ↓ Destination Tables <em style="font-style:normal;text-transform:none;font-size:.58rem;color:#3a2a0e;letter-spacing:0;">outputs</em>
          <span class="seccnt" id="cnt-dst">0</span><span class="arr">▾</span>
        </div><div class="secbody" id="body-dst"></div>
      </div>
    </div>
  </div>
</div>

<div id="sb">
  <span>zoom <b id="sb-z">100%</b></span>
  <span>visible <b id="sb-v">0</b></span>
  <span>drag box · pan background · scroll zoom · click for details</span>
</div>

<script>
const NODES = {nodes_json};
const EDGES = {edges_json};
const FINAL = "{FINAL}";

// ── Colour palette ─────────────────────────────────────────────────────────────
const STROKE = {{final:'#059669',ikg:'#2563eb',tmp:'#7c3aed',ext:'#d97706'}};
const FILL   = {{final:'#f0fdf4',ikg:'#eff6ff',tmp:'#f5f3ff',ext:'#fffbeb'}};
const HDR_C  = {{final:'#dcfce7',ikg:'#dbeafe',tmp:'#ede9fe',ext:'#fef3c7'}};
const TEXT_C = {{final:'#065f46',ikg:'#1e40af',tmp:'#5b21b6',ext:'#92400e'}};
const SUB_C  = {{final:'#6ee7b7',ikg:'#93c5fd',tmp:'#c4b5fd',ext:'#fde68a'}};
const TL     = {{final:'Final Output',ikg:'IKG Table',tmp:'Temp Table',ext:'External/EDW'}};
const HDR_H=30, ROW_H=17, PREV_N=4;

// ── Index ──────────────────────────────────────────────────────────────────────
const NM={{}};
NODES.forEach(function(n){{NM[n.id]=n;}});
const EF={{}}, ET={{}};
EDGES.forEach(function(e){{
  if(!EF[e.from]) EF[e.from]=[];
  if(!ET[e.to])   ET[e.to]=[];
  EF[e.from].push(e); ET[e.to].push(e);
}});

// ── State ──────────────────────────────────────────────────────────────────────
const cv=document.getElementById('cv'), ctx=cv.getContext('2d');
let tx=0,ty=0,sc=1;
let selId=null, hlNodes=new Set(), hiddenT=new Set(), srchQ='';
let panning=null, dragN=null, dox=0,doy=0, mdx=0,mdy=0, nodeMoved=false;

function s2w(sx,sy){{return{{x:(sx-tx)/sc,y:(sy-ty)/sc}};}}
function isVis(n){{
  if(n.ntype!=='final'&&hiddenT.has(n.ntype)) return false;
  if(srchQ&&n.id.toLowerCase().indexOf(srchQ)<0) return false;
  return true;
}}
function hitNode(wx,wy){{
  for(var i=NODES.length-1;i>=0;i--){{
    var n=NODES[i]; if(!isVis(n)) continue;
    if(wx>=n.x&&wx<=n.x+n.w&&wy>=n.y&&wy<=n.y+n.h) return n;
  }}
  return null;
}}

// ── Render ─────────────────────────────────────────────────────────────────────
function render(){{
  ctx.clearRect(0,0,cv.width,cv.height);
  // Deep space background gradient
  var grad=ctx.createRadialGradient(cv.width*.25,cv.height*.2,0,cv.width*.5,cv.height*.5,cv.width*.8);
  grad.addColorStop(0,'#f0f7ff'); grad.addColorStop(.5,'#f8fafc'); grad.addColorStop(1,'#f1f5f9');
  ctx.fillStyle=grad; ctx.fillRect(0,0,cv.width,cv.height);

  ctx.save(); ctx.translate(tx,ty); ctx.scale(sc,sc);

  // Subtle dot grid
  var gs=40,ox=-tx/sc,oy=-ty/sc,W=cv.width/sc,H=cv.height/sc;
  ctx.fillStyle='#94a3b820';
  for(var gx=Math.floor(ox/gs)*gs;gx<ox+W;gx+=gs)
    for(var gy=Math.floor(oy/gs)*gs;gy<oy+H;gy+=gs){{
      ctx.beginPath(); ctx.arc(gx,gy,1,0,Math.PI*2); ctx.fill();
    }}

  // Edges first (behind boxes)
  EDGES.forEach(function(e){{
    var a=NM[e.from],b=NM[e.to];
    if(!a||!b||!isVis(a)||!isVis(b)) return;
    drawEdge(e,a,b);
  }});

  // Boxes on top
  NODES.forEach(function(n){{if(isVis(n)) drawNode(n);}});

  ctx.restore();
  document.getElementById('sb-v').textContent=NODES.filter(isVis).length;
}}

function drawEdge(e,a,b){{
  var hl=selId&&(e.from===selId||e.to===selId);
  var x1=a.x+a.w, y1=a.y+a.h/2;
  var x2=b.x,     y2=b.y+b.h/2;
  var cp=Math.abs(x2-x1)*0.45;

  if(hl){{
    // Glowing highlighted edge
    var col=e.to===selId?'#059669':'#d97706';
    // Glow layer
    ctx.beginPath(); ctx.moveTo(x1,y1);
    ctx.bezierCurveTo(x1+cp,y1,x2-cp,y2,x2,y2);
    ctx.strokeStyle=col; ctx.lineWidth=6/sc; ctx.globalAlpha=0.12; ctx.stroke();
    ctx.lineWidth=3/sc; ctx.globalAlpha=0.3; ctx.stroke();
    ctx.lineWidth=1.5/sc; ctx.globalAlpha=1; ctx.stroke();

    // Arrow tip
    var ang=Math.atan2(y2-(y1+(y2-y1)*.7),x2-(x1+(x2-x1)*.7)),as=8/sc;
    ctx.beginPath(); ctx.moveTo(x2,y2);
    ctx.lineTo(x2-as*Math.cos(ang-.4),y2-as*Math.sin(ang-.4));
    ctx.lineTo(x2-as*Math.cos(ang+.4),y2-as*Math.sin(ang+.4));
    ctx.closePath(); ctx.fillStyle=col; ctx.globalAlpha=1; ctx.fill();
  }} else {{
    // Normal edge — thin, luminous, subtle
    ctx.beginPath(); ctx.moveTo(x1,y1);
    ctx.bezierCurveTo(x1+cp,y1,x2-cp,y2,x2,y2);
    // Gradient along the edge direction
    var eg=ctx.createLinearGradient(x1,y1,x2,y2);
    eg.addColorStop(0,STROKE[a.ntype]+'44');
    eg.addColorStop(1,STROKE[b.ntype]+'44');
    ctx.strokeStyle=eg;
    ctx.lineWidth=1.2/sc; ctx.globalAlpha=0.5; ctx.stroke();
    ctx.globalAlpha=1;
  }}
}}

function rrect(x,y,w,h,r){{
  var tl,tr,bl,br;
  if(typeof r==='number'){{tl=tr=bl=br=r;}}
  else{{tl=r.tl||0;tr=r.tr||0;bl=r.bl||0;br=r.br||0;}}
  ctx.beginPath();
  ctx.moveTo(x+tl,y);ctx.lineTo(x+w-tr,y);ctx.arcTo(x+w,y,x+w,y+tr,tr);
  ctx.lineTo(x+w,y+h-br);ctx.arcTo(x+w,y+h,x+w-br,y+h,br);
  ctx.lineTo(x+bl,y+h);ctx.arcTo(x,y+h,x,y+h-bl,bl);
  ctx.lineTo(x,y+tl);ctx.arcTo(x,y,x+tl,y,tl);
  ctx.closePath();
}}

function drawNode(n){{
  var s=STROKE[n.ntype], f=FILL[n.ntype], h=HDR_C[n.ntype], t=TEXT_C[n.ntype], sub=SUB_C[n.ntype];
  var isFinal=(n.id===FINAL), isHl=(selId===n.id);
  var conn=hlNodes.has(n.id);
  var dim=selId&&!isHl&&!conn;
  var searched=srchQ&&n.id.toLowerCase().indexOf(srchQ)>=0;

  ctx.globalAlpha = dim ? 0.25 : 1;

  // Outer glow for highlighted / final
  if(isHl||isFinal){{
    ctx.shadowColor=s; ctx.shadowBlur=isHl?24:12;
    // Extra glow ring
    ctx.strokeStyle=s+'33'; ctx.lineWidth=6/sc;
    rrect(n.x-3,n.y-3,n.w+6,n.h+6,10); ctx.stroke();
    ctx.shadowBlur=0;
  }}

  // Box fill with subtle gradient
  // Drop shadow
  ctx.shadowColor='#00000018'; ctx.shadowBlur=12; ctx.shadowOffsetX=0; ctx.shadowOffsetY=3;
  var bg=ctx.createLinearGradient(n.x,n.y,n.x,n.y+n.h);
  bg.addColorStop(0,f); bg.addColorStop(1,f);
  rrect(n.x,n.y,n.w,n.h,8); ctx.fillStyle=bg; ctx.fill();
  ctx.shadowColor='transparent'; ctx.shadowBlur=0; ctx.shadowOffsetY=0;

  // Header gradient
  var hg=ctx.createLinearGradient(n.x,n.y,n.x+n.w,n.y+HDR_H);
  hg.addColorStop(0,h); hg.addColorStop(1,f);
  rrect(n.x,n.y,n.w,HDR_H,{{tl:8,tr:8,bl:0,br:0}}); ctx.fillStyle=hg; ctx.fill();

  // Border with glow
  ctx.strokeStyle = isHl ? s : (searched ? '#f59e0b' : s+'88');
  ctx.lineWidth = isHl||isFinal ? 1.5/sc : 1/sc;
  rrect(n.x,n.y,n.w,n.h,8); ctx.stroke();

  // Top accent bar — glowing
  ctx.shadowColor=s+'44'; ctx.shadowBlur=4;
  ctx.fillStyle=s;
  rrect(n.x,n.y,n.w,3,{{tl:8,tr:8,bl:0,br:0}}); ctx.fill();
  ctx.shadowBlur=0; ctx.shadowColor='transparent';

  // Table name
  ctx.fillStyle=dim?sub:t;
  ctx.font=(isFinal?'600 ':'')+' 10.5px "JetBrains Mono",monospace';
  ctx.textAlign='left'; ctx.textBaseline='middle';
  var lbl=n.id, mw=n.w-50;
  while(ctx.measureText(lbl).width>mw&&lbl.length>4) lbl=lbl.slice(0,-4)+'…';
  ctx.fillText(lbl,n.x+11,n.y+HDR_H/2+1);

  // Level pill
  ctx.font='7px "Sora",sans-serif'; ctx.textAlign='right';
  ctx.fillStyle=dim?'#1a2030':s+'aa';
  ctx.fillText(isFinal?'● OUT':'L'+n.level,n.x+n.w-7,n.y+HDR_H/2+1);

  // Column rows
  if(n.cols&&n.cols.length>0&&!dim){{
    ctx.font='10.5px "JetBrains Mono",monospace'; ctx.textAlign='left';
    n.cols.slice(0,PREV_N).forEach(function(col,i){{
      var cy=n.y+HDR_H+8+i*ROW_H+ROW_H/2-2;
      // Dot
      ctx.fillStyle=s+'cc';
      ctx.beginPath();ctx.arc(n.x+13,cy,2.5,0,Math.PI*2);ctx.fill();
      // Column name
      ctx.fillStyle=t;
      var clbl=col;
      while(ctx.measureText(clbl).width>n.w-32&&clbl.length>3) clbl=clbl.slice(0,-3)+'…';
      ctx.fillText(clbl,n.x+22,cy+1);
    }});
    if(n.cols.length>PREV_N){{
      ctx.fillStyle=sub;
      ctx.font='8px "Sora",sans-serif';
      ctx.fillText('+'+( n.cols.length-PREV_N)+' more',n.x+11,n.y+HDR_H+8+PREV_N*ROW_H+4);
    }}
  }}
  ctx.globalAlpha=1; ctx.shadowBlur=0;
}}

// ── Mouse ──────────────────────────────────────────────────────────────────────
cv.addEventListener('mousedown',function(e){{
  mdx=e.clientX; mdy=e.clientY; nodeMoved=false;
  var p=s2w(e.offsetX,e.offsetY), hit=hitNode(p.x,p.y);
  if(hit){{dragN=hit;dox=p.x-hit.x;doy=p.y-hit.y;}}
  else panning={{mx:e.clientX,my:e.clientY,tx:tx,ty:ty}};
  cv.style.cursor='grabbing';
}});
cv.addEventListener('mousemove',function(e){{
  if(dragN){{
    if(Math.abs(e.clientX-mdx)>3||Math.abs(e.clientY-mdy)>3) nodeMoved=true;
    var p=s2w(e.offsetX,e.offsetY);
    dragN.x=p.x-dox; dragN.y=p.y-doy; render();
  }} else if(panning){{
    tx=panning.tx+(e.clientX-panning.mx); ty=panning.ty+(e.clientY-panning.my); render();
  }}
}});
cv.addEventListener('mouseup',function(e){{
  var dn=dragN,wp=panning;
  var pdx=wp?Math.abs(e.clientX-panning.mx):0, pdy=wp?Math.abs(e.clientY-panning.my):0;
  var click=dn&&!nodeMoved, pclick=wp&&pdx<4&&pdy<4;
  dragN=null; panning=null; nodeMoved=false; cv.style.cursor='grab';
  if(click){{ selectNode(dn.id); }}
  else if(pclick){{ clearSel(); closePanel(); }}
}});
cv.addEventListener('mouseleave',function(){{dragN=null;panning=null;cv.style.cursor='grab';}});
cv.addEventListener('wheel',function(e){{
  e.preventDefault();
  var d=e.deltaY>0?.88:1.14;
  tx=e.offsetX-d*(e.offsetX-tx); ty=e.offsetY-d*(e.offsetY-ty);
  sc=Math.max(.04,Math.min(6,sc*d));
  document.getElementById('sb-z').textContent=Math.round(sc*100)+'%';
  render();
}},{{passive:false}});

// ── Selection ──────────────────────────────────────────────────────────────────
function selectNode(id){{
  selId=id; hlNodes=new Set([id]);
  (EF[id]||[]).forEach(function(e){{hlNodes.add(e.to);}});
  (ET[id]||[]).forEach(function(e){{hlNodes.add(e.from);}});
  render(); openPanel(id);
}}
function clearSel(){{selId=null;hlNodes=new Set();render();}}

// ── Panel ──────────────────────────────────────────────────────────────────────
function openPanel(id){{
  var nd=NM[id]; if(!nd) return;
  document.getElementById('panel').classList.add('open');
  resize();
  document.getElementById('ph-title').textContent=id;
  var inE=ET[id]||[], outE=EF[id]||[];
  document.getElementById('ph-meta').innerHTML=
    '<span style="color:'+STROKE[nd.ntype]+'88">'+TL[nd.ntype]+'</span>'+
    ' &nbsp;·&nbsp; Level '+nd.level+
    ' &nbsp;·&nbsp; '+nd.cols.length+' cols'+
    ' &nbsp;·&nbsp; '+inE.length+' in · '+outE.length+' out';
  buildCols(nd.cols);
  buildRelSec('src',inE,true);
  buildRelSec('dst',outE,false);
}}

function buildCols(cols){{
  document.getElementById('cnt-cols').textContent=cols.length;
  var body=document.getElementById('body-cols'); body.innerHTML='';
  if(!cols.length){{
    var em=document.createElement('div');
    em.style.cssText='padding:10px 16px;color:#334155;font-size:.68rem;font-style:italic;';
    em.textContent='No columns extracted (external/base table)';
    body.appendChild(em); return;
  }}
  cols.forEach(function(col,i){{
    var row=document.createElement('div'); row.className='col-row';
    var idx=document.createElement('span'); idx.className='col-idx'; idx.textContent=(i+1)+'.';
    var nm=document.createElement('span');  nm.className='col-name';  nm.textContent=col;
    row.appendChild(idx); row.appendChild(nm); body.appendChild(row);
  }});
}}

function buildRelSec(sec,edges,isSource){{
  document.getElementById('cnt-'+sec).textContent=edges.length;
  var body=document.getElementById('body-'+sec); body.innerHTML='';
  if(!edges.length){{
    var em=document.createElement('div');
    em.style.cssText='padding:10px 16px;color:#334155;font-size:.68rem;font-style:italic;';
    em.textContent=isSource?'No source tables (base input)':'No downstream tables (leaf output)';
    body.appendChild(em); return;
  }}
  edges.forEach(function(e){{
    var oid=isSource?e.from:e.to, nd2=NM[oid];
    var nt=nd2?nd2.ntype:'ext', col=STROKE[nt]||'#64748b';
    var fa=e.src_alias||'', ta=e.tgt_alias||'', da=isSource?fa:ta;
    var row=document.createElement('div'); row.className='trow';
    row.addEventListener('click',function(){{
      selectNode(oid); focusNode(oid);
    }});
    var top=document.createElement('div'); top.className='trow-top';
    var dot=document.createElement('div'); dot.className='tdot';
    dot.style.cssText='background:'+col+';box-shadow:0 0 6px '+col+'66;';
    var nm=document.createElement('div'); nm.className='tname'; nm.textContent=oid;
    top.appendChild(dot); top.appendChild(nm);
    if(da){{var ab=document.createElement('span');ab.className='talias';
      ab.textContent=da;top.appendChild(ab);}}
    row.appendChild(top);
    var jks=e.jk||[];
    if(jks.length){{
      var tbl=document.createElement('table'); tbl.className='jktbl';
      var thead=tbl.createTHead(), hr=thead.insertRow();
      ['Source','','Dest'].forEach(function(t){{var el=document.createElement('th');el.textContent=t;hr.appendChild(el);}});
      var tb2=tbl.createTBody();
      jks.forEach(function(jk){{
        var tr=tb2.insertRow();
        if(jk.left&&jk.right){{
          tr.insertCell().className='jk-s'; tr.cells[0].textContent=jk.left;
          tr.insertCell().className='jk-e'; tr.cells[1].textContent='=';
          tr.insertCell().className='jk-d'; tr.cells[2].textContent=jk.right;
        }} else if(jk.col){{
          var td=tr.insertCell(); td.className='jk-s'; td.textContent=jk.col; td.colSpan=3;
        }}
      }});
      row.appendChild(tbl);
    }} else {{
      var nj=document.createElement('div'); nj.className='no-jk';
      nj.textContent='no explicit join key'; row.appendChild(nj);
    }}
    body.appendChild(row);
  }});
}}

function closePanel(){{document.getElementById('panel').classList.remove('open');clearSel();resize();}}
function toggleSec(s){{document.getElementById('hdr-'+s).classList.toggle('collapsed');}}
function focusNode(id){{
  var n=NM[id]; if(!n) return;
  tx=cv.width/2-(n.x+n.w/2)*sc; ty=cv.height/2-(n.y+n.h/2)*sc; render();
}}

// ── Camera ─────────────────────────────────────────────────────────────────────
function resetView(){{clearSel();closePanel();fitAll();}}
function fitAll(){{
  hiddenT=new Set();
  ['ikg','tmp','ext'].forEach(function(t){{document.getElementById('btn-'+t).classList.add('on');}});
  fitVisible();
}}
function fitVisible(){{
  var vis=NODES.filter(isVis); if(!vis.length) return;
  var x1=Math.min.apply(null,vis.map(function(n){{return n.x;}}));
  var x2=Math.max.apply(null,vis.map(function(n){{return n.x+n.w;}}));
  var y1=Math.min.apply(null,vis.map(function(n){{return n.y;}}));
  var y2=Math.max.apply(null,vis.map(function(n){{return n.y+n.h;}}));
  var pad=60;
  sc=Math.min((cv.width-pad*2)/(x2-x1||1),(cv.height-pad*2)/(y2-y1||1),1.4);
  tx=(cv.width-(x2-x1)*sc)/2-x1*sc; ty=(cv.height-(y2-y1)*sc)/2-y1*sc;
  document.getElementById('sb-z').textContent=Math.round(sc*100)+'%';
  render();
}}
function doZoom(f){{
  var cx=cv.width/2,cy=cv.height/2;
  tx=cx-f*(cx-tx); ty=cy-f*(cy-ty);
  sc=Math.max(.04,Math.min(6,sc*f));
  document.getElementById('sb-z').textContent=Math.round(sc*100)+'%'; render();
}}
function toggleType(t){{
  var btn=document.getElementById('btn-'+t);
  if(hiddenT.has(t)){{hiddenT.delete(t);btn.classList.add('on');}}
  else{{hiddenT.add(t);btn.classList.remove('on');}}
  render();
}}
function onSearch(){{srchQ=document.getElementById('srch').value.toLowerCase().trim();render();}}
function clearSearch(){{srchQ='';document.getElementById('srch').value='';render();}}

function resize(){{
  var w=document.getElementById('cv-wrap');
  cv.width=w.clientWidth; cv.height=w.clientHeight; render();
}}
window.addEventListener('resize',resize);
window.addEventListener('load',function(){{resize();fitAll();}});
</script>
</body>
</html>
"""
    return html



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
