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
    for m in re.finditer(r'\bWITH\b(.*?)(?=\bSELECT\b)', clean_sql,
                         re.IGNORECASE | re.DOTALL):
        for nm in re.finditer(r'\b(\w+)\s+AS\s*\(', m.group(1), re.IGNORECASE):
            cte_names.add(nm.group(1).lower())

    skip_kw = {'select','where','on','set','lateral','only','rows','unnest','values',
               'null','true','false','current','row','all','distinct'}
    all_ref = set()
    for m in re.finditer(r'\b(?:FROM|JOIN)\s+(?:\{\{[^}]+\}\}\.)?(\w+)',
                         clean_sql, re.IGNORECASE):
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
        existing = set(tuple(j) if isinstance(j, list) else j
                       for j in merged_edges[key]['jk'])
        for jc in r.get('join_cols', []):
            if isinstance(jc, (list, tuple)) and len(jc) == 2:
                item = [str(jc[0]), str(jc[1])]; k2 = tuple(item)
            else:
                item = str(jc); k2 = item
            if k2 not in existing:
                existing.add(k2)
                merged_edges[key]['jk'].append(
                    {'left': item[0], 'right': item[1]}
                    if isinstance(item, list) else {'col': item})
    edges_list = list(merged_edges.values())

    # ── Topological levels ────────────────────────────────────────────────────
    from collections import defaultdict as _dd, deque as _dq
    in_em  = _dd(set); out_em = _dd(set)
    for e in edges_list:
        in_em[e['to']].add(e['from'])
        out_em[e['from']].add(e['to'])

    in_deg = {t: len(in_em[t]) for t in all_tables}
    queue  = _dq([t for t in all_tables if in_deg[t] == 0])
    level  = {t: 0 for t in queue}
    while queue:
        t = queue.popleft()
        for nxt in out_em[t]:
            if nxt in all_tables:
                in_deg[nxt]  = max(0, in_deg.get(nxt, 0) - 1)
                level[nxt]   = max(level.get(nxt, 0), level[t] + 1)
                if in_deg[nxt] == 0: queue.append(nxt)
    for t in all_tables:
        if t not in level: level[t] = 0

    def node_type(n):
        if n == FINAL:       return 'final'
        if n in created_set: return 'ikg'
        if n in temp_set:    return 'tmp'
        return 'ext'

    # ── Layout ────────────────────────────────────────────────────────────────
    max_level = max(level.values()) if level else 0
    by_level  = _dd(list)
    for t in all_tables:
        by_level[level[t]].append(t)
    for lv in by_level:
        by_level[lv].sort(key=lambda t: -(len(in_em[t]) + len(out_em[t])))

    BOX_W   = 220
    HDR_H   = 28
    ROW_H   = 16
    PREV_N  = 4
    FOOT_H  = 10
    COL_GAP = 80
    ROW_GAP = 20
    PAD_X   = 60
    PAD_Y   = 50

    def box_h(cols):
        rows = min(len(cols), PREV_N)
        return HDR_H + (rows * ROW_H + 6 if rows else 0) + FOOT_H

    sorted_lvs = sorted(by_level.keys(), reverse=True)
    xCursor = PAD_X
    colX = {}
    for lv in sorted_lvs:
        colX[lv] = xCursor
        xCursor += BOX_W + COL_GAP

    nodes_data = []
    for lv in sorted_lvs:
        yCursor = PAD_Y
        for t in by_level[lv]:
            cols = col_map.get(t, [])
            bh   = box_h(cols)
            nodes_data.append({
                'id':    t,
                'x':     colX[lv],
                'y':     yCursor,
                'w':     BOX_W,
                'h':     bh,
                'ntype': node_type(t),
                'level': level.get(t, 0),
                'cols':  cols,
            })
            yCursor += bh + ROW_GAP

    ikg_count = len([n for n in nodes_data if n['ntype'] in ('ikg','final')])
    tmp_count = len([n for n in nodes_data if n['ntype'] == 'tmp'])
    ext_count = len([n for n in nodes_data if n['ntype'] == 'ext'])
    rel_count = len(edges_list)

    import json as _json
    nodes_json = _json.dumps(nodes_data, separators=(',',':'))
    edges_json = _json.dumps(edges_list, separators=(',',':'))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{FINAL} — ER Diagram (Sigma)</title>
<script src="https://cdn.jsdelivr.net/npm/graphology@0.25.4/dist/graphology.umd.min.js" crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/npm/sigma@2.4.0/build/sigma.min.js" crossorigin="anonymous"></script>
<style>
:root{{--bg:#090e18;--panel-bg:#0b1422;--tb-bg:#0d1525;--border:#162030;
      --text:#e2e8f0;--muted:#4a6070;
      --final:#2ecc71;--ikg:#2a78b8;--tmp:#7840c8;--ext:#c06020;}}
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0;}}
body{{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;
      overflow:hidden;height:100vh;display:flex;flex-direction:column;}}
#hdr{{background:linear-gradient(90deg,#1a3356,#0d2040);padding:10px 18px;
      display:flex;align-items:center;gap:12px;border-bottom:1px solid var(--border);flex-shrink:0;}}
#hdr h1{{font-size:.92rem;font-weight:700;color:#5aa8e8;}}
.sub{{font-size:.67rem;color:#2d4060;margin-top:1px;}}
.sp{{padding:2px 10px;border-radius:20px;font-size:.65rem;font-weight:600;}}
.sp-b{{background:#0f2045;color:#5aa8e8;border:1px solid #1e4080;}}
.sp-p{{background:#1e1040;color:#a070e8;border:1px solid #3a2070;}}
.sp-o{{background:#2a1500;color:#e09050;border:1px solid #603010;}}
.sp-g{{background:#0f3020;color:#4ec87a;border:1px solid #2a6040;}}
#tb{{background:var(--tb-bg);border-bottom:1px solid var(--border);
     padding:5px 12px;display:flex;align-items:center;gap:6px;flex-shrink:0;flex-wrap:wrap;}}
.btn{{background:#111d30;color:#7a90a8;border:1px solid #1e2d40;border-radius:4px;
      padding:4px 11px;font-size:.71rem;cursor:pointer;transition:all .12s;}}
.btn:hover{{background:#1a2d44;color:#c8d8e8;}}.btn.on{{background:#122040;color:#5aa8e8;border-color:#1e4070;}}
.sep{{width:1px;height:18px;background:#1a2535;margin:0 2px;}}
#srch{{background:#111d30;color:var(--text);border:1px solid #1e2d40;border-radius:4px;
       padding:4px 10px;font-size:.71rem;width:175px;outline:none;}}
#srch:focus{{border-color:#2a5090;}}#srch::placeholder{{color:#2a3848;}}
#leg{{display:flex;gap:10px;margin-left:auto;align-items:center;}}
.li{{display:flex;align-items:center;gap:4px;font-size:.65rem;color:var(--muted);}}
.lb{{width:12px;height:8px;border-radius:2px;}}
#wrap{{flex:1;display:flex;overflow:hidden;}}
#cv-wrap{{flex:1;min-width:0;position:relative;overflow:hidden;cursor:grab;}}
#cv{{display:block;position:absolute;top:0;left:0;cursor:grab;}}
#panel{{width:420px;min-width:420px;background:var(--panel-bg);border-left:1px solid var(--border);
        display:none;flex-direction:column;overflow:hidden;flex-shrink:0;}}
#panel.open{{display:flex;}}
#ph{{background:#091020;padding:11px 13px 9px;border-bottom:1px solid var(--border);
     flex-shrink:0;position:relative;}}
#ph-title{{font-size:.87rem;font-weight:700;color:#5aa8e8;word-break:break-all;margin-bottom:3px;padding-right:24px;}}
#ph-meta{{font-size:.67rem;color:var(--muted);}}
#ph-x{{position:absolute;right:12px;top:12px;background:none;border:none;color:#2a3848;
       cursor:pointer;font-size:1rem;transition:color .1s;}}
#ph-x:hover{{color:var(--text);}}
#pb{{flex:1;overflow-y:auto;}}
#pb::-webkit-scrollbar{{width:4px;}}
#pb::-webkit-scrollbar-track{{background:#080f1c;}}
#pb::-webkit-scrollbar-thumb{{background:#1a2840;border-radius:3px;}}
.sec{{border-bottom:1px solid #0f1e30;}}
.sechdr{{padding:8px 13px;font-size:.66rem;font-weight:700;text-transform:uppercase;
         letter-spacing:.8px;display:flex;align-items:center;gap:7px;
         cursor:pointer;user-select:none;transition:background .1s;}}
.sechdr:hover{{background:#0d1a2a;}}
.sechdr.c-cols{{color:#3a6888;}}.sechdr.c-src{{color:#2a6840;}}.sechdr.c-dst{{color:#6a4010;}}
.seccnt{{margin-left:auto;background:#0f1e2e;color:#3a6888;font-size:.61rem;padding:1px 7px;border-radius:10px;font-weight:700;}}
.arr{{font-size:.62rem;color:#1e2d3e;transition:transform .15s;}}
.sechdr.collapsed .arr{{transform:rotate(-90deg);}}
.secbody{{display:block;}}.sechdr.collapsed+.secbody{{display:none;}}
.col-row{{padding:4px 13px 4px 22px;display:flex;align-items:center;gap:8px;border-bottom:1px solid #0a1520;font-size:.73rem;}}
.col-row:last-child{{border-bottom:none;}}
.col-idx{{color:#1e3040;font-size:.6rem;min-width:18px;font-family:monospace;}}
.col-name{{color:#90b8d8;font-family:'Consolas','Courier New',monospace;}}
.trow{{padding:7px 11px 7px 15px;border-bottom:1px solid #0a1828;cursor:pointer;transition:background .1s;}}
.trow:hover{{background:#0d1c2e;}}.trow-top{{display:flex;align-items:flex-start;gap:7px;margin-bottom:4px;}}
.tdot{{width:7px;height:7px;border-radius:2px;flex-shrink:0;margin-top:4px;}}
.tname{{font-size:.75rem;font-weight:600;color:#b0c8d8;flex:1;word-break:break-all;}}
.talias{{font-size:.62rem;color:#1e4060;background:#081522;border:1px solid #102030;border-radius:3px;
         padding:1px 6px;font-family:'Consolas','Courier New',monospace;white-space:nowrap;}}
.jktbl{{width:calc(100% - 14px);border-collapse:collapse;margin:2px 0 2px 14px;}}
.jktbl th{{font-size:.59rem;color:#1e4050;font-weight:700;text-transform:uppercase;letter-spacing:.4px;padding:2px 6px 2px 0;border-bottom:1px solid #0f1e2e;}}
.jktbl td{{font-size:.66rem;font-family:'Consolas','Courier New',monospace;padding:2px 6px 2px 0;vertical-align:top;border-bottom:1px solid #081520;}}
.jktbl tr:last-child td{{border-bottom:none;}}
.jk-s{{color:#4ec87a;}}.jk-e{{color:#1e2d3a;padding:0 2px;}}.jk-d{{color:#e8a040;}}
.no-jk{{font-size:.62rem;color:#162028;padding:2px 0 2px 14px;font-style:italic;}}
#sb{{background:var(--tb-bg);border-top:1px solid var(--border);padding:3px 13px;
     font-size:.65rem;color:#1e3040;display:flex;gap:16px;flex-shrink:0;}}
#sb b{{color:#2a4050;}}
</style>
</head>
<body>
<div id="hdr">
  <div>
    <h1>&#128202; {FINAL} &mdash; ER Diagram</h1>
    <div class="sub">Powered by Sigma.js &bull; Drag boxes to move &bull; Click any box for details &bull; Scroll to zoom</div>
  </div>
  <span class="sp sp-b">{ikg_count} IKG tables</span>
  <span class="sp sp-p">{tmp_count} Temp tables</span>
  <span class="sp sp-o">{ext_count} External tables</span>
  <span class="sp sp-g">{rel_count} Relationships</span>
</div>
<div id="tb">
  <button class="btn" onclick="resetView()">&#8635; Reset</button>
  <button class="btn" onclick="doZoom(1.25)">+ Zoom</button>
  <button class="btn" onclick="doZoom(0.8)">&#8722; Zoom</button>
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
  <div id="cv-wrap" style="flex:1;position:relative;overflow:hidden;min-width:0;">
    <canvas id="cv" style="display:block;position:absolute;top:0;left:0;"></canvas>
  </div>
  <div id="panel">
    <div id="ph">
      <button id="ph-x" onclick="closePanel()">&#10005;</button>
      <div id="ph-title"></div><div id="ph-meta"></div>
    </div>
    <div id="pb">
      <div class="sec">
        <div class="sechdr c-cols" id="hdr-cols" onclick="toggleSec('cols')">
          &#128196;&nbsp;Columns<span class="seccnt" id="cnt-cols">0</span><span class="arr">&#9660;</span>
        </div><div class="secbody" id="body-cols"></div>
      </div>
      <div class="sec">
        <div class="sechdr c-src" id="hdr-src" onclick="toggleSec('src')">
          &#8593;&nbsp;Source Tables <em style="color:#1a4828;font-weight:400;font-size:.6rem;text-transform:none">(inputs)</em>
          <span class="seccnt" id="cnt-src">0</span><span class="arr">&#9660;</span>
        </div><div class="secbody" id="body-src"></div>
      </div>
      <div class="sec">
        <div class="sechdr c-dst collapsed" id="hdr-dst" onclick="toggleSec('dst')">
          &#8595;&nbsp;Destination Tables <em style="color:#482810;font-weight:400;font-size:.6rem;text-transform:none">(outputs)</em>
          <span class="seccnt" id="cnt-dst">0</span><span class="arr">&#9660;</span>
        </div><div class="secbody" id="body-dst"></div>
      </div>
    </div>
  </div>
</div>
<div id="sb">
  <span>Zoom: <b id="sb-z">100%</b></span>
  <span>Visible: <b id="sb-v">0</b></span>
  <span>Drag box to move &bull; Drag background to pan &bull; Scroll to zoom &bull; Click box for details</span>
</div>

<script>
// ── Data ──────────────────────────────────────────────────────────────────────
const NODES = {nodes_json};
const EDGES = {edges_json};
const FINAL = "{FINAL}";

const TC  = {{final:'#2ecc71',ikg:'#2a78b8',tmp:'#7840c8',ext:'#c06020'}};
const TL  = {{final:'Final Output',ikg:'IKG Table',tmp:'Temp Table',ext:'External/EDW'}};
const ST  = {{
  final:{{fill:'#081e10',stroke:'#2ecc71',hdr:'#0a3018',text:'#7ae8a0',sub:'#2a6040'}},
  ikg:  {{fill:'#071525',stroke:'#2a78b8',hdr:'#0a1e38',text:'#6ab0d8',sub:'#1a4060'}},
  tmp:  {{fill:'#12082a',stroke:'#7840c8',hdr:'#180d38',text:'#a878e8',sub:'#3a1870'}},
  ext:  {{fill:'#1a0a00',stroke:'#c06020',hdr:'#240e00',text:'#e09050',sub:'#602810'}},
}};
const HDR_H=28, ROW_H=16, PREV_N=4;

// ── Indices ───────────────────────────────────────────────────────────────────
const NM={{}};
NODES.forEach(function(n){{NM[n.id]=n;}});
const edgesFrom={{}}, edgesTo={{}};
EDGES.forEach(function(e){{
  if(!edgesFrom[e.from]) edgesFrom[e.from]=[];
  if(!edgesTo[e.to])     edgesTo[e.to]=[];
  edgesFrom[e.from].push(e);
  edgesTo[e.to].push(e);
}});

// ── Canvas & camera state ─────────────────────────────────────────────────────
const cv  = document.getElementById('cv');
const ctx = cv.getContext('2d');
let tx=0, ty=0, sc=1;

// ── Interaction state ─────────────────────────────────────────────────────────
let selectedId   = null;
let hlNodes      = new Set();
let hiddenTypes  = new Set();
let srchQ        = '';
let panning      = null;
let draggingNode = null;
let dragOffX=0, dragOffY=0;
let mouseDownX=0, mouseDownY=0;
let nodeDragMoved = false;

// ── Coordinate helpers ────────────────────────────────────────────────────────
function s2w(sx,sy){{return{{x:(sx-tx)/sc, y:(sy-ty)/sc}};}}
function w2s(wx,wy){{return{{x:wx*sc+tx,   y:wy*sc+ty}};}}

// ── Visibility ────────────────────────────────────────────────────────────────
function isVis(n){{
  if(n.ntype!=='final' && hiddenTypes.has(n.ntype)) return false;
  if(srchQ && n.id.toLowerCase().indexOf(srchQ)<0) return false;
  return true;
}}

// ── Hit testing ───────────────────────────────────────────────────────────────
function hitNode(wx,wy){{
  for(var i=NODES.length-1;i>=0;i--){{
    var n=NODES[i];
    if(!isVis(n)) continue;
    if(wx>=n.x && wx<=n.x+n.w && wy>=n.y && wy<=n.y+n.h) return n;
  }}
  return null;
}}

// ── Render ────────────────────────────────────────────────────────────────────
function render(){{
  ctx.clearRect(0,0,cv.width,cv.height);
  ctx.save(); ctx.translate(tx,ty); ctx.scale(sc,sc);

  // Grid
  var gs=40, ox=-tx/sc, oy=-ty/sc, W=cv.width/sc, H=cv.height/sc;
  ctx.strokeStyle='#0c1525'; ctx.lineWidth=1/sc;
  for(var x=Math.floor(ox/gs)*gs;x<ox+W;x+=gs){{
    ctx.beginPath();ctx.moveTo(x,oy);ctx.lineTo(x,oy+H);ctx.stroke();}}
  for(var y=Math.floor(oy/gs)*gs;y<oy+H;y+=gs){{
    ctx.beginPath();ctx.moveTo(ox,y);ctx.lineTo(ox+W,y);ctx.stroke();}}

  // Edges (drawn first, under boxes)
  EDGES.forEach(function(e){{
    var a=NM[e.from], b=NM[e.to];
    if(!a||!b||!isVis(a)||!isVis(b)) return;
    drawEdge(e,a,b);
  }});

  // Boxes (on top)
  NODES.forEach(function(n){{if(isVis(n)) drawNode(n);}});

  ctx.restore();

  // Status
  document.getElementById('sb-v').textContent = NODES.filter(isVis).length;
}}

function drawEdge(e,a,b){{
  var hl = selectedId && (e.from===selectedId || e.to===selectedId);
  // Connect right edge of source (a) to left edge of target (b)
  // source is the table being read FROM; target is the table being written TO
  // In our layout: higher level (final) is LEFT, lower level (sources) is RIGHT
  // Edge goes: right of source box → left of target box
  var x1=a.x+a.w, y1=a.y+a.h/2;
  var x2=b.x,     y2=b.y+b.h/2;
  var cp=Math.abs(x2-x1)*0.45;

  ctx.beginPath();
  ctx.moveTo(x1,y1);
  ctx.bezierCurveTo(x1+cp,y1, x2-cp,y2, x2,y2);
  ctx.strokeStyle = hl ? (e.to===selectedId?'#3adc80':'#e89030') : '#182840';
  ctx.lineWidth   = hl ? 2/sc : 1/sc;
  ctx.globalAlpha = hl ? 1 : 0.5;
  ctx.stroke();
  ctx.globalAlpha = 1;

  // Arrow tip at x2,y2 pointing left
  if(hl){{
    var col = e.to===selectedId ? '#3adc80' : '#e89030';
    var as  = 7/sc;
    var ang = Math.atan2(y2-(y1+(y2-y1)*0.8), x2-(x1+(x2-x1)*0.8));
    ctx.beginPath();
    ctx.moveTo(x2,y2);
    ctx.lineTo(x2-as*Math.cos(ang-0.4), y2-as*Math.sin(ang-0.4));
    ctx.lineTo(x2-as*Math.cos(ang+0.4), y2-as*Math.sin(ang+0.4));
    ctx.closePath();
    ctx.fillStyle=col; ctx.fill();
  }}
}}

function drawNode(n){{
  var s=ST[n.ntype], isFinal=(n.id===FINAL), isHl=(selectedId===n.id);
  var conn=hlNodes.has(n.id);
  var dim=selectedId && !isHl && !conn;
  var searched=srchQ && n.id.toLowerCase().indexOf(srchQ)>=0;

  if(isHl||isFinal){{ctx.shadowColor=s.stroke;ctx.shadowBlur=isHl?16:8;}}
  ctx.globalAlpha=dim?0.2:1;

  rrect(n.x,n.y,n.w,n.h,6); ctx.fillStyle=s.fill; ctx.fill();
  rrect(n.x,n.y,n.w,HDR_H,{{tl:6,tr:6,bl:0,br:0}}); ctx.fillStyle=s.hdr; ctx.fill();
  ctx.strokeStyle=isHl?'#fff':(searched?'#e8d040':s.stroke);
  ctx.lineWidth=isHl||isFinal?2/sc:1/sc;
  rrect(n.x,n.y,n.w,n.h,6); ctx.stroke();
  rrect(n.x,n.y,n.w,3,{{tl:6,tr:6,bl:0,br:0}}); ctx.fillStyle=s.stroke; ctx.fill();
  ctx.shadowBlur=0;

  // Table name
  ctx.fillStyle=dim?'#1e2d3e':s.text;
  ctx.font=(isFinal?'bold ':'')+' 10.5px Segoe UI,system-ui,sans-serif';
  ctx.textAlign='left'; ctx.textBaseline='middle';
  var lbl=n.id, mw=n.w-52;
  while(ctx.measureText(lbl).width>mw&&lbl.length>4) lbl=lbl.slice(0,-4)+'...';
  ctx.fillText(lbl, n.x+10, n.y+HDR_H/2+1);

  // Level badge
  ctx.font='8px Segoe UI,system-ui,sans-serif';
  ctx.fillStyle=dim?'#1a2838':s.sub; ctx.textAlign='right';
  ctx.fillText(isFinal?'OUT':'L'+n.level, n.x+n.w-5, n.y+HDR_H/2+1);

  // Columns preview
  if(n.cols&&n.cols.length>0&&!dim){{
    var cols=n.cols.slice(0,PREV_N);
    ctx.font='9px Consolas,Courier New,monospace'; ctx.textAlign='left';
    cols.forEach(function(col,i){{
      var cy=n.y+HDR_H+5+i*ROW_H+ROW_H/2;
      ctx.fillStyle=s.sub;
      ctx.beginPath();ctx.arc(n.x+14,cy,2,0,Math.PI*2);ctx.fill();
      ctx.fillStyle='#5a7888';
      var clbl=col;
      while(ctx.measureText(clbl).width>n.w-28&&clbl.length>3) clbl=clbl.slice(0,-3)+'..';
      ctx.fillText(clbl,n.x+21,cy+1);
    }});
    if(n.cols.length>PREV_N){{
      ctx.fillStyle=s.sub; ctx.font='8px Segoe UI,system-ui,sans-serif';
      ctx.fillText('+'+(n.cols.length-PREV_N)+' more',n.x+10,n.y+HDR_H+5+PREV_N*ROW_H+3);
    }}
  }}
  ctx.globalAlpha=1;
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

// ── Mouse interaction ─────────────────────────────────────────────────────────
cv.addEventListener('mousedown',function(e){{
  mouseDownX=e.clientX; mouseDownY=e.clientY; nodeDragMoved=false;
  var p=s2w(e.offsetX,e.offsetY);
  var hit=hitNode(p.x,p.y);
  if(hit){{
    draggingNode=hit; dragOffX=p.x-hit.x; dragOffY=p.y-hit.y;
    cv.style.cursor='grabbing';
  }} else {{
    panning={{mx:e.clientX,my:e.clientY,tx:tx,ty:ty}};
    cv.style.cursor='grabbing';
  }}
}});

cv.addEventListener('mousemove',function(e){{
  if(draggingNode){{
    if(Math.abs(e.clientX-mouseDownX)>3||Math.abs(e.clientY-mouseDownY)>3)
      nodeDragMoved=true;
    var p=s2w(e.offsetX,e.offsetY);
    draggingNode.x=p.x-dragOffX; draggingNode.y=p.y-dragOffY;
    render();
  }} else if(panning){{
    tx=panning.tx+(e.clientX-panning.mx);
    ty=panning.ty+(e.clientY-panning.my);
    render();
  }}
}});

cv.addEventListener('mouseup',function(e){{
  var hitN=draggingNode, wasPan=panning;
  var panDx=wasPan?Math.abs(e.clientX-panning.mx):0;
  var panDy=wasPan?Math.abs(e.clientY-panning.my):0;
  var wasClick=hitN&&!nodeDragMoved;
  var wasPanClick=wasPan&&panDx<4&&panDy<4;
  draggingNode=null; panning=null; nodeDragMoved=false;
  cv.style.cursor='grab';
  if(wasClick){{ selectNode(hitN.id); }}
  else if(wasPanClick){{ clearSelection(); closePanel(); }}
}});

cv.addEventListener('mouseleave',function(){{
  draggingNode=null; panning=null; cv.style.cursor='grab';
}});

cv.addEventListener('wheel',function(e){{
  e.preventDefault();
  var d=e.deltaY>0?0.88:1.14;
  tx=e.offsetX-d*(e.offsetX-tx); ty=e.offsetY-d*(e.offsetY-ty);
  sc=Math.max(0.04,Math.min(6,sc*d));
  document.getElementById('sb-z').textContent=Math.round(sc*100)+'%';
  render();
}},{{passive:false}});

// ── Selection ─────────────────────────────────────────────────────────────────
function selectNode(id){{
  selectedId=id; hlNodes=new Set([id]);
  var inE=edgesTo[id]||[], outE=edgesFrom[id]||[];
  inE.forEach(function(e){{hlNodes.add(e.from);}});
  outE.forEach(function(e){{hlNodes.add(e.to);}});
  render(); openPanel(id);
}}
function clearSelection(){{selectedId=null;hlNodes=new Set();render();}}

// ── Panel ─────────────────────────────────────────────────────────────────────
function openPanel(id){{
  var nd=NM[id]; if(!nd) return;
  document.getElementById('panel').classList.add('open');
  resize();  // shrink canvas to make room for panel
  document.getElementById('ph-title').textContent=id;
  var inE=edgesTo[id]||[], outE=edgesFrom[id]||[];
  document.getElementById('ph-meta').textContent=
    TL[nd.ntype]+' \u2022 Level '+nd.level+
    ' \u2022 '+nd.cols.length+' cols'+
    ' \u2022 '+inE.length+' inputs \u2022 '+outE.length+' outputs';
  document.getElementById('ph-meta').style.color=TC[nd.ntype]+'88';
  buildCols(nd.cols);
  buildRelSec('src',inE,true);
  buildRelSec('dst',outE,false);
}}

function buildCols(cols){{
  document.getElementById('cnt-cols').textContent=cols.length;
  var body=document.getElementById('body-cols'); body.innerHTML='';
  if(!cols.length){{
    var em=document.createElement('div');
    em.style.cssText='padding:7px 13px;color:#1a2e3e;font-size:.7rem;font-style:italic;';
    em.textContent='No columns found (external/base table)';
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
    em.style.cssText='padding:7px 13px;color:#162030;font-size:.7rem;font-style:italic;';
    em.textContent=isSource?'No source tables':'No downstream tables';
    body.appendChild(em); return;
  }}
  edges.forEach(function(e){{
    var otherId=isSource?e.from:e.to;
    var nd2=NM[otherId], ntype=nd2?nd2.ntype:'ext', col=TC[ntype]||'#718096';
    var fa=e.src_alias||'', ta=e.tgt_alias||'', da=isSource?fa:ta;
    var row=document.createElement('div'); row.className='trow';
    row.addEventListener('click',function(){{
      selectNode(otherId);
      focusNode(otherId);
    }});
    var top=document.createElement('div'); top.className='trow-top';
    var dot=document.createElement('div'); dot.className='tdot';
    dot.style.background=col; dot.style.border='1px solid '+col+'55';
    var nm=document.createElement('div'); nm.className='tname'; nm.textContent=otherId;
    top.appendChild(dot); top.appendChild(nm);
    if(da){{var ab=document.createElement('span'); ab.className='talias';
      ab.textContent='alias: '+da; top.appendChild(ab);}}
    row.appendChild(top);
    var jks=e.jk||[];
    if(jks.length){{
      var tbl=document.createElement('table'); tbl.className='jktbl';
      var thead=tbl.createTHead(),hr=thead.insertRow();
      function th(t){{var el=document.createElement('th');el.textContent=t;hr.appendChild(el);}}
      th('Source ('+(fa||e.from.slice(0,14))+')');th('');th('Dest ('+(ta||e.to.slice(0,14))+')');
      var tb2=tbl.createTBody();
      jks.forEach(function(jk){{
        var tr=tb2.insertRow();
        if(jk.left&&jk.right){{
          tr.insertCell().className='jk-s';tr.cells[0].textContent=jk.left;
          tr.insertCell().className='jk-e';tr.cells[1].textContent='=';
          tr.insertCell().className='jk-d';tr.cells[2].textContent=jk.right;
        }}else if(jk.col){{
          var td=tr.insertCell();td.className='jk-s';td.textContent=jk.col;td.colSpan=3;
        }}
      }});
      row.appendChild(tbl);
    }}else{{
      var nj=document.createElement('div');nj.className='no-jk';
      nj.textContent='(no explicit join key)';row.appendChild(nj);
    }}
    body.appendChild(row);
  }});
}}

function closePanel(){{document.getElementById('panel').classList.remove('open');clearSelection();resize();}}
function toggleSec(s){{document.getElementById('hdr-'+s).classList.toggle('collapsed');}}
function focusNode(id){{
  var n=NM[id]; if(!n) return;
  var r=cv.getBoundingClientRect();
  tx=r.width/2-(n.x+n.w/2)*sc;
  ty=r.height/2-(n.y+n.h/2)*sc;
  render();
}}

// ── Camera controls ───────────────────────────────────────────────────────────
function resetView(){{
  clearSelection(); closePanel(); fitAll();
}}
function fitAll(){{
  hiddenTypes=new Set();
  ['ikg','tmp','ext'].forEach(function(t){{
    document.getElementById('btn-'+t).classList.add('on');
  }});
  fitVisible();
}}
function fitVisible(){{
  var vis=NODES.filter(isVis); if(!vis.length) return;
  var x1=Math.min.apply(null,vis.map(function(n){{return n.x;}}));
  var x2=Math.max.apply(null,vis.map(function(n){{return n.x+n.w;}}));
  var y1=Math.min.apply(null,vis.map(function(n){{return n.y;}}));
  var y2=Math.max.apply(null,vis.map(function(n){{return n.y+n.h;}}));
  var pad=50;
  sc=Math.min((cv.width-pad*2)/(x2-x1||1),(cv.height-pad*2)/(y2-y1||1),1.4);
  tx=(cv.width-(x2-x1)*sc)/2-x1*sc;
  ty=(cv.height-(y2-y1)*sc)/2-y1*sc;
  document.getElementById('sb-z').textContent=Math.round(sc*100)+'%';
  render();
}}
function doZoom(f){{
  var cx=cv.width/2,cy=cv.height/2;
  tx=cx-f*(cx-tx); ty=cy-f*(cy-ty);
  sc=Math.max(0.04,Math.min(6,sc*f));
  document.getElementById('sb-z').textContent=Math.round(sc*100)+'%';
  render();
}}
function toggleType(t){{
  var btn=document.getElementById('btn-'+t);
  if(hiddenTypes.has(t)){{hiddenTypes.delete(t);btn.classList.add('on');}}
  else{{hiddenTypes.add(t);btn.classList.remove('on');}}
  render();
}}
function onSearch(){{srchQ=document.getElementById('srch').value.toLowerCase().trim();render();}}
function clearSearch(){{srchQ='';document.getElementById('srch').value='';render();}}

// ── Resize ────────────────────────────────────────────────────────────────────
function resize(){{
  var wrapper=document.getElementById('cv-wrap');
  cv.width=wrapper.clientWidth; cv.height=wrapper.clientHeight;
  render();
}}
window.addEventListener('resize',resize);

window.addEventListener('load',function(){{
  resize(); fitAll();
}});
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
