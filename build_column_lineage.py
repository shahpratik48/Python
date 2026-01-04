"""
Build semantic column lineage from a data dictionary (XLSX) using similarity + an LLM.

Input XLSX must contain (or be mappable to) these fields:
  - table_name
  - column_name
  - business attribute name
  - description

This produces a lineage edge list (CSV/JSON) that you can load into a graph DB.

Notes / limits:
  - With only a data dictionary (no SQL/ETL code), this builds *semantic lineage*
    (same business meaning / likely derivations), not guaranteed execution lineage.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import yaml
from pydantic import BaseModel, Field, ValidationError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


# -----------------------------
# Models
# -----------------------------


@dataclass(frozen=True)
class ColumnMeta:
    table_name: str
    column_name: str
    business_attribute_name: str
    description: str

    @property
    def column_id(self) -> str:
        return f"{self.table_name}.{self.column_name}"

    def as_prompt_dict(self) -> Dict[str, str]:
        return {
            "table_name": self.table_name,
            "column_name": self.column_name,
            "business_attribute_name": self.business_attribute_name,
            "description": self.description,
        }


class LineageEdge(BaseModel):
    source_table: str
    source_column: str
    target_table: str
    target_column: str
    relation_type: str = Field(
        ...,
        description=(
            "One of: same_as | derived_from | aggregated_from | lookup_from | filtered_from | unknown"
        ),
    )
    transformation: str = Field(
        ...,
        description="Plain-English transformation (or 'none' for same_as).",
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: str = Field(
        ...,
        description="Short explanation referencing business attribute name / description.",
    )


class LineageResponse(BaseModel):
    edges: List[LineageEdge] = Field(default_factory=list)
    notes: str = ""


@dataclass(frozen=True)
class Candidate:
    source_idx: int
    score: float


# -----------------------------
# Config / IO helpers
# -----------------------------


def _norm_header(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return str(x).strip()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def read_data_dictionary_xlsx(
    xlsx_path: Path,
    sheet_name: Optional[Any],
    column_mapping: Optional[Dict[str, str]],
) -> List[ColumnMeta]:
    # pandas behavior:
    # - sheet_name omitted or 0 -> first sheet (DataFrame)
    # - sheet_name=None -> ALL sheets (dict[str, DataFrame])
    # To keep configs simple, treat None/"" as "first sheet".
    sn = sheet_name
    if sn is None or (isinstance(sn, str) and not sn.strip()):
        sn = 0

    df = pd.read_excel(xlsx_path, sheet_name=sn)
    if isinstance(df, dict):
        # If a caller supplied sheet_name=None (or a list), pick the first sheet deterministically.
        if not df:
            raise ValueError("Excel workbook has no readable sheets.")
        df = next(iter(df.values()))

    if df is None or df.empty:
        raise ValueError("Excel sheet is empty or could not be read.")

    # Auto-detect columns if mapping not provided.
    columns_by_norm = {_norm_header(c): c for c in df.columns}

    required_norm = {
        "table_name": ["tablename", "table", "table_name"],
        "column_name": ["columnname", "column", "column_name", "field", "fieldname"],
        "business_attribute_name": [
            "businessattributename",
            "businessattribute",
            "business attribute name",
            "business_attribute_name",
            "businessattributename",
            "attributename",
            "attribute",
        ],
        "description": ["description", "desc", "definition", "businessdefinition"],
    }

    def pick_col(norm_keys: List[str]) -> Optional[str]:
        for k in norm_keys:
            nk = _norm_header(k)
            if nk in columns_by_norm:
                return columns_by_norm[nk]
        return None

    if column_mapping:
        # Config mapping uses logical keys to actual excel headers.
        mapped = {}
        for logical_key in required_norm.keys():
            if logical_key not in column_mapping:
                raise ValueError(f"Missing column_mapping for '{logical_key}'.")
            mapped[logical_key] = column_mapping[logical_key]
    else:
        mapped = {}
        for logical_key, keys in required_norm.items():
            col = pick_col(keys)
            if not col:
                raise ValueError(
                    f"Could not auto-detect required column '{logical_key}'. "
                    f"Available columns: {list(df.columns)}"
                )
            mapped[logical_key] = col

    records: List[ColumnMeta] = []
    for _, row in df.iterrows():
        table_name = _safe_str(row.get(mapped["table_name"]))
        column_name = _safe_str(row.get(mapped["column_name"]))
        business_attribute_name = _safe_str(row.get(mapped["business_attribute_name"]))
        description = _safe_str(row.get(mapped["description"]))
        if not table_name or not column_name:
            continue
        records.append(
            ColumnMeta(
                table_name=table_name,
                column_name=column_name,
                business_attribute_name=business_attribute_name,
                description=description,
            )
        )

    if not records:
        raise ValueError("No usable rows found (table_name/column_name missing).")
    return records


# -----------------------------
# Similarity / candidates
# -----------------------------


def column_text(meta: ColumnMeta) -> str:
    # A compact representation for semantic similarity.
    parts = [
        f"business_attribute: {meta.business_attribute_name}",
        f"description: {meta.description}",
        f"table: {meta.table_name}",
        f"column: {meta.column_name}",
    ]
    return " | ".join([p for p in parts if p and not p.endswith(": ")])


def build_candidates(
    cols: List[ColumnMeta],
    top_k: int,
    min_score: float,
    include_same_table: bool,
) -> List[List[Candidate]]:
    texts = [column_text(c) for c in cols]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
    )
    X = vectorizer.fit_transform(texts)
    sim = cosine_similarity(X, X)

    # Business-attribute grouping for boosting candidates.
    ba_to_indices: Dict[str, List[int]] = {}
    for i, c in enumerate(cols):
        ba = _norm_header(c.business_attribute_name)
        if ba:
            ba_to_indices.setdefault(ba, []).append(i)

    all_candidates: List[List[Candidate]] = []
    for i, c in enumerate(cols):
        candidates: Dict[int, float] = {}

        # 1) Always include same business-attribute group (if any).
        ba = _norm_header(c.business_attribute_name)
        if ba and ba in ba_to_indices:
            for j in ba_to_indices[ba]:
                if j == i:
                    continue
                if (not include_same_table) and (cols[j].table_name == c.table_name):
                    continue
                candidates[j] = max(candidates.get(j, 0.0), float(sim[i, j]) + 0.15)

        # 2) Add similarity top-k.
        scored = sorted(
            ((j, float(sim[i, j])) for j in range(len(cols)) if j != i),
            key=lambda x: x[1],
            reverse=True,
        )
        for j, s in scored:
            if (not include_same_table) and (cols[j].table_name == c.table_name):
                continue
            if s < min_score:
                break
            candidates[j] = max(candidates.get(j, 0.0), s)
            if len(candidates) >= top_k:
                break

        # Final sorted list.
        cand_list = [
            Candidate(source_idx=j, score=min(1.0, s)) for j, s in candidates.items()
        ]
        cand_list.sort(key=lambda x: x.score, reverse=True)
        all_candidates.append(cand_list[:top_k])

    return all_candidates


def table_layer_hint(table_name: str) -> int:
    """
    Heuristic ordering to help directionality:
      smaller number = more upstream
    """
    t = table_name.lower()
    if any(x in t for x in ["raw", "landing", "ingest"]):
        return 0
    if any(x in t for x in ["stg", "stage", "staging"]):
        return 1
    if any(x in t for x in ["ods", "core"]):
        return 2
    if any(x in t for x in ["dim", "fact", "mart", "report", "rpt", "bi"]):
        return 3
    return 2


def suggest_direction(source: ColumnMeta, target: ColumnMeta) -> str:
    ls = table_layer_hint(source.table_name)
    lt = table_layer_hint(target.table_name)
    if ls < lt:
        return "source_to_target"
    if ls > lt:
        return "target_to_source"
    return "unknown"


# -----------------------------
# LLM client(s)
# -----------------------------


class LLMClient:
    def complete_json(self, *, system: str, user: str) -> Dict[str, Any]:
        raise NotImplementedError


class OpenAIChatClient(LLMClient):
    def __init__(self, model: str):
        from openai import OpenAI  # type: ignore

        self._client = OpenAI()
        self._model = model

    def complete_json(self, *, system: str, user: str) -> Dict[str, Any]:
        # Use a plain JSON instruction + robust parsing to support broad model set.
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
        )
        text = resp.choices[0].message.content or ""
        return extract_json_obj(text)


class OllamaClient(LLMClient):
    def __init__(self, model: str, base_url: str):
        import requests  # type: ignore

        self._requests = requests
        self._model = model
        self._base_url = base_url.rstrip("/")

    def complete_json(self, *, system: str, user: str) -> Dict[str, Any]:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": 0.1},
        }
        r = self._requests.post(f"{self._base_url}/api/chat", json=payload, timeout=180)
        r.raise_for_status()
        data = r.json()
        text = (data.get("message") or {}).get("content") or ""
        return extract_json_obj(text)


def build_llm_client(cfg: Dict[str, Any]) -> Optional[LLMClient]:
    provider = (cfg.get("provider") or "").lower().strip()
    if not provider or provider == "none":
        return None
    if provider == "openai":
        model = cfg.get("model") or "gpt-4.1-mini"
        return OpenAIChatClient(model=model)
    if provider == "ollama":
        model = cfg.get("model") or "llama3.1"
        base_url = cfg.get("base_url") or "http://localhost:11434"
        return OllamaClient(model=model, base_url=base_url)
    raise ValueError(f"Unsupported llm.provider: {provider}")


# -----------------------------
# Prompting / JSON parsing
# -----------------------------


SYSTEM_PROMPT = """You are a data lineage expert.
You infer likely COLUMN-LEVEL lineage using only a data dictionary (table/column names, business attribute name, descriptions).
Return ONLY valid JSON that matches the requested schema.
If you are uncertain, return no edges or use low confidence (<=0.4) and relation_type 'unknown'.
Never invent tables/columns not present in the input.
"""


def build_user_prompt(
    *,
    target: ColumnMeta,
    candidates: List[Tuple[ColumnMeta, float]],
    max_edges: int,
) -> str:
    cand_payload = []
    for c, score in candidates:
        cand_payload.append(
            {
                "table_name": c.table_name,
                "column_name": c.column_name,
                "business_attribute_name": c.business_attribute_name,
                "description": c.description,
                "similarity_score": round(score, 4),
                "direction_hint": suggest_direction(c, target),
            }
        )

    schema = {
        "edges": [
            {
                "source_table": "string",
                "source_column": "string",
                "target_table": "string",
                "target_column": "string",
                "relation_type": "same_as|derived_from|aggregated_from|lookup_from|filtered_from|unknown",
                "transformation": "string",
                "confidence": "0..1",
                "evidence": "string",
            }
        ],
        "notes": "string",
    }

    return (
        "Given the TARGET column and a list of CANDIDATE source columns, infer the most likely lineage edges.\n"
        f"Rules:\n"
        f"- Output at most {max_edges} edges.\n"
        "- Prefer candidates with same business_attribute_name and compatible descriptions.\n"
        "- If the target looks like a surrogate key or audit field, usually no lineage unless clear.\n"
        "- Use direction_hint when plausible, but you may override it.\n"
        "- If nothing is convincing, return edges=[] with a note.\n\n"
        f"JSON schema (example types):\n{json.dumps(schema, indent=2)}\n\n"
        f"TARGET:\n{json.dumps(target.as_prompt_dict(), indent=2)}\n\n"
        f"CANDIDATES:\n{json.dumps(cand_payload, indent=2)}\n"
    )


def extract_json_obj(text: str) -> Dict[str, Any]:
    """
    Robustly extract first JSON object from model output.
    """
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)

    # Try to find the first {...} block.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("Could not extract JSON object from model output.")


# -----------------------------
# Caching
# -----------------------------


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_cache(cache_path: Path) -> Dict[str, Dict[str, Any]]:
    if not cache_path.exists():
        return {}
    data: Dict[str, Dict[str, Any]] = {}
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "key" in obj and "value" in obj:
                data[obj["key"]] = obj["value"]
    return data


def append_cache(cache_path: Path, key: str, value: Dict[str, Any]) -> None:
    with cache_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")


# -----------------------------
# Export
# -----------------------------


def write_edges_csv(edges: List[LineageEdge], path: Path) -> None:
    import csv

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "source_table",
                "source_column",
                "target_table",
                "target_column",
                "relation_type",
                "transformation",
                "confidence",
                "evidence",
            ],
        )
        w.writeheader()
        for e in edges:
            w.writerow(e.model_dump())


def write_edges_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# -----------------------------
# Main pipeline
# -----------------------------


def run(cfg: Dict[str, Any]) -> int:
    input_cfg = cfg.get("input") or {}
    out_cfg = cfg.get("output") or {}
    cand_cfg = cfg.get("candidates") or {}
    llm_cfg = cfg.get("llm") or {}

    xlsx_path = Path(input_cfg.get("path") or "")
    if not xlsx_path.exists():
        raise FileNotFoundError(
            f"Input xlsx not found: {xlsx_path}. "
            "Copy the file locally or mount it into this environment."
        )

    sheet_name = input_cfg.get("sheet_name")
    column_mapping = input_cfg.get("column_mapping")

    out_dir = Path(out_cfg.get("dir") or "out_lineage")
    ensure_out_dir(out_dir)

    cols = read_data_dictionary_xlsx(
        xlsx_path=xlsx_path,
        sheet_name=sheet_name,
        column_mapping=column_mapping,
    )

    top_k = int(cand_cfg.get("top_k") or 20)
    min_score = float(cand_cfg.get("min_score") or 0.15)
    include_same_table = bool(cand_cfg.get("include_same_table") or False)
    max_edges_per_target = int(cand_cfg.get("max_edges_per_target") or 2)

    candidates = build_candidates(
        cols=cols,
        top_k=top_k,
        min_score=min_score,
        include_same_table=include_same_table,
    )

    llm_client = build_llm_client(llm_cfg)
    cache_path = out_dir / "llm_cache.jsonl"
    cache = load_cache(cache_path)

    edges: List[LineageEdge] = []
    per_target_debug: Dict[str, Any] = {}

    # If no LLM configured, fall back to similarity-only edges.
    if llm_client is None:
        for i, target in enumerate(cols):
            for cand in candidates[i]:
                src = cols[cand.source_idx]
                edges.append(
                    LineageEdge(
                        source_table=src.table_name,
                        source_column=src.column_name,
                        target_table=target.table_name,
                        target_column=target.column_name,
                        relation_type="unknown",
                        transformation="similarity_candidate",
                        confidence=max(0.0, min(1.0, cand.score)),
                        evidence="Created without LLM; based on TF-IDF similarity.",
                    )
                )
        payload = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "mode": "similarity_only",
            "edges": [e.model_dump() for e in edges],
        }
        write_edges_csv(edges, out_dir / "lineage_edges.csv")
        write_edges_json(out_dir / "lineage_edges.json", payload)
        return 0

    for i, target in enumerate(tqdm(cols, desc="Inferring lineage")):
        cand = candidates[i]
        if not cand:
            continue
        cand_metas = [(cols[c.source_idx], c.score) for c in cand]

        user_prompt = build_user_prompt(
            target=target,
            candidates=cand_metas,
            max_edges=max_edges_per_target,
        )
        cache_key = sha256_str(SYSTEM_PROMPT + "\n" + user_prompt)

        if cache_key in cache:
            raw = cache[cache_key]
        else:
            raw = llm_client.complete_json(system=SYSTEM_PROMPT, user=user_prompt)
            append_cache(cache_path, cache_key, raw)

        per_target_debug[target.column_id] = {
            "target": target.as_prompt_dict(),
            "candidates": [
                {**m.as_prompt_dict(), "similarity_score": s} for m, s in cand_metas
            ],
            "raw_llm_json": raw,
        }

        try:
            parsed = LineageResponse.model_validate(raw)
        except ValidationError:
            # If validation fails, skip rather than producing invalid edges.
            continue

        # Keep only edges that reference the actual target.
        for e in parsed.edges:
            if (
                _norm_header(e.target_table) == _norm_header(target.table_name)
                and _norm_header(e.target_column) == _norm_header(target.column_name)
            ):
                edges.append(e)

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "mode": "llm_confirmed",
        "edge_count": len(edges),
        "edges": [e.model_dump() for e in edges],
    }
    write_edges_csv(edges, out_dir / "lineage_edges.csv")
    write_edges_json(out_dir / "lineage_edges.json", payload)
    write_edges_json(out_dir / "debug_per_target.json", per_target_debug)
    return 0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build column lineage from xlsx dictionary.")
    p.add_argument(
        "--config",
        required=True,
        help="Path to YAML config (see lineage_config.example.yaml).",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cfg = load_config(Path(args.config))
    return run(cfg)


if __name__ == "__main__":
    raise SystemExit(main())

