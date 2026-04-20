### Goal
Build **column-level lineage** from a data-dictionary Excel file using:
- **Candidate generation**: TF‑IDF similarity + business-attribute grouping
- **LLM confirmation**: emits strict JSON edges (source → target) with relation type + confidence
- **Exports**: `CSV` + `JSON` edge lists (ready for graph loading)

This is **semantic lineage** (best-effort meaning-based links) because the only input is metadata. If you also have SQL/ETL code, you can extend this to execution lineage by parsing queries and letting the LLM reconcile ambiguous mappings.

---

### Input requirements (from your XLSX)
The sheet must contain these columns (exact header names can vary; auto-detection is included):
- `table_name`
- `column_name`
- `business attribute name`
- `description`

If your headers don’t match, set `input.column_mapping` in the config.

---

### Setup
Install dependencies:

```bash
python3 -m pip install -r requirements-lineage.txt
```

If using OpenAI:
- Set `OPENAI_API_KEY` in your environment.

---

### Run
**Preferred:** use the notebook `column_lineage_main.ipynb` as the main entrypoint.

1) Put `ikg data dictionary.xlsx` in the same folder as the notebook (this repo root).
   Your original path is Windows: `c:\dev\ikg data dictionary.xlsx`

2) Update the config:
- `lineage_config.example.yaml` → set `input.path` to your `.xlsx`

3) Execute:

```bash
python3 build_column_lineage.py --config lineage_config.example.yaml
```

---

### Outputs
Written to `output.dir` (default `out_lineage/`):
- `lineage_edges.csv` — edge list for BI/graph tools
- `lineage_edges.json` — same edges in JSON
- `debug_per_target.json` — per-target candidates + raw LLM JSON (useful for review)
- `llm_cache.jsonl` — request/response cache keyed by prompt hash

CSV columns:
- `source_table`, `source_column`
- `target_table`, `target_column`
- `relation_type` (same_as | derived_from | aggregated_from | lookup_from | filtered_from | unknown)
- `transformation`, `confidence`, `evidence`

---

### Recommended operating mode (practical)
- **Phase 1 (cheap, fast)**: run with `llm.provider: none` to generate candidate edges.
- **Phase 2 (higher quality)**: turn on the LLM and re-run; review `debug_per_target.json`.
- **Phase 3 (human-in-the-loop)**: curate “gold” edges for your domain; re-run to measure precision/recall.

---

### Extending to “real” lineage (if you have SQL/ETL later)
Add parsers for:
- dbt models (`target/run/manifest.json` + SQL)
- stored procedures / views
- Spark / Databricks notebooks

Then:
- parse column expressions → deterministic lineage where possible
- use the LLM only to resolve ambiguous joins/aliases and naming mismatches

