# nlg_dag_agent

An agent that statically analyzes the NLG Airflow codebase (`nlg-dags` +
`nlg-src`) and builds the full hierarchy:

```
DAG  ->  sub-DAG / TaskGroup  ->  nested sub-DAG(s)  ->  task/sub-task  ->  method  ->  file
```

exactly as requested: parent DAG `nlg_master`, walking down through
`check_odm_run_completion`, `populate_latest_odm_insights`, `nlg_preprocess`,
`nlg_rules`, `nlg_product_client_briefings`, `nlg_product_others`,
`nlg_json_insights`, `nlg_post_process`, etc.

It also does impact analysis: *"if I change file/method/target_type X, which
tasks and DAGs are affected?"*, and tags every task with `target_type`,
`product`, and any `si_*.yml` insight-type files it maps to.

**Tested against the attached `nlg-dag.zip` / `nlg-src.zip`: 33 DAGs, 293
tasks, 100% of methods resolved to a concrete file — with the LLM turned
off.** The LLM is optional, used only as a fallback / for free-text Q&A.

## Why it's accurate without an LLM

The hierarchy extraction is pure Python `ast` analysis, tuned to the two
authoring idioms actually used in this repo:

1. `with TaskGroup(...) as x:` wrapped by a `def get_xxx():` function,
   called from `with DAG(...) as dag:` (e.g. `nlg_master.py` /
   `nlg_taskgroups.py`), **and**
2. `dag = DAG('some_id', ...)` plain assignment with tasks referencing
   `dag=dag` as separate module-level statements (e.g.
   `nlg_prospect_master.py`, `staat_ds_nlg.py`).

It also specifically detects the dynamic-dispatch idiom used for most rule/
product tasks:

```python
def task_run_nlg_main(**kwargs):
    nlg_obj = NLG(args=nlg_args, ...)
    action_str = f"nlg_obj.{nlg_args['action']}()"
    eval(action_str)
```

Given a task with `op_kwargs={"action": "generate_rule_narratives", ...}`,
the agent resolves this to `NLG.generate_rule_narratives` in
`nlg_main_class.py` — without needing an LLM call.

`PostgresOperator` tasks (raw SQL, no python method) are recorded as
`method: "N/A"`, `note: "Postgres operator with sql code"`, per the spec.

## Files

| File | Purpose |
|---|---|
| `nlg_dag_agent/` | The package. Import it the same way from a notebook or an Airflow DAG. |
| `run_nlg_hierarchy_agent.ipynb` | Jupyter notebook — run interactively, prompts for keys. |
| `airflow_dag_nlg_hierarchy_agent.py` | Drop-in Airflow DAG — non-interactive, reads Airflow Variables. |
| `sample_output/nlg_hierarchy.json` | Full extracted hierarchy for the attached zips, for reference. |

### `nlg_dag_agent/` internals

| Module | Purpose |
|---|---|
| `config.py` | All settings (LLM + GitLab), mirrors your provided config, all env-var overridable. Secret resolution: env var → Airflow Variable → interactive prompt. |
| `source_loader.py` | `LocalSourceRepo` (a directory on disk) and `GitLabSourceRepo` (reads via GitLab REST API, no clone needed). |
| `dag_ast_parser.py` | The core static extractor: DAG → TaskGroup → Task, for both authoring styles found in this repo. |
| `src_scanner.py` | Indexes `nlg-src`: class → methods, function → file, and `target_type` → resource files (yaml/sql/ini), including `si_*.yml` → insight_type. |
| `dispatch_resolver.py` | Detects the `eval(f"{obj}.{kwargs['action']}()")` dynamic-dispatch idiom. |
| `llm_client.py` | Azure OpenAI / internal LLM Gateway REST client (used only as fallback + for `ask()`). |
| `pipeline.py` | Orchestrates everything: `build_hierarchy()`, `run_impact_analysis()`, `ask()`, tree printing, JSON export. |
| `__main__.py` | CLI (`python -m nlg_dag_agent build / impact`). |

## Quick start — Jupyter

```python
from nlg_dag_agent import Settings, build_hierarchy, run_impact_analysis, ask

settings = Settings()
settings.local_dags_dir = "./nlg-dag/dags"   # or leave unset to use GitLab
settings.local_src_dir  = "./nlg-src/src"
settings.resolve_secrets(interactive=True)    # prompts for GitLab token / LLM key only if needed

hierarchy = build_hierarchy(settings)
hierarchy.print_tree()
hierarchy.save_json("nlg_hierarchy.json")

run_impact_analysis(hierarchy, changed_file="nlg_main_class.py")
run_impact_analysis(hierarchy, target_type="client_windfall")
ask(hierarchy, "What breaks if I change generate_rule_narratives?")
```

See `run_nlg_hierarchy_agent.ipynb` for the full walkthrough.

## Quick start — Airflow

1. Copy `nlg_dag_agent/` and `airflow_dag_nlg_hierarchy_agent.py` into your
   `dags/` folder (alongside `nlg_master.py`).
2. Set the Airflow Variable `GENESIS_DDLC_IKG_GIT_SECRET` (already used
   elsewhere in this codebase) — it's reused for both the GitLab token and
   the LLM key by default; point `GITLAB_TOKEN_VAR` / `LLM_AIRFLOW_SECRET_VAR`
   env vars at different Variables if they should differ.
3. Trigger `nlg_hierarchy_agent` on its own schedule to keep a fresh
   `nlg_hierarchy.json` artifact, or trigger it on-demand with:
   ```json
   {"changed_file": "nlg_main_class.py"}
   {"target_type": "client_windfall"}
   {"question": "What breaks if I change generate_rule_narratives?"}
   ```
   via `dag_run.conf`.

## Quick start — GitLab (no local checkout)

```python
settings = Settings()   # NLG_PROJECT_PATH / NLG_DAGS_SUBPATH / NLG_SRC_SUBPATH
                          # already default to dags/nlg/dags and dags/nlg/src
settings.resolve_secrets(interactive=True)   # prompts for GitLab token + LLM key
hierarchy = build_hierarchy(settings)
```

## Performance

`build_hierarchy()` was originally slow in **GitLab mode** because it fetched
every single file with its own HTTP request (~2,000+ round-trips just for
`nlg-src`). It now:

1. Downloads the whole `dags/nlg/dags` and `dags/nlg/src` subtrees in **one
   request each** via GitLab's `repository/archive.tar.gz` endpoint, and
   reads every file out of the in-memory tarball.
2. Falls back to the old tree+raw-file approach only if the archive
   endpoint isn't reachable (older GitLab, permissions, proxy) — but that
   fallback now fetches files **in parallel** (24 concurrent requests)
   instead of one at a time.
3. Caches the downloaded contents on disk (`settings.gitlab_cache_dir`,
   default `./.nlg_dag_agent_cache`), keyed by the ref's current commit
   SHA — a second run against an unchanged branch does **zero** network
   calls. Delete the cache dir, or bump the ref, to force a re-download.
4. `target_type` → resource-file matching (`si_*.yml`, config, rules...) is
   now a one-time O(files) index built once, instead of an O(files) scan
   repeated for every task — this matters once a repo has hundreds of
   rule tasks.

Local mode (`local_dags_dir` / `local_src_dir`) was already fast (disk I/O),
but `read_all()` now also parallelizes file reads for very large or
network-mounted checkouts.

On the attached `nlg-dag.zip`/`nlg-src.zip` (2,157 files, 33 DAGs, 293
tasks), a full local build takes well under 2 seconds.

If you still see GitLab mode being slow, the likely causes are (in order):
- the archive endpoint being blocked by a proxy/WAF, silently forcing the
  parallel-fallback path — check for a log line/exception around
  `_download_archive`;
- LLM fallback being triggered for many unresolved tasks (each is a network
  call) — set `llm_provider = "none"` or check `resolution="unresolved"`
  counts to see how many tasks actually need it;
- a cold cache on the very first run against a new ref (expected — the
  *second* run should be near-instant).



- Execution order (`>>` chains) isn't extracted, only containment
  (DAG/sub-DAG/task/method/file) — add it in `dag_ast_parser._walk_stmts`
  if you need it.
- `target_type` → resource-file matching relies on the naming conventions
  observed in this repo (`group_logic/<target_type>/`, `rules/<target_type>/`,
  `si_<target_type>.yml`, ...). If new conventions appear, extend
  `src_scanner.resources_for_target_type`.
- The internal LLM Gateway's exact request/response contract wasn't in the
  provided config screenshot, so `llm_client._call_gateway` assumes an
  OpenAI-compatible `/v1/chat/completions` shape — adjust if it differs.
- CI/offline testing here used the attached `nlg-dag.zip` / `nlg-src.zip`;
  swap in `settings.local_dags_dir` / `local_src_dir` for any other checkout,
  or switch to GitLab mode with no code changes.
