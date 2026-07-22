"""
Configuration for nlg_dag_agent.

Everything is overridable via environment variables so the exact same
module works locally, in a notebook, or inside Airflow (where env vars /
Airflow Variables are typically used instead of a .env file).

Secrets (LLM API key, GitLab token) are NEVER hard-coded. Use
`Settings.resolve_secrets(interactive=...)` to populate them:
  - interactive=True  (typical notebook use): env var -> Airflow Variable
    (if Airflow is importable) -> interactive prompt (getpass).
  - interactive=False (typical Airflow task use): env var -> Airflow
    Variable -> raise if still missing.
"""
from __future__ import annotations

import os
import getpass
from dataclasses import dataclass, field
from typing import Optional, Dict


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.getenv(name, default)


@dataclass
class Settings:
    # ------------------------------------------------------------------
    # LLM settings
    # ------------------------------------------------------------------
    # "azure_openai" -> Azure OpenAI (GPT-4.1)
    # "azure_gateway" -> internal LLM Gateway (llama3.1:8b or similar)
    # "none"          -> disable LLM entirely, static analysis only
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "azure_openai"))

    azure_openai_endpoint: str = field(default_factory=lambda: _env(
        "AZURE_OPENAI_ENDPOINT", "https://cirruspl-staat-ste-dev-ai.openai.azure.com/openai/v1/"))
    azure_openai_deployment: str = field(default_factory=lambda: _env("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1"))
    azure_openai_api_version: str = field(default_factory=lambda: _env("AZURE_OPENAI_API_VERSION", "2024-02-01"))
    azure_openai_api_key: Optional[str] = field(default_factory=lambda: _env("AZURE_OPENAI_API_KEY", ""))

    llm_gateway_url: str = field(default_factory=lambda: _env(
        "LLM_GATEWAY_URL", "https://llm.genesis-dev.azpriv-cloud.ubs.net"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "gpt-4.1"))
    llm_api_key: Optional[str] = field(default_factory=lambda: _env("LLM_API_KEY", ""))

    llm_temperature: float = field(default_factory=lambda: float(_env("LLM_TEMPERATURE", "0.1")))
    llm_max_tokens: int = field(default_factory=lambda: int(_env("LLM_MAX_TOKENS", "20000")))

    # Non-interactive fallback: name of the Airflow Variable holding the LLM key
    llm_airflow_secret_var: str = field(default_factory=lambda: _env("LLM_AIRFLOW_SECRET_VAR", "GENESIS_DDLC_IKG_GIT_SECRET"))

    # ------------------------------------------------------------------
    # Git / GitLab settings
    # ------------------------------------------------------------------
    gitlab_url: str = field(default_factory=lambda: _env("GITLAB_URL", "https://devcloud.ubs.net"))
    gitlab_token: Optional[str] = field(default_factory=lambda: _env("GITLAB_TOKEN", ""))
    gitlab_token_var: str = field(default_factory=lambda: _env("GITLAB_TOKEN_VAR", "GENESIS_DDLC_IKG_GIT_SECRET"))
    default_base_branch: str = field(default_factory=lambda: _env("DEFAULT_BASE_BRANCH", "develop"))

    ikg_project_path: str = field(default_factory=lambda: _env(
        "IKG_PROJECT_PATH",
        "ubs/gwma/smart-technology-and-analytics/staat-data-science/staat-ds-genesis/genesis-platform/ikg-dags"))
    nlg_project_path: str = field(default_factory=lambda: _env(
        "NLG_PROJECT_PATH",
        "ubs/gwma/smart-technology-and-analytics/staat-data-science/staat-ds-genesis/genesis-platform/nlg-dags"))
    odm_project_path: str = field(default_factory=lambda: _env(
        "ODM_PROJECT_PATH",
        "ubs/gwma/smart-technology-and-analytics/staat-data-science/staat-ds-genesis/genesis-platform/odm-dags"))

    # Within the nlg-dags GitLab project, the DAG code lives at
    # dags/nlg/dags/*.py and the shared library lives at dags/nlg/src/**.
    # (matches: "Read nlg-dags/dags/nlg/dags & nlg-dags/dags/nlg/src")
    nlg_dags_subpath: str = field(default_factory=lambda: _env("NLG_DAGS_SUBPATH", "dags/nlg/dags"))
    nlg_src_subpath: str = field(default_factory=lambda: _env("NLG_SRC_SUBPATH", "dags/nlg/src"))

    # Optional on-disk cache for GitLab downloads (huge speedup on repeat
    # runs against an unchanged ref -- skips re-downloading entirely).
    # Set to None/"" to disable.
    gitlab_cache_dir: Optional[str] = field(default_factory=lambda: _env("NLG_GITLAB_CACHE_DIR", "./.nlg_dag_agent_cache"))

    @property
    def project_path_by_repo(self) -> Dict[str, str]:
        return {
            "ikg": self.ikg_project_path,
            "nlg": self.nlg_project_path,
            "odm": self.odm_project_path,
        }

    # ------------------------------------------------------------------
    # Local / offline mode (no GitLab access, e.g. a checked-out repo or
    # an extracted zip such as the ones provided during development).
    # If both are set, LOCAL mode is used in preference to GitLab.
    # ------------------------------------------------------------------
    local_dags_dir: Optional[str] = field(default_factory=lambda: _env("NLG_LOCAL_DAGS_DIR"))
    local_src_dir: Optional[str] = field(default_factory=lambda: _env("NLG_LOCAL_SRC_DIR"))

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    output_dir: str = field(default_factory=lambda: _env("NLG_AGENT_OUTPUT_DIR", "./nlg_dag_agent_output"))

    # ------------------------------------------------------------------
    # Secret resolution
    # ------------------------------------------------------------------
    def _airflow_variable(self, key: str) -> Optional[str]:
        try:
            from airflow.models import Variable  # type: ignore
            return Variable.get(key, default_var=None)
        except Exception:
            return None

    def resolve_secrets(self, interactive: bool = True) -> "Settings":
        """Populate llm_api_key / azure_openai_api_key / gitlab_token in-place."""
        # GitLab token
        if not self.gitlab_token:
            self.gitlab_token = self._airflow_variable(self.gitlab_token_var)
        if not self.gitlab_token and interactive:
            self.gitlab_token = getpass.getpass("GitLab personal access token: ").strip() or None

        # LLM key (Azure OpenAI or gateway, whichever provider is active)
        if self.llm_provider == "azure_openai":
            if not self.azure_openai_api_key:
                self.azure_openai_api_key = self._airflow_variable(self.llm_airflow_secret_var)
            if not self.azure_openai_api_key and interactive:
                self.azure_openai_api_key = getpass.getpass("Azure OpenAI API key: ").strip() or None
        elif self.llm_provider == "azure_gateway":
            if not self.llm_api_key:
                self.llm_api_key = self._airflow_variable(self.llm_airflow_secret_var)
            if not self.llm_api_key and interactive:
                self.llm_api_key = getpass.getpass("LLM Gateway API key: ").strip() or None
        return self

    @property
    def llm_enabled(self) -> bool:
        if self.llm_provider == "none":
            return False
        if self.llm_provider == "azure_openai":
            return bool(self.azure_openai_api_key)
        if self.llm_provider == "azure_gateway":
            return bool(self.llm_api_key)
        return False
