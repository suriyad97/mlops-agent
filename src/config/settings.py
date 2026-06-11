"""Central configuration via pydantic-settings. All values come from .env or environment."""
from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM (GitHub Models)
    github_token: str = ""
    github_model: str = "gpt-4o"

    # Source control
    repo_pat: str = ""          # PAT used to clone/push the target repository
    azdo_pat: str = ""          # Azure DevOps PAT (falls back to repo_pat)
    azdo_org_url: str = ""      # e.g. https://dev.azure.com/myorg (derived from repo URL if empty)
    azdo_project: str = ""      # derived from repo URL if empty

    # Azure
    azure_subscription_id: str = ""
    azure_tenant_id: str = ""
    aml_resource_group: str = Field(
        default="", validation_alias=AliasChoices("AML_RESOURCE_GROUP", "AZURE_RESOURCE_GROUP")
    )
    aml_workspace: str = Field(
        default="", validation_alias=AliasChoices("AML_WORKSPACE", "AZURE_ML_WORKSPACE")
    )
    aml_compute_target: str = Field(
        default="", validation_alias=AliasChoices("AML_COMPUTE_TARGET", "AZURE_ML_COMPUTE_TARGET")
    )
    aml_environment_name: str = Field(
        default="", validation_alias=AliasChoices("AML_ENVIRONMENT_NAME", "AZURE_ML_ENVIRONMENT_NAME")
    )
    aml_experiment_name: str = Field(
        default="remediation-agent",
        validation_alias=AliasChoices("AML_EXPERIMENT_NAME", "AZURE_ML_EXPERIMENT_NAME"),
    )
    acr_name: str = ""

    # Observability (Arize Phoenix)
    phoenix_enabled: bool = True
    phoenix_collector_endpoint: str = "http://localhost:6006"
    phoenix_project_name: str = "repo-remediation-agent"

    # Runtime behaviour
    workspace_dir: str = "repos"
    memory_path: str = ".agent_memory/memory.json"
    max_retries: int = 3
    max_reflection_rounds: int = 2
    remediation_branch: str = "remediation/agent"
    log_level: str = "INFO"

    @property
    def effective_azdo_pat(self) -> str:
        return self.azdo_pat or self.repo_pat


@lru_cache
def get_settings() -> Settings:
    return Settings()
