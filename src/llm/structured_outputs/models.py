"""Pydantic models for every structured LLM output in the platform."""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# --- Repository analysis ---------------------------------------------------

class MissingAsset(BaseModel):
    path: str = Field(description="Relative path of the file to create")
    asset_type: Literal[
        "docker", "azdo_pipeline", "aml_environment", "aml_component",
        "aml_pipeline", "docs", "tests", "config", "other",
    ] = "other"
    description: str
    template_hint: str = ""


class GapReport(BaseModel):
    missing_files: List[MissingAsset] = Field(default_factory=list)
    existing_mlops_assets: List[str] = Field(default_factory=list)
    summary: str = ""


class RemediationTask(BaseModel):
    name: str
    agent: Literal["docker", "azdo", "aml"]
    rationale: str = ""


class RemediationPlan(BaseModel):
    tasks: List[RemediationTask] = Field(default_factory=list)
    summary: str = ""


class ProjectAnalysis(BaseModel):
    language: str = "python"
    framework: str = ""
    entry_point: str = ""
    python_version: str = "3.12"
    dependencies_file: str = ""
    has_training_code: bool = False
    has_serving_code: bool = False
    notes: str = ""


# --- Generation + reflection -----------------------------------------------

class GeneratedFile(BaseModel):
    path: str
    content: str


class DockerArtifacts(BaseModel):
    dockerfile: str
    dockerignore: str
    build_script: str = ""


class AzdoPipeline(BaseModel):
    pipeline_yaml: str = Field(description="Content of azure-pipelines.yml")
    variable_groups: List[GeneratedFile] = Field(default_factory=list)
    service_connections: List[GeneratedFile] = Field(default_factory=list)


class AmlAsset(BaseModel):
    path: str = Field(description="Relative path, e.g. aml/environment.yml")
    content: str


class ReviewResult(BaseModel):
    approved: bool
    critique: str = ""
    suggestions: List[str] = Field(default_factory=list)


# --- Pipeline repair --------------------------------------------------------

class FailureAnalysis(BaseModel):
    failed_step: str = ""
    root_cause: str
    category: Literal[
        "dependency", "dockerfile", "pipeline_yaml", "aml_asset",
        "code", "auth", "infrastructure", "unknown",
    ] = "unknown"
    confidence: float = 0.5


class FileFix(BaseModel):
    file_path: str
    new_content: str
    explanation: str = ""


class FixBundle(BaseModel):
    fixes: List[FileFix] = Field(default_factory=list)
    summary: str = ""
    retriable: bool = True


class PullRequestDescription(BaseModel):
    title: str
    description: str
