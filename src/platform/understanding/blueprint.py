"""Repository Blueprint: deterministic graph facts + LLM-inferred ML profile."""
from typing import List, Literal, Optional

import networkx as nx
from pydantic import BaseModel, Field

from src.llm.factory import get_provider
from src.llm.prompts.base import MLOPS_EXPERT_PREAMBLE
from src.platform.understanding.retrieval import (
    assets_by_role,
    entry_points,
    find_evidence,
    graph_digest,
    graph_stats,
)
from src.tools.file_tools import read_file


class MlProfile(BaseModel):
    project_type: Literal[
        "binary_classification", "multiclass_classification", "regression",
        "forecasting", "nlp", "computer_vision", "custom",
    ] = "custom"
    target_variable: str = ""
    metrics: List[str] = Field(default_factory=list)
    endpoint_strategy: Literal["realtime", "batch", "both", "none"] = "none"
    drift_kinds: List[str] = Field(default_factory=list)
    confidence: float = 0.5
    reasoning: str = ""


class RepositoryBlueprint(BaseModel):
    profile: MlProfile
    entry_points: List[dict] = Field(default_factory=list)
    aml_assets: List[str] = Field(default_factory=list)
    azdo_pipelines: List[str] = Field(default_factory=list)
    github_workflows: List[str] = Field(default_factory=list)
    graph_stats: dict = Field(default_factory=dict)


PROFILE_SYSTEM = MLOPS_EXPERT_PREAMBLE + """
Infer the ML profile of this repository from the knowledge-graph digest and
code excerpts. Ground every field in the evidence:
- project_type from the modeling code (estimators, losses, label handling)
- target_variable from dataset column usage in training/evaluation code
- metrics from evaluation calls (accuracy_score, f1, rmse, ...)
- endpoint_strategy from deployment assets (online endpoint = realtime,
  batch endpoint = batch; both if both; none if no deployment)
- drift_kinds from monitoring assets/code (data / prediction / concept).
  Concept drift requires actual label availability — only include it if
  labels are clearly available post-deployment.
Set confidence (0-1) and a 1-2 sentence reasoning.
"""

_EVIDENCE_INDICATORS = [
    "fit", "predict", "train", "evaluate", "accuracy", "f1", "rmse", "mae",
    "roc_auc", "score", "metric", "target", "label", "drift", "monitor",
]


def _profile_excerpts(repo_path: str, graph: nx.DiGraph, max_files: int = 3) -> str:
    """Source excerpts for the files with the densest ML evidence."""
    hits = find_evidence(graph, _EVIDENCE_INDICATORS)
    seen: List[str] = []
    for hit in hits:
        file = hit.get("file")
        if file and file.endswith(".py") and file not in seen:
            seen.append(file)
        if len(seen) >= max_files:
            break
    excerpts = ""
    for file in seen:
        try:
            excerpts += f"\n--- {file} ---\n{read_file(repo_path, file, max_bytes=2500)}"
        except Exception:
            continue
    return excerpts


def extract_blueprint(repo_path: str, graph: nx.DiGraph) -> RepositoryBlueprint:
    provider = get_provider()
    digest = graph_digest(graph, max_chars=5000)
    excerpts = _profile_excerpts(repo_path, graph)

    profile = provider.complete_structured(
        MlProfile,
        system=PROFILE_SYSTEM,
        user=f"Knowledge graph digest:\n{digest}\n\nCode excerpts:{excerpts}",
    )

    return RepositoryBlueprint(
        profile=profile,
        entry_points=entry_points(graph),
        aml_assets=assets_by_role(graph, "aml_asset"),
        azdo_pipelines=assets_by_role(graph, "azdo_pipeline"),
        github_workflows=assets_by_role(graph, "github_workflow"),
        graph_stats=graph_stats(graph),
    )
