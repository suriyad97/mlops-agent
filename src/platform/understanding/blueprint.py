"""Repository Blueprint: deterministic graph facts + LLM-inferred ML profile."""
from pathlib import Path
from typing import List, Literal, Optional, Tuple

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
    train_script: str = "src/train.py"
    score_script: str = "src/score.py"
    batch_score_script: str = "src/batch_score.py"
    drift_script: str = "src/monitoring/detect_drift.py"
    evaluate_thresholds_script: str = "src/monitoring/evaluate_thresholds.py"


class RepositoryBlueprint(BaseModel):
    profile: MlProfile
    entry_points: List[dict] = Field(default_factory=list)
    aml_assets: List[str] = Field(default_factory=list)
    azdo_pipelines: List[str] = Field(default_factory=list)
    github_workflows: List[str] = Field(default_factory=list)
    dockerfiles: List[str] = Field(default_factory=list)
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
NOTE: train_script, score_script, and batch_score_script are set deterministically
after this call — do not try to infer them; leave them at their defaults.
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


_TRAIN_CALL_MARKERS = frozenset({"fit", "train", "train_step", "training_step", "fit_transform", "partial_fit"})


def _find_script_paths(graph: nx.DiGraph) -> Tuple[str, str, str, str, str]:
    """Deterministically detect train/score/batch_score entry points from graph structure.

    Returns (train_script, score_script, batch_score_script) as repo-relative paths.
    Falls back to conventional defaults when evidence is insufficient.
    """
    score_candidates: List[str] = []
    batch_score_candidates: List[str] = []
    train_candidates: List[str] = []
    drift_candidates: List[str] = []

    _DRIFT_LIBS = frozenset({"evidently", "alibi", "alibi_detect", "nannyml", "whylogs", "great_expectations"})
    _DRIFT_NAME_HINTS = frozenset({"drift", "monitor", "detect"})

    for node, attrs in graph.nodes(data=True):
        if not node.startswith("file:"):
            continue
        rel = attrs.get("path", "")
        if not rel.endswith(".py"):
            continue

        func_names: set = set()
        all_calls: set = set()
        func_sigs: dict = {}
        for n2, a2 in graph.nodes(data=True):
            if n2.startswith(f"def:{rel}:"):
                name = a2.get("name", "")
                func_names.add(name)
                all_calls.update(a2.get("calls", []))
                func_sigs[name] = a2.get("signature", "")

        # AML online scoring contract: defines init() + run()
        if "init" in func_names and "run" in func_names:
            score_candidates.append(rel)

        # AML batch scoring contract: defines run() with mini_batch in signature
        if "run" in func_names and "mini_batch" in func_sigs.get("run", ""):
            batch_score_candidates.append(rel)

        # Training script: entry point with known training function calls
        if attrs.get("entry_point") and all_calls & _TRAIN_CALL_MARKERS:
            train_candidates.append(rel)

        # Drift/monitoring scripts: entry points whose name or imports suggest monitoring
        stem = Path(rel).stem.lower()
        if attrs.get("entry_point") and (
            any(h in stem for h in _DRIFT_NAME_HINTS)
            or all_calls & _DRIFT_LIBS
        ):
            drift_candidates.append(rel)

    def _prefer(candidates: List[str], preferred: List[str], default: str) -> str:
        for name in preferred:
            for c in candidates:
                if Path(c).name == name:
                    return c
        return candidates[0] if candidates else default

    train_script = _prefer(train_candidates, ["train.py"], "src/train.py")
    score_script = _prefer(score_candidates, ["score.py"], "src/score.py")
    batch_score_script = _prefer(batch_score_candidates, ["batch_score.py"], "src/batch_score.py")
    drift_script = _prefer(drift_candidates, ["detect_drift.py", "monitor.py"], "src/monitoring/detect_drift.py")
    evaluate_thresholds_script = _prefer(
        drift_candidates, ["evaluate_thresholds.py", "thresholds.py"], "src/monitoring/evaluate_thresholds.py"
    )
    return train_script, score_script, batch_score_script, drift_script, evaluate_thresholds_script


def extract_blueprint(repo_path: str, graph: nx.DiGraph) -> RepositoryBlueprint:
    provider = get_provider()
    digest = graph_digest(graph, max_chars=5000)
    excerpts = _profile_excerpts(repo_path, graph)

    profile = provider.complete_structured(
        MlProfile,
        system=PROFILE_SYSTEM,
        user=f"Knowledge graph digest:\n{digest}\n\nCode excerpts:{excerpts}",
    )

    # Override script paths with deterministic graph evidence — more reliable than LLM inference
    train_script, score_script, batch_score_script, drift_script, evaluate_thresholds_script = _find_script_paths(graph)
    profile.train_script = train_script
    profile.score_script = score_script
    profile.batch_score_script = batch_score_script
    profile.drift_script = drift_script
    profile.evaluate_thresholds_script = evaluate_thresholds_script

    return RepositoryBlueprint(
        profile=profile,
        entry_points=entry_points(graph),
        aml_assets=assets_by_role(graph, "aml_asset"),
        azdo_pipelines=assets_by_role(graph, "azdo_pipeline"),
        github_workflows=assets_by_role(graph, "github_workflow"),
        dockerfiles=assets_by_role(graph, "dockerfile"),
        graph_stats=graph_stats(graph),
    )
