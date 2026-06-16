"""Repository contract manifest — the user-confirmed checklist of what ML code exists.

The standard MLOps block (azdopipelines/, MLpipelines/, aml/) is ALWAYS generated
in full for the chosen endpoint strategy. The only variable is the user's ML code:
for each pipeline-bound stage, the contract declares whether the logic exists, where,
and whether it satisfies the thin execution contract (CLI args / AML init+run).

propose_contract() pre-fills the checklist from deterministic graph evidence; the user
then confirms or corrects it (the human is ground truth). The resolved manifest drives
generation: each stage becomes wired / adapter / scaffold.
"""
from pathlib import Path
from typing import Dict, List, Literal, Set

import networkx as nx
from pydantic import BaseModel, Field

# The seven pipeline-bound stages. Everything else in the standard block
# (AML YAMLs, AzDO pipelines, Dockerfile, register.yml) is always generated
# standard and is NOT governed by the checklist.
#   (stage key, display name, standard path, owning capability)
STAGE_REGISTRY: List[tuple] = [
    ("training",         "Training",             "src/train.py",                          "training"),
    ("evaluation",       "Evaluation / metrics", "src/evaluate.py",                       "training"),
    ("scoring_realtime", "Realtime scoring",     "src/score.py",                          "realtime_deployment"),
    ("scoring_batch",    "Batch scoring",        "src/batch_score.py",                    "batch_deployment"),
    ("drift",            "Drift detection",      "src/monitoring/detect_drift.py",        "monitoring"),
    ("thresholds",       "Threshold evaluation", "src/monitoring/evaluate_thresholds.py", "retraining"),
    ("rollback",         "Model rollback",       "src/rollback/rollback_model.py",        "rollback"),
]

# Stages relevant per endpoint strategy (mirrors evaluator._STRATEGY_CAPS).
_STRATEGY_STAGES: Dict[str, Set[str] | None] = {
    "realtime": {"training", "evaluation", "scoring_realtime", "drift", "thresholds", "rollback"},
    "batch":    {"training", "evaluation", "scoring_batch", "drift", "thresholds", "rollback"},
    "both":     None,   # all stages
    "none":     {"training", "evaluation"},
}

Mode = Literal["wired", "adapter", "scaffold"]


class ContractStage(BaseModel):
    stage: str
    display_name: str
    standard_path: str
    capability: str
    present: bool = False           # the logic exists somewhere in the repo
    detected_path: str = ""         # where the scan found it (may differ from standard_path)
    detected_symbol: str = ""       # best-guess function/class to wrap in an adapter
    meets_contract: bool = False    # compliant entry point (CLI args / init+run)
    mode: Mode = "scaffold"         # resolved generation action
    note: str = ""

    def resolve(self) -> "ContractStage":
        """Derive the generation mode from the (possibly user-edited) declarations.

        - wired:    compliant entry point exists — pipeline points at it (detected_path)
                    via the script-path param; nothing is generated.
        - adapter:  logic exists but not contract-compliant — generate a thin adapter at
                    the standard path that imports the user's symbol (their code untouched).
        - scaffold: nothing exists — generate a TODO scaffold at the standard path.
        """
        if not self.present:
            mode: Mode = "scaffold"
        elif self.meets_contract:
            mode = "wired"
        else:
            mode = "adapter"
        return self.model_copy(update={"mode": mode})

    def pipeline_path(self) -> str:
        """The path the generated pipeline should call for this stage."""
        return self.detected_path if self.mode == "wired" else self.standard_path


class ContractManifest(BaseModel):
    stages: List[ContractStage] = Field(default_factory=list)
    # dev-time-only declarations — acknowledged, never turned into pipeline stages
    has_eda: bool = False
    has_shap: bool = False
    has_feature_engineering: bool = False

    def stage(self, key: str) -> ContractStage | None:
        return next((s for s in self.stages if s.stage == key), None)

    def for_strategy(self, endpoint_strategy: str) -> "ContractManifest":
        allowed = _STRATEGY_STAGES.get(endpoint_strategy)
        if allowed is None:
            kept = list(self.stages)
        else:
            kept = [s for s in self.stages if s.stage in allowed]
        return self.model_copy(update={"stages": kept})

    def resolved(self) -> "ContractManifest":
        return self.model_copy(update={"stages": [s.resolve() for s in self.stages]})


# ---------------------------------------------------------------------------
# Deterministic pre-fill from the knowledge graph
# ---------------------------------------------------------------------------

class _FileFacts(BaseModel):
    funcs: Set[str] = Field(default_factory=set)
    calls: Set[str] = Field(default_factory=set)
    sigs: Dict[str, str] = Field(default_factory=dict)
    cli_args: List[str] = Field(default_factory=list)
    entry_point: bool = False


_TRAIN_MARKERS = frozenset({"fit", "train", "train_step", "training_step", "fit_transform", "partial_fit"})
_EVAL_MARKERS = frozenset({"evaluate", "score", "accuracy_score", "f1_score", "mean_squared_error",
                           "roc_auc_score", "classification_report", "r2_score"})
_DRIFT_MARKERS = frozenset({"evidently", "alibi", "alibi_detect", "nannyml", "whylogs",
                            "great_expectations", "psi", "wasserstein_distance", "ks_2samp"})


def _collect_facts(graph: nx.DiGraph) -> Dict[str, _FileFacts]:
    facts: Dict[str, _FileFacts] = {}
    for node, attrs in graph.nodes(data=True):
        if node.startswith("file:") and str(attrs.get("path", "")).endswith(".py"):
            facts[attrs["path"]] = _FileFacts(
                cli_args=list(attrs.get("cli_args", [])),
                entry_point=bool(attrs.get("entry_point", False)),
            )
    for node, attrs in graph.nodes(data=True):
        if attrs.get("kind") in ("function", "class"):
            rel = attrs.get("file", "")
            f = facts.get(rel)
            if f is None:
                continue
            name = attrs.get("name", "")
            f.funcs.add(name)
            f.calls.update(attrs.get("calls", []))
            f.sigs[name] = attrs.get("signature", "")
    return facts


def _has_cli(f: _FileFacts) -> bool:
    return bool(f.cli_args) or f.entry_point


def _best_symbol(f: _FileFacts, markers: Set[str]) -> str:
    """Pick the most adapter-worthy function: one whose body calls a marker, else any def."""
    for name, _sig in f.sigs.items():
        if name in ("main", "__init__"):
            continue
        return name
    return ""


def _detect_stage(key: str, std_path: str, facts: Dict[str, _FileFacts]) -> ContractStage:
    """Find the best candidate file + compliance for one stage from graph facts."""
    display = next(d for k, d, _p, _c in STAGE_REGISTRY if k == key)
    capability = next(c for k, _d, _p, c in STAGE_REGISTRY if k == key)
    base = ContractStage(stage=key, display_name=display, standard_path=std_path, capability=capability)

    std_name = Path(std_path).name
    candidates: List[tuple] = []   # (priority, path, meets_contract, symbol)

    for rel, f in facts.items():
        meets = False
        symbol = ""
        if key == "scoring_realtime":
            if "init" in f.funcs and "run" in f.funcs:
                meets, symbol = True, "run"
                candidates.append((0, rel, meets, symbol))
            elif {"predict", "load_model", "serve"} & f.calls or "fastapi" in f.calls:
                candidates.append((2, rel, False, _best_symbol(f, set())))
        elif key == "scoring_batch":
            if "run" in f.funcs and "mini_batch" in f.sigs.get("run", ""):
                meets, symbol = True, "run"
                candidates.append((0, rel, meets, symbol))
            elif "batch" in rel.lower() and ({"predict", "score"} & f.calls):
                candidates.append((2, rel, False, _best_symbol(f, set())))
        elif key == "training":
            if f.entry_point and f.calls & _TRAIN_MARKERS:
                candidates.append((0 if _has_cli(f) else 1, rel, _has_cli(f), _best_symbol(f, _TRAIN_MARKERS)))
            elif f.calls & _TRAIN_MARKERS:
                candidates.append((2, rel, False, _best_symbol(f, _TRAIN_MARKERS)))
        elif key == "evaluation":
            if f.calls & _EVAL_MARKERS:
                candidates.append((1 if _has_cli(f) else 2, rel, _has_cli(f), _best_symbol(f, _EVAL_MARKERS)))
        elif key == "drift":
            stem = Path(rel).stem.lower()
            if ("drift" in stem or "monitor" in stem) or (f.calls & _DRIFT_MARKERS):
                candidates.append((1 if _has_cli(f) else 2, rel, _has_cli(f), _best_symbol(f, _DRIFT_MARKERS)))
        elif key == "thresholds":
            stem = Path(rel).stem.lower()
            if "threshold" in stem or {"threshold", "breach", "exceeds"} & f.calls:
                candidates.append((1 if _has_cli(f) else 2, rel, _has_cli(f), _best_symbol(f, set())))
        elif key == "rollback":
            stem = Path(rel).stem.lower()
            if "rollback" in stem or {"rollback", "revert", "fallback"} & f.calls:
                candidates.append((1, rel, False, _best_symbol(f, set())))

    if not candidates:
        return base

    # Prefer the conventionally-named file, then lowest priority number (best evidence)
    def _key(c: tuple) -> tuple:
        prio, path, _m, _s = c
        return (Path(path).name != std_name, prio)

    candidates.sort(key=_key)
    _prio, path, meets, symbol = candidates[0]
    return base.model_copy(update={
        "present": True,
        "detected_path": path,
        "detected_symbol": symbol,
        "meets_contract": meets,
        "note": ("compliant entry point" if meets else "needs adapter — non-standard entry point"),
    }).resolve()


def propose_contract(graph: nx.DiGraph) -> ContractManifest:
    """Pre-fill the checklist from deterministic graph evidence (user confirms/corrects later)."""
    facts = _collect_facts(graph)
    stages = [_detect_stage(key, std_path, facts) for key, _d, std_path, _c in STAGE_REGISTRY]

    # dev-time-only signals (best-effort, informational)
    all_py = list(facts.keys())
    has_eda = any("eda" in p.lower() or "explor" in p.lower() for p in all_py)
    has_shap = any("shap" in f.calls or "shap" in p.lower() for p, f in facts.items())
    has_fe = any("feature" in p.lower() or {"fit_transform", "StandardScaler", "OneHotEncoder"} & f.calls
                 for p, f in facts.items())

    return ContractManifest(stages=stages, has_eda=has_eda, has_shap=has_shap,
                            has_feature_engineering=has_fe)
