"""Endpoint requirement plan: resolve backward from the endpoint, tag ownership, tick status.

When the user picks an endpoint strategy, the platform must reason from the endpoint
through its full dependency chain and decide WHAT is required and WHO owns each piece —
before generating anything. Three ownership buckets:

  data_scientist  — ML *code logic* (train / score / evaluate / drift / threshold rules)
  platform        — everything that wraps that code (env defs, AML YAMLs, endpoints,
                    CI/CT/CD/monitoring pipelines, model registration, data collection)
  infra           — cloud prerequisites (RG, ACR, workspace, compute, data, connections)

The plan is the gate shown before generation: the DS sees only their rows; the platform
rows are auto-generated; the infra rows point at the prerequisite check.
"""
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from src.platform.evaluation.evaluator import standard_generation_set
from src.platform.understanding.contract import STAGE_REGISTRY, ContractManifest

Owner = Literal["data_scientist", "platform", "infra"]
Status = Literal[
    "present",        # DS code exists and is compliant — reused as-is
    "adapter",        # DS code exists but needs a generated adapter
    "scaffold",       # DS code absent — scaffold generated, DS must implement
    "exists",         # platform file already present at the standard path
    "will_generate",  # platform file will be generated
    "prerequisite",   # infra — verify via the infrastructure check
    "user_action",    # the user must do this (e.g. upload training data)
]

# Contract stage standard_path -> stage key (the DS-owned code deliverables)
_DS_PATH_TO_STAGE = {std: key for key, _disp, std, _cap in STAGE_REGISTRY}

# Stages whose scaffold is model-specific (a stub the DS MUST implement → blocks the endpoint).
# Other code stages (evaluation, drift, thresholds, rollback) ship a COMPLETE platform default,
# so when the DS hasn't written their own they become the platform's job, not a blocker.
_DS_STUB_STAGES = {"training", "scoring_realtime", "scoring_batch"}

# Friendly per-deliverable labels for DS code rows
_DELIVERABLE_LABEL = {
    "src/train.py": "Training code",
    "src/evaluate.py": "Evaluation code",
    "src/score.py": "Scoring code (init/run)",
    "src/batch_score.py": "Batch scoring code (run)",
    "src/monitoring/detect_drift.py": "Drift detection logic",
    "src/monitoring/evaluate_thresholds.py": "Drift-threshold rule",
    "src/rollback/rollback_model.py": "Model rollback logic",
}

_STATUS_ICON = {
    "present": "✅", "adapter": "🔧", "scaffold": "⬜",
    "exists": "✅", "will_generate": "🔲",
    "prerequisite": "☁️", "user_action": "⚠️",
}
_OWNER_ICON = {"data_scientist": "🧑‍🔬", "platform": "🤖", "infra": "☁️"}
_STATUS_LABEL = {
    "present": "reuse your code", "adapter": "adapter wraps your code",
    "scaffold": "you must implement", "exists": "present", "will_generate": "will generate",
    "prerequisite": "verify in Infra check", "user_action": "you must do this",
}


class RequirementItem(BaseModel):
    order: int
    requirement: str
    owner: Owner
    status: Status
    deliverables: List[str] = Field(default_factory=list)
    detail: str = ""


class RequirementPlan(BaseModel):
    endpoint_strategy: str
    items: List[RequirementItem] = Field(default_factory=list)

    @property
    def ds_items(self) -> List[RequirementItem]:
        return [i for i in self.items if i.owner == "data_scientist"]

    @property
    def blocking(self) -> List[RequirementItem]:
        """Items that block a working endpoint until the user acts (DS scaffold or infra/user)."""
        return [i for i in self.items if i.status in ("scaffold", "user_action")]

    def markdown(self) -> str:
        lines = [
            f"## Requirement plan — `{self.endpoint_strategy}` endpoint",
            "",
            "Resolved backward from the endpoint. 🧑‍🔬 = your job (data scientist) · "
            "🤖 = auto-generated (platform) · ☁️ = infra/ops.",
            "",
            "| # | Requirement | Owner | Status |",
            "|---|---|---|---|",
        ]
        for it in self.items:
            owner = f"{_OWNER_ICON[it.owner]} {it.owner.replace('_', ' ')}"
            status = f"{_STATUS_ICON[it.status]} {_STATUS_LABEL[it.status]}"
            detail = f" — {it.detail}" if it.detail else ""
            lines.append(f"| {it.order} | {it.requirement}{detail} | {owner} | {status} |")
        ds = self.ds_items
        block = self.blocking
        lines += [
            "",
            f"**Your job (data scientist): {len(ds)} item(s).** "
            f"The platform generates the other {len(self.items) - len(ds)}.",
        ]
        if block:
            lines.append(
                "⚠️ Blocking before a working endpoint: "
                + "; ".join(f"{b.requirement} ({_STATUS_LABEL[b.status]})" for b in block)
            )
        return "\n".join(lines)


# Human-readable requirement labels per (capability, component)
_REQ_LABEL = {
    ("environment_lifecycle", "container_definition"): "Container / conda definition",
    ("environment_lifecycle", "aml_environment"): "AML environment asset",
    ("environment_lifecycle", "azdo_env_pipeline"): "CI pipeline (build + register environment)",
    ("training", "code_functions"): "Training & evaluation code",
    ("training", "aml_training_assets"): "Training pipeline (AML)",
    ("training", "azdo_orchestration"): "CT pipeline (submit training)",
    ("model_lifecycle", "registration_mechanism"): "Model registration",
    ("model_lifecycle", "promotion_mechanism"): "Model promotion pipeline",
    ("model_lifecycle", "approval_gates"): "Approval gates",
    ("rollback", "endpoint_rollback"): "Rollback CD path (blue/green revert)",
    ("rollback", "pipeline_rollback"): "Rollback validation gates / runbook",
    ("realtime_deployment", "online_endpoint_assets"): "Online endpoint + deployment",
    ("realtime_deployment", "scoring_code"): "Scoring code (init/run)",
    ("realtime_deployment", "azdo_cd_realtime"): "CD pipeline (realtime)",
    ("batch_deployment", "batch_endpoint_assets"): "Batch endpoint + deployment",
    ("batch_deployment", "inference_code"): "Batch scoring code",
    ("batch_deployment", "azdo_cd_batch"): "CD pipeline (batch)",
    ("monitoring", "drift_detection_code"): "Drift detection logic",
    ("monitoring", "monitoring_aml_pipeline"): "Monitoring pipeline (AML)",
    ("monitoring", "azdo_monitoring"): "Monitoring schedule + alerts",
    ("retraining", "trigger_mechanism"): "Drift-threshold rule",
    ("retraining", "retraining_pipeline"): "Retraining pipeline (AML)",
    ("retraining", "azdo_ct"): "CT pipeline (retrain → register → promote)",
}


def _classify_code(path: str, contract: Optional[ContractManifest]) -> tuple[Owner, Status, str]:
    """Owner/status/detail for a code deliverable, following reality:
    - the DS wrote it (wired/adapter) → DS owns it;
    - absent + model-specific stub (train/score) → DS must implement (blocks);
    - absent + a complete platform default exists (eval/drift/thresholds/rollback) → platform's job.
    """
    key = _DS_PATH_TO_STAGE[path]
    st = contract.stage(key) if contract else None
    mode = st.mode if st else "scaffold"
    if mode == "wired":
        return "data_scientist", "present", (f"reuse {st.detected_path}" if st and st.detected_path else "")
    if mode == "adapter":
        return "data_scientist", "adapter", (f"wrap {st.detected_path}" if st and st.detected_path else "")
    # scaffold / absent
    if key in _DS_STUB_STAGES:
        return "data_scientist", "scaffold", "model-specific — you must implement"
    return "platform", "will_generate", "default provided — customize if needed"


def endpoint_requirement_plan(
    repo_path: str,
    endpoint_strategy: str,
    contract: Optional[ContractManifest] = None,
) -> RequirementPlan:
    """Assemble the ownership-tagged, dependency-ordered, ticked requirement plan."""
    base = Path(repo_path) if repo_path else None
    plan_items: List[RequirementItem] = []
    n = 0

    def add(requirement: str, owner: Owner, status: Status,
            deliverables: Optional[List[str]] = None, detail: str = "") -> None:
        nonlocal n
        n += 1
        plan_items.append(RequirementItem(
            order=n, requirement=requirement, owner=owner, status=status,
            deliverables=deliverables or [], detail=detail,
        ))

    # 1. Infra prerequisites first (verify via the infra check; the plan does not call Azure)
    add("Cloud prerequisites (ACR · workspace · compute · ARM connection)", "infra",
        "prerequisite", detail="run the Infra check")

    gen = standard_generation_set(endpoint_strategy)
    training_seen = False
    for item in gen.items:
        cap, comp = item.capability, item.component
        label = _REQ_LABEL.get((cap, comp), f"{cap}.{comp}")

        # DS-code deliverables: one row each, owner/status follow reality (see _classify_code)
        ds_files = [d for d in item.deliverables if d in _DS_PATH_TO_STAGE]
        for d in ds_files:
            owner, status, detail = _classify_code(d, contract)
            add(_DELIVERABLE_LABEL.get(d, Path(d).name), owner, status,
                deliverables=[d], detail=detail)

        # Platform-owned files: one row per component group
        platform_files = [d for d in item.deliverables if d not in _DS_PATH_TO_STAGE]
        if platform_files:
            present = bool(base) and all((base / f).exists() for f in platform_files)
            add(label, "platform", "exists" if present else "will_generate",
                deliverables=platform_files)

        # After the training pipeline, training data must be in blob (user responsibility)
        if cap == "training" and not training_seen:
            training_seen = True
            add("Training data in Blob Storage", "infra", "user_action",
                detail="upload before the CT pipeline runs")

    # Strategy-specific monitoring data source (the chain the user cares about)
    if endpoint_strategy in ("realtime", "both"):
        add("Inference data collection (MDC on the online deployment)", "platform",
            "will_generate", deliverables=["MLpipelines/deployment.yml"],
            detail="logs request inputs/outputs to blob for the monitoring pipeline")
    if endpoint_strategy in ("batch", "both"):
        add("Batch output files as monitoring input", "platform", "will_generate",
            detail="batch predictions.csv feeds the drift pipeline — no collector needed")

    return RequirementPlan(endpoint_strategy=endpoint_strategy, items=plan_items)
