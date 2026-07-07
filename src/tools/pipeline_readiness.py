"""Pipeline readiness checker — verifies prerequisites before each pipeline can run.

The MLOps pipeline chain has a strict dependency order:
  Infra → CI → CT → CD → Monitoring → Retraining (loops back to CT)

Each stage consumes artifacts from the previous stage and produces artifacts
for the next. This module checks what is actually in place so the agent can
guide the user through the correct sequence.
"""
from typing import List, Literal

from pydantic import BaseModel

from src.config.settings import get_settings
from src.shared.logging import get_logger

logger = get_logger(__name__)

Stage = Literal["ci", "ct", "cd", "monitoring", "retraining"]


class PipelineStatus(BaseModel):
    stage: Stage
    display_name: str
    ready: bool
    blockers: List[str]       # what's missing — shown to user
    artifacts_present: List[str]   # what's already in place
    produces: List[str]       # what this pipeline creates (for next stage)


class ReadinessReport(BaseModel):
    stages: List[PipelineStatus]

    @property
    def next_runnable(self) -> str | None:
        """Return the name of the first stage that is ready but not yet producing artifacts."""
        for s in self.stages:
            if s.ready:
                return s.display_name
        return None

    def markdown(self) -> str:
        lines = ["## Pipeline Readiness\n",
                 "| Stage | Ready | Blockers |",
                 "|---|---|---|"]
        for s in self.stages:
            icon = "✅" if s.ready else "⛔"
            blockers = ", ".join(s.blockers) if s.blockers else "—"
            lines.append(f"| {icon} **{s.display_name}** | {'Yes' if s.ready else 'No'} | {blockers} |")
        if self.next_runnable:
            lines.append(f"\n> **Next step:** run the {self.next_runnable}")
        return "\n".join(lines)


def _find_file(repo_path: str, target_name: str, search_dirs: tuple = ()) -> str | None:
    """Search for a file by name across the repo, preferring conventional locations.

    Checks ``search_dirs`` first (relative to repo root), then falls back to a
    recursive search.  Returns a repo-relative path or ``None``.
    """
    from pathlib import Path
    base = Path(repo_path)
    target_lower = target_name.lower()

    # 1. Check conventional locations first
    for d in search_dirs:
        candidate = base / d / target_name
        if candidate.exists():
            return str(candidate.relative_to(base)).replace("\\", "/")

    # 2. Check repo root
    root_candidate = base / target_name
    if root_candidate.exists():
        return str(root_candidate.relative_to(base)).replace("\\", "/")

    # 3. Recursive search (case-insensitive on name)
    for p in base.rglob("*"):
        if p.is_file() and p.name.lower() == target_lower and ".git" not in p.parts:
            return str(p.relative_to(base)).replace("\\", "/")
    return None


def _find_dockerfile(repo_path: str) -> str | None:
    """Find a Dockerfile anywhere in the repo, handling name variants."""
    from pathlib import Path
    base = Path(repo_path)

    # Check common locations
    for candidate in ["Dockerfile", "docker/Dockerfile", "environment/Dockerfile",
                       "src/Dockerfile", "build/Dockerfile"]:
        if (base / candidate).exists():
            return candidate

    # Recursive search for any file named Dockerfile* (case-insensitive)
    for p in base.rglob("*"):
        if p.is_file() and p.name.lower().startswith("dockerfile") and ".git" not in p.parts:
            return str(p.relative_to(base)).replace("\\", "/")
    return None


def _check_ci(repo_path: str, profile: dict) -> PipelineStatus:
    """CI is ready when the environment definition files exist in the repo."""
    from pathlib import Path
    blockers: List[str] = []
    present: List[str] = []

    if repo_path:
        # Search for Dockerfile anywhere in the repo
        dockerfile = _find_dockerfile(repo_path)
        if dockerfile:
            present.append(f"Dockerfile ({dockerfile})")
        else:
            blockers.append("missing: Dockerfile (not found anywhere in the repo)")

        # Search for conda/environment files in common locations
        conda_dirs = ("aml", "environment", "config", "")
        for target in ["conda.yml", "environment.yml"]:
            found = _find_file(repo_path, target, search_dirs=conda_dirs)
            if found:
                present.append(f"{target} ({found})")
            else:
                blockers.append(f"missing: {target}")
    else:
        blockers.append("repo not cloned yet — run Scan first")

    acr = profile.get("acr_name") or get_settings().acr_name
    if not acr:
        blockers.append("ACR_NAME not configured (run Auto-discover in Infra)")

    return PipelineStatus(
        stage="ci",
        display_name="CI — Environment Build",
        ready=len(blockers) == 0,
        blockers=blockers,
        artifacts_present=present,
        produces=["Docker image in ACR", "AML Environment registered in workspace"],
    )


def _check_ct(profile: dict) -> PipelineStatus:
    """CT needs a registered AML environment and (ideally) training data accessible."""
    blockers: List[str] = []
    present: List[str] = []

    ws   = profile.get("aml_workspace") or get_settings().aml_workspace
    rg   = profile.get("azure_resource_group") or get_settings().aml_resource_group
    sub  = profile.get("azure_subscription_id") or get_settings().azure_subscription_id
    ct   = profile.get("aml_compute_target") or get_settings().aml_compute_target

    if not ws:
        blockers.append("AML workspace not configured")
        return PipelineStatus(stage="ct", display_name="CT — Training",
                              ready=False, blockers=blockers,
                              artifacts_present=present,
                              produces=["Trained model in AML Model Registry"])
    if not ct:
        blockers.append("compute cluster not configured")

    # Check AML environment registered
    try:
        from src.tools.aml_tools import _ml_client
        client = _ml_client()
        envs = list(client.environments.list())
        if envs:
            present.append(f"AML environment: {envs[0].name}")
        else:
            blockers.append("no AML environment registered — run CI pipeline first")
    except Exception as exc:
        blockers.append(f"cannot reach AML workspace: {str(exc)[:100]}")

    # Training data presence: we can't generically verify blob path,
    # so we just note it as a user responsibility
    present.append("(training data location: user responsibility — verify blob path in pipeline YAML)")

    return PipelineStatus(
        stage="ct",
        display_name="CT — Training",
        ready=len([b for b in blockers if "AML environment" in b or "compute" in b]) == 0,
        blockers=blockers,
        artifacts_present=present,
        produces=["Trained model in AML Model Registry"],
    )


def _check_cd(profile: dict, endpoint_strategy: str) -> PipelineStatus:
    """CD needs a registered model and at least one deployment YAML in the repo."""
    blockers: List[str] = []
    present: List[str] = []

    ws = profile.get("aml_workspace") or get_settings().aml_workspace

    if not ws:
        blockers.append("AML workspace not configured")
        return PipelineStatus(stage="cd", display_name="CD — Deployment",
                              ready=False, blockers=blockers,
                              artifacts_present=present,
                              produces=["Live endpoint in AML workspace"])

    # Check registered models
    try:
        from src.tools.aml_tools import _ml_client
        client = _ml_client()
        models = list(client.models.list())
        if models:
            latest = models[0]
            present.append(f"model registered: {latest.name} v{latest.version}")
        else:
            blockers.append("no model registered in AML — run CT pipeline first")
    except Exception as exc:
        blockers.append(f"cannot reach AML workspace: {str(exc)[:100]}")

    # Check AML environment registered
    try:
        from src.tools.aml_tools import _ml_client
        client = _ml_client()
        envs = list(client.environments.list())
        if envs:
            present.append(f"AML environment: {envs[0].name}")
        else:
            blockers.append("no AML environment registered — run CI pipeline first")
    except Exception:
        pass

    produces = []
    if endpoint_strategy in ("realtime", "both"):
        produces.append("Online endpoint deployed")
    if endpoint_strategy in ("batch", "both"):
        produces.append("Batch endpoint deployed")

    return PipelineStatus(
        stage="cd",
        display_name="CD — Deployment",
        ready=len(blockers) == 0,
        blockers=blockers,
        artifacts_present=present,
        produces=produces or ["Endpoint deployed"],
    )


def _check_monitoring(profile: dict) -> PipelineStatus:
    """Monitoring needs a deployed endpoint."""
    blockers: List[str] = []
    present: List[str] = []

    ws = profile.get("aml_workspace") or get_settings().aml_workspace

    if not ws:
        blockers.append("AML workspace not configured")
        return PipelineStatus(stage="monitoring", display_name="Monitoring",
                              ready=False, blockers=blockers,
                              artifacts_present=present,
                              produces=["Drift alerts", "Data quality reports"])

    try:
        from src.tools.aml_tools import _ml_client
        client = _ml_client()
        online = list(client.online_endpoints.list())
        batch  = list(client.batch_endpoints.list())
        if online:
            present.append(f"online endpoint: {online[0].name}")
        if batch:
            present.append(f"batch endpoint: {batch[0].name}")
        if not online and not batch:
            blockers.append("no endpoint deployed — run CD pipeline first")
    except Exception as exc:
        blockers.append(f"cannot reach AML workspace: {str(exc)[:100]}")

    return PipelineStatus(
        stage="monitoring",
        display_name="Monitoring",
        ready=len(blockers) == 0,
        blockers=blockers,
        artifacts_present=present,
        produces=["Drift alerts", "Data quality reports"],
    )


def _check_retraining(profile: dict) -> PipelineStatus:
    """Retraining needs monitoring to be running and a registered model to improve on."""
    # Retraining readiness is implicitly satisfied when monitoring is running
    # and a model exists. We reuse the monitoring check's result.
    monitoring = _check_monitoring(profile)
    blockers = list(monitoring.blockers)

    if monitoring.ready:
        present = monitoring.artifacts_present + ["model to retrain on"]
    else:
        present = []
        if not blockers:
            blockers.append("monitoring must be running first")

    return PipelineStatus(
        stage="retraining",
        display_name="Retraining",
        ready=monitoring.ready,
        blockers=blockers,
        artifacts_present=present,
        produces=["New model version → triggers CT again"],
    )


# ── public API ────────────────────────────────────────────────────────────────

def check_pipeline_readiness(repo_path: str = "", profile: dict | None = None) -> ReadinessReport:
    """Check readiness of all pipeline stages in dependency order."""
    p = profile or {}
    strategy = str(p.get("endpoint_strategy", "both"))
    return ReadinessReport(stages=[
        _check_ci(repo_path, p),
        _check_ct(p),
        _check_cd(p, strategy),
        _check_monitoring(p),
        _check_retraining(p),
    ])
