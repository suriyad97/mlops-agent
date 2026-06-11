"""Tiered validation framework (R2: validate locally, commit late).

T1 static     — YAML/Dockerfile/pipeline rules (pure code)
T2 local_exec — python byte-compilation of generated .py files
T3 aml_schema — azure-ai-ml client-side loaders (schema-correct without submission;
                actual dev-workspace submissions stay an explicit chat action)
T4 azdo_preview — AzDO previewRun: server-side parse/expand, zero runs created

Each tier returns per-target failures; tiers run in order and all are reported.
"""
import py_compile
import tempfile
from pathlib import Path
from typing import Callable, Dict, List

from pydantic import BaseModel, Field

from src.shared.logging import get_logger
from src.tools.validation_tools import (
    validate_aml_asset_yaml,
    validate_azdo_pipeline_yaml,
    validate_dockerfile,
    validate_yaml,
)

logger = get_logger(__name__)


class Failure(BaseModel):
    target: str
    error: str


class TierResult(BaseModel):
    tier: str
    passed: bool
    failures: List[Failure] = Field(default_factory=list)
    setup_required: List[str] = Field(default_factory=list)   # human org-setup tasks, not file defects
    skipped: bool = False
    note: str = ""


class ValidationReport(BaseModel):
    tiers: List[TierResult] = Field(default_factory=list)
    passed: bool = False
    healed_files: List[str] = Field(default_factory=list)

    def failures_text(self) -> str:
        return "\n".join(
            f"[{t.tier}] {f.target}: {f.error}"
            for t in self.tiers for f in t.failures
        )


def _is_azdo(path: str) -> bool:
    return path.startswith("azdopipelines/") or path == "azure-pipelines.yml"


def _is_aml(path: str) -> bool:
    return path.startswith("aml/") and path.endswith((".yml", ".yaml"))


def tier_static(repo_path: str, files: List[str]) -> TierResult:
    failures = []
    for rel in files:
        content = (Path(repo_path) / rel).read_text(encoding="utf-8", errors="ignore")
        if _is_azdo(rel):
            ok, err = validate_azdo_pipeline_yaml(content)
        elif _is_aml(rel) and "conda" not in rel:
            ok, err = validate_aml_asset_yaml(content)
        elif rel.endswith((".yml", ".yaml")):
            ok, err = validate_yaml(content)
        elif rel.endswith("Dockerfile") or Path(rel).name == "Dockerfile":
            ok, err = validate_dockerfile(content)
        else:
            continue
        if not ok:
            failures.append(Failure(target=rel, error=err))
    return TierResult(tier="static", passed=not failures, failures=failures)


def tier_local_exec(repo_path: str, files: List[str]) -> TierResult:
    failures = []
    for rel in files:
        if not rel.endswith(".py"):
            continue
        try:
            with tempfile.NamedTemporaryFile(suffix=".pyc", delete=True) as tmp:
                py_compile.compile(str(Path(repo_path) / rel), cfile=tmp.name, doraise=True)
        except py_compile.PyCompileError as exc:
            failures.append(Failure(target=rel, error=str(exc.msg)[:500]))
        except OSError:
            # windows cannot reopen the NamedTemporaryFile — compile without cfile
            try:
                py_compile.compile(str(Path(repo_path) / rel), doraise=True)
            except py_compile.PyCompileError as exc:
                failures.append(Failure(target=rel, error=str(exc.msg)[:500]))
    return TierResult(tier="local_exec", passed=not failures, failures=failures)


_AML_LOADERS: Dict[str, str] = {
    "commandComponent": "load_component",
    "pipelineJob": "load_job",
    "environment.schema": "load_environment",
    "managedOnlineEndpoint": "load_online_endpoint",
    "managedOnlineDeployment": "load_online_deployment",
    "batchEndpoint": "load_batch_endpoint",
    "modelBatchDeployment": "load_batch_deployment",
}


def tier_aml_schema(repo_path: str, files: List[str]) -> TierResult:
    targets = [f for f in files if _is_aml(f) and "conda" not in f]
    if not targets:
        return TierResult(tier="aml_schema", passed=True, note="no AML assets to validate")
    try:
        from azure.ai import ml as aml_module
    except ImportError:
        return TierResult(tier="aml_schema", passed=True, skipped=True,
                          note="azure-ai-ml not installed")
    failures = []
    for rel in targets:
        full = Path(repo_path) / rel
        content = full.read_text(encoding="utf-8", errors="ignore")
        loader_name = next((fn for hint, fn in _AML_LOADERS.items() if hint in content), None)
        if not loader_name:
            continue
        loader: Callable = getattr(aml_module, loader_name, None)
        if loader is None:
            continue
        try:
            loader(source=str(full))
        except Exception as exc:
            failures.append(Failure(target=rel, error=str(exc)[:500]))
    return TierResult(tier="aml_schema", passed=not failures, failures=failures)


def tier_azdo_preview(repo_path: str, files: List[str], branch: str = "main") -> TierResult:
    targets = [f for f in files if _is_azdo(f)]
    if not targets:
        return TierResult(tier="azdo_preview", passed=True, note="no AzDO pipelines to validate")
    from pathlib import PurePosixPath

    from src.config.settings import get_settings
    from src.shared.exceptions import ConfigurationError, ToolError
    from src.tools.azdo_tools import create_azure_pipeline, preview_pipeline_yaml

    settings = get_settings()
    if not settings.azdo_org_url or not settings.azdo_project:
        return TierResult(tier="azdo_preview", passed=True, skipped=True,
                          note="AzDO not configured")
    try:
        repo_name = PurePosixPath(Path(repo_path).name).name
        pipeline_id = create_azure_pipeline(name=f"{repo_name}-remediation", repo_name=repo_name)
    except (ConfigurationError, ToolError) as exc:
        return TierResult(tier="azdo_preview", passed=True, skipped=True,
                          note=f"preview unavailable: {str(exc)[:200]}")
    # Org-state errors the FILES cannot fix: surfaced as setup tasks, not failures.
    # (AzDO Environments with approval checks must be created once by a human — R5.)
    _SETUP_PATTERNS = (
        "could not be found. The environment does not exist",
        "has not been authorized for use",
        "service connection",
    )
    failures, setup_required = [], []
    for rel in targets:
        content = (Path(repo_path) / rel).read_text(encoding="utf-8", errors="ignore")
        result = preview_pipeline_yaml(pipeline_id, content, branch=branch)
        if result["valid"]:
            continue
        if any(p in result["error"] for p in _SETUP_PATTERNS):
            setup_required.append(f"{rel}: {result['error'][:300]}")
        else:
            failures.append(Failure(target=rel, error=result["error"]))
    note = "AzDO org setup required (see .azuredevops/environments.md)" if setup_required else ""
    return TierResult(tier="azdo_preview", passed=not failures, failures=failures,
                      setup_required=setup_required, note=note)


def run_validation(repo_path: str, files: List[str], branch: str = "main") -> ValidationReport:
    report = ValidationReport()
    report.tiers.append(tier_static(repo_path, files))
    report.tiers.append(tier_local_exec(repo_path, files))
    report.tiers.append(tier_aml_schema(repo_path, files))
    report.tiers.append(tier_azdo_preview(repo_path, files, branch=branch))
    report.passed = all(t.passed for t in report.tiers)
    return report
