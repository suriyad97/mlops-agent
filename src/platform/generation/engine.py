"""Generation engine: template retrieval -> Jinja2 render -> LLM adaptation -> working tree.

80% template / 20% LLM: structure comes from the curated library; one bounded
LLM pass per component specializes repo-specific details. If adaptation fails
validation, the pure template render is kept. Files are written to the cloned
working tree only — committing is a separate, user-gated stage.
"""
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import BaseModel, Field

from src.llm.factory import get_provider
from src.llm.prompts.base import MLOPS_EXPERT_PREAMBLE
from src.shared.logging import get_logger
from src.tools.file_tools import write_file
from src.tools.validation_tools import validate_yaml

logger = get_logger(__name__)
TEMPLATES_DIR = Path(__file__).parent.parent / "templates"


class TemplateFile(BaseModel):
    src: str
    dest: str


class TemplateManifest(BaseModel):
    capability: str
    components: Dict[str, dict]   # component -> {files: [{src, dest}]}

    def files_for(self, component: str) -> List[TemplateFile]:
        spec = self.components.get(component, {})
        return [TemplateFile.model_validate(f) for f in spec.get("files", [])]


class AdaptedFiles(BaseModel):
    files: List[dict] = Field(default_factory=list)   # [{path, content}]


class GeneratedComponent(BaseModel):
    capability: str
    component: str
    files: List[str] = Field(default_factory=list)
    adapted: bool = False
    note: str = ""


class GenerationReport(BaseModel):
    components: List[GeneratedComponent] = Field(default_factory=list)
    written_files: List[str] = Field(default_factory=list)
    summary: str = ""


ADAPT_SYSTEM = MLOPS_EXPERT_PREAMBLE + """
You are adapting enterprise template files to a specific repository. Keep the
template's structure, stages, and security posture EXACTLY — only specialize
repo-specific details (paths to real scripts, real input names, column names,
metric names) using the provided repository context. Return every file,
complete. Do not add new files. Do not remove approval gates or stages.
"""


def load_manifest(capability: str) -> Optional[TemplateManifest]:
    manifest_path = TEMPLATES_DIR / capability / "template.yaml"
    if not manifest_path.exists():
        return None
    return TemplateManifest.model_validate(yaml.safe_load(manifest_path.read_text(encoding="utf-8")))


def render_component(capability: str, component: str, params: dict) -> Dict[str, str]:
    """Jinja2-render a component's files. Returns {dest: content}."""
    manifest = load_manifest(capability)
    if not manifest:
        return {}
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR / capability)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )
    rendered: Dict[str, str] = {}
    for tf in manifest.files_for(component):
        rendered[tf.dest] = env.get_template(tf.src).render(**params)
    return rendered


def _validate_files(files: Dict[str, str]) -> str:
    errors = []
    for dest, content in files.items():
        if dest.endswith((".yml", ".yaml")):
            ok, err = validate_yaml(content)
            if not ok:
                errors.append(f"{dest}: {err}")
        if not content.strip():
            errors.append(f"{dest}: empty content")
    return "; ".join(errors)


def _adapt(rendered: Dict[str, str], repo_context: str) -> Optional[Dict[str, str]]:
    """One LLM pass to specialize the rendered files; None when rejected."""
    provider = get_provider()
    blocks = "\n".join(f"--- {dest} ---\n{content}" for dest, content in rendered.items())
    try:
        result = provider.complete_structured(
            AdaptedFiles,
            system=ADAPT_SYSTEM,
            user=f"Repository context:\n{repo_context}\n\nTemplate-rendered files to adapt:\n{blocks}",
        )
    except Exception as exc:
        logger.warning("adaptation failed, keeping template render: %s", exc)
        return None
    adapted = {f["path"]: f["content"] for f in result.files if f.get("path") in rendered}
    if set(adapted) != set(rendered):
        logger.warning("adaptation dropped files, keeping template render")
        return None
    if _validate_files(adapted):
        logger.warning("adaptation failed validation, keeping template render")
        return None
    return adapted


def build_params(project_name: str, profile: dict, settings) -> dict:
    safe = project_name.replace("_", "-").lower()
    return {
        "project_name": project_name,
        "safe_name": safe,
        "model_name": f"{safe}-model",
        "environment_name": settings.aml_environment_name or f"{safe}-env",
        "compute_target": settings.aml_compute_target or "serverless",
        "experiment_name": settings.aml_experiment_name or safe,
        "endpoint_name": f"{safe}-endpoint",
        "default_branch": "main",
        "project_type": profile.get("project_type", "custom"),
        "target_variable": profile.get("target_variable", ""),
        "metrics": profile.get("metrics", []),
    }


def generate(
    repo_path: str,
    gap_items: List[dict],
    params: dict,
    repo_context: str = "",
    adapt: bool = True,
    on_progress=None,          # optional callable(msg: str) — called after each component
) -> GenerationReport:
    report = GenerationReport()
    for item in gap_items:
        capability, component = item["capability"], item["component"]
        rendered = render_component(capability, component, params)
        if not rendered:
            report.components.append(GeneratedComponent(
                capability=capability, component=component,
                note="no template available — needs LLM-only generation",
            ))
            continue

        static_error = _validate_files(rendered)
        if static_error:
            report.components.append(GeneratedComponent(
                capability=capability, component=component,
                note=f"template render invalid: {static_error}",
            ))
            continue

        final_files, adapted = rendered, False
        if adapt and repo_context:
            adapted_files = _adapt(rendered, repo_context)
            if adapted_files:
                final_files, adapted = adapted_files, True

        written = [write_file(repo_path, dest, content) for dest, content in final_files.items()]
        report.components.append(GeneratedComponent(
            capability=capability, component=component, files=written, adapted=adapted,
        ))
        report.written_files.extend(written)
        if on_progress:
            status = "adapted" if adapted else "rendered"
            on_progress(f"✓ {capability}/{component} — {len(written)} file(s) written [{status}]")

    generated = sum(1 for c in report.components if c.files)
    report.summary = (
        f"{generated}/{len(report.components)} components generated "
        f"({len(report.written_files)} files written to the working tree, NOT committed)"
    )
    return report
