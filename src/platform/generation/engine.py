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
    files: List[dict] = Field(default_factory=list)   # [{path: str, content: str|dict}]


class GeneratedComponent(BaseModel):
    capability: str
    component: str
    files: List[str] = Field(default_factory=list)
    adapted: bool = False
    note: str = ""


class GenerationReport(BaseModel):
    components: List[GeneratedComponent] = Field(default_factory=list)
    written_files: List[str] = Field(default_factory=list)
    created_files: List[str] = Field(default_factory=list)    # did NOT exist before — newly created
    updated_files: List[str] = Field(default_factory=list)    # already existed — overwritten with standard
    adapter_files: List[str] = Field(default_factory=list)    # thin wrappers over user code
    scaffold_files: List[str] = Field(default_factory=list)   # TODO stubs — user must implement
    wired_skipped: List[str] = Field(default_factory=list)    # user code reused as-is (not written)
    superseded_files: List[str] = Field(default_factory=list)  # legacy non-standard pipelines to remove
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
        if not isinstance(content, str):
            errors.append(f"{dest}: non-string content ({type(content).__name__})")
            continue
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

    # Build adapted dict, type-checking content values
    adapted: Dict[str, str] = {}
    for f in result.files:
        path = f.get("path")
        content = f.get("content")
        if path not in rendered:
            continue
        # LLM sometimes returns content as a parsed dict/list instead of a string
        # (i.e. it gave us parsed YAML instead of the YAML text). Convert back to string.
        if isinstance(content, (dict, list)):
            try:
                import yaml as _yaml
                content = _yaml.dump(content, allow_unicode=True, sort_keys=False)
                logger.warning("adaptation returned dict content for %s — converted to YAML string", path)
            except Exception:
                logger.warning("adaptation returned non-string content for %s — rejecting", path)
                return None
        if not isinstance(content, str):
            logger.warning("adaptation returned non-string content for %s (type %s) — rejecting", path, type(content))
            return None
        adapted[path] = content

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
        "acr_name": profile.get("acr_name") or settings.acr_name or "",
        # All script paths are REPO-RELATIVE. Every AML component/deployment uses the
        # repository root as its code root (code: ../ or ../../), so user code anywhere
        # in the tree is reachable and importable by adapters.
        "train_script": profile.get("train_script", "src/train.py"),
        "evaluate_script": profile.get("evaluate_script", "src/evaluate.py"),
        "score_script": profile.get("score_script", "src/score.py"),
        "batch_score_script": profile.get("batch_score_script", "src/batch_score.py"),
        "drift_script": profile.get("drift_script", "src/monitoring/detect_drift.py"),
        "evaluate_thresholds_script": profile.get("evaluate_thresholds_script", "src/monitoring/evaluate_thresholds.py"),
        # Tunable pipeline parameters (user-overridable upfront; baked into generated assets)
        **_pipeline_params(profile.get("pipeline_params") or {}),
        # Data-plane paths (blob/datastore); empty falls back to registered data assets
        **_data_path_params(profile.get("data_paths") or {}),
    }


# Tunable knobs surfaced upfront and baked into the standard assets. Defaults are
# sensible production starting points; the user overrides them per project.
_PARAM_DEFAULTS = {
    "optuna_trials": 20,            # HPO trials in the training scaffold
    "drift_threshold": 0.2,         # PSI alert threshold in detect_drift
    "retrain_threshold": 0.25,      # PSI severity that triggers retraining
    "monitoring_cron": "0 6 * * *", # drift-check schedule (daily 06:00 UTC)
    "instance_type": "Standard_DS2_v2",
    "instance_count": 1,
}

# Data-plane path params; "" means "fall back to azureml:<name>@latest data asset".
_DATA_PATH_DEFAULTS = {
    "training_data_path": "",
    "eval_data_path": "",
    "batch_input_path": "",
    "batch_output_path": "",
    "baseline_data_path": "",
    "current_data_path": "",
}


def _pipeline_params(overrides: dict) -> dict:
    return {k: overrides.get(k, v) for k, v in _PARAM_DEFAULTS.items()}


def _data_path_params(overrides: dict) -> dict:
    return {k: overrides.get(k, v) for k, v in _DATA_PATH_DEFAULTS.items()}


# ---------------------------------------------------------------------------
# Contract-aware code generation (adapter / scaffold / wired)
# ---------------------------------------------------------------------------

def _module_path(rel: str) -> str:
    """recommender/model.py -> recommender.model (for adapter imports)."""
    return rel.replace("\\", "/").removesuffix(".py").replace("/", ".")


def _adapter_source(stage_key: str, detected_path: str, symbol: str) -> str:
    """A thin, reviewable adapter at the standard path that wraps the user's symbol.

    The user's own module is imported, never edited. Arg wiring is best-effort and
    flagged with TODO — the user confirms it. Each adapter satisfies the same
    execution contract the standard pipeline expects.
    """
    module = _module_path(detected_path)
    sym = symbol or "main"
    header = (
        "# GENERATED ADAPTER — do not put logic here. This bridges the standard\n"
        f"# entry point to your existing code in {detected_path}. Review the call below.\n"
    )
    if stage_key in ("training", "evaluation"):
        return header + (
            "import argparse\n"
            f"from {module} import {sym}\n\n"
            "def main():\n"
            "    parser = argparse.ArgumentParser()\n"
            '    parser.add_argument("--data-path", required=True)\n'
            '    parser.add_argument("--output-dir", required=True)\n'
            "    args = parser.parse_args()\n"
            f"    {sym}(args.data_path, args.output_dir)  # TODO: match your function signature\n\n"
            'if __name__ == "__main__":\n'
            "    main()\n"
        )
    if stage_key == "scoring_realtime":
        return header + (
            f"from {module} import {sym}  # TODO: import your model-loading / predict logic\n\n"
            "def init():\n"
            "    global _model\n"
            "    _model = None  # TODO: load your model here (AML mounts it at AZUREML_MODEL_DIR)\n\n"
            "def run(raw_data):\n"
            f"    return {sym}(raw_data)  # TODO: adapt input parsing / output shape\n"
        )
    if stage_key == "scoring_batch":
        return header + (
            f"from {module} import {sym}\n\n"
            "def init():\n"
            "    global _model\n"
            "    _model = None  # TODO: load your model here\n\n"
            "def run(mini_batch):\n"
            "    results = []\n"
            "    for item in mini_batch:\n"
            f"        results.append({sym}(item))  # TODO: adapt per-item scoring\n"
            "    return results\n"
        )
    if stage_key == "drift":
        return header + (
            "import argparse\n"
            f"from {module} import {sym}\n\n"
            "def main():\n"
            "    parser = argparse.ArgumentParser()\n"
            '    parser.add_argument("--baseline-data", required=True)\n'
            '    parser.add_argument("--current-data", required=True)\n'
            '    parser.add_argument("--output", required=True)\n'
            "    args = parser.parse_args()\n"
            f"    {sym}(args.baseline_data, args.current_data, args.output)  # TODO: match signature\n\n"
            'if __name__ == "__main__":\n'
            "    main()\n"
        )
    if stage_key == "thresholds":
        return header + (
            "import argparse\n"
            f"from {module} import {sym}\n\n"
            "def main():\n"
            "    parser = argparse.ArgumentParser()\n"
            '    parser.add_argument("--drift-report", required=True)\n'
            "    args = parser.parse_args()\n"
            f"    {sym}(args.drift_report)  # TODO: match signature; exit non-zero to block retrain\n\n"
            'if __name__ == "__main__":\n'
            "    main()\n"
        )
    # rollback / fallback
    return header + (
        "import argparse\n"
        f"from {module} import {sym}\n\n"
        "def main():\n"
        "    parser = argparse.ArgumentParser()\n"
        "    args, _ = parser.parse_known_args()\n"
        f"    {sym}()  # TODO: wire arguments\n\n"
        'if __name__ == "__main__":\n'
        "    main()\n"
    )


def apply_contract_paths(params: dict, contract) -> dict:
    """Set the script-path params (train_script, score_script, ...) from the contract.

    For 'wired' stages the pipeline points at the user's detected path; for adapter/
    scaffold stages it points at the standard path (where we generate the file).
    """
    if contract is None:
        return params
    key_to_param = {
        "training": "train_script",
        "evaluation": "evaluate_script",
        "scoring_realtime": "score_script",
        "scoring_batch": "batch_score_script",
        "drift": "drift_script",
        "thresholds": "evaluate_thresholds_script",
    }
    out = dict(params)
    for st in contract.stages:
        param = key_to_param.get(st.stage)
        if not param:
            continue
        out[param] = st.pipeline_path()   # repo-relative; AML code root is the repo root
    return out


def generate(
    repo_path: str,
    gap_items: List[dict],
    params: dict,
    repo_context: str = "",
    adapt: bool = True,
    on_progress=None,          # optional callable(msg: str) — called after each component
    contract=None,             # optional ContractManifest — gates user-ML-code deliverables
    existing_pipeline_files: Optional[List[str]] = None,  # azdo/aml asset paths from the graph
) -> GenerationReport:
    report = GenerationReport()

    # Contract present: point pipelines at the right script paths, and build a
    # standard_path -> stage map so user-ML-code deliverables can be gated per mode.
    if contract is not None:
        params = apply_contract_paths(params, contract)
        stage_by_path = {st.standard_path: st for st in contract.stages}
    else:
        stage_by_path = {}

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

        # Apply contract modes per-file: a code deliverable may be wired (skip),
        # adapter (wrap user code), or scaffold (keep TODO template). Pipeline/infra
        # deliverables have no stage and are always written standard.
        written: List[str] = []
        notes: List[str] = []

        def _write(dest: str, content: str) -> None:
            existed = (Path(repo_path) / dest).exists()
            written.append(write_file(repo_path, dest, content))
            (report.updated_files if existed else report.created_files).append(dest)

        for dest, content in final_files.items():
            st = stage_by_path.get(dest)
            if st is None:
                _write(dest, content)
                continue
            if st.mode == "wired":
                report.wired_skipped.append(dest)
                notes.append(f"{st.stage}: reuse {st.detected_path}")
                continue
            if st.mode == "adapter":
                content = _adapter_source(st.stage, st.detected_path, st.detected_symbol)
                report.adapter_files.append(dest)
                notes.append(f"{st.stage}: adapter → {st.detected_path}")
            else:  # scaffold
                report.scaffold_files.append(dest)
                notes.append(f"{st.stage}: scaffold (implement TODO)")
            _write(dest, content)

        report.components.append(GeneratedComponent(
            capability=capability, component=component, files=written, adapted=adapted,
            note="; ".join(notes),
        ))
        report.written_files.extend(written)
        if on_progress:
            status = "adapted" if adapted else "rendered"
            extra = f" [{'; '.join(notes)}]" if notes else ""
            on_progress(f"✓ {capability}/{component} — {len(written)} file(s) written [{status}]{extra}")

    # Superseded: existing AzDO/AML pipeline files that are NOT part of the standard
    # set we just generated — flagged for human-approved removal at commit time.
    standard_paths = {d for item in gap_items for d in item.get("deliverables", [])}
    report.superseded_files = sorted(
        p for p in (existing_pipeline_files or [])
        if p not in standard_paths and p not in report.written_files
    )

    generated = sum(1 for c in report.components if c.files)
    extra = []
    if report.updated_files:
        extra.append(f"{len(report.created_files)} new, {len(report.updated_files)} already existed (overwritten)")
    if report.adapter_files:
        extra.append(f"{len(report.adapter_files)} adapter(s)")
    if report.scaffold_files:
        extra.append(f"{len(report.scaffold_files)} scaffold(s) to implement")
    if report.wired_skipped:
        extra.append(f"{len(report.wired_skipped)} user script(s) reused")
    if report.superseded_files:
        extra.append(f"{len(report.superseded_files)} legacy file(s) superseded — review & delete")
    tail = f"; {', '.join(extra)}" if extra else ""
    report.summary = (
        f"{generated}/{len(report.components)} components generated "
        f"({len(report.written_files)} files written to the working tree, NOT committed{tail})"
    )
    return report
