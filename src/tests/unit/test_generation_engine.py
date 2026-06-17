import yaml

from src.platform.catalog.loader import load_catalog
from src.platform.generation.engine import (
    generate,
    load_manifest,
    render_component,
)

PARAMS = {
    "project_name": "logistic_regression",
    "safe_name": "logistic-regression",
    "model_name": "logistic-regression-model",
    "environment_name": "logreg-env",
    "compute_target": "serverless",
    "experiment_name": "logreg-exp",
    "endpoint_name": "logistic-regression-endpoint",
    "default_branch": "main",
    "project_type": "binary_classification",
    "target_variable": "Survived",
    "metrics": ["accuracy", "f1"],
    "acr_name": "myacr",
    "train_script": "src/train.py",
    "evaluate_script": "src/evaluate.py",
    "score_script": "src/score.py",
    "batch_score_script": "src/batch_score.py",
    "drift_script": "src/monitoring/detect_drift.py",
    "evaluate_thresholds_script": "src/monitoring/evaluate_thresholds.py",
    # tunable pipeline parameters
    "optuna_trials": 20,
    "drift_threshold": 0.2,
    "retrain_threshold": 0.25,
    "register_threshold": 0.0,
    "monitoring_cron": "0 6 * * *",
    "instance_type": "Standard_DS2_v2",
    "instance_count": 1,
    # data-plane paths (empty → fall back to registered data assets)
    "training_data_path": "",
    "eval_data_path": "",
    "batch_input_path": "",
    "batch_output_path": "",
    "baseline_data_path": "",
    "current_data_path": "",
    # per-environment Azure targets
    "sc_dev": "sc-dev", "rg_dev": "rg-dev", "ws_dev": "ws-dev",
    "sc_qa": "sc-qa", "rg_qa": "rg-qa", "ws_qa": "ws-qa",
    "sc_prod": "sc-prod", "rg_prod": "rg-prod", "ws_prod": "ws-prod",
}


def test_every_catalog_deliverable_component_has_a_template():
    catalog = load_catalog()
    for name, spec in catalog.items():
        manifest = load_manifest(name)
        assert manifest is not None, f"no template manifest for {name}"
        for component in spec.deliverables:
            assert component in manifest.components, f"{name}.{component} has no template"


def test_all_templates_render_and_yaml_parses():
    catalog = load_catalog()
    for name, spec in catalog.items():
        for component in spec.deliverables:
            rendered = render_component(name, component, PARAMS)
            assert rendered, f"{name}.{component} rendered nothing"
            for dest, content in rendered.items():
                assert content.strip(), f"{name}.{component} -> {dest} empty"
                if dest.endswith((".yml", ".yaml")):
                    yaml.safe_load(content)  # raises on invalid YAML


def test_rendered_pipeline_keeps_approval_environments():
    rendered = render_component("environment_lifecycle", "azdo_env_pipeline", PARAMS)
    content = rendered["azdopipelines/ci-environment.yml"]
    assert "logistic-regression-qa" in content
    assert "logistic-regression-prod" in content


def test_generate_writes_to_working_tree(tmp_path):
    gap_items = [
        {"capability": "monitoring", "component": "drift_detection_code"},
        {"capability": "rollback", "component": "pipeline_rollback"},
    ]
    report = generate(str(tmp_path), gap_items, PARAMS, repo_context="", adapt=False)
    assert len(report.written_files) == 2
    assert (tmp_path / "src" / "monitoring" / "detect_drift.py").exists()
    assert (tmp_path / ".azuredevops" / "rollback-runbook.md").exists()
    assert "NOT committed" in report.summary


def test_validators_reject_non_string_content():
    """Regression: the LLM adaptation pass can return parsed YAML (a dict) instead of
    YAML text. That must be rejected cleanly, never raise 'dict' object has no
    attribute 'read' from PyYAML and crash generation."""
    from src.tools.validation_tools import validate_yaml
    from src.platform.generation.engine import _validate_files

    ok, msg = validate_yaml({"parsed": "yaml"})
    assert ok is False and "YAML text" in msg
    ok, _ = validate_yaml("a: 1")
    assert ok is True
    assert "non-string" in _validate_files({"x.yml": {"a": 1}})
    assert _validate_files({"x.yml": "a: 1"}) == ""


def test_generate_handles_unknown_component(tmp_path):
    report = generate(str(tmp_path), [{"capability": "training", "component": "nonexistent"}],
                      PARAMS, adapt=False)
    assert report.written_files == []
    assert "no template" in report.components[0].note
