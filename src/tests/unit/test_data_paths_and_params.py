"""Data-path verification + tunable-parameter wiring."""
from src.config.settings import get_settings
from src.platform.generation.engine import build_params, render_component
from src.tools.data_path_tools import verify_data_paths


def _status_by_name(report):
    return {c.name: c.status for c in report.checks}


def test_verify_flags_required_paths_when_empty():
    report = verify_data_paths({"endpoint_strategy": "realtime"})
    by = _status_by_name(report)
    # required-for-realtime paths show as missing_config; optional ones are omitted
    assert by.get("Training data") == "missing_config"
    assert by.get("Monitoring baseline") == "missing_config"
    assert "Evaluation data" not in by   # optional → not cluttered


def test_verify_classifies_path_schemes():
    report = verify_data_paths({"endpoint_strategy": "none", "data_paths": {
        "training_data_path": "azureml://datastores/ds/paths/training/",
        "eval_data_path": "gobbledygook://nope",
    }})
    by = _status_by_name(report)
    assert by["Training data"] == "ok"        # valid datastore URI (existence unverified)
    assert by["Evaluation data"] == "error"   # unrecognized scheme


def test_batch_paths_required_only_for_batch_strategy():
    realtime = _status_by_name(verify_data_paths({"endpoint_strategy": "realtime"}))
    batch = _status_by_name(verify_data_paths({"endpoint_strategy": "batch"}))
    assert "Batch input" not in realtime
    assert batch.get("Batch input") == "missing_config"


def test_params_default_and_override_flow_into_templates():
    settings = get_settings()
    base = {"endpoint_strategy": "both", "project_type": "binary_classification", "target_variable": "y"}

    p_def = build_params("p", base, settings)
    assert p_def["drift_threshold"] == 0.2 and p_def["optuna_trials"] == 20
    assert p_def["training_data_path"] == ""   # empty → template falls back to asset

    over = dict(base, pipeline_params={"drift_threshold": 0.4, "monitoring_cron": "0 */2 * * *"},
                data_paths={"training_data_path": "azureml://datastores/d/paths/t/"})
    p_over = build_params("p", over, settings)
    assert p_over["drift_threshold"] == 0.4
    drift = render_component("monitoring", "drift_detection_code", p_over)["src/monitoring/detect_drift.py"]
    assert "PSI_ALERT_THRESHOLD = 0.4" in drift
    pipe = render_component("training", "aml_training_assets", p_over)["MLpipelines/training_pipeline.yml"]
    assert "azureml://datastores/d/paths/t/" in pipe
