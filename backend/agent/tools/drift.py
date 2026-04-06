"""
Drift Monitoring Tools
=======================
Trigger and fetch results from drift monitoring pipelines.
Three types: data drift, prediction drift, concept drift.
All authentication is via Service Principal — no `az login` required.
"""

import os
from langchain_core.tools import tool
from azure.identity import ClientSecretCredential
from azure.ai.ml import MLClient, load_job


# ── ML Client Setup ────────────────────────────────────────────────────────────

def get_ml_client() -> MLClient:
    credential = ClientSecretCredential(
        tenant_id=os.environ["AZURE_TENANT_ID"],
        client_id=os.environ["AZURE_CLIENT_ID"],
        client_secret=os.environ["AZURE_CLIENT_SECRET"],
    )
    return MLClient(
        credential=credential,
        subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
        resource_group_name=os.environ["AML_RESOURCE_GROUP"],
        workspace_name=os.environ["AML_WORKSPACE_NAME"],
    )


def _studio_url(job) -> str:
    services = getattr(job, "services", None) or {}
    studio = services.get("Studio") or services.get("studio") or {}
    return getattr(studio, "endpoint", "") or studio.get("endpoint", "")


def _submit_pipeline(pipeline_file: str, inputs: dict) -> dict:
    """Helper: load a pipeline YAML, set inputs, and submit via SDK."""
    try:
        ml_client = get_ml_client()
        pipeline_job = load_job(source=pipeline_file)
        for key, value in inputs.items():
            setattr(pipeline_job.inputs, key, value)
        submitted = ml_client.jobs.create_or_update(pipeline_job)
        return {
            "success": True,
            "job_id": submitted.name,
            "status": submitted.status,
            "studio_url": _studio_url(submitted),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Drift Pipeline Tools ───────────────────────────────────────────────────────

@tool
def run_data_drift_pipeline(reference_data_path: str, current_data_path: str) -> dict:
    """
    Triggers the data drift monitoring pipeline.
    Compares the distribution of current input data vs reference training data.

    Args:
        reference_data_path: Path to the reference (training) dataset
        current_data_path: Path to the current production input data

    Returns:
        dict with 'job_id', 'status', 'studio_url'
    """
    return _submit_pipeline("pipelines/data_drift_pipeline.yml", {
        "reference_data": reference_data_path,
        "current_data": current_data_path,
    })


@tool
def run_prediction_drift_pipeline(reference_predictions_path: str, current_predictions_path: str) -> dict:
    """
    Triggers the prediction drift monitoring pipeline.
    Compares the distribution of model predictions over time.

    Args:
        reference_predictions_path: Path to reference prediction outputs
        current_predictions_path: Path to recent prediction outputs

    Returns:
        dict with 'job_id', 'status', 'studio_url'
    """
    return _submit_pipeline("pipelines/prediction_drift_pipeline.yml", {
        "reference_predictions": reference_predictions_path,
        "current_predictions": current_predictions_path,
    })


@tool
def run_concept_drift_pipeline(predictions_path: str, actuals_path: str) -> dict:
    """
    Triggers the concept drift pipeline.
    Compares model predictions against ground truth actuals to detect
    degradation in model accuracy over time.

    Args:
        predictions_path: Path to model prediction outputs
        actuals_path: Path to ground truth actual labels

    Returns:
        dict with 'job_id', 'status', 'studio_url'
    """
    return _submit_pipeline("pipelines/concept_drift_pipeline.yml", {
        "predictions": predictions_path,
        "actuals": actuals_path,
    })


@tool
def get_latest_drift_report() -> dict:
    """
    Fetches the latest drift report by inspecting the most recent completed
    drift jobs in the Azure ML workspace. Returns scores for data drift,
    prediction drift, and concept drift along with threshold status.

    Returns:
        dict with drift scores, thresholds, and overall status
    """
    ml_client = get_ml_client()

    thresholds = {
        "data_drift": float(os.environ.get("DATA_DRIFT_THRESHOLD", "0.15")),
        "prediction_drift": float(os.environ.get("PREDICTION_DRIFT_THRESHOLD", "0.10")),
        "concept_drift": float(os.environ.get("CONCEPT_DRIFT_THRESHOLD", "0.05")),
    }

    drift_results = {
        "data_drift_score": None,
        "prediction_drift_score": None,
        "concept_drift_score": None,
        "data_drift_status": "unknown",
        "prediction_drift_status": "unknown",
        "concept_drift_status": "unknown",
        "threshold_breached": False,
        "last_updated": None,
        "thresholds": thresholds,
    }

    # Map of experiment name fragment → which drift type it feeds
    drift_experiment_map = {
        "data_drift": "data_drift_score",
        "prediction_drift": "prediction_drift_score",
        "concept_drift": "concept_drift_score",
    }

    try:
        for job in ml_client.jobs.list():
            if job.status != "Completed":
                continue
            experiment = getattr(job, "experiment_name", "") or ""
            for keyword, score_key in drift_experiment_map.items():
                if keyword in experiment and drift_results[score_key] is None:
                    # Extract score from job tags if present
                    score = job.tags.get("drift_score") or job.tags.get(keyword)
                    if score is not None:
                        drift_results[score_key] = float(score)
                    creation_context = getattr(job, "creation_context", None)
                    if creation_context and drift_results["last_updated"] is None:
                        drift_results["last_updated"] = str(creation_context.created_at)

        # Compute statuses
        for drift_type, threshold_key in [
            ("data_drift", "data_drift"),
            ("prediction_drift", "prediction_drift"),
            ("concept_drift", "concept_drift"),
        ]:
            score = drift_results[f"{drift_type}_score"]
            threshold = thresholds[threshold_key]
            if score is None:
                drift_results[f"{drift_type}_status"] = "unknown"
            elif score > threshold:
                drift_results[f"{drift_type}_status"] = "red"
                drift_results["threshold_breached"] = True
            elif score > threshold * 0.75:
                drift_results[f"{drift_type}_status"] = "amber"
            else:
                drift_results[f"{drift_type}_status"] = "green"

    except Exception as e:
        drift_results["error"] = str(e)

    return drift_results
