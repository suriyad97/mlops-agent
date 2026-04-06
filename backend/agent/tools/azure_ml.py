"""
Azure ML Tools
===============
Interact with Azure ML Workspace via the Azure ML SDK v2.
Covers job submission, status polling, model registry, and job logs.
All authentication is via Service Principal — no `az login` required.
"""

import os
from datetime import datetime
from langchain_core.tools import tool
from azure.identity import ClientSecretCredential
from azure.ai.ml import MLClient, load_job
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes


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
    """Extract the AML Studio URL from a job object."""
    services = getattr(job, "services", None) or {}
    studio = services.get("Studio") or services.get("studio") or {}
    return getattr(studio, "endpoint", "") or studio.get("endpoint", "")


# ── Job Tools ──────────────────────────────────────────────────────────────────

@tool
def trigger_ct_pipeline(data_path: str, experiment_name: str = "ct_pipeline") -> dict:
    """
    Triggers the Continuous Training (CT) pipeline in Azure ML.
    Validates data path exists before submission.

    Args:
        data_path: Azure ML datastore path to training data
                   (e.g. "azureml://datastores/workspaceblobstore/paths/data/train/")
        experiment_name: AML experiment name to group runs under

    Returns:
        dict with 'job_id', 'status', 'studio_url'
    """
    try:
        ml_client = get_ml_client()
        pipeline_job = load_job(source="pipelines/ct_pipeline.yml")
        pipeline_job.inputs.data_path = data_path
        pipeline_job.experiment_name = experiment_name

        submitted = ml_client.jobs.create_or_update(pipeline_job)
        return {
            "success": True,
            "job_id": submitted.name,
            "status": submitted.status,
            "studio_url": _studio_url(submitted),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool
def trigger_inference_pipeline(input_data_path: str, output_path: str = "") -> dict:
    """
    Triggers the batch inference pipeline using the current champion model.
    Should only be run after confirming data schema validation passes.

    Args:
        input_data_path: Path to the unseen data for scoring
        output_path: Where to write predictions (optional, uses YAML default if empty)

    Returns:
        dict with 'job_id', 'status', 'studio_url'
    """
    try:
        ml_client = get_ml_client()
        pipeline_job = load_job(source="pipelines/inference_pipeline.yml")
        pipeline_job.inputs.input_data = input_data_path
        if output_path:
            pipeline_job.outputs.predictions = output_path

        submitted = ml_client.jobs.create_or_update(pipeline_job)
        return {
            "success": True,
            "job_id": submitted.name,
            "status": submitted.status,
            "studio_url": _studio_url(submitted),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool
def poll_aml_job_status(job_id: str) -> dict:
    """
    Polls the current status of an Azure ML job.
    Call this repeatedly until status is 'Completed' or 'Failed'.

    Args:
        job_id: The Azure ML job name/ID returned when the job was submitted

    Returns:
        dict with 'job_id', 'status', 'duration_minutes', 'error_message' (if failed)
    """
    try:
        ml_client = get_ml_client()
        job = ml_client.jobs.get(job_id)

        duration_minutes = 0.0
        creation_context = getattr(job, "creation_context", None)
        if creation_context:
            created_at = getattr(creation_context, "created_at", None)
            last_modified = getattr(creation_context, "last_modified_at", None)
            if created_at and last_modified:
                duration_minutes = (last_modified - created_at).total_seconds() / 60

        error_msg = None
        error = getattr(job, "error", None)
        if error:
            error_msg = getattr(error, "message", str(error))

        return {
            "job_id": job_id,
            "status": job.status,
            "duration_minutes": round(duration_minutes, 2),
            "error_message": error_msg,
            "studio_url": _studio_url(job),
        }
    except Exception as e:
        return {"job_id": job_id, "status": "Unknown", "error": str(e)}


@tool
def get_aml_job_logs(job_id: str, tail_lines: int = 50) -> dict:
    """
    Downloads and returns the last N lines of logs from an Azure ML job.
    Use this when a job fails to diagnose the error.

    Args:
        job_id: The Azure ML job name/ID
        tail_lines: Number of log lines to return from the end (default: 50)

    Returns:
        dict with 'job_id' and 'logs' (string)
    """
    try:
        import tempfile, pathlib
        ml_client = get_ml_client()

        # Download logs to a temp directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            ml_client.jobs.download(
                name=job_id,
                download_path=tmp_dir,
                all=False,  # only logs, not outputs
            )
            log_lines = []
            for log_file in pathlib.Path(tmp_dir).rglob("*.txt"):
                log_lines.extend(log_file.read_text(errors="replace").splitlines())

        tail = "\n".join(log_lines[-tail_lines:]) if log_lines else "(no logs found)"
        return {"job_id": job_id, "logs": tail}
    except Exception as e:
        return {"job_id": job_id, "logs": f"(log retrieval failed: {e})"}


@tool
def list_aml_jobs(status_filter: str = "", max_results: int = 20) -> dict:
    """
    Lists recent Azure ML jobs with their status.
    Use this to show the user what has been running.

    Args:
        status_filter: Filter by status e.g. "Failed", "Running", "Completed" (empty = all)
        max_results: Maximum number of jobs to return

    Returns:
        dict with 'jobs' list containing job_id, status, type, created_at
    """
    ml_client = get_ml_client()
    jobs = []
    for job in ml_client.jobs.list():
        if status_filter and job.status != status_filter:
            continue
        creation_context = getattr(job, "creation_context", None)
        created_at = str(creation_context.created_at) if creation_context else ""
        jobs.append({
            "job_id": job.name,
            "status": job.status,
            "display_name": getattr(job, "display_name", job.name),
            "created_at": created_at,
        })
        if len(jobs) >= max_results:
            break
    return {"jobs": jobs, "count": len(jobs)}


# ── Model Registry Tools ───────────────────────────────────────────────────────

@tool
def register_model(model_path: str, model_name: str, version: str, label: str = "challenger") -> dict:
    """
    Registers a trained model in the Azure ML model registry with a label.
    Newly trained models are always labeled 'challenger'.

    Args:
        model_path: AzureML job output path to the model artifact
        model_name: Name to register the model under
        version: Version string
        label: 'challenger' (newly trained) or 'champion' (promoted)

    Returns:
        dict with 'success', 'model_name', 'version', 'label'
    """
    ml_client = get_ml_client()
    model = Model(
        path=model_path,
        name=model_name,
        version=version,
        type=AssetTypes.CUSTOM_MODEL,
        tags={"label": label},
    )
    registered = ml_client.models.create_or_update(model)
    return {
        "success": True,
        "model_name": registered.name,
        "version": registered.version,
        "label": label,
    }


@tool
def compare_models(challenger_name: str, challenger_version: str,
                   champion_name: str, champion_version: str,
                   primary_metric: str = "auc") -> dict:
    """
    Compares challenger and champion models using their logged metrics.
    Returns which model performed better on the primary metric.

    Args:
        challenger_name: Name of the challenger model in registry
        challenger_version: Version of the challenger model
        champion_name: Name of the champion model in registry
        champion_version: Version of the champion model
        primary_metric: The metric to compare on (e.g. "auc", "f1", "rmse")

    Returns:
        dict with 'winner', 'challenger_metrics', 'champion_metrics', 'recommendation'
    """
    ml_client = get_ml_client()
    challenger = ml_client.models.get(challenger_name, challenger_version)
    champion = ml_client.models.get(champion_name, champion_version)

    challenger_metrics = challenger.tags.get("metrics", {})
    champion_metrics = champion.tags.get("metrics", {})

    challenger_score = float(challenger_metrics.get(primary_metric, 0))
    champion_score = float(champion_metrics.get(primary_metric, 0))

    # For error metrics (RMSE, MAE), lower is better
    error_metrics = ["rmse", "mae", "mse", "mape"]
    if primary_metric.lower() in error_metrics:
        winner = "challenger" if challenger_score < champion_score else "champion"
        improvement = champion_score - challenger_score
    else:
        winner = "challenger" if challenger_score > champion_score else "champion"
        improvement = challenger_score - champion_score

    return {
        "winner": winner,
        "challenger_metrics": challenger_metrics,
        "champion_metrics": champion_metrics,
        "primary_metric": primary_metric,
        "improvement": round(improvement, 4),
        "recommendation": (
            f"Promote challenger to champion ({primary_metric}: {challenger_score:.4f} vs {champion_score:.4f})"
            if winner == "challenger"
            else f"Keep current champion ({primary_metric}: {champion_score:.4f} is better)"
        ),
    }


@tool
def get_model_registry(label_filter: str = "") -> dict:
    """
    Fetches models from the Azure ML model registry.
    Use this to show the user which models are registered and their labels.

    Args:
        label_filter: Filter by label e.g. "champion", "challenger" (empty = all)

    Returns:
        dict with 'models' list
    """
    ml_client = get_ml_client()
    models = []
    for model in ml_client.models.list():
        label = model.tags.get("label", "untagged")
        if label_filter and label != label_filter:
            continue
        creation_context = getattr(model, "creation_context", None)
        models.append({
            "name": model.name,
            "version": model.version,
            "label": label,
            "metrics": model.tags.get("metrics", {}),
            "created_at": str(creation_context.created_at) if creation_context else "",
        })
    return {"models": models, "count": len(models)}
