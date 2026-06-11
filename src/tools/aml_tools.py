"""Stateless Azure ML tools via azure-ai-ml SDK (imported lazily)."""
from pathlib import Path
from typing import Any, Dict

from src.config.settings import get_settings
from src.shared.exceptions import ConfigurationError, ToolError
from src.shared.logging import get_logger

logger = get_logger(__name__)


def _ml_client():
    try:
        from azure.ai.ml import MLClient
        from azure.identity import DefaultAzureCredential
    except ImportError as exc:
        raise ToolError("azure-ai-ml / azure-identity not installed") from exc
    settings = get_settings()
    if not all([settings.azure_subscription_id, settings.aml_resource_group, settings.aml_workspace]):
        raise ConfigurationError(
            "AZURE_SUBSCRIPTION_ID, AML_RESOURCE_GROUP and AML_WORKSPACE must be configured"
        )
    return MLClient(
        DefaultAzureCredential(),
        subscription_id=settings.azure_subscription_id,
        resource_group_name=settings.aml_resource_group,
        workspace_name=settings.aml_workspace,
    )


def create_aml_environment(repo_path: str, environment_yaml: str) -> str:
    from azure.ai.ml import load_environment
    path = Path(repo_path) / environment_yaml
    env = load_environment(source=str(path))
    result = _ml_client().environments.create_or_update(env)
    logger.info("AML environment %s:%s created", result.name, result.version)
    return f"{result.name}:{result.version}"


def create_aml_component(repo_path: str, component_yaml: str) -> str:
    from azure.ai.ml import load_component
    path = Path(repo_path) / component_yaml
    component = load_component(source=str(path))
    result = _ml_client().components.create_or_update(component)
    logger.info("AML component %s:%s created", result.name, result.version)
    return f"{result.name}:{result.version}"


def run_aml_pipeline(repo_path: str, pipeline_yaml: str, experiment_name: str = "remediation-agent") -> Dict[str, Any]:
    """Submit a pipeline job defined in YAML and wait for terminal status."""
    from azure.ai.ml import load_job
    client = _ml_client()
    job = load_job(source=str(Path(repo_path) / pipeline_yaml))
    job.experiment_name = experiment_name
    submitted = client.jobs.create_or_update(job)
    logger.info("AML job %s submitted", submitted.name)
    client.jobs.stream(submitted.name)  # blocks until terminal state
    final = client.jobs.get(submitted.name)
    return {"name": final.name, "status": final.status, "studio_url": final.studio_url}


def get_workspace_details() -> Dict[str, Any]:
    client = _ml_client()
    ws = client.workspaces.get(get_settings().aml_workspace)
    return {
        "name": ws.name,
        "location": ws.location,
        "resource_group": ws.resource_group,
        "description": ws.description or "",
        "mlflow_tracking_uri": ws.mlflow_tracking_uri or "",
        "studio_url": getattr(ws, "studio_url", "") or "",
    }


def list_aml_jobs(top: int = 10) -> list:
    client = _ml_client()
    jobs = []
    for job in client.jobs.list():
        jobs.append({
            "name": job.name,
            "status": getattr(job, "status", ""),
            "type": getattr(job, "type", ""),
            "experiment": getattr(job, "experiment_name", ""),
            "created": str(getattr(getattr(job, "creation_context", None), "created_at", "")),
        })
        if len(jobs) >= top:
            break
    return jobs


def list_aml_compute() -> list:
    client = _ml_client()
    return [
        {
            "name": c.name,
            "type": getattr(c, "type", ""),
            "state": getattr(c, "provisioning_state", ""),
            "size": getattr(c, "size", ""),
        }
        for c in client.compute.list()
    ]


def list_aml_environments(top: int = 10) -> list:
    client = _ml_client()
    envs = []
    for env in client.environments.list():
        envs.append({"name": env.name, "latest_version": getattr(env, "latest_version", "")})
        if len(envs) >= top:
            break
    return envs


def get_aml_job_logs(job_name: str, max_chars: int = 30_000) -> str:
    client = _ml_client()
    try:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            client.jobs.download(job_name, download_path=tmp, all=True)
            chunks = []
            for log_file in sorted(Path(tmp).rglob("*.txt")):
                chunks.append(f"--- {log_file.name} ---\n{log_file.read_text(errors='ignore')}")
            return "\n".join(chunks)[-max_chars:]
    except Exception as exc:
        raise ToolError(f"could not download AML logs for {job_name}: {exc}") from exc
