"""
Azure DevOps Pipeline Tools
=============================
Trigger and monitor AzDO pipelines via the REST API.
Used primarily for the CI pipeline (Docker build → ACR → AML env).
"""

import os
import base64
import requests
from langchain_core.tools import tool

ADO_ORG = os.environ.get("AZURE_DEVOPS_ORG", "")
ADO_PROJECT = os.environ.get("AZURE_DEVOPS_PROJECT", "")
ADO_PAT = os.environ.get("AZURE_DEVOPS_PAT", "")

BASE_URL = f"https://dev.azure.com/{ADO_ORG}/{ADO_PROJECT}/_apis"
HEADERS = {
    "Authorization": "Basic " + base64.b64encode(f":{ADO_PAT}".encode()).decode(),
    "Content-Type": "application/json",
}
API_VERSION = "api-version=7.1"


@tool
def list_azdo_pipelines() -> dict:
    """
    Lists all Azure DevOps pipelines in the project.
    Use this to discover available pipeline IDs before triggering.

    Returns:
        dict with 'pipelines' list containing id, name, folder
    """
    url = f"{BASE_URL}/pipelines?{API_VERSION}"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    pipelines = [
        {"id": p["id"], "name": p["name"], "folder": p.get("folder", "/")}
        for p in response.json().get("value", [])
    ]
    return {"pipelines": pipelines, "count": len(pipelines)}


@tool
def trigger_azdo_pipeline(pipeline_id: int, branch: str = "main", parameters: dict = {}) -> dict:
    """
    Triggers an Azure DevOps pipeline run.
    Use this to start the CI pipeline (Docker build + ACR push + AML env registration).

    Args:
        pipeline_id: The numeric ID of the AzDO pipeline to trigger
        branch: The branch to run the pipeline on (default: "main")
        parameters: Optional dict of pipeline parameters to pass

    Returns:
        dict with 'run_id', 'status', 'pipeline_name', 'url'
    """
    url = f"{BASE_URL}/pipelines/{pipeline_id}/runs?{API_VERSION}"
    payload = {
        "resources": {"repositories": {"self": {"refName": f"refs/heads/{branch}"}}},
        "templateParameters": parameters,
    }
    response = requests.post(url, headers=HEADERS, json=payload)
    response.raise_for_status()
    run = response.json()
    return {
        "run_id": run["id"],
        "status": run["state"],
        "pipeline_name": run["pipeline"]["name"],
        "url": run.get("_links", {}).get("web", {}).get("href", ""),
    }


@tool
def get_azdo_run_status(pipeline_id: int, run_id: int) -> dict:
    """
    Gets the current status of an Azure DevOps pipeline run.
    Poll this until state is 'completed'.

    Args:
        pipeline_id: The numeric pipeline ID
        run_id: The numeric run ID returned when the pipeline was triggered

    Returns:
        dict with 'run_id', 'state', 'result', 'duration_minutes'
    """
    url = f"{BASE_URL}/pipelines/{pipeline_id}/runs/{run_id}?{API_VERSION}"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    run = response.json()

    # Calculate duration if available
    duration = 0
    if run.get("createdDate") and run.get("finishedDate"):
        from datetime import datetime
        created = datetime.fromisoformat(run["createdDate"].replace("Z", "+00:00"))
        finished = datetime.fromisoformat(run["finishedDate"].replace("Z", "+00:00"))
        duration = (finished - created).total_seconds() / 60

    return {
        "run_id": run_id,
        "state": run.get("state", "unknown"),      # inProgress, completed, canceling
        "result": run.get("result", "unknown"),    # succeeded, failed, canceled
        "duration_minutes": round(duration, 1),
        "url": run.get("_links", {}).get("web", {}).get("href", ""),
    }
