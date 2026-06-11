"""Stateless Azure DevOps REST tools: pipelines, runs, logs, pull requests."""
import base64
import time
from typing import Any, Dict, List, Optional

import requests

from src.config.settings import get_settings
from src.shared.exceptions import ConfigurationError, ToolError
from src.shared.logging import get_logger

logger = get_logger(__name__)
API_VERSION = "7.1"


def _base_url() -> str:
    settings = get_settings()
    if not settings.azdo_org_url or not settings.azdo_project:
        raise ConfigurationError("AZDO_ORG_URL and AZDO_PROJECT must be configured")
    return f"{settings.azdo_org_url.rstrip('/')}/{settings.azdo_project}"


def _headers() -> Dict[str, str]:
    pat = get_settings().effective_azdo_pat
    if not pat:
        raise ConfigurationError("AZDO_PAT (or REPO_PAT) must be configured")
    token = base64.b64encode(f":{pat}".encode()).decode()
    return {"Authorization": f"Basic {token}", "Content-Type": "application/json"}


def _request(method: str, url: str, **kwargs) -> Dict[str, Any]:
    response = requests.request(method, url, headers=_headers(), timeout=60, **kwargs)
    if response.status_code >= 300:
        raise ToolError(f"AzDO API {method} {url} -> {response.status_code}: {response.text[:2000]}")
    return response.json() if response.text else {}


def get_repository_id(repo_name: str) -> str:
    data = _request("GET", f"{_base_url()}/_apis/git/repositories/{repo_name}?api-version={API_VERSION}")
    return data["id"]


def create_azure_pipeline(name: str, repo_name: str, yaml_path: str = "azure-pipelines.yml") -> int:
    """Create a YAML pipeline bound to an Azure Repos repository. Returns pipeline id."""
    existing = _request("GET", f"{_base_url()}/_apis/pipelines?api-version={API_VERSION}")
    for p in existing.get("value", []):
        if p["name"] == name:
            logger.info("pipeline '%s' already exists (id=%s)", name, p["id"])
            return p["id"]
    body = {
        "name": name,
        "configuration": {
            "type": "yaml",
            "path": f"/{yaml_path}",
            "repository": {"id": get_repository_id(repo_name), "name": repo_name, "type": "azureReposGit"},
        },
    }
    data = _request("POST", f"{_base_url()}/_apis/pipelines?api-version={API_VERSION}", json=body)
    return data["id"]


def run_azure_pipeline(pipeline_id: int, branch: str,
                       parameters: Optional[Dict[str, str]] = None) -> int:
    body: Dict[str, Any] = {
        "resources": {"repositories": {"self": {"refName": f"refs/heads/{branch}"}}},
    }
    if parameters:
        body["templateParameters"] = parameters
    data = _request("POST", f"{_base_url()}/_apis/pipelines/{pipeline_id}/runs?api-version={API_VERSION}", json=body)
    return data["id"]


def get_run(pipeline_id: int, run_id: int) -> Dict[str, Any]:
    """Current state of a single pipeline run (state, result, timing)."""
    return _request(
        "GET",
        f"{_base_url()}/_apis/pipelines/{pipeline_id}/runs/{run_id}?api-version={API_VERSION}",
    )


def wait_for_run(pipeline_id: int, run_id: int, timeout_seconds: int = 3600, poll: int = 20) -> Dict[str, Any]:
    """Poll until the run finishes. Returns the final run object."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        run = get_run(pipeline_id, run_id)
        if run.get("state") == "completed":
            return run
        time.sleep(poll)
    raise ToolError(f"pipeline run {run_id} did not complete within {timeout_seconds}s")


def get_pipeline_logs(pipeline_id: int, run_id: int, max_chars: int = 30_000) -> str:
    """Concatenated log content for a pipeline run (truncated from the tail)."""
    listing = _request(
        "GET",
        f"{_base_url()}/_apis/pipelines/{pipeline_id}/runs/{run_id}/logs"
        f"?$expand=signedContent&api-version={API_VERSION}",
    )
    chunks: List[str] = []
    for log in listing.get("logs", []):
        url = (log.get("signedContent") or {}).get("url")
        if not url:
            continue
        try:
            text = requests.get(url, timeout=60).text
            chunks.append(f"--- log {log['id']} ---\n{text}")
        except requests.RequestException as exc:
            chunks.append(f"--- log {log['id']} unavailable: {exc} ---")
    combined = "\n".join(chunks)
    return combined[-max_chars:]


def create_pull_request(
    repo_name: str,
    source_branch: str,
    target_branch: str,
    title: str,
    description: str = "",
) -> Optional[str]:
    repo_id = get_repository_id(repo_name)
    body = {
        "sourceRefName": f"refs/heads/{source_branch}",
        "targetRefName": f"refs/heads/{target_branch}",
        "title": title,
        "description": description[:4000],
    }
    try:
        data = _request(
            "POST",
            f"{_base_url()}/_apis/git/repositories/{repo_id}/pullrequests?api-version={API_VERSION}",
            json=body,
        )
    except ToolError as exc:
        if "TF401179" in str(exc):  # active PR already exists for this source/target
            logger.info("pull request already exists for %s -> %s", source_branch, target_branch)
            return None
        raise
    web = get_settings().azdo_org_url.rstrip("/")
    return f"{web}/{get_settings().azdo_project}/_git/{repo_name}/pullrequest/{data['pullRequestId']}"


def list_pipelines() -> List[Dict[str, Any]]:
    data = _request("GET", f"{_base_url()}/_apis/pipelines?api-version={API_VERSION}")
    return [{"id": p["id"], "name": p["name"], "folder": p.get("folder", "")} for p in data.get("value", [])]


def list_pipeline_runs(pipeline_id: int, top: int = 10) -> List[Dict[str, Any]]:
    data = _request("GET", f"{_base_url()}/_apis/pipelines/{pipeline_id}/runs?api-version={API_VERSION}")
    return [
        {
            "run_id": r["id"],
            "state": r.get("state"),
            "result": r.get("result", ""),
            "created": r.get("createdDate", ""),
            "finished": r.get("finishedDate", ""),
            "name": r.get("name", ""),
        }
        for r in data.get("value", [])[:top]
    ]


def list_pull_requests(repo_name: str, status: str = "active") -> List[Dict[str, Any]]:
    repo_id = get_repository_id(repo_name)
    data = _request(
        "GET",
        f"{_base_url()}/_apis/git/repositories/{repo_id}/pullrequests"
        f"?searchCriteria.status={status}&api-version={API_VERSION}",
    )
    return [
        {
            "id": pr["pullRequestId"],
            "title": pr.get("title", ""),
            "source": pr.get("sourceRefName", "").replace("refs/heads/", ""),
            "target": pr.get("targetRefName", "").replace("refs/heads/", ""),
            "status": pr.get("status", ""),
            "created_by": (pr.get("createdBy") or {}).get("displayName", ""),
        }
        for pr in data.get("value", [])
    ]


def list_repositories() -> List[Dict[str, Any]]:
    data = _request("GET", f"{_base_url()}/_apis/git/repositories?api-version={API_VERSION}")
    return [
        {"name": r["name"], "default_branch": r.get("defaultBranch", ""), "url": r.get("webUrl", "")}
        for r in data.get("value", [])
    ]


def preview_pipeline_yaml(pipeline_id: int, yaml_content: str, branch: str = "main") -> Dict[str, Any]:
    """Server-side YAML validation WITHOUT execution (previewRun + yamlOverride).

    Azure DevOps fully parses and expands the pipeline (catching unauthorized
    variable groups, unknown tasks, schema errors) but creates no run.
    Returns {"valid": bool, "error": str, "final_yaml": str}.
    """
    body = {
        "previewRun": True,
        "yamlOverride": yaml_content,
        "resources": {"repositories": {"self": {"refName": f"refs/heads/{branch}"}}},
    }
    try:
        data = _request(
            "POST",
            f"{_base_url()}/_apis/pipelines/{pipeline_id}/preview?api-version={API_VERSION}",
            json=body,
        )
        return {"valid": True, "error": "", "final_yaml": data.get("finalYaml", "")}
    except ToolError as exc:
        return {"valid": False, "error": str(exc)[:1500], "final_yaml": ""}


def get_default_branch(repo_name: str) -> str:
    data = _request("GET", f"{_base_url()}/_apis/git/repositories/{repo_name}?api-version={API_VERSION}")
    return data.get("defaultBranch", "refs/heads/main").replace("refs/heads/", "")


def list_branches_for_url(repo_url: str, pat: str) -> List[str]:
    """List branches using an explicit URL + PAT (no settings required)."""
    from urllib.parse import urlparse
    parsed = urlparse(repo_url)
    parts = [p for p in parsed.path.split("/") if p]
    if "dev.azure.com" not in (parsed.hostname or "") or "_git" not in parts:
        raise ToolError("not a valid Azure DevOps repo URL (expected dev.azure.com/.../org/project/_git/repo)")
    git_idx = parts.index("_git")
    org = parts[0]
    project = parts[git_idx - 1] if git_idx >= 2 else ""
    repo = parts[git_idx + 1] if git_idx + 1 < len(parts) else ""
    if not project or not repo:
        raise ToolError("could not parse org/project/repo from URL")
    token = base64.b64encode(f":{pat}".encode()).decode()
    headers = {"Authorization": f"Basic {token}", "Content-Type": "application/json"}
    url = (
        f"https://dev.azure.com/{org}/{project}/_apis/git/repositories/{repo}"
        f"/refs?filter=heads/&api-version={API_VERSION}"
    )
    resp = requests.get(url, headers=headers, timeout=30)
    if resp.status_code >= 300:
        raise ToolError(f"AzDO branches API {resp.status_code}: {resp.text[:500]}")
    return [
        ref["name"].replace("refs/heads/", "")
        for ref in resp.json().get("value", [])
    ]
