"""
Azure Repo Tools
=================
Talk to Azure DevOps Repos via the REST API using a PAT.
These tools allow the agent to read, list, and commit files in the repo.

Tools:
  - list_repo_files       : List all files in a folder of the repo
  - read_file_from_repo   : Read any single file
  - commit_file_to_repo   : Commit a generated file back to the repo
  - scan_azure_repo       : HIGH-LEVEL — full repo scan + structured project summary
"""

import os
import base64
import requests
from langchain_core.tools import tool

ADO_ORG = os.environ.get("AZURE_DEVOPS_ORG", "")
ADO_PROJECT = os.environ.get("AZURE_DEVOPS_PROJECT", "")
ADO_REPO = os.environ.get("AZURE_DEVOPS_REPO", "")
ADO_PAT = os.environ.get("AZURE_DEVOPS_PAT", "")

BASE_URL = f"https://dev.azure.com/{ADO_ORG}/{ADO_PROJECT}/_apis/git/repositories/{ADO_REPO}"
HEADERS = {
    "Authorization": "Basic " + base64.b64encode(f":{ADO_PAT}".encode()).decode(),
    "Content-Type": "application/json",
}
API_VERSION = "api-version=7.1"


@tool
def list_repo_files(directory_path: str = "/") -> dict:
    """
    Lists all files in a given directory of the Azure Repo.
    Use this to understand the project structure before scanning.

    Args:
        directory_path: The folder path to list (e.g. "/" for root, "/src" for src folder)

    Returns:
        dict with 'files' list and 'count'
    """
    url = f"{BASE_URL}/items?scopePath={directory_path}&recursionLevel=Full&{API_VERSION}"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    items = response.json().get("value", [])
    files = [item["path"] for item in items if not item.get("isFolder", False)]
    return {"files": files, "count": len(files)}


@tool
def read_file_from_repo(file_path: str, branch: str = "main") -> dict:
    """
    Reads the content of a file from the Azure Repo.
    Use this to inspect training scripts, requirements.txt, existing configs, etc.

    Args:
        file_path: Path to the file in the repo (e.g. "/train.py", "/requirements.txt")
        branch: The branch to read from (default: "main")

    Returns:
        dict with 'content' (string) and 'file_path'
    """
    url = f"{BASE_URL}/items?path={file_path}&versionDescriptor.version={branch}&{API_VERSION}"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    content = response.text
    return {"file_path": file_path, "content": content}


@tool
def commit_file_to_repo(file_path: str, content: str, commit_message: str, branch: str = "main") -> dict:
    """
    Commits a new or updated file to the Azure Repo.
    Use this to save generated Dockerfiles, pipeline YAMLs, etc.

    Args:
        file_path: Path where the file should be saved (e.g. "/Dockerfile")
        content: The full text content of the file
        commit_message: A clear commit message describing what was added
        branch: The branch to commit to (default: "main")

    Returns:
        dict with 'success', 'commit_id', and 'file_path'
    """
    # First check if file exists to decide between add vs edit
    check_url = f"{BASE_URL}/items?path={file_path}&{API_VERSION}"
    check_response = requests.get(check_url, headers=HEADERS)
    file_exists = check_response.status_code == 200

    # Get latest commit on the branch
    ref_url = f"{BASE_URL}/refs?filter=heads/{branch}&{API_VERSION}"
    ref_response = requests.get(ref_url, headers=HEADERS)
    ref_response.raise_for_status()
    old_object_id = ref_response.json()["value"][0]["objectId"]

    payload = {
        "refUpdates": [{"name": f"refs/heads/{branch}", "oldObjectId": old_object_id}],
        "commits": [{
            "comment": commit_message,
            "changes": [{
                "changeType": "edit" if file_exists else "add",
                "item": {"path": file_path},
                "newContent": {
                    "content": content,
                    "contentType": "rawtext"
                }
            }]
        }]
    }

    push_url = f"{BASE_URL}/pushes?{API_VERSION}"
    response = requests.post(push_url, headers=HEADERS, json=payload)
    response.raise_for_status()
    commit_id = response.json()["commits"][0]["commitId"]
    return {"success": True, "commit_id": commit_id, "file_path": file_path}


# ── Key file patterns the scanner looks for ───────────────────────────────────

_ENTRYPOINT_NAMES = {
    "train.py", "training.py", "run_training.py", "fit.py",
    "score.py", "scoring.py", "inference.py", "predict.py",
    "main.py", "run.py", "pipeline.py",
}
_CONFIG_NAMES = {
    "config.yaml", "config.yml", "params.yaml", "params.yml",
    "hyperparameters.yaml", "hyperparameters.json",
    "settings.yaml", "settings.json",
}
_SCHEMA_NAMES = {
    "schema.yaml", "schema.json", "data_schema.yaml",
    "input_schema.json", "output_schema.json",
}
_ARTIFACT_NAMES = {
    "Dockerfile", "dockerfile",
    "environment.yaml", "environment.yml", "conda.yaml", "conda.yml",
    "ct_pipeline.yaml", "ct_pipeline.yml",
    "inference_pipeline.yaml", "inference_pipeline.yml",
    "drift_pipeline.yaml", "drift_pipeline.yml",
    "azure-pipelines.yml", "azure-pipelines.yaml",
}


@tool
def scan_azure_repo(branch: str = "main") -> dict:
    """
    HIGH-LEVEL REPO SCANNER. Call this first when a user asks you to understand
    their ML project. It does the following in a single call:

      1. Lists every file in the Azure Repo
      2. Identifies key ML files: training entrypoints, requirements,
         config files, data schema files, and any existing pipeline YAMLs
      3. Reads the content of each identified file (capped at 4000 chars each)
      4. Returns a rich structured summary so you can understand the project
         without making multiple separate tool calls

    Args:
        branch: The branch to scan (default: "main")

    Returns:
        dict with:
          - all_files: full list of every file path in the repo
          - file_count: total number of files
          - entrypoints: {filename: content} for detected training/scoring scripts
          - requirements: content of requirements.txt / setup.py if found
          - configs: {filename: content} for config/params files
          - schemas: {filename: content} for data schema files
          - existing_artifacts: list of already-committed Dockerfiles / pipeline YAMLs
          - missing_artifacts: list of artifacts the agent will need to generate
          - framework_hints: phrases found in requirements that hint at ML framework
          - summary: a short human-readable summary of findings
    """
    # ── Step 1: List all files ─────────────────────────────────────────────────
    list_url = f"{BASE_URL}/items?scopePath=/&recursionLevel=Full&{API_VERSION}"
    r = requests.get(list_url, headers=HEADERS)
    r.raise_for_status()
    items = r.json().get("value", [])
    all_files = [item["path"] for item in items if not item.get("isFolder", False)]

    # ── Step 2: Classify files ─────────────────────────────────────────────────
    entrypoint_paths, config_paths, schema_paths = [], [], []
    req_path = None
    existing_artifact_paths, missing_artifact_names = [], []

    for path in all_files:
        basename = path.split("/")[-1]
        basename_lower = basename.lower()

        if basename in _ENTRYPOINT_NAMES:
            entrypoint_paths.append(path)
        if basename in _CONFIG_NAMES:
            config_paths.append(path)
        if basename in _SCHEMA_NAMES:
            schema_paths.append(path)
        if basename_lower in ("requirements.txt", "setup.py", "pyproject.toml"):
            req_path = path
        if basename in _ARTIFACT_NAMES:
            existing_artifact_paths.append(path)

    existing_basenames = {p.split("/")[-1] for p in existing_artifact_paths}
    for artifact in _ARTIFACT_NAMES:
        if artifact not in existing_basenames:
            missing_artifact_names.append(artifact)

    # ── Step 3: Read key file contents ────────────────────────────────────────
    def _read(path: str) -> str:
        try:
            url = f"{BASE_URL}/items?path={path}&versionDescriptor.version={branch}&{API_VERSION}"
            resp = requests.get(url, headers=HEADERS)
            resp.raise_for_status()
            return resp.text[:4000]  # cap to avoid context overflow
        except Exception as e:
            return f"[Error reading {path}: {e}]"

    entrypoints = {p.split("/")[-1]: _read(p) for p in entrypoint_paths[:5]}
    configs     = {p.split("/")[-1]: _read(p) for p in config_paths[:3]}
    schemas     = {p.split("/")[-1]: _read(p) for p in schema_paths[:3]}
    requirements = _read(req_path) if req_path else "Not found"

    # ── Step 4: Detect framework hints from requirements ──────────────────────
    framework_hints = []
    req_lower = requirements.lower()
    for fw in ["sklearn", "scikit-learn", "xgboost", "lightgbm", "catboost",
               "tensorflow", "keras", "torch", "pytorch", "transformers",
               "prophet", "statsmodels", "mlflow", "automl"]:
        if fw in req_lower:
            framework_hints.append(fw)

    # ── Step 5: Build summary ─────────────────────────────────────────────────
    summary_parts = [
        f"Found {len(all_files)} files total.",
        f"Entrypoints detected: {list(entrypoints.keys()) or 'none'}.",
        f"Config files: {list(configs.keys()) or 'none'}.",
        f"Schema files: {list(schemas.keys()) or 'none'}.",
        f"Requirements: {'found at ' + req_path if req_path else 'not found'}.",
        f"Frameworks detected: {framework_hints or 'unknown'}.",
        f"Existing MLOps artifacts: {[p.split('/')[-1] for p in existing_artifact_paths] or 'none'}.",
        f"Missing artifacts to generate: {missing_artifact_names}."
    ]

    return {
        "all_files":          all_files,
        "file_count":         len(all_files),
        "entrypoints":        entrypoints,
        "requirements":       requirements,
        "configs":            configs,
        "schemas":            schemas,
        "existing_artifacts": [p.split("/")[-1] for p in existing_artifact_paths],
        "missing_artifacts":  missing_artifact_names,
        "framework_hints":    framework_hints,
        "summary":            " ".join(summary_parts),
    }
