"""
Azure Repo Tools
=================
Talk to Azure DevOps Repos via the REST API using a PAT.
These tools allow the agent to read, list, and commit files in the repo.
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
