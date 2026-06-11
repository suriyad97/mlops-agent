"""Per-project execution context.

Single-user local platform: scoping is done by setting process env from the
project record (PAT, AzDO org/project derived from its repo URL) and clearing
the settings cache. A multi-user deployment would replace this with
request-scoped settings injection.
"""
import os
from urllib.parse import urlparse

from src.app.crypto import decrypt
from src.app.db import Project
from src.config.settings import get_settings


def derive_azdo(repo_url: str) -> tuple[str, str]:
    parsed = urlparse(repo_url)
    parts = [p for p in parsed.path.split("/") if p]
    if "dev.azure.com" not in (parsed.hostname or "") or "_git" not in parts:
        return "", ""
    git_idx = parts.index("_git")
    org = parts[0]
    project = parts[git_idx - 1] if git_idx >= 2 else parts[git_idx + 1]
    return f"https://dev.azure.com/{org}", project


def activate(project: Project) -> None:
    """Make tools (git/AzDO/AML) operate as this project."""
    pat = decrypt(project.pat_encrypted)
    org_url, azdo_project = derive_azdo(project.repo_url)
    os.environ["REPO_PAT"] = pat
    os.environ["AZDO_PAT"] = pat
    if org_url:
        os.environ["AZDO_ORG_URL"] = org_url
        os.environ["AZDO_PROJECT"] = azdo_project
    get_settings.cache_clear()
