"""Stateless git tools (clone, branch, commit, push) via GitPython."""
import os
import shutil
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse, urlunparse

from git import InvalidGitRepositoryError, Repo

from src.config.settings import get_settings
from src.shared.exceptions import ToolError
from src.shared.logging import get_logger

logger = get_logger(__name__)


def _inject_pat(url: str) -> str:
    """Embed the repo PAT into an https clone URL."""
    pat = os.getenv("REPO_PAT") or get_settings().repo_pat
    if not pat or "@" in urlparse(url).netloc:
        return url
    parsed = urlparse(url)
    port = f":{parsed.port}" if parsed.port else ""
    return urlunparse(parsed._replace(netloc=f":{pat}@{parsed.hostname}{port}"))


def _clean_url(repo_url: str) -> str:
    """Strip browser query params / fragments from an AzDO repo URL before cloning."""
    parsed = urlparse(repo_url)
    return urlunparse(parsed._replace(query="", fragment=""))


def clone_repo(repo_url: str, workspace: Optional[str] = None) -> str:
    clean = _clean_url(repo_url)
    repo_name = Path(urlparse(clean).path).name   # last path segment, no query junk

    # If the selected workspace IS itself the repo checkout (user picked the repo
    # folder directly), reuse it in place rather than nesting a clone inside it.
    if workspace:
        ws = Path(workspace)
        if (ws / ".git").is_dir():
            try:
                Repo(str(ws))
                logger.info("using selected folder as repo checkout: %s", ws)
                return str(ws)
            except InvalidGitRepositoryError:
                pass

    target = Path(workspace or get_settings().workspace_dir) / repo_name
    if target.exists():
        try:
            Repo(str(target))
            logger.info("reusing existing clone at %s", target)
            return str(target)
        except InvalidGitRepositoryError:
            shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        Repo.clone_from(_inject_pat(clean), target)
    except Exception as exc:
        raise ToolError(f"clone failed for {repo_url}: {exc}") from exc
    return str(target)


def create_branch(repo_path: str, branch_name: str) -> str:
    repo = Repo(repo_path)
    if branch_name in repo.heads:
        repo.heads[branch_name].checkout()
    else:
        repo.create_head(branch_name).checkout()
    return branch_name


def current_branch(repo_path: str) -> str:
    return Repo(repo_path).active_branch.name


def git_commit(repo_path: str, relative_paths: List[str], message: str) -> str:
    repo = Repo(repo_path)
    repo.index.add(relative_paths)
    if not repo.index.diff("HEAD") and not repo.untracked_files:
        logger.info("nothing to commit")
        return repo.head.commit.hexsha
    commit = repo.index.commit(message)
    return commit.hexsha


def git_push(repo_path: str, branch_name: str) -> None:
    repo = Repo(repo_path)
    try:
        origin = repo.remote("origin")
        push_url = _inject_pat(next(origin.urls))
        origin.set_url(push_url)
        result = origin.push(refspec=f"{branch_name}:{branch_name}", set_upstream=True)
        for info in result:
            if info.flags & info.ERROR:
                raise ToolError(f"push rejected: {info.summary}")
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"push failed: {exc}") from exc
