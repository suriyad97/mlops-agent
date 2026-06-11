"""Stateless filesystem tools scoped to a repository checkout."""
from pathlib import Path
from typing import Dict, List

from src.shared.exceptions import ToolError

_KEY_FILE_NAMES = {
    "readme.md", "requirements.txt", "pyproject.toml", "setup.py", "setup.cfg",
    "package.json", "makefile", "dockerfile", "environment.yml", "conda.yaml",
}
MAX_KEY_FILE_BYTES = 2000


def scan_repo(repo_path: str) -> List[str]:
    """All repo-relative file paths, excluding .git internals."""
    base = Path(repo_path)
    if not base.is_dir():
        raise ToolError(f"not a directory: {repo_path}")
    return sorted(
        str(p.relative_to(base)).replace("\\", "/")
        for p in base.rglob("*")
        if p.is_file() and ".git" not in p.parts
    )


def read_file(repo_path: str, relative_path: str, max_bytes: int = 50_000) -> str:
    target = (Path(repo_path) / relative_path).resolve()
    if Path(repo_path).resolve() not in target.parents and target != Path(repo_path).resolve():
        raise ToolError(f"path escapes repository: {relative_path}")
    if not target.is_file():
        raise ToolError(f"file not found: {relative_path}")
    return target.read_text(encoding="utf-8", errors="ignore")[:max_bytes]


def write_file(repo_path: str, relative_path: str, content: str) -> str:
    target = (Path(repo_path) / relative_path).resolve()
    if Path(repo_path).resolve() not in target.parents:
        raise ToolError(f"path escapes repository: {relative_path}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return relative_path


def list_files(repo_path: str, subdir: str = "") -> List[str]:
    base = Path(repo_path) / subdir
    if not base.is_dir():
        return []
    return sorted(p.name + ("/" if p.is_dir() else "") for p in base.iterdir())


def read_key_files(repo_path: str) -> Dict[str, str]:
    """Contents of well-known config/docs files, truncated, keyed by relative path."""
    contents: Dict[str, str] = {}
    for rel in scan_repo(repo_path):
        if Path(rel).name.lower() in _KEY_FILE_NAMES:
            contents[rel] = read_file(repo_path, rel, max_bytes=MAX_KEY_FILE_BYTES)
    return contents
