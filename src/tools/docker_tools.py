"""Stateless Docker / ACR tools (shell out to docker and az CLIs)."""
import subprocess
from typing import List

from src.config.settings import get_settings
from src.shared.exceptions import ToolError
from src.shared.logging import get_logger

logger = get_logger(__name__)


def _run(cmd: List[str], timeout: int = 1800) -> str:
    logger.info("exec: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise ToolError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr[-4000:]}")
    return proc.stdout


def build_docker_image(repo_path: str, image_name: str, tag: str = "latest") -> str:
    full = f"{image_name}:{tag}"
    _run(["docker", "build", "-t", full, repo_path])
    return full


def acr_login() -> None:
    acr = get_settings().acr_name
    if not acr:
        raise ToolError("ACR_NAME not configured")
    _run(["az", "acr", "login", "--name", acr], timeout=300)


def push_docker_image(image_name: str, tag: str = "latest") -> str:
    acr = get_settings().acr_name
    remote = f"{acr}.azurecr.io/{image_name}:{tag}"
    _run(["docker", "tag", f"{image_name}:{tag}", remote])
    _run(["docker", "push", remote])
    return remote
