"""In-memory registry for long-running background jobs (remediation runs)."""
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from src.shared.logging import get_logger

logger = get_logger(__name__)

_jobs: Dict[str, Dict[str, Any]] = {}
_lock = threading.Lock()


def start_job(kind: str, target: Callable[[], Any], detail: str = "") -> str:
    job_id = uuid.uuid4().hex[:8]
    with _lock:
        _jobs[job_id] = {
            "id": job_id,
            "kind": kind,
            "detail": detail,
            "status": "running",
            "started": datetime.now(timezone.utc).isoformat(),
            "result": None,
            "error": None,
        }

    def _run() -> None:
        try:
            result = target()
            with _lock:
                _jobs[job_id]["status"] = "completed"
                _jobs[job_id]["result"] = result
        except Exception as exc:
            logger.exception("job %s failed", job_id)
            with _lock:
                _jobs[job_id]["status"] = "failed"
                _jobs[job_id]["error"] = str(exc)[:2000]

    threading.Thread(target=_run, name=f"job-{job_id}", daemon=True).start()
    return job_id


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def list_jobs() -> list:
    with _lock:
        return [
            {k: v for k, v in job.items() if k != "result"}
            for job in _jobs.values()
        ]
