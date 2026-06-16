"""Platform API v2: projects, threads, chat, scan.

Run: uvicorn src.app.main:app --port 8000
"""
import json
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

from fastapi import Depends, FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from src.app import services  # noqa: E402
from src.app.crypto import encrypt  # noqa: E402
from src.app.db import Message, Project, Report, Thread, get_session, init_db  # noqa: E402
from src.shared.logging import get_logger  # noqa: E402
from src.shared.observability import setup_observability  # noqa: E402

logger = get_logger(__name__)

app = FastAPI(title="MLOps Platform API v2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    init_db()
    setup_observability()


# --- schemas -----------------------------------------------------------------

class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    repo_url: str
    pat: str = Field(min_length=1)
    base_branch: str = "main"


class ProjectOut(BaseModel):
    id: str
    name: str
    repo_url: str
    stage: str
    profile: dict
    local_repo_path: str = ""
    base_branch: str = "main"


class ProfilePatch(BaseModel):
    profile: dict


class ThreadOut(BaseModel):
    id: str
    title: str


class MessageOut(BaseModel):
    role: str
    content: str


class ChatIn(BaseModel):
    content: str = Field(min_length=1)


def _project_out(p: Project) -> ProjectOut:
    return ProjectOut(
        id=p.id, name=p.name, repo_url=p.repo_url, stage=p.stage,
        profile=p.profile or {},
        local_repo_path=p.local_repo_path or "",
        base_branch=p.base_branch or "main",
    )


def _get_project(session: Session, project_id: str) -> Project:
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(404, "project not found")
    return project


# --- projects ----------------------------------------------------------------

@app.post("/api/projects", response_model=ProjectOut)
def create_project(body: ProjectCreate, session: Session = Depends(get_session)) -> ProjectOut:
    # No auto-scan here: scanning clones the repo, and the clone location is the
    # user-selected folder chosen on the Scan button. Cloning at creation would
    # land in the default workspace inside this agent repo, which we never want.
    project = Project(
        id=services.new_id(),
        name=body.name,
        repo_url=body.repo_url,
        pat_encrypted=encrypt(body.pat),   # PAT never stored or logged in plaintext
        base_branch=body.base_branch,
    )
    session.add(project)
    session.commit()
    return _project_out(project)


@app.get("/api/projects", response_model=List[ProjectOut])
def list_projects(session: Session = Depends(get_session)) -> List[ProjectOut]:
    return [_project_out(p) for p in session.query(Project).order_by(Project.created_at.desc())]


class ScanIn(BaseModel):
    local_path: str = ""
    branch: str = ""


@app.delete("/api/projects/{project_id}")
def delete_project(project_id: str, session: Session = Depends(get_session)) -> dict:
    project = _get_project(session, project_id)
    session.delete(project)
    session.commit()
    return {"deleted": project_id}


@app.get("/api/projects/{project_id}/branches")
def project_branches(project_id: str, session: Session = Depends(get_session)) -> dict:
    """List branches for a project using its stored (encrypted) PAT."""
    from src.app.crypto import decrypt
    from src.tools.azdo_tools import list_branches_for_url
    project = _get_project(session, project_id)
    try:
        pat = decrypt(project.pat_encrypted)
        branches = list_branches_for_url(project.repo_url, pat)
        return {"branches": branches}
    except Exception as exc:
        raise HTTPException(400, str(exc)[:300])


@app.post("/api/projects/{project_id}/scan")
def scan(project_id: str, body: ScanIn | None = None,
         session: Session = Depends(get_session)) -> dict:
    project = _get_project(session, project_id)
    if body and body.branch:
        project.base_branch = body.branch
        session.commit()
    try:
        return services.scan_project(session, project,
                                     local_path=(body.local_path if body else ""))
    except Exception as exc:
        logger.exception("scan failed")
        raise HTTPException(500, f"scan failed: {str(exc)[:300]}")


@app.get("/api/browse")
def browse_folder() -> dict:
    """Open a native OS folder-picker dialog and return the selected path."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askdirectory(title="Select folder to clone repository into")
        root.destroy()
        return {"path": path or ""}
    except Exception as exc:
        raise HTTPException(500, f"folder dialog failed: {exc}")


class BranchesIn(BaseModel):
    repo_url: str
    pat: str


@app.post("/api/repos/branches")
def get_branches(body: BranchesIn) -> dict:
    """List branches for an AzDO repo URL using the provided PAT."""
    try:
        from src.tools.azdo_tools import list_branches_for_url
        branches = list_branches_for_url(body.repo_url, body.pat)
        return {"branches": branches}
    except Exception as exc:
        raise HTTPException(400, str(exc)[:300])


@app.post("/api/projects/{project_id}/evaluate")
def evaluate(project_id: str, session: Session = Depends(get_session)) -> dict:
    project = _get_project(session, project_id)
    try:
        return services.evaluate_project(session, project)
    except FileNotFoundError:
        raise HTTPException(400, "no knowledge graph — run a scan first")
    except Exception as exc:
        logger.exception("evaluation failed")
        raise HTTPException(500, f"evaluation failed: {str(exc)[:300]}")


class GenerateIn(BaseModel):
    capabilities: Optional[List[str]] = None


@app.post("/api/projects/{project_id}/generate")
def generate(project_id: str, body: GenerateIn | None = None,
             session: Session = Depends(get_session)) -> dict:
    project = _get_project(session, project_id)
    try:
        return services.generate_project(
            session, project, capabilities=(body.capabilities if body else None))
    except Exception as exc:
        logger.exception("generation failed")
        raise HTTPException(500, f"generation failed: {str(exc)[:300]}")


@app.get("/api/projects/{project_id}/contract")
def get_contract(project_id: str, session: Session = Depends(get_session)) -> dict:
    """The code checklist: detected/declared user-ML-code stages for the endpoint strategy."""
    project = _get_project(session, project_id)
    try:
        return services.get_contract(session, project)
    except Exception as exc:
        logger.exception("contract fetch failed")
        raise HTTPException(500, f"contract fetch failed: {str(exc)[:300]}")


class ContractIn(BaseModel):
    contract: dict


@app.put("/api/projects/{project_id}/contract")
def save_contract(project_id: str, body: ContractIn,
                  session: Session = Depends(get_session)) -> dict:
    """Persist the user-confirmed checklist (drives wired/adapter/scaffold at generation)."""
    project = _get_project(session, project_id)
    try:
        return services.save_contract(session, project, body.contract)
    except Exception as exc:
        logger.exception("contract save failed")
        raise HTTPException(400, f"contract save failed: {str(exc)[:300]}")


@app.get("/api/projects/{project_id}/requirement-plan")
def requirement_plan(project_id: str, session: Session = Depends(get_session)) -> dict:
    """Backward dependency plan for the endpoint strategy: requirement · owner · status."""
    project = _get_project(session, project_id)
    try:
        return services.get_requirement_plan(session, project)
    except Exception as exc:
        logger.exception("requirement plan failed")
        raise HTTPException(500, f"requirement plan failed: {str(exc)[:300]}")


@app.get("/api/projects/{project_id}/generation-report")
def generation_report(project_id: str, session: Session = Depends(get_session)) -> dict:
    """Latest stored generation report — what was generated, persisted across reloads."""
    project = _get_project(session, project_id)
    try:
        return services.get_generation_report(session, project)
    except Exception as exc:
        logger.exception("generation report fetch failed")
        raise HTTPException(500, f"generation report fetch failed: {str(exc)[:300]}")


@app.post("/api/projects/{project_id}/validate")
def validate(project_id: str, session: Session = Depends(get_session)) -> dict:
    project = _get_project(session, project_id)
    try:
        return services.validate_project(session, project)
    except Exception as exc:
        logger.exception("validation failed")
        raise HTTPException(500, f"validation failed: {str(exc)[:300]}")


class CommitIn(BaseModel):
    message: str = ""


@app.post("/api/projects/{project_id}/commit")
def commit(project_id: str, body: CommitIn | None = None,
           session: Session = Depends(get_session)) -> dict:
    project = _get_project(session, project_id)
    try:
        return services.commit_project(session, project, message=(body.message if body else ""))
    except Exception as exc:
        logger.exception("commit failed")
        raise HTTPException(500, f"commit failed: {str(exc)[:300]}")


@app.post("/api/projects/{project_id}/register-pipelines")
def register_pipelines_endpoint(project_id: str, session: Session = Depends(get_session)) -> dict:
    project = _get_project(session, project_id)
    try:
        return services.register_pipelines(session, project)
    except Exception as exc:
        logger.exception("pipeline registration failed")
        raise HTTPException(500, f"registration failed: {str(exc)[:300]}")


@app.patch("/api/projects/{project_id}/profile", response_model=ProjectOut)
def patch_profile(project_id: str, body: ProfilePatch, session: Session = Depends(get_session)) -> ProjectOut:
    project = _get_project(session, project_id)
    merged = dict(project.profile or {})
    merged.update(body.profile)
    project.profile = merged
    session.commit()
    return _project_out(project)


@app.get("/api/projects/{project_id}/reports")
def get_reports(project_id: str, kind: Optional[str] = None, session: Session = Depends(get_session)) -> list:
    _get_project(session, project_id)
    query = session.query(Report).filter(Report.project_id == project_id)
    if kind:
        query = query.filter(Report.kind == kind)
    return [
        {"kind": r.kind, "payload": r.payload, "created_at": r.created_at.isoformat()}
        for r in query.order_by(Report.created_at.desc())
    ]


# --- threads + chat ----------------------------------------------------------

@app.get("/api/projects/{project_id}/threads", response_model=List[ThreadOut])
def list_threads(project_id: str, session: Session = Depends(get_session)) -> List[ThreadOut]:
    _get_project(session, project_id)
    threads = (
        session.query(Thread)
        .filter(Thread.project_id == project_id)
        .order_by(Thread.created_at.desc())
    )
    return [ThreadOut(id=t.id, title=t.title) for t in threads]


@app.post("/api/projects/{project_id}/threads", response_model=ThreadOut)
def create_thread(project_id: str, session: Session = Depends(get_session)) -> ThreadOut:
    _get_project(session, project_id)
    thread = Thread(id=services.new_id(), project_id=project_id)
    session.add(thread)
    session.commit()
    return ThreadOut(id=thread.id, title=thread.title)


@app.delete("/api/threads/{thread_id}")
def delete_thread(thread_id: str, session: Session = Depends(get_session)) -> dict:
    thread = session.get(Thread, thread_id)
    if thread:
        session.delete(thread)
        session.commit()
    return {"deleted": thread_id}


@app.get("/api/threads/{thread_id}/messages", response_model=List[MessageOut])
def get_messages(thread_id: str, session: Session = Depends(get_session)) -> List[MessageOut]:
    thread = session.get(Thread, thread_id)
    if not thread:
        raise HTTPException(404, "thread not found")
    return [MessageOut(role=m.role, content=m.content) for m in thread.messages]


@app.post("/api/threads/{thread_id}/messages", response_model=MessageOut)
def post_message(thread_id: str, body: ChatIn, session: Session = Depends(get_session)) -> MessageOut:
    thread = session.get(Thread, thread_id)
    if not thread:
        raise HTTPException(404, "thread not found")
    try:
        reply = services.chat_turn(session, thread, body.content)
        return MessageOut(role="assistant", content=reply)
    except Exception as exc:
        logger.exception("chat turn failed")
        raise HTTPException(500, str(exc)[:500])


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


@app.post("/api/threads/{thread_id}/messages/stream")
def post_message_stream(thread_id: str, body: ChatIn) -> StreamingResponse:
    """Stream agent steps (thinking, tool calls, observations) then the final reply as SSE.

    Uses its own DB session: the request returns immediately with StreamingResponse, so a
    dependency-injected session would already be closed before the generator runs.
    """
    from src.app.db import SessionLocal

    def gen():
        with SessionLocal() as s:
            thread = s.get(Thread, thread_id)
            if not thread:
                yield _sse({"type": "error", "message": "thread not found"})
                return
            try:
                for event in services.chat_turn_stream(s, thread, body.content):
                    yield _sse(event)
            except Exception as exc:
                logger.exception("stream chat failed")
                yield _sse({"type": "error", "message": str(exc)[:300]})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/projects/{project_id}/infra-check")
def infra_check(project_id: str, session: Session = Depends(get_session)) -> dict:
    """Run all six Azure + AzDO prerequisite checks for a project.

    Project profile values are used first (set by auto-discovery), falling back to .env.
    """
    from src.app.project_context import activate
    from src.tools.azure_infra_tools import check_all_prerequisites
    project = _get_project(session, project_id)
    activate(project)
    try:
        report = check_all_prerequisites(profile_overrides=dict(project.profile or {}))
        return report.model_dump()
    except Exception as exc:
        logger.exception("infra check failed")
        raise HTTPException(500, f"infra check failed: {str(exc)[:300]}")


@app.get("/api/projects/{project_id}/verify-data-paths")
def verify_data_paths_endpoint(project_id: str, session: Session = Depends(get_session)) -> dict:
    """Validate the project's configured data-plane paths (format + best-effort blob existence)."""
    from src.app.project_context import activate
    from src.tools.data_path_tools import verify_data_paths
    project = _get_project(session, project_id)
    activate(project)
    try:
        report = verify_data_paths(dict(project.profile or {}))
        return report.model_dump()
    except Exception as exc:
        logger.exception("data path verification failed")
        raise HTTPException(500, f"data path verification failed: {str(exc)[:300]}")


@app.post("/api/projects/{project_id}/infra-discover")
def infra_discover(project_id: str, session: Session = Depends(get_session)) -> dict:
    """Auto-discover Azure config from AzDO service connections.

    Saves discovered values to the project profile, then re-runs the infra check
    using the updated profile so the caller gets fresh results in one round-trip.
    """
    from src.app.project_context import activate
    from src.tools.azure_infra_tools import auto_discover_from_azdo, check_all_prerequisites
    project = _get_project(session, project_id)
    activate(project)
    try:
        discovered = auto_discover_from_azdo()

        # Persist only the fields that were actually discovered (don't overwrite existing)
        _KEY_MAP = {
            "azure_subscription_id": "subscription_id",
            "azure_resource_group":  "resource_group",
            "acr_name":              "acr_name",
            "aml_workspace":         "aml_workspace",
            "aml_compute_target":    "aml_compute_target",
        }
        if discovered.discovered:
            merged = dict(project.profile or {})
            for profile_key, disc_field in _KEY_MAP.items():
                val = getattr(discovered, disc_field, "")
                if val and profile_key in discovered.discovered:
                    merged[profile_key] = val
            project.profile = merged
            session.commit()

        report = check_all_prerequisites(profile_overrides=dict(project.profile or {}))
        return {"discovered": discovered.model_dump(), "report": report.model_dump()}
    except Exception as exc:
        logger.exception("infra discover failed")
        raise HTTPException(500, f"infra discover failed: {str(exc)[:300]}")


@app.get("/api/projects/{project_id}/pipeline-readiness")
def pipeline_readiness(project_id: str, session: Session = Depends(get_session)) -> dict:
    """Check readiness of all pipeline stages in dependency order."""
    from src.app.project_context import activate
    from src.tools.pipeline_readiness import check_pipeline_readiness
    project = _get_project(session, project_id)
    activate(project)
    report = check_pipeline_readiness(
        repo_path=project.local_repo_path or "",
        profile=dict(project.profile or {}),
    )
    return report.model_dump()


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}
