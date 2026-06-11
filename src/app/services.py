"""App services: project scan and project-scoped chat turns."""
import queue
import threading
import uuid
from pathlib import Path

from sqlalchemy.orm import Session

from src.app.db import Message, Project, Report, Thread
from src.app.project_context import activate
from src.shared.logging import get_logger

logger = get_logger(__name__)


def new_id() -> str:
    return uuid.uuid4().hex[:16]


class NoRepoPath(Exception):
    """Raised when an operation needs the project's cloned repo but none is set."""


def _repo_path(project: Project) -> str:
    """The project's working tree. Never falls back to cloning inside the agent repo."""
    path = project.local_repo_path
    if not path or not Path(path).is_dir():
        raise NoRepoPath(
            "no cloned repository for this project — click Scan and pick the folder "
            "containing the repository (e.g. the clone in your Downloads folder) first"
        )
    return path


def _graph_tools(project_id: str) -> list:
    """Knowledge-graph query tools, bound to this project's saved graph."""
    from src.shared.react import ReactTool
    from src.platform.understanding import retrieval

    def _load():
        return retrieval.load_graph(project_id)

    def digest() -> str:
        try:
            return retrieval.graph_digest(_load())
        except FileNotFoundError:
            return "No knowledge graph yet — run a scan from the Projects page first."

    def what(path: str) -> str:
        try:
            return retrieval.what_does(_load(), path)
        except FileNotFoundError:
            return "No knowledge graph yet — run a scan first."

    def refs(path: str) -> str:
        try:
            result = retrieval.who_references(_load(), path)
            if isinstance(result, str):
                return result    # FILE NOT FOUND + suggestions
            return ", ".join(result) or "the file exists but nothing references it"
        except FileNotFoundError:
            return "No knowledge graph yet — run a scan first."

    def latest_report(kind: str) -> str:
        import json
        from src.app.db import Report, SessionLocal
        with SessionLocal() as s:
            report = (
                s.query(Report)
                .filter(Report.project_id == project_id, Report.kind == kind)
                .order_by(Report.created_at.desc())
                .first()
            )
            if not report:
                return f"No '{kind}' report yet. Available kinds: blueprint, capability, gap. Run a scan/evaluation first."
            return json.dumps(report.payload)[:7000]

    def generate_assets(capabilities: str = ""):
        """Chat-gated generation (R1): streams per-component progress as thinking events."""
        import json
        from src.app.db import SessionLocal
        from src.config.settings import get_settings
        from src.platform.generation.engine import build_params, generate
        from src.platform.understanding.retrieval import graph_digest, load_graph

        wanted = [c.strip() for c in capabilities.split(",") if c.strip()] or None

        with SessionLocal() as s:
            project = s.get(Project, project_id)
            if not project:
                yield json.dumps({"error": "project not found"})
                return

            gap = (
                s.query(Report)
                .filter_by(project_id=project_id, kind="gap")
                .order_by(Report.created_at.desc())
                .first()
            )
            if not gap or not gap.payload.get("items"):
                yield json.dumps({"error": "no gap report — run an evaluation first"})
                return

            items = gap.payload["items"]
            if wanted:
                items = [i for i in items if i["capability"] in wanted]
            if not items:
                yield json.dumps({"error": f"no gap items for {wanted}"})
                return

            try:
                local_path = _repo_path(project)
            except NoRepoPath as exc:
                yield json.dumps({"error": str(exc)})
                return
            params = build_params(project.name, project.profile or {}, get_settings())
            try:
                repo_context = graph_digest(load_graph(project_id), max_chars=2500)
            except FileNotFoundError:
                repo_context = ""

        # Collect progress messages via a queue; generate() runs synchronously
        # but we yield thinking events between components via on_progress callback.
        _SENTINEL = object()
        progress_q: queue.Queue = queue.Queue()
        result_holder: list = []

        def _run():
            try:
                report = generate(
                    local_path, items, params,
                    repo_context=repo_context,
                    on_progress=lambda msg: progress_q.put({"type": "thinking", "content": msg}),
                )
                from src.app.db import SessionLocal as SL
                with SL() as s2:
                    p2 = s2.get(Project, project_id)
                    s2.add(Report(id=new_id(), project_id=project_id, kind="generation",
                                  payload=report.model_dump()))
                    if p2:
                        p2.stage = "generated"
                    s2.commit()
                result_holder.append(json.dumps(report.model_dump())[:7000])
            except Exception as exc:
                result_holder.append(json.dumps({"error": str(exc)[:300]}))
            finally:
                progress_q.put(_SENTINEL)

        t = threading.Thread(target=_run, daemon=True)
        t.start()

        # Drain the queue — yield thinking dicts in real time, then the result string
        while True:
            item = progress_q.get()
            if item is _SENTINEL:
                break
            yield item   # {"type": "thinking", "content": "..."}

        yield result_holder[0] if result_holder else json.dumps({"error": "no result"})


    def validate_assets() -> str:
        import json
        from src.app.db import Project, SessionLocal
        with SessionLocal() as s:
            result = validate_project(s, s.get(Project, project_id))
        return json.dumps(result)[:7000]

    def commit_assets(message: str = "") -> str:
        import json
        from src.app.db import Project, SessionLocal
        with SessionLocal() as s:
            result = commit_project(s, s.get(Project, project_id), message=message)
        return json.dumps(result)[:4000]

    def register_generated_pipelines() -> str:
        import json
        from src.app.db import Project, SessionLocal
        with SessionLocal() as s:
            result = register_pipelines(s, s.get(Project, project_id))
        return json.dumps(result)[:4000]

    def aml_job_logs(job_name: str) -> str:
        from src.tools.aml_tools import get_aml_job_logs
        try:
            return get_aml_job_logs(job_name, max_chars=6000)
        except Exception as exc:
            return f"log retrieval failed: {exc}"

    def submit_aml_pipeline(pipeline_yaml: str) -> str:
        """Submit an AML pipeline from a YAML inside THIS project's cloned repo."""
        import json
        from src.api import jobs
        from src.app.db import Project, SessionLocal
        from src.tools.aml_tools import run_aml_pipeline
        with SessionLocal() as s:
            project = s.get(Project, project_id)
            if not project:
                return json.dumps({"error": "project not found"})
            try:
                repo_path = _repo_path(project)
            except NoRepoPath as exc:
                return json.dumps({"error": str(exc)})
            activate(project)

        def _run():
            return run_aml_pipeline(repo_path, pipeline_yaml)
        job_id = jobs.start_job("aml_pipeline", _run, detail=pipeline_yaml)
        return json.dumps({"job_id": job_id, "status": "running",
                           "pipeline_yaml": pipeline_yaml, "repo_path": repo_path})

    return [
        ReactTool("register_generated_pipelines",
                  "Register the committed AzDO pipeline YAMLs as pipeline definitions in Azure "
                  "DevOps (creates definitions only — runs nothing). Requires stage 'committed'. "
                  "After this, the user can ask to trigger them (each trigger needs approval).",
                  register_generated_pipelines, {"type": "object", "properties": {}}),
        ReactTool("get_aml_job_logs",
                  "Download and return the log tail of an Azure ML job by name.",
                  aml_job_logs,
                  {"type": "object", "properties": {"job_name": {"type": "string"}}, "required": ["job_name"]}),
        ReactTool("submit_aml_pipeline",
                  "ACTION: submit an Azure ML pipeline job from a YAML path inside THIS project's "
                  "cloned repository (the folder you selected on Scan; background, may incur compute "
                  "cost). The path is relative to the repo root, e.g. 'aml/pipelines/retraining_pipeline.yml'.",
                  submit_aml_pipeline,
                  {"type": "object", "properties": {"pipeline_yaml": {"type": "string"}},
                   "required": ["pipeline_yaml"]}),
        ReactTool("validate_generated_assets",
                  "Run tiered validation on the generated files: static checks, python "
                  "compilation, AML schema loaders, AzDO previewRun (server-side parse, zero "
                  "runs). Self-healing fixes failures automatically (bounded). Run after "
                  "generation, before any commit.",
                  validate_assets, {"type": "object", "properties": {}}),
        ReactTool("commit_generated_assets",
                  "ACTION (HARD GATE): commit + push the validated files to the remediation "
                  "branch. Requires stage 'validated_local'. Call ONLY after the user explicitly "
                  "confirms the commit in a SEPARATE message — restate what will be committed "
                  "and wait for their yes first. Optional commit message.",
                  commit_assets,
                  {"type": "object", "properties": {"message": {"type": "string"}}}),
        ReactTool("generate_missing_assets",
                  "ACTION (user-gated): generate the missing MLOps assets from the latest gap report "
                  "using enterprise templates. Writes to the local working tree only — NOTHING is "
                  "committed. Call ONLY when the user explicitly asks to generate. Optional "
                  "'capabilities' = comma-separated subset (e.g. 'monitoring,retraining'). After it "
                  "returns, show the user the per-component report and ask whether anything is missing.",
                  generate_assets,
                  {"type": "object", "properties": {"capabilities": {"type": "string"}}}),
        ReactTool("get_report",
                  "Latest stored report for this project. kind: blueprint | capability | gap.",
                  latest_report,
                  {"type": "object", "properties": {"kind": {"type": "string"}}, "required": ["kind"]}),
        ReactTool("repo_graph_digest",
                  "Knowledge-graph digest of this project's repository: entry points, AML assets, AzDO pipelines, key code.",
                  digest, {"type": "object", "properties": {}}),
        ReactTool("what_does_file_do",
                  "Functions, imports and references of a repository file (from the knowledge graph).",
                  what, {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}),
        ReactTool("who_references_file",
                  "Which files import/reference/submit the given repository file.",
                  refs, {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}),
    ]


def scan_project(session: Session, project: Project, local_path: str = "") -> dict:
    """P2 scan: clone/use local path → knowledge graph → blueprint → profile auto-fill.

    If local_path is provided (user-selected folder), the repo is cloned/reused there
    and the path stored on the project. If a path is already stored, it is reused.
    Never clones into the agent repo's default workspace — raises NoRepoPath instead.
    Graph build is deterministic (zero LLM); profile inference is one
    structured LLM call grounded in the graph digest + evidence excerpts.
    """
    activate(project)
    from src.platform.understanding.blueprint import extract_blueprint
    from src.platform.understanding.graph_builder import build_graph
    from src.platform.understanding.retrieval import save_graph
    from src.tools.git_tools import clone_repo
    from git import Repo as GitRepo

    if local_path:
        # local_path is the user-selected workspace folder. If it already *is* a clone
        # of this repo (or contains one), reuse it; otherwise clone into it. Never clone
        # into the agent repo's default workspace.
        actual_path = clone_repo(project.repo_url, workspace=local_path)
        project.local_repo_path = actual_path
        session.commit()
        local_path = actual_path
    elif project.local_repo_path:
        local_path = project.local_repo_path
    else:
        raise NoRepoPath(
            "no folder selected — click Scan and pick the folder where the repository "
            "is cloned (e.g. your Downloads folder)"
        )

    # Checkout the project's target branch if specified
    branch = project.base_branch or "main"
    try:
        repo = GitRepo(local_path)
        if repo.active_branch.name != branch:
            if branch in repo.heads:
                repo.heads[branch].checkout()
            else:
                repo.git.checkout("-b", branch, f"origin/{branch}")
    except Exception:
        pass  # best-effort; scan continues on current branch
    graph = build_graph(local_path)
    save_graph(graph, project.id)
    blueprint = extract_blueprint(local_path, graph)

    session.add(Report(id=new_id(), project_id=project.id, kind="blueprint",
                       payload=blueprint.model_dump()))

    profile = dict(project.profile or {})
    inferred = blueprint.profile
    profile.update({
        "project_type": profile.get("project_type") or inferred.project_type,
        "target_variable": profile.get("target_variable") or inferred.target_variable,
        "metrics": profile.get("metrics") or inferred.metrics,
        "endpoint_strategy": profile.get("endpoint_strategy") or inferred.endpoint_strategy,
        "drift_kinds": profile.get("drift_kinds") or inferred.drift_kinds,
        "entry_points": [ep["path"] for ep in blueprint.entry_points][:5],
        "inference_confidence": inferred.confidence,
        "inference_reasoning": inferred.reasoning,
    })
    project.profile = profile
    project.stage = "scanned"
    session.commit()
    return {"profile": profile, "blueprint": blueprint.model_dump()}


def evaluate_project(session: Session, project: Project) -> dict:
    """P3: capability evaluation + gap analysis from the knowledge graph."""
    activate(project)
    from src.platform.evaluation.evaluator import evaluate, gap_analysis
    from src.platform.understanding.retrieval import load_graph

    graph = load_graph(project.id)
    capability_report = evaluate(graph)
    gap_report = gap_analysis(capability_report)

    session.add(Report(id=new_id(), project_id=project.id, kind="capability",
                       payload=capability_report.model_dump()))
    session.add(Report(id=new_id(), project_id=project.id, kind="gap",
                       payload=gap_report.model_dump()))
    project.stage = "evaluated"
    session.commit()
    return {"capability_report": capability_report.model_dump(),
            "gap_report": gap_report.model_dump()}


def generate_project(session: Session, project: Project,
                     capabilities: list | None = None) -> dict:
    """P4: template-based generation for the project's open gaps (user-gated).

    Writes to the cloned working tree only — committing is a later, separately
    gated stage (R1/R2).
    """
    activate(project)
    from src.config.settings import get_settings
    from src.platform.generation.engine import build_params, generate
    from src.platform.understanding.retrieval import graph_digest, load_graph
    from src.tools.git_tools import clone_repo

    gap = (
        session.query(Report)
        .filter(Report.project_id == project.id, Report.kind == "gap")
        .order_by(Report.created_at.desc())
        .first()
    )
    if not gap or not gap.payload.get("items"):
        return {"error": "no gap report with open items — run an evaluation first"}

    items = gap.payload["items"]
    if capabilities:
        items = [i for i in items if i["capability"] in capabilities]
        if not items:
            return {"error": f"no gap items for capabilities {capabilities}"}

    local_path = _repo_path(project)
    params = build_params(project.name, project.profile or {}, get_settings())
    try:
        repo_context = graph_digest(load_graph(project.id), max_chars=2500)
    except FileNotFoundError:
        repo_context = ""

    report = generate(local_path, items, params, repo_context=repo_context)

    session.add(Report(id=new_id(), project_id=project.id, kind="generation",
                       payload=report.model_dump()))
    project.stage = "generated"
    session.commit()
    return report.model_dump()


def _latest_generated_files(session: Session, project_id: str) -> list:
    generation = (
        session.query(Report)
        .filter(Report.project_id == project_id, Report.kind == "generation")
        .order_by(Report.created_at.desc())
        .first()
    )
    return (generation.payload.get("written_files") if generation else []) or []


def validate_project(session: Session, project: Project) -> dict:
    """P5: tiered validation of generated files + bounded self-healing (R2)."""
    activate(project)
    from src.app.db import FailureRecord
    from src.platform.healing.healer import heal
    from src.platform.validation.framework import run_validation
    from src.tools.git_tools import clone_repo, current_branch

    files = _latest_generated_files(session, project.id)
    if not files:
        return {"error": "no generated files to validate — run generation first"}

    local_path = _repo_path(project)
    branch = current_branch(local_path)

    def record_failure(signature: str, rca: str, fix: dict, outcome: str) -> None:
        session.add(FailureRecord(id=new_id(), project_id=project.id,
                                  signature=signature, rca=rca, fix=fix, outcome=outcome))
        session.commit()

    report = run_validation(local_path, files, branch=branch)
    if not report.passed:
        report = heal(
            local_path, report,
            revalidate=lambda: run_validation(local_path, files, branch=branch),
            record_failure=record_failure,
        )

    session.add(Report(id=new_id(), project_id=project.id, kind="validation",
                       payload=report.model_dump()))
    project.stage = "validated_local" if report.passed else "validation_failed"
    session.commit()
    return report.model_dump()


def commit_project(session: Session, project: Project, message: str = "") -> dict:
    """P5 commit gate: only after local validation passed AND user confirmed in chat."""
    if project.stage != "validated_local":
        return {"error": f"commit blocked — stage is '{project.stage}', requires 'validated_local'. "
                         "Run validation (and fix failures) first."}
    activate(project)
    from src.config.settings import get_settings
    from src.tools.git_tools import clone_repo, create_branch, git_commit, git_push

    files = _latest_generated_files(session, project.id)
    validation = (
        session.query(Report)
        .filter(Report.project_id == project.id, Report.kind == "validation")
        .order_by(Report.created_at.desc())
        .first()
    )
    healed = (validation.payload.get("healed_files") if validation else []) or []
    to_commit = sorted({*files, *(h for h in healed if not h.startswith("NOT RETRIABLE"))})

    local_path = _repo_path(project)
    branch = get_settings().remediation_branch
    create_branch(local_path, branch)
    commit_hash = git_commit(
        local_path, to_commit,
        message or "feat: add MLOps capability assets (validated locally) via platform",
    )
    git_push(local_path, branch)
    project.stage = "committed"
    session.commit()
    return {"commit_hash": commit_hash, "branch": branch, "files": to_commit}


def azdo_pipeline_files(files: list) -> list:
    """Generated files that are AzDO pipeline definitions."""
    return [f for f in files
            if f == "azure-pipelines.yml" or (f.startswith("azdopipelines/") and f.endswith((".yml", ".yaml")))]


def register_pipelines(session: Session, project: Project) -> dict:
    """P6: register the committed AzDO pipeline YAMLs as pipeline definitions.

    Creates definitions only — never runs anything (R5: every trigger needs
    explicit human approval in chat).
    """
    if project.stage not in ("committed", "operational"):
        return {"error": f"register blocked — stage is '{project.stage}', requires 'committed'. "
                         "Commit the validated files first."}
    activate(project)
    from pathlib import Path

    from src.tools.azdo_tools import create_azure_pipeline

    files = azdo_pipeline_files(_latest_generated_files(session, project.id))
    if not files:
        return {"error": "no generated AzDO pipeline files to register"}

    repo_name = Path(project.repo_url).stem
    registered = []
    for rel in files:
        name = f"{Path(rel).stem}"
        pipeline_id = create_azure_pipeline(name=name, repo_name=repo_name, yaml_path=rel)
        registered.append({"name": name, "pipeline_id": pipeline_id, "yaml": rel})

    session.add(Report(id=new_id(), project_id=project.id, kind="operations",
                       payload={"registered_pipelines": registered}))
    project.stage = "operational"
    session.commit()
    return {"registered_pipelines": registered,
            "note": "definitions created; every run still requires your approval in chat"}


def chat_turn(session: Session, thread: Thread, user_text: str) -> str:
    """Persist the user message, run the project-scoped agent, persist the reply."""
    activate(thread.project)
    from src.api.chat_agent import chat

    session.add(Message(id=new_id(), thread_id=thread.id, role="user", content=user_text))
    session.commit()
    session.refresh(thread)

    history = [
        {"role": m.role, "content": m.content}
        for m in thread.messages
    ]
    context_note = (
        f"[project context] name={thread.project.name} repo={thread.project.repo_url} "
        f"stage={thread.project.stage} profile={thread.project.profile}"
    )
    history.insert(0, {"role": "user", "content": context_note})

    reply = chat(history, extra_tools=_graph_tools(thread.project.id))
    session.add(Message(id=new_id(), thread_id=thread.id, role="assistant", content=reply))

    if thread.title == "New thread" and len(thread.messages) <= 2:
        thread.title = user_text[:60]
    session.commit()
    return reply


def chat_turn_stream(session: Session, thread: Thread, user_text: str):
    """Streaming variant of chat_turn: yields agent events, persists the final reply."""
    activate(thread.project)
    from src.api.chat_agent import chat_stream

    session.add(Message(id=new_id(), thread_id=thread.id, role="user", content=user_text))
    session.commit()
    session.refresh(thread)

    history = [{"role": m.role, "content": m.content} for m in thread.messages]
    context_note = (
        f"[project context] name={thread.project.name} repo={thread.project.repo_url} "
        f"stage={thread.project.stage} profile={thread.project.profile}"
    )
    history.insert(0, {"role": "user", "content": context_note})

    final = ""
    for event in chat_stream(history, extra_tools=_graph_tools(thread.project.id)):
        if event.get("type") == "final":
            final = event.get("content", "")
        yield event


    session.add(Message(id=new_id(), thread_id=thread.id, role="assistant", content=final))
    if thread.title == "New thread" and len(thread.messages) <= 2:
        thread.title = user_text[:60]
    session.commit()

