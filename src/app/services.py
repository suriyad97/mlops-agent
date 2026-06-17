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


def load_or_propose_contract(project: Project, graph) -> "object":
    """Return the project's contract manifest — the stored (user-edited) one if present,
    otherwise a fresh proposal pre-filled from the knowledge graph."""
    from src.platform.understanding.contract import ContractManifest, propose_contract
    stored = (project.profile or {}).get("contract")
    if stored:
        try:
            return ContractManifest.model_validate(stored)
        except Exception:
            pass
    return propose_contract(graph)


def _build_generation_inputs(project: Project, graph):
    """Assemble everything generation needs for the golden-path (full standard set).

    Returns (items, params, contract, existing_pipeline_files, repo_context).
    Generation is driven by the strategy's complete standard set — NOT by a gap report.
    """
    from src.config.settings import get_settings
    from src.platform.evaluation.evaluator import standard_generation_set
    from src.platform.generation.engine import build_params
    from src.platform.understanding.retrieval import assets_by_role, graph_digest

    strategy = (project.profile or {}).get("endpoint_strategy", "both")
    items = [i.model_dump() for i in standard_generation_set(strategy).items]
    contract = load_or_propose_contract(project, graph).for_strategy(strategy).resolved()
    existing = assets_by_role(graph, "azdo_pipeline") + assets_by_role(graph, "aml_asset")
    params = build_params(project.name, project.profile or {}, get_settings())
    try:
        repo_context = graph_digest(graph, max_chars=2500)
    except Exception:
        repo_context = ""
    return items, params, contract, existing, repo_context


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
                return (f"No '{kind}' report yet. Available kinds: blueprint, capability, gap, generation. "
                        f"For what was generated use get_generation_report; for what's required use "
                        f"endpoint_requirement_plan (neither needs an evaluation).")
            return json.dumps(report.payload)[:7000]

    def generate_assets(capabilities: str = ""):
        """Chat-gated generation (R1): streams per-component progress as thinking events.

        Golden path: always generates the FULL standard set for the project's endpoint
        strategy, gated per user-ML-code stage by the contract manifest (wired/adapter/
        scaffold). Does not require a gap report.
        """
        import json
        from src.app.db import SessionLocal
        from src.platform.generation.engine import generate
        from src.platform.understanding.retrieval import load_graph

        wanted = [c.strip() for c in capabilities.split(",") if c.strip()] or None

        with SessionLocal() as s:
            project = s.get(Project, project_id)
            if not project:
                yield json.dumps({"error": "project not found"})
                return
            try:
                local_path = _repo_path(project)
            except NoRepoPath as exc:
                yield json.dumps({"error": str(exc)})
                return
            try:
                graph = load_graph(project_id)
            except FileNotFoundError:
                yield json.dumps({"error": "no knowledge graph — run a scan first"})
                return

            items, params, contract, existing, repo_context = _build_generation_inputs(project, graph)
            if wanted:
                items = [i for i in items if i["capability"] in wanted]
                contract = contract.model_copy(update={
                    "stages": [st for st in contract.stages if st.capability in wanted]})
            if not items:
                yield json.dumps({"error": f"no standard components for {wanted}"})
                return

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
                    contract=contract,
                    existing_pipeline_files=existing,
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


    def check_infra() -> str:
        """Run Azure + AzDO infrastructure prerequisite checks and return a markdown table."""
        from src.app.db import SessionLocal
        from src.tools.azure_infra_tools import check_all_prerequisites
        try:
            with SessionLocal() as s:
                p = s.get(Project, project_id)
                profile = dict(p.profile or {}) if p else {}
            report = check_all_prerequisites(profile_overrides=profile)
            return report.markdown()
        except Exception as exc:
            return f"Prerequisite check failed: {exc}"

    def check_readiness() -> str:
        """Check pipeline stage readiness in dependency order (CI→CT→CD→Monitoring→Retraining).
        Returns a table showing what's ready, what's blocking, and the recommended next step."""
        from src.app.db import SessionLocal
        from src.tools.pipeline_readiness import check_pipeline_readiness
        try:
            with SessionLocal() as s:
                p = s.get(Project, project_id)
                repo_path = (p.local_repo_path or "") if p else ""
                profile = dict(p.profile or {}) if p else {}
            report = check_pipeline_readiness(repo_path=repo_path, profile=profile)
            return report.markdown()
        except Exception as exc:
            return f"Readiness check failed: {exc}"

    def requirement_plan_tool() -> str:
        """Backward dependency plan for the project's endpoint strategy: what is required,
        who owns each piece (data scientist / platform / infra), and whether it is done.
        ALWAYS show this before generating, so the user sees their job vs the platform's."""
        from src.app.db import SessionLocal
        from src.platform.planning.requirement_plan import endpoint_requirement_plan
        from src.platform.understanding.retrieval import load_graph
        try:
            graph = load_graph(project_id)
        except FileNotFoundError:
            graph = None
        try:
            with SessionLocal() as s:
                p = s.get(Project, project_id)
                if not p:
                    return "project not found"
                repo_path = p.local_repo_path or ""
                strategy = (p.profile or {}).get("endpoint_strategy", "both")
                contract = (load_or_propose_contract(p, graph).for_strategy(strategy).resolved()
                            if graph is not None else None)
            plan = endpoint_requirement_plan(repo_path, strategy, contract)
            return plan.markdown()
        except Exception as exc:
            return f"Requirement plan failed: {exc}"

    def get_generation_plan_tool() -> str:
        """Grouped present-vs-missing plan for the project's endpoint strategy."""
        import json
        from src.app.db import Report as DBReport, SessionLocal
        from src.platform.evaluation.evaluator import CapabilityReport, generation_plan

        with SessionLocal() as s:
            project = s.get(Project, project_id)
            if not project:
                return "project not found"
            cap_row = (
                s.query(DBReport)
                .filter_by(project_id=project_id, kind="capability")
                .order_by(DBReport.created_at.desc())
                .first()
            )
            if not cap_row:
                return ("No capability report (golden-path generation doesn't need one). "
                        "For what's required and its status use endpoint_requirement_plan; "
                        "for what was already generated use get_generation_report.")
            cap_report = CapabilityReport.model_validate(cap_row.payload)
            strategy = (project.profile or {}).get("endpoint_strategy", "both")
            repo_path = project.local_repo_path or ""

        plan = generation_plan(cap_report, repo_path=repo_path, endpoint_strategy=strategy)
        return plan.markdown()

    def generation_report_tool() -> str:
        """What has actually been GENERATED for this project: files written, adapters,
        scaffolds to implement, user scripts reused (wired), and legacy files superseded.
        Golden-path generation needs NO evaluation or gap report — read this directly."""
        from src.app.db import Report as DBReport, SessionLocal
        with SessionLocal() as s:
            row = (
                s.query(DBReport)
                .filter_by(project_id=project_id, kind="generation")
                .order_by(DBReport.created_at.desc())
                .first()
            )
            if not row:
                return ("Nothing has been generated yet for this project. Run generation "
                        "(Readiness step → 'Generate pipelines', or ask me to 'generate the pipelines').")
            p = row.payload or {}

        def _fmt(items: list) -> str:
            return "\n".join(f"  - `{x}`" for x in items) or "  _(none)_"

        attention = p.get("needs_attention", [])
        gap_line = ("**Nothing is missing** — the full standard set for this strategy is in place. "
                    "Components that reuse your scripts are complete, not missing."
                    if not attention else
                    f"**{len(attention)} component(s) genuinely need attention** (no template / invalid):")
        return "\n".join([
            f"## Generation report\n\n{p.get('summary', '')}\n",
            gap_line, (_fmt(attention) if attention else ""),
            f"\n### Files written ({len(p.get('written_files', []))})", _fmt(p.get('written_files', [])),
            f"\n### Adapters — wrap your existing code ({len(p.get('adapter_files', []))})", _fmt(p.get('adapter_files', [])),
            f"\n### Scaffolds — you must implement these ({len(p.get('scaffold_files', []))})", _fmt(p.get('scaffold_files', [])),
            f"\n### Your scripts reused as-is / wired — COMPLETE ({len(p.get('wired_skipped', []))})", _fmt(p.get('wired_skipped', [])),
            f"\n### Superseded — review & delete ({len(p.get('superseded_files', []))})", _fmt(p.get('superseded_files', [])),
        ])

    def list_script_parameters() -> str:
        """List the parameters (argparse CLI args) each of the user's ML scripts defines —
        e.g. a training script's --register-threshold or --min-accuracy. Read from the
        knowledge graph (which captures every add_argument call). Use this to answer
        'what parameters do my scripts take / what can I tune'."""
        from src.app.db import SessionLocal as SL
        from src.platform.understanding.retrieval import load_graph
        try:
            graph = load_graph(project_id)
        except FileNotFoundError:
            return "No knowledge graph — run a scan first."
        with SL() as s:
            p = s.get(Project, project_id)
            if not p:
                return "project not found"
            profile = dict(p.profile or {})
            strategy = profile.get("endpoint_strategy", "both")
            contract = load_or_propose_contract(p, graph).for_strategy(strategy).resolved()
            overrides = profile.get("pipeline_params") or {}

        lines, seen = ["## Parameters your scripts define\n"], set()
        for st in contract.stages:
            path = st.detected_path or st.standard_path
            if not path or path in seen:
                continue
            seen.add(path)
            cli = graph.nodes.get(f"file:{path}", {}).get("cli_args", [])
            if cli:
                lines.append(f"- **{st.stage}** `{path}` → {', '.join(cli)}")
        if len(lines) == 1:
            lines.append("(no argparse parameters detected in the detected scripts)")
        lines.append("\n## Tunable pipeline parameters (baked into the generated pipelines)")
        defaults = {"drift_threshold": 0.2, "retrain_threshold": 0.25, "register_threshold": 0.0,
                    "optuna_trials": 20, "monitoring_cron": "0 6 * * *",
                    "instance_type": "Standard_DS2_v2", "instance_count": 1}
        for k, dv in defaults.items():
            cur = overrides.get(k, dv)
            lines.append(f"  - {k} = {cur}" + ("  (overridden)" if k in overrides else "  (default)"))
        lines.append("\nTo change one, ask e.g. \"set register_threshold to 0.85\" and I'll update it "
                     "(call set_pipeline_parameter), then re-generate to bake it in.")
        return "\n".join(lines)

    def set_pipeline_parameter(name: str, value: str) -> str:
        """Set/override a tunable pipeline parameter (drift_threshold, retrain_threshold,
        register_threshold, optuna_trials, monitoring_cron, instance_type, instance_count, …).
        Stored in the project profile; baked into the pipelines on the next generation."""
        from src.app.db import SessionLocal as SL
        numeric = {"drift_threshold", "retrain_threshold", "register_threshold",
                   "optuna_trials", "instance_count"}
        with SL() as s:
            p = s.get(Project, project_id)
            if not p:
                return "project not found"
            profile = dict(p.profile or {})
            pp = dict(profile.get("pipeline_params") or {})
            v = (value or "").strip()
            if name in numeric:
                try:
                    pp[name] = float(v) if "." in v else int(v)
                except ValueError:
                    return f"'{value}' is not a number — {name} expects a numeric value."
            else:
                pp[name] = v
            profile["pipeline_params"] = pp
            p.profile = profile
            s.commit()
        return f"✓ Set {name} = {pp[name]}. Re-generate the pipelines (Readiness → Generate) to apply it."

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

    def check_file_on_disk(path: str) -> str:
        """Directly inspect a file OR directory on disk in the project's cloned repo.

        Use this when the user says something is present but your knowledge graph
        shows it as missing. The filesystem is ground truth — trust it over the graph.
        - For a FILE: returns a content preview (first 40 lines).
        - For a DIRECTORY: returns a listing of its contents (files + subdirectories).
        - Otherwise: a clear 'not found' message.
        """
        from pathlib import Path
        from src.app.db import SessionLocal
        with SessionLocal() as s:
            p = s.get(Project, project_id)
            repo_path = (p.local_repo_path or "") if p else ""
        if not repo_path:
            return "Cannot check — repo not cloned yet. Run Scan first."
        full = Path(repo_path) / path
        if not full.exists():
            # Try case-insensitive search as a fallback
            parent = full.parent
            if parent.is_dir():
                matches = [f for f in parent.iterdir() if f.name.lower() == full.name.lower()]
                if matches:
                    full = matches[0]
                else:
                    return f"NOT FOUND on disk: {path}\nThe knowledge graph and filesystem agree — this does not exist. It needs to be generated."
            else:
                return f"NOT FOUND on disk: {path} (parent directory {parent} also missing)"
        # Directory: list contents (reading a dir as a file raises 'permission denied' on Windows)
        if full.is_dir():
            try:
                entries = sorted(full.iterdir(), key=lambda f: (f.is_file(), f.name.lower()))
            except Exception as exc:
                return f"DIRECTORY on disk: {path}/ (could not list: {exc})"
            if not entries:
                return f"DIRECTORY on disk: {path}/ is EMPTY."
            rel = Path(repo_path)
            listing = "\n".join(
                f"  {'📁' if e.is_dir() else '📄'} {e.relative_to(rel).as_posix()}{'/' if e.is_dir() else ''}"
                for e in entries
            )
            return f"DIRECTORY on disk: {path}/ — {len(entries)} entr{'y' if len(entries) == 1 else 'ies'}:\n{listing}"
        try:
            lines = full.read_text(encoding="utf-8", errors="replace").splitlines()
            preview = "\n".join(lines[:40])
            return f"FOUND on disk: {full}\n\n```\n{preview}\n{'...(truncated)' if len(lines) > 40 else ''}\n```"
        except Exception as exc:
            return f"EXISTS on disk: {full} (could not read: {exc})"

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
        ReactTool("endpoint_requirement_plan",
                  "Produce the REQUIREMENT PLAN for the project's endpoint strategy: resolves backward "
                  "from the endpoint through its full dependency chain and lists every requirement, who "
                  "owns it (🧑‍🔬 data scientist = ML code · 🤖 platform = auto-generated YAML/pipelines · "
                  "☁️ infra = cloud prerequisites), and whether it's done. ALWAYS call this and show it to "
                  "the user BEFORE generating assets, so they see exactly what is their job vs the "
                  "platform's, and what is still blocking a working endpoint.",
                  requirement_plan_tool, {"type": "object", "properties": {}}),
        ReactTool("check_pipeline_readiness",
                  "Check readiness of all pipeline stages in the MLOps dependency chain: "
                  "CI (environment build) → CT (training) → CD (deployment) → Monitoring → Retraining. "
                  "Each stage shows what's present, what's blocking, and what it will produce. "
                  "ALWAYS call this before recommending that the user trigger any pipeline — "
                  "it tells you whether prerequisites from earlier stages are satisfied. "
                  "For example, CD cannot run without a registered model (from CT), and CT cannot "
                  "run without an AML environment (from CI).",
                  check_readiness, {"type": "object", "properties": {}}),
        ReactTool("check_infrastructure_prerequisites",
                  "Scan whether all Azure + AzDO infrastructure prerequisites are in place before "
                  "any pipeline can run: Resource Group, ACR (Container Registry), AML Workspace, "
                  "AML Compute Cluster, AzDO→Azure RM service connection, AzDO→ACR service connection. "
                  "Returns a status table (✅ Ready / ⚠️ Not configured / ❌ Not found). "
                  "Call this FIRST whenever the user asks about running, triggering, or setting up "
                  "any pipeline — show the table so they know exactly what needs to be created.",
                  check_infra, {"type": "object", "properties": {}}),
        ReactTool("list_script_parameters",
                  "List the parameters your ML scripts define (argparse CLI args, e.g. a training "
                  "script's --register-threshold / --min-accuracy) AND the current tunable pipeline "
                  "parameters with their values. Use when the user asks what they can tune / what "
                  "parameters their scripts take.",
                  list_script_parameters, {"type": "object", "properties": {}}),
        ReactTool("set_pipeline_parameter",
                  "Set/override a tunable pipeline parameter (drift_threshold, retrain_threshold, "
                  "register_threshold, optuna_trials, monitoring_cron, instance_type, instance_count). "
                  "Persists to the project; re-generation bakes it into the pipelines. Call this when "
                  "the user asks to change a parameter, e.g. 'set register_threshold to 0.85'.",
                  set_pipeline_parameter,
                  {"type": "object", "properties": {"name": {"type": "string"}, "value": {"type": "string"}},
                   "required": ["name", "value"]}),
        ReactTool("get_generation_report",
                  "What has actually been GENERATED for this project: files written, adapters "
                  "(wrappers over the user's code), scaffolds the user must implement, the user's "
                  "scripts reused/wired, and legacy files superseded. ALWAYS use THIS to answer "
                  "'what was generated / created / what files / what's missing'. Golden-path "
                  "generation needs NO evaluation or gap report — never tell the user to evaluate "
                  "first for these questions.",
                  generation_report_tool, {"type": "object", "properties": {}}),
        ReactTool("get_generation_plan",
                  "Structured PRESENT-vs-MISSING plan (needs a capability report from an evaluation). "
                  "Prefer endpoint_requirement_plan (no evaluation needed) and get_generation_report "
                  "for golden-path projects.",
                  get_generation_plan_tool, {"type": "object", "properties": {}}),
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
        ReactTool("check_file_on_disk",
                  "Directly inspect a FILE or DIRECTORY in the project's cloned repo by checking the "
                  "filesystem — bypasses the knowledge graph entirely. Pass a file path to preview its "
                  "contents, or a DIRECTORY path (e.g. 'MLpipelines', 'src') to LIST what is inside it. "
                  "ALWAYS call this when the user says something is present that you believe is missing, "
                  "or when you need to see what a folder actually contains. The filesystem is ground "
                  "truth: if it returns FOUND/DIRECTORY, correct your earlier assessment. Never claim "
                  "'permission denied' for a folder — call this with the folder path to list it. "
                  "path is relative to the repo root, e.g. 'azdopipelines/ct-train.yml' or 'MLpipelines'.",
                  check_file_on_disk,
                  {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}),
        ReactTool("get_report",
                  "Latest stored report for this project. kind: blueprint | capability | gap | generation. "
                  "For 'what was generated' prefer get_generation_report; for 'what's required' prefer "
                  "endpoint_requirement_plan.",
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
    repo_path = project.local_repo_path or ""
    endpoint_strategy = (project.profile or {}).get("endpoint_strategy", "both")
    # repo_path enables filesystem self-correction inside _score()
    capability_report = evaluate(graph, repo_path=repo_path)
    # filter gap to capabilities on the path to the project's target endpoint
    gap_report = gap_analysis(capability_report, endpoint_strategy=endpoint_strategy)

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
    """P4: golden-path generation of the FULL standard MLOps block (user-gated).

    Always emits the complete standard set for the project's endpoint strategy, gated
    per user-ML-code stage by the contract manifest (wired/adapter/scaffold). Legacy
    non-standard pipelines are flagged as superseded for human-approved removal.
    Writes to the cloned working tree only — committing is a later, separately gated stage.
    """
    activate(project)
    from src.platform.generation.engine import generate
    from src.platform.understanding.retrieval import load_graph

    try:
        graph = load_graph(project.id)
    except FileNotFoundError:
        return {"error": "no knowledge graph — run a scan first"}

    items, params, contract, existing, repo_context = _build_generation_inputs(project, graph)
    if capabilities:
        items = [i for i in items if i["capability"] in capabilities]
        contract = contract.model_copy(update={
            "stages": [st for st in contract.stages if st.capability in capabilities]})
        if not items:
            return {"error": f"no standard components for capabilities {capabilities}"}

    local_path = _repo_path(project)
    report = generate(local_path, items, params, repo_context=repo_context,
                      contract=contract, existing_pipeline_files=existing)

    session.add(Report(id=new_id(), project_id=project.id, kind="generation",
                       payload=report.model_dump()))
    project.stage = "generated"
    session.commit()
    return report.model_dump()


# Stage key -> profile script-path field (so build_params / readiness see the resolved paths)
_STAGE_PROFILE_KEY = {
    "training": "train_script",
    "scoring_realtime": "score_script",
    "scoring_batch": "batch_score_script",
    "drift": "drift_script",
    "thresholds": "evaluate_thresholds_script",
}


def get_contract(session: Session, project: Project) -> dict:
    """Return the project's contract checklist — stored (user-edited) if present, else a
    fresh proposal pre-filled from the knowledge graph, filtered to the endpoint strategy."""
    activate(project)
    from src.platform.understanding.retrieval import load_graph

    try:
        graph = load_graph(project.id)
    except FileNotFoundError:
        return {"error": "no knowledge graph — run a scan first"}
    strategy = (project.profile or {}).get("endpoint_strategy", "both")
    contract = load_or_propose_contract(project, graph).for_strategy(strategy).resolved()
    return {"endpoint_strategy": strategy, "contract": contract.model_dump()}


def save_contract(session: Session, project: Project, manifest: dict) -> dict:
    """Persist the user-confirmed checklist into the project profile and mirror the
    resolved pipeline paths into the script-path fields used by generation/readiness."""
    activate(project)
    from src.platform.understanding.contract import ContractManifest

    cm = ContractManifest.model_validate(manifest).resolved()
    profile = dict(project.profile or {})
    profile["contract"] = cm.model_dump()
    for st in cm.stages:
        key = _STAGE_PROFILE_KEY.get(st.stage)
        if key:
            profile[key] = st.pipeline_path()
    project.profile = profile
    session.commit()
    return {"saved": True, "contract": cm.model_dump()}


def get_generation_report(session: Session, project: Project) -> dict:
    """The latest stored generation report (what was written/adapted/scaffolded/superseded),
    or an empty marker when nothing has been generated yet."""
    row = (
        session.query(Report)
        .filter(Report.project_id == project.id, Report.kind == "generation")
        .order_by(Report.created_at.desc())
        .first()
    )
    if not row:
        return {"generated": False}
    payload = dict(row.payload or {})
    payload["generated"] = True
    return payload


def get_requirement_plan(session: Session, project: Project) -> dict:
    """The endpoint requirement plan: backward dependency chain, ownership, tick status."""
    activate(project)
    from src.platform.planning.requirement_plan import endpoint_requirement_plan
    from src.platform.understanding.retrieval import load_graph

    try:
        graph = load_graph(project.id)
    except FileNotFoundError:
        graph = None
    strategy = (project.profile or {}).get("endpoint_strategy", "both")
    contract = (load_or_propose_contract(project, graph).for_strategy(strategy).resolved()
                if graph is not None else None)
    plan = endpoint_requirement_plan(project.local_repo_path or "", strategy, contract)
    return {"plan": plan.model_dump(), "markdown": plan.markdown()}


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


def project_state_card(session: Session, project: Project) -> str:
    """The COMMON project state every chat thread reads from, recomputed each turn and
    injected into the system prompt (so it never gets dropped by the message-window slice).

    Authoritative snapshot: profile + detected ML code (contract) + last generation +
    data-path config. Keeps every thread — new, old, short, long — grounded identically.
    """
    profile = dict(project.profile or {})
    lines = [
        "\n\n=== CURRENT PROJECT STATE (authoritative; refreshed every turn — trust this over "
        "your own assumptions, and use tools for deeper detail) ===",
        f"Project: {project.name} · stage: {project.stage} · "
        f"endpoint strategy: {profile.get('endpoint_strategy', 'unset')}",
        f"Repo: {project.repo_url}",
        f"ML profile: type={profile.get('project_type', '?')}, "
        f"target={profile.get('target_variable', '?')}, metrics={profile.get('metrics', [])}",
    ]

    # Your ML code — wired / adapter / scaffold (from the contract)
    try:
        from src.platform.understanding.retrieval import load_graph
        graph = load_graph(project.id)
        strategy = profile.get("endpoint_strategy", "both")
        contract = load_or_propose_contract(project, graph).for_strategy(strategy).resolved()
        code = []
        for st in contract.stages:
            where = f" → {st.detected_path}" if st.mode == "wired" else ""
            code.append(f"  - {st.stage}: {st.mode}{where}")
        if code:
            lines.append("Your ML code (contract):")
            lines.extend(code)
    except Exception:
        lines.append("Knowledge graph: not scanned yet (run a scan).")

    # Data paths configured (blob/datastore) — prerequisites
    dp = profile.get("data_paths") or {}
    if dp:
        lines.append("Data paths configured: " + ", ".join(sorted(dp.keys())))
    else:
        lines.append("Data paths: none configured yet.")

    # Last generation report — what exists / scaffolds / superseded
    gen = (
        session.query(Report)
        .filter(Report.project_id == project.id, Report.kind == "generation")
        .order_by(Report.created_at.desc())
        .first()
    )
    if gen and gen.payload:
        p = gen.payload
        lines.append(f"Last generation: {p.get('summary', '')}")
        attn = p.get("needs_attention", [])
        if attn:
            lines.append("  Needs attention (no template/invalid): " + ", ".join(attn))
        else:
            lines.append("  NOTE: components with no generated file because they REUSE the user's "
                         "scripts are COMPLETE — there are NO missing components.")
        if p.get("scaffold_files"):
            lines.append("  Scaffolds to implement: " + ", ".join(p["scaffold_files"]))
        if p.get("superseded_files"):
            lines.append("  Superseded (review & delete): " + ", ".join(p["superseded_files"][:12]))
    else:
        lines.append("Generation: nothing generated yet.")

    lines.append("=== END PROJECT STATE ===")
    return "\n".join(lines)


def chat_turn(session: Session, thread: Thread, user_text: str) -> str:
    """Persist the user message, run the project-scoped agent, persist the reply."""
    activate(thread.project)
    from src.api.chat_agent import chat

    session.add(Message(id=new_id(), thread_id=thread.id, role="user", content=user_text))
    session.commit()
    session.refresh(thread)

    history = [{"role": m.role, "content": m.content} for m in thread.messages]
    card = project_state_card(session, thread.project)

    reply = chat(history, extra_tools=_graph_tools(thread.project.id), system_suffix=card)
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
    card = project_state_card(session, thread.project)

    final = ""
    for event in chat_stream(history, extra_tools=_graph_tools(thread.project.id), system_suffix=card):
        if event.get("type") == "final":
            final = event.get("content", "")
        yield event


    session.add(Message(id=new_id(), thread_id=thread.id, role="assistant", content=final))
    if thread.title == "New thread" and len(thread.messages) <= 2:
        thread.title = user_text[:60]
    session.commit()

