# Architecture v2 — Capability-Driven MLOps Onboarding Platform

Implementation blueprint. Supersedes the v1 file-driven design.

---

## 1. Principles

1. **Capability-driven**: the unit of reasoning is a business capability ("can this repo
   train a model?"), never a filename. Files are evidence, not truth.
2. **Evidence-based scoring**: every capability is COMPLETE / PARTIAL / MISSING with a
   weighted score and named missing components.
3. **Template-first generation**: 80% adaptation of curated enterprise templates,
   20% LLM reasoning to specialize them to the repo.
4. **Validate locally, commit late**: nothing reaches the remote until static, local-exec,
   AML, and AzDO-preview validation all pass.
5. **Autonomous within stages, human-gated between**: two hard gates — generation
   kickoff and commit — implemented as LangGraph interrupts resumed from chat.
6. **Graph-based understanding**: the LLM never sees the whole repo; it sees a knowledge
   graph digest and queries the graph on demand.

---

## 2. System overview

```
┌─ React app ──────────────────────────────────────────────────┐
│  Page 1: Projects (repo URL + PAT, Scan, profile editor)     │
│  Page 2: Chat (project dropdown, threads sidebar, messages)  │
└──────────────┬───────────────────────────────────────────────┘
               │ REST (FastAPI)
┌──────────────▼───────────────────────────────────────────────┐
│  App layer: projects / threads / messages / workflow API     │
│  SQLite: projects, threads, messages, workflow_state,        │
│          reports, failure_memory                             │
└──────────────┬───────────────────────────────────────────────┘
               │ invoke / resume (thread_id = project workflow)
┌──────────────▼───────────────────────────────────────────────┐
│  LangGraph supervisor (SqliteSaver checkpointer)             │
│   Repository Understanding → Capability Evaluation →         │
│   Gap Analysis → ⏸ GATE(generate) → Template Retrieval →     │
│   Generation → Validation → Self-Healing ⟲ →                 │
│   ⏸ GATE(commit) → Commit → Deployment → Monitoring          │
└──────────────┬───────────────────────────────────────────────┘
               │ stateless tools
┌──────────────▼───────────────────────────────────────────────┐
│  git · file · graph (tree-sitter) · az cli · AzDO REST ·     │
│  AML SDK · validators · template engine                      │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. Folder structure

```
src/
├── app/                        # FastAPI application layer
│   ├── api/                    # routers: projects, threads, chat, workflow
│   ├── db/                     # SQLAlchemy models + migrations: projects, threads,
│   │                           #   messages, workflow_state, reports, failure_memory
│   ├── security/               # PAT encryption (Fernet), secret redaction
│   └── services/               # chat command router, workflow driver
├── platform/
│   ├── catalog/                # CAPABILITY CATALOG (YAML ontology, one file per capability)
│   ├── templates/              # TEMPLATE LIBRARY
│   │   ├── environment/  training/  model_lifecycle/  batch_deployment/
│   │   ├── realtime_deployment/  monitoring/  retraining/  rollback/
│   │   └── (each: template files + template.yaml manifest with parameters)
│   ├── understanding/          # repo graph builder, blueprint extractor, graph retrieval
│   ├── evaluation/             # capability evaluator, scoring, gap analysis
│   ├── generation/             # template engine + LLM adaptation
│   ├── validation/             # tiered validation framework
│   └── healing/                # RCA + patch + retry framework
├── graphs/                     # LangGraph: supervisor + agent subgraphs
│   ├── supervisor/             # state, routers, graph, interrupts
│   └── agents/                 # one package per specialist agent
├── tools/                      # stateless (extends v1: + graph_tools, az_cli_tools,
│                               #   template_tools, azdo preview-validate)
├── llm/  memory/  shared/  config/   # carried from v1, extended
└── tests/
web/                            # React: pages/Projects, pages/Chat, threads, dropdown
```

---

## 4. Capability catalog (the ontology)

One YAML per capability under `src/platform/catalog/`. Schema:

```yaml
# catalog/training.yaml
capability: training
purpose: Produce a trained, evaluated model artifact from data
depends_on: [environment_lifecycle]
required_functions:          # business functions, not files
  - data_loading
  - feature_engineering
  - model_training
  - model_evaluation
implementation_options:      # valid ways a repo may realize this
  - python_scripts + aml_pipeline + azdo_pipeline
  - notebook_converted_pipeline + azdo_pipeline
evidence:                    # SEMANTIC evidence specs consumed by evaluator
  - kind: code_semantic
    function: model_training
    indicators: [model.fit, Trainer, estimator training loops, sklearn/torch/lightgbm usage]
  - kind: aml_asset
    asset: pipeline_job_or_component
    role: training
  - kind: azdo_pipeline
    role: submits_training      # AzDO must ORCHESTRATE the AML asset
scoring:
  code_functions: 30
  aml_assets: 40
  azdo_orchestration: 30
deliverables:                # what generation must produce when missing
  - aml: components (train/evaluate), pipeline job
  - azdo: ct-train pipeline yaml
  - scripts: only if code functions missing
validation_rules:
  - static: aml schema check, azdo preview
  - local: submit training pipeline via az ml, expect Completed
relationships:
  enables: [model_lifecycle, retraining]
```

The 9 capabilities: `repository_discovery, environment_lifecycle, training,
model_lifecycle, batch_deployment, realtime_deployment, monitoring, retraining, rollback`.

**Catalog is data, not prompts.** The evaluator and generator read it; prompts reference
it by interpolation. Editing the ontology never means editing agent code.

Key encoded knowledge:
- environment_lifecycle = build → dev ACR → dev env → validate → QA → Prod (root, dependency-free)
- AzDO orchestrates, AML executes: AML assets WITHOUT an AzDO pipeline that submits them ⇒ PARTIAL
- model_lifecycle may be implemented purely in pipelines — never flag "missing .py" if
  registration/promotion happens in YAML
- monitoring: concept drift requires label availability — evaluator must check for it
- deployment strategy (batch vs realtime) inferred from project profile + endpoint evidence

---

## 5. Repository understanding framework

Pipeline (deterministic, local — tree-sitter; graphify CLI as optional accelerator):

```
clone → file tree → code graph (defs/calls/imports) → asset graph
  (AML YAML → scripts they invoke; AzDO YAML → AML assets they submit;
   configs → consumers) → dependency graph → REPOSITORY BLUEPRINT
```

**Blueprint** (Pydantic, stored in Project Memory):

```python
class RepositoryBlueprint(BaseModel):
    project_type: Literal["binary_classification","multiclass_classification",
                          "regression","forecasting","nlp","computer_vision","custom"]
    target_variable: str | None
    metrics: list[str]
    endpoint_strategy: Literal["realtime","batch","both","none"]
    drift_kinds: list[str]              # data / prediction / concept
    entry_points: list[EntryPoint]      # path, function, args
    aml_assets: list[AssetRef]          # kind, path, references
    azdo_pipelines: list[PipelineRef]   # path, what it submits/orchestrates
    graph_stats: GraphStats
```

Project profile fields (type, target, metrics, endpoint, drift) are inferred by the LLM
from **graph retrieval results** (entry-point signatures, evaluation code, dataset
columns touched), shown on Page 1 for the user to edit/confirm.

**Graph retrieval** (token control — repo never sent whole):
- `graph_digest(max_tokens)` — ranked by centrality
- `who_references(path)` / `what_does(path)` / `entry_point_signature(path)`
- `find_evidence(indicators: list[str])` — semantic grep over node summaries; this is
  what the capability evaluator calls per evidence spec

---

## 6. Capability evaluator

```
Blueprint + Graph → per-capability evidence collection (find_evidence + asset graph)
  → LLM judgment per evidence item (FOUND / WEAK / ABSENT, with citation into graph)
  → deterministic weighted scoring from catalog weights
  → CapabilityReport
```

```python
class CapabilityStatus(BaseModel):
    capability: str
    score: int                 # 0-100, weights from catalog
    status: Literal["complete","partial","missing"]
    evidence_found: list[EvidenceHit]    # each cites graph nodes (auditable)
    missing: list[str]                   # component names from catalog deliverables
class CapabilityReport(BaseModel):
    capabilities: list[CapabilityStatus]
    gap_summary: str
```

The LLM judges evidence; the **arithmetic is deterministic** — same evidence ⇒ same score.
Gap analysis = catalog deliverables minus found components, ordered by `depends_on`
(environment_lifecycle first, rollback last).

---

## 7. Template library + generation

Each template dir has a `template.yaml` manifest:

```yaml
name: ct-train-register
capability: training
files: [azdo/ct-train.yml.j2, aml/train_component.yml.j2]
parameters: [project_name, entry_point, entry_args, environment_name,
             compute_target, model_name, metrics]
constraints: [requires environment_lifecycle complete]
```

Generation per missing component:
1. **Template Retrieval Agent** picks the template via catalog mapping + blueprint
2. Jinja2 render with blueprint/profile parameters (the deterministic 80%)
3. **LLM adaptation pass** (the 20%): adjust entry-point args, metric names, column
   names — diff-reviewed against template to prevent drift from standards
4. Write to working tree (never committed yet)

---

## 8. Validation framework (tiered, fail-fast)

```
T1 static     yaml/schema lint, dockerfile rules, az ml validate (client-side schema)
T2 local exec python import checks, pytest if present, dry-run entry points
T3 AML        az cli submissions from local: env build, training pipeline,
              deployment (dev), monitoring job — expect terminal success
T4 AzDO       previewRun=true per generated pipeline (server parses/expands YAML,
              creates NO run)  ⇒ catches unauthorized variable groups, bad tasks
```

Validation rules come from the catalog per capability. Results aggregate into a
`ValidationReport` shown in chat. T3 runs against the **dev** environment only;
QA/Prod promotion happens post-commit through the generated AzDO pipelines.

---

## 9. Self-healing framework

Carried from v1 (proven live) and generalized:

```
failure → collect logs (az cli output / AML job logs / preview API errors)
        → RCA (ReAct over repo graph + logs; failure memory consulted for similar fixes)
        → FixBundle (complete file contents, only implicated files)
        → fix validation (static + reviewer + implicated-files constraint)
        → patch working tree → re-run failed tier only
bounded by max_retries per tier; exhaustion ⇒ surface to chat with RCA + manual steps
every cycle recorded to Failure Memory: {failure, root_cause, fix, outcome}
```

---

## 10. Memory design

| Store | Backing | Contents |
|---|---|---|
| Project Memory | SQLite (`projects`, `reports`) | blueprint, capability report, gap report, profile, architecture/deployment decisions |
| Failure Memory | SQLite (`failure_memory`) | failure signature, RCA, fix, outcome — retrieved by similarity during RCA |
| Knowledge Memory | repo files (versioned) | capability catalog, template library, validation rules, enterprise standards |

Knowledge Memory lives in git deliberately: ontology changes are code-reviewed.

---

## 11. LangGraph design

**State (supervisor):**

```python
class PlatformState(TypedDict, total=False):
    project_id: str
    repo_url: str; local_repo_path: str
    blueprint: dict; profile: dict
    capability_report: dict; gap_report: dict
    generation_plan: list          # ordered missing components
    generated_files: Annotated[list, operator.add]
    validation_report: dict
    heal_attempts: dict            # per-tier counters
    commit_hash: str; pr_url: str
    deployment_status: dict; monitoring_status: dict
    stage: str                     # mirrors workflow_state table
    awaiting: Literal["", "generate_approval", "commit_approval"]
```

**Agents (subgraphs):** Repository Understanding, Capability Evaluation, Gap Analysis,
Template Retrieval, Generation, Validation, Self-Healing, Commit, Deployment, Monitoring.

**Control flow:** supervisor router reads `stage` + reports; conditional edges:
- after gap analysis → `interrupt()` (generate gate) — graph checkpointed, chat shows report
- validation fail → Self-Healing → re-validate (bounded) → exhaustion → chat
- validation pass → `interrupt()` (commit gate)
- after commit → Deployment (trigger AzDO pipelines; they submit AML) → Monitoring

**Checkpointing:** `SqliteSaver`, `thread_id = project_id`. Chat commands resume
interrupts (`Command(resume=...)`). This is how "user says generate/commit in chat"
maps onto the graph — no polling, no bespoke state machine.

---

## 12. App layer (chat, threads, projects)

SQLite schema:

```
projects(id, name, repo_url, pat_encrypted, profile_json, stage, created_at)
threads(id, project_id→projects, title, created_at)
messages(id, thread_id→threads, role, content, created_at)
reports(id, project_id, kind, payload_json, created_at)      # blueprint/capability/gap/validation
failure_memory(id, project_id, signature, rca, fix_json, outcome, created_at)
```

API:

```
POST /api/projects                {name, repo_url, pat}        # PAT encrypted on write
POST /api/projects/{id}/scan                                   # clone+graph+blueprint+profile
GET  /api/projects/{id}/report                                 # capability/gap reports
PATCH /api/projects/{id}/profile                               # user edits auto-filled profile
GET/POST /api/projects/{id}/threads
GET/POST /api/threads/{id}/messages                            # POST = chat turn
POST /api/projects/{id}/workflow/resume   {gate, decision}     # generate/commit approvals
GET  /api/projects/{id}/workflow                               # stage + pending gate
```

Chat turns route through an intent layer: ops questions → tool-calling agent (v1 chat
agent, kept); workflow commands ("generate", "commit", "looks good") → graph resume.

**Frontend:** Page 1 Projects (form, Scan progress, editable profile card);
Page 2 Chat (project dropdown → loads that project's threads only; thread sidebar
with new/rename/delete; gap/validation reports rendered as cards; gate prompts
rendered as explicit approve buttons that POST resume).

---

## 13. Security design

- PAT encrypted at rest (Fernet; key from env `PLATFORM_SECRET_KEY`, not in repo)
- PAT never enters prompts, logs, or generated files; redaction filter on the logging
  formatter and on chat-visible tool outputs
- Per-project workspace isolation under `repos/{project_id}/`
- Audit trail: every gate decision + commit + deployment recorded in `reports`
- Generated assets never contain secrets — variable groups / service connections only

## 14. Scalability design

- Long operations (scan, validation, healing) run on a worker pool (ThreadPool now;
  the API stays responsive; jobs surfaced via the existing background-job registry)
- SQLite → Postgres is a connection-string swap (SQLAlchemy); SqliteSaver → PostgresSaver
- Stateless API + per-project graph threads ⇒ horizontal scaling when needed
- Production shape: containerized API+worker, static web build behind reverse proxy,
  or Azure Container Apps + Azure Files for workspaces

---

## 15. Phased delivery

| Phase | Scope | Builds on |
|---|---|---|
| **P1 Foundation** | SQLite + app layer (projects/threads/messages/PAT), two-page frontend, project dropdown + threads | v1 chat console |
| **P2 Understanding** | tree-sitter repo graph, blueprint extractor, profile auto-fill, Scan flow | P1 |
| **P3 Ontology** | capability catalog (9 YAMLs), evaluator + scoring, gap report in chat | P2 |
| **P4 Generation** | template library, retrieval + Jinja2 + LLM adaptation, generate gate | P3 |
| **P5 Validation+Healing** | tiered validation (az cli local, previewRun), self-healing loop, commit gate + commit | P4 |
| **P6 Operations** | deployment trigger, monitoring, retraining, rollback capabilities end-to-end | P5 |

Each phase is independently shippable and testable against `logistic_regression`.
