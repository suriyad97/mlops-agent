# Product Requirements — MLOps Remediation Platform v2

Status: capturing — user is still adding requirements. Do not implement until confirmed.

## R1 — Human-in-the-loop workflow (no autonomous PR)

The platform must NOT raise PRs or commit on its own. Staged flow, each stage gated by the user from chat:

1. Clone repo locally
2. Scan it → **publish gap report to the user**: what's missing, what an MLOps repo should have
3. User commands generation from chat → agent generates files
4. Confirmation loop: keep generating until ALL required files are generated
5. Show final generation report → ask the user "anything missing?"

## R2 — Local validation before commit

After generation, test everything FROM LOCAL before any commit, to avoid failed pipeline runs polluting AzDO:

- Run/submit pipelines from local using **az CLI**:
  - Environment creation
  - Training pipeline
  - CD pipelines (deployment)
  - CT pipelines (continuous training)
- Only when ALL of these pass → commit to the repo

## R3 — Chat application structure

- **Conversation threads**: ChatGPT-like session creation/switching (sidebar)
- **Two pages**:
  1. **Project setup page**: user enters project details — AzDO repo URL, PAT.
     A single **Scan** action: clones, builds the knowledge graph (graphify-style),
     and auto-fills project characteristics:
     - project type (regression / classification / …)
     - metrics used
     - target variable
     - endpoint type (real-time / batch)
     - drift monitoring (data drift / concept drift)
     - (etc. — user-editable after auto-fill)
  2. **Chat page**: threaded conversations driving the staged workflow (R1, R2)

## Decisions (answered by user)

- **PAT**: entered with the repo URL in the chat interface, per project
- **Commit gate**: explicit confirmation in chat after all local validation passes
- **Threads**: scoped per project; project dropdown in chat loads only that project's
  threads; new project → fresh thread space; stored in local SQLite
- **AzDO local validation**: preview/validate API accepted (implied by "Azure DevOps
  Validation" stage in final spec)

## R4 — Final architecture spec (v2, supersedes v1 design)

Full spec provided 2026-06-10. Key points:

- **Capability-driven, NOT file-driven**: infer business capabilities from evidence
  ("can this repo train a model?"), never deterministic filename checks. Must work
  across arbitrary naming conventions.
- **9 enterprise capabilities**: repository_discovery, environment_lifecycle, training,
  model_lifecycle, batch_deployment, realtime_deployment, monitoring, retraining, rollback
- **Capability catalog** = enterprise ontology (purpose, dependencies, required functions,
  implementation options, AML/AzDO assets, validation rules, deliverables, evidence
  patterns, relationships). Business knowledge, not prompts.
- **Capability evaluator**: evidence extraction → weighted scoring → complete/partial/missing
  + structured gap output
- **Environment lifecycle** = build → push dev ACR → dev AML env → validate → promote QA →
  validate → promote Prod → validate (dependency-free root capability)
- **Template library**: 80% template adaptation / 20% LLM reasoning; templates per capability
- **Repository understanding**: file tree + dependency graph + code graph + asset analysis →
  Repository Blueprint; graph retrieval, never full repo to LLM (token control)
- **Validation-first**: generate → local validation → local execution → AML validation →
  AzDO validation (preview) → only then commit/push/PR
- **Self-healing**: logs → RCA → fix → patch → retry, with failure memory
- **Memory**: Project Memory (blueprint, reports, decisions), Failure Memory (failures/fixes/
  outcomes), Knowledge Memory (catalog, templates, validation rules, standards)
- **LangGraph**: supervisor + specialist agents (Repository Understanding, Capability
  Evaluation, Gap Analysis, Template Retrieval, Generation, Validation, Self-Healing,
  Commit, Deployment, Monitoring); checkpoints; human approval nodes
- **AzDO = orchestration, AML = execution**: capability incomplete if AML assets exist
  without corresponding AzDO execution pipelines
- **Project types**: binary/multiclass classification, regression, forecasting, NLP, CV, custom
- **Autonomy reconciliation**: autonomous within stages; human gates at (1) generation
  kickoff and (2) commit (per R1 + commit-gate decision)

See docs/ARCHITECTURE_V2.md for the full blueprint.

## R5 — Human approval for every AzDO pipeline (added 2026-06-10)

- EVERY Azure DevOps pipeline trigger requires explicit human approval — platform-level
  chat confirmation before any run is started, even when the user named the pipeline.
- Generated multi-stage pipelines (QA/Prod promotion) must use AzDO Environment
  approval checks so promotions also gate inside Azure DevOps itself.
