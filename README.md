# MLOps Onboarding & Remediation Platform

Capability-driven agentic platform that onboards any Azure ML repository:
scans it into a knowledge graph, audits its MLOps maturity against an
enterprise capability ontology, generates missing assets from templates,
validates everything locally (zero failed cloud runs), self-heals, and — only
after your confirmation — commits and operates the pipelines.

Human-in-the-loop by design: generation, commits, and **every** Azure DevOps
pipeline trigger require explicit approval in chat.

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env          # GITHUB_TOKEN (LLM), PLATFORM_SECRET_KEY, optional Azure config

# backend
python -m uvicorn src.app.main:app --port 8000
# frontend
cd web && npm install && npm run dev   # http://localhost:5173
```

1. **Projects page** — register a project (AzDO repo URL + PAT, stored encrypted), hit **Scan**
2. Review/edit the auto-inferred profile (project type, target, metrics, endpoint, drift)
3. **Chat page** — threads are scoped per project (dropdown). Drive the workflow:
   - "evaluate the capabilities" → maturity scores + gap report
   - "generate the missing assets" → template-based generation (working tree only)
   - "validate" → 4-tier local validation + self-healing
   - "commit" → gated; requires `validated_local` + your explicit confirmation
   - "register the pipelines" → AzDO definitions created (no runs)
   - "trigger <pipeline>" → agent restates and waits for your approval, every time

## Architecture (v2)

```
React (Projects + threaded Chat)
  └─ FastAPI app layer ── SQLite (projects, threads, messages, reports, failure memory)
       └─ workflow stages: configured → scanned → evaluated → generated
                           → validated_local → committed → operational
            ├─ understanding/  knowledge graph (AST + asset edges) + blueprint
            ├─ catalog/        capability ontology (9 capabilities, YAML)
            ├─ evaluation/     evidence → LLM judgment → deterministic scoring
            ├─ generation/     Jinja2 template library (29 files) + bounded LLM adaptation
            ├─ validation/     static → py-compile → AML schema → AzDO previewRun
            ├─ healing/        RCA → fix → re-validate (bounded, memory-backed)
            └─ tools/          git, files, AzDO REST, AML SDK, validators (stateless)
```

Key properties:
- **Capability-driven, not file-driven** — capabilities are inferred from graph
  evidence; "AzDO orchestrates, AML executes" is enforced via graph `submits` edges
- **Token-bounded** — the LLM sees graph digests and evidence excerpts, never the repo
- **Deterministic scoring** — LLM judges evidence; weights/statuses are pure code
- **Validate-local-first** — AzDO `previewRun` parses pipelines server-side with
  zero runs created; org-setup tasks (approval environments, service connections)
  are reported separately from file defects
- **Observability** — Arize Phoenix traces every graph node and LLM call
  (`python -m phoenix.server.main serve`, http://localhost:6006)

Design docs: [docs/ARCHITECTURE_V2.md](docs/ARCHITECTURE_V2.md) ·
[docs/REQUIREMENTS.md](docs/REQUIREMENTS.md)

## Tests

```bash
pytest src/tests -q   # 59 tests
```
