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

1. **Projects page** — register a project by providing an AzDO repo URL (with PAT) **OR** a local folder path (PAT bypass for local testing), hit **Scan**
2. Review/edit the auto-inferred profile (project type, target, metrics, endpoint, drift)
3. **Chat page** — threads are scoped per project (dropdown). Drive the workflow:
   - "evaluate the capabilities" → maturity scores + gap report
   - "generate the missing assets" → template-based generation (working tree only)
   - "validate" → 4-tier local validation + self-healing
   - "commit" → gated; requires `validated_local` + your explicit confirmation
   - "register the pipelines" → AzDO definitions created (no runs)
   - "trigger <pipeline>" → agent restates and waits for your approval, every time

## Architecture (v2)

### System components

```mermaid
flowchart TB
    subgraph client["React SPA — web/"]
        proj["Projects page<br/>register · scan · profile"]
        chat["Chat page<br/>threaded, SSE streaming"]
    end

    subgraph api["FastAPI — src/app/main.py"]
        rest["REST + SSE endpoints"]
        svc["services.py<br/>stage driver + project-scoped chat tools"]
    end

    subgraph platform["src/platform/ — the engine"]
        understanding["understanding<br/>knowledge graph + blueprint"]
        catalog["catalog<br/>8 capability YAMLs + implicit discovery"]
        evaluation["evaluation<br/>evidence → LLM judge → score"]
        generation["generation<br/>Jinja2 templates + bounded LLM adapt"]
        validation["validation<br/>4-tier, fail-fast"]
        healing["healing<br/>RCA → fix → re-validate"]
    end

    subgraph agent["Chat agent — src/api/chat_agent.py"]
        react["ReAct loop + SSE step streaming"]
        gtools["generic tools<br/>AzDO/AML read, trigger, watch_run"]
        ptools["project tools<br/>graph query, generate, validate, commit"]
    end

    tools["src/tools/ — stateless<br/>git · file · AzDO REST · AML SDK · validators"]
    db[("SQLite<br/>projects · threads · messages<br/>reports · failure_memory")]
    kg[(".graphs/PROJECT.json<br/>knowledge graph")]
    llm["GitHub Models LLM"]
    phx["Arize Phoenix :6006"]

    proj & chat --> rest --> svc
    svc --> platform
    svc --> agent
    agent --> tools
    platform --> tools
    understanding --> kg
    evaluation & generation --> llm
    react --> llm
    svc --- db
    react -.->|traces| phx
    evaluation -.->|traces| phx
```

### Workflow stages (the state machine)

The project `stage` column is the single source of truth. Each transition is a
service function; the two **human gates** (generate, commit) and every pipeline
trigger require explicit chat approval.

```mermaid
stateDiagram-v2
    [*] --> configured: create project
    configured --> scanned: scan — clone → graph → blueprint → profile
    scanned --> evaluated: evaluate — capability report + gap report
    evaluated --> generated: GATE generate (user-confirmed)
    generated --> validated_local: validate — 4-tier
    generated --> validation_failed: validate fails
    validation_failed --> validated_local: self-heal + re-validate
    validated_local --> committed: GATE commit (user-confirmed)
    committed --> operational: register AzDO pipeline definitions
    operational --> operational: trigger run (approval EVERY time) → watch_pipeline_run
```

### Knowledge graph (repository understanding)

`understanding/graph_builder.py` builds a `networkx` DiGraph from the repo — no
LLM. The evaluator and chat query it; the LLM only ever sees digests, never the
raw repo. The `submits` edge encodes the platform's core rule: **AzDO
orchestrates, AML executes**.

```mermaid
flowchart LR
    f1["file: train.py"] -->|defines| d1["def: train()"]
    f1 -->|imports| f2["file: data_prep.py"]
    d1 -->|calls| d2["def: load_data()"]
    azdo["file: ct-train.yml<br/>role = azdo_pipeline"] -->|references| aml["file: training_pipeline.yml<br/>role = aml_asset"]
    azdo ==>|submits| aml

    classDef pipe fill:#1e3a5f,stroke:#3b82f6,color:#fff;
    class azdo,aml pipe;
```

| Node kind | Meaning | Edge kinds |
|---|---|---|
| `file:<path>` | every repo file (lang, role, entry-point flag) | `imports`, `references`, `submits` |
| `def:<path>:<name>` | Python function/class (signature, doc, calls) | `defines`, `calls` |

### Capability dependency graph

The catalog is a DAG (`depends_on` / `enables`). `dependency_order()`
topo-sorts it so the gap report is fixed root-first — `environment_lifecycle`
before anything, `retraining`/`rollback` last. Arrows point **enables →**.

```mermaid
flowchart TD
    rd["repository_discovery<br/>(implicit — the scan)"]:::implicit --> el[environment_lifecycle]
    el --> tr[training]
    el --> bd[batch_deployment]
    el --> rt[realtime_deployment]
    el --> mon[monitoring]
    el --> ret[retraining]
    tr --> ml[model_lifecycle]
    tr --> ret
    ml --> bd
    ml --> rt
    ml --> rb[rollback]
    bd --> mon
    rt --> mon
    mon --> ret

    classDef implicit fill:#3b1f4f,stroke:#a855f7,color:#fff;
```

### Chat turn with live step streaming

Every assistant turn is a ReAct loop streamed over SSE. Tool calls, their
observations, and any reasoning appear live in the UI; generator tools
(`generate_assets`, `watch_pipeline_run`) emit intermediate `thinking` events —
e.g. a triggered pipeline streams its status + log tail until it finishes.

```mermaid
sequenceDiagram
    actor U as User
    participant W as React Chat
    participant S as /messages/stream
    participant CS as chat_stream (ReAct)
    participant T as Tools
    participant L as LLM

    U->>W: send message
    W->>S: POST (Server-Sent Events)
    S->>CS: chat_turn_stream
    loop until no tool calls
        CS->>L: completion(history + tools)
        L-->>CS: tool_calls
        CS-->>W: event: step
        CS->>T: tool.fn(args)
        T-->>CS: observation (or live yields)
        CS-->>W: event: thinking / observation
    end
    L-->>CS: final answer
    CS-->>W: event: final
    W-->>U: render steps + markdown reply
```

Key properties:
- **Capability-driven, not file-driven** — capabilities are inferred from graph
  evidence; "AzDO orchestrates, AML executes" is enforced via graph `submits` edges
- **Token-bounded** — the LLM sees graph digests and evidence excerpts, never the repo
- **Deterministic scoring** — LLM judges evidence; weights/statuses are pure code
- **Validate-local-first** — AzDO `previewRun` parses pipelines server-side with
  zero runs created; org-setup tasks (approval environments, service connections)
  are reported separately from file defects
- **Conversational guardrails** — A pre-flight topic classifier checks messages (with
  recent context) to strictly block non-ML/Data queries while naturally permitting 
  affirmations (e.g. "yes", "proceed") in context.
- **Observability** — Arize Phoenix traces every graph node and LLM call
  (`python -m phoenix.server.main serve`, http://localhost:6006)

Design docs: [docs/ARCHITECTURE_V2.md](docs/ARCHITECTURE_V2.md) ·
[docs/REQUIREMENTS.md](docs/REQUIREMENTS.md)

## Tests

```bash
pytest src/tests -q   # 46 tests
```
