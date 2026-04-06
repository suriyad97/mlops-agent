# MLOps Agent — Complete Architecture

## Overview

A conversational AI agent that manages the **entire Azure ML lifecycle** through natural language.
Built on LangGraph (multi-agent), FastAPI (backend), and Chainlit (chat UI).

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER (Browser)                               │
│                   http://localhost:8000                             │
└────────────────────────────┬────────────────────────────────────────┘
                             │ Chat messages / button clicks
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    CHAINLIT FRONTEND                                 │
│                  frontend/chainlit_app.py                           │
│                                                                     │
│  • Renders chat messages + streaming tokens                         │
│  • Shows tool calls as collapsible cl.Step blocks                   │
│  • Human-in-the-loop via cl.AskActionMessage                       │
│  • On-demand dashboard card (type "status")                         │
│  • Repo settings via cl.ChatSettings sidebar                        │
│  • Quick action buttons (Scan Repo, Run Training, etc.)             │
└────────────────────────────┬────────────────────────────────────────┘
                             │ POST /chat  (SSE stream)
                             │ POST /feedback (approval)
                             │ GET  /status/{thread_id}
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   FASTAPI BACKEND                                    │
│                  backend/main.py  :8001                             │
│                                                                     │
│  POST /chat      → runs LangGraph agent, streams SSE events         │
│  POST /feedback  → resumes interrupted graph with user decision     │
│  GET  /status    → returns current state snapshot for dashboard     │
│  GET  /health    → health check                                     │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  LANGGRAPH AGENT GRAPH                               │
│                 backend/agent/graph.py                              │
│                                                                     │
│   START                                                             │
│     │                                                               │
│     ▼                                                               │
│  ┌──────────────────────────────────────┐                           │
│  │         SUPERVISOR NODE              │  ← Claude Sonnet          │
│  │   (routes intent, decides tools)     │                           │
│  └────┬─────┬──────┬──────┬────────────┘                           │
│       │     │      │      │                                         │
│       ▼     ▼      ▼      ▼                                         │
│  ┌────┐ ┌──────┐ ┌─────┐ ┌──────────┐                              │
│  │Env │ │Train │ │Infer│ │Monitoring│  ← ReAct sub-agents           │
│  │Agent│ │Agent│ │Agent│ │Agent     │    (each has own tools)       │
│  └────┘ └──────┘ └─────┘ └──────────┘                              │
│       │     │      │      │                                         │
│       └─────┴──────┴──────┘                                         │
│                    │                                                │
│                    ▼                                                │
│            TOOL EXECUTOR NODE                                       │
│         (runs supervisor's direct tools)                            │
│                    │                                                │
│                    ▼                                                │
│               back to SUPERVISOR → END                              │
│                                                                     │
│  Checkpointer: SqliteSaver → mlops_agent_memory.db                  │
│  (every thread_id has isolated conversation history)                │
└────────────────────────────┬────────────────────────────────────────┘
                             │
            ┌────────────────┼────────────────┐
            ▼                ▼                ▼
┌───────────────┐  ┌──────────────────┐  ┌───────────────────┐
│  AZURE REPOS  │  │   AZURE ML       │  │  AZURE DEVOPS     │
│  (Git Repo)   │  │   WORKSPACE      │  │  PIPELINES        │
└───────────────┘  └──────────────────┘  └───────────────────┘
```

---

## Agent Nodes & Responsibilities

### Supervisor Node
- **LLM**: Claude Sonnet via `langchain-anthropic`
- **Decides**: which sub-agent to route to, or which tool to call directly
- **Handles directly**: repo scanning, artifact generation, sanity checks, status queries
- **Routes to sub-agents**: environment setup, training, inference, monitoring

### Sub-Agents (ReAct pattern)
Each sub-agent is a `create_react_agent` loop — it reasons, calls tools, observes, repeats.

| Agent | Trigger | Tools |
|---|---|---|
| **Environment Agent** | CI pipeline, ACR, AML env | `trigger_azdo_pipeline`, `get_azdo_run_status`, `list_azdo_pipelines` |
| **Training Agent** | CT pipeline, model reg | `trigger_ct_pipeline`, `poll_aml_job_status`, `register_model`, `compare_models` |
| **Inference Agent** | Batch scoring | `trigger_inference_pipeline`, `poll_aml_job_status`, `run_data_schema_check` |
| **Monitoring Agent** | Drift detection | `run_data_drift_pipeline`, `run_prediction_drift_pipeline`, `get_latest_drift_report` |

---

## Tool Inventory

### Azure Repo Tools (`tools/azure_repo.py`)
| Tool | What it does |
|---|---|
| `scan_azure_repo()` | **Full repo scan** — lists all files, reads key ones, detects framework, returns missing artifacts list |
| `list_repo_files()` | Lists files in a directory |
| `read_file_from_repo()` | Reads content of any file |
| `commit_file_to_repo()` | Commits generated files (Dockerfile, YAMLs) back to the repo |

### Azure ML Tools (`tools/azure_ml.py`)
| Tool | What it does |
|---|---|
| `trigger_ct_pipeline()` | Submits the CT (training) pipeline job to AML |
| `trigger_inference_pipeline()` | Submits batch inference pipeline job |
| `poll_aml_job_status()` | Polls live job status (Running/Completed/Failed) |
| `get_aml_job_logs()` | Downloads and returns last N lines of job logs |
| `list_aml_jobs()` | Lists recent AML jobs with status |
| `register_model()` | Registers model in AML registry with champion/challenger label |
| `compare_models()` | Compares challenger vs champion metrics, picks winner |
| `get_model_registry()` | Fetches all registered models with metrics |

### Azure DevOps Tools (`tools/azdo_pipelines.py`)
| Tool | What it does |
|---|---|
| `trigger_azdo_pipeline()` | Triggers a CI/CD pipeline run in Azure DevOps |
| `get_azdo_run_status()` | Gets status of a pipeline run |
| `list_azdo_pipelines()` | Lists all available pipelines |

### Artifact Generation Tools (`tools/artifacts.py`)
| Tool | What it does |
|---|---|
| `generate_dockerfile()` | Generates Dockerfile from repo scan results |
| `generate_aml_environment_yaml()` | Generates AML environment YAML |
| `generate_ct_pipeline_yaml()` | Generates CT pipeline YAML |
| `generate_inference_pipeline_yaml()` | Generates inference pipeline YAML |
| `generate_drift_pipeline_yaml()` | Generates drift monitoring pipeline YAML |
| `generate_azdo_ci_yaml()` | Generates Azure DevOps CI pipeline YAML |

### Sanity Check Tools (`tools/sanity.py`)
| Tool | What it does |
|---|---|
| `check_entry_points_exist()` | Verifies train.py / score.py exist |
| `check_dependencies_valid()` | Validates requirements.txt packages |
| `check_aml_resources_exist()` | Checks AML compute, datastore, environment exist |
| `validate_pipeline_yaml()` | Validates generated pipeline YAML syntax |
| `run_data_schema_check()` | Validates input data schema before inference |

### Drift Monitoring Tools (`tools/drift.py`)
| Tool | What it does |
|---|---|
| `run_data_drift_pipeline()` | Triggers data drift detection pipeline |
| `run_prediction_drift_pipeline()` | Triggers prediction drift pipeline |
| `run_concept_drift_pipeline()` | Triggers concept drift pipeline |
| `get_latest_drift_report()` | Fetches latest drift scores |

---

## Full End-to-End Flow (New Project)

```
User: "Scan my repo and set everything up"

① scan_azure_repo()
   → finds train.py (xgboost), requirements.txt
   → missing: Dockerfile, all pipeline YAMLs

② Agent presents project understanding to user
   → "I found an XGBoost classification project. Is primary metric AUC?"

③ User confirms → "Yes, AUC"

④ generate_dockerfile()
   generate_aml_environment_yaml()
   generate_ct_pipeline_yaml()
   generate_inference_pipeline_yaml()
   generate_drift_pipeline_yaml()
   generate_azdo_ci_yaml()

⑤ check_entry_points_exist()
   check_dependencies_valid()
   validate_pipeline_yaml()             ← sanity checks

⑥ commit_file_to_repo() × 6           ← push all to Azure Repo

⑦ ⏸ INTERRUPT → "Trigger the CI pipeline now?"
   User: "Yes"

⑧ trigger_azdo_pipeline()             ← builds Docker image → pushes to ACR → creates AML env

⑨ poll_aml_job_status() × N           ← watches until complete

⑩ Agent: "CI complete ✅. Ready to run training when you have data."
```

---

## Data Flow: SSE Streaming

```
LangGraph event          FastAPI SSE payload           Chainlit renders
─────────────────────    ──────────────────────────    ─────────────────────
on_chain_start      →    {type: "node_start"}      →   ▶ supervisor (Step)
on_tool_start       →    {type: "tool_call"}        →   🔧 scan_azure_repo (Step)
on_tool_end         →    {type: "tool_result"}      →     output shown in Step
on_chat_model_stream→    {type: "llm_token"}        →   streamed into Message
interrupt           →    {type: "interrupt"}        →   cl.AskActionMessage
done                →    {type: "done"}             →   stream closes
```

---

## State Schema (`agent/state.py`)

The `MLOpsState` TypedDict flows through every graph node:

```python
messages              # Full conversation history
repo_name             # Azure Repo being worked on
project_understanding # Output of repo scan
pipeline_runs         # List of all AML job runs
champion_model        # Current production model
challenger_model      # Newly trained candidate
data_drift_score      # Latest drift metrics
prediction_drift_score
concept_drift_score
human_review_approved # Boolean — did user approve?
current_job_id        # Actively running job
current_job_status    # Running/Completed/Failed
```

---

## Authentication

```
Service Principal (preferred — no browser login)
  AZURE_TENANT_ID
  AZURE_CLIENT_ID        → ClientSecretCredential → MLClient (Azure ML SDK v2)
  AZURE_CLIENT_SECRET
  AZURE_SUBSCRIPTION_ID

Personal Access Token
  AZURE_DEVOPS_PAT       → Basic Auth header → Azure DevOps REST API

Anthropic
  ANTHROPIC_API_KEY      → ChatAnthropic → Claude Sonnet
```

---

## File Structure

```
mlops-agent/
├── frontend/
│   ├── chainlit_app.py          # Chainlit UI (chat + steps + approval)
│   ├── app.py                   # Legacy Streamlit UI (backup)
│   ├── public/custom.css        # Dark theme CSS
│   └── .chainlit/config.toml   # Chainlit config
│
├── backend/
│   ├── main.py                  # FastAPI app (SSE endpoints)
│   └── agent/
│       ├── graph.py             # LangGraph supervisor + sub-agents
│       ├── state.py             # MLOpsState TypedDict
│       └── tools/
│           ├── azure_repo.py    # Repo tools (scan, read, commit)
│           ├── azure_ml.py      # AML tools (jobs, models, logs)
│           ├── azdo_pipelines.py # Azure DevOps tools
│           ├── artifacts.py     # YAML/Dockerfile generators
│           ├── sanity.py        # Pre-flight checks
│           └── drift.py         # Drift monitoring tools
│
├── .env                         # Credentials (never committed)
├── .env.example                 # Template
├── .gitignore
└── requirements.txt
```

---

## Ports

| Service | Port | Command |
|---|---|---|
| Chainlit UI | 8000 | `python -m chainlit run frontend/chainlit_app.py --port 8000` |
| FastAPI Backend | 8001 | `python -m uvicorn main:app --port 8001` (from `backend/`) |
