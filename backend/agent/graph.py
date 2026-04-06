"""
MLOps Agent — LangGraph Graph Definition
==========================================
Supervisor + 4 Sub-agents pattern.

Flow:
  User Input
    → Supervisor (routes intent)
      → Environment Agent  (CI pipeline, ACR, AML env)
      → Training Agent     (CT pipeline, model registration, comparison)
      → Inference Agent    (batch inference, data validation)
      → Monitoring Agent   (data drift, prediction drift, concept drift)
    ← Results back to Supervisor
    → Respond to User
"""

import os
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent

from agent.state import MLOpsState
from agent.tools.azure_repo import (
    read_file_from_repo,
    list_repo_files,
    commit_file_to_repo,
)
from agent.tools.azure_ml import (
    trigger_ct_pipeline,
    trigger_inference_pipeline,
    poll_aml_job_status,
    register_model,
    compare_models,
    get_aml_job_logs,
    list_aml_jobs,
    get_model_registry,
)
from agent.tools.azdo_pipelines import (
    trigger_azdo_pipeline,
    get_azdo_run_status,
    list_azdo_pipelines,
)
from agent.tools.drift import (
    run_data_drift_pipeline,
    run_prediction_drift_pipeline,
    run_concept_drift_pipeline,
    get_latest_drift_report,
)
from agent.tools.sanity import (
    check_entry_points_exist,
    check_dependencies_valid,
    check_aml_resources_exist,
    validate_pipeline_yaml,
    run_data_schema_check,
)
from agent.tools.artifacts import (
    generate_dockerfile,
    generate_aml_environment_yaml,
    generate_ct_pipeline_yaml,
    generate_inference_pipeline_yaml,
    generate_drift_pipeline_yaml,
    generate_azdo_ci_yaml,
)

# ── LLM ───────────────────────────────────────────────────────────────────────

llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    api_key=os.environ["ANTHROPIC_API_KEY"],
    streaming=True,
)

# ── System Prompts ─────────────────────────────────────────────────────────────

SUPERVISOR_PROMPT = """You are the MLOps Supervisor Agent managing an Azure ML workspace.
Your job is to understand user intent and coordinate the right actions.

You oversee these pipelines:
- CI Pipeline: Builds Docker image → pushes to ACR → creates AML environment
- CT Pipeline: Validates data → trains model → registers as challenger
- Model Comparison: Compares challenger vs champion → promotes winner
- Inference Pipeline: Batch scoring using champion model (runs twice a month)
- Drift Monitoring: Data drift, prediction drift, concept drift

When a user gives raw source code (no Dockerfile or pipeline YAMLs), your first job is to:
1. Scan the repo thoroughly and present your understanding of the project
2. Ask the user to confirm the ML task type and primary evaluation metric
3. Generate all missing artifacts (Dockerfile, YAMLs)
4. Run sanity checks before touching any Azure resources
5. Get human approval at each critical decision point

Always be clear about what you are doing and why. When jobs are running,
poll their status and keep the user informed. Never assume — always confirm
before destructive or irreversible actions like model promotion."""

ENVIRONMENT_AGENT_PROMPT = """You are the Environment Agent for an MLOps system on Azure ML.
Your responsibilities:
- Trigger and monitor the CI pipeline (Docker build → ACR push → AML env creation)
- Check if Docker images exist in ACR
- Verify AML environments are registered and healthy
- Report clearly on build failures with actionable diagnostics

Always check the current status before triggering a new run."""

TRAINING_AGENT_PROMPT = """You are the Training Agent for an MLOps system on Azure ML.
Your responsibilities:
- Trigger the Continuous Training (CT) pipeline
- Poll job status until completion
- Register trained models with the 'challenger' label
- Run model comparison between challenger and champion
- Present comparison results clearly and ask for human approval before promotion

Always verify data path exists before triggering training."""

INFERENCE_AGENT_PROMPT = """You are the Inference Agent for an MLOps system on Azure ML.
Your responsibilities:
- Trigger batch inference pipeline using the current champion model
- Validate input data schema before submission
- Poll job status and report output location on completion
- This pipeline runs twice a month — always confirm before triggering"""

MONITORING_AGENT_PROMPT = """You are the Monitoring Agent for an MLOps system on Azure ML.
Your responsibilities:
- Run data drift, prediction drift, and concept drift pipelines
- Fetch and interpret the latest drift reports
- Alert the user clearly when thresholds are breached
- Suggest retraining when drift is significant, but always ask before triggering"""

# ── Sub-Agents (ReAct) ────────────────────────────────────────────────────────

environment_agent = create_react_agent(
    llm,
    tools=[trigger_azdo_pipeline, get_azdo_run_status, list_azdo_pipelines,
           read_file_from_repo, list_repo_files],
    state_modifier=ENVIRONMENT_AGENT_PROMPT,
)

training_agent = create_react_agent(
    llm,
    tools=[trigger_ct_pipeline, poll_aml_job_status, register_model,
           compare_models, get_aml_job_logs, get_model_registry],
    state_modifier=TRAINING_AGENT_PROMPT,
)

inference_agent = create_react_agent(
    llm,
    tools=[trigger_inference_pipeline, poll_aml_job_status,
           get_aml_job_logs, run_data_schema_check],
    state_modifier=INFERENCE_AGENT_PROMPT,
)

monitoring_agent = create_react_agent(
    llm,
    tools=[run_data_drift_pipeline, run_prediction_drift_pipeline,
           run_concept_drift_pipeline, get_latest_drift_report,
           poll_aml_job_status],
    state_modifier=MONITORING_AGENT_PROMPT,
)

# ── Supervisor Node ────────────────────────────────────────────────────────────

supervisor_llm = llm.bind_tools([
    # Repo & artifact tools
    read_file_from_repo, list_repo_files, commit_file_to_repo,
    # Artifact generation
    generate_dockerfile, generate_aml_environment_yaml,
    generate_ct_pipeline_yaml, generate_inference_pipeline_yaml,
    generate_drift_pipeline_yaml, generate_azdo_ci_yaml,
    # Sanity checks
    check_entry_points_exist, check_dependencies_valid,
    check_aml_resources_exist, validate_pipeline_yaml, run_data_schema_check,
    # AML status
    list_aml_jobs, get_model_registry, get_latest_drift_report,
])


def supervisor_node(state: MLOpsState) -> MLOpsState:
    """
    Central supervisor that understands user intent and either:
    - Handles it directly (repo scan, artifact generation, sanity checks)
    - Routes to a sub-agent (training, inference, monitoring, environment)
    - Asks the human for a decision (interrupt points)
    """
    messages = [SystemMessage(content=SUPERVISOR_PROMPT)] + state["messages"]
    response = supervisor_llm.invoke(messages)
    return {"messages": [response]}


def route_supervisor(state: MLOpsState) -> str:
    """
    Conditional edge: after the supervisor responds, decide where to go next.
    """
    last_message = state["messages"][-1]

    # If the supervisor called a tool, execute it
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        tool_name = last_message.tool_calls[0]["name"]

        # Route to sub-agents based on tool intent
        if tool_name in ("trigger_azdo_pipeline", "get_azdo_run_status"):
            return "environment_agent"
        if tool_name in ("trigger_ct_pipeline", "register_model", "compare_models"):
            return "training_agent"
        if tool_name in ("trigger_inference_pipeline",):
            return "inference_agent"
        if tool_name in ("run_data_drift_pipeline", "run_prediction_drift_pipeline",
                         "run_concept_drift_pipeline"):
            return "monitoring_agent"

        # Supervisor handles it directly (repo tools, artifact gen, sanity checks)
        return "tool_executor"

    # No tool calls — we're done
    return END


def tool_executor_node(state: MLOpsState) -> MLOpsState:
    """
    Executes tool calls made directly by the supervisor node.
    """
    from langchain_core.messages import ToolMessage
    from langchain_core.tools import BaseTool

    all_tools = {
        t.name: t for t in [
            read_file_from_repo, list_repo_files, commit_file_to_repo,
            generate_dockerfile, generate_aml_environment_yaml,
            generate_ct_pipeline_yaml, generate_inference_pipeline_yaml,
            generate_drift_pipeline_yaml, generate_azdo_ci_yaml,
            check_entry_points_exist, check_dependencies_valid,
            check_aml_resources_exist, validate_pipeline_yaml, run_data_schema_check,
            list_aml_jobs, get_model_registry, get_latest_drift_report,
        ]
    }

    last_message = state["messages"][-1]
    results = []
    for tool_call in last_message.tool_calls:
        tool = all_tools.get(tool_call["name"])
        if tool:
            output = tool.invoke(tool_call["args"])
            results.append(ToolMessage(content=str(output), tool_call_id=tool_call["id"]))
        else:
            results.append(ToolMessage(
                content=f"Tool '{tool_call['name']}' not found.",
                tool_call_id=tool_call["id"]
            ))

    return {"messages": results}


def environment_agent_node(state: MLOpsState) -> MLOpsState:
    result = environment_agent.invoke({"messages": state["messages"]})
    return {"messages": result["messages"], "active_agent": "environment"}


def training_agent_node(state: MLOpsState) -> MLOpsState:
    result = training_agent.invoke({"messages": state["messages"]})
    return {"messages": result["messages"], "active_agent": "training"}


def inference_agent_node(state: MLOpsState) -> MLOpsState:
    result = inference_agent.invoke({"messages": state["messages"]})
    return {"messages": result["messages"], "active_agent": "inference"}


def monitoring_agent_node(state: MLOpsState) -> MLOpsState:
    result = monitoring_agent.invoke({"messages": state["messages"]})
    return {"messages": result["messages"], "active_agent": "monitoring"}


# ── Build Graph ────────────────────────────────────────────────────────────────

def build_graph():
    builder = StateGraph(MLOpsState)

    # Add nodes
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("tool_executor", tool_executor_node)
    builder.add_node("environment_agent", environment_agent_node)
    builder.add_node("training_agent", training_agent_node)
    builder.add_node("inference_agent", inference_agent_node)
    builder.add_node("monitoring_agent", monitoring_agent_node)

    # Entry point
    builder.add_edge(START, "supervisor")

    # Supervisor routes conditionally
    builder.add_conditional_edges("supervisor", route_supervisor, {
        "tool_executor": "tool_executor",
        "environment_agent": "environment_agent",
        "training_agent": "training_agent",
        "inference_agent": "inference_agent",
        "monitoring_agent": "monitoring_agent",
        END: END,
    })

    # After tool execution or sub-agents, always return to supervisor
    builder.add_edge("tool_executor", "supervisor")
    builder.add_edge("environment_agent", "supervisor")
    builder.add_edge("training_agent", "supervisor")
    builder.add_edge("inference_agent", "supervisor")
    builder.add_edge("monitoring_agent", "supervisor")

    # ── Checkpointer ──────────────────────────────────────────────────────────
    # SqliteSaver writes all conversation state to a local .db file.
    # This means:
    #   - Conversations persist across FastAPI restarts
    #   - Long-running AML jobs don't lose context if the server blips
    #   - Every thread_id gets its own isolated conversation history
    #   - You can open mlops_agent_memory.db with any SQLite viewer to inspect
    #
    # To switch to Postgres for production:
    #   from langgraph.checkpoint.postgres import PostgresSaver
    #   checkpointer = PostgresSaver.from_conn_string(os.environ["POSTGRES_URL"])

    db_path = os.environ.get("MEMORY_DB_PATH", "mlops_agent_memory.db")
    checkpointer = SqliteSaver.from_conn_string(db_path)

    return builder.compile(checkpointer=checkpointer)
