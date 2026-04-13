"""
Supervisor Node
================
The central router node. Reads user intent, either handles it directly
via tools or routes to a specialised sub-agent.
"""

import os
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage
from langgraph.graph import END

from mlops_agent.state import MLOpsState
from mlops_agent.configuration import AgentConfiguration
from mlops_agent.prompts.supervisor import SUPERVISOR_PROMPT
from mlops_agent.tools.azure.repo import (
    scan_azure_repo, read_file_from_repo, list_repo_files, commit_file_to_repo,
)
from mlops_agent.tools.azure.ml import list_aml_jobs, get_model_registry
from mlops_agent.tools.mlops.artifacts import (
    generate_dockerfile, generate_aml_environment_yaml,
    generate_ct_pipeline_yaml, generate_inference_pipeline_yaml,
    generate_drift_pipeline_yaml, generate_azdo_ci_yaml,
)
from mlops_agent.tools.mlops.sanity import (
    check_entry_points_exist, check_dependencies_valid,
    check_aml_resources_exist, validate_pipeline_yaml, run_data_schema_check,
)
from mlops_agent.tools.mlops.drift import get_latest_drift_report


def build_supervisor_llm(config: AgentConfiguration) -> ChatAnthropic:
    """Create and bind the supervisor LLM with all its direct tools."""
    llm = ChatAnthropic(
        model=config.model_name,
        api_key=os.environ["ANTHROPIC_API_KEY"],
        streaming=config.streaming,
    )
    return llm.bind_tools([
        # Repo tools
        scan_azure_repo, read_file_from_repo, list_repo_files, commit_file_to_repo,
        # Artifact generation
        generate_dockerfile, generate_aml_environment_yaml,
        generate_ct_pipeline_yaml, generate_inference_pipeline_yaml,
        generate_drift_pipeline_yaml, generate_azdo_ci_yaml,
        # Sanity checks
        check_entry_points_exist, check_dependencies_valid,
        check_aml_resources_exist, validate_pipeline_yaml, run_data_schema_check,
        # AML status queries
        list_aml_jobs, get_model_registry, get_latest_drift_report,
    ])


def make_supervisor_node(config: AgentConfiguration):
    """Factory: returns a supervisor_node function bound to the given config."""
    supervisor_llm = build_supervisor_llm(config)

    def supervisor_node(state: MLOpsState) -> MLOpsState:
        """
        Central supervisor: reads user intent, decides tools or sub-agent routing.
        """
        messages = [SystemMessage(content=SUPERVISOR_PROMPT)] + state["messages"]
        response = supervisor_llm.invoke(messages)
        return {"messages": [response]}

    return supervisor_node


def route_supervisor(state: MLOpsState) -> str:
    """
    Conditional edge after the supervisor responds.
    Routes to a sub-agent, tool executor, or END.
    """
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        tool_name = last_message.tool_calls[0]["name"]

        if tool_name in ("trigger_azdo_pipeline", "get_azdo_run_status"):
            return "environment_agent"
        if tool_name in ("trigger_ct_pipeline", "register_model", "compare_models"):
            return "training_agent"
        if tool_name in ("trigger_inference_pipeline",):
            return "inference_agent"
        if tool_name in ("run_data_drift_pipeline", "run_prediction_drift_pipeline",
                         "run_concept_drift_pipeline"):
            return "monitoring_agent"

        # Supervisor handles directly (repo scan, artifact gen, sanity checks)
        return "tool_executor"

    return END
