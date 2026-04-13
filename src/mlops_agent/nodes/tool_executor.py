"""
Tool Executor Node
===================
Handles tool calls made directly by the Supervisor node.
Runs the tool, wraps the result in a ToolMessage, and returns to supervisor.
"""

from langchain_core.messages import ToolMessage

from mlops_agent.state import MLOpsState
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


# Registry of all tools the supervisor can call directly
_SUPERVISOR_TOOLS = {
    t.name: t for t in [
        scan_azure_repo, read_file_from_repo, list_repo_files, commit_file_to_repo,
        generate_dockerfile, generate_aml_environment_yaml,
        generate_ct_pipeline_yaml, generate_inference_pipeline_yaml,
        generate_drift_pipeline_yaml, generate_azdo_ci_yaml,
        check_entry_points_exist, check_dependencies_valid,
        check_aml_resources_exist, validate_pipeline_yaml, run_data_schema_check,
        list_aml_jobs, get_model_registry, get_latest_drift_report,
    ]
}


def tool_executor_node(state: MLOpsState) -> MLOpsState:
    """
    Execute every tool call in the last supervisor message.
    Returns ToolMessage results for each call.
    """
    last_message = state["messages"][-1]
    results = []

    for tool_call in last_message.tool_calls:
        tool = _SUPERVISOR_TOOLS.get(tool_call["name"])
        if tool:
            output = tool.invoke(tool_call["args"])
            results.append(
                ToolMessage(content=str(output), tool_call_id=tool_call["id"])
            )
        else:
            results.append(
                ToolMessage(
                    content=f"Tool '{tool_call['name']}' not found.",
                    tool_call_id=tool_call["id"],
                )
            )

    return {"messages": results}
