"""
Environment Agent Node
=======================
Handles CI pipeline operations:
  - Docker image builds
  - ACR pushes
  - AML environment registration

Uses a ReAct sub-agent with Azure DevOps tools.
"""

import os
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

from mlops_agent.state import MLOpsState
from mlops_agent.configuration import AgentConfiguration
from mlops_agent.prompts.agents import ENVIRONMENT_AGENT_PROMPT
from mlops_agent.tools.azure.devops import (
    trigger_azdo_pipeline, get_azdo_run_status, list_azdo_pipelines,
)
from mlops_agent.tools.azure.repo import read_file_from_repo, list_repo_files


def make_environment_agent_node(config: AgentConfiguration):
    """Factory: returns an environment_agent_node bound to the given config."""
    llm = ChatAnthropic(
        model=config.model_name,
        api_key=os.environ["ANTHROPIC_API_KEY"],
        streaming=config.streaming,
    )
    agent = create_react_agent(
        llm,
        tools=[trigger_azdo_pipeline, get_azdo_run_status, list_azdo_pipelines,
               read_file_from_repo, list_repo_files],
        state_modifier=ENVIRONMENT_AGENT_PROMPT,
    )

    def environment_agent_node(state: MLOpsState) -> MLOpsState:
        result = agent.invoke({"messages": state["messages"]})
        return {"messages": result["messages"], "active_agent": "environment"}

    return environment_agent_node
