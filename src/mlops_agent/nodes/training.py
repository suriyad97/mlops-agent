"""
Training Agent Node
====================
Handles the full CT pipeline lifecycle:
  - Pipeline submission
  - Job status polling
  - Model registration (challenger)
  - Champion vs challenger comparison
  - Promotion decisions
"""

import os
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

from mlops_agent.state import MLOpsState
from mlops_agent.configuration import AgentConfiguration
from mlops_agent.prompts.agents import TRAINING_AGENT_PROMPT
from mlops_agent.tools.azure.ml import (
    trigger_ct_pipeline, poll_aml_job_status, register_model,
    compare_models, get_aml_job_logs, get_model_registry,
)


def make_training_agent_node(config: AgentConfiguration):
    """Factory: returns a training_agent_node bound to the given config."""
    llm = ChatAnthropic(
        model=config.model_name,
        api_key=os.environ["ANTHROPIC_API_KEY"],
        streaming=config.streaming,
    )
    agent = create_react_agent(
        llm,
        tools=[trigger_ct_pipeline, poll_aml_job_status, register_model,
               compare_models, get_aml_job_logs, get_model_registry],
        state_modifier=TRAINING_AGENT_PROMPT,
    )

    def training_agent_node(state: MLOpsState) -> MLOpsState:
        result = agent.invoke({"messages": state["messages"]})
        return {"messages": result["messages"], "active_agent": "training"}

    return training_agent_node
