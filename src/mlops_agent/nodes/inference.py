"""
Inference Agent Node
=====================
Handles batch inference operations:
  - Data schema validation
  - Pipeline submission using champion model
  - Job status tracking
  - Output location reporting
"""

import os
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

from mlops_agent.state import MLOpsState
from mlops_agent.configuration import AgentConfiguration
from mlops_agent.prompts.agents import INFERENCE_AGENT_PROMPT
from mlops_agent.tools.azure.ml import (
    trigger_inference_pipeline, poll_aml_job_status, get_aml_job_logs,
)
from mlops_agent.tools.mlops.sanity import run_data_schema_check


def make_inference_agent_node(config: AgentConfiguration):
    """Factory: returns an inference_agent_node bound to the given config."""
    llm = ChatAnthropic(
        model=config.model_name,
        api_key=os.environ["ANTHROPIC_API_KEY"],
        streaming=config.streaming,
    )
    agent = create_react_agent(
        llm,
        tools=[trigger_inference_pipeline, poll_aml_job_status,
               get_aml_job_logs, run_data_schema_check],
        state_modifier=INFERENCE_AGENT_PROMPT,
    )

    def inference_agent_node(state: MLOpsState) -> MLOpsState:
        result = agent.invoke({"messages": state["messages"]})
        return {"messages": result["messages"], "active_agent": "inference"}

    return inference_agent_node
