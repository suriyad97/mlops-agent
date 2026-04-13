"""
Monitoring Agent Node
======================
Handles drift monitoring operations:
  - Data drift detection
  - Prediction drift detection
  - Concept drift detection
  - Threshold breach alerting
  - Retraining suggestions
"""

import os
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

from mlops_agent.state import MLOpsState
from mlops_agent.configuration import AgentConfiguration
from mlops_agent.prompts.agents import MONITORING_AGENT_PROMPT
from mlops_agent.tools.azure.ml import poll_aml_job_status
from mlops_agent.tools.mlops.drift import (
    run_data_drift_pipeline, run_prediction_drift_pipeline,
    run_concept_drift_pipeline, get_latest_drift_report,
)


def make_monitoring_agent_node(config: AgentConfiguration):
    """Factory: returns a monitoring_agent_node bound to the given config."""
    llm = ChatAnthropic(
        model=config.model_name,
        api_key=os.environ["ANTHROPIC_API_KEY"],
        streaming=config.streaming,
    )
    agent = create_react_agent(
        llm,
        tools=[run_data_drift_pipeline, run_prediction_drift_pipeline,
               run_concept_drift_pipeline, get_latest_drift_report,
               poll_aml_job_status],
        state_modifier=MONITORING_AGENT_PROMPT,
    )

    def monitoring_agent_node(state: MLOpsState) -> MLOpsState:
        result = agent.invoke({"messages": state["messages"]})
        return {"messages": result["messages"], "active_agent": "monitoring"}

    return monitoring_agent_node
