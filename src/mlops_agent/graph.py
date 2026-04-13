"""
MLOps Agent — LangGraph Graph
==============================
This file ONLY wires nodes and edges.
All business logic lives in nodes/, tools/, and prompts/.

Flow:
  START → supervisor → [tool_executor | environment_agent |
                         training_agent | inference_agent |
                         monitoring_agent] → supervisor → END
"""

import os
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from mlops_agent.state import MLOpsState
from mlops_agent.configuration import AgentConfiguration
from mlops_agent.nodes.supervisor import make_supervisor_node, route_supervisor
from mlops_agent.nodes.tool_executor import tool_executor_node
from mlops_agent.nodes.environment import make_environment_agent_node
from mlops_agent.nodes.training import make_training_agent_node
from mlops_agent.nodes.inference import make_inference_agent_node
from mlops_agent.nodes.monitoring import make_monitoring_agent_node


def build_graph(config: AgentConfiguration | None = None):
    """
    Build and compile the MLOps agent graph.

    Args:
        config: Optional AgentConfiguration. Uses defaults if not provided.

    Returns:
        Compiled LangGraph graph with SqliteSaver checkpointer.
    """
    if config is None:
        config = AgentConfiguration()

    # ── Create node functions from factories ──────────────────────────────────
    supervisor_node       = make_supervisor_node(config)
    environment_agent_node = make_environment_agent_node(config)
    training_agent_node   = make_training_agent_node(config)
    inference_agent_node  = make_inference_agent_node(config)
    monitoring_agent_node = make_monitoring_agent_node(config)

    # ── Build graph ───────────────────────────────────────────────────────────
    builder = StateGraph(MLOpsState)

    builder.add_node("supervisor",         supervisor_node)
    builder.add_node("tool_executor",      tool_executor_node)
    builder.add_node("environment_agent",  environment_agent_node)
    builder.add_node("training_agent",     training_agent_node)
    builder.add_node("inference_agent",    inference_agent_node)
    builder.add_node("monitoring_agent",   monitoring_agent_node)

    # Entry
    builder.add_edge(START, "supervisor")

    # Routing
    builder.add_conditional_edges("supervisor", route_supervisor, {
        "tool_executor":      "tool_executor",
        "environment_agent":  "environment_agent",
        "training_agent":     "training_agent",
        "inference_agent":    "inference_agent",
        "monitoring_agent":   "monitoring_agent",
        END: END,
    })

    # All nodes return to supervisor
    for node in ("tool_executor", "environment_agent", "training_agent",
                 "inference_agent", "monitoring_agent"):
        builder.add_edge(node, "supervisor")

    # ── Checkpointer ──────────────────────────────────────────────────────────
    # SqliteSaver: persists conversation state across restarts.
    # Switch to PostgresSaver for production multi-instance deployments.
    checkpointer = SqliteSaver.from_conn_string(config.memory_db_path)

    return builder.compile(checkpointer=checkpointer)
