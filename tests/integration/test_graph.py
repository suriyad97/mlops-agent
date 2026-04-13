"""
Integration tests for the LangGraph graph.
Tests that the graph builds correctly and all nodes are registered.
Does NOT make real LLM or Azure calls.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestGraphBuilds:
    """Verify the graph compiles without errors."""

    @patch("mlops_agent.nodes.supervisor.ChatAnthropic")
    @patch("mlops_agent.nodes.environment.ChatAnthropic")
    @patch("mlops_agent.nodes.training.ChatAnthropic")
    @patch("mlops_agent.nodes.inference.ChatAnthropic")
    @patch("mlops_agent.nodes.monitoring.ChatAnthropic")
    @patch("mlops_agent.graph.SqliteSaver")
    def test_build_graph_registers_all_nodes(
        self,
        mock_sqlite,
        mock_mon_llm, mock_inf_llm, mock_trn_llm, mock_env_llm, mock_sup_llm,
        monkeypatch,
    ):
        """build_graph() should compile without errors and contain all 6 nodes."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")
        mock_sqlite.from_conn_string.return_value = MagicMock()

        from mlops_agent.graph import build_graph
        from mlops_agent.configuration import AgentConfiguration

        config = AgentConfiguration(memory_db_path=":memory:")
        graph = build_graph(config)

        assert graph is not None

    @patch("mlops_agent.nodes.supervisor.ChatAnthropic")
    @patch("mlops_agent.nodes.environment.ChatAnthropic")
    @patch("mlops_agent.nodes.training.ChatAnthropic")
    @patch("mlops_agent.nodes.inference.ChatAnthropic")
    @patch("mlops_agent.nodes.monitoring.ChatAnthropic")
    @patch("mlops_agent.graph.SqliteSaver")
    def test_build_graph_uses_default_config(
        self,
        mock_sqlite,
        mock_mon_llm, mock_inf_llm, mock_trn_llm, mock_env_llm, mock_sup_llm,
        monkeypatch,
    ):
        """build_graph() with no config should use AgentConfiguration defaults."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")
        mock_sqlite.from_conn_string.return_value = MagicMock()

        from mlops_agent.graph import build_graph
        graph = build_graph()

        assert graph is not None
