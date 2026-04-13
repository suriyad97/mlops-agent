"""
Unit tests for Azure ML tools.
All MLClient calls are mocked — no real AML workspace needed.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestPollAmlJobStatus:
    """Tests for poll_aml_job_status."""

    @patch("mlops_agent.tools.azure.ml.get_ml_client")
    def test_returns_running_status(self, mock_client_factory):
        mock_client = MagicMock()
        mock_job = MagicMock()
        mock_job.status = "Running"
        mock_job.creation_context = None
        mock_job.error = None
        mock_job.services = {}
        mock_client.jobs.get.return_value = mock_job
        mock_client_factory.return_value = mock_client

        from mlops_agent.tools.azure.ml import poll_aml_job_status
        result = poll_aml_job_status.invoke({"job_id": "test-job-123"})

        assert result["status"] == "Running"
        assert result["job_id"] == "test-job-123"

    @patch("mlops_agent.tools.azure.ml.get_ml_client")
    def test_returns_error_on_failed_job(self, mock_client_factory):
        mock_client = MagicMock()
        mock_job = MagicMock()
        mock_job.status = "Failed"
        mock_job.creation_context = None
        mock_job.error.message = "OOM error in training step"
        mock_job.services = {}
        mock_client.jobs.get.return_value = mock_job
        mock_client_factory.return_value = mock_client

        from mlops_agent.tools.azure.ml import poll_aml_job_status
        result = poll_aml_job_status.invoke({"job_id": "test-job-456"})

        assert result["status"] == "Failed"
        assert result["error_message"] is not None


class TestCompareModels:
    """Tests for compare_models — metric comparison logic."""

    @patch("mlops_agent.tools.azure.ml.get_ml_client")
    def test_challenger_wins_on_higher_auc(self, mock_client_factory):
        mock_client = MagicMock()

        challenger = MagicMock()
        challenger.tags = {"label": "challenger", "metrics": {"auc": "0.92"}}
        champion = MagicMock()
        champion.tags = {"label": "champion", "metrics": {"auc": "0.88"}}

        mock_client.models.get.side_effect = [challenger, champion]
        mock_client_factory.return_value = mock_client

        from mlops_agent.tools.azure.ml import compare_models
        result = compare_models.invoke({
            "challenger_name": "churn-model", "challenger_version": "2",
            "champion_name": "churn-model", "champion_version": "1",
            "primary_metric": "auc",
        })

        assert result["winner"] == "challenger"
        assert result["improvement"] == pytest.approx(0.04, abs=0.001)

    @patch("mlops_agent.tools.azure.ml.get_ml_client")
    def test_champion_wins_on_lower_rmse(self, mock_client_factory):
        mock_client = MagicMock()

        challenger = MagicMock()
        challenger.tags = {"label": "challenger", "metrics": {"rmse": "0.25"}}
        champion = MagicMock()
        champion.tags = {"label": "champion", "metrics": {"rmse": "0.18"}}

        mock_client.models.get.side_effect = [challenger, champion]
        mock_client_factory.return_value = mock_client

        from mlops_agent.tools.azure.ml import compare_models
        result = compare_models.invoke({
            "challenger_name": "price-model", "challenger_version": "2",
            "champion_name": "price-model", "champion_version": "1",
            "primary_metric": "rmse",
        })

        assert result["winner"] == "champion"
