"""
Unit tests for Azure Repo tools.
All HTTP calls are mocked — no real Azure connection needed.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestScanAzureRepo:
    """Tests for the high-level scan_azure_repo tool."""

    @patch("mlops_agent.tools.azure.repo.requests.get")
    def test_scan_returns_file_list(self, mock_get):
        """scan_azure_repo should list all files and classify them."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "value": [
                {"path": "/train.py", "isFolder": False},
                {"path": "/requirements.txt", "isFolder": False},
                {"path": "/src", "isFolder": True},
            ]
        }
        mock_resp.text = "xgboost==1.7.0\nscikit-learn==1.3.0"
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        from mlops_agent.tools.azure.repo import scan_azure_repo
        result = scan_azure_repo.invoke({"branch": "main"})

        assert "train.py" in result["entrypoints"]
        assert "xgboost" in result["framework_hints"]
        assert result["file_count"] == 2  # folders excluded

    @patch("mlops_agent.tools.azure.repo.requests.get")
    def test_scan_detects_missing_artifacts(self, mock_get):
        """scan_azure_repo should report missing Dockerfiles and pipeline YAMLs."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"value": [
            {"path": "/train.py", "isFolder": False},
        ]}
        mock_resp.text = ""
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        from mlops_agent.tools.azure.repo import scan_azure_repo
        result = scan_azure_repo.invoke({"branch": "main"})

        assert "Dockerfile" in result["missing_artifacts"]
        assert "ct_pipeline.yaml" in result["missing_artifacts"]


class TestListRepoFiles:
    """Tests for list_repo_files."""

    @patch("mlops_agent.tools.azure.repo.requests.get")
    def test_returns_file_paths(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"value": [
            {"path": "/train.py", "isFolder": False},
            {"path": "/src", "isFolder": True},
        ]}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        from mlops_agent.tools.azure.repo import list_repo_files
        result = list_repo_files.invoke({"directory_path": "/"})

        assert result["count"] == 1
        assert "/train.py" in result["files"]
