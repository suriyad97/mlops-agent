"""
MLOps Agent State Schema
========================
The shared state dictionary that flows through every node in the LangGraph graph.
Every node reads from this and writes back to it.
"""

from typing import Annotated, Any, Optional
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class MLOpsState(TypedDict):
    # ── Conversation ──────────────────────────────────────────────
    messages: Annotated[list, add_messages]   # Full chat history

    # ── Repo & Project Understanding ──────────────────────────────
    repo_name: Optional[str]                  # Azure Repo name
    repo_branch: Optional[str]               # Branch to work on
    repo_files: Optional[list[str]]          # List of files found in repo
    project_understanding: Optional[dict]    # Output of repo scan:
                                             #   framework, task_type, entry_points,
                                             #   dependencies, target_column, metrics
    task_type_confirmed: Optional[str]       # "classification", "regression",
                                             # "multiclass", "timeseries", "other"
    primary_metric: Optional[str]           # "auc", "f1", "rmse", "mape" etc.

    # ── Generated Artifacts ───────────────────────────────────────
    dockerfile_generated: Optional[bool]
    aml_env_yaml_generated: Optional[bool]
    ct_pipeline_yaml_generated: Optional[bool]
    inference_pipeline_yaml_generated: Optional[bool]
    drift_pipeline_yaml_generated: Optional[bool]
    azdo_ci_yaml_generated: Optional[bool]
    artifacts_committed: Optional[bool]

    # ── Sanity Check Results ──────────────────────────────────────
    sanity_checks_passed: Optional[bool]
    sanity_check_results: Optional[dict]     # { check_name: { passed: bool, message: str } }

    # ── Human Decision Points ──────────────────────────────────────
    human_review_approved: Optional[bool]    # After code review presentation
    human_artifacts_approved: Optional[bool] # After artifact generation
    human_promotion_decision: Optional[str]  # "promote" or "keep_champion"

    # ── Azure ML Job Tracking ──────────────────────────────────────
    current_job_id: Optional[str]
    current_job_type: Optional[str]          # "ci", "ct", "inference", "drift", "comparison"
    current_job_status: Optional[str]        # "Running", "Completed", "Failed", "Queued"
    job_error_message: Optional[str]
    job_logs_summary: Optional[str]

    # ── Pipeline Run History ───────────────────────────────────────
    pipeline_runs: Optional[list[dict]]      # List of all runs with status

    # ── Model Registry ─────────────────────────────────────────────
    champion_model: Optional[dict]           # { name, version, metrics, registered_at }
    challenger_model: Optional[dict]         # { name, version, metrics, registered_at }
    comparison_result: Optional[dict]        # { winner, challenger_metrics, champion_metrics }

    # ── Drift Monitoring ───────────────────────────────────────────
    data_drift_score: Optional[float]
    prediction_drift_score: Optional[float]
    concept_drift_score: Optional[float]
    drift_threshold_breached: Optional[bool]
    retraining_triggered: Optional[bool]

    # ── Active Sub-Agent ───────────────────────────────────────────
    active_agent: Optional[str]              # "environment", "training",
                                             # "inference", "monitoring"
    next_action: Optional[str]               # Hint for the supervisor
    error: Optional[str]                     # Any unhandled error message
