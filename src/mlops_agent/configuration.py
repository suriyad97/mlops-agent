"""
Agent Configuration
====================
Centralised configuration for the MLOps agent.
Values are loaded from environment variables with sensible defaults.
Pass a custom AgentConfiguration to build_graph() to override.
"""

import os
from dataclasses import dataclass, field


@dataclass
class AgentConfiguration:
    # ── LLM ───────────────────────────────────────────────────────────────────
    model_name: str = field(
        default_factory=lambda: os.environ.get("AGENT_MODEL", "claude-sonnet-4-6")
    )
    streaming: bool = True

    # ── Memory ────────────────────────────────────────────────────────────────
    memory_db_path: str = field(
        default_factory=lambda: os.environ.get("MEMORY_DB_PATH", "mlops_agent_memory.db")
    )

    # ── Drift Thresholds ──────────────────────────────────────────────────────
    data_drift_threshold: float = field(
        default_factory=lambda: float(os.environ.get("DATA_DRIFT_THRESHOLD", "0.15"))
    )
    prediction_drift_threshold: float = field(
        default_factory=lambda: float(os.environ.get("PREDICTION_DRIFT_THRESHOLD", "0.10"))
    )
    concept_drift_threshold: float = field(
        default_factory=lambda: float(os.environ.get("CONCEPT_DRIFT_THRESHOLD", "0.05"))
    )

    # ── Job Polling ───────────────────────────────────────────────────────────
    max_poll_retries: int = 60          # max times to poll a running job
    poll_interval_seconds: int = 30     # seconds between polls

    # ── Repo Scanner ─────────────────────────────────────────────────────────
    max_file_read_chars: int = 4000     # cap per file to avoid context overflow
    max_entrypoints_to_read: int = 5

    def __post_init__(self):
        """Validate critical thresholds."""
        for attr in ("data_drift_threshold", "prediction_drift_threshold", "concept_drift_threshold"):
            val = getattr(self, attr)
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{attr} must be between 0.0 and 1.0, got {val}")


# Module-level default instance
default_config = AgentConfiguration()
