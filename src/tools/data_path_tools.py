"""Verify user-provided data-plane paths (blob/datastore) used by the pipelines.

The user pastes blob/datastore paths upfront; these get baked into the generated
pipelines (training input, monitoring baseline/current, batch I/O). This validates
the path FORMAT deterministically and makes a BEST-EFFORT existence check for blob
HTTPS URLs via DefaultAzureCredential — degrading gracefully when the storage SDK or
credentials aren't available (format-valid but unverified is still useful).
"""
import re
from typing import List, Optional, Tuple

from src.tools.azure_infra_tools import InfraReport, PrereqItem

# scheme prefix -> human label
_SCHEMES = {
    "azureml://": "AML datastore URI",
    "azureml:":   "AML data asset",
    "https://":   "Blob HTTPS URL",
    "abfss://":   "ADLS Gen2 URI",
    "wasbs://":   "Blob (wasbs) URI",
}

# (profile key, label, required_for, strategies that need it)
_DATA_PATH_SPEC: List[Tuple[str, str, str, set]] = [
    ("training_data_path", "Training data",       "CT — training input",        {"realtime", "batch", "both", "none"}),
    ("eval_data_path",     "Evaluation data",     "evaluate step (held-out)",   set()),   # optional, falls back
    ("batch_input_path",   "Batch input",         "batch scoring input",        {"batch", "both"}),
    ("batch_output_path",  "Batch output",        "batch predictions output",   {"batch", "both"}),
    ("baseline_data_path", "Monitoring baseline", "drift reference distribution", {"realtime", "batch", "both"}),
    ("current_data_path",  "Monitoring current",  "drift comparison window",    set()),   # optional (MDC/batch output)
]

_BLOB_HTTPS = re.compile(r"^https://([a-z0-9]+)\.blob\.core\.windows\.net/([^/]+)/?(.*)$", re.I)


def _classify(path: str) -> Optional[str]:
    """Return the scheme label if the path uses a recognized scheme, else None."""
    for prefix, label in _SCHEMES.items():
        if path.startswith(prefix):
            return label
    return None


def _blob_exists(path: str) -> Optional[bool]:
    """Best-effort existence for a blob HTTPS URL. None when it cannot be checked."""
    m = _BLOB_HTTPS.match(path)
    if not m:
        return None
    account, container, prefix = m.group(1), m.group(2), m.group(3)
    try:
        from azure.identity import DefaultAzureCredential
        from azure.storage.blob import ContainerClient
    except Exception:
        return None  # SDK not installed — cannot verify
    # the credential chain logs its attempts noisily; this is a best-effort probe, so quiet it
    import logging
    logging.getLogger("azure.identity").setLevel(logging.CRITICAL)
    try:
        cc = ContainerClient(
            account_url=f"https://{account}.blob.core.windows.net",
            container_name=container,
            credential=DefaultAzureCredential(),
        )
        # a folder "path" has no blob of its own; list one blob under the prefix
        for _ in cc.list_blobs(name_starts_with=prefix.rstrip("/"), results_per_page=1).by_page():
            return True
        # container reachable but prefix empty
        return False
    except Exception:
        return None  # auth/network/permission — treat as unverified, not a hard failure


def _verify_one(key: str, label: str, required_for: str, required: bool, value: str) -> Optional[PrereqItem]:
    value = (value or "").strip()
    if not value:
        if not required:
            return None  # optional + empty → don't clutter the report
        return PrereqItem(
            name=label, status="missing_config", required_for=required_for,
            detail="no path provided — pipeline falls back to azureml:<asset>@latest",
            fix=f"Paste the blob/datastore path for {label.lower()} (e.g. "
                f"azureml://datastores/<datastore>/paths/<folder>/ or "
                f"https://<account>.blob.core.windows.net/<container>/<folder>/), or register the data asset.",
        )

    scheme = _classify(value)
    if scheme is None:
        return PrereqItem(
            name=label, status="error", required_for=required_for, detail=f"unrecognized path scheme: {value[:60]}",
            fix="Use azureml://datastores/…, azureml:<asset>@<version>, https://<account>.blob.core.windows.net/…, "
                "abfss://… or wasbs://…",
        )

    exists = _blob_exists(value)
    if exists is True:
        return PrereqItem(name=label, status="ok", required_for=required_for, detail=f"{scheme} · verified in blob")
    if exists is False:
        return PrereqItem(
            name=label, status="not_found", required_for=required_for,
            detail=f"{scheme} · no data found at this path",
            fix="Upload data to this path, or correct it. The container is reachable but the folder is empty.",
        )
    # exists is None → format valid, existence not checkable
    return PrereqItem(name=label, status="ok", required_for=required_for,
                      detail=f"{scheme} · format valid (existence not verified)")


def verify_data_paths(profile: dict | None) -> InfraReport:
    """Validate the configured data paths for the project's endpoint strategy."""
    profile = profile or {}
    data_paths = profile.get("data_paths") or {}
    strategy = profile.get("endpoint_strategy", "both")

    checks: List[PrereqItem] = []
    for key, label, required_for, strategies in _DATA_PATH_SPEC:
        required = strategy in strategies
        item = _verify_one(key, label, required_for, required, data_paths.get(key, ""))
        if item is not None:
            checks.append(item)
    return InfraReport(checks=checks)
