"""Self-healing loop for validation failures: RCA -> fix -> re-validate, bounded.

Fixes are constrained to the files that failed validation. Every cycle is
recorded to failure memory so future RCAs see prior fixes for similar errors.
"""
from pathlib import Path
from typing import Callable, List

from src.llm.factory import get_provider
from src.llm.prompts.base import MLOPS_EXPERT_PREAMBLE
from src.llm.structured_outputs.models import FixBundle
from src.platform.validation.framework import ValidationReport
from src.shared.logging import get_logger
from src.tools.file_tools import read_file, write_file

logger = get_logger(__name__)

HEAL_SYSTEM = MLOPS_EXPERT_PREAMBLE + """
Generated MLOps assets failed validation. Fix them. For each file that must
change, return its COMPLETE corrected content. Only touch the files listed as
failing. Keep approval gates and pipeline stages intact.
Set retriable=false ONLY when NONE of the failures can be fixed by editing
files. If SOME failures are fixable, fix all of those (retriable=true) and
list the human-only tasks (missing service connections, permissions) in the
summary instead.
"""


def heal(
    repo_path: str,
    report: ValidationReport,
    revalidate: Callable[[], ValidationReport],
    record_failure: Callable[[str, str, dict, str], None],
    max_attempts: int = 2,
) -> ValidationReport:
    """Attempt to fix failures and re-validate. Returns the final report."""
    provider = get_provider()
    current = report

    for attempt in range(1, max_attempts + 1):
        if current.passed:
            return current
        failing_files = sorted({f.target for t in current.tiers for f in t.failures})
        failures_text = current.failures_text()

        user = f"Validation failures:\n{failures_text}\n\nCurrent content of failing files:"
        for rel in failing_files[:4]:
            try:
                user += f"\n--- {rel} ---\n{read_file(repo_path, rel, max_bytes=4000)}"
            except Exception:
                continue

        try:
            bundle = provider.complete_structured(FixBundle, system=HEAL_SYSTEM, user=user)
        except Exception as exc:
            logger.warning("heal attempt %d: fix generation failed: %s", attempt, exc)
            record_failure(failures_text[:300], "fix generation failed", {}, "heal_error")
            return current

        if not bundle.retriable:
            record_failure(failures_text[:300], bundle.summary, bundle.model_dump(), "not_retriable")
            current.healed_files.append(f"NOT RETRIABLE: {bundle.summary[:200]}")
            return current

        applied: List[str] = []
        for fix in bundle.fixes:
            if fix.file_path not in failing_files:
                logger.warning("heal: skipping fix for non-failing file %s", fix.file_path)
                continue
            write_file(repo_path, fix.file_path, fix.new_content)
            applied.append(fix.file_path)

        if not applied:
            record_failure(failures_text[:300], "no applicable fixes", bundle.model_dump(), "no_fix")
            return current

        new_report = revalidate()
        new_report.healed_files = current.healed_files + applied
        outcome = "fixed" if new_report.passed else "still_failing"
        record_failure(failures_text[:300], bundle.summary, bundle.model_dump(), outcome)
        logger.info("heal attempt %d: applied=%s outcome=%s", attempt, applied, outcome)
        current = new_report

    return current
