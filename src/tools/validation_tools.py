"""Stateless static validators for generated artifacts."""
from typing import List, Tuple

import yaml


def validate_yaml(content: str) -> Tuple[bool, str]:
    # Guard non-string input: PyYAML treats a dict/list as a stream and calls .read()
    # on it, raising "'dict' object has no attribute 'read'" (an AttributeError, NOT a
    # YAMLError) — which would otherwise escape and crash generation. Reject cleanly.
    if not isinstance(content, str):
        return False, f"expected YAML text, got {type(content).__name__}"
    try:
        yaml.safe_load(content)
        return True, ""
    except yaml.YAMLError as exc:
        return False, f"YAML parse error: {exc}"


def validate_dockerfile(content: str) -> Tuple[bool, str]:
    problems: List[str] = []
    lines = [l.strip() for l in content.splitlines() if l.strip() and not l.strip().startswith("#")]
    if not lines:
        return False, "Dockerfile is empty"
    if not lines[0].upper().startswith(("FROM", "ARG")):
        problems.append("first instruction must be FROM (or ARG before FROM)")
    if not any(l.upper().startswith("FROM") for l in lines):
        problems.append("missing FROM instruction")
    if any(l.upper().startswith("FROM") and l.rstrip().endswith(":latest") for l in lines):
        problems.append("base image pinned to :latest — pin an explicit version")
    if not any(l.upper().startswith(("CMD", "ENTRYPOINT")) for l in lines):
        problems.append("missing CMD or ENTRYPOINT")
    if any("ADD http" in l for l in lines):
        problems.append("avoid ADD with remote URLs — use curl in RUN instead")
    return (not problems), "; ".join(problems)


def _find_variable_groups(node) -> List[str]:
    """Recursively collect variable-group names referenced anywhere in the doc."""
    groups: List[str] = []
    if isinstance(node, dict):
        variables = node.get("variables")
        if isinstance(variables, list):
            groups.extend(
                str(entry["group"]) for entry in variables
                if isinstance(entry, dict) and "group" in entry
            )
        for value in node.values():
            groups.extend(_find_variable_groups(value))
    elif isinstance(node, list):
        for item in node:
            groups.extend(_find_variable_groups(item))
    return groups


def validate_azdo_pipeline_yaml(content: str, allow_variable_groups: bool = False) -> Tuple[bool, str]:
    ok, err = validate_yaml(content)
    if not ok:
        return False, err
    doc = yaml.safe_load(content) or {}
    problems: List[str] = []
    if not any(k in doc for k in ("steps", "jobs", "stages")):
        problems.append("pipeline must define steps, jobs, or stages")
    if "trigger" not in doc and "pr" not in doc:
        problems.append("no trigger defined — add trigger: or pr:")
    if not allow_variable_groups:
        groups = _find_variable_groups(doc)
        if groups:
            problems.append(
                f"references variable group(s) {groups} that are not configured/authorized "
                "in the project — inline the variables instead"
            )
    return (not problems), "; ".join(problems)


def validate_aml_asset_yaml(content: str, expected_schema_fragment: str = "") -> Tuple[bool, str]:
    ok, err = validate_yaml(content)
    if not ok:
        return False, err
    doc = yaml.safe_load(content) or {}
    problems: List[str] = []
    if "$schema" not in doc:
        problems.append("missing $schema — AML assets should declare their schema")
    if expected_schema_fragment and expected_schema_fragment not in str(doc.get("$schema", "")):
        problems.append(f"$schema does not look like a {expected_schema_fragment} schema")
    if "name" not in doc and "display_name" not in doc:
        problems.append("missing name (or display_name for pipeline jobs)")
    return (not problems), "; ".join(problems)
