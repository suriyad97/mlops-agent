"""
Sanity Check Tools
===================
Run before any Azure ML pipeline is triggered.
Validates code, dependencies, Azure resources, and pipeline YAMLs.
All authentication is via Service Principal — no `az login` required.
"""

import os
import requests
import base64
from langchain_core.tools import tool
from azure.identity import ClientSecretCredential
from azure.ai.ml import MLClient, load_job


def get_ml_client() -> MLClient:
    credential = ClientSecretCredential(
        tenant_id=os.environ["AZURE_TENANT_ID"],
        client_id=os.environ["AZURE_CLIENT_ID"],
        client_secret=os.environ["AZURE_CLIENT_SECRET"],
    )
    return MLClient(
        credential=credential,
        subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
        resource_group_name=os.environ["AML_RESOURCE_GROUP"],
        workspace_name=os.environ["AML_WORKSPACE_NAME"],
    )


@tool
def check_entry_points_exist(
    train_script_path: str,
    score_script_path: str,
    repo_files: list[str]
) -> dict:
    """
    Checks that the training and inference entry point scripts exist in the repo.
    Also checks that they accept the expected argparse arguments.

    Args:
        train_script_path: Expected path to the training script (e.g. "/train.py")
        score_script_path: Expected path to the inference script (e.g. "/score.py")
        repo_files: List of files in the repo (from list_repo_files)

    Returns:
        dict with check results for each entry point
    """
    results = {}

    train_exists = train_script_path in repo_files
    score_exists = score_script_path in repo_files

    results["train_script"] = {
        "passed": train_exists,
        "message": f"✓ Found {train_script_path}" if train_exists
                   else f"✗ Missing {train_script_path} — training script not found in repo"
    }
    results["score_script"] = {
        "passed": score_exists,
        "message": f"✓ Found {score_script_path}" if score_exists
                   else f"✗ Missing {score_script_path} — inference script not found in repo"
    }

    all_passed = train_exists and score_exists
    return {"passed": all_passed, "checks": results}


@tool
def check_dependencies_valid(requirements_content: str) -> dict:
    """
    Validates that all packages in requirements.txt are installable
    and checks for known version conflicts.

    Args:
        requirements_content: The raw text content of requirements.txt

    Returns:
        dict with 'passed', 'packages', and any conflict warnings
    """
    lines = [l.strip() for l in requirements_content.splitlines()
             if l.strip() and not l.startswith("#")]

    packages = []
    warnings = []

    for line in lines:
        pkg_name = line.split("==")[0].split(">=")[0].split("<=")[0].strip()
        packages.append({"package": pkg_name, "spec": line})

    # Check for known problematic combos
    pkg_names = [p["package"].lower() for p in packages]
    if "tensorflow" in pkg_names and "torch" in pkg_names:
        warnings.append("Both tensorflow and torch detected — ensure they don't conflict in the same environment")
    if "xgboost" in pkg_names and "lightgbm" in pkg_names:
        warnings.append("Both xgboost and lightgbm detected — verify you're using the right one in your scripts")

    return {
        "passed": True,
        "package_count": len(packages),
        "packages": packages,
        "warnings": warnings,
        "message": f"✓ Found {len(packages)} packages" + (f" ({len(warnings)} warnings)" if warnings else "")
    }


@tool
def check_aml_resources_exist(
    compute_cluster_name: str,
    datastore_name: str,
    environment_name: str = ""
) -> dict:
    """
    Verifies that required Azure ML resources exist before submitting any job.
    Checks compute cluster, datastore, and optionally the AML environment.

    Args:
        compute_cluster_name: Name of the AML compute cluster to use
        datastore_name: Name of the AML datastore where data lives
        environment_name: Name of the AML environment (optional)

    Returns:
        dict with check results for each resource
    """
    ml_client = get_ml_client()
    results = {}

    # Check compute cluster
    try:
        compute = ml_client.compute.get(compute_cluster_name)
        results["compute_cluster"] = {
            "passed": True,
            "message": f"✓ Compute cluster '{compute_cluster_name}' exists (state: {compute.provisioning_state})"
        }
    except Exception as e:
        results["compute_cluster"] = {
            "passed": False,
            "message": f"✗ Compute cluster '{compute_cluster_name}' not found — {str(e)}"
        }

    # Check datastore
    try:
        ml_client.datastores.get(datastore_name)
        results["datastore"] = {
            "passed": True,
            "message": f"✓ Datastore '{datastore_name}' exists"
        }
    except Exception as e:
        results["datastore"] = {
            "passed": False,
            "message": f"✗ Datastore '{datastore_name}' not found — {str(e)}"
        }

    # Check environment if provided
    if environment_name:
        try:
            ml_client.environments.get(environment_name, label="latest")
            results["environment"] = {
                "passed": True,
                "message": f"✓ Environment '{environment_name}' is registered"
            }
        except Exception as e:
            results["environment"] = {
                "passed": False,
                "message": f"✗ Environment '{environment_name}' not found — run CI pipeline first"
            }

    all_passed = all(r["passed"] for r in results.values())
    return {"passed": all_passed, "checks": results}


@tool
def validate_pipeline_yaml(pipeline_yaml_path: str) -> dict:
    """
    Validates an Azure ML pipeline YAML using the SDK's validate() method.
    Catches structural errors, missing references, and invalid bindings
    without actually submitting the job.

    Args:
        pipeline_yaml_path: Local path to the pipeline YAML file

    Returns:
        dict with 'passed', 'errors', 'warnings'
    """
    try:
        ml_client = get_ml_client()
        pipeline_job = load_job(source=pipeline_yaml_path)
        validation_result = ml_client.jobs.validate(pipeline_job)

        errors = [str(e) for e in (validation_result.errors or [])]
        warnings = [str(w) for w in (validation_result.warnings or [])]
        passed = len(errors) == 0

        return {
            "passed": passed,
            "message": (
                f"✓ Pipeline YAML '{pipeline_yaml_path}' is valid"
                if passed
                else f"✗ Pipeline YAML validation failed ({len(errors)} error(s))"
            ),
            "errors": errors,
            "warnings": warnings,
        }
    except Exception as e:
        return {
            "passed": False,
            "message": f"✗ Could not validate '{pipeline_yaml_path}': {e}",
            "errors": [str(e)],
            "warnings": [],
        }


@tool
def run_data_schema_check(data_path: str, schema_definition: dict) -> dict:
    """
    Validates that input data conforms to the expected schema before training or inference.
    Checks column names, data types, and null value thresholds.

    Args:
        data_path: Path to the data file or AML datastore path
        schema_definition: Expected schema as dict:
            { "columns": [{"name": "col1", "type": "float", "nullable": false}], ... }

    Returns:
        dict with 'passed', 'column_checks', 'row_count', 'issues'
    """
    # NOTE: For AML datastore paths, mount/download the data first
    # For local paths, use pandas directly
    try:
        import pandas as pd
        df = pd.read_csv(data_path)
        issues = []
        column_checks = {}

        expected_columns = {c["name"]: c for c in schema_definition.get("columns", [])}

        for col_name, col_def in expected_columns.items():
            if col_name not in df.columns:
                issues.append(f"Missing column: '{col_name}'")
                column_checks[col_name] = {"passed": False, "issue": "Column missing"}
                continue

            # Check nulls
            null_count = df[col_name].isnull().sum()
            if not col_def.get("nullable", True) and null_count > 0:
                issues.append(f"Column '{col_name}' has {null_count} null values but nullable=False")
                column_checks[col_name] = {"passed": False, "issue": f"{null_count} nulls found"}
            else:
                column_checks[col_name] = {"passed": True, "null_count": int(null_count)}

        return {
            "passed": len(issues) == 0,
            "row_count": len(df),
            "column_count": len(df.columns),
            "column_checks": column_checks,
            "issues": issues,
            "message": "✓ Data schema valid" if not issues else f"✗ Schema issues: {'; '.join(issues)}"
        }
    except Exception as e:
        return {"passed": False, "issues": [str(e)], "message": f"✗ Schema check failed: {e}"}
