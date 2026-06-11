from src.tools.validation_tools import (
    validate_aml_asset_yaml,
    validate_azdo_pipeline_yaml,
    validate_dockerfile,
    validate_yaml,
)


def test_valid_yaml():
    ok, err = validate_yaml("a: 1\nb:\n  - x\n")
    assert ok and not err


def test_invalid_yaml():
    ok, err = validate_yaml("a: [unclosed")
    assert not ok and "YAML" in err


def test_good_dockerfile():
    content = "FROM python:3.12-slim\nRUN pip install -r requirements.txt\nUSER app\nCMD ['python', 'main.py']\n"
    ok, err = validate_dockerfile(content)
    assert ok, err


def test_dockerfile_latest_tag_flagged():
    ok, err = validate_dockerfile("FROM python:latest\nCMD ['python']\n")
    assert not ok
    assert "latest" in err


def test_dockerfile_missing_cmd_flagged():
    ok, err = validate_dockerfile("FROM python:3.12-slim\nRUN echo hi\n")
    assert not ok
    assert "CMD" in err


def test_azdo_pipeline_requires_steps_and_trigger():
    ok, err = validate_azdo_pipeline_yaml("variables:\n  a: 1\n")
    assert not ok
    good = "trigger:\n  - main\nsteps:\n  - script: echo hi\n"
    ok, err = validate_azdo_pipeline_yaml(good)
    assert ok, err


def test_azdo_pipeline_rejects_variable_groups_by_default():
    content = (
        "trigger:\n  - main\n"
        "variables:\n  - group: python-ci-config\n"
        "steps:\n  - script: echo hi\n"
    )
    ok, err = validate_azdo_pipeline_yaml(content)
    assert not ok
    assert "python-ci-config" in err


def test_azdo_pipeline_rejects_nested_variable_groups():
    content = (
        "trigger:\n  - main\n"
        "stages:\n"
        "  - stage: CI\n"
        "    variables:\n      - group: secrets\n"
        "    jobs:\n      - job: a\n        steps:\n          - script: echo hi\n"
    )
    ok, err = validate_azdo_pipeline_yaml(content)
    assert not ok
    assert "secrets" in err


def test_azdo_pipeline_variable_groups_allowed_when_flagged():
    content = (
        "trigger:\n  - main\n"
        "variables:\n  - group: python-ci-config\n"
        "steps:\n  - script: echo hi\n"
    )
    ok, err = validate_azdo_pipeline_yaml(content, allow_variable_groups=True)
    assert ok, err


def test_azdo_pipeline_inline_variables_fine():
    content = (
        "trigger:\n  - main\n"
        "variables:\n  PIP_CACHE: $(Pipeline.Workspace)/.pip\n"
        "steps:\n  - script: echo hi\n"
    )
    ok, err = validate_azdo_pipeline_yaml(content)
    assert ok, err


def test_aml_asset_requires_schema_and_name():
    ok, err = validate_aml_asset_yaml("foo: bar\n")
    assert not ok
    good = (
        "$schema: https://azuremlschemas.azureedge.net/latest/environment.schema.json\n"
        "name: train-env\n"
    )
    ok, err = validate_aml_asset_yaml(good, "environment")
    assert ok, err
