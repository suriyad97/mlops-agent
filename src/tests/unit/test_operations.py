from src.app.services import azdo_pipeline_files


def test_azdo_pipeline_files_filters_correctly():
    files = [
        "aml/pipelines/monitoring_pipeline.yml",
        "azdopipelines/cm-monitoring.yml",
        "azdopipelines/ct-retrain.yml",
        "azure-pipelines.yml",
        "src/monitoring/detect_drift.py",
        ".azuredevops/rollback-runbook.md",
    ]
    assert azdo_pipeline_files(files) == [
        "azdopipelines/cm-monitoring.yml",
        "azdopipelines/ct-retrain.yml",
        "azure-pipelines.yml",
    ]


def test_azdo_pipeline_files_empty():
    assert azdo_pipeline_files(["src/train.py"]) == []
