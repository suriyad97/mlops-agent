import pytest

from src.platform.validation.framework import (
    run_validation,
    tier_local_exec,
    tier_static,
)


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "azdopipelines").mkdir()
    (tmp_path / "aml").mkdir()
    (tmp_path / "src").mkdir()
    return tmp_path


def test_static_catches_bad_azdo_pipeline(repo):
    (repo / "azdopipelines" / "bad.yml").write_text(
        "trigger: [main]\nvariables:\n  - group: nope\nsteps:\n  - script: echo hi\n"
    )
    result = tier_static(str(repo), ["azdopipelines/bad.yml"])
    assert not result.passed
    assert "nope" in result.failures[0].error


def test_static_passes_good_files(repo):
    (repo / "azdopipelines" / "good.yml").write_text(
        "trigger: [main]\nsteps:\n  - script: echo hi\n"
    )
    (repo / "aml" / "env.yml").write_text(
        "$schema: https://azuremlschemas.azureedge.net/latest/environment.schema.json\nname: env\n"
    )
    result = tier_static(str(repo), ["azdopipelines/good.yml", "aml/env.yml"])
    assert result.passed, result.failures


def test_local_exec_catches_syntax_error(repo):
    (repo / "src" / "broken.py").write_text("def broken(:\n    pass\n")
    (repo / "src" / "fine.py").write_text("def fine():\n    return 1\n")
    result = tier_local_exec(str(repo), ["src/broken.py", "src/fine.py"])
    assert not result.passed
    assert result.failures[0].target == "src/broken.py"


def test_run_validation_aggregates_and_skips_unconfigured(repo, monkeypatch):
    monkeypatch.delenv("AZDO_ORG_URL", raising=False)
    monkeypatch.delenv("AZDO_PROJECT", raising=False)
    from src.config.settings import get_settings
    get_settings.cache_clear()

    (repo / "src" / "ok.py").write_text("x = 1\n")
    (repo / "azdopipelines" / "ok.yml").write_text("trigger: [main]\nsteps:\n  - script: echo hi\n")
    report = run_validation(str(repo), ["src/ok.py", "azdopipelines/ok.yml"])
    tiers = {t.tier: t for t in report.tiers}
    assert tiers["static"].passed
    assert tiers["local_exec"].passed
    assert tiers["azdo_preview"].skipped  # AzDO not configured in this test
    assert report.passed
    get_settings.cache_clear()


def test_failures_text_format(repo):
    (repo / "src" / "broken.py").write_text("def broken(:\n")
    report = run_validation(str(repo), ["src/broken.py"])
    assert "[local_exec] src/broken.py" in report.failures_text()
