"""Golden-path generation: contract pre-fill, strategy-driven full set, wired/adapter/scaffold."""
import networkx as nx
import pytest

from src.config.settings import get_settings
from src.platform.evaluation.evaluator import standard_generation_set
from src.platform.generation.engine import build_params, generate
from src.platform.understanding.contract import (
    STAGE_REGISTRY,
    ContractManifest,
    propose_contract,
)


def _file(g, rel, *, entry=False, cli=None):
    g.add_node(f"file:{rel}", kind="file", path=rel, entry_point=entry, cli_args=cli or [])


def _func(g, rel, name, *, calls=None, sig=""):
    g.add_node(f"def:{rel}:{name}", kind="function", name=name, file=rel,
               calls=calls or [], signature=sig)


@pytest.fixture
def mixed_repo_graph():
    """A repo with: compliant training (wired), non-compliant drift (adapter), rest absent."""
    g = nx.DiGraph()
    _file(g, "recommender/model.py", entry=True, cli=["--data-path", "--output-dir"])
    _func(g, "recommender/model.py", "train_als", calls=["fit", "read_csv"], sig="train_als(data_path, output_dir)")
    _file(g, "src/score.py")
    _func(g, "src/score.py", "init", calls=["load_model"], sig="init()")
    _func(g, "src/score.py", "run", calls=["predict"], sig="run(data)")
    _file(g, "analysis/drift.py")
    _func(g, "analysis/drift.py", "compute_psi", calls=["evidently"], sig="compute_psi(a, b)")
    return g


# --- standard_generation_set ------------------------------------------------

def test_standard_set_is_full_and_dependency_ordered():
    s = standard_generation_set("realtime")
    caps = [it.capability for it in s.items]
    # environment first, retraining last; batch capability excluded for realtime strategy
    assert caps[0] == "environment_lifecycle"
    assert caps[-1] == "retraining"
    assert "batch_deployment" not in caps
    assert "realtime_deployment" in caps
    # every item carries its deliverable paths (the standard template set)
    assert all(it.deliverables for it in s.items)


def test_standard_set_strategy_filtering():
    assert any(it.capability == "batch_deployment" for it in standard_generation_set("batch").items)
    none_caps = {it.capability for it in standard_generation_set("none").items}
    assert none_caps == {"environment_lifecycle", "training", "model_lifecycle"}


# --- propose_contract -------------------------------------------------------

def test_propose_contract_resolves_modes(mixed_repo_graph):
    m = propose_contract(mixed_repo_graph)
    by = {s.stage: s for s in m.stages}
    # compliant CLI training at a non-standard path -> wired, pipeline points at it
    assert by["training"].mode == "wired"
    assert by["training"].pipeline_path() == "recommender/model.py"
    # compliant init+run scoring already at standard path -> wired
    assert by["scoring_realtime"].mode == "wired"
    # drift logic present but no CLI/entrypoint -> adapter at the standard path
    assert by["drift"].mode == "adapter"
    assert by["drift"].detected_symbol == "compute_psi"
    # nothing for evaluation -> scaffold
    assert by["evaluation"].mode == "scaffold"


def test_contract_registry_covers_seven_stages():
    assert len(STAGE_REGISTRY) == 7
    m = propose_contract(nx.DiGraph())
    assert all(s.mode == "scaffold" for s in m.stages)   # empty repo -> all scaffold


# --- end-to-end generation with the contract --------------------------------

def test_generate_full_set_with_contract(tmp_path, mixed_repo_graph):
    contract = propose_contract(mixed_repo_graph).for_strategy("realtime").resolved()
    items = [it.model_dump() for it in standard_generation_set("realtime").items]
    params = build_params("reco_proj", {"endpoint_strategy": "realtime"}, get_settings())
    existing = [".azuredevops/old-train.yml", "azdopipelines/ci-environment.yml"]

    rep = generate(str(tmp_path), items, params, repo_context="", adapt=False,
                   contract=contract, existing_pipeline_files=existing)

    # wired training: src/train.py must NOT be written (user's file is reused)
    assert not (tmp_path / "src" / "train.py").exists()
    assert "src/train.py" in rep.wired_skipped

    # adapter drift: file exists and imports the user's module/symbol, untouched
    drift = tmp_path / "src" / "monitoring" / "detect_drift.py"
    assert drift.exists() and "src/monitoring/detect_drift.py" in rep.adapter_files
    body = drift.read_text()
    assert "from analysis.drift import compute_psi" in body
    assert "ADAPTER" in body

    # scaffold evaluation: standard template written
    assert (tmp_path / "src" / "evaluate.py").exists()
    assert "src/evaluate.py" in rep.scaffold_files

    # superseded: legacy pipeline not in the standard set flagged; standard one is not
    assert ".azuredevops/old-train.yml" in rep.superseded_files
    assert "azdopipelines/ci-environment.yml" not in rep.superseded_files


def test_generate_without_contract_writes_everything(tmp_path):
    """Backward-compat: contract=None keeps the legacy behavior (all code files written)."""
    items = [{"capability": "training", "component": "code_functions",
              "deliverables": ["src/train.py", "src/evaluate.py"]}]
    params = build_params("p", {}, get_settings())
    rep = generate(str(tmp_path), items, params, repo_context="", adapt=False)
    assert (tmp_path / "src" / "train.py").exists()
    assert (tmp_path / "src" / "evaluate.py").exists()
    assert rep.wired_skipped == [] and rep.adapter_files == []
