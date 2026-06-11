import pytest

from src.platform.catalog.loader import dependency_order, load_catalog
from src.platform.evaluation.evaluator import (
    CapabilityReport,
    EvidenceJudgment,
    _score,
    gap_analysis,
)


def test_catalog_loads_and_validates():
    catalog = load_catalog()
    assert len(catalog) == 8
    assert "training" in catalog and "environment_lifecycle" in catalog
    for spec in catalog.values():
        assert sum(e.weight for e in spec.evidence) == 100


def test_dependency_order_roots_first():
    catalog = load_catalog()
    order = dependency_order(catalog)
    assert order.index("environment_lifecycle") < order.index("training")
    assert order.index("training") < order.index("model_lifecycle")
    assert order.index("model_lifecycle") < order.index("realtime_deployment")
    assert order.index("monitoring") < order.index("retraining")


def _judgment(eid: str, status: str) -> EvidenceJudgment:
    return EvidenceJudgment(evidence_id=eid, status=status)


def test_scoring_complete():
    spec = load_catalog()["training"]
    judgments = {f"training.{e.id}": _judgment(f"training.{e.id}", "found") for e in spec.evidence}
    status = _score(spec, judgments)
    assert status.score == 100
    assert status.status == "complete"
    assert status.missing_components == []


def test_scoring_partial_with_weak_and_absent():
    spec = load_catalog()["training"]
    judgments = {
        "training.code_functions": _judgment("training.code_functions", "found"),    # 30
        "training.aml_training_assets": _judgment("training.aml_training_assets", "weak"),  # 20
        "training.azdo_orchestration": _judgment("training.azdo_orchestration", "absent"),  # 0
    }
    status = _score(spec, judgments)
    assert status.score == 50
    assert status.status == "partial"
    assert set(status.missing_components) == {"aml_training_assets", "azdo_orchestration"}


def test_scoring_missing_when_all_absent():
    spec = load_catalog()["rollback"]
    judgments = {f"rollback.{e.id}": _judgment(f"rollback.{e.id}", "absent") for e in spec.evidence}
    status = _score(spec, judgments)
    assert status.score == 0
    assert status.status == "missing"


def test_gap_analysis_orders_by_dependency_and_maps_deliverables():
    catalog = load_catalog()
    statuses = []
    for name in ("retraining", "environment_lifecycle"):   # deliberately wrong order
        spec = catalog[name]
        judgments = {f"{name}.{e.id}": _judgment(f"{name}.{e.id}", "absent") for e in spec.evidence}
        statuses.append(_score(spec, judgments))
    report = CapabilityReport(capabilities=statuses)
    gaps = gap_analysis(report)
    capabilities_in_order = [item.capability for item in gaps.items]
    assert capabilities_in_order.index("environment_lifecycle") < capabilities_in_order.index("retraining")
    env_items = [i for i in gaps.items if i.capability == "environment_lifecycle"]
    assert any("aml/environment.yml" in i.deliverables for i in env_items)
