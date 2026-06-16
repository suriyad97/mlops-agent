"""Endpoint requirement plan: ownership, status, dependency order, MDC, strategy filtering."""
import networkx as nx
import yaml

from src.config.settings import get_settings
from src.platform.generation.engine import build_params, render_component
from src.platform.planning.requirement_plan import endpoint_requirement_plan
from src.platform.understanding.contract import propose_contract


def _graph_train_wired_drift_adapter() -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_node("file:src/train.py", kind="file", path="src/train.py",
               entry_point=True, cli_args=["--data-path", "--output-dir"])
    g.add_node("def:src/train.py:main", kind="function", name="main",
               file="src/train.py", calls=["fit"], signature="main()")
    g.add_node("file:analysis/drift.py", kind="file", path="analysis/drift.py",
               entry_point=False, cli_args=[])
    g.add_node("def:analysis/drift.py:compute_psi", kind="function", name="compute_psi",
               file="analysis/drift.py", calls=["evidently"], signature="compute_psi(a, b)")
    return g


def test_plan_ownership_and_blocking_realtime(tmp_path):
    contract = propose_contract(_graph_train_wired_drift_adapter()).for_strategy("realtime").resolved()
    plan = endpoint_requirement_plan(str(tmp_path), "realtime", contract)

    by_req = {i.requirement: i for i in plan.items}

    # DS wrote train.py (compliant) -> their job, reuse
    assert by_req["Training code"].owner == "data_scientist"
    assert by_req["Training code"].status == "present"

    # No scoring code -> model-specific stub -> DS must implement (blocks)
    assert by_req["Scoring code (init/run)"].owner == "data_scientist"
    assert by_req["Scoring code (init/run)"].status == "scaffold"

    # Drift logic exists but non-compliant -> DS owns it via adapter
    assert by_req["Drift detection logic"].owner == "data_scientist"
    assert by_req["Drift detection logic"].status == "adapter"

    # Pipelines/endpoints are the platform's job
    assert by_req["Online endpoint + deployment"].owner == "platform"
    assert by_req["CI pipeline (build + register environment)"].owner == "platform"

    # Infra + user rows present
    assert by_req["Cloud prerequisites (ACR · workspace · compute · ARM connection)"].owner == "infra"
    assert by_req["Training data in Blob Storage"].status == "user_action"

    # Blocking = the two real gaps (scoring stub + training data upload)
    blocking = {b.requirement for b in plan.blocking}
    assert "Scoring code (init/run)" in blocking
    assert "Training data in Blob Storage" in blocking
    # drift adapter and platform defaults do NOT block
    assert "Drift detection logic" not in blocking


def test_plan_default_code_is_platform_job(tmp_path):
    """Stages with a complete platform default (drift/thresholds/rollback) are the platform's
    job when the DS hasn't written their own — not a DS blocker."""
    contract = propose_contract(nx.DiGraph()).for_strategy("batch").resolved()  # empty repo
    plan = endpoint_requirement_plan(str(tmp_path), "batch", contract)
    by_req = {i.requirement: i for i in plan.items}
    # drift/threshold/rollback absent -> platform generates a default
    assert by_req["Drift detection logic"].owner == "platform"
    assert by_req["Drift-threshold rule"].owner == "platform"
    # but scoring (model-specific) is still the DS's blocking job
    assert by_req["Batch scoring code (run)"].owner == "data_scientist"
    assert by_req["Batch scoring code (run)"].status == "scaffold"


def test_plan_strategy_filters_and_mdc(tmp_path):
    rt = endpoint_requirement_plan(str(tmp_path), "realtime", None)
    bt = endpoint_requirement_plan(str(tmp_path), "batch", None)
    rt_reqs = {i.requirement for i in rt.items}
    bt_reqs = {i.requirement for i in bt.items}

    # realtime has MDC + online endpoint, not batch endpoint
    assert any("MDC" in r for r in rt_reqs)
    assert "Online endpoint + deployment" in rt_reqs
    assert "Batch endpoint + deployment" not in rt_reqs

    # batch has the output-file monitoring note + batch endpoint, no MDC
    assert any("Batch output files" in r for r in bt_reqs)
    assert "Batch endpoint + deployment" in bt_reqs
    assert not any("MDC" in r for r in bt_reqs)


def test_plan_is_dependency_ordered():
    plan = endpoint_requirement_plan("", "realtime", None)
    reqs = [i.requirement for i in plan.items]
    # environment before training before endpoint before monitoring
    assert reqs.index("Container / conda definition") < reqs.index("Training code")
    assert reqs.index("Training code") < reqs.index("Online endpoint + deployment")
    assert reqs.index("Online endpoint + deployment") < reqs.index("Monitoring pipeline (AML)")


def test_realtime_deployment_renders_data_collector():
    params = build_params("reco", {"endpoint_strategy": "realtime"}, get_settings())
    rendered = render_component("realtime_deployment", "online_endpoint_assets", params)
    doc = yaml.safe_load(rendered["MLpipelines/deployment.yml"])
    assert "data_collector" in doc
    assert set(doc["data_collector"]["collections"]) == {"model_inputs", "model_outputs"}
    assert doc["code_configuration"]["code"] == "../"   # repo root
