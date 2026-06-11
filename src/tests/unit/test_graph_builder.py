import pytest

from src.platform.understanding.graph_builder import build_graph
from src.platform.understanding.retrieval import (
    assets_by_role,
    entry_points,
    find_evidence,
    graph_digest,
    load_graph,
    save_graph,
    what_does,
    who_references,
)

TRAIN_PY = '''
import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression
from utils import load_data

def train_model(data_path, max_iter):
    """Train the classifier."""
    df = load_data(data_path)
    model = LogisticRegression(max_iter=max_iter)
    model.fit(df.drop(columns=["survived"]), df["survived"])
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path")
    parser.add_argument("--max-iter", type=int)
    args = parser.parse_args()
    train_model(args.data_path, args.max_iter)
'''

UTILS_PY = '''
import pandas as pd

def load_data(path):
    """Load the dataset."""
    return pd.read_csv(path)
'''

AML_PIPELINE = '''
$schema: https://azuremlschemas.azureedge.net/latest/pipelineJob.schema.json
name: training-pipeline
jobs:
  train:
    command: python train.py --data-path ${{inputs.data}} --max-iter 100
    code: .
'''

AZDO_PIPELINE = '''
trigger: [main]
stages:
  - stage: Train
    jobs:
      - job: submit
        steps:
          - script: az ml job create --file aml/pipeline.yml
'''


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "train.py").write_text(TRAIN_PY)
    (tmp_path / "utils.py").write_text(UTILS_PY)
    (tmp_path / "aml").mkdir()
    (tmp_path / "aml" / "pipeline.yml").write_text(AML_PIPELINE)
    (tmp_path / "azure-pipelines.yml").write_text(AZDO_PIPELINE)
    return str(tmp_path)


def test_defs_and_imports(repo):
    graph = build_graph(repo)
    assert "def:train.py:train_model" in graph
    assert graph.nodes["def:train.py:train_model"]["signature"] == "train_model(data_path, max_iter)"
    assert graph.has_edge("file:train.py", "file:utils.py")  # imports utils


def test_call_edges(repo):
    graph = build_graph(repo)
    assert graph.has_edge("def:train.py:train_model", "def:utils.py:load_data")


def test_entry_point_with_cli_args(repo):
    graph = build_graph(repo)
    eps = entry_points(graph)
    train = next(e for e in eps if e["path"] == "train.py")
    assert "--data-path" in train["cli_args"]


def test_yaml_roles_and_orchestration(repo):
    graph = build_graph(repo)
    assert assets_by_role(graph, "aml_asset") == ["aml/pipeline.yml"]
    assert assets_by_role(graph, "azdo_pipeline") == ["azure-pipelines.yml"]
    # azdo submits aml; aml references train.py
    assert graph.has_edge("file:azure-pipelines.yml", "file:aml/pipeline.yml")
    assert graph["file:azure-pipelines.yml"]["file:aml/pipeline.yml"]["kind"] in ("submits", "references")
    assert graph.has_edge("file:aml/pipeline.yml", "file:train.py")


def test_who_references_and_what_does(repo):
    graph = build_graph(repo)
    assert "aml/pipeline.yml" in who_references(graph, "train.py")
    description = what_does(graph, "train.py")
    assert "train_model" in description


def test_who_references_missing_file_suggests_alternatives(repo):
    graph = build_graph(repo)
    result = who_references(graph, "mlpipelines/training/train.py")
    assert isinstance(result, str)
    assert "FILE NOT FOUND" in result
    assert "train.py" in result  # suggests the real train.py


def test_what_does_missing_file_suggests(repo):
    graph = build_graph(repo)
    result = what_does(graph, "src/utils.py")
    assert "FILE NOT FOUND" in result
    assert "utils.py" in result


def test_find_evidence(repo):
    graph = build_graph(repo)
    hits = find_evidence(graph, ["fit", "LogisticRegression", "train"])
    assert hits
    assert any("train" in str(h["file"]) for h in hits)


def test_digest_contains_key_sections(repo):
    graph = build_graph(repo)
    digest = graph_digest(graph)
    assert "ENTRY POINTS" in digest
    assert "AML ASSETS" in digest
    assert "train.py" in digest


def test_save_and_load_roundtrip(repo, tmp_path, monkeypatch):
    import src.platform.understanding.retrieval as retrieval
    monkeypatch.setattr(retrieval, "GRAPHS_DIR", tmp_path / "graphs")
    graph = build_graph(repo)
    save_graph(graph, "testproj")
    loaded = load_graph("testproj")
    assert loaded.number_of_nodes() == graph.number_of_nodes()
    assert "def:train.py:train_model" in loaded
