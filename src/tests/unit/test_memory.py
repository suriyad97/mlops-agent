from src.memory.json_store import JsonMemoryStore


def test_add_and_search(tmp_path):
    store = JsonMemoryStore(path=str(tmp_path / "mem.json"))
    store.record_fix("ModuleNotFoundError pandas missing", "added pandas to requirements", ["requirements.txt"])
    store.record_failure("docker build OOM", "build killed")

    fixes = store.search(kind="fix")
    assert len(fixes) == 1
    assert fixes[0].payload["files"] == ["requirements.txt"]
    assert len(store.all()) == 2


def test_similar_fixes_ranks_by_overlap(tmp_path):
    store = JsonMemoryStore(path=str(tmp_path / "mem.json"))
    store.record_fix("ModuleNotFoundError pandas missing import", "add pandas", ["requirements.txt"])
    store.record_fix("docker daemon unreachable", "start docker", [])

    hits = store.similar_fixes("pipeline failed with ModuleNotFoundError: pandas")
    assert hits
    assert "pandas" in hits[0].key


def test_persistence_across_instances(tmp_path):
    path = str(tmp_path / "mem.json")
    JsonMemoryStore(path=path).record_fix("err sig", "summary", [])
    assert len(JsonMemoryStore(path=path).all()) == 1
