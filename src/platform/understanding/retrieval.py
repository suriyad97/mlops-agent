"""Graph retrieval: token-bounded digest, queries, semantic evidence search.

The LLM never receives the repository — only what these functions return.
"""
import json
from pathlib import Path
from typing import Dict, List

import networkx as nx

GRAPHS_DIR = Path(".graphs")


def save_graph(graph: nx.DiGraph, project_id: str) -> str:
    GRAPHS_DIR.mkdir(exist_ok=True)
    target = GRAPHS_DIR / f"{project_id}.json"
    data = nx.node_link_data(graph, edges="links")
    target.write_text(json.dumps(data), encoding="utf-8")
    return str(target)


def load_graph(project_id: str) -> nx.DiGraph:
    target = GRAPHS_DIR / f"{project_id}.json"
    data = json.loads(target.read_text(encoding="utf-8"))
    return nx.node_link_graph(data, edges="links", directed=True)


def graph_stats(graph: nx.DiGraph) -> Dict[str, int]:
    kinds: Dict[str, int] = {}
    for _, attrs in graph.nodes(data=True):
        kinds[attrs.get("kind", "?")] = kinds.get(attrs.get("kind", "?"), 0) + 1
    return {"nodes": graph.number_of_nodes(), "edges": graph.number_of_edges(), **kinds}


def entry_points(graph: nx.DiGraph) -> List[dict]:
    result = []
    for node, attrs in graph.nodes(data=True):
        if attrs.get("kind") == "file" and attrs.get("entry_point"):
            result.append({
                "path": attrs["path"],
                "cli_args": attrs.get("cli_args", []),
                "functions": [
                    graph.nodes[d].get("signature", "")
                    for d in graph.successors(node)
                    if graph.nodes[d].get("kind") == "function"
                ],
            })
    return result


def assets_by_role(graph: nx.DiGraph, role: str) -> List[str]:
    return [
        attrs["path"] for _, attrs in graph.nodes(data=True)
        if attrs.get("kind") == "file" and attrs.get("role") == role
    ]


def suggest_paths(graph: nx.DiGraph, path: str, limit: int = 5) -> List[str]:
    """Close matches for a path that isn't in the graph (same filename or substring)."""
    name = path.rsplit("/", 1)[-1].lower()
    stem = name.rsplit(".", 1)[0]
    candidates = []
    for _, attrs in graph.nodes(data=True):
        if attrs.get("kind") != "file":
            continue
        candidate = attrs["path"]
        lowered = candidate.lower()
        if lowered.endswith("/" + name) or lowered == name:
            candidates.append((0, candidate))          # exact filename match
        elif stem and stem in lowered:
            candidates.append((1, candidate))          # stem appears in path
    candidates.sort()
    return [c for _, c in candidates[:limit]]


def who_references(graph: nx.DiGraph, path: str) -> List[str] | str:
    """Referrers of a file; a descriptive string when the file doesn't exist."""
    node = f"file:{path}"
    if node not in graph:
        suggestions = suggest_paths(graph, path)
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        return f"FILE NOT FOUND: '{path}' is not in the repository.{hint}"
    return [
        graph.nodes[p].get("path", p)
        for p in graph.predecessors(node)
        if graph[p][node].get("kind") in ("references", "imports", "submits")
    ]


def what_does(graph: nx.DiGraph, path: str) -> str:
    node = f"file:{path}"
    if node not in graph:
        suggestions = suggest_paths(graph, path)
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        return f"FILE NOT FOUND: '{path}' is not in the repository.{hint}"
    attrs = graph.nodes[node]
    lines = [f"{path} ({attrs.get('lang')}, role={attrs.get('role') or 'code'})"]
    for successor in graph.successors(node):
        s_attrs = graph.nodes[successor]
        if s_attrs.get("kind") in ("function", "class"):
            lines.append(f"  - {s_attrs.get('summary', '')}")
        elif graph[node][successor].get("kind") in ("imports", "references", "submits"):
            lines.append(f"  -> {graph[node][successor]['kind']} {s_attrs.get('path', successor)}")
    return "\n".join(lines)


def find_evidence(graph: nx.DiGraph, indicators: List[str]) -> List[dict]:
    """Substring search over node summaries — the evaluator's evidence primitive."""
    hits = []
    lowered = [i.lower() for i in indicators]
    for node, attrs in graph.nodes(data=True):
        haystack = " ".join([
            str(attrs.get("summary", "")), str(attrs.get("name", "")),
            str(attrs.get("path", "")), str(attrs.get("doc", "")),
            " ".join(attrs.get("calls", []) or []),
        ]).lower()
        matched = [i for i in lowered if i in haystack]
        if matched:
            hits.append({
                "node": node,
                "kind": attrs.get("kind"),
                "file": attrs.get("file") or attrs.get("path"),
                "matched": matched,
                "summary": str(attrs.get("summary", ""))[:200],
            })
    hits.sort(key=lambda h: len(h["matched"]), reverse=True)
    return hits[:25]


def graph_digest(graph: nx.DiGraph, max_chars: int = 6000) -> str:
    """Centrality-ranked digest: entry points, assets, orchestration, key code."""
    sections: List[str] = []

    eps = entry_points(graph)
    if eps:
        sections.append("ENTRY POINTS:")
        for ep in eps[:8]:
            args = f" args={ep['cli_args']}" if ep["cli_args"] else ""
            sections.append(f"  {ep['path']}{args}")
            for fn in ep["functions"][:6]:
                sections.append(f"    def {fn}")

    for role, label in (("aml_asset", "AML ASSETS"), ("azdo_pipeline", "AZDO PIPELINES"),
                        ("github_workflow", "GITHUB WORKFLOWS")):
        paths = assets_by_role(graph, role)
        if paths:
            sections.append(f"{label}:")
            for p in paths[:15]:
                refs = [
                    graph.nodes[s].get("path", "")
                    for s in graph.successors(f"file:{p}")
                    if graph[f"file:{p}"][s].get("kind") in ("references", "submits")
                ]
                suffix = f" -> {', '.join(refs[:4])}" if refs else ""
                sections.append(f"  {p}{suffix}")

    centrality = nx.degree_centrality(graph)
    code_files = sorted(
        (n for n, a in graph.nodes(data=True) if a.get("kind") == "file" and a.get("lang") == "python"),
        key=lambda n: centrality.get(n, 0), reverse=True,
    )
    sections.append("KEY CODE (by connectivity):")
    for node in code_files[:10]:
        path = graph.nodes[node]["path"]
        defs = [
            graph.nodes[d].get("signature", "")
            for d in graph.successors(node)
            if graph.nodes[d].get("kind") in ("function", "class")
        ]
        sections.append(f"  {path}: {', '.join(defs[:8])}")

    return "\n".join(sections)[:max_chars]
