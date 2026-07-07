"""Repository knowledge graph builder — deterministic, local, no LLM.

Nodes:
  file:<rel>           every repo file (kind=file, lang, role hints)
  def:<rel>:<name>     functions/classes from Python AST (signature, doc, summary)

Edges (kind=):
  defines     file -> def
  imports     file -> file              (repo-internal import resolution)
  calls       def -> def                (best-effort, name-matched within repo)
  references  yaml/config file -> file  (YAML mentions a repo file path)
  submits     azdo yaml -> aml yaml     (orchestration: AzDO submits AML asset)
"""
import ast
import re
from pathlib import Path
from typing import Dict, List

import networkx as nx
import yaml

from src.tools.file_tools import scan_repo

PY_EXT = {".py"}
YAML_EXT = {".yml", ".yaml"}
SKIP_DIRS = {"node_modules", ".venv", "venv", "__pycache__", "dist", "build"}
MAX_FILE_BYTES = 200_000

# Extensionless / special files the graph should recognise by name.
DOCKERFILE_NAMES = frozenset({"dockerfile"})
DOCKERFILE_PREFIXES = ("dockerfile.",)   # dockerfile.dev, dockerfile.prod etc.
EXTENSIONLESS_ROLES = {
    "makefile": "build_script",
    "procfile": "runtime_config",
    ".dockerignore": "docker_config",
    ".gitignore": "vcs_config",
}


def _lang(path: str) -> str:
    suffix = Path(path).suffix.lower()
    return {".py": "python", ".yml": "yaml", ".yaml": "yaml", ".sh": "shell",
            ".md": "markdown", ".json": "json", ".txt": "text"}.get(suffix, "other")


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = [a.arg for a in node.args.args]
    return f"{node.name}({', '.join(args)})"


def _called_names(node: ast.AST) -> List[str]:
    names = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Name):
                names.append(func.id)
            elif isinstance(func, ast.Attribute):
                names.append(func.attr)
    return names


def _add_python_file(graph: nx.DiGraph, rel: str, source: str, module_index: Dict[str, str]) -> None:
    file_node = f"file:{rel}"
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return

    has_main_guard = "__main__" in source
    uses_argparse = "argparse" in source or "ArgumentParser" in source
    cli_args = re.findall(r"add_argument\(\s*['\"](--?[\w-]+)", source)
    graph.nodes[file_node].update({
        "entry_point": has_main_guard or uses_argparse,
        "cli_args": cli_args,
    })

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = (
                [a.name for a in node.names] if isinstance(node, ast.Import)
                else [node.module] if node.module else []
            )
            for module in modules:
                target = module_index.get(module) or module_index.get(module.split(".")[0])
                if target and target != rel:
                    graph.add_edge(file_node, f"file:{target}", kind="imports")

    for node in tree.body if isinstance(tree, ast.Module) else []:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = node.name
            def_node = f"def:{rel}:{name}"
            if isinstance(node, ast.ClassDef):
                signature, kind = name, "class"
            else:
                signature, kind = _signature(node), "function"
            doc = (ast.get_docstring(node) or "").split("\n")[0][:120]
            called = _called_names(node)[:30]
            graph.add_node(
                def_node,
                kind=kind,
                name=name,
                file=rel,
                signature=signature,
                doc=doc,
                summary=f"{signature} | {doc} | calls: {', '.join(called[:12])}",
                calls=called,
            )
            graph.add_edge(file_node, def_node, kind="defines")


def _resolve_calls(graph: nx.DiGraph) -> None:
    """Best-effort call edges: match called names to repo-defined functions."""
    by_name: Dict[str, List[str]] = {}
    for node, attrs in graph.nodes(data=True):
        if attrs.get("kind") in ("function", "class"):
            by_name.setdefault(attrs["name"], []).append(node)
    for node, attrs in list(graph.nodes(data=True)):
        for called in attrs.get("calls", []):
            for target in by_name.get(called, []):
                if target != node:
                    graph.add_edge(node, target, kind="calls")


_AML_SCHEMA_HINTS = ("azureml", "commandComponent", "pipelineJob", "environment.schema",
                     "managedOnlineEndpoint", "managedOnlineDeployment", "batchEndpoint")

# AML YAML `type:` values (used when $schema is absent)
_AML_TYPE_VALUES = frozenset({
    "command", "pipeline", "mltable", "mlmodel", "sweep", "automl",
    "managed_online_endpoint", "batch_endpoint",
    "managed_online_deployment", "batch_deployment",
})

# Folder prefixes that strongly imply role regardless of YAML content.
# MLpipelines/ holds all AML pipeline/endpoint/component YAMLs.
# azdopipelines/ holds all Azure DevOps CI/CT/CD YAML pipelines.
_AML_PATH_HINTS    = ("aml/", "aml\\", "mlpipelines/", "mlpipelines\\")
_AZDO_PATH_HINTS   = ("azdopipelines/", "azdopipelines\\", ".azuredevops/", ".azuredevops\\")


def _yaml_role(rel: str, doc: dict, raw: str) -> str:
    rel_lower = rel.replace("\\", "/").lower()
    schema = str(doc.get("$schema", "")) if isinstance(doc, dict) else ""

    # 1. Schema / content hints (highest confidence)
    if any(h in schema or h in raw[:2000] for h in _AML_SCHEMA_HINTS):
        return "aml_asset"

    # 2. YAML `type:` field (AML pipelines often omit $schema)
    if isinstance(doc, dict) and str(doc.get("type", "")).lower() in _AML_TYPE_VALUES:
        return "aml_asset"

    # 3. AzDO top-level structure keys
    if isinstance(doc, dict) and any(k in doc for k in ("trigger", "stages", "jobs", "steps", "pr", "pool")):
        return "azdo_pipeline"

    # 4. Path-based fallback — folder tells us the role even without perfect YAML
    if any(rel_lower.startswith(h) or f"/{h.rstrip('/')}" in rel_lower for h in _AML_PATH_HINTS):
        return "aml_asset"
    if any(rel_lower.startswith(h) or f"/{h.rstrip('/')}" in rel_lower for h in _AZDO_PATH_HINTS):
        return "azdo_pipeline"

    if "workflows" in rel and rel.startswith(".github"):
        return "github_workflow"
    return "config"


def _add_dockerfile(graph: nx.DiGraph, rel: str, raw: str, all_files: List[str]) -> None:
    """Tag a Dockerfile node with its role and key metadata."""
    file_node = f"file:{rel}"
    # Extract base image(s) for summary
    from_images = re.findall(r"^FROM\s+([^\s]+)", raw, re.MULTILINE)
    has_cmd = bool(re.search(r"^(CMD|ENTRYPOINT)\s", raw, re.MULTILINE))
    summary_parts = [f"dockerfile: base={', '.join(from_images[:3]) or '?'}"]
    if has_cmd:
        summary_parts.append("has CMD/ENTRYPOINT")
    graph.nodes[file_node].update({
        "role": "dockerfile",
        "summary": " | ".join(summary_parts),
    })
    # Cross-reference: Dockerfile may COPY repo files
    for other in all_files:
        if other == rel:
            continue
        name = Path(other).name
        if (other in raw or (name in raw and len(name) > 6)) and Path(other).suffix in (
            ".py", ".txt", ".cfg", ".toml", ".yml", ".yaml", ".sh",
        ):
            graph.add_edge(file_node, f"file:{other}", kind="references")


def _add_extensionless(graph: nx.DiGraph, rel: str, role: str) -> None:
    """Tag an extensionless file node (Makefile, Procfile, etc.) with its role."""
    file_node = f"file:{rel}"
    graph.nodes[file_node].update({
        "role": role,
        "summary": f"{role}: {rel}",
    })


def _add_yaml_file(graph: nx.DiGraph, rel: str, raw: str, all_files: List[str]) -> None:
    file_node = f"file:{rel}"
    try:
        doc = yaml.safe_load(raw)
    except yaml.YAMLError:
        doc = None
    role = _yaml_role(rel, doc if isinstance(doc, dict) else {}, raw)
    top_keys = list(doc.keys())[:12] if isinstance(doc, dict) else []
    graph.nodes[file_node].update({
        "role": role,
        "summary": f"{role}: keys={', '.join(map(str, top_keys))}",
    })

    # references: YAML mentions repo files (scripts, other yamls)
    for other in all_files:
        if other == rel:
            continue
        name = Path(other).name
        if (other in raw or (name in raw and len(name) > 6)) and Path(other).suffix in (".py", ".yml", ".yaml", ".sh"):
            graph.add_edge(file_node, f"file:{other}", kind="references")


def _mark_submits(graph: nx.DiGraph) -> None:
    """AzDO pipeline -> AML asset orchestration edges (via existing references)."""
    for src, dst, attrs in list(graph.edges(data=True)):
        if attrs.get("kind") != "references":
            continue
        src_role = graph.nodes[src].get("role", "")
        dst_role = graph.nodes[dst].get("role", "")
        if src_role == "azdo_pipeline" and dst_role == "aml_asset":
            graph.add_edge(src, dst, kind="submits")


def build_graph(repo_path: str) -> nx.DiGraph:
    graph = nx.DiGraph()
    base = Path(repo_path)
    files = [f for f in scan_repo(repo_path) if not any(d in f.split("/") for d in SKIP_DIRS)]

    # module index for import resolution: "pkg.mod" -> "pkg/mod.py"
    module_index: Dict[str, str] = {}
    for rel in files:
        if Path(rel).suffix in PY_EXT:
            module_index[rel[:-3].replace("/", ".")] = rel
            module_index[Path(rel).stem] = rel

    for rel in files:
        graph.add_node(f"file:{rel}", kind="file", path=rel, lang=_lang(rel),
                       summary=rel, role="", entry_point=False)

    for rel in files:
        path = base / rel
        if path.stat().st_size > MAX_FILE_BYTES:
            continue
        suffix = Path(rel).suffix.lower()
        name_lower = Path(rel).name.lower()
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if suffix in PY_EXT:
            _add_python_file(graph, rel, raw, module_index)
        elif suffix in YAML_EXT:
            _add_yaml_file(graph, rel, raw, files)
        elif name_lower in DOCKERFILE_NAMES or name_lower.startswith(DOCKERFILE_PREFIXES):
            _add_dockerfile(graph, rel, raw, files)
        elif name_lower in EXTENSIONLESS_ROLES:
            _add_extensionless(graph, rel, EXTENSIONLESS_ROLES[name_lower])

    _resolve_calls(graph)
    _mark_submits(graph)
    return graph
