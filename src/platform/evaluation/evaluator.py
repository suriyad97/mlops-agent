"""Capability evaluator: evidence collection -> LLM judgment -> deterministic scoring.

The LLM judges WHETHER collected evidence truly realizes each capability
component (semantics). The arithmetic — weights, scores, statuses, gap
ordering — is pure code: same judgments always produce the same report.
One structured LLM call evaluates all capabilities.

Self-correction: after LLM judgment, deliverable files that exist on disk
override an 'absent' or 'weak' verdict with 'found' (filesystem is ground truth).
"""
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Literal

import networkx as nx
from pydantic import BaseModel, Field

from src.llm.factory import get_provider
from src.llm.prompts.base import MLOPS_EXPERT_PREAMBLE
from src.platform.catalog.loader import CapabilitySpec, dependency_order, load_catalog
from src.platform.understanding.retrieval import assets_by_role, find_evidence

# Capabilities to include for each endpoint strategy (None = include all)
_STRATEGY_CAPS: Dict[str, set | None] = {
    "realtime": {"environment_lifecycle", "training", "model_lifecycle",
                 "realtime_deployment", "monitoring", "retraining", "rollback"},
    "batch":    {"environment_lifecycle", "training", "model_lifecycle",
                 "batch_deployment", "monitoring", "retraining", "rollback"},
    "both":     None,
    "none":     {"environment_lifecycle", "training", "model_lifecycle"},
}


class EvidenceJudgment(BaseModel):
    evidence_id: str                      # "<capability>.<evidence id>"
    status: Literal["found", "weak", "absent"]
    citations: List[str] = Field(default_factory=list)
    note: str = ""


class EvaluationJudgments(BaseModel):
    judgments: List[EvidenceJudgment]


class CapabilityStatus(BaseModel):
    capability: str
    score: int
    status: Literal["complete", "partial", "missing"]
    evidence: List[EvidenceJudgment] = Field(default_factory=list)
    missing_components: List[str] = Field(default_factory=list)


class CapabilityReport(BaseModel):
    capabilities: List[CapabilityStatus]
    summary: str = ""


class GapItem(BaseModel):
    capability: str
    component: str
    deliverables: List[str] = Field(default_factory=list)
    reason: str = ""


class GapReport(BaseModel):
    items: List[GapItem]
    order: List[str] = Field(default_factory=list)   # capability dependency order


class PlanItem(BaseModel):
    category: str
    file: str
    capability: str
    component: str
    status: Literal["present", "missing"]
    note: str = ""


class GenerationPlan(BaseModel):
    endpoint_strategy: str
    items: List[PlanItem]

    def markdown(self) -> str:
        present = [i for i in self.items if i.status == "present"]
        missing = [i for i in self.items if i.status == "missing"]
        lines = [f"## Generation Plan\n\n**Target:** `{self.endpoint_strategy}` endpoint\n"]
        for label, section, icon in [("Present", present, "✅"), ("Missing — will generate", missing, "🔲")]:
            if not section:
                continue
            lines.append(f"### {icon} {label}\n")
            by_cat: Dict[str, List[PlanItem]] = defaultdict(list)
            for item in section:
                by_cat[item.category].append(item)
            for cat in sorted(by_cat):
                lines.append(f"**{cat}**")
                for item in by_cat[cat]:
                    note = f" _{item.note}_" if item.note else ""
                    if item.status == "present":
                        lines.append(f"- `{item.file}` ({item.capability}){note}")
                    else:
                        lines.append(f"- `{item.file}` → _{item.capability}.{item.component}_")
                lines.append("")
        lines.append(f"**{len(present)} files present · {len(missing)} files to generate.**")
        return "\n".join(lines)


def _fs_verify(repo_path: str, spec: CapabilitySpec,
               judgment: EvidenceJudgment) -> EvidenceJudgment:
    """Override absent/weak → found when ALL deliverable files exist on disk.

    The LLM judges from the graph (an approximation). The filesystem is ground
    truth: if the files are there, the component is there.
    """
    if judgment.status == "found" or not repo_path:
        return judgment
    ev_id = judgment.evidence_id.split(".", 1)[-1]
    deliverables = spec.deliverables.get(ev_id, [])
    if not deliverables:
        return judgment
    base = Path(repo_path)
    existing = [d for d in deliverables if (base / d).exists()]
    if not existing:
        return judgment
    if len(existing) == len(deliverables):
        return judgment.model_copy(update={
            "status": "found",
            "citations": existing,
            "note": (f"[fs-verified] {judgment.note}").strip(),
        })
    # partial: some but not all — promote absent→weak, keep weak as-is
    if judgment.status == "absent":
        return judgment.model_copy(update={
            "status": "weak",
            "citations": existing,
            "note": (f"[fs-partial {len(existing)}/{len(deliverables)}] {judgment.note}").strip(),
        })
    return judgment


def _deliverable_category(path: str) -> str:
    p = path.replace("\\", "/").lower()
    if "azdopipelines" in p or ".azuredevops" in p:
        return "AzDO Pipeline"
    if p.startswith("aml/") or "/aml/" in p:
        return "AML Pipeline / Asset"
    if p.endswith(".py"):
        return "Python Script"
    return "Config / Other"


JUDGE_SYSTEM = MLOPS_EXPERT_PREAMBLE + """
You judge whether collected repository evidence truly realizes each capability
component. For EVERY evidence id listed, return exactly one judgment:
- found:  the evidence clearly implements the component's description
- weak:   something related exists but is incomplete for the description
- absent: nothing implements it
Cite file paths from the provided evidence. Judge SEMANTICS against the
description — do not require specific file names. An AzDO component is only
'found' if a pipeline actually orchestrates/submits the related assets.
Remember: concept drift monitoring requires ground-truth labels.
"""


def _inventory(graph: nx.DiGraph) -> str:
    """Shared asset inventory, serialized once per judgment call."""
    aml = assets_by_role(graph, "aml_asset")
    azdo = assets_by_role(graph, "azdo_pipeline")
    submits = [
        (graph.nodes[u].get("path"), graph.nodes[v].get("path"))
        for u, v, a in graph.edges(data=True) if a.get("kind") == "submits"
    ]
    def _line(path: str) -> str:
        summary = str(graph.nodes.get(f"file:{path}", {}).get("summary", ""))[:90]
        return f"  {path}  [{summary}]"

    lines = ["REPOSITORY INVENTORY (shared context for all judgments):"]
    lines.append("AML assets:")
    lines.extend(_line(p) for p in aml[:18]) if aml else lines.append("  none")
    lines.append("AzDO pipelines:")
    lines.extend(_line(p) for p in azdo[:18]) if azdo else lines.append("  none")
    lines.append("Orchestration (azdo submits aml): " +
                 ("; ".join(f"{s} -> {t}" for s, t in submits[:12]) or "NONE DETECTED"))
    lines.append(
        "NOTE: pipeline file names and YAML keys are meaningful evidence — e.g. a pipeline "
        "named like 'ci-setup-environment' that references environment assets DOES realize "
        "an environment pipeline; 'qa-champion-challenger' implements model comparison/promotion."
    )
    return "\n".join(lines)


def _capability_section(graph: nx.DiGraph, spec: CapabilitySpec) -> str:
    """Evidence section for one capability (asset lists live in the inventory)."""
    lines = [f"## {spec.capability}: {spec.purpose}"]
    for ev in spec.evidence:
        lines.append(f"[{spec.capability}.{ev.id}] ({ev.kind}, weight {ev.weight}) {ev.description}")
        if ev.kind == "code_semantic":
            hits = find_evidence(graph, ev.indicators)[:5]
            for hit in hits:
                lines.append(f"    hit: {hit['file']} :: {hit['summary'][:100]}")
            if not hits:
                lines.append("    (no code hits)")
        else:
            lines.append("    judge against the repository inventory above")
    return "\n".join(lines)


def _score(spec: CapabilitySpec, judgments: Dict[str, EvidenceJudgment],
           repo_path: str = "") -> CapabilityStatus:
    score = 0
    evidence: List[EvidenceJudgment] = []
    missing: List[str] = []
    for ev in spec.evidence:
        judgment = judgments.get(
            f"{spec.capability}.{ev.id}",
            EvidenceJudgment(evidence_id=f"{spec.capability}.{ev.id}", status="absent",
                             note="no judgment returned"),
        )
        # Filesystem ground truth: files on disk override LLM verdict
        judgment = _fs_verify(repo_path, spec, judgment)
        evidence.append(judgment)
        if judgment.status == "found":
            score += ev.weight
        elif judgment.status == "weak":
            score += ev.weight // 2
            missing.append(ev.id)
        else:
            missing.append(ev.id)
    if not missing:
        status = "complete"
    elif score == 0:
        status = "missing"
    else:
        status = "partial"
    return CapabilityStatus(capability=spec.capability, score=score, status=status,
                            evidence=evidence, missing_components=missing)


CAPABILITIES_PER_CALL = 3


def _judge(provider, inventory: str, sections: List[str]) -> Dict[str, EvidenceJudgment]:
    result = provider.complete_structured(
        EvaluationJudgments,
        system=JUDGE_SYSTEM,
        user=f"{inventory}\n\nEvidence to judge (one judgment per [id]):\n\n" + "\n\n".join(sections),
    )
    return {j.evidence_id: j for j in result.judgments}


def evaluate(graph: nx.DiGraph, repo_path: str = "") -> CapabilityReport:
    catalog = load_catalog()
    order = dependency_order(catalog)
    provider = get_provider()
    inventory = _inventory(graph)

    # chunked judgment calls: bounded input AND bounded output per call
    judgments: Dict[str, EvidenceJudgment] = {}
    for i in range(0, len(order), CAPABILITIES_PER_CALL):
        chunk = order[i:i + CAPABILITIES_PER_CALL]
        sections = [_capability_section(graph, catalog[name]) for name in chunk]
        judgments.update(_judge(provider, inventory, sections))

    # repair pass: re-ask once for any ids the model failed to return
    expected = {f"{name}.{e.id}" for name in order for e in catalog[name].evidence}
    unjudged = sorted(expected - set(judgments))
    if unjudged:
        missing_caps = sorted({eid.split(".")[0] for eid in unjudged})
        sections = [_capability_section(graph, catalog[name]) for name in missing_caps]
        judgments.update(_judge(provider, inventory, sections))

    # _score applies filesystem verification (repo_path) on top of LLM verdicts
    statuses = [_score(catalog[name], judgments, repo_path) for name in order]
    complete = sum(1 for s in statuses if s.status == "complete")
    summary = (
        f"{complete}/{len(statuses)} capabilities complete; "
        + "; ".join(f"{s.capability}={s.status}({s.score})" for s in statuses)
    )
    return CapabilityReport(capabilities=statuses, summary=summary)


def standard_generation_set(endpoint_strategy: str = "both") -> GapReport:
    """The FULL standard deliverable set for an endpoint strategy, in dependency order.

    This is the golden-path driver: generation always emits the complete standard MLOps
    block for the strategy, independent of what the repository already contains. The
    per-stage decision of wired/adapter/scaffold (for user ML code) is layered on top
    via the contract manifest — it does NOT remove items from this set. Pipeline/AML/
    infra deliverables are always (re)generated to the one standard template.
    """
    catalog = load_catalog()
    order = dependency_order(catalog)
    allowed = _STRATEGY_CAPS.get(endpoint_strategy)   # None = all capabilities
    items: List[GapItem] = []
    for name in order:
        if allowed is not None and name not in allowed:
            continue
        spec = catalog[name]
        for component, deliverables in spec.deliverables.items():
            items.append(GapItem(
                capability=name, component=component,
                deliverables=deliverables, reason="standard set",
            ))
    return GapReport(items=items, order=order)


def gap_analysis(report: CapabilityReport, endpoint_strategy: str = "both") -> GapReport:
    catalog = load_catalog()
    order = dependency_order(catalog)
    allowed = _STRATEGY_CAPS.get(endpoint_strategy)  # None = all caps allowed
    items: List[GapItem] = []
    judgment_by_id = {
        j.evidence_id: j for s in report.capabilities for j in s.evidence
    }
    for name in order:
        if allowed is not None and name not in allowed:
            continue
        status = next((s for s in report.capabilities if s.capability == name), None)
        if not status or status.status == "complete":
            continue
        spec = catalog[name]
        for component in status.missing_components:
            judgment = judgment_by_id.get(f"{name}.{component}")
            items.append(GapItem(
                capability=name,
                component=component,
                deliverables=spec.deliverables.get(component, []),
                reason=(judgment.note if judgment else "") or
                       (f"{judgment.status} evidence" if judgment else "absent"),
            ))
    return GapReport(items=items, order=order)


def generation_plan(
    report: CapabilityReport,
    repo_path: str = "",
    endpoint_strategy: str = "both",
) -> GenerationPlan:
    """Build the present-vs-missing plan, filtered to the endpoint strategy.

    Files that exist on disk are marked present regardless of LLM verdict —
    this is the same filesystem-ground-truth logic as _fs_verify.
    """
    catalog = load_catalog()
    order = dependency_order(catalog)
    allowed = _STRATEGY_CAPS.get(endpoint_strategy)
    judgment_by_id = {j.evidence_id: j for s in report.capabilities for j in s.evidence}

    items: List[PlanItem] = []
    seen: set = set()   # deduplicate deliverable paths across components
    for name in order:
        if allowed is not None and name not in allowed:
            continue
        spec = catalog[name]
        for ev in spec.evidence:
            eid = f"{name}.{ev.id}"
            j = judgment_by_id.get(eid)
            llm_status = j.status if j else "absent"
            for fpath in spec.deliverables.get(ev.id, []):
                if fpath in seen:
                    continue
                seen.add(fpath)
                on_disk = bool(repo_path) and (Path(repo_path) / fpath).exists()
                if llm_status == "found" or on_disk:
                    item_status: Literal["present", "missing"] = "present"
                    note = "[fs-verified]" if (on_disk and llm_status != "found") else ""
                else:
                    item_status = "missing"
                    note = ""
                items.append(PlanItem(
                    category=_deliverable_category(fpath),
                    file=fpath,
                    capability=name,
                    component=ev.id,
                    status=item_status,
                    note=note,
                ))
    return GenerationPlan(endpoint_strategy=endpoint_strategy, items=items)
