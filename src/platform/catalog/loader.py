"""Capability catalog loader — the enterprise ontology, validated at load time.

repository_discovery is implicit: it is the platform's own scan capability and
is marked complete when a knowledge graph + blueprint exist for the project.
"""
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

CATALOG_DIR = Path(__file__).parent


class EvidenceSpec(BaseModel):
    id: str
    kind: Literal["code_semantic", "aml_asset", "azdo_pipeline"]
    description: str
    indicators: List[str] = Field(default_factory=list)
    role: str = ""
    weight: int


class CapabilitySpec(BaseModel):
    capability: str
    purpose: str
    depends_on: List[str] = Field(default_factory=list)
    enables: List[str] = Field(default_factory=list)
    evidence: List[EvidenceSpec]
    deliverables: Dict[str, List[str]] = Field(default_factory=dict)
    validation: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def weights_sum_to_100(self) -> "CapabilitySpec":
        total = sum(e.weight for e in self.evidence)
        if total != 100:
            raise ValueError(f"{self.capability}: evidence weights sum to {total}, expected 100")
        return self


@lru_cache
def load_catalog() -> Dict[str, CapabilitySpec]:
    catalog: Dict[str, CapabilitySpec] = {}
    for path in sorted(CATALOG_DIR.glob("*.yaml")):
        spec = CapabilitySpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        catalog[spec.capability] = spec
    # referential integrity
    for spec in catalog.values():
        for dep in spec.depends_on:
            if dep not in catalog and dep != "repository_discovery":
                raise ValueError(f"{spec.capability} depends on unknown capability {dep}")
    return catalog


def dependency_order(catalog: Dict[str, CapabilitySpec]) -> List[str]:
    """Topological order: dependency-free capabilities first."""
    ordered: List[str] = []
    remaining = dict(catalog)
    while remaining:
        ready = [
            name for name, spec in remaining.items()
            if all(d in ordered or d not in catalog for d in spec.depends_on)
        ]
        if not ready:
            raise ValueError(f"dependency cycle among: {list(remaining)}")
        for name in sorted(ready):
            ordered.append(name)
            remaining.pop(name)
    return ordered
