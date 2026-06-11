"""Pluggable long-term memory abstraction.

Short-term memory is the LangGraph state itself; this layer persists knowledge
across runs: previous fixes, common failures, successful remediations.
"""
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

MemoryKind = Literal["fix", "failure", "remediation", "repair_history"]


class MemoryRecord(BaseModel):
    kind: MemoryKind
    key: str = Field(description="Short lookup key, e.g. error signature or asset type")
    payload: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class MemoryStore(ABC):
    @abstractmethod
    def add(self, record: MemoryRecord) -> None: ...

    @abstractmethod
    def search(self, kind: Optional[MemoryKind] = None, key_contains: str = "") -> List[MemoryRecord]: ...

    @abstractmethod
    def all(self) -> List[MemoryRecord]: ...

    # Convenience wrappers -------------------------------------------------
    def record_fix(self, error_signature: str, fix_summary: str, files: List[str]) -> None:
        self.add(MemoryRecord(kind="fix", key=error_signature,
                              payload={"summary": fix_summary, "files": files}))

    def record_failure(self, error_signature: str, detail: str) -> None:
        self.add(MemoryRecord(kind="failure", key=error_signature, payload={"detail": detail}))

    def record_remediation(self, repo_url: str, generated_files: List[str]) -> None:
        self.add(MemoryRecord(kind="remediation", key=repo_url,
                              payload={"generated_files": generated_files}))

    def similar_fixes(self, error_text: str, limit: int = 3) -> List[MemoryRecord]:
        """Naive relevance: token overlap between the error and stored keys."""
        tokens = {t.lower() for t in error_text.split() if len(t) > 3}
        scored = []
        for rec in self.search(kind="fix"):
            overlap = len(tokens & {t.lower() for t in rec.key.split()})
            if overlap:
                scored.append((overlap, rec))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [rec for _, rec in scored[:limit]]
