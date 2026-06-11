"""File-backed JSON memory store (default implementation)."""
import json
from pathlib import Path
from typing import List, Optional

from src.config.settings import get_settings
from src.memory.base import MemoryKind, MemoryRecord, MemoryStore


class JsonMemoryStore(MemoryStore):
    def __init__(self, path: Optional[str] = None) -> None:
        self._path = Path(path or get_settings().memory_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.write_text("[]", encoding="utf-8")

    def _load(self) -> List[MemoryRecord]:
        raw = json.loads(self._path.read_text(encoding="utf-8") or "[]")
        return [MemoryRecord.model_validate(item) for item in raw]

    def _save(self, records: List[MemoryRecord]) -> None:
        self._path.write_text(
            json.dumps([r.model_dump() for r in records], indent=2),
            encoding="utf-8",
        )

    def add(self, record: MemoryRecord) -> None:
        records = self._load()
        records.append(record)
        self._save(records)

    def search(self, kind: Optional[MemoryKind] = None, key_contains: str = "") -> List[MemoryRecord]:
        return [
            r for r in self._load()
            if (kind is None or r.kind == kind)
            and (not key_contains or key_contains.lower() in r.key.lower())
        ]

    def all(self) -> List[MemoryRecord]:
        return self._load()


_default_store: Optional[JsonMemoryStore] = None


def get_memory() -> JsonMemoryStore:
    global _default_store
    if _default_store is None:
        _default_store = JsonMemoryStore()
    return _default_store
