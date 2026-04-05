from __future__ import annotations

import json
from pathlib import Path

from agent_rag.config import METADATA_DIR, ensure_data_dirs
from agent_rag.data.schemas import DocumentRecord


class MetadataStore:
    """Simple JSON-backed metadata store for the MVP."""

    def __init__(self) -> None:
        ensure_data_dirs()

    def _path_for(self, document_id: str) -> Path:
        return METADATA_DIR / f"{document_id}.json"

    def save_document(self, record: DocumentRecord) -> None:
        path = self._path_for(record.document_id)
        path.write_text(
            json.dumps(record.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get_document(self, document_id: str) -> DocumentRecord | None:
        path = self._path_for(document_id)
        if not path.exists():
            return None
        return DocumentRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def list_documents(self) -> list[DocumentRecord]:
        records: list[DocumentRecord] = []
        for path in sorted(METADATA_DIR.glob("*.json")):
            records.append(
                DocumentRecord.model_validate_json(path.read_text(encoding="utf-8"))
            )
        return records

    def upsert_documents(self, records: list[DocumentRecord]) -> None:
        for record in records:
            self.save_document(record)
