from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from agent_rag.config import (
    BYALDI_INDEX_ROOT,
    BYALDI_STAGING_ROOT,
    DEFAULT_INDEX_NAME,
    DEFAULT_RETRIEVAL_MODEL_NAME,
    ensure_data_dirs,
)
from agent_rag.data.schemas import DocumentRecord, IndexBuildResponse


class ByaldiIndexService:
    """
    Build and query ColQwen2/ColPali late-interaction indices through Byaldi.

    We index rendered page images instead of raw PDFs so later stages can reuse
    the exact same page assets for reranking and VLM generation.
    """

    def __init__(
        self,
        index_root: Path = BYALDI_INDEX_ROOT,
        staging_root: Path = BYALDI_STAGING_ROOT,
        default_model_name: str = DEFAULT_RETRIEVAL_MODEL_NAME,
    ) -> None:
        ensure_data_dirs()
        self.index_root = index_root
        self.staging_root = staging_root
        self.default_model_name = default_model_name
        self._loaded_rags: dict[str, Any] = {}

    def _resolve_device(self) -> str:
        try:
            import torch
        except ImportError:
            return "cpu"

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _import_byaldi(self) -> Any:
        try:
            from byaldi import RAGMultiModalModel
        except ImportError as exc:
            raise RuntimeError(
                "Byaldi is not installed. Run `pip install byaldi` or `pip install -e .` after adding the dependency."
            ) from exc
        return RAGMultiModalModel

    def _index_path(self, index_name: str) -> Path:
        return self.index_root / index_name

    def index_exists(self, index_name: str = DEFAULT_INDEX_NAME) -> bool:
        return self._index_path(index_name).exists()

    def _prepare_staging_dir(self, index_name: str) -> Path:
        staging_dir = self.staging_root / index_name
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        staging_dir.mkdir(parents=True, exist_ok=True)
        return staging_dir

    def _link_or_copy(self, source: Path, destination: Path) -> None:
        try:
            os.symlink(source, destination)
        except OSError:
            shutil.copy2(source, destination)

    def build_index(
        self,
        documents: list[DocumentRecord],
        index_name: str = DEFAULT_INDEX_NAME,
        model_name: str | None = None,
        overwrite: bool = True,
        store_collection_with_index: bool = False,
    ) -> IndexBuildResponse:
        if not documents:
            raise ValueError("No documents available to index.")

        RAGMultiModalModel = self._import_byaldi()
        resolved_model_name = model_name or self.default_model_name
        staging_dir = self._prepare_staging_dir(index_name)

        doc_ids: list[int] = []
        metadata: list[dict[str, Any]] = []
        indexed_pages = 0

        for page_idx, document in enumerate(documents):
            for page in document.pages:
                image_path = page.image_path
                if not image_path:
                    continue
                source = Path(str(image_path))
                if not source.exists():
                    continue

                index_doc_id = indexed_pages
                destination = staging_dir / f"{index_doc_id:06d}_{source.name}"
                self._link_or_copy(source, destination)

                doc_ids.append(index_doc_id)
                metadata.append(
                    {
                        "document_id": document.document_id,
                        "title": document.title,
                        "page_number": int(page.page_number),
                        "image_path": str(source),
                    }
                )
                indexed_pages += 1

        if indexed_pages == 0:
            raise ValueError("No rendered page images were found. Register documents first.")

        rag = RAGMultiModalModel.from_pretrained(
            resolved_model_name,
            index_root=str(self.index_root),
            device=self._resolve_device(),
        )
        rag.index(
            input_path=str(staging_dir),
            index_name=index_name,
            store_collection_with_index=store_collection_with_index,
            doc_ids=doc_ids,
            metadata=metadata,
            overwrite=overwrite,
        )
        self._loaded_rags[index_name] = rag

        manifest = {
            "index_name": index_name,
            "model_name": resolved_model_name,
            "indexed_documents": [doc.document_id for doc in documents],
            "indexed_pages": indexed_pages,
            "staging_path": str(staging_dir),
        }
        (self._index_path(index_name) / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return IndexBuildResponse(
            index_name=index_name,
            model_name=resolved_model_name,
            indexed_documents=[doc.document_id for doc in documents],
            indexed_pages=indexed_pages,
            staging_path=str(staging_dir),
            index_root=str(self._index_path(index_name)),
        )

    def _load_or_get_rag(self, index_name: str) -> Any:
        if index_name in self._loaded_rags:
            return self._loaded_rags[index_name]

        RAGMultiModalModel = self._import_byaldi()
        rag = RAGMultiModalModel.from_index(
            index_name,
            index_root=str(self.index_root),
            device=self._resolve_device(),
        )
        self._loaded_rags[index_name] = rag
        return rag

    def search(self, query: str, index_name: str, top_k: int) -> list[dict[str, Any]]:
        if not self.index_exists(index_name):
            raise FileNotFoundError(
                f"Index '{index_name}' does not exist under {self.index_root}."
            )

        rag = self._load_or_get_rag(index_name)
        results = rag.search(query, k=top_k)
        normalized: list[dict[str, Any]] = []

        for item in results:
            payload = item.dict() if hasattr(item, "dict") else dict(item)
            metadata = payload.get("metadata") or {}
            normalized.append(
                {
                    "document_id": metadata.get("document_id"),
                    "title": metadata.get("title"),
                    "page_number": int(metadata.get("page_number", payload.get("page_num", 1))),
                    "image_path": metadata.get("image_path"),
                    "score": float(payload.get("score", 0.0)),
                    "raw_doc_id": payload.get("doc_id"),
                }
            )

        return normalized
