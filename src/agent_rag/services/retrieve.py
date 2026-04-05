from __future__ import annotations

from agent_rag.config import DEFAULT_INDEX_NAME
from agent_rag.data.schemas import DocumentRecord, EvidencePage
from agent_rag.services.index import ByaldiIndexService


class MetadataRetriever:
    """Fallback retriever used before or without a trained multimodal index."""

    def retrieve(
        self,
        question: str,
        documents: list[DocumentRecord],
        top_k: int,
    ) -> tuple[list[EvidencePage], dict[str, object]]:
        evidences: list[EvidencePage] = []
        lowered_question = question.lower()

        for document in documents:
            score = 0.2
            if any(token in document.title.lower() for token in lowered_question.split()):
                score = 0.6

            first_page = document.pages[0] if document.pages else None
            evidences.append(
                EvidencePage(
                    document_id=document.document_id,
                    title=document.title,
                    page_number=int(first_page.page_number if first_page else 1),
                    image_path=first_page.image_path if first_page else None,
                    score=score,
                    retrieval_score=score,
                    rationale="Metadata-based placeholder retrieval.",
                )
            )

        evidences.sort(key=lambda item: item.score or 0.0, reverse=True)
        return evidences[:top_k], {
            "retriever": "metadata_placeholder",
            "index_name": None,
        }


class Retriever:
    """
    Hybrid retriever.

    Preferred path:
    - Byaldi + ColQwen2 index over rendered page images

    Fallback path:
    - simple metadata retrieval so the API remains usable before the index exists
    """

    def __init__(self) -> None:
        self.metadata_retriever = MetadataRetriever()
        self.byaldi = ByaldiIndexService()

    def build_index(
        self,
        documents: list[DocumentRecord],
        index_name: str,
        model_name: str,
        overwrite: bool,
        store_collection_with_index: bool,
    ):
        return self.byaldi.build_index(
            documents=documents,
            index_name=index_name,
            model_name=model_name,
            overwrite=overwrite,
            store_collection_with_index=store_collection_with_index,
        )

    def retrieve(
        self,
        question: str,
        documents: list[DocumentRecord],
        top_k: int,
        index_name: str | None = None,
    ) -> tuple[list[EvidencePage], dict[str, object]]:
        target_index = index_name or DEFAULT_INDEX_NAME

        if self.byaldi.index_exists(target_index):
            allowed_document_ids = {doc.document_id for doc in documents}
            byaldi_results = self.byaldi.search(
                query=question,
                index_name=target_index,
                top_k=max(top_k * 3, top_k),
            )

            evidences: list[EvidencePage] = []
            for item in byaldi_results:
                if allowed_document_ids and item["document_id"] not in allowed_document_ids:
                    continue
                evidences.append(
                    EvidencePage(
                        document_id=str(item["document_id"]),
                        title=str(item["title"]),
                        page_number=int(item["page_number"]),
                        image_path=item["image_path"],
                        score=float(item["score"]),
                        retrieval_score=float(item["score"]),
                        rationale="Byaldi late-interaction retrieval over rendered page images.",
                    )
                )
                if len(evidences) >= top_k:
                    break

            if evidences:
                return evidences, {
                    "retriever": "byaldi_colqwen2",
                    "index_name": target_index,
                }

        return self.metadata_retriever.retrieve(
            question=question,
            documents=documents,
            top_k=top_k,
        )
