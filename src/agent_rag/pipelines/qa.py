from __future__ import annotations

from agent_rag.data.schemas import QueryRequest, QueryResponse
from agent_rag.services.generate import AnswerGenerator
from agent_rag.services.rerank import MonoVLMReranker
from agent_rag.services.retrieve import Retriever
from agent_rag.services.storage import MetadataStore


class QAPipeline:
    def __init__(
        self,
        store: MetadataStore,
        retriever: Retriever,
        reranker: MonoVLMReranker,
        generator: AnswerGenerator,
    ) -> None:
        self.store = store
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator

    def run(self, request: QueryRequest) -> QueryResponse:
        documents = self.store.list_documents()
        if request.document_ids:
            allowed = set(request.document_ids)
            documents = [doc for doc in documents if doc.document_id in allowed]

        retrieval_k = request.top_k
        if request.use_reranker:
            retrieval_k = request.top_k * request.rerank_candidate_multiplier

        evidences, retrieval_debug = self.retriever.retrieve(
            question=request.question,
            documents=documents,
            top_k=retrieval_k,
            index_name=request.index_name,
        )
        rerank_debug: dict[str, object] = {
            "reranker": None,
            "reranker_status": "disabled",
        }
        if request.use_reranker:
            try:
                evidences, rerank_debug = self.reranker.rerank(
                    question=request.question,
                    evidences=evidences,
                    top_k=request.top_k,
                    model_name=request.reranker_model_name,
                )
            except Exception as exc:
                evidences = evidences[: request.top_k]
                rerank_debug = {
                    "reranker": "monoqwen2_vl",
                    "reranker_model_name": request.reranker_model_name,
                    "reranker_status": "fallback_to_retrieval",
                    "reranker_error": str(exc),
                }
        else:
            evidences = evidences[: request.top_k]

        generation_debug: dict[str, object] = {
            "generator": "placeholder_answer_generator",
            "generator_status": "disabled",
        }
        if request.use_generation_model:
            try:
                answer, generation_debug = self.generator.generate(
                    question=request.question,
                    evidences=evidences,
                    model_name=request.generation_model_name,
                    max_new_tokens=request.generation_max_new_tokens,
                    max_images=request.generation_max_images,
                )
            except Exception as exc:
                answer, generation_debug = self.generator._placeholder_answer(
                    request.question,
                    evidences,
                ), {
                    "generator": "qwen2_5_vl",
                    "generator_model_name": request.generation_model_name,
                    "generator_status": "fallback_to_placeholder",
                    "generator_error": str(exc),
                }
        else:
            answer = self.generator._placeholder_answer(
                request.question,
                evidences,
            )
        return QueryResponse(
            answer=answer,
            evidences=evidences,
            debug={
                "documents_considered": [doc.document_id for doc in documents],
                **retrieval_debug,
                **rerank_debug,
                **generation_debug,
            },
        )
