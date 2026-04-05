from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from agent_rag.config import FRONTEND_DIR, UPLOADS_DIR
from agent_rag.data.schemas import (
    DatasetImportRequest,
    DatasetImportResponse,
    DocumentRecord,
    DocumentRegistrationRequest,
    EvaluationRequest,
    EvaluationResponse,
    IndexBuildRequest,
    IndexBuildResponse,
    QASynthesisRequest,
    QASynthesisResponse,
    QueryRequest,
    QueryResponse,
)
from agent_rag.pipelines.qa import QAPipeline
from agent_rag.services.evaluate import MRAGEvaluator, OpenAIAnswerJudge
from agent_rag.services.generate import AnswerGenerator
from agent_rag.services.ingest import DocumentIngestService
from agent_rag.services.rerank import MonoVLMReranker
from agent_rag.services.retrieve import Retriever
from agent_rag.services.storage import MetadataStore


router = APIRouter()
store = MetadataStore()
ingest_service = DocumentIngestService()
qa_pipeline = QAPipeline(
    store=store,
    retriever=Retriever(),
    reranker=MonoVLMReranker(),
    generator=AnswerGenerator(),
)
evaluator = MRAGEvaluator(
    store=store,
    qa_pipeline=qa_pipeline,
    judge=OpenAIAnswerJudge(),
)


@router.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/documents/register", response_model=DocumentRecord)
def register_document(request: DocumentRegistrationRequest) -> DocumentRecord:
    try:
        record = ingest_service.register_document(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    store.save_document(record)
    return record


@router.post("/documents/upload", response_model=DocumentRecord)
async def upload_document(
    file: UploadFile = File(...),
    document_id: str = Form(...),
    title: str = Form(...),
    authors: str = Form(default=""),
    tags: str = Form(default=""),
) -> DocumentRecord:
    suffix = Path(file.filename or "upload.pdf").suffix or ".pdf"
    upload_path = UPLOADS_DIR / f"{document_id}{suffix}"

    with upload_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    request = DocumentRegistrationRequest(
        document_id=document_id,
        title=title,
        pdf_path=str(upload_path),
        authors=[item.strip() for item in authors.split(",") if item.strip()],
        tags=[item.strip() for item in tags.split(",") if item.strip()],
    )

    try:
        record = ingest_service.register_document(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    store.save_document(record)
    return record


@router.get("/documents", response_model=list[DocumentRecord])
def list_documents() -> list[DocumentRecord]:
    return store.list_documents()


@router.post("/datasets/pdfvqa/import", response_model=DatasetImportResponse)
def import_pdfvqa(request: DatasetImportRequest) -> DatasetImportResponse:
    try:
        records = ingest_service.import_pdfvqa(request)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    for record in records:
        store.save_document(record)

    return DatasetImportResponse(
        dataset_name=request.dataset_name,
        split=request.split,
        imported_documents=[record.document_id for record in records],
        imported_count=len(records),
    )


@router.post("/datasets/pdfvqa/synthesize-qa", response_model=QASynthesisResponse)
def synthesize_pdfvqa_qa(request: QASynthesisRequest) -> QASynthesisResponse:
    documents = store.list_documents()
    if request.document_ids:
        allowed = set(request.document_ids)
        documents = [doc for doc in documents if doc.document_id in allowed]
    else:
        documents = [
            doc for doc in documents
            if doc.source and doc.source.kind == "dataset"
        ]

    if request.limit is not None:
        documents = documents[: request.limit]

    if not documents:
        raise HTTPException(
            status_code=404,
            detail="No matching dataset-backed documents found for QA synthesis.",
        )

    try:
        updated_documents, synthesized_pairs, synthesis_debug = qa_pipeline.generator.synthesize_factoid_qa(
            documents=documents,
            model_name=request.model_name,
            questions_per_page=request.questions_per_page,
            max_new_tokens=request.max_new_tokens,
            filter_groundedness=request.filter_groundedness,
            filter_standalone=request.filter_standalone,
            prompt_template=request.prompt_template,
            overwrite_synthetic=request.overwrite_synthetic,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    store.upsert_documents(updated_documents)

    return QASynthesisResponse(
        processed_documents=len(updated_documents),
        synthesized_pairs=synthesized_pairs,
        skipped_documents=list(synthesis_debug.get("skipped_documents", [])),
    )


@router.post("/indices/build", response_model=IndexBuildResponse)
def build_index(request: IndexBuildRequest) -> IndexBuildResponse:
    documents = store.list_documents()
    if request.document_ids:
        allowed = set(request.document_ids)
        documents = [doc for doc in documents if doc.document_id in allowed]

    if not documents:
        raise HTTPException(status_code=404, detail="No matching documents found to index.")

    try:
        return qa_pipeline.retriever.build_index(
            documents=documents,
            index_name=request.index_name,
            model_name=request.model_name,
            overwrite=request.overwrite,
            store_collection_with_index=request.store_collection_with_index,
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    return qa_pipeline.run(request)


@router.post("/evaluations/pdfvqa/run", response_model=EvaluationResponse)
def run_pdfvqa_evaluation(request: EvaluationRequest) -> EvaluationResponse:
    try:
        return evaluator.run(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Evaluation failed: {exc}") from exc
