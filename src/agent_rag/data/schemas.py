from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DocumentRegistrationRequest(BaseModel):
    document_id: str = Field(..., description="Unique ID for the document.")
    title: str = Field(..., description="Paper title.")
    pdf_path: str = Field(..., description="Absolute or project-relative PDF path.")
    authors: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class DocumentSource(BaseModel):
    kind: str = Field(..., description="Source kind such as 'pdf' or 'dataset'.")
    dataset_name: str | None = None
    split: str | None = None
    row_index: int | None = None


class QAPair(BaseModel):
    question: str
    answer: str
    qa_type: str | None = None
    task_type: str | None = None
    source: str | None = None
    page_number: int | None = None
    groundedness_passed: bool | None = None
    standalone_passed: bool | None = None
    critique_metadata: dict[str, Any] = Field(default_factory=dict)


class PageRecord(BaseModel):
    page_number: int
    image_path: str | None = None
    width: int | None = None
    height: int | None = None
    texts: list[str] = Field(default_factory=list)
    object_image_paths: list[str] = Field(default_factory=list)
    object_ids: list[str] = Field(default_factory=list)
    bboxes: list[list[int]] = Field(default_factory=list)
    relations: list[list[str]] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    raw_metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentRecord(BaseModel):
    document_id: str
    title: str
    pdf_path: str | None = None
    authors: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    source: DocumentSource | None = None
    pages: list[PageRecord] = Field(default_factory=list)
    qa_pairs: list[QAPair] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DatasetImportRequest(BaseModel):
    dataset_name: str = Field(default="gigant/pdfvqa")
    split: str = Field(default="validation")
    start_index: int = Field(default=0, ge=0)
    limit: int = Field(default=10, ge=1, le=200)
    document_id_prefix: str = Field(default="pdfvqa")
    persist_object_images: bool = Field(
        default=True,
        description="Whether to persist object-level images from the dataset row.",
    )


class DatasetImportResponse(BaseModel):
    dataset_name: str
    split: str
    imported_documents: list[str]
    imported_count: int


class QASynthesisRequest(BaseModel):
    document_ids: list[str] | None = Field(
        default=None,
        description="Optional subset of stored documents to synthesize QA for.",
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        le=200,
        description="Optional cap on how many documents to process.",
    )
    model_name: str = Field(
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        description="Qwen2.5-VL checkpoint for QA synthesis.",
    )
    questions_per_page: int = Field(
        default=3,
        ge=1,
        le=10,
        description="How many candidate questions to generate per page before filtering.",
    )
    max_new_tokens: int = Field(default=128, ge=32, le=512)
    overwrite_synthetic: bool = Field(
        default=False,
        description="Whether to replace existing synthetic QA pairs.",
    )
    filter_groundedness: bool = Field(
        default=True,
        description="Whether to filter candidate questions by groundedness.",
    )
    filter_standalone: bool = Field(
        default=True,
        description="Whether to filter candidate questions by standalone-ness.",
    )
    prompt_template: str | None = Field(
        default=None,
        description="Optional custom QA generation prompt template.",
    )


class QASynthesisResult(BaseModel):
    document_id: str
    question: str
    answer: str
    page_number: int
    groundedness_passed: bool
    standalone_passed: bool
    groundedness_judgment: str | None = None
    standalone_judgment: str | None = None
    accepted: bool = True


class QASynthesisResponse(BaseModel):
    processed_documents: int
    synthesized_pairs: list[QASynthesisResult]
    skipped_documents: list[str] = Field(default_factory=list)


class EvaluationRequest(BaseModel):
    document_ids: list[str] | None = Field(
        default=None,
        description="Optional subset of stored documents to evaluate.",
    )
    limit: int = Field(default=20, ge=1, le=500)
    qa_sources: list[str] | None = Field(
        default=None,
        description="Optional subset of QA sources to evaluate, such as qwen2.5-vl.",
    )
    index_name: str | None = Field(default=None)
    top_k: int = Field(default=3, ge=1, le=10)
    use_reranker: bool = Field(default=False)
    reranker_model_name: str = Field(default="lightonai/MonoQwen2-VL-v0.1")
    rerank_candidate_multiplier: int = Field(default=3, ge=1, le=10)
    use_generation_model: bool = Field(default=True)
    generation_model_name: str = Field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    generation_max_new_tokens: int = Field(default=256, ge=32, le=1024)
    generation_max_images: int = Field(default=1, ge=1, le=8)
    judge_model_name: str = Field(default="gpt-5.4-mini")
    question_weight: float = Field(default=0.5, ge=0.0)
    page_weight: float = Field(default=0.5, ge=0.0)


class EvaluationExampleResult(BaseModel):
    document_id: str
    title: str
    page_number: int
    question: str
    reference_answer: str
    predicted_answer: str
    evidence_page_numbers: list[int] = Field(default_factory=list)
    correct: bool | None = None
    accurate: bool | None = None
    factual: bool | None = None
    raw_score: int
    normalized_score: float
    rationale: str | None = None
    judge_model_name: str


class EvaluationPageSummary(BaseModel):
    document_id: str
    title: str
    page_number: int
    question_count: int
    mean_raw_score: float
    mean_normalized_score: float


class EvaluationAggregate(BaseModel):
    example_count: int
    page_count: int
    mean_raw_score: float
    mean_normalized_score: float
    page_balanced_normalized_score: float
    weighted_normalized_score: float
    question_weight: float
    page_weight: float
    page_summaries: list[EvaluationPageSummary] = Field(default_factory=list)


class EvaluationResponse(BaseModel):
    run_id: str
    report_path: str
    processed_examples: int
    aggregate: EvaluationAggregate
    examples: list[EvaluationExampleResult] = Field(default_factory=list)


class QueryRequest(BaseModel):
    question: str = Field(..., description="User question about the paper corpus.")
    top_k: int = Field(default=3, ge=1, le=10)
    index_name: str | None = Field(
        default=None,
        description="Optional retrieval index name. Falls back to metadata retrieval when unavailable.",
    )
    use_reranker: bool = Field(
        default=False,
        description="Whether to rerank retrieved page images with MonoQwen2-VL.",
    )
    reranker_model_name: str = Field(
        default="lightonai/MonoQwen2-VL-v0.1",
        description="Multimodal reranker checkpoint.",
    )
    rerank_candidate_multiplier: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Retrieve more candidates before reranking, then keep only top_k.",
    )
    use_generation_model: bool = Field(
        default=True,
        description="Whether to answer with Qwen2.5-VL using the retrieved page images.",
    )
    generation_model_name: str = Field(
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        description="Vision-language answer generation model checkpoint.",
    )
    generation_max_new_tokens: int = Field(
        default=256,
        ge=32,
        le=1024,
        description="Maximum number of generated tokens for the final answer.",
    )
    generation_max_images: int = Field(
        default=1,
        ge=1,
        le=8,
        description="Maximum number of retrieved page images to pass into the generator.",
    )
    document_ids: list[str] | None = Field(
        default=None,
        description="Optional subset of registered documents to search.",
    )


class EvidencePage(BaseModel):
    document_id: str
    title: str
    page_number: int
    image_path: str | None = None
    score: float | None = None
    retrieval_score: float | None = None
    rerank_score: float | None = None
    rationale: str | None = None


class QueryResponse(BaseModel):
    answer: str
    evidences: list[EvidencePage]
    debug: dict[str, Any] = Field(default_factory=dict)


class IndexBuildRequest(BaseModel):
    index_name: str = Field(
        default="paper_index",
        description="Name of the retrieval index to create or overwrite.",
    )
    model_name: str = Field(
        default="vidore/colqwen2-v1.0",
        description="Retrieval model checkpoint. For Byaldi, use the colpali-engine compatible checkpoint.",
    )
    document_ids: list[str] | None = Field(
        default=None,
        description="Optional subset of registered documents to index.",
    )
    overwrite: bool = Field(default=True)
    store_collection_with_index: bool = Field(default=False)


class IndexBuildResponse(BaseModel):
    index_name: str
    model_name: str
    indexed_documents: list[str]
    indexed_pages: int
    staging_path: str
    index_root: str
