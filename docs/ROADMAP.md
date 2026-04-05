# Roadmap

## Phase 0: Project Scaffold

Goals:

- Establish clear module boundaries
- Provide a basic API
- Provide local metadata storage

Deliverables:

- Document registration endpoint
- Query endpoint
- Placeholder retrieval and generation modules

## Phase 1: PDF Page Rendering

Goals:

- Render PDFs into page images
- Preserve page numbers and file paths
- Prepare for page-level retrieval

Current progress:

- `PyMuPDF` is already integrated for page-by-page PNG rendering
- Metadata already records `page_number`, `image_path`, `width`, and `height`
- A dataset-backed ingest path is available for `gigant/pdfvqa`
- `questions` and `answers` from dataset rows are stored as QA pairs for evaluation
- Qwen2.5-VL can now synthesize multiple factoid QA candidates from each imported `page` image
- Candidate QA pairs can be filtered by groundedness and standalone-ness before persistence

Suggested next steps:

- Add `caption hints` and `section hints`
- Add scaling and compression strategies for very large pages
- Support batch import of entire PDF folders
- Add batch QA synthesis scripts and dataset export utilities for evaluation runs
- Add offline synthesis jobs with resumable progress tracking

Acceptance criteria:

- Uploading a PDF produces a full rendered page-image directory

## Phase 2: Multimodal Retrieval

Goals:

- Use a visual document retrieval model for recall
- Avoid OCR text chunking as the primary path

Current progress:

- Byaldi is integrated as the retrieval framework
- Rendered page images can be indexed directly
- The default model is `vidore/colqwen2-v1.0`
- It remains configurable back to `vidore/colqwen2-v0.1`

Suggested next steps:

- Add an index status endpoint
- Support incremental indexing instead of always rebuilding from scratch
- Add richer retrieval debug output to query results

Acceptance criteria:

- For a single paper, questions like “Which page contains the method figure?”
  or “Which page contains the experiment table?” return sensible pages

## Phase 3: Multimodal Reranking

Goals:

- Improve ranking quality for figure-heavy, table-heavy, and detail-heavy pages

Current progress:

- A reranking service scaffold is integrated with `lightonai/MonoQwen2-VL-v0.1`
- Query-time reranking can be enabled on demand
- The system can retrieve more candidates, rerank them, and keep only top-k

Suggested next steps:

- Add batch scoring to reduce per-page reranking latency
- Add lightweight priors for figure and table pages
- Record both retrieval scores and rerank scores for evaluation

Acceptance criteria:

- `Recall@5` and `MRR` improve measurably over retrieval-only mode

## Phase 4: Answer Generation

Goals:

- Answer questions directly from retrieved page images
- Return page-level evidence

Current progress:

- A generation service scaffold is integrated with `Qwen/Qwen2.5-VL-3B-Instruct`
- Top-k page images can be passed directly into the VLM
- If generation fails, the system falls back to a placeholder answer

Suggested next steps:

- Improve prompting so outputs more reliably include `[pX]` citations
- Add context compression for multi-image answering
- Compare `3B` and `7B` quality on paper QA tasks

Acceptance criteria:

- The system can answer summary, method, experiment, and chart-interpretation questions

## Phase 5: Evaluation

Goals:

- Diagnose whether bottlenecks come from retrieval, reranking, or generation

Suggested implementation:

- Retrieval metrics: `Recall@k`, `MRR`, `nDCG`
- Generation metrics: `EM`, `F1`, `LLM Judge`
- Datasets: `PDFVQA`, `ChartQA`, and custom paper QA data
- OpenAI-judge scoring over `question + GT answer + mRAG answer`
- Weighted aggregation over QA-level means and page-balanced means

Acceptance criteria:

- Each module has its own metrics and there is also an end-to-end score

## Phase 6: SFT and Research Enhancements

Goals:

- Improve chart understanding and fine-grained QA ability

Suggested implementation:

- Fine-tune on `PDFVQA`, `ChartQA`, and custom datasets
- Also experiment with SFT for the reranker

Acceptance criteria:

- Fine-grained QA and chart QA improve over the untuned baseline
