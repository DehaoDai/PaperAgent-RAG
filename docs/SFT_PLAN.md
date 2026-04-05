# SFT Plan

## Goal

Improve the multimodal generation quality of `Qwen2.5-VL` for paper-style PDF QA, with a special focus on:

- chart and table summarization
- short grounded answers on document pages
- page-region understanding such as headings and local sections
- reducing over-generation when the ground-truth answer is a short phrase

This project should treat generator SFT and reranker SFT as two separate tracks.

## Recommendation

Start with generator SFT first.

Reason:

- Current evaluation errors are dominated by answer generation and page-layout understanding.
- The system already retrieves the correct page in many cases, but the generated answer is often too long, region-confused, or insufficiently grounded.
- Reranker SFT becomes more valuable after we confirm that retrieval candidates are already close to correct and generation remains the bottleneck.

## Track A: Generator SFT

Target model:

- `Qwen/Qwen2.5-VL-3B-Instruct` for local prototyping
- `Qwen/Qwen2.5-VL-7B-Instruct` for a stronger later-stage run if hardware allows

Primary objectives:

1. improve paper-page QA
2. improve chart and table summary quality
3. make answers shorter and more faithful
4. strengthen page-region and section-label understanding

### Recommended dataset mixture

Phase 1 should use the datasets we already have or can prepare quickly:

- `gigant/pdfvqa` original QA pairs
- accepted synthetic QA pairs generated from our pipeline

Phase 2 should mix in more task-specific data:

- `ChartQA`
- `DocVQA`
- `Docmatix`

### Suggested mixture ratio

For the first useful training run:

- `40%` PDFVQA original QA
- `30%` accepted synthetic QA from paper pages
- `30%` chart/table-oriented QA or summary data

If chart and table performance is the main target, shift to:

- `30%` PDFVQA original QA
- `20%` accepted synthetic QA
- `50%` chart/table data

## Track B: Reranker SFT

Target model:

- `lightonai/MonoQwen2-VL-v0.1` style reranker

Primary objective:

- improve ranking quality over visually similar but semantically different pages

Training task:

- pointwise or pairwise page relevance scoring

Positive examples:

- ground-truth answer page

Hard negative examples:

- same paper, wrong page, similar section title
- same paper, another figure page
- same paper, another table page
- top-k retrieval pages that look relevant but do not contain the answer

This track should start only after generator SFT baseline experiments are running.

## Data policy

Accepted synthetic QA pairs should pass all of the following:

- groundedness
- standalone
- non-empty answer
- answer length sanity check

High-risk layout-only questions should either be filtered or tagged separately.

Examples:

- `What is the left section about?`
- `What is the bottom section?`
- `What is the last section in this page?`

These are useful for a future layout-aware model, but they currently degrade end-to-end QA if mixed blindly with normal factoid QA.

The first-pass export tool in this repo supports filtering these questions out before training.

## Training sample format

The first export target should be JSONL with multimodal chat-style messages.

Each record should look like:

```json
{
  "id": "pdfvqa-debug-validation-0::1::0",
  "messages": [
    {
      "role": "system",
      "content": "You are a paper page question-answering assistant. Answer only from the provided PDF page image. If the answer is a short heading, label, or phrase, return only that phrase."
    },
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "/abs/path/to/page.png"},
        {"type": "text", "text": "Question: What is the topic of the bottom left section?"}
      ]
    },
    {
      "role": "assistant",
      "content": "Background"
    }
  ],
  "metadata": {
    "document_id": "pdfvqa-debug-validation-0",
    "page_number": 1,
    "source": "qwen2.5-vl",
    "qa_type": "synthetic_factoid"
  }
}
```

## Experiment order

### Stage 1

- export current PDFVQA original QA
- export accepted synthetic QA
- train a small generator SFT run
- evaluate with current OpenAI-judge pipeline

### Stage 2

- add chart/table data
- compare against Stage 1
- inspect whether low-score region questions remain dominant

### Stage 3

- build reranker training set
- train reranker
- compare:
  - baseline
  - generator SFT only
  - reranker SFT only
  - generator SFT + reranker SFT

## Minimum ablations

Always keep these comparisons:

1. no SFT
2. generator SFT on PDFVQA only
3. generator SFT on PDFVQA + synthetic QA
4. generator SFT on PDFVQA + synthetic QA + chart data
5. reranker SFT added on top

## Metrics

Use both answer-level and page-level metrics:

- retrieval: `Recall@k`, `MRR`
- answer quality: OpenAI-judge score
- aggregate:
  - `mean_normalized_score`
  - `page_balanced_normalized_score`
  - `weighted_normalized_score`

## Immediate deliverables

This repo should first support:

1. exporting multimodal SFT JSONL from stored metadata
2. filtering by QA source
3. filtering by accepted synthetic QA flags
4. emitting dataset statistics for sanity checks

Training code can then be connected to:

- `trl`
- `transformers`
- `LLaMA-Factory`

depending on the final stack choice
