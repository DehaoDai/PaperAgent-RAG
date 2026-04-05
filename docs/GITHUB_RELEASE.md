# GitHub Release Kit

This file contains copy-ready text for the GitHub repository homepage and the first public release.

## Repository Name

`PaperAgent-RAG`

## Short Repository Description

Multimodal RAG for paper-style PDF QA with ColQwen2 retrieval, MonoQwen2-VL reranking, Qwen2.5-VL answering, judged evaluation, and SFT prep.

## About Section

PaperAgent-RAG is an open multimodal RAG starter kit for paper-style PDF question answering. It supports page-image ingestion, ColQwen2 late-interaction retrieval, optional MonoQwen2-VL reranking, Qwen2.5-VL answer generation, PDFVQA-based data workflows, OpenAI-judged evaluation, and first-pass SFT data preparation.

## Suggested Topics

Use these as GitHub repository topics:

- `rag`
- `multimodal-rag`
- `pdf-qa`
- `document-ai`
- `vision-language-model`
- `qwen2-5-vl`
- `colqwen2`
- `colpali`
- `fastapi`
- `llm`
- `vlm`
- `retrieval`
- `reranking`
- `docvqa`
- `pdfvqa`
- `sft`
- `llama-factory`
- `research-tooling`

## Social Preview Caption

Open multimodal RAG starter kit for paper-style PDFs: retrieve page images with ColQwen2, rerank with MonoQwen2-VL, answer with Qwen2.5-VL, evaluate with an OpenAI judge, and prepare SFT data in one repo.

## Pinned Repository One-Liner

Paper-first multimodal RAG with retrieval, reranking, generation, evaluation, and SFT prep.

## First Release Title

`v0.1.0 - Public Starter Release`

## First Release Notes

### Highlights

- Added a complete multimodal paper QA starter workflow
- Added PDF-to-page-image ingestion for local papers
- Added dataset import flow for `gigant/pdfvqa`
- Added ColQwen2 + Byaldi page-image retrieval
- Added optional MonoQwen2-VL reranking
- Added Qwen2.5-VL answer generation
- Added synthetic QA generation with groundedness and standalone filtering
- Added OpenAI-judged evaluation with normalized and page-balanced scoring
- Added SFT export, train/eval split, and LLaMA-Factory run packaging
- Added a lightweight FastAPI web UI

### What Is Included

- local PDF upload and registration
- PDFVQA import and synthetic QA generation
- retrieval, reranking, generation, and evaluation endpoints
- browser-based demo console
- starter SFT workflow for Qwen2.5-VL
- GitHub-friendly setup with `.env.example`, `Makefile`, and CI smoke checks

### Quick Start

```bash
git clone https://github.com/DehaoDai/PaperAgent-RAG
cd PaperAgent-RAG
cp .env.example .env
make setup-app
make run
```

### Notes

- First-time use will download models from Hugging Face.
- On Apple Silicon, Qwen2.5-VL generation may run on CPU for stability.
- Training uses a separate `.venv-sft` environment to avoid dependency conflicts.

## Release Announcement Post

I just open-sourced `PaperAgent-RAG`, a multimodal RAG starter kit for paper-style PDF question answering.

It includes:

- PDF page-image ingestion
- ColQwen2 retrieval with Byaldi
- MonoQwen2-VL reranking
- Qwen2.5-VL answer generation
- PDFVQA import and synthetic QA generation
- OpenAI-judged evaluation
- SFT export and LLaMA-Factory preparation

The goal is to make it easier to prototype research-oriented paper agents on top of page images instead of relying only on OCR plus text chunking.

## Maintainer Checklist Before Publishing

- confirm the `LICENSE` file matches your intended public release terms
- confirm no secrets exist in `.env`, notebooks, or committed metadata
- ensure `workspace_data/` is not committed
- confirm README and release text point to `https://github.com/DehaoDai/PaperAgent-RAG`
- optionally add a real screenshot from the running web UI
