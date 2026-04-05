# Contributing

Thanks for your interest in improving PaperAgent-RAG.

## Local Setup

### App environment

```bash
make setup-app
cp .env.example .env
make run
```

### Training environment

```bash
make setup-sft
```

We intentionally keep two virtual environments:

- `.venv` for the FastAPI app, retrieval, reranking, generation, and evaluation
- `.venv-sft` for LLaMA-Factory training

This avoids dependency conflicts between document retrieval and fine-tuning tools.

## Development Guidelines

- Keep edits ASCII unless the file already uses Unicode and there is a good reason.
- Prefer small, reviewable pull requests.
- Run local checks before opening a PR:

```bash
make check
```

## What We Welcome

- bug fixes
- documentation improvements
- model integration improvements
- evaluation and reproducibility improvements
- frontend usability improvements

## Pull Requests

Please include:

- a short description of the change
- how you tested it
- any environment assumptions

If your change touches retrieval, reranking, or evaluation, include a small before/after example when possible.
