from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

DATA_ROOT = PROJECT_ROOT / "workspace_data"
PAPERS_DIR = DATA_ROOT / "papers"
RENDERED_PAGES_DIR = DATA_ROOT / "rendered_pages"
DATASET_ASSETS_DIR = DATA_ROOT / "dataset_assets"
INDEX_DIR = DATA_ROOT / "indices"
METADATA_DIR = DATA_ROOT / "metadata"
EVALUATIONS_DIR = DATA_ROOT / "evaluations"
UPLOADS_DIR = DATA_ROOT / "uploads"
BYALDI_INDEX_ROOT = INDEX_DIR / "byaldi"
BYALDI_STAGING_ROOT = INDEX_DIR / "staging"
DEFAULT_INDEX_NAME = "paper_index"
DEFAULT_RETRIEVAL_MODEL_NAME = "vidore/colqwen2-v1.0"
DEFAULT_RERANKER_MODEL_NAME = "lightonai/MonoQwen2-VL-v0.1"
DEFAULT_RERANKER_PROCESSOR_NAME = "Qwen/Qwen2-VL-2B-Instruct"
DEFAULT_GENERATION_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_EVAL_JUDGE_MODEL_NAME = os.getenv("OPENAI_EVAL_MODEL", "gpt-5.4-mini")
FRONTEND_DIR = PROJECT_ROOT / "frontend"
FRONTEND_ASSETS_DIR = FRONTEND_DIR / "assets"


def ensure_data_dirs() -> None:
    """Create local storage folders for the MVP."""
    for path in (
        DATA_ROOT,
        PAPERS_DIR,
        RENDERED_PAGES_DIR,
        DATASET_ASSETS_DIR,
        INDEX_DIR,
        METADATA_DIR,
        EVALUATIONS_DIR,
        UPLOADS_DIR,
        BYALDI_INDEX_ROOT,
        BYALDI_STAGING_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)
