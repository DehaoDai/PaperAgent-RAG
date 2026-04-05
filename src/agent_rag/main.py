from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from agent_rag.api.routes import router
from agent_rag.config import (
    FRONTEND_ASSETS_DIR,
    RENDERED_PAGES_DIR,
    ensure_data_dirs,
)


ensure_data_dirs()

app = FastAPI(
    title="PaperAgent-RAG",
    version="0.1.0",
    description="Multimodal RAG MVP scaffold for paper question answering.",
)
app.mount("/assets", StaticFiles(directory=FRONTEND_ASSETS_DIR), name="assets")
app.mount("/rendered_pages", StaticFiles(directory=RENDERED_PAGES_DIR), name="rendered_pages")
app.include_router(router)
