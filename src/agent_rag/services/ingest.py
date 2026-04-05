from __future__ import annotations

import shutil
from pathlib import Path

import fitz
from PIL import Image

from agent_rag.config import (
    DATASET_ASSETS_DIR,
    PAPERS_DIR,
    RENDERED_PAGES_DIR,
    ensure_data_dirs,
)
from agent_rag.data.schemas import (
    DatasetImportRequest,
    DocumentRecord,
    DocumentRegistrationRequest,
    DocumentSource,
    PageRecord,
    QAPair,
)


class DocumentIngestService:
    """
    Ingest PDF files and render them into page images for page-level retrieval.
    """

    def __init__(self, dpi: int = 300) -> None:
        self.dpi = dpi
        ensure_data_dirs()

    def _copy_pdf_to_workspace(self, source_pdf: Path, document_id: str) -> Path:
        document_dir = PAPERS_DIR / document_id
        document_dir.mkdir(parents=True, exist_ok=True)
        destination = document_dir / source_pdf.name
        if source_pdf != destination:
            shutil.copy2(source_pdf, destination)
        return destination

    def _render_pdf_pages(self, pdf_path: Path, document_id: str) -> list[PageRecord]:
        output_dir = RENDERED_PAGES_DIR / document_id
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use a zoom matrix derived from DPI so exported PNGs keep enough detail
        # for page-level multimodal retrieval and downstream VLM reasoning.
        zoom = self.dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pages: list[PageRecord] = []

        with fitz.open(pdf_path) as pdf:
            for page_index, page in enumerate(pdf, start=1):
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                image_path = output_dir / f"page_{page_index:04d}.png"
                pixmap.save(image_path)
                pages.append(
                    PageRecord(
                        page_number=page_index,
                        image_path=str(image_path),
                        width=pixmap.width,
                        height=pixmap.height,
                    )
                )

        return pages

    def _save_dataset_page_image(
        self,
        image: Image.Image,
        document_id: str,
    ) -> PageRecord:
        output_dir = RENDERED_PAGES_DIR / document_id
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image = image.convert("RGB")
        image_path = output_dir / "page_0001.png"
        image.save(image_path)
        return PageRecord(
            page_number=1,
            image_path=str(image_path),
            width=image.width,
            height=image.height,
        )

    def _save_dataset_object_images(
        self,
        images: list[Image.Image],
        document_id: str,
        persist_object_images: bool,
    ) -> list[str]:
        if not persist_object_images:
            return []

        object_dir = DATASET_ASSETS_DIR / document_id / "objects"
        if object_dir.exists():
            shutil.rmtree(object_dir)
        object_dir.mkdir(parents=True, exist_ok=True)

        image_paths: list[str] = []
        for index, image in enumerate(images, start=1):
            rgb_image = image.convert("RGB")
            path = object_dir / f"object_{index:04d}.png"
            rgb_image.save(path)
            image_paths.append(str(path))
        return image_paths

    def register_document(
        self,
        request: DocumentRegistrationRequest,
    ) -> DocumentRecord:
        pdf_path = Path(request.pdf_path).expanduser()
        if not pdf_path.is_absolute():
            pdf_path = Path.cwd() / pdf_path
        pdf_path = pdf_path.resolve()

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        workspace_pdf_path = self._copy_pdf_to_workspace(
            source_pdf=pdf_path,
            document_id=request.document_id,
        )
        rendered_pages = self._render_pdf_pages(
            pdf_path=workspace_pdf_path,
            document_id=request.document_id,
        )

        return DocumentRecord(
            document_id=request.document_id,
            title=request.title,
            pdf_path=str(workspace_pdf_path),
            authors=request.authors,
            tags=request.tags,
            source=DocumentSource(kind="pdf"),
            pages=rendered_pages,
        )

    def import_pdfvqa(
        self,
        request: DatasetImportRequest,
    ) -> list[DocumentRecord]:
        from itertools import islice

        from datasets import load_dataset

        dataset = load_dataset(
            request.dataset_name,
            split=request.split,
            streaming=True,
        )

        records: list[DocumentRecord] = []
        for relative_index, row in enumerate(
            islice(dataset, request.start_index, request.start_index + request.limit)
        ):
            row_index = request.start_index + relative_index
            document_id = f"{request.document_id_prefix}-{request.split}-{row_index}"
            title = f"{request.dataset_name} {request.split} sample {row_index}"

            page_record = self._save_dataset_page_image(
                image=row["page"],
                document_id=document_id,
            )
            page_record.texts = list(row.get("texts") or [])
            page_record.object_image_paths = self._save_dataset_object_images(
                images=list(row.get("images") or []),
                document_id=document_id,
                persist_object_images=request.persist_object_images,
            )
            page_record.object_ids = list(row.get("object_ids") or [])
            page_record.bboxes = [list(item) for item in (row.get("bboxes") or [])]
            page_record.relations = [list(item) for item in (row.get("relations") or [])]
            page_record.categories = list(row.get("categories") or [])
            page_record.raw_metadata = {
                "text_count": len(page_record.texts),
                "object_count": len(page_record.object_ids),
                "category_count": len(page_record.categories),
            }

            questions = list(row.get("questions") or [])
            answers = list(row.get("answers") or [])
            types = list(row.get("types") or [])
            task_types = list(row.get("task types") or [])
            qa_pairs: list[QAPair] = []
            for index, question in enumerate(questions):
                qa_pairs.append(
                    QAPair(
                        question=question,
                        answer=answers[index] if index < len(answers) else "",
                        qa_type=types[index] if index < len(types) else None,
                        task_type=task_types[index] if index < len(task_types) else None,
                        page_number=1,
                    )
                )

            record = DocumentRecord(
                document_id=document_id,
                title=title,
                source=DocumentSource(
                    kind="dataset",
                    dataset_name=request.dataset_name,
                    split=request.split,
                    row_index=row_index,
                ),
                pages=[page_record],
                qa_pairs=qa_pairs,
                metadata={
                    "dataset_name": request.dataset_name,
                    "split": request.split,
                    "row_index": row_index,
                },
            )
            records.append(record)

        return records
