from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from agent_rag.data.schemas import DocumentRecord, QAPair
from agent_rag.services.storage import MetadataStore


DEFAULT_SYSTEM_PROMPT = (
    "You are a paper page question-answering assistant. "
    "Answer only from the provided PDF page image. "
    "If the answer is a short heading, label, boolean value, or phrase, "
    "return only that short answer without extra explanation."
)

LAYOUT_RISK_TOKENS = (
    "left section",
    "right section",
    "top left",
    "top right",
    "bottom left",
    "bottom right",
    "bottom section",
    "top section",
    "first section",
    "last section",
)


def _should_keep_pair(
    qa_pair: QAPair,
    allowed_sources: set[str] | None,
    accepted_only: bool,
    exclude_layout_questions: bool,
) -> bool:
    if not qa_pair.question or not qa_pair.answer:
        return False
    if allowed_sources is not None and qa_pair.source not in allowed_sources:
        return False
    if exclude_layout_questions:
        lowered_question = qa_pair.question.lower()
        if any(token in lowered_question for token in LAYOUT_RISK_TOKENS):
            return False
    if accepted_only:
        if qa_pair.source == "qwen2.5-vl":
            if qa_pair.groundedness_passed is False or qa_pair.standalone_passed is False:
                return False
    return True


def _build_record(
    document: DocumentRecord,
    qa_pair: QAPair,
    qa_index: int,
    system_prompt: str,
) -> dict[str, object] | None:
    page_number = qa_pair.page_number or 1
    page = next((item for item in document.pages if item.page_number == page_number), None)
    if page is None or not page.image_path:
        return None

    return {
        "id": f"{document.document_id}::{page_number}::{qa_index}",
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": page.image_path},
                    {"type": "text", "text": f"Question: {qa_pair.question}"},
                ],
            },
            {
                "role": "assistant",
                "content": qa_pair.answer,
            },
        ],
        "metadata": {
            "document_id": document.document_id,
            "title": document.title,
            "page_number": page_number,
            "source": qa_pair.source,
            "qa_type": qa_pair.qa_type,
            "task_type": qa_pair.task_type,
            "groundedness_passed": qa_pair.groundedness_passed,
            "standalone_passed": qa_pair.standalone_passed,
        },
    }


def export_dataset(
    *,
    output_path: Path,
    qa_sources: set[str] | None,
    accepted_only: bool,
    exclude_layout_questions: bool,
    limit: int | None,
    system_prompt: str,
) -> dict[str, object]:
    store = MetadataStore()
    documents = store.list_documents()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept_records = 0
    source_counter: Counter[str] = Counter()
    document_counter: Counter[str] = Counter()
    page_counter: Counter[str] = Counter()

    with output_path.open("w", encoding="utf-8") as handle:
        for document in documents:
            for qa_index, qa_pair in enumerate(document.qa_pairs):
                if limit is not None and kept_records >= limit:
                    break
                if not _should_keep_pair(
                    qa_pair=qa_pair,
                    allowed_sources=qa_sources,
                    accepted_only=accepted_only,
                    exclude_layout_questions=exclude_layout_questions,
                ):
                    continue

                record = _build_record(
                    document=document,
                    qa_pair=qa_pair,
                    qa_index=qa_index,
                    system_prompt=system_prompt,
                )
                if record is None:
                    continue

                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept_records += 1
                source_counter[str(qa_pair.source or "unknown")] += 1
                document_counter[document.document_id] += 1
                page_counter[f"{document.document_id}::p{qa_pair.page_number or 1}"] += 1
            if limit is not None and kept_records >= limit:
                break

    return {
        "output_path": str(output_path),
        "record_count": kept_records,
        "source_distribution": dict(source_counter),
        "document_count": len(document_counter),
        "page_count": len(page_counter),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export multimodal SFT dataset from stored PaperAgent-RAG metadata.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--qa-sources",
        type=str,
        default="",
        help="Comma-separated QA sources to keep, such as qwen2.5-vl.",
    )
    parser.add_argument(
        "--accepted-only",
        action="store_true",
        help="Keep only accepted synthetic QA pairs when source is qwen2.5-vl.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of exported examples.",
    )
    parser.add_argument(
        "--exclude-layout-questions",
        action="store_true",
        help="Filter out high-risk layout questions such as left/right/top/bottom section prompts.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt to embed into each SFT record.",
    )
    args = parser.parse_args()

    qa_sources = {item.strip() for item in args.qa_sources.split(",") if item.strip()} or None
    summary = export_dataset(
        output_path=args.output,
        qa_sources=qa_sources,
        accepted_only=args.accepted_only,
        exclude_layout_questions=args.exclude_layout_questions,
        limit=args.limit,
        system_prompt=args.system_prompt,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
