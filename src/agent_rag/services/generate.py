from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from PIL import Image

from agent_rag.config import DEFAULT_GENERATION_MODEL_NAME
from agent_rag.data.schemas import DocumentRecord, EvidencePage, QAPair, QASynthesisResult


DEFAULT_QA_GENERATION_PROMPT = """You are shown one page image extracted from a PDF.
Generate {questions_per_page} distinct fact-based question-answer pairs for use in Retrieval-Augmented Generation (RAG).

Requirements
1. Grounded - Each answer MUST appear verbatim in the page image. Do not use outside knowledge or reasoning.
2. Specific - Ask for a concrete item such as a date, name, section heading, figure label, table entry, or short phrase.
3. Search-style - Word each question like something a user might type into a search engine.
4. Standalone - Each question must be fully understandable on its own without mentioning an image, page, figure above, or document.
5. Concise answer - Return the shortest exact text span that answers the question, preferably under 20 tokens.
6. Diversity - The questions should target different facts from the page whenever possible.
7. Output only the requested pairs and no commentary.

Return your result in exactly this format and nothing else:

Output:::
Factoid question 1: (question 1)
Answer 1: (answer 1)
Factoid question 2: (question 2)
Answer 2: (answer 2)
...

Now here is the PDF page image.

Output:::"""

DEFAULT_GROUNDEDNESS_CRITIQUE_PROMPT = """You are checking whether a question can be answered directly from a PDF page image.

Question: {question}
Answer: {answer}

Criterion
- Groundedness means the answer is directly supported by the page image and does not require outside knowledge.

Respond in exactly this format and nothing else:
Verdict: YES or NO
Reason: (one short sentence)
"""

DEFAULT_STANDALONE_CRITIQUE_PROMPT = """You are checking whether a question is understandable on its own.

Question: {question}

Criterion
- Standalone means the question is fully understandable without extra context such as "this figure", "the page above", "this paper", or similar references.

Respond in exactly this format and nothing else:
Verdict: YES or NO
Reason: (one short sentence)
"""


class AnswerGenerator:
    """Generate final answers from retrieved page images with Qwen2.5-VL."""

    def __init__(self, default_model_name: str = DEFAULT_GENERATION_MODEL_NAME) -> None:
        self.default_model_name = default_model_name
        self._loaded_model_name: str | None = None
        self._processor: Any = None
        self._model: Any = None
        self._device: str | None = None

    def _resolve_device(self) -> str:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        # Qwen2.5-VL generation currently hits MPS temporary NDArray limits
        # on this local setup, so prefer CPU over MPS for stability.
        return "cpu"

    def _resolve_dtype(self):
        import torch

        device = self._resolve_device()
        if device == "cuda":
            return torch.bfloat16
        if device == "mps":
            return torch.float16
        return torch.float32

    def _ensure_model(self, model_name: str) -> None:
        if self._loaded_model_name == model_name and self._model is not None:
            return

        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self._device = self._resolve_device()
        self._processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=224 * 28 * 28,
            max_pixels=640 * 28 * 28,
        )
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=self._resolve_dtype(),
        )
        self._model.to(self._device)
        self._model.eval()
        if hasattr(self._model, "generation_config"):
            self._model.generation_config.temperature = None
        self._loaded_model_name = model_name

    def _run_single_image_prompt(
        self,
        image_path: str,
        prompt: str,
        model_name: str,
        max_new_tokens: int,
    ) -> str:
        self._ensure_model(model_name)

        with Image.open(image_path) as raw_image:
            image = raw_image.convert("RGB")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self._processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self._device)

        output_ids = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        return self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0].strip()

    def _placeholder_answer(self, question: str, evidences: list[EvidencePage]) -> str:
        if not evidences:
            return "No usable evidence was retrieved, so the system cannot answer this question yet."

        evidence_summary = "；".join(
            f"{item.title} page {item.page_number}"
            for item in evidences
        )
        return (
            f"This is a placeholder answer built from the current retrieved evidence. "
            f"Question: {question}. Candidate evidence: {evidence_summary}. "
            "Once the full multimodal retrieval and VLM answer path is available, "
            "the response will be grounded more directly in the page images."
        )

    def generate(
        self,
        question: str,
        evidences: list[EvidencePage],
        model_name: str | None = None,
        max_new_tokens: int = 256,
        max_images: int = 3,
    ) -> tuple[str, dict[str, object]]:
        if not evidences:
            return self._placeholder_answer(question, evidences), {
                "generator": "qwen2_5_vl",
                "generator_status": "no_evidence",
            }

        resolved_model_name = model_name or self.default_model_name
        effective_max_images = max_images
        if self._resolve_device() == "mps":
            effective_max_images = min(effective_max_images, 1)
        selected = [
            item for item in evidences[:effective_max_images]
            if item.image_path and Path(item.image_path).exists()
        ]
        if not selected:
            return self._placeholder_answer(question, evidences), {
                "generator": "qwen2_5_vl",
                "generator_model_name": resolved_model_name,
                "generator_status": "fallback_no_images",
            }

        self._ensure_model(resolved_model_name)

        page_list = ", ".join(f"p{item.page_number}" for item in selected)
        prompt = (
            "You are a paper question-answering assistant. Answer only using the provided "
            "paper page images.\n"
            "Requirements:\n"
            "1. Use only evidence from the images and do not hallucinate.\n"
            "2. Keep the answer concise and accurate.\n"
            "3. Cite supporting pages using the format [pX].\n"
            "4. If the evidence is insufficient, say 'Insufficient evidence.'\n\n"
            f"Candidate pages: {page_list}\n"
            f"Question: {question}"
        )

        if len(selected) == 1:
            answer = self._run_single_image_prompt(
                image_path=selected[0].image_path,
                prompt=prompt,
                model_name=resolved_model_name,
                max_new_tokens=max_new_tokens,
            )
        else:
            pil_images = []
            for evidence in selected:
                with Image.open(evidence.image_path) as raw_image:
                    pil_images.append(raw_image.convert("RGB"))

            messages = [
                {
                    "role": "user",
                    "content": (
                        [{"type": "image", "image": image} for image in pil_images]
                        + [{"type": "text", "text": prompt}]
                    ),
                }
            ]
            text = self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self._processor(
                text=[text],
                images=pil_images,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self._device)

            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
            answer = self._processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )[0].strip()

        if not answer:
            answer = self._placeholder_answer(question, evidences)
            return answer, {
                "generator": "qwen2_5_vl",
                "generator_model_name": resolved_model_name,
                "generator_device": self._device or self._resolve_device(),
                "generator_status": "fallback_empty_output",
            }

        return answer, {
            "generator": "qwen2_5_vl",
            "generator_model_name": resolved_model_name,
            "generator_device": self._device or self._resolve_device(),
            "generator_status": "applied",
            "generator_pages_used": [item.page_number for item in selected],
            "generator_effective_max_images": effective_max_images,
        }

    def _parse_synthesized_qa(self, output: str) -> tuple[str, str]:
        question_match = re.search(
            r"Factoid question:\s*(.+)",
            output,
            flags=re.IGNORECASE,
        )
        answer_match = re.search(
            r"Answer:\s*(.+)",
            output,
            flags=re.IGNORECASE,
        )
        if not question_match or not answer_match:
            raise ValueError(
                "Failed to parse synthesized QA. Expected lines starting with "
                "'Factoid question:' and 'Answer:'."
            )

        question = question_match.group(1).strip()
        answer = answer_match.group(1).strip()
        if not question or not answer:
            raise ValueError("Synthesized QA is missing a non-empty question or answer.")
        return question, answer

    def _parse_multiple_synthesized_qas(self, output: str) -> list[tuple[str, str]]:
        question_pattern = re.compile(r"^Factoid question(?:\s+\d+)?:\s*(.+)$", re.IGNORECASE)
        answer_pattern = re.compile(r"^Answer(?:\s+\d+)?:\s*(.+)$", re.IGNORECASE)

        pairs: list[tuple[str, str]] = []
        pending_question: str | None = None
        for raw_line in output.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            question_match = question_pattern.match(line)
            if question_match:
                pending_question = " ".join(question_match.group(1).strip().split())
                continue
            answer_match = answer_pattern.match(line)
            if answer_match and pending_question:
                answer = " ".join(answer_match.group(1).strip().split())
                if pending_question and answer:
                    pairs.append((pending_question, answer))
                pending_question = None

        if not pairs:
            single_question, single_answer = self._parse_synthesized_qa(output)
            return [(single_question, single_answer)]
        return pairs

    def _parse_critique_verdict(self, output: str) -> tuple[bool, str]:
        verdict_match = re.search(r"Verdict:\s*(YES|NO)", output, flags=re.IGNORECASE)
        fallback_verdict_match = re.search(r"\b(YES|NO)\b", output, flags=re.IGNORECASE)
        reason_match = re.search(r"Reason:\s*(.+)", output, flags=re.IGNORECASE | re.DOTALL)
        if verdict_match:
            verdict = verdict_match.group(1).strip().upper() == "YES"
        elif fallback_verdict_match:
            verdict = fallback_verdict_match.group(1).strip().upper() == "YES"
        else:
            raise ValueError("Failed to parse critique verdict. Expected a YES/NO judgment.")

        if reason_match:
            reason = " ".join(reason_match.group(1).strip().split())
        else:
            lines = [line.strip() for line in output.splitlines() if line.strip()]
            trailing_lines = []
            verdict_consumed = False
            for line in lines:
                if not verdict_consumed and re.search(r"\b(YES|NO)\b", line, flags=re.IGNORECASE):
                    verdict_consumed = True
                    continue
                if verdict_consumed:
                    trailing_lines.append(line)
            reason = " ".join(" ".join(trailing_lines).split())
        return verdict, reason

    def _critique_candidate(
        self,
        image_path: str,
        question: str,
        answer: str,
        model_name: str,
        max_new_tokens: int,
        criterion: str,
    ) -> tuple[bool, str, str]:
        if criterion == "groundedness":
            prompt = DEFAULT_GROUNDEDNESS_CRITIQUE_PROMPT.format(
                question=question,
                answer=answer,
            )
        elif criterion == "standalone":
            prompt = DEFAULT_STANDALONE_CRITIQUE_PROMPT.format(question=question)
        else:
            raise ValueError(f"Unsupported critique criterion: {criterion}")

        output = self._run_single_image_prompt(
            image_path=image_path,
            prompt=prompt,
            model_name=model_name,
            max_new_tokens=max_new_tokens,
        )
        verdict, reason = self._parse_critique_verdict(output)
        return verdict, reason, output

    def synthesize_factoid_qa(
        self,
        documents: list[DocumentRecord],
        model_name: str | None = None,
        questions_per_page: int = 3,
        max_new_tokens: int = 128,
        filter_groundedness: bool = True,
        filter_standalone: bool = True,
        prompt_template: str | None = None,
        overwrite_synthetic: bool = False,
    ) -> tuple[list[DocumentRecord], list[QASynthesisResult], dict[str, object]]:
        resolved_model_name = model_name or self.default_model_name
        resolved_prompt = (prompt_template or DEFAULT_QA_GENERATION_PROMPT).format(
            questions_per_page=questions_per_page,
        )
        synthesized_results: list[QASynthesisResult] = []
        updated_documents: list[DocumentRecord] = []
        skipped_documents: list[str] = []

        for document in documents:
            if not document.pages:
                skipped_documents.append(document.document_id)
                updated_documents.append(document)
                continue

            page = document.pages[0]
            if not page.image_path or not Path(page.image_path).exists():
                skipped_documents.append(document.document_id)
                updated_documents.append(document)
                continue

            output = self._run_single_image_prompt(
                image_path=page.image_path,
                prompt=resolved_prompt,
                model_name=resolved_model_name,
                max_new_tokens=max_new_tokens,
            )
            candidate_pairs = self._parse_multiple_synthesized_qas(output)
            accepted_pairs: list[QAPair] = []

            for question, answer in candidate_pairs[:questions_per_page]:
                groundedness_passed = True
                standalone_passed = True
                groundedness_reason = ""
                standalone_reason = ""

                if filter_groundedness:
                    groundedness_passed, groundedness_reason, _ = self._critique_candidate(
                        image_path=page.image_path,
                        question=question,
                        answer=answer,
                        model_name=resolved_model_name,
                        max_new_tokens=64,
                        criterion="groundedness",
                    )
                if filter_standalone:
                    standalone_passed, standalone_reason, _ = self._critique_candidate(
                        image_path=page.image_path,
                        question=question,
                        answer=answer,
                        model_name=resolved_model_name,
                        max_new_tokens=64,
                        criterion="standalone",
                    )

                accepted = groundedness_passed and standalone_passed
                synthesized_results.append(
                    QASynthesisResult(
                        document_id=document.document_id,
                        question=question,
                        answer=answer,
                        page_number=page.page_number,
                        groundedness_passed=groundedness_passed,
                        standalone_passed=standalone_passed,
                        groundedness_judgment=groundedness_reason or None,
                        standalone_judgment=standalone_reason or None,
                        accepted=accepted,
                    )
                )

                if not accepted:
                    continue

                accepted_pairs.append(
                    QAPair(
                        question=question,
                        answer=answer,
                        qa_type="synthetic_factoid",
                        task_type="rag_qa_generation",
                        source="qwen2.5-vl",
                        page_number=page.page_number,
                        groundedness_passed=groundedness_passed,
                        standalone_passed=standalone_passed,
                        critique_metadata={
                            "groundedness_reason": groundedness_reason,
                            "standalone_reason": standalone_reason,
                        },
                    )
                )

            retained_pairs = document.qa_pairs
            if overwrite_synthetic:
                retained_pairs = [
                    pair for pair in document.qa_pairs
                    if pair.source != "qwen2.5-vl"
                ]
            updated_document = document.model_copy(
                update={"qa_pairs": [*retained_pairs, *accepted_pairs]},
            )
            updated_documents.append(updated_document)

        return updated_documents, synthesized_results, {
            "synthesis_model_name": resolved_model_name,
            "synthesis_device": self._device or self._resolve_device(),
            "synthesis_prompt_template": resolved_prompt,
            "questions_per_page": questions_per_page,
            "filter_groundedness": filter_groundedness,
            "filter_standalone": filter_standalone,
            "skipped_documents": skipped_documents,
        }
