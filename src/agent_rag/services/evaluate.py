from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from agent_rag.config import DEFAULT_EVAL_JUDGE_MODEL_NAME, EVALUATIONS_DIR
from agent_rag.data.schemas import (
    EvaluationAggregate,
    EvaluationExampleResult,
    EvaluationPageSummary,
    EvaluationRequest,
    EvaluationResponse,
    QueryRequest,
)
from agent_rag.pipelines.qa import QAPipeline
from agent_rag.services.storage import MetadataStore


DEFAULT_EVALUATION_PROMPT = """You are grading an mRAG system answer against a reference answer.

Question:
{question}

Reference answer:
{reference_answer}

Candidate answer:
{predicted_answer}

Evaluate whether the candidate answer is correct, accurate, and factual based on the reference answer.

Scoring rubric
Score 1: The response is completely incorrect, inaccurate, and/or not factual.
Score 2: The response is mostly incorrect, inaccurate, and/or not factual.
Score 3: The response is somewhat correct, accurate, and/or factual.
Score 4: The response is mostly correct, accurate, and factual.
Score 5: The response is completely correct, accurate, and factual.

Return exactly this format and nothing else:
Correct: YES or NO
Accurate: YES or NO
Factual: YES or NO
Score: 1, 2, 3, 4, or 5
Rationale: one short paragraph
"""


class OpenAIAnswerJudge:
    def __init__(self, default_model_name: str = DEFAULT_EVAL_JUDGE_MODEL_NAME) -> None:
        self.default_model_name = default_model_name

    def _build_client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The `openai` package is required for evaluation. Install project dependencies again."
            ) from exc

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it before running ChatGPT-based evaluation."
            )
        return OpenAI()

    def _parse_bool(self, output: str, field_name: str) -> bool | None:
        match = re.search(rf"{field_name}:\s*(YES|NO)", output, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip().upper() == "YES"
        return None

    def _parse_result(self, output: str) -> tuple[bool | None, bool | None, bool | None, int, str]:
        correct = self._parse_bool(output, "Correct")
        accurate = self._parse_bool(output, "Accurate")
        factual = self._parse_bool(output, "Factual")

        score_match = re.search(r"Score:\s*([1-5])", output, flags=re.IGNORECASE)
        if not score_match:
            fallback_score_match = re.search(r"\b([1-5])\b", output)
            if not fallback_score_match:
                raise ValueError("Failed to parse evaluation score from judge output.")
            raw_score = int(fallback_score_match.group(1))
        else:
            raw_score = int(score_match.group(1))

        rationale_match = re.search(r"Rationale:\s*(.+)", output, flags=re.IGNORECASE | re.DOTALL)
        rationale = " ".join((rationale_match.group(1).strip() if rationale_match else "").split())
        return correct, accurate, factual, raw_score, rationale

    def evaluate(
        self,
        *,
        question: str,
        reference_answer: str,
        predicted_answer: str,
        model_name: str | None = None,
    ) -> dict[str, object]:
        client = self._build_client()
        resolved_model_name = model_name or self.default_model_name
        prompt = DEFAULT_EVALUATION_PROMPT.format(
            question=question,
            reference_answer=reference_answer,
            predicted_answer=predicted_answer,
        )
        response = client.responses.create(
            model=resolved_model_name,
            input=prompt,
            reasoning={"effort": "none"},
            text={"verbosity": "low"},
        )
        output = (response.output_text or "").strip()
        correct, accurate, factual, raw_score, rationale = self._parse_result(output)
        return {
            "correct": correct,
            "accurate": accurate,
            "factual": factual,
            "raw_score": raw_score,
            "normalized_score": (raw_score - 1) / 4.0,
            "rationale": rationale,
            "judge_model_name": resolved_model_name,
            "judge_raw_output": output,
            "request_id": getattr(response, "_request_id", None),
        }


class MRAGEvaluator:
    def __init__(
        self,
        *,
        store: MetadataStore,
        qa_pipeline: QAPipeline,
        judge: OpenAIAnswerJudge,
    ) -> None:
        self.store = store
        self.qa_pipeline = qa_pipeline
        self.judge = judge

    def _collect_examples(self, request: EvaluationRequest):
        documents = self.store.list_documents()
        if request.document_ids:
            allowed = set(request.document_ids)
            documents = [doc for doc in documents if doc.document_id in allowed]

        examples = []
        for document in documents:
            for qa_pair in document.qa_pairs:
                if not qa_pair.question or not qa_pair.answer:
                    continue
                if request.qa_sources and qa_pair.source not in set(request.qa_sources):
                    continue
                examples.append((document, qa_pair))
                if len(examples) >= request.limit:
                    return examples
        return examples

    def _save_report(self, payload: dict[str, object]) -> tuple[str, Path]:
        run_id = datetime.now().strftime("eval_%Y%m%d_%H%M%S")
        report_path = EVALUATIONS_DIR / f"{run_id}.json"
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return run_id, report_path

    def run(self, request: EvaluationRequest) -> EvaluationResponse:
        examples = self._collect_examples(request)
        if not examples:
            raise ValueError("No QA pairs matched the evaluation request.")

        results: list[EvaluationExampleResult] = []
        page_buckets: dict[tuple[str, int], list[EvaluationExampleResult]] = defaultdict(list)

        for document, qa_pair in examples:
            query_response = self.qa_pipeline.run(
                QueryRequest(
                    question=qa_pair.question,
                    top_k=request.top_k,
                    index_name=request.index_name,
                    use_reranker=request.use_reranker,
                    reranker_model_name=request.reranker_model_name,
                    rerank_candidate_multiplier=request.rerank_candidate_multiplier,
                    use_generation_model=request.use_generation_model,
                    generation_model_name=request.generation_model_name,
                    generation_max_new_tokens=request.generation_max_new_tokens,
                    generation_max_images=request.generation_max_images,
                    document_ids=[document.document_id],
                )
            )
            judgment = self.judge.evaluate(
                question=qa_pair.question,
                reference_answer=qa_pair.answer,
                predicted_answer=query_response.answer,
                model_name=request.judge_model_name,
            )
            page_number = qa_pair.page_number or 1
            result = EvaluationExampleResult(
                document_id=document.document_id,
                title=document.title,
                page_number=page_number,
                question=qa_pair.question,
                reference_answer=qa_pair.answer,
                predicted_answer=query_response.answer,
                evidence_page_numbers=[item.page_number for item in query_response.evidences],
                correct=judgment["correct"],
                accurate=judgment["accurate"],
                factual=judgment["factual"],
                raw_score=int(judgment["raw_score"]),
                normalized_score=float(judgment["normalized_score"]),
                rationale=str(judgment["rationale"]),
                judge_model_name=str(judgment["judge_model_name"]),
            )
            results.append(result)
            page_buckets[(document.document_id, page_number)].append(result)

        page_summaries: list[EvaluationPageSummary] = []
        for (document_id, page_number), bucket in page_buckets.items():
            mean_raw = sum(item.raw_score for item in bucket) / len(bucket)
            mean_normalized = sum(item.normalized_score for item in bucket) / len(bucket)
            title = bucket[0].title
            page_summaries.append(
                EvaluationPageSummary(
                    document_id=document_id,
                    title=title,
                    page_number=page_number,
                    question_count=len(bucket),
                    mean_raw_score=mean_raw,
                    mean_normalized_score=mean_normalized,
                )
            )
        page_summaries.sort(key=lambda item: (item.document_id, item.page_number))

        example_mean_raw = sum(item.raw_score for item in results) / len(results)
        example_mean_normalized = sum(item.normalized_score for item in results) / len(results)
        page_balanced_normalized = (
            sum(item.mean_normalized_score for item in page_summaries) / len(page_summaries)
            if page_summaries else 0.0
        )

        total_weight = request.question_weight + request.page_weight
        if total_weight == 0:
            normalized_question_weight = 0.5
            normalized_page_weight = 0.5
        else:
            normalized_question_weight = request.question_weight / total_weight
            normalized_page_weight = request.page_weight / total_weight

        weighted_normalized_score = (
            normalized_question_weight * example_mean_normalized
            + normalized_page_weight * page_balanced_normalized
        )

        aggregate = EvaluationAggregate(
            example_count=len(results),
            page_count=len(page_summaries),
            mean_raw_score=example_mean_raw,
            mean_normalized_score=example_mean_normalized,
            page_balanced_normalized_score=page_balanced_normalized,
            weighted_normalized_score=weighted_normalized_score,
            question_weight=normalized_question_weight,
            page_weight=normalized_page_weight,
            page_summaries=page_summaries,
        )

        report_payload = {
            "aggregate": aggregate.model_dump(mode="json"),
            "examples": [item.model_dump(mode="json") for item in results],
            "request": request.model_dump(mode="json"),
        }
        run_id, report_path = self._save_report(report_payload)
        return EvaluationResponse(
            run_id=run_id,
            report_path=str(report_path),
            processed_examples=len(results),
            aggregate=aggregate,
            examples=results,
        )
