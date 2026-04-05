from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from agent_rag.config import (
    DEFAULT_RERANKER_MODEL_NAME,
    DEFAULT_RERANKER_PROCESSOR_NAME,
)
from agent_rag.data.schemas import EvidencePage


class MonoVLMReranker:
    """
    Pointwise multimodal reranker using MonoQwen2-VL.

    The model is trained with a MonoT5-style objective:
    it predicts whether a page image is relevant to the query by generating
    "True" or "False". We turn those logits into a relevance score.
    """

    def __init__(
        self,
        default_model_name: str = DEFAULT_RERANKER_MODEL_NAME,
        processor_name: str = DEFAULT_RERANKER_PROCESSOR_NAME,
    ) -> None:
        self.default_model_name = default_model_name
        self.processor_name = processor_name
        self._loaded_model_name: str | None = None
        self._processor: Any = None
        self._model: Any = None
        self._device: str | None = None
        self._true_token_id: int | None = None
        self._false_token_id: int | None = None

    def _resolve_device(self) -> str:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
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

        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self._device = self._resolve_device()
        self._processor = AutoProcessor.from_pretrained(self.processor_name)
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=self._resolve_dtype(),
        )
        self._model.to(self._device)
        self._model.eval()
        self._loaded_model_name = model_name

        true_ids = self._processor.tokenizer.encode("True", add_special_tokens=False)
        false_ids = self._processor.tokenizer.encode("False", add_special_tokens=False)
        if not true_ids or not false_ids:
            raise RuntimeError("Failed to resolve True/False tokens for reranker scoring.")
        self._true_token_id = true_ids[0]
        self._false_token_id = false_ids[0]

    def _score_single(self, question: str, image_path: str, model_name: str) -> float:
        import torch

        self._ensure_model(model_name)

        prompt = (
            "Assert the relevance of the previous image document to the following query, "
            "answer True or False. The query is: {query}"
        ).format(query=question)

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
                return_tensors="pt",
            )
            inputs = inputs.to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits_for_last_token = outputs.logits[:, -1, :]
            relevance_score = torch.softmax(
                logits_for_last_token[:, [self._true_token_id, self._false_token_id]],
                dim=-1,
            )
        return float(relevance_score[0, 0].item())

    def rerank(
        self,
        question: str,
        evidences: list[EvidencePage],
        top_k: int,
        model_name: str | None = None,
    ) -> tuple[list[EvidencePage], dict[str, object]]:
        if not evidences:
            return [], {
                "reranker": "monoqwen2_vl",
                "reranker_status": "no_candidates",
            }

        resolved_model_name = model_name or self.default_model_name
        rescored: list[EvidencePage] = []

        for evidence in evidences:
            if not evidence.image_path or not Path(evidence.image_path).exists():
                rescored.append(
                    evidence.model_copy(
                        update={
                            "rerank_score": None,
                            "rationale": "Reranker skipped because image_path is missing.",
                        }
                    )
                )
                continue

            rerank_score = self._score_single(
                question=question,
                image_path=evidence.image_path,
                model_name=resolved_model_name,
            )
            rescored.append(
                evidence.model_copy(
                    update={
                        "rerank_score": rerank_score,
                        "score": rerank_score,
                        "rationale": "MonoQwen2-VL pointwise reranking over retrieved page images.",
                    }
                )
            )

        rescored.sort(
            key=lambda item: item.rerank_score if item.rerank_score is not None else -1.0,
            reverse=True,
        )
        return rescored[:top_k], {
            "reranker": "monoqwen2_vl",
            "reranker_model_name": resolved_model_name,
            "reranker_device": self._device or self._resolve_device(),
            "reranker_status": "applied",
        }
