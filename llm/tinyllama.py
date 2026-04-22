"""TinyLlama local inference wrapper."""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional

import config as cfg


@dataclass
class LLMResponse:
    answer: str
    tokens_input: int
    tokens_output: int
    latency: float   # seconds


_MODEL_CACHE: dict[str, tuple] = {}  # model_name → (model, tokenizer)


def _load_model(model_name: str):
    """Lazy-load and cache the model + tokenizer."""
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.backends.mps.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()
    _MODEL_CACHE[model_name] = (model, tokenizer)
    return model, tokenizer


def _build_prompt(context: str, query: str) -> str:
    return (
        "<|system|>\n"
        "You are a helpful assistant. Answer the question based ONLY on the provided context.\n"
        "</s>\n"
        "<|user|>\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "</s>\n"
        "<|assistant|>\n"
    )


class TinyLlamaModel:
    """Wrapper around TinyLlama for single-call inference."""

    def __init__(self, model_name: str = cfg.TINYLLAMA_MODEL):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self):
        if self._model is None:
            self._model, self._tokenizer = _load_model(self.model_name)

    def generate(
        self,
        context: str,
        query: str,
        max_new_tokens: int = cfg.TINYLLAMA_MAX_NEW_TOKENS,
        temperature: float = 0.1,
    ) -> LLMResponse:
        """Run inference and return answer + metadata."""
        self._ensure_loaded()
        import torch

        prompt = _build_prompt(context, query)
        t0 = time.time()

        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to(self._model.device)

        n_input = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        latency = time.time() - t0
        new_ids = output_ids[0][n_input:]
        answer = self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        return LLMResponse(
            answer=answer,
            tokens_input=n_input,
            tokens_output=len(new_ids),
            latency=latency,
        )

    def chunks_to_context(self, chunks: list) -> str:
        """Join chunk texts into a single context string."""
        return "\n\n---\n\n".join(c.text for c in chunks)
