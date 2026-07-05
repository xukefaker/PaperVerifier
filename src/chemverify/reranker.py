from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .devices import resolve_torch_device


def _resolve_device(device: str | None) -> str:
    return resolve_torch_device(device, purpose="Reranking")


class BaseReranker:
    backend_name = "unknown"

    @property
    def model_name(self) -> str:
        raise NotImplementedError

    def score_pairs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        raise NotImplementedError


@dataclass(slots=True)
class RerankerConfig:
    model_name: str
    device: str | None = None
    batch_size: int = 8


class CrossEncoderReranker(BaseReranker):
    def __init__(self, config: RerankerConfig) -> None:
        self.config = config
        try:
            from sentence_transformers import CrossEncoder
        except Exception as exc:
            raise RuntimeError("sentence-transformers is required for reranking.") from exc

        resolved_device = _resolve_device(config.device)
        init_kwargs = {
            "trust_remote_code": True,
            "device": resolved_device,
        }
        try:
            self._model = CrossEncoder(config.model_name, **init_kwargs)
        except Exception as exc:
            raise RuntimeError(f"Failed to load cross-encoder model '{config.model_name}'.") from exc
        self.backend_name = f"cross-encoder:{config.model_name}"
        self.device = resolved_device

    @property
    def model_name(self) -> str:
        return self.config.model_name

    def score_pairs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        if not pairs:
            return np.zeros(0, dtype=np.float32)
        scores = self._model.predict(
            pairs,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(scores, dtype=np.float32).reshape(-1)
