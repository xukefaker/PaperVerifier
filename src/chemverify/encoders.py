from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .devices import resolve_torch_device


def _resolve_device(device: str | None) -> str:
    return resolve_torch_device(device, purpose="Dense retrieval")


class BaseEncoder:
    backend_name = "unknown"

    @property
    def model_name(self) -> str:
        raise NotImplementedError

    def encode(
        self,
        texts: list[str],
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        raise NotImplementedError


@dataclass(slots=True)
class EncoderConfig:
    model_name: str
    device: str | None = None
    batch_size: int = 8


class SentenceTransformerEncoder(BaseEncoder):
    def __init__(self, config: EncoderConfig) -> None:
        self.config = config
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise RuntimeError("sentence-transformers is required for dense retrieval.") from exc

        resolved_device = _resolve_device(config.device)
        init_kwargs = {"device": resolved_device}
        try:
            self._model = SentenceTransformer(config.model_name, **init_kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load sentence-transformer model '{config.model_name}'."
            ) from exc
        self.backend_name = f"sentence-transformers:{config.model_name}"
        self.device = resolved_device

    @property
    def model_name(self) -> str:
        return self.config.model_name

    def encode(
        self,
        texts: list[str],
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        batch_size = max(1, int(self.config.batch_size))
        outer_batch_size = batch_size
        chunks: list[np.ndarray] = []
        total = len(texts)
        completed = 0

        for start in range(0, total, outer_batch_size):
            batch = texts[start : start + outer_batch_size]
            embeddings = self._model.encode(
                batch,
                normalize_embeddings=True,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            array = np.asarray(embeddings, dtype=np.float32)
            chunks.append(array)
            completed += len(batch)
            if progress_callback is not None:
                progress_callback(completed, total)

        return np.vstack(chunks).astype(np.float32, copy=False)
