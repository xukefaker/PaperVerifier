from __future__ import annotations

import sys
import types

from chemverify.config import Settings
from chemverify.devices import resolve_mineru_device, resolve_torch_device
from chemverify.encoders import EncoderConfig, SentenceTransformerEncoder
from chemverify.reranker import CrossEncoderReranker, RerankerConfig


def _install_fake_torch(
    monkeypatch,
    *,
    cuda_available: bool,
    cuda_version: str | None = "12.1",
    mps_available: bool = False,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(
            __version__="test-torch",
            version=types.SimpleNamespace(cuda=cuda_version),
            cuda=types.SimpleNamespace(is_available=lambda: cuda_available),
            backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps_available)),
        ),
    )


def test_sentence_transformer_encoder_prefers_cuda_when_available(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class _FakeSentenceTransformer:
        def __init__(self, model_name: str, **kwargs) -> None:
            calls.append({"model_name": model_name, "kwargs": kwargs})

    _install_fake_torch(monkeypatch, cuda_available=True)
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=_FakeSentenceTransformer),
    )

    encoder = SentenceTransformerEncoder(EncoderConfig("test-model"))

    assert encoder.device == "cuda:0"
    assert calls == [{"model_name": "test-model", "kwargs": {"device": "cuda:0"}}]


def test_sentence_transformer_encoder_falls_back_to_cpu(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class _FakeSentenceTransformer:
        def __init__(self, model_name: str, **kwargs) -> None:
            calls.append({"model_name": model_name, "kwargs": kwargs})

    _install_fake_torch(monkeypatch, cuda_available=False)
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=_FakeSentenceTransformer),
    )

    encoder = SentenceTransformerEncoder(EncoderConfig("test-model"))

    assert encoder.device == "cpu"
    assert calls == [{"model_name": "test-model", "kwargs": {"device": "cpu"}}]


def test_cross_encoder_reranker_prefers_cuda_when_available(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class _FakeCrossEncoder:
        def __init__(self, model_name: str, **kwargs) -> None:
            calls.append({"model_name": model_name, "kwargs": kwargs})

    _install_fake_torch(monkeypatch, cuda_available=True)
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(CrossEncoder=_FakeCrossEncoder),
    )

    reranker = CrossEncoderReranker(RerankerConfig("test-reranker"))

    assert reranker.device == "cuda:0"
    assert calls == [{"model_name": "test-reranker", "kwargs": {"trust_remote_code": True, "device": "cuda:0"}}]


def test_settings_device_env_sets_all_runtime_devices(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CHEMVERIFY_DEVICE", "cuda")
    monkeypatch.delenv("CHEMVERIFY_MINERU_DEVICE", raising=False)
    monkeypatch.delenv("CHEMVERIFY_DENSE_DEVICE", raising=False)
    monkeypatch.delenv("CHEMVERIFY_RERANKER_DEVICE", raising=False)

    settings = Settings.from_env(tmp_path)

    assert settings.mineru_device == "cuda"
    assert settings.dense_device == "cuda"
    assert settings.reranker_device == "cuda"


def test_cuda_request_falls_back_to_cpu_when_torch_has_no_cuda(monkeypatch) -> None:
    _install_fake_torch(monkeypatch, cuda_available=False, cuda_version=None)

    assert resolve_torch_device("cuda", purpose="Dense retrieval") == "cpu"
    assert resolve_mineru_device("cuda", purpose="MinerU PDF parsing") == "cpu"


def test_auto_device_uses_mps_for_torch_but_cpu_for_mineru(monkeypatch) -> None:
    _install_fake_torch(monkeypatch, cuda_available=False, cuda_version=None, mps_available=True)

    assert resolve_torch_device("auto", purpose="Dense retrieval") == "mps"
    assert resolve_mineru_device("auto", purpose="MinerU PDF parsing") == "cpu"
