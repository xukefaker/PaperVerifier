from __future__ import annotations

import logging

logger = logging.getLogger(__name__)
_WARNED: set[tuple[str, str]] = set()


def torch_cuda_report() -> dict[str, str | bool]:
    try:
        import torch
    except Exception as exc:
        return {
            "torch_imported": False,
            "torch_version": "not installed",
            "torch_cuda_version": "",
            "cuda_available": False,
            "mps_available": False,
            "gpu_name": "",
            "error": repr(exc),
        }

    cuda_module = getattr(torch, "cuda", None)
    mps_module = getattr(getattr(torch, "backends", None), "mps", None)
    cuda_version = str(getattr(getattr(torch, "version", None), "cuda", "") or "")
    cuda_available = bool(cuda_module and cuda_module.is_available())
    mps_available = bool(mps_module and mps_module.is_available())
    gpu_name = ""
    if cuda_available:
        try:
            gpu_name = str(cuda_module.get_device_name(0))
        except Exception:
            gpu_name = "CUDA device 0"
    elif mps_available:
        gpu_name = "Apple MPS"
    return {
        "torch_imported": True,
        "torch_version": str(getattr(torch, "__version__", "unknown")),
        "torch_cuda_version": cuda_version,
        "cuda_available": cuda_available,
        "mps_available": mps_available,
        "gpu_name": gpu_name,
        "error": "",
    }


def resolve_torch_device(device: str | None, *, purpose: str) -> str:
    requested = (device or "auto").strip().lower()
    report = torch_cuda_report()
    if requested in {"", "auto", "accelerated"}:
        if report["cuda_available"]:
            return "cuda:0"
        if report.get("mps_available"):
            return "mps"
        _warn_cpu(purpose, "No CUDA or Apple MPS backend is available to PyTorch.")
        return "cpu"

    if requested.startswith("cuda"):
        if report["cuda_available"]:
            return requested if ":" in requested else "cuda:0"
        _warn_cpu(
            purpose,
            f"CUDA was requested, but PyTorch cannot use CUDA "
            f"(torch_cuda={report['torch_cuda_version'] or 'none'}, cuda_available={report['cuda_available']}).",
        )
        return "cpu"

    if requested == "mps":
        if report.get("mps_available"):
            return "mps"
        _warn_cpu(purpose, "Apple MPS was requested, but PyTorch cannot use MPS.")
        return "cpu"

    if requested == "cpu":
        if not report["cuda_available"] and not report.get("mps_available"):
            _warn_cpu(purpose, "No accelerated PyTorch backend is available.")
        return "cpu"

    return requested


def resolve_mineru_device(device: str | None, *, purpose: str) -> str:
    requested = (device or "auto").strip().lower()
    report = torch_cuda_report()
    if requested in {"", "auto", "accelerated"}:
        if report["cuda_available"]:
            return "cuda"
        _warn_cpu(purpose, "No CUDA backend is available to PyTorch.")
        return "cpu"

    if requested.startswith("cuda"):
        if report["cuda_available"]:
            return "cuda"
        _warn_cpu(
            purpose,
            f"CUDA was requested, but PyTorch cannot use CUDA "
            f"(torch_cuda={report['torch_cuda_version'] or 'none'}, cuda_available={report['cuda_available']}).",
        )
        return "cpu"

    if requested == "mps":
        _warn_cpu(purpose, "MinerU parsing uses CUDA or CPU in ChemVerify; falling back from MPS to CPU.")
        return "cpu"

    return requested


def _warn_cpu(purpose: str, reason: str) -> None:
    key = (purpose, reason)
    if key in _WARNED:
        return
    _WARNED.add(key)
    logger.warning(
        "[bold yellow]Runtime[/] | cpu_fallback purpose=%s reason=%s note=CPU will work but can be slow",
        purpose,
        reason,
    )


def require_cuda_ready(device: str | None, *, purpose: str) -> None:
    resolve_torch_device(device, purpose=purpose)
