"""The one device choice an evaluation command runs every model on."""

from __future__ import annotations

DEVICES = ("auto", "cpu", "mps", "cuda")


def resolve_device(choice: str) -> str:
    """`auto` picks CUDA, then MPS, then CPU. An explicit device that is not
    available is an error, never a silent fallback, so a report can say where
    its numbers came from."""
    import torch

    available = {
        "cpu": True,
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
    }
    if choice == "auto":
        return next(d for d in ("cuda", "mps", "cpu") if available[d])
    if choice not in available:
        raise ValueError(f"unknown device {choice!r}; choose one of {', '.join(DEVICES)}")
    if not available[choice]:
        raise ValueError(f"device {choice!r} is not available on this machine")
    return choice
