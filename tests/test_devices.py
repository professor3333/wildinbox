"""One explicit device reaches every evaluation stage, and CPU scoring works end
to end: weights verified, caches keyed by device, identical on a re-run."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from wildinbox import cli
from wildinbox.config import load_config
from wildinbox.datasets.spec import Partition
from wildinbox.devices import resolve_device
from wildinbox.evaluation.data import ImageRow

from .conftest import EXAMPLE_CONFIG


def test_auto_prefers_cuda_then_mps_then_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("auto") == "mps"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == "cuda"


def test_an_unavailable_device_is_an_error_not_a_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device("cpu") == "cpu"
    for device in ("mps", "cuda"):
        with pytest.raises(ValueError, match="not available"):
            resolve_device(device)


class _Stop(Exception):
    pass


COMMANDS = {
    "evaluate": ("wildinbox.evaluation.run", "evaluate"),
    "calibrate": ("wildinbox.evaluation.calibration", "run"),
    "unfamiliar": ("wildinbox.evaluation.unfamiliar", "run"),
    "final-test": ("wildinbox.evaluation.final_test", "run"),
    "final-evaluation": ("wildinbox.evaluation.final_eval", "run"),
    "update gate": ("wildinbox.training.gate", "run"),
}


@pytest.mark.parametrize("command", COMMANDS)
def test_every_evaluation_command_passes_its_device_on(
    command: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    module, name = COMMANDS[command]
    seen: list[str] = []

    def fake(*_: Any, device: str, **__: Any) -> None:
        seen.append(device)
        raise _Stop

    monkeypatch.setattr(importlib.import_module(module), name, fake)
    with pytest.raises(_Stop):
        cli.main([*command.split(), "--device", "cpu"])
    assert seen == ["cpu"]


def test_final_evaluation_reruns_and_rebuilds_on_the_requested_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from wildinbox.evaluation import final_eval

    seen: dict[str, str] = {}

    def reproduce(plan: Any, config: Path, device: str) -> dict[str, Any]:
        seen["reproduction"] = device
        return {}

    def build(plan: Any, config: Path, device: str) -> Any:
        seen["records"] = device
        raise _Stop

    monkeypatch.setattr(final_eval, "load_plan", lambda p: {})
    monkeypatch.setattr(final_eval, "verify_freeze", lambda plan: {})
    monkeypatch.setattr(final_eval, "reproduce_final_test", reproduce)
    monkeypatch.setattr(final_eval, "build_records", build)
    with pytest.raises(_Stop):
        final_eval.run(tmp_path / "plan.yaml", tmp_path / "cfg.yaml", tmp_path, device="cpu")
    assert seen == {"reproduction": "cpu", "records": "cpu"}


def _model_dir(tmp_path: Path, classes: list[str]) -> Path:
    from wildinbox.training.finetune import build_model

    pre = load_config(EXAMPLE_CONFIG).preprocessing
    torch.manual_seed(0)
    model = build_model(len(classes), None)
    out = tmp_path / "model"
    out.mkdir()
    torch.save(model.state_dict(), out / "model.pt")
    (out / "meta.json").write_text(
        json.dumps({"classes": classes, "preprocessing_version": pre.fingerprint()})
    )
    return out


def test_finetuned_model_scores_on_cpu_reproducibly(tmp_path: Path) -> None:
    """A real EfficientNet-B0 on CPU: the weights digest identifies the file, a
    re-run reproduces the scores exactly (from its cache and from scratch), and
    a cache written on another device is not reused."""
    import hashlib

    from wildinbox.evaluation.predictors import FinetunedPredictor
    from wildinbox.evaluation.run import _score

    run_cfg = load_config(EXAMPLE_CONFIG)
    images = tmp_path / "images"
    images.mkdir()
    rows = []
    for i in range(4):
        arr = np.random.default_rng(i).integers(0, 256, (120, 160, 3), dtype=np.uint8)
        Image.fromarray(arr).save(images / f"{i}.jpg")
        rows.append(
            ImageRow(
                f"img-{i}", f"ev-{i}", Partition.CALIBRATION, "cam", f"{i}.jpg",
                None, None, "empty", False,
            )
        )  # fmt: skip
    classes = list(run_cfg.classes)
    model_dir = _model_dir(tmp_path, classes)
    ctx = SimpleNamespace(
        run=run_cfg,
        images_root=images,
        cfg=SimpleNamespace(extraction=SimpleNamespace(num_workers=0)),
    )

    cpu = FinetunedPredictor(ctx, model_dir, "cpu")  # type: ignore[arg-type]
    digest = hashlib.sha256((model_dir / "model.pt").read_bytes()).hexdigest()[:12]
    assert cpu.weights_digest == digest
    first = _score(cpu, rows)
    cache = model_dir / "eval-cache" / digest / "calibration.npz"
    with np.load(cache) as z:
        assert json.loads(str(z["identity"]))["device"] == "cpu"
    assert [s.probs for s in _score(cpu, rows)] == [s.probs for s in first]  # from the cache
    cache.unlink()
    fresh = FinetunedPredictor(ctx, model_dir, "cpu")  # type: ignore[arg-type]
    assert [s.probs for s in _score(fresh, rows)] == [s.probs for s in first]  # recomputed
    for s in first:
        assert set(s.probs) == set(classes) and abs(sum(s.probs.values()) - 1) < 1e-5

    # A cache computed on another device must not stand in for this one.
    with np.load(cache) as z:
        saved = {k: z[k] for k in z.files}
    identity = json.loads(str(saved["identity"]))
    saved["identity"] = np.array(json.dumps({**identity, "device": "mps"}, sort_keys=True))
    saved["embeddings"] = np.zeros_like(saved["embeddings"])
    np.savez_compressed(cache, **saved)
    assert [s.probs for s in _score(fresh, rows)] == [s.probs for s in first]
