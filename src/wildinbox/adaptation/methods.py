"""Ways to adapt the released model to one camera from its reviewed events.

Each method sees only the camera's reviewed events (ground truth standing in
for reviews) and the released model's outputs, and returns probabilities for
any of that camera's frames over its own classes. None of them retrains the
network; per-camera fine-tuning is a later, costlier candidate.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from wildinbox.adaptation.data import CameraEvent, FrameOutputs
from wildinbox.class_map import EMPTY_CLASS

# Roles whose label is one of the model's classes (datasets.events.Role values).
LABELLED_ROLES = {"supported_species", EMPTY_CLASS}
UNSUPPORTED_ROLE = "unsupported_animal"
# An extra output for animals outside the supported classes. Never accepted
# automatically: an event suggested as OTHER goes to review.
OTHER = "other_animal"


@dataclass(frozen=True)
class Adapted:
    classes: tuple[str, ...]
    score: Callable[[np.ndarray], np.ndarray]  # frame row indices -> (n, classes) probs


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    out: np.ndarray = e / e.sum(axis=1, keepdims=True)
    return out


class Method(Protocol):
    @property
    def name(self) -> str: ...

    def adapt(self, reviewed: list[CameraEvent], out: FrameOutputs) -> Adapted: ...


@dataclass(frozen=True)
class Release:
    """The released model with its calibration: no adaptation."""

    temperature: float
    name: str = "release"

    def adapt(self, reviewed: list[CameraEvent], out: FrameOutputs) -> Adapted:
        return Adapted(out.classes, lambda idx: softmax(out.log_probs[idx] / self.temperature))


@dataclass(frozen=True)
class PriorShift:
    """Reweight calibrated probabilities by the camera's class mix among its
    reviewed events (add-`alpha` smoothed). The released model was trained
    with class-balanced sampling, so its own prior is taken as uniform."""

    temperature: float
    alpha: float = 1.0
    name: str = "prior_shift"

    def adapt(self, reviewed: list[CameraEvent], out: FrameOutputs) -> Adapted:
        counts = Counter(e.label for e in reviewed if e.role in LABELLED_ROLES)
        k = len(out.classes)
        prior = np.array([counts[c] + self.alpha for c in out.classes], dtype=np.float64)
        weight = prior / prior.sum() * k

        def score(idx: np.ndarray) -> np.ndarray:
            p = softmax(out.log_probs[idx] / self.temperature) * weight
            normed: np.ndarray = p / p.sum(axis=1, keepdims=True)
            return normed

        return Adapted(out.classes, score)


def reviewed_frames(reviewed: list[CameraEvent], with_other: bool) -> tuple[list[str], list[str]]:
    """Frame ids and labels a camera's reviews provide: frames annotated like
    their event; unsupported animals as OTHER when the head has that class."""
    ids: list[str] = []
    labels: list[str] = []
    for e in reviewed:
        target: str
        if e.role in LABELLED_ROLES and e.label is not None:
            target = e.label
        elif with_other and e.role == UNSUPPORTED_ROLE:
            target = OTHER
        else:
            continue
        for i, lab in zip(e.image_ids, e.image_labels, strict=True):
            if lab is not None and lab == e.label:
                ids.append(i)
                labels.append(target)
    return ids, labels


@dataclass(frozen=True)
class CameraHead:
    """A new linear classifier on the released model's frozen features, fit on
    base frames plus the camera's reviewed frames, which carry `camera_share`
    of the total sample weight.

    - iteration 1 (`with_other=False`): base = training frames. The network
      was trained on them, so their features are nearly separable and the
      head is overconfident everywhere else.
    - `with_other=True`: base = frames from cameras the network never trained
      on, and an OTHER class from unsupported animals (in the base and among
      the camera's reviews).
    """

    base_features: np.ndarray
    base_labels: tuple[str, ...]
    with_other: bool = False
    camera_share: float = 0.5
    c: float = 0.1
    name: str = "camera_head"

    def adapt(self, reviewed: list[CameraEvent], out: FrameOutputs) -> Adapted:
        classes = out.classes + ((OTHER,) if self.with_other else ())
        index = {c: i for i, c in enumerate(classes)}
        # camera_share 0: the camera's reviews are ignored entirely (ablation)
        ids, labels = reviewed_frames(reviewed, self.with_other) if self.camera_share else ([], [])
        x, y = self.base_features, np.array([index[c] for c in self.base_labels])
        w = np.ones(len(y))
        if ids:
            x = np.vstack([x, out.features[out.rows(ids)]])
            y = np.concatenate([y, [index[c] for c in labels]])
            per_frame = self.camera_share / (1 - self.camera_share) * len(w) / len(ids)
            w = np.concatenate([w, np.full(len(ids), per_frame)])
        scaler = StandardScaler().fit(x)
        model = LogisticRegression(C=self.c, max_iter=3000).fit(
            scaler.transform(x), y, sample_weight=w
        )
        fitted = list(model.classes_)

        def score(idx: np.ndarray) -> np.ndarray:
            p = model.predict_proba(scaler.transform(out.features[idx]))
            full = np.zeros((len(idx), len(classes)))
            full[:, fitted] = p  # a class absent from the fit gets probability 0
            return full

        return Adapted(classes, score)
