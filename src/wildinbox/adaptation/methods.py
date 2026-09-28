"""Ways to adapt the released model to one camera from its reviewed events.

Each method sees only the camera's reviewed events (ground truth standing in
for reviews) and the released model's outputs, and returns calibrated-style
probabilities for any of that camera's frames. None of them retrains the
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

Scorer = Callable[[np.ndarray], np.ndarray]  # frame row indices -> (n, classes) probs


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    out: np.ndarray = e / e.sum(axis=1, keepdims=True)
    return out


class Method(Protocol):
    @property
    def name(self) -> str: ...

    def adapt(self, reviewed: list[CameraEvent], out: FrameOutputs) -> Scorer: ...


@dataclass(frozen=True)
class Release:
    """The released model with its calibration: no adaptation."""

    temperature: float
    name: str = "release"

    def adapt(self, reviewed: list[CameraEvent], out: FrameOutputs) -> Scorer:
        return lambda idx: softmax(out.log_probs[idx] / self.temperature)


@dataclass(frozen=True)
class PriorShift:
    """Reweight calibrated probabilities by the camera's class mix among its
    reviewed events (add-`alpha` smoothed). The released model was trained
    with class-balanced sampling, so its own prior is taken as uniform."""

    temperature: float
    alpha: float = 1.0
    name: str = "prior_shift"

    def adapt(self, reviewed: list[CameraEvent], out: FrameOutputs) -> Scorer:
        counts = Counter(e.label for e in reviewed if e.role in LABELLED_ROLES)
        k = len(out.classes)
        prior = np.array([counts[c] + self.alpha for c in out.classes], dtype=np.float64)
        weight = prior / prior.sum() * k

        def score(idx: np.ndarray) -> np.ndarray:
            p = softmax(out.log_probs[idx] / self.temperature) * weight
            normed: np.ndarray = p / p.sum(axis=1, keepdims=True)
            return normed

        return score


@dataclass(frozen=True)
class CameraHead:
    """A new linear classifier on the released model's frozen features, fit on
    a class-balanced sample of training frames plus the camera's reviewed
    frames, which carry `camera_share` of the total sample weight."""

    train_features: np.ndarray
    train_labels: np.ndarray  # class indices
    camera_share: float = 0.5
    c: float = 0.1
    name: str = "camera_head"

    def adapt(self, reviewed: list[CameraEvent], out: FrameOutputs) -> Scorer:
        cls = {c: i for i, c in enumerate(out.classes)}
        ids, labels = [], []
        for e in reviewed:
            if e.role not in LABELLED_ROLES:
                continue
            for i, lab in zip(e.image_ids, e.image_labels, strict=True):
                if lab is not None and lab == e.label:  # frames annotated like their event
                    ids.append(i)
                    labels.append(cls[lab])
        x_train, y_train = self.train_features, self.train_labels
        w_train = np.ones(len(y_train))
        if ids:
            x_cam = out.features[out.rows(ids)]
            x = np.vstack([x_train, x_cam])
            y = np.concatenate([y_train, labels])
            w_cam = np.full(
                len(ids), self.camera_share / (1 - self.camera_share) * len(y_train) / len(ids)
            )
            w = np.concatenate([w_train, w_cam])
        else:
            x, y, w = x_train, y_train, w_train
        scaler = StandardScaler().fit(x)
        model = LogisticRegression(C=self.c, max_iter=2000).fit(
            scaler.transform(x), y, sample_weight=w
        )
        order = np.array([list(model.classes_).index(i) for i in range(len(out.classes))])

        def score(idx: np.ndarray) -> np.ndarray:
            p: np.ndarray = model.predict_proba(scaler.transform(out.features[idx]))[:, order]
            return p

        return score
