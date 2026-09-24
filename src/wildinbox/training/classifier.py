"""Linear classifier on standardized embeddings.

Fitted with scikit-learn, stored as plain arrays (no pickle), and applied with
numpy, so the artifact is safe to load and cheap to serve.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wildinbox.training.spec import ClassifierSpec


@dataclass(frozen=True)
class LinearClassifier:
    classes: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray  # (n_classes, dim), rows in `classes` order
    intercept: np.ndarray  # (n_classes,)

    def logits(self, x: np.ndarray) -> np.ndarray:
        z: np.ndarray = ((x - self.mean) / self.scale) @ self.coef.T + self.intercept
        return z

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        z = self.logits(x)
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        p: np.ndarray = e / e.sum(axis=1, keepdims=True)
        return p

    def save(self, path: Path) -> None:
        np.savez(
            path,
            classes=np.array(self.classes),
            mean=self.mean,
            scale=self.scale,
            coef=self.coef,
            intercept=self.intercept,
        )

    @classmethod
    def load(cls, path: Path) -> LinearClassifier:
        with np.load(path, allow_pickle=False) as z:
            return cls(
                tuple(z["classes"].tolist()), z["mean"], z["scale"], z["coef"], z["intercept"]
            )


def fit(
    x: np.ndarray, labels: list[str], classes: list[str], spec: ClassifierSpec, seed: int
) -> LinearClassifier:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    missing = sorted(set(classes) - set(labels))
    if missing:
        raise ValueError(f"no training examples for classes {missing}")
    unknown = sorted(set(labels) - set(classes))
    if unknown:
        raise ValueError(f"training labels outside the class list: {unknown}")
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(
        C=spec.c, class_weight=spec.class_weight, max_iter=spec.max_iter, random_state=seed
    )
    model.fit(scaler.transform(x), labels)
    order = [list(model.classes_).index(c) for c in classes]  # sklearn sorts; restore config order
    return LinearClassifier(
        classes=tuple(classes),
        mean=scaler.mean_.astype(np.float32),
        scale=scaler.scale_.astype(np.float32),
        coef=model.coef_[order].astype(np.float32),
        intercept=model.intercept_[order].astype(np.float32),
    )
