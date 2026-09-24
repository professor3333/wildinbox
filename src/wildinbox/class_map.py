"""The single source of truth for class names and their model output order.

Training, evaluation, inference, and the decision policy all build a ClassMap
from the run configuration. No other module may hard-code class names or
indices.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

EMPTY_CLASS = "empty"


class ClassMap:
    """Ordered, immutable mapping between class names and output indices."""

    def __init__(self, classes: Sequence[str]) -> None:
        names = tuple(classes)
        if len(set(names)) != len(names):
            raise ValueError(f"class names must be unique, got {list(names)}")
        if EMPTY_CLASS not in names:
            raise ValueError(f"classes must include {EMPTY_CLASS!r}, got {list(names)}")
        self._names = names
        self._index = {name: i for i, name in enumerate(names)}

    @property
    def names(self) -> tuple[str, ...]:
        return self._names

    @property
    def species(self) -> tuple[str, ...]:
        """Supported species, i.e. every class except empty, in output order."""
        return tuple(n for n in self._names if n != EMPTY_CLASS)

    @property
    def empty_index(self) -> int:
        return self._index[EMPTY_CLASS]

    def __len__(self) -> int:
        return len(self._names)

    def __contains__(self, name: object) -> bool:
        return name in self._index

    def index_of(self, name: str) -> int:
        try:
            return self._index[name]
        except KeyError:
            raise KeyError(
                f"{name!r} is not a supported class; supported: {list(self._names)}"
            ) from None

    def name_of(self, index: int) -> str:
        if not 0 <= index < len(self._names):
            raise IndexError(f"class index {index} out of range 0..{len(self._names) - 1}")
        return self._names[index]

    def fingerprint(self) -> str:
        """Short stable hash of names *and* order; changes if either changes."""
        payload = json.dumps(list(self._names)).encode()
        return hashlib.sha256(payload).hexdigest()[:12]

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ClassMap) and self._names == other._names

    def __hash__(self) -> int:
        return hash(self._names)

    def __repr__(self) -> str:
        return f"ClassMap({list(self._names)!r})"
