"""Capture events and their conservative ground truth.

Rules (see docs/dataset.md):

1. An event containing any quarantined frame is excluded; its ground truth
   cannot be trusted.
2. Event ground truth uses the annotations of every non-quarantined frame,
   including exact duplicates excluded from the image set, so removing a
   duplicate image can never turn an animal event into an empty one.
3. An event is animal-containing if at least one frame contains an animal.
4. Exactly one animal species -> that species. More than one -> `mixed`,
   species unresolved (no resolution rule is applied).
5. No animal but a non-animal category (e.g. car) -> non_animal; neither an
   animal positive nor an empty negative.
6. `empty` only when every frame is explicitly annotated empty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from wildinbox.class_map import EMPTY_CLASS
from wildinbox.datasets.spec import Kind, Partition, Taxonomy
from wildinbox.ingestion.inventory import Status, normalize_label


class Role(StrEnum):
    EMPTY = EMPTY_CLASS
    SUPPORTED = "supported_species"
    UNSUPPORTED = "unsupported_animal"
    MIXED = "mixed_species"
    NON_ANIMAL = "non_animal"


class FitExclusion(StrEnum):
    NOT_TRAINING_PARTITION = "not_training_partition"
    EVENT_NOT_TRAINABLE = "event_not_supported_or_empty"
    FRAME_LABEL_DIFFERS = "frame_label_differs_from_event"


@dataclass(frozen=True)
class Frame:
    source_id: str
    status: Status
    camera_id: str
    sequence_id: str
    source_file: str
    frame_num: int | None
    captured_at: str | None
    storage_path: str | None
    sha256: str | None
    labels: tuple[str, ...]  # normalized from the original annotations


@dataclass
class Event:
    event_id: str
    camera_id: str
    sequence_id: str
    frames: list[Frame]
    partition: Partition | None = None
    excluded_reason: str | None = None
    animals: tuple[str, ...] = ()
    non_animals: tuple[str, ...] = ()
    role: Role | None = None
    label: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def animal_present(self) -> bool:
        return bool(self.animals)

    @property
    def images(self) -> list[Frame]:
        """Frames whose image is used downstream (accepted records only)."""
        return [f for f in self.frames if f.status is Status.ACCEPTED]


def frame_labels(original_labels: list[str]) -> tuple[str, ...]:
    return tuple(sorted({normalize_label(x) for x in original_labels}))


def apply_ground_truth(event: Event, taxonomy: Taxonomy) -> None:
    """Set animals / non_animals / exclusion from frame annotations (rules 1-3, 5-6)."""
    if any(f.status is Status.QUARANTINED for f in event.frames):
        event.excluded_reason = "contains_quarantined_frame"
        return
    labels = {lab for f in event.frames for lab in f.labels}
    kinds = {lab: taxonomy.kind_of(lab) for lab in labels}
    event.animals = tuple(sorted(lab for lab, k in kinds.items() if k is Kind.ANIMAL))
    event.non_animals = tuple(sorted(lab for lab, k in kinds.items() if k is Kind.NON_ANIMAL))
    if not event.images:
        event.excluded_reason = "no_usable_images"
    empty_frames = [f for f in event.frames if f.labels == (EMPTY_CLASS,)]
    if event.animals and empty_frames:
        event.notes.append(f"{len(empty_frames)} empty-annotated frame(s) in an animal event")


def assign_role(event: Event, supported: frozenset[str]) -> None:
    """Rules 3-6, once the supported species are known."""
    if event.excluded_reason:
        return
    if len(event.animals) > 1:
        event.role, event.label = Role.MIXED, None
    elif len(event.animals) == 1:
        species = event.animals[0]
        event.label = species
        event.role = Role.SUPPORTED if species in supported else Role.UNSUPPORTED
    elif event.non_animals:
        event.role, event.label = Role.NON_ANIMAL, "+".join(event.non_animals)
    else:
        event.role, event.label = Role.EMPTY, EMPTY_CLASS


def fit_exclusion(event: Event, frame: Frame) -> FitExclusion | None:
    """Why an image may not be used to fit the classifier, or None if it may."""
    if event.partition is not Partition.TRAIN:
        return FitExclusion.NOT_TRAINING_PARTITION
    if event.role not in (Role.SUPPORTED, Role.EMPTY):
        return FitExclusion.EVENT_NOT_TRAINABLE
    if frame.labels != (event.label,):
        return FitExclusion.FRAME_LABEL_DIFFERS
    return None
