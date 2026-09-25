"""Timed review study endpoints (configs/study/review_study.yaml).

Study choices are logged here and never become production reviews, so
participants who label the same events do not overwrite each other.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import datetime
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session, sessionmaker

from wildinbox.storage.models import StudyParticipant, StudyPlan, StudyRating, StudyTrial
from wildinbox.study.design import ARMS, CONDITIONS, assignment


class PlanIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sets: dict[str, list[uuid.UUID]]
    truth: dict[uuid.UUID, str | None]


class JoinIn(BaseModel):
    code: str = Field(min_length=1, max_length=50, pattern=r"^[A-Za-z0-9_-]+$")


class TrialIn(BaseModel):
    participant: str = Field(min_length=1, max_length=50)
    block: int = Field(ge=0, le=2)
    condition: str
    event_id: uuid.UUID
    label: str | None = Field(default=None, max_length=100)
    seconds: float = Field(ge=0)
    interactions: int = Field(ge=0)
    shown_at: datetime
    decided_at: datetime


class RatingIn(BaseModel):
    participant: str = Field(min_length=1, max_length=50)
    block: int = Field(ge=1, le=2)
    condition: str
    difficulty: int = Field(ge=1, le=5)


def add_study_routes(
    app: FastAPI,
    sessions: Callable[[], sessionmaker[Session]],
    classes: list[str],
    error: Callable[[int, str, str], Exception],
) -> None:
    def plan_or_404(s: Session, plan_id: uuid.UUID) -> StudyPlan:
        plan = s.get(StudyPlan, plan_id)
        if plan is None:
            raise error(404, "not_found", f"study plan {plan_id} not found")
        return plan

    def participant_or_404(s: Session, plan_id: uuid.UUID, code: str) -> StudyParticipant:
        p = s.scalar(
            select(StudyParticipant).where(
                StudyParticipant.plan_id == plan_id, StudyParticipant.code == code
            )
        )
        if p is None:
            raise error(404, "not_joined", f"participant {code} has not joined this study")
        return p

    @app.post("/study/plans", status_code=201)
    def create_plan(body: PlanIn) -> dict[str, Any]:
        if set(body.sets) != {"A", "B", "practice"}:
            raise error(422, "invalid_plan", "sets must be exactly A, B, and practice")
        ids = [e for v in body.sets.values() for e in v]
        if len(ids) != len(set(ids)):
            raise error(422, "invalid_plan", "an event appears in more than one set")
        if missing := [str(e) for e in ids if e not in body.truth]:
            raise error(422, "invalid_plan", f"no ground truth for {missing[:3]}")
        with sessions()() as s:
            plan = StudyPlan(
                name=body.name,
                protocol_sha256=body.protocol_sha256,
                sets={k: [str(e) for e in v] for k, v in body.sets.items()},
                truth={str(k): v for k, v in body.truth.items()},
            )
            s.add(plan)
            s.commit()
            return {"id": str(plan.id)}

    @app.get("/study/plans/{plan_id}")
    def get_plan(plan_id: uuid.UUID) -> dict[str, Any]:
        with sessions()() as s:
            plan = plan_or_404(s, plan_id)
            joined = s.scalar(
                select(func.count())
                .select_from(StudyParticipant)
                .where(StudyParticipant.plan_id == plan_id)
            )
            return {
                "id": str(plan.id),
                "name": plan.name,
                "sets": {k: len(v) for k, v in plan.sets.items()},
                "participants": joined,
            }

    @app.post("/study/plans/{plan_id}/participants")
    def join(plan_id: uuid.UUID, body: JoinIn) -> dict[str, Any]:
        """Join (or re-join) a study; the arm follows the order of joining."""
        with sessions()() as s:
            plan = plan_or_404(s, plan_id)
            p = s.scalar(
                select(StudyParticipant).where(
                    StudyParticipant.plan_id == plan_id, StudyParticipant.code == body.code
                )
            )
            if p is None:
                s.execute(select(StudyPlan).where(StudyPlan.id == plan_id).with_for_update())
                n = (
                    s.scalar(
                        select(func.count())
                        .select_from(StudyParticipant)
                        .where(StudyParticipant.plan_id == plan_id)
                    )
                    or 0
                )
                p = StudyParticipant(plan_id=plan_id, code=body.code, arm=n % len(ARMS))
                s.add(p)
                s.commit()
            return {"participant": p.code, "classes": classes, **assignment(plan.sets, p.arm)}

    def expected(plan: StudyPlan, arm: int, block: int, condition: str, event_id: str) -> bool:
        a = assignment(plan.sets, arm)
        if block == 0:
            return any(
                x["condition"] == condition and event_id in x["events"] for x in a["practice"]
            )
        b = a["blocks"][block - 1]
        return b["condition"] == condition and event_id in b["events"]

    @app.post("/study/plans/{plan_id}/trials", status_code=201)
    def log_trial(plan_id: uuid.UUID, body: TrialIn) -> Any:
        if body.condition not in CONDITIONS:
            raise error(422, "invalid_trial", f"condition must be one of {CONDITIONS}")
        with sessions()() as s:
            plan = plan_or_404(s, plan_id)
            p = participant_or_404(s, plan_id, body.participant)
            if not expected(plan, p.arm, body.block, body.condition, str(body.event_id)):
                raise error(
                    422, "invalid_trial", "this event is not in that block for this participant"
                )
            inserted = s.execute(
                insert(StudyTrial)
                .values(plan_id=plan_id, **body.model_dump())
                .on_conflict_do_nothing(index_elements=["plan_id", "participant", "event_id"])
                .returning(StudyTrial.id)
            ).first()
            s.commit()
            if inserted is None:  # already logged: a retried request
                return JSONResponse({"duplicate": True}, status_code=200)
            return {"logged": True}

    @app.post("/study/plans/{plan_id}/ratings", status_code=201)
    def rate(plan_id: uuid.UUID, body: RatingIn) -> dict[str, Any]:
        with sessions()() as s:
            plan_or_404(s, plan_id)
            participant_or_404(s, plan_id, body.participant)
            s.execute(
                insert(StudyRating)
                .values(plan_id=plan_id, **body.model_dump())
                .on_conflict_do_update(
                    index_elements=["plan_id", "participant", "block"],
                    set_={"difficulty": body.difficulty},
                )
            )
            s.commit()
            return {"rated": True}

    @app.get("/study/plans/{plan_id}/export")
    def export(plan_id: uuid.UUID) -> dict[str, Any]:
        """Everything the analysis needs, including ground truth (organizer only)."""
        with sessions()() as s:
            plan = plan_or_404(s, plan_id)
            participants = s.scalars(
                select(StudyParticipant).where(StudyParticipant.plan_id == plan_id)
            ).all()
            trials = s.scalars(
                select(StudyTrial).where(StudyTrial.plan_id == plan_id).order_by(StudyTrial.id)
            ).all()
            ratings = s.scalars(select(StudyRating).where(StudyRating.plan_id == plan_id)).all()
            return {
                "plan": {
                    "id": str(plan.id),
                    "name": plan.name,
                    "protocol_sha256": plan.protocol_sha256,
                    "sets": plan.sets,
                    "truth": plan.truth,
                    "classes": classes,
                },
                "participants": [
                    {"code": p.code, "arm": p.arm, "joined_at": p.joined_at.isoformat()}
                    for p in participants
                ],
                "trials": [
                    {
                        "participant": t.participant,
                        "block": t.block,
                        "condition": t.condition,
                        "event_id": str(t.event_id),
                        "label": t.label,
                        "seconds": t.seconds,
                        "interactions": t.interactions,
                        "shown_at": t.shown_at.isoformat(),
                        "decided_at": t.decided_at.isoformat(),
                    }
                    for t in trials
                ],
                "ratings": [
                    {
                        "participant": r.participant,
                        "block": r.block,
                        "condition": r.condition,
                        "difficulty": r.difficulty,
                    }
                    for r in ratings
                ],
            }
