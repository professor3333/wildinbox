"""Engine and session helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker


@lru_cache(maxsize=8)
def engine_for(url: str) -> Engine:
    return create_engine(url, pool_pre_ping=True)


def session_factory(url: str) -> sessionmaker[Session]:
    return sessionmaker(engine_for(url), expire_on_commit=False)


@contextmanager
def session_scope(url: str) -> Iterator[Session]:
    """A session that commits on success and rolls back on error."""
    session = session_factory(url)()
    try:
        yield session
        session.commit()
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()
