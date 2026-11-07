from __future__ import annotations

from collections.abc import Callable, Generator

from fastapi import HTTPException
from sqlalchemy.orm import Session

SessionFactory = Callable[[], Session]
_session_factory: SessionFactory | None = None


def configure_session_factory(factory: SessionFactory) -> None:
    global _session_factory
    _session_factory = factory


def get_db() -> Generator[Session, None, None]:
    if _session_factory is None:
        raise HTTPException(503, "Database session factory is not initialized")
    session = _session_factory()
    try:
        yield session
    finally:
        session.close()
