from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from .database import ChatMessage, Statement


def save_statement(session: Session, source_names: str, scrubbed_text: str) -> Statement:
    record = Statement(source_names=source_names, scrubbed_text=scrubbed_text)
    session.add(record)
    session.commit()
    session.refresh(record)
    return record


def get_statement(session: Session, statement_id: str) -> Statement | None:
    return session.get(Statement, statement_id)


def save_analysis(session: Session, statement: Statement, analysis: str) -> None:
    statement.analysis = analysis
    session.commit()


def get_chat_history(session: Session, statement_id: str, limit: int = 10) -> list[ChatMessage]:
    return list(session.scalars(
        select(ChatMessage)
        .where(ChatMessage.statement_id == statement_id)
        .order_by(ChatMessage.created_at.desc())
        .limit(limit)
    ).all()[::-1])


def save_chat_turn(
    session: Session, statement_id: str, question: str, answer: str
) -> None:
    session.add_all([
        ChatMessage(statement_id=statement_id, role="user", content=question),
        ChatMessage(statement_id=statement_id, role="assistant", content=answer),
    ])
    session.commit()


def read_chat_history(session: Session, statement_id: str) -> list[ChatMessage]:
    return list(session.scalars(
        select(ChatMessage)
        .where(ChatMessage.statement_id == statement_id)
        .order_by(ChatMessage.created_at)
    ).all())
