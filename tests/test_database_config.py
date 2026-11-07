from pathlib import Path

import pytest

import app.dependencies as dependencies
from app.database import create_database


def test_docker_compose_allows_preview_only_without_groq_key():
    compose = Path(__file__).parents[1].joinpath("docker-compose.yml").read_text(
        encoding="utf-8"
    )
    assert "GROQ_API_KEY: ${GROQ_API_KEY:-}" in compose


def test_database_configuration_rejects_sqlite(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///not-postgres.db")
    with pytest.raises(RuntimeError, match="PostgreSQL"):
        create_database()


def test_request_database_dependency_closes_injected_session(monkeypatch):
    class FakeSession:
        closed = False

        def close(self):
            self.closed = True

    session = FakeSession()
    monkeypatch.setattr(dependencies, "_session_factory", lambda: session)
    dependency = dependencies.get_db()
    assert next(dependency) is session
    dependency.close()
    assert session.closed
