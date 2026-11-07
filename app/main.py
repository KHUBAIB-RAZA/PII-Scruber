from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from .database import Statement, create_database
from .dependencies import configure_session_factory, get_db
from .llm import (
    ANALYSIS_INCOMPLETE_MARKER,
    AnalysisIncomplete,
    ProviderRequestError,
    StatementTooLarge,
    UngroundedResponse,
    analysis_segment_statuses,
    analyze,
    chat,
    estimate_provider_requests,
    local_chat_explanation,
    preview_analysis_payloads,
    preview_chat_payloads,
)
from .repository import (
    get_chat_history as repository_get_chat_history,
    get_statement as repository_get_statement,
    read_chat_history,
    save_analysis,
    save_chat_turn,
    save_statement,
)
from .scrubber import scrub_and_check, scrub_text
from .uploads import scrub_uploads


def _analysis_status(analysis: str | None) -> str:
    if not analysis:
        return "not_run"
    if analysis.startswith(ANALYSIS_INCOMPLETE_MARKER):
        return "incomplete"
    return "complete"


def _sanitize_stored_statement(statement: Statement) -> bool:
    """Re-scrub legacy records before they can be returned or sent to a provider."""
    sanitized = scrub_and_check(statement.scrubbed_text)
    changed = sanitized != statement.scrubbed_text
    if changed:
        statement.scrubbed_text = sanitized
        statement.analysis = None
    if statement.analysis:
        statement.analysis = scrub_and_check(statement.analysis)
    for message in statement.messages:
        safe_content = scrub_and_check(message.content)
        if safe_content != message.content:
            message.content = safe_content
    return changed


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_session_factory(create_database())
    yield


app = FastAPI(title="Local Financial Statement Scrubber", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
def home():
    return FileResponse("app/static/index.html")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/statements/analyze")
def upload_and_analyze(
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    try:
        source_names, scrubbed = scrub_uploads(files)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    record = save_statement(db, source_names, scrubbed)
    try:
        analysis = analyze(scrubbed)
    except AnalysisIncomplete as exc:
        save_analysis(db, record, exc.analysis)
        return {
            "id": record.id,
            "files": source_names,
            "scrubbed_text": scrubbed,
            "analysis": record.analysis,
            "analysis_status": "incomplete",
            "analysis_segments": exc.segment_statuses,
            "message": str(exc),
        }
    except StatementTooLarge as exc:
        raise HTTPException(413, f"{exc} Statement ID: {record.id}.") from exc
    except UngroundedResponse as exc:
        raise HTTPException(422, f"{exc} Statement ID: {record.id}.") from exc
    except ProviderRequestError as exc:
        retry_hint = " Retry analysis from the saved statement view." if exc.retryable else ""
        raise HTTPException(
            exc.status_code, f"{exc}{retry_hint} Statement ID: {record.id}."
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(502, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, "Analysis provider request failed") from exc
    stored = repository_get_statement(db, record.id)
    save_analysis(db, stored, analysis)
    return {
        "id": record.id,
        "files": source_names,
        "scrubbed_text": scrubbed,
        "analysis": analysis,
        "analysis_status": "complete",
        "analysis_segments": analysis_segment_statuses(analysis, scrubbed),
    }


@app.post("/api/statements/preview")
def upload_and_preview(
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    """Scrub and persist uploads, then show exact analysis payloads without calling Groq."""
    try:
        source_names, scrubbed = scrub_uploads(files)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    record = Statement(source_names=source_names, scrubbed_text=scrubbed)
    try:
        prompt_payloads = preview_analysis_payloads(scrubbed)
    except StatementTooLarge as exc:
        raise HTTPException(413, str(exc)) from exc
    db.add(record)
    db.commit()
    db.refresh(record)
    _sanitize_stored_statement(record)
    db.commit()
    return {
        "id": record.id,
        "files": record.source_names,
        "scrubbed_text": record.scrubbed_text,
        "analysis": record.analysis,
        "analysis_status": _analysis_status(record.analysis),
        "analysis_segments": analysis_segment_statuses(
            record.analysis, record.scrubbed_text
        ),
        "provider_payloads": prompt_payloads,
        "provider_request_estimate": estimate_provider_requests(len(prompt_payloads)),
    }


@app.get("/api/statements/{statement_id}")
def get_statement(statement_id: str, db: Session = Depends(get_db)):
    statement = repository_get_statement(db, statement_id)
    if statement is None:
        raise HTTPException(404, "Statement not found")
    _sanitize_stored_statement(statement)
    db.commit()
    return {
        "id": statement.id,
        "files": statement.source_names,
        "scrubbed_text": statement.scrubbed_text,
        "analysis": statement.analysis,
        "analysis_status": _analysis_status(statement.analysis),
        "analysis_segments": analysis_segment_statuses(
            statement.analysis, statement.scrubbed_text
        ),
    }


@app.get("/api/statements/{statement_id}/groq-preview")
def get_groq_preview(statement_id: str, db: Session = Depends(get_db)):
    """Inspect stored scrubbed text and exact planned analysis payload; no provider call."""
    statement = repository_get_statement(db, statement_id)
    if statement is None:
        raise HTTPException(404, "Statement not found")
    _sanitize_stored_statement(statement)
    db.commit()
    try:
        prompt_payloads = preview_analysis_payloads(statement.scrubbed_text)
    except StatementTooLarge as exc:
        raise HTTPException(413, str(exc)) from exc
    return {
        "id": statement.id,
        "files": statement.source_names,
        "scrubbed_text": statement.scrubbed_text,
        "analysis": statement.analysis,
        "analysis_status": _analysis_status(statement.analysis),
        "analysis_segments": analysis_segment_statuses(
            statement.analysis, statement.scrubbed_text
        ),
        "provider_payloads": prompt_payloads,
        "provider_request_estimate": estimate_provider_requests(len(prompt_payloads)),
    }


@app.post("/api/statements/{statement_id}/analyze")
def retry_analysis(statement_id: str, db: Session = Depends(get_db)):
    statement = repository_get_statement(db, statement_id)
    if statement is None:
        raise HTTPException(404, "Statement not found")
    _sanitize_stored_statement(statement)
    db.commit()
    try:
        analysis = analyze(statement.scrubbed_text)
    except AnalysisIncomplete as exc:
        save_analysis(db, statement, exc.analysis)
        return {
            "id": statement.id,
            "files": statement.source_names,
            "scrubbed_text": statement.scrubbed_text,
            "analysis": statement.analysis,
            "analysis_status": "incomplete",
            "analysis_segments": exc.segment_statuses,
            "message": str(exc),
        }
    except StatementTooLarge as exc:
        raise HTTPException(413, str(exc)) from exc
    except UngroundedResponse as exc:
        raise HTTPException(422, str(exc)) from exc
    except ProviderRequestError as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(502, str(exc)) from exc
    save_analysis(db, statement, analysis)
    return {
        "id": statement.id,
        "files": statement.source_names,
        "scrubbed_text": statement.scrubbed_text,
        "analysis": statement.analysis,
        "analysis_status": "complete",
        "analysis_segments": analysis_segment_statuses(
            statement.analysis, statement.scrubbed_text
        ),
    }


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


@app.post("/api/statements/{statement_id}/chat-preview")
def preview_chat_request(
    statement_id: str,
    request: ChatRequest,
    db: Session = Depends(get_db),
):
    """Preview exact scrubbed chat payloads without contacting the provider."""
    statement = repository_get_statement(db, statement_id)
    if statement is None:
        raise HTTPException(404, "Statement not found")
    _sanitize_stored_statement(statement)
    db.commit()
    safe_question = scrub_text(request.question)
    local_explanation = local_chat_explanation(safe_question)
    if local_explanation:
        return {
            "id": statement.id,
            "question": safe_question,
            "local_only": True,
            "message": local_explanation,
            "provider_payloads": [],
        }
    history = [
        {"role": message.role, "content": message.content}
        for message in repository_get_chat_history(db, statement_id)
    ]
    try:
        payloads = preview_chat_payloads(
            statement.scrubbed_text, safe_question, history
        )
    except StatementTooLarge as exc:
        raise HTTPException(413, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {
        "id": statement.id,
        "question": safe_question,
        "local_only": False,
        "provider_payloads": payloads,
    }


@app.post("/api/statements/{statement_id}/chat")
def chat_about_statement(
    statement_id: str,
    request: ChatRequest,
    db: Session = Depends(get_db),
):
    statement = repository_get_statement(db, statement_id)
    if statement is None:
        raise HTTPException(404, "Statement not found")
    _sanitize_stored_statement(statement)
    db.commit()
    history = [
        {"role": message.role, "content": message.content}
        for message in repository_get_chat_history(db, statement_id)
    ]
    safe_question = scrub_text(request.question)
    try:
        answer = chat(statement.scrubbed_text, safe_question, history)
    except StatementTooLarge as exc:
        raise HTTPException(413, str(exc)) from exc
    except UngroundedResponse as exc:
        raise HTTPException(422, str(exc)) from exc
    except ProviderRequestError as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, "Chat provider request failed") from exc
    save_chat_turn(db, statement_id, safe_question, answer)
    return {"answer": answer, "local_only": local_chat_explanation(safe_question) is not None}


@app.get("/api/statements/{statement_id}/chat")
def get_chat_history(statement_id: str, db: Session = Depends(get_db)):
    statement = repository_get_statement(db, statement_id)
    if statement is None:
        raise HTTPException(404, "Statement not found")
    _sanitize_stored_statement(statement)
    db.commit()
    messages = read_chat_history(db, statement_id)
    return {"messages": [{"role": message.role, "content": message.content} for message in messages]}
