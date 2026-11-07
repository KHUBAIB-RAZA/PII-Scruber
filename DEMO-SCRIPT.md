# End-to-end demo recording checklist

No screen-recording capability was available in the build environment, so this is a recording script, not a video or proof of a completed live demo. Record the following after configuring a working Groq key and PostgreSQL instance:

1. Show `docker compose up -d db`, set `DATABASE_URL` and `GROQ_API_KEY`, then run `uvicorn app.main:app --reload`; open the app at `http://localhost:8000`.
2. Select the provided TXT, PDF, DOCX, and CSV statements and submit **Scrub, save & preview only**. Show that the UI reports no provider call, then show the saved UUID, scrubbed database text, per-file headers, detectable section/page/table/header/footer markers, redaction placeholders, and retained merchants, names, dates, and amounts. Demonstrate that multiple bank sections from one DOCX remain present. Do not zoom in on or read raw originals.
3. Expand the exact planned Groq payloads for all segments. Show system/user messages, model/settings, JSON mode and the fallback body. Verify that every statement excerpt shown is scrubbed. Explain that this is a preview and nothing is sent yet; click **Analyze this saved statement with Groq** only after review, then show the evidence-cited analysis.
4. In a separate terminal, query PostgreSQL (`SELECT id, source_names, length(scrubbed_text), analysis IS NOT NULL FROM statements;`) and show that only scrubbed text is stored. Optionally query the `chat_messages` table after chat.
5. Ask a monthly-total question and show the explicit local-only limitation response (no provider call), then ask about itemized dated expenses or merchant frequency. Inspect the sanitized question/history/source chat preview, then explicitly click **Send this preview to Groq**. Show cited grounded answers; ask something absent from the source and show abstention or grounding rejection.
6. Briefly explain local extraction/redaction before provider calls, the separate PostgreSQL document/chat schema, Groq model/prompt selection, regex and OCR limitations, and an improvement: tested country-specific detectors plus deterministic transaction normalization.

To inspect an existing record, paste its ID into **Load saved preview**. This calls `GET /api/statements/{id}/groq-preview`, locally re-scrubs the legacy row before showing it, and does not call Groq. API clients can use `POST /api/statements/preview` to save and preview an upload, `GET /api/statements/{id}/groq-preview` for a stored analysis payload, or `POST /api/statements/{id}/chat-preview` for chat. Only use `POST /api/statements/{id}/analyze` or the explicit UI Analyze button after inspecting the preview. The older `POST /api/statements/analyze` endpoint still analyzes immediately for compatibility.

If analysis fails, the statement is already saved. The preview shows initial/fallback request estimates; adaptive retries may use more, but each run is capped at 64 requests. Rate-limit/server failures are not automatically retried because that could worsen quota exhaustion. Use the same ID and retry only after reviewing the safe provider diagnostic; do not re-upload unless you intend to create a separate statement record.

If constrained JSON mode fails but plain JSON recovery succeeds, show the normal evidence-validated analysis. If the single fallback is malformed or fails grounding, it remains rejected; use the saved-statement retry button after resolving the provider/model issue.

Show synthetic examples of a redacted UK sort code and a preserved valid date; note that old saved records are re-scrubbed before retry and recurrence claims require two distinct cited transaction dates.

Never display the raw statement text in terminal logs or send it to any cloud service.
