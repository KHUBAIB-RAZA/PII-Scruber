from __future__ import annotations

from fastapi import HTTPException, UploadFile

from .scrubber import combine_files, safe_filename, scrub_and_check

MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_TOTAL_BYTES = 40 * 1024 * 1024


def scrub_uploads(files: list[UploadFile]) -> tuple[str, str]:
    """Validate bounded uploads, extract and scrub locally, returning safe name/text."""
    if not files:
        raise HTTPException(400, "Upload at least one statement file")
    payloads = []
    total = 0
    for upload in files:
        data = upload.file.read(MAX_FILE_BYTES + 1)
        if len(data) > MAX_FILE_BYTES:
            raise HTTPException(413, "Each file must be 10 MB or smaller")
        total += len(data)
        if total > MAX_TOTAL_BYTES:
            raise HTTPException(413, "Combined upload must be 40 MB or smaller")
        payloads.append((upload.filename or "statement.txt", data))
    scrubbed = scrub_and_check(combine_files(payloads))
    source_names = ", ".join(safe_filename(name) for name, _ in payloads)
    return source_names, scrubbed
