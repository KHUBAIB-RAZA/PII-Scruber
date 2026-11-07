from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .extraction import extract_text
from .redaction import (
    IBAN_CONTIGUOUS,
    IBAN_GROUPED,
    PLAIN_SORT_CODE,
    REDACTION_RULES,
    REPLACEMENTS,
    _redact_grouped_iban,
    _redact_plain_sort_code,
    contains_obvious_identifier,
    scrub_and_check,
    scrub_text,
)


def scrub_file(filename: str, data: bytes) -> str:
    return scrub_text(extract_text(filename, data))


def safe_filename(filename: str) -> str:
    return scrub_text(Path(filename.replace("\\", "/")).name)


def combine_files(files: Iterable[tuple[str, bytes]]) -> str:
    """Return one scrubbed string, with identifiers removed from body and file header."""
    sections = []
    for filename, data in files:
        safe_name = safe_filename(filename)
        sections.append(f"---- File: {safe_name} ----\n{scrub_file(filename, data)}")
    if not sections:
        raise ValueError("At least one statement file is required")
    return "\n\n".join(sections)


def scrub_folder(folder: Path) -> str:
    if not folder.is_dir():
        raise ValueError(f"Not a directory: {folder}")
    files = sorted(path for path in folder.rglob("*") if path.is_file())
    if not files:
        raise ValueError("The input folder contains no files")
    return combine_files((str(path.relative_to(folder)), path.read_bytes()) for path in files)
