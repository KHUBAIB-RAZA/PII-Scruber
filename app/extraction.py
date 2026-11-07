from __future__ import annotations

import csv
import io
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".csv", ".txt", ".log"}

SECTION_HEADING = re.compile(
    r"^\s*(?:(?:\S.{0,80}\s+)?(?:bank|account|wallet|card)\s+statement|"
    r"statement\s+(?:of\s+account|period|summary)|"
    r"(?:account|transaction|activity)\s+summary)\b.*$",
    re.I,
)


def _decode_text(data: bytes) -> str:
    return data.decode("utf-8-sig", errors="replace")


def mark_statement_sections(text: str) -> str:
    """Insert generic boundaries at recognizable statement headings without dropping text."""
    marked = []
    section_number = 0
    for line in text.splitlines():
        if SECTION_HEADING.match(line.strip()):
            section_number += 1
            marked.append(f"---- Statement section {section_number} ----")
        marked.append(line)
    return "\n".join(marked)


def _docx_paragraph_text(paragraph) -> str:
    return "".join(node.text or "" for node in paragraph.iter() if node.tag.endswith("}t"))


def _docx_block_text(container) -> list[str]:
    """Read paragraphs and nested tables in document order, including drawing text."""
    from docx.oxml.ns import qn

    def walk(parent):
        output = []
        for child in parent.iterchildren():
            if child.tag == qn("w:p"):
                value = _docx_paragraph_text(child)
                if value.strip():
                    output.append(value)
            elif child.tag == qn("w:tbl"):
                for row in child.findall(qn("w:tr")):
                    cells = []
                    for cell in row.findall(qn("w:tc")):
                        cells.append(" ".join(walk(cell)))
                    output.append(" | ".join(cells))
        return output

    return walk(container)


def _docx_auxiliary_text(data: bytes) -> list[str]:
    extracted = []
    with zipfile.ZipFile(io.BytesIO(data)) as package:
        for name in package.namelist():
            if not re.fullmatch(r"word/(?:comments|footnotes|endnotes)\.xml", name):
                continue
            root = ET.fromstring(package.read(name))
            for paragraph in root.iter():
                if not paragraph.tag.endswith("}p"):
                    continue
                value = "".join(
                    node.text or "" for node in paragraph.iter() if node.tag.endswith("}t")
                )
                if value.strip():
                    extracted.append(value)
    return extracted


def _table_rows_missing_from_text(rows: list[list[object]], page_text: str) -> list[list[object]]:
    """Keep table rows only when their cell sequence is absent from page text."""
    normalized_text = re.sub(r"\s+", " ", page_text).casefold()
    missing = []
    for row in rows:
        cells = [str(cell).strip() for cell in row]
        cells = [cell for cell in cells if cell]
        fingerprint = " ".join(cells)
        normalized_fingerprint = re.sub(r"\s+", " ", fingerprint).casefold()
        if normalized_fingerprint and normalized_fingerprint not in normalized_text:
            missing.append(row)
    return missing


def _extract_pdf(data: bytes) -> str:
    import pymupdf

    with pymupdf.open(stream=data, filetype="pdf") as document:
        page_text = []
        has_extractable_text = False
        for index, page in enumerate(document, 1):
            content = page.get_text("text")
            tables = page.find_tables().tables
            if tables:
                table_lines = []
                for table_index, table in enumerate(tables, 1):
                    missing_rows = _table_rows_missing_from_text(
                        table.extract(), content
                    )
                    if missing_rows:
                        table_lines.append(f"---- Detected table {table_index} ----")
                        table_lines.extend(
                            " | ".join(str(cell or "") for cell in row)
                            for row in missing_rows
                        )
                content += "\n" + "\n".join(table_lines)
            extras = []
            for widget in page.widgets() or []:
                if widget.field_value:
                    extras.append(str(widget.field_value))
            for annotation in page.annots() or []:
                info = annotation.info
                extras.extend(value for value in (info.get("title"), info.get("content")) if value)
            if extras:
                content += "\n" + "\n".join(extras)
            if content.strip():
                has_extractable_text = True
                page_text.append(f"---- Page {index} ----\n{content}")
            else:
                page_text.append(
                    f"---- Page {index}: no selectable text (blank or scanned; OCR not enabled) ----"
                )
        extracted = "\n".join(page_text)
        if not has_extractable_text:
            raise ValueError("PDF contains no extractable text; scanned PDFs require local OCR")
        return mark_statement_sections(extracted)


def _extract_docx(data: bytes) -> str:
    from docx import Document

    document = Document(io.BytesIO(data))
    lines = _docx_block_text(document.element.body)
    for section_index, section in enumerate(document.sections, 1):
        header = _docx_block_text(section.header._element)
        footer = _docx_block_text(section.footer._element)
        if header:
            lines.extend([f"---- Section {section_index} header ----", *header])
        if footer:
            lines.extend([f"---- Section {section_index} footer ----", *footer])
    auxiliary = _docx_auxiliary_text(data)
    if auxiliary:
        lines.extend(["---- DOCX comments and notes ----", *auxiliary])
    body = "\n".join(lines)
    if not body.strip():
        raise ValueError("DOCX contains no extractable text")
    return mark_statement_sections(body)


def extract_text(filename: str, data: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return _extract_pdf(data)
    if ext == ".docx":
        return _extract_docx(data)
    if ext == ".csv":
        rows = csv.reader(io.StringIO(_decode_text(data)))
        return mark_statement_sections(
            "\n".join(" | ".join(cell.strip() for cell in row) for row in rows)
        )
    if ext in {".txt", ".log"}:
        return mark_statement_sections(_decode_text(data))
    raise ValueError(f"Unsupported file type: {ext or '(no extension)'}")
