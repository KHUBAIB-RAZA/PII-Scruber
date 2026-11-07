from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class RedactionRule:
    name: str
    category: str
    replacement: str
    pattern: re.Pattern[str]


IBAN_CONTIGUOUS = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b", re.I)
IBAN_GROUPED = re.compile(
    r"\b[A-Z]{2}\d{2}(?:[ \t-][A-Z0-9]{4}){2,7}(?:[ \t-][A-Z0-9]{1,4})?\b",
    re.I,
)
PLAIN_SORT_CODE = re.compile(r"\b\d{2}-\d{2}-\d{2}\b")
PLUS_PHONE = re.compile(r"(?<!\w)\+\d[\d\s().-]{7,20}\d(?!\w)")
SORT_CODE_LABEL = re.compile(
    r"\b(?:sort(?:ing)?(?:\s+code)?|s\s*/\s*c)\s*[:#-]?\s*"
    r"\d{2}[\s-]?\d{2}[\s-]?\d{2}\b",
    re.I,
)

# Ordered by specificity: structured identifiers first, then contextual IDs and
# contact/address patterns, with long numeric IDs last to avoid clipping amounts.
REDACTION_RULES = [
    RedactionRule("iban_contiguous", "iban", "<IBAN>", IBAN_CONTIGUOUS),
    RedactionRule(
        "grouped_iban", "iban", "<IBAN>",
        re.compile(r"\b[A-Z]{2}\d{2}(?:[ \t-][A-Z0-9]{4}){2,7}(?:[ \t-][A-Z0-9]{1,4})?\b", re.I),
    ),
    RedactionRule(
        "uk_sort_code_labeled", "sort_code", "<SORT_CODE>",
        re.compile(
            r"\b(?:sort(?:ing)?(?:\s+code)?|s\s*/\s*c)\s*[:#-]?\s*"
            r"\d{2}[\s-]?\d{2}[\s-]?\d{2}\b",
            re.I,
        ),
    ),
    RedactionRule("uk_sort_code_unlabeled", "sort_code", "<SORT_CODE>", PLAIN_SORT_CODE),
    RedactionRule(
        "email", "email", "<EMAIL_OR_WALLET_ID>",
        re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    ),
    RedactionRule(
        "payment_handle", "payment_handle", "<PAYMENT_HANDLE>",
        re.compile(r"\b[a-zA-Z0-9._-]{2,}@[a-zA-Z][a-zA-Z0-9_-]{2,}\b"),
    ),
    RedactionRule(
        "reference", "reference", "<REFERENCE_ID>",
        re.compile(
            r"\b(?:reference|ref|trace|transaction|txn|confirmation|customer\s*"
            r"(?:no\.?|number|#|id))\s*(?:(?:no\.?|number|#|id)\s*[:=]?\s*|[:=]\s*)"
            r"[A-Z0-9-]{7,}\b",
            re.I,
        ),
    ),
    RedactionRule(
        "account_number", "account_number", "<ACCOUNT_ID>",
        re.compile(
            r"\b(?:account|acct|a/c)\s*(?:(?:no\.?|number|#|id)\s*[:=]?\s*|[:=]\s*)"
            r"(?:[A-Z0-9][A-Z0-9-]{5,33}|\d{2,}(?:[ -]\d{2,})+)\b",
            re.I,
        ),
    ),
    RedactionRule(
        "routing_number", "routing_number", "<ROUTING_ID>",
        re.compile(
            r"\b(?:routing|transit)\s*(?:no\.?|number|#)?\s*[:=]?\s*"
            r"\d(?:[ -]?\d){5,14}\b",
            re.I,
        ),
    ),
    RedactionRule(
        "bank_code", "bank_code", "<BANK_CODE>",
        re.compile(
            r"\b(?:(?:bank|branch)\s+code\s*(?:no\.?|number|#)?\s*[:=]?\s*"
            r"[A-Z0-9]{6,12}|(?:swift|bic)\s*(?:code|no\.?|number|#)?"
            r"\s*[:=]?\s*[A-Z0-9]{8}(?:[A-Z0-9]{3})?)\b",
            re.I,
        ),
    ),
    RedactionRule(
        "national_id_cnic", "national_id", "<NATIONAL_ID>",
        re.compile(r"(?<!\d)\d{5}-\d{7}-\d(?!\d)"),
    ),
    RedactionRule(
        "national_id_ssn", "national_id", "<NATIONAL_ID>",
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    ),
    RedactionRule(
        "national_id_nino", "national_id", "<NATIONAL_ID>",
        re.compile(r"\b[A-CEGHJ-PR-TW-Z]{2}\d{6}[A-D]\b", re.I),
    ),
    RedactionRule(
        "national_id_labeled", "national_id", "<NATIONAL_ID>",
        re.compile(
            r"\b(?:national\s*(?:identity|id)|identity|ssn|sin|cnic|nino)\s*"
            r"(?:(?:no\.?|number|id|#)\s*[:=]?\s*|[:=]\s*)[A-Z0-9-]{6,20}\b",
            re.I,
        ),
    ),
    RedactionRule(
        "card_number_labeled", "card_number", "<CARD_NUMBER>",
        re.compile(r"\b(?:credit|debit|card)\s*(?:no\.?|number|#)?\s*[:=]?\s*\d[\d -]{4,}\d\b", re.I),
    ),
    RedactionRule(
        "card_number_long", "card_number", "<CARD_NUMBER>",
        re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)"),
    ),
    RedactionRule(
        "ifsc", "bank_code", "<IFSC>",
        re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b", re.I),
    ),
    RedactionRule(
        "phone_labeled", "phone", "<PHONE>",
        re.compile(
            r"\b(?:phone|mobile|telephone|tel)\s*(?:no\.?|number|#)?\s*[:=]?\s*"
            r"(?:\+\d[\d\s().-]{7,20}\d|"
            r"(?:\(\d{2,4}\)|\d{2,4})[\s.-]*\d{3,4}[\s.-]*\d{3,4})\b",
            re.I,
        ),
    ),
    RedactionRule("phone_international", "phone", "<PHONE>", PLUS_PHONE),
    RedactionRule(
        "phone_pattern", "phone", "<PHONE>",
        re.compile(
            r"(?<!\w)(?:\+\d{1,3}[\s().-]*)?(?:\(\d{2,4}\)|\d{2,4})"
            r"[\s.-]+\d{3,4}[\s.-]+\d{3,4}(?!\w)"
        ),
    ),
    RedactionRule(
        "long_numeric_id", "long_numeric_id", "<LONG_ID>",
        re.compile(r"(?<![\d.,])\d{9,}(?![\d.,])"),
    ),
    RedactionRule(
        "address_labeled", "address", "<ADDRESS>",
        re.compile(r"\b(?:home|billing|postal|residential|customer)?\s*address\s*[:=]\s*[^|\r\n;]+", re.I),
    ),
    RedactionRule(
        "address_street", "address", "<ADDRESS>",
        re.compile(r"\b\d{1,6}\s+(?:[\w.'-]+\s+){0,5}(?:street|st\.?|road|rd\.?|avenue|ave\.?|lane|ln\.?|drive|dr\.?|boulevard|blvd\.?|court|ct\.?|place|pl\.?)\b(?:,\s*[^|\r\n;]+)?", re.I),
    ),
]

# Compatibility alias for downstream code that enumerated the older patterns.
REPLACEMENTS = tuple(
    (rule.replacement, rule.pattern)
    for rule in REDACTION_RULES
    if rule.name not in {
        "iban_contiguous",
        "grouped_iban",
        "uk_sort_code_labeled",
        "uk_sort_code_unlabeled",
    }
)


def _redact_grouped_iban(match: re.Match[str]) -> str:
    value = match.group()
    compact = re.sub(r"[ \t-]", "", value)
    if not 15 <= len(compact) <= 34:
        return value
    groups = re.split(r"[ \t-]+", value)
    if len(groups) > 1 and (
        len(groups[0]) != 4
        or any(len(group) != 4 for group in groups[1:-1])
        or not 1 <= len(groups[-1]) <= 4
    ):
        return value
    return "<IBAN>"


def _redact_plain_sort_code(match: re.Match[str]) -> str:
    value = match.group()
    first, second, third = (int(part) for part in value.split("-"))
    for year in (2000 + third, 1900 + third):
        for month, day in ((first, second), (second, first)):
            try:
                date(year, month, day)
            except ValueError:
                continue
            return value
    return "<SORT_CODE>"


def scrub_text(text: str) -> str:
    """Apply ordered, named local redaction rules without external lookup."""
    for rule in REDACTION_RULES:
        if rule.name == "iban_contiguous":
            text = rule.pattern.sub(rule.replacement, text)
        elif rule.name == "grouped_iban":
            text = rule.pattern.sub(_redact_grouped_iban, text)
        elif rule.name == "uk_sort_code_labeled":
            text = rule.pattern.sub("<SORT_CODE>", text)
        elif rule.name == "uk_sort_code_unlabeled":
            text = rule.pattern.sub(_redact_plain_sort_code, text)
        else:
            text = rule.pattern.sub(rule.replacement, text)
    return text


def contains_obvious_identifier(text: str) -> bool:
    """Fail closed if a known high-confidence direct identifier remains."""
    if IBAN_CONTIGUOUS.search(text) or any(
        _redact_grouped_iban(match) == "<IBAN>" for match in IBAN_GROUPED.finditer(text)
    ):
        return True
    if SORT_CODE_LABEL.search(text):
        return True
    if any(pattern.search(text) for _, pattern in REPLACEMENTS):
        return True
    if any(
        _redact_plain_sort_code(match) == "<SORT_CODE>"
        for match in PLAIN_SORT_CODE.finditer(text)
    ):
        return True
    return bool(PLUS_PHONE.search(text))


def scrub_and_check(text: str) -> str:
    safe = scrub_text(text)
    if contains_obvious_identifier(safe):
        raise ValueError("Potential identifier remains after local scrubbing; refusing external transmission")
    return safe
