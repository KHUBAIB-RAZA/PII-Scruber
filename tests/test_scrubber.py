from pathlib import Path
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from docx import Document

from app.extraction import _table_rows_missing_from_text
from app.scrubber import (
    combine_files,
    contains_obvious_identifier,
    extract_text,
    scrub_and_check,
    scrub_folder,
    scrub_text,
)
from app.redaction import REDACTION_RULES


def test_scrub_removes_identifiers_and_preserves_analysis_values():
    source = (
        "Account holder: Amina Khan\n"
        "Account number: 1234567890 National ID: AB123456C\n"
        "Paid Foodpanda and Netflix.com on 2026-04-12.\n"
        "Amount: PKR 1,250.50; balance: $9,876.40.\n"
        "IBAN: PK36SCBL0000001123456702 phone: +92 300 1234567\n"
        "Email: amina.khan@example.com SSN: 123-45-6789\n"
        "Card: 4111 1111 1111 1111; Debit: 1234 56\n"
        "Reference: 123456789012345\n"
        "Billing address: 12 Sample Road, Lahore"
    )
    result = scrub_text(source)
    for private_value in (
        "PK36SCBL0000001123456702", "+92 300 1234567", "amina.khan@example.com",
        "123-45-6789", "4111 1111 1111 1111", "123456789012345", "12 Sample Road",
        "1234 56",
        "1234567890", "AB123456C",
    ):
        assert private_value not in result
    for analytical_value in (
        "Amina Khan", "Foodpanda", "Netflix.com", "2026-04-12", "PKR 1,250.50", "$9,876.40",
    ):
        assert analytical_value in result
    assert "<IBAN>" in result
    assert "<PHONE>" in result
    assert "<EMAIL_OR_WALLET_ID>" in result
    assert "<NATIONAL_ID>" in result
    assert "<CARD_NUMBER>" in result
    assert "<ADDRESS>" in result
    assert "Account holder: Amina Khan" in result


@pytest.mark.parametrize(
    ("source", "private_value", "placeholder"),
    [
        ("IBAN PK36SCBL0000001123456702", "PK36SCBL0000001123456702", "<IBAN>"),
        ("IBAN GB82 WEST 1234 5698 7654 32", "GB82 WEST 1234 5698 7654 32", "<IBAN>"),
        ("SORT 402219", "402219", "<SORT_CODE>"),
        ("Sort Code: 40-22-19", "40-22-19", "<SORT_CODE>"),
        ("S/C: 40 22 19", "40 22 19", "<SORT_CODE>"),
        ("Sort Code 402219", "402219", "<SORT_CODE>"),
        ("Account number: 1234567890", "1234567890", "<ACCOUNT_ID>"),
        ("A/C No: GB1234567890", "GB1234567890", "<ACCOUNT_ID>"),
        ("Routing number: 021000021", "021000021", "<ROUTING_ID>"),
        ("SWIFT BIC: DEUTDEFF500", "DEUTDEFF500", "<BANK_CODE>"),
        ("Bank code: 123456", "123456", "<BANK_CODE>"),
        ("Card number: 4111 1111 1111 1111", "4111 1111 1111 1111", "<CARD_NUMBER>"),
        ("National ID: AB123456C", "AB123456C", "<NATIONAL_ID>"),
        ("CNIC: 12345-1234567-1", "12345-1234567-1", "<NATIONAL_ID>"),
        ("IFSC: HDFC0123456", "HDFC0123456", "<IFSC>"),
        ("Phone: +92 300 1234567", "+92 300 1234567", "<PHONE>"),
        ("Email: test.person@example.com", "test.person@example.com", "<EMAIL_OR_WALLET_ID>"),
        ("Wallet id: person@walletabc", "person@walletabc", "<PAYMENT_HANDLE>"),
        ("Reference: ABCDE123456", "ABCDE123456", "<REFERENCE_ID>"),
        ("Customer number: ABCDE123456", "ABCDE123456", "<REFERENCE_ID>"),
        ("Account address: 12 Sample Road, Lahore", "12 Sample Road, Lahore", "<ADDRESS>"),
        ("Transaction id: 123456789012345", "123456789012345", "<REFERENCE_ID>"),
        ("Transfer: 123456789", "123456789", "<LONG_ID>"),
        ("Flat 10 High Street, London", "10 High Street, London", "<ADDRESS>"),
    ],
)
def test_named_redaction_rules_cover_synthetic_identifier_categories(
    source, private_value, placeholder
):
    result = scrub_and_check(source)
    assert private_value not in result
    assert placeholder in result


def test_redaction_rules_are_ordered_and_categorized():
    assert REDACTION_RULES
    assert len({rule.name for rule in REDACTION_RULES}) == len(REDACTION_RULES)
    assert all(rule.category and rule.replacement for rule in REDACTION_RULES)
    names = [rule.name for rule in REDACTION_RULES]
    assert names.index("iban_contiguous") < names.index("uk_sort_code_labeled")
    assert names.index("grouped_iban") < names.index("account_number")
    assert names.index("uk_sort_code_labeled") < names.index("phone_pattern")


def test_account_and_phone_redaction_preserve_adjacent_merchant_amount_date_and_card_last_four():
    source = (
        "Account number: 1234567890 | Foodpanda | 2026-04-04 | GBP 14.25 | Card ending 1234\n"
        "Phone: +92 300 1234567 | Netflix.com | 2026-04-05 | GBP 9.99"
    )
    result = scrub_and_check(source)
    assert "<ACCOUNT_ID>" in result
    assert "<PHONE>" in result
    for preserved in ("Foodpanda", "Netflix.com", "2026-04-04", "2026-04-05", "GBP 14.25", "GBP 9.99", "1234"):
        assert preserved in result


@pytest.mark.parametrize("date_like", ["04-22-19", "22-04-19", "2026-04-04"])
def test_valid_date_forms_are_not_treated_as_plain_sort_codes(date_like):
    result = scrub_and_check(f"Statement date: {date_like}")
    assert date_like in result
    assert "<SORT_CODE>" not in result


def test_csv_extraction_and_combined_headers():
    data = b"Date,Merchant,Amount\n2026-04-01,Netflix.com,12.99\n"
    assert extract_text("transactions.csv", data) == (
        "Date | Merchant | Amount\n2026-04-01 | Netflix.com | 12.99"
    )
    combined = combine_files([("transactions.csv", data), ("notes.txt", b"Paid Foodpanda")])
    assert combined.count("---- File:") == 2
    assert "Netflix.com" in combined and "Foodpanda" in combined


def test_folder_in_combined_string_out(tmp_path: Path):
    (tmp_path / "a.txt").write_text("Merchant: TESCO STORES", encoding="utf-8")
    (tmp_path / "b.csv").write_text("Date,Amount\n2026-04-02,4.50\n", encoding="utf-8")
    combined = scrub_folder(tmp_path)
    assert "---- File: a.txt ----" in combined
    assert "---- File: b.csv ----" in combined
    assert "TESCO STORES" in combined
    assert "2026-04-02 | 4.50" in combined


def test_empty_and_unsupported_files_fail_explicitly():
    with pytest.raises(ValueError, match="At least one"):
        combine_files([])
    with pytest.raises(ValueError, match="Unsupported"):
        extract_text("statement.xlsx", b"not supported")


def test_egress_check_scrubs_identifiers_before_transmission():
    assert contains_obvious_identifier("contact alice@example.com")
    safe = scrub_and_check("contact alice@example.com")
    assert "alice@example.com" not in safe
    assert "<EMAIL_OR_WALLET_ID>" in safe
    assert not contains_obvious_identifier(scrub_and_check("Paid Netflix.com 2026-04-02 $1,200.00"))


def test_address_detection_handles_csv_fields_and_unlabeled_street_addresses():
    result = scrub_text("Amina Khan | Address: 12 Example Road, Lahore; Balance: PKR 1,200.00")
    assert "12 Example Road" not in result
    assert "Amina Khan" in result
    assert "Balance: PKR 1,200.00" in result


@pytest.mark.parametrize("source", [
    "SORT 402219",
    "SORT CODE 40-22-19",
    "40-22-19",
])
def test_uk_sort_codes_removed_without_touching_dates_or_amounts(source):
    result = scrub_text(
        f"{source} | Date 2026-04-04 | Amount GBP 40.22 | Card ending 1234"
    )
    assert "402219" not in result
    assert "40-22-19" not in result
    assert "<SORT_CODE>" in result
    assert "2026-04-04" in result
    assert "04-22-19" in scrub_text("Date: 04-22-19")
    assert "22-04-19" in scrub_text("Date: 22-04-19")
    assert "GBP 40.22" in result
    assert "1234" in result


def test_docx_extracts_body_tables_headers_footers_and_multiple_bank_sections():
    document = Document()
    document.sections[0].header.paragraphs[0].text = "Statement delivery notice"
    document.add_paragraph("Crescent Bank Statement")
    document.add_paragraph("Account Holder: Amina Khan")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Date"
    table.cell(0, 1).text = "Amount"
    table.cell(1, 0).text = "2026-04-01 Foodpanda"
    table.cell(1, 1).text = "PKR 250"
    document.add_paragraph("International Bank Statement")
    document.add_paragraph("2026-04-02 Netflix.com PKR 900")
    document.sections[0].footer.paragraphs[0].text = "Customer Service"
    stream = BytesIO()
    document.save(stream)

    extracted = extract_text("combined.docx", stream.getvalue())
    assert "---- Statement section 1 ----" in extracted
    assert "---- Statement section 2 ----" in extracted
    assert "---- Section 1 header ----" in extracted
    assert "---- Section 1 footer ----" in extracted
    assert "Date | Amount" in extracted
    assert "2026-04-01 Foodpanda | PKR 250" in extracted
    assert "2026-04-02 Netflix.com PKR 900" in extracted


def test_docx_extracts_comment_and_footnote_text():
    document = Document()
    document.add_paragraph("Bank Statement")
    stream = BytesIO()
    document.save(stream)
    original = stream.getvalue()
    updated = BytesIO()
    with ZipFile(BytesIO(original)) as source, ZipFile(updated, "w", ZIP_DEFLATED) as target:
        for item in source.infolist():
            target.writestr(item, source.read(item.filename))
        target.writestr(
            "word/comments.xml",
            '<w:comments xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            '<w:comment><w:p><w:r><w:t>Commented merchant note</w:t></w:r></w:p></w:comment>'
            '</w:comments>',
        )
        target.writestr(
            "word/footnotes.xml",
            '<w:footnotes xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            '<w:footnote><w:p><w:r><w:t>Footnote transaction detail</w:t></w:r></w:p></w:footnote>'
            '</w:footnotes>',
        )
    extracted = extract_text("annotated.docx", updated.getvalue())
    assert "Commented merchant note" in extracted
    assert "Footnote transaction detail" in extracted


def test_pdf_extraction_marks_every_page_and_preserves_blank_page_warning():
    import fitz

    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "First Bank Statement")
    pdf.new_page()
    page = pdf.new_page()
    page.insert_text((72, 72), "Second Account Statement")
    data = pdf.tobytes()
    pdf.close()

    extracted = extract_text("multi.pdf", data)
    assert "---- Page 1 ----" in extracted
    assert "---- Page 2: no selectable text (blank or scanned; OCR not enabled) ----" in extracted
    assert "---- Page 3 ----" in extracted
    assert "---- Statement section 1 ----" in extracted
    assert "---- Statement section 2 ----" in extracted


def test_pdf_extraction_preserves_detectable_table_cells():
    import fitz

    pdf = fitz.open()
    page = pdf.new_page()
    for x in (72, 180, 300):
        page.draw_line((x, 72), (x, 132))
    for y in (72, 102, 132):
        page.draw_line((72, y), (300, y))
    page.insert_text((80, 94), "Date")
    page.insert_text((188, 94), "Amount")
    page.insert_text((80, 124), "2026-04-04")
    page.insert_text((188, 124), "PKR 550")
    data = pdf.tobytes()
    pdf.close()

    extracted = extract_text("table.pdf", data)
    assert "---- Detected table 1 ----" not in extracted
    assert "Date" in extracted and "Amount" in extracted
    assert "2026-04-04" in extracted and "PKR 550" in extracted


def test_pdf_table_extraction_keeps_rows_missing_from_page_text():
    rows = [
        ["Date", "Merchant"],
        ["2026-04-04", "Foodpanda"],
    ]
    assert _table_rows_missing_from_text(
        rows, "Date Merchant\n2026-04-04"
    ) == [["2026-04-04", "Foodpanda"]]


def test_scanned_only_pdf_fails_instead_of_silently_returning_empty_text():
    import fitz

    pdf = fitz.open()
    pdf.new_page()
    data = pdf.tobytes()
    pdf.close()
    with pytest.raises(ValueError, match="scanned PDFs require local OCR"):
        extract_text("scan.pdf", data)
