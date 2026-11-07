import json
import sys
from types import SimpleNamespace

import pytest
from fastapi.routing import APIRoute

import app.analysis as llm
from app.dependencies import get_db
from app.analysis import (
    AnalysisRequestLimit,
    AnalysisIncomplete,
    COMPLETION_MAX_TOKENS,
    ProviderRequestError,
    StatementTooLarge,
    UngroundedResponse,
    analyze,
    chat,
    classify_provider_error,
    _completion,
    estimate_provider_requests,
    preview_analysis_payloads,
    preview_chat_payloads,
    split_source,
    validate_grounded_response,
)
import app.main as main

SOURCE = "2026-04-02 | Foodpanda | PKR 1,200.00"


def test_statement_routes_use_injected_request_database_session():
    protected_paths = {
        "/api/statements/analyze",
        "/api/statements/preview",
        "/api/statements/{statement_id}",
        "/api/statements/{statement_id}/groq-preview",
        "/api/statements/{statement_id}/analyze",
        "/api/statements/{statement_id}/chat-preview",
        "/api/statements/{statement_id}/chat",
    }
    routes = [
        route
        for route in main.app.routes
        if isinstance(route, APIRoute) and route.path in protected_paths
    ]
    assert {route.path for route in routes} >= protected_paths
    assert all(
        any(dependency.call is get_db for dependency in route.dependant.dependencies)
        for route in routes
    )


def test_accepts_exact_source_quote_and_supported_amount():
    candidate = json.dumps({
        "answer": "The Foodpanda transaction was PKR 1,200.00.",
        "evidence": ["2026-04-02 | Foodpanda | PKR 1,200.00"],
        "refuse": False,
    })
    assert validate_grounded_response(candidate, SOURCE)["answer"].startswith("The Foodpanda")


def test_accepts_equivalent_thousands_and_decimal_formatting():
    candidate = json.dumps({
        "answer": "The transaction is PKR 1200.",
        "evidence": ["2026-04-02 | Foodpanda | PKR 1,200.00"],
        "refuse": False,
    })
    assert validate_grounded_response(candidate, SOURCE)["answer"].startswith("The transaction")


def test_response_evidence_and_answer_are_rescrubbed_before_return():
    source = "Sort code 40-22-19 | Foodpanda | 2026-04-02 | GBP 12.00"
    candidate = json.dumps({
        "answer": "Sort code 40-22-19 was used for Foodpanda GBP 12.00.",
        "evidence": ["Sort code 40-22-19 | Foodpanda | 2026-04-02 | GBP 12.00"],
        "refuse": False,
    })
    result = validate_grounded_response(candidate, source)
    assert "40-22-19" not in result["answer"]
    assert all("40-22-19" not in item for item in result["evidence"])
    assert "<SORT_CODE>" in result["answer"]
    assert "<SORT_CODE>" in result["evidence"][0]


def test_single_transaction_cannot_support_frequency_claim():
    source = "2026-04-02 | Foodpanda | PKR 1,200.00"
    candidate = json.dumps({
        "answer": "This was a weekly grocery purchase.",
        "evidence": [source],
        "refuse": False,
    })
    with pytest.raises(UngroundedResponse, match="two distinct dated transactions"):
        validate_grounded_response(candidate, source)


def test_repeated_citation_of_same_date_does_not_support_recurrence():
    source = "2026-04-02 | Foodpanda | PKR 1,200.00"
    candidate = json.dumps({
        "answer": "This was a monthly food purchase.",
        "evidence": [source, source],
        "refuse": False,
    })
    with pytest.raises(UngroundedResponse, match="two distinct dated transactions"):
        validate_grounded_response(candidate, source)


def test_two_distinct_dated_transactions_can_support_frequency_claim():
    source = (
        "2026-04-02 | Foodpanda | PKR 1,200.00\n"
        "2026-04-09 | Foodpanda | PKR 1,450.00"
    )
    candidate = json.dumps({
        "answer": "Two weekly purchases are shown.",
        "evidence": [source],
        "refuse": False,
    })
    assert "weekly" in validate_grounded_response(candidate, source)["answer"]


def test_recurrence_claim_with_multiple_dates_must_not_include_identifier():
    source = (
        "Sort code 40-22-19 | Foodpanda | 2026-04-02 | GBP 12.00\n"
        "Sort code 40-22-19 | Foodpanda | 2026-04-09 | GBP 14.00"
    )
    sanitized_source = llm.scrub_and_check(source)
    candidate = json.dumps({
        "answer": "These were weekly Foodpanda purchases.",
        "evidence": [source],
        "refuse": False,
    })
    result = validate_grounded_response(candidate, source)
    assert "40-22-19" not in json.dumps(result)
    assert result["evidence"][0] == sanitized_source


def test_arithmetic_not_directly_supported_by_citations_is_rejected_with_recovery_guidance():
    candidate = json.dumps({
        "answer": "The total was PKR 2,400.00.",
        "evidence": ["2026-04-02 | Foodpanda | PKR 1,200.00"],
        "refuse": False,
    })
    with pytest.raises(UngroundedResponse, match="Computed totals, counts, and percentages"):
        validate_grounded_response(candidate, SOURCE)


def test_large_analysis_is_chunked_and_each_reply_is_grounded_to_its_source_chunk(monkeypatch):
    monkeypatch.setattr(llm, "ANALYSIS_CHUNK_CHARS", 24)
    source = "Foodpanda lunch | PKR 120\nNetflix monthly | PKR 20\nTESCO groceries | PKR 80\n"
    chunks = split_source(source)
    seen = []

    def completion(messages, *_):
        prompt = messages[-1]["content"]
        chunk = prompt.split("Include exact evidence quotes from this segment.\n\n", 1)[1]
        evidence = chunk.strip()
        assert len(chunk) <= llm.ANALYSIS_CHUNK_CHARS
        seen.append(evidence)
        return json.dumps({"answer": "A listed transaction is present.", "evidence": [evidence], "refuse": False})

    monkeypatch.setattr(llm, "_completion", completion)
    result = analyze(source)
    assert len(seen) == len(chunks) > 1
    assert all(quote in source for quote in seen)
    assert "Evidence:" in result


def test_complete_json_is_accepted_even_when_provider_finish_reason_is_length(monkeypatch):
    source = "Foodpanda lunch | PKR 120"
    payload = json.dumps({
        "answer": "Foodpanda was listed.",
        "evidence": [source],
        "refuse": False,
    })

    class FakeGroq:
        def __init__(self, api_key):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=lambda **_: SimpleNamespace(
                    choices=[SimpleNamespace(
                        finish_reason="length",
                        message=SimpleNamespace(content=payload, refusal=None),
                    )]
                ))
            )

    monkeypatch.setenv("GROQ_API_KEY", "synthetic-test-key")
    monkeypatch.setitem(sys.modules, "groq", SimpleNamespace(Groq=FakeGroq))
    result = analyze(source)
    assert "Foodpanda was listed." in result
    assert source in result


def test_incomplete_json_is_rejected_without_exposing_partial_claim(monkeypatch):
    monkeypatch.setattr(llm, "_completion", lambda *_args, **_kwargs: (
        '{"answer":"UNVERIFIED_PRIVATE_CLAIM'
    ))
    with pytest.raises(AnalysisIncomplete) as error:
        analyze("Foodpanda transaction | GBP 12.00")
    assert error.value.failed_segment == 1
    assert error.value.segment_count == 1
    assert "No text from that response was accepted." in error.value.analysis
    assert "UNVERIFIED_PRIVATE_CLAIM" not in error.value.analysis
    assert "UNVERIFIED_PRIVATE_CLAIM" not in str(error.value)


def test_mixed_completed_and_truncated_segments_keep_only_validated_results(monkeypatch):
    monkeypatch.setattr(llm, "ANALYSIS_CHUNK_CHARS", 24)
    source = "Merchant Alpha | GBP 12\nMerchant Beta | GBP 8\n"
    chunks = split_source(source)
    assert len(chunks) == 2
    calls = []

    def completion(messages, *_args, **_kwargs):
        calls.append(messages)
        chunk = messages[-1]["content"].split(
            "Include exact evidence quotes from this segment.\n\n", 1
        )[1].strip()
        if len(calls) == 2:
            return '{"answer":"truncated'
        return json.dumps({
            "answer": "A listed transaction is present.",
            "evidence": [chunk],
            "refuse": False,
        })

    monkeypatch.setattr(llm, "_completion", completion)
    with pytest.raises(AnalysisIncomplete) as error:
        analyze(source)
    assert error.value.failed_segment == 2
    assert error.value.segment_count == 2
    assert "A listed transaction is present." in error.value.analysis
    assert "Evidence:" in error.value.analysis
    assert "truncated" not in error.value.analysis
    assert "Segment coverage:" in error.value.analysis
    assert [segment["status"] for segment in error.value.segment_statuses] == [
        "complete", "partial"
    ]


def test_analysis_payloads_use_reduced_output_budget():
    payload = preview_analysis_payloads("Foodpanda | GBP 12.00")[0]
    for mode in ("primary", "json_recovery"):
        assert payload[mode]["max_completion_tokens"] == COMPLETION_MAX_TOKENS
        assert payload[mode]["reasoning_effort"] == "low"
        assert "max_tokens" not in payload[mode]
    assert COMPLETION_MAX_TOKENS == 1400
    assert "<=40 words" in payload["primary"]["messages"][1]["content"]
    assert "two distinct dated cited transactions" in payload["primary"]["messages"][0]["content"]
    assert "reasoning_effort" not in json.dumps(payload["primary"]["messages"])


def test_provider_request_estimate_is_bounded_and_explains_adaptive_cost():
    estimate = estimate_provider_requests(5)
    assert estimate["initial_requests"] == 5
    assert estimate["including_one_json_fallback_per_segment"] == 10
    assert estimate["hard_run_limit"] == 64
    assert "Adaptive recovery may use additional requests" in estimate["note"]


def test_analysis_does_not_split_nonrecoverable_grounding_failures(monkeypatch):
    source = "Merchant " + ("source text " * 80) + " | GBP 12.00"
    calls = []

    def completion(_messages, *_args, **_kwargs):
        calls.append(1)
        return json.dumps({
            "answer": "The expense was GBP 999.00.",
            "evidence": ["Merchant"],
            "refuse": False,
        })

    monkeypatch.setattr(llm, "_completion", completion)
    with pytest.raises(AnalysisIncomplete) as error:
        analyze(source)
    assert len(calls) == 1
    assert error.value.failure_category == "unsupported_number"
    assert llm.SPLITTABLE_FAILURES == {
        "invalid_json", "output_length", "request_too_large"
    }


def test_invalid_large_segment_is_adaptively_split_and_subchunks_are_validated(monkeypatch):
    monkeypatch.setattr(llm, "ANALYSIS_CHUNK_CHARS", 2000)
    lines = ["Merchant Alpha " + "x" * 590, "Merchant Beta " + "y" * 590]
    source = "\n".join(lines)
    calls = []

    def completion(messages, *_args, **_kwargs):
        content = messages[-1]["content"]
        chunk = content.split(
            "Include exact evidence quotes from this segment.\n\n", 1
        )[1]
        calls.append((content, chunk))
        if len(calls) == 1:
            return '{"answer":"truncated'
        return json.dumps({
            "answer": "A listed transaction is present.",
            "evidence": [chunk.strip()],
            "refuse": False,
        })

    monkeypatch.setattr(llm, "_completion", completion)
    result = analyze(source)
    assert len(calls) == 3
    assert "smaller recovery part" in calls[1][0]
    assert "smaller recovery part" in calls[2][0]
    assert calls[1][1] + calls[2][1] == source
    assert result.count("A listed transaction is present.") == 2
    assert "Evidence:" in result


def test_empty_length_response_splits_chunk_and_recovery_payload_is_low_effort(monkeypatch):
    monkeypatch.setattr(llm, "ANALYSIS_CHUNK_CHARS", 1800)
    source = "Merchant Alpha " + "x" * 550 + "\nMerchant Beta " + "y" * 550
    calls = []

    class FakeGroq:
        def __init__(self, api_key):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self.create)
            )

        def create(self, **payload):
            calls.append(payload)
            user_message = payload["messages"][-1]["content"]
            chunk = user_message.split(
                "Include exact evidence quotes from this segment.\n\n", 1
            )[1]
            assert payload["model"] == "openai/gpt-oss-20b"
            assert payload["reasoning_effort"] == "low"
            assert payload["max_completion_tokens"] == COMPLETION_MAX_TOKENS
            if len(chunk) > 700:
                return SimpleNamespace(choices=[SimpleNamespace(
                    finish_reason="length",
                    message=SimpleNamespace(content=None, refusal=None),
                )])
            return SimpleNamespace(choices=[SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=json.dumps({
                    "answer": "A merchant transaction is listed.",
                    "evidence": [chunk.strip()],
                    "refuse": False,
                }), refusal=None),
            )])

    monkeypatch.setenv("GROQ_API_KEY", "synthetic-test-key")
    monkeypatch.setitem(sys.modules, "groq", SimpleNamespace(Groq=FakeGroq))
    result = analyze(source)
    assert len(calls) == 3
    assert result.count("A merchant transaction is listed.") == 2
    assert all(call["reasoning_effort"] == "low" for call in calls)
    assert all(call["max_completion_tokens"] == 1400 for call in calls)


def test_empty_length_responses_split_to_floor_then_stop_safely(monkeypatch):
    monkeypatch.setattr(llm, "ANALYSIS_CHUNK_CHARS", 1000)
    source = "merchant " + ("x" * 800)
    calls = []

    class FakeGroq:
        def __init__(self, api_key):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self.create)
            )

        def create(self, **_payload):
            calls.append(1)
            return SimpleNamespace(choices=[SimpleNamespace(
                finish_reason="length",
                message=SimpleNamespace(content=None, refusal=None),
            )])

    monkeypatch.setenv("GROQ_API_KEY", "synthetic-test-key")
    monkeypatch.setitem(sys.modules, "groq", SimpleNamespace(Groq=FakeGroq))
    with pytest.raises(AnalysisIncomplete) as error:
        analyze(source)
    assert 1 < len(calls) <= 64
    assert error.value.failure_category == "output_length"
    assert "output_length" in str(error.value)
    assert "merchant" not in str(error.value)
    assert "No text from that response was accepted." in error.value.analysis


def test_failed_recovery_subchunk_keeps_valid_sibling_and_reports_category(monkeypatch):
    monkeypatch.setattr(llm, "ANALYSIS_CHUNK_CHARS", 2000)
    source = "Merchant Alpha " + "x" * 480 + "\n" + "Merchant Beta " + "y" * 480
    calls = []

    def completion(messages, *_args, **_kwargs):
        chunk = messages[-1]["content"].split(
            "Include exact evidence quotes from this segment.\n\n", 1
        )[1]
        calls.append(chunk)
        if len(calls) == 1 or "Merchant Alpha" in chunk:
            return '{"answer":"truncated'
        return json.dumps({
            "answer": "A listed transaction is present.",
            "evidence": [chunk.strip()],
            "refuse": False,
        })

    monkeypatch.setattr(llm, "_completion", completion)
    with pytest.raises(AnalysisIncomplete) as error:
        analyze(source)
    assert len(calls) == 5
    assert error.value.failure_category == "invalid_json"
    assert "reason: invalid_json" in str(error.value)
    assert "A listed transaction is present." in error.value.analysis
    assert "truncated" not in error.value.analysis
    assert error.value.segment_statuses[0]["status"] == "partial"
    assert error.value.segment_statuses[0]["validated_subchunks"] >= 1
    assert error.value.segment_statuses[0]["failed_subchunks"] >= 1
    assert "Merchant Alpha" not in str(error.value)
    assert "Merchant Beta" not in str(error.value)


def test_adaptive_analysis_stops_at_hard_provider_request_budget(monkeypatch):
    monkeypatch.setattr(llm, "ANALYSIS_CHUNK_CHARS", 6000)
    monkeypatch.setattr(llm, "MAX_PROVIDER_REQUESTS", 5)
    source = "transaction " + ("x" * 4300)
    calls = []

    class FakeGroq:
        def __init__(self, api_key):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self.create)
            )

        def create(self, **payload):
            calls.append(1)
            if "response_format" in payload:
                raise type("BadRequestError", (Exception,), {"status_code": 400})(
                    "failed_generation json_validate_failed"
                )
            return SimpleNamespace(choices=[SimpleNamespace(
                finish_reason="length",
                message=SimpleNamespace(content='{"answer":"truncated', refusal=None),
            )])

    monkeypatch.setenv("GROQ_API_KEY", "synthetic-test-key")
    monkeypatch.setitem(sys.modules, "groq", SimpleNamespace(Groq=FakeGroq))
    with pytest.raises(AnalysisIncomplete) as error:
        analyze(source)
    assert len(calls) == 5
    assert "request_limit" in error.value.failure_category
    assert "request_limit" in str(error.value)
    assert "transaction" not in str(error.value)


def test_request_budget_counts_json_recovery_provider_call(monkeypatch):
    from app.analysis import _RequestBudget

    budget = _RequestBudget(limit=1)
    budget.consume()
    with pytest.raises(AnalysisRequestLimit):
        budget.consume()


def test_large_chat_is_chunked_and_each_answer_cites_only_its_chunk(monkeypatch):
    monkeypatch.setattr(llm, "ANALYSIS_CHUNK_CHARS", 24)
    source = "Foodpanda lunch | PKR 120\nNetflix monthly | PKR 20\n"
    seen = []

    def completion(messages, *_):
        prompt = messages[-1]["content"]
        chunk = prompt.split("SOURCE SEGMENT:\n", 1)[1].split("\nQUESTION:\n", 1)[0]
        evidence = chunk.strip()
        assert len(chunk) <= llm.ANALYSIS_CHUNK_CHARS
        seen.append(evidence)
        return json.dumps({"answer": "This transaction is listed.", "evidence": [evidence], "refuse": False})

    monkeypatch.setattr(llm, "_completion", completion)
    answer = chat(source, "Which payments are listed?", [])
    assert len(seen) == len(split_source(source)) > 1
    assert all(quote in source for quote in seen)
    assert "Evidence:" in answer


def test_monthly_aggregate_chat_gets_clear_local_explanation_without_provider(monkeypatch):
    question = "What are monthly expenses of Marcus T. Alvarez?"

    def forbidden(*_args, **_kwargs):
        pytest.fail("aggregate request must not call Groq")

    monkeypatch.setattr(llm, "_completion", forbidden)
    answer = chat(
        "2026-04-04 | Synthetic Market | GBP 12.00",
        question,
        [],
    )
    assert "cannot reliably calculate monthly expense totals" in answer
    assert "individual dated expense entries" in answer
    assert "Marcus T. Alvarez" not in answer


def test_complete_analysis_reports_segment_coverage():
    coverage = llm.analysis_segment_statuses("Validated result", "A\nB\n")
    assert len(coverage) == 1
    assert coverage[0] == {
        "segment": 1,
        "segment_count": 1,
        "status": "complete",
        "validated_subchunks": 1,
        "failed_subchunks": 0,
        "categories": [],
    }


def test_incomplete_analysis_coverage_metadata_round_trips_without_source_text():
    source = "Synthetic source A\nSynthetic source B"
    marker = (
        "[ANALYSIS INCOMPLETE] Segment 2 of 2 (reason: invalid_json).\n"
        "Segment coverage:\n"
        "- segment 1/2: complete; validated_subchunks=1; failed_subchunks=0; categories=none\n"
        "- segment 2/2: partial; validated_subchunks=1; failed_subchunks=1; categories=invalid_json\n"
    )
    coverage = llm.analysis_segment_statuses(marker, source)
    assert [part["status"] for part in coverage] == ["complete", "partial"]
    assert coverage[1]["categories"] == ["invalid_json"]
    assert "Synthetic source" not in json.dumps(coverage)


def test_too_many_chunks_fail_before_any_provider_request(monkeypatch):
    monkeypatch.setattr(llm, "ANALYSIS_CHUNK_CHARS", 10)
    monkeypatch.setattr(llm, "MAX_ANALYSIS_CHUNKS", 2)
    called = False

    def completion(*_):
        nonlocal called
        called = True
        raise AssertionError("provider must not be called for over-limit input")

    monkeypatch.setattr(llm, "_completion", completion)
    with pytest.raises(StatementTooLarge, match="scrubbed statement is too large"):
        analyze("a" * 21)
    assert not called


@pytest.mark.parametrize(("error", "status", "message"), [
    (type("Provider413", (Exception,), {"status_code": 413})("payload too large"), 413, "too large"),
    (type("Provider429", (Exception,), {"status_code": 429})("rate limit"), 429, "rate-limiting"),
    (TimeoutError("request timed out"), 503, "could not be reached"),
    (RuntimeError("unavailable"), 502, "could not complete"),
])
def test_provider_failures_have_safe_recoverable_classification(error, status, message):
    classified = classify_provider_error(error)
    assert isinstance(classified, ProviderRequestError)
    assert classified.status_code == status
    assert message in str(classified)
    assert "scrubbed statement is saved" in str(classified)


@pytest.mark.parametrize(("status", "code", "kind", "category"), [
    (400, "invalid_request_error", "invalid_request_error", "bad_request"),
    (429, "rate_limit_exceeded", "rate_limit_error", "rate_limited"),
    (503, "server_error", "server_error", "provider_server_error"),
])
def test_diagnostics_show_only_http_status_and_allowlisted_metadata(status, code, kind, category):
    error = type("ProviderError", (Exception,), {
        "status_code": status,
        "body": {
            "error": {
                "code": code,
                "type": kind,
                "message": "PRIVATE_PROVIDER_BODY statement-text=SAFE_TEST api-key=secret",
            }
        },
    })("PRIVATE_EXCEPTION_MESSAGE")
    classified = classify_provider_error(error, chunk_index=2, chunk_count=3)
    text = str(classified)
    assert f"HTTP {status}" in text
    assert f"category={category}" in text
    assert f"code={code}" in text
    assert f"type={kind}" in text
    assert "segment 2 of 3" in text
    for secret in ("PRIVATE_PROVIDER_BODY", "PRIVATE_EXCEPTION_MESSAGE", "SAFE_TEST", "secret"):
        assert secret not in text


def test_unknown_provider_codes_and_types_are_not_echoed():
    error = type("ProviderError", (Exception,), {
        "status_code": 400,
        "body": {"error": {
            "code": "customer-id-123",
            "type": "sensitive-custom-type",
            "message": "private provider message",
        }},
    })("private exception message")
    text = str(classify_provider_error(error))
    assert "HTTP 400" in text
    assert "category=bad_request" in text
    assert "customer-id-123" not in text
    assert "sensitive-custom-type" not in text
    assert "private" not in text


def test_unknown_exception_includes_safe_class_and_next_step_only():
    error = type("UnexpectedSDKFailure", (Exception,), {})(
        "statement text and api key must never be shown"
    )
    classified = classify_provider_error(error)
    text = str(classified)
    assert "category=provider_error" in text
    assert "exception=UnexpectedSDKFailure" in text
    assert "next_action=check server logs and provider configuration" in text
    assert "statement text" not in text
    assert "api key" not in text


def test_api_status_error_reads_status_from_response_and_omits_unknown_code():
    error = type("APIStatusError", (Exception,), {
        "response": SimpleNamespace(status_code=400),
        "body": {"error": {
            "code": "unrecognized-sensitive-code",
            "message": "provider body containing statement content",
        }},
    })("private APIStatusError text")
    text = str(classify_provider_error(error))
    assert "HTTP 400" in text
    assert "category=bad_request" in text
    assert "exception=APIStatusError" in text
    assert "next_action=check model/request compatibility" in text
    for private in ("unrecognized-sensitive-code", "provider body", "statement content", "private APIStatusError"):
        assert private not in text


def test_api_connection_error_class_is_recognized_without_provider_message():
    error = type("APIConnectionError", (Exception,), {})(
        "private host, request contents, and credentials"
    )
    classified = classify_provider_error(error)
    text = str(classified)
    assert classified.status_code == 503
    assert "category=network_error" in text
    assert "exception=APIConnectionError" in text
    assert "next_action=check server connectivity" in text
    assert "private host" not in text
    assert "credentials" not in text


@pytest.mark.parametrize(
    ("finish_reason", "refusal", "expected_finish", "expected_refusal", "category"),
    [
        ("length", None, "length", "no", "output_length"),
        ("stop", "private refusal detail containing statement text", "stop", "yes", "empty_response"),
        ("private-provider-value", None, "unknown", "no", "empty_response"),
    ],
)
def test_empty_provider_content_has_safe_diagnostic_and_segment(
    monkeypatch, finish_reason, refusal, expected_finish, expected_refusal, category
):
    class FakeGroq:
        def __init__(self, api_key):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=lambda **_: SimpleNamespace(
                    choices=[SimpleNamespace(
                        finish_reason=finish_reason,
                        message=SimpleNamespace(content=None, refusal=refusal),
                    )]
                ))
            )

    monkeypatch.setenv("GROQ_API_KEY", "synthetic-test-key")
    monkeypatch.setitem(sys.modules, "groq", SimpleNamespace(Groq=FakeGroq))
    with pytest.raises(ProviderRequestError) as error:
        _completion([{"role": "user", "content": "SAFE_SYNTHETIC_SOURCE"}], 4, 9)

    text = str(error.value)
    assert error.value.status_code == 502
    assert error.value.retryable is False
    assert category in text
    assert "no usable" in text or "completion limit was reached" in text
    assert "segment 4 of 9" in text
    assert f"category={category}" in text
    assert "exception=EmptyProviderResponse" in text
    assert f"finish_reason={expected_finish}" in text
    assert f"refusal_present={expected_refusal}" in text
    assert "next_action=allow a smaller-chunk retry or check model output compatibility" in text
    for private in (
        "SAFE_SYNTHETIC_SOURCE",
        "private refusal detail",
        "statement text",
        "synthetic-test-key",
    ):
        assert private not in text


def test_empty_plain_json_recovery_response_preserves_safe_metadata_once(monkeypatch):
    calls = []

    class FakeGroq:
        def __init__(self, api_key):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self.create)
            )

        def create(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise type("BadRequestError", (Exception,), {"status_code": 400})(
                    "failed_generation json_validate_failed"
                )
            return SimpleNamespace(choices=[SimpleNamespace(
                finish_reason="length",
                message=SimpleNamespace(content="", refusal=None),
            )])

    monkeypatch.setenv("GROQ_API_KEY", "synthetic-test-key")
    monkeypatch.setitem(sys.modules, "groq", SimpleNamespace(Groq=FakeGroq))
    with pytest.raises(ProviderRequestError) as error:
        _completion([{"role": "user", "content": "SAFE_SYNTHETIC_SOURCE"}], 2, 6)

    text = str(error.value)
    assert len(calls) == 2
    assert "single plain-JSON recovery attempt failed" in text
    assert "segment 2 of 6" in text
    assert "category=output_length" in text
    assert "finish_reason=length" in text
    assert "refusal_present=no" in text
    assert "SAFE_SYNTHETIC_SOURCE" not in text
    assert "synthetic-test-key" not in text


def test_provider_sdk_exception_is_mapped_before_reaching_api(monkeypatch):
    class FakeGroq:
        def __init__(self, api_key):
            assert api_key == "synthetic-test-key"
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **_: (_ for _ in ()).throw(
                        type("RateLimitError", (Exception,), {"status_code": 429})("rate limit")
                    )
                )
            )

    monkeypatch.setenv("GROQ_API_KEY", "synthetic-test-key")
    monkeypatch.setitem(sys.modules, "groq", SimpleNamespace(Groq=FakeGroq))
    with pytest.raises(ProviderRequestError) as error:
        _completion([])
    assert error.value.status_code == 429
    assert "wait and retry" in str(error.value)


def test_groq_failed_generation_400_has_safe_json_mode_guidance_and_segment():
    error = type("BadRequestError", (Exception,), {"status_code": 400})(
        "failed_generation: json_validate_failed"
    )
    classified = classify_provider_error(error, chunk_index=3, chunk_count=7)
    assert classified.status_code == 502
    assert classified.retryable is True
    assert "required JSON response" in str(classified)
    assert "segment 3 of 7" in str(classified)
    assert "response mode" in str(classified)
    assert "failed_generation" not in str(classified)


def test_groq_json_validation_message_without_status_is_classified():
    classified = classify_provider_error(
        RuntimeError("json_validate_failed"),
        chunk_index=2,
        chunk_count=2,
    )
    assert "segment 2 of 2" in str(classified)
    assert classified.retryable is True


def test_json_validation_failure_gets_exactly_one_compact_recovery_retry(monkeypatch):
    calls = []

    class FakeGroq:
        def __init__(self, api_key):
            assert api_key == "synthetic-test-key"
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self.create)
            )

        def create(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise type("BadRequestError", (Exception,), {"status_code": 400})(
                    "failed_generation: json_validate_failed"
                )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"answer":"ok","evidence":["source quote"],"refuse":false}'))]
            )

    monkeypatch.setenv("GROQ_API_KEY", "synthetic-test-key")
    monkeypatch.setitem(sys.modules, "groq", SimpleNamespace(Groq=FakeGroq))
    result = _completion([{"role": "user", "content": "scrubbed source quote"}], 2, 4)
    assert result.startswith("{")
    assert len(calls) == 2
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert "response_format" not in calls[1]
    retry_messages = calls[1]["messages"]
    assert "scrubbed source quote" in retry_messages[0]["content"]
    assert "Return only one valid JSON object" in retry_messages[-1]["content"]
    assert calls[0]["messages"] == calls[1]["messages"][:-1]


def test_plain_json_fallback_response_is_still_strictly_grounded(monkeypatch):
    calls = []
    source = "Foodpanda lunch | PKR 120"

    class FakeGroq:
        def __init__(self, api_key):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise type("BadRequestError", (Exception,), {"status_code": 400})(
                    "failed_generation json_validate_failed"
                )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
                    "answer": "Foodpanda was PKR 120.",
                    "evidence": ["Foodpanda lunch | PKR 120"],
                    "refuse": False,
                })))]
            )

    monkeypatch.setenv("GROQ_API_KEY", "synthetic-test-key")
    monkeypatch.setitem(sys.modules, "groq", SimpleNamespace(Groq=FakeGroq))
    result = analyze(source)
    assert "Foodpanda was PKR 120." in result
    assert "Evidence:\n- Foodpanda lunch | PKR 120" in result
    assert len(calls) == 2
    assert "response_format" not in calls[1]


def test_analysis_preview_payload_matches_actual_chunked_groq_requests(monkeypatch):
    monkeypatch.setattr(llm, "ANALYSIS_CHUNK_CHARS", 32)
    source = (
        "Sort code 40-22-19 | Foodpanda | GBP 12.00\n"
        "Netflix subscription | GBP 8.00\n"
    )
    preview = preview_analysis_payloads(source)
    assert len(preview) > 1
    observed = []

    class FakeGroq:
        def __init__(self, api_key):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self.create)
            )

        def create(self, **kwargs):
            observed.append(kwargs)
            user_text = kwargs["messages"][-1]["content"]
            chunk = user_text.split(
                "Include exact evidence quotes from this segment.\n\n", 1
            )[1]
            return SimpleNamespace(choices=[SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=json.dumps({
                    "answer": "A listed transaction is present.",
                    "evidence": [chunk.strip()],
                    "refuse": False,
                }), refusal=None),
            )])

    monkeypatch.setenv("GROQ_API_KEY", "synthetic-test-key")
    monkeypatch.setitem(sys.modules, "groq", SimpleNamespace(Groq=FakeGroq))
    result = analyze(source)
    assert "listed transaction" in result
    assert observed == [segment["primary"] for segment in preview]
    serialized = json.dumps(preview)
    assert "40-22-19" not in serialized
    assert "<SORT_CODE>" in serialized
    assert all(segment["json_recovery"]["messages"][-1]["content"].startswith("Retry once.") for segment in preview)


def test_analysis_preview_matches_actual_json_recovery_request(monkeypatch):
    source = "Foodpanda lunch | GBP 12.00"
    preview = preview_analysis_payloads(source)[0]
    observed = []

    class FakeGroq:
        def __init__(self, api_key):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self.create)
            )

        def create(self, **kwargs):
            observed.append(kwargs)
            if len(observed) == 1:
                raise type("BadRequestError", (Exception,), {"status_code": 400})(
                    "failed_generation json_validate_failed"
                )
            return SimpleNamespace(choices=[SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=json.dumps({
                    "answer": "A listed transaction is present.",
                    "evidence": [source],
                    "refuse": False,
                }), refusal=None),
            )])

    monkeypatch.setenv("GROQ_API_KEY", "synthetic-test-key")
    monkeypatch.setitem(sys.modules, "groq", SimpleNamespace(Groq=FakeGroq))
    result = analyze(source)
    assert "listed transaction" in result
    assert observed == [preview["primary"], preview["json_recovery"]]


def test_chat_preview_matches_shared_builder_and_scrubs_question_and_history():
    source = "Foodpanda lunch | GBP 12.00"
    preview = preview_chat_payloads(
        source,
        "What happened to sort code 40-22-19?",
        [
            {"role": "user", "content": "Prior question includes 40-22-19"},
            {"role": "assistant", "content": "A safe reply."},
        ],
    )
    assert len(preview) == 1
    payload = preview[0]["primary"]
    serialized = json.dumps(payload)
    assert "40-22-19" not in serialized
    assert "<SORT_CODE>" in serialized
    assert "Foodpanda lunch | GBP 12.00" in serialized
    assert "A safe reply." in serialized
    assert payload == llm.build_chat_request_payload(
        "Foodpanda lunch | GBP 12.00",
        llm.scrub_and_check("What happened to sort code 40-22-19?"),
        llm._safe_chat_history([
            {"role": "user", "content": "Prior question includes 40-22-19"},
            {"role": "assistant", "content": "A safe reply."},
        ]),
        1,
        1,
    )


def test_chat_preview_payload_equals_actual_mocked_groq_request(monkeypatch):
    source = "Foodpanda lunch | GBP 12.00"
    question = "What was this purchase?"
    history = [{"role": "assistant", "content": "A safe prior answer."}]
    preview = preview_chat_payloads(source, question, history)
    observed = []

    class FakeGroq:
        def __init__(self, api_key):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self.create)
            )

        def create(self, **kwargs):
            observed.append(kwargs)
            chunk = kwargs["messages"][-1]["content"].split(
                "SOURCE SEGMENT:\n", 1
            )[1].split("\nQUESTION:\n", 1)[0]
            return SimpleNamespace(choices=[SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=json.dumps({
                    "answer": "A transaction is listed.",
                    "evidence": [chunk],
                    "refuse": False,
                }), refusal=None),
            )])

    monkeypatch.setenv("GROQ_API_KEY", "synthetic-test-key")
    monkeypatch.setitem(sys.modules, "groq", SimpleNamespace(Groq=FakeGroq))
    result = chat(source, question, history)
    assert "transaction is listed" in result
    assert observed == [segment["primary"] for segment in preview]


@pytest.mark.parametrize("fallback_content", [
    "not json",
    '{"answer":"The total was 999.","evidence":["Foodpanda lunch | PKR 120"],"refuse":false}',
    '{"answer":"A claim.","evidence":["invented transaction"],"refuse":false}',
])
def test_invalid_or_ungrounded_plain_json_fallback_is_rejected(monkeypatch, fallback_content):
    calls = []
    source = "SAFE_SYNTHETIC_SOURCE Foodpanda lunch | PKR 120"

    class FakeGroq:
        def __init__(self, api_key):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise type("BadRequestError", (Exception,), {"status_code": 400})(
                    "failed_generation json_validate_failed"
                )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=fallback_content))]
            )

    monkeypatch.setenv("GROQ_API_KEY", "synthetic-test-key")
    monkeypatch.setitem(sys.modules, "groq", SimpleNamespace(Groq=FakeGroq))
    with pytest.raises(AnalysisIncomplete) as error:
        analyze(source)
    assert len(calls) == 2
    assert "segment 1 of 1" in str(error.value)
    assert "SAFE_SYNTHETIC_SOURCE" not in str(error.value)
    assert fallback_content not in str(error.value)


def test_failed_json_recovery_does_not_loop_and_omits_provider_payload(monkeypatch):
    calls = []

    class FakeGroq:
        def __init__(self, api_key):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            calls.append(kwargs)
            raise type("BadRequestError", (Exception,), {"status_code": 400})(
                "failed_generation json_validate_failed confidential-provider-body"
            )

    monkeypatch.setenv("GROQ_API_KEY", "synthetic-test-key")
    monkeypatch.setitem(sys.modules, "groq", SimpleNamespace(Groq=FakeGroq))
    with pytest.raises(ProviderRequestError) as error:
        _completion([{"role": "user", "content": "scrubbed source quote"}], 3, 5)
    assert len(calls) == 2
    assert "segment 3 of 5" in str(error.value)
    assert "check model response-format compatibility" in str(error.value).lower()
    assert "confidential-provider-body" not in str(error.value)
    assert "scrubbed source quote" not in str(error.value)


@pytest.mark.parametrize(("fallback_error", "status", "category", "safe_code"), [
    (
        type("BadRequestError", (Exception,), {
            "status_code": 400,
            "body": {"error": {"code": "invalid_request_error", "message": "secret request body"}},
        })("private statement and key"),
        502,
        "bad_request",
        "code=invalid_request_error",
    ),
    (
        type("RateLimitError", (Exception,), {
            "status_code": 429,
            "body": {"error": {"code": "rate_limit_exceeded", "message": "secret"}},
        })("private statement"),
        429,
        "rate_limited",
        "code=rate_limit_exceeded",
    ),
    (
        type("ServerError", (Exception,), {
            "status_code": 503,
            "body": {"error": {"type": "server_error", "message": "secret"}},
        })("private statement"),
        503,
        "provider_server_error",
        "type=server_error",
    ),
    (ConnectionError("private statement and key"), 503, "network_error", None),
    (
        type("APIStatusError", (Exception,), {
            "response": SimpleNamespace(status_code=400),
            "body": {"error": {"code": "unrecognized-code", "message": "private body"}},
        })("private status error"),
        502,
        "bad_request",
        None,
    ),
    (
        type("APIConnectionError", (Exception,), {})("private network error"),
        503,
        "network_error",
        None,
    ),
])
def test_plain_json_fallback_failure_includes_safe_diagnostic(
    monkeypatch, fallback_error, status, category, safe_code
):
    calls = []

    class FakeGroq:
        def __init__(self, api_key):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise type("BadRequestError", (Exception,), {"status_code": 400})(
                    "failed_generation json_validate_failed"
                )
            raise fallback_error

    monkeypatch.setenv("GROQ_API_KEY", "synthetic-test-key")
    monkeypatch.setitem(sys.modules, "groq", SimpleNamespace(Groq=FakeGroq))
    with pytest.raises(ProviderRequestError) as error:
        _completion([{"role": "user", "content": "SAFE_SYNTHETIC_STATEMENT"}], 1, 1)
    text = str(error.value)
    assert len(calls) == 2
    assert error.value.status_code == status
    assert f"category={category}" in text
    assert f"exception={type(fallback_error).__name__}" in text
    if safe_code:
        assert safe_code in text
    for secret in ("SAFE_SYNTHETIC_STATEMENT", "private statement", "secret", "synthetic-test-key"):
        assert secret not in text


def test_grounding_failure_identifies_segment_without_echoing_source(monkeypatch):
    monkeypatch.setattr(llm, "ANALYSIS_CHUNK_CHARS", 24)
    source = "UNIQUE_PRIVATE_TEXT transaction 120\nAnother line\n"

    def completion(*_):
        return json.dumps({"answer": "Unsupported.", "evidence": ["invented"], "refuse": False})

    monkeypatch.setattr(llm, "_completion", completion)
    with pytest.raises(AnalysisIncomplete) as error:
        analyze(source)
    assert "segment 1 of" in str(error.value)
    assert "UNIQUE_PRIVATE_TEXT" not in str(error.value)
    assert "invented" not in error.value.analysis


def test_retry_endpoint_reuses_saved_scrubbed_statement_without_creating_record(monkeypatch):
    statement = SimpleNamespace(
        id="saved-id",
        source_names="statement.txt",
        scrubbed_text="scrubbed-only body",
        analysis=None,
        messages=[],
    )
    calls = {"get": 0, "add": 0, "commit": 0}

    class FakeDB:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def get(self, model, key):
            calls["get"] += 1
            assert model is main.Statement and key == "saved-id"
            return statement

        def add(self, _record):
            calls["add"] += 1

        def commit(self):
            calls["commit"] += 1

    seen = []
    monkeypatch.setattr(main, "analyze", lambda text: seen.append(text) or "new analysis")
    response = main.retry_analysis("saved-id", db=FakeDB())
    assert seen == ["scrubbed-only body"]
    assert response["analysis"] == "new analysis"
    assert response["analysis_status"] == "complete"
    assert calls == {"get": 1, "add": 0, "commit": 2}


def test_retry_endpoint_persists_and_reports_only_validated_partial_analysis(monkeypatch):
    statement = SimpleNamespace(
        id="partial-id",
        source_names="statement.txt",
        scrubbed_text="scrubbed statement",
        analysis=None,
        messages=[],
    )
    commits = []

    class FakeDB:
        def get(self, *_):
            return statement

        def commit(self):
            commits.append(statement.analysis)

    partial = (
        "[ANALYSIS INCOMPLETE] Segment 2 of 2 did not produce a complete response "
        "that passed every validation check. No text from that response was accepted.\n\n"
        "Previously completed, validated segment results only:\n"
        "A listed transaction is present.\n\nEvidence:\n- Merchant Alpha | GBP 12"
    )
    monkeypatch.setattr(
        main,
        "analyze",
        lambda _source: (_ for _ in ()).throw(
            AnalysisIncomplete(
                partial,
                2,
                2,
                "model response was incomplete",
                "invalid_json",
                [{
                    "segment": 1,
                    "segment_count": 2,
                    "status": "complete",
                    "validated_subchunks": 1,
                    "failed_subchunks": 0,
                    "categories": [],
                }, {
                    "segment": 2,
                    "segment_count": 2,
                    "status": "partial",
                    "validated_subchunks": 1,
                    "failed_subchunks": 1,
                    "categories": ["invalid_json"],
                }],
            )
        ),
    )
    response = main.retry_analysis("partial-id", db=FakeDB())
    assert response["analysis_status"] == "incomplete"
    assert response["analysis"] == partial
    assert response["analysis_segments"][1]["status"] == "partial"
    assert response["message"].startswith(
        "Analysis incomplete at segment 2 of 2 (reason: invalid_json)."
    )
    assert "model response was incomplete" in response["message"]
    assert commits == [None, partial]


def test_retry_rescrubs_legacy_record_and_persists_before_provider(monkeypatch):
    statement = SimpleNamespace(
        id="legacy-id",
        source_names="statement.txt",
        scrubbed_text="Sort code 40-22-19 | Foodpanda | GBP 12.00",
        analysis="Old analysis citing sort code 40-22-19",
        messages=[],
    )
    saved_before_analysis = []

    class FakeDB:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def get(self, *_):
            return statement

        def commit(self):
            saved_before_analysis.append((statement.scrubbed_text, statement.analysis))


    def fake_analyze(source):
        assert "<SORT_CODE>" in source
        assert "40-22-19" not in source
        assert saved_before_analysis[0][0] == source
        assert saved_before_analysis[0][1] is None
        return "Safe analysis"

    monkeypatch.setattr(main, "analyze", fake_analyze)
    response = main.retry_analysis("legacy-id", db=FakeDB())
    assert "40-22-19" not in response["scrubbed_text"]
    assert "<SORT_CODE>" in response["scrubbed_text"]
    assert response["analysis"] == "Safe analysis"


def test_saved_groq_preview_rescrubs_database_record_without_provider_call(monkeypatch):
    statement = SimpleNamespace(
        id="preview-id",
        source_names="statement.txt",
        scrubbed_text="Sort code 40-22-19 | Foodpanda | GBP 12.00",
        analysis="Old analysis",
        messages=[],
    )
    commits = []

    class FakeDB:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def get(self, model, key):
            assert model is main.Statement
            assert key == "preview-id"
            return statement

        def commit(self):
            commits.append((statement.scrubbed_text, statement.analysis))


    def forbidden_provider_call(*_args, **_kwargs):
        raise AssertionError("preview must not call Groq")

    monkeypatch.setattr(main, "analyze", forbidden_provider_call)
    monkeypatch.setattr(llm, "_completion", forbidden_provider_call)
    response = main.get_groq_preview("preview-id", db=FakeDB())

    assert commits[0][1] is None
    assert "<SORT_CODE>" in commits[0][0]
    assert response["id"] == "preview-id"
    assert response["files"] == "statement.txt"
    assert "40-22-19" not in json.dumps(response)
    assert "Foodpanda | GBP 12.00" in response["scrubbed_text"]
    assert response["provider_payloads"][0]["primary"]["model"] == llm.MODEL
    assert response["provider_request_estimate"]["initial_requests"] == 1
    assert response["provider_request_estimate"]["including_one_json_fallback_per_segment"] == 2


def test_upload_preview_saves_scrubbed_text_without_provider_call(monkeypatch):
    from io import BytesIO

    upload = SimpleNamespace(filename="synthetic.txt", file=BytesIO(
        b"Sort code 40-22-19 | Foodpanda | GBP 12.00"
    ))
    saved = []

    class FakeDB:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def add(self, record):
            saved.append(record)
            record.id = "new-preview-id"
            record.messages = []

        def commit(self):
            pass

        def refresh(self, record):
            record.id = "new-preview-id"

    monkeypatch.setattr(llm, "_completion", lambda *_args, **_kwargs: pytest.fail("Groq called"))
    response = main.upload_and_preview([upload], db=FakeDB())
    assert response["id"] == "new-preview-id"
    assert "40-22-19" not in response["scrubbed_text"]
    assert "<SORT_CODE>" in response["scrubbed_text"]
    assert saved[0].scrubbed_text == response["scrubbed_text"]
    assert "provider_payloads" in response
    assert response["provider_request_estimate"]["hard_run_limit"] == 64


def test_chat_preview_api_includes_sanitized_history_without_provider_call(monkeypatch):
    statement = SimpleNamespace(
        id="chat-preview-id",
        source_names="statement.txt",
        scrubbed_text="Foodpanda lunch | GBP 12.00",
        analysis=None,
        messages=[],
    )
    saved_chat = [
        SimpleNamespace(role="user", content="Earlier question includes 40-22-19"),
        SimpleNamespace(role="assistant", content="Prior answer."),
    ]

    class FakeQuery:
        def where(self, *_):
            return self

        def order_by(self, *_):
            return self

        def limit(self, *_):
            return self

        def all(self):
            return saved_chat

    class FakeDB:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def get(self, *_):
            return statement

        def scalars(self, *_):
            return FakeQuery()

        def commit(self):
            pass


    def forbidden_provider_call(*_args, **_kwargs):
        raise AssertionError("chat preview must not call Groq")

    monkeypatch.setattr(main, "chat", forbidden_provider_call)
    monkeypatch.setattr(llm, "_completion", forbidden_provider_call)
    response = main.preview_chat_request(
        "chat-preview-id",
        main.ChatRequest(question="What about sort code 40-22-19?"),
        db=FakeDB(),
    )
    serialized = json.dumps(response)
    assert "40-22-19" not in serialized
    assert "<SORT_CODE>" in serialized
    assert "Prior answer." in serialized
    assert response["provider_payloads"][0]["primary"]["model"] == llm.MODEL


def test_chat_preview_for_monthly_total_explains_local_limitation_without_groq(monkeypatch):
    statement = SimpleNamespace(
        id="monthly-id",
        source_names="synthetic.docx",
        scrubbed_text="2026-04-04 | Synthetic Market | GBP 12.00",
        analysis=None,
        messages=[],
    )

    class FakeDB:
        def get(self, *_):
            return statement

        def commit(self):
            pass

    monkeypatch.setattr(llm, "_completion", lambda *_: pytest.fail("Groq called"))
    response = main.preview_chat_request(
        "monthly-id",
        main.ChatRequest(question="What are monthly expenses of Marcus T. Alvarez?"),
        db=FakeDB(),
    )
    assert response["local_only"] is True
    assert response["provider_payloads"] == []
    assert "cannot reliably calculate monthly expense totals" in response["message"]
    assert "Marcus T. Alvarez" not in response["message"]


def test_chat_api_returns_local_aggregate_explanation_and_saves_turn(monkeypatch):
    statement = SimpleNamespace(
        id="monthly-chat-id",
        source_names="synthetic.docx",
        scrubbed_text="2026-04-04 | Synthetic Market | GBP 12.00",
        analysis=None,
        messages=[],
    )
    saved = []

    class EmptyQuery:
        def where(self, *_):
            return self

        def order_by(self, *_):
            return self

        def limit(self, *_):
            return self

        def all(self):
            return []

    class FakeDB:
        def get(self, *_):
            return statement

        def commit(self):
            pass

        def scalars(self, *_):
            return EmptyQuery()

        def add_all(self, messages):
            saved.extend(messages)

    monkeypatch.setattr(
        main,
        "chat",
        lambda source, question, history: llm.chat(source, question, history),
    )
    monkeypatch.setattr(llm, "_completion", lambda *_: pytest.fail("Groq called"))
    response = main.chat_about_statement(
        "monthly-chat-id",
        main.ChatRequest(question="What are monthly expenses of Marcus T. Alvarez?"),
        db=FakeDB(),
    )
    assert response["local_only"] is True
    assert "cannot reliably calculate monthly expense totals" in response["answer"]
    assert len(saved) == 2
    assert saved[1].content == response["answer"]


def test_chat_api_exposes_actionable_provider_error(monkeypatch):
    statement = SimpleNamespace(
        id="failed-chat-id",
        source_names="synthetic.txt",
        scrubbed_text="Synthetic transaction | GBP 12.00",
        analysis=None,
        messages=[],
    )

    class EmptyQuery:
        def where(self, *_):
            return self

        def order_by(self, *_):
            return self

        def limit(self, *_):
            return self

        def all(self):
            return []

    class FakeDB:
        def get(self, *_):
            return statement

        def commit(self):
            pass

        def scalars(self, *_):
            return EmptyQuery()

    safe_detail = "provider failure category=rate_limited; retry later"
    monkeypatch.setattr(main, "chat", lambda *_: (_ for _ in ()).throw(
        ProviderRequestError(safe_detail, status_code=429, category="rate_limited")
    ))
    with pytest.raises(main.HTTPException) as error:
        main.chat_about_statement(
            "failed-chat-id",
            main.ChatRequest(question="Which transaction is shown?"),
            db=FakeDB(),
        )
    assert error.value.status_code == 429
    assert error.value.detail == safe_detail


def test_legacy_sort_code_is_removed_before_any_outgoing_prompt(monkeypatch):
    legacy = "SORT 402219 | Foodpanda | 2026-04-02 | GBP 12.00"
    statement = SimpleNamespace(
        id="legacy-sort-id",
        source_names="statement.txt",
        scrubbed_text=legacy,
        analysis="Old analysis cites SORT 402219",
        messages=[],
    )
    commits = []

    class FakeDB:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def get(self, *_):
            return statement

        def commit(self):
            commits.append(statement.scrubbed_text)

    monkeypatch.setattr(main, "analyze", llm.analyze)
    sent_prompts = []

    def fake_completion(messages, *_):
        sent_prompts.extend(message["content"] for message in messages)
        source_chunk = messages[-1]["content"].split(
            "Include exact evidence quotes from this segment.\n\n", 1
        )[1]
        return json.dumps({
            "answer": "Foodpanda GBP 12.00.",
            "evidence": [source_chunk.strip()],
            "refuse": False,
        })

    monkeypatch.setattr(llm, "_completion", fake_completion)
    response = main.retry_analysis("legacy-sort-id", db=FakeDB())
    assert commits[0] == statement.scrubbed_text
    assert "402219" not in statement.scrubbed_text
    assert "<SORT_CODE>" in statement.scrubbed_text
    assert sent_prompts and all("402219" not in prompt for prompt in sent_prompts)
    assert "402219" not in response["analysis"]
    assert "402219" not in response["scrubbed_text"]


def test_direct_identifiers_are_absent_from_outgoing_prompt_and_citation(monkeypatch):
    source = (
        "Sort Code: 40-22-19 | Account number: 1234567890 | "
        "Phone: +92 300 1234567 | Email: alice@example.com | "
        "Foodpanda | 2026-04-04 | GBP 12.00 | Card ending 1234"
    )
    outgoing = []

    def fake_completion(messages, *_args, **_kwargs):
        outgoing.extend(message["content"] for message in messages)
        user_content = messages[-1]["content"]
        chunk = user_content.split(
            "Include exact evidence quotes from this segment.\n\n", 1
        )[1]
        return json.dumps({
            "answer": "Foodpanda was GBP 12.00.",
            "evidence": [chunk],
            "refuse": False,
        })

    monkeypatch.setattr(llm, "_completion", fake_completion)
    result = analyze(source)
    for private in (
        "40-22-19",
        "1234567890",
        "+92 300 1234567",
        "alice@example.com",
    ):
        assert all(private not in prompt for prompt in outgoing)
        assert private not in result
    assert "Foodpanda" in result
    assert "2026-04-04" in result
    assert "GBP 12.00" in result
    assert "1234" in result
    assert "<SORT_CODE>" in result
    assert "<ACCOUNT_ID>" in result
    assert "<PHONE>" in result
    assert "<EMAIL_OR_WALLET_ID>" in result


@pytest.mark.parametrize("candidate", [
    '{"answer":"It was PKR 9000.","evidence":["2026-04-02 | Foodpanda | PKR 1,200.00"],"refuse":false}',
    '{"answer":"A claim.","evidence":["invented source"],"refuse":false}',
    '{"answer":"A claim.","evidence":[],"refuse":false}',
    '{"answer":"A claim.","evidence":[],"refuse":"no"}',
    "not json",
])
def test_rejects_unverifiable_model_outputs(candidate):
    with pytest.raises(UngroundedResponse):
        validate_grounded_response(candidate, SOURCE)
