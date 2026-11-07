from __future__ import annotations

from typing import Any


class EmptyProviderResponse(RuntimeError):
    def __init__(self, finish_reason: str | None, refusal_present: bool):
        super().__init__("Provider returned no usable response content")
        allowed_finish_reasons = {
            "stop", "length", "content_filter", "tool_calls", "function_call",
        }
        self.finish_reason = (
            finish_reason
            if isinstance(finish_reason, str) and finish_reason in allowed_finish_reasons
            else "unknown"
        )
        self.refusal_present = bool(refusal_present)


def create_completion(payload: dict[str, Any]) -> str:
    """Make one Groq SDK call using a payload built by the analysis layer."""
    import os

    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured")
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(**payload)
    choice = response.choices[0]
    message = choice.message
    content = message.content
    if not content:
        raise EmptyProviderResponse(
            getattr(choice, "finish_reason", None),
            bool(getattr(message, "refusal", None)),
        )
    return content
