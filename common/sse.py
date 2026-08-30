"""Minimal Server-Sent Events helpers (no extra dependency)."""

from __future__ import annotations

import json
from typing import Any, Dict

from .models import TokenChunk


def sse(data: Dict[str, Any]) -> str:
    """Format a dict as a single SSE ``data:`` frame."""
    return f"data: {json.dumps(data, separators=(',', ':'))}\n\n"


def sse_chunk(chunk: TokenChunk) -> str:
    return sse(chunk.model_dump())


def sse_done() -> str:
    """Terminal frame following the OpenAI streaming convention."""
    return "data: [DONE]\n\n"
