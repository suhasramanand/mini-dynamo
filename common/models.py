"""Pydantic request/response schemas shared across services."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# Client-facing (router)
# --------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    """A client request for text generation."""

    session_id: str = Field(..., description="Conversation/session id used for KV-cache sticky routing.")
    prompt: str = Field(..., description="Prompt text.")
    max_tokens: Optional[int] = Field(None, ge=1, le=4096)
    stream: bool = Field(True, description="Stream tokens via SSE.")


class TokenChunk(BaseModel):
    """One streamed token (or the terminal event)."""

    token: str = ""
    index: int = 0
    finish_reason: Optional[str] = None  # "stop" | "length" | None while streaming


# --------------------------------------------------------------------------
# Prefill stage
# --------------------------------------------------------------------------
class PrefillRequest(BaseModel):
    session_id: str
    prompt: str
    max_tokens: int = 64


class PrefillResponse(BaseModel):
    session_id: str
    kv_cache_id: str
    num_prompt_tokens: int
    kv_size_mb: float
    prefill_ms: float
    backend: str          # name of the prefill node that produced the KV cache
    evicted: int = 0      # number of cache entries evicted to make room


# --------------------------------------------------------------------------
# Decode stage
# --------------------------------------------------------------------------
class DecodeRequest(BaseModel):
    session_id: str
    kv_cache_id: str
    prompt: str = ""              # kept for colocated / re-prefill fallback
    num_prompt_tokens: int = 0
    kv_size_mb: float = 0.0
    max_tokens: int = 64
    prefill_backend: str = ""     # where the KV cache currently lives


# --------------------------------------------------------------------------
# Service discovery / health
# --------------------------------------------------------------------------
class BackendInfo(BaseModel):
    name: str
    url: str
    role: str                     # prefill | decode
    queue_depth: int = 0
    active_batch: int = 0
    mem_used_mb: float = 0.0
    mem_total_mb: float = 0.0
    utilization: float = 0.0      # 0..1

    @property
    def healthy(self) -> bool:
        return self.utilization < 0.999


class HealthResponse(BaseModel):
    status: str = "ok"
    role: str = ""
    name: str = ""
    mode: str = "disaggregated"
    mem_used_mb: float = 0.0
    mem_total_mb: float = 0.0
    utilization: float = 0.0
    queue_depth: int = 0
    active_batch: int = 0
