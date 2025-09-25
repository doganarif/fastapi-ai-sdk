"""
FastAPI AI SDK - A Pythonic helper library for Vercel AI SDK backend implementation.

This library provides a complete set of utilities for building FastAPI backends
that are compatible with the Vercel AI SDK frontend.
"""

from .decorators import ai_endpoint, streaming_endpoint, tool_endpoint
from .models import (  # Message lifecycle; Text parts; Reasoning parts; Source references; File reference; Structured data; Tool calls; Steps; Error; Base
    DataEvent,
    ErrorEvent,
    FileEvent,
    FinishEvent,
    FinishStepEvent,
    ReasoningDeltaEvent,
    ReasoningEndEvent,
    ReasoningStartEvent,
    SourceDocumentEvent,
    SourceURLEvent,
    StartEvent,
    StartStepEvent,
    StreamEvent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ToolInputAvailableEvent,
    ToolInputDeltaEvent,
    ToolInputStartEvent,
    ToolOutputAvailableEvent,
)
from .response import create_ai_stream_response
from .stream import AIStream, AIStreamBuilder

__version__ = "0.1.0"

__all__ = [
    # Models
    "StartEvent",
    "FinishEvent",
    "TextStartEvent",
    "TextDeltaEvent",
    "TextEndEvent",
    "ReasoningStartEvent",
    "ReasoningDeltaEvent",
    "ReasoningEndEvent",
    "SourceURLEvent",
    "SourceDocumentEvent",
    "FileEvent",
    "DataEvent",
    "ToolInputStartEvent",
    "ToolInputDeltaEvent",
    "ToolInputAvailableEvent",
    "ToolOutputAvailableEvent",
    "StartStepEvent",
    "FinishStepEvent",
    "ErrorEvent",
    "StreamEvent",
    # Stream utilities
    "AIStreamBuilder",
    "AIStream",
    # Response helpers
    "create_ai_stream_response",
    # Decorators
    "ai_endpoint",
    "streaming_endpoint",
    "tool_endpoint",
]
