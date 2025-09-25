"""
Pydantic models for all Vercel AI SDK event types.

This module defines all the event types that can be streamed to the Vercel AI SDK frontend.
"""

from abc import ABC
from typing import Any, Dict, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StreamEvent(BaseModel, ABC):
    """
    Base class for all stream events.

    All events must have a 'type' field that identifies the event type.
    """

    model_config = ConfigDict(extra="forbid")

    type: str

    def to_sse(self) -> str:
        """
        Convert the event to SSE format.

        Returns:
            SSE formatted string with 'data:' prefix and newlines.
        """
        return f"data: {self.model_dump_json(exclude_none=True, by_alias=True)}\n\n"


# Message lifecycle events


class StartEvent(StreamEvent):
    """Event indicating the start of a message."""

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["start"] = "start"
    message_id: str = Field(..., alias="messageId")


class FinishEvent(StreamEvent):
    """Event indicating the end of a message."""

    type: Literal["finish"] = "finish"


# Text part events


class TextStartEvent(StreamEvent):
    """Event indicating the start of a text part."""

    type: Literal["text-start"] = "text-start"
    id: str


class TextDeltaEvent(StreamEvent):
    """Event containing a text delta."""

    type: Literal["text-delta"] = "text-delta"
    id: str
    delta: str


class TextEndEvent(StreamEvent):
    """Event indicating the end of a text part."""

    type: Literal["text-end"] = "text-end"
    id: str


# Reasoning part events


class ReasoningStartEvent(StreamEvent):
    """Event indicating the start of a reasoning part."""

    type: Literal["reasoning-start"] = "reasoning-start"
    id: str


class ReasoningDeltaEvent(StreamEvent):
    """Event containing a reasoning delta."""

    type: Literal["reasoning-delta"] = "reasoning-delta"
    id: str
    delta: str


class ReasoningEndEvent(StreamEvent):
    """Event indicating the end of a reasoning part."""

    type: Literal["reasoning-end"] = "reasoning-end"
    id: str


# Source reference events


class SourceURLEvent(StreamEvent):
    """Event containing a URL source reference."""

    type: Literal["source-url"] = "source-url"
    source_id: str = Field(..., alias="sourceId")
    url: str

    model_config = ConfigDict(populate_by_name=True)


class SourceDocumentEvent(StreamEvent):
    """Event containing a document source reference."""

    type: Literal["source-document"] = "source-document"
    source_id: str = Field(..., alias="sourceId")
    media_type: str = Field(..., alias="mediaType")
    title: str

    model_config = ConfigDict(populate_by_name=True)


# File reference event


class FileEvent(StreamEvent):
    """Event containing a file reference."""

    type: Literal["file"] = "file"
    url: str
    media_type: str = Field(..., alias="mediaType")

    model_config = ConfigDict(populate_by_name=True)


# Structured data event


class DataEvent(StreamEvent):
    """
    Event containing custom structured data.

    The type should follow the pattern 'data-{name}' where {name}
    identifies the type of data being sent.
    """

    data: Dict[str, Any]

    @field_validator("type")
    @classmethod
    def validate_data_type(cls, v: str) -> str:
        """Ensure type starts with 'data-'."""
        if not v.startswith("data-"):
            raise ValueError("DataEvent type must start with 'data-'")
        return v

    @classmethod
    def create(cls, name: str, data: Dict[str, Any]) -> "DataEvent":
        """
        Create a DataEvent with a specific name.

        Args:
            name: The name suffix for the data type (will be prefixed with 'data-')
            data: The structured data to send

        Returns:
            A DataEvent instance
        """
        return cls(type=f"data-{name}", data=data)


# Tool call events


class ToolInputStartEvent(StreamEvent):
    """Event indicating the start of tool input."""

    type: Literal["tool-input-start"] = "tool-input-start"
    tool_call_id: str = Field(..., alias="toolCallId")
    tool_name: str = Field(..., alias="toolName")

    model_config = ConfigDict(populate_by_name=True)


class ToolInputDeltaEvent(StreamEvent):
    """Event containing tool input delta."""

    type: Literal["tool-input-delta"] = "tool-input-delta"
    tool_call_id: str = Field(..., alias="toolCallId")
    input_text_delta: str = Field(..., alias="inputTextDelta")

    model_config = ConfigDict(populate_by_name=True)


class ToolInputAvailableEvent(StreamEvent):
    """Event indicating tool input is available."""

    type: Literal["tool-input-available"] = "tool-input-available"
    tool_call_id: str = Field(..., alias="toolCallId")
    tool_name: str = Field(..., alias="toolName")
    input: Dict[str, Any]

    model_config = ConfigDict(populate_by_name=True)


class ToolOutputAvailableEvent(StreamEvent):
    """Event indicating tool output is available."""

    type: Literal["tool-output-available"] = "tool-output-available"
    tool_call_id: str = Field(..., alias="toolCallId")
    output: Dict[str, Any]

    model_config = ConfigDict(populate_by_name=True)


# Step events


class StartStepEvent(StreamEvent):
    """Event indicating the start of a step."""

    type: Literal["start-step"] = "start-step"


class FinishStepEvent(StreamEvent):
    """Event indicating the end of a step."""

    type: Literal["finish-step"] = "finish-step"


# Error event


class ErrorEvent(StreamEvent):
    """Event containing an error message."""

    type: Literal["error"] = "error"
    error_text: str = Field(..., alias="errorText")

    model_config = ConfigDict(populate_by_name=True)


# Type aliases for convenience
AnyStreamEvent = Union[
    StartEvent,
    FinishEvent,
    TextStartEvent,
    TextDeltaEvent,
    TextEndEvent,
    ReasoningStartEvent,
    ReasoningDeltaEvent,
    ReasoningEndEvent,
    SourceURLEvent,
    SourceDocumentEvent,
    FileEvent,
    DataEvent,
    ToolInputStartEvent,
    ToolInputDeltaEvent,
    ToolInputAvailableEvent,
    ToolOutputAvailableEvent,
    StartStepEvent,
    FinishStepEvent,
    ErrorEvent,
]
