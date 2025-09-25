"""Tests for Pydantic models."""

import json

import pytest

from fastapi_ai_sdk.models import (
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
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ToolInputAvailableEvent,
    ToolInputDeltaEvent,
    ToolInputStartEvent,
    ToolOutputAvailableEvent,
)


class TestStreamEvent:
    """Tests for the base StreamEvent class."""

    def test_to_sse_format(self):
        """Test SSE formatting of events."""
        event = StartEvent(message_id="test_123")
        sse = event.to_sse()

        assert sse.startswith("data: ")
        assert sse.endswith("\n\n")
        assert '"type":"start"' in sse
        assert '"messageId":"test_123"' in sse

    def test_json_serialization(self):
        """Test JSON serialization."""
        event = TextDeltaEvent(id="txt_1", delta="Hello")
        json_str = event.model_dump_json(exclude_none=True)
        data = json.loads(json_str)

        assert data["type"] == "text-delta"
        assert data["id"] == "txt_1"
        assert data["delta"] == "Hello"


class TestMessageLifecycle:
    """Tests for message lifecycle events."""

    def test_start_event(self):
        """Test StartEvent creation and serialization."""
        event = StartEvent(message_id="msg_123")

        assert event.type == "start"
        assert event.message_id == "msg_123"

        data = event.model_dump(by_alias=True)
        assert data["messageId"] == "msg_123"

    def test_finish_event(self):
        """Test FinishEvent creation."""
        event = FinishEvent()

        assert event.type == "finish"
        data = event.model_dump()
        assert len(data) == 1  # Only type field


class TestTextEvents:
    """Tests for text part events."""

    def test_text_start(self):
        """Test TextStartEvent."""
        event = TextStartEvent(id="txt_1")

        assert event.type == "text-start"
        assert event.id == "txt_1"

    def test_text_delta(self):
        """Test TextDeltaEvent."""
        event = TextDeltaEvent(id="txt_1", delta="Hello world")

        assert event.type == "text-delta"
        assert event.id == "txt_1"
        assert event.delta == "Hello world"

    def test_text_end(self):
        """Test TextEndEvent."""
        event = TextEndEvent(id="txt_1")

        assert event.type == "text-end"
        assert event.id == "txt_1"


class TestReasoningEvents:
    """Tests for reasoning part events."""

    def test_reasoning_start(self):
        """Test ReasoningStartEvent."""
        event = ReasoningStartEvent(id="r_1")

        assert event.type == "reasoning-start"
        assert event.id == "r_1"

    def test_reasoning_delta(self):
        """Test ReasoningDeltaEvent."""
        event = ReasoningDeltaEvent(id="r_1", delta="Thinking about...")

        assert event.type == "reasoning-delta"
        assert event.id == "r_1"
        assert event.delta == "Thinking about..."

    def test_reasoning_end(self):
        """Test ReasoningEndEvent."""
        event = ReasoningEndEvent(id="r_1")

        assert event.type == "reasoning-end"
        assert event.id == "r_1"


class TestSourceEvents:
    """Tests for source reference events."""

    def test_source_url(self):
        """Test SourceURLEvent."""
        event = SourceURLEvent(source_id="src_1", url="https://example.com")

        assert event.type == "source-url"
        assert event.source_id == "src_1"
        assert event.url == "https://example.com"

        data = event.model_dump(by_alias=True)
        assert data["sourceId"] == "src_1"

    def test_source_document(self):
        """Test SourceDocumentEvent."""
        event = SourceDocumentEvent(
            source_id="src_2", media_type="application/pdf", title="Report.pdf"
        )

        assert event.type == "source-document"
        assert event.source_id == "src_2"
        assert event.media_type == "application/pdf"
        assert event.title == "Report.pdf"

        data = event.model_dump(by_alias=True)
        assert data["sourceId"] == "src_2"
        assert data["mediaType"] == "application/pdf"


class TestFileEvent:
    """Tests for file reference event."""

    def test_file_event(self):
        """Test FileEvent."""
        event = FileEvent(url="https://example.com/image.png", media_type="image/png")

        assert event.type == "file"
        assert event.url == "https://example.com/image.png"
        assert event.media_type == "image/png"

        data = event.model_dump(by_alias=True)
        assert data["mediaType"] == "image/png"


class TestDataEvent:
    """Tests for structured data event."""

    def test_data_event_creation(self):
        """Test DataEvent creation."""
        event = DataEvent(
            type="data-weather", data={"city": "Berlin", "temperature": 18}
        )

        assert event.type == "data-weather"
        assert event.data == {"city": "Berlin", "temperature": 18}

    def test_data_event_create_method(self):
        """Test DataEvent.create() helper."""
        event = DataEvent.create("weather", {"city": "Berlin", "temperature": 18})

        assert event.type == "data-weather"
        assert event.data["city"] == "Berlin"
        assert event.data["temperature"] == 18

    def test_data_event_validation(self):
        """Test DataEvent type validation."""
        with pytest.raises(ValueError, match="must start with 'data-'"):
            DataEvent(type="weather", data={"test": "value"})  # Missing 'data-' prefix


class TestToolEvents:
    """Tests for tool call events."""

    def test_tool_input_start(self):
        """Test ToolInputStartEvent."""
        event = ToolInputStartEvent(tool_call_id="call_123", tool_name="get_weather")

        assert event.type == "tool-input-start"
        assert event.tool_call_id == "call_123"
        assert event.tool_name == "get_weather"

        data = event.model_dump(by_alias=True)
        assert data["toolCallId"] == "call_123"
        assert data["toolName"] == "get_weather"

    def test_tool_input_delta(self):
        """Test ToolInputDeltaEvent."""
        event = ToolInputDeltaEvent(tool_call_id="call_123", input_text_delta="Berlin")

        assert event.type == "tool-input-delta"
        assert event.tool_call_id == "call_123"
        assert event.input_text_delta == "Berlin"

        data = event.model_dump(by_alias=True)
        assert data["toolCallId"] == "call_123"
        assert data["inputTextDelta"] == "Berlin"

    def test_tool_input_available(self):
        """Test ToolInputAvailableEvent."""
        event = ToolInputAvailableEvent(
            tool_call_id="call_123", tool_name="get_weather", input={"city": "Berlin"}
        )

        assert event.type == "tool-input-available"
        assert event.tool_call_id == "call_123"
        assert event.tool_name == "get_weather"
        assert event.input == {"city": "Berlin"}

    def test_tool_output_available(self):
        """Test ToolOutputAvailableEvent."""
        event = ToolOutputAvailableEvent(
            tool_call_id="call_123", output={"temperature": 18, "condition": "sunny"}
        )

        assert event.type == "tool-output-available"
        assert event.tool_call_id == "call_123"
        assert event.output["temperature"] == 18
        assert event.output["condition"] == "sunny"


class TestStepEvents:
    """Tests for step events."""

    def test_start_step(self):
        """Test StartStepEvent."""
        event = StartStepEvent()

        assert event.type == "start-step"

    def test_finish_step(self):
        """Test FinishStepEvent."""
        event = FinishStepEvent()

        assert event.type == "finish-step"


class TestErrorEvent:
    """Tests for error event."""

    def test_error_event(self):
        """Test ErrorEvent."""
        event = ErrorEvent(error_text="Something went wrong")

        assert event.type == "error"
        assert event.error_text == "Something went wrong"

        data = event.model_dump(by_alias=True)
        assert data["errorText"] == "Something went wrong"


class TestModelValidation:
    """Tests for model validation and edge cases."""

    def test_extra_fields_forbidden(self):
        """Test that extra fields are not allowed."""
        with pytest.raises(ValueError):
            StartEvent(message_id="test", extra_field="not_allowed")  # This should fail

    def test_field_aliases(self):
        """Test that field aliases work correctly."""
        # Test with snake_case
        event1 = StartEvent(message_id="test")
        # Test with camelCase
        event2 = StartEvent(messageId="test")

        assert event1.message_id == "test"
        assert event2.message_id == "test"

        # Both should serialize to camelCase
        data1 = event1.model_dump(by_alias=True)
        data2 = event2.model_dump(by_alias=True)

        assert data1["messageId"] == "test"
        assert data2["messageId"] == "test"

    def test_none_exclusion(self):
        """Test that None values are excluded from serialization."""
        event = StartEvent(message_id="test")
        json_str = event.model_dump_json(exclude_none=True)

        # Should not contain any null values
        assert "null" not in json_str
