"""Tests for stream building and management."""

import json

import pytest

from fastapi_ai_sdk.models import (
    ErrorEvent,
    FinishEvent,
    StartEvent,
    TextDeltaEvent,
)
from fastapi_ai_sdk.stream import (
    AIStream,
    AIStreamBuilder,
    create_simple_text_stream,
)


class TestAIStreamBuilder:
    """Tests for AIStreamBuilder class."""

    def test_builder_initialization(self):
        """Test builder initialization."""
        builder = AIStreamBuilder("test_msg_123")

        assert builder._message_id == "test_msg_123"
        assert not builder._started
        assert not builder._finished
        assert len(builder._events) == 0

    def test_auto_generated_message_id(self):
        """Test automatic message ID generation."""
        builder = AIStreamBuilder()

        assert builder._message_id.startswith("msg_")
        assert len(builder._message_id) > 4

    def test_start_method(self):
        """Test start method."""
        builder = AIStreamBuilder("test_123")
        builder.start()

        assert builder._started
        assert len(builder._events) == 1
        assert isinstance(builder._events[0], StartEvent)
        assert builder._events[0].message_id == "test_123"

    def test_start_called_twice(self):
        """Test that calling start twice raises an error."""
        builder = AIStreamBuilder()
        builder.start()

        with pytest.raises(RuntimeError, match="already been started"):
            builder.start()

    def test_finish_method(self):
        """Test finish method."""
        builder = AIStreamBuilder()
        builder.start().finish()

        assert builder._finished
        assert isinstance(builder._events[-1], FinishEvent)

    def test_finish_called_twice(self):
        """Test that calling finish twice raises an error."""
        builder = AIStreamBuilder()
        builder.start().finish()

        with pytest.raises(RuntimeError, match="already been finished"):
            builder.finish()

    def test_text_method(self):
        """Test text method."""
        builder = AIStreamBuilder()
        builder.text("Hello world", text_id="txt_1")

        # Should have TextStartEvent, TextDeltaEvent, TextEndEvent
        assert len(builder._events) == 3
        assert builder._events[0].type == "text-start"
        assert builder._events[0].id == "txt_1"
        assert builder._events[1].type == "text-delta"
        assert builder._events[1].delta == "Hello world"
        assert builder._events[2].type == "text-end"

    def test_text_with_chunking(self):
        """Test text method with chunking."""
        builder = AIStreamBuilder()
        builder.text("Hello world", chunk_size=5)

        # Should have start, 3 deltas (5+5+1 chars), and end
        text_deltas = [e for e in builder._events if e.type == "text-delta"]
        assert len(text_deltas) == 3
        assert text_deltas[0].delta == "Hello"
        assert text_deltas[1].delta == " worl"
        assert text_deltas[2].delta == "d"

    def test_reasoning_method(self):
        """Test reasoning method."""
        builder = AIStreamBuilder()
        builder.reasoning("Thinking about the answer...", reasoning_id="r_1")

        # Should have ReasoningStartEvent, ReasoningDeltaEvent, ReasoningEndEvent
        assert len(builder._events) == 3
        assert builder._events[0].type == "reasoning-start"
        assert builder._events[0].id == "r_1"
        assert builder._events[1].type == "reasoning-delta"
        assert builder._events[1].delta == "Thinking about the answer..."
        assert builder._events[2].type == "reasoning-end"

    def test_data_method(self):
        """Test data method."""
        builder = AIStreamBuilder()
        builder.data("weather", {"city": "Berlin", "temp": 18})

        assert len(builder._events) == 1
        event = builder._events[0]
        assert event.type == "data-weather"
        assert event.data == {"city": "Berlin", "temp": 18}

    def test_tool_call_method(self):
        """Test tool_call method."""
        builder = AIStreamBuilder()
        builder.tool_call(
            "get_weather",
            {"city": "Berlin"},
            {"temperature": 18},
            tool_call_id="call_1",
        )

        # Should have start, available input, and output events
        assert len(builder._events) >= 3
        assert builder._events[0].type == "tool-input-start"
        assert builder._events[0].tool_call_id == "call_1"

        # Find input available event
        input_event = next(
            e for e in builder._events if e.type == "tool-input-available"
        )
        assert input_event.input == {"city": "Berlin"}

        # Find output event
        output_event = next(
            e for e in builder._events if e.type == "tool-output-available"
        )
        assert output_event.output == {"temperature": 18}

    def test_tool_call_streaming_input(self):
        """Test tool_call with streaming input."""
        builder = AIStreamBuilder()
        builder.tool_call(
            "get_weather", {"city": "Berlin"}, stream_input=True, tool_call_id="call_1"
        )

        # Should have delta events for streaming input
        delta_events = [e for e in builder._events if e.type == "tool-input-delta"]
        assert len(delta_events) > 0

        # Concatenate all deltas should form the JSON
        all_deltas = "".join(e.input_text_delta for e in delta_events)
        assert json.loads(all_deltas) == {"city": "Berlin"}

    def test_step_method(self):
        """Test step method."""
        builder = AIStreamBuilder()

        def step_func(b: AIStreamBuilder):
            b.text("Inside step")

        builder.step(step_func)

        # Should have start step, text events, finish step
        assert builder._events[0].type == "start-step"
        assert builder._events[-1].type == "finish-step"
        # Text events in between
        text_events = [e for e in builder._events if "text" in e.type]
        assert len(text_events) > 0

    def test_error_method(self):
        """Test error method."""
        builder = AIStreamBuilder()
        builder.error("Something went wrong")

        assert len(builder._events) == 1
        assert builder._events[0].type == "error"
        assert builder._events[0].error_text == "Something went wrong"

    def test_method_chaining(self):
        """Test that methods can be chained."""
        builder = (
            AIStreamBuilder()
            .start()
            .text("Hello")
            .data("info", {"key": "value"})
            .error("Oops")
            .finish()
        )

        assert builder._started
        assert builder._finished
        assert len(builder._events) > 0

    @pytest.mark.asyncio
    async def test_stream_generation(self):
        """Test async stream generation."""
        builder = AIStreamBuilder()
        builder.text("Hello")

        events = []
        async for event in builder.stream():
            events.append(event)

        # Should auto-start and auto-finish
        assert events[0].type == "start"
        assert events[-1].type == "finish"

        # Should contain text events
        text_events = [e for e in events if "text" in e.type]
        assert len(text_events) > 0

    @pytest.mark.asyncio
    async def test_build_method(self):
        """Test build method returns AIStream."""
        builder = AIStreamBuilder()
        builder.text("Hello")

        stream = builder.build()

        assert isinstance(stream, AIStream)

        # Collect all SSE events
        sse_events = []
        async for sse in stream:
            sse_events.append(sse)

        # All events should be SSE formatted
        assert all(e.startswith("data: ") for e in sse_events[:-1])
        assert sse_events[-1] == "data: [DONE]\n\n"


class TestAIStream:
    """Tests for AIStream class."""

    @pytest.mark.asyncio
    async def test_stream_iteration(self):
        """Test basic stream iteration."""

        async def generator():
            yield StartEvent(message_id="test")
            yield TextDeltaEvent(id="txt_1", delta="Hello")
            yield FinishEvent()

        stream = AIStream(generator(), auto_close=False)

        events = []
        async for sse in stream:
            events.append(sse)

        assert len(events) == 3
        assert all(e.startswith("data: ") and e.endswith("\n\n") for e in events)

    @pytest.mark.asyncio
    async def test_auto_close(self):
        """Test auto-close functionality."""

        async def generator():
            yield StartEvent(message_id="test")
            yield TextDeltaEvent(id="txt_1", delta="Hello")

        stream = AIStream(generator(), auto_close=True)

        events = []
        async for sse in stream:
            events.append(sse)

        # Should have original events plus finish and [DONE]
        assert events[-2].startswith('data: {"type":"finish"')
        assert events[-1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in stream."""

        async def generator():
            yield StartEvent(message_id="test")
            raise ValueError("Test error")

        stream = AIStream(generator(), auto_close=True)

        events = []
        with pytest.raises(ValueError):
            async for sse in stream:
                events.append(sse)

        # Should have sent error event before raising
        error_events = [e for e in events if '"type":"error"' in e]
        assert len(error_events) == 1

    @pytest.mark.asyncio
    async def test_pipe_transformation(self):
        """Test pipe transformation."""

        async def generator():
            yield StartEvent(message_id="test")
            yield TextDeltaEvent(id="txt_1", delta="Hello")
            yield TextDeltaEvent(id="txt_1", delta=" World")

        def uppercase_text(event):
            if isinstance(event, TextDeltaEvent):
                event.delta = event.delta.upper()
            return event

        stream = AIStream(generator(), auto_close=False)
        piped = stream.pipe(uppercase_text)

        events = []
        async for sse in piped:
            events.append(sse)

        # Text should be uppercase
        assert '"delta":"HELLO"' in events[1]
        assert '"delta":" WORLD"' in events[2]

    @pytest.mark.asyncio
    async def test_filter(self):
        """Test filter functionality."""

        async def generator():
            yield StartEvent(message_id="test")
            yield TextDeltaEvent(id="txt_1", delta="Keep")
            yield ErrorEvent(error_text="Filter me")
            yield TextDeltaEvent(id="txt_1", delta="Also keep")

        def no_errors(event):
            return not isinstance(event, ErrorEvent)

        stream = AIStream(generator(), auto_close=False)
        filtered = stream.filter(no_errors)

        events = []
        async for sse in filtered:
            events.append(sse)

        # Should not contain error event
        assert not any('"type":"error"' in e for e in events)
        assert len(events) == 3  # Start + 2 text deltas


class TestTextStreamContext:
    """Tests for TextStreamContext context manager."""

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test TextStreamContext as async context manager."""
        builder = AIStreamBuilder()

        async with builder.text_stream("txt_test") as ctx:
            ctx.write("Hello").write(" ").write("World")

        # Should have start, 3 deltas, and end
        assert len(builder._events) == 5
        assert builder._events[0].type == "text-start"
        assert builder._events[0].id == "txt_test"
        assert builder._events[1].delta == "Hello"
        assert builder._events[2].delta == " "
        assert builder._events[3].delta == "World"
        assert builder._events[4].type == "text-end"


class TestSimpleTextStream:
    """Tests for create_simple_text_stream helper."""

    @pytest.mark.asyncio
    async def test_simple_text_streaming(self):
        """Test simple text streaming."""
        events = []

        async for sse in create_simple_text_stream(
            "Hello World", chunk_size=5, delay=0.01
        ):
            events.append(sse)

        # Should have start, text-start, deltas, text-end, finish, [DONE]
        assert events[0].startswith('data: {"type":"start"')
        assert events[1].startswith('data: {"type":"text-start"')
        assert events[-2].startswith('data: {"type":"finish"')
        assert events[-1] == "data: [DONE]\n\n"

        # Check deltas
        delta_events = [e for e in events if '"type":"text-delta"' in e]
        assert len(delta_events) > 0


class TestStreamIntegration:
    """Integration tests for stream building and processing."""

    @pytest.mark.asyncio
    async def test_complete_conversation_flow(self):
        """Test a complete conversation flow."""
        builder = AIStreamBuilder("conversation_1")

        # Build a complex stream
        builder.start()
        builder.reasoning("Let me think about this question...")
        builder.text("Based on my analysis, ")
        builder.tool_call(
            "search",
            {"query": "Python FastAPI"},
            {"results": ["FastAPI docs", "Tutorial"]},
        )
        builder.text("I found some relevant information.")
        builder.data("summary", {"key_points": ["Point 1", "Point 2"]})
        builder.finish()

        # Convert to stream and collect events
        stream = builder.build()
        events = []
        async for sse in stream:
            if sse != "data: [DONE]\n\n":
                # Parse JSON from SSE
                json_str = sse.replace("data: ", "").strip()
                event_data = json.loads(json_str)
                events.append(event_data)

        # Verify event sequence
        assert events[0]["type"] == "start"
        assert any(e["type"] == "reasoning-start" for e in events)
        assert any(e["type"] == "text-delta" for e in events)
        assert any(e["type"] == "tool-input-start" for e in events)
        assert any(e["type"] == "tool-output-available" for e in events)
        assert any(e["type"] == "data-summary" for e in events)
        assert events[-1]["type"] == "finish"

    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """Test error handling and recovery."""

        async def problematic_generator():
            yield StartEvent(message_id="test")
            yield TextDeltaEvent(id="txt_1", delta="Starting...")

            # Simulate an error condition
            try:
                raise ValueError("Simulated error")
            except ValueError as e:
                yield ErrorEvent(error_text=str(e))

            # Continue after error
            yield TextDeltaEvent(id="txt_1", delta=" Recovered!")
            yield FinishEvent()

        stream = AIStream(problematic_generator(), auto_close=False)

        events = []
        async for sse in stream:
            events.append(sse)

        # Should contain error event but continue
        assert any('"type":"error"' in e for e in events)
        assert any('"delta":" Recovered!"' in e for e in events)
        assert any('"type":"finish"' in e for e in events)
