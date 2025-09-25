"""
Stream builder and helper classes for creating AI SDK compatible streams.

This module provides utilities to build and manage event streams that are
compatible with the Vercel AI SDK frontend.
"""

import asyncio
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
)

from .models import (
    AnyStreamEvent,
    DataEvent,
    ErrorEvent,
    FinishEvent,
    FinishStepEvent,
    ReasoningDeltaEvent,
    ReasoningEndEvent,
    ReasoningStartEvent,
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


class AIStream:
    """
    A wrapper for async generators that produce AI SDK events.

    This class provides utilities for managing and transforming event streams.
    """

    def __init__(
        self,
        generator: AsyncGenerator[AnyStreamEvent, None],
        auto_close: bool = True,
    ):
        """
        Initialize an AIStream.

        Args:
            generator: An async generator that yields stream events
            auto_close: Whether to automatically send finish event and [DONE] marker
        """
        self._generator = generator
        self._auto_close = auto_close
        self._closed = False

    async def __aiter__(self) -> AsyncIterator[str]:
        """
        Iterate over the stream, yielding SSE formatted events.

        Yields:
            SSE formatted strings ready to be sent to the client
        """
        try:
            async for event in self._generator:
                yield event.to_sse()

            if self._auto_close and not self._closed:
                yield FinishEvent().to_sse()
                yield "data: [DONE]\n\n"
                self._closed = True
        except Exception as e:
            # Send error event if an exception occurs
            yield ErrorEvent(errorText=str(e)).to_sse()
            if self._auto_close:
                yield "data: [DONE]\n\n"
            raise

    def pipe(
        self,
        transform: Callable[[AnyStreamEvent], Optional[AnyStreamEvent]],
    ) -> "AIStream":
        """
        Pipe the stream through a transformation function.

        Args:
            transform: A function that transforms events (can return None to filter)

        Returns:
            A new AIStream with the transformation applied
        """

        async def transformed_generator() -> AsyncGenerator[AnyStreamEvent, None]:
            async for event in self._generator:
                transformed = transform(event)
                if transformed is not None:
                    yield transformed

        return AIStream(transformed_generator(), auto_close=self._auto_close)

    def filter(
        self,
        predicate: Callable[[AnyStreamEvent], bool],
    ) -> "AIStream":
        """
        Filter events in the stream.

        Args:
            predicate: A function that returns True for events to keep

        Returns:
            A new AIStream with the filter applied
        """

        async def filtered_generator() -> AsyncGenerator[AnyStreamEvent, None]:
            async for event in self._generator:
                if predicate(event):
                    yield event

        return AIStream(filtered_generator(), auto_close=self._auto_close)


class AIStreamBuilder:
    """
    A builder class for constructing AI SDK compatible event streams.

    This class provides a fluent interface for building complex event streams
    with proper lifecycle management.
    """

    def __init__(self, message_id: Optional[str] = None):
        """
        Initialize the stream builder.

        Args:
            message_id: Optional message ID (will be generated if not provided)
        """
        self._message_id = message_id or f"msg_{uuid.uuid4().hex[:8]}"
        self._events: List[AnyStreamEvent] = []
        self._started = False
        self._finished = False
        self._current_text_id: Optional[str] = None
        self._current_reasoning_id: Optional[str] = None
        self._in_step = False

    def start(self) -> "AIStreamBuilder":
        """
        Add a start event to the stream.

        Returns:
            Self for method chaining

        Raises:
            RuntimeError: If start has already been called
        """
        if self._started:
            raise RuntimeError("Stream has already been started")

        self._events.append(StartEvent(messageId=self._message_id))
        self._started = True
        return self

    def finish(self) -> "AIStreamBuilder":
        """
        Add a finish event to the stream.

        Returns:
            Self for method chaining

        Raises:
            RuntimeError: If finish has already been called
        """
        if self._finished:
            raise RuntimeError("Stream has already been finished")

        # Close any open parts
        if self._current_text_id:
            self._events.append(TextEndEvent(id=self._current_text_id))
            self._current_text_id = None

        if self._current_reasoning_id:
            self._events.append(ReasoningEndEvent(id=self._current_reasoning_id))
            self._current_reasoning_id = None

        if self._in_step:
            self._events.append(FinishStepEvent())
            self._in_step = False

        self._events.append(FinishEvent())
        self._finished = True
        return self

    def text(
        self,
        content: str,
        text_id: Optional[str] = None,
        chunk_size: Optional[int] = None,
    ) -> "AIStreamBuilder":
        """
        Add text content to the stream.

        Args:
            content: The text content to add
            text_id: Optional text part ID (will be generated if not provided)
            chunk_size: Optional chunk size for splitting content

        Returns:
            Self for method chaining
        """
        text_id = text_id or f"txt_{uuid.uuid4().hex[:8]}"

        # Start text part
        self._events.append(TextStartEvent(id=text_id))

        # Add content (optionally chunked)
        if chunk_size:
            chunks = [
                content[i: i + chunk_size] for i in range(0, len(content), chunk_size)
            ]
            for chunk in chunks:
                self._events.append(TextDeltaEvent(id=text_id, delta=chunk))
        else:
            self._events.append(TextDeltaEvent(id=text_id, delta=content))

        # End text part
        self._events.append(TextEndEvent(id=text_id))

        return self

    def text_stream(
        self,
        text_id: Optional[str] = None,
    ) -> "TextStreamContext":
        """
        Create a context manager for streaming text.

        Args:
            text_id: Optional text part ID (will be generated if not provided)

        Returns:
            A context manager for streaming text
        """
        return TextStreamContext(self, text_id)

    def reasoning(
        self,
        content: str,
        reasoning_id: Optional[str] = None,
        chunk_size: Optional[int] = None,
    ) -> "AIStreamBuilder":
        """
        Add reasoning content to the stream.

        Args:
            content: The reasoning content to add
            reasoning_id: Optional reasoning part ID (will be generated if not provided)
            chunk_size: Optional chunk size for splitting content

        Returns:
            Self for method chaining
        """
        reasoning_id = reasoning_id or f"r_{uuid.uuid4().hex[:8]}"

        # Start reasoning part
        self._events.append(ReasoningStartEvent(id=reasoning_id))

        # Add content (optionally chunked)
        if chunk_size:
            chunks = [
                content[i: i + chunk_size] for i in range(0, len(content), chunk_size)
            ]
            for chunk in chunks:
                self._events.append(ReasoningDeltaEvent(id=reasoning_id, delta=chunk))
        else:
            self._events.append(ReasoningDeltaEvent(id=reasoning_id, delta=content))

        # End reasoning part
        self._events.append(ReasoningEndEvent(id=reasoning_id))

        return self

    def data(
        self,
        name: str,
        data: Dict[str, Any],
    ) -> "AIStreamBuilder":
        """
        Add structured data to the stream.

        Args:
            name: The name of the data type (will be prefixed with 'data-')
            data: The structured data to add

        Returns:
            Self for method chaining
        """
        self._events.append(DataEvent.create(name, data))
        return self

    def tool_call(
        self,
        tool_name: str,
        input_data: Dict[str, Any],
        output_data: Optional[Dict[str, Any]] = None,
        tool_call_id: Optional[str] = None,
        stream_input: bool = False,
    ) -> "AIStreamBuilder":
        """
        Add a tool call to the stream.

        Args:
            tool_name: The name of the tool being called
            input_data: The input data for the tool
            output_data: Optional output data from the tool
            tool_call_id: Optional tool call ID (will be generated if not provided)
            stream_input: Whether to stream the input as deltas

        Returns:
            Self for method chaining
        """
        tool_call_id = tool_call_id or f"call_{uuid.uuid4().hex[:8]}"

        # Start tool input
        self._events.append(
            ToolInputStartEvent(toolCallId=tool_call_id, toolName=tool_name)
        )

        # Stream or provide input
        if stream_input:
            import json

            input_str = json.dumps(input_data)
            for char in input_str:
                self._events.append(
                    ToolInputDeltaEvent(toolCallId=tool_call_id, inputTextDelta=char)
                )

        # Make input available
        self._events.append(
            ToolInputAvailableEvent(
                toolCallId=tool_call_id,
                toolName=tool_name,
                input=input_data,
            )
        )

        # Add output if available
        if output_data is not None:
            self._events.append(
                ToolOutputAvailableEvent(toolCallId=tool_call_id, output=output_data)
            )

        return self

    def step(
        self, func: Optional[Callable[["AIStreamBuilder"], None]] = None
    ) -> "AIStreamBuilder":
        """
        Add a step to the stream.

        Args:
            func: Optional function to execute within the step

        Returns:
            Self for method chaining
        """
        self._events.append(StartStepEvent())

        if func:
            func(self)

        self._events.append(FinishStepEvent())

        return self

    def error(self, error_text: str) -> "AIStreamBuilder":
        """
        Add an error to the stream.

        Args:
            error_text: The error message

        Returns:
            Self for method chaining
        """
        self._events.append(ErrorEvent(errorText=error_text))
        return self

    def add_event(self, event: AnyStreamEvent) -> "AIStreamBuilder":
        """
        Add a custom event to the stream.

        Args:
            event: The event to add

        Returns:
            Self for method chaining
        """
        self._events.append(event)
        return self

    async def stream(self) -> AsyncGenerator[AnyStreamEvent, None]:
        """
        Generate the stream of events.

        Yields:
            Stream events in order
        """
        # Auto-start if not started
        if not self._started:
            # Insert start event at beginning
            self._events.insert(0, StartEvent(messageId=self._message_id))
            self._started = True

        # Yield all events
        for event in self._events:
            yield event
            # Small delay to simulate streaming
            await asyncio.sleep(0)

        # Auto-finish if not finished
        if not self._finished:
            self.finish()
            # Yield the finish event
            yield self._events[-1]

    def build(self) -> AIStream:
        """
        Build and return an AIStream.

        Returns:
            An AIStream instance
        """
        return AIStream(self.stream())


class TextStreamContext:
    """Context manager for streaming text content."""

    def __init__(self, builder: AIStreamBuilder, text_id: Optional[str] = None):
        """
        Initialize the text stream context.

        Args:
            builder: The stream builder to use
            text_id: Optional text part ID
        """
        self._builder = builder
        self._text_id = text_id or f"txt_{uuid.uuid4().hex[:8]}"

    async def __aenter__(self) -> "TextStreamContext":
        """Enter the context and start the text part."""
        self._builder._events.append(TextStartEvent(id=self._text_id))
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the context and end the text part."""
        self._builder._events.append(TextEndEvent(id=self._text_id))

    def write(self, delta: str) -> "TextStreamContext":
        """
        Write a text delta.

        Args:
            delta: The text delta to write

        Returns:
            Self for method chaining
        """
        self._builder._events.append(TextDeltaEvent(id=self._text_id, delta=delta))
        return self


async def create_simple_text_stream(
    text: str,
    chunk_size: int = 10,
    delay: float = 0.1,
) -> AsyncGenerator[str, None]:
    """
    Create a simple text streaming generator.

    Args:
        text: The text to stream
        chunk_size: Size of each chunk
        delay: Delay between chunks in seconds

    Yields:
        SSE formatted events
    """
    builder = AIStreamBuilder()
    builder.start()

    # Start message
    yield builder._events[0].to_sse()

    # Start text
    text_id = f"txt_{uuid.uuid4().hex[:8]}"
    yield TextStartEvent(id=text_id).to_sse()

    # Stream chunks
    for i in range(0, len(text), chunk_size):
        chunk = text[i: i + chunk_size]
        yield TextDeltaEvent(id=text_id, delta=chunk).to_sse()
        await asyncio.sleep(delay)

    # End text
    yield TextEndEvent(id=text_id).to_sse()

    # Finish
    yield FinishEvent().to_sse()
    yield "data: [DONE]\n\n"
