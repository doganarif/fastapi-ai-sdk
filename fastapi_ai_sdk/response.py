"""
FastAPI response utilities for AI SDK streaming.

This module provides helper functions for creating FastAPI responses
that are compatible with the Vercel AI SDK.
"""

from collections.abc import AsyncGenerator
from typing import Any, Dict, Optional, Union

from fastapi.responses import StreamingResponse

from .stream import AIStream

# Default headers for AI SDK compatibility
DEFAULT_AI_HEADERS = {
    "x-vercel-ai-ui-message-stream": "v1",
    "Cache-Control": "no-cache, no-transform",
    "X-Accel-Buffering": "no",  # Disable Nginx buffering
}


def create_ai_stream_response(
    stream: Union[AIStream, AsyncGenerator[str, None]],
    headers: Optional[Dict[str, str]] = None,
    status_code: int = 200,
) -> StreamingResponse:
    """
    Create a FastAPI StreamingResponse for AI SDK streams.

    Args:
        stream: An AIStream instance or async generator yielding SSE events
        headers: Optional additional headers to include
        status_code: HTTP status code (default: 200)

    Returns:
        A FastAPI StreamingResponse configured for AI SDK
    """
    response_headers = DEFAULT_AI_HEADERS.copy()
    if headers:
        response_headers.update(headers)

    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers=response_headers,
        status_code=status_code,
    )


class AIStreamResponse(StreamingResponse):
    """
    A specialized StreamingResponse for AI SDK streams.

    This class extends FastAPI's StreamingResponse with AI SDK specific
    configuration and utilities.
    """

    def __init__(
        self,
        stream: Union[AIStream, AsyncGenerator[str, None]],
        status_code: int = 200,
        headers: Optional[Dict[str, str]] = None,
        media_type: str = "text/event-stream",
        background: Optional[Any] = None,
    ):
        """
        Initialize an AIStreamResponse.

        Args:
            stream: An AIStream instance or async generator yielding SSE events
            status_code: HTTP status code
            headers: Optional additional headers
            media_type: Media type (default: text/event-stream)
            background: Optional background task
        """
        response_headers = DEFAULT_AI_HEADERS.copy()
        if headers:
            response_headers.update(headers)

        super().__init__(
            content=stream,
            status_code=status_code,
            headers=response_headers,
            media_type=media_type,
            background=background,
        )

    @classmethod
    def from_ai_stream(
        cls,
        ai_stream: AIStream,
        **kwargs: Any,
    ) -> "AIStreamResponse":
        """
        Create an AIStreamResponse from an AIStream.

        Args:
            ai_stream: The AIStream to create response from
            **kwargs: Additional arguments for the response

        Returns:
            An AIStreamResponse instance
        """
        return cls(stream=ai_stream, **kwargs)


async def stream_text_response(
    text: str,
    chunk_size: int = 10,
    delay: float = 0.1,
    message_id: Optional[str] = None,
) -> StreamingResponse:
    """
    Create a streaming response for simple text content.

    Args:
        text: The text to stream
        chunk_size: Size of each chunk
        delay: Delay between chunks in seconds
        message_id: Optional message ID

    Returns:
        A FastAPI StreamingResponse
    """
    import asyncio

    from .stream import AIStreamBuilder

    async def generate() -> AsyncGenerator[str, None]:
        builder = AIStreamBuilder(message_id=message_id)

        # Start message
        yield builder.start()._events[0].to_sse()

        # Start text
        text_id = f"txt_{id(text)}"
        from .models import FinishEvent, TextDeltaEvent, TextEndEvent, TextStartEvent

        yield TextStartEvent(id=text_id).to_sse()

        # Stream chunks
        for i in range(0, len(text), chunk_size):
            chunk = text[i : i + chunk_size]
            yield TextDeltaEvent(id=text_id, delta=chunk).to_sse()
            await asyncio.sleep(delay)

        # End text
        yield TextEndEvent(id=text_id).to_sse()

        # Finish
        yield FinishEvent().to_sse()
        yield "data: [DONE]\n\n"

    return create_ai_stream_response(generate())


async def stream_json_response(
    data_name: str,
    data: Dict,
    message_id: Optional[str] = None,
) -> StreamingResponse:
    """
    Create a streaming response for structured data.

    Args:
        data_name: The name for the data type
        data: The structured data to send
        message_id: Optional message ID

    Returns:
        A FastAPI StreamingResponse
    """
    from .stream import AIStreamBuilder

    builder = AIStreamBuilder(message_id=message_id)
    builder.start().data(data_name, data).finish()

    return create_ai_stream_response(builder.build())
