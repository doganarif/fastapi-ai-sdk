"""
Decorators for FastAPI endpoints to simplify AI SDK integration.

This module provides decorators that make it easy to create
AI SDK compatible endpoints in FastAPI.
"""

import functools
import inspect
from typing import Any, Callable, Optional

from fastapi import Response
from fastapi.responses import StreamingResponse

from .response import create_ai_stream_response
from .stream import AIStream, AIStreamBuilder


async def _handle_builder_result(result: AIStreamBuilder) -> StreamingResponse:
    """Handle AIStreamBuilder return type."""
    # The stream() method will handle auto-start internally
    stream = result.build()
    return create_ai_stream_response(stream)


async def _handle_unknown_result(
    result: Any, message_id: Optional[str]
) -> StreamingResponse:
    """Handle unknown return types by converting to appropriate stream."""
    if isinstance(result, str):
        from .response import stream_text_response

        return await stream_text_response(result, message_id=message_id)
    elif isinstance(result, dict):
        from .response import stream_json_response

        return await stream_json_response("response", result, message_id=message_id)
    else:
        raise ValueError(
            f"Unsupported return type: {type(result)}. "
            "Expected AIStreamBuilder, AIStream, async generator, "
            "StreamingResponse, string, or dict."
        )


def ai_endpoint(
    auto_start: bool = True,
    auto_finish: bool = True,
    message_id_param: Optional[str] = None,
    include_request: bool = False,
) -> Callable:
    """
    Decorator for FastAPI endpoints to automatically handle AI SDK streaming.

    This decorator:
    - Automatically wraps the response in an AI SDK compatible streaming response
    - Optionally auto-starts and auto-finishes the stream
    - Can extract message ID from request parameters

    Args:
        auto_start: Whether to automatically start the stream
        auto_finish: Whether to automatically finish the stream
        message_id_param: Name of the parameter to use for message ID
        include_request: Whether to pass the FastAPI Request object to the function

    Returns:
        Decorated function that returns an AI SDK compatible response

    Example:
        ```python
        @app.post("/chat")
        @ai_endpoint(message_id_param="message_id")
        async def chat(message_id: Optional[str] = None):
            builder = AIStreamBuilder(message_id)
            builder.text("Hello from FastAPI!")
            return builder
        ```
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Response:
            # Extract message_id if parameter specified
            message_id = None
            if message_id_param and message_id_param in kwargs:
                message_id = kwargs.get(message_id_param)

            # Call the original function
            result = (
                await func(*args, **kwargs)
                if inspect.iscoroutinefunction(func)
                else func(*args, **kwargs)
            )

            # Handle different return types
            if isinstance(result, AIStreamBuilder):
                return await _handle_builder_result(result)

            elif isinstance(result, AIStream):
                return create_ai_stream_response(result)

            elif inspect.isasyncgen(result):
                # Result is an async generator object
                # Wrap in AIStream to handle event-to-SSE conversion
                stream = AIStream(result)
                return create_ai_stream_response(stream)

            elif isinstance(result, (StreamingResponse, Response)):
                return result

            else:
                return await _handle_unknown_result(result, message_id)

        return wrapper

    return decorator


def streaming_endpoint(
    chunk_size: int = 10,
    delay: float = 0.1,
) -> Callable:
    """
    Decorator for simple text streaming endpoints.

    This decorator automatically streams string responses with configurable
    chunking and delay.

    Args:
        chunk_size: Size of each text chunk
        delay: Delay between chunks in seconds

    Returns:
        Decorated function that streams text responses

    Example:
        ```python
        @app.get("/stream")
        @streaming_endpoint(chunk_size=20, delay=0.05)
        async def stream_text():
            return "This text will be streamed to the client!"
        ```
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Response:
            # Call the original function
            result = (
                await func(*args, **kwargs)
                if inspect.iscoroutinefunction(func)
                else func(*args, **kwargs)
            )

            if not isinstance(result, str):
                raise ValueError(
                    f"streaming_endpoint decorator expects string return, got {type(result)}"
                )

            from .response import stream_text_response

            return await stream_text_response(
                result,
                chunk_size=chunk_size,
                delay=delay,
            )

        return wrapper

    return decorator


def tool_endpoint(tool_name: str) -> Callable:
    """
    Decorator for tool call endpoints.

    This decorator wraps the endpoint to handle tool calls with proper
    AI SDK event formatting.

    Args:
        tool_name: The name of the tool

    Returns:
        Decorated function that handles tool calls

    Example:
        ```python
        @app.post("/tools/weather")
        @tool_endpoint("get_weather")
        async def get_weather(city: str):
            # Tool implementation
            return {"temperature": 20, "condition": "sunny"}
        ```
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Response:
            import uuid

            # Extract input from kwargs or request body
            input_data = kwargs.copy()

            # Remove FastAPI injected parameters
            for key in ["request", "response", "background_tasks"]:
                input_data.pop(key, None)

            # Generate tool call ID
            tool_call_id = f"call_{uuid.uuid4().hex[:8]}"

            # Call the tool function
            try:
                output = (
                    await func(*args, **kwargs)
                    if inspect.iscoroutinefunction(func)
                    else func(*args, **kwargs)
                )

                # Build response stream
                builder = AIStreamBuilder()
                builder.start().tool_call(
                    tool_name=tool_name,
                    input_data=input_data,
                    output_data=output,
                    tool_call_id=tool_call_id,
                ).finish()

                return create_ai_stream_response(builder.build())

            except Exception as e:
                # Handle errors
                builder = AIStreamBuilder()
                builder.start().error(f"Tool {tool_name} failed: {str(e)}").finish()
                return create_ai_stream_response(builder.build())

        return wrapper

    return decorator
