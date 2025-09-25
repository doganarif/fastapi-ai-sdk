"""
Example showing integration with actual LLMs (OpenAI, Anthropic, etc).

Run with:
    OPENAI_API_KEY=your_key uvicorn examples.llm_integration:app --reload
"""

import asyncio
import os
from collections.abc import AsyncGenerator
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from fastapi_ai_sdk import (
    AIStream,
    AIStreamBuilder,
    ai_endpoint,
    create_ai_stream_response,
)
from fastapi_ai_sdk.models import (
    ErrorEvent,
    FinishEvent,
    ReasoningDeltaEvent,
    ReasoningEndEvent,
    ReasoningStartEvent,
    StartEvent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
)

app = FastAPI(title="LLM Integration Example")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    """Chat message model."""

    role: str  # "user", "assistant", "system"
    content: str


class ChatRequest(BaseModel):
    """Chat request model."""

    messages: list[ChatMessage]
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stream: bool = True


# Mock OpenAI streaming (replace with actual OpenAI client)
async def mock_openai_stream(
    messages: list[ChatMessage],
    model: str,
    temperature: float,
) -> AsyncGenerator[str, None]:
    """
    Mock OpenAI streaming response.

    In production, replace this with actual OpenAI API calls:

    ```python
    import openai

    async def openai_stream(messages, model, temperature):
        client = openai.AsyncOpenAI()
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=temperature,
            stream=True
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    ```
    """
    # Simulate thinking time
    await asyncio.sleep(0.5)

    # Generate mock response based on last message
    last_message = messages[-1].content if messages else "Hello"

    response = f"""I understand you're asking about: "{last_message}"

Based on my analysis, here are the key points:

1. This is a mock response demonstrating streaming capabilities
2. In production, this would connect to actual LLM APIs
3. The streaming format is compatible with Vercel AI SDK

Each chunk of this response is being streamed gradually to create a smooth user experience.
This approach is particularly effective for longer responses where users can start reading
while the rest of the content is being generated.

The FastAPI AI SDK makes it easy to integrate any LLM backend with the Vercel AI SDK frontend,
providing a seamless streaming experience across your entire application stack."""

    # Stream response word by word
    words = response.split()
    for i, word in enumerate(words):
        # Add space except for first word
        if i > 0:
            yield " "
        yield word
        # Simulate variable typing speed
        await asyncio.sleep(0.02 if len(word) < 5 else 0.05)


@app.post("/api/chat/openai")
@ai_endpoint()
async def chat_with_openai(request: ChatRequest):
    """
    Chat endpoint with OpenAI-style streaming.

    Args:
        request: Chat request with messages and parameters

    Returns:
        Streaming response compatible with Vercel AI SDK
    """
    builder = AIStreamBuilder()

    # Start the stream
    builder.start()

    # Add reasoning (optional - shows thinking process)
    if request.model.startswith("o1"):  # Models with reasoning
        builder._events.append(ReasoningStartEvent(id="reasoning_1"))
        builder._events.append(
            ReasoningDeltaEvent(
                id="reasoning_1",
                delta="Analyzing the conversation context and preparing response...",
            )
        )
        builder._events.append(ReasoningEndEvent(id="reasoning_1"))

    # Start text streaming
    text_id = "response_text"
    builder._events.append(TextStartEvent(id=text_id))

    try:
        # Stream from LLM
        async for chunk in mock_openai_stream(
            request.messages,
            request.model,
            request.temperature,
        ):
            builder._events.append(TextDeltaEvent(id=text_id, delta=chunk))

        # End text
        builder._events.append(TextEndEvent(id=text_id))

    except Exception as e:
        builder._events.append(ErrorEvent(error_text=f"LLM Error: {str(e)}"))

    # Finish the stream
    builder.finish()

    return builder


@app.post("/api/chat/streaming")
async def custom_streaming_chat(request: ChatRequest):
    """
    Custom streaming implementation with direct event control.

    Args:
        request: Chat request

    Returns:
        Raw SSE stream
    """

    async def generate_events():
        """Generate AI SDK events."""
        # Start message
        yield StartEvent(message_id="custom_msg")

        # Optional reasoning
        if request.model.startswith("o1"):
            reasoning_id = "r_1"
            yield ReasoningStartEvent(id=reasoning_id)

            reasoning_text = "Processing your request and formulating a response..."
            for char in reasoning_text:
                yield ReasoningDeltaEvent(id=reasoning_id, delta=char)
                await asyncio.sleep(0.01)

            yield ReasoningEndEvent(id=reasoning_id)

        # Text response
        text_id = "txt_1"
        yield TextStartEvent(id=text_id)

        try:
            # Stream from LLM
            async for chunk in mock_openai_stream(
                request.messages,
                request.model,
                request.temperature,
            ):
                yield TextDeltaEvent(id=text_id, delta=chunk)

        except Exception as e:
            yield ErrorEvent(error_text=str(e))

        finally:
            yield TextEndEvent(id=text_id)
            yield FinishEvent()

    # Create and return streaming response
    stream = AIStream(generate_events())
    return create_ai_stream_response(stream)


@app.post("/api/chat/anthropic")
@ai_endpoint()
async def chat_with_anthropic(request: ChatRequest):
    """
    Chat endpoint mimicking Anthropic Claude streaming.

    In production, use the actual Anthropic client:

    ```python
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic()
    stream = await client.messages.create(
        model="claude-3-opus-20240229",
        messages=messages,
        stream=True,
    )
    ```

    Args:
        request: Chat request

    Returns:
        Streaming response
    """
    builder = AIStreamBuilder()

    # Anthropic models often show their thinking process
    builder.reasoning(
        "I'll analyze your request and provide a comprehensive response. "
        "Let me consider the context and formulate the best answer..."
    )

    # Generate response
    text_content = (
        f"Based on your message about '{request.messages[-1].content if request.messages else 'your query'}', "
        "here's my response:\n\n"
        "This example demonstrates how to integrate Anthropic's Claude models "
        "with the FastAPI AI SDK. The streaming format remains consistent "
        "regardless of which LLM provider you use, making it easy to switch "
        "between different models or providers.\n\n"
        "The key advantages are:\n"
        "• Consistent API interface\n"
        "• Seamless frontend integration\n"
        "• Provider-agnostic implementation\n"
        "• Full streaming support"
    )

    # Stream the response with chunking
    builder.text(text_content, chunk_size=30)

    return builder


@app.post("/api/chat/multimodal")
@ai_endpoint()
async def multimodal_chat(
    request: ChatRequest,
    image_url: Optional[str] = None,
):
    """
    Example of multimodal chat with image analysis.

    Args:
        request: Chat request
        image_url: Optional image URL for analysis

    Returns:
        Streaming response with image analysis
    """
    builder = AIStreamBuilder()

    if image_url:
        # Add file reference
        from fastapi_ai_sdk.models import FileEvent

        builder.add_event(FileEvent(url=image_url, media_type="image/png"))

        # Simulate image analysis
        builder.reasoning("Analyzing the provided image...")

        # Add analysis result
        builder.text(
            "I can see the image you've provided. "
            "[In production, this would use vision-capable models like GPT-4V or Claude 3] "
        )

        # Add structured data about the image
        builder.data(
            "image_analysis",
            {
                "url": image_url,
                "detected_objects": ["example", "objects"],
                "description": "Image analysis would appear here",
                "confidence": 0.95,
            },
        )

    # Continue with text response
    builder.text(
        f"\nRegarding your message: '{request.messages[-1].content if request.messages else 'your query'}'\n\n"
        "This demonstrates multimodal capabilities where the AI can process "
        "both text and images, providing comprehensive analysis and responses."
    )

    return builder


@app.get("/")
async def root():
    """API documentation."""
    return {
        "title": "LLM Integration Examples",
        "description": "Examples of integrating various LLMs with FastAPI AI SDK",
        "endpoints": [
            {
                "path": "/api/chat/openai",
                "description": "OpenAI-style chat completion",
                "model_examples": ["gpt-3.5-turbo", "gpt-4", "o1-preview"],
            },
            {
                "path": "/api/chat/anthropic",
                "description": "Anthropic Claude-style chat",
                "model_examples": ["claude-3-opus", "claude-3-sonnet"],
            },
            {
                "path": "/api/chat/streaming",
                "description": "Custom streaming with direct event control",
            },
            {
                "path": "/api/chat/multimodal",
                "description": "Multimodal chat with image support",
                "supports": ["text", "images"],
            },
        ],
        "request_format": {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello!"},
            ],
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
            "stream": True,
        },
        "note": "These are mock implementations. In production, replace with actual LLM API calls.",
    }


if __name__ == "__main__":
    import uvicorn

    # Check for API keys (optional)
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. Using mock responses.")

    uvicorn.run(app, host="0.0.0.0", port=8000)
