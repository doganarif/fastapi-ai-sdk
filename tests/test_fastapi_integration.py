"""Tests for FastAPI integration and decorators."""

import json
from typing import Optional

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from fastapi_ai_sdk import (
    AIStream,
    AIStreamBuilder,
    ai_endpoint,
    create_ai_stream_response,
    streaming_endpoint,
    tool_endpoint,
)
from fastapi_ai_sdk.response import (
    AIStreamResponse,
    stream_json_response,
    stream_text_response,
)

# Create test FastAPI app
app = FastAPI()


# Test endpoints
@app.get("/simple-stream")
@ai_endpoint()
async def simple_stream():
    """Simple streaming endpoint."""
    builder = AIStreamBuilder()
    builder.text("Hello from FastAPI!")
    return builder


@app.post("/chat")
@ai_endpoint(message_id_param="message_id")
async def chat_endpoint(message: str, message_id: Optional[str] = None):
    """Chat endpoint with message ID."""
    builder = AIStreamBuilder(message_id)
    builder.text(f"You said: {message}")
    return builder


@app.get("/text-stream")
@streaming_endpoint(chunk_size=5, delay=0.01)
async def text_stream_endpoint():
    """Text streaming endpoint."""
    return "This is a streaming text response"


@app.post("/tools/weather")
@tool_endpoint("get_weather")
async def weather_tool(city: str, units: str = "celsius"):
    """Weather tool endpoint."""
    return {
        "temperature": 20,
        "condition": "sunny",
        "city": city,
        "units": units,
    }


@app.get("/manual-stream")
async def manual_stream():
    """Manually created stream response."""
    builder = AIStreamBuilder()
    builder.start().text("Manual response").finish()
    return create_ai_stream_response(builder.build())


@app.get("/async-generator")
@ai_endpoint()
async def async_generator_endpoint():
    """Endpoint that returns an async generator."""

    async def generate():
        from fastapi_ai_sdk.models import FinishEvent, StartEvent, TextDeltaEvent

        yield StartEvent(message_id="gen_1")
        yield TextDeltaEvent(id="txt_1", delta="Generated ")
        yield TextDeltaEvent(id="txt_1", delta="text")
        yield FinishEvent()

    return generate()


@app.get("/dict-response")
@ai_endpoint()
async def dict_response_endpoint():
    """Endpoint that returns a dict."""
    return {"message": "This is a dict response", "value": 42}


@app.get("/string-response")
@ai_endpoint()
async def string_response_endpoint():
    """Endpoint that returns a string."""
    return "This is a simple string response"


class TestFastAPIIntegration:
    """Tests for FastAPI integration."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_simple_stream(self, client):
        """Test simple streaming endpoint."""
        with client.stream("GET", "/simple-stream") as response:
            assert response.status_code == 200
            assert (
                response.headers["content-type"] == "text/event-stream; charset=utf-8"
            )
            assert response.headers["x-vercel-ai-ui-message-stream"] == "v1"

            # Collect events
            events = []
            for line in response.iter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    json_str = line.replace("data: ", "")
                    events.append(json.loads(json_str))

            # Check events
            assert events[0]["type"] == "start"
            assert any(e["type"] == "text-delta" for e in events)
            assert events[-1]["type"] == "finish"

    def test_chat_endpoint_with_message_id(self, client):
        """Test chat endpoint with message ID parameter."""
        with client.stream(
            "POST", "/chat", params={"message": "Hello", "message_id": "custom_123"}
        ) as response:
            assert response.status_code == 200

            # Parse first event
            first_line = next(response.iter_lines())
            if first_line.startswith("data: "):
                event = json.loads(first_line.replace("data: ", ""))
                assert event["type"] == "start"
                assert event["messageId"] == "custom_123"

    def test_streaming_decorator(self, client):
        """Test streaming_endpoint decorator."""
        with client.stream("GET", "/text-stream") as response:
            assert response.status_code == 200

            # Collect text deltas
            text_parts = []
            for line in response.iter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    event = json.loads(line.replace("data: ", ""))
                    if event["type"] == "text-delta":
                        text_parts.append(event["delta"])

            # Reassemble text
            full_text = "".join(text_parts)
            assert full_text == "This is a streaming text response"

    def test_tool_endpoint(self, client):
        """Test tool_endpoint decorator."""
        with client.stream(
            "POST", "/tools/weather", params={"city": "Berlin", "units": "fahrenheit"}
        ) as response:
            assert response.status_code == 200

            # Collect events
            events = []
            for line in response.iter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    events.append(json.loads(line.replace("data: ", "")))

            # Find tool events
            tool_input = next(
                (e for e in events if e["type"] == "tool-input-available"), None
            )
            tool_output = next(
                (e for e in events if e["type"] == "tool-output-available"), None
            )

            assert tool_input is not None
            assert tool_input["toolName"] == "get_weather"
            assert tool_input["input"]["city"] == "Berlin"

            assert tool_output is not None
            assert tool_output["output"]["temperature"] == 20
            assert tool_output["output"]["city"] == "Berlin"

    def test_manual_stream_response(self, client):
        """Test manually created stream response."""
        with client.stream("GET", "/manual-stream") as response:
            assert response.status_code == 200
            assert response.headers["x-vercel-ai-ui-message-stream"] == "v1"

            # Check content
            content = list(response.iter_lines())
            assert any("Manual response" in line for line in content)

    def test_async_generator_endpoint(self, client):
        """Test endpoint that returns async generator."""
        with client.stream("GET", "/async-generator") as response:
            assert response.status_code == 200

            # Collect text
            text_parts = []
            for line in response.iter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    event = json.loads(line.replace("data: ", ""))
                    if event["type"] == "text-delta":
                        text_parts.append(event["delta"])

            assert "".join(text_parts) == "Generated text"

    def test_dict_response_endpoint(self, client):
        """Test endpoint that returns a dict."""
        with client.stream("GET", "/dict-response") as response:
            assert response.status_code == 200

            # Find data event
            for line in response.iter_lines():
                if line.startswith("data: ") and '"type":"data-response"' in line:
                    event = json.loads(line.replace("data: ", ""))
                    assert event["data"]["message"] == "This is a dict response"
                    assert event["data"]["value"] == 42
                    break

    def test_string_response_endpoint(self, client):
        """Test endpoint that returns a string."""
        with client.stream("GET", "/string-response") as response:
            assert response.status_code == 200

            # Collect text
            text_parts = []
            for line in response.iter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    event = json.loads(line.replace("data: ", ""))
                    if event["type"] == "text-delta":
                        text_parts.append(event["delta"])

            full_text = "".join(text_parts)
            assert full_text == "This is a simple string response"


class TestResponseUtilities:
    """Tests for response utility functions."""

    @pytest.mark.asyncio
    async def test_stream_text_response(self):
        """Test stream_text_response function."""
        response = await stream_text_response(
            "Test text", chunk_size=4, delay=0.01, message_id="test_123"
        )

        assert isinstance(response, StreamingResponse)
        assert response.media_type == "text/event-stream"
        assert response.headers["x-vercel-ai-ui-message-stream"] == "v1"

    @pytest.mark.asyncio
    async def test_stream_json_response(self):
        """Test stream_json_response function."""
        response = await stream_json_response(
            "weather", {"city": "Berlin", "temp": 18}, message_id="test_456"
        )

        assert isinstance(response, StreamingResponse)
        assert response.media_type == "text/event-stream"

    def test_ai_stream_response_class(self):
        """Test AIStreamResponse class."""
        builder = AIStreamBuilder()
        builder.text("Test")
        stream = builder.build()

        response = AIStreamResponse(stream)

        assert response.status_code == 200
        assert response.media_type == "text/event-stream"
        assert "x-vercel-ai-ui-message-stream" in response.headers

    def test_ai_stream_response_from_method(self):
        """Test AIStreamResponse.from_ai_stream method."""
        builder = AIStreamBuilder()
        builder.text("Test")
        stream = builder.build()

        response = AIStreamResponse.from_ai_stream(
            stream, status_code=201, headers={"Custom-Header": "value"}
        )

        assert response.status_code == 201
        assert response.headers["Custom-Header"] == "value"
        assert response.headers["x-vercel-ai-ui-message-stream"] == "v1"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_empty_stream(self, client):
        """Test handling of empty streams."""

        @app.get("/empty-stream")
        @ai_endpoint()
        async def empty_stream():
            return AIStreamBuilder()

        with client.stream("GET", "/empty-stream") as response:
            assert response.status_code == 200

            # Should still have start and finish events
            events = []
            for line in response.iter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    events.append(json.loads(line.replace("data: ", "")))

            assert events[0]["type"] == "start"
            assert events[-1]["type"] == "finish"

    @pytest.mark.asyncio
    async def test_error_in_stream(self):
        """Test error handling in streams."""

        async def error_generator():
            from fastapi_ai_sdk.models import StartEvent

            yield StartEvent(message_id="error_test")
            raise ValueError("Test error")

        stream = AIStream(error_generator())

        events = []
        with pytest.raises(ValueError):
            async for sse in stream:
                events.append(sse)

        # Should have error event before exception
        assert any('"type":"error"' in e for e in events)

    def test_invalid_decorator_usage(self, client):
        """Test invalid decorator usage."""

        @app.get("/invalid")
        @ai_endpoint()
        async def invalid_endpoint():
            return 12345  # Invalid return type

        # Should raise ValueError for unsupported type
        with pytest.raises(ValueError, match="Unsupported return type"):
            client.get("/invalid")

    def test_tool_endpoint_error(self, client):
        """Test tool endpoint error handling."""

        @app.post("/tools/broken")
        @tool_endpoint("broken_tool")
        async def broken_tool():
            raise RuntimeError("Tool is broken")

        with client.stream("POST", "/tools/broken") as response:
            assert response.status_code == 200

            # Should contain error event
            for line in response.iter_lines():
                if line.startswith("data: ") and '"type":"error"' in line:
                    event = json.loads(line.replace("data: ", ""))
                    assert "Tool broken_tool failed" in event["errorText"]
                    break


@pytest.mark.asyncio
class TestAsyncIntegration:
    """Async integration tests using httpx."""

    async def test_async_client_streaming(self):
        """Test streaming with async httpx client."""
        transport = httpx.ASGITransport(app=app)
        async with (
            httpx.AsyncClient(transport=transport, base_url="http://test") as client,
            client.stream("GET", "/simple-stream") as response,
        ):
            assert response.status_code == 200

            events = []
            async for line in response.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    events.append(json.loads(line.replace("data: ", "")))

            assert events[0]["type"] == "start"
            assert any(e["type"] == "text-delta" for e in events)
