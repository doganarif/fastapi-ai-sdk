# FastAPI AI SDK

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI--green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A Pythonic helper library for building FastAPI applications that integrate with the [Vercel AI SDK](https://sdk.vercel.ai/). This library provides a seamless way to stream AI responses from your FastAPI backend to your Next.js frontend.

## Features

- **Full Vercel AI SDK Compatibility** - Implements the complete AI SDK protocol specification
- **Type-Safe with Pydantic** - Full type hints and validation for all events
- **Streaming Support** - Built-in Server-Sent Events (SSE) streaming
- **Easy Integration** - Simple decorators and utilities for FastAPI
- **Flexible Builder Pattern** - Intuitive API for constructing AI streams
- **Well Tested** - Comprehensive test coverage
- **Fully Documented** - Complete documentation with examples

## Installation

```bash
pip install fastapi-ai-sdk
```

## Quick Start

### Basic Example

```python
from fastapi import FastAPI
from fastapi_ai_sdk import AIStreamBuilder, ai_endpoint

app = FastAPI()

@app.post("/api/chat")
@ai_endpoint()
async def chat(message: str):
    """Simple chat endpoint that streams a response."""
    builder = AIStreamBuilder()
    builder.text(f"You said: {message}")
    return builder
```

### Frontend Integration (Next.js)

```tsx
import { useChat } from "@ai-sdk/react";

export default function Chat() {
  const { messages, input, handleInputChange, handleSubmit } = useChat({
    api: "http://localhost:8000/api/chat",
  });

  return (
    <div>
      {messages.map((msg) => (
        <div key={msg.id}>{msg.content}</div>
      ))}
      <form onSubmit={handleSubmit}>
        <input value={input} onChange={handleInputChange} />
        <button type="submit">Send</button>
      </form>
    </div>
  );
}
```

## Documentation

### Stream Events

The library supports all Vercel AI SDK event types:

- **Message Lifecycle**: `start`, `finish`
- **Text Streaming**: `text-start`, `text-delta`, `text-end`
- **Reasoning**: `reasoning-start`, `reasoning-delta`, `reasoning-end`
- **Tool Calls**: `tool-input-start`, `tool-input-delta`, `tool-input-available`, `tool-output-available`
- **Structured Data**: Custom `data-*` events
- **File References**: URLs and documents
- **Error Handling**: Error events with messages

### Using the Stream Builder

```python
from fastapi_ai_sdk import AIStreamBuilder

# Create a builder
builder = AIStreamBuilder(message_id="optional_id")

# Add different types of content
builder.start()  # Start the stream
builder.text("Here's some text")  # Add text content
builder.reasoning("Let me think about this...")  # Add reasoning
builder.data("weather", {"temperature": 20, "city": "Berlin"})  # Add structured data
builder.tool_call(  # Add tool usage
    "get_weather",
    input_data={"city": "Berlin"},
    output_data={"temperature": 20}
)
builder.finish()  # End the stream

# Build and return the stream
return builder.build()
```

### Decorators

#### `@ai_endpoint` - Automatic AI SDK Response Handling

```python
@app.post("/chat")
@ai_endpoint()
async def chat(message: str):
    builder = AIStreamBuilder()
    builder.text(f"Response: {message}")
    return builder
```

#### `@streaming_endpoint` - Simple Text Streaming

```python
@app.get("/stream")
@streaming_endpoint(chunk_size=10, delay=0.1)
async def stream():
    return "This text will be streamed chunk by chunk"
```

#### `@tool_endpoint` - Tool Call Handling

```python
@app.post("/tools/weather")
@tool_endpoint("get_weather")
async def get_weather(city: str):
    # Your tool logic here
    return {"temperature": 20, "condition": "sunny"}
```

### Advanced Examples

#### Streaming with Reasoning and Tools

```python
@app.post("/api/advanced-chat")
@ai_endpoint()
async def advanced_chat(query: str):
    builder = AIStreamBuilder()

    # Start with reasoning
    builder.reasoning("Analyzing your query...")

    # Make a tool call
    weather_data = await get_weather_data("Berlin")
    builder.tool_call(
        "get_weather",
        input_data={"city": "Berlin"},
        output_data=weather_data
    )

    # Stream the response
    builder.text(f"Based on the weather data: {weather_data}")

    return builder
```

#### Custom Async Generators

```python
from fastapi_ai_sdk import create_ai_stream_response

@app.get("/api/generate")
async def generate():
    async def event_generator():
        from fastapi_ai_sdk.models import StartEvent, TextDeltaEvent, FinishEvent

        yield StartEvent(message_id="gen_1")

        for word in ["Hello", " ", "from", " ", "FastAPI"]:
            yield TextDeltaEvent(id="txt_1", delta=word)
            await asyncio.sleep(0.1)

        yield FinishEvent()

    return create_ai_stream_response(event_generator())
```

#### Chunked Text Streaming

```python
@app.post("/api/story")
@ai_endpoint()
async def generate_story(prompt: str):
    builder = AIStreamBuilder()

    story = await generate_long_story(prompt)  # Your story generation logic

    # Stream with custom chunk size
    builder.text(story, chunk_size=50)  # Streams in 50-character chunks

    return builder
```

## Testing

Run the test suite:

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests with coverage
pytest --cov=fastapi_ai_sdk --cov-report=term-missing

# Run specific test file
pytest tests/test_models.py

# Run with verbose output
pytest -v
```

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/doganarif/fastapi-ai-sdk.git
cd fastapi-ai-sdk

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run linting
black fastapi_ai_sdk tests
isort fastapi_ai_sdk tests
flake8 fastapi_ai_sdk tests
mypy fastapi_ai_sdk
```

### Code Style

This project uses:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## Author

**Arif Dogan**

- Email: me@arif.sh
- GitHub: [@doganarif](https://github.com/doganarif)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgments

- [Vercel AI SDK](https://sdk.vercel.ai/) for the excellent frontend SDK
- [FastAPI](https://fastapi.tiangolo.com/) for the amazing web framework
- [Pydantic](https://docs.pydantic.dev/) for data validation

## Resources

- [Vercel AI SDK Documentation](https://sdk.vercel.ai/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Server-Sent Events Specification](https://html.spec.whatwg.org/multipage/server-sent-events.html)

---

<p align="center">Made with ❤️ for the FastAPI and AI community from Arif</p>
