"""
Advanced AI Assistant with tools, reasoning, and structured data.

Run with:
    uvicorn examples.advanced_ai_assistant:app --reload
"""

import asyncio
import random
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from fastapi_ai_sdk import (
    AIStreamBuilder,
    ai_endpoint,
    tool_endpoint,
)

app = FastAPI(title="Advanced AI Assistant")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request models
class ChatRequest(BaseModel):
    """Chat request model."""

    message: str
    use_tools: bool = True
    include_reasoning: bool = True
    session_id: Optional[str] = None


class WeatherRequest(BaseModel):
    """Weather tool request."""

    city: str
    units: str = "celsius"


# Mock database for session storage
sessions: Dict[str, List[str]] = {}


# Tool implementations
async def get_weather(city: str, units: str = "celsius") -> Dict:
    """
    Mock weather API call.

    Args:
        city: City name
        units: Temperature units (celsius/fahrenheit)

    Returns:
        Weather data
    """
    # Simulate API delay
    await asyncio.sleep(0.5)

    # Mock weather data
    temp_c = random.randint(10, 30)
    temp_f = (temp_c * 9 / 5) + 32

    conditions = ["sunny", "cloudy", "rainy", "snowy"]

    return {
        "city": city,
        "temperature": temp_f if units == "fahrenheit" else temp_c,
        "units": units,
        "condition": random.choice(conditions),
        "humidity": random.randint(30, 80),
        "wind_speed": random.randint(5, 25),
        "timestamp": datetime.now().isoformat(),
    }


async def search_knowledge(query: str) -> List[Dict]:
    """
    Mock knowledge base search.

    Args:
        query: Search query

    Returns:
        Search results
    """
    # Simulate search delay
    await asyncio.sleep(0.3)

    # Mock search results
    results = [
        {
            "title": "FastAPI Documentation",
            "snippet": "FastAPI is a modern web framework for Python...",
            "url": "https://fastapi.tiangolo.com",
            "relevance": 0.95,
        },
        {
            "title": "Vercel AI SDK",
            "snippet": "The AI SDK is a TypeScript library for building AI applications...",
            "url": "https://sdk.vercel.ai",
            "relevance": 0.88,
        },
    ]

    return results


async def calculate(expression: str) -> float:
    """
    Simple calculator tool.

    Args:
        expression: Mathematical expression

    Returns:
        Calculation result
    """
    try:
        # DANGER: eval is used here for simplicity - never do this in production!
        # Use a proper expression parser instead
        result = eval(expression, {"__builtins__": {}}, {})
        return float(result)
    except Exception as e:
        raise ValueError(f"Invalid expression: {e}")


async def handle_weather_request(builder: AIStreamBuilder, message_lower: str):
    """Handle weather-related requests."""
    # Extract city (simplified)
    cities = ["london", "new york", "tokyo", "berlin", "paris"]
    city = next((c for c in cities if c in message_lower), "London")

    # Make tool call
    weather_data = await get_weather(city.title())
    builder.tool_call(
        "get_weather",
        input_data={"city": city.title()},
        output_data=weather_data,
    )

    # Add response based on weather
    builder.text(
        f"The current weather in {weather_data['city']} is "
        f"{weather_data['condition']} with a temperature of "
        f"{weather_data['temperature']}°{weather_data['units'][0].upper()}. "
        f"Humidity is at {weather_data['humidity']}% and "
        f"wind speed is {weather_data['wind_speed']} km/h."
    )

    # Add structured data
    builder.data("weather", weather_data)


async def handle_search_request(builder: AIStreamBuilder, message: str):
    """Handle search-related requests."""
    # Perform search
    search_results = await search_knowledge(message)

    builder.tool_call(
        "search_knowledge",
        input_data={"query": message},
        output_data={"results": search_results},
    )

    builder.text("I found the following relevant information:\n\n")

    for i, result in enumerate(search_results, 1):
        builder.text(
            f"{i}. **{result['title']}**\n"
            f"   {result['snippet']}\n"
            f"   [Learn more]({result['url']})\n\n"
        )

    # Add structured search results
    builder.data("search_results", {"results": search_results})


async def handle_calculation_request(builder: AIStreamBuilder, message: str):
    """Handle calculation-related requests."""
    import re

    expression = re.search(r"[\d+\-*/().\s]+", message)
    if expression:
        expr = expression.group().strip()
        try:
            result = await calculate(expr)

            builder.tool_call(
                "calculator",
                input_data={"expression": expr},
                output_data={"result": result},
            )

            builder.text(f"The result of {expr} is **{result}**")

            # Add calculation data
            builder.data("calculation", {"expression": expr, "result": result})
        except ValueError as e:
            builder.error(str(e))
    else:
        builder.text("I couldn't find a valid mathematical expression to calculate.")


@app.post("/api/assistant")
@ai_endpoint()
async def ai_assistant(request: ChatRequest):
    """
    Advanced AI assistant with multiple capabilities.

    Args:
        request: Chat request with configuration

    Returns:
        Streaming AI response with tools and reasoning
    """
    builder = AIStreamBuilder()

    # Store message in session
    if request.session_id:
        if request.session_id not in sessions:
            sessions[request.session_id] = []
        sessions[request.session_id].append(request.message)

    # Start with reasoning if enabled
    if request.include_reasoning:
        builder.reasoning(
            f"Analyzing the query: '{request.message}'. "
            "Determining the best approach to respond..."
        )

    # Detect intent and use appropriate tools
    message_lower = request.message.lower()

    if request.use_tools:
        # Weather query detection
        if any(word in message_lower for word in ["weather", "temperature", "climate"]):
            await handle_weather_request(builder, message_lower)

        # Search query detection
        elif any(
            word in message_lower for word in ["search", "find", "look up", "what is"]
        ):
            await handle_search_request(builder, request.message)

        # Math calculation detection
        elif any(word in message_lower for word in ["calculate", "compute", "solve"]):
            await handle_calculation_request(builder, request.message)

        else:
            # Default response without tools
            builder.text(
                f'I received your message: "{request.message}"\n\n'
                "I can help you with:\n"
                "- Weather information (ask about weather in any city)\n"
                "- Knowledge search (ask me to search for information)\n"
                "- Calculations (ask me to calculate mathematical expressions)\n\n"
                "How can I assist you today?"
            )
    else:
        # Simple response without tools
        builder.text(f"You said: {request.message}")

    # Add session info if available
    if request.session_id and request.session_id in sessions:
        history_count = len(sessions[request.session_id])
        builder.data(
            "session_info",
            {
                "session_id": request.session_id,
                "message_count": history_count,
                "last_messages": sessions[request.session_id][-3:],
            },
        )

    # Add metadata
    builder.data(
        "metadata",
        {
            "timestamp": datetime.now().isoformat(),
            "tools_enabled": request.use_tools,
            "reasoning_enabled": request.include_reasoning,
        },
    )

    return builder


@app.post("/api/tools/weather")
@tool_endpoint("get_weather")
async def weather_endpoint(request: WeatherRequest):
    """
    Weather tool endpoint.

    Args:
        request: Weather request

    Returns:
        Weather data
    """
    return await get_weather(request.city, request.units)


@app.post("/api/tools/search")
@tool_endpoint("search_knowledge")
async def search_endpoint(query: str):
    """
    Search tool endpoint.

    Args:
        query: Search query

    Returns:
        Search results
    """
    return await search_knowledge(query)


@app.post("/api/tools/calculate")
@tool_endpoint("calculator")
async def calculator_endpoint(expression: str):
    """
    Calculator tool endpoint.

    Args:
        expression: Mathematical expression

    Returns:
        Calculation result
    """
    try:
        result = await calculate(expression)
        return {"result": result, "expression": expression}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """
    Get session history.

    Args:
        session_id: Session identifier

    Returns:
        Session history
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": session_id,
        "messages": sessions[session_id],
        "message_count": len(sessions[session_id]),
    }


@app.delete("/api/sessions/{session_id}")
async def clear_session(session_id: str):
    """
    Clear session history.

    Args:
        session_id: Session identifier

    Returns:
        Success message
    """
    if session_id in sessions:
        del sessions[session_id]
        return {"message": f"Session {session_id} cleared"}

    return {"message": "Session not found"}


@app.get("/")
async def root():
    """Root endpoint with API documentation."""
    return {
        "title": "Advanced AI Assistant API",
        "description": "AI assistant with tools, reasoning, and structured data",
        "endpoints": {
            "assistant": {
                "path": "/api/assistant",
                "method": "POST",
                "description": "Main AI assistant endpoint",
                "body": {
                    "message": "string (required)",
                    "use_tools": "boolean (default: true)",
                    "include_reasoning": "boolean (default: true)",
                    "session_id": "string (optional)",
                },
            },
            "tools": {
                "weather": "/api/tools/weather (POST)",
                "search": "/api/tools/search (POST)",
                "calculate": "/api/tools/calculate (POST)",
            },
            "sessions": {
                "get": "/api/sessions/{session_id} (GET)",
                "clear": "/api/sessions/{session_id} (DELETE)",
            },
        },
        "example_queries": [
            "What's the weather in London?",
            "Search for information about FastAPI",
            "Calculate 15 * 23 + 42",
            "Tell me about the Vercel AI SDK",
        ],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
