"""MCP-style tool catalog (schemas for agents / clients)."""

from __future__ import annotations

from typing import Any


def tool_schemas() -> list[dict[str, Any]]:
    return [
        {
            "name": "web_search",
            "description": "Search the public web for current information. Use for news, facts, docs.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
        {
            "name": "weather",
            "description": "Current weather for a city/place (Open-Meteo).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City or place name"},
                },
                "required": ["location"],
            },
        },
    ]


def list_mcp_tools() -> dict[str, Any]:
    """Subset shaped like MCP tools/list result."""
    return {"tools": tool_schemas()}
