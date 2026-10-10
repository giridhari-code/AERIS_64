"""MCP-style tool catalog (schemas for agents / clients)."""

from __future__ import annotations

from typing import Any


def tool_schemas() -> list[dict[str, Any]]:
    return [
        {
            "name": "web_search",
            "description": "Search the public web for current information, news, docs.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
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
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
        {
            "name": "time_now",
            "description": "Current date and time, optional IANA timezone (e.g. Asia/Kolkata).",
            "inputSchema": {
                "type": "object",
                "properties": {"timezone": {"type": "string", "description": "IANA tz name"}},
            },
        },
        {
            "name": "calculator",
            "description": "Evaluate a basic arithmetic expression (+ - * / ** %).",
            "inputSchema": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
        {
            "name": "wikipedia",
            "description": "Short encyclopedia summary for a topic.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "lang": {"type": "string", "default": "en"},
                },
                "required": ["topic"],
            },
        },
        {
            "name": "currency_convert",
            "description": "Convert amount between currencies (ECB/Frankfurter).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number"},
                    "from": {"type": "string", "description": "ISO code e.g. USD"},
                    "to": {"type": "string", "description": "ISO code e.g. INR"},
                },
                "required": ["amount", "from", "to"],
            },
        },
    ]


def list_mcp_tools() -> dict[str, Any]:
    return {"tools": tool_schemas()}
