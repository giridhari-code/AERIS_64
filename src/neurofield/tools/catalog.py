"""MCP-style tool catalog (schemas for agents / clients).

For MCP / structured-pipeline systems the matching evaluation metrics are:
  - Field extraction accuracy
  - Pipeline completion rate

Agent-level validity and completion are measured in
``neurofield.agent.loop.AgentResult.metrics``.
"""

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
        {
            "name": "git",
            "description": (
                "Safe git operations for this repository: status, log, diff, add, "
                "commit, show, branch. Push is disabled unless NEUROFIELD_GIT_ALLOW_PUSH=1."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Git subcommand, e.g. 'status', 'log', 'diff', 'add src/', 'commit'",
                    },
                    "message": {
                        "type": "string",
                        "description": "Commit message (required when command is commit)",
                    },
                    "path": {
                        "type": "string",
                        "description": "Optional working directory (defaults to repo root)",
                    },
                },
                "required": ["command"],
            },
        },
        {
            "name": "rag_retrieve",
            "description": (
                "Retrieval-Augmented Generation: search the domain knowledge base "
                "(vector store) for chunks similar to the query, then rank them."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 4},
                },
                "required": ["query"],
            },
        },
        {
            "name": "tool_search",
            "description": (
                "Claude-style Tool Search: given a natural-language need, rank and return "
                "the most relevant tools from the full catalog. Use this when you have many "
                "tools and want only the best matches instead of loading everything."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What capability or information you need",
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 5,
                        "description": "How many tools to return",
                    },
                },
                "required": ["query"],
            },
        },

        {
            "name": "code_execution",
            "description": "Run Python code in an isolated sandbox and return stdout/stderr (Code Interpreter).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python source to execute"},
                    "language": {"type": "string", "default": "python"},
                },
                "required": ["code"],
            },
        },
        {
            "name": "code_interpreter",
            "description": "Alias of code_execution — isolated Python interpreter.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "language": {"type": "string", "default": "python"},
                },
                "required": ["code"],
            },
        },
        {
            "name": "bash",
            "description": "Run a restricted shell command inside the sandbox (allow-listed binaries only).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command"},
                },
                "required": ["command"],
            },
        },
        {
            "name": "web_fetch",
            "description": "Fetch a public URL and return readable plain text of the page.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "max_chars": {"type": "integer", "default": 8000},
                },
                "required": ["url"],
            },
        },
        {
            "name": "file_search",
            "description": "Search file names and content inside the repository (safe paths only).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "path": {"type": "string", "description": "Optional subdirectory to limit search"},
                },
                "required": ["query"],
            },
        },
        {
            "name": "image_generation",
            "description": "Generate an image from a text prompt (requires configured image provider).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "size": {"type": "string", "default": "1024x1024"},
                },
                "required": ["prompt"],
            },
        },
        {
            "name": "computer_use",
            "description": "Computer-use actions (screenshot, click, type, etc.). Requires a desktop backend.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "coordinate": {"type": "array", "items": {"type": "integer"}},
                    "text": {"type": "string"},
                },
                "required": ["action"],
            },
        },

    ]


def list_mcp_tools() -> dict[str, Any]:
    return {"tools": tool_schemas()}
