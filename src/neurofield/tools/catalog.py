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
                "Rank tools in the catalog against a natural-language need and return "
                "the best matches. Use when the tool set is large and only relevant "
                "tools should be considered."
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


        {
            "name": "cowork_file",
            "description": "Read, write, edit, or batch-process local workspace files. Deletes require allow_delete=true.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "description": "read|write|edit|delete|batch"},
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                    "allow_delete": {"type": "boolean", "default": False},
                },
                "required": ["action"],
            },
        },
        {
            "name": "cowork_plan",
            "description": "Build a structured multi-step plan for a complex objective (autonomous execution when run_step is wired).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "objective": {"type": "string"},
                    "steps_json": {"type": "string", "description": "JSON list of {id,title,action,args}"},
                },
                "required": ["objective"],
            },
        },
        {
            "name": "cowork_subagents",
            "description": "Coordinate parallel sub-agent workstreams with shared budgets.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "tasks_json": {"type": "string", "description": "JSON list of {name, ...}"},
                },
                "required": ["tasks_json"],
            },
        },
        {
            "name": "cowork_task",
            "description": "Background/scheduled task create, status, list (persisted progress and reasoning).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "name": {"type": "string"},
                    "task_id": {"type": "string"},
                    "schedule": {"type": "string"},
                },
            },
        },
        {
            "name": "cowork_memory",
            "description": "Shared memory between chat and cowork sessions (optional per-task off switch).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "session_id": {"type": "string"},
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                    "memory_enabled": {"type": "boolean", "default": True},
                },
            },
        },
        {
            "name": "cowork_document",
            "description": "Create markdown reports, CSV/Excel-compatible tables, or presentation outlines in the workspace.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string"},
                    "path": {"type": "string"},
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                    "headers_json": {"type": "string"},
                    "rows_json": {"type": "string"},
                    "slides_json": {"type": "string"},
                },
                "required": ["kind", "path"],
            },
        },
        {
            "name": "cowork_draft_edit",
            "description": "In-place edit of a selected span inside a Markdown/text draft.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "selected_text": {"type": "string"},
                    "replacement": {"type": "string"},
                },
                "required": ["path", "selected_text", "replacement"],
            },
        },
        {
            "name": "cowork_skill",
            "description": "Save/load/list reusable skills; call registered external connectors.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "name": {"type": "string"},
                    "instructions": {"type": "string"},
                },
            },
        },

    ]


def list_mcp_tools() -> dict[str, Any]:
    """Full static catalog (may exceed per-request limit).

    For a single model turn use ``tools_for_request`` / ``mcp_list_tools``
    which enforce ``max_tools_per_request`` (default 250).
    """
    schemas = tool_schemas()
    return {"tools": schemas, "count": len(schemas)}
