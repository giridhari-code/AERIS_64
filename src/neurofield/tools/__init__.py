"""External tools: web search, weather, MCP-style catalog."""

from neurofield.tools.catalog import list_mcp_tools, tool_schemas
from neurofield.tools.orchestrate import enrich_prompt_with_tools

__all__ = ["list_mcp_tools", "tool_schemas", "enrich_prompt_with_tools"]
