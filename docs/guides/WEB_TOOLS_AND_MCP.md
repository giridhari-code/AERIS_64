# Web search, weather & MCP-style tools

Without tools the small LM **cannot** know today's weather or live facts.  
AERIS adds **server-side tools** that fetch real data and inject it into the chat prompt.

## Enable / disable

```bash
export NEUROFIELD_WEB_TOOLS=1   # default on
# export NEUROFIELD_WEB_TOOLS=0  # disable auto inject on /v1/chat
```

Needs **outbound internet** from the machine running uvicorn.

## Auto on chat

If the user asks about weather / mausam / temperature, the server:

1. Calls Open-Meteo (geocode + current weather)
2. Prepends facts to the model prompt
3. Model generates a reply **conditioned on those facts**

Similar path for explicit search / news style queries (DuckDuckGo best-effort).

**Honest limit:** the tiny model may still paraphrase poorly; facts come from tools, not from weights.

## HTTP API

```bash
# MCP-style catalog
curl -s http://127.0.0.1:8000/v1/tools/mcp

# Weather
curl -s -X POST http://127.0.0.1:8000/v1/tools/weather \
  -H 'Content-Type: application/json' \
  -d '{"location":"Delhi"}'

# Web search
curl -s -X POST http://127.0.0.1:8000/v1/tools/web_search \
  -H 'Content-Type: application/json' \
  -d '{"query":"Open-Meteo API","max_results":3}'
```

## MCP

`GET /v1/tools/mcp` returns a **tools/list-shaped** JSON (`name`, `description`, `inputSchema`).  
Full MCP stdio server is optional future work; HTTP schemas are enough for many clients.

## Code

| File | Role |
|------|------|
| `src/neurofield/tools/weather.py` | Open-Meteo |
| `src/neurofield/tools/web_search.py` | DuckDuckGo lite / instant |
| `src/neurofield/tools/catalog.py` | MCP-style schemas |
| `src/neurofield/tools/orchestrate.py` | Intent → tool → enrich prompt |
| `src/neurofield/serving/server.py` | Wire + endpoints |

## Example

User: `Aj Delhi ka weather kitna hai?`  
Server injects: `Weather now in Delhi, India: 32°C, ...`  
Model sees facts before answering.
