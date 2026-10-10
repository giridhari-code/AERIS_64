# Tools connector — web, weather, time, calc, wiki, FX (MCP-style)

Small LMs cannot know live facts. Server tools fetch data and inject into `/v1/chat`.

## Tool set

| Tool | Endpoint | Source |
|------|----------|--------|
| web_search | `POST /v1/tools/web_search` | DuckDuckGo |
| weather | `POST /v1/tools/weather` | Open-Meteo |
| time_now | `POST /v1/tools/time` | Server clock / timezone |
| calculator | `POST /v1/tools/calculator` | Safe arithmetic |
| wikipedia | `POST /v1/tools/wikipedia` | Wikipedia REST |
| currency_convert | `POST /v1/tools/currency` | Frankfurter/ECB |
| MCP catalog | `GET /v1/tools/mcp` | JSON schemas |

## Enable

```bash
export NEUROFIELD_WEB_TOOLS=1   # chat auto-routing (default)
# NEUROFIELD_WEB_TOOLS=0 to disable inject
```

Outbound internet required for web/weather/wiki/FX.

## Chat auto examples

- `Aj Delhi ka weather kitna hai?` → weather  
- `What time is it in India?` → time  
- `Calculate 12.5 * 8 + 3` → calculator  
- `Who is APJ Abdul Kalam?` → wikipedia  
- `100 USD to INR` → currency  
- `latest news about ISRO` → web search  

## MCP

`GET /v1/tools/mcp` returns tools/list-shaped schemas for agent clients.

## Code

`src/neurofield/tools/*` + wired in `serving/server.py`.
