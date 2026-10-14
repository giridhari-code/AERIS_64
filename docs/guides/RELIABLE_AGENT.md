# Reliable agent behaviour (SI_v0)

## Goal

Predictable tool use + hard budgets — not open-ended autonomy.

## Loop

1. User text  
2. **Enrich** via allow-listed tools (`orchestrate.enrich_prompt_with_tools`)  
3. Optional **generate_fn** (NeuroField checkpoint)  
4. Stop: answer | tool_fail_budget | empty | max_steps  

## Config

| Field | Default | Meaning |
|-------|---------|---------|
| `max_steps` | 4 | Cap |
| `max_tool_failures` | 2 | Abort tools |
| `allowed_tools` | weather, search, calc, time, wiki, FX | Allow-list |
| `always_enrich` | True | Facts before generate |

## CLI

```bash
PYTHONPATH=src python scripts/agent/run_agent.py "what time is it"
PYTHONPATH=src python scripts/agent/run_agent.py "calculate 12*17" --json
```

## Serve

Chat path already calls `enrich_prompt_with_tools` when tools enabled.  
Agent module is the **explicit** reliability API for scripts/evals.

## Reliability rules

- Live numbers come from tools, not model imagination  
- Every tool event is logged (`ToolEvent`)  
- Failures do not crash the process  
- No infinite replan loop in v0  

## Next (SI_v1)

- Multi-step plan JSON from model (validated schema)  
- Critique pass  
- Task suite success % in `scripts/eval`
