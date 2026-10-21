# 0z code — full company-standard agent

**Product:** 0z code  
**Rule:** No programming source code (write / edit / execute).  
**Loop:** Policy → Plan → ToolBus → Synthesize (optional LM).

## Package

```text
packages/sentinel/zeroz/
  policy.py      product policy + refusals
  planner.py     multi-step tool plans
  toolbus.py     allow-listed tool dispatch + schemas
  workspace.py   safe read-only FS helpers
  runtime.py     full agent turn
  session.py     transcript + audit JSONL
  server.py      FastAPI /v1/chat
  cli.py         CLI
configs/zeroz.yaml
```

## CLI

```bash
export PYTHONPATH=packages
python -m sentinel.zeroz.cli --policy
python -m sentinel.zeroz.cli --no-model "Weather in Mumbai?"
python -m sentinel.zeroz.cli --no-model "write python code"   # refused
python -m sentinel.zeroz.cli -i --config configs/zeroz.yaml
```

## HTTP API

```bash
pip install fastapi uvicorn pydantic
export PYTHONPATH=packages
python -m sentinel.zeroz.server
# GET  /health
# GET  /v1/policy
# GET  /v1/tools
# POST /v1/chat  {"message":"...","use_model":false}
```

## After 109M train (optional polish)

```bash
python -m sentinel.zeroz.cli --checkpoint runs/aeris_109m_pretrain/final \
  --device cuda "Summarize private AI benefits for Indian SMEs"
```
