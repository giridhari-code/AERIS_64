# Company chat stack — system prompt + tools + SFT

## Layers

```text
1. Special tokens   <|system|> <|user|> <|assistant|> <|end|>
2. System prompt    rules + tool list (same in train & serve)
3. Tool-first path  SI_v1 / orchestrate (reliable, no LM needed)
4. Model tool JSON  optional after SFT on data/samples/sft_system_tools.txt
5. UI apps/web      :3000 → proxy → :8000 /v1/chat
```

## Tokens

| Token | Role |
|-------|------|
| `<\|system\|>` | instructions + tools |
| `<\|user\|>` | user message |
| `<\|assistant\|>` | model reply / tool JSON |
| `<\|end\|>` | end of turn |

## Env

| Variable | Default | Meaning |
|----------|---------|---------|
| `NEUROFIELD_SYSTEM_PROMPT` | company default | override system text |
| `NEUROFIELD_CHAT_TOOLS_DIRECT` | 1 | SI_v1/tool-first before LM |
| `NEUROFIELD_MODEL_TOOL_CALLS` | 1 | if LM emits JSON tool, execute it |
| `NEUROFIELD_WEB_TOOLS` | 1 | enable tool package |

## Train (same format as serve)

```bash
PYTHONPATH=src python scripts/train/train_company.py \
  --data data/samples/sft_system_tools.txt \
  --preset tiny --steps 500 \
  --out docs/runs/sft_tools_v1 --device cpu
```

## Serve

System prompt is set at app startup from env or `default_system_prompt()`.

## Rule

**Train format == serve format.** Mismatch → model ignores system / tools.
