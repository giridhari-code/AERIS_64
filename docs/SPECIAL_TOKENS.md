# Chat special tokens

## Tokens

| Token | Role |
|-------|------|
| `<\|user\|>` | Start of user turn |
| `<\|assistant\|>` | Start of assistant turn |
| `<\|end\|>` | End of turn |
| `<\|pad\|>` / `<\|unk\|>` | Pad / unknown |

Aliases when encoding: `User:` → `<|user|>`, `Assistant:` → `<|assistant|>`.

## Why

Previously the server stopped generation with **regex** on decoded text (`User:`, blank lines).  
That is brittle. With specials in the **tokenizer vocab**, generation can stop on **token id** (`<|end|>` or `<|user|>`).

## Train format (recommended)

```text
<|user|>hello<|end|><|assistant|>Hello. I am AERIS_64.<|end|>
```

Or legacy (still encoded via aliases if specials exist):

```text
User: hello
Assistant: Hello. I am AERIS_64.
```

## Legacy checkpoints

Old char vocab **without** specials → server keeps **text stop** fallback.  
Retrain with current `CharTokenizer.from_text` / BPE to bake specials into vocab.
