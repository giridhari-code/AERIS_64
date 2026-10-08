#!/usr/bin/env python3
"""
Mini Reinforcement Learning for AERIS (toy scale).

What this is:
  - REINFORCE-style policy gradient on short prompts
  - Reward from simple heuristics + optional preference pairs
  - Continues from a supervised checkpoint (--resume)

What this is NOT:
  - Full PPO / Anthropic RLAIF / large reward model
  - A path to Claude-level intelligence by itself

Reward signals (default):
  + length in a sensible band
  + contains product name tokens when identity prompt
  - garbage patterns / User: leak
  + optional: higher reward if closer to "chosen" preference text (token overlap)
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from neurofield.checkpoint import load_checkpoint, save_checkpoint
from neurofield.config import ModelConfig, SafetyConfig, model_preset
from neurofield.model import NeuroField
from neurofield.tokenizer.char import CharTokenizer


def _load_prompts(path: str) -> list[str]:
    p = Path(path)
    if not p.is_file():
        return ["hello", "who are you", "thank you"]
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
    return lines or ["hello"]


def _load_prefs(path: str) -> list[dict]:
    p = Path(path)
    if not p.is_file():
        return []
    out = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln:
            out.append(json.loads(ln))
    return out


def reward_fn(prompt: str, text: str, prefs: list[dict]) -> float:
    t = (text or "").strip()
    if not t:
        return -2.0
    r = 0.0
    # length band (char-level models)
    n = len(t)
    if 8 <= n <= 120:
        r += 1.0
    elif n < 4:
        r -= 1.5
    elif n > 200:
        r -= 1.0
    # identity
    pl = prompt.lower()
    if any(k in pl for k in ("who are you", "your name", "hello", "hi")):
        if re.search(r"aeris|neurofield", t, re.I):
            r += 1.5
    # penalties
    if re.search(r"(?i)\buser\s*:", t):
        r -= 2.0
    if re.search(r"(.)\1{6,}", t):
        r -= 1.0
    if re.search(r"(?i)ntannt|nunsink|whenl", t):
        r -= 2.0
    # preference overlap
    for pr in prefs:
        if pr.get("prompt", "").lower().strip() == pl.strip():
            chosen = pr.get("chosen") or ""
            if chosen and chosen.lower()[:20] in t.lower():
                r += 2.0
            rej = pr.get("rejected") or ""
            if rej and len(rej) > 5 and rej.lower()[:15] in t.lower():
                r -= 1.5
            break
    return r


def sample_continuation(model, encode, decode, prompt: str, device, max_new: int = 40, temperature: float = 0.8):
    """Sample tokens; return (token_ids including prompt, logprobs of generated, text)."""
    ids = encode(prompt)
    if not ids:
        ids = [0]
    generated = list(ids)
    logps: list[torch.Tensor] = []
    cur = torch.tensor([ids], dtype=torch.long, device=device)
    state = None
    model.eval()
    for _ in range(max_new):
        out = model(cur, state=state, return_audit=False)
        state = out.state
        logits = out.logits[:, -1, :] / max(temperature, 1e-5)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        nid = dist.sample()
        logps.append(dist.log_prob(nid))
        generated.append(int(nid.item()))
        cur = nid.view(1, 1)
        # stop on end-ish
        piece = decode([int(nid.item())])
        if piece in ("\n",) and len(generated) - len(ids) > 10:
            break
    cont = generated[len(ids) :]
    text = decode(cont)
    return generated, logps, text


def main() -> None:
    ap = argparse.ArgumentParser(description="AERIS mini RL (REINFORCE)")
    ap.add_argument("--resume", default="docs/AERIS_64")
    ap.add_argument("--prompts", default="data/rl_prompts.txt")
    ap.add_argument("--prefs", default="data/align/preferences.jsonl")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--preset", default="tiny")
    ap.add_argument("--out", default="docs/AERIS_64")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    prompts = _load_prompts(args.prompts)
    prefs = _load_prefs(args.prefs)

    # tokenizer from checkpoint vocab or prompts
    root = Path(args.resume)
    text_blob = "\n".join(prompts)
    tok = CharTokenizer.from_text(text_blob, with_specials=True)
    if (root / "vocab.json").is_file():
        try:
            raw = json.loads((root / "vocab.json").read_text(encoding="utf-8"))
            if "stoi" in raw:
                tok = CharTokenizer(stoi=raw["stoi"], itos={int(k): v for k, v in raw.get("itos", {}).items()})
                print("using checkpoint vocab")
        except Exception as e:
            print("vocab fallback", e)

    encode, decode = tok.encode, tok.decode
    preset = model_preset(args.preset)
    # prefer checkpoint config dims
    d_model = preset.d_model
    vocab_size = tok.vocab_size
    if (root / "config.json").is_file():
        try:
            cfgj = json.loads((root / "config.json").read_text())
            d_model = int(cfgj.get("d_model", d_model))
            vocab_size = int(cfgj.get("vocab_size", vocab_size))
        except Exception:
            pass

    cfg = ModelConfig(
        d_model=d_model,
        d_k=max(8, d_model // 2),
        d_v=max(8, d_model // 2),
        n_skills=preset.n_skills,
        top_k=preset.top_k,
        k_max=preset.k_max,
        dendrite_window=preset.dendrite_window,
        vocab_size=vocab_size,
        max_seq_len=preset.max_seq_len,
        tie_embeddings=True,
        dropout=0.0,
    )
    model = NeuroField(cfg, SafetyConfig(enable_audit=False)).to(device)
    if root.exists():
        try:
            ck = load_checkpoint(args.resume)
            model.load_state_dict(ck["model"], strict=False)
            print(f"resumed {args.resume}")
        except Exception as e:
            print("resume failed", e)

    opt = AdamW(model.parameters(), lr=args.lr)
    print(f"RL steps={args.steps} prompts={len(prompts)} device={device}")
    t0 = time.time()
    baseline = 0.0
    beta = 0.9

    for step in range(1, args.steps + 1):
        prompt = prompts[(step - 1) % len(prompts)]
        # format lightly
        full_prompt = f"User: {prompt}\nAssistant:"
        model.train()
        _, logps, text = sample_continuation(
            model, encode, decode, full_prompt, device, max_new=args.max_new, temperature=args.temperature
        )
        r = reward_fn(prompt, text, prefs)
        baseline = beta * baseline + (1 - beta) * r
        advantage = r - baseline
        if not logps:
            continue
        loss = -(advantage) * torch.stack(logps).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 20 == 0 or step == 1:
            print(f"  rl {step} reward={r:.2f} baseline={baseline:.2f} text={text[:60]!r}")

    config_dict = {
        "d_model": cfg.d_model,
        "d_k": cfg.d_k,
        "d_v": cfg.d_v,
        "n_skills": cfg.n_skills,
        "top_k": cfg.top_k,
        "k_max": cfg.k_max,
        "dendrite_window": cfg.dendrite_window,
        "vocab_size": cfg.vocab_size,
        "tie_embeddings": True,
        "tokenizer": "char",
    }
    out = save_checkpoint(
        args.out,
        model,
        config=config_dict,
        stoi=getattr(tok, "stoi", None),
        itos=getattr(tok, "itos", None),
        meta={
            "product": "AERIS_64",
            "stage": "rl_reinforce",
            "steps": args.steps,
            "note": "mini REINFORCE; not full PPO/RLAIF",
        },
    )
    (Path(out) / "tokenizer.json").write_text(
        json.dumps(tok.to_vocab_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Done in {time.time()-t0:.1f}s saved {out}")


if __name__ == "__main__":
    main()
