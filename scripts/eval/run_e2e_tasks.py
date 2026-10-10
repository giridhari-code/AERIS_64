#!/usr/bin/env python3
"""Run free end-to-end tasks against a local checkpoint or HTTP API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# repo root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def make_local_generate(checkpoint: str, device: str = "cpu"):
    import torch
    from neurofield.checkpoint.infer import load_model_from_checkpoint

    model, tok, cfg, meta = load_model_from_checkpoint(checkpoint, None, device=device)
    model.eval()
    encode = tok.encode
    decode = tok.decode
    device_t = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    model = model.to(device_t)

    def generate(prompt: str, max_tokens: int = 64, temperature: float = 0.2) -> str:
        ids = encode(f"User: {prompt}\nAssistant:")
        if not ids:
            ids = [0]
        cur = torch.tensor([ids], dtype=torch.long, device=device_t)
        state = None
        out_ids: list[int] = []
        with torch.no_grad():
            for _ in range(max_tokens):
                o = model(cur, state=state)
                state = o.state
                logits = o.logits[:, -1, :]
                if temperature <= 1e-5:
                    nid = int(logits.argmax(dim=-1).item())
                else:
                    probs = torch.softmax(logits.float() / temperature, dim=-1)
                    nid = int(torch.multinomial(probs, 1).item())
                out_ids.append(nid)
                cur = torch.tensor([[nid]], dtype=torch.long, device=device_t)
        return decode(out_ids)

    return generate


def make_http_generate(base_url: str):
    import urllib.request

    def generate(prompt: str, max_tokens: int = 64, temperature: float = 0.2) -> str:
        body = json.dumps(
            {
                "prompt": prompt,
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "auto_length": True,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            base_url.rstrip("/") + "/v1/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return str(data.get("text") or data.get("completion") or data.get("message") or data)

    return generate


def main() -> None:
    ap = argparse.ArgumentParser(description="Free E2E tasks for AERIS")
    ap.add_argument("--tasks", default=str(ROOT / "data/e2e_tasks/free_tasks.jsonl"))
    ap.add_argument("--checkpoint", default="", help="local checkpoint folder")
    ap.add_argument("--api", default="", help="e.g. http://127.0.0.1:8000")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from neurofield.tasks.e2e import run_task_file

    if args.api:
        gen = make_http_generate(args.api)
        mode = f"api:{args.api}"
    elif args.checkpoint:
        gen = make_local_generate(args.checkpoint, device=args.device)
        mode = f"ckpt:{args.checkpoint}"
    else:
        print("Need --checkpoint or --api")
        sys.exit(2)

    results = run_task_file(args.tasks, gen)
    n = len(results)
    graded = [r for r in results if r.passed is not None]
    passed = sum(1 for r in graded if r.passed)
    print(f"mode={mode} tasks={n} graded={len(graded)} passed={passed}")
    for r in results:
        flag = {True: "PASS", False: "FAIL", None: "FREE"}[r.passed]
        print(f"  [{flag}] {r.id:16s} {r.latency_ms:7.1f}ms  {r.output[:60]!r}")

    payload = {
        "mode": mode,
        "passed": passed,
        "graded": len(graded),
        "results": [r.to_dict() for r in results],
    }
    out = Path(args.out) if args.out else ROOT / "docs" / "e2e_last_run.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
