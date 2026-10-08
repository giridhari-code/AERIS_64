#!/usr/bin/env python3
"""
Full RLHF-style stack for AERIS (still small-model scale).

Stages:
  1) reward  — pairwise ranking reward model (Bradley-Terry on sequence logprobs)
  2) dpo     — Direct Preference Optimization vs frozen reference
  3) ppo     — sample + reward model score + clipped policy objective + KL

This is a complete *pipeline* (RM → preference opt → on-policy RL), not a 3-line demo.
It is still NOT Anthropic/OpenAI compute scale.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from neurofield.checkpoint import load_checkpoint, save_checkpoint
from neurofield.config import ModelConfig, SafetyConfig, model_preset
from neurofield.model import NeuroField
from neurofield.rlhf.dpo import dpo_loss
from neurofield.rlhf.ppo import ppo_policy_loss
from neurofield.rlhf.rewards import pairwise_reward_loss
from neurofield.tokenizer.char import CharTokenizer


def load_prefs(path: str) -> list[dict]:
    p = Path(path)
    if not p.is_file():
        return [
            {"prompt": "hello", "chosen": "Hello. I am AERIS_64.", "rejected": "ntannt garbage"},
            {"prompt": "who are you", "chosen": "I am AERIS_64.", "rejected": "I am chatgpt claude"},
        ]
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def pack_pair(tok, prompt: str, answer: str, max_len: int = 96):
    text = f"User: {prompt}\nAssistant: {answer}"
    ids = tok.encode(text)[: max_len + 1]
    if len(ids) < 3:
        ids = ids + [0] * (3 - len(ids))
    x = torch.tensor(ids[:-1], dtype=torch.long)
    y = torch.tensor(ids[1:], dtype=torch.long)
    return x, y


def build_model(tok, resume: str, preset: str, device: torch.device) -> NeuroField:
    root = Path(resume)
    d_model = model_preset(preset).d_model
    vocab_size = tok.vocab_size
    n_skills = model_preset(preset).n_skills
    if (root / "config.json").is_file():
        cfgj = json.loads((root / "config.json").read_text())
        d_model = int(cfgj.get("d_model", d_model))
        vocab_size = int(cfgj.get("vocab_size", vocab_size))
        n_skills = int(cfgj.get("n_skills", n_skills))
    preset_cfg = model_preset(preset)
    cfg = ModelConfig(
        d_model=d_model,
        d_k=max(8, d_model // 2),
        d_v=max(8, d_model // 2),
        n_skills=n_skills,
        top_k=preset_cfg.top_k,
        k_max=preset_cfg.k_max,
        dendrite_window=preset_cfg.dendrite_window,
        vocab_size=vocab_size,
        max_seq_len=preset_cfg.max_seq_len,
        tie_embeddings=True,
        dropout=0.0,
    )
    model = NeuroField(cfg, SafetyConfig(enable_audit=False)).to(device)
    if root.exists():
        try:
            ck = load_checkpoint(resume)
            model.load_state_dict(ck["model"], strict=False)
            print(f"loaded weights {resume}")
        except Exception as e:
            print("load failed", e)
    return model


def stage_reward(model, tok, prefs, device, steps, lr):
    opt = AdamW(model.parameters(), lr=lr)
    model.train()
    print(f"[reward] steps={steps}")
    for step in range(1, steps + 1):
        pr = prefs[(step - 1) % len(prefs)]
        cx, cy = pack_pair(tok, pr["prompt"], pr["chosen"])
        rx, ry = pack_pair(tok, pr["prompt"], pr["rejected"])
        loss = pairwise_reward_loss(
            model,
            cx.unsqueeze(0).to(device),
            cy.unsqueeze(0).to(device),
            rx.unsqueeze(0).to(device),
            ry.unsqueeze(0).to(device),
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 25 == 0 or step == 1:
            print(f"  reward {step} loss={float(loss):.4f}")
    return model


def stage_dpo(policy, ref, tok, prefs, device, steps, lr, beta):
    opt = AdamW(policy.parameters(), lr=lr)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    print(f"[dpo] steps={steps} beta={beta}")
    for step in range(1, steps + 1):
        policy.train()
        pr = prefs[(step - 1) % len(prefs)]
        cx, cy = pack_pair(tok, pr["prompt"], pr["chosen"])
        rx, ry = pack_pair(tok, pr["prompt"], pr["rejected"])
        loss = dpo_loss(
            policy,
            ref,
            cx.unsqueeze(0).to(device),
            cy.unsqueeze(0).to(device),
            rx.unsqueeze(0).to(device),
            ry.unsqueeze(0).to(device),
            beta=beta,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        if step % 25 == 0 or step == 1:
            print(f"  dpo {step} loss={float(loss):.4f}")
    return policy


def sample_response(model, tok, prompt: str, device, max_new=32, temperature=0.9):
    ids = tok.encode(f"User: {prompt}\nAssistant:")
    generated = list(ids)
    logps = []
    cur = torch.tensor([ids], dtype=torch.long, device=device)
    state = None
    model.eval()
    for _ in range(max_new):
        out = model(cur, state=state)
        state = out.state
        logits = out.logits[:, -1, :] / max(temperature, 1e-5)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        nid = dist.sample()
        logps.append(dist.log_prob(nid))
        generated.append(int(nid.item()))
        cur = nid.view(1, 1)
    text = tok.decode(generated[len(ids) :])
    logprob_sum = torch.stack(logps).sum() if logps else torch.tensor(0.0, device=device)
    return text, logprob_sum, generated


@torch.no_grad()
def score_with_rm(rm, tok, prompt: str, answer: str, device) -> float:
    x, y = pack_pair(tok, prompt, answer)
    from neurofield.rlhf.rewards import sequence_logprob

    s = sequence_logprob(rm, x.unsqueeze(0).to(device), y.unsqueeze(0).to(device))
    return float(s.item())


def stage_ppo(policy, ref, rm, tok, prompts, device, steps, lr, clip_eps, kl_coef):
    opt = AdamW(policy.parameters(), lr=lr)
    ref.eval()
    rm.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    for p in rm.parameters():
        p.requires_grad_(False)
    print(f"[ppo] steps={steps}")
    baseline = 0.0
    for step in range(1, steps + 1):
        prompt = prompts[(step - 1) % len(prompts)]
        # sample under policy
        text, logp_new, _ = sample_response(policy, tok, prompt, device)
        with torch.no_grad():
            # old logprob ≈ detach new for single-step
            logp_old = logp_new.detach()
            # ref logprob on same tokens — recompute cheaply via NLL on produced text
            x, y = pack_pair(tok, prompt, text)
            from neurofield.rlhf.rewards import sequence_logprob

            logp_ref = sequence_logprob(ref, x.unsqueeze(0).to(device), y.unsqueeze(0).to(device)).squeeze()
            reward = score_with_rm(rm, tok, prompt, text, device)
        baseline = 0.9 * baseline + 0.1 * reward
        adv = reward - baseline
        policy.train()
        # re-evaluate logprob with grad: teacher force on sampled text
        x, y = pack_pair(tok, prompt, text)
        from neurofield.rlhf.rewards import sequence_logprob

        logp_grad = sequence_logprob(policy, x.unsqueeze(0).to(device), y.unsqueeze(0).to(device)).squeeze()
        loss = ppo_policy_loss(
            logp_grad,
            logp_old if logp_old.dim() == 0 else logp_old,
            torch.tensor(adv, device=device),
            clip_eps=clip_eps,
            kl_coef=kl_coef,
            logprob_ref=logp_ref,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        if step % 20 == 0 or step == 1:
            print(f"  ppo {step} reward={reward:.3f} adv={adv:.3f} text={text[:50]!r}")
    return policy


def save_model(model, tok, out, stage, extra_meta=None):
    cfg = model.cfg if hasattr(model, "cfg") else None
    # read dims from embed
    d_model = model.embed.embedding_dim
    vocab_size = model.embed.num_embeddings
    config_dict = {
        "d_model": d_model,
        "d_k": max(8, d_model // 2),
        "d_v": max(8, d_model // 2),
        "n_skills": 4,
        "top_k": 2,
        "k_max": 2,
        "dendrite_window": 3,
        "vocab_size": vocab_size,
        "tie_embeddings": True,
        "tokenizer": "char",
    }
    meta = {"product": "AERIS_64", "stage": stage, "pipeline": "rlhf_full"}
    if extra_meta:
        meta.update(extra_meta)
    path = save_checkpoint(out, model, config=config_dict, stoi=getattr(tok, "stoi", None), itos=getattr(tok, "itos", None), meta=meta)
    (Path(path) / "tokenizer.json").write_text(json.dumps(tok.to_vocab_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", path)
    return path


def main():
    ap = argparse.ArgumentParser(description="Full RLHF-style: reward + DPO + PPO")
    ap.add_argument("--resume", default="docs/AERIS_64")
    ap.add_argument("--prefs", default="data/align/preferences.jsonl")
    ap.add_argument("--prompts", default="data/rl_prompts.txt")
    ap.add_argument("--stages", default="reward,dpo,ppo", help="comma: reward,dpo,ppo")
    ap.add_argument("--reward-steps", type=int, default=150)
    ap.add_argument("--dpo-steps", type=int, default=150)
    ap.add_argument("--ppo-steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dpo-beta", type=float, default=0.1)
    ap.add_argument("--ppo-clip", type=float, default=0.2)
    ap.add_argument("--kl-coef", type=float, default=0.05)
    ap.add_argument("--preset", default="tiny")
    ap.add_argument("--out", default="docs/AERIS_64")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    prefs = load_prefs(args.prefs)
    prompts_path = Path(args.prompts)
    prompts = [
        ln.strip()
        for ln in (prompts_path.read_text(encoding="utf-8").splitlines() if prompts_path.is_file() else ["hello", "who are you"])
        if ln.strip() and not ln.startswith("#")
    ]

    blob = "\n".join(p["prompt"] + p["chosen"] + p["rejected"] for p in prefs) + "\n".join(prompts)
    tok = CharTokenizer.from_text(blob, with_specials=True)
    root = Path(args.resume)
    if (root / "vocab.json").is_file():
        try:
            raw = json.loads((root / "vocab.json").read_text(encoding="utf-8"))
            if "stoi" in raw:
                tok = CharTokenizer(stoi=raw["stoi"], itos={int(k): v for k, v in raw.get("itos", {}).items()})
        except Exception:
            pass

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    t0 = time.time()

    policy = build_model(tok, args.resume, args.preset, device)
    ref = copy.deepcopy(policy).to(device)
    ref.eval()
    rm = copy.deepcopy(policy).to(device)

    if "reward" in stages:
        rm = stage_reward(rm, tok, prefs, device, args.reward_steps, args.lr)
        save_model(rm, tok, str(Path(args.out).parent / "AERIS_reward"), "reward_model")

    if "dpo" in stages:
        policy = stage_dpo(policy, ref, tok, prefs, device, args.dpo_steps, args.lr, args.dpo_beta)

    if "ppo" in stages:
        policy = stage_ppo(
            policy, ref, rm, tok, prompts, device, args.ppo_steps, args.lr * 0.5, args.ppo_clip, args.kl_coef
        )

    save_model(policy, tok, args.out, "rlhf_policy", {"stages": stages})
    print(f"Full RLHF pipeline done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
