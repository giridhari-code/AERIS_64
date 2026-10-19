"""All company-standard pipeline stages."""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch.optim import AdamW

from sentinel.data.dataset import build_dataloader
from sentinel.pipeline.context import PipelineContext
from sentinel.pipeline.lora import apply_lora, lora_parameters, merge_lora_
from sentinel.pipeline.pref_data import byte_encode, preference_loader, sft_loader
from sentinel.pipeline.registry import register
from sentinel.rlhf.dpo import dpo_loss
from sentinel.rlhf.ppo import ppo_policy_loss
from sentinel.rlhf.rewards import pairwise_reward_loss, sequence_logprob
from sentinel.training.trainer import Trainer

logger = logging.getLogger(__name__)


def _encode_fn(ctx: PipelineContext):
    vs = int(ctx.cfg.model.vocab_size)

    def enc(text: str) -> list[int]:
        return byte_encode(text, vocab_size=vs)

    return enc


def _lm_train(ctx: PipelineContext, stage: str, overrides: dict[str, Any] | None = None) -> None:
    """Shared causal-LM training via Trainer (pretrain / midtrain / long_context)."""
    o = overrides or {}
    tc = ctx.cfg.training
    for k, v in o.items():
        if hasattr(tc, k):
            setattr(tc, k, v)
    train_path = ctx.train_bin or ctx.cfg.data.train_path
    if not train_path:
        raise ValueError(f"{stage}: need train_bin / data.train_path (run prepare_data first)")
    tc.output_dir = str(ctx.stage_dir(stage))
    train_loader = build_dataloader(
        train_path,
        seq_len=tc.seq_len,
        batch_size=tc.batch_size,
        streaming=bool(o.get("streaming", False)),
        num_workers=int(o.get("num_workers", 0)),
        dtype=str(o.get("dtype", "uint16")),
        stride=int(o.get("stride", 1)),
    )
    val_loader = None
    if ctx.val_bin or ctx.cfg.data.val_path:
        val_loader = build_dataloader(
            ctx.val_bin or ctx.cfg.data.val_path,
            seq_len=tc.seq_len,
            batch_size=tc.batch_size,
            streaming=False,
            num_workers=0,
            stride=max(1, tc.seq_len),
            shuffle_train=False,
            drop_last=False,
        )
    trainer = Trainer(ctx.model, ctx.cfg, train_loader, val_loader)
    trainer.train()
    ctx.model = trainer.model
    ctx.save_model(stage)


@register("pretrain")
def stage_pretrain(ctx: PipelineContext) -> None:
    """Large-scale next-token pretraining on token bins."""
    sc = ctx.stage_cfg.get("pretrain", {})
    logger.info("=== STAGE: pretrain ===")
    _lm_train(ctx, "pretrain", sc)


@register("midtrain")
def stage_midtrain(ctx: PipelineContext) -> None:
    """Continued pretraining on higher-quality / domain mix (lower LR)."""
    sc = dict(ctx.stage_cfg.get("midtrain", {}))
    sc.setdefault("lr", float(ctx.cfg.training.lr) * 0.3)
    sc.setdefault("max_steps", min(5000, int(ctx.cfg.training.max_steps)))
    logger.info("=== STAGE: midtrain ===")
    _lm_train(ctx, "midtrain", sc)


@register("long_context")
def stage_long_context(ctx: PipelineContext) -> None:
    """Extend context: longer seq_len, adjusted truncate window / slots."""
    sc = dict(ctx.stage_cfg.get("long_context", {}))
    seq = int(sc.get("seq_len", max(512, ctx.cfg.training.seq_len * 2)))
    sc["seq_len"] = seq
    sc.setdefault("lr", float(ctx.cfg.training.lr) * 0.2)
    sc.setdefault("batch_size", max(1, int(ctx.cfg.training.batch_size) // 2))
    sc.setdefault("max_steps", min(3000, int(ctx.cfg.training.max_steps)))
    # model context ceiling
    ctx.cfg.model.max_seq_len = max(int(ctx.cfg.model.max_seq_len), seq)
    ctx.model.cfg.max_seq_len = ctx.cfg.model.max_seq_len
    ctx.model._truncate_window = max(16, int(sc.get("truncate_write_window", ctx.model._truncate_window)))
    logger.info("=== STAGE: long_context seq_len=%d ===", seq)
    _lm_train(ctx, "long_context", sc)


@register("sft")
def stage_sft(ctx: PipelineContext) -> None:
    """Supervised fine-tuning on instruction / dialogue data."""
    sc = ctx.stage_cfg.get("sft", {})
    path = ctx.sft_path or sc.get("data")
    if not path:
        raise ValueError("sft: set --sft-data or stage_cfg.sft.data")
    logger.info("=== STAGE: sft (%s) ===", path)
    steps = int(sc.get("max_steps", 1000))
    lr = float(sc.get("lr", 1e-4))
    max_len = int(sc.get("max_len", ctx.cfg.training.seq_len))
    bs = int(sc.get("batch_size", min(4, ctx.cfg.training.batch_size)))
    loader = sft_loader(path, _encode_fn(ctx), bs, max_len)
    opt = AdamW((p for p in ctx.model.parameters() if p.requires_grad), lr=lr)
    ctx.model.train()
    step = 0
    while step < steps:
        for batch in loader:
            if step >= steps:
                break
            x = batch["input_ids"].to(ctx.device)
            y = batch["targets"].to(ctx.device)
            out = ctx.model(x, targets=y)
            loss = out.loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ctx.model.parameters(), 1.0)
            opt.step()
            step += 1
            if ctx.is_main and step % 50 == 0:
                logger.info("sft step %d loss=%.4f", step, float(loss.detach()))
    ctx.save_model("sft")


@register("reasoning")
def stage_reasoning(ctx: PipelineContext) -> None:
    """SFT on chain-of-thought / reasoning traces (same loop, dedicated data)."""
    sc = dict(ctx.stage_cfg.get("reasoning", {}))
    path = ctx.reasoning_path or sc.get("data") or ctx.sft_path
    if not path:
        raise ValueError("reasoning: set --reasoning-data or --sft-data")
    ctx.sft_path = path
    ctx.stage_cfg["sft"] = {**sc, "data": path}
    logger.info("=== STAGE: reasoning ===")
    stage_sft(ctx)
    # rename artifact
    ctx.save_model("reasoning")


@register("preference_bt")
def stage_preference_bt(ctx: PipelineContext) -> None:
    """Bradley-Terry reward model: P(chosen ≻ rejected) = σ(s_c - s_r)."""
    sc = ctx.stage_cfg.get("preference_bt", {})
    path = ctx.prefs_path or sc.get("data")
    if not path:
        raise ValueError("preference_bt: set --prefs")
    logger.info("=== STAGE: preference_bt (Bradley-Terry) ===")
    steps = int(sc.get("max_steps", 500))
    lr = float(sc.get("lr", 5e-5))
    loader = preference_loader(
        path, _encode_fn(ctx), int(sc.get("batch_size", 2)), int(sc.get("max_len", 128))
    )
    opt = AdamW(ctx.model.parameters(), lr=lr)
    ctx.model.train()
    step = 0
    while step < steps:
        for batch in loader:
            if step >= steps:
                break
            loss = pairwise_reward_loss(
                ctx.model,
                batch["chosen_x"].to(ctx.device),
                batch["chosen_y"].to(ctx.device),
                batch["rejected_x"].to(ctx.device),
                batch["rejected_y"].to(ctx.device),
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if ctx.is_main and step % 50 == 0:
                logger.info("bt step %d loss=%.4f", step, float(loss.detach()))
    ctx.reward_model = ctx.model  # same weights act as RM scores via logprob
    ctx.save_model("preference_bt")


@register("dpo")
def stage_dpo(ctx: PipelineContext) -> None:
    """Direct Preference Optimization (Rafailov et al.)."""
    sc = ctx.stage_cfg.get("dpo", {})
    path = ctx.prefs_path or sc.get("data")
    if not path:
        raise ValueError("dpo: set --prefs")
    if ctx.ref_model is None:
        ctx.freeze_as_ref()
    logger.info("=== STAGE: dpo ===")
    beta = float(sc.get("beta", 0.1))
    steps = int(sc.get("max_steps", 500))
    lr = float(sc.get("lr", 5e-5))
    loader = preference_loader(
        path, _encode_fn(ctx), int(sc.get("batch_size", 2)), int(sc.get("max_len", 128))
    )
    opt = AdamW((p for p in ctx.model.parameters() if p.requires_grad), lr=lr)
    ctx.model.train()
    step = 0
    while step < steps:
        for batch in loader:
            if step >= steps:
                break
            loss = dpo_loss(
                ctx.model,
                ctx.ref_model,
                batch["chosen_x"].to(ctx.device),
                batch["chosen_y"].to(ctx.device),
                batch["rejected_x"].to(ctx.device),
                batch["rejected_y"].to(ctx.device),
                beta=beta,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if ctx.is_main and step % 50 == 0:
                logger.info("dpo step %d loss=%.4f", step, float(loss.detach()))
    ctx.save_model("dpo")


@register("orpo")
def stage_orpo(ctx: PipelineContext) -> None:
    """Odds-Ratio Preference Optimization (reference-free preference)."""
    sc = ctx.stage_cfg.get("orpo", {})
    path = ctx.prefs_path or sc.get("data")
    if not path:
        raise ValueError("orpo: set --prefs")
    logger.info("=== STAGE: orpo ===")
    steps = int(sc.get("max_steps", 500))
    lr = float(sc.get("lr", 5e-5))
    beta = float(sc.get("beta", 0.1))
    loader = preference_loader(
        path, _encode_fn(ctx), int(sc.get("batch_size", 2)), int(sc.get("max_len", 128))
    )
    opt = AdamW(ctx.model.parameters(), lr=lr)
    ctx.model.train()
    step = 0
    while step < steps:
        for batch in loader:
            if step >= steps:
                break
            # SFT on chosen + odds-ratio penalty on rejected
            out_c = ctx.model(
                batch["chosen_x"].to(ctx.device),
                targets=batch["chosen_y"].to(ctx.device),
            )
            log_c = sequence_logprob(
                ctx.model, batch["chosen_x"].to(ctx.device), batch["chosen_y"].to(ctx.device)
            )
            log_r = sequence_logprob(
                ctx.model, batch["rejected_x"].to(ctx.device), batch["rejected_y"].to(ctx.device)
            )
            # ORPO: -log σ(β * log odds)
            odds = log_c - log_r
            or_loss = -torch.nn.functional.logsigmoid(beta * odds).mean()
            loss = out_c.loss + or_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if ctx.is_main and step % 50 == 0:
                logger.info("orpo step %d loss=%.4f", step, float(loss.detach()))
    ctx.save_model("orpo")


@register("lora_pref")
def stage_lora_pref(ctx: PipelineContext) -> None:
    """LoRA adapters + DPO (parameter-efficient preference tuning)."""
    sc = ctx.stage_cfg.get("lora_pref", {})
    path = ctx.prefs_path or sc.get("data")
    if not path:
        raise ValueError("lora_pref: set --prefs")
    logger.info("=== STAGE: lora_pref ===")
    rank = int(sc.get("rank", 8))
    alpha = float(sc.get("alpha", 16.0))
    adapted = apply_lora(ctx.model, rank=rank, alpha=alpha)
    ctx.lora_enabled = True
    if ctx.is_main:
        logger.info("LoRA adapted %d modules (rank=%d)", len(adapted), rank)
    if ctx.ref_model is None:
        ctx.freeze_as_ref()
    # only train LoRA params
    params = lora_parameters(ctx.model)
    if not params:
        raise RuntimeError("lora_pref: no LoRA parameters found")
    beta = float(sc.get("beta", 0.1))
    steps = int(sc.get("max_steps", 400))
    lr = float(sc.get("lr", 1e-4))
    loader = preference_loader(
        path, _encode_fn(ctx), int(sc.get("batch_size", 2)), int(sc.get("max_len", 128))
    )
    opt = AdamW(params, lr=lr)
    ctx.model.train()
    step = 0
    while step < steps:
        for batch in loader:
            if step >= steps:
                break
            loss = dpo_loss(
                ctx.model,
                ctx.ref_model,
                batch["chosen_x"].to(ctx.device),
                batch["chosen_y"].to(ctx.device),
                batch["rejected_x"].to(ctx.device),
                batch["rejected_y"].to(ctx.device),
                beta=beta,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if ctx.is_main and step % 50 == 0:
                logger.info("lora_pref step %d loss=%.4f", step, float(loss.detach()))
    if sc.get("merge", True):
        n = merge_lora_(ctx.model)
        if ctx.is_main:
            logger.info("merged %d LoRA modules into base", n)
        ctx.lora_enabled = False
    ctx.save_model("lora_pref")


@register("grpo")
def stage_grpo(ctx: PipelineContext) -> None:
    """Group Relative Policy Optimization (group-normalized advantages)."""
    sc = ctx.stage_cfg.get("grpo", {})
    path = ctx.prefs_path or sc.get("data")
    if not path:
        raise ValueError("grpo: set --prefs")
    if ctx.ref_model is None:
        ctx.freeze_as_ref()
    logger.info("=== STAGE: grpo ===")
    # Treat preference pairs as group of 2: chosen vs rejected scores
    steps = int(sc.get("max_steps", 400))
    lr = float(sc.get("lr", 5e-5))
    clip = float(sc.get("clip_eps", 0.2))
    loader = preference_loader(
        path, _encode_fn(ctx), int(sc.get("batch_size", 2)), int(sc.get("max_len", 128))
    )
    opt = AdamW(ctx.model.parameters(), lr=lr)
    ctx.model.train()
    step = 0
    while step < steps:
        for batch in loader:
            if step >= steps:
                break
            cx = batch["chosen_x"].to(ctx.device)
            cy = batch["chosen_y"].to(ctx.device)
            rx = batch["rejected_x"].to(ctx.device)
            ry = batch["rejected_y"].to(ctx.device)
            with torch.no_grad():
                old_c = sequence_logprob(ctx.ref_model, cx, cy)
                old_r = sequence_logprob(ctx.ref_model, rx, ry)
            new_c = sequence_logprob(ctx.model, cx, cy)
            new_r = sequence_logprob(ctx.model, rx, ry)
            # group of 2: relative advantage
            scores = torch.stack([new_c.detach(), new_r.detach()], dim=-1)
            mean = scores.mean(dim=-1, keepdim=True)
            std = scores.std(dim=-1, keepdim=True).clamp(min=1e-4)
            adv_c = ((new_c.detach() - mean.squeeze(-1)) / std.squeeze(-1))
            adv_r = ((new_r.detach() - mean.squeeze(-1)) / std.squeeze(-1))
            loss_c = ppo_policy_loss(new_c, old_c, adv_c, clip_eps=clip)
            loss_r = ppo_policy_loss(new_r, old_r, adv_r, clip_eps=clip)
            loss = (loss_c + loss_r).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if ctx.is_main and step % 50 == 0:
                logger.info("grpo step %d loss=%.4f", step, float(loss.detach()))
    ctx.save_model("grpo")


@register("rlhf_ppo")
def stage_rlhf_ppo(ctx: PipelineContext) -> None:
    """On-policy PPO with reward = logprob margin vs rejected (or RM if set)."""
    sc = ctx.stage_cfg.get("rlhf_ppo", {})
    path = ctx.prefs_path or sc.get("data")
    if not path:
        raise ValueError("rlhf_ppo: set --prefs")
    if ctx.ref_model is None:
        ctx.freeze_as_ref()
    logger.info("=== STAGE: rlhf_ppo ===")
    steps = int(sc.get("max_steps", 300))
    lr = float(sc.get("lr", 3e-5))
    clip = float(sc.get("clip_eps", 0.2))
    kl_coef = float(sc.get("kl_coef", 0.05))
    loader = preference_loader(
        path, _encode_fn(ctx), int(sc.get("batch_size", 2)), int(sc.get("max_len", 128))
    )
    opt = AdamW(ctx.model.parameters(), lr=lr)
    ctx.model.train()
    step = 0
    while step < steps:
        for batch in loader:
            if step >= steps:
                break
            cx = batch["chosen_x"].to(ctx.device)
            cy = batch["chosen_y"].to(ctx.device)
            rx = batch["rejected_x"].to(ctx.device)
            ry = batch["rejected_y"].to(ctx.device)
            with torch.no_grad():
                old = sequence_logprob(ctx.ref_model, cx, cy)
                ref = sequence_logprob(ctx.ref_model, cx, cy)
                # reward: BT-style margin
                r_c = sequence_logprob(ctx.model, cx, cy)
                r_r = sequence_logprob(ctx.model, rx, ry)
                reward = (r_c - r_r).detach()
            new = sequence_logprob(ctx.model, cx, cy)
            adv = reward  # simple advantage
            loss = ppo_policy_loss(new, old, adv, clip_eps=clip, kl_coef=kl_coef, logprob_ref=ref)
            loss = loss.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if ctx.is_main and step % 50 == 0:
                logger.info("rlhf_ppo step %d loss=%.4f", step, float(loss.detach()))
    ctx.save_model("rlhf_ppo")


@register("opd")
def stage_opd(ctx: PipelineContext) -> None:
    """Online Preference Distillation: distill preferred policy into student via KL + CE.

    Teacher = current ref (or frozen preference-tuned weights);
    student = active model. Minimizes KL(student || teacher) on chosen responses
    plus standard CE — online distillation for deployment-sized models.
    """
    sc = ctx.stage_cfg.get("opd", {})
    path = ctx.prefs_path or ctx.sft_path or sc.get("data")
    if not path:
        raise ValueError("opd: set --prefs or --sft-data")
    if ctx.ref_model is None:
        ctx.freeze_as_ref()
    logger.info("=== STAGE: opd (online preference distillation) ===")
    steps = int(sc.get("max_steps", 400))
    lr = float(sc.get("lr", 5e-5))
    kl_w = float(sc.get("kl_weight", 0.5))
    # use chosen side of prefs or sft
    try:
        loader = preference_loader(
            path, _encode_fn(ctx), int(sc.get("batch_size", 2)), int(sc.get("max_len", 128))
        )
        use_pref = True
    except Exception:
        loader = sft_loader(
            path, _encode_fn(ctx), int(sc.get("batch_size", 2)), int(sc.get("max_len", 128))
        )
        use_pref = False
    opt = AdamW(ctx.model.parameters(), lr=lr)
    ctx.model.train()
    step = 0
    while step < steps:
        for batch in loader:
            if step >= steps:
                break
            if use_pref:
                x = batch["chosen_x"].to(ctx.device)
                y = batch["chosen_y"].to(ctx.device)
            else:
                x = batch["input_ids"].to(ctx.device)
                y = batch["targets"].to(ctx.device)
            out = ctx.model(x, targets=y)
            with torch.no_grad():
                t_out = ctx.ref_model(x, targets=y)
                t_logp = torch.nn.functional.log_softmax(t_out.logits, dim=-1)
            s_logp = torch.nn.functional.log_softmax(out.logits, dim=-1)
            # KL(student || teacher) tokenwise
            kl = torch.nn.functional.kl_div(s_logp, t_logp.exp(), reduction="batchmean")
            loss = out.loss + kl_w * kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if ctx.is_main and step % 50 == 0:
                logger.info("opd step %d loss=%.4f kl=%.4f", step, float(loss.detach()), float(kl.detach()))
    ctx.save_model("opd")
