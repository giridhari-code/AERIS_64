"""One place that turns a checkpoint path into (model, tokenizer, config).

Used by the server, the CLI and the eval scripts so they cannot drift apart.
Shapes in the weights are the source of truth for dimensions; a config.json that
is empty/missing/wrong no longer yields a mis-shaped or half-random model.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from neurofield.checkpoint.infer import infer_model_dims, remap_legacy_keys
from neurofield.checkpoint.io import load_checkpoint
from neurofield.config import ModelConfig, NeuroFieldConfig, SafetyConfig, load_config
from neurofield.model import NeuroField
from neurofield.tokenizer.bpe import BPETokenizer
from neurofield.tokenizer.char import CharTokenizer
from neurofield.tokenizer.hf import HFTokenizer

logger = logging.getLogger(__name__)


def load_tokenizer(checkpoint: str | Path, ckpt: dict[str, Any]):
    root = Path(checkpoint)
    tok_path = root / "tokenizer.json" if root.is_dir() else root.with_name("tokenizer.json")
    if tok_path.is_file():
        data = json.loads(tok_path.read_text(encoding="utf-8"))
        if HFTokenizer.is_hf_json(data):
            return HFTokenizer(tok_path)
        if data.get("type") == "bpe":
            return BPETokenizer.load(tok_path)
        return CharTokenizer.from_vocab_dict(data)
    return CharTokenizer(ckpt.get("stoi") or {}, ckpt.get("itos") or {})


def resolve_model_config(
    ckpt: dict[str, Any], config_path: Optional[str] = None
) -> tuple[ModelConfig, SafetyConfig, NeuroFieldConfig]:
    """Declared config supplies non-shape knobs; weight shapes win for dimensions."""
    state = ckpt["model"]
    if config_path:
        full = load_config(config_path)
        declared = dict(full.model.__dict__)
    else:
        full = NeuroFieldConfig()
        raw = ckpt.get("config") or {}
        if isinstance(raw, dict) and isinstance(raw.get("model"), dict):
            raw = raw["model"]
        fields = set(ModelConfig.__dataclass_fields__)
        given = {k: v for k, v in raw.items() if k in fields} if isinstance(raw, dict) else {}
        declared = {**ModelConfig().__dict__, **given}

    inferred = infer_model_dims(state)
    if not inferred:
        raise ValueError("Checkpoint has no recognisable NeuroField tensors (embed.weight ...)")
    for k, v in inferred.items():
        if k in declared and declared[k] != v:
            logger.warning("config says %s=%s but weights say %s; using weights", k, declared[k], v)
        declared[k] = v
    model_cfg = ModelConfig(**{k: declared[k] for k in ModelConfig.__dataclass_fields__ if k in declared})
    merged = NeuroFieldConfig(
        model=model_cfg, training=full.training, data=full.data, safety=full.safety, logging=full.logging
    )
    return model_cfg, full.safety, merged


def load_model_from_checkpoint(
    checkpoint: str | Path,
    config_path: Optional[str] = None,
    device: str = "cpu",
    strict: bool = True,
) -> tuple[NeuroField, Any, NeuroFieldConfig, dict[str, Any]]:
    """Return (model.eval() on `device`, tokenizer, merged config, meta)."""
    ckpt = load_checkpoint(checkpoint)
    ckpt["model"], renamed = remap_legacy_keys(ckpt["model"])
    if renamed:
        logger.warning("Renamed legacy tensor keys: %s", renamed)
    tok = load_tokenizer(checkpoint, ckpt)
    model_cfg, safety_cfg, full_cfg = resolve_model_config(ckpt, config_path)
    if tok.vocab_size > model_cfg.vocab_size:
        raise ValueError(
            f"Tokenizer vocab ({tok.vocab_size}) > model vocab ({model_cfg.vocab_size}): "
            "wrong tokenizer for this checkpoint."
        )
    model = NeuroField(model_cfg, safety_cfg)
    res = model.load_state_dict(ckpt["model"], strict=False)
    missing = list(getattr(res, "missing_keys", []))
    unexpected = list(getattr(res, "unexpected_keys", []))
    if unexpected:
        logger.warning("Ignoring %d unexpected keys: %s", len(unexpected), unexpected[:5])
    if missing:
        msg = f"{len(missing)} weights missing from checkpoint (would stay random): {missing[:5]}"
        if strict:
            raise RuntimeError(msg)
        logger.error(msg)
    model.to(device)
    model.eval()
    return model, tok, full_cfg, dict(ckpt.get("meta") or {})
