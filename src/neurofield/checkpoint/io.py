"""
Single-folder checkpoint format (industry style):

  my_model/
    model.safetensors   # weights only (safe, no pickle)
    config.json         # architecture + vocab size
    vocab.json          # optional stoi/itos for char models
    meta.json           # training meta (steps, source, …)

Why not only .pt?
  - torch.save uses pickle → arbitrary code execution risk
  - safetensors is the HuggingFace / industry standard for weight files
  - one folder = no multi-file confusion about which .pt is "the" model
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional

import torch
try:
    from safetensors.torch import load_file, save_file
    _HAS_ST = True
except ImportError:  # pragma: no cover
    _HAS_ST = False
    load_file = save_file = None  # type: ignore



def _torch_load(path, map_location: str = "cpu"):
    """Load a legacy .pt WITHOUT executing pickle code unless explicitly allowed.

    weights_only=True only rebuilds tensors and plain containers. Files that
    need arbitrary unpickling are refused unless NEUROFIELD_ALLOW_PICKLE=1
    (use that only for files you created yourself).
    """
    try:
        return torch.load(str(path), map_location=map_location, weights_only=True)
    except Exception as e:  # pickle.UnpicklingError, RuntimeError, ...
        if os.environ.get("NEUROFIELD_ALLOW_PICKLE", "0") == "1":
            return torch.load(str(path), map_location=map_location, weights_only=False)
        raise RuntimeError(
            f"Refused to unpickle {path} ({e}). Convert it to safetensors "
            "(neurofield.checkpoint.io.convert_pt_to_safetensors) or, for a file "
            "you trust, set NEUROFIELD_ALLOW_PICKLE=1."
        ) from e


def _save_tensors(tensors: dict, path: str) -> None:
    """Save weights as safetensors only (no pickle .pt)."""
    if not _HAS_ST:
        raise ImportError("safetensors is required: pip install safetensors")
    if not path.endswith(".safetensors"):
        path = path.rsplit(".", 1)[0] + ".safetensors"
    save_file(tensors, path)


def _load_tensors(path: str, device: str = "cpu") -> dict:
    """Load weights from a .safetensors file only (no .pt / pickle)."""
    path_p = Path(path)
    if path_p.suffix != ".safetensors":
        raise ValueError(
            f"Refusing to load {path_p}: only .safetensors is supported (no .pt fallback)."
        )
    if not _HAS_ST:
        raise ImportError("safetensors is required: pip install safetensors")
    if not path_p.is_file():
        raise FileNotFoundError(f"No tensor file at {path}")
    return load_file(str(path_p), device=device)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    *,
    config: Optional[dict[str, Any]] = None,
    stoi: Optional[dict[str, int]] = None,
    itos: Optional[dict[int, str]] = None,
    meta: Optional[dict[str, Any]] = None,
) -> Path:
    """
    Save ONE model into a folder (or create path as folder).
    Returns the folder path.
    """
    root = Path(path)
    if root.suffix in {".pt", ".pth", ".bin", ".safetensors"}:
        root = root.with_suffix("")  # turn file-like name into folder
    root.mkdir(parents=True, exist_ok=True)

    state = model.state_dict()
    # safetensors forbids tensors that share storage (e.g. tied embed/head).
    # Clone any shared tensors so each key is independent on disk.
    tensors = {}
    seen_data_ptrs: dict[int, str] = {}
    for k, v in state.items():
        t = v.detach().cpu()
        ptr = t.untyped_storage().data_ptr() if t.numel() > 0 else id(t)
        if ptr in seen_data_ptrs:
            t = t.clone()
        else:
            seen_data_ptrs[ptr] = k
        tensors[k] = t.contiguous()
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        if not _HAS_ST:
            raise ImportError("safetensors is required: pip install safetensors")
        _save_tensors(tensors, str(td_path / "model.safetensors"))
        if config is not None:
            (td_path / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        if stoi is not None or itos is not None:
            vocab = {
                "stoi": stoi or {},
                "itos": {str(k): v for k, v in (itos or {}).items()},
            }
            (td_path / "vocab.json").write_text(json.dumps(vocab, indent=2), encoding="utf-8")
        meta_out = dict(meta or {})
        meta_out["format"] = "neurofield-safetensors-v1"
        meta_out["weights"] = "model.safetensors"
        meta_out.setdefault("framework", "pytorch")
        (td_path / "meta.json").write_text(json.dumps(meta_out, indent=2), encoding="utf-8")
        if root.exists():
            shutil.rmtree(root)
        shutil.copytree(td_path, root)
    return root


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """
    Load from:
      - folder with model.safetensors
      - single .safetensors file
      - .pt / .pth are NOT loaded (no pickle fallback)

    Returns dict with keys: model (state_dict), config, stoi, itos, meta
    """
    path = Path(path)
    out: dict[str, Any] = {
        "model": None,
        "config": {},
        "stoi": {},
        "itos": {},
        "meta": {},
    }

    # --- Folder (preferred) ---
    if path.is_dir():
        st_path = path / "model.safetensors"
        if not st_path.is_file():
            raise FileNotFoundError(
                f"No model.safetensors in {path}. "
                "This project does not load .pt checkpoints. Re-train or convert to safetensors."
            )
        out["model"] = _load_tensors(str(st_path), "cpu")
        if (path / "config.json").is_file():
            out["config"] = json.loads((path / "config.json").read_text(encoding="utf-8"))
        if (path / "vocab.json").is_file():
            vocab = json.loads((path / "vocab.json").read_text(encoding="utf-8"))
            out["stoi"] = vocab.get("stoi") or {}
            out["itos"] = {int(k): v for k, v in (vocab.get("itos") or {}).items()}
        if (path / "meta.json").is_file():
            out["meta"] = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        return out

    # --- Single safetensors file ---
    if path.suffix == ".safetensors":
        out["model"] = _load_tensors(str(path), "cpu")
        # sidecar config/vocab if present
        for name, key in [("config.json", "config"), ("vocab.json", "vocab"), ("meta.json", "meta")]:
            side = path.with_name(name)
            if side.is_file():
                data = json.loads(side.read_text(encoding="utf-8"))
                if key == "vocab":
                    out["stoi"] = data.get("stoi") or {}
                    out["itos"] = {int(k): v for k, v in (data.get("itos") or {}).items()}
                else:
                    out[key] = data
        return out

    # --- .pt / .pth / .bin: no fallback ---
    if path.suffix in {".pt", ".pth", ".bin"}:
        raise ValueError(
            f"Refusing to load {path}: .pt pickle checkpoints are disabled. "
            "Convert once with convert_pt_to_safetensors() or re-train to model.safetensors."
        )

    raise ValueError(f"Unsupported checkpoint path: {path} (need folder with model.safetensors or a .safetensors file)")



def convert_pt_to_safetensors(
    pt_path: str | Path,
    out_dir: str | Path,
    model: Optional[torch.nn.Module] = None,
) -> Path:
    """Convert a legacy .pt file into the folder safetensors format."""
    import shutil
    import tempfile

    ckpt = load_checkpoint(pt_path)
    root = Path(out_dir)
    state = ckpt["model"]
    if not isinstance(state, dict):
        raise TypeError("Checkpoint has no state_dict")

    tensors: dict[str, torch.Tensor] = {}
    seen: set[int] = set()
    for k, v in state.items():
        if not torch.is_tensor(v):
            continue
        tv = v.detach().cpu()
        ptr = tv.untyped_storage().data_ptr() if tv.numel() > 0 else id(tv)
        if ptr in seen:
            tv = tv.clone()
        else:
            seen.add(ptr)
        tensors[k] = tv.contiguous()

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        if not _HAS_ST:
            raise ImportError("safetensors is required: pip install safetensors")
        _save_tensors(tensors, str(td_path / "model.safetensors"))
        cfg = ckpt.get("config") or {}
        if hasattr(cfg, "to_dict"):
            cfg = cfg.to_dict()
        if isinstance(cfg, dict) and "model" in cfg and isinstance(cfg["model"], dict):
            cfg = cfg["model"]
        (td_path / "config.json").write_text(
            json.dumps(cfg, indent=2, default=str), encoding="utf-8"
        )
        if ckpt.get("stoi") or ckpt.get("itos"):
            vocab = {
                "stoi": ckpt.get("stoi") or {},
                "itos": {str(k): v for k, v in (ckpt.get("itos") or {}).items()},
            }
            (td_path / "vocab.json").write_text(json.dumps(vocab, indent=2), encoding="utf-8")
        meta = dict(ckpt.get("meta") or {})
        meta["format"] = "neurofield-safetensors-v1"
        meta["converted_from"] = str(pt_path)
        (td_path / "meta.json").write_text(
            json.dumps(meta, indent=2, default=str), encoding="utf-8"
        )
        if root.exists():
            shutil.rmtree(root)
        shutil.copytree(td_path, root)
    return root
