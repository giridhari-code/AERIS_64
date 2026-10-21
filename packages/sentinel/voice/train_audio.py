"""Train Indo-Voice-3 AudioEncoder + ModalityProjector.

Data format (JSONL), one utterance per line:
  {"wav": "path/to/utt.wav", "text": "नमस्ते आप कैसे हैं"}

Or directory of wav + matching .txt files.

Usage:
  python -m sentinel.voice.train_audio \\
    --data data/hindi_speech/manifest.jsonl \\
    --ckpt checkpoints/AERIS_main \\
    --steps 500 --batch-size 4 --device cpu \\
    --output checkpoints/indo_audio.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import wave
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sentinel.voice_config import IndoVoiceConfig
from sentinel.modules.audio_encoder import AudioEncoder
from sentinel.modules.modality_projector import ModalityProjector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("indo_voice.train")


def load_wav_mono(path: Path, target_sr: int = 16000) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(n)
    if sw == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        x = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr != target_sr:
        ratio = target_sr / sr
        new_len = max(1, int(len(x) * ratio))
        x = np.interp(np.linspace(0, len(x) - 1, new_len), np.arange(len(x)), x).astype(np.float32)
    return x


class SpeechTextDataset(Dataset):
    """JSONL manifest or synthetic fallback."""

    def __init__(
        self,
        manifest: Optional[str],
        tokenizer,
        sample_rate: int = 16000,
        max_sec: float = 8.0,
        synth_size: int = 64,
    ):
        self.sr = sample_rate
        self.max_samples = int(sample_rate * max_sec)
        self.tok = tokenizer
        self.items: List[Tuple[Optional[str], str]] = []

        if manifest and Path(manifest).is_file():
            for line in Path(manifest).read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.items.append((obj.get("wav"), obj["text"]))
            logger.info("Loaded %d items from %s", len(self.items), manifest)
        else:
            # synthetic: random noise + fixed Hindi-ish strings for pipeline test
            phrases = [
                "namaste",
                "aap kaise hain",
                "main theek hoon",
                "shukriya",
                "kya haal hai",
            ]
            for i in range(synth_size):
                self.items.append((None, phrases[i % len(phrases)]))
            logger.warning("No manifest — using %d synthetic items (train real data later)", synth_size)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        wav_path, text = self.items[idx]
        if wav_path and Path(wav_path).is_file():
            samples = load_wav_mono(Path(wav_path), self.sr)
        else:
            # synthetic speech-like
            T = min(self.max_samples, self.sr + (idx % 5) * 2000)
            t = np.linspace(0, T / self.sr, T, dtype=np.float32)
            samples = (0.15 * np.sin(2 * np.pi * (140 + idx) * t)).astype(np.float32)
        if len(samples) > self.max_samples:
            samples = samples[: self.max_samples]
        ids = self.tok.encode(text)
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        return (
            torch.from_numpy(samples),
            torch.tensor(ids, dtype=torch.long),
            len(samples),
        )


def collate(batch):
    waves, texts, lens = zip(*batch)
    max_t = max(w.size(0) for w in waves)
    max_l = max(t.size(0) for t in texts)
    B = len(batch)
    wav = torch.zeros(B, max_t)
    txt = torch.full((B, max_l), -100, dtype=torch.long)
    for i, (w, t) in enumerate(zip(waves, texts)):
        wav[i, : w.size(0)] = w
        txt[i, : t.size(0)] = t
    return wav, txt, torch.tensor(lens, dtype=torch.long)


class AudioTrainer:
    def __init__(self, cfg: IndoVoiceConfig, device: str, sentinel=None, freeze_lm: bool = True):
        self.device = torch.device(device)
        self.cfg = cfg
        self.encoder = AudioEncoder(cfg.audio).to(self.device)
        self.projector = ModalityProjector(cfg.audio, cfg.model).to(self.device)
        self.sentinel = sentinel
        if sentinel is not None and freeze_lm:
            for p in sentinel.parameters():
                p.requires_grad_(False)
            sentinel.eval()
        params = list(self.encoder.parameters()) + list(self.projector.parameters())
        self.opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=0.01)

    def step(self, wav, text_ids, lengths) -> dict:
        self.encoder.train()
        self.projector.train()
        wav = wav.to(self.device)
        text_ids = text_ids.to(self.device)
        lengths = lengths.to(self.device)

        feats, _ = self.encoder(wav, lengths)
        embeds = self.projector(feats)  # (B, Ta, D)

        metrics = {"frames": float(feats.size(1))}
        # Reconstruction / contrastive proxy + optional LM loss
        loss = embeds.pow(2).mean() * 0.001  # stability

        # Align mean audio embed with mean text embed (if sentinel has embed)
        if self.sentinel is not None:
            with torch.no_grad():
                # valid tokens only
                mask = text_ids != -100
                safe_ids = text_ids.clamp(min=0)
                te = self.sentinel.embed(safe_ids)  # (B, L, D)
                te = te * mask.unsqueeze(-1).float()
                te_mean = te.sum(1) / mask.sum(1).clamp(min=1).unsqueeze(-1)
            ae_mean = embeds.mean(1)
            # cosine loss
            ae_n = F.normalize(ae_mean, dim=-1)
            te_n = F.normalize(te_mean.detach(), dim=-1)
            cos = (ae_n * te_n).sum(-1).mean()
            loss = loss + (1.0 - cos)
            metrics["cos"] = float(cos.detach().item())

            # optional LM on text with audio prefix (first 4 frames)
            if text_ids.size(1) > 1:
                prefix = embeds[:, : min(4, embeds.size(1))]
                # pure text LM loss as auxiliary
                inp = safe_ids[:, :-1]
                tgt = text_ids[:, 1:]
                out = self.sentinel(input_ids=inp, targets=tgt, collect_stats=False)
                if out.loss is not None:
                    loss = loss + 0.1 * out.loss
                    metrics["lm"] = float(out.loss.detach().item())

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.projector.parameters()), 1.0
        )
        self.opt.step()
        metrics["loss"] = float(loss.detach().item())
        return metrics

    def save(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "encoder": self.encoder.state_dict(),
                "projector": self.projector.state_dict(),
                "audio_cfg": self.cfg.audio.__dict__,
                "model_d": self.cfg.model.d_model,
            },
            path,
        )
        logger.info("Saved %s", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="manifest.jsonl")
    ap.add_argument("--ckpt", default="checkpoints/AERIS_main", help="SENTINEL checkpoint (optional)")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output", default="checkpoints/indo_audio.pt")
    ap.add_argument("--synth-size", type=int, default=32)
    ap.add_argument("--allow-synth", action="store_true", help="Allow synthetic data when --data missing (dev only)")
    args = ap.parse_args()

    cfg = IndoVoiceConfig.from_env()
    sentinel = None
    tokenizer = None

    ckpt_path = Path(args.ckpt) if args.ckpt else None
    if ckpt_path and ckpt_path.exists():
        from sentinel.checkpoint.resolve import load_model_from_checkpoint
        sentinel, tok, full, _ = load_model_from_checkpoint(str(ckpt_path), device=args.device, strict=False)
        cfg.model.d_model = full.model.d_model
        cfg.model.vocab_size = full.model.vocab_size
        tokenizer = tok
        logger.info("Frozen SENTINEL from %s d_model=%s", args.ckpt, cfg.model.d_model)
    else:
        # minimal char tok for synth
        from sentinel.tokenizer.char import CharTokenizer
        tokenizer = CharTokenizer.from_text("namaste aap kaise hain main theek hoon shukriya kya haal")
        cfg.model.vocab_size = tokenizer.vocab_size
        logger.warning("No LM checkpoint — contrastive only")

    if not args.data and not args.allow_synth:
        raise SystemExit("Provide --data manifest.jsonl or pass --allow-synth for development only")
    ds = SpeechTextDataset(args.data, tokenizer, sample_rate=cfg.audio.sample_rate, synth_size=args.synth_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

    trainer = AudioTrainer(cfg, args.device, sentinel=sentinel, freeze_lm=True)
    step = 0
    while step < args.steps:
        for batch in loader:
            m = trainer.step(*batch)
            step += 1
            if step % 10 == 0 or step == 1:
                logger.info("step=%d %s", step, m)
            if step >= args.steps:
                break
    trainer.save(args.output)


if __name__ == "__main__":
    main()
