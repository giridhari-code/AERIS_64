"""Which cause makes decode differ from the full pass? Turn the suspects off one by one.

Put this file in the repo root (~/AERIS_64) and run:   PYTHONPATH=src python decode_gap_experiment.py

A  baseline
B  the dendrite error path off   (gamma = 0, a_s = 0)
C  error path AND the read fed to the field off   (W_m = 0)
D  same as C with k_max = 2 (more than one depth pass)

If C is about 0 and D is not, the remaining gap comes from the memory of passes after the first.
"""
import torch

from neurofield.config import AblationConfig, ModelConfig, SafetyConfig
from neurofield.model import NeuroField

V = 40


def make(k_max):
    cfg = ModelConfig(d_model=32, d_k=16, d_v=16, n_skills=2, top_k=1, k_max=k_max,
                      vocab_size=V, max_seq_len=64, dendrite_window=3, dropout=0.0)
    m = NeuroField(cfg, SafetyConfig(enable_audit=False, truncate_write_window=64))
    m.set_ablation(AblationConfig(use_slow_memory=False))
    return m.eval()


def gap(m, x):
    with torch.no_grad():
        full = m(x).logits[:, -1]
        step = m(x[:, -1:], state=m(x[:, :-1]).state).logits[:, -1]
    return (full - step).abs().max().item()


def off(m, error=False, rshift=False):
    with torch.no_grad():
        if error:
            m.dendrite.gamma.zero_()
            m.neuromod.a_s.zero_()
        if rshift:
            m.field.W_m.weight.zero_()
    return m


x = torch.randint(0, V, (1, 12))
for label, k, err, rs in [
    ("A  k=1 baseline            ", 1, False, False),
    ("B  k=1 error path off      ", 1, True, False),
    ("C  k=1 error + r-shift off ", 1, True, True),
    ("D  k=2 error + r-shift off ", 2, True, True),
]:
    torch.manual_seed(0)
    print(label, "%.6f" % gap(off(make(k), err, rs), x))
