# Colab: test System-1 / System-2 under Metacognition

## 1. Setup

```python
!git clone https://github.com/giridhari-code/AERIS_64.git
%cd AERIS_64
!pip -q install torch --index-url https://download.pytorch.org/whl/cpu
```

## 2. Run dual-process smoke

```python
import os
os.environ["PYTHONPATH"] = "packages"
!python scripts/research/test_dual_process_colab.py
```

Expected:
- easy cases → `system1`, `k=1`
- hard conflict → `system2`, `k>=2`
- force flags override auto

## 3. Interactive probe

```python
import torch
from sentinel.modules.metacognition import Metacognition

meta = Metacognition(k_max=4, track_global=False)
meta.training = False

def probe(surprise, conflict, force=None):
    rel = torch.tensor([surprise] * 4)
    conf = torch.sigmoid(-2.0 * (rel - 1.0))
    confl = torch.tensor([conflict] * 4)
    mode, k = meta.dual_process(rel, confl, conf, force=force)
    print(f"surprise={surprise} conflict={conflict} force={force} → {mode} k={k}")

probe(0.3, 0.1)       # System-1
probe(1.5, 1.0)       # System-2
probe(1.5, 1.0, "system1")  # force fast
probe(0.2, 0.1, "system2")  # force deliberate
```

## 4. Optional: full model forward (GPU)

If you have a checkpoint:

```python
# load SENTINEL, run generate on "hi" vs hard puzzle, log decision.k / process_mode from meta step
```

Dual-process lives in `packages/sentinel/modules/metacognition.py` (`ControlDecision.process_mode`, `dual_process()`).
