# Learned Global Workspace (J-space)

Source idea: Gurnee et al. (Anthropic, 2026), *Verbalizable Representations Form a Global Workspace in
Language Models* — a small set of internal patterns that the model can **report, control, reason with and
broadcast**, found with the **Jacobian lens (J-lens)**.

## Mapping paper → code

| Paper | `modules/workspace.py` |
|---|---|
| `lens(h) = softmax(W_U · norm(J h))` | `lens_state(h)` = `norm(h + B A h)`; `read_tokens(h, W_U)` |
| `J = I` is the logit lens | `lens_out` is zero-init ⇒ J = I at step 0 |
| J-space = sparse non-negative combo of lens vectors, k ≲ 25 | `code()`: ReLU scores over `atoms`, top-`ws_k_active` kept |
| broadcast hub read by many systems | `broadcast(code)` → `NeuralField.W_js`, `SkillRouter.ws_bias` |
| reportable / verbalizable | `read_tokens`, `atom_words` |
| delete J-space (ablation) | `AblationConfig(use_workspace=False)` |
| inject / swap a concept | `model._ws_steer = vec` (added before competition) |

Not copied: the paper's corpus-averaged Jacobian. Here `J` is *learned* (rank-r correction of I), because a
~100M model has no frozen Jacobian to average; `lambda_lens` is the optional training signal that makes the
workspace verbalizable.

## Enable

```yaml
model:
  workspace_mode: learned   # default: rule (legacy, unchanged)
  ws_atoms: 128
  ws_k_active: 8
  ws_rank: 16
  lambda_lens: 0.0          # experimental
```
or `sentinel-train --config configs/workspace.yaml`. New parameters ⇒ train from scratch or load
with `strict=False` (enabling is exactly neutral at init: consumers are zero-init).

Branch note (`fix/sentinel-debug`): decode now uses the same depth `k_max` as training (`adaptive_decode_depth: false`
by default) and `scale_tied_logits: true` is available for NEW runs. `configs/workspace.yaml` sets the latter; give your
baseline run the same line, otherwise the A/B compares two different output scalings.

## Properties (all covered by `tests/test_workspace.py`)

- exactly neutral at init vs. rule mode; rule mode bit-identical to the pre-change code
- deterministic + session-isolated (no module-level workspace state), also with `adaptive_decode_depth: true`
- full-sequence == token-by-token logits with the workspace on
- code is ≥ 0 and has ≤ `ws_k_active` non-zeros per position
- gradients reach atoms / lens / broadcast; `lambda_lens` trains J even while consumers are still zero

## Inspect

```bash
python scripts/research/workspace_probe.py --checkpoint runs/aeris_109m_ws/best --text "..."
```

## Cost (d_model=1024, 109M config)

+174k params (0.16%), ≈172k MACs per position per field loop (< 1% of the router/skill MLPs).

## What is NOT shown yet

Quality gains. Tiny CPU smoke runs (150 steps, byte-level demo corpus) only show that learned mode trains
as stably as rule mode (val CE 2.77 vs 2.73, within noise) and that `lambda_lens=0.05` slows early
convergence (3.12). Decide on your own data with an A/B: `default.yaml` vs `workspace.yaml`
(vs `lambda_lens: 0.02`), same seed, same data, compare val CE and your reasoning evals; use
`use_workspace=False` at eval to measure how much the trained model relies on it.
