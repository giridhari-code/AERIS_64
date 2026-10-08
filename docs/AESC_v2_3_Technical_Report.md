# AESC v2.3: Two-Speed Memory Architecture (Revised Technical Report)

**Oct 9, 2026** · Giridhari Karmakar · giriisdev@gmail.com  
**Code:** https://github.com/giridhari-code/AERIS_64

## Abstract

AESC / NeuroField is a recurrent LM with a k-step neural field, predictive dendrite, top-k skill routing, and two-speed memory. Fast weights use a surprise-gated delta rule; slow weights use backprop + surprise-weighted replay.

v2 (Oct 2, 2026) fixed eleven failure modes and claimed 1.000 recall on a 9.92M reference model. **v2.3** documents production defects that blocked that result in the public tree, the fixes, measured module contributions, brain-inspired metacognition, and a tool sandbox.

After fixes, production NeuroField (`d_model=64`) reaches **0.973** eval accuracy on 4-pair recall at 800 steps vs **0.344** with fast memory off (chance 0.125). **H1 remains untested.**

## Key results

| Condition (800 steps) | Eval acc |
|----------------------|----------|
| full | **0.973** |
| no_fast_mem | 0.344 |
| Delta | **+0.629** |

| Ablation (400 steps) | Eval | Delta vs full |
|----------------------|------|---------------|
| full | 0.555 | — |
| no_fast_mem | 0.426 | -0.129 |
| no_slow_mem | 0.543 | -0.012 |
| no_neuromod | 0.438 | -0.117 |
| no_metacog | 0.367 | -0.188 |
| skills_only | 0.395 | -0.160 |

## Failure modes 12–18 (new)

12. Decay too fast → lambda ≈ 0.97–0.998  
13. W_q vs W_k mismatch → W_q ← W_k at init  
14. Zero d_prev write → skip  
15. Slow M_s noise → zero init  
16. Gate starved → bias=1, a_s=3  
17. bar_S collapse → floor + median k  
18. No replay in ablation → every 50 steps  

## Metacognition (brain-inspired)

- **Monitor:** relative surprise `S_rel = S / bar_S`, router conflict, confidence = sigmoid(-(S_rel - 1)), arousal = 0.7*S_rel + 0.3*conflict  
- **Control:** k (effort), gate_bias delta (encoding), want_replay (consolidation)  
- Neuromod: `g = sigmoid(a_s*(S_rel - 1) + a_c*|h| + b + delta)`

## Sandbox

`/v1/tools/shell|python|audit|reset` — session isolation, allow-list, no network by default.

## Non-claims

- H1 not shown  
- Single-seed CPU ablations  
- Not biological fidelity  
- Demo checkpoints are not assistants  

Full PDF: `docs/AESC_v2_3_Technical_Report.pdf`
