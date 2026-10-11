"""Rebuild docs/paper/AESC_v2_3_Technical_Report.pdf (ReportLab).

The PDF is a binary artefact, so text edits are made HERE and the file is regenerated:
    python scripts/docs/make_tech_report_pdf.py
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = Path(__file__).resolve().parents[2] / "docs" / "paper" / "AESC_v2_3_Technical_Report.pdf"
FOOT = "AESC / NeuroField v2.3 \u2014 Revised Technical Report"
AR = "-&gt;"  # Helvetica has no arrow glyph and the Symbol font rendered as black boxes

ss = getSampleStyleSheet()
body = ParagraphStyle("body", parent=ss["Normal"], fontName="Helvetica", fontSize=9, leading=12, spaceAfter=6)
title = ParagraphStyle("title", parent=body, fontName="Helvetica-Bold", fontSize=16, leading=20, alignment=TA_CENTER, spaceAfter=6)
sub = ParagraphStyle("sub", parent=body, fontName="Helvetica-Oblique", alignment=TA_CENTER, spaceAfter=8)
meta = ParagraphStyle("meta", parent=body, alignment=TA_CENTER, fontSize=8.5, leading=11, spaceAfter=2)
h = ParagraphStyle("h", parent=body, fontName="Helvetica-Bold", fontSize=11, leading=14, spaceBefore=10, spaceAfter=5)
h2 = ParagraphStyle("h2", parent=h, fontSize=9.5, leading=12, spaceBefore=6, spaceAfter=3)
bullet = ParagraphStyle("bullet", parent=body, spaceAfter=5)
cap = ParagraphStyle("cap", parent=body, fontName="Helvetica-Oblique", fontSize=8, leading=10, alignment=TA_CENTER, spaceBefore=3, spaceAfter=8)
cell = ParagraphStyle("cell", parent=body, fontSize=8, leading=10, spaceAfter=0)
cellh = ParagraphStyle("cellh", parent=cell, fontName="Helvetica-Bold")


def P(t, s=body):
    return Paragraph(t, s)


def B(t):
    return Paragraph("\u2022 " + t, bullet)


def table(rows, widths):
    data = [[Paragraph(c, cellh if i == 0 else cell) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * inch for w in widths], repeatRows=1)
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setStrokeColor(colors.HexColor("#999999"))
    canvas.line(0.75 * inch, 0.7 * inch, letter[0] - 0.75 * inch, 0.7 * inch)
    canvas.drawString(0.75 * inch, 0.55 * inch, FOOT)
    canvas.drawRightString(letter[0] - 0.75 * inch, 0.55 * inch, f"Page {doc.page}")
    canvas.restoreState()


s = []
s += [P("AESC v2.3: Two-Speed Memory Architecture", title),
      P("(Revised Technical Report \u2014 Memory Fixes, Brain-Inspired Metacognition, Contribution Results)", sub),
      P("Oct 9, 2026 \u00b7 Giridhari Karmakar \u00b7 giriisdev@gmail.com", meta),
      P("Code: https://github.com/giridhari-code/AERIS_64 (commit 70d54bd)", meta), Spacer(1, 10)]

s += [P("Abstract", h),
      P("AESC (Adaptive Event-State Computation), also released as NeuroField, is a recurrent language model with a k-step "
        "neural field, a predictive dendrite, top-k skill routing, and two-speed memory. Fast weights are written token by token "
        "by a surprise-gated delta rule; slow weights are trained by backpropagation and surprise-weighted replay."),
      P("The previous revision (v2 / Oct 2, 2026) fixed eleven failure modes from the first 10M implementation and reported that "
        "a 9.92M reference model solved 4-pair associative recall at 1.000 accuracy after 800 steps. This v2.3 report documents "
        "additional production defects that prevented the public codebase from reproducing that result, the fixes applied, "
        "measured module contributions, a brain-inspired metacognition controller, and a session-isolated tool sandbox. After the "
        "memory-path fixes, a d_model=64 production NeuroField reaches <b>0.973</b> eval accuracy on 4-pair recall at 800 steps with "
        "fast memory on, versus <b>0.344</b> with fast memory off (chance 0.125).")]

s += [P("1. Introduction and status of claims", h),
      P("This report makes no claim about how a small model compares with a much larger transformer. What changed in v2.3 is that "
        "the production repository now exhibits a clear, measurable contribution from fast memory on the same synthetic recall "
        "task used in Section 7 of v2."),
      P("<b>Expectation (unchanged):</b> on plain next-token loss a larger transformer is expected to win; the goal is to narrow the gap.")]

s += [P("2. Contributions of this revision", h),
      B("Diagnosis and fix of production memory-path defects that nullified fast-memory contribution in the public code (Section 4)."),
      B("Measured ablation of fast memory, slow memory, neuromodulator, and metacognition on 4-pair recall (Section 5)."),
      B("Brain-inspired metacognition: monitor (surprise, conflict, confidence, arousal) and control (k, encoding bias, consolidation request) (Section 6)."),
      B("Session-isolated tool sandbox aligned with safety controls (Section 7)."),
      B("Updated design checklist and open items (Sections 8-9).")]

s += [P("3. Architecture (unchanged data path)", h),
      P(f"Token order remains: embedding {AR} dendrite (local window + delayed error) {AR} metacognition {AR} k-step neural field {AR} "
        f"predictor {AR} top-k skill router {AR} neuromodulator gate {AR} fast write/read {AR} slow read {AR} logits. Only the fast matrix is "
        "written by a local delta rule; slow weights consolidate via surprise-weighted replay. Timing remains leakage-safe: "
        "nothing computed from x(t+1) reaches the logits that predict x(t+1)."),
      P("Fast write (Appendix A form): k_t = normalize(W_k d_{t-1}), v_t = W_v o_t, M_t = Lambda M_{t-1} + g_t k_t (v_t - "
        "M_{t-1}^T k_t)^T. Read: q_t = normalize(W_q d_t), r_t = M_t^T q_t.")]

s += [P("4. Additional failure modes fixed in v2.3", h),
      P("Rows 1-11 remain as in v2. The following defects were present in the production tree after v2 and prevented fast memory "
        "from contributing on recall until fixed."),
      KeepTogether([table([
          ["#", "Failure mode", "Evidence", "Fix"],
          ["12", "Decay too fast (lambda~0.73 on some dims)", "Associations wiped before query", "lambda init ~ 0.97-0.998 (linspace 3.5 to 6.5)"],
          ["13", "W_q random vs W_k", "Poor retrieval early", "W_q &lt;- W_k at init"],
          ["14", "Write on zero d_prev", "Garbage first write", "Skip write if ||d_prev||~0"],
          ["15", "Slow M_s xavier noise", "Noise masks fast path", "M_s = 0; P_s = 0 init"],
          ["16", "Gate starved (bias=0)", "Weak writes early", "bias=1.0, a_s=3.0"],
          ["17", "bar_S near zero", "rel_S explosion", "bar_floor=0.05; median k"],
          ["18", "No slow replay in ablation runs", "Slow path unused", "replay every 50 steps"],
      ], [0.35, 2.0, 1.9, 2.0]), P("Table 1. Production defects fixed in v2.3 (in addition to v2 rows 1-11).", cap)]),
      P("Isolated write-read unit check after the fix: cosine similarity between stored value projection and read vector is 1.000 "
        "under a single write with g=1.")]

s += [P("5. Contribution experiments", h),
      P("Task: 4 key-value pairs, 8 keys and 8 values, answer at the last position (chance 0.125). Model: production NeuroField, "
        "d_model=64, d_k=d_v=32, 4 skills, top-k=2, AdamW lr=1e-3, batch 8, CPU, single seed unless noted. Ablations use "
        "AblationConfig switches."),
      P("5.1 Before vs after memory-path fix (800 steps)", h2),
      KeepTogether([table([
          ["Condition", "Eval acc (before)", "Eval acc (after)"],
          ["full (all modules)", "0.383", "0.973"],
          ["no_fast_mem", "0.355", "0.344"],
          ["Delta (full - no_fast)", "+0.028 (null)", "+0.629 (clear)"],
      ], [2.6, 2.1, 2.1]), P("Table 2. Fast-memory contribution appears only after v2.3 fixes. Train best reaches 1.000 on full.", cap)]),
      P("5.2 Module ablations at 400 steps (after fix)", h2),
      KeepTogether([table([
          ["Ablation", "Eval acc", "Delta vs full"],
          ["full", "0.555", "-"],
          ["no_fast_mem", "0.426", "-0.129"],
          ["no_slow_mem", "0.543", "-0.012"],
          ["no_neuromod", "0.438", "-0.117"],
          ["no_metacog", "0.367", "-0.188"],
          ["skills_only", "0.395", "-0.160"],
      ], [2.6, 2.1, 2.1]), P("Table 3. Fast memory, neuromodulator, and metacognition help on pure recall; slow memory is near zero (expected). "
                              "Single seed \u2014 multi-seed GPU runs required before strong claims.", cap)]),
      P("These results support the mechanistic claim that the fast delta-rule path is necessary for high recall accuracy in this "
        "implementation. They do not include a comparison against transformer baselines.")]

s += [P("6. Brain-inspired metacognition", h),
      P("v2 metacognition was a minimal rule set: relative surprise baseline and a discrete k schedule. v2.3 expands it into an "
        "explicit monitor-control loop without claiming biological fidelity."),
      P("<b>Monitor.</b> Relative surprise S_rel = S / bar_S. Conflict from previous-step router entropy. Confidence = sigmoid(-(S_rel - "
        "1)). Arousal = 0.7*S_rel + 0.3*conflict."),
      P("<b>Control.</b> Field depth k increases with surprise and conflict. Encoding bias delta tags writes (high surprise / low "
        "confidence raises gate bias). Neuromodulator: g = sigmoid(a_s*(S_rel - 1) + a_c*|h| + b + delta). Consolidation request "
        "want_replay fires after a cumulative surprise budget and minimum token count; the trainer runs slow replay and "
        "acknowledge_replay."),
      P("The controller remains rule-based with calibration scalars. A learned metacognitive policy is future work.")]

s += [P("7. Safety and tool sandbox", h),
      P("v2.3 adds a production subprocess sandbox for tool use, aligned with session isolation:"),
      B("Per-session work directory and call budget; TTL eviction of idle sessions."),
      B("Binary allow-list; network disabled by default; CPU / address-space / wall-clock limits; audit log."),
      B("HTTP: POST /v1/tools/shell, /v1/tools/python; GET /v1/tools/audit; POST /v1/tools/reset."),
      B("Model session reset also drops the sandbox session."),
      P("Research-grade isolation (subprocess + rlimits), not a multi-tenant hardware jail. Red-team evaluation of memory "
        "poisoning remains future work.")]

s += [P("8. Updated design checklist (additions)", h),
      B("lambda on fast memory remains above ~0.95 on all key dims at init."),
      B("W_q and W_k aligned at init; write skipped when d_prev is the zero state."),
      B("Slow M_s and P_s do not inject random noise before the first replay."),
      B("Neuromodulator gate mean at init is materially above 0.5 on unit Gaussian state."),
      B("Metacognition exposes k, gate_bias, confidence, arousal; bar_S has a positive floor."),
      B("Contribution ablation reports full vs no_fast_mem on 4-pair recall before other claims."),
      B("Sandbox audit fields present when tools are enabled.")]

s += [P("9. Limitations and non-claims", h),
      B("No comparison against transformer or Mamba baselines is reported."),
      B("Reported ablations are single-seed CPU runs on one synthetic task."),
      B("Slow memory is not stressed by 4-pair recall; continual learning and long-copy are needed."),
      B("Metacognition is not a learned policy and is not a claim of biological realism."),
      B("Sandbox is not a full multi-tenant secure enclave."),
      B("Demo checkpoints in the repo remain tiny memorisers, not assistants."),
      B("Token-by-token recurrence is slower to train than a same-size transformer; scale GPU metrics pending.")]

s += [P("10. Related work (pointer)", h),
      P("Precedents unchanged from v2: fast weights (Schmidhuber; Ba et al.; DeltaNet), surprise-driven test-time memory "
        "(Titans), state-space models (Mamba), predictive coding (Rao and Ballard), complementary learning systems "
        "(McClelland et al.), looped computation (Universal Transformers), sparse routing (Shazeer et al.). The contribution "
        "remains the combination, audit signals, and measured production fixes.")]

s += [P("11. Conclusion", h),
      P("v2.3 closes a gap between the v2 reference claim and the production codebase: after correcting decay, key/query "
        "alignment, slow-path noise, gate starvation, and metacognitive baselines, fast memory shows a large, necessary "
        "contribution on associative recall (0.973 vs 0.344 at 800 steps in the reported configuration). Brain-inspired "
        "metacognition and a tool sandbox extend the control and safety surface. The next decisive experiments are multi-seed, "
        "compute-matched comparisons against transformers and Mamba on the full Section 7 task suite."),
      Spacer(1, 14),
      P("Code and configs: https://github.com/giridhari-code/AERIS_64 \u00b7 Contact: giriisdev@gmail.com", meta)]

doc = SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                        topMargin=0.75 * inch, bottomMargin=0.9 * inch,
                        title="AESC v2.3: Two-Speed Memory Architecture (Revised Technical Report)",
                        author="Giridhari Karmakar")
doc.build(s, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
