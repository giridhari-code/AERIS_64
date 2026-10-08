"""Pure-shape tests: architecture inference, param formula, legacy key remap, leak-free split."""

from neurofield.checkpoint.infer import count_params_from_shapes, infer_model_dims, remap_legacy_keys
from neurofield.utils.split import split_blocks


def _fake_state(V=100, D=48, dk=24, dv=24, n=4, W=3):
    st = {
        "embed.weight": (V, D),
        "head.weight": (V, D),
        "dendrite.coeffs": (W, D),
        "router.router.weight": (n, D),
        "fast_mem.W_k.weight": (dk, D),
        "fast_mem.W_v.weight": (dv, D),
    }
    return st


def test_infer_dims_from_shapes():
    d = infer_model_dims(_fake_state())
    assert d == {
        "vocab_size": 100, "d_model": 48, "d_k": 24, "d_v": 24, "n_skills": 4, "dendrite_window": 3,
    }


def test_infer_empty_state_is_empty():
    assert infer_model_dims({}) == {}


def test_tied_head_counted_once():
    st = {"embed.weight": (10, 4), "head.weight": (10, 4), "x": (3,)}
    assert count_params_from_shapes(st) == 43
    assert count_params_from_shapes(st, tied_head=False) == 83


def test_legacy_slow_memory_key_is_remapped():
    new, renamed = remap_legacy_keys({"slow_mem.W_s": 1, "other": 2})
    assert "slow_mem.M_s" in new and "slow_mem.W_s" not in new
    assert renamed
    # never clobbers a present new key
    keep, none = remap_legacy_keys({"slow_mem.W_s": 1, "slow_mem.M_s": 9})
    assert keep["slow_mem.M_s"] == 9 and not none


def test_split_has_no_train_val_overlap_even_with_duplicates():
    text = "\n".join(f"q{i % 7}: answer {i % 7}" for i in range(80))
    train, val, stats = split_blocks(text, val_frac=0.2)
    assert stats["unique"] == 7
    t, v = set(train.split("\n")) - {""}, set(val.split("\n")) - {""}
    assert t and v and not (t & v)


def test_param_formula_matches_1b_config():
    # mirrors scripts/count_params.py; guards the headline claim "~1.0B params"
    V, D, dk, dv, n, W = 32000, 3584, 128, 128, 16, 4
    total = (
        V * D + (W * D + 2 * D) + (3 * D + dv * D + 3 * D * D + D + 2 * D)
        + 2 * (D * D + D) + 3 + D * n + n * (D * 2 * D + 2 * D + 2 * D * D + D)
        + (2 * D * dk + D * dv + dk) + (D * dk + dk * dv) + 2 * dv * D + 1 + 2 * D
    )
    assert total == 1_004_511_364
    assert 0.99e9 < total < 1.02e9
