from sentinel.buddy import hatch_buddy, list_animation_types
from sentinel.buddy.animations import ANIMATION_TYPES, build_animations


def test_all_anim_types_present():
    b = hatch_buddy("anim-user")
    for name in ANIMATION_TYPES:
        assert name in b.animations
        assert len(b.animations[name]) >= 1


def test_build_from_sprite():
    anims = build_animations(["  (o o)  ", " <(   )> ", "  ^^ ^^  "])
    assert "blink" in anims and len(anims["blink"]) >= 2


def test_list_types():
    assert len(list_animation_types()) == len(ANIMATION_TYPES)
