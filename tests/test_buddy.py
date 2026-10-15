from sentinel.buddy import hatch_buddy, handle_buddy_command, is_buddy_command
from sentinel.buddy.species import SPECIES


def test_eighteen_species():
    assert len(SPECIES) == 18


def test_deterministic():
    a = hatch_buddy("alice")
    b = hatch_buddy("alice")
    assert a.species_id == b.species_id
    assert a.name == b.name
    assert a.rarity == b.rarity
    assert a.shiny == b.shiny


def test_different_users_may_differ():
    # not guaranteed different but vastly likely; at least hatch works
    a = hatch_buddy("alice")
    b = hatch_buddy("bob-other-id-xyz")
    assert a.user_id != b.user_id


def test_commands():
    assert is_buddy_command("/buddy")
    assert is_buddy_command("/buddy pet")
    assert is_buddy_command("--buddy on")
    assert is_buddy_command("--Buddy off")
    assert not is_buddy_command("hello")
    text = handle_buddy_command("/buddy", "u1")
    b = hatch_buddy("u1")
    assert b.name in text and b.species_name in text
    assert handle_buddy_command("/buddy on", "u1") == "BUDDY_ON"
    assert handle_buddy_command("/buddy off", "u1") == "BUDDY_OFF"
    assert handle_buddy_command("--buddy on", "u1") == "BUDDY_ON"
    assert handle_buddy_command("--Buddy off", "u1") == "BUDDY_OFF"
    pet = handle_buddy_command("/buddy pet", "u1")
    assert "♥" in pet or "heart" in pet.lower() or "*" in pet
