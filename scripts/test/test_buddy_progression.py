from pathlib import Path
import tempfile
import sentinel.buddy.progression as pr
from sentinel.buddy.progression import ProgressionStore, evolve_sprite
from sentinel.buddy.engine import handle_buddy_command, hatch_buddy, is_buddy_command

def main():
    with tempfile.TemporaryDirectory() as td:
        pr._STORE = ProgressionStore(Path(td) / "p.json")
        uid = "qa_user"
        assert is_buddy_command("/buddy care")
        for _ in range(45):
            pr._STORE.add_care(uid, 1)
        assert pr._STORE.get(uid).stage >= 1
        lines = evolve_sprite(["  (o o)  "], 1)
        assert "*" in lines[0]
        assert "progression" in hatch_buddy(uid).to_public_dict()
        print("ok", handle_buddy_command("/buddy care", uid)[:60])

if __name__ == "__main__":
    main()
