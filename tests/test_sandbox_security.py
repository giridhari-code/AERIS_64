"""Regression tests for the sandbox hardening (no shell, containment, cleanup)."""

from __future__ import annotations

import pytest

from neurofield.sandbox import SandboxConfig, SandboxExecutor, SandboxRegistry


@pytest.fixture()
def ex(tmp_path):
    return SandboxExecutor(SandboxConfig(timeout_sec=5, work_root=str(tmp_path)), session_id="sec-test")


def test_shell_metacharacters_are_literal(ex):
    r = ex.run_shell("echo hi; id")
    assert r.ok and r.stdout.strip() == "hi; id"
    assert "uid=" not in r.stdout
    r = ex.run_shell("echo a && echo b | cat")
    assert r.stdout.strip() == "a && echo b | cat"


@pytest.mark.parametrize("cmd", ["bash -c id", "sh -c id", "/bin/cat /etc/hostname", "./x", "curl http://x"])
def test_disallowed_binaries_and_paths(ex, cmd):
    r = ex.run_shell(cmd)
    assert not r.ok and "allow-list" in r.stderr


def test_unparseable_command_rejected(ex):
    r = ex.run_shell("echo 'unterminated")
    assert not r.ok and "unparseable" in r.stderr


def test_write_file_cannot_escape_or_hit_sibling_prefix(ex):
    sibling = ex.work_dir.parent / (ex.work_dir.name + "_evil")
    sibling.mkdir()
    with pytest.raises(ValueError):
        ex.write_file(f"../{sibling.name}/pwn.txt", "x")
    with pytest.raises(ValueError):
        ex.write_file("/tmp/abs.txt", "x")
    assert ex.write_file("ok/a.txt", "x").read_text() == "x"


def test_session_dirs_do_not_collide_on_long_ids():
    a = SandboxExecutor._safe_id("A" * 64 + "1")
    b = SandboxExecutor._safe_id("A" * 64 + "2")
    assert a != b


def test_file_size_limit(ex):
    r = ex.run_python("open('big','w').write('x' * (40 * 1024 * 1024))")
    assert not r.ok


def test_offline_run_has_no_network_when_isolation_available(ex):
    r = ex.run_python(
        "import socket\ns=socket.socket(); s.settimeout(2)\n"
        "try:\n    s.connect(('1.1.1.1', 53)); print('CONNECTED')\n"
        "except OSError: print('blocked')"
    )
    if not r.audit.get("net_isolated"):
        pytest.skip("host cannot create a network namespace")
    assert r.stdout.strip() == "blocked"


def test_require_isolation_fails_closed(tmp_path, monkeypatch):
    import neurofield.sandbox.executor as e

    monkeypatch.setattr(e, "_NET_ISOLATION_PREFIX", [])  # pretend unshare is unavailable
    ex = SandboxExecutor(SandboxConfig(work_root=str(tmp_path), require_isolation=True), session_id="iso-test")
    r = ex.run_shell("echo hi")
    assert not r.ok and "isolation" in r.stderr


def test_registry_eviction_deletes_workdir(tmp_path):
    reg = SandboxRegistry(SandboxConfig(work_root=str(tmp_path)), max_sessions=1)
    first = reg.get("session-one")
    first.run_shell("echo a")
    d = first.executor.work_dir
    assert d.exists()
    reg.get("session-two")  # evicts the oldest
    assert not d.exists()
    second = reg.get("session-two")
    reg.drop("session-two")
    assert not second.executor.work_dir.exists()
