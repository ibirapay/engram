#!/usr/bin/env python3
"""approval gate: ids and approvals are bound to the store that queued them, --show
reads through the gate and secret-scans every file it would let a caller summarize,
and only a pending proposal can be approved."""
import importlib.util, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GATE = ROOT / "bin" / "engram_telegram_gate.py"


def _gate(slug):
    os.environ["CLAUDE_MEMORY_SLUG"] = slug
    sp = importlib.util.spec_from_file_location(f"gate_{slug}", GATE)
    g = importlib.util.module_from_spec(sp); sp.loader.exec_module(g)
    d = Path(tempfile.mkdtemp()); g.MEM = d; g.Q = d / ".approvals"
    return g, d


def test_ids_differ_per_store():
    a, da = _gate("-tenant-a"); b, db = _gate("-tenant-b")
    params = {"name": "same_file.md"}
    assert a._id("orphan_prune", params) != b._id("orphan_prune", params), "same id across stores"
    shutil.rmtree(da); shutil.rmtree(db)
    print("ok — the same op on the same file name gets a different id per store")


def test_apply_refuses_a_foreign_store():
    g, d = _gate("-tenant-a")
    ok, why = g._apply_one({"op": "orphan_prune", "store": "-tenant-b", "params": {"name": "x.md"},
                            "source_hashes": {}})
    assert not ok and "belongs to store -tenant-b" in why, why
    assert g._store_ok({"op": "orphan_prune"}), "legacy proposal without a store must stay valid"
    shutil.rmtree(d)
    print("ok — apply refuses a proposal queued by another store")


def _cli(d, slug, *args):
    env = {**os.environ, "HOME": str(d), "CLAUDE_MEMORY_SLUG": slug, "TELEGRAM_BOT_TOKEN": ""}
    mem = Path(d, ".claude", "projects", slug, "memory")
    return subprocess.run([sys.executable, str(GATE), *args], env=env, capture_output=True, text=True), mem


def test_cli_approve_only_pending_same_store_and_show_scans():
    home = Path(tempfile.mkdtemp())
    _, mem = _cli(home, "-tenant-a", "--list")
    (mem / ".approvals" / "pending").mkdir(parents=True, exist_ok=True)
    (mem / "keep.md").write_text("---\nname: keep\ndescription: d\n---\n\nplain fact\n")
    (mem / "leak.md").write_text("---\nname: leak\ndescription: d\n---\n\n"
                                 "token: ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8\n")
    prop = {"id": "merge_apply-t1", "op": "merge_apply", "store": "-tenant-a",
            "params": {"umbrella": "keep.md", "absorbed": ["leak.md"],
                       "umbrella_content": "---\nname: keep\ndescription: d\n---\n\nplain fact\n"},
            "preview": "p", "source_hashes": {}, "created_at": 0, "expires_at": 2**40}
    (mem / ".approvals" / "pending" / "merge_apply-t1.json").write_text(json.dumps(prop))
    foreign = {**prop, "id": "orphan_prune-t2", "op": "orphan_prune", "store": "-tenant-b",
               "params": {"name": "keep.md"}}
    (mem / ".approvals" / "pending" / "orphan_prune-t2.json").write_text(json.dumps(foreign))

    r, _ = _cli(home, "-tenant-a", "--show", "merge_apply-t1")
    assert r.returncode == 0, r.stdout + r.stderr
    shown = json.loads(r.stdout)
    assert shown["describe"] == "absorb leak.md into keep.md", shown["describe"]
    assert shown["secret_scan"]["keep.md"] is None, shown["secret_scan"]
    assert shown["secret_scan"]["umbrella_content"] is None, shown["secret_scan"]
    assert shown["secret_scan"]["leak.md"], f"secret not flagged: {shown['secret_scan']}"

    # Fail closed: a file that is not there was not scanned, and the save-memory
    # override must not make the scan pass silently.
    gone = {**prop, "id": "merge_apply-t3", "params": {**prop["params"], "absorbed": ["gone.md"]}}
    (mem / ".approvals" / "pending" / "merge_apply-t3.json").write_text(json.dumps(gone))
    r, _ = _cli(home, "-tenant-a", "--show", "merge_apply-t3")
    assert json.loads(r.stdout)["secret_scan"]["gone.md"], "missing file reported clean"
    os.environ["MEMORY_ALLOW_SECRET"] = "1"
    try:
        r, _ = _cli(home, "-tenant-a", "--show", "merge_apply-t1")
    finally:
        del os.environ["MEMORY_ALLOW_SECRET"]
    assert json.loads(r.stdout)["secret_scan"]["leak.md"], "MEMORY_ALLOW_SECRET silenced the scan"

    r, _ = _cli(home, "-tenant-a", "--approve", "orphan_prune-t2")
    assert r.returncode == 1 and "refused: belongs to store -tenant-b" in r.stdout, r.stdout
    assert (mem / ".approvals" / "pending" / "orphan_prune-t2.json").exists(), "foreign proposal moved"

    expired = mem / ".approvals" / "expired"; expired.mkdir(parents=True, exist_ok=True)
    (expired / "merge_apply-old.json").write_text(json.dumps({**prop, "id": "merge_apply-old"}))
    r, _ = _cli(home, "-tenant-a", "--approve", "merge_apply-old")
    assert r.returncode == 1 and "already expired" in r.stdout, r.stdout

    r, _ = _cli(home, "-tenant-a", "--approve", "merge_apply-t1")
    assert r.returncode == 0 and r.stdout.strip() == "approved", r.stdout + r.stderr
    shutil.rmtree(home)
    print("ok — --show scans files + proposed text; --approve: pending + same store only")


if __name__ == "__main__":
    test_ids_differ_per_store()
    test_apply_refuses_a_foreign_store()
    test_cli_approve_only_pending_same_store_and_show_scans()
