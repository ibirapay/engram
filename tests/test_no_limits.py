#!/usr/bin/env python3
"""Every night reads everything new (2026-11-26): harvest drained 4 transcripts a run, one window
each, and 430 fell behind; the graph synced 25 memories a night; the daemon killed any task after
an hour. Now each cap defaults to 0 = none, a transcript is read to its end in one run with its
progress saved after every window, and the caps still apply when set."""
import importlib.util, json, os, sys, tempfile, types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _ul(txt):
    return json.dumps({"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": txt}]}}) + "\n"


def harvest_drains_every_transcript(max_files=None):
    home = Path(tempfile.mkdtemp()); os.environ["CLAUDE_MEMORY_SLUG"] = "-t"
    calls = []
    mai = types.ModuleType("memory_ai")
    mai.load = lambda: {"harvest": {"idle_minutes": 0, **({"max_files_per_run": max_files} if max_files else {})}}
    mai.expert_model = lambda role, cfg=None: "m"
    mai.local_enabled = lambda c: True
    mai.ollama_generate = lambda prompt, role=None, cfg=None: (calls.append(prompt), "[]")[1]
    sys.modules["memory_ai"] = mai
    sys.modules["engram_sources"] = types.SimpleNamespace(export=lambda dry_run=False: {})
    sp = importlib.util.spec_from_file_location("mh_nl", ROOT / "bin" / "memory_harvest.py")
    mh = importlib.util.module_from_spec(sp); sp.loader.exec_module(mh)
    proj = home / "proj"; proj.mkdir()
    mh.PROJ_DIR, mh.STAGING = proj, proj / "memory" / ".staging"
    mh.STATE_PATH = home / "state.json"
    for name in ("a.jsonl", "b.jsonl", "c.jsonl", "d.jsonl", "e.jsonl"):     # more than the old cap of 4
        (proj / name).write_text(_ul("A" * 200) + _ul("B" * 200) + _ul("C" * 200))
    saves = []
    real_save = mh.save_state
    mh.save_state = lambda st: (saves.append(1), real_save(st))[1]
    argv, sys.argv = sys.argv, ["mh", "--max-chars", "250"]
    try:
        mh.main()
    finally:
        sys.argv = argv
    state = json.loads(mh.STATE_PATH.read_text())["files"]
    return calls, saves, state, proj


def main():
    calls, saves, state, proj = harvest_drains_every_transcript()
    sizes = {p.name: p.stat().st_size for p in proj.glob("*.jsonl")}
    assert all(state.get(n, {}).get("offset") == s for n, s in sizes.items()), (state, sizes)
    assert len(calls) == 15, f"5 transcripts x 3 windows each, got {len(calls)}"
    assert len(saves) >= 15, f"progress must be saved after every window, saved {len(saves)}x"
    # a cap, when set, still applies (to transcripts, each still drained to its end)
    calls, _, state, proj = harvest_drains_every_transcript(max_files=1)
    assert len(calls) == 3 and sum(1 for n in state if state[n]["offset"] > 0) == 1, (len(calls), state)
    print("ok — harvest reads every transcript to its end in one run, saving after every window")

    # the daemon: no per-task timeout and every pending memory, unless configured
    def daemon(conf):
        mai = types.ModuleType("memory_ai"); mai.load = lambda: conf
        mai.local_enabled = lambda c: True; mai.vector_enabled = lambda c: True
        sys.modules["memory_ai"] = mai
        os.environ["ENGRAM_LOG_DIR"] = tempfile.mkdtemp()
        sp = importlib.util.spec_from_file_location("d_nl", ROOT / "daemon" / "engram-daemon.py")
        d = importlib.util.module_from_spec(sp); sp.loader.exec_module(d)
        return d
    d = daemon({})
    assert d._task_timeout() is None and d._graph_sync_limit() >= 1_000_000
    d = daemon({"daemon": {"task_timeout_seconds": 7200, "graph_sync_limit": 40}})
    assert d._task_timeout() == 7200 and d._graph_sync_limit() == 40
    print("ok — the daemon has no task timeout and syncs every pending memory unless configured")


if __name__ == "__main__":
    main()
