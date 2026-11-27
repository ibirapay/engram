#!/usr/bin/env python3
"""Power-cut safety (2026-11-27: 15 graduated memories came back empty). Every store writer puts the
bytes on disk before it returns, through a temp file in the same dir, fsync, and a rename, so a
power cut leaves the old file or the new one, never an empty or torn one:
- save_memory.sh / save_memory_content_only.sh (memory_write_durable): sync is called on the temp
  file, the target is a NEW inode (renamed into place, not truncated in place), no temp left;
- memory_harvest.py (durable_write): staged candidates and state.json fsync before the rename."""
import os, subprocess, sys, tempfile, types, importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "bin"


def shell_writers():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t); cl = t / ".claude"; cl.mkdir()
        for f in ("memory_lib.sh", "save_memory.sh", "save_memory_content_only.sh", "memory_index_build.py",
                  "engram_secrets.py"):
            (cl / f).write_bytes((BIN / f).read_bytes()); (cl / f).chmod(0o755)
        shim = t / "shim"; shim.mkdir(); log = t / "sync.log"
        (shim / "sync").write_text(f'#!/bin/sh\necho "$@" >> {log}\nexec /bin/sync "$@"\n'); (shim / "sync").chmod(0o755)
        env = {k: v for k, v in os.environ.items() if k != "CLAUDE_MEMORY_REPO"}
        env.update(HOME=str(t), CLAUDE_MEMORY_SLUG="-durable", PATH=f"{shim}:{os.environ['PATH']}")
        mem = cl / "projects" / "-durable" / "memory"
        body = "---\nname: project_a\ndescription: a\nmetadata:\n  type: project\n---\n\n## Summary\nv{}\n"
        p = mem / "project_a.md"
        for v, script, args in ((0, "save_memory.sh", ["project_a.md", "a"]),            # new file
                                (1, "save_memory.sh", ["project_a.md", "a"]),            # rewrite
                                (2, "save_memory_content_only.sh", ["project_a.md"])):   # rewrite
            ino = p.stat().st_ino if p.exists() else None
            log.write_text("")
            r = subprocess.run(["bash", str(cl / script), *args], input=body.format(v), env=env,
                               capture_output=True, text=True)
            # content_only always pushes (no remote here): only the local write is under test
            assert r.returncode == 0 or "ERROR pushing" in r.stderr, (script, r.stderr)
            assert p.read_text() == body.format(v), (script, p.read_text())
            assert ".project_a.md.tmp." in log.read_text(), f"{script} v{v}: temp file not synced before the rename"
            assert ino is None or p.stat().st_ino != ino, f"{script} v{v}: truncated in place instead of renamed into place"
            assert not [f for f in os.listdir(mem) if ".tmp." in f], f"{script} v{v}: temp file left behind"

def harvest_writer():
    sys.modules.setdefault("memory_ai", types.ModuleType("memory_ai"))
    spec = importlib.util.spec_from_file_location("mh", BIN / "memory_harvest.py")
    mh = importlib.util.module_from_spec(spec); spec.loader.exec_module(mh)
    calls, real_fsync, real_replace = [], os.fsync, os.replace
    mh.os.fsync = lambda fd: (calls.append(("fsync", os.readlink(f"/proc/self/fd/{fd}"))), real_fsync(fd))[1]
    mh.os.replace = lambda a, b: (calls.append(("replace", str(a))), real_replace(a, b))[1]
    try:
        with tempfile.TemporaryDirectory() as d:
            mh.STATE_PATH = Path(d) / "state.json"
            mh.save_state({"files": {"t.jsonl": {"offset": 7}}})
            assert mh.load_state()["files"]["t.jsonl"]["offset"] == 7
            tmp = [c for c in calls if c[0] == "replace"][0][1]
            assert calls.index(("fsync", tmp)) < calls.index(("replace", tmp)), f"state.json: no fsync before the rename: {calls}"
            assert ("fsync", d) in calls, f"the dir was not synced after the rename: {calls}"
            calls.clear()
            mh.STAGING = Path(d) / "staging"; mh.STAGING.mkdir()
            p = mh.write_candidate({"name": "x", "type": "project", "description": "x", "body": "b", "cites": ["U1"],
                                    "confidence": 0.9}, "user-direct", "s", {"U1"})
            assert p.read_text().count("name:") >= 1 and ("fsync", str(p.parent / f".{p.name}.tmp.{os.getpid()}")) in calls, calls
            assert not [f for f in os.listdir(mh.STAGING) if ".tmp." in f]
    finally:
        mh.os.fsync, mh.os.replace = real_fsync, real_replace


if __name__ == "__main__":
    shell_writers()
    harvest_writer()
    print("ok — store writers are power-cut safe (temp + fsync + rename, dir synced)")
