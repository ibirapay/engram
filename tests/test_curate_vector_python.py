#!/usr/bin/env python3
"""auto-curate under a python WITHOUT qdrant-client (the memory-tidy skill runs system
python3) must re-exec once under the vector venv, and with no venv must exit non-zero,
never print "nothing to do" and exit 0 (that hid a dead curator for weeks)."""
import sys, types, importlib.util, tempfile, shutil, os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent


class Exec(Exception):
    pass


def main():
    for m in ("memory_ai", "engram_secrets", "memory_distill_verified"):
        sys.modules[m] = types.ModuleType(m)
    sys.modules["qdrant_client"] = None            # import qdrant_client -> ImportError
    d = tempfile.mkdtemp(); os.environ["HOME"] = d; os.environ["CLAUDE_MEMORY_SLUG"] = "-t"
    for k in ("ENGRAM_VECTOR_PYTHON", "ENGRAM_CURATE_REEXEC"):
        os.environ.pop(k, None)
    sp = importlib.util.spec_from_file_location("ac", ROOT / "bin" / "memory_auto_curate.py")
    ac = importlib.util.module_from_spec(sp); sp.loader.exec_module(ac)

    def fake_execv(path, argv):
        raise Exec((path, argv))
    real_execv, os.execv = os.execv, fake_execv
    try:
        # 1. no venv -> exit 2
        try:
            ac._ensure_vector_python(); raise AssertionError("returned without qdrant-client")
        except SystemExit as e:
            assert e.code == 2, f"want exit 2, got {e.code}"

        # 2. venv present -> one re-exec into it, same argv
        vpy = Path(d) / ".claude/vector/venv/bin/python"
        vpy.parent.mkdir(parents=True); vpy.write_text("#!/bin/sh\n"); vpy.chmod(0o755)
        sys.argv = ["memory_auto_curate.py", "--apply"]
        try:
            ac._ensure_vector_python(); raise AssertionError("did not re-exec")
        except Exec as e:
            path, argv = e.args[0]
            assert path == str(vpy) and argv[-1] == "--apply", e.args

        # 3. already re-exec'd once (env guard set by step 2) -> exit 2, no loop
        try:
            ac._ensure_vector_python(); raise AssertionError("looped")
        except SystemExit as e:
            assert e.code == 2
    finally:
        os.execv = real_execv
        shutil.rmtree(d)
    print("ok — auto-curate re-execs once into the vector venv, else exits 2")


if __name__ == "__main__":
    main()
