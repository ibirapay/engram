#!/usr/bin/env python3
"""Hermes gets automatic recall on every turn: its pre_llm_call payload goes to engram-recall-hook
(with the slug the installer passed) and the <relevant-memory> block comes back as {"context": ...}.
Nothing comes back for an empty message or one that already carries the block, and any failure
fails open (no output, exit 0)."""
import json, os, stat, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ADAPTER = ROOT / "integrations" / "hermes" / "engram-recall-hook.py"


def run(payload, hook_body, *args):
    d = Path(tempfile.mkdtemp())
    (d / "rust").mkdir()
    hook = d / "rust" / "engram-recall-hook"
    hook.write_text(hook_body); hook.chmod(hook.stat().st_mode | stat.S_IEXEC)
    env = dict(os.environ, ENGRAM_BIN=str(d))
    r = subprocess.run([sys.executable, str(ADAPTER), *args], input=json.dumps(payload),
                       capture_output=True, text=True, env=env)
    return r


ECHO = ("#!/usr/bin/env python3\nimport json,sys\nreq=json.load(sys.stdin)\n"
        "print('<relevant-memory>\\n- seen: '+req['prompt']+' | args: '+' '.join(sys.argv[1:])+' | sid: '+req['session_id']+'\\n</relevant-memory>')\n")


def main():
    turn = {"hook_event_name": "pre_llm_call", "session_id": "s1", "cwd": "/tmp",
            "extra": {"user_message": "how do I rotate the relay key?"}}
    r = run(turn, ECHO, "--slug", "-home-x")
    out = json.loads(r.stdout)["context"]
    assert r.returncode == 0 and "<relevant-memory>" in out, r
    assert "seen: how do I rotate the relay key?" in out and "args: --slug -home-x" in out and "sid: s1" in out, out
    # a multimodal message: the text parts
    turn["extra"]["user_message"] = [{"type": "text", "text": "rotate"}, {"type": "image_url", "image_url": "x"}]
    assert "seen: rotate" in json.loads(run(turn, ECHO).stdout)["context"]
    # nothing to add: empty, or the block is already there
    for msg in ("", "  ", "<relevant-memory>\n- x\n</relevant-memory>"):
        turn["extra"]["user_message"] = msg
        assert run(turn, ECHO).stdout == "", msg
    # the hook prints nothing (nothing relevant): no context
    turn["extra"]["user_message"] = "hi"
    assert run(turn, "#!/bin/sh\ncat >/dev/null\n").stdout == ""
    # fail open: a crashing hook, a missing hook, a malformed payload
    r = run(turn, "#!/bin/sh\nexit 3\n")
    assert r.returncode == 0 and r.stdout == "", r
    r = subprocess.run([sys.executable, str(ADAPTER)], input="not json", capture_output=True, text=True,
                       env=dict(os.environ, ENGRAM_BIN="/nonexistent"))
    assert r.returncode == 0 and r.stdout == "" and "failed open" in r.stderr, r
    print("ok — Hermes recall hook: payload in, <relevant-memory> context out, fails open")


if __name__ == "__main__":
    main()
