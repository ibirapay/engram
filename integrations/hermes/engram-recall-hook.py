#!/usr/bin/env python3
"""engram for Hermes Agent: automatic recall before every model call.

Registered as a Hermes shell hook on `pre_llm_call` (install.sh does it; Hermes asks for a one-time
consent, or `hermes --accept-hooks`). Hermes sends the turn as JSON on stdin; the user's message is in
`extra.user_message`. This hands it to engram's own recall hook (the same binary Claude Code runs on
UserPromptSubmit and the opencode plugin runs on every prompt) and returns its <relevant-memory>
block as {"context": ...}, which Hermes appends to the user message (never the system prompt).

Arguments after the script name go to engram-recall-hook as they are (install.sh passes
`--slug <store>`, so the store does not depend on the directory Hermes runs in).
Fails open: any error prints to stderr and injects nothing. engram-recall-hook itself injects a
memory at most once per session, so a tool loop that fires the hook again adds nothing new.
"""
import json
import os
import subprocess
import sys

BIN = os.environ.get("ENGRAM_BIN") or os.path.expanduser("~/.claude")
HOOK = os.path.join(BIN, "rust", "engram-recall-hook")


def user_text(message) -> str:
    if isinstance(message, str):
        return message
    if isinstance(message, list):           # multimodal: keep the text parts
        return " ".join(p.get("text", "") for p in message if isinstance(p, dict) and p.get("type") == "text")
    return ""


def main() -> None:
    try:
        payload = json.load(sys.stdin)
        text = user_text((payload.get("extra") or {}).get("user_message")).strip()
        if not text or "<relevant-memory>" in text:
            return
        request = {"prompt": text, "session_id": payload.get("session_id") or "",
                   "cwd": payload.get("cwd") or os.getcwd(), "hook_event_name": "UserPromptSubmit"}
        out = subprocess.run([HOOK, *sys.argv[1:]], input=json.dumps(request), capture_output=True,
                             text=True, timeout=15).stdout.strip()
        if out:
            print(json.dumps({"context": out}, ensure_ascii=False))
    except Exception as exc:                 # never block a turn on recall
        print(f"engram recall hook failed open: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
