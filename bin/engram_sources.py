#!/usr/bin/env python3
"""Export other agents' sessions into the transcript shape harvest already reads.

engram's ENCODE stage (memory_harvest.py) reads Claude Code JSONL transcripts. opencode
and Hermes keep their sessions in SQLite, so this turns each session into one JSONL file
of Claude-shaped `user` / `assistant` text events under

    ~/.claude/projects/<slug>/.sources/<agent>-<session id>.jsonl

and harvest globs that directory too. Only the human's words and the assistant's prose are
exported: tool calls and results, reasoning, synthetic/ignored parts, sub-agent and cron
sessions never leave their database. The `.sources/` directory sits outside `*.jsonl` of
the project dir on purpose, so Claude Code's own session list is not polluted.

Idempotent and append-safe: a file is rewritten only when its content changed, and a
session only ever grows at the end, so harvest's byte-offset watermark stays valid.

    engram_sources.py              # export everything new
    engram_sources.py --dry-run    # report, write nothing
Env: ENGRAM_OPENCODE_DB, ENGRAM_HERMES_DB, CLAUDE_MEMORY_SLUG.
"""
import datetime
import json
import os
import sqlite3
import sys
from pathlib import Path

HOME = Path.home()
HERMES_SOURCES = ("cli", "local", "telegram", "oneshot")  # not subagent / cron


def slug() -> str:
    return os.environ.get("CLAUDE_MEMORY_SLUG") or str(HOME).replace("/", "-")


def sources_dir() -> Path:
    return HOME / ".claude" / "projects" / slug() / ".sources"


def _db(env: str, default: Path):
    p = Path(os.environ.get(env) or default)
    if not p.exists():
        return None
    return sqlite3.connect(f"file:{p}?mode=ro", uri=True)


def _event(role: str, text: str, ts: float) -> str:
    iso = datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).isoformat()
    return json.dumps({"type": role, "timestamp": iso,
                       "message": {"role": role, "content": [{"type": "text", "text": text}]}},
                      ensure_ascii=False)


def opencode_sessions(con):
    """Yield (session id, [(role, text, epoch)]) for top-level sessions of this project."""
    want = slug()
    for sid, directory in con.execute(
            "select id, directory from session where parent_id is null order by time_created"):
        if directory.replace("/", "-") != want:
            continue
        turns = []
        for mid, mdata in con.execute(
                "select id, data from message where session_id=? order by time_created, id", (sid,)):
            role = json.loads(mdata).get("role")
            if role not in ("user", "assistant"):
                continue
            for pdata, created in con.execute(
                    "select data, time_created from part where message_id=? order by time_created, id",
                    (mid,)):
                p = json.loads(pdata)
                text = (p.get("text") or "").strip()
                if p.get("type") == "text" and text and not p.get("synthetic") and not p.get("ignored"):
                    turns.append((role, text, created / 1000))
        yield sid, turns


def hermes_sessions(con):
    marks = ",".join("?" * len(HERMES_SOURCES))
    for (sid,) in con.execute(
            f"select id from sessions where source in ({marks}) order by started_at", HERMES_SOURCES):
        turns = [(r, c.strip(), t) for r, c, t in con.execute(
            "select role, content, timestamp from messages where session_id=? "
            "and role in ('user','assistant') and content is not null order by id", (sid,))
            if c.strip()]
        yield sid, turns


def write_session(out: Path, name: str, turns, dry_run: bool) -> str:
    """Returns 'new' | 'updated' | 'same' | 'empty'."""
    if not turns:
        return "empty"
    body = "\n".join(_event(r, t, ts) for r, t, ts in turns) + "\n"
    path = out / f"{name}.jsonl"
    old = path.read_text(encoding="utf-8") if path.exists() else None
    if old == body:
        return "same"
    if not dry_run:
        out.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(body, encoding="utf-8")
        os.utime(tmp, (turns[-1][2], turns[-1][2]))  # harvest's idle-grace sees the real last activity
        tmp.replace(path)
    return "new" if old is None else "updated"


def export(dry_run: bool = False) -> dict:
    out = sources_dir()
    result = {}
    for agent, env, default, reader in (
        ("opencode", "ENGRAM_OPENCODE_DB", HOME / ".local/share/opencode/opencode.db", opencode_sessions),
        ("hermes", "ENGRAM_HERMES_DB", HOME / ".hermes/state.db", hermes_sessions),
    ):
        con = _db(env, default)
        if con is None:
            continue
        counts = {"new": 0, "updated": 0, "same": 0, "empty": 0}
        try:
            for sid, turns in reader(con):
                counts[write_session(out, f"{agent}-{sid}", turns, dry_run)] += 1
        finally:
            con.close()
        result[agent] = counts
    return result


if __name__ == "__main__":
    print(json.dumps(export("--dry-run" in sys.argv[1:])))
