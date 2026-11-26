#!/usr/bin/env python3
"""opencode and Hermes sessions become harvestable transcripts: only the human's words and the
assistant's prose are exported, foreign noise is skipped, the export is idempotent and append-safe,
and harvest's own parser segments the result."""
import importlib.util, json, os, sqlite3, sys, tempfile, types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(name, rel):
    sys.modules.setdefault("memory_ai", types.ModuleType("memory_ai"))
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def make_opencode(path, directory):
    c = sqlite3.connect(path)
    c.executescript("""create table session(id text, directory text, parent_id text, time_created int);
      create table message(id text, session_id text, time_created int, data text);
      create table part(id text, message_id text, session_id text, time_created int, data text);""")
    c.execute("insert into session values('s1',?,NULL,1)", (directory,))
    c.execute("insert into session values('sub',?,'s1',2)", (directory,))       # sub-agent: skipped
    c.execute("insert into session values('other','/elsewhere',NULL,3)")        # other project: skipped
    for i, (role, parts) in enumerate([
        ("user", [{"type": "text", "text": "use sqlite for the graph"}]),
        ("assistant", [{"type": "reasoning", "text": "secret thoughts"},
                       {"type": "tool", "text": "ls"},
                       {"type": "text", "text": "ok, sqlite it is"},
                       {"type": "text", "text": "injected", "synthetic": True}]),
    ]):
        c.execute("insert into message values(?,?,?,?)", (f"m{i}", "s1", 1000 + i, json.dumps({"role": role})))
        for j, p in enumerate(parts):
            c.execute("insert into part values(?,?,?,?,?)", (f"p{i}{j}", f"m{i}", "s1", 1000 + i, json.dumps(p)))
    c.execute("insert into message values('mx','sub',5,?)", (json.dumps({"role": "user"}),))
    c.execute("insert into part values('px','mx','sub',5,?)", (json.dumps({"type": "text", "text": "sub-agent chatter"}),))
    c.commit(); c.close()


def make_hermes(path):
    c = sqlite3.connect(path)
    c.executescript("""create table sessions(id text, source text, started_at real);
      create table messages(id integer primary key autoincrement, session_id text, role text, content text, timestamp real);""")
    for sid, src in (("h1", "cli"), ("h2", "cron"), ("h3", "subagent")):
        c.execute("insert into sessions values(?,?,1)", (sid, src))
        for role, text in (("user", "remember I prefer pt-BR"), ("tool", "tool output"), ("assistant", "noted")):
            c.execute("insert into messages(session_id,role,content,timestamp) values(?,?,?,2)", (sid, role, text))
    c.commit(); c.close()


def main():
    with tempfile.TemporaryDirectory() as home:
        os.environ["CLAUDE_MEMORY_SLUG"] = "-proj"
        src = load("engram_sources", "bin/engram_sources.py")
        src.HOME = Path(home)
        oc, hm = Path(home) / "oc.db", Path(home) / "hm.db"
        make_opencode(oc, "/proj"); make_hermes(hm)
        os.environ["ENGRAM_OPENCODE_DB"], os.environ["ENGRAM_HERMES_DB"] = str(oc), str(hm)

        r = src.export()
        assert r["opencode"]["new"] == 1, r          # only the top-level session of this project
        assert r["hermes"]["new"] == 1, r            # cli only: cron and subagent skipped
        out = src.sources_dir()
        assert sorted(p.name for p in out.glob("*.jsonl")) == ["hermes-h1.jsonl", "opencode-s1.jsonl"]
        text = (out / "opencode-s1.jsonl").read_text()
        assert "secret thoughts" not in text and "injected" not in text and "sub-agent chatter" not in text
        assert "tool output" not in (out / "hermes-h1.jsonl").read_text()

        mtime = (out / "opencode-s1.jsonl").stat().st_mtime
        assert src.export() == {"opencode": {"new": 0, "updated": 0, "same": 1, "empty": 0},
                                "hermes": {"new": 0, "updated": 0, "same": 1, "empty": 0}}
        assert (out / "opencode-s1.jsonl").stat().st_mtime == mtime, "unchanged session must not be rewritten"

        # a session that grows keeps its old bytes as a prefix, so harvest's byte offset stays valid
        before = (out / "opencode-s1.jsonl").read_bytes()
        c = sqlite3.connect(oc)
        c.execute("insert into message values('m9','s1',2000,?)", (json.dumps({"role": "user"}),))
        c.execute("insert into part values('p9','m9','s1',2000,?)", (json.dumps({"type": "text", "text": "and a new turn"}),))
        c.commit(); c.close()
        assert src.export()["opencode"]["updated"] == 1
        assert (out / "opencode-s1.jsonl").read_bytes().startswith(before)

        mh = load("mh", "bin/memory_harvest.py")
        segs = mh.segment_events((out / "opencode-s1.jsonl").read_text().splitlines())
        assert [(s["kind"], s["text"]) for s in segs] == [
            ("user-direct", "use sqlite for the graph"), ("assistant", "ok, sqlite it is"),
            ("user-direct", "and a new turn")], segs
    print("ok")


if __name__ == "__main__":
    main()
