#!/usr/bin/env python3
"""engram-tui.py — the engram console. Browse, search, recall, inspect, edit.

Replaces the old FastAPI+React GUI. Same seven views, no browser, no server, no
dependencies: stdlib `curses`, the Rust `engram-recall` CLI for recall and the SQLite
graph file (`index/graph.db`) read with stdlib `sqlite3`. Runs on system python3 — a venv is only needed for the things
that genuinely need one (vector sync), which are subprocessed exactly as before.

Mutations are NOT reimplemented here: `s` and `d` shell out to save_memory.sh and
delete_memory.sh, so the TUI goes through the same gates the CLI does and cannot
bypass the safety model.

  engram-tui.py            # or: python3 ~/.claude/engram-tui.py
Keys: 1-7 view  j/k arrows move  enter open  / search  e edit  s save  d delete
      r refresh  q quit
"""
from __future__ import annotations
import curses
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import urllib.request
from pathlib import Path

ENGRAM_BIN = Path(os.environ.get("ENGRAM_BIN", Path.home() / ".claude"))
sys.path.insert(0, str(ENGRAM_BIN))
try:
    import memory_ai
except Exception as e:                                   # pragma: no cover
    sys.exit(f"engram engine not importable from {ENGRAM_BIN}: {e}")

TABS = ["Dashboard", "Memories", "Recall", "Vector", "Graph", "Skills", "Queues"]


def resolve_slug() -> str:
    """Which store: explicit env, then the operator pin in engram.env, then $HOME slugified."""
    slug = os.environ.get("CLAUDE_MEMORY_SLUG", "")
    if not slug:
        try:
            for line in (ENGRAM_BIN / "engram.env").read_text().splitlines():
                line = line.strip().removeprefix("export ").strip()
                if line.startswith("CLAUDE_MEMORY_SLUG="):
                    slug = line.split("=", 1)[1].strip().strip('"').strip("'")
        except OSError:
            pass
    return slug or str(Path.home()).replace("/", "-")


def store() -> Path:
    return Path.home() / ".claude" / "projects" / resolve_slug() / "memory"


def _recall(query: str, k: int) -> list[dict]:
    """Hybrid recall through the Rust CLI (keyword index + vector + the SQLite graph)."""
    exe = ENGRAM_BIN / "rust" / "engram-recall"
    r = subprocess.run([str(exe), query, "--k", str(k), "--slug", resolve_slug()],
                       capture_output=True, text=True, timeout=60)
    if r.returncode:
        raise RuntimeError((r.stderr or "engram-recall failed").strip()[-200:])
    return json.loads(r.stdout).get("results", [])


def _graph_rows(sql: str, params: tuple = ()) -> list[tuple]:
    import sqlite3
    db = Path(os.environ.get("ENGRAM_GRAPH_DB", ENGRAM_BIN / "index" / "graph.db"))
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5)
    try:
        return [tuple(r) for r in con.execute(sql, params)]
    finally:
        con.close()


def _get(url: str, headers: dict | None = None, timeout: float = 3.0) -> dict:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _frontmatter(text: str) -> dict:
    """name/description/type out of a memory's YAML front matter (type is nested
    under metadata:, hence the second branch)."""
    meta = {}
    if text.startswith("---"):
        end = text.find("\n---", 3)
        for ln in text[3:end if end != -1 else len(text)].splitlines():
            if ":" not in ln:
                continue
            key, val = ln.split(":", 1)
            key, val = key.strip(), val.strip().strip('"')
            if key in ("name", "description") and not ln.startswith(" "):
                meta[key] = val
            elif key == "type":
                meta["type"] = val
    return meta


def _sh(cmd: list, stdin: str | None = None, timeout: int = 1800):
    return subprocess.run(cmd, input=stdin, text=True, capture_output=True, timeout=timeout)


# ---- view data ------------------------------------------------------------
def rows_dashboard() -> list[dict]:
    cfg = memory_ai.load()
    out = [{"label": f"store        {store()}"},
           {"label": f"memories     {len(list(store().glob('*.md'))) - (1 if (store()/'MEMORY.md').exists() else 0)}"}]
    host = (cfg.get("ollama") or {}).get("host", "http://localhost:11434")
    backend = cfg.get("backend", "?")
    try:
        tags = _get(f"{host.rstrip('/')}/api/tags")
        out.append({"label": f"backend      {backend} — reachable, {len(tags.get('models', []))} models"})
    except Exception as e:
        out.append({"label": f"backend      {backend} — UNREACHABLE ({str(e)[:40]})"})
    vs = cfg.get("vector_store", {}) or {}
    if not memory_ai.vector_enabled(cfg):
        out.append({"label": "vector       disabled"})
    else:
        url = vs.get("url", "http://127.0.0.1:6333")
        coll = vs.get("collection", "engram_memory")
        try:
            n = _get(f"{url}/collections/{coll}")["result"]["points_count"]
            out.append({"label": f"vector       {n} points in {coll}"})
        except Exception as e:
            out.append({"label": f"vector       UNREACHABLE ({str(e)[:40]})"})
    g = graph_stats()
    out.append({"label": f"graph        {g}"})
    st = ENGRAM_BIN / "logs" / "daemon_state.json"
    try:
        out.append({"label": f"daemon       {json.dumps(json.loads(st.read_text()))[:100]}"})
    except Exception:
        out.append({"label": "daemon       no state file"})
    inject = (memory_ai.recall_cfg(cfg).get("inject") or {})
    out.append({"label": f"auto-recall  {'on' if inject.get('enabled', True) else 'off'} "
                         f"(k={inject.get('k', 4)}, {inject.get('timeout_ms', 2500)}ms)"})
    return out


def rows_memories(query: str = "") -> list[dict]:
    items = []
    for p in sorted(store().glob("*.md")):
        if p.name == "MEMORY.md":
            continue
        fm = _frontmatter(p.read_text(errors="ignore"))
        label = f"{fm.get('type', 'reference')[:4]:<4} {p.name:<52} {fm.get('description', '')}"
        if query and query.lower() not in label.lower():
            continue
        items.append({"label": label, "file": p.name})
    return items


def rows_recall(query: str = "") -> list[dict]:
    if not query:
        return [{"label": "press / and type a query — fuses graph + vector + keyword (RRF)"}]
    try:
        results = _recall(query, 10)
    except Exception as e:
        return [{"label": f"recall unavailable: {str(e)[:70]}"}]
    rows = [{"label": f"[{'+'.join(r['sources']):<22}] {r['name']}: {r['description']}",
             "file": r["file"], "detail": "\n".join(r.get("facts", []))} for r in results]
    return rows or [{"label": f"(nothing matched: {query})"}]


def rows_vector(query: str = "") -> list[dict]:
    cfg = memory_ai.load()
    if not memory_ai.vector_enabled(cfg):
        return [{"label": "vector store disabled (install.sh --vector)"}]
    if not query:
        vs = cfg.get("vector_store", {}) or {}
        url, coll = vs.get("url", "http://127.0.0.1:6333"), vs.get("collection", "engram_memory")
        try:
            info = _get(f"{url}/collections/{coll}")["result"]
            head = [{"label": f"collection {coll}: {info['points_count']} points, "
                              f"status {info.get('status')}"},
                    {"label": "press / to search  ·  S to re-sync the index from the .md store"}]
        except Exception as e:
            head = [{"label": f"Qdrant unreachable: {str(e)[:60]}"}]
        return head
    try:
        hits = [r for r in _recall(query, 15) if "vector" in r.get("sources", [])]
    except Exception as e:
        return [{"label": f"vector search unavailable: {str(e)[:70]}"}]
    return [{"label": f"{r['name']}: {(r.get('description') or '')[:90]}",
             "file": r["file"]} for r in hits] or [{"label": f"(no vector hits: {query})"}]


def graph_stats() -> str:
    try:
        mem, facts, triples = _graph_rows(
            "SELECT (SELECT COUNT(*) FROM memories), "
            "(SELECT COUNT(*) FROM facts WHERE valid_until IS NULL), "
            "(SELECT COUNT(*) FROM triples WHERE status = 'active' AND valid_until IS NULL)")[0]
        return f"{mem} memories, {facts} facts, {triples} triples (SQLite graph)"
    except Exception as e:
        return f"unavailable ({str(e)[:40]})"


def rows_graph(query: str = "") -> list[dict]:
    if not query:
        return [{"label": graph_stats()},
                {"label": "press / and type an entity name to see its 1-hop facts"}]
    try:
        e = query.lower()
        rows = _graph_rows(
            "SELECT relation, CASE WHEN subject_l = ?1 THEN object ELSE subject END, "
            "subject || ' ' || relation || ' ' || object FROM triples "
            "WHERE status = 'active' AND valid_until IS NULL AND (subject_l = ?1 OR object_l = ?1) "
            "LIMIT 50", (e,))
    except Exception as err:
        return [{"label": f"graph unavailable: {str(err)[:60]}"}]
    return [{"label": f"{rel} -> {other}", "detail": fact or ""}
            for rel, other, fact in rows] or [{"label": f"(no entity named {query})"}]


def rows_skills() -> list[dict]:
    base = Path.home() / ".claude" / "skills"
    rows = []
    for d in sorted(base.iterdir()) if base.exists() else []:
        if not d.is_dir() or d.name.startswith("."):
            continue
        desc = ""
        sk = d / "SKILL.md"
        if sk.exists():
            lines = sk.read_text(errors="ignore").splitlines()
            for i, ln in enumerate(lines):
                if not ln.startswith("description:"):
                    continue
                desc = ln.split(":", 1)[1].strip()
                if desc in (">", ">-", "|", "|-", ""):   # folded/literal block scalar
                    desc = " ".join(x.strip() for x in lines[i + 1:i + 4] if x.startswith(" "))
                desc = desc[:200]
                break
        rows.append({"label": f"{d.name:<40} {desc}"})
    pend = base / ".pending"
    rows += [{"label": f"PENDING  {p.name}"} for p in sorted(pend.glob("*"))] if pend.exists() else []
    return rows or [{"label": "(no skills installed)"}]


def rows_queues() -> list[dict]:
    rows = []
    for area in (".staging", ".quarantine"):
        d = store() / area
        files = sorted(d.glob("*.md")) if d.exists() else []
        rows.append({"label": f"── {area}  ({len(files)})"})
        rows += [{"label": f"   {p.name}", "path": str(p)} for p in files]
    return rows


# ---- the app --------------------------------------------------------------
class App:
    def __init__(self, scr):
        self.scr = scr
        self.tab = 0
        self.sel = 0
        self.top = 0
        self.query = ""
        self.status = "engram"
        self.rows: list[dict] = []
        self.load()

    def load(self):
        name = TABS[self.tab]
        try:
            if name == "Dashboard":
                self.rows = rows_dashboard()
            elif name == "Memories":
                self.rows = rows_memories(self.query)
            elif name == "Recall":
                self.rows = rows_recall(self.query)
            elif name == "Vector":
                self.rows = rows_vector(self.query)
            elif name == "Graph":
                self.rows = rows_graph(self.query)
            elif name == "Skills":
                self.rows = rows_skills()
            else:
                self.rows = rows_queues()
        except Exception as e:
            self.rows = [{"label": f"error: {e}"}]
        self.sel = min(self.sel, max(0, len(self.rows) - 1))

    # -- drawing
    def draw(self):
        scr = self.scr
        scr.erase()
        h, w = scr.getmaxyx()
        head = f" engram  ·  {store()}  ·  {len(list(store().glob('*.md')))} files "
        scr.addnstr(0, 0, head.ljust(w), w, curses.A_REVERSE)
        x = 0
        for i, t in enumerate(TABS):
            label = f" {i+1}:{t} "
            if x + len(label) < w:
                scr.addnstr(1, x, label, w - x,
                            curses.A_BOLD | curses.A_UNDERLINE if i == self.tab else curses.A_DIM)
            x += len(label)
        body_h = h - 4
        if self.sel < self.top:
            self.top = self.sel
        elif self.sel >= self.top + body_h:
            self.top = self.sel - body_h + 1
        for i in range(body_h):
            idx = self.top + i
            if idx >= len(self.rows):
                break
            line = self.rows[idx].get("label", "")
            attr = curses.A_REVERSE if idx == self.sel else curses.A_NORMAL
            scr.addnstr(2 + i, 0, ("> " if idx == self.sel else "  ") + line, w - 1, attr)
        q = f"  [/{self.query}]" if self.query else ""
        scr.addnstr(h - 2, 0, f" {self.status}{q}".ljust(w), w, curses.A_REVERSE)
        scr.addnstr(h - 1, 0, " 1-7 view · jk/↑↓ move · ⏎ open · / search · e edit · "
                              "s save · d delete · r refresh · q quit", w - 1, curses.A_DIM)
        scr.refresh()

    def pager(self, text: str, title: str = ""):
        """Read-only scrollable view — memory bodies, facts, command output."""
        h, w = self.scr.getmaxyx()
        lines = []
        for para in text.splitlines():
            lines += textwrap.wrap(para, max(20, w - 2)) or [""]
        top = 0
        while True:
            self.scr.erase()
            self.scr.addnstr(0, 0, f" {title} ".ljust(w), w, curses.A_REVERSE)
            for i in range(h - 2):
                if top + i >= len(lines):
                    break
                self.scr.addnstr(1 + i, 0, lines[top + i], w - 1)
            # w-1, never w: writing the bottom-right cell is an error in curses
            self.scr.addnstr(h - 1, 0, " jk/↑↓ scroll · q back".ljust(w - 1), w - 1, curses.A_REVERSE)
            self.scr.refresh()
            c = self.scr.getch()
            if c in (ord("q"), 27):
                return
            if c in (curses.KEY_DOWN, ord("j")) and top + h - 2 < len(lines):
                top += 1
            elif c in (curses.KEY_UP, ord("k")) and top:
                top -= 1
            elif c == curses.KEY_NPAGE:
                top = min(top + h - 3, max(0, len(lines) - 1))
            elif c == curses.KEY_PPAGE:
                top = max(0, top - (h - 3))

    def ask(self, prompt: str) -> str:
        h, w = self.scr.getmaxyx()
        curses.echo()
        curses.curs_set(1)
        self.scr.addnstr(h - 2, 0, f" {prompt} ".ljust(w), w, curses.A_REVERSE)
        self.scr.move(h - 2, len(prompt) + 2)
        try:
            s = self.scr.getstr().decode(errors="ignore").strip()
        except Exception:
            s = ""
        curses.noecho()
        curses.curs_set(0)
        return s

    def shell(self, fn):
        """Leave curses, run something interactive/slow, come back."""
        curses.endwin()
        try:
            return fn()
        finally:
            self.scr.refresh()
            curses.doupdate()

    # -- actions
    def current(self) -> dict:
        return self.rows[self.sel] if self.rows else {}

    def open_selected(self):
        row = self.current()
        path = row.get("path") or (str(store() / row["file"]) if row.get("file") else "")
        if path and Path(path).exists():
            self.pager(Path(path).read_text(errors="ignore"), Path(path).name)
        elif row.get("detail"):
            self.pager(row["detail"], row.get("label", "")[:60])

    def edit_selected(self):
        row = self.current()
        path = row.get("path") or (str(store() / row["file"]) if row.get("file") else "")
        if not path or not Path(path).exists():
            self.status = "nothing to edit here"
            return
        editor = os.environ.get("EDITOR", "vi")
        self.shell(lambda: subprocess.call([editor, path]))
        self.status = f"edited {Path(path).name}"
        self.load()

    def delete_selected(self):
        row = self.current()
        if not row.get("file"):
            self.status = "nothing to delete here"
            return
        if self.ask(f"delete {row['file']}? type yes:") != "yes":
            self.status = "delete cancelled"
            return
        script = ENGRAM_BIN / "delete_memory.sh"
        if not script.exists():
            self.status = "delete_memory.sh not installed"
            return
        r = _sh(["bash", str(script), row["file"]])
        self.status = f"deleted {row['file']}" if r.returncode == 0 else f"delete failed: {r.stderr[-60:]}"
        self.load()

    def save_new(self):
        fname = self.ask("new memory filename (foo.md):")
        if not fname:
            return
        if not fname.endswith(".md"):
            fname += ".md"
        desc = self.ask("one-line description:") or "(no description)"
        with tempfile.NamedTemporaryFile("w+", suffix=".md", delete=False) as tf:
            tf.write(f"# {fname}\n\n")
            tmp = tf.name
        self.shell(lambda: subprocess.call([os.environ.get("EDITOR", "vi"), tmp]))
        content = Path(tmp).read_text()
        Path(tmp).unlink(missing_ok=True)
        if not content.strip():
            self.status = "empty — not saved"
            return
        r = _sh(["bash", str(ENGRAM_BIN / "save_memory.sh"), fname, desc], stdin=content)
        self.status = f"saved {fname}" if r.returncode == 0 else f"save failed: {r.stderr[-60:]}"
        self.load()

    def vector_sync(self):
        script = ENGRAM_BIN / "vector" / "vector_sync.py"
        if not script.exists():
            self.status = "vector_sync.py not installed"
            return
        py = ENGRAM_BIN / "vector" / "venv" / "bin" / "python"
        cmd = [str(py) if py.exists() else sys.executable, str(script), "--insert"]
        self.status = "syncing…"
        self.draw()
        r = _sh(cmd)
        self.pager((r.stdout or "") + (r.stderr or ""), "vector_sync --insert")
        self.load()

    def run(self):
        while True:
            self.draw()
            c = self.scr.getch()
            if c in (ord("q"), 27):
                return
            elif ord("1") <= c <= ord("7"):
                self.tab, self.sel, self.top, self.query = c - ord("1"), 0, 0, ""
                self.load()
            elif c in (curses.KEY_DOWN, ord("j")):
                self.sel = min(self.sel + 1, max(0, len(self.rows) - 1))
            elif c in (curses.KEY_UP, ord("k")):
                self.sel = max(0, self.sel - 1)
            elif c == curses.KEY_NPAGE:
                self.sel = min(self.sel + 10, max(0, len(self.rows) - 1))
            elif c == curses.KEY_PPAGE:
                self.sel = max(0, self.sel - 10)
            elif c in (curses.KEY_ENTER, 10, 13):
                self.open_selected()
            elif c == ord("/"):
                self.query = self.ask("search:")
                self.sel = self.top = 0
                self.status = "searching…"
                self.draw()
                self.load()
                self.status = f"{len(self.rows)} rows"
            elif c == ord("e"):
                self.edit_selected()
            elif c == ord("d"):
                self.delete_selected()
            elif c == ord("s"):
                self.save_new()
            elif c == ord("S") and TABS[self.tab] == "Vector":
                self.vector_sync()
            elif c == ord("r"):
                self.status = "refreshing…"
                self.draw()
                self.load()
                self.status = "refreshed"


def main():
    if "--help" in sys.argv or "-h" in sys.argv:
        sys.exit(__doc__)
    def _run(scr):
        curses.curs_set(0)
        App(scr).run()
    curses.wrapper(_run)


if __name__ == "__main__":
    main()
