#!/usr/bin/env python3
"""memory_auto_curate.py — DETERMINISTIC, lossless, recoverable auto-consolidation.

The interactive /memory-curate uses LLM JUDGMENT + a human gate to decide what to
merge. This is the unattended sibling: it merges only what is DETERMINISTICALLY a
near-duplicate (Qdrant ANN cosine >= a high threshold, SAME type), so no judgment
call is delegated to a model. Two different operations come out of that:
  * SUPERSEDE  — the members are canonically IDENTICAL copies (same body after
                 whitespace/boilerplate normalization, same description and type).
                 Genuinely lossless: the keeper is written back byte-for-byte and no
                 model is invoked. Auto-applies. Containment alone is NOT enough —
                 see _supersede_keeper.
  * COMPRESS   — members say the same thing differently, so an umbrella is generated.
                 This IS lossy; `sentence_coverage` is a rejection heuristic, NOT proof
                 of losslessness (it is lexical prefix matching, so two rewordings of
                 one fact never "cover" each other). Safety here comes from the Codex/
                 human gate plus REVERSIBILITY, not from the score.
  * RECOVERABLE — absorbed sources go to .quarantine/ (undo) and are de-indexed via
                 delete_memory.sh, which snapshots to .trash/ (90-day undo) first.
  * BOUNDED    — at most `max_merges_per_run` clusters per run.
Everything a model can't make safe (pruning distinct memories, deleting suspects)
stays in the human-gated /memory-curate. This never deletes a memory that isn't
absorbed into an umbrella first.

Gated by auto_curate.enabled in engram.yaml AND --apply. Dry-run otherwise.

Usage:
  memory_auto_curate.py                 # dry-run: print the merges it would do
  memory_auto_curate.py --apply         # apply (needs auto_curate.enabled: true)
"""
from __future__ import annotations
import json, os, re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path.home() / ".claude"))
sys.path.insert(0, str(Path.home() / ".claude" / "vector"))
import memory_ai
import engram_secrets
import memory_distill_verified as mdv

HOME = Path.home()
SLUG = os.environ.get("CLAUDE_MEMORY_SLUG") or str(HOME).replace("/", "-")
MEM = HOME / ".claude" / "projects" / SLUG / "memory"
SAVE = HOME / ".claude" / "save_memory.sh"
DELETE = HOME / ".claude" / "delete_memory.sh"

AC_DEFAULTS = {
    "enabled": False,
    "merge_threshold": 0.92,      # cosine >= this to auto-merge (stricter than the 0.86 finder)
    "max_merges_per_run": 2,
    # Who reviews a merge before it can auto-apply:
    #   auto   (default) — use Codex if it's installed; otherwise fall back to HUMAN
    #                      (every merge -> Telegram approval queue). Best for portability:
    #                      most users have Claude but NOT Codex.
    #   codex  — force Codex review (near-lossless+clean auto-applies, else human queue).
    #   human  — NO Codex; EVERY merge goes to the human Telegram queue (nothing auto-
    #            applies). Safe default when you have only one agent.
    # There is intentionally no "no reviewer, just auto-apply" mode — a compressed merge
    # always gets either Codex or a human before it touches the store.
    "review_gate": "auto",
    # Autonomous mode: NO human queue anywhere. A COMPRESS merge auto-applies when
    # Codex says APPROVE (lossy coverage allowed — safety is REVERSIBILITY: sources
    # stay in .quarantine/ for 30 days); any other verdict is SKIPPED, not queued.
    # Orphan prunes apply straight to .trash/ (90-day undo). FAIL-CLOSED: with no
    # Codex reviewer available an autonomous run mutates NOTHING (it does not fall
    # back to a human queue).
    "autonomous": False,
    # When the primary reviewer FAILS MECHANICALLY (no CLI, rate-limited, unparseable
    # output) the verdict goes to a Claude-headless fallback once — the owner's
    # standing review rule (Codex first, Claude when Codex is down). A real REJECT
    # never falls back. Safety does not depend on the reviewer: quarantine + reconcile.
    "fallback_reviewer": True,
    # Orphan pruning: PROPOSE (never auto) stale memories with no merge target for a
    # one-tap Telegram prune. frequency==0 means "never discussed", NOT "worthless" (a
    # disaster-recovery runbook is freq-0 for a year), so this is human-gated by design
    # — unless autonomous: true, which prunes directly to .trash.
    "prune_orphans": True,
    "orphan_age_days": 90,
    "max_prunes_per_run": 5,
    # When Codex DOES review: auto-apply only a near-LOSSLESS merge (a fact outside
    # Codex's slice could be dropped otherwise); everything lossy -> the human queue.
    "min_auto_coverage": 0.90,
}


def _resolve_gate(ac) -> str:
    """codex | human. 'auto' -> codex iff the advisor script AND the codex CLI are both
    present; else human (Telegram approval). So a user without Codex still gets a safe
    reviewer (themselves, one tap) instead of unreviewed auto-merges."""
    import shutil
    g = (ac.get("review_gate") or "auto").lower()
    if g in ("codex", "human"):
        return g
    return "codex" if (ADVISOR.exists() and shutil.which("codex")) else "human"

ADVISOR = HOME / ".claude" / "skills" / "code-advisor" / "scripts" / "code_advisor.py"


def ac_cfg(cfg):
    out = dict(AC_DEFAULTS); out.update((cfg.get("auto_curate") or {})); return out


_FM_RE = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n", re.S)


def _fm(p: Path):
    """(name, description, type) from an ANCHORED frontmatter block.

    Previously this regex-searched the WHOLE file, so body prose containing a line
    like `type: project` could spoof a memory's type — and two malformed files both
    yielding "" counted as the same type, clearing the same-type merge guard.
    """
    t = p.read_text(errors="ignore")
    m = _FM_RE.match(t)
    if not m:
        return p.stem, "", ""
    try:
        import yaml
        fm = yaml.safe_load(m.group(1))
    except Exception:
        fm = None
    if not isinstance(fm, dict):
        return p.stem, "", ""
    meta = fm.get("metadata") if isinstance(fm.get("metadata"), dict) else {}
    return (str(fm.get("name") or p.stem).strip(),
            str(fm.get("description") or "").strip(),
            str(fm.get("type") or meta.get("type") or "").strip())


def _fm_raw(p: Path) -> dict:
    """The whole parsed frontmatter mapping ({} when absent/unparseable)."""
    m = _FM_RE.match(p.read_text(errors="ignore"))
    if not m:
        return {}
    try:
        import yaml
        fm = yaml.safe_load(m.group(1))
    except Exception:
        return {}
    return fm if isinstance(fm, dict) else {}


def _clusters(pairs, threshold):
    """Above-threshold pairs -> CLIQUES (every member similar to every other member).

    This used to be union-find, whose transitive closure merges A-B-C whenever A≈B and
    B≈C — even when A and C are materially different. For an operation that deletes
    files that is not safe, so a component is only kept whole when it is complete;
    otherwise it degrades to its individual pairs.
    """
    edges = {(a, b) if a < b else (b, a) for score, a, b in pairs if score >= threshold}
    adj = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b); adj.setdefault(b, set()).add(a)
    seen, used, out = set(), set(), []
    for node in sorted(adj):                       # connected components
        if node in seen:
            continue
        comp, stack = set(), [node]
        while stack:
            x = stack.pop()
            if x in comp:
                continue
            comp.add(x); stack.extend(adj[x] - comp)
        seen |= comp
        n = len(comp)
        if n >= 2 and not (comp & used) and all(
                (a, b) in edges for i, a in enumerate(sorted(comp))
                for b in sorted(comp)[i + 1:]):
            out.append(comp); used |= comp         # complete -> safe as one cluster
        else:
            # Degrade to pairs, but keep the output a DISJOINT matching: a file may
            # take part in at most one merge per run. Overlapping pairs (A-B and B-C)
            # would merge B into A and then into C, and undoing them out of order
            # restores an intermediate B over the original and drops the last backup.
            for a, b in sorted(edges):
                if a in comp and b in comp and a not in used and b not in used:
                    out.append({a, b}); used |= {a, b}
    return out


def _body_of(distilled: str) -> str:
    """Strip an outer code fence, then a leading frontmatter block; keep the body."""
    import engram_telegram_gate as _gate
    distilled = _gate.unwrap_code_fence(distilled)
    m = _FM_RE.match(distilled)
    return distilled[m.end():].lstrip("\n") if m else distilled


def _src_sha(files) -> dict:
    """sha256 of every participating file, captured at ANALYSIS time.

    _apply_merge re-checks these before it mutates anything. Without it the whole
    analysis is time-of-check/time-of-use: a concurrent save (harvest, a hook, the
    user) between the containment test and the write could let a newly added fact be
    quarantined, or overwrite a just-updated keeper with stale content — while the run
    still reports "lossless". Queued proposals make the window hours long, not ms.
    """
    import hashlib
    return {f: (hashlib.sha256((MEM / f).read_bytes()).hexdigest()
                if (MEM / f).is_file() else "") for f in files}


# ── supersede: the lossless path ────────────────────────────────────────────
# Per-file harvest boilerplate: the ONLY lines allowed to differ between two copies,
# because the harvester stamps them per file. Matched as WHOLE LINES, anchored, with
# no DOTALL: an earlier version used `_Provenance:.*?_` under re.S, whose non-greedy
# run to the next underscore could swallow real body text across many lines — so two
# memories with genuinely different content could canonicalize to the same string and
# one would be quarantined unread. A line-anchored pattern cannot span body content.
_BOILER_LINE_RE = re.compile(
    r"(?:_Provenance:[^\n_]*_|<!--\s*staged by memory_harvest\b[^\n]*-->)\Z")


def _canonical_body(text: str) -> str:
    """Body for the identity test: frontmatter and per-file harvest boilerplate lines
    removed, line endings and trailing whitespace normalized, blank lines dropped.
    Everything semantic — case, punctuation, operators, numbers — survives verbatim."""
    m = _FM_RE.match(text)
    body = text[m.end():] if m else text
    out = []
    for ln in body.replace("\r\n", "\n").split("\n"):
        ln = ln.rstrip()
        if ln.strip() and not _BOILER_LINE_RE.match(ln.strip()):
            out.append(ln)
    return "\n".join(out)


def _supersede_keeper(files):
    """The keeper when the members are CANONICALLY IDENTICAL duplicates, else None.

    Ungated supersede deletes a file with no model and no human in the loop, so the
    only thing it may act on is a true copy: identical canonical body AND identical
    description and type.

    Containment is deliberately NOT enough, even at 100%. Containment proves the text
    is present in the keeper, not that it still MEANS the same thing there — a larger
    stale document that quotes a current rule in order to call it obsolete contains
    every shingle of the smaller authoritative one, would be picked as keeper (it is
    longer), and would silently quarantine the authority. No lexical test can tell
    those apart, so anything short of identity goes to the review path, where a model
    or a human reads it. Cosine cannot help either: it is symmetric.

    Consequence, stated plainly: this fires only on genuine double-harvests and copies
    left by a store merge. Reworded duplicates, supersets, and near-misses all still
    cost a review — that is the intended trade.
    """
    canon = {f: _canonical_body((MEM / f).read_text(errors="ignore")) for f in files}
    if any(not c.strip() for c in canon.values()):
        return None
    if len({c for c in canon.values()}) != 1:          # not a true copy -> review
        return None
    # The ENTIRE frontmatter must match, not just description+type. _canonical_body
    # strips frontmatter, so without this two memories with identical generic bodies
    # but different scope — "Ethereum treasury rule" vs "Bitcoin treasury rule", which
    # differ only in `name` — would look like copies and one would be quarantined
    # unread. Real duplicate-harvest copies match here exactly: when a store merge
    # suffixes a colliding FILENAME the frontmatter (name, description, metadata,
    # originSessionId) is carried over untouched, so nothing legitimate is lost.
    fms = [json.dumps(_fm_raw(MEM / f), sort_keys=True, default=str) for f in files]
    if len(set(fms)) != 1:
        return None
    # Bodies are identical, so any member is a valid keeper; pick deterministically.
    return max(sorted(files), key=lambda f: len((MEM / f).read_text(errors="ignore")))


def _score(p: Path, scores) -> float:
    return float(scores.get(p.name, 0.0))


def _ensure_vector_python():
    """Re-exec under the vector venv when qdrant-client is not importable here."""
    try:
        import qdrant_client  # noqa: F401
        return
    except ImportError:
        pass
    vpy = os.environ.get("ENGRAM_VECTOR_PYTHON") or str(Path.home() / ".claude/vector/venv/bin/python")
    if os.environ.get("ENGRAM_CURATE_REEXEC") != "1" and Path(vpy).is_file():
        os.environ["ENGRAM_CURATE_REEXEC"] = "1"      # one hop only: a broken venv must not loop
        os.execv(vpy, [vpy, str(Path(__file__).resolve()), *sys.argv[1:]])
    print(f"[auto-curate] ERROR: qdrant-client not importable by {sys.executable} "
          f"(vector venv: {vpy})", file=sys.stderr)
    sys.exit(2)


def main():
    apply = "--apply" in sys.argv
    cfg = memory_ai.load()
    ac = ac_cfg(cfg)
    if apply and not ac["enabled"]:
        print("[auto-curate] --apply ignored: auto_curate.enabled is false. Dry-run.", file=sys.stderr)
        apply = False

    # fixation scores (to pick which member becomes the umbrella)
    try:
        st = json.loads((MEM / ".fixation_state.json").read_text()).get("memories", {})
        scores = {k: (v.get("last_score") or 0.0) for k, v in st.items()}
    except Exception:
        scores = {}

    # deterministic near-dup pairs via the Qdrant ANN finder. qdrant-client lives in the
    # vector venv, not the system python the memory-tidy skill invokes: re-exec there
    # once. A store we cannot reach is a FAILURE (exit 2), never "nothing to do" — that
    # exit-0 message hid a dead curator for weeks.
    _ensure_vector_python()
    try:
        from vector_store import EngramVectorStore
        store = EngramVectorStore(cfg); store.ensure_collection()
        pairs = store.find_duplicates(threshold=ac["merge_threshold"], max_pairs=100)
    except Exception as e:
        print(f"[auto-curate] ERROR: vector store unavailable ({e})", file=sys.stderr)
        sys.exit(2)

    clusters = _clusters(pairs, ac["merge_threshold"])
    # SAME-type only (never merge a feedback into a project); require all files present
    typed = []
    for c in clusters:
        files = [f for f in c if (MEM / f).is_file()]
        if len(files) < 2:
            continue
        types = {_fm(MEM / f)[2] for f in files}
        if len(types) == 1:
            typed.append(sorted(files))
        else:
            print(f"[auto-curate] skip mixed-type cluster {files}")
    typed = typed[:ac["max_merges_per_run"]]

    gate_mode = _resolve_gate(ac)     # "codex" or "human"
    print(f"# memory_auto_curate — apply={apply} enabled={ac['enabled']} "
          f"threshold={ac['merge_threshold']} cap={ac['max_merges_per_run']} review_gate={gate_mode}")
    print(f"near-dup clusters this run: {len(typed)}")
    import engram_telegram_gate as gate
    done = queued = 0
    for files in typed:
        paths = [MEM / f for f in files]
        # Snapshot BEFORE any analysis. Hashing after the containment test (or after a
        # minutes-long distillation) would record whatever a concurrent writer had just
        # produced, so _apply_merge would wave through an umbrella computed from the OLD
        # bytes while a newly added fact sat unread in the source.
        sha0 = _src_sha(files)
        # ── SUPERSEDE first: if the members are identical copies, this is not a
        # compression problem at all. Keep one BYTE-FOR-BYTE and quarantine the rest —
        # no LLM, no coverage score, nothing generated that could be lossy or malformed.
        # Without this path an exact duplicate goes through umbrella generation, scores
        # ~0.5 coverage (the gate is lexical, so two copies of one fact never "cover"
        # each other), and lands in the human queue for no reason.
        keeper = _supersede_keeper(files)
        if keeper:
            absorbed = [f for f in files if f != keeper]
            content = (MEM / keeper).read_text(errors="ignore")
            kname, kdesc, _kt = _fm(MEM / keeper)
            # merge_id must depend on CONTENT, not just filenames: a later merge that
            # recreates the same filenames would otherwise reuse this transaction dir
            # and overwrite an unresolved backup, destroying the earlier undo.
            merge_id = __import__("hashlib").sha256(
                (keeper + "".join(sorted(absorbed)) + json.dumps(sha0, sort_keys=True)
                 + "supersede").encode()).hexdigest()[:10]
            params = {"umbrella": keeper, "umbrella_content": content, "desc": kdesc,
                      "absorbed": absorbed, "merge_id": merge_id,
                      "src_sha": sha0}
            preview = (f"Supersede: {absorbed} are canonically IDENTICAL copies of "
                       f"{keeper}. Keeper unchanged; absorbed→quarantine.")
            print(f"\n· {files} -> {keeper}  [SUPERSEDE — lossless]")
            if not apply:
                print(f"  would: {preview}"); continue
            if gate_mode == "human" and not ac.get("autonomous"):  # explicit no-auto policy
                # (autonomous DOES apply supersede: byte-identical keeper, zero model judgment)
                gate.propose("merge_apply", params, f"Supersede (lossless). {preview}",
                             files=files, codex_verdict="supersede-deterministic")
                print("  QUEUED (human gate)"); queued += 1
                continue
            ok, detail = gate._apply_merge(params)
            if ok:
                gate.notify_undo("merge_undo",
                                 {"umbrella": keeper, "absorbed": absorbed, "merge_id": merge_id},
                                 f"🧠 superseded (lossless): {preview}\nTap UNDO to reverse.")
                print(f"  APPLIED ({detail})"); done += 1
            else:
                print(f"  FAILED ({detail})")
            continue
        umbrella = max(paths, key=lambda p: _score(p, scores))   # keep the highest-trust member's name
        name, desc, typ = _fm(umbrella)
        # COMPRESS for real (no preserve_sources) — the umbrella is a tight consolidation.
        # Losslessness is provided by REVERSIBILITY (sources -> quarantine), not by
        # appending everything. sentence_coverage is recorded, not gated.
        final, report = mdv.distill_cluster(umbrella.stem, files, cfg=cfg, preserve_sources=False)
        body = _body_of(final)
        if not body.strip():
            print(f"· {files}: HOLD (empty umbrella) — skipping"); continue
        doc = f"---\nname: {umbrella.stem}\ndescription: {desc}\nmetadata:\n  type: {typ}\n---\n\n{body}"
        okdoc, whydoc = gate.validate_memory_doc(doc)
        if not okdoc:
            print(f"· {files}: HOLD (malformed umbrella: {whydoc}) — skipping"); continue
        # Credential RECONCILIATION (replaces the blanket secret-HOLD, 2026-11-12):
        # every REMOTE hop (distill prompt, Codex blob) already redacts AGGRESSIVELY,
        # so a credential in the cluster is not a leak risk — the real risk is the
        # umbrella silently DROPPING one. So a cluster merges even when it touches
        # credentials, as long as each credential found in any source still appears
        # verbatim in the final doc. Only CREDENTIAL_RE (high-precision) counts —
        # txids/EVM-addresses/UUIDs are public data the aggressive scanner used to
        # block on, which is why secret-bearing clusters never consolidated.
        missing = sorted({c for f in files
                          for c in engram_secrets.credentials((MEM / f).read_text(errors="ignore"))
                          if c not in doc})
        if missing:
            print(f"· {files}: HOLD (umbrella drops {len(missing)} credential(s): "
                  f"{[c[:50] for c in missing[:3]]}) — skipping"); continue
        absorbed = [f for f in files if f != umbrella.name]
        # transaction id so a second merge on the same umbrella can't clobber the
        # first merge's backup (each merge's originals live under their own dir)
        import hashlib as _hl
        merge_id = _hl.sha256((umbrella.name + "".join(sorted(absorbed)) + doc
                               + json.dumps(sha0, sort_keys=True)).encode()).hexdigest()[:10]
        params = {"umbrella": umbrella.name, "umbrella_content": doc, "desc": desc,
                  "absorbed": absorbed, "merge_id": merge_id, "src_sha": sha0}
        preview = (f"Merge {len(files)} near-dups → {umbrella.name} (compressed "
                   f"{report.get('draft_chars')} chars, prose-cov {report.get('sentence_coverage')}). "
                   f"Absorbed→quarantine: {', '.join(absorbed)}")
        print(f"\n· {files} -> {umbrella.name}")
        if not apply:
            print(f"  would: {preview}"); continue
        cov = report.get("sentence_coverage", 0.0)
        # HUMAN gate (no Codex): every merge goes to the Telegram approval queue.
        if gate_mode == "human":
            if ac.get("autonomous"):
                print(f"· {files}: SKIP (autonomous but no Codex reviewer) — fail-closed"); continue
            gate.propose("merge_apply", params, f"Needs your approval (no Codex). {preview}",
                         files=files, codex_verdict="human-gate")
            print("  QUEUED for human approval (human gate — no Codex)"); queued += 1
            continue
        # CODEX gate: auto-apply only a near-LOSSLESS + clean merge; lossy or non-clean
        # -> human queue. coverage is on the FULL sources (not Codex's truncated slice).
        # autonomous mode widens this: Codex APPROVE alone is enough (reversibility,
        # not coverage, is the safety), everything else is SKIPPED below.
        verdict = _codex_verdict(doc, files, ac)
        if verdict == "APPROVE" and (ac.get("autonomous") or cov >= ac["min_auto_coverage"]):
            ok, detail = gate._apply_merge(params)
            # Report the ACTUAL outcome — this used to print AUTO-APPLIED before
            # checking ok, so a refused or partial merge still read as success.
            print(f"  {'AUTO-APPLIED' if ok else 'FAILED'} (cov={cov}; {detail})")
            if ok:
                gate.notify_undo("merge_undo",
                                 {"umbrella": umbrella.name, "absorbed": absorbed, "merge_id": merge_id},
                                 f"🧠 auto-merged (near-lossless): {preview}\nTap UNDO to reverse.")
                done += 1
        else:
            if verdict != "APPROVE":                    # name the REAL reason — cov may be fine
                why = f"Codex {verdict}"
            else:
                why = f"lossy compression (cov={cov} < {ac['min_auto_coverage']})"
            if ac.get("autonomous"):
                print(f"· {files}: SKIP ({why}) — not queued"); continue
            gate.propose("merge_apply", params, f"Needs approval — {why}. {preview}",
                         files=files, codex_verdict=verdict)
            print(f"  QUEUED for human approval ({why})"); queued += 1

    pruned = _orphan_prune_pass(cfg, ac, store, gate, apply) if ac.get("prune_orphans") else 0
    print(f"\napplied {done}, queued {queued}, orphan-prune proposed {pruned}."
          if apply else "\n(dry-run — set auto_curate.enabled + --apply)")


def _orphan_prune_pass(cfg, ac, store, gate, apply) -> int:
    """PROPOSE stale orphans for a one-tap Telegram prune (never auto). An orphan is:
    type∈{project,reference}, age>=orphan_age_days, frequency==0 (never discussed), NO
    close neighbor (no merge target), not a suspect, and not a skill source."""
    try:
        r = subprocess.run([sys.executable, str(HOME / ".claude" / "memory_score.py"), "--json"],
                           capture_output=True, text=True, timeout=300)
        scored = json.loads(r.stdout).get("memories", [])
    except Exception as e:
        print(f"[orphan-prune] scorer unavailable ({e}) — skipping"); return 0
    # set of files that HAVE a merge target (close neighbor >= 0.80) — those are not orphans
    close = set()
    try:
        for _s, a, b in store.find_duplicates(threshold=0.80, max_pairs=2000):
            close.add(a); close.add(b)
    except Exception:
        pass
    age_min = float(ac.get("orphan_age_days", 90))
    prunable = []
    for m in scored:
        name = m.get("name", "")
        if m.get("type") not in ("project", "reference"):     # never user/feedback
            continue
        if float(m.get("age_days", 0)) < age_min or m.get("frequency", 1) != 0 or m.get("suspicion"):
            continue
        if name in close:                                     # has a merge target -> not an orphan
            continue
        try:
            if "Promoted to skill:" in (MEM / name).read_text(errors="ignore"):
                continue                                      # skill sources are load-bearing
        except Exception:
            continue
        prunable.append(name)
    n = 0
    for name in prunable[:int(ac.get("max_prunes_per_run", 5))]:
        preview = f"Prune stale orphan {name}: aged ≥{int(age_min)}d, never recalled, no merge target. → .trash (recoverable)."
        if apply and ac.get("autonomous"):
            # autonomous: prune straight to .trash/ (90-day undo), no Telegram proposal
            r = subprocess.run(["bash", str(DELETE), name], capture_output=True, text=True, timeout=60)
            print(f"  PRUNED: {name}" if r.returncode == 0 else f"  prune FAILED: {name}: {r.stderr.strip()[:150]}")
        elif apply:
            gate.propose("orphan_prune", {"name": name}, preview, files=[name])
        else:
            print(f"  would propose prune: {name}")
        n += 1
    if prunable:
        mode = "pruned" if ac.get("autonomous") else "proposed"
        print(f"[orphan-prune] {len(prunable)} orphan(s); {mode} {n} (cap {ac.get('max_prunes_per_run', 5)})")
    return n


def _strict_verdict(stdout: str) -> str:
    """Strict token rules — injected source text could try to spoof a verdict.
    Require: exactly ONE verdict token anywhere, it is APPROVE, and it is on the LAST
    non-empty line. Anything else is not an approval (fail-closed)."""
    import re as _re
    verdicts = _re.findall(r"VERDICT:\s*(APPROVE|REJECT)", stdout or "", _re.I)
    lines = [ln.strip() for ln in (stdout or "").splitlines() if ln.strip()]
    last = lines[-1].upper() if lines else ""
    if len(verdicts) == 1 and verdicts[0].upper() == "APPROVE" and last == "VERDICT: APPROVE":
        return "APPROVE"
    return "REJECT" if verdicts else "DEFER-no-verdict"


def _claude_fallback_verdict(task, blob) -> str:
    """Fallback reviewer per the owner's standing rule (Codex first; Claude headless
    when Codex is logged out/rate-limited/absent). Same strict token gate. NOTE the
    weaker independence — the generation backend is also Claude — the hard safety is
    unchanged: 30-day quarantine + the reconciliation guard, not reviewer variety."""
    import shutil
    if not shutil.which("claude"):
        return "DEFER-no-claude"
    try:
        r = subprocess.run(
            ["claude", "-p", "--output-format", "text", "--no-session-persistence"],
            input=f"{task}\n\n{blob}", capture_output=True, text=True, timeout=240)
    except Exception as e:
        return f"DEFER-claude-{type(e).__name__}"
    if r.returncode != 0:
        return "DEFER-claude-rc"
    return _strict_verdict(r.stdout)


def _codex_verdict(umbrella_doc, source_files, ac) -> str:
    """Ask the Codex advisor whether the compressed umbrella loses information vs its
    sources or introduces a secret. Returns APPROVE only on an explicit clean verdict.
    A MECHANICAL failure (no advisor, rate-limited, no token) routes to the fallback
    reviewer (Claude headless) when fallback_reviewer is on; a real REJECT stays."""
    task = ("Does this umbrella drop any durable fact from the sources, or add a secret/"
            "hallucination? Reply a final line exactly 'VERDICT: APPROVE' or 'VERDICT: REJECT'.")
    # Redact EVERYTHING before it reaches the advisor subprocess/backend/logs — the
    # whole umbrella doc (its frontmatter description may hold a secret) AND every
    # source. A scanner-missed credential in a legacy memory must not leak.
    safe_umbrella = engram_secrets.redact(umbrella_doc)[0]
    srcs = "\n\n".join(f"### {f}\n{engram_secrets.redact((MEM/f).read_text(errors='ignore')[:2000])[0]}"
                       for f in source_files)
    blob = f"PROPOSED UMBRELLA:\n{safe_umbrella[:4000]}\n\nSOURCES IT REPLACES:\n{srcs[:6000]}"
    fb = bool(ac.get("fallback_reviewer", True))
    if not ADVISOR.exists():
        return _claude_fallback_verdict(task, blob) if fb else "DEFER-no-advisor"
    try:
        r = subprocess.run(
            [sys.executable, str(ADVISOR), "--mode", "review", "--stdin", "--task", task],
            input=blob, capture_output=True, text=True, timeout=180)
    except Exception as e:
        return _claude_fallback_verdict(task, blob) if fb else f"DEFER-{type(e).__name__}"
    if r.returncode != 0:
        return _claude_fallback_verdict(task, blob) if fb else "DEFER-advisor-rc"
    v = _strict_verdict(r.stdout)
    if v.startswith("DEFER") and fb:      # token gate got nothing parseable — try fallback once
        return _claude_fallback_verdict(task, blob)
    return v


if __name__ == "__main__":
    main()
