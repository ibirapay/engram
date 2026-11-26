#!/usr/bin/env python3
"""Graduation drains its queue on its own (2026-11-26): a cap of 8 per run in name order let
never-resolving holds take 7 slots every run, so one candidate a day was evaluated against an
intake of 150-250. Now every candidate goes through the cheap gates, user-direct first and then
oldest first; failed provenance/confidence gates are terminal (.expired); model time is bounded by
a budget and a breaker, and what does not fit is deferred; a near-copy of something graduated
earlier in the same run is dropped."""
import contextlib, importlib.util, io, json, os, sys, tempfile, types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run(cands, ag=None, verdict=lambda body: "SAFE", apply=False, clock_step=0.0):
    """Drive main() on a temp store. cands: [(name, provenance, confidence, tag, age_hours)];
    candidates with the same tag embed identically (cosine 1), different tags are orthogonal.
    Returns (decisions in order, staging dir)."""
    d = Path(tempfile.mkdtemp())
    os.environ["CLAUDE_MEMORY_SLUG"] = "-t"
    mem = d / "memory"; staging = mem / ".staging"; staging.mkdir(parents=True)
    (mem / "project_live.md").write_text("---\nname: project_live\ndescription: tag-live\n---\nlive body\n")
    tags = {}

    def embed(text, cfg=None, role=None):
        tag = next((t for t in text.split() if t.startswith("tag-")), "tag-none")
        i = tags.setdefault(tag, len(tags))
        return [1.0 if k == i else 0.0 for k in range(64)]
    now = 1_800_000_000
    for name, prov, conf, tag, age_h in cands:
        p = staging / name
        p.write_text(f"---\nname: {name[:-3]}\ndescription: {tag} candidate\nmetadata:\n  type: project\n"
                     f"harvest:\n  status: quarantined\n  provenance: {prov}\n  confidence: {conf}\n"
                     f"  source_session: s\n---\n\nbody of {tag}\n")
        os.utime(p, (now - age_h * 3600, now - age_h * 3600))
    mai = types.ModuleType("memory_ai")
    mai.load = lambda: {"auto_graduate": {"enabled": True, "allow_provenance": ["user-direct", "assistant"],
                                          "staging_ttl_days": 0, **(ag or {})}}
    mai.ollama_embed = embed
    mai.is_transient_fact = lambda name, desc: False
    sys.modules["memory_ai"] = mai
    sp = importlib.util.spec_from_file_location("sa_auto", ROOT / "bin" / "memory_stage_apply.py")
    sa = importlib.util.module_from_spec(sp); sp.loader.exec_module(sa)
    sa.MEM_DIR, sa.STAGING, sa.QUAR = mem, staging, mem / ".quarantine"
    sa._EMBED_CACHE = mem / ".stage_embed_cache.json"
    sa.memory_ai = mai
    t = {"now": 0.0}

    def model(body, cfg):
        t["now"] += clock_step          # each model call takes clock_step seconds
        return verdict(body)
    sa.injection_verdict = model
    sa.graduate = lambda fname, content, desc, apply: "graduated" if apply else "would-graduate"
    sa._clock = lambda: t["now"]
    argv, sys.argv = sys.argv, ["sa", "--json"] + (["--apply"] if apply else [])
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            sa.main()
    finally:
        sys.argv = argv
    return json.loads(out.getvalue())["decisions"], staging


def test_whole_queue_in_priority_order():
    # 12 candidates (more than the old cap of 8), names chosen so name order != priority order
    cands = [(f"a_assistant_{i}.md", "assistant", 0.9, f"tag-a{i}", 10 + i) for i in range(10)]
    cands += [("z_user_new.md", "user-direct", 0.9, "tag-u1", 1), ("y_user_old.md", "user-direct", 0.9, "tag-u2", 50)]
    decs, _ = _run(cands)
    order = [x["file"] for x in decs]
    assert len(decs) == 12, f"the queue was capped: {len(decs)}"
    assert order[:2] == ["y_user_old.md", "z_user_new.md"], f"user-direct (oldest first) must lead: {order}"
    assert order[2:] == [f"a_assistant_{i}.md" for i in range(9, -1, -1)], f"then oldest first: {order}"
    print("ok — whole queue every run, user-direct first, then oldest first")


def test_terminal_holds_leave_the_queue():
    cands = [("b_low.md", "assistant", 0.5, "tag-b", 5), ("c_unverified.md", "unverified", 1.0, "tag-c", 5),
             ("d_good.md", "assistant", 0.9, "tag-d", 5)]
    decs, staging = _run(cands, apply=True)
    acts = {x["file"]: x["action"] for x in decs}
    assert acts == {"b_low.md": "expire-terminal", "c_unverified.md": "expire-terminal", "d_good.md": "graduated"}, acts
    assert sorted(p.name for p in (staging / ".expired").glob("*.md")) == ["b_low.md", "c_unverified.md"]
    assert not list(staging.glob("*.md")), "a terminal candidate stayed in the queue"
    print("ok — failed provenance/confidence gates are terminal (.expired), never re-taken")


def test_model_time_budget_defers_the_rest():
    cands = [(f"e_{i}.md", "assistant", 0.9, f"tag-e{i}", 10 - i) for i in range(5)]
    decs, staging = _run(cands, ag={"llm_budget_seconds": 10}, clock_step=6.0, apply=True)
    acts = [x["action"] for x in decs]
    assert acts == ["graduated", "graduated", "defer", "defer", "defer"], acts
    assert len(list(staging.glob("*.md"))) == 3, "deferred candidates must stay in the queue"
    print("ok — model time is bounded; what does not fit is deferred and kept")


def test_breaker_stops_a_dead_model():
    def boom(body):
        raise RuntimeError("backend down")
    cands = [(f"f_{i}.md", "assistant", 0.9, f"tag-f{i}", 10 - i) for i in range(5)]
    decs, staging = _run(cands, verdict=boom, apply=True)
    acts = [x["action"] for x in decs]
    assert acts == ["hold", "hold", "hold", "defer", "defer"], acts
    assert "failed 3x" in decs[3]["reason"], decs[3]
    assert len(list(staging.glob("*.md"))) == 5, "a transient failure must keep every candidate"
    # the breaker counts failures IN A ROW: a success in between resets it
    seq = iter([RuntimeError, RuntimeError, "SAFE", RuntimeError, RuntimeError, "SAFE"])

    def flaky(body):
        r = next(seq)
        if r is RuntimeError:
            raise RuntimeError("blip")
        return r
    cands = [(f"j_{i}.md", "assistant", 0.9, f"tag-j{i}", 10 - i) for i in range(6)]
    decs, _ = _run(cands, verdict=flaky)
    assert [x["action"] for x in decs] == ["hold", "hold", "would-graduate", "hold", "hold", "would-graduate"], decs
    print("ok — three model failures in a row stop the model for the run; nothing is lost")


def test_near_copy_in_the_same_run_drops():
    cands = [("g_first.md", "assistant", 0.9, "tag-same", 9), ("h_second.md", "assistant", 0.9, "tag-same", 8),
             ("i_store_dup.md", "assistant", 0.9, "tag-live", 7)]
    decs, _ = _run(cands)
    acts = {x["file"]: (x["action"], x["reason"]) for x in decs}
    assert acts["g_first.md"][0] == "would-graduate", acts
    assert acts["h_second.md"][0] == "drop-duplicate" and "(this run) g_first.md" in acts["h_second.md"][1], acts
    assert acts["i_store_dup.md"][0] == "drop-duplicate" and "project_live.md" in acts["i_store_dup.md"][1], acts
    print("ok — a near-copy of something graduated earlier in the run is dropped")


def test_activity_summary_reports_ttl_and_model_time():
    os.environ["CLAUDE_MEMORY_SLUG"] = "-t"
    sp = importlib.util.spec_from_file_location("gate_auto", ROOT / "bin" / "engram_telegram_gate.py")
    g = importlib.util.module_from_spec(sp); sp.loader.exec_module(g)
    d = Path(tempfile.mkdtemp())
    g.PIPELINE_LOG = d / "pipeline.log"; g.ACTIVITY_STAMP = d / ".last_activity"
    g.PIPELINE_LOG.write_text(
        "[2026-11-27T05:00:00] pipeline start (apply='--apply')\ntranscripts processed: 3\ncandidates staged: 40\n"
        "pruned 12 stale staged file(s) (> 14d)\nmodel time used: 931s of 1800\n"
        "processed 90 staged candidate(s): {'graduated': 50, 'drop-duplicate': 40}\n")
    sent = []
    g.notify = sent.append
    g.activity_summary()
    assert sent and "expired by TTL: 12" in sent[0] and "model time: 931s of 1800" in sent[0], sent
    print("ok — the Telegram run summary shows TTL expiries and model time")


if __name__ == "__main__":
    test_whole_queue_in_priority_order()
    test_terminal_holds_leave_the_queue()
    test_model_time_budget_defers_the_rest()
    test_breaker_stops_a_dead_model()
    test_near_copy_in_the_same_run_drops()
    test_activity_summary_reports_ttl_and_model_time()
