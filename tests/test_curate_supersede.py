#!/usr/bin/env python3
"""auto-curate safety: the supersede path, clique clustering, anchored frontmatter,
and the document validator that stops a malformed umbrella reaching the store."""
import sys, types, importlib.util, os, tempfile, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load(mod, path, **stubs):
    for k, v in stubs.items():
        sys.modules[k] = v
    sp = importlib.util.spec_from_file_location(mod, path)
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return m


def _gate():
    os.environ.setdefault("CLAUDE_MEMORY_SLUG", "-t")
    return _load("engram_telegram_gate", ROOT / "bin" / "engram_telegram_gate.py")


def _curate(memdir):
    g = _gate()
    mai = types.ModuleType("memory_ai"); mai.load = lambda: {}
    sec = types.ModuleType("engram_secrets")
    sec.looks_secret = lambda t: False; sec.redact = lambda t: (t, 0)
    mdv = types.ModuleType("memory_distill_verified")
    m = _load("memory_auto_curate", ROOT / "bin" / "memory_auto_curate.py",
              memory_ai=mai, engram_secrets=sec, memory_distill_verified=mdv,
              engram_telegram_gate=g)
    m.MEM = Path(memdir)
    return m


# ── document validation ─────────────────────────────────────────────────────
def test_validator_rejects_the_real_malformed_shapes():
    g = _gate()
    good = "---\nname: a\ndescription: d\n---\n\nbody text\n"
    assert g.validate_memory_doc(good)[0], "valid doc rejected"

    # the shape actually observed in a queued proposal: fence + double frontmatter
    bad = "---\nname: a\ndescription: d\n---\n\n```markdown\n---\nname: a\n---\n\nbody\n```\n"
    ok, why = g.validate_memory_doc(bad)
    assert not ok and "SECOND frontmatter" in why, f"double frontmatter accepted: {why}"

    # REGRESSION (real payload, merge_apply-43f08623b060): the fence CLOSES MID-BODY
    # because distill appends a "Hard facts" section after it, so a whole-body unwrap
    # sees nothing. This escaped the first version of the validator.
    real = ("---\nname: a\ndescription: d\nmetadata:\n  type: project\n---\n\n"
            "```markdown\n---\nname: a\ndescription: d\nmetadata:\n  type: project\n---\n\n"
            "## Summary\nreal summary\n```\n\n_Facts note._\n### Hard facts\n- `x.py` (file)\n")
    ok, why = g.validate_memory_doc(real)
    assert not ok and "SECOND frontmatter" in why, f"real malformed payload accepted: {why}"

    for doc, label in [("```markdown\n---\nname: a\n---\nbody\n```", "outer fence"),
                       ("no frontmatter at all", "missing frontmatter"),
                       ("---\nname: a\n---\n\n   \n", "empty body"),
                       ("", "empty")]:
        assert not g.validate_memory_doc(doc)[0], f"{label} accepted"
    print("ok — validator rejects fenced / double-frontmatter / empty docs")


def test_unwrap_fence_leaves_inner_code_blocks_alone():
    g = _gate()
    assert g.unwrap_code_fence("```markdown\nhello\n```") == "hello"
    body = "text\n\n```bash\nls -la\n```\n\nmore text"
    assert g.unwrap_code_fence(body) == body, "inner code block was mangled"
    print("ok — unwraps only a whole-text fence, never an inner block")


# ── supersede ───────────────────────────────────────────────────────────────
def test_supersede_only_on_identical_copies():
    d = tempfile.mkdtemp()
    sup = "The quick brown fox jumps over the lazy dog near the river bank at dawn every day."
    fm = "---\nname: {n}\ndescription: same desc\nmetadata:\n  type: project\n---\n\n"
    # true copies: different filenames + different provenance boilerplate, same content
    Path(d, "copy_a.md").write_text(fm.format(n="shared") + sup +
        "\n\n_Provenance: auto-harvested from session AAA._\n")
    Path(d, "copy_b.md").write_text(fm.format(n="shared") + sup +
        "\n\n_Provenance: auto-harvested from session BBB._\n")
    m = _curate(d)
    assert m._supersede_keeper(["copy_a.md", "copy_b.md"]) is not None, \
        "identical copies (differing only in provenance) were not superseded"

    # A SUPERSET must NOT auto-supersede. Containment proves the text is present, not
    # that it still means the same thing: a larger STALE doc can quote the current rule
    # in order to call it obsolete, and would be picked as keeper (codex review, HIGH).
    Path(d, "small.md").write_text(fm.format(n="small") + sup + "\n")
    Path(d, "big_stale.md").write_text(fm.format(n="big_stale") + sup +
        " The above is OBSOLETE; withdrawals are now disabled.\n")
    assert m._supersede_keeper(["small.md", "big_stale.md"]) is None, \
        "superset auto-superseded — a stale doc quoting a rule as obsolete would win"

    # rewording, differing description, and operator flips all decline too
    Path(d, "reworded.md").write_text(fm.format(n="reworded") +
        "A speedy auburn vulpine leaps above an idle canine beside the stream.\n")
    assert m._supersede_keeper(["small.md", "reworded.md"]) is None, "rewording superseded"
    Path(d, "otherdesc.md").write_text(
        "---\nname: otherdesc\ndescription: DIFFERENT\nmetadata:\n  type: project\n---\n\n" + sup + "\n")
    assert m._supersede_keeper(["small.md", "otherdesc.md"]) is None, \
        "differing description ignored — it can carry a fact absent from the body"
    # SCOPE-BEARING NAMES (codex review, HIGH): identical generic bodies that differ
    # only by `name` are different rules, not copies.
    scoped = "Treasury withdrawals require two signatures.\n"
    Path(d, "eth_rule.md").write_text(
        "---\nname: eth_treasury_rule\ndescription: same desc\nmetadata:\n  type: project\n---\n\n" + scoped)
    Path(d, "btc_rule.md").write_text(
        "---\nname: btc_treasury_rule\ndescription: same desc\nmetadata:\n  type: project\n---\n\n" + scoped)
    assert m._supersede_keeper(["eth_rule.md", "btc_rule.md"]) is None, \
        "different `name` treated as a copy — a chain-specific rule would be dropped"
    Path(d, "lt.md").write_text(fm.format(n="lt") + "Withdrawals capped at < 500 per day.\n")
    Path(d, "gt.md").write_text(fm.format(n="gt") + "Withdrawals capped at > 500 per day.\n")
    assert m._supersede_keeper(["lt.md", "gt.md"]) is None, \
        "`<` vs `>` normalized to the same text — a contradictory rule could be dropped"
    # BOILERPLATE STRIP MUST NOT SPAN BODY TEXT (codex review, HIGH): a non-greedy
    # `_Provenance:.*?_` under re.S ran to the next underscore, so a crafted line could
    # swallow real content and make two different memories canonicalize identically.
    poisoned = (fm.format(n="shared") + "_Provenance: x_ REAL RULE: send to 0xAAA\n"
                "more body with an _emphasis_ marker\n")
    clean = fm.format(n="shared") + "_Provenance: x_ REAL RULE: send to 0xBBB\n" \
            "more body with an _emphasis_ marker\n"
    Path(d, "poison_a.md").write_text(poisoned)
    Path(d, "poison_b.md").write_text(clean)
    assert m._supersede_keeper(["poison_a.md", "poison_b.md"]) is None, \
        "boilerplate regex swallowed body text — two different rules looked identical"
    shutil.rmtree(d)
    print("ok — supersede fires ONLY on identical copies; supersets/rewordings decline")


# ── clustering + frontmatter ────────────────────────────────────────────────
def test_clusters_require_a_clique():
    m = _curate(tempfile.mkdtemp())
    # A≈B and B≈C but NOT A≈C: union-find would delete across the whole chain
    out = m._clusters([(0.95, "a", "b"), (0.95, "b", "c")], 0.92)
    assert {"a", "b", "c"} not in out, "transitive closure produced an unsafe 3-cluster"
    # ...and the fallback pairs must be DISJOINT (codex review, HIGH): overlapping
    # A-B and B-C merges undo out of order and destroy the original B.
    assert out == [{"a", "b"}], f"overlapping pairs emitted: {out}"
    flat = [f for c in out for f in c]
    assert len(flat) == len(set(flat)), f"a file appears in two merges: {out}"
    assert {"a", "b", "c"} in m._clusters(
        [(0.95, "a", "b"), (0.95, "b", "c"), (0.95, "a", "c")], 0.92), "complete clique split"
    print("ok — cliques only; A-B-C chain never becomes one cluster")


def test_fm_is_anchored():
    d = tempfile.mkdtemp()
    Path(d, "spoof.md").write_text(
        "---\nname: real\ndescription: d\nmetadata:\n  type: reference\n---\n\n"
        "Body prose that mentions type: project in passing.\n")
    m = _curate(d)
    assert m._fm(Path(d, "spoof.md"))[2] == "reference", "body text spoofed the type"
    Path(d, "nofm.md").write_text("type: project\n\njust a body\n")
    assert m._fm(Path(d, "nofm.md"))[2] == "", "unanchored type accepted"
    shutil.rmtree(d)
    print("ok — frontmatter parsed anchored; body cannot spoof type")


def test_apply_merge_refuses_stale_sources():
    """A proposal can sit in the approval queue for hours; sources must be re-verified."""
    import hashlib
    g = _gate()
    d = tempfile.mkdtemp(); g.MEM = Path(d)
    doc = "---\nname: k\ndescription: d\n---\n\nkeeper body\n"
    Path(d, "k.md").write_text(doc)
    Path(d, "a.md").write_text("---\nname: a\ndescription: d\n---\n\nabsorbed\n")
    sha = {f: hashlib.sha256(Path(d, f).read_bytes()).hexdigest() for f in ("k.md", "a.md")}
    Path(d, "a.md").write_text("---\nname: a\ndescription: d\n---\n\nabsorbed + A NEW FACT\n")
    ok, why = g._apply_merge({"umbrella": "k.md", "umbrella_content": doc, "desc": "d",
                              "absorbed": ["a.md"], "merge_id": "t1", "src_sha": sha})
    assert not ok and "changed since" in why, f"stale source merged anyway: {why}"
    # FAIL CLOSED for a legacy proposal with no snapshot (codex review, HIGH): an empty
    # dict would make the verification loop a silent no-op.
    ok, why = g._apply_merge({"umbrella": "k.md", "umbrella_content": doc, "desc": "d",
                              "absorbed": ["a.md"], "merge_id": "t2"})
    assert not ok and "no complete source snapshot" in why, f"legacy proposal applied: {why}"
    # ...and for a truncated snapshot that omits one of the files being touched
    ok, why = g._apply_merge({"umbrella": "k.md", "umbrella_content": doc, "desc": "d",
                              "absorbed": ["a.md"], "merge_id": "t3",
                              "src_sha": {"k.md": sha["k.md"]}})
    assert not ok and "no complete source snapshot" in why, f"truncated snapshot applied: {why}"
    assert "A NEW FACT" in Path(d, "a.md").read_text(), "the new fact was destroyed"
    shutil.rmtree(d)
    print("ok — _apply_merge refuses when a source changed after analysis")


def test_apply_merge_refuses_to_reuse_a_transaction_dir():
    """A pre-existing txn dir means an unresolved backup; reusing it destroys that undo."""
    import hashlib
    g = _gate(); d = tempfile.mkdtemp(); g.MEM = Path(d)
    doc = "---\nname: k\ndescription: d\n---\n\nkeeper body\n"
    Path(d, "k.md").write_text(doc)
    prior = Path(d, ".quarantine", "merge-dup1"); prior.mkdir(parents=True)
    (prior / "precious.orig").write_text("an earlier merge's only backup")
    sha = {"k.md": hashlib.sha256(Path(d, "k.md").read_bytes()).hexdigest()}
    ok, why = g._apply_merge({"umbrella": "k.md", "umbrella_content": doc, "desc": "d",
                              "absorbed": [], "merge_id": "dup1", "src_sha": sha})
    assert not ok and "already exists" in why, f"reused a transaction dir: {why}"
    assert (prior / "precious.orig").read_text() == "an earlier merge's only backup", \
        "prior backup was clobbered"
    shutil.rmtree(d)
    print("ok — refuses to reuse a transaction dir, prior undo preserved")


def test_apply_merge_rejects_a_file_in_an_outstanding_transaction():
    """A file absorbed by one un-undone merge must not be the umbrella of another."""
    import hashlib
    g = _gate(); d = tempfile.mkdtemp(); g.MEM = Path(d)
    doc = "---\nname: b\ndescription: d\n---\n\nbody\n"
    Path(d, "b.md").write_text(doc)
    prior = Path(d, ".quarantine", "merge-earlier"); prior.mkdir(parents=True)
    (prior / "b.md").write_text("b as absorbed by an earlier, un-undone merge")
    sha = {"b.md": hashlib.sha256(Path(d, "b.md").read_bytes()).hexdigest()}
    ok, why = g._apply_merge({"umbrella": "b.md", "umbrella_content": doc, "desc": "d",
                              "absorbed": [], "merge_id": "later", "src_sha": sha})
    assert not ok and "outstanding merge backup" in why, f"chained merge allowed: {why}"
    shutil.rmtree(d)
    print("ok — refuses a file already held by an outstanding transaction")


def test_an_earlier_merge_settles_after_settle_days():
    """An earlier merge nobody undid for settle_days no longer locks its files out of curation:
    its backup goes to .trash and its UNDO record closes; a younger one still blocks."""
    import hashlib, json, time
    g = _gate(); d = Path(tempfile.mkdtemp()); g.MEM = d; g.Q = d / ".approvals"; g.HOME = d
    (d / ".claude").mkdir()
    save = d / ".claude" / "save_memory.sh"      # fake writer: stdin -> the memory file
    save.write_text(f'#!/bin/sh\ncat > "{d}/$1"\n'); save.chmod(0o755)
    doc = "---\nname: b\ndescription: d\n---\n\nbody\n"
    (d / "b.md").write_text(doc)
    prior = d / ".quarantine" / "merge-earlier"; prior.mkdir(parents=True)
    (prior / "b.md.orig").write_text("b before the earlier merge")
    for uid, mid in (("u1", "earlier"), ("u2", "unrelated")):
        (g._dir("undo") / f"{uid}.json").write_text(json.dumps(
            {"id": uid, "op": "merge_undo", "params": {"umbrella": "b.md", "absorbed": [], "merge_id": mid}}))
    sha = {"b.md": hashlib.sha256((d / "b.md").read_bytes()).hexdigest()}
    params = {"umbrella": "b.md", "umbrella_content": doc.replace("body", "merged"), "desc": "d",
              "absorbed": [], "merge_id": "later", "src_sha": sha, "settle_days": 7}
    six = time.time() - 6 * 86400
    os.utime(prior, (six, six))
    ok, why = g._apply_merge(params)
    assert not ok and "settles after 7 days" in why and prior.is_dir(), why
    eight = time.time() - 8 * 86400
    os.utime(prior, (eight, eight))
    ok, why = g._apply_merge(params)
    assert ok and "settled earlier merge(s) ['merge-earlier']" in why, why
    assert not prior.exists() and len(list((d / ".trash").glob("merge-earlier-*"))) == 1
    assert (g._dir("undone") / "u1.json").is_file() and (g._dir("undo") / "u2.json").is_file(), \
        "only the settled merge's undo record closes"
    assert "merged" in (d / "b.md").read_text() and (d / ".quarantine" / "merge-later" / "b.md.orig").is_file()
    shutil.rmtree(d)
    print("ok — an earlier merge settles after settle_days; a younger one still blocks")


def test_descriptions_with_a_colon_survive_every_document_builder():
    """A description holding ": " (real ones do: "Fake OTC (live on prod): a Mesa ...") used to be
    written as a plain YAML scalar, which is a ScannerError: curate held the umbrella on every pass
    and harvest/graduation silently lost the name and description."""
    import yaml
    desc = 'Fake OTC (live on prod 2026-11-15): a Mesa-OTC order, priced "as is" #1'
    g = _gate()
    m = _curate(tempfile.mkdtemp())
    doc = m._umbrella_doc("project_x", desc, "project", "## Summary\nbody\n")
    ok, why = g.validate_memory_doc(doc)
    assert ok, f"umbrella with a colon in the description rejected: {why}"
    assert yaml.safe_load(doc.split("---")[1])["description"] == desc, "description changed in transit"

    sys.modules.setdefault("memory_ai", types.ModuleType("memory_ai"))
    d = Path(tempfile.mkdtemp())
    mh = _load("memory_harvest", ROOT / "bin" / "memory_harvest.py")
    mh.STAGING = d
    staged = mh.write_candidate({"name": "x", "type": "project", "description": desc, "body": "b"},
                                "user-direct", "sess", set())
    fm = yaml.safe_load(staged.read_text().split("---")[1])
    assert fm["description"] == desc and fm["name"] == staged.stem, f"harvest candidate unparseable: {fm}"

    sa = _load("memory_stage_apply", ROOT / "bin" / "memory_stage_apply.py")
    content = sa.canonical_content({"name": "x", "description": desc, "metadata": {"type": "project"}},
                                   "body", "user-direct", "sess")
    meta, _ = sa.parse_frontmatter(content)
    assert meta.get("description") == desc and meta.get("name") == "x", f"graduated memory lost its metadata: {meta}"
    shutil.rmtree(d)
    print("ok — a colon in the description round-trips through curate, harvest and graduation")


# ── the compress decision, through main() ──────────────────────────────────
def _run_compress(report, autonomous=True, body_a="alpha holds one fact", more=None, cap=None,
                  umbrella_body="umbrella body\n", store=None, dry=False, verdict="APPROVE", ac_extra=None,
                  apply_result=(True, "ok")):
    """Drive main() through compress merges, every outside dependency stubbed: the vector store pairs
    project_a/project_b (plus `more` pairs: {name: body}), distill returns `report`, the reviewer
    approves. engram_secrets is the real one (its regexes decide the credential and value holds).
    Returns the calls and everything main() printed."""
    import contextlib, io
    d = store or Path(tempfile.mkdtemp())      # a given store is reused (and kept) across runs
    fm = "---\nname: {n}\ndescription: d\nmetadata:\n  type: project\n---\n\n{b}\n"
    bodies = {"project_a.md": body_a, "project_b.md": "beta holds another fact", **(more or {})}
    for f, b in bodies.items():
        if not (d / f).exists():
            (d / f).write_text(fm.format(n=f[:-3], b=b))
    names = sorted(bodies)
    pairs = [(0.99, names[i], names[i + 1]) for i in range(0, len(names) - 1, 2)]
    calls = {"apply": 0, "verify": 0, "distill": 0, "notify": [], "propose": []}
    gate = types.ModuleType("engram_telegram_gate")
    gate.validate_memory_doc = lambda doc: (True, "")
    gate.unwrap_code_fence = lambda text: text
    gate._apply_merge = lambda p: (calls.__setitem__("apply", calls["apply"] + 1), apply_result)[1]
    gate.notify_undo = lambda op, p, text: calls["notify"].append(text)
    gate.propose = lambda op, p, text, files=None, verdict=None: calls["propose"].append(text)
    store = types.ModuleType("vector_store")
    store.EngramVectorStore = lambda cfg: types.SimpleNamespace(
        ensure_collection=lambda: None, find_duplicates=lambda threshold, max_pairs: pairs)
    mdv = types.ModuleType("memory_distill_verified")

    def distill(key, files, cfg=None, preserve_sources=None):
        calls["distill"] += 1
        return umbrella_body, report
    mdv.distill_cluster = distill
    ac = {"enabled": True, "autonomous": autonomous, "review_gate": "self", "prune_orphans": False,
          **(ac_extra or {})}
    if cap:
        ac["max_merges_per_run"] = cap
    mai = types.ModuleType("memory_ai"); mai.load = lambda: {"auto_curate": ac}
    sec = _load("engram_secrets", ROOT / "bin" / "engram_secrets.py")
    stubs = {"vector_store": store, "qdrant_client": types.ModuleType("qdrant_client")}
    saved = {k: sys.modules.get(k) for k in stubs}
    sys.modules.update(stubs)
    argv, sys.argv = sys.argv, ["memory_auto_curate.py"] + ([] if dry else ["--apply"])
    out = io.StringIO()
    try:
        m = _load("memory_auto_curate", ROOT / "bin" / "memory_auto_curate.py", memory_ai=mai,
                  engram_secrets=sec, memory_distill_verified=mdv, engram_telegram_gate=gate)
        m.MEM = d

        def verify(doc, files):
            calls["verify"] += 1
            return verdict
        m._self_verdict = verify
        with contextlib.redirect_stdout(out):
            m.main()
    finally:
        sys.argv = argv
        for k, v in saved.items():
            sys.modules.pop(k) if v is None else sys.modules.__setitem__(k, v)
        if store is None:
            shutil.rmtree(d)
    calls["out"] = out.getvalue()
    return calls


BASE = {"hard_facts": 40, "final_fact_coverage": 1.0, "sentence_coverage": 0.0, "draft_chars": 1000,
        "appendix_carried_chars": 0}


def test_a_bloated_merge_never_applies_and_the_message_says_what_was_kept():
    """Measured 2026-11-26: two merges applied as "near-lossless" with a prose score of 0.0 while
    commands, flags and refs were gone. A merge whose rewrite left more than half the umbrella to
    the appendix must not apply on its own, and the notice must report the facts actually kept."""
    bloated = {**BASE, "appendix_facts": 30, "appendix_chars": 650, "appendix_fresh_chars": 600}
    calls = _run_compress(bloated)
    assert calls["apply"] == 0 and not calls["propose"], f"bloated merge applied or queued: {calls}"
    assert calls["verify"] == 0, "a merge that cannot apply still paid for a review call"
    calls = _run_compress(bloated, autonomous=False)
    assert calls["apply"] == 0 and len(calls["propose"]) == 1 and "restated 60%" in calls["propose"][0], calls
    # lines carried from an earlier merge's appendix (never shown to the model) are out of BOTH sides
    # of the share: in the numerator they would block re-merging a merged umbrella forever...
    calls = _run_compress({**BASE, "appendix_facts": 30, "appendix_chars": 900,
                           "appendix_fresh_chars": 50, "appendix_carried_chars": 800})
    assert calls["apply"] == 1, f"carried-over appendix lines blocked the merge: {calls}"
    # ...and in the denominator they dilute a real loss: 150 fresh of the 200 not carried is 75%
    calls = _run_compress({**BASE, "appendix_facts": 30, "appendix_chars": 980,
                           "appendix_fresh_chars": 150, "appendix_carried_chars": 800})
    assert calls["apply"] == 0, f"carried lines diluted a bloated rewrite into applying: {calls}"
    calls = _run_compress({**BASE, "appendix_facts": 3, "appendix_chars": 150, "appendix_fresh_chars": 100})
    assert calls["apply"] == 1 and calls["verify"] == 1, calls
    note = calls["notify"][0]
    assert "near-lossless" not in note and "40/40 concrete facts kept, 3 only in the appendix (10%" in note, note
    print("ok — a bloated merge never applies on its own; the notice reports the facts kept")


def test_public_values_reach_the_umbrella_or_the_merge_is_held():
    """The distiller swaps txids, addresses and full hashes for stand-ins and restores them, so a
    cluster that holds one merges (it used to be held forever). Curate still checks the result: an
    umbrella that lost a value is held, without printing it. A session id is not a value."""
    report = {**BASE, "appendix_facts": 0, "appendix_chars": 0, "appendix_fresh_chars": 0}
    txid = "9f2c" * 16
    calls = _run_compress(report, body_a=f"refund sent, txid {txid}", umbrella_body=f"refund txid {txid}\n")
    assert calls["apply"] == 1, f"a merge whose umbrella keeps the txid was held: {calls}"
    for autonomous in (True, False):
        calls = _run_compress(report, autonomous, body_a=f"refund sent, txid {txid}")
        assert (calls["apply"], calls["verify"], calls["propose"]) == (0, 0, []), calls
        assert "lost 1 public value" in calls["out"] and txid not in calls["out"], calls["out"]
    addr = "0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed"
    calls = _run_compress(report, body_a=f"refund sent to {addr}")
    assert calls["apply"] == 0, f"an umbrella that lost an address was applied: {calls}"
    calls = _run_compress(report, body_a="originSessionId: 5f1e2d3c-4b5a-6978-8a9b-0c1d2e3f4a5b\nplain fact")
    assert calls["apply"] == 1, f"a session id held the merge: {calls}"
    print("ok — public values reach the umbrella; one that lost a value is held, unprinted")


def test_held_clusters_take_no_slot():
    """A cluster held from its sources (a credential) costs no distill and no slot: with a cap of 1
    the clean cluster after it still merges, and two clean clusters still get only one attempt."""
    report = {**BASE, "appendix_facts": 0, "appendix_chars": 0, "appendix_fresh_chars": 0}
    token = "ghp_" + "A1b2C3d4E5f6G7h8I9j0" * 2
    calls = _run_compress(report, body_a=f"GITHUB_TOKEN={token}", cap=1,
                          more={"project_c.md": "gamma fact", "project_d.md": "delta fact"})
    assert calls["distill"] == 1 and calls["apply"] == 1, f"a held cluster used up the cap: {calls}"
    calls = _run_compress(report, cap=1, more={"project_c.md": "gamma fact", "project_d.md": "delta fact"})
    assert calls["distill"] == 1 and calls["apply"] == 1, f"the cap no longer bounds merge attempts: {calls}"
    print("ok — a held cluster takes no slot; the cap still bounds attempts")


def test_a_credential_hold_never_prints_the_credential():
    """The credential HOLD printed the first 50 chars of each credential, i.e. the whole of a GitHub
    or AWS token, into the daemon log (and the transcript when the memory-tidy skill runs curate)."""
    token = "ghp_" + "A1b2C3d4E5f6G7h8I9j0" * 2
    calls = _run_compress({**BASE, "appendix_facts": 0, "appendix_chars": 0, "appendix_fresh_chars": 0},
                          body_a=f"deploy key GITHUB_TOKEN={token}")
    assert calls["apply"] == 0 and calls["distill"] == 0, calls
    assert "HOLD" in calls["out"] and token[:12] not in calls["out"], calls["out"]
    print("ok — a credential hold reports a count, never the value")


def test_a_skipped_cluster_is_not_distilled_again_until_it_changes():
    """A cluster skipped after a distill call (here: bloated) is not distilled again on the next
    run; an edit to one of its files, or retry_days passing, brings it back."""
    import json, time
    bloated = {**BASE, "appendix_facts": 30, "appendix_chars": 650, "appendix_fresh_chars": 600}
    d = Path(tempfile.mkdtemp())
    first = _run_compress(bloated, store=d)
    assert first["distill"] == 1 and "SKIP (this rewrite restated 60%" in first["out"], first["out"]
    again = _run_compress(bloated, store=d)
    assert again["distill"] == 0 and "SKIP (unchanged since: this rewrite restated" in again["out"], again["out"]
    (d / "project_b.md").write_text((d / "project_b.md").read_text() + "\nnew fact\n")
    edited = _run_compress(bloated, store=d)
    assert edited["distill"] == 1, "an edited cluster was not retried"
    skips = json.loads((d / ".curate_skips.json").read_text())
    for v in skips.values():
        v["at"] = int(time.time()) - 8 * 86400
    (d / ".curate_skips.json").write_text(json.dumps(skips))
    aged = _run_compress(bloated, store=d)
    assert aged["distill"] == 1, "a skip older than retry_days was not retried"
    shutil.rmtree(d)
    # a dry run writes nothing, not even a remembered skip (this hold fires before the dry-run exit)
    d = Path(tempfile.mkdtemp())
    report = {**BASE, "appendix_facts": 0, "appendix_chars": 0, "appendix_fresh_chars": 0}
    dry = _run_compress(report, body_a=f"txid {'9f2c' * 16}", store=d, dry=True)
    assert "lost 1 public value" in dry["out"] and not (d / ".curate_skips.json").exists(), dry["out"]
    shutil.rmtree(d)
    print("ok — a skipped cluster is not distilled again until it changes or retry_days pass")


def test_review_round_holds_and_skip_rules():
    """Independent review 2026-11-26: a value that cannot be told apart from a secret holds BEFORE
    distill; an umbrella the writer would block is held, not attempted; a down reviewer is not
    remembered as an answer; a changed setting retries; a skip expires exactly retry_days later."""
    import json, time
    report = {**BASE, "appendix_facts": 0, "appendix_chars": 0, "appendix_fresh_chars": 0}
    pk = "4c0883a69102937d6231471b5dbb6204fe5129617082792ae468d01a3f362318"
    calls = _run_compress(report, body_a=f"deployer 0x{pk}")
    assert calls["distill"] == 0 and "cannot be told apart from a secret" in calls["out"], calls["out"]
    assert pk not in calls["out"]
    calls = _run_compress(report, umbrella_body="umbrella " + "A1" * 45 + "\n")      # case-insensitive, like grep -iE
    assert calls["apply"] == 0 and "secret guard would block" in calls["out"], calls["out"]
    # an empty model reply is not a judgment: no apply, no queue, not remembered
    d0 = Path(tempfile.mkdtemp())
    empty = _run_compress({**report, "gen_empty": True}, store=d0, autonomous=False)
    assert empty["apply"] == 0 and not empty["propose"] and "returned nothing" in empty["out"], empty
    assert not (d0 / ".curate_skips.json").exists(), "an empty reply was remembered"
    shutil.rmtree(d0)

    # a DEFER verdict (reviewer down) is not remembered: the next run distills again
    d = Path(tempfile.mkdtemp())
    deferred = _run_compress(report, store=d, verdict="DEFER-RuntimeError")
    assert "SKIP (self-review DEFER-RuntimeError)" in deferred["out"]
    assert not (d / ".curate_skips.json").exists(), "a down reviewer was remembered as an answer"
    # a failed write says nothing about the cluster; waiting for an earlier merge to settle does
    failed = _run_compress(report, store=d, apply_result=(False, "umbrella write failed: push — rolled back"))
    assert failed["apply"] == 1 and not (d / ".curate_skips.json").exists(), "a write failure was remembered"
    waiting = _run_compress(report, store=d, apply_result=(False, "b.md is referenced by an outstanding merge backup"))
    assert waiting["apply"] == 1 and (d / ".curate_skips.json").exists(), "an outstanding backup was not remembered"
    (d / ".curate_skips.json").unlink()
    # a REJECT is remembered; changing a decisive setting retries it
    _run_compress(report, store=d, verdict="REJECT")
    assert _run_compress(report, store=d, verdict="REJECT")["distill"] == 0
    assert _run_compress(report, store=d, verdict="REJECT", ac_extra={"max_appendix_ratio": 0.6})["distill"] == 1
    # the window: a skip recorded exactly retry_days before this run's start has expired
    skips = json.loads((d / ".curate_skips.json").read_text())
    for v in skips.values():
        v["at"] = int(time.time()) - 7 * 86400
    (d / ".curate_skips.json").write_text(json.dumps(skips))
    assert _run_compress(report, store=d, verdict="REJECT", ac_extra={"max_appendix_ratio": 0.6})["distill"] == 1
    shutil.rmtree(d)
    print("ok — review round: holds before distill, writer guard, deterministic-only skips, config and age retry")


def test_settle_waits_for_a_successful_write():
    """A failed umbrella write must leave the earlier merge's backup and its UNDO record in place."""
    import hashlib, json, time
    g = _gate(); d = Path(tempfile.mkdtemp()); g.MEM = d; g.Q = d / ".approvals"; g.HOME = d
    (d / ".claude").mkdir()
    save = d / ".claude" / "save_memory.sh"
    save.write_text("#!/bin/sh\necho 'push failed' >&2\nexit 1\n"); save.chmod(0o755)
    doc = "---\nname: b\ndescription: d\n---\n\nbody\n"
    (d / "b.md").write_text(doc)
    prior = d / ".quarantine" / "merge-earlier"; prior.mkdir(parents=True)
    (prior / "b.md.orig").write_text("b before")
    (g._dir("undo") / "u1.json").write_text(json.dumps(
        {"id": "u1", "op": "merge_undo", "params": {"umbrella": "b.md", "absorbed": [], "merge_id": "earlier"}}))
    old = time.time() - 8 * 86400
    os.utime(prior, (old, old))
    sha = {"b.md": hashlib.sha256(doc.encode()).hexdigest()}
    ok, why = g._apply_merge({"umbrella": "b.md", "umbrella_content": doc.replace("body", "merged"), "desc": "d",
                              "absorbed": [], "merge_id": "later", "src_sha": sha, "settle_days": 7})
    assert not ok and "write failed" in why, why
    assert prior.is_dir() and (g._dir("undo") / "u1.json").is_file(), "a failed write settled the earlier merge"
    # a settle that fails AFTER a good write is reported; the merge itself stands
    save.write_text(f'#!/bin/sh\ncat > "{d}/$1"\n')

    def broken(_d):
        raise OSError("disk full")
    g._settle = broken
    ok, why = g._apply_merge({"umbrella": "b.md", "umbrella_content": doc.replace("body", "merged"), "desc": "d",
                              "absorbed": [], "merge_id": "later", "src_sha": sha, "settle_days": 7})
    assert ok and "WARNING could not settle" in why and "disk full" in why, why
    shutil.rmtree(d)
    print("ok — settle happens only after the umbrella write succeeded")


if __name__ == "__main__":
    test_validator_rejects_the_real_malformed_shapes()
    test_unwrap_fence_leaves_inner_code_blocks_alone()
    test_supersede_only_on_identical_copies()
    test_clusters_require_a_clique()
    test_fm_is_anchored()
    test_apply_merge_refuses_stale_sources()
    test_apply_merge_refuses_to_reuse_a_transaction_dir()
    test_apply_merge_rejects_a_file_in_an_outstanding_transaction()
    test_an_earlier_merge_settles_after_settle_days()
    test_descriptions_with_a_colon_survive_every_document_builder()
    test_a_bloated_merge_never_applies_and_the_message_says_what_was_kept()
    test_public_values_reach_the_umbrella_or_the_merge_is_held()
    test_held_clusters_take_no_slot()
    test_a_credential_hold_never_prints_the_credential()
    test_a_skipped_cluster_is_not_distilled_again_until_it_changes()
    test_review_round_holds_and_skip_rules()
    test_settle_waits_for_a_successful_write()
