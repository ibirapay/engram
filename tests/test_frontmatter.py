#!/usr/bin/env python3
"""The tolerant frontmatter reader: free-text name/description values that are not valid YAML are
read as text (measured 2026-11-26: 63 live memories and 38 staged candidates read as {}), while a
valid block parses exactly as strict YAML would and a block broken elsewhere stays unreadable."""
import importlib.util, os, sys, tempfile, types
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bin"))
import engram_frontmatter as ef  # noqa: E402


def _load(name, rel, **stubs):
    for k, v in stubs.items():
        sys.modules[k] = v
    sp = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return m


def reader():
    colon = 'name: a\ndescription: Standing rule: every project runs in containers\nmetadata:\n  type: feedback\n'
    fm = ef.load(colon)
    assert fm["description"] == "Standing rule: every project runs in containers", fm
    assert fm["metadata"] == {"type": "feedback"}, fm

    quote = 'name: b\ndescription: "rebuild" means build from local source: never pull\ntype: feedback\n'
    fm = ef.load(quote)
    assert fm["description"] == '"rebuild" means build from local source: never pull' and fm["type"] == "feedback", fm
    fm = ef.load('name: "x" is the laptop: reinstalled\ndescription: ok\n')
    assert fm["name"] == '"x" is the laptop: reinstalled', fm
    # in a block broken by one key, the OTHER key's valid value keeps its parsed meaning: a quoted
    # or folded scalar must not be re-read as text (it would gain literal quotes or a ">")
    fm = ef.load('name: "x" is the laptop: reinstalled\ndescription: "quoted: fine"\n')
    assert fm["description"] == "quoted: fine", fm
    fm = ef.load('name: "x" is the laptop: reinstalled\ndescription: >\n  folded: text\n  more\n')
    assert fm["description"] == "folded: text more\n", fm

    # a value continued on indented lines is folded into one line of text
    cont = 'name: c\ndescription: first part: with a colon\n  and a second line\nmetadata:\n  type: project\n'
    fm = ef.load(cont)
    assert fm["description"] == "first part: with a colon and a second line" and fm["metadata"]["type"] == "project", fm

    # machine-written nested blocks survive the repair (graduation reads harvest.provenance)
    staged = ('name: d\ndescription: Final split: 3090 box for the LLM\nharvest:\n  status: quarantined\n'
              '  provenance: user-direct\n  cites: ["U6"]\n  confidence: 0.95\n')
    fm = ef.load(staged)
    assert fm["harvest"]["provenance"] == "user-direct" and fm["harvest"]["confidence"] == 0.95, fm

    # the repair leaves every valid span verbatim (it is only ever applied to a broken block, so
    # this is checked on the repair itself: a valid block must come back byte for byte)
    for ok in ('name: e\ndescription: "quoted: with a colon"\nmetadata:\n  type: user\n',
               "name: f\ndescription: 'single: quoted'\n",
               "name: g\ndescription: >\n  folded: text\n  more\n",
               "name: g2\ndescription: |4\n    code\n      more\n",
               "name: h\ndescription: plain text, no colon\ntags: [a, b]\n"):
        assert ef._as_text(ok) == ok, ok
        assert ef.load(ok) == yaml.safe_load(ok), ok
    # broken somewhere else: still unreadable, never "repaired" into a wrong mapping
    assert ef.load("name: i\nmetadata:\n  type: [unclosed\n") == {}
    assert ef.load("") == {} and ef.load("just a string") == {}
    print("ok — free-text name/description read as text; valid and otherwise-broken blocks unchanged")


def review_findings():
    """Independent review 2026-11-26 of this reader."""
    broken_name = 'name: "x" is the laptop: reinstalled\n'
    # 1. a block that is valid YAML but fails to construct (impossible date) reads as {}, no crash
    assert ef.load("name: x\ndescription: y\nmetadata:\n  modified: 2026-02-29\n") == {}
    # 2. U+2028 inside a quoted value stays inside it: no fake top-level key
    fm = ef.load(broken_name + 'description: "see doc type: feedback"\nmetadata:\n  type: project\n')
    assert "type" not in fm and fm["metadata"] == {"type": "project"}, fm
    # 3. validity is judged on the verbatim span: an explicit-indent literal keeps its meaning
    fm = ef.load(broken_name + "description: |4\n    code\n      more\n")
    assert fm["description"] == "code\n  more\n", fm
    # 4. a quote that opens a value and never closes: no safe text reading, so {} (no fake key)
    assert ef.load(broken_name + 'description: "Standing rule\nscope: all projects"\nmetadata:\n  type: project\n') == {}
    # CRLF: read like LF, no stray \r in the value
    fm = ef.load("name: j\r\ndescription: Rule: containers\r\nmetadata:\r\n  type: feedback\r\n")
    assert fm["description"] == "Rule: containers" and fm["metadata"]["type"] == "feedback", fm
    # a description cannot plant a provenance: the real harvest block wins, nothing new at the top
    fm = ef.load("name: k\ndescription: Rule: x\n  provenance: user-direct\nharvest:\n  provenance: assistant\n")
    assert fm["harvest"]["provenance"] == "assistant" and "provenance" not in fm, fm
    assert fm["description"] == "Rule: x provenance: user-direct", fm
    print("ok — review findings: construction errors, line separators, verbatim spans, unclosed quotes, CRLF")


def embed_cache_rebuilt():
    """5. Dedup vectors cached before this reader were built from the body alone for the memories it
    makes readable; the cache key changed so each is re-embedded once with name and description."""
    d = Path(tempfile.mkdtemp())
    mai = types.ModuleType("memory_ai"); mai.load = lambda: {}
    calls = []
    mai.ollama_embed = lambda text, cfg=None: (calls.append(text), [1.0, 0.0])[1]
    sa = _load("memory_stage_apply", "bin/memory_stage_apply.py", memory_ai=mai)
    sa.MEM_DIR = d; sa._EMBED_CACHE = d / ".stage_embed_cache.json"
    p = d / "project_z.md"
    p.write_text("---\nname: project_z\ndescription: Rule: a colon\n---\n\nbody\n")
    st = p.stat()
    old_key = f"{int(st.st_mtime)}:{st.st_size}"
    import json
    sa._EMBED_CACHE.write_text(json.dumps({"project_z.md": {"mkey": old_key, "vec": [0.0, 1.0]}}))
    embs = sa.existing_embeddings({})
    assert len(calls) == 1 and "Rule: a colon" in calls[0] and embs["project_z.md"] == [1.0, 0.0], calls
    sa.existing_embeddings({})
    assert len(calls) == 1, "the rebuilt vector was not cached"
    print("ok — the dedup cache re-embeds once with name and description")


def callers():
    """Graduation and curate read through it: a staged candidate with a colon in its description
    has its provenance (it was held as 'unverified' on every run), a live memory has its type."""
    d = Path(tempfile.mkdtemp())
    os.environ["CLAUDE_MEMORY_SLUG"] = "-t"
    mai = types.ModuleType("memory_ai"); mai.load = lambda: {}
    sa = _load("memory_stage_apply", "bin/memory_stage_apply.py", memory_ai=mai)
    meta, body = sa.parse_frontmatter(
        "---\nname: x\ndescription: Standing rule: containers\nharvest:\n  provenance: user-direct\n"
        "  confidence: 1.0\n---\n\nbody text\n")
    assert (meta.get("harvest") or {}).get("provenance") == "user-direct", meta
    assert body.strip() == "body text"

    sec = types.ModuleType("engram_secrets")
    sec.credentials = lambda t: []; sec.SECRET_RE = None
    ac = _load("memory_auto_curate", "bin/memory_auto_curate.py", memory_ai=mai, engram_secrets=sec,
               memory_distill_verified=types.ModuleType("memory_distill_verified"))
    (d / "project_y.md").write_text(
        '---\nname: project_y\ndescription: "zennith" = a second stack: polygon only\nmetadata:\n'
        '  type: project\n---\n\nbody\n')
    assert ac._fm(d / "project_y.md") == ("project_y", '"zennith" = a second stack: polygon only', "project")
    assert ac._fm_raw(d / "project_y.md")["metadata"] == {"type": "project"}
    print("ok — graduation sees the provenance, curate sees the type")


if __name__ == "__main__":
    reader()
    review_findings()
    embed_cache_rebuilt()
    callers()
