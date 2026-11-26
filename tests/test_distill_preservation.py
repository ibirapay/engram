#!/usr/bin/env python3
"""C8 regression: final_fact_coverage (regex token classes) can read ~1.0 while a
whole PROSE fact was dropped. sentence_coverage must expose that, and
preserve_sources must make the merge lossless. Guards against coverage-gated
destructive merges."""
import sys, types, importlib.util, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
mai = types.ModuleType("memory_ai"); mai.load = lambda: {}
sys.modules["memory_ai"] = mai
spec = importlib.util.spec_from_file_location("mdv", ROOT / "bin" / "memory_distill_verified.py")
mdv = importlib.util.module_from_spec(spec); spec.loader.exec_module(mdv)


def main():
    tmp = ROOT / "tests" / "_tmp_c8"; tmp.mkdir(exist_ok=True); mdv.MEM = tmp
    (tmp / "project_a.md").write_text(
        "---\nname: a\n---\nThe rebalancer must never fill dust below one thousand "
        "reais because fees eat the spread.\n")
    (tmp / "project_b.md").write_text("---\nname: b\n---\nDeploy on port 9000.\n")
    # LLM drops the prose fact, keeps only the tokened one
    mdv.ollama_stream = lambda p, c: ("Umbrella: deploy on port 9000.", "stop", 0.0)

    _, rep = mdv.distill_cluster("project", ["project_a.md", "project_b.md"])
    assert rep["final_fact_coverage"] >= 0.99, "token coverage should look perfect"
    assert rep["sentence_coverage"] < 1.0, "sentence_coverage must expose dropped prose"

    final2, rep2 = mdv.distill_cluster("project", ["project_a.md", "project_b.md"],
                                       preserve_sources=True)
    assert "never fill dust below one thousand reais" in final2, "preserve mode must be lossless"
    assert rep2["sources_preserved"] is True
    shutil.rmtree(tmp)
    print("ok — C8 distill preservation (token cov hid loss; sentence cov caught it; preserve lossless)")


def concrete_tokens():
    """Tokens outside the old regex classes (commands, flags, bare refs, paths under any root,
    "porta N") used to vanish when the model rewrote the prose, while the merge still read as
    near-lossless. The appendix must now carry each one verbatim."""
    tmp = ROOT / "tests" / "_tmp_tokens"; tmp.mkdir(exist_ok=True); mdv.MEM = tmp
    # ONE fact class per line: a restated line carries every token on it, so a token sharing a
    # line with another missing one would survive even if its own class stopped being extracted
    isolated = {
        "Built with `cargo build --locked --release --features liquid` here.": "cargo build --locked --release --features liquid",
        "The image was built at 4b96c4a last week.": "4b96c4a",
        "Blocks live in /daemon/liquidv1/blocks/ on that node.": "/daemon/liquidv1/blocks/",
        "Nightly copies go to ~/ops/backups/daily on main.": "~/ops/backups/daily",
        "From outside the pod it is porta 18444 as published.": "porta 18444",
        "Retry with `--keep` when it stalls.": "`--keep`",
    }
    (tmp / "reference_a.md").write_text(
        "---\nname: a\n---\n" + "\n".join(isolated) + "\n"
        "Pinned `6cad780be93835b962af735171fae872f9526503` for good.\n"
        "- Mount /srv/data/x, run `podman ps -a`, then `podman logs x`.\n"
        # over 80 chars in backticks is a sentence, kept as prose: never restated as a token
        "The app said `Depósito na rede tron só é executado pela Mesa OTC, ligue a Mesa OTC antes de tentar de novo`.\n")
    uuid = "550e8400-e29b-41d4-a716-44665544000f"
    # a long line (over 160 chars) is not restated whole: its tokens are listed one by one
    (tmp / "reference_b.md").write_text(
        f"---\nname: b\n---\nSee https://github.com/a/b/c, N/A, id {uuid}.\n"
        "Restore: `pg_restore --migrations /mnt/backups/premt-1` then check `DepositDto`, and only after "
        "the restore finished and the row counts matched the export, which took most of the night.\n")
    # the rewrite drops every token but says "keep": a separator-insensitive match must not
    # count the word as the `--keep` flag
    mdv.ollama_stream = lambda p, c: ("Umbrella: build electrs, keep the backup, check the DTO.", "stop", 0.0)

    final, rep = mdv.distill_cluster("reference", ["reference_a.md", "reference_b.md"])
    for tok in [*isolated.values(), "pg_restore --migrations /mnt/backups/premt-1", "DepositDto"]:
        assert tok in final, f"concrete token lost in a rewrite: {tok}"
    assert rep["final_fact_coverage"] == 1.0, rep
    appendix = final.split("### Hard facts", 1)[1]
    # a masked secret is not a fact to keep: its line holds nothing else, so it is not restated
    assert "Pinned" not in appendix, "the redaction placeholder was kept as a fact"
    # the number the curate gate reads is the size of the entries actually written (none carried)
    entries = [ln for ln in appendix.splitlines() if ln.startswith("- ")]
    assert rep["appendix_fresh_chars"] == sum(len(ln) + 1 for ln in entries), rep
    assert rep["appendix_carried_chars"] == 0, rep
    # not facts: a URL is not a path, "N/A" is not a path, a UUID piece is not a git ref
    toks = {d for d, _ in mdv.extract_facts((tmp / "reference_b.md").read_text()).values()}
    assert not any("github.com" in t or t.startswith("/a/") or t == "N/A" for t in toks), toks
    assert not any(t in uuid for t in toks), f"a piece of a UUID became a git ref: {toks}"
    assert "Mesa OTC" not in appendix, "a sentence in backticks was restated as a token"
    # a short source line is restated ONCE (bullet marker dropped), holding all three of its tokens
    assert appendix.count("podman ps -a") == 1, appendix
    assert "- Mount /srv/data/x" in appendix and "- - " not in appendix, appendix
    # one entry per token: the path inside the long line's restore command is not listed again
    assert appendix.count("/mnt/backups/premt-1") == 1, appendix
    assert 0 < rep["appendix_chars"] < rep["draft_chars"], rep
    shutil.rmtree(tmp)
    print("ok — commands, flags, refs and any-root paths survive a lossy rewrite verbatim")


def remerge_is_stable():
    """An umbrella merged again (its appendix is now part of a source) keeps every fact at the
    same size. Entries used to come back with one more "(from x.md)" per generation, or nested
    inside each other, and the appendix headers turned into caveats."""
    tmp = ROOT / "tests" / "_tmp_remerge"; tmp.mkdir(exist_ok=True); mdv.MEM = tmp
    (tmp / "project_a.md").write_text(
        "---\nname: a\n---\nBlocks live in /data/x/blocks/ there. Never run `rm -rf /data/x` here.\n\n"
        "Never restart the node during the initial sync.\n")
    (tmp / "project_b.md").write_text("---\nname: b\n---\nOther prose.\n")
    mdv.ollama_stream = lambda p, c: ("Umbrella prose without tokens.", "stop", 0.0)
    sizes = []
    for _ in range(3):
        final, _rep = mdv.distill_cluster("project", ["project_a.md", "project_b.md"])
        (tmp / "project_a.md").write_text("---\nname: a\n---\n" + final)   # merged again next time
        sizes.append(len(final))
    assert sizes[0] == sizes[1] == sizes[2], f"the umbrella grows on every re-merge: {sizes}"
    assert "/data/x/blocks/" in final and "rm -rf /data/x" in final, "a fact was lost on re-merge"
    # no stacked tag; the caveat inside the restated line is not written a second time; the
    # caveat on its own line is restated once, as a caveat
    assert ") (from " not in final, final
    assert final.count("Never run `rm -rf /data/x`") == 1 and final.count("Never restart the node") == 1, final
    # an old-format entry (token, tag, " — ", context) reads back as plain text; scaffolding drops
    old = ("x\n\n## Reference facts (verbatim — auto-preserved from sources)\n_Hard facts/caveats from x._\n\n"
           "### Hard facts\n- `/data/y` (path, from a.md) — Data in /data/y.\n<details><summary>s.md</summary>\n")
    assert mdv.prior_appendix_lines(old) == ["`/data/y` Data in /data/y."], mdv.prior_appendix_lines(old)

    # the merged umbrella meets a fresh note: its carried lines are restated again (the model never
    # sees them) but they are not THIS rewrite's loss, so the gate's share must not see them
    (tmp / "project_c.md").write_text("---\nname: c\n---\nA fresh note with nothing concrete.\n")
    final, rep = mdv.distill_cluster("project", ["project_a.md", "project_c.md"])
    assert "rm -rf /data/x" in final, "a carried fact was lost"
    assert rep["appendix_carried_chars"] > 0 and rep["appendix_fresh_chars"] == 0, rep
    shutil.rmtree(tmp)
    print("ok — merging an umbrella again keeps its facts at the same size, and they do not count as new loss")


def identifiers_fences_and_frontmatter():
    """Review findings 2026-11-26: an identifier inside a longer one is not the same fact; a command
    in a fenced block is code; a token the model put in its own frontmatter is not kept (curate
    keeps only the body); a source file name used as a subsection label is not a fact."""
    tmp = ROOT / "tests" / "_tmp_ident"; tmp.mkdir(exist_ok=True); mdv.MEM = tmp
    # over 200 chars, and not secret-shaped (a long alnum run would be masked before extraction)
    LONG = "printf '%s ' " + "word-" * 40
    assert len(LONG) > 200 and not mdv.looks_secret(LONG)
    (tmp / "reference_a.md").write_text(
        "---\nname: a\n---\nSet ENGRAM_URL to the gateway.\nENGRAM_URL_FALLBACK points at the local daemon.\n"
        "Rebuild with:\n```bash\ncargo build --locked --release --features liquid\n" + LONG + "\n```\n"
        "Restore with `pg_restore --clean` first.\n")
    (tmp / "reference_b.md").write_text("---\nname: b\n---\n### reference_c.md\nOther prose.\n")

    # the model keeps only the longer identifier: the shorter one's statement must still survive
    mdv.ollama_stream = lambda p, c: ("ENGRAM_URL_FALLBACK points at the local daemon.", "stop", 0.0)
    final, rep = mdv.distill_cluster("reference", ["reference_a.md", "reference_b.md"])
    assert "Set ENGRAM_URL to the gateway." in final, "ENGRAM_URL counted as present inside ENGRAM_URL_FALLBACK"
    # the model drops both: both lines stay (they are different statements, not one inside the other)
    mdv.ollama_stream = lambda p, c: ("Umbrella.", "stop", 0.0)
    final, rep = mdv.distill_cluster("reference", ["reference_a.md", "reference_b.md"])
    assert "Set ENGRAM_URL to the gateway." in final and "ENGRAM_URL_FALLBACK points" in final, final
    assert rep["final_fact_coverage"] == 1.0, rep
    # the fenced command is code, kept verbatim; an over-long fenced line is not a token
    assert "cargo build --locked --release --features liquid" in final, "fenced command lost"
    assert LONG not in final, "an over-long fenced line was restated as a token"
    # the subsection label naming a source file is not a fact to carry forward
    assert "reference_c.md" not in {d for d, _ in mdv.extract_facts((tmp / "reference_b.md").read_text()).values()}

    # a token the model wrote only into its own frontmatter does not count as kept
    mdv.ollama_stream = lambda p, c: (
        "---\nname: a\ndescription: run `pg_restore --clean` first\n---\nUmbrella body.", "stop", 0.0)
    final, rep = mdv.distill_cluster("reference", ["reference_a.md", "reference_b.md"])
    assert "Restore with `pg_restore --clean` first." in final.split("Reference facts", 1)[1], \
        "a token only in the model's frontmatter counted as kept"
    assert rep["final_fact_coverage"] == 1.0, rep
    shutil.rmtree(tmp)
    print("ok — identifiers by whole word, fenced commands, and the model's frontmatter does not count")


if __name__ == "__main__":
    main()
    concrete_tokens()
    remerge_is_stable()
    identifiers_fences_and_frontmatter()
