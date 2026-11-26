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
    assert "6cad780be93835b962af735171fae872f9526503" not in final
    assert "Pinned" not in appendix, "the redaction placeholder was kept as a fact"
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


if __name__ == "__main__":
    main()
    concrete_tokens()
