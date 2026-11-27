#!/usr/bin/env python3
"""Which masked values distillation may restore: an ALLOWLIST of positively public shapes. A
denylist ("masked minus known credentials") restored bare private keys and base64 secrets into a
pushed memory (independent review 2026-11-26)."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bin"))
import engram_secrets as es  # noqa: E402

TX = "ab" * 32
PK = "4c0883a69102937d6231471b5dbb6204fe5129617082792ae468d01a3f362318"
ADDR = "0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed"


def main():
    corpus = json.loads((ROOT / "tests" / "fixtures" / "secret_samples.json").read_text())
    leaked = [s[:24] for s in corpus["secret"] if es.public_values(s)]
    assert not leaked, f"corpus secrets classified public: {leaked}"
    # the shared corpus, on the Python side (the Rust crate checks the same file): every secret is
    # masked, no clean sample is
    assert all(es.looks_secret(s) for s in corpus["secret"]), [s[:30] for s in corpus["secret"] if not es.looks_secret(s)]
    assert not any(es.looks_secret(s) for s in corpus["clean"]), [s[:30] for s in corpus["clean"] if es.looks_secret(s)]

    public = [f"sent to {ADDR}", f"to {ADDR.lower()}", f"payer `{ADDR}` paid",
              "address bc1qar0srrr7xfkvy5l643lydnw9re59gtzzwf5mdq", "liquid lq1qq2xvpcvfup5j8zscjq05u2wxxjcyewk7979f3mmz5l7uw5pqmx6xf5xy50hsn6vhkm5euwt72x878eq6zxx2z58hd7zrsg9qn",
              "path /home/user/src/engram/crates/engramgraph/src/lib", "see ~/ops/tools/engram/crates/engramgraph/srcx"]
    for t in public:
        assert es.public_values(t), f"not restored: {t!r}"
    # (text, the part that must never come back as public). A bare hex is never public, labelled
    # or not: a tx hash and a private key are the same 64 hex (three review passes broke labels).
    secret = [(f"refund tx {TX}", TX), (f"txid: `{TX}`", TX), (f"tx hash 0x{TX}", TX),
              ("commit 6cad780be93835b962af735171fae872f9526503", "6cad780b"),(f"deployer pk 0x{PK}", PK),                       # a secret word on the line
              (f"deployer 0x{PK}", PK),                          # 0x + 64 hex with no tx label
              (f'{{"private_key": "{PK}"}}', PK),
              ("aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "wJalrXUtnFEMI"),
              (f"nostr hex {TX}", TX),
              (f"tx {TX} and the private key is in vault", TX),
              ("nsec1vl029mgpspedva04g90vltkh6fvh240zqtv9k0t9af8935ke9laqsnlfe5", "nsec1"),
              ("QWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXo0NTY3ODkwREU=", "QWJj"),
              ("secret at /home/user/src/engram/crates/engramgraph/src/lib", "home/user"),
              # the label on the previous line belongs to the value right after it, not this one
              (f"commit 6cad780be93835b962af735171fae872f9526503 then\n{PK[:40]} unlabelled", PK[:40]),
              # review round 2: a label that is not adjacent, or a secret word anywhere in the paragraph
              (f"Webhook signing key (HMAC-SHA256): {PK}", PK),
              (f"The PRIVATE_KEY for the tx relayer is {PK}", PK),
              (f"Boltz reverse swap, claim tx pending\npreimage {PK}", PK),
              (f'{{"block": 840000, "signer_private_key": "{PK}"}}', PK),
              (f"**Hot wallet private key** (never share):\nsigns every refund tx\n{PK}", PK),
              (f"GitHub webhook HMAC-SHA1 key: {PK[:40]}", PK[:40]),
              (f"Refund tx sent\nin `0x{TX}`", TX),              # label on the line above: held, by design
              # a key inside a URL path is not a "path"
              (f"RPC: https://black-sly-frost.quiknode.pro/{PK[:40]}/", PK[:40]),
              (f"RPC https://rpc.ankr.com/eth/{PK}", PK),
              ("rpc https://api.example.com/key/abcdefghijklmnopqrstuvwxyzabcdefghij/x", "abcdefghij"),
              # one case per guard, each with no other guard to catch it
              (f"refund tx\n{TX}", TX),                                   # a label must be on the SAME line
              (f"hotWalletKey stored; refund tx {TX}", TX),                # camelCase is split before matching
              (f"The key for refund tx {TX}", TX),                         # "key" alone is a secret word
              ("see https://api.example.com/v1/abcdefgh/ijklmnop/qrstuvwx/yz0123456789abcd", "yz0123"),  # in a URL
              ("dir abcdefghijklmnopqrstuvwxyzabcd/xyz/files/x", "abcdefghij"),                # a 24+ segment
              ("dir store/4c0883a69102937d62/files/archive/xy", "4c0883"),                    # a hex segment
              # review round 3: a path inside a URL with a dotless host or no scheme; a zero-width
              # character inside a secret word; a secret word as a prefix (plural, compound, pt-BR)
              ("alerts go to http://iaserver/ntfy/k3xq9m2v7w8z4p5r6t2y8nq/send/x", "k3xq9m"),
              ("rpc.example.com/api/v1/k3xq9m2v7w8z4p5r6t2y8nq/eth/main/x", "k3xq9m"),
              ("k\u200bey dir /home/user/src/engram/crates/engramgraph/src/lib", "home/user"),
              ("tokens in /home/user/src/engram/crates/engramgraph/src/lib", "home/user"),
              ("chave em /home/user/src/engram/crates/engramgraph/src/lib", "home/user"),
              ("blob wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "wJalr"),   # mixed case with slashes: base64, not a path
              # review round 4: a capability path without its scheme or host
              ("alerts: curl -d up iaserver/ntfy/k3xq9m2v7w8z4p5r6t2y8nq/send", "k3xq9m"),
              ("POST /hooks/k3xq9m2v7w8z4p5r6t2y8nq/deploy/now", "k3xq9m"),
              # a seed written with slashes is not a path: a path's masked run starts with /
              ("backup words: abandon/ability/able/about/above/absent/absorb/abstract/absurd/abuse/access/accident", "abandon")]
    for text, part in secret:
        vals = es.public_values(text)
        assert not any(part in v for v in vals), f"restored: {text!r} -> {vals}"

    # a seed arm must not swallow the next label as its last word and leave that label's value in clear
    for sample in ["**Seed phrase:** abandon ability able about above absent absorb abstract absurd abuse access "
                   "accident\npassword: Hunter2Hunter2",
                   "seed words: zoo zoo zoo zoo zoo zoo zoo zoo zoo zoo zoo wrong token=abc123def456",
                   "mnemonic: abandon abandon abandon abandon abandon ability\npassword: Hunter2Hunter2"]:
        masked = es.redact(sample)[0]
        assert "Hunter2" not in masked and "abc123def456" not in masked, masked
    # credentials() returns the WORDS of a seed written the markdown way, like the plain form
    words = "abandon ability able about above absent absorb abstract absurd abuse access accident"
    assert es.credentials(f"**Seed phrase:** {words}") == [words], es.credentials(f"**Seed phrase:** {words}")
    # cluster level: public in one file, suspect in another -> public nowhere, and held
    path = "/home/user/src/engram/crates/engramgraph/src/lib"
    assert es.classify_values([f"see {path}"])[0] == {path}
    pub, held = es.classify_values([f"see {path}", f"secret at {path}"])
    assert path not in pub and path in held, (pub, held)
    # a value inside a credential anywhere in the cluster is never public (nor "held")
    pub, held = es.classify_values([f"refund tx {TX}", f"api_key={TX}9z"])
    assert TX not in pub, pub
    # the swap works on match spans: a public value's copy inside a credential span is masked
    text = f"refund tx {TX}\nsecret: {TX}"
    out, n = es.swap_values(text, {TX}, lambda v: "«v1»")
    assert out == "refund tx «v1»\n«redacted-secret»" and n == 1, out
    # a short named secret no credential covers is held (the merge would mask it away); the
    # originSessionId metadata a merge drops on purpose is not
    pub, held = es.classify_values(["db password: hunter2hunter"])
    assert not pub and held, (pub, held)
    pub, held = es.classify_values(["originSessionId: 5f1e2d3c-4b5a-6978-8a9b-0c1d2e3f4a5b"])
    assert not held, held
    print("ok — public values by allowlist: corpus secrets, bare keys and base64 never restored")


def writer_guard_parity():
    """WRITER_BLOCK_RE must block exactly what save_memory.sh's guard (grep -iE) blocks."""
    import re, subprocess
    corpus = json.loads((ROOT / "tests" / "fixtures" / "secret_samples.json").read_text())
    samples = corpus["secret"] + corpus["clean"] + [
        "A1" * 45, "a1" * 45, "token: 0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed", "plain prose",
        f"refund tx {TX}", "PASSWORD = Hunter2Hunter2"]
    samples += ["secret:\n    secretName: wallet-api-tls",           # \s must not reach across lines
                "Bearer\vabcdefghijklmnopqrstuvwxyz0123",           # \v does not split a line for grep
                "Bearer\u00a0abcdefghijklmnopqrstuvwxyz0123",       # a no-break space, per-line path
                "Seed phrase:\n```\n" + "\u00a0".join(["abandon", "ability", "able", "about", "above", "absent",
                 "absorb", "abstract", "absurd", "abuse", "access", "accident"]) + "\n```"]   # whole-file path
    import tempfile
    for s in samples:
        # the REAL guard: memory_lib.sh's memory_scan_secret (per line + the whole-file seed pass)
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as fh:
            fh.write(s + "\n")
        shell = subprocess.run(["bash", "-c", 'source "$1"; memory_scan_secret "$2" >/dev/null', "_",
                                str(ROOT / "bin" / "memory_lib.sh"), fh.name]).returncode == 0
        Path(fh.name).unlink()
        assert shell == es.writer_would_block(s), f"writer guard drift on {s[:40]!r}: shell={shell}"
        if re.search(r"(?i)mnemonic|seed|recovery", s) and s in corpus["secret"]:
            assert shell, f"the writer guard lets a seed phrase through: {s[:40]!r}"
        if s in corpus["clean"]:
            assert not shell, f"the writer guard blocks clean prose: {s[:40]!r}"
    # a stray non-UTF-8 byte on the line must not hide a hit ("binary file matches")
    with tempfile.NamedTemporaryFile("wb", suffix=".md", delete=False) as fh:
        fh.write(b"api_key=sk-live-abcdefghijklmnop1234 conta da loja \xe7\n")
    shell = subprocess.run(["bash", "-c", 'source "$1"; memory_scan_secret "$2" >/dev/null', "_",
                            str(ROOT / "bin" / "memory_lib.sh"), fh.name]).returncode == 0
    Path(fh.name).unlink()
    assert shell, "a Latin-1 byte on the line hid a credential from the writer guard"
    print("ok — the writer-guard pattern matches save_memory.sh's grep -iE on every sample")


if __name__ == "__main__":
    main()
    writer_guard_parity()
