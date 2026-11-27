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

    public = [f"refund tx {TX}", f"refund tx 0x{TX} on polygon", f"sent to {ADDR}", f"to {ADDR.lower()}",
              "commit 6cad780be93835b962af735171fae872f9526503",
              "address bc1qar0srrr7xfkvy5l643lydnw9re59gtzzwf5mdq", "path home/user/src/engram/crates/engramgraph/src/lib",
              f"txid: `{TX}`", f"block hash **{TX}**", f"tx hash 0x{TX}"]
    for t in public:
        assert es.public_values(t), f"not restored: {t!r}"
    # (text, the part that must never come back as public)
    secret = [(f"deployer pk 0x{PK}", PK),                       # a secret word on the line
              (f"deployer 0x{PK}", PK),                          # 0x + 64 hex with no tx label
              (f'{{"private_key": "{PK}"}}', PK),
              ("aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "wJalrXUtnFEMI"),
              (f"nostr hex {TX}", TX),
              (f"tx {TX} and the private key is in vault", TX),
              ("nsec1vl029mgpspedva04g90vltkh6fvh240zqtv9k0t9af8935ke9laqsnlfe5", "nsec1"),
              ("QWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXo0NTY3ODkwREU=", "QWJj"),
              ("secret at home/user/src/engram/crates/engramgraph/src/lib", "home/user"),
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
              ("dir store/4c0883a69102937d62/files/archive/xy", "4c0883")]                    # a hex segment
    for text, part in secret:
        vals = es.public_values(text)
        assert not any(part in v for v in vals), f"restored: {text!r} -> {vals}"

    # cluster level: public in one file, unlabelled in another -> not public, and held
    pub, held = es.classify_values([f"refund tx {TX}", f"see {TX}"])
    assert TX not in pub and TX in held, (pub, held)
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
    ere = re.search(r"^_MEMORY_SECRET_ERE='(.*)'$", (ROOT / "bin" / "memory_lib.sh").read_text(), re.M).group(1)
    corpus = json.loads((ROOT / "tests" / "fixtures" / "secret_samples.json").read_text())
    samples = corpus["secret"] + corpus["clean"] + [
        "A1" * 45, "a1" * 45, "token: 0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed", "plain prose",
        f"refund tx {TX}", "PASSWORD = Hunter2Hunter2"]
    for s in samples:
        shell = subprocess.run(["grep", "-qiE", "--", ere], input=s, text=True).returncode == 0
        assert shell == bool(es.WRITER_BLOCK_RE.search(s)), f"writer guard drift on {s[:40]!r}: shell={shell}"
    print("ok — the writer-guard pattern matches save_memory.sh's grep -iE on every sample")


if __name__ == "__main__":
    main()
    writer_guard_parity()
