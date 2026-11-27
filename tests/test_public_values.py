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
              f"Refund tx sent\nin `0x{TX}`"]                       # label wrapped onto the previous line
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
              (f"commit 6cad780be93835b962af735171fae872f9526503 then\n{PK[:40]} unlabelled", PK[:40])]
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
    print("ok — public values by allowlist: corpus secrets, bare keys and base64 never restored")


if __name__ == "__main__":
    main()
