#!/usr/bin/env python3
"""engram_secrets.py — ONE shared secret detector/redactor for the memory system.

Used by everything on the Python side that could send memory text somewhere it
shouldn't: distillation (redact before the LLM prompt, which may be a REMOTE
backend), vector embedding (redact before the embedding provider, which may be an
off-box Ollama/LM Studio host), index generation (redact descriptions before they
land in MEMORY.md), and stage-apply dedup. The shell writers use the mirrored
high-precision block ERE in memory_lib.sh.

This pattern is intentionally AGGRESSIVE — it is used for REDACTION (masking a
stray hash/UUID costs nothing) and for HOLDING content off remote backends. The
shell *block* guard is separately tuned high-precision so it rarely false-blocks
a save. Both must cover the crypto material this fleet actually handles:
macaroons, BIP39 seed phrases, WIF/xprv keys.
"""
import re

SECRET_RE = re.compile(
    r"(?i)("
    # BIP39 / recovery phrases: keyword then 6+ space/comma-separated words —
    # listed FIRST so it masks the whole phrase, not just the first word.
    r"(?:mnemonic|seed[_-]?phrase|recovery[_-]?phrase)\s*[:=]\s*(?:[a-z]+[\s,]+){5,}[a-z]+"
    # named credential = VALUE (incl. crypto key material)
    r"|(?:client[_-]?secret|webhook[_-]?secret|api[_-]?key|apikey|password|passwd|secret|token|access[_-]?token|mnemonic|seed[_-]?phrase|recovery[_-]?phrase|private[_-]?key|priv[_-]?key|macaroon)\s*[:=]\s*['\"]?[^\s'\"]{6,}"
    r"|originSessionId\s*[:=]?\s*[0-9a-fA-F-]{8,}"
    r"|-----BEGIN[ A-Z]*PRIVATE KEY"
    r"|Bearer\s+[A-Za-z0-9._\-]{20,}"
    r"|xprv[a-zA-Z0-9]{20,}"                 # BIP32 extended private key
    r"|\b[5KL][1-9A-HJ-NP-Za-km-z]{50,51}\b" # WIF private key
    r"|AKIA[0-9A-Z]{16}"
    r"|gh[pousr]_[A-Za-z0-9]{20,}"
    r"|sk-[A-Za-z0-9_-]{20,}"
    r"|[0-9a-f]{32,}"                        # long hex (hashes, raw macaroons)
    r"|[A-Za-z0-9+/]{40,}={0,2}"             # long base64 (tokens, blobs)
    r")")


# HIGH-PRECISION credential detector, used by memory_auto_curate's credential
# RECONCILIATION (a merge is allowed iff every credential found in the sources still
# appears verbatim in the umbrella). Unlike SECRET_RE it must NOT fire on public data
# (txids, EVM addresses, UUIDs, long hex): a false hit only blocks a merge, but the old
# blanket scanner blocked every cluster that touched a txid. Group "v" is the value to
# keep verbatim; without it the whole match is the credential. Mirrors the shell block
# ERE in memory_lib.sh, plus: a name=value pair needs 16+ chars AND a digit.
CREDENTIAL_RE = re.compile(
    r"(?i:(?:mnemonic|seed[_-]?phrase|recovery[_-]?phrase)\s*[:=]\s*(?P<v>(?:[a-z]+[\s,]+){5,}[a-z]+))"
    r"|(?i:(?:client[_-]?secret|webhook[_-]?secret|api[_-]?key|apikey|password|passwd|secret|token|"
    r"access[_-]?token|private[_-]?key|priv[_-]?key|macaroon)\s*[:=]\s*['\"]?)"
    r"(?P<v2>(?=[^\s'\"]*\d)[^\s'\"]{16,})"
    r"|-----BEGIN[ A-Z]*PRIVATE KEY-----"
    r"|Bearer\s+[A-Za-z0-9._\-]{20,}"
    r"|xprv[a-zA-Z0-9]{20,}"
    r"|\b[5KL][1-9A-HJ-NP-Za-km-z]{50,51}\b"
    r"|AKIA[0-9A-Z]{16}"
    r"|gh[pousr]_[A-Za-z0-9]{20,}"
    r"|sk-[A-Za-z0-9_-]{20,}")


def credentials(text: str):
    """Credential strings in `text` that must survive a merge verbatim (deduped, in order)."""
    out = []
    for m in CREDENTIAL_RE.finditer(text or ""):
        c = m.group("v") or m.group("v2") or m.group(0)
        if c not in out:
            out.append(c)
    return out


# PUBLIC values: what SECRET_RE masks that is positively public data, by an ALLOWLIST of shapes
# (anything else masked stays masked). A denylist ("masked minus known credentials") restored a
# bare 64-hex private key or a base64 secret into a pushed memory (review 2026-11-26).
#   0x + exactly 40 hex                 an EVM address (a private key is 32 bytes, 64 hex)
#   bech32 with a chain-address prefix  bc1 / tb1 / bcrt1 / lq1 / tlq1 / ex1 / el1 / ert1 (nsec1 is a key: not listed)
#   64 hex labelled tx/txid/hash/sha256/digest/block, 40 hex labelled commit/sha/rev
#                                       and no secret word anywhere on the line
#   a non-hex run of lowercase, digits and at least one / (a path: random base64 is mixed case;
#                                       40+ chars of [a-z0-9/] by chance is ~1e-10)
# A value that is ALSO seen in any other context in the cluster (inside a credential, unlabelled,
# next to a secret word) is not public anywhere.
_SECRET_WORD_RE = re.compile(r"(?i)\b(priv(ate)?|secret|seed|mnemonic|macaroon|passw(or)?d|pk|wif|xprv|nsec|"
                             r"token|api[_ -]?key|auth)\b")
_TX_LABEL_RE = re.compile(r"(?i)\b(tx|txid|txhash|tx_hash|hash|sha-?256|digest|block)\b")
_COMMIT_LABEL_RE = re.compile(r"(?i)\b(commit|sha1?|rev(ision)?)\b")
_BECH32_RE = re.compile(r"(?:bc1|tb1|bcrt1|lq1|tlq1|ex1|el1|ert1)[ac-hj-np-z02-9]{25,}")
_HEX_RE = re.compile(r"[0-9a-fA-F]+")
_BLOB_RE = re.compile(r"[0-9a-fA-F]{32,}|[A-Za-z0-9+/]{40,}={0,2}")    # what SECRET_RE masks as a bare run


def _is_public(m) -> bool:
    v, text = m.group(0), m.string
    # the window is the value's line AND the one before: markdown wraps a sentence, so the label
    # ("refund tx") often ends the previous line. The secret-word check uses the same window.
    cur_start = text.rfind("\n", 0, m.start())
    line_start = text.rfind("\n", 0, max(cur_start, 0)) + 1 if cur_start > 0 else 0
    line_end = text.find("\n", m.end()); line_end = len(text) if line_end < 0 else line_end
    # a label belongs to the NEXT value only: start after any earlier masked run in the window
    label_start = line_start
    for prev in SECRET_RE.finditer(text, line_start, m.start()):
        label_start = prev.end()
    before, line = text[label_start:m.start()], text[line_start:line_end]
    # the run may carry its own 0x (SECRET_RE's base64 alternative takes it) or follow one
    prefixed = v[:2].lower() == "0x" or text[max(0, m.start() - 2):m.start()].lower() == "0x"
    h = v[2:] if v[:2].lower() == "0x" else v
    if _HEX_RE.fullmatch(h):                             # hex: only by length + label, never "identifier"
        if prefixed and len(h) == 40:
            return True                                  # an EVM address
        if _SECRET_WORD_RE.search(line):
            return False
        if len(h) == 64:
            return bool(_TX_LABEL_RE.search(before))     # tx/block hash, sha256 digest
        if len(h) == 40 and not prefixed:
            return bool(_COMMIT_LABEL_RE.search(before)) # git commit
        return False
    if _BECH32_RE.fullmatch(v):
        return True                                      # a chain address
    if _SECRET_WORD_RE.search(line):
        return False
    # a path: lowercase segments joined by / (a bare lowercase run is NOT enough: nsec1..., a
    # lowercase bech32 private key, is one)
    return "/" in v and bool(re.fullmatch(r"[a-z0-9/]+", v))


def classify_values(texts) -> tuple:
    """(public, held) over a whole cluster. public: bare masked runs on the allowlist in every
    context they appear in, which distillation may swap for stand-ins and restore. held: bare
    masked runs that are neither public nor a recognised credential (a merge would mask them, so
    curate holds it). Named secrets and credentials are in neither: they stay masked."""
    texts = [t or "" for t in texts]
    creds = [c for t in texts for c in credentials(t)]
    ok, bad = set(), set()
    for t in texts:
        for m in SECRET_RE.finditer(t):
            v = m.group(0)
            if not _BLOB_RE.fullmatch(v):
                continue                                   # a named secret: stays masked
            if any(v in c or c in v for c in creds) or not _is_public(m):
                bad.add(v)
            else:
                ok.add(v)
    public = ok - bad
    held = {v for v in bad if not any(v in c or c in v for c in creds)}
    return public, held


def public_values(text: str) -> set:
    return classify_values([text])[0]


def swap_values(text: str, public: set, stand_in) -> tuple:
    """One pass over SECRET_RE's match spans: a public value becomes stand_in(value), everything
    else the redaction marker. Never a global str.replace, which also rewrote a value's copy
    inside a credential span and left the credential unmasked. Returns (text, n_masked)."""
    n = [0]

    def sub(m):
        if m.group(0) in public:
            return stand_in(m.group(0))
        n[0] += 1
        return "«redacted-secret»"
    return SECRET_RE.sub(sub, text or ""), n[0]


def redact(text: str):
    """Mask secret-looking substrings. Returns (masked_text, n_masked)."""
    return SECRET_RE.subn("«redacted-secret»", text or "")


def looks_secret(line: str) -> bool:
    return bool(SECRET_RE.search(line or ""))


def _corpus():
    """The shared corpus, also used by the engram-secrets Rust crate. One file, two
    implementations — that is what keeps the Python and Rust detectors in lockstep
    instead of each growing its own idea of what a credential looks like."""
    import json
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "secret_samples.json"
    d = json.loads(p.read_text())
    return d["secret"], d["clean"]


if __name__ == "__main__":  # tiny self-check
    samples_secret, samples_clean = _corpus()
    for s in samples_secret:
        assert looks_secret(s), f"missed secret: {s}"
    for s in samples_clean:
        assert not looks_secret(s), f"false positive: {s}"
    assert redact("api_key=sk-proj-abcdefghijklmnopqrstuvwxyz1234")[1] == 1
    # credentials(): real secrets are found, with the VALUE as the thing to keep verbatim...
    assert credentials("api_key=sk-proj-abcdefghijklmnopqrstuvwxyz1234") == ["sk-proj-abcdefghijklmnopqrstuvwxyz1234"]
    assert credentials("mnemonic: abandon abandon abandon abandon abandon ability") == ["abandon abandon abandon abandon abandon ability"]
    for s in ("macaroon=0201036c6e6402eb01030a10", "AKIAIOSFODNN7EXAMPLE", "ghp_abcdefghijklmnopqrstuvwxyz0123",
              "Authorization: Bearer abcdefghijklmnopqrstuvwxyz0123", "-----BEGIN RSA PRIVATE KEY-----",
              "L1aW4aubDFB7yfras2S1mN3bqg9nwySY8nkoLmJebSLD5BWv3ENZ"):
        assert credentials(s), f"credentials() missed: {s}"
    # ...and public data / prose are NOT credentials (the reason this is not SECRET_RE).
    for s in ("tx 3f2a9c1b8d7e6f5a4b3c2d1e0f9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d3e2f1a",
              "0x71C7656EC7ab88b098defB751B7401B5f6d8976F",
              "originSessionId=9f8e7d6c-1234-4abc-9def-0123456789ab",
              "password: correct-horse-battery", "token = account",
              "rotate the api_key via the console, never in a memory"):
        assert credentials(s) == [], f"credentials() false positive: {s}"
    assert credentials("a ghp_abcdefghijklmnopqrstuvwxyz0123 b ghp_abcdefghijklmnopqrstuvwxyz0123") == ["ghp_abcdefghijklmnopqrstuvwxyz0123"]
    print(f"ok — engram_secrets self-check ({len(samples_secret)} secret, "
          f"{len(samples_clean)} clean samples)")
