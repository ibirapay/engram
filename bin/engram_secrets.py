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


# The shape of a value SECRET_RE masks that is PUBLIC data rather than a credential: a txid,
# an address, a full commit/file hash (long hex or base64-like runs with no name attached).
_PUBLIC_SHAPE_RE = re.compile(r"[0-9a-fA-F]{32,}|[A-Za-z0-9+/]{40,}={0,2}")


def public_values(text: str) -> set:
    """Long hex/base64 runs that redact() masks but that are not credentials (credentials(),
    session ids and named secrets are left out). Distillation swaps them for placeholders so the
    model never sees them and the umbrella gets them back verbatim."""
    creds = credentials(text)
    return {m.group(0) for m in SECRET_RE.finditer(text or "")
            if _PUBLIC_SHAPE_RE.fullmatch(m.group(0)) and not any(m.group(0) in c for c in creds)}


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
