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
    r"(?:mnemonic|seed[\s_-]?(?:phrase|words)|recovery[\s_-]?phrase)[*_]*(?:\s*\([^)\n]{0,40}\))?[*_]*\s*[:=][\s*_`'\x22]*(?:[a-z]{3,8}[\s,]+){5,}[a-z]{3,8}"
    # named credential = VALUE (incl. crypto key material)
    r"|(?:client[_-]?secret|webhook[_-]?secret|api[_-]?key|apikey|password|passwd|secret|token|access[_-]?token|mnemonic|seed[\s_-]?(?:phrase|words)|recovery[\s_-]?phrase|private[_-]?key|priv[_-]?key|macaroon)\s*[:=]\s*['\"]?[^\s'\"]{6,}"
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
    r"(?i:(?:mnemonic|seed[\s_-]?(?:phrase|words)|recovery[\s_-]?phrase)[*_]*(?:\s*\([^)\n]{0,40}\))?[*_]*\s*[:=][\s*_`'\x22]*(?P<v>(?:[a-z]{3,8}[\s,]+){5,}[a-z]{3,8}))"
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
# that CANNOT be a secret (anything else masked stays masked). Three independent review passes
# (2026-11-26) broke every rule that tried to tell a tx hash from a private key by its label:
# a 64-hex is a 64-hex. Measured on the author's store, label rules served 3 values; the shapes
# kept here serve the other ~100 (76 EVM addresses, 10 bech32 addresses, 15 paths).
#   0x + exactly 40 hex                 an EVM address (a private key is 32 bytes, 64 hex)
#   bech32 with a chain-address prefix  bc1 / tb1 / bcrt1 / lq1 / tlq1 / ex1 / el1 / ert1 (nsec1 is a key: not listed)
#   a filesystem path                   the run starts its word (not inside a URL or after a host),
#                                       lowercase, digits and /, every segment short and not hex,
#                                       and no secret word in its paragraph
# A value that is ALSO seen in any other context in the cluster is not public anywhere.
_SECRET_WORD_PREFIXES = (
    "key", "pk", "sk", "seed", "wif", "nsec", "xprv", "hmac", "preimage", "mnemonic", "macaroon",
    "apikey", "privkey", "passphrase", "cred", "priv", "secret", "passw", "pwd", "auth", "sign",
    "token", "otp", "jwt", "bearer", "cookie", "salt", "entropy", "blinder",
    "senha", "chave", "segredo", "semente")
_BECH32_RE = re.compile(r"(?:bc1|tb1|bcrt1|lq1|tlq1|ex1|el1|ert1)[ac-hj-np-z02-9]{25,}")
_HEX_RE = re.compile(r"[0-9a-fA-F]+")
_BLOB_RE = re.compile(r"[0-9a-fA-F]{32,}|[A-Za-z0-9+/]{40,}={0,2}")    # what SECRET_RE masks as a bare run


def _secret_word_near(text: str, start: int, end: int) -> bool:
    """A secret word anywhere in the paragraph around [start, end): normalised (NFKC, invisible
    characters dropped), names split on _ - and camelCase, every word matched as a prefix."""
    import unicodedata
    p0 = text.rfind("\n\n", 0, start); p0 = 0 if p0 < 0 else p0
    p1 = text.find("\n\n", end); p1 = len(text) if p1 < 0 else p1
    para = unicodedata.normalize("NFKC", text[p0:p1])
    para = "".join(c for c in para if unicodedata.category(c) != "Cf")
    para = re.sub(r"([a-z])([A-Z])", r"\1 \2", para).lower()
    return any(w.startswith(_SECRET_WORD_PREFIXES) for w in re.findall(r"[a-z0-9]+", para))


def _is_public(m) -> bool:
    v, text = m.group(0), m.string
    # the run may carry its own 0x (SECRET_RE's base64 alternative takes it) or follow one
    prefixed = v[:2].lower() == "0x" or text[max(0, m.start() - 2):m.start()].lower() == "0x"
    h = v[2:] if v[:2].lower() == "0x" else v
    if _HEX_RE.fullmatch(h):
        nxt = text[m.end():m.end() + 1]
        return prefixed and len(h) == 40 and not _HEX_RE.fullmatch(nxt or "g")    # an EVM address
    if _BECH32_RE.fullmatch(v):
        return True                                                               # a chain address
    # a filesystem path: the run must START its word (a URL's path follows a scheme or a host),
    # be lowercase with short non-hex segments, and have no secret word around it
    word_start = max(text.rfind(c, 0, m.start()) for c in " \t\n`'\"(<[") + 1
    if text[word_start:m.start()] not in ("", "~", "~/", ".", "./", "..", "../") or v.startswith("//"):
        return False
    if "/" not in v or not re.fullmatch(r"[a-z0-9/]+", v):
        return False
    # a long segment, a hex one, or a 12+ one mixing letters with 2+ digits (a random token: the
    # capability part of a webhook or ntfy URL written without its scheme)
    if any(len(seg) >= 24 or re.fullmatch(r"[0-9a-f]{16,}", seg)
           or re.fullmatch(r"(?=(?:.*\d){2})(?=.*[a-z])[a-z0-9]{12,}", seg) for seg in v.split("/")):
        return False
    return not _secret_word_near(text, m.start(), m.end())


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
                # a named secret: stays masked. One no credential covers (a short "password: x")
                # would be lost to the merge without curate's credential hold: count it as held.
                # originSessionId is frontmatter metadata a merge drops on purpose, not a secret.
                if not v.lower().startswith("originsessionid") and not any(c in v for c in creds):
                    bad.add(v)
                continue
            if any(v in c or c in v for c in creds) or not _is_public(m):
                bad.add(v)
            else:
                ok.add(v)
    public = ok - bad
    held = {v for v in bad if not any(v in c or c in v for c in creds)}
    return public, held


# memory_lib.sh _MEMORY_SECRET_ERE, the guard save_memory.sh applies (grep -iE): curate checks an
# umbrella against it before writing, instead of failing the write and retrying every run.
WRITER_BLOCK_RE = re.compile(
    r"((mnemonic|seed[\s_-]?(?:phrase|words)|recovery[\s_-]?phrase)[*_]*(\s*\([^)\n]{0,40}\))?[*_]*\s*[:=][\s*_`\x22]*([a-z]{3,8}[\s,]+){5,}[a-z]{3,8}"
    r"|(client[_-]?secret|webhook[_-]?secret|api[_-]?key|apikey|password|passwd|secret|token|access[_-]?token"
    r"|mnemonic|seed[\s_-]?(?:phrase|words)|recovery[\s_-]?phrase|private[_-]?key|priv[_-]?key|macaroon)\s*[:=]\s*.?[A-Za-z0-9_/+-]{8,}"
    r"|BEGIN\s+[A-Z ]*PRIVATE KEY|-----BEGIN\s|Bearer\s+[A-Za-z0-9._-]{20,}|xprv[a-zA-Z0-9]{20,}"
    r"|[5KL][1-9A-HJ-NP-Za-km-z]{50,51}|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}"
    r"|[0-9a-f]{80,})", re.I)


# memory_lib.sh _MEMORY_SEED_ERE: the seed arm alone, matched over the whole file (grep -z).
SEED_BLOCK_RE = re.compile(
    r"(mnemonic|seed[\s_-]?(?:phrase|words)|recovery[\s_-]?phrase)[*_]*(\s*\([^)]{0,40}\))?[*_]*\s*[:=]"
    r"[\s*_`\x22]*([a-z]{3,8}[\s,]+){5,}[a-z]{3,8}", re.I)


def writer_would_block(text: str) -> bool:
    """What save_memory.sh's guard blocks: its pattern line by line (grep splits on \\n only, so
    \\s must not reach across lines and \\v or \\f must not split one), plus the seed phrase over
    the whole text (grep -z)."""
    text = text or ""
    return any(WRITER_BLOCK_RE.search(ln) for ln in text.split("\n")) or bool(SEED_BLOCK_RE.search(text))


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
