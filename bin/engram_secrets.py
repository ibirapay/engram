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
# (anything else masked stays masked). Two independent reviews (2026-11-26) broke looser versions:
# a denylist restored bare private keys; a label anywhere in a two-line window let
# "signing key (HMAC-SHA256): <hex>" through, and a URL's API key passed as a "path". So:
#   0x + exactly 40 hex                 an EVM address (a private key is 32 bytes, 64 hex)
#   bech32 with a chain-address prefix  bc1 / tb1 / bcrt1 / lq1 / tlq1 / ex1 / el1 / ert1 (nsec1 is a key: not listed)
#   64 hex RIGHT AFTER tx / txid / tx hash / block hash / hash, 40 hex right after commit / rev
#                                       (same line, only punctuation between), AND no secret word
#                                       anywhere in the paragraph
#   a filesystem path: lowercase, digits and /, not inside a URL, every segment short and not hex,
#                                       no secret word in the paragraph
# A value that is ALSO seen in any other context in the cluster (inside a credential, unlabelled,
# near a secret word) is not public anywhere.
_SECRET_WORDS = {"key", "keys", "pk", "seed", "seeds", "wif", "nsec", "xprv", "hmac", "preimage",
                 "mnemonic", "macaroon", "apikey", "privkey", "passphrase", "cred", "creds"}
_SECRET_PREFIXES = ("priv", "secret", "passw", "auth", "credential", "sign")
_TX_ADJ_RE = re.compile(r"(?i)\b(tx|txid|tx[ _-]?(?:id|hash)|block[ _-]?hash|hash)\s*[:=#]?\s*[`'\"*(\[]*\s*(?:0x)?$")
_COMMIT_ADJ_RE = re.compile(r"(?i)\b(commit|rev(?:ision)?)\s*[:=#]?\s*[`'\"*(\[]*\s*$")
_BECH32_RE = re.compile(r"(?:bc1|tb1|bcrt1|lq1|tlq1|ex1|el1|ert1)[ac-hj-np-z02-9]{25,}")
_HEX_RE = re.compile(r"[0-9a-fA-F]+")
_BLOB_RE = re.compile(r"[0-9a-fA-F]{32,}|[A-Za-z0-9+/]{40,}={0,2}")    # what SECRET_RE masks as a bare run


def _secret_word_near(text: str, start: int, end: int) -> bool:
    """A secret word anywhere in the paragraph around [start, end): names split on _ - and
    camelCase first, so PRIVATE_KEY, signerPrivateKey and signing-key all count."""
    p0 = text.rfind("\n\n", 0, start); p0 = 0 if p0 < 0 else p0
    p1 = text.find("\n\n", end); p1 = len(text) if p1 < 0 else p1
    para = re.sub(r"([a-z])([A-Z])", r"\1 \2", text[p0:p1]).lower()
    return any(w in _SECRET_WORDS or w.startswith(_SECRET_PREFIXES) for w in re.findall(r"[a-z0-9]+", para))


def _is_public(m) -> bool:
    v, text = m.group(0), m.string
    line_start = text.rfind("\n", 0, m.start()) + 1
    before = text[line_start:m.start()]                  # same line only: a label must be adjacent
    # the run may carry its own 0x (SECRET_RE's base64 alternative takes it) or follow one
    prefixed = v[:2].lower() == "0x" or text[max(0, m.start() - 2):m.start()].lower() == "0x"
    h = v[2:] if v[:2].lower() == "0x" else v
    if _HEX_RE.fullmatch(h):                             # hex: only by length + adjacent label
        if prefixed and len(h) == 40:
            return True                                  # an EVM address
        if len(h) == 64:
            ok = bool(_TX_ADJ_RE.search(before))
        elif len(h) == 40 and not prefixed:
            ok = bool(_COMMIT_ADJ_RE.search(before))
        else:
            return False
        return ok and not _secret_word_near(text, m.start(), m.end())
    if _BECH32_RE.fullmatch(v):
        return True                                      # a chain address
    # a filesystem path: not inside a URL, short non-hex segments, no secret word around
    word_start = max(text.rfind(c, 0, m.start()) for c in " \t\n`'\"(<[") + 1
    if "://" in text[word_start:m.start()] or "/" not in v or not re.fullmatch(r"[a-z0-9/]+", v):
        return False
    if any(len(seg) >= 24 or re.fullmatch(r"[0-9a-f]{16,}", seg) for seg in v.split("/")):
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
    r"((mnemonic|seed[_-]?phrase|recovery[_-]?phrase)\s*[:=]\s*([a-z]+[\s,]+){5,}[a-z]+"
    r"|(client[_-]?secret|webhook[_-]?secret|api[_-]?key|apikey|password|passwd|secret|token|access[_-]?token"
    r"|mnemonic|seed[_-]?phrase|recovery[_-]?phrase|private[_-]?key|priv[_-]?key|macaroon)\s*[:=]\s*.?[A-Za-z0-9_/+-]{8,}"
    r"|BEGIN\s+[A-Z ]*PRIVATE KEY|-----BEGIN\s|Bearer\s+[A-Za-z0-9._-]{20,}|xprv[a-zA-Z0-9]{20,}"
    r"|[5KL][1-9A-HJ-NP-Za-km-z]{50,51}|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}"
    r"|[0-9a-f]{80,})", re.I)


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
