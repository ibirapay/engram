#!/usr/bin/env python3
"""Memory frontmatter, read the way people and models actually write it.

A memory's `name:` and `description:` are free text, written by hand, by Claude, by the harvester
and by older engram versions, and often not valid YAML:

    description: Standing rule: all projects run in containers      (": " inside a plain scalar)
    description: "rebuild" means build from local source              (a quote that does not close it)

Strict YAML rejects the whole block, so a reader got {} and the memory silently lost its name,
description, type and harvest provenance: curate skipped it as untyped, and graduation held a
staged candidate as "provenance unverified" on every run, forever (measured 2026-11-26: 63 of 1120
live memories and 38 of 804 staged candidates). The Rust reader was already lenient, so recall saw
them fine; only the Python stages did not.

`load()` parses strictly first. Only when that fails does it retry with each top-level `name:` /
`description:` value that is not a valid scalar on its own read as plain text. A valid block is
never touched, and nothing is written back to disk.

Documents engram GENERATES (curate umbrellas) are still validated strictly by
engram_telegram_gate.validate_memory_doc: this is for reading what already exists.
"""
from __future__ import annotations

import json
import re

import yaml

FREE_TEXT_KEYS = ("name", "description")
_KEY_RE = re.compile(r"^(%s):[ \t]*(.*)$" % "|".join(FREE_TEXT_KEYS))


def _span_ok(span: str) -> bool:
    """The span (a key line and its continuation lines, verbatim) is valid YAML on its own."""
    try:
        return isinstance(yaml.safe_load(span), dict)
    except Exception:          # not only YAMLError: a bad date or int raises ValueError
        return False


def _as_text(block: str) -> str:
    """The block with every top-level free-text value that is not a valid scalar on its own (with
    its indented continuation lines) rewritten as one JSON string, a valid YAML scalar."""
    # split on \n only: splitlines() also breaks on U+2028, \v, \f..., which YAML keeps inside a
    # quoted value, and would turn the rest of such a value into a fake top-level key
    lines = block.replace("\r\n", "\n").split("\n")
    out, i = [], 0
    while i < len(lines):
        m = _KEY_RE.match(lines[i])
        if not m:
            out.append(lines[i]); i += 1
            continue
        key, parts, j = m.group(1), [m.group(2)], i + 1
        while j < len(lines) and lines[j][:1] in (" ", "\t") and lines[j].strip():
            parts.append(lines[j].strip()); j += 1
        value = " ".join(p for p in parts if p).strip()
        if value and not _span_ok("\n".join(lines[i:j])):
            # a quote that opens the value and never closes may continue on unindented lines
            # YAML would read as part of it: there is no safe text reading, give up
            if value[0] in "\"'" and value[0] not in value[1:]:
                raise ValueError("unclosed quoted value")
            out.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
        else:
            out.extend(lines[i:j])
        i = j
    return "\n".join(out)


def load(block: str) -> dict:
    """The frontmatter mapping of a block (the text between the --- fences), {} when absent or
    unreadable even as free text."""
    # Exception, not YAMLError: a well-formed block can still fail to construct (an impossible
    # date such as 2026-02-29 raises ValueError) and must read as {}, never crash the stage
    try:
        fm = yaml.safe_load(block or "")
    except Exception:
        try:
            fm = yaml.safe_load(_as_text(block))
        except Exception:
            return {}
    return fm if isinstance(fm, dict) else {}


if __name__ == "__main__":
    import sys
    from pathlib import Path
    fm_re = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n", re.S)
    # usage: engram_frontmatter.py FILE...  -> prints the files that still do not parse
    bad = 0
    for f in sys.argv[1:]:
        m = fm_re.match(Path(f).read_text(errors="ignore"))
        if m and not load(m.group(1)):
            bad += 1; print(f)
    print(f"{bad} of {len(sys.argv) - 1} unreadable", file=sys.stderr)
