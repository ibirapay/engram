#!/usr/bin/env bash
# memory_audit.sh must treat MEMORY_FULL.md as part of the index: a store that
# keeps a hot MEMORY.md plus MEMORY_FULL.md had every cold memory reported as
# "not indexed" (663 of 781 on a real store), drowning the real findings.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fail(){ echo "FAIL test_memory_audit_full_index — $1"; exit 1; }
command -v jq >/dev/null 2>&1 || { echo "SKIP test_memory_audit_full_index — no jq"; exit 0; }

H="$(mktemp -d)"; trap 'rm -rf "$H"' EXIT
mkdir -p "$H/.claude/projects/-t/memory"
cp "$ROOT/bin/memory_lib.sh" "$H/.claude/"
D="$H/.claude/projects/-t/memory"
mem(){ printf -- '---\nname: %s\ndescription: d %s\nmetadata:\n  type: project\n---\n%s\n' "$1" "$1" "$(head -c 300 /dev/zero | tr '\0' x)" > "$D/$1.md"; }
mem hot; mem cold; mem lost
echo '- [hot](hot.md) — d'   > "$D/MEMORY.md"
echo '- [cold](cold.md) — d' > "$D/MEMORY_FULL.md"

out="$(HOME="$H" CLAUDE_MEMORY_SLUG=-t bash "$ROOT/bin/memory_audit.sh" --json)" || fail "audit exited non-zero"
missing="$(jq -r '.missing_index | join(",")' <<<"$out")"
[[ "$missing" == "lost.md" ]] || fail "missing_index should be exactly lost.md, got '$missing'"
jq -e '.malformed_frontmatter | length == 0' <<<"$out" >/dev/null || fail "MEMORY_FULL.md was audited as a memory"
# editing a cold memory must not duplicate it into the hot index
printf 'body\n' | HOME="$H" CLAUDE_MEMORY_SLUG=-t bash "$ROOT/bin/save_memory.sh" cold.md "d cold" >/dev/null 2>&1 \
  || fail "save_memory.sh failed on a cold memory"
grep -qF '](cold.md)' "$D/MEMORY.md" && fail "save_memory.sh copied a cold memory into MEMORY.md"
grep -qF '](cold.md)' "$D/MEMORY_FULL.md" || fail "save_memory.sh dropped the cold index line"

# deleting a cold memory must take its MEMORY_FULL.md line with it
HOME="$H" CLAUDE_MEMORY_SLUG=-t bash "$ROOT/bin/delete_memory.sh" cold.md >/dev/null 2>&1 \
  || fail "delete_memory.sh failed"
[[ -e "$D/cold.md" ]] && fail "delete_memory.sh left the file"
grep -qF '](cold.md)' "$D/MEMORY_FULL.md" && fail "delete_memory.sh left a dangling MEMORY_FULL.md line"
echo "PASS test_memory_audit_full_index"
