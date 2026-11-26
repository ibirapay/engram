#!/usr/bin/env bash
# test_install_agents.sh — install.sh gives opencode and Hermes the same recall Claude Code has:
# MCP registration, the opencode prompt plugin, and the Hermes SOUL.md block. Drives the
# installer functions with fake CLIs in a sandboxed HOME (never touches a real ~/.hermes or opencode).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fails=0; ok() { echo "  ok   $*"; }; bad() { echo "  FAIL $*"; fails=$((fails+1)); }
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
unset XDG_CONFIG_HOME HERMES_HOME
export HOME="$T/home"; mkdir -p "$HOME" "$T/bin" "$HOME/.claude/rust"
printf '#!/bin/sh\n' > "$HOME/.claude/rust/engram-mcp"; chmod +x "$HOME/.claude/rust/engram-mcp"
cat > "$T/bin/opencode" <<'X'
#!/usr/bin/env bash
[[ "$1 $2" == "mcp add" ]] && { printf '%s\n' "$*" >> "$CALLS"; mkdir -p "$HOME/.config/opencode"; echo '{"mcp":{"engram":{}}}' > "$HOME/.config/opencode/opencode.json"; }
exit 0
X
cat > "$T/bin/hermes" <<'X'
#!/usr/bin/env bash
[[ "$1 $2" == "mcp list" ]] && exit 0
[[ "$1 $2" == "mcp add" ]] && printf '%s\n' "$*" >> "$CALLS"
exit 0
X
chmod +x "$T/bin/"*; export PATH="$T/bin:$PATH" CALLS="$T/calls"; : > "$CALLS"
say() { :; }; warn() { echo "WARN $*" >> "$T/warns"; }
CLAUDE="$HOME/.claude"; SLUG="-s"; WANT_HERMES=auto; WANT_OPENCODE=auto
for f in opencode_register hermes_soul hermes_register; do eval "$(sed -n "/^$f()/,/^}/p" "$REPO/install.sh")"; done

hermes_register engram "$CLAUDE/rust/engram-mcp" --slug "$SLUG"
grep -q -- "mcp add engram .*--env ENGRAM_CONFIG=$CLAUDE/engram.yaml .*--command $CLAUDE/rust/engram-mcp --args --slug -s\$" "$CALLS" \
  && ok "hermes: engram-mcp registered with config env, --args last" || bad "hermes registration: $(cat "$CALLS")"

opencode_register
grep -q -- "mcp add --global .*ENGRAM_CONFIG=.* engram -- $CLAUDE/rust/engram-mcp" "$CALLS" \
  && ok "opencode: MCP server registered" || bad "opencode registration: $(cat "$CALLS")"
cmp -s "$REPO/integrations/opencode/engram-recall.ts" "$HOME/.config/opencode/plugins/engram-recall.ts" \
  && ok "opencode: recall plugin installed" || bad "opencode plugin missing"
n=$(wc -l < "$CALLS"); opencode_register
[[ "$(wc -l < "$CALLS")" == "$n" ]] && ok "opencode: second run does not re-register" || bad "opencode re-registered"

hermes_soul; hermes_soul
[[ "$(grep -c 'engram:begin' "$HOME/.hermes/SOUL.md")" == 1 ]] && grep -q mcp__engram_memory_recall "$HOME/.hermes/SOUL.md" \
  && ok "hermes: SOUL.md block written once, idempotent" || bad "SOUL.md: $(cat "$HOME/.hermes/SOUL.md")"
printf 'my own rules\n' > "$HOME/.hermes/SOUL.md"; hermes_soul
head -1 "$HOME/.hermes/SOUL.md" | grep -q 'my own rules' && ok "hermes: existing SOUL.md content kept" || bad "SOUL.md content lost"
printf '## Shared memory (engram)\nhand written\n' > "$HOME/.hermes/SOUL.md"; hermes_soul
grep -q 'engram:begin' "$HOME/.hermes/SOUL.md" && bad "overwrote a hand-written engram section" || ok "hermes: hand-written engram section left alone"

WANT_OPENCODE=no; WANT_HERMES=no; rm -rf "$HOME/.config/opencode" "$HOME/.hermes"; : > "$CALLS"
opencode_register; hermes_soul; hermes_register engram x
[[ ! -s "$CALLS" && ! -e "$HOME/.config/opencode" && ! -e "$HOME/.hermes" ]] && ok "--no-opencode / --no-hermes do nothing" || bad "opt-out ignored"

echo "── test_install_agents: $([[ $fails -eq 0 ]] && echo PASS || echo "FAIL ($fails)") ──"; exit $((fails>0))
