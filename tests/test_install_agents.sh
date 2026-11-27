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
for f in managed_block opencode_register hermes_soul hermes_register hermes_recall_hook; do eval "$(sed -n "/^$f()/,/^}/p" "$REPO/install.sh")"; done
# the engram skills and commands Claude Code has, for opencode to link
mkdir -p "$CLAUDE/skills/memory-tidy" "$CLAUDE/commands"; echo x > "$CLAUDE/skills/memory-tidy/SKILL.md"
echo x > "$CLAUDE/commands/memory-checkpoint.md"

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
OC="$HOME/.config/opencode"
[[ "$(readlink "$OC/skills/memory-tidy")" == "$CLAUDE/skills/memory-tidy" && "$(readlink "$OC/command/memory-checkpoint.md")" == "$CLAUDE/commands/memory-checkpoint.md" ]] \
  && ok "opencode: engram skills and commands linked" || bad "opencode skills/commands: $(ls -l "$OC/skills" "$OC/command" 2>&1)"
[[ "$(grep -c 'engram:begin' "$OC/AGENTS.md")" == 1 ]] && grep -q 'save_memory.sh' "$OC/AGENTS.md" \
  && ok "opencode: AGENTS.md engram block written once" || bad "AGENTS.md: $(cat "$OC/AGENTS.md")"
# another plugin already running engram-recall-hook (a user's own): no second recall plugin
rm -rf "$OC/plugins"; mkdir -p "$OC/plugins"; echo 'run("engram-recall-hook")' > "$OC/plugins/claude-hooks.ts"
opencode_register
[[ ! -e "$OC/plugins/engram-recall.ts" ]] && ok "opencode: no duplicate recall plugin next to one that runs the hook" \
  || bad "a second recall plugin was installed"

hermes_soul; hermes_soul
[[ "$(grep -c 'engram:begin' "$HOME/.hermes/SOUL.md")" == 1 ]] && grep -q mcp__engram_memory_recall "$HOME/.hermes/SOUL.md" \
  && ok "hermes: SOUL.md block written once, idempotent" || bad "SOUL.md: $(cat "$HOME/.hermes/SOUL.md")"
printf 'my own rules\n' > "$HOME/.hermes/SOUL.md"; hermes_soul
head -1 "$HOME/.hermes/SOUL.md" | grep -q 'my own rules' && ok "hermes: existing SOUL.md content kept" || bad "SOUL.md content lost"
printf '## Shared memory (engram)\nhand written\n' > "$HOME/.hermes/SOUL.md"; hermes_soul
grep -q 'engram:begin' "$HOME/.hermes/SOUL.md" && bad "overwrote a hand-written engram section" || ok "hermes: hand-written engram section left alone"

# Hermes recall hook: installed, registered under hooks.pre_llm_call once, other config untouched
mkdir -p "$HOME/.hermes"
printf 'model: m\nhooks:\n  pre_tool_call:\n    - matcher: terminal\n      command: /x/advisor.py\n      timeout: 300\n\ntools:\n  a: 1\n' > "$HOME/.hermes/config.yaml"
hermes_recall_hook; hermes_recall_hook
python3 - "$HOME/.hermes/config.yaml" "$CLAUDE/hermes-recall-hook.py --slug -s" <<'PY' && ok "hermes: recall hook registered once, the rest of the config kept" || bad "hermes recall hook: $(cat "$HOME/.hermes/config.yaml")"
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
pre = c["hooks"]["pre_llm_call"]
assert [h["command"] for h in pre] == [sys.argv[2]], pre
assert c["hooks"]["pre_tool_call"][0]["command"] == "/x/advisor.py" and c["model"] == "m" and c["tools"] == {"a": 1}, c
PY
[[ -x "$CLAUDE/hermes-recall-hook.py" ]] && ok "hermes: recall hook adapter installed" || bad "adapter missing"
printf 'model: m\n' > "$HOME/.hermes/config.yaml"; hermes_recall_hook
python3 -c "import sys,yaml; c=yaml.safe_load(open(sys.argv[1])); assert c['hooks']['pre_llm_call'][0]['command'].endswith('--slug -s') and c['model']=='m'" "$HOME/.hermes/config.yaml" \
  && ok "hermes: a config without hooks gets the section" || bad "no hooks section: $(cat "$HOME/.hermes/config.yaml")"

WANT_OPENCODE=no; WANT_HERMES=no; rm -rf "$HOME/.config/opencode" "$HOME/.hermes"; : > "$CALLS"
opencode_register; hermes_soul; hermes_register engram x; hermes_recall_hook
[[ ! -s "$CALLS" && ! -e "$HOME/.config/opencode" && ! -e "$HOME/.hermes" ]] && ok "--no-opencode / --no-hermes do nothing" || bad "opt-out ignored"

echo "── test_install_agents: $([[ $fails -eq 0 ]] && echo PASS || echo "FAIL ($fails)") ──"; exit $((fails>0))
