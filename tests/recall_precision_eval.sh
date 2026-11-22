#!/usr/bin/env bash
# Live precision check for the prompt-hook recall. Unlike graph_recall_eval
# (which compares two backends for agreement), this asserts OUTCOMES for real
# owner prompts: the memory that answers the question must be injected, and the
# off-topic ones that motivated fix/recall-precision must not be.
#
#   tests/recall_precision_eval.sh [path-to-engram-recall-hook]
#
# Needs the live store, Qdrant and Neo4j — it is an integration probe, not a
# unit test. Each case gets a fresh session id so the hook's per-session dedup
# never hides a failure on re-run.
set -u
HOOK="${1:-$HOME/.claude/rust/engram-recall-hook}"
fail=0

run() { # tag prompt -> stdout is the injection
  printf '{"prompt": %s, "session_id": "eval-%s-%s", "cwd": "%s"}' \
    "$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$2")" \
    "$1" "$$" "$HOME" | "$HOOK"
}

expect() { # tag haystack
  if ! grep -qF -- "$3" <<< "$2"; then
    echo "FAIL [$1]: expected '$3' to be injected"; fail=1
  fi
}
forbid() {
  if grep -qF -- "$3" <<< "$2"; then
    echo "FAIL [$1]: '$3' must not be injected"; fail=1
  fi
}

A=$(run A "Me ajude a analisar o software engram de memoria para llm, quando os hooks puxam conhecimentos relevantes, precisamos de memórias mais precisas. Maybe we need to rebuild the graph or the vector db?")
expect A-engram "$A" "project_engram_manual_only"
forbid A-engram "$A" "fala-sp"                 # vector-noise from the before run

B=$(run B "quanto custa a energia da mineracao do Moises este mes e quanto eu minerei de BTC")
expect B-moises "$B" "reference_moises_host_energy_billing"   # missed before: accents + function words
forbid B-moises "$B" "claude_usage_notify"                    # matched only "quanto custa"/"de/da/do"

C=$(run C "como estao os anuncios do google ads do drmolina esta semana")
expect C-gads "$C" "project_telemedicine_site"
forbid C-gads "$C" "docker ps status"          # "esta" substring-matched "restarts" before

E=$(run E "qual filme vou assistir hoje a noite com a familia no cinema")
forbid E-unrelated "$E" "Graph facts"          # nothing on this prompt should reach the fact leg

[ "$fail" = 0 ] && echo "recall-precision eval: all cases passed"
exit "$fail"
