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
# Say what is actually being tested. The default is the DEPLOYED binary, which is
# the right thing to ship but a foot-gun after a code change: on 2026-11-22 this
# script passed against the Nov-21 build while the new one sat in target/release.
echo "recall-precision eval: hook=$HOOK mtime=$(stat -c %y "$HOOK" 2>/dev/null || echo missing)"
# NOTE: this probe covers the code AND this host's config together — case D's
# forbidden vector-leg noise depends on vector_store.recall.threshold. The
# code-only guard for the ranking itself is: cargo test -p engram-retrieval

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

A=$(run A-engram "Me ajude a analisar o software engram de memoria para llm, quando os hooks puxam conhecimentos relevantes, precisamos de memórias mais precisas. Maybe we need to rebuild the graph or the vector db?")
# Expected: the memory about REBUILDING THE VECTOR INDEX — the prompt literally asks
# "maybe we need to rebuild the graph or the vector db". It asserted
# project_engram_manual_only before 2026-11-22, because that happened to be the top
# hit of the old ranking; the memory is about engram's operating MODE, which this
# prompt never asks about. With field weights that actually apply, the two memories
# about graph-insert failures and index rebuilds (both on topic) now rank higher, and
# manual-only drops out. Re-pointed at the memory that answers the question, NOT
# relaxed to fit the new code. (The strings grepped here are the printed NAME fields,
# not the filenames — this store mixes hyphenated names with underscored files.)
expect A-engram "$A" "reference-engram-index-needs-delete-then-insert"
forbid A-engram "$A" "fala-sp"                 # vector-noise from the before run

B=$(run B-moises "quanto custa a energia da mineracao do Moises este mes e quanto eu minerei de BTC")
expect B-moises "$B" "reference_moises_host_energy_billing"   # missed before: accents + function words
forbid B-moises "$B" "claude_usage_notify"                    # matched only "quanto custa"/"de/da/do"

C=$(run C-gads "como estao os anuncios do google ads do drmolina esta semana")
expect C-gads "$C" "project_telemedicine_site"
forbid C-gads "$C" "docker ps status"          # "esta" substring-matched "restarts" before

# D: the owner's prompt of 2026-11-22, the one that showed the precision fix was still
# not precise. It names a topic ("robosats coordinator") that four memories are NAMED
# after. Every one of them lost, because (a) `reference-robosats-gateway` tokenized as
# one token so the topic word never reached the name — the ×3 name weight applied to
# nothing; (b) the name×3/description×2 field weighting was inert under BM25 tf
# saturation; and (c) the vector leg, whose top hit for this prompt is 0.6565 on a
# memory with an EMPTY description, voted twice for its own noise and beat a precise
# single-leg lexical hit.
D=$(run D-robosats "Vamos continuar o plano da implementação do robosats coordinator, veja na pasta canonica de planos o que já temos pronto e vamos prosseguir")
expect D-robosats "$D" "project_robosats_pagcoin_coordinator"
expect D-robosats "$D" "project_robosats_temple_dispute_20261120"
expect D-robosats "$D" "reference_robosats_gateway"
expect D-robosats "$D" "project_robosats_arb_bot"
forbid D-robosats "$D" "autonomous curate"                 # top of the vector leg, empty description
forbid D-robosats "$D" "Mercado Livre"                     # matched the prompt's conversational words
forbid D-robosats "$D" "feedback_official_download_own_channel"  # #1 of the OLD keyword leg: "vamos"+"temos"
forbid D-robosats "$D" "dsec-training-scope-doc"           # matched only "plano"+"pasta"

E=$(run E-unrelated "qual filme vou assistir hoje a noite com a familia no cinema")
forbid E-unrelated "$E" "Graph facts"          # nothing on this prompt should reach the fact leg

[ "$fail" = 0 ] && echo "recall-precision eval: all cases passed"
exit "$fail"
