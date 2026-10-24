---
name: memory-tidy
description: Use when the user asks to clean up, tidy, consolidate, organize, or de-duplicate their memories / memory store — e.g. "tidy my memories", "clean up my memory", "consolidate memories", "merge duplicate memories". Triggers engram's auto-consolidation pass on demand instead of waiting for the weekly schedule, then walks every proposal that needs the user's OK one at a time.
---

Every script below runs with the vector venv interpreter,
`~/.claude/vector/venv/bin/python`, written out in full in each command (a shell
variable does not survive between tool calls). Plain `python3` has no PyYAML, cannot parse `engram.yaml`, and then reports
"auto_curate.enabled is false" even when it is true.

## Step 1: run the consolidation pass

    ~/.claude/vector/venv/bin/python ~/.claude/memory_auto_curate.py --apply </dev/null

What it does:
- **Merges near-duplicate memories** (same type, very similar text) into one umbrella
  memory. Merges are *reviewed* (Codex if installed, else queued for the user) and
  *reversible*: absorbed sources go to `.quarantine/` for a 30-day probation.
- **Proposes stale orphans** (old, never-recalled, no merge target) for pruning. Never
  auto-deleted; approved prunes go to `.trash/` (90-day undo).

It respects `auto_curate.enabled` and `auto_curate.review_gate` in `~/.claude/engram.yaml`
(if `enabled: false`, tell the user to flip it).

## Step 2: drop expired proposals, list the rest

    ~/.claude/vector/venv/bin/python ~/.claude/engram_telegram_gate.py --poll </dev/null
    ~/.claude/vector/venv/bin/python ~/.claude/engram_telegram_gate.py --list </dev/null

`--poll` drops proposals past their 72h TTL (an expired one can never be approved) and
applies anything already approved. Each dropped one prints a line naming its files
("absorb X into Y"); keep those names for the final report. The engram daemon has NO
Telegram token, so its own one-tap buttons never reach the user: THIS walkthrough is
the only way proposals get answered.

## Step 3: one proposal at a time

For each pending id from `--list`, read it THROUGH THE GATE (never from a path you
build yourself: the gate resolves the store it will approve in):

    ~/.claude/vector/venv/bin/python ~/.claude/engram_telegram_gate.py --show <id> </dev/null

It prints the proposal plus `describe` (which files become what), `store_ok`, and
`secret_scan` (per file and for `umbrella_content`: `null` = clean, anything else =
possible secret or scan unavailable). If `store_ok` is false, do not present it: report
it as refused.

1. Build a card in the user's language, plain words, no jargon (no "near-dup",
   "cosine", "TTL", "umbrella"). Name every memory by its file name AND its frontmatter
   description; never refer to a proposal only by its id.
   - `merge_apply`: "Juntar `<absorbed>` dentro de `<umbrella>`". For each file, its
     description and 1-2 sentences of what it holds (read it locally); then what the
     merged memory would say (trim to ~1200 chars) and whether any fact from the
     sources is missing from it.
   - `orphan_prune`: "Apagar `<name>`" (recoverable from `.trash/` for 90 days), its
     description, a 2-sentence summary, and why it was flagged (old, never recalled).
   - Any other op: its `op`, `describe` and `preview`.
   - **Secret scan:** for any file (or `umbrella_content`) whose `secret_scan` is not
     `null`, put NO summary and NO merged text in the card. Write "conteúdo omitido:
     possível segredo (linhas N); revise o arquivo localmente" instead. The card is
     sent to a remote chat.
2. Ask with your question tool (Hermes: `clarify`; Claude Code: `AskUserQuestion`),
   choices exactly `Approve` / `Reject` / `Skip`.
   On Hermes/Telegram the WHOLE card goes INSIDE the clarify `question` text (end it
   with `O que fazer?`). Text written next to the clarify call is not delivered on
   Telegram. No fenced code blocks.
3. Approve: `~/.claude/vector/venv/bin/python ~/.claude/engram_telegram_gate.py --approve <id>`.
   Reject: `~/.claude/vector/venv/bin/python ~/.claude/engram_telegram_gate.py --reject <id>`.
   Skip: leave it pending (it expires at its TTL).
   A "refused" or "already <state>" answer is reported, not worked around.

After the last one, apply the approvals:

    ~/.claude/vector/venv/bin/python ~/.claude/engram_telegram_gate.py --poll </dev/null

Report each `<id>: ok|FAIL — detail` line it prints, with the files it names. A FAIL
(e.g. a source changed since the proposal) is reported, never retried by hand.

## Step 4: final report

Same language and rules as the cards: plain words, files named, no ids on their own.

1. First line: what changed in the store. If nothing was merged or deleted, say so
   plainly ("Nada foi alterado").
2. One section per outcome, each item as "`<file>` (description) -> what happened":
   merged automatically (with the 30-day undo), approved and applied, rejected,
   skipped (still pending, with its expiry), and expired unanswered (dropped, nothing
   changed; the next pass proposes it again only if it still qualifies).
3. If the pass found nothing to merge, say that the automatic step has nothing left
   and that any further consolidation is a manual review. Only offer it with a
   concrete list (file names + descriptions) of what would be reviewed.

Never apply a merge or prune the user did not approve.
