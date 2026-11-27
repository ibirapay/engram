## Shared memory (engram)

You share one long-term memory store with the user's other agents (Claude Code, Hermes Agent). It
holds their projects, infrastructure, decisions and preferences. Treat it as your own memory.

- **It comes to you on its own**: each prompt arrives with a `<relevant-memory>` block of the
  memories that match it (names and one-line descriptions). When one matters, recall it in full with
  the engram MCP server's `memory_recall` tool (`q` = the question, optionally `k`).
- **Recall before acting** when the work depends on a past decision, a runbook, an incident or the
  user's preferences, and again before writing an operational script (deploys, migrations, anything
  that moves money or data): a proven version may already exist.
- **Write to it** when you learn something durable that is not derivable from the code. Pass the
  body through a QUOTED heredoc so the shell cannot expand it:
  `~/.claude/save_memory.sh <name>.md "<one-line description>" <<'EOF'` then the body, then `EOF`
  on its own line. One fact per file; recall first and update an existing file rather than creating a
  near-duplicate.
- Your sessions are read every night: durable facts from them are staged, screened and graduated
  into the store automatically. The skills `memory-tidy` and `memory-promote` and the commands
  `/memory-checkpoint`, `/memory-clean-review` and `/memory-reformat` work the same as in Claude Code.
