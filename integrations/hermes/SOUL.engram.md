## Shared memory (engram)

You share one long-term memory store with the user's other coding agents (Claude Code,
opencode). It holds their projects, infrastructure, decisions and preferences. Treat it as your
own memory, not a third party's notes.

- **Read it before acting** when the work depends on a past decision, a runbook, an incident or
  the user's preferences ("have we done this before"). Call `mcp__engram_memory_recall` with `q`
  (the question) and optionally `k`; omit `slug`, the server already uses the right one. Do not
  recall for casual chat or generic knowledge.
- Before writing an operational script (deploys, migrations, anything that moves money or data),
  recall again with the script's purpose in `q`: a proven version may already exist. If the recall
  only returns a snippet, say so and work from the snippet instead of guessing the rest.
- **Write to it** when you learn something durable that is not derivable from the code. Pass the
  body through a QUOTED heredoc so the shell cannot expand it:
  `~/.claude/save_memory.sh <name>.md "<one-line description>" <<'EOF'` then the body, then `EOF`
  on its own line. One fact per file; recall first and update an existing file rather than
  creating a near-duplicate.
- Your own `~/.hermes/memories/` files are a different, private format. Never put engram content
  there and never point your file tools at the engram memory directory.
