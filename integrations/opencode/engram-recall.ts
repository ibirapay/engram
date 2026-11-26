// engram for opencode (v2 plugin API): automatic recall on every prompt, the same way the
// Claude Code UserPromptSubmit hook does it. The hook binary reads the prompt, searches the
// memory store (keyword + vector + graph) and prints a <relevant-memory> block; we append it
// to the END of the user turn (not the system prompt: a changing system prompt would bust a
// local model's KV cache).
// The MCP tools (memory_recall, memory_recall_hybrid) are registered separately by install.sh
// (`opencode mcp add`), so the model can also ask for a recall on purpose.
import { spawnSync } from "child_process"
import os from "os"
import path from "path"

const BIN = process.env.ENGRAM_BIN ?? path.join(os.homedir(), ".claude")
const HOOK = path.join(BIN, "rust", "engram-recall-hook")

export default {
  id: "engram-recall",
  async setup(ctx: any) {
    const cwd: string = ctx.location?.directory ?? process.cwd()
    // opencode can fire the prompt hook twice for one user message; recall once per distinct text.
    const done = new Set<string>()

    await ctx.session.hook("prompt", (event: any) => {
      const text = event?.prompt?.text
      if (typeof text !== "string" || text.includes("<relevant-memory>")) return
      const key = `${event.sessionID ?? ""}\n${text}`
      if (done.has(key)) return
      done.add(key)
      const input = JSON.stringify({ prompt: text, session_id: event.sessionID ?? "", cwd, hook_event_name: "UserPromptSubmit" })
      const r = spawnSync(HOOK, [], { input, timeout: 15000, encoding: "utf8" })
      const out = (r.stdout || "").trim()
      if (r.status === 0 && out) event.prompt.text = `${text}\n\n${out}`
    })
  },
}
