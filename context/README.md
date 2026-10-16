# Muse build history — the original agent conversation

These two files are the **complete record** of the prior agent session that
built this codebase. They are NOT in git (89MB trajectory) — they ship as
release assets on this repo so the git history stays light.

| File | Size | What it is |
|---|---|---|
| `conversation-2026-10-16-153003.txt` | ~592KB, 14,413 lines | Readable transcript — user prompts (`❯`) + agent steps (`◆`). **Read tail-to-head** for the chronological intent. |
| `trajectory-2026-10-16-152947.json` | ~89MB | Raw Muse trajectory JSON — every tool call + file diff. Machine-usable; parse selectively, don't load whole. |

## Fetch

```bash
ALIVE_REPO=GC-WORK11/alive-engine ./context/fetch.sh   # requires gh auth
```

## Why it matters

The transcript is the *true spec*: the user's rejected directions
(hardcoded command lists, dev dashboards, impossible FPS claims), the
orb→capsule UI demands, the "sense at 30 / understand at 240" breakthrough
framing, and every bug hit while building (ort API shifts, nokhwa
`open_stream`, MutexGuard-across-await, X11 focus quirks).

When `documents/` and the transcript disagree, trust the code — but read
the transcript to understand *why* the code looks the way it does.
