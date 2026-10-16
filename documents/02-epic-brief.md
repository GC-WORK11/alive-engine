# ALive — Epic Brief

## One line

**ALive is a desktop input method for people who don't speak and don't want
to type: a camera watches quietly in the background, and signing drives text
and actions into whatever app has focus.**

Not a translator demo. Not a voice clone. A real input layer — sign as
keyboard+mouse replacement.

---

## What the product is

A minimal always-on desktop companion:

- A **camera runs unobtrusively in the background**. No video window. No
  dashboard. The orb — a small floating capsule that lives on top of
  everything — is the entire UI.
- It sees **hands, fingers, pose** in real time, copes with normal room
  lighting (auto-exposure lock, adaptive gain path, motion gate that ignores
  exposure snaps), and knows when someone is signing versus just sitting
  there (motion gate + endpointer, not "always interpreting").
- Signing produces **committed text** that is typed into whatever the user
  is focused on — chat box, doc, terminal — the same way Wispr Flow types
  dictated words.
- Two modes:
  - **Prompt mode** (`Ctrl+Shift+P`): sign → words → typed into the focused
    composer. "Sign it, it appears."
  - **Computer mode** (`Ctrl+Shift+C`): sign → a free-form instruction → a
    tool-using agent operates the machine (open, search, click, type, copy,
    wait) — *not* a hardcoded list of commands.
- Every action is **journaled, redacted, and reversible where possible**;
  risky steps ask first. Dry-run is the default until the user opts in.

The end goal the user described: *"He just opens the app, camera works in
the background — don't show in the UI. Milliseconds. The capsule gets bigger
when signing, shows the interpretation in real time, and when I sign ENTER
it goes into the composer box I was in. Later — computer use."*

That is exactly this product, mid-build.

---

## Why it exists (the purpose)

Three honest user groups:

1. **People who can't or don't want to speak** — Deaf and hard-of-hearing
   users for whom sign is their primary language. Today their input options
   are keyboard, phone keyboard, or worse. Voice assistants simply don't
   serve them. Sign→text+action is a *parity* feature, not a luxury.
2. **Hands-busy / hands-free computing** — cooking, lab work, accessibility
   setups, motor-impairment contexts where a keyboard is expensive but a
   gesture is cheap.
3. **Power users who want ambient computer control** — the Wispr-Flow
   pattern (ambient → capsule → commit) applied to vision instead of voice.

**The strategy (from the PRD, still correct):** don't compete on dictation
speed — sign ≈ 2–3 signs/sec vs speech ≈ 4–5 words/sec; Wispr wins raw
typing. Compete on **action density**: one sign = one completed task
("reply yes", "open mail and paste the last draft"), which raw dictation
can't do without an agent loop. The win metric is **sign→done**, not
words-per-minute.

---

## Why this hackathon (WeMakeDevs × Wispr — "No Hands: The Wispr Challenge")

- The challenge literally asks: **what would you build if you had hands-free
  computing?** ALive is that question answered for people who can't use
  their voice either — the most underserved input modality there is.
- Wispr Flow's core loop — ambient listening, capsule UI, commit into any
  app, agent-style computer use — is *the exact product shape* ALive
  reimplements on the vision channel. It shows deep understanding of what
  makes Flow good while extending it somewhere voice can't go.
- It's buildable in the window: the hard parts (camera→pose→segment→match→
  act) already exist in this repo; the remaining work is wiring, polish,
  and honest scoping — not research.
- Builder-track fit: it's a real desktop tool the judges can run and sign
  into, not a prompt demo.

**What wins the demo:** a 2-minute video that shows (a) camera already on,
(b) sign → committed word typed into a real app, (c) sign → agent does a
small task with a confirm gate, (d) the orb breathing through its states.
No slides needed.

---

## The flows, end to end

### Flow A — Prompt mode (sign → text into focused app)

```
camera 30fps ─► motion gate (cheap pixel delta; skips on stillness,
                skips on exposure snaps)
            ─► frame slot (latest only)
            ─► infer thread: YOLOX detects person once → RTMW poses
               the tracked bbox each tick (133 keypoints)
            ─► result slot ─► main loop
            ─► hand_centroid per hand (score-weighted, ≥6 keys)
            ─► Endpointer: move → pause 350ms = Tentative (speculative
               display), pause 700ms + EOS = Commit
            ─► committed hand path ─► DTW vs per-user templates
            ─► gloss ─► lexicon intent / Jev decision
            ─► OrbBar 'committed' event ─► overlay_commit_type
            ─► text typed into whatever window has focus
            ─► orb auto-hides
```

### Flow B — Computer mode (sign → agent acts)

```
same perception ─► committed sign text
              ─► agent_act {text, history, pending, approve}
              ─► optional Jev triage (tool choice + safety nouls)
              ─► DeepSeek chat loop (≤6 steps): plans tool calls
              ─► each call validated against the 15-verb ToolId set
              ─► LocalExecutor: dry-run default / confirm gate on tier-2
                 verbs / argv-only (no shell) / every step journaled
              ─► AgentPendingDto pauses on confirm → Enter resumes
              ─► ActionStarted/Completed events → orb states
              ─► done → orb auto-hides
```

### Flow C — Calibration (how the matcher learns "your" signs)

```
user performs a sign a few times ─► template_add stores normalized
exemplars (≤5, FIFO) ─► persisted to alive-state/templates.json
──► recognition = nearest DTW with a reject threshold; unknown motions
reject, never guess.
```

### Flow D — Confirmation & undo

```
tier-2 verb or Jev flags risky/confirm ─► Route::Confirm
──► NeedsConfirmation preview ─► journal awaiting_confirm
──► user approves ─► execute + journal Done (+undo payload when one
exists, e.g. screenshot→delete_file) ─► undo_action consumes it once.
```

### Orb state flow (what the user actually sees)

```
hidden ──(hotkey / signing)──► Activating ──► CameraActive
    ──► SignerDetected ──► Interpreting (listening state on the orb)
    ──► IntentForming/Executing (thinking/talking on the orb)
    ──► Done ──► auto-hide
    Esc anywhere = hide.  Errors = red state + auto-hide.
```

---

## Feature inventory — honest status

| Feature | State |
|---|---|
| Background camera, unobtrusive | ✅ works (no video UI by design) |
| Latest-frame pipeline (fresh not fast) | ✅ works |
| Person detect → tracked 133-kp pose | ✅ works (~105ms pose p50, overlapped) |
| Motion gate w/ exposure rejection | ✅ works |
| Sign segmentation (tentative+commit) | ✅ works, tested |
| Command-gesture recognition (DTW) | ✅ works — small calibrated vocab |
| **General sign-language recognition** | ❌ trait seam only (`NullEncoder`) |
| ASL pack | ✅ 26-entry command set |
| CSL pack (default) | ✅ parity with ASL; default unvalidated |
| Jev typed decision layer | ✅ wired; needs API key |
| DeepSeek tool loop (free-form agent) | ✅ wired; needs API key; model id unverified |
| 15-tool safe executor + journal + undo | ✅ works, dry-run default |
| Prompt/computer hotkeys + orb states | ✅ wired; orb→OrbState mapping partial |
| **240Hz upsampler + Kalman + skeleton overlay** | ⚠ built & tested, **not wired in** |
| AGC dark-room gain | ⚠ built, **not wired in** |
| Visible composer/capsule text | ❌ types into focused app directly; no visible composer UI |
| Onboarding / pack picker | ❌ |
| Wayland input path | ⚠ wtype supported; X11-first today |

---

## What "done" looks like for the submission

1. The wired-30% landed (upsampler ticks, overlay leads/ribbons, AGC).
2. A 5–8 sign calibrated vocabulary demonstrated live (sign → word →
   typed; sign → command → agent).
3. One free-form computer-use demo with a visible confirm gate.
4. Honest latency telemetry on screen (fps, pose ms, commit ms).
5. A demo video ≤2 min + Wispr usage card + public repo link.

## The non-negotiables (project law, keep or lose trust)

- Never claim sensor FPS above what the camera emits — prediction and
  interpolation are *labeled* virtual ticks, fenced from recognition.
- Never guess: unknown sign → reject; unknown tool → refuse; unknown JSON
  shape → error. Honest degrade everywhere.
- Dry-run default; confirm on externally-visible actions; journal
  everything, redact text.
- The orb renders *pipeline states*, not marketing states.
- UI is chrome-less: one orb, zero panels in the user path.
