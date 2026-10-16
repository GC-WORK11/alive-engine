# MASTER PROMPT — ALive handoff (paste this to the fresh agent)

```
You are taking over ALive — a Rust+Tauri desktop input method that lets a
person control a computer and produce text by SIGNING on camera, with zero
speech and zero typing. This is a working engine mid-build, not a concept.
The repo you just cloned contains ~9,700 lines of tested, architecture-
first code. Your job is to finish the last 30%, polish the capsule UI, and
ship it as a hackathon submission (WeMakeDevs × Wispr "No Hands" challenge,
deadline Oct 30, 11:59pm IST).

REPO: https://github.com/GC-WORK11/alive-engine (private — you need access)
Models are NOT in git (>100MB). Fetch them:
  ALIVE_REPO=GC-WORK11/alive-engine ./models/fetch.sh   (gh auth required)
Verify: sha256sum -c models/SHA256SUMS

BUILD HISTORY (the true spec — read when confused):
  ALIVE_REPO=GC-WORK11/alive-engine ./context/fetch.sh
  → conversation-*.txt: prior agent transcript, user prompts marked ❯.
    READ TAIL-TO-HEAD for chronological intent (14K lines).
  → trajectory-*.json (89MB): raw Muse trajectory, parse selectively.
  The transcript explains WHY the code is shaped this way — rejected
  directions, UI demands, and every bug already fixed once.

═══════════════════════════════════════════════════════════════════════
PART 1 — WHAT THE PRODUCT IS
═══════════════════════════════════════════════════════════════════════

ALive = sign as an input method. A camera watches quietly in the
background — never shown in the UI. When the user signs, a floating
capsule expands, shows the interpretation in real time, and the committed
text/actions land in whatever app has focus. Two modes:

- PROMPT mode (Ctrl+Shift+P): sign → words → typed into the focused app.
- COMPUTER mode (Ctrl+Shift+C): sign → free-form instruction → a
  tool-using agent operates the machine (open, search, click, type,
  scroll…). NOT a hardcoded command list — a generic 15-verb tool set +
  an OpenAI-compatible LLM loop.

The user's exact vision (non-negotiable):
  "He opens the app, camera works in the background — don't show it in
  the UI. In milliseconds — contrast, skin, fingers, shadows, lighting —
  it just works. The capsule gets bigger when signing, shows the
  interpretation in real time, and when I sign ENTER it lands in the
  composer box I was in. Then computer-use mode."

The user REJECTED: developer dashboards, duplicate windows, matte-blur
takeovers, cheap buttons, hardcoded "open gmail" commands, and any claim
of impossible camera FPS.

═══════════════════════════════════════════════════════════════════════
PART 2 — THE TARGET UI (the capsule, not an orb-only page)
═══════════════════════════════════════════════════════════════════════

Design references are IN the repo: documents/design/capsule-mockup-1.png
and capsule-mockup-2.png. Build to those. What they show:

- A single dark rounded CAPSULE (pill) floating above the desktop —
  bottom-left or top-center. NOT a bare orb on a blank window.
- Inside the capsule, left to right:
    [ALive label] · [mode/hotkey hints ⌃⇧P prompt ⌃⇧C computer]
    · [status text: "Signed: OPEN BROWSER"]
    · [ISL/language badge] · [the animated orb on the right edge]
- Footer line under the capsule: "No hands · signed-native compute".
- The orb itself (src/components/ui/orb.tsx already exists — a GLSL
  perlin-noise shader circle) lives INSIDE the capsule as the status
  light: LOW amplitude/calm breathing when idle, rising amplitude +
  color shift when listening/working, green pulse on done, warm/red on
  error or confirm-needed.
- The capsule EXPANDS (width spring) while signing is active or a state
  needs attention, contracts when idle. Transparent window, always on
  top, skip taskbar, no decorations.
- The window is already configured this way (180×180, transparent,
  alwaysOnTop, hidden — src-tauri/tauri.conf.json). Expanding it for
  text means resizing the window (Rust setSize) or a second larger
  capsule window. Keep ALL text minimal — status line only, never a
  panel.

═══════════════════════════════════════════════════════════════════════
PART 3 — WHAT EXISTS (verified state of the code)
═══════════════════════════════════════════════════════════════════════

Rust workspace, 9 crates + Tauri shell (~9,700 LOC, heavily tested):

- alive-core      — OrbState(12), EventBus, Config (640×480@30, pack=csl,
                    dry_run_actions=true, 350/700ms endpointing)
- alive-perception — CaptureSource (nokhwa MJPEG + YUYV path, synthetic,
                    imagedir) + ONNX backend: YOLOX-m person detect +
                    RTMW-l 133-keypoint pose, SimCC decode, affine warp,
                    scratch-reuse (bit-identical tested). AGC gain fn
                    exists but IS UNUSED.
- alive-temporal   — Endpointer: Idle→Signing→Pausing, Tentative@350ms,
                    Commit@700ms+EOS≥0.85, cancel on hands-lost.
- alive-track      — ⚠ THE BREAKTHROUGH, WRITTEN + TESTED BUT NOT WIRED:
                    kalman.rs (constant-velocity Kalman, teleport gate,
                    coast states) + upsample.rs (causal PCHIP → 240Hz
                    labeled ticks, anti-hallucination rules R1–R7).
                    NOTHING in src-tauri calls these yet.
- alive-sign       — LanguagePack loader (pack.toml+lexicon+intents,
                    packs/asl + packs/csl ship), DTW trajectory matcher
                    (HONEST SCOPE: calibrated command gestures only —
                    NOT general sign-language recognition), NBest,
                    SignEncoder trait + NullEncoder seam.
- alive-intent     — JevClient (typed SystemOne triage:
                    {model,state,questions}→answers, needs
                    ALIVE_JEV_API_KEY), ToolLlmClient (OpenAI-compatible
                    loop, defaults api.deepseek.com + deepseek-flash —
                    VERIFY that model id; env ALIVE_LLM_ENDPOINT/MODEL,
                    key ALIVE_LLM_API_KEY or DEEPSEEK_API_KEY).
- alive-act        — ToolId: 15 generic verbs, OpenAI schemas, risk
                    tiers, argv-only executor (NO SHELL), xdotool/wtype/
                    xdg-open/clip/screenshot, dry-run default, redacted
                    journal, one-shot undo. X11-first, wtype for Wayland.
- alive-journal    — append-only JSONL (alive-state/journal.jsonl).
- alive-ipc        — DTOs mirrored 1:1 by src/types.ts.
- src-tauri        — pipeline: camera → motion gate (hysteresis +
                    exposure rejection) → latest-frame slot → infer
                    thread → result slot → endpointer → handle_commit →
                    matcher → decide → execute. Commands: get_status,
                    agent_act, overlay_show/hide/state,
                    overlay_commit_type, templates… Global hotkeys wired.
                    PiP thread + aliveframe:// protocol exist.
- Frontend         — src/OrbBar.tsx (mode + agent state → orb),
                    orb.tsx (the shader orb — REUSE INSIDE capsule),
                    types.ts, tauri.ts (mocks for browser dev).

ALSO IN REPO: documents/ — the full brain (01 architecture walkthrough
with per-module analysis, 02 epic brief, 03 tech stack, 04 ui/ux flows,
05 roadmap). READ documents/ FIRST. The three root files
alive-prd-sota.md / alive-build-plan.md / alive-deep-research.md are the
original design contracts — still accurate.

MEASURED NUMBERS (CPU): pose-on-bbox ~105ms p50 (12 threads), detect
~430ms (background cadence only), effective pose ~10–13Hz overlapped,
capture ~25–30fps. Commit contract: p50 ≤400ms, p95 ≤900ms.

═══════════════════════════════════════════════════════════════════════
PART 4 — THE JOB, IN ORDER
═══════════════════════════════════════════════════════════════════════

1. BUILD + VERIFY FIRST. cargo test --workspace; npm ci && npm run build;
   cargo build -p alive; ALIVE_REPO=GC-WORK11/alive-engine
   ./models/fetch.sh. Then cargo run --bin alive-demo (headless E2E
   prints one JSON verdict). On Linux X11 run the real app and verify
   hotkeys + orb render before touching anything.
2. WIRE THE BREAKTHROUGH (the missing 30%):
   - run_pipeline: construct Upsampler + HandTracker; feed PoseSample
     from result-slot landmarks via hand_centroid_full (HAND_A 91..112,
     HAND_B 112..133, min_score 0.4, ≥6 keys); push ticks into the
     endpointer/trail path; note_ticks(real+virtual count).
   - Tracker: observe() on reals, mark_missed() on absence,
     estimate_at(now) → OverlayFrame.leads (UI-only, fenced).
   - Write shared.overlay every pose: segs (skeleton bones), ribbons
     (upsampled centroid history), leads (Kalman predict). The PiP draw
     path already exists — it currently draws an empty frame.
   - Wire agc_gain (motion pass luma → gain) into capture or warp.
   - Switch ≤480p capture to YUYV (FrameFormat::YUYV) w/ MJPEG fallback.
   - Proof: status.upsample_hz ~240 while signing; overlay shows ribbons.
3. BUILD THE CAPSULE UI per mockups (see PART 2): expand-on-activity,
   status text line, ISL/pack badge, footer, orb-as-status-light with
   low→high amplitude states mapped from ALL 12 OrbStates (not just the
   3 agent states used now). Esc always dismisses.
4. COMPOSER + ENTER: buffer committed glosses into a visible capsule
   line; add an "ENTER" template/intent that types the buffer into the
   focused app and clears. (overlay_commit_type exists and types into
   focus.)
5. DECISION LAYER: verify live endpoints/models; keep Jev for triage,
   DeepSeek for the tool loop; support a local OpenAI-compatible server
   (llama.cpp/vLLM/ollama) via ALIVE_LLM_ENDPOINT with zero code change.
6. DEMO-VOCAB CALIBRATION: 5–8 signs ×3 samples into templates.json;
   freeze them for the video.
7. HACKATHON DELIVERABLES: ≤2-min demo video, Wispr usage-card
   screenshot, public/judge-usable build, problem/user/process write-up.

═══════════════════════════════════════════════════════════════════════
PART 5 — PROJECT LAW (break these = lose the user's trust)
═══════════════════════════════════════════════════════════════════════

- HONEST MATH: sensor FPS ≠ understanding ticks. Virtual ticks are
  labeled Provenance::Virtual and fenced from recognition (R7: ≥3 reals
  spanning ≥250ms required to commit). Prediction (Kalman) is UI-only.
  NEVER claim "720fps camera" or "0 latency" — the contract is "sense at
  sensor rate, understand at 240Hz, react within a display frame".
- NEVER GUESS: unknown sign → reject; unknown tool → refuse; unknown
  JSON shape → error; no API key → honest "not configured".
- SAFE ACTIONS: dry_run_actions=true default; tier-2 verbs (type_text,
  key_press, clicks, scroll, open_app) behind confirm; argv-only
  execution — model output never becomes a shell; typed text redacted
  from the journal; undo is one-shot.
- NO CHROME in the user path: one capsule. Debug/telemetry behind a flag.
- Follow existing conventions; prefer editing over new files; keep the
  frontend thin — data path stays in Rust.
- Don't hold MutexGuard across .await (this was a real bug; keep it out).

═══════════════════════════════════════════════════════════════════════
PART 6 — CONTEXT YOU DIDN'T ASK FOR BUT NEED
═══════════════════════════════════════════════════════════════════════

- Why CSL is default: largest signer population globally — provisional,
  unvalidated with native signers. The mockups show "ISL" — for the
  user's own demo and India hackathon, an ISL pack (or a picker) is the
  right default. Packs are data directories; adding one = zero code.
- Trajectory matcher ≠ sign language: it recognizes user-calibrated
  MOTION gestures. For the demo that IS the product (calibrated command
  grammar) — say it honestly. Real sign recognition = a landmark→gloss
  ONNX behind the existing SignEncoder trait (R1 milestone).
- Strategy (from the PRD): don't race Wispr on words/sec (sign ≈2–3/s
  vs speech ≈4–5/s); win on action density — one sign = one done task.
- The user communicates via voice dictation (Wispr); messages may be
  garbled — infer intent, don't nitpick typos. They want autonomy +
  honesty, brutal over flattering, verified over asserted.
- X11 quirks seen: overlay Esc needs windowfocus first; scratch-window
  typing via xdotool works ("typed N chars"); keep windows focusable.
```

After the prompt, tell the agent: "Start with documents/README.md →
01-architecture.md, then run the build+verify steps before any edits."
