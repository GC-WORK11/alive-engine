# ALive — Forward Path: Tools, Shorthand, Releases

## 1. The immediate punch list (ordered by demo leverage)

Work the repo in this order — each item is measurable and bounded:

1. **Wire the breakthrough (the missing 30%):**
   - In `run_pipeline`: construct `Upsampler` + `HandTracker`; on each
     result-slot pose build `PoseSample{present, a, b}` from
     `hand_centroid_full`; `upsampler.push()` → feed `Tick`s to the
     endpointer/trail; `note_ticks(ticks.len())`.
   - Tracker: `hands[i].observe(t_us, x, y, conf)` on real poses,
     `mark_missed` on absence; `estimate_at(now)` → `OverlayFrame.leads`.
   - Write `shared.overlay` every real pose: `segs` (bones from the 133
     keypoints), `ribbons` (upsampled centroid trails), `leads`.
   - Call `agc_gain(luma)` inside `motion_score`'s sample pass → scale
     warp input before pose (or the preview first — safer).
   - Prefer `FrameFormat::YUYV` at ≤480p; keep MJPEG fallback.
   - *Proof:* `status.upsample_hz` reads ~240 while signing;
     PiP shows trail ribbons + lead dots; commit latency unchanged.
2. **Composer line in the capsule:** 1-line text row bound to
   `partial_gloss`/`committed` (target capsule UX). + an `ENTER` template
   so a sign can commit buffered text into the focused app.
3. **Verify the model ids:** ping DeepSeek's model list; fix
   `ALIVE_LLM_MODEL` default if `deepseek-flash` isn't real (likely
   `deepseek-chat`); confirm `api.typesafe.ai` Jev endpoint or run a
   local `/systemone`-compatible server.
4. **Calibrate a demo vocabulary** (5–8 signs ×3 samples each) against
   real camera lighting; store in `alive-state/templates.json`; freeze
   for the video.
5. **Orb → OrbState mapping** (12 states → color/motion params) so the
   orb is the honest status surface it's designed to be.
6. **Onboarding v0:** camera check → pack pick (ASL/CSL) → 3-sample
   calibration. One plain modal window is fine — chrome is banned in the
   *runtime* path, not in setup.
7. **Telemetry HUD for the demo video only:** fps, pose ms, commit ms,
   upsample_hz — a dev flag, off for users.
8. **`git init` reality:** first commit + tagged `v0.1.0-demo` before
   submission — the repo currently has zero history.

## 2. Verification rituals (run before every claim)

```
cargo test --workspace            # all crates, deterministic
cargo run --bin alive-demo        # headless E2E: camera→pose→match→act→journal
npm run build                     # tsc + vite
cargo build -p alive              # the Tauri bin
# live: DISPLAY=:1, xdotool hotkeys, scratch-window typing proof
```

Numbers to report honestly in the demo: capture fps (sensor), pose p50
ms, commit p50/p95 ms (PRD contract: ≤400/≤900), upsample_hz (labeled
*ticks*, not frames), dropped-frame count.

## 3. Communication tools & rituals

- **Build with Wispr Flow open** — dictate prompts, docs, issues,
  commit messages. The submission needs a **Wispr usage-card
  screenshot**; this project also *is* a Wispr-pattern tool, so dogfood
  it visibly.
- **Screen recording:** OBS Studio (system you're on) — record the
  2-min demo in one continuous take; the orb + a real app window in
  frame; camera PiP inset optional (debug flag only).
- **Issues/notes:** keep `documents/` as the living brain — file future
  decisions here rather than in chat threads.
- **Demo script skeleton (90–120s):**
  1. cold open: "I can't always talk, and I don't want to type" (5s)
  2. orb wakes on `Ctrl+Shift+P`, sign a word → text lands in a chat
     box (25s)
  3. sign two more words, sign ENTER → full line committed (20s)
  4. `Ctrl+Shift+C`: sign "open browser and search weather" → agent
     plans → confirm gate → executes (30s)
  5. close on the honest telemetry: "30 real fps in, 240 understanding
     ticks out, commit under half a second" (15s)

## 4. Internal shorthand (speak this, save minutes)

| Term | Meaning |
|---|---|
| **tick** | one understanding step (real pose *or* 240Hz virtual) |
| **bracket** | time between two real poses — the unit the upsampler fills |
| **virtual** | interpolated tick; `Provenance::Virtual` — fenced from recognition triggers |
| **R1–R7** | the upsampler's anti-hallucination rules (per-hand, confidence, stillness, window, v-clamp, real-gate) |
| **EOS** | end-of-sign probability; the 700ms+0.85 commit gate |
| **NBest** | renormalized top-N gloss list |
| **pack** | a language directory (toml+lexicon+intents+eval) |
| **tier-2** | externally-visible verbs behind the confirm gate |
| **dry-run** | plan+journal without executing (the default) |
| **slot** | `Mutex<Option<T>> + seq` latest-only channel between threads |
| **gate** | the motion hysteresis that decides "worth inferring" |
| **lead** | Kalman predict-to-now dot (UI-only, never fed to matcher) |
| **capsule** | the expanding orb container (target UX) |
| **S1/S2** | System-1 (Jev typed triage) / System-2 (LLM tool loop) |

## 5. Release plan

**R0 — hackathon submission (the only deadline that exists):**
wired-30% done, calibrated vocab, one agent demo, ≤2-min video, Wispr
card, public repo, honest README. Ship it even if sign recognition is
still the command grammar — scope it truthfully in the writeup.

**R0.5 — "it actually works on my machine" cleanup:** onboarding,
pack picker, commit-sign, orb state colors, Linux install script
(xdotool/wtype/xclip presence check → `tool_availability` report at
startup).

**R1 — real recognition:** `SignEncoder` ONNX (landmark→gloss over
21-kp windows; WLASL/CSL-corpus-trained); fingerspelling bridge; pack
evals run against `eval/phrases.json`.

**R2 — platform & providers:** Wayland input parity (ydotool/atspi),
Windows port (enigo/SendInput), local-LLM default config
(llama.cpp/vLLM endpoint presets), GPU ONNX EP auto-select.

**R3 — packs as a product:** pack format v1 freeze + docs, community
packs (ISL/BSL/Auslan…), per-user calibration export/import,
promote-3x-to-script (PRD's moat: repeated sign sequences become named
actions).

**Future (keep honest):** mobile sensor/companion via the documented
seam (doc 04 §mobile) — don't market it until something runs on a
phone. Eye/eyebrow channel as *command* input (not text — research
says 10–25wpm gaze typing loses to everything).

## 6. The one-paragraph truth to keep on the wall

**Camera fps is physics. Understanding rate is math. Commit latency is
design.** ALive's architecture already separates them: sense at sensor
rate, understand at 240Hz (labeled virtuals, real-gated recognition),
react in one frame of the display. The breakthrough is real — it was
simply never wired. Wire it, scope the vocabulary honestly, and the
demo tells itself.
