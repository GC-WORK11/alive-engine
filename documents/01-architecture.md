# ALive — Architecture Walkthrough

**What this is:** A complete tour of every line of code in this repository —
what exists, what it does, how it connects end-to-end, and (honestly) what is
written-but-not-wired. Read this top to bottom and you can rebuild the system
from memory.

**Codebase size:** ~9,700 lines across a Rust workspace (9 crates + a Tauri
shell) and a React/Three.js frontend. No git commits yet — `master` has zero
history, this document is the state of record.

---

## 1. The system in one picture

```
 ┌─────────── background (Rust, always alive) ───────────┐
 │                                                        │
 │  camera ──► motion gate ──► frame slot (latest only)   │
 │     30fps      cheap           │                         │
 │                                ▼                         │
 │                        inference thread                  │
 │                   YOLOX detect / RTMW pose (133 kp)      │
 │                                │                         │
 │                                ▼                         │
 │                         result slot ──► endpointer       │
 │                              (350/700ms pauses)          │
 │                                │                         │
 │            committed hand path ▼ trajectory matcher      │
 │                     DTW vs per-user templates            │
 │                                │                         │
 │                       gloss ──► decision layer           │
 │                    Jev (typed) or DeepSeek (tools)       │
 │                                │                         │
 │                    tool call ──► executor (dry-run first)│
 │                            journal (JSONL, redacted)     │
 └────────────────────────────────────────────────────────┘
                  │ events (tokio broadcast)
                  ▼
 ┌───────── foreground (Tauri WebView) ─────────┐
 │  orb.tsx — a single shader circle. No chrome.│
 │  Ctrl+Shift+P = prompt mode                  │
 │  Ctrl+Shift+C = computer-use mode            │
 └──────────────────────────────────────────────┘
```

The pipeline is **producer/consumer with latest-frame semantics**: slow
stages never pile up behind fast ones. Capture, inference, and the
picture-in-picture encoder are three independent loops; each publishes to a
`Mutex<Option<T>>` slot with a sequence counter, and consumers drop stale
data. That is the actual "no lag" mechanism — not speed, but *freshness*.

---

## 2. Crate map

| Crate | Lines | Role |
|---|---|---|
| `alive-core` | 212 | Shared types: AppMode, OrbState, EventBus, Config |
| `alive-perception` | 513 + 664 | Camera sources + ONNX pose/detector |
| `alive-temporal` | 396 | Streaming segmentation (endpointer) |
| `alive-track` | 1,115 | Kalman prediction + PCHIP upsampler ⚠ *unwired* |
| `alive-sign` | 296 + 417 | Language packs + trajectory matcher (DTW) |
| `alive-intent` | 757 + 349 | Decision layer: Jev client + tool-LLM client |
| `alive-act` | 1,723 | Tool registry + local executor (safe argv) |
| `alive-journal` | 206 | Append-only JSONL audit log + undo registry |
| `alive-ipc` | 166 | DTOs mirrored 1:1 by `src/types.ts` |
| `src-tauri` | 2,082 | Desktop shell: pipeline loop, commands, overlay |

Frontend: `src/` = 813 lines (OrbBar + orb shader + bridge + types).

---

## 3. Crate-by-crate, module-by-module

### 3.1 `alive-core` — the vocabulary everything shares

*File: `crates/alive-core/src/lib.rs` (212 lines)*

- **`AppMode`**: `Dictation` (default) or `Action`. Serializes as
  `snake_case` — the wire strings are `"dictation"` / `"action"`.
- **`OrbState`**: 12 states — `Idle, Activating, CameraActive,
  SignerDetected, Interpreting, IntentForming, Ready, Executing, Done,
  ConfirmNeeded, Error, Recovery`. This is the app's true state machine;
  the orb renders *from these and only these* (honest UI rule).
- **`EventKind`**: 10 event variants — `OrbStateChanged`, `PartialGloss`,
  `Committed`, `IntentDecided`, `ActionStarted`, `ActionCompleted`,
  `ConfirmRequested`, `Error`, `Info`, `PerfSample`. Tagged
  `type: snake_case` on the wire.
- **`PipelineEvent`**: `{seq, at_ms, kind}` — every event monotonic.
- **`EventBus`**: `tokio::broadcast` channel (cap ≥16) + `AtomicU64` seq.
  Rust publishes, the Tauri layer forwards to the WebView as `alive-event`.
- **`Config`** (defaults = the latency contract):
  `camera_index:0, 640×480@30`, `language_pack:"csl"`,
  `jev_endpoint:"https://api.typesafe.ai/v1/systemone"`,
  `jev_model:"jev-1.13.0"`, `act_threshold:0.85`, `confirm_threshold:0.6`,
  `tentative_pause_ms:350`, `commit_pause_ms:700`, `min_segment_ms:250`,
  `boundary_p:0.85`, **`dry_run_actions:true`** (actions are simulated
  until the user opts in — a hard safety default), `app_allowlist:[]`.
- **Tests**: default contract values; bus ordering/seq.

### 3.2 `alive-perception` — eyes in

*File: `crates/alive-perception/src/lib.rs` (513) + `onnx_backend.rs` (664)*

**lib.rs:**

- `now_ms()` — wall-clock ms, single time source.
- **AGC block** (`AGC_TARGET_MEAN=96, AGC_MAX_GAIN=3, agc_gain(luma)`):
  computes a 1–3× gain from the motion pass's luma mean — zero extra frame
  passes. ⚠ **Dead code right now**: `agc_gain` is never called.
- **`Frame`**: `{seq, at_capture_ms, w, h, data: RGB8}`. Invariant:
  `data.len() == w*h*3`.
- **`CaptureSource` trait**: `next_frame() -> Option<Frame>` + `label()`.
  Implementations:
  - `SyntheticSource` — deterministic moving bar (benches/tests).
  - `ImageDirSource` — sorted png/jpg/bmp files, once through.
  - `NokhwaSource` (feature `camera`) — wraps `nokhwa::Camera`.
    - `lock_sensor_max_mode(index)` runs `v4l2-ctl --set-ctrl
      exposure_dynamic_framerate=0,auto_exposure=3,gain=0,sharpness=5,
      backlight_compensation=1,white_balance_automatic=1` before stream
      open — keeps auto-exposure but forbids the FPS collapse in dark
      rooms. Skippable via `ALIVE_NO_SENSOR_LOCK`.
    - `open_with_format` supports MJPEG **and YUYV** (uncompressed, no
      JPEG decode). ⚠ The app still opens via `open_with` → **MJPEG**;
      YUYV exists but isn't selected by the pipeline.
    - `next_frame_timed` splits wait/decode ms for the profiler.
- **ROI ops**: `crop_rgb` (clamped, panic-free), `resize_rgb` (Triangle
  filter).
- **`Landmarks`**: `{at_ms, present, points: Vec<[f32;3]>, scores}` —
  model-native order, 133 keypoints for whole-body.
- **`LandmarkExtractor` trait**: `infer`, `keypoints`, `name`, `rescan`
  (default no-op), `tracking` (default false). `Send` bound — the
  extractor lives on the inference thread.
- `NullExtractor` — reports absent, never guesses (CI/bench honest).
- `flatten_landmarks` — xyz*N flat vector for `motion_energy`.
- Tests: synthetic determinism, crop clamp, resize dims, null honest,
  imagedir ordering.

**onnx_backend.rs** — the real vision math:

- Constants: `RTMW_W=192, RTMW_H=256, RTMW_K=133, XBINS=384, YBINS=512,
  DET_SIZE=640, BBOX_PADDING=1.25`; ImageNet-ish `MEAN/STD`.
- `Bbox` + `CropAffine::from_bbox` — mmpose top-down affine (rot-0,
  border 0, uniform scale off bbox **width**), with `to_crop`/`to_frame`
  round-trips.
- `bilinear_rgb` — 4-tap sample, border-0 (like cv2 default).
- `warp_normalize_pose_into` — warp+normalize to CHW f32 **into caller
  scratch** (hot path zero-alloc).
- `Letterbox` + `letterbox_bgr_into` — 640-letterbox, pad 114, **BGR
  order**, raw 0–255 (the detector was exported `to_rgb=false`; verified
  against `pipeline.json` ground truth).
- `simcc_decode(_into)` — exact mmpose SimCCLabelDecode, no DARK:
  `coord = argmax/2`, `score = raw max` over both axes (softmax would
  crush the peaky logits to ~1/bins — verified on bus.jpg).
- `RtmwPose` — session with `with_intra_threads`, pre-sized `warp`
  buffer, pooled `crop_pts`. `infer_bbox` → warp → `session.run` →
  extract `simcc_x/y` → decode → map back through affine. **Presence =
  ≥12 keypoints scoring ≥0.4** (raw-max calibrated on empty-room
  negatives ~0.1).
- `YoloxPerson` — `dets` [N×5] already NMS'd in-graph; score filter,
  letterbox→frame map, largest-area first.
- `WholeBodyTracker` — holds pose+det+`bbox`. `infer` returns **absent**
  when no bbox (never runs RTMW on a full empty frame — that hallucinates
  mid-score scatter and burns 64ms). Pose failure clears the bbox → app
  re-detects rather than re-posing. `rescan` = refresh_detection.
- Tests: affine round-trip, SimCC peak math, letterbox math, preprocess
  shapes, **preprocess bit-identical** (scratch reuse changes no ULP).

### 3.3 `alive-temporal` — where one sign ends and the next begins

*File: `crates/alive-temporal/src/lib.rs` (396)*

- `LandmarkFrame {at_ms, present, points}` — `present=false` means hands
  left the frame: a *cancel* signal, not silence.
- `motion_energy(prev, cur)` — mean |Δ| per coordinate.
- `RingBuffer<T>` — fixed ring, oldest-first iteration (window ops).
- `EndpointConfig` — `tentative_pause 350ms`, `commit_pause 700ms`,
  `min_segment 250ms` of *active* motion, `boundary_p 0.85`,
  `low_motion 0.02`, `max_pause 2000ms`.
- `EosScorer` trait — pluggable P(end-of-sign); the learned TCN slots in
  later. `LowMotionEos` ships now: fraction of recent frames under the
  stillness floor — converges to 1.0 during a real pause.
- `Endpointer` — `Idle → Signing → Pausing` state machine, one frame at
  a time, fully causal:
  - `Tentative` fires once at first 350ms pause (speculative UI only).
  - `Commit` fires at ≥700ms pause **AND** `eos_p ≥ boundary_p`, with
    `active_ms ≥ 250` (debounce — flicker never commits).
  - `Cancelled` on `present=false` mid-segment.
  - `max_pause` abandons stale segments silently (no zombie commits).
- Tests: tentative→commit timing at 30fps cadence, flicker never commits,
  drop cancels, low-eos blocks commit, ring ordering, motion_energy math.

### 3.4 `alive-track` — the breakthrough that's written but not wired

*Files: `kalman.rs` (401), `upsample.rs` (704), `lib.rs` (10)*

This is the math the user calls the "70% breakthrough". It compiles, it's
thoroughly tested — **and nothing in the app calls it.** `src-tauri`
lists it as a dependency but `main.rs` never constructs an `Upsampler`,
`HandTracker`, or `OneEuro`. The `OverlayFrame` it was built to feed is
never populated. This is the missing 30%.

**kalman.rs** — predictive hand tracking (UI/overlay/reaction ONLY):
- `AxisKalman` — scalar constant-velocity Kalman per axis, Joseph-form
  covariance update, `predict_dt` (clamped dt ≤0.5s, Q_pos·dt +
  Q_vel·dt³/3 acceleration model), `update` with innovation clamp ±150px
  (one bad frame can't sling the track).
- `TrackState`: `Uninit → Tentative → Confirmed → Coast`.
- `HandTrack` — 2 axes + confidence EMA (α=0.3) + gates:
  - teleport gate: jump >170px from prediction → re-init (no velocity
    sling).
  - confirmed after ≥2 updates and conf_ema>0.15.
  - `mark_missed`: >120ms gap → Coast, >400ms → reset.
  - `estimate_at(t)` — predicts forward ≤400ms; `velocity()`,
    `uncertainty()` (1σ px) for overlay gating.
- `HandTracker` — both hands.
- `OneEuro` — adaptive smoother for display scalars (min_cutoff + β·|dx|;
  still = smooth, fast = responsive).
- Tests: constant-velocity convergence beats zero-order-hold ≥2×, still
  hold doesn't drift, teleport re-inits, coast-out, OneEuro jitter/step.

**upsample.rs** — causal temporal super-resolution, ~10Hz poses → 240Hz
understanding ticks:
- `TICK_US=4167` (240Hz grid), `MAX_BRACKET 200ms`, `MIN_BRACKET 8ms`,
  `STILL_PX=2`, `VCAP 2400px/s`.
- `Centroid {x,y,score,n_keys}`, `PoseSample {t_cap_us, present, a, b}`
  (two hands), `Tick {t_cap_us, prov, a, b}`.
- **`Provenance`**: every tick is `Real{seq}` or `Virtual{left_seq,
  right_seq, alpha, scheme}` — downstream always knows measured vs
  interpolated. `Scheme = Pchip | Linear | Hold`.
- `Upsampler.push(sample)` → virtual ticks then the real tick, per
  bracket. Per-hand channels with previous + previous-previous knots
  (PCHIP tangents).
- **Anti-hallucination rules R1–R7** (the honest-math core):
  - R1 per-hand independence — a missing hand kills that channel's
    virtuals; no cross-hand borrowing.
  - R2 caller gates trail entry on centroid confidence.
  - R3 confidence blend — low-confidence brackets blend PCHIP → linear.
  - R4 stillness (<2px) → `Hold` ticks, never cubic wiggle.
  - R5 bracket >200ms → endpoints only + tangent reset.
  - R6 velocity clamp — virtual speed > max(3× secant, 2400px/s) → whole
    bracket linear.
  - R7 `real_span_ok` — recognition needs ≥3 real poses spanning ≥250ms;
    virtuals alone can never trigger a commit.
- Fritsch–Carlson tangents (`fc_interior`, `fc_endpoint` — non-uniform
  grid), `hermite` evaluator.
- Causality: virtuals emit when the right endpoint arrives (~one bracket
  of lag, honest, inside the 350/700ms endpointing budget). Prediction
  (the future) belongs to `kalman` and is fenced to UI.
- Tests (suite A): circle RMSE <4px across all 24 decimation phases,
  flick overshoot <6px (PCHIP monotonicity), hold wiggle ≤1.5px, V-turn
  apex fidelity <8px + no pre-corner ringing, cost <25µs/tick.

### 3.5 `alive-sign` — motion → gloss

*Files: `lib.rs` (296), `matcher.rs` (417)*

**lib.rs — language packs as data:**
- A pack is a directory: `pack.toml` + `lexicon.json` + `intents.json` +
  `eval/`. Adding a sign language = adding a directory, zero core changes.
- `PackManifest {id, display_name, tier, version, adapter}`,
  `EntryKind {Command, Dictation, Fingerspell, Control}`,
  `LexiconEntry {gloss, aliases, kind, intent}`,
  `IntentDef {intent, action, risk_tier, slots, patterns}`.
- `LanguagePack::load(dir)` — parses all three files, builds a
  lower-cased `alias_index` (gloss + every alias).
- `match_gloss` — calibrated-vocabulary fast path, case-insensitive.
- `intent_options()` — catalog + `"dictate"` + `"cancel"`, ≤255 (Jev
  Choice protocol cap).
- `GlossHypothesis`/`NBest` — sort desc, top-N, renormalize to Σ1;
  degenerate input → uniform, never NaN.
- `SignEncoder` trait + `NullEncoder` — the future learned encoder's
  seam; null honestly returns `UNKNOWN` at confidence 1.
- Tests: ASL + CSL pack loading, bilingual alias matching, lexicon↔intent
  resolution, ASL/CSL parity (26 entries each), NBest math.

**matcher.rs — calibrated motion commands (HONEST SCOPE):**
- Header says it plainly: this recognizes **motion-distinct command
  gestures** (swipe, shake, circle, push) from hand-centroid paths,
  calibrated per user with ≥3 samples. **It is not general sign-language
  recognition** — that needs the trained encoder + corpus (M1+).
  Unknown motions are *rejected, never guessed*.
- `HAND_A = 91..112`, `HAND_B = 112..133` — COCO-WholeBody ranges
  (verified on bus.jpg).
- `hand_centroid(_full)` — score-weighted centroid; needs ≥6 keys ≥0.4.
  Returns (x,y,mean_score,n_keys) for PCHIP confidence blending.
- `resample` — arc-length → exactly 32 points.
- `normalize` — centroid origin + RMS scale. Translation/scale invariant,
  **direction sensitive** (left swipe ≠ right swipe).
- `dtw(a,b,band=4)` — Sakoe-Chiba banded DTW, mean cost per aligned step.
- `TemplateBank` — ≤5 normalized exemplars per gloss (FIFO),
  `reject_threshold` (default 0.45 in tests), `best()` returns
  (gloss, best_d, second_d) for margin-aware callers; `recognize()` →
  `score = 1/(1+d)` or None on reject. JSON persist + `glosses()` map.
- Tests: shift/scale invariance, direction sensitivity, shape separation
  with margin, recognize+reject (squiggle/short/reverse), JSON round-trip
  + FIFO cap, centroid weighting/gating.

**Packs on disk**: `packs/asl` (26 entries, tier A), `packs/csl`
(26 entries — 打开浏览器/打开邮件/回复是/复制/截图/输入文字/确认/命令模式…
with English aliases, tier C). `Config.language_pack` defaults to `csl` —
chosen for signer-population size, **not validated** by any native-signer
testing. Treat as provisional.

### 3.6 `alive-intent` — the decision layer

*Files: `lib.rs` (757), `tool_llm.rs` (349)*

**lib.rs — Jev-style typed decisions:**
- `JevQuestion` — `choice` (1..=255 options), `score` (2..=10 levels),
  `noul`. Builder asserts = protocol guarantees.
- `DecisionRequest {state, questions}` — all questions in ONE call.
- `alive_questions()` — the standard 8: intent Choice + route Choice
  (deterministic/llm/human) + 4 Nouls (addressing_computer,
  needs_confirmation, is_risky, is_multi_step) + 2 Scores
  (confidence, risk).
- `tool_dispatch_questions()` — the **generic** variant: one Choice over
  the 15 tool names + `"none"` + the same safety Nouls. This is what
  replaces the hardcoded intent catalog.
- Tolerant parsing (`parse_jev_response` handles answers as map / array /
  flat; `parse_probs` accepts map or {option|label|choice,
  probability|p} entries; `noul_of`, `score_of`). Unknown shapes are
  **errors, never guesses**.
- `DecisionProvider` trait (`decide`, `name`) — the seam.
- `JevClient` — reqwest POST, bearer auth, 10s timeout, key from
  `ALIVE_JEV_API_KEY`. `decide()` → `Decision`; `decide_dispatch()` →
  `ToolDispatch`.
- `MockProvider` — scripted decisions for tests/offline dev.
- **`calibrated_confidence(n, peak) = (n·peak − 1)/(n − 1)`** —
  chance-corrected peak; `route()` → Act ≥0.85 / Confirm ≥0.6 / Ask.
- `route_decision` — takes the *pessimist* of calibrated peak and vendor
  confidence; any confirm/risk flag or risk_score≥4 demotes Act→Confirm;
  `addressing_computer` gate ≥0.6.
- Tests: PRD confidence formula, route thresholds, question shapes
  (8 + dispatch-5), parser accepting all three wire forms, dispatch
  happy/error paths.

**tool_llm.rs — the System-Two half:**
- `ToolLlmClient` — OpenAI-compatible chat-completions client. Defaults:
  `https://api.deepseek.com/chat/completions`, model `deepseek-flash`,
  env `ALIVE_LLM_ENDPOINT/MODEL`, key `ALIVE_LLM_API_KEY` or
  `DEEPSEEK_API_KEY`. Missing key → honest Transport error. ⚠ The model
  id `deepseek-flash` (comment claims "V4.1-Flash") is **unverified**
  against DeepSeek's live catalog — confirm before demo.
- `chat(messages, tools)` → `LlmReply {content, tool_calls}`; strict
  parse — missing choices/message/function/name or unparseable arguments
  are errors.
- Message builders: `sys_msg`, `user_msg`, `assistant_msg` (echoes
  content + tool_calls verbatim for multi-step loops), `tool_msg`.
- `OPERATOR_SYSTEM_PROMPT` — terse operator rules: do exactly what was
  asked, prefer fewer steps, never invent paths/URLs/apps, stop on
  refusal, plain text when done.
- Tests: reply parsing (text-only, multi-tool, bad shapes), transcript
  builders round-trip, **live stub-HTTP wire test** (OpenAI shape on the
  wire verified without a vendor key), missing-key honesty.

### 3.7 `alive-act` — hands on the machine (with rules)

*File: `crates/alive-act/src/lib.rs` (1,723)*

Hard rules enforced in code: **no shell, ever** (argv only, model output
never becomes argv[0]); `open_app` requires allowlist; URLs
http/https/mailto only; dry-run plans+journals without executing; every
execution journaled with redacted args; undo payloads one-shot.

- **`ToolId` — the closed verb set (15):**
  `open_app, open_url, file_search, browser_search, ai_query,
  clipboard_copy, screenshot, type_text, key_press, mouse_move,
  mouse_click, mouse_drag, scroll, wait_ms, cursor_position`.
  - `all()` stable order = Jev options + LLM schemas come from ONE list.
  - `risk_tier`: tier 2 (externally visible, behind confirm) =
    open_app, type_text, key_press, mouse_click, mouse_drag, scroll.
  - `tool_schemas()` — OpenAI function schemas generated from the same
    arg shapes the executor reads (`additionalProperties:false`).
- `ActionRequest {action_id, intent, tool, args, risk_tier}`;
  `ActionOutcome`: `Done{summary,undo,verified}` |
  `NeedsConfirmation{preview}` | `Refused{reason}` | `Failed{error}`.
- **`availability()`** — honest degrade: probes PATH for xdg-open/gio,
  wl-copy/xclip/xsel, grim/gnome-screenshot/scrot/import,
  wtype/xdotool/ydotool; reports `session_type` + `desktop` + per-tool
  `via`. The UI/agent sees what *this* machine can actually do.
- **`LocalExecutor{dry_run, allowlist}`**, per-tool handlers:
  - `open_app` — name sanitization (no `/ ; & | $ \`` or spaces) +
    allowlist → detached spawn (null stdio).
  - `open_url` — scheme + ≤2048 → `xdg-open` or `gio open`.
  - `file_search` — spawn_blocking walk, depth 5 / 5000 entries / 20
    hits, skips .git/node_modules/target/Trash etc.
  - `browser_search` — DDG URL → `open_url` (composability).
  - `ai_query` — `ALIVE_MOCK_LLM=1` only; otherwise honest failure.
  - `clipboard_copy` — pipes stdin to wl-copy/xclip/xsel (≤100k).
  - `screenshot` — grim→gnome-screenshot→scrot→import →
    `/tmp/alive-shots/`, yields `delete_file` undo.
  - `type_text` — `wtype` (Wayland) or `xdotool type --delay 0`
    (≤5000 chars).
  - `key_press` — `canonical_key_combo` allowlist-grammar (named keys +
    1–4 tokens, no `;`/`$`/junk) → `xdotool key --clearmodifiers`.
  - `mouse_move/click/drag/scroll` — xdotool argv; coords u32 ≤32767
    (floats/negatives/overflow **refused**, never silently clamped);
    buttons left/middle/right=1/2/3; wheel 4–7; scroll 1–20 notches.
  - `wait_ms` ≤30s (built-in sleep); `cursor_position` parses
    `getmouselocation --shell`.
- `redact_args` — type_text/clipboard_copy/ai_query/key_press journal
  only `{"chars": n}` — typed text never hits the log.
- `execute_and_journal`, `apply_undo` (noop/delete_file only; unknown op
  = error).
- Tests: availability probe, dry-run purity, URL scheme table, app
  allowlist + injection-name refusals, file_search fixture, ai_query
  honesty, journal wiring + real undo round-trip, urlencode, redaction,
  **registry consistency** (names↔schemas↔required-args all aligned),
  key-combo grammar table, mouse arg normalizers.

### 3.8 `alive-journal` — the audit spine

*File: `crates/alive-journal/src/lib.rs` (206)*

- `ActionStatus`: Proposed/AwaitingConfirm/Done/Undone/Failed/Refused.
- `UndoOp {op, payload}` — opaque; only the creating executor interprets.
- `JournalEntry {seq, at_ms, action_id, intent, tool, args_redacted,
  status, summary, undo, verified}`.
- `Journal::open` (append-create), `append` (assigns seq + timestamp,
  stores undo in memory, writes+flushes line), `take_undo` (**one-shot by
  design** — undo can never resurrect stale state), `read_all` (skips
  blanks). Path: `alive-state/journal.jsonl`.
- Tests: seq/timestamp assignment, one-shot undo, blank-line tolerance.

### 3.9 `alive-ipc` — the wire contract

*File: `crates/alive-ipc/src/lib.rs` (166)*

DTOs the TypeScript mirrors 1:1 (`src/types.ts`): `StatusDto` (14 fields
incl. `upsample_hz`), `HypDto`, `DecideRequestDto`, `DecideResponseDto`,
`ExecuteRequestDto`, `OutcomeDto`, `AgentPendingDto` (opaque pending
approval round-trip), `AgentRequestDto` (`text, app, url, history,
pending, approve, max_steps` — stateless server, client carries state),
`AgentStepDto`, `AgentResponseDto` (`done|awaiting_confirm|ignored|
error`), `ToolDto`, `AvailabilityDto`, `JournalEntryDto` +
`From<&JournalEntry>`.

### 3.10 `src-tauri` — the shell that owns the threads

*File: `src-tauri/src/main.rs` (2,082) + `src/bin/demo.rs` (309)*

**Shared state** (one `Shared` struct): orb_state, mode, bus, pack,
journal, provider, executor, **frame slot** (`Mutex<Option<Frame>>` +
`AtomicU64 seq`), **result slot** (same shape for `Landmarks`), motion
EMA, armed flag, fps/capture/landmark EMA meters, `ticks_total`,
`upsample_hz`, hand trail, **`overlay: Mutex<OverlayFrame>` ⚠ never
written**, overlay_mode, action_seq, pending_action.

- `OverlayFrame {w,h,seq,segs,ribbons,leads}` — designed for skeleton
  segments + PCHIP trail ribbons + Kalman lead dots. **Dead: only
  `OverlayFrame::default()` exists**; the PiP thread draws an empty
  frame. The fields exist; the writers don't.
- Perf meters: `note_frame/note_infer/note_ticks` — EMA-windowed rates.
  `note_ticks` gets `n=1` per real pose only → `upsample_hz` currently
  reports ~pose rate, not 240Hz.
- **Tauri commands**: `get_status`, `set_mode`, `decide`, `execute`,
  `undo_action`, `journal_recent`, `arm_camera`, `disarm`,
  `overlay_show/hide/state`, `overlay_commit_type`, `agent_act`,
  `tool_availability`, `pack_info`, `templates` (list/add/clear).
- **`aliveframe://` custom protocol** — serves the latest PiP JPEG from
  the frame slot (the frontend asks for `frame.jpg?seq=N`).
- **Global shortcuts** (`tauri-plugin-global-shortcut`): `Ctrl+Shift+P`
  → prompt mode, `Ctrl+Shift+C` → computer mode — both show the overlay
  + emit `overlay-mode` to the WebView.
- **`run_pipeline`** (~line 1103): opens camera at cfg 640×480@30 →
  `SyntheticSource` fallback (honest degrade); loads `WholeBodyTracker`
  if both ONNX files exist; ONNX intra-threads = `available_parallelism
  −2, min 4`; spawns the inference thread.
- **Motion gate** (`GatePolicy` + `motion_score`): every-4th-pixel
  green-channel delta, noise floor 12, hysteresis `MOTION_WAKE=0.003 /
  MOTION_SLEEP=0.0015`, >45% global-shift or coherent large shift →
  exposure/scene change (not hand motion). Wake latch, heartbeat (~2s),
  rescan on track loss, **no synthesized motion during gaps**.
- **Inference thread**: waits on frame-slot seq changes → motion-gated →
  `extractor.infer(latest_frame)` → publishes `Landmarks` to result slot.
- **Main loop**: consumes result-slot seq → `hand_centroid` trails →
  `endpointer.push` → `Tentative` → `PartialGloss` event → `Commit` →
  `handle_commit`.
- **`handle_commit`** (~line 1547): gathers per-hand paths over the
  segment → dominant path → `TemplateBank::recognize` → `NBest` → Jev
  `decide` (if keyed) else `mock_decide` (lexicon-driven) →
  `route_decision` → publishes `Committed` + `IntentDecided` → executor
  (dry-run) + journal.
- **`agent_act`** (~mid-file): `AgentRequestDto` → optional Jev
  `decide_dispatch` triage → `ToolLlmClient.chat` loop (≤max_steps) →
  validate every call against `ToolId` → execute via `LocalExecutor`
  (dry-run/confirm gating, `AgentPendingDto` round-trip for approvals) →
  journal + `ActionStarted/Completed` events.
- **PiP thread**: frame slot → box-downsample to ~320×180 → overlay draw
  (segs/ribbons/leads — currently empty) → JPEG q70 → back to frame slot;
  ~80ms tick. Encoding is off the hot path.
- **App init** (~line 1677): stderr logger → `LanguagePack::load` →
  `Journal::open(alive-state/journal.jsonl)` → load persisted templates
  (`alive-state/templates.json`) → shared state → bridge
  `EventBus→alive-event` → register shortcuts → overlay window from
  `tauri.conf.json` (180×180, transparent, undecorated, always-on-top,
  skip-taskbar, hidden, focusable, non-resizable).
- **`demo.rs`** — headless end-to-end proof, no WebView: matcher
  recognize+reject, pack load+gloss→intent, camera+detect+pose with
  measured ms, decide via mock (+ live Jev when `ALIVE_JEV_API_KEY`),
  live file_search + journal + undo round-trip, endpointer
  tentative→commit. Prints one JSON verdict, exits 0/1.

### 3.11 Frontend — one orb, no chrome

- **`src/main.tsx`** (10): mounts `OrbBar`.
- **`src/OrbBar.tsx`** (136): `mode: 'prompt'|'computer'|null`,
  `AgentState` for the orb. Listens `overlay-mode` + `overlay_state` to
  wake; subscribes `alive-event`: `partial_gloss/action_started` →
  `listening`; `committed` → prompt mode calls `overlay_commit_type`
  (types into the focused window) then auto-hides, computer mode calls
  `agent_act` (history+pending carried client-side); `action_completed/
  error` → timed hide. `Esc` hides; `Enter` approves a pending tool call
  (`approve:true`). Renders `<div.orb-stage><Orb/></div>` — that's the
  entire DOM.
- **`src/components/ui/orb.tsx`** (499): react-three-fiber `Canvas` +
  one `CircleGeometry(3.5,64)` + custom `ShaderMaterial`.
  - Uniforms: `uTime, uAnimation, uColor1/2, uInputVolume,
    uOutputVolume, uOpacity, uInverted, uOffsets[7]` +
    `uPerlinTexture` (bundled `src/assets/perlin.png` — no CDN).
  - `splitmix32` seeded RNG → 7 per-oval phase offsets.
  - `AgentState` drives target in/out volumes (idle breathing /
    listening / talking curves); `useFrame` lerps volume + color +
    animation speed — the orb's motion is the state readout.
  - Fragment shader: polar uv → 7 perlin-modulated ovals orbiting
    (`drawOval` soft edges + flattened gradients) + 2 noise rings
    (`sharpRing`,`smoothRing` screen-blended white) + `flow` texture
    distortion scaling with `uOutputVolume` → 4-stop `colorRamp`
    (black→color1→color2→white) with `uInverted` flip for dark mode
    (MutationObserver on `<html>.dark`).
  - `webglcontextlost` → `forceContextRestore` (Wry/WebKit quirk guard).
- **`src/types.ts`** (126): mirrors the IPC DTOs + event tag union.
- **`src/tauri.ts`** (42): `isTauri()` via `__TAURI_INTERNALS__`,
  `MOCKS` for browser dev (clearly labeled), `cmd()` invoke wrapper,
  `onAliveEvent()` → `alive-event`, `frameUrl()` for the protocol.
- **`src/styles.css`** (26): transparent everything, `overflow:hidden`,
  `.orb-stage` fixed fill.

---

## 4. Protocols & boundaries

- **Event bus** — `EventBus` (tokio broadcast) → `alive-event` Tauri
  emit → `onAliveEvent` listener. Seq-monotonic; the orb renders truth.
- **Frame protocol** — `aliveframe://localhost/frame.jpg?seq=N` serves
  the latest PiP JPEG (custom protocol; the WebView never sees raw
  frames).
- **IPC** — `invoke()` commands ↔ `alive-ipc` DTOs ↔ `src/types.ts` (the
  comment says it: *if Rust changes, this changes*).
- **Agent loop** — stateless server: the client carries `history` +
  `pending` verbatim; `AgentPendingDto` is the opaque approval token;
  `approve:true` resumes.
- **Journal** — `alive-state/journal.jsonl`, one JSON object per line,
  redacted args, seq-ordered; `alive-state/templates.json` for the
  matcher bank.
- **Packs** — `packs/{id}/{pack.toml,lexicon.json,intents.json,eval/}`;
  `alive-state/` holds runtime state (git-ignored territory).

## 5. The honest wiring diagram (what's real vs what's missing)

| Subsystem | Written | Wired | Notes |
|---|---|---|---|
| Camera capture (MJPEG) | ✅ | ✅ | YUYV path exists, not selected |
| Sensor max-mode lock | ✅ | ✅ | v4l2-ctl best-effort |
| AGC dark-room gain | ✅ | ❌ | `agc_gain` never called |
| Motion gate + hysteresis | ✅ | ✅ | exposure-aware, heartbeat |
| Latest-frame/result slots | ✅ | ✅ | freshness > throughput |
| YOLOX→RTMW track+pose | ✅ | ✅ | ~105ms pose p50 @12 threads |
| Endpointer (350/700) | ✅ | ✅ | causal, tested |
| DTW template matcher | ✅ | ✅ | command gestures, NOT sign lang |
| Pack loader (ASL/CSL) | ✅ | ✅ | CSL default unvalidated |
| Jev typed decisions | ✅ | ✅ | needs `ALIVE_JEV_API_KEY` |
| DeepSeek tool loop | ✅ | ✅ | needs `ALIVE_LLM_API_KEY`; model id unverified |
| 15-tool executor + journal | ✅ | ✅ | dry-run default |
| **Kalman prediction** | ✅ | ❌ | overlay leads never populated |
| **240Hz upsampler** | ✅ | ❌ | `Upsampler` never constructed |
| **Skeleton/ribbon overlay** | ✅ | ❌ | `OverlayFrame` never written |
| Sign-language encoder | trait only | — | `NullEncoder` honest UNKNOWN |
| TCN EOS scorer | trait only | — | `LowMotionEos` ships |
| Composer/typing UX | partial | ⚠ | commits type into focus; no visible composer yet |

**The 30% the project never got:** instantiate `Upsampler` +
`HandTracker` in `run_pipeline`, feed `PoseSample`s from the result slot,
count `Tick`s into `note_ticks`, write `OverlayFrame` (segs from the 133
keypoints, ribbons from upsampled centroids, leads from
`HandTrack::estimate_at(now)`), call `agc_gain` from the motion pass's
luma mean, and prefer YUYV at ≤480p. All the hard math already exists and
is unit-proven.

## 6. Test coverage inventory

Every crate carries `#[cfg(test)]` suites (unit-level, deterministic, no
network/camera except where noted): core (contract defaults, bus order),
perception (determinism, crop/resize, null honesty, affine/SimCC/letterbox
math, **bit-identical preprocess**), temporal (endpoint timing contracts,
flicker/drop/eos gates), track (Kalman convergence/hold/teleport/coast +
PCHIP suite A1–A5), sign (pack loads + parity + matcher invariance/
direction/reject/round-trip), intent (formula, routes, wire-shape
tolerance, stub-HTTP round-trip, missing-key honesty), act (refusal
tables, registry consistency, undo, redaction), journal (seq, one-shot,
blanks). Plus `demo.rs` = the real end-to-end runner (camera + ONNX +
executor + journal, JSON verdict).

## 7. Known sharp edges (fix list, ordered)

1. **alive-track unwired** — the whole point of the breakthrough;
  see §5 table. Highest-leverage change in the repo.
2. Trajectory matcher ≠ sign language — scopes the demo honestly:
  "calibrated command grammar" unless/until a real encoder lands.
3. `deepseek-flash` model id + Jev endpoint — verify against live
  catalogs before the demo; both fail closed (honest errors) today.
4. CSL default — population-largest pack ≠ right default for this user;
  pack selection UI/onboarding missing.
5. `overlay_commit_type` types into whatever has focus — the "composer"
  is the OS itself; a visible capsule composer doesn't exist yet.
6. Orb shows AgentState only; `OrbState` (12 core states) isn't mapped
  to visuals — the honest-state rule is half-implemented.
7. No commits — the entire history is this repo's working tree.
