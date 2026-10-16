# ALive — Tech Stack (current + where to extend)

The rules of this document: every layer lists (a) what's in use *now*,
(b) the measured numbers behind it, (c) where a better/alternative
technology plugs in and why. Speed and cost are called out per layer.

---

## 1. Stack map — what's actually running today

| Layer | Technology | Version/Note |
|---|---|---|
| Shell | **Tauri 2** (custom-protocol feature) | 2.x; single 180×180 transparent always-on-top overlay window; CSP null |
| Frontend | **React 18 + TypeScript + Vite 5** | 18.3.1 / 5.5.4 / 5.4.8 |
| Orb render | **Three.js + @react-three/fiber + drei** | 0.186 / 8.18 / 9.122; custom GLSL shader orb; `perlin.png` bundled |
| OS glue | `tauri-plugin-global-shortcut` 2.x | Ctrl+Shift+P / Ctrl+Shift+C |
| Language | **Rust 2021**, workspace `resolver=2` | 9 crates + `src-tauri` bin |
| Async | **tokio 1** | rt-multi-thread, sync, time, net; broadcast event bus |
| Camera | **nokhwa 0.10** → V4L2/UVC | MJPEG (active) + YUYV (available, unused); `v4l2-ctl` sensor lock |
| Inference | **ONNX Runtime via `ort` 2.0.0-rc.13** | CPU, intra-op threads = cores−2 (min 4) |
| Models | **RTMW-l 256×192** (133 keypoints, 228MB) + **YOLOX-m humanart** (101MB) | `models/` + SHA256SUMS |
| Images | `image` 0.25 | JPEG decode/encode, resize, warp |
| HTTP (LLM) | `reqwest` | Jev + OpenAI-compatible clients, 10s/60s timeouts |
| Decision (System 1) | **Jev-style typed API** — `alive_intent::JevClient` | `POST {model,state,questions}` → typed answers; endpoint `api.typesafe.ai/v1/systemone`, key `ALIVE_JEV_API_KEY` |
| Agent (System 2) | **OpenAI-compatible tool loop** — `ToolLlmClient` | default `api.deepseek.com` + `deepseek-flash` (⚠ id unverified), key `ALIVE_LLM_API_KEY`/`DEEPSEEK_API_KEY` |
| Action exec | **argv-only subprocess**: xdotool, wtype, xdg-open/gio, wl-copy/xclip/xsel, grim/gnome-screenshot/scrot/import | X11-first, Wayland partial |
| Journal | JSONL append-only (`alive-state/journal.jsonl`) | redacted args, one-shot undo |
| Testing | `cargo test --workspace` + `alive-demo` binary | unit suites per crate + headless E2E |

---

## 2. Perception layer — speed budget & where to gain

**Measured on this machine (CPU, RTMW-l + YOLOX-m):**

| Stage | p50 | Note |
|---|---|---|
| capture wait | ~0.2–0.5ms | latest-frame slot, never blocks |
| MJPEG decode | ~2.3ms | YUYV would drop this to ~0 + zero artifacts at ≤480p |
| PiP resize+JPEG | ~9ms | off hot path, 80ms tick, 320×180 q70 |
| pose on tracked bbox | **~105ms** | dominates; intra-threads 10–12 plateaued |
| full-frame detect | ~430ms | runs only on acquire/loss, background-cadenced |
| **overlapped effective rate** | **~10–13Hz pose** | throughput = slowest stage, not the sum |

**Integration points, ordered by payoff:**

1. **Wire what exists (free):** feed `PoseSample`s into `Upsampler`
   (240Hz causal ticks, <25µs/tick CPU cost) + `HandTracker` (Kalman
   predict-to-now for overlay leads). This is the "sense at 30,
   understand at 240" promise — *already built, needs ~100 lines of
   wiring* in `run_pipeline`.
2. **Switch ≤480p capture to YUYV** — one enum in `open_with_format`;
   kills JPEG decode latency and block artifacts.
3. **GPU execution provider** — `ort` supports CUDA/TensorRT/CoreML
   features; RTMW-l on even an iGPU drops pose to ~15–30ms → real-time
   pose at sensor rate. Big single jump available.
4. **Smaller/specialized model path** — RTMW-m or a hand-only crop model
   (MediaPipe-class 21-kp hand net on the hand ROI) for the
   centroid/motion path, keeping whole-body only when a person is
   tracked. Cheaper per-tick where it matters.
5. **DirectML/NNAPI** for Windows/Android ports via the same `ort` seam —
   `LandmarkExtractor` is already the abstraction.

## 3. Temporal/track layer — already SOTA-shape, zero cost

- `alive-temporal`: pure Rust, no deps. Latency contract lives here
  (350/700ms). Future: plug a learned `EosScorer` (small TCN over
  motion windows, ~1–2MB) behind the existing trait — no structural
  change.
- `alive-track`: pure Rust math. PCHIP = Fritsch–Carlson tangents +
  Hermite; Kalman = constant-velocity Joseph-form. Cost ~µs/tick.
  No replacement needed — needs *wiring*, not swapping.

## 4. Recognition layer — the real cost/perf decision

- **Today:** DTW trajectory templates — O(templates × 32²-band) per
  commit, microseconds, zero model files. Correct choice for command
  grammar.
- **For real sign language (the honest path):** a landmark→gloss
  encoder — e.g., a TCN or small transformer over 21-kp×T hand windows
  (+optional face/torso context), trained on WLASL/How2Sign-class data,
  exported to ONNX, and dropped behind the **already-existing**
  `SignEncoder` trait. Vocab 300–2000 glosses is a realistic on-CPU
  budget (few-ms infer per commit). **This is THE feature-level
  decision** — it turns ALive from "gesture commands" into "sign input
  method".
- **Fingerspelling bridge (cheap intermediate win):** static handshape
  classifier per frame (logreg/小型 CNN over 21-kp), majority-vote over
  ~10 frames → letters. Unlocks arbitrary words before full CSL/ASL.

## 5. Decision/agent layer — providers are env-swappable

Current wiring: JevClient (typed System-1 triage) + ToolLlmClient
(generic System-2 loop). Both are *endpoint + key* configured:

- **Jev** (`ALIVE_JEV_API_KEY`): claimed ~70–500ms/decision,
  ~$0.042/M input tokens → **~1¢/user-hour** at a few decisions/min.
  Keep as the typed-triage default.
- **DeepSeek** (`ALIVE_LLM_API_KEY`): OpenAI-compatible, tool calling,
  cheapest capable frontier-class API. **Verify `deepseek-flash` id
  before demo** — fall back to a confirmed id (`deepseek-chat`).
- **Local System-1 (offline path):** any OpenAI-compatible server
  (llama.cpp server, vLLM, ollama) pointed at `ALIVE_LLM_ENDPOINT` works
  *today with zero code* — the client is generic. For Jev-protocol local
  models (StartLux-Decision-class, GGUF/vLLM), point a Jev-compatible
  `/systemone` shim at `jev_endpoint` — the wire shape is already
  `{model,state,questions}`.
- Cost model: perception+temporal+track+match = **$0 (all local)**;
  decisions = fractions of a cent each; agent turns = DeepSeek token
  cost only when computer mode is used. **A full local-LLM config =
  $0 marginal cost.**

## 6. Action layer — X11 today, Wayland honest

- argv-only execution through xdotool/wtype/xdg-open/gio/clip tools —
  no shell, no injection surface. `availability()` probes PATH and
  reports per-tool `via` → the agent sees what *this* box can do.
- **Integration:** add an `atspi`/ydotool path + PipeWire screenshot
  for pure-Wayland parity (xdotool calls are X11-only); on Windows,
  PowerShell-free equivalents via `SendInput` through a tiny helper or
  the `enigo`/`rdev` crate behind the same executor seam.
- Speed: each tool is one fork+exec (~1–5ms) — action latency is not a
  bottleneck; LLM planning time is (bounded by max_steps + 60s timeout).

## 7. Shell/UI layer — cheap to extend, keep it thin

- Tauri gives a ~3–6MB binary (vs ~150MB Electron), WebKit on Linux.
  The overlay is a single transparent undecorated window; the frame
  channel is the `aliveframe://` custom protocol (no HTTP server).
- Three.js orb is GPU-shader-driven; CPU cost ~nil; the WebGL-context
  restore handler guards Wry quirks.
- **Integration:** expanding "capsule" = CSS/window resize on the same
  window (or a second window for composer text); PiP feed already
  exists for the debug view. Keep frontend = chrome only; data path
  stays Rust.

## 8. Cost sheet — what it actually costs to run/scale

| Item | Dev | End-user marginal cost |
|---|---|---|
| Perception/track/match | free (local CPU/GPU) | 0 |
| Jev decisions | $0.042/Mtok | ~1¢/hour heavy use |
| DeepSeek agent turns | ~$0.10–0.30/Mtok class | cents/hour, only in computer mode |
| Local LLM alternative | free weights (CC-BY-NC for some — check license) | 0 (electricity) |
| Packaging | cargo tauri build → deb/rpm/AppImage | ~10–20MB binary + 330MB models |
| Models | RTMW-l 228MB + YOLOX-m 101MB | one-time download; can ship a fetcher |

## 9. Where each upgrade slots (the "integration map")

```
camera ─► [YUYV now] ─► [GPU EP / hand-crop model] ─► pose
pose ─► [wire Upsampler+Kalman] ─► 240Hz ticks ─► endpointer
segment ─► [DTW now | SignEncoder ONNX next | fingerspell bridge] ─► gloss
gloss ─► [Jev | any /systemone-compatible local | heuristic] ─► intent
agent ─► [DeepSeek | any OpenAI-compatible server incl. llama.cpp] ─► tools
tools ─► [xdotool X11 | wtype/atspi Wayland | enigo Windows] ─► OS
```

Every seam is already an interface (`CaptureSource`, `LandmarkExtractor`,
`EosScorer`, `SignEncoder`, `DecisionProvider`, `LocalExecutor`) —
the architecture was built to swap, not to fork.
