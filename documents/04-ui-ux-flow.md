# ALive — Frontend, UI/UX Flows

## Design law (non-negotiable)

1. **One orb, zero chrome.** The user-facing surface is a floating
   180×180 transparent always-on-top window containing exactly one
   shader-driven orb. No panels, no buttons, no tabs, no debug text in
   the user path.
2. **The orb renders truth.** Its motion is driven by real pipeline
   states — `AgentState: null | thinking | listening | talking` today,
   wired from `alive-event`s (partial_gloss/action_started → listening,
   committed→agent → talking, done → idle/fade). It never plays a
   canned "AI-ish" animation disconnected from state.
3. **It appears when needed and gets out of the way.** Hidden by
   default; a hotkey or activity wakes it; success/error auto-hide on a
   timer; `Esc` always dismisses.
4. **Chrome-less does not mean information-less** — state is
   communicated through motion, scale, and color of one object, the way
   Wispr's capsule communicates listening/processing/done.

---

## The orb itself (what's actually rendered)

`src/components/ui/orb.tsx` — a react-three-fiber `Canvas` with ONE
`CircleGeometry(3.5, 64)` and a custom `ShaderMaterial`. There is no
DOM UI inside the orb; everything is GPU shader.

- **Structure:** polar-space fragment shader → 7 noise-modulated soft
  ovals orbiting the center (`drawOval`, flattened gradients, per-oval
  phase offsets from a seeded `splitmix32`) + 2 secondary noise rings
  (`sharpRing`, `smoothRing`, screen-blended white) + a perlin-texture
  flow distortion (`flow()`) that twists the field with `uOutputVolume`.
  Final color = a 4-stop luminance ramp
  `black → uColor1 → uColor2 → white`, with `uInverted` flip when the
  root element carries `.dark` (MutationObserver).
- **Motion as state:** `useFrame` lerps two volumes — `uInputVolume`
  (how strongly the field distorts/extends) and `uOutputVolume` (how
  much the ring churns) — plus animation speed, toward per-state
  curves: idle = slow breathing baseline (~0.3 out), listening =
  rhythmic input swell (0.55±0.35 sin), talking = dual churn
  (0.65/0.75±0.22). Speed and color lerp at 0.08–0.2/frame → smooth,
  never jumpy.
- **Texture:** `src/assets/perlin.png` bundled — no CDN, works inside
  the Tauri CSP-less sandbox.
- **Robustness:** `webglcontextlost → forceContextRestore` guards the
  Wry/WebKit context-loss quirk.
- **The window** (`tauri.conf.json`): `transparent:true,
  decorations:false, alwaysOnTop:true, skipTaskbar:true,
  resizable:false, focusable:true, visible:false`. The page itself
  (`styles.css`) is `background:transparent; overflow:hidden` — the orb
  floats on the desktop with no rectangle around it.

## UX flows, state by state

### Wake & mode select

| Trigger | Result |
|---|---|
| `Ctrl+Shift+P` (global) | `overlay_show` + `overlay-mode {mode:"prompt"}` → orb appears, resets to prompt mode |
| `Ctrl+Shift+C` (global) | same for `mode:"computer"` |
| `Esc` | `overlay_hide` |
| auto-timer | ~0.9–1.8s after done/error → `overlay_hide` |

### Prompt-mode flow (sign → text)

```
orb appears (idle breathing)
  → user signs → partial_gloss events → orb = "listening" (input swell)
  → endpointer Commit → 'committed' event
  → cmd overlay_commit_type {text} → typed into the FOCUSED app
  → ~0.9s later orb hides itself
```

### Computer-mode flow (sign → agent)

```
orb appears → committed sign text → agent_act {text, history, pending}
  → orb = "talking" while the LLM loop runs
  → tool needs confirm → AgentPendingDto returned, orb waits
  → Enter = approve (rerun with pending+approve:true)
  → done/ignored/error → orb auto-hides ~1.8s
```

### What's *missing* vs the target capsule UX (be honest)

| Target (user's words) | Current state | Gap to close |
|---|---|---|
| "mini pill that gets bigger when signing" | fixed 180×180 square | expand width on `partial_gloss`/activity: window `setSize` or CSS transform on `.orb-stage` |
| "shows the interpretation in real time" | orb only — no text rendered | a thin text line inside the capsule fed by `partial_gloss`/`committed` (the capsule can hold 1 line of text without becoming a panel) |
| "sign ENTER and it lands in the composer" | `committed` → `overlay_commit_type` → typed into focused window | commit-key sign needs a template entry (`ENTER`) + the text to accumulate (composer = target app itself today) |
| orb = 12 pipeline states | orb shows 3 agent states | map `OrbState` → visual params (color per state: idle=neutral, listening=cool, thinking=violet, error=red, done=green pulse) — the shader already accepts `uColor1/2` lerp |
| PiP preview for calibration/debug | `aliveframe://` protocol + PiP thread exist | hidden debug flag to show the PiP JPEG inside an expanded capsule — dev-only, off by default |
| onboarding (camera check, pack pick, calibrate 3 signs) | nothing | first-run flow = blocking modal window (a real window is fine for setup — chrome is only banned in the runtime path) |
| tray icon / quit | none | system-tray entry: enable/disable, pack, quit |

## Interaction contract (keep these invariants)

- **Esc always works** even when the webview lacks focus → the Rust
  side focuses the overlay before send (the known X11 quirk; keep the
  `xdotool windowfocus` pre-step in `overlay_show`).
- **Never steal focus mid-typing.** `overlay_commit_type` types into
  whatever has focus *at commit time* — the overlay must not focus
  itself after show; if needed, `focusable:false` during type-out and
  restore after.
- **No silent rewrites.** A `partial_gloss` that gets superseded fades
  out rather than mutating committed text.
- **Confirm stays visible.** An `awaiting_confirm` pending call pins
  the capsule open (no auto-hide) until Enter/Esc.

## Mobile app (future — honest scoping)

Nothing mobile exists today; the seam is designed for it:

- **Option A — phone as sensor:** phone camera → landmark stream →
  WebSocket/gRPC → the same `result slot` contract (`Landmarks` DTO).
  Rust core unchanged; mobile is a dumb sensor + haptic confirm.
- **Option B — companion remote:** desktop stays the brain; a small
  PWA/native app shows orb state + pending confirms (approval from
  across the room). Needs only the `alive-event` stream +
  `agent_act`/`overlay` commands — all already IPC-shaped.
- **Option C — on-device:** MediaPipe 21-kp hands on phone GPU →
  the existing DTW matcher + packs ported (matcher is dependency-free
  math) → offline command grammar; full sign encoder needs the ONNX
  step anyway.
- Recommendation for the demo: **don't promise mobile**. Promise the
  seam. The desktop story alone carries the submission.

## Visual tokens (suggested, for the expanded capsule)

- Idle: `#CADCFC → #A0B9D1` (current, cool neutral)
- Listening: `#A8C8FF → #6E9FFF` (blue swell)
- Thinking/agent: `#C9B8FF → #8E7BFF` (violet)
- Done: `#A8E6CF → #5FCCA0` (one green pulse, then hide)
- Error/confirm: `#FFB4A2 → #E76F51` (warm, attention)
- Timing: state lerp ≤250ms felt-lag; auto-hide 900ms (success) /
  1800ms (error); capsule expand spring ~180ms overshoot-free.
- Type: if text lands in-capsule — 13–14px, monospace-ish for gloss,
  single line, fade+slide-in only.
