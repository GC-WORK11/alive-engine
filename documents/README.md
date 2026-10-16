# ALive — Project Documents

The complete walkthrough of this repository: what we built, why, how it
connects, and where it goes next. Read in order — each doc builds on the
last.

| # | Doc | What's inside |
|---|---|---|
| 01 | [architecture.md](01-architecture.md) | **The full tour.** Every crate, module, and protocol — what each line of code does, how data flows end-to-end, test coverage, and an honest table of what's wired vs written-but-dead (including the unwired 240Hz upsampler + Kalman "breakthrough"). |
| 02 | [epic-brief.md](02-epic-brief.md) | **The product.** What ALive is, who it's for, why it fits the WeMakeDevs × Wispr "No Hands" challenge, every UX flow end-to-end, and the honest feature-status inventory. |
| 03 | [tech-stack.md](03-tech-stack.md) | **The stack.** Everything in use today with measured numbers, where each upgrade plugs in (GPU EP, sign encoder, local LLMs, Wayland), and the real cost sheet. |
| 04 | [ui-ux-flow.md](04-ui-ux-flow.md) | **The surface.** The shader orb, the overlay window, state→visual mapping, every interaction flow, the gaps to the target "capsule", and honest mobile scoping. |
| 05 | [roadmap-and-tools.md](05-roadmap-and-tools.md) | **The path.** The ordered punch list (wire the breakthrough first), verification rituals, comms/tools, internal shorthand, release plan R0→R3. |

## The 10-second version

ALive = **sign as an input method**: a background camera → motion-gated
latest-frame pipeline → YOLOX+RTMW pose → 350/700ms endpointer → DTW
command matcher → Jev/DeepSeek decision → 15-tool journaled executor →
one shader orb on top of everything.

The code is ~9,700 lines, heavily tested, and structurally sound. Its
best part — the causal PCHIP upsampler + Kalman predictor that turns
~10Hz poses into 240Hz labeled ticks — is written and proven, but was
never wired into the pipeline. That wiring is job #1.

## Ground rules this project lives by

- **Honest by construction:** sensor FPS ≠ understanding ticks; unknown →
  reject, never guess; dry-run default; redacted journal.
- **No chrome:** one orb; the UI renders pipeline truth.
- **Swap-ready seams:** `CaptureSource`, `LandmarkExtractor`,
  `EosScorer`, `SignEncoder`, `DecisionProvider`, `LocalExecutor` are all
  interfaces — nothing in the core needs forking to upgrade.
