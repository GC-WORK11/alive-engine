# ALive — SOTA PRD (buildable)

**Date:** 2026-09-24 · **Status:** PRD from 10-agent breakthrough sweep (7 researchers + critic + gap-followup + synthesis)
**Prior docs:** `alive-deep-research.md`, `alive-build-plan.md` (both still valid; this PRD tightens them with measured numbers)
**Reader:** the builder agent. If a number here lacks a source, treat it as must-measure, not fact.

---

## 0. Contract in one paragraph

ALive is **sign as an input method**, not a translator. V1 = **ASL-only**, calibrated per-user, two modes (dictation + sign→action), Linux-first Tauri+Rust desktop orb. Honest accuracy contract: **90%+ on your vocabulary, for you, with invisible calibration** — not open-domain sign understanding. Latency contract: **p50 commit ≤400ms, p95 ≤900ms, barge-in ≤300ms**; perception never blocks on cloud. Jev is the **text-only decision layer** (1 call, parallel Choice/Score/Noul, ~1¢/user-hour), never vision. Moat = the command layer nobody ships: intent grammar + confidence routing + confirmation/undo + promote-3x-to-script.

---

## 1. The three brutal questions — answered with numbers

### Q1. "Wispr Flow does words/sec, we do signs/sec — are we fucked?"

**On raw dictation throughput: yes, you lose. On task throughput: you can win. That is the entire strategy.**

Measured linguistics (sweep L2):

- Speech ≈ **4–5 words/sec** sustained; sign ≈ **2–3 signs/sec**, roughly **1 proposition per 1–2s**.
- You cannot beat Wispr Flow at free-text speed. Don't try.
- **Breakthrough = don't compete on dictation; compete on action density.** One command-sign ("reply yes to Rahul, archive") = a full task. Speech dictation cannot do that reliably without an agent loop. ALive's win metric is **sign→done**, not words-per-minute.

What makes it *feel* instant despite the rate gap (all licensed by evidence):

1. **Speculative partials:** unique sign identifiable at **~240ms / 35%** of production vs **~330ms / 83%** for full disambiguation. Emit dashed candidate tokens at 240ms, commit later. Never silent-rewrite.
2. **Endpoint tuning:** tentative emit at **300–400ms** low-motion pause; commit at **600–800ms + EOS token**. Human turn-taking mode is **0–200ms, mean 208ms**; past **300/500/800ms** the interaction cliff hits (abandonment, LLM-judge agreement drops to ~70%). Stay under 800ms p95 or feel broken.
3. **Windowed-dictation + ISLR beats CTC** for streaming (sweep finding). Build the temporal head as causal TCN + explicit EOS over a **1.2s window, 0.3–0.5s right context, 2.4s history, stride 1**, not CTC.
4. **Fingerspelling fast-path:** fingerspelling runs **5–8 letters/sec** and causes **12–35% of SL2T errors**. Separate ISLR fingerspell detector + Jev N-best disambiguation; don't let the general encoder eat it.
5. **Command-grammar shortcut:** closed-vocab intents (≤255 Choice options) resolve faster and more accurately than free dictation. Mode B will feel faster than Mode A. Ship both, lead demos with B.

### Q2. "Jev for vision/compute/end-to-end — what are its unknown qualities?"

**Brutal: Jev takes TEXT ONLY. No pixels, no streaming, no chaining, English-first. There is no hidden vision mode. The exploits below are all real and all stay inside that box.**

Verified (docs + benchmark + status, sweep J1):

- Endpoint `POST https://api.typesafe.ai/v1/systemone`, model `jev-1.13.0` / `jev-latest`. Primitives: **Choice (≤255 opts → choice+probabilities+confidence), Score (2–10 levels), Noul (0–1)**.
- **All questions in one call run in parallel; adding questions barely changes latency.** Mixed 3-type fanout in 1 call is the core exploit.
- Pricing **$0.042/M input tokens, output free** → ~413-token state ≈ **$0.000017/decision → ~$0.01/user-hour at 600 utt/hr**. At 10 qps sustained ≈ $7/hr (never do this; Jev is per-utterance, not per-frame).
- Latency: vendor 70–500ms; independent **p50 ~400ms, p95 ~500ms**. Design for p95 500, not vendor-best 70.
- Accuracy **91.7% on tool-call risk classification (n=60), ECE ~0.07, zero wrong answers at confidence 1.0 (0/5)**. Calibrated enough to route on.
- Uptime **99.829%, 4 incidents in Sept** (9 days old). Context **64k (32k state + longest question)**, **1,200 req/min**. No fine-tune, no stream, one judgment per question, degrades with irrelevant context, CJK weak.

Real exploit patterns (the "unknown qualities" that are actually just underused):

1. **1-call fanout:** intent Choice + slots Choice + `needs_confirmation / is_risky / is_multi_step / user_is_addressing_computer` Nouls + confidence/risk/urgency Scores — **one round-trip**.
2. **Spec + commit without 2nd RT:** ask speculative questions (partial gloss) AND commit questions in the same call; resolve in code when EOS fires. No second Jev call on commit.
3. **N-best in state:** pass gloss N-best + probabilities + landmark stats (motion energy, EOS p, hold duration) as a compact state object. Jev re-scores; code commits.
4. **Confidence routing with formula:** `conf = (n·peak − 1)/(n − 1)`; **act ≥0.85, confirm 0.6–0.85, refuse/ask <0.6**. Escalation budget 10–15%.
5. **Complexity router in 1 call:** `Choice(route: deterministic / LLM / human)` + category + room/device/action slots together.
6. **Post-action verifier:** `Noul(state_actually_changed?)` against before/after a11y snapshot. Never trust "done".
7. **30fps verbalization is UNTESTED vs 32k/64k + 1200rpm** — so Jev is per-utterance/per-decision ONLY, never per-frame. Perception loop must not await Jev.

### Q3. "Which sign language for English + global + huge TAM at MVP?"

**ASL. No contest for V1. Not because it is biggest globally, but because it is the only one that is shippable + measurable + Jev-compatible on day 1.**

- ASL is the sign language of English; Jev is English-first (CJK weak) → the text handoff is clean.
- Only commercial-clean isolated corpus (**PopSign, CC BY 4.0**) + only public benchmark (**FLEURS-ASL**: human 13.0/64.6, baseline 3.7/37.2 — headroom is real, gaming is impossible).
- Google SL2T existence proof: **100k+ hours, 50+ languages joint > single-language, 70 BLEURT zero-shot**, MediaPipe-landmarks-only on Pixel 11.
- BSL/Auslan: fragmented dialects, weaker clean data. ISL: better long-term market, but iSign/ISLTranslate are **non-commercial** → requires own $20–40k corpus before shipping. DGS/Libras: same data problem.
- Architecture stays multilingual from day 1 (1 encoder + N packs), but V1 ships **Tier-A ASL only**; ISL is pack #2.
- **Honest gap (sweep unresolved): TAM $/users for ASL/DGS not sourced.** Do not invent TAM slides. The shippable-TAM argument is: English-output + benchmark + clean data + Jev-EN = only V1 that can launch and be measured. Size revenue from pricing ($14/$29), not from invented signer counts.

---

## 2. Product definition

**Modes (one pipeline):**

- **Mode A — Sign Dictation:** sign → live text at cursor → accept/edit. Calibrated-vocabulary first (user's real phrases + fingerspelling), not open-domain translation.
- **Mode B — Sign → Action:** sign → intent+slots → Jev typed decision → executed action with risk tiers, confirmation, undo. The differentiator.

**V1 scope:** ASL Tier-A · Linux-first (X11 solid, Wayland degraded-but-honest) · local-first zero-OAuth actions · opt-in telemetry · nothing-recorded default.
**V1 actions:** open/switch app · open URL · focus window · local file search · browser search · AI query · clipboard/screenshot · basic browser form fill.
**Non-goals:** interpreter replacement (never medical/legal) · teaching app/dictionary · text→sign avatars · generic chatbot · mobile app · MCP client · universality claims · Gmail (deferred: restricted-scope CASA $540–1,800/yr + weeks).

---

## 3. Latency contract (CI-enforced)

Full online budget (sweep L2/P3/J1):

| Stage | Budget |
|---|---|
| Frame interval | 33.3ms @30fps (16.7 @60 — capture option, not default) |
| USB + decode | 2–5ms |
| Landmarks (whole-body-m) | 13.5ms optimistic / up to 36.3ms realistic per frame |
| Frame → orb transport | custom URI protocol ~46ms, ~0ms main-thread block (never `invoke`: measured 1811ms stall) |
| Endpointing | tentative 300–400ms pause → speculative; commit 600–800ms + EOS p>0.8–0.9, min segment 0.2–0.3s |
| First partial gloss | ≤300ms from motion onset |
| Commit | p50 ≤400ms, p95 ≤900ms |
| Jev decision (async) | p50 ~400ms, p95 ~500ms; never blocks perception |
| Barge-in / interrupt recovery | ≤300ms |
| T1 action execute | <300ms · T3 CUA escape hatch 3–8s/step (gated, never default) |
| Undo window | 5–10s inline + session ⌘Z |

**Rules:** cloud LLM/Jev never on the perception critical path · `LIVE_STREAM`-style immediate dispatch for the interactive path, `VIDEO`-style blocking only for the temporal re-score pass · detect-and-track + frame-skip (latest-frame) + model tiering + thermal shedding (thin laptops lose 20–40% sustained) · existence proof this budget is shippable: Talksign Palm 1.0 **84.2% semantic @29ms**.

---

## 4. Perception spec (low-RAM, CPU-real-time, no Python)

**Primary: RTMW/DWPose 133 whole-body keypoints as ONE ONNX model via `ort`. Fallback: MediaPipe Tasks (Hand+Face+Pose). Decide at M0 by head-to-head on the dev machine.**

Sweep numbers (P3, CPU unless noted):

- RTMPose-m **75.8 AP, 90 FPS on i7, 430 FPS on GTX1660** · whole-body-m **60.6 AP / 13.5ms / 2.2 GFLOPs** · -l 63.1/23.4ms · -l384 66.5/44.6ms · RTMW **70.1–70.2** · DWPose-m 0.606 (body 0.685 / hand 0.527 / face 0.828) · MediaPipe hand **17.1ms CPU / 12.3ms GPU** · thermal fallback nano **0.99M params / 0.31 GFLOPs, skip-5**.
- **Mouth > eyes, decisively:** body-only 80.53 → +eyes 82.77 (ns) → **+mouth 88.24 (+7.7pts)** · SignMouth 24.71 vs 24.32 baseline (PHX), 10.75 How2Sign. Spend face budget on mouth ROI, not gaze/eyebrow taxonomy in V1.
- **ROI failures are real:** hand-crop heuristic IoU **57 vs 63 learned MLP**, worst-case **3 vs 16**. Ship `mediapipe-hand-crop-fix` equivalent; MediaPipe Holistic distorts under head pitch and drops tracks at distance.
- Capture **1080p, crop ROIs at native res down to 192–256px model input**; keep hand ROI ≥200px long side (the exact 200px knee is **unmeasured** — M0 must measure accuracy vs ROI size).
- Runtime `ort`, **pin exact version** (sweep saw 1.28; prior doc pinned 2.0.0-rc.x — M0 spike decides and pins; hard ORT coupling). EPs: CoreML/DirectML/OpenVINO/CPU. Never WebGPU in WebView on Linux (broken in WebKitGTK). No Python sidecar (multi-GB, GIL jitter).
- Capture in Rust (`nokhwa` 0.10.x). **Never `getUserMedia`** (WebKitGTK denies media by default). Capture in Rust, chrome in WebView.

---

## 5. Language packs (1 pipeline + N packs)

| Tier | Coverage | V1 |
|---|---|---|
| A Full | dictation + action vocab + calibration | **ASL** |
| B Command | action vocab + calibration, no free dictation | — (scaffold) |
| C Community | user/org lexicon, labelled unverified | — (scaffold) |

Pack = `pack.toml + adapter.onnx + tokenizer + lexicon.json + intents.json + eval/`. Adding a pack = zero core changes (M4 verification). Language ID head from day 1. Joint multilingual training once ≥2 packs have data (expect the SL2T joint>single effect).

---

## 6. Jev exploit spec (normative for the builder)

**Call shape (one call per utterance/decision, all parallel):**

```json
POST https://api.typesafe.ai/v1/systemone  { "model": "jev-1.13.0" }
{
  "state": {
    "gloss_nbest": [{"gloss": "REPLY YES RAHUL", "p": 0.81}, {"gloss": "REPLY RAHUL", "p": 0.12}],
    "slots_raw": {"person": "Rahul", "action_hint": "reply"},
    "signal": {"eos_p": 0.9, "hold_ms": 650, "motion_energy": 0.2, "fingerspell_p": 0.05},
    "context": {"app": "browser", "url": "mail…", "last_action": "opened thread 412"}
  },
  "questions": [
    {"type": "choice", "key": "intent", "options": ["reply_email", "archive", "search", "open_app", "dictate", "cancel"], "text": "Which intent?"},
    {"type": "choice", "key": "route", "options": ["deterministic", "llm", "human"], "text": "Which route?"},
    {"type": "noul", "key": "addressing_computer", "text": "User is addressing the computer (not touching face / chatting)?"},
    {"type": "noul", "key": "needs_confirmation", "text": "This needs explicit confirmation?"},
    {"type": "noul", "key": "is_risky", "text": "This is risky or irreversible?"},
    {"type": "noul", "key": "is_multi_step", "text": "This needs multiple steps?"},
    {"type": "score", "key": "confidence", "levels": 5, "text": "Confidence the top intent is correct?"},
    {"type": "score", "key": "risk", "levels": 5, "text": "Risk if executed now?"}
  ]
}
```

**Router (code, not model):** `conf=(n·peak−1)/(n−1)`; **≥0.85 act · 0.6–0.85 confirm/preview · <0.6 ask/ignore**; `addressing_computer <0.6` → drop silently (anti-false-trigger). Escalation 10–15%. Post-action: second 1-call `Noul(state_actually_changed?)` vs before/after snapshot.
**Fallbacks:** Jev timeout/error/low-conf → function-calling LLM → human ask. `DecisionProvider` trait (Jev default, pinned version); re-evaluate quarterly. **Never per-frame; 1200rpm + 32k/64k context are utterance-scale budgets.**

---

## 7. Action layer + safety (normative)

**Tiers:** T1 OS-native deterministic (<300ms, $0): xdg-open/open/start, deep links, focus, indexed search, clipboard/screenshot → ship first. T2 browser/a11y ($0.001–0.01/step): CDP `snapshot → click(ref) → type → read-back`; a11y tree 200–400 tok vs screenshot 3–5k. T3 CUA 3–8s/step: allowlisted escape hatch with step/time/cost caps + confirm writes. **Promote any CUA flow that succeeds 3× into a deterministic script** (this is the cost/latency compounder).
**Risk:** T1 notify (act+journal+undo, no block) · T2 confirm-once/undo (reversible state, 5–10s undo) · T3 draft-then-commit (reviewable artifact → distinct held-sign confirm + echo, never remembered) · T4 hard gate (2-factor + pause or disallow). Idempotency keys, no blind retry on Send/Submit/Pay/Delete, typed errors, immutable journal, one-click human takeover. Tool output/page text = untrusted (envelope + provenance + per-task allowlist + hash tool descriptions; never build shell from model output).
**Platform honesty:** Wayland = no portable window enum/target, no global shortcuts, overlay layer-shell broken → build on inputs + a11y + portal RemoteDesktop/libei, not window handles. macOS = notarized DMG ($99/yr), no MAS (transparency needs private API + Accessibility). Windows = WebView2 bootstrapper, UIPI blocks elevated targets (report, don't crash).

---

## 8. Orb UX spec (state legibility is the law)

States (visually distinct, animation communicates): idle → activating → camera-active → signer-detected → interpreting → intent-forming → ready → executing → done → confirm-needed → error → recovery. Never show "calling Jev / TCN / router".
**Rules:** hold-to-sign consent gate + hard stop · camera = small floating PiP w/ blurred bg + physical off indicator, never video-call · partial gloss ribbon with dashed candidates (no silent retype — Siri's #1 anti-pattern) · correct by re-signing the unit, cancel by dropping hands · confirm = **distinct held sign, asymmetric dwell 1.0–1.5s + visible echo + cancel** (never involuntary nod/blink) · user-chosen output: type-at-cursor / overlay card / speak-aloud / action-preview from the same semantic state · nothing-recorded default, narrow FOV, per-purpose opt-in, ASL-video consent.
**Deaf etiquette (gates, not values):** nothing-about-us-without-us (Deaf leadership, paid $25–75/hr, certified interpreters, recruit via Deaf orgs) · gloves are poison (19% comfort vs 91% app) · never universality, never interpreter-replacement, refuse medical/legal · latency > polish (synced-incomplete beats lagging-perfect).

---

## 9. System design (builder view)

```
1080p30 nokhwa ─▶ ROI crop (native) ─▶ RTMW-133 ONNX (ort) ─▶ ringbuf 2.4s ─▶ causal TCN+EoS ─▶ shared encoder ─▶ ASL pack ─┬─ Mode A text head ─▶ cursor
                                                                                                                              └─ Mode B intent+slots ─▶ Jev 1-call ─▶ router ─▶ T1/T2/T3 ─▶ journal+undo
                                                                                                                                                              ─▶ orb feedback (URI frames, invoke for keypoints/control only)
```

**Repo (Cargo workspace + Tauri):** `crates/alive-core` (state machine/event bus) · `alive-perception` (capture+landmarks+ROI) · `alive-temporal` (TCN+EoS, ringbufs) · `alive-sign` (encoder+packs+langID) · `alive-intent` (DecisionProvider, Jev, LLM fallback, router) · `alive-act` (os/{linux,macos,windows}+browser+cua) · `alive-journal` (append-only log, undo/compensation, export) · `alive-ipc` (commands/events/frame protocol) · `packs/asl/` · `src/` (orb/PiP/ribbon/confirm/journal) · `src-tauri/` · `eval/` (FLEURS-ASL harness, cross-signer CI, latency benches).
**Stack pins:** Tauri 2.11.x (minor) · React+TS+Vite+Tailwind+shadcn · `nokhwa` 0.10.x · `ort` exact-pin at M0 · `atspi`/`axuielement`/`uiautomation` · updater Ed25519 on GitHub Releases · ~$220/yr signing.

---

## 10. Costs + pricing (per active user/mo)

Jev ~1¢/hr · blended LLM ~$0.0014/utt (normal 2,600 utt ≈ $3.64) · on-device perception $0 (margin shield) · cloud vision fatal ($0.58/min @1fps) → opt-in ≤1fps · totals Light $2.38 / Normal $5.14 / Heavy $15–16 / heavy-agent $36–50 (must meter). Pricing: Free 300 utt on-device-only → Pro $14 (3,000 utt, 25 agent tasks, 500 cloud frames) → Power $29 → agent packs $5/100. **Never meter on-device perception.** Validation program budget ~$11–15k/6wk (WoZ → battery → field; SUS ≥68 + UEQ).

---

## 11. Milestones + verification (builder checklist)

- **M0 spikes (numbers or re-plan):** Rust cam→landmarks 30fps X11+Wayland p95 ≤120ms · URI frames ~0ms block · RTMW-vs-MediaPipe head-to-head (fps, jitter, rotated-palm ROI) · Jev hello-world from Rust (p50/p95, typed round-trip, $/call) · orb transparent always-on-top + click-through + hold-to-sign, Wayland graceful. **Exit:** table of measured vs target; p95 frame >150ms or untracked rotated palms = re-plan.
- **M1 Mode A (ASL dictation v0):** calibrated-vocab + fingerspell path · partials ≤300ms, commit p50 ≤400 · re-sign to correct, drop-hands cancel, 5–10s undo. Verify: 20 held-out phrases/internal signer ≥95% top-1 · p95 commit ≤900 · zero visible self-retype.
- **M2 Mode B (Jev actions):** 1-call fanout + router + T1 executor + journal/undo + outcome verification. Verify: 30-utt battery ≥90% routing · escalation 10–15% · zero irreversible w/o confirm · Jev ≤$0.02/user-hr.
- **M3 trust:** T1–T4 + confidence routing + held-sign confirm + the 3 missing studies (confirm-gesture sign-vs-nod-vs-tap; latency-abandonment 500ms/1s/2s; camera-consent). Verify: 0 false confirms/100 adversarial trials · users can narrate orb state.
- **M4 multilingual scaffold + ISL:** langID + pack loader, ASL refactored out, ISL scaffold; own-corpus plan (12–20 signers, ≥4 regions, 5–8 sessions, 2–3 geometries, 50–100h, $20–40k, split research-vs-shipped consent). Verify: cross-signer per-pack numbers published · new pack = zero core changes.
- **M5 T2 + Calendar:** CDP/a11y + CUA hatch + promote-3x; Calendar direct API (`calendar.events`, no CASA). Verify: 20-step browser task with read-back · zero unverified "success" · typed recoverable failures.
- **M6 productize:** AppImage/deb/rpm + DMG + MSI/NSIS+WebView2 · per-OS permission onboarding · ASL+English privacy onboarding · metering. Verify: clean-install 3 OSes · deny-paths degrade · SUS ≥68 field.

---

## 12. Kill criteria (stop, don't pivot-blindly)

Top-1 intent <85% after Phase-1 iter · p95 e2e >800ms · >30% refuse persistent camera · >20% actions need >1 retry · median engaged user lands heavy-agent with WTP <$20 · Jev ECE >0.1 or wrong-at-1.0 ever · can't recruit via Deaf orgs with enthusiasm / recurring "surveillance tool" feedback.

---

## 13. Unresolved — must measure, never assume (carried from sweep)

TAM $/users ASL/DGS (none sourced) · orb HW size/power/BOM (none) · per-action P95/acc beyond T1/T3 (only T1/T3 bounded) · 200px ROI knee + RTMW-exact-CPU + Holistic-CPU-ms (unmeasured) · 30fps verbalization token-rate vs 32k/64k/1200rpm (untested — hence per-utterance Jev only) · fingerspelling fix numbers (SL2T 12–35% errs known, no proven fix measured — M1 must produce them).

---

## 14. Builder handoff — first 5 tasks

1. Scaffold the workspace + `eval/` latency bench (frame-interval + USB + landmark + transport columns) before any product code.
2. M0 spike 1+2+3 in one branch: nokhwa → RTMW-ONNX (ort) → URI frames → fps/jitter/ROI table.
3. M0 spike 4: Jev client in Rust with the §6 call shape; record p50/p95 + $/call + confidence histogram.
4. M0 spike 5: orb shell + hold-to-sign + PiP; Wayland degrade path documented.
5. M1 vertical slice: 20-phrase calibrated dictation with partials + commit + undo, evaluated on FLEURS-ASL sd-test for comparability to SL2T 70 BLEURT.

> Keep asking: if I sat down and signed something useful cold, would the computer just do it — no tutorial, no pipeline on screen? If no, the pipeline got ahead of the product.
