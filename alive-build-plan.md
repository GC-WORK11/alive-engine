# ALive — Build Plan (V1 → Global)

**Date:** 2026-09-24 · Companion research: `alive-deep-research.md`
**Repo:** `/home/govinda/sign-flow` (empty — greenfield)

---

## 1. What we are building

> A person sits at a computer, opens ALive, signs naturally, and the computer understands what they mean and does the task — continuously, without recording, uploading, or speaking.

Two modes, **one pipeline**:

- **Mode A — Sign Dictation.** Sign → live text at the cursor → user accepts/edits. Immediately useful in *every* sign language (output is the user's own written language). This is the "Wispr Flow for sign."
- **Mode B — Sign → Action.** Sign → intent → **Jev** typed decision (with confidence) → executed computer action, with risk tiers, confirmation, and undo. This is the differentiator nobody ships.

**Design contract (say this out loud, do not hide it):** V1 is accurate on **your vocabulary, for you, with per-user calibration** — not open-domain sign understanding. The literature is unambiguous: gloss-free translation drops from ~22 BLEU signer-dependent to **3.6 BLEU cross-signer**. Build the product so calibration is invisible, not so the caveat is hidden.

---

## 2. The pipeline (peak flow)

```
                          ┌───────────── LANGUAGE-AGNOSTIC (built once) ─────────────┐
 camera 1080p30 ──▶ landmarks ──▶ segmentation ──▶ SHARED SIGN ENCODER ──┐
 (Rust/nokhwa)      (MediaPipe     (causal TCN +   (pose-based, gloss-    │
                     or RTMW ONNX)  EOS token)      free, multilingual)    │
                                                                          │
                          └───────────────────────────────────────────────┘
                                                                          │
        ┌───────────────────────── LANGUAGE PACK (per sign language) ─────┤
        │  language ID · output tokenizer · command lexicon · eval set    │
        └───────────────┬─────────────────────────────────┬───────────────┘
                        │                                 │
              Mode A: TEXT HEAD                  Mode B: INTENT HEAD
        (gloss-free translation →          (closed-vocab intent + slots,
         English/Hindi/… text at cursor)    per-user calibrated)
                        │                                 │
                        │                                 ▼
                        │                     SEMANTIC STATE (text + JSON)
                        │                                 │
                        │                                 ▼
                        │                    ┌──── JE V (TypeSafe) ────┐
                        │                    │ choice / score / noul    │
                        │                    │ + calibrated confidence  │
                        │                    └───────────┬──────────────┘
                        │                                │
                        │                    ┌───────────▼──────────────┐
                        │                    │ DECISION ROUTER           │
                        │                    │ tier · confidence · confirm│
                        │                    └───────────┬──────────────┘
                        │                                │
                        │        ┌───────────────────────┼────────────────────┐
                        │        ▼                       ▼                    ▼
                        │   T1 OS-native           T2 browser/a11y      T3 gated CUA
                        │   (open, focus, URL,     (CDP snapshot →      (escape hatch,
                        │    file search) $0       click(ref) → type)   allowlisted, 3–8s)
                        │        └───────────────────────┴────────────────────┘
                        │                                │
                        └──────────────▶ ORB FEEDBACK ◀──┴── ACTION JOURNAL + UNDO
```

**Why this shape is both global and cheap:**
- Everything above the language pack is **language-independent** — landmarks don't know what language you sign. Google's SL2T proves the economics: **joint training over 50+ sign languages beats single-language models** (100k+ hours, 70 BLEURT zero-shot on FLEURS-ASL).
- Adding a language = **one pack (adapter + tokenizer + lexicon + eval set)**, not one codebase.
- Jev operates on **text**, so it is language-agnostic by construction — the decision layer never needs to know whether the user signed ASL or ISL.

---

## 3. Language strategy (the "300 sign languages" answer)

~**300 sign languages** exist (est.; the number is genuinely unknown). We do not claim universality — the Deaf community explicitly rejects that (200+ languages, plus dialects, Black ASL, late learners, one-handed signers).

| Tier | Coverage | What works | Examples (target) |
|---|---|---|---|
| **A — Full** | Dictation + action vocab + calibration | Everything | ASL, ISL, BSL, DGS, Libras |
| **B — Command** | Action vocab + calibration, no free-form dictation | Mode B fully; Mode A limited | Most languages |
| **C — Community pack** | User/org-submitted lexicon, labelled unverified | Mode B, disclosed quality | Long tail |

**V1 shipping language: ASL** — it is the sign language of English, has the only commercial-clean isolated corpus (PopSign, CC BY 4.0), the only public benchmark (**FLEURS-ASL**), Google's existence proof, and the deepest tooling.
**Pack #2: ISL** — better *market* (5.07M hearing-disabled per Census 2011, <300 certified interpreters, NEP-2020 standardisation), weaker data (iSign/ISLTranslate are non-commercial) → requires our own recorded corpus.

**Build the multilingual encoder scaffold from day one** (language embedding + per-pack adapter). Do not build an ASL-only model and call it V1 — that is the junkyard path.

---

## 4. Stack decisions (validated, pinned)

| Layer | Choice | Why / binding constraint |
|---|---|---|
| Shell | **Tauri 2.11.x** (pin minor) | 14 MB RAM vs Electron 200–300 MB. **WebView = chrome only.** |
| Core | **Rust** | Owns camera, ML, control, journal. A Tauri maintainer won't fully recommend Linux Tauri today → keep the Electron/CEF escape hatch documented. |
| UI | React + TS + Vite + Tailwind + shadcn/ui | Validated. Never drive React state at 30 fps — canvas/`<img>` only. |
| Capture | **`nokhwa` 0.10.x** | **Never `getUserMedia`** — WebKitGTK denies media by default. |
| Perception | **MediaPipe Tasks** (primary) → **RTMW 133-kp ONNX** (fallback/likely winner) | RTMW = one model for hands+face+body, Apache-2.0, RTMPose-m ≈ 11 ms CPU. |
| Inference runtime | **`ort` (pin exact version)** | Hard ORT coupling. EPs: CoreML / DirectML / OpenVINO / CPU. Never WebGPU in the WebView on Linux. |
| Frame transport | **Custom URI protocol** | `invoke` measured at **1,811 ms** frame stall vs **46 ms**. Non-negotiable. |
| Decision | **Jev** `jev-1.13.0` behind `DecisionProvider` | Text-only in. ~1¢/user-hour. LLM fallback for low confidence. |
| Control | Tier 1 native → Tier 2 CDP/a11y → Tier 3 gated CUA | CUA is 3–8 s/step — never the default. |
| Integrations | Direct APIs only; **no MCP client in V1** | MCP = RCE class + tool poisoning + confused-deputy OAuth. |
| Packaging | deb/rpm/AppImage → DMG (notarized, **not MAS**) → MSI/NSIS + WebView2 bootstrapper | ~$220/yr total signing cost. |

---

## 5. Repo layout (Cargo workspace + Tauri app)

```
sign-flow/
├─ crates/
│  ├─ alive-core/        # orchestrator: state machine, event bus, config
│  ├─ alive-perception/  # capture (nokhwa) + landmarks (MediaPipe/RTMW) + ROI crop
│  ├─ alive-temporal/    # causal TCN + EOS endpointing, ring buffers
│  ├─ alive-sign/        # shared multilingual encoder + language packs + language ID
│  ├─ alive-intent/      # DecisionProvider trait, Jev client, LLM fallback, routing
│  ├─ alive-act/         # action traits + os/{linux,macos,windows} + browser + cua
│  ├─ alive-journal/     # append-only action log, undo/compensation, audit export
│  └─ alive-ipc/         # Tauri commands, events, custom frame protocol
├─ packs/                # language packs as data (asl/, isl/, bsl/…)
│  └─ asl/ pack.toml · lexicon.json · intents.json · eval/ · adapter.onnx
├─ src/                  # React: orb, camera pip, gloss ribbon, confirm sheet, journal
├─ src-tauri/            # Tauri shell, capabilities, updater, perms onboarding
└─ eval/                 # FLEURS-ASL harness, cross-signer CI, latency benchmarks
```

---

## 6. Milestones

### M0 — Risk retirement spikes (do not skip)
Prove the five things that can kill the product, with numbers, before writing product code.
1. Rust camera → landmarks at 30 fps on Linux (X11 **and** Wayland): measure capture→landmark latency; target p95 ≤120 ms.
2. Frame delivery via custom URI protocol to the orb: target main-thread block ≈0 ms, no frame drops.
3. **MediaPipe Tasks vs RTMW ONNX** head-to-head on the dev machine: fps, hand-landmark jitter, ROI failures on rotated palms.
4. **Jev hello-world from Rust**: exact API shape, p50/p95 latency, typed `Choice`+`Noul` round-trip, cost per call.
5. Orb shell: transparent always-on-top window + Rust cursor-poll click-through + hold-to-sign gate; confirm Wayland degrades gracefully.

**Exit:** a table of measured numbers vs targets. If p95 frame latency >150 ms or hand tracking fails on rotated palms without a fix, re-plan before M1.

### M1 — Mode A: Sign Dictation v0 (ASL)
- Calibrated-vocabulary dictation first (phrases a user actually needs + fingerspelling), **not** open-domain translation. Honest, useful, shippable.
- Hold-to-sign → partial gloss ribbon (≤300 ms first partial) → commit (p50 ≤400 ms) → text inserted at cursor.
- Correction by re-signing the unit; cancel by dropping hands; 5–10 s inline undo.
- **Verification:** 20 held-out phrases per fixed internal signer ≥95% top-1; p95 commit ≤900 ms; no visible self-retyping (the #1 anti-pattern).

### M2 — Mode B: Sign → Action with Jev
- `DecisionProvider` + Jev: `Choice`(intent) · `Noul`(needs_confirmation / is_risky / user_is_addressing_computer) · `Score`(confidence) in **one parallel call**.
- Tier-1 executor: open/switch app, open URL, focus window, local file search, browser search, AI query, clipboard, screenshot.
- Action journal + undo, typed error taxonomy, independent outcome verification.
- **Verification:** 30-utterance battery ≥90% correct action routing; escalation rate 10–15% of decisions; zero irreversible actions without confirmation; Jev cost per user-hour ≤$0.02.

### M3 — Safety, confidence, and trust
- Risk tiers T1 notify / T2 confirm-once+undo / T3 draft-then-commit / T4 hard gate (two-factor + pause). Never auto-remember T3/T4.
- Confidence routing: high → act · mid → preview/dry-run · low → ask/ignore.
- Confirmation gesture implemented as a **distinct held sign with asymmetric dwell (1.0–1.5 s) + visible echo + cancel**; never an involuntary expression.
- **Run the missing studies:** (a) confirm-gesture study (sign vs nod vs tap), (b) latency-tolerance curve (500 ms/1 s/2 s, measuring abandonment), (c) camera-consent study (on-device vs cloud).
- **Verification:** no false confirmations in a 100-trial adversarial session; escalation stays 10–15%; participants can articulate what the orb is doing.

### M4 — Multilingual: language ID + packs + ISL
- Language ID head + pack loader; ASL pack refactored out of core; ISL pack scaffolded.
- Joint multilingual training (shared encoder, per-language adapters) once ≥2 packs have data — expect the SL2T effect (shared structure improves both).
- Own-corpus recording: 12–20 native signers, ≥4 regions, 5–8 sessions, 2–3 camera geometries; ~50–100 h → budget **$20–40k**, dominated by annotation; separate consent for research vs shipped weights.
- **Verification:** cross-signer accuracy reported per pack (this is the number that matters); adding a pack requires **zero core changes**; per-pack quality published honestly (Tier A/B/C).

### M5 — Tier 2 control + integrations
- Browser control over CDP/a11y tree first (`snapshot → click(ref) → type → read-back`), ~$0.001–0.01/step; CUA behind allowlist + caps as escape hatch; **promote any CUA flow that succeeds 3× to a deterministic script.**
- Google Calendar via direct API (`calendar.events` — sensitive scope, **no CASA**). Gmail deferred until after the CASA cycle (~$540–1,800/yr + weeks) — or local IMAP for own-account only.
- **Verification:** 20-step browser task completes with read-back verification; zero unverified "success" reports; failure states are typed and recoverable.

### M6 — Productization
- Packaging: AppImage/deb/rpm; notarized DMG; MSI + NSIS with WebView2 bootstrapper. Updater via `tauri-plugin-updater` (Ed25519 mandatory) on GitHub Releases.
- First-run permission onboarding per OS (Linux portal session, macOS Accessibility + Screen Recording + per-app Automation, Windows UIPI report).
- Opt-in telemetry, local-first. Privacy onboarding in ASL video + English. Physical camera-off indicator. Nothing-recorded default.
- Metering: utterances, agent credits, cloud-vision frames, clip storage. **Never meter on-device perception.**
- **Verification:** clean-machine install on all three OSes; permission-denied paths degrade rather than crash; SUS ≥68 in the field study.

---

## 7. Safety model (adopted, not invented)

| Tier | Examples | Behavior | Confirmation |
|---|---|---|---|
| **T1 Notify** | open/switch app, focus, search, scroll, read | Act immediately, journal, non-blocking undo | none |
| **T2 Confirm-once / undo** | create event, queue draft, rename/move file | Act into reversible state, 5–10 s undo | one gesture or undo window |
| **T3 Draft-then-commit** | send to a person, post, submit form, hard delete | Produce reviewable artifact → distinct confirm commits | explicit sign + visible echo; never remembered |
| **T4 Hard gate** | payments, credential/permission changes, bulk delete | Two-factor + forced pause, or disallowed | second independent factor |

Principles: reversibility is the primary risk axis · least privilege (scoped, expiring capabilities) · confirm only when the human can evaluate (show evidence/diff) · preview/undo over blocking dialogs · escalation budget 10–15% · immutable journal · earn autonomy, never assume it · treat all tool output and page text as untrusted data.

---

## 8. Non-goals (protect the focus)

Not building: an interpreter replacement (never medical/legal) · a sign-language teaching app or dictionary · text→sign generation/avatars in V1 · a generic chatbot or agent wrapper · a dashboard or settings maze · a video-call-style camera UI · a mobile app in V1 · an MCP client or plugin marketplace in V1 · universality claims of any kind.

---

## 9. Open questions (need decisions)

1. **Corpus funding:** record our own ASL + ISL corpus (~$20–40k) or start with PopSign/INCLUDE only? (Recommend: record, because the command vocabulary is out-of-domain for every public corpus.)
2. **On-device vs cloud dictation:** full continuous translation is too heavy for a laptop today → do we ship Tier-B/command-only for Mode A until an encoder is trained, or use a server-side translation model (privacy trade-off, and the shape that triggers the community's camera concerns)?
3. **Linux Wayland:** ship Linux with degraded window control at launch, or gate the orb to X11/macOS/Windows and treat Wayland as beta?
4. **Which languages get funded packs first** after ASL/ISL (BSL? DGS? Libras? — market vs data trade-off).
5. **Jev dependency:** default-on with LLM fallback (recommended), or local-only for the privacy-maximal tier?

---

## 10. The one question to keep asking

> If I sat in front of this for the first time, could I naturally sign something useful and have my computer do it without thinking about the technology behind it?

If the answer is no, the pipeline got ahead of the product.
