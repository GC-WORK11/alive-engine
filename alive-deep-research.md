# ALive — Deep Research Report

**Date:** 2026-09-24
**Scope:** 20 parallel deep-dive research agents + direct primary-source verification (Google DeepMind, TypeSafe docs, Talksign, arXiv).
**Status:** Research complete. Build plan is in `alive-build-plan.md`.

---

## 0. Verdict in one paragraph

ALive is a strong idea with a **narrow, defensible core**: nobody ships *sign → meaning → computer action*. Every live product (Google SL2T, Talksign, Sorenson, Sign-Speak, Signapse, Rylo) ends in **text, speech, or an avatar** — none of them executes a computer action from a sign. Google is the platform risk (it ships ASL→text on Pixel 11 and hands *text* to Gemini), so ALive must not compete on recognition quality. ALive competes on the **command layer**: intent resolution, confidence routing, confirmation, undo, and reliable action execution. That is unoccupied, and it is where the moat is. The honest accuracy contract is "**90%+ on your vocabulary, for you, with per-user calibration**" — not open-domain sign understanding, which collapses cross-signer (BLEU 22 → 3.6 in EMNLP 2025 findings).

**On "support every sign language":** there are an estimated **~300 sign languages** (Wikipedia; the true number is unknown, new ones emerge via creolization), and the world's **~70M Deaf people** rely on them. The reason "all of them" is affordable rather than suicidal is that **the pipeline is language-agnostic up to the semantic layer**. Google proved this: SL2T is jointly trained on **100,000+ hours across 50+ sign languages** and states that **joint training across languages and dialects outperforms single-language models** by learning shared underlying structure. Landmarks (hands/face/body) do not care which language is being signed. So ALive supports N languages as **one encoder + N language packs**, not N products.

---

## 1. Product conclusion

- **Is ALive a strong product idea?** Yes, but only as an *input method*, not as a translator. The translation fight is already lost to Google (100k hrs, 70 BLEURT zero-shot on FLEURS-ASL, free on Pixel 11). The input-method fight — sign as a way to *operate* a computer — has no incumbent.
- **First real user:** signing users who would otherwise type English (Deaf professionals, signing students, power users). They have an acute, quantifiable pain, they are already fluent in the input modality (no learning cost), and 71–91% of them report that existing sign tech was built without them.
- **Strongest use case:** the **Gmail flow** — open → find email → draft reply → confirm send — because it proves *meaning*, not *words*. Second strongest: **sign dictation** (Mode A), which is immediately useful for every signer in every sign language.
- **Biggest risk:** not the model — the **signer-dependence cliff**. Accuracy that looks great on one signer collapses on the next. V1 must be sold as per-user-calibrated, and the product must be designed so calibration is invisible.

---

## 2. Existing landscape (what already works)

| Product | What it does | Real-time | Platform | Status |
|---|---|---|---|---|
| **Google SL2T** (DeepMind, 2026-08-12) | ASL→text dictation; on-device MediaPipe Holistic → server translation; gloss-free; can hand text to Gemini | Yes (streaming, latency explicitly optimised) | Pixel 11 (Gboard, Live Transcribe) | Live, free |
| **Talksign Palm 1.0 / Echo 1.0** (2026-05-20) | ASL→text **84.2% semantic / 79.6% word accuracy** (71k samples, SAGE encoder, 133 landmarks); text→ASL avatars | Yes | Mobile; desktop rollout announced Aug 2026 | Live |
| **Sorenson** (acquired OmniBridge + Hand Talk) | ASL↔English conversation POCs; avatar content | Partial | Kiosk/POC | Pilots |
| **Sign-Speak** (Deaf-led) | ASL recognition → text/voice; captioning; API | Yes | Web/API | Live |
| **Signapse** | English→BSL/ASL generative avatars; live streams | Yes | API/signage | Live ($0–$780/mo) |
| **Rylo (ex-Nagish, acquired sign.mt)** | Multilingual sign↔spoken, free; $85M raise | Yes | Mobile/web | Live |
| **SignAll** | ASL Chat + multi-camera rig | Yes | Desktop rig | Status uncertain |
| **SLAIT** | Was real-time ASL (~200 words) → **pivoted to ASL education** (2023) | — | — | Pivoted |
| Google SignGemma | Announced on-device ASL→text | Claimed | Mobile/laptop | **No public weights** |
| **Leap Motion / Myo** | Gesture/sEMG input | — | — | **Commercial failures** (cautionary) |

**Critical:** none of these perform computer actions. The only sign→action paths that exist are Google's *indirect* route (sign → text → Gemini executes the English) and academic/hackathon demos.

---

## 3. The gap (what is still missing)

1. **Sign-as-command.** Zero products map sign → system action. The whole field maps sign → human-readable output.
2. **Command grammar** — constrained lexicons, high recall, graceful fallback. Translators optimise BLEU; control needs latency and precision.
3. **Confirmation / correction / undo.** No translator needs reversibility; every input method does. Nobody has built it.
4. **Confidence as a product surface.** "I'm 60% sure you mean X" is a control primitive, not a translation concern.
5. **Per-user calibration.** All shipping systems are signer-independent, one-size-fits-all. SL2T handles only handedness (10% left-handed) and one-handed signing.
6. **Non-manual markers as UI affordances.** Every system models face/body *to improve translation*; nobody uses brows/gaze/nod as *control modifiers*.
7. **Multilingual sign as one system.** SL2T is multilingual but mobile-only, closed, and translation-only.

---

## 4. Sign languages: the honest answer to "global"

**There is no universal sign language.** International Sign (IS) exists but is a contact pidgin for international events, not a full natural language with a standard corpus — it cannot be the target. The count is ~300 sign languages, not one.

**But "all of them" is an architecture decision, not 300 roadmaps**, because:

| Pipeline stage | Language-dependent? |
|---|---|
| Camera capture, landmark extraction (hands/face/body) | **No** |
| Segmentation / endpointing (motion-based) | **Mostly no** |
| Shared sign encoder (pose-based, gloss-free) | **Shared, jointly trained** — Google: 50+ languages jointly > single-language |
| Language ID + spoken-language output | **Yes** — a small head/adapter per language |
| Intent resolution (Jev) | **No** — operates on text/structured state |
| Action execution | **No** |

**Therefore the model is: 1 pipeline + N "language packs" (encoder adapter + output tokenizer + command lexicon + evaluation set).** Adding a language is a data problem, not a re-architecture.

**Quality must be tiered and disclosed** (the Deaf community explicitly rejects universality claims — 200+ languages, plus Black ASL, late learners, one-handed signers, regional dialects):

- **Tier A — Full:** dictation + action vocabulary + per-user calibration (ASL, ISL, BSL, DGS, Libras initially)
- **Tier B — Dictionary + calibration:** isolated/command vocabulary only, no free-form dictation (most languages)
- **Tier C — Community packs:** user/org-submitted vocabulary, clearly labelled as community, unverified quality

**Recommendation for V1 shipping language:** **ASL** — it is the sign language of English, has the largest commercial-clean isolated corpus (PopSign, CC BY 4.0), the only public benchmark (**FLEURS-ASL**), an existence proof from Google, and the deepest tooling. **ISL ships as the second pack** (India-first advantages: 5.07M hearing-disabled per Census 2011, <300 certified interpreters, NEP-2020 standardisation mandate, cheaper annotation labour) — the research argues ISL is the better *market*, but ASL is the better *first proof* because the data and evaluation exist. Build the multilingual encoder from day one so ISL is a pack, not a rewrite.

---

## 5. V1 definition (narrow surface, general core)

**Two modes, one pipeline:**

- **Mode A — Sign Dictation:** sign → live text at the cursor → user accepts/edits. Works for *any* supported sign language immediately (output is the user's own written language). This is the "Wispr Flow for sign."
- **Mode B — Sign → Action:** sign → intent → **Jev** typed decision → executed computer action with risk tiers, confirmation and undo. This is the differentiator.

**V1 action set (local-first, zero OAuth):** open/switch app · open URL · focus window · local file search · browser search · AI assistant query · clipboard/screenshot · basic browser form fill. **No Gmail/Slack in V1** (Google restricted-scope verification + annual CASA assessment is a real deliverable — see §8).

**V1 language:** ASL (Tier A) with the multilingual encoder scaffolded and ISL as pack #2.
**V1 platform:** **Linux first** (matches dev machine), arch-portable control layer behind a capability-probed abstraction.

**V1 is successful when** a real signer can open ALive, sign naturally, be understood, complete a useful computer task, continue with another request, correct a mistake, and trust the result — without a tutorial.

---

## 6. Technical approach

### 6.1 Perception
- **Primary:** MediaPipe Tasks (Hand Landmarker + Face Landmarker + Pose Landmarker lite), Apache-2.0, all three OSes, CPU-capable; mirrored into Rust via the Tasks C API (`mediapipe-rs` wraps it; hands/pose need ~1–2 weeks FFI work) or a Python sidecar as a temporary fallback.
- **Fallback / likely winner at scale:** **RTMW/DWPose 133 whole-body keypoints as a single ONNX model** via the `ort` crate — one model for hands+face+body (both signing and pointing), fully Apache-2.0, no FFI, RTMPose-m ≈ 11 ms CPU.
- **Known defects to design around:** MediaPipe's hand ROI heuristic breaks for palms perpendicular to the camera (IoU 57% heuristic vs 63% learned, worst case 3%) → use `mediapipe-hand-crop-fix`; face mesh distorts under head pitch and can drop tracks at distance; `LIVE_STREAM` drops frames under load (use VIDEO mode for the temporal path).
- **Resolution:** keep the hand ROI ≥200 px on its long side → capture 1080p and crop ROIs at native resolution rather than feeding full frames.
- **Do not build on:** YOLO-pose (AGPL), HaMeR/WiLoR/Sapiens (GPU-bound + non-commercial), Maxine (RTX-only), ML Kit (mobile-only).

### 6.2 Streaming / endpointing (the hard, unsolved-for-gesture part)
Design validated by literature: causal dilated TCN over landmarks with an explicit **end-of-sign (EOS) token**, analysis window ~1.2 s, right context 0.3–0.5 s, history 2.4 s, stride 1 frame.
- **Tentative endpoint:** 300–400 ms of low-motion pause → emit *speculative* segment.
- **Commit endpoint:** 600–800 ms pause **AND** EOS fired → commit (this converges cues; in ASR it lifted commit precision 33.6 → 87.5).
- **Debounce:** boundary p > 0.8–0.9 and minimum segment 0.2–0.3 s.
- **Two-pass:** low-latency tentative → re-score with lookahead before hand-off.
- **Why:** human turn-taking gap averages **200 ms** (Stivers et al., PNAS 2009); >700–800 ms "feels broken."

### 6.3 Non-manual markers
Mouth is the money feature: **+6–8 accuracy points on hard classes** (Body+Mouth 88.2% vs Body 80.5%); **eyes add ~0** once body is present. Role shift (reported speech — "tell Rahul I'll be late") matters but has **no mature detector**; V1 detects only its *boundary* (sustained torso yaw + head-pose + gaze change) as a heuristic. Do not attempt full mouth-morpheme taxonomy or fine eyebrow semantics in V1.

### 6.4 Performance (measured, not guessed)
- **Frames must never cross Tauri `invoke`:** measured 1,811 ms frame stall (a 6.3 MB frame becomes a 22.5 MB JSON number array) vs **46 ms and ~0 ms main-thread block via a custom URI protocol**. Use `register_uri_scheme_protocol` for pixels; `invoke`/Channel for keypoints and control.
- **All inference in Rust** via `ort` with per-platform EPs (CoreML / DirectML / OpenVINO / CPU). `burn`/`candle` are not viable here (WebGPU dispatch 24–71 µs, 1–2% of peak). WebGPU **does not work in WebKitGTK** on Linux — never put production inference in the WebView.
- **Local perception budget:** ~20–35 ms optimistic / 60–120 ms realistic per frame. **Cloud LLM is the real latency risk** (400–1,500 ms p95) → it must never block the perception loop.
- **Thermal:** thin laptops lose 20–40% sustained throughput; use detect-and-track, frame skipping with latest-frame semantics, model tiering, and thermal-aware shedding (`ProcessInfo.thermalState`, `pmset -g therm`).

### 6.5 Desktop shell (Tauri 2)
Verdict: **yes, with the WebView treated as chrome only.** tauri 2.11.5 (2026-07-01). Knowns to design around:
- Linux WebKitGTK is the weakest link (a Tauri maintainer won't fully recommend Tauri on Linux today) → **capture in Rust, never `getUserMedia`** (WebKitGTK denies media by default; needs manual settings + X11 backend).
- **Wayland:** global shortcuts unsupported; overlay layer-shell broken upstream; **no portable way to enumerate/target windows**; synthetic input requires the portal (`RemoteDesktop` + libei) — so the action layer must be built on *inputs + accessibility targets*, not window handles.
- macOS: transparency needs `macOSPrivateApi` + Accessibility permission → **Mac App Store is off the table**; ship a notarized DMG ($99/yr). Windows: bundle the WebView2 bootstrapper; UIPI silently blocks automation of elevated apps.
- Keep the Electron-on-Linux escape hatch documented; CEF backend is the maintainers' long-term fix.

### 6.6 Stack validation
Tauri 2 + Rust + React/TS + Vite + Tailwind + shadcn/ui — **validated**, with these bindings: Rust owns camera + ML + OS control + journal; React owns the orb/chrome; `nokhwa` 0.10.x for capture; `ort` 2.0.0-rc.x pinned exactly (hard ORT version coupling); `atspi`/`axuielement`/`uiautomation` for the semantic layer; Python sidecar **not justified** (multi-GB bundles) unless a non-exportable model forces it.

---

## 7. Jev's role and economics (verified from primary sources)

**Jev by TypeSafe AI is real** — launched 2026-09-15, SF, $40M seed led by DCVC, founder Diogo Almeida (RLHF co-inventor). Note the name collision: it is **not** the old Scala "Typesafe" (now Akka).

- **Endpoint:** `POST https://api.typesafe.ai/v1/systemone`, `model: jev-1.13.0` (alias `jev-latest`).
- **Primitives:** **Choice** (≤255 options → `choice` + `probabilities` + `confidence`), **Score** (2–10 levels → weighted score + legend + confidence), **Noul** (yes/no → 0–1 probability). **All questions run in parallel in one call** and adding questions barely changes latency.
- **Pricing:** **$0.042 per 1M input tokens; output tokens free.** Rate limits 250k tok/s, 1,200 req/min (dynamic during early access). Context 64k (32k for state + longest question).
- **Input is TEXT ONLY** — no image/audio/video. **The vision pipeline is entirely ours**; Jev is the decision layer, exactly as the brief's §13–14 says. Pre-process landmarks/recognised signs into text/structured `state`.
- **Latency:** vendor 70–500 ms; independent benchmarks p50 ≈176–422 ms, p95 ≈336–542 ms, accuracy 91.7% on tool-call risk classification (n=60), calibration held (no wrong answer at confidence 1.0).
- **Limits:** no fine-tuning (same weights for all), English-first, no streaming, one judgment per question, degrades with irrelevant context.
- **Cost per user-hour:** ~400-token state → **$0.0000168/decision → ~1¢/user-hour at 600 utterances/hour**. Compare a frontier LLM at ~$3.90/hour for the same volume (**~390× more**).
- **Verdict:** **Jev is the right decision layer and the economics are trivially good.** Use it behind a `DecisionProvider` interface (default Jev, pinned version), with a function-calling LLM as the low-confidence fallback. Do not hard-wire; it is 9 days old with a 99.83% uptime page and fresh incidents.

**Where Jev earns its place in ALive:** intent Choice (which action), Noul gates (`needs_confirmation`, `is_risky`, `is_multi_step`, `user_is_addressing_computer` — the anti-false-trigger guard), Score (confidence/risk/urgency), tool-argument extraction, and post-action verification ("did the state actually change?").

---

## 8. Computer control (safest practical path)

**Three tiers, in priority order:**
1. **OS-native deterministic** (ships first, $0, sub-300 ms): `xdg-open`/`open`/`start`, deep links (`mailto:`, `slack://`, `ms-settings:`), app activation, window focus, indexed file search (Tracker/`mdfind`/Everything), CLI wrappers.
2. **Browser + accessibility tree** (weeks, ~$0.001–0.01/step): Playwright/CDP or Rustwright, `snapshot → click(ref) → type → read-back`. A11y tree = 200–400 tokens vs 3,000–5,000 for screenshot vision (**7.5–12.5× cheaper**).
3. **Computer-use model as a gated escape hatch** (last resort): Claude/Gemini CUA for canvas-rendered or legacy surfaces only. **3–8 s per step** — incompatible with ALive's instant feel, so never the default; bound with allowlists, step/time/cost caps, and human confirmation for writes. **Promote any CUA flow that succeeds 3× into a deterministic script.**

**Reliability:** verify outcomes independently (never trust "done"), idempotency keys, no blind retries on Send/Submit/Pay/Delete, typed error taxonomy, immutable action journal, one-click human takeover.

**Prompt-injection defense:** treat all tool output and page text as untrusted data (envelope it), provenance-tag every result, per-task tool allowlists, confirm every write, hash tool descriptions to detect rug-pulls, never build shell commands from model output.

**Integrations:** V1 = local files + OS + browser + AI. **Defer Gmail** — restricted scopes trigger annual CASA assessment ($540–1,800/yr + weeks of process). Calendar (`calendar.events`) is *sensitive*, not restricted → no CASA, ship in V2. **Do not ship an MCP client in V1** (MCP servers = arbitrary code execution, documented RCE class, tool poisoning, confused-deputy OAuth, supply-chain backdoors). Add MCP later, remote-first, behind a gateway.

---

## 9. UX

**Design law:** the failure bar is not "fast models" — it's **state legibility**. Siri's in-place retype, Gemini's 2–3 s interrupt lag and Wispr's cloud stalls are the same error: the interface stopped telling the truth about what it was doing.

**Measured targets (evidence-backed):**

| Milestone | Target |
|---|---|
| Activation → armed | ≤100 ms p95 |
| First feedback (skeleton lock) | ≤100–200 ms |
| First partial gloss | ≤300 ms from motion onset |
| Endpointing (motion stops → commit) | 300–600 ms |
| Commit (text finalized) | p50 ≤400 ms, p95 ≤900 ms |
| Barge-in / interruption recovery | ≤300 ms |
| Undo window | 5–10 s inline + session ⌘Z |

**Key decisions:**
- **Hold-to-sign** as the consent gate and hard stop — the sign analog of push-to-talk. Suppresses false triggers (face-touching occurs 15.7–23×/hour and looks like signing onset) and gives a physical interrupt.
- **Camera = small floating picture-in-picture, never a video call.** Silent computing evidence says users pay for privacy-by-default; 35% of DHH respondents worry about *showing their surroundings*.
- **Orb states** (each visually distinct, animation *communicates* not decorates): idle → activating → camera-active → signer-detected → interpreting → intent-forming → ready → executing → done → confirm-needed → error → recovery.
- **No pipeline exposure.** Never show "calling Jev," "temporal encoder," "tool router."
- **Partial gloss ribbon with dashed "candidate" tokens** instead of silent rewrites (Siri's retype is the #1 anti-pattern).
- **Correct by re-signing the unit**, cancel by dropping hands/gesture, undo always available.

---

## 10. Silent computing (the broader market — honestly sized)

Evidence is real but **contextual, not mass-market**: 6% use voice assistants in public, 1.3% at work, 47% feel awkward talking to a machine. Top barriers: social exposure, disturbing others, confidentiality. Counter-evidence: Jabra/LSE (Oct 2025) finds voice crossing early-adopter thresholds for AI work; Pew (2026) finds 49% of US adults use AI chatbots. Meta shipped an sEMG Neural Band at $799, but its handwriting decode is **20.9 WPM vs 36 WPM on a phone keyboard** — silent input is *slower*, and wears hardware.

**Caveat ALive must confront: "silent" ≠ "unobtrusive."** Mid-air signing is *visible*. The wedge is people who are *already* signers, plus private-by-default contexts; the day-1 pitch is **not** "silent for everyone."

---

## 11. Deaf-user evidence and etiquette (non-negotiable)

- **"Nothing about us without us" is a design gate, not a value statement.** 81% of Deaf respondents worry about "hearing people making money"; 71% about "few deaf people leading"; 91% (ASLT survey) worry about hearing people profiting from ASL. Only 62% said "yes, develop" — conditional support is the base case.
- **Gloves are poison:** only 19% of signers are comfortable with gloves vs **91% with a mobile app**. The r/deaf thread "Mods, can we please ban sign language gloves?" is the community's stated position.
- **Never claim universality** (200+ languages, dialects, Black ASL, late learners, one-handed signers). Never position as an interpreter replacement; refuse medical/legal. Never medicalise ("hearing loss" framing is rejected).
- **Latency matters more than polish:** captions delayed 7–12 s score 3.66/7 vs 5.09 synced; "I would rather have incomplete captions in sync than better captions lagging behind."
- **Privacy:** every concern category in a 61-signer study was privacy-related (video misuse 68%, being recognised 41%, **showing surroundings 35%**); only 6% had none. Design: on-device by default, hard "nothing recorded," narrow field of view, blurred background, separate opt-in per purpose (testing vs training), camera-off physical indicator, ASL-available consent.
- **Confirmation gestures:** there is **no published study** on nod vs sign vs tap for consequential actions — this is an open empirical question for ALive's own research, not something to invent from intuition.

---

## 12. Costs and pricing

- **Jev:** ~1¢/user-hour (600 utterances/hr) — negligible.
- **Cloud LLM:** ~$0.0014/utterance (blended) → Normal user (2,600 utt/mo) ≈ **$3.64/mo**.
- **On-device perception: $0 marginal.** This is the gross-margin shield — it is the reason ALive can be priced like a consumer product. Real-time cloud vision is fatal: **$0.58/min ≈ $34.80/hour** at 1 fps with the cheapest Gemini, so cloud vision must be opt-in and ≤1 fps.
- **Monthly per active user:** Light **$2.38** / Normal **$5.14** / Heavy **$15.10–15.82**; heavy *agent* usage **$36–50** (unviable unless metered).
- **Pricing:** Free (300 utterances/mo, on-device only) → **Pro $14/mo** (3,000 utterances, 25 agent tasks, 500 cloud-vision frames) → **Power $29** → **agent credit packs $5/100** (never bundle unlimited agents). Adjacent products cluster $8–20/mo. Meter utterances, agent credits, cloud-vision frames, clip storage — **never meter on-device perception.**

---

## 13. Risk list

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| 1 | **Signer-dependence cliff** — great for one signer, useless for the next | Existential | Per-user calibration as a first-class flow; signer-dependent V1 contract; measure cross-signer in CI |
| 2 | **Google ships the command layer** (SL2T → Gemini already hints it) | High | Own the command grammar, confirmation/undo, and per-user calibration — things a dictation model cannot provide |
| 3 | **Deaf community rejects it as extractive** | High | Deaf leadership (not consultation), non-replacement scope, published safeguards, data governance |
| 4 | **Linux/Wayland platform limits** (no window targeting, no global shortcuts) | Medium | Capability-probed action layer; portal-first; documented degradation; Electron/CEF escape hatch |
| 5 | **Latency creeps past the "continuous" threshold** | Medium | Never block on cloud; async decision; strict budget table in CI |
| 6 | **License contamination** (How2Sign/OpenASL/WLASL/iSign are non-commercial) | Medium | Commercial-clean training only (PopSign CC BY 4.0, INCLUDE CC BY 4.0) + own recorded corpus |
| 7 | **False triggers / camera always-on distrust** | Medium | Hold-to-sign gate, Noul "user_is_addressing_computer" gate, nothing-recorded default, physical indicator |
| 8 | **Jev is 9 days old** (vendor/rate-limit/uptime risk) | Low–Med | `DecisionProvider` abstraction, LLM fallback, pin version, re-evaluate quarterly |
| 9 | **Unit economics of heavy agent use** | Low | Meter agent credits; keep perception on-device |
| 10 | **Scope sprawl across 300 languages** | Medium | Tiered language packs with disclosed quality; never claim universality |

---

## 14. Validation plan (before scaling)

**Wizard-of-Oz first** — never build the model before proving the interaction.
- **Phase 0 (wk 0–2, n=5–6):** WoZ sign→action, think-aloud, ASL interpreter. Measures mental model and latency tolerance.
- **Phase 1 (wk 2–4, n=6–8):** WoZ + task battery, NASA-TLX, SEQ, trust probe, **correction cost** (retries/utterance — economics-relevant, since every retry bills).
- **Phase 2 (wk 4–6, n=8–12):** on-device MVP field study, 2 weeks, SUS (≥68) + UEQ.
- **Missing studies ALive must run:** (a) confirm-gesture study (nod / explicit sign / tap) with visible echo + cancel; (b) latency-tolerance curve (500 ms / 1 s / 2 s) measuring *task abandonment*, not preference; (c) camera-consent study (on-device vs cloud, nothing-recorded default); (d) longitudinal diary, 8–12 signers, 2–4 weeks.
- **Recruiting and ethics:** through Deaf organisations and gatekeepers (not "disability lists"); screen for ASL fluency, not diagnoses; **certified interpreters mandatory** (never a family member), paid prep time, two interpreters rotating past 60 min; pay **$25–75/hr** (or $100–150 per 60–90 min session), promptly; bilingual instruments (ASL video + English); separate consent for testing vs training; prefer DHH moderators.
- **Budget:** ~$11–15k for the 6-week program.
- **Kill criteria:** top-1 intent accuracy <85% after Phase 1 iteration · p95 end-to-end >800 ms · >30% refuse persistent camera-on · >20% of actions need >1 retry · median engaged user lands in heavy-agent usage with WTP <$20 · cannot recruit through Deaf orgs with genuine enthusiasm, or recurring "surveillance tool" feedback.

**Public benchmark:** evaluate on **FLEURS-ASL** (Google's ASL→English benchmark, sd-test) so ALive's numbers are comparable to SL2T's 70 BLEURT.

---

## 15. Primary sources (selection)

**Sign AI:** [DeepMind SL2T](https://deepmind.google/blog/putting-sign-language-ai-into-users-hands/) (2026-08-12) · [Talksign Palm 1.0](https://techcabal.com/2026/05/20/talksign-launches-ai-powered-palm-1-0-and-echo-1-0/) (2026-05-20) · [FLEURS-ASL](https://www.kaggle.com/datasets/googleai/fleurs-asl) · [SignGPT arXiv:2609.21709](https://arxiv.org/abs/2609.21709) · [Online CSLR, EMNLP 2024](https://arxiv.org/abs/2401.05336) · [Gloss-Free SLT unbiased eval, CVIU 2026](https://arxiv.org/html/2603.13240) · [Rethinking SLT: signer dependence, EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.997/) · [25 years of CSLR](https://www.sciencedirect.com/science/article/pii/S0306457324001341) · [iSign, ACL Findings 2024](https://arxiv.org/html/2407.05404v1)
**Jev:** [typesafe.ai](https://typesafe.ai/) · [docs](https://docs.typesafe.ai/introduction) · [models](https://docs.typesafe.ai/models) · [API](https://docs.typesafe.ai/api) · [launch post](https://typesafe.ai/blog/introducing-system-one-models-and-jev) · [jev-benchmark](https://github.com/themsquared/jev-benchmark) · [status](https://status.typesafe.ai)
**Deaf community:** [Global DHH perspectives, arXiv:2512.08839](https://arxiv.org/html/2512.08839v1) · [ASLT US survey, ASSETS'23](https://www.microsoft.com/en-us/research/wp-content/uploads/2023/10/Automatic_Sign_Language_Translation_Perspectives.pdf) · [Gallaudet, arXiv:2410.21358](https://arxiv.org/html/2410.21358v2) · [Caption latency, ASSETS'26](https://arxiv.org/html/2609.11408v1) · [Dataset privacy, ASSETS'20](https://danibragg.com/papers/Exploring_Collection_of_Sign_Language_Datasets.pdf) · [Deaf-led call, arXiv:2403.02563](https://arxiv.org/html/2403.02563v1) · [Why sign gloves don't help, The Atlantic](https://www.theatlantic.com/technology/archive/2017/11/why-sign-language-gloves-dont-help-deaf-people/545441/)
**Perception/perf:** [MediaPipe Holistic](https://developers.google.com/edge/mediapipe/solutions/vision/holistic_landmarker) · [hand ROI flaw, arXiv:2405.03545](https://arxiv.org/html/2405.03545) · [MPH non-manuals, arXiv:2403.10367](https://arxiv.org/html/2403.10367v1) · [RTMW](https://arxiv.org/html/2407.08634v1) · [Facial features ablation, arXiv:2507.20884](https://arxiv.org/html/2507.20884v1) · [SignMouth](https://arxiv.org/html/2509.10266v2) · [Tauri IPC bandwidth wall](https://www.mechanicalrock.io/blog/it-s-physics-not-a-bug-the-ipc-bandwidth-wall-in-webview-apps) · [ORT CoreML EP](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html) · [nokhwa](https://docs.rs/nokhwa/latest/nokhwa/) · [ort](https://ort.pyke.io/)
**Control/safety:** [XDG RemoteDesktop portal](https://flatpak.github.io/xdg-desktop-portal/docs/doc-org.freedesktop.portal.RemoteDesktop.html) · [atspi](https://docs.rs/atspi/latest/atspi/index.html) · [axuielement](https://docs.rs/axuielement/latest/axuielement/index.html) · [uiautomation](https://docs.rs/crate/uiautomation/latest) · [MS HITL pattern](https://github.com/microsoft/ai-agent-runbooks/blob/main/02-patterns/Human-in-the-Loop-Review-and-Approval/Human-in-the-Loop-Review-and-Approval.md) · [Anthropic trustworthy agents](https://www.anthropic.com/research/trustworthy-agents) · [OpenAI Operator system card](https://cdn.openai.com/operator_system_card.pdf) · [Levels of Autonomy, arXiv:2506.12469](https://arxiv.org/abs/2506.12469) · [MCP security](https://modelcontextprotocol.io/docs/2026-07-28/tutorials/security/security_best_practices) · [Gmail restricted scopes](https://developers.google.com/identity/protocols/oauth2/production-readiness/restricted-scope-verification)
**UX/economics:** [Stivers et al., PNAS 2009](https://www.pnas.org/doi/10.1073/pnas.0903616106) · [Turn-taking latency](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2015.00731/full) · [Wispr Flow](https://wisprflow.ai/) · [Tauri overlay](https://blog.manasight.gg/why-i-chose-tauri-v2-for-a-desktop-overlay/) · [Section508 usability testing](https://www.section508.gov/develop/usability-testing-with-people-with-disabilities/)
**Datasets (license-checked):** [PopSign CC BY 4.0](https://signdata.cc.gatech.edu/view/datasets/popsign_v1_0/index.html) · [INCLUDE CC BY 4.0](https://zenodo.org/records/4010759) · [ISL-CSLTR CC BY 4.0](https://data.mendeley.com/datasets/kcmpdxky7p/1)
