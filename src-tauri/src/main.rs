//! ALive desktop app: Tauri shell + pipeline thread + commands + frame protocol.
//!
//! Data flow (all Rust, WebView is chrome only):
//! camera -> detect/pose (RTMW) -> motion -> endpointer -> trajectory match
//! -> lexicon -> DecisionProvider (Jev if ALIVE_JEV_API_KEY, else mock) ->
//! router -> EventBus -> WebView events. The pipeline NEVER executes actions;
//! the frontend calls `execute` explicitly (T1 auto-run only when the user
//! flips "Live actions" on; dry-run is the default).

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering},
    Arc, Mutex,
};
use std::time::Instant;

use alive_core::{AppMode, Config, EventBus, EventKind, OrbState};
use alive_intent::{
    alive_questions, calibrated_confidence, ChoiceAnswer, Decision, DecisionProvider,
    DecisionRequest, JevClient,
};
use alive_ipc::*;
use alive_sign::matcher::{hand_centroid, TemplateBank, HAND_A, HAND_B};
use alive_sign::{LanguagePack, NBest};
use tauri::{Emitter, Manager, State};

// Motion gate thresholds: fraction of subsampled pixels that moved.
// Wake opens inference; sleep (hysteresis) closes it. A finger flick moves
// ~2-5% of samples; high-gain noise moves ~0.05%.
const MOTION_WAKE: f32 = 0.003;
const MOTION_SLEEP: f32 = 0.0015;

/// Subsampled luma motion score: fraction of sampled pixels whose G channel
/// moved past the noise floor. ~0.2ms at 480p. Returns 0.0 with no history
/// (first frame / resolution change) and on global-shift frames (>45% moved
/// = exposure snap or flash, not a hand).
fn motion_score(data: &[u8], width: u32, height: u32, prev: &mut Vec<u8>) -> f32 {
    const STRIDE: usize = 4;
    const NOISE: u8 = 12;
    let n = width as usize * height as usize;
    let sampled = n.div_ceil(STRIDE);
    if prev.len() != sampled || data.len() < n * 3 {
        prev.clear();
        prev.extend(
            (0..n)
                .step_by(STRIDE)
                .map(|i| data.get(i * 3 + 1).copied().unwrap_or(0)),
        );
        return 0.0;
    }
    let mut moved = 0usize;
    let mut signed: i64 = 0;
    let mut abs_sum: i64 = 0;
    for (j, slot) in prev.iter_mut().enumerate() {
        let g = data[j * STRIDE * 3 + 1];
        let d = g as i16 - *slot as i16;
        if g.abs_diff(*slot) > NOISE {
            moved += 1;
        }
        signed += d as i64;
        abs_sum += d.unsigned_abs() as i64;
        *slot = g;
    }
    let frac = moved as f32 / sampled as f32;
    if frac > 0.45 {
        return 0.0; // global shift: flash / scene cut
    }
    // Exposure coherence: a large coherently-signed shift is a gain snap,
    // not a hand (hands stay under ~10% of frame).
    if frac > 0.25 && abs_sum > 0 && signed.abs() * 10 > abs_sum * 7 {
        return 0.0;
    }
    frac
}

/// Motion-gate policy: pure decision logic (no atomics, unit-tested).
/// Ticks on every fresh armed frame; answers run-pose? and rescan?
#[derive(Debug)]
struct GatePolicy {
    wake_latch: bool,
    slow_ema: f32,
    slow_hits: u8,
    was_armed: bool,
    last_run_ms: u64,
    last_scan_ms: u64,
    last_seen_ms: u64,
}

#[derive(Debug, PartialEq, Eq)]
struct GateDecision {
    run: bool,
    rescan: bool,
}

impl GatePolicy {
    fn new(now_ms: u64) -> Self {
        Self {
            wake_latch: false,
            slow_ema: 0.0,
            slow_hits: 0,
            was_armed: false,
            // Heartbeats due immediately at startup.
            last_run_ms: now_ms.saturating_sub(10_000),
            last_scan_ms: now_ms.saturating_sub(120_000),
            last_seen_ms: 0,
        }
    }

    fn tick(
        &mut self,
        motion: f32,
        now_ms: u64,
        armed: bool,
        ever_tracked: bool,
        tracking: bool,
        first: bool,
    ) -> GateDecision {
        let arm_edge = armed && !self.was_armed;
        self.was_armed = armed;
        if !armed {
            self.wake_latch = false;
            self.slow_ema = 0.0;
            self.slow_hits = 0;
            return GateDecision {
                run: false,
                rescan: false,
            };
        }
        // Fast latch with hysteresis.
        let mut latch_edge = false;
        if motion >= MOTION_WAKE {
            if !self.wake_latch {
                latch_edge = true;
            }
            self.wake_latch = true;
        } else if motion < MOTION_SLEEP {
            self.wake_latch = false;
        }
        // Slow-sign integrator: sub-threshold flickers accumulate.
        self.slow_ema += motion - self.slow_ema / 15.0;
        if self.slow_ema > 0.01 {
            self.slow_hits = self.slow_hits.saturating_add(1);
        } else {
            self.slow_hits = 0;
        }
        let slow_wake = self.slow_hits >= 6;
        if tracking {
            self.last_seen_ms = now_ms;
        }
        // Entry sweep till first lock; heartbeat after.
        let sweep_due = !ever_tracked && now_ms.saturating_sub(self.last_run_ms) >= 5000;
        let beat_due = ever_tracked && now_ms.saturating_sub(self.last_run_ms) >= 2000;
        let run = self.wake_latch || slow_wake || arm_edge || sweep_due || beat_due;
        // Rescan backstop: short when recently active, long in dead rooms.
        let recently_active = now_ms.saturating_sub(self.last_seen_ms) < 30_000;
        let backstop_ms = if recently_active { 5000 } else { 60_000 };
        let backstop_due = now_ms.saturating_sub(self.last_scan_ms) >= backstop_ms;
        let rescan = first || (!tracking && (arm_edge || latch_edge || slow_wake || backstop_due));
        if run {
            self.last_run_ms = now_ms;
        }
        if rescan {
            self.last_scan_ms = now_ms;
        }
        GateDecision { run, rescan }
    }
}

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

struct Shared {
    config: Mutex<Config>,
    pack: LanguagePack,
    bank: Mutex<TemplateBank>,
    bus: EventBus,
    journal: Mutex<alive_journal::Journal>,
    journal_path: PathBuf,
    orb: Mutex<OrbState>,
    provider: Mutex<String>,
    vision: Mutex<String>,
    fps: Mutex<f64>,
    cap_ms: Mutex<f64>,
    lm_ms: Mutex<f64>,
    frame_jpeg: Mutex<Vec<u8>>,
    last_trail: Mutex<Vec<(f32, f32)>>,
    // Frame slot: capture writes, infer reads (drops stale). Arc: publish +
    // consume are refcount bumps, zero frame memcpy (~36MB/s eliminated).
    slot_seq: AtomicU64,
    slot_frame: Mutex<Option<Arc<alive_perception::Frame>>>,
    // Result slot: infer writes landmarks, main consumes.
    result_seq: AtomicU64,
    result: Mutex<Option<Arc<alive_perception::Landmarks>>>,
    cap_fps: Mutex<f64>,
    seq: AtomicU64,
    running: AtomicBool,
    armed: AtomicBool,
    has_camera: AtomicBool,
    // Motion gate telemetry: 0.0..1.0 moved-pixel fraction as u32 bits.
    motion_bits: AtomicU32,
    // Tracker state mirror (extractor lives in the infer thread).
    tracking: AtomicBool,
    ever_tracked: AtomicBool,
    // Motion gate state (set by infer each tick; read by main).
    gate_open: AtomicBool,
    // Overlay snapshot for the PiP thread (frame-space coords).
    overlay: Mutex<Arc<OverlayFrame>>,
    // Understanding ticks emitted (upsampler virtuals + real).
    ticks_total: AtomicU64,
    ticks_window: Mutex<(Instant, u64)>,
    upsample_hz: Mutex<f64>,
    // Active user-overlay mode (None = hidden). Read by the overlay
    // webview on mount so it never misses the show event.
    overlay_mode: Mutex<Option<String>>,
}

/// Skeleton + ribbons + lead dots drawn over the PiP preview.
/// All points are frame-space pixels; the PiP thread scales them.
#[derive(Debug, Clone, Default)]
struct OverlayFrame {
    /// Body skeleton segments (133-kp subset links).
    segs: Vec<((i16, i16), (i16, i16))>,
    /// Per-hand trail ribbons (PCHIP history, newest last).
    ribbons: [Vec<(i16, i16)>; 2],
    /// Per-hand lead dots (Kalman predict-to-now).
    leads: [Option<(i16, i16)>; 2],
}

type Ctx<'a> = State<'a, Arc<Shared>>;

fn set_orb(s: &Shared, to: OrbState) {
    let mut o = s.orb.lock().unwrap();
    let from = *o;
    if from == to {
        return;
    }
    *o = to;
    s.bus.publish(EventKind::OrbStateChanged { from, to });
}

fn orb_name(o: OrbState) -> String {
    serde_json::to_string(&o)
        .unwrap_or_default()
        .trim_matches('"')
        .to_string()
}

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

fn candidate_roots() -> Vec<PathBuf> {
    let mut v = vec![PathBuf::from(".")];
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            v.push(dir.to_path_buf());
            if let Some(p) = dir.parent() {
                v.push(p.to_path_buf());
            }
        }
    }
    v.push(PathBuf::from("/home/govinda/sign-flow"));
    v
}

fn find_dir(name: &str) -> Option<PathBuf> {
    candidate_roots()
        .into_iter()
        .map(|r| r.join(name))
        .find(|p| p.is_dir())
}

fn state_dir() -> PathBuf {
    if let Some(p) = find_dir("alive-state") {
        return p;
    }
    let p = candidate_roots()[0].join("alive-state");
    let _ = std::fs::create_dir_all(&p);
    p
}

// ---------------------------------------------------------------------------
// Offline decision path (mock driven by the lexicon, not scripts)
// ---------------------------------------------------------------------------

fn mock_decide(pack: &LanguagePack, dto: &DecideRequestDto) -> Decision {
    use std::collections::HashMap;
    let top = dto.nbest.first();
    let (intent, conf) = match top.and_then(|h| pack.match_gloss(&h.gloss).map(|e| (e, h.p))) {
        Some((entry, p)) => {
            let name = entry
                .intent
                .clone()
                .unwrap_or_else(|| "dictate".to_string());
            (name, p.clamp(0.05, 0.99))
        }
        None => (
            "dictate".to_string(),
            top.map(|h| h.p * 0.4).unwrap_or(0.1).clamp(0.05, 0.6),
        ),
    };
    let tier = pack.intent(&intent).map(|d| d.risk_tier).unwrap_or(1);
    let mut probs = HashMap::new();
    probs.insert(intent.clone(), conf);
    probs.insert("dictate".to_string(), (1.0 - conf).max(0.01));
    probs.insert("cancel".to_string(), 0.01);
    let sum: f64 = probs.values().sum();
    for v in probs.values_mut() {
        *v /= sum;
    }
    let peak = probs.get(&intent).copied().unwrap_or(0.0);
    Decision {
        intent: ChoiceAnswer {
            choice: intent,
            probabilities: probs.clone(),
            confidence: calibrated_confidence(probs.len(), peak),
        },
        route: "deterministic".to_string(),
        addressing_computer: 0.9,
        needs_confirmation: if tier >= 2 { 1.0 } else { 0.0 },
        is_risky: if tier >= 3 { 1.0 } else { 0.0 },
        is_multi_step: 0.0,
        confidence_score: conf * 5.0,
        risk_score: tier as f64,
    }
}

fn decide_request(pack: &LanguagePack, dto: &DecideRequestDto) -> DecisionRequest {
    DecisionRequest {
        state: serde_json::json!({
            "gloss_nbest": dto.nbest.iter().map(|h| serde_json::json!({"gloss": h.gloss, "p": h.p})).collect::<Vec<_>>(),
            "signal": {"eos_p": dto.eos_p, "hold_ms": dto.hold_ms, "motion_energy": dto.motion_energy, "fingerspell_p": dto.fingerspell_p},
            "context": {"app": dto.app, "url": dto.url, "last_action": dto.last_action},
        }),
        questions: alive_questions(&pack.intent_options()),
    }
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

#[tauri::command]
fn get_status(s: Ctx<'_>) -> StatusDto {
    let cfg = s.config.lock().unwrap();
    StatusDto {
        orb_state: orb_name(*s.orb.lock().unwrap()),
        mode: serde_json::to_string(&cfg.mode)
            .unwrap_or_default()
            .trim_matches('"')
            .to_string(),
        provider: s.provider.lock().unwrap().clone(),
        vision: s.vision.lock().unwrap().clone(),
        fps: *s.fps.lock().unwrap(),
        capture_ms: *s.cap_ms.lock().unwrap(),
        landmark_ms: *s.lm_ms.lock().unwrap(),
        camera: s.has_camera.load(Ordering::Relaxed),
        pack: cfg.language_pack.clone(),
        running: s.running.load(Ordering::Relaxed),
        armed: s.armed.load(Ordering::Relaxed),
        motion: f32::from_bits(s.motion_bits.load(Ordering::Relaxed)),
        tracking: s.tracking.load(Ordering::Relaxed),
        upsample_hz: *s.upsample_hz.lock().unwrap() as f32,
    }
}

#[tauri::command]
fn set_mode(s: Ctx<'_>, mode: String) -> Result<String, String> {
    let m = match mode.as_str() {
        "dictation" => AppMode::Dictation,
        "action" => AppMode::Action,
        _ => return Err(format!("unknown mode {mode:?}")),
    };
    s.config.lock().unwrap().mode = m;
    Ok(mode)
}

#[tauri::command]
fn set_armed(s: Ctx<'_>, armed: bool) -> String {
    s.armed.store(armed, Ordering::Relaxed);
    if armed {
        set_orb(&s, OrbState::CameraActive);
        "armed".to_string()
    } else {
        set_orb(&s, OrbState::Idle);
        "idle".to_string()
    }
}

#[tauri::command]
fn set_live_actions(s: Ctx<'_>, live: bool) -> String {
    s.config.lock().unwrap().dry_run_actions = !live;
    if live {
        "live".to_string()
    } else {
        "dry-run".to_string()
    }
}

/// Start the pipeline thread once. Shared by the `start_session` command
/// and the global-hotkey overlay path.
fn start_pipeline_once(shared: &Arc<Shared>) -> Result<bool, String> {
    if shared.running.load(Ordering::Relaxed) {
        return Ok(false);
    }
    shared.running.store(true, Ordering::Relaxed);
    let s2 = shared.clone();
    std::thread::Builder::new()
        .name("alive-pipeline".into())
        .spawn(move || run_pipeline(s2))
        .map_err(|e| e.to_string())?;
    Ok(true)
}

#[tauri::command]
fn start_session(s: Ctx<'_>) -> Result<String, String> {
    eprintln!("[alive] start_session called");
    if start_pipeline_once(s.inner())? {
        Ok("started".to_string())
    } else {
        Ok("already running".to_string())
    }
}

#[tauri::command]
fn stop_session(s: Ctx<'_>) -> String {
    eprintln!("[alive] stop_session called");
    s.running.store(false, Ordering::Relaxed);
    set_orb(&s, OrbState::Idle);
    "stopped".to_string()
}

#[tauri::command]
async fn decide(s: Ctx<'_>, dto: DecideRequestDto) -> Result<DecideResponseDto, String> {
    let cfg = s.config.lock().unwrap().clone();
    let req = decide_request(&s.pack, &dto);
    let (decision, name): (Decision, String) = if std::env::var("ALIVE_JEV_API_KEY").is_ok() {
        let client = JevClient::from_env(cfg.jev_endpoint.clone(), cfg.jev_model.clone())
            .map_err(|e| e.to_string())?;
        match client.decide(&req).await {
            Ok(d) => (d, "jev".to_string()),
            Err(e) => (
                mock_decide(&s.pack, &dto),
                format!("mock(jev-fallback:{e})"),
            ),
        }
    } else {
        (mock_decide(&s.pack, &dto), "mock".to_string())
    };
    *s.provider.lock().unwrap() = name;
    let r = alive_intent::route_decision(&decision, cfg.act_threshold, cfg.confirm_threshold);
    s.bus.publish(EventKind::IntentDecided {
        intent: r.intent.clone(),
        confidence: r.confidence,
        route: format!("{:?}", r.route).to_lowercase(),
    });
    Ok(DecideResponseDto {
        intent: r.intent,
        confidence: r.confidence,
        route: format!("{:?}", r.route).to_lowercase(),
        addressing_computer: r.addressing_computer,
        needs_confirmation: r.needs_confirmation,
        is_risky: r.is_risky,
        is_multi_step: r.is_multi_step,
        provider_route: r.provider_route,
    })
}

fn tool_from_str(t: &str) -> Result<alive_act::ToolId, String> {
    use alive_act::ToolId::*;
    Ok(match t {
        "open_app" => OpenApp,
        "open_url" => OpenUrl,
        "file_search" => FileSearch,
        "browser_search" => BrowserSearch,
        "ai_query" => AiQuery,
        "clipboard_copy" => ClipboardCopy,
        "screenshot" => Screenshot,
        "type_text" => TypeText,
        "key_press" => KeyPress,
        "mouse_move" => MouseMove,
        "mouse_click" => MouseClick,
        "mouse_drag" => MouseDrag,
        "scroll" => Scroll,
        "wait_ms" => WaitMs,
        "cursor_position" => CursorPosition,
        _ => return Err(format!("unknown tool {t:?}")),
    })
}

#[tauri::command]
fn execute(s: Ctx<'_>, dto: ExecuteRequestDto) -> Result<OutcomeDto, String> {
    eprintln!(
        "[alive] execute {} {:?} dry={}",
        dto.action_id, dto.tool, dto.dry_run
    );
    let cfg = s.config.lock().unwrap().clone();
    let tool = tool_from_str(&dto.tool)?;
    // Safety: the global dry-run switch can only escalate toward safety.
    let dry_run = dto.dry_run || cfg.dry_run_actions;
    let req = alive_act::ActionRequest {
        action_id: dto.action_id.clone(),
        intent: dto.intent.clone(),
        tool,
        args: dto.args,
        risk_tier: dto.risk_tier,
    };
    s.bus.publish(EventKind::ActionStarted {
        action_id: req.action_id.clone(),
        intent: req.intent.clone(),
    });
    set_orb(&s, OrbState::Executing);
    let exec = alive_act::LocalExecutor::new(dry_run, cfg.app_allowlist.clone());
    // Sync command (Tauri worker thread): block_on a private runtime so no
    // MutexGuard is ever held across an await point.
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?;
    let mut guard = s.journal.lock().unwrap();
    let outcome = rt
        .block_on(alive_act::execute_and_journal(&exec, &mut guard, &req))
        .map_err(|e| e.to_string())?;
    drop(guard);
    let (verified, status) = match &outcome {
        alive_act::ActionOutcome::Done { verified, .. } => (*verified, "done"),
        alive_act::ActionOutcome::NeedsConfirmation { .. } => (None, "awaiting_confirm"),
        alive_act::ActionOutcome::Refused { .. } => (None, "refused"),
        alive_act::ActionOutcome::Failed { .. } => (None, "failed"),
    };
    let ok = status == "done";
    s.bus.publish(EventKind::ActionCompleted {
        action_id: req.action_id.clone(),
        ok,
        summary: outcome.summary(),
    });
    set_orb(&s, if ok { OrbState::Done } else { OrbState::Error });
    Ok(OutcomeDto {
        status: status.to_string(),
        summary: outcome.summary(),
        verified,
    })
}

/// One validated agent tool call (model output, checked before use).
struct AgentCall {
    id: String,
    tool: alive_act::ToolId,
    name: String,
    args: serde_json::Value,
}

fn agent_call_from(id: &str, name: &str, args: &serde_json::Value) -> Result<AgentCall, String> {
    let tool =
        alive_act::ToolId::from_name(name).ok_or_else(|| format!("unknown tool {name:?}"))?;
    if !args.is_object() {
        return Err(format!("tool {name}: args must be an object"));
    }
    Ok(AgentCall {
        id: id.to_string(),
        tool,
        name: name.to_string(),
        args: args.clone(),
    })
}

/// Resumed histories are trusted for shape only: objects with a known role.
fn valid_agent_history(h: &[serde_json::Value]) -> bool {
    if h.len() > 64 {
        return false;
    }
    h.iter().all(|m| {
        m.get("role")
            .and_then(|r| r.as_str())
            .is_some_and(|r| matches!(r, "system" | "user" | "assistant" | "tool"))
    })
}

fn trunc_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    s.chars().take(max).collect::<String>() + "…"
}

/// Execute one agent step: journal + events, mirroring `execute`.
async fn agent_run_step(
    s: &Shared,
    exec: &alive_act::LocalExecutor,
    action_id: String,
    call: &AgentCall,
) -> AgentStepDto {
    let req = alive_act::ActionRequest {
        action_id: action_id.clone(),
        intent: "agent".to_string(),
        tool: call.tool,
        args: call.args.clone(),
        risk_tier: call.tool.risk_tier(),
    };
    s.bus.publish(EventKind::ActionStarted {
        action_id: action_id.clone(),
        intent: "agent".to_string(),
    });
    // Never hold the std MutexGuard across .await (Tauri needs Send
    // futures): execute first, then lock only for the sync append.
    let outcome = exec.execute(&req).await;
    // UI settle: acting verbs change the screen; live runs wait 250ms so
    // the next step observes the result. Dry-run touches nothing, skips it.
    if !exec.dry_run
        && req.risk_tier >= 2
        && matches!(outcome, alive_act::ActionOutcome::Done { .. })
    {
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
    }
    let (undo, verified) = match &outcome {
        alive_act::ActionOutcome::Done { undo, verified, .. } => (undo.clone(), *verified),
        _ => (None, None),
    };
    let summary = outcome.summary();
    let ok = matches!(outcome, alive_act::ActionOutcome::Done { .. });
    if let Err(e) = s
        .journal
        .lock()
        .unwrap()
        .append(alive_journal::JournalEntry {
            seq: 0,
            at_ms: 0,
            action_id: req.action_id.clone(),
            intent: req.intent.clone(),
            tool: serde_json::to_string(&req.tool)
                .unwrap_or_default()
                .trim_matches('"')
                .to_string(),
            args_redacted: alive_act::redact_args(req.tool, &req.args),
            status: outcome.status(),
            summary: summary.clone(),
            undo,
            verified,
        })
    {
        let summary = format!("journal failed: {e}");
        s.bus.publish(EventKind::ActionCompleted {
            action_id: req.action_id,
            ok: false,
            summary: summary.clone(),
        });
        return AgentStepDto {
            tool: call.name.clone(),
            summary,
        };
    }
    s.bus.publish(EventKind::ActionCompleted {
        action_id: req.action_id,
        ok,
        summary: summary.clone(),
    });
    AgentStepDto {
        tool: call.name.clone(),
        summary,
    }
}

/// End-to-end agent turn: Jev triages, DeepSeek operates tools, loop runs to
/// done / confirm-gate / step budget. No intent catalog: the instruction is
/// free text, the closed set is generic tool verbs. Stateless across calls —
/// `history` + `pending` resume a paused run verbatim.
#[tauri::command]
async fn agent_act(s: Ctx<'_>, dto: AgentRequestDto) -> Result<AgentResponseDto, String> {
    use alive_intent::tool_llm::{
        assistant_msg, sys_msg, tool_msg, user_msg, ToolLlmClient, OPERATOR_SYSTEM_PROMPT,
    };

    let cfg = s.config.lock().unwrap().clone();
    let fresh = dto.history.is_empty() && dto.pending.is_none();
    if fresh && dto.text.trim().is_empty() {
        return Err("empty instruction".to_string());
    }

    // Jev triage on fresh tasks: addressing gate + provider label.
    // No key / triage failure: proceed (explicit invocation), say so loudly.
    let mut provider = "deepseek-flash".to_string();
    if fresh && std::env::var("ALIVE_JEV_API_KEY").is_ok() {
        let tri_req = DecisionRequest {
            state: serde_json::json!({"text": dto.text, "app": dto.app, "url": dto.url}),
            questions: alive_intent::tool_dispatch_questions(&alive_act::tool_names()),
        };
        match alive_intent::JevClient::from_env(cfg.jev_endpoint.clone(), cfg.jev_model.clone()) {
            Ok(client) => match client.decide_dispatch(&tri_req).await {
                Ok(d) => {
                    provider = "jev+deepseek-flash".to_string();
                    if d.addressing_computer < 0.6 {
                        return Ok(AgentResponseDto {
                            status: "ignored".to_string(),
                            summary: "not addressing the computer".to_string(),
                            steps: vec![],
                            history: vec![],
                            pending: None,
                        });
                    }
                }
                Err(e) => s.bus.publish(EventKind::Error {
                    message: format!("jev triage failed ({e}); proceeding without gate"),
                }),
            },
            Err(e) => s.bus.publish(EventKind::Error {
                message: format!("jev client ({e}); proceeding without gate"),
            }),
        }
    }
    *s.provider.lock().unwrap() = provider;

    let llm = ToolLlmClient::from_env().map_err(|e| e.to_string())?;
    let schemas = alive_act::tool_schemas();
    let mut messages: Vec<serde_json::Value> = if fresh {
        let mut ctx = format!("Instruction: {}", dto.text.trim());
        if !dto.app.trim().is_empty() {
            ctx.push_str(&format!("\nFocused app: {}", dto.app.trim()));
        }
        if !dto.url.trim().is_empty() {
            ctx.push_str(&format!("\nOpen URL: {}", dto.url.trim()));
        }
        vec![sys_msg(OPERATOR_SYSTEM_PROMPT), user_msg(&ctx)]
    } else {
        if !valid_agent_history(&dto.history) {
            return Err("bad history: must be <=64 role-tagged messages".to_string());
        }
        dto.history.clone()
    };

    let exec = alive_act::LocalExecutor::new(cfg.dry_run_actions, cfg.app_allowlist.clone());
    set_orb(&s, OrbState::Executing);
    let mut steps: Vec<AgentStepDto> = vec![];
    let mut n = 0u64;
    let mut next_id = || {
        n += 1;
        format!("g{}s{n}", s.seq.fetch_add(1, Ordering::Relaxed),)
    };

    // Resume: run the approved pending call first. Approval is one-shot:
    // later tier-2 calls in this run pause again.
    if let Some(p) = &dto.pending {
        if !dto.approve {
            set_orb(&s, OrbState::ConfirmNeeded);
            return Ok(AgentResponseDto {
                status: "awaiting_confirm".to_string(),
                summary: format!("approve {} to continue", p.tool),
                steps,
                history: messages,
                pending: dto.pending.clone(),
            });
        }
        let call = agent_call_from(&p.id, &p.tool, &p.args)?;
        let step = agent_run_step(&s, &exec, next_id(), &call).await;
        messages.push(tool_msg(
            &call.id,
            &call.name,
            &trunc_chars(&step.summary, 2000),
        ));
        steps.push(step);
    }

    let budget = if dto.max_steps == 0 {
        6
    } else {
        dto.max_steps.clamp(1, 12)
    };
    for _ in 0..budget {
        let reply = match llm.chat(&messages, &schemas).await {
            Ok(r) => r,
            Err(e) => {
                set_orb(&s, OrbState::Error);
                return Ok(AgentResponseDto {
                    status: "error".to_string(),
                    summary: format!("llm error: {e}"),
                    steps,
                    history: messages,
                    pending: None,
                });
            }
        };
        messages.push(assistant_msg(&reply));
        if reply.tool_calls.is_empty() {
            let summary = if reply.content.trim().is_empty() {
                if steps.is_empty() {
                    "nothing to do".to_string()
                } else {
                    steps
                        .iter()
                        .map(|st| format!("{}: {}", st.tool, st.summary))
                        .collect::<Vec<_>>()
                        .join("\n")
                }
            } else {
                reply.content.clone()
            };
            set_orb(&s, OrbState::Done);
            return Ok(AgentResponseDto {
                status: "done".to_string(),
                summary,
                steps,
                history: messages,
                pending: None,
            });
        }
        for tc in &reply.tool_calls {
            let call = match agent_call_from(&tc.id, &tc.name, &tc.args) {
                Ok(c) => c,
                Err(e) => {
                    set_orb(&s, OrbState::Error);
                    return Ok(AgentResponseDto {
                        status: "error".to_string(),
                        summary: e,
                        steps,
                        history: messages,
                        pending: None,
                    });
                }
            };
            if call.tool.risk_tier() >= 2 {
                set_orb(&s, OrbState::ConfirmNeeded);
                return Ok(AgentResponseDto {
                    status: "awaiting_confirm".to_string(),
                    summary: format!(
                        "approve {} {}",
                        call.name,
                        trunc_chars(&call.args.to_string(), 300)
                    ),
                    steps,
                    history: messages,
                    pending: Some(AgentPendingDto {
                        id: call.id,
                        tool: call.name,
                        args: call.args,
                    }),
                });
            }
            let step = agent_run_step(&s, &exec, next_id(), &call).await;
            messages.push(tool_msg(
                &call.id,
                &call.name,
                &trunc_chars(&step.summary, 2000),
            ));
            steps.push(step);
        }
    }
    set_orb(&s, OrbState::Done);
    Ok(AgentResponseDto {
        status: "done".to_string(),
        summary: format!("stopped after {budget} steps (limit); send continue to resume"),
        steps,
        history: messages,
        pending: None,
    })
}

/// Show the user overlay in a mode: pipeline warms, camera arms, orb appears.
fn overlay_show(app: &tauri::AppHandle, shared: &Arc<Shared>, mode: &str) -> Result<(), String> {
    if let Err(e) = start_pipeline_once(shared) {
        shared.bus.publish(EventKind::Error {
            message: format!("overlay start: {e}"),
        });
    }
    shared.armed.store(true, Ordering::Relaxed);
    set_orb(shared, OrbState::CameraActive);
    let win = app
        .get_webview_window("overlay")
        .ok_or("overlay window missing")?;
    *shared.overlay_mode.lock().unwrap() = Some(mode.to_string());
    win.show().map_err(|e| e.to_string())?;
    // Middle-top, just below the edge: centered on the current monitor.
    if let Ok(Some(mon)) = win.current_monitor() {
        let ms = *mon.size();
        let mp = *mon.position();
        let x = mp.x + (ms.width.max(180) as i32 - 180) / 2;
        let pos = tauri::Position::Physical(tauri::PhysicalPosition { x, y: mp.y + 40 });
        if let Err(e) = win.set_position(pos) {
            eprintln!("[alive] overlay position: {e}");
        }
    }
    if let Err(e) = win.set_focus() {
        eprintln!("[alive] overlay focus: {e}");
    }
    win.emit("overlay-mode", serde_json::json!({"mode": mode}))
        .map_err(|e| e.to_string())?;
    eprintln!("[alive] overlay shown ({mode})");
    Ok(())
}

fn overlay_hide_now(app: &tauri::AppHandle, shared: &Arc<Shared>) {
    shared.armed.store(false, Ordering::Relaxed);
    set_orb(shared, OrbState::Idle);
    *shared.overlay_mode.lock().unwrap() = None;
    if let Some(win) = app.get_webview_window("overlay") {
        let _ = win.hide();
    }
}

#[tauri::command]
fn overlay_state(app: tauri::AppHandle, s: Ctx<'_>) -> serde_json::Value {
    let visible = app
        .get_webview_window("overlay")
        .and_then(|w| w.is_visible().ok())
        .unwrap_or(false);
    serde_json::json!({
        "mode": *s.overlay_mode.lock().unwrap(),
        "visible": visible,
    })
}

/// Hotkey toggle: visible overlay dismisses, hidden overlay appears in mode.
fn overlay_toggle(app: &tauri::AppHandle, shared: &Arc<Shared>, mode: &str) {
    let visible = app
        .get_webview_window("overlay")
        .and_then(|w| w.is_visible().ok())
        .unwrap_or(false);
    if visible {
        overlay_hide_now(app, shared);
    } else if let Err(e) = overlay_show(app, shared, mode) {
        eprintln!("[alive] overlay show failed: {e}");
        shared.bus.publish(EventKind::Error {
            message: format!("overlay show: {e}"),
        });
    }
}

#[tauri::command]
fn overlay_hide(app: tauri::AppHandle, s: Ctx<'_>) -> String {
    overlay_hide_now(&app, s.inner());
    "hidden".to_string()
}

/// Type a committed transcription at the user's cursor. Hides the overlay
/// first so focus falls back to the previous app, waits for the window
/// manager, then types live: the hotkey press is explicit consent, so this
/// bypasses the global dry-run switch (agent tools never do).
#[tauri::command]
fn overlay_commit_type(app: tauri::AppHandle, s: Ctx<'_>, text: String) -> Result<String, String> {
    let text = text.trim().to_string();
    if text.is_empty() {
        return Err("empty text".to_string());
    }
    if text.len() > 5000 {
        return Err("text too long".to_string());
    }
    if let Some(win) = app.get_webview_window("overlay") {
        let _ = win.hide();
    }
    s.armed.store(false, Ordering::Relaxed);
    std::thread::sleep(std::time::Duration::from_millis(180));
    let cfg = s.config.lock().unwrap().clone();
    let exec = alive_act::LocalExecutor::new(false, cfg.app_allowlist.clone());
    let req = alive_act::ActionRequest {
        action_id: format!("ov{}", s.seq.fetch_add(1, Ordering::Relaxed)),
        intent: "overlay-prompt".to_string(),
        tool: alive_act::ToolId::TypeText,
        args: serde_json::json!({"text": text}),
        risk_tier: 1,
    };
    s.bus.publish(EventKind::ActionStarted {
        action_id: req.action_id.clone(),
        intent: req.intent.clone(),
    });
    set_orb(&s, OrbState::Executing);
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?;
    let mut guard = s.journal.lock().unwrap();
    let outcome = rt
        .block_on(alive_act::execute_and_journal(&exec, &mut guard, &req))
        .map_err(|e| e.to_string())?;
    drop(guard);
    let ok = matches!(outcome, alive_act::ActionOutcome::Done { .. });
    s.bus.publish(EventKind::ActionCompleted {
        action_id: req.action_id,
        ok,
        summary: outcome.summary(),
    });
    set_orb(&s, if ok { OrbState::Done } else { OrbState::Error });
    if ok {
        Ok(outcome.summary())
    } else {
        Err(outcome.summary())
    }
}

#[tauri::command]
fn undo_last(s: Ctx<'_>) -> Result<String, String> {
    let entries = alive_journal::Journal::read_all(&s.journal_path).map_err(|e| e.to_string())?;
    let target = entries
        .iter()
        .rev()
        .find(|e| e.status == alive_journal::ActionStatus::Done && e.undo.is_some())
        .cloned();
    match target {
        Some(e) => {
            let op = e.undo.clone().unwrap();
            let msg = alive_act::apply_undo(&op).map_err(|e| e.to_string())?;
            s.journal
                .lock()
                .unwrap()
                .append(alive_journal::JournalEntry {
                    seq: 0,
                    at_ms: 0,
                    action_id: format!("{}:undo", e.action_id),
                    intent: e.intent.clone(),
                    tool: e.tool.clone(),
                    args_redacted: serde_json::Value::Null,
                    status: alive_journal::ActionStatus::Undone,
                    summary: format!("undid {}: {msg}", e.action_id),
                    undo: None,
                    verified: None,
                })
                .map_err(|e| e.to_string())?;
            Ok(format!("undid {}: {msg}", e.action_id))
        }
        None => Err("nothing undoable".to_string()),
    }
}

#[tauri::command]
fn journal_recent(s: Ctx<'_>, n: usize) -> Result<Vec<JournalEntryDto>, String> {
    let entries = alive_journal::Journal::read_all(&s.journal_path).map_err(|e| e.to_string())?;
    let n = n.clamp(1, 200);
    Ok(entries
        .iter()
        .rev()
        .take(n)
        .map(JournalEntryDto::from)
        .collect())
}

#[tauri::command]
fn tool_availability() -> AvailabilityDto {
    let r = alive_act::availability();
    AvailabilityDto {
        session_type: r.session_type,
        desktop: r.desktop,
        tools: r
            .tools
            .into_iter()
            .map(|t| ToolDto {
                tool: serde_json::to_string(&t.tool)
                    .unwrap_or_default()
                    .trim_matches('"')
                    .to_string(),
                available: t.available,
                via: t.via,
                note: t.note,
            })
            .collect(),
    }
}

#[tauri::command]
fn pack_info(s: Ctx<'_>) -> serde_json::Value {
    serde_json::json!({
        "id": s.pack.manifest.id,
        "display_name": s.pack.manifest.display_name,
        "tier": s.pack.manifest.tier,
        "entries": s.pack.lexicon.entries.len(),
        "intents": s.pack.intent_options(),
        "defs": s.pack.intents.intents,
        "templates": s.bank.lock().unwrap().glosses(),
    })
}

#[tauri::command]
fn last_trail(s: Ctx<'_>) -> Vec<(f32, f32)> {
    s.last_trail.lock().unwrap().clone()
}

#[tauri::command]
fn record_sample(s: Ctx<'_>, gloss: String, points: Vec<(f32, f32)>) -> Result<String, String> {
    if gloss.trim().is_empty() {
        return Err("empty gloss".to_string());
    }
    let mut bank = s.bank.lock().unwrap();
    if !bank.add_sample(gloss.trim(), &points) {
        return Err("need at least 8 tracked points; sign bigger".to_string());
    }
    let json = bank.to_json().map_err(|e| e.to_string())?;
    let path = state_dir().join(format!("templates-{}.json", s.pack.manifest.id));
    std::fs::write(&path, json).map_err(|e| e.to_string())?;
    Ok(format!(
        "recorded {gloss} ({} samples)",
        bank.glosses().get(gloss.trim()).copied().unwrap_or(0)
    ))
}

// ---------------------------------------------------------------------------
// Pipeline thread
// ---------------------------------------------------------------------------

struct TrailSample {
    at_ms: u64,
    hand_a: Option<(f32, f32)>,
    hand_b: Option<(f32, f32)>,
}

fn run_pipeline(shared: Arc<Shared>) {
    use alive_perception::{CaptureSource, LandmarkExtractor};
    use alive_temporal::{EndpointConfig, Endpointer, EosScorer, LowMotionEos, RingBuffer};

    set_orb(&shared, OrbState::Activating);
    let cfg = shared.config.lock().unwrap().clone();

    // Camera (480p MJPEG default + sensor max-mode lock; synthetic fallback
    // keeps the UI alive honestly).
    let mut source: Box<dyn CaptureSource> = match alive_perception::NokhwaSource::open_with(
        cfg.camera_index,
        cfg.camera_width,
        cfg.camera_height,
        cfg.camera_fps,
    ) {
        Ok(c) => {
            shared.has_camera.store(true, Ordering::Relaxed);
            Box::new(c)
        }
        Err(e) => {
            shared.bus.publish(EventKind::Error {
                message: format!("no camera ({e}); synthetic preview"),
            });
            Box::new(alive_perception::SyntheticSource::new(640, 480, None))
        }
    };

    // Vision backend: RTMW tracker if models exist, else honest null.
    // ORT intra threads: all cores minus two for capture/UI (sweep p50:
    // 2t=154 4t=124 6t=115 8t=111 10t=110 12t=105; 10t keeps the
    // foreground smooth for 4ms of invisible background cost).
    let ort_threads = std::thread::available_parallelism()
        .map(|n| n.get().saturating_sub(2).max(4))
        .unwrap_or(8);
    let mut extractor: Box<dyn LandmarkExtractor> = match find_dir("models") {
        Some(d) => match alive_perception::onnx_backend::WholeBodyTracker::load(&d, ort_threads) {
            Ok(t) => {
                *shared.vision.lock().unwrap() = "rtmw-l-wholebody".to_string();
                Box::new(t)
            }
            Err(e) => {
                *shared.vision.lock().unwrap() = "off".to_string();
                shared.bus.publish(EventKind::Error {
                    message: format!("model load failed ({e}); vision off"),
                });
                Box::new(alive_perception::NullExtractor)
            }
        },
        None => {
            *shared.vision.lock().unwrap() = "off".to_string();
            shared.bus.publish(EventKind::Error {
                message: "vision off: models/ missing".to_string(),
            });
            Box::new(alive_perception::NullExtractor)
        }
    };

    // Infer thread: owns the extractor (ort Session is Send; the camera is
    // !Send and stays in main). Throughput becomes max(capture, infer)
    // instead of capture+infer (P0: 6x measured).
    let inf_shared = shared.clone();
    let inf_handle = std::thread::Builder::new()
        .name("alive-infer".into())
        .spawn(move || {
            let mut inf_n = 0u64;
            let mut last_fseq = 0u64;
            let mut policy = GatePolicy::new(alive_perception::now_ms());
            while inf_shared.running.load(Ordering::Relaxed) {
                let armed = inf_shared.armed.load(Ordering::Relaxed);
                if !armed {
                    last_fseq = inf_shared.slot_seq.load(Ordering::Acquire);
                    policy.tick(0.0, alive_perception::now_ms(), false, false, false, false);
                    std::thread::sleep(std::time::Duration::from_millis(50));
                    continue;
                }
                let fseq = inf_shared.slot_seq.load(Ordering::Acquire);
                if fseq == last_fseq {
                    std::thread::sleep(std::time::Duration::from_millis(3));
                    continue;
                }
                last_fseq = fseq;
                // Motion gate (v2 policy): skip inference while still.
                let motion = f32::from_bits(inf_shared.motion_bits.load(Ordering::Relaxed));
                let dec = policy.tick(
                    motion,
                    alive_perception::now_ms(),
                    true,
                    inf_shared.ever_tracked.load(Ordering::Relaxed),
                    extractor.tracking(),
                    inf_n == 0,
                );
                inf_shared.gate_open.store(dec.run, Ordering::Relaxed);
                if !dec.run {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                    continue;
                }
                let frame = match inf_shared.slot_frame.lock().unwrap().clone() {
                    Some(f) => f,
                    None => continue,
                };
                inf_n += 1;
                // Detection refresh per policy (edges + backstops).
                // Never hitches active signing (tracked path skips it).
                if dec.rescan {
                    extractor.rescan(&frame);
                }
                let t1 = Instant::now();
                let lm = match extractor.infer(&frame) {
                    Ok(l) => l,
                    Err(e) => {
                        inf_shared.bus.publish(EventKind::Error {
                            message: format!("vision: {e}"),
                        });
                        continue;
                    }
                };
                let now_tracking = extractor.tracking();
                inf_shared.tracking.store(now_tracking, Ordering::Relaxed);
                if now_tracking {
                    inf_shared.ever_tracked.store(true, Ordering::Relaxed);
                }
                *inf_shared.lm_ms.lock().unwrap() =
                    (t1.elapsed().as_secs_f64() * 1000.0 * 10.0).round() / 10.0;
                *inf_shared.result.lock().unwrap() = Some(Arc::new(lm));
                inf_shared.result_seq.fetch_add(1, Ordering::Release);
            }
        })
        .expect("spawn infer thread");

    // PiP thread: preview encode off the main thread (main-thread PiP
    // cost ~9ms → ~0). Snapshots Arc frames, box-samples, draws the
    // overlay, JPEG-encodes into reused buffers, swap-publishes.
    let pip_shared = shared.clone();
    let pip_handle = std::thread::Builder::new()
        .name("alive-pip".into())
        .spawn(move || {
            let mut small = vec![0u8; 320 * 180 * 3];
            let mut jpeg = Vec::with_capacity(65536);
            let mut last_pseq = 0u64;
            let mut last_encode = Instant::now() - std::time::Duration::from_secs(1);
            while pip_shared.running.load(Ordering::Relaxed) {
                if last_encode.elapsed() < std::time::Duration::from_millis(80) {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                    continue;
                }
                last_encode = Instant::now();
                let pseq = pip_shared.slot_seq.load(Ordering::Acquire);
                if pseq == last_pseq {
                    continue;
                }
                last_pseq = pseq;
                let frame = match pip_shared.slot_frame.lock().unwrap().clone() {
                    Some(f) => f,
                    None => continue,
                };
                box_downsample(&frame.data, frame.width, frame.height, &mut small, 320, 180);
                let ov = pip_shared.overlay.lock().unwrap().clone();
                draw_overlay(&mut small, 320, 180, frame.width, frame.height, &ov);
                jpeg.clear();
                let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, 70);
                if enc
                    .encode(&small, 320, 180, image::ExtendedColorType::Rgb8)
                    .is_err()
                {
                    continue;
                }
                if jpeg.len() > 65536 {
                    // Pathological content: realign capacity, drop one
                    // preview frame (invisible at 12Hz), never corrupt.
                    jpeg = Vec::with_capacity(65536);
                    continue;
                }
                // Swap-publish: both sides keep their capacities, zero
                // alloc steady-state. URI handler untouched.
                std::mem::swap(&mut *pip_shared.frame_jpeg.lock().unwrap(), &mut jpeg);
                pip_shared.seq.fetch_add(1, Ordering::Relaxed);
            }
        })
        .expect("spawn pip thread");

    let mut endpointer = Endpointer::new(EndpointConfig {
        tentative_pause_ms: cfg.tentative_pause_ms,
        commit_pause_ms: cfg.commit_pause_ms,
        min_segment_ms: cfg.min_segment_ms,
        boundary_p: cfg.boundary_p,
        ..Default::default()
    });
    let eos = LowMotionEos { floor: 0.02 };
    let mut recent_motion: RingBuffer<f32> = RingBuffer::new(15);
    let mut prev_flat: Vec<f32> = vec![];
    let mut trail: VecDeque<TrailSample> = VecDeque::with_capacity(600);
    let mut fps_ema = 0.0f64;
    let mut prev_luma: Vec<u8> = vec![];
    let mut last_present = false;

    set_orb(&shared, OrbState::CameraActive);
    shared.bus.publish(EventKind::Info {
        message: "session live".to_string(),
    });

    // Main loop: capture + publish + consume landmark results + temporal.
    let mut last_rseq = 0u64;
    let mut prev_tick = Instant::now();
    while shared.running.load(Ordering::Relaxed) {
        // Capture (camera is !Send: it lives here and never moves).
        let t0 = Instant::now();
        let frame = match source.next_frame() {
            Ok(Some(f)) => f,
            Ok(None) => {
                eprintln!("[alive] capture: source exhausted");
                break;
            }
            Err(e) => {
                eprintln!("[alive] capture error: {e}");
                shared.bus.publish(EventKind::Error {
                    message: format!("capture: {e}"),
                });
                break;
            }
        };
        let cap_ms = t0.elapsed().as_secs_f64() * 1000.0;
        // Motion score every frame (~0.2ms): drives the infer gate + UI.
        let mscore = motion_score(&frame.data, frame.width, frame.height, &mut prev_luma);
        shared
            .motion_bits
            .store(mscore.to_bits(), Ordering::Relaxed);
        // PiP encode lives on the alive-pip thread now (main cost ~0).
        *shared.slot_frame.lock().unwrap() = Some(Arc::new(frame));
        shared.slot_seq.fetch_add(1, Ordering::Release);
        // Loop-rate meter, every iteration armed or not (the 0.0fps fix).
        let now = Instant::now();
        let dt = now.duration_since(prev_tick).as_secs_f64();
        prev_tick = now;
        if dt > 0.0 {
            let inst = 1.0 / dt;
            fps_ema = if fps_ema == 0.0 {
                inst
            } else {
                fps_ema * 0.9 + inst * 0.1
            };
        }
        *shared.fps.lock().unwrap() = (fps_ema * 10.0).round() / 10.0;
        *shared.cap_fps.lock().unwrap() = (fps_ema * 10.0).round() / 10.0;
        *shared.cap_ms.lock().unwrap() = (cap_ms * 10.0).round() / 10.0;
        // Consume the latest landmarks (infer thread produces them).
        if !shared.armed.load(Ordering::Relaxed) {
            continue;
        }
        let rseq = shared.result_seq.load(Ordering::Acquire);
        if rseq != last_rseq {
            last_rseq = rseq;
            let lm = match shared.result.lock().unwrap().clone() {
                Some(l) => l,
                None => continue,
            };
            last_present = lm.present;
            note_ticks(&shared, 1);

            if lm.present {
                set_orb(&shared, OrbState::SignerDetected);
            }

            // Trail (both hands, gated at 0.4 / min 6 keys — matches matcher).
            if lm.present && lm.points.len() == 133 {
                let a = hand_centroid(&lm.points, &lm.scores, HAND_A, 0.4, 6);
                let b = hand_centroid(&lm.points, &lm.scores, HAND_B, 0.4, 6);
                trail.push_back(TrailSample {
                    at_ms: lm.at_ms,
                    hand_a: a,
                    hand_b: b,
                });
                while trail.len() > 600 {
                    trail.pop_front();
                }
            }

            // Temporal.
            let flat = alive_perception::flatten_landmarks(&lm);
            let motion = alive_temporal::motion_energy(&prev_flat, &flat);
            prev_flat = flat;
            recent_motion.push(motion);
            let window: Vec<f32> = recent_motion.iter_oldest_first().copied().collect();
            let eos_p = eos.score(&window);

            if let Some(ev) = endpointer.push(lm.at_ms, motion, eos_p, lm.present) {
                handle_endpoint_event(&shared, &mut trail, ev);
            }
        } else if last_present && !shared.gate_open.load(Ordering::Relaxed) && mscore < MOTION_SLEEP
        {
            // Gate-held stillness: no pose will come, so feed explicit
            // stillness at capture cadence — pauses accumulate and commits
            // fire on time instead of starving. Triple-guarded: we were
            // seeing a signer, infer is provably skipping, and this fresh
            // frame is still (never synthesize during real motion gaps).
            recent_motion.push(0.0);
            let window: Vec<f32> = recent_motion.iter_oldest_first().copied().collect();
            let eos_p = eos.score(&window);
            if let Some(ev) = endpointer.push(alive_perception::now_ms(), 0.0, eos_p, true) {
                handle_endpoint_event(&shared, &mut trail, ev);
            }
        }
    }

    set_orb(&shared, OrbState::Idle);
    let _ = inf_handle.join();
    let _ = pip_handle.join();
}

/// Area-weighted box downsample RGB8 → RGB8. Display path (PiP):
/// ~0.4ms for 640x480 → 320x180, zero allocs (caller buffers).
fn box_downsample(src: &[u8], sw: u32, sh: u32, dst: &mut [u8], dw: u32, dh: u32) {
    debug_assert_eq!(src.len(), sw as usize * sh as usize * 3);
    debug_assert_eq!(dst.len(), dw as usize * dh as usize * 3);
    if sw == 0 || sh == 0 || dw == 0 || dh == 0 {
        return;
    }
    let x_ratio = sw as f32 / dw as f32;
    let y_ratio = sh as f32 / dh as f32;
    for dy in 0..dh as usize {
        let y0 = (dy as f32 * y_ratio) as usize;
        let y1 = (((dy + 1) as f32 * y_ratio).ceil() as usize).min(sh as usize);
        for dx in 0..dw as usize {
            let x0 = (dx as f32 * x_ratio) as usize;
            let x1 = (((dx + 1) as f32 * x_ratio).ceil() as usize).min(sw as usize);
            let mut acc = [0u32; 3];
            let mut n = 0u32;
            for sy in y0..y1 {
                for sx in x0..x1 {
                    let o = (sy * sw as usize + sx) * 3;
                    acc[0] += src[o] as u32;
                    acc[1] += src[o + 1] as u32;
                    acc[2] += src[o + 2] as u32;
                    n += 1;
                }
            }
            let o = (dy * dw as usize + dx) * 3;
            let n = n.max(1);
            dst[o] = (acc[0] / n) as u8;
            dst[o + 1] = (acc[1] / n) as u8;
            dst[o + 2] = (acc[2] / n) as u8;
        }
    }
}

/// Draw the overlay snapshot onto a downsampled RGB8 buffer.
/// Amber skeleton, fading warm ribbons, green lead dots.
fn draw_overlay(dst: &mut [u8], dw: u32, dh: u32, fw: u32, fh: u32, ov: &OverlayFrame) {
    if dw == 0 || dh == 0 || fw == 0 || fh == 0 {
        return;
    }
    let sx = dw as f32 / fw as f32;
    let sy = dh as f32 / fh as f32;
    let map = |x: i16, y: i16| ((x as f32 * sx) as i32, (y as f32 * sy) as i32);
    let mut dot = |x: i32, y: i32, r: i32, col: [u8; 3]| {
        for oy in -r..=r {
            for ox in -r..=r {
                if ox * ox + oy * oy > r * r {
                    continue;
                }
                let (xx, yy) = (x + ox, y + oy);
                if xx < 0 || yy < 0 || xx >= dw as i32 || yy >= dh as i32 {
                    continue;
                }
                let o = ((yy as usize) * (dw as usize) + (xx as usize)) * 3;
                dst[o] = col[0];
                dst[o + 1] = col[1];
                dst[o + 2] = col[2];
            }
        }
    };
    for &((x1, y1), (x2, y2)) in &ov.segs {
        let (ax, ay) = map(x1, y1);
        let (bx, by) = map(x2, y2);
        let steps = (ax - bx).abs().max((ay - by).abs()).max(1);
        for i in 0..=steps {
            let t = i as f32 / steps as f32;
            dot(
                ax + ((bx - ax) as f32 * t) as i32,
                ay + ((by - ay) as f32 * t) as i32,
                1,
                [255, 176, 102],
            );
        }
    }
    for ribbon in &ov.ribbons {
        let len = ribbon.len().max(1) as f32;
        for (i, &p) in ribbon.iter().enumerate() {
            let fade = (i as f32 / len * 200.0 + 55.0) as u8;
            let (x, y) = map(p.0, p.1);
            dot(
                x,
                y,
                1,
                [fade, (fade as f32 * 0.85) as u8, (fade as f32 * 0.7) as u8],
            );
        }
    }
    for lead in ov.leads.iter().flatten() {
        let (x, y) = map(lead.0, lead.1);
        dot(x, y, 2, [127, 208, 138]);
    }
}

/// Record understanding ticks (real poses + upsampled virtuals) and
/// maintain a 1s-windowed Hz readout for the UI.
fn note_ticks(shared: &Shared, n: u64) {
    shared.ticks_total.fetch_add(n, Ordering::Relaxed);
    let mut win = shared.ticks_window.lock().unwrap();
    win.1 += n;
    let el = win.0.elapsed().as_secs_f64();
    if el >= 1.0 {
        *shared.upsample_hz.lock().unwrap() = (win.1 as f64 / el * 10.0).round() / 10.0;
        *win = (Instant::now(), 0);
    }
}

fn handle_endpoint_event(
    shared: &Shared,
    trail: &mut VecDeque<TrailSample>,
    ev: alive_temporal::EndpointEvent,
) {
    use alive_temporal::EndpointEvent::*;
    match ev {
        Tentative { .. } => {
            set_orb(shared, OrbState::Interpreting);
            shared.bus.publish(EventKind::PartialGloss {
                text: String::new(),
                speculative: true,
            });
        }
        Commit {
            start_ms, end_ms, ..
        } => {
            handle_commit(shared, trail, start_ms, end_ms);
            while trail.front().is_some_and(|s| s.at_ms < end_ms) {
                trail.pop_front();
            }
        }
        Cancelled { .. } => {
            set_orb(shared, OrbState::CameraActive);
        }
    }
}

fn handle_commit(shared: &Shared, trail: &VecDeque<TrailSample>, start_ms: u64, end_ms: u64) {
    set_orb(shared, OrbState::IntentForming);
    // Dominant-hand path over the segment.
    let mut path_a: Vec<(f32, f32)> = vec![];
    let mut path_b: Vec<(f32, f32)> = vec![];
    for s in trail
        .iter()
        .filter(|s| s.at_ms >= start_ms && s.at_ms <= end_ms)
    {
        if let Some(p) = s.hand_a {
            path_a.push(p);
        }
        if let Some(p) = s.hand_b {
            path_b.push(p);
        }
    }
    let path = if path_a.len() >= path_b.len() {
        path_a
    } else {
        path_b
    };
    *shared.last_trail.lock().unwrap() = path.clone();
    let matched = shared.bank.lock().unwrap().recognize(&path);

    let cfg = shared.config.lock().unwrap().clone();
    match matched {
        Some((gloss, score)) => {
            let nbest = NBest::new(
                vec![
                    alive_sign::GlossHypothesis {
                        gloss: gloss.clone(),
                        confidence: score as f64,
                    },
                    alive_sign::GlossHypothesis {
                        gloss: "UNKNOWN".to_string(),
                        confidence: (1.0 - score as f64).max(0.01),
                    },
                ],
                2,
            );
            let dto = DecideRequestDto {
                nbest: nbest
                    .hyps
                    .iter()
                    .map(|h| HypDto {
                        gloss: h.gloss.clone(),
                        p: h.confidence,
                    })
                    .collect(),
                eos_p: 1.0,
                hold_ms: end_ms.saturating_sub(start_ms),
                motion_energy: 0.0,
                fingerspell_p: 0.0,
                app: String::new(),
                url: String::new(),
                last_action: String::new(),
            };
            // Synchronous decision inside the pipeline thread (once per
            // committed segment, never per frame, so perception never blocks).
            let decision = if std::env::var("ALIVE_JEV_API_KEY").is_ok() {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap();
                rt.block_on(async {
                    let c = JevClient::from_env(cfg.jev_endpoint.clone(), cfg.jev_model.clone())
                        .map_err(|e| e.to_string())?;
                    let req = decide_request(&shared.pack, &dto);
                    c.decide(&req).await.map_err(|e| e.to_string())
                })
                .map(|d| {
                    *shared.provider.lock().unwrap() = "jev".to_string();
                    d
                })
                .unwrap_or_else(|e| {
                    *shared.provider.lock().unwrap() = format!("mock(jev-fallback:{e})");
                    mock_decide(&shared.pack, &dto)
                })
            } else {
                *shared.provider.lock().unwrap() = "mock".to_string();
                mock_decide(&shared.pack, &dto)
            };
            let r =
                alive_intent::route_decision(&decision, cfg.act_threshold, cfg.confirm_threshold);
            shared.bus.publish(EventKind::Committed {
                text: gloss.clone(),
                segment_ms: end_ms.saturating_sub(start_ms),
            });
            shared.bus.publish(EventKind::IntentDecided {
                intent: r.intent.clone(),
                confidence: r.confidence,
                route: format!("{:?}", r.route).to_lowercase(),
            });
            set_orb(shared, OrbState::Ready);
        }
        None => {
            shared.bus.publish(EventKind::Committed {
                text: String::new(),
                segment_ms: end_ms.saturating_sub(start_ms),
            });
            shared.bus.publish(EventKind::Error {
                message: format!(
                    "motion not recognized ({} tracked points) — record it under Tools > Commands",
                    path.len()
                ),
            });
            set_orb(shared, OrbState::CameraActive);
        }
    }
}

// ---------------------------------------------------------------------------
// App entry
// ---------------------------------------------------------------------------

/// Minimal stderr logger so Tauri/wry/WebKit errors are visible.
struct StderrLogger;
impl log::Log for StderrLogger {
    fn enabled(&self, m: &log::Metadata) -> bool {
        m.level() <= log::Level::Warn
    }
    fn log(&self, r: &log::Record) {
        if self.enabled(r.metadata()) {
            eprintln!("[{}] {}", r.target(), r.args());
        }
    }
    fn flush(&self) {}
}
static STDERR_LOGGER: StderrLogger = StderrLogger;

fn main() {
    let _ = log::set_logger(&STDERR_LOGGER);
    log::set_max_level(log::LevelFilter::Warn);
    let cfg = Config::default();
    let packs_dir = find_dir("packs").expect("packs/ directory not found");
    let pack = LanguagePack::load(&packs_dir.join(&cfg.language_pack))
        .expect("language pack failed to load");
    let sdir = state_dir();
    let journal = alive_journal::Journal::open(&sdir.join("journal.jsonl")).expect("journal open");
    let bank_path = sdir.join(format!("templates-{}.json", pack.manifest.id));
    let bank = std::fs::read_to_string(&bank_path)
        .ok()
        .and_then(|s| TemplateBank::from_json(&s).ok())
        .unwrap_or_else(|| TemplateBank::new(0.45));

    let shared = Arc::new(Shared {
        config: Mutex::new(cfg),
        pack,
        bank: Mutex::new(bank),
        bus: EventBus::new(256),
        journal: Mutex::new(journal),
        journal_path: sdir.join("journal.jsonl"),
        orb: Mutex::new(OrbState::Idle),
        provider: Mutex::new("mock".to_string()),
        vision: Mutex::new("pending".to_string()),
        fps: Mutex::new(0.0),
        cap_ms: Mutex::new(0.0),
        lm_ms: Mutex::new(0.0),
        frame_jpeg: Mutex::new(vec![]),
        last_trail: Mutex::new(vec![]),
        slot_seq: AtomicU64::new(0),
        slot_frame: Mutex::new(None),
        result_seq: AtomicU64::new(0),
        result: Mutex::new(None),
        cap_fps: Mutex::new(0.0),
        seq: AtomicU64::new(0),
        running: AtomicBool::new(false),
        armed: AtomicBool::new(false),
        has_camera: AtomicBool::new(false),
        motion_bits: AtomicU32::new(0),
        tracking: AtomicBool::new(false),
        ever_tracked: AtomicBool::new(false),
        gate_open: AtomicBool::new(true),
        overlay: Mutex::new(Arc::new(OverlayFrame::default())),
        ticks_total: AtomicU64::new(0),
        ticks_window: Mutex::new((Instant::now(), 0)),
        upsample_hz: Mutex::new(0.0),
        overlay_mode: Mutex::new(None),
    });

    let frame_shared = shared.clone();
    let bus = shared.bus.clone();

    // Global hotkeys (Linux/X11 via XGrabKey): ctrl+shift+p prompts at
    // the cursor, ctrl+shift+c runs computer-use. Same key toggles dismiss.
    use tauri_plugin_global_shortcut::{
        Code, GlobalShortcutExt, Modifiers, Shortcut, ShortcutState,
    };
    let prompt_sc = Shortcut::new(Some(Modifiers::CONTROL | Modifiers::SHIFT), Code::KeyP);
    let computer_sc = Shortcut::new(Some(Modifiers::CONTROL | Modifiers::SHIFT), Code::KeyC);
    let (prompt_id, computer_id) = (prompt_sc.id(), computer_sc.id());

    tauri::Builder::default()
        .manage(shared)
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(move |app, _shortcut, event| {
                    if event.state != ShortcutState::Pressed {
                        return;
                    }
                    let shared: State<Arc<Shared>> = app.state();
                    if event.id == prompt_id {
                        overlay_toggle(app, &shared, "prompt");
                    } else if event.id == computer_id {
                        overlay_toggle(app, &shared, "computer");
                    }
                })
                .build(),
        )
        .register_uri_scheme_protocol("aliveframe", move |_ctx, req| {
            let path = req.uri().path().to_string();
            if path != "/frame.jpg" {
                return tauri::http::Response::builder()
                    .status(404)
                    .body(Vec::new())
                    .unwrap();
            }
            let jpeg = frame_shared.frame_jpeg.lock().unwrap().clone();
            if jpeg.is_empty() {
                return tauri::http::Response::builder()
                    .status(404)
                    .body(Vec::new())
                    .unwrap();
            }
            tauri::http::Response::builder()
                .status(200)
                .header("Content-Type", "image/jpeg")
                .header("Cache-Control", "no-store")
                .body(jpeg)
                .unwrap()
        })
        .setup(move |app| {
            let handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                let mut rx = bus.subscribe();
                while let Ok(ev) = rx.recv().await {
                    let _ = handle.emit("alive-event", &ev);
                }
            });
            if let Err(e) = app.global_shortcut().register(prompt_sc) {
                eprintln!("[alive] hotkey ctrl+shift+p refused: {e}");
            }
            if let Err(e) = app.global_shortcut().register(computer_sc) {
                eprintln!("[alive] hotkey ctrl+shift+c refused: {e}");
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_status,
            set_mode,
            set_armed,
            set_live_actions,
            start_session,
            stop_session,
            decide,
            execute,
            agent_act,
            overlay_hide,
            overlay_commit_type,
            overlay_state,
            undo_last,
            journal_recent,
            tool_availability,
            pack_info,
            record_sample,
            last_trail
        ])
        .run(tauri::generate_context!())
        .expect("tauri run");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agent_validation_helpers() {
        // Known tools pass, unknown names and non-object args fail.
        let c =
            agent_call_from("c1", "open_url", &serde_json::json!({"url": "https://x"})).unwrap();
        assert_eq!(c.name, "open_url");
        assert!(agent_call_from("c1", "launch_missiles", &serde_json::json!({})).is_err());
        assert!(agent_call_from("c1", "open_url", &serde_json::json!("nope")).is_err());
        // History shape gate.
        assert!(valid_agent_history(&[]));
        assert!(valid_agent_history(&[
            serde_json::json!({"role": "system", "content": "s"}),
            serde_json::json!({"role": "tool", "content": "r"}),
        ]));
        assert!(!valid_agent_history(&[
            serde_json::json!({"role": "hacker"})
        ]));
        assert!(!valid_agent_history(&[serde_json::json!({"nope": 1})]));
        assert!(!valid_agent_history(&vec![
            serde_json::json!({"role": "user"});
            65
        ]));
        // Truncation is char-based, never panics on unicode.
        assert_eq!(trunc_chars("abcdef", 4), "abcd…");
        assert_eq!(trunc_chars("héllo", 99), "héllo");
        assert_eq!(trunc_chars("请发送", 2), "请发…");
    }

    fn gray_frame(w: u32, h: u32, v: u8) -> Vec<u8> {
        vec![v; (w * h) as usize * 3]
    }

    #[test]
    fn gate_still_wake_and_global_shift() {
        let (w, h) = (64u32, 48u32);
        let mut prev = vec![];
        // First frame reseeds and reports still.
        assert_eq!(motion_score(&gray_frame(w, h, 20), w, h, &mut prev), 0.0);
        // Identical frame: still.
        assert_eq!(motion_score(&gray_frame(w, h, 20), w, h, &mut prev), 0.0);
        // Small noise under the floor: still.
        assert_eq!(motion_score(&gray_frame(w, h, 25), w, h, &mut prev), 0.0);
        // A moved block (top-left 16x12 bright): wakes.
        let mut moved = gray_frame(w, h, 25);
        for y in 0..12 {
            for x in 0..16 {
                moved[(y * w as usize + x) * 3 + 1] = 200;
            }
        }
        let s = motion_score(&moved, w, h, &mut prev);
        assert!(s >= MOTION_WAKE, "block must wake, got {s}");
        assert!(s < 0.45, "block is not global, got {s}");
        // Whole-frame flash: rejected as global shift.
        assert_eq!(motion_score(&gray_frame(w, h, 250), w, h, &mut prev), 0.0);
    }

    #[test]
    fn gate_exposure_coherence() {
        let (w, h) = (64u32, 48u32);
        let n = (w * h) as usize;
        // Coherent +40 lift on 30% of pixels (gain snap): suppressed.
        let mut prev = vec![];
        motion_score(&gray_frame(w, h, 100), w, h, &mut prev);
        let mut snap = gray_frame(w, h, 100);
        for i in 0..(n * 3 / 10) {
            snap[i * 3 + 1] = 140;
        }
        assert_eq!(motion_score(&snap, w, h, &mut prev), 0.0);
        // Same 30% split half-brighter/half-darker (real change): reported.
        // (Split by halves, not parity: stride-4 sampling only sees even
        // pixels, so parity patterns alias to coherent.)
        let mut prev = vec![];
        motion_score(&gray_frame(w, h, 100), w, h, &mut prev);
        let mut mixed = gray_frame(w, h, 100);
        for i in 0..(n * 3 / 10) {
            mixed[i * 3 + 1] = if i < n * 3 / 20 { 140 } else { 60 };
        }
        let s = motion_score(&mixed, w, h, &mut prev);
        assert!(s > 0.25 && s < 0.45, "mixed change must report, got {s}");
    }

    #[test]
    fn policy_disarmed_never_runs() {
        let mut p = GatePolicy::new(1_000_000);
        for t in 0..10 {
            let d = p.tick(0.5, 1_000_000 + t * 33, false, false, false, false);
            assert_eq!(
                d,
                GateDecision {
                    run: false,
                    rescan: false
                }
            );
        }
    }

    #[test]
    fn policy_arm_edge_scans_immediately() {
        let mut p = GatePolicy::new(1_000_000);
        p.tick(0.0, 1_000_000, false, false, false, false);
        let d = p.tick(0.0, 1_000_033, true, false, false, true);
        assert!(d.run && d.rescan, "arm edge must scan, got {d:?}");
        // Next still tick: sweep/backstop not due → quiet.
        let d = p.tick(0.0, 1_000_066, true, false, false, false);
        assert_eq!(
            d,
            GateDecision {
                run: false,
                rescan: false
            }
        );
    }

    #[test]
    fn policy_wake_latch_hysteresis() {
        let mut p = GatePolicy::new(1_000_000);
        // Wake at threshold; untracked so each run also rescans on the edge.
        let d = p.tick(0.003, 1_000_000, true, true, true, false);
        assert!(d.run, "wake threshold must run");
        // Mid-band holds the latch open (no flutter).
        let d = p.tick(0.002, 1_000_033, true, true, true, false);
        assert!(d.run, "hysteresis band must hold");
        // Below sleep closes.
        let d = p.tick(0.0005, 1_000_066, true, true, true, false);
        assert!(!d.run, "below sleep must close");
    }

    #[test]
    fn policy_dead_room_stays_quiet() {
        // THE anti-burn proof: 30s of stillness, never tracked →
        // a handful of sweep runs, exactly one rescan (the arm edge).
        let mut p = GatePolicy::new(1_000_000);
        let mut runs = 0u32;
        let mut rescans = 0u32;
        for i in 0..900 {
            let t = 1_000_000 + i * 33;
            let d = p.tick(0.0001, t, true, false, false, i == 0);
            runs += d.run as u32;
            rescans += d.rescan as u32;
        }
        assert!(runs <= 8, "dead room must barely run, got {runs}");
        assert_eq!(rescans, 1, "dead room rescans once (arm edge)");
    }

    #[test]
    fn policy_slow_sign_wakes() {
        // Sub-threshold flicker accumulates into a wake within ~200ms.
        let mut p = GatePolicy::new(1_000_000);
        p.tick(0.0, 1_000_000, true, true, true, false); // heartbeat run
        let mut runs = 0u32;
        for i in 1..30 {
            let d = p.tick(0.0012, 1_000_000 + i * 33, true, true, true, false);
            runs += d.run as u32;
        }
        assert!(runs >= 1, "slow sign must wake, got {runs} runs");
        // Control: true stillness stays quiet (no heartbeat inside 660ms).
        let mut p = GatePolicy::new(2_000_000);
        p.tick(0.0, 2_000_000, true, true, true, false);
        let mut runs = 0u32;
        for i in 1..20 {
            let d = p.tick(0.0, 2_000_000 + i * 33, true, true, true, false);
            runs += d.run as u32;
        }
        assert_eq!(runs, 0, "stillness must not wake");
    }

    #[test]
    fn policy_recent_loss_rescans_fast() {
        let mut p = GatePolicy::new(1_000_000);
        // Tracked at t0 (marks last_seen), then lost.
        p.tick(0.0, 1_000_000, true, true, true, true);
        let mut rescans = 0u32;
        for i in 1..200 {
            let d = p.tick(0.0, 1_000_000 + i * 33, true, true, false, false);
            rescans += d.rescan as u32;
        }
        assert!(rescans >= 1, "recent loss must rescan within ~5s");
    }

    #[test]
    fn pip_box_downsample_exact() {
        // 4x4 solid red → 2x2 solid red.
        let src = vec![200u8, 10, 30].repeat(16);
        assert_eq!(src.len(), 4 * 4 * 3);
        let mut dst = vec![0u8; 2 * 2 * 3];
        box_downsample(&src, 4, 4, &mut dst, 2, 2);
        assert_eq!(dst, vec![200u8, 10, 30].repeat(4));
        // 2x2 distinct quadrants → 1x1 mean.
        let src = vec![0u8, 0, 0, 100, 0, 0, 0, 100, 0, 0, 0, 100];
        let mut dst = vec![0u8; 3];
        box_downsample(&src, 2, 2, &mut dst, 1, 1);
        assert_eq!(dst, vec![25u8, 25, 25]);
        // Degenerate dims never panic.
        let mut dst = vec![7u8; 3];
        box_downsample(&[], 0, 0, &mut dst, 1, 1);
        assert_eq!(dst, vec![7u8; 3]);
    }

    #[test]
    fn pip_overlay_draws_and_clips() {
        let mut buf = vec![0u8; 32 * 18 * 3];
        // Empty overlay: untouched.
        draw_overlay(&mut buf, 32, 18, 640, 480, &OverlayFrame::default());
        assert!(buf.iter().all(|&b| b == 0));
        // Lead dot lands green; wild coords clip silently.
        let ov = OverlayFrame {
            segs: vec![],
            ribbons: [vec![], vec![]],
            leads: [Some((320, 240)), Some((9999, -9999))],
        };
        draw_overlay(&mut buf, 32, 18, 640, 480, &ov);
        // Center maps to (16, 9); radius-2 disc must be greenish.
        let o = (9 * 32 + 16) * 3;
        assert_eq!([buf[o], buf[o + 1], buf[o + 2]], [127, 208, 138]);
        // A segment draws amber somewhere along its length.
        let mut buf = vec![0u8; 32 * 18 * 3];
        let ov = OverlayFrame {
            segs: vec![((0, 0), (640, 480))],
            ribbons: [vec![], vec![]],
            leads: [None, None],
        };
        draw_overlay(&mut buf, 32, 18, 640, 480, &ov);
        let amber = buf
            .chunks_exact(3)
            .filter(|p| p[0] == 255 && p[1] == 176 && p[2] == 102)
            .count();
        assert!(amber > 10, "diagonal must paint, got {amber}");
    }

    #[test]
    fn pip_jpeg_encode_path() {
        let small = vec![128u8; 320 * 180 * 3];
        let mut jpeg = Vec::with_capacity(65536);
        let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, 70);
        enc.encode(&small, 320, 180, image::ExtendedColorType::Rgb8)
            .expect("encode");
        assert!(jpeg.len() > 100, "jpeg must have bytes");
        assert_eq!(&jpeg[0..2], &[0xFF, 0xD8], "jpeg magic");
        assert!(jpeg.len() < 65536, "fits the swap cap");
    }

    #[test]
    fn gate_reseeds_on_resolution_change() {
        let mut prev = vec![];
        assert_eq!(
            motion_score(&gray_frame(64, 48, 10), 64, 48, &mut prev),
            0.0
        );
        assert_eq!(
            prev.len(),
            (64usize * 48).div_ceil(4),
            "sample count follows geometry"
        );
        // New geometry reseeds instead of panicking or misreading.
        assert_eq!(
            motion_score(&gray_frame(32, 24, 10), 32, 24, &mut prev),
            0.0
        );
        assert_eq!(prev.len(), (32usize * 24).div_ceil(4));
    }
}
