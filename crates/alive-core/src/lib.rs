//! alive-core: shared pipeline types, app state machine, event bus, config.
//!
//! Every crate speaks these types. No ML, no OS calls, no network here.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

/// Wall-clock milliseconds. Single time source for the whole pipeline.
pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum AppMode {
    #[default]
    Dictation,
    Action,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum OrbState {
    #[default]
    Idle,
    Activating,
    CameraActive,
    SignerDetected,
    Interpreting,
    IntentForming,
    Ready,
    Executing,
    Done,
    ConfirmNeeded,
    Error,
    Recovery,
}

/// Every observable pipeline transition. The orb renders truthfully from these.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EventKind {
    OrbStateChanged {
        from: OrbState,
        to: OrbState,
    },
    PartialGloss {
        text: String,
        speculative: bool,
    },
    Committed {
        text: String,
        segment_ms: u64,
    },
    IntentDecided {
        intent: String,
        confidence: f64,
        route: String,
    },
    ActionStarted {
        action_id: String,
        intent: String,
    },
    ActionCompleted {
        action_id: String,
        ok: bool,
        summary: String,
    },
    ConfirmRequested {
        action_id: String,
        preview: String,
    },
    Error {
        message: String,
    },
    Info {
        message: String,
    },
    PerfSample {
        capture_ms: f64,
        landmark_ms: f64,
        fps: f64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineEvent {
    pub seq: u64,
    pub at_ms: u64,
    pub kind: EventKind,
}

/// Fan-out bus: Rust pipeline publishes, Tauri layer forwards to the WebView.
#[derive(Debug, Clone)]
pub struct EventBus {
    tx: broadcast::Sender<PipelineEvent>,
    seq: Arc<AtomicU64>,
}

impl EventBus {
    pub fn new(capacity: usize) -> Self {
        let (tx, _) = broadcast::channel(capacity.max(16));
        Self {
            tx,
            seq: Arc::new(AtomicU64::new(1)),
        }
    }

    pub fn publish(&self, kind: EventKind) {
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        let _ = self.tx.send(PipelineEvent {
            seq,
            at_ms: now_ms(),
            kind,
        });
    }

    pub fn subscribe(&self) -> broadcast::Receiver<PipelineEvent> {
        self.tx.subscribe()
    }
}

/// Whole-app configuration. Defaults encode the PRD latency contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub mode: AppMode,
    pub camera_index: u32,
    pub camera_width: u32,
    pub camera_height: u32,
    pub camera_fps: u32,
    pub language_pack: String,
    pub hold_to_sign: bool,
    pub jev_endpoint: String,
    pub jev_model: String,
    /// act >= this confidence
    pub act_threshold: f64,
    /// confirm in [confirm_threshold, act_threshold)
    pub confirm_threshold: f64,
    pub tentative_pause_ms: u64,
    pub commit_pause_ms: u64,
    pub min_segment_ms: u64,
    pub boundary_p: f32,
    /// When true, actions are planned + journaled but never executed.
    pub dry_run_actions: bool,
    /// Desktop IDs / command names allowed for open_app. Empty = deny all.
    pub app_allowlist: Vec<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            mode: AppMode::Dictation,
            camera_index: 0,
            camera_width: 640,
            camera_height: 480,
            camera_fps: 30,
            language_pack: "csl".to_string(),
            hold_to_sign: true,
            jev_endpoint: "https://api.typesafe.ai/v1/systemone".to_string(),
            jev_model: "jev-1.13.0".to_string(),
            act_threshold: 0.85,
            confirm_threshold: 0.6,
            tentative_pause_ms: 350,
            commit_pause_ms: 700,
            min_segment_ms: 250,
            boundary_p: 0.85,
            dry_run_actions: true,
            app_allowlist: vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_prd_contract() {
        let c = Config::default();
        assert!((c.act_threshold - 0.85).abs() < f64::EPSILON);
        assert!((c.confirm_threshold - 0.6).abs() < f64::EPSILON);
        assert_eq!(c.tentative_pause_ms, 350);
        assert_eq!(c.commit_pause_ms, 700);
        assert_eq!(c.min_segment_ms, 250);
        assert!(c.dry_run_actions, "actions must default to dry-run");
    }

    #[tokio::test]
    async fn bus_delivers_in_order_with_seq() {
        let bus = EventBus::new(16);
        let mut rx = bus.subscribe();
        bus.publish(EventKind::PartialGloss {
            text: "HELLO".into(),
            speculative: true,
        });
        bus.publish(EventKind::Committed {
            text: "HELLO".into(),
            segment_ms: 900,
        });
        let a = rx.recv().await.unwrap();
        let b = rx.recv().await.unwrap();
        assert_eq!(a.seq + 1, b.seq);
        assert!(a.at_ms > 0 && b.at_ms >= a.at_ms);
    }
}
