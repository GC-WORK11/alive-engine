//! alive-ipc: DTOs shared between the Rust core and the Tauri frontend.
//! The frontend TypeScript mirrors these shapes 1:1 (see src/types.ts).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusDto {
    pub orb_state: String,
    pub mode: String,
    pub provider: String,
    pub vision: String,
    pub fps: f64,
    pub capture_ms: f64,
    pub landmark_ms: f64,
    pub camera: bool,
    pub pack: String,
    pub running: bool,
    pub armed: bool,
    pub motion: f32,
    pub tracking: bool,
    pub upsample_hz: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypDto {
    pub gloss: String,
    pub p: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecideRequestDto {
    #[serde(default)]
    pub nbest: Vec<HypDto>,
    #[serde(default)]
    pub eos_p: f32,
    #[serde(default)]
    pub hold_ms: u64,
    #[serde(default)]
    pub motion_energy: f32,
    #[serde(default)]
    pub fingerspell_p: f32,
    #[serde(default)]
    pub app: String,
    #[serde(default)]
    pub url: String,
    #[serde(default)]
    pub last_action: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecideResponseDto {
    pub intent: String,
    pub confidence: f64,
    pub route: String,
    pub addressing_computer: bool,
    pub needs_confirmation: bool,
    pub is_risky: bool,
    pub is_multi_step: bool,
    pub provider_route: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecuteRequestDto {
    pub action_id: String,
    pub intent: String,
    pub tool: String,
    #[serde(default)]
    pub args: serde_json::Value,
    pub risk_tier: u8,
    pub dry_run: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomeDto {
    pub status: String,
    pub summary: String,
    pub verified: Option<bool>,
}

/// One gated tool call awaiting user approval (opaque round-trip).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentPendingDto {
    pub id: String,
    pub tool: String,
    pub args: serde_json::Value,
}

/// End-to-end agent turn: free-form instruction in, tool loop out.
/// `history` + `pending` resume a paused run verbatim (stateless server).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRequestDto {
    pub text: String,
    #[serde(default)]
    pub app: String,
    #[serde(default)]
    pub url: String,
    #[serde(default)]
    pub history: Vec<serde_json::Value>,
    #[serde(default)]
    pub pending: Option<AgentPendingDto>,
    #[serde(default)]
    pub approve: bool,
    #[serde(default)]
    pub max_steps: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStepDto {
    pub tool: String,
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResponseDto {
    /// done | awaiting_confirm | ignored | error
    pub status: String,
    pub summary: String,
    #[serde(default)]
    pub steps: Vec<AgentStepDto>,
    #[serde(default)]
    pub history: Vec<serde_json::Value>,
    #[serde(default)]
    pub pending: Option<AgentPendingDto>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDto {
    pub tool: String,
    pub available: bool,
    pub via: Option<String>,
    pub note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvailabilityDto {
    pub session_type: String,
    pub desktop: String,
    pub tools: Vec<ToolDto>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JournalEntryDto {
    pub seq: u64,
    pub at_ms: u64,
    pub action_id: String,
    pub intent: String,
    pub tool: String,
    pub status: String,
    pub summary: String,
    pub verified: Option<bool>,
}

impl From<&alive_journal::JournalEntry> for JournalEntryDto {
    fn from(e: &alive_journal::JournalEntry) -> Self {
        Self {
            seq: e.seq,
            at_ms: e.at_ms,
            action_id: e.action_id.clone(),
            intent: e.intent.clone(),
            tool: e.tool.clone(),
            status: format!("{:?}", e.status).to_lowercase(),
            summary: e.summary.clone(),
            verified: e.verified,
        }
    }
}
