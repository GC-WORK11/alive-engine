// Mirrors alive-ipc DTOs + alive-core events 1:1. If Rust changes, this changes.

export type OrbState =
  | 'idle' | 'activating' | 'camera_active' | 'signer_detected' | 'interpreting'
  | 'intent_forming' | 'ready' | 'executing' | 'done' | 'confirm_needed'
  | 'error' | 'recovery';

export type AppMode = 'dictation' | 'action';

export interface StatusDto {
  orb_state: OrbState;
  mode: AppMode;
  provider: string;
  vision: string;
  fps: number;
  capture_ms: number;
  landmark_ms: number;
  camera: boolean;
  pack: string;
  running: boolean;
  armed: boolean;
  motion: number;
  tracking: boolean;
  upsample_hz: number;
}

export interface HypDto {
  gloss: string;
  p: number;
}

export interface DecideResponseDto {
  intent: string;
  confidence: number;
  route: 'act' | 'confirm' | 'ask';
  addressing_computer: boolean;
  needs_confirmation: boolean;
  is_risky: boolean;
  is_multi_step: boolean;
  provider_route: string;
}

export interface OutcomeDto {
  status: string;
  summary: string;
  verified: boolean | null;
}

export interface AgentPendingDto {
  id: string;
  tool: string;
  args: unknown;
}

export interface AgentStepDto {
  tool: string;
  summary: string;
}

export interface AgentResponseDto {
  status: 'done' | 'awaiting_confirm' | 'ignored' | 'error';
  summary: string;
  steps: AgentStepDto[];
  history: unknown[];
  pending: AgentPendingDto | null;
}

export interface ToolDto {
  tool: string;
  available: boolean;
  via: string | null;
  note: string;
}

export interface AvailabilityDto {
  session_type: string;
  desktop: string;
  tools: ToolDto[];
}

export interface JournalEntryDto {
  seq: number;
  at_ms: number;
  action_id: string;
  intent: string;
  tool: string;
  status: string;
  summary: string;
  verified: boolean | null;
}

export interface IntentDef {
  intent: string;
  action: string;
  risk_tier: number;
  slots: string[];
  patterns: string[];
}

export interface PackInfo {
  id: string;
  display_name: string;
  tier: string;
  entries: number;
  intents: string[];
  defs: IntentDef[];
  templates: Record<string, number>;
}

export type PipelineEvent =
  | { type: 'orb_state_changed'; from: OrbState; to: OrbState }
  | { type: 'partial_gloss'; text: string; speculative: boolean }
  | { type: 'committed'; text: string; segment_ms: number }
  | { type: 'intent_decided'; intent: string; confidence: number; route: string }
  | { type: 'action_started'; action_id: string; intent: string }
  | { type: 'action_completed'; action_id: string; ok: boolean; summary: string }
  | { type: 'confirm_requested'; action_id: string; preview: string }
  | { type: 'error'; message: string }
  | { type: 'info'; message: string }
  | { type: 'perf_sample'; capture_ms: number; landmark_ms: number; fps: number };

export interface BusEnvelope {
  seq: number;
  at_ms: number;
  kind: PipelineEvent;
}
