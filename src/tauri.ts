// Tauri bridge with an honest browser-dev fallback (static mocks, loudly labeled).
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import type { BusEnvelope } from './types';

export function isTauri(): boolean {
  return typeof window !== 'undefined' && '__TAURI_INTERNALS__' in window;
}

const MOCKS: Record<string, unknown> = {
  get_status: {
    orb_state: 'idle', mode: 'dictation', provider: 'mock', vision: 'mock',
    fps: 0, capture_ms: 0, landmark_ms: 0, camera: false, pack: 'csl',
    running: false, armed: false,
  },
  pack_info: {
    id: 'csl', display_name: 'Chinese Sign Language (简体命令集)', tier: 'C', entries: 26,
    intents: ['open_app', 'dictate', 'cancel'], defs: [], templates: {},
  },
  tool_availability: { session_type: 'browser', desktop: 'mock', tools: [] },
  journal_recent: [],
};

export async function cmd<T>(name: string, args?: Record<string, unknown>): Promise<T> {
  if (!isTauri()) {
    if (name in MOCKS) return MOCKS[name] as T;
    console.warn(`[mock] invoke ${name} ignored in browser dev`);
    return undefined as T;
  }
  return invoke<T>(name, args);
}

export async function onAliveEvent(cb: (e: BusEnvelope) => void): Promise<() => void> {
  if (!isTauri()) return () => {};
  const unlisten = await listen<BusEnvelope>('alive-event', (e) => cb(e.payload));
  return unlisten;
}

/** PiP frame URL with cache-busting seq. */
export function frameUrl(seq: number): string {
  return `aliveframe://localhost/frame.jpg?seq=${seq}`;
}
