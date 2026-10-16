import { useCallback, useEffect, useRef, useState } from 'react';
import { listen } from '@tauri-apps/api/event';
import { AgentState, Orb } from './components/ui/orb';
import { cmd, onAliveEvent } from './tauri';
import type { AgentPendingDto, AgentResponseDto, BusEnvelope } from './types';

type Mode = 'prompt' | 'computer';

// Orb only. No words, no captions, no chrome.
export default function OrbBar() {
  const [mode, setMode] = useState<Mode | null>(null);
  const [agent, setAgent] = useState<AgentState>(null);
  const [pending, setPending] = useState<AgentPendingDto | null>(null);
  const [history, setHistory] = useState<unknown[]>([]);
  const [busy, setBusy] = useState(false);
  const modeRef = useRef<Mode | null>(null);
  modeRef.current = mode;
  const hideTimer = useRef<number | null>(null);

  const hide = useCallback(async () => {
    if (hideTimer.current) window.clearTimeout(hideTimer.current);
    try {
      await cmd('overlay_hide');
    } catch {
      /* browser dev */
    }
  }, []);

  const later = useCallback(
    (ms: number) => {
      if (hideTimer.current) window.clearTimeout(hideTimer.current);
      hideTimer.current = window.setTimeout(() => void hide(), ms);
    },
    [hide],
  );

  const reset = useCallback((m: Mode) => {
    setMode(m);
    setAgent(null);
    setPending(null);
    setHistory([]);
    setBusy(false);
  }, []);

  const runAgent = useCallback(
    async (text: string, hist: unknown[], pend: AgentPendingDto | null, approve: boolean) => {
      setBusy(true);
      try {
        const out = await cmd<AgentResponseDto>('agent_act', {
          dto: { text, app: '', url: '', history: hist, pending: pend, approve, max_steps: 6 },
        });
        setHistory(out.history ?? []);
        setPending(out.pending ?? null);
        if (out.status === 'done' || out.status === 'ignored' || out.status === 'error') {
          later(1800);
        }
      } catch {
        later(1800);
      } finally {
        setBusy(false);
      }
    },
    [later],
  );

  useEffect(() => {
    let off = () => {};
    listen<{ mode: Mode }>('overlay-mode', (e) => reset(e.payload.mode))
      .then((fn) => {
        off = fn;
      })
      .catch(() => {});
    cmd<{ mode: Mode | null }>('overlay_state')
      .then((st) => {
        if (st.mode) reset(st.mode);
      })
      .catch(() => {});
    return () => off();
  }, [reset]);

  useEffect(() => {
    let off = () => {};
    onAliveEvent((e: BusEnvelope) => {
      const k = e.kind;
      if (modeRef.current === null) return;
      switch (k.type) {
        case 'partial_gloss':
        case 'action_started':
          setAgent('listening');
          break;
        case 'committed': {
          if (!k.text) break;
          if (modeRef.current === 'prompt') {
            setAgent('listening');
            cmd<string>('overlay_commit_type', { text: k.text })
              .then(() => later(900))
              .catch(() => later(1600));
          } else {
            setAgent('talking');
            void runAgent(k.text, [], null, false);
          }
          break;
        }
        case 'action_completed':
          if (!k.ok && modeRef.current === 'prompt') later(1600);
          break;
        case 'error':
          later(1600);
          break;
        default:
          break;
      }
    }).then((offFn) => {
      off = offFn;
    });
    return () => off();
  }, [runAgent, later]);

  useEffect(() => {
    const dn = (e: KeyboardEvent) => {
      if (e.key === 'Escape') void hide();
      if (e.key === 'Enter' && pending && !busy) {
        setAgent('talking');
        void runAgent('', history, pending, true);
      }
    };
    window.addEventListener('keydown', dn);
    return () => window.removeEventListener('keydown', dn);
  }, [hide, pending, busy, history, runAgent]);

  return (
    <div className="orb-stage">
      <Orb seed={7} agentState={agent} className="orb-fill" />
    </div>
  );
}
