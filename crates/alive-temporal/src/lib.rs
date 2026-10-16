//! alive-temporal: streaming segmentation over landmark streams.
//!
//! Causal by construction: `Endpointer::push` sees one frame at a time and
//! emits PRD §3 events — Tentative at 300–400ms of stillness (speculative UI),
//! Commit at 600–800ms AND EosScorer above boundary_p. Hands lost = Cancelled.
//! No ML weights here; the EOS model is a pluggable trait.

use serde::{Deserialize, Serialize};

/// One landmark frame, flattened xyz * N keypoints. `present=false` means the
/// signer left the frame / hands dropped (a cancel signal, not silence).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LandmarkFrame {
    pub at_ms: u64,
    pub present: bool,
    pub points: Vec<f32>,
}

/// Mean absolute per-coordinate displacement. Length mismatch or empty = 0.
pub fn motion_energy(prev: &[f32], cur: &[f32]) -> f32 {
    if prev.is_empty() || prev.len() != cur.len() {
        return 0.0;
    }
    let mut s = 0.0f32;
    for (a, b) in prev.iter().zip(cur.iter()) {
        s += (a - b).abs();
    }
    s / prev.len() as f32
}

/// Fixed-capacity ring. Oldest-first iteration for temporal windows.
#[derive(Debug, Clone)]
pub struct RingBuffer<T> {
    buf: Vec<T>,
    cap: usize,
    head: usize,
    len: usize,
}

impl<T> RingBuffer<T> {
    pub fn new(cap: usize) -> Self {
        assert!(cap > 0, "ring capacity must be > 0");
        Self {
            buf: Vec::with_capacity(cap),
            cap,
            head: 0,
            len: 0,
        }
    }

    pub fn push(&mut self, v: T) {
        if self.len < self.cap {
            self.buf.push(v);
            self.len += 1;
        } else {
            self.buf[self.head] = v;
        }
        self.head = (self.head + 1) % self.cap;
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn last(&self) -> Option<&T> {
        if self.len == 0 {
            None
        } else {
            Some(&self.buf[(self.head + self.cap - 1) % self.cap])
        }
    }

    pub fn iter_oldest_first(&self) -> impl Iterator<Item = &T> {
        (0..self.len).map(move |i| &self.buf[(self.head + self.cap - self.len + i) % self.cap])
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointConfig {
    pub tentative_pause_ms: u64,
    pub commit_pause_ms: u64,
    /// Minimum *active* (moving) time before a segment can emit. Debounce.
    pub min_segment_ms: u64,
    pub boundary_p: f32,
    /// motion_energy below this counts as still.
    pub low_motion: f32,
    /// Stillness beyond this without a commit silently resets (abandoned).
    pub max_pause_ms: u64,
}

impl Default for EndpointConfig {
    fn default() -> Self {
        Self {
            tentative_pause_ms: 350,
            commit_pause_ms: 700,
            min_segment_ms: 250,
            boundary_p: 0.85,
            low_motion: 0.02,
            max_pause_ms: 2000,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum EndpointEvent {
    Tentative {
        start_ms: u64,
        end_ms: u64,
    },
    Commit {
        start_ms: u64,
        end_ms: u64,
        eos_p: f32,
    },
    Cancelled {
        at_ms: u64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EpState {
    Idle,
    Signing,
    Pausing,
}

/// P(end-of-sign) given recent per-frame motion values. The learned TCN
/// implements this trait later; the heuristic below ships first.
pub trait EosScorer {
    fn score(&self, recent_motion: &[f32]) -> f32;
}

/// Fraction of recent frames under the stillness floor. After a real pause
/// this converges to 1.0, which is exactly the commit gate.
#[derive(Debug, Clone, Copy)]
pub struct LowMotionEos {
    pub floor: f32,
}

impl EosScorer for LowMotionEos {
    fn score(&self, recent_motion: &[f32]) -> f32 {
        if recent_motion.is_empty() {
            return 0.0;
        }
        let still = recent_motion.iter().filter(|&&m| m < self.floor).count();
        still as f32 / recent_motion.len() as f32
    }
}

pub struct Endpointer {
    cfg: EndpointConfig,
    state: EpState,
    seg_start: u64,
    prev_ms: u64,
    last_motion_ms: u64,
    active_ms: u64,
    tentative_fired: bool,
}

impl Endpointer {
    pub fn new(cfg: EndpointConfig) -> Self {
        Self {
            cfg,
            state: EpState::Idle,
            seg_start: 0,
            prev_ms: 0,
            last_motion_ms: 0,
            active_ms: 0,
            tentative_fired: false,
        }
    }

    fn reset(&mut self) {
        self.state = EpState::Idle;
        self.active_ms = 0;
        self.tentative_fired = false;
    }

    /// Feed one frame. Returns at most one event. Gaps in `at_ms` are fine.
    pub fn push(
        &mut self,
        at_ms: u64,
        motion: f32,
        eos_p: f32,
        present: bool,
    ) -> Option<EndpointEvent> {
        if !present {
            if self.state != EpState::Idle {
                self.reset();
                return Some(EndpointEvent::Cancelled { at_ms });
            }
            return None;
        }
        let dt = at_ms.saturating_sub(self.prev_ms);
        self.prev_ms = at_ms;
        let moving = motion >= self.cfg.low_motion;

        match self.state {
            EpState::Idle => {
                if moving {
                    self.state = EpState::Signing;
                    self.seg_start = at_ms;
                    self.last_motion_ms = at_ms;
                    self.active_ms = 0;
                    self.tentative_fired = false;
                }
                None
            }
            EpState::Signing => {
                if moving {
                    self.last_motion_ms = at_ms;
                    self.active_ms = self.active_ms.saturating_add(dt.min(250));
                    None
                } else {
                    self.state = EpState::Pausing;
                    self.check_pause(at_ms, eos_p)
                }
            }
            EpState::Pausing => {
                if moving {
                    self.state = EpState::Signing;
                    self.last_motion_ms = at_ms;
                    self.active_ms = self.active_ms.saturating_add(dt.min(250));
                    None
                } else {
                    self.check_pause(at_ms, eos_p)
                }
            }
        }
    }

    fn check_pause(&mut self, at_ms: u64, eos_p: f32) -> Option<EndpointEvent> {
        let pause = at_ms.saturating_sub(self.last_motion_ms);
        if pause >= self.cfg.max_pause_ms {
            // Abandoned segment (e.g. sub-minimum flicker + long stillness).
            self.reset();
            return None;
        }
        if self.active_ms < self.cfg.min_segment_ms {
            return None;
        }
        if !self.tentative_fired && pause >= self.cfg.tentative_pause_ms {
            self.tentative_fired = true;
            return Some(EndpointEvent::Tentative {
                start_ms: self.seg_start,
                end_ms: at_ms,
            });
        }
        if pause >= self.cfg.commit_pause_ms && eos_p >= self.cfg.boundary_p {
            let ev = EndpointEvent::Commit {
                start_ms: self.seg_start,
                end_ms: at_ms,
                eos_p,
            };
            self.reset();
            return Some(ev);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_orders_oldest_first_across_wrap() {
        let mut r = RingBuffer::new(3);
        for i in 0..5 {
            r.push(i);
        }
        let v: Vec<_> = r.iter_oldest_first().copied().collect();
        assert_eq!(v, vec![2, 3, 4]);
        assert_eq!(r.last(), Some(&4));
    }

    #[test]
    fn motion_energy_basics() {
        assert_eq!(motion_energy(&[], &[]), 0.0);
        assert_eq!(motion_energy(&[1.0], &[1.0, 2.0]), 0.0);
        assert!((motion_energy(&[0.0, 0.0], &[1.0, 3.0]) - 2.0).abs() < 1e-6);
    }

    /// Drive 30fps: 600ms signing, then stillness. Expect Tentative ~350ms
    /// into the pause and Commit ~700ms in with eos=1.
    #[test]
    fn tentative_then_commit_on_pause() {
        let mut ep = Endpointer::new(EndpointConfig::default());
        let eos = LowMotionEos { floor: 0.02 };
        // 15 frames @30fps = 0.5s right context (PRD: 0.3-0.5s).
        let mut recent = RingBuffer::new(15);
        let mut events = Vec::new();
        // 600ms of motion at 30fps (frame every 33ms), t=0..600
        let mut t = 0u64;
        while t < 600 {
            recent.push(0.5f32);
            let p = eos.score(&recent.iter_oldest_first().copied().collect::<Vec<_>>());
            if let Some(e) = ep.push(t, 0.5, p, true) {
                events.push((t, e));
            }
            t += 33;
        }
        // Stillness until t=1600
        while t < 1600 {
            recent.push(0.001f32);
            let p = eos.score(&recent.iter_oldest_first().copied().collect::<Vec<_>>());
            if let Some(e) = ep.push(t, 0.001, p, true) {
                events.push((t, e));
            }
            t += 33;
        }
        assert_eq!(events.len(), 2, "expected tentative+commit, got {events:?}");
        match &events[0].1 {
            EndpointEvent::Tentative { .. } => {
                assert!(
                    (580 + 350..=580 + 450).contains(&events[0].0),
                    "tentative at {}",
                    events[0].0
                );
            }
            e => panic!("first event must be tentative, got {e:?}"),
        }
        match &events[1].1 {
            EndpointEvent::Commit { eos_p, .. } => {
                assert!(
                    (580 + 700..=580 + 800).contains(&events[1].0),
                    "commit at {}",
                    events[1].0
                );
                assert!(*eos_p >= 0.85);
            }
            e => panic!("second event must be commit, got {e:?}"),
        }
    }

    #[test]
    fn flicker_never_commits_and_resets() {
        let mut ep = Endpointer::new(EndpointConfig::default());
        let mut t = 0u64;
        // 60ms flicker
        while t < 60 {
            assert!(ep.push(t, 0.5, 0.0, true).is_none());
            t += 33;
        }
        // 2.5s stillness: no events, silent reset via max_pause
        while t < 2600 {
            assert!(
                ep.push(t, 0.001, 1.0, true).is_none(),
                "flicker emitted at {t}"
            );
            t += 33;
        }
    }

    #[test]
    fn dropped_hands_cancel() {
        let mut ep = Endpointer::new(EndpointConfig::default());
        let mut t = 0u64;
        while t < 400 {
            ep.push(t, 0.5, 0.0, true);
            t += 33;
        }
        let e = ep.push(500, 0.0, 0.0, false);
        assert!(matches!(e, Some(EndpointEvent::Cancelled { at_ms: 500 })));
        // After cancel, idle again: absent hands emit nothing.
        assert!(ep.push(600, 0.0, 0.0, false).is_none());
    }

    #[test]
    fn commit_requires_eos_gate() {
        // Long pause with eos_p pinned low must NOT commit (only tentative).
        let mut ep = Endpointer::new(EndpointConfig::default());
        let mut t = 0u64;
        while t < 500 {
            ep.push(t, 0.5, 0.0, true);
            t += 33;
        }
        let mut saw_tentative = false;
        while t < 2400 {
            if let Some(e) = ep.push(t, 0.001, 0.1, true) {
                match e {
                    EndpointEvent::Tentative { .. } => saw_tentative = true,
                    EndpointEvent::Commit { .. } => panic!("committed with eos_p=0.1"),
                    EndpointEvent::Cancelled { .. } => panic!("unexpected cancel"),
                }
            }
            t += 33;
        }
        assert!(saw_tentative);
    }
}
