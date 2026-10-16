//! Predictive hand tracking: constant-velocity Kalman per axis, two hands.
//!
//! FENCING (hard rule): prediction feeds the overlay, the orb, and reaction
//! triggers ONLY. The matcher, the endpointer, and the trail consume causal
//! interpolation ([`crate::upsample`]) and real poses. Prediction extrapolates
//! (it guesses the future); understanding must never hallucinate.
//!
//! Units: pixels, seconds. Time base: capture timestamps (µs).

/// Scalar constant-velocity Kalman filter, Joseph-form covariance.
#[derive(Debug, Clone)]
pub struct AxisKalman {
    x: [f64; 2],
    p: [[f64; 2]; 2],
    q_pos: f64,
    q_vel: f64,
    r_base: f64,
}

impl AxisKalman {
    pub fn new(pos: f64, q_pos: f64, q_vel: f64, r_base: f64) -> Self {
        Self {
            x: [pos, 0.0],
            p: [[25.0, 0.0], [0.0, 2500.0]],
            q_pos,
            q_vel,
            r_base,
        }
    }

    pub fn reset(&mut self, pos: f64) {
        self.x = [pos, 0.0];
        self.p = [[25.0, 0.0], [0.0, 2500.0]];
    }

    pub fn predict_dt(&mut self, dt: f64) {
        let dt = dt.clamp(0.0, 0.5);
        let (p0, v0) = (self.x[0], self.x[1]);
        self.x = [p0 + v0 * dt, v0];
        let [[a, b], [_, d]] = [self.p[0], self.p[1]];
        let c = b;
        let dt2 = dt * dt;
        let dt3 = dt2 * dt;
        self.p = [
            [
                a + dt * (b + c) + dt2 * d + self.q_pos * dt + self.q_vel * dt3 / 3.0,
                b + dt * d + self.q_vel * dt2 / 2.0,
            ],
            [
                c + dt * d + self.q_vel * dt2 / 2.0,
                d + self.q_vel * dt,
            ],
        ];
        self.p[1][0] = self.p[0][1];
        debug_assert!(self.p[0][0].is_finite() && self.p[1][1].is_finite());
    }

    pub fn update(&mut self, z: f64, r: f64) {
        let r = (r * self.r_base).max(1e-6);
        let s = self.p[0][0] + r;
        if !s.is_finite() || s <= 0.0 {
            return;
        }
        let k0 = self.p[0][0] / s;
        let k1 = self.p[1][0] / s;
        let y = z - self.x[0];
        let yc = y.clamp(-150.0, 150.0);
        self.x[0] += k0 * yc;
        self.x[1] += k1 * yc;
        let i00 = 1.0 - k0;
        let p00 = self.p[0][0];
        let p01 = self.p[0][1];
        let p11 = self.p[1][1];
        self.p[0][0] = i00 * i00 * p00 + k0 * k0 * r;
        self.p[0][1] = i00 * (p01 - k1 * p00) + k0 * k1 * r;
        self.p[1][1] = k1 * k1 * p00 - 2.0 * k1 * p01 + p11 + k1 * k1 * r;
        self.p[1][0] = self.p[0][1];
    }

    pub fn pos(&self) -> f64 {
        self.x[0]
    }

    pub fn vel(&self) -> f64 {
        self.x[1]
    }

    pub fn pos_var(&self) -> f64 {
        self.p[0][0]
    }
}

/// Track state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackState {
    Uninit,
    Tentative,
    Confirmed,
    Coast,
}

/// One hand: 2 Kalman axes + confidence EMA + coast logic.
#[derive(Debug, Clone)]
pub struct HandTrack {
    kx: AxisKalman,
    ky: AxisKalman,
    state: TrackState,
    t_last_us: u64,
    conf_ema: f64,
    updates: u32,
}

impl HandTrack {
    pub fn new() -> Self {
        Self {
            kx: AxisKalman::new(0.0, 4.0, 2500.0, 36.0),
            ky: AxisKalman::new(0.0, 4.0, 2500.0, 36.0),
            state: TrackState::Uninit,
            t_last_us: 0,
            conf_ema: 0.0,
            updates: 0,
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }

    pub fn state(&self) -> TrackState {
        self.state
    }

    pub fn last_update_us(&self) -> u64 {
        self.t_last_us
    }

    /// Observe a centroid. `conf01`: 0..1 fused centroid confidence.
    pub fn observe(&mut self, t_us: u64, x: f32, y: f32, conf01: f32) {
        let conf = conf01.clamp(0.0, 1.0) as f64;
        self.conf_ema += 0.3 * (conf - self.conf_ema);
        if self.state == TrackState::Uninit {
            self.kx.reset(x as f64);
            self.ky.reset(y as f64);
            self.t_last_us = t_us;
            self.state = TrackState::Tentative;
            self.updates = 1;
            return;
        }
        let dt = (t_us.saturating_sub(self.t_last_us) as f64 / 1e6).clamp(0.0, 0.5);
        let pred_x = self.kx.pos() + self.kx.vel() * dt;
        let pred_y = self.ky.pos() + self.ky.vel() * dt;
        let jump = ((x as f64 - pred_x).powi(2) + (y as f64 - pred_y).powi(2)).sqrt();
        if jump > 170.0 {
            self.kx.reset(x as f64);
            self.ky.reset(y as f64);
            self.t_last_us = t_us;
            self.state = TrackState::Tentative;
            self.updates = 1;
            return;
        }
        self.kx.predict_dt(dt);
        self.ky.predict_dt(dt);
        let r = (1.2 - self.conf_ema).clamp(0.2, 1.2);
        self.kx.update(x as f64, r);
        self.ky.update(y as f64, r);
        self.t_last_us = t_us;
        self.updates += 1;
        if self.updates >= 2 && self.conf_ema > 0.15 {
            self.state = TrackState::Confirmed;
        }
    }

    pub fn mark_missed(&mut self, t_us: u64) {
        if self.state == TrackState::Uninit {
            return;
        }
        let gap_ms = t_us.saturating_sub(self.t_last_us) / 1000;
        if gap_ms > 400 {
            self.reset();
        } else if gap_ms > 120 {
            self.state = TrackState::Coast;
        }
    }

    /// Position predicted at `t_us` (may be ahead of last observation).
    /// None while uninitialised or after coast-out.
    pub fn estimate_at(&self, t_us: u64) -> Option<(f32, f32)> {
        match self.state {
            TrackState::Uninit => None,
            TrackState::Tentative | TrackState::Confirmed | TrackState::Coast => {
                if t_us.saturating_sub(self.t_last_us) > 400_000 {
                    return None;
                }
                let dt = (t_us.saturating_sub(self.t_last_us) as f64 / 1e6).clamp(0.0, 0.5);
                let mut kx = self.kx.clone();
                let mut ky = self.ky.clone();
                kx.predict_dt(dt);
                ky.predict_dt(dt);
                Some((kx.pos() as f32, ky.pos() as f32))
            }
        }
    }

    /// Current velocity in px/s (for the orb / reaction triggers).
    pub fn velocity(&self) -> (f32, f32) {
        (self.kx.vel() as f32, self.ky.vel() as f32)
    }

    /// Position uncertainty (1σ px) for overlay gating.
    pub fn uncertainty(&self) -> f32 {
        (self.kx.pos_var().max(0.0).sqrt() + self.ky.pos_var().max(0.0).sqrt()) as f32 * 0.5
    }
}

impl Default for HandTrack {
    fn default() -> Self {
        Self::new()
    }
}

/// Both hands, indexed 0 = A, 1 = B.
#[derive(Debug, Clone, Default)]
pub struct HandTracker {
    pub hands: [HandTrack; 2],
}

impl HandTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        self.hands[0].reset();
        self.hands[1].reset();
    }
}

/// One-Euro adaptive smoother for display scalars (orb level, overlay alpha).
#[derive(Debug, Clone)]
pub struct OneEuro {
    x_hat: f64,
    dx_hat: f64,
    t_last_us: u64,
    init: bool,
    min_cutoff: f64,
    beta: f64,
    d_cutoff: f64,
}

impl OneEuro {
    pub fn new(min_cutoff: f64, beta: f64, d_cutoff: f64) -> Self {
        Self {
            x_hat: 0.0,
            dx_hat: 0.0,
            t_last_us: 0,
            init: false,
            min_cutoff,
            beta,
            d_cutoff,
        }
    }

    pub fn reset(&mut self) {
        self.init = false;
        self.dx_hat = 0.0;
    }

    fn alpha(&self, cutoff: f64, dt: f64) -> f64 {
        if dt <= 0.0 {
            return 1.0;
        }
        let tau = 1.0 / (2.0 * std::f64::consts::PI * cutoff);
        1.0 / (1.0 + tau / dt)
    }

    pub fn filter(&mut self, t_us: u64, x: f64) -> f64 {
        if !self.init {
            self.x_hat = x;
            self.t_last_us = t_us;
            self.init = true;
            return x;
        }
        let dt = (t_us.saturating_sub(self.t_last_us) as f64 / 1e6).clamp(1e-4, 1.0);
        let dx = (x - self.x_hat) / dt;
        let a_d = self.alpha(self.d_cutoff, dt);
        self.dx_hat += a_d * (dx - self.dx_hat);
        let cutoff = self.min_cutoff + self.beta * self.dx_hat.abs();
        let a = self.alpha(cutoff, dt);
        self.x_hat += a * (x - self.x_hat);
        self.t_last_us = t_us;
        self.x_hat
    }

    pub fn value(&self) -> f64 {
        self.x_hat
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Constant-velocity ground truth: filter must converge below the
    /// measurement noise floor and predict ahead accurately.
    #[test]
    fn kalman_converges_constant_velocity() {
        let mut h = HandTrack::new();
        // 300 px/s diagonal for 2s @9.5Hz, ±6px uniform noise (deterministic).
        let mut seed = 0x12345u32;
        let mut rnd = move || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f64 / (u32::MAX as f64 / 12.0) - 6.0
        };
        let mut t = 0u64;
        for i in 0..20 {
            let gt = i as f64 * (300.0 / 9.5);
            h.observe(t, (gt + rnd()) as f32, (gt + rnd()) as f32, 0.9);
            t += 105_263;
        }
        assert_eq!(h.state(), TrackState::Confirmed);
        // Predict 105ms ahead: must beat zero-order-hold on the last sample.
        let t_pred = t + 105_263;
        let (px, py) = h.estimate_at(t_pred).unwrap();
        let gt = 20.0 * (300.0 / 9.5) + 300.0 * 0.105263;
        let err_k = ((px as f64 - gt).powi(2) + (py as f64 - gt).powi(2)).sqrt();
        let last = 19.0 * (300.0 / 9.5);
        let err_z = ((last - gt).powi(2) * 2.0).sqrt();
        assert!(err_k < err_z * 0.5, "kalman {err_k:.1} vs zoh {err_z:.1}");
        assert!(err_k < 12.0, "converged error {err_k:.1}px");
    }

    /// Still hold: velocity collapses, position stays (no drift).
    #[test]
    fn kalman_hold_is_still() {
        let mut h = HandTrack::new();
        let mut t = 0u64;
        for _ in 0..15 {
            h.observe(t, 320.0, 240.0, 0.9);
            t += 105_263;
        }
        let (vx, vy) = h.velocity();
        assert!(vx.abs() < 8.0 && vy.abs() < 8.0, "v=({vx:.1},{vy:.1})");
        let (px, py) = h.estimate_at(t + 200_000).unwrap();
        assert!((px - 320.0).abs() < 3.0 && (py - 240.0).abs() < 3.0);
    }

    /// Teleport gate: a >170px jump re-inits instead of slinging velocity.
    #[test]
    fn kalman_teleport_reinits() {
        let mut h = HandTrack::new();
        let mut t = 0u64;
        for _ in 0..5 {
            h.observe(t, 100.0, 100.0, 0.9);
            t += 105_263;
        }
        h.observe(t, 500.0, 400.0, 0.9);
        assert_eq!(h.state(), TrackState::Tentative);
        let (vx, vy) = h.velocity();
        assert!(vx.abs() + vy.abs() < 1.0, "no sling velocity");
        let (px, py) = h.estimate_at(t).unwrap();
        assert!((px - 500.0).abs() < 1.0 && (py - 400.0).abs() < 1.0);
    }

    /// Coast-out: 400ms without observations → None (never stale).
    #[test]
    fn kalman_coast_out_returns_none() {
        let mut h = HandTrack::new();
        h.observe(1_000_000, 100.0, 100.0, 0.9);
        h.observe(1_105_263, 105.0, 105.0, 0.9);
        assert!(h.estimate_at(1_105_263 + 300_000).is_some());
        h.mark_missed(1_105_263 + 200_000);
        assert_eq!(h.state(), TrackState::Coast);
        assert!(h.estimate_at(1_105_263 + 500_000).is_none());
        h.mark_missed(1_105_263 + 500_000);
        assert_eq!(h.state(), TrackState::Uninit);
    }

    /// OneEuro: still values pass almost unchanged; steps respond fast.
    #[test]
    fn one_euro_smooths_jitter_tracks_steps() {
        let mut f = OneEuro::new(1.0, 0.02, 1.0);
        let mut t = 0u64;
        let mut last = 0.0;
        // Long settle so the step fully converges before jitter is measured.
        for i in 0..120 {
            let x = if i < 30 { 0.5 } else { 1.0 };
            last = f.filter(t, x);
            t += 16_667;
        }
        assert!((last - 1.0).abs() < 0.01, "step must converge, got {last}");
        // Jitter ±0.02 around 1.0 must mostly vanish.
        let mut spread = 0.0f64;
        for i in 0..60 {
            let x = 1.0 + if i % 2 == 0 { 0.02 } else { -0.02 };
            let y = f.filter(t, x);
            spread = spread.max((y - 1.0).abs());
            t += 16_667;
        }
        assert!(spread < 0.012, "jitter must shrink, got {spread}");
    }
}
