//! Temporal super-resolution: ~10Hz real poses → 240Hz understanding ticks.
//!
//! Dual path per bracket (previous real → current real):
//! - hand centroids: monotone cubic (PCHIP) with confidence blending,
//! - provenance on every tick so downstream always knows real vs virtual.
//!
//! This module is CAUSAL (no lookahead): brackets emit when the right
//! endpoint arrives, so virtuals lag ~105ms. That lag is honest and lives
//! inside the 350/700ms endpointing budget. Prediction (the future) belongs
//! to [`crate::kalman`] and is fenced to UI/overlay only.
//!
//! Anti-hallucination rules (R1–R7):
//! - R1 per-hand independence: a missing hand on either end kills that
//!   channel's virtuals (no cross-hand borrowing).
//! - R2 handled by the caller: centroid confidence gates trail entry.
//! - R3 confidence blend: low-confidence brackets blend PCHIP → linear.
//! - R4 stillness: <2px brackets emit Hold (linear) ticks, never cubic.
//! - R5 window: brackets over 200ms emit endpoints only + reset tangents.
//! - R6 velocity clamp: virtual speed ≤ max(3× secant, 2400px/s) or the
//!   whole bracket falls back to linear.
//! - R7 real-aware gate (caller side): recognition needs ≥3 reals spanning
//!   ≥250ms; see [`real_span_ok`].

/// Target virtual tick grid spacing (240Hz).
pub const TICK_US: u64 = 4167;
/// Max bracket we interpolate (R5).
pub const MAX_BRACKET_US: u64 = 200_000;
/// Brackets at/below this emit endpoints only (R4 passthrough).
pub const MIN_BRACKET_US: u64 = 8_000;
/// Stillness displacement threshold in px (R4).
pub const STILL_PX: f32 = 2.0;
/// Absolute virtual-speed cap in px/s (R6).
pub const VCAP_PX_S: f64 = 2400.0;

/// A hand centroid observation.
#[derive(Debug, Clone, Copy)]
pub struct Centroid {
    pub x: f32,
    pub y: f32,
    /// Mean keypoint score 0..1.
    pub score: f32,
    /// Contributing keypoints (0..21).
    pub n_keys: u8,
}

/// One real pose feeding the upsampler.
#[derive(Debug, Clone, Copy)]
pub struct PoseSample {
    pub t_cap_us: u64,
    pub present: bool,
    pub a: Option<Centroid>,
    pub b: Option<Centroid>,
}

/// Interpolation scheme behind a virtual tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scheme {
    /// Monotone cubic (confident, moving).
    Pchip,
    /// Linear fallback (low confidence or velocity-clamp trip).
    Linear,
    /// Linear by stillness rule (R4).
    Hold,
}

/// Provenance of a tick: real poses vs virtual interpolation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Provenance {
    Real { seq: u64 },
    Virtual {
        left_seq: u64,
        right_seq: u64,
        alpha: f32,
        scheme: Scheme,
    },
}

impl Provenance {
    pub fn is_real(&self) -> bool {
        matches!(self, Provenance::Real { .. })
    }
}

/// One understanding tick: real or virtual, both hands.
#[derive(Debug, Clone, Copy)]
pub struct Tick {
    pub t_cap_us: u64,
    pub prov: Provenance,
    pub a: Option<(f32, f32)>,
    pub b: Option<(f32, f32)>,
}

/// R7 gate helper: (n_real, span_us) over a tick slice.
pub fn real_span(ticks: &[Tick]) -> (usize, u64) {
    let mut n = 0usize;
    let mut t0 = u64::MAX;
    let mut t1 = 0u64;
    for t in ticks {
        if t.prov.is_real() {
            n += 1;
            t0 = t0.min(t.t_cap_us);
            t1 = t1.max(t.t_cap_us);
        }
    }
    if n == 0 {
        (0, 0)
    } else {
        (n, t1 - t0)
    }
}

/// R7: recognition needs ≥3 reals spanning ≥250ms.
pub fn real_span_ok(ticks: &[Tick]) -> bool {
    let (n, span) = real_span(ticks);
    n >= 3 && span >= 250_000
}

/// Per-hand bracket state (R1 independence).
#[derive(Debug, Clone, Copy, Default)]
struct Channel {
    /// Previous-previous knot (t_us, x, y) for PCHIP tangents.
    pp: Option<(u64, f64, f64)>,
    /// Previous knot.
    p: Option<(u64, f64, f64, f32, u8)>,
}

/// Bracket upsampler. Feed real poses, get virtual ticks + the real.
#[derive(Debug, Default)]
pub struct Upsampler {
    ch: [Channel; 2],
    seq: u64,
}

impl Upsampler {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Push one real pose. Returns the bracket ticks (virtuals, oldest
    /// first) followed by the real tick itself. First sample returns just
    /// the real (no bracket yet).
    pub fn push(&mut self, s: PoseSample) -> Vec<Tick> {
        let mut out = Vec::new();
        let seq = self.seq;
        self.seq += 1;
        if !s.present {
            // Track lost: reset tangents (R5-style), emit the real marker.
            self.ch = [Channel::default(), Channel::default()];
            out.push(Tick {
                t_cap_us: s.t_cap_us,
                prov: Provenance::Real { seq },
                a: None,
                b: None,
            });
            return out;
        }
        let hands = [s.a, s.b];
        // Snapshot left edges BEFORE brackets advance the channels.
        let left = [self.ch[0].p.map(|k| k.0), self.ch[1].p.map(|k| k.0)];
        // Per-channel virtuals (may differ in count only if a hand drops;
        // timestamps share the sample clock).
        let mut va: Vec<(u64, f32, f32, Scheme)> = vec![];
        let mut vb: Vec<(u64, f32, f32, Scheme)> = vec![];
        for (i, h) in hands.iter().enumerate() {
            let v = self.channel_bracket(i, s.t_cap_us, *h, seq);
            if i == 0 {
                va = v;
            } else {
                vb = v;
            }
        }
        // Merge by timestamp (both channels share the grid when both
        // present; missing channels yield None on that tick).
        let mut ia = 0usize;
        let mut ib = 0usize;
        while ia < va.len() || ib < vb.len() {
            let ta = va.get(ia).map(|v| v.0).unwrap_or(u64::MAX);
            let tb = vb.get(ib).map(|v| v.0).unwrap_or(u64::MAX);
            let t = ta.min(tb);
            let mut pa = None;
            let mut pb = None;
            let mut scheme = Scheme::Linear;
            if ta == t {
                pa = Some((va[ia].1, va[ia].2));
                scheme = va[ia].3;
                ia += 1;
            }
            if tb == t {
                pb = Some((vb[ib].1, vb[ib].2));
                scheme = vb[ib].3;
                ib += 1;
            }
            let l = if ta == t { left[0] } else { left[1] }.unwrap_or(t);
            let dt = t.saturating_sub(l);
            let span = s.t_cap_us.saturating_sub(l);
            let alpha = if span > 0 { dt as f32 / span as f32 } else { 0.0 };
            out.push(Tick {
                t_cap_us: t,
                prov: Provenance::Virtual {
                    left_seq: seq.saturating_sub(1),
                    right_seq: seq,
                    alpha,
                    scheme,
                },
                a: pa,
                b: pb,
            });
        }
        out.push(Tick {
            t_cap_us: s.t_cap_us,
            prov: Provenance::Real { seq },
            a: s.a.map(|c| (c.x, c.y)),
            b: s.b.map(|c| (c.x, c.y)),
        });
        out
    }

    /// Emit one channel's bracket virtuals; update tangent memory.
    /// Returns (t_us, x, y, scheme) oldest-first.
    fn channel_bracket(
        &mut self,
        i: usize,
        t1: u64,
        cur: Option<Centroid>,
        _seq: u64,
    ) -> Vec<(u64, f32, f32, Scheme)> {
        let ch = &mut self.ch[i];
        let Some(c) = cur else {
            // R1: missing hand kills this channel's bracket.
            ch.pp = None;
            ch.p = None;
            return vec![];
        };
        let (x1, y1) = (c.x as f64, c.y as f64);
        let Some((t0, x0, y0, s0, n0)) = ch.p else {
            ch.p = Some((t1, x1, y1, c.score, c.n_keys));
            return vec![];
        };
        let dt_us = t1.saturating_sub(t0);
        let mut out = vec![];
        // R5: oversized bracket → endpoints only + tangent reset.
        if dt_us > MAX_BRACKET_US {
            ch.pp = None;
            ch.p = Some((t1, x1, y1, c.score, c.n_keys));
            return out;
        }
        // Passthrough: tiny bracket → endpoints only (tangents kept).
        if dt_us <= MIN_BRACKET_US {
            ch.pp = ch.p.map(|k| (k.0, k.1, k.2));
            ch.p = Some((t1, x1, y1, c.score, c.n_keys));
            return out;
        }
        let dx = x1 - x0;
        let dy = y1 - y0;
        let disp = (dx * dx + dy * dy).sqrt();
        // Grid strictly inside (t0, t1).
        let mut grid: Vec<u64> = vec![];
        let mut t = t0 + TICK_US;
        while t < t1 {
            grid.push(t);
            t += TICK_US;
        }
        if grid.is_empty() {
            ch.pp = ch.p.map(|k| (k.0, k.1, k.2));
            ch.p = Some((t1, x1, y1, c.score, c.n_keys));
            return out;
        }
        // R4: still bracket → Hold (linear) ticks.
        if disp < STILL_PX as f64 {
            for &tg in &grid {
                let a = (tg - t0) as f64 / dt_us as f64;
                out.push((
                    tg,
                    (x0 + dx * a) as f32,
                    (y0 + dy * a) as f32,
                    Scheme::Hold,
                ));
            }
            ch.pp = ch.p.map(|k| (k.0, k.1, k.2));
            ch.p = Some((t1, x1, y1, c.score, c.n_keys));
            return out;
        }
        // PCHIP tangents (px/s): interior m0, one-sided m1.
        let h1 = dt_us as f64 / 1e6;
        let d1 = (dx / h1, dy / h1);
        let m0 = match ch.pp {
            Some((tp, xp, yp)) => {
                let h0 = (t0.saturating_sub(tp) as f64 / 1e6).max(1e-6);
                let d0 = ((x0 - xp) / h0, (y0 - yp) / h0);
                (
                    fc_interior(h0, h1, d0.0, d1.0),
                    fc_interior(h0, h1, d0.1, d1.1),
                )
            }
            // No history: zero-slope start (no phantom velocity).
            None => (0.0, 0.0),
        };
        let no_history = ch.pp.is_none();
        let m1 = match ch.pp {
            Some((tp, xp, yp)) => {
                let h0 = (t0.saturating_sub(tp) as f64 / 1e6).max(1e-6);
                let d0 = ((x0 - xp) / h0, (y0 - yp) / h0);
                (
                    fc_endpoint(h0, h1, d0.0, d1.0),
                    fc_endpoint(h0, h1, d0.1, d1.1),
                )
            }
            None => (0.0, 0.0),
        };
        // R3: confidence blend PCHIP ↔ linear.
        let w = {
            let smin = s0.min(c.score) as f64;
            let nmin = (n0.min(c.n_keys) as f64 / 6.0).min(1.0);
            ((smin - 0.1) / 0.5).clamp(0.0, 1.0) * nmin
        };
        // Candidate PCHIP ticks + R6 velocity verify (fallback: linear).
        let mut cand: Vec<(f64, f64)> = Vec::with_capacity(grid.len());
        for &tg in &grid {
            let s = (tg - t0) as f64 / dt_us as f64;
            let px = hermite(x0, m0.0, x1, m1.0, h1, s);
            let py = hermite(y0, m0.1, y1, m1.1, h1, s);
            let lx = x0 + dx * s;
            let ly = y0 + dy * s;
            cand.push((w * px + (1.0 - w) * lx, w * py + (1.0 - w) * ly));
        }
        let mut vmax = 0.0f64;
        let mut px = x0;
        let mut py = y0;
        let mut pt = t0;
        for (k, &(cx, cy)) in cand.iter().enumerate() {
            let dt = (grid[k] - pt) as f64 / 1e6;
            if dt > 0.0 {
                vmax = vmax.max((((cx - px).powi(2) + (cy - py).powi(2)).sqrt()) / dt);
            }
            px = cx;
            py = cy;
            pt = grid[k];
        }
        // Final hop to the endpoint counts too.
        let dt = (t1 - pt) as f64 / 1e6;
        if dt > 0.0 {
            vmax = vmax.max((((x1 - px).powi(2) + (y1 - py).powi(2)).sqrt()) / dt);
        }
        let secant = disp / h1;
        // No tangent history (first bracket / post-gap): linear is the
        // honest scheme — PCHIP without 3 knots invents curvature.
        let scheme = if no_history {
            Scheme::Linear
        } else if vmax > secant * 3.0 && vmax > VCAP_PX_S {
            Scheme::Linear
        } else {
            Scheme::Pchip
        };
        for (k, &tg) in grid.iter().enumerate() {
            let (cx, cy) = if scheme == Scheme::Pchip {
                cand[k]
            } else {
                let s = (tg - t0) as f64 / dt_us as f64;
                (x0 + dx * s, y0 + dy * s)
            };
            out.push((tg, cx as f32, cy as f32, scheme));
        }
        ch.pp = ch.p.map(|k| (k.0, k.1, k.2));
        ch.p = Some((t1, x1, y1, c.score, c.n_keys));
        out
    }
}

/// Fritsch–Carlson interior tangent (non-uniform grid), px/s.
fn fc_interior(h0: f64, h1: f64, d0: f64, d1: f64) -> f64 {
    if d0 == 0.0 || d1 == 0.0 || d0.signum() != d1.signum() {
        return 0.0;
    }
    let w1 = 2.0 * h1 + h0;
    let w2 = h1 + 2.0 * h0;
    (w1 + w2) / (w1 / d0 + w2 / d1)
}

/// One-sided endpoint tangent with clamping, px/s.
fn fc_endpoint(h0: f64, h1: f64, d0: f64, d1: f64) -> f64 {
    let mut m = ((2.0 * h1 + h0) * d1 - h1 * d0) / (h0 + h1).max(1e-9);
    if m.signum() != d1.signum() {
        m = 0.0;
    } else if d0.signum() != d1.signum() && m.abs() > 3.0 * d1.abs() {
        m = 3.0 * d1.abs() * d1.signum();
    }
    m
}

/// Cubic Hermite evaluation. s in [0,1], h in seconds, m in px/s.
fn hermite(p0: f64, m0: f64, p1: f64, m1: f64, h: f64, s: f64) -> f64 {
    let s2 = s * s;
    let s3 = s2 * s;
    let h00 = 2.0 * s3 - 3.0 * s2 + 1.0;
    let h10 = s3 - 2.0 * s2 + s;
    let h01 = -2.0 * s3 + 3.0 * s2;
    let h11 = s3 - s2;
    h00 * p0 + h10 * h * m0 + h01 * p1 + h11 * h * m1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(t_us: u64, x: f64, y: f64) -> PoseSample {
        PoseSample {
            t_cap_us: t_us,
            present: true,
            a: Some(Centroid {
                x: x as f32,
                y: y as f32,
                score: 0.9,
                n_keys: 12,
            }),
            b: None,
        }
    }

    fn run(samples: &[PoseSample]) -> Vec<Tick> {
        let mut u = Upsampler::new();
        let mut ticks = vec![];
        for s in samples {
            ticks.extend(u.push(*s));
        }
        ticks
    }

    /// Dense 240Hz ground truth, linear-interpolated at any t.
    fn gt_interp(gt: &[(u64, (f64, f64))], t: u64) -> (f64, f64) {
        let i = gt.partition_point(|g| g.0 < t);
        if i == 0 {
            return gt[0].1;
        }
        if i >= gt.len() {
            return gt[gt.len() - 1].1;
        }
        let (t0, p0) = gt[i - 1];
        let (t1, p1) = gt[i];
        let a = (t - t0) as f64 / (t1 - t0).max(1) as f64;
        (p0.0 + (p1.0 - p0.0) * a, p0.1 + (p1.1 - p0.1) * a)
    }

    fn decimate(
        gt: &[(u64, (f64, f64))],
        every: usize,
        phase: usize,
    ) -> Vec<PoseSample> {
        gt.iter()
            .enumerate()
            .filter(|(i, _)| i % every == phase % every)
            .map(|(_, &(t, p))| sample(t, p.0, p.1))
            .collect()
    }

    /// A1: circle RMSE < 4px across all 24 decimation phases.
    #[test]
    fn suite_a_circle_rmse() {
        let mut gt: Vec<(u64, (f64, f64))> = vec![];
        for i in 0..288 {
            let t = i as f64 / 240.0;
            let a = 2.0 * std::f64::consts::PI * t / 1.2;
            gt.push((
                (i as u64 * 1_000_000) / 240,
                (320.0 + 120.0 * a.cos(), 240.0 + 120.0 * a.sin()),
            ));
        }
        for phase in 0..24 {
            let ticks = run(&decimate(&gt, 24, phase));
            let mut se = 0.0f64;
            let mut n = 0u64;
            for tk in &ticks {
                if let Some((x, y)) = tk.a {
                    let (gx, gy) = gt_interp(&gt, tk.t_cap_us);
                    se += (x as f64 - gx).powi(2) + (y as f64 - gy).powi(2);
                    n += 1;
                }
            }
            let rmse = (se / n.max(1) as f64).sqrt();
            assert!(rmse < 4.0, "phase {phase}: rmse {rmse:.2}px");
        }
    }

    /// A2: fast flick overshoot < 6px (PCHIP monotonicity).
    #[test]
    fn suite_a_flick_no_overshoot() {
        let mut gt: Vec<(u64, (f64, f64))> = vec![];
        for i in 0..72 {
            // smoothstep 100 → 300 in 150ms, hold.
            let t = i as f64 / 240.0;
            let s = (t / 0.15).clamp(0.0, 1.0);
            let s = s * s * (3.0 - 2.0 * s);
            gt.push((
                (i as u64 * 1_000_000) / 240,
                (100.0 + 200.0 * s, 200.0),
            ));
        }
        let ticks = run(&decimate(&gt, 24, 0));
        for tk in &ticks {
            if let Some((x, y)) = tk.a {
                assert!(
                    x > 94.0 && x < 306.0,
                    "overshoot: x={x} at t={}",
                    tk.t_cap_us
                );
                assert!((y - 200.0).abs() < 1.0, "lateral wiggle: y={y}");
            }
        }
    }

    /// A3: static hold → Hold scheme, wiggle ≤ 1.5px.
    #[test]
    fn suite_a_hold_still() {
        let samples: Vec<PoseSample> = (0..12)
            .map(|i| sample(i * 105_263, 320.0, 240.0))
            .collect();
        let ticks = run(&samples);
        let mut nvirt = 0u32;
        for tk in &ticks {
            match tk.prov {
                Provenance::Real { .. } => {}
                Provenance::Virtual { scheme, .. } => {
                    nvirt += 1;
                    assert_eq!(scheme, Scheme::Hold, "still must Hold");
                    if let Some((x, y)) = tk.a {
                        assert!(
                            (x - 320.0).abs() < 1.5 && (y - 240.0).abs() < 1.5,
                            "hold wiggle: ({x},{y})"
                        );
                    }
                }
            }
        }
        assert!(nvirt > 50, "holds still emit ticks, got {nvirt}");
    }

    /// A4: sharp V corner — apex fidelity, no pre-corner ringing.
    #[test]
    fn suite_a_vturn_apex() {
        let mut gt: Vec<(u64, (f64, f64))> = vec![];
        for i in 0..145 {
            let t = i as f64 / 240.0;
            let p = if t < 0.3 {
                (100.0 + 200.0 * (t / 0.3), 100.0)
            } else {
                (300.0, 100.0 + 200.0 * ((t - 0.3) / 0.3))
            };
            gt.push(((i as u64 * 1_000_000) / 240, p));
        }
        // Knots exactly on the corner (phase 0: 0,100,...,600ms).
        let ticks = run(&decimate(&gt, 24, 0));
        let mut apex = f64::INFINITY;
        for tk in &ticks {
            if let Some((x, y)) = tk.a {
                apex = apex.min(((x - 300.0).powi(2) + (y - 100.0).powi(2)).sqrt() as f64);
                if tk.t_cap_us < 300_000 {
                    assert!(
                        (y - 100.0).abs() < 3.0 && x < 303.0,
                        "pre-corner ring: ({x:.1},{y:.1}) at t={}",
                        tk.t_cap_us
                    );
                }
            }
        }
        assert!(apex < 8.0, "apex fidelity {apex:.1}px");
    }

    /// A5: cost — 200 brackets must stay far under budget.
    #[test]
    fn suite_a_cost() {
        let samples: Vec<PoseSample> = (0..201)
            .map(|i| {
                let t = i as f64 * 0.105;
                sample(
                    i as u64 * 105_000,
                    320.0 + 100.0 * (t * 3.0).sin(),
                    240.0 + 60.0 * (t * 5.0).cos(),
                )
            })
            .collect();
        let t0 = std::time::Instant::now();
        let mut u = Upsampler::new();
        let mut nticks = 0u64;
        for s in &samples {
            nticks += u.push(*s).len() as u64;
        }
        let el = t0.elapsed();
        assert!(nticks > 4000, "must emit virtuals, got {nticks}");
        let per_tick_us = el.as_micros() as f64 / nticks as f64;
        assert!(
            per_tick_us < 25.0,
            "too slow: {per_tick_us:.1}µs/tick ({nticks} ticks)"
        );
    }

    /// Bracket rules: ≤8ms passthrough, >200ms gap reset.
    #[test]
    fn bracket_rules_passthrough_and_gap() {
        let mut u = Upsampler::new();
        assert_eq!(u.push(sample(1_000_000, 0.0, 0.0)).len(), 1);
        // 5ms later: real only.
        let t = u.push(sample(1_005_000, 50.0, 0.0));
        assert_eq!(t.len(), 1);
        assert!(t[0].prov.is_real());
        // 500ms later: gap → real only, tangents reset.
        let t = u.push(sample(1_505_000, 100.0, 0.0));
        assert_eq!(t.len(), 1);
        // Next normal bracket: first-bracket-linear (no history).
        let t = u.push(sample(1_605_000, 150.0, 0.0));
        let virts: Vec<_> = t.iter().filter(|k| !k.prov.is_real()).collect();
        assert!(!virts.is_empty());
        for v in virts {
            match v.prov {
                Provenance::Virtual { scheme, .. } => {
                    assert_eq!(scheme, Scheme::Linear)
                }
                _ => unreachable!(),
            }
        }
    }

    /// Provenance: virtuals carry bracket seqs + monotonic alpha in (0,1).
    #[test]
    fn provenance_brackets_and_alpha() {
        let samples: Vec<PoseSample> = (0..4)
            .map(|i| sample(i * 100_000, i as f64 * 30.0, 0.0))
            .collect();
        let ticks = run(&samples);
        let mut last_alpha = -1.0f32;
        let mut last_right = 0u64;
        for tk in &ticks {
            if let Provenance::Virtual {
                left_seq,
                right_seq,
                alpha,
                ..
            } = tk.prov
            {
                assert_eq!(left_seq + 1, right_seq);
                assert!(alpha > 0.0 && alpha < 1.0, "alpha {alpha}");
                if right_seq == last_right {
                    assert!(alpha > last_alpha, "alpha must rise in-bracket");
                }
                last_alpha = alpha;
                last_right = right_seq;
            }
        }
    }

    /// R1: a missing hand kills only its own channel.
    #[test]
    fn missing_hand_kills_channel_only() {
        let mut u = Upsampler::new();
        let mk = |t: u64, b: bool| PoseSample {
            t_cap_us: t,
            present: true,
            a: Some(Centroid {
                x: 10.0,
                y: 0.0,
                score: 0.9,
                n_keys: 12,
            }),
            b: if b {
                Some(Centroid {
                    x: 50.0,
                    y: 0.0,
                    score: 0.9,
                    n_keys: 12,
                })
            } else {
                None
            },
        };
        u.push(mk(1_000_000, true));
        u.push(mk(1_100_000, true));
        // B drops: A keeps virtuals, B yields nothing.
        let t = u.push(mk(1_200_000, false));
        let va = t.iter().filter(|k| !k.prov.is_real() && k.a.is_some()).count();
        let vb = t.iter().filter(|k| k.b.is_some()).count();
        assert!(va > 10, "A channel must interpolate");
        assert_eq!(vb, 0, "B channel must be silent");
    }

    /// R7 gate math.
    #[test]
    fn r7_real_span_gate() {
        let samples: Vec<PoseSample> = (0..6)
            .map(|i| sample(i * 100_000, i as f64 * 10.0, 0.0))
            .collect();
        let ticks = run(&samples);
        let (n, span) = real_span(&ticks);
        assert_eq!(n, 6);
        assert_eq!(span, 500_000);
        assert!(real_span_ok(&ticks));
        // Short span fails R7 even with ticks present.
        let short: Vec<PoseSample> = (0..6)
            .map(|i| sample(i * 10_000, i as f64 * 10.0, 0.0))
            .collect();
        assert!(!real_span_ok(&run(&short)));
    }
}
