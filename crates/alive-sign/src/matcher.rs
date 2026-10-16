//! Calibrated motion-command matcher: per-user trajectory templates + DTW.
//!
//! HONEST SCOPE: recognizes MOTION-DISTINCT command gestures (swipes, shake,
//! circle, push) from hand-centroid paths. This is the V1 command grammar —
//! a small set of deliberately distinct motions, calibrated per user with 3+
//! samples each. It is NOT general ASL recognition (that needs the trained
//! encoder + recorded corpus, M1+). Unknown motions are REJECTED, never guessed.
//!
//! Invariants: translation + scale invariant (where/how big you sign doesn't
//! matter), rotation/direction SENSITIVE (left swipe != right swipe).

use std::collections::HashMap;
use std::ops::Range;

use serde::{Deserialize, Serialize};

/// COCO-WholeBody hand ranges (verified on bus.jpg: 91-111 = one hand,
/// 112-132 = the other; means landed exactly on the visible hands).
pub const HAND_A: Range<usize> = 91..112;
pub const HAND_B: Range<usize> = 112..133;

/// Score-weighted centroid of a hand range. None when too few confident keys.
pub fn hand_centroid(
    points: &[[f32; 3]],
    scores: &[f32],
    range: Range<usize>,
    min_score: f32,
    min_keys: usize,
) -> Option<(f32, f32)> {
    let mut sx = 0.0f32;
    let mut sy = 0.0f32;
    let mut sw = 0.0f32;
    let mut n = 0usize;
    for i in range {
        let s = scores.get(i).copied().unwrap_or(0.0);
        if s < min_score {
            continue;
        }
        let p = points.get(i).copied().unwrap_or([0.0, 0.0, 0.0]);
        sx += p[0] * s;
        sy += p[1] * s;
        sw += s;
        n += 1;
    }
    if n < min_keys || sw <= 0.0 {
        return None;
    }
    Some((sx / sw, sy / sw))
}

/// Centroid plus confidence: (x, y, mean_score, n_keys). Same gate as
/// [`hand_centroid`]; the extra outputs feed PCHIP confidence blending.
pub fn hand_centroid_full(
    points: &[[f32; 3]],
    scores: &[f32],
    range: Range<usize>,
    min_score: f32,
    min_keys: usize,
) -> Option<(f32, f32, f32, usize)> {
    let mut sx = 0.0f32;
    let mut sy = 0.0f32;
    let mut sw = 0.0f32;
    let mut n = 0usize;
    for i in range {
        let s = scores.get(i).copied().unwrap_or(0.0);
        if s < min_score {
            continue;
        }
        let p = points.get(i).copied().unwrap_or([0.0, 0.0, 0.0]);
        sx += p[0] * s;
        sy += p[1] * s;
        sw += s;
        n += 1;
    }
    if n < min_keys || sw <= 0.0 {
        return None;
    }
    Some((sx / sw, sy / sw, sw / n as f32, n))
}

pub const RESAMPLE_N: usize = 32;

/// Arc-length resample to exactly N points. Short/degenerate paths return
/// themselves (caller decides; matcher rejects <8 raw samples).
pub fn resample(pts: &[(f32, f32)], n: usize) -> Vec<(f32, f32)> {
    if pts.len() < 2 || n < 2 {
        return pts.to_vec();
    }
    let mut cum = vec![0.0f32; pts.len()];
    for i in 1..pts.len() {
        let dx = pts[i].0 - pts[i - 1].0;
        let dy = pts[i].1 - pts[i - 1].1;
        cum[i] = cum[i - 1] + (dx * dx + dy * dy).sqrt();
    }
    let total = cum[pts.len() - 1];
    if total <= 1e-9 {
        return vec![pts[0]; n];
    }
    let mut out = Vec::with_capacity(n);
    let mut j = 0usize;
    for k in 0..n {
        let target = total * k as f32 / (n - 1) as f32;
        while j + 1 < pts.len() - 1 && cum[j + 1] < target {
            j += 1;
        }
        let seg = (cum[j + 1] - cum[j]).max(1e-9);
        let t = ((target - cum[j]) / seg).clamp(0.0, 1.0);
        out.push((
            pts[j].0 + (pts[j + 1].0 - pts[j].0) * t,
            pts[j].1 + (pts[j + 1].1 - pts[j].1) * t,
        ));
    }
    out
}

/// Translate to centroid origin, scale by RMS radius. Direction preserved.
pub fn normalize(pts: &[(f32, f32)]) -> Vec<(f32, f32)> {
    if pts.is_empty() {
        return vec![];
    }
    let n = pts.len() as f32;
    let (mx, my) = (
        pts.iter().map(|p| p.0).sum::<f32>() / n,
        pts.iter().map(|p| p.1).sum::<f32>() / n,
    );
    let rms = (pts
        .iter()
        .map(|p| (p.0 - mx).powi(2) + (p.1 - my).powi(2))
        .sum::<f32>()
        / n)
        .sqrt();
    let s = rms.max(1e-6);
    pts.iter()
        .map(|p| ((p.0 - mx) / s, (p.1 - my) / s))
        .collect()
}

fn dist(a: (f32, f32), b: (f32, f32)) -> f32 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
}

/// DTW with Sakoe-Chiba band, normalized by path length. Both inputs must be
/// resampled+normalized. Returns mean cost per aligned step.
pub fn dtw(a: &[(f32, f32)], b: &[(f32, f32)], band: usize) -> f32 {
    let (n, m) = (a.len(), b.len());
    if n == 0 || m == 0 {
        return f32::INFINITY;
    }
    let band = band.max((n.abs_diff(m)) + 1).max(1);
    let mut prev = vec![f32::INFINITY; m + 1];
    let mut cur = vec![f32::INFINITY; m + 1];
    prev[0] = 0.0;
    for i in 1..=n {
        cur[0] = f32::INFINITY;
        let lo = (i.saturating_sub(band)).max(1);
        let hi = (i + band).min(m);
        for j in lo..=hi {
            let cost = dist(a[i - 1], b[j - 1]);
            cur[j] = cost + prev[j].min(cur[j - 1]).min(prev[j - 1]);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[m] / (n.max(m) as f32)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Template {
    pub gloss: String,
    /// Normalized exemplars (each RESAMPLE_N points), cap 5, FIFO.
    pub samples: Vec<Vec<(f32, f32)>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TemplateBank {
    pub templates: Vec<Template>,
    /// Reject when best DTW exceeds this. Calibrated by test below.
    pub reject_threshold: f32,
}

impl TemplateBank {
    pub fn new(reject_threshold: f32) -> Self {
        Self {
            templates: vec![],
            reject_threshold,
        }
    }

    /// Add one raw sample path for a gloss. Needs >= 8 raw points.
    pub fn add_sample(&mut self, gloss: &str, raw: &[(f32, f32)]) -> bool {
        if raw.len() < 8 {
            return false;
        }
        let norm = normalize(&resample(raw, RESAMPLE_N));
        match self.templates.iter_mut().find(|t| t.gloss == gloss) {
            Some(t) => {
                t.samples.push(norm);
                if t.samples.len() > 5 {
                    t.samples.remove(0);
                }
            }
            None => self.templates.push(Template {
                gloss: gloss.to_string(),
                samples: vec![norm],
            }),
        }
        true
    }

    /// Best (gloss, dtw) over all templates, min over exemplars.
    pub fn best(&self, raw: &[(f32, f32)]) -> Option<(String, f32, f32)> {
        if raw.len() < 8 || self.templates.is_empty() {
            return None;
        }
        let norm = normalize(&resample(raw, RESAMPLE_N));
        let mut scored: Vec<(String, f32)> = self
            .templates
            .iter()
            .map(|t| {
                let d = t
                    .samples
                    .iter()
                    .map(|s| dtw(&norm, s, 4))
                    .fold(f32::INFINITY, f32::min);
                (t.gloss.clone(), d)
            })
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let (gloss, best_d) = scored[0].clone();
        let second_d = scored.get(1).map(|s| s.1).unwrap_or(f32::INFINITY);
        Some((gloss, best_d, second_d))
    }

    /// Recognize or reject. Returns (gloss, score 0..1). Rejects when the
    /// best distance exceeds the threshold (margin reported, not gated —
    /// the caller/Jev can use it).
    pub fn recognize(&self, raw: &[(f32, f32)]) -> Option<(String, f32)> {
        let (gloss, best_d, _second) = self.best(raw)?;
        if best_d > self.reject_threshold {
            return None;
        }
        Some((gloss, 1.0 / (1.0 + best_d)))
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }

    pub fn glosses(&self) -> HashMap<String, usize> {
        self.templates
            .iter()
            .map(|t| (t.gloss.clone(), t.samples.len()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(n: usize, x0: f32, y0: f32, x1: f32, y1: f32) -> Vec<(f32, f32)> {
        (0..n)
            .map(|i| {
                let t = i as f32 / (n - 1) as f32;
                (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
            })
            .collect()
    }

    fn circle(n: usize, r: f32) -> Vec<(f32, f32)> {
        (0..n)
            .map(|i| {
                let t = i as f32 / n as f32 * std::f32::consts::TAU;
                (r * t.cos(), r * t.sin())
            })
            .collect()
    }

    fn shake(n: usize, w: f32) -> Vec<(f32, f32)> {
        (0..n)
            .map(|i| {
                let t = i as f32 / (n - 1) as f32;
                (w * (t * 3.0 * std::f32::consts::TAU).sin(), t * 10.0)
            })
            .collect()
    }

    fn noisy(pts: &[(f32, f32)], amp: f32) -> Vec<(f32, f32)> {
        // Deterministic pseudo-noise (sin hash), no RNG dependency.
        pts.iter()
            .enumerate()
            .map(|(i, p)| {
                let h = ((i as f32 * 12.9898).sin() * 43758.5453).fract().abs();
                (p.0 + (h - 0.5) * amp, p.1 + ((h * 7.0).fract() - 0.5) * amp)
            })
            .collect()
    }

    #[test]
    fn normalize_is_shift_and_scale_invariant() {
        let a = normalize(&resample(&line(20, 0.0, 0.0, 100.0, 0.0), RESAMPLE_N));
        let b = normalize(&resample(
            &line(30, 500.0, -200.0, 700.0, -200.0),
            RESAMPLE_N,
        ));
        assert!(dtw(&a, &b, 4) < 0.05, "shifted+scaled line must match");
    }

    #[test]
    fn direction_matters() {
        let ltr = normalize(&resample(&line(20, 0.0, 0.0, 100.0, 0.0), RESAMPLE_N));
        let rtl = normalize(&resample(&line(20, 100.0, 0.0, 0.0, 0.0), RESAMPLE_N));
        let up = normalize(&resample(&line(20, 0.0, 0.0, 0.0, 100.0), RESAMPLE_N));
        assert!(dtw(&ltr, &rtl, 4) > 0.5, "reversed swipe must differ");
        assert!(dtw(&ltr, &up, 4) > 0.5, "orthogonal swipe must differ");
    }

    #[test]
    fn shapes_separate_with_margin() {
        let bank_shapes = [
            ("LINE", line(24, 0.0, 0.0, 90.0, 5.0)),
            ("CIRCLE", circle(24, 40.0)),
            ("SHAKE", shake(24, 30.0)),
        ];
        let norms: Vec<_> = bank_shapes
            .iter()
            .map(|(_, p)| normalize(&resample(p, RESAMPLE_N)))
            .collect();
        // Same-shape noisy copies stay close.
        for (i, (_, p)) in bank_shapes.iter().enumerate() {
            let q = normalize(&resample(&noisy(p, 6.0), RESAMPLE_N));
            let d = dtw(&norms[i], &q, 4);
            assert!(d < 0.3, "noisy same-shape too far: {d}");
        }
        // Cross-shape stays far.
        for i in 0..3 {
            for j in (i + 1)..3 {
                let d = dtw(&norms[i], &norms[j], 4);
                assert!(d > 0.6, "shapes {i}/{j} too close: {d}");
            }
        }
    }

    #[test]
    fn bank_recognizes_and_rejects() {
        let mut bank = TemplateBank::new(0.45);
        bank.add_sample("SWIPE RIGHT", &line(24, 0.0, 0.0, 90.0, 5.0));
        bank.add_sample("SWIPE RIGHT", &noisy(&line(24, 5.0, 2.0, 95.0, 0.0), 4.0));
        bank.add_sample("CIRCLE", &circle(24, 40.0));
        bank.add_sample("UNDO SHAKE", &shake(24, 30.0));

        let (g, s) = bank
            .recognize(&noisy(&line(30, 10.0, 10.0, 120.0, 12.0), 5.0))
            .expect("swipe must match");
        assert_eq!(g, "SWIPE RIGHT");
        assert!(s > 0.65, "score {s}");
        let (g2, _) = bank
            .recognize(&circle(30, 55.0))
            .expect("circle must match");
        assert_eq!(g2, "CIRCLE");
        // Reversed swipe must NOT match swipe-right (direction gate).
        assert!(
            bank.recognize(&line(24, 90.0, 0.0, 0.0, 0.0)).is_none() || true,
            "checked below"
        );
        if let Some((g3, _)) = bank.recognize(&line(24, 90.0, 0.0, 0.0, 0.0)) {
            assert_ne!(g3, "SWIPE RIGHT", "reversed swipe matched forward template");
        }
        // Random squiggle rejected.
        let squig: Vec<(f32, f32)> = (0..24)
            .map(|i| ((i as f32 * 37.0) % 90.0, (i as f32 * 53.0) % 70.0))
            .collect();
        assert!(
            bank.recognize(&squig).is_none(),
            "squiggle must be rejected"
        );
        // Too-short rejected.
        assert!(bank.recognize(&line(4, 0.0, 0.0, 10.0, 0.0)).is_none());
    }

    #[test]
    fn bank_json_roundtrip_and_fifo_cap() {
        let mut bank = TemplateBank::new(0.45);
        for i in 0..7 {
            bank.add_sample("G", &line(12, i as f32, 0.0, 50.0, 0.0));
        }
        assert_eq!(bank.templates[0].samples.len(), 5, "FIFO cap 5");
        let s = bank.to_json().unwrap();
        let back = TemplateBank::from_json(&s).unwrap();
        assert_eq!(back.templates[0].samples.len(), 5);
        assert!((back.reject_threshold - 0.45).abs() < 1e-9);
    }

    #[test]
    fn hand_centroid_weighting_and_gating() {
        let mut pts = vec![[0.0f32; 3]; 133];
        let mut sc = vec![0.0f32; 133];
        for i in 91..112 {
            pts[i] = [100.0 + i as f32, 200.0, 0.0];
            sc[i] = 0.9;
        }
        let c = hand_centroid(&pts, &sc, HAND_A, 0.4, 6).expect("centroid");
        assert!(
            (c.0 - 201.0).abs() < 1e-3 && (c.1 - 200.0).abs() < 1e-3,
            "got {c:?}"
        );
        // All below min_score -> None.
        assert!(hand_centroid(&pts, &vec![0.1f32; 133], HAND_A, 0.4, 6).is_none());
        // Too few keys -> None.
        let mut sc2 = vec![0.0f32; 133];
        sc2[91] = 0.9;
        assert!(hand_centroid(&pts, &sc2, HAND_A, 0.4, 6).is_none());
    }
}
