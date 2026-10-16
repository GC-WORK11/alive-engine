//! ONNX whole-body backend: YOLOX-m person detector + RTMW-l 133-keypoint pose.
//!
//! Preprocessing matches the exported `pipeline.json` ground truth exactly:
//! - Pose: TopDown center/scale (padding 1.25) -> affine warp to 192x256 ->
//!   RGB normalize (123.675/116.28/103.53, 58.395/57.12/57.375). Post: SimCC
//!   split_ratio 2.0, no DARK -> coord = argmax/2, softmax max as score.
//! - Detector: letterbox to 640 (pad 114, keep ratio) in BGR order, raw
//!   0-255 floats. Outputs dets [1,N,5] + labels [1,N], NMS already in-graph.
//!
//! Latency policy (measured on this machine: pose ~67ms, detect ~250ms CPU):
//! pose runs in the foreground on a tracked bbox; detection refreshes on
//! track start, on loss, and on explicit request (the app moves refresh to a
//! background thread so the pose loop never hitches).

use std::path::Path;

use super::{Frame, LandmarkExtractor, Landmarks, PerceptionError};

pub const RTMW_W: u32 = 192;
pub const RTMW_H: u32 = 256;
pub const RTMW_K: usize = 133;
pub const RTMW_XBINS: usize = 384;
pub const RTMW_YBINS: usize = 512;
pub const DET_SIZE: u32 = 640;
pub const BBOX_PADDING: f32 = 1.25;

const MEAN: [f32; 3] = [123.675, 116.28, 103.53];
const STD: [f32; 3] = [58.395, 57.12, 57.375];

#[derive(Debug, Clone, Copy)]
pub struct Bbox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub score: f32,
}

impl Bbox {
    pub fn full_frame(w: u32, h: u32) -> Self {
        Self {
            x1: 0.0,
            y1: 0.0,
            x2: w as f32,
            y2: h as f32,
            score: 0.0,
        }
    }
    pub fn width(&self) -> f32 {
        (self.x2 - self.x1).max(1.0)
    }
    pub fn height(&self) -> f32 {
        (self.y2 - self.y1).max(1.0)
    }
    pub fn clamp(&self, w: u32, h: u32) -> Self {
        Self {
            x1: self.x1.clamp(0.0, w as f32),
            y1: self.y1.clamp(0.0, h as f32),
            x2: self.x2.clamp(0.0, w as f32),
            y2: self.y2.clamp(0.0, h as f32),
            score: self.score,
        }
    }
}

/// Rot-0 top-down affine (mmpose `get_affine_transform`, border 0).
/// Uniform scale referenced to the bbox WIDTH (mmpose uses src_w for both
/// axes), so tall boxes gain vertical context instead of being squeezed.
#[derive(Debug, Clone, Copy)]
pub struct CropAffine {
    pub sx: f32,
    pub sy: f32,
    pub tx: f32,
    pub ty: f32,
}

impl CropAffine {
    pub fn from_bbox(b: &Bbox, out_w: u32, out_h: u32) -> Self {
        let cx = (b.x1 + b.x2) * 0.5;
        let cy = (b.y1 + b.y2) * 0.5;
        let s = out_w as f32 / (b.width() * BBOX_PADDING);
        Self {
            sx: s,
            sy: s,
            tx: out_w as f32 * 0.5 - s * cx,
            ty: out_h as f32 * 0.5 - s * cy,
        }
    }

    pub fn to_crop(&self, x: f32, y: f32) -> (f32, f32) {
        (self.sx * x + self.tx, self.sy * y + self.ty)
    }

    pub fn to_frame(&self, x: f32, y: f32) -> (f32, f32) {
        ((x - self.tx) / self.sx, (y - self.ty) / self.sy)
    }
}

fn bilinear_rgb(frame: &Frame, x: f32, y: f32) -> [f32; 3] {
    let (w, h) = (frame.width as i64, frame.height as i64);
    let x0 = x.floor() as i64;
    let y0 = y.floor() as i64;
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let mut out = [0.0f32; 3];
    for (dx, wx) in [(0, 1.0 - fx), (1, fx)] {
        for (dy, wy) in [(0, 1.0 - fy), (1, fy)] {
            let px = x0 + dx;
            let py = y0 + dy;
            if px < 0 || py < 0 || px >= w || py >= h {
                continue; // border 0, like cv2 warpAffine default
            }
            let o = ((py as usize) * (frame.width as usize) + (px as usize)) * 3;
            let wt = wx * wy;
            out[0] += frame.data[o] as f32 * wt;
            out[1] += frame.data[o + 1] as f32 * wt;
            out[2] += frame.data[o + 2] as f32 * wt;
        }
    }
    out
}

/// Warp + normalize a bbox crop into CHW float32 (RGB order).
/// Writes into caller scratch (hot path: zero alloc). Overwrites every
/// element, so dirty scratch is fine.
pub fn warp_normalize_pose_into(frame: &Frame, affine: &CropAffine, chw: &mut [f32]) {
    debug_assert_eq!(chw.len(), 3 * RTMW_W as usize * RTMW_H as usize);
    let (w, h) = (RTMW_W as usize, RTMW_H as usize);
    for dy in 0..h {
        for dx in 0..w {
            let sx = (dx as f32 - affine.tx) / affine.sx;
            let sy = (dy as f32 - affine.ty) / affine.sy;
            let px = bilinear_rgb(frame, sx, sy);
            for c in 0..3 {
                chw[c * w * h + dy * w + dx] = (px[c] - MEAN[c]) / STD[c];
            }
        }
    }
}

/// Allocating wrapper (benches, probes, tests).
pub fn warp_normalize_pose(frame: &Frame, affine: &CropAffine) -> Vec<f32> {
    let mut v = vec![0.0f32; 3 * RTMW_W as usize * RTMW_H as usize];
    warp_normalize_pose_into(frame, affine, &mut v);
    v
}

pub struct Letterbox {
    pub scale: f32,
    pub new_w: u32,
    pub new_h: u32,
}

impl Letterbox {
    pub fn for_frame(w: u32, h: u32) -> Self {
        let scale = (DET_SIZE as f32 / w as f32).min(DET_SIZE as f32 / h as f32);
        Self {
            scale,
            new_w: (w as f32 * scale).round() as u32,
            new_h: (h as f32 * scale).round() as u32,
        }
    }

    pub fn to_frame(&self, x: f32, y: f32) -> (f32, f32) {
        (x / self.scale, y / self.scale)
    }
}

/// Letterbox + BGR interleave into CHW float32, raw 0-255 (mean 0 / std 1).
/// Writes into caller scratch (hot path: zero alloc). Pre-fills pad 114.
pub fn letterbox_bgr_into(frame: &Frame, lb: &Letterbox, chw: &mut [f32]) {
    let s = DET_SIZE as usize;
    debug_assert_eq!(chw.len(), 3 * s * s);
    chw.fill(114.0f32);
    let (nw, nh) = (lb.new_w as usize, lb.new_h as usize);
    for dy in 0..nh {
        for dx in 0..nw {
            let sx = dx as f32 / lb.scale;
            let sy = dy as f32 / lb.scale;
            let px = bilinear_rgb(frame, sx, sy);
            // BGR order (detector pipeline: to_rgb=false).
            chw[0 * s * s + dy * s + dx] = px[2];
            chw[1 * s * s + dy * s + dx] = px[1];
            chw[2 * s * s + dy * s + dx] = px[0];
        }
    }
}

/// Allocating wrapper (benches, probes, tests).
pub fn letterbox_bgr(frame: &Frame, lb: &Letterbox) -> Vec<f32> {
    let s = DET_SIZE as usize;
    let mut v = vec![0.0f32; 3 * s * s];
    letterbox_bgr_into(frame, lb, &mut v);
    v
}

/// SimCC decode exactly like mmpose `SimCCLabelDecode` (no DARK):
/// coord = argmax/2 per axis, score = raw max over both axes (NO softmax —
/// the exported logits are peaky but low-margin, and softmax-max crushes
/// them to ~1/bins; verified visually on bus.jpg).
/// `simcc_x` is [K*XBINS] row-major, `simcc_y` is [K*YBINS] row-major.
pub fn simcc_decode(
    simcc_x: &[f32],
    simcc_y: &[f32],
    k: usize,
) -> Result<(Vec<(f32, f32)>, Vec<f32>), PerceptionError> {
    if simcc_x.len() != k * RTMW_XBINS {
        return Err(PerceptionError::Backend(format!(
            "simcc_x len {} != {k}*{RTMW_XBINS}",
            simcc_x.len()
        )));
    }
    if simcc_y.len() != k * RTMW_YBINS {
        return Err(PerceptionError::Backend(format!(
            "simcc_y len {} != {k}*{RTMW_YBINS}",
            simcc_y.len()
        )));
    }
    let mut pts = Vec::with_capacity(k);
    let mut scores = Vec::with_capacity(k);
    simcc_decode_into(simcc_x, simcc_y, k, &mut pts, &mut scores)?;
    Ok((pts, scores))
}

/// SimCC decode into caller scratch (hot path: zero alloc).
/// `clear` + push reproduces identical contents; strict `>` keeps
/// first-max-wins tie behavior.
pub fn simcc_decode_into(
    simcc_x: &[f32],
    simcc_y: &[f32],
    k: usize,
    pts: &mut Vec<(f32, f32)>,
    scores: &mut Vec<f32>,
) -> Result<(), PerceptionError> {
    if simcc_x.len() != k * RTMW_XBINS {
        return Err(PerceptionError::Backend(format!(
            "simcc_x len {} != {k}*{RTMW_XBINS}",
            simcc_x.len()
        )));
    }
    if simcc_y.len() != k * RTMW_YBINS {
        return Err(PerceptionError::Backend(format!(
            "simcc_y len {} != {k}*{RTMW_YBINS}",
            simcc_y.len()
        )));
    }
    pts.clear();
    scores.clear();
    pts.reserve(k);
    scores.reserve(k);
    for i in 0..k {
        let (mut bx, mut mx) = (0usize, f32::NEG_INFINITY);
        let row_x = &simcc_x[i * RTMW_XBINS..(i + 1) * RTMW_XBINS];
        for (b, &v) in row_x.iter().enumerate() {
            if v > mx {
                mx = v;
                bx = b;
            }
        }
        let (mut by, mut my) = (0usize, f32::NEG_INFINITY);
        let row_y = &simcc_y[i * RTMW_YBINS..(i + 1) * RTMW_YBINS];
        for (b, &v) in row_y.iter().enumerate() {
            if v > my {
                my = v;
                by = b;
            }
        }
        pts.push((bx as f32 / 2.0, by as f32 / 2.0));
        scores.push(mx.max(my));
    }
    Ok(())
}

fn ort_err(e: impl std::fmt::Display) -> PerceptionError {
    PerceptionError::Backend(e.to_string())
}

pub struct RtmwPose {
    session: ort::session::Session,
    warp: Vec<f32>,
    crop_pts: Vec<(f32, f32)>,
}

impl RtmwPose {
    pub fn load(path: &Path, intra_threads: usize) -> Result<Self, PerceptionError> {
        let session = ort::session::Session::builder()
            .map_err(ort_err)?
            .with_intra_threads(intra_threads.max(1))
            .map_err(ort_err)?
            .commit_from_file(path)
            .map_err(ort_err)?;
        Ok(Self {
            session,
            // Pre-sized once: hot path never allocates the big buffers.
            warp: vec![0.0f32; 3 * RTMW_W as usize * RTMW_H as usize],
            crop_pts: Vec::with_capacity(RTMW_K),
        })
    }

    /// Pose on a bbox; returns frame-space keypoints + scores.
    pub fn infer_bbox(&mut self, frame: &Frame, bbox: &Bbox) -> Result<Landmarks, PerceptionError> {
        let affine = CropAffine::from_bbox(bbox, RTMW_W, RTMW_H);
        warp_normalize_pose_into(frame, &affine, &mut self.warp);
        let input = ort::value::TensorRef::from_array_view((
            [1usize, 3, RTMW_H as usize, RTMW_W as usize],
            &self.warp[..],
        ))
        .map_err(ort_err)?;
        let outputs = self.session.run(ort::inputs![input]).map_err(ort_err)?;
        let (_, sx) = outputs["simcc_x"]
            .try_extract_tensor::<f32>()
            .map_err(ort_err)?;
        let (_, sy) = outputs["simcc_y"]
            .try_extract_tensor::<f32>()
            .map_err(ort_err)?;
        // Scores move into Landmarks (owned); only the transient crop points
        // are pooled. 532B fresh per infer is malloc-trivial, never mmap.
        let mut scores = Vec::with_capacity(RTMW_K);
        simcc_decode_into(sx, sy, RTMW_K, &mut self.crop_pts, &mut scores)?;
        let mut points = Vec::with_capacity(RTMW_K);
        for (cx, cy) in &self.crop_pts {
            let (fx, fy) = affine.to_frame(*cx, *cy);
            points.push([fx, fy, 0.0]);
        }
        // Presence = at least 12 keypoints with raw-max score >= 0.4.
        // (Raw-max, not softmax: verified on bus.jpg; threshold calibrated
        // against empty-room negatives which peak ~0.1.)
        let confident = scores.iter().filter(|&&s| s >= 0.4).count();
        Ok(Landmarks {
            at_ms: frame.at_capture_ms,
            present: confident >= 12,
            points,
            scores,
        })
    }
}

pub struct YoloxPerson {
    session: ort::session::Session,
    det: Vec<f32>,
}

impl YoloxPerson {
    pub fn load(path: &Path, intra_threads: usize) -> Result<Self, PerceptionError> {
        let session = ort::session::Session::builder()
            .map_err(ort_err)?
            .with_intra_threads(intra_threads.max(1))
            .map_err(ort_err)?
            .commit_from_file(path)
            .map_err(ort_err)?;
        let s = DET_SIZE as usize;
        Ok(Self {
            session,
            det: vec![0.0f32; 3 * s * s],
        })
    }

    /// Person boxes in frame coords, largest-first. Empty = nobody found.
    pub fn detect(&mut self, frame: &Frame, score_thr: f32) -> Result<Vec<Bbox>, PerceptionError> {
        let lb = Letterbox::for_frame(frame.width, frame.height);
        letterbox_bgr_into(frame, &lb, &mut self.det);
        let s = DET_SIZE as usize;
        let input = ort::value::TensorRef::from_array_view(([1usize, 3, s, s], &self.det[..]))
            .map_err(ort_err)?;
        let outputs = self.session.run(ort::inputs![input]).map_err(ort_err)?;
        let (_, dets) = outputs["dets"]
            .try_extract_tensor::<f32>()
            .map_err(ort_err)?;
        if dets.len() % 5 != 0 {
            return Err(PerceptionError::Backend(format!(
                "dets len {} not x5",
                dets.len()
            )));
        }
        let mut boxes = Vec::new();
        for d in dets.chunks_exact(5) {
            let score = d[4];
            if score < score_thr {
                continue;
            }
            let (x1, y1) = lb.to_frame(d[0], d[1]);
            let (x2, y2) = lb.to_frame(d[2], d[3]);
            boxes.push(
                Bbox {
                    x1,
                    y1,
                    x2: x2.max(x1 + 1.0),
                    y2: y2.max(y1 + 1.0),
                    score,
                }
                .clamp(frame.width, frame.height),
            );
        }
        boxes.sort_by(|a, b| {
            (b.width() * b.height())
                .partial_cmp(&(a.width() * a.height()))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(boxes)
    }
}

/// Foreground tracker: pose every call on the tracked bbox; detection only
/// on `refresh_detection` (frame 0, on loss, or app-driven background tick).
pub struct WholeBodyTracker {
    pose: RtmwPose,
    det: YoloxPerson,
    bbox: Option<Bbox>,
    pub score_thr: f32,
}

impl WholeBodyTracker {
    pub fn load(models_dir: &Path, intra_threads: usize) -> Result<Self, PerceptionError> {
        let pose = RtmwPose::load(&models_dir.join("rtmw-l-256x192.onnx"), intra_threads)?;
        let det = YoloxPerson::load(&models_dir.join("yolox-m-humanart.onnx"), intra_threads)?;
        Ok(Self {
            pose,
            det,
            bbox: None,
            score_thr: 0.5,
        })
    }

    pub fn refresh_detection(&mut self, frame: &Frame) -> Result<Vec<Bbox>, PerceptionError> {
        let boxes = self.det.detect(frame, self.score_thr)?;
        if let Some(best) = boxes.first() {
            self.bbox = Some(*best);
        }
        Ok(boxes)
    }

    pub fn bbox(&self) -> Option<Bbox> {
        self.bbox
    }

    pub fn set_bbox(&mut self, b: Option<Bbox>) {
        self.bbox = b;
    }
}

impl LandmarkExtractor for WholeBodyTracker {
    fn infer(&mut self, frame: &Frame) -> Result<Landmarks, PerceptionError> {
        // No track, no pose: running RTMW on a full empty frame burns 64ms
        // and hallucinates mid-score scatter (verified on empty wall).
        // The app refreshes detection on track start, on loss, and every N
        // frames from a background thread.
        let Some(bbox) = self.bbox else {
            return Ok(Landmarks {
                at_ms: frame.at_capture_ms,
                present: false,
                points: vec![],
                scores: vec![],
            });
        };
        let lm = self.pose.infer_bbox(frame, &bbox)?;
        if !lm.present {
            // Track lost: clear so the app re-detects instead of re-posing.
            self.bbox = None;
        }
        Ok(lm)
    }

    fn keypoints(&self) -> usize {
        RTMW_K
    }

    fn name(&self) -> &'static str {
        "rtmw-l-wholebody"
    }

    fn rescan(&mut self, frame: &Frame) {
        let _ = self.refresh_detection(frame);
    }

    fn tracking(&self) -> bool {
        self.bbox.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::super::CaptureSource;
    use super::super::LandmarkExtractor;
    use super::super::SyntheticSource;
    use super::*;

    #[test]
    fn affine_roundtrips() {
        let b = Bbox {
            x1: 100.0,
            y1: 50.0,
            x2: 500.0,
            y2: 650.0,
            score: 1.0,
        };
        let a = CropAffine::from_bbox(&b, RTMW_W, RTMW_H);
        for (x, y) in [(100.0, 50.0), (300.0, 350.0), (500.0, 650.0), (0.0, 0.0)] {
            let (cx, cy) = a.to_crop(x, y);
            let (rx, ry) = a.to_frame(cx, cy);
            assert!(
                (rx - x).abs() < 1e-3 && (ry - y).abs() < 1e-3,
                "roundtrip {x},{y}"
            );
        }
        // Bbox center maps to crop center.
        let (cx, cy) = a.to_crop(300.0, 350.0);
        assert!((cx - 96.0).abs() < 1e-3 && (cy - 128.0).abs() < 1e-3);
    }

    #[test]
    fn simcc_decode_peak_math() {
        let k = 2;
        let mut x = vec![-10.0f32; k * RTMW_XBINS];
        let mut y = vec![-10.0f32; k * RTMW_YBINS];
        x[100] = 10.0; // -> 50.0
        y[200] = 10.0; // -> 100.0
        x[RTMW_XBINS + 10] = 5.0;
        y[RTMW_YBINS + 20] = 5.0;
        let (pts, scores) = simcc_decode(&x, &y, k).unwrap();
        assert!((pts[0].0 - 50.0).abs() < 1e-6);
        assert!((pts[0].1 - 100.0).abs() < 1e-6);
        // Raw-max scoring (mmpose): score == peak logit, max over both axes.
        assert!((scores[0] - 10.0).abs() < 1e-6, "got {}", scores[0]);
        assert!((scores[1] - 5.0).abs() < 1e-6, "got {}", scores[1]);
        assert!(simcc_decode(&x[..10], &y, k).is_err(), "bad len must error");
    }

    #[test]
    fn letterbox_math_720p() {
        let lb = Letterbox::for_frame(1280, 720);
        assert!((lb.scale - 0.5).abs() < 1e-6);
        assert_eq!((lb.new_w, lb.new_h), (640, 360));
        let (x, y) = lb.to_frame(320.0, 180.0);
        assert!((x - 640.0).abs() < 1e-3 && (y - 360.0).abs() < 1e-3);
    }

    #[test]
    fn preprocess_shapes() {
        let mut s = SyntheticSource::new(320, 240, Some(1));
        let f = s.next_frame().unwrap().unwrap();
        let a = CropAffine::from_bbox(&Bbox::full_frame(320, 240), RTMW_W, RTMW_H);
        let chw = warp_normalize_pose(&f, &a);
        assert_eq!(chw.len(), 3 * 192 * 256);
        assert!(chw.iter().all(|v| v.is_finite()));
        let lb = Letterbox::for_frame(320, 240);
        let det = letterbox_bgr(&f, &lb);
        assert_eq!(det.len(), 3 * 640 * 640);
        // Padded corner must be exactly 114 (pad val).
        let o = 2 * 640 * 640 + 639 * 640 + 639;
        assert!((det[o] - 114.0).abs() < 1e-6);
    }

    fn synth_frame(w: u32, h: u32, seed: u64) -> super::super::Frame {
        SyntheticSource::new(w, h, Some(seed))
            .next_frame()
            .unwrap()
            .unwrap()
    }

    fn assert_bits_eq(a: &[f32], b: &[f32], what: &str) {
        assert_eq!(a.len(), b.len(), "{what}: len");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{what}: bit mismatch at {i}: {x} vs {y}"
            );
        }
    }

    /// Scratch-reuse must not change numerics by even 1 ULP: fresh-alloc
    /// vs `_into` on dirty scratch (run twice to prove full overwrite).
    #[test]
    fn preprocess_bitidentical() {
        let bboxes = [
            Bbox::full_frame(640, 480),
            Bbox {
                x1: 300.0,
                y1: 10.0,
                x2: 340.0,
                y2: 470.0,
                score: 1.0,
            }, // tall
            Bbox {
                x1: -50.0,
                y1: -30.0,
                x2: 200.0,
                y2: 500.0,
                score: 1.0,
            }, // edge-clamped
        ];
        for (fi, seed) in [1u64, 7, 42].iter().enumerate() {
            let f = synth_frame(640, 480, *seed);
            for (bi, b) in bboxes.iter().enumerate() {
                let a = CropAffine::from_bbox(b, RTMW_W, RTMW_H);
                let fresh = warp_normalize_pose(&f, &a);
                // Dirty scratch with hostile bits, written twice.
                let mut scratch = vec![-12345.678f32; fresh.len()];
                warp_normalize_pose_into(&f, &a, &mut scratch);
                warp_normalize_pose_into(&f, &a, &mut scratch);
                assert_bits_eq(&fresh, &scratch, &format!("warp f{fi} b{bi}"));
            }
            // Letterbox across aspects; pad bits exactly 114.0.
            for (w, h) in [(640u32, 480u32), (1280, 720), (480, 640)] {
                let g = synth_frame(w, h, *seed);
                let lb = Letterbox::for_frame(w, h);
                let fresh = letterbox_bgr(&g, &lb);
                let mut scratch = vec![999.0f32; fresh.len()];
                letterbox_bgr_into(&g, &lb, &mut scratch);
                letterbox_bgr_into(&g, &lb, &mut scratch);
                assert_bits_eq(&fresh, &scratch, &format!("letterbox {w}x{h}"));
                let s = DET_SIZE as usize;
                let o = 2 * s * s + (s - 1) * s + (s - 1);
                assert_eq!(scratch[o].to_bits(), 114.0f32.to_bits(), "pad bits");
            }
        }
        // SimCC: ties keep first-max-wins in both paths.
        let k = 3usize;
        let mut x = vec![-5.0f32; k * RTMW_XBINS];
        let mut y = vec![-5.0f32; k * RTMW_YBINS];
        x[10] = 7.0;
        x[11] = 7.0; // tie: index 10 must win
        y[20] = 7.0;
        y[21] = 7.0;
        let (p0, s0) = simcc_decode(&x, &y, k).unwrap();
        let mut p1 = vec![(1.0f32, 2.0f32); 99]; // dirty, wrong len
        let mut s1 = vec![3.0f32; 99];
        simcc_decode_into(&x, &y, k, &mut p1, &mut s1).unwrap();
        simcc_decode_into(&x, &y, k, &mut p1, &mut s1).unwrap();
        assert_eq!(p0, p1);
        assert_eq!(s0, s1);
        assert!((p0[0].0 - 5.0).abs() < 1e-9, "tie must pick first: {p0:?}");
    }

    /// End-to-end determinism: same frame + threads=1 → bit-identical
    /// landmarks across runs. (ORT >1 thread may vary ±1 ULP in parallel
    /// reductions, so the gate pins threads=1.)
    #[test]
    fn infer_deterministic_threads1() {
        let dir = std::path::Path::new("../../models");
        if !dir.join("rtmw-l-256x192.onnx").exists() {
            eprintln!("SKIP infer_deterministic_threads1: models/ missing");
            return;
        }
        let f = synth_frame(640, 480, 3);
        let mut t = WholeBodyTracker::load(dir, 1).expect("load models");
        t.set_bbox(Some(Bbox::full_frame(640, 480)));
        let l1 = t.infer(&f).expect("infer 1");
        // Second tracker instance: fresh sessions, same math.
        let mut t2 = WholeBodyTracker::load(dir, 1).expect("load models");
        t2.set_bbox(Some(Bbox::full_frame(640, 480)));
        let l2 = t2.infer(&f).expect("infer 2");
        assert_eq!(l1.present, l2.present);
        assert_eq!(l1.points.len(), l2.points.len());
        for (i, (a, b)) in l1.points.iter().zip(l2.points.iter()).enumerate() {
            for c in 0..3 {
                assert_eq!(a[c].to_bits(), b[c].to_bits(), "point {i} ch {c}");
            }
        }
        for (i, (a, b)) in l1.scores.iter().zip(l2.scores.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "score {i}");
        }
    }
}
