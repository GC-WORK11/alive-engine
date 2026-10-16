//! alive-perception: capture sources, ROI ops, landmark extractor backends.
//!
//! Layers: `CaptureSource` (camera / image-dir / synthetic) produces RGB8
//! `Frame`s; `LandmarkExtractor` turns frames into keypoints. The ONNX
//! whole-body backend lands behind the `onnx` feature once the model hunt
//! returns pinned URLs; until then `NullExtractor` reports honestly.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, thiserror::Error)]
pub enum PerceptionError {
    #[error("camera: {0}")]
    Camera(String),
    #[error("decode: {0}")]
    Decode(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("backend: {0}")]
    Backend(String),
    #[error("unsupported: {0}")]
    Unsupported(String),
}

pub fn now_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Zero-pass adaptive gain (AGC): dark-room contrast for free.
// ---------------------------------------------------------------------------

/// Luma mean (G channel) the gain steers toward. Indoor rooms sit ~80-120,
/// so this is a no-op in normal light and a lift only in the dark.
pub const AGC_TARGET_MEAN: f32 = 96.0;
/// Hard ceiling: 3x lifts a mean-30 room to target without blowing noise.
pub const AGC_MAX_GAIN: f32 = 3.0;

/// Sample-time gain from the observed luma mean of the motion subsample.
/// 1.0 in normal light (preprocess stays bit-identical), up to 3x in the
/// dark. The mean comes from the motion pass's existing samples — zero
/// extra frame passes, ~0ms. Pure + unit-tested.
pub fn agc_gain(luma_mean: f32) -> f32 {
    if !luma_mean.is_finite() || luma_mean <= 0.0 {
        return 1.0;
    }
    (AGC_TARGET_MEAN / luma_mean).clamp(1.0, AGC_MAX_GAIN)
}

// ---------------------------------------------------------------------------
// Frames + sources
// ---------------------------------------------------------------------------

/// One RGB8 frame. `data.len() == width*height*3` is the invariant.
#[derive(Debug, Clone)]
pub struct Frame {
    pub seq: u64,
    pub at_capture_ms: u64,
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

impl Frame {
    pub fn rgb_image(&self) -> Option<image::RgbImage> {
        image::RgbImage::from_raw(self.width, self.height, self.data.clone())
    }
}

pub trait CaptureSource {
    fn next_frame(&mut self) -> Result<Option<Frame>, PerceptionError>;
    fn label(&self) -> String;
}

/// Deterministic synthetic frames (moving bar). For benches + tests.
pub struct SyntheticSource {
    width: u32,
    height: u32,
    seq: u64,
    max_frames: Option<u64>,
}

impl SyntheticSource {
    pub fn new(width: u32, height: u32, max_frames: Option<u64>) -> Self {
        Self {
            width: width.max(8),
            height: height.max(8),
            seq: 0,
            max_frames,
        }
    }
}

impl CaptureSource for SyntheticSource {
    fn next_frame(&mut self) -> Result<Option<Frame>, PerceptionError> {
        if let Some(max) = self.max_frames {
            if self.seq >= max {
                return Ok(None);
            }
        }
        let (w, h) = (self.width as usize, self.height as usize);
        let mut data = vec![0u8; w * h * 3];
        let bar_x = ((self.seq * 8) % self.width as u64) as usize;
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 3;
                let v = if x == bar_x {
                    255
                } else {
                    ((x + y) % 256) as u8
                };
                data[i] = v;
                data[i + 1] = (y % 256) as u8;
                data[i + 2] = (x % 256) as u8;
            }
        }
        let f = Frame {
            seq: self.seq,
            at_capture_ms: now_ms(),
            width: self.width,
            height: self.height,
            data,
        };
        self.seq += 1;
        Ok(Some(f))
    }

    fn label(&self) -> String {
        format!("synthetic {}x{}", self.width, self.height)
    }
}

/// Ordered image files from a directory (png/jpg). Sorted, loops once.
pub struct ImageDirSource {
    paths: Vec<PathBuf>,
    idx: usize,
}

impl ImageDirSource {
    pub fn open(dir: &Path) -> Result<Self, PerceptionError> {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)?
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.extension().and_then(|e| e.to_str()).is_some_and(|e| {
                    matches!(e.to_lowercase().as_str(), "png" | "jpg" | "jpeg" | "bmp")
                })
            })
            .collect();
        paths.sort();
        if paths.is_empty() {
            return Err(PerceptionError::Unsupported(format!(
                "no images in {}",
                dir.display()
            )));
        }
        Ok(Self { paths, idx: 0 })
    }
}

impl CaptureSource for ImageDirSource {
    fn next_frame(&mut self) -> Result<Option<Frame>, PerceptionError> {
        if self.idx >= self.paths.len() {
            return Ok(None);
        }
        let p = &self.paths[self.idx];
        self.idx += 1;
        let img =
            image::open(p).map_err(|e| PerceptionError::Decode(format!("{}: {e}", p.display())))?;
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width(), rgb.height());
        Ok(Some(Frame {
            seq: self.idx as u64 - 1,
            at_capture_ms: now_ms(),
            width: w,
            height: h,
            data: rgb.into_raw(),
        }))
    }

    fn label(&self) -> String {
        format!("imagedir {} files", self.paths.len())
    }
}

#[cfg(feature = "camera")]
mod camera {
    use super::*;

    /// V4L2/UVC camera via nokhwa. Requests the highest resolution the
    /// device offers (PRD: 1080p capture, ROI-crop at native res).
    pub struct NokhwaSource {
        cam: nokhwa::Camera,
        seq: u64,
    }

    /// Best-effort "max mode" sensor lock: auto-exposure stays on (adapts to
    /// bright and dark rooms) but dynamic-framerate is OFF, so the firmware
    /// caps exposure at the frame budget and boosts gain instead of
    /// collapsing to ~9fps in the dark. Runs before the stream opens;
    /// a no-op when v4l2-ctl is missing or refuses.
    fn lock_sensor_max_mode(index: u32) {
        if std::env::var("ALIVE_NO_SENSOR_LOCK").is_ok() {
            eprintln!("[alive] sensor lock skipped (ALIVE_NO_SENSOR_LOCK)");
            return;
        }
        let dev = format!("/dev/video{index}");
        let ctrls = "exposure_dynamic_framerate=0,auto_exposure=3,gain=0,\
             sharpness=5,backlight_compensation=1,white_balance_automatic=1";
        match std::process::Command::new("v4l2-ctl")
            .args(["-d", &dev, "--set-ctrl", ctrls])
            .output()
        {
            Ok(o) if o.status.success() => {
                eprintln!("[alive] sensor max-mode locked ({dev})")
            }
            Ok(o) => eprintln!(
                "[alive] sensor lock refused: {}",
                String::from_utf8_lossy(&o.stderr).trim()
            ),
            Err(e) => eprintln!("[alive] v4l2-ctl missing ({e}); auto-exposure stays"),
        }
    }

    impl NokhwaSource {
        pub fn open(index: u32) -> Result<Self, PerceptionError> {
            Self::open_with(index, 1280, 720, 30)
        }

        pub fn open_with(
            index: u32,
            width: u32,
            height: u32,
            fps: u32,
        ) -> Result<Self, PerceptionError> {
            use nokhwa::utils::FrameFormat;
            Self::open_with_format(index, width, height, fps, FrameFormat::MJPEG)
        }

        /// Format-selectable open. YUYV (uncompressed) skips JPEG decode
        /// entirely — much lower latency + zero block artifacts at <=480p.
        pub fn open_with_format(
            index: u32,
            width: u32,
            height: u32,
            fps: u32,
            format: nokhwa::utils::FrameFormat,
        ) -> Result<Self, PerceptionError> {
            use nokhwa::utils::{
                CameraFormat, CameraIndex, RequestedFormat, RequestedFormatType, Resolution,
            };
            let idx = CameraIndex::Index(index);
            let want = CameraFormat::new(Resolution::new(width, height), format, fps);
            lock_sensor_max_mode(index);
            let fmt = RequestedFormat::new::<nokhwa::pixel_format::RgbFormat>(
                RequestedFormatType::Closest(want),
            );
            let mut cam = nokhwa::Camera::new(idx, fmt)
                .map_err(|e| PerceptionError::Camera(format!("open cam {index}: {e}")))?;
            cam.open_stream()
                .map_err(|e| PerceptionError::Camera(format!("stream cam {index}: {e}")))?;
            Ok(Self { cam, seq: 0 })
        }

        /// Frame plus (wait_ms, decode_ms) split so benches prove where time goes.
        pub fn next_frame_timed(&mut self) -> Result<Option<(Frame, f64, f64)>, PerceptionError> {
            let t0 = std::time::Instant::now();
            let buf = self
                .cam
                .frame()
                .map_err(|e| PerceptionError::Camera(e.to_string()))?;
            let wait_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let img: image::RgbImage = buf
                .decode_image::<nokhwa::pixel_format::RgbFormat>()
                .map_err(|e| PerceptionError::Decode(e.to_string()))?;
            let decode_ms = t0.elapsed().as_secs_f64() * 1000.0 - wait_ms;
            let (w, h) = (img.width(), img.height());
            let f = Frame {
                seq: self.seq,
                at_capture_ms: now_ms(),
                width: w,
                height: h,
                data: img.into_raw(),
            };
            self.seq += 1;
            Ok(Some((f, wait_ms, decode_ms)))
        }

        pub fn camera_info(&self) -> String {
            format!("backend={:?}", self.cam.backend())
        }
    }

    impl CaptureSource for NokhwaSource {
        fn next_frame(&mut self) -> Result<Option<Frame>, PerceptionError> {
            Ok(self.next_frame_timed()?.map(|(f, _, _)| f))
        }

        fn label(&self) -> String {
            format!("nokhwa {}", self.camera_info())
        }
    }
}

#[cfg(feature = "camera")]
pub use camera::NokhwaSource;

#[cfg(feature = "onnx")]
pub mod onnx_backend;

// ---------------------------------------------------------------------------
// ROI ops
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Roi {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

/// Crop with bounds clamping. Never panics on out-of-range ROIs.
pub fn crop_rgb(frame: &Frame, roi: &Roi) -> Frame {
    let x = roi.x.min(frame.width.saturating_sub(1));
    let y = roi.y.min(frame.height.saturating_sub(1));
    let w = roi.w.min(frame.width - x).max(1);
    let h = roi.h.min(frame.height - y).max(1);
    let mut data = vec![0u8; (w * h * 3) as usize];
    for row in 0..h {
        let src_y = (y + row) as usize;
        let src_off = (src_y * frame.width as usize + x as usize) * 3;
        let dst_off = (row * w) as usize * 3;
        data[dst_off..dst_off + (w as usize * 3)]
            .copy_from_slice(&frame.data[src_off..src_off + (w as usize * 3)]);
    }
    Frame {
        seq: frame.seq,
        at_capture_ms: frame.at_capture_ms,
        width: w,
        height: h,
        data,
    }
}

pub fn resize_rgb(frame: &Frame, w: u32, h: u32) -> Frame {
    let img = match frame.rgb_image() {
        Some(i) => i,
        None => {
            return Frame {
                seq: frame.seq,
                at_capture_ms: frame.at_capture_ms,
                width: 0,
                height: 0,
                data: vec![],
            }
        }
    };
    let r = image::imageops::resize(
        &img,
        w.max(1),
        h.max(1),
        image::imageops::FilterType::Triangle,
    );
    Frame {
        seq: frame.seq,
        at_capture_ms: frame.at_capture_ms,
        width: r.width(),
        height: r.height(),
        data: r.into_raw(),
    }
}

// ---------------------------------------------------------------------------
// Landmarks
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Landmarks {
    pub at_ms: u64,
    pub present: bool,
    /// xyz per keypoint, model-native order (133 for whole-body).
    pub points: Vec<[f32; 3]>,
    pub scores: Vec<f32>,
}

pub trait LandmarkExtractor: Send {
    fn infer(&mut self, frame: &Frame) -> Result<Landmarks, PerceptionError>;
    fn keypoints(&self) -> usize;
    fn name(&self) -> &'static str;
    /// Refresh person detection (background cadence). Default: no-op.
    fn rescan(&mut self, _frame: &Frame) {}
    /// True when a person track is currently held.
    fn tracking(&self) -> bool {
        false
    }
}

/// Reports absent instead of guessing. Used in CI and no-model benches.
pub struct NullExtractor;

impl LandmarkExtractor for NullExtractor {
    fn infer(&mut self, frame: &Frame) -> Result<Landmarks, PerceptionError> {
        Ok(Landmarks {
            at_ms: frame.at_capture_ms,
            present: false,
            points: vec![],
            scores: vec![],
        })
    }
    fn keypoints(&self) -> usize {
        0
    }
    fn name(&self) -> &'static str {
        "null"
    }
}

/// Flatten for the temporal layer (`motion_energy` input).
pub fn flatten_landmarks(lm: &Landmarks) -> Vec<f32> {
    let mut v = Vec::with_capacity(lm.points.len() * 3);
    for p in &lm.points {
        v.extend_from_slice(p);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_is_deterministic() {
        let mut a = SyntheticSource::new(64, 48, Some(3));
        let mut b = SyntheticSource::new(64, 48, Some(3));
        for _ in 0..3 {
            let fa = a.next_frame().unwrap().unwrap();
            let fb = b.next_frame().unwrap().unwrap();
            assert_eq!(fa.data, fb.data);
        }
        assert!(a.next_frame().unwrap().is_none());
    }

    #[test]
    fn crop_clamps_and_preserves_pixels() {
        let mut s = SyntheticSource::new(32, 32, Some(1));
        let f = s.next_frame().unwrap().unwrap();
        let c = crop_rgb(
            &f,
            &Roi {
                x: 1000,
                y: 1000,
                w: 500,
                h: 500,
            },
        );
        assert_eq!((c.width, c.height), (1, 1));
        let c2 = crop_rgb(
            &f,
            &Roi {
                x: 4,
                y: 4,
                w: 8,
                h: 8,
            },
        );
        assert_eq!((c2.width, c2.height), (8, 8));
        // Top-left pixel of crop == source pixel (4,4).
        let src_off = (4 * 32 + 4) * 3;
        assert_eq!(&c2.data[0..3], &f.data[src_off..src_off + 3]);
    }

    #[test]
    fn resize_dims() {
        let mut s = SyntheticSource::new(64, 64, Some(1));
        let f = s.next_frame().unwrap().unwrap();
        let r = resize_rgb(&f, 192, 256);
        assert_eq!((r.width, r.height), (192, 256));
        assert_eq!(r.data.len(), 192 * 256 * 3);
    }

    #[test]
    fn null_extractor_reports_absent() {
        let mut s = SyntheticSource::new(16, 16, Some(1));
        let f = s.next_frame().unwrap().unwrap();
        let mut n = NullExtractor;
        let lm = n.infer(&f).unwrap();
        assert!(!lm.present);
        assert!(flatten_landmarks(&lm).is_empty());
    }

    #[test]
    fn imagedir_loads_sorted() {
        let dir = tempfile::tempdir().unwrap();
        let mk = |n: &str| {
            let img = image::RgbImage::from_pixel(8, 8, image::Rgb([1, 2, 3]));
            img.save(dir.path().join(n)).unwrap();
        };
        mk("b.png");
        mk("a.png");
        std::fs::write(dir.path().join("note.txt"), "skip me").unwrap();
        let mut src = ImageDirSource::open(dir.path()).unwrap();
        assert_eq!(src.label(), "imagedir 2 files");
        assert!(src.next_frame().unwrap().is_some());
        assert!(src.next_frame().unwrap().is_some());
        assert!(src.next_frame().unwrap().is_none());
    }
}
