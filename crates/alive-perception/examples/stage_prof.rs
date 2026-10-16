//! Stage profiler: hard numbers for every pipeline stage on live input.
//! Prints p50/p95 per stage + sequential sum vs overlapped lower bound.
//! Usage: cargo run -p alive-perception --features onnx --example stage_prof --
//!   [--frames N] [--models DIR] [--source camera|PATH] [--width W] [--height H]

use alive_perception::onnx_backend::{
    letterbox_bgr, warp_normalize_pose, Bbox, CropAffine, Letterbox, RTMW_H, RTMW_W,
};
use alive_perception::{CaptureSource, LandmarkExtractor, SyntheticSource};
use std::time::Instant;

fn pct(v: &mut [f64], p: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((p / 100.0) * (v.len() as f64 - 1.0)).round() as usize]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut frames_n = 60usize;
    let mut models = String::from("models");
    let mut source = String::from("camera");
    let mut w = 640u32;
    let mut h = 480u32;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--frames" => {
                frames_n = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(60);
                i += 2;
            }
            "--models" => {
                models = args.get(i + 1).cloned().unwrap_or(models);
                i += 2;
            }
            "--source" => {
                source = args.get(i + 1).cloned().unwrap_or(source);
                i += 2;
            }
            "--width" => {
                w = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(640);
                i += 2;
            }
            "--height" => {
                h = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(480);
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }

    let mut tracker =
        alive_perception::onnx_backend::WholeBodyTracker::load(std::path::Path::new(&models), 10)
            .expect("load models");

    #[cfg(feature = "camera")]
    let mut cam = if source == "camera" {
        Some(alive_perception::NokhwaSource::open_with(0, w, h, 30).expect("open camera"))
    } else {
        None
    };
    let single = if source != "camera" {
        let img = image::open(&source).expect("open image").to_rgb8();
        Some(alive_perception::Frame {
            seq: 0,
            at_capture_ms: 0,
            width: img.width(),
            height: img.height(),
            data: img.into_raw(),
        })
    } else {
        None
    };
    let mut synth = SyntheticSource::new(w, h, None);

    // Warmup (models + camera auto-exposure settle).
    for _ in 0..5 {
        #[cfg(feature = "camera")]
        let f = if let Some(c) = cam.as_mut() {
            c.next_frame().unwrap().unwrap()
        } else if let Some(img) = single.clone() {
            img
        } else {
            synth.next_frame().unwrap().unwrap()
        };
        #[cfg(not(feature = "camera"))]
        let f = single
            .clone()
            .unwrap_or_else(|| synth.next_frame().unwrap().unwrap());
        let _ = tracker.refresh_detection(&f);
        let _ = tracker.infer(&f);
    }

    let mut wait = vec![];
    let mut decode = vec![];
    let mut resize = vec![];
    let mut jpeg = vec![];
    let mut rescan = vec![];
    let mut pose = vec![];
    let mut flatx = vec![];
    let mut total = vec![];

    for n in 0..frames_n {
        let t_all = Instant::now();
        // Capture with split.
        let t0 = Instant::now();
        #[cfg(feature = "camera")]
        let frame = if let Some(c) = cam.as_mut() {
            let (f, w, d) = c.next_frame_timed().unwrap().unwrap();
            wait.push(w);
            decode.push(d);
            f
        } else if let Some(img) = single.clone() {
            wait.push(0.0);
            decode.push(0.0);
            img
        } else {
            let f = synth.next_frame().unwrap().unwrap();
            wait.push(t0.elapsed().as_secs_f64() * 1000.0);
            decode.push(0.0);
            f
        };
        #[cfg(not(feature = "camera"))]
        let frame = {
            wait.push(0.0);
            decode.push(0.0);
            single.clone().unwrap()
        };
        let _ = n;

        // PiP path: resize 320x180 + JPEG q75 (matches the app).
        let t = Instant::now();
        let rgb = frame.rgb_image().unwrap();
        let small = image::imageops::resize(&rgb, 320, 180, image::imageops::FilterType::Triangle);
        resize.push(t.elapsed().as_secs_f64() * 1000.0);
        let t = Instant::now();
        let mut buf = Vec::new();
        let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, 75);
        enc.encode_image(&image::DynamicImage::ImageRgb8(small))
            .unwrap();
        jpeg.push(t.elapsed().as_secs_f64() * 1000.0);

        // Detect (every frame here to measure; app uses cadence).
        let t = Instant::now();
        let boxes = tracker.refresh_detection(&frame).unwrap();
        rescan.push(t.elapsed().as_secs_f64() * 1000.0);
        let _ = boxes;

        // Pose (tracked path).
        let t = Instant::now();
        let lm = tracker.infer(&frame).unwrap();
        pose.push(t.elapsed().as_secs_f64() * 1000.0);
        let _ = lm.present;

        // Flatten + motion.
        let t = Instant::now();
        let flat = alive_perception::flatten_landmarks(&lm);
        let _ = alive_temporal::motion_energy(&flat, &flat);
        flatx.push(t.elapsed().as_secs_f64() * 1000.0);

        total.push(t_all.elapsed().as_secs_f64() * 1000.0);
    }

    // Preprocess-only micro splits (no ORT): letterbox + warp on frame size.
    let mut s = SyntheticSource::new(w, h, Some(1));
    let f = s.next_frame().unwrap().unwrap();
    let t = Instant::now();
    for _ in 0..20 {
        let lb = Letterbox::for_frame(w, h);
        let _ = letterbox_bgr(&f, &lb);
    }
    let letterbox_ms = t.elapsed().as_secs_f64() * 1000.0 / 20.0;
    let t = Instant::now();
    for _ in 0..20 {
        let a = CropAffine::from_bbox(
            &Bbox {
                x1: 100.0,
                y1: 100.0,
                x2: 500.0,
                y2: 600.0,
                score: 1.0,
            },
            RTMW_W,
            RTMW_H,
        );
        let _ = warp_normalize_pose(&f, &a);
    }
    let warp_ms = t.elapsed().as_secs_f64() * 1000.0 / 20.0;

    let row = |v: &mut Vec<f64>| format!("p50 {:7.2}  p95 {:7.2}", pct(v, 50.0), pct(v, 95.0));
    println!("stage profile: {frames_n} frames, source={source}");
    println!("  capture_wait  {}", row(&mut wait));
    println!("  mjpeg_decode  {}", row(&mut decode));
    println!("  pip_resize    {}", row(&mut resize));
    println!("  pip_jpeg      {}", row(&mut jpeg));
    println!("  detect_full   {}", row(&mut rescan));
    println!("  pose_tracked  {}", row(&mut pose));
    println!("  flatten+mot   {}", row(&mut flatx));
    println!("  TOTAL_seq     {}", row(&mut total));
    println!("micro: letterbox {letterbox_ms:.2}ms  warp {warp_ms:.2}ms");
    let seq = pct(&mut total.clone(), 50.0);
    let overlap = pct(&mut wait.clone(), 50.0)
        .max(pct(&mut decode.clone(), 50.0) + pct(&mut pose.clone(), 50.0));
    println!(
        "math: sequential_p50 {seq:.1}ms ({:.1} fps) | overlapped_lower_bound {overlap:.1}ms ({:.1} fps)",
        1000.0 / seq.max(0.01),
        1000.0 / overlap.max(0.01)
    );
}
