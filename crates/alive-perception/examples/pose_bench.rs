//! Live whole-body bench: camera -> YOLOX detect -> RTMW 133-kp pose.
//! Prints JSON timings + saves annotated PNGs as visual proof.
//! Usage: cargo run -p alive-perception --features onnx --example pose_bench --
//!   [--frames N] [--models DIR] [--save PREFIX] [--source camera|synthetic|PATH_TO_IMAGE]
//!   [--threads N] [--force-bbox]

use alive_perception::onnx_backend::{Bbox, WholeBodyTracker};
use alive_perception::{CaptureSource, Frame, LandmarkExtractor, SyntheticSource};
use std::time::Instant;

fn pct(sorted: &mut [f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted[((p / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize]
}

fn annotate(frame: &Frame, lm: &alive_perception::Landmarks) -> image::RgbImage {
    let mut img = frame.rgb_image().unwrap();
    for (i, p) in lm.points.iter().enumerate() {
        let x = p[0].round() as i64;
        let y = p[1].round() as i64;
        if x < 0 || y < 0 || x >= img.width() as i64 || y >= img.height() as i64 {
            continue;
        }
        let s = lm.scores.get(i).copied().unwrap_or(0.0).clamp(0.0, 1.0);
        let col = image::Rgb([(255.0 * (1.0 - s)) as u8, (255.0 * s) as u8, 60]);
        for dy in -2..=2 {
            for dx in -2..=2 {
                let (xx, yy) = (x + dx, y + dy);
                if xx >= 0 && yy >= 0 && xx < img.width() as i64 && yy < img.height() as i64 {
                    img.put_pixel(xx as u32, yy as u32, col);
                }
            }
        }
    }
    img
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut frames_n = 60usize;
    let mut models = String::from("models");
    let mut save = String::from("/tmp/alive-pose");
    let mut source = String::from("camera");
    let mut threads = 8usize;
    let mut force_bbox = false;
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
            "--save" => {
                save = args.get(i + 1).cloned().unwrap_or(save);
                i += 2;
            }
            "--source" => {
                source = args.get(i + 1).cloned().unwrap_or(source);
                i += 2;
            }
            "--threads" => {
                threads = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(8);
                i += 2;
            }
            "--force-bbox" => {
                force_bbox = true;
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    }

    let t_load = Instant::now();
    let mut tracker = match WholeBodyTracker::load(std::path::Path::new(&models), threads) {
        Ok(t) => t,
        Err(e) => {
            println!(
                "{}",
                serde_json::json!({"ok": false, "error": e.to_string()})
            );
            std::process::exit(2);
        }
    };
    eprintln!("models loaded in {:.1}s", t_load.elapsed().as_secs_f64());

    // Frame source: camera | single image | synthetic.
    #[cfg(feature = "camera")]
    let mut cam = if source == "camera" {
        match alive_perception::NokhwaSource::open(0) {
            Ok(c) => {
                eprintln!("camera: {}", c.label());
                Some(c)
            }
            Err(e) => {
                println!(
                    "{}",
                    serde_json::json!({"ok": false, "error": e.to_string()})
                );
                std::process::exit(2);
            }
        }
    } else {
        None
    };
    let single: Option<Frame> = if source != "camera" && source != "synthetic" {
        let img = image::open(&source).expect("open image");
        let rgb = img.to_rgb8();
        Some(Frame {
            seq: 0,
            at_capture_ms: 0,
            width: rgb.width(),
            height: rgb.height(),
            data: rgb.into_raw(),
        })
    } else {
        None
    };
    let mut synth = SyntheticSource::new(640, 480, None);
    let use_synth = source == "synthetic";

    let mut pose_ms: Vec<f64> = Vec::new();
    let mut det_ms: Vec<f64> = Vec::new();
    let mut present_n = 0usize;
    let mut score_sum = 0.0f64;
    let mut confident_max = 0usize;
    let mut top_max = 0.0f32;
    let mut first_boxes = serde_json::Value::Null;
    let mut saved = 0u32;

    for f in 0..frames_n {
        let mut frame = if use_synth {
            synth.next_frame().unwrap().unwrap()
        } else if let Some(img) = single.clone() {
            Frame {
                seq: f as u64,
                ..img
            }
        } else {
            #[cfg(feature = "camera")]
            {
                match cam.as_mut().unwrap().next_frame().unwrap() {
                    Some(fr) => fr,
                    None => break,
                }
            }
            #[cfg(not(feature = "camera"))]
            {
                synth.next_frame().unwrap().unwrap()
            }
        };
        frame.seq = f as u64;

        // Force-bbox mode: skip detection, re-seed a full-frame track every
        // frame (isolates per-frame tracked-pose cost for thread sweeps).
        if force_bbox {
            tracker.set_bbox(Some(Bbox::full_frame(frame.width, frame.height)));
        }
        if !force_bbox && (f == 0 || f % 30 == 29) {
            let t0 = Instant::now();
            match tracker.refresh_detection(&frame) {
                Ok(boxes) => {
                    det_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
                    if f == 0 {
                        first_boxes = serde_json::json!(boxes.iter().map(|b| {
                            serde_json::json!({"x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2, "score": b.score})
                        }).collect::<Vec<_>>());
                        eprintln!("frame0 boxes: {first_boxes}");
                    }
                }
                Err(e) => {
                    println!(
                        "{}",
                        serde_json::json!({"ok": false, "error": format!("detect: {e}")})
                    );
                    std::process::exit(2);
                }
            }
        }

        let t0 = Instant::now();
        let lm = match tracker.infer(&frame) {
            Ok(l) => l,
            Err(e) => {
                println!(
                    "{}",
                    serde_json::json!({"ok": false, "error": format!("pose: {e}")})
                );
                std::process::exit(2);
            }
        };
        pose_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        if lm.present {
            present_n += 1;
        }
        if !lm.scores.is_empty() {
            score_sum += lm.scores.iter().sum::<f32>() as f64 / lm.scores.len() as f64;
            confident_max = confident_max.max(lm.scores.iter().filter(|&&s| s >= 0.4).count());
            top_max = top_max.max(lm.scores.iter().copied().fold(0.0f32, f32::max));
        }
        if saved < 3 && (f == 0 || lm.present) {
            let img = annotate(&frame, &lm);
            let path = format!("{save}-{f}.png");
            img.save(&path).expect("save png");
            eprintln!(
                "saved {path} present={} kps={}",
                lm.present,
                lm.points.len()
            );
            saved += 1;
        }
    }

    let n = pose_ms.len().max(1);
    println!(
        "{}",
        serde_json::json!({
            "ok": true,
            "frames": n,
            "present_rate": (present_n as f64 / n as f64 * 1000.0).round() / 1000.0,
            "confident_max": confident_max,
            "top_score_max": (top_max * 1000.0).round() / 1000.0,
            "mean_score": (score_sum / n as f64 * 1000.0).round() / 1000.0,
            "pose_p50_ms": (pct(&mut pose_ms.clone(), 50.0) * 10.0).round() / 10.0,
            "pose_p95_ms": (pct(&mut pose_ms.clone(), 95.0) * 10.0).round() / 10.0,
            "det_p50_ms": (pct(&mut det_ms.clone(), 50.0) * 10.0).round() / 10.0,
            "frame0_boxes": first_boxes,
        })
    );
}
