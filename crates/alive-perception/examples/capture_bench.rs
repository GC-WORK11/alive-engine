//! Capture benchmark: frames, fps, p50/p95 per-frame latency as JSON.
//! Usage: cargo run -p alive-perception --example capture_bench -- [--frames N] [--source camera|synthetic] [--width W] [--height H] [--format mjpeg|yuyv] [--save PREFIX]

use alive_perception::{CaptureSource, SyntheticSource};
use std::time::Instant;

fn pct(sorted: &mut [f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let i = ((p / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[i.min(sorted.len() - 1)]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut frames_n = 300usize;
    let mut source = "camera".to_string();
    let mut format = "mjpeg".to_string();
    let mut save = String::new();
    let mut w = 1280u32;
    let mut h = 720u32;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--frames" => {
                frames_n = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(300);
                i += 2;
            }
            "--source" => {
                source = args.get(i + 1).cloned().unwrap_or(source);
                i += 2;
            }
            "--width" => {
                w = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(1280);
                i += 2;
            }
            "--height" => {
                h = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(720);
                i += 2;
            }
            "--format" => {
                format = args.get(i + 1).cloned().unwrap_or(format);
                i += 2;
            }
            "--save" => {
                save = args.get(i + 1).cloned().unwrap_or(save);
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }

    #[cfg(feature = "camera")]
    let mut cam: Option<alive_perception::NokhwaSource> = None;
    #[cfg(feature = "camera")]
    if source == "camera" {
        let fmt = match format.to_lowercase().as_str() {
            "yuyv" | "yuy2" => nokhwa::utils::FrameFormat::YUYV,
            _ => nokhwa::utils::FrameFormat::MJPEG,
        };
        match alive_perception::NokhwaSource::open_with_format(0, w, h, 30, fmt) {
            Ok(c) => {
                eprintln!("camera: {}", c.label());
                cam = Some(c);
            }
            Err(e) => {
                println!(
                    "{}",
                    serde_json::json!({"ok": false, "error": e.to_string()})
                );
                std::process::exit(2);
            }
        }
    }
    #[cfg(not(feature = "camera"))]
    if source == "camera" {
        println!(
            "{}",
            serde_json::json!({"ok": false, "error": "built without camera feature"})
        );
        std::process::exit(2);
    }

    let mut synth = SyntheticSource::new(w, h, None);
    let mut lat: Vec<f64> = Vec::with_capacity(frames_n);
    let mut wait: Vec<f64> = Vec::with_capacity(frames_n);
    let mut dec: Vec<f64> = Vec::with_capacity(frames_n);
    let mut got = 0usize;
    let mut res = (w, h);
    let t_all = Instant::now();
    for _ in 0..frames_n {
        let t0 = Instant::now();
        #[cfg(feature = "camera")]
        let f = if let Some(c) = cam.as_mut() {
            match c.next_frame_timed() {
                Ok(Some((frame, wms, dms))) => {
                    wait.push(wms);
                    dec.push(dms);
                    Ok(Some(frame))
                }
                Ok(None) => Ok(None),
                Err(e) => Err(e),
            }
        } else {
            synth.next_frame()
        };
        #[cfg(not(feature = "camera"))]
        let f = synth.next_frame();
        let dt = t0.elapsed().as_secs_f64() * 1000.0;
        match f {
            Ok(Some(frame)) => {
                got += 1;
                res = (frame.width, frame.height);
                lat.push(dt);
                if !save.is_empty() && frame.seq % 20 == 0 {
                    let path = format!("{}-{}.png", save, frame.seq);
                    if let Some(rgb) = frame.rgb_image() {
                        let _ = rgb.save(&path);
                    }
                }
            }
            Ok(None) => break,
            Err(e) => {
                println!(
                    "{}",
                    serde_json::json!({"ok": false, "error": e.to_string()})
                );
                std::process::exit(2);
            }
        }
    }
    let wall = t_all.elapsed().as_secs_f64();
    let out = serde_json::json!({
        "ok": true,
        "source": source,
        "frames": got,
        "width": res.0,
        "height": res.1,
        "wall_s": (wall * 1000.0).round() / 1000.0,
        "fps": if wall > 0.0 { (got as f64 / wall * 10.0).round() / 10.0 } else { 0.0 },
        "p50_ms": (pct(&mut lat.clone(), 50.0) * 100.0).round() / 100.0,
        "p95_ms": (pct(&mut lat.clone(), 95.0) * 100.0).round() / 100.0,
        "wait_p50_ms": (pct(&mut wait.clone(), 50.0) * 100.0).round() / 100.0,
        "decode_p50_ms": (pct(&mut dec.clone(), 50.0) * 100.0).round() / 100.0,
    });
    println!("{out}");
}
