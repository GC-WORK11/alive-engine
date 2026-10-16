//! Debug probe: dump the pose crop PNG + raw simcc stats for one image.
//! Usage: cargo run -p alive-perception --features onnx --example crop_probe -- IMG OUT_CROP

use alive_perception::onnx_backend::{Bbox, CropAffine, RTMW_H, RTMW_W};
use alive_perception::ImageDirSource;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let img_path = args
        .get(1)
        .cloned()
        .unwrap_or("/tmp/onnxprobe/rtmw/output_pytorch.jpg".into());
    let crop_path = args.get(2).cloned().unwrap_or("/tmp/alive-crop.png".into());

    // Load single image via a temp dir source trick: just open with image crate.
    let img = image::open(&img_path).expect("open image").to_rgb8();
    let (w, h) = (img.width(), img.height());
    let frame = alive_perception::Frame {
        seq: 0,
        at_capture_ms: 0,
        width: w,
        height: h,
        data: img.into_raw(),
    };
    eprintln!("image {img_path} {w}x{h}");

    let mut tracker =
        alive_perception::onnx_backend::WholeBodyTracker::load(std::path::Path::new("models"), 8)
            .unwrap();
    let boxes = tracker.refresh_detection(&frame).unwrap();
    eprintln!("boxes: {boxes:?}");
    let bbox = boxes
        .first()
        .copied()
        .unwrap_or_else(|| Bbox::full_frame(w, h));

    // Crop with the same affine, save de-normalized view.
    let affine = CropAffine::from_bbox(&bbox, RTMW_W, RTMW_H);
    let chw = alive_perception::onnx_backend::warp_normalize_pose(&frame, &affine);
    let (cw, ch) = (RTMW_W as usize, RTMW_H as usize);
    let mean = [123.675f32, 116.28, 103.53];
    let std = [58.395f32, 57.12, 57.375];
    let mut rgb = image::RgbImage::new(RTMW_W, RTMW_H);
    for y in 0..ch {
        for x in 0..cw {
            let mut px = [0u8; 3];
            for c in 0..3 {
                let v = chw[c * cw * ch + y * cw + x] * std[c] + mean[c];
                px[c] = v.clamp(0.0, 255.0) as u8;
            }
            rgb.put_pixel(x as u32, y as u32, image::Rgb(px));
        }
    }
    rgb.save(&crop_path).unwrap();
    eprintln!("saved crop {crop_path}");

    // Raw simcc stats via a fresh pose session call path: run infer and dump.
    // (Reach in through infer_bbox on the same bbox.)
    let mut pose = load_pose_only();
    let lm = pose.infer_bbox(&frame, &bbox).unwrap();
    for (name, range) in [
        ("body0-16", 0..17),
        ("feet17-22", 17..23),
        ("face23-90", 23..91),
        ("hand91-111", 91..112),
        ("hand112-132", 112..133),
    ] {
        let pts: Vec<[f32; 3]> = range.clone().map(|i| lm.points[i]).collect();
        let sc: Vec<f32> = range.map(|i| lm.scores[i]).collect();
        let n = pts.len() as f32;
        let mx: f32 = pts.iter().map(|p| p[0]).sum::<f32>() / n;
        let my: f32 = pts.iter().map(|p| p[1]).sum::<f32>() / n;
        let ms: f32 = sc.iter().sum::<f32>() / n;
        eprintln!("{name}: mean=({mx:.0},{my:.0}) meanscore={ms:.2}");
    }
    let mut top: Vec<(usize, f32)> = lm.scores.iter().copied().enumerate().collect();
    top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    eprintln!("top10 (kp,score): {:?}", &top[..10.min(top.len())]);
    eprintln!(
        "coords of top3: {:?}",
        top.iter()
            .take(3)
            .map(|(i, _)| lm.points[*i])
            .collect::<Vec<_>>()
    );
    let _ = ImageDirSource::open; // keep import used on some cfgs
}

fn load_pose_only() -> alive_perception::onnx_backend::RtmwPose {
    alive_perception::onnx_backend::RtmwPose::load(
        std::path::Path::new("models/rtmw-l-256x192.onnx"),
        8,
    )
    .unwrap()
}
