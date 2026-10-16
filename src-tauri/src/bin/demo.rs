//! Headless end-to-end proof (no WebView): exercises every REAL component.
//!
//! HONEST BOUNDARY: free-ASL classification needs the trained encoder + corpus
//! (M1+). Here the trajectory matcher proves calibrated-command recognition on
//! synthetic paths, and a canned NBest proves decide -> route -> execute ->
//! journal -> undo. When ALIVE_JEV_API_KEY is set, the decide step also runs
//! live against Jev and reports real latency.

use std::collections::HashMap;

use alive_intent::{
    alive_questions, calibrated_confidence, ChoiceAnswer, Decision, DecisionProvider,
    DecisionRequest, MockProvider,
};
use alive_sign::matcher::TemplateBank;
use alive_sign::LanguagePack;

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

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let mut steps = serde_json::Map::new();
    let mut ok = true;

    // 1. Matcher: calibrated recognition + rejection.
    {
        let mut bank = TemplateBank::new(0.45);
        bank.add_sample("SWIPE RIGHT", &line(24, 0.0, 0.0, 90.0, 0.0));
        bank.add_sample("CIRCLE", &circle(24, 40.0));
        let r1 = bank.recognize(&line(24, 5.0, 3.0, 100.0, 2.0));
        let squig: Vec<(f32, f32)> = (0..24)
            .map(|i| ((i as f32 * 37.0) % 90.0, (i as f32 * 53.0) % 70.0))
            .collect();
        let r2 = bank.recognize(&squig);
        match (r1, r2) {
            (Some((g, s)), None) if g == "SWIPE RIGHT" && s > 0.6 => {
                steps.insert(
                    "matcher".into(),
                    serde_json::json!({"ok": true, "score": s}),
                );
            }
            other => fail(
                &mut steps,
                &mut ok,
                "matcher",
                format!("unexpected {other:?}"),
            ),
        }
    }

    // 2. Pack: lexicon + intent resolution.
    let pack = match LanguagePack::load(std::path::Path::new("packs/asl")) {
        Ok(p) => {
            let e = p.match_gloss("open browser");
            match e.and_then(|e| e.intent.clone()) {
                Some(i) if i == "open_app" => {
                    steps.insert(
                        "pack".into(),
                        serde_json::json!({"ok": true, "entries": p.lexicon.entries.len()}),
                    );
                    p
                }
                other => {
                    fail(
                        &mut steps,
                        &mut ok,
                        "pack",
                        format!("lexicon mismatch: {other:?}"),
                    );
                    finish(ok, steps);
                }
            }
        }
        Err(e) => {
            fail(&mut steps, &mut ok, "pack", e.to_string());
            finish(ok, steps);
        }
    };

    // 3. Capture + pose smoke (room-dependent presence; errors are failures).
    {
        use alive_perception::{CaptureSource, LandmarkExtractor};
        let t0 = std::time::Instant::now();
        let mut source: Box<dyn CaptureSource> =
            match alive_perception::NokhwaSource::open_with(0, 1280, 720, 30) {
                Ok(c) => Box::new(c),
                Err(e) => {
                    steps.insert(
                        "capture".into(),
                        serde_json::json!({"ok": false, "error": e.to_string()}),
                    );
                    ok = false;
                    finish(ok, steps);
                }
            };
        let mut got = 0;
        for _ in 0..5 {
            match source.next_frame() {
                Ok(Some(_)) => got += 1,
                Ok(None) => break,
                Err(e) => {
                    fail(&mut steps, &mut ok, "capture", e.to_string());
                    finish(ok, steps);
                }
            }
        }
        let vision = {
            let models = std::path::Path::new("models");
            if models.join("rtmw-l-256x192.onnx").is_file() {
                match alive_perception::onnx_backend::WholeBodyTracker::load(models, 8) {
                    Ok(mut t) => {
                        // One detect+pose round on a fresh frame.
                        let f = source.next_frame().unwrap().unwrap();
                        let det_ms = {
                            let dt = std::time::Instant::now();
                            let boxes = t.refresh_detection(&f).unwrap_or_default();
                            (dt.elapsed().as_secs_f64() * 1000.0, boxes.len())
                        };
                        let (present, pose_ms) = {
                            let dt = std::time::Instant::now();
                            let lm = t.infer(&f).unwrap();
                            (lm.present, dt.elapsed().as_secs_f64() * 1000.0)
                        };
                        serde_json::json!({"backend": "rtmw-l", "det_ms": det_ms.0, "boxes": det_ms.1, "present": present, "pose_ms": pose_ms})
                    }
                    Err(e) => serde_json::json!({"backend": "error", "error": e.to_string()}),
                }
            } else {
                serde_json::json!({"backend": "null", "note": "models missing"})
            }
        };
        steps.insert(
            "capture".into(),
            serde_json::json!({"ok": true, "frames": got, "ms": t0.elapsed().as_secs_f64() * 1000.0, "vision": vision}),
        );
    }

    // 4. Decide: canned NBest -> lexicon mock -> route. Live Jev if keyed.
    {
        let req = DecisionRequest {
            state: serde_json::json!({"gloss_nbest": [{"gloss": "OPEN BROWSER", "p": 0.9}]}),
            questions: alive_questions(&pack.intent_options()),
        };
        // Lexicon-driven mock (mirrors the app's offline path).
        let entry = pack.match_gloss("OPEN BROWSER").unwrap();
        let intent = entry.intent.clone().unwrap();
        let mut probs = HashMap::new();
        probs.insert(intent.clone(), 0.9);
        probs.insert("dictate".to_string(), 0.1);
        let mock = MockProvider::from_fn(move |_| Decision {
            intent: ChoiceAnswer {
                choice: intent.clone(),
                probabilities: probs.clone(),
                confidence: calibrated_confidence(2, 0.9),
            },
            route: "deterministic".to_string(),
            addressing_computer: 0.9,
            ..Default::default()
        });
        let d = mock.decide(&req).await.unwrap();
        let r = alive_intent::route_decision(&d, 0.85, 0.6);
        // n=2 toy: peak 0.9 -> calibrated 0.8 -> Confirm. Assert the contract.
        if r.intent == "open_app" && r.route == alive_intent::Route::Confirm {
            steps.insert(
                "decide_mock".into(),
                serde_json::json!({"ok": true, "intent": r.intent, "route": "confirm"}),
            );
        } else {
            fail(
                &mut steps,
                &mut ok,
                "decide_mock",
                format!("{} {:?}", r.intent, r.route),
            );
        }

        if std::env::var("ALIVE_JEV_API_KEY").is_ok() {
            let t0 = std::time::Instant::now();
            let client = alive_intent::JevClient::from_env(
                "https://api.typesafe.ai/v1/systemone".to_string(),
                "jev-1.13.0".to_string(),
            )
            .unwrap();
            match client.decide(&req).await {
                Ok(jd) => {
                    let jr = alive_intent::route_decision(&jd, 0.85, 0.6);
                    steps.insert(
                        "decide_jev".into(),
                        serde_json::json!({"ok": true, "intent": jr.intent, "ms": t0.elapsed().as_secs_f64() * 1000.0}),
                    );
                }
                Err(e) => fail(&mut steps, &mut ok, "decide_jev", e.to_string()),
            }
        } else {
            steps.insert(
                "decide_jev".into(),
                serde_json::json!({"ok": true, "skipped": "no ALIVE_JEV_API_KEY"}),
            );
        }
    }

    // 5. Execute live (read-only tool) + journal + undo.
    {
        let dir = std::env::temp_dir().join(format!("alive-demo-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("demo-report.txt"), "x").unwrap();
        let jp = dir.join("j.jsonl");
        let mut journal = alive_journal::Journal::open(&jp).unwrap();
        let exec = alive_act::LocalExecutor::new(false, vec![]);
        let req = alive_act::ActionRequest {
            action_id: "demo1".into(),
            intent: "file_search".into(),
            tool: alive_act::ToolId::FileSearch,
            args: serde_json::json!({"query": "report", "root": dir}),
            risk_tier: 1,
        };
        let outcome = alive_act::execute_and_journal(&exec, &mut journal, &req)
            .await
            .unwrap();
        let back = alive_journal::Journal::read_all(&jp).unwrap();
        if outcome.summary().starts_with("1 hits") && back.len() == 1 {
            // Undo roundtrip on a scratch file.
            let victim = dir.join("v.txt");
            std::fs::write(&victim, "bye").unwrap();
            let op = alive_journal::UndoOp {
                op: "delete_file".into(),
                payload: serde_json::json!({"path": victim.to_string_lossy()}),
            };
            let u = alive_act::apply_undo(&op);
            if u.is_ok() && !victim.exists() {
                steps.insert(
                    "execute".into(),
                    serde_json::json!({"ok": true, "summary": outcome.summary()}),
                );
            } else {
                fail(
                    &mut steps,
                    &mut ok,
                    "execute",
                    format!("undo failed: {u:?}"),
                );
            }
        } else {
            fail(&mut steps, &mut ok, "execute", outcome.summary());
        }
    }

    // 6. Endpointing contract on a synthetic burst.
    {
        use alive_temporal::{EndpointConfig, EndpointEvent, Endpointer};
        let mut ep = Endpointer::new(EndpointConfig::default());
        let mut seen = (false, false);
        let mut t = 0u64;
        while t < 600 {
            ep.push(t, 0.5, 0.0, true);
            t += 33;
        }
        while t < 1600 {
            match ep.push(t, 0.001, 1.0, true) {
                Some(EndpointEvent::Tentative { .. }) => seen.0 = true,
                Some(EndpointEvent::Commit { .. }) => seen.1 = true,
                _ => {}
            }
            t += 33;
        }
        if seen == (true, true) {
            steps.insert("endpointing".into(), serde_json::json!({"ok": true}));
        } else {
            fail(&mut steps, &mut ok, "endpointing", format!("{seen:?}"));
        }
    }

    finish(ok, steps)
}

fn fail(
    steps: &mut serde_json::Map<String, serde_json::Value>,
    ok: &mut bool,
    step: &str,
    msg: String,
) {
    *ok = false;
    steps.insert(
        step.to_string(),
        serde_json::json!({"ok": false, "error": msg}),
    );
}

fn finish(ok: bool, steps: serde_json::Map<String, serde_json::Value>) -> ! {
    println!("{}", serde_json::json!({"ok": ok, "steps": steps}));
    std::process::exit(if ok { 0 } else { 1 })
}
