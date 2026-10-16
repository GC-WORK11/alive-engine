//! alive-intent: the decision layer.
//!
//! `DecisionProvider` is the seam: Jev by default, `MockProvider` in tests,
//! a function-calling LLM as the low-confidence fallback (later).
//! Jev wire format follows PRD §6: one call, all questions in parallel.
//! The response parser is tolerant of map-vs-array answer shapes because the
//! vendor schemas drift; unknown shapes are errors, never guesses.

use std::collections::HashMap;
use std::time::Duration;

use serde::Serialize;

pub mod tool_llm;

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct JevQuestion {
    #[serde(rename = "type")]
    pub qtype: String,
    pub key: String,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub levels: Option<u8>,
}

impl JevQuestion {
    pub fn choice(key: &str, text: &str, options: Vec<String>) -> Self {
        assert!(
            !options.is_empty() && options.len() <= 255,
            "choice needs 1..=255 options"
        );
        Self {
            qtype: "choice".into(),
            key: key.into(),
            text: text.into(),
            options: Some(options),
            levels: None,
        }
    }
    pub fn score(key: &str, text: &str, levels: u8) -> Self {
        assert!((2..=10).contains(&levels), "score needs 2..=10 levels");
        Self {
            qtype: "score".into(),
            key: key.into(),
            text: text.into(),
            options: None,
            levels: Some(levels),
        }
    }
    pub fn noul(key: &str, text: &str) -> Self {
        Self {
            qtype: "noul".into(),
            key: key.into(),
            text: text.into(),
            options: None,
            levels: None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct DecisionRequest {
    pub state: serde_json::Value,
    pub questions: Vec<JevQuestion>,
}

/// The standard ALive question set (PRD §6): intent + route + 4 Noul gates
/// + 2 Scores, always in ONE call.
pub fn alive_questions(intent_options: &[String]) -> Vec<JevQuestion> {
    vec![
        JevQuestion::choice("intent", "Which user intent does this signed utterance express?", intent_options.to_vec()),
        JevQuestion::choice("route", "Which execution route fits best?", vec!["deterministic".into(), "llm".into(), "human".into()]),
        JevQuestion::noul("addressing_computer", "The user is addressing the computer (not touching their face, not chatting with someone else)."),
        JevQuestion::noul("needs_confirmation", "This action needs explicit user confirmation before executing."),
        JevQuestion::noul("is_risky", "This action is risky, irreversible, or externally visible."),
        JevQuestion::noul("is_multi_step", "This request needs multiple sequential steps."),
        JevQuestion::score("confidence", "Confidence the top intent is correct.", 5),
        JevQuestion::score("risk", "Risk if executed right now.", 5),
    ]
}

// ---------------------------------------------------------------------------
// Parsed answers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct ChoiceAnswer {
    pub choice: String,
    pub probabilities: HashMap<String, f64>,
    pub confidence: f64,
}

#[derive(Debug, Clone, Default)]
pub struct ScoreAnswer {
    pub score: f64,
    pub confidence: f64,
}

/// Jev triage over generic tools (typed dispatch): one Choice over the
/// tool verbs plus the safety/multi-step gates. The option set is tools,
/// never tasks — this is what replaces the hardcoded intent catalog.
pub fn tool_dispatch_questions(tool_names: &[String]) -> Vec<JevQuestion> {
    let mut opts = tool_names.to_vec();
    opts.push("none".to_string());
    vec![
        JevQuestion::choice(
            "tool",
            "Which single tool call carries out this request? Pick none if it needs several steps or no tool fits.",
            opts,
        ),
        JevQuestion::noul("addressing_computer", "The user is addressing the computer (not touching their face, not chatting with someone else)."),
        JevQuestion::noul("needs_confirmation", "This action needs explicit user confirmation before executing."),
        JevQuestion::noul("is_risky", "This action is risky, irreversible, or externally visible."),
        JevQuestion::noul("is_multi_step", "This request needs multiple sequential steps."),
    ]
}

#[derive(Debug, Clone, Default)]
pub struct ToolDispatch {
    pub tool: ChoiceAnswer,
    pub addressing_computer: f64,
    pub needs_confirmation: f64,
    pub is_risky: f64,
    pub is_multi_step: f64,
}

pub fn dispatch_from_answers(
    map: &HashMap<String, serde_json::Value>,
) -> Result<ToolDispatch, IntentError> {
    let tool_v = map
        .get("tool")
        .ok_or_else(|| IntentError::Parse("missing tool answer".into()))?;
    let tool = parse_choice(tool_v);
    if tool.choice.is_empty() {
        return Err(IntentError::Parse("empty tool choice".into()));
    }
    Ok(ToolDispatch {
        tool,
        addressing_computer: noul_of(map, "addressing_computer"),
        needs_confirmation: noul_of(map, "needs_confirmation"),
        is_risky: noul_of(map, "is_risky"),
        is_multi_step: noul_of(map, "is_multi_step"),
    })
}

#[derive(Debug, Clone, Default)]
pub struct Decision {
    pub intent: ChoiceAnswer,
    pub route: String,
    pub addressing_computer: f64,
    pub needs_confirmation: f64,
    pub is_risky: f64,
    pub is_multi_step: f64,
    pub confidence_score: f64,
    pub risk_score: f64,
}

#[derive(Debug, thiserror::Error)]
pub enum IntentError {
    #[error("transport: {0}")]
    Transport(String),
    #[error("jev error {0}: {1}")]
    Api(String, String),
    #[error("parse: {0}")]
    Parse(String),
}

fn as_f64(v: &serde_json::Value) -> Option<f64> {
    match v {
        serde_json::Value::Number(n) => n.as_f64(),
        serde_json::Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }
}

/// Accept probabilities as a map OR an array of {option|label|choice, probability|p}.
fn parse_probs(v: &serde_json::Value) -> HashMap<String, f64> {
    let mut m = HashMap::new();
    match v {
        serde_json::Value::Object(o) => {
            for (k, pv) in o {
                if let Some(p) = as_f64(pv) {
                    m.insert(k.clone(), p);
                }
            }
        }
        serde_json::Value::Array(a) => {
            for e in a {
                let label = e
                    .get("option")
                    .or_else(|| e.get("label"))
                    .or_else(|| e.get("choice"))
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string();
                let p = e
                    .get("probability")
                    .or_else(|| e.get("p"))
                    .and_then(as_f64)
                    .unwrap_or(0.0);
                if !label.is_empty() {
                    m.insert(label, p);
                }
            }
        }
        _ => {}
    }
    m
}

fn parse_choice(v: &serde_json::Value) -> ChoiceAnswer {
    let choice = v
        .get("choice")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();
    let probabilities = v.get("probabilities").map(parse_probs).unwrap_or_default();
    let confidence = v
        .get("confidence")
        .and_then(as_f64)
        .unwrap_or_else(|| probabilities.values().copied().fold(0.0f64, f64::max));
    ChoiceAnswer {
        choice,
        probabilities,
        confidence,
    }
}

/// Tolerant Jev response parser. Handles:
/// (a) {"answers": {key: {...}}}  (b) {"answers": [{key, ...}]}  (c) flat {key: {...}}
pub fn parse_jev_response(
    body: &serde_json::Value,
) -> Result<HashMap<String, serde_json::Value>, IntentError> {
    let mut out = HashMap::new();
    if let Some(ans) = body.get("answers") {
        match ans {
            serde_json::Value::Object(o) => {
                for (k, v) in o {
                    out.insert(k.clone(), v.clone());
                }
            }
            serde_json::Value::Array(a) => {
                for e in a {
                    if let Some(k) = e.get("key").and_then(|x| x.as_str()) {
                        out.insert(k.to_string(), e.clone());
                    }
                }
            }
            _ => {
                return Err(IntentError::Parse(
                    "answers is neither map nor array".into(),
                ))
            }
        }
    } else if let Some(o) = body.as_object() {
        for (k, v) in o {
            out.insert(k.clone(), v.clone());
        }
    } else {
        return Err(IntentError::Parse("response is not a JSON object".into()));
    }
    if out.is_empty() {
        return Err(IntentError::Parse("no answers in response".into()));
    }
    Ok(out)
}

fn noul_of(map: &HashMap<String, serde_json::Value>, key: &str) -> f64 {
    map.get(key)
        .and_then(|v| v.get("noul").and_then(as_f64).or_else(|| as_f64(v)))
        .unwrap_or(0.0)
}

fn score_of(map: &HashMap<String, serde_json::Value>, key: &str) -> f64 {
    map.get(key)
        .and_then(|v| v.get("score").and_then(as_f64))
        .unwrap_or(0.0)
}

pub fn decision_from_answers(
    map: &HashMap<String, serde_json::Value>,
) -> Result<Decision, IntentError> {
    let intent_v = map
        .get("intent")
        .ok_or_else(|| IntentError::Parse("missing intent answer".into()))?;
    let intent = parse_choice(intent_v);
    if intent.choice.is_empty() {
        return Err(IntentError::Parse("empty intent choice".into()));
    }
    let route = map
        .get("route")
        .map(parse_choice)
        .map(|c| c.choice)
        .unwrap_or_else(|| "deterministic".to_string());
    Ok(Decision {
        intent,
        route,
        addressing_computer: noul_of(map, "addressing_computer"),
        needs_confirmation: noul_of(map, "needs_confirmation"),
        is_risky: noul_of(map, "is_risky"),
        is_multi_step: noul_of(map, "is_multi_step"),
        confidence_score: score_of(map, "confidence"),
        risk_score: score_of(map, "risk"),
    })
}

// ---------------------------------------------------------------------------
// Provider trait + Jev client + mock
// ---------------------------------------------------------------------------

// Native async fn: providers are used as generics, never as trait objects,
// so the missing auto-trait bounds do not matter here.
#[allow(async_fn_in_trait)]
pub trait DecisionProvider: Send + Sync {
    async fn decide(&self, req: &DecisionRequest) -> Result<Decision, IntentError>;
    fn name(&self) -> &'static str;
}

#[derive(Debug, Clone)]
pub struct JevClient {
    endpoint: String,
    model: String,
    api_key: String,
    timeout: Duration,
    http: reqwest::Client,
}

impl JevClient {
    pub fn new(endpoint: String, model: String, api_key: String) -> Result<Self, IntentError> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .user_agent("alive/0.1")
            .build()
            .map_err(|e| IntentError::Transport(e.to_string()))?;
        Ok(Self {
            endpoint,
            model,
            api_key,
            timeout: Duration::from_secs(10),
            http,
        })
    }

    pub fn from_env(endpoint: String, model: String) -> Result<Self, IntentError> {
        let key = std::env::var("ALIVE_JEV_API_KEY")
            .map_err(|_| IntentError::Transport("ALIVE_JEV_API_KEY not set".into()))?;
        Self::new(endpoint, model, key)
    }

    pub fn request_body(&self, req: &DecisionRequest) -> serde_json::Value {
        serde_json::json!({
            "model": self.model,
            "state": req.state,
            "questions": req.questions,
        })
    }

    async fn post(&self, req: &DecisionRequest) -> Result<serde_json::Value, IntentError> {
        let body = self.request_body(req);
        let resp = self
            .http
            .post(&self.endpoint)
            .bearer_auth(&self.api_key)
            .json(&body)
            .timeout(self.timeout)
            .send()
            .await
            .map_err(|e| IntentError::Transport(e.to_string()))?;
        let status = resp.status();
        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| IntentError::Parse(format!("invalid JSON: {e}")))?;
        if !status.is_success() {
            return Err(IntentError::Api(status.to_string(), json.to_string()));
        }
        Ok(json)
    }

    /// Typed tool dispatch: same wire call, parsed as a tool triage
    /// (for [`tool_dispatch_questions`]) instead of an intent decision.
    pub async fn decide_dispatch(
        &self,
        req: &DecisionRequest,
    ) -> Result<ToolDispatch, IntentError> {
        let json = self.post(req).await?;
        dispatch_from_answers(&parse_jev_response(&json)?)
    }
}

impl DecisionProvider for JevClient {
    async fn decide(&self, req: &DecisionRequest) -> Result<Decision, IntentError> {
        let json = self.post(req).await?;
        decision_from_answers(&parse_jev_response(&json)?)
    }

    fn name(&self) -> &'static str {
        "jev"
    }
}

/// Scripted provider for tests, demos, and no-key development.
pub struct MockProvider {
    f: Box<dyn Fn(&DecisionRequest) -> Decision + Send + Sync>,
}

impl MockProvider {
    pub fn from_fn(f: impl Fn(&DecisionRequest) -> Decision + Send + Sync + 'static) -> Self {
        Self { f: Box::new(f) }
    }

    /// Always returns this intent at the given confidence, deterministic route.
    pub fn fixed(intent: &str, confidence: f64) -> Self {
        let intent = intent.to_string();
        Self::from_fn(move |_| Decision {
            intent: ChoiceAnswer {
                choice: intent.clone(),
                probabilities: HashMap::from([
                    (intent.clone(), confidence),
                    ("dictate".into(), 1.0 - confidence),
                ]),
                confidence,
            },
            route: "deterministic".to_string(),
            addressing_computer: 1.0,
            ..Default::default()
        })
    }
}

impl DecisionProvider for MockProvider {
    async fn decide(&self, req: &DecisionRequest) -> Result<Decision, IntentError> {
        Ok((self.f)(req))
    }
    fn name(&self) -> &'static str {
        "mock"
    }
}

// ---------------------------------------------------------------------------
// Router (code, not model)
// ---------------------------------------------------------------------------

/// Chance-corrected peak: conf = (n·peak − 1)/(n − 1).
pub fn calibrated_confidence(n_opts: usize, peak_p: f64) -> f64 {
    if n_opts <= 1 {
        return peak_p.clamp(0.0, 1.0);
    }
    ((n_opts as f64 * peak_p - 1.0) / (n_opts as f64 - 1.0)).clamp(0.0, 1.0)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Route {
    Act,
    Confirm,
    Ask,
}

pub fn route(confidence: f64, act_t: f64, confirm_t: f64) -> Route {
    if confidence >= act_t {
        Route::Act
    } else if confidence >= confirm_t {
        Route::Confirm
    } else {
        Route::Ask
    }
}

#[derive(Debug, Clone)]
pub struct Routing {
    pub intent: String,
    pub confidence: f64,
    pub route: Route,
    pub addressing_computer: bool,
    pub needs_confirmation: bool,
    pub is_risky: bool,
    pub is_multi_step: bool,
    pub provider_route: String,
}

pub fn route_decision(d: &Decision, act_t: f64, confirm_t: f64) -> Routing {
    let peak = d
        .intent
        .probabilities
        .values()
        .copied()
        .fold(d.intent.confidence, f64::max);
    let n = d.intent.probabilities.len().max(1);
    let mut conf = calibrated_confidence(n, peak);
    // Vendor confidence and calibrated peak must agree; take the pessimist.
    conf = conf.min(d.intent.confidence);
    let mut r = route(conf, act_t, confirm_t);
    let needs_confirmation = d.needs_confirmation >= 0.5;
    let is_risky = d.is_risky >= 0.5;
    if (needs_confirmation || is_risky || d.risk_score >= 4.0) && r == Route::Act {
        r = Route::Confirm;
    }
    Routing {
        intent: d.intent.choice.clone(),
        confidence: conf,
        route: r,
        addressing_computer: d.addressing_computer >= 0.6,
        needs_confirmation,
        is_risky,
        is_multi_step: d.is_multi_step >= 0.5,
        provider_route: d.route.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conf_formula_matches_prd() {
        assert!((calibrated_confidence(2, 1.0) - 1.0).abs() < 1e-9);
        assert!((calibrated_confidence(2, 0.5) - 0.0).abs() < 1e-9);
        // PRD example: n=6, peak=0.81 -> 0.772
        assert!((calibrated_confidence(6, 0.81) - 0.772).abs() < 1e-9);
        assert_eq!(calibrated_confidence(1, 0.7), 0.7);
    }

    #[test]
    fn route_thresholds() {
        assert_eq!(route(0.9, 0.85, 0.6), Route::Act);
        assert_eq!(route(0.85, 0.85, 0.6), Route::Act);
        assert_eq!(route(0.7, 0.85, 0.6), Route::Confirm);
        assert_eq!(route(0.6, 0.85, 0.6), Route::Confirm);
        assert_eq!(route(0.59, 0.85, 0.6), Route::Ask);
    }

    #[test]
    fn questions_shape_is_prd_section_6() {
        let opts = vec!["a".to_string(), "b".to_string()];
        let q = alive_questions(&opts);
        assert_eq!(q.len(), 8);
        assert_eq!(q[0].qtype, "choice");
        assert_eq!(q[0].options.as_ref().unwrap().len(), 2);
        assert_eq!(q[2].qtype, "noul");
        assert_eq!(q[6].levels, Some(5));
    }

    #[test]
    fn dispatch_questions_cover_tools_plus_none() {
        let tools = vec!["open_url".to_string(), "type_text".to_string()];
        let q = tool_dispatch_questions(&tools);
        assert_eq!(q.len(), 5);
        assert_eq!(q[0].qtype, "choice");
        let opts = q[0].options.as_ref().unwrap();
        assert_eq!(opts.len(), 3);
        assert!(opts.contains(&"none".to_string()));
        let body = serde_json::json!({
            "answers": {
                "tool": {"choice": "open_url",
                         "probabilities": {"open_url": 0.9, "none": 0.1},
                         "confidence": 0.9},
                "addressing_computer": {"noul": 0.95},
                "is_multi_step": {"noul": 0.1}
            }
        });
        let d = dispatch_from_answers(&parse_jev_response(&body).unwrap()).unwrap();
        assert_eq!(d.tool.choice, "open_url");
        assert!((d.addressing_computer - 0.95).abs() < 1e-9);
        assert!(parse_jev_response(&serde_json::json!({"answers": {}})).is_err());
        let no_tool = serde_json::json!({"answers": {"route": {"choice": "llm"}}});
        assert!(dispatch_from_answers(&parse_jev_response(&no_tool).unwrap()).is_err());
    }

    #[test]
    fn parse_accepts_map_array_and_flat() {
        let map_form = serde_json::json!({
            "answers": {
                "intent": {"choice": "open_app", "probabilities": {"open_app": 0.9, "dictate": 0.1}, "confidence": 0.9},
                "addressing_computer": {"noul": 1.0}
            }
        });
        let d = decision_from_answers(&parse_jev_response(&map_form).unwrap()).unwrap();
        assert_eq!(d.intent.choice, "open_app");

        let array_form = serde_json::json!({
            "answers": [
                {"key": "intent", "choice": "open_app", "probabilities": [{"option": "open_app", "probability": 0.9}], "confidence": 0.9},
                {"key": "addressing_computer", "noul": 0.2}
            ]
        });
        let d = decision_from_answers(&parse_jev_response(&array_form).unwrap()).unwrap();
        assert_eq!(d.intent.choice, "open_app");
        assert!((d.addressing_computer - 0.2).abs() < 1e-9);

        let flat = serde_json::json!({
            "intent": {"choice": "cancel", "probabilities": {"cancel": 1.0}, "confidence": 1.0}
        });
        let d = decision_from_answers(&parse_jev_response(&flat).unwrap()).unwrap();
        assert_eq!(d.intent.choice, "cancel");
    }

    #[test]
    fn parse_rejects_empty_and_missing_intent() {
        assert!(parse_jev_response(&serde_json::json!({"answers": {}})).is_err());
        let no_intent = serde_json::json!({"answers": {"route": {"choice": "llm"}}});
        let m = parse_jev_response(&no_intent).unwrap();
        assert!(decision_from_answers(&m).is_err());
    }

    #[test]
    fn risky_downgrades_act_to_confirm() {
        let d = Decision {
            intent: ChoiceAnswer {
                choice: "send_email".into(),
                probabilities: HashMap::from([
                    ("send_email".into(), 0.95),
                    ("dictate".into(), 0.05),
                ]),
                confidence: 0.95,
            },
            is_risky: 1.0,
            ..Default::default()
        };
        let r = route_decision(&d, 0.85, 0.6);
        assert_eq!(r.route, Route::Confirm);
        assert!(r.is_risky);
    }

    #[tokio::test]
    async fn mock_provider_roundtrip() {
        // n=2 toy distribution: peak 0.95 -> calibrated (2*.95-1)/1 = 0.90 -> Act.
        let m = MockProvider::fixed("open_app", 0.95);
        let req = DecisionRequest {
            state: serde_json::json!({"gloss": "OPEN BROWSER"}),
            questions: alive_questions(&["open_app".into(), "dictate".into()]),
        };
        let d = m.decide(&req).await.unwrap();
        let r = route_decision(&d, 0.85, 0.6);
        assert_eq!(r.intent, "open_app");
        assert_eq!(r.route, Route::Act);
    }

    /// Real wire-format verification without a Jev key: a tiny stub HTTP
    /// server captures the POST and returns canned answers.
    #[tokio::test]
    async fn jev_client_posts_prd_shape_to_stub() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut s, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 16384];
            let n = s.read(&mut buf).await.unwrap();
            let raw = String::from_utf8_lossy(&buf[..n]).to_string();
            let body_str = raw.split("\r\n\r\n").nth(1).unwrap_or("").to_string();
            let canned = serde_json::json!({
                "answers": {
                    "intent": {"choice": "open_app", "probabilities": {"open_app": 0.92, "dictate": 0.08}, "confidence": 0.92},
                    "route": {"choice": "deterministic", "probabilities": {"deterministic": 1.0}, "confidence": 1.0},
                    "addressing_computer": {"noul": 0.99}
                }
            })
            .to_string();
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                canned.len(), canned
            );
            s.write_all(resp.as_bytes()).await.unwrap();
            raw.lines().next().unwrap_or("").to_string()
                + "\nAUTH:"
                + &raw
                    .lines()
                    .find(|l| l.to_lowercase().starts_with("authorization:"))
                    .unwrap_or("")
                    .to_string()
                + "\nBODY:"
                + &body_str
        });

        let client = JevClient::new(
            format!("http://{addr}/v1/systemone"),
            "jev-1.13.0".into(),
            "test-key".into(),
        )
        .unwrap();
        let req = DecisionRequest {
            state: serde_json::json!({"gloss_nbest": [{"gloss": "OPEN BROWSER", "p": 0.9}]}),
            questions: alive_questions(&["open_app".into(), "dictate".into()]),
        };
        let d = client.decide(&req).await.expect("stub roundtrip");
        assert_eq!(d.intent.choice, "open_app");
        assert!(d.addressing_computer > 0.9);

        let captured = server.await.unwrap();
        assert!(
            captured.starts_with("POST /v1/systemone"),
            "request line: {captured}"
        );
        assert!(captured.contains("Bearer test-key"), "auth: {captured}");
        let body_json: serde_json::Value =
            serde_json::from_str(captured.split("\nBODY:").nth(1).unwrap()).unwrap();
        assert_eq!(body_json["model"], "jev-1.13.0");
        assert_eq!(body_json["questions"].as_array().unwrap().len(), 8);
        assert!(body_json["state"]["gloss_nbest"].is_array());
    }

    #[tokio::test]
    async fn decide_dispatch_parses_tool_triage() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut s, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 16384];
            let n = s.read(&mut buf).await.unwrap();
            let raw = String::from_utf8_lossy(&buf[..n]).to_string();
            let body_str = raw.split("\r\n\r\n").nth(1).unwrap_or("").to_string();
            let canned = serde_json::json!({
                "answers": {
                    "tool": {"choice": "open_url",
                             "probabilities": {"open_url": 0.88, "none": 0.12},
                             "confidence": 0.88},
                    "addressing_computer": {"noul": 0.97},
                    "is_multi_step": {"noul": 0.05}
                }
            })
            .to_string();
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                canned.len(), canned
            );
            s.write_all(resp.as_bytes()).await.unwrap();
            body_str
        });

        let client = JevClient::new(
            format!("http://{addr}/v1/systemone"),
            "jev-1.13.0".into(),
            "test-key".into(),
        )
        .unwrap();
        let req = DecisionRequest {
            state: serde_json::json!({"text": "open example.com"}),
            questions: tool_dispatch_questions(&["open_url".into(), "type_text".into()]),
        };
        let d = client.decide_dispatch(&req).await.expect("stub roundtrip");
        assert_eq!(d.tool.choice, "open_url");
        assert!(d.addressing_computer > 0.9);

        let body_json: serde_json::Value = serde_json::from_str(&server.await.unwrap()).unwrap();
        assert_eq!(body_json["questions"].as_array().unwrap().len(), 5);
        assert_eq!(body_json["questions"][0]["key"], "tool");
    }
}
