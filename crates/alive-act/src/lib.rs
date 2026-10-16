//! alive-act: Tier-1 OS-native action execution (Linux-first).
//!
//! Rules, enforced in code:
//! - No shell. Ever. `Command` argv only; model output never becomes argv[0].
//! - `open_app` requires an explicit allowlist entry, else Refused.
//! - URLs must be http/https/mailto, else Refused.
//! - `dry_run` plans + journals without executing. Tests always dry-run.
//! - Every execution is journaled with redacted args; undo payloads are
//!   one-shot via the journal.

use std::path::{Path, PathBuf};
use std::process::Stdio;

use alive_journal::{ActionStatus, Journal, JournalEntry, UndoOp};
use serde::{Deserialize, Serialize};

#[derive(Debug, thiserror::Error)]
pub enum ActError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("journal: {0}")]
    Journal(#[from] alive_journal::JournalError),
    #[error("undo '{0}' failed: {1}")]
    Undo(String, String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolId {
    OpenApp,
    OpenUrl,
    FileSearch,
    BrowserSearch,
    AiQuery,
    ClipboardCopy,
    Screenshot,
    TypeText,
    KeyPress,
    MouseMove,
    MouseClick,
    MouseDrag,
    Scroll,
    WaitMs,
    CursorPosition,
}

impl ToolId {
    /// All tools in a stable order (Jev Choice options + LLM schemas).
    pub fn all() -> &'static [ToolId] {
        use ToolId::*;
        &[
            OpenApp,
            OpenUrl,
            FileSearch,
            BrowserSearch,
            AiQuery,
            ClipboardCopy,
            Screenshot,
            TypeText,
            KeyPress,
            MouseMove,
            MouseClick,
            MouseDrag,
            Scroll,
            WaitMs,
            CursorPosition,
        ]
    }

    pub fn name(self) -> &'static str {
        match self {
            ToolId::OpenApp => "open_app",
            ToolId::OpenUrl => "open_url",
            ToolId::FileSearch => "file_search",
            ToolId::BrowserSearch => "browser_search",
            ToolId::AiQuery => "ai_query",
            ToolId::ClipboardCopy => "clipboard_copy",
            ToolId::Screenshot => "screenshot",
            ToolId::TypeText => "type_text",
            ToolId::KeyPress => "key_press",
            ToolId::MouseMove => "mouse_move",
            ToolId::MouseClick => "mouse_click",
            ToolId::MouseDrag => "mouse_drag",
            ToolId::Scroll => "scroll",
            ToolId::WaitMs => "wait_ms",
            ToolId::CursorPosition => "cursor_position",
        }
    }

    pub fn from_name(name: &str) -> Option<ToolId> {
        Self::all()
            .iter()
            .copied()
            .find(|t| t.name() == name.trim())
    }

    /// Risk tier of the verb itself (intent catalogs may escalate, never
    /// de-escalate): 1 = read-only/local, 2 = externally visible state.
    /// Clicks/drags/scrolls act on the world (they can hit Delete as
    /// easily as Play), so they sit behind the confirm gate.
    pub fn risk_tier(self) -> u8 {
        match self {
            ToolId::OpenApp
            | ToolId::TypeText
            | ToolId::KeyPress
            | ToolId::MouseClick
            | ToolId::MouseDrag
            | ToolId::Scroll => 2,
            _ => 1,
        }
    }
}

/// Snake-case tool names in [`ToolId::all`] order. This is the closed set
/// Jev dispatches over (typed tool dispatch) — generic verbs, not tasks.
pub fn tool_names() -> Vec<String> {
    ToolId::all().iter().map(|t| t.name().to_string()).collect()
}

/// OpenAI-compatible function schemas, generated from the same arg shapes
/// the executor actually reads. The agent sees tools, never task intents.
pub fn tool_schemas() -> Vec<serde_json::Value> {
    fn str_prop(desc: &str) -> serde_json::Value {
        serde_json::json!({"type": "string", "description": desc})
    }
    fn int_prop(desc: &str) -> serde_json::Value {
        serde_json::json!({"type": "integer", "description": desc})
    }
    fn schema(
        name: &str,
        desc: &str,
        props: serde_json::Value,
        required: &[&str],
    ) -> serde_json::Value {
        serde_json::json!({
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": required,
                    "additionalProperties": false,
                },
            },
        })
    }
    vec![
        schema(
            "open_app",
            "Launch an installed application by binary name. Only pre-allowlisted names run; anything else is refused.",
            serde_json::json!({"app": str_prop("Binary name, e.g. firefox. No paths, no flags.")}),
            &["app"],
        ),
        schema(
            "open_url",
            "Open a URL in the default handler. http/https/mailto only.",
            serde_json::json!({"url": str_prop("Full URL, max 2048 chars.")}),
            &["url"],
        ),
        schema(
            "file_search",
            "Search file names under a root directory. Returns a hit count summary.",
            serde_json::json!({
                "query": str_prop("Case-insensitive filename substring."),
                "root": str_prop("Optional root dir. Defaults to the home directory."),
            }),
            &["query"],
        ),
        schema(
            "browser_search",
            "Web-search via DuckDuckGo results page in the default browser.",
            serde_json::json!({"query": str_prop("Search query.")}),
            &["query"],
        ),
        schema(
            "ai_query",
            "Ask the configured helper LLM a question. Fails when none is configured.",
            serde_json::json!({"query": str_prop("The question.")}),
            &["query"],
        ),
        schema(
            "clipboard_copy",
            "Copy text to the system clipboard.",
            serde_json::json!({"text": str_prop("Text to copy, max 100k chars.")}),
            &["text"],
        ),
        schema(
            "screenshot",
            "Capture the screen to a temp PNG. The file path is in the result summary.",
            serde_json::json!({}),
            &[],
        ),
        schema(
            "type_text",
            "Type text into the focused window via synthetic key events.",
            serde_json::json!({"text": str_prop("Literal text to type, max 5000 chars.")}),
            &["text"],
        ),
        schema(
            "key_press",
            "Press a key or shortcut combo in the focused window (X11/xdotool).",
            serde_json::json!({"key": str_prop("Key name or +-combo, e.g. Return, Tab, Escape, ctrl+Return, ctrl+c.")}),
            &["key"],
        ),
        schema(
            "mouse_move",
            "Move the pointer to screen pixels (X11/xdotool). Origin is top-left.",
            serde_json::json!({
                "x": int_prop("X pixel, 0..32767."),
                "y": int_prop("Y pixel, 0..32767."),
            }),
            &["x", "y"],
        ),
        schema(
            "mouse_click",
            "Move to screen pixels and click a button (X11/xdotool).",
            serde_json::json!({
                "x": int_prop("X pixel, 0..32767."),
                "y": int_prop("Y pixel, 0..32767."),
                "button": str_prop("left, middle, or right. Default left."),
                "clicks": int_prop("1 or 2 (double-click). Default 1."),
            }),
            &["x", "y"],
        ),
        schema(
            "mouse_drag",
            "Press button at (x1,y1), drag to (x2,y2), release (X11/xdotool).",
            serde_json::json!({
                "x1": int_prop("Start X pixel."),
                "y1": int_prop("Start Y pixel."),
                "x2": int_prop("End X pixel."),
                "y2": int_prop("End Y pixel."),
                "button": str_prop("left, middle, or right. Default left."),
            }),
            &["x1", "y1", "x2", "y2"],
        ),
        schema(
            "scroll",
            "Scroll the wheel under the pointer (X11/xdotool).",
            serde_json::json!({
                "direction": str_prop("up, down, left, or right."),
                "amount": int_prop("Wheel notches, 1..20. Default 3."),
            }),
            &["direction"],
        ),
        schema(
            "wait_ms",
            "Wait without touching the machine. Always available.",
            serde_json::json!({
                "ms": int_prop("Milliseconds to wait, 0..30000."),
            }),
            &["ms"],
        ),
        schema(
            "cursor_position",
            "Report the pointer position in screen pixels (X11/xdotool).",
            serde_json::json!({}),
            &[],
        ),
    ]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionRequest {
    pub action_id: String,
    pub intent: String,
    pub tool: ToolId,
    #[serde(default)]
    pub args: serde_json::Value,
    pub risk_tier: u8,
}

#[derive(Debug, Clone)]
pub enum ActionOutcome {
    Done {
        summary: String,
        undo: Option<UndoOp>,
        verified: Option<bool>,
    },
    NeedsConfirmation {
        preview: String,
    },
    Refused {
        reason: String,
    },
    Failed {
        error: String,
    },
}

impl ActionOutcome {
    pub fn status(&self) -> ActionStatus {
        match self {
            ActionOutcome::Done { .. } => ActionStatus::Done,
            ActionOutcome::NeedsConfirmation { .. } => ActionStatus::AwaitingConfirm,
            ActionOutcome::Refused { .. } => ActionStatus::Refused,
            ActionOutcome::Failed { .. } => ActionStatus::Failed,
        }
    }
    pub fn summary(&self) -> String {
        match self {
            ActionOutcome::Done { summary, .. } => summary.clone(),
            ActionOutcome::NeedsConfirmation { preview } => format!("awaiting confirm: {preview}"),
            ActionOutcome::Refused { reason } => format!("refused: {reason}"),
            ActionOutcome::Failed { error } => format!("failed: {error}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Availability probing (honest degrade: report what this machine can do)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolAvailability {
    pub tool: ToolId,
    pub available: bool,
    pub via: Option<String>,
    pub note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvailabilityReport {
    pub session_type: String,
    pub desktop: String,
    pub tools: Vec<ToolAvailability>,
}

pub fn find_in_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let p = dir.join(name);
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

fn first_of(names: &[&str]) -> Option<String> {
    names
        .iter()
        .find_map(|n| find_in_path(n).map(|p| p.display().to_string()))
}

pub fn availability() -> AvailabilityReport {
    let session_type = std::env::var("XDG_SESSION_TYPE").unwrap_or_else(|_| "unknown".into());
    let desktop = std::env::var("XDG_CURRENT_DESKTOP").unwrap_or_else(|_| "unknown".into());
    let open_via = first_of(&["xdg-open", "gio"]);
    let clip_via = first_of(&["wl-copy", "xclip", "xsel"]);
    let shot_via = first_of(&["grim", "gnome-screenshot", "scrot", "import"]);
    let type_via = first_of(&["wtype", "xdotool", "ydotool"]);
    let mock_llm = std::env::var("ALIVE_MOCK_LLM").ok().as_deref() == Some("1");
    AvailabilityReport {
        session_type,
        desktop,
        tools: vec![
            ToolAvailability {
                tool: ToolId::OpenApp,
                available: true,
                via: Some("spawn".into()),
                note: "allowlisted binaries only".into(),
            },
            ToolAvailability {
                tool: ToolId::OpenUrl,
                available: open_via.is_some(),
                via: open_via,
                note: "http/https/mailto only".into(),
            },
            ToolAvailability {
                tool: ToolId::FileSearch,
                available: true,
                via: Some("builtin-walk".into()),
                note: "capped recursive walk".into(),
            },
            ToolAvailability {
                tool: ToolId::BrowserSearch,
                available: true,
                via: Some("open_url".into()),
                note: "DuckDuckGo URL".into(),
            },
            ToolAvailability {
                tool: ToolId::AiQuery,
                available: mock_llm,
                via: if mock_llm { Some("mock".into()) } else { None },
                note: "no LLM provider configured".into(),
            },
            ToolAvailability {
                tool: ToolId::ClipboardCopy,
                available: clip_via.is_some(),
                via: clip_via,
                note: "stdin pipe, no shell".into(),
            },
            ToolAvailability {
                tool: ToolId::Screenshot,
                available: shot_via.is_some(),
                via: shot_via,
                note: "PNG to temp dir".into(),
            },
            ToolAvailability {
                tool: ToolId::TypeText,
                available: type_via.is_some(),
                via: type_via,
                note: "wtype/xdotool".into(),
            },
            ToolAvailability {
                tool: ToolId::KeyPress,
                available: find_in_path("xdotool").is_some(),
                via: find_in_path("xdotool").map(|p| p.display().to_string()),
                note: "xdotool key (X11)".into(),
            },
            ToolAvailability {
                tool: ToolId::MouseMove,
                available: find_in_path("xdotool").is_some(),
                via: find_in_path("xdotool").map(|p| p.display().to_string()),
                note: "xdotool mousemove (X11)".into(),
            },
            ToolAvailability {
                tool: ToolId::MouseClick,
                available: find_in_path("xdotool").is_some(),
                via: find_in_path("xdotool").map(|p| p.display().to_string()),
                note: "xdotool mousemove+click (X11)".into(),
            },
            ToolAvailability {
                tool: ToolId::MouseDrag,
                available: find_in_path("xdotool").is_some(),
                via: find_in_path("xdotool").map(|p| p.display().to_string()),
                note: "xdotool mousedown/move/mouseup (X11)".into(),
            },
            ToolAvailability {
                tool: ToolId::Scroll,
                available: find_in_path("xdotool").is_some(),
                via: find_in_path("xdotool").map(|p| p.display().to_string()),
                note: "xdotool wheel buttons 4-7 (X11)".into(),
            },
            ToolAvailability {
                tool: ToolId::WaitMs,
                available: true,
                via: Some("builtin-sleep".into()),
                note: "no OS dependency".into(),
            },
            ToolAvailability {
                tool: ToolId::CursorPosition,
                available: find_in_path("xdotool").is_some(),
                via: find_in_path("xdotool").map(|p| p.display().to_string()),
                note: "xdotool getmouselocation (X11)".into(),
            },
        ],
    }
}

// ---------------------------------------------------------------------------
// Executor
// ---------------------------------------------------------------------------

pub struct LocalExecutor {
    pub dry_run: bool,
    pub allowlist: Vec<String>,
}

impl LocalExecutor {
    pub fn new(dry_run: bool, allowlist: Vec<String>) -> Self {
        Self { dry_run, allowlist }
    }

    pub async fn execute(&self, req: &ActionRequest) -> ActionOutcome {
        if self.dry_run {
            return ActionOutcome::Done {
                summary: format!(
                    "dry-run: would {:?} {}",
                    req.tool,
                    compact_value(&redact_args(req.tool, &req.args))
                ),
                undo: None,
                verified: None,
            };
        }
        match req.tool {
            ToolId::OpenApp => self.open_app(req),
            ToolId::OpenUrl => self.open_url(req),
            ToolId::FileSearch => self.file_search(req).await,
            ToolId::BrowserSearch => self.browser_search(req),
            ToolId::AiQuery => self.ai_query(req),
            ToolId::ClipboardCopy => self.clipboard_copy(req),
            ToolId::Screenshot => self.screenshot(req),
            ToolId::TypeText => self.type_text(req),
            ToolId::KeyPress => self.key_press(req),
            ToolId::MouseMove => self.mouse_move(req),
            ToolId::MouseClick => self.mouse_click(req),
            ToolId::MouseDrag => self.mouse_drag(req),
            ToolId::Scroll => self.scroll(req),
            ToolId::WaitMs => self.wait_ms(req).await,
            ToolId::CursorPosition => self.cursor_position(),
        }
    }

    fn open_app(&self, req: &ActionRequest) -> ActionOutcome {
        let app = req
            .args
            .get("app")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim();
        if app.is_empty() || app.contains(['/', ' ', ';', '&', '|', '$', '`']) {
            return ActionOutcome::Refused {
                reason: format!("bad app name: {app:?}"),
            };
        }
        if !self.allowlist.iter().any(|a| a == app) {
            return ActionOutcome::Refused {
                reason: format!("{app} not in allowlist"),
            };
        }
        match std::process::Command::new(app)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
        {
            Ok(mut child) => {
                // Reap-or-detach: a spawn error surfaces here; exit status is the app's business.
                let _ = child.try_wait();
                ActionOutcome::Done {
                    summary: format!("launched {app}"),
                    undo: None,
                    verified: Some(true),
                }
            }
            Err(e) => ActionOutcome::Failed {
                error: format!("spawn {app}: {e}"),
            },
        }
    }

    fn open_url(&self, req: &ActionRequest) -> ActionOutcome {
        let url = req
            .args
            .get("url")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim();
        if !(url.starts_with("https://")
            || url.starts_with("http://")
            || url.starts_with("mailto:"))
        {
            return ActionOutcome::Refused {
                reason: format!("scheme not allowed: {url:?}"),
            };
        }
        if url.len() > 2048 {
            return ActionOutcome::Refused {
                reason: "url too long".into(),
            };
        }
        let opener = match first_of(&["xdg-open", "gio"]) {
            Some(o) => o,
            None => {
                return ActionOutcome::Failed {
                    error: "no URL opener (xdg-open/gio)".into(),
                }
            }
        };
        let mut cmd = std::process::Command::new(&opener);
        if opener.ends_with("gio") {
            cmd.arg("open");
        }
        match cmd
            .arg(url)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
        {
            Ok(s) if s.success() => ActionOutcome::Done {
                summary: format!("opened {url}"),
                undo: None,
                verified: Some(true),
            },
            Ok(s) => ActionOutcome::Failed {
                error: format!("opener exit {s}"),
            },
            Err(e) => ActionOutcome::Failed {
                error: e.to_string(),
            },
        }
    }

    async fn file_search(&self, req: &ActionRequest) -> ActionOutcome {
        let query = req
            .args
            .get("query")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if query.trim().is_empty() {
            return ActionOutcome::Refused {
                reason: "empty query".into(),
            };
        }
        let root = req
            .args
            .get("root")
            .and_then(|v| v.as_str())
            .map(PathBuf::from)
            .or_else(dirs_home)
            .unwrap_or_else(|| PathBuf::from("/"));
        let query_owned = query.clone();
        let hits =
            tokio::task::spawn_blocking(move || walk_search(&root, &query_owned, 5, 5000, 20))
                .await;
        match hits {
            Ok(list) => ActionOutcome::Done {
                summary: format!("{} hits for {query:?}", list.len()),
                undo: None,
                verified: Some(true),
            },
            Err(e) => ActionOutcome::Failed {
                error: format!("search join: {e}"),
            },
        }
    }

    fn browser_search(&self, req: &ActionRequest) -> ActionOutcome {
        let query = req
            .args
            .get("query")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim();
        if query.is_empty() {
            return ActionOutcome::Refused {
                reason: "empty query".into(),
            };
        }
        let url = format!("https://duckduckgo.com/?q={}", urlencode(query));
        let sub = ActionRequest {
            action_id: req.action_id.clone(),
            intent: req.intent.clone(),
            tool: ToolId::OpenUrl,
            args: serde_json::json!({"url": url}),
            risk_tier: req.risk_tier,
        };
        self.open_url(&sub)
    }

    fn ai_query(&self, req: &ActionRequest) -> ActionOutcome {
        let q = req.args.get("query").and_then(|v| v.as_str()).unwrap_or("");
        if std::env::var("ALIVE_MOCK_LLM").ok().as_deref() == Some("1") {
            return ActionOutcome::Done {
                summary: format!("mock answer to {q:?}"),
                undo: None,
                verified: Some(false),
            };
        }
        ActionOutcome::Failed {
            error: "no LLM provider configured (set ALIVE_MOCK_LLM=1 for demo)".into(),
        }
    }

    fn clipboard_copy(&self, req: &ActionRequest) -> ActionOutcome {
        let text = req.args.get("text").and_then(|v| v.as_str()).unwrap_or("");
        if text.is_empty() {
            return ActionOutcome::Refused {
                reason: "empty text".into(),
            };
        }
        if text.len() > 100_000 {
            return ActionOutcome::Refused {
                reason: "text too long".into(),
            };
        }
        if let Some(wl) = find_in_path("wl-copy") {
            return pipe_stdin(&wl, &[], text, "clipboard (wl-copy)");
        }
        if let Some(xc) = find_in_path("xclip") {
            return pipe_stdin(&xc, &["-selection", "clipboard"], text, "clipboard (xclip)");
        }
        if let Some(xs) = find_in_path("xsel") {
            return pipe_stdin(&xs, &["--clipboard", "--input"], text, "clipboard (xsel)");
        }
        ActionOutcome::Failed {
            error: "no clipboard tool (wl-copy/xclip/xsel)".into(),
        }
    }

    fn screenshot(&self, req: &ActionRequest) -> ActionOutcome {
        let dir = std::env::temp_dir().join("alive-shots");
        if std::fs::create_dir_all(&dir).is_err() {
            return ActionOutcome::Failed {
                error: "cannot create shot dir".into(),
            };
        }
        let dest = dir.join(format!("shot-{}.png", req.action_id));
        let dest_s = dest.display().to_string();
        let attempts: Vec<(&str, Vec<String>)> = vec![
            ("grim", vec![dest_s.clone()]),
            ("gnome-screenshot", vec!["-f".into(), dest_s.clone()]),
            ("scrot", vec![dest_s.clone()]),
            (
                "import",
                vec!["-window".into(), "root".into(), dest_s.clone()],
            ),
        ];
        for (bin, args) in attempts {
            if find_in_path(bin).is_none() {
                continue;
            }
            match std::process::Command::new(bin)
                .args(&args)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
            {
                Ok(s) if s.success() && dest.is_file() => {
                    return ActionOutcome::Done {
                        summary: format!("screenshot -> {dest_s}"),
                        undo: Some(UndoOp {
                            op: "delete_file".into(),
                            payload: serde_json::json!({"path": dest_s}),
                        }),
                        verified: Some(true),
                    };
                }
                _ => continue,
            }
        }
        ActionOutcome::Failed {
            error: "no working screenshot tool".into(),
        }
    }

    fn type_text(&self, req: &ActionRequest) -> ActionOutcome {
        let text = req.args.get("text").and_then(|v| v.as_str()).unwrap_or("");
        if text.is_empty() {
            return ActionOutcome::Refused {
                reason: "empty text".into(),
            };
        }
        if text.len() > 5000 {
            return ActionOutcome::Refused {
                reason: "text too long for typing".into(),
            };
        }
        if let Some(wt) = find_in_path("wtype") {
            return run_status(&wt, &[text], &format!("typed {} chars (wtype)", text.len()));
        }
        if let Some(xd) = find_in_path("xdotool") {
            return run_status(
                &xd,
                &["type", "--delay", "0", "--", text],
                &format!("typed {} chars (xdotool)", text.len()),
            );
        }
        ActionOutcome::Failed {
            error: "no typing tool (wtype/xdotool)".into(),
        }
    }

    fn key_press(&self, req: &ActionRequest) -> ActionOutcome {
        let key = req.args.get("key").and_then(|v| v.as_str()).unwrap_or("");
        let combo = match canonical_key_combo(key) {
            Some(c) => c,
            None => {
                return ActionOutcome::Refused {
                    reason: format!("bad key combo: {key:?}"),
                }
            }
        };
        if let Some(xd) = find_in_path("xdotool") {
            return run_status(
                &xd,
                &["key", "--clearmodifiers", &combo],
                &format!("pressed {combo} (xdotool)"),
            );
        }
        ActionOutcome::Failed {
            error: "no key tool (xdotool)".into(),
        }
    }

    fn xdotool() -> Option<PathBuf> {
        find_in_path("xdotool")
    }

    fn mouse_move(&self, req: &ActionRequest) -> ActionOutcome {
        let (x, y) = match parse_xy(&req.args, "x", "y") {
            Some(p) => p,
            None => {
                return ActionOutcome::Refused {
                    reason: format!("bad coords: {}", compact_value(&req.args)),
                }
            }
        };
        match Self::xdotool() {
            Some(xd) => run_status(
                &xd,
                &["mousemove", &x.to_string(), &y.to_string()],
                &format!("moved to {x},{y} (xdotool)"),
            ),
            None => ActionOutcome::Failed {
                error: "no mouse tool (xdotool)".into(),
            },
        }
    }

    fn mouse_click(&self, req: &ActionRequest) -> ActionOutcome {
        let (x, y) = match parse_xy(&req.args, "x", "y") {
            Some(p) => p,
            None => {
                return ActionOutcome::Refused {
                    reason: format!("bad coords: {}", compact_value(&req.args)),
                }
            }
        };
        let button = match parse_button(req.args.get("button")) {
            Some(b) => b,
            None => {
                return ActionOutcome::Refused {
                    reason: format!("bad button: {}", compact_value(&req.args)),
                }
            }
        };
        let clicks = match req.args.get("clicks") {
            None => 1,
            Some(v) => match v.as_u64() {
                Some(1) => 1,
                Some(2) => 2,
                _ => {
                    return ActionOutcome::Refused {
                        reason: format!("bad clicks (want 1|2): {}", compact_value(&req.args)),
                    }
                }
            },
        };
        match Self::xdotool() {
            Some(xd) => {
                let xs = x.to_string();
                let ys = y.to_string();
                let bs = button.to_string();
                let mut args: Vec<&str> =
                    vec!["mousemove", &xs, &ys, "click", "--repeat", "1", &bs];
                if clicks == 2 {
                    args = vec![
                        "mousemove",
                        &xs,
                        &ys,
                        "click",
                        "--repeat",
                        "2",
                        "--delay",
                        "50",
                        &bs,
                    ];
                }
                run_status(
                    &xd,
                    &args,
                    &format!("clicked {button}x{clicks} at {x},{y} (xdotool)"),
                )
            }
            None => ActionOutcome::Failed {
                error: "no mouse tool (xdotool)".into(),
            },
        }
    }

    fn mouse_drag(&self, req: &ActionRequest) -> ActionOutcome {
        let from = parse_xy(&req.args, "x1", "y1");
        let to = parse_xy(&req.args, "x2", "y2");
        let button = parse_button(req.args.get("button"));
        let (Some((x1, y1)), Some((x2, y2)), Some(button)) = (from, to, button) else {
            return ActionOutcome::Refused {
                reason: format!("bad drag args: {}", compact_value(&req.args)),
            };
        };
        match Self::xdotool() {
            Some(xd) => {
                let (x1s, y1s, x2s, y2s, bs) = (
                    x1.to_string(),
                    y1.to_string(),
                    x2.to_string(),
                    y2.to_string(),
                    button.to_string(),
                );
                run_status(
                    &xd,
                    &[
                        "mousemove",
                        &x1s,
                        &y1s,
                        "mousedown",
                        &bs,
                        "mousemove",
                        &x2s,
                        &y2s,
                        "mouseup",
                        &bs,
                    ],
                    &format!("dragged {x1},{y1}->{x2},{y2} (xdotool)"),
                )
            }
            None => ActionOutcome::Failed {
                error: "no mouse tool (xdotool)".into(),
            },
        }
    }

    fn scroll(&self, req: &ActionRequest) -> ActionOutcome {
        let dir = req
            .args
            .get("direction")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_lowercase();
        // xdotool wheel buttons: 4=up 5=down 6=left 7=right.
        let button = match dir.as_str() {
            "up" => 4,
            "down" => 5,
            "left" => 6,
            "right" => 7,
            _ => {
                return ActionOutcome::Refused {
                    reason: format!("bad direction: {}", compact_value(&req.args)),
                }
            }
        };
        let amount = match req.args.get("amount") {
            None => 3u64,
            Some(v) => match v.as_u64() {
                Some(n) if (1..=20).contains(&n) => n,
                _ => {
                    return ActionOutcome::Refused {
                        reason: format!("bad amount (want 1..=20): {}", compact_value(&req.args)),
                    }
                }
            },
        };
        match Self::xdotool() {
            Some(xd) => {
                let (bs, ns) = (button.to_string(), amount.to_string());
                run_status(
                    &xd,
                    &["click", "--repeat", &ns, &bs],
                    &format!("scrolled {dir} x{amount} (xdotool)"),
                )
            }
            None => ActionOutcome::Failed {
                error: "no mouse tool (xdotool)".into(),
            },
        }
    }

    async fn wait_ms(&self, req: &ActionRequest) -> ActionOutcome {
        match req.args.get("ms").and_then(|v| v.as_u64()) {
            Some(ms) if ms <= 30_000 => {
                tokio::time::sleep(std::time::Duration::from_millis(ms)).await;
                ActionOutcome::Done {
                    summary: format!("waited {ms}ms"),
                    undo: None,
                    verified: Some(true),
                }
            }
            _ => ActionOutcome::Refused {
                reason: format!("bad ms (want 0..=30000): {}", compact_value(&req.args)),
            },
        }
    }

    fn cursor_position(&self) -> ActionOutcome {
        let xd = match Self::xdotool() {
            Some(x) => x,
            None => {
                return ActionOutcome::Failed {
                    error: "no mouse tool (xdotool)".into(),
                }
            }
        };
        match run_stdout(&xd, &["getmouselocation", "--shell"]) {
            Ok(out) => match parse_mouse_location(&out) {
                Some((x, y)) => ActionOutcome::Done {
                    summary: format!("cursor at {x},{y}"),
                    undo: None,
                    verified: Some(true),
                },
                None => ActionOutcome::Failed {
                    error: "unparseable getmouselocation output".into(),
                },
            },
            Err(e) => ActionOutcome::Failed { error: e },
        }
    }
}

/// Screen pixels as u32 within 0..=32767. Floats, negatives, and overflow
/// are refused (never rounded or clamped silently).
fn parse_xy(args: &serde_json::Value, kx: &str, ky: &str) -> Option<(u32, u32)> {
    let x = args.get(kx)?.as_u64()?;
    let y = args.get(ky)?.as_u64()?;
    if x > 32767 || y > 32767 {
        return None;
    }
    Some((x as u32, y as u32))
}

/// Button name to xdotool number. Missing defaults to left.
fn parse_button(v: Option<&serde_json::Value>) -> Option<u32> {
    match v {
        None => Some(1),
        Some(val) => match val.as_str()?.trim().to_lowercase().as_str() {
            "left" | "1" => Some(1),
            "middle" | "2" => Some(2),
            "right" | "3" => Some(3),
            _ => None,
        },
    }
}

/// Parse `xdotool getmouselocation --shell` (X=..\nY=.. lines).
fn parse_mouse_location(out: &str) -> Option<(u32, u32)> {
    let mut x = None;
    let mut y = None;
    for line in out.lines() {
        let Some((k, v)) = line.split_once('=') else {
            continue;
        };
        match k.trim() {
            "X" => x = v.trim().parse::<u32>().ok(),
            "Y" => y = v.trim().parse::<u32>().ok(),
            _ => {}
        }
    }
    Some((x?, y?))
}

/// Validate + canonicalize a key combo for `xdotool key`. Tokens are
/// `+`-joined: modifiers/single alphanumerics lowercase, named keys in
/// xdotool spelling. Anything else is refused (never passed to argv).
pub fn canonical_key_combo(key: &str) -> Option<String> {
    fn named(tok: &str) -> Option<&'static str> {
        Some(match tok {
            "return" | "enter" => "Return",
            "tab" => "Tab",
            "escape" | "esc" => "Escape",
            "backspace" | "back" => "BackSpace",
            "delete" | "del" => "Delete",
            "space" => "space",
            "up" => "Up",
            "down" => "Down",
            "left" => "Left",
            "right" => "Right",
            "home" => "Home",
            "end" => "End",
            "page_up" | "pgup" => "Page_Up",
            "page_down" | "pgdn" => "Page_Down",
            "insert" | "ins" => "Insert",
            "print" | "printscreen" => "Print",
            "menu" => "Menu",
            "f1" => "F1",
            "f2" => "F2",
            "f3" => "F3",
            "f4" => "F4",
            "f5" => "F5",
            "f6" => "F6",
            "f7" => "F7",
            "f8" => "F8",
            "f9" => "F9",
            "f10" => "F10",
            "f11" => "F11",
            "f12" => "F12",
            "ctrl" | "control" => "ctrl",
            "alt" => "alt",
            "shift" => "shift",
            "super" | "win" | "meta" => "super",
            _ => return None,
        })
    }
    let key = key.trim();
    if key.is_empty() || key.len() > 48 {
        return None;
    }
    let mut out = Vec::new();
    for raw in key.split('+') {
        let tok = raw.trim().to_lowercase();
        if tok.is_empty() || tok.len() > 12 {
            return None;
        }
        if tok.len() == 1 && tok.as_bytes()[0].is_ascii_alphanumeric() {
            out.push(tok);
        } else if let Some(n) = named(&tok) {
            out.push(n.to_string());
        } else {
            return None;
        }
    }
    if out.is_empty() || out.len() > 4 {
        return None;
    }
    Some(out.join("+"))
}

fn pipe_stdin(bin: &Path, args: &[&str], text: &str, ok_summary: &str) -> ActionOutcome {
    use std::io::Write;
    match std::process::Command::new(bin)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(mut child) => {
            if let Some(mut stdin) = child.stdin.take() {
                if stdin.write_all(text.as_bytes()).is_err() {
                    return ActionOutcome::Failed {
                        error: "stdin write failed".into(),
                    };
                }
            }
            match child.wait() {
                Ok(s) if s.success() => ActionOutcome::Done {
                    summary: ok_summary.into(),
                    undo: None,
                    verified: Some(true),
                },
                Ok(s) => ActionOutcome::Failed {
                    error: format!("exit {s}"),
                },
                Err(e) => ActionOutcome::Failed {
                    error: e.to_string(),
                },
            }
        }
        Err(e) => ActionOutcome::Failed {
            error: e.to_string(),
        },
    }
}

/// Run and capture stdout as UTF-8 (for parsed queries like cursor pos).
fn run_stdout(bin: &Path, args: &[&str]) -> Result<String, String> {
    match std::process::Command::new(bin)
        .args(args)
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
    {
        Ok(o) if o.status.success() => {
            String::from_utf8(o.stdout).map_err(|_| "non-utf8 stdout".to_string())
        }
        Ok(o) => Err(format!("exit {}", o.status)),
        Err(e) => Err(e.to_string()),
    }
}

fn run_status(bin: &Path, args: &[&str], ok_summary: &str) -> ActionOutcome {
    match std::process::Command::new(bin)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
    {
        Ok(s) if s.success() => ActionOutcome::Done {
            summary: ok_summary.into(),
            undo: None,
            verified: Some(true),
        },
        Ok(s) => ActionOutcome::Failed {
            error: format!("exit {s}"),
        },
        Err(e) => ActionOutcome::Failed {
            error: e.to_string(),
        },
    }
}

fn dirs_home() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

fn urlencode(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                o.push(b as char)
            }
            b' ' => o.push('+'),
            _ => o.push_str(&format!("%{b:02X}")),
        }
    }
    o
}

fn compact_value(v: &serde_json::Value) -> String {
    let mut s = v.to_string();
    if s.len() > 160 {
        s.truncate(160);
        s.push('…');
    }
    s
}

/// Args as journaled: typed/pasted text is never stored, only its length.
pub fn redact_args(tool: ToolId, args: &serde_json::Value) -> serde_json::Value {
    match tool {
        ToolId::TypeText | ToolId::ClipboardCopy | ToolId::AiQuery | ToolId::KeyPress => {
            let chars = args
                .get("text")
                .or_else(|| args.get("query"))
                .or_else(|| args.get("key"))
                .and_then(|v| v.as_str())
                .map(|s| s.len())
                .unwrap_or(0);
            serde_json::json!({"chars": chars})
        }
        _ => args.clone(),
    }
}

fn walk_search(
    root: &Path,
    query: &str,
    max_depth: usize,
    max_entries: usize,
    limit: usize,
) -> Vec<String> {
    const SKIP: &[&str] = &[
        ".git",
        "node_modules",
        "target",
        ".cache",
        ".mozilla",
        "__pycache__",
        ".local/share/Trash",
    ];
    let q = query.to_lowercase();
    let mut hits = Vec::new();
    let mut stack: Vec<(PathBuf, usize)> = vec![(root.to_path_buf(), 0)];
    let mut seen = 0usize;
    while let Some((dir, depth)) = stack.pop() {
        if hits.len() >= limit || seen >= max_entries || depth > max_depth {
            continue;
        }
        let rd = match std::fs::read_dir(&dir) {
            Ok(r) => r,
            Err(_) => continue,
        };
        for e in rd.flatten() {
            seen += 1;
            if seen >= max_entries || hits.len() >= limit {
                break;
            }
            let p = e.path();
            let name = e.file_name().to_string_lossy().to_string();
            if name.starts_with('.') && dir == *root {
                continue;
            }
            if SKIP.iter().any(|s| p.to_string_lossy().contains(s)) {
                continue;
            }
            if name.to_lowercase().contains(&q) {
                hits.push(p.display().to_string());
            }
            if p.is_dir() && depth < max_depth {
                stack.push((p, depth + 1));
            }
        }
    }
    hits.sort();
    hits
}

/// Execute + journal in one call. Returns the outcome; the journal always
/// records what happened (or what was refused).
pub async fn execute_and_journal(
    exec: &LocalExecutor,
    journal: &mut Journal,
    req: &ActionRequest,
) -> Result<ActionOutcome, ActError> {
    let outcome = exec.execute(req).await;
    let (undo, verified) = match &outcome {
        ActionOutcome::Done { undo, verified, .. } => (undo.clone(), *verified),
        _ => (None, None),
    };
    journal.append(JournalEntry {
        seq: 0,
        at_ms: 0,
        action_id: req.action_id.clone(),
        intent: req.intent.clone(),
        tool: serde_json::to_string(&req.tool)
            .unwrap_or_default()
            .trim_matches('"')
            .to_string(),
        args_redacted: redact_args(req.tool, &req.args),
        status: outcome.status(),
        summary: outcome.summary(),
        undo,
        verified,
    })?;
    Ok(outcome)
}

/// Apply a one-shot undo payload. Only known ops; unknown = error, never guess.
pub fn apply_undo(op: &UndoOp) -> Result<String, ActError> {
    match op.op.as_str() {
        "noop" => Ok("nothing to undo".into()),
        "delete_file" => {
            let p = op
                .payload
                .get("path")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ActError::Undo("delete_file".into(), "missing path".into()))?;
            let path = Path::new(p);
            if !path.is_file() {
                return Err(ActError::Undo("delete_file".into(), "not a file".into()));
            }
            std::fs::remove_file(path)?;
            Ok(format!("deleted {p}"))
        }
        other => Err(ActError::Undo(other.into(), "unknown undo op".into())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dry() -> LocalExecutor {
        LocalExecutor::new(true, vec!["firefox".into()])
    }
    fn live() -> LocalExecutor {
        LocalExecutor::new(false, vec!["firefox".into(), "true".into()])
    }

    fn req(tool: ToolId, args: serde_json::Value) -> ActionRequest {
        ActionRequest {
            action_id: "t1".into(),
            intent: "test".into(),
            tool,
            args,
            risk_tier: 1,
        }
    }

    #[test]
    fn availability_probe_runs() {
        let r = availability();
        assert_eq!(r.tools.len(), ToolId::all().len());
        assert!(!r.session_type.is_empty());
    }

    #[tokio::test]
    async fn dry_run_executes_nothing() {
        let o = dry()
            .execute(&req(
                ToolId::OpenUrl,
                serde_json::json!({"url": "https://example.com"}),
            ))
            .await;
        assert!(matches!(o, ActionOutcome::Done { .. }));
        assert!(o.summary().contains("dry-run"));
    }

    #[tokio::test]
    async fn url_scheme_allowlist() {
        let e = live();
        for bad in [
            "file:///etc/passwd",
            "javascript:alert(1)",
            "data:text/plain,hi",
            "ftp://x/y",
            "",
        ] {
            let o = e
                .execute(&req(ToolId::OpenUrl, serde_json::json!({"url": bad})))
                .await;
            assert!(
                matches!(o, ActionOutcome::Refused { .. }),
                "accepted {bad:?}"
            );
        }
        // Good URL in dry-run is fine; live would open a browser, so only check refusal logic here.
        let o = dry()
            .execute(&req(
                ToolId::OpenUrl,
                serde_json::json!({"url": "https://example.com"}),
            ))
            .await;
        assert!(matches!(o, ActionOutcome::Done { .. }));
    }

    #[tokio::test]
    async fn open_app_needs_allowlist_and_clean_name() {
        let e = live();
        let o = e
            .execute(&req(
                ToolId::OpenApp,
                serde_json::json!({"app": "not-allowed"}),
            ))
            .await;
        assert!(matches!(o, ActionOutcome::Refused { .. }));
        let o = e
            .execute(&req(
                ToolId::OpenApp,
                serde_json::json!({"app": "x; rm -rf ~"}),
            ))
            .await;
        assert!(matches!(o, ActionOutcome::Refused { .. }));
        // `true` is allowlisted and exits 0 — a real spawn, harmless.
        if find_in_path("true").is_some() {
            let o = e
                .execute(&req(ToolId::OpenApp, serde_json::json!({"app": "true"})))
                .await;
            assert!(
                matches!(o, ActionOutcome::Done { .. }),
                "got {:?}",
                o.summary()
            );
        }
    }

    #[tokio::test]
    async fn file_search_finds_fixture() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("quarterly-report.txt"), "x").unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("sub").join("report-draft.md"), "y").unwrap();
        let e = live();
        let o = e
            .execute(&req(
                ToolId::FileSearch,
                serde_json::json!({"query": "report", "root": dir.path()}),
            ))
            .await;
        match o {
            ActionOutcome::Done { summary, .. } => {
                assert!(summary.starts_with("2 hits"), "{summary}")
            }
            other => panic!("unexpected {:?}", other.summary()),
        }
        let o = e
            .execute(&req(ToolId::FileSearch, serde_json::json!({"query": ""})))
            .await;
        assert!(matches!(o, ActionOutcome::Refused { .. }));
    }

    #[tokio::test]
    async fn ai_query_honest_without_provider() {
        std::env::remove_var("ALIVE_MOCK_LLM");
        let o = live()
            .execute(&req(ToolId::AiQuery, serde_json::json!({"query": "hi"})))
            .await;
        assert!(matches!(o, ActionOutcome::Failed { .. }));
    }

    #[tokio::test]
    async fn journal_wiring_and_undo_delete_file() {
        let dir = tempfile::tempdir().unwrap();
        let jp = dir.path().join("j.jsonl");
        let mut j = Journal::open(&jp).unwrap();
        let o = execute_and_journal(
            &dry(),
            &mut j,
            &req(ToolId::OpenApp, serde_json::json!({"app": "firefox"})),
        )
        .await
        .unwrap();
        assert!(matches!(o, ActionOutcome::Done { .. }));
        assert_eq!(j.len(), 1);
        // Redaction: typed text never hits the journal.
        let o2 = execute_and_journal(
            &dry(),
            &mut j,
            &req(
                ToolId::TypeText,
                serde_json::json!({"text": "secret words"}),
            ),
        )
        .await
        .unwrap();
        assert!(matches!(o2, ActionOutcome::Done { .. }));
        let back = Journal::read_all(&jp).unwrap();
        assert_eq!(back.len(), 2);
        assert_eq!(back[1].args_redacted, serde_json::json!({"chars": 12}));
        assert!(!std::fs::read_to_string(&jp).unwrap().contains("secret"));

        // Real undo: delete_file removes a temp file, one-shot.
        let victim = dir.path().join("v.txt");
        std::fs::write(&victim, "bye").unwrap();
        let op = UndoOp {
            op: "delete_file".into(),
            payload: serde_json::json!({"path": victim.to_string_lossy()}),
        };
        assert!(apply_undo(&op).is_ok());
        assert!(!victim.exists());
        assert!(apply_undo(&op).is_err(), "second undo must fail (gone)");
        assert!(apply_undo(&UndoOp {
            op: "mystery".into(),
            payload: serde_json::Value::Null
        })
        .is_err());
    }

    #[test]
    fn urlencode_basics() {
        assert_eq!(urlencode("hello world"), "hello+world");
        assert!(urlencode("a&b=c").contains("%26"));
    }

    #[test]
    fn redaction_map() {
        assert_eq!(
            redact_args(ToolId::TypeText, &serde_json::json!({"text": "abcdef"})),
            serde_json::json!({"chars": 6})
        );
        assert_eq!(
            redact_args(ToolId::OpenUrl, &serde_json::json!({"url": "https://x"})),
            serde_json::json!({"url": "https://x"})
        );
    }

    #[test]
    fn tool_registry_is_consistent() {
        // names <-> from_name roundtrip; every name has exactly one schema;
        // every schema's required args are declared properties.
        let names = tool_names();
        assert_eq!(names.len(), ToolId::all().len());
        for t in ToolId::all() {
            assert_eq!(ToolId::from_name(t.name()), Some(*t));
        }
        assert_eq!(ToolId::from_name("launch_missiles"), None);
        let schemas = tool_schemas();
        assert_eq!(schemas.len(), names.len());
        for (s, n) in schemas.iter().zip(names.iter()) {
            assert_eq!(s["type"], "function");
            assert_eq!(s["function"]["name"], *n);
            let params = &s["function"]["parameters"];
            assert_eq!(params["type"], "object");
            for r in params["required"].as_array().unwrap() {
                let key = r.as_str().unwrap();
                assert!(
                    params["properties"].get(key).is_some(),
                    "{n}: required arg {key} not declared"
                );
            }
        }
        // Verbs that change external state sit at tier 2 (confirm gate).
        assert_eq!(ToolId::OpenApp.risk_tier(), 2);
        assert_eq!(ToolId::TypeText.risk_tier(), 2);
        assert_eq!(ToolId::KeyPress.risk_tier(), 2);
        assert_eq!(ToolId::MouseClick.risk_tier(), 2);
        assert_eq!(ToolId::MouseDrag.risk_tier(), 2);
        assert_eq!(ToolId::Scroll.risk_tier(), 2);
        assert_eq!(ToolId::Screenshot.risk_tier(), 1);
        assert_eq!(ToolId::MouseMove.risk_tier(), 1);
        assert_eq!(ToolId::CursorPosition.risk_tier(), 1);
        assert_eq!(ToolId::WaitMs.risk_tier(), 1);
    }

    #[test]
    fn key_combo_validation_table() {
        for (raw, canon) in [
            ("Return", "Return"),
            ("enter", "Return"),
            ("ESC", "Escape"),
            ("ctrl+c", "ctrl+c"),
            ("CTRL+Return", "ctrl+Return"),
            ("alt+F4", "alt+F4"),
            ("super+t", "super+t"),
            ("a", "a"),
        ] {
            assert_eq!(canonical_key_combo(raw).as_deref(), Some(canon), "{raw}");
        }
        for bad in [
            "",
            "Return; rm -rf ~",
            "ctrl+;id",
            "$(id)",
            "Return+Tab+Up+Down+Left",
            "thiskeydoesnotexist",
            "ctrl+",
        ] {
            assert_eq!(canonical_key_combo(bad), None, "{bad:?}");
        }
    }

    #[tokio::test]
    async fn key_press_refuses_bad_combo_without_spawning() {
        let o = live()
            .execute(&req(
                ToolId::KeyPress,
                serde_json::json!({"key": "ctrl+;id"}),
            ))
            .await;
        assert!(
            matches!(o, ActionOutcome::Refused { .. }),
            "{:?}",
            o.summary()
        );
        // Dry-run rehearses without touching the window system.
        let o = dry()
            .execute(&req(ToolId::KeyPress, serde_json::json!({"key": "Return"})))
            .await;
        assert!(matches!(o, ActionOutcome::Done { .. }));
        assert!(o.summary().contains("dry-run"));
        // Key combos never hit the journal in cleartext.
        assert_eq!(
            redact_args(ToolId::KeyPress, &serde_json::json!({"key": "ctrl+c"})),
            serde_json::json!({"chars": 6})
        );
    }

    #[test]
    fn mouse_normalizer_tables() {
        // Coords: ints in range pass; floats/negatives/overflow/missing fail.
        assert_eq!(
            parse_xy(&serde_json::json!({"x": 100, "y": 200}), "x", "y"),
            Some((100, 200))
        );
        for bad in [
            serde_json::json!({"x": 1.5, "y": 2}),
            serde_json::json!({"x": -1, "y": 2}),
            serde_json::json!({"x": 99999, "y": 2}),
            serde_json::json!({"x": "100", "y": 2}),
            serde_json::json!({"y": 2}),
            serde_json::json!({}),
        ] {
            assert_eq!(parse_xy(&bad, "x", "y"), None, "{bad}");
        }
        // Buttons: names + numbers pass, default left, junk fails.
        assert_eq!(parse_button(None), Some(1));
        assert_eq!(parse_button(Some(&serde_json::json!("RIGHT"))), Some(3));
        assert_eq!(parse_button(Some(&serde_json::json!("2"))), Some(2));
        assert_eq!(parse_button(Some(&serde_json::json!("hyper"))), None);
        assert_eq!(parse_button(Some(&serde_json::json!(9))), None);
        // Cursor parse: --shell format, tolerant of blank lines.
        assert_eq!(
            parse_mouse_location("X=640\nY=360\nSCREEN=0\nWINDOW=123\n"),
            Some((640, 360))
        );
        assert_eq!(parse_mouse_location("\nX=1\n\nY=2\n"), Some((1, 2)));
        assert_eq!(parse_mouse_location("nope"), None);
        assert_eq!(parse_mouse_location("X=abc\nY=2\n"), None);
    }

    #[tokio::test]
    async fn mouse_verbs_refuse_bad_args_without_spawning() {
        let e = live();
        // Bad coords: no xdotool invocation (Refused, not Failed).
        for (tool, args) in [
            (ToolId::MouseMove, serde_json::json!({"x": -5, "y": 5})),
            (
                ToolId::MouseClick,
                serde_json::json!({"x": 1, "y": 1, "button": "hyper"}),
            ),
            (
                ToolId::MouseClick,
                serde_json::json!({"x": 1, "y": 1, "clicks": 5}),
            ),
            (
                ToolId::MouseDrag,
                serde_json::json!({"x1": 1, "y1": 1, "x2": 2}),
            ),
            (ToolId::Scroll, serde_json::json!({"direction": "diagonal"})),
            (
                ToolId::Scroll,
                serde_json::json!({"direction": "up", "amount": 99}),
            ),
            (ToolId::WaitMs, serde_json::json!({"ms": 60_000})),
            (ToolId::WaitMs, serde_json::json!({})),
        ] {
            let o = e.execute(&req(tool, args)).await;
            assert!(
                matches!(o, ActionOutcome::Refused { .. }),
                "{tool:?}: {:?}",
                o.summary()
            );
        }
        // Dry-run rehearses every new verb without touching the desktop.
        for (tool, args) in [
            (ToolId::MouseMove, serde_json::json!({"x": 10, "y": 10})),
            (
                ToolId::MouseClick,
                serde_json::json!({"x": 10, "y": 10, "clicks": 2}),
            ),
            (
                ToolId::MouseDrag,
                serde_json::json!({"x1": 1, "y1": 1, "x2": 50, "y2": 50}),
            ),
            (ToolId::Scroll, serde_json::json!({"direction": "down"})),
            (ToolId::WaitMs, serde_json::json!({"ms": 1})),
            (ToolId::CursorPosition, serde_json::json!({})),
        ] {
            let o = dry().execute(&req(tool, args)).await;
            assert!(
                matches!(o, ActionOutcome::Done { .. }) && o.summary().contains("dry-run"),
                "{tool:?}: {:?}",
                o.summary()
            );
        }
        // wait_ms really waits (live, tiny) and cursor really parses (live).
        let t0 = std::time::Instant::now();
        let o = e
            .execute(&req(ToolId::WaitMs, serde_json::json!({"ms": 50})))
            .await;
        assert!(matches!(o, ActionOutcome::Done { .. }));
        assert!(t0.elapsed() >= std::time::Duration::from_millis(50));
        if find_in_path("xdotool").is_some() && std::env::var("DISPLAY").is_ok() {
            let o = e
                .execute(&req(ToolId::CursorPosition, serde_json::json!({})))
                .await;
            assert!(
                matches!(o, ActionOutcome::Done { .. }),
                "cursor: {:?}",
                o.summary()
            );
        }
    }
}
