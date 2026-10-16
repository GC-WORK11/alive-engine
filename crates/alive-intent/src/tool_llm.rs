//! Tool-capable chat LLM: DeepSeek-V4.1-Flash on the official global endpoint.
//!
//! This is the System Two half of ALive: Jev triages (addressing? risky?
//! multi-step? which tool for the fast path?) and this client does the
//! open-ended work — free-form instructions in, tool calls out, looped
//! against executor observations until done. No intent catalog involved:
//! the model sees generic computer verbs, never hardcoded tasks.
//!
//! DeepSeek's API is OpenAI-compatible (chat completions + function
//! calling, Bearer auth). Current model id is `deepseek-flash`, which
//! serves V4.1-Flash with tool use.
//!
//! Configured by environment, like Jev:
//! - `ALIVE_LLM_ENDPOINT` (default `https://api.deepseek.com/chat/completions`)
//! - `ALIVE_LLM_MODEL` (default `deepseek-flash`)
//! - `ALIVE_LLM_API_KEY` or `DEEPSEEK_API_KEY` (required; absent = honest
//!   "not configured" error)

use std::time::Duration;

use super::IntentError;

pub const DEFAULT_ENDPOINT: &str = "https://api.deepseek.com/chat/completions";
pub const DEFAULT_MODEL: &str = "deepseek-flash";

/// One function call requested by the model. `args` is parsed JSON —
/// unparseable argument strings are errors, never guessed.
#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub args: serde_json::Value,
}

/// The model's reply: optional text plus zero or more tool calls.
#[derive(Debug, Clone, Default)]
pub struct LlmReply {
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
}

#[derive(Debug, Clone)]
pub struct ToolLlmClient {
    endpoint: String,
    model: String,
    api_key: String,
    timeout: Duration,
    http: reqwest::Client,
}

impl ToolLlmClient {
    pub fn new(endpoint: String, model: String, api_key: String) -> Result<Self, IntentError> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(60))
            .user_agent("alive/0.1")
            .build()
            .map_err(|e| IntentError::Transport(e.to_string()))?;
        Ok(Self {
            endpoint,
            model,
            api_key,
            timeout: Duration::from_secs(60),
            http,
        })
    }

    pub fn from_env() -> Result<Self, IntentError> {
        let key = std::env::var("ALIVE_LLM_API_KEY")
            .or_else(|_| std::env::var("DEEPSEEK_API_KEY"))
            .map_err(|_| IntentError::Transport("ALIVE_LLM_API_KEY not set".into()))?;
        let endpoint =
            std::env::var("ALIVE_LLM_ENDPOINT").unwrap_or_else(|_| DEFAULT_ENDPOINT.into());
        let model = std::env::var("ALIVE_LLM_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.into());
        Self::new(endpoint, model, key)
    }

    pub fn request_body(
        &self,
        messages: &[serde_json::Value],
        tools: &[serde_json::Value],
    ) -> serde_json::Value {
        serde_json::json!({
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
        })
    }

    pub async fn chat(
        &self,
        messages: &[serde_json::Value],
        tools: &[serde_json::Value],
    ) -> Result<LlmReply, IntentError> {
        let body = self.request_body(messages, tools);
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
        parse_chat_reply(&json)
    }
}

/// Parse an OpenAI-style chat completion into text + tool calls.
/// Unknown shapes are errors; argument strings must be valid JSON.
pub fn parse_chat_reply(body: &serde_json::Value) -> Result<LlmReply, IntentError> {
    let msg = body
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .ok_or_else(|| IntentError::Parse("missing choices[0].message".into()))?;
    let content = msg
        .get("content")
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string();
    let mut tool_calls = Vec::new();
    if let Some(arr) = msg.get("tool_calls").and_then(|t| t.as_array()) {
        for tc in arr {
            let id = tc
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let f = tc
                .get("function")
                .ok_or_else(|| IntentError::Parse("tool_call without function".into()))?;
            let name = f
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            if name.is_empty() {
                return Err(IntentError::Parse("tool_call without name".into()));
            }
            let args_str = f.get("arguments").and_then(|v| v.as_str()).unwrap_or("");
            let args: serde_json::Value = serde_json::from_str(args_str)
                .map_err(|e| IntentError::Parse(format!("bad tool arguments: {e}")))?;
            tool_calls.push(ToolCall { id, name, args });
        }
    }
    Ok(LlmReply {
        content,
        tool_calls,
    })
}

/// Message builders (thin JSON, same shape the wire expects).
pub fn sys_msg(content: &str) -> serde_json::Value {
    serde_json::json!({"role": "system", "content": content})
}
pub fn user_msg(content: &str) -> serde_json::Value {
    serde_json::json!({"role": "user", "content": content})
}
/// Assistant turn echo for multi-step loops (content + the calls it made).
pub fn assistant_msg(reply: &LlmReply) -> serde_json::Value {
    serde_json::json!({
        "role": "assistant",
        "content": reply.content,
        "tool_calls": reply.tool_calls.iter().map(|tc| {
            serde_json::json!({
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.args.to_string()},
            })
        }).collect::<Vec<_>>(),
    })
}
pub fn tool_msg(call_id: &str, name: &str, result: &str) -> serde_json::Value {
    serde_json::json!({
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": result,
    })
}

/// System prompt for the computer-operator loop. Short on purpose:
/// schemas carry the arg details, the executor enforces the rules.
pub const OPERATOR_SYSTEM_PROMPT: &str = "You operate the user's computer through tools only. \
Rules: do exactly what was asked, nothing extra; prefer fewer steps; \
never invent file paths, URLs, or app names — use search tools to find them; \
if a tool is refused or fails, say so and stop that branch; \
when the task is complete, reply with plain text and no more tool calls.";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_text_only_reply() {
        let body = serde_json::json!({
            "choices": [{"message": {"role": "assistant", "content": "done"}}]
        });
        let r = parse_chat_reply(&body).unwrap();
        assert_eq!(r.content, "done");
        assert!(r.tool_calls.is_empty());
    }

    #[test]
    fn parse_tool_calls_reply() {
        let body = serde_json::json!({
            "choices": [{"message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "type": "function",
                     "function": {"name": "browser_search", "arguments": "{\"query\": \"weather\"}"}},
                    {"id": "c2", "type": "function",
                     "function": {"name": "screenshot", "arguments": "{}"}},
                ],
            }}]
        });
        let r = parse_chat_reply(&body).unwrap();
        assert_eq!(r.tool_calls.len(), 2);
        assert_eq!(r.tool_calls[0].name, "browser_search");
        assert_eq!(r.tool_calls[0].args["query"], "weather");
        assert_eq!(r.tool_calls[1].name, "screenshot");
    }

    #[test]
    fn parse_rejects_bad_shapes() {
        assert!(parse_chat_reply(&serde_json::json!({})).is_err());
        let bad_args = serde_json::json!({
            "choices": [{"message": {"tool_calls": [
                {"id": "c1", "function": {"name": "open_url", "arguments": "{oops"}}
            ]}}]
        });
        assert!(parse_chat_reply(&bad_args).is_err());
        let no_name = serde_json::json!({
            "choices": [{"message": {"tool_calls": [
                {"id": "c1", "function": {"arguments": "{}"}}
            ]}}]
        });
        assert!(parse_chat_reply(&no_name).is_err());
    }

    #[test]
    fn transcript_roundtrips_through_builders() {
        let reply = LlmReply {
            content: "searching".into(),
            tool_calls: vec![ToolCall {
                id: "c1".into(),
                name: "browser_search".into(),
                args: serde_json::json!({"query": "x"}),
            }],
        };
        let a = assistant_msg(&reply);
        assert_eq!(a["tool_calls"][0]["function"]["name"], "browser_search");
        // Re-parse: the echoed arguments string must stay valid JSON.
        let back: serde_json::Value = serde_json::from_str(
            a["tool_calls"][0]["function"]["arguments"]
                .as_str()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(back["query"], "x");
        let t = tool_msg("c1", "browser_search", "opened ok");
        assert_eq!(t["role"], "tool");
        assert_eq!(sys_msg("s")["role"], "system");
        assert_eq!(user_msg("u")["role"], "user");
    }

    /// Wire-format verification without a vendor key: stub HTTP server.
    #[tokio::test]
    async fn chat_posts_openai_shape_to_stub() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut s, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 32768];
            let n = s.read(&mut buf).await.unwrap();
            let raw = String::from_utf8_lossy(&buf[..n]).to_string();
            let body_str = raw.split("\r\n\r\n").nth(1).unwrap_or("").to_string();
            let canned = serde_json::json!({
                "choices": [{"message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "call_1", "type": "function",
                         "function": {"name": "open_url",
                                     "arguments": "{\"url\": \"https://example.com\"}"}},
                    ],
                }}]
            })
            .to_string();
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                canned.len(), canned
            );
            s.write_all(resp.as_bytes()).await.unwrap();
            body_str
        });

        let client = ToolLlmClient::new(
            format!("http://{addr}/v1/chat/completions"),
            "stub-model".into(),
            "test-key".into(),
        )
        .unwrap();
        let tools = vec![serde_json::json!({"type": "function"})];
        let r = client
            .chat(&[user_msg("open example")], &tools)
            .await
            .expect("stub roundtrip");
        assert_eq!(r.tool_calls.len(), 1);
        assert_eq!(r.tool_calls[0].args["url"], "https://example.com");

        let body_json: serde_json::Value = serde_json::from_str(&server.await.unwrap()).unwrap();
        assert_eq!(body_json["model"], "stub-model");
        assert_eq!(body_json["tool_choice"], "auto");
        assert_eq!(body_json["messages"][0]["role"], "user");
        assert_eq!(body_json["tools"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn missing_key_is_honest_error() {
        std::env::remove_var("ALIVE_LLM_API_KEY");
        std::env::remove_var("DEEPSEEK_API_KEY");
        assert!(matches!(
            ToolLlmClient::from_env(),
            Err(IntentError::Transport(_))
        ));
    }

    #[test]
    fn deepseek_defaults() {
        assert_eq!(
            DEFAULT_ENDPOINT,
            "https://api.deepseek.com/chat/completions"
        );
        assert_eq!(DEFAULT_MODEL, "deepseek-flash");
    }
}
