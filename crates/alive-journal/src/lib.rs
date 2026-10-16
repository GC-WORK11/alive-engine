//! alive-journal: append-only JSONL action log + in-memory undo registry.
//!
//! Every proposed/executed/refused action is a line. Undo payloads are kept
//! in memory (never re-read from disk) so undo can never resurrect stale state.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

#[derive(Debug, thiserror::Error)]
pub enum JournalError {
    #[error("journal io {0}: {1}")]
    Io(String, #[source] std::io::Error),
    #[error("journal parse line {0}: {1}")]
    Parse(usize, String),
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionStatus {
    Proposed,
    AwaitingConfirm,
    Done,
    Undone,
    Failed,
    Refused,
}

/// Opaque undo payload. The executor that created it is the only one that
/// may interpret it (checked by `op` tag at undo time).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UndoOp {
    pub op: String,
    #[serde(default)]
    pub payload: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JournalEntry {
    pub seq: u64,
    pub at_ms: u64,
    pub action_id: String,
    pub intent: String,
    pub tool: String,
    #[serde(default)]
    pub args_redacted: serde_json::Value,
    pub status: ActionStatus,
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub undo: Option<UndoOp>,
    #[serde(default)]
    pub verified: Option<bool>,
}

pub struct Journal {
    path: PathBuf,
    w: BufWriter<File>,
    seq: u64,
    undos: HashMap<String, UndoOp>,
}

impl Journal {
    pub fn open(path: &Path) -> Result<Self, JournalError> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| JournalError::Io(parent.display().to_string(), e))?;
            }
        }
        let f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| JournalError::Io(path.display().to_string(), e))?;
        Ok(Self {
            path: path.to_path_buf(),
            w: BufWriter::new(f),
            seq: 0,
            undos: HashMap::new(),
        })
    }

    pub fn append(&mut self, mut e: JournalEntry) -> Result<u64, JournalError> {
        e.seq = self.seq;
        self.seq += 1;
        if e.at_ms == 0 {
            e.at_ms = now_ms();
        }
        if let Some(u) = e.undo.clone() {
            self.undos.insert(e.action_id.clone(), u);
        }
        serde_json::to_writer(&mut self.w, &e)
            .map_err(|e| JournalError::Parse(self.seq as usize, e.to_string()))?;
        self.w
            .write_all(b"\n")
            .map_err(|e| JournalError::Io(self.path.display().to_string(), e))?;
        self.w
            .flush()
            .map_err(|e| JournalError::Io(self.path.display().to_string(), e))?;
        Ok(e.seq)
    }

    /// Take (remove) the undo payload for an action. One-shot by design.
    pub fn take_undo(&mut self, action_id: &str) -> Option<UndoOp> {
        self.undos.remove(action_id)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn len(&self) -> u64 {
        self.seq
    }

    pub fn is_empty(&self) -> bool {
        self.seq == 0
    }

    /// Read back a journal file (for UI + audit export). Skips blank lines.
    pub fn read_all(path: &Path) -> Result<Vec<JournalEntry>, JournalError> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| JournalError::Io(path.display().to_string(), e))?;
        let mut out = Vec::new();
        for (i, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let e: JournalEntry = serde_json::from_str(line)
                .map_err(|e| JournalError::Parse(i + 1, e.to_string()))?;
            out.push(e);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(id: &str) -> JournalEntry {
        JournalEntry {
            seq: 0,
            at_ms: 0,
            action_id: id.to_string(),
            intent: "open_app".to_string(),
            tool: "open_app".to_string(),
            args_redacted: serde_json::json!({"app": "firefox"}),
            status: ActionStatus::Done,
            summary: "opened firefox".to_string(),
            undo: Some(UndoOp {
                op: "noop".into(),
                payload: serde_json::Value::Null,
            }),
            verified: Some(true),
        }
    }

    #[test]
    fn append_assigns_seq_and_timestamp() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.jsonl");
        let mut j = Journal::open(&p).unwrap();
        assert!(j.is_empty());
        let s0 = j.append(entry("a1")).unwrap();
        let s1 = j.append(entry("a2")).unwrap();
        assert_eq!((s0, s1), (0, 1));
        assert_eq!(j.len(), 2);
        let back = Journal::read_all(&p).unwrap();
        assert_eq!(back.len(), 2);
        assert!(back[0].at_ms > 0);
        assert_eq!(back[1].action_id, "a2");
    }

    #[test]
    fn undo_is_one_shot() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.jsonl");
        let mut j = Journal::open(&p).unwrap();
        j.append(entry("a1")).unwrap();
        assert!(j.take_undo("a1").is_some());
        assert!(j.take_undo("a1").is_none());
    }

    #[test]
    fn read_all_skips_blank_lines() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.jsonl");
        std::fs::write(&p, "\n{\"seq\":0,\"at_ms\":1,\"action_id\":\"x\",\"intent\":\"i\",\"tool\":\"t\",\"status\":\"done\"}\n\n").unwrap();
        let back = Journal::read_all(&p).unwrap();
        assert_eq!(back.len(), 1);
    }
}
