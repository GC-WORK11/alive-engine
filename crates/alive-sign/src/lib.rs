//! alive-sign: language packs (Tier A/B/C), lexicon matching, N-best gloss types.
//!
//! A pack is data: `pack.toml + lexicon.json + intents.json + eval/`.
//! Adding a language = adding a directory. Zero core changes (M4 gate).

pub mod matcher;

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

#[derive(Debug, thiserror::Error)]
pub enum SignError {
    #[error("pack io {0}: {1}")]
    Io(String, #[source] std::io::Error),
    #[error("pack parse {0}: {1}")]
    Parse(String, String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackManifest {
    pub id: String,
    pub display_name: String,
    pub tier: String,
    pub version: String,
    #[serde(default)]
    pub adapter: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntryKind {
    Command,
    Dictation,
    Fingerspell,
    Control,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LexiconEntry {
    pub gloss: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    pub kind: EntryKind,
    #[serde(default)]
    pub intent: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lexicon {
    pub entries: Vec<LexiconEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentDef {
    pub intent: String,
    pub action: String,
    #[serde(default)]
    pub risk_tier: u8,
    #[serde(default)]
    pub slots: Vec<String>,
    #[serde(default)]
    pub patterns: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentCatalog {
    pub intents: Vec<IntentDef>,
}

fn read(path: &Path) -> Result<String, SignError> {
    std::fs::read_to_string(path).map_err(|e| SignError::Io(path.display().to_string(), e))
}

pub struct LanguagePack {
    pub manifest: PackManifest,
    pub lexicon: Lexicon,
    pub intents: IntentCatalog,
    alias_index: HashMap<String, usize>,
}

impl LanguagePack {
    pub fn load(dir: &Path) -> Result<Self, SignError> {
        let manifest: PackManifest = toml::from_str(&read(&dir.join("pack.toml"))?)
            .map_err(|e| SignError::Parse("pack.toml".into(), e.to_string()))?;
        let lexicon: Lexicon = serde_json::from_str(&read(&dir.join("lexicon.json"))?)
            .map_err(|e| SignError::Parse("lexicon.json".into(), e.to_string()))?;
        let intents: IntentCatalog = serde_json::from_str(&read(&dir.join("intents.json"))?)
            .map_err(|e| SignError::Parse("intents.json".into(), e.to_string()))?;

        let mut alias_index = HashMap::new();
        for (i, e) in lexicon.entries.iter().enumerate() {
            alias_index.insert(e.gloss.to_lowercase(), i);
            for a in &e.aliases {
                alias_index.insert(a.to_lowercase(), i);
            }
        }
        Ok(Self {
            manifest,
            lexicon,
            intents,
            alias_index,
        })
    }

    /// Calibrated-vocabulary fast path: exact gloss/alias match (case-insensitive).
    pub fn match_gloss(&self, gloss: &str) -> Option<&LexiconEntry> {
        self.alias_index
            .get(&gloss.trim().to_lowercase())
            .map(|&i| &self.lexicon.entries[i])
    }

    pub fn intent(&self, name: &str) -> Option<&IntentDef> {
        self.intents.intents.iter().find(|d| d.intent == name)
    }

    /// Choice options for the Jev intent question (capped at 255 by protocol).
    pub fn intent_options(&self) -> Vec<String> {
        let mut v: Vec<String> = self
            .intents
            .intents
            .iter()
            .map(|d| d.intent.clone())
            .collect();
        v.push("dictate".to_string());
        v.push("cancel".to_string());
        v.truncate(255);
        v
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlossHypothesis {
    pub gloss: String,
    pub confidence: f64,
}

/// Top-N hypotheses, sorted desc and renormalized to sum 1.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NBest {
    pub hyps: Vec<GlossHypothesis>,
}

impl NBest {
    pub fn new(mut hyps: Vec<GlossHypothesis>, top_n: usize) -> Self {
        hyps.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hyps.truncate(top_n.max(1));
        let sum: f64 = hyps.iter().map(|h| h.confidence.max(0.0)).sum();
        if sum > 0.0 {
            for h in &mut hyps {
                h.confidence /= sum;
            }
        } else if !hyps.is_empty() {
            let u = 1.0 / hyps.len() as f64;
            for h in &mut hyps {
                h.confidence = u;
            }
        }
        Self { hyps }
    }

    pub fn top(&self) -> Option<&GlossHypothesis> {
        self.hyps.first()
    }
}

/// Future learned encoder (shared multilingual + per-pack adapter).
/// Ships as a trait so the heuristic/lexicon path and the ONNX path are swappable.
pub trait SignEncoder {
    fn encode(&self, flat_window: &[f32], keypoints: usize) -> NBest;
}

/// Honest placeholder: reports UNKNOWN at full confidence instead of guessing.
pub struct NullEncoder;

impl SignEncoder for NullEncoder {
    fn encode(&self, _flat_window: &[f32], _keypoints: usize) -> NBest {
        NBest::new(
            vec![GlossHypothesis {
                gloss: "UNKNOWN".to_string(),
                confidence: 1.0,
            }],
            1,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asl_dir() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../packs/asl")
    }

    fn csl_dir() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../packs/csl")
    }

    #[test]
    fn asl_pack_loads_and_matches() {
        let pack = LanguagePack::load(&asl_dir()).expect("packs/asl must load");
        assert_eq!(pack.manifest.id, "asl");
        assert!(!pack.lexicon.entries.is_empty());
        // Case-insensitive gloss + alias matching.
        let e = pack.match_gloss("reply YES").expect("alias match");
        assert_eq!(e.intent.as_deref(), Some("reply_email"));
        let c = pack.match_gloss("cancel").expect("control match");
        assert_eq!(c.kind, EntryKind::Control);
        assert!(pack.match_gloss("definitely not a sign").is_none());
        // Every command entry with an intent resolves in the catalog.
        for e in &pack.lexicon.entries {
            if let Some(intent) = &e.intent {
                assert!(pack.intent(intent).is_some(), "missing intent {intent}");
            }
        }
        let opts = pack.intent_options();
        assert!(opts.contains(&"dictate".to_string()));
        assert!(opts.len() <= 255);
    }

    #[test]
    fn csl_pack_loads_and_matches() {
        let pack = LanguagePack::load(&csl_dir()).expect("packs/csl must load");
        assert_eq!(pack.manifest.id, "csl");
        assert_eq!(pack.manifest.tier, "C");
        assert!(!pack.lexicon.entries.is_empty());
        // Chinese gloss + English alias both resolve (bilingual matching).
        let e = pack.match_gloss("回复是").expect("csl gloss match");
        assert_eq!(e.intent.as_deref(), Some("reply_email"));
        let a = pack.match_gloss("reply yes").expect("english alias match");
        assert_eq!(a.intent.as_deref(), Some("reply_email"));
        let c = pack.match_gloss("取消").expect("control match");
        assert_eq!(c.kind, EntryKind::Control);
        assert!(pack.match_gloss("definitely not a sign").is_none());
        // Same intent catalog shape as ASL: every intent resolves.
        for e in &pack.lexicon.entries {
            if let Some(intent) = &e.intent {
                assert!(pack.intent(intent).is_some(), "missing intent {intent}");
            }
        }
        // Same command surface as ASL (parity: 26 entries).
        let asl = LanguagePack::load(&asl_dir()).expect("packs/asl must load");
        assert_eq!(pack.lexicon.entries.len(), asl.lexicon.entries.len());
        assert_eq!(
            pack.intents.intents.len(),
            asl.intents.intents.len(),
            "intent catalog parity"
        );
    }

    #[test]
    fn nbest_sorts_truncates_normalizes() {
        let n = NBest::new(
            vec![
                GlossHypothesis {
                    gloss: "B".into(),
                    confidence: 0.2,
                },
                GlossHypothesis {
                    gloss: "A".into(),
                    confidence: 0.6,
                },
                GlossHypothesis {
                    gloss: "C".into(),
                    confidence: 0.2,
                },
            ],
            2,
        );
        assert_eq!(n.hyps.len(), 2);
        assert_eq!(n.top().unwrap().gloss, "A");
        let sum: f64 = n.hyps.iter().map(|h| h.confidence).sum();
        assert!((sum - 1.0).abs() < 1e-9);
        // Degenerate input -> uniform, never NaN.
        let u = NBest::new(
            vec![
                GlossHypothesis {
                    gloss: "X".into(),
                    confidence: 0.0,
                },
                GlossHypothesis {
                    gloss: "Y".into(),
                    confidence: -1.0,
                },
            ],
            5,
        );
        assert!((u.hyps[0].confidence - 0.5).abs() < 1e-9);
    }
}
