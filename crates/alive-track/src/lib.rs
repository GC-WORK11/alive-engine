//! ALive understanding engine.
//!
//! - [`kalman`]: predictive hand tracking (UI/overlay/reaction only).
//! - [`upsample`]: causal temporal super-resolution (matcher/endpointer).

pub mod kalman;
pub mod upsample;

pub use kalman::{HandTrack, HandTracker, OneEuro, TrackState};
pub use upsample::{real_span, real_span_ok, PoseSample, Provenance, Scheme, Tick, Upsampler, Centroid};
