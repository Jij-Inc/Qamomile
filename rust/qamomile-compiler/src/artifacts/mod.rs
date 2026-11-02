//! Persistent envelopes, ABIs, result codecs, and bounded execution data.

pub(crate) mod abi;
pub(crate) mod artifact;
pub(crate) mod artifact_abi;
pub(crate) mod artifact_family;
pub(crate) mod execution_snapshot;
pub(crate) mod expectation;
pub(crate) mod result_codec;
pub(crate) mod result_postprocessing;
pub(crate) mod runtime;
pub(crate) mod sample_counts;
pub(crate) mod simulation;
pub(crate) mod statistics;

#[allow(clippy::all, clippy::pedantic, dead_code)]
pub(crate) mod wire;
