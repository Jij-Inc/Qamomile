//! Target-neutral compiler transformations over verified `QCore`.

pub(crate) mod adjoint_lowering;
pub(crate) mod classical_cse;
pub(crate) mod cleanup;
pub(crate) mod cleanup_synthesis;
pub(crate) mod evaluation;
pub(crate) mod gate_rewrite;
pub(crate) mod optimization_pipeline;
pub(crate) mod reversible_regions;
pub(crate) mod semantic_identity;
pub(crate) mod specialization;
pub(crate) mod standard_optimization;
pub(crate) mod static_diagnostics;
pub(crate) mod static_values;
