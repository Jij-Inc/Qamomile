//! Generic approximation evidence attached to a late realization plan.
//!
//! Approximation is a property of an implementation of a semantic operation,
//! not a new semantic opcode or data representation in persistent `QCore`.
//! Producers may use representation-specific proofs internally, while every
//! consumer sees the same output-distance contract.

/// Public metric identity for the distance between ideal and realized output.
pub const OUTPUT_TRACE_DISTANCE_BOUND_METRIC: &str = "qamomile.circuit.output_trace_distance_bound";

/// Accuracy evidence carried by one selected realization.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ApproximationCertificate<T> {
    /// The realization preserves the semantic operation exactly.
    Exact,
    /// A proved upper bound on output trace distance.
    OutputTraceDistanceBound(T),
}

impl<T> ApproximationCertificate<T> {
    /// Construct a checked realization-provided trace-distance certificate.
    #[must_use]
    pub const fn output_trace_distance_bound(upper: T) -> Self {
        Self::OutputTraceDistanceBound(upper)
    }

    /// Transform the certificate payload without changing its meaning.
    #[must_use]
    pub fn map<U>(self, map: impl FnOnce(T) -> U) -> ApproximationCertificate<U> {
        match self {
            Self::Exact => ApproximationCertificate::Exact,
            Self::OutputTraceDistanceBound(upper) => {
                ApproximationCertificate::OutputTraceDistanceBound(map(upper))
            }
        }
    }
}
