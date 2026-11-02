//! Canonical persistence for immutable polynomial optimization problems.

use num_bigint::BigUint;
use prost::Message;
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::budget;
use crate::polynomial::{
    CanonicalPolynomial, MAX_POLYNOMIAL_COMPONENTS, MAX_POLYNOMIAL_TERMS, MAX_POLYNOMIAL_VARIABLES,
    PolynomialDomain, PolynomialError, normalize_polynomial,
};
use crate::wire;

const SCHEMA_MAJOR: u32 = 1;
const SCHEMA_MINOR: u32 = 0;
const ARTIFACT_KIND: &str = "qamomile.optimization.polynomial";

/// Maximum encoded size of one canonical optimization-problem artifact.
pub const MAX_OPTIMIZATION_PROBLEM_BYTES: usize = 64 * 1024 * 1024;

/// One integrity-checked immutable polynomial optimization problem.
#[derive(Clone, Debug, PartialEq)]
pub struct VerifiedOptimizationProblem {
    polynomial: CanonicalPolynomial,
    problem_digest: [u8; 32],
}

impl VerifiedOptimizationProblem {
    /// Return the canonical polynomial carried by this problem artifact.
    #[must_use]
    pub const fn polynomial(&self) -> &CanonicalPolynomial {
        &self.polynomial
    }

    /// Return the SHA-256 identity of the canonical unsealed problem.
    #[must_use]
    pub const fn problem_digest(&self) -> &[u8; 32] {
        &self.problem_digest
    }
}

/// Failure at the canonical optimization-problem persistence boundary.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum OptimizationProblemError {
    /// The complete encoded artifact exceeds its total byte budget.
    #[error("Q7570: optimization problem exceeds the wire-size limit")]
    TooLarge,
    /// Borrowed Protobuf traversal found malformed or excessive structure.
    #[error("Q7571: invalid optimization problem wire: {0}")]
    InvalidWire(String),
    /// Prost could not decode the bounded payload.
    #[error("Q7571: malformed optimization problem: {0}")]
    Malformed(String),
    /// The artifact schema cannot be consumed by this compiler.
    #[error("Q7572: unsupported optimization problem schema {major}.{minor}")]
    UnsupportedSchema {
        /// Schema major observed on the wire.
        major: u32,
        /// Schema minor observed on the wire.
        minor: u32,
    },
    /// The artifact kind or polynomial domain is not in the closed vocabulary.
    #[error("Q7573: invalid optimization problem identity")]
    InvalidIdentity,
    /// The digest does not cover the canonical unsealed problem.
    #[error("Q7574: optimization problem digest mismatch")]
    DigestMismatch,
    /// The message has an alternative encoding of the same or invalid data.
    #[error("Q7575: optimization problem is not canonically encoded")]
    NonCanonical,
    /// The decoded mathematical model violates the polynomial contract.
    #[error(transparent)]
    Polynomial(#[from] PolynomialError),
}

/// Encode and integrity-seal one canonical polynomial problem.
///
/// # Errors
///
/// Returns [`OptimizationProblemError::TooLarge`] when the canonical payload
/// exceeds the portable artifact byte budget.
pub fn encode_optimization_problem(
    polynomial: &CanonicalPolynomial,
) -> Result<Vec<u8>, OptimizationProblemError> {
    let mut problem = unsealed_problem(polynomial);
    let digest = digest_unsealed_problem(&problem)?;
    problem.problem_digest.extend_from_slice(&digest);
    let payload = problem.encode_to_vec();
    ensure_wire_size(payload.len())?;
    Ok(payload)
}

/// Return the stable semantic identity of one canonical polynomial problem.
///
/// # Errors
///
/// Returns [`OptimizationProblemError::TooLarge`] when the canonical unsealed
/// representation exceeds the portable artifact byte budget.
pub fn optimization_problem_digest(
    polynomial: &CanonicalPolynomial,
) -> Result<[u8; 32], OptimizationProblemError> {
    digest_unsealed_problem(&unsealed_problem(polynomial))
}

/// Decode, integrity-check, and validate one polynomial problem artifact.
///
/// # Errors
///
/// Returns [`OptimizationProblemError`] when the payload is oversized,
/// malformed, incompatible, corrupted, noncanonical, or semantically invalid.
pub fn decode_optimization_problem(
    payload: &[u8],
) -> Result<VerifiedOptimizationProblem, OptimizationProblemError> {
    let mut problem = decode_message(payload)?;
    validate_identity(&problem)?;
    let observed: [u8; 32] = problem
        .problem_digest
        .as_slice()
        .try_into()
        .map_err(|_| OptimizationProblemError::DigestMismatch)?;
    problem.problem_digest.clear();
    let expected: [u8; 32] = Sha256::digest(problem.encode_to_vec()).into();
    if observed != expected {
        return Err(OptimizationProblemError::DigestMismatch);
    }
    let polynomial = decode_polynomial(&problem)?;
    if unsealed_problem(&polynomial) != problem {
        return Err(OptimizationProblemError::NonCanonical);
    }
    problem.problem_digest.extend_from_slice(&observed);
    if problem.encode_to_vec() != payload {
        return Err(OptimizationProblemError::NonCanonical);
    }
    Ok(VerifiedOptimizationProblem {
        polynomial,
        problem_digest: observed,
    })
}

fn decode_message(
    payload: &[u8],
) -> Result<wire::OptimizationProblemProto, OptimizationProblemError> {
    ensure_wire_size(payload.len())?;
    budget::preflight_optimization_problem(payload).map_err(|error| match error {
        budget::PreflightError::Budget(detail) | budget::PreflightError::Malformed(detail) => {
            OptimizationProblemError::InvalidWire(detail)
        }
    })?;
    wire::OptimizationProblemProto::decode(payload)
        .map_err(|error| OptimizationProblemError::Malformed(error.to_string()))
}

const fn ensure_wire_size(size: usize) -> Result<(), OptimizationProblemError> {
    if size > MAX_OPTIMIZATION_PROBLEM_BYTES {
        Err(OptimizationProblemError::TooLarge)
    } else {
        Ok(())
    }
}

fn validate_identity(
    problem: &wire::OptimizationProblemProto,
) -> Result<(), OptimizationProblemError> {
    if problem.schema_major != SCHEMA_MAJOR || problem.schema_minor > SCHEMA_MINOR {
        return Err(OptimizationProblemError::UnsupportedSchema {
            major: problem.schema_major,
            minor: problem.schema_minor,
        });
    }
    if problem.artifact_kind != ARTIFACT_KIND {
        return Err(OptimizationProblemError::InvalidIdentity);
    }
    Ok(())
}

fn decode_polynomial(
    problem: &wire::OptimizationProblemProto,
) -> Result<CanonicalPolynomial, OptimizationProblemError> {
    let domain = match wire::optimization_problem_proto::Domain::try_from(problem.domain) {
        Ok(wire::optimization_problem_proto::Domain::Binary) => PolynomialDomain::Binary,
        Ok(wire::optimization_problem_proto::Domain::Spin) => PolynomialDomain::Spin,
        _ => return Err(OptimizationProblemError::InvalidIdentity),
    };
    if problem.terms.len() > MAX_POLYNOMIAL_TERMS
        || problem
            .terms
            .iter()
            .try_fold(0_usize, |total, term| {
                total.checked_add(term.variables.len())
            })
            .is_none_or(|count| count > MAX_POLYNOMIAL_COMPONENTS)
    {
        return Err(OptimizationProblemError::Polynomial(
            PolynomialError::InvalidTerms,
        ));
    }
    if problem.variables.len() > MAX_POLYNOMIAL_VARIABLES {
        return Err(OptimizationProblemError::Polynomial(
            PolynomialError::InvalidVariables,
        ));
    }
    let terms = problem
        .terms
        .iter()
        .map(|term| {
            term.variables
                .iter()
                .map(|variable| decode_variable(variable))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let coefficients = problem
        .terms
        .iter()
        .map(|term| term.coefficient)
        .collect::<Vec<_>>();
    let variables = problem
        .variables
        .iter()
        .map(|variable| decode_variable(variable))
        .collect::<Result<Vec<_>, _>>()?;
    normalize_polynomial(
        domain,
        &terms,
        &coefficients,
        problem.constant,
        Some(&variables),
    )
    .map_err(OptimizationProblemError::Polynomial)
}

fn decode_variable(value: &str) -> Result<BigUint, OptimizationProblemError> {
    let variable =
        BigUint::parse_bytes(value.as_bytes(), 16).ok_or(OptimizationProblemError::NonCanonical)?;
    if variable.to_str_radix(16) != value {
        return Err(OptimizationProblemError::NonCanonical);
    }
    Ok(variable)
}

fn unsealed_problem(polynomial: &CanonicalPolynomial) -> wire::OptimizationProblemProto {
    let domain = match polynomial.domain() {
        PolynomialDomain::Binary => wire::optimization_problem_proto::Domain::Binary,
        PolynomialDomain::Spin => wire::optimization_problem_proto::Domain::Spin,
    };
    wire::OptimizationProblemProto {
        schema_major: SCHEMA_MAJOR,
        schema_minor: SCHEMA_MINOR,
        artifact_kind: ARTIFACT_KIND.to_owned(),
        domain: domain.into(),
        terms: polynomial
            .terms()
            .iter()
            .map(|term| wire::PolynomialTermProto {
                variables: term
                    .variables()
                    .iter()
                    .map(|variable| variable.to_str_radix(16))
                    .collect(),
                coefficient: term.coefficient(),
            })
            .collect(),
        constant: polynomial.constant(),
        variables: polynomial
            .variables()
            .iter()
            .map(|variable| variable.to_str_radix(16))
            .collect(),
        problem_digest: Vec::new(),
    }
}

fn digest_unsealed_problem(
    problem: &wire::OptimizationProblemProto,
) -> Result<[u8; 32], OptimizationProblemError> {
    let payload = problem.encode_to_vec();
    ensure_wire_size(payload.len())?;
    Ok(Sha256::digest(payload).into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> CanonicalPolynomial {
        normalize_polynomial(
            PolynomialDomain::Binary,
            &[vec![BigUint::from(9_u8), BigUint::from(2_u8)]],
            &[1.5],
            -0.25,
            Some(&[BigUint::from(2_u8), BigUint::from(9_u8)]),
        )
        .expect("canonical polynomial")
    }

    fn spin_model() -> CanonicalPolynomial {
        normalize_polynomial(
            PolynomialDomain::Spin,
            &[vec![BigUint::from(9_u8), BigUint::from(9_u8)]],
            &[1.5],
            -0.25,
            Some(&[BigUint::from(2_u8), BigUint::from(9_u8)]),
        )
        .expect("canonical spin polynomial")
    }

    fn reseal(problem: &mut wire::OptimizationProblemProto) -> Vec<u8> {
        problem.problem_digest.clear();
        let digest: [u8; 32] = Sha256::digest(problem.encode_to_vec()).into();
        problem.problem_digest.extend_from_slice(&digest);
        problem.encode_to_vec()
    }

    #[test]
    fn round_trips_one_canonical_problem_deterministically() {
        for polynomial in [model(), spin_model()] {
            let first = encode_optimization_problem(&polynomial).expect("encode problem");
            let second = encode_optimization_problem(&polynomial).expect("encode problem again");
            let verified = decode_optimization_problem(&first).expect("decode problem");

            assert_eq!(first, second);
            assert_eq!(verified.polynomial(), &polynomial);
            assert_eq!(verified.problem_digest().len(), 32);
        }
    }

    #[test]
    fn rejects_digest_tampering_and_resealed_noncanonical_order() {
        let payload = encode_optimization_problem(&model()).expect("encode problem");
        let mut tampered = wire::OptimizationProblemProto::decode(payload.as_slice())
            .expect("decode test payload");
        tampered.terms[0].coefficient = 2.0;
        assert_eq!(
            decode_optimization_problem(&tampered.encode_to_vec()),
            Err(OptimizationProblemError::DigestMismatch),
        );

        let mut reordered = wire::OptimizationProblemProto::decode(payload.as_slice())
            .expect("decode test payload");
        reordered.variables.reverse();
        assert_eq!(
            decode_optimization_problem(&reseal(&mut reordered)),
            Err(OptimizationProblemError::NonCanonical),
        );
    }

    #[test]
    fn rejects_unknown_identity_malformed_wire_and_size_boundary() {
        let payload = encode_optimization_problem(&model()).expect("encode problem");
        let mut future = wire::OptimizationProblemProto::decode(payload.as_slice())
            .expect("decode test payload");
        future.schema_minor = SCHEMA_MINOR + 1;
        assert_eq!(
            decode_optimization_problem(&future.encode_to_vec()),
            Err(OptimizationProblemError::UnsupportedSchema {
                major: SCHEMA_MAJOR,
                minor: SCHEMA_MINOR + 1,
            }),
        );
        let mut wrong_kind = wire::OptimizationProblemProto::decode(payload.as_slice())
            .expect("decode test payload");
        wrong_kind.artifact_kind = "qamomile.optimization.future".to_owned();
        assert_eq!(
            decode_optimization_problem(&wrong_kind.encode_to_vec()),
            Err(OptimizationProblemError::InvalidIdentity),
        );
        assert!(matches!(
            decode_optimization_problem(&[0xff]),
            Err(OptimizationProblemError::InvalidWire(_))
        ));
        assert_eq!(
            ensure_wire_size(MAX_OPTIMIZATION_PROBLEM_BYTES + 1),
            Err(OptimizationProblemError::TooLarge),
        );
    }
}
