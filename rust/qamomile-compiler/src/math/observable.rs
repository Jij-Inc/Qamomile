//! Canonical backend-neutral Pauli observable algebra.
//!
//! Public Python values use the existing portable `LiteralProto` schema, while
//! this module owns normalization, exact Pauli-word multiplication, and the
//! Hermiticity/commutation facts consumed by compilation and estimation.

use std::collections::BTreeMap;

use num_bigint::BigInt;
use num_complex::Complex64;
use num_traits::ToPrimitive;
use prost::Message;
use thiserror::Error;

use crate::binary64::{
    SUBNORMAL_EXPONENT, exact_sum, float_to_subnormal_units, fwht_integer_in_place,
    scaled_integer_to_f64,
};
use crate::literals::validate_literal_references;
use crate::{budget, wire};

/// One normalized portable observable and its compiler-derived facts.
#[derive(Clone, Debug, PartialEq)]
pub struct NormalizedObservable {
    literal: Vec<u8>,
    hermitian: bool,
}

/// One bounded dense-matrix Pauli decomposition and its truncation evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct DensePauliDecomposition {
    observable: NormalizedObservable,
    truncation_error_bound: f64,
}

/// Dense Hamiltonian and overlap matrices projected onto Pauli product states.
#[derive(Clone, Debug, PartialEq)]
pub struct ProjectedSubspace {
    size: usize,
    hamiltonian: Vec<Complex64>,
    overlap: Vec<Complex64>,
}

impl ProjectedSubspace {
    /// Return the common square matrix dimension.
    #[must_use]
    pub const fn size(&self) -> usize {
        self.size
    }

    /// Return row-major projected Hamiltonian entries.
    #[must_use]
    pub fn hamiltonian(&self) -> &[Complex64] {
        &self.hamiltonian
    }

    /// Return row-major overlap entries.
    #[must_use]
    pub fn overlap(&self) -> &[Complex64] {
        &self.overlap
    }
}

impl DensePauliDecomposition {
    /// Return the canonical complex Pauli sum.
    #[must_use]
    pub const fn observable(&self) -> &NormalizedObservable {
        &self.observable
    }

    /// Return a conservative one-norm bound for deliberately omitted terms.
    #[must_use]
    pub const fn truncation_error_bound(&self) -> f64 {
        self.truncation_error_bound
    }
}

impl NormalizedObservable {
    /// Return the canonical portable `LiteralProto` bytes.
    #[must_use]
    pub fn literal(&self) -> &[u8] {
        &self.literal
    }

    /// Return whether every coefficient is exactly real.
    #[must_use]
    pub const fn is_hermitian(&self) -> bool {
        self.hermitian
    }
}

/// Failure to decode or evaluate one portable Pauli observable.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum ObservableError {
    /// The standalone literal exceeds the shared structural budget.
    #[error("Q7401: observable literal exceeds the structural budget")]
    Budget,
    /// The bytes are not one canonical portable literal.
    #[error("Q7402: observable literal is not canonical")]
    NonCanonical,
    /// The literal does not encode the closed Pauli-sum domain.
    #[error("Q7403: malformed portable Hamiltonian: {0}")]
    Malformed(&'static str),
    /// A logical index or inferred width cannot be represented.
    #[error("Q7404: observable logical width is not representable")]
    WidthOverflow,
    /// A dense matrix has an invalid shape or entry count.
    #[error("Q7405: invalid dense matrix: {0}")]
    InvalidMatrix(&'static str),
    /// A dense matrix exceeds the bounded exponential decomposition domain.
    #[error("Q7406: dense matrix exceeds the 256 x 256 decomposition budget")]
    MatrixBudget,
    /// A matrix or tolerance contains a non-finite value.
    #[error("Q7407: dense matrix and tolerances must be finite")]
    NonFiniteMatrix,
    /// A requested absolute tolerance is negative.
    #[error("Q7408: dense matrix tolerances must be non-negative")]
    NegativeTolerance,
    /// The input is not Hermitian within the requested absolute tolerance.
    #[error("Q7409: input matrix is not Hermitian within absolute tolerance")]
    NonHermitianMatrix,
    /// A purported Hermitian matrix produced a complex Pauli coefficient.
    #[error("Q7410: Pauli decomposition has non-negligible imaginary residue")]
    ImaginaryResidue,
    /// A resulting Pauli coefficient cannot be represented as binary64.
    #[error("Q7411: Pauli coefficient exceeds the finite binary64 range")]
    CoefficientRange,
    /// A sampled Pauli product-state basis is malformed.
    #[error("Q7412: invalid sampled Pauli subspace: {0}")]
    InvalidSubspace(&'static str),
    /// A sampled projection exceeds its explicit polynomial work budget.
    #[error("Q7413: sampled Pauli subspace exceeds the projection budget")]
    SubspaceBudget,
    /// The observable coefficient one-norm is outside finite binary64.
    #[error("Q7414: observable one-norm exceeds the finite binary64 range")]
    OneNormRange,
    /// A logical-index substitution is outside the source width or collapses a word.
    #[error("Q7415: invalid observable remap: {0}")]
    InvalidRemap(&'static str),
    /// Division of a dense Hermitian matrix by exact zero was requested.
    #[error("Q7416: Hermitian matrix division by zero")]
    MatrixDivisionByZero,
    /// Dense Hermitian operands have different shapes.
    #[error("Q7417: Shape mismatch for Hermitian matrices")]
    MatrixShapeMismatch,
}

/// Maximum dense entries accepted before exponential Pauli decomposition.
pub const MAX_DENSE_MATRIX_ENTRIES: usize = 256 * 256;
const HERMITIAN_COEFFICIENT_IMAG_TOLERANCE: f64 = 1e-10;
const DEFAULT_HERMITIAN_TOLERANCE: f64 = 1e-10;
const MAX_SUBSPACE_CELLS: usize = 1_000_000;
const MAX_SUBSPACE_WORK: usize = 50_000_000;

/// Normalize one portable Hamiltonian and derive its semantic facts.
///
/// # Errors
///
/// Returns [`ObservableError`] when the payload is noncanonical, exceeds the
/// shared literal budget, or does not inhabit the portable Pauli-sum domain.
pub fn normalize_observable(payload: &[u8]) -> Result<NormalizedObservable, ObservableError> {
    let observable = Observable::decode(payload)?;
    Ok(observable.normalized())
}

/// Return the exact-sum coefficient one-norm of one canonical observable.
///
/// # Errors
///
/// Returns [`ObservableError`] when the payload is invalid or its finite
/// coefficient magnitudes cannot be summed into binary64.
pub fn observable_lcu_one_norm(payload: &[u8]) -> Result<f64, ObservableError> {
    let observable = Observable::decode(payload)?;
    exact_sum(
        std::iter::once(observable.constant.norm()).chain(
            observable
                .terms
                .values()
                .map(|coefficient| coefficient.norm()),
        ),
    )
    .ok_or(ObservableError::OneNormRange)
}

/// Build one canonical real Pauli sum from checked native coordinates.
///
/// # Errors
///
/// Returns [`ObservableError`] when width, factors, coefficients, or the
/// resulting portable literal violate the common observable domain.
pub fn normalize_real_observable(
    num_qubits: usize,
    constant: f64,
    terms: &[(Vec<(usize, Pauli)>, f64)],
) -> Result<NormalizedObservable, ObservableError> {
    let num_qubits = u64::try_from(num_qubits).map_err(|_| ObservableError::WidthOverflow)?;
    if !constant.is_finite() {
        return Err(ObservableError::Malformed(
            "real coefficients must be finite",
        ));
    }
    let mut grouped = BTreeMap::<PauliWord, Vec<f64>>::new();
    for (raw_word, coefficient) in terms {
        if !coefficient.is_finite() || raw_word.is_empty() {
            return Err(ObservableError::Malformed(
                "real Pauli terms must be nonempty and finite",
            ));
        }
        let mut word = Vec::with_capacity(raw_word.len());
        let mut previous = None;
        for (index, pauli) in raw_word {
            let index = u64::try_from(*index).map_err(|_| ObservableError::WidthOverflow)?;
            if index >= num_qubits || previous.is_some_and(|value| value >= index) {
                return Err(ObservableError::Malformed(
                    "Pauli factors must be in-range unique ordered indices",
                ));
            }
            previous = Some(index);
            word.push((index, *pauli));
        }
        grouped.entry(word).or_default().push(*coefficient);
    }
    let mut normalized_terms = BTreeMap::new();
    for (word, coefficients) in grouped {
        let coefficient = exact_sum(coefficients).ok_or(ObservableError::CoefficientRange)?;
        if coefficient != 0.0 {
            normalized_terms.insert(word, Complex64::new(coefficient, 0.0));
        }
    }
    let observable = Observable {
        terms: normalized_terms,
        constant: Complex64::new(constant, 0.0),
        num_qubits,
    }
    .normalized();
    budget::preflight_literal(observable.literal()).map_err(|_| ObservableError::Budget)?;
    Ok(observable)
}

/// Validate one dense matrix and return its logical qubit width.
///
/// Shape, finite-value, and budget checks are unconditional. The Hermitian
/// proof can be skipped for trusted algebraic intermediates while retaining
/// every structural check.
///
/// # Errors
///
/// Returns [`ObservableError`] when the shape, entries, tolerance, budget, or
/// requested Hermitian proof is invalid.
pub fn validate_hermitian_matrix(
    shape: &[usize],
    entries: &[Complex64],
    require_hermitian: bool,
    atol: f64,
) -> Result<u64, ObservableError> {
    let (dimension, num_qubits) = validate_dense_matrix(shape, entries)?;
    validate_tolerance(atol)?;
    if require_hermitian {
        for row in 0..dimension {
            for column in 0..=row {
                let left = entries[row * dimension + column];
                let right = entries[column * dimension + row].conj();
                if (left - right).norm() > atol {
                    return Err(ObservableError::NonHermitianMatrix);
                }
            }
        }
    }
    Ok(num_qubits)
}

/// Add a real multiple of one dense Hermitian matrix to another.
///
/// Both operands retain the structural and finite-value guarantees of the
/// public dense-matrix domain. The result is checked with the public default
/// Hermitian tolerance, matching construction of the former Python result.
///
/// # Errors
///
/// Returns [`ObservableError`] when either operand is malformed, their shapes
/// differ, the scale or result is non-finite, or the result is not Hermitian.
pub fn add_scaled_hermitian_matrices(
    left_shape: &[usize],
    left_entries: &[Complex64],
    right_shape: &[usize],
    right_entries: &[Complex64],
    right_scale: f64,
) -> Result<Vec<Complex64>, ObservableError> {
    validate_dense_matrix(left_shape, left_entries)?;
    validate_dense_matrix(right_shape, right_entries)?;
    if left_shape != right_shape {
        return Err(ObservableError::MatrixShapeMismatch);
    }
    if !right_scale.is_finite() {
        return Err(ObservableError::NonFiniteMatrix);
    }
    let result = left_entries
        .iter()
        .zip(right_entries)
        .map(|(left, right)| *left + *right * right_scale)
        .collect::<Vec<_>>();
    validate_hermitian_matrix(left_shape, &result, true, DEFAULT_HERMITIAN_TOLERANCE)?;
    Ok(result)
}

/// Scale or divide one dense Hermitian matrix by a real scalar.
///
/// # Errors
///
/// Returns [`ObservableError`] when the matrix is malformed, the scalar or
/// result is non-finite, exact-zero division is requested, or the result is
/// not Hermitian within the public default tolerance.
pub fn scale_hermitian_matrix(
    shape: &[usize],
    entries: &[Complex64],
    scalar: f64,
    divide: bool,
) -> Result<Vec<Complex64>, ObservableError> {
    validate_dense_matrix(shape, entries)?;
    if !scalar.is_finite() {
        return Err(ObservableError::NonFiniteMatrix);
    }
    if divide && scalar == 0.0 {
        return Err(ObservableError::MatrixDivisionByZero);
    }
    let result = if divide {
        entries
            .iter()
            .map(|entry| *entry / scalar)
            .collect::<Vec<_>>()
    } else {
        entries
            .iter()
            .map(|entry| *entry * scalar)
            .collect::<Vec<_>>()
    };
    validate_hermitian_matrix(shape, &result, true, DEFAULT_HERMITIAN_TOLERANCE)?;
    Ok(result)
}

/// Decompose a finite dense matrix into the canonical real Pauli basis.
///
/// The transform uses symplectic masks `(x_mask, z_mask)` and maps bit zero to
/// logical qubit zero, the least-significant computational-basis bit. Every
/// Walsh-Hadamard butterfly is accumulated as exact integer multiples of the
/// smallest binary64 subnormal before a single final rounding.
///
/// # Errors
///
/// Returns [`ObservableError`] when the matrix or tolerance is invalid, the
/// exponential transform budget is exceeded, the matrix has a significant
/// anti-Hermitian component, or a coefficient is not representable.
pub fn decompose_hermitian_matrix(
    shape: &[usize],
    entries: &[Complex64],
    tol: f64,
) -> Result<NormalizedObservable, ObservableError> {
    Ok(decompose_dense_pauli_matrix(shape, entries, tol, true)?.observable)
}

/// Decompose an arbitrary finite dense matrix into the complex Pauli basis.
///
/// # Errors
///
/// Returns [`ObservableError`] when the input is malformed, exceeds its
/// bounded exponential domain, or produces an unrepresentable result.
pub fn decompose_pauli_matrix(
    shape: &[usize],
    entries: &[Complex64],
    atol: f64,
) -> Result<DensePauliDecomposition, ObservableError> {
    decompose_dense_pauli_matrix(shape, entries, atol, false)
}

fn decompose_dense_pauli_matrix(
    shape: &[usize],
    entries: &[Complex64],
    tolerance: f64,
    require_real: bool,
) -> Result<DensePauliDecomposition, ObservableError> {
    let (dimension, num_qubits) = validate_dense_matrix(shape, entries)?;
    validate_tolerance(tolerance)?;
    let num_qubits_usize =
        usize::try_from(num_qubits).map_err(|_| ObservableError::WidthOverflow)?;
    let coefficient_denominator_exponent = SUBNORMAL_EXPONENT
        .checked_add(num_qubits_usize)
        .ok_or(ObservableError::WidthOverflow)?;
    let mut terms = BTreeMap::new();
    let mut constant = Complex64::new(0.0, 0.0);
    let mut truncation_error_bound = 0.0;
    let mut normalization_upper_bound = 0.0;

    for x_mask in 0..dimension {
        let mut real = Vec::with_capacity(dimension);
        let mut imaginary = Vec::with_capacity(dimension);
        for basis_index in 0..dimension {
            let value = entries[(basis_index ^ x_mask) * dimension + basis_index];
            real.push(float_to_subnormal_units(value.re));
            imaginary.push(float_to_subnormal_units(value.im));
        }
        fwht_integer_in_place(&mut real);
        fwht_integer_in_place(&mut imaginary);

        for z_mask in 0..dimension {
            let phase = (x_mask & z_mask).count_ones() & 3;
            let (real_units, imaginary_units) = match phase {
                0 => (real[z_mask].clone(), imaginary[z_mask].clone()),
                1 => (imaginary[z_mask].clone(), -real[z_mask].clone()),
                2 => (-real[z_mask].clone(), -imaginary[z_mask].clone()),
                3 => (-imaginary[z_mask].clone(), real[z_mask].clone()),
                _ => unreachable!("population count modulo four is closed"),
            };
            let mut coefficient = Complex64::new(
                scaled_integer_to_f64(&real_units, coefficient_denominator_exponent)
                    .ok_or(ObservableError::CoefficientRange)?,
                scaled_integer_to_f64(&imaginary_units, coefficient_denominator_exponent)
                    .ok_or(ObservableError::CoefficientRange)?,
            );
            if !coefficient.re.is_finite() || !coefficient.im.is_finite() {
                return Err(ObservableError::CoefficientRange);
            }
            if require_real && coefficient.im.abs() > HERMITIAN_COEFFICIENT_IMAG_TOLERANCE {
                return Err(ObservableError::ImaginaryResidue);
            }
            if require_real {
                coefficient.im = 0.0;
            }
            let magnitude = coefficient.norm();
            if !magnitude.is_finite() {
                return Err(ObservableError::CoefficientRange);
            }
            if magnitude <= tolerance {
                truncation_error_bound =
                    add_truncation_upper_bound(truncation_error_bound, magnitude)?;
                continue;
            }
            if !require_real {
                normalization_upper_bound =
                    add_normalization_upper_bound(normalization_upper_bound, magnitude)?;
            }
            if x_mask == 0 && z_mask == 0 {
                constant = coefficient;
                continue;
            }
            let mut word = Vec::with_capacity(num_qubits_usize);
            for index in 0..num_qubits {
                let bit = 1_usize << index;
                let pauli = match (x_mask & bit != 0, z_mask & bit != 0) {
                    (false, false) => continue,
                    (true, false) => Pauli::X,
                    (false, true) => Pauli::Z,
                    (true, true) => Pauli::Y,
                };
                word.push((index, pauli));
            }
            terms.insert(word, coefficient);
        }
    }

    let normalized = Observable {
        terms,
        constant,
        num_qubits,
    }
    .normalized();
    budget::preflight_literal(normalized.literal()).map_err(|_| ObservableError::Budget)?;
    Ok(DensePauliDecomposition {
        observable: normalized,
        truncation_error_bound,
    })
}

fn add_truncation_upper_bound(total: f64, value: f64) -> Result<f64, ObservableError> {
    if value == 0.0 {
        return Ok(total);
    }
    let sum = total + value;
    if !sum.is_finite() {
        return Err(ObservableError::CoefficientRange);
    }
    Ok(sum.next_up())
}

fn add_normalization_upper_bound(total: f64, value: f64) -> Result<f64, ObservableError> {
    if total == 0.0 {
        return Ok(value);
    }
    let sum = total + value;
    if !sum.is_finite() || !sum.next_up().is_finite() {
        return Err(ObservableError::CoefficientRange);
    }
    Ok(sum.next_up())
}

fn validate_dense_matrix(
    shape: &[usize],
    entries: &[Complex64],
) -> Result<(usize, u64), ObservableError> {
    let [rows, columns] = shape else {
        return Err(ObservableError::InvalidMatrix(
            "expected a two-dimensional shape",
        ));
    };
    if *rows == 0 || rows != columns {
        return Err(ObservableError::InvalidMatrix(
            "expected a non-empty square matrix",
        ));
    }
    if !rows.is_power_of_two() {
        return Err(ObservableError::InvalidMatrix(
            "dimension must be a power of two",
        ));
    }
    let entry_count = rows
        .checked_mul(*columns)
        .ok_or(ObservableError::MatrixBudget)?;
    if entry_count > MAX_DENSE_MATRIX_ENTRIES {
        return Err(ObservableError::MatrixBudget);
    }
    if entries.len() != entry_count {
        return Err(ObservableError::InvalidMatrix(
            "entry count does not match the matrix shape",
        ));
    }
    if entries
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(ObservableError::NonFiniteMatrix);
    }
    Ok((*rows, u64::from(rows.trailing_zeros())))
}

fn validate_tolerance(tolerance: f64) -> Result<(), ObservableError> {
    if !tolerance.is_finite() {
        return Err(ObservableError::NonFiniteMatrix);
    }
    if tolerance < 0.0 {
        return Err(ObservableError::NegativeTolerance);
    }
    Ok(())
}

/// Multiply two portable Hamiltonians using the exact Pauli algebra.
///
/// # Errors
///
/// Returns [`ObservableError`] when either operand is malformed or a resulting
/// logical width cannot be represented.
pub fn multiply_observables(
    left: &[u8],
    right: &[u8],
) -> Result<NormalizedObservable, ObservableError> {
    let left = Observable::decode(left)?;
    let right = Observable::decode(right)?;
    let mut terms = BTreeMap::new();
    let mut constant = left.constant * right.constant;
    for (left_word, left_coefficient) in &left.terms {
        for (right_word, right_coefficient) in &right.terms {
            let (word, phase) = multiply_words(left_word, right_word);
            let coefficient = *left_coefficient * *right_coefficient * phase;
            if word.is_empty() {
                constant += coefficient;
            } else {
                merge_term(&mut terms, word, coefficient);
            }
        }
    }
    for (word, coefficient) in &right.terms {
        merge_term(&mut terms, word.clone(), left.constant * *coefficient);
    }
    for (word, coefficient) in &left.terms {
        merge_term(&mut terms, word.clone(), right.constant * *coefficient);
    }
    Observable {
        terms,
        constant,
        num_qubits: left.num_qubits.max(right.num_qubits),
    }
    .checked_normalized()
}

/// Apply one complex scale and identity shift to a portable Hamiltonian.
///
/// # Errors
///
/// Returns [`ObservableError`] when the operand or scalar inputs are malformed,
/// or when a resulting coefficient is outside finite binary64.
pub fn affine_observable(
    payload: &[u8],
    scale: Complex64,
    shift: Complex64,
) -> Result<NormalizedObservable, ObservableError> {
    if !finite_complex(scale) || !finite_complex(shift) {
        return Err(ObservableError::Malformed(
            "observable affine scalars must be finite",
        ));
    }
    let mut observable = Observable::decode(payload)?;
    for coefficient in observable.terms.values_mut() {
        *coefficient *= scale;
    }
    observable.constant = observable.constant * scale + shift;
    observable.checked_normalized()
}

/// Add a complex multiple of one portable Hamiltonian to another.
///
/// # Errors
///
/// Returns [`ObservableError`] when either operand or the scale is malformed,
/// or when a resulting coefficient is outside finite binary64.
pub fn add_scaled_observable(
    left: &[u8],
    right: &[u8],
    right_scale: Complex64,
) -> Result<NormalizedObservable, ObservableError> {
    if !finite_complex(right_scale) {
        return Err(ObservableError::Malformed(
            "observable affine scalars must be finite",
        ));
    }
    let mut left = Observable::decode(left)?;
    let right = Observable::decode(right)?;
    for (word, coefficient) in right.terms {
        merge_term(&mut left.terms, word, coefficient * right_scale);
    }
    left.constant += right.constant * right_scale;
    left.num_qubits = left.num_qubits.max(right.num_qubits);
    left.checked_normalized()
}

/// Substitute logical indices in one portable Hamiltonian.
///
/// Explicit source width never shrinks, while an in-range source may map to a
/// wider destination. A substitution that collapses two factors of one Pauli
/// word is not a qubit relabeling and is rejected rather than assigned an
/// order-dependent local product.
///
/// # Errors
///
/// Returns [`ObservableError`] when the operand or mapping is malformed, a
/// source lies outside the explicit width, a destination overflows the width,
/// or two factors of one word map to the same destination.
pub fn remap_observable(
    payload: &[u8],
    indices: &[(u64, u64)],
) -> Result<NormalizedObservable, ObservableError> {
    let observable = Observable::decode(payload)?;
    let mut mapping = BTreeMap::new();
    let mut num_qubits = observable.num_qubits;
    for &(source, target) in indices {
        if source >= observable.num_qubits {
            return Err(ObservableError::InvalidRemap(
                "source index is outside the observable width",
            ));
        }
        if mapping.insert(source, target).is_some() {
            return Err(ObservableError::InvalidRemap(
                "source indices must be unique",
            ));
        }
        num_qubits = num_qubits.max(
            target
                .checked_add(1)
                .ok_or(ObservableError::WidthOverflow)?,
        );
    }
    let mut terms = BTreeMap::new();
    for (word, coefficient) in observable.terms {
        let mut remapped = BTreeMap::new();
        for (source, pauli) in word {
            let target = mapping.get(&source).copied().unwrap_or(source);
            if remapped.insert(target, pauli).is_some() {
                return Err(ObservableError::InvalidRemap(
                    "two factors of one word map to the same index",
                ));
            }
        }
        merge_term(&mut terms, remapped.into_iter().collect(), coefficient);
    }
    Observable {
        terms,
        constant: observable.constant,
        num_qubits,
    }
    .checked_normalized()
}

/// Determine whether all non-identity terms commute pairwise.
///
/// # Errors
///
/// Returns [`ObservableError`] when the operand is malformed. The quadratic
/// term-pair scan is deliberately separate from ordinary normalization so
/// constructing a large observable does not pay for an unused property.
pub fn observable_is_pairwise_commuting(payload: &[u8]) -> Result<bool, ObservableError> {
    let observable = Observable::decode(payload)?;
    let words = observable.terms.keys().collect::<Vec<_>>();
    Ok(words.iter().enumerate().all(|(position, left)| {
        words[position + 1..]
            .iter()
            .all(|right| words_commute(left, right))
    }))
}

/// Project one canonical Pauli sum onto sampled X/Y/Z product states.
///
/// Basis codes are `0 = X`, `1 = Y`, and `2 = Z`. Sample bit zero denotes
/// the positive eigenstate and bit one the negative eigenstate. Both output
/// matrices are row-major and use the source sample order.
///
/// # Errors
///
/// Returns [`ObservableError`] for malformed observable data, sample/basis
/// shapes or values, non-finite arithmetic, or an excessive projection.
pub fn project_pauli_subspace(
    payload: &[u8],
    samples: &[Vec<u8>],
    bases: &[Vec<u8>],
) -> Result<ProjectedSubspace, ObservableError> {
    let (observable, width, cells) = validate_subspace(payload, samples, bases)?;
    let size = samples.len();
    let mut hamiltonian = Vec::with_capacity(cells);
    let mut overlap = Vec::with_capacity(cells);
    for bra in 0..size {
        for ket in 0..size {
            let mut overlap_value = Complex64::new(1.0, 0.0);
            for index in 0..width {
                overlap_value *= local_pauli_element(
                    None,
                    bases[bra][index],
                    samples[bra][index],
                    bases[ket][index],
                    samples[ket][index],
                );
            }
            let mut hamiltonian_value = observable.constant * overlap_value;
            for (word, coefficient) in &observable.terms {
                hamiltonian_value += *coefficient
                    * pauli_word_element(
                        word,
                        &samples[bra],
                        &bases[bra],
                        &samples[ket],
                        &bases[ket],
                    )?;
            }
            if !overlap_value.re.is_finite()
                || !overlap_value.im.is_finite()
                || !hamiltonian_value.re.is_finite()
                || !hamiltonian_value.im.is_finite()
            {
                return Err(ObservableError::InvalidSubspace(
                    "projected entries must be finite",
                ));
            }
            overlap.push(canonical_complex(overlap_value));
            hamiltonian.push(canonical_complex(hamiltonian_value));
        }
    }
    Ok(ProjectedSubspace {
        size,
        hamiltonian,
        overlap,
    })
}

fn validate_subspace(
    payload: &[u8],
    samples: &[Vec<u8>],
    bases: &[Vec<u8>],
) -> Result<(Observable, usize, usize), ObservableError> {
    let observable = Observable::decode(payload)?;
    let width =
        usize::try_from(observable.num_qubits).map_err(|_| ObservableError::WidthOverflow)?;
    if samples.len() != bases.len() {
        return Err(ObservableError::InvalidSubspace(
            "sample and basis row counts differ",
        ));
    }
    for row in samples {
        if row.len() != width || row.iter().any(|bit| *bit > 1) {
            return Err(ObservableError::InvalidSubspace(
                "samples must be fixed-width bit rows",
            ));
        }
    }
    for row in bases {
        if row.len() != width || row.iter().any(|basis| *basis > 2) {
            return Err(ObservableError::InvalidSubspace(
                "bases must be fixed-width X/Y/Z rows",
            ));
        }
    }
    if !observable.constant.re.is_finite()
        || !observable.constant.im.is_finite()
        || observable
            .terms
            .values()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(ObservableError::InvalidSubspace(
            "observable coefficients must be finite",
        ));
    }
    let cells = samples
        .len()
        .checked_mul(samples.len())
        .ok_or(ObservableError::SubspaceBudget)?;
    if cells > MAX_SUBSPACE_CELLS {
        return Err(ObservableError::SubspaceBudget);
    }
    let work = cells
        .checked_mul(observable.terms.len().saturating_add(1))
        .and_then(|value| value.checked_mul(width.max(1)))
        .ok_or(ObservableError::SubspaceBudget)?;
    if work > MAX_SUBSPACE_WORK {
        return Err(ObservableError::SubspaceBudget);
    }
    Ok((observable, width, cells))
}

fn pauli_word_element(
    word: &PauliWord,
    bra_sample: &[u8],
    bra_bases: &[u8],
    ket_sample: &[u8],
    ket_bases: &[u8],
) -> Result<Complex64, ObservableError> {
    let mut value = Complex64::new(1.0, 0.0);
    let mut factor_position = 0;
    for index in 0..bra_sample.len() {
        let index_u64 = u64::try_from(index).map_err(|_| ObservableError::WidthOverflow)?;
        let pauli = if word
            .get(factor_position)
            .is_some_and(|(factor_index, _)| *factor_index == index_u64)
        {
            let pauli = word[factor_position].1;
            factor_position += 1;
            Some(pauli)
        } else {
            None
        };
        value *= local_pauli_element(
            pauli,
            bra_bases[index],
            bra_sample[index],
            ket_bases[index],
            ket_sample[index],
        );
        if value == Complex64::new(0.0, 0.0) {
            break;
        }
    }
    Ok(value)
}

fn local_pauli_element(
    pauli: Option<Pauli>,
    bra_basis: u8,
    bra_bit: u8,
    ket_basis: u8,
    ket_bit: u8,
) -> Complex64 {
    let inverse_sqrt_two = std::f64::consts::FRAC_1_SQRT_2;
    let state = |basis: u8, bit: u8| match (basis, bit) {
        (0, 0) => [
            Complex64::new(inverse_sqrt_two, 0.0),
            Complex64::new(inverse_sqrt_two, 0.0),
        ],
        (0, 1) => [
            Complex64::new(inverse_sqrt_two, 0.0),
            Complex64::new(-inverse_sqrt_two, 0.0),
        ],
        (1, 0) => [
            Complex64::new(inverse_sqrt_two, 0.0),
            Complex64::new(0.0, inverse_sqrt_two),
        ],
        (1, 1) => [
            Complex64::new(inverse_sqrt_two, 0.0),
            Complex64::new(0.0, -inverse_sqrt_two),
        ],
        (2, 0) => [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        (2, 1) => [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        _ => unreachable!("validated Pauli basis state"),
    };
    let bra = state(bra_basis, bra_bit);
    let ket = state(ket_basis, ket_bit);
    let applied = match pauli {
        None => ket,
        Some(Pauli::X) => [ket[1], ket[0]],
        Some(Pauli::Y) => [
            Complex64::new(0.0, -1.0) * ket[1],
            Complex64::new(0.0, 1.0) * ket[0],
        ],
        Some(Pauli::Z) => [ket[0], -ket[1]],
    };
    bra[0].conj() * applied[0] + bra[1].conj() * applied[1]
}

/// One non-identity single-lane Pauli operator.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Pauli {
    /// Pauli X.
    X,
    /// Pauli Y.
    Y,
    /// Pauli Z.
    Z,
}

pub(crate) type PauliWord = Vec<(u64, Pauli)>;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Observable {
    pub(crate) terms: BTreeMap<PauliWord, Complex64>,
    pub(crate) constant: Complex64,
    pub(crate) num_qubits: u64,
}

impl Observable {
    pub(crate) fn decode(payload: &[u8]) -> Result<Self, ObservableError> {
        budget::preflight_literal(payload).map_err(|_| ObservableError::Budget)?;
        let literal =
            wire::LiteralProto::decode(payload).map_err(|_| ObservableError::NonCanonical)?;
        validate_literal_references(vec![&literal]).map_err(|_| ObservableError::NonCanonical)?;
        let entries =
            mapping_entries(&literal).ok_or(ObservableError::Malformed("expected a mapping"))?;
        if entries.len() != 4
            || entries
                .iter()
                .map(|entry| entry.key.as_str())
                .collect::<Vec<_>>()
                != ["constant", "kind", "num_qubits", "terms"]
        {
            return Err(ObservableError::Malformed("invalid field set"));
        }
        if literal_string(entry(entries, "kind")?) != Some("qamomile.pauli_sum.v1") {
            return Err(ObservableError::Malformed("unknown kind"));
        }
        let mut num_qubits = literal_u64(entry(entries, "num_qubits")?)
            .ok_or(ObservableError::Malformed("invalid num_qubits"))?;
        let mut constant = literal_complex(entry(entries, "constant")?)
            .ok_or(ObservableError::Malformed("invalid constant"))?;
        let raw_terms = tuple_values(entry(entries, "terms")?)
            .ok_or(ObservableError::Malformed("invalid terms"))?;
        let mut terms = BTreeMap::new();
        for raw_term in raw_terms {
            let [coefficient, factors] = tuple_values(raw_term)
                .and_then(|values| <&[wire::LiteralProto; 2]>::try_from(values).ok())
                .ok_or(ObservableError::Malformed("invalid term"))?;
            let coefficient = literal_complex(coefficient)
                .ok_or(ObservableError::Malformed("invalid coefficient"))?;
            let raw_factors =
                tuple_values(factors).ok_or(ObservableError::Malformed("invalid Pauli word"))?;
            let mut word = Vec::with_capacity(raw_factors.len());
            let mut previous = None;
            for raw_factor in raw_factors {
                let [pauli, index] = tuple_values(raw_factor)
                    .and_then(|values| <&[wire::LiteralProto; 2]>::try_from(values).ok())
                    .ok_or(ObservableError::Malformed("invalid Pauli factor"))?;
                let pauli = match literal_string(pauli) {
                    Some("X") => Pauli::X,
                    Some("Y") => Pauli::Y,
                    Some("Z") => Pauli::Z,
                    _ => return Err(ObservableError::Malformed("invalid Pauli name")),
                };
                let index =
                    literal_u64(index).ok_or(ObservableError::Malformed("invalid Pauli index"))?;
                if previous.is_some_and(|value| value >= index) {
                    return Err(ObservableError::Malformed(
                        "Pauli factors must have unique ordered indices",
                    ));
                }
                previous = Some(index);
                num_qubits =
                    num_qubits.max(index.checked_add(1).ok_or(ObservableError::WidthOverflow)?);
                word.push((index, pauli));
            }
            if word.is_empty() {
                constant += coefficient;
            } else {
                merge_term(&mut terms, word, coefficient);
            }
        }
        terms.retain(|_, coefficient| !is_zero(*coefficient));
        for coefficient in terms.values_mut() {
            *coefficient = canonical_complex(*coefficient);
        }
        if !constant.re.is_finite()
            || !constant.im.is_finite()
            || terms
                .values()
                .any(|coefficient| !coefficient.re.is_finite() || !coefficient.im.is_finite())
        {
            return Err(ObservableError::Malformed(
                "observable coefficients must be finite",
            ));
        }
        Ok(Self {
            terms,
            constant: canonical_complex(constant),
            num_qubits,
        })
    }

    fn normalized(mut self) -> NormalizedObservable {
        self.terms.retain(|_, coefficient| !is_zero(*coefficient));
        for coefficient in self.terms.values_mut() {
            *coefficient = canonical_complex(*coefficient);
        }
        self.constant = canonical_complex(self.constant);
        let hermitian =
            self.constant.im == 0.0 && self.terms.values().all(|coefficient| coefficient.im == 0.0);
        NormalizedObservable {
            literal: self.encode().encode_to_vec(),
            hermitian,
        }
    }

    fn checked_normalized(self) -> Result<NormalizedObservable, ObservableError> {
        if !finite_complex(self.constant)
            || self
                .terms
                .values()
                .any(|coefficient| !finite_complex(*coefficient))
        {
            return Err(ObservableError::CoefficientRange);
        }
        Ok(self.normalized())
    }

    pub(crate) fn encode(&self) -> wire::LiteralProto {
        mapping_literal([
            ("constant", complex_literal(self.constant)),
            ("kind", string_literal("qamomile.pauli_sum.v1")),
            ("num_qubits", integer_literal(self.num_qubits)),
            (
                "terms",
                tuple_literal(self.terms.iter().map(|(word, coefficient)| {
                    tuple_literal([
                        complex_literal(*coefficient),
                        tuple_literal(word.iter().map(|(index, pauli)| {
                            tuple_literal([
                                string_literal(match pauli {
                                    Pauli::X => "X",
                                    Pauli::Y => "Y",
                                    Pauli::Z => "Z",
                                }),
                                integer_literal(*index),
                            ])
                        })),
                    ])
                })),
            ),
        ])
    }
}

fn merge_term(terms: &mut BTreeMap<PauliWord, Complex64>, word: PauliWord, coefficient: Complex64) {
    let value = terms.entry(word).or_default();
    *value += coefficient;
    if is_zero(*value) {
        *value = Complex64::new(0.0, 0.0);
    }
}

fn multiply_words(left: &PauliWord, right: &PauliWord) -> (PauliWord, Complex64) {
    let mut factors = left.iter().copied().collect::<BTreeMap<_, _>>();
    let mut phase = Complex64::new(1.0, 0.0);
    for &(index, right_pauli) in right {
        let Some(left_pauli) = factors.get(&index).copied() else {
            factors.insert(index, right_pauli);
            continue;
        };
        if left_pauli == right_pauli {
            factors.remove(&index);
            continue;
        }
        let (result, contribution) = match (left_pauli, right_pauli) {
            (Pauli::X, Pauli::Y) => (Pauli::Z, Complex64::new(0.0, 1.0)),
            (Pauli::Y, Pauli::X) => (Pauli::Z, Complex64::new(0.0, -1.0)),
            (Pauli::Y, Pauli::Z) => (Pauli::X, Complex64::new(0.0, 1.0)),
            (Pauli::Z, Pauli::Y) => (Pauli::X, Complex64::new(0.0, -1.0)),
            (Pauli::Z, Pauli::X) => (Pauli::Y, Complex64::new(0.0, 1.0)),
            (Pauli::X, Pauli::Z) => (Pauli::Y, Complex64::new(0.0, -1.0)),
            _ => unreachable!("equal Pauli factors were removed"),
        };
        factors.insert(index, result);
        phase *= contribution;
    }
    (factors.into_iter().collect(), phase)
}

pub(crate) fn words_commute(left: &PauliWord, right: &PauliWord) -> bool {
    let mut left_index = 0;
    let mut right_index = 0;
    let mut anticommutes = false;
    while left_index < left.len() && right_index < right.len() {
        match left[left_index].0.cmp(&right[right_index].0) {
            std::cmp::Ordering::Less => left_index += 1,
            std::cmp::Ordering::Greater => right_index += 1,
            std::cmp::Ordering::Equal => {
                anticommutes ^= left[left_index].1 != right[right_index].1;
                left_index += 1;
                right_index += 1;
            }
        }
    }
    !anticommutes
}

/// Return whether two canonical Pauli words commute.
#[must_use]
pub fn pauli_words_commute(left: &[(u64, Pauli)], right: &[(u64, Pauli)]) -> bool {
    words_commute(&left.to_vec(), &right.to_vec())
}

fn canonical_complex(value: Complex64) -> Complex64 {
    Complex64::new(
        if value.re == 0.0 { 0.0 } else { value.re },
        if value.im == 0.0 { 0.0 } else { value.im },
    )
}

fn finite_complex(value: Complex64) -> bool {
    value.re.is_finite() && value.im.is_finite()
}

fn is_zero(value: Complex64) -> bool {
    value.re == 0.0 && value.im == 0.0
}

fn entry<'a>(
    entries: &'a [wire::literal_proto::Entry],
    name: &str,
) -> Result<&'a wire::LiteralProto, ObservableError> {
    entries
        .iter()
        .find(|entry| entry.key == name)
        .and_then(|entry| entry.value.as_ref())
        .ok_or(ObservableError::Malformed("missing field"))
}

fn mapping_entries(literal: &wire::LiteralProto) -> Option<&[wire::literal_proto::Entry]> {
    let wire::literal_proto::Value::MappingValue(mapping) = literal.value.as_ref()? else {
        return None;
    };
    Some(&mapping.entries)
}

fn tuple_values(literal: &wire::LiteralProto) -> Option<&[wire::LiteralProto]> {
    let wire::literal_proto::Value::TupleValue(sequence) = literal.value.as_ref()? else {
        return None;
    };
    Some(&sequence.values)
}

fn literal_string(literal: &wire::LiteralProto) -> Option<&str> {
    let wire::literal_proto::Value::StringValue(value) = literal.value.as_ref()? else {
        return None;
    };
    Some(value)
}

fn literal_complex(literal: &wire::LiteralProto) -> Option<Complex64> {
    use wire::literal_proto::Value;

    match literal.value.as_ref()? {
        Value::IntValue(value) => value.to_f64().map(|value| Complex64::new(value, 0.0)),
        Value::BigintValue(value) => BigInt::from_signed_bytes_be(value)
            .to_f64()
            .map(|value| Complex64::new(value, 0.0)),
        Value::RealValue(value) => Some(Complex64::new(*value, 0.0)),
        Value::ComplexValue(value) => Some(Complex64::new(value.real, value.imaginary)),
        _ => None,
    }
}

fn literal_u64(literal: &wire::LiteralProto) -> Option<u64> {
    use wire::literal_proto::Value;

    match literal.value.as_ref()? {
        Value::IntValue(value) => u64::try_from(*value).ok(),
        Value::BigintValue(value) => BigInt::from_signed_bytes_be(value).to_u64(),
        _ => None,
    }
}

fn mapping_literal<const N: usize>(entries: [(&str, wire::LiteralProto); N]) -> wire::LiteralProto {
    wire::LiteralProto {
        value: Some(wire::literal_proto::Value::MappingValue(
            wire::literal_proto::Mapping {
                entries: entries
                    .into_iter()
                    .map(|(key, value)| wire::literal_proto::Entry {
                        key: key.to_owned(),
                        value: Some(value),
                    })
                    .collect(),
            },
        )),
    }
}

fn tuple_literal(values: impl IntoIterator<Item = wire::LiteralProto>) -> wire::LiteralProto {
    wire::LiteralProto {
        value: Some(wire::literal_proto::Value::TupleValue(
            wire::literal_proto::Sequence {
                values: values.into_iter().collect(),
            },
        )),
    }
}

fn string_literal(value: &str) -> wire::LiteralProto {
    wire::LiteralProto {
        value: Some(wire::literal_proto::Value::StringValue(value.to_owned())),
    }
}

fn complex_literal(value: Complex64) -> wire::LiteralProto {
    wire::LiteralProto {
        value: Some(wire::literal_proto::Value::ComplexValue(
            wire::literal_proto::Complex {
                real: value.re,
                imaginary: value.im,
            },
        )),
    }
}

fn integer_literal(value: u64) -> wire::LiteralProto {
    let value = if let Ok(value) = i64::try_from(value) {
        wire::literal_proto::Value::IntValue(value)
    } else {
        wire::literal_proto::Value::BigintValue(BigInt::from(value).to_signed_bytes_be())
    };
    wire::LiteralProto { value: Some(value) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observable(
        terms: impl IntoIterator<Item = (Complex64, PauliWord)>,
        constant: Complex64,
        width: u64,
    ) -> Vec<u8> {
        mapping_literal([
            ("constant", complex_literal(constant)),
            ("kind", string_literal("qamomile.pauli_sum.v1")),
            ("num_qubits", integer_literal(width)),
            (
                "terms",
                tuple_literal(terms.into_iter().map(|(coefficient, factors)| {
                    tuple_literal([
                        complex_literal(coefficient),
                        tuple_literal(factors.into_iter().map(|(index, pauli)| {
                            tuple_literal([
                                string_literal(match pauli {
                                    Pauli::X => "X",
                                    Pauli::Y => "Y",
                                    Pauli::Z => "Z",
                                }),
                                integer_literal(index),
                            ])
                        })),
                    ])
                })),
            ),
        ])
        .encode_to_vec()
    }

    #[test]
    fn normalizes_duplicate_terms_identity_and_signed_zero() {
        let x = vec![(0, Pauli::X)];
        let normalized = normalize_observable(&observable(
            [
                (Complex64::new(2.0, -0.0), x.clone()),
                (Complex64::new(-1.0, 0.0), x),
                (Complex64::new(3.0, 0.0), vec![]),
            ],
            Complex64::new(-3.0, -0.0),
            0,
        ))
        .expect("normalize observable");
        let decoded = Observable::decode(normalized.literal()).expect("decode normalized");

        assert_eq!(decoded.constant.re.to_bits(), 0.0_f64.to_bits());
        assert_eq!(decoded.constant.im.to_bits(), 0.0_f64.to_bits());
        assert_eq!(
            decoded.terms[&vec![(0, Pauli::X)]],
            Complex64::new(1.0, 0.0)
        );
        assert_eq!(decoded.num_qubits, 1);
        assert!(normalized.is_hermitian());
    }

    #[test]
    fn observable_one_norm_uses_the_canonical_exact_sum() {
        let payload = observable(
            [(Complex64::new(3.0, 4.0), vec![(0, Pauli::X)])],
            Complex64::new(-2.0, 0.0),
            1,
        );

        assert_eq!(
            observable_lcu_one_norm(&payload)
                .expect("observable one-norm")
                .to_bits(),
            7.0_f64.to_bits()
        );
        assert_eq!(
            observable_lcu_one_norm(&observable(
                [(Complex64::new(f64::MAX, 0.0), vec![(0, Pauli::Z)])],
                Complex64::new(f64::MAX, 0.0),
                1,
            )),
            Err(ObservableError::OneNormRange)
        );
    }

    #[test]
    fn observable_normalization_rejects_nonfinite_coefficients() {
        assert!(matches!(
            normalize_observable(&observable(
                [(Complex64::new(f64::INFINITY, 0.0), vec![(0, Pauli::X)])],
                Complex64::new(0.0, 0.0),
                1,
            )),
            Err(ObservableError::Malformed(
                "observable coefficients must be finite"
            ))
        ));
    }

    #[test]
    fn native_real_projection_retains_exact_cancellation_residue() {
        let residue = f64::from_bits(1);
        let word = vec![(0_usize, Pauli::Z)];
        let normalized = normalize_real_observable(
            1,
            0.0,
            &[
                (word.clone(), f64::MAX),
                (word.clone(), -f64::MAX),
                (word, residue),
            ],
        )
        .expect("real observable");
        let decoded = Observable::decode(normalized.literal()).expect("decode normalized");

        assert_eq!(
            decoded.terms[&vec![(0, Pauli::Z)]].re.to_bits(),
            residue.to_bits()
        );
    }

    #[test]
    fn multiplies_pauli_words_with_ordered_complex_phase() {
        let x = observable(
            [(Complex64::new(1.0, 0.0), vec![(0, Pauli::X)])],
            Complex64::new(0.0, 0.0),
            1,
        );
        let y = observable(
            [(Complex64::new(1.0, 0.0), vec![(0, Pauli::Y)])],
            Complex64::new(0.0, 0.0),
            1,
        );
        let product = multiply_observables(&x, &y).expect("multiply observables");
        let decoded = Observable::decode(product.literal()).expect("decode product");

        assert_eq!(
            decoded.terms[&vec![(0, Pauli::Z)]],
            Complex64::new(0.0, 1.0)
        );
        assert!(!product.is_hermitian());
    }

    #[test]
    fn closes_affine_observable_algebra_in_native_normal_form() {
        let x = observable(
            [(Complex64::new(2.0, 0.0), vec![(0, Pauli::X)])],
            Complex64::new(3.0, 0.0),
            1,
        );
        let z = observable(
            [(Complex64::new(1.0, 0.0), vec![(2, Pauli::Z)])],
            Complex64::new(-1.0, 0.0),
            3,
        );
        let affine = affine_observable(&x, Complex64::new(0.0, 1.0), Complex64::new(1.0, -3.0))
            .expect("affine observable");
        let combined =
            add_scaled_observable(&x, &z, Complex64::new(-2.0, 0.0)).expect("linear combination");
        let affine = Observable::decode(affine.literal()).expect("decode affine observable");
        let combined = Observable::decode(combined.literal()).expect("decode linear combination");

        assert_eq!(affine.terms[&vec![(0, Pauli::X)]], Complex64::new(0.0, 2.0));
        assert_eq!(affine.constant, Complex64::new(1.0, 0.0));
        assert_eq!(affine.num_qubits, 1);
        assert_eq!(
            combined.terms[&vec![(0, Pauli::X)]],
            Complex64::new(2.0, 0.0)
        );
        assert_eq!(
            combined.terms[&vec![(2, Pauli::Z)]],
            Complex64::new(-2.0, 0.0)
        );
        assert_eq!(combined.constant, Complex64::new(5.0, 0.0));
        assert_eq!(combined.num_qubits, 3);
    }

    #[test]
    fn observable_algebra_rejects_nonfinite_results() {
        let maximum = observable(
            [(Complex64::new(f64::MAX, 0.0), vec![(0, Pauli::X)])],
            Complex64::new(0.0, 0.0),
            1,
        );
        let two = observable([], Complex64::new(2.0, 0.0), 1);

        assert_eq!(
            affine_observable(&maximum, Complex64::new(2.0, 0.0), Complex64::new(0.0, 0.0),),
            Err(ObservableError::CoefficientRange)
        );
        assert_eq!(
            multiply_observables(&maximum, &two),
            Err(ObservableError::CoefficientRange)
        );
    }

    #[test]
    fn remaps_logical_indices_without_collapsing_pauli_axes() {
        let source = observable(
            [
                (Complex64::new(1.0, 0.0), vec![(0, Pauli::X)]),
                (Complex64::new(2.0, 0.0), vec![(1, Pauli::X)]),
            ],
            Complex64::new(3.0, 0.0),
            4,
        );
        let remapped = remap_observable(&source, &[(0, 1), (1, 6)]).expect("remap observable");
        let remapped = Observable::decode(remapped.literal()).expect("decode remapped observable");

        assert_eq!(
            remapped.terms[&vec![(1, Pauli::X)]],
            Complex64::new(1.0, 0.0)
        );
        assert_eq!(
            remapped.terms[&vec![(6, Pauli::X)]],
            Complex64::new(2.0, 0.0)
        );
        assert_eq!(remapped.constant, Complex64::new(3.0, 0.0));
        assert_eq!(remapped.num_qubits, 7);

        let product = observable(
            [(Complex64::new(1.0, 0.0), vec![(0, Pauli::X), (1, Pauli::Z)])],
            Complex64::new(0.0, 0.0),
            2,
        );
        assert_eq!(
            remap_observable(&product, &[(1, 0)]),
            Err(ObservableError::InvalidRemap(
                "two factors of one word map to the same index"
            ))
        );
        assert_eq!(
            remap_observable(&product, &[(2, 3)]),
            Err(ObservableError::InvalidRemap(
                "source index is outside the observable width"
            ))
        );
    }

    #[test]
    fn closes_the_complete_single_lane_pauli_multiplication_table() {
        let cases = [
            (Pauli::X, Pauli::X, None, Complex64::new(1.0, 0.0)),
            (Pauli::Y, Pauli::Y, None, Complex64::new(1.0, 0.0)),
            (Pauli::Z, Pauli::Z, None, Complex64::new(1.0, 0.0)),
            (Pauli::X, Pauli::Y, Some(Pauli::Z), Complex64::new(0.0, 1.0)),
            (
                Pauli::Y,
                Pauli::X,
                Some(Pauli::Z),
                Complex64::new(0.0, -1.0),
            ),
            (Pauli::Y, Pauli::Z, Some(Pauli::X), Complex64::new(0.0, 1.0)),
            (
                Pauli::Z,
                Pauli::Y,
                Some(Pauli::X),
                Complex64::new(0.0, -1.0),
            ),
            (Pauli::Z, Pauli::X, Some(Pauli::Y), Complex64::new(0.0, 1.0)),
            (
                Pauli::X,
                Pauli::Z,
                Some(Pauli::Y),
                Complex64::new(0.0, -1.0),
            ),
        ];
        for (left, right, expected_pauli, expected_coefficient) in cases {
            let left = observable(
                [(Complex64::new(1.0, 0.0), vec![(0, left)])],
                Complex64::new(0.0, 0.0),
                1,
            );
            let right = observable(
                [(Complex64::new(1.0, 0.0), vec![(0, right)])],
                Complex64::new(0.0, 0.0),
                1,
            );
            let product = multiply_observables(&left, &right).expect("Pauli product");
            let product = Observable::decode(product.literal()).expect("decode product");
            if let Some(pauli) = expected_pauli {
                assert_eq!(product.terms[&vec![(0, pauli)]], expected_coefficient);
                assert_eq!(product.constant, Complex64::new(0.0, 0.0));
            } else {
                assert!(product.terms.is_empty());
                assert_eq!(product.constant, expected_coefficient);
            }
        }
    }

    #[test]
    fn distributes_identity_and_nonidentity_terms_without_zero_residue() {
        let left = observable(
            [(Complex64::new(1.0, 0.0), vec![(0, Pauli::X)])],
            Complex64::new(2.0, 0.0),
            1,
        );
        let right = observable(
            [(Complex64::new(1.0, 0.0), vec![(0, Pauli::Y)])],
            Complex64::new(3.0, 0.0),
            1,
        );
        let product = multiply_observables(&left, &right).expect("distributed product");
        let product = Observable::decode(product.literal()).expect("decode product");

        assert_eq!(product.constant, Complex64::new(6.0, 0.0));
        assert_eq!(
            product.terms[&vec![(0, Pauli::X)]],
            Complex64::new(3.0, 0.0)
        );
        assert_eq!(
            product.terms[&vec![(0, Pauli::Y)]],
            Complex64::new(2.0, 0.0)
        );
        assert_eq!(
            product.terms[&vec![(0, Pauli::Z)]],
            Complex64::new(0.0, 1.0)
        );
    }

    #[test]
    fn derives_pairwise_commutation_from_canonical_words() {
        let commuting = normalize_observable(&observable(
            [
                (Complex64::new(1.0, 0.0), vec![(0, Pauli::X)]),
                (Complex64::new(1.0, 0.0), vec![(0, Pauli::X), (1, Pauli::Z)]),
            ],
            Complex64::new(0.0, 0.0),
            2,
        ))
        .expect("commuting observable");
        let noncommuting = normalize_observable(&observable(
            [
                (Complex64::new(1.0, 0.0), vec![(0, Pauli::X)]),
                (Complex64::new(1.0, 0.0), vec![(0, Pauli::Z)]),
            ],
            Complex64::new(0.0, 0.0),
            1,
        ))
        .expect("noncommuting observable");

        assert!(observable_is_pairwise_commuting(commuting.literal()).expect("commutation"));
        assert!(!observable_is_pairwise_commuting(noncommuting.literal()).expect("commutation"));
    }

    #[test]
    fn word_commutation_counts_anticommuting_lanes_modulo_two() {
        let xx = [(0, Pauli::X), (1, Pauli::X)];
        let zz = [(0, Pauli::Z), (1, Pauli::Z)];

        assert!(pauli_words_commute(&xx, &zz));
        assert!(!pauli_words_commute(&xx[..1], &zz[..1]));
    }

    #[test]
    fn rejects_noncanonical_pauli_factor_order() {
        let malformed = observable(
            [(Complex64::new(1.0, 0.0), vec![(1, Pauli::X), (0, Pauli::Z)])],
            Complex64::new(0.0, 0.0),
            2,
        );

        assert_eq!(
            normalize_observable(&malformed),
            Err(ObservableError::Malformed(
                "Pauli factors must have unique ordered indices"
            ))
        );
    }

    #[test]
    fn decomposes_dense_matrix_with_little_endian_pauli_words() {
        let identity = [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];
        let x = [
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        let z = [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-1.0, 0.0),
        ];
        let mut matrix = Vec::with_capacity(16);
        for row in 0..4 {
            for column in 0..4 {
                matrix.push(
                    x[row / 2 * 2 + column / 2] * z[row % 2 * 2 + column % 2]
                        + identity[row / 2 * 2 + column / 2] * x[row % 2 * 2 + column % 2] * 0.5,
                );
            }
        }

        let normalized =
            decompose_hermitian_matrix(&[4, 4], &matrix, 0.0).expect("decompose Hermitian matrix");
        let decoded = Observable::decode(normalized.literal()).expect("decode observable");

        assert_eq!(decoded.num_qubits, 2);
        assert_eq!(decoded.constant, Complex64::new(0.0, 0.0));
        assert_eq!(
            decoded.terms[&vec![(0, Pauli::Z), (1, Pauli::X)]],
            Complex64::new(1.0, 0.0)
        );
        assert_eq!(
            decoded.terms[&vec![(0, Pauli::X)]],
            Complex64::new(0.5, 0.0)
        );
    }

    #[test]
    fn exact_fwht_retains_extreme_and_subnormal_identity_coefficients() {
        for coefficient in [1e308, f64::from_bits(1)] {
            let matrix = [
                Complex64::new(coefficient, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(coefficient, 0.0),
            ];
            let normalized =
                decompose_hermitian_matrix(&[2, 2], &matrix, 0.0).expect("decompose identity");
            let decoded = Observable::decode(normalized.literal()).expect("decode identity");

            assert_eq!(decoded.constant, Complex64::new(coefficient, 0.0));
            assert!(decoded.terms.is_empty());
        }
    }

    #[test]
    fn decomposes_a_general_complex_matrix_with_truncation_evidence() {
        let tiny = 1e-14;
        let matrix = [
            Complex64::new(tiny, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-tiny, 0.0),
        ];

        let decomposition =
            decompose_pauli_matrix(&[2, 2], &matrix, 1e-12).expect("decompose matrix");
        let decoded =
            Observable::decode(decomposition.observable().literal()).expect("decode matrix");

        assert_eq!(decoded.terms.len(), 2);
        assert_eq!(
            decoded.terms[&vec![(0, Pauli::X)]],
            Complex64::new(0.5, 0.0)
        );
        assert_eq!(
            decoded.terms[&vec![(0, Pauli::Y)]],
            Complex64::new(0.0, 0.5)
        );
        assert!(decomposition.truncation_error_bound() >= tiny);
        assert!(decomposition.truncation_error_bound() < tiny.next_up().next_up());
        assert!(!decomposition.observable().is_hermitian());
    }

    #[test]
    fn rejects_a_general_decomposition_with_unbounded_lcu_normalization() {
        let largest = f64::MAX;
        let matrix = [
            Complex64::new(largest, 0.0),
            Complex64::new(largest, 0.0),
            Complex64::new(largest, 0.0),
            Complex64::new(-largest, 0.0),
        ];

        assert_eq!(
            decompose_pauli_matrix(&[2, 2], &matrix, 0.0),
            Err(ObservableError::CoefficientRange)
        );
    }

    #[test]
    fn dense_hermitian_arithmetic_owns_shape_range_and_zero_semantics() {
        let x = [
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        let identity = [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];

        assert_eq!(
            add_scaled_hermitian_matrices(&[2, 2], &x, &[2, 2], &identity, -2.0)
                .expect("subtract scaled identity"),
            vec![
                Complex64::new(-2.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(-2.0, 0.0),
            ]
        );
        assert_eq!(
            scale_hermitian_matrix(&[2, 2], &x, 2.0, true).expect("divide matrix"),
            x.iter().map(|entry| *entry * 0.5).collect::<Vec<_>>()
        );
        assert_eq!(
            add_scaled_hermitian_matrices(&[2, 2], &x, &[1, 1], &identity[..1], 1.0),
            Err(ObservableError::MatrixShapeMismatch)
        );
        assert_eq!(
            scale_hermitian_matrix(&[2, 2], &x, 0.0, true),
            Err(ObservableError::MatrixDivisionByZero)
        );
        assert_eq!(
            scale_hermitian_matrix(&[2, 2], &x, f64::INFINITY, false),
            Err(ObservableError::NonFiniteMatrix)
        );
        assert_eq!(
            scale_hermitian_matrix(
                &[2, 2],
                &[Complex64::new(0.0, 0.0); 4],
                f64::MIN_POSITIVE * f64::EPSILON,
                true,
            )
            .expect("divide zero matrix by the smallest positive scalar"),
            vec![Complex64::new(0.0, 0.0); 4]
        );
        assert_eq!(
            scale_hermitian_matrix(
                &[2, 2],
                &[
                    Complex64::new(1e308, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1e308, 0.0),
                ],
                1e-308,
                true,
            ),
            Err(ObservableError::NonFiniteMatrix)
        );
    }

    #[test]
    fn dense_matrix_validation_is_strict_and_bounded() {
        let non_hermitian = [
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];

        assert_eq!(
            validate_hermitian_matrix(&[2, 2], &non_hermitian, true, 1e-10),
            Err(ObservableError::NonHermitianMatrix)
        );
        assert_eq!(
            validate_hermitian_matrix(&[2, 2], &non_hermitian, false, 1e-10),
            Ok(1)
        );
        assert_eq!(
            decompose_hermitian_matrix(&[2, 2], &non_hermitian, 0.0),
            Err(ObservableError::ImaginaryResidue)
        );
        assert_eq!(
            validate_hermitian_matrix(&[3, 3], &[], false, 0.0),
            Err(ObservableError::InvalidMatrix(
                "dimension must be a power of two"
            ))
        );
        assert_eq!(
            validate_hermitian_matrix(&[512, 512], &[], false, 0.0),
            Err(ObservableError::MatrixBudget)
        );
        assert_eq!(
            validate_hermitian_matrix(&[1, 1], &[Complex64::new(f64::NAN, 0.0)], false, 0.0),
            Err(ObservableError::NonFiniteMatrix)
        );
        assert_eq!(
            validate_hermitian_matrix(&[1, 1], &[Complex64::new(1.0, 0.0)], false, -1.0),
            Err(ObservableError::NegativeTolerance)
        );
    }

    #[test]
    fn projects_mixed_pauli_product_states_with_overlap() {
        let z = observable(
            [(Complex64::new(1.0, 0.0), vec![(0, Pauli::Z)])],
            Complex64::new(0.0, 0.0),
            1,
        );
        let projected = project_pauli_subspace(&z, &[vec![0], vec![0]], &[vec![2], vec![0]])
            .expect("project mixed basis");
        let inverse_sqrt_two = std::f64::consts::FRAC_1_SQRT_2;

        assert_eq!(projected.size(), 2);
        let expected_overlap = [1.0, inverse_sqrt_two, inverse_sqrt_two, 1.0];
        assert!(
            projected
                .overlap()
                .iter()
                .zip(expected_overlap)
                .all(|(actual, expected)| (actual.re - expected).abs() < 1e-15 && actual.im == 0.0)
        );
        assert_eq!(
            projected.hamiltonian(),
            &[
                Complex64::new(1.0, 0.0),
                Complex64::new(inverse_sqrt_two, 0.0),
                Complex64::new(inverse_sqrt_two, 0.0),
                Complex64::new(0.0, 0.0),
            ]
        );
    }

    #[test]
    fn sampled_subspace_rejects_malformed_rows_and_excessive_cells() {
        let identity = observable([], Complex64::new(1.0, 0.0), 1);

        assert_eq!(
            project_pauli_subspace(&identity, &[vec![2]], &[vec![2]]),
            Err(ObservableError::InvalidSubspace(
                "samples must be fixed-width bit rows"
            ))
        );
        let samples = vec![vec![0]; 1_001];
        let bases = vec![vec![2]; 1_001];
        assert_eq!(
            project_pauli_subspace(&identity, &samples, &bases),
            Err(ObservableError::SubspaceBudget)
        );
    }
}
