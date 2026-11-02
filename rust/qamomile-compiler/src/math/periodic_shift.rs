//! Canonical finite sums of multidimensional periodic translations.

use std::collections::BTreeMap;

use num_bigint::{BigInt, BigUint, Sign};
use num_complex::Complex64;
use thiserror::Error;

use crate::binary64::exact_sum;

/// Maximum flattened quantum width accepted for sparse periodic offsets.
pub const MAX_PERIODIC_SHIFT_WIDTH: usize = 4096;
/// Maximum number of sparse coefficient rows accepted at one boundary.
pub const MAX_PERIODIC_SHIFT_TERMS: usize = 1_000_000;
/// Maximum number of dense matrix cells inspected for periodic structure.
pub const MAX_PERIODIC_MATRIX_ENTRIES: usize = 256 * 256;

/// One nonzero coefficient at one canonical multidimensional offset.
#[derive(Clone, Debug, PartialEq)]
pub struct PeriodicShiftTerm {
    offset: Vec<BigUint>,
    coefficient: Complex64,
}

impl PeriodicShiftTerm {
    /// Return one non-negative modular residue per axis.
    #[must_use]
    pub fn offset(&self) -> &[BigUint] {
        &self.offset
    }

    /// Return the finite nonzero shift coefficient.
    #[must_use]
    pub const fn coefficient(&self) -> Complex64 {
        self.coefficient
    }
}

/// One canonical periodic-shift decomposition in lexicographic offset order.
#[derive(Clone, Debug, PartialEq)]
pub struct PeriodicShiftDecomposition {
    terms: Vec<PeriodicShiftTerm>,
}

impl PeriodicShiftDecomposition {
    /// Return the canonical nonzero term row.
    #[must_use]
    pub fn terms(&self) -> &[PeriodicShiftTerm] {
        &self.terms
    }
}

/// Failure to normalize or recognize a periodic-shift decomposition.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum PeriodicShiftError {
    /// The periodic register layout is empty or contains a zero-width axis.
    #[error("Q7531: periodic shift requires nonempty positive register sizes")]
    InvalidLayout,
    /// Sparse modular arithmetic would exceed the explicit width budget.
    #[error("Q7532: periodic shift exceeds the {MAX_PERIODIC_SHIFT_WIDTH}-qubit width budget")]
    WidthBudget,
    /// Sparse coefficient and offset rows disagree or exceed their budget.
    #[error("Q7533: malformed periodic shift coefficient rows")]
    InvalidTerms,
    /// An offset has the wrong arity for the periodic axes.
    #[error("Q7534: offset must contain one component per register axis")]
    InvalidOffset,
    /// A coefficient contains a non-finite component.
    #[error("Q7535: periodic shift coefficients must be finite")]
    NonFiniteCoefficient,
    /// Aliased finite coefficients have an unrepresentable sum.
    #[error("Q7536: periodic shift coefficient sum exceeds the finite binary64 range")]
    CoefficientRange,
    /// The LCU one-norm cannot be represented as binary64.
    #[error("Q7537: periodic shift normalization exceeds the finite binary64 range")]
    NormalizationRange,
    /// A dense matrix shape or entry count is invalid.
    #[error("Q7538: invalid periodic shift matrix shape")]
    InvalidMatrix,
    /// Dense recognition exceeds its explicit quadratic work budget.
    #[error("Q7539: periodic shift matrix exceeds the 256 x 256 recognition budget")]
    MatrixBudget,
    /// The dense matrix is not a constant-coefficient periodic translation sum.
    #[error("Q7540: matrix is not a constant-coefficient periodic-shift matrix")]
    NonPeriodicMatrix,
}

/// Canonicalize signed modular aliases and combine their finite coefficients.
///
/// Offset components are arbitrary-precision signed integers. Axis zero is the
/// least-significant flattened axis. Coefficient components are accumulated as
/// exact integers in units of the smallest binary64 subnormal and rounded once
/// per canonical term, making the result independent of mapping iteration
/// order.
///
/// # Errors
///
/// Returns [`PeriodicShiftError`] when layout, row, finite-value, width, sum,
/// or one-norm contracts are violated.
pub fn normalize_periodic_shift_terms(
    register_sizes: &[usize],
    offsets: &[Vec<BigInt>],
    coefficients: &[Complex64],
) -> Result<PeriodicShiftDecomposition, PeriodicShiftError> {
    validate_layout(register_sizes)?;
    if offsets.len() != coefficients.len() || offsets.len() > MAX_PERIODIC_SHIFT_TERMS {
        return Err(PeriodicShiftError::InvalidTerms);
    }
    let moduli = register_sizes
        .iter()
        .map(|width| BigInt::from(1_u8) << width)
        .collect::<Vec<_>>();
    let mut grouped = BTreeMap::<Vec<BigUint>, Vec<Complex64>>::new();
    for (offset, coefficient) in offsets.iter().zip(coefficients) {
        if offset.len() != register_sizes.len() {
            return Err(PeriodicShiftError::InvalidOffset);
        }
        if !coefficient.re.is_finite() || !coefficient.im.is_finite() {
            return Err(PeriodicShiftError::NonFiniteCoefficient);
        }
        let canonical = offset
            .iter()
            .zip(&moduli)
            .map(|(component, modulus)| modular_residue(component, modulus))
            .collect::<Vec<_>>();
        grouped.entry(canonical).or_default().push(*coefficient);
    }

    let mut terms = Vec::with_capacity(grouped.len());
    for (offset, values) in grouped {
        let real = exact_sum(values.iter().map(|value| value.re))
            .ok_or(PeriodicShiftError::CoefficientRange)?;
        let imaginary = exact_sum(values.iter().map(|value| value.im))
            .ok_or(PeriodicShiftError::CoefficientRange)?;
        let coefficient = Complex64::new(canonical_zero(real), canonical_zero(imaginary));
        if coefficient != Complex64::new(0.0, 0.0) {
            terms.push(PeriodicShiftTerm {
                offset,
                coefficient,
            });
        }
    }
    validate_normalization(&terms)?;
    Ok(PeriodicShiftDecomposition { terms })
}

/// Recognize an exact dense constant-coefficient periodic-shift matrix.
///
/// # Errors
///
/// Returns [`PeriodicShiftError`] when the matrix or layout is malformed,
/// exceeds the dense budget, contains non-finite entries, or violates the
/// exact multidimensional translation law.
pub fn decompose_periodic_shift_matrix(
    register_sizes: &[usize],
    shape: &[usize],
    entries: &[Complex64],
) -> Result<PeriodicShiftDecomposition, PeriodicShiftError> {
    let total_width = validate_layout(register_sizes)?;
    if total_width > 8 {
        return Err(PeriodicShiftError::MatrixBudget);
    }
    let dimension = 1_usize << total_width;
    if shape != [dimension, dimension] {
        return Err(PeriodicShiftError::InvalidMatrix);
    }
    let entry_count = dimension
        .checked_mul(dimension)
        .ok_or(PeriodicShiftError::MatrixBudget)?;
    if entry_count > MAX_PERIODIC_MATRIX_ENTRIES {
        return Err(PeriodicShiftError::MatrixBudget);
    }
    if entries.len() != entry_count {
        return Err(PeriodicShiftError::InvalidMatrix);
    }
    if entries
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(PeriodicShiftError::NonFiniteCoefficient);
    }

    let dimensions = register_sizes
        .iter()
        .map(|width| 1_usize << width)
        .collect::<Vec<_>>();
    let coordinates = (0..dimension)
        .map(|index| periodic_coordinates(index, &dimensions))
        .collect::<Vec<_>>();
    for (column, column_coordinates) in coordinates.iter().enumerate() {
        for (offset_index, offset) in coordinates.iter().enumerate() {
            let translated = offset
                .iter()
                .zip(column_coordinates)
                .zip(&dimensions)
                .map(|((left, right), axis)| (left + right) % axis)
                .collect::<Vec<_>>();
            let row = flatten_periodic_coordinates(&translated, &dimensions);
            if entries[row * dimension + column] != entries[offset_index * dimension] {
                return Err(PeriodicShiftError::NonPeriodicMatrix);
            }
        }
    }

    let offsets = coordinates
        .into_iter()
        .map(|offset| offset.into_iter().map(BigInt::from).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let coefficients = (0..dimension)
        .map(|row| entries[row * dimension])
        .collect::<Vec<_>>();
    normalize_periodic_shift_terms(register_sizes, &offsets, &coefficients)
}

fn validate_layout(register_sizes: &[usize]) -> Result<usize, PeriodicShiftError> {
    if register_sizes.is_empty() || register_sizes.contains(&0) {
        return Err(PeriodicShiftError::InvalidLayout);
    }
    let total_width = register_sizes
        .iter()
        .try_fold(0_usize, |total, width| total.checked_add(*width))
        .ok_or(PeriodicShiftError::WidthBudget)?;
    if total_width > MAX_PERIODIC_SHIFT_WIDTH {
        return Err(PeriodicShiftError::WidthBudget);
    }
    Ok(total_width)
}

fn modular_residue(component: &BigInt, modulus: &BigInt) -> BigUint {
    let mut residue = component % modulus;
    if residue.sign() == Sign::Minus {
        residue += modulus;
    }
    residue
        .to_biguint()
        .expect("a normalized modular residue is non-negative")
}

fn canonical_zero(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}

fn validate_normalization(terms: &[PeriodicShiftTerm]) -> Result<(), PeriodicShiftError> {
    let scale = terms
        .iter()
        .map(|term| term.coefficient.norm())
        .fold(0.0_f64, f64::max);
    if !scale.is_finite() {
        return Err(PeriodicShiftError::NormalizationRange);
    }
    if scale == 0.0 {
        return Ok(());
    }
    let scaled_sum = terms
        .iter()
        .map(|term| term.coefficient.norm() / scale)
        .sum::<f64>();
    let normalization = scale * scaled_sum;
    if !normalization.is_finite() {
        return Err(PeriodicShiftError::NormalizationRange);
    }
    Ok(())
}

fn periodic_coordinates(mut index: usize, dimensions: &[usize]) -> Vec<usize> {
    dimensions
        .iter()
        .map(|dimension| {
            let coordinate = index % dimension;
            index /= dimension;
            coordinate
        })
        .collect()
}

fn flatten_periodic_coordinates(coordinates: &[usize], dimensions: &[usize]) -> usize {
    coordinates
        .iter()
        .zip(dimensions)
        .fold((0, 1), |(result, stride), (coordinate, dimension)| {
            (result + coordinate * stride, stride * dimension)
        })
        .0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_aliases_use_exact_order_independent_sums() {
        let residue = f64::from_bits(1);
        let decomposition = normalize_periodic_shift_terms(
            &[2],
            &[
                vec![BigInt::from(-1)],
                vec![BigInt::from(3)],
                vec![BigInt::from(7)],
            ],
            &[
                Complex64::new(f64::MAX, 0.0),
                Complex64::new(-f64::MAX, 0.0),
                Complex64::new(residue, 0.0),
            ],
        )
        .expect("normalize terms");

        assert_eq!(decomposition.terms().len(), 1);
        assert_eq!(decomposition.terms()[0].offset(), &[BigUint::from(3_u8)]);
        assert_eq!(
            decomposition.terms()[0].coefficient().re.to_bits(),
            residue.to_bits()
        );
    }

    #[test]
    fn recognizes_multidimensional_periodic_matrix() {
        let mut entries = vec![Complex64::new(0.0, 0.0); 16];
        for column in 0..4 {
            entries[((column + 1) % 2 + 2 * (column / 2)) * 4 + column] = Complex64::new(1.0, 0.0);
        }
        let decomposition =
            decompose_periodic_shift_matrix(&[1, 1], &[4, 4], &entries).expect("recognize matrix");

        assert_eq!(decomposition.terms().len(), 1);
        assert_eq!(
            decomposition.terms()[0].offset(),
            &[BigUint::from(1_u8), BigUint::from(0_u8)]
        );
    }

    #[test]
    fn rejects_nonperiodic_and_unbounded_dense_matrices() {
        let nonperiodic = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(2.0, 0.0),
        ];
        assert_eq!(
            decompose_periodic_shift_matrix(&[1], &[2, 2], &nonperiodic),
            Err(PeriodicShiftError::NonPeriodicMatrix)
        );
        assert_eq!(
            decompose_periodic_shift_matrix(&[9], &[512, 512], &[]),
            Err(PeriodicShiftError::MatrixBudget)
        );
    }
}
