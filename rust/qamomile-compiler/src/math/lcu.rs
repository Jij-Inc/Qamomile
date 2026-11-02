//! Bounded coefficient plans for linear combinations of unitaries.

use std::collections::BTreeMap;

use num_bigint::BigInt;
use num_complex::Complex64;
use thiserror::Error;

use crate::binary64::{
    SUBNORMAL_EXPONENT, exact_sum, float_to_subnormal_units, scaled_integer_to_f64,
};
use crate::state_preparation::MAX_AMPLITUDE_COUNT;

/// Maximum raw coefficient rows accepted by one LCU plan.
pub const MAX_LCU_INPUT_TERMS: usize = 1_000_000;
/// Maximum power-of-two SELECT cases admitted by the shared PREPARE path.
pub const MAX_LCU_CASE_COUNT: usize = MAX_AMPLITUDE_COUNT;

/// One nonzero canonical coefficient and its preparation coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct LcuCoefficientTerm {
    group: usize,
    coefficient: Complex64,
    amplitude: f64,
    phase: f64,
}

impl LcuCoefficientTerm {
    /// Return the first-occurrence callable group.
    #[must_use]
    pub const fn group(&self) -> usize {
        self.group
    }

    /// Return the exactly accumulated finite coefficient.
    #[must_use]
    pub const fn coefficient(&self) -> Complex64 {
        self.coefficient
    }

    /// Return the nonnegative normalized PREPARE amplitude.
    #[must_use]
    pub const fn amplitude(&self) -> f64 {
        self.amplitude
    }

    /// Return the finite coefficient phase closed into SELECT.
    #[must_use]
    pub const fn phase(&self) -> f64 {
        self.phase
    }
}

/// One immutable LCU coefficient, selector, and PREPARE plan.
#[derive(Clone, Debug, PartialEq)]
pub struct LcuCoefficientPlan {
    terms: Vec<LcuCoefficientTerm>,
    normalization: f64,
    signal_width: usize,
    case_count: usize,
}

impl LcuCoefficientPlan {
    /// Return canonical nonzero groups in first-occurrence order.
    #[must_use]
    pub fn terms(&self) -> &[LcuCoefficientTerm] {
        &self.terms
    }

    /// Return the finite coefficient one-norm, or one for the zero operator.
    #[must_use]
    pub const fn normalization(&self) -> f64 {
        self.normalization
    }

    /// Return the positive selector-register width.
    #[must_use]
    pub const fn signal_width(&self) -> usize {
        self.signal_width
    }

    /// Return the power-of-two SELECT case count.
    #[must_use]
    pub const fn case_count(&self) -> usize {
        self.case_count
    }
}

/// Failure to derive one finite LCU coefficient plan.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum LcuError {
    /// Group identities or coefficient rows are structurally invalid.
    #[error("Q7611: malformed LCU coefficient groups")]
    InvalidGroups,
    /// The raw input exceeds the explicit sparse row budget.
    #[error("Q7612: LCU input exceeds the {MAX_LCU_INPUT_TERMS}-term budget")]
    InputBudget,
    /// A coefficient contains a non-finite coordinate.
    #[error("Q7613: LCU coefficients must be finite")]
    NonFiniteCoefficient,
    /// Exact alias accumulation cannot be represented as binary64.
    #[error("Q7614: LCU coefficient sum exceeds the finite binary64 range")]
    CoefficientRange,
    /// The coefficient one-norm cannot be represented as binary64.
    #[error("Q7615: LCU normalization exceeds the finite binary64 range")]
    NormalizationRange,
    /// Selector padding would exceed the common amplitude-coordinate budget.
    #[error("Q7616: LCU selector exceeds the bounded preparation domain")]
    PreparationBudget,
    /// A normalized amplitude or phase is not representable.
    #[error("Q7617: LCU preparation coordinates exceed the finite binary64 range")]
    CoordinateRange,
}

#[derive(Default)]
struct ExactComplexSum {
    real: BigInt,
    imaginary: BigInt,
}

impl ExactComplexSum {
    fn add(&mut self, value: Complex64) {
        self.real += float_to_subnormal_units(value.re);
        self.imaginary += float_to_subnormal_units(value.im);
    }

    fn finish(&self) -> Result<Complex64, LcuError> {
        let real = scaled_integer_to_f64(&self.real, SUBNORMAL_EXPONENT)
            .ok_or(LcuError::CoefficientRange)?;
        let imaginary = scaled_integer_to_f64(&self.imaginary, SUBNORMAL_EXPONENT)
            .ok_or(LcuError::CoefficientRange)?;
        Ok(Complex64::new(
            canonical_zero(real),
            canonical_zero(imaginary),
        ))
    }
}

/// Derive canonical coefficients and bounded PREPARE/SELECT coordinates.
///
/// `group_count` names the contiguous first-occurrence callable identities
/// `0..group_count`. Multiple input rows may refer to one group. Their real and
/// imaginary coordinates are accumulated exactly in binary64 subnormal units,
/// making cancellation independent of Python iteration order.
///
/// # Errors
///
/// Returns [`LcuError`] for malformed groups, excessive input or selector
/// materialization, non-finite coefficients, an unrepresentable exact sum or
/// one-norm, or invalid preparation coordinates.
pub fn plan_lcu_coefficients(
    group_count: usize,
    rows: &[(usize, Complex64)],
) -> Result<LcuCoefficientPlan, LcuError> {
    if rows.len() > MAX_LCU_INPUT_TERMS || group_count > MAX_LCU_INPUT_TERMS {
        return Err(LcuError::InputBudget);
    }
    if (rows.is_empty() && group_count != 0) || (!rows.is_empty() && group_count == 0) {
        return Err(LcuError::InvalidGroups);
    }
    let mut grouped = BTreeMap::<usize, ExactComplexSum>::new();
    for (group, coefficient) in rows {
        if *group >= group_count {
            return Err(LcuError::InvalidGroups);
        }
        if !coefficient.re.is_finite() || !coefficient.im.is_finite() {
            return Err(LcuError::NonFiniteCoefficient);
        }
        grouped.entry(*group).or_default().add(*coefficient);
    }
    if grouped.len() != group_count {
        return Err(LcuError::InvalidGroups);
    }

    let mut coefficients = Vec::with_capacity(group_count);
    for (group, sum) in grouped {
        let coefficient = sum.finish()?;
        if coefficient != Complex64::new(0.0, 0.0) {
            coefficients.push((group, coefficient));
        }
    }
    if coefficients.is_empty() {
        return Ok(LcuCoefficientPlan {
            terms: Vec::new(),
            normalization: 1.0,
            signal_width: 1,
            case_count: 2,
        });
    }

    let normalization = exact_sum(coefficients.iter().map(|(_, value)| value.norm()))
        .ok_or(LcuError::NormalizationRange)?;
    if !normalization.is_finite() || normalization <= 0.0 {
        return Err(LcuError::NormalizationRange);
    }
    let signal_width = usize::max(1, coefficients.len().next_power_of_two().ilog2() as usize);
    let case_count = 1_usize
        .checked_shl(u32::try_from(signal_width).map_err(|_| LcuError::PreparationBudget)?)
        .ok_or(LcuError::PreparationBudget)?;
    if case_count > MAX_LCU_CASE_COUNT {
        return Err(LcuError::PreparationBudget);
    }
    let normalization_root = normalization.sqrt();
    let terms = coefficients
        .into_iter()
        .map(|(group, coefficient)| {
            let amplitude = coefficient.norm().sqrt() / normalization_root;
            let phase = coefficient.arg();
            if !amplitude.is_finite() || amplitude <= 0.0 || !phase.is_finite() {
                return Err(LcuError::CoordinateRange);
            }
            Ok(LcuCoefficientTerm {
                group,
                coefficient,
                amplitude,
                phase,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(LcuCoefficientPlan {
        terms,
        normalization,
        signal_width,
        case_count,
    })
}

fn canonical_zero(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_aliases_drive_one_coherent_plan() {
        let residue = f64::from_bits(1);
        let plan = plan_lcu_coefficients(
            2,
            &[
                (0, Complex64::new(f64::MAX, 0.0)),
                (1, Complex64::new(0.0, -1.0)),
                (0, Complex64::new(-f64::MAX, 0.0)),
                (0, Complex64::new(residue, 0.0)),
            ],
        )
        .expect("bounded plan");

        assert_eq!(plan.terms().len(), 2);
        assert_eq!(plan.terms()[0].group(), 0);
        assert_eq!(plan.terms()[0].coefficient(), Complex64::new(residue, 0.0));
        assert_eq!(plan.terms()[1].group(), 1);
        assert_eq!(plan.normalization().to_bits(), 1.0_f64.to_bits());
        assert_eq!(plan.signal_width(), 1);
        assert_eq!(plan.case_count(), 2);
        assert!((plan.terms()[1].phase() + std::f64::consts::FRAC_PI_2).abs() < f64::EPSILON);
    }

    #[test]
    fn cancellation_uses_the_explicit_zero_operator_plan() {
        let plan = plan_lcu_coefficients(
            1,
            &[
                (0, Complex64::new(2.0, -3.0)),
                (0, Complex64::new(-2.0, 3.0)),
            ],
        )
        .expect("zero plan");

        assert!(plan.terms().is_empty());
        assert_eq!(plan.normalization().to_bits(), 1.0_f64.to_bits());
        assert_eq!(plan.signal_width(), 1);
        assert_eq!(plan.case_count(), 2);
    }

    #[test]
    fn malformed_nonfinite_and_excessive_inputs_fail_closed() {
        assert_eq!(
            plan_lcu_coefficients(2, &[(0, Complex64::new(1.0, 0.0))]),
            Err(LcuError::InvalidGroups)
        );
        assert_eq!(
            plan_lcu_coefficients(1, &[(0, Complex64::new(f64::NAN, 0.0))]),
            Err(LcuError::NonFiniteCoefficient)
        );
        assert_eq!(
            plan_lcu_coefficients(1, &[(0, Complex64::new(f64::MAX, 0.0)); 2]),
            Err(LcuError::CoefficientRange)
        );
        assert_eq!(
            plan_lcu_coefficients(MAX_LCU_INPUT_TERMS + 1, &[]),
            Err(LcuError::InputBudget)
        );
    }
}
