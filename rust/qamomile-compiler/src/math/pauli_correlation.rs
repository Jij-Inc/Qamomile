//! Bounded Pauli-correlation encoding plans over canonical polynomials.

use num_bigint::BigUint;
use thiserror::Error;

use crate::{
    CanonicalPolynomial, MAX_POLYNOMIAL_VARIABLES, PolynomialDomain, PolynomialError,
    convert_polynomial,
};

/// Maximum supported Pauli correlator order.
pub const MAX_PCE_ORDER: usize = 64;
/// Maximum physical width searched for a PCE plan.
pub const MAX_PCE_QUBITS: usize = 1_000_000;
/// Maximum total factors materialized across PCE words.
pub const MAX_PCE_FACTORS: usize = 10_000_000;

/// One Pauli kind in a correlation word.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CorrelationPauli {
    /// Pauli X.
    X,
    /// Pauli Y.
    Y,
    /// Pauli Z.
    Z,
}

/// One deterministic PCE coordinate.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CorrelationWord {
    factors: Vec<(usize, CorrelationPauli)>,
}

impl CorrelationWord {
    /// Return the ordered physical-qubit factors.
    #[must_use]
    pub fn factors(&self) -> &[(usize, CorrelationPauli)] {
        &self.factors
    }
}

/// One checked PCE coordinate and decoding plan.
#[derive(Clone, Debug, PartialEq)]
pub struct PauliCorrelationPlan {
    source: CanonicalPolynomial,
    spin: CanonicalPolynomial,
    width: usize,
    words: Vec<CorrelationWord>,
}

impl PauliCorrelationPlan {
    /// Construct one bounded deterministic PCE plan.
    ///
    /// # Errors
    ///
    /// Returns [`PauliCorrelationError`] for nonquadratic models, invalid
    /// widths, or requests exceeding explicit combinatorial budgets.
    pub fn new(
        source: &CanonicalPolynomial,
        correlator_order: usize,
        width: Option<usize>,
    ) -> Result<Self, PauliCorrelationError> {
        let minimum = minimum_pce_qubits(source.variables().len(), correlator_order)?;
        if source.terms().iter().any(|term| term.variables().len() > 2) {
            return Err(PauliCorrelationError::NonQuadratic);
        }
        let width = width.unwrap_or(minimum);
        if width < minimum || width > MAX_PCE_QUBITS {
            return Err(PauliCorrelationError::InvalidWidth);
        }
        let factor_count = source
            .variables()
            .len()
            .checked_mul(correlator_order)
            .ok_or(PauliCorrelationError::CapacityBudget)?;
        if factor_count > MAX_PCE_FACTORS {
            return Err(PauliCorrelationError::CapacityBudget);
        }
        let spin = convert_polynomial(source, PolynomialDomain::Spin)?;
        let words = enumerate_words(width, correlator_order, source.variables().len());
        Ok(Self {
            source: source.clone(),
            spin,
            width,
            words,
        })
    }

    /// Return the physical register width.
    #[must_use]
    pub const fn width(&self) -> usize {
        self.width
    }

    /// Return one word per source variable.
    #[must_use]
    pub fn words(&self) -> &[CorrelationWord] {
        &self.words
    }

    /// Evaluate the relaxed spin objective on bounded expectations.
    ///
    /// # Errors
    ///
    /// Returns [`PauliCorrelationError`] for wrong, non-finite, or out-of-range
    /// coordinates and unrepresentable energy.
    pub fn relaxed_cost(&self, values: &[f64]) -> Result<f64, PauliCorrelationError> {
        if values.len() != self.words.len()
            || values
                .iter()
                .any(|value| !value.is_finite() || !(-1.0..=1.0).contains(value))
        {
            return Err(PauliCorrelationError::InvalidExpectations);
        }
        self.spin.energy_reals(values).map_err(Into::into)
    }

    /// Sign-round expectations into the source domain and compute its energy.
    ///
    /// # Errors
    ///
    /// Returns [`PauliCorrelationError`] under the same coordinate and energy
    /// contracts as [`Self::relaxed_cost`].
    pub fn decode(&self, values: &[f64]) -> Result<(Vec<i8>, f64), PauliCorrelationError> {
        self.relaxed_cost(values)?;
        let spins = values
            .iter()
            .map(|value| if *value >= 0.0 { 1_i8 } else { -1_i8 })
            .collect::<Vec<_>>();
        let decoded = match self.source.domain() {
            PolynomialDomain::Binary => spins.iter().map(|spin| (1 - spin) / 2).collect(),
            PolynomialDomain::Spin => spins,
        };
        let energy = self.source.energy(&decoded)?;
        Ok((decoded, energy))
    }
}

/// Failure to construct or evaluate a PCE plan.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum PauliCorrelationError {
    /// Correlator order or capacity search exceeds a bound.
    #[error("Q7571: PCE variable count or correlator order exceeds its capacity budget")]
    CapacityBudget,
    /// A requested physical width is too small or excessive.
    #[error("Q7572: PCE physical width is too small or excessive")]
    InvalidWidth,
    /// PCE currently accepts only quadratic source objectives.
    #[error("Q7573: Pauli correlation encoding requires a quadratic model")]
    NonQuadratic,
    /// An expectation row has the wrong shape or physical range.
    #[error("Q7574: PCE expectations must be finite values in [-1, 1]")]
    InvalidExpectations,
    /// The shared polynomial normal form rejected conversion or evaluation.
    #[error(transparent)]
    Polynomial(#[from] PolynomialError),
}

/// Return the smallest width containing enough distinct order-k Pauli words.
///
/// # Errors
///
/// Returns [`PauliCorrelationError`] when order or the resulting width exceeds
/// the explicit plan budgets.
pub fn minimum_pce_qubits(
    num_variables: usize,
    correlator_order: usize,
) -> Result<usize, PauliCorrelationError> {
    if num_variables > MAX_POLYNOMIAL_VARIABLES
        || correlator_order == 0
        || correlator_order > MAX_PCE_ORDER
    {
        return Err(PauliCorrelationError::CapacityBudget);
    }
    if num_variables == 0 {
        return Ok(correlator_order);
    }
    let mut low = correlator_order;
    let mut high = correlator_order.max(1);
    while !has_capacity(high, correlator_order, num_variables) {
        low = high + 1;
        high = high
            .checked_mul(2)
            .map(|value| value.min(MAX_PCE_QUBITS))
            .ok_or(PauliCorrelationError::CapacityBudget)?;
        if high < low
            || high == MAX_PCE_QUBITS && !has_capacity(high, correlator_order, num_variables)
        {
            return Err(PauliCorrelationError::CapacityBudget);
        }
    }
    while low < high {
        let middle = low + (high - low) / 2;
        if has_capacity(middle, correlator_order, num_variables) {
            high = middle;
        } else {
            low = middle + 1;
        }
    }
    Ok(low)
}

fn has_capacity(width: usize, order: usize, required: usize) -> bool {
    if width < order {
        return false;
    }
    let mut combinations = BigUint::from(1_u8);
    for index in 1..=order {
        combinations *= width - order + index;
        combinations /= index;
    }
    let order = u32::try_from(order).unwrap_or(u32::MAX);
    combinations * BigUint::from(3_u8).pow(order) >= BigUint::from(required)
}

fn enumerate_words(width: usize, order: usize, count: usize) -> Vec<CorrelationWord> {
    let mut words = Vec::with_capacity(count);
    if count == 0 {
        return words;
    }
    let mut support = (0..order).collect::<Vec<_>>();
    loop {
        let mut assignment = 0_usize;
        loop {
            let mut digits = assignment;
            let mut paulis = vec![CorrelationPauli::X; order];
            for position in (0..order).rev() {
                paulis[position] = match digits % 3 {
                    0 => CorrelationPauli::X,
                    1 => CorrelationPauli::Y,
                    _ => CorrelationPauli::Z,
                };
                digits /= 3;
            }
            if digits != 0 {
                break;
            }
            words.push(CorrelationWord {
                factors: support.iter().copied().zip(paulis).collect(),
            });
            if words.len() == count {
                return words;
            }
            assignment = assignment.saturating_add(1);
        }
        let Some(position) = (0..order)
            .rev()
            .find(|position| support[*position] < width - order + *position)
        else {
            return words;
        };
        support[position] += 1;
        for next in position + 1..order {
            support[next] = support[next - 1] + 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacity_and_word_order_match_the_pce_convention() {
        assert_eq!(minimum_pce_qubits(9, 2), Ok(2));
        assert_eq!(minimum_pce_qubits(10, 2), Ok(3));
        let source = crate::normalize_polynomial(
            PolynomialDomain::Spin,
            &[vec![10_u8.into()], vec![20_u8.into()], vec![30_u8.into()]],
            &[1.0, 1.0, 1.0],
            0.0,
            None,
        )
        .expect("source");
        let plan = PauliCorrelationPlan::new(&source, 1, None).expect("plan");
        assert_eq!(plan.width(), 1);
        assert_eq!(
            plan.words()
                .iter()
                .map(|word| word.factors()[0].1)
                .collect::<Vec<_>>(),
            vec![
                CorrelationPauli::X,
                CorrelationPauli::Y,
                CorrelationPauli::Z
            ]
        );
    }

    #[test]
    fn relaxed_cost_and_decode_share_the_source_polynomial() {
        let source = crate::normalize_polynomial(
            PolynomialDomain::Spin,
            &[
                vec![10_u8.into()],
                vec![10_u8.into(), 20_u8.into()],
                vec![20_u8.into()],
            ],
            &[1.0, 0.5, -2.0],
            0.25,
            None,
        )
        .expect("source");
        let plan = PauliCorrelationPlan::new(&source, 2, None).expect("plan");
        assert!((plan.relaxed_cost(&[0.4, -0.5]).expect("cost") - 1.55).abs() < 1e-12);
        assert_eq!(
            plan.decode(&[0.4, -0.5]).expect("decode"),
            (vec![1, -1], 2.75)
        );
    }

    #[test]
    fn topology_and_materialization_budgets_fail_before_unbounded_work() {
        let linear = crate::normalize_polynomial(
            PolynomialDomain::Spin,
            &[vec![0_u8.into()]],
            &[1.0],
            0.0,
            None,
        )
        .expect("linear source");
        let cubic = crate::normalize_polynomial(
            PolynomialDomain::Binary,
            &[vec![0_u8.into(), 1_u8.into(), 2_u8.into()]],
            &[1.0],
            0.0,
            None,
        )
        .expect("cubic source");

        assert_eq!(
            PauliCorrelationPlan::new(&cubic, 2, None),
            Err(PauliCorrelationError::NonQuadratic)
        );
        assert_eq!(
            PauliCorrelationPlan::new(&linear, MAX_PCE_ORDER + 1, None),
            Err(PauliCorrelationError::CapacityBudget)
        );
        assert_eq!(
            minimum_pce_qubits(MAX_POLYNOMIAL_VARIABLES + 1, 1),
            Err(PauliCorrelationError::CapacityBudget)
        );
    }
}
