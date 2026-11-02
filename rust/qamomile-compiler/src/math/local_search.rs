//! Deterministic single-variable descent for canonical sparse polynomials.

use std::collections::BTreeMap;

use thiserror::Error;

use crate::binary64::exact_sum;
use crate::polynomial::{CanonicalPolynomial, PolynomialDomain, PolynomialTerm};

/// Maximum number of variables accepted by one local-search request.
pub const MAX_LOCAL_SEARCH_VARIABLES: usize = 1_000_000;
/// Maximum number of nonconstant polynomial terms accepted by one request.
pub const MAX_LOCAL_SEARCH_TERMS: usize = 1_000_000;
/// Maximum structural multiplication work in one complete variable sweep.
pub const MAX_LOCAL_SEARCH_SWEEP_WORK: usize = 10_000_000;
/// Maximum cumulative structural work accepted by one search.
pub const MAX_LOCAL_SEARCH_WORK: usize = 100_000_000;
/// Maximum number of accepted improving flips in one search.
pub const MAX_LOCAL_SEARCH_STEPS: usize = 1_000_000;

/// Deterministic improving-flip selection rule.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImprovementMethod {
    /// Select the first improving dense variable position.
    First,
    /// Select the variable with the most negative energy delta.
    Best,
}

/// One completed local-search row.
#[derive(Clone, Debug, PartialEq)]
pub struct LocalSearchResult {
    values: Vec<i8>,
    energy: f64,
    steps: usize,
}

impl LocalSearchResult {
    /// Return the final values in dense model-variable order.
    #[must_use]
    pub fn values(&self) -> &[i8] {
        &self.values
    }

    /// Return the exactly accumulated finite objective value.
    #[must_use]
    pub const fn energy(&self) -> f64 {
        self.energy
    }

    /// Return the number of accepted strictly improving flips.
    #[must_use]
    pub const fn steps(&self) -> usize {
        self.steps
    }
}

/// Minimize one already canonical polynomial by deterministic value flips.
///
/// # Errors
///
/// Returns [`LocalSearchError`] when the dense projection, assignment, energy,
/// or cumulative work exceeds the checked local-search contracts.
pub fn minimize_polynomial(
    polynomial: &CanonicalPolynomial,
    initial_values: &[i8],
    method: ImprovementMethod,
    max_steps: Option<usize>,
) -> Result<LocalSearchResult, LocalSearchError> {
    let dense = polynomial
        .variables()
        .iter()
        .enumerate()
        .map(|(position, variable)| (variable, position))
        .collect::<BTreeMap<_, _>>();
    let term_positions = polynomial
        .terms()
        .iter()
        .map(|term| {
            term.variables()
                .iter()
                .map(|variable| dense[variable])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let coefficients = polynomial
        .terms()
        .iter()
        .map(PolynomialTerm::coefficient)
        .collect::<Vec<_>>();
    minimize_sparse_polynomial(
        polynomial.domain(),
        polynomial.variables().len(),
        &term_positions,
        &coefficients,
        polynomial.constant(),
        initial_values,
        method,
        max_steps,
    )
}

/// Failure to validate or execute a bounded local-search request.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum LocalSearchError {
    /// The sparse polynomial is noncanonical or structurally excessive.
    #[error("Q7553: malformed or excessive local-search polynomial")]
    InvalidModel,
    /// The initial row disagrees with the selected variable domain.
    #[error("Q7554: local-search state disagrees with the model domain")]
    InvalidState,
    /// A coefficient or constant is not finite.
    #[error("Q7555: local-search coefficients must be finite")]
    NonFiniteCoefficient,
    /// The requested or realized search exceeds a deterministic work budget.
    #[error("Q7556: local search exceeds its deterministic work budget")]
    WorkBudget,
    /// Exact binary64 accumulation cannot represent an objective or delta.
    #[error("Q7557: local-search energy exceeds the finite binary64 range")]
    EnergyRange,
}

/// Minimize one canonical sparse polynomial by deterministic single-value flips.
///
/// Term positions must be nonempty, strictly increasing within each term, and
/// lexicographically increasing across terms. Coefficients must be finite and
/// nonzero. Every delta and the final energy are accumulated exactly in
/// binary64 subnormal units and rounded only once.
///
/// # Errors
///
/// Returns [`LocalSearchError`] for malformed models or states, non-finite
/// numerical input or results, and requests that exceed explicit work limits.
#[allow(clippy::too_many_arguments)]
pub fn minimize_sparse_polynomial(
    domain: PolynomialDomain,
    num_variables: usize,
    term_positions: &[Vec<usize>],
    coefficients: &[f64],
    constant: f64,
    initial_values: &[i8],
    method: ImprovementMethod,
    max_steps: Option<usize>,
) -> Result<LocalSearchResult, LocalSearchError> {
    validate_request(
        domain,
        num_variables,
        term_positions,
        coefficients,
        constant,
        initial_values,
        max_steps,
    )?;
    let affected = affected_terms(num_variables, term_positions);
    let mut values = initial_values.to_vec();
    let mut steps = 0_usize;
    let mut work = 0_usize;
    while max_steps.is_none_or(|limit| steps < limit) {
        let position = next_flip(
            domain,
            &values,
            term_positions,
            coefficients,
            &affected,
            method,
            &mut work,
        )?;
        let Some(position) = position else {
            break;
        };
        values[position] = match domain {
            PolynomialDomain::Binary => 1 - values[position],
            PolynomialDomain::Spin => -values[position],
        };
        steps = steps.checked_add(1).ok_or(LocalSearchError::WorkBudget)?;
        if steps > MAX_LOCAL_SEARCH_STEPS {
            return Err(LocalSearchError::WorkBudget);
        }
    }
    let energy = polynomial_energy(&values, term_positions, coefficients, constant)?;
    Ok(LocalSearchResult {
        values,
        energy: canonical_zero(energy),
        steps,
    })
}

#[allow(clippy::too_many_arguments)]
fn validate_request(
    domain: PolynomialDomain,
    num_variables: usize,
    term_positions: &[Vec<usize>],
    coefficients: &[f64],
    constant: f64,
    initial_values: &[i8],
    max_steps: Option<usize>,
) -> Result<(), LocalSearchError> {
    if num_variables > MAX_LOCAL_SEARCH_VARIABLES
        || term_positions.len() != coefficients.len()
        || term_positions.len() > MAX_LOCAL_SEARCH_TERMS
    {
        return Err(LocalSearchError::InvalidModel);
    }
    if !constant.is_finite() || coefficients.iter().any(|value| !value.is_finite()) {
        return Err(LocalSearchError::NonFiniteCoefficient);
    }
    if max_steps.is_some_and(|limit| limit > MAX_LOCAL_SEARCH_STEPS) {
        return Err(LocalSearchError::WorkBudget);
    }
    let mut previous: Option<&[usize]> = None;
    let mut sweep_work = num_variables;
    for (positions, coefficient) in term_positions.iter().zip(coefficients) {
        if positions.is_empty()
            || *coefficient == 0.0
            || positions.windows(2).any(|pair| pair[0] >= pair[1])
            || positions
                .last()
                .is_some_and(|position| *position >= num_variables)
            || previous.is_some_and(|row| row >= positions)
        {
            return Err(LocalSearchError::InvalidModel);
        }
        let term_work = positions
            .len()
            .checked_mul(positions.len())
            .ok_or(LocalSearchError::InvalidModel)?;
        sweep_work = sweep_work
            .checked_add(term_work)
            .ok_or(LocalSearchError::InvalidModel)?;
        previous = Some(positions);
    }
    if sweep_work > MAX_LOCAL_SEARCH_SWEEP_WORK {
        return Err(LocalSearchError::InvalidModel);
    }
    if initial_values.len() != num_variables
        || initial_values
            .iter()
            .any(|value| !valid_value(domain, *value))
    {
        return Err(LocalSearchError::InvalidState);
    }
    Ok(())
}

fn valid_value(domain: PolynomialDomain, value: i8) -> bool {
    match domain {
        PolynomialDomain::Binary => matches!(value, 0 | 1),
        PolynomialDomain::Spin => matches!(value, -1 | 1),
    }
}

fn affected_terms(num_variables: usize, terms: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut affected = vec![Vec::new(); num_variables];
    for (term_index, positions) in terms.iter().enumerate() {
        for position in positions {
            affected[*position].push(term_index);
        }
    }
    affected
}

#[allow(clippy::too_many_arguments)]
fn next_flip(
    domain: PolynomialDomain,
    values: &[i8],
    terms: &[Vec<usize>],
    coefficients: &[f64],
    affected: &[Vec<usize>],
    method: ImprovementMethod,
    work: &mut usize,
) -> Result<Option<usize>, LocalSearchError> {
    let mut best = None;
    let mut best_delta = 0.0_f64;
    for position in 0..values.len() {
        spend_work(work, 1)?;
        let mut contributions = Vec::with_capacity(affected[position].len());
        for term_index in &affected[position] {
            let term = &terms[*term_index];
            spend_work(work, term.len())?;
            let product = term
                .iter()
                .filter(|term_position| **term_position != position)
                .map(|term_position| f64::from(values[*term_position]))
                .product::<f64>();
            contributions.push(coefficients[*term_index] * product);
        }
        let affected_sum = exact_sum(contributions).ok_or(LocalSearchError::EnergyRange)?;
        let factor = match domain {
            PolynomialDomain::Binary => 1 - 2 * values[position],
            PolynomialDomain::Spin => -2 * values[position],
        };
        let delta = f64::from(factor) * affected_sum;
        if !delta.is_finite() {
            return Err(LocalSearchError::EnergyRange);
        }
        if delta >= 0.0 {
            continue;
        }
        if method == ImprovementMethod::First {
            return Ok(Some(position));
        }
        if best.is_none() || delta < best_delta {
            best = Some(position);
            best_delta = delta;
        }
    }
    Ok(best)
}

fn spend_work(work: &mut usize, amount: usize) -> Result<(), LocalSearchError> {
    *work = work
        .checked_add(amount)
        .ok_or(LocalSearchError::WorkBudget)?;
    if *work > MAX_LOCAL_SEARCH_WORK {
        return Err(LocalSearchError::WorkBudget);
    }
    Ok(())
}

fn polynomial_energy(
    values: &[i8],
    terms: &[Vec<usize>],
    coefficients: &[f64],
    constant: f64,
) -> Result<f64, LocalSearchError> {
    exact_sum(
        std::iter::once(constant).chain(terms.iter().zip(coefficients).map(
            |(positions, coefficient)| {
                coefficient
                    * positions
                        .iter()
                        .map(|position| f64::from(values[*position]))
                        .product::<f64>()
            },
        )),
    )
    .ok_or(LocalSearchError::EnergyRange)
}

fn canonical_zero(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_and_best_choose_deterministic_binary_flips() {
        let terms = vec![vec![0], vec![1]];
        let coefficients = vec![-1.0, -3.0];

        let first = minimize_sparse_polynomial(
            PolynomialDomain::Binary,
            2,
            &terms,
            &coefficients,
            0.0,
            &[0, 0],
            ImprovementMethod::First,
            Some(1),
        )
        .expect("first improvement");
        let best = minimize_sparse_polynomial(
            PolynomialDomain::Binary,
            2,
            &terms,
            &coefficients,
            0.0,
            &[0, 0],
            ImprovementMethod::Best,
            Some(1),
        )
        .expect("best improvement");

        assert_eq!(first.values(), &[1, 0]);
        assert_eq!(first.energy().to_bits(), (-1.0_f64).to_bits());
        assert_eq!(best.values(), &[0, 1]);
        assert_eq!(best.energy().to_bits(), (-3.0_f64).to_bits());
    }

    #[test]
    fn arbitrary_order_spin_search_reaches_a_one_flip_minimum() {
        let terms = vec![vec![0], vec![0, 1, 2], vec![1]];
        let coefficients = vec![0.5, -3.0, -1.0];
        let result = minimize_sparse_polynomial(
            PolynomialDomain::Spin,
            3,
            &terms,
            &coefficients,
            0.25,
            &[1, 1, 1],
            ImprovementMethod::Best,
            None,
        )
        .expect("spin search");

        for position in 0..3 {
            let mut neighbor = result.values().to_vec();
            neighbor[position] = -neighbor[position];
            let neighbor_energy =
                polynomial_energy(&neighbor, &terms, &coefficients, 0.25).expect("neighbor energy");
            assert!(neighbor_energy >= result.energy());
        }
    }

    #[test]
    fn exact_delta_retains_a_subnormal_cancellation_residue() {
        let residue = f64::from_bits(1);
        let result = minimize_sparse_polynomial(
            PolynomialDomain::Binary,
            3,
            &[vec![0], vec![0, 1], vec![0, 2]],
            &[f64::MAX, -f64::MAX, -residue],
            0.0,
            &[0, 1, 1],
            ImprovementMethod::First,
            Some(1),
        )
        .expect("exact delta");

        assert_eq!(result.values(), &[1, 1, 1]);
        assert_eq!(result.energy().to_bits(), (-residue).to_bits());
    }

    #[test]
    fn malformed_state_model_and_energy_fail_closed() {
        assert_eq!(
            minimize_sparse_polynomial(
                PolynomialDomain::Binary,
                1,
                &[vec![0, 0]],
                &[1.0],
                0.0,
                &[0],
                ImprovementMethod::Best,
                None,
            ),
            Err(LocalSearchError::InvalidModel)
        );
        assert_eq!(
            minimize_sparse_polynomial(
                PolynomialDomain::Binary,
                1,
                &[vec![0]],
                &[1.0],
                0.0,
                &[2],
                ImprovementMethod::Best,
                None,
            ),
            Err(LocalSearchError::InvalidState)
        );
        assert_eq!(
            minimize_sparse_polynomial(
                PolynomialDomain::Binary,
                1,
                &[vec![0]],
                &[f64::MAX],
                f64::MAX,
                &[1],
                ImprovementMethod::Best,
                Some(0),
            ),
            Err(LocalSearchError::EnergyRange)
        );
        assert_eq!(
            minimize_sparse_polynomial(
                PolynomialDomain::Binary,
                1,
                &[vec![0]],
                &[1.0],
                0.0,
                &[0],
                ImprovementMethod::Best,
                Some(MAX_LOCAL_SEARCH_STEPS + 1),
            ),
            Err(LocalSearchError::WorkBudget)
        );
    }

    #[test]
    fn canonical_polynomial_enters_the_same_search_engine() {
        let polynomial = crate::normalize_polynomial(
            PolynomialDomain::Binary,
            &[vec![4_u8.into()], vec![9_u8.into()]],
            &[-1.0, -3.0],
            0.0,
            None,
        )
        .expect("canonical polynomial");
        let result = minimize_polynomial(&polynomial, &[0, 0], ImprovementMethod::Best, Some(1))
            .expect("native model search");

        assert_eq!(result.values(), &[0, 1]);
        assert_eq!(result.energy().to_bits(), (-3.0_f64).to_bits());
    }
}
