//! Bounded host coordinates for product-formula composition.
//!
//! A schedule is transient library data. It is closed into ordinary `QCore`
//! loops and abstract Pauli evolution instead of becoming an algorithm opcode.

use thiserror::Error;

/// Maximum number of elementary term visits in one generated schedule.
pub const MAX_SUZUKI_SCHEDULE_ENTRIES: usize = 1_000_000;

/// Failure to construct one bounded Suzuki product-formula schedule.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum ProductFormulaError {
    /// A Hamiltonian splitting needs at least two ordered terms.
    #[error("Q7521: term_count must be at least two")]
    InvalidTermCount,
    /// Supported recursive formulas have order one or positive even order.
    #[error("Q7522: order must be one or a positive even integer")]
    InvalidOrder,
    /// Recursive expansion would exceed the explicit output budget.
    #[error("Q7523: Suzuki schedule exceeds the {MAX_SUZUKI_SCHEDULE_ENTRIES}-entry budget")]
    Budget,
    /// A coefficient cannot be represented as finite binary64.
    #[error("Q7524: Suzuki schedule coefficient exceeds the finite binary64 range")]
    CoefficientRange,
}

/// Construct an adjacent-normalized Suzuki product-formula schedule.
///
/// Each result row is `(term_index, time_scale)`. Order one emits the ordered
/// Lie product. Order two emits the symmetric formula. Every higher even order
/// applies Suzuki's five-block recursion, fusing equal terms at block
/// boundaries as it builds the next immutable row. Construction is iterative,
/// so invalid large orders fail by budget rather than recursion depth.
///
/// # Errors
///
/// Returns [`ProductFormulaError`] when the term count or order is invalid,
/// expansion exceeds the entry budget, or a coefficient is non-finite.
pub fn suzuki_product_formula_schedule(
    term_count: usize,
    order: usize,
) -> Result<Vec<(usize, f64)>, ProductFormulaError> {
    if term_count < 2 {
        return Err(ProductFormulaError::InvalidTermCount);
    }
    if order != 1 && (order < 2 || !order.is_multiple_of(2)) {
        return Err(ProductFormulaError::InvalidOrder);
    }
    if order == 1 {
        if term_count > MAX_SUZUKI_SCHEDULE_ENTRIES {
            return Err(ProductFormulaError::Budget);
        }
        return Ok((0..term_count).map(|term| (term, 1.0)).collect());
    }

    let second_order_length = term_count
        .checked_mul(2)
        .and_then(|value| value.checked_sub(1))
        .ok_or(ProductFormulaError::Budget)?;
    if second_order_length > MAX_SUZUKI_SCHEDULE_ENTRIES {
        return Err(ProductFormulaError::Budget);
    }
    let mut schedule = Vec::with_capacity(second_order_length);
    schedule.extend((0..term_count - 1).map(|term| (term, 0.5)));
    schedule.push((term_count - 1, 1.0));
    schedule.extend((0..term_count - 1).rev().map(|term| (term, 0.5)));

    for current_order in (4..=order).step_by(2) {
        let upper_bound = schedule
            .len()
            .checked_mul(5)
            .ok_or(ProductFormulaError::Budget)?;
        if upper_bound.saturating_sub(4) > MAX_SUZUKI_SCHEDULE_ENTRIES {
            return Err(ProductFormulaError::Budget);
        }
        let exponent_denominator =
            u32::try_from(current_order - 1).map_err(|_| ProductFormulaError::Budget)?;
        let coefficient = 1.0 / (4.0 - 4.0_f64.powf(1.0 / f64::from(exponent_denominator)));
        if !coefficient.is_finite() {
            return Err(ProductFormulaError::CoefficientRange);
        }
        let middle = 1.0 - 4.0 * coefficient;
        let mut next = Vec::with_capacity(upper_bound.saturating_sub(4));
        for scale in [coefficient, coefficient, middle, coefficient, coefficient] {
            append_scaled(&mut next, &schedule, scale)?;
        }
        schedule = next;
    }
    Ok(schedule)
}

fn append_scaled(
    output: &mut Vec<(usize, f64)>,
    source: &[(usize, f64)],
    multiplier: f64,
) -> Result<(), ProductFormulaError> {
    for &(term, coefficient) in source {
        let scaled = coefficient * multiplier;
        if !scaled.is_finite() {
            return Err(ProductFormulaError::CoefficientRange);
        }
        if let Some((last_term, last_coefficient)) = output.last_mut()
            && *last_term == term
        {
            *last_coefficient += scaled;
            if !last_coefficient.is_finite() {
                return Err(ProductFormulaError::CoefficientRange);
            }
        } else {
            output.push((term, scaled));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fourth_order_schedule_is_palindromic_and_normalized() {
        let schedule = suzuki_product_formula_schedule(2, 4).expect("derive schedule");

        assert_eq!(schedule.len(), 11);
        let suzuki_coefficient = 1.0 / (4.0 - 4.0_f64.powf(1.0 / 3.0));
        assert!((schedule[0].1 - 0.5 * suzuki_coefficient).abs() < 1e-15);
        assert!(schedule.iter().zip(schedule.iter().rev()).all(
            |(&(left_term, left_scale), &(right_term, right_scale))| {
                left_term == right_term && (left_scale - right_scale).abs() < 1e-15
            }
        ));
        for term in 0..2 {
            let total = schedule
                .iter()
                .filter(|(index, _)| *index == term)
                .map(|(_, scale)| scale)
                .sum::<f64>();
            assert!((total - 1.0).abs() < 1e-15);
        }
    }

    #[test]
    fn first_and_second_order_schedules_preserve_term_order() {
        assert_eq!(
            suzuki_product_formula_schedule(3, 1).expect("first order"),
            vec![(0, 1.0), (1, 1.0), (2, 1.0)]
        );
        assert_eq!(
            suzuki_product_formula_schedule(3, 2).expect("second order"),
            vec![(0, 0.5), (1, 0.5), (2, 1.0), (1, 0.5), (0, 0.5)]
        );
    }

    #[test]
    fn invalid_or_excessive_schedules_fail_before_expansion() {
        assert_eq!(
            suzuki_product_formula_schedule(1, 2),
            Err(ProductFormulaError::InvalidTermCount)
        );
        assert_eq!(
            suzuki_product_formula_schedule(2, 3),
            Err(ProductFormulaError::InvalidOrder)
        );
        assert_eq!(
            suzuki_product_formula_schedule(MAX_SUZUKI_SCHEDULE_ENTRIES, 2),
            Err(ProductFormulaError::Budget)
        );
    }
}
