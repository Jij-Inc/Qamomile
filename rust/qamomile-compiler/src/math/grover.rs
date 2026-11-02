//! Bounded native planning for Grover amplitude amplification.

use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{One, Zero};
use thiserror::Error;

/// Largest search-register width accepted by native Grover planning.
pub const MAX_GROVER_QUBITS: usize = 16_384;

const EXTRA_DECIMAL_DIGITS: usize = 20;
const CONVERGENCE_DECIMAL_DIGITS: usize = 30;

/// Failure to derive a bounded Grover iteration count.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum GroverPlanError {
    /// The search register has no basis states.
    #[error("Q7590: Grover search width must be positive")]
    EmptySearch,
    /// The search register exceeds the native planning budget.
    #[error("Q7591: Grover search width exceeds the {MAX_GROVER_QUBITS}-qubit budget")]
    SearchTooWide,
    /// The marked-state count is outside the search space.
    #[error("Q7592: Grover marked-state count must be positive and cannot exceed the search space")]
    InvalidMarkedCount,
    /// Guarded fixed-point evaluation did not reach a stable integer.
    #[error("Q7593: Grover iteration planning did not converge")]
    DidNotConverge,
}

/// Return the conventional Grover iteration count.
///
/// The result is `floor((pi / 4) * sqrt(2**num_qubits / num_marked))`.
/// Chudnovsky evaluation and integer square roots use bounded arbitrary-
/// precision integers. Two independently refined fixed-point evaluations must
/// agree before the integer is returned.
///
/// # Errors
///
/// Returns [`GroverPlanError`] when either count is outside its supported
/// domain or guarded evaluation cannot establish a stable integer.
pub fn grover_iteration_count(
    num_qubits: usize,
    num_marked: &BigUint,
) -> Result<BigUint, GroverPlanError> {
    if num_qubits == 0 {
        return Err(GroverPlanError::EmptySearch);
    }
    if num_qubits > MAX_GROVER_QUBITS {
        return Err(GroverPlanError::SearchTooWide);
    }
    let search_space = BigUint::one() << num_qubits;
    if num_marked.is_zero() || num_marked > &search_space {
        return Err(GroverPlanError::InvalidMarkedCount);
    }

    let result_digits = (num_qubits * 30_103).div_ceil(200_000);
    let precision = result_digits.max(50) + EXTRA_DECIMAL_DIGITS;
    let first = iteration_count_at_precision(&search_space, num_marked, precision);
    let second = iteration_count_at_precision(
        &search_space,
        num_marked,
        precision + CONVERGENCE_DECIMAL_DIGITS,
    );
    if first == second {
        Ok(first)
    } else {
        Err(GroverPlanError::DidNotConverge)
    }
}

fn iteration_count_at_precision(
    search_space: &BigUint,
    num_marked: &BigUint,
    precision: usize,
) -> BigUint {
    let scale = BigUint::from(10_u8).pow(u32::try_from(precision).expect("bounded precision"));
    let radicand = search_space * &scale * &scale / num_marked;
    let root = integer_sqrt(&radicand);
    decimal_pi_scaled(&scale, precision) * root / (BigUint::from(4_u8) * &scale * &scale)
}

fn decimal_pi_scaled(scale: &BigUint, precision: usize) -> BigUint {
    let signed_scale = BigInt::from(scale.clone());
    let mut multiplier = BigInt::one();
    let mut linear = BigInt::from(13_591_409_u32);
    let mut exponential = BigInt::one();
    let mut coefficient = BigInt::from(6_u8);
    let mut series = &linear * &signed_scale;
    for index in 1..=(precision / 14 + 2) {
        let cubic = &coefficient * &coefficient * &coefficient;
        multiplier =
            (cubic - BigInt::from(16_u8) * &coefficient) * multiplier / BigInt::from(index).pow(3);
        linear += 545_140_134_u32;
        exponential *= -262_537_412_640_768_000_i64;
        series += &multiplier * &linear * &signed_scale / &exponential;
        coefficient += 12_u8;
    }
    let square_root = integer_sqrt(&(BigUint::from(10_005_u32) * scale * scale));
    let numerator = BigInt::from(426_880_u32) * BigInt::from(square_root) * signed_scale;
    let value = numerator / series;
    let (sign, digits) = value.to_bytes_be();
    debug_assert_eq!(sign, Sign::Plus);
    BigUint::from_bytes_be(&digits)
}

fn integer_sqrt(value: &BigUint) -> BigUint {
    if value.is_zero() {
        return BigUint::zero();
    }
    let mut estimate = BigUint::one() << value.bits().div_ceil(2);
    loop {
        let next = (&estimate + value / &estimate) >> 1;
        if next >= estimate {
            return estimate;
        }
        estimate = next;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::ToPrimitive;

    #[test]
    fn derives_known_iteration_counts() {
        assert_eq!(
            grover_iteration_count(4, &BigUint::one()).unwrap(),
            BigUint::from(3_u8)
        );
        assert_eq!(
            grover_iteration_count(10, &BigUint::one()).unwrap(),
            BigUint::from(25_u8)
        );
        assert_eq!(
            grover_iteration_count(3, &BigUint::from(8_u8)).unwrap(),
            BigUint::zero()
        );
    }

    #[test]
    fn agrees_with_the_small_binary64_reference_domain() {
        for num_qubits in 1..=30 {
            let search_space = 1_u32 << num_qubits;
            for num_marked in [1, search_space / 3, search_space / 2, search_space] {
                let num_marked = num_marked.max(1);
                let expected = ((std::f64::consts::PI / 4.0)
                    * (f64::from(search_space) / f64::from(num_marked)).sqrt())
                .floor()
                .to_u64()
                .unwrap();
                assert_eq!(
                    grover_iteration_count(num_qubits, &BigUint::from(num_marked)).unwrap(),
                    BigUint::from(expected)
                );
            }
        }
    }

    #[test]
    fn preserves_the_existing_large_reference_value() {
        let expected = BigUint::parse_bytes(
            concat!(
                "105304677233626590548617053711398470263139993283723136513986712720259514",
                "455690247299484713430619315866109428242290833713318232291563997903855884",
                "43550958149"
            )
            .as_bytes(),
            10,
        )
        .unwrap();
        assert_eq!(
            grover_iteration_count(1024, &BigUint::one()).unwrap(),
            expected
        );
    }

    #[test]
    fn rejects_invalid_or_unbounded_search_spaces() {
        assert_eq!(
            grover_iteration_count(0, &BigUint::one()),
            Err(GroverPlanError::EmptySearch)
        );
        assert_eq!(
            grover_iteration_count(MAX_GROVER_QUBITS + 1, &BigUint::one()),
            Err(GroverPlanError::SearchTooWide)
        );
        assert_eq!(
            grover_iteration_count(2, &BigUint::from(5_u8)),
            Err(GroverPlanError::InvalidMarkedCount)
        );
    }

    #[test]
    fn integer_square_root_is_flooring() {
        for value in 0_u32..1000 {
            let root = integer_sqrt(&BigUint::from(value));
            assert!(&root * &root <= BigUint::from(value));
            assert!((&root + 1_u8) * (&root + 1_u8) > BigUint::from(value));
        }
    }
}
