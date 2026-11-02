//! Bounded integer plans for reversible arithmetic and short-DLP factoring.

use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{One, Zero};
use thiserror::Error;

/// Largest register accepted by linear wrapping/modular addition planning.
pub const MAX_ARITHMETIC_WIDTH: usize = 4096;
/// Largest register accepted by the quadratic modular-multiplication factory.
pub const MAX_MODULAR_MULTIPLICATION_WIDTH: usize = 512;
/// Largest phase-estimation register accepted by order-finding construction.
pub const MAX_ORDER_FINDING_PRECISION: usize = 4096;

/// Failure to derive a bounded number-theory plan.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum NumberTheoryError {
    /// An arithmetic register has no components.
    #[error("Q7630: arithmetic width must be positive")]
    EmptyWidth,
    /// A linear arithmetic plan exceeds its generation budget.
    #[error("Q7631: arithmetic width exceeds the {MAX_ARITHMETIC_WIDTH}-qubit budget")]
    WidthBudget,
    /// A modulus does not fit strictly inside the selected register.
    #[error("Q7632: modulus must satisfy 2 <= modulus < 2**width")]
    InvalidModulus,
    /// Modular addition cannot reserve its required overflow component.
    #[error(
        "Q7638: modular arithmetic width must be below the {MAX_ARITHMETIC_WIDTH}-qubit budget"
    )]
    ModularWidthBudget,
    /// A modular multiplier is not a unit.
    #[error("Q7633: value must be coprime to modulus")]
    NonUnit,
    /// Quadratic multiplication synthesis exceeds its explicit budget.
    #[error(
        "Q7634: modular multiplication width exceeds the {MAX_MODULAR_MULTIPLICATION_WIDTH}-qubit budget"
    )]
    MultiplicationBudget,
    /// An order-finding group configuration is invalid.
    #[error("Q7640: order-finding base must satisfy 1 < base < modulus and be coprime to modulus")]
    InvalidOrderFindingGroup,
    /// The order-finding modulus is too small.
    #[error("Q7641: order-finding modulus must be greater than two")]
    InvalidOrderFindingModulus,
    /// A phase-estimation register has no components.
    #[error("Q7642: order-finding precision must be positive")]
    EmptyPrecision,
    /// A phase-estimation register exceeds its generation budget.
    #[error(
        "Q7643: order-finding precision exceeds the {MAX_ORDER_FINDING_PRECISION}-qubit budget"
    )]
    PrecisionBudget,
}

/// A positive modulus fitting a nonempty register, independent of synthesis limits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CheckedModulus(i64);

impl CheckedModulus {
    /// Check the total-domain modular operation without materializing its domain.
    pub(crate) fn new(width: u64, modulus: i64) -> Option<Self> {
        (width > 0 && modulus > 0 && (width >= 63 || modulus <= (1_i64 << width)))
            .then_some(Self(modulus))
    }

    /// Normalize signed parameters, including the minimum signed integer.
    pub(crate) fn normalize(self, value: i64) -> i64 {
        value.rem_euclid(self.0)
    }

    /// Decide whether modular addition fixes the complete register domain.
    pub(crate) fn is_additive_identity(self, value: i64) -> bool {
        self.normalize(value) == 0
    }

    /// Check that a multiplier has an inverse, including the trivial ring.
    pub(crate) fn unit(self, value: i64) -> Option<ModularUnit> {
        let residue = self.normalize(value);
        let (mut left, mut right) = (residue, self.0);
        while right != 0 {
            (left, right) = (right, left % right);
        }
        (left == 1).then_some(ModularUnit {
            modulus: self,
            residue,
        })
    }
}

/// A normalized unit of a checked modular domain.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ModularUnit {
    modulus: CheckedModulus,
    residue: i64,
}

impl ModularUnit {
    /// Return the normalized multiplier.
    pub(crate) const fn residue(self) -> i64 {
        self.residue
    }

    /// Decide exact identity on both valid residues and the fixed complement.
    pub(crate) fn is_identity(self) -> bool {
        self.residue == self.modulus.normalize(1)
    }

    /// Return the checked inverse using bounded wide Euclidean intermediates.
    pub(crate) fn inverse(self) -> Self {
        let (mut previous, mut current) = (1_i128, 0_i128);
        let (mut remainder, mut next) = (i128::from(self.residue), i128::from(self.modulus.0));
        while next != 0 {
            let quotient = remainder / next;
            (remainder, next) = (next, remainder - quotient * next);
            (previous, current) = (current, previous - quotient * current);
        }
        let residue = i64::try_from(previous.rem_euclid(i128::from(self.modulus.0)))
            .expect("a residue of a positive i64 modulus fits i64");
        Self { residue, ..self }
    }
}

/// One canonical modular multiplication schedule.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModularMultiplicationPlan {
    width: usize,
    modulus: BigUint,
    multiplier: BigUint,
    inverse: BigUint,
    accumulate: Vec<BigUint>,
    erase: Vec<BigUint>,
}

impl ModularMultiplicationPlan {
    /// Return the value-register width.
    #[must_use]
    pub const fn width(&self) -> usize {
        self.width
    }

    /// Return the canonical positive modulus.
    #[must_use]
    pub const fn modulus(&self) -> &BigUint {
        &self.modulus
    }

    /// Return the canonical multiplier residue.
    #[must_use]
    pub const fn multiplier(&self) -> &BigUint {
        &self.multiplier
    }

    /// Return the canonical modular inverse.
    #[must_use]
    pub const fn inverse(&self) -> &BigUint {
        &self.inverse
    }

    /// Return one shifted multiplier residue per multiplicand component.
    #[must_use]
    pub fn accumulate(&self) -> &[BigUint] {
        &self.accumulate
    }

    /// Return one shifted inverse residue per product component.
    #[must_use]
    pub fn erase(&self) -> &[BigUint] {
        &self.erase
    }
}

/// One validated Ekerå–Håstad short-DLP coordinate plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EkeraHastadPlan {
    width: usize,
    parameter_width: usize,
    inverse_y: BigUint,
}

impl EkeraHastadPlan {
    /// Return the modular value width.
    #[must_use]
    pub const fn width(&self) -> usize {
        self.width
    }

    /// Return the selected short parameter width.
    #[must_use]
    pub const fn parameter_width(&self) -> usize {
        self.parameter_width
    }

    /// Return the long phase-estimation width.
    #[must_use]
    pub const fn long_precision(&self) -> usize {
        2 * self.parameter_width
    }

    /// Return the short phase-estimation width.
    #[must_use]
    pub const fn short_precision(&self) -> usize {
        self.parameter_width
    }

    /// Return the complete modular-multiplication target width.
    #[must_use]
    pub const fn target_width(&self) -> usize {
        2 * self.width + 2
    }

    /// Return the canonical inverse of `g**(modulus + 1)` modulo `modulus`.
    #[must_use]
    pub const fn inverse_y(&self) -> &BigUint {
        &self.inverse_y
    }
}

/// Return the set-bit offsets of a signed wrapping-addition constant.
///
/// # Errors
///
/// Returns [`NumberTheoryError`] when `width` is empty or exceeds the linear
/// arithmetic budget.
pub fn wrapping_addition_offsets(
    width: usize,
    value: &BigInt,
) -> Result<Vec<usize>, NumberTheoryError> {
    validate_width(width)?;
    let modulus = BigUint::one() << width;
    let normalized = canonical_residue(value, &modulus);
    let bounded_width = u16::try_from(width).map_err(|_| NumberTheoryError::WidthBudget)?;
    Ok((0..bounded_width)
        .filter(|offset| normalized.bit(u64::from(*offset)))
        .map(usize::from)
        .collect())
}

/// Return the canonical residue for modular constant addition.
///
/// # Errors
///
/// Returns [`NumberTheoryError`] when the width or modulus is invalid.
pub fn modular_addition_residue(
    width: usize,
    value: &BigInt,
    modulus: &BigUint,
) -> Result<BigUint, NumberTheoryError> {
    validate_modulus(width, modulus)?;
    Ok(canonical_residue(value, modulus))
}

/// Derive one bounded modular multiplication schedule.
///
/// # Errors
///
/// Returns [`NumberTheoryError`] when the register, modulus, or multiplier is
/// invalid, or quadratic factory generation would exceed its budget.
pub fn plan_modular_multiplication(
    width: usize,
    value: &BigInt,
    modulus: &BigUint,
) -> Result<ModularMultiplicationPlan, NumberTheoryError> {
    validate_modulus(width, modulus)?;
    if width > MAX_MODULAR_MULTIPLICATION_WIDTH {
        return Err(NumberTheoryError::MultiplicationBudget);
    }
    let multiplier = canonical_residue(value, modulus);
    let inverse = modular_inverse(&multiplier, modulus).ok_or(NumberTheoryError::NonUnit)?;
    let accumulate = shifted_residues(&multiplier, modulus, width);
    let erase = shifted_residues(&inverse, modulus, width);
    Ok(ModularMultiplicationPlan {
        width,
        modulus: modulus.clone(),
        multiplier,
        inverse,
        accumulate,
        erase,
    })
}

/// Derive one bounded Ekerå–Håstad short-DLP coordinate plan.
///
/// # Errors
///
/// Returns [`NumberTheoryError`] when the group or selected parameter width is
/// invalid or exceeds a generation budget.
pub fn plan_ekera_hastad(
    generator: &BigUint,
    modulus: &BigUint,
    parameter_width: Option<usize>,
) -> Result<EkeraHastadPlan, NumberTheoryError> {
    let width = validate_order_group(generator, modulus)?;
    let selected = parameter_width.unwrap_or(width.div_ceil(2) + 1);
    validate_precision(selected)?;
    let long_precision = selected
        .checked_mul(2)
        .ok_or(NumberTheoryError::PrecisionBudget)?;
    validate_precision(long_precision)?;
    let exponent = modulus + BigUint::one();
    let y = generator.modpow(&exponent, modulus);
    let inverse_y = modular_inverse(&y, modulus).ok_or(NumberTheoryError::NonUnit)?;
    Ok(EkeraHastadPlan {
        width,
        parameter_width: selected,
        inverse_y,
    })
}

fn validate_width(width: usize) -> Result<(), NumberTheoryError> {
    if width == 0 {
        return Err(NumberTheoryError::EmptyWidth);
    }
    if width > MAX_ARITHMETIC_WIDTH {
        return Err(NumberTheoryError::WidthBudget);
    }
    Ok(())
}

fn validate_modulus(width: usize, modulus: &BigUint) -> Result<(), NumberTheoryError> {
    validate_width(width)?;
    if width == MAX_ARITHMETIC_WIDTH {
        return Err(NumberTheoryError::ModularWidthBudget);
    }
    if modulus < &BigUint::from(2_u8) || modulus >= &(BigUint::one() << width) {
        return Err(NumberTheoryError::InvalidModulus);
    }
    Ok(())
}

fn validate_order_group(base: &BigUint, modulus: &BigUint) -> Result<usize, NumberTheoryError> {
    if modulus <= &BigUint::from(2_u8) {
        return Err(NumberTheoryError::InvalidOrderFindingModulus);
    }
    let width = usize::try_from(modulus.bits()).map_err(|_| NumberTheoryError::WidthBudget)?;
    if width > MAX_MODULAR_MULTIPLICATION_WIDTH {
        return Err(NumberTheoryError::MultiplicationBudget);
    }
    if base <= &BigUint::one()
        || base >= modulus
        || greatest_common_divisor(base, modulus) != BigUint::one()
    {
        return Err(NumberTheoryError::InvalidOrderFindingGroup);
    }
    Ok(width)
}

fn validate_precision(precision: usize) -> Result<(), NumberTheoryError> {
    if precision == 0 {
        return Err(NumberTheoryError::EmptyPrecision);
    }
    if precision > MAX_ORDER_FINDING_PRECISION {
        return Err(NumberTheoryError::PrecisionBudget);
    }
    Ok(())
}

fn canonical_residue(value: &BigInt, modulus: &BigUint) -> BigUint {
    let signed_modulus = BigInt::from(modulus.clone());
    let mut residue = value % &signed_modulus;
    if residue.sign() == Sign::Minus {
        residue += signed_modulus;
    }
    residue
        .to_biguint()
        .expect("a canonical residue is non-negative")
}

fn shifted_residues(value: &BigUint, modulus: &BigUint, width: usize) -> Vec<BigUint> {
    let mut current = value.clone();
    let mut residues = Vec::with_capacity(width);
    for _ in 0..width {
        residues.push(current.clone());
        current = (current << 1) % modulus;
    }
    residues
}

fn modular_inverse(value: &BigUint, modulus: &BigUint) -> Option<BigUint> {
    let mut previous = BigInt::zero();
    let mut current = BigInt::one();
    let mut remainder = BigInt::from(modulus.clone());
    let mut next_remainder = BigInt::from(value.clone());
    while !next_remainder.is_zero() {
        let quotient = &remainder / &next_remainder;
        (previous, current) = (current.clone(), previous - &quotient * current);
        (remainder, next_remainder) = (
            next_remainder.clone(),
            remainder - quotient * next_remainder,
        );
    }
    if remainder != BigInt::one() {
        return None;
    }
    Some(canonical_residue(&previous, modulus))
}

fn greatest_common_divisor(left: &BigUint, right: &BigUint) -> BigUint {
    let mut left = left.clone();
    let mut right = right.clone();
    while !right.is_zero() {
        let remainder = &left % &right;
        left = right;
        right = remainder;
    }
    left
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_modular_domains_cover_signed_extrema_and_trivial_ring() {
        for (width, modulus) in [(0, 1), (3, 0), (3, -1), (3, 9)] {
            assert_eq!(CheckedModulus::new(width, modulus), None);
        }
        let trivial = CheckedModulus::new(1, 1).unwrap();
        for factor in [i64::MIN, -1, 0, 1, i64::MAX] {
            let unit = trivial.unit(factor).unwrap();
            assert_eq!(unit.residue(), 0);
            assert!(unit.is_identity());
            assert_eq!(unit.inverse(), unit);
            assert!(trivial.is_additive_identity(factor));
        }
        assert!(CheckedModulus::new(u64::MAX, i64::MAX).is_some());
        let signed = CheckedModulus::new(63, i64::MAX).unwrap();
        let unit = signed.unit(i64::MIN).unwrap();
        assert_eq!(unit.residue(), i64::MAX - 1);
        assert_eq!(unit.inverse(), unit);
        assert!(!unit.is_identity());
    }

    #[test]
    fn checked_modular_units_match_exhaustive_small_integer_domains() {
        for modulus in 1..=16_i64 {
            let checked = CheckedModulus::new(4, modulus).unwrap();
            for factor in -32..=32 {
                let expected = (0..modulus).find(|inverse| {
                    (factor * inverse).rem_euclid(modulus) == 1_i64.rem_euclid(modulus)
                });
                let unit = checked.unit(factor);
                assert_eq!(unit.is_some(), expected.is_some());
                if let Some(unit) = unit {
                    assert_eq!(Some(unit.inverse().residue()), expected);
                    assert_eq!(unit.inverse().inverse(), unit);
                    assert_eq!(
                        unit.is_identity(),
                        factor.rem_euclid(modulus) == 1 % modulus
                    );
                }
            }
        }
    }

    #[test]
    fn wrapping_offsets_use_signed_arbitrary_precision_residues() {
        assert_eq!(
            wrapping_addition_offsets(4, &BigInt::from(-1)).unwrap(),
            vec![0, 1, 2, 3]
        );
        let huge: BigInt = -(BigInt::one() << 10_000_usize) + 5_u8;
        assert_eq!(wrapping_addition_offsets(3, &huge).unwrap(), vec![0, 2]);
    }

    #[test]
    fn modular_plan_derives_inverse_and_shifted_residues() {
        let plan = plan_modular_multiplication(3, &BigInt::from(-3), &BigUint::from(5_u8)).unwrap();
        assert_eq!(plan.multiplier(), &BigUint::from(2_u8));
        assert_eq!(plan.inverse(), &BigUint::from(3_u8));
        assert_eq!(plan.accumulate(), &[2_u8, 4, 3].map(BigUint::from));
        assert_eq!(plan.erase(), &[3_u8, 1, 2].map(BigUint::from));
    }

    #[test]
    fn modular_plan_matches_small_signed_unit_domains_exhaustively() {
        for modulus in 2_i32..64 {
            for value in -128_i32..=128 {
                let result = plan_modular_multiplication(
                    6,
                    &BigInt::from(value),
                    &BigUint::from(u32::try_from(modulus).unwrap()),
                );
                let residue = value.rem_euclid(modulus);
                if small_gcd(residue, modulus) == 1 {
                    let plan = result.unwrap();
                    assert_eq!(
                        plan.multiplier(),
                        &BigUint::from(u32::try_from(residue).unwrap())
                    );
                    assert_eq!(
                        plan.multiplier() * plan.inverse() % plan.modulus(),
                        BigUint::one()
                    );
                    for (index, shifted) in plan.accumulate().iter().enumerate() {
                        let expected = (BigUint::from(u32::try_from(residue).unwrap()) << index)
                            % plan.modulus();
                        assert_eq!(shifted, &expected);
                    }
                } else {
                    assert_eq!(result, Err(NumberTheoryError::NonUnit));
                }
            }
        }
    }

    #[test]
    fn short_dlp_plan_owns_width_precision_and_inverse() {
        let ekera = plan_ekera_hastad(&BigUint::from(2_u8), &BigUint::from(15_u8), None).unwrap();
        assert_eq!(
            (
                ekera.width(),
                ekera.parameter_width(),
                ekera.long_precision(),
                ekera.short_precision(),
                ekera.target_width(),
            ),
            (4, 3, 6, 3, 10)
        );
        assert_eq!(ekera.inverse_y(), &BigUint::one());
    }

    #[test]
    fn invalid_domains_and_generation_budgets_fail_closed() {
        assert_eq!(
            modular_addition_residue(2, &BigInt::one(), &BigUint::from(4_u8)),
            Err(NumberTheoryError::InvalidModulus)
        );
        assert_eq!(
            modular_addition_residue(MAX_ARITHMETIC_WIDTH, &BigInt::one(), &BigUint::from(3_u8),),
            Err(NumberTheoryError::ModularWidthBudget)
        );
        assert_eq!(
            plan_modular_multiplication(3, &BigInt::from(2_u8), &BigUint::from(4_u8)),
            Err(NumberTheoryError::NonUnit)
        );
        assert_eq!(
            plan_ekera_hastad(&BigUint::from(3_u8), &BigUint::from(6_u8), None,),
            Err(NumberTheoryError::InvalidOrderFindingGroup)
        );
        assert_eq!(
            plan_ekera_hastad(
                &BigUint::from(2_u8),
                &BigUint::from(15_u8),
                Some(MAX_ORDER_FINDING_PRECISION),
            ),
            Err(NumberTheoryError::PrecisionBudget)
        );
    }

    fn small_gcd(mut left: i32, mut right: i32) -> i32 {
        while right != 0 {
            (left, right) = (right, left % right);
        }
        left.abs()
    }
}
