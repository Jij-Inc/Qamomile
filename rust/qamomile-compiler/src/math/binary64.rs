//! Exact-integer accumulation helpers for finite binary64 values.

use num_bigint::{BigInt, BigUint, Sign};
use num_traits::ToPrimitive;

pub(crate) const SUBNORMAL_EXPONENT: usize = 1074;

pub(crate) struct ExactSum {
    total: Option<BigInt>,
}

impl ExactSum {
    pub(crate) fn new() -> Self {
        Self {
            total: Some(BigInt::from(0_u8)),
        }
    }

    pub(crate) fn push(&mut self, value: f64) -> bool {
        let Some(total) = &mut self.total else {
            return false;
        };
        if !value.is_finite() {
            self.total = None;
            return false;
        }
        *total += float_to_subnormal_units(value);
        true
    }

    pub(crate) fn finish(self) -> Option<f64> {
        self.total
            .and_then(|total| scaled_integer_to_f64(&total, SUBNORMAL_EXPONENT))
    }
}

pub(crate) fn float_to_subnormal_units(value: f64) -> BigInt {
    if value == 0.0 {
        return BigInt::from(0_u8);
    }
    let bits = value.to_bits();
    let sign = if bits >> 63 == 0 {
        Sign::Plus
    } else {
        Sign::Minus
    };
    let exponent_bits = ((bits >> 52) & 0x7ff) as usize;
    let fraction = bits & ((1_u64 << 52) - 1);
    let (significand, shift) = if exponent_bits == 0 {
        (fraction, 0)
    } else {
        ((1_u64 << 52) | fraction, exponent_bits - 1)
    };
    BigInt::from_biguint(sign, BigUint::from(significand) << shift)
}

pub(crate) fn exact_sum(values: impl IntoIterator<Item = f64>) -> Option<f64> {
    let mut total = ExactSum::new();
    for value in values {
        if !total.push(value) {
            return None;
        }
    }
    total.finish()
}

/// Round an exact signed ratio to nearest binary64, with ties to even.
pub(crate) fn ratio_to_f64(numerator: &BigInt, denominator: &BigUint) -> Option<f64> {
    if denominator == &BigUint::from(0_u8) {
        return None;
    }
    if numerator.sign() == Sign::NoSign {
        return Some(0.0);
    }
    let sign = if numerator.sign() == Sign::Minus {
        1_u64 << 63
    } else {
        0
    };
    let numerator = numerator.magnitude();
    let mut exponent = i64::try_from(numerator.bits())
        .ok()?
        .checked_sub(i64::try_from(denominator.bits()).ok()?)?;
    if exponent > 1024 {
        return None;
    }
    if exponent < -1075 {
        return Some(f64::from_bits(sign));
    }
    let below_power = if exponent >= 0 {
        numerator < &(denominator << usize::try_from(exponent).ok()?)
    } else {
        (numerator << usize::try_from(-exponent).ok()?) < *denominator
    };
    if below_power {
        exponent -= 1;
    }
    let subnormal = exponent < -1022;
    let shift = if subnormal { 1074 } else { 52 - exponent };
    let (n, d) = if shift >= 0 {
        (
            numerator << usize::try_from(shift).ok()?,
            denominator.clone(),
        )
    } else {
        (
            numerator.clone(),
            denominator << usize::try_from(-shift).ok()?,
        )
    };
    let mut quotient = &n / &d;
    let doubled_remainder = (&n % &d) << 1_usize;
    if doubled_remainder > d || (doubled_remainder == d && quotient.bit(0)) {
        quotient += 1_u8;
    }
    let mut significand = quotient.to_u64()?;
    if subnormal {
        return Some(f64::from_bits(sign | significand));
    }
    if significand == 1_u64 << 53 {
        significand >>= 1;
        exponent += 1;
    }
    if exponent > 1023 {
        return None;
    }
    let biased = u64::try_from(exponent + 1023).ok()?;
    Some(f64::from_bits(
        sign | (biased << 52) | (significand - (1_u64 << 52)),
    ))
}

pub(crate) fn fwht_integer_in_place(values: &mut [BigInt]) {
    let mut step = 1;
    while step < values.len() {
        for start in (0..values.len()).step_by(2 * step) {
            for offset in 0..step {
                let left = values[start + offset].clone();
                let right = values[start + step + offset].clone();
                values[start + offset] = &left + &right;
                values[start + step + offset] = left - right;
            }
        }
        step *= 2;
    }
}

pub(crate) fn scaled_integer_to_f64(value: &BigInt, denominator_exponent: usize) -> Option<f64> {
    let sign = match value.sign() {
        Sign::NoSign => return Some(0.0),
        Sign::Plus => 1.0,
        Sign::Minus => -1.0,
    };
    let magnitude = value.magnitude();
    let bits = usize::try_from(magnitude.bits()).ok()?;
    let unbiased_exponent = i32::try_from(bits)
        .ok()
        .and_then(|bits| bits.checked_sub(1))
        .and_then(|bits| {
            i32::try_from(denominator_exponent)
                .ok()
                .and_then(|denominator| bits.checked_sub(denominator))
        })?;
    if unbiased_exponent < -1022 {
        let shift = denominator_exponent.checked_sub(SUBNORMAL_EXPONENT)?;
        let units = round_shift_right(magnitude, shift).to_u64()?;
        return Some(sign * f64::from_bits(units));
    }
    let shift = bits.saturating_sub(53);
    let significand = round_shift_right(magnitude, shift).to_u64()?;
    let scale_exponent = i32::try_from(shift).ok().and_then(|shift| {
        i32::try_from(denominator_exponent)
            .ok()
            .and_then(|denominator| shift.checked_sub(denominator))
    })?;
    let significand = significand.to_f64()?;
    let result = sign * significand * 2.0_f64.powi(scale_exponent);
    result.is_finite().then_some(result)
}

fn round_shift_right(value: &BigUint, shift: usize) -> BigUint {
    if shift == 0 {
        return value.clone();
    }
    let mut rounded = value >> shift;
    let remainder = value - (&rounded << shift);
    let half = BigUint::from(1_u8) << (shift - 1);
    if remainder > half
        || (remainder == half && (&rounded & BigUint::from(1_u8)) == BigUint::from(1_u8))
    {
        rounded += 1_u8;
    }
    rounded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_ratio_rounding_handles_ties_subnormals_and_overflow() {
        use num_traits::One;
        let denominator = BigUint::one() << 53_usize;
        assert_eq!(
            ratio_to_f64(&BigInt::from(&denominator + 1_u8), &denominator),
            Some(1.0)
        );
        assert_eq!(
            ratio_to_f64(&BigInt::from(&denominator + 3_u8), &denominator),
            Some(f64::from_bits(1.0_f64.to_bits() + 2))
        );
        let subnormal = BigUint::one() << 1075_usize;
        assert_eq!(ratio_to_f64(&BigInt::from(1_u8), &subnormal), Some(0.0));
        assert_eq!(
            ratio_to_f64(&BigInt::from(3_u8), &subnormal),
            Some(f64::from_bits(2))
        );
        assert_eq!(
            ratio_to_f64(&-(BigInt::one() << 1024_usize), &BigUint::one()),
            None
        );
        assert_eq!(ratio_to_f64(&BigInt::one(), &BigUint::from(0_u8)), None);
    }

    #[test]
    fn exact_ratio_round_trips_finite_binary64_extremes() {
        use num_traits::One;
        let denominator = BigUint::one() << SUBNORMAL_EXPONENT;
        for value in [
            f64::MAX,
            -f64::MAX,
            f64::MIN_POSITIVE,
            f64::from_bits(1),
            -0.1,
            1.0 / 3.0,
        ] {
            assert_eq!(
                ratio_to_f64(&float_to_subnormal_units(value), &denominator),
                Some(value)
            );
        }
    }

    #[test]
    fn exact_sum_retains_cancellation_and_subnormal_residue() {
        let residue = f64::from_bits(1);
        assert_eq!(exact_sum([f64::MAX, -f64::MAX, residue]), Some(residue));
    }

    #[test]
    fn exact_sum_rejects_nonfinite_input_and_output() {
        assert_eq!(exact_sum([f64::INFINITY]), None);
        assert_eq!(exact_sum([f64::MAX, f64::MAX]), None);
    }

    #[test]
    fn streaming_exact_sum_matches_the_bounded_batch_helper() {
        let values = [f64::MAX, -f64::MAX, f64::from_bits(1)];
        let mut sum = ExactSum::new();
        assert!(values.into_iter().all(|value| sum.push(value)));
        assert_eq!(sum.finish(), exact_sum(values));
    }
}
