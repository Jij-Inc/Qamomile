//! Exact binary64-to-integer objective encoding for bounded GAS arithmetic.

use num_bigint::BigInt;
use num_traits::{Signed, Zero};
use thiserror::Error;

use super::{CanonicalPolynomial, PolynomialDomain};
use crate::binary64::{SUBNORMAL_EXPONENT, float_to_subnormal_units};

/// Maximum sign-inclusive register width for exact GAS integer encoding.
pub const MAX_EXACT_GAS_BITS: usize = 54;

/// Failure to encode the original binary objective without rounding.
#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum IntegerEncodingError {
    /// GAS needs a finite threshold and a BINARY-domain objective.
    #[error("Q7570: exact GAS encoding requires a BINARY model and finite threshold")]
    InvalidInput,
    /// The common power-of-two scale or scaled coefficients are not finite.
    #[error("Q7570: exact GAS encoding exceeds finite binary64 scaling")]
    ScaleRange,
    /// The exact objective difference would exceed the construction budget.
    #[error(
        "Q7570: exact GAS encoding needs {required} output bits; maximum is {MAX_EXACT_GAS_BITS}"
    )]
    Width {
        /// Sign-inclusive width needed by the exact objective envelope.
        required: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polynomial::normalize_polynomial;
    use num_bigint::BigUint;

    fn model(coefficients: &[f64], constant: f64) -> CanonicalPolynomial {
        let terms = (0..coefficients.len())
            .map(|index| vec![BigUint::from(index)])
            .collect::<Vec<_>>();
        normalize_polynomial(
            PolynomialDomain::Binary,
            &terms,
            coefficients,
            constant,
            None,
        )
        .expect("valid test polynomial")
    }

    #[test]
    fn fractional_encoding_and_negative_ceil_are_exact() {
        let source = model(&[0.25, -0.5], 0.125);
        let encoded = ExactBinaryEncoding::new(&source, -0.2).expect("exact encoding");
        assert_eq!(encoded.scale().to_bits(), 8.0_f64.to_bits());
        assert_eq!(encoded.threshold(), &BigInt::from(-1));
        assert_eq!(encoded.polynomial().constant().to_bits(), 1.0_f64.to_bits());
        assert_eq!(
            encoded.polynomial().terms()[0].coefficient().to_bits(),
            2.0_f64.to_bits()
        );
        assert_eq!(
            encoded.polynomial().terms()[1].coefficient().to_bits(),
            (-4.0_f64).to_bits()
        );
        assert_eq!(source.constant().to_bits(), 0.125_f64.to_bits());
        assert_eq!(encoded.output_bits(), 4);
    }

    #[test]
    fn encoding_keeps_small_coefficients_and_rejects_scale_overflow() {
        let encoded = ExactBinaryEncoding::new(&model(&[2.0_f64.powi(-45)], 0.0), 0.0)
            .expect("small exact coefficient");
        assert_eq!(
            encoded.polynomial().terms()[0].coefficient().to_bits(),
            1.0_f64.to_bits()
        );
        assert_eq!(encoded.scale().to_bits(), 2.0_f64.powi(45).to_bits());
        assert_eq!(
            ExactBinaryEncoding::new(&model(&[f64::from_bits(1)], 0.0), 0.0),
            Err(IntegerEncodingError::ScaleRange)
        );
    }

    #[test]
    fn sign_inclusive_width_budget_is_checked_before_scaling() {
        let allowed = ExactBinaryEncoding::new(&model(&[2.0_f64.powi(53) - 1.0], 0.0), 0.0)
            .expect("last allowed positive integer");
        assert_eq!(allowed.output_bits(), MAX_EXACT_GAS_BITS);
        assert_eq!(
            ExactBinaryEncoding::new(&model(&[2.0_f64.powi(53)], 0.0), 0.0),
            Err(IntegerEncodingError::Width { required: 55 })
        );
        assert_eq!(
            ExactBinaryEncoding::new(&model(&[1.0], 0.0), f64::INFINITY),
            Err(IntegerEncodingError::InvalidInput)
        );
    }
}

/// Checked exact objective, threshold, and minimal two's-complement width.
#[derive(Clone, Debug, PartialEq)]
pub struct ExactBinaryEncoding {
    polynomial: CanonicalPolynomial,
    scale: f64,
    threshold: BigInt,
    output_bits: usize,
}

impl ExactBinaryEncoding {
    /// Scale all original coefficients exactly and round only the strict bound.
    ///
    /// The least common power of two makes each binary64 coefficient integral.
    /// An integer objective is below a real threshold exactly when it is below
    /// its ceiling. Bounds are accumulated as integers, without cancellation or
    /// binary64 rounding. No assignment enumeration or circuit expansion occurs.
    ///
    /// # Errors
    ///
    /// Returns [`IntegerEncodingError`] for a non-BINARY model, nonfinite
    /// threshold, nonfinite scale, or an over-budget objective register.
    pub fn new(source: &CanonicalPolynomial, threshold: f64) -> Result<Self, IntegerEncodingError> {
        if source.domain != PolynomialDomain::Binary || !threshold.is_finite() {
            return Err(IntegerEncodingError::InvalidInput);
        }
        let mut fraction_bits = 0_usize;
        for value in
            std::iter::once(source.constant).chain(source.terms.iter().map(|term| term.coefficient))
        {
            let zeros = float_to_subnormal_units(value)
                .trailing_zeros()
                .unwrap_or(1074);
            let required = usize::try_from(1074_u64.saturating_sub(zeros))
                .map_err(|_| IntegerEncodingError::ScaleRange)?;
            fraction_bits = fraction_bits.max(required);
        }
        if fraction_bits > 1023 {
            return Err(IntegerEncodingError::ScaleRange);
        }
        let exponent =
            i32::try_from(fraction_bits).map_err(|_| IntegerEncodingError::ScaleRange)?;
        let scale = 2.0_f64.powi(exponent);
        let shift = SUBNORMAL_EXPONENT - fraction_bits;
        let threshold_units = float_to_subnormal_units(threshold);
        let floor = &threshold_units >> shift;
        let remainder = threshold_units - (&floor << shift);
        let threshold = floor + u8::from(!remainder.is_zero());
        let constant = float_to_subnormal_units(source.constant) >> shift;
        let mut minimum = &constant - &threshold;
        let mut maximum = minimum.clone();
        for term in &source.terms {
            let coefficient = float_to_subnormal_units(term.coefficient) >> shift;
            if coefficient.is_negative() {
                minimum += coefficient;
            } else {
                maximum += coefficient;
            }
        }
        let negative_bits = if minimum.is_negative() {
            (-minimum - 1_u8).bits() + 1
        } else {
            1
        };
        let positive_bits = if maximum.is_positive() {
            maximum.bits() + 1
        } else {
            1
        };
        let output_bits = usize::try_from(negative_bits.max(positive_bits).max(2))
            .map_err(|_| IntegerEncodingError::ScaleRange)?;
        if output_bits > MAX_EXACT_GAS_BITS {
            return Err(IntegerEncodingError::Width {
                required: output_bits,
            });
        }
        let mut polynomial = source.clone();
        polynomial.constant *= scale;
        for term in &mut polynomial.terms {
            term.coefficient *= scale;
        }
        if !polynomial.constant.is_finite()
            || polynomial
                .terms
                .iter()
                .any(|term| !term.coefficient.is_finite())
        {
            return Err(IntegerEncodingError::ScaleRange);
        }
        Ok(Self {
            polynomial,
            scale,
            threshold,
            output_bits,
        })
    }

    /// Return the exactly scaled polynomial, retaining original variable IDs.
    #[must_use]
    pub const fn polynomial(&self) -> &CanonicalPolynomial {
        &self.polynomial
    }

    /// Return the finite positive power-of-two multiplier.
    #[must_use]
    pub const fn scale(&self) -> f64 {
        self.scale
    }

    /// Return the exact integer threshold for a strict comparison.
    #[must_use]
    pub const fn threshold(&self) -> &BigInt {
        &self.threshold
    }

    /// Return the minimal safe sign-inclusive objective register width.
    #[must_use]
    pub const fn output_bits(&self) -> usize {
        self.output_bits
    }
}
