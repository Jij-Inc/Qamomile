//! Bounded statistics of typed classical samples, independent of measurement physics.

use std::collections::HashSet;

use num_bigint::{BigInt, BigUint};
use num_rational::Ratio;
use num_traits::{One, ToPrimitive, Zero};
use thiserror::Error;

use crate::binary64::{SUBNORMAL_EXPONENT, exact_sum, float_to_subnormal_units};
use crate::result_codec::checked_result_abi;
use crate::runtime::project_literal_component;
use crate::wire::literal_proto::Value;
use crate::{SampleCountSummary, ScalarParameterPathComponent};

type BigRational = Ratio<BigInt>;

/// Maximum number of readouts or independent estimates in one reduction.
pub const MAX_STATISTICS_TERMS: usize = 4096;
/// Maximum aggregate path depth.
pub const MAX_STATISTICS_PATH: usize = 64;
/// Maximum exact count width used by moment arithmetic.
pub const MAX_STATISTICS_COUNT_BITS: u64 = 4096;
const MAX_WORK: usize = 100_000_000;

/// Rejection of malformed, ambiguous, or unrepresentable sample statistics.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum StatisticsError {
    /// The declared result ABI or selected value is incompatible with the readout.
    #[error("Q7400: invalid statistics sample shape: {0}")]
    Shape(String),
    /// Coefficients, scores, moments, or estimates cannot be represented finitely.
    #[error("Q7401: statistics are outside the finite binary64 range")]
    NumericalRange,
    /// A sample reduction exceeds a bounded processing limit.
    #[error("Q7402: statistics exceed the processing budget")]
    Budget,
    /// A parity includes duplicate coordinates.
    #[error("Q7403: parity indices must be distinct")]
    DuplicateIndex,
}

/// An explicit classical readout; it does not assert any quantum observable.
#[derive(Clone, Debug, PartialEq)]
pub struct SampleReadout {
    path: Box<[ScalarParameterPathComponent]>,
    kind: ReadoutKind,
}

#[derive(Clone, Debug, PartialEq)]
enum ReadoutKind {
    Numeric,
    Parity(Box<[usize]>),
}

impl SampleReadout {
    /// Select one numeric scalar, treating Bool and Bit as zero or one.
    ///
    /// # Errors
    /// Returns [`StatisticsError::Budget`] for an oversized path.
    pub fn value(path: Vec<ScalarParameterPathComponent>) -> Result<Self, StatisticsError> {
        Self::new(path, ReadoutKind::Numeric)
    }

    /// Read the product of selected zero/one coordinates as a parity in +/-1.
    ///
    /// The selected value must be a tuple of bits. An empty index list denotes
    /// constant +1. Indices address logical decoded values, never transport bits.
    ///
    /// # Errors
    /// Returns [`StatisticsError`] for excessive or duplicate indices or paths.
    pub fn parity(
        path: Vec<ScalarParameterPathComponent>,
        indices: Vec<usize>,
    ) -> Result<Self, StatisticsError> {
        if indices.len() > MAX_STATISTICS_TERMS {
            return Err(StatisticsError::Budget);
        }
        if indices.iter().collect::<HashSet<_>>().len() != indices.len() {
            return Err(StatisticsError::DuplicateIndex);
        }
        Self::new(path, ReadoutKind::Parity(indices.into_boxed_slice()))
    }

    fn new(
        path: Vec<ScalarParameterPathComponent>,
        kind: ReadoutKind,
    ) -> Result<Self, StatisticsError> {
        if path.len() > MAX_STATISTICS_PATH
            || path
                .iter()
                .any(|p| matches!(p, ScalarParameterPathComponent::Field(s) if s.len() > 4096))
        {
            return Err(StatisticsError::Budget);
        }
        Ok(Self {
            path: path.into_boxed_slice(),
            kind,
        })
    }

    fn work(&self) -> usize {
        self.path.len()
            + match &self.kind {
                ReadoutKind::Numeric => 1,
                ReadoutKind::Parity(indices) => indices.len() + 1,
            }
    }

    fn read(&self, root: &crate::wire::LiteralProto) -> Result<f64, StatisticsError> {
        let selected = self
            .path
            .iter()
            .try_fold(root, project_literal_component)
            .ok_or_else(|| StatisticsError::Shape("invalid aggregate path".to_owned()))?;
        match &self.kind {
            ReadoutKind::Numeric => numeric(selected),
            ReadoutKind::Parity(indices) => {
                let Some(Value::TupleValue(sequence)) = &selected.value else {
                    return Err(StatisticsError::Shape(
                        "parity requires a decoded bit tuple".to_owned(),
                    ));
                };
                let mut odd = false;
                for &index in indices {
                    let value = sequence.values.get(index).ok_or_else(|| {
                        StatisticsError::Shape("parity index is out of range".to_owned())
                    })?;
                    odd ^= match value.value {
                        Some(Value::BoolValue(bit)) => bit,
                        Some(Value::IntValue(0)) => false,
                        Some(Value::IntValue(1)) => true,
                        _ => {
                            return Err(StatisticsError::Shape(
                                "parity requires zero/one bit coordinates".to_owned(),
                            ));
                        }
                    };
                }
                Ok(if odd { -1.0 } else { 1.0 })
            }
        }
    }
}

fn numeric(value: &crate::wire::LiteralProto) -> Result<f64, StatisticsError> {
    let value = match &value.value {
        Some(Value::RealValue(v)) => *v,
        Some(Value::BoolValue(v)) => f64::from(u8::from(*v)),
        Some(Value::IntValue(v)) => v.to_f64().ok_or(StatisticsError::NumericalRange)?,
        Some(Value::BigintValue(v)) => BigInt::from_signed_bytes_be(v)
            .to_f64()
            .ok_or(StatisticsError::NumericalRange)?,
        _ => {
            return Err(StatisticsError::Shape(
                "mean requires one numeric scalar; select an aggregate field explicitly".to_owned(),
            ));
        }
    };
    finite(value)
}

/// A checked estimate with uncertainty of the estimated mean, not shot variance.
#[derive(Clone, Debug, PartialEq)]
pub struct StatisticalEstimate {
    value: f64,
    variance: Option<f64>,
    shots: BigUint,
}

impl StatisticalEstimate {
    /// Import a backend estimate after its uncertainty convention is normalized.
    ///
    /// This records a summary, not invented observations or a sample variance.
    /// # Errors
    /// Rejects nonfinite values, negative errors, invalid counts, and known
    /// uncertainty from only one observation.
    pub fn from_summary(
        value: f64,
        standard_error: Option<f64>,
        shots: BigUint,
    ) -> Result<Self, StatisticsError> {
        finite(value)?;
        if shots.is_zero() || shots.bits() > MAX_STATISTICS_COUNT_BITS {
            return Err(StatisticsError::Budget);
        }
        let variance = standard_error
            .map(|error| {
                finite(error)?;
                if error < 0.0 || shots == BigUint::one() {
                    return Err(StatisticsError::NumericalRange);
                }
                let variance = finite(error * error)?;
                if error > 0.0 && variance == 0.0 {
                    return Err(StatisticsError::NumericalRange);
                }
                Ok(variance)
            })
            .transpose()?;
        Ok(Self {
            value,
            variance,
            shots,
        })
    }

    /// Return the estimated value.
    #[must_use]
    pub const fn value(&self) -> f64 {
        self.value
    }
    /// Return the estimated variance of this mean, if identifiable from samples.
    #[must_use]
    pub const fn variance(&self) -> Option<f64> {
        self.variance
    }
    /// Return the plug-in standard error under the declared independence assumptions.
    #[must_use]
    pub fn standard_error(&self) -> Option<f64> {
        self.variance.map(f64::sqrt)
    }
    /// Return the total number of observations consumed, without expansion.
    #[must_use]
    pub const fn shots(&self) -> &BigUint {
        &self.shots
    }
}

/// One sample mean together with the unbiased variance of individual outcomes.
#[derive(Clone, Debug, PartialEq)]
pub struct SampleMean {
    estimate: StatisticalEstimate,
    sample_variance: Option<f64>,
}

impl SampleMean {
    /// Return the estimated mean and its uncertainty.
    #[must_use]
    pub const fn estimate(&self) -> &StatisticalEstimate {
        &self.estimate
    }
    /// Return the unbiased sample variance; one observation yields None.
    #[must_use]
    pub const fn sample_variance(&self) -> Option<f64> {
        self.sample_variance
    }
}

/// Estimate a linear score from a single joint histogram, retaining covariance.
///
/// Frequency weights are exact counts, not importance weights. Each finite
/// score is accumulated in exact integer binary64 units before moment division.
/// No loop or allocation scales with the expanded shot count.
///
/// # Errors
/// Returns [`StatisticsError`] for incompatible types, invalid readouts,
/// nonfinite input/output or exceeded row/readout/numeric work budgets.
pub fn sample_mean(
    samples: &SampleCountSummary,
    result_abi: &[u8],
    terms: &[(f64, SampleReadout)],
    offset: f64,
) -> Result<SampleMean, StatisticsError> {
    finite(offset)?;
    if terms.len() > MAX_STATISTICS_TERMS || samples.shots().bits() > MAX_STATISTICS_COUNT_BITS {
        return Err(StatisticsError::Budget);
    }
    let work = terms.iter().try_fold(
        samples
            .most_likely()
            .len()
            .checked_add(result_abi.len())
            .and_then(|w| w.checked_add(1))
            .ok_or(StatisticsError::Budget)?,
        |w, (c, r)| {
            finite(*c)?;
            // Each field projection may scan the record layout; account for
            // the full ABI size at every path component before decoding rows.
            r.path
                .len()
                .checked_mul(result_abi.len())
                .and_then(|path_work| path_work.checked_add(r.work()))
                .and_then(|readout_work| w.checked_add(readout_work))
                .ok_or(StatisticsError::Budget)
        },
    )?;
    let numeric_work =
        usize::try_from(samples.shots().bits()).map_err(|_| StatisticsError::Budget)? + 4200;
    if work
        .checked_add(numeric_work)
        .and_then(|w| w.checked_mul(samples.entries().len()))
        .is_none_or(|w| w > MAX_WORK)
    {
        return Err(StatisticsError::Budget);
    }
    let abi = checked_result_abi(result_abi).map_err(|e| StatisticsError::Shape(e.to_string()))?;
    let mut sum = BigInt::zero();
    let mut squares = BigInt::zero();
    for (bits, count) in samples.entries() {
        if count.is_zero() {
            continue;
        }
        let value = abi
            .decode_literal_bits(bits)
            .map_err(|e| StatisticsError::Shape(e.to_string()))?;
        let mut contributions = Vec::with_capacity(terms.len() + 1);
        contributions.push(offset);
        for (coefficient, readout) in terms {
            contributions.push(finite(coefficient * readout.read(&value)?)?);
        }
        let score = exact_sum(contributions).ok_or(StatisticsError::NumericalRange)?;
        let units = float_to_subnormal_units(score);
        let count = BigInt::from(count.clone());
        sum += &units * &count;
        squares += &units * &units * count;
    }
    let n = BigInt::from(samples.shots().clone());
    let value = ratio(&sum, &(&n << SUBNORMAL_EXPONENT))?;
    let (sample_variance, variance) = if samples.shots().is_one() {
        (None, None)
    } else {
        let numerator = &n * squares - &sum * &sum;
        let denominator = (&n * (&n - 1_u8)) << (2 * SUBNORMAL_EXPONENT);
        (
            Some(ratio(&numerator, &denominator)?),
            Some(ratio(&numerator, &(&denominator * &n))?),
        )
    };
    Ok(SampleMean {
        estimate: StatisticalEstimate {
            value,
            variance,
            shots: samples.shots().clone(),
        },
        sample_variance,
    })
}

/// Combine estimates whose underlying observations the caller declares independent.
///
/// Empty input represents the constant offset with zero shots and zero error.
/// A nonzero-weight component with unknown uncertainty makes the sum unknown.
/// No independence is inferred from backend identity, seeds, or numerical values.
///
/// # Errors
/// Returns [`StatisticsError`] for excessive terms, nonfinite coefficients or
/// unrepresentable output statistics.
pub fn combine_independent_estimates(
    terms: &[(f64, StatisticalEstimate)],
    offset: f64,
) -> Result<StatisticalEstimate, StatisticsError> {
    finite(offset)?;
    if terms.len() > MAX_STATISTICS_TERMS {
        return Err(StatisticsError::Budget);
    }
    let mut mean = rational(offset)?;
    let mut variance = Some(BigRational::zero());
    let mut shots = BigUint::zero();
    for (coefficient, estimate) in terms {
        let c = rational(*coefficient)?;
        shots += &estimate.shots;
        mean += &c * rational(estimate.value)?;
        if !c.is_zero() {
            variance = match (variance, estimate.variance) {
                (Some(v), Some(e)) => Some(v + &c * &c * rational(e)?),
                _ => None,
            };
        }
    }
    if shots.bits() > MAX_STATISTICS_COUNT_BITS {
        return Err(StatisticsError::Budget);
    }
    Ok(StatisticalEstimate {
        value: rational_f64(&mean)?,
        variance: variance.as_ref().map(rational_f64).transpose()?,
        shots,
    })
}

fn finite(value: f64) -> Result<f64, StatisticsError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(StatisticsError::NumericalRange)
    }
}

fn rational(value: f64) -> Result<BigRational, StatisticsError> {
    finite(value)?;
    Ok(BigRational::new(
        float_to_subnormal_units(value),
        BigInt::one() << SUBNORMAL_EXPONENT,
    ))
}

fn ratio(numerator: &BigInt, denominator: &BigInt) -> Result<f64, StatisticsError> {
    rational_f64(&BigRational::new(numerator.clone(), denominator.clone()))
}

fn rational_f64(value: &BigRational) -> Result<f64, StatisticsError> {
    let projected = crate::binary64::ratio_to_f64(value.numer(), value.denom().magnitude())
        .ok_or(StatisticsError::NumericalRange)?;
    if !projected.is_finite() || (projected == 0.0 && !value.is_zero()) {
        return Err(StatisticsError::NumericalRange);
    }
    Ok(projected)
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::wire;
    use prost::Message;

    #[test]
    fn imported_summaries_are_checked_before_combination() {
        let estimate = StatisticalEstimate::from_summary(2.0, Some(0.5), 8u32.into()).unwrap();
        assert_eq!(estimate.standard_error(), Some(0.5));
        assert_eq!(estimate.variance(), Some(0.25));
        assert!(StatisticalEstimate::from_summary(0.0, Some(0.0), 1u32.into()).is_err());
        assert!(StatisticalEstimate::from_summary(0.0, None, 1u32.into()).is_ok());
        for error in [f64::NAN, f64::INFINITY, -1.0, f64::MAX, f64::MIN_POSITIVE] {
            assert!(StatisticalEstimate::from_summary(0.0, Some(error), 2u32.into()).is_err());
        }
        assert!(StatisticalEstimate::from_summary(f64::NAN, None, 2u32.into()).is_err());
        assert!(StatisticalEstimate::from_summary(0.0, None, BigUint::zero()).is_err());
    }

    fn bits_abi() -> Vec<u8> {
        wire::ResultAbiProto {
            r#type: Some(wire::TypeProto {
                kind: wire::type_proto::Kind::Array.into(),
                name: "Bits".to_owned(),
                arguments: vec![wire::TypeProto {
                    kind: wire::type_proto::Kind::Scalar.into(),
                    name: "Bit".to_owned(),
                    ..Default::default()
                }],
                dimensions: vec![wire::DimensionProto {
                    value: Some(wire::dimension_proto::Value::Fixed(2)),
                }],
                ..Default::default()
            }),
            source_order: vec![0, 1],
            native_width: 2,
        }
        .encode_to_vec()
    }

    fn histogram(count: BigInt) -> SampleCountSummary {
        crate::normalize_sample_counts(
            &[
                (vec![0, 0].into_boxed_slice(), count.clone()),
                (vec![1, 1].into_boxed_slice(), count),
            ],
            Some(2),
        )
        .unwrap()
    }

    #[test]
    fn paired_parities_keep_positive_and_negative_covariance() {
        let samples = histogram(2.into());
        let left = SampleReadout::parity(vec![], vec![0]).unwrap();
        let right = SampleReadout::parity(vec![], vec![1]).unwrap();
        let positive = sample_mean(
            &samples,
            &bits_abi(),
            &[(1.0, left.clone()), (1.0, right.clone())],
            3.0,
        )
        .unwrap();
        assert_eq!(positive.estimate().value(), 3.0);
        assert_eq!(positive.sample_variance(), Some(16.0 / 3.0));
        let negative =
            sample_mean(&samples, &bits_abi(), &[(1.0, left), (-1.0, right)], 3.0).unwrap();
        assert_eq!(negative.sample_variance(), Some(0.0));
    }

    #[test]
    fn invalid_readouts_and_numeric_budgets_fail_before_reduction() {
        assert_eq!(
            SampleReadout::parity(vec![], vec![0, 0]),
            Err(StatisticsError::DuplicateIndex)
        );
        assert_eq!(
            SampleReadout::value(vec![
                ScalarParameterPathComponent::Index(0);
                MAX_STATISTICS_PATH + 1
            ]),
            Err(StatisticsError::Budget)
        );
        let value = SampleReadout::value(vec![]).unwrap();
        assert!(matches!(
            sample_mean(&histogram(1.into()), &bits_abi(), &[(1.0, value)], 0.0),
            Err(StatisticsError::Shape(_))
        ));
        assert_eq!(
            sample_mean(
                &histogram(BigInt::one() << usize::try_from(MAX_STATISTICS_COUNT_BITS).unwrap()),
                &bits_abi(),
                &[],
                0.0
            ),
            Err(StatisticsError::Budget)
        );
    }

    #[test]
    fn independent_constant_has_no_fictitious_sample_variance() {
        let result = combine_independent_estimates(&[], 2.5).unwrap();
        assert_eq!(result.value(), 2.5);
        assert_eq!(result.standard_error(), Some(0.0));
        assert!(result.shots().is_zero());
    }
}
