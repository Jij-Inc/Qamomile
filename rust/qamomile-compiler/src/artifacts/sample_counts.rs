//! Canonical logical sample histograms shared by every execution endpoint.

use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{Signed, ToPrimitive, Zero};
use thiserror::Error;

/// Largest number of distinct rows accepted in one logical histogram.
pub const MAX_SAMPLE_COUNT_ROWS: usize = 1_000_000;
/// Largest logical row width accepted by histogram normalization.
pub const MAX_SAMPLE_COUNT_WIDTH: usize = 1_000_000;
/// Largest total number of logical bit cells accepted in one histogram.
pub const MAX_SAMPLE_COUNT_CELLS: usize = 10_000_000;
/// Largest arbitrary-precision count accepted per row or total.
pub const MAX_SAMPLE_COUNT_BITS: u64 = 1_000_000;

/// Failure to normalize one logical sample histogram.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum SampleCountError {
    /// A logical row contains a value outside the bit domain.
    #[error("Q8111: sample keys must contain only 0 or 1")]
    InvalidBit,
    /// Logical rows do not have the declared uniform width.
    #[error("Q8113: sampling counts disagree with the logical result width")]
    WidthMismatch,
    /// Two input rows denote the same logical sample.
    #[error("Q8116: sampling count keys must be unique")]
    DuplicateRow,
    /// A row count is negative.
    #[error("Q8107: sampling counts cannot be negative")]
    NegativeCount,
    /// No positive observation is present.
    #[error("Q8106: sampling counts must contain observations")]
    NoObservations,
    /// The histogram has too many distinct rows.
    #[error("Q8125: sampling counts exceed the {MAX_SAMPLE_COUNT_ROWS}-row budget")]
    RowBudget,
    /// A logical row is wider than the supported bound.
    #[error("Q8126: sampling keys exceed the {MAX_SAMPLE_COUNT_WIDTH}-bit width budget")]
    WidthBudget,
    /// A count or total is too large for bounded processing.
    #[error("Q8127: sampling counts exceed the {MAX_SAMPLE_COUNT_BITS}-bit numeric budget")]
    CountBudget,
    /// The complete histogram exceeds its bit-cell work budget.
    #[error("Q8129: sampling counts exceed the {MAX_SAMPLE_COUNT_CELLS}-cell work budget")]
    CellBudget,
}

/// One canonical logical histogram and its shared summary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SampleCountSummary {
    entries: Vec<(Box<[u8]>, BigUint)>,
    shots: BigUint,
    most_likely: Box<[u8]>,
}

impl SampleCountSummary {
    /// Return lexicographically ordered logical rows and exact counts.
    #[must_use]
    pub fn entries(&self) -> &[(Box<[u8]>, BigUint)] {
        &self.entries
    }

    /// Return the exact total observation count.
    #[must_use]
    pub const fn shots(&self) -> &BigUint {
        &self.shots
    }

    /// Return the most frequent row, breaking ties lexicographically upward.
    #[must_use]
    pub fn most_likely(&self) -> &[u8] {
        &self.most_likely
    }

    /// Return the empirical probability of one logical row.
    #[must_use]
    pub fn probability(&self, bits: &[u8]) -> Option<f64> {
        if bits.len() != self.most_likely.len() || bits.iter().any(|bit| *bit > 1) {
            return None;
        }
        let count = self
            .entries
            .binary_search_by(|(row, _)| row.as_ref().cmp(bits))
            .ok()
            .map_or(&BigUint::ZERO, |index| &self.entries[index].1);
        biguint_ratio(count, &self.shots)
    }
}

/// Normalize one logical histogram and derive its reusable summary.
///
/// # Errors
///
/// Returns [`SampleCountError`] when rows are malformed, inconsistent,
/// duplicated, empty, negative, or outside an explicit processing budget.
pub fn normalize_sample_counts(
    rows: &[(Box<[u8]>, BigInt)],
    expected_width: Option<usize>,
) -> Result<SampleCountSummary, SampleCountError> {
    if rows.len() > MAX_SAMPLE_COUNT_ROWS {
        return Err(SampleCountError::RowBudget);
    }
    if rows.is_empty() {
        return Err(SampleCountError::NoObservations);
    }
    let width = rows[0].0.len();
    if width > MAX_SAMPLE_COUNT_WIDTH
        || expected_width.is_some_and(|expected| expected > MAX_SAMPLE_COUNT_WIDTH)
    {
        return Err(SampleCountError::WidthBudget);
    }
    if rows.iter().any(|(bits, _)| {
        bits.len() != width || expected_width.is_some_and(|expected| bits.len() != expected)
    }) {
        return Err(SampleCountError::WidthMismatch);
    }
    if width
        .checked_mul(rows.len())
        .is_none_or(|cells| cells > MAX_SAMPLE_COUNT_CELLS)
    {
        return Err(SampleCountError::CellBudget);
    }
    if rows
        .iter()
        .flat_map(|(bits, _)| bits.iter())
        .any(|bit| *bit > 1)
    {
        return Err(SampleCountError::InvalidBit);
    }

    let mut entries = Vec::with_capacity(rows.len());
    for (bits, count) in rows {
        if count.is_negative() {
            return Err(SampleCountError::NegativeCount);
        }
        let count = count.to_biguint().ok_or(SampleCountError::NegativeCount)?;
        if count.bits() > MAX_SAMPLE_COUNT_BITS {
            return Err(SampleCountError::CountBudget);
        }
        entries.push((bits.clone(), count));
    }
    entries.sort_by(|left, right| left.0.cmp(&right.0));
    if entries.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return Err(SampleCountError::DuplicateRow);
    }

    let mut shots = BigUint::ZERO;
    for (_, count) in &entries {
        shots += count;
        if shots.bits() > MAX_SAMPLE_COUNT_BITS {
            return Err(SampleCountError::CountBudget);
        }
    }
    if shots.is_zero() {
        return Err(SampleCountError::NoObservations);
    }
    let most_likely = entries
        .iter()
        .max_by(|left, right| left.1.cmp(&right.1).then_with(|| left.0.cmp(&right.0)))
        .ok_or(SampleCountError::NoObservations)?
        .0
        .clone();
    Ok(SampleCountSummary {
        entries,
        shots,
        most_likely,
    })
}

pub(crate) fn biguint_ratio(numerator: &BigUint, denominator: &BigUint) -> Option<f64> {
    if denominator.is_zero() {
        return None;
    }
    if numerator.is_zero() {
        return Some(0.0);
    }
    let (numerator_mantissa, numerator_shift) = scaled_mantissa(numerator)?;
    let (denominator_mantissa, denominator_shift) = scaled_mantissa(denominator)?;
    let exponent = numerator_shift.checked_sub(denominator_shift)?;
    Some(((numerator_mantissa / denominator_mantissa) * 2_f64.powi(exponent)).clamp(0.0, 1.0))
}

fn scaled_mantissa(value: &BigUint) -> Option<(f64, i32)> {
    let shift = value.bits().saturating_sub(64);
    let shift_usize = usize::try_from(shift).ok()?;
    let mantissa = (value >> shift_usize).to_f64()?;
    Some((mantissa, i32::try_from(shift).ok()?))
}

pub(crate) fn bigint_ratio(numerator: &BigInt, denominator: &BigUint) -> Option<f64> {
    let magnitude = biguint_ratio(numerator.magnitude(), denominator)?;
    Some(if numerator.sign() == Sign::Minus {
        -magnitude
    } else {
        magnitude
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(bits: &[u8], count: BigInt) -> (Box<[u8]>, BigInt) {
        (bits.into(), count)
    }

    #[test]
    fn normalizes_arbitrary_precision_counts_and_tie_breaking() {
        let large: BigInt = BigInt::from(1_u8) << 100_usize;
        let summary = normalize_sample_counts(
            &[
                row(&[1, 0], large.clone()),
                row(&[0, 1], large),
                row(&[0, 0], BigInt::ZERO),
            ],
            Some(2),
        )
        .expect("sample summary");

        assert_eq!(summary.entries()[0].0.as_ref(), &[0, 0]);
        assert_eq!(summary.shots(), &(BigUint::from(1_u8) << 101));
        assert_eq!(summary.most_likely(), &[1, 0]);
        assert_eq!(summary.probability(&[0, 1]), Some(0.5));
        assert_eq!(summary.probability(&[1, 1]), Some(0.0));
    }

    #[test]
    fn rejects_malformed_histograms() {
        assert_eq!(
            normalize_sample_counts(&[row(&[0], BigInt::from(-1))], Some(1)),
            Err(SampleCountError::NegativeCount)
        );
        assert_eq!(
            normalize_sample_counts(
                &[row(&[0], BigInt::from(1)), row(&[0], BigInt::from(2))],
                Some(1),
            ),
            Err(SampleCountError::DuplicateRow)
        );
        assert_eq!(
            normalize_sample_counts(&[row(&[2], BigInt::from(1))], Some(1)),
            Err(SampleCountError::InvalidBit)
        );
        assert_eq!(
            normalize_sample_counts(&[row(&[0], BigInt::ZERO)], Some(1)),
            Err(SampleCountError::NoObservations)
        );
    }

    #[test]
    fn probability_scales_extreme_counts_without_binary64_overflow() {
        let huge: BigInt = BigInt::from(1_u8) << 100_000_usize;
        let tiny = BigInt::from(1_u8);
        let summary = normalize_sample_counts(&[row(&[0], huge), row(&[1], tiny)], Some(1))
            .expect("extreme sample summary");

        assert_eq!(summary.probability(&[0]), Some(1.0));
        assert_eq!(summary.probability(&[1]), Some(0.0));
        assert_eq!(summary.probability(&[0, 1]), None);
    }
}
