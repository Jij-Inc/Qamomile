//! Explicit success-conditioned statistics and bounded phase-schedule inspection.

use super::signal_algebra::{MAX_SIGNAL_BRANCHES, SignalError};
use num_complex::Complex64;

/// Joint success and conditional parity estimates with simultaneous intervals.
#[derive(Clone, Debug)]
pub struct SignalEstimate {
    shots: u64,
    successes: u64,
    probability: f64,
    probability_interval: (f64, f64),
    conditional: Option<f64>,
    conditional_interval: (f64, f64),
}

impl SignalEstimate {
    /// Return all shots, including failures.
    #[must_use]
    pub const fn shots(&self) -> u64 {
        self.shots
    }
    /// Return observed successful shots.
    #[must_use]
    pub const fn successes(&self) -> u64 {
        self.successes
    }
    /// Return observed success probability.
    #[must_use]
    pub const fn probability(&self) -> f64 {
        self.probability
    }
    /// Return the Hoeffding success interval.
    #[must_use]
    pub const fn probability_interval(&self) -> (f64, f64) {
        self.probability_interval
    }
    /// Return conditional Z parity, absent when success was not observed.
    #[must_use]
    pub const fn conditional(&self) -> Option<f64> {
        self.conditional
    }
    /// Return the conditional Hoeffding interval, uninformative at zero success.
    #[must_use]
    pub const fn conditional_interval(&self) -> (f64, f64) {
        self.conditional_interval
    }
}

/// Reduce ordinary success-first result bits without silently discarding failures.
///
/// # Errors
/// Rejects malformed samples, indices, count overflow, or unbounded work.
#[allow(clippy::cast_precision_loss)]
pub fn signal_estimate(
    rows: &[(Vec<u8>, u64)],
    indices: &[usize],
    confidence: f64,
) -> Result<SignalEstimate, SignalError> {
    if rows.is_empty()
        || rows.len() > 4096
        || !confidence.is_finite()
        || !(0.0..1.0).contains(&confidence)
        || confidence == 0.0
    {
        return Err(SignalError::Budget);
    }
    let width = rows[0].0.len();
    if width == 0 || width > 4097 || rows.len().saturating_mul(width) > (1 << 20) {
        return Err(SignalError::Budget);
    }
    let mut unique = indices.to_vec();
    unique.sort_unstable();
    unique.dedup();
    if unique.len() != indices.len() || indices.iter().any(|i| *i >= width - 1) {
        return Err(SignalError::Space);
    }
    let mut shots = 0_u64;
    let mut successes = 0_u64;
    let mut negative = 0_u64;
    for (bits, count) in rows {
        if bits.len() != width || *count == 0 || bits.iter().any(|b| *b > 1) {
            return Err(SignalError::Space);
        }
        shots = shots
            .checked_add(*count)
            .filter(|n| *n <= (1 << 53))
            .ok_or(SignalError::Budget)?;
        if bits[0] == 1 {
            successes += count;
            if indices.iter().fold(0, |parity, i| parity ^ bits[i + 1]) == 1 {
                negative += count;
            }
        }
    }
    let probability = successes as f64 / shots as f64;
    let logarithm = (4.0 / (1.0 - confidence)).ln();
    let radius = (logarithm / (2.0 * shots as f64)).sqrt();
    let probability_interval = (
        (probability - radius).max(0.0),
        (probability + radius).min(1.0),
    );
    let (conditional, conditional_interval) = if successes == 0 {
        (None, (-1.0, 1.0))
    } else {
        let mean = 1.0 - 2.0 * negative as f64 / successes as f64;
        let radius = (2.0 * logarithm / successes as f64).sqrt();
        (
            Some(mean),
            ((mean - radius).max(-1.0), (mean + radius).min(1.0)),
        )
    };
    Ok(SignalEstimate {
        shots,
        successes,
        probability,
        probability_interval,
        conditional,
        conditional_interval,
    })
}

/// Check finite phase schedules before source generation or numerical inspection.
///
/// # Errors
/// Rejects nonfinite coordinates or more than 256 rounds.
pub fn signal_schedule(phases: &[(f64, f64)]) -> Result<Vec<(f64, f64)>, SignalError> {
    if phases.len() > MAX_SIGNAL_BRANCHES {
        return Err(SignalError::Budget);
    }
    if phases.iter().any(|(a, b)| !a.is_finite() || !b.is_finite()) {
        return Err(SignalError::Scalar);
    }
    Ok(phases.to_vec())
}

/// Evaluate the canonical two-dimensional amplitude response, not a uniform proof.
///
/// # Errors
/// Rejects invalid amplitudes or phase schedules.
pub fn signal_schedule_response(
    phases: &[(f64, f64)],
    amplitude: f64,
) -> Result<Complex64, SignalError> {
    signal_schedule(phases)?;
    if !amplitude.is_finite() || !(-1.0..=1.0).contains(&amplitude) {
        return Err(SignalError::Scalar);
    }
    let complement = (1.0 - amplitude * amplitude).max(0.0).sqrt();
    let mut a = Complex64::new(amplitude, 0.0);
    let mut b = Complex64::new(complement, 0.0);
    for &(input, output) in phases {
        a *= Complex64::from_polar(1.0, output);
        let left = amplitude * a + complement * b;
        let right = -complement * a + amplitude * b;
        let rotated = left * Complex64::from_polar(1.0, input);
        a = amplitude * rotated - complement * right;
        b = complement * rotated + amplitude * right;
    }
    Ok(a)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "The observed frequency 3/4 is exactly representable in binary64"
    )]
    fn estimates_retain_failure_counts_and_zero_success() {
        let e = signal_estimate(&[(vec![1, 1], 3), (vec![0, 0], 1)], &[0], 0.95).unwrap();
        assert_eq!(e.probability, 0.75);
        assert_eq!(e.conditional, Some(-1.0));
        let e = signal_estimate(&[(vec![0, 0], 4)], &[0], 0.95).unwrap();
        assert_eq!(e.conditional, None);
        assert_eq!(e.conditional_interval, (-1.0, 1.0));
    }
    #[test]
    fn schedules_distinguish_grover_and_fixed_point() {
        let pi = std::f64::consts::PI;
        assert!(
            (signal_schedule_response(&[(pi, pi)], 0.5)
                .unwrap()
                .norm_sqr()
                - 1.0)
                .abs()
                < 1e-14
        );
        assert!(
            (signal_schedule_response(&[(pi / 3.0, pi / 3.0)], 0.5)
                .unwrap()
                .norm_sqr()
                - 37.0 / 64.0)
                .abs()
                < 1e-14
        );
    }
}
