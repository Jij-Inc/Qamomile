//! Certified reciprocal singular-value transformations over a gapped spectrum.

use num_bigint::{BigInt, Sign};
use num_rational::Ratio;
use num_traits::{One, Signed, ToPrimitive, Zero};
use thiserror::Error;

use crate::qsvt::{
    CertifiedQsvtPolynomial, MAX_CERTIFIED_QSVT_DEGREE, QsvtPolynomialError,
    synthesize_qsvt_polynomial, verify_qsvt_polynomial,
};

type BigRational = Ratio<BigInt>;

/// Largest odd reciprocal-polynomial degree accepted by the producer.
pub const MAX_RECIPROCAL_QSVT_DEGREE: usize = MAX_CERTIFIED_QSVT_DEGREE - 1;
/// Maximum charged exact-rational coefficient-construction work per request.
pub const MAX_RECIPROCAL_COEFFICIENT_WORK: usize = 1 << 28;
const TARGET_CHEBYSHEV_L1_NUMERATOR: i64 = 1;
const TARGET_CHEBYSHEV_L1_DENOMINATOR: i64 = 4;

/// Failure to synthesize or replay a reciprocal singular-value transform.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum ReciprocalQsvtError {
    /// The block normalization or promised minimum singular value is invalid.
    #[error("Q7721: reciprocal SVT requires 0 < minimum_singular_value <= normalization")]
    InvalidDomain,
    /// The requested total uniform error is not finite and positive.
    #[error("Q7722: reciprocal SVT maximum error must be finite and positive")]
    InvalidTolerance,
    /// The requested degree bound cannot contain a positive odd polynomial.
    #[error("Q7723: reciprocal SVT degree must be between 1 and {MAX_RECIPROCAL_QSVT_DEGREE}")]
    DegreeBudget,
    /// No polynomial within the degree bound proves the requested approximation.
    #[error("Q7724: reciprocal SVT did not meet the requested error within its degree budget")]
    ApproximationBudget,
    /// Exact rational construction or binary64 projection exceeded its bounded domain.
    #[error("Q7725: reciprocal SVT coefficient construction exceeded its numeric budget")]
    NumericBudget,
    /// A reciprocal or nested polynomial certificate failed independent replay.
    #[error("Q7726: invalid reciprocal SVT certificate")]
    InvalidCertificate,
    /// The requested exact-rational construction exceeds its work budget.
    #[error(
        "Q7727: reciprocal SVT coefficient construction exceeds the {MAX_RECIPROCAL_COEFFICIENT_WORK}-unit work budget"
    )]
    WorkBudget,
    /// The bounded QSVT phase producer rejected the selected polynomial.
    #[error(transparent)]
    Polynomial(#[from] QsvtPolynomialError),
}

/// One independently replayable reciprocal singular-value transformation.
#[derive(Clone, Debug, PartialEq)]
pub struct CertifiedReciprocalQsvt {
    normalization: f64,
    minimum_singular_value: f64,
    normalized_gap: f64,
    reciprocal_scale: f64,
    operator_scale: f64,
    approximation_error_bound: f64,
    total_error_bound: f64,
    polynomial: CertifiedQsvtPolynomial,
}

impl CertifiedReciprocalQsvt {
    /// Return the exact block-encoding normalization supplied to synthesis.
    #[must_use]
    pub const fn normalization(&self) -> f64 {
        self.normalization
    }

    /// Return the promised physical minimum singular value.
    #[must_use]
    pub const fn minimum_singular_value(&self) -> f64 {
        self.minimum_singular_value
    }

    /// Return the normalized spectral gap `minimum_singular_value / normalization`.
    #[must_use]
    pub const fn normalized_gap(&self) -> f64 {
        self.normalized_gap
    }

    /// Return the numerator in the normalized target `reciprocal_scale / x`.
    #[must_use]
    pub const fn reciprocal_scale(&self) -> f64 {
        self.reciprocal_scale
    }

    /// Return the numerator multiplying physical reciprocal singular values.
    #[must_use]
    pub const fn operator_scale(&self) -> f64 {
        self.operator_scale
    }

    /// Return the polynomial-to-reciprocal uniform error over the gapped domain.
    #[must_use]
    pub const fn approximation_error_bound(&self) -> f64 {
        self.approximation_error_bound
    }

    /// Return the combined approximation and phase-realization error bound.
    #[must_use]
    pub const fn total_error_bound(&self) -> f64 {
        self.total_error_bound
    }

    /// Return the independently replayed QSVT polynomial realization.
    #[must_use]
    pub const fn polynomial(&self) -> &CertifiedQsvtPolynomial {
        &self.polynomial
    }
}

/// Synthesize a reciprocal transform for singular values outside a normalized gap.
///
/// The selected odd polynomial is the closed-form minimax approximation to
/// `1/x` on `[-1, -a] union [a, 1]`, scaled so its Chebyshev coefficient L1
/// norm is at most one quarter. Coefficients and approximation error are built
/// from exact rational arithmetic over the binary64 input values. The QSVT
/// phase candidate and the complete reciprocal claim are independently replayed
/// before returning.
///
/// # Errors
///
/// Returns [`ReciprocalQsvtError`] when the spectral promise, tolerance, degree,
/// coefficient construction, phase synthesis, or replayed certificate is invalid.
pub fn synthesize_reciprocal_qsvt(
    normalization: f64,
    minimum_singular_value: f64,
    max_error: f64,
    max_degree: usize,
    max_iterations: usize,
) -> Result<CertifiedReciprocalQsvt, ReciprocalQsvtError> {
    let normalized_gap =
        validate_request(normalization, minimum_singular_value, max_error, max_degree)?;
    let approximation_limit = max_error / 2.0;
    if !approximation_limit.is_finite() || approximation_limit <= 0.0 {
        return Err(ReciprocalQsvtError::InvalidTolerance);
    }
    let mut selected = None;
    let mut construction_work = 0_usize;
    for degree in (1..=max_degree).step_by(2) {
        construction_work = construction_work
            .checked_add(reciprocal_construction_work(normalized_gap, degree)?)
            .ok_or(ReciprocalQsvtError::WorkBudget)?;
        if construction_work > MAX_RECIPROCAL_COEFFICIENT_WORK {
            return Err(ReciprocalQsvtError::WorkBudget);
        }
        let candidate = reciprocal_candidate(normalized_gap, degree)?;
        if candidate.approximation_error_bound <= approximation_limit {
            selected = Some(candidate);
            break;
        }
    }
    let candidate = selected.ok_or(ReciprocalQsvtError::ApproximationBudget)?;
    let realization_limit = max_error - candidate.approximation_error_bound;
    if !realization_limit.is_finite() || realization_limit <= 0.0 {
        return Err(ReciprocalQsvtError::ApproximationBudget);
    }
    let polynomial = synthesize_qsvt_polynomial(
        &candidate.chebyshev_coefficients,
        realization_limit,
        max_iterations,
    )?;
    verify_reciprocal_qsvt(
        normalization,
        minimum_singular_value,
        candidate.reciprocal_scale,
        &candidate.chebyshev_coefficients,
        polynomial.qsp_phases(),
        max_error,
    )
}

/// Replay a reciprocal polynomial, phase realization, and combined error claim.
///
/// # Errors
///
/// Returns [`ReciprocalQsvtError`] when any coordinate differs from the
/// canonical exact-rational construction or the combined bound exceeds
/// `max_error`.
pub fn verify_reciprocal_qsvt(
    normalization: f64,
    minimum_singular_value: f64,
    reciprocal_scale: f64,
    chebyshev_coefficients: &[f64],
    qsp_phases: &[f64],
    max_error: f64,
) -> Result<CertifiedReciprocalQsvt, ReciprocalQsvtError> {
    let degree = chebyshev_coefficients
        .len()
        .checked_sub(1)
        .ok_or(ReciprocalQsvtError::InvalidCertificate)?;
    let normalized_gap =
        validate_request(normalization, minimum_singular_value, max_error, degree)?;
    if degree.is_multiple_of(2) {
        return Err(ReciprocalQsvtError::InvalidCertificate);
    }
    if reciprocal_construction_work(normalized_gap, degree)? > MAX_RECIPROCAL_COEFFICIENT_WORK {
        return Err(ReciprocalQsvtError::WorkBudget);
    }
    let canonical = reciprocal_candidate(normalized_gap, degree)?;
    if reciprocal_scale.to_bits() != canonical.reciprocal_scale.to_bits()
        || chebyshev_coefficients.len() != canonical.chebyshev_coefficients.len()
        || chebyshev_coefficients
            .iter()
            .zip(&canonical.chebyshev_coefficients)
            .any(|(left, right)| left.to_bits() != right.to_bits())
        || canonical.approximation_error_bound > max_error
    {
        return Err(ReciprocalQsvtError::InvalidCertificate);
    }
    let realization_limit = max_error - canonical.approximation_error_bound;
    let polynomial = verify_qsvt_polynomial(chebyshev_coefficients, qsp_phases, realization_limit)
        .map_err(|_| ReciprocalQsvtError::InvalidCertificate)?;
    let total_error_bound = upward_sum(
        canonical.approximation_error_bound,
        polynomial.uniform_error_bound(),
    )?;
    if total_error_bound > max_error {
        return Err(ReciprocalQsvtError::InvalidCertificate);
    }
    let operator_scale = reciprocal_scale * normalization;
    if !operator_scale.is_finite() {
        return Err(ReciprocalQsvtError::NumericBudget);
    }
    Ok(CertifiedReciprocalQsvt {
        normalization,
        minimum_singular_value,
        normalized_gap,
        reciprocal_scale,
        operator_scale,
        approximation_error_bound: canonical.approximation_error_bound,
        total_error_bound,
        polynomial,
    })
}

struct ReciprocalCandidate {
    reciprocal_scale: f64,
    chebyshev_coefficients: Vec<f64>,
    approximation_error_bound: f64,
}

fn validate_request(
    normalization: f64,
    minimum_singular_value: f64,
    max_error: f64,
    max_degree: usize,
) -> Result<f64, ReciprocalQsvtError> {
    if !normalization.is_finite()
        || !minimum_singular_value.is_finite()
        || normalization <= 0.0
        || minimum_singular_value <= 0.0
        || minimum_singular_value > normalization
    {
        return Err(ReciprocalQsvtError::InvalidDomain);
    }
    if !max_error.is_finite() || max_error <= 0.0 {
        return Err(ReciprocalQsvtError::InvalidTolerance);
    }
    if max_degree == 0 || max_degree > MAX_RECIPROCAL_QSVT_DEGREE {
        return Err(ReciprocalQsvtError::DegreeBudget);
    }
    let normalized_gap = minimum_singular_value / normalization;
    if !(normalized_gap.is_finite() && 0.0 < normalized_gap && normalized_gap <= 1.0) {
        return Err(ReciprocalQsvtError::InvalidDomain);
    }
    Ok(normalized_gap)
}

fn reciprocal_candidate(
    normalized_gap: f64,
    degree: usize,
) -> Result<ReciprocalCandidate, ReciprocalQsvtError> {
    if degree == 1 && normalized_gap.to_bits() == 1.0_f64.to_bits() {
        return Ok(ReciprocalCandidate {
            reciprocal_scale: 1.0,
            chebyshev_coefficients: vec![0.0, 1.0],
            approximation_error_bound: 0.0,
        });
    }
    let gap = rational_from_f64(normalized_gap)?;
    let unscaled = optimal_reciprocal_chebyshev(&gap, degree)?;
    let l1 = unscaled
        .iter()
        .fold(BigRational::zero(), |total, value| total + value.abs());
    if l1.is_zero() {
        return Err(ReciprocalQsvtError::NumericBudget);
    }
    let target_l1 = BigRational::new(
        BigInt::from(TARGET_CHEBYSHEV_L1_NUMERATOR),
        BigInt::from(TARGET_CHEBYSHEV_L1_DENOMINATOR),
    );
    let mut scale = rational_to_f64_down(&(target_l1 / l1))?;
    if scale <= 0.0 {
        return Err(ReciprocalQsvtError::NumericBudget);
    }
    loop {
        let scale_ratio = rational_from_f64(scale)?;
        let scaled = unscaled
            .iter()
            .map(|value| value * &scale_ratio)
            .collect::<Vec<_>>();
        let coefficients = scaled
            .iter()
            .map(rational_to_f64_nearest)
            .collect::<Result<Vec<_>, _>>()?;
        let projected_l1 = coefficients
            .iter()
            .try_fold(BigRational::zero(), |total, value| {
                Ok::<_, ReciprocalQsvtError>(total + rational_from_f64(value.abs())?)
            })?;
        if projected_l1 > BigRational::new(BigInt::one(), BigInt::from(4_u8)) {
            scale *= 0.5;
            if scale <= 0.0 || !scale.is_finite() {
                return Err(ReciprocalQsvtError::NumericBudget);
            }
            continue;
        }
        let rounding_error = coefficients.iter().zip(&scaled).try_fold(
            BigRational::zero(),
            |total, (projected, exact)| {
                Ok::<_, ReciprocalQsvtError>(total + (rational_from_f64(*projected)? - exact).abs())
            },
        )?;
        let n = degree.div_ceil(2);
        let ideal_error = (BigRational::one() - &gap)
            .pow(i32::try_from(n).map_err(|_| ReciprocalQsvtError::DegreeBudget)?)
            / (&gap
                * (BigRational::one() + &gap)
                    .pow(i32::try_from(n - 1).map_err(|_| ReciprocalQsvtError::DegreeBudget)?));
        let approximation_error = ideal_error * scale_ratio + rounding_error;
        return Ok(ReciprocalCandidate {
            reciprocal_scale: scale,
            chebyshev_coefficients: coefficients,
            approximation_error_bound: rational_to_f64_up(&approximation_error)?,
        });
    }
}

fn reciprocal_construction_work(
    normalized_gap: f64,
    degree: usize,
) -> Result<usize, ReciprocalQsvtError> {
    let gap = rational_from_f64(normalized_gap)?;
    let coordinate_bits = usize::try_from(
        gap.numer()
            .magnitude()
            .bits()
            .checked_add(gap.denom().magnitude().bits())
            .ok_or(ReciprocalQsvtError::WorkBudget)?,
    )
    .map_err(|_| ReciprocalQsvtError::WorkBudget)?;
    let extent = degree
        .checked_add(1)
        .ok_or(ReciprocalQsvtError::WorkBudget)?;
    coordinate_bits
        .checked_mul(extent)
        .and_then(|work| work.checked_mul(extent))
        .and_then(|work| work.checked_mul(extent))
        .ok_or(ReciprocalQsvtError::WorkBudget)
}

fn optimal_reciprocal_chebyshev(
    gap: &BigRational,
    degree: usize,
) -> Result<Vec<BigRational>, ReciprocalQsvtError> {
    if degree == 0 || degree.is_multiple_of(2) {
        return Err(ReciprocalQsvtError::DegreeBudget);
    }
    let n = degree.div_ceil(2);
    let one = BigRational::one();
    let two = BigRational::from_integer(BigInt::from(2_u8));
    let four = BigRational::from_integer(BigInt::from(4_u8));
    let alpha = (&one + gap) / (&two * (&one - gap));
    if alpha.is_zero() {
        return Err(ReciprocalQsvtError::NumericBudget);
    }
    let ratio = (&one - gap) / (&one + gap);
    let gap_squared = gap * gap;
    let z = vec![
        -(&one + &gap_squared) / (&one - &gap_squared),
        BigRational::zero(),
        &two / (&one - &gap_squared),
    ];
    let l_one = poly_scale(
        &poly_add(&z, std::slice::from_ref(&ratio)),
        &(BigRational::one() / &alpha),
    );
    let l_n = if n == 1 {
        l_one
    } else {
        let numerator = poly_add(
            &poly_add(&poly_multiply(&z, &z), &poly_scale(&z, &(&ratio / &two))),
            &[-BigRational::one() / &two],
        );
        let l_two = poly_scale(&numerator, &(BigRational::one() / (&alpha * &alpha)));
        if n == 2 {
            l_two
        } else {
            let mut previous = l_one;
            let mut current = l_two;
            for _ in 3..=n {
                let next = poly_subtract(
                    &poly_scale(&poly_multiply(&z, &current), &(BigRational::one() / &alpha)),
                    &poly_scale(&previous, &(BigRational::one() / (&four * &alpha * &alpha))),
                );
                previous = current;
                current = next;
            }
            current
        }
    };
    let mut prefactor = (&one + gap) * (&one + gap) / (&four * gap);
    if n % 2 == 1 {
        prefactor = -prefactor;
    }
    let numerator = poly_subtract(&[BigRational::one()], &poly_scale(&l_n, &prefactor));
    if !numerator.first().is_some_and(num_traits::Zero::is_zero)
        || numerator
            .iter()
            .enumerate()
            .any(|(power, value)| power % 2 == 1 && !value.is_zero())
    {
        return Err(ReciprocalQsvtError::NumericBudget);
    }
    let power = numerator.into_iter().skip(1).collect::<Vec<_>>();
    let chebyshev = power_to_chebyshev(&power);
    if chebyshev.len() != degree + 1
        || chebyshev
            .iter()
            .enumerate()
            .any(|(index, value)| index % 2 == 0 && !value.is_zero())
    {
        return Err(ReciprocalQsvtError::NumericBudget);
    }
    Ok(chebyshev)
}

fn poly_add(left: &[BigRational], right: &[BigRational]) -> Vec<BigRational> {
    let size = left.len().max(right.len());
    (0..size)
        .map(|index| {
            left.get(index).cloned().unwrap_or_else(BigRational::zero)
                + right.get(index).cloned().unwrap_or_else(BigRational::zero)
        })
        .collect()
}

fn poly_subtract(left: &[BigRational], right: &[BigRational]) -> Vec<BigRational> {
    poly_add(left, &poly_scale(right, &-BigRational::one()))
}

fn poly_scale(values: &[BigRational], scale: &BigRational) -> Vec<BigRational> {
    values.iter().map(|value| value * scale).collect()
}

fn poly_multiply(left: &[BigRational], right: &[BigRational]) -> Vec<BigRational> {
    let mut result = vec![BigRational::zero(); left.len() + right.len() - 1];
    for (left_index, left_value) in left.iter().enumerate() {
        for (right_index, right_value) in right.iter().enumerate() {
            result[left_index + right_index] += left_value * right_value;
        }
    }
    result
}

fn power_to_chebyshev(power: &[BigRational]) -> Vec<BigRational> {
    let mut result = vec![BigRational::zero(); power.len()];
    let mut basis = vec![BigRational::one()];
    for coefficient in power {
        for (index, value) in basis.iter().enumerate() {
            result[index] += coefficient * value;
        }
        basis = chebyshev_multiply_x(&basis);
    }
    result
}

fn chebyshev_multiply_x(values: &[BigRational]) -> Vec<BigRational> {
    let mut result = vec![BigRational::zero(); values.len() + 1];
    for (index, value) in values.iter().enumerate() {
        if index == 0 {
            result[1] += value;
        } else {
            result[index - 1] += value / BigInt::from(2_u8);
            result[index + 1] += value / BigInt::from(2_u8);
        }
    }
    result
}

fn rational_from_f64(value: f64) -> Result<BigRational, ReciprocalQsvtError> {
    if !value.is_finite() {
        return Err(ReciprocalQsvtError::NumericBudget);
    }
    if value == 0.0 {
        return Ok(BigRational::zero());
    }
    let bits = value.to_bits();
    let sign = if bits >> 63 == 0 {
        Sign::Plus
    } else {
        Sign::Minus
    };
    let exponent_bits = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1_u64 << 52) - 1);
    let (significand, exponent) = if exponent_bits == 0 {
        (fraction, -1074)
    } else {
        ((1_u64 << 52) | fraction, exponent_bits - 1023 - 52)
    };
    let numerator = BigInt::from_biguint(sign, significand.into());
    if exponent >= 0 {
        Ok(BigRational::from_integer(
            numerator
                << usize::try_from(exponent).map_err(|_| ReciprocalQsvtError::NumericBudget)?,
        ))
    } else {
        Ok(BigRational::new(
            numerator,
            BigInt::one()
                << usize::try_from(-exponent).map_err(|_| ReciprocalQsvtError::NumericBudget)?,
        ))
    }
}

// Both integer heads contain at most 53 significant bits, so these casts are exact.
#[allow(clippy::cast_precision_loss)]
fn rational_to_f64_nearest(value: &BigRational) -> Result<f64, ReciprocalQsvtError> {
    if value.is_zero() {
        return Ok(0.0);
    }
    let numerator = value.numer().magnitude();
    let denominator = value.denom().magnitude();
    let numerator_shift = usize::try_from(numerator.bits())
        .map_err(|_| ReciprocalQsvtError::NumericBudget)?
        .saturating_sub(53);
    let denominator_shift = usize::try_from(denominator.bits())
        .map_err(|_| ReciprocalQsvtError::NumericBudget)?
        .saturating_sub(53);
    let numerator_head = (numerator >> numerator_shift)
        .to_u64()
        .ok_or(ReciprocalQsvtError::NumericBudget)? as f64;
    let denominator_head = (denominator >> denominator_shift)
        .to_u64()
        .ok_or(ReciprocalQsvtError::NumericBudget)? as f64;
    let exponent = i32::try_from(numerator_shift)
        .ok()
        .and_then(|left| {
            i32::try_from(denominator_shift)
                .ok()
                .and_then(|right| left.checked_sub(right))
        })
        .ok_or(ReciprocalQsvtError::NumericBudget)?;
    let magnitude = numerator_head / denominator_head * 2.0_f64.powi(exponent);
    let projected = if value.is_negative() {
        -magnitude
    } else {
        magnitude
    };
    projected
        .is_finite()
        .then_some(projected)
        .ok_or(ReciprocalQsvtError::NumericBudget)
}

fn rational_to_f64_down(value: &BigRational) -> Result<f64, ReciprocalQsvtError> {
    let mut projected = rational_to_f64_nearest(value)?;
    for _ in 0..8 {
        if rational_from_f64(projected)? > *value {
            projected = next_down(projected);
            continue;
        }
        let next = next_up(projected);
        if next.is_finite() && rational_from_f64(next)? <= *value {
            projected = next;
            continue;
        }
        return Ok(projected);
    }
    Err(ReciprocalQsvtError::NumericBudget)
}

fn rational_to_f64_up(value: &BigRational) -> Result<f64, ReciprocalQsvtError> {
    let mut projected = rational_to_f64_nearest(value)?;
    for _ in 0..8 {
        if rational_from_f64(projected)? < *value {
            projected = next_up(projected);
            continue;
        }
        let previous = next_down(projected);
        if previous.is_finite() && rational_from_f64(previous)? >= *value {
            projected = previous;
            continue;
        }
        return Ok(projected);
    }
    Err(ReciprocalQsvtError::NumericBudget)
}

fn next_down(value: f64) -> f64 {
    if value == 0.0 {
        -f64::from_bits(1)
    } else if value > 0.0 {
        f64::from_bits(value.to_bits() - 1)
    } else {
        f64::from_bits(value.to_bits() + 1)
    }
}

fn next_up(value: f64) -> f64 {
    if value == 0.0 {
        f64::from_bits(1)
    } else if value > 0.0 {
        f64::from_bits(value.to_bits() + 1)
    } else {
        f64::from_bits(value.to_bits() - 1)
    }
}

fn upward_sum(left: f64, right: f64) -> Result<f64, ReciprocalQsvtError> {
    let exact = rational_from_f64(left)? + rational_from_f64(right)?;
    rational_to_f64_up(&exact)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_unit_gap_reuses_the_linear_qsvt_transform() {
        let certificate = synthesize_reciprocal_qsvt(2.0, 2.0, 1e-12, 15, 8).expect("unit gap");

        assert_eq!(certificate.normalized_gap().to_bits(), 1.0_f64.to_bits());
        assert_eq!(certificate.reciprocal_scale().to_bits(), 1.0_f64.to_bits());
        assert_eq!(certificate.operator_scale().to_bits(), 2.0_f64.to_bits());
        assert_eq!(
            certificate.approximation_error_bound().to_bits(),
            0.0_f64.to_bits()
        );
        assert_eq!(certificate.total_error_bound().to_bits(), 0.0_f64.to_bits());
        assert_eq!(certificate.polynomial().degree(), 1);
    }

    #[test]
    fn synthesizes_and_replays_a_nontrivial_gapped_reciprocal() {
        let certificate = synthesize_reciprocal_qsvt(1.0, 0.8, 0.01, 15, 16).expect("reciprocal");

        assert_eq!(certificate.polynomial().degree(), 3);
        assert!(certificate.reciprocal_scale() > 0.0);
        assert!(certificate.approximation_error_bound() < 0.005);
        assert!(certificate.total_error_bound() <= 0.01);
        assert_eq!(
            verify_reciprocal_qsvt(
                1.0,
                0.8,
                certificate.reciprocal_scale(),
                certificate.polynomial().chebyshev_coefficients(),
                certificate.polynomial().qsp_phases(),
                0.01,
            ),
            Ok(certificate)
        );
    }

    #[test]
    fn reciprocal_requests_and_certificates_fail_closed() {
        assert_eq!(
            synthesize_reciprocal_qsvt(1.0, 0.0, 0.01, 15, 16),
            Err(ReciprocalQsvtError::InvalidDomain)
        );
        assert_eq!(
            synthesize_reciprocal_qsvt(1.0, 0.8, 1e-12, 1, 16),
            Err(ReciprocalQsvtError::ApproximationBudget)
        );
        let certificate = synthesize_reciprocal_qsvt(1.0, 0.8, 0.01, 15, 16).expect("reciprocal");
        let mut corrupted = certificate.polynomial().chebyshev_coefficients().to_vec();
        corrupted[1] += 0.01;
        assert_eq!(
            verify_reciprocal_qsvt(
                1.0,
                0.8,
                certificate.reciprocal_scale(),
                &corrupted,
                certificate.polynomial().qsp_phases(),
                0.01,
            ),
            Err(ReciprocalQsvtError::InvalidCertificate)
        );

        let cumulative_work = (1..=MAX_RECIPROCAL_QSVT_DEGREE)
            .step_by(2)
            .try_fold(0_usize, |work, degree| {
                work.checked_add(reciprocal_construction_work(0.125, degree).ok()?)
            })
            .expect("bounded test arithmetic");
        assert!(cumulative_work > MAX_RECIPROCAL_COEFFICIENT_WORK);
    }
}
