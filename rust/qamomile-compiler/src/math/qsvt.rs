//! Bounded phase synthesis and coordinates for quantum singular value transforms.

use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI, TAU};

use num_complex::Complex64;
use thiserror::Error;

use crate::binary64::exact_sum;

/// Maximum phase count accepted by one convention transform.
pub const MAX_QSVT_PHASES: usize = 1 << 20;
/// Maximum degree accepted by the certified symmetric phase solver.
pub const MAX_CERTIFIED_QSVT_DEGREE: usize = 256;
/// Maximum damped-Newton iterations accepted by one synthesis request.
pub const MAX_QSVT_SYNTHESIS_ITERATIONS: usize = 4096;
/// Maximum charged scalar/matrix work in one synthesis request.
pub const MAX_QSVT_SYNTHESIS_WORK: usize = 1 << 29;
const MAX_NEWTON_TARGET_L1: f64 = 0.25;
const MAX_NEWTON_BACKTRACKS: usize = 24;
const QSVT_NUMERICAL_MARGIN_FACTOR: f64 = 64.0;

type Matrix2 = [[Complex64; 2]; 2];

/// Failure to convert one QSP phase sequence.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum QsvtPhaseError {
    /// A phase convention requires at least one coordinate.
    #[error("Q7621: QSVT requires at least one phase")]
    Empty,
    /// Phase materialization exceeds the explicit coordinate budget.
    #[error("Q7622: QSVT phase count exceeds the {MAX_QSVT_PHASES}-entry budget")]
    Budget,
    /// A source phase is not finite.
    #[error("Q7623: QSVT phases must be finite")]
    NonFinite,
    /// Offset or wrapping arithmetic left the finite binary64 domain.
    #[error("Q7624: QSVT phase conversion exceeds the finite binary64 range")]
    CoordinateRange,
}

/// Failure to synthesize or replay a certified real QSVT polynomial.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum QsvtPolynomialError {
    /// Chebyshev coordinates are empty, non-finite, or non-canonical.
    #[error("Q7701: QSVT Chebyshev coefficients must be finite and canonical")]
    InvalidCoefficients,
    /// Nonzero coordinates disagree with the polynomial degree parity.
    #[error("Q7702: QSVT Chebyshev coefficients must have the degree parity")]
    InvalidParity,
    /// The requested target is outside the solver's proved convergence domain.
    #[error("Q7703: QSVT synthesis target is outside the proved bounded domain")]
    UnsupportedTarget,
    /// The requested uniform error is not finite and positive.
    #[error("Q7704: QSVT maximum error must be finite and positive")]
    InvalidTolerance,
    /// The target degree exceeds the explicit synthesis budget.
    #[error("Q7705: QSVT synthesis degree exceeds the {MAX_CERTIFIED_QSVT_DEGREE}-degree budget")]
    DegreeBudget,
    /// The requested iteration count is zero or excessive.
    #[error(
        "Q7706: QSVT synthesis iterations must be between 1 and {MAX_QSVT_SYNTHESIS_ITERATIONS}"
    )]
    IterationBudget,
    /// The bounded quadratic solver would exceed its work budget.
    #[error("Q7707: QSVT synthesis exceeds its {MAX_QSVT_SYNTHESIS_WORK}-unit work budget")]
    WorkBudget,
    /// Binary64 synthesis or certificate replay left its finite domain.
    #[error("Q7708: QSVT synthesis exceeds the finite binary64 range")]
    NumericalRange,
    /// A candidate is malformed or does not meet its claimed error bound.
    #[error("Q7709: invalid QSVT polynomial certificate")]
    InvalidCertificate,
    /// The bounded solver did not produce a certificate within the requested error.
    #[error("Q7710: QSVT phase synthesis did not converge within its bounded work")]
    DidNotConverge,
    /// Projector-coordinate conversion failed after synthesis.
    #[error(transparent)]
    Phase(#[from] QsvtPhaseError),
}

/// One replayable real-polynomial QSVT certificate.
#[derive(Clone, Debug, PartialEq)]
pub struct CertifiedQsvtPolynomial {
    chebyshev_coefficients: Vec<f64>,
    qsp_phases: Vec<f64>,
    projector_phases: Vec<f64>,
    uniform_error_bound: f64,
    target_magnitude_bound: f64,
    realized_magnitude_bound: f64,
    heralded: bool,
}

impl CertifiedQsvtPolynomial {
    /// Return canonical first-kind Chebyshev coefficients in ascending order.
    #[must_use]
    pub fn chebyshev_coefficients(&self) -> &[f64] {
        &self.chebyshev_coefficients
    }

    /// Return symmetric canonical QSP `Wx` phases used by certificate replay.
    #[must_use]
    pub fn qsp_phases(&self) -> &[f64] {
        &self.qsp_phases
    }

    /// Return projector-convention phases consumed by Qamomile's QSVT body.
    #[must_use]
    pub fn projector_phases(&self) -> &[f64] {
        &self.projector_phases
    }

    /// Return the polynomial degree and block-encoding query count.
    #[must_use]
    pub fn degree(&self) -> usize {
        self.chebyshev_coefficients.len() - 1
    }

    /// Return whether the certified target polynomial is odd.
    #[must_use]
    pub fn is_odd(&self) -> bool {
        self.degree() % 2 == 1
    }

    /// Return the proved uniform target-to-realization error over `[-1, 1]`.
    #[must_use]
    pub const fn uniform_error_bound(&self) -> f64 {
        self.uniform_error_bound
    }

    /// Return the coefficient-L1 target magnitude bound over `[-1, 1]`.
    #[must_use]
    pub const fn target_magnitude_bound(&self) -> f64 {
        self.target_magnitude_bound
    }

    /// Return the realized block magnitude bound over `[-1, 1]`.
    #[must_use]
    pub const fn realized_magnitude_bound(&self) -> f64 {
        self.realized_magnitude_bound
    }

    /// Return whether observing the signal subspace heralds the transformed block.
    #[must_use]
    pub const fn is_heralded(&self) -> bool {
        self.heralded
    }
}

/// Convert QSP `Wx` phases into canonical projector-convention coordinates.
///
/// The endpoint corrections follow the Qamomile projector convention. Every
/// result is reduced to the half-open interval `[-pi, pi)`. A degree-zero
/// sequence receives both endpoint corrections because its sole coordinate is
/// simultaneously the first and last phase.
///
/// # Errors
///
/// Returns [`QsvtPhaseError`] when the input is empty, non-finite, excessive,
/// or cannot be shifted and wrapped within finite binary64 arithmetic.
pub fn convert_qsp_wx_phases(phases: &[f64]) -> Result<Vec<f64>, QsvtPhaseError> {
    if phases.is_empty() {
        return Err(QsvtPhaseError::Empty);
    }
    if phases.len() > MAX_QSVT_PHASES {
        return Err(QsvtPhaseError::Budget);
    }
    if phases.iter().any(|phase| !phase.is_finite()) {
        return Err(QsvtPhaseError::NonFinite);
    }
    let degree = phases.len() - 1;
    let degree_factor = if degree == 0 {
        -1.0
    } else {
        let factor = degree
            .checked_mul(2)
            .and_then(|value| value.checked_sub(1))
            .and_then(|value| u32::try_from(value).ok())
            .ok_or(QsvtPhaseError::Budget)?;
        f64::from(factor)
    };
    let mut converted = phases.to_vec();
    converted[0] += degree_factor * FRAC_PI_4;
    for phase in converted.iter_mut().take(degree).skip(1) {
        *phase -= FRAC_PI_2;
    }
    converted[degree] -= FRAC_PI_4;
    for phase in &mut converted {
        if !phase.is_finite() {
            return Err(QsvtPhaseError::CoordinateRange);
        }
        *phase = (*phase + PI).rem_euclid(TAU) - PI;
        if !phase.is_finite() {
            return Err(QsvtPhaseError::CoordinateRange);
        }
    }
    Ok(converted)
}

/// Synthesize symmetric phases for one bounded real Chebyshev polynomial.
///
/// Coefficients are ordered from `T_0` through `T_d`. A single `T_d` term has
/// a direct analytic realization. Other targets whose coefficient L1 norm is
/// at most one quarter use a bounded damped-Newton solve over symmetric QSP
/// coordinates. The resulting candidate is independently replayed into a
/// coefficient-L1 uniform error certificate before it is returned.
///
/// # Errors
///
/// Returns [`QsvtPolynomialError`] when the polynomial, tolerance, or bounded
/// work request is invalid, or when the solver cannot establish the requested
/// uniform error within `max_iterations`.
pub fn synthesize_qsvt_polynomial(
    chebyshev_coefficients: &[f64],
    max_error: f64,
    max_iterations: usize,
) -> Result<CertifiedQsvtPolynomial, QsvtPolynomialError> {
    let (degree, target_magnitude_bound) = validate_chebyshev_coefficients(chebyshev_coefficients)?;
    validate_tolerance(max_error)?;
    validate_synthesis_iterations(max_iterations)?;

    if let Some(coefficient) = single_leading_coefficient(chebyshev_coefficients)
        && coefficient.abs() <= 1.0
    {
        let phase = canonical_qsp_phase(coefficient.acos());
        if !phase.is_finite() {
            return Err(QsvtPolynomialError::NumericalRange);
        }
        let mut qsp_phases = vec![0.0; degree + 1];
        if degree == 0 {
            qsp_phases[0] = phase;
        } else {
            qsp_phases[0] = phase / 2.0;
            qsp_phases[degree] = phase / 2.0;
        }
        return verify_qsvt_polynomial(chebyshev_coefficients, &qsp_phases, max_error).map_err(
            |error| match error {
                QsvtPolynomialError::InvalidCertificate => QsvtPolynomialError::DidNotConverge,
                other => other,
            },
        );
    }

    validate_synthesis_work(degree, max_iterations)?;
    if degree == 0 || target_magnitude_bound > MAX_NEWTON_TARGET_L1 {
        return Err(QsvtPolynomialError::UnsupportedTarget);
    }
    solve_symmetric_phases(chebyshev_coefficients, max_error, max_iterations)
}

/// Replay a symmetric QSP candidate into a bounded polynomial certificate.
///
/// The target and realized polynomial are compared at a complete Chebyshev
/// interpolation grid. The reconstructed difference coefficients give a
/// uniform bound because `|T_j(x)| <= 1` on `[-1, 1]`; a conservative
/// binary64 margin covers the bounded matrix products and DCT accumulation.
///
/// # Errors
///
/// Returns [`QsvtPolynomialError`] when coefficients or phases are malformed,
/// or the replayed uniform error exceeds `max_error`.
pub fn verify_qsvt_polynomial(
    chebyshev_coefficients: &[f64],
    qsp_phases: &[f64],
    max_error: f64,
) -> Result<CertifiedQsvtPolynomial, QsvtPolynomialError> {
    let (degree, target_magnitude_bound) = validate_chebyshev_coefficients(chebyshev_coefficients)?;
    validate_tolerance(max_error)?;
    if qsp_phases.len() != degree + 1
        || qsp_phases.iter().any(|phase| !phase.is_finite())
        || qsp_phases.iter().any(|phase| !(-PI..PI).contains(phase))
        || qsp_phases
            .iter()
            .zip(qsp_phases.iter().rev())
            .any(|(left, right)| left.to_bits() != right.to_bits())
    {
        return Err(QsvtPolynomialError::InvalidCertificate);
    }
    let uniform_error_bound = certified_uniform_error(chebyshev_coefficients, qsp_phases)?;
    if uniform_error_bound > max_error {
        return Err(QsvtPolynomialError::InvalidCertificate);
    }
    let projector_phases = convert_qsp_wx_phases(qsp_phases)?;
    let realized_magnitude_bound = (target_magnitude_bound + uniform_error_bound).min(1.0);
    if !realized_magnitude_bound.is_finite() {
        return Err(QsvtPolynomialError::NumericalRange);
    }
    Ok(CertifiedQsvtPolynomial {
        chebyshev_coefficients: chebyshev_coefficients.to_vec(),
        qsp_phases: qsp_phases.to_vec(),
        projector_phases,
        uniform_error_bound,
        target_magnitude_bound,
        realized_magnitude_bound,
        heralded: true,
    })
}

fn validate_chebyshev_coefficients(
    coefficients: &[f64],
) -> Result<(usize, f64), QsvtPolynomialError> {
    if coefficients.is_empty()
        || coefficients
            .iter()
            .any(|coefficient| !coefficient.is_finite())
        || (coefficients.len() > 1 && coefficients.last() == Some(&0.0))
    {
        return Err(QsvtPolynomialError::InvalidCoefficients);
    }
    let degree = coefficients.len() - 1;
    if degree > MAX_CERTIFIED_QSVT_DEGREE {
        return Err(QsvtPolynomialError::DegreeBudget);
    }
    if coefficients
        .iter()
        .enumerate()
        .any(|(index, coefficient)| index % 2 != degree % 2 && *coefficient != 0.0)
    {
        return Err(QsvtPolynomialError::InvalidParity);
    }
    let target_magnitude_bound = exact_sum(coefficients.iter().map(|value| value.abs()))
        .ok_or(QsvtPolynomialError::NumericalRange)?;
    Ok((degree, target_magnitude_bound))
}

fn validate_tolerance(max_error: f64) -> Result<(), QsvtPolynomialError> {
    if !max_error.is_finite() || max_error <= 0.0 {
        return Err(QsvtPolynomialError::InvalidTolerance);
    }
    Ok(())
}

fn validate_synthesis_iterations(max_iterations: usize) -> Result<(), QsvtPolynomialError> {
    if max_iterations == 0 || max_iterations > MAX_QSVT_SYNTHESIS_ITERATIONS {
        return Err(QsvtPolynomialError::IterationBudget);
    }
    Ok(())
}

fn validate_synthesis_work(
    degree: usize,
    max_iterations: usize,
) -> Result<(), QsvtPolynomialError> {
    let extent = degree
        .checked_add(2)
        .ok_or(QsvtPolynomialError::WorkBudget)?;
    let reduced = degree.midpoint(2);
    let cubic = reduced
        .checked_mul(reduced)
        .and_then(|value| value.checked_mul(reduced))
        .ok_or(QsvtPolynomialError::WorkBudget)?;
    let trial_work = extent
        .checked_mul(reduced)
        .and_then(|value| value.checked_mul(MAX_NEWTON_BACKTRACKS + 2))
        .and_then(|value| value.checked_add(cubic))
        .ok_or(QsvtPolynomialError::WorkBudget)?;
    let work = trial_work
        .checked_mul(max_iterations + 1)
        .ok_or(QsvtPolynomialError::WorkBudget)?;
    if work > MAX_QSVT_SYNTHESIS_WORK {
        return Err(QsvtPolynomialError::WorkBudget);
    }
    Ok(())
}

fn canonical_qsp_phase(phase: f64) -> f64 {
    (phase + PI).rem_euclid(TAU) - PI
}

fn single_leading_coefficient(coefficients: &[f64]) -> Option<f64> {
    coefficients[..coefficients.len() - 1]
        .iter()
        .all(|coefficient| *coefficient == 0.0)
        .then_some(coefficients[coefficients.len() - 1])
}

fn expand_symmetric_phases(reduced: &[f64], degree: usize) -> Vec<f64> {
    (0..=degree)
        .map(|position| reduced[position.min(degree - position)])
        .collect()
}

fn solve_symmetric_phases(
    coefficients: &[f64],
    max_error: f64,
    max_iterations: usize,
) -> Result<CertifiedQsvtPolynomial, QsvtPolynomialError> {
    let degree = coefficients.len() - 1;
    let reduced_count = degree.midpoint(2);
    let reduced_count_f64 = bounded_qsvt_coordinate(reduced_count);
    let nodes = (0..reduced_count)
        .map(|node| (bounded_qsvt_coordinate(2 * node + 1) * PI / (4.0 * reduced_count_f64)).cos())
        .collect::<Vec<_>>();
    let targets = nodes
        .iter()
        .map(|node| evaluate_chebyshev(coefficients, *node))
        .collect::<Result<Vec<_>, _>>()?;
    let mut reduced_phases = vec![0.0; reduced_count];
    reduced_phases[0] = FRAC_PI_4;

    for iteration in 0..=max_iterations {
        let qsp_phases = expand_symmetric_phases(&reduced_phases, degree);
        if let Ok(certificate) = verify_qsvt_polynomial(coefficients, &qsp_phases, max_error) {
            return Ok(certificate);
        }
        if iteration == max_iterations {
            break;
        }
        let (residuals, jacobian) = qsp_residual_jacobian(&nodes, &targets, &qsp_phases)?;
        let residual_norm = max_absolute(&residuals)?;
        let right = residuals
            .into_iter()
            .map(|value| -value)
            .collect::<Vec<_>>();
        let step =
            solve_linear_system(jacobian, right).ok_or(QsvtPolynomialError::DidNotConverge)?;
        let mut accepted = None;
        for backtrack in 0..=MAX_NEWTON_BACKTRACKS {
            let scale = 2.0_f64
                .powi(-i32::try_from(backtrack).map_err(|_| QsvtPolynomialError::WorkBudget)?);
            let candidate = reduced_phases
                .iter()
                .zip(&step)
                .map(|(phase, delta)| canonical_qsp_phase(*phase + scale * delta))
                .collect::<Vec<_>>();
            if candidate.iter().any(|phase| !phase.is_finite()) {
                return Err(QsvtPolynomialError::NumericalRange);
            }
            let candidate_phases = expand_symmetric_phases(&candidate, degree);
            let candidate_residuals = nodes
                .iter()
                .zip(&targets)
                .map(|(node, target)| Ok(qsp_value(*node, &candidate_phases)? - target))
                .collect::<Result<Vec<_>, QsvtPolynomialError>>()?;
            if max_absolute(&candidate_residuals)? < residual_norm {
                accepted = Some(candidate);
                break;
            }
        }
        reduced_phases = accepted.ok_or(QsvtPolynomialError::DidNotConverge)?;
    }
    Err(QsvtPolynomialError::DidNotConverge)
}

fn qsp_residual_jacobian(
    nodes: &[f64],
    targets: &[f64],
    qsp_phases: &[f64],
) -> Result<(Vec<f64>, Vec<Vec<f64>>), QsvtPolynomialError> {
    let mut residuals = Vec::with_capacity(nodes.len());
    let mut jacobian = Vec::with_capacity(nodes.len());
    for (node, target) in nodes.iter().zip(targets) {
        let (value, derivatives) = qsp_value_and_gradient(*node, qsp_phases)?;
        let residual = value - target;
        if !residual.is_finite() || derivatives.iter().any(|value| !value.is_finite()) {
            return Err(QsvtPolynomialError::NumericalRange);
        }
        residuals.push(residual);
        jacobian.push(derivatives);
    }
    Ok((residuals, jacobian))
}

fn max_absolute(values: &[f64]) -> Result<f64, QsvtPolynomialError> {
    values.iter().try_fold(0.0_f64, |maximum, value| {
        value
            .is_finite()
            .then(|| maximum.max(value.abs()))
            .ok_or(QsvtPolynomialError::NumericalRange)
    })
}

fn solve_linear_system(mut matrix: Vec<Vec<f64>>, mut right: Vec<f64>) -> Option<Vec<f64>> {
    let size = right.len();
    if matrix.len() != size || matrix.iter().any(|row| row.len() != size) {
        return None;
    }
    for column in 0..size {
        let pivot = (column..size).max_by(|left, right_index| {
            matrix[*left][column]
                .abs()
                .total_cmp(&matrix[*right_index][column].abs())
        })?;
        let pivot_value = matrix[pivot][column];
        if !pivot_value.is_finite() || pivot_value.abs() <= f64::EPSILON {
            return None;
        }
        matrix.swap(column, pivot);
        right.swap(column, pivot);
        let pivot_row = matrix[column].clone();
        for (row, row_values) in matrix.iter_mut().enumerate().skip(column + 1) {
            let factor = row_values[column] / pivot_row[column];
            if !factor.is_finite() {
                return None;
            }
            for (entry, value) in row_values.iter_mut().enumerate().skip(column) {
                *value -= factor * pivot_row[entry];
            }
            right[row] -= factor * right[column];
        }
    }
    let mut result = vec![0.0; size];
    for row in (0..size).rev() {
        let residual = right[row]
            - matrix[row][row + 1..]
                .iter()
                .zip(&result[row + 1..])
                .map(|(coefficient, value)| coefficient * value)
                .sum::<f64>();
        let value = residual / matrix[row][row];
        if !value.is_finite() {
            return None;
        }
        result[row] = value;
    }
    Some(result)
}

fn qsp_value_and_gradient(
    x: f64,
    qsp_phases: &[f64],
) -> Result<(f64, Vec<f64>), QsvtPolynomialError> {
    let degree = qsp_phases.len() - 1;
    let signal = signal_matrix(x)?;
    let rotations = qsp_phases
        .iter()
        .map(|phase| phase_matrix(*phase))
        .collect::<Vec<_>>();
    let mut prefixes = Vec::with_capacity(qsp_phases.len());
    let mut product = identity_matrix();
    for (position, rotation) in rotations.iter().enumerate() {
        prefixes.push(product);
        product = matrix_multiply(product, *rotation);
        if position < degree {
            product = matrix_multiply(product, signal);
        }
    }
    let mut suffixes = vec![identity_matrix(); qsp_phases.len()];
    for position in (0..degree).rev() {
        suffixes[position] = matrix_multiply(
            signal,
            matrix_multiply(rotations[position + 1], suffixes[position + 1]),
        );
    }
    let reduced_count = degree.midpoint(2);
    let mut derivatives = vec![0.0; reduced_count];
    for position in 0..=degree {
        let derivative = matrix_multiply(
            prefixes[position],
            matrix_multiply(phase_derivative(qsp_phases[position]), suffixes[position]),
        )[0][0]
            .re;
        derivatives[position.min(degree - position)] += derivative;
    }
    if !product[0][0].re.is_finite() || derivatives.iter().any(|derivative| !derivative.is_finite())
    {
        return Err(QsvtPolynomialError::NumericalRange);
    }
    Ok((product[0][0].re, derivatives))
}

fn qsp_value(x: f64, qsp_phases: &[f64]) -> Result<f64, QsvtPolynomialError> {
    let signal = signal_matrix(x)?;
    let mut product = identity_matrix();
    for (position, phase) in qsp_phases.iter().enumerate() {
        product = matrix_multiply(product, phase_matrix(*phase));
        if position + 1 < qsp_phases.len() {
            product = matrix_multiply(product, signal);
        }
    }
    product[0][0]
        .re
        .is_finite()
        .then_some(product[0][0].re)
        .ok_or(QsvtPolynomialError::NumericalRange)
}

fn certified_uniform_error(
    coefficients: &[f64],
    qsp_phases: &[f64],
) -> Result<f64, QsvtPolynomialError> {
    if coefficients
        .iter()
        .take(coefficients.len() - 1)
        .all(|coefficient| *coefficient == 0.0)
        && coefficients.last() == Some(&1.0)
        && qsp_phases.iter().all(|phase| *phase == 0.0)
    {
        return Ok(0.0);
    }
    let degree = coefficients.len() - 1;
    let node_count = if degree.is_multiple_of(2) {
        degree + 2
    } else {
        degree + 1
    };
    let node_count_f64 = bounded_qsvt_coordinate(node_count);
    let mut residuals = Vec::with_capacity(node_count);
    for node in 0..node_count {
        let theta = bounded_qsvt_coordinate(2 * node + 1) * PI / (2.0 * node_count_f64);
        let x = theta.cos();
        residuals.push(qsp_value(x, qsp_phases)? - evaluate_chebyshev(coefficients, x)?);
    }
    let mut coefficient_errors = Vec::with_capacity(node_count);
    for order in 0..node_count {
        let scale = if order == 0 {
            1.0 / node_count_f64
        } else {
            2.0 / node_count_f64
        };
        let terms = residuals.iter().enumerate().map(|(node, residual)| {
            let theta = bounded_qsvt_coordinate(2 * node + 1) * PI / (2.0 * node_count_f64);
            residual * (bounded_qsvt_coordinate(order) * theta).cos() * scale
        });
        coefficient_errors.push(
            exact_sum(terms)
                .filter(|value| value.is_finite())
                .ok_or(QsvtPolynomialError::NumericalRange)?,
        );
    }
    let interpolation_bound = exact_sum(coefficient_errors.iter().map(|value| value.abs()))
        .ok_or(QsvtPolynomialError::NumericalRange)?;
    let numerical_margin = QSVT_NUMERICAL_MARGIN_FACTOR * f64::EPSILON * node_count_f64.powi(3);
    let bound = interpolation_bound + numerical_margin;
    bound
        .is_finite()
        .then_some(bound)
        .ok_or(QsvtPolynomialError::NumericalRange)
}

fn evaluate_chebyshev(coefficients: &[f64], x: f64) -> Result<f64, QsvtPolynomialError> {
    let mut next = 0.0;
    let mut following = 0.0;
    for coefficient in coefficients.iter().skip(1).rev() {
        let current = 2.0 * x * next - following + coefficient;
        following = next;
        next = current;
    }
    let value = coefficients[0] + x * next - following;
    value
        .is_finite()
        .then_some(value)
        .ok_or(QsvtPolynomialError::NumericalRange)
}

fn identity_matrix() -> Matrix2 {
    [
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
    ]
}

fn signal_matrix(x: f64) -> Result<Matrix2, QsvtPolynomialError> {
    let complement = (1.0 - x * x).max(0.0).sqrt();
    if !complement.is_finite() {
        return Err(QsvtPolynomialError::NumericalRange);
    }
    Ok([
        [Complex64::new(x, 0.0), Complex64::new(0.0, complement)],
        [Complex64::new(0.0, complement), Complex64::new(x, 0.0)],
    ])
}

fn phase_matrix(phase: f64) -> Matrix2 {
    let positive = Complex64::from_polar(1.0, phase);
    [
        [positive, Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), positive.conj()],
    ]
}

fn phase_derivative(phase: f64) -> Matrix2 {
    let positive = Complex64::from_polar(1.0, phase);
    [
        [
            Complex64::new(0.0, 1.0) * positive,
            Complex64::new(0.0, 0.0),
        ],
        [
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, -1.0) * positive.conj(),
        ],
    ]
}

fn matrix_multiply(left: Matrix2, right: Matrix2) -> Matrix2 {
    let mut result = [[Complex64::new(0.0, 0.0); 2]; 2];
    for row in 0..2 {
        for column in 0..2 {
            result[row][column] = left[row][0] * right[0][column] + left[row][1] * right[1][column];
        }
    }
    result
}

fn bounded_qsvt_coordinate(value: usize) -> f64 {
    f64::from(u32::try_from(value).expect("QSVT coordinates are bounded below u32"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_endpoints_including_the_degree_zero_overlap() {
        let scalar = convert_qsp_wx_phases(&[0.25]).expect("scalar phase");
        assert!((scalar[0] - (0.25 - FRAC_PI_2)).abs() < 4.0 * f64::EPSILON);

        let phases = convert_qsp_wx_phases(&[0.1, 0.2, 0.3]).expect("quadratic phases");
        assert!((phases[0] - (0.1 + 3.0 * FRAC_PI_4)).abs() < 4.0 * f64::EPSILON);
        assert!((phases[1] - (0.2 - FRAC_PI_2)).abs() < 4.0 * f64::EPSILON);
        assert!((phases[2] - (0.3 - FRAC_PI_4)).abs() < 4.0 * f64::EPSILON);
    }

    #[test]
    fn wraps_to_one_canonical_half_open_interval() {
        let phases =
            convert_qsp_wx_phases(&[5.0 * PI / 2.0, -7.0 * PI / 2.0]).expect("wrapped phases");
        assert!(phases.iter().all(|phase| (-PI..PI).contains(phase)));
    }

    #[test]
    fn malformed_and_unrepresentable_inputs_fail_closed() {
        assert_eq!(convert_qsp_wx_phases(&[]), Err(QsvtPhaseError::Empty));
        assert_eq!(
            convert_qsp_wx_phases(&[f64::NAN]),
            Err(QsvtPhaseError::NonFinite)
        );
        let large = convert_qsp_wx_phases(&[f64::MAX, 0.0]).expect("large finite phase");
        assert!(large.iter().all(|phase| (-PI..PI).contains(phase)));
    }

    #[test]
    fn certifies_the_direct_chebyshev_family_exactly() {
        let certificate = synthesize_qsvt_polynomial(&[0.0, 0.0, 0.0, 1.0], 1e-15, 1)
            .expect("Chebyshev certificate");

        assert_eq!(certificate.degree(), 3);
        assert!(certificate.is_odd());
        assert_eq!(certificate.qsp_phases(), &[0.0; 4]);
        assert_eq!(
            certificate.uniform_error_bound().to_bits(),
            0.0_f64.to_bits()
        );
        assert_eq!(
            certificate.target_magnitude_bound().to_bits(),
            1.0_f64.to_bits()
        );
        assert_eq!(
            certificate.realized_magnitude_bound().to_bits(),
            1.0_f64.to_bits()
        );
        assert_eq!(
            verify_qsvt_polynomial(
                certificate.chebyshev_coefficients(),
                certificate.qsp_phases(),
                certificate.uniform_error_bound() + f64::MIN_POSITIVE,
            ),
            Ok(certificate)
        );

        let negative_constant =
            synthesize_qsvt_polynomial(&[-1.0], 1e-12, 1).expect("negative constant certificate");
        assert_eq!(negative_constant.qsp_phases(), &[-PI]);
        assert!(negative_constant.uniform_error_bound() <= 1e-12);

        let scaled = synthesize_qsvt_polynomial(&[0.0, 0.0, -0.5], 1e-12, 1)
            .expect("scaled Chebyshev certificate");
        assert!(scaled.uniform_error_bound() <= 1e-12);
    }

    #[test]
    fn synthesizes_and_replays_a_nontrivial_small_norm_polynomial() {
        let coefficients = [0.002, 0.0, 0.003];
        let certificate =
            synthesize_qsvt_polynomial(&coefficients, 1e-7, 512).expect("synthesized phases");

        assert_eq!(certificate.degree(), 2);
        assert!(!certificate.is_odd());
        assert_eq!(certificate.qsp_phases().len(), 3);
        assert_eq!(
            certificate.qsp_phases()[0].to_bits(),
            certificate.qsp_phases()[2].to_bits()
        );
        assert!(certificate.uniform_error_bound() <= 1e-7);
        assert_eq!(
            verify_qsvt_polynomial(
                &coefficients,
                certificate.qsp_phases(),
                certificate.uniform_error_bound(),
            ),
            Ok(certificate)
        );
    }

    #[test]
    fn synthesis_and_certificate_boundaries_fail_closed() {
        assert_eq!(
            synthesize_qsvt_polynomial(&[0.1, 0.1], 1e-6, 32),
            Err(QsvtPolynomialError::InvalidParity)
        );
        assert_eq!(
            synthesize_qsvt_polynomial(&[0.2, 0.0, 0.2], 1e-6, 32),
            Err(QsvtPolynomialError::UnsupportedTarget)
        );
        assert_eq!(
            synthesize_qsvt_polynomial(&[0.0, 1.0], 1e-6, 0),
            Err(QsvtPolynomialError::IterationBudget)
        );
        let mut oversized = vec![0.0; MAX_CERTIFIED_QSVT_DEGREE + 1];
        oversized[0] = 1e-6;
        oversized[MAX_CERTIFIED_QSVT_DEGREE] = 1e-6;
        assert_eq!(
            synthesize_qsvt_polynomial(&oversized, 1e-6, MAX_QSVT_SYNTHESIS_ITERATIONS,),
            Err(QsvtPolynomialError::WorkBudget)
        );

        let certificate =
            synthesize_qsvt_polynomial(&[0.0, 1.0], 1e-15, 1).expect("linear certificate");
        let mut corrupted = certificate.qsp_phases().to_vec();
        corrupted[0] = 0.1;
        assert_eq!(
            verify_qsvt_polynomial(&[0.0, 1.0], &corrupted, 1e-6),
            Err(QsvtPolynomialError::InvalidCertificate)
        );
    }
}
