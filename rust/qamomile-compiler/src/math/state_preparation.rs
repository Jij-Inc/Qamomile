//! Native classical coordinates for portable quantum state preparation.
//!
//! The compiler foundation owns normalization, phase collapse, and the
//! Gray-permuted Walsh transform used by Qamomile's Möttönen construction.
//! Gate emission remains ordinary `QCore` rotation and CNOT composition.

use std::f64::consts::{PI, TAU};

use num_complex::Complex64;
use thiserror::Error;

/// Maximum number of amplitudes accepted by one coordinate transform.
pub const MAX_AMPLITUDE_COUNT: usize = 1 << 20;

/// Exact-phase Möttönen coordinates for one normalized state vector.
#[derive(Clone, Debug, PartialEq)]
pub struct AmplitudeCoordinates {
    width: usize,
    ry: Vec<f64>,
    rz: Vec<f64>,
    phase: f64,
}

impl AmplitudeCoordinates {
    /// Return the logical qubit width.
    #[must_use]
    pub const fn width(&self) -> usize {
        self.width
    }

    /// Return level-ordered Gray-walk Y-rotation angles.
    #[must_use]
    pub fn ry(&self) -> &[f64] {
        &self.ry
    }

    /// Return level-ordered Gray-walk Z-rotation angles.
    #[must_use]
    pub fn rz(&self) -> &[f64] {
        &self.rz
    }

    /// Return the retained global phase.
    #[must_use]
    pub const fn phase(&self) -> f64 {
        self.phase
    }
}

/// Failure to derive bounded state-preparation coordinates.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum StatePreparationError {
    /// The vector shape cannot describe a positive-width quantum state.
    #[error("Q7501: amplitude count must be a power of two and at least two")]
    InvalidLength,
    /// The vector would exceed the explicit preprocessing domain.
    #[error("Q7502: amplitude vector exceeds the {MAX_AMPLITUDE_COUNT}-entry coordinate budget")]
    Budget,
    /// An amplitude is not a finite complex number.
    #[error("Q7503: each amplitude must be finite")]
    NonFinite,
    /// The vector has no support and therefore cannot be normalized.
    #[error("Q7504: amplitudes must not all be zero")]
    ZeroVector,
    /// Floating-point coordinate synthesis did not remain finite.
    #[error("Q7505: amplitude coordinates exceed the finite binary64 range")]
    CoordinateRange,
    /// A branch table cannot address a uniformly controlled rotation.
    #[error("Q7511: branch angle count must be a nonempty power of two")]
    InvalidBranchCount,
    /// A branch table would exceed the explicit preprocessing domain.
    #[error("Q7512: branch angle table exceeds the {MAX_AMPLITUDE_COUNT}-entry coordinate budget")]
    BranchBudget,
    /// A branch rotation angle is not finite.
    #[error("Q7513: branch angles must contain only finite values")]
    NonFiniteBranchAngle,
    /// The Walsh transform produced a non-finite rotation coordinate.
    #[error("Q7514: uniformly controlled rotation coordinates exceed the finite binary64 range")]
    BranchCoordinateRange,
}

/// Normalize a finite vector and derive its exact-phase Möttönen coordinates.
///
/// The returned vectors each contain `2**width - 1` angles ordered by forward
/// target level. Per-level angles use paper Eq. (3)'s normalized Walsh basis in
/// binary-reflected Gray-walk order. Normalization is scale-first, avoiding
/// overflow for otherwise representable finite inputs. Runtime is
/// `O(N log N)` and working memory is `O(N)` for `N` input amplitudes.
///
/// # Errors
///
/// Returns [`StatePreparationError`] when the vector length, finite-value,
/// nonzero-norm, bounded-work, or binary64-output contract is violated.
pub fn mottonen_amplitude_coordinates(
    amplitudes: &[Complex64],
) -> Result<AmplitudeCoordinates, StatePreparationError> {
    let size = amplitudes.len();
    if size < 2 || !size.is_power_of_two() {
        return Err(StatePreparationError::InvalidLength);
    }
    if size > MAX_AMPLITUDE_COUNT {
        return Err(StatePreparationError::Budget);
    }
    if amplitudes
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(StatePreparationError::NonFinite);
    }

    let scale = amplitudes
        .iter()
        .map(|value| value.norm())
        .fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return Err(StatePreparationError::ZeroVector);
    }
    let scaled_norm = amplitudes
        .iter()
        .map(|value| {
            let scaled = *value / scale;
            scaled.norm_sqr()
        })
        .sum::<f64>()
        .sqrt();
    let mut current = amplitudes
        .iter()
        .map(|value| (*value / scale) / scaled_norm)
        .collect::<Vec<_>>();
    let width = size.ilog2() as usize;
    let mut magnitude_levels = (0..width)
        .map(|level| vec![0.0; 1 << level])
        .collect::<Vec<_>>();
    let mut phase_levels = (0..width)
        .map(|level| vec![0.0; 1 << level])
        .collect::<Vec<_>>();

    for forward_level in (0..width).rev() {
        let mut collapsed = Vec::with_capacity(current.len() / 2);
        for (pair, values) in current.as_chunks::<2>().0.iter().enumerate() {
            let left = values[0];
            let right = values[1];
            let left_magnitude = left.norm();
            let right_magnitude = right.norm();
            let magnitude = left_magnitude.hypot(right_magnitude);
            magnitude_levels[forward_level][pair] = 2.0 * right_magnitude.atan2(left_magnitude);
            let average_phase = if left_magnitude > 0.0 && right_magnitude > 0.0 {
                let relative_phase = wrapped_phase(right.arg() - left.arg());
                phase_levels[forward_level][pair] = relative_phase;
                left.arg() + relative_phase / 2.0
            } else if left_magnitude > 0.0 {
                left.arg()
            } else if right_magnitude > 0.0 {
                right.arg()
            } else {
                0.0
            };
            collapsed.push(Complex64::from_polar(magnitude, average_phase));
        }
        current = collapsed;
    }

    let ry = flatten_levels(gray_walk_basis(magnitude_levels));
    let rz = flatten_levels(gray_walk_basis(phase_levels));
    let phase = current[0].arg();
    if ry
        .iter()
        .chain(&rz)
        .chain([&phase])
        .any(|value| !value.is_finite())
    {
        return Err(StatePreparationError::CoordinateRange);
    }
    Ok(AmplitudeCoordinates {
        width,
        ry,
        rz,
        phase,
    })
}

/// Convert address-ordered branch rotations into cyclic Gray-walk coordinates.
///
/// This is the direct normalized Walsh-Hadamard transform for a uniformly
/// controlled rotation. Unlike Möttönen's recursively indexed level rows,
/// the input is already in the target's little-endian address order and does
/// not need the level-local bit reversal. Runtime is `O(N log N)` and working
/// memory is `O(N)` for `N` branch angles.
///
/// # Errors
///
/// Returns [`StatePreparationError`] when the table is empty, not a power of
/// two, too large, non-finite, or produces an unrepresentable coordinate.
pub fn uniformly_controlled_rotation_coordinates(
    branch_angles: &[f64],
) -> Result<Vec<f64>, StatePreparationError> {
    let count = branch_angles.len();
    if count == 0 || !count.is_power_of_two() {
        return Err(StatePreparationError::InvalidBranchCount);
    }
    if count > MAX_AMPLITUDE_COUNT {
        return Err(StatePreparationError::BranchBudget);
    }
    if branch_angles.iter().any(|angle| !angle.is_finite()) {
        return Err(StatePreparationError::NonFiniteBranchAngle);
    }
    let mut coordinates = branch_angles.to_vec();
    gray_permuted_walsh_in_place(&mut coordinates);
    if coordinates.iter().any(|angle| !angle.is_finite()) {
        return Err(StatePreparationError::BranchCoordinateRange);
    }
    Ok(coordinates)
}

/// Return the control components for one complete cyclic Gray walk.
///
/// Width zero is the identity schedule. Larger widths are bounded by the same
/// table-size contract as uniformly controlled coordinate synthesis. Runtime
/// and output memory are both `O(2**width)`.
///
/// # Errors
///
/// Returns [`StatePreparationError::BranchBudget`] when the requested walk
/// exceeds the bounded coordinate-table domain.
pub fn cyclic_gray_control_schedule(width: usize) -> Result<Vec<usize>, StatePreparationError> {
    let maximum_width = MAX_AMPLITUDE_COUNT.ilog2() as usize;
    if width > maximum_width {
        return Err(StatePreparationError::BranchBudget);
    }
    if width == 0 {
        return Ok(Vec::new());
    }
    let count = 1_usize << width;
    Ok((1..=count)
        .map(|step| {
            if step == count {
                width - 1
            } else {
                step.trailing_zeros() as usize
            }
        })
        .collect())
}

/// Return angle, target, and control indices for Möttönen's Gray walks.
///
/// The order matches the level-flattened coordinate rows returned by
/// [`mottonen_amplitude_coordinates`]. Runtime and output memory are both
/// `O(2**width)`.
///
/// # Errors
///
/// Returns [`StatePreparationError::InvalidLength`] for width zero and
/// [`StatePreparationError::Budget`] when the corresponding amplitude vector
/// exceeds the preprocessing domain.
pub fn mottonen_rotation_schedule(
    width: usize,
) -> Result<Vec<(usize, usize, usize)>, StatePreparationError> {
    if width == 0 {
        return Err(StatePreparationError::InvalidLength);
    }
    let maximum_width = MAX_AMPLITUDE_COUNT.ilog2() as usize;
    if width > maximum_width {
        return Err(StatePreparationError::Budget);
    }
    let mut schedule = Vec::with_capacity((1_usize << width) - 2);
    for level in 1..width {
        let steps = 1_usize << level;
        let angle_offset = steps - 1;
        for (offset, changed) in cyclic_gray_control_schedule(level)?.into_iter().enumerate() {
            schedule.push((
                angle_offset + offset,
                width - 1 - level,
                width - 1 - changed,
            ));
        }
    }
    Ok(schedule)
}

fn wrapped_phase(mut phase: f64) -> f64 {
    if phase > PI {
        phase -= TAU;
    } else if phase <= -PI {
        phase += TAU;
    }
    phase
}

fn gray_walk_basis(mut levels: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    for (control_width, level) in levels.iter_mut().enumerate().skip(1) {
        let source = level.clone();
        for (index, value) in level.iter_mut().enumerate() {
            *value = source[reverse_bits(index, control_width)];
        }
        gray_permuted_walsh_in_place(level);
    }
    levels
}

fn gray_permuted_walsh_in_place(values: &mut [f64]) {
    walsh_hadamard_in_place(values);
    let hadamard = values.to_vec();
    let denominator =
        f64::from(u32::try_from(values.len()).expect("bounded coordinate tables fit in u32"));
    for (index, value) in values.iter_mut().enumerate() {
        *value = hadamard[index ^ (index >> 1)] / denominator;
    }
}

fn walsh_hadamard_in_place(values: &mut [f64]) {
    let mut half_width = 1;
    while half_width < values.len() {
        for start in (0..values.len()).step_by(2 * half_width) {
            for offset in 0..half_width {
                let left_index = start + offset;
                let right_index = left_index + half_width;
                let left = values[left_index];
                let right = values[right_index];
                values[left_index] = left + right;
                values[right_index] = left - right;
            }
        }
        half_width *= 2;
    }
}

fn reverse_bits(mut value: usize, width: usize) -> usize {
    let mut reversed = 0;
    for _ in 0..width {
        reversed = (reversed << 1) | (value & 1);
        value >>= 1;
    }
    reversed
}

fn flatten_levels(levels: Vec<Vec<f64>>) -> Vec<f64> {
    levels.into_iter().flatten().collect()
}

#[cfg(test)]
mod tests {
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    use super::*;

    #[test]
    fn basis_state_coordinates_have_one_pi_rotation() {
        let coordinates = mottonen_amplitude_coordinates(&[
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ])
        .expect("derive coordinates");

        assert_eq!(coordinates.width(), 2);
        assert_eq!(coordinates.ry(), &[PI, FRAC_PI_2, -FRAC_PI_2]);
        assert_eq!(coordinates.rz(), &[0.0, 0.0, 0.0]);
        assert!(coordinates.phase().abs() < f64::EPSILON);
    }

    #[test]
    fn complex_pair_retains_relative_and_global_phase() {
        let coordinates =
            mottonen_amplitude_coordinates(&[Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)])
                .expect("derive coordinates");

        assert!((coordinates.ry()[0] - FRAC_PI_2).abs() < 1e-15);
        assert!((coordinates.rz()[0] - FRAC_PI_2).abs() < 1e-15);
        assert!((coordinates.phase() - FRAC_PI_4).abs() < 1e-15);
    }

    #[test]
    fn scale_first_normalization_accepts_large_finite_values() {
        let coordinates = mottonen_amplitude_coordinates(&[
            Complex64::new(f64::MAX, 0.0),
            Complex64::new(f64::MAX, 0.0),
        ])
        .expect("derive coordinates");

        assert!((coordinates.ry()[0] - FRAC_PI_2).abs() < 1e-15);
    }

    #[test]
    fn malformed_vectors_fail_closed() {
        assert_eq!(
            mottonen_amplitude_coordinates(&[Complex64::new(1.0, 0.0)]),
            Err(StatePreparationError::InvalidLength)
        );
        assert_eq!(
            mottonen_amplitude_coordinates(&[Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),]),
            Err(StatePreparationError::ZeroVector)
        );
        assert_eq!(
            mottonen_amplitude_coordinates(&[
                Complex64::new(f64::INFINITY, 0.0),
                Complex64::new(0.0, 0.0),
            ]),
            Err(StatePreparationError::NonFinite)
        );
    }

    #[test]
    fn uniformly_controlled_coordinates_use_cyclic_gray_order() {
        let coordinates = uniformly_controlled_rotation_coordinates(&[1.0, 2.0, 4.0, 8.0])
            .expect("derive coordinates");

        assert_eq!(coordinates, vec![3.75, -1.25, 0.75, -2.25]);
    }

    #[test]
    fn uniformly_controlled_coordinates_fail_closed() {
        assert_eq!(
            uniformly_controlled_rotation_coordinates(&[]),
            Err(StatePreparationError::InvalidBranchCount)
        );
        assert_eq!(
            uniformly_controlled_rotation_coordinates(&[0.0, 1.0, 2.0]),
            Err(StatePreparationError::InvalidBranchCount)
        );
        assert_eq!(
            uniformly_controlled_rotation_coordinates(&[0.0, f64::NAN]),
            Err(StatePreparationError::NonFiniteBranchAngle)
        );
        assert_eq!(
            uniformly_controlled_rotation_coordinates(&[f64::MAX, f64::MAX]),
            Err(StatePreparationError::BranchCoordinateRange)
        );
    }

    #[test]
    fn gray_control_and_mottonen_schedules_share_one_native_walk() {
        assert_eq!(
            cyclic_gray_control_schedule(2).expect("Gray controls"),
            vec![0, 1, 0, 1]
        );
        assert_eq!(
            mottonen_rotation_schedule(3).expect("Möttönen schedule"),
            vec![
                (1, 1, 2),
                (2, 1, 2),
                (3, 0, 2),
                (4, 0, 1),
                (5, 0, 2),
                (6, 0, 1),
            ]
        );
        assert!(
            cyclic_gray_control_schedule(0)
                .expect("empty walk")
                .is_empty()
        );
        assert!(
            mottonen_rotation_schedule(1)
                .expect("single-qubit schedule")
                .is_empty()
        );
    }

    #[test]
    fn gray_schedules_share_coordinate_budgets() {
        assert_eq!(
            cyclic_gray_control_schedule(21),
            Err(StatePreparationError::BranchBudget)
        );
        assert_eq!(
            mottonen_rotation_schedule(0),
            Err(StatePreparationError::InvalidLength)
        );
        assert_eq!(
            mottonen_rotation_schedule(21),
            Err(StatePreparationError::Budget)
        );
    }
}
