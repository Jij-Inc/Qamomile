//! Bounded fermionic-QAOA host plans over canonical binary polynomials.

use std::collections::{BTreeMap, BTreeSet};

use num_bigint::BigUint;
use thiserror::Error;

use crate::{
    CanonicalPolynomial, NormalizedObservable, ObservableError, Pauli, PolynomialDomain,
    PolynomialError, convert_polynomial, normalize_real_observable,
};

/// Maximum number of fermionic modes in one host plan.
pub const MAX_FQAOA_MODES: usize = 4096;
/// Maximum occupied-orbital matrix cells materialized by one plan.
pub const MAX_FQAOA_ORBITAL_CELLS: usize = 1_000_000;
/// Maximum scalar updates admitted by Givens synthesis.
pub const MAX_FQAOA_GIVENS_WORK: usize = 50_000_000;
/// Maximum decoded sample rows accepted in one call.
pub const MAX_FQAOA_DECODE_ROWS: usize = 1_000_000;
/// Maximum decoded bit cells accepted in one call.
pub const MAX_FQAOA_DECODE_CELLS: usize = 10_000_000;

type GivensSchedule = (Vec<(usize, usize)>, Vec<f64>);

/// One original decision variable and its fermionic grid coordinate.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FermionicMode {
    variable: BigUint,
    site: usize,
    orbital: usize,
}

impl FermionicMode {
    /// Construct one mode row without narrowing the original variable ID.
    #[must_use]
    pub fn new(variable: BigUint, site: usize, orbital: usize) -> Self {
        Self {
            variable,
            site,
            orbital,
        }
    }
}

/// One decoded FQAOA sample in canonical source-variable order.
#[derive(Clone, Debug, PartialEq)]
pub struct FqaoaDecodedSample {
    values: Vec<i8>,
    energy: f64,
    occurrences: usize,
}

impl FqaoaDecodedSample {
    /// Return values in canonical source-variable order.
    #[must_use]
    pub fn values(&self) -> &[i8] {
        &self.values
    }

    /// Return the source objective energy.
    #[must_use]
    pub const fn energy(&self) -> f64 {
        self.energy
    }

    /// Return the sample multiplicity.
    #[must_use]
    pub const fn occurrences(&self) -> usize {
        self.occurrences
    }
}

/// One immutable fermionic geometry, preparation, cost, and decode plan.
#[derive(Clone, Debug, PartialEq)]
pub struct FqaoaPlan {
    source: CanonicalPolynomial,
    num_sites: usize,
    num_orbitals: usize,
    variable_order: Vec<usize>,
    givens_pairs: Vec<(usize, usize)>,
    givens_angles: Vec<f64>,
    hamiltonian: NormalizedObservable,
}

impl FqaoaPlan {
    /// Derive a bounded plan from one binary model and explicit mode geometry.
    ///
    /// # Errors
    ///
    /// Returns [`FqaoaError`] for an invalid model, grid, particle sector,
    /// numerical schedule, observable, or structural budget.
    pub fn new(
        source: &CanonicalPolynomial,
        modes: &[FermionicMode],
        particles: usize,
    ) -> Result<Self, FqaoaError> {
        if source.domain() != PolynomialDomain::Binary {
            return Err(FqaoaError::BinaryModel);
        }
        if source.terms().iter().any(|term| term.variables().len() > 2) {
            return Err(FqaoaError::NonQuadratic);
        }
        let (num_sites, num_orbitals, variable_order) =
            validate_geometry(source, modes, particles)?;
        let (givens_pairs, givens_angles) = if particles == 0 || particles == modes.len() {
            (Vec::new(), Vec::new())
        } else {
            let orbitals = fermi_orbitals(particles, modes.len())?;
            givens_schedule(&orbitals)?
        };
        let hamiltonian = physical_hamiltonian(source, &variable_order)?;
        Ok(Self {
            source: source.clone(),
            num_sites,
            num_orbitals,
            variable_order,
            givens_pairs,
            givens_angles,
            hamiltonian,
        })
    }

    /// Return the dense site count.
    #[must_use]
    pub const fn num_sites(&self) -> usize {
        self.num_sites
    }

    /// Return the dense orbital count.
    #[must_use]
    pub const fn num_orbitals(&self) -> usize {
        self.num_orbitals
    }

    /// Return source-variable positions in physical periodic mode order.
    #[must_use]
    pub fn variable_order(&self) -> &[usize] {
        &self.variable_order
    }

    /// Return the inverse preparation's adjacent mode pairs.
    #[must_use]
    pub fn givens_pairs(&self) -> &[(usize, usize)] {
        &self.givens_pairs
    }

    /// Return the inverse preparation's finite angles.
    #[must_use]
    pub fn givens_angles(&self) -> &[f64] {
        &self.givens_angles
    }

    /// Return the physical-mode cost in the common Pauli normal form.
    #[must_use]
    pub const fn hamiltonian(&self) -> &NormalizedObservable {
        &self.hamiltonian
    }

    /// Decode physical bit rows into canonical source-variable assignments.
    ///
    /// # Errors
    ///
    /// Returns [`FqaoaError`] for malformed or excessive rows and
    /// unrepresentable source energy.
    pub fn decode(&self, rows: &[(Vec<u8>, usize)]) -> Result<Vec<FqaoaDecodedSample>, FqaoaError> {
        if rows.is_empty() {
            return Err(FqaoaError::InvalidSamples);
        }
        if rows.len() > MAX_FQAOA_DECODE_ROWS {
            return Err(FqaoaError::DecodeBudget);
        }
        let cells = rows
            .len()
            .checked_mul(self.variable_order.len())
            .ok_or(FqaoaError::DecodeBudget)?;
        if cells > MAX_FQAOA_DECODE_CELLS {
            return Err(FqaoaError::DecodeBudget);
        }
        let mut seen = BTreeSet::new();
        let mut decoded = Vec::with_capacity(rows.len());
        for (bits, occurrences) in rows {
            if bits.len() != self.variable_order.len()
                || bits.iter().any(|bit| *bit > 1)
                || *occurrences == 0
                || !seen.insert(bits.clone())
            {
                return Err(FqaoaError::InvalidSamples);
            }
            let mut values = vec![0_i8; self.variable_order.len()];
            for (physical, source) in self.variable_order.iter().copied().enumerate() {
                values[source] = i8::from(bits[physical] != 0);
            }
            let energy = self.source.energy(&values)?;
            decoded.push(FqaoaDecodedSample {
                values,
                energy,
                occurrences: *occurrences,
            });
        }
        Ok(decoded)
    }
}

/// Failure to derive or use one FQAOA plan.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum FqaoaError {
    /// FQAOA requires the binary source domain.
    #[error("Q7591: FQAOA requires a binary model")]
    BinaryModel,
    /// FQAOA accepts only quadratic costs.
    #[error("Q7592: FQAOA requires a quadratic cost model")]
    NonQuadratic,
    /// Mode rows do not form one unique dense rectangular grid.
    #[error("Q7593: FQAOA variables must form a unique dense rectangular (site, orbital) grid")]
    InvalidGeometry,
    /// Source variables and fermionic modes are not the same closed universe.
    #[error("Q7594: FQAOA does not support transformed auxiliary variables without a mode")]
    AuxiliaryVariables,
    /// The particle sector is outside the mode count.
    #[error("Q7595: particles must lie between zero and the mode count")]
    InvalidParticles,
    /// Geometry or numerical synthesis exceeds an explicit bound.
    #[error("Q7596: FQAOA preparation exceeds its structural work budget")]
    PreparationBudget,
    /// Givens synthesis produced a non-finite coordinate.
    #[error("Q7597: FQAOA preparation produced a non-finite coordinate")]
    NonFinitePreparation,
    /// Sample rows violate shape, bit, uniqueness, or multiplicity contracts.
    #[error("Q7598: FQAOA samples must be unique positive fixed-width bit rows")]
    InvalidSamples,
    /// Decoding exceeds the explicit row or cell budget.
    #[error("Q7599: FQAOA decoding exceeds its work budget")]
    DecodeBudget,
    /// The shared polynomial rejected conversion or energy evaluation.
    #[error(transparent)]
    Polynomial(#[from] PolynomialError),
    /// The common observable normal form rejected the physical Hamiltonian.
    #[error(transparent)]
    Observable(#[from] ObservableError),
}

fn validate_geometry(
    source: &CanonicalPolynomial,
    modes: &[FermionicMode],
    particles: usize,
) -> Result<(usize, usize, Vec<usize>), FqaoaError> {
    if modes.len() < 2 || modes.len() > MAX_FQAOA_MODES {
        return Err(FqaoaError::InvalidGeometry);
    }
    if particles > modes.len() {
        return Err(FqaoaError::InvalidParticles);
    }
    let source_positions = source
        .variables()
        .iter()
        .enumerate()
        .map(|(position, variable)| (variable, position))
        .collect::<BTreeMap<_, _>>();
    let mode_variables = modes
        .iter()
        .map(|mode| &mode.variable)
        .collect::<BTreeSet<_>>();
    if mode_variables.len() != modes.len()
        || mode_variables.len() != source.variables().len()
        || mode_variables
            .iter()
            .any(|variable| !source_positions.contains_key(*variable))
    {
        return Err(FqaoaError::AuxiliaryVariables);
    }
    let num_sites = modes
        .iter()
        .map(|mode| mode.site)
        .max()
        .and_then(|value| value.checked_add(1))
        .ok_or(FqaoaError::InvalidGeometry)?;
    let num_orbitals = modes
        .iter()
        .map(|mode| mode.orbital)
        .max()
        .and_then(|value| value.checked_add(1))
        .ok_or(FqaoaError::InvalidGeometry)?;
    if num_sites.checked_mul(num_orbitals) != Some(modes.len()) {
        return Err(FqaoaError::InvalidGeometry);
    }
    let by_coordinate = modes
        .iter()
        .map(|mode| ((mode.site, mode.orbital), &mode.variable))
        .collect::<BTreeMap<_, _>>();
    if by_coordinate.len() != modes.len() {
        return Err(FqaoaError::InvalidGeometry);
    }
    let mut variable_order = Vec::with_capacity(modes.len());
    for orbital in 0..num_orbitals {
        for site in 0..num_sites {
            let variable = by_coordinate
                .get(&(site, orbital))
                .ok_or(FqaoaError::InvalidGeometry)?;
            variable_order.push(source_positions[*variable]);
        }
    }
    Ok((num_sites, num_orbitals, variable_order))
}

/// Return real orthonormal occupied ring orbitals under explicit budgets.
///
/// # Errors
///
/// Returns [`FqaoaError`] for invalid particle/width domains, excessive
/// materialization, or a non-finite coordinate.
fn fermi_orbitals(particles: usize, width: usize) -> Result<Vec<Vec<f64>>, FqaoaError> {
    if !(2..=MAX_FQAOA_MODES).contains(&width) || particles > width {
        return Err(FqaoaError::InvalidParticles);
    }
    let cells = particles
        .checked_mul(width)
        .ok_or(FqaoaError::PreparationBudget)?;
    if cells > MAX_FQAOA_ORBITAL_CELLS {
        return Err(FqaoaError::PreparationBudget);
    }
    let width_real = f64::from(u32::try_from(width).map_err(|_| FqaoaError::PreparationBudget)?);
    let mut rows = vec![vec![0.0; width]; particles];
    if particles.is_multiple_of(2) {
        for wave in 0..particles / 2 {
            let reflected = particles - 1 - wave;
            let (upper, lower) = rows.split_at_mut(reflected);
            let sine_row = &mut upper[wave];
            let cosine_row = &mut lower[0];
            for (mode, (sine, cosine)) in sine_row.iter_mut().zip(cosine_row).enumerate() {
                let wave_real =
                    f64::from(u32::try_from(wave).map_err(|_| FqaoaError::PreparationBudget)?);
                let mode_real =
                    f64::from(u32::try_from(mode + 1).map_err(|_| FqaoaError::PreparationBudget)?);
                let angle = 2.0 * std::f64::consts::PI * (wave_real + 0.5) * mode_real / width_real;
                *sine = (2.0 / width_real).sqrt() * angle.sin();
                *cosine = (2.0 / width_real).sqrt() * angle.cos();
            }
        }
    } else {
        rows[0].fill((1.0 / width_real).sqrt());
        for wave in 0..particles / 2 {
            let sine_index = wave + 1;
            let cosine_index = particles - 1 - wave;
            let (upper, lower) = rows.split_at_mut(cosine_index);
            let sine_row = &mut upper[sine_index];
            let cosine_row = &mut lower[0];
            for (mode, (sine, cosine)) in sine_row.iter_mut().zip(cosine_row).enumerate() {
                let wave_real =
                    f64::from(u32::try_from(wave + 1).map_err(|_| FqaoaError::PreparationBudget)?);
                let mode_real =
                    f64::from(u32::try_from(mode + 1).map_err(|_| FqaoaError::PreparationBudget)?);
                let angle = 2.0 * std::f64::consts::PI * wave_real * mode_real / width_real;
                *sine = (2.0 / width_real).sqrt() * angle.sin();
                *cosine = (2.0 / width_real).sqrt() * angle.cos();
            }
        }
    }
    if rows.iter().flatten().any(|value| !value.is_finite()) {
        return Err(FqaoaError::NonFinitePreparation);
    }
    Ok(rows)
}

fn givens_schedule(orbitals: &[Vec<f64>]) -> Result<GivensSchedule, FqaoaError> {
    let particles = orbitals.len();
    if particles == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let width = orbitals[0].len();
    if particles == width {
        return Ok((Vec::new(), Vec::new()));
    }
    let work = particles
        .checked_mul(particles)
        .and_then(|value| value.checked_mul(width))
        .ok_or(FqaoaError::PreparationBudget)?;
    if work > MAX_FQAOA_GIVENS_WORK {
        return Err(FqaoaError::PreparationBudget);
    }
    let mut matrix = orbitals.to_vec();
    for column in (width - particles..width).rev() {
        let row_limit = column
            .checked_add(particles)
            .and_then(|value| value.checked_sub(width))
            .ok_or(FqaoaError::PreparationBudget)?;
        for row in 0..row_limit {
            let norm = matrix[row][column].hypot(matrix[row + 1][column]);
            if norm == 0.0 {
                continue;
            }
            let sine = -matrix[row][column] / norm;
            let cosine = matrix[row + 1][column] / norm;
            let upper = matrix[row].clone();
            let lower = matrix[row + 1].clone();
            for index in 0..width {
                matrix[row][index] = cosine * upper[index] + sine * lower[index];
                matrix[row + 1][index] = -sine * upper[index] + cosine * lower[index];
            }
        }
    }
    let mut eliminations = Vec::new();
    for row in 0..particles {
        for column in (row..row + width - particles).rev() {
            let norm = matrix[row][column].hypot(matrix[row][column + 1]);
            if norm == 0.0 {
                continue;
            }
            let cosine = matrix[row][column] / norm;
            let sine = -matrix[row][column + 1] / norm;
            let angle = sine.atan2(cosine);
            if !angle.is_finite() {
                return Err(FqaoaError::NonFinitePreparation);
            }
            eliminations.push(((column, column + 1), angle));
            let left = matrix
                .iter()
                .map(|values| values[column])
                .collect::<Vec<_>>();
            let right = matrix
                .iter()
                .map(|values| values[column + 1])
                .collect::<Vec<_>>();
            for target in 0..particles {
                matrix[target][column] = cosine * left[target] - sine * right[target];
                matrix[target][column + 1] = sine * left[target] + cosine * right[target];
            }
        }
    }
    eliminations.reverse();
    let (pairs, angles): (Vec<_>, Vec<_>) = eliminations.into_iter().unzip();
    Ok((pairs, angles))
}

fn physical_hamiltonian(
    source: &CanonicalPolynomial,
    variable_order: &[usize],
) -> Result<NormalizedObservable, FqaoaError> {
    let spin = convert_polynomial(source, PolynomialDomain::Spin)?;
    let physical = variable_order
        .iter()
        .copied()
        .enumerate()
        .map(|(physical, source)| (source, physical))
        .collect::<BTreeMap<_, _>>();
    let positions = spin
        .variables()
        .iter()
        .enumerate()
        .map(|(position, variable)| (variable, position))
        .collect::<BTreeMap<_, _>>();
    let terms = spin
        .terms()
        .iter()
        .map(|term| {
            let mut word = term
                .variables()
                .iter()
                .map(|variable| (physical[&positions[variable]], Pauli::Z))
                .collect::<Vec<_>>();
            word.sort_unstable_by_key(|(index, _)| *index);
            (word, term.coefficient())
        })
        .collect::<Vec<_>>();
    Ok(normalize_real_observable(
        variable_order.len(),
        spin.constant(),
        &terms,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source() -> CanonicalPolynomial {
        crate::normalize_polynomial(
            PolynomialDomain::Binary,
            &[
                vec![40_u8.into()],
                vec![7_u8.into()],
                vec![20_u8.into()],
                vec![3_u8.into()],
                vec![40_u8.into(), 3_u8.into()],
            ],
            &[1.0, 2.0, -1.0, 0.0, 0.5],
            0.0,
            Some(&[40_u8.into(), 7_u8.into(), 20_u8.into(), 3_u8.into()]),
        )
        .expect("source")
    }

    #[test]
    fn plan_owns_periodic_order_preparation_cost_and_decode() {
        let modes = vec![
            FermionicMode::new(40_u8.into(), 0, 0),
            FermionicMode::new(7_u8.into(), 0, 1),
            FermionicMode::new(20_u8.into(), 1, 0),
            FermionicMode::new(3_u8.into(), 1, 1),
        ];
        let plan = FqaoaPlan::new(&source(), &modes, 2).expect("plan");

        assert_eq!(plan.num_sites(), 2);
        assert_eq!(plan.num_orbitals(), 2);
        assert_eq!(plan.variable_order(), &[3, 2, 1, 0]);
        assert_eq!(plan.givens_pairs().len(), plan.givens_angles().len());
        assert!(plan.hamiltonian().is_hermitian());
        let decoded = plan.decode(&[(vec![1, 0, 1, 0], 3)]).expect("decode");
        assert_eq!(decoded[0].values(), &[0, 1, 0, 1]);
        assert_eq!(decoded[0].occurrences(), 3);
    }

    #[test]
    fn odd_ring_orbitals_are_orthonormal() {
        let orbitals = fermi_orbitals(3, 8).expect("orbitals");
        for (left, left_row) in orbitals.iter().enumerate() {
            for (right, right_row) in orbitals.iter().enumerate() {
                let inner = left_row
                    .iter()
                    .zip(right_row)
                    .map(|(left, right)| left * right)
                    .sum::<f64>();
                let expected = if left == right { 1.0 } else { 0.0 };
                assert!((inner - expected).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn invalid_geometry_sector_and_samples_fail_closed() {
        let source = source();
        let valid = vec![
            FermionicMode::new(40_u8.into(), 0, 0),
            FermionicMode::new(7_u8.into(), 0, 1),
            FermionicMode::new(20_u8.into(), 1, 0),
            FermionicMode::new(3_u8.into(), 1, 1),
        ];
        let malformed = vec![
            FermionicMode::new(40_u8.into(), 0, 0),
            FermionicMode::new(7_u8.into(), 0, 2),
            FermionicMode::new(20_u8.into(), 1, 0),
            FermionicMode::new(3_u8.into(), 1, 1),
        ];
        assert_eq!(
            FqaoaPlan::new(&source, &malformed, 2),
            Err(FqaoaError::InvalidGeometry)
        );
        assert_eq!(
            FqaoaPlan::new(&source, &malformed, 5),
            Err(FqaoaError::InvalidParticles)
        );
        let plan = FqaoaPlan::new(&source, &valid, 4).expect("full-sector plan");
        assert!(plan.givens_pairs().is_empty());
        assert_eq!(
            plan.decode(&[(vec![0, 1, 2, 0], 1)]),
            Err(FqaoaError::InvalidSamples)
        );
        assert_eq!(
            plan.decode(&[(vec![0, 1, 1, 0], 1), (vec![0, 1, 1, 0], 2)]),
            Err(FqaoaError::InvalidSamples)
        );
        assert_eq!(
            fermi_orbitals(1_000, 2_000),
            Err(FqaoaError::PreparationBudget)
        );
    }
}
