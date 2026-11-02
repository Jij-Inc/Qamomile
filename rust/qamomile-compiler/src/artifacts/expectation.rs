//! Backend-neutral Pauli measurement planning and raw-sample parity.
//!
//! Measurement programs remain ordinary verified `QCore` programs. Typed
//! estimates and their uncertainty belong to the shared statistics module.

use std::collections::BTreeMap;

use num_bigint::{BigInt, BigUint};
use num_traits::{ToPrimitive, Zero};
use thiserror::Error;

use crate::observable::{Observable, Pauli};
use crate::sample_counts::{MAX_SAMPLE_COUNT_BITS, SampleCountSummary, bigint_ratio};

/// Maximum number of independently sampled Pauli terms in one estimate.
pub const MAX_EXPECTATION_TERMS: usize = 1 << 16;
/// Maximum total parity-factor visits in one estimate.
pub const MAX_EXPECTATION_WORK: usize = 1 << 24;
const EXPECTATION_SEED_DOMAIN: u64 = 0x5141_4D4F_4D49_4C45;
const EXPECTATION_GROUP_SEED_DOMAIN: u64 = 0x4752_4F55_505F_514D;
const EXPECTATION_COMMUTING_SEED_DOMAIN: u64 = 0x434F_4D4D_5F51_4D32;
const EXPECTATION_SEED_STEP: u32 = 0x7F4A_7C15;
/// Maximum Clifford gates materialized for one commuting measurement group.
pub const MAX_COMMUTING_MEASUREMENT_GATES: usize = 1 << 16;

/// One deterministic qubit-wise-commuting measurement group.
#[derive(Clone, Debug, PartialEq)]
pub struct PauliMeasurementGroup {
    basis: Vec<(u64, Pauli)>,
    term_indices: Vec<usize>,
    shots: BigUint,
    seed: u64,
}

impl PauliMeasurementGroup {
    /// Return the source-ordered non-identity basis factors.
    #[must_use]
    pub fn basis(&self) -> &[(u64, Pauli)] {
        &self.basis
    }

    /// Return canonical observable term positions measured by this group.
    #[must_use]
    pub fn term_indices(&self) -> &[usize] {
        &self.term_indices
    }

    /// Return the exact positive shot allocation for this group.
    #[must_use]
    pub const fn shots(&self) -> &BigUint {
        &self.shots
    }

    /// Return the endpoint-portable seed for this group.
    #[must_use]
    pub const fn seed(&self) -> u64 {
        self.seed
    }
}

/// One transient Rust-owned QWC measurement plan.
#[derive(Clone, Debug, PartialEq)]
pub struct PauliMeasurementPlan {
    groups: Vec<PauliMeasurementGroup>,
}

struct PendingMeasurementGroup {
    basis: BTreeMap<u64, Pauli>,
    term_indices: Vec<usize>,
}

impl PauliMeasurementPlan {
    /// Return deterministic measurement groups in first-fit order.
    #[must_use]
    pub fn groups(&self) -> &[PauliMeasurementGroup] {
        &self.groups
    }
}

/// One signed Z-product readout after a commuting Clifford diagonalization.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommutingPauliReadout {
    term_index: usize,
    factors: Vec<u64>,
    negative: bool,
}

impl CommutingPauliReadout {
    /// Return the canonical observable term position.
    #[must_use]
    pub const fn term_index(&self) -> usize {
        self.term_index
    }

    /// Return the ordered logical Z factors measured after diagonalization.
    #[must_use]
    pub fn factors(&self) -> &[u64] {
        &self.factors
    }

    /// Return whether the Clifford image is the negative Z product.
    #[must_use]
    pub const fn is_negative(&self) -> bool {
        self.negative
    }
}

/// One exact Clifford gate in a transient commuting-measurement plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CommutingMeasurementGate {
    Hadamard { target: u64 },
    Sdg { target: u64 },
    Cx { control: u64, target: u64 },
}

/// One deterministic generally commuting measurement group.
#[derive(Clone, Debug, PartialEq)]
pub struct CommutingPauliMeasurementGroup {
    term_indices: Vec<usize>,
    readouts: Vec<CommutingPauliReadout>,
    gates: Vec<CommutingMeasurementGate>,
    shots: BigUint,
    seed: u64,
}

impl CommutingPauliMeasurementGroup {
    /// Return canonical observable term positions measured by this group.
    #[must_use]
    pub fn term_indices(&self) -> &[usize] {
        &self.term_indices
    }

    /// Return signed Z-product readouts in canonical term order.
    #[must_use]
    pub fn readouts(&self) -> &[CommutingPauliReadout] {
        &self.readouts
    }

    /// Return the exact positive shot allocation for this group.
    #[must_use]
    pub const fn shots(&self) -> &BigUint {
        &self.shots
    }

    /// Return the endpoint-portable seed for this group.
    #[must_use]
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// Return the exact Clifford gate count materialized before measurement.
    #[must_use]
    pub const fn gate_count(&self) -> usize {
        self.gates.len()
    }
}

/// One transient Rust-owned generally commuting measurement plan.
#[derive(Clone, Debug, PartialEq)]
pub struct CommutingPauliMeasurementPlan {
    groups: Vec<CommutingPauliMeasurementGroup>,
}

impl CommutingPauliMeasurementPlan {
    /// Return deterministic measurement groups in first-fit order.
    #[must_use]
    pub fn groups(&self) -> &[CommutingPauliMeasurementGroup] {
        &self.groups
    }
}

/// Failure to plan Pauli measurements or read raw-sample parity.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum PauliExpectationError {
    /// The requested denominator is zero.
    #[error("Q7311: expectation shots must be positive")]
    InvalidShots,
    /// Factor coordinates are not a unique ordered subset of every row.
    #[error("Q7314: expectation factors must be unique, ordered, and within the sample width")]
    InvalidFactors,
    /// The requested aggregation exceeds an explicit work bound.
    #[error("Q7315: expectation aggregation exceeds its work budget")]
    Budget,
    /// Binary64 aggregation would produce a non-finite result.
    #[error("Q7316: expectation statistics exceed the finite binary64 range")]
    NumericalRange,
    /// QWC planning requires one canonical Hermitian observable.
    #[error("Q7317: grouped expectation requires a Hermitian observable")]
    NonHermitian,
    /// A total shot budget cannot assign a positive shot count to every group.
    #[error("Q7318: expectation total shots must cover every measurement group")]
    InsufficientShots,
    /// Group rows do not form one valid complete QWC term partition.
    #[error("Q7319: malformed grouped expectation samples")]
    InvalidGroups,
    /// The canonical observable could not be decoded.
    #[error(transparent)]
    Observable(#[from] crate::observable::ObservableError),
}

/// Derive portable random streams for Pauli-term measurements.
///
/// The source observable is already in canonical term order when this plan is
/// requested. Seed zero preserves the endpoint-default randomness contract.
/// Every nonzero base seed is mixed into a 32-bit start, then an odd wrapping
/// stride assigns distinct SDK-portable seeds to every supported term ordinal.
/// Runtime and output allocation are linear in `term_count`.
///
/// # Errors
///
/// Returns [`PauliExpectationError::Budget`] before allocation when the term
/// count exceeds [`MAX_EXPECTATION_TERMS`].
fn expectation_term_seeds(
    base_seed: u64,
    term_count: usize,
) -> Result<Vec<u64>, PauliExpectationError> {
    expectation_seeds(base_seed, term_count, EXPECTATION_SEED_DOMAIN)
}

fn expectation_seeds(
    base_seed: u64,
    count: usize,
    domain: u64,
) -> Result<Vec<u64>, PauliExpectationError> {
    if count > MAX_EXPECTATION_TERMS {
        return Err(PauliExpectationError::Budget);
    }
    if base_seed == 0 {
        return Ok(vec![0; count]);
    }
    let mixed = mix_seed(base_seed ^ domain).to_le_bytes();
    let start = u32::from_le_bytes([mixed[0], mixed[1], mixed[2], mixed[3]]);
    let mut seeds = Vec::with_capacity(count);
    for ordinal in 0..count {
        let ordinal = u32::try_from(ordinal).map_err(|_| PauliExpectationError::Budget)?;
        seeds.push(u64::from(
            start.wrapping_add(ordinal.wrapping_mul(EXPECTATION_SEED_STEP)),
        ));
    }
    Ok(seeds)
}

/// Plan deterministic first-fit qubit-wise-commuting measurements.
///
/// `total_shots` is divided exactly across the resulting groups. Every group
/// receives at least one shot, the first groups receive any integer remainder,
/// and nonzero group seeds occupy a domain distinct from independent-term
/// seeds. The plan is transient and depends only on the canonical observable.
///
/// # Errors
///
/// Returns [`PauliExpectationError`] when the observable is malformed or
/// non-Hermitian, grouping exceeds the shared work budget, or the total shot
/// budget cannot cover every group.
pub fn plan_qwc_pauli_expectation(
    observable: &[u8],
    total_shots: &BigUint,
    base_seed: u64,
) -> Result<PauliMeasurementPlan, PauliExpectationError> {
    plan_qwc_pauli_expectation_with_limit(observable, total_shots, base_seed, MAX_EXPECTATION_WORK)
}

/// Plan one measurement per term using a single total-shot budget.
///
/// # Errors
/// Returns the same bounded observable and allocation errors as QWC planning.
pub fn plan_independent_pauli_expectation(
    observable: &[u8],
    total_shots: &BigUint,
    base_seed: u64,
) -> Result<PauliMeasurementPlan, PauliExpectationError> {
    plan_local_pauli_expectation(
        observable,
        total_shots,
        base_seed,
        MAX_EXPECTATION_WORK,
        LocalGrouping::Independent,
    )
}

#[derive(Clone, Copy)]
enum LocalGrouping {
    Independent,
    Qwc,
}

/// Plan deterministic first-fit measurements for generally commuting Pauli terms.
///
/// Every group is accompanied by a Rust-derived Clifford diagonalization and a
/// signed Z-product readout for each canonical term. The plan is transient;
/// only the resulting ordinary gates and sampled program cross later compiler
/// stages.
///
/// # Errors
///
/// Returns [`PauliExpectationError`] when the observable, shot allocation,
/// grouping work, Clifford synthesis, or output gate budget is invalid.
pub fn plan_commuting_pauli_expectation(
    observable: &[u8],
    total_shots: &BigUint,
    base_seed: u64,
) -> Result<CommutingPauliMeasurementPlan, PauliExpectationError> {
    plan_commuting_pauli_expectation_with_limit(
        observable,
        total_shots,
        base_seed,
        MAX_EXPECTATION_WORK,
    )
}

fn plan_commuting_pauli_expectation_with_limit(
    observable: &[u8],
    total_shots: &BigUint,
    base_seed: u64,
    work_limit: usize,
) -> Result<CommutingPauliMeasurementPlan, PauliExpectationError> {
    if total_shots.is_zero() {
        return Err(PauliExpectationError::InvalidShots);
    }
    if total_shots.bits() > MAX_SAMPLE_COUNT_BITS {
        return Err(PauliExpectationError::Budget);
    }
    let observable = Observable::decode(observable)?;
    validate_grouped_observable(&observable)?;
    let terms = observable.terms.keys().collect::<Vec<_>>();
    let mut work = 0_usize;
    let mut pending = Vec::<Vec<usize>>::new();
    for (term_index, word) in terms.iter().enumerate() {
        charge_expectation_work(&mut work, word.len().max(1), work_limit)?;
        let mut selected = None;
        for (group_index, group) in pending.iter().enumerate() {
            let mut commutes = true;
            for other_index in group {
                let other = terms[*other_index];
                charge_expectation_work(
                    &mut work,
                    word.len().saturating_add(other.len()).max(1),
                    work_limit,
                )?;
                if !crate::observable::pauli_words_commute(word, other) {
                    commutes = false;
                    break;
                }
            }
            if commutes {
                selected = Some(group_index);
                break;
            }
        }
        let group_index = selected.unwrap_or_else(|| {
            pending.push(Vec::new());
            pending.len() - 1
        });
        pending[group_index].push(term_index);
    }

    if BigUint::from(pending.len()) > *total_shots {
        return Err(PauliExpectationError::InsufficientShots);
    }
    let divisor = BigUint::from(pending.len().max(1));
    let quotient = total_shots / &divisor;
    let remainder = (total_shots % &divisor)
        .to_usize()
        .ok_or(PauliExpectationError::Budget)?;
    let seeds = expectation_seeds(base_seed, pending.len(), EXPECTATION_COMMUTING_SEED_DOMAIN)?;
    let mut groups = Vec::with_capacity(pending.len());
    for (index, (term_indices, seed)) in pending.into_iter().zip(seeds).enumerate() {
        let (gates, readouts) =
            diagonalize_commuting_terms(&terms, &term_indices, &mut work, work_limit)?;
        groups.push(CommutingPauliMeasurementGroup {
            term_indices,
            readouts,
            gates,
            shots: &quotient + BigUint::from(index < remainder),
            seed,
        });
    }
    Ok(CommutingPauliMeasurementPlan { groups })
}

pub(crate) fn commuting_group_diagonalization(
    observable: &[u8],
    term_indices: &[usize],
) -> Result<(Vec<CommutingMeasurementGate>, Vec<CommutingPauliReadout>), PauliExpectationError> {
    let observable = Observable::decode(observable)?;
    validate_grouped_observable(&observable)?;
    let terms = observable.terms.keys().collect::<Vec<_>>();
    let mut work = 0_usize;
    diagonalize_commuting_terms(&terms, term_indices, &mut work, MAX_EXPECTATION_WORK)
}

fn validate_grouped_observable(observable: &Observable) -> Result<(), PauliExpectationError> {
    if observable.constant.im != 0.0
        || observable
            .terms
            .values()
            .any(|coefficient| coefficient.im != 0.0)
    {
        return Err(PauliExpectationError::NonHermitian);
    }
    if observable.terms.len() > MAX_EXPECTATION_TERMS {
        return Err(PauliExpectationError::Budget);
    }
    Ok(())
}

fn charge_expectation_work(
    work: &mut usize,
    amount: usize,
    limit: usize,
) -> Result<(), PauliExpectationError> {
    *work = work
        .checked_add(amount)
        .filter(|value| *value <= limit)
        .ok_or(PauliExpectationError::Budget)?;
    Ok(())
}

#[derive(Clone)]
struct SymplecticRow {
    x: Vec<bool>,
    z: Vec<bool>,
    negative: bool,
}

impl SymplecticRow {
    fn bit(&self, column: usize) -> bool {
        if column < self.x.len() {
            self.x[column]
        } else {
            self.z[column - self.x.len()]
        }
    }

    fn xor_assign(&mut self, other: &Self) {
        for (left, right) in self.x.iter_mut().zip(&other.x) {
            *left ^= right;
        }
        for (left, right) in self.z.iter_mut().zip(&other.z) {
            *left ^= right;
        }
    }

    fn apply_h(&mut self, target: usize) {
        self.negative ^= self.x[target] && self.z[target];
        std::mem::swap(&mut self.x[target], &mut self.z[target]);
    }

    fn apply_sdg(&mut self, target: usize) {
        self.negative ^= self.x[target] && !self.z[target];
        self.z[target] ^= self.x[target];
    }

    fn apply_cx(&mut self, control: usize, target: usize) {
        self.negative ^= self.x[control] && self.z[target] && (self.x[target] == self.z[control]);
        self.x[target] ^= self.x[control];
        self.z[control] ^= self.z[target];
    }
}

fn diagonalize_commuting_terms(
    terms: &[&crate::observable::PauliWord],
    term_indices: &[usize],
    work: &mut usize,
    work_limit: usize,
) -> Result<(Vec<CommutingMeasurementGate>, Vec<CommutingPauliReadout>), PauliExpectationError> {
    if term_indices.is_empty()
        || term_indices.windows(2).any(|pair| pair[0] >= pair[1])
        || term_indices.iter().any(|index| *index >= terms.len())
    {
        return Err(PauliExpectationError::InvalidGroups);
    }
    for (offset, left_index) in term_indices.iter().enumerate() {
        for right_index in &term_indices[offset + 1..] {
            charge_expectation_work(
                work,
                terms[*left_index]
                    .len()
                    .saturating_add(terms[*right_index].len())
                    .max(1),
                work_limit,
            )?;
            if !crate::observable::pauli_words_commute(terms[*left_index], terms[*right_index]) {
                return Err(PauliExpectationError::InvalidGroups);
            }
        }
    }
    let active_lanes = term_indices
        .iter()
        .flat_map(|index| terms[*index].iter().map(|(lane, _)| *lane))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let lane_positions = active_lanes
        .iter()
        .enumerate()
        .map(|(position, lane)| (*lane, position))
        .collect::<BTreeMap<_, _>>();
    let width = active_lanes.len();
    charge_expectation_work(
        work,
        width
            .checked_mul(term_indices.len().saturating_add(width))
            .ok_or(PauliExpectationError::Budget)?,
        work_limit,
    )?;
    let mut originals = term_indices
        .iter()
        .map(|index| symplectic_row(terms[*index], &lane_positions, width))
        .collect::<Result<Vec<_>, _>>()?;
    let mut basis = independent_rows(&originals, work, work_limit)?;
    let gates = synthesize_commuting_diagonalization(
        &active_lanes,
        &mut basis,
        &mut originals,
        work,
        work_limit,
    )?;
    let readouts = commuting_readouts(term_indices, originals, &active_lanes)?;
    Ok((gates, readouts))
}

fn synthesize_commuting_diagonalization(
    active_lanes: &[u64],
    basis: &mut [SymplecticRow],
    originals: &mut [SymplecticRow],
    work: &mut usize,
    work_limit: usize,
) -> Result<Vec<CommutingMeasurementGate>, PauliExpectationError> {
    let width = active_lanes.len();
    let mut gates = Vec::new();
    for row_index in 0..basis.len() {
        for (target, logical_target) in active_lanes.iter().copied().enumerate() {
            match (basis[row_index].x[target], basis[row_index].z[target]) {
                (true, false) => append_commuting_gate(
                    CommutingMeasurementGate::Hadamard {
                        target: logical_target,
                    },
                    target,
                    None,
                    basis,
                    originals,
                    &mut gates,
                    work,
                    work_limit,
                )?,
                (true, true) => {
                    append_commuting_gate(
                        CommutingMeasurementGate::Sdg {
                            target: logical_target,
                        },
                        target,
                        None,
                        basis,
                        originals,
                        &mut gates,
                        work,
                        work_limit,
                    )?;
                    append_commuting_gate(
                        CommutingMeasurementGate::Hadamard {
                            target: logical_target,
                        },
                        target,
                        None,
                        basis,
                        originals,
                        &mut gates,
                        work,
                        work_limit,
                    )?;
                }
                _ => {}
            }
        }
        let pivot = basis[row_index]
            .z
            .iter()
            .position(|value| *value)
            .ok_or(PauliExpectationError::InvalidGroups)?;
        let support = basis[row_index]
            .z
            .iter()
            .enumerate()
            .filter_map(|(position, value)| (*value && position != pivot).then_some(position))
            .collect::<Vec<_>>();
        for control in support {
            append_commuting_gate(
                CommutingMeasurementGate::Cx {
                    control: active_lanes[control],
                    target: active_lanes[pivot],
                },
                pivot,
                Some(control),
                basis,
                originals,
                &mut gates,
                work,
                work_limit,
            )?;
        }
        if basis[row_index].x.iter().any(|value| *value)
            || basis[row_index]
                .z
                .iter()
                .enumerate()
                .any(|(position, value)| *value && position != pivot)
        {
            return Err(PauliExpectationError::InvalidGroups);
        }
        let pivot_row = basis[row_index].clone();
        for other in basis.iter_mut().skip(row_index + 1) {
            if other.z[pivot] {
                charge_expectation_work(work, 2 * width, work_limit)?;
                other.xor_assign(&pivot_row);
            }
        }
    }
    Ok(gates)
}

fn commuting_readouts(
    term_indices: &[usize],
    originals: Vec<SymplecticRow>,
    active_lanes: &[u64],
) -> Result<Vec<CommutingPauliReadout>, PauliExpectationError> {
    let mut readouts = Vec::with_capacity(originals.len());
    for (term_index, row) in term_indices.iter().copied().zip(originals) {
        if row.x.iter().any(|value| *value) {
            return Err(PauliExpectationError::InvalidGroups);
        }
        let factors = row
            .z
            .iter()
            .enumerate()
            .filter_map(|(position, value)| value.then_some(active_lanes[position]))
            .collect::<Vec<_>>();
        if factors.is_empty() {
            return Err(PauliExpectationError::InvalidGroups);
        }
        readouts.push(CommutingPauliReadout {
            term_index,
            factors,
            negative: row.negative,
        });
    }
    Ok(readouts)
}

fn symplectic_row(
    word: &[(u64, Pauli)],
    lane_positions: &BTreeMap<u64, usize>,
    width: usize,
) -> Result<SymplecticRow, PauliExpectationError> {
    let mut row = SymplecticRow {
        x: vec![false; width],
        z: vec![false; width],
        negative: false,
    };
    for (lane, pauli) in word {
        let position = *lane_positions
            .get(lane)
            .ok_or(PauliExpectationError::InvalidGroups)?;
        match pauli {
            Pauli::X => row.x[position] = true,
            Pauli::Y => {
                row.x[position] = true;
                row.z[position] = true;
            }
            Pauli::Z => row.z[position] = true,
        }
    }
    Ok(row)
}

fn independent_rows(
    rows: &[SymplecticRow],
    work: &mut usize,
    work_limit: usize,
) -> Result<Vec<SymplecticRow>, PauliExpectationError> {
    let width = rows.first().map_or(0, |row| row.x.len());
    let mut echelon = vec![None::<SymplecticRow>; 2 * width];
    let mut basis = Vec::new();
    for row in rows {
        let mut reduced = row.clone();
        reduced.negative = false;
        for (column, slot) in echelon.iter_mut().enumerate() {
            if !reduced.bit(column) {
                continue;
            }
            if let Some(pivot) = slot.as_ref() {
                charge_expectation_work(work, 2 * width, work_limit)?;
                reduced.xor_assign(pivot);
            } else {
                *slot = Some(reduced.clone());
                basis.push(reduced);
                break;
            }
        }
    }
    Ok(basis)
}

#[allow(clippy::too_many_arguments)]
fn append_commuting_gate(
    gate: CommutingMeasurementGate,
    target: usize,
    control: Option<usize>,
    basis: &mut [SymplecticRow],
    originals: &mut [SymplecticRow],
    gates: &mut Vec<CommutingMeasurementGate>,
    work: &mut usize,
    work_limit: usize,
) -> Result<(), PauliExpectationError> {
    if gates.len() >= MAX_COMMUTING_MEASUREMENT_GATES {
        return Err(PauliExpectationError::Budget);
    }
    charge_expectation_work(
        work,
        basis.len().saturating_add(originals.len()).max(1),
        work_limit,
    )?;
    for row in basis.iter_mut().chain(originals) {
        match control {
            None if matches!(gate, CommutingMeasurementGate::Hadamard { .. }) => {
                row.apply_h(target);
            }
            None => row.apply_sdg(target),
            Some(control) => row.apply_cx(control, target),
        }
    }
    gates.push(gate);
    Ok(())
}

fn plan_qwc_pauli_expectation_with_limit(
    observable: &[u8],
    total_shots: &BigUint,
    base_seed: u64,
    work_limit: usize,
) -> Result<PauliMeasurementPlan, PauliExpectationError> {
    plan_local_pauli_expectation(
        observable,
        total_shots,
        base_seed,
        work_limit,
        LocalGrouping::Qwc,
    )
}

fn plan_local_pauli_expectation(
    observable: &[u8],
    total_shots: &BigUint,
    base_seed: u64,
    work_limit: usize,
    grouping: LocalGrouping,
) -> Result<PauliMeasurementPlan, PauliExpectationError> {
    if total_shots.is_zero() {
        return Err(PauliExpectationError::InvalidShots);
    }
    if total_shots.bits() > MAX_SAMPLE_COUNT_BITS {
        return Err(PauliExpectationError::Budget);
    }
    let observable = Observable::decode(observable)?;
    validate_grouped_observable(&observable)?;

    let mut work = 0_usize;
    let mut groups = Vec::<PendingMeasurementGroup>::new();
    for (term_index, word) in observable.terms.keys().enumerate() {
        work = work
            .checked_add(word.len())
            .filter(|value| *value <= work_limit)
            .ok_or(PauliExpectationError::Budget)?;
        let mut selected = None;
        for (group_index, group) in groups.iter().enumerate().take(match grouping {
            LocalGrouping::Qwc => groups.len(),
            LocalGrouping::Independent => 0,
        }) {
            work = work
                .checked_add(word.len().max(1))
                .filter(|value| *value <= work_limit)
                .ok_or(PauliExpectationError::Budget)?;
            if word_is_qwc_with_basis(&group.basis, word) {
                selected = Some(group_index);
                break;
            }
        }
        let group_index = selected.unwrap_or_else(|| {
            groups.push(PendingMeasurementGroup {
                basis: BTreeMap::new(),
                term_indices: Vec::new(),
            });
            groups.len() - 1
        });
        let group = &mut groups[group_index];
        group.basis.extend(word.iter().copied());
        group.term_indices.push(term_index);
    }

    if BigUint::from(groups.len()) > *total_shots {
        return Err(PauliExpectationError::InsufficientShots);
    }
    let divisor = BigUint::from(groups.len().max(1));
    let quotient = total_shots / &divisor;
    let remainder = (total_shots % &divisor)
        .to_usize()
        .ok_or(PauliExpectationError::Budget)?;
    let seeds = match grouping {
        LocalGrouping::Qwc => {
            expectation_seeds(base_seed, groups.len(), EXPECTATION_GROUP_SEED_DOMAIN)?
        }
        LocalGrouping::Independent => expectation_term_seeds(base_seed, groups.len())?,
    };
    Ok(PauliMeasurementPlan {
        groups: groups
            .into_iter()
            .zip(seeds)
            .enumerate()
            .map(|(index, (group, seed))| PauliMeasurementGroup {
                basis: group.basis.into_iter().collect(),
                term_indices: group.term_indices,
                shots: &quotient + BigUint::from(index < remainder),
                seed,
            })
            .collect(),
    })
}

/// Re-derive one QWC basis from canonical observable term positions.
pub(crate) fn qwc_group_basis(
    observable: &[u8],
    term_indices: &[usize],
) -> Result<Vec<(u64, Pauli)>, PauliExpectationError> {
    let observable = Observable::decode(observable)?;
    if term_indices.is_empty()
        || term_indices.windows(2).any(|pair| pair[0] >= pair[1])
        || observable.constant.im != 0.0
        || observable
            .terms
            .values()
            .any(|coefficient| coefficient.im != 0.0)
    {
        return Err(PauliExpectationError::InvalidGroups);
    }
    let terms = observable.terms.keys().collect::<Vec<_>>();
    let mut basis = BTreeMap::<u64, Pauli>::new();
    for term_index in term_indices {
        let pauli_word = terms
            .get(*term_index)
            .ok_or(PauliExpectationError::InvalidGroups)?;
        if !extend_qwc_basis(&mut basis, pauli_word) {
            return Err(PauliExpectationError::InvalidGroups);
        }
    }
    Ok(basis.into_iter().collect())
}

fn word_is_qwc_with_basis(basis: &BTreeMap<u64, Pauli>, word: &[(u64, Pauli)]) -> bool {
    word.iter().all(|(index, pauli)| {
        basis
            .get(index)
            .is_none_or(|basis_pauli| basis_pauli == pauli)
    })
}

fn extend_qwc_basis(basis: &mut BTreeMap<u64, Pauli>, word: &[(u64, Pauli)]) -> bool {
    if !word_is_qwc_with_basis(basis, word) {
        return false;
    }
    basis.extend(word.iter().copied());
    true
}

const fn mix_seed(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

/// Estimate one Pauli-Z product directly from a canonical sample summary.
///
/// # Errors
///
/// Returns [`PauliExpectationError`] when factor coordinates or the bounded
/// histogram computation are invalid.
pub fn estimate_sample_parity(
    counts: &SampleCountSummary,
    factors: &[usize],
) -> Result<f64, PauliExpectationError> {
    // SampleCountSummary already guarantees nonempty uniform bit rows, bounded
    // counts, and a positive total. Only this readout's coordinates and work
    // need validation here.
    let width = counts.most_likely().len();
    if factors.is_empty()
        || factors.windows(2).any(|pair| pair[0] >= pair[1])
        || factors.iter().any(|index| *index >= width)
    {
        return Err(PauliExpectationError::InvalidFactors);
    }
    let work = counts
        .entries()
        .len()
        .checked_mul(
            factors
                .len()
                .checked_add(1)
                .ok_or(PauliExpectationError::Budget)?,
        )
        .ok_or(PauliExpectationError::Budget)?;
    if work > MAX_EXPECTATION_WORK {
        return Err(PauliExpectationError::Budget);
    }

    let mut signed = BigInt::from(0_u8);
    for (bits, count) in counts.entries() {
        let odd = factors
            .iter()
            .fold(0_u8, |parity, index| parity ^ bits[*index]);
        let signed_count = BigInt::from(count.clone());
        if odd == 0 {
            signed += signed_count;
        } else {
            signed -= signed_count;
        }
    }
    bigint_ratio(&signed, counts.shots())
        .filter(|value| value.is_finite())
        .ok_or(PauliExpectationError::NumericalRange)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observable(width: usize, terms: &[(Vec<(usize, Pauli)>, f64)]) -> Vec<u8> {
        crate::normalize_real_observable(width, 0.0, terms)
            .expect("observable")
            .literal()
            .to_vec()
    }

    #[test]
    fn term_seed_plan_is_stable_distinct_and_bounded() {
        let seeds = expectation_term_seeds(7, 4).expect("seed plan");

        assert_eq!(expectation_term_seeds(7, 4), Ok(seeds.clone()));
        let unique = seeds
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(unique.len(), seeds.len());
        assert!(seeds.iter().all(|seed| u32::try_from(*seed).is_ok()));
        assert_eq!(expectation_term_seeds(0, 4), Ok(vec![0; 4]));
        assert_eq!(expectation_term_seeds(7, 0), Ok(Vec::new()));
        assert_eq!(
            expectation_term_seeds(7, MAX_EXPECTATION_TERMS + 1),
            Err(PauliExpectationError::Budget)
        );
    }

    #[test]
    fn independent_plan_uses_total_budget_and_rejects_short_budget() {
        let observable = observable(2, &[(vec![(0, Pauli::Z)], 1.0), (vec![(1, Pauli::Z)], 2.0)]);
        let plan =
            plan_independent_pauli_expectation(&observable, &BigUint::from(5_u8), 7).expect("plan");
        assert_eq!(plan.groups().len(), 2);
        assert_eq!(plan.groups()[0].shots(), &BigUint::from(3_u8));
        assert_eq!(plan.groups()[1].shots(), &BigUint::from(2_u8));
        assert_eq!(plan.groups()[0].term_indices(), &[0]);
        assert_eq!(plan.groups()[1].term_indices(), &[1]);
        assert_eq!(
            plan.groups()
                .iter()
                .map(PauliMeasurementGroup::seed)
                .collect::<Vec<_>>(),
            expectation_term_seeds(7, 2).expect("seeds")
        );
        assert_eq!(
            plan_independent_pauli_expectation(&observable, &BigUint::from(1_u8), 7),
            Err(PauliExpectationError::InsufficientShots)
        );
        assert_eq!(
            plan_independent_pauli_expectation(&observable, &BigUint::from(0_u8), 7),
            Err(PauliExpectationError::InvalidShots)
        );
        assert_eq!(
            plan_independent_pauli_expectation(
                &observable,
                &(BigUint::from(1_u8) << MAX_SAMPLE_COUNT_BITS),
                7
            ),
            Err(PauliExpectationError::Budget)
        );
    }

    #[test]
    fn qwc_plan_is_deterministic_and_allocates_one_total_shot_budget() {
        let observable = observable(
            2,
            &[
                (vec![(0, Pauli::X)], 1.0),
                (vec![(0, Pauli::Z)], 2.0),
                (vec![(1, Pauli::Z)], 3.0),
            ],
        );
        let plan =
            plan_qwc_pauli_expectation(&observable, &BigUint::from(5_u8), 7).expect("QWC plan");

        assert_eq!(plan.groups().len(), 2);
        assert_eq!(plan.groups()[0].basis(), &[(0, Pauli::X), (1, Pauli::Z)]);
        assert_eq!(plan.groups()[0].term_indices(), &[0, 2]);
        assert_eq!(plan.groups()[0].shots(), &BigUint::from(3_u8));
        assert_eq!(plan.groups()[1].basis(), &[(0, Pauli::Z)]);
        assert_eq!(plan.groups()[1].term_indices(), &[1]);
        assert_eq!(plan.groups()[1].shots(), &BigUint::from(2_u8));
        assert_ne!(plan.groups()[0].seed(), plan.groups()[1].seed());
        assert_ne!(
            plan.groups()[0].seed(),
            expectation_term_seeds(7, 2).expect("term seeds")[0]
        );
        assert_eq!(
            plan_qwc_pauli_expectation(&observable, &BigUint::from(1_u8), 7),
            Err(PauliExpectationError::InsufficientShots)
        );
    }

    #[test]
    fn qwc_plan_checks_its_exact_grouping_work_boundary() {
        let observable = observable(1, &[(vec![(0, Pauli::X)], 1.0), (vec![(0, Pauli::Z)], 1.0)]);

        assert!(
            plan_qwc_pauli_expectation_with_limit(&observable, &BigUint::from(2_u8), 0, 3,).is_ok()
        );
        assert_eq!(
            plan_qwc_pauli_expectation_with_limit(&observable, &BigUint::from(2_u8), 0, 2,),
            Err(PauliExpectationError::Budget)
        );
    }

    #[test]
    fn commuting_plan_diagonalizes_a_non_qwc_group() {
        let observable = observable(
            2,
            &[
                (vec![(0, Pauli::X), (1, Pauli::X)], 1.0),
                (vec![(0, Pauli::Y), (1, Pauli::Y)], -1.0),
                (vec![(0, Pauli::Z), (1, Pauli::Z)], 1.0),
            ],
        );
        let plan = plan_commuting_pauli_expectation(&observable, &BigUint::from(4_u8), 7)
            .expect("commuting plan");

        assert_eq!(plan.groups().len(), 1);
        let group = &plan.groups()[0];
        assert_eq!(group.term_indices(), &[0, 1, 2]);
        assert_eq!(group.readouts().len(), 3);
        assert!(group.gate_count() > 0);
        assert_eq!(group.shots(), &BigUint::from(4_u8));
        assert!(
            group
                .readouts()
                .iter()
                .all(|readout| !readout.factors().is_empty())
        );
        assert!(
            group
                .readouts()
                .iter()
                .any(CommutingPauliReadout::is_negative)
        );
    }

    #[test]
    fn commuting_plan_separates_noncommuting_terms_and_bounds_work() {
        let observable = observable(1, &[(vec![(0, Pauli::X)], 1.0), (vec![(0, Pauli::Z)], 1.0)]);

        let plan = plan_commuting_pauli_expectation(&observable, &BigUint::from(2_u8), 0)
            .expect("two groups");
        assert_eq!(plan.groups().len(), 2);
        assert_eq!(
            plan_commuting_pauli_expectation_with_limit(&observable, &BigUint::from(2_u8), 0, 0,),
            Err(PauliExpectationError::Budget)
        );
    }

    #[test]
    fn clifford_sign_updates_cover_y_and_cnot_phases() {
        let mut hadamard_y = SymplecticRow {
            x: vec![true],
            z: vec![true],
            negative: false,
        };
        hadamard_y.apply_h(0);
        assert!(hadamard_y.x[0] && hadamard_y.z[0] && hadamard_y.negative);

        let mut sdg_x = SymplecticRow {
            x: vec![true],
            z: vec![false],
            negative: false,
        };
        sdg_x.apply_sdg(0);
        assert!(sdg_x.x[0] && sdg_x.z[0] && sdg_x.negative);

        let mut cross = SymplecticRow {
            x: vec![true, false],
            z: vec![false, true],
            negative: false,
        };
        cross.apply_cx(0, 1);
        assert_eq!(cross.x, vec![true, true]);
        assert_eq!(cross.z, vec![true, true]);
        assert!(cross.negative);
    }

    #[test]
    fn diagonalizer_covers_every_two_qubit_pauli_pair() {
        let words = (1_u8..16)
            .map(|code| {
                (0_u64..2)
                    .filter_map(|lane| {
                        let coordinate = (code >> (2 * lane)) & 3;
                        let pauli = match coordinate {
                            0 => return None,
                            1 => Pauli::X,
                            2 => Pauli::Y,
                            _ => Pauli::Z,
                        };
                        Some((lane, pauli))
                    })
                    .collect::<crate::observable::PauliWord>()
            })
            .collect::<Vec<_>>();

        for left in 0..words.len() {
            for right in left + 1..words.len() {
                let terms = vec![&words[left], &words[right]];
                let mut work = 0;
                let result =
                    diagonalize_commuting_terms(&terms, &[0, 1], &mut work, MAX_EXPECTATION_WORK);
                if crate::observable::pauli_words_commute(&words[left], &words[right]) {
                    let (_, readouts) = result.expect("commuting pair");
                    assert_eq!(readouts.len(), 2);
                    assert!(readouts.iter().all(|readout| !readout.factors().is_empty()));
                } else {
                    assert_eq!(result, Err(PauliExpectationError::InvalidGroups));
                }
            }
        }
    }

    #[test]
    fn sample_summary_parity_retains_arbitrary_precision_counts() {
        let huge = BigInt::from(1_u8) << 100_000_usize;
        let summary = crate::sample_counts::normalize_sample_counts(
            &[
                (vec![0].into_boxed_slice(), huge),
                (vec![1].into_boxed_slice(), BigInt::from(1_u8)),
            ],
            Some(1),
        )
        .expect("sample summary");

        let estimate = estimate_sample_parity(&summary, &[0]).expect("parity estimate");

        assert_eq!(estimate.to_bits(), 1.0_f64.to_bits());
    }
}
