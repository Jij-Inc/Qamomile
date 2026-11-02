//! Bounded late lowering for exact evolution under commuting Pauli sums.
//!
//! `QCore` retains one abstract `quantum.pauli_evolution` operation. This module
//! owns the target-neutral validation and gate schedule used by concrete
//! circuit emitters, so Python adapters never reimplement compiler semantics.

use prost::Message;
use thiserror::Error;

use crate::lowering::realization::RealizationCapabilities;
use crate::observable::{Observable, ObservableError, Pauli, words_commute};

/// Maximum emitted gates in one exact Pauli-evolution schedule.
pub const MAX_PAULI_EVOLUTION_STEPS: usize = 250_000;

/// Maximum factor visits admitted by pairwise commutation validation.
pub const MAX_PAULI_EVOLUTION_COMMUTATION_WORK: usize = 50_000_000;

/// One target-neutral gate in an exact Pauli-evolution schedule.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PauliEvolutionGate {
    /// Hadamard basis change.
    H,
    /// Phase gate.
    S,
    /// Adjoint phase gate.
    Sdg,
    /// Controlled-X parity accumulation.
    Cx,
    /// Z rotation whose angle is `scale * time`.
    Rz,
    /// Two-qubit ZZ rotation whose angle is `scale * time`.
    Rzz,
    /// Global phase whose angle is `scale * time`.
    GlobalPhase,
}

impl PauliEvolutionGate {
    /// Return the stable backend-neutral gate spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::H => "h",
            Self::S => "s",
            Self::Sdg => "sdg",
            Self::Cx => "cx",
            Self::Rz => "rz",
            Self::Rzz => "rzz",
            Self::GlobalPhase => "global_phase",
        }
    }
}

/// One gate and its logical selectors in a Pauli-evolution schedule.
#[derive(Clone, Debug, PartialEq)]
pub struct PauliEvolutionStep {
    gate: PauliEvolutionGate,
    indices: Box<[u64]>,
    scale: f64,
}

impl PauliEvolutionStep {
    /// Return the gate kind.
    #[must_use]
    pub const fn gate(&self) -> PauliEvolutionGate {
        self.gate
    }

    /// Return logical qubit selectors.
    #[must_use]
    pub fn indices(&self) -> &[u64] {
        &self.indices
    }

    /// Return the multiplier applied to the evolution time.
    #[must_use]
    pub const fn scale(&self) -> f64 {
        self.scale
    }
}

/// Failure to validate or lower one exact Pauli evolution.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum PauliEvolutionError {
    /// The observable literal is invalid or noncanonical.
    #[error(transparent)]
    Observable(#[from] ObservableError),
    /// Exact evolution requires real coefficients.
    #[error("Q8036: Pauli evolution requires a Hermitian observable")]
    NonHermitian,
    /// The exact factorization is valid only for commuting terms.
    #[error("Q8037: exact Pauli evolution requires pairwise-commuting terms")]
    NonCommuting,
    /// The observable addresses a logical qubit outside the supplied state.
    #[error("Q8038: Hamiltonian width {observable_width} exceeds state width {state_width}")]
    Width {
        /// Minimum logical width declared by the observable.
        observable_width: u64,
        /// Available state width.
        state_width: u64,
    },
    /// Pairwise validation would exceed its explicit polynomial work budget.
    #[error("Q8039: Pauli evolution exceeds the commutation work budget")]
    CommutationBudget,
    /// The concrete schedule would exceed its explicit output budget.
    #[error("Q8040: Pauli evolution exceeds the schedule budget")]
    ScheduleBudget,
}

/// Produce an exact schedule using only explicitly supported realizations.
///
/// The payload is a canonical observable literal. Each angle is a scale
/// multiplied by a literal or runtime evolution time at emission.
///
/// # Errors
///
/// Returns [`PauliEvolutionError`] for malformed or noncanonical observables,
/// complex coefficients, noncommuting terms, width mismatches, or excessive
/// validation/output work.
pub fn plan_pauli_evolution_with_capabilities(
    payload: &[u8],
    state_width: u64,
    capabilities: RealizationCapabilities,
) -> Result<Vec<PauliEvolutionStep>, PauliEvolutionError> {
    let observable = validated_observable(payload, state_width)?;
    let step_count = pauli_evolution_step_count(&observable, capabilities)?;

    let mut steps = Vec::with_capacity(step_count);
    if observable.constant.re != 0.0 {
        steps.push(step(
            PauliEvolutionGate::GlobalPhase,
            [],
            -observable.constant.re,
        ));
    }
    for (word, coefficient) in &observable.terms {
        if capabilities.supports_rzz() && is_two_qubit_zz(word) {
            steps.push(step(
                PauliEvolutionGate::Rzz,
                [word[0].0, word[1].0],
                2.0 * coefficient.re,
            ));
            continue;
        }
        for &(index, pauli) in word {
            match pauli {
                Pauli::X => steps.push(step(PauliEvolutionGate::H, [index], 1.0)),
                Pauli::Y => {
                    steps.push(step(PauliEvolutionGate::Sdg, [index], 1.0));
                    steps.push(step(PauliEvolutionGate::H, [index], 1.0));
                }
                Pauli::Z => {}
            }
        }
        for pair in word.windows(2) {
            steps.push(step(PauliEvolutionGate::Cx, [pair[0].0, pair[1].0], 1.0));
        }
        let target = word
            .last()
            .ok_or(ObservableError::Malformed(
                "normalized Pauli terms must be nonempty",
            ))?
            .0;
        steps.push(step(PauliEvolutionGate::Rz, [target], 2.0 * coefficient.re));
        for pair in word.windows(2).rev() {
            steps.push(step(PauliEvolutionGate::Cx, [pair[0].0, pair[1].0], 1.0));
        }
        for &(index, pauli) in word.iter().rev() {
            match pauli {
                Pauli::X => steps.push(step(PauliEvolutionGate::H, [index], 1.0)),
                Pauli::Y => {
                    steps.push(step(PauliEvolutionGate::H, [index], 1.0));
                    steps.push(step(PauliEvolutionGate::S, [index], 1.0));
                }
                Pauli::Z => {}
            }
        }
    }
    debug_assert_eq!(steps.len(), step_count);
    Ok(steps)
}

/// Validate exact evolution without generating a decomposition.
///
/// # Errors
/// Rejects noncanonical, non-Hermitian, noncommuting or out-of-width input.
pub fn validate_pauli_evolution(
    payload: &[u8],
    state_width: u64,
) -> Result<(), PauliEvolutionError> {
    validated_observable(payload, state_width).map(|_| ())
}

fn validated_observable(
    payload: &[u8],
    state_width: u64,
) -> Result<Observable, PauliEvolutionError> {
    let observable = Observable::decode(payload)?;
    if observable.encode().encode_to_vec() != payload {
        return Err(ObservableError::NonCanonical.into());
    }
    if observable.constant.im != 0.0
        || observable
            .terms
            .values()
            .any(|coefficient| coefficient.im != 0.0)
    {
        return Err(PauliEvolutionError::NonHermitian);
    }
    if observable.num_qubits > state_width {
        return Err(PauliEvolutionError::Width {
            observable_width: observable.num_qubits,
            state_width,
        });
    }

    let words = observable.terms.keys().collect::<Vec<_>>();
    let mut commutation_work = 0_usize;
    for (position, left) in words.iter().enumerate() {
        for right in &words[position + 1..] {
            commutation_work = commutation_work
                .checked_add(left.len())
                .and_then(|work| work.checked_add(right.len()))
                .filter(|work| *work <= MAX_PAULI_EVOLUTION_COMMUTATION_WORK)
                .ok_or(PauliEvolutionError::CommutationBudget)?;
            if !words_commute(left, right) {
                return Err(PauliEvolutionError::NonCommuting);
            }
        }
    }
    Ok(observable)
}

fn is_two_qubit_zz(word: &[(u64, Pauli)]) -> bool {
    matches!(word, [(_, Pauli::Z), (_, Pauli::Z)])
}

fn pauli_evolution_step_count(
    observable: &Observable,
    capabilities: RealizationCapabilities,
) -> Result<usize, PauliEvolutionError> {
    let mut step_count = usize::from(observable.constant.re != 0.0);
    for word in observable.terms.keys() {
        if capabilities.supports_rzz() && is_two_qubit_zz(word) {
            step_count = step_count
                .checked_add(1)
                .filter(|count| *count <= MAX_PAULI_EVOLUTION_STEPS)
                .ok_or(PauliEvolutionError::ScheduleBudget)?;
            continue;
        }
        let basis_steps = word.iter().try_fold(0_usize, |count, (_, pauli)| {
            count.checked_add(match pauli {
                Pauli::X => 1,
                Pauli::Y => 2,
                Pauli::Z => 0,
            })
        });
        let parity_steps = word
            .len()
            .checked_sub(1)
            .and_then(|value| value.checked_mul(2));
        let basis_steps = basis_steps
            .and_then(|value| value.checked_mul(2))
            .ok_or(PauliEvolutionError::ScheduleBudget)?;
        step_count = step_count
            .checked_add(basis_steps)
            .and_then(|count| count.checked_add(parity_steps?))
            .and_then(|count| count.checked_add(1))
            .filter(|count| *count <= MAX_PAULI_EVOLUTION_STEPS)
            .ok_or(PauliEvolutionError::ScheduleBudget)?;
    }
    Ok(step_count)
}

fn step<const N: usize>(
    gate: PauliEvolutionGate,
    indices: [u64; N],
    scale: f64,
) -> PauliEvolutionStep {
    PauliEvolutionStep {
        gate,
        indices: Vec::from(indices).into_boxed_slice(),
        scale,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use num_complex::Complex64;
    use prost::Message;

    use super::*;

    fn observable(
        num_qubits: u64,
        constant: Complex64,
        terms: impl IntoIterator<Item = (Vec<(u64, Pauli)>, Complex64)>,
    ) -> Vec<u8> {
        Observable {
            terms: terms.into_iter().collect::<BTreeMap<_, _>>(),
            constant,
            num_qubits,
        }
        .encode()
        .encode_to_vec()
    }

    fn projected(steps: &[PauliEvolutionStep]) -> Vec<(&'static str, Vec<u64>, f64)> {
        steps
            .iter()
            .map(|step| (step.gate().as_str(), step.indices().to_vec(), step.scale()))
            .collect()
    }

    #[test]
    fn shares_one_exact_xyz_and_global_phase_schedule() {
        let literal = observable(
            3,
            Complex64::new(0.25, 0.0),
            [
                (vec![(0, Pauli::X)], Complex64::new(0.5, 0.0)),
                (
                    vec![(1, Pauli::Y), (2, Pauli::Z)],
                    Complex64::new(-0.75, 0.0),
                ),
            ],
        );
        let steps =
            plan_pauli_evolution_with_capabilities(&literal, 3, RealizationCapabilities::EMPTY)
                .expect("valid commuting evolution");

        assert_eq!(
            projected(&steps),
            vec![
                ("global_phase", vec![], -0.25),
                ("h", vec![0], 1.0),
                ("rz", vec![0], 1.0),
                ("h", vec![0], 1.0),
                ("sdg", vec![1], 1.0),
                ("h", vec![1], 1.0),
                ("cx", vec![1, 2], 1.0),
                ("rz", vec![2], -1.5),
                ("cx", vec![1, 2], 1.0),
                ("h", vec![1], 1.0),
                ("s", vec![1], 1.0),
            ]
        );
    }

    #[test]
    fn selects_rzz_only_for_an_explicit_exact_capability() {
        let literal = observable(
            2,
            Complex64::new(0.0, 0.0),
            [(
                vec![(0, Pauli::Z), (1, Pauli::Z)],
                Complex64::new(-0.75, 0.0),
            )],
        );

        let fallback =
            plan_pauli_evolution_with_capabilities(&literal, 2, RealizationCapabilities::EMPTY)
                .expect("parity ladder");
        let optimized =
            plan_pauli_evolution_with_capabilities(&literal, 2, RealizationCapabilities::RZZ)
                .expect("RZZ realization");

        assert_eq!(
            projected(&fallback),
            vec![
                ("cx", vec![0, 1], 1.0),
                ("rz", vec![1], -1.5),
                ("cx", vec![0, 1], 1.0),
            ]
        );
        assert_eq!(projected(&optimized), vec![("rzz", vec![0, 1], -1.5)]);
        assert_eq!(
            RealizationCapabilities::from_labels(&["unknown".to_owned()]),
            Err(crate::RealizationError::UnknownCapability(
                "unknown".to_owned()
            ))
        );
        assert_eq!(
            RealizationCapabilities::from_labels(&["rzz".to_owned(), "rzz".to_owned()]),
            Err(crate::RealizationError::DuplicateCapability(
                "rzz".to_owned()
            ))
        );
    }

    #[test]
    fn rejects_nonhermitian_noncommuting_width_and_noncanonical_inputs() {
        let nonhermitian = observable(1, Complex64::new(0.0, 1.0), std::iter::empty());
        assert_eq!(
            plan_pauli_evolution_with_capabilities(
                &nonhermitian,
                1,
                RealizationCapabilities::EMPTY
            ),
            Err(PauliEvolutionError::NonHermitian)
        );

        let noncommuting = observable(
            1,
            Complex64::new(0.0, 0.0),
            [
                (vec![(0, Pauli::X)], Complex64::new(1.0, 0.0)),
                (vec![(0, Pauli::Z)], Complex64::new(1.0, 0.0)),
            ],
        );
        assert_eq!(
            plan_pauli_evolution_with_capabilities(
                &noncommuting,
                1,
                RealizationCapabilities::EMPTY
            ),
            Err(PauliEvolutionError::NonCommuting)
        );

        let too_wide = observable(
            2,
            Complex64::new(0.0, 0.0),
            [(vec![(1, Pauli::Z)], Complex64::new(1.0, 0.0))],
        );
        assert_eq!(
            plan_pauli_evolution_with_capabilities(&too_wide, 1, RealizationCapabilities::EMPTY),
            Err(PauliEvolutionError::Width {
                observable_width: 2,
                state_width: 1,
            })
        );

        let normalized = observable(
            0,
            Complex64::new(0.0, 0.0),
            [(vec![(0, Pauli::Z)], Complex64::new(1.0, 0.0))],
        );
        assert_eq!(
            plan_pauli_evolution_with_capabilities(&normalized, 1, RealizationCapabilities::EMPTY),
            Err(PauliEvolutionError::Observable(
                ObservableError::NonCanonical
            ))
        );
    }

    #[test]
    fn rejects_pairwise_validation_past_its_work_budget() {
        let terms = (0..7_100_u64).map(|index| {
            (
                vec![(index, Pauli::X)],
                Complex64::new(f64::from((index % 7) as u8) + 1.0, 0.0),
            )
        });
        let literal = observable(7_100, Complex64::new(0.0, 0.0), terms);

        assert_eq!(
            plan_pauli_evolution_with_capabilities(&literal, 7_100, RealizationCapabilities::EMPTY),
            Err(PauliEvolutionError::CommutationBudget)
        );
    }

    #[test]
    fn rejects_concrete_schedules_past_the_output_budget() {
        let terms = (0..42_u64).map(|offset| {
            (
                (offset..offset + 1_000)
                    .map(|index| (index, Pauli::Y))
                    .collect::<Vec<_>>(),
                Complex64::new(f64::from((offset % 7) as u8) + 1.0, 0.0),
            )
        });
        let literal = observable(1_041, Complex64::new(0.0, 0.0), terms);

        assert_eq!(
            plan_pauli_evolution_with_capabilities(&literal, 1_041, RealizationCapabilities::EMPTY),
            Err(PauliEvolutionError::ScheduleBudget)
        );
    }
}
