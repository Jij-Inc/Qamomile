//! Late exact realizations of little-endian wrapping register addition.

use crate::lowering::realization::{
    LoweringConfig, Realization, RealizationError, RealizationFamily,
};

/// Maximum steps in one arithmetic realization.
pub const MAX_ARITHMETIC_STEPS: u64 = 100_000;
/// Maximum total wire selectors, including coherent controls.
pub const MAX_ARITHMETIC_SELECTORS: u64 = 1_000_000;

/// One primitive instruction in a checked arithmetic schedule.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArithmeticGate {
    /// Preserve native wrapping register addition at the target boundary.
    AddInto,
    /// Preserve native wrapping register subtraction at the target boundary.
    SubtractInto,
    /// Apply one Hadamard gate.
    Hadamard,
    /// Apply one Pauli-X gate.
    PauliX,
    /// Apply one controlled phase rotation.
    ControlledPhase,
}

impl ArithmeticGate {
    /// Return the canonical target instruction spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AddInto => "add_into",
            Self::SubtractInto => "subtract_into",
            Self::Hadamard => "h",
            Self::PauliX => "x",
            Self::ControlledPhase => "cp",
        }
    }

    const fn adjoint(self) -> Self {
        match self {
            Self::AddInto => Self::SubtractInto,
            Self::SubtractInto => Self::AddInto,
            Self::Hadamard | Self::PauliX | Self::ControlledPhase => self,
        }
    }
}

/// A target-independent elementary step with an optional exact rotation.
#[derive(Clone, Debug, PartialEq)]
pub struct ArithmeticStep {
    /// Whether surrounding coherent controls apply to this step. Fourier
    /// basis changes are unconditional and cancel on the inactive branch.
    inherit_controls: bool,
    /// Closed primitive instruction kind.
    gate: ArithmeticGate,
    /// Logical selectors in the original aggregate state.
    indices: Vec<u64>,
    /// Additional coherent controls, all active on one.
    controls: Vec<u64>,
    /// Angle in radians; no small rotations are truncated.
    angle: Option<f64>,
}

impl ArithmeticStep {
    pub(super) fn pauli_x(target: u64, controls: Vec<u64>) -> Self {
        Self {
            inherit_controls: true,
            gate: ArithmeticGate::PauliX,
            indices: vec![target],
            controls,
            angle: None,
        }
    }

    /// Return whether surrounding coherent controls apply to this step.
    #[must_use]
    pub const fn inherits_controls(&self) -> bool {
        self.inherit_controls
    }

    /// Return the closed primitive instruction kind.
    #[must_use]
    pub const fn gate(&self) -> ArithmeticGate {
        self.gate
    }

    /// Return logical target selectors in the original aggregate state.
    #[must_use]
    pub fn indices(&self) -> &[u64] {
        &self.indices
    }

    /// Return additional positive coherent controls.
    #[must_use]
    pub fn controls(&self) -> &[u64] {
        &self.controls
    }

    /// Return the exact rotation angle when this is a parametric step.
    #[must_use]
    pub const fn angle(&self) -> Option<f64> {
        self.angle
    }

    fn adjoint(mut self) -> Self {
        self.gate = self.gate.adjoint();
        self.angle = self.angle.map(|angle| -angle);
        self
    }
}

/// Checked realization identity and a bounded schedule; native has one step.
#[derive(Clone, Debug, PartialEq)]
pub struct ArithmeticPlan {
    /// Selected implementation identity.
    realization: Realization,
    /// Exact steps, preserving all input wires and using no ancillas.
    steps: Vec<ArithmeticStep>,
}

impl ArithmeticPlan {
    pub(super) const fn new(realization: Realization, steps: Vec<ArithmeticStep>) -> Self {
        Self { realization, steps }
    }

    /// Return the selected exact realization.
    #[must_use]
    pub const fn realization(&self) -> Realization {
        self.realization
    }

    /// Borrow the checked schedule in execution order.
    #[must_use]
    pub fn steps(&self) -> &[ArithmeticStep] {
        &self.steps
    }

    /// Consume this plan into its checked schedule.
    #[must_use]
    pub fn into_steps(self) -> Vec<ArithmeticStep> {
        self.steps
    }

    /// Consume this plan into the exact inverse schedule.
    #[must_use]
    pub fn into_adjoint(mut self) -> Self {
        self.steps = self
            .steps
            .into_iter()
            .rev()
            .map(ArithmeticStep::adjoint)
            .collect();
        self
    }
}

/// Realize `|a>|b> -> |a>|a+b mod 2^n>` with equal little-endian halves.
///
/// The increment construction uses O(n²) steps and O(n³) selectors; Fourier
/// addition uses O(n²) steps/selectors. Checked preflight precedes allocation.
///
/// # Errors
/// Rejects zero/odd widths and schedules exceeding either output budget.
pub fn plan_addition(
    state_width: u64,
    config: LoweringConfig,
) -> Result<ArithmeticPlan, RealizationError> {
    if state_width == 0 || !state_width.is_multiple_of(2) {
        return Err(RealizationError::AdditionWidth);
    }
    let n = state_width / 2;
    let mut realization = config.select(RealizationFamily::AddInto);
    if config.selects_addition() && realization.kind() != Realization::AddNative {
        let choices = [Realization::AddIncrements, Realization::AddQft];
        if let Some(choice) = choices
            .into_iter()
            .filter_map(|choice| addition_cost(n, state_width, choice))
            .min_by_key(|cost| (cost.selectors, cost.steps))
        {
            realization = crate::ConfiguredRealization::exact(choice.realization);
        }
    }
    let cost = addition_cost(n, state_width, realization.kind())
        .ok_or(RealizationError::ScheduleBudget)?;
    build_addition(n, state_width, cost.realization, cost.steps)
}

/// A complete candidate whose work and allocation costs fit both native bounds.
struct AdditionCost {
    realization: Realization,
    steps: u64,
    selectors: u64,
}

fn addition_cost(n: u64, state_width: u64, realization: Realization) -> Option<AdditionCost> {
    let cost = match realization {
        Realization::AddNative => AdditionCost {
            realization,
            steps: 1,
            selectors: state_width,
        },
        Realization::AddIncrements => AdditionCost {
            realization,
            steps: n.checked_add(1)?.checked_mul(n)? / 2,
            selectors: n
                .checked_add(1)?
                .checked_mul(n)?
                .checked_mul(n.checked_add(5)?)?
                / 6,
        },
        Realization::AddQft => AdditionCost {
            realization,
            steps: n
                .checked_mul(n)?
                .checked_mul(3)?
                .checked_add(n.checked_mul(3)?)?
                / 2,
            selectors: n.checked_mul(n)?.checked_mul(3)?.checked_add(n)?,
        },
        _ => unreachable!("family selection returns an addition realization"),
    };
    (cost.steps <= MAX_ARITHMETIC_STEPS && cost.selectors <= MAX_ARITHMETIC_SELECTORS)
        .then_some(cost)
}

fn build_addition(
    n: u64,
    state_width: u64,
    realization: Realization,
    count: u64,
) -> Result<ArithmeticPlan, RealizationError> {
    let mut steps =
        Vec::with_capacity(usize::try_from(count).map_err(|_| RealizationError::ScheduleBudget)?);
    match realization {
        Realization::AddNative => steps.push(ArithmeticStep {
            inherit_controls: true,
            gate: ArithmeticGate::AddInto,
            indices: (0..state_width).collect(),
            controls: Vec::new(),
            angle: None,
        }),
        Realization::AddIncrements => {
            for addend in 0..n {
                for target in (addend..n).rev() {
                    let mut controls = Vec::with_capacity(
                        usize::try_from(target - addend + 1)
                            .map_err(|_| RealizationError::ScheduleBudget)?,
                    );
                    controls.push(addend);
                    controls.extend((addend..target).map(|bit| n + bit));
                    steps.push(ArithmeticStep {
                        inherit_controls: true,
                        gate: ArithmeticGate::PauliX,
                        indices: vec![n + target],
                        controls,
                        angle: None,
                    });
                }
            }
        }
        Realization::AddQft => {
            // QFT without swaps; Fourier output order is accounted for below.
            for target in (0..n).rev() {
                steps.push(single(ArithmeticGate::Hadamard, n + target));
                for control in (0..target).rev() {
                    let mut step = phase(n + control, n + target, angle(target - control));
                    step.inherit_controls = false;
                    steps.push(step);
                }
            }
            for addend in 0..n {
                for target in addend..n {
                    steps.push(phase(addend, n + target, angle(target - addend)));
                }
            }
            for target in 0..n {
                for control in 0..target {
                    let mut step = phase(n + control, n + target, -angle(target - control));
                    step.inherit_controls = false;
                    steps.push(step);
                }
                steps.push(single(ArithmeticGate::Hadamard, n + target));
            }
        }
        _ => unreachable!("family selection returns an addition realization"),
    }
    debug_assert_eq!(steps.len() as u64, count);
    Ok(ArithmeticPlan::new(realization, steps))
}

fn single(gate: ArithmeticGate, target: u64) -> ArithmeticStep {
    ArithmeticStep {
        inherit_controls: false,
        gate,
        indices: vec![target],
        controls: Vec::new(),
        angle: None,
    }
}

fn phase(control: u64, target: u64, angle: f64) -> ArithmeticStep {
    ArithmeticStep {
        inherit_controls: true,
        gate: ArithmeticGate::ControlledPhase,
        indices: vec![control, target],
        controls: Vec::new(),
        angle: Some(angle),
    }
}

fn angle(distance: u64) -> f64 {
    // The schedule budget bounds distance below binary64's exponent limit.
    std::f64::consts::PI * 2.0_f64.powi(-i32::try_from(distance).expect("bounded Fourier distance"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_cost_selection_respects_explicit_and_native_choices() {
        let automatic = LoweringConfig::EMPTY.minimizing_addition_selectors();
        for n in [1, 4, 8, 16, 32] {
            let selected = plan_addition(2 * n, automatic).unwrap();
            let candidates = ["add.increments", "add.qft"].map(|label| {
                plan_addition(
                    2 * n,
                    LoweringConfig::from_labels(&[], &[label.into()]).unwrap(),
                )
                .unwrap()
            });
            let cost = |plan: &ArithmeticPlan| {
                plan.steps()
                    .iter()
                    .map(|s| s.indices().len() + s.controls().len())
                    .sum::<usize>()
            };
            assert_eq!(cost(&selected), candidates.iter().map(cost).min().unwrap());
        }
        let explicit = LoweringConfig::from_labels(&[], &["add.qft".into()]).unwrap();
        assert_eq!(
            plan_addition(4, explicit).unwrap(),
            plan_addition(4, explicit.minimizing_addition_selectors()).unwrap()
        );
        let native =
            LoweringConfig::from_labels(&["add_into".into()], &["add.native".into()]).unwrap();
        assert_eq!(
            plan_addition(32, native.minimizing_addition_selectors())
                .unwrap()
                .steps()[0]
                .gate(),
            ArithmeticGate::AddInto
        );
        assert_eq!(
            plan_addition(u64::MAX - 1, automatic),
            Err(RealizationError::ScheduleBudget)
        );
    }

    #[test]
    fn checked_plan_owns_exact_adjoint_construction() {
        let config =
            LoweringConfig::from_labels(&["add_into".to_owned()], &["add.native".to_owned()])
                .unwrap();
        let plan = plan_addition(4, config).unwrap();
        assert_eq!(plan.steps()[0].gate(), ArithmeticGate::AddInto);

        let inverse = plan.into_adjoint();
        assert_eq!(inverse.steps()[0].gate(), ArithmeticGate::SubtractInto);
        assert_eq!(inverse.steps()[0].angle(), None);
    }

    #[test]
    fn schedules_match_preflight_and_restore_fourier_basis() {
        for n in 1..8 {
            for choice in ["add.increments", "add.qft"] {
                let config = LoweringConfig::from_labels(&[], &[choice.into()]).unwrap();
                let plan = plan_addition(2 * n, config).unwrap();
                assert!(plan.steps().iter().all(|step| {
                    step.indices()
                        .iter()
                        .chain(step.controls())
                        .all(|wire| *wire < 2 * n)
                }));
                if choice == "add.qft" {
                    assert_eq!(
                        plan.steps()
                            .iter()
                            .filter(|step| !step.inherits_controls())
                            .count() as u64,
                        n * (n + 1)
                    );
                    assert_eq!(plan.steps().len() as u64, 3 * n * (n + 1) / 2);
                }
            }
        }
    }

    #[test]
    fn limits_reject_before_allocation() {
        for width in [0, 1, 3, u64::MAX] {
            assert_eq!(
                plan_addition(width, LoweringConfig::EMPTY),
                Err(RealizationError::AdditionWidth)
            );
        }
        assert_eq!(
            plan_addition(u64::MAX - 1, LoweringConfig::EMPTY),
            Err(RealizationError::ScheduleBudget)
        );
    }
}
