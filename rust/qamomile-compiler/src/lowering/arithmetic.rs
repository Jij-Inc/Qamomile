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
    /// Apply one T phase gate.
    T,
    /// Apply one adjoint T phase gate.
    Tdg,
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
            Self::T => "t",
            Self::Tdg => "tdg",
        }
    }

    const fn adjoint(self) -> Self {
        match self {
            Self::AddInto => Self::SubtractInto,
            Self::SubtractInto => Self::AddInto,
            Self::T => Self::Tdg,
            Self::Tdg => Self::T,
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
/// addition uses O(n²) steps/selectors; ripple-carry addition uses O(n) steps
/// and selectors. Checked preflight precedes allocation.
///
/// Without an explicit choice, a native target capability, or the
/// minimum-selector policy, the portable default minimizes the estimated
/// CNOT count of the standard decomposition of each candidate.
///
/// # Errors
/// Rejects zero/odd widths and schedules exceeding either output budget.
pub fn plan_addition(
    state_width: u64,
    config: LoweringConfig,
) -> Result<ArithmeticPlan, RealizationError> {
    plan_controlled_addition(state_width, config, 0)
}

/// Plan addition whose control-inheriting steps receive `outer_controls`.
///
/// Surrounding controls change the automatic portable choice and replace a
/// default native choice; explicit and minimum-selector selections are kept.
///
/// # Errors
/// Rejects zero/odd widths and schedules exceeding either output budget.
pub(crate) fn plan_controlled_addition(
    state_width: u64,
    config: LoweringConfig,
    outer_controls: u64,
) -> Result<ArithmeticPlan, RealizationError> {
    if state_width == 0 || !state_width.is_multiple_of(2) {
        return Err(RealizationError::AdditionWidth);
    }
    let n = state_width / 2;
    let mut realization = config.select(RealizationFamily::AddInto);
    if config.selects_addition() && realization.kind() != Realization::AddNative {
        let choices = [
            Realization::AddIncrements,
            Realization::AddQft,
            Realization::AddRipple,
        ];
        if let Some(choice) = choices
            .into_iter()
            .filter_map(|choice| addition_cost(n, state_width, choice))
            .min_by_key(|cost| (cost.selectors, cost.steps))
        {
            realization = crate::ConfiguredRealization::exact(choice.realization);
        }
    } else if config.addition_is_automatic()
        || (outer_controls > 0
            && config.addition_is_default()
            && realization.kind() == Realization::AddNative)
    {
        // A coherently controlled native adder is synthesized by generic SDK
        // control of every gate; the portable ripple adder controls only the
        // 2n-1 CNOTs between its self-inverse conjugations.
        let choices = [
            Realization::AddRipple,
            Realization::AddQft,
            Realization::AddIncrements,
        ];
        if let Some(choice) = choices
            .into_iter()
            .filter_map(|choice| {
                let cnots = estimated_cnots(n, choice, outer_controls)?;
                addition_cost(n, state_width, choice).map(|cost| (cnots, cost))
            })
            .min_by_key(|(cnots, cost)| (*cnots, cost.steps))
        {
            realization = crate::ConfiguredRealization::exact(choice.1.realization);
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
        Realization::AddRipple if n == 1 => AdditionCost {
            realization,
            steps: 1,
            selectors: 2,
        },
        Realization::AddRipple => AdditionCost {
            realization,
            steps: n.checked_mul(28)?.checked_sub(25)?,
            selectors: n.checked_mul(44)?.checked_sub(38)?,
        },
        _ => unreachable!("family selection returns an addition realization"),
    };
    (cost.steps <= MAX_ARITHMETIC_STEPS && cost.selectors <= MAX_ARITHMETIC_SELECTORS)
        .then_some(cost)
}

/// Estimate CNOTs of a standard decomposition under `outer` extra controls.
///
/// Multi-controlled X uses the ancilla-free costs 0, 1, 6, 14, 36 and 84 for up
/// to five controls, then grows linearly; a controlled phase with extra
/// controls costs one multi-controlled X plus two CNOTs. Increments are only
/// estimated for one or two bits, where they need at most two internal controls.
fn estimated_cnots(n: u64, realization: Realization, outer: u64) -> Option<u64> {
    let x = |controls: u64| -> Option<u64> {
        match controls {
            0 => Some(0),
            1 => Some(1),
            2 => Some(6),
            3 => Some(14),
            4 => Some(36),
            _ => controls.checked_sub(5)?.checked_mul(56)?.checked_add(84),
        }
    };
    let phase = if outer == 0 {
        2
    } else {
        x(outer.checked_add(1)?)?.checked_add(2)?
    };
    match realization {
        Realization::AddIncrements => match n {
            1 => x(outer.checked_add(1)?),
            2 => x(outer.checked_add(2)?)?.checked_add(x(outer.checked_add(1)?)?.checked_mul(2)?),
            _ => None,
        },
        Realization::AddQft => {
            let basis = n.checked_mul(n.checked_sub(1)?)?.checked_mul(2)?;
            let rotations = n.checked_mul(n.checked_add(1)?)? / 2;
            basis.checked_add(rotations.checked_mul(phase)?)
        }
        Realization::AddRipple if n == 1 => x(outer.checked_add(1)?),
        Realization::AddRipple => {
            let inherited = n.checked_mul(2)?.checked_sub(1)?;
            n.checked_mul(14)?
                .checked_sub(12)?
                .checked_add(inherited.checked_mul(x(outer.checked_add(1)?)?)?)
        }
        _ => None,
    }
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
        Realization::AddRipple => ripple_addition(n, &mut steps),
        _ => unreachable!("family selection returns an addition realization"),
    }
    debug_assert_eq!(steps.len() as u64, count);
    Ok(ArithmeticPlan::new(realization, steps))
}

/// Gidney's ancilla-free modular adder (arXiv:1706.07884, Fig. 15).
///
/// The top addend bit carries the ripple. Fan-out layers and the controlled
/// swap blocks `S_i` are self-inverse conjugations around the CNOTs that write
/// the target, so only those `2n - 1` CNOTs inherit surrounding controls:
/// `C(W M W) = W C(M) W` and `C(X S M' S Y) = C(X) S C(M') S C(Y)`. Each `S_i`
/// uses a relative-phase Toffoli whose phase cancels against its mirror image.
fn ripple_addition(n: u64, steps: &mut Vec<ArithmeticStep>) {
    let addend = |bit: u64| bit;
    let target = |bit: u64| n + bit;
    if n == 1 {
        steps.push(ArithmeticStep::pauli_x(target(0), vec![addend(0)]));
        return;
    }
    let carry = addend(n - 1);
    let fan_out = |steps: &mut Vec<ArithmeticStep>| {
        for wire in (0..n - 1).map(addend).chain((0..n).map(target)) {
            steps.push(local_x(wire, vec![carry]));
        }
    };
    let swap_block = |steps: &mut Vec<ArithmeticStep>, bit: u64| {
        let (first, second, wire) = (target(bit), carry, addend(bit));
        steps.push(local_x(carry, vec![wire]));
        steps.push(single(ArithmeticGate::Hadamard, wire));
        steps.push(single(ArithmeticGate::T, wire));
        steps.push(local_x(wire, vec![second]));
        steps.push(single(ArithmeticGate::Tdg, wire));
        steps.push(local_x(wire, vec![first]));
        steps.push(single(ArithmeticGate::T, wire));
        steps.push(local_x(wire, vec![second]));
        steps.push(single(ArithmeticGate::Tdg, wire));
        steps.push(single(ArithmeticGate::Hadamard, wire));
        steps.push(local_x(carry, vec![wire]));
    };
    fan_out(steps);
    for bit in 0..n - 1 {
        steps.push(ArithmeticStep::pauli_x(target(bit), vec![carry]));
        swap_block(steps, bit);
    }
    steps.push(ArithmeticStep::pauli_x(target(n - 1), vec![carry]));
    for bit in (0..n - 1).rev() {
        swap_block(steps, bit);
        steps.push(ArithmeticStep::pauli_x(target(bit), vec![addend(bit)]));
    }
    fan_out(steps);
}

fn local_x(target: u64, controls: Vec<u64>) -> ArithmeticStep {
    ArithmeticStep {
        inherit_controls: false,
        ..ArithmeticStep::pauli_x(target, controls)
    }
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
            let candidates = ["add.increments", "add.qft", "add.ripple"].map(|label| {
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
            for choice in ["add.increments", "add.qft", "add.ripple"] {
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

    /// Simulate a schedule on basis states, with optional outer controls.
    fn simulate(plan: &ArithmeticPlan, input: u64, outer: bool) -> u64 {
        let mut state = input;
        for step in plan.steps() {
            let bit = |wire: u64| state >> wire & 1 == 1;
            let active = (!step.inherits_controls() || outer)
                && step.controls().iter().all(|wire| bit(*wire));
            match step.gate() {
                ArithmeticGate::PauliX if active => state ^= 1 << step.indices()[0],
                ArithmeticGate::PauliX => {}
                _ => panic!("basis simulation needs X-only blocks"),
            }
        }
        state
    }

    #[test]
    fn ripple_matches_increments_on_classical_inputs_without_phase_blocks() {
        // Replace every relative-phase Toffoli block by an exact CCX so the
        // schedule can be checked as a classical permutation.
        for n in 1..6_u64 {
            let config = LoweringConfig::from_labels(&[], &["add.ripple".into()]).unwrap();
            let plan = plan_addition(2 * n, config).unwrap();
            assert_eq!(plan.realization(), Realization::AddRipple);
            let mut classical = Vec::new();
            let mut iter = plan.steps().iter().peekable();
            while let Some(step) = iter.next() {
                if step.gate() == ArithmeticGate::Hadamard {
                    let wire = step.indices()[0];
                    let block = (0..8).map(|_| iter.next().unwrap()).collect::<Vec<_>>();
                    assert_eq!(block[7].gate(), ArithmeticGate::Hadamard);
                    let mut controls = vec![block[1].controls()[0], block[3].controls()[0]];
                    controls.sort_unstable();
                    classical.push(local_x(wire, controls));
                } else {
                    classical.push(step.clone());
                }
            }
            let classical = ArithmeticPlan::new(Realization::AddRipple, classical);
            let mask = (1 << n) - 1;
            for input in 0..1_u64 << (2 * n) {
                let (a, b) = (input & mask, input >> n);
                let sum = a | ((a + b) & mask) << n;
                assert_eq!(simulate(&classical, input, true), sum);
                assert_eq!(simulate(&classical, input, false), input);
                let inverse = classical.clone().into_adjoint();
                assert_eq!(simulate(&inverse, sum, true), input);
            }
        }
    }

    #[test]
    fn automatic_addition_minimizes_estimated_cnots() {
        let kinds = (1..=8)
            .map(|n| {
                plan_addition(2 * n, LoweringConfig::EMPTY)
                    .unwrap()
                    .realization()
            })
            .collect::<Vec<_>>();
        assert_eq!(
            kinds,
            [
                Realization::AddRipple,
                Realization::AddIncrements,
                Realization::AddQft,
                Realization::AddQft,
                Realization::AddRipple,
                Realization::AddRipple,
                Realization::AddRipple,
                Realization::AddRipple,
            ]
        );
        let explicit = LoweringConfig::from_labels(&[], &["add.increments".into()]).unwrap();
        assert_eq!(
            plan_addition(16, explicit).unwrap().realization(),
            Realization::AddIncrements
        );
        // Doubly controlled Fourier rotations are more expensive than the
        // 2n-1 control-inheriting CNOTs of the ripple adder.
        assert_eq!(
            plan_controlled_addition(6, LoweringConfig::EMPTY, 2)
                .unwrap()
                .realization(),
            Realization::AddRipple
        );
        let native = LoweringConfig::from_labels(&["add_into".into()], &[]).unwrap();
        assert_eq!(
            plan_controlled_addition(16, native, 0)
                .unwrap()
                .realization(),
            Realization::AddNative
        );
        assert_eq!(
            plan_controlled_addition(16, native, 1)
                .unwrap()
                .realization(),
            Realization::AddRipple
        );
        let pinned =
            LoweringConfig::from_labels(&["add_into".into()], &["add.native".into()]).unwrap();
        assert_eq!(
            plan_controlled_addition(16, pinned, 1)
                .unwrap()
                .realization(),
            Realization::AddNative
        );
        for n in 1..=16 {
            assert_eq!(
                estimated_cnots(n, Realization::AddRipple, 0),
                Some(if n == 1 { 1 } else { 16 * n - 13 })
            );
            assert_eq!(
                estimated_cnots(n, Realization::AddQft, 0),
                Some(3 * n * n - n)
            );
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
