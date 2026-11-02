//! Target-neutral modular-arithmetic realization plans.
//!
//! Semantic modular operations remain in persistent `QCore`.  At the final
//! target boundary this module either preserves an accepted native composite
//! or builds one bounded primitive plan shared by circuit consumers. Exact and
//! approximate implementations carry the same generic accuracy contract.

use crate::lowering::approximation::ApproximationCertificate;
use crate::lowering::arithmetic_permutation::{ArithmeticOperation, plan_arithmetic_permutation};
use crate::lowering::realization::{
    LoweringConfig, Realization, RealizationCapabilities, RealizationError, RealizationFamily,
};
use crate::number_theory::CheckedModulus;

#[cfg(test)]
#[path = "modular_small_tests.rs"]
mod small_tests;

/// Maximum primitive gates in one materialized modular realization.
pub const MAX_MODULAR_REALIZATION_STEPS: usize = 1_000_000;

/// Maximum cumulative wire and control selectors in a materialized plan.
pub const MAX_MODULAR_REALIZATION_SELECTORS: usize = 1_000_000;

/// Gates admitted by the checked modular realization boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModularGate {
    /// Apply Hadamard to one target.
    H,
    /// Apply Pauli-X to one target.
    X,
    /// Apply controlled-X to one control and one target.
    Cx,
    /// Apply doubly controlled-X to two controls and one target.
    Ccx,
    /// Exchange two targets, optionally under additional positive controls.
    Swap,
    /// Discard one internal environment wire and prepare zero.
    Reset,
}

impl ModularGate {
    /// Return the stable target instruction spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::H => "h",
            Self::X => "x",
            Self::Cx => "cx",
            Self::Ccx => "ccx",
            Self::Reset => "reset",
            Self::Swap => "swap",
        }
    }

    const fn arity(self) -> usize {
        match self {
            Self::H | Self::X | Self::Reset => 1,
            Self::Cx | Self::Swap => 2,
            Self::Ccx => 3,
        }
    }
}

/// One checked primitive step over the plan's relative wire space.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModularStep {
    gate: ModularGate,
    wires: Box<[u64]>,
    controls: Box<[u64]>,
}

impl ModularStep {
    /// Return the primitive gate.
    #[must_use]
    pub const fn gate(&self) -> ModularGate {
        self.gate
    }

    /// Return additional positive controls accepted by the next consumer.
    #[must_use]
    pub fn controls(&self) -> &[u64] {
        &self.controls
    }

    /// Return the ordered gate operands, excluding additional controls.
    #[must_use]
    pub fn wires(&self) -> &[u64] {
        &self.wires
    }
}

/// One bounded portable plan with explicit clean scratch requirements.
#[derive(Clone, Debug, PartialEq)]
pub struct ModularGatePlan {
    realization: Realization,
    child_realizations: Vec<Realization>,
    state_width: u64,
    control_width: u64,
    scratch_width: u64,
    approximation: ApproximationCertificate<f64>,
    steps: Vec<ModularStep>,
    step_count: usize,
    has_reset: bool,
}

impl ModularGatePlan {
    /// Return the selected implementation identity.
    #[must_use]
    pub const fn realization(&self) -> Realization {
        self.realization
    }

    /// Return distinct child implementations actually materialized by this plan.
    #[must_use]
    pub fn child_realizations(&self) -> &[Realization] {
        &self.child_realizations
    }

    /// Return the semantic state width at the start of the relative wire row.
    #[must_use]
    pub const fn state_width(&self) -> u64 {
        self.state_width
    }

    /// Return the number of surrounding coherent controls after the state row.
    #[must_use]
    pub const fn control_width(&self) -> u64 {
        self.control_width
    }

    /// Return the clean scratch width after state and control wires.
    #[must_use]
    pub const fn scratch_width(&self) -> u64 {
        self.scratch_width
    }

    /// Return exactness or the proved output trace-distance upper bound.
    #[must_use]
    pub const fn approximation(&self) -> ApproximationCertificate<f64> {
        self.approximation
    }

    /// Borrow the primitive schedule in execution order.
    #[must_use]
    pub fn steps(&self) -> &[ModularStep] {
        &self.steps
    }

    /// Consume the plan into its primitive schedule.
    #[must_use]
    pub fn into_steps(self) -> Vec<ModularStep> {
        self.steps
    }
}

/// Target-aware disposition of one semantic modular operation.
#[derive(Clone, Debug, PartialEq)]
pub enum ModularRealizationPlan {
    /// Preserve the semantic operation for a target that accepts its contract.
    PreserveNative(Realization),
    /// Emit the selected portable realization.
    Gates(ModularGatePlan),
}

impl ModularRealizationPlan {
    /// Return the selected implementation identity.
    #[must_use]
    pub const fn realization(&self) -> Realization {
        match self {
            Self::PreserveNative(realization) => *realization,
            Self::Gates(plan) => plan.realization(),
        }
    }
}

/// Plan exact modular exponentiation at the final target boundary.
///
/// Relative wires are ordered as semantic state, surrounding coherent
/// controls, then clean scratch.  A native-capable target receives
/// [`ModularRealizationPlan::PreserveNative`] unless an explicit portable
/// override was selected in `config`.
///
/// # Errors
/// Returns a stable realization error for invalid dimensions, non-unit bases,
/// checked-arithmetic overflow, or a schedule beyond the work budget.
pub fn plan_mod_pow(
    target_width: u64,
    exponent_width: u64,
    base: i64,
    modulus: i64,
    inverse: bool,
    control_width: u64,
    config: LoweringConfig,
) -> Result<ModularRealizationPlan, RealizationError> {
    plan_mod_pow_with_schedule(
        target_width,
        exponent_width,
        base,
        modulus,
        inverse,
        control_width,
        config,
        true,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "Shared validation and schedule generation use identical semantic inputs"
)]
fn plan_mod_pow_with_schedule(
    target_width: u64,
    exponent_width: u64,
    base: i64,
    modulus: i64,
    inverse: bool,
    control_width: u64,
    config: LoweringConfig,
    materialize: bool,
) -> Result<ModularRealizationPlan, RealizationError> {
    let configured = config.select(RealizationFamily::ModPow);
    let realization = configured.kind();
    if exponent_width == 0 {
        return Err(RealizationError::ArithmeticWidth);
    }
    let state_width = target_width
        .checked_add(exponent_width)
        .ok_or(RealizationError::ScheduleBudget)?;
    validate_modulus(target_width, modulus)?;
    let domain =
        CheckedModulus::new(target_width, modulus).ok_or(RealizationError::ArithmeticParameters)?;
    let base_unit = domain
        .unit(base)
        .ok_or(RealizationError::ArithmeticParameters)?;
    let base_unit = if inverse {
        base_unit.inverse()
    } else {
        base_unit
    };
    let effective_base = base_unit.residue();
    if realization == Realization::ModPowNative {
        return Ok(ModularRealizationPlan::PreserveNative(realization));
    }
    let mut builder =
        PlanBuilder::new(state_width, control_width, realization, config.capabilities)?;
    builder.materialize = materialize;
    let target = (0..target_width).collect::<Vec<_>>();
    let exponent = (target_width..state_width).collect::<Vec<_>>();
    if !base_unit.is_identity() {
        match realization {
            Realization::ModPowSquareAndMultiply => {
                let mut factor = effective_base;
                for control in exponent {
                    if domain
                        .unit(factor)
                        .ok_or(RealizationError::ArithmeticParameters)?
                        .is_identity()
                    {
                        break;
                    }
                    builder.controls.push(control);
                    let result = builder.emit_selected_mod_mul(&target, factor, modulus, config);
                    builder.controls.pop();
                    result?;
                    factor = modular_product(factor, factor, modulus)?;
                }
            }
            Realization::ModPowWindowedLookup => {
                builder.emit_windowed_mod_pow(&target, &exponent, effective_base, modulus)?;
            }
            Realization::ModPowCosetWindowed => {
                let padding_bits = configured
                    .coset_padding_bits()
                    .ok_or_else(|| {
                        RealizationError::InvalidParameters(
                            "coset modular exponentiation requires padding_bits".to_owned(),
                        )
                    })?
                    .get();
                builder.emit_coset_windowed_mod_pow(
                    &target,
                    &exponent,
                    effective_base,
                    modulus,
                    usize::from(padding_bits),
                )?;
                let event_count = target_width
                    .div_ceil(3)
                    .checked_mul(exponent_width.div_ceil(3))
                    .and_then(|value| value.checked_mul(2))
                    .ok_or(RealizationError::ScheduleBudget)?;
                // The materialized schedule bounds the event count below 2^53.
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "Each event emits gates within the million-step schedule budget"
                )]
                let deviation = (event_count as f64) * 2.0_f64.powi(-i32::from(padding_bits));
                let upper = (2.0 * deviation.min(1.0).sqrt()).min(1.0);
                builder.approximation =
                    ApproximationCertificate::output_trace_distance_bound(upper);
            }
            _ => return Err(RealizationError::ArithmeticParameters),
        }
    }
    Ok(ModularRealizationPlan::Gates(builder.finish()?))
}

/// Plan exact modular multi-exponentiation with independent coordinates.
///
/// Relative semantic wires are ordered as all packed coordinate registers,
/// followed by the shared modular target. Each coordinate contributes one
/// controlled schoolbook multiplication per exponent bit. The balanced
/// product-tree choice is currently an analysis realization and is rejected
/// here instead of silently emitting a different circuit.
///
/// # Errors
/// Returns a stable realization error for inconsistent dimensions, invalid
/// modular units, an analysis-only realization, checked-arithmetic overflow,
/// or a schedule beyond the work budget.
#[expect(
    clippy::too_many_arguments,
    reason = "Explicit semantic dimensions and lowering options mirror the modular operation contract"
)]
pub fn plan_mod_multi_pow(
    coordinate_count: u64,
    coordinate_width: u64,
    target_width: u64,
    base_width: u64,
    bases: &[i64],
    modulus: i64,
    inverse: bool,
    control_width: u64,
    config: LoweringConfig,
) -> Result<ModularGatePlan, RealizationError> {
    plan_mod_multi_pow_with_schedule(
        coordinate_count,
        coordinate_width,
        target_width,
        base_width,
        bases,
        modulus,
        inverse,
        control_width,
        config,
        true,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "Shared validation and schedule generation use identical semantic inputs"
)]
fn plan_mod_multi_pow_with_schedule(
    coordinate_count: u64,
    coordinate_width: u64,
    target_width: u64,
    base_width: u64,
    bases: &[i64],
    modulus: i64,
    inverse: bool,
    control_width: u64,
    config: LoweringConfig,
    materialize: bool,
) -> Result<ModularGatePlan, RealizationError> {
    if coordinate_count == 0
        || coordinate_width == 0
        || target_width == 0
        || base_width == 0
        || usize::try_from(coordinate_count).ok() != Some(bases.len())
    {
        return Err(RealizationError::ArithmeticWidth);
    }
    let realization = config.select(RealizationFamily::ModMultiPow).kind();
    if realization != Realization::ModMultiPowIndependent {
        return Err(RealizationError::ArithmeticParameters);
    }
    validate_modulus(target_width, modulus)?;
    if base_width < 63 {
        let limit = 1_i128
            .checked_shl(
                u32::try_from(base_width).map_err(|_| RealizationError::ArithmeticParameters)?,
            )
            .ok_or(RealizationError::ArithmeticParameters)?;
        if bases
            .iter()
            .any(|base| i128::from(base.rem_euclid(modulus)) >= limit)
        {
            return Err(RealizationError::ArithmeticParameters);
        }
    }
    let coordinate_total = coordinate_count
        .checked_mul(coordinate_width)
        .ok_or(RealizationError::ScheduleBudget)?;
    let state_width = coordinate_total
        .checked_add(target_width)
        .ok_or(RealizationError::ScheduleBudget)?;
    let mut builder =
        PlanBuilder::new(state_width, control_width, realization, config.capabilities)?;
    builder.materialize = materialize;
    let target = (coordinate_total..state_width).collect::<Vec<_>>();
    for (coordinate, base) in bases.iter().copied().enumerate() {
        let domain = CheckedModulus::new(target_width, modulus)
            .ok_or(RealizationError::ArithmeticParameters)?;
        let unit = domain
            .unit(base)
            .ok_or(RealizationError::ArithmeticParameters)?;
        let mut factor = if inverse { unit.inverse() } else { unit }.residue();
        let start = u64::try_from(coordinate)
            .map_err(|_| RealizationError::ScheduleBudget)?
            .checked_mul(coordinate_width)
            .ok_or(RealizationError::ScheduleBudget)?;
        for offset in 0..coordinate_width {
            if domain
                .unit(factor)
                .ok_or(RealizationError::ArithmeticParameters)?
                .is_identity()
            {
                break;
            }
            {
                builder.controls.push(
                    start
                        .checked_add(offset)
                        .ok_or(RealizationError::ScheduleBudget)?,
                );
                let result = builder.emit_selected_mod_mul(&target, factor, modulus, config);
                builder.controls.pop();
                result?;
            }
            factor = modular_product(factor, factor, modulus)?;
        }
    }
    builder.finish()
}

/// Plan one exact schoolbook modular multiplication.
///
/// # Errors
/// Returns a stable realization error for invalid dimensions, a non-unit
/// factor, checked-arithmetic overflow, or a schedule beyond the work budget.
pub fn plan_schoolbook_mod_mul(
    width: u64,
    factor: i64,
    modulus: i64,
    inverse: bool,
    control_width: u64,
    capabilities: RealizationCapabilities,
) -> Result<ModularGatePlan, RealizationError> {
    plan_schoolbook_mod_mul_with_schedule(
        width,
        factor,
        modulus,
        inverse,
        control_width,
        capabilities,
        true,
    )
}

fn plan_schoolbook_mod_mul_with_schedule(
    width: u64,
    factor: i64,
    modulus: i64,
    inverse: bool,
    control_width: u64,
    capabilities: RealizationCapabilities,
    materialize: bool,
) -> Result<ModularGatePlan, RealizationError> {
    let mut builder = PlanBuilder::new(
        width,
        control_width,
        Realization::ModMulSchoolbookRipple,
        capabilities,
    )?;
    builder.materialize = materialize;
    validate_modulus(width, modulus)?;
    let domain =
        CheckedModulus::new(width, modulus).ok_or(RealizationError::ArithmeticParameters)?;
    let unit = domain
        .unit(factor)
        .ok_or(RealizationError::ArithmeticParameters)?;
    let effective_factor = if inverse { unit.inverse() } else { unit }.residue();
    let state = (0..width).collect::<Vec<_>>();
    builder.emit_schoolbook_mod_mul(&state, effective_factor, modulus)?;
    builder.finish()
}

/// Plan one exact retained-register modular multiply-accumulate.
///
/// Relative semantic wires are ordered as accumulator, left factor, and right
/// factor. The plan acts only when all three values are below the classical
/// modulus and is the identity elsewhere.
///
/// # Errors
/// Returns a stable realization error for invalid dimensions or modulus,
/// checked-arithmetic overflow, or a schedule beyond the work budget.
pub fn plan_schoolbook_mod_mul_add(
    width: u64,
    modulus: i64,
    inverse: bool,
    control_width: u64,
    capabilities: RealizationCapabilities,
) -> Result<ModularGatePlan, RealizationError> {
    plan_schoolbook_mod_mul_add_with_schedule(
        width,
        modulus,
        inverse,
        control_width,
        capabilities,
        true,
    )
}

fn plan_schoolbook_mod_mul_add_with_schedule(
    width: u64,
    modulus: i64,
    inverse: bool,
    control_width: u64,
    capabilities: RealizationCapabilities,
    materialize: bool,
) -> Result<ModularGatePlan, RealizationError> {
    validate_modulus(width, modulus)?;
    let state_width = width
        .checked_mul(3)
        .ok_or(RealizationError::ScheduleBudget)?;
    let mut builder = PlanBuilder::new(
        state_width,
        control_width,
        Realization::ModMulAddSchoolbookRipple,
        capabilities,
    )?;
    builder.materialize = materialize;
    let width = usize::try_from(width).map_err(|_| RealizationError::ScheduleBudget)?;
    let accumulator = (0..width).map(|wire| wire as u64).collect::<Vec<_>>();
    let left = (width..2 * width)
        .map(|wire| wire as u64)
        .collect::<Vec<_>>();
    let right = (2 * width..3 * width)
        .map(|wire| wire as u64)
        .collect::<Vec<_>>();
    builder.emit_schoolbook_mod_mul_add(&accumulator, &left, &right, modulus, inverse)?;
    builder.finish()
}

/// Closed inputs for shared modular validation and deferred schedule generation.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) enum ModularRequest {
    Multiply {
        width: u64,
        factor: i64,
        modulus: i64,
        inverse: bool,
        capabilities: RealizationCapabilities,
    },
    MultiplyAdd {
        width: u64,
        modulus: i64,
        inverse: bool,
        capabilities: RealizationCapabilities,
    },
    Power {
        target_width: u64,
        exponent_width: u64,
        base: i64,
        modulus: i64,
        inverse: bool,
        config: LoweringConfig,
    },
    MultiPower {
        coordinate_count: u64,
        coordinate_width: u64,
        target_width: u64,
        base_width: u64,
        bases: Box<[i64]>,
        modulus: i64,
        inverse: bool,
        config: LoweringConfig,
    },
}

/// Checked resource/layout facts, without an allocated gate schedule.
#[derive(Clone, Debug)]
pub(crate) struct CheckedModularDefinition {
    request: ModularRequest,
    metadata: ModularLayout,
}

/// Validated layout and work facts, deliberately not an executable gate plan.
#[derive(Clone, Debug)]
pub(crate) struct ModularLayout {
    state_width: u64,
    scratch_width: u64,
    realization: Realization,
    child_realizations: Vec<Realization>,
    step_count: usize,
    has_reset: bool,
}

impl ModularLayout {
    pub(crate) fn state_width(&self) -> u64 {
        self.state_width
    }
    pub(crate) fn scratch_width(&self) -> u64 {
        self.scratch_width
    }
    pub(crate) fn realization(&self) -> Realization {
        self.realization
    }
    pub(crate) fn child_realizations(&self) -> &[Realization] {
        &self.child_realizations
    }
}

impl ModularRequest {
    fn plan(&self, materialize: bool) -> Result<ModularGatePlan, RealizationError> {
        match self {
            Self::Multiply {
                width,
                factor,
                modulus,
                inverse,
                capabilities,
            } => plan_schoolbook_mod_mul_with_schedule(
                *width,
                *factor,
                *modulus,
                *inverse,
                0,
                *capabilities,
                materialize,
            ),
            Self::MultiplyAdd {
                width,
                modulus,
                inverse,
                capabilities,
            } => plan_schoolbook_mod_mul_add_with_schedule(
                *width,
                *modulus,
                *inverse,
                0,
                *capabilities,
                materialize,
            ),
            Self::Power {
                target_width,
                exponent_width,
                base,
                modulus,
                inverse,
                config,
            } => {
                match plan_mod_pow_with_schedule(
                    *target_width,
                    *exponent_width,
                    *base,
                    *modulus,
                    *inverse,
                    0,
                    *config,
                    materialize,
                )? {
                    ModularRealizationPlan::Gates(plan) => Ok(plan),
                    ModularRealizationPlan::PreserveNative(_) => {
                        Err(RealizationError::ArithmeticParameters)
                    }
                }
            }
            Self::MultiPower {
                coordinate_count,
                coordinate_width,
                target_width,
                base_width,
                bases,
                modulus,
                inverse,
                config,
            } => plan_mod_multi_pow_with_schedule(
                *coordinate_count,
                *coordinate_width,
                *target_width,
                *base_width,
                bases,
                *modulus,
                *inverse,
                0,
                *config,
                materialize,
            ),
        }
    }

    pub(crate) fn check(self) -> Result<CheckedModularDefinition, RealizationError> {
        let plan = self.plan(false)?;
        let metadata = ModularLayout {
            state_width: plan.state_width,
            scratch_width: plan.scratch_width,
            realization: plan.realization,
            child_realizations: plan.child_realizations,
            step_count: plan.step_count,
            has_reset: plan.has_reset,
        };
        Ok(CheckedModularDefinition {
            request: self,
            metadata,
        })
    }
}

impl CheckedModularDefinition {
    pub(crate) fn metadata(&self) -> &ModularLayout {
        &self.metadata
    }
    pub(crate) fn is_unitary(&self) -> bool {
        !self.metadata.has_reset
    }
    pub(crate) fn materialize(&self) -> Result<ModularGatePlan, RealizationError> {
        let plan = self.request.plan(true)?;
        if plan.scratch_width != self.metadata.scratch_width
            || plan.steps.len() != self.metadata.step_count
        {
            return Err(RealizationError::ArithmeticParameters);
        }
        Ok(plan)
    }
}

struct PlanBuilder {
    realization: Realization,
    child_realizations: Vec<Realization>,
    state_width: u64,
    control_width: u64,
    next_wire: u64,
    controls: Vec<u64>,
    workspace: Vec<u64>,
    mcx_chain: Vec<u64>,
    capabilities: RealizationCapabilities,
    selector_count: usize,
    approximation: ApproximationCertificate<f64>,
    steps: Vec<ModularStep>,
    step_count: usize,
    has_reset: bool,
    materialize: bool,
}

impl PlanBuilder {
    fn new(
        state_width: u64,
        control_width: u64,
        realization: Realization,
        capabilities: RealizationCapabilities,
    ) -> Result<Self, RealizationError> {
        let next_wire = state_width
            .checked_add(control_width)
            .ok_or(RealizationError::ScheduleBudget)?;
        if next_wire > MAX_MODULAR_REALIZATION_SELECTORS as u64 {
            return Err(RealizationError::ScheduleBudget);
        }
        Ok(Self {
            realization,
            child_realizations: Vec::new(),
            state_width,
            control_width,
            next_wire,
            controls: (state_width..next_wire).collect(),
            workspace: Vec::new(),
            mcx_chain: Vec::new(),
            capabilities,
            selector_count: 0,
            approximation: ApproximationCertificate::Exact,
            steps: Vec::new(),
            step_count: 0,
            has_reset: false,
            materialize: true,
        })
    }

    fn finish(self) -> Result<ModularGatePlan, RealizationError> {
        let input_width = self
            .state_width
            .checked_add(self.control_width)
            .ok_or(RealizationError::ScheduleBudget)?;
        let scratch_width = self
            .next_wire
            .checked_sub(input_width)
            .ok_or(RealizationError::ScheduleBudget)?;
        Ok(ModularGatePlan {
            realization: self.realization,
            child_realizations: self.child_realizations,
            state_width: self.state_width,
            control_width: self.control_width,
            scratch_width,
            approximation: self.approximation,
            steps: self.steps,
            step_count: self.step_count,
            has_reset: self.has_reset,
        })
    }

    fn reserve_clean_wires(&mut self, count: usize) -> Result<Vec<u64>, RealizationError> {
        let count = u64::try_from(count).map_err(|_| RealizationError::ScheduleBudget)?;
        let end = self
            .next_wire
            .checked_add(count)
            .ok_or(RealizationError::ScheduleBudget)?;
        if end > MAX_MODULAR_REALIZATION_SELECTORS as u64 {
            return Err(RealizationError::ScheduleBudget);
        }
        let wires = (self.next_wire..end).collect();
        self.next_wire = end;
        Ok(wires)
    }

    fn schoolbook_workspace(&mut self, count: usize) -> Result<Vec<u64>, RealizationError> {
        if self.workspace.len() < count {
            let missing = count - self.workspace.len();
            let added = self.reserve_clean_wires(missing)?;
            self.workspace.extend(added);
        }
        Ok(self.workspace[..count].to_vec())
    }

    fn push(
        &mut self,
        gate: ModularGate,
        wires: impl IntoIterator<Item = u64>,
    ) -> Result<(), RealizationError> {
        self.push_controlled(gate, wires, &[])
    }

    fn push_controlled(
        &mut self,
        gate: ModularGate,
        wires: impl IntoIterator<Item = u64>,
        controls: &[u64],
    ) -> Result<(), RealizationError> {
        let wires = wires.into_iter().collect::<Vec<_>>();
        if wires.len() != gate.arity()
            || (!controls.is_empty() && !matches!(gate, ModularGate::X | ModularGate::Swap))
        {
            return Err(RealizationError::ArithmeticParameters);
        }
        let selectors = wires
            .len()
            .checked_add(controls.len())
            .and_then(|count| self.selector_count.checked_add(count))
            .filter(|count| *count <= MAX_MODULAR_REALIZATION_SELECTORS)
            .ok_or(RealizationError::ScheduleBudget)?;
        if self.step_count >= MAX_MODULAR_REALIZATION_STEPS {
            return Err(RealizationError::ScheduleBudget);
        }
        self.validate_selectors(&wires, controls)?;
        self.step_count += 1;
        self.has_reset |= gate == ModularGate::Reset;
        if self.materialize {
            self.steps.push(ModularStep {
                gate,
                wires: wires.into_boxed_slice(),
                controls: controls.into(),
            });
        }
        self.selector_count = selectors;
        Ok(())
    }

    fn validate_selectors(&self, wires: &[u64], controls: &[u64]) -> Result<(), RealizationError> {
        let count = wires
            .len()
            .checked_add(controls.len())
            .filter(|count| *count <= MAX_MODULAR_REALIZATION_SELECTORS)
            .ok_or(RealizationError::ScheduleBudget)?;
        let mut distinct = Vec::with_capacity(count);
        distinct.extend_from_slice(wires);
        distinct.extend_from_slice(controls);
        if distinct.iter().any(|wire| *wire >= self.next_wire) {
            return Err(RealizationError::ArithmeticParameters);
        }
        distinct.sort_unstable();
        distinct.dedup();
        if distinct.len() != count {
            return Err(RealizationError::ArithmeticParameters);
        }
        Ok(())
    }

    fn push_with_local_controls(
        &mut self,
        gate: ModularGate,
        wires: impl IntoIterator<Item = u64>,
        controls: &[u64],
    ) -> Result<(), RealizationError> {
        let wires = wires.into_iter().collect::<Vec<_>>();
        if wires.len() != gate.arity()
            || !matches!(gate, ModularGate::X | ModularGate::Cx | ModularGate::Ccx)
        {
            return Err(RealizationError::ArithmeticParameters);
        }
        self.controls
            .len()
            .checked_add(controls.len())
            .and_then(|count| count.checked_add(wires.len()))
            .filter(|count| *count <= MAX_MODULAR_REALIZATION_SELECTORS)
            .ok_or(RealizationError::ScheduleBudget)?;
        let target = *wires.last().ok_or(RealizationError::ArithmeticParameters)?;
        let mut all_controls = self.controls.clone();
        all_controls.extend_from_slice(controls);
        all_controls.extend_from_slice(&wires[..wires.len() - 1]);
        self.emit_clean_mcx(&all_controls, target)
    }

    fn emit_controlled_swap(
        &mut self,
        left: u64,
        right: u64,
        controls: &[u64],
    ) -> Result<(), RealizationError> {
        self.controls
            .len()
            .checked_add(controls.len())
            .and_then(|count| count.checked_add(2))
            .filter(|count| *count <= MAX_MODULAR_REALIZATION_SELECTORS)
            .ok_or(RealizationError::ScheduleBudget)?;
        let mut all_controls = self.controls.clone();
        all_controls.extend_from_slice(controls);
        self.validate_selectors(&[left, right], &all_controls)?;
        if self
            .capabilities
            .contains(RealizationCapabilities::CONTROLLED_SWAP)
        {
            return self.push_controlled(ModularGate::Swap, [left, right], &all_controls);
        }
        // Conjugation cancels even when a local control is inactive.
        self.push_with_local_controls(ModularGate::Cx, [right, left], &[])?;
        self.push_with_local_controls(ModularGate::Cx, [left, right], controls)?;
        self.push_with_local_controls(ModularGate::Cx, [right, left], &[])
    }

    fn emit_clean_mcx(&mut self, controls: &[u64], target: u64) -> Result<(), RealizationError> {
        self.validate_selectors(&[target], controls)?;
        if controls.len() > 2 && self.capabilities.contains(RealizationCapabilities::MCX) {
            return self.push_controlled(ModularGate::X, [target], controls);
        }
        match controls {
            [] => self.push(ModularGate::X, [target]),
            [control] => self.push(ModularGate::Cx, [*control, target]),
            [left, right] => self.push(ModularGate::Ccx, [*left, *right, target]),
            _ => {
                let required = controls.len() - 2;
                let gate_count = controls
                    .len()
                    .checked_mul(2)
                    .and_then(|count| count.checked_sub(3))
                    .ok_or(RealizationError::ScheduleBudget)?;
                let selectors = gate_count
                    .checked_mul(3)
                    .and_then(|count| self.selector_count.checked_add(count))
                    .filter(|count| *count <= MAX_MODULAR_REALIZATION_SELECTORS)
                    .ok_or(RealizationError::ScheduleBudget)?;
                let _ = selectors;
                self.step_count
                    .checked_add(gate_count)
                    .filter(|count| *count <= MAX_MODULAR_REALIZATION_STEPS)
                    .ok_or(RealizationError::ScheduleBudget)?;
                if self.mcx_chain.len() < required {
                    let added = self.reserve_clean_wires(required - self.mcx_chain.len())?;
                    self.mcx_chain.extend(added);
                }
                let chain = self.mcx_chain[..required].to_vec();
                self.push(ModularGate::Ccx, [controls[0], controls[1], chain[0]])?;
                for index in 2..controls.len() - 1 {
                    self.push(
                        ModularGate::Ccx,
                        [chain[index - 2], controls[index], chain[index - 1]],
                    )?;
                }
                self.push(
                    ModularGate::Ccx,
                    [chain[required - 1], controls[controls.len() - 1], target],
                )?;
                for index in (2..controls.len() - 1).rev() {
                    self.push(
                        ModularGate::Ccx,
                        [chain[index - 2], controls[index], chain[index - 1]],
                    )?;
                }
                self.push(ModularGate::Ccx, [controls[0], controls[1], chain[0]])
            }
        }
    }

    fn emit_windowed_mod_pow(
        &mut self,
        target: &[u64],
        exponent: &[u64],
        base: i64,
        modulus: i64,
    ) -> Result<(), RealizationError> {
        const WINDOW: usize = 3;
        let mut window_base = base;
        for exponent_window in exponent.chunks(WINDOW) {
            self.emit_windowed_factor_mul(target, exponent_window, window_base, modulus)?;
            for _ in 0..exponent_window.len() {
                window_base = modular_product(window_base, window_base, modulus)?;
            }
        }
        Ok(())
    }

    fn emit_coset_windowed_mod_pow(
        &mut self,
        target: &[u64],
        exponent: &[u64],
        base: i64,
        modulus: i64,
        padding_bits: usize,
    ) -> Result<(), RealizationError> {
        const WINDOW: usize = 3;
        let mut window_base = base;
        for exponent_window in exponent.chunks(WINDOW) {
            self.emit_coset_windowed_factor_mul(
                target,
                exponent_window,
                window_base,
                modulus,
                padding_bits,
            )?;
            for _ in 0..exponent_window.len() {
                window_base = modular_product(window_base, window_base, modulus)?;
            }
        }
        Ok(())
    }

    fn emit_coset_windowed_factor_mul(
        &mut self,
        state: &[u64],
        exponent: &[u64],
        base: i64,
        modulus: i64,
        padding_bits: usize,
    ) -> Result<(), RealizationError> {
        const WINDOW: usize = 3;
        let width = state.len();
        if width == 0
            || width >= 63
            || exponent.is_empty()
            || padding_bits == 0
            || modulus <= 0
            || i128::from(modulus) > (1_i128 << width)
            || CheckedModulus::new(64, modulus)
                .and_then(|domain| domain.unit(base))
                .is_none()
        {
            return Err(RealizationError::ArithmeticParameters);
        }
        let encoded_width = width
            .checked_add(padding_bits)
            .ok_or(RealizationError::ScheduleBudget)?;
        let required_workspace = encoded_width
            .checked_mul(3)
            .and_then(|value| value.checked_add(padding_bits))
            .and_then(|value| value.checked_add(3))
            .ok_or(RealizationError::ScheduleBudget)?;
        let workspace = self.schoolbook_workspace(required_workspace)?;
        let (accumulator, rest) = workspace.split_at(encoded_width);
        let (selector, rest) = rest.split_at(padding_bits);
        let (flags, rest) = rest.split_at(2);
        let valid = flags[0];
        let overflow = flags[1];
        let (lookup, rest) = rest.split_at(encoded_width);
        let (constant, rest) = rest.split_at(encoded_width);
        let carry = rest[0];

        self.emit_less_than_constant_xor(
            state,
            valid,
            overflow,
            &constant[..=width],
            carry,
            modulus,
        )?;
        self.emit_coset_encode(accumulator, selector, constant, carry, modulus, false)?;
        self.emit_coset_windowed_accumulation(
            accumulator,
            state,
            exponent,
            lookup,
            carry,
            base,
            modulus,
            false,
            valid,
            WINDOW,
        )?;
        self.emit_coset_encode(accumulator, selector, constant, carry, modulus, true)?;
        for wire in accumulator[width..].iter().chain(selector).copied() {
            self.push(ModularGate::Reset, [wire])?;
        }

        for (&left, &right) in state.iter().zip(&accumulator[..width]) {
            self.emit_controlled_swap(left, right, &[valid])?;
        }

        let inverse =
            modular_inverse(base, modulus).ok_or(RealizationError::ArithmeticParameters)?;
        self.emit_coset_encode(accumulator, selector, constant, carry, modulus, false)?;
        self.emit_coset_windowed_accumulation(
            accumulator,
            state,
            exponent,
            lookup,
            carry,
            inverse,
            modulus,
            true,
            valid,
            WINDOW,
        )?;
        self.emit_coset_encode(accumulator, selector, constant, carry, modulus, true)?;
        for wire in accumulator.iter().chain(selector).copied() {
            self.push(ModularGate::Reset, [wire])?;
        }
        self.emit_less_than_constant_xor(
            state,
            valid,
            overflow,
            &constant[..=width],
            carry,
            modulus,
        )
    }

    fn emit_coset_encode(
        &mut self,
        value: &[u64],
        selector: &[u64],
        constant: &[u64],
        carry: u64,
        modulus: i64,
        inverse: bool,
    ) -> Result<(), RealizationError> {
        if value.len() != constant.len() || selector.len() >= value.len() || modulus <= 0 {
            return Err(RealizationError::ArithmeticParameters);
        }
        if inverse {
            for (offset, control) in selector.iter().copied().enumerate().rev() {
                self.emit_positive_ripple_constant_add(
                    &value[offset..],
                    &constant[offset..],
                    carry,
                    modulus,
                    &[control],
                    true,
                )?;
            }
            for wire in selector.iter().rev().copied() {
                self.push(ModularGate::H, [wire])?;
            }
            return Ok(());
        }
        for wire in selector.iter().copied() {
            self.push(ModularGate::H, [wire])?;
        }
        for (offset, control) in selector.iter().copied().enumerate() {
            self.emit_positive_ripple_constant_add(
                &value[offset..],
                &constant[offset..],
                carry,
                modulus,
                &[control],
                false,
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_coset_windowed_accumulation(
        &mut self,
        accumulator: &[u64],
        source: &[u64],
        exponent: &[u64],
        lookup: &[u64],
        carry: u64,
        base: i64,
        modulus: i64,
        inverse: bool,
        valid: u64,
        window: usize,
    ) -> Result<(), RealizationError> {
        for (chunk_index, source_chunk) in source.chunks(window).enumerate() {
            let offset = chunk_index
                .checked_mul(window)
                .ok_or(RealizationError::ScheduleBudget)?;
            let mut address = exponent.to_vec();
            address.extend_from_slice(source_chunk);
            let rows = 1_usize
                .checked_shl(
                    u32::try_from(address.len()).map_err(|_| RealizationError::ScheduleBudget)?,
                )
                .ok_or(RealizationError::ScheduleBudget)?;
            let exponent_mask = (1_usize << exponent.len()) - 1;
            let mut table = Vec::with_capacity(rows);
            for row in 0..rows {
                let exponent_value = row & exponent_mask;
                let source_value = row >> exponent.len();
                let factor = modular_power(base, exponent_value, modulus)?;
                let shifted = (i128::try_from(source_value)
                    .map_err(|_| RealizationError::ScheduleBudget)?
                    << offset)
                    .rem_euclid(i128::from(modulus));
                let value = (shifted * i128::from(factor)).rem_euclid(i128::from(modulus));
                table.push(i64::try_from(value).map_err(|_| RealizationError::ScheduleBudget)?);
            }
            self.emit_lookup_xor(&address, lookup, &table)?;
            if inverse {
                self.emit_cuccaro_inverse(lookup, accumulator, carry, &[valid])?;
            } else {
                self.emit_cuccaro(lookup, accumulator, carry, &[valid])?;
            }
            self.emit_lookup_xor(&address, lookup, &table)?;
        }
        Ok(())
    }

    fn emit_positive_ripple_constant_add(
        &mut self,
        target: &[u64],
        constant: &[u64],
        carry: u64,
        value: i64,
        controls: &[u64],
        inverse: bool,
    ) -> Result<(), RealizationError> {
        if target.len() != constant.len() || target.is_empty() || value < 0 {
            return Err(RealizationError::ArithmeticParameters);
        }
        let normalized = u64::try_from(value).map_err(|_| RealizationError::ScheduleBudget)?;
        for (index, wire) in constant.iter().copied().enumerate().take(64) {
            if normalized & (1_u64 << index) != 0 {
                self.push_with_local_controls(ModularGate::X, [wire], controls)?;
            }
        }
        if inverse {
            self.emit_cuccaro_inverse(constant, target, carry, controls)?;
        } else {
            self.emit_cuccaro(constant, target, carry, controls)?;
        }
        for (index, wire) in constant.iter().copied().enumerate().take(64) {
            if normalized & (1_u64 << index) != 0 {
                self.push_with_local_controls(ModularGate::X, [wire], controls)?;
            }
        }
        Ok(())
    }

    fn emit_windowed_factor_mul(
        &mut self,
        state: &[u64],
        exponent: &[u64],
        base: i64,
        modulus: i64,
    ) -> Result<(), RealizationError> {
        const WINDOW: usize = 3;
        let width = state.len();
        if width == 0 || width >= 63 || exponent.is_empty() || modulus <= 0 {
            return Err(RealizationError::ArithmeticParameters);
        }
        if i128::from(modulus) > (1_i128 << width)
            || CheckedModulus::new(64, modulus)
                .and_then(|domain| domain.unit(base))
                .is_none()
        {
            return Err(RealizationError::ArithmeticParameters);
        }
        let required_workspace = width
            .checked_mul(3)
            .and_then(|value| value.checked_add(6))
            .ok_or(RealizationError::ScheduleBudget)?;
        let workspace = self.schoolbook_workspace(required_workspace)?;
        let (accumulator, rest) = workspace.split_at(width);
        let (flags, rest) = rest.split_at(3);
        let valid = flags[0];
        let reduction = flags[1];
        let overflow = flags[2];
        let (lookup, rest) = rest.split_at(width + 1);
        let (constant, rest) = rest.split_at(width + 1);
        let carry = rest[0];

        self.emit_less_than_constant_xor(state, valid, overflow, constant, carry, modulus)?;
        self.emit_windowed_accumulation(
            accumulator,
            state,
            exponent,
            lookup,
            reduction,
            overflow,
            constant,
            carry,
            base,
            modulus,
            false,
            valid,
            WINDOW,
        )?;
        for (&left, &right) in state.iter().zip(accumulator) {
            self.emit_controlled_swap(left, right, &[valid])?;
        }
        let inverse =
            modular_inverse(base, modulus).ok_or(RealizationError::ArithmeticParameters)?;
        self.emit_windowed_accumulation(
            accumulator,
            state,
            exponent,
            lookup,
            reduction,
            overflow,
            constant,
            carry,
            inverse,
            modulus,
            true,
            valid,
            WINDOW,
        )?;
        self.emit_less_than_constant_xor(state, valid, overflow, constant, carry, modulus)
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_windowed_accumulation(
        &mut self,
        accumulator: &[u64],
        source: &[u64],
        exponent: &[u64],
        lookup: &[u64],
        reduction: u64,
        overflow: u64,
        constant: &[u64],
        carry: u64,
        base: i64,
        modulus: i64,
        inverse: bool,
        valid: u64,
        window: usize,
    ) -> Result<(), RealizationError> {
        for (chunk_index, source_chunk) in source.chunks(window).enumerate() {
            let offset = chunk_index
                .checked_mul(window)
                .ok_or(RealizationError::ScheduleBudget)?;
            let mut address = exponent.to_vec();
            address.extend_from_slice(source_chunk);
            let rows = 1_usize
                .checked_shl(
                    u32::try_from(address.len()).map_err(|_| RealizationError::ScheduleBudget)?,
                )
                .ok_or(RealizationError::ScheduleBudget)?;
            let exponent_mask = (1_usize << exponent.len()) - 1;
            let mut table = Vec::with_capacity(rows);
            for row in 0..rows {
                let exponent_value = row & exponent_mask;
                let source_value = row >> exponent.len();
                let factor = modular_power(base, exponent_value, modulus)?;
                let shifted = (i128::try_from(source_value)
                    .map_err(|_| RealizationError::ScheduleBudget)?
                    << offset)
                    .rem_euclid(i128::from(modulus));
                let value = (shifted * i128::from(factor)).rem_euclid(i128::from(modulus));
                table.push(i64::try_from(value).map_err(|_| RealizationError::ScheduleBudget)?);
            }
            self.emit_lookup_xor(&address, lookup, &table)?;
            self.emit_modular_register_add(
                accumulator,
                lookup,
                reduction,
                overflow,
                constant,
                carry,
                modulus,
                &[valid],
                inverse,
            )?;
            self.emit_lookup_xor(&address, lookup, &table)?;
        }
        Ok(())
    }

    fn emit_lookup_xor(
        &mut self,
        address: &[u64],
        output: &[u64],
        table: &[i64],
    ) -> Result<(), RealizationError> {
        if table.len() != 1_usize << address.len() {
            return Err(RealizationError::ArithmeticParameters);
        }
        let saved = std::mem::take(&mut self.controls);
        let result = (|| {
            for (row, value) in table.iter().copied().enumerate() {
                let negative = address
                    .iter()
                    .copied()
                    .enumerate()
                    .filter_map(|(bit, wire)| ((row >> bit) & 1 == 0).then_some(wire))
                    .collect::<Vec<_>>();
                for wire in negative.iter().copied() {
                    self.push(ModularGate::X, [wire])?;
                }
                for (bit, wire) in output.iter().copied().enumerate() {
                    if (i128::from(value) & (1_i128 << bit)) != 0 {
                        self.emit_clean_mcx(address, wire)?;
                    }
                }
                for wire in negative.iter().rev().copied() {
                    self.push(ModularGate::X, [wire])?;
                }
            }
            Ok(())
        })();
        self.controls = saved;
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_modular_register_add(
        &mut self,
        value: &[u64],
        addend: &[u64],
        reduction: u64,
        overflow: u64,
        constant: &[u64],
        carry: u64,
        modulus: i64,
        controls: &[u64],
        inverse: bool,
    ) -> Result<(), RealizationError> {
        let mut extended = value.to_vec();
        extended.push(overflow);
        if inverse {
            self.emit_cuccaro_inverse(addend, &extended, carry, controls)?;
            self.push_with_local_controls(ModularGate::X, [overflow], controls)?;
            self.push_with_local_controls(ModularGate::Cx, [overflow, reduction], controls)?;
            self.push_with_local_controls(ModularGate::X, [overflow], controls)?;
            self.emit_cuccaro(addend, &extended, carry, controls)?;
            let mut conditional = controls.to_vec();
            conditional.push(reduction);
            self.emit_ripple_constant_add(&extended, constant, carry, modulus, &conditional, true)?;
            self.push_with_local_controls(ModularGate::Cx, [overflow, reduction], controls)?;
            self.emit_ripple_constant_add(&extended, constant, carry, -modulus, controls, true)?;
            return self.emit_cuccaro_inverse(addend, &extended, carry, controls);
        }
        self.emit_cuccaro(addend, &extended, carry, controls)?;
        self.emit_ripple_constant_add(&extended, constant, carry, -modulus, controls, false)?;
        self.push_with_local_controls(ModularGate::Cx, [overflow, reduction], controls)?;
        let mut conditional = controls.to_vec();
        conditional.push(reduction);
        self.emit_ripple_constant_add(&extended, constant, carry, modulus, &conditional, false)?;
        self.emit_cuccaro_inverse(addend, &extended, carry, controls)?;
        self.push_with_local_controls(ModularGate::X, [overflow], controls)?;
        self.push_with_local_controls(ModularGate::Cx, [overflow, reduction], controls)?;
        self.push_with_local_controls(ModularGate::X, [overflow], controls)?;
        self.emit_cuccaro(addend, &extended, carry, controls)
    }

    fn emit_selected_mod_mul(
        &mut self,
        state: &[u64],
        factor: i64,
        modulus: i64,
        config: LoweringConfig,
    ) -> Result<(), RealizationError> {
        let width = u64::try_from(state.len()).map_err(|_| RealizationError::ArithmeticWidth)?;
        let choice = config.select_mod_mul_child(width).kind();
        if !self.child_realizations.contains(&choice) {
            self.child_realizations.push(choice);
        }
        if choice == Realization::ModMulSchoolbookRipple {
            return self.emit_schoolbook_mod_mul(state, factor, modulus);
        }
        let plan = plan_arithmetic_permutation(
            ArithmeticOperation::ModMul,
            width,
            &[factor, modulus],
            false,
        )?;
        for step in plan.into_steps() {
            let target = state[usize::try_from(step.indices()[0])
                .map_err(|_| RealizationError::ArithmeticWidth)?];
            let controls = step
                .controls()
                .iter()
                .map(|index| {
                    usize::try_from(*index)
                        .ok()
                        .and_then(|index| state.get(index))
                        .copied()
                        .ok_or(RealizationError::ArithmeticWidth)
                })
                .collect::<Result<Vec<_>, _>>()?;
            self.push_with_local_controls(ModularGate::X, [target], &controls)?;
        }
        Ok(())
    }

    fn emit_schoolbook_mod_mul(
        &mut self,
        state: &[u64],
        factor: i64,
        modulus: i64,
    ) -> Result<(), RealizationError> {
        let width = state.len();
        let width_u64 = u64::try_from(width).map_err(|_| RealizationError::ArithmeticWidth)?;
        validate_modulus(width_u64, modulus)?;
        let unit = CheckedModulus::new(width_u64, modulus)
            .and_then(|domain| domain.unit(factor))
            .ok_or(RealizationError::ArithmeticParameters)?;
        if unit.is_identity() {
            return Ok(());
        }
        let required_workspace = width
            .checked_mul(2)
            .and_then(|value| value.checked_add(5))
            .ok_or(RealizationError::ScheduleBudget)?;
        let workspace = self.schoolbook_workspace(required_workspace)?;
        let (accumulator, rest) = workspace.split_at(width);
        let (flags, rest) = rest.split_at(3);
        let valid = flags[0];
        let reduction = flags[1];
        let overflow = flags[2];
        let (constant, rest) = rest.split_at(width + 1);
        let carry = rest[0];

        self.emit_less_than_constant_xor(state, valid, overflow, constant, carry, modulus)?;
        let mut shifted = factor.rem_euclid(modulus);
        for (index, source) in state.iter().copied().enumerate() {
            self.emit_modular_constant_add(
                accumulator,
                reduction,
                overflow,
                constant,
                carry,
                shifted,
                modulus,
                &[valid, source],
                false,
            )?;
            if index + 1 != width {
                shifted = (shifted * 2).rem_euclid(modulus);
            }
        }
        for (&left, &right) in state.iter().zip(accumulator) {
            self.emit_controlled_swap(left, right, &[valid])?;
        }
        let inverse = unit.inverse().residue();
        let mut shifted_inverse = inverse;
        for (index, source) in state.iter().copied().enumerate() {
            self.emit_modular_constant_add(
                accumulator,
                reduction,
                overflow,
                constant,
                carry,
                shifted_inverse,
                modulus,
                &[valid, source],
                true,
            )?;
            if index + 1 != width {
                shifted_inverse = (shifted_inverse * 2).rem_euclid(modulus);
            }
        }
        self.emit_less_than_constant_xor(state, valid, overflow, constant, carry, modulus)
    }

    fn emit_schoolbook_mod_mul_add(
        &mut self,
        accumulator: &[u64],
        left: &[u64],
        right: &[u64],
        modulus: i64,
        inverse: bool,
    ) -> Result<(), RealizationError> {
        let width = accumulator.len();
        if width == 0
            || width >= 63
            || left.len() != width
            || right.len() != width
            || modulus <= 0
            || i128::from(modulus) > (1_i128 << width)
        {
            return Err(RealizationError::ArithmeticParameters);
        }
        // The deepest step combines outer controls, three validity flags, one
        // product selector, one reduction flag, and a Toffoli's two controls.
        let required_workspace = width
            .checked_mul(4)
            .and_then(|value| value.checked_add(7))
            .ok_or(RealizationError::ScheduleBudget)?;
        let workspace = self.schoolbook_workspace(required_workspace)?;
        let (product, rest) = workspace.split_at(2 * width);
        let (padding, rest) = rest.split_at(width);
        let (valid, rest) = rest.split_at(3);
        let reduction = rest[0];
        let overflow = rest[1];
        let (constant, rest) = rest[2..].split_at(width + 1);
        let carry = rest[0];

        for (value, flag) in [accumulator, left, right].into_iter().zip(valid) {
            self.emit_less_than_constant_xor(value, *flag, overflow, constant, carry, modulus)?;
        }

        let validity = valid.to_vec();
        for (index, selector) in right.iter().copied().enumerate() {
            let mut addend = left.to_vec();
            addend.extend_from_slice(&padding[..width - index]);
            let mut controls = validity.clone();
            controls.push(selector);
            self.emit_cuccaro(&addend, &product[index..], carry, &controls)?;
        }

        let mut residue = 1_i64.rem_euclid(modulus);
        for product_bit in product.iter().copied() {
            let mut controls = validity.clone();
            controls.push(product_bit);
            self.emit_modular_constant_add(
                accumulator,
                reduction,
                overflow,
                constant,
                carry,
                residue,
                modulus,
                &controls,
                inverse,
            )?;
            residue = modular_product(residue, 2, modulus)?;
        }

        for (index, selector) in right.iter().copied().enumerate().rev() {
            let mut addend = left.to_vec();
            addend.extend_from_slice(&padding[..width - index]);
            let mut controls = validity.clone();
            controls.push(selector);
            self.emit_cuccaro_inverse(&addend, &product[index..], carry, &controls)?;
        }

        for (value, flag) in [accumulator, left, right].into_iter().zip(valid).rev() {
            self.emit_less_than_constant_xor(value, *flag, overflow, constant, carry, modulus)?;
        }
        Ok(())
    }

    fn emit_less_than_constant_xor(
        &mut self,
        value: &[u64],
        flag: u64,
        overflow: u64,
        constant: &[u64],
        carry: u64,
        modulus: i64,
    ) -> Result<(), RealizationError> {
        let mut extended = value.to_vec();
        extended.push(overflow);
        self.emit_ripple_constant_add(&extended, constant, carry, -modulus, &[], false)?;
        self.push_with_local_controls(ModularGate::Cx, [overflow, flag], &[])?;
        self.emit_ripple_constant_add(&extended, constant, carry, -modulus, &[], true)
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_modular_constant_add(
        &mut self,
        value: &[u64],
        reduction: u64,
        overflow: u64,
        constant: &[u64],
        carry: u64,
        addend: i64,
        modulus: i64,
        controls: &[u64],
        inverse: bool,
    ) -> Result<(), RealizationError> {
        if inverse {
            return self.emit_modular_constant_add_inverse(
                value, reduction, overflow, constant, carry, addend, modulus, controls,
            );
        }
        let mut extended = value.to_vec();
        extended.push(overflow);
        self.emit_ripple_constant_add(&extended, constant, carry, addend, controls, false)?;
        self.emit_ripple_constant_add(&extended, constant, carry, -modulus, controls, false)?;
        self.push_with_local_controls(ModularGate::Cx, [overflow, reduction], controls)?;
        let mut conditional = controls.to_vec();
        conditional.push(reduction);
        self.emit_ripple_constant_add(&extended, constant, carry, modulus, &conditional, false)?;
        self.emit_ripple_constant_add(&extended, constant, carry, addend, controls, true)?;
        self.push_with_local_controls(ModularGate::X, [overflow], controls)?;
        self.push_with_local_controls(ModularGate::Cx, [overflow, reduction], controls)?;
        self.push_with_local_controls(ModularGate::X, [overflow], controls)?;
        self.emit_ripple_constant_add(&extended, constant, carry, addend, controls, false)
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_modular_constant_add_inverse(
        &mut self,
        value: &[u64],
        reduction: u64,
        overflow: u64,
        constant: &[u64],
        carry: u64,
        addend: i64,
        modulus: i64,
        controls: &[u64],
    ) -> Result<(), RealizationError> {
        let mut extended = value.to_vec();
        extended.push(overflow);
        self.emit_ripple_constant_add(&extended, constant, carry, addend, controls, true)?;
        self.push_with_local_controls(ModularGate::X, [overflow], controls)?;
        self.push_with_local_controls(ModularGate::Cx, [overflow, reduction], controls)?;
        self.push_with_local_controls(ModularGate::X, [overflow], controls)?;
        self.emit_ripple_constant_add(&extended, constant, carry, addend, controls, false)?;
        let mut conditional = controls.to_vec();
        conditional.push(reduction);
        self.emit_ripple_constant_add(&extended, constant, carry, modulus, &conditional, true)?;
        self.push_with_local_controls(ModularGate::Cx, [overflow, reduction], controls)?;
        self.emit_ripple_constant_add(&extended, constant, carry, -modulus, controls, true)?;
        self.emit_ripple_constant_add(&extended, constant, carry, addend, controls, true)
    }

    fn emit_ripple_constant_add(
        &mut self,
        target: &[u64],
        constant: &[u64],
        carry: u64,
        value: i64,
        controls: &[u64],
        inverse: bool,
    ) -> Result<(), RealizationError> {
        if target.len() != constant.len() || target.is_empty() || target.len() >= 127 {
            return Err(RealizationError::ArithmeticWidth);
        }
        let modulus = 1_i128 << target.len();
        let normalized = i128::from(value).rem_euclid(modulus) as u128;
        for (index, wire) in constant.iter().copied().enumerate() {
            if normalized & (1_u128 << index) != 0 {
                self.push_with_local_controls(ModularGate::X, [wire], controls)?;
            }
        }
        if inverse {
            self.emit_cuccaro_inverse(constant, target, carry, controls)?;
        } else {
            self.emit_cuccaro(constant, target, carry, controls)?;
        }
        for (index, wire) in constant.iter().copied().enumerate() {
            if normalized & (1_u128 << index) != 0 {
                self.push_with_local_controls(ModularGate::X, [wire], controls)?;
            }
        }
        Ok(())
    }

    fn emit_cuccaro(
        &mut self,
        addend: &[u64],
        target: &[u64],
        carry: u64,
        controls: &[u64],
    ) -> Result<(), RealizationError> {
        self.emit_majority(addend[0], target[0], carry, controls)?;
        for index in 0..addend.len() - 1 {
            self.emit_majority(
                addend[index + 1],
                target[index + 1],
                addend[index],
                controls,
            )?;
        }
        for index in (0..addend.len() - 1).rev() {
            self.emit_unmajority(
                addend[index + 1],
                target[index + 1],
                addend[index],
                controls,
            )?;
        }
        self.emit_unmajority(addend[0], target[0], carry, controls)
    }

    fn emit_cuccaro_inverse(
        &mut self,
        addend: &[u64],
        target: &[u64],
        carry: u64,
        controls: &[u64],
    ) -> Result<(), RealizationError> {
        self.emit_unmajority_inverse(addend[0], target[0], carry, controls)?;
        for index in 0..addend.len() - 1 {
            self.emit_unmajority_inverse(
                addend[index + 1],
                target[index + 1],
                addend[index],
                controls,
            )?;
        }
        for index in (0..addend.len() - 1).rev() {
            self.emit_majority_inverse(
                addend[index + 1],
                target[index + 1],
                addend[index],
                controls,
            )?;
        }
        self.emit_majority_inverse(addend[0], target[0], carry, controls)
    }

    fn emit_majority(
        &mut self,
        a: u64,
        b: u64,
        c: u64,
        controls: &[u64],
    ) -> Result<(), RealizationError> {
        self.push_with_local_controls(ModularGate::Cx, [a, b], controls)?;
        self.push_with_local_controls(ModularGate::Cx, [a, c], controls)?;
        self.push_with_local_controls(ModularGate::Ccx, [c, b, a], controls)
    }

    fn emit_majority_inverse(
        &mut self,
        a: u64,
        b: u64,
        c: u64,
        controls: &[u64],
    ) -> Result<(), RealizationError> {
        self.push_with_local_controls(ModularGate::Ccx, [c, b, a], controls)?;
        self.push_with_local_controls(ModularGate::Cx, [a, c], controls)?;
        self.push_with_local_controls(ModularGate::Cx, [a, b], controls)
    }

    fn emit_unmajority(
        &mut self,
        a: u64,
        b: u64,
        c: u64,
        controls: &[u64],
    ) -> Result<(), RealizationError> {
        self.push_with_local_controls(ModularGate::Ccx, [c, b, a], controls)?;
        self.push_with_local_controls(ModularGate::Cx, [a, c], controls)?;
        self.push_with_local_controls(ModularGate::Cx, [c, b], controls)
    }

    fn emit_unmajority_inverse(
        &mut self,
        a: u64,
        b: u64,
        c: u64,
        controls: &[u64],
    ) -> Result<(), RealizationError> {
        self.push_with_local_controls(ModularGate::Cx, [c, b], controls)?;
        self.push_with_local_controls(ModularGate::Cx, [a, c], controls)?;
        self.push_with_local_controls(ModularGate::Ccx, [c, b, a], controls)
    }
}

fn validate_modulus(width: u64, modulus: i64) -> Result<(), RealizationError> {
    if width == 0 || width >= 63 || modulus <= 0 || i128::from(modulus) > (1_i128 << width) {
        return Err(RealizationError::ArithmeticParameters);
    }
    Ok(())
}

fn modular_inverse(value: i64, modulus: i64) -> Option<i64> {
    CheckedModulus::new(64, modulus)
        .and_then(|domain| domain.unit(value))
        .map(|unit| unit.inverse().residue())
}

fn modular_product(left: i64, right: i64, modulus: i64) -> Result<i64, RealizationError> {
    if modulus <= 0 {
        return Err(RealizationError::ArithmeticParameters);
    }
    let value = (i128::from(left) * i128::from(right)).rem_euclid(i128::from(modulus));
    i64::try_from(value).map_err(|_| RealizationError::ScheduleBudget)
}

fn modular_power(
    mut base: i64,
    mut exponent: usize,
    modulus: i64,
) -> Result<i64, RealizationError> {
    let mut result = 1_i64.rem_euclid(modulus);
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = modular_product(result, base, modulus)?;
        }
        exponent >>= 1;
        if exponent != 0 {
            base = modular_product(base, base, modulus)?;
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simulate(plan: &ModularGatePlan, target: u64, exponent: u64, control: bool) -> Vec<bool> {
        simulate_words(plan, &[(target, 3), (exponent, 2)], control)
    }

    fn simulate_words(plan: &ModularGatePlan, words: &[(u64, usize)], control: bool) -> Vec<bool> {
        let total_width = plan.state_width() + plan.control_width() + plan.scratch_width();
        let mut bits = vec![false; usize::try_from(total_width).unwrap()];
        let mut offset = 0;
        for (value, width) in words {
            for index in 0..*width {
                bits[offset + index] = value & (1 << index) != 0;
            }
            offset += width;
        }
        assert_eq!(offset, usize::try_from(plan.state_width()).unwrap());
        if plan.control_width() == 1 {
            bits[offset] = control;
        }
        for step in plan.steps() {
            let wires = step
                .wires()
                .iter()
                .map(|wire| usize::try_from(*wire).unwrap())
                .collect::<Vec<_>>();
            if !step
                .controls()
                .iter()
                .all(|wire| bits[usize::try_from(*wire).unwrap()])
            {
                continue;
            }
            match step.gate() {
                ModularGate::X => bits[wires[0]] ^= true,
                ModularGate::Cx if bits[wires[0]] => bits[wires[1]] ^= true,
                ModularGate::Ccx if bits[wires[0]] && bits[wires[1]] => {
                    bits[wires[2]] ^= true;
                }
                ModularGate::Cx | ModularGate::Ccx => {}
                ModularGate::Swap => bits.swap(wires[0], wires[1]),
                ModularGate::Reset => bits[wires[0]] = false,
                ModularGate::H => panic!("exact realization emitted an approximate gate"),
            }
        }
        bits
    }

    fn read_word(bits: &[bool], offset: usize, width: usize) -> u64 {
        bits[offset..offset + width]
            .iter()
            .enumerate()
            .fold(0, |value, (index, bit)| value | (u64::from(*bit) << index))
    }

    #[test]
    fn mcx_retention_and_clean_fallback_agree_on_every_small_basis_state() {
        for control_count in 0..=5 {
            let state_width = control_count + 1;
            let controls = (0..control_count).collect::<Vec<_>>();
            let plans = [RealizationCapabilities::EMPTY, RealizationCapabilities::MCX].map(
                |capabilities| {
                    let mut builder = PlanBuilder::new(
                        state_width,
                        0,
                        Realization::ModMulSchoolbookRipple,
                        capabilities,
                    )
                    .unwrap();
                    builder.emit_clean_mcx(&controls, control_count).unwrap();
                    builder.finish().unwrap()
                },
            );
            assert_eq!(plans[1].scratch_width(), 0);
            assert_eq!(plans[1].steps().len(), 1);
            assert_eq!(plans[0].scratch_width(), control_count.saturating_sub(2));
            for input in 0..(1 << state_width) {
                let active = input & ((1 << control_count) - 1) == (1 << control_count) - 1;
                let expected = input ^ if active { 1 << control_count } else { 0 };
                for plan in &plans {
                    let bits = simulate_words(
                        plan,
                        &[(input, usize::try_from(state_width).unwrap())],
                        false,
                    );
                    assert_eq!(
                        read_word(&bits, 0, usize::try_from(state_width).unwrap()),
                        expected
                    );
                    assert!(
                        bits[usize::try_from(state_width).unwrap()..]
                            .iter()
                            .all(|bit| !bit)
                    );
                }
            }
        }
    }

    #[test]
    fn retained_swaps_match_clean_fallback_with_local_and_outer_controls() {
        for outer in 0..=2 {
            for local in 0..=3 {
                let width = local + 2;
                let plans = [
                    RealizationCapabilities::EMPTY,
                    RealizationCapabilities::CONTROLLED_SWAP,
                ]
                .map(|capabilities| {
                    let mut builder = PlanBuilder::new(
                        width,
                        outer,
                        Realization::ModMulSchoolbookRipple,
                        capabilities,
                    )
                    .unwrap();
                    builder
                        .emit_controlled_swap(0, 1, &(2..width).collect::<Vec<_>>())
                        .unwrap();
                    builder.finish().unwrap()
                });
                assert_eq!(plans[1].steps().len(), 1);
                assert_eq!(plans[1].scratch_width(), 0);
                for input in 0..(1 << (width + outer)) {
                    // simulate_words initializes arbitrary control rows as ordinary bits.
                    let mut results = Vec::new();
                    for plan in &plans {
                        let mut raw = plan.clone();
                        raw.state_width += raw.control_width;
                        raw.control_width = 0;
                        let bits = simulate_words(
                            &raw,
                            &[(input, usize::try_from(width + outer).unwrap())],
                            false,
                        );
                        assert!(
                            bits[usize::try_from(width + outer).unwrap()..]
                                .iter()
                                .all(|bit| !bit)
                        );
                        results.push(read_word(&bits, 0, usize::try_from(width + outer).unwrap()));
                    }
                    let active = input >> 2 == (1 << (local + outer)) - 1;
                    let expected = if active && (input & 1) != ((input >> 1) & 1) {
                        input ^ 3
                    } else {
                        input
                    };
                    assert_eq!(results, vec![expected, expected]);
                }
            }
        }
    }

    #[test]
    fn swap_validation_and_budget_fail_before_emission() {
        for capabilities in [
            RealizationCapabilities::EMPTY,
            RealizationCapabilities::CONTROLLED_SWAP,
        ] {
            let mut builder =
                PlanBuilder::new(4, 1, Realization::ModMulSchoolbookRipple, capabilities).unwrap();
            for (left, right, controls) in [
                (0, 0, vec![]),
                (0, 1, vec![0]),
                (0, 1, vec![2, 2]),
                (0, 1, vec![4]),
                (0, 5, vec![]),
            ] {
                assert!(matches!(
                    builder.emit_controlled_swap(left, right, &controls),
                    Err(RealizationError::ArithmeticParameters)
                ));
                assert!(builder.steps.is_empty());
                assert_eq!(builder.next_wire, 5);
            }
            builder.selector_count = MAX_MODULAR_REALIZATION_SELECTORS;
            assert!(matches!(
                builder.emit_controlled_swap(0, 1, &[2]),
                Err(RealizationError::ScheduleBudget)
            ));
            assert!(builder.steps.is_empty());
        }
    }

    #[test]
    fn modular_control_validation_rejects_aliases_and_wrong_gate_kinds() {
        for capabilities in [RealizationCapabilities::EMPTY, RealizationCapabilities::MCX] {
            let mut builder =
                PlanBuilder::new(5, 0, Realization::ModMulSchoolbookRipple, capabilities).unwrap();
            for (controls, target) in [(vec![0, 0, 1], 4), (vec![0, 1, 4], 4), (vec![0, 1, 5], 4)] {
                assert!(matches!(
                    builder.emit_clean_mcx(&controls, target),
                    Err(RealizationError::ArithmeticParameters)
                ));
            }
            assert!(matches!(
                builder.push_with_local_controls(ModularGate::H, [4], &[0]),
                Err(RealizationError::ArithmeticParameters)
            ));
            assert!(matches!(
                builder.push_with_local_controls(ModularGate::Reset, [4], &[0]),
                Err(RealizationError::ArithmeticParameters)
            ));
            assert_eq!(builder.next_wire, 5);
            assert!(builder.steps.is_empty());
        }
    }

    #[test]
    fn mcx_selector_and_expansion_budgets_precede_scratch_allocation() {
        for capabilities in [RealizationCapabilities::EMPTY, RealizationCapabilities::MCX] {
            let mut builder =
                PlanBuilder::new(5, 0, Realization::ModMulSchoolbookRipple, capabilities).unwrap();
            builder.selector_count = MAX_MODULAR_REALIZATION_SELECTORS - 3;
            assert!(matches!(
                builder.emit_clean_mcx(&[0, 1, 2, 3], 4),
                Err(RealizationError::ScheduleBudget)
            ));
            assert_eq!(builder.next_wire, 5);
            assert!(builder.steps.is_empty());
        }
        assert!(matches!(
            PlanBuilder::new(
                1,
                u64::MAX,
                Realization::ModMulSchoolbookRipple,
                RealizationCapabilities::MCX
            ),
            Err(RealizationError::ScheduleBudget)
        ));
    }

    #[test]
    fn nested_modular_controls_share_one_lazy_chain() {
        let mut builder = PlanBuilder::new(
            4,
            2,
            Realization::ModMulSchoolbookRipple,
            RealizationCapabilities::EMPTY,
        )
        .unwrap();
        builder
            .push_with_local_controls(ModularGate::Cx, [0, 3], &[1])
            .unwrap();
        let scratch_end = builder.next_wire;
        builder
            .push_with_local_controls(ModularGate::Cx, [0, 3], &[1])
            .unwrap();
        assert_eq!(builder.next_wire, scratch_end);
        assert_eq!(builder.mcx_chain.len(), 2);
        let workspace = builder.schoolbook_workspace(2).unwrap();
        assert!(
            workspace
                .iter()
                .all(|wire| !builder.mcx_chain.contains(wire))
        );
        let mut retained = PlanBuilder::new(
            4,
            2,
            Realization::ModMulSchoolbookRipple,
            RealizationCapabilities::MCX,
        )
        .unwrap();
        retained
            .push_with_local_controls(ModularGate::Cx, [0, 3], &[1])
            .unwrap();
        let plan = retained.finish().unwrap();
        assert_eq!(plan.steps()[0].controls(), &[4, 5, 1, 0]);
        assert_eq!(plan.steps()[0].wires(), &[3]);
        assert_eq!(plan.scratch_width(), 0);
    }

    #[test]
    fn native_mod_pow_capability_preserves_the_semantic_operation() {
        let config = LoweringConfig::from_labels(&["mod_pow".to_owned()], &[]).unwrap();
        assert_eq!(
            plan_mod_pow(3, 2, 2, 5, false, 0, config).unwrap(),
            ModularRealizationPlan::PreserveNative(Realization::ModPowNative),
        );
    }

    #[test]
    fn explicit_portable_mod_pow_overrides_native_preservation() {
        let config = LoweringConfig::from_labels(
            &["mod_pow".to_owned()],
            &["mod_pow.square_and_multiply".to_owned()],
        )
        .unwrap();
        let ModularRealizationPlan::Gates(plan) =
            plan_mod_pow(3, 2, 2, 5, false, 0, config).unwrap()
        else {
            panic!("portable override must materialize a gate plan");
        };
        assert_eq!(plan.realization(), Realization::ModPowSquareAndMultiply);
        assert!(!plan.steps().is_empty());
    }

    #[test]
    fn portable_mod_pow_is_bounded_and_uses_clean_scratch() {
        for realization in ["mod_pow.square_and_multiply", "mod_pow.windowed_lookup"] {
            let config = LoweringConfig::from_labels(&[], &[realization.to_owned()]).unwrap();
            let ModularRealizationPlan::Gates(plan) =
                plan_mod_pow(3, 2, 2, 5, false, 1, config).unwrap()
            else {
                panic!("portable realization must materialize a gate plan");
            };
            assert_eq!(plan.state_width(), 5);
            assert_eq!(plan.control_width(), 1);
            assert!(plan.scratch_width() > 0);
            assert!(plan.steps().len() <= MAX_MODULAR_REALIZATION_STEPS);
        }
    }

    #[test]
    fn exact_mod_pow_plans_preserve_total_domain_controls_and_clean_scratch() {
        for realization in ["mod_pow.square_and_multiply", "mod_pow.windowed_lookup"] {
            for capabilities in [vec![], vec!["mcx".to_owned(), "controlled_swap".to_owned()]] {
                let config =
                    LoweringConfig::from_labels(&capabilities, &[realization.to_owned()]).unwrap();
                for inverse in [false, true] {
                    let ModularRealizationPlan::Gates(plan) =
                        plan_mod_pow(3, 2, 2, 5, inverse, 1, config).unwrap()
                    else {
                        panic!("portable realization must materialize a gate plan");
                    };
                    let base = if inverse { 3 } else { 2 };
                    for control in [false, true] {
                        for target in 0..8 {
                            for exponent in 0..4 {
                                let bits = simulate(&plan, target, exponent, control);
                                let factor = u64::try_from(
                                    modular_power(base, usize::try_from(exponent).unwrap(), 5)
                                        .unwrap(),
                                )
                                .unwrap();
                                let expected = if control && target < 5 {
                                    target * factor % 5
                                } else {
                                    target
                                };
                                assert_eq!(read_word(&bits, 0, 3), expected);
                                assert_eq!(read_word(&bits, 3, 2), exponent);
                                assert_eq!(bits[5], control);
                                assert!(bits[6..].iter().all(|bit| !bit));
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn independent_multi_pow_preserves_coordinates_and_cleans_scratch() {
        for inverse in [false, true] {
            let plan =
                plan_mod_multi_pow(2, 2, 3, 2, &[2, 3], 5, inverse, 0, LoweringConfig::EMPTY)
                    .unwrap();
            assert_eq!(plan.realization(), Realization::ModMultiPowIndependent);
            for left in 0..4 {
                for right in 0..4 {
                    for target in 0..8 {
                        let bits =
                            simulate_words(&plan, &[(left, 2), (right, 2), (target, 3)], false);
                        let base_left = if inverse { 3 } else { 2 };
                        let base_right = if inverse { 2 } else { 3 };
                        let left_factor =
                            modular_power(base_left, usize::try_from(left).unwrap(), 5).unwrap();
                        let right_factor =
                            modular_power(base_right, usize::try_from(right).unwrap(), 5).unwrap();
                        let factor =
                            u64::try_from(modular_product(left_factor, right_factor, 5).unwrap())
                                .unwrap();
                        let expected = if target < 5 {
                            target * factor % 5
                        } else {
                            target
                        };
                        assert_eq!(read_word(&bits, 0, 2), left);
                        assert_eq!(read_word(&bits, 2, 2), right);
                        assert_eq!(read_word(&bits, 4, 3), expected);
                        assert!(bits[7..].iter().all(|bit| !bit));
                    }
                }
            }
        }
    }

    #[test]
    fn product_tree_multi_pow_is_not_silently_lowered_as_independent() {
        let config =
            LoweringConfig::from_labels(&[], &["mod_multi_pow.product_tree".to_owned()]).unwrap();
        assert_eq!(
            plan_mod_multi_pow(2, 2, 3, 2, &[2, 3], 5, false, 0, config),
            Err(RealizationError::ArithmeticParameters),
        );
    }

    #[test]
    fn multi_pow_rejects_a_base_outside_its_declared_leaf_width() {
        assert_eq!(
            plan_mod_multi_pow(1, 1, 3, 1, &[3], 5, false, 0, LoweringConfig::EMPTY),
            Err(RealizationError::ArithmeticParameters),
        );
    }

    #[test]
    fn schoolbook_mod_mul_add_is_total_controlled_and_cleans_scratch() {
        for inverse in [false, true] {
            let plan =
                plan_schoolbook_mod_mul_add(2, 3, inverse, 1, RealizationCapabilities::EMPTY)
                    .unwrap();
            assert_eq!(plan.state_width(), 6);
            assert_eq!(plan.control_width(), 1);
            assert!(plan.scratch_width() > 0);
            for control in [false, true] {
                for accumulator in 0..4 {
                    for left in 0..4 {
                        for right in 0..4 {
                            let bits = simulate_words(
                                &plan,
                                &[(accumulator, 2), (left, 2), (right, 2)],
                                control,
                            );
                            let valid = accumulator < 3 && left < 3 && right < 3;
                            let product = left * right % 3;
                            let expected = if control && valid {
                                if inverse {
                                    (accumulator + 3 - product) % 3
                                } else {
                                    (accumulator + product) % 3
                                }
                            } else {
                                accumulator
                            };
                            assert_eq!(read_word(&bits, 0, 2), expected);
                            assert_eq!(read_word(&bits, 2, 2), left);
                            assert_eq!(read_word(&bits, 4, 2), right);
                            assert_eq!(bits[6], control);
                            assert!(bits[7..].iter().all(|bit| !bit));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn coset_mod_pow_materializes_channel_steps_and_accuracy() {
        let config = LoweringConfig::from_labels(
            &[],
            &["mod_pow.coset_windowed[padding_bits=8]".to_owned()],
        )
        .unwrap();
        let ModularRealizationPlan::Gates(plan) =
            plan_mod_pow(3, 2, 2, 5, false, 1, config).unwrap()
        else {
            panic!("coset realization must materialize a gate plan");
        };
        assert_eq!(plan.realization(), Realization::ModPowCosetWindowed);
        assert!(
            plan.steps()
                .iter()
                .any(|step| step.gate() == ModularGate::H)
        );
        assert!(
            plan.steps()
                .iter()
                .any(|step| step.gate() == ModularGate::Reset)
        );
        let ApproximationCertificate::OutputTraceDistanceBound(upper) = plan.approximation() else {
            panic!("coset realization must carry an approximation certificate");
        };
        assert!(0.0 < upper && upper <= 1.0);
    }

    #[test]
    fn invalid_modular_parameters_fail_closed() {
        for (base, modulus) in [(2, 0), (2, 6), (2, 9)] {
            assert!(matches!(
                plan_mod_pow(3, 2, base, modulus, false, 0, LoweringConfig::EMPTY),
                Err(RealizationError::ArithmeticParameters),
            ));
        }
    }
}

#[cfg(test)]
mod deferred_tests {
    use super::{LoweringConfig, ModularRequest, RealizationCapabilities};

    #[test]
    fn checked_modular_recipes_preflight_without_retaining_schedules() {
        let capabilities =
            RealizationCapabilities::from_labels(&["mcx".to_owned(), "controlled_swap".to_owned()])
                .unwrap();
        let mut config = LoweringConfig::EMPTY;
        config.capabilities = capabilities;
        let requests = [
            ModularRequest::Multiply {
                width: 2,
                factor: 2,
                modulus: 3,
                inverse: false,
                capabilities,
            },
            ModularRequest::MultiplyAdd {
                width: 2,
                modulus: 3,
                inverse: true,
                capabilities,
            },
            ModularRequest::Power {
                target_width: 2,
                exponent_width: 2,
                base: 2,
                modulus: 3,
                inverse: true,
                config,
            },
            ModularRequest::MultiPower {
                coordinate_count: 2,
                coordinate_width: 1,
                target_width: 2,
                base_width: 2,
                bases: vec![2, 2].into_boxed_slice(),
                modulus: 3,
                inverse: false,
                config,
            },
        ];
        for request in requests {
            assert!(request.plan(false).unwrap().steps.is_empty());
            let checked = request.check().unwrap();
            assert!(checked.metadata.step_count > 0);
            assert!(checked.is_unitary());
            let materialized = checked.materialize().unwrap();
            assert_eq!(materialized.steps.len(), checked.metadata.step_count);
            assert_eq!(materialized.scratch_width, checked.metadata.scratch_width);
            assert_eq!(
                materialized.child_realizations,
                checked.metadata.child_realizations
            );
        }
    }
}
