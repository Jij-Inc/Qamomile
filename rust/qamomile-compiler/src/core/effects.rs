//! Closed denotational effects, callable kinds, and target requirements.
//!
//! These facts are compiler-owned. They are reconstructed from the verified
//! operation graph and are never accepted from producer-provided caches.

use crate::operations::{GateOpcode, Opcode, QCoreOpcode, QuantumOpcode};
use crate::types::Stage;

/// Observable behavior retained by the core language.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum Effect {
    /// Coherent quantum-state evolution is performed.
    Quantum,
    /// Fresh coherent storage is introduced.
    Allocation,
    /// Coherent storage leaves the callable interface.
    Release,
    /// Quantum information is observed classically.
    Measurement,
    /// Coherent data is irreversibly discarded.
    Discard,
    /// Quantum state is irreversibly reset.
    Reset,
    /// Control depends on data available only within a shot.
    ShotControl,
    /// Control depends on launch-time host data.
    HostControl,
    /// Repeated executions may produce distinct outcomes.
    Probabilistic,
}

impl Effect {
    /// Return the stable language spelling of this effect.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Quantum => "quantum",
            Self::Allocation => "allocation",
            Self::Release => "release",
            Self::Measurement => "measurement",
            Self::Discard => "discard",
            Self::Reset => "reset",
            Self::ShotControl => "shot_control",
            Self::HostControl => "host_control",
            Self::Probabilistic => "probabilistic",
        }
    }
}

/// Compact immutable set of semantic effects.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EffectSet(u16);

impl EffectSet {
    pub(crate) const EMPTY: Self = Self(0);

    pub(crate) const fn only(effect: Effect) -> Self {
        Self(1_u16 << effect as u8)
    }

    pub(crate) const fn with(self, effect: Effect) -> Self {
        Self(self.0 | (1_u16 << effect as u8))
    }

    pub(crate) const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Return whether this set contains one effect.
    #[must_use]
    pub const fn contains(self, effect: Effect) -> bool {
        self.0 & (1_u16 << effect as u8) != 0
    }

    /// Iterate over present effects in stable semantic order.
    pub fn iter(self) -> impl Iterator<Item = Effect> {
        const ALL: [Effect; 9] = [
            Effect::Quantum,
            Effect::Allocation,
            Effect::Release,
            Effect::Measurement,
            Effect::Discard,
            Effect::Reset,
            Effect::ShotControl,
            Effect::HostControl,
            Effect::Probabilistic,
        ];
        ALL.into_iter().filter(move |effect| self.contains(*effect))
    }

    pub(crate) const fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }
}

/// Denotational class of a verified callable.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CallableKind {
    /// A purely classical map.
    Classical,
    /// A reversible coherent map.
    Unitary,
    /// A coherent embedding that may allocate or release state.
    Isometry,
    /// An irreversible quantum channel.
    Channel,
    /// A quantum instrument that exposes measurement outcomes.
    Instrument,
}

impl CallableKind {
    pub(crate) fn parse(value: &str) -> Option<Self> {
        match value {
            "classical" => Some(Self::Classical),
            "unitary" => Some(Self::Unitary),
            "isometry" => Some(Self::Isometry),
            "channel" => Some(Self::Channel),
            "instrument" => Some(Self::Instrument),
            _ => None,
        }
    }

    /// Return the stable language spelling of this denotational class.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Classical => "classical",
            Self::Unitary => "unitary",
            Self::Isometry => "isometry",
            Self::Channel => "channel",
            Self::Instrument => "instrument",
        }
    }
}

/// Independently proved callable transformation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum Capability {
    /// The callable has a reconstructed inverse transformation.
    Adjointable,
    /// The callable supports coherent control.
    Controllable,
}

impl Capability {
    /// Return the stable language spelling of this transformation proof.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Adjointable => "adjointable",
            Self::Controllable => "controllable",
        }
    }
}

/// Compact immutable set of callable capabilities.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CapabilitySet(u8);

impl CapabilitySet {
    pub(crate) const EMPTY: Self = Self(0);
    pub(crate) const COHERENT: Self =
        Self((1_u8 << Capability::Adjointable as u8) | (1_u8 << Capability::Controllable as u8));

    /// Return whether this set contains one transformation capability.
    #[must_use]
    pub const fn contains(self, capability: Capability) -> bool {
        self.0 & (1_u8 << capability as u8) != 0
    }

    /// Iterate over established capabilities in stable semantic order.
    pub fn iter(self) -> impl Iterator<Item = Capability> {
        const ALL: [Capability; 2] = [Capability::Adjointable, Capability::Controllable];
        ALL.into_iter()
            .filter(move |capability| self.contains(*capability))
    }
}

/// Capability that an execution target must provide.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum Requirement {
    /// The target must evaluate classical operations.
    ClassicalCompute,
    /// The entry artifact accepts launch-time parameters.
    ParameterizedExecution,
    /// The target must represent coherent quantum state.
    Quantum,
    /// The target must support measurement.
    Measurement,
    /// Measurement must be available before execution terminates.
    MidCircuitMeasurement,
    /// Shot-time branching must be available.
    DynamicBranch,
    /// Shot-time looping must be available.
    DynamicLoop,
    /// The target must record and replay a finite dynamic-loop history.
    DynamicLoopHistory,
    /// The target must implement coherent quantum control.
    CoherentControl,
    /// The host must dispatch among specialized executions.
    HostDispatch,
    /// The target must reset coherent state.
    Reset,
    /// The target must discard coherent state.
    Discard,
    /// The target must implement parameterized rotations.
    Rotation,
    /// A rotation angle depends on measurements within a placed quantum region.
    DynamicRotation,
    /// A fixed classical aggregate is indexed only at launch or shot time.
    DynamicClassicalIndex,
    /// Qubit selection is unavailable until launch or shot time.
    DynamicQubitSelection,
    /// Execution crosses an opaque external callable boundary.
    ExternalCall,
}

impl Requirement {
    /// Return the stable execution-contract spelling of this requirement.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ClassicalCompute => "classical_compute",
            Self::ParameterizedExecution => "parameterized_execution",
            Self::Quantum => "quantum",
            Self::Measurement => "measurement",
            Self::MidCircuitMeasurement => "mid_circuit_measurement",
            Self::DynamicBranch => "dynamic_branch",
            Self::DynamicLoop => "dynamic_loop",
            Self::DynamicLoopHistory => "dynamic_loop_history",
            Self::CoherentControl => "coherent_control",
            Self::HostDispatch => "host_dispatch",
            Self::Reset => "reset",
            Self::Discard => "discard",
            Self::Rotation => "rotation",
            Self::DynamicRotation => "dynamic_rotation",
            Self::DynamicClassicalIndex => "dynamic_classical_index",
            Self::DynamicQubitSelection => "dynamic_qubit_selection",
            Self::ExternalCall => "external_call",
        }
    }

    /// Parse one stable execution-contract spelling.
    #[must_use]
    pub fn from_contract_str(value: &str) -> Option<Self> {
        Self::all()
            .into_iter()
            .find(|requirement| requirement.as_str() == value)
    }

    /// Return every requirement in stable target-contract order.
    #[must_use]
    pub const fn all() -> [Self; 17] {
        [
            Self::ClassicalCompute,
            Self::ParameterizedExecution,
            Self::Quantum,
            Self::Measurement,
            Self::MidCircuitMeasurement,
            Self::DynamicBranch,
            Self::DynamicLoop,
            Self::DynamicLoopHistory,
            Self::CoherentControl,
            Self::HostDispatch,
            Self::Reset,
            Self::Discard,
            Self::Rotation,
            Self::DynamicRotation,
            Self::DynamicClassicalIndex,
            Self::DynamicQubitSelection,
            Self::ExternalCall,
        ]
    }
}

/// Compact immutable set of target requirements.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RequirementSet(u32);

impl RequirementSet {
    pub(crate) const EMPTY: Self = Self(0);

    pub(crate) const fn only(requirement: Requirement) -> Self {
        Self(1_u32 << requirement as u8)
    }

    pub(crate) const fn with(self, requirement: Requirement) -> Self {
        Self(self.0 | (1_u32 << requirement as u8))
    }

    pub(crate) const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub(crate) const fn without(self, requirement: Requirement) -> Self {
        Self(self.0 & !(1_u32 << requirement as u8))
    }

    /// Return whether this set contains one target requirement.
    #[must_use]
    pub const fn contains(self, requirement: Requirement) -> bool {
        self.0 & (1_u32 << requirement as u8) != 0
    }

    /// Return whether the set contains no requirements.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Return requirements present here and absent from `other`.
    #[must_use]
    pub const fn difference(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }

    /// Iterate over present requirements in stable target-contract order.
    pub fn iter(self) -> impl Iterator<Item = Requirement> {
        Requirement::all()
            .into_iter()
            .filter(move |requirement| self.contains(*requirement))
    }
}

/// Return the direct body effect of one operation.
pub(crate) const fn operation_effects(opcode: Opcode) -> EffectSet {
    use Effect::{Allocation, Discard, Measurement, Probabilistic, Quantum, Release, Reset};

    match opcode {
        Opcode::Classical(_)
        | Opcode::QCore(
            QCoreOpcode::Constant
            | QCoreOpcode::Reshape
            | QCoreOpcode::StageLift
            | QCoreOpcode::Tuple
            | QCoreOpcode::Unpack
            | QCoreOpcode::Call
            | QCoreOpcode::If
            | QCoreOpcode::For
            | QCoreOpcode::While,
        ) => EffectSet::EMPTY,
        Opcode::QCore(
            QCoreOpcode::Adjoint
            | QCoreOpcode::AdjointCall
            | QCoreOpcode::ControlCall
            | QCoreOpcode::SelectCall
            | QCoreOpcode::PhaseBy
            | QCoreOpcode::XorBy
            | QCoreOpcode::PermuteBy
            | QCoreOpcode::Forget,
        )
        | Opcode::Quantum(
            QuantumOpcode::GlobalPhase
            | QuantumOpcode::PauliEvolution
            | QuantumOpcode::AddInto
            | QuantumOpcode::SubtractInto
            | QuantumOpcode::AddConstant
            | QuantumOpcode::SubtractConstant
            | QuantumOpcode::ModAdd
            | QuantumOpcode::ModSub
            | QuantumOpcode::ModAddConstant
            | QuantumOpcode::ModSubConstant
            | QuantumOpcode::MulAdd
            | QuantumOpcode::MulSub
            | QuantumOpcode::ModMulAdd
            | QuantumOpcode::ModMulSub
            | QuantumOpcode::ModMul
            | QuantumOpcode::ModDiv
            | QuantumOpcode::ModPow
            | QuantumOpcode::ModPowInverse
            | QuantumOpcode::ModMultiPow
            | QuantumOpcode::ModMultiPowInverse
            | QuantumOpcode::LessThanXor
            | QuantumOpcode::Gate(_),
        ) => EffectSet::only(Quantum),
        Opcode::Quantum(QuantumOpcode::Alloc | QuantumOpcode::Prepare) => {
            EffectSet::only(Quantum).with(Allocation)
        }
        Opcode::Quantum(QuantumOpcode::Measure) => EffectSet::only(Quantum)
            .with(Measurement)
            .with(Probabilistic)
            .with(Release),
        Opcode::Quantum(
            QuantumOpcode::MeasureX | QuantumOpcode::MeasureY | QuantumOpcode::MeasureZ,
        ) => EffectSet::only(Quantum)
            .with(Measurement)
            .with(Probabilistic),
        Opcode::Quantum(QuantumOpcode::MeasureReset) => EffectSet::only(Quantum)
            .with(Measurement)
            .with(Probabilistic)
            .with(Reset),
        Opcode::Quantum(QuantumOpcode::Discard) => {
            EffectSet::only(Quantum).with(Discard).with(Release)
        }
        Opcode::Quantum(QuantumOpcode::Reset) => EffectSet::only(Quantum).with(Reset),
        Opcode::Quantum(QuantumOpcode::Release | QuantumOpcode::Unprepare) => {
            EffectSet::only(Quantum).with(Release)
        }
    }
}

/// Return the direct target requirement of one operation.
pub(crate) const fn operation_requirements(opcode: Opcode) -> RequirementSet {
    use Requirement::{ClassicalCompute, CoherentControl, MidCircuitMeasurement, Rotation};

    match opcode {
        Opcode::Classical(_) => RequirementSet::only(ClassicalCompute),
        Opcode::QCore(QCoreOpcode::ControlCall | QCoreOpcode::SelectCall) => {
            RequirementSet::only(CoherentControl)
        }
        Opcode::Quantum(
            QuantumOpcode::MeasureReset
            | QuantumOpcode::MeasureX
            | QuantumOpcode::MeasureY
            | QuantumOpcode::MeasureZ,
        ) => RequirementSet::only(MidCircuitMeasurement),
        Opcode::Quantum(
            QuantumOpcode::GlobalPhase
            | QuantumOpcode::PauliEvolution
            | QuantumOpcode::Gate(
                GateOpcode::P
                | GateOpcode::Rx
                | GateOpcode::Ry
                | GateOpcode::Rz
                | GateOpcode::Cp
                | GateOpcode::Rzz,
            ),
        ) => RequirementSet::only(Rotation),
        Opcode::QCore(_)
        | Opcode::Quantum(
            QuantumOpcode::Alloc
            | QuantumOpcode::Prepare
            | QuantumOpcode::Measure
            | QuantumOpcode::Discard
            | QuantumOpcode::Reset
            | QuantumOpcode::Release
            | QuantumOpcode::Unprepare
            | QuantumOpcode::AddInto
            | QuantumOpcode::SubtractInto
            | QuantumOpcode::AddConstant
            | QuantumOpcode::SubtractConstant
            | QuantumOpcode::ModAdd
            | QuantumOpcode::ModSub
            | QuantumOpcode::ModAddConstant
            | QuantumOpcode::ModSubConstant
            | QuantumOpcode::MulAdd
            | QuantumOpcode::MulSub
            | QuantumOpcode::ModMulAdd
            | QuantumOpcode::ModMulSub
            | QuantumOpcode::ModMul
            | QuantumOpcode::ModDiv
            | QuantumOpcode::ModPow
            | QuantumOpcode::ModPowInverse
            | QuantumOpcode::ModMultiPow
            | QuantumOpcode::ModMultiPowInverse
            | QuantumOpcode::LessThanXor
            | QuantumOpcode::Gate(_),
        ) => RequirementSet::EMPTY,
    }
}

/// Return the effect introduced by availability-dependent control.
pub(crate) const fn control_effect(stage: Option<Stage>) -> EffectSet {
    match stage {
        Some(Stage::Param) => EffectSet::only(Effect::HostControl),
        Some(Stage::Shot) => EffectSet::only(Effect::ShotControl),
        None | Some(Stage::Const) => EffectSet::EMPTY,
    }
}

/// Convert semantic effects to the minimum target requirements they imply.
pub(crate) const fn requirements_for_effects(effects: EffectSet) -> RequirementSet {
    let mut requirements = RequirementSet::EMPTY;
    if effects.contains(Effect::Quantum) {
        requirements = requirements.with(Requirement::Quantum);
    }
    if effects.contains(Effect::Measurement) {
        requirements = requirements.with(Requirement::Measurement);
    }
    if effects.contains(Effect::ShotControl) {
        requirements = requirements.with(Requirement::DynamicBranch);
    }
    if effects.contains(Effect::HostControl) {
        requirements = requirements.with(Requirement::HostDispatch);
    }
    if effects.contains(Effect::Reset) {
        requirements = requirements.with(Requirement::Reset);
    }
    if effects.contains(Effect::Discard) {
        requirements = requirements.with(Requirement::Discard);
    }
    requirements
}

/// Infer one callable's denotational kind from its verified signature and effects.
pub(crate) const fn infer_callable_kind(
    inputs_have_quantum: bool,
    output_is_quantum: bool,
    effects: EffectSet,
) -> CallableKind {
    if !inputs_have_quantum && !output_is_quantum && !effects.contains(Effect::Quantum) {
        CallableKind::Classical
    } else if effects.contains(Effect::Measurement) {
        CallableKind::Instrument
    } else if effects.intersects(EffectSet::only(Effect::Discard).with(Effect::Reset)) {
        CallableKind::Channel
    } else if effects.intersects(EffectSet::only(Effect::Allocation).with(Effect::Release)) {
        CallableKind::Isometry
    } else {
        CallableKind::Unitary
    }
}

/// Return whether an asserted kind safely over-approximates a derived kind.
pub(crate) const fn asserted_kind_accepts(asserted: CallableKind, derived: CallableKind) -> bool {
    match asserted {
        CallableKind::Classical => matches!(derived, CallableKind::Classical),
        CallableKind::Unitary => matches!(derived, CallableKind::Unitary),
        CallableKind::Isometry => {
            matches!(derived, CallableKind::Unitary | CallableKind::Isometry)
        }
        CallableKind::Channel => matches!(
            derived,
            CallableKind::Unitary | CallableKind::Isometry | CallableKind::Channel
        ),
        CallableKind::Instrument => matches!(derived, CallableKind::Instrument),
    }
}

/// Return conservative effects for an opaque external callable.
pub(crate) const fn external_effects(kind: CallableKind) -> Option<EffectSet> {
    match kind {
        CallableKind::Classical => None,
        CallableKind::Unitary => Some(EffectSet::only(Effect::Quantum)),
        CallableKind::Isometry => Some(EffectSet::only(Effect::Quantum).with(Effect::Allocation)),
        CallableKind::Channel => Some(EffectSet::only(Effect::Quantum).with(Effect::Probabilistic)),
        CallableKind::Instrument => Some(
            EffectSet::only(Effect::Quantum)
                .with(Effect::Measurement)
                .with(Effect::Probabilistic),
        ),
    }
}

/// Return transformations proved by a derived callable kind.
pub(crate) const fn capabilities_for(kind: CallableKind) -> CapabilitySet {
    match kind {
        CallableKind::Unitary => CapabilitySet::COHERENT,
        CallableKind::Classical
        | CallableKind::Isometry
        | CallableKind::Channel
        | CallableKind::Instrument => CapabilitySet::EMPTY,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measurement_reset_requires_mid_circuit_measurement_and_reset() {
        let opcode = Opcode::Quantum(QuantumOpcode::MeasureReset);
        let direct = operation_requirements(opcode);
        let derived = requirements_for_effects(operation_effects(opcode));

        assert!(direct.contains(Requirement::MidCircuitMeasurement));
        assert!(derived.contains(Requirement::Measurement));
        assert!(derived.contains(Requirement::Reset));
    }

    #[test]
    fn callable_kind_lattice_distinguishes_reversible_and_irreversible_maps() {
        assert_eq!(
            infer_callable_kind(true, true, EffectSet::only(Effect::Quantum)),
            CallableKind::Unitary
        );
        assert_eq!(
            infer_callable_kind(
                true,
                true,
                EffectSet::only(Effect::Quantum).with(Effect::Reset),
            ),
            CallableKind::Channel
        );
        assert_eq!(
            infer_callable_kind(
                true,
                false,
                EffectSet::only(Effect::Quantum).with(Effect::Measurement),
            ),
            CallableKind::Instrument
        );
    }

    #[test]
    fn only_derived_unitaries_receive_coherent_capabilities() {
        let unitary = capabilities_for(CallableKind::Unitary);
        assert!(unitary.contains(Capability::Adjointable));
        assert!(unitary.contains(Capability::Controllable));
        assert_eq!(
            capabilities_for(CallableKind::Isometry),
            CapabilitySet::EMPTY
        );
    }

    #[test]
    fn assertion_compatibility_is_an_explicit_overapproximation_lattice() {
        assert!(asserted_kind_accepts(
            CallableKind::Channel,
            CallableKind::Unitary
        ));
        assert!(asserted_kind_accepts(
            CallableKind::Isometry,
            CallableKind::Unitary
        ));
        assert!(!asserted_kind_accepts(
            CallableKind::Unitary,
            CallableKind::Isometry
        ));
        assert!(!asserted_kind_accepts(
            CallableKind::Instrument,
            CallableKind::Channel
        ));
    }

    #[test]
    fn all_classical_operations_require_classical_compute() {
        assert!(
            operation_requirements(Opcode::Classical(
                crate::operations::ClassicalOpcode::CompareChain,
            ))
            .contains(Requirement::ClassicalCompute)
        );
    }

    #[test]
    fn an_external_classical_assertion_has_no_quantum_effect_contract() {
        assert_eq!(external_effects(CallableKind::Classical), None);
    }
}
