//! Target-neutral static circuit projection for verified programs.
//!
//! The projection deliberately starts with a closed, exact subset. Unsupported
//! dynamic values fail before any partial circuit escapes, while allocation,
//! primitive discrete gates, measurement, and ordinary calls share the native
//! specialization-time scalar semantics used by resource analysis.
//! Bounded measurement-driven loop exits become scalar selects and conditional
//! instruction blocks at projection, preserving carry values and loop-else flow.

use std::collections::{BTreeSet, HashMap, HashSet};

use super::shot_execution::{CircuitScalarCheck, ClosedQuantumFrontier};
use std::sync::{Arc, OnceLock};

#[path = "circuit/boolean_store.rs"]
mod boolean_store;
#[path = "circuit/deferred.rs"]
mod deferred;
mod optimization;
pub use deferred::CircuitDefinition;
use deferred::{CallCacheEntry, CallCacheKey};

use num_bigint::BigInt;
use num_traits::ToPrimitive;
use prost::Message;
use thiserror::Error;

use crate::contracts::requires_launch_scalar_evaluation;
use crate::control_state::ControlState;
use crate::lowering::arithmetic::plan_addition;
use crate::lowering::arithmetic_permutation::{ArithmeticOperation, plan_arithmetic_permutation};
use crate::lowering::modular_realization::{
    ModularGatePlan, ModularRealizationPlan, ModularRequest, plan_mod_multi_pow, plan_mod_pow,
    plan_schoolbook_mod_mul, plan_schoolbook_mod_mul_add,
};
use crate::lowering::realization::{
    ConfiguredRealization, LoweringConfig, Realization, RealizationCapabilities, RealizationError,
    RealizationFamily,
};
use crate::numeric_values::NumericEvaluationError;
use crate::operations::{PrimitiveGateSpec, QuantumOpcode, primitive_gate_spec};
use crate::pauli_evolution::{
    PauliEvolutionError, PauliEvolutionGate, plan_pauli_evolution_with_capabilities,
    validate_pauli_evolution,
};
use crate::phase_lifting;
use crate::static_values::{
    self, StaticScalar, StaticValue, normalize_aggregate_index, range_len as static_range_len,
    truth as static_truth,
};
use crate::types::{SemanticType, TypeArena, TypeId};
use crate::wire;

const MAX_STATIC_CIRCUIT_INSTRUCTIONS: usize = 1_000_000;
const MAX_STATIC_LOOP_ITERATIONS: u128 = 1_000_000;
const MAX_STATIC_CIRCUIT_VARIANTS: usize = 256;
// Bound host stack use and total expansion work when projecting recursive calls.
const MAX_CIRCUIT_CALL_DEPTH: usize = 64;
const MAX_CIRCUIT_OPERATION_VISITS: u128 = 10_000_000;

/// Stable index of one hash-consed scalar circuit expression.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CircuitScalarId(u32);

impl CircuitScalarId {
    pub(crate) const fn from_index(index: u32) -> Self {
        Self(index)
    }

    #[cfg(test)]
    pub(crate) const fn test_new(index: u32) -> Self {
        Self(index)
    }

    /// Return the zero-based scalar arena index.
    #[must_use]
    pub const fn index(self) -> u32 {
        self.0
    }
}

/// Closed scalar type universe supported by the circuit-expression DAG.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CircuitScalarType {
    /// Classical truth value.
    Bool,
    /// Signed integer value.
    Int,
    /// IEEE-754 binary64 value.
    Real,
}

/// One canonical structural component selecting a scalar parameter leaf.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum CircuitParameterPath {
    /// Zero-based position in an array, word, or anonymous product.
    Index(u64),
    /// Stable field name in a named record.
    Field(String),
}

/// One closed exact value supplied at a verified launch boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CircuitLaunchValue {
    /// Classical truth value.
    Bool(bool),
    /// Signed integer value.
    Int(i64),
    /// Canonical signed arbitrary-precision integer bytes.
    BigInt(Box<[u8]>),
    /// IEEE-754 binary64 value retained by exact bits.
    RealBits(u64),
    /// Complex binary64 coordinates retained by exact bits.
    ComplexBits(u64, u64),
    /// One source-ordered positional aggregate.
    Aggregate(Box<[Self]>),
    /// One named record in semantic declaration order.
    Record(Box<[(String, Self)]>),
}

/// One named entrypoint value supplied to native execution or materialization.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CircuitLaunchBinding {
    name: String,
    value: CircuitLaunchValue,
}

impl CircuitLaunchBinding {
    /// Construct one named launch binding.
    #[must_use]
    pub fn new(name: String, value: CircuitLaunchValue) -> Self {
        Self { name, value }
    }

    /// Return the entrypoint parameter name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Return the closed launch value.
    #[must_use]
    pub const fn value(&self) -> &CircuitLaunchValue {
        &self.value
    }
}

/// One closed portable value retained as scalar-operation metadata.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum CircuitScalarAttributeValue {
    /// Exact signed integer metadata.
    Int(i64),
    /// Exact Unicode string metadata.
    String(String),
    /// Source-ordered tuple of exact Unicode strings.
    Strings(Box<[String]>),
}

/// One named attribute on a target-neutral scalar operation.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct CircuitScalarAttribute {
    name: String,
    value: CircuitScalarAttributeValue,
}

impl CircuitScalarAttribute {
    /// Construct one checked scalar attribute inside the compiler crate.
    pub(crate) fn from_parts(name: String, value: CircuitScalarAttributeValue) -> Self {
        Self { name, value }
    }

    /// Return the canonical attribute name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Return the closed portable attribute value.
    #[must_use]
    pub const fn value(&self) -> &CircuitScalarAttributeValue {
        &self.value
    }
}

/// One immutable node in the target-neutral scalar expression DAG.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum CircuitScalarNode {
    /// Exact Boolean literal.
    Bool(bool),
    /// Exact signed integer literal.
    Int(i64),
    /// Exact IEEE-754 binary64 literal stored by bits for total equality.
    RealBits(u64),
    /// One source-ordered device measurement available to runtime control.
    Measurement(u64),
    /// Scoped index of one retained loop; never a launch or result value.
    LoopIndex(u32),
    /// One scalar returned by a structured runtime call frame.
    CallValue {
        /// Stable call-result storage index.
        index: u64,
        /// Verified scalar type of the returned leaf.
        scalar_type: CircuitScalarType,
    },
    /// One scalar launch parameter selected by a canonical structural path.
    Parameter {
        /// User-facing entrypoint parameter name.
        name: String,
        /// Zero or more fixed aggregate projections from the launch slot.
        path: Box<[CircuitParameterPath]>,
        /// Verified scalar leaf type.
        scalar_type: CircuitScalarType,
    },
    /// One classical operation over scalar DAG operands and exact metadata.
    Apply {
        /// Canonical classical opcode.
        opcode: String,
        /// Verified operation result type.
        result_type: CircuitScalarType,
        /// Source-ordered operation metadata.
        attributes: Box<[CircuitScalarAttribute]>,
        /// Source-ordered scalar operands.
        operands: Box<[CircuitScalarId]>,
    },
}

impl CircuitScalarNode {
    /// Return the verified result type of this scalar node.
    #[must_use]
    pub const fn scalar_type(&self) -> CircuitScalarType {
        match self {
            Self::Bool(_) | Self::Measurement(_) => CircuitScalarType::Bool,
            Self::Int(_) | Self::LoopIndex(_) => CircuitScalarType::Int,
            Self::RealBits(_) => CircuitScalarType::Real,
            Self::CallValue { scalar_type, .. } | Self::Parameter { scalar_type, .. } => {
                *scalar_type
            }
            Self::Apply { result_type, .. } => *result_type,
        }
    }
}

/// One typed parameter retained by a target-neutral circuit instruction.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CircuitParameter {
    /// Source-ordered measurement-storage index for an instruction.
    MeasurementIndex(u64),
    /// Scalar angle or phase expression retained by a unitary instruction.
    Scalar(CircuitScalarId),
    /// One scalar leaf written when a structured call frame returns.
    CallResult {
        /// Stable call-result storage slot.
        target: u64,
        /// Verified scalar type of the returned leaf.
        scalar_type: CircuitScalarType,
    },
    /// Canonical portable Hamiltonian literal owned by one instruction.
    Hamiltonian(Box<[u8]>),
    /// Compiler-verified ANF, shared with the portable Boolean realization.
    BooleanPlan(phase_lifting::BooleanLiftPlan),
    /// Shared checked local definition, owned for the executable lifetime.
    Definition(CircuitDefinition),
}

/// One target-neutral instruction in a statically projected circuit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StaticInstruction {
    opcode: String,
    qubits: Box<[u64]>,
    parameters: Box<[CircuitParameter]>,
    controls: Box<[u64]>,
    condition: Option<CircuitScalarId>,
    regions: Box<[StaticBlock]>,
}

/// One nested target-neutral instruction block.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StaticBlock {
    instructions: Box<[StaticInstruction]>,
    return_value: Option<StaticCircuitValue>,
}

impl StaticBlock {
    pub(crate) fn instructions_mut(&mut self) -> &mut [StaticInstruction] {
        &mut self.instructions
    }

    pub(crate) fn replace_instructions(&mut self, instructions: Vec<StaticInstruction>) {
        self.instructions = instructions.into_boxed_slice();
    }
    #[cfg(test)]
    pub(crate) fn test_new(instructions: Vec<StaticInstruction>) -> Self {
        Self {
            instructions: instructions.into_boxed_slice(),
            return_value: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn test_returning(
        instructions: Vec<StaticInstruction>,
        return_value: StaticCircuitValue,
    ) -> Self {
        Self {
            instructions: instructions.into_boxed_slice(),
            return_value: Some(return_value),
        }
    }

    /// Return source-ordered instructions in this block.
    #[must_use]
    pub fn instructions(&self) -> &[StaticInstruction] {
        &self.instructions
    }

    /// Return the typed value carried by a runtime return exit, when present.
    #[must_use]
    pub fn return_value(&self) -> Option<StaticCircuitResultValue> {
        self.return_value
            .as_ref()
            .and_then(project_static_circuit_result)
    }

    /// Return whether this block exits its enclosing callable.
    #[must_use]
    pub const fn has_runtime_return(&self) -> bool {
        self.return_value.is_some()
    }

    pub(crate) const fn runtime_return_value(&self) -> Option<&StaticCircuitValue> {
        self.return_value.as_ref()
    }
}

impl StaticInstruction {
    pub(crate) fn regions_mut(&mut self) -> &mut [StaticBlock] {
        &mut self.regions
    }

    pub(crate) fn parameters_mut(&mut self) -> &mut [CircuitParameter] {
        &mut self.parameters
    }

    pub(crate) fn remap_segment(
        &mut self,
        wires: &HashMap<u64, u64>,
        measurements: &HashMap<u64, u64>,
    ) -> Result<(), StaticCircuitError> {
        for wire in self.qubits.iter_mut().chain(self.controls.iter_mut()) {
            *wire = *wires
                .get(wire)
                .ok_or_else(|| StaticCircuitError::Missing("segment wire".into()))?;
        }
        for parameter in &mut self.parameters {
            if let CircuitParameter::MeasurementIndex(index) = parameter {
                *index = *measurements.get(index).ok_or_else(|| {
                    StaticCircuitError::Missing("segment measurement index".to_owned())
                })?;
            }
        }
        Ok(())
    }
    #[cfg(test)]
    pub(crate) fn test_new(
        opcode: &str,
        qubits: Vec<u64>,
        parameters: Vec<CircuitParameter>,
        controls: Vec<u64>,
        condition: Option<CircuitScalarId>,
        regions: Vec<StaticBlock>,
    ) -> Self {
        Self {
            opcode: opcode.to_owned(),
            qubits: qubits.into_boxed_slice(),
            parameters: parameters.into_boxed_slice(),
            controls: controls.into_boxed_slice(),
            condition,
            regions: regions.into_boxed_slice(),
        }
    }

    /// Return the logical opcode without the `quantum.` prefix.
    #[must_use]
    pub fn opcode(&self) -> &str {
        &self.opcode
    }

    /// Return source-ordered logical wire indices.
    #[must_use]
    pub fn qubits(&self) -> &[u64] {
        &self.qubits
    }

    /// Return typed target-neutral instruction parameters.
    #[must_use]
    pub fn parameters(&self) -> &[CircuitParameter] {
        &self.parameters
    }

    /// Return source-ordered coherent control wires.
    #[must_use]
    pub fn controls(&self) -> &[u64] {
        &self.controls
    }

    /// Return the optional runtime condition expression.
    #[must_use]
    pub const fn condition(&self) -> Option<CircuitScalarId> {
        self.condition
    }

    /// Return source-ordered structured regions.
    #[must_use]
    pub fn regions(&self) -> &[StaticBlock] {
        &self.regions
    }
}

/// A closed target-neutral circuit projected without runtime specialization.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StaticCircuit {
    optimization_stages: Vec<String>,
    closed_cuts: Box<[ClosedQuantumFrontier]>,
    scalar_checks: Box<[CircuitScalarCheck]>,
    realizations: Box<[ConfiguredRealization]>,
    width: u64,
    measurement_count: u64,
    operation_count: usize,
    instructions: Box<[StaticInstruction]>,
    scalars: Box<[CircuitScalarNode]>,
    result_measurements: Box<[u64]>,
    result_value: Option<StaticCircuitValue>,
}

/// Diagnostic work charged while projecting one target-neutral circuit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StaticCircuitWork {
    operation_visits: u128,
}

impl StaticCircuitWork {
    /// Return the number of dynamically visited `QCore` operations.
    ///
    /// This value describes compiler work and is not a circuit resource metric.
    #[must_use]
    pub const fn operation_visits(self) -> u128 {
        self.operation_visits
    }
}

/// One public result expression retained by a target-neutral static circuit.
///
/// Scalar nodes refer to the circuit's compact scalar arena. Literal payloads
/// are canonical `LiteralProto` bytes so this public boundary does not expose
/// the compiler's private constant-propagation representation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StaticCircuitResultValue {
    /// The sole unit value.
    Unit,
    /// One exact portable literal encoded canonically.
    Literal(Box<[u8]>),
    /// One scalar-expression root.
    Scalar(CircuitScalarId),
    /// One positional product or array.
    Aggregate(Box<[Self]>),
    /// One record in semantic declaration order.
    Record(Box<[(String, Self)]>),
}

/// One portable value retained across static-circuit execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum StaticCircuitValue {
    /// The sole unit value.
    Unit,
    /// One exact value outside the runtime scalar-expression universe.
    Portable(StaticValue),
    /// One scalar-expression root.
    Scalar(CircuitScalarId),
    /// Compile-time identity of an owned quantum value crossing a return.
    Wires(Box<[u64]>),
    /// One positional product or array.
    Aggregate(Box<[Self]>),
    /// One record in semantic declaration order.
    Record(Box<[(String, Self)]>),
}

/// A retained result paired with its compact, result-reachable scalar arena.
pub(crate) type CompactResultPlan = (Box<[CircuitScalarNode]>, StaticCircuitValue);

/// One finite launch parameter selected by host dispatch.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CircuitDispatchParameter {
    parameter_index: usize,
    name: String,
    uint_width: Option<u64>,
}

impl CircuitDispatchParameter {
    /// Reconstruct one checked parameter identity from a persistent selector.
    pub(crate) fn from_parts(
        parameter_index: usize,
        name: String,
        uint_width: Option<u64>,
    ) -> Self {
        Self {
            parameter_index,
            name,
            uint_width,
        }
    }

    /// Return the source-ordered entrypoint parameter index.
    #[must_use]
    pub const fn parameter_index(&self) -> usize {
        self.parameter_index
    }

    /// Return the user-facing entrypoint parameter name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Return the fixed unsigned-word width represented by one Python integer.
    #[must_use]
    pub const fn uint_width(&self) -> Option<u64> {
        self.uint_width
    }
}

/// One verified Boolean expression evaluated before backend execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CircuitDispatchPredicate {
    root: CircuitScalarId,
    scalars: Box<[CircuitScalarNode]>,
}

impl CircuitDispatchPredicate {
    /// Reconstruct one checked predicate from a persistent selection plan.
    pub(crate) fn from_parts(root: CircuitScalarId, scalars: Box<[CircuitScalarNode]>) -> Self {
        Self { root, scalars }
    }

    /// Return the root Boolean expression identifier.
    #[must_use]
    pub const fn root(&self) -> CircuitScalarId {
        self.root
    }

    /// Return the closed, topologically ordered scalar-expression arena.
    #[must_use]
    pub fn scalars(&self) -> &[CircuitScalarNode] {
        &self.scalars
    }

    /// Return unique launch parameter names referenced by this predicate.
    #[must_use]
    pub fn parameter_names(&self) -> Vec<&str> {
        unique_scalar_parameter_names(self.scalars.iter())
    }
}

/// One truth assignment along a circuit-family decision-tree path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CircuitDispatchDecision {
    predicate_index: usize,
    value: bool,
}

impl CircuitDispatchDecision {
    /// Reconstruct one checked decision from a persistent selection plan.
    pub(crate) const fn from_parts(predicate_index: usize, value: bool) -> Self {
        Self {
            predicate_index,
            value,
        }
    }

    /// Return the zero-based family predicate index.
    #[must_use]
    pub const fn predicate_index(self) -> usize {
        self.predicate_index
    }

    /// Return the truth value required by this variant.
    #[must_use]
    pub const fn value(self) -> bool {
        self.value
    }
}

/// One statically projected circuit and its exact launch dispatch key.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StaticCircuitVariant {
    decisions: Box<[CircuitDispatchDecision]>,
    circuit: StaticCircuit,
}

impl StaticCircuitVariant {
    /// Return the verified predicate assignments selecting this circuit.
    #[must_use]
    pub fn decisions(&self) -> &[CircuitDispatchDecision] {
        &self.decisions
    }

    /// Return the statically projected circuit for these bindings.
    #[must_use]
    pub const fn circuit(&self) -> &StaticCircuit {
        &self.circuit
    }
}

/// A finite target-neutral family for launch-time host dispatch.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StaticCircuitFamily {
    dispatch_parameters: Box<[CircuitDispatchParameter]>,
    predicates: Box<[CircuitDispatchPredicate]>,
    variants: Box<[StaticCircuitVariant]>,
}

impl StaticCircuitFamily {
    pub(crate) fn map_circuits(
        mut self,
        mut transform: impl FnMut(StaticCircuit) -> Result<StaticCircuit, StaticCircuitError>,
    ) -> Result<Self, StaticCircuitError> {
        self.variants = self
            .variants
            .into_vec()
            .into_iter()
            .map(|variant| {
                Ok(StaticCircuitVariant {
                    decisions: variant.decisions,
                    circuit: transform(variant.circuit)?,
                })
            })
            .collect::<Result<Vec<_>, StaticCircuitError>>()?
            .into_boxed_slice();
        Ok(self)
    }
    /// Return the source-ordered scalar parameters used by host control.
    #[must_use]
    pub fn dispatch_parameters(&self) -> &[CircuitDispatchParameter] {
        &self.dispatch_parameters
    }

    /// Return the canonical launch predicates used by the decision tree.
    #[must_use]
    pub fn predicates(&self) -> &[CircuitDispatchPredicate] {
        &self.predicates
    }

    /// Return every circuit leaf in canonical false-first tree order.
    #[must_use]
    pub fn variants(&self) -> &[StaticCircuitVariant] {
        &self.variants
    }
}

impl StaticCircuit {
    /// Return source-proved empty quantum frontiers and measurement prefixes.
    #[must_use]
    pub fn closed_cuts(&self) -> &[ClosedQuantumFrontier] {
        &self.closed_cuts
    }

    pub(crate) fn scalar_checks(&self) -> &[CircuitScalarCheck] {
        &self.scalar_checks
    }

    pub(crate) fn from_closed_segment(
        realizations: Vec<ConfiguredRealization>,
        width: u64,
        measurement_count: u64,
        instructions: Vec<StaticInstruction>,
        scalars: Box<[CircuitScalarNode]>,
        result: StaticCircuitValue,
        direct_measurements: bool,
    ) -> Self {
        Self {
            optimization_stages: Vec::new(),
            realizations: realizations.into_boxed_slice(),
            width,
            measurement_count,
            operation_count: instruction_count(&instructions),
            instructions: instructions.into_boxed_slice(),
            scalars,
            result_measurements: if direct_measurements {
                (0..measurement_count).collect()
            } else {
                Box::new([])
            },
            result_value: Some(result),
            closed_cuts: Box::new([]),
            scalar_checks: Box::new([]),
        }
    }
    /// Return the exact implementations selected during this projection.
    #[must_use]
    pub fn realizations(&self) -> &[ConfiguredRealization] {
        &self.realizations
    }
    #[cfg(test)]
    pub(crate) fn test_new(
        width: u64,
        measurement_count: u64,
        instructions: Vec<StaticInstruction>,
        scalars: Vec<CircuitScalarNode>,
        result_measurements: Vec<u64>,
    ) -> Self {
        let operation_count = instruction_count(&instructions);
        Self {
            optimization_stages: Vec::new(),
            realizations: Box::new([]),
            width,
            measurement_count,
            operation_count,
            instructions: instructions.into_boxed_slice(),
            scalars: scalars.into_boxed_slice(),
            result_measurements: result_measurements.into_boxed_slice(),
            result_value: None,
            closed_cuts: Box::new([]),
            scalar_checks: Box::new([]),
        }
    }

    #[cfg(test)]
    pub(crate) fn test_new_with_result(
        width: u64,
        measurement_count: u64,
        instructions: Vec<StaticInstruction>,
        scalars: Vec<CircuitScalarNode>,
        result_value: StaticCircuitValue,
    ) -> Self {
        let operation_count = instruction_count(&instructions);
        Self {
            optimization_stages: Vec::new(),
            realizations: Box::new([]),
            width,
            measurement_count,
            operation_count,
            instructions: instructions.into_boxed_slice(),
            scalars: scalars.into_boxed_slice(),
            result_measurements: Box::new([]),
            result_value: Some(result_value),
            closed_cuts: Box::new([]),
            scalar_checks: Box::new([]),
        }
    }

    /// Return the total number of allocated logical wires.
    #[must_use]
    pub const fn width(&self) -> u64 {
        self.width
    }

    /// Return the number of source-ordered measurement bits.
    #[must_use]
    pub const fn measurement_count(&self) -> u64 {
        self.measurement_count
    }

    /// Return the flat logical instruction sequence.
    #[must_use]
    pub fn instructions(&self) -> &[StaticInstruction] {
        &self.instructions
    }

    /// Return the total number of instructions across every structured block.
    #[must_use]
    pub const fn operation_count(&self) -> usize {
        self.operation_count
    }

    /// Return the hash-consed scalar expression arena.
    #[must_use]
    pub fn scalars(&self) -> &[CircuitScalarNode] {
        &self.scalars
    }

    /// Return unique launch parameter names retained by the compact scalar DAG.
    #[must_use]
    pub fn retained_parameter_names(&self) -> Vec<&str> {
        unique_scalar_parameter_names(self.scalars.iter())
    }

    /// Return source-ordered measurements retained by the entrypoint result.
    #[must_use]
    pub fn result_measurements(&self) -> &[u64] {
        &self.result_measurements
    }

    /// Return the public typed result-expression tree.
    ///
    /// Literal leaves use the same canonical protobuf representation as the
    /// rest of the compiler's portable ABI. A missing value is only possible
    /// for an entrypoint with no retained return operand.
    #[must_use]
    pub fn result_value(&self) -> Option<StaticCircuitResultValue> {
        self.result_value
            .as_ref()
            .and_then(project_static_circuit_result)
    }

    /// Return logical wire order for a single quantum-returning component.
    /// This is circuit composition metadata, never an executable result ABI.
    #[must_use]
    pub fn output_wires(&self) -> Option<&[u64]> {
        match self.result_value.as_ref()? {
            StaticCircuitValue::Wires(wires) => Some(wires),
            _ => None,
        }
    }

    /// Borrow the private result tree used by native execution.
    pub(crate) const fn runtime_result_value(&self) -> Option<&StaticCircuitValue> {
        self.result_value.as_ref()
    }

    pub(crate) fn has_terminal_checks(&self) -> bool {
        self.scalar_checks
            .iter()
            .any(|check| check.instruction_index == self.instructions.len())
    }

    /// Copy the retained result and only its reachable scalar DAG.
    pub(crate) fn compact_result_plan(
        &self,
    ) -> Result<Option<CompactResultPlan>, StaticCircuitError> {
        let Some(mut result) = self.result_value.clone() else {
            return Ok(None);
        };
        if self.has_terminal_checks() {
            let checks = self
                .scalar_checks
                .iter()
                .filter(|check| check.instruction_index == self.instructions.len())
                .map(|check| StaticCircuitValue::Scalar(check.root))
                .collect();
            result = StaticCircuitValue::Aggregate(
                vec![StaticCircuitValue::Aggregate(checks), result].into(),
            );
        }
        let scalars = compact_retained_result_scalars(&mut result, &self.scalars)?;
        Ok(Some((scalars, result)))
    }

    /// Reconstruct the scalar/result subset carried by a checked result plan.
    pub(crate) fn from_result_plan(
        measurement_count: u64,
        scalars: Box<[CircuitScalarNode]>,
        result: StaticCircuitValue,
    ) -> Self {
        Self {
            width: 0,
            optimization_stages: Vec::new(),
            realizations: Box::new([]),
            measurement_count,
            operation_count: 0,
            instructions: Box::new([]),
            scalars,
            result_measurements: Box::new([]),
            result_value: Some(result),
            closed_cuts: Box::new([]),
            scalar_checks: Box::new([]),
        }
    }
}

fn unique_scalar_parameter_names<'a>(
    nodes: impl Iterator<Item = &'a CircuitScalarNode>,
) -> Vec<&'a str> {
    let mut seen = HashSet::new();
    nodes
        .filter_map(|node| match node {
            CircuitScalarNode::Parameter { name, .. } if seen.insert(name.as_str()) => {
                Some(name.as_str())
            }
            _ => None,
        })
        .collect()
}

fn project_static_circuit_result(value: &StaticCircuitValue) -> Option<StaticCircuitResultValue> {
    match value {
        StaticCircuitValue::Unit => Some(StaticCircuitResultValue::Unit),
        StaticCircuitValue::Portable(value) => Some(StaticCircuitResultValue::Literal(
            static_values::value_to_literal(value)?
                .encode_to_vec()
                .into_boxed_slice(),
        )),
        StaticCircuitValue::Scalar(identifier) => {
            Some(StaticCircuitResultValue::Scalar(*identifier))
        }
        StaticCircuitValue::Wires(_) => None,
        StaticCircuitValue::Aggregate(values) => Some(StaticCircuitResultValue::Aggregate(
            values
                .iter()
                .map(project_static_circuit_result)
                .collect::<Option<Vec<_>>>()?
                .into_boxed_slice(),
        )),
        StaticCircuitValue::Record(fields) => Some(StaticCircuitResultValue::Record(
            fields
                .iter()
                .map(|(name, value)| Some((name.clone(), project_static_circuit_result(value)?)))
                .collect::<Option<Vec<_>>>()?
                .into_boxed_slice(),
        )),
    }
}

/// Failure to project a verified program without inventing target semantics.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum StaticCircuitError {
    /// Exact standard optimization failed before target-neutral projection.
    #[error("{0}")]
    Optimization(String),
    /// A requested realization is unavailable or exceeds its generation budget.
    #[error(transparent)]
    Realization(#[from] RealizationError),
    /// The verified program requires a circuit feature outside this exact slice.
    #[error("Q4301: static native circuit projection does not support {0}")]
    Unsupported(String),
    /// A verified structural reference was unexpectedly unavailable.
    #[error("Q4302: static native circuit projection lost {0}")]
    Missing(String),
    /// A statically known selector is outside the owned logical state.
    #[error("Q4303: invalid static circuit selector: {0}")]
    InvalidSelector(String),
    /// Logical wire or measurement numbering cannot be represented.
    #[error("Q4304: static native circuit projection overflow")]
    Overflow,
    /// Exact Pauli evolution was requested for a non-Hermitian observable.
    #[error("Q8036: Pauli evolution requires a Hermitian observable")]
    NonHermitianPauliEvolution,
    /// Exact Pauli evolution was requested for noncommuting terms.
    #[error("Q8037: exact Pauli evolution requires pairwise-commuting terms")]
    NonCommutingPauliEvolution,
    /// A circuit target cannot retain runtime classical aggregate addressing.
    #[error(
        "Q7246: circuit targets cannot dynamically address a classical aggregate; use a fixed Python for loop or specialize the index"
    )]
    DynamicClassicalAggregateIndex,
    /// Projection discovered a new launch predicate requiring host dispatch.
    #[error("Q4305: static native circuit projection requires launch predicate dispatch")]
    HostDispatch(CircuitDispatchPredicate),
    /// Exact launch values are required to materialize topology or checked scalars.
    #[error("Q4306: static native circuit projection requires launch materialization for {0}")]
    LaunchMaterialization(String),
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
enum CircuitValue {
    Scalar(CircuitScalarId),
    Portable(StaticValue),
    Hamiltonian(Box<[u8]>),
    Parameter {
        name: String,
        path: Box<[CircuitParameterPath]>,
        shape: ParameterShape,
    },
    Aggregate(Box<[CircuitValue]>),
    Wires(Box<[u64]>),
    Unit,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
enum ParameterShape {
    Array(u64),
    Product(u64),
    Record(Box<[String]>),
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Flow {
    kind: String,
    values: Vec<CircuitValue>,
}

enum OperationFlow {
    Values(Vec<CircuitValue>),
    Exit(Flow),
}

/// Predicates and carried values for one statically expanded loop. Exits are
/// represented with existing scalar selects and conditional blocks, so targets
/// consume the same measured-control representation as ordinary `if` statements.
#[derive(Clone)]
struct ExpandedLoop {
    running: CircuitScalarId,
    continuing: CircuitScalarId,
    exit_values: Vec<CircuitValue>,
}

type CompletedFamilyProjection = (Vec<bool>, Vec<CircuitDispatchPredicate>, StaticCircuit);

#[derive(Clone)]
struct StaticLowerer<'a> {
    entry_body: Option<&'a wire::BlockProto>,
    cut_operations: BTreeSet<usize>,
    closed_cuts: Vec<ClosedQuantumFrontier>,
    scalar_checks: Vec<CircuitScalarCheck>,
    measurement_guards: Vec<(CircuitScalarId, bool)>,
    realizations: BTreeSet<ConfiguredRealization>,
    boolean_plans: &'a phase_lifting::VerifiedBooleanLiftPlans,
    lowering: LoweringConfig,
    functions: HashMap<&'a str, &'a wire::FunctionProto>,
    next_wire: u64,
    next_measurement: u64,
    next_call_value: u64,
    instructions: Vec<StaticInstruction>,
    scalar_ids: HashMap<CircuitScalarNode, CircuitScalarId>,
    scalars: Vec<CircuitScalarNode>,
    controls: Vec<u64>,
    next_loop_index: u32,
    loop_projection_work: u128,
    coherent_partitions: HashMap<Box<[u64]>, Box<[usize]>>,
    schoolbook_workspace: Vec<u64>,
    instruction_count: usize,
    host_decisions: &'a [bool],
    dispatch_predicates: Vec<CircuitDispatchPredicate>,
    launch_scalars: &'a HashMap<LaunchScalarKey, StaticValue>,
    known_scalars: HashMap<CircuitScalarId, StaticValue>,
    result_types: Vec<wire::TypeProto>,
    expanded_loops: Vec<ExpandedLoop>,
    operation_visits: u128,
    aggregate_projection_work: usize,
    call_cache: HashMap<CallCacheKey, CallCacheEntry>,
    arithmetic_definitions: HashMap<
        crate::lowering::arithmetic_permutation::CheckedArithmeticPermutation,
        CircuitDefinition,
    >,
    modular_definitions: HashMap<ModularRequest, (CircuitDefinition, usize)>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct LaunchScalarKey {
    name: String,
    path: Box<[CircuitParameterPath]>,
}

pub(crate) fn project_with_lowering(
    package: &wire::QCorePackageProto,
    boolean_plans: &phase_lifting::VerifiedBooleanLiftPlans,
    lowering: LoweringConfig,
) -> Result<StaticCircuit, StaticCircuitError> {
    project_with_host_decisions(package, boolean_plans, lowering, &[], &HashMap::new())
        .map(|(circuit, _, _)| circuit)
}

pub(crate) fn project_with_work(
    package: &wire::QCorePackageProto,
    boolean_plans: &phase_lifting::VerifiedBooleanLiftPlans,
    lowering: LoweringConfig,
) -> Result<(StaticCircuit, StaticCircuitWork), StaticCircuitError> {
    project_with_host_decisions(package, boolean_plans, lowering, &[], &HashMap::new())
        .map(|(circuit, _, operation_visits)| (circuit, StaticCircuitWork { operation_visits }))
}

pub(crate) fn project_at(
    package: &wire::QCorePackageProto,
    boolean_plans: &phase_lifting::VerifiedBooleanLiftPlans,
    dispatch_parameter_indices: &[usize],
    bindings: &[CircuitLaunchBinding],
    lowering: LoweringConfig,
) -> Result<StaticCircuit, StaticCircuitError> {
    let entry = package
        .functions
        .iter()
        .find(|function| function.symbol == package.entrypoint)
        .ok_or_else(|| StaticCircuitError::Missing("entrypoint".to_owned()))?;
    let runtime_indices =
        unbound_control_parameter_indices(package, entry, dispatch_parameter_indices)?;
    let controlled_parameters = validate_dispatch_parameters(entry, &runtime_indices)?;
    let launch_scalars = validate_launch_bindings(entry, &controlled_parameters, bindings)?;
    project_with_host_decisions(package, boolean_plans, lowering, &[], &launch_scalars)
        .map(|(circuit, _, _)| circuit)
}

pub(crate) fn project_for_invocation(
    package: &wire::QCorePackageProto,
    boolean_plans: &phase_lifting::VerifiedBooleanLiftPlans,
    dispatch_parameter_indices: &[usize],
    bindings: &[CircuitLaunchBinding],
    lowering: LoweringConfig,
) -> Result<StaticCircuit, StaticCircuitError> {
    let launch_scalars = validated_launch_scalars(package, bindings)?;
    let entry = package
        .functions
        .iter()
        .find(|function| function.symbol == package.entrypoint)
        .ok_or_else(|| StaticCircuitError::Missing("entrypoint".to_owned()))?;
    let runtime_indices =
        unbound_control_parameter_indices(package, entry, dispatch_parameter_indices)?;
    validate_dispatch_parameters(entry, &runtime_indices)?;
    project_with_host_decisions(package, boolean_plans, lowering, &[], &launch_scalars)
        .map(|(circuit, _, _)| circuit)
}

/// Validate the complete exact entrypoint `Param` ABI without projecting it.
pub(crate) fn validate_launch_parameters(
    package: &wire::QCorePackageProto,
    bindings: &[CircuitLaunchBinding],
) -> Result<(), StaticCircuitError> {
    validated_launch_scalars(package, bindings).map(drop)
}

fn validated_launch_scalars(
    package: &wire::QCorePackageProto,
    bindings: &[CircuitLaunchBinding],
) -> Result<HashMap<LaunchScalarKey, StaticValue>, StaticCircuitError> {
    let entry = package
        .functions
        .iter()
        .find(|function| function.symbol == package.entrypoint)
        .ok_or_else(|| StaticCircuitError::Missing("entrypoint".to_owned()))?;
    let parameter_indices = entry
        .parameters
        .iter()
        .enumerate()
        .filter_map(|(index, parameter)| {
            parameter
                .value
                .as_ref()
                .and_then(|value| value.r#type.as_ref())
                .filter(|type_spec| {
                    type_spec.kind() == wire::type_proto::Kind::Staged && type_spec.name == "Param"
                })
                .map(|_| index)
        })
        .collect::<Vec<_>>();
    let parameters = validate_dispatch_parameters(entry, &parameter_indices)?;
    validate_launch_bindings(entry, &parameters, bindings)
}

pub(crate) fn project_family(
    package: &wire::QCorePackageProto,
    boolean_plans: &phase_lifting::VerifiedBooleanLiftPlans,
    dispatch_parameter_indices: &[usize],
    lowering: LoweringConfig,
) -> Result<StaticCircuitFamily, StaticCircuitError> {
    let entry = package
        .functions
        .iter()
        .find(|function| function.symbol == package.entrypoint)
        .ok_or_else(|| StaticCircuitError::Missing("entrypoint".to_owned()))?;
    let mut runtime_indices =
        unbound_control_parameter_indices(package, entry, dispatch_parameter_indices)?;
    let completed = explore_dispatch_tree(package, boolean_plans, lowering)?;
    let mut predicates = Vec::<CircuitDispatchPredicate>::new();
    let mut variants = Vec::with_capacity(completed.len());
    for (decision_values, path_predicates, circuit) in completed {
        let mut decisions = Vec::with_capacity(path_predicates.len());
        for (predicate, value) in path_predicates.into_iter().zip(decision_values) {
            let predicate_index = predicates
                .iter()
                .position(|known| known == &predicate)
                .unwrap_or_else(|| {
                    predicates.push(predicate);
                    predicates.len() - 1
                });
            decisions.push(CircuitDispatchDecision {
                predicate_index,
                value,
            });
        }
        variants.push(StaticCircuitVariant {
            circuit,
            decisions: decisions.into_boxed_slice(),
        });
    }
    let referenced_names = predicates
        .iter()
        .flat_map(|predicate| predicate.scalars.iter())
        .filter_map(|node| match node {
            CircuitScalarNode::Parameter { name, .. } => Some(name.as_str()),
            _ => None,
        })
        .collect::<std::collections::HashSet<_>>();
    let predicate_indices = entry
        .parameters
        .iter()
        .enumerate()
        .filter(|(_, parameter)| referenced_names.contains(parameter.name.as_str()))
        .map(|(index, _)| index)
        .filter(|index| !runtime_indices.contains(index))
        .collect::<Vec<_>>();
    runtime_indices.extend(predicate_indices);
    runtime_indices.sort_unstable();
    let dispatch_parameters = validate_dispatch_parameters(entry, &runtime_indices)?
        .into_iter()
        .filter(|(_, name)| referenced_names.contains(name.as_str()))
        .map(|(parameter_index, name)| {
            let uint_width = entry
                .parameters
                .get(parameter_index)
                .and_then(|parameter| parameter.value.as_ref())
                .and_then(|value| value.r#type.as_ref())
                .and_then(fixed_uint_width);
            CircuitDispatchParameter {
                parameter_index,
                name,
                uint_width,
            }
        })
        .collect::<Vec<_>>();
    Ok(StaticCircuitFamily {
        dispatch_parameters: dispatch_parameters.into_boxed_slice(),
        predicates: predicates.into_boxed_slice(),
        variants: variants.into_boxed_slice(),
    })
}

fn validate_dispatch_parameters(
    entry: &wire::FunctionProto,
    parameter_indices: &[usize],
) -> Result<Vec<(usize, String)>, StaticCircuitError> {
    parameter_indices
        .iter()
        .map(|parameter_index| {
            let parameter = entry.parameters.get(*parameter_index).ok_or_else(|| {
                StaticCircuitError::Missing(format!(
                    "dispatch parameter #{parameter_index} in {:?}",
                    entry.symbol
                ))
            })?;
            let parameter_type = parameter
                .value
                .as_ref()
                .and_then(|value| value.r#type.as_ref())
                .ok_or_else(|| {
                    StaticCircuitError::Missing(format!(
                        "dispatch parameter type for {:?}",
                        parameter.name
                    ))
                })?;
            if parameter_type.kind() != wire::type_proto::Kind::Staged
                || parameter_type.name != "Param"
                || (circuit_scalar_type(parameter_type).is_none()
                    && !is_complex_scalar_type(parameter_type)
                    && parameter_shape(parameter_type)?.is_none())
            {
                return Err(StaticCircuitError::Unsupported(format!(
                    "host dispatch parameter {:?} is not a supported Param value",
                    parameter.name
                )));
            }
            Ok((*parameter_index, parameter.name.clone()))
        })
        .collect()
}

pub(crate) fn unbound_control_parameter_indices(
    package: &wire::QCorePackageProto,
    entry: &wire::FunctionProto,
    parameter_indices: &[usize],
) -> Result<Vec<usize>, StaticCircuitError> {
    use wire::type_proto::Kind;

    let constants = package
        .const_bindings
        .iter()
        .map(|binding| binding.name.as_str())
        .collect::<std::collections::HashSet<_>>();
    let mut runtime = Vec::with_capacity(parameter_indices.len());
    for parameter_index in parameter_indices.iter().copied() {
        let parameter = entry.parameters.get(parameter_index).ok_or_else(|| {
            StaticCircuitError::Missing(format!(
                "dispatch parameter #{parameter_index} in {:?}",
                entry.symbol
            ))
        })?;
        let type_spec = parameter
            .value
            .as_ref()
            .and_then(|value| value.r#type.as_ref())
            .ok_or_else(|| {
                StaticCircuitError::Missing(format!(
                    "dispatch parameter type for {:?}",
                    parameter.name
                ))
            })?;
        if type_spec.kind() == Kind::Staged
            && type_spec.name == "Const"
            && constants.contains(parameter.name.as_str())
        {
            continue;
        }
        runtime.push(parameter_index);
    }
    Ok(runtime)
}

fn validate_launch_bindings(
    entry: &wire::FunctionProto,
    controlled_parameters: &[(usize, String)],
    bindings: &[CircuitLaunchBinding],
) -> Result<HashMap<LaunchScalarKey, StaticValue>, StaticCircuitError> {
    let supplied = bindings
        .iter()
        .map(|binding| (binding.name.as_str(), binding))
        .collect::<HashMap<_, _>>();
    if supplied.len() != bindings.len() {
        return Err(StaticCircuitError::Unsupported(
            "duplicate launch materialization binding".to_owned(),
        ));
    }
    let expected = controlled_parameters
        .iter()
        .map(|(_, name)| name.as_str())
        .collect::<std::collections::HashSet<_>>();
    if supplied
        .keys()
        .copied()
        .collect::<std::collections::HashSet<_>>()
        != expected
    {
        return Err(StaticCircuitError::Unsupported(
            "launch materialization bindings disagree with the parameter ABI".to_owned(),
        ));
    }
    let mut scalars = HashMap::new();
    for (parameter_index, name) in controlled_parameters {
        let parameter_type = entry
            .parameters
            .get(*parameter_index)
            .and_then(|parameter| parameter.value.as_ref())
            .and_then(|value| value.r#type.as_ref())
            .ok_or_else(|| {
                StaticCircuitError::Missing(format!("launch parameter type for {name:?}"))
            })?;
        let binding = supplied.get(name.as_str()).ok_or_else(|| {
            StaticCircuitError::Missing(format!("launch materialization binding for {name:?}"))
        })?;
        collect_launch_scalars(
            name,
            binding.value(),
            parameter_type,
            &mut Vec::new(),
            &mut scalars,
        )?;
    }
    Ok(scalars)
}

fn collect_launch_scalars(
    name: &str,
    value: &CircuitLaunchValue,
    type_spec: &wire::TypeProto,
    path: &mut Vec<CircuitParameterPath>,
    scalars: &mut HashMap<LaunchScalarKey, StaticValue>,
) -> Result<(), StaticCircuitError> {
    use wire::type_proto::Kind;

    let current = strip_staged_type(type_spec)
        .ok_or_else(|| StaticCircuitError::Missing("launch binding type".to_owned()))?;
    if let Some(expected) = circuit_scalar_type(current) {
        let scalar = launch_scalar(value)
            .filter(|value| circuit_scalar_type_for_value(value) == Some(expected))
            .ok_or_else(|| {
                StaticCircuitError::Unsupported(format!(
                    "launch binding {name:?} disagrees with its scalar ABI"
                ))
            })?;
        let key = LaunchScalarKey {
            name: name.to_owned(),
            path: path.clone().into_boxed_slice(),
        };
        scalars.insert(key, scalar);
        return Ok(());
    }
    if is_complex_scalar_type(current) {
        let scalar = launch_scalar(value)
            .filter(|value| matches!(value, StaticValue::ComplexBits(_, _)))
            .ok_or_else(|| {
                StaticCircuitError::Unsupported(format!(
                    "launch binding {name:?} disagrees with its Complex ABI"
                ))
            })?;
        let key = LaunchScalarKey {
            name: name.to_owned(),
            path: path.clone().into_boxed_slice(),
        };
        scalars.insert(key, scalar);
        return Ok(());
    }
    if current.kind() == Kind::Array && current.name == "UInt" {
        return collect_launch_uint(name, value, current, path, scalars);
    }
    match current.kind() {
        Kind::Array => collect_launch_array(name, value, current, path, scalars),
        Kind::Product => {
            let CircuitLaunchValue::Aggregate(values) = value else {
                return Err(launch_aggregate_error(name));
            };
            if values.len() != current.arguments.len() {
                return Err(launch_aggregate_error(name));
            }
            for (index, (component, component_type)) in
                values.iter().zip(&current.arguments).enumerate()
            {
                path.push(CircuitParameterPath::Index(
                    u64::try_from(index).map_err(|_| StaticCircuitError::Overflow)?,
                ));
                collect_launch_scalars(name, component, component_type, path, scalars)?;
                path.pop();
            }
            Ok(())
        }
        Kind::Record => {
            let CircuitLaunchValue::Record(fields) = value else {
                return Err(launch_aggregate_error(name));
            };
            if fields.len() != current.field_names.len()
                || current.field_names.len() != current.arguments.len()
            {
                return Err(launch_aggregate_error(name));
            }
            let supplied = fields
                .iter()
                .map(|(field, value)| (field.as_str(), value))
                .collect::<HashMap<_, _>>();
            if supplied.len() != fields.len() {
                return Err(launch_aggregate_error(name));
            }
            for (field, field_type) in current.field_names.iter().zip(&current.arguments) {
                let component = supplied
                    .get(field.as_str())
                    .ok_or_else(|| launch_aggregate_error(name))?;
                path.push(CircuitParameterPath::Field(field.clone()));
                collect_launch_scalars(name, component, field_type, path, scalars)?;
                path.pop();
            }
            Ok(())
        }
        _ => Err(StaticCircuitError::Unsupported(format!(
            "launch binding {name:?} has an unsupported control type"
        ))),
    }
}

fn collect_launch_uint(
    name: &str,
    value: &CircuitLaunchValue,
    type_spec: &wire::TypeProto,
    path: &mut Vec<CircuitParameterPath>,
    scalars: &mut HashMap<LaunchScalarKey, StaticValue>,
) -> Result<(), StaticCircuitError> {
    use wire::dimension_proto::Value;

    let [dimension] = type_spec.dimensions.as_slice() else {
        return Err(launch_aggregate_error(name));
    };
    let Some(Value::Fixed(width)) = dimension.value else {
        return Err(StaticCircuitError::Unsupported(
            "launch materialization requires a fixed UInt width".to_owned(),
        ));
    };
    let width = usize::try_from(width).map_err(|_| StaticCircuitError::Overflow)?;
    let integer = match value {
        CircuitLaunchValue::Int(value) => BigInt::from(*value),
        CircuitLaunchValue::BigInt(value) => BigInt::from_signed_bytes_be(value),
        _ => return Err(launch_aggregate_error(name)),
    };
    if integer < BigInt::ZERO || integer.bits() > u64::try_from(width).unwrap_or(u64::MAX) {
        return Err(launch_aggregate_error(name));
    }
    for index in 0..width {
        path.push(CircuitParameterPath::Index(
            u64::try_from(index).map_err(|_| StaticCircuitError::Overflow)?,
        ));
        let key = LaunchScalarKey {
            name: name.to_owned(),
            path: path.clone().into_boxed_slice(),
        };
        let shift = width - index - 1;
        let bit = ((&integer >> shift) & BigInt::from(1_u8)) == BigInt::from(1_u8);
        scalars.insert(key, StaticValue::Scalar(StaticScalar::Bool(bit)));
        path.pop();
    }
    Ok(())
}

fn collect_launch_array(
    name: &str,
    value: &CircuitLaunchValue,
    type_spec: &wire::TypeProto,
    path: &mut Vec<CircuitParameterPath>,
    scalars: &mut HashMap<LaunchScalarKey, StaticValue>,
) -> Result<(), StaticCircuitError> {
    use wire::dimension_proto::Value;

    fn descend(
        name: &str,
        value: &CircuitLaunchValue,
        dimensions: &[wire::DimensionProto],
        element_type: &wire::TypeProto,
        path: &mut Vec<CircuitParameterPath>,
        scalars: &mut HashMap<LaunchScalarKey, StaticValue>,
    ) -> Result<(), StaticCircuitError> {
        let (dimension, remaining) = dimensions
            .split_first()
            .ok_or_else(|| StaticCircuitError::Missing("launch array dimension".to_owned()))?;
        let extent = match dimension.value.as_ref() {
            Some(Value::Fixed(value)) => usize::try_from(*value).ok(),
            Some(Value::Symbol(_)) | None => None,
        }
        .ok_or_else(|| {
            StaticCircuitError::Unsupported(
                "launch materialization requires fixed aggregate shapes".to_owned(),
            )
        })?;
        let CircuitLaunchValue::Aggregate(values) = value else {
            return Err(launch_aggregate_error(name));
        };
        if values.len() != extent {
            return Err(launch_aggregate_error(name));
        }
        for (index, component) in values.iter().enumerate() {
            path.push(CircuitParameterPath::Index(
                u64::try_from(index).map_err(|_| StaticCircuitError::Overflow)?,
            ));
            if remaining.is_empty() {
                collect_launch_scalars(name, component, element_type, path, scalars)?;
            } else {
                descend(name, component, remaining, element_type, path, scalars)?;
            }
            path.pop();
        }
        Ok(())
    }
    let element_type = type_spec
        .arguments
        .first()
        .ok_or_else(|| StaticCircuitError::Missing("launch array element type".to_owned()))?;
    descend(
        name,
        value,
        &type_spec.dimensions,
        element_type,
        path,
        scalars,
    )
}

fn launch_scalar(value: &CircuitLaunchValue) -> Option<StaticValue> {
    match value {
        CircuitLaunchValue::Bool(value) => Some(StaticValue::Scalar(StaticScalar::Bool(*value))),
        CircuitLaunchValue::Int(value) => Some(StaticValue::Scalar(StaticScalar::Int(*value))),
        CircuitLaunchValue::BigInt(value) => Some(StaticValue::BigInt(value.clone().into())),
        CircuitLaunchValue::RealBits(value) => {
            Some(StaticValue::Scalar(StaticScalar::RealBits(*value)))
        }
        CircuitLaunchValue::ComplexBits(real, imaginary) => {
            Some(StaticValue::ComplexBits(*real, *imaginary))
        }
        CircuitLaunchValue::Aggregate(_) | CircuitLaunchValue::Record(_) => None,
    }
}

const fn circuit_scalar_type_for_value(value: &StaticValue) -> Option<CircuitScalarType> {
    match value {
        StaticValue::Scalar(StaticScalar::Bool(_)) => Some(CircuitScalarType::Bool),
        StaticValue::Scalar(StaticScalar::Int(_)) | StaticValue::BigInt(_) => {
            Some(CircuitScalarType::Int)
        }
        StaticValue::Scalar(StaticScalar::RealBits(_)) => Some(CircuitScalarType::Real),
        StaticValue::Unit
        | StaticValue::ComplexBits(_, _)
        | StaticValue::String(_)
        | StaticValue::Bytes(_)
        | StaticValue::Aggregate(_)
        | StaticValue::Record(_)
        | StaticValue::Mapping(_) => None,
    }
}

fn launch_aggregate_error(name: &str) -> StaticCircuitError {
    StaticCircuitError::Unsupported(format!(
        "launch binding {name:?} disagrees with its aggregate ABI"
    ))
}

fn explore_dispatch_tree(
    package: &wire::QCorePackageProto,
    boolean_plans: &phase_lifting::VerifiedBooleanLiftPlans,
    lowering: LoweringConfig,
) -> Result<Vec<CompletedFamilyProjection>, StaticCircuitError> {
    let mut pending = std::collections::VecDeque::from([Vec::<bool>::new()]);
    let mut completed = Vec::new();
    let mut instruction_count = 0_usize;
    while let Some(decisions) = pending.pop_front() {
        match project_with_host_decisions(
            package,
            boolean_plans,
            lowering,
            &decisions,
            &HashMap::new(),
        ) {
            Ok((circuit, predicates, _)) => {
                instruction_count = instruction_count
                    .checked_add(circuit.operation_count())
                    .filter(|count| *count <= MAX_STATIC_CIRCUIT_INSTRUCTIONS)
                    .ok_or_else(|| {
                        StaticCircuitError::Unsupported(format!(
                            "circuit family exceeds {MAX_STATIC_CIRCUIT_INSTRUCTIONS} total instructions"
                        ))
                    })?;
                completed.push((decisions, predicates, circuit));
            }
            Err(StaticCircuitError::HostDispatch(_)) => {
                if pending.len() + completed.len() + 2 > MAX_STATIC_CIRCUIT_VARIANTS {
                    return Err(StaticCircuitError::Unsupported(format!(
                        "more than {MAX_STATIC_CIRCUIT_VARIANTS} host-dispatch variants"
                    )));
                }
                let mut when_false = decisions.clone();
                when_false.push(false);
                pending.push_back(when_false);
                let mut when_true = decisions;
                when_true.push(true);
                pending.push_back(when_true);
            }
            Err(error) => return Err(error),
        }
    }
    Ok(completed)
}

pub(crate) fn instruction_count(instructions: &[StaticInstruction]) -> usize {
    let mut count = 0_usize;
    let mut pending = vec![instructions];
    while let Some(instructions) = pending.pop() {
        count += instructions.len();
        pending.extend(
            instructions
                .iter()
                .flat_map(|instruction| instruction.regions.iter())
                .map(|region| region.instructions.as_ref()),
        );
    }
    count
}

#[allow(clippy::too_many_lines)]
fn project_with_host_decisions(
    package: &wire::QCorePackageProto,
    boolean_plans: &phase_lifting::VerifiedBooleanLiftPlans,
    lowering: LoweringConfig,
    host_decisions: &[bool],
    launch_scalars: &HashMap<LaunchScalarKey, StaticValue>,
) -> Result<(StaticCircuit, Vec<CircuitDispatchPredicate>, u128), StaticCircuitError> {
    let mut lowerer = StaticLowerer {
        entry_body: package
            .functions
            .iter()
            .find(|f| f.symbol == package.entrypoint)
            .and_then(|f| f.body.as_ref()),
        cut_operations: super::shot_execution::closed_entry_cuts(package),
        closed_cuts: Vec::new(),
        scalar_checks: Vec::new(),
        measurement_guards: Vec::new(),
        realizations: BTreeSet::new(),
        boolean_plans,
        lowering,
        functions: package
            .functions
            .iter()
            .map(|function| (function.symbol.as_str(), function))
            .collect(),
        next_wire: 0,
        next_measurement: 0,
        next_call_value: 0,
        instructions: Vec::new(),
        scalar_ids: HashMap::new(),
        scalars: Vec::new(),
        controls: Vec::new(),
        next_loop_index: 0,
        loop_projection_work: 0,
        call_cache: HashMap::new(),
        arithmetic_definitions: HashMap::new(),
        modular_definitions: HashMap::new(),
        coherent_partitions: HashMap::new(),
        schoolbook_workspace: Vec::new(),
        instruction_count: 0,
        host_decisions,
        dispatch_predicates: Vec::new(),
        launch_scalars,
        known_scalars: HashMap::new(),
        result_types: Vec::new(),
        expanded_loops: Vec::new(),
        operation_visits: 0,
        aggregate_projection_work: 0,
    };
    let entry = lowerer
        .functions
        .get(package.entrypoint.as_str())
        .copied()
        .ok_or_else(|| StaticCircuitError::Missing("entrypoint".to_owned()))?;
    let bindings = package
        .const_bindings
        .iter()
        .map(|binding| (binding.name.as_str(), binding.value.as_ref()))
        .collect::<HashMap<_, _>>();
    let mut arguments = Vec::with_capacity(entry.parameters.len());
    for parameter in &entry.parameters {
        if parameter
            .value
            .as_ref()
            .is_some_and(|value| value.has_region)
        {
            return Err(StaticCircuitError::Unsupported(
                "external live quantum entry parameter".to_owned(),
            ));
        }
        let parameter_type = parameter
            .value
            .as_ref()
            .and_then(|value| value.r#type.as_ref())
            .ok_or_else(|| {
                StaticCircuitError::Missing(format!(
                    "entry parameter type for {:?}",
                    parameter.name
                ))
            })?;
        let value = if let Some(literal) = bindings.get(parameter.name.as_str()).copied().flatten()
        {
            lowerer.literal_value(literal, parameter_type)?
        } else {
            lowerer.parameter_value(parameter.name.clone(), Box::new([]), parameter_type)?
        };
        arguments.push(value);
    }
    let flow = lowerer.emit_function(entry, &arguments)?;
    if flow.kind != "return" {
        return Err(StaticCircuitError::Missing(format!(
            "return terminator in {:?}",
            entry.symbol
        )));
    }
    let mut result_measurements = Vec::new();
    let mut result_value = if let [value] = flow.values.as_slice() {
        if !collect_measurements(value, &lowerer.scalars, &mut result_measurements) {
            result_measurements.clear();
        }
        entry
            .result_type
            .as_ref()
            .and_then(|result_type| retain_result_value(value, result_type))
    } else if !flow.values.is_empty() {
        return Err(StaticCircuitError::Unsupported(
            "multiple entrypoint return values".to_owned(),
        ));
    } else {
        None
    };
    lowerer.retain_boolean_stores()?;
    let mut instructions = lowerer.instructions;
    let operation_count = instruction_count(&instructions);
    let scalars = compact_circuit_scalars(
        &mut instructions,
        result_value.as_mut(),
        &lowerer.scalars,
        &mut lowerer.scalar_checks,
    )?;
    let circuit = StaticCircuit {
        optimization_stages: Vec::new(),
        closed_cuts: lowerer.closed_cuts.into_boxed_slice(),
        scalar_checks: lowerer.scalar_checks.into_boxed_slice(),
        realizations: lowerer.realizations.into_iter().collect(),
        width: lowerer.next_wire,
        measurement_count: lowerer.next_measurement,
        operation_count,
        instructions: instructions.into_boxed_slice(),
        scalars,
        result_measurements: result_measurements.into_boxed_slice(),
        result_value,
    };
    Ok((
        circuit,
        lowerer.dispatch_predicates,
        lowerer.operation_visits,
    ))
}

impl StaticLowerer<'_> {
    fn literal_value(
        &mut self,
        literal: &wire::LiteralProto,
        type_spec: &wire::TypeProto,
    ) -> Result<CircuitValue, StaticCircuitError> {
        let value = static_values::typed_value_literal(literal, type_spec).ok_or_else(|| {
            StaticCircuitError::Unsupported("circuit literal/type relation".to_owned())
        })?;
        self.static_value_to_circuit(&value)
    }

    fn static_value_to_circuit(
        &mut self,
        value: &StaticValue,
    ) -> Result<CircuitValue, StaticCircuitError> {
        match value {
            StaticValue::Unit => Ok(CircuitValue::Unit),
            StaticValue::Scalar(value) => Ok(CircuitValue::Scalar(
                self.intern_literal(&static_values::to_literal(*value))?,
            )),
            StaticValue::Aggregate(values) => Ok(CircuitValue::Aggregate(
                values
                    .iter()
                    .map(|value| {
                        value
                            .as_ref()
                            .ok_or_else(|| {
                                StaticCircuitError::Missing(
                                    "complete circuit aggregate literal".to_owned(),
                                )
                            })
                            .and_then(|value| self.static_value_to_circuit(value))
                    })
                    .collect::<Result<Vec<_>, _>>()?
                    .into_boxed_slice(),
            )),
            StaticValue::Record(fields) => Ok(CircuitValue::Aggregate(
                fields
                    .iter()
                    .map(|(_, value)| {
                        value
                            .as_ref()
                            .ok_or_else(|| {
                                StaticCircuitError::Missing(
                                    "complete circuit record literal".to_owned(),
                                )
                            })
                            .and_then(|value| self.static_value_to_circuit(value))
                    })
                    .collect::<Result<Vec<_>, _>>()?
                    .into_boxed_slice(),
            )),
            StaticValue::Mapping(_) => {
                let literal = static_values::value_to_literal(value).ok_or_else(|| {
                    StaticCircuitError::Missing("complete Hamiltonian literal".to_owned())
                })?;
                Ok(CircuitValue::Hamiltonian(
                    literal.encode_to_vec().into_boxed_slice(),
                ))
            }
            StaticValue::BigInt(_)
            | StaticValue::ComplexBits(_, _)
            | StaticValue::String(_)
            | StaticValue::Bytes(_) => Ok(CircuitValue::Portable(value.clone())),
        }
    }

    fn parameter_value(
        &mut self,
        name: String,
        path: Box<[CircuitParameterPath]>,
        current_type: &wire::TypeProto,
    ) -> Result<CircuitValue, StaticCircuitError> {
        let launch_value = self
            .launch_scalars
            .get(&LaunchScalarKey {
                name: name.clone(),
                path: path.clone(),
            })
            .cloned();
        if let Some(scalar_type) = circuit_scalar_type(current_type) {
            let identifier = self.intern_scalar(CircuitScalarNode::Parameter {
                name,
                path,
                scalar_type,
            })?;
            if let Some(value) = launch_value {
                if circuit_scalar_type_for_value(&value) != Some(scalar_type) {
                    return Err(StaticCircuitError::Unsupported(
                        "launch binding scalar type disagrees with the entrypoint ABI".to_owned(),
                    ));
                }
                self.known_scalars.insert(identifier, value);
            }
            return Ok(CircuitValue::Scalar(identifier));
        }
        if is_complex_scalar_type(current_type) {
            let value = launch_value.ok_or_else(|| {
                StaticCircuitError::Unsupported(
                    "Complex circuit parameters require exact launch materialization".to_owned(),
                )
            })?;
            if !matches!(value, StaticValue::ComplexBits(_, _)) {
                return Err(StaticCircuitError::Unsupported(
                    "Complex launch value disagrees with the entrypoint ABI".to_owned(),
                ));
            }
            return self.static_value_to_circuit(&value);
        }
        if let Some(shape) = parameter_shape(current_type)? {
            return Ok(CircuitValue::Parameter { name, path, shape });
        }
        Err(StaticCircuitError::Unsupported(format!(
            "circuit parameter with type {:?}",
            display_type(current_type)
        )))
    }

    fn intern_scalar(
        &mut self,
        node: CircuitScalarNode,
    ) -> Result<CircuitScalarId, StaticCircuitError> {
        if let Some(identifier) = self.scalar_ids.get(&node) {
            return Ok(*identifier);
        }
        let index = u32::try_from(self.scalars.len()).map_err(|_| StaticCircuitError::Overflow)?;
        let identifier = CircuitScalarId(index);
        self.scalars.push(node.clone());
        self.scalar_ids.insert(node, identifier);
        Ok(identifier)
    }

    fn intern_literal(
        &mut self,
        literal: &wire::LiteralProto,
    ) -> Result<CircuitScalarId, StaticCircuitError> {
        use wire::literal_proto::Value;

        let node = match literal.value.as_ref() {
            Some(Value::BoolValue(value)) => CircuitScalarNode::Bool(*value),
            Some(Value::IntValue(value)) => CircuitScalarNode::Int(*value),
            Some(Value::RealValue(value)) => CircuitScalarNode::RealBits(value.to_bits()),
            _ => {
                return Err(StaticCircuitError::Unsupported(
                    "non-scalar circuit literal".to_owned(),
                ));
            }
        };
        let value = match &node {
            CircuitScalarNode::Bool(value) => StaticScalar::Bool(*value),
            CircuitScalarNode::Int(value) => StaticScalar::Int(*value),
            CircuitScalarNode::RealBits(value) => StaticScalar::RealBits(*value),
            _ => unreachable!("literal nodes are closed scalar values"),
        };
        let identifier = self.intern_scalar(node)?;
        self.known_scalars
            .insert(identifier, StaticValue::Scalar(value));
        Ok(identifier)
    }

    fn scalar_node(&self, identifier: CircuitScalarId) -> Option<&CircuitScalarNode> {
        self.scalars.get(usize::try_from(identifier.0).ok()?)
    }

    fn static_value(&self, value: &CircuitValue) -> Option<StaticValue> {
        let identifier = match value {
            CircuitValue::Portable(value) => return Some(value.clone()),
            CircuitValue::Hamiltonian(payload) => {
                let literal = wire::LiteralProto::decode(payload.as_ref()).ok()?;
                return static_values::value_literal(&literal);
            }
            CircuitValue::Aggregate(values) => {
                return Some(StaticValue::Aggregate(
                    values
                        .iter()
                        .map(|value| self.static_value(value).map(Some))
                        .collect::<Option<Vec<_>>>()?
                        .into(),
                ));
            }
            CircuitValue::Scalar(identifier) => identifier,
            _ => return None,
        };
        if let Some(value) = self.known_scalars.get(identifier) {
            return Some(value.clone());
        }
        match self.scalar_node(*identifier)? {
            CircuitScalarNode::Bool(value) => Some(StaticValue::Scalar(StaticScalar::Bool(*value))),
            CircuitScalarNode::Int(value) => Some(StaticValue::Scalar(StaticScalar::Int(*value))),
            CircuitScalarNode::RealBits(value) => {
                Some(StaticValue::Scalar(StaticScalar::RealBits(*value)))
            }
            CircuitScalarNode::LoopIndex(_)
            | CircuitScalarNode::Measurement(_)
            | CircuitScalarNode::CallValue { .. }
            | CircuitScalarNode::Parameter { .. }
            | CircuitScalarNode::Apply { .. } => None,
        }
    }

    fn static_int(&self, value: &CircuitValue) -> Option<i64> {
        match self.static_value(value)? {
            StaticValue::Scalar(StaticScalar::Int(value)) => Some(value),
            StaticValue::BigInt(value) => BigInt::from_signed_bytes_be(&value).to_i64(),
            _ => None,
        }
    }

    fn literal_bits(&self, value: &CircuitValue) -> Option<Vec<bool>> {
        match value {
            CircuitValue::Scalar(_) => match self.static_value(value)? {
                StaticValue::Scalar(StaticScalar::Bool(value)) => Some(vec![value]),
                StaticValue::Scalar(StaticScalar::Int(value @ (0 | 1))) => Some(vec![value == 1]),
                _ => None,
            },
            CircuitValue::Aggregate(values) => values
                .iter()
                .map(|value| match self.static_value(value)? {
                    StaticValue::Scalar(StaticScalar::Bool(value)) => Some(value),
                    StaticValue::Scalar(StaticScalar::Int(value @ (0 | 1))) => Some(value == 1),
                    _ => None,
                })
                .collect(),
            CircuitValue::Parameter {
                name,
                path,
                shape: ParameterShape::Array(width),
            } => (0..*width)
                .map(|index| {
                    let mut component_path = Vec::with_capacity(path.len() + 1);
                    component_path.extend_from_slice(path);
                    component_path.push(CircuitParameterPath::Index(index));
                    match self.launch_scalars.get(&LaunchScalarKey {
                        name: name.clone(),
                        path: component_path.into_boxed_slice(),
                    })? {
                        StaticValue::Scalar(StaticScalar::Bool(value)) => Some(*value),
                        StaticValue::Scalar(StaticScalar::Int(value @ (0 | 1))) => {
                            Some(*value == 1)
                        }
                        _ => None,
                    }
                })
                .collect(),
            _ => None,
        }
    }

    fn host_dispatch_decision(
        &mut self,
        condition: CircuitScalarId,
    ) -> Result<bool, StaticCircuitError> {
        let predicate = compact_dispatch_predicate(condition, &self.scalars)?;
        let predicate_position = self
            .dispatch_predicates
            .iter()
            .position(|known| known == &predicate);
        let decision = if let Some(position) = predicate_position {
            self.host_decisions.get(position).copied()
        } else {
            let position = self.dispatch_predicates.len();
            let decision = self.host_decisions.get(position).copied();
            self.dispatch_predicates.push(predicate.clone());
            decision
        };
        decision.ok_or(StaticCircuitError::HostDispatch(predicate))
    }

    fn prepare_bits(&mut self, value: &CircuitValue) -> Result<Vec<bool>, StaticCircuitError> {
        if let Some(bits) = self.literal_bits(value) {
            return Ok(bits);
        }
        let CircuitValue::Parameter {
            name,
            path,
            shape: ParameterShape::Array(width),
        } = value
        else {
            return Err(StaticCircuitError::Unsupported(
                "non-static quantum.prepare data".to_owned(),
            ));
        };
        let name = name.clone();
        let path = path.clone();
        (0..*width)
            .map(|index| {
                let mut component_path = Vec::with_capacity(path.len() + 1);
                component_path.extend_from_slice(&path);
                component_path.push(CircuitParameterPath::Index(index));
                let condition = self.intern_scalar(CircuitScalarNode::Parameter {
                    name: name.clone(),
                    path: component_path.into_boxed_slice(),
                    scalar_type: CircuitScalarType::Bool,
                })?;
                self.host_dispatch_decision(condition)
            })
            .collect()
    }

    fn emit_function(
        &mut self,
        function: &wire::FunctionProto,
        arguments: &[CircuitValue],
    ) -> Result<Flow, StaticCircuitError> {
        if self.result_types.len() >= MAX_CIRCUIT_CALL_DEPTH {
            return Err(StaticCircuitError::Unsupported(format!(
                "call expansion exceeds the {MAX_CIRCUIT_CALL_DEPTH}-frame depth budget"
            )));
        }
        if function.external {
            return Err(StaticCircuitError::Unsupported(format!(
                "external callable {:?}",
                function.symbol
            )));
        }
        if arguments.len() != function.parameters.len() {
            return Err(StaticCircuitError::Missing(format!(
                "argument ABI for {:?}",
                function.symbol
            )));
        }
        let environment = function
            .parameters
            .iter()
            .zip(arguments)
            .map(|(parameter, value)| {
                parameter
                    .value
                    .as_ref()
                    .map(|definition| (definition.id, value.clone()))
                    .ok_or_else(|| {
                        StaticCircuitError::Missing(format!(
                            "parameter value in {:?}",
                            function.symbol
                        ))
                    })
            })
            .collect::<Result<HashMap<_, _>, _>>()?;
        let body = function.body.as_ref().ok_or_else(|| {
            StaticCircuitError::Missing(format!("body for {:?}", function.symbol))
        })?;
        let result_type = function.result_type.clone().ok_or_else(|| {
            StaticCircuitError::Missing(format!("result type for {:?}", function.symbol))
        })?;
        self.result_types.push(result_type);
        let outer_loops = std::mem::take(&mut self.expanded_loops);
        let result = self.emit_block(body, &environment, &[]);
        self.expanded_loops = outer_loops;
        self.result_types.pop();
        result
    }

    fn emit_block(
        &mut self,
        block: &wire::BlockProto,
        outer: &HashMap<u64, CircuitValue>,
        arguments: &[CircuitValue],
    ) -> Result<Flow, StaticCircuitError> {
        if arguments.len() != block.arguments.len() {
            return Err(StaticCircuitError::Missing(
                "structured-region argument ABI".to_owned(),
            ));
        }
        let mut environment = outer.clone();
        for (definition, value) in block.arguments.iter().zip(arguments) {
            environment.insert(definition.id, value.clone());
        }
        let entry_block = self.result_types.len() == 1
            && self
                .entry_body
                .is_some_and(|entry| std::ptr::eq(entry, block));
        for (operation_index, operation) in block.operations.iter().enumerate() {
            self.operation_visits = self.operation_visits.saturating_add(1);
            if self.operation_visits > MAX_CIRCUIT_OPERATION_VISITS {
                return Err(StaticCircuitError::Unsupported(format!(
                    "call expansion exceeds the {MAX_CIRCUIT_OPERATION_VISITS}-operation work budget"
                )));
            }
            let guard = self.expanded_loops.last().map(|frame| frame.continuing);
            let start = self.instructions.len();
            let outcome = self.emit_operation(operation, &environment)?;
            if let Some(condition) = guard {
                self.guard_instructions(start, condition)?;
            }
            let values = match outcome {
                OperationFlow::Values(values) => values,
                OperationFlow::Exit(flow) => return Ok(flow),
            };
            if values.len() != operation.results.len() {
                return Err(StaticCircuitError::Missing(format!(
                    "result ABI for {:?}",
                    operation.opcode
                )));
            }
            for (result, value) in operation.results.iter().zip(values) {
                environment.insert(result.id, value);
            }
            if entry_block && self.cut_operations.contains(&operation_index) {
                let cut = ClosedQuantumFrontier {
                    instruction_index: self.instructions.len(),
                    measurement_prefix: self.next_measurement,
                };
                if self.closed_cuts.last() != Some(&cut) {
                    self.closed_cuts.push(cut);
                }
            }
        }
        let terminator = block
            .terminator
            .as_ref()
            .ok_or_else(|| StaticCircuitError::Missing("block terminator".to_owned()))?;
        let values = terminator
            .operands
            .iter()
            .map(|operand| {
                environment.get(operand).cloned().ok_or_else(|| {
                    StaticCircuitError::Missing(format!("terminator operand %{operand}"))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let flow = Flow {
            kind: terminator.kind.clone(),
            values,
        };
        self.record_expanded_loop_exit(&flow)?;
        Ok(flow)
    }

    fn guard_instructions(
        &mut self,
        start: usize,
        condition: CircuitScalarId,
    ) -> Result<(), StaticCircuitError> {
        if self.instructions.len() == start
            || matches!(
                self.scalar_node(condition),
                Some(CircuitScalarNode::Bool(true))
            )
        {
            return Ok(());
        }
        let instructions = self.instructions.split_off(start);
        if self.instruction_count >= MAX_STATIC_CIRCUIT_INSTRUCTIONS {
            return Err(StaticCircuitError::Unsupported(format!(
                "more than {MAX_STATIC_CIRCUIT_INSTRUCTIONS} circuit instructions"
            )));
        }
        self.instruction_count += 1;
        self.instructions.push(StaticInstruction {
            opcode: "if".to_owned(),
            qubits: Box::new([]),
            parameters: Box::new([]),
            controls: Box::new([]),
            condition: Some(condition),
            regions: vec![
                StaticBlock {
                    instructions: instructions.into_boxed_slice(),
                    return_value: None,
                },
                StaticBlock {
                    instructions: Box::new([]),
                    return_value: None,
                },
            ]
            .into_boxed_slice(),
        });
        Ok(())
    }

    fn select_scalar(
        &mut self,
        condition: CircuitScalarId,
        when_true: CircuitScalarId,
        when_false: CircuitScalarId,
    ) -> Result<CircuitScalarId, StaticCircuitError> {
        let value = self.merge_dynamic_if_value(
            condition,
            &CircuitValue::Scalar(when_true),
            &CircuitValue::Scalar(when_false),
        )?;
        Self::scalar_value_identifier(&value, "loop predicate")
    }

    fn record_expanded_loop_exit(&mut self, flow: &Flow) -> Result<(), StaticCircuitError> {
        if !matches!(flow.kind.as_str(), "break" | "continue") {
            return Ok(());
        }
        let Some(mut frame) = self.expanded_loops.pop() else {
            return Err(StaticCircuitError::Missing(
                "expanded loop exit scope".to_owned(),
            ));
        };
        let inactive = self.intern_scalar(CircuitScalarNode::Bool(false))?;
        frame.exit_values =
            self.merge_dynamic_if_values(frame.continuing, &flow.values, &frame.exit_values)?;
        if flow.kind == "break" {
            frame.running = self.select_scalar(frame.continuing, inactive, frame.running)?;
        }
        frame.continuing = inactive;
        self.expanded_loops.push(frame);
        Ok(())
    }

    fn retain_coherent_region(
        &mut self,
        start: usize,
        label: &str,
        controls: Vec<u64>,
    ) -> Result<(), StaticCircuitError> {
        let instructions = self.instructions.split_off(start);
        if !coherent_instructions(&instructions) || contains_structured_loop(&instructions) {
            return Err(StaticCircuitError::Unsupported(
                "nonunitary gate definition".to_owned(),
            ));
        }
        let mut qubits = BTreeSet::new();
        collect_instruction_wires(&instructions, &mut qubits);
        if qubits.iter().any(|wire| controls.contains(wire)) {
            return Err(StaticCircuitError::InvalidSelector(
                "aliased coherent region controls".to_owned(),
            ));
        }
        let qubits = qubits.into_iter().collect::<Vec<_>>();
        let map = qubits
            .iter()
            .copied()
            .enumerate()
            .map(|(local, wire)| (wire, local as u64))
            .collect();
        let mut instructions = instructions;
        deferred::remap_instruction_wires(&mut instructions, &map)?;
        let scalars = compact_circuit_scalars(&mut instructions, None, &self.scalars, &mut [])?;
        let mut parameters = vec![CircuitParameter::Definition(CircuitDefinition::body(
            qubits.len() as u64,
            instructions,
            scalars.clone(),
        ))];
        // Retain launch/index dependencies in the enclosing scalar arena too.
        for node in &scalars {
            if matches!(
                node,
                CircuitScalarNode::Parameter { .. } | CircuitScalarNode::LoopIndex(_)
            ) {
                parameters.push(CircuitParameter::Scalar(self.intern_scalar(node.clone())?));
            }
        }
        self.push_instruction(&format!("box:{label}"), qubits, parameters)?;
        let instruction = self.instructions.last_mut().expect("region was inserted");
        instruction.controls = controls.into_boxed_slice();
        Ok(())
    }

    fn emit_retained_callable(
        &mut self,
        callee_name: &str,
        callee: &wire::FunctionProto,
        arguments: &[CircuitValue],
    ) -> Result<Flow, StaticCircuitError> {
        if !self
            .lowering
            .capabilities
            .contains(RealizationCapabilities::COHERENT_REGIONS)
        {
            return self.emit_function(callee, arguments);
        }
        let cache_key = self.call_cache_key(callee_name, arguments);
        if let Some((key, wires)) = &cache_key
            && let Some(entry) = self.call_cache.get(key).cloned()
        {
            return self.replay_cached_call(&entry, wires);
        }
        let start_wire = self.next_wire;
        let start_measurement = self.next_measurement;
        let start_workspace = self.schoolbook_workspace.len();
        let controls = std::mem::take(&mut self.controls);
        let start = self.instructions.len();
        let result = self.emit_function(callee, arguments);
        let flow = result?;
        if coherent_instructions(&self.instructions[start..]) {
            if contains_structured_loop(&self.instructions[start..]) {
                distribute_controls(&mut self.instructions[start..], &controls)?;
            } else if !self.instructions[start..].is_empty() {
                let label = named_call_label(callee_name).unwrap_or_else(|| callee_name.to_owned());
                self.retain_coherent_region(start, &label, controls.clone())?;
            }
        } else if !controls.is_empty() {
            return Err(StaticCircuitError::Unsupported(
                "nonunitary controlled callable".to_owned(),
            ));
        }
        self.controls = controls;
        if self.next_wire.checked_sub(start_wire)
            == u64::try_from(self.schoolbook_workspace.len() - start_workspace).ok()
            && self.next_measurement == start_measurement
            && let Some((key, wires)) = cache_key
        {
            self.cache_call(key, &wires, start, &flow);
        }
        Ok(flow)
    }

    fn emit_operation(
        &mut self,
        operation: &wire::OperationProto,
        environment: &HashMap<u64, CircuitValue>,
    ) -> Result<OperationFlow, StaticCircuitError> {
        let semantic_opcode = if operation.opcode == "qcore.adjoint" {
            string_attribute(operation, "original").unwrap_or(&operation.opcode)
        } else {
            &operation.opcode
        };
        let semantic = matches!(
            semantic_opcode,
            "qcore.phase_by"
                | "qcore.xor_by"
                | "qcore.permute_by"
                | "quantum.add_constant"
                | "quantum.subtract_constant"
                | "quantum.mod_add"
                | "quantum.mod_sub"
                | "quantum.mod_add_constant"
                | "quantum.mod_sub_constant"
                | "quantum.mul_add"
                | "quantum.mul_sub"
                | "quantum.mod_mul_add"
                | "quantum.mod_mul_sub"
                | "quantum.mod_mul"
                | "quantum.mod_div"
                | "quantum.mod_pow"
                | "quantum.mod_pow_inverse"
                | "quantum.mod_multi_pow"
                | "quantum.mod_multi_pow_inverse"
                | "quantum.less_than_xor"
        );
        if !semantic
            || !self
                .lowering
                .capabilities
                .contains(RealizationCapabilities::COHERENT_REGIONS)
        {
            return self.emit_operation_unwrapped(operation, environment);
        }
        let controls = std::mem::take(&mut self.controls);
        let start = self.instructions.len();
        let result = self.emit_operation_unwrapped(operation, environment)?;
        if self.instructions.len() == start + 1
            && matches!(
                self.instructions[start].opcode.as_str(),
                "deferred"
                    | "phase_oracle"
                    | "xor_oracle"
                    | "linear_function"
                    | "linear_function_inverse"
            )
        {
            self.instructions[start].controls = controls.clone().into_boxed_slice();
        } else {
            self.retain_coherent_region(start, &operation.opcode, controls.clone())?;
        }
        self.controls = controls;
        Ok(result)
    }

    fn emit_operation_unwrapped(
        &mut self,
        operation: &wire::OperationProto,
        environment: &HashMap<u64, CircuitValue>,
    ) -> Result<OperationFlow, StaticCircuitError> {
        let operands = operation
            .operands
            .iter()
            .map(|operand| {
                environment
                    .get(operand)
                    .cloned()
                    .ok_or_else(|| StaticCircuitError::Missing(format!("operand %{operand}")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        match operation.opcode.as_str() {
            "qcore.if" => return self.emit_if(operation, &operands),
            "qcore.for" => return self.emit_for(operation, &operands),
            "qcore.while" => return self.emit_while(operation, &operands),
            _ => {}
        }
        let values = match operation.opcode.as_str() {
            "qcore.constant" => {
                let literal = attribute(operation, "value").ok_or_else(|| {
                    StaticCircuitError::Missing("qcore.constant value".to_owned())
                })?;
                let type_spec = operation
                    .results
                    .first()
                    .and_then(|result| result.r#type.as_ref())
                    .ok_or_else(|| {
                        StaticCircuitError::Missing("qcore.constant result type".to_owned())
                    })?;
                let value = self.literal_value(literal, type_spec)?;
                Ok(vec![value])
            }
            "qcore.stage_lift" => Ok(operands),
            "qcore.reshape" => self.emit_reshape(operation, &operands),
            "tuple" | "array" | "qcore.tuple" => {
                Ok(vec![CircuitValue::Aggregate(operands.into_boxed_slice())])
            }
            "qcore.unpack" => match operands.as_slice() {
                [CircuitValue::Aggregate(values)] => Ok(values.to_vec()),
                _ => Err(StaticCircuitError::Missing(
                    "aggregate qcore.unpack operand".to_owned(),
                )),
            },
            "qcore.call" => self.emit_call(operation, &operands),
            "qcore.phase_by" => self.emit_phase_by(operation, &operands),
            "qcore.xor_by" => self.emit_xor_by(operation, &operands),
            "qcore.permute_by" => self.emit_permute_by(operation, &operands, false),
            "qcore.adjoint_call" => self.emit_adjoint_call(operation, &operands),
            "qcore.adjoint" => self.emit_adjoint(operation, &operands),
            "qcore.control_call" => self.emit_control_call(operation, &operands),
            "qcore.select_call" => self.emit_select_call(operation, &operands),
            "classical.index" => self.emit_parameter_index(operation, &operands),
            "classical.len" if operands.len() == 1 => {
                let length = Self::collection_extent(&operands[0]);
                match length {
                    Ok(length) => Ok(vec![CircuitValue::Scalar(self.intern_scalar(
                        CircuitScalarNode::Int(
                            i64::try_from(length).map_err(|_| StaticCircuitError::Overflow)?,
                        ),
                    )?)]),
                    Err(_) => self.emit_symbolic_classical(operation, &operands),
                }
            }
            "classical.slice" => self.emit_classical_slice(operation, &operands),
            "classical.update" => self.emit_classical_update(operation, &operands),
            "classical.project" => self.emit_parameter_project(operation, &operands),
            opcode if opcode.starts_with("classical.") => {
                self.emit_symbolic_classical(operation, &operands)
            }
            opcode if opcode.starts_with("quantum.") => self.emit_quantum(operation, &operands),
            opcode => Err(StaticCircuitError::Unsupported(format!(
                "operation {opcode:?}"
            ))),
        }?;
        Ok(OperationFlow::Values(values))
    }

    fn emit_reshape(
        &mut self,
        operation: &wire::OperationProto,
        operands: &[CircuitValue],
    ) -> Result<Vec<CircuitValue>, StaticCircuitError> {
        let [source] = operands else {
            return Err(StaticCircuitError::Missing(
                "qcore.reshape operand ABI".to_owned(),
            ));
        };
        let target_type = operation
            .results
            .first()
            .and_then(|result| result.r#type.as_ref())
            .ok_or_else(|| StaticCircuitError::Missing("qcore.reshape result type".to_owned()))?;
        let mut wires = Vec::new();
        flatten_coherent_wires(source, &mut wires)?;
        if is_top_level_quantum_type(target_type) {
            if let (Some(arity), CircuitValue::Aggregate(parts)) =
                (top_level_quantum_product_arity(target_type), source)
                && arity == parts.len()
            {
                let widths = parts
                    .iter()
                    .map(|part| {
                        let mut part_wires = Vec::new();
                        flatten_coherent_wires(part, &mut part_wires)?;
                        Ok(part_wires.len())
                    })
                    .collect::<Result<Vec<_>, StaticCircuitError>>()?;
                self.coherent_partitions
                    .insert(wires.clone().into_boxed_slice(), widths.into_boxed_slice());
            }
            return Ok(vec![CircuitValue::Wires(wires.into_boxed_slice())]);
        }
        if let (CircuitValue::Wires(source_wires), Some(arity)) =
            (source, product_of_quantum_arity(target_type))
            && self
                .coherent_partitions
                .get(source_wires.as_ref())
                .is_some_and(|widths| widths.len() == arity)
        {
            let widths = self
                .coherent_partitions
                .get(source_wires.as_ref())
                .expect("partition count checked");
            let mut offset = 0_usize;
            let parts = widths
                .iter()
                .map(|width| {
                    let end = offset
                        .checked_add(*width)
                        .ok_or(StaticCircuitError::Overflow)?;
                    let part = source_wires.get(offset..end).ok_or_else(|| {
                        StaticCircuitError::Missing("qcore.reshape partition width".to_owned())
                    })?;
                    offset = end;
                    Ok(CircuitValue::Wires(part.to_vec().into_boxed_slice()))
                })
                .collect::<Result<Vec<_>, StaticCircuitError>>()?;
            if offset == source_wires.len() {
                return Ok(vec![CircuitValue::Aggregate(parts.into_boxed_slice())]);
            }
        }
        let mut position = 0_usize;
        let reshaped = pack_coherent_wires(target_type, &wires, &mut position)?;
        if position != wires.len() {
            return Err(StaticCircuitError::Missing(
                "qcore.reshape width equation".to_owned(),
            ));
        }
        Ok(vec![reshaped])
    }

    fn emit_phase_by(
        &mut self,
        operation: &wire::OperationProto,
        operands: &[CircuitValue],
    ) -> Result<Vec<CircuitValue>, StaticCircuitError> {
        let [state] = operands else {
            return Err(StaticCircuitError::Missing(
                "qcore.phase_by operand ABI".to_owned(),
            ));
        };
        let wires = require_wires(Some(state), "qcore.phase_by")?;
        let predicate = string_attribute(operation, "predicate")
            .ok_or_else(|| StaticCircuitError::Missing("phase predicate".to_owned()))?;
        let plan = self.boolean_plans.get(predicate).ok_or_else(|| {
            StaticCircuitError::Missing(format!("verified phase plan for {predicate:?}"))
        })?;
        if usize::try_from(plan.input_width()).ok() != Some(wires.len()) {
            return Err(StaticCircuitError::Missing(
                "phase predicate width equation".to_owned(),
            ));
        }
        if self
            .lowering
            .capabilities
            .contains(RealizationCapabilities::COHERENT_REGIONS)
        {
            self.push_instruction(
                "phase_oracle",
                wires.iter().copied(),
                [CircuitParameter::BooleanPlan(plan.clone())],
            )?;
            return Ok(vec![state.clone()]);
        }
        for monomial in plan.monomials() {
            if monomial.variables().is_empty() {
                let angle = self
                    .intern_scalar(CircuitScalarNode::RealBits(std::f64::consts::PI.to_bits()))?;
                self.push_instruction(
                    "global_phase",
                    std::iter::empty(),
                    [CircuitParameter::Scalar(angle)],
                )?;
                continue;
            }
            let selected = monomial
                .variables()
                .iter()
                .map(|index| {
                    usize::try_from(*index)
                        .ok()
                        .and_then(|index| wires.get(index))
                        .copied()
                        .ok_or_else(|| {
                            StaticCircuitError::Missing("phase monomial input position".to_owned())
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let (&target, controls) = selected
                .split_last()
                .ok_or_else(|| StaticCircuitError::Missing("phase monomial".to_owned()))?;
            let control_count = self.controls.len();
            self.controls.extend_from_slice(controls);
            let result = self.push_instruction("z", [target], std::iter::empty());
            self.controls.truncate(control_count);
            result?;
        }
        Ok(vec![state.clone()])
    }

    fn emit_xor_by(
        &mut self,
        operation: &wire::OperationProto,
        operands: &[CircuitValue],
    ) -> Result<Vec<CircuitValue>, StaticCircuitError> {
        let [CircuitValue::Aggregate(parts)] = operands else {
            return Err(StaticCircuitError::Missing(
                "qcore.xor_by operand ABI".to_owned(),
            ));
        };
        let [source, target] = parts.as_ref() else {
            return Err(StaticCircuitError::Missing(
                "qcore.xor_by packed state ABI".to_owned(),
            ));
        };
        let source_wires = require_wires(Some(source), "qcore.xor_by source")?;
        let target_wires = require_wires(Some(target), "qcore.xor_by target")?;
        let predicate = string_attribute(operation, "predicate")
            .ok_or_else(|| StaticCircuitError::Missing("Boolean predicate".to_owned()))?;
        let plan = self.boolean_plans.get(predicate).ok_or_else(|| {
            StaticCircuitError::Missing(format!("verified Boolean plan for {predicate:?}"))
        })?;
        if usize::try_from(plan.input_width()).ok() != Some(source_wires.len()) {
            return Err(StaticCircuitError::Missing(
                "Boolean predicate source-width equation".to_owned(),
            ));
        }
        if plan.outputs().len() != target_wires.len() {
            return Err(StaticCircuitError::Missing(
                "Boolean predicate target-width equation".to_owned(),
            ));
        }
        if self
            .lowering
            .capabilities
            .contains(RealizationCapabilities::COHERENT_REGIONS)
        {
            self.push_instruction(
                "xor_oracle",
                source_wires.iter().chain(target_wires).copied(),
                [CircuitParameter::BooleanPlan(plan.clone())],
            )?;
            return Ok(vec![CircuitValue::Aggregate(
                vec![source.clone(), target.clone()].into_boxed_slice(),
            )]);
        }
        for (output, target_wire) in plan.outputs().iter().zip(target_wires) {
            for monomial in output.monomials() {
                let controls = monomial
                    .variables()
                    .iter()
                    .map(|index| {
                        usize::try_from(*index)
                            .ok()
                            .and_then(|index| source_wires.get(index))
                            .copied()
                            .ok_or_else(|| {
                                StaticCircuitError::Missing(
                                    "Boolean monomial input position".to_owned(),
                                )
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let control_count = self.controls.len();
                self.controls.extend_from_slice(&controls);
                let result = self.push_instruction("x", [*target_wire], std::iter::empty());
                self.controls.truncate(control_count);
                result?;
            }
        }
        Ok(vec![CircuitValue::Aggregate(
            vec![source.clone(), target.clone()].into_boxed_slice(),
        )])
    }

    fn emit_permute_by(
        &mut self,
        operation: &wire::OperationProto,
        operands: &[CircuitValue],
        adjoint: bool,
    ) -> Result<Vec<CircuitValue>, StaticCircuitError> {
        let [state] = operands else {
            return Err(StaticCircuitError::Missing(
                "qcore.permute_by operand ABI".to_owned(),
            ));
        };
        let wires = require_wires(Some(state), "qcore.permute_by")?;
        let predicate = string_attribute(operation, "predicate")
            .ok_or_else(|| StaticCircuitError::Missing("Boolean permutation".to_owned()))?;
        let plan = self.boolean_plans.get(predicate).ok_or_else(|| {
            StaticCircuitError::Missing(format!("verified Boolean plan for {predicate:?}"))
        })?;
        if plan.outputs().len() != wires.len() {
            return Err(StaticCircuitError::Missing(
                "Boolean permutation width equation".to_owned(),
            ));
        }
        if self
            .lowering
            .capabilities
            .contains(RealizationCapabilities::COHERENT_REGIONS)
            && wires.len() <= 1024
            && plan.outputs().iter().all(|output| {
                output
                    .monomials()
                    .iter()
                    .all(|monomial| monomial.variables().len() == 1)
            })
        {
            self.push_instruction(
                if adjoint {
                    "linear_function_inverse"
                } else {
                    "linear_function"
                },
                wires.iter().copied(),
                [CircuitParameter::BooleanPlan(plan.clone())],
            )?;
            return Ok(vec![state.clone()]);
        }
        match (plan.permutation_order(), plan.basis_transpositions()) {
            (Some(order), None) if order.len() == wires.len() => {
                if adjoint {
                    for target in order.iter().rev() {
                        self.emit_permutation_target(plan, wires, *target, true)?;
                    }
                } else {
                    for target in order {
                        self.emit_permutation_target(plan, wires, *target, false)?;
                    }
                }
            }
            (None, Some(transpositions)) => {
                if adjoint {
                    for transposition in transpositions.iter().rev() {
                        self.emit_basis_transposition(wires, *transposition)?;
                    }
                } else {
                    for transposition in transpositions {
                        self.emit_basis_transposition(wires, *transposition)?;
                    }
                }
            }
            _ => {
                return Err(StaticCircuitError::Missing(format!(
                    "verified permutation plan for {predicate:?}"
                )));
            }
        }
        Ok(vec![state.clone()])
    }

    fn emit_basis_transposition(
        &mut self,
        wires: &[u64],
        transposition: phase_lifting::BooleanBasisTransposition,
    ) -> Result<(), StaticCircuitError> {
        let width = wires.len();
        let differing = transposition.left() ^ transposition.right();
        let state_limit = 1_u64
            .checked_shl(u32::try_from(width).map_err(|_| StaticCircuitError::Overflow)?)
            .ok_or(StaticCircuitError::Overflow)?;
        if differing == 0
            || transposition.left() >= state_limit
            || transposition.right() >= state_limit
        {
            return Err(StaticCircuitError::Missing(
                "Boolean basis transposition range".to_owned(),
            ));
        }
        let mut current = transposition.left();
        let mut gates = Vec::new();
        for target in 0..width {
            if differing & (1_u64 << target) == 0 {
                continue;
            }
            gates.push((target, current));
            current ^= 1_u64 << target;
        }
        if current != transposition.right() {
            return Err(StaticCircuitError::Missing(
                "Boolean basis transposition path".to_owned(),
            ));
        }
        for (target, basis_state) in gates.iter().chain(gates.iter().rev().skip(1)).copied() {
            self.emit_basis_state_flip(wires, target, basis_state)?;
        }
        Ok(())
    }

    fn emit_basis_state_flip(
        &mut self,
        wires: &[u64],
        target: usize,
        basis_state: u64,
    ) -> Result<(), StaticCircuitError> {
        let zero_controls = wires
            .iter()
            .enumerate()
            .filter_map(|(index, wire)| {
                (index != target && basis_state & (1_u64 << index) == 0).then_some(*wire)
            })
            .collect::<Vec<_>>();
        for wire in &zero_controls {
            self.push_instruction("x", [*wire], [])?;
        }
        let outer_control_count = self.controls.len();
        self.controls.extend(
            wires
                .iter()
                .enumerate()
                .filter_map(|(index, wire)| (index != target).then_some(*wire)),
        );
        let result = self.push_instruction("x", [wires[target]], []);
        self.controls.truncate(outer_control_count);
        result?;
        for wire in zero_controls.iter().rev() {
            self.push_instruction("x", [*wire], [])?;
        }
        Ok(())
    }

    fn emit_permutation_target(
        &mut self,
        plan: &phase_lifting::BooleanLiftPlan,
        wires: &[u64],
        target: u64,
        adjoint: bool,
    ) -> Result<(), StaticCircuitError> {
        let target_index = usize::try_from(target)
            .ok()
            .filter(|index| *index < wires.len())
            .ok_or_else(|| {
                StaticCircuitError::Missing("Boolean permutation target position".to_owned())
            })?;
        let output = plan.outputs().get(target_index).ok_or_else(|| {
            StaticCircuitError::Missing("Boolean permutation output position".to_owned())
        })?;
        if adjoint {
            for monomial in output.monomials().iter().rev() {
                self.emit_permutation_monomial(wires, target_index, monomial)?;
            }
        } else {
            for monomial in output.monomials() {
                self.emit_permutation_monomial(wires, target_index, monomial)?;
            }
        }
        Ok(())
    }

    fn emit_permutation_monomial(
        &mut self,
        wires: &[u64],
        target: usize,
        monomial: &phase_lifting::BooleanMonomial,
    ) -> Result<(), StaticCircuitError> {
        if monomial.variables() == [u64::try_from(target).unwrap_or(u64::MAX)] {
            return Ok(());
        }
        let controls = monomial
            .variables()
            .iter()
            .map(|index| {
                usize::try_from(*index)
                    .ok()
                    .filter(|index| *index != target)
                    .and_then(|index| wires.get(index))
                    .copied()
                    .ok_or_else(|| {
                        StaticCircuitError::Missing(
                            "Boolean permutation control position".to_owned(),
                        )
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let control_count = self.controls.len();
        self.controls.extend_from_slice(&controls);
        let result = self.push_instruction("x", [wires[target]], std::iter::empty());
        self.controls.truncate(control_count);
        result
    }

    fn charge_aggregate_projection(&mut self, items: usize) -> Result<(), StaticCircuitError> {
        self.aggregate_projection_work = self
            .aggregate_projection_work
            .checked_add(items)
            .ok_or(StaticCircuitError::Overflow)?;
        if self.aggregate_projection_work > MAX_STATIC_CIRCUIT_INSTRUCTIONS {
            return Err(StaticCircuitError::Unsupported(
                "aggregate projection exceeds its total work budget".to_owned(),
            ));
        }
        Ok(())
    }

    fn collection_extent(source: &CircuitValue) -> Result<usize, StaticCircuitError> {
        match source {
            CircuitValue::Aggregate(values) => Ok(values.len()),
            CircuitValue::Parameter {
                shape: ParameterShape::Array(extent),
                ..
            } => usize::try_from(*extent).map_err(|_| StaticCircuitError::Overflow),
            _ => Err(StaticCircuitError::Unsupported(
                "classical collection projection".to_owned(),
            )),
        }
    }

    fn collection_element(
        &mut self,
        operation: &wire::OperationProto,
        source: &CircuitValue,
        index: usize,
    ) -> Result<CircuitValue, StaticCircuitError> {
        match source {
            CircuitValue::Aggregate(values) => values
                .get(index)
                .cloned()
                .ok_or(StaticCircuitError::Overflow),
            CircuitValue::Parameter { name, path, .. } => {
                let result = operation.results.first().ok_or_else(|| {
                    StaticCircuitError::Missing("collection result type".to_owned())
                })?;
                let mut arena = crate::types::TypeArena::default();
                let scope = arena.intern_scope("collection_projection");
                let root = arena
                    .resolve_wire_type(
                        result.r#type.as_ref().ok_or_else(|| {
                            StaticCircuitError::Missing("collection type".to_owned())
                        })?,
                        &scope,
                    )
                    .map_err(StaticCircuitError::Missing)?;
                let element = arena
                    .sequence_element(root)
                    .map_err(StaticCircuitError::Missing)?
                    .ok_or_else(|| StaticCircuitError::Missing("collection element".to_owned()))?;
                let mut projected = operation.clone();
                projected.results[0].r#type = Some(
                    arena
                        .encode_wire_type(element)
                        .map_err(StaticCircuitError::Missing)?,
                );
                let value = self.project_parameter(
                    &projected,
                    name,
                    path,
                    CircuitParameterPath::Index(index as u64),
                )?;
                value.into_iter().next().ok_or_else(|| {
                    StaticCircuitError::Missing("projected collection element".to_owned())
                })
            }
            _ => Err(StaticCircuitError::Unsupported(
                "classical collection projection".to_owned(),
            )),
        }
    }

    fn emit_classical_slice(
        &mut self,
        operation: &wire::OperationProto,
        operands: &[CircuitValue],
    ) -> Result<Vec<CircuitValue>, StaticCircuitError> {
        use crate::core::slice_shape::{SliceBound, operation_bounds};
        let source = operands
            .first()
            .ok_or_else(|| StaticCircuitError::Missing("slice source".to_owned()))?;
        let descriptors = operation_bounds(operation).map_err(StaticCircuitError::Missing)?;
        let mut supplied = operands.iter().skip(1);
        let mut bounds = Vec::new();
        for descriptor in descriptors {
            if descriptor == SliceBound::Omitted {
                bounds.push(None);
            } else {
                let value = supplied
                    .next()
                    .and_then(|v| self.static_int(v))
                    .ok_or(StaticCircuitError::DynamicClassicalAggregateIndex)?;
                bounds.push(Some(BigInt::from(value)));
            }
        }
        let positions = static_values::normalize_aggregate_slice(
            Self::collection_extent(source)?,
            bounds[0].as_ref(),
            bounds[1].as_ref(),
            bounds[2].as_ref(),
        )
        .map_err(|e| StaticCircuitError::Unsupported(e.to_string()))?;
        self.charge_aggregate_projection(positions.len())?;
        let values = positions
            .iter()
            .map(|i| self.collection_element(operation, source, *i))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(vec![CircuitValue::Aggregate(values.into_boxed_slice())])
    }

    fn emit_classical_update(
        &mut self,
        operation: &wire::OperationProto,
        operands: &[CircuitValue],
    ) -> Result<Vec<CircuitValue>, StaticCircuitError> {
        let [source, index, replacement] = operands else {
            return Err(StaticCircuitError::Missing("update operands".to_owned()));
        };
        let extent = Self::collection_extent(source)?;
        if extent > static_values::MAX_AGGREGATE_SLICE_ITEMS {
            return Err(StaticCircuitError::Unsupported(
                "aggregate update exceeds its item budget".to_owned(),
            ));
        }
        let index = self
            .static_int(index)
            .ok_or(StaticCircuitError::DynamicClassicalAggregateIndex)?;
        let position =
            normalize_aggregate_index(extent, &BigInt::from(index)).ok_or_else(|| {
                StaticCircuitError::InvalidSelector(
                    "aggregate update index is out of bounds".to_owned(),
                )
            })?;
        self.charge_aggregate_projection(extent)?;
        let mut values = Vec::with_capacity(extent);
        for i in 0..extent {
            values.push(if i == position {
                replacement.clone()
            } else {
                self.collection_element(operation, source, i)?
            });
        }
        Ok(vec![CircuitValue::Aggregate(values.into_boxed_slice())])
    }

    fn emit_parameter_index(
        &mut self,
        operation: &wire::OperationProto,
        operands: &[CircuitValue],
    ) -> Result<Vec<CircuitValue>, StaticCircuitError> {
        let [parameter, selector] = operands else {
            return Err(StaticCircuitError::Missing(
                "classical.index operand ABI".to_owned(),
            ));
        };
        let index = self
            .static_int(selector)
            .ok_or(StaticCircuitError::DynamicClassicalAggregateIndex)?;
        match parameter {
            CircuitValue::Aggregate(values) => {
                Ok(vec![aggregate_value_at(values, index, "classical.index")?])
            }
            CircuitValue::Parameter { name, path, shape } => {
                let component = parameter_component(shape, index)?;
                self.project_parameter(operation, name, path, component)
            }
            _ => self.emit_symbolic_classical(operation, operands),
        }
    }

    fn emit_parameter_project(
        &mut self,
        operation: &wire::OperationProto,
        operands: &[CircuitValue],
    ) -> Result<Vec<CircuitValue>, StaticCircuitError> {
        let [source] = operands else {
            return Err(StaticCircuitError::Missing(
                "classical.project operand ABI".to_owned(),
            ));
        };
        let position = int_attribute(operation, "position").ok_or_else(|| {
            StaticCircuitError::Missing("parameter projection position".to_owned())
        })?;
        match source {
            CircuitValue::Aggregate(values) => Ok(vec![aggregate_value_at(
                values,
                position,
                "classical.project",
            )?]),
            CircuitValue::Parameter { name, path, shape } => {
                let component = parameter_component(shape, position)?;
                self.project_parameter(operation, name, path, component)
            }
            _ => self.emit_symbolic_classical(operation, operands),
        }
    }

    fn project_parameter(
        &mut self,
        operation: &wire::OperationProto,
        name: &str,
        path: &[CircuitParameterPath],
        component: CircuitParameterPath,
    ) -> Result<Vec<CircuitValue>, StaticCircuitError> {
        let result_type = operation
            .results
            .first()
            .and_then(|result| result.r#type.as_ref())
            .ok_or_else(|| {
                StaticCircuitError::Missing(format!(
                    "parameter projection result type for {:?}",
                    operation.opcode
                ))
            })?;
        let mut projected_path = Vec::with_capacity(path.len() + 1);
        projected_path.extend_from_slice(path);
        projected_path.push(component);
        let value = self.parameter_value(
            name.to_owned(),
            projected_path.into_boxed_slice(),
            result_type,
        )?;
        Ok(vec![value])
    }

    fn emit_symbolic_classical(
        &mut self,
        operation: &wire::OperationProto,
        operands: &[CircuitValue],
    ) -> Result<Vec<CircuitValue>, StaticCircuitError> {
        let opcode = operation.opcode.as_str();
        let scalar_environment = operation
            .operands
            .iter()
            .zip(operands)
            .map(|(operand, value)| (*operand, self.static_value(value)))
            .collect::<HashMap<_, _>>();
        let static_result = self.evaluate_static_classical(operation, &scalar_environment)?;
        if let Some(value) = static_result.as_ref().and_then(StaticValue::scalar) {
            let node = match value {
                StaticScalar::Bool(value) => CircuitScalarNode::Bool(value),
                StaticScalar::Int(value) => CircuitScalarNode::Int(value),
                StaticScalar::RealBits(value) => CircuitScalarNode::RealBits(value),
            };
            return Ok(vec![CircuitValue::Scalar(self.intern_scalar(node)?)]);
        }
        if requires_launch_scalar_evaluation(operation)
            && self.measurement_guards.is_empty()
            && operation
                .results
                .first()
                .and_then(|result| result.r#type.as_ref())
                .is_some_and(|result| {
                    result.kind() == wire::type_proto::Kind::Staged && result.name == "Param"
                })
        {
            let reason = if opcode == "classical.sqrt" {
                "sqrt operands for native domain validation"
            } else {
                "Real cast operands for native numeric conversion"
            };
            return Err(StaticCircuitError::LaunchMaterialization(reason.to_owned()));
        }
        let result_type = operation
            .results
            .first()
            .and_then(|result| result.r#type.as_ref())
            .and_then(circuit_scalar_type);
        if result_type.is_none()
            && let Some(value) = &static_result
        {
            return Ok(vec![self.static_value_to_circuit(value)?]);
        }
        let scalar_operands = operands
            .iter()
            .map(|value| Self::scalar_value_identifier(value, opcode))
            .collect::<Result<Vec<_>, _>>();
        if scalar_operands.is_err()
            && let Some(value) = &static_result
        {
            return Ok(vec![self.static_value_to_circuit(value)?]);
        }
        let scalar_operands = scalar_operands?;
        let result_type = result_type.ok_or_else(|| {
            StaticCircuitError::Missing(format!("scalar result type for {opcode:?}"))
        })?;
        let identifier = self.intern_scalar(CircuitScalarNode::Apply {
            opcode: opcode.to_owned(),
            result_type,
            attributes: circuit_scalar_attributes(operation)?,
            operands: scalar_operands.into_boxed_slice(),
        })?;
        if let Some(value) = static_result {
            self.known_scalars.insert(identifier, value);
        }
        if result_type != CircuitScalarType::Bool
            || matches!(opcode, "classical.index" | "classical.map_get")
        {
            let neutral = self.intern_scalar(match result_type {
                CircuitScalarType::Real => CircuitScalarNode::RealBits(0.0_f64.to_bits()),
                CircuitScalarType::Int => CircuitScalarNode::Int(0),
                CircuitScalarType::Bool => CircuitScalarNode::Bool(false),
            })?;
            let mut root = identifier;
            for (condition, when_true) in self.measurement_guards.clone().into_iter().rev() {
                root = if when_true {
                    self.select_scalar(condition, root, neutral)?
                } else {
                    self.select_scalar(condition, neutral, root)?
                };
            }
            self.scalar_checks.push(CircuitScalarCheck {
                instruction_index: self.instructions.len(),
                root,
            });
        }
        Ok(vec![CircuitValue::Scalar(identifier)])
    }

    fn evaluate_static_classical(
        &self,
        operation: &wire::OperationProto,
        environment: &HashMap<u64, Option<StaticValue>>,
    ) -> Result<Option<StaticValue>, StaticCircuitError> {
        // Measurement guards belong to the native lazy evaluator; folding here
        // could evaluate a partial expression in an untaken source branch.
        if !self.measurement_guards.is_empty() {
            return Ok(None);
        }
        if operation.opcode == "classical.map_get" || requires_launch_scalar_evaluation(operation) {
            static_values::evaluate_value_checked(operation, environment).map_err(|error| {
                let reason = match error {
                    NumericEvaluationError::Undefined(reason)
                    | NumericEvaluationError::Limit(reason) => reason,
                };
                StaticCircuitError::Unsupported(reason.to_owned())
            })
        } else {
            Ok(static_values::evaluate_value(operation, environment))
        }
    }

    fn scalar_value_identifier(
        value: &CircuitValue,
        opcode: &str,
    ) -> Result<CircuitScalarId, StaticCircuitError> {
        match value {
            CircuitValue::Scalar(identifier) => Ok(*identifier),
            _ => Err(StaticCircuitError::Unsupported(format!(
                "non-scalar symbolic classical operation {opcode:?}"
            ))),
        }
    }

    fn emit_call(
        &mut self,
        operation: &wire::OperationProto,
        operands: &[CircuitValue],
    ) -> Result<Vec<CircuitValue>, StaticCircuitError> {
        let callee_name = string_attribute(operation, "callee")
            .ok_or_else(|| StaticCircuitError::Missing("callee".to_owned()))?;
        let callee = self
            .functions
            .get(callee_name)
            .copied()
            .ok_or_else(|| StaticCircuitError::Missing(format!("callable {callee_name:?}")))?;
        let start = self.instructions.len();
        let flow = self.emit_retained_callable(callee_name, callee, operands)?;
        if flow.kind != "return" {
            return Err(StaticCircuitError::Missing(format!(
                "return terminator in {callee_name:?}"
            )));
        }
        let instructions = self.instructions.split_off(start);
        let mut runtime_returns = Vec::new();
        collect_runtime_returns(&instructions, &mut runtime_returns);
        if runtime_returns.is_empty() {
            if let Some(label) = named_call_label(callee_name).filter(|_| {
                self.lowering.capabilities.supports_named_call()
                    && !self
                        .lowering
                        .capabilities
                        .contains(RealizationCapabilities::COHERENT_REGIONS)
            }) {
                if self.instruction_count >= MAX_STATIC_CIRCUIT_INSTRUCTIONS {
                    return Err(StaticCircuitError::Unsupported(format!(
                        "more than {MAX_STATIC_CIRCUIT_INSTRUCTIONS} circuit instructions"
                    )));
                }
                self.instruction_count += 1;
                let mut qubits = std::collections::BTreeSet::new();
                collect_instruction_wires(&instructions, &mut qubits);
                self.instructions.push(StaticInstruction {
                    opcode: format!("box:{label}"),
                    qubits: qubits.into_iter().collect::<Vec<_>>().into_boxed_slice(),
                    parameters: Box::new([]),
                    controls: Box::new([]),
                    condition: None,
                    regions: vec![StaticBlock {
                        instructions: instructions.into_boxed_slice(),
                        return_value: None,
                    }]
                    .into_boxed_slice(),
                });
                return Ok(flow.values);
            }
            self.instructions.extend(instructions);
            return Ok(flow.values);
        }
        let result_type = callee.result_type.as_ref().ok_or_else(|| {
            StaticCircuitError::Missing(format!("result type for {callee_name:?}"))
        })?;
        let [normal_value] = flow.values.as_slice() else {
            return Err(StaticCircuitError::Unsupported(
                "runtime call result ABI".to_owned(),
            ));
        };
        let normal_result = retain_result_value(normal_value, result_type).ok_or_else(|| {
            StaticCircuitError::Unsupported("runtime call result type".to_owned())
        })?;
        for runtime_return in runtime_returns {
            if !runtime_results_compatible(&normal_result, runtime_return, &self.scalars) {
                return Err(StaticCircuitError::Unsupported(
                    "runtime call return alternatives".to_owned(),
                ));
            }
        }
        let mut parameters = Vec::new();
        let value = self.runtime_call_value(normal_value, &mut parameters)?;
        if self.instruction_count >= MAX_STATIC_CIRCUIT_INSTRUCTIONS {
            return Err(StaticCircuitError::Unsupported(format!(
                "more than {MAX_STATIC_CIRCUIT_INSTRUCTIONS} circuit instructions"
            )));
        }
        self.instruction_count += 1;
        self.instructions.push(StaticInstruction {
            opcode: "call".to_owned(),
            qubits: Box::new([]),
            parameters: parameters.into_boxed_slice(),
            controls: Box::new([]),
            condition: None,
            regions: vec![StaticBlock {
                instructions: instructions.into_boxed_slice(),
                return_value: Some(normal_result),
            }]
            .into_boxed_slice(),
        });
        Ok(vec![value])
    }

    fn runtime_call_value(
        &mut self,
        value: &CircuitValue,
        parameters: &mut Vec<CircuitParameter>,
    ) -> Result<CircuitValue, StaticCircuitError> {
        match value {
            CircuitValue::Scalar(identifier) => {
                let scalar_type = self
                    .scalar_node(*identifier)
                    .map(CircuitScalarNode::scalar_type)
                    .ok_or_else(|| StaticCircuitError::Missing("call result scalar".to_owned()))?;
                let index = self.next_call_value;
                self.next_call_value = self
                    .next_call_value
                    .checked_add(1)
                    .ok_or(StaticCircuitError::Overflow)?;
                parameters.push(CircuitParameter::CallResult {
                    target: index,
                    scalar_type,
                });
                Ok(CircuitValue::Scalar(self.intern_scalar(
                    CircuitScalarNode::CallValue { index, scalar_type },
                )?))
            }
            CircuitValue::Portable(value) => Ok(CircuitValue::Portable(value.clone())),
            CircuitValue::Aggregate(values) => Ok(CircuitValue::Aggregate(
                values
                    .iter()
                    .map(|value| self.runtime_call_value(value, parameters))
                    .collect::<Result<Vec<_>, _>>()?
                    .into_boxed_slice(),
            )),
            CircuitValue::Unit => Ok(CircuitValue::Unit),
            CircuitValue::Wires(wires) => Ok(CircuitValue::Wires(wires.clone())),
            CircuitValue::Hamiltonian(_) | CircuitValue::Parameter { .. } => Err(
                StaticCircuitError::Unsupported("runtime call result type".to_owned()),
            ),
        }
    }

    fn emit_control_call(
        &mut self,
        operation: &wire::OperationProto,
        operands: &[CircuitValue],
    ) -> Result<Vec<CircuitValue>, StaticCircuitError> {
        let callee_name = string_attribute(operation, "callee")
            .ok_or_else(|| StaticCircuitError::Missing("controlled callee".to_owned()))?;
        let callee = self
            .functions
            .get(callee_name)
            .copied()
            .ok_or_else(|| StaticCircuitError::Missing(format!("callable {callee_name:?}")))?;
        let offset = operands
            .len()
            .checked_sub(callee.parameters.len())
            .filter(|offset| matches!(offset, 1 | 2))
            .ok_or_else(|| {
                StaticCircuitError::Missing("controlled callable argument ABI".to_owned())
            })?;
        let all_controls = require_wires(operands.first(), "qcore.control_call")?;
        let selected_controls = if offset == 2 {
            let component = operands
                .get(1)
                .and_then(|value| self.static_int(value))
                .ok_or_else(|| {
                    StaticCircuitError::Unsupported("dynamic controlled-call component".to_owned())
                })?;
            vec![wire_at(
                all_controls,
                component,
                "controlled-call component",
            )?]
        } else {
            all_controls.to_vec()
        };
        let control_state = attribute(operation, "control_state")
            .and_then(ControlState::decode)
            .ok_or_else(|| StaticCircuitError::Missing("controlled basis state".to_owned()))?;
        let result = self.emit_controlled_callee(
            callee_name,
            callee,
            &selected_controls,
            &control_state,
            &operands[offset..],
        )?;
        Ok(vec![CircuitValue::Aggregate(
            vec![CircuitValue::Wires(all_controls.into()), result].into_boxed_slice(),
        )])
    }

    fn emit_select_call(
        &mut self,
        operation: &wire::OperationProto,
        operands: &[CircuitValue],
    ) -> Result<Vec<CircuitValue>, StaticCircuitError> {
        let index = require_wires(operands.first(), "qcore.select_call")?;
        let mut current = operands
            .get(1)
            .cloned()
            .ok_or_else(|| StaticCircuitError::Missing("selected callable state".to_owned()))?;
        let classical = operands.get(2..).unwrap_or_default();
        let callees = string_tuple_attribute(operation, "callees")
            .ok_or_else(|| StaticCircuitError::Missing("selected callable table".to_owned()))?;
        for (control_state, callee_name) in callees.into_iter().enumerate() {
            let callee =
                self.functions.get(callee_name).copied().ok_or_else(|| {
                    StaticCircuitError::Missing(format!("callable {callee_name:?}"))
                })?;
            let mut arguments = Vec::with_capacity(classical.len() + 1);
            arguments.push(current);
            arguments.extend(classical.iter().cloned());
            current = self.emit_controlled_callee(
                callee_name,
                callee,
                index,
                &ControlState::exact_u64(
                    u64::try_from(control_state).map_err(|_| StaticCircuitError::Overflow)?,
                ),
                &arguments,
            )?;
        }
        Ok(vec![CircuitValue::Aggregate(
            vec![CircuitValue::Wires(index.into()), current].into_boxed_slice(),
        )])
    }

    fn emit_controlled_callee(
        &mut self,
        callee_name: &str,
        callee: &wire::FunctionProto,
        controls: &[u64],
        control_state: &ControlState,
        arguments: &[CircuitValue],
    ) -> Result<CircuitValue, StaticCircuitError> {
        if !control_state
            .fits(u64::try_from(controls.len()).map_err(|_| StaticCircuitError::Overflow)?)
        {
            return Err(StaticCircuitError::InvalidSelector(format!(
                "controlled basis state {control_state}"
            )));
        }
        let zero_controls = controls
            .iter()
            .enumerate()
            .filter_map(|(index, wire)| {
                (!control_state.component_is_active(index)).then_some(*wire)
            })
            .collect::<Vec<_>>();
        for wire in &zero_controls {
            self.push_instruction("x", [*wire], [])?;
        }
        let outer_control_count = self.controls.len();
        self.controls.extend(controls.iter().copied());
        let flow = self.emit_retained_callable(callee_name, callee, arguments);
        self.controls.truncate(outer_control_count);
        let flow = flow?;
        for wire in zero_controls.iter().rev() {
            self.push_instruction("x", [*wire], [])?;
        }
        if flow.kind != "return" || flow.values.len() != 1 {
            return Err(StaticCircuitError::Missing(format!(
                "controlled return ABI for {callee_name:?}"
            )));
        }
        Ok(flow
            .values
            .into_iter()
            .next()
            .expect("one controlled result was checked"))
    }

    fn emit_adjoint_call(
        &mut self,
        operation: &wire::OperationProto,
        operands: &[CircuitValue],
    ) -> Result<Vec<CircuitValue>, StaticCircuitError> {
        let callee_name = string_attribute(operation, "callee")
            .ok_or_else(|| StaticCircuitError::Missing("adjoint callee".to_owned()))?;
        let callee = self
            .functions
            .get(callee_name)
            .copied()
            .ok_or_else(|| StaticCircuitError::Missing(format!("callable {callee_name:?}")))?;
        let packed = operands
            .first()
            .ok_or_else(|| StaticCircuitError::Missing("adjoint packed state".to_owned()))?;
        let mut wires = Vec::new();
        flatten_coherent_wires(packed, &mut wires)?;
        let linear_parameters = callee
            .parameters
            .iter()
            .filter(|parameter| {
                parameter
                    .value
                    .as_ref()
                    .is_some_and(|definition| definition.has_region)
            })
            .collect::<Vec<_>>();
        let direct_quantum_adjoint = linear_parameters.len() == 1
            && linear_parameters.first().is_some_and(|parameter| {
                parameter
                    .value
                    .as_ref()
                    .and_then(|definition| definition.r#type.as_ref())
                    .is_some_and(is_top_level_quantum_type)
            })
            && matches!(packed, CircuitValue::Wires(_));
        let mut wire_position = 0_usize;
        let mut classical = operands.get(1..).unwrap_or_default().iter();
        let arguments = callee
            .parameters
            .iter()
            .map(|parameter| {
                let definition = parameter.value.as_ref().ok_or_else(|| {
                    StaticCircuitError::Missing(format!("parameter value in {callee_name:?}"))
                })?;
                if definition.has_region {
                    if direct_quantum_adjoint {
                        wire_position = wires.len();
                        return Ok(packed.clone());
                    }
                    let parameter_type = definition.r#type.as_ref().ok_or_else(|| {
                        StaticCircuitError::Missing(format!(
                            "linear parameter type in {callee_name:?}"
                        ))
                    })?;
                    pack_coherent_wires(parameter_type, &wires, &mut wire_position)
                } else {
                    classical.next().cloned().ok_or_else(|| {
                        StaticCircuitError::Missing(format!(
                            "classical adjoint argument in {callee_name:?}"
                        ))
                    })
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        if wire_position != wires.len() || classical.next().is_some() {
            return Err(StaticCircuitError::Missing(format!(
                "adjoint argument ABI for {callee_name:?}"
            )));
        }
        let instruction_start = self.instructions.len();
        let flow = self.emit_retained_callable(callee_name, callee, &arguments)?;
        if flow.kind != "return" || flow.values.len() != 1 {
            return Err(StaticCircuitError::Missing(format!(
                "adjoint return ABI for {callee_name:?}"
            )));
        }
        let forward = self.instructions.split_off(instruction_start);
        let inverse = forward
            .into_iter()
            .rev()
            .map(|instruction| self.invert_instruction(instruction))
            .collect::<Result<Vec<_>, _>>()?;
        self.instructions.extend(inverse);
        let result_type = operation
            .results
            .first()
            .and_then(|result| result.r#type.as_ref())
            .ok_or_else(|| StaticCircuitError::Missing("adjoint result type".to_owned()))?;
        if direct_quantum_adjoint && is_top_level_quantum_type(result_type) {
            return Ok(vec![packed.clone()]);
        }
        let mut result_position = 0_usize;
        let result = pack_coherent_wires(result_type, &wires, &mut result_position)?;
        if result_position != wires.len() {
            return Err(StaticCircuitError::Missing(
                "adjoint result width equation".to_owned(),
            ));
        }
        Ok(vec![result])
    }

    fn emit_adjoint(
        &mut self,
        operation: &wire::OperationProto,
        operands: &[CircuitValue],
    ) -> Result<Vec<CircuitValue>, StaticCircuitError> {
        let original = string_attribute(operation, "original")
            .ok_or_else(|| StaticCircuitError::Missing("adjoint original".to_owned()))?;
        if matches!(original, "qcore.control_call" | "qcore.select_call") {
            self.emit_coherent_adjoint(operation, operands, original)?;
            return operands
                .first()
                .cloned()
                .map(|value| vec![value])
                .ok_or_else(|| StaticCircuitError::Missing("adjoint state".to_owned()));
        }
        let synthetic = adjoint_original_operation(operation, original)?;
        if matches!(
            original,
            "qcore.phase_by" | "qcore.xor_by" | "qcore.permute_by"
        ) {
            return if original == "qcore.phase_by" {
                self.emit_phase_by(&synthetic, operands)
            } else if original == "qcore.permute_by" {
                self.emit_permute_by(&synthetic, operands, true)
            } else {
                self.emit_xor_by(&synthetic, operands)
            };
        }
        let instruction_start = self.instructions.len();
        let results = self.emit_quantum(&synthetic, operands)?;
        let forward = self.instructions.split_off(instruction_start);
        let inverse = forward
            .into_iter()
            .rev()
            .map(|instruction| self.invert_instruction(instruction))
            .collect::<Result<Vec<_>, _>>()?;
        self.instructions.extend(inverse);
        Ok(results)
    }

    fn emit_coherent_adjoint(
        &mut self,
        operation: &wire::OperationProto,
        operands: &[CircuitValue],
        original: &str,
    ) -> Result<(), StaticCircuitError> {
        let parts = match operands.first() {
            Some(CircuitValue::Aggregate(parts)) if parts.len() == 2 => parts,
            _ => {
                return Err(StaticCircuitError::Missing(
                    "coherent adjoint packed state".to_owned(),
                ));
            }
        };
        let selector = require_wires(parts.first(), original)?;
        let target = parts
            .get(1)
            .cloned()
            .ok_or_else(|| StaticCircuitError::Missing("coherent adjoint target".to_owned()))?;
        let classical = operands.get(1..).unwrap_or_default();
        if original == "qcore.control_call" {
            let callee_name = mapping_string_attribute(operation, "original_attributes", "callee")
                .ok_or_else(|| {
                    StaticCircuitError::Missing("controlled adjoint callee".to_owned())
                })?;
            let callee =
                self.functions.get(callee_name).copied().ok_or_else(|| {
                    StaticCircuitError::Missing(format!("callable {callee_name:?}"))
                })?;
            let offset = classical
                .len()
                .checked_add(2)
                .and_then(|count| count.checked_sub(callee.parameters.len()))
                .filter(|offset| matches!(offset, 1 | 2))
                .ok_or_else(|| {
                    StaticCircuitError::Missing("controlled adjoint argument ABI".to_owned())
                })?;
            let selected = if offset == 2 {
                let component = classical
                    .first()
                    .and_then(|value| self.static_int(value))
                    .ok_or_else(|| {
                        StaticCircuitError::Unsupported(
                            "dynamic controlled-adjoint component".to_owned(),
                        )
                    })?;
                vec![wire_at(
                    selector,
                    component,
                    "controlled-adjoint component",
                )?]
            } else {
                selector.to_vec()
            };
            let control_state = mapping_entry(operation, "original_attributes", "control_state")
                .and_then(ControlState::decode)
                .ok_or_else(|| {
                    StaticCircuitError::Missing("controlled adjoint state".to_owned())
                })?;
            let mut arguments = Vec::with_capacity(classical.len() - (offset - 1) + 1);
            arguments.push(target);
            arguments.extend(classical[offset - 1..].iter().cloned());
            return self.emit_controlled_adjoint_callee(
                callee_name,
                callee,
                &selected,
                &control_state,
                &arguments,
            );
        }
        let callees =
            mapping_string_tuple_attribute(operation, "original_attributes", "callees")
                .ok_or_else(|| StaticCircuitError::Missing("selected adjoint table".to_owned()))?;
        for (control_state, callee_name) in callees.into_iter().enumerate().rev() {
            let callee =
                self.functions.get(callee_name).copied().ok_or_else(|| {
                    StaticCircuitError::Missing(format!("callable {callee_name:?}"))
                })?;
            let mut arguments = Vec::with_capacity(classical.len() + 1);
            arguments.push(target.clone());
            arguments.extend(classical.iter().cloned());
            self.emit_controlled_adjoint_callee(
                callee_name,
                callee,
                selector,
                &ControlState::exact_u64(
                    u64::try_from(control_state).map_err(|_| StaticCircuitError::Overflow)?,
                ),
                &arguments,
            )?;
        }
        Ok(())
    }

    fn emit_controlled_adjoint_callee(
        &mut self,
        callee_name: &str,
        callee: &wire::FunctionProto,
        controls: &[u64],
        control_state: &ControlState,
        arguments: &[CircuitValue],
    ) -> Result<(), StaticCircuitError> {
        if !control_state
            .fits(u64::try_from(controls.len()).map_err(|_| StaticCircuitError::Overflow)?)
        {
            return Err(StaticCircuitError::InvalidSelector(format!(
                "controlled adjoint basis state {control_state}"
            )));
        }
        let zero_controls = controls
            .iter()
            .enumerate()
            .filter_map(|(index, wire)| {
                (!control_state.component_is_active(index)).then_some(*wire)
            })
            .collect::<Vec<_>>();
        for wire in &zero_controls {
            self.push_instruction("x", [*wire], [])?;
        }
        let outer_control_count = self.controls.len();
        self.controls.extend(controls.iter().copied());
        let instruction_start = self.instructions.len();
        let flow = self.emit_retained_callable(callee_name, callee, arguments);
        self.controls.truncate(outer_control_count);
        let flow = flow?;
        if flow.kind != "return" || flow.values.len() != 1 {
            return Err(StaticCircuitError::Missing(format!(
                "controlled adjoint return ABI for {callee_name:?}"
            )));
        }
        let forward = self.instructions.split_off(instruction_start);
        let inverse = forward
            .into_iter()
            .rev()
            .map(|instruction| self.invert_instruction(instruction))
            .collect::<Result<Vec<_>, _>>()?;
        self.instructions.extend(inverse);
        for wire in zero_controls.iter().rev() {
            self.push_instruction("x", [*wire], [])?;
        }
        Ok(())
    }

    #[expect(
        clippy::too_many_lines,
        reason = "Keep exhaustive structured and primitive adjoint dispatch together"
    )]
    fn invert_instruction(
        &mut self,
        instruction: StaticInstruction,
    ) -> Result<StaticInstruction, StaticCircuitError> {
        if self
            .lowering
            .capabilities
            .contains(RealizationCapabilities::COHERENT_REGIONS)
            && (instruction.opcode.starts_with("box:") || instruction.opcode == "deferred")
        {
            let mut instruction = instruction;
            instruction.opcode = format!("inverse:{}", instruction.opcode);
            return Ok(instruction);
        }
        if instruction.opcode == "inverse:deferred" {
            return Ok(StaticInstruction {
                opcode: "deferred".to_owned(),
                ..instruction
            });
        }
        if let Some(opcode) = instruction.opcode.strip_prefix("inverse:box:") {
            let opcode = format!("box:{opcode}");
            return Ok(StaticInstruction {
                opcode,
                ..instruction
            });
        }
        if instruction.opcode == "for" && instruction.condition.is_none() {
            let mut instruction = instruction;
            let bounds = instruction
                .parameters
                .iter()
                .take(3)
                .map(|parameter| match parameter {
                    CircuitParameter::Scalar(id) => Ok(*id),
                    _ => Err(StaticCircuitError::Missing("loop bound".to_owned())),
                })
                .collect::<Result<Vec<_>, _>>()?;
            let bounds: [CircuitScalarId; 3] = bounds
                .try_into()
                .map_err(|_| StaticCircuitError::Missing("loop bounds".to_owned()))?;
            for (slot, value) in instruction
                .parameters
                .iter_mut()
                .zip(self.reverse_for_bounds(&bounds)?)
            {
                *slot = CircuitParameter::Scalar(value);
            }
            for block in &mut instruction.regions {
                block.instructions = std::mem::take(&mut block.instructions)
                    .into_vec()
                    .into_iter()
                    .rev()
                    .map(|item| self.invert_instruction(item))
                    .collect::<Result<Vec<_>, _>>()?
                    .into_boxed_slice();
            }
            return Ok(instruction);
        }
        if instruction.condition.is_some() || !instruction.regions.is_empty() {
            return Err(StaticCircuitError::Unsupported(
                "adjoint structured circuit instruction".to_owned(),
            ));
        }
        if matches!(instruction.opcode.as_str(), "phase_oracle" | "xor_oracle") {
            return Ok(instruction);
        }
        if matches!(
            instruction.opcode.as_str(),
            "linear_function" | "linear_function_inverse"
        ) {
            let opcode = if instruction.opcode == "linear_function" {
                "linear_function_inverse"
            } else {
                "linear_function"
            }
            .to_owned();
            return Ok(StaticInstruction {
                opcode,
                ..instruction
            });
        }
        let operation = QuantumOpcode::parse_name(&instruction.opcode).ok_or_else(|| {
            StaticCircuitError::Unsupported(format!(
                "adjoint circuit instruction {:?}",
                instruction.opcode
            ))
        })?;
        let opcode = operation
            .exact_adjoint()
            .ok_or_else(|| {
                StaticCircuitError::Unsupported(format!(
                    "adjoint circuit instruction {:?}",
                    instruction.opcode
                ))
            })?
            .name()
            .to_owned();
        let parameters = if operation == QuantumOpcode::PauliEvolution {
            let [hamiltonian, CircuitParameter::Scalar(time)] = instruction.parameters.as_ref()
            else {
                return Err(StaticCircuitError::Missing(
                    "adjoint Pauli evolution operands".to_owned(),
                ));
            };
            vec![
                hamiltonian.clone(),
                CircuitParameter::Scalar(self.negate_scalar(*time)?),
            ]
            .into_boxed_slice()
        } else if operation.adjoint_negates_parameter() {
            let [CircuitParameter::Scalar(identifier)] = instruction.parameters.as_ref() else {
                return Err(StaticCircuitError::Missing(format!(
                    "adjoint angle for {:?}",
                    instruction.opcode
                )));
            };
            vec![CircuitParameter::Scalar(self.negate_scalar(*identifier)?)].into_boxed_slice()
        } else {
            instruction.parameters
        };
        Ok(StaticInstruction {
            opcode,
            qubits: instruction.qubits,
            parameters,
            controls: instruction.controls,
            condition: instruction.condition,
            regions: instruction.regions,
        })
    }

    fn negate_scalar(
        &mut self,
        identifier: CircuitScalarId,
    ) -> Result<CircuitScalarId, StaticCircuitError> {
        let node = match self.scalar_node(identifier) {
            Some(node) if node.scalar_type() == CircuitScalarType::Real => {
                CircuitScalarNode::Apply {
                    opcode: "classical.neg".to_owned(),
                    result_type: CircuitScalarType::Real,
                    attributes: Box::new([]),
                    operands: vec![identifier].into_boxed_slice(),
                }
            }
            _ => {
                return Err(StaticCircuitError::Missing("real adjoint angle".to_owned()));
            }
        };
        self.intern_scalar(node)
    }

    fn emit_if(
        &mut self,
        operation: &wire::OperationProto,
        operands: &[CircuitValue],
    ) -> Result<OperationFlow, StaticCircuitError> {
        let adjoint_body = string_attribute(operation, "body_mode") == Some("adjoint");
        let condition_value = operands
            .first()
            .ok_or_else(|| StaticCircuitError::Missing("qcore.if condition".to_owned()))?;
        if let Some(condition) = self
            .static_value(condition_value)
            .and_then(|value| value.scalar())
            .map(static_truth)
        {
            return self.emit_static_if(operation, operands, condition, adjoint_body);
        }
        let condition = Self::scalar_value_identifier(condition_value, "qcore.if")?;
        if scalar_depends_on_parameter(condition, &self.scalars)
            && !scalar_depends_on_measurement(condition, &self.scalars)
        {
            let decision = self.host_dispatch_decision(condition)?;
            return self.emit_static_if(operation, operands, decision, adjoint_body);
        }
        if !scalar_depends_on_measurement(condition, &self.scalars) {
            return Err(StaticCircuitError::Unsupported(
                "dynamic qcore.if condition without a device measurement".to_owned(),
            ));
        }
        if !self.controls.is_empty() {
            return Err(StaticCircuitError::Unsupported(
                "measurement control inside a coherent controlled call".to_owned(),
            ));
        }
        let captures = operands.get(1..).unwrap_or_default();
        let mut blocks = Vec::with_capacity(operation.regions.len());
        let mut flows = Vec::with_capacity(operation.regions.len());
        let branch_wire_start = self.next_wire;
        let mut branch_wire_end = branch_wire_start;
        let outer_loops = self.expanded_loops.clone();
        let mut branch_loops = Vec::with_capacity(operation.regions.len());
        for (index, region) in operation.regions.iter().enumerate() {
            self.next_wire = branch_wire_start;
            self.expanded_loops.clone_from(&outer_loops);
            self.measurement_guards.push((condition, index == 0));
            let branch = self.emit_dynamic_if_branch(region, captures, adjoint_body);
            self.measurement_guards.pop();
            let (block, flow) = branch?;
            branch_wire_end = branch_wire_end.max(self.next_wire);
            blocks.push(block);
            flows.push(flow);
            branch_loops.push(self.expanded_loops.clone());
        }
        self.next_wire = branch_wire_end;
        self.expanded_loops = outer_loops;
        if let [when_true, when_false] = branch_loops.as_slice() {
            self.expanded_loops = self.merge_expanded_loops(condition, when_true, when_false)?;
        }
        let returning = flows.iter().filter(|flow| flow.kind == "return").count();
        if returning != 0 && returning != flows.len() {
            let result_type = self
                .result_types
                .last()
                .ok_or_else(|| StaticCircuitError::Missing("function result type".to_owned()))?;
            for (block, flow) in blocks.iter_mut().zip(&flows) {
                if flow.kind != "return" {
                    continue;
                }
                let [value] = flow.values.as_slice() else {
                    return Err(StaticCircuitError::Unsupported(
                        "runtime return value ABI".to_owned(),
                    ));
                };
                block.return_value =
                    Some(retain_result_value(value, result_type).ok_or_else(|| {
                        StaticCircuitError::Unsupported("runtime return value type".to_owned())
                    })?);
            }
        }
        let outcome = self.merge_dynamic_if_flows(condition, captures, &flows, adjoint_body)?;
        if blocks
            .iter()
            .all(|block| block.instructions.is_empty() && block.return_value.is_none())
            && (flows.iter().all(|flow| flow.kind == "yield")
                || flows.iter().all(|flow| flow.kind == "return"))
        {
            return Ok(outcome);
        }
        if self.instruction_count >= MAX_STATIC_CIRCUIT_INSTRUCTIONS {
            return Err(StaticCircuitError::Unsupported(format!(
                "more than {MAX_STATIC_CIRCUIT_INSTRUCTIONS} circuit instructions"
            )));
        }
        self.instruction_count += 1;
        self.instructions.push(StaticInstruction {
            opcode: "if".to_owned(),
            qubits: Box::new([]),
            parameters: Box::new([]),
            controls: Box::new([]),
            condition: Some(condition),
            regions: blocks.into_boxed_slice(),
        });
        Ok(outcome)
    }

    fn merge_expanded_loops(
        &mut self,
        condition: CircuitScalarId,
        when_true: &[ExpandedLoop],
        when_false: &[ExpandedLoop],
    ) -> Result<Vec<ExpandedLoop>, StaticCircuitError> {
        if when_true.len() != when_false.len() {
            return Err(StaticCircuitError::Missing("branch loop scopes".to_owned()));
        }
        when_true
            .iter()
            .zip(when_false)
            .map(|(when_true, when_false)| {
                Ok(ExpandedLoop {
                    running: self.select_scalar(
                        condition,
                        when_true.running,
                        when_false.running,
                    )?,
                    continuing: self.select_scalar(
                        condition,
                        when_true.continuing,
                        when_false.continuing,
                    )?,
                    exit_values: self.merge_dynamic_if_values(
                        condition,
                        &when_true.exit_values,
                        &when_false.exit_values,
                    )?,
                })
            })
            .collect()
    }

    fn merge_dynamic_if_flows(
        &mut self,
        condition: CircuitScalarId,
        captures: &[CircuitValue],
        flows: &[Flow],
        adjoint_body: bool,
    ) -> Result<OperationFlow, StaticCircuitError> {
        if flows.len() != 2
            || flows.iter().any(|flow| {
                !matches!(
                    flow.kind.as_str(),
                    "yield" | "return" | "break" | "continue" | "dynamic_exit"
                )
            })
        {
            return Err(StaticCircuitError::Unsupported(
                "dynamic qcore.if branch terminator".to_owned(),
            ));
        }
        let yielding = flows
            .iter()
            .filter(|flow| flow.kind == "yield")
            .collect::<Vec<_>>();
        if adjoint_body && yielding.len() != 2 {
            return Err(StaticCircuitError::Unsupported(
                "dynamic qcore.if branch terminator".to_owned(),
            ));
        }
        if adjoint_body {
            return Ok(OperationFlow::Values(captures.to_vec()));
        }
        match yielding.as_slice() {
            [when_true, when_false] => self
                .merge_dynamic_if_values(condition, &when_true.values, &when_false.values)
                .map(OperationFlow::Values),
            [flow] => Ok(OperationFlow::Values(flow.values.clone())),
            [] if flows.iter().all(|flow| flow.kind == "return") => self
                .merge_dynamic_if_values(condition, &flows[0].values, &flows[1].values)
                .map(|values| {
                    OperationFlow::Exit(Flow {
                        kind: "return".to_owned(),
                        values,
                    })
                }),
            [] if flows.iter().all(|flow| flow.kind != "return") => Ok(OperationFlow::Exit(Flow {
                kind: "dynamic_exit".to_owned(),
                values: Vec::new(),
            })),
            _ => Err(StaticCircuitError::Unsupported(
                "dynamic qcore.if branch terminator".to_owned(),
            )),
        }
    }

    fn emit_dynamic_if_branch(
        &mut self,
        region: &wire::RegionProto,
        captures: &[CircuitValue],
        adjoint_body: bool,
    ) -> Result<(StaticBlock, Flow), StaticCircuitError> {
        let block = region
            .blocks
            .first()
            .ok_or_else(|| StaticCircuitError::Missing("qcore.if branch".to_owned()))?;
        let start = self.instructions.len();
        let flow = self.emit_block(block, &HashMap::new(), captures)?;
        let instructions = if adjoint_body {
            self.take_adjoint_instructions(start)?
        } else {
            self.instructions.split_off(start)
        };
        Ok((
            StaticBlock {
                instructions: instructions.into_boxed_slice(),
                return_value: None,
            },
            flow,
        ))
    }

    fn merge_dynamic_if_values(
        &mut self,
        condition: CircuitScalarId,
        when_true: &[CircuitValue],
        when_false: &[CircuitValue],
    ) -> Result<Vec<CircuitValue>, StaticCircuitError> {
        if when_true.len() != when_false.len() {
            return Err(dynamic_if_result_error());
        }
        when_true
            .iter()
            .zip(when_false)
            .map(|(when_true, when_false)| {
                self.merge_dynamic_if_value(condition, when_true, when_false)
            })
            .collect()
    }

    fn merge_dynamic_if_value(
        &mut self,
        condition: CircuitScalarId,
        when_true: &CircuitValue,
        when_false: &CircuitValue,
    ) -> Result<CircuitValue, StaticCircuitError> {
        if let Some(CircuitScalarNode::Bool(selected)) = self.scalar_node(condition) {
            return Ok(if *selected { when_true } else { when_false }.clone());
        }
        if when_true == when_false {
            return Ok(when_true.clone());
        }
        match (when_true, when_false) {
            (CircuitValue::Scalar(when_true), CircuitValue::Scalar(when_false)) => {
                let true_type = self
                    .scalar_node(*when_true)
                    .map(CircuitScalarNode::scalar_type);
                let false_type = self
                    .scalar_node(*when_false)
                    .map(CircuitScalarNode::scalar_type);
                if true_type.is_none() || true_type != false_type {
                    return Err(dynamic_if_result_error());
                }
                let identifier = self.intern_scalar(CircuitScalarNode::Apply {
                    opcode: "classical.select".to_owned(),
                    result_type: true_type.expect("checked equal scalar types"),
                    attributes: Box::new([]),
                    operands: vec![condition, *when_true, *when_false].into_boxed_slice(),
                })?;
                Ok(CircuitValue::Scalar(identifier))
            }
            (CircuitValue::Aggregate(when_true), CircuitValue::Aggregate(when_false)) => self
                .merge_dynamic_if_values(condition, when_true, when_false)
                .map(|values| CircuitValue::Aggregate(values.into_boxed_slice())),
            _ => Err(dynamic_if_result_error()),
        }
    }

    fn emit_static_if(
        &mut self,
        operation: &wire::OperationProto,
        operands: &[CircuitValue],
        condition: bool,
        adjoint_body: bool,
    ) -> Result<OperationFlow, StaticCircuitError> {
        let region = operation
            .regions
            .get(usize::from(!condition))
            .and_then(|region| region.blocks.first())
            .ok_or_else(|| StaticCircuitError::Missing("qcore.if branch".to_owned()))?;
        let instruction_start = self.instructions.len();
        let flow = self.emit_block(
            region,
            &HashMap::new(),
            operands.get(1..).unwrap_or_default(),
        )?;
        if flow.kind != "yield" {
            if adjoint_body {
                return Err(StaticCircuitError::Unsupported(format!(
                    "adjoint qcore.if branch terminator {:?}",
                    flow.kind
                )));
            }
            return Ok(OperationFlow::Exit(flow));
        }
        if adjoint_body {
            let inverse = self.take_adjoint_instructions(instruction_start)?;
            self.instructions.extend(inverse);
            return Ok(OperationFlow::Values(
                operands.get(1..).unwrap_or_default().to_vec(),
            ));
        }
        Ok(OperationFlow::Values(flow.values))
    }

    fn take_adjoint_instructions(
        &mut self,
        start: usize,
    ) -> Result<Vec<StaticInstruction>, StaticCircuitError> {
        self.instructions
            .split_off(start)
            .into_iter()
            .rev()
            .map(|instruction| self.invert_instruction(instruction))
            .collect()
    }

    fn emit_static_loop_block(
        &mut self,
        block: &wire::BlockProto,
        arguments: &[CircuitValue],
    ) -> Result<Flow, StaticCircuitError> {
        self.emit_block(block, &HashMap::new(), arguments)
    }

    #[expect(
        clippy::too_many_lines,
        reason = "Keep structured-loop eligibility and exact fallback together"
    )]
    fn emit_for(
        &mut self,
        operation: &wire::OperationProto,
        operands: &[CircuitValue],
    ) -> Result<OperationFlow, StaticCircuitError> {
        let [lower, upper, stride] = self.static_for_bounds(operands)?;
        let trip_count = static_range_len(lower, upper, stride)
            .ok_or_else(|| StaticCircuitError::Missing("non-zero qcore.for step".to_owned()))?;
        if trip_count > MAX_STATIC_LOOP_ITERATIONS {
            return Err(StaticCircuitError::Unsupported(format!(
                "qcore.for expansion of {trip_count} iterations"
            )));
        }
        let reverse = bool_attribute(operation, "reverse").unwrap_or(false);
        let adjoint_body = string_attribute(operation, "body_mode") == Some("adjoint");
        let body = operation
            .regions
            .first()
            .and_then(|region| region.blocks.first())
            .ok_or_else(|| StaticCircuitError::Missing("qcore.for body".to_owned()))?;
        let mut carries = operands.get(3..).unwrap_or_default().to_vec();
        if trip_count > 0
            && self.lowering.capabilities.supports_native_for()
            && operands.get(..3).is_some_and(|bounds| {
                bounds
                    .iter()
                    .all(|value| !circuit_value_depends_on_parameter(value, &self.scalars))
            })
            && (native_for_body_eligible(operation, body)
                || (self
                    .lowering
                    .capabilities
                    .contains(RealizationCapabilities::INDEXED_FOR)
                    && structured_for_body_eligible(operation, body)))
        {
            if !self
                .lowering
                .capabilities
                .contains(RealizationCapabilities::INDEXED_FOR)
            {
                return self.emit_native_for(body, operands, &carries, reverse, adjoint_body);
            }
            // Bound the aggregate cost of copying prefixes for speculative loops.
            let work = self.instruction_count as u128
                + self.scalars.len() as u128
                + self.functions.len() as u128
                + u128::from(self.next_wire);
            self.loop_projection_work = self.loop_projection_work.saturating_add(work);
            if self.loop_projection_work <= MAX_STATIC_CIRCUIT_INSTRUCTIONS as u128 {
                let mut candidate = self.clone();
                if let Ok(flow) =
                    candidate.emit_native_for(body, operands, &carries, reverse, adjoint_body)
                {
                    *self = candidate;
                    return Ok(flow);
                }
            }
        }
        let mut broke = false;
        let mut running = self.intern_scalar(CircuitScalarNode::Bool(true))?;
        for position in 0..trip_count {
            let logical_position = if reverse {
                trip_count - position - 1
            } else {
                position
            };
            let index = static_loop_index(lower, stride, logical_position)?;
            let mut arguments = Vec::with_capacity(carries.len() + 1);
            let index = self.intern_scalar(CircuitScalarNode::Int(index))?;
            arguments.push(CircuitValue::Scalar(index));
            arguments.extend(carries.iter().cloned());
            let instruction_start = self.instructions.len();
            self.expanded_loops.push(ExpandedLoop {
                running,
                continuing: running,
                exit_values: carries.clone(),
            });
            let flow = self.emit_static_loop_block(body, &arguments)?;
            let frame = self
                .expanded_loops
                .pop()
                .ok_or_else(|| StaticCircuitError::Missing("expanded loop scope".to_owned()))?;
            running = frame.running;
            if adjoint_body && !matches!(flow.kind.as_str(), "yield" | "continue") {
                return Err(StaticCircuitError::Unsupported(format!(
                    "adjoint qcore.for body terminator {:?}",
                    flow.kind
                )));
            }
            match flow.kind.as_str() {
                "yield" => {
                    if adjoint_body {
                        let inverse = self.take_adjoint_instructions(instruction_start)?;
                        self.instructions.extend(inverse);
                    } else {
                        carries = self.merge_dynamic_if_values(
                            frame.continuing,
                            &flow.values,
                            &frame.exit_values,
                        )?;
                    }
                }
                "break" | "continue" | "dynamic_exit" => {
                    if adjoint_body {
                        let inverse = self.take_adjoint_instructions(instruction_start)?;
                        self.instructions.extend(inverse);
                    } else {
                        carries = frame.exit_values;
                    }
                    if matches!(
                        self.scalar_node(running),
                        Some(CircuitScalarNode::Bool(false))
                    ) {
                        broke = true;
                        break;
                    }
                }
                kind => {
                    if kind == "return" {
                        return Ok(OperationFlow::Exit(flow));
                    }
                    return Err(StaticCircuitError::Unsupported(format!(
                        "qcore.for body terminator {kind:?}"
                    )));
                }
            }
        }
        if !broke
            && let Some(otherwise) = operation
                .regions
                .get(1)
                .and_then(|region| region.blocks.first())
        {
            if adjoint_body {
                return Ok(OperationFlow::Values(carries));
            }
            let start = self.instructions.len();
            let flow = self.emit_block(otherwise, &HashMap::new(), &carries)?;
            self.guard_instructions(start, running)?;
            if flow.kind != "yield" {
                if flow.kind == "return" {
                    if !matches!(
                        self.scalar_node(running),
                        Some(CircuitScalarNode::Bool(true))
                    ) {
                        return Err(StaticCircuitError::Unsupported(
                            "path-dependent qcore.for else return".to_owned(),
                        ));
                    }
                    return Ok(OperationFlow::Exit(flow));
                }
                return Err(StaticCircuitError::Unsupported(format!(
                    "qcore.for else terminator {:?}",
                    flow.kind
                )));
            }
            carries = self.merge_dynamic_if_values(running, &flow.values, &carries)?;
        }
        Ok(OperationFlow::Values(carries))
    }

    fn emit_while(
        &mut self,
        operation: &wire::OperationProto,
        operands: &[CircuitValue],
    ) -> Result<OperationFlow, StaticCircuitError> {
        let mut carries = operands.to_vec();
        let mut iterations = 0;
        loop {
            let flow = self.emit_static_loop_block(&operation.regions[0].blocks[0], &carries)?;
            let test = flow
                .values
                .first()
                .and_then(|value| self.static_value(value))
                .and_then(|value| value.scalar())
                .map(static_truth);
            carries = flow.values[1..].to_vec();
            match test {
                Some(false) => {
                    let flow =
                        self.emit_static_loop_block(&operation.regions[2].blocks[0], &carries)?;
                    return Ok(if flow.kind == "yield" {
                        OperationFlow::Values(flow.values)
                    } else {
                        OperationFlow::Exit(flow)
                    });
                }
                Some(true) => {}
                None => {
                    let id = Self::scalar_value_identifier(&flow.values[0], "qcore.while")?;
                    if scalar_depends_on_parameter(id, &self.scalars)
                        && !scalar_depends_on_measurement(id, &self.scalars)
                    {
                        return Err(StaticCircuitError::LaunchMaterialization(
                            "qcore.while condition".to_owned(),
                        ));
                    }
                    return Err(StaticCircuitError::Unsupported(
                        "dynamic qcore.while condition requires runtime loop support".to_owned(),
                    ));
                }
            }
            if iterations >= MAX_STATIC_LOOP_ITERATIONS {
                return Err(StaticCircuitError::Unsupported(
                    "qcore.while expansion exceeds iteration budget".to_owned(),
                ));
            }
            iterations += 1;
            let flow = self.emit_static_loop_block(&operation.regions[1].blocks[0], &carries)?;
            match flow.kind.as_str() {
                "yield" | "continue" => carries = flow.values,
                "break" => return Ok(OperationFlow::Values(flow.values)),
                "return" => return Ok(OperationFlow::Exit(flow)),
                kind => {
                    return Err(StaticCircuitError::Unsupported(format!(
                        "qcore.while body terminator {kind:?}"
                    )));
                }
            }
        }
    }

    fn emit_native_for(
        &mut self,
        body: &wire::BlockProto,
        operands: &[CircuitValue],
        carries: &[CircuitValue],
        reverse: bool,
        adjoint_body: bool,
    ) -> Result<OperationFlow, StaticCircuitError> {
        let [lower, upper, stride] = operands
            .get(..3)
            .ok_or_else(|| StaticCircuitError::Missing("native qcore.for bounds".to_owned()))?
        else {
            return Err(StaticCircuitError::Missing(
                "native qcore.for bounds".to_owned(),
            ));
        };
        let bounds = [
            Self::scalar_value_identifier(lower, "qcore.for")?,
            Self::scalar_value_identifier(upper, "qcore.for")?,
            Self::scalar_value_identifier(stride, "qcore.for")?,
        ];
        let mut arguments = Vec::with_capacity(carries.len() + 1);
        let index = if self
            .lowering
            .capabilities
            .contains(RealizationCapabilities::INDEXED_FOR)
        {
            let index = self.next_loop_index;
            self.next_loop_index = index.checked_add(1).ok_or(StaticCircuitError::Overflow)?;
            Some(self.intern_scalar(CircuitScalarNode::LoopIndex(index))?)
        } else {
            None
        };
        arguments.push(CircuitValue::Scalar(index.unwrap_or(bounds[0])));
        arguments.extend(carries.iter().cloned());
        let start = self.instructions.len();
        let start_wire = self.next_wire;
        let start_measurement = self.next_measurement;
        let start_workspace = self.schoolbook_workspace.len();
        let start_scalar = self.scalars.len();
        let flow = self.emit_static_loop_block(body, &arguments)?;
        let scratch_growth = u64::try_from(self.schoolbook_workspace.len() - start_workspace)
            .map_err(|_| StaticCircuitError::Overflow)?;
        let integer_index_arithmetic = self.scalars[start_scalar..].iter().any(|node| {
            matches!(node, CircuitScalarNode::Apply { result_type: CircuitScalarType::Int, operands, .. }
                if operands.iter().any(|operand| scalar_depends_on_loop_index(*operand, &self.scalars)))
        });
        if flow.kind != "yield"
            || flow.values != carries
            || self.next_wire.checked_sub(start_wire) != Some(scratch_growth)
            || self.next_measurement != start_measurement
            || !coherent_instructions(&self.instructions[start..])
            || integer_index_arithmetic
        {
            return Err(StaticCircuitError::Unsupported(
                "native qcore.for body must preserve its circuit carries".to_owned(),
            ));
        }
        let instructions = if adjoint_body {
            self.take_adjoint_instructions(start)?
        } else {
            self.instructions.split_off(start)
        };
        // Require an exactly representable reverse range before retaining it.
        let reversed = self.reverse_for_bounds(&bounds)?;
        let bounds = if reverse { reversed } else { bounds };
        if self.instruction_count >= MAX_STATIC_CIRCUIT_INSTRUCTIONS {
            return Err(StaticCircuitError::Unsupported(format!(
                "more than {MAX_STATIC_CIRCUIT_INSTRUCTIONS} circuit instructions"
            )));
        }
        self.instruction_count += 1;
        self.instructions.push(StaticInstruction {
            opcode: "for".to_owned(),
            qubits: Box::new([]),
            parameters: bounds
                .into_iter()
                .chain(index)
                .map(CircuitParameter::Scalar)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            controls: Box::new([]),
            condition: None,
            regions: vec![StaticBlock {
                instructions: instructions.into_boxed_slice(),
                return_value: None,
            }]
            .into_boxed_slice(),
        });
        Ok(OperationFlow::Values(carries.to_vec()))
    }

    fn reverse_for_bounds(
        &mut self,
        bounds: &[CircuitScalarId; 3],
    ) -> Result<[CircuitScalarId; 3], StaticCircuitError> {
        let values = bounds.map(CircuitValue::Scalar);
        let [lower, upper, stride] = self.static_for_bounds(&values)?;
        let count = static_range_len(lower, upper, stride).ok_or(StaticCircuitError::Overflow)?;
        let last = static_loop_index(lower, stride, count.saturating_sub(1))?;
        let stop = lower
            .checked_sub(stride)
            .ok_or(StaticCircuitError::Overflow)?;
        let reverse_stride = stride.checked_neg().ok_or(StaticCircuitError::Overflow)?;
        Ok([
            self.intern_scalar(CircuitScalarNode::Int(last))?,
            self.intern_scalar(CircuitScalarNode::Int(stop))?,
            self.intern_scalar(CircuitScalarNode::Int(reverse_stride))?,
        ])
    }

    fn static_for_bounds(&self, operands: &[CircuitValue]) -> Result<[i64; 3], StaticCircuitError> {
        let bounds = operands
            .get(..3)
            .ok_or_else(|| StaticCircuitError::Missing("qcore.for bounds".to_owned()))?;
        let mut values = [0_i64; 3];
        for (index, value) in bounds.iter().enumerate() {
            values[index] = self.static_int(value).ok_or_else(|| {
                if circuit_value_depends_on_parameter(value, &self.scalars) {
                    StaticCircuitError::LaunchMaterialization("qcore.for bounds".to_owned())
                } else {
                    StaticCircuitError::Unsupported("non-static qcore.for bounds".to_owned())
                }
            })?;
        }
        Ok(values)
    }

    fn emit_quantum(
        &mut self,
        operation: &wire::OperationProto,
        operands: &[CircuitValue],
    ) -> Result<Vec<CircuitValue>, StaticCircuitError> {
        if let Some(gate) = primitive_gate_spec(&operation.opcode) {
            return self.emit_primitive_gate(operation, operands, gate);
        }
        match operation.opcode.as_str() {
            "quantum.alloc" => self.emit_alloc(operands),
            "quantum.prepare" => self.emit_prepare(operands),
            "quantum.measure" => self.emit_measure(operation, operands),
            "quantum.measure_reset" => self.emit_measure_reset(operation, operands),
            "quantum.measure_x" | "quantum.measure_y" | "quantum.measure_z" => {
                self.emit_basis_measurement(operation, operands)
            }
            "quantum.release" => {
                require_wires(operands.first(), "quantum.release")?;
                Ok(Vec::new())
            }
            "quantum.discard" => {
                require_wires(operands.first(), "quantum.discard")?;
                Ok(vec![CircuitValue::Unit])
            }
            "quantum.unprepare" => self.emit_unprepare(operands),
            "quantum.reset" => {
                let state = require_wires(operands.first(), "quantum.reset")?;
                for wire in state {
                    self.push_instruction("reset", [*wire], [])?;
                }
                Ok(vec![CircuitValue::Wires(state.into())])
            }
            "quantum.global_phase" => self.emit_global_phase(operands),
            "quantum.pauli_evolution" => self.emit_pauli_evolution(operands),
            "quantum.add_into" => self.emit_addition(operands, false),
            "quantum.subtract_into" => self.emit_addition(operands, true),
            "quantum.add_constant"
            | "quantum.subtract_constant"
            | "quantum.mod_add"
            | "quantum.mod_sub"
            | "quantum.mod_add_constant"
            | "quantum.mod_sub_constant"
            | "quantum.mul_add"
            | "quantum.mul_sub"
            | "quantum.mod_mul_add"
            | "quantum.mod_mul_sub"
            | "quantum.mod_mul"
            | "quantum.mod_div"
            | "quantum.less_than_xor" => self.emit_arithmetic_permutation(operation, operands),
            "quantum.mod_pow" | "quantum.mod_pow_inverse" => self.emit_mod_pow(operation, operands),
            "quantum.mod_multi_pow" | "quantum.mod_multi_pow_inverse" => {
                self.emit_mod_multi_pow(operation, operands)
            }
            opcode => Err(StaticCircuitError::Unsupported(format!(
                "quantum operation {opcode:?}"
            ))),
        }
    }

    fn emit_alloc(
        &mut self,
        operands: &[CircuitValue],
    ) -> Result<Vec<CircuitValue>, StaticCircuitError> {
        let width = operands
            .first()
            .and_then(|value| self.static_int(value))
            .ok_or_else(|| {
                StaticCircuitError::Unsupported("non-static allocation width".to_owned())
            })?;
        let width = u64::try_from(width)
            .ok()
            .filter(|width| *width > 0)
            .ok_or_else(|| {
                StaticCircuitError::InvalidSelector(format!("allocation width {width}"))
            })?;
        let end = self
            .next_wire
            .checked_add(width)
            .ok_or(StaticCircuitError::Overflow)?;
        let wires = (self.next_wire..end).collect::<Vec<_>>();
        self.next_wire = end;
        Ok(vec![CircuitValue::Wires(wires.into_boxed_slice())])
    }

    fn emit_measure(
        &mut self,
        operation: &wire::OperationProto,
        operands: &[CircuitValue],
    ) -> Result<Vec<CircuitValue>, StaticCircuitError> {
        let wires = require_wires(operands.first(), "quantum.measure")?;
        let mut measurements = Vec::with_capacity(wires.len());
        for wire in wires {
            measurements.push(self.push_measurement(*wire)?);
        }
        let result_type = operation
            .results
            .first()
            .and_then(|result| result.r#type.as_ref())
            .ok_or_else(|| StaticCircuitError::Missing("quantum.measure result type".to_owned()))?;
        Ok(vec![pack_measurement_values(measurements, result_type)?])
    }

    fn emit_prepare(
        &mut self,
        operands: &[CircuitValue],
    ) -> Result<Vec<CircuitValue>, StaticCircuitError> {
        let value = operands
            .first()
            .ok_or_else(|| StaticCircuitError::Missing("quantum.prepare data".to_owned()))?;
        if let CircuitValue::Scalar(condition) = value
            && scalar_depends_on_measurement(*condition, &self.scalars)
            && self.static_value(value).is_none()
            && self
                .scalar_node(*condition)
                .is_some_and(|node| node.scalar_type() == CircuitScalarType::Bool)
        {
            let wire = self.next_wire;
            self.next_wire = self
                .next_wire
                .checked_add(1)
                .ok_or(StaticCircuitError::Overflow)?;
            let start = self.instructions.len();
            self.push_instruction("x", [wire], [])?;
            let instructions = self.instructions.split_off(start);
            self.instructions.push(StaticInstruction {
                opcode: "if".into(),
                qubits: Box::new([]),
                parameters: Box::new([]),
                controls: Box::new([]),
                condition: Some(*condition),
                regions: vec![
                    StaticBlock {
                        instructions: instructions.into_boxed_slice(),
                        return_value: None,
                    },
                    StaticBlock {
                        instructions: Box::new([]),
                        return_value: None,
                    },
                ]
                .into_boxed_slice(),
            });
            self.instruction_count += 1;
            return Ok(vec![CircuitValue::Wires(vec![wire].into_boxed_slice())]);
        }
        let bits = self.prepare_bits(value)?;
        let width = u64::try_from(bits.len()).map_err(|_| StaticCircuitError::Overflow)?;
        let end = self
            .next_wire
            .checked_add(width)
            .ok_or(StaticCircuitError::Overflow)?;
        let wires = (self.next_wire..end).collect::<Vec<_>>();
        self.next_wire = end;
        for (wire, bit) in wires.iter().zip(bits) {
            if bit {
                self.push_instruction("x", [*wire], [])?;
            }
        }
        Ok(vec![CircuitValue::Wires(wires.into_boxed_slice())])
    }

    fn emit_unprepare(
        &mut self,
        operands: &[CircuitValue],
    ) -> Result<Vec<CircuitValue>, StaticCircuitError> {
        let state = require_wires(operands.first(), "quantum.unprepare")?;
        let bits = operands
            .get(1)
            .and_then(|value| self.literal_bits(value))
            .ok_or_else(|| {
                StaticCircuitError::Unsupported("non-static quantum.unprepare data".to_owned())
            })?;
        if state.len() != bits.len() {
            return Err(StaticCircuitError::Missing(
                "quantum.unprepare width equation".to_owned(),
            ));
        }
        for (wire, bit) in state.iter().zip(bits) {
            if bit {
                self.push_instruction("x", [*wire], [])?;
            }
        }
        Ok(Vec::new())
    }

    fn emit_measure_reset(
        &mut self,
        operation: &wire::OperationProto,
        operands: &[CircuitValue],
    ) -> Result<Vec<CircuitValue>, StaticCircuitError> {
        let state = require_wires(operands.first(), "quantum.measure_reset")?;
        let mut measurements = Vec::with_capacity(state.len());
        for wire in state {
            measurements.push(self.push_measurement(*wire)?);
            self.push_instruction("reset", [*wire], [])?;
        }
        let result_type = operation
            .results
            .first()
            .and_then(|result| result.r#type.as_ref())
            .and_then(|result| strip_staged_type(result))
            .filter(|result| {
                result.kind() == wire::type_proto::Kind::Product && result.arguments.len() == 2
            })
            .and_then(|result| result.arguments.first())
            .ok_or_else(|| {
                StaticCircuitError::Missing("quantum.measure_reset result type".to_owned())
            })?;
        let measurement_value = pack_measurement_values(measurements, result_type)?;
        Ok(vec![CircuitValue::Aggregate(
            vec![measurement_value, CircuitValue::Wires(state.into())].into_boxed_slice(),
        )])
    }

    fn emit_basis_measurement(
        &mut self,
        operation: &wire::OperationProto,
        operands: &[CircuitValue],
    ) -> Result<Vec<CircuitValue>, StaticCircuitError> {
        let state = require_wires(operands.first(), &operation.opcode)?;
        let arguments = gate_arguments(operation, operands.get(1..).unwrap_or_default())?;
        let value = arguments
            .keywords
            .get("target")
            .copied()
            .or_else(|| arguments.positional.first());
        let selector = match value {
            Some(value) => self.static_int(value).ok_or_else(|| {
                StaticCircuitError::Unsupported("non-static measurement target".to_owned())
            })?,
            None => 0,
        };
        let wire = wire_at(state, selector, "measurement target")?;
        if self
            .lowering
            .capabilities
            .contains(RealizationCapabilities::PAULI_MEASUREMENT)
        {
            let measurement = self.push_measurement(wire)?;
            operation.opcode.trim_start_matches("quantum.").clone_into(
                &mut self
                    .instructions
                    .last_mut()
                    .expect("measurement was inserted")
                    .opcode,
            );
            return Ok(vec![CircuitValue::Aggregate(
                vec![measurement, CircuitValue::Wires(state.into())].into_boxed_slice(),
            )]);
        }
        match operation.opcode.as_str() {
            "quantum.measure_x" => self.push_instruction("h", [wire], [])?,
            "quantum.measure_y" => {
                self.push_instruction("sdg", [wire], [])?;
                self.push_instruction("h", [wire], [])?;
            }
            "quantum.measure_z" => {}
            _ => return Err(StaticCircuitError::Missing("measurement opcode".to_owned())),
        }
        let measurement = self.push_measurement(wire)?;
        match operation.opcode.as_str() {
            "quantum.measure_x" => self.push_instruction("h", [wire], [])?,
            "quantum.measure_y" => {
                self.push_instruction("h", [wire], [])?;
                self.push_instruction("s", [wire], [])?;
            }
            "quantum.measure_z" => {}
            _ => return Err(StaticCircuitError::Missing("measurement opcode".to_owned())),
        }
        Ok(vec![CircuitValue::Aggregate(
            vec![measurement, CircuitValue::Wires(state.into())].into_boxed_slice(),
        )])
    }

    fn push_measurement(&mut self, wire: u64) -> Result<CircuitValue, StaticCircuitError> {
        let index = self.next_measurement;
        self.next_measurement = index.checked_add(1).ok_or(StaticCircuitError::Overflow)?;
        self.push_instruction(
            "measure",
            [wire],
            [CircuitParameter::MeasurementIndex(index)],
        )?;
        Ok(CircuitValue::Scalar(
            self.intern_scalar(CircuitScalarNode::Measurement(index))?,
        ))
    }

    fn emit_primitive_gate(
        &mut self,
        operation: &wire::OperationProto,
        operands: &[CircuitValue],
        gate: PrimitiveGateSpec,
    ) -> Result<Vec<CircuitValue>, StaticCircuitError> {
        let state = require_wires(operands.first(), &operation.opcode)?;
        let arguments = gate_arguments(operation, operands.get(1..).unwrap_or_default())?;
        let angle = if gate.has_angle() {
            Some(
                arguments
                    .positional
                    .first()
                    .and_then(scalar_identifier)
                    .ok_or_else(|| {
                        StaticCircuitError::Missing(format!("angle for {:?}", operation.opcode))
                    })?,
            )
        } else {
            None
        };
        let selector_offset = usize::from(gate.has_angle());
        let has_selector = gate.selectors().iter().enumerate().any(|(position, name)| {
            arguments.keywords.contains_key(name)
                || arguments
                    .positional
                    .get(position + selector_offset)
                    .is_some()
        });
        if gate.broadcasts_when_unselected() && !has_selector {
            for wire in state {
                let parameters = angle.map(CircuitParameter::Scalar);
                self.push_instruction(gate.name(), [*wire], parameters)?;
            }
            return Ok(vec![CircuitValue::Wires(state.into())]);
        }
        let mut selected = Vec::with_capacity(gate.selectors().len());
        for (position, name) in gate.selectors().iter().enumerate() {
            let value = arguments
                .keywords
                .get(name)
                .copied()
                .or_else(|| arguments.positional.get(position + selector_offset));
            let selector = match value {
                Some(value) => self.static_int(value).ok_or_else(|| {
                    StaticCircuitError::Unsupported(format!(
                        "non-static selector {name:?} for {:?}",
                        operation.opcode
                    ))
                })?,
                None => gate
                    .selector_default(position)
                    .and_then(|value| i64::try_from(value).ok())
                    .ok_or(StaticCircuitError::Overflow)?,
            };
            selected.push(wire_at(state, selector, name)?);
        }
        require_distinct_selectors(&operation.opcode, &selected)?;
        let parameters = angle.map(CircuitParameter::Scalar);
        self.push_instruction(gate.name(), selected, parameters)?;
        Ok(vec![CircuitValue::Wires(state.into())])
    }

    fn emit_global_phase(
        &mut self,
        operands: &[CircuitValue],
    ) -> Result<Vec<CircuitValue>, StaticCircuitError> {
        let state = require_wires(operands.first(), "quantum.global_phase")?;
        let angle = operands
            .get(1)
            .and_then(scalar_identifier)
            .ok_or_else(|| StaticCircuitError::Missing("global phase angle".to_owned()))?;
        self.push_instruction("global_phase", [], [CircuitParameter::Scalar(angle)])?;
        Ok(vec![CircuitValue::Wires(state.into())])
    }

    fn emit_addition(
        &mut self,
        operands: &[CircuitValue],
        inverse: bool,
    ) -> Result<Vec<CircuitValue>, StaticCircuitError> {
        let state = require_wires(operands.first(), "quantum.add_into")?;
        let plan = plan_addition(state.len() as u64, self.lowering)?;
        self.realizations
            .insert(ConfiguredRealization::exact(plan.realization()));
        let plan = if inverse { plan.into_adjoint() } else { plan };
        for step in plan.into_steps() {
            let select = |indices: &[u64]| {
                indices
                    .iter()
                    .map(|index| {
                        let index =
                            usize::try_from(*index).map_err(|_| StaticCircuitError::Overflow)?;
                        state
                            .get(index)
                            .copied()
                            .ok_or(StaticCircuitError::Realization(
                                crate::RealizationError::AdditionWidth,
                            ))
                    })
                    .collect::<Result<Vec<_>, StaticCircuitError>>()
            };
            let qubits = select(step.indices())?;
            let controls = select(step.controls())?;
            let parameters = match step.angle() {
                Some(angle) => vec![CircuitParameter::Scalar(
                    self.intern_scalar(CircuitScalarNode::RealBits(angle.to_bits()))?,
                )],
                None => Vec::new(),
            };
            let saved = if step.inherits_controls() {
                None
            } else {
                Some(std::mem::take(&mut self.controls))
            };
            let outer = self.controls.len();
            self.controls.extend(controls);
            let result = self.push_instruction(step.gate().as_str(), qubits, parameters);
            self.controls.truncate(outer);
            if let Some(saved) = saved {
                self.controls = saved;
            }
            result?;
        }
        Ok(vec![CircuitValue::Wires(state.into())])
    }

    #[expect(
        clippy::too_many_lines,
        reason = "Keep arithmetic realization selection and emission in one dispatch"
    )]
    fn emit_arithmetic_permutation(
        &mut self,
        operation: &wire::OperationProto,
        operands: &[CircuitValue],
    ) -> Result<Vec<CircuitValue>, StaticCircuitError> {
        let state = require_wires(operands.first(), &operation.opcode)?;
        let constants = operands
            .iter()
            .skip(1)
            .map(|value| {
                self.static_int(value).ok_or_else(|| {
                    StaticCircuitError::Unsupported(format!(
                        "{} requires specialized Const integers",
                        operation.opcode
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if matches!(
            operation.opcode.as_str(),
            "quantum.mod_mul_add" | "quantum.mod_mul_sub"
        ) {
            let [modulus] = constants.as_slice() else {
                return Err(StaticCircuitError::Realization(
                    crate::RealizationError::ArithmeticParameters,
                ));
            };
            let width = self
                .coherent_partitions
                .get(state)
                .and_then(|widths| match widths.as_ref() {
                    [accumulator, left, right] if accumulator == left && left == right => {
                        Some(*accumulator)
                    }
                    _ => None,
                })
                .or_else(|| (state.len() % 3 == 0).then_some(state.len() / 3))
                .filter(|width| *width > 0)
                .ok_or(StaticCircuitError::Realization(
                    crate::RealizationError::ArithmeticWidth,
                ))?;
            if self.try_deferred_modular(
                state,
                ModularRequest::MultiplyAdd {
                    width: width as u64,
                    modulus: *modulus,
                    inverse: operation.opcode == "quantum.mod_mul_sub",
                    capabilities: self.lowering.capabilities,
                },
            )? {
                return Ok(vec![CircuitValue::Wires(state.into())]);
            }
            let plan = plan_schoolbook_mod_mul_add(
                u64::try_from(width).map_err(|_| StaticCircuitError::Overflow)?,
                *modulus,
                operation.opcode == "quantum.mod_mul_sub",
                u64::try_from(self.controls.len()).map_err(|_| StaticCircuitError::Overflow)?,
                self.lowering.capabilities,
            )?;
            self.emit_modular_gate_plan(state, plan)?;
            self.realizations.insert(ConfiguredRealization::exact(
                Realization::ModMulAddSchoolbookRipple,
            ));
            return Ok(vec![CircuitValue::Wires(state.into())]);
        }
        if matches!(
            operation.opcode.as_str(),
            "quantum.mod_mul" | "quantum.mod_div"
        ) && self.lowering.select(RealizationFamily::ModMul).kind()
            == Realization::ModMulSchoolbookRipple
        {
            let [factor, modulus] = constants.as_slice() else {
                return Err(StaticCircuitError::Realization(
                    crate::RealizationError::ArithmeticParameters,
                ));
            };
            if self.try_deferred_modular(
                state,
                ModularRequest::Multiply {
                    width: state.len() as u64,
                    factor: *factor,
                    modulus: *modulus,
                    inverse: operation.opcode == "quantum.mod_div",
                    capabilities: self.lowering.capabilities,
                },
            )? {
                return Ok(vec![CircuitValue::Wires(state.into())]);
            }
            let plan = plan_schoolbook_mod_mul(
                u64::try_from(state.len()).map_err(|_| StaticCircuitError::Overflow)?,
                *factor,
                *modulus,
                operation.opcode == "quantum.mod_div",
                u64::try_from(self.controls.len()).map_err(|_| StaticCircuitError::Overflow)?,
                self.lowering.capabilities,
            )?;
            self.emit_modular_gate_plan(state, plan)?;
            self.realizations.insert(ConfiguredRealization::exact(
                Realization::ModMulSchoolbookRipple,
            ));
            return Ok(vec![CircuitValue::Wires(state.into())]);
        }
        let arithmetic = ArithmeticOperation::try_from(operation.opcode.as_str())?;
        if self
            .lowering
            .capabilities
            .contains(RealizationCapabilities::COHERENT_REGIONS)
        {
            let recipe =
                crate::lowering::arithmetic_permutation::CheckedArithmeticPermutation::new(
                    arithmetic,
                    state.len() as u64,
                    &constants,
                )?;
            self.realizations.insert(ConfiguredRealization::exact(
                Realization::ArithmeticPortable,
            ));
            let definition = self
                .arithmetic_definitions
                .entry(recipe.clone())
                .or_insert_with(|| CircuitDefinition::arithmetic(state.len() as u64, recipe))
                .clone();
            self.push_instruction(
                "deferred",
                state.iter().copied(),
                [CircuitParameter::Definition(definition)],
            )?;
            return Ok(vec![CircuitValue::Wires(state.into())]);
        }
        let plan = plan_arithmetic_permutation(arithmetic, state.len() as u64, &constants, false)?;
        self.realizations
            .insert(ConfiguredRealization::exact(plan.realization()));
        for step in plan.into_steps() {
            let target = state
                .get(usize::try_from(step.indices()[0]).map_err(|_| StaticCircuitError::Overflow)?)
                .copied()
                .ok_or(StaticCircuitError::Realization(
                    crate::RealizationError::ArithmeticWidth,
                ))?;
            let controls = step
                .controls()
                .iter()
                .map(|index| {
                    state
                        .get(usize::try_from(*index).map_err(|_| StaticCircuitError::Overflow)?)
                        .copied()
                        .ok_or(StaticCircuitError::Realization(
                            crate::RealizationError::ArithmeticWidth,
                        ))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let outer = self.controls.len();
            self.controls.extend(controls);
            debug_assert!(step.angle().is_none());
            debug_assert!(step.inherits_controls());
            let emitted = self.push_instruction(step.gate().as_str(), [target], []);
            self.controls.truncate(outer);
            emitted?;
        }
        Ok(vec![CircuitValue::Wires(state.into())])
    }

    fn emit_mod_pow(
        &mut self,
        operation: &wire::OperationProto,
        operands: &[CircuitValue],
    ) -> Result<Vec<CircuitValue>, StaticCircuitError> {
        let state = require_wires(operands.first(), &operation.opcode)?;
        let constants = operands
            .get(1..)
            .unwrap_or_default()
            .iter()
            .map(|value| {
                self.static_int(value).ok_or_else(|| {
                    StaticCircuitError::Unsupported(format!(
                        "{} requires specialized Const integers",
                        operation.opcode
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let [base, modulus] = constants.as_slice() else {
            return Err(StaticCircuitError::Realization(
                crate::RealizationError::ArithmeticParameters,
            ));
        };
        let (target_width, exponent_width) = self
            .coherent_partitions
            .get(state)
            .and_then(|widths| match widths.as_ref() {
                [target, exponent] => Some((*target, *exponent)),
                _ => None,
            })
            .map_or_else(|| mod_pow_widths(operation), Ok)?;
        if state.len() != target_width + exponent_width {
            return Err(StaticCircuitError::Realization(
                crate::RealizationError::ArithmeticWidth,
            ));
        }
        let realization = self.lowering.select(RealizationFamily::ModPow);
        if realization.kind() != Realization::ModPowNative
            && self.try_deferred_modular(
                state,
                ModularRequest::Power {
                    target_width: target_width as u64,
                    exponent_width: exponent_width as u64,
                    base: *base,
                    modulus: *modulus,
                    inverse: operation.opcode == "quantum.mod_pow_inverse",
                    config: self.lowering,
                },
            )?
        {
            self.realizations.insert(realization);
            return Ok(vec![CircuitValue::Wires(state.into())]);
        }
        let plan = plan_mod_pow(
            u64::try_from(target_width).map_err(|_| StaticCircuitError::Overflow)?,
            u64::try_from(exponent_width).map_err(|_| StaticCircuitError::Overflow)?,
            *base,
            *modulus,
            operation.opcode == "quantum.mod_pow_inverse",
            u64::try_from(self.controls.len()).map_err(|_| StaticCircuitError::Overflow)?,
            self.lowering,
        )?;
        match plan {
            ModularRealizationPlan::PreserveNative(_) => {
                let target_width = self.intern_scalar(CircuitScalarNode::Int(
                    i64::try_from(target_width).map_err(|_| StaticCircuitError::Overflow)?,
                ))?;
                let exponent_width = self.intern_scalar(CircuitScalarNode::Int(
                    i64::try_from(exponent_width).map_err(|_| StaticCircuitError::Overflow)?,
                ))?;
                let base = self.intern_scalar(CircuitScalarNode::Int(*base))?;
                let modulus = self.intern_scalar(CircuitScalarNode::Int(*modulus))?;
                self.push_instruction(
                    operation.opcode.trim_start_matches("quantum."),
                    state.iter().copied(),
                    [
                        CircuitParameter::Scalar(target_width),
                        CircuitParameter::Scalar(exponent_width),
                        CircuitParameter::Scalar(base),
                        CircuitParameter::Scalar(modulus),
                    ],
                )?;
            }
            ModularRealizationPlan::Gates(plan) => {
                self.emit_modular_gate_plan(state, plan)?;
            }
        }
        self.realizations.insert(realization);
        Ok(vec![CircuitValue::Wires(state.into())])
    }

    fn emit_mod_multi_pow(
        &mut self,
        operation: &wire::OperationProto,
        operands: &[CircuitValue],
    ) -> Result<Vec<CircuitValue>, StaticCircuitError> {
        let state = require_wires(operands.first(), &operation.opcode)?;
        let bases = match operands.get(1) {
            Some(CircuitValue::Aggregate(values)) => values
                .iter()
                .map(|value| {
                    self.static_int(value).ok_or_else(|| {
                        StaticCircuitError::Unsupported(format!(
                            "{} requires specialized Const bases",
                            operation.opcode
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
            _ => {
                return Err(StaticCircuitError::Unsupported(format!(
                    "{} requires a specialized Const base vector",
                    operation.opcode
                )));
            }
        };
        let constants = operands
            .get(2..)
            .unwrap_or_default()
            .iter()
            .map(|value| {
                self.static_int(value).ok_or_else(|| {
                    StaticCircuitError::Unsupported(format!(
                        "{} requires specialized Const integers",
                        operation.opcode
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let [base_width, modulus, coordinate_width] = constants.as_slice() else {
            return Err(StaticCircuitError::Realization(
                crate::RealizationError::ArithmeticParameters,
            ));
        };
        let (coordinate_total, target_width) = self
            .coherent_partitions
            .get(state)
            .and_then(|widths| match widths.as_ref() {
                [coordinates, target] => Some((*coordinates, *target)),
                _ => None,
            })
            .map_or_else(|| mod_multi_pow_widths(operation), Ok)?;
        let coordinate_width = usize::try_from(*coordinate_width)
            .ok()
            .filter(|width| *width > 0)
            .ok_or(StaticCircuitError::Realization(
                crate::RealizationError::ArithmeticWidth,
            ))?;
        if state.len() != coordinate_total + target_width
            || bases
                .len()
                .checked_mul(coordinate_width)
                .ok_or(StaticCircuitError::Overflow)?
                != coordinate_total
        {
            return Err(StaticCircuitError::Realization(
                crate::RealizationError::ArithmeticWidth,
            ));
        }
        if self.try_deferred_modular(
            state,
            ModularRequest::MultiPower {
                coordinate_count: bases.len() as u64,
                coordinate_width: coordinate_width as u64,
                target_width: target_width as u64,
                base_width: u64::try_from(*base_width).map_err(|_| {
                    StaticCircuitError::Realization(RealizationError::ArithmeticWidth)
                })?,
                bases: bases.clone().into_boxed_slice(),
                modulus: *modulus,
                inverse: operation.opcode == "quantum.mod_multi_pow_inverse",
                config: self.lowering,
            },
        )? {
            return Ok(vec![CircuitValue::Wires(state.into())]);
        }
        let plan = plan_mod_multi_pow(
            u64::try_from(bases.len()).map_err(|_| StaticCircuitError::Overflow)?,
            u64::try_from(coordinate_width).map_err(|_| StaticCircuitError::Overflow)?,
            u64::try_from(target_width).map_err(|_| StaticCircuitError::Overflow)?,
            u64::try_from(*base_width).map_err(|_| {
                StaticCircuitError::Realization(crate::RealizationError::ArithmeticWidth)
            })?,
            &bases,
            *modulus,
            operation.opcode == "quantum.mod_multi_pow_inverse",
            u64::try_from(self.controls.len()).map_err(|_| StaticCircuitError::Overflow)?,
            self.lowering,
        )?;
        let realization = ConfiguredRealization::exact(plan.realization());
        self.emit_modular_gate_plan(state, plan)?;
        self.realizations.insert(realization);
        Ok(vec![CircuitValue::Wires(state.into())])
    }

    fn try_deferred_modular(
        &mut self,
        state: &[u64],
        request: ModularRequest,
    ) -> Result<bool, StaticCircuitError> {
        if !self
            .lowering
            .capabilities
            .contains(RealizationCapabilities::COHERENT_REGIONS)
            || !self.controls.is_empty()
        {
            return Ok(false);
        }
        if let Some((definition, scratch_width)) = self.modular_definitions.get(&request).cloned() {
            let scratch = self.schoolbook_workspace(scratch_width)?;
            self.push_instruction(
                "deferred",
                state.iter().chain(&scratch).copied(),
                [CircuitParameter::Definition(definition)],
            )?;
            return Ok(true);
        }
        let checked = request.clone().check()?;
        if !checked.is_unitary() {
            return Ok(false);
        }
        let plan = checked.metadata();
        if plan.state_width() != state.len() as u64 {
            return Err(StaticCircuitError::Realization(
                RealizationError::ArithmeticWidth,
            ));
        }
        self.realizations
            .insert(ConfiguredRealization::exact(plan.realization()));
        self.realizations.extend(
            plan.child_realizations()
                .iter()
                .copied()
                .map(ConfiguredRealization::exact),
        );
        let scratch = self.schoolbook_workspace(
            usize::try_from(plan.scratch_width()).map_err(|_| StaticCircuitError::Overflow)?,
        )?;
        let wires = state.iter().chain(&scratch).copied().collect::<Vec<_>>();
        let definition = CircuitDefinition::modular(wires.len() as u64, checked);
        self.modular_definitions
            .insert(request, (definition.clone(), scratch.len()));
        self.push_instruction(
            "deferred",
            wires,
            [CircuitParameter::Definition(definition)],
        )?;
        Ok(true)
    }

    fn emit_modular_gate_plan(
        &mut self,
        state: &[u64],
        plan: ModularGatePlan,
    ) -> Result<(), StaticCircuitError> {
        if plan.state_width()
            != u64::try_from(state.len()).map_err(|_| StaticCircuitError::Overflow)?
            || plan.control_width()
                != u64::try_from(self.controls.len()).map_err(|_| StaticCircuitError::Overflow)?
        {
            return Err(StaticCircuitError::Realization(
                RealizationError::ArithmeticWidth,
            ));
        }
        self.realizations.extend(
            plan.child_realizations()
                .iter()
                .copied()
                .map(ConfiguredRealization::exact),
        );
        let scratch_width =
            usize::try_from(plan.scratch_width()).map_err(|_| StaticCircuitError::Overflow)?;
        let scratch = self.schoolbook_workspace(scratch_width)?;
        let saved_controls = std::mem::take(&mut self.controls);
        let mapped = state
            .iter()
            .chain(&saved_controls)
            .chain(&scratch)
            .copied()
            .collect::<Vec<_>>();
        let emitted = (|| {
            for step in plan.into_steps() {
                let wires = step
                    .wires()
                    .iter()
                    .map(|index| {
                        mapped
                            .get(
                                usize::try_from(*index)
                                    .map_err(|_| StaticCircuitError::Overflow)?,
                            )
                            .copied()
                            .ok_or(StaticCircuitError::Realization(
                                RealizationError::ArithmeticWidth,
                            ))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                self.controls = step
                    .controls()
                    .iter()
                    .map(|index| {
                        mapped
                            .get(
                                usize::try_from(*index)
                                    .map_err(|_| StaticCircuitError::Overflow)?,
                            )
                            .copied()
                            .ok_or(StaticCircuitError::Realization(
                                RealizationError::ArithmeticWidth,
                            ))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                self.push_instruction(step.gate().as_str(), wires, [])?;
                self.controls.clear();
            }
            Ok(())
        })();
        self.controls = saved_controls;
        emitted
    }

    fn reserve_clean_wires(&mut self, count: usize) -> Result<Vec<u64>, StaticCircuitError> {
        let count = u64::try_from(count).map_err(|_| StaticCircuitError::Overflow)?;
        let end = self
            .next_wire
            .checked_add(count)
            .ok_or(StaticCircuitError::Overflow)?;
        let result = (self.next_wire..end).collect();
        self.next_wire = end;
        Ok(result)
    }

    fn schoolbook_workspace(&mut self, count: usize) -> Result<Vec<u64>, StaticCircuitError> {
        if self.schoolbook_workspace.len() < count {
            let missing = count - self.schoolbook_workspace.len();
            let added = self.reserve_clean_wires(missing)?;
            self.schoolbook_workspace.extend(added);
        }
        Ok(self.schoolbook_workspace[..count].to_vec())
    }

    fn emit_pauli_evolution(
        &mut self,
        operands: &[CircuitValue],
    ) -> Result<Vec<CircuitValue>, StaticCircuitError> {
        let state = require_wires(operands.first(), "quantum.pauli_evolution")?;
        let Some(CircuitValue::Hamiltonian(hamiltonian)) = operands.get(1) else {
            return Err(StaticCircuitError::Missing(
                "Hamiltonian for quantum.pauli_evolution".to_owned(),
            ));
        };
        let time = operands
            .get(2)
            .and_then(scalar_identifier)
            .ok_or_else(|| StaticCircuitError::Missing("Pauli evolution time".to_owned()))?;
        let width = u64::try_from(state.len()).map_err(|_| StaticCircuitError::Overflow)?;
        let choice = self.lowering.select(RealizationFamily::PauliEvolution);
        self.realizations.insert(choice);
        if choice.kind() == Realization::PauliNative {
            validate_pauli_evolution(hamiltonian, width)
                .map_err(|error| static_pauli_evolution_error(&error))?;
            self.push_instruction(
                "pauli_evolution",
                state.iter().copied(),
                [
                    CircuitParameter::Hamiltonian(hamiltonian.clone()),
                    CircuitParameter::Scalar(time),
                ],
            )?;
            return Ok(vec![CircuitValue::Wires(state.into())]);
        }
        let schedule = plan_pauli_evolution_with_capabilities(
            hamiltonian,
            width,
            if choice.kind() == Realization::PauliRzz {
                RealizationCapabilities::RZZ
            } else {
                RealizationCapabilities::EMPTY
            },
        )
        .map_err(|error| static_pauli_evolution_error(&error))?;
        for step in schedule {
            match step.gate() {
                PauliEvolutionGate::GlobalPhase => {
                    let angle = self.scale_scalar(step.scale(), time)?;
                    self.push_instruction(
                        step.gate().as_str(),
                        [],
                        [CircuitParameter::Scalar(angle)],
                    )?;
                }
                PauliEvolutionGate::Rz => {
                    let [index] = step.indices() else {
                        return Err(StaticCircuitError::Missing(
                            "native Pauli evolution Rz selector".to_owned(),
                        ));
                    };
                    let angle = self.scale_scalar(step.scale(), time)?;
                    self.push_instruction(
                        step.gate().as_str(),
                        [pauli_wire(state, *index)?],
                        [CircuitParameter::Scalar(angle)],
                    )?;
                }
                PauliEvolutionGate::Rzz => {
                    let [left, right] = step.indices() else {
                        return Err(StaticCircuitError::Missing(
                            "native Pauli evolution Rzz selectors".to_owned(),
                        ));
                    };
                    let angle = self.scale_scalar(step.scale(), time)?;
                    self.push_instruction(
                        step.gate().as_str(),
                        [pauli_wire(state, *left)?, pauli_wire(state, *right)?],
                        [CircuitParameter::Scalar(angle)],
                    )?;
                }
                _ => {
                    let selected = step
                        .indices()
                        .iter()
                        .map(|index| pauli_wire(state, *index))
                        .collect::<Result<Vec<_>, _>>()?;
                    self.push_instruction(step.gate().as_str(), selected, [])?;
                }
            }
        }
        Ok(vec![CircuitValue::Wires(state.into())])
    }

    fn scale_scalar(
        &mut self,
        scale: f64,
        identifier: CircuitScalarId,
    ) -> Result<CircuitScalarId, StaticCircuitError> {
        if let Some(CircuitScalarNode::RealBits(value)) = self.scalar_node(identifier) {
            return self.intern_scalar(CircuitScalarNode::RealBits(
                (scale * f64::from_bits(*value)).to_bits(),
            ));
        }
        let scale = self.intern_scalar(CircuitScalarNode::RealBits(scale.to_bits()))?;
        self.intern_scalar(CircuitScalarNode::Apply {
            opcode: "classical.mul".to_owned(),
            result_type: CircuitScalarType::Real,
            attributes: Box::new([]),
            operands: vec![scale, identifier].into_boxed_slice(),
        })
    }

    fn push_instruction(
        &mut self,
        opcode: &str,
        qubits: impl IntoIterator<Item = u64>,
        parameters: impl IntoIterator<Item = CircuitParameter>,
    ) -> Result<(), StaticCircuitError> {
        if self.instruction_count >= MAX_STATIC_CIRCUIT_INSTRUCTIONS {
            return Err(StaticCircuitError::Unsupported(format!(
                "more than {MAX_STATIC_CIRCUIT_INSTRUCTIONS} circuit instructions"
            )));
        }
        self.instruction_count += 1;
        let qubits = qubits.into_iter().collect::<Vec<_>>();
        if qubits.iter().any(|wire| *wire >= self.next_wire) {
            return Err(StaticCircuitError::InvalidSelector(format!(
                "{opcode} references an unallocated wire"
            )));
        }
        if self.controls.iter().any(|wire| *wire >= self.next_wire) {
            return Err(StaticCircuitError::InvalidSelector(format!(
                "{opcode} references an unallocated control wire"
            )));
        }
        let mut distinct_controls = self.controls.clone();
        distinct_controls.sort_unstable();
        distinct_controls.dedup();
        if distinct_controls.len() != self.controls.len()
            || self.controls.iter().any(|wire| qubits.contains(wire))
        {
            return Err(StaticCircuitError::InvalidSelector(format!(
                "{opcode} has aliased coherent controls"
            )));
        }
        let mut instruction = StaticInstruction {
            opcode: opcode.to_owned(),
            qubits: qubits.into_boxed_slice(),
            parameters: parameters
                .into_iter()
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            controls: self.controls.clone().into_boxed_slice(),
            condition: None,
            regions: Box::new([]),
        };
        // A finite choice of launch angles is expressible as quantum control;
        // arbitrary measured arithmetic still requires dynamic-rotation support.
        if matches!(opcode, "rx" | "ry" | "rz" | "p" | "rzz" | "cp")
            && let [CircuitParameter::Scalar(root)] = instruction.parameters.as_ref()
            && let Some(CircuitScalarNode::Apply {
                opcode, operands, ..
            }) = self.scalar_node(*root)
            && opcode == "classical.select"
            && let [condition, left, right] = operands.as_ref()
            && scalar_depends_on_measurement(*condition, &self.scalars)
            && [left, right].iter().all(|id| {
                matches!(
                    self.scalar_node(**id),
                    Some(CircuitScalarNode::RealBits(_) | CircuitScalarNode::Parameter { .. })
                )
            })
        {
            if self.instruction_count > MAX_STATIC_CIRCUIT_INSTRUCTIONS - 2 {
                return Err(StaticCircuitError::Unsupported(
                    "finite angle branch instruction budget".to_owned(),
                ));
            }
            let (condition, left, right) = (*condition, *left, *right);
            self.instruction_count += 2;
            let mut when_true = instruction.clone();
            when_true.parameters = vec![CircuitParameter::Scalar(left)].into_boxed_slice();
            let mut when_false = instruction;
            when_false.parameters = vec![CircuitParameter::Scalar(right)].into_boxed_slice();
            instruction = StaticInstruction {
                opcode: "if".to_owned(),
                qubits: Box::new([]),
                parameters: Box::new([]),
                controls: Box::new([]),
                condition: Some(condition),
                regions: vec![
                    StaticBlock {
                        instructions: vec![when_true].into_boxed_slice(),
                        return_value: None,
                    },
                    StaticBlock {
                        instructions: vec![when_false].into_boxed_slice(),
                        return_value: None,
                    },
                ]
                .into_boxed_slice(),
            };
        }
        self.instructions.push(instruction);
        Ok(())
    }
}

fn static_loop_index(
    lower: i64,
    stride: i64,
    logical_position: u128,
) -> Result<i64, StaticCircuitError> {
    let offset = i128::try_from(logical_position)
        .ok()
        .and_then(|position| position.checked_mul(i128::from(stride)))
        .ok_or(StaticCircuitError::Overflow)?;
    i128::from(lower)
        .checked_add(offset)
        .and_then(|value| i64::try_from(value).ok())
        .ok_or(StaticCircuitError::Overflow)
}

fn named_call_label(symbol: &str) -> Option<String> {
    const PREFIX: &str = "__qamomile_box_";
    let local = symbol.rsplit('.').next()?;
    let encoded = local.strip_prefix(PREFIX)?.split_once("__")?.0;
    if encoded.is_empty() || encoded.len() % 2 != 0 {
        return None;
    }
    let bytes = encoded
        .as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| {
            let digits = std::str::from_utf8(pair).ok()?;
            u8::from_str_radix(digits, 16).ok()
        })
        .collect::<Option<Vec<_>>>()?;
    String::from_utf8(bytes)
        .ok()
        .filter(|label| !label.is_empty())
}

fn collect_instruction_wires(
    instructions: &[StaticInstruction],
    output: &mut std::collections::BTreeSet<u64>,
) {
    for instruction in instructions {
        output.extend(instruction.qubits.iter().copied());
        output.extend(instruction.controls.iter().copied());
        for region in &instruction.regions {
            collect_instruction_wires(&region.instructions, output);
        }
    }
}

fn contains_structured_loop(instructions: &[StaticInstruction]) -> bool {
    instructions.iter().any(|item| {
        item.opcode == "for"
            || item
                .regions
                .iter()
                .any(|block| contains_structured_loop(&block.instructions))
    })
}

fn distribute_controls(
    instructions: &mut [StaticInstruction],
    controls: &[u64],
) -> Result<(), StaticCircuitError> {
    for instruction in instructions {
        if instruction.opcode == "for" {
            for block in &mut instruction.regions {
                distribute_controls(&mut block.instructions, controls)?;
            }
        } else {
            if controls.iter().any(|wire| {
                instruction.qubits.contains(wire) || instruction.controls.contains(wire)
            }) {
                return Err(StaticCircuitError::InvalidSelector(
                    "aliased coherent region controls".to_owned(),
                ));
            }
            let mut combined = controls.to_vec();
            combined.extend_from_slice(&instruction.controls);
            instruction.controls = combined.into_boxed_slice();
        }
    }
    Ok(())
}

fn coherent_instructions(instructions: &[StaticInstruction]) -> bool {
    instructions.iter().all(|instruction| {
        instruction.condition.is_none()
            && (instruction.opcode.starts_with("box:")
                || instruction.opcode.starts_with("inverse:box:")
                || matches!(
                    instruction.opcode.as_str(),
                    "for"
                        | "phase_oracle"
                        | "xor_oracle"
                        | "linear_function"
                        | "linear_function_inverse"
                )
                || QuantumOpcode::parse_name(&instruction.opcode)
                    .is_some_and(|opcode| opcode.exact_adjoint().is_some()))
            && instruction.regions.iter().all(|block| {
                block.return_value.is_none() && coherent_instructions(&block.instructions)
            })
    })
}

fn structured_for_body_eligible(operation: &wire::OperationProto, body: &wire::BlockProto) -> bool {
    let identity_else = operation
        .regions
        .get(1)
        .and_then(|region| region.blocks.first())
        .is_none_or(|otherwise| {
            otherwise.operations.is_empty()
                && otherwise.terminator.as_ref().is_some_and(|term| {
                    term.kind == "yield"
                        && term.operands
                            == otherwise
                                .arguments
                                .iter()
                                .map(|arg| arg.id)
                                .collect::<Vec<_>>()
                })
        });
    identity_else
        && body
            .terminator
            .as_ref()
            .is_some_and(|term| term.kind == "yield")
        && body.operations.iter().all(|item| {
            (item.opcode == "qcore.for"
                && item
                    .regions
                    .first()
                    .and_then(|region| region.blocks.first())
                    .is_some_and(|body| structured_for_body_eligible(item, body)))
                || (item.regions.is_empty()
                    && (matches!(
                        item.opcode.as_str(),
                        "qcore.constant"
                            | "qcore.stage_lift"
                            | "qcore.reshape"
                            | "qcore.tuple"
                            | "qcore.unpack"
                            | "qcore.call"
                            | "qcore.control_call"
                            | "qcore.adjoint_call"
                            | "qcore.select_call"
                            | "qcore.phase_by"
                            | "qcore.xor_by"
                            | "qcore.permute_by"
                            | "classical.add"
                            | "classical.sub"
                            | "classical.mul"
                            | "classical.truediv"
                            | "classical.neg"
                    ) || QuantumOpcode::parse_name(item.opcode.trim_start_matches("quantum."))
                        .is_some_and(|opcode| opcode.exact_adjoint().is_some())))
        })
}

fn native_for_body_eligible(operation: &wire::OperationProto, body: &wire::BlockProto) -> bool {
    const COHERENT_OPCODES: &[&str] = &[
        "qcore.constant",
        "qcore.stage_lift",
        "quantum.ccx",
        "quantum.cp",
        "quantum.cx",
        "quantum.cz",
        "quantum.global_phase",
        "quantum.h",
        "quantum.p",
        "quantum.rx",
        "quantum.ry",
        "quantum.rz",
        "quantum.rzz",
        "quantum.s",
        "quantum.sdg",
        "quantum.swap",
        "quantum.t",
        "quantum.tdg",
        "quantum.x",
        "quantum.y",
        "quantum.z",
    ];
    let Some(index) = body.arguments.first().map(|value| value.id) else {
        return false;
    };
    let dependent = std::collections::HashSet::from([index]);
    for item in &body.operations {
        if !COHERENT_OPCODES.contains(&item.opcode.as_str())
            || !item.regions.is_empty()
            || item
                .operands
                .iter()
                .any(|operand| dependent.contains(operand))
        {
            return false;
        }
    }
    let identity_else = operation
        .regions
        .get(1)
        .and_then(|region| region.blocks.first())
        .is_none_or(|otherwise| {
            otherwise.operations.is_empty()
                && otherwise.terminator.as_ref().is_some_and(|terminator| {
                    terminator.kind == "yield"
                        && terminator.operands
                            == otherwise
                                .arguments
                                .iter()
                                .map(|argument| argument.id)
                                .collect::<Vec<_>>()
                })
        });
    body.terminator
        .as_ref()
        .is_some_and(|terminator| terminator.kind == "yield")
        && identity_else
}

struct GateArguments<'a> {
    positional: &'a [CircuitValue],
    keywords: HashMap<&'a str, &'a CircuitValue>,
}

fn gate_arguments<'a>(
    operation: &'a wire::OperationProto,
    operands: &'a [CircuitValue],
) -> Result<GateArguments<'a>, StaticCircuitError> {
    let names = string_tuple_attribute(operation, "keywords")
        .ok_or_else(|| StaticCircuitError::Missing("gate keywords".to_owned()))?;
    let positional_count = operands
        .len()
        .checked_sub(names.len())
        .ok_or_else(|| StaticCircuitError::Missing("gate argument ABI".to_owned()))?;
    let keywords = names
        .into_iter()
        .zip(&operands[positional_count..])
        .collect();
    Ok(GateArguments {
        positional: &operands[..positional_count],
        keywords,
    })
}

fn require_wires<'a>(
    value: Option<&'a CircuitValue>,
    opcode: &str,
) -> Result<&'a [u64], StaticCircuitError> {
    match value {
        Some(CircuitValue::Wires(wires)) => Ok(wires),
        _ => Err(StaticCircuitError::Missing(format!(
            "logical wires for {opcode:?}"
        ))),
    }
}

fn scalar_identifier(value: &CircuitValue) -> Option<CircuitScalarId> {
    match value {
        CircuitValue::Scalar(identifier) => Some(*identifier),
        _ => None,
    }
}

fn circuit_value_depends_on_parameter(value: &CircuitValue, scalars: &[CircuitScalarNode]) -> bool {
    match value {
        CircuitValue::Scalar(identifier) => scalar_depends_on_parameter(*identifier, scalars),
        CircuitValue::Parameter { .. } => true,
        CircuitValue::Aggregate(values) => values
            .iter()
            .any(|value| circuit_value_depends_on_parameter(value, scalars)),
        CircuitValue::Hamiltonian(_)
        | CircuitValue::Portable(_)
        | CircuitValue::Wires(_)
        | CircuitValue::Unit => false,
    }
}

fn scalar_depends_on_loop_index(
    identifier: CircuitScalarId,
    scalars: &[CircuitScalarNode],
) -> bool {
    match scalars.get(identifier.index() as usize) {
        Some(CircuitScalarNode::LoopIndex(_)) => true,
        Some(CircuitScalarNode::Apply { operands, .. }) => operands
            .iter()
            .any(|operand| scalar_depends_on_loop_index(*operand, scalars)),
        _ => false,
    }
}

fn scalar_depends_on_measurement(
    identifier: CircuitScalarId,
    scalars: &[CircuitScalarNode],
) -> bool {
    match usize::try_from(identifier.index())
        .ok()
        .and_then(|index| scalars.get(index))
    {
        Some(CircuitScalarNode::Measurement(_) | CircuitScalarNode::CallValue { .. }) => true,
        Some(CircuitScalarNode::Apply { operands, .. }) => operands
            .iter()
            .any(|operand| scalar_depends_on_measurement(*operand, scalars)),
        _ => false,
    }
}

fn scalar_depends_on_parameter(identifier: CircuitScalarId, scalars: &[CircuitScalarNode]) -> bool {
    match usize::try_from(identifier.index())
        .ok()
        .and_then(|index| scalars.get(index))
    {
        Some(CircuitScalarNode::Parameter { .. }) => true,
        Some(CircuitScalarNode::Apply { operands, .. }) => operands
            .iter()
            .any(|operand| scalar_depends_on_parameter(*operand, scalars)),
        _ => false,
    }
}

fn compact_dispatch_predicate(
    root: CircuitScalarId,
    scalars: &[CircuitScalarNode],
) -> Result<CircuitDispatchPredicate, StaticCircuitError> {
    let mut compact = Vec::new();
    let compact_root = copy_scalar_node(root, scalars, &mut compact, &mut HashMap::new())?;
    if compact
        .get(compact_root.index() as usize)
        .map(CircuitScalarNode::scalar_type)
        != Some(CircuitScalarType::Bool)
    {
        return Err(StaticCircuitError::Missing(
            "Boolean launch predicate root".to_owned(),
        ));
    }
    Ok(CircuitDispatchPredicate {
        root: compact_root,
        scalars: compact.into_boxed_slice(),
    })
}

pub(crate) fn compact_circuit_scalars(
    instructions: &mut [StaticInstruction],
    result: Option<&mut StaticCircuitValue>,
    scalars: &[CircuitScalarNode],
    checks: &mut [CircuitScalarCheck],
) -> Result<Box<[CircuitScalarNode]>, StaticCircuitError> {
    let mut compact = Vec::new();
    let mut remap = HashMap::new();
    for instruction in instructions {
        remap_instruction_scalars(instruction, scalars, &mut compact, &mut remap)?;
    }
    if let Some(result) = result {
        remap_result_scalars(result, scalars, &mut compact, &mut remap)?;
    }
    for check in checks {
        check.root = copy_scalar_node(check.root, scalars, &mut compact, &mut remap)?;
    }
    Ok(compact.into_boxed_slice())
}

fn compact_retained_result_scalars(
    result: &mut StaticCircuitValue,
    source: &[CircuitScalarNode],
) -> Result<Box<[CircuitScalarNode]>, StaticCircuitError> {
    let mut pending = Vec::new();
    collect_result_scalar_roots(result, &mut pending);
    let mut reachable = vec![false; source.len()];
    while let Some(identifier) = pending.pop() {
        let index =
            usize::try_from(identifier.index()).map_err(|_| StaticCircuitError::Overflow)?;
        let node = source
            .get(index)
            .ok_or_else(|| StaticCircuitError::Missing("result scalar".to_owned()))?;
        if std::mem::replace(&mut reachable[index], true) {
            continue;
        }
        if let CircuitScalarNode::Apply { operands, .. } = node {
            pending.extend(operands.iter().copied());
        }
    }

    let mut remap = vec![None; source.len()];
    let mut compact = Vec::new();
    for (source_index, node) in source.iter().enumerate() {
        if !reachable[source_index] {
            continue;
        }
        let copied = match node {
            CircuitScalarNode::Apply {
                opcode,
                result_type,
                attributes,
                operands,
            } => CircuitScalarNode::Apply {
                opcode: opcode.clone(),
                result_type: *result_type,
                attributes: attributes.clone(),
                operands: operands
                    .iter()
                    .map(|operand| {
                        remap
                            .get(operand.index() as usize)
                            .copied()
                            .flatten()
                            .ok_or_else(|| {
                                StaticCircuitError::Missing(
                                    "topological result scalar operand".to_owned(),
                                )
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()?
                    .into_boxed_slice(),
            },
            other => other.clone(),
        };
        let identifier = CircuitScalarId(
            u32::try_from(compact.len()).map_err(|_| StaticCircuitError::Overflow)?,
        );
        remap[source_index] = Some(identifier);
        compact.push(copied);
    }
    remap_result_scalar_roots(result, &remap)?;
    Ok(compact.into_boxed_slice())
}

fn collect_result_scalar_roots(result: &StaticCircuitValue, roots: &mut Vec<CircuitScalarId>) {
    match result {
        StaticCircuitValue::Scalar(identifier) => roots.push(*identifier),
        StaticCircuitValue::Aggregate(values) => {
            for value in values {
                collect_result_scalar_roots(value, roots);
            }
        }
        StaticCircuitValue::Record(fields) => {
            for (_, value) in fields {
                collect_result_scalar_roots(value, roots);
            }
        }
        StaticCircuitValue::Unit
        | StaticCircuitValue::Portable(_)
        | StaticCircuitValue::Wires(_) => {}
    }
}

fn remap_result_scalar_roots(
    result: &mut StaticCircuitValue,
    remap: &[Option<CircuitScalarId>],
) -> Result<(), StaticCircuitError> {
    match result {
        StaticCircuitValue::Scalar(identifier) => {
            *identifier = remap
                .get(identifier.index() as usize)
                .copied()
                .flatten()
                .ok_or_else(|| StaticCircuitError::Missing("result scalar root".to_owned()))?;
        }
        StaticCircuitValue::Aggregate(values) => {
            for value in values {
                remap_result_scalar_roots(value, remap)?;
            }
        }
        StaticCircuitValue::Record(fields) => {
            for (_, value) in fields {
                remap_result_scalar_roots(value, remap)?;
            }
        }
        StaticCircuitValue::Unit
        | StaticCircuitValue::Portable(_)
        | StaticCircuitValue::Wires(_) => {}
    }
    Ok(())
}

fn remap_result_scalars(
    result: &mut StaticCircuitValue,
    source: &[CircuitScalarNode],
    target: &mut Vec<CircuitScalarNode>,
    remap: &mut HashMap<CircuitScalarId, CircuitScalarId>,
) -> Result<(), StaticCircuitError> {
    match result {
        StaticCircuitValue::Scalar(identifier) => {
            *identifier = copy_scalar_node(*identifier, source, target, remap)?;
        }
        StaticCircuitValue::Aggregate(values) => {
            for value in values {
                remap_result_scalars(value, source, target, remap)?;
            }
        }
        StaticCircuitValue::Record(fields) => {
            for (_, value) in fields {
                remap_result_scalars(value, source, target, remap)?;
            }
        }
        StaticCircuitValue::Unit
        | StaticCircuitValue::Portable(_)
        | StaticCircuitValue::Wires(_) => {}
    }
    Ok(())
}

fn remap_instruction_scalars(
    instruction: &mut StaticInstruction,
    source: &[CircuitScalarNode],
    target: &mut Vec<CircuitScalarNode>,
    remap: &mut HashMap<CircuitScalarId, CircuitScalarId>,
) -> Result<(), StaticCircuitError> {
    for parameter in &mut instruction.parameters {
        match parameter {
            CircuitParameter::Scalar(identifier) => {
                *identifier = copy_scalar_node(*identifier, source, target, remap)?;
            }
            CircuitParameter::MeasurementIndex(_)
            | CircuitParameter::CallResult { .. }
            | CircuitParameter::Hamiltonian(_)
            | CircuitParameter::BooleanPlan(_)
            | CircuitParameter::Definition(_) => {}
        }
    }
    if let Some(condition) = &mut instruction.condition {
        *condition = copy_scalar_node(*condition, source, target, remap)?;
    }
    for region in &mut instruction.regions {
        if let Some(result) = &mut region.return_value {
            remap_result_scalars(result, source, target, remap)?;
        }
        for nested in &mut region.instructions {
            remap_instruction_scalars(nested, source, target, remap)?;
        }
    }
    Ok(())
}

fn copy_scalar_node(
    identifier: CircuitScalarId,
    source: &[CircuitScalarNode],
    target: &mut Vec<CircuitScalarNode>,
    remap: &mut HashMap<CircuitScalarId, CircuitScalarId>,
) -> Result<CircuitScalarId, StaticCircuitError> {
    if let Some(mapped) = remap.get(&identifier) {
        return Ok(*mapped);
    }
    let node = source
        .get(identifier.index() as usize)
        .ok_or_else(|| StaticCircuitError::Missing("circuit scalar".to_owned()))?;
    let copied = match node {
        CircuitScalarNode::Apply {
            opcode,
            result_type,
            attributes,
            operands,
        } => CircuitScalarNode::Apply {
            opcode: opcode.clone(),
            result_type: *result_type,
            attributes: attributes.clone(),
            operands: operands
                .iter()
                .map(|operand| copy_scalar_node(*operand, source, target, remap))
                .collect::<Result<Vec<_>, _>>()?
                .into_boxed_slice(),
        },
        other => other.clone(),
    };
    let index = u32::try_from(target.len()).map_err(|_| StaticCircuitError::Overflow)?;
    let mapped = CircuitScalarId(index);
    target.push(copied);
    remap.insert(identifier, mapped);
    Ok(mapped)
}

fn mapping_entries(literal: &wire::LiteralProto) -> Option<&[wire::literal_proto::Entry]> {
    let wire::literal_proto::Value::MappingValue(mapping) = literal.value.as_ref()? else {
        return None;
    };
    Some(&mapping.entries)
}

fn mapping_entry<'a>(
    operation: &'a wire::OperationProto,
    attribute_name: &str,
    key: &str,
) -> Option<&'a wire::LiteralProto> {
    mapping_entries(attribute(operation, attribute_name)?)?
        .iter()
        .find(|entry| entry.key == key)?
        .value
        .as_ref()
}

fn mapping_string_attribute<'a>(
    operation: &'a wire::OperationProto,
    attribute_name: &str,
    key: &str,
) -> Option<&'a str> {
    literal_string(mapping_entry(operation, attribute_name, key)?)
}

fn mapping_string_tuple_attribute<'a>(
    operation: &'a wire::OperationProto,
    attribute_name: &str,
    key: &str,
) -> Option<Vec<&'a str>> {
    tuple_values(mapping_entry(operation, attribute_name, key)?)?
        .iter()
        .map(literal_string)
        .collect()
}

fn adjoint_original_operation(
    operation: &wire::OperationProto,
    original: &str,
) -> Result<wire::OperationProto, StaticCircuitError> {
    let entries =
        mapping_entries(attribute(operation, "original_attributes").ok_or_else(|| {
            StaticCircuitError::Missing("adjoint original attributes".to_owned())
        })?)
        .ok_or_else(|| StaticCircuitError::Missing("adjoint original attributes".to_owned()))?;
    Ok(wire::OperationProto {
        opcode: original.to_owned(),
        operands: operation.operands.clone(),
        results: operation.results.clone(),
        attributes: entries
            .iter()
            .map(|entry| wire::AttributeProto {
                name: entry.key.clone(),
                value: entry.value.clone(),
            })
            .collect(),
        span: operation.span.clone(),
        ..Default::default()
    })
}

fn tuple_values(literal: &wire::LiteralProto) -> Option<&[wire::LiteralProto]> {
    let wire::literal_proto::Value::TupleValue(sequence) = literal.value.as_ref()? else {
        return None;
    };
    Some(&sequence.values)
}

fn literal_string(literal: &wire::LiteralProto) -> Option<&str> {
    let wire::literal_proto::Value::StringValue(value) = literal.value.as_ref()? else {
        return None;
    };
    Some(value)
}

fn static_pauli_evolution_error(error: &PauliEvolutionError) -> StaticCircuitError {
    match error {
        PauliEvolutionError::NonHermitian => StaticCircuitError::NonHermitianPauliEvolution,
        PauliEvolutionError::NonCommuting => StaticCircuitError::NonCommutingPauliEvolution,
        PauliEvolutionError::Width {
            observable_width,
            state_width,
        } => StaticCircuitError::InvalidSelector(format!(
            "Hamiltonian width {observable_width} exceeds state width {state_width}"
        )),
        PauliEvolutionError::Observable(_)
        | PauliEvolutionError::CommutationBudget
        | PauliEvolutionError::ScheduleBudget => StaticCircuitError::Unsupported(error.to_string()),
    }
}

fn pauli_wire(state: &[u64], index: u64) -> Result<u64, StaticCircuitError> {
    let position = usize::try_from(index).map_err(|_| StaticCircuitError::Overflow)?;
    state.get(position).copied().ok_or_else(|| {
        StaticCircuitError::InvalidSelector(format!("Pauli index {index} is outside state width"))
    })
}

fn flatten_coherent_wires(
    value: &CircuitValue,
    output: &mut Vec<u64>,
) -> Result<(), StaticCircuitError> {
    match value {
        CircuitValue::Wires(wires) => output.extend(wires.iter().copied()),
        CircuitValue::Aggregate(values) => {
            for value in values {
                flatten_coherent_wires(value, output)?;
            }
        }
        _ => {
            return Err(StaticCircuitError::Missing(
                "coherent qcore.reshape operand".to_owned(),
            ));
        }
    }
    Ok(())
}

fn pack_coherent_wires(
    source: &wire::TypeProto,
    wires: &[u64],
    position: &mut usize,
) -> Result<CircuitValue, StaticCircuitError> {
    let mut types = TypeArena::default();
    let scope = types.intern_scope("static-circuit");
    let source = types.resolve_wire_type(source, &scope).map_err(|error| {
        StaticCircuitError::Unsupported(format!("coherent circuit type: {error}"))
    })?;
    pack_coherent_type(&types, source, wires, position)
}

fn pack_coherent_type(
    types: &TypeArena,
    source: TypeId,
    wires: &[u64],
    position: &mut usize,
) -> Result<CircuitValue, StaticCircuitError> {
    let current = types.strip_stage(source);
    match types.get(current) {
        Some(SemanticType::Quantum(payload)) => {
            let width = types
                .coherent_payload_width(*payload)
                .map_err(|_| StaticCircuitError::Overflow)?
                .ok_or_else(|| {
                    StaticCircuitError::Unsupported(
                        "qcore.reshape target has no fixed coherent width".to_owned(),
                    )
                })?;
            let width = usize::try_from(width).map_err(|_| StaticCircuitError::Overflow)?;
            let end = position
                .checked_add(width)
                .ok_or(StaticCircuitError::Overflow)?;
            let selected = wires.get(*position..end).ok_or_else(|| {
                StaticCircuitError::Missing("qcore.reshape source width".to_owned())
            })?;
            *position = end;
            Ok(CircuitValue::Wires(selected.to_vec().into_boxed_slice()))
        }
        Some(SemanticType::Product(elements)) => {
            let values = elements
                .iter()
                .map(|element| pack_coherent_type(types, *element, wires, position))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(CircuitValue::Aggregate(values.into_boxed_slice()))
        }
        Some(SemanticType::Record { fields, .. }) => {
            let values = fields
                .iter()
                .map(|(_, field_type)| pack_coherent_type(types, *field_type, wires, position))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(CircuitValue::Aggregate(values.into_boxed_slice()))
        }
        _ => Err(StaticCircuitError::Unsupported(format!(
            "qcore.reshape target type {:?}",
            types.get(current)
        ))),
    }
}

pub(crate) fn require_distinct_selectors<T: Clone + Ord>(
    opcode: &str,
    selected: &[T],
) -> Result<(), StaticCircuitError> {
    let mut distinct = selected.to_vec();
    distinct.sort_unstable();
    distinct.dedup();
    if distinct.len() != selected.len() {
        return Err(StaticCircuitError::InvalidSelector(format!(
            "{opcode} selectors must be distinct"
        )));
    }
    Ok(())
}

fn collect_measurements(
    value: &CircuitValue,
    scalars: &[CircuitScalarNode],
    output: &mut Vec<u64>,
) -> bool {
    match value {
        CircuitValue::Scalar(identifier) => {
            let Some(CircuitScalarNode::Measurement(index)) = usize::try_from(identifier.index())
                .ok()
                .and_then(|index| scalars.get(index))
            else {
                return false;
            };
            output.push(*index);
        }
        CircuitValue::Aggregate(values) => {
            return values
                .iter()
                .all(|value| collect_measurements(value, scalars, output));
        }
        CircuitValue::Unit => {}
        CircuitValue::Portable(_)
        | CircuitValue::Hamiltonian(_)
        | CircuitValue::Parameter { .. }
        | CircuitValue::Wires(_) => {
            return false;
        }
    }
    true
}

fn pack_measurement_values(
    mut values: Vec<CircuitValue>,
    type_spec: &wire::TypeProto,
) -> Result<CircuitValue, StaticCircuitError> {
    use wire::type_proto::Kind;

    let current = strip_staged_type(type_spec)
        .ok_or_else(|| StaticCircuitError::Missing("measurement result type".to_owned()))?;
    if current.kind() == Kind::Scalar && matches!(current.name.as_str(), "Bit" | "Bool") {
        if values.len() != 1 {
            return Err(StaticCircuitError::Missing(
                "scalar measurement result width".to_owned(),
            ));
        }
        return Ok(values.pop().expect("one checked measurement value"));
    }
    Ok(CircuitValue::Aggregate(values.into_boxed_slice()))
}

fn retain_result_value(
    value: &CircuitValue,
    type_spec: &wire::TypeProto,
) -> Option<StaticCircuitValue> {
    use wire::type_proto::Kind;

    let mut current = type_spec;
    while Kind::try_from(current.kind).ok()? == Kind::Staged {
        current = current.arguments.first()?;
    }
    match value {
        CircuitValue::Unit => Some(StaticCircuitValue::Unit),
        CircuitValue::Portable(value) => Some(StaticCircuitValue::Portable(value.clone())),
        CircuitValue::Scalar(identifier) => Some(StaticCircuitValue::Scalar(*identifier)),
        CircuitValue::Wires(wires) if Kind::try_from(current.kind).ok()? == Kind::Quantum => {
            Some(StaticCircuitValue::Wires(wires.clone()))
        }
        CircuitValue::Aggregate(values) => match Kind::try_from(current.kind).ok()? {
            Kind::Record
                if current.field_names.len() == values.len()
                    && current.arguments.len() == values.len() =>
            {
                Some(StaticCircuitValue::Record(
                    current
                        .field_names
                        .iter()
                        .zip(&current.arguments)
                        .zip(values)
                        .map(|((name, field_type), value)| {
                            Some((name.clone(), retain_result_value(value, field_type)?))
                        })
                        .collect::<Option<Vec<_>>>()?
                        .into_boxed_slice(),
                ))
            }
            Kind::Product if current.arguments.len() == values.len() => {
                Some(StaticCircuitValue::Aggregate(
                    values
                        .iter()
                        .zip(&current.arguments)
                        .map(|(value, element_type)| retain_result_value(value, element_type))
                        .collect::<Option<Vec<_>>>()?
                        .into_boxed_slice(),
                ))
            }
            Kind::Array if current.name == "Array" && !current.dimensions.is_empty() => {
                retain_array_result(values, current.arguments.first()?, current.dimensions.len())
            }
            Kind::Array if current.name == "Bits" && current.dimensions.len() == 1 => {
                // Measured words are scalar-bit rows, but unlike Array their
                // element type is implicit. Retain them when a sibling result
                // (for example a derived success flag) needs postprocessing.
                let element = wire::TypeProto {
                    kind: Kind::Scalar as i32,
                    name: "Bit".to_owned(),
                    ..wire::TypeProto::default()
                };
                retain_array_result(values, &element, 1)
            }
            _ => None,
        },
        CircuitValue::Hamiltonian(_) | CircuitValue::Parameter { .. } | CircuitValue::Wires(_) => {
            None
        }
    }
}

fn collect_runtime_returns<'a>(
    instructions: &'a [StaticInstruction],
    returns: &mut Vec<&'a StaticCircuitValue>,
) {
    for instruction in instructions {
        if instruction.opcode() == "call" {
            continue;
        }
        for region in instruction.regions() {
            if let Some(value) = region.runtime_return_value() {
                returns.push(value);
            }
            collect_runtime_returns(region.instructions(), returns);
        }
    }
}

fn runtime_results_compatible(
    expected: &StaticCircuitValue,
    candidate: &StaticCircuitValue,
    scalars: &[CircuitScalarNode],
) -> bool {
    match (expected, candidate) {
        (StaticCircuitValue::Unit, StaticCircuitValue::Unit) => true,
        (StaticCircuitValue::Portable(expected), StaticCircuitValue::Portable(candidate)) => {
            expected == candidate
        }
        (StaticCircuitValue::Scalar(expected), StaticCircuitValue::Scalar(candidate)) => {
            let scalar_type = |identifier: &CircuitScalarId| {
                usize::try_from(identifier.index())
                    .ok()
                    .and_then(|index| scalars.get(index))
                    .map(CircuitScalarNode::scalar_type)
            };
            scalar_type(expected).is_some() && scalar_type(expected) == scalar_type(candidate)
        }
        (StaticCircuitValue::Wires(expected), StaticCircuitValue::Wires(candidate)) => {
            expected == candidate
        }
        (StaticCircuitValue::Aggregate(expected), StaticCircuitValue::Aggregate(candidate)) => {
            expected.len() == candidate.len()
                && expected.iter().zip(candidate).all(|(expected, candidate)| {
                    runtime_results_compatible(expected, candidate, scalars)
                })
        }
        (StaticCircuitValue::Record(expected), StaticCircuitValue::Record(candidate)) => {
            expected.len() == candidate.len()
                && expected.iter().zip(candidate).all(
                    |((expected_name, expected), (candidate_name, candidate))| {
                        expected_name == candidate_name
                            && runtime_results_compatible(expected, candidate, scalars)
                    },
                )
        }
        _ => false,
    }
}

fn mod_pow_widths(operation: &wire::OperationProto) -> Result<(usize, usize), StaticCircuitError> {
    let source = operation
        .results
        .first()
        .and_then(|result| result.r#type.as_ref())
        .ok_or_else(|| StaticCircuitError::Missing("mod_pow state type".to_owned()))?;
    let mut types = TypeArena::default();
    let scope = types.intern_scope("mod-pow-circuit");
    let source = types.resolve_wire_type(source, &scope).map_err(|error| {
        StaticCircuitError::Unsupported(format!("mod_pow circuit type: {error}"))
    })?;
    let payload = match types.get(types.strip_stage(source)) {
        Some(SemanticType::Quantum(payload)) => *payload,
        _ => {
            return Err(StaticCircuitError::Realization(
                crate::RealizationError::ArithmeticWidth,
            ));
        }
    };
    let Some(SemanticType::Product(elements)) = types.get(types.strip_stage(payload)) else {
        return Err(StaticCircuitError::Realization(
            crate::RealizationError::ArithmeticWidth,
        ));
    };
    let [target, exponent] = elements.as_ref() else {
        return Err(StaticCircuitError::Realization(
            crate::RealizationError::ArithmeticWidth,
        ));
    };
    let width = |element| {
        types
            .coherent_payload_width(element)
            .map_err(|_| StaticCircuitError::Overflow)?
            .and_then(|value| usize::try_from(value).ok())
            .filter(|value| *value > 0)
            .ok_or(StaticCircuitError::Realization(
                crate::RealizationError::ArithmeticWidth,
            ))
    };
    Ok((width(*target)?, width(*exponent)?))
}

fn mod_multi_pow_widths(
    operation: &wire::OperationProto,
) -> Result<(usize, usize), StaticCircuitError> {
    let source = operation
        .results
        .first()
        .and_then(|result| result.r#type.as_ref())
        .ok_or_else(|| StaticCircuitError::Missing("mod_multi_pow state type".to_owned()))?;
    let mut types = TypeArena::default();
    let scope = types.intern_scope("mod-multi-pow-circuit");
    let source = types.resolve_wire_type(source, &scope).map_err(|error| {
        StaticCircuitError::Unsupported(format!("mod_multi_pow circuit type: {error}"))
    })?;
    let payload = match types.get(types.strip_stage(source)) {
        Some(SemanticType::Quantum(payload)) => *payload,
        _ => {
            return Err(StaticCircuitError::Realization(
                crate::RealizationError::ArithmeticWidth,
            ));
        }
    };
    let Some(SemanticType::Product(elements)) = types.get(types.strip_stage(payload)) else {
        return Err(StaticCircuitError::Realization(
            crate::RealizationError::ArithmeticWidth,
        ));
    };
    let [coordinates, target] = elements.as_ref() else {
        return Err(StaticCircuitError::Realization(
            crate::RealizationError::ArithmeticWidth,
        ));
    };
    let width = |element| {
        types
            .coherent_payload_width(element)
            .map_err(|_| StaticCircuitError::Overflow)?
            .and_then(|value| usize::try_from(value).ok())
            .filter(|value| *value > 0)
            .ok_or(StaticCircuitError::Realization(
                crate::RealizationError::ArithmeticWidth,
            ))
    };
    Ok((width(*coordinates)?, width(*target)?))
}

fn retain_array_result(
    values: &[CircuitValue],
    element_type: &wire::TypeProto,
    rank: usize,
) -> Option<StaticCircuitValue> {
    if rank == 0 {
        return None;
    }
    Some(StaticCircuitValue::Aggregate(
        values
            .iter()
            .map(|value| {
                if rank == 1 {
                    retain_result_value(value, element_type)
                } else if let CircuitValue::Aggregate(values) = value {
                    retain_array_result(values, element_type, rank - 1)
                } else {
                    None
                }
            })
            .collect::<Option<Vec<_>>>()?
            .into_boxed_slice(),
    ))
}

/// Validate concrete quantum selectors at direct SDK emission without a circuit projection.
///
/// Work and storage are proportional to the selector count, never the state width.
/// This shares the bounds and alias checks used by static circuit lowering.
///
/// # Errors
/// Returns `InvalidSelector` for negative, out-of-bounds, or repeated indices,
/// or `Overflow` if a host index cannot be represented as a logical wire.
pub fn validate_circuit_selectors(
    opcode: &str,
    width: usize,
    selectors: &[i64],
) -> Result<(), StaticCircuitError> {
    let selected = selectors
        .iter()
        .map(|selector| {
            let index = circuit_wire_index(width, *selector, opcode)?;
            u64::try_from(index).map_err(|_| StaticCircuitError::Overflow)
        })
        .collect::<Result<Vec<_>, _>>()?;
    require_distinct_selectors(opcode, &selected)
}

fn circuit_wire_index(
    width: usize,
    selector: i64,
    label: &str,
) -> Result<usize, StaticCircuitError> {
    let index = usize::try_from(selector).map_err(|_| {
        StaticCircuitError::InvalidSelector(format!("{label} {selector} is negative"))
    })?;
    if index >= width {
        return Err(StaticCircuitError::InvalidSelector(format!(
            "{label} {selector} is outside width {width}"
        )));
    }
    Ok(index)
}

fn wire_at(state: &[u64], selector: i64, label: &str) -> Result<u64, StaticCircuitError> {
    Ok(state[circuit_wire_index(state.len(), selector, label)?])
}

fn circuit_scalar_type(source: &wire::TypeProto) -> Option<CircuitScalarType> {
    use wire::type_proto::Kind;

    let mut current = source;
    while current.kind() == Kind::Staged {
        let [inner] = current.arguments.as_slice() else {
            return None;
        };
        current = inner;
    }
    if current.kind() != Kind::Scalar {
        return None;
    }
    match current.name.as_str() {
        "Bit" | "Bool" => Some(CircuitScalarType::Bool),
        "Int" => Some(CircuitScalarType::Int),
        "Real" => Some(CircuitScalarType::Real),
        _ => None,
    }
}

fn is_complex_scalar_type(source: &wire::TypeProto) -> bool {
    use wire::type_proto::Kind;

    strip_staged_type(source)
        .is_some_and(|current| current.kind() == Kind::Scalar && current.name == "Complex")
}

fn is_top_level_quantum_type(source: &wire::TypeProto) -> bool {
    use wire::type_proto::Kind;

    strip_staged_type(source).is_some_and(|current| current.kind() == Kind::Quantum)
}

fn top_level_quantum_product_arity(source: &wire::TypeProto) -> Option<usize> {
    use wire::type_proto::Kind;

    let quantum = strip_staged_type(source)?;
    if quantum.kind() != Kind::Quantum {
        return None;
    }
    let [payload] = quantum.arguments.as_slice() else {
        return None;
    };
    let product = strip_staged_type(payload)?;
    (product.kind() == Kind::Product).then_some(product.arguments.len())
}

fn product_of_quantum_arity(source: &wire::TypeProto) -> Option<usize> {
    use wire::type_proto::Kind;

    let product = strip_staged_type(source)?;
    (product.kind() == Kind::Product && product.arguments.iter().all(is_top_level_quantum_type))
        .then_some(product.arguments.len())
}

fn parameter_shape(source: &wire::TypeProto) -> Result<Option<ParameterShape>, StaticCircuitError> {
    use wire::dimension_proto::Value;
    use wire::type_proto::Kind;

    let current = strip_staged_type(source)
        .ok_or_else(|| StaticCircuitError::Missing("parameter type".to_owned()))?;
    match current.kind() {
        Kind::Array => {
            let extent = current
                .dimensions
                .first()
                .and_then(|dimension| dimension.value.as_ref())
                .and_then(|dimension| match dimension {
                    Value::Fixed(value) => Some(*value),
                    Value::Symbol(_) => None,
                })
                .ok_or_else(|| {
                    StaticCircuitError::Unsupported(
                        "aggregate circuit parameters require fixed shapes".to_owned(),
                    )
                })?;
            Ok(Some(ParameterShape::Array(extent)))
        }
        Kind::Product => Ok(Some(ParameterShape::Product(
            u64::try_from(current.arguments.len()).map_err(|_| StaticCircuitError::Overflow)?,
        ))),
        Kind::Record => Ok(Some(ParameterShape::Record(
            current.field_names.clone().into_boxed_slice(),
        ))),
        _ => Ok(None),
    }
}

fn strip_staged_type(source: &wire::TypeProto) -> Option<&wire::TypeProto> {
    use wire::type_proto::Kind;

    let mut current = source;
    while current.kind() == Kind::Staged {
        let [inner] = current.arguments.as_slice() else {
            return None;
        };
        current = inner;
    }
    Some(current)
}

fn fixed_uint_width(source: &wire::TypeProto) -> Option<u64> {
    use wire::dimension_proto::Value;
    use wire::type_proto::Kind;

    let current = strip_staged_type(source)?;
    if current.kind() != Kind::Array || current.name != "UInt" {
        return None;
    }
    let [dimension] = current.dimensions.as_slice() else {
        return None;
    };
    match dimension.value.as_ref()? {
        Value::Fixed(width) if *width > 0 => Some(*width),
        Value::Fixed(_) | Value::Symbol(_) => None,
    }
}

fn parameter_component(
    shape: &ParameterShape,
    index: i64,
) -> Result<CircuitParameterPath, StaticCircuitError> {
    let extent = match shape {
        ParameterShape::Array(extent) | ParameterShape::Product(extent) => *extent,
        ParameterShape::Record(fields) => {
            u64::try_from(fields.len()).map_err(|_| StaticCircuitError::Overflow)?
        }
    };
    let extent_usize = usize::try_from(extent).map_err(|_| StaticCircuitError::Overflow)?;
    let normalized =
        normalize_aggregate_index(extent_usize, &BigInt::from(index)).ok_or_else(|| {
            StaticCircuitError::InvalidSelector(format!(
                "parameter index {index} is outside extent {extent}"
            ))
        })?;
    if let ParameterShape::Record(fields) = shape {
        return Ok(CircuitParameterPath::Field(
            fields
                .get(normalized)
                .cloned()
                .ok_or_else(|| StaticCircuitError::Missing("record field name".to_owned()))?,
        ));
    }
    Ok(CircuitParameterPath::Index(
        u64::try_from(normalized).map_err(|_| StaticCircuitError::Overflow)?,
    ))
}

fn aggregate_value_at(
    values: &[CircuitValue],
    index: i64,
    label: &str,
) -> Result<CircuitValue, StaticCircuitError> {
    let normalized =
        normalize_aggregate_index(values.len(), &BigInt::from(index)).ok_or_else(|| {
            StaticCircuitError::InvalidSelector(format!(
                "{label} {index} is outside extent {}",
                values.len()
            ))
        })?;
    values
        .get(normalized)
        .cloned()
        .ok_or(StaticCircuitError::Overflow)
}

fn dynamic_if_result_error() -> StaticCircuitError {
    StaticCircuitError::Unsupported(
        "dynamic qcore.if branches return incompatible values".to_owned(),
    )
}

fn display_type(source: &wire::TypeProto) -> String {
    let current = strip_staged_type(source).unwrap_or(source);
    if current.name.is_empty() {
        format!("{:?}", current.kind())
    } else {
        current.name.clone()
    }
}

fn attribute<'a>(
    operation: &'a wire::OperationProto,
    name: &str,
) -> Option<&'a wire::LiteralProto> {
    operation
        .attributes
        .iter()
        .find(|attribute| attribute.name == name)
        .and_then(|attribute| attribute.value.as_ref())
}

fn circuit_scalar_attributes(
    operation: &wire::OperationProto,
) -> Result<Box<[CircuitScalarAttribute]>, StaticCircuitError> {
    use wire::literal_proto::Value;

    operation
        .attributes
        .iter()
        .map(|attribute| {
            let literal = attribute.value.as_ref().ok_or_else(|| {
                StaticCircuitError::Missing(format!(
                    "scalar attribute {:?} on {:?}",
                    attribute.name, operation.opcode
                ))
            })?;
            let value = match literal.value.as_ref() {
                Some(Value::IntValue(value)) => CircuitScalarAttributeValue::Int(*value),
                Some(Value::StringValue(value)) => {
                    CircuitScalarAttributeValue::String(value.clone())
                }
                Some(Value::TupleValue(sequence)) => {
                    let values = sequence
                        .values
                        .iter()
                        .map(|literal| match literal.value.as_ref() {
                            Some(Value::StringValue(value)) => Ok(value.clone()),
                            _ => Err(StaticCircuitError::Unsupported(format!(
                                "non-string tuple scalar attribute {:?} on {:?}",
                                attribute.name, operation.opcode
                            ))),
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    CircuitScalarAttributeValue::Strings(values.into_boxed_slice())
                }
                _ => {
                    return Err(StaticCircuitError::Unsupported(format!(
                        "scalar attribute literal {:?} on {:?}",
                        attribute.name, operation.opcode
                    )));
                }
            };
            Ok(CircuitScalarAttribute {
                name: attribute.name.clone(),
                value,
            })
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Vec::into_boxed_slice)
}

fn string_attribute<'a>(operation: &'a wire::OperationProto, name: &str) -> Option<&'a str> {
    use wire::literal_proto::Value;

    match attribute(operation, name)?.value.as_ref()? {
        Value::StringValue(value) => Some(value),
        _ => None,
    }
}

fn bool_attribute(operation: &wire::OperationProto, name: &str) -> Option<bool> {
    use wire::literal_proto::Value;

    match attribute(operation, name)?.value.as_ref()? {
        Value::BoolValue(value) => Some(*value),
        _ => None,
    }
}

fn int_attribute(operation: &wire::OperationProto, name: &str) -> Option<i64> {
    use wire::literal_proto::Value;

    match attribute(operation, name)?.value.as_ref()? {
        Value::IntValue(value) => Some(*value),
        _ => None,
    }
}

fn string_tuple_attribute<'a>(
    operation: &'a wire::OperationProto,
    name: &str,
) -> Option<Vec<&'a str>> {
    use wire::literal_proto::Value;

    let Value::TupleValue(sequence) = attribute(operation, name)?.value.as_ref()? else {
        return None;
    };
    sequence
        .values
        .iter()
        .map(|value| match value.value.as_ref()? {
            Value::StringValue(value) => Some(value.as_str()),
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_emission_selectors_share_bounds_and_alias_checks() {
        assert!(validate_circuit_selectors("quantum.cx", 2, &[0, 1]).is_ok());
        assert!(validate_circuit_selectors("quantum.x", usize::MAX, &[0]).is_ok());
        for selected in [&[-1][..], &[2], &[i64::MIN], &[i64::MAX], &[1, 1]] {
            assert!(matches!(
                validate_circuit_selectors("quantum.cx", 2, selected),
                Err(StaticCircuitError::InvalidSelector(_))
            ));
        }
    }

    #[test]
    fn retained_parameter_names_are_unique_in_scalar_arena_order() {
        let parameter = |name: &str, path| CircuitScalarNode::Parameter {
            name: name.to_owned(),
            path,
            scalar_type: CircuitScalarType::Real,
        };
        let circuit = StaticCircuit::test_new(
            0,
            0,
            vec![],
            vec![
                parameter("theta", Box::new([])),
                CircuitScalarNode::RealBits(0),
                parameter(
                    "values",
                    vec![CircuitParameterPath::Index(0)].into_boxed_slice(),
                ),
                parameter(
                    "theta",
                    vec![CircuitParameterPath::Index(1)].into_boxed_slice(),
                ),
            ],
            vec![],
        );

        assert_eq!(circuit.retained_parameter_names(), vec!["theta", "values"]);
    }

    #[test]
    fn dispatch_predicate_parameter_names_are_unique_in_scalar_arena_order() {
        let parameter = |name: &str, path| CircuitScalarNode::Parameter {
            name: name.to_owned(),
            path,
            scalar_type: CircuitScalarType::Int,
        };
        let predicate = CircuitDispatchPredicate::from_parts(
            CircuitScalarId::from_index(3),
            vec![
                parameter(
                    "values",
                    vec![CircuitParameterPath::Index(1)].into_boxed_slice(),
                ),
                CircuitScalarNode::Int(0.into()),
                parameter(
                    "values",
                    vec![CircuitParameterPath::Index(0)].into_boxed_slice(),
                ),
                CircuitScalarNode::Bool(true),
            ]
            .into_boxed_slice(),
        );

        assert_eq!(predicate.parameter_names(), vec!["values"]);
    }

    #[test]
    fn operation_count_includes_nested_structured_blocks() {
        let leaf = StaticInstruction::test_new("x", vec![0], vec![], vec![], None, vec![]);
        let branch = StaticBlock::test_new(vec![leaf]);
        let structured = StaticInstruction::test_new(
            "if",
            vec![],
            vec![],
            vec![],
            None,
            vec![branch, StaticBlock::test_new(vec![])],
        );
        let circuit = StaticCircuit::test_new(1, 0, vec![structured], vec![], vec![]);

        assert_eq!(circuit.operation_count(), 2);
    }
}
