//! Derived whole-program callable contracts.
//!
//! Contracts are reconstructed from locally verified bodies and the normalized
//! strongly connected call graph. No producer-supplied effect cache or transformation
//! proof crosses this boundary.

use thiserror::Error;

use crate::call_graph::{CallFlavor, StaticCallGraph};
use crate::control_dependence::ControlDependenceSummary;
use crate::effects::{
    CallableKind, Capability, CapabilitySet, EffectSet, Requirement, RequirementSet,
    asserted_kind_accepts, capabilities_for, control_effect, external_effects, infer_callable_kind,
    operation_effects, operation_requirements, requirements_for_effects,
};
use crate::equations::LocalEquationFacts;
use crate::operations::{LocallyCheckedOperations, Opcode, QCoreOpcode, QuantumOpcode};
use crate::types::{Stage, TypeArena};
use crate::verification::QCoreVerificationWork;
use crate::wire;

/// One compiler-derived callable summary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CallableContract {
    kind: CallableKind,
    effects: EffectSet,
    capabilities: CapabilitySet,
    requirements: RequirementSet,
}

impl CallableContract {
    /// Return the reconstructed denotational kind.
    #[must_use]
    pub const fn kind(self) -> CallableKind {
        self.kind
    }

    /// Return the closed transitive effect set.
    #[must_use]
    pub const fn effects(self) -> EffectSet {
        self.effects
    }

    /// Return independently established callable transformations.
    #[must_use]
    pub const fn capabilities(self) -> CapabilitySet {
        self.capabilities
    }

    /// Return the closed target requirement set.
    #[must_use]
    pub const fn requirements(self) -> RequirementSet {
        self.requirements
    }
}

/// Retained semantic summaries for the complete checked function graph.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ProgramContracts {
    contracts: Box<[CallableContract]>,
    control_dependence: ControlDependenceSummary,
}

impl ProgramContracts {
    /// Return one callable contract by stable function index.
    #[must_use]
    pub(crate) fn contract(&self, function: usize) -> Option<CallableContract> {
        self.contracts.get(function).copied()
    }

    /// Return the retained whole-program control-dependence result.
    #[must_use]
    pub(crate) const fn control_dependence(&self) -> &ControlDependenceSummary {
        &self.control_dependence
    }
}

/// Failure to derive a closed callable contract.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub(crate) enum ContractError {
    /// Recursive closure exhausted its bounded compiler work allowance.
    #[error("Q4007: recursive callable-contract analysis exceeds its work budget")]
    RecursiveWorkBudget,
    /// A retained graph or value-type row is unexpectedly unavailable.
    #[error("Q4002: missing callable-contract facts for {function:?}")]
    MissingFacts { function: String },
    /// A producer assertion is weaker than the body-derived denotation.
    #[error("Q1050: {function:?} asserts {asserted:?} but derives {derived:?}")]
    IncompatibleAssertion {
        function: String,
        asserted: CallableKind,
        derived: CallableKind,
    },
    /// An adjoint call targets a callable without a reconstructed proof.
    #[error("Q1029: invalid adjoint callable {callee:?}")]
    InvalidAdjointCallable { callee: String },
    /// A coherent call targets a callable without a reconstructed proof.
    #[error("Q1051: invalid controlled callable {callee:?}")]
    InvalidControlledCallable { callee: String },
    /// A Boolean lift targets a callable outside the proved classical predicate class.
    #[error("Q7680: invalid Boolean lift predicate {predicate:?}")]
    InvalidBooleanPredicate { predicate: String },
    /// A normalized call cannot map formal parameters onto actual operands.
    #[error("Q1052: transformed call has an invalid control-argument ABI")]
    InvalidCallOperandMapping,
}

/// Derive closed contracts to a fixed point within each recursive component.
///
/// External declarations receive effects, requirements, and mathematical
/// transformation capabilities from their explicit asserted contract. The
/// `ExternalCall` requirement still forces a target realization to uphold that
/// contract; body-backed callables derive the same facts from their equations.
///
/// # Errors
///
/// Returns [`ContractError`] for incomplete retained facts, incompatible kind
/// assertions, unsupported transformed calls, or inconsistent call operands.
#[cfg(test)]
pub(crate) fn derive(
    package: &wire::QCorePackageProto,
    arena: &TypeArena,
    operations: &LocallyCheckedOperations,
    equations: &LocalEquationFacts,
    call_graph: &StaticCallGraph,
    control_dependence: ControlDependenceSummary,
) -> Result<ProgramContracts, ContractError> {
    let mut work = QCoreVerificationWork::default();
    derive_with_work(
        package,
        arena,
        operations,
        equations,
        call_graph,
        control_dependence,
        &mut work,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn derive_with_work(
    package: &wire::QCorePackageProto,
    arena: &TypeArena,
    operations: &LocallyCheckedOperations,
    equations: &LocalEquationFacts,
    call_graph: &StaticCallGraph,
    control_dependence: ControlDependenceSummary,
    verification_work: &mut QCoreVerificationWork,
) -> Result<ProgramContracts, ContractError> {
    verification_work.charge_contract_derivation(call_graph.function_count());
    let placeholder = CallableContract {
        kind: CallableKind::Classical,
        effects: EffectSet::EMPTY,
        capabilities: CapabilitySet::EMPTY,
        requirements: RequirementSet::EMPTY,
    };
    let mut contracts = vec![placeholder; call_graph.function_count()];
    let context = DerivationContext {
        package,
        arena,
        operations,
        equations,
        call_graph,
        control: &control_dependence,
    };
    for component in call_graph.components() {
        let start_work = verification_work.contract_derivation_steps();
        if !component.is_recursive() {
            let index = component.members()[0];
            verification_work.charge_contract_derivation(1);
            contracts[index] = derive_one(
                index,
                &contracts,
                &context,
                DerivationPhase::Validate,
                verification_work,
            )?;
            continue;
        }
        // Start from body-local facts: effects grow and coherent capabilities
        // can only be removed as the component's transitive facts converge.
        for &index in component.members() {
            contracts[index] = derive_one(
                index,
                &contracts,
                &context,
                DerivationPhase::Local,
                verification_work,
            )?;
        }
        loop {
            let mut changed = false;
            for &index in component.members() {
                verification_work.charge_contract_derivation(1);
                let derived = derive_one(
                    index,
                    &contracts,
                    &context,
                    DerivationPhase::Closure,
                    verification_work,
                )?;
                changed |= contracts[index] != derived;
                contracts[index] = derived;
            }
            if verification_work
                .contract_derivation_steps()
                .saturating_sub(start_work)
                > crate::budget::MAX_RECURSIVE_ANALYSIS_WORK
            {
                return Err(ContractError::RecursiveWorkBudget);
            }
            if !changed {
                break;
            }
        }
        for &index in component.members() {
            derive_one(
                index,
                &contracts,
                &context,
                DerivationPhase::Validate,
                verification_work,
            )?;
        }
    }

    Ok(ProgramContracts {
        contracts: contracts.into_boxed_slice(),
        control_dependence,
    })
}

struct DerivationContext<'a> {
    package: &'a wire::QCorePackageProto,
    arena: &'a TypeArena,
    operations: &'a LocallyCheckedOperations,
    equations: &'a LocalEquationFacts,
    call_graph: &'a StaticCallGraph,
    control: &'a ControlDependenceSummary,
}

/// Separate monotone contract inference from checks requiring closed facts.
#[derive(Clone, Copy, Eq, PartialEq)]
enum DerivationPhase {
    Local,
    Closure,
    Validate,
}

fn derive_one(
    function_index: usize,
    known: &[CallableContract],
    context: &DerivationContext<'_>,
    phase: DerivationPhase,
    work: &mut QCoreVerificationWork,
) -> Result<CallableContract, ContractError> {
    let function = context
        .package
        .functions
        .get(function_index)
        .ok_or_else(|| {
            missing(
                context
                    .call_graph
                    .symbol(function_index)
                    .unwrap_or("<unknown>"),
            )
        })?;
    let value_types = context
        .operations
        .value_types(&function.symbol)
        .ok_or_else(|| missing(&function.symbol))?;
    if function.external {
        work.charge_contract_derivation(function.parameters.len());
        return external_contract(function, value_types, context.arena);
    }

    let (effects, mut requirements) = direct_contract(function, value_types, context.arena, work);
    work.charge_contract_derivation(function.parameters.len());
    if has_parameter_stage(function, value_types, context.arena, Stage::Param) {
        requirements = requirements.with(Requirement::ParameterizedExecution);
    }
    let coherent = !context.equations.has_proof_obligations(&function.symbol);
    let (effects, requirements, coherent) = if phase == DerivationPhase::Local {
        (effects, requirements, coherent)
    } else {
        close_calls(
            function_index,
            value_types,
            known,
            context,
            effects,
            requirements,
            coherent,
            phase == DerivationPhase::Validate,
            work,
        )?
    };
    let result_type = context
        .operations
        .result_type(&function.symbol)
        .ok_or_else(|| missing(&function.symbol))?;
    let inputs_have_quantum = function.parameters.iter().any(|parameter| {
        parameter
            .value
            .as_ref()
            .and_then(|value| value_types.get(&value.id))
            .is_some_and(|type_id| context.arena.is_quantum(*type_id))
    });
    let kind = infer_callable_kind(
        inputs_have_quantum,
        context.arena.is_quantum(result_type),
        effects,
    );
    if phase == DerivationPhase::Validate
        && let Some(asserted) = CallableKind::parse(&function.asserted_kind)
        && !asserted_kind_accepts(asserted, kind)
    {
        return Err(ContractError::IncompatibleAssertion {
            function: function.symbol.clone(),
            asserted,
            derived: kind,
        });
    }
    Ok(CallableContract {
        kind,
        effects,
        capabilities: if coherent {
            capabilities_for(kind)
        } else {
            CapabilitySet::EMPTY
        },
        requirements: requirements.union(requirements_for_effects(effects)),
    })
}

fn external_contract(
    function: &wire::FunctionProto,
    value_types: &std::collections::HashMap<u64, crate::TypeId>,
    arena: &TypeArena,
) -> Result<CallableContract, ContractError> {
    let kind =
        CallableKind::parse(&function.asserted_kind).ok_or_else(|| missing(&function.symbol))?;
    let effects = external_effects(kind).unwrap_or(EffectSet::EMPTY);
    let mut requirements = requirements_for_effects(effects).with(Requirement::ExternalCall);
    if has_parameter_stage(function, value_types, arena, Stage::Param) {
        requirements = requirements.with(Requirement::ParameterizedExecution);
    }
    Ok(CallableContract {
        kind,
        effects,
        capabilities: capabilities_for(kind),
        requirements,
    })
}

#[allow(clippy::too_many_arguments)]
fn close_calls(
    function_index: usize,
    value_types: &std::collections::HashMap<u64, crate::TypeId>,
    known: &[CallableContract],
    context: &DerivationContext<'_>,
    mut effects: EffectSet,
    mut requirements: RequirementSet,
    mut coherent: bool,
    validate: bool,
    work: &mut QCoreVerificationWork,
) -> Result<(EffectSet, RequirementSet, bool), ContractError> {
    let sites = context
        .call_graph
        .sites_for_caller(function_index)
        .ok_or_else(|| missing("<caller>"))?;
    for site in sites {
        work.charge_contract_derivation(1);
        for &callee_index in site.callees() {
            work.charge_contract_derivation(1);
            let contract = known
                .get(callee_index)
                .copied()
                .ok_or_else(|| missing("<callee>"))?;
            let callee_name = context
                .call_graph
                .symbol(callee_index)
                .unwrap_or("<unknown>");
            if validate {
                require_transformation(site.flavor(), callee_name, contract)?;
            }
            if site.flavor() == CallFlavor::BooleanLift {
                continue;
            }
            if site.flavor() == CallFlavor::HistoryAdjoint {
                requirements = requirements.with(Requirement::DynamicLoopHistory);
            }
            coherent = coherent
                && site.flavor() != CallFlavor::HistoryAdjoint
                && (contract.kind() == CallableKind::Classical
                    || (contract.capabilities().contains(Capability::Adjointable)
                        && contract.capabilities().contains(Capability::Controllable)));
            effects = effects.union(contract.effects());
            requirements = requirements.union(
                contract
                    .requirements()
                    .without(Requirement::ParameterizedExecution),
            );
            let callee = context
                .package
                .functions
                .get(callee_index)
                .ok_or_else(|| missing(callee_name))?;
            let offset = call_argument_offset(site.flavor(), site.operands(), callee)?;
            requirements = call_selection_requirement(
                site.flavor(),
                site.operands(),
                offset,
                value_types,
                context.arena,
                requirements,
            )?;
            for &formal in context
                .control
                .parameters(callee_index)
                .ok_or_else(|| missing(callee_name))?
            {
                work.charge_contract_derivation(1);
                let actual = formal
                    .checked_add(offset)
                    .and_then(|index| site.operands().get(index))
                    .ok_or(ContractError::InvalidCallOperandMapping)?;
                let actual_type = value_types
                    .get(actual)
                    .copied()
                    .ok_or(ContractError::InvalidCallOperandMapping)?;
                effects = effects.union(control_effect(
                    context.arena.latest_availability(actual_type),
                ));
            }
        }
    }
    Ok((effects, requirements, coherent))
}

fn call_selection_requirement(
    flavor: CallFlavor,
    operands: &[u64],
    offset: usize,
    value_types: &std::collections::HashMap<u64, crate::TypeId>,
    arena: &TypeArena,
    requirements: RequirementSet,
) -> Result<RequirementSet, ContractError> {
    if flavor != CallFlavor::Control || offset != 2 {
        return Ok(requirements);
    }
    let selector_type = operands
        .get(1)
        .and_then(|value_id| value_types.get(value_id))
        .copied()
        .ok_or(ContractError::InvalidCallOperandMapping)?;
    Ok(
        if matches!(
            arena.stage_of(selector_type),
            Some(Stage::Param | Stage::Shot)
        ) {
            requirements.with(Requirement::DynamicQubitSelection)
        } else {
            requirements
        },
    )
}

fn missing(function: &str) -> ContractError {
    ContractError::MissingFacts {
        function: function.to_owned(),
    }
}

fn require_transformation(
    flavor: CallFlavor,
    callee: &str,
    contract: CallableContract,
) -> Result<(), ContractError> {
    match flavor {
        CallFlavor::Ordinary => Ok(()),
        CallFlavor::HistoryAdjoint if contract.kind() == CallableKind::Unitary => Ok(()),
        CallFlavor::Adjoint if contract.capabilities().contains(Capability::Adjointable) => Ok(()),
        CallFlavor::Control | CallFlavor::Select
            if contract.capabilities().contains(Capability::Controllable) =>
        {
            Ok(())
        }
        CallFlavor::Adjoint | CallFlavor::HistoryAdjoint => {
            Err(ContractError::InvalidAdjointCallable {
                callee: callee.to_owned(),
            })
        }
        CallFlavor::Control | CallFlavor::Select => Err(ContractError::InvalidControlledCallable {
            callee: callee.to_owned(),
        }),
        CallFlavor::BooleanLift if contract.kind() == CallableKind::Classical => Ok(()),
        CallFlavor::BooleanLift => Err(ContractError::InvalidBooleanPredicate {
            predicate: callee.to_owned(),
        }),
    }
}

fn call_argument_offset(
    flavor: CallFlavor,
    operands: &[u64],
    callee: &wire::FunctionProto,
) -> Result<usize, ContractError> {
    match flavor {
        CallFlavor::Ordinary
        | CallFlavor::Adjoint
        | CallFlavor::HistoryAdjoint
        | CallFlavor::BooleanLift => Ok(0),
        CallFlavor::Control => {
            let offset = operands.len().saturating_sub(callee.parameters.len());
            if matches!(offset, 1 | 2) {
                Ok(offset)
            } else {
                Err(ContractError::InvalidCallOperandMapping)
            }
        }
        CallFlavor::Select if operands.len().saturating_sub(callee.parameters.len()) == 1 => Ok(1),
        CallFlavor::Select => Err(ContractError::InvalidCallOperandMapping),
    }
}

fn direct_contract(
    function: &wire::FunctionProto,
    value_types: &std::collections::HashMap<u64, crate::TypeId>,
    arena: &TypeArena,
    work: &mut QCoreVerificationWork,
) -> (EffectSet, RequirementSet) {
    let mut effects = EffectSet::EMPTY;
    let mut requirements = RequirementSet::EMPTY;
    for operation in walk_operations(function) {
        work.charge_contract_derivation(
            1_usize
                .saturating_add(operation.operands.len())
                .saturating_add(operation.results.len())
                .saturating_add(operation.attributes.len())
                .saturating_add(operation.regions.len()),
        );
        let opcode = Opcode::try_from(operation.opcode.as_str())
            .expect("operation vocabulary was checked before contract derivation");
        effects = effects.union(operation_effects(opcode));
        if matches!(
            opcode,
            Opcode::QCore(QCoreOpcode::If | QCoreOpcode::For | QCoreOpcode::While)
        ) {
            effects = effects.union(control_effect(stage_attribute(operation)));
        }
        requirements = requirements.union(operation_requirements(opcode));
        // These operands must be checked/materialized before SDK emission,
        // independently of a target's parameter-expression vocabulary.
        if (matches!(opcode, Opcode::Quantum(QuantumOpcode::Prepare))
            || requires_launch_scalar_evaluation(operation))
            && operation
                .operands
                .first()
                .and_then(|value_id| value_types.get(value_id))
                .is_some_and(|type_id| arena.stage_of(*type_id) == Some(Stage::Param))
        {
            effects = effects.union(control_effect(Some(Stage::Param)));
        }
        if let Some(original) = adjoint_original(operation) {
            requirements = requirements.union(operation_requirements(original));
        }
        if is_shot_loop(operation) {
            requirements = requirements.with(Requirement::DynamicLoop);
        }
        if dynamic_selector_ids(operation)
            .iter()
            .filter_map(|value_id| value_types.get(value_id))
            .any(|type_id| matches!(arena.stage_of(*type_id), Some(Stage::Param | Stage::Shot)))
        {
            requirements = requirements.with(Requirement::DynamicQubitSelection);
        }
        if dynamic_classical_index_ids(operation)
            .iter()
            .filter_map(|value_id| value_types.get(value_id))
            .any(|type_id| matches!(arena.stage_of(*type_id), Some(Stage::Param | Stage::Shot)))
        {
            requirements = requirements.with(Requirement::DynamicClassicalIndex);
        }
    }
    requirements = requirements.union(requirements_for_effects(effects));
    (effects, requirements)
}

fn has_parameter_stage(
    function: &wire::FunctionProto,
    value_types: &std::collections::HashMap<u64, crate::TypeId>,
    arena: &TypeArena,
    stage: Stage,
) -> bool {
    function.parameters.iter().any(|parameter| {
        parameter
            .value
            .as_ref()
            .and_then(|value| value_types.get(&value.id))
            .is_some_and(|type_id| arena.contains_stage(*type_id, stage))
    })
}

fn stage_attribute(operation: &wire::OperationProto) -> Option<Stage> {
    match string_attribute(operation, "stage") {
        Some("Const") => Some(Stage::Const),
        Some("Param") => Some(Stage::Param),
        Some("Shot") => Some(Stage::Shot),
        _ => None,
    }
}

fn is_shot_loop(operation: &wire::OperationProto) -> bool {
    if matches!(operation.opcode.as_str(), "qcore.for" | "qcore.while")
        && stage_attribute(operation) == Some(Stage::Shot)
    {
        return true;
    }
    if operation.opcode != "qcore.while" {
        return false;
    }
    let mut blocks = operation
        .regions
        .get(1)
        .into_iter()
        .flat_map(|region| &region.blocks)
        .map(|block| (block, false))
        .collect::<Vec<_>>();
    while let Some((block, controlled)) = blocks.pop() {
        if controlled
            && block
                .terminator
                .as_ref()
                .is_some_and(|exit| matches!(exit.kind.as_str(), "break" | "return"))
        {
            return true;
        }
        for nested in &block.operations {
            let controlled = controlled
                || (nested.opcode == "qcore.if" && stage_attribute(nested) == Some(Stage::Shot));
            let regions = match nested.opcode.as_str() {
                "qcore.for" => nested.regions.get(1..).unwrap_or_default(),
                "qcore.while" => nested.regions.get(2..).unwrap_or_default(),
                _ => &nested.regions,
            };
            for region in regions {
                blocks.extend(region.blocks.iter().map(|block| (block, controlled)));
            }
        }
    }
    false
}

fn adjoint_original(operation: &wire::OperationProto) -> Option<Opcode> {
    (operation.opcode == "qcore.adjoint")
        .then(|| string_attribute(operation, "original"))
        .flatten()
        .and_then(|opcode| Opcode::try_from(opcode).ok())
}

pub(crate) fn dynamic_selector_ids(operation: &wire::OperationProto) -> Vec<u64> {
    match operation.opcode.as_str() {
        "quantum.measure_x" | "quantum.measure_y" | "quantum.measure_z" => {
            operation.operands.get(1..).unwrap_or_default().to_vec()
        }
        "qcore.adjoint" => {
            let count =
                mapping_string_tuple_len(operation, "original_attributes", "keywords").unwrap_or(0);
            operation
                .operands
                .get(operation.operands.len().saturating_sub(count)..)
                .unwrap_or_default()
                .to_vec()
        }
        opcode if opcode.starts_with("quantum.") => {
            let count = string_tuple_attribute_len(operation, "keywords").unwrap_or(0);
            operation
                .operands
                .get(operation.operands.len().saturating_sub(count)..)
                .unwrap_or_default()
                .to_vec()
        }
        _ => Vec::new(),
    }
}

pub(crate) fn launch_selector_ids(operation: &wire::OperationProto) -> Vec<u64> {
    let mut selectors = dynamic_selector_ids(operation);
    selectors.extend(dynamic_classical_index_ids(operation));
    if operation.opcode == "quantum.prepare" || requires_launch_scalar_evaluation(operation) {
        selectors.extend(operation.operands.first().copied());
    }
    selectors
}

/// Identify scalars whose native numeric evaluation must precede SDK emission.
pub(crate) fn requires_launch_scalar_evaluation(operation: &wire::OperationProto) -> bool {
    operation.opcode == "classical.sqrt"
        || (operation.opcode == "classical.cast"
            && string_attribute(operation, "target") == Some("Real"))
}

fn dynamic_classical_index_ids(operation: &wire::OperationProto) -> Vec<u64> {
    match operation.opcode.as_str() {
        "classical.index" => operation.operands.get(1..2).unwrap_or_default().to_vec(),
        _ => Vec::new(),
    }
}

fn walk_operations(function: &wire::FunctionProto) -> impl Iterator<Item = &wire::OperationProto> {
    enum Work<'a> {
        Block(&'a wire::BlockProto),
        Operation(&'a wire::OperationProto),
    }

    let mut work = function.body.iter().map(Work::Block).collect::<Vec<_>>();
    std::iter::from_fn(move || {
        while let Some(item) = work.pop() {
            match item {
                Work::Block(block) => {
                    work.extend(block.operations.iter().rev().map(Work::Operation));
                }
                Work::Operation(operation) => {
                    for region in operation.regions.iter().rev() {
                        work.extend(region.blocks.iter().rev().map(Work::Block));
                    }
                    return Some(operation);
                }
            }
        }
        None
    })
}

fn string_attribute<'a>(operation: &'a wire::OperationProto, name: &str) -> Option<&'a str> {
    use wire::literal_proto::Value;

    operation
        .attributes
        .iter()
        .find(|attribute| attribute.name == name)
        .and_then(|attribute| attribute.value.as_ref())
        .and_then(|literal| match literal.value.as_ref() {
            Some(Value::StringValue(value)) => Some(value.as_str()),
            _ => None,
        })
}

fn string_tuple_attribute_len(operation: &wire::OperationProto, name: &str) -> Option<usize> {
    use wire::literal_proto::Value;

    operation
        .attributes
        .iter()
        .find(|attribute| attribute.name == name)
        .and_then(|attribute| attribute.value.as_ref())
        .and_then(|literal| match literal.value.as_ref() {
            Some(Value::TupleValue(sequence))
                if sequence
                    .values
                    .iter()
                    .all(|item| matches!(item.value, Some(Value::StringValue(_)))) =>
            {
                Some(sequence.values.len())
            }
            _ => None,
        })
}

fn mapping_string_tuple_len(
    operation: &wire::OperationProto,
    attribute_name: &str,
    key: &str,
) -> Option<usize> {
    use wire::literal_proto::Value;

    let mapping = operation
        .attributes
        .iter()
        .find(|attribute| attribute.name == attribute_name)?
        .value
        .as_ref()?
        .value
        .as_ref();
    let Some(Value::MappingValue(mapping)) = mapping else {
        return None;
    };
    let value = mapping
        .entries
        .iter()
        .find(|entry| entry.key == key)?
        .value
        .as_ref()?
        .value
        .as_ref();
    match value {
        Some(Value::TupleValue(sequence))
            if sequence
                .values
                .iter()
                .all(|item| matches!(item.value, Some(Value::StringValue(_)))) =>
        {
            Some(sequence.values.len())
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effects::{Effect, Requirement};

    fn scalar(name: &str) -> wire::TypeProto {
        wire::TypeProto {
            kind: wire::type_proto::Kind::Scalar.into(),
            name: name.to_owned(),
            ..Default::default()
        }
    }

    fn staged(stage: &str, inner: wire::TypeProto) -> wire::TypeProto {
        wire::TypeProto {
            kind: wire::type_proto::Kind::Staged.into(),
            name: stage.to_owned(),
            arguments: vec![inner],
            ..Default::default()
        }
    }

    fn record(name: &str, fields: Vec<(&str, wire::TypeProto)>) -> wire::TypeProto {
        wire::TypeProto {
            kind: wire::type_proto::Kind::Record.into(),
            name: name.to_owned(),
            field_names: fields
                .iter()
                .map(|(field_name, _)| (*field_name).to_owned())
                .collect(),
            arguments: fields
                .into_iter()
                .map(|(_, field_type)| field_type)
                .collect(),
            ..Default::default()
        }
    }

    fn quantum_bit() -> wire::TypeProto {
        wire::TypeProto {
            kind: wire::type_proto::Kind::Quantum.into(),
            arguments: vec![wire::TypeProto {
                kind: wire::type_proto::Kind::Array.into(),
                name: "Bits".to_owned(),
                arguments: vec![scalar("Bit")],
                dimensions: vec![wire::DimensionProto {
                    value: Some(wire::dimension_proto::Value::Fixed(1)),
                }],
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn value(id: u64, r#type: wire::TypeProto) -> wire::ValueProto {
        wire::ValueProto {
            id,
            r#type: Some(r#type),
            ..Default::default()
        }
    }

    fn parameter(name: &str, id: u64, r#type: wire::TypeProto) -> wire::ParameterProto {
        wire::ParameterProto {
            name: name.to_owned(),
            value: Some(value(id, r#type)),
            ..Default::default()
        }
    }

    fn attribute(name: &str, value: wire::literal_proto::Value) -> wire::AttributeProto {
        wire::AttributeProto {
            name: name.to_owned(),
            value: Some(wire::LiteralProto { value: Some(value) }),
        }
    }

    fn call(
        callee: &str,
        operands: Vec<u64>,
        result: Option<wire::ValueProto>,
    ) -> wire::OperationProto {
        wire::OperationProto {
            opcode: "qcore.call".to_owned(),
            operands,
            results: result.into_iter().collect(),
            attributes: vec![attribute(
                "callee",
                wire::literal_proto::Value::StringValue(callee.to_owned()),
            )],
            ..Default::default()
        }
    }

    fn concrete(
        symbol: &str,
        parameters: Vec<wire::ParameterProto>,
        result_type: wire::TypeProto,
        operations: Vec<wire::OperationProto>,
        result_id: Option<u64>,
        asserted_kind: &str,
    ) -> wire::FunctionProto {
        wire::FunctionProto {
            symbol: symbol.to_owned(),
            parameters,
            result_type: Some(result_type),
            body: Some(wire::BlockProto {
                operations,
                terminator: Some(wire::TerminatorProto {
                    kind: "return".to_owned(),
                    operands: result_id.into_iter().collect(),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            asserted_kind: asserted_kind.to_owned(),
            ..Default::default()
        }
    }

    fn external(
        symbol: &str,
        parameters: Vec<wire::ParameterProto>,
        result: wire::TypeProto,
    ) -> wire::FunctionProto {
        wire::FunctionProto {
            symbol: symbol.to_owned(),
            parameters,
            result_type: Some(result),
            body: Some(wire::BlockProto {
                terminator: Some(wire::TerminatorProto {
                    kind: "external".to_owned(),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            asserted_kind: "unitary".to_owned(),
            external: true,
            ..Default::default()
        }
    }

    fn derive_test(
        functions: Vec<wire::FunctionProto>,
    ) -> Result<(ProgramContracts, StaticCallGraph), ContractError> {
        let package = wire::QCorePackageProto {
            functions,
            ..Default::default()
        };
        let mut arena = crate::types::validate_program_types(&package).expect("valid test types");
        let operations =
            crate::operations::validate_program_operations_with_types(&package, &mut arena)
                .expect("valid test operation graph");
        let graph = crate::call_graph::build(&package).expect("valid test call graph");
        let control = crate::control_dependence::analyze(&package, &graph)
            .expect("valid test control dependence");
        let equations = LocalEquationFacts::default();
        let contracts = derive(&package, &arena, &operations, &equations, &graph, control)?;
        Ok((contracts, graph))
    }

    #[test]
    fn closes_external_requirements_through_a_chain_and_diamond() {
        let unit = scalar("Unit");
        let leaf = external("test.leaf", vec![], unit.clone());
        let left = concrete(
            "test.left",
            vec![],
            unit.clone(),
            vec![call("test.leaf", vec![], Some(value(1, unit.clone())))],
            Some(1),
            "",
        );
        let right = concrete(
            "test.right",
            vec![],
            unit.clone(),
            vec![call("test.leaf", vec![], Some(value(1, unit.clone())))],
            Some(1),
            "",
        );
        let root = concrete(
            "test.root",
            vec![],
            unit.clone(),
            vec![
                call("test.left", vec![], Some(value(1, unit.clone()))),
                call("test.right", vec![], Some(value(2, unit))),
            ],
            Some(2),
            "",
        );

        let (contracts, graph) =
            derive_test(vec![leaf, left, right, root]).expect("closed contracts");
        assert_eq!(graph.callee_first_order(), [0, 1, 2, 3]);
        let leaf_contract = contracts.contract(0).expect("leaf contract");
        assert!(leaf_contract.effects().contains(Effect::Quantum));
        assert!(
            leaf_contract
                .requirements()
                .contains(Requirement::ExternalCall)
        );
        assert_eq!(leaf_contract.capabilities(), CapabilitySet::COHERENT);
        let root_contract = contracts.contract(3).expect("root contract");
        assert!(root_contract.effects().contains(Effect::Quantum));
        assert!(
            root_contract
                .requirements()
                .contains(Requirement::ExternalCall)
        );
        assert!(
            !root_contract
                .requirements()
                .contains(Requirement::ParameterizedExecution)
        );
        assert_eq!(root_contract.capabilities(), CapabilitySet::COHERENT);
    }

    #[test]
    fn external_unitary_contracts_authorize_mathematical_transforms() {
        let quantum = quantum_bit();
        let external_function = external(
            "test.external",
            vec![parameter("state", 1, quantum.clone())],
            quantum.clone(),
        );
        let root_function = concrete(
            "test.root",
            vec![parameter("state", 1, quantum.clone())],
            quantum.clone(),
            vec![wire::OperationProto {
                opcode: "qcore.adjoint_call".to_owned(),
                operands: vec![1],
                results: vec![value(2, quantum)],
                attributes: vec![attribute(
                    "callee",
                    wire::literal_proto::Value::StringValue("test.external".to_owned()),
                )],
                ..Default::default()
            }],
            Some(2),
            "",
        );

        let (contracts, _) =
            derive_test(vec![external_function, root_function]).expect("external contract");
        assert_eq!(
            contracts.contract(0).expect("external").capabilities(),
            CapabilitySet::COHERENT
        );
        assert_eq!(
            contracts.contract(1).expect("wrapper").capabilities(),
            CapabilitySet::COHERENT
        );
        assert!(
            contracts
                .contract(1)
                .expect("wrapper")
                .requirements()
                .contains(Requirement::ExternalCall)
        );
    }

    #[test]
    fn accepts_all_transforms_only_for_a_derived_unitary() {
        let quantum = quantum_bit();
        let unitary_function = concrete(
            "test.unitary",
            vec![parameter("state", 1, quantum.clone())],
            quantum.clone(),
            vec![wire::OperationProto {
                opcode: "quantum.x".to_owned(),
                operands: vec![1],
                results: vec![value(2, quantum.clone())],
                attributes: vec![attribute(
                    "keywords",
                    wire::literal_proto::Value::TupleValue(wire::literal_proto::Sequence::default()),
                )],
                ..Default::default()
            }],
            Some(2),
            "unitary",
        );
        let root_function = concrete(
            "test.root",
            vec![
                parameter("selector", 1, quantum.clone()),
                parameter("state", 2, quantum.clone()),
            ],
            scalar("Unit"),
            vec![
                wire::OperationProto {
                    opcode: "qcore.adjoint_call".to_owned(),
                    operands: vec![2],
                    attributes: vec![attribute(
                        "callee",
                        wire::literal_proto::Value::StringValue("test.unitary".to_owned()),
                    )],
                    ..Default::default()
                },
                wire::OperationProto {
                    opcode: "qcore.control_call".to_owned(),
                    operands: vec![1, 2],
                    attributes: vec![
                        attribute(
                            "callee",
                            wire::literal_proto::Value::StringValue("test.unitary".to_owned()),
                        ),
                        attribute("control_state", wire::literal_proto::Value::IntValue(1)),
                    ],
                    ..Default::default()
                },
                wire::OperationProto {
                    opcode: "qcore.select_call".to_owned(),
                    operands: vec![1, 2],
                    attributes: vec![attribute(
                        "callees",
                        wire::literal_proto::Value::TupleValue(wire::literal_proto::Sequence {
                            values: vec![wire::LiteralProto {
                                value: Some(wire::literal_proto::Value::StringValue(
                                    "test.unitary".to_owned(),
                                )),
                            }],
                        }),
                    )],
                    ..Default::default()
                },
            ],
            None,
            "",
        );

        let (contracts, _) =
            derive_test(vec![root_function, unitary_function]).expect("valid transforms");
        assert_eq!(
            contracts.contract(1).expect("callee").kind(),
            CallableKind::Unitary
        );
        assert_eq!(
            contracts.contract(1).expect("callee").capabilities(),
            CapabilitySet::COHERENT
        );
    }

    #[test]
    fn instantiates_param_and_shot_control_at_call_sites() {
        let unit = scalar("Unit");
        let controlled_function = concrete(
            "test.controlled",
            vec![
                parameter("flag", 1, scalar("Bool")),
                parameter("unit", 2, unit.clone()),
            ],
            unit.clone(),
            vec![wire::OperationProto {
                opcode: "qcore.if".to_owned(),
                operands: vec![1],
                attributes: vec![attribute(
                    "stage",
                    wire::literal_proto::Value::StringValue("Value".to_owned()),
                )],
                ..Default::default()
            }],
            Some(2),
            "",
        );
        let staged_caller = |symbol: &str, stage: &str| {
            concrete(
                symbol,
                vec![
                    parameter("flag", 1, staged(stage, scalar("Bool"))),
                    parameter("unit", 2, unit.clone()),
                ],
                unit.clone(),
                vec![call(
                    "test.controlled",
                    vec![1, 2],
                    Some(value(3, staged(stage, unit.clone()))),
                )],
                Some(3),
                "",
            )
        };

        let (contracts, _) = derive_test(vec![
            controlled_function,
            staged_caller("test.param", "Param"),
            staged_caller("test.shot", "Shot"),
        ])
        .expect("stage-instantiated control");
        let param = contracts.contract(1).expect("param caller");
        assert!(param.effects().contains(Effect::HostControl));
        assert!(param.requirements().contains(Requirement::HostDispatch));
        assert!(
            param
                .requirements()
                .contains(Requirement::ParameterizedExecution)
        );
        let shot = contracts.contract(2).expect("shot caller");
        assert!(shot.effects().contains(Effect::ShotControl));
        assert!(shot.requirements().contains(Requirement::DynamicBranch));
        assert!(
            !shot
                .requirements()
                .contains(Requirement::ParameterizedExecution)
        );
    }

    #[test]
    fn instantiates_control_from_stages_nested_in_aggregate_arguments() {
        let unit = scalar("Unit");
        let declared_flags = record("Flags", vec![("enabled", scalar("Bool"))]);
        let runtime_flags = record("Flags", vec![("enabled", staged("Param", scalar("Bool")))]);
        let controlled = concrete(
            "test.controlled",
            vec![parameter("flags", 1, declared_flags)],
            unit.clone(),
            vec![wire::OperationProto {
                opcode: "qcore.if".to_owned(),
                operands: vec![1],
                attributes: vec![attribute(
                    "stage",
                    wire::literal_proto::Value::StringValue("Value".to_owned()),
                )],
                ..Default::default()
            }],
            None,
            "",
        );
        let caller = concrete(
            "test.caller",
            vec![parameter("flags", 1, runtime_flags)],
            unit.clone(),
            vec![call("test.controlled", vec![1], Some(value(2, unit)))],
            Some(2),
            "",
        );

        let (contracts, _) =
            derive_test(vec![caller, controlled]).expect("nested availability contract");
        let caller = contracts.contract(0).expect("caller contract");
        assert!(caller.effects().contains(Effect::HostControl));
        assert!(caller.requirements().contains(Requirement::HostDispatch));
        assert!(
            caller
                .requirements()
                .contains(Requirement::ParameterizedExecution)
        );
    }

    #[test]
    fn enforces_kind_assertions_and_strengthened_requirements() {
        let quantum = quantum_bit();
        let incompatible = concrete(
            "test.bad",
            vec![parameter("state", 1, quantum.clone())],
            quantum.clone(),
            vec![wire::OperationProto {
                opcode: "quantum.reset".to_owned(),
                operands: vec![1],
                results: vec![value(2, quantum.clone())],
                ..Default::default()
            }],
            Some(2),
            "unitary",
        );
        assert!(matches!(
            derive_test(vec![incompatible]),
            Err(ContractError::IncompatibleAssertion {
                asserted: CallableKind::Unitary,
                derived: CallableKind::Channel,
                ..
            })
        ));

        let measure_reset = concrete(
            "test.measure_reset",
            vec![parameter("state", 1, quantum.clone())],
            quantum.clone(),
            vec![wire::OperationProto {
                opcode: "quantum.measure_reset".to_owned(),
                operands: vec![1],
                results: vec![value(2, quantum)],
                ..Default::default()
            }],
            Some(2),
            "",
        );
        let shot_loop = concrete(
            "test.shot_loop",
            vec![parameter("bound", 1, staged("Shot", scalar("Int")))],
            scalar("Unit"),
            vec![wire::OperationProto {
                opcode: "qcore.for".to_owned(),
                operands: vec![1, 1, 1],
                attributes: vec![attribute(
                    "stage",
                    wire::literal_proto::Value::StringValue("Shot".to_owned()),
                )],
                ..Default::default()
            }],
            None,
            "",
        );
        let (contracts, _) =
            derive_test(vec![measure_reset, shot_loop]).expect("valid requirement contracts");
        assert!(
            contracts
                .contract(0)
                .expect("measurement contract")
                .requirements()
                .contains(Requirement::MidCircuitMeasurement)
        );
        let loop_contract = contracts.contract(1).expect("loop contract");
        assert!(
            loop_contract
                .requirements()
                .contains(Requirement::DynamicLoop)
        );
        assert!(
            loop_contract
                .requirements()
                .contains(Requirement::DynamicBranch)
        );
    }

    #[test]
    fn retains_dynamic_selection_from_an_adjoint_gate() {
        let dynamic_adjoint = concrete(
            "test.dynamic_adjoint",
            vec![
                parameter("state", 1, quantum_bit()),
                parameter("target", 2, staged("Param", scalar("Int"))),
            ],
            quantum_bit(),
            vec![wire::OperationProto {
                opcode: "qcore.adjoint".to_owned(),
                operands: vec![1, 2],
                results: vec![value(3, quantum_bit())],
                attributes: vec![
                    attribute(
                        "original",
                        wire::literal_proto::Value::StringValue("quantum.x".to_owned()),
                    ),
                    attribute(
                        "original_attributes",
                        wire::literal_proto::Value::MappingValue(wire::literal_proto::Mapping {
                            entries: vec![wire::literal_proto::Entry {
                                key: "keywords".to_owned(),
                                value: Some(wire::LiteralProto {
                                    value: Some(wire::literal_proto::Value::TupleValue(
                                        wire::literal_proto::Sequence {
                                            values: vec![wire::LiteralProto {
                                                value: Some(
                                                    wire::literal_proto::Value::StringValue(
                                                        "target".to_owned(),
                                                    ),
                                                ),
                                            }],
                                        },
                                    )),
                                }),
                            }],
                        }),
                    ),
                    attribute(
                        "proof",
                        wire::literal_proto::Value::StringValue("modular-adjoint".to_owned()),
                    ),
                ],
                ..Default::default()
            }],
            Some(3),
            "",
        );
        let (contracts, _) =
            derive_test(vec![dynamic_adjoint]).expect("valid adjoint requirement contract");
        assert!(
            contracts
                .contract(0)
                .expect("adjoint contract")
                .requirements()
                .contains(Requirement::DynamicQubitSelection)
        );
    }

    #[test]
    fn retains_dynamic_classical_index_as_an_independent_requirement() {
        let vector = wire::TypeProto {
            kind: wire::type_proto::Kind::Array.into(),
            name: "Array".to_owned(),
            arguments: vec![scalar("Real")],
            dimensions: vec![wire::DimensionProto {
                value: Some(wire::dimension_proto::Value::Fixed(2)),
            }],
            ..Default::default()
        };
        let dynamic_index = concrete(
            "test.dynamic_index",
            vec![
                parameter("values", 1, staged("Param", vector)),
                parameter("index", 2, staged("Param", scalar("Int"))),
            ],
            scalar("Real"),
            vec![wire::OperationProto {
                opcode: "classical.index".to_owned(),
                operands: vec![1, 2],
                results: vec![value(3, staged("Param", scalar("Real")))],
                ..Default::default()
            }],
            Some(3),
            "",
        );

        let (contracts, _) =
            derive_test(vec![dynamic_index]).expect("valid dynamic-index contract");
        let requirements = contracts
            .contract(0)
            .expect("dynamic-index contract")
            .requirements();

        assert!(requirements.contains(Requirement::DynamicClassicalIndex));
        assert!(!requirements.contains(Requirement::DynamicQubitSelection));
    }
}
