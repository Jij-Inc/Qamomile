//! Function-local SSA, ownership, and structured-control verification.
//!
//! This pass owns the `QCore` lexical-scope model. Nested regions are
//! explicit-capture scopes, operation results do not dominate
//! their own regions, and every value containing coherent data is consumed
//! exactly once. It also establishes the complete region signatures of
//! `qcore.if`, `qcore.for`, and `qcore.while`, including control availability.
//!
//! Local equations for quantum operations, callable ABIs, transitive effects,
//! and callable contracts remain deliberately deferred. Consequently this pass
//! strengthens `LocallyCheckedProgram`; it does not create a fully verified IR.

use std::collections::{HashMap, HashSet};

use thiserror::Error;

use crate::operations::LocallyCheckedOperations;
use crate::type_relations::is_assignable;
use crate::types::{ScalarType, SemanticType, Stage, TypeArena, TypeId};
use crate::wire;

/// Transient work charged by decoded `QCore` semantic verification.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct QCoreVerificationWork {
    literal_nodes: u128,
    type_components: u128,
    operation_kernel_nodes: u128,
    attribute_nodes: u128,
    call_graph_steps: u128,
    declaration_nodes: u128,
    static_domain_steps: u128,
    equation_steps: u128,
    call_abi_steps: u128,
    phase_lifting_steps: u128,
    control_dependence_steps: u128,
    contract_derivation_steps: u128,
    functions: u128,
    blocks: u128,
    operations: u128,
    values: u128,
    operands: u128,
    topology: u128,
    ownership_steps: u128,
    control_steps: u128,
}

impl QCoreVerificationWork {
    /// Return portable literal nodes visited before semantic decoding.
    #[must_use]
    pub const fn literal_node_visits(&self) -> u128 {
        self.literal_nodes
    }

    /// Return wire-type nodes, dimensions, and field labels inspected.
    #[must_use]
    pub const fn type_component_visits(&self) -> u128 {
        self.type_components
    }

    /// Return closed-dialect, value-index, and operand-resolution visits.
    #[must_use]
    pub const fn operation_kernel_visits(&self) -> u128 {
        self.operation_kernel_nodes
    }

    /// Return blocks, operations, terminators, and attributes inspected.
    #[must_use]
    pub const fn attribute_node_visits(&self) -> u128 {
        self.attribute_nodes
    }

    /// Return static call-graph node, edge, and ordering steps.
    #[must_use]
    pub const fn call_graph_steps(&self) -> u128 {
        self.call_graph_steps
    }

    /// Return declaration, binding, and concrete-type roots inspected.
    #[must_use]
    pub const fn declaration_node_visits(&self) -> u128 {
        self.declaration_nodes
    }

    /// Return static-domain functions, blocks, operations, and captures inspected.
    #[must_use]
    pub const fn static_domain_steps(&self) -> u128 {
        self.static_domain_steps
    }

    /// Return local equation indexing, operand, result, and proof steps.
    #[must_use]
    pub const fn equation_steps(&self) -> u128 {
        self.equation_steps
    }

    /// Return dependent call-ABI indexing, traversal, and matching steps.
    #[must_use]
    pub const fn call_abi_steps(&self) -> u128 {
        self.call_abi_steps
    }

    /// Return Boolean phase-plan indexing, traversal, and proof steps.
    #[must_use]
    pub const fn phase_lifting_steps(&self) -> u128 {
        self.phase_lifting_steps
    }

    /// Return whole-program control-dependence propagation steps.
    #[must_use]
    pub const fn control_dependence_steps(&self) -> u128 {
        self.control_dependence_steps
    }

    /// Return callable-contract derivation and closure steps.
    #[must_use]
    pub const fn contract_derivation_steps(&self) -> u128 {
        self.contract_derivation_steps
    }

    /// Return the number of function declarations visited.
    #[must_use]
    pub const fn function_visits(&self) -> u128 {
        self.functions
    }

    /// Return the number of lexical blocks visited.
    #[must_use]
    pub const fn block_visits(&self) -> u128 {
        self.blocks
    }

    /// Return the number of operations visited.
    #[must_use]
    pub const fn operation_visits(&self) -> u128 {
        self.operations
    }

    /// Return the number of typed SSA values inspected.
    #[must_use]
    pub const fn value_visits(&self) -> u128 {
        self.values
    }

    /// Return the number of operand uses consumed.
    #[must_use]
    pub const fn operand_visits(&self) -> u128 {
        self.operands
    }

    /// Return the number of topology pairs compared.
    #[must_use]
    pub const fn topology_comparisons(&self) -> u128 {
        self.topology
    }

    /// Return ownership-trie path steps charged by insertions and removals.
    #[must_use]
    pub const fn ownership_path_steps(&self) -> u128 {
        self.ownership_steps
    }

    /// Return structured-control and stage-analysis steps.
    #[must_use]
    pub const fn control_flow_steps(&self) -> u128 {
        self.control_steps
    }

    /// Return all charged local-verification work.
    #[must_use]
    pub const fn verification_steps(&self) -> u128 {
        self.literal_nodes
            .saturating_add(self.type_components)
            .saturating_add(self.operation_kernel_nodes)
            .saturating_add(self.attribute_nodes)
            .saturating_add(self.call_graph_steps)
            .saturating_add(self.declaration_nodes)
            .saturating_add(self.static_domain_steps)
            .saturating_add(self.equation_steps)
            .saturating_add(self.call_abi_steps)
            .saturating_add(self.phase_lifting_steps)
            .saturating_add(self.control_dependence_steps)
            .saturating_add(self.contract_derivation_steps)
            .saturating_add(self.functions)
            .saturating_add(self.blocks)
            .saturating_add(self.operations)
            .saturating_add(self.values)
            .saturating_add(self.operands)
            .saturating_add(self.topology)
            .saturating_add(self.ownership_steps)
            .saturating_add(self.control_steps)
    }

    fn visit_function(&mut self) {
        self.functions = self.functions.saturating_add(1);
    }

    pub(crate) fn visit_literal(&mut self) {
        self.literal_nodes = self.literal_nodes.saturating_add(1);
    }

    pub(crate) fn charge_types(&mut self, count: usize) {
        self.type_components = self.type_components.saturating_add(count as u128);
    }

    pub(crate) fn charge_operation_kernel(&mut self, count: usize) {
        self.operation_kernel_nodes = self.operation_kernel_nodes.saturating_add(count as u128);
    }

    pub(crate) fn charge_attributes(&mut self, count: usize) {
        self.attribute_nodes = self.attribute_nodes.saturating_add(count as u128);
    }

    pub(crate) fn charge_call_graph(&mut self, count: usize) {
        self.call_graph_steps = self.call_graph_steps.saturating_add(count as u128);
    }

    pub(crate) fn charge_declarations(&mut self, count: usize) {
        self.declaration_nodes = self.declaration_nodes.saturating_add(count as u128);
    }

    pub(crate) fn charge_static_domains(&mut self, count: usize) {
        self.static_domain_steps = self.static_domain_steps.saturating_add(count as u128);
    }

    pub(crate) fn charge_equations(&mut self, count: usize) {
        self.equation_steps = self.equation_steps.saturating_add(count as u128);
    }

    pub(crate) fn charge_call_abi(&mut self, count: usize) {
        self.call_abi_steps = self.call_abi_steps.saturating_add(count as u128);
    }

    pub(crate) fn charge_phase_lifting(&mut self, count: usize) {
        self.phase_lifting_steps = self.phase_lifting_steps.saturating_add(count as u128);
    }

    pub(crate) fn charge_control_dependence(&mut self, count: usize) {
        self.control_dependence_steps = self.control_dependence_steps.saturating_add(count as u128);
    }

    pub(crate) fn charge_contract_derivation(&mut self, count: usize) {
        self.contract_derivation_steps =
            self.contract_derivation_steps.saturating_add(count as u128);
    }

    fn visit_block(&mut self) {
        self.blocks = self.blocks.saturating_add(1);
    }

    fn visit_operation(&mut self) {
        self.operations = self.operations.saturating_add(1);
    }

    fn visit_value(&mut self) {
        self.values = self.values.saturating_add(1);
    }

    fn visit_operand(&mut self) {
        self.operands = self.operands.saturating_add(1);
    }

    fn visit_topology(&mut self) {
        self.topology = self.topology.saturating_add(1);
    }

    fn charge_ownership(&mut self, steps: usize) {
        self.ownership_steps = self.ownership_steps.saturating_add(steps as u128);
    }

    fn charge_control(&mut self, steps: usize) {
        self.control_steps = self.control_steps.saturating_add(steps as u128);
    }
}

/// A function-local SSA, ownership, or structured-control failure.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub(crate) enum LocalVerificationError {
    /// One use is not visible at its lexical program point.
    #[error("Q1013: SSA value %{value_id} does not dominate this use")]
    NonDominatingUse { value_id: u64 },
    /// A coherent value omitted its ownership identity.
    #[error("Q1012: linear value %{value_id} has no ownership region")]
    MissingOwnershipRegion { value_id: u64 },
    /// A classical value spuriously carries an ownership identity.
    #[error("Q1044: non-linear value %{value_id} has an ownership region")]
    RegionOnNonLinearValue { value_id: u64 },
    /// A coherent value is not consumed exactly once.
    #[error("Q2001: linear value %{value_id} must be consumed exactly once; observed {uses} uses")]
    InvalidLinearUseCount { value_id: u64, uses: usize },
    /// Two simultaneously live coherent values have overlapping owners.
    #[error("Q2002: live ownership region for value %{value_id} aliases another live value")]
    AliasedLiveRegion { value_id: u64 },
    /// A retained operation fact is internally incomplete.
    #[error("Q4002: missing retained type root for SSA value %{value_id}")]
    MissingTypeRoot { value_id: u64 },
    /// A structured operation has the wrong coarse shape.
    #[error("{0}")]
    InvalidStructuredShape(String),
    /// A structured operation's control value has the wrong type.
    #[error("{0}")]
    InvalidControlType(String),
    /// A structured operation carries an invalid control stage.
    #[error("Q1049: {opcode} has an invalid control stage")]
    InvalidControlStage { opcode: String },
    /// One structured region does not have exactly one block.
    #[error("Q1042: {opcode} regions must contain exactly one block")]
    InvalidRegionBlockCount { opcode: String },
    /// A region's explicit capture signature disagrees with its sources.
    #[error("Q1009: region signature mismatch in {opcode}")]
    RegionSignatureMismatch { opcode: String },
    /// A structured result disagrees with its loop-carried source.
    #[error("Q1043: result topology mismatch in {opcode}")]
    ResultTopologyMismatch { opcode: String },
    /// A structured region yields the wrong result topology.
    #[error("Q1010: yield/result mismatch in {opcode}")]
    YieldResultMismatch { opcode: String },
    /// A structured classical result loses a persistent control dependency.
    #[error("Q1053: {opcode} result omits its control dependency")]
    MissingControlDependency { opcode: String },
    /// A concrete function exposes unsupported root block parameters.
    #[error("Q1054: concrete function {function:?} has root block arguments")]
    RootBlockArguments { function: String },
    /// A return does not satisfy its containing callable's result ABI.
    #[error("Q1023: function return does not match its declared result type")]
    ReturnTypeMismatch,
    /// An unreachable exit spuriously carries values.
    #[error("Q1024: unreachable terminator has operands")]
    UnreachableHasOperands,
    /// A control transfer is invalid in its lexical structured context.
    #[error("Q1055: {kind:?} terminator is invalid in this structured context")]
    InvalidTerminatorContext { kind: String },
    /// A break or continue carries the wrong loop state.
    #[error("Q1055: {kind} terminator does not match its loop carry topology")]
    InvalidLoopExitTopology { kind: String },
    /// A structured operation has ambiguous or forged semantic metadata.
    #[error("Q1056: {opcode} has an invalid structured attribute schema")]
    InvalidStructuredAttributes { opcode: String },
    /// An external declaration has executable body state or no trusted kind.
    #[error("Q1047: external function {function:?} has an invalid declaration")]
    InvalidExternalDeclaration { function: String },
    /// A concrete function uses the declaration-only external terminator.
    #[error("Q1048: concrete function {function:?} uses an external terminator")]
    ConcreteExternalTerminator { function: String },
}

#[derive(Clone, Copy)]
struct ValueFact<'a> {
    source: &'a wire::ValueProto,
    type_id: TypeId,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Availability {
    Value,
    Const,
    Param,
    Shot,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BlockRole {
    Root,
    Region,
}

#[derive(Clone, Copy)]
struct ControlContext<'a> {
    role: BlockRole,
    loop_results: Option<&'a [wire::ValueProto]>,
}

impl Availability {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Value => "Value",
            Self::Const => "Const",
            Self::Param => "Param",
            Self::Shot => "Shot",
        }
    }
}

/// Verify lexical SSA, exact linear consumption, and structured-region ABIs.
///
/// # Errors
///
/// Returns [`LocalVerificationError`] at the first authority-ordered failure.
pub(crate) fn verify_program_with_work(
    package: &wire::QCorePackageProto,
    arena: &TypeArena,
    operations: &LocallyCheckedOperations,
    work: &mut QCoreVerificationWork,
) -> Result<(), LocalVerificationError> {
    for function in &package.functions {
        work.visit_function();
        let values = operations.value_types(&function.symbol).ok_or_else(|| {
            LocalVerificationError::InvalidStructuredShape(format!(
                "Q4002: missing retained value roots for function {:?}",
                function.symbol
            ))
        })?;
        let result_type = operations.result_type(&function.symbol).ok_or_else(|| {
            LocalVerificationError::InvalidStructuredShape(format!(
                "Q4002: missing retained result type for function {:?}",
                function.symbol
            ))
        })?;
        let mut verifier = FunctionVerifier {
            arena,
            values,
            result_type,
            uses: HashMap::new(),
            linear_definitions: Vec::new(),
            assignability: HashMap::new(),
            forwarded_values: HashMap::new(),
            work,
        };
        let mut visible = HashMap::new();
        let mut parameter_regions = LiveRegions::default();
        for parameter in &function.parameters {
            let value = parameter
                .value
                .as_ref()
                .ok_or(LocalVerificationError::MissingTypeRoot { value_id: u64::MAX })?;
            let fact = verifier.define(value)?;
            verifier.insert_live(fact, &mut parameter_regions)?;
            visible.insert(value.id, fact);
        }
        if function.external {
            // Externals have no local body execution, but their parameter IDs,
            // types, and owners still cross the native trust boundary.
            let body = function
                .body
                .as_ref()
                .expect("canonical structure requires a function body");
            let valid_kind = matches!(
                function.asserted_kind.as_str(),
                "unitary" | "isometry" | "channel" | "instrument"
            );
            let valid_terminator = body.terminator.as_ref().is_some_and(|terminator| {
                terminator.kind == "external" && terminator.operands.is_empty()
            });
            if !valid_kind
                || !body.arguments.is_empty()
                || !body.operations.is_empty()
                || !valid_terminator
            {
                return Err(LocalVerificationError::InvalidExternalDeclaration {
                    function: function.symbol.clone(),
                });
            }
            continue;
        }
        let body = function
            .body
            .as_ref()
            .expect("canonical structure requires a function body");
        if body
            .terminator
            .as_ref()
            .is_some_and(|terminator| terminator.kind == "external")
        {
            return Err(LocalVerificationError::ConcreteExternalTerminator {
                function: function.symbol.clone(),
            });
        }
        if !body.arguments.is_empty() {
            return Err(LocalVerificationError::RootBlockArguments {
                function: function.symbol.clone(),
            });
        }
        verifier.verify_block(
            body,
            visible,
            ControlContext {
                role: BlockRole::Root,
                loop_results: None,
            },
        )?;
        for &value_id in &verifier.linear_definitions {
            verifier.work.visit_value();
            let uses = verifier.uses.get(&value_id).copied().unwrap_or(0);
            if uses != 1 {
                return Err(LocalVerificationError::InvalidLinearUseCount { value_id, uses });
            }
        }
    }
    Ok(())
}

struct FunctionVerifier<'a, 'work> {
    arena: &'a TypeArena,
    values: &'a HashMap<u64, TypeId>,
    result_type: TypeId,
    uses: HashMap<u64, usize>,
    linear_definitions: Vec<u64>,
    assignability: HashMap<(TypeId, TypeId), bool>,
    forwarded_values: HashMap<u64, u64>,
    work: &'work mut QCoreVerificationWork,
}

impl<'a> FunctionVerifier<'a, '_> {
    fn fact(
        &mut self,
        value: &'a wire::ValueProto,
    ) -> Result<ValueFact<'a>, LocalVerificationError> {
        self.work.visit_value();
        let type_id = self
            .values
            .get(&value.id)
            .copied()
            .ok_or(LocalVerificationError::MissingTypeRoot { value_id: value.id })?;
        Ok(ValueFact {
            source: value,
            type_id,
        })
    }

    fn define(
        &mut self,
        value: &'a wire::ValueProto,
    ) -> Result<ValueFact<'a>, LocalVerificationError> {
        let fact = self.fact(value)?;
        let linear = self.arena.is_quantum(fact.type_id);
        if linear && value.region.is_none() {
            return Err(LocalVerificationError::MissingOwnershipRegion { value_id: value.id });
        }
        if !linear && value.region.is_some() {
            return Err(LocalVerificationError::RegionOnNonLinearValue { value_id: value.id });
        }
        if linear {
            self.linear_definitions.push(value.id);
        }
        Ok(fact)
    }

    fn use_operands(
        &mut self,
        operands: &[u64],
        visible: &HashMap<u64, ValueFact<'a>>,
        live: &mut LiveRegions<'a>,
    ) -> Result<Vec<ValueFact<'a>>, LocalVerificationError> {
        operands
            .iter()
            .map(|value_id| {
                self.work.visit_operand();
                let value = visible.get(value_id).copied().ok_or(
                    LocalVerificationError::NonDominatingUse {
                        value_id: *value_id,
                    },
                )?;
                *self.uses.entry(*value_id).or_default() += 1;
                if self.arena.is_quantum(value.type_id) {
                    let region = value
                        .source
                        .region
                        .as_ref()
                        .expect("linear definitions have checked owners");
                    self.work
                        .charge_ownership(region.projections.len().saturating_add(1));
                    live.remove(region);
                }
                Ok(value)
            })
            .collect()
    }

    fn verify_block(
        &mut self,
        block: &'a wire::BlockProto,
        mut visible: HashMap<u64, ValueFact<'a>>,
        context: ControlContext<'a>,
    ) -> Result<Vec<ValueFact<'a>>, LocalVerificationError> {
        self.work.visit_block();
        let mut live = LiveRegions::default();
        for value in visible.values().copied() {
            self.insert_live(value, &mut live)?;
        }
        for argument in &block.arguments {
            let fact = self.define(argument)?;
            self.insert_live(fact, &mut live)?;
            visible.insert(argument.id, fact);
        }
        for operation in &block.operations {
            self.work.visit_operation();
            let operands = self.use_operands(&operation.operands, &visible, &mut live)?;
            self.verify_operation_regions(operation, &operands, context.loop_results)?;
            for result in &operation.results {
                let fact = self.define(result)?;
                self.insert_live(fact, &mut live)?;
                visible.insert(result.id, fact);
            }
        }
        let terminator = block
            .terminator
            .as_ref()
            .expect("operation vocabulary pass requires every terminator");
        let values = self.use_operands(&terminator.operands, &visible, &mut live)?;
        self.verify_terminator(terminator, &values, context)?;
        Ok(values)
    }

    fn insert_live(
        &mut self,
        value: ValueFact<'a>,
        live: &mut LiveRegions<'a>,
    ) -> Result<(), LocalVerificationError> {
        if !self.arena.is_quantum(value.type_id) {
            return Ok(());
        }
        let region = value
            .source
            .region
            .as_ref()
            .expect("linear definitions have checked owners");
        self.work
            .charge_ownership(region.projections.len().saturating_add(1).saturating_mul(2));
        if live.insert(region) {
            Ok(())
        } else {
            Err(LocalVerificationError::AliasedLiveRegion {
                value_id: value.source.id,
            })
        }
    }

    fn verify_terminator(
        &mut self,
        terminator: &wire::TerminatorProto,
        values: &[ValueFact<'a>],
        context: ControlContext<'a>,
    ) -> Result<(), LocalVerificationError> {
        self.work.charge_control(1);
        match terminator.kind.as_str() {
            "return" => {
                if values.len() != 1 || !self.type_assignable(self.result_type, values[0].type_id) {
                    return Err(LocalVerificationError::ReturnTypeMismatch);
                }
            }
            "unreachable" => {
                if !values.is_empty() {
                    return Err(LocalVerificationError::UnreachableHasOperands);
                }
            }
            "break" | "continue" => {
                let Some(results) = context.loop_results else {
                    return Err(LocalVerificationError::InvalidTerminatorContext {
                        kind: terminator.kind.clone(),
                    });
                };
                let expected = self.facts(results)?;
                if values.len() != expected.len()
                    || values
                        .iter()
                        .zip(&expected)
                        .any(|(value, result)| !self.same_topology(*value, *result))
                {
                    return Err(LocalVerificationError::InvalidLoopExitTopology {
                        kind: terminator.kind.clone(),
                    });
                }
            }
            "yield" => {
                if context.role == BlockRole::Root {
                    return Err(LocalVerificationError::InvalidTerminatorContext {
                        kind: terminator.kind.clone(),
                    });
                }
            }
            "external" => {
                return Err(LocalVerificationError::InvalidTerminatorContext {
                    kind: terminator.kind.clone(),
                });
            }
            _ => unreachable!("operation vocabulary pass checked terminators"),
        }
        if context.role == BlockRole::Root
            && !matches!(terminator.kind.as_str(), "return" | "unreachable")
        {
            return Err(LocalVerificationError::InvalidStructuredShape(format!(
                "Q1016: function ends with {:?}, not return",
                terminator.kind
            )));
        }
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn verify_operation_regions(
        &mut self,
        operation: &'a wire::OperationProto,
        operands: &[ValueFact<'a>],
        enclosing_loop: Option<&'a [wire::ValueProto]>,
    ) -> Result<(), LocalVerificationError> {
        self.work.charge_control(1);
        match operation.opcode.as_str() {
            "qcore.if" => self.verify_if(operation, operands, enclosing_loop),
            "qcore.for" => self.verify_for(operation, operands, enclosing_loop),
            "qcore.while" => self.verify_while(operation, operands, enclosing_loop),
            _ => Ok(()),
        }
    }

    fn verify_if(
        &mut self,
        operation: &'a wire::OperationProto,
        operands: &[ValueFact<'a>],
        enclosing_loop: Option<&'a [wire::ValueProto]>,
    ) -> Result<(), LocalVerificationError> {
        self.charge_structured_control(operation);
        if operation.regions.len() != 2 || operands.is_empty() {
            return Err(LocalVerificationError::InvalidStructuredShape(
                "Q1005: qcore.if requires a condition and two regions".to_owned(),
            ));
        }
        if !self.is_bool_or_bit(operands[0].type_id) {
            return Err(LocalVerificationError::InvalidControlType(
                "Q1039: qcore.if condition must be Bool or Bit".to_owned(),
            ));
        }
        let stage = self.availability(operands[0].type_id);
        Self::require_stage(operation, stage)?;
        let results = self.facts(&operation.results)?;
        verify_structured_attributes(operation, operands.len() - 1, results.len())?;
        for region in &operation.regions {
            let block = region_entry(region, operation)?;
            self.verify_region_arguments(&block.arguments, &operands[1..], operation)?;
            let yielded = self.verify_block(
                block,
                HashMap::new(),
                ControlContext {
                    role: BlockRole::Region,
                    loop_results: enclosing_loop,
                },
            )?;
            if block
                .terminator
                .as_ref()
                .is_some_and(|terminator| terminator.kind == "yield")
            {
                self.verify_topology(&yielded, &results, operation, false)?;
            }
        }
        self.verify_if_result_stages(operation, &results, stage)
    }

    fn verify_while(
        &mut self,
        operation: &'a wire::OperationProto,
        carries: &[ValueFact<'a>],
        enclosing_loop: Option<&'a [wire::ValueProto]>,
    ) -> Result<(), LocalVerificationError> {
        self.charge_structured_control(operation);
        if operation.regions.len() != 3 {
            return Err(LocalVerificationError::InvalidStructuredShape(
                "Q1006: qcore.while requires condition, body, and else regions".to_owned(),
            ));
        }
        let results = self.facts(&operation.results)?;
        verify_structured_attributes(operation, carries.len(), results.len())?;
        let all_paths_return = results.is_empty()
            && block_exits_callable(region_entry(&operation.regions[2], operation)?)
            && !while_body_has_break(region_entry(&operation.regions[1], operation)?);
        if !all_paths_return {
            self.verify_topology(&results, carries, operation, true)?;
        }
        let condition = region_entry(&operation.regions[0], operation)?;
        self.verify_region_arguments(&condition.arguments, carries, operation)?;
        let yielded = self.verify_block(
            condition,
            HashMap::new(),
            ControlContext {
                role: BlockRole::Region,
                loop_results: None,
            },
        )?;
        if condition
            .terminator
            .as_ref()
            .is_none_or(|value| value.kind != "yield")
            || yielded.len() != carries.len() + 1
            || !self.is_bool_or_bit(yielded[0].type_id)
        {
            return Err(LocalVerificationError::InvalidControlType(
                "Q1039: while condition must yield Bool or Bit followed by its carries".to_owned(),
            ));
        }
        self.verify_topology(&yielded[1..], carries, operation, false)?;
        let stage = self.availability(yielded[0].type_id);
        Self::require_stage(operation, stage)?;
        for index in 1..3 {
            let block = region_entry(&operation.regions[index], operation)?;
            self.verify_region_arguments(&block.arguments, carries, operation)?;
            let yielded = self.verify_block(
                block,
                HashMap::new(),
                ControlContext {
                    role: BlockRole::Region,
                    loop_results: if index == 1 {
                        Some(if all_paths_return {
                            &condition.arguments
                        } else {
                            &operation.results
                        })
                    } else {
                        enclosing_loop
                    },
                },
            )?;
            if block.terminator.as_ref().is_some_and(|value| {
                value.kind == "yield"
                    || (index == 1 && matches!(value.kind.as_str(), "break" | "continue"))
            }) {
                self.verify_topology(
                    &yielded,
                    if all_paths_return { carries } else { &results },
                    operation,
                    false,
                )?;
            }
        }
        for (index, result) in results.iter().enumerate() {
            let forwarded = operation
                .regions
                .iter()
                .enumerate()
                .all(|(region_index, region)| {
                    let block = &region.blocks[0];
                    block.terminator.as_ref().is_some_and(|exit| {
                        if matches!(exit.kind.as_str(), "return" | "unreachable") {
                            return true;
                        }
                        let yielded = exit.operands.get(index + usize::from(region_index == 0));
                        yielded.map(|id| self.forwarded_values.get(id).copied().unwrap_or(*id))
                            == block.arguments.get(index).map(|argument| argument.id)
                    })
                });
            self.verify_control_result_stage(operation, *result, forwarded, stage)?;
            if forwarded {
                let input = operation.operands[index];
                let origin = self.forwarded_values.get(&input).copied().unwrap_or(input);
                self.forwarded_values.insert(result.source.id, origin);
            }
        }
        Ok(())
    }

    fn verify_for(
        &mut self,
        operation: &'a wire::OperationProto,
        operands: &[ValueFact<'a>],
        enclosing_loop: Option<&'a [wire::ValueProto]>,
    ) -> Result<(), LocalVerificationError> {
        self.charge_structured_control(operation);
        if !matches!(operation.regions.len(), 1 | 2) || operands.len() < 3 {
            return Err(LocalVerificationError::InvalidStructuredShape(
                "Q1006: malformed qcore.for operation".to_owned(),
            ));
        }
        if reverse_attribute(operation).is_none() {
            return Err(LocalVerificationError::InvalidStructuredShape(
                "Q1006: qcore.for reverse must be Boolean".to_owned(),
            ));
        }
        if operands[..3]
            .iter()
            .any(|value| !self.is_int(value.type_id))
        {
            return Err(LocalVerificationError::InvalidControlType(
                "Q1041: qcore.for bounds must be Int".to_owned(),
            ));
        }
        let stage = operands[..3]
            .iter()
            .map(|value| self.availability(value.type_id))
            .max()
            .expect("three bounds have a maximum stage");
        Self::require_stage(operation, stage)?;
        let carries = &operands[3..];
        let results = self.facts(&operation.results)?;
        verify_structured_attributes(operation, carries.len(), results.len())?;

        let body = region_entry(&operation.regions[0], operation)?;
        let otherwise = operation
            .regions
            .get(1)
            .map(|region| region_entry(region, operation))
            .transpose()?;
        let all_paths_return = results.is_empty()
            && block_exits_callable(body)
            && otherwise.is_some_and(block_exits_callable);
        if results.is_empty() && !carries.is_empty() && !all_paths_return {
            return Err(LocalVerificationError::InvalidStructuredShape(
                "Q1007: result-free qcore.for can fall through".to_owned(),
            ));
        }
        if !all_paths_return {
            self.verify_topology(&results, carries, operation, true)?;
        }
        if body.arguments.len() != carries.len() + 1 {
            return Err(LocalVerificationError::InvalidStructuredShape(
                "Q1007: qcore.for body argument mismatch".to_owned(),
            ));
        }
        let induction = self.fact(&body.arguments[0])?;
        if !self.is_int(induction.type_id) || self.availability(induction.type_id) != stage {
            return Err(LocalVerificationError::InvalidControlType(
                "Q1041: qcore.for induction value must be Int".to_owned(),
            ));
        }
        self.verify_region_arguments(&body.arguments[1..], carries, operation)?;
        let yielded = self.verify_block(
            body,
            HashMap::new(),
            ControlContext {
                role: BlockRole::Region,
                loop_results: Some(&operation.results),
            },
        )?;
        if body.terminator.as_ref().is_some_and(|terminator| {
            matches!(terminator.kind.as_str(), "yield" | "continue" | "break")
        }) {
            self.verify_topology(&yielded, &results, operation, false)?;
        }

        if let Some(otherwise) = otherwise {
            self.verify_region_arguments(
                &otherwise.arguments,
                if all_paths_return { carries } else { &results },
                operation,
            )?;
            let yielded = self.verify_block(
                otherwise,
                HashMap::new(),
                ControlContext {
                    role: BlockRole::Region,
                    loop_results: enclosing_loop,
                },
            )?;
            if otherwise
                .terminator
                .as_ref()
                .is_some_and(|terminator| terminator.kind == "yield")
            {
                self.verify_topology(&yielded, &results, operation, false)?;
            }
        }
        if all_paths_return {
            Ok(())
        } else {
            self.verify_loop_result_stages(operation, &results, stage, 1)
        }
    }

    fn facts(
        &mut self,
        values: &'a [wire::ValueProto],
    ) -> Result<Vec<ValueFact<'a>>, LocalVerificationError> {
        values.iter().map(|value| self.fact(value)).collect()
    }

    fn verify_region_arguments(
        &mut self,
        arguments: &'a [wire::ValueProto],
        sources: &[ValueFact<'a>],
        operation: &wire::OperationProto,
    ) -> Result<(), LocalVerificationError> {
        let arguments = self.facts(arguments)?;
        self.verify_topology(&arguments, sources, operation, false)
            .map_err(|_| LocalVerificationError::RegionSignatureMismatch {
                opcode: operation.opcode.clone(),
            })
    }

    fn verify_topology(
        &mut self,
        values: &[ValueFact<'a>],
        sources: &[ValueFact<'a>],
        operation: &wire::OperationProto,
        result_topology: bool,
    ) -> Result<(), LocalVerificationError> {
        let matches = values.len() == sources.len()
            && values
                .iter()
                .zip(sources)
                .all(|(value, source)| self.same_topology(*value, *source));
        if matches {
            return Ok(());
        }
        if result_topology {
            Err(LocalVerificationError::ResultTopologyMismatch {
                opcode: operation.opcode.clone(),
            })
        } else {
            Err(LocalVerificationError::YieldResultMismatch {
                opcode: operation.opcode.clone(),
            })
        }
    }

    fn same_topology(&mut self, left: ValueFact<'a>, right: ValueFact<'a>) -> bool {
        self.work.visit_topology();
        left.type_id == right.type_id
            && (!self.arena.is_quantum(left.type_id)
                || left.source.region.as_ref() == right.source.region.as_ref())
    }

    fn is_bool_or_bit(&self, type_id: TypeId) -> bool {
        matches!(
            self.arena.get(self.arena.strip_stage(type_id)),
            Some(SemanticType::Scalar(ScalarType::Bool | ScalarType::Bit))
        )
    }

    fn is_int(&self, type_id: TypeId) -> bool {
        matches!(
            self.arena.get(self.arena.strip_stage(type_id)),
            Some(SemanticType::Scalar(ScalarType::Int))
        )
    }

    fn availability(&self, type_id: TypeId) -> Availability {
        match self.arena.stage_of(type_id) {
            None => Availability::Value,
            Some(Stage::Const) => Availability::Const,
            Some(Stage::Param) => Availability::Param,
            Some(Stage::Shot) => Availability::Shot,
        }
    }

    fn require_stage(
        operation: &wire::OperationProto,
        expected: Availability,
    ) -> Result<(), LocalVerificationError> {
        if string_attribute(operation, "stage") == Some(expected.as_str()) {
            Ok(())
        } else {
            Err(LocalVerificationError::InvalidControlStage {
                opcode: operation.opcode.clone(),
            })
        }
    }

    fn verify_if_result_stages(
        &mut self,
        operation: &'a wire::OperationProto,
        results: &[ValueFact<'a>],
        control_stage: Availability,
    ) -> Result<(), LocalVerificationError> {
        for (index, result) in results.iter().enumerate() {
            self.work.charge_control(1);
            let mut yielding_count = 0_usize;
            let mut first_position = None;
            let mut same_position = true;
            for block in operation
                .regions
                .iter()
                .filter_map(|region| region.blocks.first())
                .filter(|block| {
                    block
                        .terminator
                        .as_ref()
                        .is_some_and(|terminator| terminator.kind == "yield")
                })
            {
                self.work.charge_control(1);
                yielding_count += 1;
                let position = block
                    .terminator
                    .as_ref()
                    .and_then(|terminator| terminator.operands.get(index))
                    .and_then(|yielded| {
                        self.work.charge_control(block.arguments.len());
                        block.arguments.iter().position(|argument| {
                            argument.id
                                == self
                                    .forwarded_values
                                    .get(yielded)
                                    .copied()
                                    .unwrap_or(*yielded)
                        })
                    });
                if yielding_count == 1 {
                    first_position = position;
                } else {
                    same_position &= position == first_position;
                }
            }
            let forwarded = yielding_count == operation.regions.len()
                && first_position.is_some()
                && same_position;
            self.verify_control_result_stage(operation, *result, forwarded, control_stage)?;
            if forwarded {
                // Nested conditionals may return an unchanged enclosing capture.
                // Record only proven identical inputs on every normal branch.
                let input = operation.operands[first_position.expect("forwarded capture") + 1];
                let origin = self.forwarded_values.get(&input).copied().unwrap_or(input);
                self.forwarded_values.insert(result.source.id, origin);
            }
        }
        Ok(())
    }

    fn verify_loop_result_stages(
        &mut self,
        operation: &'a wire::OperationProto,
        results: &[ValueFact<'a>],
        control_stage: Availability,
        carry_offset: usize,
    ) -> Result<(), LocalVerificationError> {
        let regions = operation.regions.as_slice();
        for (index, result) in results.iter().enumerate() {
            self.work.charge_control(1);
            let mut forwarded = true;
            let mut observed_exit = false;
            for (region_index, region) in regions.iter().enumerate() {
                self.work.charge_control(1);
                let block = region
                    .blocks
                    .first()
                    .expect("region shape was checked before stage analysis");
                let terminator = block
                    .terminator
                    .as_ref()
                    .expect("operation vocabulary pass requires every terminator");
                if !matches!(terminator.kind.as_str(), "yield" | "break" | "continue") {
                    continue;
                }
                observed_exit = true;
                let argument_index = if region_index == 0 {
                    index + carry_offset
                } else {
                    index
                };
                forwarded &= terminator.operands.get(index)
                    == block
                        .arguments
                        .get(argument_index)
                        .map(|argument| &argument.id);
            }
            self.verify_control_result_stage(
                operation,
                *result,
                forwarded && observed_exit,
                control_stage,
            )?;
        }
        Ok(())
    }

    fn verify_control_result_stage(
        &mut self,
        operation: &wire::OperationProto,
        result: ValueFact<'a>,
        forwarded: bool,
        control_stage: Availability,
    ) -> Result<(), LocalVerificationError> {
        self.work.charge_control(1);
        if self.arena.is_quantum(result.type_id)
            || forwarded
            || self.availability(result.type_id) >= control_stage
        {
            Ok(())
        } else {
            Err(LocalVerificationError::MissingControlDependency {
                opcode: operation.opcode.clone(),
            })
        }
    }

    fn charge_structured_control(&mut self, operation: &wire::OperationProto) {
        self.work.charge_control(
            operation
                .attributes
                .len()
                .saturating_add(operation.regions.len()),
        );
    }

    fn type_assignable(&mut self, expected: TypeId, actual: TypeId) -> bool {
        if let Some(result) = self.assignability.get(&(expected, actual)) {
            return *result;
        }
        let result = is_assignable(self.arena, expected, actual);
        self.assignability.insert((expected, actual), result);
        result
    }
}

#[derive(Default)]
struct LiveRegions<'a> {
    roots: HashMap<u64, LiveRegionNode<'a>>,
}

#[derive(Default)]
struct LiveRegionNode<'a> {
    live_subtree: usize,
    occupied: bool,
    children: HashMap<&'a str, LiveRegionNode<'a>>,
}

impl<'a> LiveRegions<'a> {
    /// Insert one owner, returning false when an ancestor, descendant, or the
    /// exact same region is already live. Sibling projections remain disjoint.
    fn insert(&mut self, region: &'a wire::RegionRefProto) -> bool {
        let root = self.roots.entry(region.root).or_default();
        if root.overlaps(&region.projections) {
            return false;
        }
        root.insert_path(&region.projections);
        true
    }

    fn remove(&mut self, region: &'a wire::RegionRefProto) {
        if let Some(root) = self.roots.get_mut(&region.root) {
            root.remove_path(&region.projections);
        }
    }
}

impl<'a> LiveRegionNode<'a> {
    fn overlaps(&self, path: &[String]) -> bool {
        if self.occupied {
            return true;
        }
        let Some((component, rest)) = path.split_first() else {
            return self.live_subtree != 0;
        };
        self.children
            .get(component.as_str())
            .is_some_and(|child| child.overlaps(rest))
    }

    fn insert_path(&mut self, path: &'a [String]) {
        self.live_subtree += 1;
        if let Some((component, rest)) = path.split_first() {
            self.children
                .entry(component.as_str())
                .or_default()
                .insert_path(rest);
        } else {
            self.occupied = true;
        }
    }

    fn remove_path(&mut self, path: &[String]) -> bool {
        if let Some((component, rest)) = path.split_first() {
            let Some(child) = self.children.get_mut(component.as_str()) else {
                return false;
            };
            if !child.remove_path(rest) {
                return false;
            }
        } else if self.occupied {
            self.occupied = false;
        } else {
            return false;
        }
        self.live_subtree -= 1;
        true
    }
}

/// Detect only breaks belonging to this loop, including nested loop else arms.
fn while_body_has_break(body: &wire::BlockProto) -> bool {
    let mut blocks = vec![body];
    while let Some(block) = blocks.pop() {
        if block
            .terminator
            .as_ref()
            .is_some_and(|exit| exit.kind == "break")
        {
            return true;
        }
        for operation in &block.operations {
            let regions = match operation.opcode.as_str() {
                "qcore.for" => operation.regions.get(1..).unwrap_or_default(),
                "qcore.while" => operation.regions.get(2..).unwrap_or_default(),
                _ => &operation.regions,
            };
            for region in regions {
                blocks.extend(&region.blocks);
            }
        }
    }
    false
}

fn verify_structured_attributes(
    operation: &wire::OperationProto,
    capture_count: usize,
    result_count: usize,
) -> Result<(), LocalVerificationError> {
    let valid = match operation.opcode.as_str() {
        "qcore.if" => {
            let base = ["capture_names", "condition_index", "output_names", "stage"];
            let inverse = [
                "body_mode",
                "capture_names",
                "condition_index",
                "original_attributes",
                "output_names",
                "proof",
                "stage",
            ];
            (exact_attribute_names(operation, &base)
                || (exact_attribute_names(operation, &inverse)
                    && string_attribute(operation, "body_mode") == Some("adjoint")
                    && string_attribute(operation, "proof") == Some("modular-adjoint")
                    && mapping_attribute(operation, "original_attributes")))
                && string_tuple_attribute_len(operation, "capture_names") == Some(capture_count)
                && int_attribute(operation, "condition_index") == Some(0)
                && string_tuple_attribute_len(operation, "output_names") == Some(result_count)
                && string_attribute(operation, "stage").is_some()
        }
        "qcore.while" => {
            exact_attribute_names(operation, &["capture_names", "stage"])
                && string_tuple_attribute_len(operation, "capture_names") == Some(capture_count)
                && (result_count == capture_count || result_count == 0)
                && string_attribute(operation, "stage").is_some()
        }
        "qcore.for" => {
            let base = ["capture_names", "induction_name", "stage"];
            let with_reverse = ["capture_names", "induction_name", "reverse", "stage"];
            let inverse = [
                "body_mode",
                "capture_names",
                "induction_name",
                "original_attributes",
                "proof",
                "reverse",
                "stage",
            ];
            ((exact_attribute_names(operation, &base)
                || exact_attribute_names(operation, &with_reverse))
                || (exact_attribute_names(operation, &inverse)
                    && string_attribute(operation, "body_mode") == Some("adjoint")
                    && string_attribute(operation, "proof") == Some("modular-adjoint")
                    && mapping_attribute(operation, "original_attributes")))
                && string_tuple_attribute_len(operation, "capture_names") == Some(capture_count)
                && string_attribute(operation, "induction_name")
                    .is_some_and(|name| !name.is_empty())
                && string_attribute(operation, "stage").is_some()
                && reverse_attribute(operation).is_some()
        }
        _ => true,
    };
    if valid {
        Ok(())
    } else {
        Err(LocalVerificationError::InvalidStructuredAttributes {
            opcode: operation.opcode.clone(),
        })
    }
}

fn mapping_attribute(operation: &wire::OperationProto, name: &str) -> bool {
    use wire::literal_proto::Value;

    operation
        .attributes
        .iter()
        .find(|attribute| attribute.name == name)
        .and_then(|attribute| attribute.value.as_ref())
        .is_some_and(|literal| matches!(literal.value, Some(Value::MappingValue(_))))
}

fn exact_attribute_names(operation: &wire::OperationProto, expected: &[&str]) -> bool {
    operation.attributes.len() == expected.len()
        && operation
            .attributes
            .iter()
            .map(|attribute| attribute.name.as_str())
            .eq(expected.iter().copied())
}

fn int_attribute(operation: &wire::OperationProto, name: &str) -> Option<i64> {
    use wire::literal_proto::Value;

    operation
        .attributes
        .iter()
        .find(|attribute| attribute.name == name)
        .and_then(|attribute| attribute.value.as_ref())
        .and_then(|literal| match literal.value.as_ref() {
            Some(Value::IntValue(value)) => Some(*value),
            _ => None,
        })
}

fn string_tuple_attribute_len(operation: &wire::OperationProto, name: &str) -> Option<usize> {
    use wire::literal_proto::Value;

    let literal = operation
        .attributes
        .iter()
        .find(|attribute| attribute.name == name)?
        .value
        .as_ref()?;
    let Value::TupleValue(sequence) = literal.value.as_ref()? else {
        return None;
    };
    let mut names = HashSet::with_capacity(sequence.values.len());
    for value in &sequence.values {
        let Some(Value::StringValue(name)) = value.value.as_ref() else {
            return None;
        };
        if name.is_empty() || !names.insert(name.as_str()) {
            return None;
        }
    }
    Some(sequence.values.len())
}

fn block_exits_callable(block: &wire::BlockProto) -> bool {
    block
        .terminator
        .as_ref()
        .is_some_and(|terminator| matches!(terminator.kind.as_str(), "return" | "unreachable"))
}

fn region_entry<'a>(
    region: &'a wire::RegionProto,
    operation: &wire::OperationProto,
) -> Result<&'a wire::BlockProto, LocalVerificationError> {
    let [block] = region.blocks.as_slice() else {
        return Err(LocalVerificationError::InvalidRegionBlockCount {
            opcode: operation.opcode.clone(),
        });
    };
    Ok(block)
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

fn reverse_attribute(operation: &wire::OperationProto) -> Option<bool> {
    use wire::literal_proto::Value;

    let Some(attribute) = operation
        .attributes
        .iter()
        .find(|attribute| attribute.name == "reverse")
    else {
        return Some(false);
    };
    match attribute
        .value
        .as_ref()
        .and_then(|literal| literal.value.as_ref())
    {
        Some(Value::BoolValue(value)) => Some(*value),
        _ => None,
    }
}
