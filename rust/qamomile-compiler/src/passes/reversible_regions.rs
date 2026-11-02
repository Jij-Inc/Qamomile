//! Derived closed reversible regions for native cleanup realization.
//!
//! Region formation is intentionally separate from inverse emission. It walks
//! checked affine producers once, classifies the closed reversible vocabulary,
//! and returns an immutable transient plan. The cleanup proof kernel remains an
//! independent consumer of the materialized program and does not trust these
//! facts as release evidence.

use std::collections::{HashMap, HashSet};

use crate::budget::MAX_MESSAGE_NODES;
use crate::operations::{LocallyCheckedOperations, Opcode, QCoreOpcode, QuantumOpcode};
use crate::types::{TypeArena, TypeId};
use crate::wire;

/// Maximum producer and owned-edge visits during one cleanup preparation.
const MAX_REVERSIBLE_REGION_WORK: usize = MAX_MESSAGE_NODES.saturating_mul(4);

/// Failure to form one closed reversible owned region.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ReversibleRegionError {
    /// A producer requires a reversible form that is not supported.
    UnsupportedProducer { opcode: String },
    /// The selected owned producer graph is not one closed region.
    InvalidDiagram { reason: &'static str },
    /// An owned value has no checked local definition.
    MissingDefinition,
    /// A called function has no verified inverse realization.
    MissingAdjoint { callee: String },
    /// Region formation exceeded its explicit work budget.
    WorkBudgetExceeded,
}

/// Compiler work charged while closed reversible regions are formed.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct ReversibleRegionWork {
    visited: usize,
}

impl ReversibleRegionWork {
    fn charge(&mut self, amount: usize) -> Result<(), ReversibleRegionError> {
        self.visited = self
            .visited
            .checked_add(amount)
            .ok_or(ReversibleRegionError::WorkBudgetExceeded)?;
        if self.visited > MAX_REVERSIBLE_REGION_WORK {
            return Err(ReversibleRegionError::WorkBudgetExceeded);
        }
        Ok(())
    }
}

/// Closed classification of one reversible-region operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReversibleStepKind {
    /// Owned tuple construction.
    Tuple,
    /// Owned tuple destruction.
    Unpack,
    /// Ownership-preserving reshape.
    Reshape,
    /// Primitive exact quantum gate.
    Primitive,
    /// Ordinary exact callable application.
    Call,
    /// Ordinary callable application inverted by history replay.
    HistoryCall,
    /// Exact adjoint callable application.
    AdjointCall,
    /// Coherent controlled callable application.
    ControlCall,
    /// Coherent selected callable application.
    SelectCall,
    /// Structured exact conditional.
    If,
    /// Structured finite loop.
    For,
}

/// Exact semantic classifier used while a reversible region is formed.
pub(crate) trait ReversibleStepClassifier {
    /// Classify one checked producer into the exact realization vocabulary.
    fn classify(
        &self,
        operation: &wire::OperationProto,
        definitions: &HashMap<u64, &wire::ValueProto>,
        value_types: &HashMap<u64, TypeId>,
        arena: &TypeArena,
    ) -> Result<ReversibleStepKind, ReversibleRegionError>;
}

/// One checked operation retained by a reversible-region plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ReversibleStep {
    kind: ReversibleStepKind,
    producer_key: u64,
}

impl ReversibleStep {
    /// Return this step's closed reversible classification.
    pub(crate) const fn kind(&self) -> ReversibleStepKind {
        self.kind
    }

    /// Return the first-result identity that locates the checked producer.
    pub(crate) const fn producer_key(&self) -> u64 {
        self.producer_key
    }
}

/// One immutable transient plan for an owned cleanup obligation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ReversibleRegionPlan {
    origin: u64,
    steps: Box<[ReversibleStep]>,
    roots: Box<[ReversibleRoot]>,
}

/// Exact terminal required after one reversible root is reconstructed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReversibleRootKind {
    /// A zero-state allocation must become a release.
    Allocation,
    /// A coherent preparation must become an unprepare.
    Preparation,
}

/// One allocation or preparation root closed by a reversible region.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ReversibleRoot {
    value: u64,
    kind: ReversibleRootKind,
}

impl ReversibleRoot {
    /// Return the owned root value identity.
    pub(crate) const fn value(self) -> u64 {
        self.value
    }

    /// Return the exact terminal kind required for this root.
    pub(crate) const fn kind(self) -> ReversibleRootKind {
        self.kind
    }
}

/// Derived reversible-region plans for every checked function.
#[derive(Debug)]
pub(crate) struct ReversibleProgramPlan<'a> {
    functions: HashMap<&'a str, ReversibleFunctionPlan<'a>>,
}

impl ReversibleProgramPlan<'_> {
    /// Resolve one function's derived reversible-region plan.
    pub(crate) fn function(&self, symbol: &str) -> Option<&ReversibleFunctionPlan<'_>> {
        self.functions.get(symbol)
    }
}

/// Shared checked index and cleanup regions for one function.
#[derive(Debug)]
pub(crate) struct ReversibleFunctionPlan<'a> {
    definitions: HashMap<u64, &'a wire::ValueProto>,
    producers: HashMap<u64, &'a wire::OperationProto>,
    regions: HashMap<u64, ReversibleRegionPlan>,
}

impl ReversibleFunctionPlan<'_> {
    /// Resolve one function-local checked value definition.
    pub(crate) fn definition(&self, identifier: u64) -> Option<&wire::ValueProto> {
        self.definitions.get(&identifier).copied()
    }

    /// Resolve one checked producer by its first-result identity.
    pub(crate) fn producer(&self, producer_key: u64) -> Option<&wire::OperationProto> {
        self.producers.get(&producer_key).copied()
    }

    /// Resolve the region derived for one cleanup origin.
    pub(crate) fn region(&self, origin: u64) -> Option<&ReversibleRegionPlan> {
        self.regions.get(&origin)
    }
}

impl ReversibleRegionPlan {
    /// Return the forgotten owned value that selected this region.
    pub(crate) const fn origin(&self) -> u64 {
        self.origin
    }

    /// Return reversible steps in forward topological order.
    pub(crate) const fn steps(&self) -> &[ReversibleStep] {
        &self.steps
    }

    /// Return allocation and preparation roots in deterministic traversal order.
    pub(crate) const fn roots(&self) -> &[ReversibleRoot] {
        &self.roots
    }
}

#[derive(Clone)]
enum RegionWorkItem {
    Visit(u64),
    Emit(ReversibleStep),
}

struct RegionContext<'data, 'index> {
    definitions: &'index HashMap<u64, &'data wire::ValueProto>,
    producer_for_value: &'index HashMap<u64, u64>,
    producers: &'index HashMap<u64, &'data wire::OperationProto>,
    value_types: &'index HashMap<u64, TypeId>,
    arena: &'index TypeArena,
    classifier: &'index dyn ReversibleStepClassifier,
}

/// Form all reversible-region plans from one locally checked program.
///
/// # Errors
///
/// Returns [`ReversibleRegionError`] when checked function facts are missing or
/// any cleanup obligation cannot form a closed reversible region.
pub(crate) fn form_program_plan<'a>(
    package: &'a wire::QCorePackageProto,
    operations: &LocallyCheckedOperations,
    arena: &TypeArena,
    classifier: &impl ReversibleStepClassifier,
) -> Result<ReversibleProgramPlan<'a>, ReversibleRegionError> {
    let mut work = ReversibleRegionWork::default();
    let mut functions = HashMap::with_capacity(package.functions.len());
    for function in &package.functions {
        let value_types = operations
            .value_types(&function.symbol)
            .ok_or(ReversibleRegionError::MissingDefinition)?;
        functions.insert(
            function.symbol.as_str(),
            form_function_plan(function, value_types, arena, classifier, &mut work)?,
        );
    }
    Ok(ReversibleProgramPlan { functions })
}

fn form_function_plan<'a>(
    function: &'a wire::FunctionProto,
    value_types: &HashMap<u64, TypeId>,
    arena: &TypeArena,
    classifier: &impl ReversibleStepClassifier,
    work: &mut ReversibleRegionWork,
) -> Result<ReversibleFunctionPlan<'a>, ReversibleRegionError> {
    let mut definitions = function
        .parameters
        .iter()
        .filter_map(|parameter| parameter.value.as_ref())
        .map(|value| (value.id, value))
        .collect::<HashMap<_, _>>();
    let mut producers = HashMap::new();
    let mut blocks = function.body.iter().collect::<Vec<_>>();
    while let Some(block) = blocks.pop() {
        work.charge(1_usize.saturating_add(block.operations.len()))?;
        definitions.extend(block.arguments.iter().map(|value| (value.id, value)));
        for operation in &block.operations {
            definitions.extend(operation.results.iter().map(|value| (value.id, value)));
            if let Some(producer_key) = operation.results.first().map(|result| result.id) {
                producers.insert(producer_key, operation);
            }
            for region in operation.regions.iter().rev() {
                blocks.extend(region.blocks.iter().rev());
            }
        }
    }
    let mut regions = HashMap::new();
    let mut blocks = function.body.iter().collect::<Vec<_>>();
    while let Some(block) = blocks.pop() {
        work.charge(
            1_usize
                .saturating_add(block.operations.len())
                .saturating_add(
                    block
                        .operations
                        .iter()
                        .map(|operation| operation.results.len())
                        .sum(),
                ),
        )?;
        let producer_for_value = block
            .operations
            .iter()
            .filter_map(|operation| operation.results.first().map(|first| (first.id, operation)))
            .flat_map(|(producer_key, operation)| {
                operation
                    .results
                    .iter()
                    .map(move |result| (result.id, producer_key))
            })
            .collect::<HashMap<_, _>>();
        for operation in &block.operations {
            if matches!(
                Opcode::try_from(operation.opcode.as_str()),
                Ok(Opcode::QCore(QCoreOpcode::Forget))
            ) {
                let origin = operation
                    .operands
                    .first()
                    .copied()
                    .ok_or(ReversibleRegionError::MissingDefinition)?;
                let context = RegionContext {
                    definitions: &definitions,
                    producer_for_value: &producer_for_value,
                    producers: &producers,
                    value_types,
                    arena,
                    classifier,
                };
                let region = form_region(origin, &context, work)?;
                regions.insert(origin, region);
            }
            for region in operation.regions.iter().rev() {
                blocks.extend(region.blocks.iter().rev());
            }
        }
    }
    Ok(ReversibleFunctionPlan {
        definitions,
        producers,
        regions,
    })
}

/// Form the closed reversible region ending at one forgotten owned value.
///
/// # Errors
///
/// Returns [`ReversibleRegionError`] when the producer graph is missing,
/// branches outside the selected owned result, contains an unsupported step,
/// has no allocation root, or exceeds the work budget.
fn form_region(
    origin: u64,
    context: &RegionContext<'_, '_>,
    work: &mut ReversibleRegionWork,
) -> Result<ReversibleRegionPlan, ReversibleRegionError> {
    let mut pending = vec![RegionWorkItem::Visit(origin)];
    let mut scheduled = HashSet::new();
    let mut steps = Vec::new();
    let mut roots = Vec::new();
    let mut seen_roots = HashSet::new();
    let mut selected_values = HashSet::new();
    while let Some(item) = pending.pop() {
        work.charge(1)?;
        match item {
            RegionWorkItem::Emit(step) => steps.push(step),
            RegionWorkItem::Visit(identifier) => {
                selected_values.insert(identifier);
                let producer_key = context
                    .producer_for_value
                    .get(&identifier)
                    .copied()
                    .ok_or(ReversibleRegionError::MissingDefinition)?;
                let producer = context
                    .producers
                    .get(&producer_key)
                    .ok_or(ReversibleRegionError::MissingDefinition)?;
                let root_kind = match Opcode::try_from(producer.opcode.as_str()) {
                    Ok(Opcode::Quantum(QuantumOpcode::Alloc)) => {
                        Some(ReversibleRootKind::Allocation)
                    }
                    Ok(Opcode::Quantum(QuantumOpcode::Prepare)) => {
                        Some(ReversibleRootKind::Preparation)
                    }
                    _ => None,
                };
                if let Some(kind) = root_kind {
                    if seen_roots.insert(identifier) {
                        roots.push(ReversibleRoot {
                            value: identifier,
                            kind,
                        });
                    }
                    continue;
                }
                let operation_key = producer.results.first().map(|result| result.id).ok_or(
                    ReversibleRegionError::InvalidDiagram {
                        reason: "contains a producer without a result",
                    },
                )?;
                if !scheduled.insert(operation_key) {
                    continue;
                }
                let kind = context.classifier.classify(
                    producer,
                    context.definitions,
                    context.value_types,
                    context.arena,
                )?;
                pending.push(RegionWorkItem::Emit(ReversibleStep {
                    kind,
                    producer_key: operation_key,
                }));
                let operands = owned_operands(producer, context.value_types, context.arena);
                work.charge(operands.len())?;
                pending.extend(operands.into_iter().rev().map(RegionWorkItem::Visit));
            }
        }
    }
    if roots.is_empty() {
        return Err(ReversibleRegionError::InvalidDiagram {
            reason: "does not have an owned allocation or preparation root",
        });
    }
    for step in &steps {
        let producer = context
            .producers
            .get(&step.producer_key)
            .ok_or(ReversibleRegionError::MissingDefinition)?;
        let results = owned_results(producer, context.value_types, context.arena);
        work.charge(results.len())?;
        if results
            .iter()
            .any(|result| !selected_values.contains(&result.id))
        {
            return Err(ReversibleRegionError::InvalidDiagram {
                reason: "has a linear branch outside its returned value",
            });
        }
    }
    Ok(ReversibleRegionPlan {
        origin,
        steps: steps.into_boxed_slice(),
        roots: roots.into_boxed_slice(),
    })
}

/// Classify the structural shape of one potential reversible step.
pub(crate) fn classify_structural_step(
    operation: &wire::OperationProto,
    definitions: &HashMap<u64, &wire::ValueProto>,
    value_types: &HashMap<u64, TypeId>,
    arena: &TypeArena,
) -> Result<ReversibleStepKind, ReversibleRegionError> {
    if operation
        .operands
        .iter()
        .chain(operation.results.iter().map(|result| &result.id))
        .any(|identifier| !definitions.contains_key(identifier))
    {
        return Err(ReversibleRegionError::MissingDefinition);
    }
    let operands = owned_operands(operation, value_types, arena);
    let results = owned_results(operation, value_types, arena);
    if operands.is_empty() || results.is_empty() {
        return Err(ReversibleRegionError::InvalidDiagram {
            reason: "contains a non-isomorphic linear operation",
        });
    }
    let opcode = Opcode::try_from(operation.opcode.as_str()).map_err(|_| {
        ReversibleRegionError::UnsupportedProducer {
            opcode: operation.opcode.clone(),
        }
    })?;
    match opcode {
        Opcode::QCore(QCoreOpcode::Tuple)
            if operands.len() == operation.operands.len()
                && results.len() == operation.results.len() =>
        {
            Ok(ReversibleStepKind::Tuple)
        }
        Opcode::QCore(QCoreOpcode::Unpack)
            if operands.len() == operation.operands.len()
                && results.len() == operation.results.len() =>
        {
            Ok(ReversibleStepKind::Unpack)
        }
        Opcode::QCore(QCoreOpcode::Reshape)
            if operands.len() == operation.operands.len()
                && results.len() == operation.results.len() =>
        {
            Ok(ReversibleStepKind::Reshape)
        }
        Opcode::QCore(QCoreOpcode::Tuple | QCoreOpcode::Unpack | QCoreOpcode::Reshape) => {
            Err(ReversibleRegionError::InvalidDiagram {
                reason: "contains a mixed classical and owned structural operation",
            })
        }
        Opcode::QCore(QCoreOpcode::PhaseBy | QCoreOpcode::XorBy | QCoreOpcode::PermuteBy)
        | Opcode::Quantum(
            QuantumOpcode::Gate(_)
            | QuantumOpcode::GlobalPhase
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
            | QuantumOpcode::ModMul
            | QuantumOpcode::ModDiv
            | QuantumOpcode::ModPow
            | QuantumOpcode::ModPowInverse
            | QuantumOpcode::LessThanXor,
        ) if !operands.is_empty() && results.len() == 1 => Ok(ReversibleStepKind::Primitive),
        Opcode::QCore(QCoreOpcode::Call) if results.len() == 1 => Ok(ReversibleStepKind::Call),
        Opcode::QCore(QCoreOpcode::AdjointCall) if operands.len() == 1 && results.len() == 1 => {
            Ok(ReversibleStepKind::AdjointCall)
        }
        Opcode::QCore(QCoreOpcode::ControlCall) if operands.len() == 2 && results.len() == 1 => {
            Ok(ReversibleStepKind::ControlCall)
        }
        Opcode::QCore(QCoreOpcode::SelectCall) if operands.len() == 2 && results.len() == 1 => {
            Ok(ReversibleStepKind::SelectCall)
        }
        Opcode::QCore(QCoreOpcode::If) if operands.len() == 1 && results.len() == 1 => {
            Ok(ReversibleStepKind::If)
        }
        Opcode::QCore(QCoreOpcode::For) if operands.len() == 1 && results.len() == 1 => {
            Ok(ReversibleStepKind::For)
        }
        _ => Err(ReversibleRegionError::UnsupportedProducer {
            opcode: operation.opcode.clone(),
        }),
    }
}

/// Return owned operands in source order.
pub(crate) fn owned_operands(
    operation: &wire::OperationProto,
    value_types: &HashMap<u64, TypeId>,
    arena: &TypeArena,
) -> Vec<u64> {
    operation
        .operands
        .iter()
        .copied()
        .filter(|operand| {
            value_types
                .get(operand)
                .is_some_and(|type_id| arena.is_quantum(*type_id))
        })
        .collect()
}

/// Return owned results in source order.
pub(crate) fn owned_results<'a>(
    operation: &'a wire::OperationProto,
    value_types: &HashMap<u64, TypeId>,
    arena: &TypeArena,
) -> Vec<&'a wire::ValueProto> {
    operation
        .results
        .iter()
        .filter(|result| {
            value_types
                .get(&result.id)
                .is_some_and(|type_id| arena.is_quantum(*type_id))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ScalarType, SemanticType};

    type TestGraph<'a> = (
        HashMap<u64, &'a wire::ValueProto>,
        HashMap<u64, u64>,
        HashMap<u64, &'a wire::OperationProto>,
    );

    struct StructuralClassifier;

    impl ReversibleStepClassifier for StructuralClassifier {
        fn classify(
            &self,
            operation: &wire::OperationProto,
            definitions: &HashMap<u64, &wire::ValueProto>,
            value_types: &HashMap<u64, TypeId>,
            arena: &TypeArena,
        ) -> Result<ReversibleStepKind, ReversibleRegionError> {
            classify_structural_step(operation, definitions, value_types, arena)
        }
    }

    struct RejectCallsClassifier;

    impl ReversibleStepClassifier for RejectCallsClassifier {
        fn classify(
            &self,
            operation: &wire::OperationProto,
            definitions: &HashMap<u64, &wire::ValueProto>,
            value_types: &HashMap<u64, TypeId>,
            arena: &TypeArena,
        ) -> Result<ReversibleStepKind, ReversibleRegionError> {
            match classify_structural_step(operation, definitions, value_types, arena)? {
                ReversibleStepKind::Call => Err(ReversibleRegionError::MissingAdjoint {
                    callee: "callee".to_owned(),
                }),
                kind => Ok(kind),
            }
        }
    }

    fn quantum_types() -> (TypeArena, TypeId, TypeId) {
        let mut arena = TypeArena::default();
        let bit = arena
            .intern_derived(SemanticType::Scalar(ScalarType::Bit))
            .expect("valid Bit type");
        let pair = arena
            .intern_derived(SemanticType::Product(vec![bit, bit].into_boxed_slice()))
            .expect("valid bit pair");
        let quantum_bit = arena
            .intern_derived(SemanticType::Quantum(bit))
            .expect("valid quantum bit");
        let quantum_pair = arena
            .intern_derived(SemanticType::Quantum(pair))
            .expect("valid quantum pair");
        (arena, quantum_bit, quantum_pair)
    }

    fn value(identifier: u64) -> wire::ValueProto {
        wire::ValueProto {
            id: identifier,
            ..Default::default()
        }
    }

    fn operation(opcode: &str, operands: &[u64], results: &[u64]) -> wire::OperationProto {
        wire::OperationProto {
            opcode: opcode.to_owned(),
            operands: operands.to_vec(),
            results: results.iter().copied().map(value).collect(),
            ..Default::default()
        }
    }

    fn graph(operations: &[wire::OperationProto]) -> TestGraph<'_> {
        let definitions = operations
            .iter()
            .flat_map(|operation| operation.results.iter())
            .map(|result| (result.id, result))
            .collect();
        let producer_for_value = operations
            .iter()
            .filter_map(|operation| operation.results.first().map(|first| (first.id, operation)))
            .flat_map(|(producer_key, operation)| {
                operation
                    .results
                    .iter()
                    .map(move |result| (result.id, producer_key))
            })
            .collect();
        let producers = operations
            .iter()
            .filter_map(|operation| operation.results.first().map(|first| (first.id, operation)))
            .collect();
        (definitions, producer_for_value, producers)
    }

    #[test]
    fn forms_steps_in_forward_order_without_copying_operations_into_the_plan() {
        let (arena, quantum_bit, _) = quantum_types();
        let operations = vec![
            operation("quantum.alloc", &[], &[1]),
            operation("quantum.h", &[1], &[2]),
            operation("quantum.x", &[2], &[3]),
        ];
        let (definitions, producer_for_value, producers) = graph(&operations);
        let value_types = HashMap::from([(1, quantum_bit), (2, quantum_bit), (3, quantum_bit)]);
        let context = RegionContext {
            definitions: &definitions,
            producer_for_value: &producer_for_value,
            producers: &producers,
            value_types: &value_types,
            arena: &arena,
            classifier: &StructuralClassifier,
        };

        let plan = form_region(3, &context, &mut ReversibleRegionWork::default())
            .expect("closed primitive region");

        assert_eq!(plan.origin(), 3);
        assert_eq!(
            plan.roots()
                .iter()
                .copied()
                .map(ReversibleRoot::value)
                .collect::<Vec<_>>(),
            vec![1]
        );
        assert_eq!(plan.roots()[0].kind(), ReversibleRootKind::Allocation);
        assert_eq!(
            plan.steps()
                .iter()
                .map(ReversibleStep::producer_key)
                .collect::<Vec<_>>(),
            vec![2, 3]
        );
        assert!(
            plan.steps()
                .iter()
                .all(|step| step.kind() == ReversibleStepKind::Primitive)
        );
    }

    #[test]
    fn retains_multiple_owned_roots_in_operand_order() {
        let (arena, quantum_bit, quantum_pair) = quantum_types();
        let operations = vec![
            operation("quantum.alloc", &[], &[1]),
            operation("quantum.alloc", &[], &[2]),
            operation("qcore.tuple", &[1, 2], &[3]),
        ];
        let (definitions, producer_for_value, producers) = graph(&operations);
        let value_types = HashMap::from([(1, quantum_bit), (2, quantum_bit), (3, quantum_pair)]);
        let context = RegionContext {
            definitions: &definitions,
            producer_for_value: &producer_for_value,
            producers: &producers,
            value_types: &value_types,
            arena: &arena,
            classifier: &StructuralClassifier,
        };

        let plan = form_region(3, &context, &mut ReversibleRegionWork::default())
            .expect("closed multi-root region");

        assert_eq!(
            plan.roots()
                .iter()
                .copied()
                .map(ReversibleRoot::value)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert_eq!(plan.steps().len(), 1);
        assert_eq!(plan.steps()[0].kind(), ReversibleStepKind::Tuple);
    }

    #[test]
    fn retains_the_exact_preparation_terminal_kind() {
        let (arena, quantum_bit, _) = quantum_types();
        let operations = vec![operation("quantum.prepare", &[], &[1])];
        let (definitions, producer_for_value, producers) = graph(&operations);
        let value_types = HashMap::from([(1, quantum_bit)]);
        let context = RegionContext {
            definitions: &definitions,
            producer_for_value: &producer_for_value,
            producers: &producers,
            value_types: &value_types,
            arena: &arena,
            classifier: &StructuralClassifier,
        };

        let plan = form_region(1, &context, &mut ReversibleRegionWork::default())
            .expect("closed preparation root");

        assert_eq!(plan.roots().len(), 1);
        assert_eq!(plan.roots()[0].value(), 1);
        assert_eq!(plan.roots()[0].kind(), ReversibleRootKind::Preparation);
    }

    #[test]
    fn rejects_an_owned_result_branch_outside_the_selected_region() {
        let (arena, quantum_bit, quantum_pair) = quantum_types();
        let operations = vec![
            operation("quantum.alloc", &[], &[1]),
            operation("qcore.unpack", &[1], &[2, 3]),
        ];
        let (definitions, producer_for_value, producers) = graph(&operations);
        let value_types = HashMap::from([(1, quantum_pair), (2, quantum_bit), (3, quantum_bit)]);
        let context = RegionContext {
            definitions: &definitions,
            producer_for_value: &producer_for_value,
            producers: &producers,
            value_types: &value_types,
            arena: &arena,
            classifier: &StructuralClassifier,
        };

        assert_eq!(
            form_region(2, &context, &mut ReversibleRegionWork::default()),
            Err(ReversibleRegionError::InvalidDiagram {
                reason: "has a linear branch outside its returned value",
            })
        );
    }

    #[test]
    fn rejects_an_irreversible_region_step() {
        let (arena, quantum_bit, _) = quantum_types();
        let operations = vec![
            operation("quantum.alloc", &[], &[1]),
            operation("quantum.measure", &[1], &[2]),
        ];
        let (definitions, producer_for_value, producers) = graph(&operations);
        let value_types = HashMap::from([(1, quantum_bit), (2, quantum_bit)]);
        let context = RegionContext {
            definitions: &definitions,
            producer_for_value: &producer_for_value,
            producers: &producers,
            value_types: &value_types,
            arena: &arena,
            classifier: &StructuralClassifier,
        };

        assert_eq!(
            form_region(2, &context, &mut ReversibleRegionWork::default()),
            Err(ReversibleRegionError::UnsupportedProducer {
                opcode: "quantum.measure".to_owned(),
            })
        );
    }

    #[test]
    fn rejects_a_structural_step_without_an_exact_realization() {
        let (arena, quantum_bit, _) = quantum_types();
        let operations = vec![
            operation("quantum.alloc", &[], &[1]),
            operation("qcore.call", &[1], &[2]),
        ];
        let (definitions, producer_for_value, producers) = graph(&operations);
        let value_types = HashMap::from([(1, quantum_bit), (2, quantum_bit)]);
        let context = RegionContext {
            definitions: &definitions,
            producer_for_value: &producer_for_value,
            producers: &producers,
            value_types: &value_types,
            arena: &arena,
            classifier: &RejectCallsClassifier,
        };

        assert_eq!(
            form_region(2, &context, &mut ReversibleRegionWork::default()),
            Err(ReversibleRegionError::MissingAdjoint {
                callee: "callee".to_owned(),
            })
        );
    }

    #[test]
    fn rejects_region_work_past_the_explicit_limit() {
        let mut work = ReversibleRegionWork {
            visited: MAX_REVERSIBLE_REGION_WORK,
        };

        assert_eq!(
            work.charge(1),
            Err(ReversibleRegionError::WorkBudgetExceeded)
        );
    }
}
