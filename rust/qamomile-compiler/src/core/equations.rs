//! Complete function-local equations outside the classical dialect.
//!
//! The operation-kernel pass resolves every wire type once into the shared
//! [`TypeArena`]. This pass consumes those retained roots after lexical SSA and
//! ownership verification, checks the remaining `qcore` and quantum equations,
//! and records cleanup claims that still need an independent proof-term pass.
//! A textual `proof` attribute is provenance only; it never discharges an
//! obligation or relaxes an ownership equation here.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use num_bigint::{BigInt, Sign};
use thiserror::Error;

use crate::control_state::ControlState;
use crate::core::slice_shape::{SliceBound, operation_bounds};
use crate::operations::{LocallyCheckedOperations, PrimitiveGateSpec, primitive_gate_spec};
use crate::types::{
    CoherentLeafWidth, Dimension, ScalarType, SemanticType, Stage, TypeArena, TypeId, WordKind,
};
use crate::verification::QCoreVerificationWork;
use crate::wire;

/// A malformed function-local non-classical equation.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub(crate) enum LocalEquationError {
    /// A retained type root is unexpectedly absent.
    #[error("Q4002: missing retained type root for SSA value %{value_id}")]
    MissingTypeRoot { value_id: u64 },
    /// A core operation has no unique local interpretation.
    #[error("Q1034: malformed {description}")]
    MalformedCore { description: String },
    /// A quantum operation has no unique local interpretation.
    #[error("Q1035: malformed {opcode} operation")]
    MalformedQuantum { opcode: String },
    /// A quantum transform fails to preserve exactly one owner.
    #[error("Q1036: {opcode} must preserve exactly one owned value")]
    InvalidOwnedTransform { opcode: String },
    /// An allocation width cannot be reconstructed from trusted local data.
    #[error(
        "Q1035: quantum.alloc width must be a positive direct constant or dependent Int binder"
    )]
    UnresolvedAllocationWidth,
}

/// One semantic cleanup claim intentionally left for proof-term verification.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ProofObligation {
    function: Arc<str>,
    opcode: &'static str,
    origin: Option<u64>,
}

/// Retained facts from complete local non-classical equation checking.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct LocalEquationFacts {
    proof_obligations: Box<[ProofObligation]>,
}

impl LocalEquationFacts {
    /// Return the number of claims still requiring proof-term verification.
    #[must_use]
    pub(crate) const fn proof_obligation_count(&self) -> usize {
        self.proof_obligations.len()
    }

    /// Return whether one function still depends on an unverified proof term.
    #[must_use]
    pub(crate) fn has_proof_obligations(&self, function: &str) -> bool {
        self.proof_obligations
            .iter()
            .any(|obligation| obligation.function.as_ref() == function)
    }

    /// Return whether one exact local operation retained a proof obligation.
    #[must_use]
    pub(crate) fn requires_proof(&self, function: &str, opcode: &str, origin: Option<u64>) -> bool {
        self.proof_obligations.iter().any(|obligation| {
            obligation.function.as_ref() == function
                && obligation.opcode == opcode
                && obligation.origin == origin
        })
    }
}

#[derive(Clone, Copy)]
struct ValueFact<'a> {
    source: &'a wire::ValueProto,
    type_id: TypeId,
}

struct FunctionVerifier<'a, 'work> {
    function: Arc<str>,
    arena: &'a TypeArena,
    values: &'a HashMap<u64, TypeId>,
    definitions: HashMap<u64, &'a wire::ValueProto>,
    producers: HashMap<u64, &'a wire::OperationProto>,
    obligations: &'a mut Vec<ProofObligation>,
    require_static_dimensions: bool,
    preparation_widths: HashMap<u64, Option<u64>>,
    work: &'work mut QCoreVerificationWork,
}

/// Verify all `qcore` and quantum equations not owned by another local pass.
///
/// Positive allocation widths are recovered only from a direct integer
/// constant, a function parameter that owns a dependent dimension binder, or
/// an availability lift / explicit structured capture of one of those values.
/// Arbitrary integer computations are deliberately not treated as shape proof.
///
/// # Errors
///
/// Returns [`LocalEquationError`] when an operation's arity, attributes, types,
/// stages, selectors, result owner, or trusted width provenance is invalid.
#[cfg(test)]
pub(crate) fn verify_program(
    package: &wire::QCorePackageProto,
    arena: &TypeArena,
    operations: &LocallyCheckedOperations,
) -> Result<LocalEquationFacts, LocalEquationError> {
    let mut work = QCoreVerificationWork::default();
    verify_program_with_work(package, arena, operations, &mut work)
}

pub(crate) fn verify_program_with_work(
    package: &wire::QCorePackageProto,
    arena: &TypeArena,
    operations: &LocallyCheckedOperations,
    work: &mut QCoreVerificationWork,
) -> Result<LocalEquationFacts, LocalEquationError> {
    verify(package, arena, operations, true, work)
}

/// Verify open module equations while retaining computed shape obligations.
///
/// # Errors
///
/// Returns [`LocalEquationError`] for malformed local equations. Allocation
/// widths that require Const evaluation remain deferred until specialization.
#[cfg(test)]
pub(crate) fn verify_module(
    package: &wire::QCorePackageProto,
    arena: &TypeArena,
    operations: &LocallyCheckedOperations,
) -> Result<LocalEquationFacts, LocalEquationError> {
    let mut work = QCoreVerificationWork::default();
    verify_module_with_work(package, arena, operations, &mut work)
}

pub(crate) fn verify_module_with_work(
    package: &wire::QCorePackageProto,
    arena: &TypeArena,
    operations: &LocallyCheckedOperations,
    work: &mut QCoreVerificationWork,
) -> Result<LocalEquationFacts, LocalEquationError> {
    verify(package, arena, operations, false, work)
}

fn verify(
    package: &wire::QCorePackageProto,
    arena: &TypeArena,
    operations: &LocallyCheckedOperations,
    require_static_dimensions: bool,
    work: &mut QCoreVerificationWork,
) -> Result<LocalEquationFacts, LocalEquationError> {
    let mut obligations = Vec::new();
    let bindings = require_static_dimensions.then(|| {
        work.charge_equations(package.const_bindings.len());
        package
            .const_bindings
            .iter()
            .filter_map(|binding| {
                binding
                    .value
                    .as_ref()
                    .map(|value| (binding.name.as_str(), value))
            })
            .collect::<HashMap<_, _>>()
    });
    for function in &package.functions {
        work.charge_equations(1_usize.saturating_add(function.parameters.len()));
        let values = operations
            .value_types(&function.symbol)
            .ok_or(LocalEquationError::MissingTypeRoot { value_id: u64::MAX })?;
        let mut definitions = HashMap::with_capacity(values.len());
        let mut producers = HashMap::new();
        for parameter in &function.parameters {
            let value = parameter
                .value
                .as_ref()
                .ok_or(LocalEquationError::MissingTypeRoot { value_id: u64::MAX })?;
            definitions.insert(value.id, value);
        }
        if let Some(body) = &function.body {
            collect_function_facts(body, &mut definitions, &mut producers, work);
        }
        let function_bindings = (function.symbol == package.entrypoint)
            .then_some(bindings.as_ref())
            .flatten();
        let function_scope = parameter_widths(function, arena, values, function_bindings, work)?;
        let mut verifier = FunctionVerifier {
            function: Arc::from(function.symbol.as_str()),
            arena,
            values,
            definitions,
            producers,
            obligations: &mut obligations,
            require_static_dimensions,
            preparation_widths: HashMap::new(),
            work,
        };
        if let Some(body) = &function.body {
            let mut bounds = HashMap::new();
            for parameter in &function.parameters {
                let value = parameter.value.as_ref().expect("verified parameter");
                if arena.stage_of(values[&value.id]) == Some(Stage::Const)
                    && is_scalar(arena, values[&value.id], ScalarType::Int)
                {
                    let bound = function_bindings
                        .and_then(|bindings| bindings.get(parameter.name.as_str()).copied())
                        .map(SliceBound::from_literal)
                        .transpose()
                        .map_err(|_| malformed_core("slice Const binding"))?
                        .unwrap_or_else(|| SliceBound::Binder(parameter.name.clone()));
                    bounds.insert(value.id, bound);
                }
            }
            verifier.verify_block(body, function_scope, bounds)?;
        }
    }
    Ok(LocalEquationFacts {
        proof_obligations: obligations.into_boxed_slice(),
    })
}

fn collect_function_facts<'a>(
    block: &'a wire::BlockProto,
    definitions: &mut HashMap<u64, &'a wire::ValueProto>,
    producers: &mut HashMap<u64, &'a wire::OperationProto>,
    work: &mut QCoreVerificationWork,
) {
    let mut blocks = vec![block];
    while let Some(block) = blocks.pop() {
        work.charge_equations(1_usize.saturating_add(block.arguments.len()));
        for argument in &block.arguments {
            definitions.insert(argument.id, argument);
        }
        for operation in &block.operations {
            work.charge_equations(1_usize.saturating_add(operation.results.len()));
            for result in &operation.results {
                definitions.insert(result.id, result);
                producers.insert(result.id, operation);
            }
            for region in operation.regions.iter().rev() {
                blocks.extend(region.blocks.iter().rev());
            }
        }
    }
}

fn parameter_widths(
    function: &wire::FunctionProto,
    arena: &TypeArena,
    values: &HashMap<u64, TypeId>,
    bindings: Option<&HashMap<&str, &wire::LiteralProto>>,
    work: &mut QCoreVerificationWork,
) -> Result<HashMap<u64, Dimension>, LocalEquationError> {
    let mut dimensions = HashMap::new();
    let scope = Arc::<str>::from(function.symbol.as_str());
    for parameter in &function.parameters {
        work.charge_equations(1);
        let value = parameter
            .value
            .as_ref()
            .ok_or(LocalEquationError::MissingTypeRoot { value_id: u64::MAX })?;
        let type_id = values
            .get(&value.id)
            .copied()
            .ok_or(LocalEquationError::MissingTypeRoot { value_id: value.id })?;
        if matches!(
            arena.get(type_id),
            Some(SemanticType::Staged {
                stage: Stage::Const,
                inner,
            }) if arena.get(*inner) == Some(&SemanticType::Scalar(ScalarType::Int))
        ) {
            let bound = bindings
                .and_then(|bindings| bindings.get(parameter.name.as_str()).copied())
                .and_then(crate::static_values::literal)
                .and_then(crate::static_values::integer)
                .and_then(|extent| u64::try_from(extent).ok())
                .filter(|extent| *extent > 0);
            dimensions.insert(
                value.id,
                bound.map_or_else(
                    || Dimension::Symbol {
                        scope: Arc::clone(&scope),
                        name: Arc::from(parameter.name.as_str()),
                    },
                    Dimension::Fixed,
                ),
            );
        }
    }
    Ok(dimensions)
}

impl<'a> FunctionVerifier<'a, '_> {
    fn fact(&self, value_id: u64) -> Result<ValueFact<'a>, LocalEquationError> {
        let source = self
            .definitions
            .get(&value_id)
            .copied()
            .ok_or(LocalEquationError::MissingTypeRoot { value_id })?;
        let type_id = self
            .values
            .get(&value_id)
            .copied()
            .ok_or(LocalEquationError::MissingTypeRoot { value_id })?;
        Ok(ValueFact { source, type_id })
    }

    fn facts(&self, values: &[u64]) -> Result<Vec<ValueFact<'a>>, LocalEquationError> {
        values.iter().map(|value| self.fact(*value)).collect()
    }

    fn result_facts(
        &self,
        operation: &wire::OperationProto,
    ) -> Result<Vec<ValueFact<'a>>, LocalEquationError> {
        operation
            .results
            .iter()
            .map(|result| self.fact(result.id))
            .collect()
    }

    fn verify_block(
        &mut self,
        block: &'a wire::BlockProto,
        mut dimensions: HashMap<u64, Dimension>,
        mut bounds: HashMap<u64, SliceBound>,
    ) -> Result<(), LocalEquationError> {
        self.work.charge_equations(1);
        for operation in &block.operations {
            self.work.charge_equations(
                1_usize
                    .saturating_add(operation.operands.len())
                    .saturating_add(operation.results.len()),
            );
            let operands = self.facts(&operation.operands)?;
            let results = self.result_facts(operation)?;
            self.verify_operation(operation, &operands, &results, &dimensions)?;
            if operation.opcode == "classical.slice" {
                let descriptors = operation_bounds(operation)
                    .map_err(|_| malformed_core("classical.slice bounds"))?;
                let mut actual = operation.operands.iter().skip(1);
                for descriptor in descriptors {
                    if descriptor == SliceBound::Omitted {
                        continue;
                    }
                    if actual.next().and_then(|id| bounds.get(id)) != Some(&descriptor) {
                        return Err(malformed_core("classical.slice bound provenance"));
                    }
                }
            }
            if let [result] = operation.results.as_slice() {
                if operation.opcode == "qcore.constant" {
                    if let Some(bound) = exact_attributes(operation, &["value"])
                        .and_then(|row| SliceBound::from_literal(row[0]).ok())
                    {
                        bounds.insert(result.id, bound);
                    }
                } else if operation.opcode == "qcore.stage_lift"
                    && let Some(bound) = operation
                        .operands
                        .first()
                        .and_then(|id| bounds.get(id))
                        .cloned()
                {
                    bounds.insert(result.id, bound);
                }
            }
            if matches!(operation.opcode.as_str(), "classical.neg" | "classical.pos")
                && let ([source], [result]) =
                    (operation.operands.as_slice(), operation.results.as_slice())
                && let Some(SliceBound::Integer(value)) = bounds.get(source).cloned()
            {
                bounds.insert(
                    result.id,
                    SliceBound::Integer(if operation.opcode == "classical.neg" {
                        -value
                    } else {
                        value
                    }),
                );
            }
            self.derive_width_evidence(operation, &mut dimensions);
            self.verify_regions(operation, &dimensions, &bounds)?;
        }
        Ok(())
    }

    fn verify_operation(
        &mut self,
        operation: &wire::OperationProto,
        operands: &[ValueFact<'a>],
        results: &[ValueFact<'a>],
        dimensions: &HashMap<u64, Dimension>,
    ) -> Result<(), LocalEquationError> {
        match operation.opcode.as_str() {
            "qcore.constant" => self.verify_constant(operation, operands, results),
            "qcore.stage_lift" => self.verify_stage_lift(operation, operands, results),
            "qcore.tuple" => self.verify_tuple(operation, operands, results),
            "qcore.unpack" => self.verify_unpack(operation, operands, results),
            "qcore.reshape" => self.verify_reshape(operation, operands, results),
            "qcore.phase_by" => self.verify_phase_by(operation, operands, results),
            "qcore.xor_by" => self.verify_xor_by(operation, operands, results),
            "qcore.permute_by" => self.verify_permute_by(operation, operands, results),
            "qcore.adjoint" => self.verify_quantum(operation, operands, results, dimensions),
            "qcore.forget" => self.verify_forget(operation, operands, results),
            opcode if opcode.starts_with("quantum.") => {
                self.verify_quantum(operation, operands, results, dimensions)
            }
            _ => Ok(()),
        }
    }

    fn verify_constant(
        &self,
        operation: &wire::OperationProto,
        operands: &[ValueFact<'a>],
        results: &[ValueFact<'a>],
    ) -> Result<(), LocalEquationError> {
        let malformed = || malformed_core("qcore.constant operation");
        let [result] = results else {
            return Err(malformed());
        };
        let Some(attributes) = exact_attributes(operation, &["value"]) else {
            return Err(malformed());
        };
        let [value] = attributes.as_slice() else {
            return Err(malformed());
        };
        if !operands.is_empty()
            || self.arena.is_quantum(result.type_id)
            || result.source.region.is_some()
            || !literal_conforms(self.arena, result.type_id, value)
        {
            return Err(malformed());
        }
        Ok(())
    }

    fn verify_stage_lift(
        &self,
        operation: &wire::OperationProto,
        operands: &[ValueFact<'a>],
        results: &[ValueFact<'a>],
    ) -> Result<(), LocalEquationError> {
        let malformed = || malformed_core("qcore.stage_lift operation");
        let ([source], [target]) = (operands, results) else {
            return Err(malformed());
        };
        if !operation.attributes.is_empty()
            || self.arena.is_quantum(source.type_id)
            || self.arena.is_quantum(target.type_id)
            || source.source.region.is_some()
            || target.source.region.is_some()
            || self.arena.strip_stage(source.type_id) != self.arena.strip_stage(target.type_id)
            || self.arena.stage_of(source.type_id) >= self.arena.stage_of(target.type_id)
        {
            return Err(malformed());
        }
        Ok(())
    }

    fn verify_tuple(
        &self,
        operation: &wire::OperationProto,
        operands: &[ValueFact<'a>],
        results: &[ValueFact<'a>],
    ) -> Result<(), LocalEquationError> {
        let malformed = || malformed_core("tuple construction");
        let [result] = results else {
            return Err(malformed());
        };
        structural_proof_marker(operation).map_err(|()| malformed())?;
        let Some(elements) = aggregate_elements(self.arena, result.type_id) else {
            return Err(malformed());
        };
        if elements.len() != operands.len()
            || elements
                .iter()
                .zip(operands)
                .any(|(expected, operand)| !expected.matches(operand.type_id, self.arena))
            || !tuple_owner_is_valid(operands, *result, self.arena)
        {
            return Err(malformed());
        }
        Ok(())
    }

    fn verify_unpack(
        &mut self,
        operation: &wire::OperationProto,
        operands: &[ValueFact<'a>],
        results: &[ValueFact<'a>],
    ) -> Result<(), LocalEquationError> {
        let malformed = || malformed_core("qcore.unpack operation");
        let [product] = operands else {
            return Err(malformed());
        };
        let proof = structural_proof_marker(operation).map_err(|()| malformed())?;
        let Some(elements) = aggregate_elements(self.arena, product.type_id) else {
            return Err(malformed());
        };
        if elements.len() != results.len()
            || elements
                .iter()
                .zip(results)
                .any(|(expected, result)| !expected.matches(result.type_id, self.arena))
        {
            return Err(malformed());
        }
        if !unpack_owners_are_valid(*product, results, self.arena) {
            if !proof {
                return Err(malformed());
            }
            self.obligations.push(ProofObligation {
                function: Arc::clone(&self.function),
                opcode: "qcore.unpack",
                origin: Some(product.source.id),
            });
        }
        Ok(())
    }

    fn verify_reshape(
        &self,
        operation: &wire::OperationProto,
        operands: &[ValueFact<'a>],
        results: &[ValueFact<'a>],
    ) -> Result<(), LocalEquationError> {
        let malformed = || malformed_core("qcore.reshape operation");
        let ([source], [target]) = (operands, results) else {
            return Err(malformed());
        };
        structural_proof_marker(operation).map_err(|()| malformed())?;
        let source_width = coherent_width_expression(self.arena, source.type_id);
        let target_width = coherent_width_expression(self.arena, target.type_id);
        if source_width.is_none()
            || source_width != target_width
            || source.source.region.as_ref() != target.source.region.as_ref()
        {
            return Err(malformed());
        }
        Ok(())
    }

    fn verify_phase_by(
        &self,
        operation: &wire::OperationProto,
        operands: &[ValueFact<'a>],
        results: &[ValueFact<'a>],
    ) -> Result<(), LocalEquationError> {
        let malformed = || malformed_core("qcore.phase_by operation");
        let Some(attributes) = exact_attributes(operation, &["predicate"]) else {
            return Err(malformed());
        };
        let [predicate] = attributes.as_slice() else {
            return Err(malformed());
        };
        if string_literal(predicate).is_none_or(str::is_empty) {
            return Err(malformed());
        }
        require_owned_transform(self.arena, operation, operands, results, 1, 1)
            .map_err(|_| malformed())
    }

    fn verify_xor_by(
        &self,
        operation: &wire::OperationProto,
        operands: &[ValueFact<'a>],
        results: &[ValueFact<'a>],
    ) -> Result<(), LocalEquationError> {
        let malformed = || malformed_core("qcore.xor_by operation");
        let Some(attributes) = exact_attributes(operation, &["predicate"]) else {
            return Err(malformed());
        };
        let [predicate] = attributes.as_slice() else {
            return Err(malformed());
        };
        if string_literal(predicate).is_none_or(str::is_empty) {
            return Err(malformed());
        }
        let [state] = operands else {
            return Err(malformed());
        };
        if !matches!(
            self.arena.get(self.arena.strip_stage(state.type_id)),
            Some(SemanticType::Product(elements)) if elements.len() == 2
                && elements.iter().all(|element| self.arena.is_quantum(*element))
        ) {
            return Err(malformed());
        }
        require_linear_transform(self.arena, operation, operands, results).map_err(|_| malformed())
    }

    fn verify_permute_by(
        &self,
        operation: &wire::OperationProto,
        operands: &[ValueFact<'a>],
        results: &[ValueFact<'a>],
    ) -> Result<(), LocalEquationError> {
        let malformed = || malformed_core("qcore.permute_by operation");
        let Some(attributes) = exact_attributes(operation, &["predicate"]) else {
            return Err(malformed());
        };
        let [predicate] = attributes.as_slice() else {
            return Err(malformed());
        };
        if string_literal(predicate).is_none_or(str::is_empty) {
            return Err(malformed());
        }
        require_owned_transform(self.arena, operation, operands, results, 1, 1)
            .map_err(|_| malformed())
    }

    fn verify_forget(
        &mut self,
        operation: &wire::OperationProto,
        operands: &[ValueFact<'a>],
        results: &[ValueFact<'a>],
    ) -> Result<(), LocalEquationError> {
        let [value] = operands else {
            return Err(malformed_core("qcore.forget operation"));
        };
        let attributes_valid = match operation.attributes.as_slice() {
            [] => true,
            [attribute]
                if attribute.name == "reason"
                    && matches!(
                        attribute
                            .value
                            .as_ref()
                            .and_then(|value| value.value.as_ref()),
                        Some(wire::literal_proto::Value::StringValue(_))
                    ) =>
            {
                true
            }
            _ => false,
        };
        if !attributes_valid || !results.is_empty() || !self.arena.is_quantum(value.type_id) {
            return Err(malformed_core("qcore.forget operation"));
        }
        self.obligations.push(ProofObligation {
            function: Arc::clone(&self.function),
            opcode: "qcore.forget",
            origin: None,
        });
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn verify_quantum(
        &mut self,
        operation: &wire::OperationProto,
        operands: &[ValueFact<'a>],
        results: &[ValueFact<'a>],
        dimensions: &HashMap<u64, Dimension>,
    ) -> Result<(), LocalEquationError> {
        let malformed = || LocalEquationError::MalformedQuantum {
            opcode: operation.opcode.clone(),
        };
        match operation.opcode.as_str() {
            "quantum.alloc" => {
                let ([width], [state]) = (operands, results) else {
                    return Err(malformed());
                };
                let payload = quantum_payload(self.arena, state.type_id);
                let declared_width = payload.and_then(|payload| match self.arena.get(payload) {
                    Some(SemanticType::Word {
                        kind: WordKind::Bits,
                        width,
                    }) => Some(width),
                    _ => None,
                });
                let matches_width = match dimensions.get(&width.source.id) {
                    Some(dimension) => declared_width == Some(dimension),
                    None if self.require_static_dimensions => {
                        return Err(LocalEquationError::UnresolvedAllocationWidth);
                    }
                    None => declared_width.is_some(),
                };
                if !operation.attributes.is_empty()
                    || !is_scalar(self.arena, width.type_id, ScalarType::Int)
                    || self.arena.stage_of(width.type_id) > Some(Stage::Const)
                    || !matches_width
                    || state.source.region.is_none()
                {
                    return Err(malformed());
                }
            }
            "quantum.prepare" => {
                let ([data], [state]) = (operands, results) else {
                    return Err(malformed());
                };
                let data_width = self.preparation_width(data.source.id, &mut HashSet::new());
                let payload_width = quantum_payload(self.arena, state.type_id)
                    .and_then(|payload| coherent_payload_width(self.arena, payload));
                if !operation.attributes.is_empty()
                    || self.arena.is_quantum(data.type_id)
                    || data_width.is_none()
                    || data_width != payload_width
                    || state.source.region.is_none()
                {
                    return Err(malformed());
                }
            }
            "quantum.measure" => {
                let ([state], [result]) = (operands, results) else {
                    return Err(malformed());
                };
                let Some(payload) = quantum_payload(self.arena, state.type_id) else {
                    return Err(malformed());
                };
                if !operation.attributes.is_empty()
                    || !is_staged(self.arena, result.type_id, Stage::Shot, payload)
                    || result.source.region.is_some()
                {
                    return Err(malformed());
                }
            }
            "quantum.measure_x" | "quantum.measure_y" | "quantum.measure_z" => {
                let [result] = results else {
                    return Err(malformed());
                };
                let keywords = keyword_tuple(operation).ok_or_else(malformed)?;
                let (state, expected_keywords): (ValueFact<'_>, &[&str]) = match operands {
                    [state] => (*state, &[]),
                    [state, target] if is_scalar(self.arena, target.type_id, ScalarType::Int) => {
                        (*state, &["target"])
                    }
                    _ => return Err(malformed()),
                };
                let Some(payload) = quantum_payload(self.arena, state.type_id) else {
                    return Err(malformed());
                };
                if keywords != expected_keywords
                    || (operands.len() == 1
                        && coherent_payload_width(self.arena, payload) != Some(1))
                    || !measurement_pair(self.arena, result.type_id, state.type_id, None)
                    || result.source.region.as_ref() != state.source.region.as_ref()
                {
                    return Err(malformed());
                }
            }
            "quantum.measure_reset" => {
                let ([state], [result]) = (operands, results) else {
                    return Err(malformed());
                };
                let Some(payload) = quantum_payload(self.arena, state.type_id) else {
                    return Err(malformed());
                };
                if !operation.attributes.is_empty()
                    || !measurement_pair(self.arena, result.type_id, state.type_id, Some(payload))
                    || result.source.region.as_ref() != state.source.region.as_ref()
                {
                    return Err(malformed());
                }
            }
            "quantum.discard" => {
                let ([state], [result]) = (operands, results) else {
                    return Err(malformed());
                };
                if !operation.attributes.is_empty()
                    || quantum_payload(self.arena, state.type_id).is_none()
                    || self.arena.get(result.type_id)
                        != Some(&SemanticType::Scalar(ScalarType::Unit))
                    || result.source.region.is_some()
                {
                    return Err(malformed());
                }
            }
            "quantum.reset" => {
                require_owned_transform(self.arena, operation, operands, results, 1, 1)?;
                if !operation.attributes.is_empty() {
                    return Err(malformed());
                }
            }
            "quantum.global_phase" => {
                require_owned_transform(self.arena, operation, operands, results, 2, 2)?;
                let Some(angle) = operands.get(1) else {
                    return Err(malformed());
                };
                if !operation.attributes.is_empty() || !is_numeric_angle(self.arena, angle.type_id)
                {
                    return Err(malformed());
                }
            }
            "quantum.pauli_evolution" => {
                require_owned_transform(self.arena, operation, operands, results, 3, 3)?;
                let (Some(hamiltonian), Some(time)) = (operands.get(1), operands.get(2)) else {
                    return Err(malformed());
                };
                if !operation.attributes.is_empty()
                    || !is_const_observable(self.arena, hamiltonian.type_id)
                    || !is_numeric_angle(self.arena, time.type_id)
                {
                    return Err(malformed());
                }
            }
            "quantum.add_into" | "quantum.subtract_into" => {
                require_owned_transform(self.arena, operation, operands, results, 1, 1)?;
                self.verify_addition_width(operation, operands)?;
                if !operation.attributes.is_empty() {
                    return Err(malformed());
                }
            }
            "quantum.add_constant"
            | "quantum.subtract_constant"
            | "quantum.mod_mul_add"
            | "quantum.mod_mul_sub" => {
                require_owned_transform(self.arena, operation, operands, results, 2, 2)?;
                self.verify_arithmetic_width(operation, operands)?;
                if !operation.attributes.is_empty()
                    || !is_const_int(self.arena, operands[1].type_id)
                {
                    return Err(malformed());
                }
            }
            "quantum.mod_add_constant"
            | "quantum.mod_sub_constant"
            | "quantum.mod_mul"
            | "quantum.mod_div" => {
                require_owned_transform(self.arena, operation, operands, results, 3, 3)?;
                self.verify_arithmetic_width(operation, operands)?;
                if !operation.attributes.is_empty()
                    || operands[1..]
                        .iter()
                        .any(|value| !is_const_int(self.arena, value.type_id))
                {
                    return Err(malformed());
                }
            }
            "quantum.mod_pow" | "quantum.mod_pow_inverse" => {
                require_owned_transform(self.arena, operation, operands, results, 3, 3)?;
                self.verify_mod_pow_width(operation, operands)?;
                if !operation.attributes.is_empty()
                    || operands[1..]
                        .iter()
                        .any(|value| !is_const_int(self.arena, value.type_id))
                {
                    return Err(malformed());
                }
            }
            "quantum.mod_multi_pow" | "quantum.mod_multi_pow_inverse" => {
                require_owned_transform(self.arena, operation, operands, results, 5, 5)?;
                self.verify_mod_multi_pow_shape(operation, operands)?;
                if !operation.attributes.is_empty()
                    || !is_const_int_vector(self.arena, operands[1].type_id)
                    || operands[2..]
                        .iter()
                        .any(|value| !is_const_int(self.arena, value.type_id))
                {
                    return Err(malformed());
                }
            }
            "quantum.mod_add"
            | "quantum.mod_sub"
            | "quantum.mul_add"
            | "quantum.mul_sub"
            | "quantum.less_than_xor" => {
                require_owned_transform(self.arena, operation, operands, results, 1, 1)?;
                self.verify_arithmetic_width(operation, operands)?;
                if !operation.attributes.is_empty() {
                    return Err(malformed());
                }
            }
            "quantum.release" => {
                let [state] = operands else {
                    return Err(malformed());
                };
                let origin = cleanup_attributes(operation).ok_or_else(malformed)?;
                if !results.is_empty() || quantum_payload(self.arena, state.type_id).is_none() {
                    return Err(malformed());
                }
                self.obligations.push(ProofObligation {
                    function: Arc::clone(&self.function),
                    opcode: "quantum.release",
                    origin: Some(origin),
                });
            }
            "quantum.unprepare" => {
                let ([state, data], []) = (operands, results) else {
                    return Err(malformed());
                };
                let origin = cleanup_attributes(operation).ok_or_else(malformed)?;
                let data_width = self.preparation_width(data.source.id, &mut HashSet::new());
                let payload_width = quantum_payload(self.arena, state.type_id)
                    .and_then(|payload| coherent_payload_width(self.arena, payload));
                if self.arena.is_quantum(data.type_id)
                    || data_width.is_none()
                    || data_width != payload_width
                {
                    return Err(malformed());
                }
                self.obligations.push(ProofObligation {
                    function: Arc::clone(&self.function),
                    opcode: "quantum.unprepare",
                    origin: Some(origin),
                });
            }
            "qcore.adjoint" => {
                self.verify_adjoint(operation, operands, results)?;
            }
            opcode => {
                let gate = primitive_gate_spec(opcode).ok_or_else(malformed)?;
                self.verify_gate(operation, operands, results, gate, None)?;
            }
        }
        Ok(())
    }

    fn preparation_width(&mut self, value_id: u64, visiting: &mut HashSet<u64>) -> Option<u64> {
        self.work.charge_equations(1);
        if let Some(cached) = self.preparation_widths.get(&value_id) {
            return *cached;
        }
        if !visiting.insert(value_id) {
            return None;
        }
        let Some(fact) = self.fact(value_id).ok() else {
            visiting.remove(&value_id);
            self.preparation_widths.insert(value_id, None);
            return None;
        };
        let base = self.arena.strip_stage(fact.type_id);
        if let Some(width) = coherent_payload_width(self.arena, base) {
            visiting.remove(&value_id);
            self.preparation_widths.insert(value_id, Some(width));
            return Some(width);
        }
        let Some(producer) = self.producers.get(&value_id).copied() else {
            visiting.remove(&value_id);
            self.preparation_widths.insert(value_id, None);
            return None;
        };
        let width = match producer.opcode.as_str() {
            "qcore.constant"
                if self.arena.get(base) == Some(&SemanticType::Scalar(ScalarType::Int))
                    && producer.attributes.len() == 1
                    && producer.attributes[0].name == "value"
                    && producer.attributes[0]
                        .value
                        .as_ref()
                        .and_then(nonnegative_u64)
                        .is_some_and(|value| value <= 1) =>
            {
                Some(1)
            }
            "qcore.tuple" => producer.operands.iter().try_fold(0_u64, |total, operand| {
                total.checked_add(self.preparation_width(*operand, visiting)?)
            }),
            "qcore.stage_lift" if producer.operands.len() == 1 => {
                self.preparation_width(producer.operands[0], visiting)
            }
            _ => None,
        };
        visiting.remove(&value_id);
        self.preparation_widths.insert(value_id, width);
        width
    }

    #[allow(clippy::too_many_lines)]
    fn verify_adjoint(
        &self,
        operation: &wire::OperationProto,
        operands: &[ValueFact<'a>],
        results: &[ValueFact<'a>],
    ) -> Result<(), LocalEquationError> {
        let malformed = || LocalEquationError::MalformedQuantum {
            opcode: operation.opcode.clone(),
        };
        let Some(attributes) =
            exact_attributes(operation, &["original", "original_attributes", "proof"])
        else {
            return Err(malformed());
        };
        let [original, original_attributes, proof] = attributes.as_slice() else {
            return Err(malformed());
        };
        let Some(original) = string_literal(original) else {
            return Err(malformed());
        };
        let Some(original_attributes) = mapping_literal(original_attributes) else {
            return Err(malformed());
        };
        if string_literal(proof) != Some("modular-adjoint") {
            return Err(malformed());
        }
        match original {
            "qcore.phase_by" => {
                require_owned_transform(self.arena, operation, operands, results, 1, 1)?;
                if !valid_boolean_lift_adjoint_attributes(original_attributes) {
                    return Err(malformed());
                }
            }
            "qcore.xor_by" | "qcore.permute_by" => {
                require_linear_transform(self.arena, operation, operands, results)?;
                if !valid_boolean_lift_adjoint_attributes(original_attributes) {
                    return Err(malformed());
                }
            }
            "qcore.control_call" | "qcore.select_call" => {
                require_linear_transform(self.arena, operation, operands, results)?;
                if !valid_coherent_adjoint_attributes(original, original_attributes) {
                    return Err(malformed());
                }
            }
            "quantum.global_phase" => {
                require_owned_transform(self.arena, operation, operands, results, 2, 2)?;
                let Some(angle) = operands.get(1) else {
                    return Err(malformed());
                };
                if !original_attributes.is_empty() || !is_numeric_angle(self.arena, angle.type_id) {
                    return Err(malformed());
                }
            }
            "quantum.pauli_evolution" => {
                require_owned_transform(self.arena, operation, operands, results, 3, 3)?;
                let (Some(hamiltonian), Some(time)) = (operands.get(1), operands.get(2)) else {
                    return Err(malformed());
                };
                if !original_attributes.is_empty()
                    || !is_const_observable(self.arena, hamiltonian.type_id)
                    || !is_numeric_angle(self.arena, time.type_id)
                {
                    return Err(malformed());
                }
            }
            "quantum.add_into" | "quantum.subtract_into" => {
                require_owned_transform(self.arena, operation, operands, results, 1, 1)?;
                self.verify_addition_width(operation, operands)?;
                if !original_attributes.is_empty() {
                    return Err(malformed());
                }
            }
            "quantum.add_constant"
            | "quantum.subtract_constant"
            | "quantum.mod_mul_add"
            | "quantum.mod_mul_sub" => {
                require_owned_transform(self.arena, operation, operands, results, 2, 2)?;
                self.verify_arithmetic_width(operation, operands)?;
                if !original_attributes.is_empty() || !is_const_int(self.arena, operands[1].type_id)
                {
                    return Err(malformed());
                }
            }
            "quantum.mod_add_constant"
            | "quantum.mod_sub_constant"
            | "quantum.mod_mul"
            | "quantum.mod_div" => {
                require_owned_transform(self.arena, operation, operands, results, 3, 3)?;
                self.verify_arithmetic_width(operation, operands)?;
                if !original_attributes.is_empty()
                    || operands[1..]
                        .iter()
                        .any(|value| !is_const_int(self.arena, value.type_id))
                {
                    return Err(malformed());
                }
            }
            "quantum.mod_pow" | "quantum.mod_pow_inverse" => {
                require_owned_transform(self.arena, operation, operands, results, 3, 3)?;
                self.verify_mod_pow_width(operation, operands)?;
                if !original_attributes.is_empty()
                    || operands[1..]
                        .iter()
                        .any(|value| !is_const_int(self.arena, value.type_id))
                {
                    return Err(malformed());
                }
            }
            "quantum.mod_multi_pow" | "quantum.mod_multi_pow_inverse" => {
                require_owned_transform(self.arena, operation, operands, results, 5, 5)?;
                self.verify_mod_multi_pow_shape(operation, operands)?;
                if !original_attributes.is_empty()
                    || !is_const_int_vector(self.arena, operands[1].type_id)
                    || operands[2..]
                        .iter()
                        .any(|value| !is_const_int(self.arena, value.type_id))
                {
                    return Err(malformed());
                }
            }
            "quantum.mod_add"
            | "quantum.mod_sub"
            | "quantum.mul_add"
            | "quantum.mul_sub"
            | "quantum.less_than_xor" => {
                require_owned_transform(self.arena, operation, operands, results, 1, 1)?;
                self.verify_arithmetic_width(operation, operands)?;
                if !original_attributes.is_empty() {
                    return Err(malformed());
                }
            }
            opcode => {
                let gate = primitive_gate_spec(opcode).ok_or_else(malformed)?;
                self.verify_gate(
                    operation,
                    operands,
                    results,
                    gate,
                    Some(original_attributes),
                )?;
            }
        }
        Ok(())
    }

    fn verify_addition_width(
        &self,
        operation: &wire::OperationProto,
        operands: &[ValueFact<'a>],
    ) -> Result<(), LocalEquationError> {
        let width = operands
            .first()
            .and_then(|state| quantum_payload(self.arena, state.type_id))
            .and_then(|payload| coherent_payload_width(self.arena, payload));
        if width.is_some_and(|width| width == 0 || width % 2 != 0) {
            return Err(LocalEquationError::MalformedQuantum {
                opcode: operation.opcode.clone(),
            });
        }
        Ok(())
    }

    fn verify_arithmetic_width(
        &self,
        operation: &wire::OperationProto,
        operands: &[ValueFact<'a>],
    ) -> Result<(), LocalEquationError> {
        let payload = operands
            .first()
            .and_then(|state| quantum_payload(self.arena, state.type_id));
        let width = payload.and_then(|payload| coherent_payload_width(self.arena, payload));
        let total_valid = match operation.opcode.as_str() {
            "quantum.mod_add"
            | "quantum.mod_sub"
            | "quantum.mod_mul_add"
            | "quantum.mod_mul_sub" => width.is_none_or(|n| n > 0 && n % 3 == 0),
            "quantum.mul_add" | "quantum.mul_sub" => width.is_none_or(|n| n > 0 && n % 4 == 0),
            "quantum.less_than_xor" => width.is_none_or(|n| n >= 3 && n % 2 == 1),
            _ => width.is_none_or(|n| n > 0),
        };
        let product_valid = match payload.and_then(|value| self.arena.get(value)) {
            Some(SemanticType::Product(elements))
                if matches!(
                    operation.opcode.as_str(),
                    "quantum.mod_add"
                        | "quantum.mod_sub"
                        | "quantum.mod_mul_add"
                        | "quantum.mod_mul_sub"
                        | "quantum.mul_add"
                        | "quantum.mul_sub"
                        | "quantum.less_than_xor"
                ) =>
            {
                let widths = elements
                    .iter()
                    .map(|element| coherent_payload_width(self.arena, *element))
                    .collect::<Vec<_>>();
                match widths.as_slice() {
                    [Some(first), Some(left), Some(right)] => match operation.opcode.as_str() {
                        "quantum.mod_add"
                        | "quantum.mod_sub"
                        | "quantum.mod_mul_add"
                        | "quantum.mod_mul_sub" => *first > 0 && first == left && left == right,
                        "quantum.mul_add" | "quantum.mul_sub" => {
                            *left > 0 && left == right && left.checked_mul(2) == Some(*first)
                        }
                        _ => *first == 1 && *left > 0 && left == right,
                    },
                    [_, _, _] => true,
                    _ => false,
                }
            }
            _ => true,
        };
        if total_valid && product_valid {
            Ok(())
        } else {
            Err(LocalEquationError::MalformedQuantum {
                opcode: operation.opcode.clone(),
            })
        }
    }

    fn verify_mod_pow_width(
        &self,
        operation: &wire::OperationProto,
        operands: &[ValueFact<'a>],
    ) -> Result<(), LocalEquationError> {
        let valid = operands
            .first()
            .and_then(|state| quantum_payload(self.arena, state.type_id))
            .and_then(|payload| self.arena.get(payload))
            .is_some_and(|payload| match payload {
                SemanticType::Product(elements) if elements.len() == 2 => {
                    elements.iter().all(|element| {
                        self.arena.is_coherent_payload(*element)
                            && coherent_payload_width(self.arena, *element).is_none_or(|n| n > 0)
                    })
                }
                _ => false,
            });
        if valid {
            Ok(())
        } else {
            Err(LocalEquationError::MalformedQuantum {
                opcode: operation.opcode.clone(),
            })
        }
    }

    fn verify_mod_multi_pow_shape(
        &self,
        operation: &wire::OperationProto,
        operands: &[ValueFact<'a>],
    ) -> Result<(), LocalEquationError> {
        let valid = operands
            .first()
            .and_then(|state| quantum_payload(self.arena, state.type_id))
            .and_then(|payload| self.arena.get(payload))
            .is_some_and(|payload| match payload {
                SemanticType::Product(elements) if elements.len() == 2 => {
                    elements.iter().all(|element| {
                        self.arena.is_coherent_payload(*element)
                            && coherent_payload_width(self.arena, *element).is_none_or(|n| n > 0)
                    })
                }
                _ => false,
            });
        if valid {
            Ok(())
        } else {
            Err(LocalEquationError::MalformedQuantum {
                opcode: operation.opcode.clone(),
            })
        }
    }

    fn verify_gate(
        &self,
        operation: &wire::OperationProto,
        operands: &[ValueFact<'a>],
        results: &[ValueFact<'a>],
        gate: PrimitiveGateSpec,
        original_attributes: Option<&[wire::literal_proto::Entry]>,
    ) -> Result<(), LocalEquationError> {
        let malformed = || LocalEquationError::MalformedQuantum {
            opcode: operation.opcode.clone(),
        };
        let minimum = 1 + usize::from(gate.has_angle());
        let maximum = minimum + gate.selectors().len();
        require_owned_transform(self.arena, operation, operands, results, minimum, maximum)?;
        let attributes = if let Some(entries) = original_attributes {
            let [entry] = entries else {
                return Err(malformed());
            };
            if entry.key != "keywords" {
                return Err(malformed());
            }
            entry.value.as_ref().ok_or_else(malformed)?
        } else {
            let values = exact_attributes(operation, &["keywords"]).ok_or_else(malformed)?;
            let [keywords] = values.as_slice() else {
                return Err(malformed());
            };
            *keywords
        };
        let Some(keywords) = string_tuple(attributes) else {
            return Err(malformed());
        };
        let Some((_, classical)) = operands.split_first() else {
            return Err(malformed());
        };
        let selectors = if gate.has_angle() {
            let Some((angle, selectors)) = classical.split_first() else {
                return Err(malformed());
            };
            if !is_numeric_angle(self.arena, angle.type_id) {
                return Err(malformed());
            }
            selectors
        } else {
            classical
        };
        if selectors
            .iter()
            .any(|selector| !is_scalar(self.arena, selector.type_id, ScalarType::Int))
            || keywords.len() != selectors.len()
            || !ordered_subset(&keywords, gate.selectors())
        {
            return Err(malformed());
        }
        Ok(())
    }

    fn derive_width_evidence(
        &self,
        operation: &wire::OperationProto,
        dimensions: &mut HashMap<u64, Dimension>,
    ) {
        let Some(result) = operation.results.first() else {
            return;
        };
        match operation.opcode.as_str() {
            "qcore.constant"
                if operation.results.len() == 1
                    && self.values.get(&result.id).is_some_and(|type_id| {
                        is_scalar(self.arena, *type_id, ScalarType::Int)
                            && self.arena.stage_of(*type_id) <= Some(Stage::Const)
                    }) =>
            {
                if let Some(value) = exact_attributes(operation, &["value"])
                    .and_then(|values| values.first().and_then(|value| positive_u64(value)))
                {
                    dimensions.insert(result.id, Dimension::Fixed(value));
                }
            }
            "qcore.stage_lift" if operation.operands.len() == 1 && operation.results.len() == 1 => {
                let result_is_static = self
                    .values
                    .get(&result.id)
                    .is_some_and(|type_id| self.arena.stage_of(*type_id) <= Some(Stage::Const));
                if result_is_static
                    && let Some(source) = operation.operands.first()
                    && let Some(dimension) = dimensions.get(source).cloned()
                {
                    dimensions.insert(result.id, dimension);
                }
            }
            _ => {}
        }
    }

    fn verify_regions(
        &mut self,
        operation: &'a wire::OperationProto,
        dimensions: &HashMap<u64, Dimension>,
        bounds: &HashMap<u64, SliceBound>,
    ) -> Result<(), LocalEquationError> {
        for (region_index, region) in operation.regions.iter().enumerate() {
            for block in &region.blocks {
                let nested = match operation.opcode.as_str() {
                    "qcore.if" => captured_dimensions(
                        &block.arguments,
                        operation.operands.get(1..).unwrap_or_default(),
                        dimensions,
                    ),
                    "qcore.for" if region_index == 0 => {
                        for_invariant_dimensions(operation, block, dimensions)
                    }
                    _ => HashMap::new(),
                };
                let nested_bounds = match operation.opcode.as_str() {
                    "qcore.if" => captured_dimensions(
                        &block.arguments,
                        operation.operands.get(1..).unwrap_or_default(),
                        bounds,
                    ),
                    "qcore.for" if region_index == 0 => {
                        for_invariant_dimensions(operation, block, bounds)
                    }
                    _ => HashMap::new(),
                };
                self.verify_block(block, nested, nested_bounds)?;
            }
        }
        Ok(())
    }
}

fn valid_boolean_lift_adjoint_attributes(attributes: &[wire::literal_proto::Entry]) -> bool {
    matches!(attributes, [predicate]
        if predicate.key == "predicate"
            && predicate.value.as_ref().and_then(string_literal).is_some_and(|value| !value.is_empty()))
}

fn valid_coherent_adjoint_attributes(
    original: &str,
    attributes: &[wire::literal_proto::Entry],
) -> bool {
    match (original, attributes) {
        ("qcore.control_call", [callee, control_state]) => {
            callee.key == "callee"
                && callee.value.as_ref().and_then(string_literal).is_some()
                && control_state.key == "control_state"
                && control_state
                    .value
                    .as_ref()
                    .and_then(ControlState::decode)
                    .is_some()
        }
        ("qcore.select_call", [callees]) => {
            callees.key == "callees"
                && callees
                    .value
                    .as_ref()
                    .and_then(string_tuple)
                    .is_some_and(|values| !values.is_empty())
        }
        _ => false,
    }
}

fn captured_dimensions<T: Clone>(
    arguments: &[wire::ValueProto],
    sources: &[u64],
    dimensions: &HashMap<u64, T>,
) -> HashMap<u64, T> {
    arguments
        .iter()
        .zip(sources)
        .filter_map(|(argument, source)| {
            dimensions
                .get(source)
                .cloned()
                .map(|dimension| (argument.id, dimension))
        })
        .collect()
}

fn for_invariant_dimensions<T: Clone>(
    operation: &wire::OperationProto,
    body: &wire::BlockProto,
    dimensions: &HashMap<u64, T>,
) -> HashMap<u64, T> {
    let Some(terminator) = &body.terminator else {
        return HashMap::new();
    };
    if !matches!(terminator.kind.as_str(), "yield" | "continue") {
        return HashMap::new();
    }
    body.arguments
        .get(1..)
        .unwrap_or_default()
        .iter()
        .zip(operation.operands.get(3..).unwrap_or_default())
        .enumerate()
        .filter_map(|(index, (argument, source))| {
            let forwarded = terminator
                .operands
                .get(index)
                .is_some_and(|value_id| *value_id == argument.id);
            forwarded
                .then(|| {
                    dimensions
                        .get(source)
                        .cloned()
                        .map(|dimension| (argument.id, dimension))
                })
                .flatten()
        })
        .collect()
}

fn malformed_core(description: &str) -> LocalEquationError {
    LocalEquationError::MalformedCore {
        description: description.to_owned(),
    }
}

#[derive(Clone, Copy)]
struct ExpectedElement {
    base: TypeId,
    stage: Option<Stage>,
}

impl ExpectedElement {
    fn matches(self, actual: TypeId, arena: &TypeArena) -> bool {
        arena.strip_stage(actual) == self.base && arena.stage_of(actual) == self.stage
    }
}

fn aggregate_elements(arena: &TypeArena, aggregate: TypeId) -> Option<Vec<ExpectedElement>> {
    let outer_stage = arena.stage_of(aggregate);
    let elements: Vec<TypeId> = match arena.get(arena.strip_stage(aggregate))? {
        SemanticType::Product(elements) => elements.to_vec(),
        SemanticType::Record { fields, .. } => {
            fields.iter().map(|(_, field_type)| *field_type).collect()
        }
        _ => return None,
    };
    Some(
        elements
            .into_iter()
            .map(|element| ExpectedElement {
                base: arena.strip_stage(element),
                stage: Stage::join(outer_stage, arena.stage_of(element)),
            })
            .collect(),
    )
}

fn tuple_owner_is_valid(
    operands: &[ValueFact<'_>],
    result: ValueFact<'_>,
    arena: &TypeArena,
) -> bool {
    let linear = operands
        .iter()
        .filter(|operand| arena.is_quantum(operand.type_id))
        .copied()
        .collect::<Vec<_>>();
    match linear.as_slice() {
        [] => result.source.region.is_none(),
        [operand] => result.source.region.as_ref() == operand.source.region.as_ref(),
        operands => {
            let Some(result_region) = result.source.region.as_ref() else {
                return false;
            };
            sibling_parent(operands).as_ref() == Some(result_region)
                || operands.iter().all(|operand| {
                    operand
                        .source
                        .region
                        .as_ref()
                        .is_some_and(|region| region.root != result_region.root)
                })
        }
    }
}

fn sibling_parent(operands: &[ValueFact<'_>]) -> Option<wire::RegionRefProto> {
    let first = operands.first()?.source.region.as_ref()?;
    let (_, parent) = first.projections.split_last()?;
    for (index, operand) in operands.iter().enumerate() {
        let region = operand.source.region.as_ref()?;
        let (projection, candidate_parent) = region.projections.split_last()?;
        if region.root != first.root
            || candidate_parent != parent
            || projection != &index.to_string()
        {
            return None;
        }
    }
    Some(wire::RegionRefProto {
        root: first.root,
        projections: parent.to_vec(),
    })
}

fn unpack_owners_are_valid(
    product: ValueFact<'_>,
    results: &[ValueFact<'_>],
    arena: &TypeArena,
) -> bool {
    let linear_indices = results
        .iter()
        .enumerate()
        .filter_map(|(index, result)| arena.is_quantum(result.type_id).then_some(index))
        .collect::<Vec<_>>();
    for index in linear_indices.iter().copied() {
        let Some(owner) = product.source.region.as_ref() else {
            return false;
        };
        let expected = if linear_indices.len() == 1 {
            owner.clone()
        } else {
            let mut projected = owner.clone();
            projected.projections.push(index.to_string());
            projected
        };
        if results[index].source.region.as_ref() != Some(&expected) {
            return false;
        }
    }
    true
}

fn coherent_width_expression(
    arena: &TypeArena,
    type_id: TypeId,
) -> Option<(u64, HashMap<Dimension, u64>)> {
    let mut fixed = 0_u64;
    let mut symbolic = HashMap::<Dimension, u64>::new();
    let complete = arena
        .visit_coherent_value_leaves(type_id, &mut |leaf| {
            let mut add = |dimension: &Dimension| -> Option<()> {
                match dimension {
                    Dimension::Fixed(width) => fixed = fixed.checked_add(*width)?,
                    Dimension::Symbol { .. } => {
                        let count = symbolic.entry(dimension.clone()).or_default();
                        *count = count.checked_add(1)?;
                    }
                }
                Some(())
            };
            match leaf {
                CoherentLeafWidth::Bit => fixed = fixed.checked_add(1).ok_or(())?,
                CoherentLeafWidth::Word(width) => add(width).ok_or(())?,
                CoherentLeafWidth::Fixed {
                    integer_bits,
                    fractional_bits,
                } => {
                    add(integer_bits).ok_or(())?;
                    add(fractional_bits).ok_or(())?;
                }
            }
            Ok::<_, ()>(true)
        })
        .ok()?;
    complete.then_some((fixed, symbolic))
}

fn coherent_payload_width(arena: &TypeArena, type_id: TypeId) -> Option<u64> {
    arena.coherent_payload_width(type_id).ok().flatten()
}

fn quantum_payload(arena: &TypeArena, type_id: TypeId) -> Option<TypeId> {
    match arena.get(arena.strip_stage(type_id)) {
        Some(SemanticType::Quantum(payload)) => Some(*payload),
        _ => None,
    }
}

fn measurement_pair(
    arena: &TypeArena,
    result: TypeId,
    state: TypeId,
    payload: Option<TypeId>,
) -> bool {
    let Some(SemanticType::Product(elements)) = arena.get(result) else {
        return false;
    };
    let [measurement, returned_state] = elements.as_ref() else {
        return false;
    };
    if *returned_state != state {
        return false;
    }
    if let Some(payload) = payload {
        is_staged(arena, *measurement, Stage::Shot, payload)
    } else {
        let Some(SemanticType::Staged {
            stage: Stage::Shot,
            inner,
        }) = arena.get(*measurement)
        else {
            return false;
        };
        arena.get(*inner) == Some(&SemanticType::Scalar(ScalarType::Bool))
    }
}

fn is_staged(arena: &TypeArena, type_id: TypeId, expected: Stage, inner: TypeId) -> bool {
    matches!(
        arena.get(type_id),
        Some(SemanticType::Staged { stage, inner: actual })
            if *stage == expected && *actual == inner
    )
}

fn is_scalar(arena: &TypeArena, type_id: TypeId, expected: ScalarType) -> bool {
    arena.get(arena.strip_stage(type_id)) == Some(&SemanticType::Scalar(expected))
}

fn is_numeric_angle(arena: &TypeArena, type_id: TypeId) -> bool {
    matches!(
        arena.get(arena.strip_stage(type_id)),
        Some(SemanticType::Scalar(ScalarType::Int | ScalarType::Real))
    )
}

fn is_const_int(arena: &TypeArena, type_id: TypeId) -> bool {
    arena.get(type_id) == Some(&SemanticType::Scalar(ScalarType::Int))
        || matches!(
            arena.get(type_id),
            Some(SemanticType::Staged {
                stage: Stage::Const,
                inner,
            }) if arena.get(*inner) == Some(&SemanticType::Scalar(ScalarType::Int))
        )
}

fn is_const_int_vector(arena: &TypeArena, type_id: TypeId) -> bool {
    matches!(
        arena.get(type_id),
        Some(SemanticType::Staged {
            stage: Stage::Const,
            inner,
        }) if matches!(
            arena.get(*inner),
            Some(SemanticType::Array { element, shape })
                if shape.len() == 1
                    && arena.get(*element) == Some(&SemanticType::Scalar(ScalarType::Int))
        )
    )
}

fn is_const_observable(arena: &TypeArena, type_id: TypeId) -> bool {
    matches!(
        arena.get(type_id),
        Some(SemanticType::Staged {
            stage: Stage::Const,
            inner,
        }) if arena.get(*inner) == Some(&SemanticType::Scalar(ScalarType::Observable))
    )
}

fn require_owned_transform(
    arena: &TypeArena,
    operation: &wire::OperationProto,
    operands: &[ValueFact<'_>],
    results: &[ValueFact<'_>],
    minimum: usize,
    maximum: usize,
) -> Result<(), LocalEquationError> {
    let valid = (minimum..=maximum).contains(&operands.len())
        && operands.split_first().is_some_and(|(state, classical)| {
            quantum_payload(arena, state.type_id).is_some()
                && classical
                    .iter()
                    .all(|operand| !arena.is_quantum(operand.type_id))
                && matches!(results, [result]
                    if result.type_id == state.type_id
                        && result.source.region.as_ref() == state.source.region.as_ref())
        });
    if valid {
        Ok(())
    } else {
        Err(LocalEquationError::InvalidOwnedTransform {
            opcode: operation.opcode.clone(),
        })
    }
}

fn require_linear_transform(
    arena: &TypeArena,
    operation: &wire::OperationProto,
    operands: &[ValueFact<'_>],
    results: &[ValueFact<'_>],
) -> Result<(), LocalEquationError> {
    let valid = operands.split_first().is_some_and(|(state, classical)| {
        arena.is_quantum(state.type_id)
            && classical
                .iter()
                .all(|operand| !arena.is_quantum(operand.type_id))
            && matches!(results, [result]
                if result.type_id == state.type_id
                    && result.source.region.as_ref() == state.source.region.as_ref())
    });
    if valid {
        Ok(())
    } else {
        Err(LocalEquationError::InvalidOwnedTransform {
            opcode: operation.opcode.clone(),
        })
    }
}

fn ordered_subset(values: &[&str], allowed: &[&str]) -> bool {
    let mut cursor = 0;
    for value in values {
        let Some(offset) = allowed[cursor..]
            .iter()
            .position(|allowed| allowed == value)
        else {
            return false;
        };
        cursor += offset + 1;
    }
    true
}

fn exact_attributes<'a>(
    operation: &'a wire::OperationProto,
    names: &[&str],
) -> Option<Vec<&'a wire::LiteralProto>> {
    if operation.attributes.len() != names.len() {
        return None;
    }
    operation
        .attributes
        .iter()
        .zip(names)
        .map(|(attribute, name)| {
            (attribute.name == *name)
                .then_some(attribute.value.as_ref())
                .flatten()
        })
        .collect()
}

fn structural_proof_marker(operation: &wire::OperationProto) -> Result<bool, ()> {
    if operation.attributes.is_empty() {
        return Ok(false);
    }
    let Some(attributes) = exact_attributes(operation, &["proof"]) else {
        return Err(());
    };
    let [proof] = attributes.as_slice() else {
        return Err(());
    };
    if string_literal(proof) == Some("modular-adjoint") {
        Ok(true)
    } else {
        Err(())
    }
}

fn cleanup_attributes(operation: &wire::OperationProto) -> Option<u64> {
    let attributes = exact_attributes(operation, &["origin", "proof"])?;
    let [origin, proof] = attributes.as_slice() else {
        return None;
    };
    if string_literal(proof) != Some("modular-adjoint") {
        return None;
    }
    nonnegative_u64(origin)
}

fn keyword_tuple(operation: &wire::OperationProto) -> Option<Vec<&str>> {
    let attributes = exact_attributes(operation, &["keywords"])?;
    let [keywords] = attributes.as_slice() else {
        return None;
    };
    string_tuple(keywords)
}

fn string_literal(literal: &wire::LiteralProto) -> Option<&str> {
    match literal.value.as_ref()? {
        wire::literal_proto::Value::StringValue(value) => Some(value),
        _ => None,
    }
}

fn string_tuple(literal: &wire::LiteralProto) -> Option<Vec<&str>> {
    let wire::literal_proto::Value::TupleValue(sequence) = literal.value.as_ref()? else {
        return None;
    };
    sequence.values.iter().map(string_literal).collect()
}

fn mapping_literal(literal: &wire::LiteralProto) -> Option<&[wire::literal_proto::Entry]> {
    let wire::literal_proto::Value::MappingValue(mapping) = literal.value.as_ref()? else {
        return None;
    };
    Some(&mapping.entries)
}

fn positive_u64(literal: &wire::LiteralProto) -> Option<u64> {
    nonnegative_u64(literal).filter(|value| *value > 0)
}

fn nonnegative_u64(literal: &wire::LiteralProto) -> Option<u64> {
    match literal.value.as_ref()? {
        wire::literal_proto::Value::IntValue(value) => u64::try_from(*value).ok(),
        wire::literal_proto::Value::BigintValue(encoded) if encoded.len() == 9 => {
            let (&0, bytes) = encoded.split_first()? else {
                return None;
            };
            let bytes: [u8; 8] = bytes.try_into().ok()?;
            Some(u64::from_be_bytes(bytes))
        }
        _ => None,
    }
}

pub(crate) fn literal_conforms(
    arena: &TypeArena,
    type_id: TypeId,
    literal: &wire::LiteralProto,
) -> bool {
    let base = arena.strip_stage(type_id);
    let Some(type_spec) = arena.get(base) else {
        return false;
    };
    match type_spec {
        SemanticType::Scalar(scalar) => scalar_literal_conforms(*scalar, literal),
        SemanticType::Array { element, shape } => {
            let Some(items) = tuple_literal(literal) else {
                return false;
            };
            sequence_conforms(arena, *element, shape, items)
        }
        SemanticType::Word { kind, width } => match kind {
            WordKind::Bits => {
                let Some(items) = tuple_literal(literal) else {
                    return false;
                };
                dimension_accepts(width, items.len())
                    && items
                        .iter()
                        .all(|item| scalar_literal_conforms(ScalarType::Bit, item))
            }
            WordKind::UInt => uint_literal_conforms(width, literal),
        },
        SemanticType::Fixed {
            integer_bits,
            fractional_bits,
        } => fixed_literal_conforms(integer_bits, fractional_bits, literal),
        SemanticType::Product(elements) => tuple_literal(literal).is_some_and(|items| {
            items.len() == elements.len()
                && items
                    .iter()
                    .zip(elements)
                    .all(|(item, element)| literal_conforms(arena, *element, item))
        }),
        SemanticType::Record { fields, .. } => mapping_literal(literal).is_some_and(|entries| {
            if entries.len() != fields.len() {
                return false;
            }
            let values = entries
                .iter()
                .filter_map(|entry| {
                    entry
                        .value
                        .as_ref()
                        .map(|value| (entry.key.as_str(), value))
                })
                .collect::<HashMap<_, _>>();
            values.len() == fields.len()
                && fields.iter().all(|(name, field_type)| {
                    values
                        .get(name.as_str())
                        .is_some_and(|value| literal_conforms(arena, *field_type, value))
                })
        }),
        SemanticType::Quantum(_) | SemanticType::Staged { .. } | SemanticType::Function { .. } => {
            false
        }
    }
}

fn scalar_literal_conforms(scalar: ScalarType, literal: &wire::LiteralProto) -> bool {
    use wire::literal_proto::Value;

    let Some(value) = literal.value.as_ref() else {
        return false;
    };
    match scalar {
        ScalarType::Unit => matches!(value, Value::UnitValue(true)),
        ScalarType::Bool => matches!(value, Value::BoolValue(_)),
        ScalarType::Bit => matches!(value, Value::BoolValue(_) | Value::IntValue(0 | 1)),
        ScalarType::Int => matches!(value, Value::IntValue(_) | Value::BigintValue(_)),
        ScalarType::Real => matches!(
            value,
            Value::IntValue(_) | Value::BigintValue(_) | Value::RealValue(_)
        ),
        ScalarType::Complex => matches!(
            value,
            Value::IntValue(_)
                | Value::BigintValue(_)
                | Value::RealValue(_)
                | Value::ComplexValue(_)
        ),
        ScalarType::Probability => match value {
            Value::IntValue(value) => (0..=1).contains(value),
            Value::RealValue(value) => value.is_finite() && (0.0..=1.0).contains(value),
            _ => false,
        },
        ScalarType::Observable => observable_literal_conforms(literal),
    }
}

fn uint_literal_conforms(width: &Dimension, literal: &wire::LiteralProto) -> bool {
    use wire::literal_proto::Value;

    let value = match literal.value.as_ref() {
        Some(Value::IntValue(value)) => BigInt::from(*value),
        Some(Value::BigintValue(value)) => BigInt::from_signed_bytes_be(value),
        _ => return false,
    };
    if value.sign() == Sign::Minus {
        return false;
    }
    match width {
        Dimension::Fixed(width) => value.bits() <= *width,
        Dimension::Symbol { .. } => true,
    }
}

fn sequence_conforms(
    arena: &TypeArena,
    element: TypeId,
    shape: &[Dimension],
    values: &[wire::LiteralProto],
) -> bool {
    let Some((extent, remaining)) = shape.split_first() else {
        return false;
    };
    if !dimension_accepts(extent, values.len()) {
        return false;
    }
    if remaining.is_empty() {
        return values
            .iter()
            .all(|value| literal_conforms(arena, element, value));
    }
    values.iter().all(|value| {
        tuple_literal(value)
            .is_some_and(|nested| sequence_conforms(arena, element, remaining, nested))
    })
}

fn dimension_accepts(dimension: &Dimension, length: usize) -> bool {
    match dimension {
        Dimension::Fixed(expected) => u64::try_from(length) == Ok(*expected),
        Dimension::Symbol { .. } => true,
    }
}

fn tuple_literal(literal: &wire::LiteralProto) -> Option<&[wire::LiteralProto]> {
    let wire::literal_proto::Value::TupleValue(sequence) = literal.value.as_ref()? else {
        return None;
    };
    Some(&sequence.values)
}

fn fixed_literal_conforms(
    integer_bits: &Dimension,
    fractional_bits: &Dimension,
    literal: &wire::LiteralProto,
) -> bool {
    use wire::literal_proto::Value;

    match literal.value.as_ref() {
        Some(Value::IntValue(value)) => {
            *value >= 0
                && match integer_bits {
                    Dimension::Symbol { .. } => true,
                    Dimension::Fixed(integer_bits) => {
                        *integer_bits >= 63
                            || u32::try_from(*integer_bits).is_ok_and(|width| {
                                u64::try_from(*value).is_ok_and(|value| {
                                    1_u64.checked_shl(width).is_some_and(|limit| value < limit)
                                })
                            })
                    }
                }
        }
        Some(Value::BigintValue(encoded)) => {
            encoded.first().is_some_and(|first| first & 0x80 == 0)
                && match integer_bits {
                    Dimension::Fixed(width) => positive_bigint_bit_length(encoded) <= *width,
                    Dimension::Symbol { .. } => true,
                }
        }
        Some(Value::RealValue(value)) => {
            value.is_finite()
                && *value >= 0.0
                && match integer_bits {
                    Dimension::Fixed(width) => below_binary_power(*value, *width),
                    Dimension::Symbol { .. } => true,
                }
                && match fractional_bits {
                    Dimension::Fixed(width) => dyadic_fraction_bits(*value) <= *width,
                    Dimension::Symbol { .. } => true,
                }
        }
        _ => false,
    }
}

fn positive_bigint_bit_length(encoded: &[u8]) -> u64 {
    let significant = encoded.strip_prefix(&[0]).unwrap_or(encoded);
    let Some(first) = significant.first() else {
        return 0;
    };
    let leading = u64::from(first.leading_zeros());
    u64::try_from(significant.len())
        .unwrap_or(u64::MAX)
        .saturating_mul(8)
        .saturating_sub(leading)
}

fn below_binary_power(value: f64, exponent: u64) -> bool {
    if exponent > 1023 {
        true
    } else {
        i32::try_from(exponent).is_ok_and(|exponent| value < 2_f64.powi(exponent))
    }
}

fn dyadic_fraction_bits(value: f64) -> u64 {
    if value == 0.0 {
        return 0;
    }
    let bits = value.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1_u64 << 52) - 1);
    let (significand, binary_exponent) = if exponent == 0 {
        (fraction, -1074)
    } else {
        ((1_u64 << 52) | fraction, exponent - 1023 - 52)
    };
    if binary_exponent >= 0 {
        return 0;
    }
    u64::try_from(-binary_exponent)
        .unwrap_or(u64::MAX)
        .saturating_sub(u64::from(significand.trailing_zeros()))
}

fn observable_literal_conforms(literal: &wire::LiteralProto) -> bool {
    let Some(entries) = mapping_literal(literal) else {
        return false;
    };
    if entries.len() != 4 {
        return false;
    }
    let get = |key: &str| {
        entries
            .iter()
            .find(|entry| entry.key == key)
            .and_then(|entry| entry.value.as_ref())
    };
    if get("kind").and_then(string_literal) != Some("qamomile.pauli_sum.v1")
        || get("num_qubits").and_then(nonnegative_u64).is_none()
        || get("constant").is_none_or(|value| !numeric_literal(value))
    {
        return false;
    }
    let Some(terms) = get("terms").and_then(tuple_literal) else {
        return false;
    };
    terms.iter().all(|term| {
        let Some(items) = tuple_literal(term) else {
            return false;
        };
        let [coefficient, factors] = items else {
            return false;
        };
        numeric_literal(coefficient)
            && tuple_literal(factors).is_some_and(|factors| {
                let mut indices = HashSet::with_capacity(factors.len());
                for factor in factors {
                    let Some(parts) = tuple_literal(factor) else {
                        return false;
                    };
                    let [pauli, index] = parts else {
                        return false;
                    };
                    if !matches!(string_literal(pauli), Some("X" | "Y" | "Z")) {
                        return false;
                    }
                    let Some(index) = nonnegative_u64(index) else {
                        return false;
                    };
                    if !indices.insert(index) {
                        return false;
                    }
                }
                true
            })
    })
}

fn numeric_literal(literal: &wire::LiteralProto) -> bool {
    matches!(
        literal.value.as_ref(),
        Some(
            wire::literal_proto::Value::IntValue(_)
                | wire::literal_proto::Value::BigintValue(_)
                | wire::literal_proto::Value::RealValue(_)
                | wire::literal_proto::Value::ComplexValue(_)
        )
    )
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn word(kind: &str, dimension: wire::dimension_proto::Value) -> wire::TypeProto {
        wire::TypeProto {
            kind: wire::type_proto::Kind::Array.into(),
            name: kind.to_owned(),
            arguments: vec![scalar("Bit")],
            dimensions: vec![wire::DimensionProto {
                value: Some(dimension),
            }],
            ..Default::default()
        }
    }

    fn bits(width: u64) -> wire::TypeProto {
        word("Bits", wire::dimension_proto::Value::Fixed(width))
    }

    fn symbolic_bits(name: &str) -> wire::TypeProto {
        word(
            "Bits",
            wire::dimension_proto::Value::Symbol(name.to_owned()),
        )
    }

    fn quantum(payload: wire::TypeProto) -> wire::TypeProto {
        wire::TypeProto {
            kind: wire::type_proto::Kind::Quantum.into(),
            arguments: vec![payload],
            ..Default::default()
        }
    }

    fn qbits(width: u64) -> wire::TypeProto {
        quantum(bits(width))
    }

    fn product(elements: Vec<wire::TypeProto>) -> wire::TypeProto {
        wire::TypeProto {
            kind: wire::type_proto::Kind::Product.into(),
            arguments: elements,
            ..Default::default()
        }
    }

    fn record(name: &str, fields: Vec<(&str, wire::TypeProto)>) -> wire::TypeProto {
        let (field_names, arguments) = fields
            .into_iter()
            .map(|(field, r#type)| (field.to_owned(), r#type))
            .unzip();
        wire::TypeProto {
            kind: wire::type_proto::Kind::Record.into(),
            name: name.to_owned(),
            arguments,
            field_names,
            ..Default::default()
        }
    }

    fn array(element: wire::TypeProto, width: u64) -> wire::TypeProto {
        wire::TypeProto {
            kind: wire::type_proto::Kind::Array.into(),
            name: "Array".to_owned(),
            arguments: vec![element],
            dimensions: vec![wire::DimensionProto {
                value: Some(wire::dimension_proto::Value::Fixed(width)),
            }],
            ..Default::default()
        }
    }

    fn fixed(integer_bits: u64, fractional_bits: u64) -> wire::TypeProto {
        wire::TypeProto {
            kind: wire::type_proto::Kind::Fixed.into(),
            name: "UFixed".to_owned(),
            dimensions: vec![
                wire::DimensionProto {
                    value: Some(wire::dimension_proto::Value::Fixed(integer_bits)),
                },
                wire::DimensionProto {
                    value: Some(wire::dimension_proto::Value::Fixed(fractional_bits)),
                },
            ],
            ..Default::default()
        }
    }

    fn region(root: u64, projections: &[&str]) -> wire::RegionRefProto {
        wire::RegionRefProto {
            root,
            projections: projections.iter().map(|item| (*item).to_owned()).collect(),
        }
    }

    fn value(
        id: u64,
        r#type: wire::TypeProto,
        ownership: Option<wire::RegionRefProto>,
    ) -> wire::ValueProto {
        wire::ValueProto {
            id,
            r#type: Some(r#type),
            has_region: ownership.is_some(),
            region: ownership,
            ..Default::default()
        }
    }

    fn literal(value: wire::literal_proto::Value) -> wire::LiteralProto {
        wire::LiteralProto { value: Some(value) }
    }

    fn attribute(name: &str, value: wire::literal_proto::Value) -> wire::AttributeProto {
        wire::AttributeProto {
            name: name.to_owned(),
            value: Some(literal(value)),
        }
    }

    fn integer(value: i64) -> wire::literal_proto::Value {
        wire::literal_proto::Value::IntValue(value)
    }

    fn string(value: &str) -> wire::literal_proto::Value {
        wire::literal_proto::Value::StringValue(value.to_owned())
    }

    fn tuple(values: Vec<wire::literal_proto::Value>) -> wire::literal_proto::Value {
        wire::literal_proto::Value::TupleValue(wire::literal_proto::Sequence {
            values: values.into_iter().map(literal).collect(),
        })
    }

    fn strings(values: &[&str]) -> wire::literal_proto::Value {
        tuple(values.iter().map(|value| string(value)).collect())
    }

    fn mapping(entries: Vec<(&str, wire::literal_proto::Value)>) -> wire::literal_proto::Value {
        wire::literal_proto::Value::MappingValue(wire::literal_proto::Mapping {
            entries: entries
                .into_iter()
                .map(|(key, value)| wire::literal_proto::Entry {
                    key: key.to_owned(),
                    value: Some(literal(value)),
                })
                .collect(),
        })
    }

    fn parameter(
        name: &str,
        id: u64,
        r#type: wire::TypeProto,
        ownership: Option<wire::RegionRefProto>,
    ) -> wire::ParameterProto {
        wire::ParameterProto {
            name: name.to_owned(),
            value: Some(value(id, r#type, ownership)),
            ..Default::default()
        }
    }

    fn package(
        parameters: Vec<wire::ParameterProto>,
        operations: Vec<wire::OperationProto>,
    ) -> wire::QCorePackageProto {
        wire::QCorePackageProto {
            functions: vec![wire::FunctionProto {
                symbol: "equations.main".to_owned(),
                parameters,
                result_type: Some(scalar("Unit")),
                body: Some(wire::BlockProto {
                    operations,
                    terminator: Some(wire::TerminatorProto {
                        kind: "unreachable".to_owned(),
                        ..Default::default()
                    }),
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn operation(
        opcode: &str,
        operands: Vec<u64>,
        results: Vec<wire::ValueProto>,
        attributes: Vec<wire::AttributeProto>,
    ) -> wire::OperationProto {
        wire::OperationProto {
            opcode: opcode.to_owned(),
            operands,
            results,
            attributes,
            ..Default::default()
        }
    }

    fn check(package: &wire::QCorePackageProto) -> Result<LocalEquationFacts, LocalEquationError> {
        let mut arena = crate::types::validate_program_types(package).expect("valid test types");
        let operations =
            crate::operations::validate_program_operations_with_types(package, &mut arena)
                .expect("valid test operation kernel");
        verify_program(package, &arena, &operations)
    }

    fn check_module(
        package: &wire::QCorePackageProto,
    ) -> Result<LocalEquationFacts, LocalEquationError> {
        let mut arena = crate::types::validate_program_types(package).expect("valid test types");
        let operations =
            crate::operations::validate_program_operations_with_types(package, &mut arena)
                .expect("valid test operation kernel");
        verify_module(package, &arena, &operations)
    }

    fn single_quantum_operation(
        opcode: &str,
        operand_types: Vec<(wire::TypeProto, Option<wire::RegionRefProto>)>,
        result_types: Vec<(wire::TypeProto, Option<wire::RegionRefProto>)>,
        attributes: Vec<wire::AttributeProto>,
    ) -> Result<LocalEquationFacts, LocalEquationError> {
        let parameters = operand_types
            .into_iter()
            .enumerate()
            .map(|(index, (r#type, ownership))| {
                let id = u64::try_from(index).expect("small test index") + 1;
                parameter(&format!("operand_{index}"), id, r#type, ownership)
            })
            .collect::<Vec<_>>();
        let operands = parameters
            .iter()
            .map(|parameter| parameter.value.as_ref().expect("test value").id)
            .collect();
        let results = result_types
            .into_iter()
            .enumerate()
            .map(|(index, (r#type, ownership))| {
                value(
                    u64::try_from(index).expect("small result index") + 100,
                    r#type,
                    ownership,
                )
            })
            .collect();
        check(&package(
            parameters,
            vec![operation(opcode, operands, results, attributes)],
        ))
    }

    #[test]
    fn constants_validate_nested_arrays_words_and_records() {
        let nested = operation(
            "qcore.constant",
            Vec::new(),
            vec![value(
                1,
                array(product(vec![scalar("Int"), bits(2)]), 2),
                None,
            )],
            vec![attribute(
                "value",
                tuple(vec![
                    tuple(vec![integer(1), tuple(vec![integer(0), integer(1)])]),
                    tuple(vec![integer(2), tuple(vec![integer(1), integer(0)])]),
                ]),
            )],
        );
        assert!(check(&package(Vec::new(), vec![nested])).is_ok());

        let wrong_shape = operation(
            "qcore.constant",
            Vec::new(),
            vec![value(1, array(scalar("Int"), 2), None)],
            vec![attribute("value", tuple(vec![integer(1)]))],
        );
        assert_eq!(
            check(&package(Vec::new(), vec![wrong_shape])),
            Err(malformed_core("qcore.constant operation"))
        );

        let record_type = record("Header", vec![("count", scalar("Int")), ("mask", bits(2))]);
        let record_constant = operation(
            "qcore.constant",
            Vec::new(),
            vec![value(1, record_type.clone(), None)],
            vec![attribute(
                "value",
                mapping(vec![
                    ("count", integer(2)),
                    ("mask", tuple(vec![integer(1), integer(0)])),
                ]),
            )],
        );
        assert!(check(&package(Vec::new(), vec![record_constant])).is_ok());

        let incomplete_record = operation(
            "qcore.constant",
            Vec::new(),
            vec![value(1, record_type, None)],
            vec![attribute("value", mapping(vec![("count", integer(2))]))],
        );
        assert_eq!(
            check(&package(Vec::new(), vec![incomplete_record])),
            Err(malformed_core("qcore.constant operation"))
        );
    }

    #[test]
    fn constants_validate_probability_fixed_and_observable() {
        let bad_probability = operation(
            "qcore.constant",
            Vec::new(),
            vec![value(1, scalar("Probability"), None)],
            vec![attribute(
                "value",
                wire::literal_proto::Value::RealValue(1.01),
            )],
        );
        assert!(check(&package(Vec::new(), vec![bad_probability])).is_err());

        let bad_fixed = operation(
            "qcore.constant",
            Vec::new(),
            vec![value(1, fixed(2, 2), None)],
            vec![attribute(
                "value",
                wire::literal_proto::Value::RealValue(0.3),
            )],
        );
        assert!(check(&package(Vec::new(), vec![bad_fixed])).is_err());

        let observable = mapping(vec![
            (
                "constant",
                wire::literal_proto::Value::ComplexValue(wire::literal_proto::Complex {
                    real: 0.0,
                    imaginary: 0.0,
                }),
            ),
            ("kind", string("qamomile.pauli_sum.v1")),
            ("num_qubits", integer(1)),
            (
                "terms",
                tuple(vec![tuple(vec![
                    wire::literal_proto::Value::RealValue(1.0),
                    tuple(vec![tuple(vec![string("Z"), integer(0)])]),
                ])]),
            ),
        ]);
        let observable_constant = operation(
            "qcore.constant",
            Vec::new(),
            vec![value(1, staged("Const", scalar("Observable")), None)],
            vec![attribute("value", observable)],
        );
        assert!(check(&package(Vec::new(), vec![observable_constant])).is_ok());

        let forged_observable = operation(
            "qcore.constant",
            Vec::new(),
            vec![value(1, scalar("Observable"), None)],
            vec![attribute(
                "value",
                mapping(vec![
                    (
                        "constant",
                        wire::literal_proto::Value::ComplexValue(wire::literal_proto::Complex {
                            real: 0.0,
                            imaginary: 0.0,
                        }),
                    ),
                    ("kind", string("qamomile.pauli_sum.v1")),
                    ("num_qubits", integer(1)),
                    (
                        "terms",
                        tuple(vec![tuple(vec![
                            wire::literal_proto::Value::RealValue(1.0),
                            tuple(vec![tuple(vec![string("A"), integer(0)])]),
                        ])]),
                    ),
                ]),
            )],
        );
        assert_eq!(
            check(&package(Vec::new(), vec![forged_observable])),
            Err(malformed_core("qcore.constant operation"))
        );
    }

    #[test]
    fn stage_lift_requires_a_strict_availability_increase_and_no_attributes() {
        let shot_parameter = parameter("source", 1, staged("Shot", scalar("Int")), None);
        let downgrade = operation(
            "qcore.stage_lift",
            vec![1],
            vec![value(2, staged("Param", scalar("Int")), None)],
            Vec::new(),
        );
        assert_eq!(
            check(&package(vec![shot_parameter], vec![downgrade])),
            Err(malformed_core("qcore.stage_lift operation"))
        );

        let plain_parameter = parameter("source", 1, scalar("Int"), None);
        let valid = operation(
            "qcore.stage_lift",
            vec![1],
            vec![value(2, staged("Const", scalar("Int")), None)],
            Vec::new(),
        );
        assert!(check(&package(vec![plain_parameter], vec![valid])).is_ok());
    }

    #[test]
    fn tuple_unpack_and_reshape_enforce_types_and_owner_topology() {
        let state = qbits(1);
        let pair = product(vec![state.clone(), state.clone()]);
        let tuple_operation = operation(
            "qcore.tuple",
            vec![1, 2],
            vec![value(3, pair.clone(), Some(region(12, &[])))],
            Vec::new(),
        );
        assert!(
            check(&package(
                vec![
                    parameter("left", 1, state.clone(), Some(region(10, &[]))),
                    parameter("right", 2, state.clone(), Some(region(11, &[]))),
                ],
                vec![tuple_operation],
            ))
            .is_ok()
        );

        let partial_sibling_tuple = operation(
            "qcore.tuple",
            vec![1, 2],
            vec![value(3, pair.clone(), Some(region(31, &[])))],
            Vec::new(),
        );
        assert!(
            check(&package(
                vec![
                    parameter("left", 1, state.clone(), Some(region(30, &["0"]))),
                    parameter("middle", 2, state.clone(), Some(region(30, &["1"]))),
                ],
                vec![partial_sibling_tuple],
            ))
            .is_ok()
        );

        let forged_sibling_tuple = operation(
            "qcore.tuple",
            vec![1, 2],
            vec![value(3, pair.clone(), Some(region(30, &["other"])))],
            Vec::new(),
        );
        assert_eq!(
            check(&package(
                vec![
                    parameter("left", 1, state.clone(), Some(region(30, &["0"]))),
                    parameter("middle", 2, state.clone(), Some(region(30, &["1"]))),
                ],
                vec![forged_sibling_tuple],
            )),
            Err(malformed_core("tuple construction"))
        );

        let valid_unpack = operation(
            "qcore.unpack",
            vec![1],
            vec![
                value(2, state.clone(), Some(region(12, &["0"]))),
                value(3, state.clone(), Some(region(12, &["1"]))),
            ],
            Vec::new(),
        );
        assert!(
            check(&package(
                vec![parameter("pair", 1, pair.clone(), Some(region(12, &[])))],
                vec![valid_unpack],
            ))
            .is_ok()
        );

        let forged_unpack = operation(
            "qcore.unpack",
            vec![1],
            vec![
                value(2, state.clone(), Some(region(10, &[]))),
                value(3, state.clone(), Some(region(11, &[]))),
            ],
            vec![attribute("proof", string("modular-adjoint"))],
        );
        let deferred = check(&package(
            vec![parameter("pair", 1, pair, Some(region(12, &[])))],
            vec![forged_unpack],
        ))
        .expect("proof-marked owner rebasing is deferred");
        assert_eq!(deferred.proof_obligation_count(), 1);

        let reshape = operation(
            "qcore.reshape",
            vec![1],
            vec![value(
                2,
                product(vec![quantum(scalar("Bit")), quantum(scalar("Bit"))]),
                Some(region(20, &[])),
            )],
            Vec::new(),
        );
        assert!(
            check(&package(
                vec![parameter("state", 1, qbits(2), Some(region(20, &[])))],
                vec![reshape],
            ))
            .is_ok()
        );
    }

    #[test]
    fn allocation_width_requires_static_evidence_and_exact_result_width() {
        let direct = package(
            Vec::new(),
            vec![
                operation(
                    "qcore.constant",
                    Vec::new(),
                    vec![value(1, scalar("Int"), None)],
                    vec![attribute("value", integer(3))],
                ),
                operation(
                    "quantum.alloc",
                    vec![1],
                    vec![value(2, qbits(3), Some(region(1, &[])))],
                    Vec::new(),
                ),
            ],
        );
        assert!(check(&direct).is_ok());

        let mut forged = direct.clone();
        forged.functions[0]
            .body
            .as_mut()
            .expect("test body")
            .operations[1]
            .results[0]
            .r#type = Some(qbits(2));
        assert!(matches!(
            check(&forged),
            Err(LocalEquationError::MalformedQuantum { .. })
        ));

        let dependent = single_quantum_operation(
            "quantum.alloc",
            vec![(staged("Const", scalar("Int")), None)],
            vec![(quantum(symbolic_bits("operand_0")), Some(region(1, &[])))],
            Vec::new(),
        );
        assert!(dependent.is_ok());

        for stage in [None, Some("Param"), Some("Shot")] {
            let width_type =
                stage.map_or_else(|| scalar("Int"), |stage| staged(stage, scalar("Int")));
            let result = single_quantum_operation(
                "quantum.alloc",
                vec![(width_type, None)],
                vec![(quantum(symbolic_bits("operand_0")), Some(region(1, &[])))],
                Vec::new(),
            );
            assert_eq!(result, Err(LocalEquationError::UnresolvedAllocationWidth));
        }
    }

    #[test]
    fn computed_allocation_width_is_not_treated_as_shape_proof() {
        let image = package(
            vec![
                parameter("left", 1, staged("Const", scalar("Int")), None),
                parameter("right", 2, staged("Const", scalar("Int")), None),
            ],
            vec![
                operation(
                    "classical.add",
                    vec![1, 2],
                    vec![value(3, staged("Const", scalar("Int")), None)],
                    Vec::new(),
                ),
                operation(
                    "quantum.alloc",
                    vec![3],
                    vec![value(
                        4,
                        quantum(symbolic_bits("left")),
                        Some(region(1, &[])),
                    )],
                    Vec::new(),
                ),
            ],
        );
        assert_eq!(
            check(&image),
            Err(LocalEquationError::UnresolvedAllocationWidth)
        );
        assert!(check_module(&image).is_ok());
    }

    #[test]
    fn prepare_and_unprepare_require_the_same_coherent_payload() {
        let prepared = single_quantum_operation(
            "quantum.prepare",
            vec![(staged("Param", bits(3)), None)],
            vec![(qbits(3), Some(region(1, &[])))],
            Vec::new(),
        );
        assert!(prepared.is_ok());

        let forged = single_quantum_operation(
            "quantum.prepare",
            vec![(staged("Param", bits(3)), None)],
            vec![(qbits(2), Some(region(1, &[])))],
            Vec::new(),
        );
        assert!(matches!(
            forged,
            Err(LocalEquationError::MalformedQuantum { .. })
        ));

        let cleanup = vec![
            attribute("origin", integer(0)),
            attribute("proof", string("modular-adjoint")),
        ];
        let unprepare = single_quantum_operation(
            "quantum.unprepare",
            vec![
                (qbits(3), Some(region(1, &[]))),
                (staged("Param", bits(2)), None),
            ],
            Vec::new(),
            cleanup,
        );
        assert!(matches!(
            unprepare,
            Err(LocalEquationError::MalformedQuantum { .. })
        ));
    }

    #[test]
    fn every_primitive_gate_and_adjoint_has_one_closed_selector_equation() {
        let gates: &[(&str, bool, &[&str])] = &[
            ("quantum.h", false, &["target"]),
            ("quantum.x", false, &["target"]),
            ("quantum.y", false, &["target"]),
            ("quantum.z", false, &["target"]),
            ("quantum.s", false, &["target"]),
            ("quantum.sdg", false, &["target"]),
            ("quantum.t", false, &["target"]),
            ("quantum.tdg", false, &["target"]),
            ("quantum.p", true, &["target"]),
            ("quantum.rx", true, &["target"]),
            ("quantum.ry", true, &["target"]),
            ("quantum.rz", true, &["target"]),
            ("quantum.cx", false, &["control", "target"]),
            ("quantum.cz", false, &["control", "target"]),
            ("quantum.ccx", false, &["control1", "control2", "target"]),
            ("quantum.cp", true, &["control", "target"]),
            ("quantum.rzz", true, &["left", "right"]),
            ("quantum.swap", false, &["left", "right"]),
        ];
        for (opcode, has_angle, selectors) in gates {
            let mut operands = vec![(qbits(4), Some(region(1, &[])))];
            if *has_angle {
                operands.push((staged("Param", scalar("Real")), None));
            }
            operands.extend(selectors.iter().map(|_| (scalar("Int"), None)));
            let direct = single_quantum_operation(
                opcode,
                operands.clone(),
                vec![(qbits(4), Some(region(1, &[])))],
                vec![attribute("keywords", strings(selectors))],
            );
            assert!(direct.is_ok(), "failed direct {opcode}: {direct:?}");

            let original_attributes = mapping(vec![("keywords", strings(selectors))]);
            let adjoint = single_quantum_operation(
                "qcore.adjoint",
                operands,
                vec![(qbits(4), Some(region(1, &[])))],
                vec![
                    attribute("original", string(opcode)),
                    attribute("original_attributes", original_attributes),
                    attribute("proof", string("modular-adjoint")),
                ],
            );
            assert!(adjoint.is_ok(), "failed adjoint {opcode}: {adjoint:?}");
        }
    }

    #[test]
    fn selector_order_and_adjoint_attribute_order_are_semantic() {
        let reordered_selectors = single_quantum_operation(
            "quantum.cx",
            vec![
                (qbits(2), Some(region(1, &[]))),
                (scalar("Int"), None),
                (scalar("Int"), None),
            ],
            vec![(qbits(2), Some(region(1, &[])))],
            vec![attribute("keywords", strings(&["target", "control"]))],
        );
        assert!(matches!(
            reordered_selectors,
            Err(LocalEquationError::MalformedQuantum { .. })
        ));

        let permuted_attributes = single_quantum_operation(
            "qcore.adjoint",
            vec![(qbits(1), Some(region(1, &[]))), (scalar("Int"), None)],
            vec![(qbits(1), Some(region(1, &[])))],
            vec![
                attribute(
                    "original_attributes",
                    mapping(vec![("keywords", strings(&["target"]))]),
                ),
                attribute("original", string("quantum.h")),
                attribute("proof", string("modular-adjoint")),
            ],
        );
        assert!(matches!(
            permuted_attributes,
            Err(LocalEquationError::MalformedQuantum { .. })
        ));

        let permuted_cleanup = single_quantum_operation(
            "quantum.release",
            vec![(qbits(1), Some(region(1, &[])))],
            Vec::new(),
            vec![
                attribute("proof", string("modular-adjoint")),
                attribute("origin", integer(0)),
            ],
        );
        assert!(matches!(
            permuted_cleanup,
            Err(LocalEquationError::MalformedQuantum { .. })
        ));
    }

    #[test]
    fn measurement_channel_and_cleanup_equations_are_closed() {
        let state = qbits(2);
        let shot_bits = staged("Shot", bits(2));
        let shot_bool = staged("Shot", scalar("Bool"));
        let measured = single_quantum_operation(
            "quantum.measure",
            vec![(state.clone(), Some(region(1, &[])))],
            vec![(shot_bits.clone(), None)],
            Vec::new(),
        );
        assert!(measured.is_ok());

        for opcode in [
            "quantum.measure_x",
            "quantum.measure_y",
            "quantum.measure_z",
        ] {
            let projected = single_quantum_operation(
                opcode,
                vec![(state.clone(), Some(region(1, &[]))), (scalar("Int"), None)],
                vec![(
                    product(vec![shot_bool.clone(), state.clone()]),
                    Some(region(1, &[])),
                )],
                vec![attribute("keywords", strings(&["target"]))],
            );
            assert!(projected.is_ok(), "failed {opcode}: {projected:?}");
        }

        let measured_bit = single_quantum_operation(
            "quantum.measure_z",
            vec![(state.clone(), Some(region(1, &[]))), (scalar("Int"), None)],
            vec![(
                product(vec![shot_bool, state.clone()]),
                Some(region(1, &[])),
            )],
            vec![attribute("keywords", strings(&["target"]))],
        );
        assert!(measured_bit.is_ok());

        let measured_single = single_quantum_operation(
            "quantum.measure_z",
            vec![(quantum(scalar("Bit")), Some(region(2, &[])))],
            vec![(
                product(vec![staged("Shot", scalar("Bool")), quantum(scalar("Bit"))]),
                Some(region(2, &[])),
            )],
            vec![attribute("keywords", strings(&[]))],
        );
        assert!(measured_single.is_ok());

        let measured_reset = single_quantum_operation(
            "quantum.measure_reset",
            vec![(state.clone(), Some(region(1, &[])))],
            vec![(
                product(vec![shot_bits, state.clone()]),
                Some(region(1, &[])),
            )],
            Vec::new(),
        );
        assert!(measured_reset.is_ok());

        let discarded = single_quantum_operation(
            "quantum.discard",
            vec![(state.clone(), Some(region(1, &[])))],
            vec![(scalar("Unit"), None)],
            Vec::new(),
        );
        assert!(discarded.is_ok());

        let reset = single_quantum_operation(
            "quantum.reset",
            vec![(state.clone(), Some(region(1, &[])))],
            vec![(state.clone(), Some(region(1, &[])))],
            Vec::new(),
        );
        assert!(reset.is_ok());

        let released = single_quantum_operation(
            "quantum.release",
            vec![(state, Some(region(1, &[])))],
            Vec::new(),
            vec![
                attribute("origin", integer(0)),
                attribute("proof", string("modular-adjoint")),
            ],
        )
        .expect("valid release schema");
        assert_eq!(released.proof_obligation_count(), 1);
    }

    #[test]
    fn global_phase_pauli_evolution_and_their_adjoint_are_exact() {
        let state = qbits(2);
        let phase_operands = vec![
            (state.clone(), Some(region(1, &[]))),
            (staged("Param", scalar("Real")), None),
        ];
        assert!(
            single_quantum_operation(
                "quantum.global_phase",
                phase_operands.clone(),
                vec![(state.clone(), Some(region(1, &[])))],
                Vec::new(),
            )
            .is_ok()
        );
        assert!(
            single_quantum_operation(
                "qcore.adjoint",
                phase_operands,
                vec![(state.clone(), Some(region(1, &[])))],
                vec![
                    attribute("original", string("quantum.global_phase")),
                    attribute("original_attributes", mapping(Vec::new())),
                    attribute("proof", string("modular-adjoint")),
                ],
            )
            .is_ok()
        );

        let evolution_operands = vec![
            (state.clone(), Some(region(1, &[]))),
            (staged("Const", scalar("Observable")), None),
            (scalar("Int"), None),
        ];
        assert!(
            single_quantum_operation(
                "quantum.pauli_evolution",
                evolution_operands.clone(),
                vec![(state.clone(), Some(region(1, &[])))],
                Vec::new(),
            )
            .is_ok()
        );
        assert!(
            single_quantum_operation(
                "qcore.adjoint",
                evolution_operands,
                vec![(state, Some(region(1, &[])))],
                vec![
                    attribute("original", string("quantum.pauli_evolution")),
                    attribute("original_attributes", mapping(Vec::new())),
                    attribute("proof", string("modular-adjoint")),
                ],
            )
            .is_ok()
        );
    }

    #[test]
    fn forget_and_unprepare_are_retained_as_unverified_obligations() {
        let forgotten = single_quantum_operation(
            "qcore.forget",
            vec![(qbits(1), Some(region(1, &[])))],
            Vec::new(),
            Vec::new(),
        )
        .expect("well-typed forget obligation");
        assert_eq!(forgotten.proof_obligation_count(), 1);

        let diagnostic = single_quantum_operation(
            "qcore.forget",
            vec![(qbits(1), Some(region(1, &[])))],
            Vec::new(),
            vec![attribute("reason", string("end of lexical lifetime"))],
        )
        .expect("diagnostic forget obligation");
        assert_eq!(diagnostic.proof_obligation_count(), 1);

        let unprepared = single_quantum_operation(
            "quantum.unprepare",
            vec![
                (qbits(1), Some(region(1, &[]))),
                (staged("Const", bits(1)), None),
            ],
            Vec::new(),
            vec![
                attribute("origin", integer(7)),
                attribute("proof", string("modular-adjoint")),
            ],
        )
        .expect("well-typed unprepare obligation");
        assert_eq!(unprepared.proof_obligation_count(), 1);
    }
}
