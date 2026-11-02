//! Typed substitution of semantic operations by verified `QCore` callables.
//!
//! The source program remains the semantic authority.  This module constructs
//! a transient, fully reverified image for resource projection or target
//! emission.  User definitions cross the boundary as canonical `QCore` modules;
//! no Python callback or process-global registry is retained.

use std::collections::{BTreeMap, BTreeSet};

use thiserror::Error;

use crate::effects::{CallableKind, Capability};
use crate::wire;
use crate::{CompilerError, VerifiedModule, VerifiedProgram, seal_package};

/// Largest explicit user realization set accepted by one projection.
pub const MAX_OPERATION_REALIZATIONS: usize = 32;

/// One semantic operation that admits a user-authored callable realization.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum SemanticOperation {
    /// Equal-width wrapping register addition.
    AddInto,
    /// Total constant modular multiplication.
    ModMul,
    /// Reversible modular multiply-accumulate over three equal-width words.
    ModMulAdd,
}

/// One verified semantic site replaced in a transient image.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum RealizationTarget {
    /// Replace every occurrence of one closed semantic operation.
    Operation(SemanticOperation),
    /// Replace calls to one verified body-backed callable.
    Callable(String),
}

impl RealizationTarget {
    fn label(&self) -> &str {
        match self {
            Self::Operation(operation) => operation.opcode(),
            Self::Callable(symbol) => symbol,
        }
    }
}

impl SemanticOperation {
    /// Decode the public intrinsic name at the untyped boundary.
    ///
    /// # Errors
    ///
    /// Returns [`OperationRealizationError::UnknownOperation`] for names that
    /// are not part of the supported realization domain.
    pub fn parse(value: &str) -> Result<Self, OperationRealizationError> {
        match value {
            "add_into" | "quantum.add_into" => Ok(Self::AddInto),
            "mod_mul" | "quantum.mod_mul" => Ok(Self::ModMul),
            "mod_mul_add" | "quantum.mod_mul_add" => Ok(Self::ModMulAdd),
            _ => Err(OperationRealizationError::UnknownOperation(
                value.to_owned(),
            )),
        }
    }

    /// Return the canonical semantic opcode replaced by this operation.
    #[must_use]
    pub const fn opcode(self) -> &'static str {
        match self {
            Self::AddInto => "quantum.add_into",
            Self::ModMul => "quantum.mod_mul",
            Self::ModMulAdd => "quantum.mod_mul_add",
        }
    }
}

/// One canonical user definition before it is linked into a source image.
#[derive(Clone, Debug)]
pub struct OperationRealizationDefinition {
    target: RealizationTarget,
    module: VerifiedModule,
    symbol: String,
}

impl OperationRealizationDefinition {
    /// Decode and resolve one user-authored `QCore` module.
    ///
    /// # Errors
    ///
    /// Returns a typed diagnostic when the operation name, module, callable,
    /// or callable contract is invalid.
    pub fn operation_from_bytes(
        operation: &str,
        module: &[u8],
        symbol: &str,
    ) -> Result<Self, OperationRealizationError> {
        let operation = SemanticOperation::parse(operation)?;
        let module = VerifiedModule::from_bytes(module)?;
        let resolved = module.resolve_entrypoint(symbol)?.to_owned();
        let contract = module
            .callable_contract(&resolved)
            .ok_or_else(|| OperationRealizationError::MissingCallable(resolved.clone()))?;
        if contract.kind() != CallableKind::Unitary
            || !contract.capabilities().contains(Capability::Adjointable)
            || !contract.capabilities().contains(Capability::Controllable)
        {
            return Err(OperationRealizationError::InvalidCallableContract(resolved));
        }
        Ok(Self {
            target: RealizationTarget::Operation(operation),
            module,
            symbol: resolved,
        })
    }

    /// Decode one verified body-backed callable replacement.
    ///
    /// # Panics
    ///
    /// Panics if a verified module resolves a callable without a local declaration,
    /// violating the internal verification invariant.
    ///
    /// # Errors
    ///
    /// Returns a typed diagnostic when either module, callable, contract, or
    /// replacement definition is invalid.
    pub fn callable_from_bytes(
        target_module: &[u8],
        target_symbol: &str,
        implementation_module: &[u8],
        implementation_symbol: &str,
    ) -> Result<Self, OperationRealizationError> {
        let target_module = VerifiedModule::from_bytes(target_module)?;
        let target_symbol = target_module.resolve_entrypoint(target_symbol)?.to_owned();
        let target_function = target_module
            .local
            .checked
            .package
            .functions
            .iter()
            .find(|function| function.symbol == target_symbol)
            .ok_or_else(|| OperationRealizationError::MissingCallable(target_symbol.clone()))?;
        if target_function.external || target_function.body.is_none() {
            return Err(OperationRealizationError::BodylessCallable(target_symbol));
        }
        let target_contract = target_module
            .callable_contract(&target_symbol)
            .ok_or_else(|| OperationRealizationError::MissingCallable(target_symbol.clone()))?;

        let module = VerifiedModule::from_bytes(implementation_module)?;
        let symbol = module.resolve_entrypoint(implementation_symbol)?.to_owned();
        let contract = module
            .callable_contract(&symbol)
            .ok_or_else(|| OperationRealizationError::MissingCallable(symbol.clone()))?;
        if contract.kind() != target_contract.kind()
            || contract.capabilities() != target_contract.capabilities()
            || !same_callable_abi(
                target_function,
                module
                    .local
                    .checked
                    .package
                    .functions
                    .iter()
                    .find(|function| function.symbol == symbol)
                    .expect("resolved callable must have a local declaration"),
            )
        {
            return Err(OperationRealizationError::CallableContractMismatch {
                target: target_symbol,
                implementation: symbol,
            });
        }
        Ok(Self {
            target: RealizationTarget::Callable(target_symbol),
            module,
            symbol,
        })
    }
}

/// Failure to construct a verified transient realization image.
#[derive(Clone, Debug, Error)]
pub enum OperationRealizationError {
    /// The requested semantic operation is not in the supported closed set.
    #[error("Q8060: unknown semantic operation realization {0:?}")]
    UnknownOperation(String),
    /// More than one definition targets the same semantic operation.
    #[error("Q8061: duplicate realization for semantic operation {0:?}")]
    DuplicateOperation(String),
    /// Too many explicit definitions were supplied.
    #[error(
        "Q8062: operation realization count exceeds the {MAX_OPERATION_REALIZATIONS}-entry budget"
    )]
    DefinitionBudget,
    /// The selected callable is absent after canonical module resolution.
    #[error("Q8063: operation realization callable {0:?} is missing")]
    MissingCallable(String),
    /// The selected callable cannot preserve exact coherent transforms.
    #[error("Q8064: operation realization callable {0:?} must be a controllable exact unitary")]
    InvalidCallableContract(String),
    /// A linked symbol conflicts with a different source declaration.
    #[error("Q8065: operation realization declaration conflicts at {0:?}")]
    ConflictingDeclaration(String),
    /// No matching operation exists, so its ABI cannot be checked at a site.
    #[error("Q8066: operation realization for {0:?} has no application site")]
    UnusedOperation(String),
    /// Callable targets must carry a verified source body.
    #[error("Q8067: callable realization target {0:?} must be body-backed")]
    BodylessCallable(String),
    /// Replacement kind and transform capabilities differ from the target.
    #[error(
        "Q8068: callable realization {implementation:?} does not preserve the contract of {target:?}"
    )]
    CallableContractMismatch {
        /// Original semantic callable.
        target: String,
        /// Candidate implementation callable.
        implementation: String,
    },
    /// Source linking rebased one target into more than one matching symbol.
    #[error("Q8069: callable realization target {0:?} is ambiguous after source linking")]
    AmbiguousCallableTarget(String),
    /// A semantic operation replacement expands into its own replacement cycle.
    #[error("Q8070: semantic operation replacement creates a recursive expansion at {0:?}")]
    RecursiveOperationReplacement(String),
    /// Complete `QCore` verification rejected the linked substitution.
    #[error("{0}")]
    Compiler(#[from] CompilerError),
}

/// Substitute user definitions into one verified closed program.
///
/// # Errors
///
/// Returns [`OperationRealizationError`] when definitions conflict, no
/// application site exists, or the rewritten image fails complete verification.
pub fn realize_program(
    program: &VerifiedProgram,
    definitions: Vec<OperationRealizationDefinition>,
) -> Result<VerifiedProgram, OperationRealizationError> {
    let package = realize_package(program.local.checked.package.clone(), definitions)?;
    let payload = seal_package(package)?;
    Ok(VerifiedProgram::from_bytes(&payload)?)
}

/// Substitute user definitions into one verified open source module.
///
/// # Errors
///
/// Returns [`OperationRealizationError`] when definitions conflict, no
/// application site exists, or the rewritten module fails verification.
pub fn realize_module(
    module: &VerifiedModule,
    definitions: Vec<OperationRealizationDefinition>,
) -> Result<VerifiedModule, OperationRealizationError> {
    let package = realize_package(module.local.checked.package.clone(), definitions)?;
    let payload = seal_package(package)?;
    Ok(VerifiedModule::from_bytes(&payload)?)
}

fn realize_package(
    mut package: wire::QCorePackageProto,
    definitions: Vec<OperationRealizationDefinition>,
) -> Result<wire::QCorePackageProto, OperationRealizationError> {
    if definitions.len() > MAX_OPERATION_REALIZATIONS {
        return Err(OperationRealizationError::DefinitionBudget);
    }
    let mut selected = BTreeMap::new();
    for definition in definitions {
        let target = definition.target.clone();
        if selected.insert(target.clone(), definition).is_some() {
            return Err(OperationRealizationError::DuplicateOperation(
                target.label().to_owned(),
            ));
        }
    }

    let mut functions = package
        .functions
        .iter()
        .map(|function| (function.symbol.clone(), function.clone()))
        .collect::<BTreeMap<_, _>>();
    let mut constants = package
        .constants
        .iter()
        .map(|constant| (constant.name.clone(), constant.clone()))
        .collect::<BTreeMap<_, _>>();
    for definition in selected.values() {
        for function in &definition.module.local.checked.package.functions {
            match functions.get(&function.symbol) {
                Some(existing) if existing != function => {
                    return Err(OperationRealizationError::ConflictingDeclaration(
                        function.symbol.clone(),
                    ));
                }
                Some(_) => {}
                None => {
                    functions.insert(function.symbol.clone(), function.clone());
                }
            }
        }
        for constant in &definition.module.local.checked.package.constants {
            match constants.get(&constant.name) {
                Some(existing) if existing != constant => {
                    return Err(OperationRealizationError::ConflictingDeclaration(
                        constant.name.clone(),
                    ));
                }
                Some(_) => {}
                None => {
                    constants.insert(constant.name.clone(), constant.clone());
                }
            }
        }
    }
    package.functions = functions.into_values().collect();
    package.constants = constants.into_values().collect();

    let mut replaced = BTreeSet::new();
    for definition in selected.values() {
        let RealizationTarget::Callable(target_symbol) = &definition.target else {
            continue;
        };
        let implementation = package
            .functions
            .iter()
            .find(|function| function.symbol == definition.symbol)
            .cloned()
            .ok_or_else(|| OperationRealizationError::MissingCallable(definition.symbol.clone()))?;
        let target_index = resolve_callable_target(&package.functions, target_symbol)?;
        let target = &mut package.functions[target_index];
        target.parameters = target
            .parameters
            .iter()
            .cloned()
            .zip(implementation.parameters)
            .map(|(mut target, implementation)| {
                target.value = implementation.value;
                target
            })
            .collect();
        target.body = implementation.body;
        replaced.insert(definition.target.clone());
    }
    let mut generated_edges = BTreeSet::new();
    for (owner, function) in package.functions.iter_mut().enumerate() {
        let Some(body) = function.body.as_mut() else {
            continue;
        };
        rewrite_block(body, &selected, &mut replaced, owner, &mut generated_edges);
    }
    for target in selected.keys() {
        if !replaced.contains(target) {
            return Err(OperationRealizationError::UnusedOperation(
                target.label().to_owned(),
            ));
        }
    }
    reject_recursive_operation_replacements(&package, &generated_edges)?;
    Ok(package)
}

fn resolve_callable_target(
    functions: &[wire::FunctionProto],
    target_symbol: &str,
) -> Result<usize, OperationRealizationError> {
    if let Some(index) = functions
        .iter()
        .position(|function| function.symbol == target_symbol)
    {
        return Ok(index);
    }
    let local_name = target_symbol.rsplit('.').next().unwrap_or(target_symbol);
    let matches = functions
        .iter()
        .enumerate()
        .filter(|(_, function)| {
            function
                .symbol
                .rsplit('.')
                .next()
                .unwrap_or(&function.symbol)
                == local_name
        })
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [index] => Ok(*index),
        [] => Err(OperationRealizationError::UnusedOperation(
            target_symbol.to_owned(),
        )),
        _ => Err(OperationRealizationError::AmbiguousCallableTarget(
            target_symbol.to_owned(),
        )),
    }
}

fn same_callable_abi(target: &wire::FunctionProto, implementation: &wire::FunctionProto) -> bool {
    target.parameters.len() == implementation.parameters.len()
        && target
            .parameters
            .iter()
            .zip(&implementation.parameters)
            .all(|(target, implementation)| {
                target
                    .value
                    .as_ref()
                    .and_then(|value| value.r#type.as_ref())
                    == implementation
                        .value
                        .as_ref()
                        .and_then(|value| value.r#type.as_ref())
                    && target.has_default == implementation.has_default
                    && target.default_value == implementation.default_value
            })
        && target.result_type == implementation.result_type
}

fn rewrite_block(
    block: &mut wire::BlockProto,
    selected: &BTreeMap<RealizationTarget, OperationRealizationDefinition>,
    replaced: &mut BTreeSet<RealizationTarget>,
    owner: usize,
    generated_edges: &mut BTreeSet<(usize, String)>,
) {
    for operation in &mut block.operations {
        for region in &mut operation.regions {
            for nested in &mut region.blocks {
                rewrite_block(nested, selected, replaced, owner, generated_edges);
            }
        }
        let target = SemanticOperation::parse(&operation.opcode)
            .ok()
            .map(RealizationTarget::Operation);
        let Some(target) = target else {
            continue;
        };
        let Some(definition) = selected.get(&target) else {
            continue;
        };
        replaced.insert(target);
        generated_edges.insert((owner, definition.symbol.clone()));
        "qcore.call".clone_into(&mut operation.opcode);
        operation.attributes = vec![wire::AttributeProto {
            name: "callee".to_owned(),
            value: Some(wire::LiteralProto {
                value: Some(wire::literal_proto::Value::StringValue(
                    definition.symbol.clone(),
                )),
            }),
        }];
    }
}

/// Reject cycles introduced by operation substitution, independently of language
/// recursion already present inside ordinary implementation calls.
fn reject_recursive_operation_replacements(
    package: &wire::QCorePackageProto,
    generated_edges: &BTreeSet<(usize, String)>,
) -> Result<(), OperationRealizationError> {
    if generated_edges.is_empty() {
        return Ok(());
    }
    let graph = crate::call_graph::build(package)
        .map_err(|error| CompilerError::InvalidCallGraph(error.to_string()))?;
    let mut components = vec![None; package.functions.len()];
    for (identifier, component) in graph
        .components()
        .iter()
        .enumerate()
        .filter(|(_, component)| component.is_recursive())
    {
        for member in component.members() {
            components[*member] = Some(identifier);
        }
    }
    let indices = package
        .functions
        .iter()
        .enumerate()
        .map(|(index, function)| (function.symbol.as_str(), index))
        .collect::<BTreeMap<_, _>>();
    for (caller, callee) in generated_edges {
        if let Some(component) = components[*caller]
            && indices
                .get(callee.as_str())
                .is_some_and(|callee| components[*callee] == Some(component))
        {
            return Err(OperationRealizationError::RecursiveOperationReplacement(
                callee.clone(),
            ));
        }
    }
    Ok(())
}
