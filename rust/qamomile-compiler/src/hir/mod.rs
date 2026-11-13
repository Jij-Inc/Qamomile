//! Private typed-HIR boundary between source syntax and ownership SSA.

mod finite_dependency_sets;
mod function_results;

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fmt::Write as _,
    sync::Arc,
};

use num_bigint::BigInt;
use prost::Message;
use sha2::{Digest, Sha256};

use crate::{
    MAX_QCORE_BYTES, MODULE_KIND, SCHEMA_MAJOR, SCHEMA_MINOR, VerifiedModule, budget, literals,
    seal_package,
    static_values::normalize_aggregate_slice,
    type_relations::{bind_function_application, unify_assignable},
    types::{Dimension, SemanticType, Stage, TypeArena},
    wire,
};
use finite_dependency_sets::{FiniteDependencyFacts, solve_finite_dependency_sets};

const HIR_KIND: &str = "qamomile.hir";
const INTERNAL_NAME_PREFIX: &str = "\0";

type RegionKey = (u64, Vec<String>);
type RegionChildren = HashMap<RegionKey, HashSet<String>>;

#[derive(Clone, Debug, Eq, PartialEq)]
enum AggregateExtent {
    Fixed(i64),
    Symbol(String),
}

enum IterationBinding {
    Range,
    Iterate,
    MapItems,
    Enumerate { start_hidden_index: usize },
    Zip,
}

/// Structural facts reconstructed from one canonical typed-HIR request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HirHeader {
    /// Stable semantic module name.
    pub module_name: String,
    /// Number of source function declarations.
    pub function_count: usize,
    /// Number of source constants.
    pub constant_count: usize,
    /// Number of typed expressions across all function bodies.
    pub expression_count: usize,
    /// Number of typed statements across all function bodies.
    pub statement_count: usize,
}

/// Diagnostic work charged while independently validating one typed-HIR image.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct HirValidationWork {
    input_bytes: usize,
    preflight_message_nodes: usize,
    semantic_node_visits: u128,
    expression_count: usize,
    statement_count: usize,
}

impl HirValidationWork {
    /// Return the number of borrowed wire bytes supplied to preflight.
    #[must_use]
    pub const fn input_bytes(self) -> usize {
        self.input_bytes
    }

    /// Return the number of wire values charged before Protobuf allocation.
    #[must_use]
    pub const fn preflight_message_nodes(self) -> usize {
        self.preflight_message_nodes
    }

    /// Return the number of semantic validation-node visits.
    ///
    /// This includes declaration, parameter, statement, expression, attribute,
    /// literal-root, and type-resolution visits. It is compiler work rather
    /// than a source or circuit resource metric.
    #[must_use]
    pub const fn semantic_node_visits(self) -> u128 {
        self.semantic_node_visits
    }

    fn charge_semantic_nodes(&mut self, count: usize) {
        self.semantic_node_visits = self.semantic_node_visits.saturating_add(count as u128);
    }
}

/// Diagnostic work charged by final typed-HIR to ownership-SSA lowering.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct HirSsaLoweringWork {
    functions: u128,
    statements: u128,
    expressions: u128,
}

impl HirSsaLoweringWork {
    /// Return the number of normalized HIR functions lowered.
    #[must_use]
    pub const fn function_visits(self) -> u128 {
        self.functions
    }

    /// Return the number of normalized HIR statements lowered.
    #[must_use]
    pub const fn statement_visits(self) -> u128 {
        self.statements
    }

    /// Return the number of normalized HIR expressions lowered.
    #[must_use]
    pub const fn expression_visits(self) -> u128 {
        self.expressions
    }

    /// Return all final ownership-SSA source-node visits.
    #[must_use]
    pub const fn source_node_visits(self) -> u128 {
        self.functions
            .saturating_add(self.statements)
            .saturating_add(self.expressions)
    }

    fn visit_function(&mut self) {
        self.functions = self.functions.saturating_add(1);
    }

    fn visit_statement(&mut self) {
        self.statements = self.statements.saturating_add(1);
    }

    fn visit_expression(&mut self) {
        self.expressions = self.expressions.saturating_add(1);
    }
}

/// Diagnostic work charged by pre-SSA typed-HIR normalization passes.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct HirNormalizationWork {
    function_results: u128,
    return_continuations: u128,
    higher_order: u128,
    function_aliases: u128,
}

impl HirNormalizationWork {
    /// Return visits charged by function-result analysis and rewriting.
    #[must_use]
    pub const fn function_result_visits(self) -> u128 {
        self.function_results
    }

    /// Return visits charged by return-continuation normalization.
    #[must_use]
    pub const fn return_continuation_visits(self) -> u128 {
        self.return_continuations
    }

    /// Return visits charged by higher-order specialization rewriting.
    #[must_use]
    pub const fn higher_order_visits(self) -> u128 {
        self.higher_order
    }

    /// Return visits charged by function-alias analysis and rewriting.
    #[must_use]
    pub const fn function_alias_visits(self) -> u128 {
        self.function_aliases
    }

    /// Return all pre-SSA normalization-node visits.
    #[must_use]
    pub const fn source_node_visits(self) -> u128 {
        self.function_results
            .saturating_add(self.return_continuations)
            .saturating_add(self.higher_order)
            .saturating_add(self.function_aliases)
    }

    pub(super) fn visit_function_result(&mut self) {
        self.function_results = self.function_results.saturating_add(1);
    }

    fn visit_return_continuation(&mut self) {
        self.return_continuations = self.return_continuations.saturating_add(1);
    }

    fn visit_higher_order(&mut self) {
        self.higher_order = self.higher_order.saturating_add(1);
    }

    fn charge_higher_order(&mut self, count: usize) {
        self.higher_order = self.higher_order.saturating_add(count as u128);
    }

    pub(super) fn visit_function_alias(&mut self) {
        self.function_aliases = self.function_aliases.saturating_add(1);
    }

    fn charge_function_aliases(&mut self, count: usize) {
        self.function_aliases = self.function_aliases.saturating_add(count as u128);
    }
}

/// Failure while decoding or independently validating private typed HIR.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum HirError {
    /// The encoded request exceeds the shared package-size budget.
    #[error("Q7001: typed HIR request exceeds the wire-size limit")]
    PackageTooLarge,
    /// Raw Protobuf structure exceeds a decode budget.
    #[error("Q7002: typed HIR request exceeds a structural budget: {0}")]
    PackageBudgetExceeded(String),
    /// Protobuf bytes cannot be decoded safely.
    #[error("Q7003: malformed typed HIR request: {0}")]
    MalformedPackage(String),
    /// The request uses an incompatible schema major.
    #[error("Q7004: unsupported typed HIR schema major {actual}; expected {expected}")]
    UnsupportedSchemaMajor {
        /// Observed schema major.
        actual: u32,
        /// Supported schema major.
        expected: u32,
    },
    /// The request uses a newer schema minor.
    #[error("Q7005: unsupported typed HIR schema minor {actual}; maximum {maximum}")]
    UnsupportedSchemaMinor {
        /// Observed schema minor.
        actual: u32,
        /// Maximum supported schema minor.
        maximum: u32,
    },
    /// The request is not the private HIR artifact kind.
    #[error("Q7006: expected typed HIR artifact {expected:?}, got {actual:?}")]
    UnexpectedArtifactKind {
        /// Required internal artifact kind.
        expected: &'static str,
        /// Observed artifact kind.
        actual: String,
    },
    /// The integrity digest is absent or incorrect.
    #[error("Q7007: typed HIR request digest mismatch")]
    DigestMismatch,
    /// The request has multiple byte encodings for the same meaning.
    #[error("Q7008: typed HIR request is not canonically encoded")]
    NonCanonicalEncoding,
    /// A declaration or tree node violates the HIR grammar.
    #[error("Q7009: invalid typed HIR structure: {0}")]
    InvalidStructure(String),
    /// A type lies outside the closed semantic universe.
    #[error("Q7010: invalid typed HIR semantic type: {0}")]
    InvalidSemanticType(String),
    /// A portable literal is absent or noncanonical.
    #[error("Q7011: invalid typed HIR portable literal: {0}")]
    InvalidPortableLiteral(String),
}

/// Failure while lowering the first native typed-HIR subset to `QCore`.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum HirLoweringError {
    /// The typed-HIR request itself is invalid.
    #[error(transparent)]
    InvalidHir(#[from] HirError),
    /// The valid request lies outside the currently migrated lowering subset.
    #[error("Q7101: native HIR lowering does not yet support {0}")]
    Unsupported(String),
    /// Native output failed the ordinary `QCore` module boundary.
    #[error("Q7102: native HIR lowering produced invalid QCore: {0}")]
    InvalidQCore(String),
}

/// Decode and independently validate one canonical typed-HIR request.
pub(crate) fn inspect(payload: &[u8]) -> Result<HirHeader, HirError> {
    inspect_with_work(payload).map(|(header, _)| header)
}

pub(crate) fn inspect_with_work(
    payload: &[u8],
) -> Result<(HirHeader, HirValidationWork), HirError> {
    decode_with_work(payload).map(|(_, header, work)| (header, work))
}

/// Seal one canonical unsealed typed-HIR request through the native wire policy.
pub(crate) fn seal(payload: &[u8]) -> Result<Vec<u8>, HirError> {
    let mut package = decode_message(payload)?;
    validate_header(&package)?;
    if !package.package_digest.is_empty() || package.encode_to_vec() != payload {
        return Err(HirError::NonCanonicalEncoding);
    }
    validate(&package, &mut HirValidationWork::default())?;
    let digest: [u8; 32] = Sha256::digest(package.encode_to_vec()).into();
    package.package_digest.extend_from_slice(&digest);
    Ok(package.encode_to_vec())
}

/// Lower one validated typed-HIR request into verified `QCore`.
pub(crate) fn lower(payload: &[u8]) -> Result<Vec<u8>, HirLoweringError> {
    lower_with_all_work(payload).map(|(module, _, _)| module)
}

pub(crate) fn lower_with_work(
    payload: &[u8],
) -> Result<(Vec<u8>, HirSsaLoweringWork), HirLoweringError> {
    lower_with_all_work(payload).map(|(module, _, ssa)| (module, ssa))
}

pub(crate) fn lower_with_normalization_work(
    payload: &[u8],
) -> Result<(Vec<u8>, HirNormalizationWork), HirLoweringError> {
    lower_with_all_work(payload).map(|(module, normalization, _)| (module, normalization))
}

fn lower_with_all_work(
    payload: &[u8],
) -> Result<(Vec<u8>, HirNormalizationWork, HirSsaLoweringWork), HirLoweringError> {
    let (package, _) = decode(payload)?;
    let mut normalization = HirNormalizationWork::default();
    let (qcore, ssa) = lower_package_with_work(&package, &mut normalization)?;
    let encoded =
        seal_package(qcore).map_err(|error| HirLoweringError::InvalidQCore(error.to_string()))?;
    let verified = VerifiedModule::from_bytes(&encoded)
        .map_err(|error| HirLoweringError::InvalidQCore(error.to_string()))?;
    Ok((verified.canonical_bytes().to_vec(), normalization, ssa))
}

fn decode(payload: &[u8]) -> Result<(wire::HirPackageProto, HirHeader), HirError> {
    decode_with_work(payload).map(|(package, header, _)| (package, header))
}

fn decode_with_work(
    payload: &[u8],
) -> Result<(wire::HirPackageProto, HirHeader, HirValidationWork), HirError> {
    let (mut package, mut work) = decode_message_with_work(payload)?;
    validate_header(&package)?;
    let observed_digest: [u8; 32] = package
        .package_digest
        .as_slice()
        .try_into()
        .map_err(|_| HirError::DigestMismatch)?;
    package.package_digest.clear();
    let expected_digest: [u8; 32] = Sha256::digest(package.encode_to_vec()).into();
    if observed_digest != expected_digest {
        return Err(HirError::DigestMismatch);
    }
    package.package_digest.extend_from_slice(&observed_digest);
    if package.encode_to_vec() != payload {
        return Err(HirError::NonCanonicalEncoding);
    }
    let header = validate(&package, &mut work)?;
    Ok((package, header, work))
}

fn decode_message(payload: &[u8]) -> Result<wire::HirPackageProto, HirError> {
    decode_message_with_work(payload).map(|(package, _)| package)
}

fn decode_message_with_work(
    payload: &[u8],
) -> Result<(wire::HirPackageProto, HirValidationWork), HirError> {
    if payload.len() > MAX_QCORE_BYTES {
        return Err(HirError::PackageTooLarge);
    }
    let preflight_message_nodes =
        budget::preflight_hir_with_work(payload).map_err(|error| match error {
            budget::PreflightError::Budget(reason) => HirError::PackageBudgetExceeded(reason),
            budget::PreflightError::Malformed(reason) => HirError::MalformedPackage(reason),
        })?;
    let package = wire::HirPackageProto::decode(payload)
        .map_err(|error| HirError::MalformedPackage(error.to_string()))?;
    Ok((
        package,
        HirValidationWork {
            input_bytes: payload.len(),
            preflight_message_nodes,
            ..Default::default()
        },
    ))
}

fn validate_header(package: &wire::HirPackageProto) -> Result<(), HirError> {
    if package.schema_major != SCHEMA_MAJOR {
        return Err(HirError::UnsupportedSchemaMajor {
            actual: package.schema_major,
            expected: SCHEMA_MAJOR,
        });
    }
    if package.schema_minor > SCHEMA_MINOR {
        return Err(HirError::UnsupportedSchemaMinor {
            actual: package.schema_minor,
            maximum: SCHEMA_MINOR,
        });
    }
    if package.artifact_kind != HIR_KIND {
        return Err(HirError::UnexpectedArtifactKind {
            expected: HIR_KIND,
            actual: package.artifact_kind.clone(),
        });
    }
    Ok(())
}

fn lower_package_with_work(
    package: &wire::HirPackageProto,
    normalization_work: &mut HirNormalizationWork,
) -> Result<(wire::QCorePackageProto, HirSsaLoweringWork), HirLoweringError> {
    let mut lowered_functions = package.functions.clone();
    function_results::lower(&mut lowered_functions, normalization_work)?;
    let mut normalized_package = package.clone();
    normalized_package.functions = lowered_functions;
    let normalized = monomorphize_higher_order_functions(&normalized_package, normalization_work)?;
    let mut work = HirSsaLoweringWork::default();
    let mut functions = Vec::with_capacity(normalized.len());
    for function in &normalized {
        functions.push(lower_function(function, &mut work)?);
    }
    Ok((
        wire::QCorePackageProto {
            schema_major: SCHEMA_MAJOR,
            schema_minor: SCHEMA_MINOR,
            artifact_kind: MODULE_KIND.to_owned(),
            module_name: package.module_name.clone(),
            functions,
            constants: package
                .constants
                .iter()
                .map(|constant| wire::ConstantProto {
                    name: constant.name.clone(),
                    r#type: constant.r#type.clone(),
                    value: constant.value.clone(),
                })
                .collect(),
            ..Default::default()
        },
        work,
    ))
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct HigherOrderSpecializationKey {
    template: String,
    candidates: Vec<String>,
    argument_types: Vec<Vec<u8>>,
}

type HigherOrderBindings = (TypeArena, Arc<str>, HashMap<Dimension, Dimension>);

const FUNCTION_YIELD_ALIAS: &str = "\0function.yield";

struct HigherOrderMonomorphizer {
    module_name: String,
    templates: HashMap<String, wire::HirFunctionProto>,
    parameter_indices: HashMap<String, Vec<usize>>,
    source_symbols: HashSet<String>,
    generated: HashMap<HigherOrderSpecializationKey, wire::HirFunctionProto>,
    active: HashMap<HigherOrderSpecializationKey, wire::HirExpressionProto>,
    used_templates: HashSet<String>,
    expanded_nodes: usize,
}

fn monomorphize_higher_order_functions(
    package: &wire::HirPackageProto,
    work: &mut HirNormalizationWork,
) -> Result<Vec<wire::HirFunctionProto>, HirLoweringError> {
    let mut templates = HashMap::new();
    let mut parameter_indices = HashMap::new();
    let mut ordinary = Vec::new();
    for source in &package.functions {
        work.visit_higher_order();
        let indices = higher_order_parameter_indices(source)?;
        let mut function = source.clone();
        if !function.external {
            function.body = Some(normalize_return_continuations(
                function.body.as_ref().ok_or_else(|| {
                    HirLoweringError::Unsupported("function without a body".to_owned())
                })?,
                work,
            )?);
        }
        if indices.is_empty() {
            ordinary.push(function);
        } else {
            if function.external {
                return Err(HirLoweringError::Unsupported(format!(
                    "external function {:?} with function-valued parameters",
                    function.symbol
                )));
            }
            parameter_indices.insert(function.symbol.clone(), indices);
            templates.insert(function.symbol.clone(), function);
        }
    }
    for function in &mut ordinary {
        work.visit_higher_order();
        if !function.external {
            let mut arena = TypeArena::default();
            let scope = arena.intern_scope(&function.symbol);
            function.body = Some(eliminate_local_function_aliases(
                function.body.as_ref().expect("normalized function body"),
                &scope,
                &mut arena,
                work,
            )?);
        }
    }
    if templates.is_empty() {
        return Ok(ordinary);
    }

    let source_symbols = package
        .functions
        .iter()
        .map(|function| {
            work.visit_higher_order();
            function.symbol.clone()
        })
        .collect();
    let mut monomorphizer = HigherOrderMonomorphizer {
        module_name: package.module_name.clone(),
        templates,
        parameter_indices,
        source_symbols,
        generated: HashMap::new(),
        active: HashMap::new(),
        used_templates: HashSet::new(),
        expanded_nodes: 0,
    };
    for function in &mut ordinary {
        work.visit_higher_order();
        if !function.external {
            monomorphizer.rewrite_block(
                function.body.as_mut().expect("normalized function body"),
                0,
                work,
            )?;
            reject_template_function_references(
                function.body.as_ref().expect("normalized function body"),
                &monomorphizer.templates,
                work,
            )?;
        }
    }
    let mut unused = monomorphizer
        .templates
        .keys()
        .filter(|symbol| !monomorphizer.used_templates.contains(*symbol))
        .cloned()
        .collect::<Vec<_>>();
    if !unused.is_empty() {
        unused.sort();
        return Err(HirLoweringError::Unsupported(format!(
            "function-valued parameters without a closed call site in {:?}",
            unused[0]
        )));
    }
    ordinary.extend(monomorphizer.generated.into_values());
    ordinary.sort_by(|left, right| left.symbol.cmp(&right.symbol));
    Ok(ordinary)
}

fn higher_order_parameter_indices(
    function: &wire::HirFunctionProto,
) -> Result<Vec<usize>, HirLoweringError> {
    let mut arena = TypeArena::default();
    let scope = arena.intern_scope(&function.symbol);
    let mut indices = Vec::new();
    for (index, parameter) in function.parameters.iter().enumerate() {
        let parameter_type = parameter.r#type.as_ref().ok_or_else(|| {
            HirLoweringError::InvalidQCore(format!(
                "function parameter {:?} has no type",
                parameter.name
            ))
        })?;
        if wire_type_is_function(parameter_type, &scope, &mut arena)? {
            indices.push(index);
        }
    }
    Ok(indices)
}

impl HigherOrderMonomorphizer {
    fn rewrite_block(
        &mut self,
        block: &mut wire::HirBlockProto,
        depth: usize,
        work: &mut HirNormalizationWork,
    ) -> Result<(), HirLoweringError> {
        if depth > budget::MAX_NESTING {
            return Err(HirLoweringError::Unsupported(
                "higher-order control-flow nesting".to_owned(),
            ));
        }
        for statement in &mut block.statements {
            work.visit_higher_order();
            if let Some(expression) = &mut statement.expression {
                self.rewrite_expression(expression, depth + 1, work)?;
            }
            for region in &mut statement.regions {
                self.rewrite_block(region, depth + 1, work)?;
            }
        }
        Ok(())
    }

    fn rewrite_expression(
        &mut self,
        expression: &mut wire::HirExpressionProto,
        depth: usize,
        work: &mut HirNormalizationWork,
    ) -> Result<(), HirLoweringError> {
        work.visit_higher_order();
        if depth > budget::MAX_NESTING {
            return Err(HirLoweringError::Unsupported(
                "higher-order expression nesting".to_owned(),
            ));
        }
        for operand in &mut expression.operands {
            self.rewrite_expression(operand, depth + 1, work)?;
        }
        for region in &mut expression.regions {
            self.rewrite_expression(region, depth + 1, work)?;
        }
        if expression.opcode != "call" {
            return Ok(());
        }
        let callee = string_attribute(&expression.attributes, "callee")?.to_owned();
        let Some(template) = self.templates.get(&callee).cloned() else {
            return Ok(());
        };
        let parameter_indices = self
            .parameter_indices
            .get(&callee)
            .cloned()
            .expect("template parameter indices");
        let combinations =
            higher_order_candidate_combinations(&parameter_indices, &expression.operands)?;
        if combinations > MAX_FUNCTION_VALUE_CANDIDATES {
            return Err(HirLoweringError::Unsupported(format!(
                "higher-order call exceeds {MAX_FUNCTION_VALUE_CANDIDATES} candidate combinations"
            )));
        }
        let callable = self.specialize_callable_product(
            &template,
            &parameter_indices,
            &expression.operands,
            0,
            work,
        )?;
        let evaluation_order = expression_evaluation_order(expression)?;
        let function_parameters = parameter_indices.iter().copied().collect::<HashSet<_>>();
        let mut retained_indices = vec![None; expression.operands.len()];
        let mut retained_count = 0_usize;
        for (index, slot) in retained_indices.iter_mut().enumerate() {
            if !function_parameters.contains(&index) {
                *slot = Some(retained_count);
                retained_count += 1;
            }
        }
        let projected_order = evaluation_order
            .into_iter()
            .filter_map(|index| retained_indices[index])
            .collect::<Vec<_>>();
        let arguments = expression
            .operands
            .drain(..)
            .enumerate()
            .filter_map(|(index, operand)| {
                (!function_parameters.contains(&index)).then_some(operand)
            })
            .collect::<Vec<_>>();
        if callable.opcode == "function_ref" {
            expression.operands = arguments;
            expression.attributes = vec![wire::AttributeProto {
                name: "callee".to_owned(),
                value: Some(wire::LiteralProto {
                    value: Some(wire::literal_proto::Value::StringValue(
                        string_attribute(&callable.attributes, "callee")?.to_owned(),
                    )),
                }),
            }];
            append_nonidentity_evaluation_order(&mut expression.attributes, &projected_order)?;
        } else {
            "invoke".clone_into(&mut expression.opcode);
            expression.operands = std::iter::once(callable).chain(arguments).collect();
            expression.attributes.clear();
            let invocation_order = std::iter::once(0)
                .chain(projected_order.iter().map(|index| index + 1))
                .collect::<Vec<_>>();
            append_nonidentity_evaluation_order(&mut expression.attributes, &invocation_order)?;
        }
        Ok(())
    }

    fn specialize_callable_product(
        &mut self,
        template: &wire::HirFunctionProto,
        function_parameters: &[usize],
        arguments: &[wire::HirExpressionProto],
        position: usize,
        work: &mut HirNormalizationWork,
    ) -> Result<wire::HirExpressionProto, HirLoweringError> {
        work.visit_higher_order();
        let Some(index) = function_parameters.get(position).copied() else {
            return self.specialize(template, function_parameters, arguments, work);
        };
        self.specialize_callable_choice(
            template,
            function_parameters,
            arguments,
            position,
            index,
            &arguments[index],
            work,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn specialize_callable_choice(
        &mut self,
        template: &wire::HirFunctionProto,
        function_parameters: &[usize],
        arguments: &[wire::HirExpressionProto],
        position: usize,
        argument_index: usize,
        choice: &wire::HirExpressionProto,
        work: &mut HirNormalizationWork,
    ) -> Result<wire::HirExpressionProto, HirLoweringError> {
        work.visit_higher_order();
        match choice.opcode.as_str() {
            "function_ref" => {
                let mut resolved = arguments.to_vec();
                resolved[argument_index] = choice.clone();
                self.specialize_callable_product(
                    template,
                    function_parameters,
                    &resolved,
                    position + 1,
                    work,
                )
            }
            "lazy_if" => {
                let [when_true, when_false] = choice.regions.as_slice() else {
                    return Err(HirLoweringError::InvalidQCore(
                        "higher-order callable choice does not have two arms".to_owned(),
                    ));
                };
                let when_true = self.specialize_callable_choice(
                    template,
                    function_parameters,
                    arguments,
                    position,
                    argument_index,
                    when_true,
                    work,
                )?;
                let when_false = self.specialize_callable_choice(
                    template,
                    function_parameters,
                    arguments,
                    position,
                    argument_index,
                    when_false,
                    work,
                )?;
                let true_type = when_true.r#type.as_ref().ok_or_else(|| {
                    HirLoweringError::InvalidQCore(
                        "specialized callable branch has no type".to_owned(),
                    )
                })?;
                let false_type = when_false.r#type.as_ref().ok_or_else(|| {
                    HirLoweringError::InvalidQCore(
                        "specialized callable branch has no type".to_owned(),
                    )
                })?;
                if function_base_type(true_type)? != function_base_type(false_type)? {
                    return Err(HirLoweringError::InvalidQCore(
                        "specialized callable branches disagree".to_owned(),
                    ));
                }
                Ok(wire::HirExpressionProto {
                    opcode: "lazy_if".to_owned(),
                    r#type: Some(function_type_at_choice_stage(
                        true_type,
                        choice.r#type.as_ref().ok_or_else(|| {
                            HirLoweringError::InvalidQCore(
                                "higher-order callable choice has no type".to_owned(),
                            )
                        })?,
                    )?),
                    operands: choice.operands.clone(),
                    attributes: choice.attributes.clone(),
                    regions: vec![when_true, when_false],
                    span: choice.span.clone(),
                })
            }
            opcode => Err(HirLoweringError::Unsupported(format!(
                "function-valued argument {} of {:?} has unsupported expression {opcode:?}",
                template.parameters[argument_index].name, template.symbol
            ))),
        }
    }

    fn specialize(
        &mut self,
        template: &wire::HirFunctionProto,
        function_parameters: &[usize],
        arguments: &[wire::HirExpressionProto],
        work: &mut HirNormalizationWork,
    ) -> Result<wire::HirExpressionProto, HirLoweringError> {
        work.visit_higher_order();
        if arguments.len() != template.parameters.len() {
            return Err(HirLoweringError::InvalidQCore(format!(
                "higher-order call to {:?} has the wrong arity",
                template.symbol
            )));
        }
        let mut candidates = Vec::with_capacity(function_parameters.len());
        for index in function_parameters {
            let argument = &arguments[*index];
            if argument.opcode != "function_ref"
                || !argument.operands.is_empty()
                || !argument.regions.is_empty()
            {
                return Err(HirLoweringError::Unsupported(format!(
                    "function-valued argument {} of {:?} is not one closed function reference",
                    template.parameters[*index].name, template.symbol
                )));
            }
            let candidate = string_attribute(&argument.attributes, "callee")?.to_owned();
            if self.templates.contains_key(&candidate) {
                return Err(HirLoweringError::Unsupported(format!(
                    "higher-order function candidate {candidate:?}"
                )));
            }
            candidates.push(candidate);
        }
        let argument_types = arguments
            .iter()
            .map(|argument| {
                argument
                    .r#type
                    .as_ref()
                    .map(prost::Message::encode_to_vec)
                    .ok_or_else(|| {
                        HirLoweringError::InvalidQCore(
                            "higher-order call argument has no type".to_owned(),
                        )
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let key = HigherOrderSpecializationKey {
            template: template.symbol.clone(),
            candidates,
            argument_types,
        };
        if let Some(function) = self.generated.get(&key) {
            return hir_function_reference(
                function,
                arguments.first().and_then(|value| value.span.clone()),
            );
        }
        if let Some(reference) = self.active.get(&key) {
            // Recursive calls reuse the checked signature reserved before the
            // body is rewritten; no second specialization is generated.
            let mut reference = reference.clone();
            reference.span = arguments.first().and_then(|value| value.span.clone());
            return Ok(reference);
        }
        if self.generated.len() + self.active.len() >= MAX_FUNCTION_VALUE_CANDIDATES {
            return Err(HirLoweringError::Unsupported(format!(
                "higher-order specialization exceeds {MAX_FUNCTION_VALUE_CANDIDATES} variants"
            )));
        }
        self.used_templates.insert(template.symbol.clone());
        let result =
            self.build_specialization(template, function_parameters, arguments, &key, work);
        self.active.remove(&key);
        let function = result?;
        let reference = hir_function_reference(
            &function,
            arguments.first().and_then(|value| value.span.clone()),
        )?;
        self.generated.insert(key, function);
        Ok(reference)
    }

    fn build_specialization(
        &mut self,
        template: &wire::HirFunctionProto,
        function_parameters: &[usize],
        arguments: &[wire::HirExpressionProto],
        key: &HigherOrderSpecializationKey,
        work: &mut HirNormalizationWork,
    ) -> Result<wire::HirFunctionProto, HirLoweringError> {
        work.visit_higher_order();
        let (mut arena, template_scope, bindings) =
            infer_higher_order_bindings(template, arguments)?;
        let mut function = template.clone();
        instantiate_hir_function_types(&mut function, &template_scope, &mut arena, &bindings)?;
        function.symbol = specialization_symbol(&self.module_name, key)?;
        if self.source_symbols.contains(&function.symbol)
            || self
                .generated
                .values()
                .any(|generated| generated.symbol == function.symbol)
        {
            return Err(HirLoweringError::Unsupported(
                "higher-order specialization symbol collision".to_owned(),
            ));
        }
        introduce_function_candidates(&mut function, function_parameters, arguments);
        let mut clone_arena = TypeArena::default();
        let clone_scope = clone_arena.intern_scope(&function.symbol);
        let normalized = preserve_function_yield_aliases(
            function.body.as_ref().expect("normalized template body"),
            &clone_scope,
            &mut clone_arena,
            work,
        )?;
        function.body = Some(normalized);
        remove_unused_introduced_function_tags(
            function.body.as_mut().expect("normalized template body"),
            template,
            function_parameters,
            work,
        )?;
        self.active.insert(
            key.clone(),
            hir_function_reference(&function, function.span.clone())?,
        );
        self.rewrite_block(
            function
                .body
                .as_mut()
                .expect("normalized specialization body"),
            0,
            work,
        )?;
        function_results::lower_generated(&mut function, work)?;
        if self.source_symbols.contains(&function.symbol)
            || self
                .generated
                .values()
                .any(|generated| generated.symbol == function.symbol)
        {
            return Err(HirLoweringError::Unsupported(
                "higher-order specialization symbol collision".to_owned(),
            ));
        }
        reject_template_function_references(
            function
                .body
                .as_ref()
                .expect("normalized specialization body"),
            &self.templates,
            work,
        )?;
        let nodes = hir_block_node_count(
            function
                .body
                .as_ref()
                .expect("normalized specialization body"),
        )?;
        self.expanded_nodes = self.expanded_nodes.checked_add(nodes).ok_or_else(|| {
            HirLoweringError::Unsupported("higher-order expansion overflow".to_owned())
        })?;
        if self.expanded_nodes > budget::MAX_MESSAGE_NODES {
            return Err(HirLoweringError::Unsupported(format!(
                "higher-order expansion exceeds {} nodes",
                budget::MAX_MESSAGE_NODES
            )));
        }
        Ok(function)
    }
}

fn hir_function_reference(
    function: &wire::HirFunctionProto,
    span: Option<wire::SourceSpanProto>,
) -> Result<wire::HirExpressionProto, HirLoweringError> {
    let mut arguments = function
        .parameters
        .iter()
        .map(|parameter| {
            parameter.r#type.clone().ok_or_else(|| {
                HirLoweringError::InvalidQCore(format!(
                    "function parameter {:?} has no type",
                    parameter.name
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    arguments.push(function.result_type.clone().ok_or_else(|| {
        HirLoweringError::InvalidQCore(format!("function {:?} has no result type", function.symbol))
    })?);
    Ok(wire::HirExpressionProto {
        opcode: "function_ref".to_owned(),
        r#type: Some(wire::TypeProto {
            kind: wire::type_proto::Kind::Function.into(),
            arguments,
            field_names: function
                .parameters
                .iter()
                .map(|parameter| parameter.name.clone())
                .collect(),
            ..Default::default()
        }),
        attributes: vec![wire::AttributeProto {
            name: "callee".to_owned(),
            value: Some(wire::LiteralProto {
                value: Some(wire::literal_proto::Value::StringValue(
                    function.symbol.clone(),
                )),
            }),
        }],
        span,
        ..Default::default()
    })
}

fn function_type_at_choice_stage(
    function_type: &wire::TypeProto,
    choice_type: &wire::TypeProto,
) -> Result<wire::TypeProto, HirLoweringError> {
    let base = function_base_type(function_type)?;
    function_base_type(choice_type)?;
    let stage = Stage::join(
        wire_function_stage(function_type)?,
        wire_function_stage(choice_type)?,
    );
    if let Some(stage) = stage {
        Ok(wire::TypeProto {
            kind: wire::type_proto::Kind::Staged.into(),
            name: stage.as_str().to_owned(),
            arguments: vec![base],
            ..Default::default()
        })
    } else {
        Ok(base)
    }
}

fn wire_function_stage(r#type: &wire::TypeProto) -> Result<Option<Stage>, HirLoweringError> {
    match wire::type_proto::Kind::try_from(r#type.kind).ok() {
        Some(wire::type_proto::Kind::Function) => Ok(None),
        Some(wire::type_proto::Kind::Staged) if r#type.arguments.len() == 1 => {
            Stage::from_name(&r#type.name).map(Some).ok_or_else(|| {
                HirLoweringError::InvalidQCore("callable choice has an invalid stage".to_owned())
            })
        }
        _ => Err(HirLoweringError::InvalidQCore(
            "callable choice has a non-function type".to_owned(),
        )),
    }
}

fn infer_higher_order_bindings(
    template: &wire::HirFunctionProto,
    arguments: &[wire::HirExpressionProto],
) -> Result<HigherOrderBindings, HirLoweringError> {
    let mut arena = TypeArena::default();
    let template_scope = arena.intern_scope(&template.symbol);
    let mut bindings = HashMap::new();
    for (parameter, argument) in template.parameters.iter().zip(arguments) {
        let expected = arena
            .resolve_wire_type(
                parameter.r#type.as_ref().ok_or_else(|| {
                    HirLoweringError::InvalidQCore("higher-order parameter has no type".to_owned())
                })?,
                &template_scope,
            )
            .map_err(HirLoweringError::InvalidQCore)?;
        let actual = arena
            .resolve_wire_type(
                argument.r#type.as_ref().ok_or_else(|| {
                    HirLoweringError::InvalidQCore("higher-order argument has no type".to_owned())
                })?,
                &template_scope,
            )
            .map_err(HirLoweringError::InvalidQCore)?;
        if !unify_assignable(&arena, expected, actual, &mut bindings) {
            return Err(HirLoweringError::InvalidQCore(format!(
                "higher-order call argument does not match parameter {:?}",
                parameter.name
            )));
        }
    }
    Ok((arena, template_scope, bindings))
}

fn introduce_function_candidates(
    function: &mut wire::HirFunctionProto,
    function_parameters: &[usize],
    arguments: &[wire::HirExpressionProto],
) {
    let parameter_set = function_parameters.iter().copied().collect::<HashSet<_>>();
    let mut introductions = Vec::with_capacity(function_parameters.len());
    for index in function_parameters {
        let parameter = &function.parameters[*index];
        let mut candidate = arguments[*index].clone();
        candidate.r#type.clone_from(&parameter.r#type);
        introductions.push(wire::HirStatementProto {
            opcode: "assign".to_owned(),
            expression: Some(candidate),
            targets: vec![parameter.name.clone()],
            span: function.span.clone(),
            ..Default::default()
        });
    }
    function.parameters = function
        .parameters
        .drain(..)
        .enumerate()
        .filter_map(|(index, parameter)| (!parameter_set.contains(&index)).then_some(parameter))
        .collect();
    function
        .body
        .as_mut()
        .expect("normalized template body")
        .statements
        .splice(0..0, introductions);
}

fn remove_unused_introduced_function_tags(
    body: &mut wire::HirBlockProto,
    template: &wire::HirFunctionProto,
    function_parameters: &[usize],
    work: &mut HirNormalizationWork,
) -> Result<(), HirLoweringError> {
    let tags = function_parameters
        .iter()
        .map(|index| function_tag_name(&template.parameters[*index].name))
        .collect::<HashSet<_>>();
    let introduction_count = function_parameters.len().min(body.statements.len());
    let mut used = HashSet::new();
    let mut blocks = body.statements[introduction_count..]
        .iter()
        .flat_map(|statement| statement.regions.iter())
        .collect::<Vec<_>>();
    for statement in &body.statements[introduction_count..] {
        work.visit_higher_order();
        if let Some(expression) = &statement.expression {
            work.charge_higher_order(expression_node_count(expression)?);
            used.extend(expression_names(expression));
        }
        if statement
            .attributes
            .iter()
            .any(|attribute| attribute.name == "captures")
        {
            used.extend(string_tuple_attribute(&statement.attributes, "captures")?);
        }
    }
    while let Some(block) = blocks.pop() {
        for statement in &block.statements {
            work.visit_higher_order();
            if let Some(expression) = &statement.expression {
                work.charge_higher_order(expression_node_count(expression)?);
                used.extend(expression_names(expression));
            }
            if statement
                .attributes
                .iter()
                .any(|attribute| attribute.name == "captures")
            {
                used.extend(string_tuple_attribute(&statement.attributes, "captures")?);
            }
            blocks.extend(&statement.regions);
        }
    }
    let mut position = 0_usize;
    body.statements.retain(|statement| {
        let remove = position < introduction_count
            && statement.opcode == "assign"
            && matches!(statement.targets.as_slice(), [target] if tags.contains(target))
            && statement
                .targets
                .iter()
                .all(|target| !used.contains(target));
        position += 1;
        !remove
    });
    Ok(())
}

fn specialization_symbol(
    module_name: &str,
    key: &HigherOrderSpecializationKey,
) -> Result<String, HirLoweringError> {
    let mut digest = Sha256::new();
    hash_specialization_field(&mut digest, key.template.as_bytes())?;
    for candidate in &key.candidates {
        hash_specialization_field(&mut digest, candidate.as_bytes())?;
    }
    for argument_type in &key.argument_types {
        hash_specialization_field(&mut digest, argument_type)?;
    }
    let mut suffix = String::with_capacity(64);
    for byte in digest.finalize() {
        write!(&mut suffix, "{byte:02x}").expect("writing to String cannot fail");
    }
    let template_name = key.template.rsplit('.').next().unwrap_or(&key.template);
    let symbol = format!("{module_name}.__fn_{template_name}_{suffix}");
    if symbol.len() > budget::MAX_STRING_BYTES {
        return Err(HirLoweringError::Unsupported(
            "higher-order specialization symbol exceeds the string budget".to_owned(),
        ));
    }
    Ok(symbol)
}

fn hash_specialization_field(digest: &mut Sha256, value: &[u8]) -> Result<(), HirLoweringError> {
    let length = u64::try_from(value.len()).map_err(|_| {
        HirLoweringError::Unsupported("higher-order specialization key size".to_owned())
    })?;
    digest.update(length.to_le_bytes());
    digest.update(value);
    Ok(())
}

fn instantiate_hir_function_types(
    function: &mut wire::HirFunctionProto,
    scope: &Arc<str>,
    arena: &mut TypeArena,
    bindings: &HashMap<Dimension, Dimension>,
) -> Result<(), HirLoweringError> {
    for parameter in &mut function.parameters {
        instantiate_hir_type(&mut parameter.r#type, scope, arena, bindings)?;
    }
    instantiate_hir_type(&mut function.result_type, scope, arena, bindings)?;
    let mut blocks = vec![function.body.as_mut().expect("template body")];
    while let Some(block) = blocks.pop() {
        for statement in &mut block.statements {
            for result_type in &mut statement.result_types {
                instantiate_wire_type(result_type, scope, arena, bindings)?;
            }
            if let Some(expression) = &mut statement.expression {
                instantiate_hir_expression_types(expression, scope, arena, bindings)?;
            }
            blocks.extend(statement.regions.iter_mut());
        }
    }
    Ok(())
}

fn instantiate_hir_expression_types(
    root: &mut wire::HirExpressionProto,
    scope: &Arc<str>,
    arena: &mut TypeArena,
    bindings: &HashMap<Dimension, Dimension>,
) -> Result<(), HirLoweringError> {
    let mut pending = vec![root];
    while let Some(expression) = pending.pop() {
        instantiate_hir_type(&mut expression.r#type, scope, arena, bindings)?;
        pending.extend(expression.operands.iter_mut());
        pending.extend(expression.regions.iter_mut());
    }
    Ok(())
}

fn instantiate_hir_type(
    r#type: &mut Option<wire::TypeProto>,
    scope: &Arc<str>,
    arena: &mut TypeArena,
    bindings: &HashMap<Dimension, Dimension>,
) -> Result<(), HirLoweringError> {
    let value = r#type.as_mut().ok_or_else(|| {
        HirLoweringError::InvalidQCore("higher-order node has no type".to_owned())
    })?;
    instantiate_wire_type(value, scope, arena, bindings)
}

fn instantiate_wire_type(
    r#type: &mut wire::TypeProto,
    scope: &Arc<str>,
    arena: &mut TypeArena,
    bindings: &HashMap<Dimension, Dimension>,
) -> Result<(), HirLoweringError> {
    let type_id = arena
        .resolve_wire_type(r#type, scope)
        .map_err(HirLoweringError::InvalidQCore)?;
    let type_id = arena
        .substitute_dimensions(type_id, bindings)
        .map_err(HirLoweringError::InvalidQCore)?;
    *r#type = arena
        .encode_wire_type(type_id)
        .map_err(HirLoweringError::InvalidQCore)?;
    Ok(())
}

fn reject_template_function_references(
    root: &wire::HirBlockProto,
    templates: &HashMap<String, wire::HirFunctionProto>,
    work: &mut HirNormalizationWork,
) -> Result<(), HirLoweringError> {
    let mut blocks = vec![root];
    let mut expressions = Vec::new();
    while let Some(block) = blocks.pop() {
        for statement in &block.statements {
            work.visit_higher_order();
            if let Some(expression) = &statement.expression {
                expressions.push(expression);
            }
            blocks.extend(&statement.regions);
        }
    }
    while let Some(expression) = expressions.pop() {
        work.visit_higher_order();
        if expression.opcode == "function_ref" {
            let symbol = string_attribute(&expression.attributes, "callee")?;
            if templates.contains_key(symbol) {
                return Err(HirLoweringError::Unsupported(format!(
                    "higher-order function reference {symbol:?}"
                )));
            }
        }
        expressions.extend(&expression.operands);
        expressions.extend(&expression.regions);
    }
    Ok(())
}

fn hir_block_node_count(root: &wire::HirBlockProto) -> Result<usize, HirLoweringError> {
    let mut count = 0_usize;
    let mut blocks = vec![root];
    while let Some(block) = blocks.pop() {
        count = count.checked_add(block.statements.len()).ok_or_else(|| {
            HirLoweringError::Unsupported("higher-order node count overflow".to_owned())
        })?;
        for statement in &block.statements {
            if let Some(expression) = &statement.expression {
                count = count
                    .checked_add(expression_node_count(expression)?)
                    .ok_or_else(|| {
                        HirLoweringError::Unsupported("higher-order node count overflow".to_owned())
                    })?;
            }
            blocks.extend(&statement.regions);
        }
    }
    Ok(count)
}

fn higher_order_candidate_combinations(
    parameter_indices: &[usize],
    arguments: &[wire::HirExpressionProto],
) -> Result<usize, HirLoweringError> {
    parameter_indices.iter().try_fold(1_usize, |total, index| {
        let argument = arguments.get(*index).ok_or_else(|| {
            HirLoweringError::InvalidQCore(
                "higher-order parameter index is outside the argument row".to_owned(),
            )
        })?;
        let candidates = callable_candidate_count(argument)?;
        total.checked_mul(candidates).ok_or_else(|| {
            HirLoweringError::Unsupported("higher-order candidate product overflow".to_owned())
        })
    })
}

fn lower_function(
    function: &wire::HirFunctionProto,
    work: &mut HirSsaLoweringWork,
) -> Result<wire::FunctionProto, HirLoweringError> {
    work.visit_function();
    let mut arena = TypeArena::default();
    let scope = arena.intern_scope(&function.symbol);
    let mut next_id = 0_u64;
    let mut next_region = 0_u64;
    let mut region_children = RegionChildren::new();
    let mut environment = HashMap::new();
    let mut parameters = Vec::with_capacity(function.parameters.len());
    for parameter in &function.parameters {
        let r#type = parameter
            .r#type
            .as_ref()
            .ok_or_else(|| HirLoweringError::Unsupported("parameter without a type".to_owned()))?;
        let type_id = arena
            .resolve_wire_type(r#type, &scope)
            .map_err(HirLoweringError::InvalidQCore)?;
        let region = arena.is_quantum(type_id).then(|| {
            let root = next_region;
            next_region += 1;
            wire::RegionRefProto {
                root,
                ..Default::default()
            }
        });
        let value = wire::ValueProto {
            id: next_id,
            r#type: Some(r#type.clone()),
            hint: parameter.name.clone(),
            has_region: region.is_some(),
            region,
        };
        next_id += 1;
        environment.insert(parameter.name.clone(), value.clone());
        parameters.push(wire::ParameterProto {
            name: parameter.name.clone(),
            value: Some(value),
            default_value: parameter.default_value.clone(),
            has_default: parameter.has_default,
        });
    }
    let span = function.span.clone();
    let body = if function.external {
        wire::BlockProto {
            terminator: Some(wire::TerminatorProto {
                kind: "external".to_owned(),
                span: span.clone(),
                ..Default::default()
            }),
            ..Default::default()
        }
    } else {
        lower_straight_body(
            function.body.as_ref().ok_or_else(|| {
                HirLoweringError::Unsupported("function without a body".to_owned())
            })?,
            &scope,
            &mut arena,
            &mut next_id,
            &mut next_region,
            &mut region_children,
            &mut environment,
            work,
        )?
    };
    Ok(wire::FunctionProto {
        symbol: function.symbol.clone(),
        parameters,
        result_type: function.result_type.clone(),
        body: Some(body),
        span,
        asserted_kind: function.declared_kind.clone(),
        external: function.external,
        intent: function.intent.clone(),
    })
}

fn eliminate_local_function_aliases(
    source: &wire::HirBlockProto,
    scope: &std::sync::Arc<str>,
    arena: &mut TypeArena,
    work: &mut HirNormalizationWork,
) -> Result<wire::HirBlockProto, HirLoweringError> {
    normalize_function_aliases(source, scope, arena, FunctionYieldMode::Lower, work)
}

fn preserve_function_yield_aliases(
    source: &wire::HirBlockProto,
    scope: &std::sync::Arc<str>,
    arena: &mut TypeArena,
    work: &mut HirNormalizationWork,
) -> Result<wire::HirBlockProto, HirLoweringError> {
    normalize_function_aliases(source, scope, arena, FunctionYieldMode::Preserve, work)
}

fn normalize_function_aliases(
    source: &wire::HirBlockProto,
    scope: &std::sync::Arc<str>,
    arena: &mut TypeArena,
    yield_mode: FunctionYieldMode,
    work: &mut HirNormalizationWork,
) -> Result<wire::HirBlockProto, HirLoweringError> {
    let catalog = FunctionAliasCatalog::build(source, scope, arena, work)?;
    let mut expanded_nodes = 0;
    normalize_function_alias_block(
        source,
        FunctionAliasEnvironment::default(),
        &catalog,
        scope,
        arena,
        &mut expanded_nodes,
        yield_mode,
        0,
        work,
    )
    .map(|(block, _)| block)
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum FunctionYieldMode {
    Lower,
    Preserve,
}

#[derive(Clone)]
struct FunctionAliasBinding {
    function_type: wire::TypeProto,
    tag_type: wire::TypeProto,
    candidates: std::sync::Arc<[String]>,
}

#[derive(Clone, Default)]
struct FunctionAliasEnvironment {
    aliases: HashMap<String, FunctionAliasBinding>,
}

type CallableFacts = FiniteDependencyFacts<FunctionAliasKey, String>;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct FunctionAliasKey {
    name: String,
    signature: Vec<u8>,
}

struct FunctionAliasCatalog {
    candidates: HashMap<FunctionAliasKey, std::sync::Arc<[String]>>,
    references: HashMap<String, wire::HirExpressionProto>,
    tags: HashMap<String, i64>,
}

impl FunctionAliasCatalog {
    fn build(
        source: &wire::HirBlockProto,
        scope: &std::sync::Arc<str>,
        arena: &mut TypeArena,
        work: &mut HirNormalizationWork,
    ) -> Result<Self, HirLoweringError> {
        let mut facts = HashMap::<FunctionAliasKey, CallableFacts>::new();
        let mut references = HashMap::<String, wire::HirExpressionProto>::new();
        let mut pending_blocks = vec![source];
        while let Some(block) = pending_blocks.pop() {
            for statement in &block.statements {
                work.visit_function_alias();
                pending_blocks.extend(statement.regions.iter());
                if statement.opcode != "assign"
                    || !expression_has_function_type(statement.expression.as_ref(), scope, arena)?
                {
                    continue;
                }
                let [target] = statement.targets.as_slice() else {
                    return Err(HirLoweringError::Unsupported(
                        "destructuring a function value".to_owned(),
                    ));
                };
                let expression = statement
                    .expression
                    .as_ref()
                    .expect("validated assignment expression");
                let key = function_alias_key(
                    target,
                    expression.r#type.as_ref().ok_or_else(|| {
                        HirLoweringError::InvalidQCore("function alias has no type".to_owned())
                    })?,
                )?;
                work.charge_function_aliases(expression_node_count(expression)?);
                collect_callable_facts(expression, facts.entry(key).or_default(), &mut references)?;
            }
        }

        let candidate_sets = solve_finite_dependency_sets(
            &facts,
            || work.visit_function_alias(),
            |count| {
                if count > MAX_FUNCTION_VALUE_CANDIDATES {
                    Err(HirLoweringError::Unsupported(format!(
                        "function value exceeds {MAX_FUNCTION_VALUE_CANDIDATES} static candidates"
                    )))
                } else {
                    Ok(())
                }
            },
        )?;
        let symbols = references.keys().cloned().collect::<Vec<_>>();
        validate_distinct_function_tags(&symbols)?;
        let tags = symbols
            .into_iter()
            .map(|symbol| {
                let tag = stable_function_tag(&symbol);
                (symbol, tag)
            })
            .collect::<HashMap<_, _>>();
        let candidates = candidate_sets
            .into_iter()
            .map(|(key, candidates)| {
                if candidates.is_empty() {
                    return Err(HirLoweringError::InvalidQCore(format!(
                        "function alias {:?} has no closed candidate",
                        key.name
                    )));
                }
                let mut candidates = candidates.into_iter().collect::<Vec<_>>();
                candidates.sort_by_key(|symbol| tags.get(symbol).copied());
                Ok((key, std::sync::Arc::from(candidates)))
            })
            .collect::<Result<HashMap<_, _>, HirLoweringError>>()?;
        Ok(Self {
            candidates,
            references,
            tags,
        })
    }

    fn candidates_for(
        &self,
        name: &str,
        function_type: &wire::TypeProto,
    ) -> Result<std::sync::Arc<[String]>, HirLoweringError> {
        self.candidates
            .get(&function_alias_key(name, function_type)?)
            .cloned()
            .ok_or_else(|| {
                HirLoweringError::InvalidQCore(format!(
                    "function alias {name:?} has no candidate catalog"
                ))
            })
    }

    fn candidates_for_yield(&self) -> Result<Arc<[String]>, HirLoweringError> {
        let mut matches = self.candidates.iter().filter_map(|(key, candidates)| {
            (key.name == FUNCTION_YIELD_ALIAS).then_some(candidates.clone())
        });
        let candidates = matches.next().ok_or_else(|| {
            HirLoweringError::InvalidQCore(
                "callable region yield has no candidate catalog".to_owned(),
            )
        })?;
        if matches.next().is_some() {
            return Err(HirLoweringError::InvalidQCore(
                "callable region yield has multiple candidate catalogs".to_owned(),
            ));
        }
        Ok(candidates)
    }

    fn from_callable(
        source: &wire::HirExpressionProto,
        name: &str,
    ) -> Result<Self, HirLoweringError> {
        let mut facts = CallableFacts::default();
        let mut references = HashMap::new();
        collect_callable_facts(source, &mut facts, &mut references)?;
        if !facts.dependencies.is_empty() || facts.direct.is_empty() {
            return Err(HirLoweringError::InvalidQCore(
                "inline callable has unresolved candidates".to_owned(),
            ));
        }
        let mut symbols = facts.direct.into_iter().collect::<Vec<_>>();
        symbols.sort_by_key(|symbol| stable_function_tag(symbol));
        if symbols.len() > MAX_FUNCTION_VALUE_CANDIDATES {
            return Err(HirLoweringError::Unsupported(format!(
                "function value exceeds {MAX_FUNCTION_VALUE_CANDIDATES} static candidates"
            )));
        }
        validate_distinct_function_tags(&symbols)?;
        let tags = symbols
            .iter()
            .map(|symbol| (symbol.clone(), stable_function_tag(symbol)))
            .collect::<HashMap<_, _>>();
        let function_type = source.r#type.as_ref().ok_or_else(|| {
            HirLoweringError::InvalidQCore("inline callable has no type".to_owned())
        })?;
        let candidates =
            HashMap::from([(function_alias_key(name, function_type)?, Arc::from(symbols))]);
        Ok(Self {
            candidates,
            references,
            tags,
        })
    }
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn normalize_function_alias_block(
    source: &wire::HirBlockProto,
    mut environment: FunctionAliasEnvironment,
    catalog: &FunctionAliasCatalog,
    scope: &std::sync::Arc<str>,
    arena: &mut TypeArena,
    expanded_nodes: &mut usize,
    yield_mode: FunctionYieldMode,
    depth: usize,
    work: &mut HirNormalizationWork,
) -> Result<(wire::HirBlockProto, FunctionAliasEnvironment), HirLoweringError> {
    if depth > budget::MAX_NESTING {
        return Err(HirLoweringError::Unsupported(
            "function alias control-flow nesting".to_owned(),
        ));
    }
    let mut statements = Vec::with_capacity(source.statements.len());
    let mut position = 0;
    while position < source.statements.len() {
        work.visit_function_alias();
        let source_statement = &source.statements[position];
        if source_statement.opcode == "if" {
            let statement = normalize_function_alias_if(
                source_statement,
                environment,
                catalog,
                scope,
                arena,
                expanded_nodes,
                yield_mode,
                depth + 1,
                work,
            )?;
            statements.push(statement.0);
            environment = statement.1;
            position += 1;
            continue;
        }
        if matches!(source_statement.opcode.as_str(), "for" | "while") {
            let statement = normalize_function_alias_loop(
                source_statement,
                environment,
                catalog,
                scope,
                arena,
                expanded_nodes,
                yield_mode,
                depth + 1,
                work,
            )?;
            statements.push(statement.0);
            environment = statement.1;
            position += 1;
            continue;
        }

        let mut statement = source_statement.clone();
        let is_function_alias = statement.opcode == "assign"
            && expression_has_function_type(statement.expression.as_ref(), scope, arena)?;
        if is_function_alias {
            let [target] = statement.targets.as_slice() else {
                return Err(HirLoweringError::Unsupported(
                    "destructuring a function value".to_owned(),
                ));
            };
            let target = target.clone();
            let expression = statement
                .expression
                .expect("validated assignment expression");
            let function_type = expression.r#type.clone().ok_or_else(|| {
                HirLoweringError::InvalidQCore("function alias has no type".to_owned())
            })?;
            work.charge_function_aliases(expression_node_count(&expression)?);
            let tag_expression = lower_callable_to_tag(
                &expression,
                &target,
                &environment,
                catalog,
                expanded_nodes,
                0,
            )?;
            work.charge_function_aliases(expression_node_count(&tag_expression)?);
            let tag_type = tag_expression.r#type.clone().ok_or_else(|| {
                HirLoweringError::InvalidQCore("function tag has no type".to_owned())
            })?;
            let function_type = function_at_tag_stage(&function_type, &tag_type)?;
            statement.expression = Some(tag_expression);
            statement.targets = vec![function_tag_name(&target)];
            let candidates = catalog.candidates_for(&target, &function_type)?;
            environment.aliases.insert(
                target,
                FunctionAliasBinding {
                    function_type,
                    tag_type,
                    candidates,
                },
            );
            statements.push(statement);
            position += 1;
            continue;
        }

        let is_function_yield = statement.opcode == "return"
            && expression_has_function_type(statement.expression.as_ref(), scope, arena)?;
        if is_function_yield && yield_mode == FunctionYieldMode::Lower {
            let expression = statement
                .expression
                .take()
                .expect("validated return expression");
            work.charge_function_aliases(expression_node_count(&expression)?);
            let tag_expression = lower_callable_to_tag(
                &expression,
                FUNCTION_YIELD_ALIAS,
                &environment,
                catalog,
                expanded_nodes,
                0,
            )?;
            work.charge_function_aliases(expression_node_count(&tag_expression)?);
            statement.expression = Some(tag_expression);
            statements.push(statement);
            position += 1;
            continue;
        }

        if let Some(expression) = &mut statement.expression {
            work.charge_function_aliases(expression_node_count(expression)?);
            rewrite_function_alias_names(
                expression,
                &environment.aliases,
                catalog,
                expanded_nodes,
                budget::MAX_MESSAGE_NODES,
                0,
            )?;
        }

        for target in &statement.targets {
            environment.aliases.remove(target);
        }
        statements.push(statement);
        position += 1;
    }
    Ok((wire::HirBlockProto { statements }, environment))
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn normalize_function_alias_if(
    source: &wire::HirStatementProto,
    mut environment: FunctionAliasEnvironment,
    catalog: &FunctionAliasCatalog,
    scope: &std::sync::Arc<str>,
    arena: &mut TypeArena,
    expanded_nodes: &mut usize,
    yield_mode: FunctionYieldMode,
    depth: usize,
    work: &mut HirNormalizationWork,
) -> Result<(wire::HirStatementProto, FunctionAliasEnvironment), HirLoweringError> {
    let [true_source, false_source] = source.regions.as_slice() else {
        return Err(HirLoweringError::InvalidQCore(
            "if does not have two regions".to_owned(),
        ));
    };
    let mut statement = source.clone();
    let condition = statement
        .expression
        .as_mut()
        .ok_or_else(|| HirLoweringError::InvalidQCore("if condition is missing".to_owned()))?;
    work.charge_function_aliases(expression_node_count(condition)?);
    rewrite_function_alias_names(
        condition,
        &environment.aliases,
        catalog,
        expanded_nodes,
        budget::MAX_MESSAGE_NODES,
        0,
    )?;

    let (true_block, _) = normalize_function_alias_block(
        true_source,
        environment.clone(),
        catalog,
        scope,
        arena,
        expanded_nodes,
        yield_mode,
        depth,
        work,
    )?;
    let (false_block, _) = normalize_function_alias_block(
        false_source,
        environment.clone(),
        catalog,
        scope,
        arena,
        expanded_nodes,
        yield_mode,
        depth,
        work,
    )?;
    statement.regions = vec![true_block, false_block];

    let captures = string_tuple_attribute(&statement.attributes, "captures")?
        .into_iter()
        .map(|name| {
            environment
                .aliases
                .get(&name)
                .map_or(name.clone(), |_| function_tag_name(&name))
        })
        .collect::<Vec<_>>();
    let outputs = string_tuple_attribute(&statement.attributes, "outputs")?;
    if outputs.len() != statement.result_types.len() {
        return Err(HirLoweringError::InvalidQCore(
            "if output name/type arity mismatch".to_owned(),
        ));
    }
    let mut lowered_outputs = Vec::with_capacity(outputs.len());
    let mut lowered_types = Vec::with_capacity(outputs.len());
    for (name, r#type) in outputs.into_iter().zip(&statement.result_types) {
        if wire_type_is_function(r#type, scope, arena)? {
            let candidates = catalog.candidates_for(&name, r#type)?;
            let tag_type = function_tag_type(r#type)?;
            lowered_outputs.push(function_tag_name(&name));
            lowered_types.push(tag_type.clone());
            environment.aliases.insert(
                name,
                FunctionAliasBinding {
                    function_type: r#type.clone(),
                    tag_type,
                    candidates,
                },
            );
        } else {
            environment.aliases.remove(&name);
            lowered_outputs.push(name);
            lowered_types.push(r#type.clone());
        }
    }
    replace_string_tuple_attribute(&mut statement.attributes, "captures", &captures)?;
    replace_string_tuple_attribute(&mut statement.attributes, "outputs", &lowered_outputs)?;
    statement.result_types = lowered_types;
    Ok((statement, environment))
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn normalize_function_alias_loop(
    source: &wire::HirStatementProto,
    mut environment: FunctionAliasEnvironment,
    catalog: &FunctionAliasCatalog,
    scope: &std::sync::Arc<str>,
    arena: &mut TypeArena,
    expanded_nodes: &mut usize,
    yield_mode: FunctionYieldMode,
    depth: usize,
    work: &mut HirNormalizationWork,
) -> Result<(wire::HirStatementProto, FunctionAliasEnvironment), HirLoweringError> {
    let mut statement = source.clone();
    let captures = string_tuple_attribute(&statement.attributes, "captures")?;
    if captures.len() != statement.result_types.len() {
        return Err(HirLoweringError::InvalidQCore(format!(
            "{} carry name/type arity mismatch",
            statement.opcode
        )));
    }
    let outputs = string_tuple_attribute(&statement.attributes, "outputs")?;
    if outputs != captures {
        return Err(HirLoweringError::InvalidQCore(format!(
            "{} output/capture row mismatch",
            statement.opcode
        )));
    }
    let mut lowered_names = Vec::with_capacity(captures.len());
    let mut lowered_types = Vec::with_capacity(captures.len());
    for (name, r#type) in captures.into_iter().zip(&statement.result_types) {
        if wire_type_is_function(r#type, scope, arena)? {
            let candidates = catalog.candidates_for(&name, r#type)?;
            let tag_type = function_tag_type(r#type)?;
            lowered_names.push(function_tag_name(&name));
            lowered_types.push(tag_type.clone());
            environment.aliases.insert(
                name,
                FunctionAliasBinding {
                    function_type: r#type.clone(),
                    tag_type,
                    candidates,
                },
            );
        } else {
            environment.aliases.remove(&name);
            lowered_names.push(name);
            lowered_types.push(r#type.clone());
        }
    }
    if let Some(expression) = &mut statement.expression {
        work.charge_function_aliases(expression_node_count(expression)?);
        rewrite_function_alias_names(
            expression,
            &environment.aliases,
            catalog,
            expanded_nodes,
            budget::MAX_MESSAGE_NODES,
            0,
        )?;
    }
    statement.regions = source
        .regions
        .iter()
        .map(|region| {
            normalize_function_alias_block(
                region,
                environment.clone(),
                catalog,
                scope,
                arena,
                expanded_nodes,
                yield_mode,
                depth,
                work,
            )
            .map(|(block, _)| block)
        })
        .collect::<Result<Vec<_>, _>>()?;
    replace_string_tuple_attribute(&mut statement.attributes, "captures", &lowered_names)?;
    replace_string_tuple_attribute(&mut statement.attributes, "outputs", &lowered_names)?;
    statement.result_types = lowered_types;
    Ok((statement, environment))
}

fn wire_type_is_function(
    r#type: &wire::TypeProto,
    scope: &std::sync::Arc<str>,
    arena: &mut TypeArena,
) -> Result<bool, HirLoweringError> {
    let type_id = arena
        .resolve_wire_type(r#type, scope)
        .map_err(HirLoweringError::InvalidQCore)?;
    Ok(matches!(
        arena.get(arena.strip_stage(type_id)),
        Some(SemanticType::Function { .. })
    ))
}

fn function_tag_name(name: &str) -> String {
    format!("{INTERNAL_NAME_PREFIX}function.tag:{name}")
}

fn function_alias_key(
    name: &str,
    function_type: &wire::TypeProto,
) -> Result<FunctionAliasKey, HirLoweringError> {
    Ok(FunctionAliasKey {
        name: name.to_owned(),
        signature: function_base_type(function_type)?.encode_to_vec(),
    })
}

fn function_base_type(
    function_type: &wire::TypeProto,
) -> Result<wire::TypeProto, HirLoweringError> {
    match wire::type_proto::Kind::try_from(function_type.kind).ok() {
        Some(wire::type_proto::Kind::Function) => Ok(function_type.clone()),
        Some(wire::type_proto::Kind::Staged) if function_type.arguments.len() == 1 => {
            let inner = &function_type.arguments[0];
            if wire::type_proto::Kind::try_from(inner.kind).ok()
                == Some(wire::type_proto::Kind::Function)
            {
                Ok(inner.clone())
            } else {
                Err(HirLoweringError::InvalidQCore(
                    "staged function alias has a non-function payload".to_owned(),
                ))
            }
        }
        _ => Err(HirLoweringError::InvalidQCore(
            "function alias has a non-function type".to_owned(),
        )),
    }
}

fn collect_callable_facts(
    expression: &wire::HirExpressionProto,
    facts: &mut CallableFacts,
    references: &mut HashMap<String, wire::HirExpressionProto>,
) -> Result<(), HirLoweringError> {
    let mut pending = vec![expression];
    while let Some(candidate) = pending.pop() {
        match candidate.opcode.as_str() {
            "function_ref" => {
                let symbol = string_attribute(&candidate.attributes, "callee")?;
                facts.direct.insert(symbol.to_owned());
                references
                    .entry(symbol.to_owned())
                    .or_insert_with(|| candidate.clone());
            }
            "name" => {
                facts.dependencies.insert(function_alias_key(
                    string_attribute(&candidate.attributes, "name")?,
                    candidate.r#type.as_ref().ok_or_else(|| {
                        HirLoweringError::InvalidQCore(
                            "function alias reference has no type".to_owned(),
                        )
                    })?,
                )?);
            }
            "lazy_if" if candidate.regions.len() == 2 => {
                pending.extend(candidate.regions.iter());
            }
            opcode => {
                return Err(HirLoweringError::Unsupported(format!(
                    "callable expression {opcode:?}"
                )));
            }
        }
    }
    Ok(())
}

fn stable_function_tag(symbol: &str) -> i64 {
    let digest = Sha256::digest(symbol.as_bytes());
    let mut bytes = [0_u8; 8];
    bytes.copy_from_slice(&digest[..8]);
    i64::from_be_bytes(bytes) & i64::MAX
}

fn validate_distinct_function_tags(symbols: &[String]) -> Result<(), HirLoweringError> {
    let mut tags = HashMap::new();
    for symbol in symbols {
        let tag = stable_function_tag(symbol);
        if let Some(previous) = tags.insert(tag, symbol)
            && previous != symbol
        {
            return Err(HirLoweringError::Unsupported(format!(
                "function candidate tag collision between {previous:?} and {symbol:?}"
            )));
        }
    }
    Ok(())
}

fn function_tag_type(r#type: &wire::TypeProto) -> Result<wire::TypeProto, HirLoweringError> {
    let scalar = |name: &str| wire::TypeProto {
        kind: wire::type_proto::Kind::Scalar.into(),
        name: name.to_owned(),
        ..Default::default()
    };
    match wire::type_proto::Kind::try_from(r#type.kind).ok() {
        Some(wire::type_proto::Kind::Function) => Ok(wire::TypeProto {
            kind: wire::type_proto::Kind::Staged.into(),
            name: "Const".to_owned(),
            arguments: vec![scalar("Int")],
            ..Default::default()
        }),
        Some(wire::type_proto::Kind::Staged) if r#type.arguments.len() == 1 => {
            Ok(wire::TypeProto {
                kind: wire::type_proto::Kind::Staged.into(),
                name: r#type.name.clone(),
                arguments: vec![scalar("Int")],
                ..Default::default()
            })
        }
        _ => Err(HirLoweringError::InvalidQCore(
            "function alias has a non-function type".to_owned(),
        )),
    }
}

fn function_at_tag_stage(
    function_type: &wire::TypeProto,
    tag_type: &wire::TypeProto,
) -> Result<wire::TypeProto, HirLoweringError> {
    let base = function_base_type(function_type)?;
    if wire::type_proto::Kind::try_from(tag_type.kind).ok() == Some(wire::type_proto::Kind::Staged)
    {
        Ok(wire::TypeProto {
            kind: wire::type_proto::Kind::Staged.into(),
            name: tag_type.name.clone(),
            arguments: vec![base],
            ..Default::default()
        })
    } else {
        Ok(base)
    }
}

fn scalar_at_tag_stage(tag_type: &wire::TypeProto, name: &str) -> wire::TypeProto {
    let scalar = wire::TypeProto {
        kind: wire::type_proto::Kind::Scalar.into(),
        name: name.to_owned(),
        ..Default::default()
    };
    if wire::type_proto::Kind::try_from(tag_type.kind).ok() == Some(wire::type_proto::Kind::Staged)
    {
        wire::TypeProto {
            kind: wire::type_proto::Kind::Staged.into(),
            name: tag_type.name.clone(),
            arguments: vec![scalar],
            ..Default::default()
        }
    } else {
        scalar
    }
}

fn lower_callable_to_tag(
    expression: &wire::HirExpressionProto,
    target: &str,
    environment: &FunctionAliasEnvironment,
    catalog: &FunctionAliasCatalog,
    expanded_nodes: &mut usize,
    depth: usize,
) -> Result<wire::HirExpressionProto, HirLoweringError> {
    let lowered = callable_tag_expression(expression, target, environment, catalog, depth)?;
    let added = expression_node_count(&lowered)?;
    *expanded_nodes = expanded_nodes.checked_add(added).ok_or_else(|| {
        HirLoweringError::Unsupported("function alias expansion overflow".to_owned())
    })?;
    if *expanded_nodes > budget::MAX_MESSAGE_NODES {
        return Err(HirLoweringError::Unsupported(format!(
            "function alias expansion exceeds {} nodes",
            budget::MAX_MESSAGE_NODES
        )));
    }
    Ok(lowered)
}

fn callable_tag_expression(
    expression: &wire::HirExpressionProto,
    target: &str,
    environment: &FunctionAliasEnvironment,
    catalog: &FunctionAliasCatalog,
    depth: usize,
) -> Result<wire::HirExpressionProto, HirLoweringError> {
    if depth > budget::MAX_NESTING {
        return Err(HirLoweringError::Unsupported(
            "function tag expression depth".to_owned(),
        ));
    }
    match expression.opcode.as_str() {
        "function_ref" => {
            let symbol = string_attribute(&expression.attributes, "callee")?;
            let function_type = expression.r#type.as_ref().ok_or_else(|| {
                HirLoweringError::InvalidQCore("function candidate has no type".to_owned())
            })?;
            let candidates = if target == FUNCTION_YIELD_ALIAS {
                catalog.candidates_for_yield()?
            } else {
                catalog.candidates_for(target, function_type)?
            };
            if !candidates.iter().any(|candidate| candidate == symbol) {
                return Err(HirLoweringError::InvalidQCore(format!(
                    "function candidate {symbol:?} is absent from alias {target:?}"
                )));
            }
            let tag = catalog.tags.get(symbol).copied().ok_or_else(|| {
                HirLoweringError::InvalidQCore(format!("function candidate {symbol:?} has no tag"))
            })?;
            Ok(function_tag_literal(tag, expression.span.clone()))
        }
        "name" => {
            let name = string_attribute(&expression.attributes, "name")?;
            let binding = environment.aliases.get(name).ok_or_else(|| {
                HirLoweringError::InvalidQCore(format!("unresolved function alias {name:?}"))
            })?;
            Ok(hir_name_expression(
                &function_tag_name(name),
                binding.tag_type.clone(),
                expression.span.clone(),
            ))
        }
        "lazy_if" => {
            let [condition] = expression.operands.as_slice() else {
                return Err(HirLoweringError::InvalidQCore(
                    "callable lazy_if does not have one condition".to_owned(),
                ));
            };
            let [when_true, when_false] = expression.regions.as_slice() else {
                return Err(HirLoweringError::InvalidQCore(
                    "callable lazy_if does not have two arms".to_owned(),
                ));
            };
            let mut condition = condition.clone();
            let mut ignored_budget = 0;
            rewrite_function_alias_names(
                &mut condition,
                &environment.aliases,
                catalog,
                &mut ignored_budget,
                budget::MAX_MESSAGE_NODES,
                depth + 1,
            )?;
            let when_true =
                callable_tag_expression(when_true, target, environment, catalog, depth + 1)?;
            let when_false =
                callable_tag_expression(when_false, target, environment, catalog, depth + 1)?;
            let true_names = expression_names(&when_true);
            let false_names = expression_names(&when_false);
            let mut captures = true_names.union(&false_names).cloned().collect::<Vec<_>>();
            captures.sort();
            Ok(wire::HirExpressionProto {
                opcode: "lazy_if".to_owned(),
                r#type: Some(function_tag_type(expression.r#type.as_ref().ok_or_else(
                    || HirLoweringError::InvalidQCore("function choice has no type".to_owned()),
                )?)?),
                operands: vec![condition],
                attributes: vec![string_tuple_qcore_attribute("captures", &captures)],
                regions: vec![when_true, when_false],
                span: expression.span.clone(),
            })
        }
        opcode => Err(HirLoweringError::Unsupported(format!(
            "callable expression {opcode:?}"
        ))),
    }
}

fn function_tag_literal(tag: i64, span: Option<wire::SourceSpanProto>) -> wire::HirExpressionProto {
    wire::HirExpressionProto {
        opcode: "literal".to_owned(),
        r#type: Some(wire::TypeProto {
            kind: wire::type_proto::Kind::Scalar.into(),
            name: "Int".to_owned(),
            ..Default::default()
        }),
        attributes: vec![wire::AttributeProto {
            name: "value".to_owned(),
            value: Some(wire::LiteralProto {
                value: Some(wire::literal_proto::Value::IntValue(tag)),
            }),
        }],
        span,
        ..Default::default()
    }
}

fn hir_name_expression(
    name: &str,
    r#type: wire::TypeProto,
    span: Option<wire::SourceSpanProto>,
) -> wire::HirExpressionProto {
    wire::HirExpressionProto {
        opcode: "name".to_owned(),
        r#type: Some(r#type),
        attributes: vec![wire::AttributeProto {
            name: "name".to_owned(),
            value: Some(wire::LiteralProto {
                value: Some(wire::literal_proto::Value::StringValue(name.to_owned())),
            }),
        }],
        span,
        ..Default::default()
    }
}

fn expression_has_function_type(
    expression: Option<&wire::HirExpressionProto>,
    scope: &std::sync::Arc<str>,
    arena: &mut TypeArena,
) -> Result<bool, HirLoweringError> {
    let expression = expression.ok_or_else(|| {
        HirLoweringError::InvalidQCore("assignment expression is missing".to_owned())
    })?;
    let type_id = arena
        .resolve_wire_type(
            expression.r#type.as_ref().ok_or_else(|| {
                HirLoweringError::InvalidQCore("assignment expression has no type".to_owned())
            })?,
            scope,
        )
        .map_err(HirLoweringError::InvalidQCore)?;
    Ok(matches!(
        arena.get(arena.strip_stage(type_id)),
        Some(SemanticType::Function { .. })
    ))
}

fn rewrite_function_alias_names(
    expression: &mut wire::HirExpressionProto,
    aliases: &HashMap<String, FunctionAliasBinding>,
    catalog: &FunctionAliasCatalog,
    expanded_nodes: &mut usize,
    max_expanded_nodes: usize,
    depth: usize,
) -> Result<(), HirLoweringError> {
    if depth > budget::MAX_NESTING {
        return Err(HirLoweringError::Unsupported(
            "function alias expansion depth".to_owned(),
        ));
    }
    if expression.opcode == "name" {
        let name = string_attribute(&expression.attributes, "name")?;
        if let Some(alias) = aliases.get(name) {
            let replacement =
                function_alias_dispatch(name, alias, catalog, expression.span.clone())?;
            let added = expression_node_count(&replacement)?;
            *expanded_nodes = expanded_nodes.checked_add(added).ok_or_else(|| {
                HirLoweringError::Unsupported("function alias expansion overflow".to_owned())
            })?;
            if *expanded_nodes > max_expanded_nodes {
                return Err(HirLoweringError::Unsupported(format!(
                    "function alias expansion exceeds {max_expanded_nodes} nodes"
                )));
            }
            *expression = replacement;
            return Ok(());
        }
    }
    for operand in &mut expression.operands {
        rewrite_function_alias_names(
            operand,
            aliases,
            catalog,
            expanded_nodes,
            max_expanded_nodes,
            depth + 1,
        )?;
    }
    for region in &mut expression.regions {
        rewrite_function_alias_names(
            region,
            aliases,
            catalog,
            expanded_nodes,
            max_expanded_nodes,
            depth + 1,
        )?;
    }
    Ok(())
}

fn function_alias_dispatch(
    name: &str,
    binding: &FunctionAliasBinding,
    catalog: &FunctionAliasCatalog,
    span: Option<wire::SourceSpanProto>,
) -> Result<wire::HirExpressionProto, HirLoweringError> {
    if binding.candidates.is_empty() || binding.candidates.len() > MAX_FUNCTION_VALUE_CANDIDATES {
        return Err(HirLoweringError::InvalidQCore(format!(
            "function alias {name:?} has an invalid candidate count"
        )));
    }
    build_function_dispatch(
        &function_tag_name(name),
        binding,
        &binding.candidates,
        catalog,
        span,
    )
}

fn build_function_dispatch(
    tag_name: &str,
    binding: &FunctionAliasBinding,
    candidates: &[String],
    catalog: &FunctionAliasCatalog,
    span: Option<wire::SourceSpanProto>,
) -> Result<wire::HirExpressionProto, HirLoweringError> {
    if candidates.is_empty() {
        return Err(HirLoweringError::InvalidQCore(
            "selected callable has no candidate".to_owned(),
        ));
    }
    if let [symbol] = candidates {
        return catalog.references.get(symbol).cloned().ok_or_else(|| {
            HirLoweringError::InvalidQCore(format!(
                "function candidate {symbol:?} has no reference"
            ))
        });
    }
    let middle = candidates.len() / 2;
    let (left, right) = candidates.split_at(middle);
    let pivot = catalog.tags.get(&right[0]).copied().ok_or_else(|| {
        HirLoweringError::InvalidQCore(format!("function candidate {:?} has no tag", right[0]))
    })?;
    let tag = hir_name_expression(tag_name, binding.tag_type.clone(), span.clone());
    let condition = wire::HirExpressionProto {
        opcode: "classical.compare_chain".to_owned(),
        r#type: Some(scalar_at_tag_stage(&binding.tag_type, "Bool")),
        operands: vec![tag, function_tag_literal(pivot, span.clone())],
        attributes: vec![string_tuple_qcore_attribute(
            "operators",
            &["lt".to_owned()],
        )],
        span: span.clone(),
        ..Default::default()
    };
    Ok(wire::HirExpressionProto {
        opcode: "lazy_if".to_owned(),
        r#type: Some(binding.function_type.clone()),
        operands: vec![condition],
        attributes: vec![string_tuple_qcore_attribute(
            "captures",
            &[tag_name.to_owned()],
        )],
        regions: vec![
            build_function_dispatch(tag_name, binding, left, catalog, span.clone())?,
            build_function_dispatch(tag_name, binding, right, catalog, span.clone())?,
        ],
        span,
    })
}

fn expression_node_count(expression: &wire::HirExpressionProto) -> Result<usize, HirLoweringError> {
    let mut pending = vec![expression];
    let mut count = 0_usize;
    while let Some(current) = pending.pop() {
        count = count.checked_add(1).ok_or_else(|| {
            HirLoweringError::Unsupported("function alias node count overflow".to_owned())
        })?;
        pending.extend(current.operands.iter());
        pending.extend(current.regions.iter());
    }
    Ok(count)
}

fn expression_names(expression: &wire::HirExpressionProto) -> HashSet<String> {
    let mut names = HashSet::new();
    let mut pending = vec![expression];
    while let Some(current) = pending.pop() {
        if current.opcode == "name"
            && let Ok(name) = string_attribute(&current.attributes, "name")
        {
            names.insert(name.to_owned());
        }
        pending.extend(current.operands.iter());
        pending.extend(current.regions.iter());
    }
    names
}

fn normalize_return_continuations(
    source: &wire::HirBlockProto,
    work: &mut HirNormalizationWork,
) -> Result<wire::HirBlockProto, HirLoweringError> {
    let mut block = source.clone();
    if hir_block_guarantees_return(&block, work)? {
        normalize_return_block(&mut block, work)?;
    }
    Ok(block)
}

fn normalize_return_block(
    block: &mut wire::HirBlockProto,
    work: &mut HirNormalizationWork,
) -> Result<(), HirLoweringError> {
    for statement in &mut block.statements {
        work.visit_return_continuation();
        for region in &mut statement.regions {
            if hir_block_guarantees_return(region, work)? {
                normalize_return_block(region, work)?;
            }
        }
    }
    for position in (0..block.statements.len().saturating_sub(1)).rev() {
        work.visit_return_continuation();
        let Some(branch) = partial_return_arm(&block.statements[position], work)? else {
            continue;
        };
        let continuation = block.statements.split_off(position + 1);
        let statement = &mut block.statements[position];
        append_return_continuation(&mut statement.regions[branch], continuation, work)?;
        clear_if_outputs(statement)?;
    }
    Ok(())
}

fn append_return_continuation(
    block: &mut wire::HirBlockProto,
    continuation: Vec<wire::HirStatementProto>,
    work: &mut HirNormalizationWork,
) -> Result<(), HirLoweringError> {
    block.statements.extend(continuation);
    normalize_return_block(block, work)
}

fn clear_if_outputs(statement: &mut wire::HirStatementProto) -> Result<(), HirLoweringError> {
    statement.result_types.clear();
    let outputs = statement
        .attributes
        .iter_mut()
        .find(|attribute| attribute.name == "outputs")
        .and_then(|attribute| attribute.value.as_mut())
        .ok_or_else(|| HirLoweringError::InvalidQCore("if outputs attribute".to_owned()))?;
    outputs.value = Some(wire::literal_proto::Value::TupleValue(
        wire::literal_proto::Sequence::default(),
    ));
    Ok(())
}

fn partial_return_arm(
    statement: &wire::HirStatementProto,
    work: &mut HirNormalizationWork,
) -> Result<Option<usize>, HirLoweringError> {
    work.visit_return_continuation();
    if statement.opcode != "if" || statement.regions.len() != 2 {
        return Ok(None);
    }
    let returns = statement
        .regions
        .iter()
        .map(|region| hir_block_guarantees_return(region, work))
        .collect::<Result<Vec<_>, _>>()?;
    let falls_through = statement
        .regions
        .iter()
        .map(|region| hir_block_falls_through(region, work))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(match (returns.as_slice(), falls_through.as_slice()) {
        ([true, false], [false, true]) => Some(1),
        ([false, true], [true, false]) => Some(0),
        _ => None,
    })
}

fn hir_block_guarantees_return(
    block: &wire::HirBlockProto,
    work: &mut HirNormalizationWork,
) -> Result<bool, HirLoweringError> {
    work.visit_return_continuation();
    let Some(statement) = block.statements.last() else {
        return Ok(false);
    };
    match statement.opcode.as_str() {
        "return" => Ok(true),
        "if" => statement
            .regions
            .iter()
            .map(|region| hir_block_guarantees_return(region, work))
            .try_fold(true, |all, returns| returns.map(|returns| all && returns)),
        "for" | "while" => bool_attribute(&statement.attributes, "all_paths_return"),
        _ => Ok(false),
    }
}

fn hir_block_falls_through(
    block: &wire::HirBlockProto,
    work: &mut HirNormalizationWork,
) -> Result<bool, HirLoweringError> {
    work.visit_return_continuation();
    let Some(statement) = block.statements.last() else {
        return Ok(true);
    };
    match statement.opcode.as_str() {
        "return" | "break" | "continue" => Ok(false),
        "if" => statement
            .regions
            .iter()
            .map(|region| hir_block_falls_through(region, work))
            .try_fold(false, |any, falls| falls.map(|falls| any || falls)),
        "for" | "while" => {
            bool_attribute(&statement.attributes, "all_paths_return").map(|returns| !returns)
        }
        _ => Ok(true),
    }
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn lower_straight_body(
    body: &wire::HirBlockProto,
    scope: &std::sync::Arc<str>,
    arena: &mut TypeArena,
    next_id: &mut u64,
    next_region: &mut u64,
    region_children: &mut RegionChildren,
    environment: &mut HashMap<String, wire::ValueProto>,
    work: &mut HirSsaLoweringWork,
) -> Result<wire::BlockProto, HirLoweringError> {
    let mut operations = Vec::new();
    let mut terminator = None;
    for statement in &body.statements {
        work.visit_statement();
        let expression = statement.expression.as_ref();
        match statement.opcode.as_str() {
            "assign" => lower_assignment(
                statement,
                scope,
                arena,
                next_id,
                next_region,
                region_children,
                environment,
                &mut operations,
                work,
            )?,
            "expression" => {
                lower_expression(
                    expression.expect("validated expression statement"),
                    scope,
                    arena,
                    next_id,
                    next_region,
                    region_children,
                    environment,
                    &mut operations,
                    work,
                )?;
            }
            "return" => {
                let value = lower_expression(
                    expression.expect("validated return expression"),
                    scope,
                    arena,
                    next_id,
                    next_region,
                    region_children,
                    environment,
                    &mut operations,
                    work,
                )?;
                terminator = Some(wire::TerminatorProto {
                    kind: "return".to_owned(),
                    operands: vec![value.id],
                    span: statement.span.clone(),
                    ..Default::default()
                });
            }
            "if" => {
                terminator = lower_structured_statement(
                    lower_if_statement,
                    statement,
                    scope,
                    arena,
                    next_id,
                    next_region,
                    region_children,
                    environment,
                    &mut operations,
                    None,
                    work,
                )?;
            }
            "for" => {
                terminator = lower_structured_statement(
                    lower_for_statement,
                    statement,
                    scope,
                    arena,
                    next_id,
                    next_region,
                    region_children,
                    environment,
                    &mut operations,
                    None,
                    work,
                )?;
            }
            "while" => {
                terminator = lower_structured_statement(
                    lower_while_statement,
                    statement,
                    scope,
                    arena,
                    next_id,
                    next_region,
                    region_children,
                    environment,
                    &mut operations,
                    None,
                    work,
                )?;
            }
            opcode => {
                return Err(HirLoweringError::Unsupported(format!(
                    "statement {opcode:?}"
                )));
            }
        }
        if terminator.is_some() {
            break;
        }
    }
    let terminator = terminator.ok_or_else(|| {
        HirLoweringError::Unsupported("straight-line body without return".to_owned())
    })?;
    let operations = schedule_dead_results(operations, &terminator);
    Ok(wire::BlockProto {
        operations,
        terminator: Some(terminator),
        ..Default::default()
    })
}

/// Lower one binding or aggregate destructuring into the current SSA block.
#[allow(clippy::too_many_arguments)]
fn lower_assignment(
    statement: &wire::HirStatementProto,
    scope: &std::sync::Arc<str>,
    arena: &mut TypeArena,
    next_id: &mut u64,
    next_region: &mut u64,
    region_children: &mut RegionChildren,
    environment: &mut HashMap<String, wire::ValueProto>,
    operations: &mut Vec<wire::OperationProto>,
    work: &mut HirSsaLoweringWork,
) -> Result<(), HirLoweringError> {
    let value = lower_expression(
        statement
            .expression
            .as_ref()
            .expect("validated assignment expression"),
        scope,
        arena,
        next_id,
        next_region,
        region_children,
        environment,
        operations,
        work,
    )?;
    // An explicit result row denotes unpacking even for a one-element aggregate.
    if statement.result_types.is_empty() {
        environment.insert(statement.targets[0].clone(), value);
        return Ok(());
    }

    let linear_results = statement
        .result_types
        .iter()
        .map(|result_type| {
            arena
                .resolve_wire_type(result_type, scope)
                .map(|type_id| arena.is_quantum(type_id))
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(HirLoweringError::InvalidQCore)?;
    let linear_count = linear_results
        .iter()
        .filter(|is_linear| **is_linear)
        .count();
    if linear_count > 0 && !value.has_region {
        return Err(HirLoweringError::InvalidQCore(
            "quantum product assignment has no owner".to_owned(),
        ));
    }
    if linear_count > 1 {
        let parent = value.region.as_ref().expect("linear product has an owner");
        let children = linear_results
            .iter()
            .enumerate()
            .filter(|(_, is_linear)| **is_linear)
            .map(|(index, _)| index.to_string())
            .collect();
        region_children.insert((parent.root, parent.projections.clone()), children);
    }

    let mut results = Vec::with_capacity(statement.targets.len());
    for (index, ((name, result_type), is_linear)) in statement
        .targets
        .iter()
        .zip(&statement.result_types)
        .zip(linear_results)
        .enumerate()
    {
        let region = if is_linear {
            let mut region = value.region.clone().expect("linear product has an owner");
            if linear_count > 1 {
                region.projections.push(index.to_string());
            }
            Some(region)
        } else {
            None
        };
        let result = new_value(result_type.clone(), name.clone(), region, next_id)?;
        environment.insert(name.clone(), result.clone());
        results.push(result);
    }
    operations.push(wire::OperationProto {
        opcode: "qcore.unpack".to_owned(),
        operands: vec![value.id],
        results,
        span: statement.span.clone(),
        ..Default::default()
    });
    Ok(())
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn lower_if_statement(
    statement: &wire::HirStatementProto,
    scope: &std::sync::Arc<str>,
    arena: &mut TypeArena,
    next_id: &mut u64,
    next_region: &mut u64,
    region_children: &mut RegionChildren,
    environment: &mut HashMap<String, wire::ValueProto>,
    operations: &mut Vec<wire::OperationProto>,
    loop_carries: Option<&[String]>,
    work: &mut HirSsaLoweringWork,
) -> Result<Option<wire::TerminatorProto>, HirLoweringError> {
    let condition = lower_expression(
        statement
            .expression
            .as_ref()
            .expect("validated if condition"),
        scope,
        arena,
        next_id,
        next_region,
        region_children,
        environment,
        operations,
        work,
    )?;
    let capture_names = string_tuple_attribute(&statement.attributes, "captures")?;
    let output_names = string_tuple_attribute(&statement.attributes, "outputs")?;
    if output_names.len() != statement.result_types.len() {
        return Err(HirLoweringError::InvalidQCore(
            "if output name/type arity mismatch".to_owned(),
        ));
    }
    let capture_values = capture_names
        .iter()
        .map(|name| {
            environment.get(name).cloned().ok_or_else(|| {
                HirLoweringError::InvalidQCore(format!("unresolved if capture {name:?}"))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut regions = Vec::with_capacity(2);
    let mut branch_outputs = Vec::with_capacity(2);
    for source in &statement.regions {
        let arguments = capture_values
            .iter()
            .zip(&capture_names)
            .map(|(value, name)| clone_region_argument(value, name, next_id))
            .collect::<Result<Vec<_>, _>>()?;
        let mut branch_environment = capture_names
            .iter()
            .cloned()
            .zip(arguments.iter().cloned())
            .collect();
        let (block, yielded) = lower_yielding_block(
            source,
            &arguments,
            &output_names,
            &statement.result_types,
            statement.span.clone(),
            scope,
            arena,
            next_id,
            next_region,
            region_children,
            &mut branch_environment,
            loop_carries,
            work,
        )?;
        let normally_yields = block
            .terminator
            .as_ref()
            .is_some_and(|terminator| terminator.kind == "yield");
        regions.push(wire::RegionProto {
            blocks: vec![block],
        });
        if normally_yields {
            branch_outputs.push(yielded);
        }
    }
    let results = match branch_outputs.first() {
        Some(first_outputs) => output_names
            .iter()
            .zip(&statement.result_types)
            .zip(first_outputs)
            .map(|((name, r#type), yielded)| {
                new_value(
                    r#type.clone(),
                    name.clone(),
                    yielded.region.clone(),
                    next_id,
                )
            })
            .collect::<Result<Vec<_>, _>>()?,
        None => Vec::new(),
    };
    let condition_type = condition
        .r#type
        .as_ref()
        .ok_or_else(|| HirLoweringError::InvalidQCore("if condition has no type".to_owned()))?;
    let condition_id = arena
        .resolve_wire_type(condition_type, scope)
        .map_err(HirLoweringError::InvalidQCore)?;
    let mut operand_ids = Vec::with_capacity(capture_values.len() + 1);
    operand_ids.push(condition.id);
    operand_ids.extend(capture_values.iter().map(|value| value.id));
    operations.push(wire::OperationProto {
        opcode: "qcore.if".to_owned(),
        operands: operand_ids,
        results: results.clone(),
        attributes: structured_if_attributes(
            &capture_names,
            &output_names,
            control_stage(arena.stage_of(condition_id)),
        ),
        regions,
        span: statement.span.clone(),
        intent: Vec::new(),
    });
    if branch_outputs.is_empty() {
        return Ok(Some(wire::TerminatorProto {
            kind: "unreachable".to_owned(),
            span: statement.span.clone(),
            ..Default::default()
        }));
    }
    replace_source_environment(environment, output_names.into_iter().zip(results));
    Ok(None)
}

/// Retain a while loop as condition, body, and normal-exit regions.
#[expect(
    clippy::too_many_lines,
    reason = "Keep the three while region signatures together"
)]
#[allow(clippy::too_many_arguments)]
fn lower_while_statement(
    statement: &wire::HirStatementProto,
    scope: &std::sync::Arc<str>,
    arena: &mut TypeArena,
    next_id: &mut u64,
    next_region: &mut u64,
    region_children: &mut RegionChildren,
    environment: &mut HashMap<String, wire::ValueProto>,
    operations: &mut Vec<wire::OperationProto>,
    outer_loop: Option<&[String]>,
    work: &mut HirSsaLoweringWork,
) -> Result<Option<wire::TerminatorProto>, HirLoweringError> {
    let all_paths_return = bool_attribute(&statement.attributes, "all_paths_return")?;
    let names = string_tuple_attribute(&statement.attributes, "captures")?;
    if names.len() != statement.result_types.len() {
        return Err(HirLoweringError::InvalidQCore(
            "while carry arity".to_owned(),
        ));
    }
    let initial = names
        .iter()
        .zip(&statement.result_types)
        .map(|(name, expected)| {
            let value = environment.get(name).cloned().ok_or_else(|| {
                HirLoweringError::InvalidQCore(format!("unresolved while carry {name:?}"))
            })?;
            lift_branch_value(
                value,
                expected,
                name,
                statement.span.clone(),
                next_id,
                operations,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let arguments = initial
        .iter()
        .zip(&names)
        .map(|(value, name)| clone_region_argument(value, name, next_id))
        .collect::<Result<Vec<_>, _>>()?;
    let local = names
        .iter()
        .cloned()
        .zip(arguments.iter().cloned())
        .collect();
    let mut condition_operations = Vec::new();
    let condition = lower_expression(
        statement
            .expression
            .as_ref()
            .expect("validated while condition"),
        scope,
        arena,
        next_id,
        next_region,
        region_children,
        &local,
        &mut condition_operations,
        work,
    )?;
    let condition_type = arena
        .resolve_wire_type(condition.r#type.as_ref().expect("condition type"), scope)
        .map_err(HirLoweringError::InvalidQCore)?;
    let stage = control_stage(arena.stage_of(condition_type));
    let condition_block = wire::BlockProto {
        arguments,
        operations: condition_operations,
        terminator: Some(wire::TerminatorProto {
            kind: "yield".to_owned(),
            operands: std::iter::once(condition.id)
                .chain(names.iter().map(|name| local[name].id))
                .collect(),
            span: statement.span.clone(),
            ..Default::default()
        }),
    };
    let mut regions = vec![wire::RegionProto {
        blocks: vec![condition_block],
    }];
    for (index, source) in statement.regions.iter().enumerate() {
        let arguments = initial
            .iter()
            .zip(&names)
            .map(|(value, name)| clone_region_argument(value, name, next_id))
            .collect::<Result<Vec<_>, _>>()?;
        let mut local = names
            .iter()
            .cloned()
            .zip(arguments.iter().cloned())
            .collect();
        let (block, _) = lower_yielding_block(
            source,
            &arguments,
            &names,
            &statement.result_types,
            statement.span.clone(),
            scope,
            arena,
            next_id,
            next_region,
            region_children,
            &mut local,
            if index == 0 { Some(&names) } else { outer_loop },
            work,
        )?;
        regions.push(wire::RegionProto {
            blocks: vec![block],
        });
    }
    let results = initial
        .iter()
        .zip(&names)
        .map(|(value, name)| clone_region_argument(value, name, next_id))
        .collect::<Result<Vec<_>, _>>()?;
    operations.push(wire::OperationProto {
        opcode: "qcore.while".to_owned(),
        operands: initial.iter().map(|value| value.id).collect(),
        results: if all_paths_return {
            Vec::new()
        } else {
            results.clone()
        },
        attributes: structured_if_attributes(&names, &names, stage)
            .into_iter()
            .filter(|attribute| matches!(attribute.name.as_str(), "capture_names" | "stage"))
            .collect(),
        regions,
        span: statement.span.clone(),
        intent: Vec::new(),
    });
    replace_source_environment(environment, names.into_iter().zip(results));
    Ok(all_paths_return.then(|| wire::TerminatorProto {
        kind: "unreachable".to_owned(),
        span: statement.span.clone(),
        ..Default::default()
    }))
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn lower_for_statement(
    statement: &wire::HirStatementProto,
    scope: &std::sync::Arc<str>,
    arena: &mut TypeArena,
    next_id: &mut u64,
    next_region: &mut u64,
    region_children: &mut RegionChildren,
    environment: &mut HashMap<String, wire::ValueProto>,
    operations: &mut Vec<wire::OperationProto>,
    outer_loop: Option<&[String]>,
    work: &mut HirSsaLoweringWork,
) -> Result<Option<wire::TerminatorProto>, HirLoweringError> {
    let all_paths_return = bool_attribute(&statement.attributes, "all_paths_return")?;
    let iterable = statement
        .expression
        .as_ref()
        .expect("validated for iterable");
    let mut bounds = Vec::with_capacity(3);
    let mut collections = Vec::new();
    let mut hidden_values = Vec::new();
    let binding = match iterable.opcode.as_str() {
        "range" if iterable.operands.len() == 3 => {
            for expression in &iterable.operands {
                bounds.push(lower_expression(
                    expression,
                    scope,
                    arena,
                    next_id,
                    next_region,
                    region_children,
                    environment,
                    operations,
                    work,
                )?);
            }
            IterationBinding::Range
        }
        "iterate" if iterable.operands.len() == 1 => {
            let value = lower_expression(
                &iterable.operands[0],
                scope,
                arena,
                next_id,
                next_region,
                region_children,
                environment,
                operations,
                work,
            )?;
            let (extent, item_type) = aggregate_iteration_type(&value)?;
            append_aggregate_bounds(
                extent,
                &value,
                statement,
                environment,
                next_id,
                operations,
                &mut bounds,
            )?;
            collections.push((value.clone(), item_type));
            hidden_values.push(value);
            IterationBinding::Iterate
        }
        "map_items" if iterable.operands.len() == 1 => {
            let value = lower_expression(
                &iterable.operands[0],
                scope,
                arena,
                next_id,
                next_region,
                region_children,
                environment,
                operations,
                work,
            )?;
            let (extent, item_type) = aggregate_iteration_type(&value)?;
            append_aggregate_bounds(
                extent,
                &value,
                statement,
                environment,
                next_id,
                operations,
                &mut bounds,
            )?;
            collections.push((value.clone(), item_type));
            hidden_values.push(value);
            IterationBinding::MapItems
        }
        "enumerate" if iterable.operands.len() == 2 => {
            let value = lower_expression(
                &iterable.operands[0],
                scope,
                arena,
                next_id,
                next_region,
                region_children,
                environment,
                operations,
                work,
            )?;
            let start = lower_expression(
                &iterable.operands[1],
                scope,
                arena,
                next_id,
                next_region,
                region_children,
                environment,
                operations,
                work,
            )?;
            let (extent, item_type) = aggregate_iteration_type(&value)?;
            append_aggregate_bounds(
                extent,
                &value,
                statement,
                environment,
                next_id,
                operations,
                &mut bounds,
            )?;
            collections.push((value.clone(), item_type));
            hidden_values.push(value);
            let start_hidden_index = hidden_values.len();
            hidden_values.push(start);
            IterationBinding::Enumerate { start_hidden_index }
        }
        "zip" if !iterable.operands.is_empty() => {
            let mut shared_extent = None;
            for expression in &iterable.operands {
                let value = lower_expression(
                    expression,
                    scope,
                    arena,
                    next_id,
                    next_region,
                    region_children,
                    environment,
                    operations,
                    work,
                )?;
                let (extent, item_type) = aggregate_iteration_type(&value)?;
                if let Some(expected) = &shared_extent
                    && expected != &extent
                {
                    return Err(HirLoweringError::InvalidQCore(
                        "zip inputs have different leading extents".to_owned(),
                    ));
                }
                shared_extent.get_or_insert_with(|| extent.clone());
                collections.push((value.clone(), item_type));
                hidden_values.push(value);
            }
            append_aggregate_bounds(
                shared_extent.expect("nonempty zip has one extent"),
                &collections[0].0,
                statement,
                environment,
                next_id,
                operations,
                &mut bounds,
            )?;
            IterationBinding::Zip
        }
        _ => {
            return Err(HirLoweringError::Unsupported(
                "for iterable outside range, enumerate, zip, Map.items(), or a fixed classical aggregate"
                    .to_owned(),
            ));
        }
    };
    let target_types = iterable_target_types(iterable)?;
    let expected_target_count = match binding {
        IterationBinding::Range | IterationBinding::Iterate => 1,
        IterationBinding::MapItems | IterationBinding::Enumerate { .. } => 2,
        IterationBinding::Zip => collections.len(),
    };
    if statement.targets.len() != expected_target_count
        || target_types.len() != expected_target_count
    {
        return Err(HirLoweringError::InvalidQCore(
            "for iterable target/type arity mismatch".to_owned(),
        ));
    }
    let mut carry_names = string_tuple_attribute(&statement.attributes, "captures")?;
    let declared_carry_count = carry_names.len();
    let mut carry_types = statement.result_types.clone();
    if declared_carry_count != carry_types.len() {
        return Err(HirLoweringError::InvalidQCore(
            "for carry name/type arity mismatch".to_owned(),
        ));
    }
    for (index, value) in hidden_values.iter().enumerate() {
        let hidden = format!("{INTERNAL_NAME_PREFIX}iterable:{index}:{}", value.id);
        carry_names.push(hidden);
        carry_types.push(value.r#type.clone().ok_or_else(|| {
            HirLoweringError::InvalidQCore("for hidden iterable value has no type".to_owned())
        })?);
    }
    let mut initial = Vec::with_capacity(carry_names.len());
    for (index, (name, expected_type)) in carry_names.iter().zip(&carry_types).enumerate() {
        if index >= declared_carry_count {
            initial.push(hidden_values[index - declared_carry_count].clone());
            continue;
        }
        let current = environment.get(name).cloned().ok_or_else(|| {
            HirLoweringError::InvalidQCore(format!("unresolved for carry {name:?}"))
        })?;
        initial.push(lift_branch_value(
            current,
            expected_type,
            name,
            statement.span.clone(),
            next_id,
            operations,
        )?);
    }
    let (stage, induction_type) = loop_stage_and_induction_type(&bounds, scope, arena)?;
    let induction_name = statement.targets.first().ok_or_else(|| {
        HirLoweringError::InvalidQCore("for loop has no induction target".to_owned())
    })?;
    let induction = new_value(induction_type, induction_name.clone(), None, next_id)?;
    let mut body_arguments = Vec::with_capacity(initial.len() + 1);
    body_arguments.push(induction.clone());
    for (value, name) in initial.iter().zip(&carry_names) {
        body_arguments.push(clone_region_argument(value, name, next_id)?);
    }
    let mut body_environment = carry_names
        .iter()
        .cloned()
        .zip(body_arguments.iter().skip(1).cloned())
        .collect::<HashMap<_, _>>();
    let hidden_names = &carry_names[declared_carry_count..];
    let mut prefix_operations = Vec::new();
    match binding {
        IterationBinding::Range => {
            body_environment.insert(induction_name.clone(), induction.clone());
        }
        IterationBinding::Iterate => {
            let collection = body_environment[&hidden_names[0]].clone();
            bind_indexed_iteration_target(
                &statement.targets[0],
                &collections[0].1,
                &collection,
                &induction,
                statement,
                next_id,
                &mut body_environment,
                &mut prefix_operations,
            )?;
        }
        IterationBinding::MapItems => {
            let collection = body_environment[&hidden_names[0]].clone();
            let entry = new_value(
                collections[0].1.clone(),
                "mapping_entry".to_owned(),
                None,
                next_id,
            )?;
            prefix_operations.push(wire::OperationProto {
                opcode: "classical.index".to_owned(),
                operands: vec![collection.id, induction.id],
                results: vec![entry.clone()],
                span: statement.span.clone(),
                ..Default::default()
            });
            let mut projected = Vec::with_capacity(2);
            for (target, item_type) in statement.targets.iter().zip(&target_types) {
                let value = new_value(item_type.clone(), target.clone(), None, next_id)?;
                body_environment.insert(target.clone(), value.clone());
                projected.push(value);
            }
            prefix_operations.push(wire::OperationProto {
                opcode: "qcore.unpack".to_owned(),
                operands: vec![entry.id],
                results: projected,
                span: statement.span.clone(),
                ..Default::default()
            });
        }
        IterationBinding::Enumerate { start_hidden_index } => {
            let index = new_value(
                target_types[0].clone(),
                statement.targets[0].clone(),
                None,
                next_id,
            )?;
            prefix_operations.push(wire::OperationProto {
                opcode: "classical.add".to_owned(),
                operands: vec![
                    induction.id,
                    body_environment[&hidden_names[start_hidden_index]].id,
                ],
                results: vec![index.clone()],
                span: statement.span.clone(),
                ..Default::default()
            });
            body_environment.insert(statement.targets[0].clone(), index);
            let collection = body_environment[&hidden_names[0]].clone();
            bind_indexed_iteration_target(
                &statement.targets[1],
                &collections[0].1,
                &collection,
                &induction,
                statement,
                next_id,
                &mut body_environment,
                &mut prefix_operations,
            )?;
        }
        IterationBinding::Zip => {
            for (index, (target, (_, item_type))) in
                statement.targets.iter().zip(&collections).enumerate()
            {
                let collection = body_environment[&hidden_names[index]].clone();
                bind_indexed_iteration_target(
                    target,
                    item_type,
                    &collection,
                    &induction,
                    statement,
                    next_id,
                    &mut body_environment,
                    &mut prefix_operations,
                )?;
            }
        }
    }
    let (mut body, _) = lower_yielding_block(
        &statement.regions[0],
        &body_arguments,
        &carry_names,
        &carry_types,
        statement.span.clone(),
        scope,
        arena,
        next_id,
        next_region,
        region_children,
        &mut body_environment,
        Some(&carry_names),
        work,
    )?;
    body.operations.splice(0..0, prefix_operations);
    let results = carry_names
        .iter()
        .zip(&carry_types)
        .zip(&initial)
        .map(|((name, r#type), value)| {
            new_value(r#type.clone(), name.clone(), value.region.clone(), next_id)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let else_arguments = results
        .iter()
        .zip(&carry_names)
        .map(|(value, name)| clone_region_argument(value, name, next_id))
        .collect::<Result<Vec<_>, _>>()?;
    let mut else_environment = carry_names
        .iter()
        .cloned()
        .zip(else_arguments.iter().cloned())
        .collect();
    let (otherwise, _) = lower_yielding_block(
        &statement.regions[1],
        &else_arguments,
        &carry_names,
        &carry_types,
        statement.span.clone(),
        scope,
        arena,
        next_id,
        next_region,
        region_children,
        &mut else_environment,
        outer_loop,
        work,
    )?;
    let result_free =
        all_paths_return && block_exits_callable(&body) && block_exits_callable(&otherwise);
    let mut operand_ids = Vec::with_capacity(bounds.len() + initial.len());
    operand_ids.extend(bounds.iter().map(|value| value.id));
    operand_ids.extend(initial.iter().map(|value| value.id));
    operations.push(wire::OperationProto {
        opcode: "qcore.for".to_owned(),
        operands: operand_ids,
        results: if result_free {
            Vec::new()
        } else {
            results.clone()
        },
        attributes: structured_for_attributes(&carry_names, induction_name, stage),
        regions: vec![
            wire::RegionProto { blocks: vec![body] },
            wire::RegionProto {
                blocks: vec![otherwise],
            },
        ],
        span: statement.span.clone(),
        intent: Vec::new(),
    });
    if !all_paths_return {
        replace_source_environment(
            environment,
            carry_names
                .into_iter()
                .take(declared_carry_count)
                .zip(results),
        );
    }
    Ok(all_paths_return.then(|| wire::TerminatorProto {
        kind: "unreachable".to_owned(),
        span: statement.span.clone(),
        ..Default::default()
    }))
}

fn append_aggregate_bounds(
    extent: AggregateExtent,
    source: &wire::ValueProto,
    statement: &wire::HirStatementProto,
    environment: &HashMap<String, wire::ValueProto>,
    next_id: &mut u64,
    operations: &mut Vec<wire::OperationProto>,
    bounds: &mut Vec<wire::ValueProto>,
) -> Result<(), HirLoweringError> {
    bounds.push(integer_constant(
        0,
        statement.span.clone(),
        next_id,
        operations,
    )?);
    bounds.push(match extent {
        AggregateExtent::Fixed(value) => {
            integer_constant(value, statement.span.clone(), next_id, operations)?
        }
        AggregateExtent::Symbol(name) => {
            if let Some(value) = environment.get(&name) {
                value.clone()
            } else {
                let source_type = source.r#type.as_ref().ok_or_else(|| {
                    HirLoweringError::InvalidQCore("iterable source type".to_owned())
                })?;
                let shape_only = wire::LiteralProto {
                    value: Some(wire::literal_proto::Value::BoolValue(true)),
                };
                let attributes = vec![wire::AttributeProto {
                    name: "shape_only".to_owned(),
                    value: Some(shape_only.clone()),
                }];
                let encoded = crate::operations::infer_classical_type(
                    "classical.len",
                    &[source_type.encode_to_vec()],
                    &[("shape_only".to_owned(), shape_only.encode_to_vec())],
                )
                .map_err(|error| HirLoweringError::InvalidQCore(error.to_string()))?;
                let result_type = wire::TypeProto::decode(encoded.as_slice())
                    .map_err(|error| HirLoweringError::InvalidQCore(error.to_string()))?;
                let value = new_value(result_type, "iterable_extent".to_owned(), None, next_id)?;
                operations.push(wire::OperationProto {
                    opcode: "classical.len".to_owned(),
                    operands: vec![source.id],
                    results: vec![value.clone()],
                    attributes,
                    span: statement.span.clone(),
                    ..Default::default()
                });
                value
            }
        }
    });
    bounds.push(integer_constant(
        1,
        statement.span.clone(),
        next_id,
        operations,
    )?);
    Ok(())
}

fn iterable_target_types(
    iterable: &wire::HirExpressionProto,
) -> Result<Vec<wire::TypeProto>, HirLoweringError> {
    let r#type = iterable
        .r#type
        .as_ref()
        .ok_or_else(|| HirLoweringError::InvalidQCore("for iterable has no type".to_owned()))?;
    if r#type.kind != wire::type_proto::Kind::Record as i32
        || !matches!(r#type.name.as_str(), "Iterable" | "Range")
    {
        return Err(HirLoweringError::InvalidQCore(
            "for iterable has no target type row".to_owned(),
        ));
    }
    Ok(r#type.arguments.clone())
}

#[allow(clippy::too_many_arguments)]
fn bind_indexed_iteration_target(
    target: &str,
    item_type: &wire::TypeProto,
    collection: &wire::ValueProto,
    induction: &wire::ValueProto,
    statement: &wire::HirStatementProto,
    next_id: &mut u64,
    environment: &mut HashMap<String, wire::ValueProto>,
    operations: &mut Vec<wire::OperationProto>,
) -> Result<(), HirLoweringError> {
    let element = new_value(item_type.clone(), target.to_owned(), None, next_id)?;
    operations.push(wire::OperationProto {
        opcode: "classical.index".to_owned(),
        operands: vec![collection.id, induction.id],
        results: vec![element.clone()],
        span: statement.span.clone(),
        ..Default::default()
    });
    environment.insert(target.to_owned(), element);
    Ok(())
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn block_exits_callable(block: &wire::BlockProto) -> bool {
    block
        .terminator
        .as_ref()
        .is_some_and(|terminator| matches!(terminator.kind.as_str(), "return" | "unreachable"))
}

/// Replace source-visible bindings while retaining enclosing lowering-only carries.
fn replace_source_environment(
    environment: &mut HashMap<String, wire::ValueProto>,
    source_values: impl IntoIterator<Item = (String, wire::ValueProto)>,
) {
    environment.retain(|name, _| name.starts_with(INTERNAL_NAME_PREFIX));
    environment.extend(source_values);
}

fn loop_stage_and_induction_type(
    bounds: &[wire::ValueProto],
    scope: &std::sync::Arc<str>,
    arena: &mut TypeArena,
) -> Result<(&'static str, wire::TypeProto), HirLoweringError> {
    let mut selected: Option<(Stage, wire::TypeProto)> = None;
    let mut value_type = None;
    for bound in bounds {
        let r#type = bound
            .r#type
            .as_ref()
            .ok_or_else(|| HirLoweringError::InvalidQCore("for bound has no type".to_owned()))?;
        value_type.get_or_insert_with(|| r#type.clone());
        let type_id = arena
            .resolve_wire_type(r#type, scope)
            .map_err(HirLoweringError::InvalidQCore)?;
        if let Some(stage) = arena.stage_of(type_id)
            && selected
                .as_ref()
                .is_none_or(|(current, _)| stage > *current)
        {
            selected = Some((stage, r#type.clone()));
        }
    }
    match selected {
        Some((stage, r#type)) => Ok((control_stage(Some(stage)), r#type)),
        None => Ok((
            "Value",
            value_type.ok_or_else(|| {
                HirLoweringError::InvalidQCore("for loop has no bounds".to_owned())
            })?,
        )),
    }
}

fn aggregate_iteration_type(
    collection: &wire::ValueProto,
) -> Result<(AggregateExtent, wire::TypeProto), HirLoweringError> {
    if collection.has_region {
        return Err(HirLoweringError::Unsupported(
            "owned aggregate iteration".to_owned(),
        ));
    }
    let r#type = collection
        .r#type
        .as_ref()
        .ok_or_else(|| HirLoweringError::InvalidQCore("for collection has no type".to_owned()))?;
    let (aggregate, stage) =
        if r#type.kind == wire::type_proto::Kind::Staged as i32 && r#type.arguments.len() == 1 {
            (&r#type.arguments[0], Some(r#type.name.clone()))
        } else {
            (r#type, None)
        };
    let (extent, element) = if aggregate.kind == wire::type_proto::Kind::Array as i32
        && aggregate.arguments.len() == 1
        && !aggregate.dimensions.is_empty()
        && matches!(aggregate.name.as_str(), "Array" | "Bits" | "UInt")
    {
        let extent = aggregate_iteration_extent(&aggregate.dimensions[0])?;
        let element = if aggregate.name == "Array" && aggregate.dimensions.len() > 1 {
            let mut element = aggregate.clone();
            element.dimensions.remove(0);
            element
        } else {
            aggregate.arguments[0].clone()
        };
        (extent, element)
    } else if matches!(
        wire::type_proto::Kind::try_from(aggregate.kind),
        Ok(wire::type_proto::Kind::Product | wire::type_proto::Kind::Record)
    ) && !aggregate.arguments.is_empty()
    {
        let extent = i64::try_from(aggregate.arguments.len())
            .map(AggregateExtent::Fixed)
            .map_err(|_| HirLoweringError::Unsupported("aggregate iteration extent".to_owned()))?;
        (extent, aggregate.arguments[0].clone())
    } else {
        return Err(HirLoweringError::Unsupported(
            "iteration outside a fixed classical aggregate".to_owned(),
        ));
    };
    Ok((extent, apply_outer_stage(element, stage)))
}

fn aggregate_iteration_extent(
    dimension: &wire::DimensionProto,
) -> Result<AggregateExtent, HirLoweringError> {
    match dimension.value.as_ref() {
        Some(wire::dimension_proto::Value::Fixed(value)) => i64::try_from(*value)
            .map(AggregateExtent::Fixed)
            .map_err(|_| HirLoweringError::Unsupported("aggregate iteration extent".to_owned())),
        Some(wire::dimension_proto::Value::Symbol(value)) => {
            Ok(AggregateExtent::Symbol(value.clone()))
        }
        None => Err(HirLoweringError::InvalidQCore(
            "aggregate iteration dimension has no value".to_owned(),
        )),
    }
}

fn apply_outer_stage(element: wire::TypeProto, stage: Option<String>) -> wire::TypeProto {
    let Some(stage) = stage else {
        return element;
    };
    if element.kind == wire::type_proto::Kind::Staged as i32
        && Stage::from_name(&element.name) >= Stage::from_name(&stage)
    {
        return element;
    }
    let inner = if element.kind == wire::type_proto::Kind::Staged as i32 {
        element
            .arguments
            .into_iter()
            .next()
            .expect("validated staged type has one argument")
    } else {
        element
    };
    wire::TypeProto {
        kind: wire::type_proto::Kind::Staged as i32,
        name: stage,
        arguments: vec![inner],
        ..Default::default()
    }
}

fn integer_constant(
    value: i64,
    span: Option<wire::SourceSpanProto>,
    next_id: &mut u64,
    operations: &mut Vec<wire::OperationProto>,
) -> Result<wire::ValueProto, HirLoweringError> {
    let result = new_value(
        wire::TypeProto {
            kind: wire::type_proto::Kind::Scalar as i32,
            name: "Int".to_owned(),
            ..Default::default()
        },
        "const".to_owned(),
        None,
        next_id,
    )?;
    operations.push(wire::OperationProto {
        opcode: "qcore.constant".to_owned(),
        results: vec![result.clone()],
        attributes: vec![wire::AttributeProto {
            name: "value".to_owned(),
            value: Some(wire::LiteralProto {
                value: Some(wire::literal_proto::Value::IntValue(value)),
            }),
        }],
        span,
        ..Default::default()
    });
    Ok(result)
}

fn aggregate_index_constant(
    position: usize,
    span: Option<wire::SourceSpanProto>,
    next_id: &mut u64,
    operations: &mut Vec<wire::OperationProto>,
) -> Result<wire::ValueProto, HirLoweringError> {
    let result = new_value(
        wire::TypeProto {
            kind: wire::type_proto::Kind::Scalar as i32,
            name: "Int".to_owned(),
            ..Default::default()
        },
        "slice_index".to_owned(),
        None,
        next_id,
    )?;
    let value = i64::try_from(position).map_or_else(
        |_| wire::literal_proto::Value::BigintValue(BigInt::from(position).to_signed_bytes_be()),
        wire::literal_proto::Value::IntValue,
    );
    operations.push(wire::OperationProto {
        opcode: "qcore.constant".to_owned(),
        results: vec![result.clone()],
        attributes: vec![wire::AttributeProto {
            name: "value".to_owned(),
            value: Some(wire::LiteralProto { value: Some(value) }),
        }],
        span,
        ..Default::default()
    });
    Ok(result)
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn lower_yielding_block(
    source: &wire::HirBlockProto,
    arguments: &[wire::ValueProto],
    outputs: &[String],
    output_types: &[wire::TypeProto],
    span: Option<wire::SourceSpanProto>,
    scope: &std::sync::Arc<str>,
    arena: &mut TypeArena,
    next_id: &mut u64,
    next_region: &mut u64,
    region_children: &mut RegionChildren,
    environment: &mut HashMap<String, wire::ValueProto>,
    loop_carries: Option<&[String]>,
    work: &mut HirSsaLoweringWork,
) -> Result<(wire::BlockProto, Vec<wire::ValueProto>), HirLoweringError> {
    let mut operations = Vec::new();
    let mut terminator = None;
    for statement in &source.statements {
        work.visit_statement();
        let expression = statement.expression.as_ref();
        match statement.opcode.as_str() {
            "assign" => lower_assignment(
                statement,
                scope,
                arena,
                next_id,
                next_region,
                region_children,
                environment,
                &mut operations,
                work,
            )?,
            "expression" => {
                lower_expression(
                    expression.expect("validated expression statement"),
                    scope,
                    arena,
                    next_id,
                    next_region,
                    region_children,
                    environment,
                    &mut operations,
                    work,
                )?;
            }
            "return" => {
                let value = lower_expression(
                    expression.expect("validated return expression"),
                    scope,
                    arena,
                    next_id,
                    next_region,
                    region_children,
                    environment,
                    &mut operations,
                    work,
                )?;
                terminator = Some(wire::TerminatorProto {
                    kind: "return".to_owned(),
                    operands: vec![value.id],
                    span: statement.span.clone(),
                    ..Default::default()
                });
            }
            "break" | "continue" => {
                let carries = loop_carries.ok_or_else(|| {
                    HirLoweringError::InvalidQCore(format!(
                        "{} terminator outside loop",
                        statement.opcode
                    ))
                })?;
                terminator = Some(wire::TerminatorProto {
                    kind: statement.opcode.clone(),
                    operands: carries
                        .iter()
                        .map(|name| {
                            environment.get(name).map(|value| value.id).ok_or_else(|| {
                                HirLoweringError::InvalidQCore(format!(
                                    "unresolved loop carry {name:?}"
                                ))
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                    span: statement.span.clone(),
                    ..Default::default()
                });
            }
            "if" => {
                terminator = lower_structured_statement(
                    lower_if_statement,
                    statement,
                    scope,
                    arena,
                    next_id,
                    next_region,
                    region_children,
                    environment,
                    &mut operations,
                    loop_carries,
                    work,
                )?;
            }
            "for" => {
                terminator = lower_structured_statement(
                    lower_for_statement,
                    statement,
                    scope,
                    arena,
                    next_id,
                    next_region,
                    region_children,
                    environment,
                    &mut operations,
                    loop_carries,
                    work,
                )?;
            }
            "while" => {
                terminator = lower_structured_statement(
                    lower_while_statement,
                    statement,
                    scope,
                    arena,
                    next_id,
                    next_region,
                    region_children,
                    environment,
                    &mut operations,
                    loop_carries,
                    work,
                )?;
            }
            opcode => {
                return Err(HirLoweringError::Unsupported(format!(
                    "yielding block statement {opcode:?}"
                )));
            }
        }
        if terminator.is_some() {
            break;
        }
    }
    let mut yielded = Vec::with_capacity(outputs.len());
    let normalize_carries = terminator.as_ref().is_none_or(|value| {
        value.kind == "yield"
            || (matches!(value.kind.as_str(), "break" | "continue")
                && loop_carries.is_some_and(|carries| carries == outputs))
    });
    if normalize_carries {
        for (name, expected_type) in outputs.iter().zip(output_types) {
            let current = environment.get(name).cloned().ok_or_else(|| {
                HirLoweringError::InvalidQCore(format!("unresolved block output {name:?}"))
            })?;
            let output = lift_branch_value(
                current,
                expected_type,
                name,
                span.clone(),
                next_id,
                &mut operations,
            )?;
            environment.insert(name.clone(), output.clone());
            yielded.push(output);
        }
    }
    let terminator = match terminator {
        Some(mut terminator)
            if matches!(terminator.kind.as_str(), "break" | "continue") && normalize_carries =>
        {
            terminator.operands = yielded.iter().map(|value| value.id).collect();
            terminator
        }
        Some(terminator) => terminator,
        None => wire::TerminatorProto {
            kind: "yield".to_owned(),
            operands: yielded.iter().map(|value| value.id).collect(),
            span,
            ..Default::default()
        },
    };
    Ok((
        wire::BlockProto {
            arguments: arguments.to_vec(),
            operations: schedule_dead_results(operations, &terminator),
            terminator: Some(terminator),
        },
        yielded,
    ))
}

type StructuredLowering = fn(
    &wire::HirStatementProto,
    &std::sync::Arc<str>,
    &mut TypeArena,
    &mut u64,
    &mut u64,
    &mut RegionChildren,
    &mut HashMap<String, wire::ValueProto>,
    &mut Vec<wire::OperationProto>,
    Option<&[String]>,
    &mut HirSsaLoweringWork,
) -> Result<Option<wire::TerminatorProto>, HirLoweringError>;

/// Lower one structured statement, regrouping carried owners when required.
///
/// A call that consumes several owners, such as `a, b = helper(a, b)` or
/// `controls, target = qm.control(f, controls, target)`, returns successors
/// that are projections of the call's ownership region. A structured region
/// must yield its carries with their entry topology, so independently owned
/// carries would no longer match. When the lowered region shows exactly that
/// shape, the statement is lowered again after packing those carries into one
/// tuple and unpacking them, which makes them sibling projections of a shared
/// owner that the call then preserves.
#[allow(clippy::too_many_arguments)]
fn lower_structured_statement(
    lower: StructuredLowering,
    statement: &wire::HirStatementProto,
    scope: &std::sync::Arc<str>,
    arena: &mut TypeArena,
    next_id: &mut u64,
    next_region: &mut u64,
    region_children: &mut RegionChildren,
    environment: &mut HashMap<String, wire::ValueProto>,
    operations: &mut Vec<wire::OperationProto>,
    loop_carries: Option<&[String]>,
    work: &mut HirSsaLoweringWork,
) -> Result<Option<wire::TerminatorProto>, HirLoweringError> {
    let saved_environment = environment.clone();
    let saved_children = region_children.clone();
    let saved_length = operations.len();
    let terminator = lower(
        statement,
        scope,
        arena,
        next_id,
        next_region,
        region_children,
        environment,
        operations,
        loop_carries,
        work,
    )?;
    let Some(groups) = operations
        .last()
        .filter(|_| operations.len() > saved_length)
        .map(|operation| fresh_carry_groups(operation, &saved_environment, region_children))
        .filter(|groups| !groups.is_empty())
    else {
        return Ok(terminator);
    };
    operations.truncate(saved_length);
    *environment = saved_environment;
    *region_children = saved_children;
    for names in groups {
        repack_owners(
            &names,
            statement.span.clone(),
            next_id,
            next_region,
            region_children,
            environment,
            operations,
        )?;
    }
    lower(
        statement,
        scope,
        arena,
        next_id,
        next_region,
        region_children,
        environment,
        operations,
        loop_carries,
        work,
    )
}

/// Find carries whose yielded successors are projections of a fresh owner.
///
/// Each returned group lists carry names ordered by projection index; it
/// covers every child of that owner and every member entered the statement
/// as an independent owner.
fn fresh_carry_groups(
    operation: &wire::OperationProto,
    environment: &HashMap<String, wire::ValueProto>,
    region_children: &RegionChildren,
) -> Vec<Vec<String>> {
    let names_attribute = if operation.opcode == "qcore.if" {
        "output_names"
    } else {
        "capture_names"
    };
    let Ok(names) = string_tuple_attribute(&operation.attributes, names_attribute) else {
        return Vec::new();
    };
    let mut groups: BTreeMap<u64, BTreeMap<usize, String>> = BTreeMap::new();
    let mut rejected = HashSet::new();
    for region in &operation.regions {
        let Some(block) = region.blocks.first() else {
            continue;
        };
        let Some(terminator) = block
            .terminator
            .as_ref()
            .filter(|terminator| terminator.kind == "yield")
        else {
            continue;
        };
        let defined = block
            .arguments
            .iter()
            .chain(block.operations.iter().flat_map(|op| op.results.iter()))
            .map(|value| (value.id, value))
            .collect::<HashMap<_, _>>();
        let argument_roots = block
            .arguments
            .iter()
            .filter_map(|value| value.region.as_ref().map(|region| region.root))
            .collect::<HashSet<_>>();
        for (name, id) in names.iter().zip(&terminator.operands) {
            let (Some(yielded), Some(entry)) = (defined.get(id), environment.get(name)) else {
                continue;
            };
            let (Some(observed), Some(expected)) = (&yielded.region, &entry.region) else {
                continue;
            };
            if observed == expected || argument_roots.contains(&observed.root) {
                continue;
            }
            let ([index], false) = (
                observed.projections.as_slice(),
                !expected.projections.is_empty(),
            ) else {
                rejected.insert(observed.root);
                continue;
            };
            let Ok(index) = index.parse::<usize>() else {
                rejected.insert(observed.root);
                continue;
            };
            let group = groups.entry(observed.root).or_default();
            if group.get(&index).is_some_and(|existing| existing != name) {
                rejected.insert(observed.root);
            }
            group.insert(index, name.clone());
        }
    }
    groups
        .into_iter()
        .filter(|(root, members)| {
            !rejected.contains(root)
                && members.len() > 1
                && members.keys().copied().eq(0..members.len())
                && region_children
                    .get(&(*root, Vec::new()))
                    .is_none_or(|children| children.len() == members.len())
        })
        .map(|(_, members)| members.into_values().collect())
        .collect()
}

/// Make independently owned values sibling projections of one fresh owner.
fn repack_owners(
    names: &[String],
    span: Option<wire::SourceSpanProto>,
    next_id: &mut u64,
    next_region: &mut u64,
    region_children: &mut RegionChildren,
    environment: &mut HashMap<String, wire::ValueProto>,
    operations: &mut Vec<wire::OperationProto>,
) -> Result<(), HirLoweringError> {
    let values = names
        .iter()
        .map(|name| {
            environment
                .get(name)
                .cloned()
                .ok_or_else(|| HirLoweringError::InvalidQCore(format!("unresolved carry {name:?}")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let element_types = values
        .iter()
        .map(|value| {
            value.r#type.clone().ok_or_else(|| {
                HirLoweringError::InvalidQCore("carried owner has no type".to_owned())
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let owner = allocate_root_region(next_region)?;
    let packed = new_value(
        wire_product_type(element_types.iter().cloned()),
        "carries".to_owned(),
        Some(owner.clone()),
        next_id,
    )?;
    operations.push(wire::OperationProto {
        opcode: "qcore.tuple".to_owned(),
        operands: values.iter().map(|value| value.id).collect(),
        results: vec![packed.clone()],
        span: span.clone(),
        ..Default::default()
    });
    region_children.insert(
        (owner.root, Vec::new()),
        (0..names.len()).map(|index| index.to_string()).collect(),
    );
    let mut results = Vec::with_capacity(names.len());
    for (index, (name, element_type)) in names.iter().zip(element_types).enumerate() {
        let mut region = owner.clone();
        region.projections.push(index.to_string());
        let value = new_value(element_type, name.clone(), Some(region), next_id)?;
        environment.insert(name.clone(), value.clone());
        results.push(value);
    }
    operations.push(wire::OperationProto {
        opcode: "qcore.unpack".to_owned(),
        operands: vec![packed.id],
        results,
        span,
        ..Default::default()
    });
    Ok(())
}

fn clone_region_argument(
    value: &wire::ValueProto,
    name: &str,
    next_id: &mut u64,
) -> Result<wire::ValueProto, HirLoweringError> {
    new_value(
        value
            .r#type
            .clone()
            .ok_or_else(|| HirLoweringError::InvalidQCore("if capture has no type".to_owned()))?,
        name.to_owned(),
        value.region.clone(),
        next_id,
    )
}

fn new_value(
    r#type: wire::TypeProto,
    hint: String,
    region: Option<wire::RegionRefProto>,
    next_id: &mut u64,
) -> Result<wire::ValueProto, HirLoweringError> {
    let value = wire::ValueProto {
        id: *next_id,
        r#type: Some(r#type),
        hint,
        has_region: region.is_some(),
        region,
    };
    *next_id = next_id
        .checked_add(1)
        .ok_or_else(|| HirLoweringError::Unsupported("SSA identifier capacity".to_owned()))?;
    Ok(value)
}

fn schedule_dead_results(
    operations: Vec<wire::OperationProto>,
    terminator: &wire::TerminatorProto,
) -> Vec<wire::OperationProto> {
    let mut uses = HashSet::new();
    for operation in &operations {
        uses.extend(operation.operands.iter().copied());
    }
    uses.extend(terminator.operands.iter().copied());
    let mut scheduled = Vec::new();
    for operation in operations {
        let dead: Vec<_> = operation
            .results
            .iter()
            .filter(|value| value.has_region && !uses.contains(&value.id))
            .cloned()
            .collect();
        let span = operation.span.clone();
        scheduled.push(operation);
        for value in dead {
            scheduled.push(wire::OperationProto {
                opcode: "qcore.forget".to_owned(),
                operands: vec![value.id],
                attributes: vec![wire::AttributeProto {
                    name: "reason".to_owned(),
                    value: Some(wire::LiteralProto {
                        value: Some(wire::literal_proto::Value::StringValue(
                            "end of lexical lifetime".to_owned(),
                        )),
                    }),
                }],
                span: span.clone(),
                ..Default::default()
            });
        }
    }
    scheduled
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn lower_expression(
    expression: &wire::HirExpressionProto,
    scope: &std::sync::Arc<str>,
    arena: &mut TypeArena,
    next_id: &mut u64,
    next_region: &mut u64,
    region_children: &mut RegionChildren,
    environment: &HashMap<String, wire::ValueProto>,
    operations: &mut Vec<wire::OperationProto>,
    work: &mut HirSsaLoweringWork,
) -> Result<wire::ValueProto, HirLoweringError> {
    work.visit_expression();
    if expression.opcode == "name" {
        let name = string_attribute(&expression.attributes, "name")?;
        return environment.get(name).cloned().ok_or_else(|| {
            HirLoweringError::InvalidQCore(format!("unresolved HIR name {name:?}"))
        });
    }
    if matches!(expression.opcode.as_str(), "invoke" | "adjoint_invoke") {
        return lower_invoke_expression(
            expression,
            scope,
            arena,
            next_id,
            next_region,
            region_children,
            environment,
            operations,
            work,
        );
    }
    if expression.opcode == "control_invoke" {
        return lower_control_invoke_expression(
            expression,
            scope,
            arena,
            next_id,
            next_region,
            region_children,
            environment,
            operations,
            work,
        );
    }
    if expression.opcode == "static_slice" {
        return lower_static_slice_expression(
            expression,
            scope,
            arena,
            next_id,
            next_region,
            region_children,
            environment,
            operations,
            work,
        );
    }
    let operands = lower_expression_operands(
        expression,
        scope,
        arena,
        next_id,
        next_region,
        region_children,
        environment,
        operations,
        work,
    )?;
    if matches!(
        expression.opcode.as_str(),
        "quantum.add_values" | "quantum.sub_values"
    ) {
        return lower_quantum_uint_transition(
            expression,
            next_id,
            next_region,
            region_children,
            operations,
            &operands,
        );
    }
    if matches!(
        expression.opcode.as_str(),
        "quantum.mod_add_values"
            | "quantum.mul_add_values"
            | "quantum.mod_mul_add_values"
            | "quantum.less_than_xor_values"
            | "quantum.mod_multi_pow_values"
            | "quantum.mod_pow_values"
    ) {
        return lower_quantum_register_transition(
            expression,
            next_id,
            next_region,
            region_children,
            operations,
            &operands,
        );
    }
    if expression.opcode == "lazy_if" {
        return lower_lazy_if_expression(
            expression,
            scope,
            arena,
            next_id,
            next_region,
            region_children,
            environment,
            operations,
            &operands,
            work,
        );
    }
    lower_ordinary_expression(
        expression,
        scope,
        arena,
        next_id,
        next_region,
        region_children,
        operations,
        &operands,
    )
}

#[allow(clippy::too_many_arguments)]
fn lower_expression_operands(
    expression: &wire::HirExpressionProto,
    scope: &Arc<str>,
    arena: &mut TypeArena,
    next_id: &mut u64,
    next_region: &mut u64,
    region_children: &mut RegionChildren,
    environment: &HashMap<String, wire::ValueProto>,
    operations: &mut Vec<wire::OperationProto>,
    work: &mut HirSsaLoweringWork,
) -> Result<Vec<wire::ValueProto>, HirLoweringError> {
    let evaluation_order = expression_evaluation_order(expression)?;
    let mut operands = vec![None; expression.operands.len()];
    for index in evaluation_order {
        operands[index] = Some(lower_expression(
            &expression.operands[index],
            scope,
            arena,
            next_id,
            next_region,
            region_children,
            environment,
            operations,
            work,
        )?);
    }
    operands
        .into_iter()
        .map(|operand| {
            operand.ok_or_else(|| {
                HirLoweringError::InvalidQCore(
                    "expression evaluation order omitted an operand".to_owned(),
                )
            })
        })
        .collect()
}

/// Lower one two-register unsigned transition through the existing aggregate adder.
///
/// Source arithmetic updates the left register and retains the right. The
/// persistent addition primitive instead retains its first payload component
/// and updates its second, so this HIR-only projection packs `(right, left)`,
/// applies the primitive, then restores source result order. Structural tuple,
/// reshape, and unpack operations emit no target instructions.
fn lower_quantum_uint_transition(
    expression: &wire::HirExpressionProto,
    next_id: &mut u64,
    next_region: &mut u64,
    region_children: &mut RegionChildren,
    operations: &mut Vec<wire::OperationProto>,
    operands: &[wire::ValueProto],
) -> Result<wire::ValueProto, HirLoweringError> {
    let [left, right] = operands else {
        return Err(HirLoweringError::InvalidQCore(
            "coherent arithmetic requires two operands".to_owned(),
        ));
    };
    let left_type = left.r#type.as_ref().ok_or_else(|| {
        HirLoweringError::InvalidQCore("left arithmetic operand has no type".to_owned())
    })?;
    let right_type = right.r#type.as_ref().ok_or_else(|| {
        HirLoweringError::InvalidQCore("right arithmetic operand has no type".to_owned())
    })?;
    let left_payload = quantum_wire_payload(left_type)?;
    let right_payload = quantum_wire_payload(right_type)?;
    let packed_product_type = wire_product_type([right_type.clone(), left_type.clone()]);
    let packed_region = allocate_root_region(next_region)?;
    let packed = new_value(
        packed_product_type.clone(),
        "arithmetic_operands".to_owned(),
        Some(packed_region.clone()),
        next_id,
    )?;
    operations.push(wire::OperationProto {
        opcode: "qcore.tuple".to_owned(),
        operands: vec![right.id, left.id],
        results: vec![packed.clone()],
        span: expression.span.clone(),
        ..Default::default()
    });

    let quantum_pair_type = wire::TypeProto {
        kind: wire::type_proto::Kind::Quantum.into(),
        arguments: vec![wire_product_type([right_payload, left_payload])],
        ..Default::default()
    };
    let quantum_pair = new_value(
        quantum_pair_type.clone(),
        "arithmetic_state".to_owned(),
        Some(packed_region.clone()),
        next_id,
    )?;
    operations.push(wire::OperationProto {
        opcode: "qcore.reshape".to_owned(),
        operands: vec![packed.id],
        results: vec![quantum_pair.clone()],
        span: expression.span.clone(),
        ..Default::default()
    });

    let updated_pair = new_value(
        quantum_pair_type,
        "arithmetic_result".to_owned(),
        Some(packed_region.clone()),
        next_id,
    )?;
    operations.push(wire::OperationProto {
        opcode: if expression.opcode == "quantum.add_values" {
            "quantum.add_into".to_owned()
        } else {
            "quantum.subtract_into".to_owned()
        },
        operands: vec![quantum_pair.id],
        results: vec![updated_pair.clone()],
        span: expression.span.clone(),
        ..Default::default()
    });

    let (updated_left, retained_right) = unpack_quantum_uint_pair(
        expression,
        &updated_pair,
        packed_product_type,
        &packed_region,
        left_type,
        right_type,
        next_id,
        region_children,
        operations,
    )?;

    emit_quantum_uint_successors(
        expression,
        &updated_left,
        &retained_right,
        next_id,
        next_region,
        operations,
    )
}

/// Pack retained quantum operands into one semantic arithmetic state.
#[allow(clippy::too_many_lines)]
fn lower_quantum_register_transition(
    expression: &wire::HirExpressionProto,
    next_id: &mut u64,
    next_region: &mut u64,
    region_children: &mut RegionChildren,
    operations: &mut Vec<wire::OperationProto>,
    operands: &[wire::ValueProto],
) -> Result<wire::ValueProto, HirLoweringError> {
    let quantum_count = if matches!(
        expression.opcode.as_str(),
        "quantum.mod_pow_values" | "quantum.mod_multi_pow_values"
    ) {
        2
    } else {
        3
    };
    if operands.len()
        != quantum_count
            + usize::from(expression.opcode == "quantum.mod_pow_values") * 2
            + usize::from(expression.opcode == "quantum.mod_multi_pow_values") * 4
            + usize::from(expression.opcode == "quantum.mod_mul_add_values")
    {
        return Err(HirLoweringError::InvalidQCore(
            "arithmetic transition has an invalid operand layout".to_owned(),
        ));
    }
    let types = operands[..quantum_count]
        .iter()
        .map(|value| {
            value.r#type.clone().ok_or_else(|| {
                HirLoweringError::InvalidQCore("arithmetic operand has no type".to_owned())
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let payloads = types
        .iter()
        .map(quantum_wire_payload)
        .collect::<Result<Vec<_>, _>>()?;
    let product_type = wire_product_type(types.clone());
    let region = allocate_root_region(next_region)?;
    let packed = new_value(
        product_type.clone(),
        "arithmetic_operands".to_owned(),
        Some(region.clone()),
        next_id,
    )?;
    operations.push(wire::OperationProto {
        opcode: "qcore.tuple".to_owned(),
        operands: operands[..quantum_count]
            .iter()
            .map(|value| value.id)
            .collect(),
        results: vec![packed.clone()],
        span: expression.span.clone(),
        ..Default::default()
    });
    let state_type = wire::TypeProto {
        kind: wire::type_proto::Kind::Quantum.into(),
        arguments: vec![wire_product_type(payloads)],
        ..Default::default()
    };
    let state = new_value(
        state_type.clone(),
        "arithmetic_state".to_owned(),
        Some(region.clone()),
        next_id,
    )?;
    operations.push(wire::OperationProto {
        opcode: "qcore.reshape".to_owned(),
        operands: vec![packed.id],
        results: vec![state.clone()],
        span: expression.span.clone(),
        ..Default::default()
    });
    let updated = new_value(
        state_type,
        "arithmetic_updated".to_owned(),
        Some(region.clone()),
        next_id,
    )?;
    let opcode = match expression.opcode.as_str() {
        "quantum.mod_add_values" => "quantum.mod_add",
        "quantum.mul_add_values" => "quantum.mul_add",
        "quantum.mod_mul_add_values" => "quantum.mod_mul_add",
        "quantum.less_than_xor_values" => "quantum.less_than_xor",
        "quantum.mod_multi_pow_values" => "quantum.mod_multi_pow",
        "quantum.mod_pow_values" => "quantum.mod_pow",
        _ => unreachable!("closed arithmetic transition"),
    };
    let mut operation_operands = vec![state.id];
    operation_operands.extend(operands[quantum_count..].iter().map(|value| value.id));
    operations.push(wire::OperationProto {
        opcode: opcode.to_owned(),
        operands: operation_operands,
        results: vec![updated.clone()],
        span: expression.span.clone(),
        ..Default::default()
    });
    let unpackable = new_value(
        product_type,
        "arithmetic_product".to_owned(),
        Some(region.clone()),
        next_id,
    )?;
    operations.push(wire::OperationProto {
        opcode: "qcore.reshape".to_owned(),
        operands: vec![updated.id],
        results: vec![unpackable.clone()],
        span: expression.span.clone(),
        ..Default::default()
    });
    region_children.insert(
        (region.root, region.projections.clone()),
        (0..quantum_count).map(|index| index.to_string()).collect(),
    );
    let successors = types
        .into_iter()
        .enumerate()
        .map(|(index, r#type)| {
            new_value(
                r#type,
                format!("arithmetic_successor_{index}"),
                Some(wire::RegionRefProto {
                    root: region.root,
                    projections: vec![index.to_string()],
                }),
                next_id,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    operations.push(wire::OperationProto {
        opcode: "qcore.unpack".to_owned(),
        operands: vec![unpackable.id],
        results: successors.clone(),
        span: expression.span.clone(),
        ..Default::default()
    });
    let result = new_value(
        expression.r#type.clone().ok_or_else(|| {
            HirLoweringError::InvalidQCore("arithmetic result has no type".to_owned())
        })?,
        "arithmetic_successors".to_owned(),
        Some(allocate_root_region(next_region)?),
        next_id,
    )?;
    operations.push(wire::OperationProto {
        opcode: "qcore.tuple".to_owned(),
        operands: successors.iter().map(|value| value.id).collect(),
        results: vec![result.clone()],
        span: expression.span.clone(),
        ..Default::default()
    });
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
fn unpack_quantum_uint_pair(
    expression: &wire::HirExpressionProto,
    updated_pair: &wire::ValueProto,
    packed_product_type: wire::TypeProto,
    packed_region: &wire::RegionRefProto,
    left_type: &wire::TypeProto,
    right_type: &wire::TypeProto,
    next_id: &mut u64,
    region_children: &mut RegionChildren,
    operations: &mut Vec<wire::OperationProto>,
) -> Result<(wire::ValueProto, wire::ValueProto), HirLoweringError> {
    let unpackable = new_value(
        packed_product_type,
        "arithmetic_product".to_owned(),
        Some(packed_region.clone()),
        next_id,
    )?;
    operations.push(wire::OperationProto {
        opcode: "qcore.reshape".to_owned(),
        operands: vec![updated_pair.id],
        results: vec![unpackable.clone()],
        span: expression.span.clone(),
        ..Default::default()
    });
    region_children.insert(
        (packed_region.root, packed_region.projections.clone()),
        ["0".to_owned(), "1".to_owned()].into_iter().collect(),
    );
    let retained_right = new_value(
        right_type.clone(),
        "retained_addend".to_owned(),
        Some(wire::RegionRefProto {
            root: packed_region.root,
            projections: vec!["0".to_owned()],
        }),
        next_id,
    )?;
    let updated_left = new_value(
        left_type.clone(),
        "updated_target".to_owned(),
        Some(wire::RegionRefProto {
            root: packed_region.root,
            projections: vec!["1".to_owned()],
        }),
        next_id,
    )?;
    operations.push(wire::OperationProto {
        opcode: "qcore.unpack".to_owned(),
        operands: vec![unpackable.id],
        results: vec![retained_right.clone(), updated_left.clone()],
        span: expression.span.clone(),
        ..Default::default()
    });
    Ok((updated_left, retained_right))
}

fn emit_quantum_uint_successors(
    expression: &wire::HirExpressionProto,
    updated_left: &wire::ValueProto,
    retained_right: &wire::ValueProto,
    next_id: &mut u64,
    next_region: &mut u64,
    operations: &mut Vec<wire::OperationProto>,
) -> Result<wire::ValueProto, HirLoweringError> {
    let result_type = expression.r#type.clone().ok_or_else(|| {
        HirLoweringError::InvalidQCore("coherent arithmetic has no result type".to_owned())
    })?;
    let result = new_value(
        result_type,
        "arithmetic_successors".to_owned(),
        Some(allocate_root_region(next_region)?),
        next_id,
    )?;
    operations.push(wire::OperationProto {
        opcode: "qcore.tuple".to_owned(),
        operands: vec![updated_left.id, retained_right.id],
        results: vec![result.clone()],
        span: expression.span.clone(),
        ..Default::default()
    });
    Ok(result)
}

fn allocate_root_region(next_region: &mut u64) -> Result<wire::RegionRefProto, HirLoweringError> {
    let region = wire::RegionRefProto {
        root: *next_region,
        projections: Vec::new(),
    };
    *next_region = next_region
        .checked_add(1)
        .ok_or_else(|| HirLoweringError::Unsupported("ownership region capacity".to_owned()))?;
    Ok(region)
}

fn quantum_wire_payload(source: &wire::TypeProto) -> Result<wire::TypeProto, HirLoweringError> {
    if source.kind != wire::type_proto::Kind::Quantum as i32 || source.arguments.len() != 1 {
        return Err(HirLoweringError::InvalidQCore(
            "coherent arithmetic operand is not Q[T]".to_owned(),
        ));
    }
    Ok(source.arguments[0].clone())
}

fn wire_product_type(elements: impl IntoIterator<Item = wire::TypeProto>) -> wire::TypeProto {
    wire::TypeProto {
        kind: wire::type_proto::Kind::Product.into(),
        arguments: elements.into_iter().collect(),
        ..Default::default()
    }
}

#[allow(clippy::too_many_arguments)]
fn lower_static_slice_expression(
    expression: &wire::HirExpressionProto,
    scope: &std::sync::Arc<str>,
    arena: &mut TypeArena,
    next_id: &mut u64,
    next_region: &mut u64,
    region_children: &mut RegionChildren,
    environment: &HashMap<String, wire::ValueProto>,
    operations: &mut Vec<wire::OperationProto>,
    work: &mut HirSsaLoweringWork,
) -> Result<wire::ValueProto, HirLoweringError> {
    let [source] = expression.operands.as_slice() else {
        return Err(HirLoweringError::InvalidQCore(
            "static slice must have one source operand".to_owned(),
        ));
    };
    if !expression.regions.is_empty()
        || expression.attributes.len() != 3
        || expression.attributes[0].name != "start"
        || expression.attributes[1].name != "step"
        || expression.attributes[2].name != "stop"
    {
        return Err(HirLoweringError::InvalidQCore(
            "static slice has an invalid attribute or region row".to_owned(),
        ));
    }
    let source = lower_expression(
        source,
        scope,
        arena,
        next_id,
        next_region,
        region_children,
        environment,
        operations,
        work,
    )?;
    let (extent, element_type) = fixed_array_slice_contract(&source)?;
    let start = optional_integer_attribute(&expression.attributes, "start")?;
    let stop = optional_integer_attribute(&expression.attributes, "stop")?;
    let stride = optional_integer_attribute(&expression.attributes, "step")?;
    let positions =
        normalize_aggregate_slice(extent, start.as_ref(), stop.as_ref(), stride.as_ref())
            .map_err(|error| HirLoweringError::InvalidQCore(error.to_string()))?;
    if positions.is_empty() {
        return Err(HirLoweringError::InvalidQCore(
            "static slice cannot produce an empty fixed array".to_owned(),
        ));
    }

    let mut elements = Vec::with_capacity(positions.len());
    for position in positions {
        let index =
            aggregate_index_constant(position, expression.span.clone(), next_id, operations)?;
        let element = new_value(
            element_type.clone(),
            "slice_element".to_owned(),
            None,
            next_id,
        )?;
        operations.push(wire::OperationProto {
            opcode: "classical.index".to_owned(),
            operands: vec![source.id, index.id],
            results: vec![element.clone()],
            span: expression.span.clone(),
            ..Default::default()
        });
        elements.push(element);
    }
    let result = new_value(
        expression.r#type.clone().ok_or_else(|| {
            HirLoweringError::InvalidQCore("static slice has no result type".to_owned())
        })?,
        "slice".to_owned(),
        None,
        next_id,
    )?;
    operations.push(wire::OperationProto {
        opcode: "array".to_owned(),
        operands: elements.iter().map(|value| value.id).collect(),
        results: vec![result.clone()],
        span: expression.span.clone(),
        ..Default::default()
    });
    Ok(result)
}

fn fixed_array_slice_contract(
    source: &wire::ValueProto,
) -> Result<(usize, wire::TypeProto), HirLoweringError> {
    if source.has_region {
        return Err(HirLoweringError::InvalidQCore(
            "static slice source must be classical".to_owned(),
        ));
    }
    let source_type = source.r#type.as_ref().ok_or_else(|| {
        HirLoweringError::InvalidQCore("static slice source has no type".to_owned())
    })?;
    let (array_type, stage) = if source_type.kind == wire::type_proto::Kind::Staged as i32
        && source_type.arguments.len() == 1
    {
        (&source_type.arguments[0], Some(source_type.name.clone()))
    } else {
        (source_type, None)
    };
    if array_type.kind != wire::type_proto::Kind::Array as i32
        || array_type.name != "Array"
        || array_type.arguments.len() != 1
        || array_type.dimensions.is_empty()
    {
        return Err(HirLoweringError::InvalidQCore(
            "static slice source must be a classical array".to_owned(),
        ));
    }
    let extent = match array_type.dimensions[0].value.as_ref() {
        Some(wire::dimension_proto::Value::Fixed(value)) => usize::try_from(*value)
            .map_err(|_| HirLoweringError::Unsupported("static slice extent".to_owned()))?,
        _ => {
            return Err(HirLoweringError::InvalidQCore(
                "static slice source must have a fixed leading extent".to_owned(),
            ));
        }
    };
    let element = if array_type.dimensions.len() == 1 {
        array_type.arguments[0].clone()
    } else {
        let mut element = array_type.clone();
        element.dimensions.remove(0);
        element
    };
    Ok((extent, apply_outer_stage(element, stage)))
}

#[allow(clippy::too_many_arguments)]
fn lower_ordinary_expression(
    expression: &wire::HirExpressionProto,
    scope: &std::sync::Arc<str>,
    arena: &mut TypeArena,
    next_id: &mut u64,
    next_region: &mut u64,
    region_children: &mut RegionChildren,
    operations: &mut Vec<wire::OperationProto>,
    operands: &[wire::ValueProto],
) -> Result<wire::ValueProto, HirLoweringError> {
    let r#type = expression
        .r#type
        .as_ref()
        .ok_or_else(|| HirLoweringError::InvalidQCore("expression has no type".to_owned()))?;
    let type_id = arena
        .resolve_wire_type(r#type, scope)
        .map_err(HirLoweringError::InvalidQCore)?;
    let opcode = match expression.opcode.as_str() {
        "literal" | "module_constant" => "qcore.constant",
        "array" if !arena.is_quantum(type_id) => "array",
        "record" | "tuple" => "qcore.tuple",
        "qcore.reshape" => "qcore.reshape",
        "call" => "qcore.call",
        "adjoint_call" => "qcore.adjoint_call",
        "phase_by" => "qcore.phase_by",
        "xor_by" => "qcore.xor_by",
        "permute_by" => "qcore.permute_by",
        "control_call" => "qcore.control_call",
        "select_call" => "qcore.select_call",
        opcode if opcode.starts_with("classical.") => opcode,
        opcode if opcode.starts_with("quantum.") => opcode,
        opcode => {
            return Err(HirLoweringError::Unsupported(format!(
                "expression {opcode:?}"
            )));
        }
    };
    let region = lower_result_region(
        expression,
        opcode,
        arena.is_quantum(type_id),
        operands,
        next_region,
        region_children,
    )?;
    let value = wire::ValueProto {
        id: *next_id,
        r#type: Some(r#type.clone()),
        hint: if opcode == "qcore.constant" {
            "const".to_owned()
        } else {
            expression
                .opcode
                .rsplit_once('.')
                .map_or(expression.opcode.clone(), |(_, hint)| hint.to_owned())
        },
        has_region: region.is_some(),
        region,
    };
    *next_id = next_id
        .checked_add(1)
        .ok_or_else(|| HirLoweringError::Unsupported("SSA identifier capacity".to_owned()))?;
    let attributes = if opcode == "qcore.constant" {
        expression
            .attributes
            .iter()
            .filter(|attribute| attribute.name == "value")
            .cloned()
            .collect()
    } else {
        expression
            .attributes
            .iter()
            .filter(|attribute| attribute.name != "evaluation_order")
            .cloned()
            .collect()
    };
    operations.push(wire::OperationProto {
        opcode: opcode.to_owned(),
        operands: operands.iter().map(|operand| operand.id).collect(),
        results: vec![value.clone()],
        attributes,
        span: expression.span.clone(),
        ..Default::default()
    });
    Ok(value)
}

const MAX_FUNCTION_VALUE_CANDIDATES: usize = 256;

#[allow(clippy::too_many_arguments)]
fn lower_control_invoke_expression(
    expression: &wire::HirExpressionProto,
    scope: &std::sync::Arc<str>,
    arena: &mut TypeArena,
    next_id: &mut u64,
    next_region: &mut u64,
    region_children: &mut RegionChildren,
    environment: &HashMap<String, wire::ValueProto>,
    operations: &mut Vec<wire::OperationProto>,
    work: &mut HirSsaLoweringWork,
) -> Result<wire::ValueProto, HirLoweringError> {
    let Some((callee, arguments)) = expression.operands.split_first() else {
        return Err(HirLoweringError::InvalidQCore(
            "control_invoke expects one callable".to_owned(),
        ));
    };
    if callee.opcode != "function_ref" {
        return Err(HirLoweringError::Unsupported(
            "controlled qkernel must resolve to one closed function".to_owned(),
        ));
    }
    let callee_symbol = string_attribute(&callee.attributes, "callee")?.to_owned();
    let evaluation_order = expression_evaluation_order(expression)?;
    if evaluation_order.first() != Some(&0) {
        return Err(HirLoweringError::InvalidQCore(
            "control_invoke must evaluate its callable first".to_owned(),
        ));
    }
    let projected_order = evaluation_order[1..]
        .iter()
        .map(|index| index - 1)
        .collect::<Vec<_>>();
    let mut controlled = expression.clone();
    "control_call".clone_into(&mut controlled.opcode);
    controlled.operands = arguments.to_vec();
    controlled
        .attributes
        .retain(|attribute| attribute.name != "evaluation_order");
    controlled.attributes.push(wire::AttributeProto {
        name: "callee".to_owned(),
        value: Some(wire::LiteralProto {
            value: Some(wire::literal_proto::Value::StringValue(callee_symbol)),
        }),
    });
    append_nonidentity_evaluation_order(&mut controlled.attributes, &projected_order)?;
    controlled
        .attributes
        .sort_by(|left, right| left.name.cmp(&right.name));
    lower_expression(
        &controlled,
        scope,
        arena,
        next_id,
        next_region,
        region_children,
        environment,
        operations,
        work,
    )
}

#[allow(clippy::too_many_arguments)]
fn lower_invoke_expression(
    expression: &wire::HirExpressionProto,
    scope: &std::sync::Arc<str>,
    arena: &mut TypeArena,
    next_id: &mut u64,
    next_region: &mut u64,
    region_children: &mut RegionChildren,
    environment: &HashMap<String, wire::ValueProto>,
    operations: &mut Vec<wire::OperationProto>,
    work: &mut HirSsaLoweringWork,
) -> Result<wire::ValueProto, HirLoweringError> {
    if !expression.regions.is_empty() {
        return Err(HirLoweringError::InvalidQCore(
            "invoke carries unexpected regions".to_owned(),
        ));
    }
    let Some((callee, arguments)) = expression.operands.split_first() else {
        return Err(HirLoweringError::InvalidQCore(
            "invoke expects one callable".to_owned(),
        ));
    };
    if callable_candidate_count(callee)? > MAX_FUNCTION_VALUE_CANDIDATES {
        return Err(HirLoweringError::Unsupported(format!(
            "function value exceeds {MAX_FUNCTION_VALUE_CANDIDATES} static candidates"
        )));
    }
    if callee.opcode == "lazy_if" {
        lower_selected_callable_application(
            callee,
            arguments,
            expression,
            scope,
            arena,
            next_id,
            next_region,
            region_children,
            environment,
            operations,
            work,
        )
    } else {
        validate_callable_application(callee, arguments, expression, scope, arena)?;
        lower_function_ref_application(
            callee,
            arguments,
            expression,
            scope,
            arena,
            next_id,
            next_region,
            region_children,
            environment,
            operations,
            work,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn lower_selected_callable_application(
    callee: &wire::HirExpressionProto,
    arguments: &[wire::HirExpressionProto],
    invocation: &wire::HirExpressionProto,
    scope: &Arc<str>,
    arena: &mut TypeArena,
    next_id: &mut u64,
    next_region: &mut u64,
    region_children: &mut RegionChildren,
    environment: &HashMap<String, wire::ValueProto>,
    operations: &mut Vec<wire::OperationProto>,
    work: &mut HirSsaLoweringWork,
) -> Result<wire::ValueProto, HirLoweringError> {
    const INLINE_CALLABLE_NAME: &str = "\0invoke.callable";

    let catalog = FunctionAliasCatalog::from_callable(callee, INLINE_CALLABLE_NAME)?;
    let mut expanded_nodes = 0;
    let tag_expression = lower_callable_to_tag(
        callee,
        INLINE_CALLABLE_NAME,
        &FunctionAliasEnvironment::default(),
        &catalog,
        &mut expanded_nodes,
        0,
    )?;
    let tag_value = lower_expression(
        &tag_expression,
        scope,
        arena,
        next_id,
        next_region,
        region_children,
        environment,
        operations,
        work,
    )?;
    let argument_order = invocation_argument_evaluation_order(invocation, arguments.len())?;
    let argument_values = lower_arguments_in_order(
        arguments,
        &argument_order,
        scope,
        arena,
        next_id,
        next_region,
        region_children,
        environment,
        operations,
        work,
    )?;
    let tag_name = format!("$invoke.callable.{}", tag_value.id);
    let argument_names = argument_values
        .iter()
        .map(|argument| format!("$invoke.argument.{}", argument.id))
        .collect::<Vec<_>>();
    let mut invocation_environment = environment.clone();
    invocation_environment.insert(tag_name.clone(), tag_value.clone());
    invocation_environment.extend(
        argument_names
            .iter()
            .cloned()
            .zip(argument_values.iter().cloned()),
    );
    let argument_refs = argument_names
        .iter()
        .zip(&argument_values)
        .map(|(name, value)| {
            hir_name_expression(
                name,
                value
                    .r#type
                    .clone()
                    .expect("validated invocation argument type"),
                invocation.span.clone(),
            )
        })
        .collect::<Vec<_>>();
    let tag_type = tag_value.r#type.clone().ok_or_else(|| {
        HirLoweringError::InvalidQCore("selected callable tag has no type".to_owned())
    })?;
    let function_type = callee.r#type.as_ref().ok_or_else(|| {
        HirLoweringError::InvalidQCore("selected callable has no type".to_owned())
    })?;
    let candidates = catalog.candidates_for(INLINE_CALLABLE_NAME, function_type)?;
    let dispatch = build_static_callable_dispatch(
        &tag_name,
        &tag_type,
        &candidates,
        &catalog,
        &argument_refs,
        invocation,
        scope,
        arena,
        invocation.span.clone(),
    )?;
    lower_expression(
        &dispatch,
        scope,
        arena,
        next_id,
        next_region,
        region_children,
        &invocation_environment,
        operations,
        work,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_static_callable_dispatch(
    tag_name: &str,
    tag_type: &wire::TypeProto,
    candidates: &[String],
    catalog: &FunctionAliasCatalog,
    arguments: &[wire::HirExpressionProto],
    invocation: &wire::HirExpressionProto,
    scope: &Arc<str>,
    arena: &mut TypeArena,
    span: Option<wire::SourceSpanProto>,
) -> Result<wire::HirExpressionProto, HirLoweringError> {
    if let [symbol] = candidates {
        let callee = catalog.references.get(symbol).ok_or_else(|| {
            HirLoweringError::InvalidQCore(format!(
                "function candidate {symbol:?} has no reference"
            ))
        })?;
        let (ordered_indices, result_type) =
            bind_hir_callable_application(callee, arguments, invocation, scope, arena)?;
        return Ok(wire::HirExpressionProto {
            opcode: if invocation.opcode == "adjoint_invoke" {
                "adjoint_call"
            } else {
                "call"
            }
            .to_owned(),
            r#type: result_type,
            operands: ordered_indices
                .iter()
                .map(|index| arguments[*index].clone())
                .collect(),
            attributes: resolved_callable_attributes(symbol, &invocation.opcode),
            span,
            ..Default::default()
        });
    }
    let middle = candidates.len() / 2;
    let (left, right) = candidates.split_at(middle);
    let pivot = catalog.tags.get(&right[0]).copied().ok_or_else(|| {
        HirLoweringError::InvalidQCore(format!("function candidate {:?} has no tag", right[0]))
    })?;
    let tag = hir_name_expression(tag_name, tag_type.clone(), span.clone());
    let condition = wire::HirExpressionProto {
        opcode: "classical.compare_chain".to_owned(),
        r#type: Some(scalar_at_tag_stage(tag_type, "Bool")),
        operands: vec![tag, function_tag_literal(pivot, span.clone())],
        attributes: vec![string_tuple_qcore_attribute(
            "operators",
            &["lt".to_owned()],
        )],
        span: span.clone(),
        ..Default::default()
    };
    let mut captures = arguments
        .iter()
        .flat_map(expression_names)
        .collect::<Vec<_>>();
    captures.push(tag_name.to_owned());
    captures.sort();
    captures.dedup();
    Ok(wire::HirExpressionProto {
        opcode: "lazy_if".to_owned(),
        r#type: invocation.r#type.clone(),
        operands: vec![condition],
        attributes: vec![string_tuple_qcore_attribute("captures", &captures)],
        regions: vec![
            build_static_callable_dispatch(
                tag_name,
                tag_type,
                left,
                catalog,
                arguments,
                invocation,
                scope,
                arena,
                span.clone(),
            )?,
            build_static_callable_dispatch(
                tag_name,
                tag_type,
                right,
                catalog,
                arguments,
                invocation,
                scope,
                arena,
                span.clone(),
            )?,
        ],
        span,
    })
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn lower_function_ref_application(
    callee: &wire::HirExpressionProto,
    arguments: &[wire::HirExpressionProto],
    invocation: &wire::HirExpressionProto,
    scope: &std::sync::Arc<str>,
    arena: &mut TypeArena,
    next_id: &mut u64,
    next_region: &mut u64,
    region_children: &mut RegionChildren,
    environment: &HashMap<String, wire::ValueProto>,
    operations: &mut Vec<wire::OperationProto>,
    work: &mut HirSsaLoweringWork,
) -> Result<wire::ValueProto, HirLoweringError> {
    if !callee.operands.is_empty() || !callee.regions.is_empty() {
        return Err(HirLoweringError::InvalidQCore(
            "function_ref carries operands or regions".to_owned(),
        ));
    }
    let symbol = string_attribute(&callee.attributes, "callee")?;
    if callee.attributes.len() != 1 {
        return Err(HirLoweringError::InvalidQCore(
            "function_ref has an invalid attribute row".to_owned(),
        ));
    }
    let argument_order = invocation_argument_evaluation_order(invocation, arguments.len())?;
    let raw_values = lower_arguments_in_order(
        arguments,
        &argument_order,
        scope,
        arena,
        next_id,
        next_region,
        region_children,
        environment,
        operations,
        work,
    )?;
    let callee_type = arena
        .resolve_wire_type(
            callee.r#type.as_ref().ok_or_else(|| {
                HirLoweringError::InvalidQCore("function_ref has no type".to_owned())
            })?,
            scope,
        )
        .map_err(HirLoweringError::InvalidQCore)?;
    let argument_types = raw_values
        .iter()
        .map(|value| {
            arena
                .resolve_wire_type(
                    value.r#type.as_ref().ok_or_else(|| {
                        HirLoweringError::InvalidQCore("function argument has no type".to_owned())
                    })?,
                    scope,
                )
                .map_err(HirLoweringError::InvalidQCore)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let (positional_count, keyword_names) = invocation_syntax(invocation, arguments.len())?;
    let binding = bind_function_application(
        arena,
        callee_type,
        &argument_types,
        positional_count,
        &keyword_names,
    )
    .map_err(HirLoweringError::InvalidQCore)?;
    let argument_names = raw_values
        .iter()
        .map(|argument| format!("$invoke.argument.{}", argument.id))
        .collect::<Vec<_>>();
    let mut invocation_environment = environment.clone();
    invocation_environment.extend(
        argument_names
            .iter()
            .cloned()
            .zip(raw_values.iter().cloned()),
    );
    let direct = wire::HirExpressionProto {
        opcode: if invocation.opcode == "adjoint_invoke" {
            "adjoint_call"
        } else {
            "call"
        }
        .to_owned(),
        r#type: invocation.r#type.clone(),
        operands: binding
            .ordered_indices()
            .iter()
            .map(|index| {
                hir_name_expression(
                    &argument_names[*index],
                    raw_values[*index]
                        .r#type
                        .clone()
                        .expect("validated function argument type"),
                    invocation.span.clone(),
                )
            })
            .collect(),
        attributes: resolved_callable_attributes(symbol, &invocation.opcode),
        span: invocation.span.clone(),
        ..Default::default()
    };
    lower_expression(
        &direct,
        scope,
        arena,
        next_id,
        next_region,
        region_children,
        &invocation_environment,
        operations,
        work,
    )
}

fn resolved_callable_attributes(symbol: &str, opcode: &str) -> Vec<wire::AttributeProto> {
    let mut attributes = vec![wire::AttributeProto {
        name: "callee".to_owned(),
        value: Some(wire::LiteralProto {
            value: Some(wire::literal_proto::Value::StringValue(symbol.to_owned())),
        }),
    }];
    if opcode == "adjoint_invoke" {
        attributes.push(wire::AttributeProto {
            name: "proof".to_owned(),
            value: Some(wire::LiteralProto {
                value: Some(wire::literal_proto::Value::StringValue(
                    "explicit-adjoint".to_owned(),
                )),
            }),
        });
    }
    attributes
}

fn invocation_argument_evaluation_order(
    invocation: &wire::HirExpressionProto,
    argument_count: usize,
) -> Result<Vec<usize>, HirLoweringError> {
    let order = expression_evaluation_order(invocation)?;
    if order.first() != Some(&0) || order.len() != argument_count + 1 {
        return Err(HirLoweringError::InvalidQCore(
            "invoke must evaluate its callable before its arguments".to_owned(),
        ));
    }
    order
        .into_iter()
        .skip(1)
        .map(|index| {
            index.checked_sub(1).ok_or_else(|| {
                HirLoweringError::InvalidQCore(
                    "invoke has an invalid argument evaluation order".to_owned(),
                )
            })
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn lower_arguments_in_order(
    arguments: &[wire::HirExpressionProto],
    evaluation_order: &[usize],
    scope: &std::sync::Arc<str>,
    arena: &mut TypeArena,
    next_id: &mut u64,
    next_region: &mut u64,
    region_children: &mut RegionChildren,
    environment: &HashMap<String, wire::ValueProto>,
    operations: &mut Vec<wire::OperationProto>,
    work: &mut HirSsaLoweringWork,
) -> Result<Vec<wire::ValueProto>, HirLoweringError> {
    let mut values = vec![None; arguments.len()];
    for index in evaluation_order {
        values[*index] = Some(lower_expression(
            &arguments[*index],
            scope,
            arena,
            next_id,
            next_region,
            region_children,
            environment,
            operations,
            work,
        )?);
    }
    values
        .into_iter()
        .map(|value| {
            value.ok_or_else(|| {
                HirLoweringError::InvalidQCore(
                    "invoke evaluation order omitted an argument".to_owned(),
                )
            })
        })
        .collect()
}

type BoundHirCallableApplication = (Box<[usize]>, Option<wire::TypeProto>);

fn bind_hir_callable_application(
    callee: &wire::HirExpressionProto,
    arguments: &[wire::HirExpressionProto],
    invocation: &wire::HirExpressionProto,
    scope: &Arc<str>,
    arena: &mut TypeArena,
) -> Result<BoundHirCallableApplication, HirLoweringError> {
    let callee_type = arena
        .resolve_wire_type(
            callee.r#type.as_ref().ok_or_else(|| {
                HirLoweringError::InvalidQCore("function candidate has no type".to_owned())
            })?,
            scope,
        )
        .map_err(HirLoweringError::InvalidQCore)?;
    let argument_types = arguments
        .iter()
        .map(|argument| {
            arena
                .resolve_wire_type(
                    argument.r#type.as_ref().ok_or_else(|| {
                        HirLoweringError::InvalidQCore("function argument has no type".to_owned())
                    })?,
                    scope,
                )
                .map_err(HirLoweringError::InvalidQCore)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let (positional_count, keyword_names) = invocation_syntax(invocation, arguments.len())?;
    let application = bind_function_application(
        arena,
        callee_type,
        &argument_types,
        positional_count,
        &keyword_names,
    )
    .map_err(HirLoweringError::InvalidQCore)?;
    let ordered_indices = application.ordered_indices().to_vec().into_boxed_slice();
    let result_type = arena
        .encode_wire_type(application.result_type())
        .map(Some)
        .map_err(HirLoweringError::InvalidQCore)?;
    Ok((ordered_indices, result_type))
}

fn validate_callable_application(
    callee: &wire::HirExpressionProto,
    arguments: &[wire::HirExpressionProto],
    invocation: &wire::HirExpressionProto,
    scope: &std::sync::Arc<str>,
    arena: &mut TypeArena,
) -> Result<(), HirLoweringError> {
    let callee_type = arena
        .resolve_wire_type(
            callee.r#type.as_ref().ok_or_else(|| {
                HirLoweringError::InvalidQCore("callable expression has no type".to_owned())
            })?,
            scope,
        )
        .map_err(HirLoweringError::InvalidQCore)?;
    let argument_types = arguments
        .iter()
        .map(|argument| {
            arena
                .resolve_wire_type(
                    argument.r#type.as_ref().ok_or_else(|| {
                        HirLoweringError::InvalidQCore("invoke argument has no type".to_owned())
                    })?,
                    scope,
                )
                .map_err(HirLoweringError::InvalidQCore)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let result_type = arena
        .resolve_wire_type(
            invocation.r#type.as_ref().ok_or_else(|| {
                HirLoweringError::InvalidQCore("invoke result has no type".to_owned())
            })?,
            scope,
        )
        .map_err(HirLoweringError::InvalidQCore)?;
    let (positional_count, keyword_names) = invocation_syntax(invocation, arguments.len())?;
    let expected_argument_count = match arena.get(arena.strip_stage(callee_type)) {
        Some(SemanticType::Function { inputs, .. }) => inputs.len(),
        _ => {
            return Err(HirLoweringError::InvalidQCore(
                "invoke callee does not have a function type".to_owned(),
            ));
        }
    };
    let binding = bind_function_application(
        arena,
        callee_type,
        &argument_types,
        positional_count,
        &keyword_names,
    )
    .map_err(|_| {
        HirLoweringError::InvalidQCore(
            if expected_argument_count == arguments.len() {
                "invoke does not match its function signature"
            } else {
                "invoke does not match its function arity"
            }
            .to_owned(),
        )
    })?;
    if binding.result_type() != result_type {
        return Err(HirLoweringError::InvalidQCore(
            "invoke does not match its function signature".to_owned(),
        ));
    }
    Ok(())
}

fn invocation_syntax(
    invocation: &wire::HirExpressionProto,
    argument_count: usize,
) -> Result<(usize, Vec<String>), HirLoweringError> {
    let binding_attributes = invocation
        .attributes
        .iter()
        .filter(|attribute| attribute.name != "evaluation_order")
        .collect::<Vec<_>>();
    if binding_attributes.is_empty() {
        return Ok((argument_count, Vec::new()));
    }
    if binding_attributes.len() != 2 {
        return Err(HirLoweringError::InvalidQCore(
            "invoke has an invalid argument-binding attribute row".to_owned(),
        ));
    }
    let keyword_names = string_tuple_attribute(&invocation.attributes, "keyword_names")?;
    let positional_count = nonnegative_usize_attribute(&invocation.attributes, "positional_count")?;
    if positional_count > argument_count || keyword_names.len() != argument_count - positional_count
    {
        return Err(HirLoweringError::InvalidQCore(
            "invoke argument-binding metadata disagrees with its operands".to_owned(),
        ));
    }
    Ok((positional_count, keyword_names))
}

fn callable_candidate_count(
    expression: &wire::HirExpressionProto,
) -> Result<usize, HirLoweringError> {
    let mut pending = vec![expression];
    let mut count = 0_usize;
    while let Some(candidate) = pending.pop() {
        match candidate.opcode.as_str() {
            "function_ref" => {
                count = count.checked_add(1).ok_or_else(|| {
                    HirLoweringError::Unsupported("function candidate count overflow".to_owned())
                })?;
            }
            "lazy_if" if candidate.regions.len() == 2 => {
                pending.extend(candidate.regions.iter());
            }
            opcode => {
                return Err(HirLoweringError::Unsupported(format!(
                    "callable expression {opcode:?}"
                )));
            }
        }
    }
    Ok(count)
}

#[allow(clippy::too_many_arguments)]
fn lower_lazy_if_expression(
    expression: &wire::HirExpressionProto,
    scope: &std::sync::Arc<str>,
    arena: &mut TypeArena,
    next_id: &mut u64,
    next_region: &mut u64,
    region_children: &mut RegionChildren,
    environment: &HashMap<String, wire::ValueProto>,
    operations: &mut Vec<wire::OperationProto>,
    operands: &[wire::ValueProto],
    work: &mut HirSsaLoweringWork,
) -> Result<wire::ValueProto, HirLoweringError> {
    let [condition] = operands else {
        return Err(HirLoweringError::InvalidQCore(
            "lazy_if does not have one condition".to_owned(),
        ));
    };
    let capture_names = string_tuple_attribute(&expression.attributes, "captures")?;
    let capture_values = capture_names
        .iter()
        .map(|name| {
            environment.get(name).cloned().ok_or_else(|| {
                HirLoweringError::InvalidQCore(format!("unresolved lazy_if capture {name:?}"))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let result_type = expression
        .r#type
        .clone()
        .ok_or_else(|| HirLoweringError::InvalidQCore("lazy_if has no result type".to_owned()))?;
    let mut regions = Vec::with_capacity(2);
    let mut branch_values = Vec::with_capacity(2);
    for branch_expression in &expression.regions {
        let arguments = capture_values
            .iter()
            .zip(&capture_names)
            .map(|(value, name)| clone_region_argument(value, name, next_id))
            .collect::<Result<Vec<_>, _>>()?;
        let branch_environment = capture_names
            .iter()
            .cloned()
            .zip(arguments.iter().cloned())
            .collect();
        let mut branch_operations = Vec::new();
        let current = lower_expression(
            branch_expression,
            scope,
            arena,
            next_id,
            next_region,
            region_children,
            &branch_environment,
            &mut branch_operations,
            work,
        )?;
        let yielded = lift_branch_value(
            current,
            &result_type,
            "if",
            expression.span.clone(),
            next_id,
            &mut branch_operations,
        )?;
        let terminator = wire::TerminatorProto {
            kind: "yield".to_owned(),
            operands: vec![yielded.id],
            span: expression.span.clone(),
            ..Default::default()
        };
        regions.push(wire::RegionProto {
            blocks: vec![wire::BlockProto {
                arguments,
                operations: schedule_dead_results(branch_operations, &terminator),
                terminator: Some(terminator),
            }],
        });
        branch_values.push(yielded);
    }
    align_fresh_branch_result_regions(&mut regions, &mut branch_values, &capture_values);
    let first = branch_values.first().ok_or_else(|| {
        HirLoweringError::InvalidQCore("lazy_if does not have a branch".to_owned())
    })?;
    let result = new_value(result_type, "if".to_owned(), first.region.clone(), next_id)?;
    let condition_type = condition.r#type.as_ref().ok_or_else(|| {
        HirLoweringError::InvalidQCore("lazy_if condition has no type".to_owned())
    })?;
    let condition_id = arena
        .resolve_wire_type(condition_type, scope)
        .map_err(HirLoweringError::InvalidQCore)?;
    let mut operand_ids = Vec::with_capacity(capture_values.len() + 1);
    operand_ids.push(condition.id);
    operand_ids.extend(capture_values.iter().map(|value| value.id));
    operations.push(wire::OperationProto {
        opcode: "qcore.if".to_owned(),
        operands: operand_ids,
        results: vec![result.clone()],
        attributes: structured_if_attributes(
            &capture_names,
            &["if".to_owned()],
            control_stage(arena.stage_of(condition_id)),
        ),
        regions,
        span: expression.span.clone(),
        intent: Vec::new(),
    });
    Ok(result)
}

fn align_fresh_branch_result_regions(
    regions: &mut [wire::RegionProto],
    branch_values: &mut [wire::ValueProto],
    captures: &[wire::ValueProto],
) {
    let Some(canonical) = branch_values
        .first()
        .and_then(|value| value.region.as_ref())
        .cloned()
    else {
        return;
    };
    let captured_roots = captures
        .iter()
        .filter_map(|value| value.region.as_ref().map(|region| region.root))
        .collect::<HashSet<_>>();
    if captured_roots.contains(&canonical.root) {
        return;
    }
    for (region, value) in regions.iter_mut().zip(branch_values.iter_mut()) {
        let Some(observed) = value.region.as_ref() else {
            return;
        };
        if observed.projections != canonical.projections || captured_roots.contains(&observed.root)
        {
            return;
        }
        let observed_root = observed.root;
        remap_region_root(region, observed_root, canonical.root);
        if let Some(value_region) = &mut value.region {
            value_region.root = canonical.root;
        }
    }
}

fn remap_region_root(region: &mut wire::RegionProto, source: u64, target: u64) {
    for block in &mut region.blocks {
        for argument in &mut block.arguments {
            remap_value_region_root(argument, source, target);
        }
        for operation in &mut block.operations {
            for result in &mut operation.results {
                remap_value_region_root(result, source, target);
            }
            for nested in &mut operation.regions {
                remap_region_root(nested, source, target);
            }
        }
    }
}

fn remap_value_region_root(value: &mut wire::ValueProto, source: u64, target: u64) {
    if let Some(region) = &mut value.region
        && region.root == source
    {
        region.root = target;
    }
}

fn lift_branch_value(
    current: wire::ValueProto,
    expected_type: &wire::TypeProto,
    hint: &str,
    span: Option<wire::SourceSpanProto>,
    next_id: &mut u64,
    operations: &mut Vec<wire::OperationProto>,
) -> Result<wire::ValueProto, HirLoweringError> {
    if current.r#type.as_ref() == Some(expected_type) {
        return Ok(current);
    }
    let lifted = new_value(expected_type.clone(), hint.to_owned(), None, next_id)?;
    operations.push(wire::OperationProto {
        opcode: "qcore.stage_lift".to_owned(),
        operands: vec![current.id],
        results: vec![lifted.clone()],
        span,
        ..Default::default()
    });
    Ok(lifted)
}

fn lower_result_region(
    expression: &wire::HirExpressionProto,
    opcode: &str,
    is_quantum: bool,
    operands: &[wire::ValueProto],
    next_region: &mut u64,
    region_children: &RegionChildren,
) -> Result<Option<wire::RegionRefProto>, HirLoweringError> {
    if !is_quantum {
        return Ok(None);
    }
    let owned_operands = operands.iter().filter(|operand| operand.has_region).count();
    if matches!(
        opcode,
        "qcore.tuple" | "qcore.call" | "qcore.control_call" | "qcore.select_call"
    ) && owned_operands > 1
        && let Some(parent) = owned_sibling_parent(operands, region_children)
    {
        return Ok(Some(parent));
    }
    if matches!(
        opcode,
        "quantum.alloc" | "quantum.prepare" | "qcore.control_call" | "qcore.select_call"
    ) || (opcode == "qcore.call" && owned_operands != 1)
        || (opcode == "qcore.tuple" && owned_operands > 1)
    {
        let root = *next_region;
        *next_region = next_region
            .checked_add(1)
            .ok_or_else(|| HirLoweringError::Unsupported("ownership region capacity".to_owned()))?;
        return Ok(Some(wire::RegionRefProto {
            root,
            ..Default::default()
        }));
    }
    let mut regions = operands.iter().filter_map(|operand| operand.region.clone());
    let region = regions.next().ok_or_else(|| {
        HirLoweringError::Unsupported(format!(
            "quantum result without an owned operand in {:?}",
            expression.opcode
        ))
    })?;
    if regions.any(|candidate| candidate != region) {
        return Err(HirLoweringError::Unsupported(format!(
            "multiple owned operands in {:?}",
            expression.opcode
        )));
    }
    Ok(Some(region))
}

fn owned_sibling_parent(
    operands: &[wire::ValueProto],
    region_children: &RegionChildren,
) -> Option<wire::RegionRefProto> {
    let owned = operands
        .iter()
        .filter_map(|operand| operand.region.as_ref())
        .collect::<Vec<_>>();
    let first = *owned.first()?;
    if owned.len() < 2 {
        return None;
    }
    let (_, parent) = first.projections.split_last()?;
    let mut components = HashSet::with_capacity(owned.len());
    for (index, region) in owned.into_iter().enumerate() {
        let (projection, candidate_parent) = region.projections.split_last()?;
        if region.root != first.root
            || candidate_parent != parent
            || projection != &index.to_string()
        {
            return None;
        }
        components.insert(projection.clone());
    }
    if region_children.get(&(first.root, parent.to_vec())) != Some(&components) {
        return None;
    }
    Some(wire::RegionRefProto {
        root: first.root,
        projections: parent.to_vec(),
    })
}

fn string_tuple_attribute(
    attributes: &[wire::AttributeProto],
    name: &str,
) -> Result<Vec<String>, HirLoweringError> {
    let values = attributes
        .iter()
        .find(|attribute| attribute.name == name)
        .and_then(|attribute| attribute.value.as_ref())
        .and_then(|literal| literal.value.as_ref())
        .and_then(|value| match value {
            wire::literal_proto::Value::TupleValue(sequence) => Some(&sequence.values),
            _ => None,
        })
        .ok_or_else(|| {
            HirLoweringError::InvalidQCore(format!("missing string tuple attribute {name:?}"))
        })?;
    values
        .iter()
        .map(|literal| match literal.value.as_ref() {
            Some(wire::literal_proto::Value::StringValue(value)) if !value.is_empty() => {
                Ok(value.clone())
            }
            _ => Err(HirLoweringError::InvalidQCore(format!(
                "invalid string tuple attribute {name:?}"
            ))),
        })
        .collect()
}

fn evaluation_order_attribute(attributes: &[wire::AttributeProto]) -> Option<Vec<usize>> {
    let values = attributes
        .iter()
        .find(|attribute| attribute.name == "evaluation_order")?
        .value
        .as_ref()?
        .value
        .as_ref()
        .and_then(|value| match value {
            wire::literal_proto::Value::TupleValue(sequence) => Some(&sequence.values),
            _ => None,
        })?;
    values
        .iter()
        .map(|literal| match literal.value.as_ref() {
            Some(wire::literal_proto::Value::IntValue(value)) => usize::try_from(*value).ok(),
            _ => None,
        })
        .collect()
}

fn evaluation_order_hir_attribute(
    values: &[usize],
) -> Result<wire::AttributeProto, HirLoweringError> {
    let values = values
        .iter()
        .map(|value| {
            i64::try_from(*value)
                .map_err(|_| HirLoweringError::Unsupported("HIR operand index".to_owned()))
                .map(|value| wire::LiteralProto {
                    value: Some(wire::literal_proto::Value::IntValue(value)),
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(wire::AttributeProto {
        name: "evaluation_order".to_owned(),
        value: Some(wire::LiteralProto {
            value: Some(wire::literal_proto::Value::TupleValue(
                wire::literal_proto::Sequence { values },
            )),
        }),
    })
}

fn append_nonidentity_evaluation_order(
    attributes: &mut Vec<wire::AttributeProto>,
    order: &[usize],
) -> Result<(), HirLoweringError> {
    if order.iter().copied().eq(0..order.len()) {
        return Ok(());
    }
    attributes.push(evaluation_order_hir_attribute(order)?);
    attributes.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(())
}

fn expression_evaluation_order(
    expression: &wire::HirExpressionProto,
) -> Result<Vec<usize>, HirLoweringError> {
    let Some(order) = evaluation_order_attribute(&expression.attributes) else {
        if expression
            .attributes
            .iter()
            .any(|attribute| attribute.name == "evaluation_order")
        {
            return Err(HirLoweringError::InvalidQCore(
                "invalid expression evaluation order".to_owned(),
            ));
        }
        return Ok((0..expression.operands.len()).collect());
    };
    if !valid_evaluation_order(&order, expression.operands.len()) {
        return Err(HirLoweringError::InvalidQCore(
            "expression evaluation order disagrees with its operands".to_owned(),
        ));
    }
    Ok(order)
}

fn valid_evaluation_order(order: &[usize], operand_count: usize) -> bool {
    if order.len() != operand_count {
        return false;
    }
    let mut seen = vec![false; operand_count];
    for index in order {
        let Some(slot) = seen.get_mut(*index) else {
            return false;
        };
        if *slot {
            return false;
        }
        *slot = true;
    }
    true
}

fn nonnegative_usize_attribute(
    attributes: &[wire::AttributeProto],
    name: &str,
) -> Result<usize, HirLoweringError> {
    let value = attributes
        .iter()
        .find(|attribute| attribute.name == name)
        .and_then(|attribute| attribute.value.as_ref())
        .and_then(|literal| literal.value.as_ref())
        .and_then(|value| match value {
            wire::literal_proto::Value::IntValue(value) => usize::try_from(*value).ok(),
            _ => None,
        })
        .ok_or_else(|| {
            HirLoweringError::InvalidQCore(format!(
                "missing nonnegative integer attribute {name:?}"
            ))
        })?;
    Ok(value)
}

fn optional_integer_attribute(
    attributes: &[wire::AttributeProto],
    name: &str,
) -> Result<Option<BigInt>, HirLoweringError> {
    let value = attributes
        .iter()
        .find(|attribute| attribute.name == name)
        .and_then(|attribute| attribute.value.as_ref())
        .and_then(|literal| literal.value.as_ref())
        .ok_or_else(|| {
            HirLoweringError::InvalidQCore(format!("missing optional integer attribute {name:?}"))
        })?;
    match value {
        wire::literal_proto::Value::UnitValue(_) => Ok(None),
        wire::literal_proto::Value::IntValue(value) => Ok(Some(BigInt::from(*value))),
        wire::literal_proto::Value::BigintValue(value) => {
            Ok(Some(BigInt::from_signed_bytes_be(value)))
        }
        _ => Err(HirLoweringError::InvalidQCore(format!(
            "invalid optional integer attribute {name:?}"
        ))),
    }
}

fn replace_string_tuple_attribute(
    attributes: &mut [wire::AttributeProto],
    name: &str,
    values: &[String],
) -> Result<(), HirLoweringError> {
    let attribute = attributes
        .iter_mut()
        .find(|attribute| attribute.name == name)
        .ok_or_else(|| {
            HirLoweringError::InvalidQCore(format!("missing string tuple attribute {name:?}"))
        })?;
    *attribute = string_tuple_qcore_attribute(name, values);
    Ok(())
}

fn bool_attribute(
    attributes: &[wire::AttributeProto],
    name: &str,
) -> Result<bool, HirLoweringError> {
    attributes
        .iter()
        .find(|attribute| attribute.name == name)
        .and_then(|attribute| attribute.value.as_ref())
        .and_then(|literal| literal.value.as_ref())
        .and_then(|value| match value {
            wire::literal_proto::Value::BoolValue(value) => Some(*value),
            _ => None,
        })
        .ok_or_else(|| HirLoweringError::InvalidQCore(format!("missing bool attribute {name:?}")))
}

fn structured_if_attributes(
    captures: &[String],
    outputs: &[String],
    stage: &str,
) -> Vec<wire::AttributeProto> {
    vec![
        string_tuple_qcore_attribute("capture_names", captures),
        wire::AttributeProto {
            name: "condition_index".to_owned(),
            value: Some(wire::LiteralProto {
                value: Some(wire::literal_proto::Value::IntValue(0)),
            }),
        },
        string_tuple_qcore_attribute("output_names", outputs),
        wire::AttributeProto {
            name: "stage".to_owned(),
            value: Some(wire::LiteralProto {
                value: Some(wire::literal_proto::Value::StringValue(stage.to_owned())),
            }),
        },
    ]
}

fn structured_for_attributes(
    captures: &[String],
    induction_name: &str,
    stage: &str,
) -> Vec<wire::AttributeProto> {
    vec![
        string_tuple_qcore_attribute("capture_names", captures),
        wire::AttributeProto {
            name: "induction_name".to_owned(),
            value: Some(wire::LiteralProto {
                value: Some(wire::literal_proto::Value::StringValue(
                    induction_name.to_owned(),
                )),
            }),
        },
        wire::AttributeProto {
            name: "stage".to_owned(),
            value: Some(wire::LiteralProto {
                value: Some(wire::literal_proto::Value::StringValue(stage.to_owned())),
            }),
        },
    ]
}

fn string_tuple_qcore_attribute(name: &str, values: &[String]) -> wire::AttributeProto {
    wire::AttributeProto {
        name: name.to_owned(),
        value: Some(wire::LiteralProto {
            value: Some(wire::literal_proto::Value::TupleValue(
                wire::literal_proto::Sequence {
                    values: values
                        .iter()
                        .map(|value| wire::LiteralProto {
                            value: Some(wire::literal_proto::Value::StringValue(value.clone())),
                        })
                        .collect(),
                },
            )),
        }),
    }
}

fn control_stage(stage: Option<Stage>) -> &'static str {
    match stage {
        None => "Value",
        Some(stage) => stage.as_str(),
    }
}

fn string_attribute<'a>(
    attributes: &'a [wire::AttributeProto],
    name: &str,
) -> Result<&'a str, HirLoweringError> {
    attributes
        .iter()
        .find(|attribute| attribute.name == name)
        .and_then(|attribute| attribute.value.as_ref())
        .and_then(|literal| literal.value.as_ref())
        .and_then(|value| match value {
            wire::literal_proto::Value::StringValue(value) => Some(value.as_str()),
            _ => None,
        })
        .ok_or_else(|| HirLoweringError::InvalidQCore(format!("missing string attribute {name:?}")))
}

fn validate(
    package: &wire::HirPackageProto,
    work: &mut HirValidationWork,
) -> Result<HirHeader, HirError> {
    work.charge_semantic_nodes(package.functions.len() + package.constants.len());
    if package.module_name.is_empty()
        || !strictly_sorted(package.functions.iter().map(|value| value.symbol.as_str()))
        || !strictly_sorted(package.constants.iter().map(|value| value.name.as_str()))
    {
        return Err(HirError::NonCanonicalEncoding);
    }
    let mut arena = TypeArena::default();
    let mut literals = Vec::new();
    let constant_scope = arena.intern_scope(&format!("{}::<constants>", package.module_name));
    for constant in &package.constants {
        work.charge_semantic_nodes(1);
        if constant.name.is_empty() {
            return Err(HirError::InvalidStructure(
                "constant name is empty".to_owned(),
            ));
        }
        resolve_type(
            constant.r#type.as_ref(),
            &constant_scope,
            &mut arena,
            "module constant",
            work,
        )?;
        work.charge_semantic_nodes(1);
        literals.push(required(
            constant.value.as_ref(),
            "module constant has no value",
        )?);
    }

    let declarations = package
        .functions
        .iter()
        .map(|function| (function.symbol.as_str(), function))
        .collect::<HashMap<_, _>>();
    work.charge_semantic_nodes(package.functions.len());
    for function in &package.functions {
        validate_function(function, &declarations, &mut arena, &mut literals, work)?;
    }
    work.charge_semantic_nodes(literals.len());
    literals::validate_literal_references(literals).map_err(HirError::InvalidPortableLiteral)?;
    Ok(HirHeader {
        module_name: package.module_name.clone(),
        function_count: package.functions.len(),
        constant_count: package.constants.len(),
        expression_count: work.expression_count,
        statement_count: work.statement_count,
    })
}

fn validate_function<'a>(
    function: &'a wire::HirFunctionProto,
    declarations: &HashMap<&str, &'a wire::HirFunctionProto>,
    arena: &mut TypeArena,
    literals: &mut Vec<&'a wire::LiteralProto>,
    work: &mut HirValidationWork,
) -> Result<(), HirError> {
    work.charge_semantic_nodes(1);
    if function.symbol.is_empty()
        || !valid_span(function.span.as_ref())
        || !matches!(
            function.declared_kind.as_str(),
            "" | "classical" | "unitary" | "isometry" | "channel" | "instrument"
        )
    {
        return Err(HirError::InvalidStructure(format!(
            "invalid function declaration {:?}",
            function.symbol
        )));
    }
    if function.external && function.declared_kind.is_empty() {
        return Err(HirError::InvalidStructure(format!(
            "external contract mismatch in {:?}",
            function.symbol
        )));
    }
    let scope = arena.intern_scope(&function.symbol);
    resolve_type(
        function.result_type.as_ref(),
        &scope,
        arena,
        "function result",
        work,
    )?;
    let mut parameter_names = HashSet::new();
    for parameter in &function.parameters {
        work.charge_semantic_nodes(1);
        if parameter.name.is_empty() || !parameter_names.insert(parameter.name.as_str()) {
            return Err(HirError::InvalidStructure(format!(
                "duplicate or empty parameter in {:?}",
                function.symbol
            )));
        }
        resolve_type(
            parameter.r#type.as_ref(),
            &scope,
            arena,
            "function parameter",
            work,
        )?;
        match (parameter.has_default, parameter.default_value.as_ref()) {
            (true, Some(value)) => {
                work.charge_semantic_nodes(1);
                literals.push(value);
            }
            (true, None) | (false, Some(_)) => {
                return Err(HirError::InvalidStructure(format!(
                    "default marker mismatch in {:?}",
                    function.symbol
                )));
            }
            (false, None) => {}
        }
    }
    let body = required(function.body.as_ref(), "function has no HIR body")?;
    if function.external && !body.statements.is_empty() {
        return Err(HirError::InvalidStructure(format!(
            "external function {:?} has a body",
            function.symbol
        )));
    }
    validate_block(body, declarations, &scope, arena, literals, work)
}

fn validate_block<'a>(
    root: &'a wire::HirBlockProto,
    declarations: &HashMap<&str, &'a wire::HirFunctionProto>,
    scope: &std::sync::Arc<str>,
    arena: &mut TypeArena,
    literals: &mut Vec<&'a wire::LiteralProto>,
    work: &mut HirValidationWork,
) -> Result<(), HirError> {
    let mut blocks = vec![root];
    let mut expressions = Vec::new();
    while let Some(block) = blocks.pop() {
        for statement in &block.statements {
            work.statement_count = work.statement_count.saturating_add(1);
            work.charge_semantic_nodes(1);
            validate_statement_shape(statement)?;
            validate_attributes(&statement.attributes, literals, work)?;
            if !valid_span(statement.span.as_ref()) {
                return Err(HirError::InvalidStructure(
                    "statement has an invalid source span".to_owned(),
                ));
            }
            for result_type in &statement.result_types {
                work.charge_semantic_nodes(1);
                arena
                    .resolve_wire_type(result_type, scope)
                    .map_err(HirError::InvalidSemanticType)?;
            }
            if let Some(expression) = &statement.expression {
                expressions.push(expression);
            }
            blocks.extend(&statement.regions);
        }
    }
    while let Some(expression) = expressions.pop() {
        work.expression_count = work.expression_count.saturating_add(1);
        work.charge_semantic_nodes(1);
        if expression.opcode.is_empty() || !valid_span(expression.span.as_ref()) {
            return Err(HirError::InvalidStructure(
                "expression has an empty opcode or invalid source span".to_owned(),
            ));
        }
        work.charge_semantic_nodes(1);
        arena
            .resolve_wire_type(
                required(
                    expression.r#type.as_ref(),
                    "expression has no semantic type",
                )?,
                scope,
            )
            .map_err(HirError::InvalidSemanticType)?;
        validate_attributes(&expression.attributes, literals, work)?;
        validate_evaluation_order_shape(expression)?;
        if expression.opcode == "function_ref" {
            validate_function_reference(expression, declarations)?;
        }
        if matches!(expression.opcode.as_str(), "invoke" | "adjoint_invoke") {
            validate_invoke_shape(expression)?;
        }
        if expression.opcode == "lazy_if" {
            if expression.regions.len() != 2 {
                return Err(HirError::InvalidStructure(
                    "lazy_if expression does not have two arms".to_owned(),
                ));
            }
        } else if !expression.regions.is_empty() {
            return Err(HirError::InvalidStructure(format!(
                "expression {:?} carries unexpected regions",
                expression.opcode
            )));
        }
        expressions.extend(&expression.operands);
        expressions.extend(&expression.regions);
    }
    Ok(())
}

fn validate_function_reference(
    expression: &wire::HirExpressionProto,
    declarations: &HashMap<&str, &wire::HirFunctionProto>,
) -> Result<(), HirError> {
    let callee = match expression.attributes.as_slice() {
        [attribute] if attribute.name == "callee" => attribute
            .value
            .as_ref()
            .and_then(|literal| literal.value.as_ref())
            .and_then(|value| match value {
                wire::literal_proto::Value::StringValue(value) if !value.is_empty() => {
                    Some(value.as_str())
                }
                _ => None,
            }),
        _ => None,
    }
    .ok_or_else(|| {
        HirError::InvalidStructure("function_ref has an invalid callee attribute".to_owned())
    })?;
    let declaration = declarations.get(callee).ok_or_else(|| {
        HirError::InvalidStructure(format!("function_ref names unknown declaration {callee:?}"))
    })?;
    let expected = wire::TypeProto {
        kind: wire::type_proto::Kind::Function.into(),
        arguments: declaration
            .parameters
            .iter()
            .map(|parameter| {
                required(
                    parameter.r#type.as_ref(),
                    "referenced function parameter has no type",
                )
                .cloned()
            })
            .chain(std::iter::once(
                required(
                    declaration.result_type.as_ref(),
                    "referenced function has no result type",
                )
                .cloned(),
            ))
            .collect::<Result<Vec<_>, _>>()?,
        field_names: declaration
            .parameters
            .iter()
            .map(|parameter| parameter.name.clone())
            .collect(),
        ..Default::default()
    };
    if expression.r#type.as_ref() != Some(&expected) {
        return Err(HirError::InvalidStructure(format!(
            "function_ref type disagrees with declaration {callee:?}"
        )));
    }
    Ok(())
}

fn validate_invoke_shape(expression: &wire::HirExpressionProto) -> Result<(), HirError> {
    if expression.operands.is_empty() {
        return Err(HirError::InvalidStructure(
            "invoke has no callable operand".to_owned(),
        ));
    }
    let binding_attributes = expression
        .attributes
        .iter()
        .filter(|attribute| attribute.name != "evaluation_order")
        .collect::<Vec<_>>();
    if binding_attributes.is_empty() {
        return Ok(());
    }
    let [keywords, positional] = binding_attributes.as_slice() else {
        return Err(HirError::InvalidStructure(
            "invoke has an invalid argument-binding attribute row".to_owned(),
        ));
    };
    if keywords.name != "keyword_names" || positional.name != "positional_count" {
        return Err(HirError::InvalidStructure(
            "invoke has unknown argument-binding attributes".to_owned(),
        ));
    }
    let keyword_count = keywords
        .value
        .as_ref()
        .and_then(|literal| literal.value.as_ref())
        .and_then(|value| match value {
            wire::literal_proto::Value::TupleValue(sequence)
                if sequence.values.iter().all(|value| {
                    matches!(
                        value.value.as_ref(),
                        Some(wire::literal_proto::Value::StringValue(name)) if !name.is_empty()
                    )
                }) =>
            {
                Some(sequence.values.len())
            }
            _ => None,
        });
    let positional_count = positional
        .value
        .as_ref()
        .and_then(|literal| literal.value.as_ref())
        .and_then(|value| match value {
            wire::literal_proto::Value::IntValue(value) => usize::try_from(*value).ok(),
            _ => None,
        });
    let argument_count = expression.operands.len() - 1;
    if positional_count
        .zip(keyword_count)
        .is_none_or(|(positional, keywords)| {
            positional > argument_count || keywords != argument_count - positional
        })
    {
        return Err(HirError::InvalidStructure(
            "invoke argument-binding metadata disagrees with its operands".to_owned(),
        ));
    }
    Ok(())
}

fn validate_evaluation_order_shape(expression: &wire::HirExpressionProto) -> Result<(), HirError> {
    let has_order = expression
        .attributes
        .iter()
        .any(|attribute| attribute.name == "evaluation_order");
    if !has_order {
        return Ok(());
    }
    if !matches!(
        expression.opcode.as_str(),
        "call"
            | "adjoint_call"
            | "control_call"
            | "control_invoke"
            | "select_call"
            | "record"
            | "invoke"
            | "adjoint_invoke"
    ) {
        return Err(HirError::InvalidStructure(format!(
            "expression {:?} carries unexpected evaluation order",
            expression.opcode
        )));
    }
    let order = evaluation_order_attribute(&expression.attributes).ok_or_else(|| {
        HirError::InvalidStructure("expression has an invalid evaluation order".to_owned())
    })?;
    if !valid_evaluation_order(&order, expression.operands.len()) {
        return Err(HirError::InvalidStructure(
            "expression evaluation order is not an operand permutation".to_owned(),
        ));
    }
    if matches!(
        expression.opcode.as_str(),
        "invoke" | "adjoint_invoke" | "control_invoke"
    ) && order.first() != Some(&0)
    {
        return Err(HirError::InvalidStructure(
            "invoke must evaluate its callable before its arguments".to_owned(),
        ));
    }
    Ok(())
}

fn validate_statement_shape(statement: &wire::HirStatementProto) -> Result<(), HirError> {
    let has_expression = statement.expression.is_some();
    let valid = match statement.opcode.as_str() {
        "assign" => {
            has_expression
                && !statement.targets.is_empty()
                && statement.regions.is_empty()
                && ((statement.targets.len() == 1 && statement.result_types.is_empty())
                    || (!statement.result_types.is_empty()
                        && statement.result_types.len() == statement.targets.len()))
        }
        "expression" | "return" => {
            has_expression
                && statement.targets.is_empty()
                && statement.regions.is_empty()
                && statement.result_types.is_empty()
        }
        "break" | "continue" => {
            !has_expression
                && statement.targets.is_empty()
                && statement.attributes.is_empty()
                && statement.regions.is_empty()
                && statement.result_types.is_empty()
        }
        "if" | "while" => {
            has_expression && statement.targets.is_empty() && statement.regions.len() == 2
        }
        "for" => has_expression && !statement.targets.is_empty() && statement.regions.len() == 2,
        _ => false,
    };
    if !valid {
        return Err(HirError::InvalidStructure(format!(
            "invalid statement shape for {:?}",
            statement.opcode
        )));
    }
    if statement.targets.iter().any(String::is_empty) {
        return Err(HirError::InvalidStructure(
            "statement target is empty".to_owned(),
        ));
    }
    Ok(())
}

fn validate_attributes<'a>(
    attributes: &'a [wire::AttributeProto],
    literals: &mut Vec<&'a wire::LiteralProto>,
    work: &mut HirValidationWork,
) -> Result<(), HirError> {
    work.charge_semantic_nodes(attributes.len());
    if !strictly_sorted(attributes.iter().map(|value| value.name.as_str())) {
        return Err(HirError::NonCanonicalEncoding);
    }
    for attribute in attributes {
        literals.push(required(
            attribute.value.as_ref(),
            "HIR attribute has no value",
        )?);
    }
    Ok(())
}

fn resolve_type(
    source: Option<&wire::TypeProto>,
    scope: &std::sync::Arc<str>,
    arena: &mut TypeArena,
    context: &str,
    work: &mut HirValidationWork,
) -> Result<(), HirError> {
    work.charge_semantic_nodes(1);
    arena
        .resolve_wire_type(
            required(source, &format!("{context} has no semantic type"))?,
            scope,
        )
        .map(|_| ())
        .map_err(HirError::InvalidSemanticType)
}

fn strictly_sorted<'a>(values: impl Iterator<Item = &'a str>) -> bool {
    let mut previous = None;
    for value in values {
        if value.is_empty() || previous.is_some_and(|item| item >= value) {
            return false;
        }
        previous = Some(value);
    }
    true
}

fn valid_span(span: Option<&wire::SourceSpanProto>) -> bool {
    span.is_some_and(|value| !value.uri.is_empty() && value.start_line != 0 && value.end_line != 0)
}

fn required<'a, T>(value: Option<&'a T>, reason: &str) -> Result<&'a T, HirError> {
    value.ok_or_else(|| HirError::InvalidStructure(reason.to_owned()))
}

#[cfg(test)]
mod persistence_tests {
    use super::*;

    fn unsealed_hir() -> wire::HirPackageProto {
        wire::HirPackageProto {
            schema_major: SCHEMA_MAJOR,
            schema_minor: SCHEMA_MINOR,
            artifact_kind: HIR_KIND.to_owned(),
            module_name: "native_hir".to_owned(),
            ..Default::default()
        }
    }

    #[test]
    fn reordered_owned_children_do_not_reuse_their_old_parent_region() {
        let child = |index: &str| wire::ValueProto {
            has_region: true,
            region: Some(wire::RegionRefProto {
                root: 7,
                projections: vec![index.to_owned()],
            }),
            ..Default::default()
        };
        let children = HashMap::from([(
            (7, Vec::new()),
            HashSet::from(["0".to_owned(), "1".to_owned()]),
        )]);
        let ordered = [child("0"), child("1")];
        let reversed = [ordered[1].clone(), ordered[0].clone()];
        assert_eq!(
            owned_sibling_parent(&ordered, &children),
            Some(wire::RegionRefProto {
                root: 7,
                projections: Vec::new(),
            })
        );
        assert_eq!(owned_sibling_parent(&reversed, &children), None);
    }

    #[test]
    fn seals_and_inspects_one_canonical_hir_request() {
        let sealed = seal(&unsealed_hir().encode_to_vec()).expect("native HIR seal");
        let (header, work) = inspect_with_work(&sealed).expect("native HIR inspect");

        assert_eq!(header.module_name, "native_hir");
        assert_eq!(header.function_count, 0);
        assert_eq!(header.constant_count, 0);
        assert_eq!(work.input_bytes(), sealed.len());
        assert!(work.preflight_message_nodes() > 0);
        assert_eq!(work.semantic_node_visits(), 0);
        assert!(matches!(seal(&sealed), Err(HirError::NonCanonicalEncoding)));
    }

    #[test]
    fn hir_sealing_rejects_an_unknown_kind() {
        let mut package = unsealed_hir();
        package.artifact_kind = "future.hir".to_owned();

        assert!(matches!(
            seal(&package.encode_to_vec()),
            Err(HirError::UnexpectedArtifactKind { .. })
        ));
    }

    fn callable_tree(leaves: usize) -> wire::HirExpressionProto {
        if leaves == 1 {
            return wire::HirExpressionProto {
                opcode: "function_ref".to_owned(),
                ..Default::default()
            };
        }
        let left = leaves / 2;
        wire::HirExpressionProto {
            opcode: "lazy_if".to_owned(),
            regions: vec![callable_tree(left), callable_tree(leaves - left)],
            ..Default::default()
        }
    }

    #[test]
    fn bounds_static_function_value_candidates_before_expansion() {
        assert_eq!(
            callable_candidate_count(&callable_tree(MAX_FUNCTION_VALUE_CANDIDATES)),
            Ok(MAX_FUNCTION_VALUE_CANDIDATES)
        );
        assert!(
            callable_candidate_count(&callable_tree(MAX_FUNCTION_VALUE_CANDIDATES + 1))
                .is_ok_and(|count| count > MAX_FUNCTION_VALUE_CANDIDATES)
        );
    }

    #[test]
    fn bounds_higher_order_candidate_products_before_specialization() {
        let arguments = vec![callable_tree(16), callable_tree(17)];

        assert_eq!(
            higher_order_candidate_combinations(&[0], &arguments),
            Ok(16)
        );
        assert!(
            higher_order_candidate_combinations(&[0, 1], &arguments)
                .is_ok_and(|count| count > MAX_FUNCTION_VALUE_CANDIDATES)
        );
        assert!(matches!(
            higher_order_candidate_combinations(&[2], &arguments),
            Err(HirLoweringError::InvalidQCore(_))
        ));
    }

    #[test]
    fn rejects_a_forged_function_application_type() {
        let scalar = |name: &str| wire::TypeProto {
            kind: wire::type_proto::Kind::Scalar.into(),
            name: name.to_owned(),
            ..Default::default()
        };
        let callee = wire::HirExpressionProto {
            opcode: "function_ref".to_owned(),
            r#type: Some(wire::TypeProto {
                kind: wire::type_proto::Kind::Function.into(),
                arguments: vec![scalar("Int"), scalar("Int")],
                ..Default::default()
            }),
            ..Default::default()
        };
        let argument = wire::HirExpressionProto {
            r#type: Some(scalar("Bool")),
            ..Default::default()
        };
        let invocation = wire::HirExpressionProto {
            r#type: Some(scalar("Int")),
            ..Default::default()
        };
        let mut arena = TypeArena::default();
        let scope = arena.intern_scope("forged");

        assert_eq!(
            validate_callable_application(
                &callee,
                std::slice::from_ref(&argument),
                &invocation,
                &scope,
                &mut arena,
            ),
            Err(HirLoweringError::InvalidQCore(
                "invoke does not match its function signature".to_owned()
            ))
        );
    }

    #[test]
    fn validates_every_input_of_a_multi_argument_function_value() {
        let scalar = |name: &str| wire::TypeProto {
            kind: wire::type_proto::Kind::Scalar.into(),
            name: name.to_owned(),
            ..Default::default()
        };
        let callee = wire::HirExpressionProto {
            opcode: "function_ref".to_owned(),
            r#type: Some(wire::TypeProto {
                kind: wire::type_proto::Kind::Function.into(),
                arguments: vec![scalar("Int"), scalar("Bool"), scalar("Real")],
                ..Default::default()
            }),
            ..Default::default()
        };
        let arguments = [
            wire::HirExpressionProto {
                r#type: Some(scalar("Int")),
                ..Default::default()
            },
            wire::HirExpressionProto {
                r#type: Some(scalar("Bool")),
                ..Default::default()
            },
        ];
        let invocation = wire::HirExpressionProto {
            r#type: Some(scalar("Real")),
            ..Default::default()
        };
        let mut arena = TypeArena::default();
        let scope = arena.intern_scope("multi");

        assert_eq!(
            validate_callable_application(&callee, &arguments, &invocation, &scope, &mut arena),
            Ok(())
        );
        assert_eq!(
            validate_callable_application(
                &callee,
                &arguments[..1],
                &invocation,
                &scope,
                &mut arena,
            ),
            Err(HirLoweringError::InvalidQCore(
                "invoke does not match its function arity".to_owned()
            ))
        );

        let nullary_callee = wire::HirExpressionProto {
            opcode: "function_ref".to_owned(),
            r#type: Some(wire::TypeProto {
                kind: wire::type_proto::Kind::Function.into(),
                arguments: vec![scalar("Real")],
                ..Default::default()
            }),
            ..Default::default()
        };
        assert_eq!(
            validate_callable_application(&nullary_callee, &[], &invocation, &scope, &mut arena,),
            Ok(())
        );
    }

    #[test]
    fn rejects_a_function_reference_with_forged_parameter_labels() {
        let scalar = |name: &str| wire::TypeProto {
            kind: wire::type_proto::Kind::Scalar.into(),
            name: name.to_owned(),
            ..Default::default()
        };
        let declaration = wire::HirFunctionProto {
            symbol: "native.named".to_owned(),
            parameters: vec![wire::HirParameterProto {
                name: "value".to_owned(),
                r#type: Some(scalar("Int")),
                ..Default::default()
            }],
            result_type: Some(scalar("Int")),
            ..Default::default()
        };
        let reference = wire::HirExpressionProto {
            opcode: "function_ref".to_owned(),
            r#type: Some(wire::TypeProto {
                kind: wire::type_proto::Kind::Function.into(),
                arguments: vec![scalar("Int"), scalar("Int")],
                field_names: vec!["forged".to_owned()],
                ..Default::default()
            }),
            attributes: vec![wire::AttributeProto {
                name: "callee".to_owned(),
                value: Some(wire::LiteralProto {
                    value: Some(wire::literal_proto::Value::StringValue(
                        declaration.symbol.clone(),
                    )),
                }),
            }],
            ..Default::default()
        };
        let declarations = HashMap::from([(declaration.symbol.as_str(), &declaration)]);

        assert!(matches!(
            validate_function_reference(&reference, &declarations),
            Err(HirError::InvalidStructure(reason))
                if reason.contains("type disagrees")
        ));
    }

    fn expression_with_evaluation_order(
        opcode: &str,
        operand_count: usize,
        order: &[usize],
    ) -> wire::HirExpressionProto {
        wire::HirExpressionProto {
            opcode: opcode.to_owned(),
            operands: vec![wire::HirExpressionProto::default(); operand_count],
            attributes: vec![evaluation_order_hir_attribute(order).expect("small test order")],
            ..Default::default()
        }
    }

    #[test]
    fn rejects_forged_hir_operand_evaluation_orders() {
        let reversed = expression_with_evaluation_order("call", 2, &[1, 0]);
        assert_eq!(validate_evaluation_order_shape(&reversed), Ok(()));

        for forged in [
            expression_with_evaluation_order("call", 2, &[0, 0]),
            expression_with_evaluation_order("call", 2, &[0, 2]),
            expression_with_evaluation_order("call", 2, &[0]),
            expression_with_evaluation_order("literal", 1, &[0]),
            expression_with_evaluation_order("invoke", 2, &[1, 0]),
        ] {
            assert!(matches!(
                validate_evaluation_order_shape(&forged),
                Err(HirError::InvalidStructure(_))
            ));
        }
    }

    #[test]
    fn bounds_local_function_alias_expansion_before_cloning() {
        let candidate = wire::HirExpressionProto {
            opcode: "function_ref".to_owned(),
            ..Default::default()
        };
        let mut aliases = HashMap::new();
        aliases.insert(
            "operation".to_owned(),
            FunctionAliasBinding {
                function_type: wire::TypeProto::default(),
                tag_type: wire::TypeProto::default(),
                candidates: std::sync::Arc::from(vec!["candidate".to_owned()]),
            },
        );
        let catalog = FunctionAliasCatalog {
            candidates: HashMap::new(),
            references: HashMap::from([("candidate".to_owned(), candidate)]),
            tags: HashMap::from([("candidate".to_owned(), 0)]),
        };
        let mut reference = wire::HirExpressionProto {
            opcode: "name".to_owned(),
            attributes: vec![wire::AttributeProto {
                name: "name".to_owned(),
                value: Some(wire::LiteralProto {
                    value: Some(wire::literal_proto::Value::StringValue(
                        "operation".to_owned(),
                    )),
                }),
            }],
            ..Default::default()
        };

        assert!(matches!(
            rewrite_function_alias_names(&mut reference, &aliases, &catalog, &mut 0, 0, 0),
            Err(HirLoweringError::Unsupported(reason))
                if reason == "function alias expansion exceeds 0 nodes"
        ));
    }
}
