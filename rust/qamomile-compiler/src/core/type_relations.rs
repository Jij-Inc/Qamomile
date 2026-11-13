//! Structural type and literal relations shared across compiler boundaries.
//!
//! Logical dependent-shape unification and recursive availability bounds are
//! deliberately separate relations. Their conjunction is assignability. This
//! keeps dimension inference reusable where staging is orthogonal while giving
//! every compiler boundary one definition of declared-type compatibility.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use prost::Message;
use thiserror::Error;

use crate::budget;
use crate::equations::literal_conforms;
use crate::literals::validate_literal_references;
use crate::types::{
    AffineDimension, Dimension, PolynomialDimension, ScalarType, SemanticType, Stage, TypeArena,
    TypeId,
};
use crate::wire;

const MAX_RELATION_TYPES: usize = 1_000_000;
const MAX_RELATION_BINDINGS: usize = 1_000_000;
const MAX_RELATION_BYTES: usize = 64 * 1024 * 1024;

/// A source-portable dependent-dimension value.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PortableDimension {
    /// One positive statically known extent.
    Fixed(u64),
    /// One nonempty symbolic extent in the observed type scope.
    Symbol(String),
}

/// One source expression that may provide a dependent-dimension value.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StaticDimensionCandidate {
    /// An arbitrary-precision source integer spelling checked when relevant.
    Integer(String),
    /// A nonempty caller-scope symbolic integer name.
    Symbol(String),
}

/// One source-facing substitution inferred from a logical type pattern.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TypeDimensionBinding {
    name: String,
    value: PortableDimension,
}

impl TypeDimensionBinding {
    /// Return the pattern binder name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Return the observed dimension value.
    #[must_use]
    pub fn value(&self) -> &PortableDimension {
        &self.value
    }
}

/// Structural projections derived from one canonical semantic type.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TypeShapeProjection {
    logical_width: Option<u64>,
    coherent_leaf_widths: Option<Box<[u64]>>,
    sequence_element: Option<Vec<u8>>,
    aggregate_elements: Option<Box<[Vec<u8>]>>,
}

impl TypeShapeProjection {
    /// Return the fixed width of one bit-like logical leaf.
    #[must_use]
    pub const fn logical_width(&self) -> Option<u64> {
        self.logical_width
    }

    /// Return ordered fixed coherent leaf widths when the type is coherent.
    #[must_use]
    pub fn coherent_leaf_widths(&self) -> Option<&[u64]> {
        self.coherent_leaf_widths.as_deref()
    }

    /// Return the canonical rank-reduced sequence element type.
    #[must_use]
    pub fn sequence_element(&self) -> Option<&[u8]> {
        self.sequence_element.as_deref()
    }

    /// Return canonical aggregate element types with outer staging distributed.
    #[must_use]
    pub fn aggregate_elements(&self) -> Option<&[Vec<u8>]> {
        self.aggregate_elements.as_deref()
    }
}

/// Constructor admissibility derived from one canonical semantic type.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TypeAdmissibilityProjection {
    coherent_payload: bool,
    stageable: bool,
    classical: bool,
}

impl TypeAdmissibilityProjection {
    /// Return whether the type may inhabit `Q[T]`.
    #[must_use]
    pub const fn coherent_payload(self) -> bool {
        self.coherent_payload
    }

    /// Return whether the type may inhabit an availability stage.
    #[must_use]
    pub const fn stageable(self) -> bool {
        self.stageable
    }

    /// Return whether the type contains no owned coherent data.
    #[must_use]
    pub const fn classical(self) -> bool {
        self.classical
    }
}

/// Failure at the standalone native type-relation boundary.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum TypeRelationError {
    /// A type or initial dimension binding is malformed or noncanonical.
    #[error("Q4510: invalid type-relation input: {0}")]
    InvalidInput(String),
}

/// Failure while mapping one source call into declaration order.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum FunctionArgumentBindingError {
    /// The encoded signature or aligned metadata is malformed or excessive.
    #[error("Q4510: invalid function-argument binding input: {0}")]
    InvalidInput(String),
    /// More positional arguments were supplied than the declaration accepts.
    #[error("Q5048: too many arguments: expected at most {expected}, got {actual}")]
    TooMany {
        /// Declared parameter count.
        expected: usize,
        /// Supplied argument count.
        actual: usize,
    },
    /// A keyword does not name a declared parameter.
    #[error("Q5049: unknown parameter {name:?}")]
    Unknown {
        /// Unknown source keyword.
        name: String,
    },
    /// One parameter was supplied positionally and/or by keyword more than once.
    #[error("Q5050: parameter {name:?} supplied twice")]
    Duplicate {
        /// Repeated source parameter name.
        name: String,
    },
    /// A required parameter has neither an argument nor a default.
    #[error("Q5051: missing argument {name:?}")]
    Missing {
        /// Missing source parameter name.
        name: String,
    },
}

impl FunctionArgumentBindingError {
    fn detail(&self) -> String {
        match self {
            Self::InvalidInput(reason) => reason.clone(),
            Self::TooMany { expected, actual } => {
                format!("function value expects {expected} arguments, got {actual}")
            }
            Self::Unknown { name } => format!("unknown function parameter {name:?}"),
            Self::Duplicate { name } => {
                format!("function parameter {name:?} was supplied twice")
            }
            Self::Missing { name } if name.is_empty() => {
                "function value is missing argument".to_owned()
            }
            Self::Missing { name } => format!("function value is missing argument {name:?}"),
        }
    }
}

/// One checked application of a labeled closed function type.
pub(crate) struct BoundFunctionApplication {
    ordered_indices: Box<[usize]>,
    result_type: TypeId,
}

impl BoundFunctionApplication {
    /// Return raw source-argument indices in declaration order.
    pub(crate) fn ordered_indices(&self) -> &[usize] {
        &self.ordered_indices
    }

    /// Return the instantiated and stage-lifted result type.
    pub(crate) const fn result_type(&self) -> TypeId {
        self.result_type
    }
}

/// Infer the result of a source-ordered function-value application.
///
/// Positional values precede keyword values in `argument_types`; keyword names
/// retain their source order. The shared binder maps that row to declaration
/// order, validates every type and dependent dimension, and preserves the
/// latest availability stage.
///
/// # Errors
///
/// Returns an error for malformed or excessive canonical type inputs and for
/// an invalid keyword/arity/type application.
pub fn infer_function_application(
    function_type: &[u8],
    argument_types: &[Vec<u8>],
    positional_count: usize,
    keyword_names: &[String],
) -> Result<Vec<u8>, TypeRelationError> {
    let keyword_bytes = keyword_names
        .iter()
        .try_fold(0_usize, |total, name| total.checked_add(name.len()));
    let DecodedFunctionApplication {
        mut arena,
        function_type,
        argument_types,
        ..
    } = decode_function_application_types(
        function_type,
        argument_types,
        keyword_names.len(),
        keyword_bytes,
    )?;
    let application = bind_function_application(
        &mut arena,
        function_type,
        &argument_types,
        positional_count,
        keyword_names,
    )
    .map_err(TypeRelationError::InvalidInput)?;
    encode_type_id(&arena, application.result_type())
}

/// Infer one declared call using source-projected static dimension evidence.
///
/// Arguments are already in declaration order after source-level keyword and
/// default binding. Each optional static value is aligned with that row. Rust
/// decides whether the corresponding integer parameter actually owns a
/// dependent dimension; irrelevant values are ignored.
///
/// # Errors
///
/// Returns an error for malformed or excessive canonical type inputs, an
/// invalid static-dimension row, or a call that violates its declaration.
pub fn infer_declared_function_application(
    function_type: &[u8],
    argument_types: &[Vec<u8>],
    static_arguments: &[Option<StaticDimensionCandidate>],
) -> Result<Vec<u8>, TypeRelationError> {
    let static_bytes = static_arguments.iter().try_fold(0_usize, |total, value| {
        total.checked_add(match value {
            Some(
                StaticDimensionCandidate::Integer(value) | StaticDimensionCandidate::Symbol(value),
            ) => value.len(),
            None => 0,
        })
    });
    let DecodedFunctionApplication {
        mut arena,
        function_type,
        argument_types,
        function_scope,
        argument_scope,
    } = decode_function_application_types(
        function_type,
        argument_types,
        static_arguments.len(),
        static_bytes,
    )?;
    if static_arguments.len() != argument_types.len() {
        return Err(TypeRelationError::InvalidInput(
            "static argument row does not match the declared call arity".to_owned(),
        ));
    }
    let base = arena.strip_stage(function_type);
    let (parameter_names, inputs) = match arena.get(base) {
        Some(SemanticType::Function {
            parameter_names,
            inputs,
            ..
        }) => (parameter_names.clone(), inputs.clone()),
        _ => {
            return Err(TypeRelationError::InvalidInput(
                "declared application target is not a function type".to_owned(),
            ));
        }
    };
    if parameter_names.len() != inputs.len() {
        return Err(TypeRelationError::InvalidInput(
            "declared application requires complete parameter names".to_owned(),
        ));
    }
    let binders = arena.symbolic_dimensions(base);
    let mut initial = Vec::new();
    for ((name, input), value) in parameter_names
        .iter()
        .zip(inputs.iter())
        .zip(static_arguments)
    {
        let key = Dimension::Symbol {
            scope: Arc::clone(&function_scope),
            name: Arc::from(name.as_str()),
        };
        let is_integer = matches!(
            arena.get(arena.strip_stage(*input)),
            Some(SemanticType::Scalar(ScalarType::Int))
        );
        if binders.contains(&key)
            && is_integer
            && let Some(value) = value
        {
            let value = match value {
                StaticDimensionCandidate::Integer(source) => {
                    let value = source.parse::<u64>().map_err(|_| {
                        TypeRelationError::InvalidInput(format!(
                            "dimension {name:?} must be a positive representable integer"
                        ))
                    })?;
                    if source != &value.to_string() {
                        return Err(TypeRelationError::InvalidInput(format!(
                            "dimension {name:?} has a noncanonical integer"
                        )));
                    }
                    PortableDimension::Fixed(value)
                }
                StaticDimensionCandidate::Symbol(value) => PortableDimension::Symbol(value.clone()),
            };
            initial.push((name.clone(), value));
        }
    }
    let dimension_bindings = decode_portable_bindings(&initial, &function_scope, &argument_scope)?;
    let application = bind_function_application_with_bindings(
        &mut arena,
        function_type,
        &argument_types,
        argument_types.len(),
        &[],
        dimension_bindings,
    )
    .map_err(TypeRelationError::InvalidInput)?;
    encode_type_id(&arena, application.result_type())
}

/// Bind raw positional and keyword arguments into one declaration row.
///
/// Returned entries contain the raw source-argument index or `None` when the
/// declaration supplies that slot's default. The plan is transient source
/// elaboration data and is never persisted in `HIR` or `QCore`.
///
/// # Errors
///
/// Returns a stable source binding diagnostic for excessive, malformed,
/// unknown, duplicate, or missing arguments.
pub fn bind_declared_function_arguments(
    function_type: &[u8],
    argument_count: usize,
    positional_count: usize,
    keyword_names: &[String],
    has_defaults: &[bool],
) -> Result<Box<[Option<usize>]>, FunctionArgumentBindingError> {
    let keyword_bytes = keyword_names
        .iter()
        .try_fold(0_usize, |total, name| total.checked_add(name.len()));
    if argument_count
        .checked_add(keyword_names.len())
        .and_then(|count| count.checked_add(has_defaults.len()))
        .is_none_or(|count| count > MAX_RELATION_TYPES)
        || keyword_bytes
            .and_then(|bytes| function_type.len().checked_add(bytes))
            .is_none_or(|bytes| bytes > MAX_RELATION_BYTES)
    {
        return Err(FunctionArgumentBindingError::InvalidInput(
            "function argument binding exceeds the relation budget".to_owned(),
        ));
    }
    let mut arena = TypeArena::default();
    let scope = arena.intern_scope("function");
    let function_type = decode_single_type(&mut arena, function_type, &scope)
        .map_err(|error| FunctionArgumentBindingError::InvalidInput(error.to_string()))?;
    let base = arena.strip_stage(function_type);
    let (parameter_names, parameter_count) = match arena.get(base) {
        Some(SemanticType::Function {
            parameter_names,
            inputs,
            ..
        }) => (parameter_names.clone(), inputs.len()),
        _ => {
            return Err(FunctionArgumentBindingError::InvalidInput(
                "argument binding target is not a function type".to_owned(),
            ));
        }
    };
    if parameter_names.len() != parameter_count {
        return Err(FunctionArgumentBindingError::InvalidInput(
            "declared argument binding requires complete parameter names".to_owned(),
        ));
    }
    bind_function_argument_indices(
        &parameter_names,
        parameter_count,
        argument_count,
        positional_count,
        keyword_names,
        Some(has_defaults),
    )
}

struct DecodedFunctionApplication {
    arena: TypeArena,
    function_type: TypeId,
    argument_types: Vec<TypeId>,
    function_scope: Arc<str>,
    argument_scope: Arc<str>,
}

fn decode_function_application_types(
    function_type: &[u8],
    argument_types: &[Vec<u8>],
    supplemental_count: usize,
    supplemental_bytes: Option<usize>,
) -> Result<DecodedFunctionApplication, TypeRelationError> {
    let type_bytes = argument_types
        .iter()
        .try_fold(function_type.len(), |total, payload| {
            total.checked_add(payload.len())
        });
    if argument_types
        .len()
        .checked_add(supplemental_count)
        .is_none_or(|count| count > MAX_RELATION_TYPES)
        || supplemental_bytes
            .and_then(|extra| type_bytes.and_then(|types| types.checked_add(extra)))
            .is_none_or(|size| size > MAX_RELATION_BYTES)
    {
        return Err(TypeRelationError::InvalidInput(
            "function application exceeds the relation budget".to_owned(),
        ));
    }
    let mut arena = TypeArena::default();
    let function_scope = arena.intern_scope("function");
    let argument_scope = arena.intern_scope("argument");
    let function_type = decode_single_type(&mut arena, function_type, &function_scope)?;
    let argument_types = argument_types
        .iter()
        .map(|source| decode_single_type(&mut arena, source, &argument_scope))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(DecodedFunctionApplication {
        arena,
        function_type,
        argument_types,
        function_scope,
        argument_scope,
    })
}

/// Infer dependent dimensions across an ordered row of logical type pairs.
///
/// Availability annotations are deliberately ignored by this relation. Use
/// [`type_is_assignable`] when an assignment boundary must enforce them too.
///
/// # Errors
///
/// Returns an error for malformed, noncanonical, or excessive inputs. A
/// well-formed row that cannot be unified returns `Ok(None)`.
pub fn infer_type_dimensions(
    expected_types: &[Vec<u8>],
    actual_types: &[Vec<u8>],
    initial_bindings: &[(String, PortableDimension)],
) -> Result<Option<Box<[TypeDimensionBinding]>>, TypeRelationError> {
    let DecodedRelation {
        arena,
        expected,
        actual,
        expected_scope,
        actual_scope,
    } = decode_relation_types(expected_types, actual_types)?;
    let mut bindings = decode_portable_bindings(initial_bindings, &expected_scope, &actual_scope)?;
    if !expected
        .iter()
        .zip(&actual)
        .all(|(expected, actual)| unify_dimensions(&arena, *expected, *actual, &mut bindings))
    {
        return Ok(None);
    }
    let mut bindings = bindings
        .into_iter()
        .filter_map(|(expected, actual)| match expected {
            Dimension::Symbol { name, .. } => Some(TypeDimensionBinding {
                name: name.to_string(),
                value: match actual {
                    Dimension::Fixed(value) => PortableDimension::Fixed(value),
                    Dimension::Symbol { name, .. } => PortableDimension::Symbol(name.to_string()),
                },
            }),
            Dimension::Fixed(_) => None,
        })
        .collect::<Vec<_>>();
    bindings.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(Some(bindings.into_boxed_slice()))
}

/// Return whether an actual type may cross one declared type boundary.
///
/// # Errors
///
/// Returns an error for malformed, noncanonical, or excessive type payloads.
pub fn type_is_assignable(expected: &[u8], actual: &[u8]) -> Result<bool, TypeRelationError> {
    let DecodedRelation {
        arena,
        expected,
        actual,
        ..
    } = decode_relation_types(&[expected.to_vec()], &[actual.to_vec()])?;
    Ok(is_assignable(&arena, expected[0], actual[0]))
}

/// Return whether two local operands differ only in outer availability.
///
/// Unlike call-site unification, this relation keeps dependent dimensions and
/// availability nested inside aggregates rigid.
///
/// # Errors
///
/// Returns an error for malformed, noncanonical, or excessive type payloads.
pub fn types_are_local_peers(left: &[u8], right: &[u8]) -> Result<bool, TypeRelationError> {
    let total_bytes = left.len().checked_add(right.len());
    if total_bytes.is_none_or(|size| size > MAX_RELATION_BYTES) {
        return Err(TypeRelationError::InvalidInput(
            "local peer types exceed the relation byte budget".to_owned(),
        ));
    }
    let mut arena = TypeArena::default();
    let scope = arena.intern_scope("source");
    let left = decode_single_type(&mut arena, left, &scope)?;
    let right = decode_single_type(&mut arena, right, &scope)?;
    Ok(arena.local_peers(left, right))
}

/// Return whether one canonical portable literal inhabits a semantic type.
///
/// This is the standalone form of the literal relation used by declarations,
/// specialization, invocation binding, selection, and persistent verification.
///
/// # Errors
///
/// Returns an error for malformed, noncanonical, or excessive type or literal
/// payloads. A canonical but ill-typed literal returns `Ok(false)`.
pub fn literal_conforms_type(
    type_source: &[u8],
    literal_source: &[u8],
) -> Result<bool, TypeRelationError> {
    let total_bytes = type_source.len().checked_add(literal_source.len());
    if total_bytes.is_none_or(|size| size > MAX_RELATION_BYTES) {
        return Err(TypeRelationError::InvalidInput(
            "type and literal exceed the relation byte budget".to_owned(),
        ));
    }
    let mut arena = TypeArena::default();
    let scope = arena.intern_scope("source");
    let type_id = decode_single_type(&mut arena, type_source, &scope)?;
    let literal = decode_single_literal(literal_source)?;
    Ok(literal_conforms(&arena, type_id, &literal))
}

/// Infer the unique unstaged semantic type of one portable source literal.
///
/// Scalars and positional tuples have a unique introduction type. Strings,
/// bytes, and mappings require an explicit declaration and return `Ok(None)`.
///
/// # Errors
///
/// Returns an error for malformed, noncanonical, excessive, or
/// unrepresentable literal payloads.
pub fn infer_literal_type(source: &[u8]) -> Result<Option<Vec<u8>>, TypeRelationError> {
    if source.len() > MAX_RELATION_BYTES {
        return Err(TypeRelationError::InvalidInput(
            "literal exceeds the relation byte budget".to_owned(),
        ));
    }
    let literal = decode_single_literal(source)?;
    let mut arena = TypeArena::default();
    let Some(type_id) = infer_literal_type_id(&mut arena, &literal)? else {
        return Ok(None);
    };
    encode_type_id(&arena, type_id).map(Some)
}

/// Substitute dependent dimensions in one canonical semantic type.
///
/// Symbols absent from `bindings` remain symbolic. Replacement symbols remain
/// in the source type's scope, so this is capture-free without inventing a
/// second name-resolution mechanism.
///
/// # Errors
///
/// Returns an error for malformed, noncanonical, excessive, or invalid input.
pub fn substitute_type_dimensions(
    source: &[u8],
    bindings: &[(String, PortableDimension)],
) -> Result<Vec<u8>, TypeRelationError> {
    let mut arena = TypeArena::default();
    let scope = arena.intern_scope("source");
    let root = decode_single_type(&mut arena, source, &scope)?;
    let bindings = decode_portable_bindings(bindings, &scope, &scope)?;
    let substituted = arena
        .substitute_dimensions(root, &bindings)
        .map_err(TypeRelationError::InvalidInput)?;
    arena
        .encode_wire_type(substituted)
        .map(|type_spec| type_spec.encode_to_vec())
        .map_err(TypeRelationError::InvalidInput)
}

/// Return the sorted dependent-dimension names in one canonical semantic type.
///
/// # Errors
///
/// Returns an error for malformed, noncanonical, or excessive input.
pub fn type_symbolic_dimensions(source: &[u8]) -> Result<Box<[String]>, TypeRelationError> {
    let mut arena = TypeArena::default();
    let scope = arena.intern_scope("source");
    let root = decode_single_type(&mut arena, source, &scope)?;
    let mut names = BTreeSet::new();
    names.extend(arena.symbolic_dimensions(root).into_iter().filter_map(
        |dimension| match dimension {
            Dimension::Symbol { name, .. } => Some(name.to_string()),
            Dimension::Fixed(_) => None,
        },
    ));
    Ok(names.into_iter().collect::<Vec<_>>().into_boxed_slice())
}

/// Derive source-facing structural facts from one canonical semantic type.
///
/// # Errors
///
/// Returns an error for malformed, noncanonical, excessive, or unrepresentable
/// input and derived types.
pub fn inspect_type_shape(source: &[u8]) -> Result<TypeShapeProjection, TypeRelationError> {
    let mut arena = TypeArena::default();
    let scope = arena.intern_scope("source");
    let root = decode_single_type(&mut arena, source, &scope)?;
    let logical_width = arena
        .logical_width(root)
        .map_err(TypeRelationError::InvalidInput)?;
    let coherent_leaf_widths = arena
        .coherent_leaf_widths(root)
        .map_err(TypeRelationError::InvalidInput)?;
    let sequence_element = arena
        .sequence_element(root)
        .map_err(TypeRelationError::InvalidInput)?
        .map(|element| encode_type_id(&arena, element))
        .transpose()?;
    let aggregate_elements = arena
        .aggregate_len(root)
        .map(|length| {
            (0..length)
                .map(|index| {
                    let element = arena
                        .aggregate_element_at(root, index)
                        .map_err(TypeRelationError::InvalidInput)?
                        .ok_or_else(|| {
                            TypeRelationError::InvalidInput(
                                "aggregate projection is internally incomplete".to_owned(),
                            )
                        })?;
                    encode_type_id(&arena, element)
                })
                .collect::<Result<Vec<_>, _>>()
                .map(Vec::into_boxed_slice)
        })
        .transpose()?;
    Ok(TypeShapeProjection {
        logical_width,
        coherent_leaf_widths,
        sequence_element,
        aggregate_elements,
    })
}

/// Derive type-constructor admissibility from one canonical semantic type.
///
/// # Errors
///
/// Returns an error for malformed, noncanonical, or excessive input.
pub fn inspect_type_admissibility(
    source: &[u8],
) -> Result<TypeAdmissibilityProjection, TypeRelationError> {
    let mut arena = TypeArena::default();
    let scope = arena.intern_scope("source");
    let root = decode_single_type(&mut arena, source, &scope)?;
    let classical = !arena.is_quantum(root);
    Ok(TypeAdmissibilityProjection {
        coherent_payload: arena.is_coherent_payload(root),
        stageable: arena.is_stageable(root),
        classical,
    })
}

/// Return the latest outer availability stage in one canonical type row.
///
/// Stages nested inside aggregates remain part of their field types and are
/// intentionally not lifted to the aggregate boundary.
///
/// # Errors
///
/// Returns an error for malformed, noncanonical, or excessive input.
pub fn common_type_stage(sources: &[Vec<u8>]) -> Result<Option<Stage>, TypeRelationError> {
    if sources.len() > MAX_RELATION_TYPES {
        return Err(TypeRelationError::InvalidInput(
            "type row exceeds the relation budget".to_owned(),
        ));
    }
    let type_bytes = sources
        .iter()
        .try_fold(0_usize, |total, payload| total.checked_add(payload.len()));
    if type_bytes.is_none_or(|size| size > MAX_RELATION_BYTES) {
        return Err(TypeRelationError::InvalidInput(
            "type row exceeds the relation byte budget".to_owned(),
        ));
    }
    let mut arena = TypeArena::default();
    let scope = arena.intern_scope("source");
    let mut common = None;
    for source in sources {
        let type_id = decode_single_type(&mut arena, source, &scope)?;
        common = Stage::join(common, arena.stage_of(type_id));
    }
    Ok(common)
}

struct DecodedRelation {
    arena: TypeArena,
    expected: Vec<TypeId>,
    actual: Vec<TypeId>,
    expected_scope: Arc<str>,
    actual_scope: Arc<str>,
}

fn decode_relation_types(
    expected_types: &[Vec<u8>],
    actual_types: &[Vec<u8>],
) -> Result<DecodedRelation, TypeRelationError> {
    if expected_types.len() != actual_types.len() {
        return Err(TypeRelationError::InvalidInput(
            "expected and actual type rows have different lengths".to_owned(),
        ));
    }
    if expected_types.len() > MAX_RELATION_TYPES {
        return Err(TypeRelationError::InvalidInput(
            "type row exceeds the relation budget".to_owned(),
        ));
    }
    let type_bytes = expected_types
        .iter()
        .chain(actual_types)
        .try_fold(0_usize, |total, payload| total.checked_add(payload.len()));
    if type_bytes.is_none_or(|size| size > MAX_RELATION_BYTES) {
        return Err(TypeRelationError::InvalidInput(
            "type row exceeds the relation byte budget".to_owned(),
        ));
    }
    let mut arena = TypeArena::default();
    let expected_scope = arena.intern_scope("expected");
    let actual_scope = arena.intern_scope("actual");
    let expected = expected_types
        .iter()
        .map(|payload| {
            arena
                .decode_canonical_wire_type(payload, &expected_scope)
                .map_err(TypeRelationError::InvalidInput)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let actual = actual_types
        .iter()
        .map(|payload| {
            arena
                .decode_canonical_wire_type(payload, &actual_scope)
                .map_err(TypeRelationError::InvalidInput)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(DecodedRelation {
        arena,
        expected,
        actual,
        expected_scope,
        actual_scope,
    })
}

fn decode_single_type(
    arena: &mut TypeArena,
    source: &[u8],
    scope: &Arc<str>,
) -> Result<TypeId, TypeRelationError> {
    if source.len() > MAX_RELATION_BYTES {
        return Err(TypeRelationError::InvalidInput(
            "type exceeds the relation byte budget".to_owned(),
        ));
    }
    arena
        .decode_canonical_wire_type(source, scope)
        .map_err(TypeRelationError::InvalidInput)
}

fn encode_type_id(arena: &TypeArena, type_id: TypeId) -> Result<Vec<u8>, TypeRelationError> {
    arena
        .encode_wire_type(type_id)
        .map(|type_spec| type_spec.encode_to_vec())
        .map_err(TypeRelationError::InvalidInput)
}

fn decode_single_literal(source: &[u8]) -> Result<wire::LiteralProto, TypeRelationError> {
    budget::preflight_literal(source)
        .map_err(|error| TypeRelationError::InvalidInput(preflight_reason(error)))?;
    let literal = wire::LiteralProto::decode(source)
        .map_err(|error| TypeRelationError::InvalidInput(error.to_string()))?;
    if literal.encode_to_vec() != source {
        return Err(TypeRelationError::InvalidInput(
            "literal payload is not canonical".to_owned(),
        ));
    }
    validate_literal_references(vec![&literal]).map_err(TypeRelationError::InvalidInput)?;
    Ok(literal)
}

fn infer_literal_type_id(
    arena: &mut TypeArena,
    literal: &wire::LiteralProto,
) -> Result<Option<TypeId>, TypeRelationError> {
    use wire::literal_proto::Value;

    let inferred = match literal.value.as_ref() {
        Some(Value::UnitValue(true)) => SemanticType::Scalar(ScalarType::Unit),
        Some(Value::BoolValue(_)) => SemanticType::Scalar(ScalarType::Bool),
        Some(Value::IntValue(_) | Value::BigintValue(_)) => SemanticType::Scalar(ScalarType::Int),
        Some(Value::RealValue(_)) => SemanticType::Scalar(ScalarType::Real),
        Some(Value::ComplexValue(_)) => SemanticType::Scalar(ScalarType::Complex),
        Some(Value::TupleValue(sequence)) => {
            let mut elements = Vec::with_capacity(sequence.values.len());
            for value in &sequence.values {
                let Some(element) = infer_literal_type_id(arena, value)? else {
                    return Ok(None);
                };
                elements.push(element);
            }
            SemanticType::Product(elements.into_boxed_slice())
        }
        Some(Value::StringValue(_) | Value::BytesValue(_) | Value::MappingValue(_)) => {
            return Ok(None);
        }
        Some(Value::UnitValue(false)) | None => {
            return Err(TypeRelationError::InvalidInput(
                "literal has no canonical value".to_owned(),
            ));
        }
    };
    arena
        .intern_derived(inferred)
        .map(Some)
        .map_err(TypeRelationError::InvalidInput)
}

fn preflight_reason(error: budget::PreflightError) -> String {
    match error {
        budget::PreflightError::Budget(reason) | budget::PreflightError::Malformed(reason) => {
            reason
        }
    }
}

fn decode_portable_bindings(
    source: &[(String, PortableDimension)],
    key_scope: &Arc<str>,
    value_scope: &Arc<str>,
) -> Result<HashMap<Dimension, Dimension>, TypeRelationError> {
    if source.len() > MAX_RELATION_BINDINGS {
        return Err(TypeRelationError::InvalidInput(
            "binding count exceeds the relation budget".to_owned(),
        ));
    }
    let binding_bytes = source.iter().try_fold(0_usize, |total, (name, value)| {
        let value_len = match value {
            PortableDimension::Fixed(_) => 0,
            PortableDimension::Symbol(value) => value.len(),
        };
        total.checked_add(name.len())?.checked_add(value_len)
    });
    if binding_bytes.is_none_or(|size| size > MAX_RELATION_BYTES) {
        return Err(TypeRelationError::InvalidInput(
            "bindings exceed the relation byte budget".to_owned(),
        ));
    }
    let mut names = HashSet::with_capacity(source.len());
    let mut bindings = HashMap::with_capacity(source.len());
    for (name, value) in source {
        if name.is_empty() || !names.insert(name.as_str()) {
            return Err(TypeRelationError::InvalidInput(
                "binding names must be unique and nonempty".to_owned(),
            ));
        }
        let value = match value {
            PortableDimension::Fixed(value) if *value > 0 => Dimension::Fixed(*value),
            PortableDimension::Fixed(_) => {
                return Err(TypeRelationError::InvalidInput(
                    "fixed dimensions must be positive".to_owned(),
                ));
            }
            PortableDimension::Symbol(value) if !value.is_empty() => Dimension::Symbol {
                scope: Arc::clone(value_scope),
                name: Arc::from(value.as_str()),
            },
            PortableDimension::Symbol(_) => {
                return Err(TypeRelationError::InvalidInput(
                    "symbolic dimensions must be nonempty".to_owned(),
                ));
            }
        };
        bindings.insert(
            Dimension::Symbol {
                scope: Arc::clone(key_scope),
                name: Arc::from(name.as_str()),
            },
            value,
        );
    }
    Ok(bindings)
}

fn bind_function_argument_indices(
    parameter_names: &[String],
    parameter_count: usize,
    argument_count: usize,
    positional_count: usize,
    keyword_names: &[String],
    has_defaults: Option<&[bool]>,
) -> Result<Box<[Option<usize>]>, FunctionArgumentBindingError> {
    if positional_count > argument_count || keyword_names.len() != argument_count - positional_count
    {
        return Err(FunctionArgumentBindingError::InvalidInput(
            "argument-binding metadata disagrees with its argument row".to_owned(),
        ));
    }
    if has_defaults.is_some_and(|defaults| defaults.len() != parameter_count) {
        return Err(FunctionArgumentBindingError::InvalidInput(
            "default bitmap does not match the declaration arity".to_owned(),
        ));
    }
    if positional_count > parameter_count {
        return Err(FunctionArgumentBindingError::TooMany {
            expected: parameter_count,
            actual: argument_count,
        });
    }
    if !keyword_names.is_empty() && parameter_names.len() != parameter_count {
        return Err(FunctionArgumentBindingError::InvalidInput(
            "function value does not expose keyword parameter names".to_owned(),
        ));
    }
    let parameter_indices = parameter_names
        .iter()
        .enumerate()
        .map(|(index, name)| (name.as_str(), index))
        .collect::<HashMap<_, _>>();
    let mut bound = vec![None; parameter_count];
    for (index, slot) in bound.iter_mut().enumerate().take(positional_count) {
        *slot = Some(index);
    }
    for (offset, keyword) in keyword_names.iter().enumerate() {
        let raw_index = positional_count + offset;
        let parameter_index = parameter_indices
            .get(keyword.as_str())
            .copied()
            .ok_or_else(|| FunctionArgumentBindingError::Unknown {
                name: keyword.clone(),
            })?;
        if bound[parameter_index].replace(raw_index).is_some() {
            return Err(FunctionArgumentBindingError::Duplicate {
                name: keyword.clone(),
            });
        }
    }
    for (index, slot) in bound.iter().enumerate() {
        if slot.is_none() && !has_defaults.is_some_and(|defaults| defaults[index]) {
            return Err(FunctionArgumentBindingError::Missing {
                name: parameter_names.get(index).cloned().unwrap_or_default(),
            });
        }
    }
    Ok(bound.into_boxed_slice())
}

/// Bind one raw source argument row to a closed function declaration row.
pub(crate) fn bind_function_application(
    arena: &mut TypeArena,
    function_type: TypeId,
    argument_types: &[TypeId],
    positional_count: usize,
    keyword_names: &[String],
) -> Result<BoundFunctionApplication, String> {
    bind_function_application_with_bindings(
        arena,
        function_type,
        argument_types,
        positional_count,
        keyword_names,
        HashMap::new(),
    )
}

fn bind_function_application_with_bindings(
    arena: &mut TypeArena,
    function_type: TypeId,
    argument_types: &[TypeId],
    positional_count: usize,
    keyword_names: &[String],
    mut dimension_bindings: HashMap<Dimension, Dimension>,
) -> Result<BoundFunctionApplication, String> {
    if positional_count > argument_types.len()
        || keyword_names.len() != argument_types.len() - positional_count
    {
        return Err("function application metadata disagrees with its argument row".to_owned());
    }
    let function_stage = arena.stage_of(function_type);
    let base = arena.strip_stage(function_type);
    let (parameter_names, inputs, output) = match arena.get(base) {
        Some(SemanticType::Function {
            parameter_names,
            inputs,
            output,
        }) => (parameter_names.clone(), inputs.clone(), *output),
        _ => return Err("function application target is not a function type".to_owned()),
    };
    let ordered_indices = bind_function_argument_indices(
        &parameter_names,
        inputs.len(),
        argument_types.len(),
        positional_count,
        keyword_names,
        None,
    )
    .map_err(|error| error.detail())?
    .into_vec()
    .into_iter()
    .map(|index| index.expect("complete function argument binding"))
    .collect::<Vec<_>>();
    for (parameter_index, (input, raw_index)) in inputs.iter().zip(&ordered_indices).enumerate() {
        if !unify_assignable(
            arena,
            *input,
            argument_types[*raw_index],
            &mut dimension_bindings,
        ) {
            return Err(format!(
                "function argument {} does not match its declared type",
                parameter_names
                    .get(parameter_index)
                    .map_or_else(|| parameter_index.to_string(), |name| format!("{name:?}"))
            ));
        }
    }
    let result_type = arena.substitute_dimensions(output, &dimension_bindings)?;
    let result_type = if arena.is_quantum(result_type) {
        result_type
    } else {
        let stage = argument_types.iter().fold(
            Stage::join(function_stage, arena.stage_of(result_type)),
            |current, argument| Stage::join(current, arena.stage_of(*argument)),
        );
        let result_type = arena.strip_stage(result_type);
        match stage {
            Some(stage) => arena.intern_derived(SemanticType::Staged {
                stage,
                inner: result_type,
            })?,
            None => result_type,
        }
    };
    Ok(BoundFunctionApplication {
        ordered_indices: ordered_indices.into_boxed_slice(),
        result_type,
    })
}

/// Unify one logical type pattern with an observed type.
pub(crate) fn unify_dimensions(
    arena: &TypeArena,
    expected: TypeId,
    actual: TypeId,
    bindings: &mut HashMap<Dimension, Dimension>,
) -> bool {
    let expected = arena.strip_stage(expected);
    let actual = arena.strip_stage(actual);
    match (arena.get(expected), arena.get(actual)) {
        (Some(SemanticType::Scalar(left)), Some(SemanticType::Scalar(right))) => left == right,
        (
            Some(SemanticType::Array {
                element: left_element,
                shape: left_shape,
            }),
            Some(SemanticType::Array {
                element: right_element,
                shape: right_shape,
            }),
        ) => {
            left_shape.len() == right_shape.len()
                && unify_dimensions(arena, *left_element, *right_element, bindings)
                && left_shape
                    .iter()
                    .zip(right_shape)
                    .all(|(left, right)| unify_dimension(left, right, bindings))
        }
        (
            Some(SemanticType::Word {
                kind: left_kind,
                width: left_width,
            }),
            Some(SemanticType::Word {
                kind: right_kind,
                width: right_width,
            }),
        ) => left_kind == right_kind && unify_dimension(left_width, right_width, bindings),
        (
            Some(SemanticType::Fixed {
                integer_bits: left_integer,
                fractional_bits: left_fractional,
            }),
            Some(SemanticType::Fixed {
                integer_bits: right_integer,
                fractional_bits: right_fractional,
            }),
        ) => {
            unify_dimension(left_integer, right_integer, bindings)
                && unify_dimension(left_fractional, right_fractional, bindings)
        }
        (Some(SemanticType::Product(left)), Some(SemanticType::Product(right))) => {
            left.len() == right.len()
                && left
                    .iter()
                    .zip(right)
                    .all(|(left, right)| unify_dimensions(arena, *left, *right, bindings))
        }
        (
            Some(SemanticType::Record {
                name: left_name,
                fields: left_fields,
            }),
            Some(SemanticType::Record {
                name: right_name,
                fields: right_fields,
            }),
        ) => {
            left_name == right_name
                && left_fields.len() == right_fields.len()
                && left_fields.iter().zip(right_fields).all(
                    |((left_name, left), (right_name, right))| {
                        left_name == right_name && unify_dimensions(arena, *left, *right, bindings)
                    },
                )
        }
        (Some(SemanticType::Quantum(left)), Some(SemanticType::Quantum(right))) => {
            unify_dimensions(arena, *left, *right, bindings)
        }
        (
            Some(SemanticType::Function {
                inputs: left_inputs,
                output: left_output,
                ..
            }),
            Some(SemanticType::Function {
                inputs: right_inputs,
                output: right_output,
                ..
            }),
        ) => {
            left_inputs.len() == right_inputs.len()
                && left_inputs
                    .iter()
                    .zip(right_inputs)
                    .all(|(left, right)| unify_dimensions(arena, *left, *right, bindings))
                && unify_dimensions(arena, *left_output, *right_output, bindings)
        }
        _ => false,
    }
}

/// Return whether recursive availability annotations permit assignment.
pub(crate) fn stages_assignable(arena: &TypeArena, expected: TypeId, actual: TypeId) -> bool {
    stages_assignable_from(arena, expected, actual, None, None)
}

fn stages_assignable_from(
    arena: &TypeArena,
    expected: TypeId,
    actual: TypeId,
    mut actual_stage: Option<Stage>,
    mut expected_bound: Option<Stage>,
) -> bool {
    let expected = match arena.get(expected) {
        Some(SemanticType::Staged { stage, inner }) => {
            expected_bound = Some(expected_bound.map_or(*stage, |bound| bound.min(*stage)));
            *inner
        }
        Some(_) => expected,
        None => return false,
    };
    let actual = match arena.get(actual) {
        Some(SemanticType::Staged { stage, inner }) => {
            actual_stage = Stage::join(actual_stage, Some(*stage));
            *inner
        }
        Some(_) => actual,
        None => return false,
    };
    if expected_bound.is_some_and(|bound| actual_stage.is_some_and(|actual| actual > bound)) {
        return false;
    }
    match (arena.get(expected), arena.get(actual)) {
        (
            Some(SemanticType::Array { element: left, .. }),
            Some(SemanticType::Array { element: right, .. }),
        )
        | (Some(SemanticType::Quantum(left)), Some(SemanticType::Quantum(right))) => {
            stages_assignable_from(arena, *left, *right, actual_stage, expected_bound)
        }
        (Some(SemanticType::Product(left)), Some(SemanticType::Product(right))) => {
            left.iter().zip(right).all(|(left, right)| {
                stages_assignable_from(arena, *left, *right, actual_stage, expected_bound)
            })
        }
        (
            Some(SemanticType::Record {
                fields: left_fields,
                ..
            }),
            Some(SemanticType::Record {
                fields: right_fields,
                ..
            }),
        ) => left_fields
            .iter()
            .zip(right_fields)
            .all(|((_, left), (_, right))| {
                stages_assignable_from(arena, *left, *right, actual_stage, expected_bound)
            }),
        (
            Some(SemanticType::Function {
                inputs: left_inputs,
                output: left_output,
                ..
            }),
            Some(SemanticType::Function {
                inputs: right_inputs,
                output: right_output,
                ..
            }),
        ) => {
            left_inputs.len() == right_inputs.len()
                && left_inputs
                    .iter()
                    .zip(right_inputs)
                    .all(|(left, right)| stages_assignable_from(arena, *left, *right, None, None))
                && stages_assignable_from(arena, *left_output, *right_output, None, None)
        }
        _ => true,
    }
}

/// Unify dimensions and require the recursive availability relation.
pub(crate) fn unify_assignable(
    arena: &TypeArena,
    expected: TypeId,
    actual: TypeId,
    bindings: &mut HashMap<Dimension, Dimension>,
) -> bool {
    unify_dimensions(arena, expected, actual, bindings)
        && stages_assignable(arena, expected, actual)
}

/// Return whether one actual type may cross one declared type boundary.
pub(crate) fn is_assignable(arena: &TypeArena, expected: TypeId, actual: TypeId) -> bool {
    unify_assignable(arena, expected, actual, &mut HashMap::new())
}

fn unify_slice_dimension(
    expected: &Dimension,
    actual: &Dimension,
    bindings: &HashMap<Dimension, Dimension>,
) -> bool {
    let instantiated = crate::types::instantiate_dimension(expected, bindings);
    match (&instantiated, actual) {
        (Dimension::Fixed(left), Dimension::Fixed(right)) => left == right,
        (Dimension::Symbol { name: left, .. }, Dimension::Symbol { name: right, .. }) => {
            left == right
        }
        _ => false,
    }
}

fn unify_dimension(
    expected: &Dimension,
    actual: &Dimension,
    bindings: &mut HashMap<Dimension, Dimension>,
) -> bool {
    match expected {
        Dimension::Fixed(_) => expected == actual,
        Dimension::Symbol { scope, name } => {
            if name.starts_with("slice(") {
                return unify_slice_dimension(expected, actual, bindings);
            }
            let polynomial = PolynomialDimension::parse(name).ok();
            let Ok(expression) = AffineDimension::parse(name) else {
                let Some(polynomial) = polynomial else {
                    return false;
                };
                if polynomial.constant == 0
                    && polynomial.terms.len() == 1
                    && polynomial.terms.keys().next().is_some_and(|symbols| {
                        symbols.len() == 1
                            && crate::types::parse_floor_quotient(&symbols[0]).is_none()
                    })
                    && polynomial.terms.values().next() == Some(&1)
                {
                    if let Some(bound) = bindings.get(expected) {
                        return bound == actual;
                    }
                    bindings.insert(expected.clone(), actual.clone());
                    return true;
                }
                if let Dimension::Symbol { name, .. } = actual
                    && name.as_ref() == polynomial.render()
                {
                    return true;
                }
                let Dimension::Fixed(actual) = actual else {
                    return false;
                };
                return polynomial.evaluate(|symbol_name| {
                    let symbol = Dimension::Symbol {
                        scope: Arc::clone(scope),
                        name: Arc::from(symbol_name),
                    };
                    match bindings.get(&symbol) {
                        Some(Dimension::Fixed(value)) => Some(*value),
                        _ => None,
                    }
                }) == Some(*actual);
            };
            if expression.constant == 0
                && expression.terms.len() == 1
                && expression.terms.values().next() == Some(&1)
            {
                if let Some(bound) = bindings.get(expected) {
                    return bound == actual;
                }
                bindings.insert(expected.clone(), actual.clone());
                return true;
            }
            let Dimension::Fixed(actual) = actual else {
                return matches!(actual, Dimension::Symbol { name, .. } if name.as_ref() == expression.render());
            };
            let mut resolved = i128::from(expression.constant);
            let mut unbound = Vec::new();
            for (name, coefficient) in expression.terms {
                let symbol = Dimension::Symbol {
                    scope: Arc::clone(scope),
                    name: Arc::from(name),
                };
                match bindings.get(&symbol) {
                    Some(Dimension::Fixed(value)) => {
                        resolved += i128::from(coefficient) * i128::from(*value);
                    }
                    Some(Dimension::Symbol { .. }) => return false,
                    None => unbound.push((symbol, coefficient)),
                }
            }
            if unbound.is_empty() {
                return resolved == i128::from(*actual);
            }
            let [(symbol, coefficient)] = unbound.as_slice() else {
                return false;
            };
            let remainder = i128::from(*actual) - resolved;
            let coefficient = i128::from(*coefficient);
            if remainder % coefficient != 0 {
                return false;
            }
            let value = remainder / coefficient;
            let Ok(value) = u64::try_from(value) else {
                return false;
            };
            if value == 0 {
                return false;
            }
            bindings.insert(symbol.clone(), Dimension::Fixed(value));
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use prost::Message;

    use super::*;
    use crate::wire;

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

    fn bits(dimension: wire::dimension_proto::Value) -> wire::TypeProto {
        wire::TypeProto {
            kind: wire::type_proto::Kind::Array.into(),
            name: "Bits".to_owned(),
            arguments: vec![scalar("Bit")],
            dimensions: vec![wire::DimensionProto {
                value: Some(dimension),
            }],
            ..Default::default()
        }
    }

    fn product(elements: Vec<wire::TypeProto>) -> wire::TypeProto {
        wire::TypeProto {
            kind: wire::type_proto::Kind::Product.into(),
            arguments: elements,
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

    fn quantum(inner: wire::TypeProto) -> wire::TypeProto {
        wire::TypeProto {
            kind: wire::type_proto::Kind::Quantum.into(),
            arguments: vec![inner],
            ..Default::default()
        }
    }

    fn function(
        parameter_names: &[&str],
        inputs: Vec<wire::TypeProto>,
        output: wire::TypeProto,
    ) -> wire::TypeProto {
        wire::TypeProto {
            kind: wire::type_proto::Kind::Function.into(),
            arguments: inputs.into_iter().chain(std::iter::once(output)).collect(),
            field_names: parameter_names
                .iter()
                .map(|name| (*name).to_owned())
                .collect(),
            ..Default::default()
        }
    }

    fn encode(type_spec: &wire::TypeProto) -> Vec<u8> {
        type_spec.encode_to_vec()
    }

    fn literal(value: wire::literal_proto::Value) -> Vec<u8> {
        wire::LiteralProto { value: Some(value) }.encode_to_vec()
    }

    #[test]
    fn logical_unification_is_scoped_and_independent_of_availability() {
        use wire::dimension_proto::Value::{Fixed, Symbol};

        let expected = encode(&staged("Param", bits(Symbol("n".to_owned()))));
        let actual = encode(&staged("Shot", bits(Fixed(4))));
        let inferred = infer_type_dimensions(
            std::slice::from_ref(&expected),
            std::slice::from_ref(&actual),
            &[],
        )
        .expect("canonical types are accepted")
        .expect("logical types unify");

        assert_eq!(
            inferred.as_ref(),
            &[TypeDimensionBinding {
                name: "n".to_owned(),
                value: PortableDimension::Fixed(4),
            }]
        );
        assert!(!type_is_assignable(&expected, &actual).expect("types are canonical"));
    }

    #[test]
    fn assignability_enforces_recursive_aggregate_stage_bounds() {
        let expected = encode(&staged(
            "Param",
            product(vec![scalar("Int"), scalar("Int")]),
        ));
        let accepted = encode(&product(vec![
            staged("Const", scalar("Int")),
            staged("Param", scalar("Int")),
        ]));
        let rejected = encode(&product(vec![
            staged("Shot", scalar("Int")),
            staged("Param", scalar("Int")),
        ]));

        assert!(type_is_assignable(&expected, &accepted).expect("types are canonical"));
        assert!(!type_is_assignable(&expected, &rejected).expect("types are canonical"));
    }

    #[test]
    fn declared_function_arguments_bind_keywords_and_defaults() {
        use wire::dimension_proto::Value::Symbol;

        let signature = function(
            &["state", "width", "angle"],
            vec![
                quantum(bits(Symbol("width".to_owned()))),
                staged("Const", scalar("Int")),
                scalar("Real"),
            ],
            quantum(bits(Symbol("width".to_owned()))),
        );
        let binding = bind_declared_function_arguments(
            &encode(&signature),
            2,
            1,
            &["angle".to_owned()],
            &[false, true, false],
        )
        .expect("mixed arguments and one default bind in declaration order");
        assert_eq!(binding.as_ref(), &[Some(0), None, Some(1)]);
        assert_eq!(
            bind_declared_function_arguments(
                &encode(&signature),
                4,
                4,
                &[],
                &[false, false, false],
            ),
            Err(FunctionArgumentBindingError::TooMany {
                expected: 3,
                actual: 4,
            })
        );
        assert_eq!(
            bind_declared_function_arguments(
                &encode(&signature),
                1,
                0,
                &["unknown".to_owned()],
                &[false, false, false],
            ),
            Err(FunctionArgumentBindingError::Unknown {
                name: "unknown".to_owned(),
            })
        );
        assert_eq!(
            bind_declared_function_arguments(
                &encode(&signature),
                2,
                1,
                &["state".to_owned()],
                &[false, false, false],
            ),
            Err(FunctionArgumentBindingError::Duplicate {
                name: "state".to_owned(),
            })
        );
        assert_eq!(
            bind_declared_function_arguments(
                &encode(&signature),
                1,
                1,
                &[],
                &[false, false, false],
            ),
            Err(FunctionArgumentBindingError::Missing {
                name: "width".to_owned(),
            })
        );
        assert!(matches!(
            bind_declared_function_arguments(&encode(&signature), 1, 1, &[], &[false],),
            Err(FunctionArgumentBindingError::InvalidInput(_))
        ));
    }

    #[test]
    fn function_application_binds_keywords_and_keeps_signature_stages_independent() {
        use wire::dimension_proto::Value::{Fixed, Symbol};

        let signature = function(
            &["state", "width", "angle"],
            vec![
                quantum(bits(Symbol("width".to_owned()))),
                staged("Const", scalar("Int")),
                scalar("Real"),
            ],
            quantum(bits(Symbol("width".to_owned()))),
        );
        let selected = encode(&staged("Param", signature.clone()));
        assert!(type_is_assignable(&selected, &selected).expect("types are canonical"));

        let result = infer_function_application(
            &selected,
            &[
                encode(&staged("Param", scalar("Real"))),
                encode(&scalar("Int")),
                encode(&quantum(bits(Fixed(2)))),
            ],
            0,
            &["angle".to_owned(), "width".to_owned(), "state".to_owned()],
        )
        .expect("reversed keyword row binds");
        assert_eq!(result, encode(&quantum(bits(Fixed(2)))));

        let ordered_arguments = [
            encode(&quantum(bits(Fixed(2)))),
            encode(&scalar("Int")),
            encode(&staged("Param", scalar("Real"))),
        ];
        let declared_result = infer_declared_function_application(
            &encode(&signature),
            &ordered_arguments,
            &[
                None,
                Some(StaticDimensionCandidate::Integer("2".to_owned())),
                None,
            ],
        )
        .expect("declared dimension value agrees with the argument shape");
        assert_eq!(declared_result, encode(&quantum(bits(Fixed(2)))));
        assert!(
            infer_declared_function_application(
                &encode(&signature),
                &ordered_arguments,
                &[
                    None,
                    Some(StaticDimensionCandidate::Integer("3".to_owned())),
                    None,
                ],
            )
            .is_err_and(|error| error.to_string().contains("declared type"))
        );
        assert!(
            infer_declared_function_application(
                &encode(&signature),
                &ordered_arguments,
                &[
                    None,
                    Some(StaticDimensionCandidate::Integer("0".to_owned())),
                    None,
                ],
            )
            .is_err_and(|error| error.to_string().contains("must be positive"))
        );
        assert!(
            infer_declared_function_application(&encode(&signature), &ordered_arguments, &[None],)
                .is_err_and(|error| error.to_string().contains("static argument row"))
        );

        let classical = function(&["value"], vec![scalar("Real")], scalar("Real"));
        let staged_result = infer_declared_function_application(
            &encode(&classical),
            &[encode(&staged("Param", scalar("Real")))],
            &[None],
        )
        .expect("argument availability lifts a classical declared result");
        assert_eq!(staged_result, encode(&staged("Param", scalar("Real"))));

        assert!(
            infer_function_application(
                &encode(&signature),
                &vec![encode(&scalar("Real")); 3],
                0,
                &["angle".to_owned(), "unknown".to_owned(), "state".to_owned()],
            )
            .is_err_and(|error| error.to_string().contains("unknown function parameter"))
        );
    }

    #[test]
    fn local_peers_and_literal_inhabitation_reuse_native_type_semantics() {
        use wire::dimension_proto::Value::Fixed;
        use wire::literal_proto::Value::{IntValue, RealValue, TupleValue};

        let left = encode(&staged("Const", bits(Fixed(2))));
        let right = encode(&staged("Shot", bits(Fixed(2))));
        let different = encode(&staged("Shot", bits(Fixed(3))));
        assert!(types_are_local_peers(&left, &right).expect("types are canonical"));
        assert!(!types_are_local_peers(&left, &different).expect("types are canonical"));

        let fixed_type = encode(&fixed(2, 2));
        assert!(
            literal_conforms_type(&fixed_type, &literal(RealValue(1.25)))
                .expect("literal is canonical")
        );
        assert!(
            !literal_conforms_type(&fixed_type, &literal(RealValue(1.125)))
                .expect("literal is canonical")
        );

        let word_type = encode(&bits(Fixed(2)));
        let bits = literal(TupleValue(wire::literal_proto::Sequence {
            values: vec![
                wire::LiteralProto {
                    value: Some(IntValue(0)),
                },
                wire::LiteralProto {
                    value: Some(IntValue(1)),
                },
            ],
        }));
        assert!(literal_conforms_type(&word_type, &bits).expect("literal is canonical"));

        let mut noncanonical = literal(IntValue(1));
        noncanonical.extend([0x50, 0x00]);
        assert!(literal_conforms_type(&encode(&scalar("Int")), &noncanonical).is_err());
    }

    #[test]
    fn literal_inference_uses_unique_native_introduction_types() {
        use wire::literal_proto::Value::{BoolValue, IntValue, StringValue, TupleValue};

        assert_eq!(
            infer_literal_type(&literal(BoolValue(true))).expect("literal is canonical"),
            Some(encode(&scalar("Bool")))
        );
        assert_eq!(
            infer_literal_type(&literal(TupleValue(wire::literal_proto::Sequence {
                values: vec![
                    wire::LiteralProto {
                        value: Some(IntValue(1)),
                    },
                    wire::LiteralProto {
                        value: Some(BoolValue(false)),
                    },
                ],
            })))
            .expect("literal is canonical"),
            Some(encode(&product(vec![scalar("Int"), scalar("Bool")])))
        );
        assert_eq!(
            infer_literal_type(&literal(StringValue("explicit".to_owned())))
                .expect("literal is canonical"),
            None
        );
    }

    #[test]
    fn substitution_and_symbol_collection_share_scoped_dimensions() {
        use wire::dimension_proto::Value::{Fixed, Symbol};

        let source = encode(&staged(
            "Param",
            product(vec![
                bits(Symbol("n".to_owned())),
                bits(Symbol("m".to_owned())),
            ]),
        ));
        let substituted = substitute_type_dimensions(
            &source,
            &[
                ("n".to_owned(), PortableDimension::Fixed(4)),
                (
                    "m".to_owned(),
                    PortableDimension::Symbol("remaining".to_owned()),
                ),
            ],
        )
        .expect("canonical substitution succeeds");
        let expected = encode(&staged(
            "Param",
            product(vec![bits(Fixed(4)), bits(Symbol("remaining".to_owned()))]),
        ));

        assert_eq!(substituted, expected);
        assert_eq!(
            type_symbolic_dimensions(&substituted)
                .expect("substituted type remains canonical")
                .as_ref(),
            &["remaining".to_owned()]
        );
    }

    #[test]
    fn shape_projection_derives_width_sequence_aggregate_and_coherent_facts() {
        use wire::dimension_proto::Value::Fixed;

        let staged_word = encode(&staged("Param", bits(Fixed(4))));
        let projection = inspect_type_shape(&staged_word).expect("type is canonical");
        assert_eq!(projection.logical_width(), Some(4));
        assert_eq!(projection.coherent_leaf_widths(), None);
        assert_eq!(
            projection.sequence_element(),
            Some(encode(&staged("Param", scalar("Bit"))).as_slice())
        );
        assert_eq!(projection.aggregate_elements(), None);

        let staged_product = encode(&staged(
            "Shot",
            product(vec![scalar("Int"), scalar("Bool")]),
        ));
        let projection = inspect_type_shape(&staged_product).expect("type is canonical");
        assert_eq!(
            projection.aggregate_elements(),
            Some(
                [
                    encode(&staged("Shot", scalar("Int"))),
                    encode(&staged("Shot", scalar("Bool"))),
                ]
                .as_slice()
            )
        );

        let coherent = encode(&quantum(product(vec![bits(Fixed(2)), fixed(1, 2)])));
        let projection = inspect_type_shape(&coherent).expect("type is canonical");
        assert_eq!(projection.logical_width(), None);
        assert_eq!(projection.coherent_leaf_widths(), Some([2, 3].as_slice()));
    }

    #[test]
    fn admissibility_projection_closes_quantum_and_staged_constructors() {
        use wire::dimension_proto::Value::Symbol;

        let coherent = inspect_type_admissibility(&encode(&product(vec![
            bits(Symbol("n".to_owned())),
            fixed(0, 3),
        ])))
        .expect("type is canonical");
        assert!(coherent.coherent_payload());
        assert!(coherent.stageable());
        assert!(coherent.classical());

        let classical =
            inspect_type_admissibility(&encode(&scalar("Real"))).expect("type is canonical");
        assert!(!classical.coherent_payload());
        assert!(classical.stageable());
        assert!(classical.classical());

        let quantum = inspect_type_admissibility(&encode(&quantum(bits(Symbol("n".to_owned())))))
            .expect("type is canonical");
        assert!(!quantum.coherent_payload());
        assert!(!quantum.stageable());
        assert!(!quantum.classical());

        let staged = inspect_type_admissibility(&encode(&staged("Param", scalar("Int"))))
            .expect("type is canonical");
        assert!(!staged.coherent_payload());
        assert!(!staged.stageable());
        assert!(staged.classical());
    }

    #[test]
    fn common_stage_projection_joins_only_outer_availability() {
        assert_eq!(common_type_stage(&[]).expect("empty row is valid"), None);
        assert_eq!(
            common_type_stage(&[
                encode(&staged("Param", scalar("Int"))),
                encode(&scalar("Bool")),
                encode(&staged("Const", scalar("Real"))),
                encode(&staged("Shot", scalar("Bit"))),
            ])
            .expect("type row is canonical"),
            Some(Stage::Shot)
        );
        assert_eq!(
            common_type_stage(&[encode(&product(vec![staged("Shot", scalar("Int"),)]))])
                .expect("aggregate is canonical"),
            None
        );
    }
}
