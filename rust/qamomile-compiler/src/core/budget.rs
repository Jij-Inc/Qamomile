//! Structural budgets for untrusted persistent Programs.

use crate::wire;

pub(crate) const MAX_MESSAGE_NODES: usize = 1_000_000;
pub(crate) const MAX_NESTING: usize = 256;
pub(crate) const MAX_STRING_BYTES: usize = 1024 * 1024;
pub(crate) const MAX_BYTES_FIELD: usize = 64 * 1024 * 1024;
/// Maximum revisited nodes when closing one recursive semantic component.
pub(crate) const MAX_RECURSIVE_ANALYSIS_WORK: u128 = 10_000_000;

/// Failure detected without allocating decoded Protobuf message vectors.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum PreflightError {
    /// The raw message exceeds one shared structural resource budget.
    Budget(String),
    /// The raw message cannot be traversed as a Protobuf wire stream.
    Malformed(String),
}

#[derive(Clone, Copy)]
enum Schema {
    Artifact,
    ExecutionSnapshot,
    ExecutionSnapshotNode,
    ExecutionReference,
    ExecutionContextEntry,
    OptimizationProblem,
    PolynomialTerm,
    CompilationReport,
    Package,
    ParameterAbi,
    ParameterSlot,
    ResultAbi,
    ResultPostprocessing,
    ResultScalar,
    ResultParameter,
    ResultPath,
    ResultApply,
    ResultValue,
    ResultSequence,
    ResultRecord,
    ResultField,
    FamilySelection,
    FamilyDispatchParameter,
    FamilyPredicate,
    FamilyVariant,
    FamilyDecision,
    HirPackage,
    HirFunction,
    HirParameter,
    HirConstant,
    HirBlock,
    HirStatement,
    HirExpression,
    ResourceAnalysisRequest,
    ResourceStudy,
    ResourceAnalysisBundle,
    ResourceAnalysisFact,
    ResourceOperationFact,
    ResourceContribution,
    ResourceMetricValue,
    ResourceValue,
    ResourceEvidence,
    ResourceRule,
    ResourceMetric,
    ResourceExpressionGraph,
    ResourceNode,
    ResourceExpressionNode,
    ResourcePredicateNode,
    ResourceNumber,
    ResourceRational,
    ResourceSymbol,
    ResourceAttribute,
    ResourceSchedulingFootprint,
    Function,
    Constant,
    Binding,
    Parameter,
    Type,
    Dimension,
    Literal,
    LiteralSequence,
    LiteralMapping,
    LiteralEntry,
    LiteralComplex,
    Value,
    RegionRef,
    Block,
    Operation,
    Region,
    Terminator,
    Attribute,
    Span,
}

#[derive(Clone, Copy)]
enum Field {
    Varint { repeated: bool, packed: bool },
    Fixed64,
    String { repeated: bool },
    Bytes,
    Message { schema: Schema, repeated: bool },
}

#[derive(Clone, Copy)]
struct ScanLimits {
    nodes: usize,
    nesting: usize,
    string_bytes: usize,
    bytes_field: usize,
}

const DEFAULT_LIMITS: ScanLimits = ScanLimits {
    nodes: MAX_MESSAGE_NODES,
    nesting: MAX_NESTING,
    string_bytes: MAX_STRING_BYTES,
    bytes_field: MAX_BYTES_FIELD,
};

/// Preflight the reachable `QCorePackageProto` schema before Prost decoding.
///
/// The scanner borrows length-delimited fields directly from `payload` and
/// visits repeated child messages one at a time. A repeated or packed field is
/// therefore charged before Prost can allocate its decoded vector. The result
/// is the exact number of wire values charged by the scan.
pub(crate) fn preflight_with_work(payload: &[u8]) -> Result<usize, PreflightError> {
    let mut nodes = 0_usize;
    scan_message(Schema::Package, payload, 0, &mut nodes, DEFAULT_LIMITS)?;
    Ok(nodes)
}

/// Preflight one portable artifact before Prost decoding.
pub(crate) fn preflight_artifact(payload: &[u8]) -> Result<(), PreflightError> {
    let mut nodes = 0_usize;
    scan_message(Schema::Artifact, payload, 0, &mut nodes, DEFAULT_LIMITS)
}

/// Preflight one bounded execution snapshot before Prost decoding.
pub(crate) fn preflight_execution_snapshot(
    payload: &[u8],
    max_nodes: usize,
    max_nesting: usize,
    max_string_bytes: usize,
) -> Result<(), PreflightError> {
    let mut nodes = 0_usize;
    scan_message(
        Schema::ExecutionSnapshot,
        payload,
        0,
        &mut nodes,
        ScanLimits {
            nodes: max_nodes,
            nesting: max_nesting,
            string_bytes: max_string_bytes,
            bytes_field: MAX_BYTES_FIELD,
        },
    )
}

/// Preflight one canonical optimization-problem artifact before Prost decoding.
pub(crate) fn preflight_optimization_problem(payload: &[u8]) -> Result<(), PreflightError> {
    let mut nodes = 0_usize;
    scan_message(
        Schema::OptimizationProblem,
        payload,
        0,
        &mut nodes,
        DEFAULT_LIMITS,
    )
}

/// Preflight the private typed-HIR request schema before Prost decoding.
pub(crate) fn preflight_hir_with_work(payload: &[u8]) -> Result<usize, PreflightError> {
    let mut nodes = 0_usize;
    scan_message(Schema::HirPackage, payload, 0, &mut nodes, DEFAULT_LIMITS)?;
    Ok(nodes)
}

/// Preflight one symbolic resource-analysis request before Prost allocation.
pub(crate) fn preflight_resource_analysis_request(payload: &[u8]) -> Result<(), PreflightError> {
    let mut nodes = 0_usize;
    scan_message(
        Schema::ResourceAnalysisRequest,
        payload,
        0,
        &mut nodes,
        DEFAULT_LIMITS,
    )
}

/// Preflight one persisted resource study before Protobuf allocation.
pub(crate) fn preflight_resource_study(payload: &[u8]) -> Result<(), PreflightError> {
    let mut nodes = 0_usize;
    scan_message(
        Schema::ResourceStudy,
        payload,
        0,
        &mut nodes,
        DEFAULT_LIMITS,
    )
}

/// Preflight a standalone artifact parameter ABI before Prost allocation.
pub(crate) fn preflight_parameter_abi(payload: &[u8]) -> Result<(), PreflightError> {
    let mut nodes = 0_usize;
    scan_message(Schema::ParameterAbi, payload, 0, &mut nodes, DEFAULT_LIMITS)
}

/// Preflight a standalone artifact result ABI before Prost allocation.
pub(crate) fn preflight_result_abi(payload: &[u8]) -> Result<(), PreflightError> {
    let mut nodes = 0_usize;
    scan_message(Schema::ResultAbi, payload, 0, &mut nodes, DEFAULT_LIMITS)
}

/// Preflight a standalone circuit-result plan before Prost allocation.
pub(crate) fn preflight_result_postprocessing(payload: &[u8]) -> Result<(), PreflightError> {
    let mut nodes = 0_usize;
    scan_message(
        Schema::ResultPostprocessing,
        payload,
        0,
        &mut nodes,
        DEFAULT_LIMITS,
    )
}

/// Preflight a standalone circuit-family selector before Prost allocation.
pub(crate) fn preflight_circuit_family_selection(payload: &[u8]) -> Result<(), PreflightError> {
    let mut nodes = 0_usize;
    scan_message(
        Schema::FamilySelection,
        payload,
        0,
        &mut nodes,
        DEFAULT_LIMITS,
    )
}

/// Preflight one standalone portable literal before Prost allocation.
pub(crate) fn preflight_literal(payload: &[u8]) -> Result<(), PreflightError> {
    let mut nodes = 0_usize;
    scan_message(Schema::Literal, payload, 0, &mut nodes, DEFAULT_LIMITS)
}

/// Preflight one standalone semantic type before Prost allocation.
pub(crate) fn preflight_type(payload: &[u8]) -> Result<(), PreflightError> {
    let mut nodes = 0_usize;
    scan_message(Schema::Type, payload, 0, &mut nodes, DEFAULT_LIMITS)
}

fn scan_message(
    schema: Schema,
    payload: &[u8],
    depth: usize,
    nodes: &mut usize,
    limits: ScanLimits,
) -> Result<(), PreflightError> {
    charge_with_limit(nodes, 1, limits.nodes).map_err(PreflightError::Budget)?;
    if depth > limits.nesting {
        return Err(PreflightError::Budget(
            "QCore package exceeds the nesting limit".to_owned(),
        ));
    }

    let mut cursor = 0_usize;
    while cursor < payload.len() {
        let key = read_varint(payload, &mut cursor)?;
        let field_number = key >> 3;
        let wire_type = u8::try_from(key & 0x07).expect("three wire-type bits fit in u8");
        if field_number == 0 || field_number > 0x1fff_ffff {
            return Err(malformed("invalid Protobuf field number"));
        }
        let field_number =
            u32::try_from(field_number).map_err(|_| malformed("invalid Protobuf field number"))?;
        if let Some(field) = schema_field(schema, field_number) {
            scan_known_field(field, wire_type, payload, &mut cursor, depth, nodes, limits)?;
        } else {
            skip_unknown_field(wire_type, payload, &mut cursor)?;
        }
    }
    Ok(())
}

fn scan_known_field(
    field: Field,
    wire_type: u8,
    payload: &[u8],
    cursor: &mut usize,
    depth: usize,
    nodes: &mut usize,
    limits: ScanLimits,
) -> Result<(), PreflightError> {
    match field {
        Field::Varint {
            repeated,
            packed: _,
        } if wire_type == 0 => {
            read_varint(payload, cursor)?;
            if repeated {
                charge_with_limit(nodes, 1, limits.nodes).map_err(PreflightError::Budget)?;
            }
        }
        Field::Varint {
            repeated: true,
            packed: true,
        } if wire_type == 2 => {
            let packed = take_length_delimited(payload, cursor)?;
            let mut packed_cursor = 0_usize;
            while packed_cursor < packed.len() {
                read_varint(packed, &mut packed_cursor)?;
                charge_with_limit(nodes, 1, limits.nodes).map_err(PreflightError::Budget)?;
            }
        }
        Field::Fixed64 if wire_type == 1 => advance(payload, cursor, 8)?,
        Field::String { repeated } if wire_type == 2 => {
            let value = take_length_delimited(payload, cursor)?;
            if value.len() > limits.string_bytes {
                return Err(PreflightError::Budget(
                    "QCore string exceeds the size limit".to_owned(),
                ));
            }
            if repeated {
                charge_with_limit(nodes, 1, limits.nodes).map_err(PreflightError::Budget)?;
            }
        }
        Field::Bytes if wire_type == 2 => {
            let value = take_length_delimited(payload, cursor)?;
            if value.len() > limits.bytes_field {
                return Err(PreflightError::Budget(
                    "QCore byte field exceeds the size limit".to_owned(),
                ));
            }
        }
        Field::Message { schema, repeated } if wire_type == 2 => {
            let child = take_length_delimited(payload, cursor)?;
            if repeated {
                charge_with_limit(nodes, 1, limits.nodes).map_err(PreflightError::Budget)?;
            }
            scan_message(schema, child, depth.saturating_add(1), nodes, limits)?;
        }
        _ => return Err(malformed("invalid wire type for known QCore field")),
    }
    Ok(())
}

// Keep one explicit row per schema field: adjacent identical arms are useful
// review evidence that every reachable generated-wire field was classified.
#[allow(clippy::match_same_arms, clippy::too_many_lines)]
fn schema_field(schema: Schema, number: u32) -> Option<Field> {
    let scalar = || Field::Varint {
        repeated: false,
        packed: false,
    };
    let repeated_scalar = || Field::Varint {
        repeated: true,
        packed: true,
    };
    let string = || Field::String { repeated: false };
    let repeated_string = || Field::String { repeated: true };
    let message = |schema| Field::Message {
        schema,
        repeated: false,
    };
    let repeated_message = |schema| Field::Message {
        schema,
        repeated: true,
    };

    match (schema, number) {
        (Schema::ExecutionSnapshot, 1 | 2) => Some(scalar()),
        (Schema::ExecutionSnapshot, 3) => Some(string()),
        (Schema::ExecutionSnapshot, 4) => Some(message(Schema::ExecutionSnapshotNode)),
        (Schema::ExecutionSnapshot, 5) => Some(Field::Bytes),
        (Schema::ExecutionSnapshotNode, 1) => Some(scalar()),
        (Schema::ExecutionSnapshotNode, 2) => Some(message(Schema::ExecutionReference)),
        (Schema::ExecutionSnapshotNode, 3) => Some(message(Schema::Literal)),
        (Schema::ExecutionSnapshotNode, 4) => Some(repeated_message(Schema::ExecutionSnapshotNode)),
        (Schema::ExecutionReference, 1 | 3 | 5) => Some(string()),
        (Schema::ExecutionReference, 2) => Some(repeated_string()),
        (Schema::ExecutionReference, 4 | 6) => Some(scalar()),
        (Schema::ExecutionReference, 7) => Some(repeated_message(Schema::ExecutionContextEntry)),
        (Schema::ExecutionContextEntry, 1 | 2) => Some(string()),

        (Schema::OptimizationProblem, 1 | 2 | 4) => Some(scalar()),
        (Schema::OptimizationProblem, 3) => Some(string()),
        (Schema::OptimizationProblem, 5) => Some(repeated_message(Schema::PolynomialTerm)),
        (Schema::OptimizationProblem, 6) => Some(Field::Fixed64),
        (Schema::OptimizationProblem, 7) => Some(repeated_string()),
        (Schema::OptimizationProblem, 8) => Some(Field::Bytes),
        (Schema::PolynomialTerm, 1) => Some(repeated_string()),
        (Schema::PolynomialTerm, 2) => Some(Field::Fixed64),

        (Schema::Artifact, 1 | 2) => Some(scalar()),
        (Schema::Artifact, 3 | 6) => Some(string()),
        (Schema::Artifact, 4 | 12) => Some(Field::Bytes),
        (Schema::Artifact, 7) => Some(message(Schema::ParameterAbi)),
        (Schema::Artifact, 8) => Some(message(Schema::ResultAbi)),
        (Schema::Artifact, 9) => Some(message(Schema::CompilationReport)),
        (Schema::Artifact, 10) => Some(repeated_message(Schema::Attribute)),
        (Schema::Artifact, 14) => Some(repeated_string()),
        (Schema::CompilationReport, 2 | 7 | 8) => Some(repeated_string()),
        (Schema::CompilationReport, 9) => Some(Field::Bytes),

        (Schema::ParameterAbi, 1) => Some(repeated_message(Schema::ParameterSlot)),
        (Schema::ParameterSlot, 1) => Some(string()),
        (Schema::ParameterSlot, 2) => Some(message(Schema::Type)),
        (Schema::ParameterSlot, 3) => Some(message(Schema::Literal)),
        (Schema::ParameterSlot, 4) => Some(scalar()),
        (Schema::ResultAbi, 1) => Some(message(Schema::Type)),
        (Schema::ResultAbi, 2) => Some(repeated_scalar()),
        (Schema::ResultAbi, 3) => Some(scalar()),
        (Schema::ResultPostprocessing, 1..=3) => Some(scalar()),
        (Schema::ResultPostprocessing, 4) => Some(repeated_message(Schema::ResultScalar)),
        (Schema::ResultPostprocessing, 5) => Some(message(Schema::ResultValue)),
        (Schema::ResultScalar, 1 | 2 | 4) => Some(scalar()),
        (Schema::ResultScalar, 3) => Some(Field::Fixed64),
        (Schema::ResultScalar, 5) => Some(message(Schema::ResultParameter)),
        (Schema::ResultScalar, 6) => Some(message(Schema::ResultApply)),
        (Schema::ResultParameter, 1) => Some(string()),
        (Schema::ResultParameter, 2) => Some(repeated_message(Schema::ResultPath)),
        (Schema::ResultParameter, 3) => Some(scalar()),
        (Schema::ResultPath, 1) => Some(scalar()),
        (Schema::ResultPath, 2) => Some(string()),
        (Schema::ResultApply, 1) => Some(string()),
        (Schema::ResultApply, 2) => Some(scalar()),
        (Schema::ResultApply, 3) => Some(repeated_message(Schema::Attribute)),
        (Schema::ResultApply, 4) => Some(repeated_scalar()),
        (Schema::ResultValue, 1 | 3) => Some(scalar()),
        (Schema::ResultValue, 2) => Some(message(Schema::Literal)),
        (Schema::ResultValue, 4) => Some(message(Schema::ResultSequence)),
        (Schema::ResultValue, 5) => Some(message(Schema::ResultRecord)),
        (Schema::ResultSequence, 1) => Some(repeated_message(Schema::ResultValue)),
        (Schema::ResultRecord, 1) => Some(repeated_message(Schema::ResultField)),
        (Schema::ResultField, 1) => Some(string()),
        (Schema::ResultField, 2) => Some(message(Schema::ResultValue)),

        (Schema::FamilySelection, 1 | 2) => Some(scalar()),
        (Schema::FamilySelection, 3) => Some(repeated_message(Schema::FamilyDispatchParameter)),
        (Schema::FamilySelection, 4) => Some(repeated_message(Schema::FamilyPredicate)),
        (Schema::FamilySelection, 5) => Some(repeated_message(Schema::FamilyVariant)),
        (Schema::FamilyDispatchParameter, 1) => Some(scalar()),
        (Schema::FamilyDispatchParameter, 2) => Some(string()),
        (Schema::FamilyPredicate, 1) => Some(repeated_message(Schema::ResultScalar)),
        (Schema::FamilyPredicate, 2) => Some(scalar()),
        (Schema::FamilyVariant, 1) => Some(repeated_message(Schema::FamilyDecision)),
        (Schema::FamilyDecision, 1 | 2) => Some(scalar()),

        (Schema::HirPackage, 1 | 2) => Some(scalar()),
        (Schema::HirPackage, 3 | 4) => Some(string()),
        (Schema::HirPackage, 5) => Some(repeated_message(Schema::HirFunction)),
        (Schema::HirPackage, 6) => Some(repeated_message(Schema::HirConstant)),
        (Schema::HirPackage, 7) => Some(Field::Bytes),
        (Schema::HirFunction, 1 | 5) => Some(string()),
        (Schema::HirFunction, 2) => Some(repeated_message(Schema::HirParameter)),
        (Schema::HirFunction, 3) => Some(message(Schema::Type)),
        (Schema::HirFunction, 4) => Some(message(Schema::HirBlock)),
        (Schema::HirFunction, 6) => Some(message(Schema::Span)),
        (Schema::HirFunction, 7) => Some(scalar()),
        (Schema::HirParameter, 1) => Some(string()),
        (Schema::HirParameter, 2) => Some(message(Schema::Type)),
        (Schema::HirParameter, 3) => Some(message(Schema::Literal)),
        (Schema::HirParameter, 4) => Some(scalar()),
        (Schema::HirConstant, 1) => Some(string()),
        (Schema::HirConstant, 2) => Some(message(Schema::Type)),
        (Schema::HirConstant, 3) => Some(message(Schema::Literal)),
        (Schema::HirBlock, 1) => Some(repeated_message(Schema::HirStatement)),
        (Schema::HirStatement, 1) => Some(string()),
        (Schema::HirStatement, 2) => Some(message(Schema::HirExpression)),
        (Schema::HirStatement, 3) => Some(repeated_string()),
        (Schema::HirStatement, 4) => Some(repeated_message(Schema::Attribute)),
        (Schema::HirStatement, 5) => Some(repeated_message(Schema::HirBlock)),
        (Schema::HirStatement, 6) => Some(message(Schema::Span)),
        (Schema::HirStatement, 7) => Some(repeated_message(Schema::Type)),
        (Schema::HirExpression, 1) => Some(string()),
        (Schema::HirExpression, 2) => Some(message(Schema::Type)),
        (Schema::HirExpression, 3) => Some(repeated_message(Schema::HirExpression)),
        (Schema::HirExpression, 4) => Some(repeated_message(Schema::Attribute)),
        (Schema::HirExpression, 5) => Some(message(Schema::Span)),
        (Schema::HirExpression, 6) => Some(repeated_message(Schema::HirExpression)),

        (Schema::ResourceAnalysisRequest, 1 | 2) => Some(scalar()),
        (Schema::ResourceAnalysisRequest, 3 | 5) => Some(Field::Bytes),
        (Schema::ResourceAnalysisRequest, 4) => Some(repeated_message(Schema::ResourceRule)),
        (Schema::ResourceStudy, 1 | 2) => Some(scalar()),
        (Schema::ResourceStudy, 3) => Some(message(Schema::ResourceAnalysisBundle)),
        (Schema::ResourceStudy, 4) => Some(repeated_message(Schema::ResourceMetricValue)),
        (Schema::ResourceStudy, 5) => Some(repeated_string()),
        (Schema::ResourceStudy, 6) => Some(Field::Bytes),
        (Schema::ResourceAnalysisBundle, 1 | 2) => Some(string()),
        (Schema::ResourceAnalysisBundle, 3) => Some(Field::Bytes),
        (Schema::ResourceAnalysisBundle, 4) => Some(repeated_message(Schema::ResourceAnalysisFact)),
        (Schema::ResourceAnalysisFact, 1..=3) => Some(string()),
        (Schema::ResourceAnalysisFact, 4) => Some(repeated_message(Schema::ResourceAttribute)),
        (Schema::ResourceAnalysisFact, 5) => Some(message(Schema::ResourceOperationFact)),
        (Schema::ResourceOperationFact, 1) => Some(string()),
        (Schema::ResourceOperationFact, 2) => Some(repeated_message(Schema::ResourceContribution)),
        (Schema::ResourceContribution, 1) => Some(string()),
        (Schema::ResourceContribution, 2) => Some(scalar()),
        (Schema::ResourceContribution, 3..=5) => Some(message(Schema::ResourceExpressionGraph)),
        (Schema::ResourceContribution, 6) => Some(repeated_string()),
        (Schema::ResourceMetricValue, 1) => Some(string()),
        (Schema::ResourceMetricValue, 2) => Some(message(Schema::ResourceValue)),
        (Schema::ResourceValue, 1) => Some(message(Schema::ResourceMetric)),
        (Schema::ResourceValue, 2) => Some(scalar()),
        (Schema::ResourceValue, 3..=5) => Some(message(Schema::ResourceExpressionGraph)),
        (Schema::ResourceValue, 6 | 8) => Some(repeated_string()),
        (Schema::ResourceValue, 7) => Some(repeated_message(Schema::ResourceEvidence)),
        (Schema::ResourceEvidence, 1) => Some(scalar()),
        (Schema::ResourceEvidence, 2) => Some(string()),
        (Schema::ResourceEvidence, 3) => Some(repeated_message(Schema::ResourceAttribute)),
        (Schema::ResourceRule, 1 | 2 | 10) => Some(string()),
        (Schema::ResourceRule, 3) => Some(message(Schema::ResourceMetric)),
        (Schema::ResourceRule, 4 | 9) => Some(scalar()),
        (Schema::ResourceRule, 5) => Some(message(Schema::ResourceExpressionGraph)),
        (Schema::ResourceRule, 6) => Some(repeated_scalar()),
        (Schema::ResourceRule, 7 | 8) => Some(repeated_string()),
        (Schema::ResourceRule, 11) => Some(message(Schema::ResourceSchedulingFootprint)),
        (Schema::ResourceMetric, 1 | 4 | 6) => Some(string()),
        (Schema::ResourceMetric, 2 | 3 | 5) => Some(scalar()),
        (Schema::ResourceExpressionGraph, 1) => Some(repeated_message(Schema::ResourceNode)),
        (Schema::ResourceExpressionGraph, 2) => Some(scalar()),
        (Schema::ResourceNode, 1) => Some(message(Schema::ResourceExpressionNode)),
        (Schema::ResourceNode, 2) => Some(message(Schema::ResourcePredicateNode)),
        (Schema::ResourceExpressionNode, 1 | 3) => Some(string()),
        (Schema::ResourceExpressionNode, 2) => Some(scalar()),
        (Schema::ResourceExpressionNode, 4 | 8) => Some(repeated_scalar()),
        (Schema::ResourceExpressionNode, 5) => Some(message(Schema::ResourceNumber)),
        (Schema::ResourceExpressionNode, 6) => Some(message(Schema::ResourceSymbol)),
        (Schema::ResourceExpressionNode, 7) => Some(repeated_message(Schema::ResourceAttribute)),
        (Schema::ResourcePredicateNode, 1) => Some(string()),
        (Schema::ResourcePredicateNode, 2 | 3) => Some(repeated_scalar()),
        (Schema::ResourceNumber, 1) => Some(string()),
        (Schema::ResourceNumber, 2) => Some(message(Schema::ResourceRational)),
        (Schema::ResourceNumber, 3) => Some(Field::Fixed64),
        (Schema::ResourceRational, 1 | 2) => Some(string()),
        (Schema::ResourceSymbol, 1 | 2 | 4 | 5) => Some(string()),
        (Schema::ResourceSymbol, 3) => Some(scalar()),
        (Schema::ResourceSymbol, 6) => Some(repeated_string()),
        (Schema::ResourceAttribute, 1 | 2) => Some(string()),
        (Schema::ResourceSchedulingFootprint, 1) => Some(repeated_string()),

        (Schema::Package, 1 | 2) => Some(scalar()),
        (Schema::Package, 3 | 4 | 9) => Some(string()),
        (Schema::Package, 5) => Some(repeated_message(Schema::Function)),
        (Schema::Package, 6) => Some(repeated_message(Schema::Constant)),
        (Schema::Package, 10) => Some(repeated_message(Schema::Binding)),
        (Schema::Package, 11) => Some(Field::Bytes),

        (Schema::Function, 1 | 10) => Some(string()),
        (Schema::Function, 2) => Some(repeated_message(Schema::Parameter)),
        (Schema::Function, 3) => Some(message(Schema::Type)),
        (Schema::Function, 4) => Some(message(Schema::Block)),
        (Schema::Function, 9) => Some(message(Schema::Span)),
        (Schema::Function, 11) => Some(scalar()),

        (Schema::Constant, 1) => Some(string()),
        (Schema::Constant, 2) => Some(message(Schema::Type)),
        (Schema::Constant, 3) => Some(message(Schema::Literal)),
        (Schema::Binding, 1) => Some(string()),
        (Schema::Binding, 2) => Some(message(Schema::Literal)),
        (Schema::Parameter, 1) => Some(string()),
        (Schema::Parameter, 2) => Some(message(Schema::Value)),
        (Schema::Parameter, 3) => Some(message(Schema::Literal)),
        (Schema::Parameter, 4) => Some(scalar()),

        (Schema::Type, 1) => Some(scalar()),
        (Schema::Type, 2) => Some(string()),
        (Schema::Type, 3) => Some(repeated_message(Schema::Type)),
        (Schema::Type, 4) => Some(repeated_message(Schema::Dimension)),
        (Schema::Type, 5) => Some(repeated_string()),
        (Schema::Dimension, 1) => Some(scalar()),
        (Schema::Dimension, 2) => Some(string()),

        (Schema::Literal, 1..=3) => Some(scalar()),
        (Schema::Literal, 4) => Some(Field::Fixed64),
        (Schema::Literal, 5) => Some(string()),
        (Schema::Literal, 6 | 10) => Some(Field::Bytes),
        (Schema::Literal, 7) => Some(message(Schema::LiteralSequence)),
        (Schema::Literal, 8) => Some(message(Schema::LiteralMapping)),
        (Schema::Literal, 9) => Some(message(Schema::LiteralComplex)),
        (Schema::LiteralSequence, 1) => Some(repeated_message(Schema::Literal)),
        (Schema::LiteralMapping, 1) => Some(repeated_message(Schema::LiteralEntry)),
        (Schema::LiteralEntry, 1) => Some(string()),
        (Schema::LiteralEntry, 2) => Some(message(Schema::Literal)),
        (Schema::LiteralComplex, 1 | 2) => Some(Field::Fixed64),

        (Schema::Value, 1 | 5) => Some(scalar()),
        (Schema::Value, 2) => Some(message(Schema::Type)),
        (Schema::Value, 3) => Some(string()),
        (Schema::Value, 4) => Some(message(Schema::RegionRef)),
        (Schema::RegionRef, 1) => Some(scalar()),
        (Schema::RegionRef, 2) => Some(repeated_string()),

        (Schema::Block, 1) => Some(repeated_message(Schema::Value)),
        (Schema::Block, 2) => Some(repeated_message(Schema::Operation)),
        (Schema::Block, 3) => Some(message(Schema::Terminator)),
        (Schema::Operation, 1) => Some(string()),
        (Schema::Operation, 2) => Some(repeated_scalar()),
        (Schema::Operation, 3) => Some(repeated_message(Schema::Value)),
        (Schema::Operation, 4) => Some(repeated_message(Schema::Attribute)),
        (Schema::Operation, 5) => Some(repeated_message(Schema::Region)),
        (Schema::Operation, 7) => Some(message(Schema::Span)),
        (Schema::Region, 1) => Some(repeated_message(Schema::Block)),
        (Schema::Terminator, 1) => Some(string()),
        (Schema::Terminator, 2) => Some(repeated_scalar()),
        (Schema::Terminator, 3) => Some(repeated_message(Schema::Attribute)),
        (Schema::Terminator, 4) => Some(message(Schema::Span)),
        (Schema::Attribute, 1) => Some(string()),
        (Schema::Attribute, 2) => Some(message(Schema::Literal)),
        (Schema::Span, 1) => Some(string()),
        (Schema::Span, 2..=5) => Some(scalar()),
        _ => None,
    }
}

fn read_varint(payload: &[u8], cursor: &mut usize) -> Result<u64, PreflightError> {
    let mut value = 0_u64;
    for index in 0..10_u32 {
        let byte = *payload
            .get(*cursor)
            .ok_or_else(|| malformed("truncated Protobuf varint"))?;
        *cursor += 1;
        if index == 9 && byte > 1 {
            return Err(malformed("overflowing Protobuf varint"));
        }
        value |= u64::from(byte & 0x7f) << (index * 7);
        if byte & 0x80 == 0 {
            return Ok(value);
        }
    }
    Err(malformed("overflowing Protobuf varint"))
}

fn take_length_delimited<'a>(
    payload: &'a [u8],
    cursor: &mut usize,
) -> Result<&'a [u8], PreflightError> {
    let length = read_varint(payload, cursor)?;
    let length = usize::try_from(length)
        .map_err(|_| malformed("Protobuf field length exceeds platform capacity"))?;
    let start = *cursor;
    advance(payload, cursor, length)?;
    Ok(&payload[start..*cursor])
}

fn advance(payload: &[u8], cursor: &mut usize, count: usize) -> Result<(), PreflightError> {
    let end = cursor
        .checked_add(count)
        .ok_or_else(|| malformed("Protobuf field length exceeds platform capacity"))?;
    if end > payload.len() {
        return Err(malformed("truncated Protobuf field"));
    }
    *cursor = end;
    Ok(())
}

fn skip_unknown_field(
    wire_type: u8,
    payload: &[u8],
    cursor: &mut usize,
) -> Result<(), PreflightError> {
    match wire_type {
        0 => {
            read_varint(payload, cursor)?;
        }
        1 => advance(payload, cursor, 8)?,
        2 => {
            take_length_delimited(payload, cursor)?;
        }
        5 => advance(payload, cursor, 4)?,
        _ => return Err(malformed("unsupported Protobuf wire type")),
    }
    Ok(())
}

fn malformed(message: &str) -> PreflightError {
    PreflightError::Malformed(message.to_owned())
}

enum Node<'a> {
    Package(&'a wire::QCorePackageProto),
    Function(&'a wire::FunctionProto),
    Constant(&'a wire::ConstantProto),
    Binding(&'a wire::BindingProto),
    Parameter(&'a wire::ParameterProto),
    Type(&'a wire::TypeProto),
    Dimension(&'a wire::DimensionProto),
    Literal(&'a wire::LiteralProto),
    LiteralSequence(&'a wire::literal_proto::Sequence),
    LiteralMapping(&'a wire::literal_proto::Mapping),
    LiteralEntry(&'a wire::literal_proto::Entry),
    LiteralComplex,
    Value(&'a wire::ValueProto),
    RegionRef(&'a wire::RegionRefProto),
    Block(&'a wire::BlockProto),
    Operation(&'a wire::OperationProto),
    Region(&'a wire::RegionProto),
    Terminator(&'a wire::TerminatorProto),
    Attribute(&'a wire::AttributeProto),
    Span(&'a wire::SourceSpanProto),
}

/// Validate the same decoded-structure budgets as the Python persistence layer.
// Keep the exhaustive wire walk in one match so adding a schema node produces
// one compiler-visible review point instead of a partially visited message.
#[allow(clippy::too_many_lines)]
pub(crate) fn validate(package: &wire::QCorePackageProto) -> Result<(), String> {
    let mut stack = vec![(Node::Package(package), 0_usize)];
    let mut nodes = 0_usize;
    while let Some((node, depth)) = stack.pop() {
        charge(&mut nodes, 1)?;
        if depth > MAX_NESTING {
            return Err("QCore package exceeds the nesting limit".to_owned());
        }
        let child_depth = depth.saturating_add(1);
        match node {
            Node::Package(value) => {
                strings([&value.artifact_kind, &value.module_name, &value.entrypoint])?;
                bytes(&value.package_digest)?;
                charge_repeated(&mut nodes, value.functions.len())?;
                charge_repeated(&mut nodes, value.constants.len())?;
                charge_repeated(&mut nodes, value.const_bindings.len())?;
                stack.extend(
                    value
                        .functions
                        .iter()
                        .map(|item| (Node::Function(item), child_depth)),
                );
                stack.extend(
                    value
                        .constants
                        .iter()
                        .map(|item| (Node::Constant(item), child_depth)),
                );
                stack.extend(
                    value
                        .const_bindings
                        .iter()
                        .map(|item| (Node::Binding(item), child_depth)),
                );
            }
            Node::Function(value) => {
                strings([&value.symbol, &value.asserted_kind])?;
                charge_repeated(&mut nodes, value.parameters.len())?;
                stack.extend(
                    value
                        .parameters
                        .iter()
                        .map(|item| (Node::Parameter(item), child_depth)),
                );
                push_optional(
                    &mut stack,
                    value.result_type.as_ref(),
                    child_depth,
                    Node::Type,
                );
                push_optional(&mut stack, value.body.as_ref(), child_depth, Node::Block);
                push_optional(&mut stack, value.span.as_ref(), child_depth, Node::Span);
            }
            Node::Constant(value) => {
                string(&value.name)?;
                push_optional(&mut stack, value.r#type.as_ref(), child_depth, Node::Type);
                push_optional(&mut stack, value.value.as_ref(), child_depth, Node::Literal);
            }
            Node::Binding(value) => {
                string(&value.name)?;
                push_optional(&mut stack, value.value.as_ref(), child_depth, Node::Literal);
            }
            Node::Parameter(value) => {
                string(&value.name)?;
                push_optional(&mut stack, value.value.as_ref(), child_depth, Node::Value);
                push_optional(
                    &mut stack,
                    value.default_value.as_ref(),
                    child_depth,
                    Node::Literal,
                );
            }
            Node::Type(value) => {
                string(&value.name)?;
                strings(value.field_names.iter())?;
                charge_repeated(&mut nodes, value.arguments.len())?;
                charge_repeated(&mut nodes, value.dimensions.len())?;
                charge_repeated(&mut nodes, value.field_names.len())?;
                stack.extend(
                    value
                        .arguments
                        .iter()
                        .map(|item| (Node::Type(item), child_depth)),
                );
                stack.extend(
                    value
                        .dimensions
                        .iter()
                        .map(|item| (Node::Dimension(item), child_depth)),
                );
            }
            Node::Dimension(value) => {
                if let Some(wire::dimension_proto::Value::Symbol(symbol)) = &value.value {
                    string(symbol)?;
                }
            }
            Node::Literal(value) => validate_literal(value, child_depth, &mut stack)?,
            Node::LiteralSequence(value) => {
                charge_repeated(&mut nodes, value.values.len())?;
                stack.extend(
                    value
                        .values
                        .iter()
                        .map(|item| (Node::Literal(item), child_depth)),
                );
            }
            Node::LiteralMapping(value) => {
                charge_repeated(&mut nodes, value.entries.len())?;
                stack.extend(
                    value
                        .entries
                        .iter()
                        .map(|item| (Node::LiteralEntry(item), child_depth)),
                );
            }
            Node::LiteralEntry(value) => {
                string(&value.key)?;
                push_optional(&mut stack, value.value.as_ref(), child_depth, Node::Literal);
            }
            Node::LiteralComplex => {}
            Node::Value(value) => {
                string(&value.hint)?;
                push_optional(&mut stack, value.r#type.as_ref(), child_depth, Node::Type);
                push_optional(
                    &mut stack,
                    value.region.as_ref(),
                    child_depth,
                    Node::RegionRef,
                );
            }
            Node::RegionRef(value) => {
                strings(value.projections.iter())?;
                charge_repeated(&mut nodes, value.projections.len())?;
            }
            Node::Block(value) => {
                charge_repeated(&mut nodes, value.arguments.len())?;
                charge_repeated(&mut nodes, value.operations.len())?;
                stack.extend(
                    value
                        .arguments
                        .iter()
                        .map(|item| (Node::Value(item), child_depth)),
                );
                stack.extend(
                    value
                        .operations
                        .iter()
                        .map(|item| (Node::Operation(item), child_depth)),
                );
                push_optional(
                    &mut stack,
                    value.terminator.as_ref(),
                    child_depth,
                    Node::Terminator,
                );
            }
            Node::Operation(value) => {
                string(&value.opcode)?;
                charge_repeated(&mut nodes, value.operands.len())?;
                charge_repeated(&mut nodes, value.results.len())?;
                charge_repeated(&mut nodes, value.attributes.len())?;
                charge_repeated(&mut nodes, value.regions.len())?;
                stack.extend(
                    value
                        .results
                        .iter()
                        .map(|item| (Node::Value(item), child_depth)),
                );
                stack.extend(
                    value
                        .attributes
                        .iter()
                        .map(|item| (Node::Attribute(item), child_depth)),
                );
                stack.extend(
                    value
                        .regions
                        .iter()
                        .map(|item| (Node::Region(item), child_depth)),
                );
                push_optional(&mut stack, value.span.as_ref(), child_depth, Node::Span);
            }
            Node::Region(value) => {
                charge_repeated(&mut nodes, value.blocks.len())?;
                stack.extend(
                    value
                        .blocks
                        .iter()
                        .map(|item| (Node::Block(item), child_depth)),
                );
            }
            Node::Terminator(value) => {
                string(&value.kind)?;
                charge_repeated(&mut nodes, value.operands.len())?;
                charge_repeated(&mut nodes, value.attributes.len())?;
                stack.extend(
                    value
                        .attributes
                        .iter()
                        .map(|item| (Node::Attribute(item), child_depth)),
                );
                push_optional(&mut stack, value.span.as_ref(), child_depth, Node::Span);
            }
            Node::Attribute(value) => {
                string(&value.name)?;
                push_optional(&mut stack, value.value.as_ref(), child_depth, Node::Literal);
            }
            Node::Span(value) => string(&value.uri)?,
        }
    }
    Ok(())
}

fn validate_literal<'a>(
    literal: &'a wire::LiteralProto,
    child_depth: usize,
    stack: &mut Vec<(Node<'a>, usize)>,
) -> Result<(), String> {
    use wire::literal_proto::Value;

    match literal.value.as_ref() {
        Some(Value::StringValue(value)) => string(value),
        Some(Value::BytesValue(value) | Value::BigintValue(value)) => bytes(value),
        Some(Value::TupleValue(value)) => {
            stack.push((Node::LiteralSequence(value), child_depth));
            Ok(())
        }
        Some(Value::MappingValue(value)) => {
            stack.push((Node::LiteralMapping(value), child_depth));
            Ok(())
        }
        Some(Value::ComplexValue(_)) => {
            stack.push((Node::LiteralComplex, child_depth));
            Ok(())
        }
        Some(
            Value::UnitValue(_) | Value::BoolValue(_) | Value::IntValue(_) | Value::RealValue(_),
        )
        | None => Ok(()),
    }
}

fn push_optional<'a, T>(
    stack: &mut Vec<(Node<'a>, usize)>,
    value: Option<&'a T>,
    depth: usize,
    wrap: impl FnOnce(&'a T) -> Node<'a>,
) {
    if let Some(value) = value {
        stack.push((wrap(value), depth));
    }
}

fn charge_repeated(nodes: &mut usize, count: usize) -> Result<(), String> {
    charge(nodes, count)
}

fn charge(nodes: &mut usize, count: usize) -> Result<(), String> {
    charge_with_limit(nodes, count, MAX_MESSAGE_NODES)
}

fn charge_with_limit(nodes: &mut usize, count: usize, limit: usize) -> Result<(), String> {
    *nodes = nodes
        .checked_add(count)
        .ok_or_else(|| "QCore package exceeds the node budget".to_owned())?;
    if *nodes > limit {
        return Err("QCore package exceeds the node budget".to_owned());
    }
    Ok(())
}

fn strings<'a>(values: impl IntoIterator<Item = &'a String>) -> Result<(), String> {
    for value in values {
        string(value)?;
    }
    Ok(())
}

fn string(value: &str) -> Result<(), String> {
    if value.len() > MAX_STRING_BYTES {
        Err("QCore string exceeds the size limit".to_owned())
    } else {
        Ok(())
    }
}

fn bytes(value: &[u8]) -> Result<(), String> {
    if value.len() > MAX_BYTES_FIELD {
        Err("QCore byte field exceeds the size limit".to_owned())
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_oversized_strings() {
        let package = wire::QCorePackageProto {
            module_name: "x".repeat(MAX_STRING_BYTES + 1),
            ..Default::default()
        };

        assert_eq!(
            validate(&package),
            Err("QCore string exceeds the size limit".to_owned())
        );
    }

    #[test]
    fn rejects_excessive_type_nesting() {
        let mut nested = wire::TypeProto::default();
        for _ in 0..=MAX_NESTING {
            nested = wire::TypeProto {
                arguments: vec![nested],
                ..Default::default()
            };
        }
        let package = wire::QCorePackageProto {
            constants: vec![wire::ConstantProto {
                r#type: Some(nested),
                ..Default::default()
            }],
            ..Default::default()
        };

        assert_eq!(
            validate(&package),
            Err("QCore package exceeds the nesting limit".to_owned())
        );
    }

    #[test]
    fn preflight_rejects_repeated_messages_at_budget_plus_one() {
        let function_count = MAX_MESSAGE_NODES / 2;
        let mut payload = Vec::with_capacity(function_count * 2);
        for _ in 0..function_count {
            payload.extend_from_slice(&[0x2a, 0x00]);
        }

        assert_eq!(
            preflight_with_work(&payload),
            Err(PreflightError::Budget(
                "QCore package exceeds the node budget".to_owned()
            ))
        );
    }

    #[test]
    fn preflight_rejects_large_typed_mapping_before_entry_allocation() {
        use prost::Message;

        let entry = wire::LiteralProto {
            value: Some(wire::literal_proto::Value::MappingValue(
                wire::literal_proto::Mapping {
                    entries: ["key", "value"]
                        .into_iter()
                        .map(|key| wire::literal_proto::Entry {
                            key: key.to_owned(),
                            value: Some(crate::literals::int_literal(0)),
                        })
                        .collect(),
                },
            )),
        }
        .encode_to_vec();
        // Build only wire bytes: no decoded vector of mapping records exists.
        let mut sequence = Vec::new();
        for _ in 0..MAX_MESSAGE_NODES / 2 {
            append_length_delimited(&mut sequence, 1, &entry);
        }
        let mut literal = Vec::new();
        append_length_delimited(&mut literal, 7, &sequence);
        let mut binding = Vec::new();
        append_length_delimited(&mut binding, 1, b"values");
        append_length_delimited(&mut binding, 2, &literal);
        let mut package = Vec::new();
        append_length_delimited(&mut package, 10, &binding);

        assert_eq!(
            preflight_with_work(&package),
            Err(PreflightError::Budget(
                "QCore package exceeds the node budget".to_owned()
            ))
        );
    }

    #[test]
    fn preflight_rejects_packed_operands_at_budget_plus_one() {
        // The enclosing package/function/block/operation path costs six nodes.
        let packed_count = MAX_MESSAGE_NODES - 5;
        let packed = vec![0_u8; packed_count];
        let mut operation = Vec::with_capacity(packed.len() + 8);
        append_length_delimited(&mut operation, 2, &packed);
        let mut block = Vec::with_capacity(operation.len() + 8);
        append_length_delimited(&mut block, 2, &operation);
        let mut function = Vec::with_capacity(block.len() + 8);
        append_length_delimited(&mut function, 4, &block);
        let mut package = Vec::with_capacity(function.len() + 8);
        append_length_delimited(&mut package, 5, &function);

        assert_eq!(
            preflight_with_work(&package),
            Err(PreflightError::Budget(
                "QCore package exceeds the node budget".to_owned()
            ))
        );
    }

    #[test]
    fn resource_preflight_rejects_repeated_rules_at_budget_plus_one() {
        let rule_count = MAX_MESSAGE_NODES / 2;
        let mut payload = Vec::with_capacity(rule_count * 2);
        for _ in 0..rule_count {
            payload.extend_from_slice(&[0x22, 0x00]);
        }

        assert_eq!(
            preflight_resource_analysis_request(&payload),
            Err(PreflightError::Budget(
                "QCore package exceeds the node budget".to_owned()
            ))
        );
    }

    fn append_length_delimited(target: &mut Vec<u8>, field: u32, payload: &[u8]) {
        append_varint(target, u64::from(field) << 3 | 2);
        append_varint(
            target,
            u64::try_from(payload.len()).expect("test payload length fits u64"),
        );
        target.extend_from_slice(payload);
    }

    fn append_varint(target: &mut Vec<u8>, mut value: u64) {
        while value >= 0x80 {
            target.push(u8::try_from(value & 0x7f).expect("seven bits fit u8") | 0x80);
            value >>= 7;
        }
        target.push(u8::try_from(value).expect("terminal varint byte fits u8"));
    }
}
