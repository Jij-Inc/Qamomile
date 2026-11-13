//! Native compiler boundary for canonical Qamomile semantic artifacts.
//!
//! This crate owns semantic verification, specialization, lowering, resource
//! analysis, and reference execution. Python captures source and consumes checked
//! projections through the separate `PyO3` crate; target adapters emit SDK objects.
//! Persistent inputs cross verified boundaries before compiler passes use them.
//!
//! Structured while loops retain a condition region, fixed-topology linear
//! carries, a body region, and a normal-exit region. Classical reference execution
//! and resolvable circuit projection are work-bounded; measured conditions retain
//! the `DynamicLoop` requirement and fail when the target cannot execute them.
//! Resource analysis introduces scoped natural iteration symbols independently
//! of execution. Scenario values and asserted upper bounds persist their user
//! assumptions; varying nested counts and unsupported liveness retain unknowns.
//!
//! The language permits annotated self and mutual recursion. The retained call
//! graph closes effects, coherent contracts, and host-control dependencies over
//! strongly connected components; local linear ownership rules apply to every
//! invocation. Acceptance does not prove termination. Execution and finite
//! circuit projection use bounded work, independently of resource estimation.
//! Additive resource analysis retains scoped recurrence equations for supported
//! Const integer arguments. Concrete evaluation is memoized and work-bounded,
//! independently of circuit expansion; termination and applicability are partial
//! expression contracts. Recursive depth, liveness, and unsupported stages retain
//! explicit unknown evidence.
//!
//! Target adjoint lowering derives shared inverse functions with a declaration
//! ABI rather than substituting call-site values into bodies. Formal classical
//! stages and dependent quantum types remain checked at ordinary calls. A bounded
//! worklist retains recursive graphs; inverse-only normalization joins terminal
//! owned returns before reversing structured regions. Quantum inverse pairs
//! cancel on their defined execution domain without totality or termination
//! proofs. Bounded native diagnostics reject known source-domain failures before
//! rewriting; unknown or erased evaluations need not retain runtime failures.
//!
//! Circuit placement distinguishes device feedback from terminal classical work.
//! Closed source-order quantum frontiers permit bounded correlated shot cohorts;
//! native scalar evaluation, joint populations, numeric guards and residual device
//! requirements remain compiler facts. SDK emission and submissions are transport.
//! The initial plan supports finite acyclic projections, preserves preparations,
//! and rejects unsupported ordered checks before provider execution.
//!
//! Source `sqrt` preserves scalar staging and returns Real for Int or Real inputs.
//! The native numeric algebra checks finite, nonnegative binary64 operands at
//! specialization or execution. Symbolic costs retain the square-root expression
//! until evaluation; target adapters do not define a second numeric semantics.
//!
//! Param casts to Real use the same native launch-evaluation boundary. Exact
//! integer arithmetic precedes each binary64 rounding step; SDK parameter
//! expressions never replace or erase a source numeric conversion. Unrelated
//! scalar parameters remain symbolic, and measurement guards preserve lazy
//! per-shot evaluation.

use prost::Message;
use sha2::{Digest, Sha256};
use std::sync::Arc;
use thiserror::Error;

mod artifacts;
mod core;
mod hir;
mod lowering;
mod math;
mod passes;
mod resources;

pub(crate) use artifacts::{
    abi, artifact, artifact_abi, artifact_family, execution_snapshot, expectation, result_codec,
    result_postprocessing, runtime, sample_counts, simulation, statistics, wire,
};
pub(crate) use core::{
    attributes, budget, call_graph, contracts, control_dependence, control_state, declarations,
    domains, effects, equations, literals, numeric_finiteness, numeric_values, operation_catalog,
    operations, structure, target_profile, type_relations, types, verification,
};
pub(crate) use lowering::{
    circuit, family_selection, launch_materializer, pauli_evolution, pauli_measurement,
    phase_lifting, planning, selection,
};
pub(crate) use math::{
    binary64, fqaoa, grover, lcu, local_search, lookup, number_theory, observable,
    optimization_problem, pauli_correlation, periodic_shift, polynomial, product_formula, qrac,
    qsvt, reciprocal_qsvt, state_preparation,
};
pub(crate) use passes::{
    adjoint_lowering, cleanup, cleanup_synthesis, evaluation, reversible_regions,
    semantic_identity, specialization, standard_optimization, static_values,
};
pub(crate) use resources::{
    resource_expression_evaluation, resource_rules, resource_study, symbolic_resources,
};

pub use adjoint_lowering::{
    AdjointLoweringError, TargetLoweredProgram, lower_adjoint_calls_for_target,
    lower_optimized_adjoint_calls_for_target,
};
pub use artifact::{
    ArtifactError, ArtifactHeader, MAX_ARTIFACT_BYTES, inspect_artifact, seal_artifact,
};
pub use artifact_abi::{ArtifactAbiError, DerivedArtifactAbi, ExecutionInterfaceError};
pub use artifact_family::{
    ArtifactFamilyError, ArtifactFamilyPayload, ArtifactFamilyVariant, ArtifactFamilyVariantInput,
    MAX_ARTIFACT_FAMILY_BYTES, inspect_artifact_family_payload, seal_artifact_family_payload,
};
pub use circuit::{
    CircuitDefinition, CircuitDispatchDecision, CircuitDispatchParameter, CircuitDispatchPredicate,
    CircuitLaunchBinding, CircuitLaunchValue, CircuitParameter, CircuitParameterPath,
    CircuitScalarAttribute, CircuitScalarAttributeValue, CircuitScalarId, CircuitScalarNode,
    CircuitScalarType, StaticBlock, StaticCircuit, StaticCircuitError, StaticCircuitFamily,
    StaticCircuitResultValue, StaticCircuitVariant, StaticCircuitWork, StaticInstruction,
    validate_circuit_selectors,
};
pub use cleanup_synthesis::CleanupSynthesisError;
pub use contracts::CallableContract;
pub use core::visualization::{
    AlgorithmDisplay, DisplayKind, DisplayNode, DisplayValue, LoopDisplay, MAX_VISUALIZATION_ITEMS,
    VisualizationError,
};
pub use effects::{
    CallableKind, Capability, CapabilitySet, Effect, EffectSet, Requirement, RequirementSet,
};
pub use evaluation::{ReferenceValueError, evaluate_classical_literals};
pub use execution_snapshot::{
    EXECUTION_SNAPSHOT_KIND, ExecutionSnapshotError, ExecutionSnapshotHeader,
    MAX_EXECUTION_CONTEXT_ENTRIES, MAX_EXECUTION_SNAPSHOT_BYTES, MAX_EXECUTION_SNAPSHOT_DEPTH,
    MAX_EXECUTION_SNAPSHOT_NODES, MAX_EXECUTION_SNAPSHOT_REFERENCES, MAX_EXECUTION_STRING_BYTES,
    inspect_execution_snapshot, seal_execution_snapshot,
};
pub use expectation::{
    CommutingPauliMeasurementGroup, CommutingPauliMeasurementPlan, CommutingPauliReadout,
    MAX_COMMUTING_MEASUREMENT_GATES, MAX_EXPECTATION_TERMS, MAX_EXPECTATION_WORK,
    PauliExpectationError, PauliMeasurementGroup, PauliMeasurementPlan, estimate_sample_parity,
    plan_commuting_pauli_expectation, plan_independent_pauli_expectation,
    plan_qwc_pauli_expectation,
};
pub use family_selection::{
    CircuitFamilySelectionError, CircuitFamilySelectionPlan, CircuitFamilySelectionVariant,
    encode_circuit_family_selection, inspect_circuit_family_selection,
    select_circuit_family_variant,
};
pub use fqaoa::{
    FermionicMode, FqaoaDecodedSample, FqaoaError, FqaoaPlan, MAX_FQAOA_DECODE_CELLS,
    MAX_FQAOA_DECODE_ROWS, MAX_FQAOA_GIVENS_WORK, MAX_FQAOA_MODES, MAX_FQAOA_ORBITAL_CELLS,
};
pub use grover::{GroverPlanError, MAX_GROVER_QUBITS, grover_iteration_count};
pub use hir::{
    HirError, HirHeader, HirLoweringError, HirNormalizationWork, HirSsaLoweringWork,
    HirValidationWork,
};
pub use launch_materializer::{
    LaunchMaterializerError, LaunchMaterializerPayload, MAX_LAUNCH_MATERIALIZER_BYTES,
    inspect_launch_materializer_artifact, inspect_launch_materializer_payload,
    seal_launch_materializer_payload,
};
pub use lcu::{
    LcuCoefficientPlan, LcuCoefficientTerm, LcuError, MAX_LCU_CASE_COUNT, MAX_LCU_INPUT_TERMS,
    plan_lcu_coefficients,
};
pub use local_search::{
    ImprovementMethod, LocalSearchError, LocalSearchResult, MAX_LOCAL_SEARCH_STEPS,
    MAX_LOCAL_SEARCH_SWEEP_WORK, MAX_LOCAL_SEARCH_TERMS, MAX_LOCAL_SEARCH_VARIABLES,
    MAX_LOCAL_SEARCH_WORK, minimize_polynomial, minimize_sparse_polynomial,
};
pub use lookup::{
    LookupError, MAX_LOOKUP_FLIPS, MAX_LOOKUP_ROWS, MAX_LOOKUP_WIDTH, SparseLookupPlan,
    XorLookupPlan, plan_sparse_lookup, plan_xor_lookup,
};
pub use lowering::arithmetic::{
    ArithmeticGate, ArithmeticPlan, ArithmeticStep, MAX_ARITHMETIC_SELECTORS, MAX_ARITHMETIC_STEPS,
    plan_addition,
};
pub use lowering::arithmetic_permutation::{
    ArithmeticOperation, MAX_PERMUTATION_WIDTH, plan_arithmetic_permutation,
};
pub use lowering::modular_realization::{
    MAX_MODULAR_REALIZATION_STEPS, ModularGate, ModularGatePlan, ModularRealizationPlan,
    ModularStep, plan_mod_pow, plan_schoolbook_mod_mul,
};
pub use lowering::operation_realization::{
    MAX_OPERATION_REALIZATIONS, OperationRealizationDefinition, OperationRealizationError,
    SemanticOperation, realize_module, realize_program,
};
pub use lowering::realization::{
    ConfiguredRealization, CosetPaddingBits, LoweringConfig, MAX_COSET_PADDING_BITS, Realization,
    RealizationCapabilities, RealizationError, RealizationFamily, RealizationInfo,
};
pub use lowering::shot_execution::{
    ClosedQuantumFrontier, MAX_SHOT_COHORT_CELLS, MAX_SHOT_COHORT_ROWS, MAX_SHOT_SUBMISSIONS,
    ShotExecutionError, ShotExecutionRecipe, ShotExecutionRequest, ShotExecutionState,
    ShotSeedPolicy, inspect_shot_execution_recipe, seal_shot_execution_recipe,
    shot_segment_result_abi, validate_shot_segment_interface,
};
pub use math::signal_algebra::{
    MAX_SIGNAL_BRANCHES, MAX_SIGNAL_DEPTH, MAX_SIGNAL_NODES, SignalContract, SignalError,
    signal_embedding, signal_product, signal_sum,
};
pub use math::signal_readout::{
    SignalEstimate, signal_estimate, signal_schedule, signal_schedule_response,
};
pub use math::signal_response::{SignalResponse, SignalResponsePlan, signal_response_plan};
pub use math::signals::{MAX_SIGNAL_REGION_WIDTH, SignalRegion, SignalRegionError};
pub use number_theory::{
    EkeraHastadPlan, MAX_ARITHMETIC_WIDTH, MAX_MODULAR_MULTIPLICATION_WIDTH,
    MAX_ORDER_FINDING_PRECISION, ModularMultiplicationPlan, NumberTheoryError,
    modular_addition_residue, plan_ekera_hastad, plan_modular_multiplication,
    wrapping_addition_offsets,
};
pub use numeric_finiteness::{
    RealFinitenessOperand, RealOperationFacts, RealOperationFactsError, RealRuntimeGuard,
    real_operation_facts,
};
pub use observable::{
    DensePauliDecomposition, MAX_DENSE_MATRIX_ENTRIES, NormalizedObservable, ObservableError,
    Pauli, ProjectedSubspace, add_scaled_hermitian_matrices, add_scaled_observable,
    affine_observable, decompose_hermitian_matrix, decompose_pauli_matrix, multiply_observables,
    normalize_observable, normalize_real_observable, observable_is_pairwise_commuting,
    observable_lcu_one_norm, pauli_words_commute, project_pauli_subspace, remap_observable,
    scale_hermitian_matrix, validate_hermitian_matrix,
};
pub use operation_catalog::{OperationCatalogEntry, operation_catalog, terminator_catalog};
pub use operations::{
    ClassicalTypeInferenceError, PrimitiveGateSpec, infer_classical_type, primitive_gate_specs,
};
pub use optimization_problem::{
    MAX_OPTIMIZATION_PROBLEM_BYTES, OptimizationProblemError, VerifiedOptimizationProblem,
    decode_optimization_problem, encode_optimization_problem, optimization_problem_digest,
};
pub use passes::classical_cse::{ClassicalCseResult, ClassicalCseWork};
pub use passes::gate_rewrite::GateRewriteRule;
pub use passes::optimization_pipeline::{
    MAX_OPTIMIZATION_PASSES, OptimizationError, OptimizationPassIdentity, OptimizationPolicy,
    OptimizationPreparation, validate_optimization_pipeline,
};
pub use passes::static_diagnostics::StaticDomainError;
pub use pauli_correlation::{
    CorrelationPauli, CorrelationWord, MAX_PCE_FACTORS, MAX_PCE_ORDER, MAX_PCE_QUBITS,
    PauliCorrelationError, PauliCorrelationPlan, minimum_pce_qubits,
};
pub use pauli_evolution::{
    MAX_PAULI_EVOLUTION_COMMUTATION_WORK, MAX_PAULI_EVOLUTION_STEPS, PauliEvolutionError,
    PauliEvolutionGate, PauliEvolutionStep, plan_pauli_evolution_with_capabilities,
};
pub use pauli_measurement::{
    PauliMeasurementError, prepare_commuting_pauli_measurement, prepare_pauli_measurement,
    prepare_qwc_pauli_measurement, validate_estimator_source,
};
pub use periodic_shift::{
    MAX_PERIODIC_MATRIX_ENTRIES, MAX_PERIODIC_SHIFT_TERMS, MAX_PERIODIC_SHIFT_WIDTH,
    PeriodicShiftDecomposition, PeriodicShiftError, PeriodicShiftTerm,
    decompose_periodic_shift_matrix, normalize_periodic_shift_terms,
};
pub use phase_lifting::{
    BooleanBasisTransposition, BooleanLiftError, BooleanLiftOutput, BooleanLiftPlan,
    BooleanMonomial, MAX_BOOLEAN_LIFT_OUTPUT_BITS, MAX_GENERAL_BOOLEAN_PERMUTATION_BITS,
    MAX_GENERAL_BOOLEAN_PERMUTATION_INSTRUCTIONS, MAX_GENERAL_BOOLEAN_PERMUTATION_WORK,
    MAX_PHASE_LIFT_DEPTH, MAX_PHASE_LIFT_INPUT_BITS, MAX_PHASE_LIFT_PRODUCT_WORK,
    MAX_PHASE_LIFT_TERMS,
};
pub use planning::{
    TargetCapabilityPlan, TargetLoweringPlan, TargetLoweringRoute, TargetPlanningError,
};
pub use polynomial::{
    CanonicalPolynomial, DecodedPolynomialSample, DecodedPolynomialSamples, ExactBinaryEncoding,
    IntegerEncodingError, MAX_EXACT_GAS_BITS, MAX_POLYNOMIAL_COMPONENTS,
    MAX_POLYNOMIAL_SAMPLE_VALUE_BITS, MAX_POLYNOMIAL_TERMS, MAX_POLYNOMIAL_TRANSFORM_DEGREE,
    MAX_POLYNOMIAL_VARIABLES, PolynomialDomain, PolynomialError, PolynomialHamiltonianError,
    PolynomialSampleError, PolynomialTerm, convert_polynomial, normalize_polynomial,
    validate_decoded_polynomial_samples,
};
pub use product_formula::{
    MAX_SUZUKI_SCHEDULE_ENTRIES, ProductFormulaError, suzuki_product_formula_schedule,
};
pub use qrac::{
    MAX_QRAC_DECODE_CELLS, MAX_QRAC_DECODE_ROWS, MAX_QRAC_GENERATED_TERMS, QracDecodedSample,
    QracError, QracPlan, QracVariant,
};
pub use qsvt::{
    CertifiedQsvtPolynomial, MAX_CERTIFIED_QSVT_DEGREE, MAX_QSVT_PHASES,
    MAX_QSVT_SYNTHESIS_ITERATIONS, MAX_QSVT_SYNTHESIS_WORK, QsvtPhaseError, QsvtPolynomialError,
    convert_qsp_wx_phases, synthesize_qsvt_polynomial, verify_qsvt_polynomial,
};
pub use reciprocal_qsvt::{
    CertifiedReciprocalQsvt, MAX_RECIPROCAL_COEFFICIENT_WORK, MAX_RECIPROCAL_QSVT_DEGREE,
    ReciprocalQsvtError, synthesize_reciprocal_qsvt, verify_reciprocal_qsvt,
};
pub use resource_expression_evaluation::{
    ResourceExpressionEvaluation, ResourceExpressionEvaluationError,
    ResourceExpressionSubstitutionError, ResourceExpressionSymbol, ResourceHotspotCandidate,
    ResourceHotspotRankingError, ResourceScalar, ResourceSymbolResolutionError,
    ResourceValueValidationError, SurfaceCodeProjectionError, SurfaceCodeResourceProjection,
    construct_resource_expression, construct_resource_literal, construct_resource_predicate,
    construct_resource_symbol, evaluate_resource_expression, free_resource_expression_symbols,
    normalize_resource_expression, project_surface_code_resources, rank_resource_hotspots,
    resolve_resource_symbol_bindings, retag_resource_expression_unit,
    substitute_resource_expression, validate_resource_unit, validate_resource_value,
};
pub use resource_rules::{
    Domain as ResourceDomain, ResourceRuleError, VerifiedResourceAnalysisRequest,
    seal_resource_analysis_request,
};
pub use resource_study::{
    IterationAssumption, ResourceStudyError, ResourceStudyHeader, assume_resource_iterations,
    inspect_resource_study, seal_resource_study,
};
pub use resources::analysis::{
    ExactLogicalCounts, ExactResourceError, FunctionStructure, StructuralCensus,
};
pub use resources::portable::{PortableGateStep, RecipeAngle, portable_gate_recipe};
pub use result_codec::{
    NativeBitOrder, ResultCodecError, decode_native_result_bitstring, decode_result_value,
    decode_sample_values, encode_result_value, validate_result_value,
};
pub use result_postprocessing::{
    ResultPostprocessingError, StaticCircuitResultProjection, StaticCircuitResultRoute,
    encode_result_postprocessing, postprocess_static_counts, project_result_postprocessing,
};
pub use runtime::{
    InvocationBindingError, MAX_SCALAR_PARAMETER_PATH_COMPONENTS, MAX_SCALAR_PARAMETERS,
    ScalarParameterBinding, ScalarParameterBindingError, ScalarParameterLeaf,
    ScalarParameterPathComponent, ScalarParameterProjectionError, bind_invocation,
    bind_scalar_parameters, project_scalar_parameters,
};
pub use sample_counts::{
    MAX_SAMPLE_COUNT_BITS, MAX_SAMPLE_COUNT_CELLS, MAX_SAMPLE_COUNT_ROWS, MAX_SAMPLE_COUNT_WIDTH,
    SampleCountError, SampleCountSummary, normalize_sample_counts,
};
pub use simulation::{StaticSampleCounts, StaticSimulationError, sample_static_circuit};
pub use specialization::SpecializationWork;
pub use standard_optimization::{
    StandardOptimizationError, StandardOptimizationOptions, StandardOptimizationWork,
};
pub use state_preparation::{
    AmplitudeCoordinates, MAX_AMPLITUDE_COUNT, StatePreparationError, cyclic_gray_control_schedule,
    mottonen_amplitude_coordinates, mottonen_rotation_schedule,
    uniformly_controlled_rotation_coordinates,
};
pub use static_values::{
    AggregateSliceError, MAX_AGGREGATE_SLICE_ITEMS, normalize_aggregate_index,
    normalize_aggregate_slice,
};
pub use statistics::{
    MAX_STATISTICS_COUNT_BITS, MAX_STATISTICS_PATH, MAX_STATISTICS_TERMS, SampleMean,
    SampleReadout, StatisticalEstimate, StatisticsError, combine_independent_estimates,
    sample_mean,
};
pub use symbolic_resources::{
    GateBasis, ResourceExpression, ResourceExpressionId, ResourceExpressionKind,
    ResourceLiteralKind, ResourcePredicateKind, ResourceRationalLiteral, ResourceSymbol,
    SymbolicCount, SymbolicCountRelation, SymbolicLogicalCounts, SymbolicResourceError,
};
pub use target_profile::{
    MAX_TARGET_PROFILE_BYTES, MAX_TARGET_PROFILE_EDGES, MAX_TARGET_PROFILE_QUBITS, TargetProfile,
    TargetProfileError, validate_target_profile,
};
pub use type_relations::{
    FunctionArgumentBindingError, PortableDimension, StaticDimensionCandidate,
    TypeAdmissibilityProjection, TypeDimensionBinding, TypeRelationError, TypeShapeProjection,
    bind_declared_function_arguments, common_type_stage, infer_declared_function_application,
    infer_function_application, infer_literal_type, infer_type_dimensions,
    inspect_type_admissibility, inspect_type_shape, literal_conforms_type,
    substitute_type_dimensions, type_is_assignable, type_symbolic_dimensions,
    types_are_local_peers,
};
pub use types::{Dimension, ScalarType, SemanticType, Stage, TypeArena, TypeId, WordKind};
pub use verification::QCoreVerificationWork;

const MODULE_KIND: &str = "qamomile.module";
const PROGRAM_KIND: &str = "qamomile.program";
const ANALYSIS_KIND: &str = "qamomile.analysis";
const SPECIALIZATION_KIND: &str = "qamomile.specialization";
const TARGET_ADJOINT_KIND: &str = "qamomile.target-adjoint";
/// Maximum encoded size of one canonical `QCore` package.
pub const MAX_QCORE_BYTES: usize = 64 * 1024 * 1024;

/// Maximum number of decoded Protobuf values in one `QCore` package.
pub const MAX_QCORE_MESSAGE_NODES: usize = budget::MAX_MESSAGE_NODES;

/// Maximum message nesting depth in one `QCore` package.
pub const MAX_QCORE_NESTING: usize = budget::MAX_NESTING;

/// Maximum UTF-8 byte length of one `QCore` string.
pub const MAX_QCORE_STRING_BYTES: usize = budget::MAX_STRING_BYTES;

/// `QCore` schema major supported by this native compiler boundary.
pub const SCHEMA_MAJOR: u32 = 5;

/// Highest `QCore` schema minor supported by this native compiler boundary.
pub const SCHEMA_MINOR: u32 = 0;

/// Integrity-checked metadata from a canonical `QCore` package envelope.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QCoreEnvelopeHeader {
    /// Schema major carried by the package.
    pub schema_major: u32,
    /// Schema minor carried by the package.
    pub schema_minor: u32,
    /// Canonical artifact kind carried by the package.
    pub artifact_kind: String,
    /// SHA-256 digest covering the canonical unsealed package.
    pub package_digest: [u8; 32],
}

/// Diagnostic work charged by one canonical `QCore` envelope inspection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QCorePersistenceWork {
    input_bytes: usize,
    preflight_message_nodes: usize,
    canonical_encode_passes: usize,
    canonical_encoded_bytes: usize,
}

impl QCorePersistenceWork {
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

    /// Return the number of canonical Protobuf encodings performed.
    #[must_use]
    pub const fn canonical_encode_passes(self) -> usize {
        self.canonical_encode_passes
    }

    /// Return the total bytes produced by canonical encoding passes.
    #[must_use]
    pub const fn canonical_encoded_bytes(self) -> usize {
        self.canonical_encoded_bytes
    }

    fn charge_encoding(&mut self, payload: &[u8]) {
        self.canonical_encode_passes = self.canonical_encode_passes.saturating_add(1);
        self.canonical_encoded_bytes = self.canonical_encoded_bytes.saturating_add(payload.len());
    }
}

/// One complete result from the unified native Reference runtime.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReferenceExecution {
    /// Canonical source-ordered logical sample counts.
    Samples(StaticSampleCounts),
    /// One canonical portable `LiteralProto` value.
    Value(Vec<u8>),
}

/// Stable metadata extracted from a locally checked Program.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProgramHeader {
    /// `QCore` schema major carried by the image.
    pub schema_major: u32,
    /// `QCore` schema minor carried by the image.
    pub schema_minor: u32,
    /// Source module identity recorded by the frontend.
    pub module_name: String,
    /// Closed entrypoint symbol selected for compilation.
    pub entrypoint: String,
    /// Number of functions in the closed transitive call graph.
    pub function_count: usize,
    /// Number of module constants retained by the program.
    pub constant_count: usize,
    /// Number of structurally unique semantic types interned for the program.
    pub type_count: usize,
    /// Number of operations across all function and structured-region blocks.
    pub operation_count: usize,
    /// Integrity digest stored in the canonical package.
    pub package_digest: [u8; 32],
}

/// Stable metadata extracted from a verified source module.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModuleHeader {
    /// `QCore` schema major carried by the image.
    pub schema_major: u32,
    /// `QCore` schema minor carried by the image.
    pub schema_minor: u32,
    /// Source module identity recorded by the frontend.
    pub module_name: String,
    /// Number of declarations in the closed static call graph.
    pub function_count: usize,
    /// Number of module constants retained by the image.
    pub constant_count: usize,
    /// Number of structurally unique semantic types interned for the module.
    pub type_count: usize,
    /// Number of operations across all function and structured-region blocks.
    pub operation_count: usize,
    /// Integrity digest stored in the canonical package.
    pub package_digest: [u8; 32],
}

/// Opaque canonical program accepted by the native local verifier.
///
/// This object deliberately retains the exact canonical bytes, their identity,
/// parsed graph, operation facts, function-local value-type roots, and interned
/// semantic type DAG. Later verification passes can therefore consume one
/// checked identity instead of decoding or interpreting the wire independently.
#[derive(Clone, Debug)]
pub struct LocallyCheckedProgram {
    canonical_bytes: Arc<[u8]>,
    header: ProgramHeader,
    checked: Arc<CheckedProgramGraph>,
}

/// Canonical program whose cleanup claims have independent native proofs.
///
/// Construction is intentionally separate from [`LocallyCheckedProgram`]: a
/// locally coherent program may still contain frontend forget markers or
/// forged cleanup provenance. Only this type is accepted by the public native
/// inspection boundary.
#[derive(Clone, Debug)]
pub struct VerifiedProgram {
    local: LocallyCheckedProgram,
    cleanup: cleanup::VerifiedCleanupFacts,
}

/// Transient verified program whose selected optimization pipeline has run.
///
/// This typestate is deliberately not persistent: target adapters consume it
/// directly, while standard, inserted, or replacement optimizers all produce
/// the same target-facing boundary from a fully verified program.
#[derive(Clone, Debug)]
pub struct TargetPreparedProgram {
    verified: VerifiedProgram,
    late: passes::optimization_pipeline::LateOptimization,
}

/// Selected symbolic program family accepted only by analysis passes.
///
/// Unlike [`VerifiedProgram`], this boundary retains unbound `Const`
/// parameters and dependent dimensions. It cannot be lowered or executed.
#[derive(Clone, Debug)]
pub struct VerifiedProgramFamily {
    local: LocallyCheckedProgram,
    pending_cleanup_obligations: usize,
}

/// Unselected source module accepted by the native open-artifact verifier.
///
/// A verified module retains symbolic `Const` parameters but has no selected
/// entrypoint or specialization bindings. It exposes declaration contracts and
/// may later be specialized or selected for analysis by a separate lifecycle.
#[derive(Clone, Debug)]
pub struct VerifiedModule {
    local: LocallyCheckedProgram,
    header: ModuleHeader,
}

/// Fully bound selected program accepted before specialization and cleanup.
///
/// This non-executable typestate verifies the complete selected call graph,
/// bindings, dependent call ABIs, ownership, contracts, and primitive static
/// domains while deliberately retaining symbolic dimensions and cleanup
/// obligations for later native transformations.
#[derive(Clone, Debug)]
pub struct VerifiedSpecializationRequest {
    local: LocallyCheckedProgram,
}

/// Selected and partially evaluated program awaiting cleanup synthesis.
///
/// This typestate is deliberately not executable and exposes no resource,
/// circuit, or target-lowering API. It proves that native specialization
/// produced an ordinary locally coherent program, while unresolved cleanup
/// obligations still prevent promotion to [`VerifiedProgram`].
#[derive(Clone, Debug)]
pub struct LocallySpecializedProgram {
    local: LocallyCheckedProgram,
}

/// Locally coherent cleanup candidate awaiting independent proof replay.
///
/// This private typestate proves only that native region realization produced
/// ordinary locally checked `QCore`. It grants no execution or target-lowering
/// capability; only [`Self::verify`] can promote it to [`VerifiedProgram`].
#[derive(Clone, Debug)]
struct RestoredProgram {
    local: LocallyCheckedProgram,
}

impl RestoredProgram {
    fn from_package(package: wire::QCorePackageProto) -> Result<Self, CompilerError> {
        let restored = seal_package(package)?;
        Ok(Self {
            local: LocallyCheckedProgram::from_bytes(&restored)?,
        })
    }

    fn verify(self) -> Result<VerifiedProgram, CompilerError> {
        self.local.verify()
    }
}

/// Failure of the native specialization and cleanup preparation pipeline.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum ProgramPreparationError {
    /// A persistent boundary or semantic verification pass rejected the image.
    #[error("{0}")]
    Compiler(#[from] CompilerError),
    /// Cleanup generation could not construct a supported inverse proof.
    #[error("{0}")]
    Cleanup(#[from] CleanupSynthesisError),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ArtifactBoundary {
    Module,
    Program,
    AnalysisSelection,
    Analysis,
    Selection,
    Specialization,
    FullyBoundSpecialization,
    TargetAdjoint,
}

impl ArtifactBoundary {
    const fn kind(self) -> &'static str {
        match self {
            Self::Module => MODULE_KIND,
            Self::Program => PROGRAM_KIND,
            Self::AnalysisSelection | Self::Analysis => ANALYSIS_KIND,
            Self::Selection | Self::Specialization | Self::FullyBoundSpecialization => {
                SPECIALIZATION_KIND
            }
            Self::TargetAdjoint => TARGET_ADJOINT_KIND,
        }
    }
}

#[allow(dead_code)]
#[derive(Debug)]
struct CheckedProgramGraph {
    package: wire::QCorePackageProto,
    type_arena: TypeArena,
    operations: operations::LocallyCheckedOperations,
    equations: equations::LocalEquationFacts,
    call_graph: call_graph::StaticCallGraph,
    boolean_plans: phase_lifting::VerifiedBooleanLiftPlans,
    contracts: contracts::ProgramContracts,
}

impl LocallyCheckedProgram {
    /// Decode a canonical program and establish the native boundary invariants.
    ///
    /// This stage checks the persistent envelope, digest, canonical structure,
    /// closed semantic type and operation universes, function-local value/type
    /// roots, lexical dominance, local classical equations, exact linear use,
    /// ownership topology, structured-region signatures, the closed recursive-capable
    /// call graph, ordinary dependent call ABIs, primitive static domains,
    /// quantum equations, whole-program control dependence, transformed-call
    /// capabilities, transitive callable contracts, and the exact entrypoint.
    ///
    /// # Errors
    ///
    /// Returns [`CompilerError`] when any currently established local invariant
    /// is violated.
    pub fn from_bytes(payload: &[u8]) -> Result<Self, CompilerError> {
        Self::from_bytes_with_boundary(payload, ArtifactBoundary::Program)
    }

    fn from_analysis_bytes(payload: &[u8]) -> Result<Self, CompilerError> {
        Self::from_bytes_with_boundary(payload, ArtifactBoundary::Analysis)
    }

    fn from_analysis_selection_bytes(payload: &[u8]) -> Result<Self, CompilerError> {
        Self::from_bytes_with_boundary(payload, ArtifactBoundary::AnalysisSelection)
    }

    fn from_module_bytes(payload: &[u8]) -> Result<Self, CompilerError> {
        Self::from_bytes_with_boundary(payload, ArtifactBoundary::Module)
    }

    fn from_specialization_bytes(payload: &[u8]) -> Result<Self, CompilerError> {
        Self::from_bytes_with_boundary(payload, ArtifactBoundary::Specialization)
    }

    fn from_fully_bound_specialization_bytes(payload: &[u8]) -> Result<Self, CompilerError> {
        Self::from_bytes_with_boundary(payload, ArtifactBoundary::FullyBoundSpecialization)
    }

    fn from_target_adjoint_bytes(payload: &[u8]) -> Result<Self, CompilerError> {
        Self::from_bytes_with_boundary(payload, ArtifactBoundary::TargetAdjoint)
    }

    fn from_selection_bytes(payload: &[u8]) -> Result<Self, CompilerError> {
        Self::from_bytes_with_boundary(payload, ArtifactBoundary::Selection)
    }

    fn from_bytes_with_boundary(
        payload: &[u8],
        boundary: ArtifactBoundary,
    ) -> Result<Self, CompilerError> {
        Self::from_bytes_with_boundary_and_work(payload, boundary).map(|(program, _)| program)
    }

    fn from_bytes_with_boundary_and_work(
        payload: &[u8],
        boundary: ArtifactBoundary,
    ) -> Result<(Self, QCoreVerificationWork), CompilerError> {
        let DecodedArtifact {
            package,
            observed_digest,
            type_arena,
            operations,
            equations,
            call_graph,
            boolean_plans,
            contracts,
            verification_work,
        } = decode_qcore_artifact(payload, boundary)?;
        let operation_count = operations.operation_count();
        let header = ProgramHeader {
            schema_major: package.schema_major,
            schema_minor: package.schema_minor,
            module_name: package.module_name.clone(),
            entrypoint: package.entrypoint.clone(),
            function_count: package.functions.len(),
            constant_count: package.constants.len(),
            type_count: type_arena.len(),
            operation_count,
            package_digest: observed_digest,
        };
        Ok((
            Self {
                canonical_bytes: payload.into(),
                header,
                checked: Arc::new(CheckedProgramGraph {
                    package,
                    type_arena,
                    operations,
                    equations,
                    call_graph,
                    boolean_plans,
                    contracts,
                }),
            },
            verification_work,
        ))
    }

    /// Return immutable program metadata.
    #[must_use]
    pub fn header(&self) -> &ProgramHeader {
        &self.header
    }

    /// Return the exact canonical bytes covered by [`Self::header`]'s digest.
    #[must_use]
    pub fn canonical_bytes(&self) -> &[u8] {
        self.canonical_bytes.as_ref()
    }

    /// Return the program-owned hash-consed semantic type DAG.
    #[must_use]
    pub fn types(&self) -> &TypeArena {
        &self.checked.type_arena
    }

    /// Return cleanup claims that still require native proof-term verification.
    #[must_use]
    pub fn unverified_proof_obligation_count(&self) -> usize {
        self.checked.equations.proof_obligation_count()
    }

    /// Return the compiler-derived contract for one closed callable symbol.
    #[must_use]
    pub fn callable_contract(&self, symbol: &str) -> Option<CallableContract> {
        self.checked
            .call_graph
            .index(symbol)
            .and_then(|index| self.checked.contracts.contract(index))
    }

    fn callable_contracts(&self) -> impl Iterator<Item = (&str, CallableContract)> + '_ {
        self.checked.package.functions.iter().map(|function| {
            let contract = self
                .callable_contract(&function.symbol)
                .expect("locally checked function must retain a callable contract");
            (function.symbol.as_str(), contract)
        })
    }

    /// Return classical formal indices that may reach local or transitive host control.
    #[must_use]
    pub fn callable_control_parameters(&self, symbol: &str) -> Option<&[usize]> {
        let index = self.checked.call_graph.index(symbol)?;
        self.checked
            .contracts
            .control_dependence()
            .parameters(index)
    }

    /// Replay all cleanup proofs and promote this program when they close.
    ///
    /// # Errors
    ///
    /// Returns [`CompilerError::InvalidCleanupProof`] when a forget remains or
    /// a release/unprepare chain does not normalize to its exact producer.
    pub fn verify(self) -> Result<VerifiedProgram, CompilerError> {
        let cleanup = verify_cleanup(&self)?;
        Ok(VerifiedProgram {
            local: self,
            cleanup,
        })
    }
}

impl VerifiedProgram {
    /// Derive a bounded, nonpersistent algorithm display without target lowering.
    ///
    /// # Errors
    /// Returns [`VisualizationError`] for an invalid option, output/work limit,
    /// or a loop whose expansion cannot retain its structured exit semantics.
    pub fn algorithm_display(
        &self,
        loops: LoopDisplay,
        inline_depth: usize,
        max_nodes: usize,
    ) -> Result<AlgorithmDisplay, VisualizationError> {
        core::visualization::project(
            &self.local.checked.package,
            &self.local.checked.type_arena,
            &self.local.checked.operations,
            loops,
            inline_depth,
            max_nodes,
        )
    }
    /// Decode, locally check, and independently verify one canonical program.
    ///
    /// # Errors
    ///
    /// Returns [`CompilerError`] when either the local semantic boundary or
    /// cleanup proof replay fails.
    pub fn from_bytes(payload: &[u8]) -> Result<Self, CompilerError> {
        LocallyCheckedProgram::from_bytes(payload)?.verify()
    }

    /// Return immutable program metadata.
    #[must_use]
    pub fn header(&self) -> &ProgramHeader {
        self.local.header()
    }

    /// Return the exact canonical bytes covered by the program digest.
    #[must_use]
    pub fn canonical_bytes(&self) -> &[u8] {
        self.local.canonical_bytes()
    }

    /// Return the debug-erased semantic identity of this verified program.
    #[must_use]
    pub fn semantic_digest(&self) -> [u8; 32] {
        semantic_identity::package_digest(&self.local.checked.package)
    }

    /// Produce a transient, fully reverified standard-optimized program.
    ///
    /// The persistent artifact remains unchanged. This exact pass removes only
    /// semantics-preserving `QCore` identities and inverse pairs before target
    /// lowering.
    ///
    /// # Errors
    ///
    /// Returns [`StandardOptimizationError`] if bounded traversal is exceeded
    /// or the rewritten package fails sealing or complete verification.
    pub fn optimize_standard(
        &self,
    ) -> Result<(Self, StandardOptimizationWork), StandardOptimizationError> {
        standard_optimization::optimize_standard(self)
    }

    /// Simplify with a conservative preflight bound of two visits per operation.
    ///
    /// # Errors
    /// Returns an optimization error before rewriting when work or depth is exceeded.
    pub fn optimize_standard_with_budget(
        &self,
        max_work: u64,
        options: StandardOptimizationOptions,
    ) -> Result<(Self, StandardOptimizationWork), StandardOptimizationError> {
        standard_optimization::optimize_standard_bounded(self, max_work, options)
    }

    /// Check an extension result against this source without serializing a receipt.
    ///
    /// # Errors
    /// Rejects a changed entrypoint or public launch/result ABI.
    pub fn validate_optimization_result(&self, result: &Self) -> Result<(), OptimizationError> {
        passes::optimization_pipeline::validate_result(self, result)
    }

    /// Run the standard pipeline and produce the sole target-facing typestate.
    ///
    /// A separately shipped optimizer may instead return a fully verified
    /// program and wrap it with [`TargetPreparedProgram::from_verified`]. It
    /// can compose after this method through [`TargetPreparedProgram::into_verified`]
    /// or before it by passing its verified result to this method.
    ///
    /// # Errors
    ///
    /// Returns [`StandardOptimizationError`] if bounded traversal is exceeded
    /// or the rewritten package fails sealing or complete verification.
    pub fn prepare_standard(
        &self,
    ) -> Result<(TargetPreparedProgram, StandardOptimizationWork), StandardOptimizationError> {
        let (verified, work) = self.optimize_standard()?;
        Ok((TargetPreparedProgram::from_verified(verified), work))
    }

    fn prepare_for_target(&self) -> Result<TargetPreparedProgram, StandardOptimizationError> {
        self.prepare_standard().map(|(prepared, _)| prepared)
    }

    /// Iterate over every exact Boolean lift plan independently verified during decode.
    pub fn boolean_lift_plans(&self) -> impl Iterator<Item = (&str, &BooleanLiftPlan)> {
        self.local.checked.boolean_plans.iter()
    }

    /// Return the program-owned hash-consed semantic type DAG.
    #[must_use]
    pub fn types(&self) -> &TypeArena {
        self.local.types()
    }

    /// Return the number of independently discharged cleanup claims.
    #[must_use]
    pub fn discharged_cleanup_count(&self) -> usize {
        self.cleanup.discharged_count()
    }

    /// Return the compiler-derived contract for one closed callable symbol.
    #[must_use]
    pub fn callable_contract(&self, symbol: &str) -> Option<CallableContract> {
        self.local.callable_contract(symbol)
    }

    /// Iterate over every declared callable and its compiler-derived contract.
    pub fn callable_contracts(&self) -> impl Iterator<Item = (&str, CallableContract)> + '_ {
        self.local.callable_contracts()
    }

    /// Return classical formal indices that may reach local or transitive host control.
    #[must_use]
    pub fn callable_control_parameters(&self, symbol: &str) -> Option<&[usize]> {
        self.local.callable_control_parameters(symbol)
    }

    /// Check that quantum ownership stays inside this execution entry.
    ///
    /// Quantum helpers remain valid verified programs for composition and inspection.
    ///
    /// # Errors
    /// Returns [`ArtifactAbiError`] if an input or output recursively contains
    /// a quantum value, or a verified signature is missing.
    pub fn validate_execution_entry(&self) -> Result<(), ArtifactAbiError> {
        artifact_abi::validate_execution_entry(
            &self.local.checked.package,
            &self.local.checked.type_arena,
            &self.local.checked.operations,
        )
    }

    /// Validate classical execution entry and retained launch/result layouts.
    ///
    /// # Errors
    /// Returns a diagnostic for quantum entry ownership or malformed ABIs.
    pub fn validate_execution_interface(
        &self,
        parameters: &[u8],
        result: &[u8],
    ) -> Result<(), ExecutionInterfaceError> {
        self.validate_execution_entry()?;
        runtime::validate_parameter_abi(parameters)?;
        result_codec::checked_result_abi(result)?;
        Ok(())
    }

    /// Derive canonical launch-parameter and logical-result ABIs.
    ///
    /// # Errors
    ///
    /// Returns [`ArtifactAbiError`] when the verified entrypoint facts are
    /// unexpectedly absent or its sampled result has no fixed pure bit layout.
    pub fn artifact_abi(&self) -> Result<DerivedArtifactAbi, ArtifactAbiError> {
        let contract = self
            .callable_contract(&self.local.header.entrypoint)
            .ok_or(ArtifactAbiError::MissingEntrypoint)?;
        artifact_abi::derive(
            &self.local.checked.package,
            &self.local.checked.type_arena,
            &self.local.checked.operations,
            contract.requirements().contains(Requirement::Measurement),
        )
    }

    /// Return entrypoint control formals without sealed constant bindings.
    /// Sealed `Const` parameters remain in formal control-dependence facts but
    /// never become runtime bindings.
    ///
    /// # Errors
    ///
    /// Returns [`StaticCircuitError`] if the entrypoint control contract or
    /// parameter type is unexpectedly absent. Consumers validate supported
    /// runtime stages at their own projection boundary.
    pub fn entry_unbound_control_parameters(&self) -> Result<Vec<usize>, StaticCircuitError> {
        let package = &self.local.checked.package;
        let entry = package
            .functions
            .iter()
            .find(|function| function.symbol == package.entrypoint)
            .ok_or_else(|| StaticCircuitError::Missing("entrypoint".to_owned()))?;
        let indices = self
            .callable_control_parameters(&package.entrypoint)
            .ok_or_else(|| StaticCircuitError::Missing("entrypoint control contract".to_owned()))?;
        circuit::unbound_control_parameter_indices(package, entry, indices)
    }

    /// Plan a closed target capability profile against this entrypoint.
    ///
    /// # Errors
    ///
    /// Returns [`TargetPlanningError`] when a capability label is unknown or
    /// duplicated at the target-profile boundary.
    pub fn plan_target(
        &self,
        supported: &[String],
    ) -> Result<TargetCapabilityPlan, TargetPlanningError> {
        planning::plan(self, supported)
    }

    /// Select a transient lowering route for one complete target profile.
    ///
    /// # Errors
    ///
    /// Returns [`TargetPlanningError`] when capability labels are unknown or
    /// duplicated, or when the target promises an invalid launch-time
    /// materialization.
    pub fn plan_target_lowering(
        &self,
        supported: &[String],
        materialized: &[String],
        circuit_family: bool,
    ) -> Result<TargetLoweringPlan, TargetPlanningError> {
        planning::plan_lowering(self, supported, materialized, circuit_family)
    }

    /// Return the native lexical inventory shared by later semantic passes.
    #[must_use]
    pub fn structural_census(&self) -> StructuralCensus {
        resources::analysis::census(&self.local.checked.package)
    }

    /// Derive exact dynamic quantum-operation counts for the static subset.
    ///
    /// # Errors
    ///
    /// Returns [`ExactResourceError`] instead of fabricating a bound when the
    /// program requires symbolic loops, dynamic control, unsupported scalar
    /// domains, or external bodies without native resource rules.
    pub fn exact_logical_counts(&self) -> Result<ExactLogicalCounts, ExactResourceError> {
        resources::analysis::exact_logical_counts(
            &self.local.checked.package,
            &self.local.checked.type_arena,
            &self.local.checked.operations,
        )
    }

    /// Derive a closed symbolic enclosure of logical operation counts.
    ///
    /// # Errors
    ///
    /// Returns [`SymbolicResourceError`] when the verified program uses a
    /// construct whose finite logical cost cannot yet be represented.
    pub fn symbolic_logical_counts(&self) -> Result<SymbolicLogicalCounts, SymbolicResourceError> {
        symbolic_resources::symbolic_logical_counts(
            &self.local.checked.package,
            &self.local.checked.type_arena,
            &self.local.checked.operations,
        )
    }

    /// Project the verified static subset into a target-neutral circuit.
    ///
    /// # Errors
    ///
    /// Returns [`StaticCircuitError`] instead of inventing target semantics
    /// when the program requires dynamic values or unsupported operations.
    pub fn static_circuit(&self) -> Result<StaticCircuit, StaticCircuitError> {
        self.prepare_for_target()
            .map_err(|error| StaticCircuitError::Optimization(error.to_string()))?
            .static_circuit()
    }

    /// Project the verified static subset for one transient realization set.
    ///
    /// # Errors
    ///
    /// Returns [`StaticCircuitError`] under the same fail-closed conditions as
    /// [`Self::static_circuit`].
    pub fn static_circuit_with_lowering(
        &self,
        capabilities: LoweringConfig,
    ) -> Result<StaticCircuit, StaticCircuitError> {
        self.prepare_for_target()
            .map_err(|error| StaticCircuitError::Optimization(error.to_string()))?
            .static_circuit_with_lowering(capabilities)
    }

    /// Project a target-neutral circuit together with diagnostic compiler work.
    ///
    /// # Errors
    ///
    /// Returns [`StaticCircuitError`] under the same fail-closed conditions as
    /// [`Self::static_circuit`]. The work value is not a logical resource count.
    pub fn static_circuit_with_work(
        &self,
    ) -> Result<(StaticCircuit, StaticCircuitWork), StaticCircuitError> {
        self.prepare_for_target()
            .map_err(|error| StaticCircuitError::Optimization(error.to_string()))?
            .static_circuit_with_work()
    }

    /// Lower finite launch-time control into a family of static circuits.
    ///
    /// The compiler derives which entry parameters reach control from the
    /// verified whole-program call graph. Scalar launch controls become a
    /// bounded Boolean decision tree with at most 256 static leaves. Other
    /// launch parameters remain symbolic circuit parameters.
    ///
    /// # Errors
    ///
    /// Returns [`StaticCircuitError`] when a control-dependent launch
    /// parameter is not scalar, the decision tree exceeds its finite work
    /// budget, or one selected program is not circuit-shaped.
    pub fn static_circuit_family(&self) -> Result<StaticCircuitFamily, StaticCircuitError> {
        self.static_circuit_family_with_lowering(LoweringConfig::EMPTY)
    }

    /// Lower a finite circuit family for one transient realization set.
    ///
    /// # Errors
    ///
    /// Returns [`StaticCircuitError`] under the same fail-closed conditions as
    /// [`Self::static_circuit_family`].
    pub fn static_circuit_family_with_lowering(
        &self,
        capabilities: LoweringConfig,
    ) -> Result<StaticCircuitFamily, StaticCircuitError> {
        self.prepare_for_target()
            .map_err(|error| StaticCircuitError::Optimization(error.to_string()))?
            .static_circuit_family_with_lowering(capabilities)
    }

    /// Materialize launch-controlled topology for one exact invocation.
    ///
    /// Only entry parameters proven to reach host control are accepted as
    /// bindings. Their values may determine static branches and finite loop
    /// expansion while the emitted circuit remains target-neutral.
    ///
    /// # Errors
    ///
    /// Returns [`StaticCircuitError`] when bindings disagree with the verified
    /// control ABI, a loop exceeds the fixed compiler-work budget, or the
    /// selected invocation is not circuit-shaped.
    pub fn static_circuit_at(
        &self,
        bindings: &[CircuitLaunchBinding],
    ) -> Result<StaticCircuit, StaticCircuitError> {
        self.static_circuit_at_with_lowering(bindings, LoweringConfig::EMPTY)
    }

    /// Materialize launch-controlled topology for one transient realization set.
    ///
    /// # Errors
    ///
    /// Returns [`StaticCircuitError`] under the same fail-closed conditions as
    /// [`Self::static_circuit_at`].
    pub fn static_circuit_at_with_lowering(
        &self,
        bindings: &[CircuitLaunchBinding],
        capabilities: LoweringConfig,
    ) -> Result<StaticCircuit, StaticCircuitError> {
        self.prepare_for_target()
            .map_err(|error| StaticCircuitError::Optimization(error.to_string()))?
            .static_circuit_at_with_lowering(bindings, capabilities)
    }

    /// Materialize one static circuit from a complete exact invocation.
    ///
    /// The complete entrypoint `Param` ABI is validated first. Only parameters
    /// proven to reach host control select topology; every other binding
    /// remains available to the emitted circuit's runtime scalar DAG.
    ///
    /// # Errors
    ///
    /// Returns [`StaticCircuitError`] when the invocation disagrees with the
    /// verified ABI, launch-controlled topology exceeds a compiler budget, or
    /// the selected invocation is not circuit-shaped.
    pub fn static_circuit_for_invocation(
        &self,
        bindings: &[CircuitLaunchBinding],
    ) -> Result<StaticCircuit, StaticCircuitError> {
        self.prepare_for_target()
            .map_err(|error| StaticCircuitError::Optimization(error.to_string()))?
            .static_circuit_for_invocation(bindings)
    }

    /// Validate every exact launch parameter against the verified entry ABI.
    ///
    /// This check is independent of circuit projection and therefore gives
    /// structured launch targets one native authority for names, scalar types,
    /// record fields, and fixed aggregate shapes.
    ///
    /// # Errors
    ///
    /// Returns [`StaticCircuitError`] when bindings are missing, unknown, or
    /// disagree with the verified `Param` type tree.
    pub fn validate_launch_parameters(
        &self,
        bindings: &[CircuitLaunchBinding],
    ) -> Result<(), StaticCircuitError> {
        circuit::validate_launch_parameters(&self.local.checked.package, bindings)
    }

    /// Evaluate one complete non-sampled value invocation in native Rust.
    ///
    /// The result is encoded as the canonical existing `QCore` portable
    /// literal, so the execution boundary does not introduce a shadow value
    /// schema.
    ///
    /// # Errors
    ///
    /// Returns [`ReferenceValueError`] when the verified program contains an
    /// operation outside the migrated classical or static-circuit subsets,
    /// selects an undefined exact operation, or exhausts a bounded execution
    /// budget.
    pub fn reference_value_for_invocation(
        &self,
        bindings: &[CircuitLaunchBinding],
        seed: u64,
    ) -> Result<Vec<u8>, ReferenceValueError> {
        self.prepare_for_target()
            .map_err(|error| ReferenceValueError::Invalid(error.to_string()))?
            .reference_value_for_invocation(bindings, seed)
    }

    /// Execute one invocation through the authoritative native Reference runtime.
    ///
    /// Sampled and non-sampled programs share launch validation and select
    /// their execution machine from the compiler-derived result ABI. Pure
    /// classical values and quantum-effectful values continue to share
    /// [`Self::reference_value_for_invocation`]'s internal transfer boundary.
    ///
    /// # Errors
    ///
    /// Returns [`ReferenceValueError`] if ABI derivation, launch validation,
    /// circuit projection, sampling, or deterministic evaluation fails.
    pub fn execute_reference(
        &self,
        bindings: &[CircuitLaunchBinding],
        shots: u64,
        seed: u64,
    ) -> Result<ReferenceExecution, ReferenceValueError> {
        self.prepare_for_target()
            .map_err(|error| ReferenceValueError::Invalid(error.to_string()))?
            .execute_reference(bindings, shots, seed)
    }
}

impl TargetPreparedProgram {
    /// Lower callable adjoints without decoding or optimizing the source again.
    ///
    /// # Errors
    ///
    /// Returns [`AdjointLoweringError`] for unsupported transforms or invalid
    /// target output, preserving the ordinary target rechecking boundary.
    pub fn lower_adjoint_calls(&self) -> Result<TargetLoweredProgram, AdjointLoweringError> {
        passes::adjoint_lowering::lower_adjoint_calls(&self.verified)
    }

    /// Promote a fully verified optimizer result to the target-facing boundary.
    ///
    /// Optimization is not a semantic requirement, so replacement pipelines
    /// need no privileged constructor. Complete `QCore` verification remains
    /// the only authority required before this transient promotion.
    #[must_use]
    pub const fn from_verified(verified: VerifiedProgram) -> Self {
        Self {
            verified,
            late: passes::optimization_pipeline::LateOptimization::NONE,
        }
    }

    /// Borrow the fully verified program produced by the selected pipeline.
    #[must_use]
    pub const fn as_verified(&self) -> &VerifiedProgram {
        &self.verified
    }

    /// Recover the verified result for insertion of another optimizer pass.
    #[must_use]
    pub fn into_verified(self) -> VerifiedProgram {
        self.verified
    }

    /// Return the canonical bytes of the fully verified prepared result.
    #[must_use]
    pub fn canonical_bytes(&self) -> &[u8] {
        self.verified.canonical_bytes()
    }

    /// Project the prepared static subset into a target-neutral circuit.
    ///
    /// # Errors
    ///
    /// Returns [`StaticCircuitError`] when the program requires dynamic values
    /// or operations outside the target-neutral circuit subset.
    pub fn static_circuit(&self) -> Result<StaticCircuit, StaticCircuitError> {
        self.static_circuit_with_lowering(LoweringConfig::EMPTY)
    }

    /// Project the prepared program for one transient realization set.
    ///
    /// # Errors
    ///
    /// Returns [`StaticCircuitError`] when lowering choices or program
    /// requirements cannot be represented by a static circuit.
    pub fn static_circuit_with_lowering(
        &self,
        capabilities: LoweringConfig,
    ) -> Result<StaticCircuit, StaticCircuitError> {
        circuit::project_with_lowering(
            &self.verified.local.checked.package,
            &self.verified.local.checked.boolean_plans,
            capabilities,
        )
        .and_then(|circuit| self.late.finish(circuit))
    }

    /// Project a prepared circuit together with diagnostic compiler work.
    ///
    /// # Errors
    ///
    /// Returns [`StaticCircuitError`] when the prepared program is not in the
    /// supported target-neutral circuit subset.
    pub fn static_circuit_with_work(
        &self,
    ) -> Result<(StaticCircuit, StaticCircuitWork), StaticCircuitError> {
        circuit::project_with_work(
            &self.verified.local.checked.package,
            &self.verified.local.checked.boolean_plans,
            LoweringConfig::EMPTY,
        )
        .and_then(|(circuit, work)| self.late.finish(circuit).map(|circuit| (circuit, work)))
    }

    /// Lower the prepared program into a finite static-circuit family.
    ///
    /// # Errors
    ///
    /// Returns [`StaticCircuitError`] when control is not finitely
    /// materializable or a selected variant is not circuit-shaped.
    pub fn static_circuit_family(&self) -> Result<StaticCircuitFamily, StaticCircuitError> {
        self.static_circuit_family_with_lowering(LoweringConfig::EMPTY)
    }

    /// Lower a prepared finite circuit family for one realization set.
    ///
    /// # Errors
    ///
    /// Returns [`StaticCircuitError`] when control, lowering, or a selected
    /// variant cannot be represented by the bounded circuit-family projection.
    pub fn static_circuit_family_with_lowering(
        &self,
        capabilities: LoweringConfig,
    ) -> Result<StaticCircuitFamily, StaticCircuitError> {
        let indices = self
            .verified
            .callable_control_parameters(&self.verified.local.header.entrypoint)
            .ok_or_else(|| StaticCircuitError::Missing("entrypoint control contract".to_owned()))?;
        circuit::project_family(
            &self.verified.local.checked.package,
            &self.verified.local.checked.boolean_plans,
            indices,
            capabilities,
        )
        .and_then(|family| family.map_circuits(|circuit| self.late.finish(circuit)))
    }

    /// Materialize prepared launch-controlled topology for exact bindings.
    ///
    /// # Errors
    ///
    /// Returns [`StaticCircuitError`] when bindings or selected topology are
    /// outside the verified finite materialization contract.
    pub fn static_circuit_at(
        &self,
        bindings: &[CircuitLaunchBinding],
    ) -> Result<StaticCircuit, StaticCircuitError> {
        self.static_circuit_at_with_lowering(bindings, LoweringConfig::EMPTY)
    }

    /// Materialize prepared launch-controlled topology for one realization set.
    ///
    /// # Errors
    ///
    /// Returns [`StaticCircuitError`] when bindings, lowering, or selected
    /// topology cannot be represented by the static projection.
    pub fn static_circuit_at_with_lowering(
        &self,
        bindings: &[CircuitLaunchBinding],
        capabilities: LoweringConfig,
    ) -> Result<StaticCircuit, StaticCircuitError> {
        let indices = self
            .verified
            .callable_control_parameters(&self.verified.local.header.entrypoint)
            .ok_or_else(|| StaticCircuitError::Missing("entrypoint control contract".to_owned()))?;
        circuit::project_at(
            &self.verified.local.checked.package,
            &self.verified.local.checked.boolean_plans,
            indices,
            bindings,
            capabilities,
        )
        .and_then(|circuit| self.late.finish(circuit))
    }

    /// Materialize one prepared static circuit from a complete invocation.
    ///
    /// # Errors
    ///
    /// Returns [`StaticCircuitError`] when the invocation disagrees with its
    /// verified ABI or the selected topology is not statically circuit-shaped.
    pub fn static_circuit_for_invocation(
        &self,
        bindings: &[CircuitLaunchBinding],
    ) -> Result<StaticCircuit, StaticCircuitError> {
        let indices = self
            .verified
            .callable_control_parameters(&self.verified.local.header.entrypoint)
            .ok_or_else(|| StaticCircuitError::Missing("entrypoint control contract".to_owned()))?;
        circuit::project_for_invocation(
            &self.verified.local.checked.package,
            &self.verified.local.checked.boolean_plans,
            indices,
            bindings,
            LoweringConfig::EMPTY,
        )
        .and_then(|circuit| self.late.finish(circuit))
    }

    /// Validate exact launch parameters against the prepared entry ABI.
    ///
    /// # Errors
    ///
    /// Returns [`StaticCircuitError`] when bindings are missing, unknown, or
    /// disagree with the verified `Param` type tree.
    pub fn validate_launch_parameters(
        &self,
        bindings: &[CircuitLaunchBinding],
    ) -> Result<(), StaticCircuitError> {
        circuit::validate_launch_parameters(&self.verified.local.checked.package, bindings)
    }

    /// Evaluate one prepared non-sampled Reference invocation.
    ///
    /// # Errors
    ///
    /// Returns [`ReferenceValueError`] when evaluation, circuit projection, or
    /// bounded deterministic simulation fails.
    pub fn reference_value_for_invocation(
        &self,
        bindings: &[CircuitLaunchBinding],
        seed: u64,
    ) -> Result<Vec<u8>, ReferenceValueError> {
        match evaluation::evaluate(&self.verified.local.checked.package, bindings) {
            Ok(value) => Ok(value),
            Err(ReferenceValueError::Unsupported(_)) => {
                let circuit = self
                    .static_circuit_for_invocation(bindings)
                    .map_err(|error| reference_circuit_error(&error))?;
                simulation::evaluate_static_circuit(&circuit, bindings, seed)
                    .map_err(|error| reference_simulation_error(&error))
            }
            Err(error) => Err(error),
        }
    }

    /// Execute one prepared invocation through the native Reference runtime.
    ///
    /// # Errors
    ///
    /// Returns [`ReferenceValueError`] if ABI derivation, launch validation,
    /// circuit projection, sampling, or deterministic evaluation fails.
    pub fn execute_reference(
        &self,
        bindings: &[CircuitLaunchBinding],
        shots: u64,
        seed: u64,
    ) -> Result<ReferenceExecution, ReferenceValueError> {
        self.verified
            .validate_execution_entry()
            .map_err(|error| ReferenceValueError::Invalid(error.to_string()))?;
        let abi = self
            .verified
            .artifact_abi()
            .map_err(|error| ReferenceValueError::Invalid(error.to_string()))?;
        if abi.sampled_width() == 0 {
            let value = self.reference_value_for_invocation(bindings, seed)?;
            result_codec::validate_result_value(abi.result_abi(), &value)
                .map_err(|error| ReferenceValueError::Invalid(error.to_string()))?;
            return Ok(ReferenceExecution::Value(value));
        }
        self.validate_launch_parameters(bindings)
            .map_err(|error| reference_circuit_error(&error))?;
        let circuit = self
            .static_circuit_for_invocation(bindings)
            .map_err(|error| reference_circuit_error(&error))?;
        simulation::sample_static_circuit(&circuit, bindings, shots, seed)
            .map(ReferenceExecution::Samples)
            .map_err(|error| reference_simulation_error(&error))
    }
}

fn reference_circuit_error(error: &StaticCircuitError) -> ReferenceValueError {
    match error {
        StaticCircuitError::Unsupported(_)
        | StaticCircuitError::DynamicClassicalAggregateIndex
        | StaticCircuitError::HostDispatch(_)
        | StaticCircuitError::LaunchMaterialization(_) => {
            ReferenceValueError::Unsupported(error.to_string())
        }
        StaticCircuitError::Missing(_)
        | StaticCircuitError::Optimization(_)
        | StaticCircuitError::Realization(_)
        | StaticCircuitError::InvalidSelector(_)
        | StaticCircuitError::Overflow
        | StaticCircuitError::NonHermitianPauliEvolution
        | StaticCircuitError::NonCommutingPauliEvolution => {
            ReferenceValueError::Invalid(error.to_string())
        }
    }
}

fn reference_simulation_error(error: &StaticSimulationError) -> ReferenceValueError {
    match error {
        StaticSimulationError::Budget(_) => ReferenceValueError::WorkBudgetExceeded,
        StaticSimulationError::Unsupported(_) => {
            ReferenceValueError::Unsupported(error.to_string())
        }
        StaticSimulationError::Missing(_)
        | StaticSimulationError::Malformed(_)
        | StaticSimulationError::Invalid(_) => ReferenceValueError::Invalid(error.to_string()),
    }
}

impl VerifiedProgramFamily {
    /// Derive a bounded view while retaining unbound Const values and shapes.
    ///
    /// # Errors
    /// Returns [`VisualizationError`] when options exceed bounds or symbolic
    /// or exiting loops cannot be expanded faithfully.
    pub fn algorithm_display(
        &self,
        loops: LoopDisplay,
        inline_depth: usize,
        max_nodes: usize,
    ) -> Result<AlgorithmDisplay, VisualizationError> {
        core::visualization::project(
            &self.local.checked.package,
            &self.local.checked.type_arena,
            &self.local.checked.operations,
            loops,
            inline_depth,
            max_nodes,
        )
    }
    /// Decode and verify one selected symbolic analysis artifact.
    ///
    /// # Errors
    ///
    /// Returns [`CompilerError`] when local semantics or family selection fail.
    pub fn from_bytes(payload: &[u8]) -> Result<Self, CompilerError> {
        let selection = LocallyCheckedProgram::from_analysis_selection_bytes(payload)?;
        let family =
            selection::select_family(&selection.checked.package, &selection.checked.call_graph)
                .map_err(|error| CompilerError::InvalidSpecialization(error.to_string()))?;
        let family = seal_package(family)?;
        let local = LocallyCheckedProgram::from_analysis_bytes(&family)?;
        let pending_cleanup_obligations = local.unverified_proof_obligation_count();
        Ok(Self {
            local,
            pending_cleanup_obligations,
        })
    }

    /// Return immutable selected-family metadata.
    #[must_use]
    pub fn header(&self) -> &ProgramHeader {
        self.local.header()
    }

    /// Return cleanup obligations retained for later specialization.
    #[must_use]
    pub const fn pending_cleanup_obligation_count(&self) -> usize {
        self.pending_cleanup_obligations
    }

    /// Derive a closed symbolic enclosure while retaining unbound Const values.
    ///
    /// # Errors
    ///
    /// Returns [`SymbolicResourceError`] when finite logical cost cannot be
    /// represented by the native expression algebra.
    pub fn symbolic_logical_counts(&self) -> Result<SymbolicLogicalCounts, SymbolicResourceError> {
        symbolic_resources::symbolic_logical_counts(
            &self.local.checked.package,
            &self.local.checked.type_arena,
            &self.local.checked.operations,
        )
    }
}

impl VerifiedModule {
    /// Decode and verify one canonical unselected Module.
    ///
    /// # Errors
    ///
    /// Returns [`CompilerError`] when the artifact selects an entrypoint,
    /// carries specialization bindings, or fails an applicable semantic pass.
    pub fn from_bytes(payload: &[u8]) -> Result<Self, CompilerError> {
        let local = LocallyCheckedProgram::from_module_bytes(payload)?;
        let common = local.header();
        let header = ModuleHeader {
            schema_major: common.schema_major,
            schema_minor: common.schema_minor,
            module_name: common.module_name.clone(),
            function_count: common.function_count,
            constant_count: common.constant_count,
            type_count: common.type_count,
            operation_count: common.operation_count,
            package_digest: common.package_digest,
        };
        Ok(Self { local, header })
    }

    /// Return immutable module metadata.
    #[must_use]
    pub const fn header(&self) -> &ModuleHeader {
        &self.header
    }

    /// Return the exact canonical bytes covered by the module digest.
    #[must_use]
    pub fn canonical_bytes(&self) -> &[u8] {
        self.local.canonical_bytes()
    }

    /// Return the module-owned hash-consed semantic type DAG.
    #[must_use]
    pub fn types(&self) -> &TypeArena {
        self.local.types()
    }

    /// Return cleanup obligations deliberately deferred until specialization.
    #[must_use]
    pub fn pending_cleanup_obligation_count(&self) -> usize {
        self.local.unverified_proof_obligation_count()
    }

    /// Return the compiler-derived contract for one declared callable.
    #[must_use]
    pub fn callable_contract(&self, symbol: &str) -> Option<CallableContract> {
        self.local.callable_contract(symbol)
    }

    /// Resolve one source-facing local or fully qualified callable selector.
    ///
    /// # Errors
    ///
    /// Returns [`CompilerError`] when the selector is absent or ambiguous.
    pub fn resolve_entrypoint(&self, selector: &str) -> Result<&str, CompilerError> {
        selection::resolve_entrypoint(&self.local.checked.package, selector)
            .map(|(_, function)| function.symbol.as_str())
            .map_err(|error| CompilerError::InvalidSpecialization(error.to_string()))
    }

    /// Return the debug-erased semantic identity of one selected callable.
    ///
    /// # Errors
    ///
    /// Returns [`CompilerError`] when the selector is absent or ambiguous.
    pub fn selected_semantic_digest(&self, selector: &str) -> Result<[u8; 32], CompilerError> {
        let symbol = self.resolve_entrypoint(selector)?;
        Ok(semantic_identity::selected_digest(
            semantic_identity::package_digest(&self.local.checked.package),
            symbol,
        ))
    }

    /// Iterate over every declared callable and its compiler-derived contract.
    pub fn callable_contracts(&self) -> impl Iterator<Item = (&str, CallableContract)> + '_ {
        self.local.callable_contracts()
    }

    /// Return classical formal indices that may reach local or transitive host control.
    #[must_use]
    pub fn callable_control_parameters(&self, symbol: &str) -> Option<&[usize]> {
        self.local.callable_control_parameters(symbol)
    }
}

impl VerifiedSpecializationRequest {
    /// Decode and verify one fully bound selected transformation request.
    ///
    /// # Errors
    ///
    /// Returns [`CompilerError`] when the request is malformed, incomplete, or
    /// violates any retained `QCore` invariant or primitive static domain.
    pub fn from_bytes(payload: &[u8]) -> Result<Self, CompilerError> {
        Ok(Self {
            local: LocallyCheckedProgram::from_fully_bound_specialization_bytes(payload)?,
        })
    }

    /// Return immutable selected-program metadata.
    #[must_use]
    pub fn header(&self) -> &ProgramHeader {
        self.local.header()
    }

    /// Return unresolved cleanup work retained for native synthesis.
    #[must_use]
    pub fn pending_cleanup_obligation_count(&self) -> usize {
        self.local.unverified_proof_obligation_count()
    }

    /// Iterate over every declared callable and its compiler-derived contract.
    pub fn callable_contracts(&self) -> impl Iterator<Item = (&str, CallableContract)> + '_ {
        self.local.callable_contracts()
    }
}

impl LocallySpecializedProgram {
    /// Decode a fully bound internal request and specialize it in Rust.
    ///
    /// # Errors
    ///
    /// Returns [`CompilerError`] if the request is malformed, binding or
    /// dependent-type specialization fails, or the rewritten package does not
    /// cross the normal locally checked program boundary.
    pub fn from_request_bytes(payload: &[u8]) -> Result<Self, CompilerError> {
        Self::from_request_bytes_with_work(payload).map(|(program, _)| program)
    }

    /// Decode and specialize a request together with diagnostic compiler work.
    ///
    /// # Errors
    ///
    /// Returns [`CompilerError`] under the same fail-closed conditions as
    /// [`Self::from_request_bytes`]. The work value is not a resource count.
    pub fn from_request_bytes_with_work(
        payload: &[u8],
    ) -> Result<(Self, SpecializationWork), CompilerError> {
        let selection = LocallyCheckedProgram::from_selection_bytes(payload)?;
        let materialized = selection::materialize(
            &selection.checked.package,
            &selection.checked.type_arena,
            &selection.checked.operations,
            &selection.checked.call_graph,
        )
        .map_err(|error| CompilerError::InvalidSpecialization(error.to_string()))?;
        let materialized = seal_package(materialized)?;
        let request = LocallyCheckedProgram::from_specialization_bytes(&materialized)?;
        let (mut package, work) = specialization::specialize_with_work(&request.checked.package)
            .map_err(|error| CompilerError::InvalidSpecialization(error.to_string()))?;
        PROGRAM_KIND.clone_into(&mut package.artifact_kind);
        let specialized = seal_package(package)?;
        let local = LocallyCheckedProgram::from_bytes(&specialized)?;
        Ok((Self { local }, work))
    }

    /// Return the canonical locally checked program bytes produced by Rust.
    #[must_use]
    pub fn canonical_bytes(&self) -> &[u8] {
        self.local.canonical_bytes()
    }

    /// Return the unresolved cleanup work that prevents execution.
    #[must_use]
    pub fn pending_cleanup_obligation_count(&self) -> usize {
        self.local.unverified_proof_obligation_count()
    }

    /// Synthesize supported cleanup chains and promote through proof replay.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramPreparationError`] when synthesis is unsupported or
    /// when the generated package fails the ordinary executable boundary.
    pub fn synthesize_cleanup(self) -> Result<VerifiedProgram, ProgramPreparationError> {
        let mut package = cleanup_synthesis::synthesize(
            &self.local.checked.package,
            &self.local.checked.type_arena,
            &self.local.checked.operations,
            &self.local.checked.call_graph,
            &self.local.checked.contracts,
        )?;
        PROGRAM_KIND.clone_into(&mut package.artifact_kind);
        Ok(RestoredProgram::from_package(package)?.verify()?)
    }

    /// Specialize and synthesize one internal request into an executable image.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramPreparationError`] when any typed transformation or
    /// independent executable verification stage fails.
    pub fn prepare_request_bytes(
        payload: &[u8],
    ) -> Result<VerifiedProgram, ProgramPreparationError> {
        Self::from_request_bytes(payload)?.synthesize_cleanup()
    }

    /// Prepare a request together with its native specialization work.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramPreparationError`] under the same fail-closed
    /// conditions as [`Self::prepare_request_bytes`].
    pub fn prepare_request_bytes_with_work(
        payload: &[u8],
    ) -> Result<(VerifiedProgram, SpecializationWork), ProgramPreparationError> {
        let (program, work) = Self::from_request_bytes_with_work(payload)?;
        Ok((program.synthesize_cleanup()?, work))
    }
}

fn verify_cleanup(
    local: &LocallyCheckedProgram,
) -> Result<cleanup::VerifiedCleanupFacts, CompilerError> {
    cleanup::verify(
        &local.checked.package,
        &local.checked.type_arena,
        &local.checked.operations,
        &local.checked.equations,
        &local.checked.call_graph,
        &local.checked.contracts,
    )
    .map_err(|error| CompilerError::InvalidCleanupProof(error.to_string()))
}

/// Failures detected at the native `QCore` persistence boundary.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum CompilerError {
    /// The serialized message exceeds the shared `QCore` wire limit.
    #[error("Q4007: QCore package exceeds the wire-size limit")]
    PackageTooLarge,
    /// A decoded message exceeds a shared structural resource budget.
    #[error("Q4007: {0}")]
    PackageBudgetExceeded(String),
    /// Protobuf decoding failed.
    #[error("Q4008: malformed QCore package: {0}")]
    MalformedPackage(String),
    /// The package uses a schema major this compiler cannot interpret.
    #[error("Q4009: unsupported QCore schema major {actual}; expected {expected}")]
    UnsupportedSchemaMajor {
        /// Version observed on the wire.
        actual: u32,
        /// Version accepted by this compiler.
        expected: u32,
    },
    /// The package uses a newer schema minor than this compiler supports.
    #[error("Q4010: unsupported QCore schema minor {actual}; maximum is {maximum}")]
    UnsupportedSchemaMinor {
        /// Version observed on the wire.
        actual: u32,
        /// Highest compatible version accepted by this compiler.
        maximum: u32,
    },
    /// The artifact kind does not match the requested typed boundary.
    #[error("Q4011: expected {expected:?}, got {actual:?}")]
    UnexpectedArtifactKind {
        /// Artifact kind required by this boundary.
        expected: &'static str,
        /// Artifact kind observed on the wire.
        actual: String,
    },
    /// The package kind is outside the closed persistent artifact universe.
    #[error("Q4011: unknown QCore image kind {actual:?}")]
    UnknownArtifactKind {
        /// Artifact kind observed on the wire.
        actual: String,
    },
    /// The package digest does not cover the canonical unsealed message.
    #[error("Q4012: QCore package digest mismatch")]
    DigestMismatch,
    /// The payload has an alternative or boundary-inconsistent encoding.
    #[error("Q4014: QCore artifact is not canonically encoded")]
    NonCanonicalEncoding,
    /// The selected artifact entrypoint is absent from the static call graph.
    #[error("Q1015: program entrypoint is missing")]
    MissingEntrypoint,
    /// A wire-carried type is outside the closed semantic universe.
    #[error("Q4002: invalid semantic type: {0}")]
    InvalidSemanticType(String),
    /// A portable literal has no unique canonical closed interpretation.
    #[error("{0}")]
    InvalidPortableLiteral(String),
    /// An operation or terminator is outside the closed native dialect kernel.
    #[error("{0}")]
    InvalidOperationKernel(String),
    /// An operation or terminator carries an ambiguous attribute row.
    #[error("{0}")]
    InvalidAttributes(String),
    /// The normalized static call graph is malformed or unresolved.
    #[error("{0}")]
    InvalidCallGraph(String),
    /// Function-local SSA, ownership, or structured control is invalid.
    #[error("{0}")]
    InvalidLocalVerification(String),
    /// A function-local core or quantum equation is invalid.
    #[error("{0}")]
    InvalidLocalEquation(String),
    /// An ordinary static call violates its declared dependent ABI.
    #[error("{0}")]
    InvalidCallAbi(String),
    /// A coherent Boolean lift lacks an exact bounded proof.
    #[error("{0}")]
    InvalidBooleanLift(String),
    /// A declaration, default, constant, or specialized binding is invalid.
    #[error("{0}")]
    InvalidDeclaration(String),
    /// Native Const substitution or static-control reduction failed.
    #[error("{0}")]
    InvalidSpecialization(String),
    /// A cleanup marker or terminal release lacks an independently replayed proof.
    #[error("{0}")]
    InvalidCleanupProof(String),
    /// A statically known primitive value lies outside its semantic domain.
    #[error("{0}")]
    InvalidStaticDomain(String),
    /// Whole-program control dependence could not be reconstructed.
    #[error("{0}")]
    InvalidControlDependence(String),
    /// A closed callable contract or transformation is invalid.
    #[error("{0}")]
    InvalidCallableContract(String),
}

/// Decode and integrity-check a canonical Qamomile Program.
///
/// This validates the persistent envelope shared with Python plus the native
/// local SSA, ownership, structured-control, static-call, call ABI, local
/// equations, and whole-program callable-contract boundary. Lowering remains a
/// later compiler stage and is intentionally not claimed by this function.
///
/// # Errors
///
/// Returns [`CompilerError`] when the payload is oversized, malformed,
/// incompatible, noncanonical, corrupted, or not a closed program package.
pub fn inspect_program(payload: &[u8]) -> Result<ProgramHeader, CompilerError> {
    Ok(VerifiedProgram::from_bytes(payload)?.local.header)
}

/// Decode and independently verify a canonical Qamomile Module.
///
/// # Errors
///
/// Returns [`CompilerError`] when the payload is oversized, malformed,
/// incompatible, noncanonical, corrupted, or not an unselected module.
pub fn inspect_module(payload: &[u8]) -> Result<ModuleHeader, CompilerError> {
    Ok(VerifiedModule::from_bytes(payload)?.header)
}

/// Decode and independently verify one private typed-HIR compiler request.
///
/// # Errors
///
/// Returns [`HirError`] when the payload is malformed, noncanonical, or lies
/// outside the closed HIR grammar and semantic type universe.
pub fn inspect_hir_image(payload: &[u8]) -> Result<HirHeader, HirError> {
    hir::inspect(payload)
}

/// Inspect one typed-HIR image together with transient native validation work.
///
/// # Errors
///
/// Returns [`HirError`] under the same fail-closed conditions as
/// [`inspect_hir_image`]. The work value is not persistent compiler data.
pub fn inspect_hir_image_with_work(
    payload: &[u8],
) -> Result<(HirHeader, HirValidationWork), HirError> {
    hir::inspect_with_work(payload)
}

/// Seal one canonical unsealed typed-HIR compiler request.
///
/// # Errors
///
/// Returns [`HirError`] when the request is oversized, malformed,
/// noncanonical, incompatible, or outside the closed HIR grammar.
pub fn seal_hir_package(payload: &[u8]) -> Result<Vec<u8>, HirError> {
    hir::seal(payload)
}

/// Lower the currently migrated typed-HIR subset into a verified `QCore` module.
///
/// # Errors
///
/// Returns [`HirLoweringError`] when the request is invalid, lies outside the
/// normally yielding single-owner subset, or cannot cross the ordinary module boundary.
pub fn lower_hir_image(payload: &[u8]) -> Result<Vec<u8>, HirLoweringError> {
    hir::lower(payload)
}

/// Lower typed HIR together with transient final ownership-SSA work.
///
/// # Errors
///
/// Returns [`HirLoweringError`] under the same fail-closed conditions as
/// [`lower_hir_image`]. The work excludes pre-SSA normalization passes.
pub fn lower_hir_image_with_work(
    payload: &[u8],
) -> Result<(Vec<u8>, HirSsaLoweringWork), HirLoweringError> {
    hir::lower_with_work(payload)
}

/// Lower typed HIR together with transient pre-SSA normalization work.
///
/// # Errors
///
/// Returns [`HirLoweringError`] under the same fail-closed conditions as
/// [`lower_hir_image`]. The work excludes final ownership-SSA lowering.
pub fn lower_hir_image_with_normalization_work(
    payload: &[u8],
) -> Result<(Vec<u8>, HirNormalizationWork), HirLoweringError> {
    hir::lower_with_normalization_work(payload)
}

/// Inspect one canonical integrity-sealed `QCore` envelope before host decoding.
///
/// This boundary performs the borrowed wire preflight, bounded Prost decode,
/// schema check, digest verification, and canonical structural checks without
/// interpreting the program body.
///
/// # Errors
///
/// Returns [`CompilerError`] when the envelope is oversized, malformed,
/// incompatible, corrupted, noncanonical, or has an unknown artifact kind.
pub fn inspect_qcore_envelope(payload: &[u8]) -> Result<QCoreEnvelopeHeader, CompilerError> {
    inspect_qcore_envelope_with_work(payload).map(|(header, _)| header)
}

/// Inspect one canonical `QCore` envelope and retain diagnostic persistence work.
///
/// # Errors
///
/// Returns [`CompilerError`] under the same fail-closed conditions as
/// [`inspect_qcore_envelope`]. The work value is not persistent program data.
pub fn inspect_qcore_envelope_with_work(
    payload: &[u8],
) -> Result<(QCoreEnvelopeHeader, QCorePersistenceWork), CompilerError> {
    let (package, package_digest, work) = decode_qcore_envelope_with_work(payload, None)?;
    Ok((
        QCoreEnvelopeHeader {
            schema_major: package.schema_major,
            schema_minor: package.schema_minor,
            artifact_kind: package.artifact_kind,
            package_digest,
        },
        work,
    ))
}

/// Inspect transient function-local verification work for one canonical module.
///
/// # Errors
///
/// Returns [`CompilerError`] when the module fails any ordinary native
/// verification stage. The returned counters are not persistent program data.
pub fn inspect_module_local_verification_work(
    payload: &[u8],
) -> Result<QCoreVerificationWork, CompilerError> {
    LocallyCheckedProgram::from_bytes_with_boundary_and_work(payload, ArtifactBoundary::Module)
        .map(|(_, work)| work)
}

/// Inspect transient function-local verification work for one canonical program.
///
/// # Errors
///
/// Returns [`CompilerError`] when the program fails any ordinary native
/// verification stage. The returned counters exclude cleanup-proof replay.
pub fn inspect_program_local_verification_work(
    payload: &[u8],
) -> Result<QCoreVerificationWork, CompilerError> {
    LocallyCheckedProgram::from_bytes_with_boundary_and_work(payload, ArtifactBoundary::Program)
        .map(|(_, work)| work)
}

/// Seal one canonical unsealed `QCore` package through the native wire policy.
///
/// The input must be a canonical `QCorePackageProto` with an empty digest.
/// Borrowed preflight and decoded structural budgets run before the package is
/// signed, so Python and native transforms cannot assign different persistence
/// identities to the same graph.
///
/// # Errors
///
/// Returns [`CompilerError`] when the unsealed package is oversized,
/// malformed, incompatible, noncanonical, or has an unknown artifact kind.
pub fn seal_qcore_package(payload: &[u8]) -> Result<Vec<u8>, CompilerError> {
    let package = decode_qcore_message(payload)?;
    validate_qcore_schema(&package)?;
    validate_known_artifact_kind(&package.artifact_kind)?;
    if !package.package_digest.is_empty()
        || package.encode_to_vec() != payload
        || !has_canonical_qcore_structure(&package)
    {
        return Err(CompilerError::NonCanonicalEncoding);
    }
    seal_package(package)
}

/// Derive the debug-erased semantic identity of one verified module.
///
/// # Errors
///
/// Returns [`CompilerError`] when `payload` is not a canonical verified module.
pub fn module_semantic_digest(payload: &[u8]) -> Result<[u8; 32], CompilerError> {
    let module = VerifiedModule::from_bytes(payload)?;
    Ok(semantic_identity::package_digest(
        &module.local.checked.package,
    ))
}

/// Resolve one source-facing callable selector in a verified module.
///
/// # Errors
///
/// Returns [`CompilerError`] when `payload` is not a canonical module or the
/// selector is absent or ambiguous.
pub fn resolve_module_entrypoint(payload: &[u8], selector: &str) -> Result<String, CompilerError> {
    let module = VerifiedModule::from_bytes(payload)?;
    module.resolve_entrypoint(selector).map(str::to_owned)
}

/// Derive the debug-erased semantic identity of one verified program.
///
/// # Errors
///
/// Returns [`CompilerError`] when `payload` is not a canonical executable program.
pub fn program_semantic_digest(payload: &[u8]) -> Result<[u8; 32], CompilerError> {
    let program = VerifiedProgram::from_bytes(payload)?;
    Ok(program.semantic_digest())
}

/// Derive one selected family identity from a verified module and source selector.
///
/// # Errors
///
/// Returns [`CompilerError`] when `payload` is not a canonical module or
/// `selector` is absent or ambiguous.
pub fn selected_module_semantic_digest(
    payload: &[u8],
    selector: &str,
) -> Result<[u8; 32], CompilerError> {
    let module = VerifiedModule::from_bytes(payload)?;
    module.selected_semantic_digest(selector)
}

struct DecodedArtifact {
    package: wire::QCorePackageProto,
    observed_digest: [u8; 32],
    type_arena: TypeArena,
    operations: operations::LocallyCheckedOperations,
    equations: equations::LocalEquationFacts,
    call_graph: call_graph::StaticCallGraph,
    boolean_plans: phase_lifting::VerifiedBooleanLiftPlans,
    contracts: contracts::ProgramContracts,
    verification_work: QCoreVerificationWork,
}

#[allow(clippy::too_many_lines)]
fn decode_qcore_artifact(
    payload: &[u8],
    boundary: ArtifactBoundary,
) -> Result<DecodedArtifact, CompilerError> {
    let (package, observed_digest) = decode_qcore_envelope(payload, Some(boundary.kind()))?;
    let mut verification_work = QCoreVerificationWork::default();
    literals::validate_program_literals_with_work(&package, &mut verification_work)
        .map_err(CompilerError::InvalidPortableLiteral)?;
    let mut type_arena = types::validate_program_types_with_work(&package, &mut verification_work)
        .map_err(CompilerError::InvalidSemanticType)?;
    let operations = operations::validate_program_operations_with_types_and_work(
        &package,
        &mut type_arena,
        &mut verification_work,
    )
    .map_err(|error| CompilerError::InvalidOperationKernel(error.to_string()))?;
    match boundary {
        ArtifactBoundary::Module => {
            if !package.entrypoint.is_empty() || !package.const_bindings.is_empty() {
                return Err(CompilerError::NonCanonicalEncoding);
            }
        }
        ArtifactBoundary::AnalysisSelection | ArtifactBoundary::Selection => {
            if package.entrypoint.is_empty() {
                return Err(CompilerError::MissingEntrypoint);
            }
        }
        ArtifactBoundary::Program
        | ArtifactBoundary::Analysis
        | ArtifactBoundary::Specialization
        | ArtifactBoundary::FullyBoundSpecialization
        | ArtifactBoundary::TargetAdjoint => {
            if package.entrypoint.is_empty()
                || !package
                    .functions
                    .iter()
                    .any(|function| function.symbol == package.entrypoint)
            {
                return Err(CompilerError::MissingEntrypoint);
            }
        }
    }
    attributes::validate_unique_attributes_with_work(&package, &mut verification_work)
        .map_err(CompilerError::InvalidAttributes)?;
    let call_graph = call_graph::build_with_work(&package, &mut verification_work)
        .map_err(|error| CompilerError::InvalidCallGraph(error.to_string()))?;
    verification::verify_program_with_work(
        &package,
        &type_arena,
        &operations,
        &mut verification_work,
    )
    .map_err(|error| CompilerError::InvalidLocalVerification(error.to_string()))?;
    match boundary {
        ArtifactBoundary::Module
        | ArtifactBoundary::AnalysisSelection
        | ArtifactBoundary::Selection => declarations::verify_module_with_work(
            &package,
            &mut type_arena,
            &operations,
            &mut verification_work,
        ),
        ArtifactBoundary::Program => declarations::verify_with_work(
            &package,
            &mut type_arena,
            &operations,
            &mut verification_work,
        ),
        ArtifactBoundary::Analysis => declarations::verify_symbolic_with_work(
            &package,
            &mut type_arena,
            &operations,
            &mut verification_work,
        ),
        ArtifactBoundary::Specialization
        | ArtifactBoundary::FullyBoundSpecialization
        | ArtifactBoundary::TargetAdjoint => declarations::verify_specialization_request_with_work(
            &package,
            &mut type_arena,
            &operations,
            &mut verification_work,
        ),
    }
    .map_err(|error| CompilerError::InvalidDeclaration(error.to_string()))?;
    let equations = match boundary {
        ArtifactBoundary::Program | ArtifactBoundary::TargetAdjoint => {
            equations::verify_program_with_work(
                &package,
                &type_arena,
                &operations,
                &mut verification_work,
            )
        }
        ArtifactBoundary::Module
        | ArtifactBoundary::AnalysisSelection
        | ArtifactBoundary::Selection
        | ArtifactBoundary::Analysis
        | ArtifactBoundary::Specialization
        | ArtifactBoundary::FullyBoundSpecialization => equations::verify_module_with_work(
            &package,
            &type_arena,
            &operations,
            &mut verification_work,
        ),
    }
    .map_err(|error| CompilerError::InvalidLocalEquation(error.to_string()))?;
    let use_abi_entry_bindings = !matches!(
        boundary,
        ArtifactBoundary::AnalysisSelection
            | ArtifactBoundary::Selection
            | ArtifactBoundary::Specialization
            | ArtifactBoundary::FullyBoundSpecialization
    );
    abi::verify_calls_with_work(
        &package,
        &type_arena,
        &operations,
        use_abi_entry_bindings,
        &mut verification_work,
    )
    .map_err(|error| CompilerError::InvalidCallAbi(error.to_string()))?;
    let boolean_plans = phase_lifting::verify_all_with_work(
        &package,
        &type_arena,
        &operations,
        &mut verification_work,
    )
    .map_err(|error| CompilerError::InvalidBooleanLift(error.to_string()))?;
    match boundary {
        ArtifactBoundary::AnalysisSelection | ArtifactBoundary::Selection => {
            domains::verify_static_domains_with_work(&package, false, &mut verification_work)
                .map_err(|error| CompilerError::InvalidStaticDomain(error.to_string()))?;
        }
        ArtifactBoundary::Specialization => {}
        ArtifactBoundary::FullyBoundSpecialization | ArtifactBoundary::TargetAdjoint => {
            domains::verify_static_domains_with_work(&package, true, &mut verification_work)
                .map_err(|error| CompilerError::InvalidStaticDomain(error.to_string()))?;
        }
        ArtifactBoundary::Module | ArtifactBoundary::Program | ArtifactBoundary::Analysis => {
            domains::verify_static_domains_with_work(&package, true, &mut verification_work)
                .map_err(|error| CompilerError::InvalidStaticDomain(error.to_string()))?;
        }
    }
    let control_dependence = control_dependence::analyze_with_work(
        &package,
        &type_arena,
        &operations,
        &call_graph,
        &mut verification_work,
    )
    .map_err(|error| CompilerError::InvalidControlDependence(error.to_string()))?;
    let contracts = contracts::derive_with_work(
        &package,
        &type_arena,
        &operations,
        &equations,
        &call_graph,
        control_dependence,
        &mut verification_work,
    )
    .map_err(|error| CompilerError::InvalidCallableContract(error.to_string()))?;
    Ok(DecodedArtifact {
        package,
        observed_digest,
        type_arena,
        operations,
        equations,
        call_graph,
        boolean_plans,
        contracts,
        verification_work,
    })
}

fn decode_qcore_message(payload: &[u8]) -> Result<wire::QCorePackageProto, CompilerError> {
    decode_qcore_message_with_work(payload).map(|(package, _)| package)
}

fn decode_qcore_message_with_work(
    payload: &[u8],
) -> Result<(wire::QCorePackageProto, QCorePersistenceWork), CompilerError> {
    if payload.len() > MAX_QCORE_BYTES {
        return Err(CompilerError::PackageTooLarge);
    }
    let preflight_message_nodes =
        budget::preflight_with_work(payload).map_err(|error| match error {
            budget::PreflightError::Budget(reason) => CompilerError::PackageBudgetExceeded(reason),
            budget::PreflightError::Malformed(reason) => CompilerError::MalformedPackage(reason),
        })?;
    let package = wire::QCorePackageProto::decode(payload)
        .map_err(|error| CompilerError::MalformedPackage(error.to_string()))?;
    budget::validate(&package).map_err(CompilerError::PackageBudgetExceeded)?;
    Ok((
        package,
        QCorePersistenceWork {
            input_bytes: payload.len(),
            preflight_message_nodes,
            canonical_encode_passes: 0,
            canonical_encoded_bytes: 0,
        },
    ))
}

fn validate_qcore_schema(package: &wire::QCorePackageProto) -> Result<(), CompilerError> {
    if package.schema_major != SCHEMA_MAJOR {
        return Err(CompilerError::UnsupportedSchemaMajor {
            actual: package.schema_major,
            expected: SCHEMA_MAJOR,
        });
    }
    if package.schema_minor > SCHEMA_MINOR {
        return Err(CompilerError::UnsupportedSchemaMinor {
            actual: package.schema_minor,
            maximum: SCHEMA_MINOR,
        });
    }
    Ok(())
}

fn validate_known_artifact_kind(kind: &str) -> Result<(), CompilerError> {
    if matches!(
        kind,
        MODULE_KIND | PROGRAM_KIND | ANALYSIS_KIND | SPECIALIZATION_KIND | TARGET_ADJOINT_KIND
    ) {
        Ok(())
    } else {
        Err(CompilerError::UnknownArtifactKind {
            actual: kind.to_owned(),
        })
    }
}

fn has_canonical_qcore_structure(package: &wire::QCorePackageProto) -> bool {
    package
        .functions
        .windows(2)
        .all(|pair| pair[0].symbol < pair[1].symbol)
        && package
            .constants
            .windows(2)
            .all(|pair| pair[0].name < pair[1].name)
        && package
            .const_bindings
            .windows(2)
            .all(|pair| pair[0].name < pair[1].name)
        && structure::validate_canonical_structure(package)
}

fn decode_qcore_envelope(
    payload: &[u8],
    expected_kind: Option<&'static str>,
) -> Result<(wire::QCorePackageProto, [u8; 32]), CompilerError> {
    decode_qcore_envelope_with_work(payload, expected_kind)
        .map(|(package, digest, _)| (package, digest))
}

fn decode_qcore_envelope_with_work(
    payload: &[u8],
    expected_kind: Option<&'static str>,
) -> Result<(wire::QCorePackageProto, [u8; 32], QCorePersistenceWork), CompilerError> {
    let (mut package, mut work) = decode_qcore_message_with_work(payload)?;
    validate_qcore_schema(&package)?;
    if let Some(expected) = expected_kind {
        if package.artifact_kind != expected {
            return Err(CompilerError::UnexpectedArtifactKind {
                expected,
                actual: package.artifact_kind,
            });
        }
    } else {
        validate_known_artifact_kind(&package.artifact_kind)?;
    }

    let observed_digest: [u8; 32] = package
        .package_digest
        .as_slice()
        .try_into()
        .map_err(|_| CompilerError::DigestMismatch)?;
    package.package_digest.clear();
    let unsealed = package.encode_to_vec();
    work.charge_encoding(&unsealed);
    let expected_digest: [u8; 32] = Sha256::digest(&unsealed).into();
    if observed_digest != expected_digest {
        return Err(CompilerError::DigestMismatch);
    }

    package.package_digest.extend_from_slice(&observed_digest);
    let canonical = package.encode_to_vec();
    work.charge_encoding(&canonical);
    if canonical != payload || !has_canonical_qcore_structure(&package) {
        return Err(CompilerError::NonCanonicalEncoding);
    }
    Ok((package, observed_digest, work))
}

fn seal_package(mut package: wire::QCorePackageProto) -> Result<Vec<u8>, CompilerError> {
    budget::validate(&package).map_err(CompilerError::PackageBudgetExceeded)?;
    package.package_digest.clear();
    let digest: [u8; 32] = Sha256::digest(package.encode_to_vec()).into();
    package.package_digest.extend_from_slice(&digest);
    let payload = package.encode_to_vec();
    if payload.len() > MAX_QCORE_BYTES {
        return Err(CompilerError::PackageTooLarge);
    }
    Ok(payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span() -> wire::SourceSpanProto {
        wire::SourceSpanProto {
            uri: "qmi://native-test".to_owned(),
            start_line: 1,
            end_line: 1,
            ..Default::default()
        }
    }

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

    fn value(id: u64, r#type: wire::TypeProto) -> wire::ValueProto {
        wire::ValueProto {
            id,
            r#type: Some(r#type),
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

    fn reseal(mut package: wire::QCorePackageProto) -> Vec<u8> {
        package.package_digest.clear();
        package.package_digest = Sha256::digest(package.encode_to_vec()).to_vec();
        package.encode_to_vec()
    }

    fn attribute(name: &str, value: wire::literal_proto::Value) -> wire::AttributeProto {
        wire::AttributeProto {
            name: name.to_owned(),
            value: Some(wire::LiteralProto { value: Some(value) }),
        }
    }

    fn string_tuple(values: &[&str]) -> wire::literal_proto::Value {
        wire::literal_proto::Value::TupleValue(wire::literal_proto::Sequence {
            values: values
                .iter()
                .map(|value| wire::LiteralProto {
                    value: Some(wire::literal_proto::Value::StringValue((*value).to_owned())),
                })
                .collect(),
        })
    }

    fn branch(
        arguments: Vec<wire::ValueProto>,
        kind: &str,
        operands: Vec<u64>,
    ) -> wire::RegionProto {
        wire::RegionProto {
            blocks: vec![wire::BlockProto {
                arguments,
                terminator: Some(wire::TerminatorProto {
                    kind: kind.to_owned(),
                    operands,
                    span: Some(span()),
                    ..Default::default()
                }),
                ..Default::default()
            }],
        }
    }

    fn encoded_program() -> Vec<u8> {
        let mut package = wire::QCorePackageProto {
            schema_major: SCHEMA_MAJOR,
            schema_minor: SCHEMA_MINOR,
            artifact_kind: PROGRAM_KIND.to_owned(),
            module_name: "native_test".to_owned(),
            functions: vec![wire::FunctionProto {
                symbol: "native_test.main".to_owned(),
                parameters: vec![wire::ParameterProto {
                    name: "unit".to_owned(),
                    value: Some(wire::ValueProto {
                        id: 1,
                        r#type: Some(scalar("Unit")),
                        ..Default::default()
                    }),
                    ..Default::default()
                }],
                result_type: Some(scalar("Unit")),
                body: Some(wire::BlockProto {
                    terminator: Some(wire::TerminatorProto {
                        kind: "return".to_owned(),
                        operands: vec![1],
                        span: Some(span()),
                        ..Default::default()
                    }),
                    ..Default::default()
                }),
                span: Some(span()),
                ..Default::default()
            }],
            entrypoint: "native_test.main".to_owned(),
            ..Default::default()
        };
        package.package_digest = Sha256::digest(package.encode_to_vec()).to_vec();
        package.encode_to_vec()
    }

    fn encoded_module() -> Vec<u8> {
        let mut package =
            wire::QCorePackageProto::decode(encoded_program().as_slice()).expect("program");
        package.artifact_kind = MODULE_KIND.to_owned();
        package.entrypoint.clear();
        let const_int = staged("Const", scalar("Int"));
        package.functions[0].parameters[0]
            .value
            .as_mut()
            .expect("parameter value")
            .r#type = Some(const_int.clone());
        package.functions[0].result_type = Some(const_int);
        reseal(package)
    }

    fn resource_study_for(program: &[u8]) -> Vec<u8> {
        let verified = VerifiedProgram::from_bytes(program).expect("resource program verifies");
        seal_resource_study(
            &wire::ResourceStudyProto {
                schema_major: 1,
                schema_minor: 0,
                bundle: Some(wire::ResourceAnalysisBundleProto {
                    entrypoint: verified.header().entrypoint.clone(),
                    analysis_version: "native-test-v1".to_owned(),
                    qcore_payload: program.to_vec(),
                    ..Default::default()
                }),
                ..Default::default()
            }
            .encode_to_vec(),
        )
        .expect("resource study seals")
    }

    fn launch_artifact(program: &[u8], resource_program: &[u8]) -> Vec<u8> {
        let verified = VerifiedProgram::from_bytes(program).expect("launch program verifies");
        let abi = verified.artifact_abi().expect("artifact ABI derives");
        let parameter_abi = if abi.parameter_abi().is_empty() {
            None
        } else {
            Some(
                wire::ParameterAbiProto::decode(abi.parameter_abi())
                    .expect("parameter ABI decodes"),
            )
        };
        let result_abi =
            wire::ResultAbiProto::decode(abi.result_abi()).expect("result ABI decodes");
        let materializer = seal_launch_materializer_payload("probe.v1", program)
            .expect("materializer payload seals");
        seal_artifact(
            &wire::ArtifactProto {
                schema_major: SCHEMA_MAJOR,
                schema_minor: SCHEMA_MINOR,
                format: "qamomile.launch-materializer.v1".to_owned(),
                payload: materializer,
                target_fingerprint: "probe-v1".to_owned(),
                parameter_abi,
                result_abi: Some(result_abi),
                report: Some(wire::CompilationReportProto {
                    resource_study: resource_study_for(resource_program),
                    ..Default::default()
                }),
                execution_requirements: vec!["host_dispatch".to_owned()],
                ..Default::default()
            }
            .encode_to_vec(),
        )
        .expect("launch artifact seals")
    }

    fn reseal_artifact(mut envelope: wire::ArtifactProto) -> Vec<u8> {
        envelope.artifact_digest.clear();
        seal_artifact(&envelope.encode_to_vec()).expect("modified artifact reseals")
    }

    fn cleanup_program(terminal: &str, origin: u64) -> Vec<u8> {
        let mut state = value(2, quantum_bit());
        state.has_region = true;
        state.region = Some(wire::RegionRefProto {
            root: 2,
            ..Default::default()
        });
        let terminal = if terminal == "qcore.forget" {
            wire::OperationProto {
                opcode: terminal.to_owned(),
                operands: vec![2],
                span: Some(span()),
                ..Default::default()
            }
        } else {
            wire::OperationProto {
                opcode: terminal.to_owned(),
                operands: vec![2],
                attributes: vec![
                    attribute(
                        "origin",
                        wire::literal_proto::Value::IntValue(
                            i64::try_from(origin).expect("test origin fits i64"),
                        ),
                    ),
                    attribute(
                        "proof",
                        wire::literal_proto::Value::StringValue("modular-adjoint".to_owned()),
                    ),
                ],
                span: Some(span()),
                ..Default::default()
            }
        };
        let package = wire::QCorePackageProto {
            schema_major: SCHEMA_MAJOR,
            schema_minor: SCHEMA_MINOR,
            artifact_kind: PROGRAM_KIND.to_owned(),
            module_name: "native_cleanup".to_owned(),
            functions: vec![wire::FunctionProto {
                symbol: "native_cleanup.main".to_owned(),
                result_type: Some(scalar("Unit")),
                body: Some(wire::BlockProto {
                    operations: vec![
                        wire::OperationProto {
                            opcode: "qcore.constant".to_owned(),
                            results: vec![value(1, scalar("Int"))],
                            attributes: vec![attribute(
                                "value",
                                wire::literal_proto::Value::IntValue(1),
                            )],
                            span: Some(span()),
                            ..Default::default()
                        },
                        wire::OperationProto {
                            opcode: "quantum.alloc".to_owned(),
                            operands: vec![1],
                            results: vec![state],
                            span: Some(span()),
                            ..Default::default()
                        },
                        terminal,
                        wire::OperationProto {
                            opcode: "qcore.constant".to_owned(),
                            results: vec![value(3, scalar("Unit"))],
                            attributes: vec![attribute(
                                "value",
                                wire::literal_proto::Value::UnitValue(true),
                            )],
                            span: Some(span()),
                            ..Default::default()
                        },
                    ],
                    terminator: Some(wire::TerminatorProto {
                        kind: "return".to_owned(),
                        operands: vec![3],
                        span: Some(span()),
                        ..Default::default()
                    }),
                    ..Default::default()
                }),
                span: Some(span()),
                ..Default::default()
            }],
            entrypoint: "native_cleanup.main".to_owned(),
            ..Default::default()
        };
        reseal(package)
    }

    #[test]
    fn accepts_a_canonical_closed_program() {
        let payload = encoded_program();
        let program = LocallyCheckedProgram::from_bytes(&payload).expect("valid program");
        let header = program.header();

        assert_eq!(header.module_name, "native_test");
        assert_eq!(header.entrypoint, "native_test.main");
        assert_eq!(header.function_count, 1);
        assert_eq!(header.type_count, 1);
        assert_eq!(header.operation_count, 0);
        assert_eq!(program.canonical_bytes(), payload);
        assert_eq!(program.types().len(), header.type_count);
        let contract = program
            .callable_contract("native_test.main")
            .expect("retained entry contract");
        assert_eq!(contract.kind(), CallableKind::Classical);
        assert_eq!(contract.effects().iter().count(), 0);
        assert_eq!(
            program.callable_control_parameters("native_test.main"),
            Some([].as_slice())
        );
    }

    #[test]
    fn seals_and_inspects_one_canonical_qcore_envelope() {
        let expected = encoded_program();
        let mut package = wire::QCorePackageProto::decode(expected.as_slice()).expect("program");
        let expected_digest = package.package_digest.clone();
        package.package_digest.clear();
        let unsealed = package.encode_to_vec();

        let sealed = seal_qcore_package(&unsealed).expect("native seal");
        assert_eq!(sealed, expected);
        let (header, work) = inspect_qcore_envelope_with_work(&sealed)
            .expect("native envelope inspection with diagnostic work");
        assert_eq!(header.schema_major, SCHEMA_MAJOR);
        assert_eq!(header.schema_minor, SCHEMA_MINOR);
        assert_eq!(header.artifact_kind, PROGRAM_KIND);
        assert_eq!(header.package_digest.as_slice(), expected_digest);
        assert_eq!(work.input_bytes(), sealed.len());
        assert!(work.preflight_message_nodes() > 0);
        assert_eq!(work.canonical_encode_passes(), 2);
        assert_eq!(
            work.canonical_encoded_bytes(),
            unsealed.len() + sealed.len()
        );

        assert_eq!(
            seal_qcore_package(&sealed),
            Err(CompilerError::NonCanonicalEncoding)
        );
    }

    #[test]
    fn seals_and_inspects_the_existing_launch_materializer_format() {
        let program = encoded_program();
        let payload = seal_launch_materializer_payload("probe.v1", &program)
            .expect("launch materializer seals");
        let materializer =
            inspect_launch_materializer_payload(&payload).expect("launch materializer inspects");

        assert_eq!(materializer.target_format(), "probe.v1");
        assert_eq!(materializer.program(), program);
    }

    #[test]
    fn launch_materializer_rejects_invalid_programs_and_noncanonical_wire() {
        assert!(seal_launch_materializer_payload("probe.v1", b"not-a-program").is_err());

        let mut corrupted =
            seal_launch_materializer_payload("probe.v1", &encoded_program()).expect("valid recipe");
        corrupted.extend_from_slice(&[0x98, 0x06, 0x00]);
        assert!(inspect_launch_materializer_payload(&corrupted).is_err());
    }

    #[test]
    fn launch_materializer_artifact_checks_every_outer_program_identity() {
        let program = encoded_program();
        let artifact = launch_artifact(&program, &program);
        let materialized = vec!["host_dispatch".to_owned()];
        let checked = inspect_launch_materializer_artifact(&artifact, "probe.v1", &materialized)
            .expect("complete launch artifact verifies");
        assert_eq!(checked.program(), program);

        let decode = || wire::ArtifactProto::decode(artifact.as_slice()).expect("artifact decodes");
        let mut wrong_parameter = decode();
        wrong_parameter
            .parameter_abi
            .as_mut()
            .expect("parameter ABI")
            .slots[0]
            .name = "other".to_owned();
        assert!(
            inspect_launch_materializer_artifact(
                &reseal_artifact(wrong_parameter),
                "probe.v1",
                &materialized,
            )
            .is_err()
        );

        let mut wrong_result = decode();
        wrong_result.result_abi.as_mut().expect("result ABI").r#type = Some(scalar("Int"));
        assert!(
            inspect_launch_materializer_artifact(
                &reseal_artifact(wrong_result),
                "probe.v1",
                &materialized,
            )
            .is_err()
        );

        let mut different =
            wire::QCorePackageProto::decode(program.as_slice()).expect("program decodes");
        different.module_name = "different_semantics".to_owned();
        let wrong_resource = launch_artifact(&program, &reseal(different));
        assert!(
            inspect_launch_materializer_artifact(&wrong_resource, "probe.v1", &materialized,)
                .is_err()
        );

        let mut wrong_contract = decode();
        wrong_contract.execution_requirements.clear();
        assert!(
            inspect_launch_materializer_artifact(
                &reseal_artifact(wrong_contract),
                "probe.v1",
                &materialized,
            )
            .is_err()
        );
        assert!(
            inspect_launch_materializer_artifact(&artifact, "other.v1", &materialized).is_err()
        );
        assert!(inspect_launch_materializer_artifact(&artifact, "probe.v1", &[]).is_err());
    }

    #[test]
    fn seals_and_inspects_one_symbolic_resource_request() {
        let mut family =
            wire::QCorePackageProto::decode(encoded_program().as_slice()).expect("program");
        family.artifact_kind = ANALYSIS_KIND.to_owned();
        let mut request = wire::ResourceAnalysisRequestProto {
            schema_major: 1,
            schema_minor: 2,
            program_family: reseal(family),
            ..Default::default()
        };
        let unsealed = request.encode_to_vec();

        let sealed = seal_resource_analysis_request(&unsealed).expect("native resource seal");
        let verified = VerifiedResourceAnalysisRequest::from_bytes(&sealed)
            .expect("native resource inspection");
        assert_eq!(verified.rule_count(), 0);
        request.request_digest = Sha256::digest(unsealed).to_vec();
        assert_eq!(sealed, request.encode_to_vec());

        assert!(matches!(
            seal_resource_analysis_request(&sealed),
            Err(ResourceRuleError::Invalid(message)) if message.starts_with("Q4225:")
        ));
    }

    #[test]
    fn semantic_identity_erases_debug_metadata_but_retains_program_meaning() {
        let original = encoded_program();
        let original_digest = program_semantic_digest(&original).expect("semantic digest");
        let mut debug_changed =
            wire::QCorePackageProto::decode(original.as_slice()).expect("program");
        let function = &mut debug_changed.functions[0];
        function.span.as_mut().expect("function span").uri = "qmsrc://changed".to_owned();
        function.parameters[0]
            .value
            .as_mut()
            .expect("parameter value")
            .hint = "changed_hint".to_owned();
        let terminator = function
            .body
            .as_mut()
            .and_then(|body| body.terminator.as_mut())
            .expect("terminator");
        terminator
            .span
            .as_mut()
            .expect("terminator span")
            .start_line = 99;

        let changed_debug_digest =
            program_semantic_digest(&reseal(debug_changed.clone())).expect("debug-only digest");
        assert_eq!(changed_debug_digest, original_digest);

        debug_changed.module_name = "different_semantics".to_owned();
        let meaning_changed =
            program_semantic_digest(&reseal(debug_changed)).expect("meaning digest");
        assert_ne!(meaning_changed, original_digest);
    }

    #[test]
    fn selected_module_identity_uses_the_resolved_callable_symbol() {
        let module = encoded_module();
        let module_digest = module_semantic_digest(&module).expect("module digest");
        let selected =
            selected_module_semantic_digest(&module, "native_test.main").expect("selected digest");

        assert_eq!(
            resolve_module_entrypoint(&module, "main").expect("local selector resolves"),
            "native_test.main"
        );
        assert_eq!(
            resolve_module_entrypoint(&module, "native_test.main")
                .expect("qualified selector resolves"),
            "native_test.main"
        );
        assert_eq!(
            selected,
            semantic_identity::selected_digest(module_digest, "native_test.main")
        );
        assert_eq!(
            selected_module_semantic_digest(&module, "main").expect("local digest"),
            selected
        );
        assert!(
            resolve_module_entrypoint(&module, "missing").is_err_and(|error| error.to_string()
                == "Q3001: expected one function named 'missing', found 0")
        );
        let oversized = "x".repeat(MAX_QCORE_STRING_BYTES + 1);
        assert!(
            resolve_module_entrypoint(&module, &oversized)
                .is_err_and(|error| error.to_string().contains("function selector exceeds"))
        );

        let mut package =
            wire::QCorePackageProto::decode(module.as_slice()).expect("canonical module");
        let mut duplicate = package.functions[0].clone();
        duplicate.symbol = "another.main".to_owned();
        package.functions.push(duplicate);
        package
            .functions
            .sort_by(|left, right| left.symbol.cmp(&right.symbol));
        let ambiguous = reseal(package);
        assert_eq!(
            resolve_module_entrypoint(&ambiguous, "main")
                .expect_err("local selector is ambiguous")
                .to_string(),
            "Q3001: expected one function named 'main', found 2"
        );
    }

    #[test]
    fn qcore_sealing_rejects_unknown_artifact_kinds() {
        let mut package =
            wire::QCorePackageProto::decode(encoded_program().as_slice()).expect("program");
        package.package_digest.clear();
        package.artifact_kind = "qamomile.unknown".to_owned();

        assert_eq!(
            seal_qcore_package(&package.encode_to_vec()),
            Err(CompilerError::UnknownArtifactKind {
                actual: "qamomile.unknown".to_owned(),
            })
        );
    }

    #[test]
    fn specialization_request_is_non_executable_and_checks_static_domains() {
        let mut package =
            wire::QCorePackageProto::decode(cleanup_program("qcore.forget", 0).as_slice())
                .expect("cleanup program");
        package.artifact_kind = SPECIALIZATION_KIND.to_owned();
        let payload = reseal(package.clone());

        let request = VerifiedSpecializationRequest::from_bytes(&payload)
            .expect("valid pending specialization request");
        assert_eq!(request.pending_cleanup_obligation_count(), 1);
        assert_eq!(request.callable_contracts().count(), 1);
        assert!(VerifiedProgram::from_bytes(&payload).is_err());

        let width = &mut package.functions[0].body.as_mut().expect("body").operations[0].attributes
            [0]
        .value;
        *width = Some(wire::LiteralProto {
            value: Some(wire::literal_proto::Value::IntValue(-1)),
        });
        let error = VerifiedSpecializationRequest::from_bytes(&reseal(package))
            .expect_err("non-positive allocation");
        assert!(error.to_string().contains("Q1046"));
    }

    #[test]
    fn target_adjoint_and_specialization_typestates_are_disjoint() {
        let mut package =
            wire::QCorePackageProto::decode(cleanup_program("qcore.forget", 0).as_slice())
                .expect("cleanup program");
        package.artifact_kind = TARGET_ADJOINT_KIND.to_owned();
        let target_payload = reseal(package.clone());

        TargetLoweredProgram::from_bytes(&target_payload).expect("target-adjoint artifact");
        let specialization_error = VerifiedSpecializationRequest::from_bytes(&target_payload)
            .expect_err("target artifact cannot be a specialization request");
        assert!(
            specialization_error
                .to_string()
                .contains(SPECIALIZATION_KIND)
        );

        package.artifact_kind = SPECIALIZATION_KIND.to_owned();
        let specialization_payload = reseal(package);
        VerifiedSpecializationRequest::from_bytes(&specialization_payload)
            .expect("specialization request");
        let target_error = TargetLoweredProgram::from_bytes(&specialization_payload)
            .expect_err("specialization request cannot be a target artifact");
        assert!(target_error.to_string().contains(TARGET_ADJOINT_KIND));
    }

    #[test]
    fn target_planning_closes_labels_and_reports_compatibility() {
        let program = VerifiedProgram::from_bytes(&encoded_program()).expect("valid program");
        let plan = program.plan_target(&[]).expect("valid empty profile");

        assert!(plan.required().is_empty());
        assert!(plan.supported().is_empty());
        assert!(plan.missing().is_empty());
        assert!(plan.is_compatible());
        let lowering = program
            .plan_target_lowering(&[], &[], false)
            .expect("valid direct lowering profile");
        assert_eq!(lowering.route(), TargetLoweringRoute::Direct);
        assert!(lowering.materialized().is_empty());
        assert!(lowering.unsupported().is_empty());
        assert_eq!(
            program
                .plan_target(&["not-a-requirement".to_owned()])
                .expect_err("unknown capability"),
            TargetPlanningError::UnknownCapability("not-a-requirement".to_owned())
        );
        assert_eq!(
            program
                .plan_target(&["quantum".to_owned(), "quantum".to_owned()])
                .expect_err("duplicate capability"),
            TargetPlanningError::DuplicateCapability("quantum".to_owned())
        );
        assert_eq!(
            program
                .plan_target_lowering(&[], &["rotation".to_owned()], false)
                .expect_err("invalid materialization promise"),
            TargetPlanningError::InvalidMaterializedCapability("rotation".to_owned())
        );
    }

    #[test]
    fn target_prepared_typestate_composes_verified_optimizer_results() {
        let program = VerifiedProgram::from_bytes(&encoded_program()).expect("valid program");
        let replacement = TargetPreparedProgram::from_verified(program.clone());

        assert_eq!(replacement.canonical_bytes(), program.canonical_bytes());
        assert_eq!(
            replacement.as_verified().semantic_digest(),
            program.semantic_digest()
        );

        let inserted = replacement.into_verified();
        let (standard, _) = inserted.prepare_standard().expect("standard insertion");
        assert_eq!(
            standard.as_verified().semantic_digest(),
            program.semantic_digest()
        );
    }

    #[test]
    fn optimization_receipt_roundtrips_source_policy_and_checked_preparation() {
        let source = VerifiedProgram::from_bytes(&encoded_program()).expect("verified source");
        let policy = OptimizationPolicy::new(7, 1000, "test.target.v1".to_owned()).expect("policy");
        let pass = OptimizationPassIdentity::new(
            "extension.example".to_owned(),
            "1".to_owned(),
            b"settings".to_vec(),
        )
        .expect("identity");
        let preparation =
            OptimizationPreparation::new(source.clone(), source.clone(), vec![pass], &policy)
                .expect("association");
        let restored = OptimizationPreparation::from_receipt(source.clone(), preparation.receipt())
            .expect("receipt");
        assert_eq!(restored.receipt(), preparation.receipt());
        assert_eq!(
            restored.source().canonical_bytes(),
            source.canonical_bytes()
        );
        assert_eq!(
            restored.prepared().canonical_bytes(),
            source.canonical_bytes()
        );
        assert_eq!(
            restored.stages().collect::<Vec<_>>(),
            ["optimization:extension.example@1"]
        );
        let other = OptimizationPreparation::new(
            source.clone(),
            source.clone(),
            Vec::new(),
            &OptimizationPolicy::new(8, 1000, "test.target.v1".to_owned()).expect("policy"),
        )
        .expect("empty pipeline");
        let empty =
            OptimizationPreparation::new(source.clone(), source.clone(), Vec::new(), &policy)
                .expect("empty pipeline");
        assert_ne!(empty.identity(), other.identity());
        for length in [0, 8, 39, 55, preparation.receipt().len() - 1] {
            assert!(
                OptimizationPreparation::from_receipt(
                    source.clone(),
                    &preparation.receipt()[..length]
                )
                .is_err()
            );
        }
        let mut corrupt = preparation.receipt().to_vec();
        corrupt[40] ^= 1;
        assert!(OptimizationPreparation::from_receipt(source, &corrupt).is_err());
    }

    #[test]
    fn optimization_configuration_and_rewrite_budgets_fail_closed() {
        let pass = OptimizationPassIdentity::new(
            "extension.example".to_owned(),
            "1".to_owned(),
            Vec::new(),
        )
        .expect("identity");
        assert!(validate_optimization_pipeline(&[pass.clone(), pass.clone()]).is_err());
        assert!(validate_optimization_pipeline(&vec![pass; 65]).is_err());
        assert!(
            OptimizationPassIdentity::new("not_namespaced".to_owned(), "1".to_owned(), Vec::new())
                .is_err()
        );
        assert!(
            OptimizationPassIdentity::new(
                "extension.example".to_owned(),
                "1".to_owned(),
                vec![0; 65537]
            )
            .is_err()
        );
        assert!(OptimizationPolicy::new(0, 0, String::new()).is_err());
        assert!(
            GateRewriteRule::new(vec!["quantum.cx".to_owned()], vec!["quantum.x".to_owned()])
                .is_err()
        );
        assert!(
            GateRewriteRule::new(
                vec!["quantum.h".to_owned()],
                vec!["quantum.x".to_owned(); 2]
            )
            .is_err()
        );
        let source = VerifiedProgram::from_bytes(&encoded_program()).expect("verified source");
        assert!(
            source
                .optimize_standard_with_budget(0, StandardOptimizationOptions::default())
                .is_err()
        );
    }

    #[test]
    fn accepts_an_unselected_module_with_unbound_const_parameters() {
        let payload = encoded_module();
        let module = VerifiedModule::from_bytes(&payload).expect("valid module");

        assert_eq!(module.header().module_name, "native_test");
        assert_eq!(module.header().function_count, 1);
        assert_eq!(module.canonical_bytes(), payload);
        assert_eq!(module.types().len(), module.header().type_count);
        assert_eq!(module.pending_cleanup_obligation_count(), 0);
        assert_eq!(
            module
                .callable_contract("native_test.main")
                .expect("function contract")
                .kind(),
            CallableKind::Classical
        );
    }

    #[test]
    fn keeps_module_and_program_artifact_boundaries_disjoint() {
        assert!(matches!(
            VerifiedProgram::from_bytes(&encoded_module()),
            Err(CompilerError::UnexpectedArtifactKind {
                expected: PROGRAM_KIND,
                ..
            })
        ));
        assert!(matches!(
            VerifiedModule::from_bytes(&encoded_program()),
            Err(CompilerError::UnexpectedArtifactKind {
                expected: MODULE_KIND,
                ..
            })
        ));

        let mut selected =
            wire::QCorePackageProto::decode(encoded_module().as_slice()).expect("module");
        selected.entrypoint = "native_test.main".to_owned();
        assert_eq!(
            VerifiedModule::from_bytes(&reseal(selected)).expect_err("selected module"),
            CompilerError::NonCanonicalEncoding
        );
    }

    #[test]
    fn module_boundary_retains_cleanup_obligations_for_specialization() {
        let mut package =
            wire::QCorePackageProto::decode(cleanup_program("qcore.forget", 0).as_slice())
                .expect("cleanup program");
        package.artifact_kind = MODULE_KIND.to_owned();
        package.entrypoint.clear();

        let module = VerifiedModule::from_bytes(&reseal(package)).expect("open module");

        assert_eq!(module.pending_cleanup_obligation_count(), 1);
    }

    #[test]
    fn native_preparation_synthesizes_and_replays_an_allocation_release() {
        let mut package =
            wire::QCorePackageProto::decode(cleanup_program("qcore.forget", 0).as_slice())
                .expect("cleanup request");
        package.artifact_kind = SPECIALIZATION_KIND.to_owned();

        let program = LocallySpecializedProgram::prepare_request_bytes(&reseal(package))
            .expect("native cleanup synthesis");
        let census = program.structural_census();

        assert_eq!(program.discharged_cleanup_count(), 1);
        assert_eq!(census.opcodes().get("quantum.release"), Some(&1));
        assert!(!census.opcodes().contains_key("qcore.forget"));
    }

    #[test]
    fn promotes_only_independently_replayed_cleanup_roots() {
        let payload = cleanup_program("quantum.release", 2);
        let local = LocallyCheckedProgram::from_bytes(&payload).expect("locally valid cleanup");
        assert_eq!(local.unverified_proof_obligation_count(), 1);

        let verified = local.verify().expect("exact allocation release");
        assert_eq!(verified.discharged_cleanup_count(), 1);
        assert_eq!(verified.canonical_bytes(), payload);
    }

    #[test]
    fn rejects_forged_cleanup_origins_and_unresolved_forget_markers() {
        let forged = VerifiedProgram::from_bytes(&cleanup_program("quantum.release", 99))
            .expect_err("forged origin must fail");
        assert_eq!(
            forged,
            CompilerError::InvalidCleanupProof(
                "Q1112: release in \"native_cleanup.main\" is not the inverse of its producer chain"
                    .to_owned(),
            )
        );
        let unresolved = VerifiedProgram::from_bytes(&cleanup_program("qcore.forget", 0))
            .expect_err("forget marker must fail");
        assert_eq!(
            unresolved,
            CompilerError::InvalidCleanupProof(
                "Q1103: unresolved Forget remains in \"native_cleanup.main\"".to_owned(),
            )
        );
    }

    #[test]
    fn rejects_digest_tampering() {
        let mut payload = encoded_program();
        let last = payload.last_mut().expect("nonempty package");
        *last ^= 1;

        assert!(matches!(
            inspect_program(&payload),
            Err(CompilerError::DigestMismatch | CompilerError::MalformedPackage(_))
        ));
    }

    #[test]
    fn rejects_a_missing_entrypoint() {
        let mut package = wire::QCorePackageProto::decode(encoded_program().as_slice())
            .expect("test package decodes");
        package.entrypoint = "native_test.missing".to_owned();
        package.package_digest.clear();
        package.package_digest = Sha256::digest(package.encode_to_vec()).to_vec();

        assert_eq!(
            inspect_program(&package.encode_to_vec()),
            Err(CompilerError::MissingEntrypoint)
        );
    }

    #[test]
    fn rejects_resealed_noncanonical_function_order() {
        let mut package = wire::QCorePackageProto::decode(encoded_program().as_slice())
            .expect("test package decodes");
        package.functions.insert(
            0,
            wire::FunctionProto {
                symbol: "native_test.z_helper".to_owned(),
                result_type: Some(wire::TypeProto {
                    kind: wire::type_proto::Kind::Scalar.into(),
                    name: "Unit".to_owned(),
                    ..Default::default()
                }),
                ..Default::default()
            },
        );
        package.package_digest.clear();
        package.package_digest = Sha256::digest(package.encode_to_vec()).to_vec();

        assert_eq!(
            inspect_program(&package.encode_to_vec()),
            Err(CompilerError::NonCanonicalEncoding)
        );
    }

    #[test]
    fn rejects_resealed_unknown_semantic_types() {
        let mut package = wire::QCorePackageProto::decode(encoded_program().as_slice())
            .expect("test package decodes");
        package.functions[0]
            .result_type
            .as_mut()
            .expect("test function has a result type")
            .name = "Atom".to_owned();
        package.package_digest.clear();
        package.package_digest = Sha256::digest(package.encode_to_vec()).to_vec();

        assert!(matches!(
            inspect_program(&package.encode_to_vec()),
            Err(CompilerError::InvalidSemanticType(_))
        ));
    }

    #[test]
    fn rejects_resealed_unknown_opcodes() {
        let mut package = wire::QCorePackageProto::decode(encoded_program().as_slice())
            .expect("test package decodes");
        let span = package.functions[0].span.clone();
        package.functions[0].body = Some(wire::BlockProto {
            operations: vec![wire::OperationProto {
                opcode: "quantum.clone".to_owned(),
                span: span.clone(),
                ..Default::default()
            }],
            terminator: Some(wire::TerminatorProto {
                kind: "return".to_owned(),
                span,
                ..Default::default()
            }),
            ..Default::default()
        });
        package.package_digest.clear();
        package.package_digest = Sha256::digest(package.encode_to_vec()).to_vec();

        assert_eq!(
            inspect_program(&package.encode_to_vec()),
            Err(CompilerError::InvalidOperationKernel(
                "Q1032: unknown QCore opcode \"quantum.clone\"".to_owned()
            ))
        );
    }

    #[test]
    fn rejects_a_forward_reference_at_the_native_boundary() {
        let mut package = wire::QCorePackageProto::decode(encoded_program().as_slice())
            .expect("test package decodes");
        let body = package.functions[0].body.as_mut().expect("test body");
        body.operations = vec![
            wire::OperationProto {
                opcode: "qcore.forget".to_owned(),
                operands: vec![2],
                span: Some(span()),
                ..Default::default()
            },
            wire::OperationProto {
                opcode: "qcore.constant".to_owned(),
                results: vec![value(2, scalar("Unit"))],
                span: Some(span()),
                ..Default::default()
            },
        ];

        assert_eq!(
            inspect_program(&reseal(package)),
            Err(CompilerError::InvalidLocalVerification(
                "Q1013: SSA value %2 does not dominate this use".to_owned()
            ))
        );
    }

    #[test]
    fn rejects_an_operation_result_used_inside_its_own_region() {
        let mut package = wire::QCorePackageProto::decode(encoded_program().as_slice())
            .expect("test package decodes");
        let function = &mut package.functions[0];
        function.parameters.push(wire::ParameterProto {
            name: "flag".to_owned(),
            value: Some(value(2, scalar("Bool"))),
            ..Default::default()
        });
        let body = function.body.as_mut().expect("test body");
        body.operations.push(wire::OperationProto {
            opcode: "qcore.if".to_owned(),
            operands: vec![2],
            results: vec![value(3, scalar("Unit"))],
            attributes: vec![
                attribute("capture_names", string_tuple(&[])),
                attribute("condition_index", wire::literal_proto::Value::IntValue(0)),
                attribute("output_names", string_tuple(&["result"])),
                attribute(
                    "stage",
                    wire::literal_proto::Value::StringValue("Value".to_owned()),
                ),
            ],
            regions: vec![
                branch(vec![], "yield", vec![3]),
                branch(vec![], "yield", vec![3]),
            ],
            span: Some(span()),
            intent: Vec::new(),
        });
        body.terminator.as_mut().expect("test terminator").operands = vec![3];

        assert_eq!(
            inspect_program(&reseal(package)),
            Err(CompilerError::InvalidLocalVerification(
                "Q1013: SSA value %3 does not dominate this use".to_owned()
            ))
        );
    }

    #[test]
    fn rejects_a_definition_escaping_between_sibling_regions() {
        let mut package = wire::QCorePackageProto::decode(encoded_program().as_slice())
            .expect("test package decodes");
        let function = &mut package.functions[0];
        function.parameters.push(wire::ParameterProto {
            name: "flag".to_owned(),
            value: Some(value(2, scalar("Bool"))),
            ..Default::default()
        });
        let producer = wire::RegionProto {
            blocks: vec![wire::BlockProto {
                operations: vec![wire::OperationProto {
                    opcode: "qcore.constant".to_owned(),
                    results: vec![value(3, scalar("Unit"))],
                    span: Some(span()),
                    ..Default::default()
                }],
                terminator: Some(wire::TerminatorProto {
                    kind: "yield".to_owned(),
                    operands: vec![3],
                    span: Some(span()),
                    ..Default::default()
                }),
                ..Default::default()
            }],
        };
        let body = function.body.as_mut().expect("test body");
        body.operations.push(wire::OperationProto {
            opcode: "qcore.if".to_owned(),
            operands: vec![2],
            results: vec![value(4, scalar("Unit"))],
            attributes: vec![
                attribute("capture_names", string_tuple(&[])),
                attribute("condition_index", wire::literal_proto::Value::IntValue(0)),
                attribute("output_names", string_tuple(&["result"])),
                attribute(
                    "stage",
                    wire::literal_proto::Value::StringValue("Value".to_owned()),
                ),
            ],
            regions: vec![producer, branch(vec![], "yield", vec![3])],
            span: Some(span()),
            intent: Vec::new(),
        });
        body.terminator.as_mut().expect("test terminator").operands = vec![4];

        assert_eq!(
            inspect_program(&reseal(package)),
            Err(CompilerError::InvalidLocalVerification(
                "Q1013: SSA value %3 does not dominate this use".to_owned()
            ))
        );
    }

    #[test]
    fn rejects_an_unconsumed_linear_parameter() {
        let mut package = wire::QCorePackageProto::decode(encoded_program().as_slice())
            .expect("test package decodes");
        let function = &mut package.functions[0];
        let quantum = function.parameters[0]
            .value
            .as_mut()
            .expect("test parameter");
        quantum.r#type = Some(quantum_bit());
        quantum.has_region = true;
        quantum.region = Some(wire::RegionRefProto {
            root: 7,
            ..Default::default()
        });
        function.parameters.push(wire::ParameterProto {
            name: "unit".to_owned(),
            value: Some(value(2, scalar("Unit"))),
            ..Default::default()
        });
        function
            .body
            .as_mut()
            .expect("test body")
            .terminator
            .as_mut()
            .expect("test terminator")
            .operands = vec![2];

        assert_eq!(
            inspect_program(&reseal(package)),
            Err(CompilerError::InvalidLocalVerification(
                "Q2001: linear value %1 must be consumed exactly once; observed 0 uses".to_owned()
            ))
        );
    }

    #[test]
    fn rejects_a_linear_value_consumed_twice() {
        let mut package = wire::QCorePackageProto::decode(encoded_program().as_slice())
            .expect("test package decodes");
        let function = &mut package.functions[0];
        let quantum = function.parameters[0]
            .value
            .as_mut()
            .expect("test parameter");
        quantum.r#type = Some(quantum_bit());
        quantum.has_region = true;
        quantum.region = Some(wire::RegionRefProto {
            root: 7,
            ..Default::default()
        });
        function.parameters.push(wire::ParameterProto {
            name: "unit".to_owned(),
            value: Some(value(2, scalar("Unit"))),
            ..Default::default()
        });
        let body = function.body.as_mut().expect("test body");
        body.operations = (0..2)
            .map(|_| wire::OperationProto {
                opcode: "qcore.forget".to_owned(),
                operands: vec![1],
                span: Some(span()),
                ..Default::default()
            })
            .collect();
        body.terminator.as_mut().expect("test terminator").operands = vec![2];

        assert_eq!(
            inspect_program(&reseal(package)),
            Err(CompilerError::InvalidLocalVerification(
                "Q2001: linear value %1 must be consumed exactly once; observed 2 uses".to_owned()
            ))
        );
    }

    #[test]
    fn rejects_overlapping_live_owner_regions() {
        let mut package = wire::QCorePackageProto::decode(encoded_program().as_slice())
            .expect("test package decodes");
        let function = &mut package.functions[0];
        function.parameters.clear();
        for id in [1, 2] {
            let mut owned = value(id, quantum_bit());
            owned.has_region = true;
            owned.region = Some(wire::RegionRefProto {
                root: 9,
                projections: if id == 1 {
                    vec![]
                } else {
                    vec!["0".to_owned()]
                },
            });
            function.parameters.push(wire::ParameterProto {
                name: format!("q{id}"),
                value: Some(owned),
                ..Default::default()
            });
        }
        function.parameters.push(wire::ParameterProto {
            name: "unit".to_owned(),
            value: Some(value(3, scalar("Unit"))),
            ..Default::default()
        });
        function
            .body
            .as_mut()
            .expect("test body")
            .terminator
            .as_mut()
            .expect("test terminator")
            .operands = vec![3];

        assert_eq!(
            inspect_program(&reseal(package)),
            Err(CompilerError::InvalidLocalVerification(
                "Q2002: live ownership region for value %2 aliases another live value".to_owned()
            ))
        );
    }

    #[test]
    fn rejects_external_parameters_with_invalid_ownership() {
        let mut package = wire::QCorePackageProto::decode(encoded_program().as_slice())
            .expect("test package decodes");
        let function = &mut package.functions[0];
        function.external = true;
        function.asserted_kind = "unitary".to_owned();
        function.parameters[0]
            .value
            .as_mut()
            .expect("test parameter")
            .r#type = Some(quantum_bit());
        function.body = Some(wire::BlockProto {
            terminator: Some(wire::TerminatorProto {
                kind: "external".to_owned(),
                span: Some(span()),
                ..Default::default()
            }),
            ..Default::default()
        });

        assert_eq!(
            inspect_program(&reseal(package)),
            Err(CompilerError::InvalidLocalVerification(
                "Q1012: linear value %1 has no ownership region".to_owned()
            ))
        );
    }

    #[test]
    fn rejects_bodyless_external_classical_declarations() {
        let mut package = wire::QCorePackageProto::decode(encoded_program().as_slice())
            .expect("test package decodes");
        let function = &mut package.functions[0];
        function.external = true;
        function.asserted_kind = "classical".to_owned();
        function.body = Some(wire::BlockProto {
            terminator: Some(wire::TerminatorProto {
                kind: "external".to_owned(),
                span: Some(span()),
                ..Default::default()
            }),
            ..Default::default()
        });

        assert_eq!(
            inspect_program(&reseal(package)),
            Err(CompilerError::InvalidLocalVerification(
                "Q1047: external function \"native_test.main\" has an invalid declaration"
                    .to_owned()
            ))
        );
    }

    #[test]
    fn rejects_concrete_root_block_arguments() {
        let mut package = wire::QCorePackageProto::decode(encoded_program().as_slice())
            .expect("test package decodes");
        package.functions[0]
            .body
            .as_mut()
            .expect("test body")
            .arguments
            .push(value(2, scalar("Unit")));

        assert_eq!(
            inspect_program(&reseal(package)),
            Err(CompilerError::InvalidLocalVerification(
                "Q1054: concrete function \"native_test.main\" has root block arguments".to_owned()
            ))
        );
    }

    #[test]
    fn rejects_forged_if_condition_metadata() {
        let mut package = wire::QCorePackageProto::decode(encoded_program().as_slice())
            .expect("test package decodes");
        let function = &mut package.functions[0];
        function.parameters.push(wire::ParameterProto {
            name: "flag".to_owned(),
            value: Some(value(2, scalar("Bool"))),
            ..Default::default()
        });
        function
            .body
            .as_mut()
            .expect("test body")
            .operations
            .push(wire::OperationProto {
                opcode: "qcore.if".to_owned(),
                operands: vec![2],
                attributes: vec![
                    attribute("capture_names", string_tuple(&[])),
                    attribute("condition_index", wire::literal_proto::Value::IntValue(1)),
                    attribute("output_names", string_tuple(&[])),
                    attribute(
                        "stage",
                        wire::literal_proto::Value::StringValue("Value".to_owned()),
                    ),
                ],
                regions: vec![
                    branch(vec![], "yield", vec![]),
                    branch(vec![], "yield", vec![]),
                ],
                span: Some(span()),
                ..Default::default()
            });

        assert_eq!(
            inspect_program(&reseal(package)),
            Err(CompilerError::InvalidLocalVerification(
                "Q1056: qcore.if has an invalid structured attribute schema".to_owned()
            ))
        );
    }

    #[test]
    fn accepts_canonical_if_metadata_and_rejects_permuted_rows() {
        let mut package = wire::QCorePackageProto::decode(encoded_program().as_slice())
            .expect("test package decodes");
        let function = &mut package.functions[0];
        function.parameters.push(wire::ParameterProto {
            name: "flag".to_owned(),
            value: Some(value(2, scalar("Bool"))),
            ..Default::default()
        });
        let operation = wire::OperationProto {
            opcode: "qcore.if".to_owned(),
            operands: vec![2],
            attributes: vec![
                attribute("capture_names", string_tuple(&[])),
                attribute("condition_index", wire::literal_proto::Value::IntValue(0)),
                attribute("output_names", string_tuple(&[])),
                attribute(
                    "stage",
                    wire::literal_proto::Value::StringValue("Value".to_owned()),
                ),
            ],
            regions: vec![
                branch(vec![], "yield", vec![]),
                branch(vec![], "yield", vec![]),
            ],
            span: Some(span()),
            ..Default::default()
        };
        function
            .body
            .as_mut()
            .expect("test body")
            .operations
            .push(operation);
        let accepted = reseal(package.clone());
        assert!(inspect_program(&accepted).is_ok());

        package.functions[0]
            .body
            .as_mut()
            .expect("test body")
            .operations[0]
            .attributes
            .swap(0, 1);
        assert_eq!(
            inspect_program(&reseal(package)),
            Err(CompilerError::InvalidLocalVerification(
                "Q1056: qcore.if has an invalid structured attribute schema".to_owned()
            ))
        );
    }

    #[test]
    fn rejects_incompatible_nested_returns() {
        let mut package = wire::QCorePackageProto::decode(encoded_program().as_slice())
            .expect("test package decodes");
        let function = &mut package.functions[0];
        function.parameters.push(wire::ParameterProto {
            name: "flag".to_owned(),
            value: Some(value(2, scalar("Bool"))),
            ..Default::default()
        });
        let body = function.body.as_mut().expect("test body");
        body.operations.push(wire::OperationProto {
            opcode: "qcore.if".to_owned(),
            operands: vec![2, 2],
            results: vec![],
            attributes: vec![
                attribute("capture_names", string_tuple(&["flag"])),
                attribute("condition_index", wire::literal_proto::Value::IntValue(0)),
                attribute("output_names", string_tuple(&[])),
                attribute(
                    "stage",
                    wire::literal_proto::Value::StringValue("Value".to_owned()),
                ),
            ],
            regions: vec![
                branch(vec![value(3, scalar("Bool"))], "return", vec![3]),
                branch(vec![value(4, scalar("Bool"))], "unreachable", vec![]),
            ],
            span: Some(span()),
            intent: Vec::new(),
        });
        body.terminator = Some(wire::TerminatorProto {
            kind: "unreachable".to_owned(),
            span: Some(span()),
            ..Default::default()
        });

        assert_eq!(
            inspect_program(&reseal(package)),
            Err(CompilerError::InvalidLocalVerification(
                "Q1023: function return does not match its declared result type".to_owned()
            ))
        );
    }

    #[test]
    fn rejects_removed_condition_terminators_as_unknown_vocabulary() {
        let mut package = wire::QCorePackageProto::decode(encoded_program().as_slice())
            .expect("test package decodes");
        let terminator = package.functions[0]
            .body
            .as_mut()
            .expect("test body")
            .terminator
            .as_mut()
            .expect("test terminator");
        terminator.kind = "condition".to_owned();

        assert_eq!(
            inspect_program(&reseal(package)),
            Err(CompilerError::InvalidOperationKernel(
                "Q1031: unknown QCore terminator \"condition\"".to_owned()
            ))
        );
    }

    #[test]
    fn rejects_breaks_with_the_wrong_loop_carry_abi() {
        let mut package = wire::QCorePackageProto::decode(encoded_program().as_slice())
            .expect("test package decodes");
        let function = &mut package.functions[0];
        for (id, name) in [(2, "start"), (3, "stop"), (4, "step")] {
            function.parameters.push(wire::ParameterProto {
                name: name.to_owned(),
                value: Some(value(id, scalar("Int"))),
                ..Default::default()
            });
        }
        function.parameters.push(wire::ParameterProto {
            name: "carry".to_owned(),
            value: Some(value(5, scalar("Unit"))),
            ..Default::default()
        });
        let body = function.body.as_mut().expect("test body");
        body.operations.push(wire::OperationProto {
            opcode: "qcore.for".to_owned(),
            operands: vec![2, 3, 4, 5],
            results: vec![value(6, scalar("Unit"))],
            attributes: vec![
                attribute("capture_names", string_tuple(&["carry"])),
                attribute(
                    "induction_name",
                    wire::literal_proto::Value::StringValue("i".to_owned()),
                ),
                attribute(
                    "stage",
                    wire::literal_proto::Value::StringValue("Value".to_owned()),
                ),
            ],
            regions: vec![
                branch(
                    vec![value(7, scalar("Int")), value(8, scalar("Unit"))],
                    "break",
                    vec![],
                ),
                branch(vec![value(9, scalar("Unit"))], "yield", vec![9]),
            ],
            span: Some(span()),
            intent: Vec::new(),
        });
        body.terminator.as_mut().expect("test terminator").operands = vec![6];

        assert_eq!(
            inspect_program(&reseal(package)),
            Err(CompilerError::InvalidLocalVerification(
                "Q1055: break terminator does not match its loop carry topology".to_owned()
            ))
        );
    }
}
