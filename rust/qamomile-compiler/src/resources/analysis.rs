//! Native structural inventory shared by lowering and resource analysis.
//!
//! This pass intentionally counts lexical sites, not dynamic executions. A
//! structured loop body therefore contributes one site here, while a later
//! symbolic resource pass assigns its trip-count algebra. Keeping that
//! distinction explicit prevents a raw wire census from being mislabeled as a
//! resource estimate.

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use thiserror::Error;

use crate::operations::{
    LocallyCheckedOperations, TerminatorKind, defers_gate_families, logical_gate_families,
    primitive_gate_spec,
};
use crate::static_values::{
    StaticScalar, StaticValue as ExactValue, evaluate_value as evaluate_exact_classical,
    integer as static_int, range_len as static_range_len,
    structural_components as exact_structural_components,
    structural_value as exact_structural_value, truth as static_truth,
    typed_value_literal as exact_typed_literal,
};
use crate::types::{Dimension, TypeArena, TypeId, compare_dimensions};
use crate::wire;

const MAX_EXACT_LOOP_INTERPRETATIONS: u128 = 4_096;

/// Lexical operation inventory for one verified callable body.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FunctionStructure {
    symbol: String,
    block_count: usize,
    region_count: usize,
    operation_count: usize,
    call_site_count: usize,
    opcodes: BTreeMap<String, usize>,
}

impl FunctionStructure {
    /// Return the normalized callable symbol.
    #[must_use]
    pub fn symbol(&self) -> &str {
        &self.symbol
    }

    /// Return the number of lexical blocks in this callable.
    #[must_use]
    pub const fn block_count(&self) -> usize {
        self.block_count
    }

    /// Return the number of regions owned by operations in this callable.
    #[must_use]
    pub const fn region_count(&self) -> usize {
        self.region_count
    }

    /// Return the number of lexical operation sites in this callable.
    #[must_use]
    pub const fn operation_count(&self) -> usize {
        self.operation_count
    }

    /// Return the number of ordinary or transformed static call sites.
    #[must_use]
    pub const fn call_site_count(&self) -> usize {
        self.call_site_count
    }

    /// Return canonical opcode spellings and their lexical site counts.
    #[must_use]
    pub fn opcodes(&self) -> &BTreeMap<String, usize> {
        &self.opcodes
    }
}

/// Whole-program lexical inventory derived after native verification.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StructuralCensus {
    entrypoint: String,
    functions: Box<[FunctionStructure]>,
    block_count: usize,
    region_count: usize,
    operation_count: usize,
    call_site_count: usize,
    opcodes: BTreeMap<String, usize>,
}

/// Exact dynamic logical counts for the closed statically analyzable subset.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExactLogicalCounts {
    analysis_operation_visits: u128,
    work: u128,
    opcodes: BTreeMap<String, u128>,
    families: Option<BTreeMap<String, u128>>,
    measurements: Option<u128>,
    resets: Option<u128>,
    input_qubits: Option<u128>,
    allocated_qubits: Option<u128>,
    circuit_qubits: Option<u128>,
    peak_qubits: Option<u128>,
    depths: Option<BTreeMap<String, u128>>,
    cleanup_work: u128,
}

impl ExactLogicalCounts {
    /// Return `QCore` operation visits made by the exact summary traversals.
    ///
    /// This diagnostic work charge covers count, allocation, peak-liveness,
    /// and dependency-timing traversal. It is not a logical resource metric.
    #[must_use]
    pub const fn analysis_operation_visits(&self) -> u128 {
        self.analysis_operation_visits
    }

    /// Return the exact number of primitive quantum operations.
    #[must_use]
    pub const fn work(&self) -> u128 {
        self.work
    }

    /// Return exact dynamic counts by canonical quantum opcode.
    #[must_use]
    pub fn opcodes(&self) -> &BTreeMap<String, u128> {
        &self.opcodes
    }

    /// Return exact dynamic counts by canonical logical gate family.
    ///
    /// Returns `None` when a coherent transform preserves semantic work but
    /// does not prove that the callee's primitive gate families are unchanged.
    #[must_use]
    pub fn families(&self) -> Option<&BTreeMap<String, u128>> {
        self.families.as_ref()
    }

    /// Return the exact number of per-qubit measurement events when proved.
    #[must_use]
    pub const fn measurements(&self) -> Option<u128> {
        self.measurements
    }

    /// Return the exact number of per-qubit reset events when proved.
    #[must_use]
    pub const fn resets(&self) -> Option<u128> {
        self.resets
    }

    /// Return the logical width supplied by the entrypoint caller when proved.
    #[must_use]
    pub const fn input_qubits(&self) -> Option<u128> {
        self.input_qubits
    }

    /// Return the sum of statically reserved logical widths when proved.
    #[must_use]
    pub const fn allocated_qubits(&self) -> Option<u128> {
        self.allocated_qubits
    }

    /// Return caller-supplied plus statically allocated logical width when proved.
    #[must_use]
    pub const fn circuit_qubits(&self) -> Option<u128> {
        self.circuit_qubits
    }

    /// Return the maximum simultaneously live logical width when proved.
    #[must_use]
    pub const fn peak_qubits(&self) -> Option<u128> {
        self.peak_qubits
    }

    /// Return exact dependency span and depth-family maxima when proved.
    #[must_use]
    pub fn depths(&self) -> Option<&BTreeMap<String, u128>> {
        self.depths.as_ref()
    }

    /// Return logical work proven to discharge cleanup obligations.
    #[must_use]
    pub const fn cleanup_work(&self) -> u128 {
        self.cleanup_work
    }
}

/// Failure to derive an exact count without inventing a resource bound.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum ExactResourceError {
    /// The verified program uses control outside the current exact subset.
    #[error("Q4201: exact native logical counts do not support {0}")]
    Unsupported(String),
    /// A correct exact count cannot be represented by the native result type.
    #[error("Q4202: exact native logical count overflow")]
    Overflow,
    /// A verified structural reference was unexpectedly unavailable.
    #[error("Q4203: exact native logical analysis lost {0}")]
    Missing(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Counts {
    work: u128,
    opcodes: BTreeMap<String, u128>,
    families: Option<BTreeMap<String, u128>>,
    measurements: Option<u128>,
    resets: Option<u128>,
    cleanup_work: u128,
}

impl Default for Counts {
    fn default() -> Self {
        Self {
            work: 0,
            opcodes: BTreeMap::new(),
            families: Some(BTreeMap::new()),
            measurements: Some(0),
            resets: Some(0),
            cleanup_work: 0,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct BlockSummary {
    counts: Counts,
    exit: TerminatorKind,
    values: Vec<Option<ExactValue>>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct ExactTimingRoot {
    scope: String,
    root: u64,
    projections: Vec<String>,
    instance: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ExactTimingFootprint {
    root: ExactTimingRoot,
    selectors: Option<Box<[u64]>>,
    width: Option<u128>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ExactTimingStep {
    footprints: Box<[ExactTimingFootprint]>,
    families: Box<[&'static str]>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ExactTimingSummary {
    steps: Vec<ExactTimingStep>,
    exit: TerminatorKind,
    values: Vec<Option<ExactValue>>,
}

type ExactTimingArguments = (Vec<Option<ExactValue>>, HashMap<u64, ExactTimingRoot>);

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct ExactLaneTimeline {
    barrier: u128,
    lanes: HashMap<u64, u128>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct StaticLoop {
    lower: i64,
    stride: i64,
    trip_count: u128,
    reverse: bool,
}

fn exact_terminator_kind(
    terminator: &wire::TerminatorProto,
) -> Result<TerminatorKind, ExactResourceError> {
    TerminatorKind::try_from(terminator.kind.as_str()).map_err(|_| {
        ExactResourceError::Missing(format!("verified terminator kind {:?}", terminator.kind))
    })
}

impl Counts {
    fn add(
        &mut self,
        opcode: &str,
        effective_opcode: &str,
        event_width: Option<u128>,
        cleanup: bool,
        count: u128,
    ) -> Result<(), ExactResourceError> {
        self.work = self
            .work
            .checked_add(count)
            .ok_or(ExactResourceError::Overflow)?;
        add_count(&mut self.opcodes, opcode, count)?;
        if count > 0 && defers_gate_families(effective_opcode) {
            self.families = None;
        }
        if let Some(families) = &mut self.families {
            for family in logical_gate_families(effective_opcode) {
                add_count(families, family, count)?;
            }
        }
        if is_measurement(opcode) {
            add_optional_count(
                &mut self.measurements,
                event_count(opcode, event_width),
                count,
            )?;
        }
        if is_reset(opcode) {
            add_optional_count(&mut self.resets, event_count(opcode, event_width), count)?;
        }
        if cleanup {
            self.cleanup_work = self
                .cleanup_work
                .checked_add(count)
                .ok_or(ExactResourceError::Overflow)?;
        }
        Ok(())
    }

    fn then(&mut self, other: &Self) -> Result<(), ExactResourceError> {
        self.work = self
            .work
            .checked_add(other.work)
            .ok_or(ExactResourceError::Overflow)?;
        for (opcode, count) in &other.opcodes {
            add_count(&mut self.opcodes, opcode, *count)?;
        }
        match (&mut self.families, &other.families) {
            (Some(destination), Some(source)) => {
                for (family, count) in source {
                    add_count(destination, family, *count)?;
                }
            }
            (_, None) => self.families = None,
            (None, Some(_)) => {}
        }
        merge_optional_count(&mut self.measurements, other.measurements)?;
        merge_optional_count(&mut self.resets, other.resets)?;
        self.cleanup_work = self
            .cleanup_work
            .checked_add(other.cleanup_work)
            .ok_or(ExactResourceError::Overflow)?;
        Ok(())
    }

    fn repeat(&self, count: u128) -> Result<Self, ExactResourceError> {
        let mut result = Self {
            work: self
                .work
                .checked_mul(count)
                .ok_or(ExactResourceError::Overflow)?,
            ..Self::default()
        };
        for (opcode, value) in &self.opcodes {
            result.opcodes.insert(
                opcode.clone(),
                value
                    .checked_mul(count)
                    .ok_or(ExactResourceError::Overflow)?,
            );
        }
        result.families = self
            .families
            .as_ref()
            .map(|families| {
                families
                    .iter()
                    .map(|(family, value)| {
                        Ok((
                            family.clone(),
                            value
                                .checked_mul(count)
                                .ok_or(ExactResourceError::Overflow)?,
                        ))
                    })
                    .collect()
            })
            .transpose()?;
        result.measurements = multiply_optional_count(self.measurements, count)?;
        result.resets = multiply_optional_count(self.resets, count)?;
        result.cleanup_work = self
            .cleanup_work
            .checked_mul(count)
            .ok_or(ExactResourceError::Overflow)?;
        Ok(result)
    }

    fn cleanup_adjoint(&self) -> Self {
        let mut opcodes = BTreeMap::new();
        if self.work != 0 {
            opcodes.insert("qcore.adjoint".to_owned(), self.work);
        }
        Self {
            work: self.work,
            opcodes,
            families: self.families.clone(),
            measurements: Some(0),
            resets: Some(0),
            cleanup_work: self.work,
        }
    }
}

fn add_count(
    counts: &mut BTreeMap<String, u128>,
    name: &str,
    count: u128,
) -> Result<(), ExactResourceError> {
    let current = counts.entry(name.to_owned()).or_default();
    *current = current
        .checked_add(count)
        .ok_or(ExactResourceError::Overflow)?;
    Ok(())
}

fn add_optional_count(
    total: &mut Option<u128>,
    value: Option<u128>,
    multiplier: u128,
) -> Result<(), ExactResourceError> {
    let Some(value) = value else {
        *total = None;
        return Ok(());
    };
    let contribution = value
        .checked_mul(multiplier)
        .ok_or(ExactResourceError::Overflow)?;
    merge_optional_count(total, Some(contribution))
}

fn merge_optional_count(
    total: &mut Option<u128>,
    value: Option<u128>,
) -> Result<(), ExactResourceError> {
    *total = match (*total, value) {
        (Some(left), Some(right)) => Some(
            left.checked_add(right)
                .ok_or(ExactResourceError::Overflow)?,
        ),
        _ => None,
    };
    Ok(())
}

fn multiply_optional_count(
    value: Option<u128>,
    multiplier: u128,
) -> Result<Option<u128>, ExactResourceError> {
    value
        .map(|value| {
            value
                .checked_mul(multiplier)
                .ok_or(ExactResourceError::Overflow)
        })
        .transpose()
}

struct ExactAnalyzer<'a> {
    functions: HashMap<&'a str, &'a wire::FunctionProto>,
    types: &'a TypeArena,
    operations: &'a LocallyCheckedOperations,
    analysis_cache: RefCell<HashMap<ExactAnalysisKey, BlockSummary>>,
    operation_visits: Cell<u128>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct ExactAnalysisKey {
    symbol: String,
    arguments: Box<[Option<ExactValue>]>,
    dimensions: Box<[(Dimension, u64)]>,
}

struct ExactDimensionBindings {
    values: HashMap<Dimension, u64>,
}

#[derive(Clone, Copy)]
struct ExactFunctionContext<'a> {
    owner: &'a str,
    dimensions: &'a ExactDimensionBindings,
}

impl StructuralCensus {
    /// Return the selected verified entrypoint.
    #[must_use]
    pub fn entrypoint(&self) -> &str {
        &self.entrypoint
    }

    /// Return function inventories in canonical package order.
    #[must_use]
    pub fn functions(&self) -> &[FunctionStructure] {
        &self.functions
    }

    /// Return the total number of lexical blocks.
    #[must_use]
    pub const fn block_count(&self) -> usize {
        self.block_count
    }

    /// Return the total number of structured regions.
    #[must_use]
    pub const fn region_count(&self) -> usize {
        self.region_count
    }

    /// Return the total number of lexical operation sites.
    #[must_use]
    pub const fn operation_count(&self) -> usize {
        self.operation_count
    }

    /// Return the total number of normalized static call sites.
    #[must_use]
    pub const fn call_site_count(&self) -> usize {
        self.call_site_count
    }

    /// Return aggregate canonical opcode spellings and lexical site counts.
    #[must_use]
    pub fn opcodes(&self) -> &BTreeMap<String, usize> {
        &self.opcodes
    }
}

pub(crate) fn census(package: &wire::QCorePackageProto) -> StructuralCensus {
    let functions = package
        .functions
        .iter()
        .map(function_structure)
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let mut opcodes = BTreeMap::new();
    for function in &functions {
        merge_counts(&mut opcodes, &function.opcodes);
    }
    StructuralCensus {
        entrypoint: package.entrypoint.clone(),
        block_count: functions.iter().map(FunctionStructure::block_count).sum(),
        region_count: functions.iter().map(FunctionStructure::region_count).sum(),
        operation_count: functions
            .iter()
            .map(FunctionStructure::operation_count)
            .sum(),
        call_site_count: functions
            .iter()
            .map(FunctionStructure::call_site_count)
            .sum(),
        functions,
        opcodes,
    }
}

fn function_structure(function: &wire::FunctionProto) -> FunctionStructure {
    let mut structure = FunctionStructure {
        symbol: function.symbol.clone(),
        block_count: 0,
        region_count: 0,
        operation_count: 0,
        call_site_count: 0,
        opcodes: BTreeMap::new(),
    };
    let mut blocks = function.body.iter().collect::<Vec<_>>();
    while let Some(block) = blocks.pop() {
        structure.block_count += 1;
        for operation in &block.operations {
            structure.operation_count += 1;
            *structure
                .opcodes
                .entry(operation.opcode.clone())
                .or_default() += 1;
            if matches!(
                operation.opcode.as_str(),
                "qcore.call"
                    | "qcore.adjoint_call"
                    | "qcore.control_call"
                    | "qcore.select_call"
                    | "qcore.phase_by"
                    | "qcore.xor_by"
                    | "qcore.permute_by"
            ) {
                structure.call_site_count += 1;
            }
            structure.region_count += operation.regions.len();
            for region in &operation.regions {
                blocks.extend(&region.blocks);
            }
        }
    }
    structure
}

fn merge_counts(destination: &mut BTreeMap<String, usize>, source: &BTreeMap<String, usize>) {
    for (opcode, count) in source {
        *destination.entry(opcode.clone()).or_default() += count;
    }
}

pub(crate) fn exact_logical_counts(
    package: &wire::QCorePackageProto,
    types: &TypeArena,
    operations: &LocallyCheckedOperations,
) -> Result<ExactLogicalCounts, ExactResourceError> {
    let analyzer = ExactAnalyzer {
        functions: package
            .functions
            .iter()
            .map(|function| (function.symbol.as_str(), function))
            .collect(),
        types,
        operations,
        analysis_cache: RefCell::new(HashMap::new()),
        operation_visits: Cell::new(0),
    };
    let entry = analyzer
        .functions
        .get(package.entrypoint.as_str())
        .copied()
        .ok_or_else(|| ExactResourceError::Missing("entrypoint".to_owned()))?;
    let bindings = package
        .const_bindings
        .iter()
        .map(|binding| (binding.name.as_str(), binding.value.as_ref()))
        .collect::<HashMap<_, _>>();
    let arguments = entry
        .parameters
        .iter()
        .map(|parameter| {
            let literal = bindings.get(parameter.name.as_str()).copied().flatten()?;
            let type_spec = parameter.value.as_ref()?.r#type.as_ref()?;
            exact_typed_literal(literal, type_spec)
        })
        .collect::<Vec<_>>();
    let dimensions = exact_dimension_bindings(entry, &arguments, &HashMap::new());
    let input_qubits = analyzer.input_width(entry, &dimensions);
    let summary = analyzer.analyze_function(entry, &arguments, &HashMap::new())?;
    let allocated_qubits = analyzer
        .allocated_function(entry, &arguments, &HashMap::new())
        .unwrap_or_default();
    let circuit_qubits = input_qubits
        .zip(allocated_qubits)
        .and_then(|(input, allocated)| input.checked_add(allocated));
    let peak_qubits = analyzer
        .peak_function(entry, &arguments, &HashMap::new())
        .unwrap_or_default();
    let mut next_timing_instance = 1;
    let depths = analyzer
        .timing_function(
            entry,
            &arguments,
            &HashMap::new(),
            &HashMap::new(),
            0,
            &mut next_timing_instance,
        )
        .ok()
        .flatten()
        .and_then(|timing| schedule_exact_depths(&timing.steps).ok());
    Ok(ExactLogicalCounts {
        analysis_operation_visits: analyzer.operation_visits.get(),
        work: summary.counts.work,
        opcodes: summary.counts.opcodes,
        families: summary.counts.families,
        measurements: summary.counts.measurements,
        resets: summary.counts.resets,
        input_qubits,
        allocated_qubits,
        circuit_qubits,
        peak_qubits,
        depths,
        cleanup_work: summary.counts.cleanup_work,
    })
}

impl ExactAnalyzer<'_> {
    fn charge_block(&self, block: &wire::BlockProto) {
        self.operation_visits.set(
            self.operation_visits
                .get()
                .saturating_add(block.operations.len() as u128),
        );
    }

    fn input_width(
        &self,
        function: &wire::FunctionProto,
        dimensions: &ExactDimensionBindings,
    ) -> Option<u128> {
        let types = self.operations.value_types(&function.symbol)?;
        function
            .parameters
            .iter()
            .filter_map(|parameter| parameter.value.as_ref().filter(|value| value.has_region))
            .try_fold(0_u128, |total, value| {
                let type_id = types.get(&value.id)?;
                total.checked_add(u128::from(coherent_width(
                    self.types, *type_id, dimensions,
                )?))
            })
    }

    fn peak_function(
        &self,
        function: &wire::FunctionProto,
        arguments: &[Option<ExactValue>],
        dimension_arguments: &HashMap<Dimension, u64>,
    ) -> Result<Option<u128>, ExactResourceError> {
        if function.external || arguments.len() != function.parameters.len() {
            return Ok(None);
        }
        let mut environment = function
            .parameters
            .iter()
            .zip(arguments)
            .map(|(parameter, value)| {
                parameter
                    .value
                    .as_ref()
                    .map(|definition| (definition.id, value.clone()))
                    .ok_or_else(|| {
                        ExactResourceError::Missing(format!(
                            "parameter value in {:?}",
                            function.symbol
                        ))
                    })
            })
            .collect::<Result<HashMap<_, _>, _>>()?;
        let dimensions = exact_dimension_bindings(function, arguments, dimension_arguments);
        let Some(types) = self.operations.value_types(&function.symbol) else {
            return Ok(None);
        };
        let mut active = HashMap::new();
        for parameter in &function.parameters {
            let Some(value) = parameter.value.as_ref().filter(|value| value.has_region) else {
                continue;
            };
            let (Some(region), Some(type_id)) = (value.region.as_ref(), types.get(&value.id))
            else {
                return Ok(None);
            };
            let Some(width) = coherent_width(self.types, *type_id, &dimensions) else {
                return Ok(None);
            };
            active.insert(region.root, u128::from(width));
        }
        let body = function.body.as_ref().ok_or_else(|| {
            ExactResourceError::Missing(format!("body for {:?}", function.symbol))
        })?;
        let roots = function_value_roots(function);
        self.peak_block(
            body,
            &mut environment,
            &mut active,
            &roots,
            ExactFunctionContext {
                owner: &function.symbol,
                dimensions: &dimensions,
            },
        )
    }

    fn peak_block(
        &self,
        block: &wire::BlockProto,
        environment: &mut HashMap<u64, Option<ExactValue>>,
        active: &mut HashMap<u64, u128>,
        roots: &HashMap<u64, u64>,
        context: ExactFunctionContext<'_>,
    ) -> Result<Option<u128>, ExactResourceError> {
        self.charge_block(block);
        let mut peak = checked_active_width(active)?;
        for operation in &block.operations {
            if operation.opcode == "qcore.call" {
                let Some(call_peak) = self.peak_call(operation, environment, active, context)?
                else {
                    return Ok(None);
                };
                peak = peak.max(call_peak);
            }
            let Some(region_peak) =
                self.peak_regions(operation, environment, active, roots, context)?
            else {
                return Ok(None);
            };
            peak = peak.max(region_peak);
            if !self.advance_peak_liveness(operation, active, roots, context) {
                return Ok(None);
            }
            peak = peak.max(checked_active_width(active)?);
            self.bind_peak_results(operation, environment, context)?;
        }
        Ok(Some(peak))
    }

    fn peak_call(
        &self,
        operation: &wire::OperationProto,
        environment: &HashMap<u64, Option<ExactValue>>,
        active: &HashMap<u64, u128>,
        context: ExactFunctionContext<'_>,
    ) -> Result<Option<u128>, ExactResourceError> {
        let callee_name = string_attribute(operation, "callee")
            .ok_or_else(|| ExactResourceError::Missing("callee".to_owned()))?;
        let callee = self
            .functions
            .get(callee_name)
            .copied()
            .ok_or_else(|| ExactResourceError::Missing(format!("callable {callee_name:?}")))?;
        let arguments = operation
            .operands
            .iter()
            .map(|operand| environment.get(operand).cloned().flatten())
            .collect::<Vec<_>>();
        let dimension_arguments = self.call_dimensions(operation, callee, context)?;
        let Some(callee_peak) = self.peak_function(callee, &arguments, &dimension_arguments)?
        else {
            return Ok(None);
        };
        let dimensions = exact_dimension_bindings(callee, &arguments, &dimension_arguments);
        let Some(input_width) = self.input_width(callee, &dimensions) else {
            return Ok(None);
        };
        let extra = callee_peak.saturating_sub(input_width);
        Ok(Some(
            checked_active_width(active)?
                .checked_add(extra)
                .ok_or(ExactResourceError::Overflow)?,
        ))
    }

    fn peak_regions(
        &self,
        operation: &wire::OperationProto,
        environment: &HashMap<u64, Option<ExactValue>>,
        active: &HashMap<u64, u128>,
        roots: &HashMap<u64, u64>,
        context: ExactFunctionContext<'_>,
    ) -> Result<Option<u128>, ExactResourceError> {
        if operation.regions.is_empty() {
            return Ok(Some(0));
        }
        let carries = operation.operands.get(3..).unwrap_or_default();
        let invariants = operation
            .regions
            .first()
            .and_then(|region| region.blocks.first())
            .map(|block| loop_carry_invariants(block, carries.len()))
            .unwrap_or_default();
        let mut peak = 0;
        for (index, region) in operation.regions.iter().enumerate() {
            let block = region.blocks.first().ok_or_else(|| {
                ExactResourceError::Missing(format!("{} region", operation.opcode))
            })?;
            let (operands, facts) = match operation.opcode.as_str() {
                "qcore.if" => (
                    operation.operands.get(1..).unwrap_or_default(),
                    RegionFacts::Captured,
                ),
                "qcore.for" if index == 0 => (carries, RegionFacts::LoopBody(&invariants)),
                "qcore.for" => (carries, RegionFacts::Captured),
                _ => return Ok(None),
            };
            let mut local = bind_region(block, operands, environment, facts)?;
            let mut nested_active = active.clone();
            let Some(nested_peak) =
                self.peak_block(block, &mut local, &mut nested_active, roots, context)?
            else {
                return Ok(None);
            };
            peak = peak.max(nested_peak);
        }
        Ok(Some(peak))
    }

    fn advance_peak_liveness(
        &self,
        operation: &wire::OperationProto,
        active: &mut HashMap<u64, u128>,
        roots: &HashMap<u64, u64>,
        context: ExactFunctionContext<'_>,
    ) -> bool {
        let result_roots = operation
            .results
            .iter()
            .filter_map(|result| result.region.as_ref().map(|region| region.root))
            .collect::<Vec<_>>();
        let Some(types) = self.operations.value_types(context.owner) else {
            return false;
        };
        for result in &operation.results {
            let Some(region) = result.region.as_ref() else {
                continue;
            };
            if active.contains_key(&region.root) {
                continue;
            }
            let Some(type_id) = types.get(&result.id) else {
                return false;
            };
            let Some(width) = coherent_width(self.types, *type_id, context.dimensions) else {
                return false;
            };
            active.insert(region.root, u128::from(width));
        }
        for operand in &operation.operands {
            let Some(root) = roots.get(operand) else {
                continue;
            };
            if !result_roots.contains(root) {
                active.remove(root);
            }
        }
        true
    }

    fn bind_peak_results(
        &self,
        operation: &wire::OperationProto,
        environment: &mut HashMap<u64, Option<ExactValue>>,
        context: ExactFunctionContext<'_>,
    ) -> Result<(), ExactResourceError> {
        if bind_structural_exact_facts(operation, environment)? {
            return Ok(());
        }
        match operation.opcode.as_str() {
            "qcore.call" => {
                let summary = self.analyze_ordinary_call(operation, environment, context)?;
                bind_exact_results(operation, environment, &summary.values)
            }
            "qcore.if" => self.bind_peak_if_results(operation, environment, context),
            "qcore.for" => {
                let summary = self.analyze_for(operation, environment, context)?;
                bind_exact_results(operation, environment, &summary.values)
            }
            opcode if opcode.starts_with("classical.") => {
                let value = evaluate_exact_classical(operation, environment);
                bind_results(operation, environment, value.as_ref());
                Ok(())
            }
            _ => {
                bind_results(operation, environment, None);
                Ok(())
            }
        }
    }

    fn bind_peak_if_results(
        &self,
        operation: &wire::OperationProto,
        environment: &mut HashMap<u64, Option<ExactValue>>,
        context: ExactFunctionContext<'_>,
    ) -> Result<(), ExactResourceError> {
        let condition = operation
            .operands
            .first()
            .and_then(|operand| environment.get(operand))
            .and_then(Option::as_ref)
            .and_then(ExactValue::scalar)
            .map(static_truth)
            .ok_or_else(|| {
                ExactResourceError::Unsupported("dynamic qcore.if condition".to_owned())
            })?;
        let branch = operation
            .regions
            .get(usize::from(!condition))
            .and_then(|region| region.blocks.first())
            .ok_or_else(|| ExactResourceError::Missing("qcore.if branch".to_owned()))?;
        let mut local = bind_region(
            branch,
            operation.operands.get(1..).unwrap_or_default(),
            environment,
            RegionFacts::Captured,
        )?;
        let summary = self.analyze_block(branch, &mut local, context)?;
        bind_exact_results(operation, environment, &summary.values)
    }

    fn timing_function(
        &self,
        function: &wire::FunctionProto,
        arguments: &[Option<ExactValue>],
        dimension_arguments: &HashMap<Dimension, u64>,
        roots: &HashMap<u64, ExactTimingRoot>,
        instance: u64,
        next_instance: &mut u64,
    ) -> Result<Option<ExactTimingSummary>, ExactResourceError> {
        if function.external || arguments.len() != function.parameters.len() {
            return Ok(None);
        }
        let mut environment = function
            .parameters
            .iter()
            .zip(arguments)
            .map(|(parameter, value)| {
                parameter
                    .value
                    .as_ref()
                    .map(|definition| (definition.id, value.clone()))
                    .ok_or_else(|| {
                        ExactResourceError::Missing(format!(
                            "parameter value in {:?}",
                            function.symbol
                        ))
                    })
            })
            .collect::<Result<HashMap<_, _>, _>>()?;
        let dimensions = exact_dimension_bindings(function, arguments, dimension_arguments);
        let body = function.body.as_ref().ok_or_else(|| {
            ExactResourceError::Missing(format!("body for {:?}", function.symbol))
        })?;
        let regions = function_value_regions(function);
        self.timing_block(
            body,
            &mut environment,
            &regions,
            roots,
            instance,
            next_instance,
            ExactFunctionContext {
                owner: &function.symbol,
                dimensions: &dimensions,
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn timing_block(
        &self,
        block: &wire::BlockProto,
        environment: &mut HashMap<u64, Option<ExactValue>>,
        regions: &HashMap<u64, wire::RegionRefProto>,
        roots: &HashMap<u64, ExactTimingRoot>,
        instance: u64,
        next_instance: &mut u64,
        context: ExactFunctionContext<'_>,
    ) -> Result<Option<ExactTimingSummary>, ExactResourceError> {
        self.charge_block(block);
        let mut steps = Vec::new();
        for operation in &block.operations {
            let nested = match operation.opcode.as_str() {
                "qcore.call" => self.timing_call(
                    operation,
                    environment,
                    regions,
                    roots,
                    instance,
                    next_instance,
                    context,
                )?,
                "qcore.if" => self.timing_if(
                    operation,
                    environment,
                    regions,
                    roots,
                    instance,
                    next_instance,
                    context,
                )?,
                "qcore.for" => self.timing_for(
                    operation,
                    environment,
                    regions,
                    roots,
                    instance,
                    next_instance,
                    context,
                )?,
                "qcore.adjoint_call" => self.timing_adjoint_call(
                    operation,
                    environment,
                    regions,
                    roots,
                    instance,
                    next_instance,
                    context,
                )?,
                "qcore.adjoint" => {
                    let original = string_attribute(operation, "original").ok_or_else(|| {
                        ExactResourceError::Missing("adjoint original opcode".to_owned())
                    })?;
                    if matches!(original, "qcore.control_call" | "qcore.select_call") {
                        return Ok(None);
                    }
                    self.timing_quantum(operation, environment, regions, roots, instance, context)?
                        .map(|step| vec![step])
                }
                "qcore.control_call" | "qcore.select_call" => return Ok(None),
                "qcore.phase_by" | "qcore.xor_by" | "qcore.permute_by" => self
                    .timing_quantum(operation, environment, regions, roots, instance, context)?
                    .map(|step| vec![step]),
                opcode if opcode.starts_with("quantum.") => self
                    .timing_quantum(operation, environment, regions, roots, instance, context)?
                    .map(|step| vec![step]),
                _ => Some(Vec::new()),
            };
            let Some(nested) = nested else {
                return Ok(None);
            };
            steps.extend(nested);
            self.bind_peak_results(operation, environment, context)?;
        }
        let terminator = block
            .terminator
            .as_ref()
            .ok_or_else(|| ExactResourceError::Missing("block terminator".to_owned()))?;
        Ok(Some(ExactTimingSummary {
            steps,
            exit: exact_terminator_kind(terminator)?,
            values: terminator
                .operands
                .iter()
                .map(|operand| environment.get(operand).cloned().flatten())
                .collect(),
        }))
    }

    #[allow(clippy::too_many_arguments)]
    fn timing_call(
        &self,
        operation: &wire::OperationProto,
        environment: &HashMap<u64, Option<ExactValue>>,
        regions: &HashMap<u64, wire::RegionRefProto>,
        roots: &HashMap<u64, ExactTimingRoot>,
        instance: u64,
        next_instance: &mut u64,
        context: ExactFunctionContext<'_>,
    ) -> Result<Option<Vec<ExactTimingStep>>, ExactResourceError> {
        let callee_name = string_attribute(operation, "callee")
            .ok_or_else(|| ExactResourceError::Missing("callee".to_owned()))?;
        let callee = self
            .functions
            .get(callee_name)
            .copied()
            .ok_or_else(|| ExactResourceError::Missing(format!("callable {callee_name:?}")))?;
        let arguments = operation
            .operands
            .iter()
            .map(|operand| environment.get(operand).cloned().flatten())
            .collect::<Vec<_>>();
        let dimensions = self.call_dimensions(operation, callee, context)?;
        let Some(bindings) =
            Self::timing_call_roots(callee, operation, regions, roots, context.owner, instance)
        else {
            return Ok(None);
        };
        let call_instance = take_timing_instance(next_instance)?;
        Ok(self
            .timing_function(
                callee,
                &arguments,
                &dimensions,
                &bindings,
                call_instance,
                next_instance,
            )?
            .map(|summary| summary.steps))
    }

    #[allow(clippy::too_many_arguments)]
    fn timing_if(
        &self,
        operation: &wire::OperationProto,
        environment: &HashMap<u64, Option<ExactValue>>,
        regions: &HashMap<u64, wire::RegionRefProto>,
        roots: &HashMap<u64, ExactTimingRoot>,
        instance: u64,
        next_instance: &mut u64,
        context: ExactFunctionContext<'_>,
    ) -> Result<Option<Vec<ExactTimingStep>>, ExactResourceError> {
        let condition = operation
            .operands
            .first()
            .and_then(|operand| environment.get(operand))
            .and_then(Option::as_ref)
            .and_then(ExactValue::scalar)
            .map(static_truth)
            .ok_or_else(|| {
                ExactResourceError::Unsupported("dynamic qcore.if condition".to_owned())
            })?;
        let branch = operation
            .regions
            .get(usize::from(!condition))
            .and_then(|region| region.blocks.first())
            .ok_or_else(|| ExactResourceError::Missing("qcore.if branch".to_owned()))?;
        let mut local = bind_region(
            branch,
            operation.operands.get(1..).unwrap_or_default(),
            environment,
            RegionFacts::Captured,
        )?;
        Ok(self
            .timing_block(
                branch,
                &mut local,
                regions,
                roots,
                instance,
                next_instance,
                context,
            )?
            .map(|summary| {
                let mut steps = summary.steps;
                if string_attribute(operation, "body_mode") == Some("adjoint") {
                    steps.reverse();
                }
                steps
            }))
    }

    #[allow(clippy::too_many_arguments)]
    fn timing_for(
        &self,
        operation: &wire::OperationProto,
        environment: &HashMap<u64, Option<ExactValue>>,
        regions: &HashMap<u64, wire::RegionRefProto>,
        roots: &HashMap<u64, ExactTimingRoot>,
        instance: u64,
        next_instance: &mut u64,
        context: ExactFunctionContext<'_>,
    ) -> Result<Option<Vec<ExactTimingStep>>, ExactResourceError> {
        let schedule = exact_static_loop(operation, environment)?;
        if schedule.trip_count > MAX_EXACT_LOOP_INTERPRETATIONS {
            return Ok(None);
        }
        let body = operation
            .regions
            .first()
            .and_then(|region| region.blocks.first())
            .ok_or_else(|| ExactResourceError::Missing("qcore.for body".to_owned()))?;
        let carries = operation.operands.get(3..).unwrap_or_default();
        let mut values = carries
            .iter()
            .map(|operand| environment.get(operand).cloned().flatten())
            .collect::<Vec<_>>();
        let mut steps = Vec::new();
        let mut broke = false;
        for position in 0..schedule.trip_count {
            let induction = exact_loop_induction(schedule, position)?;
            let arguments = std::iter::once(Some(ExactValue::Scalar(StaticScalar::Int(induction))))
                .chain(values.iter().cloned())
                .collect();
            let mut local = bind_region_values(body, arguments)?;
            let Some(summary) = self.timing_block(
                body,
                &mut local,
                regions,
                roots,
                instance,
                next_instance,
                context,
            )?
            else {
                return Ok(None);
            };
            let mut iteration_steps = summary.steps;
            if string_attribute(operation, "body_mode") == Some("adjoint") {
                iteration_steps.reverse();
            }
            steps.extend(iteration_steps);
            match summary.exit {
                TerminatorKind::Yield | TerminatorKind::Continue => values = summary.values,
                TerminatorKind::Break => {
                    broke = true;
                    values = summary.values;
                    break;
                }
                _ => return Ok(None),
            }
        }
        if !broke
            && let Some(otherwise) = operation
                .regions
                .get(1)
                .and_then(|region| region.blocks.first())
        {
            let mut local = bind_region_values(otherwise, values)?;
            let Some(summary) = self.timing_block(
                otherwise,
                &mut local,
                regions,
                roots,
                instance,
                next_instance,
                context,
            )?
            else {
                return Ok(None);
            };
            if summary.exit != TerminatorKind::Yield {
                return Ok(None);
            }
            steps.extend(summary.steps);
        }
        Ok(Some(steps))
    }

    #[allow(clippy::too_many_arguments)]
    fn timing_adjoint_call(
        &self,
        operation: &wire::OperationProto,
        environment: &HashMap<u64, Option<ExactValue>>,
        regions: &HashMap<u64, wire::RegionRefProto>,
        roots: &HashMap<u64, ExactTimingRoot>,
        instance: u64,
        next_instance: &mut u64,
        context: ExactFunctionContext<'_>,
    ) -> Result<Option<Vec<ExactTimingStep>>, ExactResourceError> {
        let callee_name = string_attribute(operation, "callee")
            .ok_or_else(|| ExactResourceError::Missing("adjoint callee".to_owned()))?;
        let callee = self.functions.get(callee_name).copied().ok_or_else(|| {
            ExactResourceError::Missing(format!("adjoint callable {callee_name:?}"))
        })?;
        let Some((arguments, bindings)) = Self::timing_adjoint_arguments(
            callee,
            operation,
            environment,
            regions,
            roots,
            instance,
            context.owner,
        ) else {
            return Ok(None);
        };
        let call_instance = take_timing_instance(next_instance)?;
        let Some(mut summary) = self.timing_function(
            callee,
            &arguments,
            &HashMap::new(),
            &bindings,
            call_instance,
            next_instance,
        )?
        else {
            return Ok(None);
        };
        summary.steps.reverse();
        Ok(Some(summary.steps))
    }

    fn timing_quantum(
        &self,
        operation: &wire::OperationProto,
        environment: &HashMap<u64, Option<ExactValue>>,
        regions: &HashMap<u64, wire::RegionRefProto>,
        roots: &HashMap<u64, ExactTimingRoot>,
        instance: u64,
        context: ExactFunctionContext<'_>,
    ) -> Result<Option<ExactTimingStep>, ExactResourceError> {
        let effective = if operation.opcode == "qcore.adjoint" {
            string_attribute(operation, "original")
                .ok_or_else(|| ExactResourceError::Missing("adjoint primitive opcode".to_owned()))?
        } else {
            operation.opcode.as_str()
        };
        if defers_gate_families(effective) {
            return Ok(None);
        }
        let value_ids = if matches!(effective, "quantum.alloc" | "quantum.prepare") {
            operation
                .results
                .iter()
                .filter(|value| value.has_region)
                .map(|value| value.id)
                .collect::<Vec<_>>()
        } else {
            operation
                .operands
                .iter()
                .filter(|operand| regions.contains_key(operand))
                .copied()
                .collect()
        };
        if value_ids.is_empty() {
            return Ok(None);
        }
        let selectors = exact_timing_selectors(operation, effective, environment)?;
        let types = self.operations.value_types(context.owner).ok_or_else(|| {
            ExactResourceError::Missing(format!("type roots for {:?}", context.owner))
        })?;
        let footprints = value_ids
            .iter()
            .map(|value_id| {
                let region = regions.get(value_id).ok_or_else(|| {
                    ExactResourceError::Missing("timing ownership region".to_owned())
                })?;
                let type_id = types
                    .get(value_id)
                    .copied()
                    .ok_or_else(|| ExactResourceError::Missing("timing value type".to_owned()))?;
                Ok(ExactTimingFootprint {
                    root: exact_timing_root(region, context.owner, roots, instance),
                    selectors: selectors.clone(),
                    width: coherent_width(self.types, type_id, context.dimensions).map(u128::from),
                })
            })
            .collect::<Result<Vec<_>, ExactResourceError>>()?;
        Ok(Some(ExactTimingStep {
            footprints: footprints.into_boxed_slice(),
            families: depth_families(effective).into_boxed_slice(),
        }))
    }

    fn timing_call_roots(
        callee: &wire::FunctionProto,
        operation: &wire::OperationProto,
        regions: &HashMap<u64, wire::RegionRefProto>,
        roots: &HashMap<u64, ExactTimingRoot>,
        owner: &str,
        instance: u64,
    ) -> Option<HashMap<u64, ExactTimingRoot>> {
        let mut bindings = HashMap::new();
        for (parameter, operand) in callee.parameters.iter().zip(&operation.operands) {
            let formal = parameter.value.as_ref()?.region.as_ref();
            let actual = regions.get(operand);
            if let (Some(formal), Some(actual)) = (formal, actual)
                && !insert_timing_root(
                    &mut bindings,
                    formal.root,
                    exact_timing_root(actual, owner, roots, instance),
                )
            {
                return None;
            }
        }
        Self::bind_timing_results(
            callee,
            operation,
            regions,
            roots,
            owner,
            instance,
            &mut bindings,
        )
        .then_some(bindings)
    }

    #[allow(clippy::too_many_arguments)]
    fn bind_timing_results(
        callee: &wire::FunctionProto,
        operation: &wire::OperationProto,
        regions: &HashMap<u64, wire::RegionRefProto>,
        roots: &HashMap<u64, ExactTimingRoot>,
        owner: &str,
        instance: u64,
        bindings: &mut HashMap<u64, ExactTimingRoot>,
    ) -> bool {
        let callee_regions = function_value_regions(callee);
        let Some(terminator) = callee
            .body
            .as_ref()
            .and_then(|body| body.terminator.as_ref())
        else {
            return false;
        };
        for (returned, result) in terminator.operands.iter().zip(&operation.results) {
            let (Some(formal), Some(actual)) =
                (callee_regions.get(returned), regions.get(&result.id))
            else {
                continue;
            };
            if !insert_timing_root(
                bindings,
                formal.root,
                exact_timing_root(actual, owner, roots, instance),
            ) {
                return false;
            }
        }
        true
    }

    #[allow(clippy::too_many_arguments)]
    fn timing_adjoint_arguments(
        callee: &wire::FunctionProto,
        operation: &wire::OperationProto,
        environment: &HashMap<u64, Option<ExactValue>>,
        regions: &HashMap<u64, wire::RegionRefProto>,
        roots: &HashMap<u64, ExactTimingRoot>,
        instance: u64,
        owner: &str,
    ) -> Option<ExactTimingArguments> {
        let quantum = callee
            .parameters
            .iter()
            .filter(|parameter| {
                parameter
                    .value
                    .as_ref()
                    .is_some_and(|value| value.has_region)
            })
            .collect::<Vec<_>>();
        if quantum.len() != 1 {
            return None;
        }
        let state = *operation.operands.first()?;
        let actual = regions.get(&state)?;
        let formal = quantum[0].value.as_ref()?.region.as_ref()?;
        let mut bindings = HashMap::from([(
            formal.root,
            exact_timing_root(actual, owner, roots, instance),
        )]);
        if !Self::bind_timing_results(
            callee,
            operation,
            regions,
            roots,
            owner,
            instance,
            &mut bindings,
        ) {
            return None;
        }
        let mut classical = operation.operands.iter().skip(1);
        let arguments = callee
            .parameters
            .iter()
            .map(|parameter| {
                let value = parameter.value.as_ref()?;
                if value.has_region {
                    Some(None)
                } else {
                    let operand = classical.next()?;
                    Some(environment.get(operand).cloned().flatten())
                }
            })
            .collect::<Option<Vec<_>>>()?;
        classical.next().is_none().then_some((arguments, bindings))
    }

    fn allocated_function(
        &self,
        function: &wire::FunctionProto,
        arguments: &[Option<ExactValue>],
        dimension_arguments: &HashMap<Dimension, u64>,
    ) -> Result<Option<u128>, ExactResourceError> {
        if function.external || arguments.len() != function.parameters.len() {
            return Ok(None);
        }
        let mut environment = function
            .parameters
            .iter()
            .zip(arguments)
            .map(|(parameter, value)| {
                parameter
                    .value
                    .as_ref()
                    .map(|definition| (definition.id, value.clone()))
                    .ok_or_else(|| {
                        ExactResourceError::Missing(format!(
                            "parameter value in {:?}",
                            function.symbol
                        ))
                    })
            })
            .collect::<Result<HashMap<_, _>, _>>()?;
        let dimensions = exact_dimension_bindings(function, arguments, dimension_arguments);
        let body = function.body.as_ref().ok_or_else(|| {
            ExactResourceError::Missing(format!("body for {:?}", function.symbol))
        })?;
        self.allocated_block(
            body,
            &mut environment,
            ExactFunctionContext {
                owner: &function.symbol,
                dimensions: &dimensions,
            },
        )
    }

    fn allocated_block(
        &self,
        block: &wire::BlockProto,
        environment: &mut HashMap<u64, Option<ExactValue>>,
        context: ExactFunctionContext<'_>,
    ) -> Result<Option<u128>, ExactResourceError> {
        self.charge_block(block);
        let mut allocated = Some(0_u128);
        for operation in &block.operations {
            if bind_structural_exact_facts(operation, environment)? {
                continue;
            }
            match operation.opcode.as_str() {
                "qcore.call" => {
                    merge_optional_count(
                        &mut allocated,
                        self.allocated_call(operation, environment, context)?,
                    )?;
                }
                "qcore.if" => {
                    merge_optional_count(
                        &mut allocated,
                        self.allocated_if(operation, environment, context)?,
                    )?;
                }
                "qcore.for" => {
                    merge_optional_count(
                        &mut allocated,
                        self.allocated_for(operation, environment, context)?,
                    )?;
                }
                "quantum.alloc" | "quantum.prepare" => {
                    merge_optional_count(
                        &mut allocated,
                        self.allocation_result_width(operation, context),
                    )?;
                    bind_results(operation, environment, None);
                }
                opcode if opcode.starts_with("classical.") => {
                    let value = evaluate_exact_classical(operation, environment);
                    bind_results(operation, environment, value.as_ref());
                }
                _ => bind_results(operation, environment, None),
            }
        }
        Ok(allocated)
    }

    fn allocated_call(
        &self,
        operation: &wire::OperationProto,
        environment: &mut HashMap<u64, Option<ExactValue>>,
        context: ExactFunctionContext<'_>,
    ) -> Result<Option<u128>, ExactResourceError> {
        let callee_name = string_attribute(operation, "callee")
            .ok_or_else(|| ExactResourceError::Missing("callee".to_owned()))?;
        let callee = self
            .functions
            .get(callee_name)
            .copied()
            .ok_or_else(|| ExactResourceError::Missing(format!("callable {callee_name:?}")))?;
        let arguments = operation
            .operands
            .iter()
            .map(|operand| environment.get(operand).cloned().flatten())
            .collect::<Vec<_>>();
        let dimensions = self.call_dimensions(operation, callee, context)?;
        let allocated = self
            .allocated_function(callee, &arguments, &dimensions)
            .unwrap_or_default();
        if let Ok(summary) = self.analyze_function(callee, &arguments, &dimensions) {
            bind_exact_results(operation, environment, &summary.values)?;
        } else {
            bind_results(operation, environment, None);
        }
        Ok(allocated)
    }

    fn allocated_if(
        &self,
        operation: &wire::OperationProto,
        environment: &mut HashMap<u64, Option<ExactValue>>,
        context: ExactFunctionContext<'_>,
    ) -> Result<Option<u128>, ExactResourceError> {
        let captures = operation.operands.get(1..).unwrap_or_default();
        let mut allocated = Some(0_u128);
        for region in &operation.regions {
            let branch = region
                .blocks
                .first()
                .ok_or_else(|| ExactResourceError::Missing("qcore.if branch".to_owned()))?;
            let mut local = bind_region(branch, captures, environment, RegionFacts::Captured)?;
            merge_optional_count(
                &mut allocated,
                self.allocated_block(branch, &mut local, context)?,
            )?;
        }
        let condition = operation
            .operands
            .first()
            .and_then(|operand| environment.get(operand))
            .and_then(Option::as_ref)
            .and_then(ExactValue::scalar)
            .map(static_truth)
            .ok_or_else(|| {
                ExactResourceError::Unsupported("dynamic qcore.if condition".to_owned())
            })?;
        let branch = operation
            .regions
            .get(usize::from(!condition))
            .and_then(|region| region.blocks.first())
            .ok_or_else(|| ExactResourceError::Missing("qcore.if branch".to_owned()))?;
        let mut local = bind_region(branch, captures, environment, RegionFacts::Captured)?;
        if let Ok(summary) = self.analyze_block(branch, &mut local, context) {
            bind_exact_results(operation, environment, &summary.values)?;
        } else {
            bind_results(operation, environment, None);
        }
        Ok(allocated)
    }

    fn allocated_for(
        &self,
        operation: &wire::OperationProto,
        environment: &mut HashMap<u64, Option<ExactValue>>,
        context: ExactFunctionContext<'_>,
    ) -> Result<Option<u128>, ExactResourceError> {
        let body = operation
            .regions
            .first()
            .and_then(|region| region.blocks.first())
            .ok_or_else(|| ExactResourceError::Missing("qcore.for body".to_owned()))?;
        let carries = operation.operands.get(3..).unwrap_or_default();
        let invariants = loop_carry_invariants(body, carries.len());
        let mut local = bind_region(
            body,
            carries,
            environment,
            RegionFacts::LoopBody(&invariants),
        )?;
        let mut allocated = self.allocated_block(body, &mut local, context)?;
        if let Some(otherwise) = operation
            .regions
            .get(1)
            .and_then(|region| region.blocks.first())
        {
            let mut local = bind_region(otherwise, carries, environment, RegionFacts::Captured)?;
            merge_optional_count(
                &mut allocated,
                self.allocated_block(otherwise, &mut local, context)?,
            )?;
        }
        if let Ok(summary) = self.analyze_for(operation, environment, context) {
            bind_exact_results(operation, environment, &summary.values)?;
        } else {
            bind_results(operation, environment, None);
        }
        Ok(allocated)
    }

    fn allocation_result_width(
        &self,
        operation: &wire::OperationProto,
        context: ExactFunctionContext<'_>,
    ) -> Option<u128> {
        let types = self.operations.value_types(context.owner)?;
        operation.results.iter().find_map(|result| {
            let type_id = types.get(&result.id)?;
            coherent_width(self.types, *type_id, context.dimensions).map(u128::from)
        })
    }

    fn event_width(
        &self,
        owner: &str,
        operation: &wire::OperationProto,
        dimensions: &ExactDimensionBindings,
    ) -> Option<u128> {
        let operand = operation.operands.first()?;
        let type_id = self.operations.value_types(owner)?.get(operand)?;
        coherent_width(self.types, *type_id, dimensions).map(u128::from)
    }

    fn analyze_function(
        &self,
        function: &wire::FunctionProto,
        arguments: &[Option<ExactValue>],
        dimension_arguments: &HashMap<Dimension, u64>,
    ) -> Result<BlockSummary, ExactResourceError> {
        let mut dimensions = dimension_arguments
            .iter()
            .map(|(dimension, value)| (dimension.clone(), *value))
            .collect::<Vec<_>>();
        dimensions.sort_unstable_by(|(left, _), (right, _)| compare_dimensions(left, right));
        let key = ExactAnalysisKey {
            symbol: function.symbol.clone(),
            arguments: arguments.into(),
            dimensions: dimensions.into(),
        };
        if let Some(summary) = self.analysis_cache.borrow().get(&key) {
            return Ok(summary.clone());
        }
        let summary = self.analyze_function_uncached(function, arguments, dimension_arguments)?;
        self.analysis_cache
            .borrow_mut()
            .insert(key, summary.clone());
        Ok(summary)
    }

    fn analyze_function_uncached(
        &self,
        function: &wire::FunctionProto,
        arguments: &[Option<ExactValue>],
        dimension_arguments: &HashMap<Dimension, u64>,
    ) -> Result<BlockSummary, ExactResourceError> {
        if function.external {
            return Err(ExactResourceError::Unsupported(format!(
                "external callable {:?}",
                function.symbol
            )));
        }
        if arguments.len() != function.parameters.len() {
            return Err(ExactResourceError::Missing(format!(
                "argument ABI for {:?}",
                function.symbol
            )));
        }
        let mut environment = function
            .parameters
            .iter()
            .zip(arguments)
            .map(|(parameter, value)| {
                parameter
                    .value
                    .as_ref()
                    .map(|definition| (definition.id, value.clone()))
                    .ok_or_else(|| {
                        ExactResourceError::Missing(format!(
                            "parameter value in {:?}",
                            function.symbol
                        ))
                    })
            })
            .collect::<Result<HashMap<_, _>, _>>()?;
        let dimensions = exact_dimension_bindings(function, arguments, dimension_arguments);
        let body = function.body.as_ref().ok_or_else(|| {
            ExactResourceError::Missing(format!("body for {:?}", function.symbol))
        })?;
        let context = ExactFunctionContext {
            owner: &function.symbol,
            dimensions: &dimensions,
        };
        let summary = self.analyze_block(body, &mut environment, context)?;
        if summary.exit != TerminatorKind::Return {
            return Err(ExactResourceError::Missing(format!(
                "return terminator in {:?}",
                function.symbol
            )));
        }
        Ok(summary)
    }

    fn analyze_block(
        &self,
        block: &wire::BlockProto,
        environment: &mut HashMap<u64, Option<ExactValue>>,
        context: ExactFunctionContext<'_>,
    ) -> Result<BlockSummary, ExactResourceError> {
        self.charge_block(block);
        let mut counts = Counts::default();
        for operation in &block.operations {
            if bind_structural_exact_facts(operation, environment)? {
                continue;
            }
            match operation.opcode.as_str() {
                "qcore.call" => {
                    let summary = self.analyze_ordinary_call(operation, environment, context)?;
                    counts.then(&summary.counts)?;
                    bind_exact_results(operation, environment, &summary.values)?;
                }
                "qcore.if" => {
                    let summary = self.analyze_if(operation, environment, context)?;
                    counts.then(&summary.counts)?;
                    bind_exact_results(operation, environment, &summary.values)?;
                }
                "qcore.for" => {
                    let summary = self.analyze_for(operation, environment, context)?;
                    counts.then(&summary.counts)?;
                    bind_exact_results(operation, environment, &summary.values)?;
                }
                "qcore.adjoint_call" => {
                    counts.then(&self.analyze_adjoint_call(operation, environment)?)?;
                    bind_results(operation, environment, None);
                }
                "qcore.control_call" | "qcore.select_call" => {
                    counts.then(&self.analyze_coherent_call(operation, environment)?)?;
                    bind_results(operation, environment, None);
                }
                opcode @ ("qcore.phase_by" | "qcore.xor_by" | "qcore.permute_by") => {
                    let event_width =
                        self.event_width(context.owner, operation, context.dimensions);
                    counts.add(
                        opcode,
                        opcode,
                        event_width,
                        is_cleanup_operation(operation),
                        1,
                    )?;
                    bind_results(operation, environment, None);
                }
                "qcore.adjoint" => {
                    counts.then(&self.analyze_adjoint_operation(
                        operation,
                        environment,
                        context,
                    )?)?;
                    bind_results(operation, environment, None);
                }
                opcode if opcode.starts_with("quantum.") => {
                    let event_width =
                        self.event_width(context.owner, operation, context.dimensions);
                    let count = exact_quantum_multiplicity(operation, opcode, event_width)?;
                    counts.add(
                        opcode,
                        opcode,
                        event_width,
                        is_cleanup_operation(operation),
                        count,
                    )?;
                    bind_results(operation, environment, None);
                }
                opcode if opcode.starts_with("classical.") => {
                    let value = evaluate_exact_classical(operation, environment);
                    bind_results(operation, environment, value.as_ref());
                }
                _ => bind_results(operation, environment, None),
            }
        }
        let terminator = block
            .terminator
            .as_ref()
            .ok_or_else(|| ExactResourceError::Missing("block terminator".to_owned()))?;
        let values = terminator
            .operands
            .iter()
            .map(|operand| environment.get(operand).cloned().flatten())
            .collect();
        Ok(BlockSummary {
            counts,
            exit: exact_terminator_kind(terminator)?,
            values,
        })
    }

    fn analyze_if(
        &self,
        operation: &wire::OperationProto,
        environment: &HashMap<u64, Option<ExactValue>>,
        context: ExactFunctionContext<'_>,
    ) -> Result<BlockSummary, ExactResourceError> {
        let condition = operation
            .operands
            .first()
            .and_then(|operand| environment.get(operand))
            .and_then(Option::as_ref)
            .and_then(ExactValue::scalar)
            .map(static_truth)
            .ok_or_else(|| {
                ExactResourceError::Unsupported("dynamic qcore.if condition".to_owned())
            })?;
        let branch = operation
            .regions
            .get(usize::from(!condition))
            .and_then(|region| region.blocks.first())
            .ok_or_else(|| ExactResourceError::Missing("qcore.if branch".to_owned()))?;
        let captures = operation.operands.get(1..).unwrap_or_default();
        let mut local = bind_region(branch, captures, environment, RegionFacts::Captured)?;
        let mut summary = self.analyze_block(branch, &mut local, context)?;
        if summary.exit != TerminatorKind::Yield {
            return Err(ExactResourceError::Unsupported(format!(
                "qcore.if branch terminator {:?}",
                summary.exit.as_str()
            )));
        }
        if string_attribute(operation, "body_mode") == Some("adjoint") {
            summary.counts = summary.counts.cleanup_adjoint();
            summary.values = captures
                .iter()
                .map(|operand| environment.get(operand).cloned().flatten())
                .collect();
        }
        Ok(summary)
    }

    fn analyze_ordinary_call(
        &self,
        operation: &wire::OperationProto,
        environment: &HashMap<u64, Option<ExactValue>>,
        context: ExactFunctionContext<'_>,
    ) -> Result<BlockSummary, ExactResourceError> {
        let callee_name = string_attribute(operation, "callee")
            .ok_or_else(|| ExactResourceError::Missing("callee".to_owned()))?;
        let callee = self
            .functions
            .get(callee_name)
            .copied()
            .ok_or_else(|| ExactResourceError::Missing(format!("callable {callee_name:?}")))?;
        let arguments = operation
            .operands
            .iter()
            .map(|operand| environment.get(operand).cloned().flatten())
            .collect::<Vec<_>>();
        let dimension_arguments = self.call_dimensions(operation, callee, context)?;
        self.analyze_function(callee, &arguments, &dimension_arguments)
    }

    fn call_dimensions(
        &self,
        operation: &wire::OperationProto,
        callee: &wire::FunctionProto,
        context: ExactFunctionContext<'_>,
    ) -> Result<HashMap<Dimension, u64>, ExactResourceError> {
        let declared_types = self.operations.value_types(&callee.symbol).ok_or_else(|| {
            ExactResourceError::Missing(format!("type roots for {:?}", callee.symbol))
        })?;
        let expected = callee
            .parameters
            .iter()
            .map(|parameter| {
                parameter
                    .value
                    .as_ref()
                    .and_then(|value| declared_types.get(&value.id))
                    .copied()
                    .ok_or_else(|| {
                        ExactResourceError::Missing(format!(
                            "parameter type in {:?}",
                            callee.symbol
                        ))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let local_types = self.operations.value_types(context.owner).ok_or_else(|| {
            ExactResourceError::Missing(format!("type roots for {:?}", context.owner))
        })?;
        let actual = operation
            .operands
            .iter()
            .map(|operand| {
                local_types.get(operand).copied().ok_or_else(|| {
                    ExactResourceError::Missing(format!("operand type in {:?}", context.owner))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let inferred = crate::abi::infer_call_dimensions(self.types, &expected, &actual)
            .ok_or_else(|| ExactResourceError::Missing("verified call dimensions".to_owned()))?;
        Ok(inferred
            .into_iter()
            .filter_map(|(formal, actual)| {
                resolve_exact_dimension(&actual, context.dimensions).map(|value| (formal, value))
            })
            .collect())
    }

    fn analyze_for(
        &self,
        operation: &wire::OperationProto,
        environment: &HashMap<u64, Option<ExactValue>>,
        context: ExactFunctionContext<'_>,
    ) -> Result<BlockSummary, ExactResourceError> {
        let bounds = operation
            .operands
            .iter()
            .take(3)
            .map(|operand| {
                environment
                    .get(operand)
                    .cloned()
                    .flatten()
                    .and_then(|value| value.scalar())
                    .and_then(static_int)
                    .ok_or_else(|| {
                        ExactResourceError::Unsupported("non-static qcore.for bounds".to_owned())
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let [lower, upper, stride] = bounds.as_slice() else {
            return Err(ExactResourceError::Missing("qcore.for bounds".to_owned()));
        };
        let trip_count = static_range_len(*lower, *upper, *stride)
            .ok_or_else(|| ExactResourceError::Missing("non-zero qcore.for step".to_owned()))?;
        let body = operation
            .regions
            .first()
            .and_then(|region| region.blocks.first())
            .ok_or_else(|| ExactResourceError::Missing("qcore.for body".to_owned()))?;
        let carry_operands = operation.operands.get(3..).unwrap_or_default();
        let schedule = StaticLoop {
            lower: *lower,
            stride: *stride,
            trip_count,
            reverse: bool_attribute(operation, "reverse").unwrap_or(false),
        };
        let mut summary = if trip_count <= MAX_EXACT_LOOP_INTERPRETATIONS {
            self.analyze_for_iterations(
                body,
                operation
                    .regions
                    .get(1)
                    .and_then(|region| region.blocks.first()),
                carry_operands,
                environment,
                schedule,
                context,
            )?
        } else {
            self.analyze_repeated_for(
                operation,
                body,
                carry_operands,
                environment,
                trip_count,
                context,
            )?
        };
        if string_attribute(operation, "body_mode") == Some("adjoint") {
            summary.counts = summary.counts.cleanup_adjoint();
        }
        Ok(summary)
    }

    fn analyze_for_iterations(
        &self,
        body: &wire::BlockProto,
        otherwise: Option<&wire::BlockProto>,
        carry_operands: &[u64],
        environment: &HashMap<u64, Option<ExactValue>>,
        schedule: StaticLoop,
        context: ExactFunctionContext<'_>,
    ) -> Result<BlockSummary, ExactResourceError> {
        let mut carries = carry_operands
            .iter()
            .map(|operand| environment.get(operand).cloned().flatten())
            .collect::<Vec<_>>();
        let mut counts = Counts::default();
        let mut broke = false;
        for position in 0..schedule.trip_count {
            let logical_position = if schedule.reverse {
                schedule.trip_count - position - 1
            } else {
                position
            };
            let offset = i128::try_from(logical_position)
                .ok()
                .and_then(|position| position.checked_mul(i128::from(schedule.stride)))
                .ok_or(ExactResourceError::Overflow)?;
            let induction = i128::from(schedule.lower)
                .checked_add(offset)
                .and_then(|value| i64::try_from(value).ok())
                .ok_or(ExactResourceError::Overflow)?;
            let arguments = std::iter::once(Some(ExactValue::Scalar(StaticScalar::Int(induction))))
                .chain(carries.iter().cloned())
                .collect::<Vec<_>>();
            let mut local = bind_region_values(body, arguments)?;
            let summary = self.analyze_block(body, &mut local, context)?;
            counts.then(&summary.counts)?;
            match summary.exit {
                TerminatorKind::Yield | TerminatorKind::Continue => carries = summary.values,
                TerminatorKind::Break => {
                    carries = summary.values;
                    broke = true;
                    break;
                }
                exit => {
                    return Err(ExactResourceError::Unsupported(format!(
                        "qcore.for body terminator {:?}",
                        exit.as_str()
                    )));
                }
            }
        }
        if !broke && let Some(otherwise) = otherwise {
            let mut local = bind_region_values(otherwise, carries)?;
            let summary = self.analyze_block(otherwise, &mut local, context)?;
            if summary.exit != TerminatorKind::Yield {
                return Err(ExactResourceError::Unsupported(format!(
                    "qcore.for else terminator {:?}",
                    summary.exit.as_str()
                )));
            }
            counts.then(&summary.counts)?;
            carries = summary.values;
        }
        Ok(BlockSummary {
            counts,
            exit: TerminatorKind::Yield,
            values: carries,
        })
    }

    fn analyze_repeated_for(
        &self,
        operation: &wire::OperationProto,
        body: &wire::BlockProto,
        carry_operands: &[u64],
        environment: &HashMap<u64, Option<ExactValue>>,
        trip_count: u128,
        context: ExactFunctionContext<'_>,
    ) -> Result<BlockSummary, ExactResourceError> {
        let invariants = loop_carry_invariants(body, carry_operands.len());
        let mut counts = if trip_count == 0 {
            Counts::default()
        } else {
            let mut local = bind_region(
                body,
                carry_operands,
                environment,
                RegionFacts::LoopBody(&invariants),
            )?;
            let summary = self.analyze_block(body, &mut local, context)?;
            if !matches!(
                summary.exit,
                TerminatorKind::Yield | TerminatorKind::Continue
            ) {
                return Err(ExactResourceError::Unsupported(format!(
                    "qcore.for body terminator {:?}",
                    summary.exit.as_str()
                )));
            }
            summary.counts.repeat(trip_count)?
        };
        let mut carries =
            loop_carry_values(carry_operands, environment, &invariants).collect::<Vec<_>>();
        if let Some(otherwise) = operation
            .regions
            .get(1)
            .and_then(|region| region.blocks.first())
        {
            let mut local = bind_region(
                otherwise,
                carry_operands,
                environment,
                if trip_count == 0 {
                    RegionFacts::Captured
                } else {
                    RegionFacts::LoopExit(&invariants)
                },
            )?;
            let summary = self.analyze_block(otherwise, &mut local, context)?;
            if summary.exit != TerminatorKind::Yield {
                return Err(ExactResourceError::Unsupported(format!(
                    "qcore.for else terminator {:?}",
                    summary.exit.as_str()
                )));
            }
            counts.then(&summary.counts)?;
            carries = summary.values;
        }
        Ok(BlockSummary {
            counts,
            exit: TerminatorKind::Yield,
            values: carries,
        })
    }

    fn analyze_adjoint_call(
        &self,
        operation: &wire::OperationProto,
        environment: &HashMap<u64, Option<ExactValue>>,
    ) -> Result<Counts, ExactResourceError> {
        let callee_name = string_attribute(operation, "callee")
            .ok_or_else(|| ExactResourceError::Missing("adjoint callee".to_owned()))?;
        let callee = self.functions.get(callee_name).copied().ok_or_else(|| {
            ExactResourceError::Missing(format!("adjoint callable {callee_name:?}"))
        })?;
        let mut classical_operands = operation.operands.iter().skip(1);
        let arguments = callee
            .parameters
            .iter()
            .map(|parameter| {
                let value = parameter.value.as_ref().ok_or_else(|| {
                    ExactResourceError::Missing(format!("parameter value in {:?}", callee.symbol))
                })?;
                if value.has_region {
                    Ok(None)
                } else {
                    let operand = classical_operands.next().ok_or_else(|| {
                        ExactResourceError::Missing("adjoint classical argument".to_owned())
                    })?;
                    Ok(environment.get(operand).cloned().flatten())
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        if classical_operands.next().is_some() {
            return Err(ExactResourceError::Missing(
                "adjoint classical argument ABI".to_owned(),
            ));
        }
        let direct = self
            .analyze_function(callee, &arguments, &HashMap::new())?
            .counts;
        let mut opcodes = BTreeMap::new();
        if direct.work != 0 {
            opcodes.insert("qcore.adjoint".to_owned(), direct.work);
        }
        Ok(Counts {
            work: direct.work,
            opcodes,
            families: direct.families,
            measurements: Some(0),
            resets: Some(0),
            cleanup_work: if string_attribute(operation, "proof") == Some("modular-adjoint") {
                direct.work
            } else {
                0
            },
        })
    }

    fn analyze_adjoint_operation(
        &self,
        operation: &wire::OperationProto,
        environment: &HashMap<u64, Option<ExactValue>>,
        context: ExactFunctionContext<'_>,
    ) -> Result<Counts, ExactResourceError> {
        let original = string_attribute(operation, "original")
            .ok_or_else(|| ExactResourceError::Missing("adjoint original opcode".to_owned()))?;
        if matches!(original, "qcore.control_call" | "qcore.select_call") {
            return self.analyze_coherent_adjoint(operation, environment);
        }
        let mut counts = Counts::default();
        let event_width = self.event_width(context.owner, operation, context.dimensions);
        let count = exact_quantum_multiplicity(operation, original, event_width)?;
        counts.add(
            "qcore.adjoint",
            original,
            event_width,
            is_cleanup_operation(operation),
            count,
        )?;
        Ok(counts)
    }

    fn analyze_coherent_call(
        &self,
        operation: &wire::OperationProto,
        environment: &HashMap<u64, Option<ExactValue>>,
    ) -> Result<Counts, ExactResourceError> {
        let callee_names = if operation.opcode == "qcore.control_call" {
            vec![
                string_attribute(operation, "callee")
                    .ok_or_else(|| ExactResourceError::Missing("control callee".to_owned()))?,
            ]
        } else {
            string_tuple_attribute(operation, "callees")
                .ok_or_else(|| ExactResourceError::Missing("select callees".to_owned()))?
        };
        let first = callee_names
            .first()
            .and_then(|name| self.functions.get(name).copied())
            .ok_or_else(|| ExactResourceError::Missing("coherent callable".to_owned()))?;
        let offset = operation
            .operands
            .len()
            .checked_sub(first.parameters.len())
            .ok_or_else(|| ExactResourceError::Missing("coherent call ABI".to_owned()))?;
        let arguments = operation
            .operands
            .get(offset..)
            .ok_or_else(|| ExactResourceError::Missing("coherent arguments".to_owned()))?
            .iter()
            .map(|operand| environment.get(operand).cloned().flatten())
            .collect::<Vec<_>>();
        let mut result = Counts::default();
        for callee_name in callee_names {
            let callee = self.functions.get(callee_name).copied().ok_or_else(|| {
                ExactResourceError::Missing(format!("coherent callable {callee_name:?}"))
            })?;
            result.then(
                &self
                    .analyze_function(callee, &arguments, &HashMap::new())?
                    .counts,
            )?;
        }
        result.families = None;
        Ok(result)
    }

    fn analyze_coherent_adjoint(
        &self,
        operation: &wire::OperationProto,
        environment: &HashMap<u64, Option<ExactValue>>,
    ) -> Result<Counts, ExactResourceError> {
        let original = string_attribute(operation, "original")
            .ok_or_else(|| ExactResourceError::Missing("coherent adjoint opcode".to_owned()))?;
        let callee_names = if original == "qcore.control_call" {
            vec![
                mapping_string_attribute(operation, "original_attributes", "callee").ok_or_else(
                    || ExactResourceError::Missing("control adjoint callee".to_owned()),
                )?,
            ]
        } else {
            mapping_string_tuple_attribute(operation, "original_attributes", "callees")
                .ok_or_else(|| ExactResourceError::Missing("select adjoint callees".to_owned()))?
        };
        let first = callee_names
            .first()
            .and_then(|name| self.functions.get(name).copied())
            .ok_or_else(|| ExactResourceError::Missing("coherent adjoint callable".to_owned()))?;
        let supplied = operation.operands.get(1..).unwrap_or_default();
        let component_count = if original == "qcore.control_call" {
            supplied
                .len()
                .checked_add(2)
                .and_then(|count| count.checked_sub(first.parameters.len()))
                .and_then(|offset| offset.checked_sub(1))
                .filter(|count| *count <= 1)
                .ok_or_else(|| {
                    ExactResourceError::Missing("control adjoint argument ABI".to_owned())
                })?
        } else {
            0
        };
        let mut classical = supplied
            .get(component_count..)
            .ok_or_else(|| ExactResourceError::Missing("coherent adjoint arguments".to_owned()))?
            .iter();
        let arguments = first
            .parameters
            .iter()
            .map(|parameter| {
                let value = parameter.value.as_ref().ok_or_else(|| {
                    ExactResourceError::Missing("coherent adjoint parameter".to_owned())
                })?;
                if value.has_region {
                    Ok(None)
                } else {
                    let operand = classical.next().ok_or_else(|| {
                        ExactResourceError::Missing("coherent adjoint argument".to_owned())
                    })?;
                    Ok(environment.get(operand).cloned().flatten())
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        if classical.next().is_some() {
            return Err(ExactResourceError::Missing(
                "coherent adjoint argument ABI".to_owned(),
            ));
        }
        let mut direct = Counts::default();
        for callee_name in callee_names {
            let callee = self.functions.get(callee_name).copied().ok_or_else(|| {
                ExactResourceError::Missing(format!("coherent callable {callee_name:?}"))
            })?;
            direct.then(
                &self
                    .analyze_function(callee, &arguments, &HashMap::new())?
                    .counts,
            )?;
        }
        let mut opcodes = BTreeMap::new();
        if direct.work != 0 {
            opcodes.insert("qcore.adjoint".to_owned(), direct.work);
        }
        Ok(Counts {
            work: direct.work,
            opcodes,
            families: None,
            measurements: Some(0),
            resets: Some(0),
            cleanup_work: if string_attribute(operation, "proof") == Some("modular-adjoint") {
                direct.work
            } else {
                0
            },
        })
    }
}

pub(crate) fn is_measurement(opcode: &str) -> bool {
    matches!(
        opcode,
        "quantum.measure"
            | "quantum.measure_reset"
            | "quantum.measure_x"
            | "quantum.measure_y"
            | "quantum.measure_z"
    )
}

pub(crate) fn is_reset(opcode: &str) -> bool {
    matches!(opcode, "quantum.reset" | "quantum.measure_reset")
}

pub(crate) fn is_cleanup_operation(operation: &wire::OperationProto) -> bool {
    matches!(
        operation.opcode.as_str(),
        "quantum.release" | "quantum.unprepare"
    ) || (operation.opcode == "qcore.adjoint"
        && string_attribute(operation, "proof") == Some("modular-adjoint"))
}

fn event_count(opcode: &str, width: Option<u128>) -> Option<u128> {
    if matches!(
        opcode,
        "quantum.measure_x" | "quantum.measure_y" | "quantum.measure_z"
    ) {
        Some(1)
    } else {
        width
    }
}

fn checked_active_width(active: &HashMap<u64, u128>) -> Result<u128, ExactResourceError> {
    active.values().try_fold(0_u128, |total, width| {
        total
            .checked_add(*width)
            .ok_or(ExactResourceError::Overflow)
    })
}

pub(crate) fn function_value_roots(function: &wire::FunctionProto) -> HashMap<u64, u64> {
    function_value_regions(function)
        .into_iter()
        .map(|(value, region)| (value, region.root))
        .collect()
}

pub(crate) fn function_value_regions(
    function: &wire::FunctionProto,
) -> HashMap<u64, wire::RegionRefProto> {
    let mut regions = HashMap::new();
    for parameter in &function.parameters {
        if let Some(value) = &parameter.value {
            record_value_region(&mut regions, value);
        }
    }
    let mut blocks = function.body.iter().collect::<Vec<_>>();
    while let Some(block) = blocks.pop() {
        for argument in &block.arguments {
            record_value_region(&mut regions, argument);
        }
        for operation in &block.operations {
            for result in &operation.results {
                record_value_region(&mut regions, result);
            }
            for region in &operation.regions {
                blocks.extend(&region.blocks);
            }
        }
    }
    regions
}

fn record_value_region(regions: &mut HashMap<u64, wire::RegionRefProto>, value: &wire::ValueProto) {
    if let Some(region) = value.region.as_ref().filter(|_| value.has_region) {
        regions.insert(value.id, region.clone());
    }
}

fn take_timing_instance(next: &mut u64) -> Result<u64, ExactResourceError> {
    let instance = *next;
    *next = next.checked_add(1).ok_or(ExactResourceError::Overflow)?;
    Ok(instance)
}

fn exact_timing_root(
    region: &wire::RegionRefProto,
    owner: &str,
    roots: &HashMap<u64, ExactTimingRoot>,
    instance: u64,
) -> ExactTimingRoot {
    if let Some(root) = roots.get(&region.root) {
        let mut mapped = root.clone();
        mapped
            .projections
            .extend(region.projections.iter().cloned());
        mapped
    } else {
        ExactTimingRoot {
            scope: owner.to_owned(),
            root: region.root,
            projections: region.projections.clone(),
            instance,
        }
    }
}

fn insert_timing_root(
    roots: &mut HashMap<u64, ExactTimingRoot>,
    formal: u64,
    actual: ExactTimingRoot,
) -> bool {
    if let Some(existing) = roots.get(&formal) {
        existing == &actual
    } else {
        roots.insert(formal, actual);
        true
    }
}

fn exact_static_loop(
    operation: &wire::OperationProto,
    environment: &HashMap<u64, Option<ExactValue>>,
) -> Result<StaticLoop, ExactResourceError> {
    let bounds = operation
        .operands
        .iter()
        .take(3)
        .map(|operand| {
            environment
                .get(operand)
                .cloned()
                .flatten()
                .and_then(|value| value.scalar())
                .and_then(static_int)
                .ok_or_else(|| {
                    ExactResourceError::Unsupported("non-static qcore.for bounds".to_owned())
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let [lower, upper, stride] = bounds.as_slice() else {
        return Err(ExactResourceError::Missing("qcore.for bounds".to_owned()));
    };
    let trip_count = static_range_len(*lower, *upper, *stride)
        .ok_or_else(|| ExactResourceError::Missing("non-zero qcore.for step".to_owned()))?;
    Ok(StaticLoop {
        lower: *lower,
        stride: *stride,
        trip_count,
        reverse: bool_attribute(operation, "reverse").unwrap_or(false),
    })
}

fn exact_loop_induction(schedule: StaticLoop, position: u128) -> Result<i64, ExactResourceError> {
    let logical_position = if schedule.reverse {
        schedule.trip_count - position - 1
    } else {
        position
    };
    let offset = i128::try_from(logical_position)
        .ok()
        .and_then(|value| value.checked_mul(i128::from(schedule.stride)))
        .ok_or(ExactResourceError::Overflow)?;
    i128::from(schedule.lower)
        .checked_add(offset)
        .and_then(|value| i64::try_from(value).ok())
        .ok_or(ExactResourceError::Overflow)
}

fn exact_timing_selectors(
    operation: &wire::OperationProto,
    effective: &str,
    environment: &HashMap<u64, Option<ExactValue>>,
) -> Result<Option<Box<[u64]>>, ExactResourceError> {
    if matches!(
        effective,
        "quantum.measure_x" | "quantum.measure_y" | "quantum.measure_z"
    ) {
        let selector = operation
            .operands
            .get(1)
            .and_then(|value| exact_selector(*value, environment))
            .unwrap_or(0);
        return Ok(Some(vec![selector].into_boxed_slice()));
    }
    let Some(gate) = primitive_gate_spec(effective) else {
        return Ok(None);
    };
    let names = gate.selectors();
    let keywords = timing_keyword_names(operation)?;
    let arguments = operation.operands.get(1..).unwrap_or_default();
    if keywords.len() > arguments.len() {
        return Err(ExactResourceError::Missing("gate selector ABI".to_owned()));
    }
    let positional_count = arguments.len() - keywords.len();
    if gate.broadcasts_when_unselected()
        && positional_count == usize::from(gate.has_angle())
        && !keywords.iter().any(|name| gate.selectors().contains(name))
    {
        return Ok(None);
    }
    let positional = &arguments[..positional_count];
    let keyword_values = keywords
        .iter()
        .zip(&arguments[positional_count..])
        .collect::<HashMap<_, _>>();
    names
        .iter()
        .enumerate()
        .map(|(index, name)| {
            let operand = keyword_values.get(name).copied().copied().or_else(|| {
                positional
                    .get(index + usize::from(gate.has_angle()))
                    .copied()
            });
            operand
                .and_then(|value| exact_selector(value, environment))
                .or_else(|| {
                    gate.selector_default(index)
                        .and_then(|value| u64::try_from(value).ok())
                })
                .ok_or_else(|| ExactResourceError::Unsupported(format!("gate selector {name:?}")))
        })
        .collect::<Result<Vec<_>, _>>()
        .map(|values| Some(values.into_boxed_slice()))
}

fn exact_selector(value: u64, environment: &HashMap<u64, Option<ExactValue>>) -> Option<u64> {
    environment
        .get(&value)
        .cloned()
        .flatten()
        .and_then(|value| value.scalar())
        .and_then(static_int)
        .and_then(|value| u64::try_from(value).ok())
}

fn timing_keyword_names(operation: &wire::OperationProto) -> Result<Vec<&str>, ExactResourceError> {
    if operation.opcode == "qcore.adjoint" {
        mapping_string_tuple_attribute(operation, "original_attributes", "keywords")
    } else {
        string_tuple_attribute(operation, "keywords")
    }
    .ok_or_else(|| ExactResourceError::Missing("gate keywords".to_owned()))
}

fn exact_quantum_multiplicity(
    operation: &wire::OperationProto,
    effective: &str,
    event_width: Option<u128>,
) -> Result<u128, ExactResourceError> {
    let Some(gate) = primitive_gate_spec(effective) else {
        return Ok(1);
    };
    if !gate.broadcasts_when_unselected() {
        return Ok(1);
    }
    let keywords = timing_keyword_names(operation)?;
    let arguments = operation.operands.get(1..).unwrap_or_default();
    if keywords.len() > arguments.len() {
        return Err(ExactResourceError::Missing("gate selector ABI".to_owned()));
    }
    let positional_count = arguments.len() - keywords.len();
    let selector_is_omitted = positional_count == usize::from(gate.has_angle())
        && !keywords.iter().any(|name| gate.selectors().contains(name));
    if selector_is_omitted {
        event_width.ok_or_else(|| {
            ExactResourceError::Unsupported(format!(
                "broadcast width for primitive gate {effective:?}"
            ))
        })
    } else {
        Ok(1)
    }
}

fn depth_families(opcode: &str) -> Vec<&'static str> {
    let mut families = logical_gate_families(opcode)
        .iter()
        .copied()
        .filter(|family| {
            matches!(
                *family,
                "gates"
                    | "clifford_gates"
                    | "rotation_gates"
                    | "t_gates"
                    | "toffoli_gates"
                    | "non_clifford_gates"
            )
        })
        .collect::<Vec<_>>();
    if matches!(
        opcode,
        "quantum.measure"
            | "quantum.measure_reset"
            | "quantum.measure_x"
            | "quantum.measure_y"
            | "quantum.measure_z"
    ) {
        families.push("measurements");
    }
    if matches!(opcode, "quantum.reset" | "quantum.measure_reset") {
        families.push("resets");
    }
    families.sort_unstable();
    families.dedup();
    families
}

const DEPTH_FAMILIES: &[&str] = &[
    "span",
    "gates",
    "clifford_gates",
    "rotation_gates",
    "t_gates",
    "toffoli_gates",
    "non_clifford_gates",
    "measurements",
    "resets",
];

fn schedule_exact_depths(
    steps: &[ExactTimingStep],
) -> Result<BTreeMap<String, u128>, ExactResourceError> {
    let mut timelines = DEPTH_FAMILIES
        .iter()
        .map(|family| {
            (
                *family,
                HashMap::<ExactTimingRoot, ExactLaneTimeline>::new(),
            )
        })
        .collect::<HashMap<_, _>>();
    let mut maxima = DEPTH_FAMILIES
        .iter()
        .map(|family| ((*family).to_owned(), 0))
        .collect::<BTreeMap<_, _>>();
    for step in steps {
        schedule_exact_step(step, &mut timelines, &mut maxima)?;
    }
    Ok(maxima)
}

fn schedule_exact_step(
    step: &ExactTimingStep,
    timelines: &mut HashMap<&'static str, HashMap<ExactTimingRoot, ExactLaneTimeline>>,
    maxima: &mut BTreeMap<String, u128>,
) -> Result<(), ExactResourceError> {
    for footprint in &step.footprints {
        if let Some(selectors) = &footprint.selectors
            && (selectors
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len()
                != selectors.len()
                || footprint.width.is_some_and(|width| {
                    selectors
                        .iter()
                        .any(|selector| u128::from(*selector) >= width)
                }))
        {
            return Err(ExactResourceError::Unsupported(
                "invalid timing selector".to_owned(),
            ));
        }
    }
    for family in DEPTH_FAMILIES {
        let roots = timelines
            .get_mut(family)
            .ok_or_else(|| ExactResourceError::Missing("depth family".to_owned()))?;
        let mut start = 0;
        for footprint in &step.footprints {
            for (root, frontier) in roots.iter() {
                if !exact_timing_roots_overlap(root, &footprint.root) {
                    continue;
                }
                if exact_timing_projection_is_ambiguous(root, footprint, frontier) {
                    return Err(ExactResourceError::Unsupported(
                        "projected timing selector alias".to_owned(),
                    ));
                }
                start = start.max(frontier.barrier);
                if root == &footprint.root
                    && let Some(selectors) = &footprint.selectors
                {
                    start = selectors.iter().fold(start, |value, selector| {
                        value.max(*frontier.lanes.get(selector).unwrap_or(&frontier.barrier))
                    });
                } else if footprint.selectors.is_none() {
                    start = frontier.lanes.values().copied().fold(start, u128::max);
                }
            }
            roots.entry(footprint.root.clone()).or_default();
        }
        let active = *family == "span" || step.families.contains(family);
        let finish = start
            .checked_add(u128::from(active))
            .ok_or(ExactResourceError::Overflow)?;
        for footprint in &step.footprints {
            let frontier = roots
                .get_mut(&footprint.root)
                .ok_or_else(|| ExactResourceError::Missing("timing frontier".to_owned()))?;
            if let Some(selectors) = &footprint.selectors {
                for selector in selectors {
                    frontier.lanes.insert(*selector, finish);
                }
            } else {
                frontier.barrier = finish;
                frontier.lanes.clear();
            }
        }
        let maximum = maxima
            .get_mut(*family)
            .ok_or_else(|| ExactResourceError::Missing("depth maximum".to_owned()))?;
        *maximum = (*maximum).max(finish);
    }
    Ok(())
}

fn exact_timing_roots_overlap(left: &ExactTimingRoot, right: &ExactTimingRoot) -> bool {
    left.scope == right.scope
        && left.root == right.root
        && left.instance == right.instance
        && (left.projections.starts_with(&right.projections)
            || right.projections.starts_with(&left.projections))
}

fn exact_timing_projection_is_ambiguous(
    existing: &ExactTimingRoot,
    footprint: &ExactTimingFootprint,
    frontier: &ExactLaneTimeline,
) -> bool {
    if existing == &footprint.root {
        return false;
    }
    let existing_is_ancestor = footprint
        .root
        .projections
        .starts_with(&existing.projections);
    (!frontier.lanes.is_empty() && existing_is_ancestor)
        || (footprint.selectors.is_some() && !existing_is_ancestor)
}

fn coherent_width(
    arena: &TypeArena,
    type_id: TypeId,
    dimensions: &ExactDimensionBindings,
) -> Option<u64> {
    arena
        .coherent_width_with(type_id, |dimension| dimension_width(dimension, dimensions))
        .ok()
        .flatten()
}

fn dimension_width(dimension: &Dimension, bindings: &ExactDimensionBindings) -> Option<u64> {
    match dimension {
        Dimension::Fixed(width) => Some(*width),
        Dimension::Symbol { .. } => bindings.values.get(dimension).copied(),
    }
}

fn exact_dimension_bindings(
    function: &wire::FunctionProto,
    arguments: &[Option<ExactValue>],
    inherited: &HashMap<Dimension, u64>,
) -> ExactDimensionBindings {
    let mut values = inherited.clone();
    values.extend(
        function
            .parameters
            .iter()
            .zip(arguments)
            .filter_map(|(parameter, value)| {
                value
                    .as_ref()
                    .and_then(ExactValue::scalar)
                    .and_then(static_int)
                    .and_then(|value| u64::try_from(value).ok())
                    .filter(|value| *value != 0)
                    .map(|value| {
                        (
                            Dimension::Symbol {
                                scope: Arc::from(function.symbol.as_str()),
                                name: Arc::from(parameter.name.as_str()),
                            },
                            value,
                        )
                    })
            }),
    );
    ExactDimensionBindings { values }
}

fn resolve_exact_dimension(
    dimension: &Dimension,
    bindings: &ExactDimensionBindings,
) -> Option<u64> {
    dimension_width(dimension, bindings)
}

fn bind_structural_exact_facts(
    operation: &wire::OperationProto,
    environment: &mut HashMap<u64, Option<ExactValue>>,
) -> Result<bool, ExactResourceError> {
    match operation.opcode.as_str() {
        "qcore.constant" => {
            let value = attribute(operation, "value").and_then(|literal| {
                operation
                    .results
                    .first()
                    .and_then(|result| result.r#type.as_ref())
                    .and_then(|type_spec| exact_typed_literal(literal, type_spec))
            });
            bind_results(operation, environment, value.as_ref());
        }
        "qcore.stage_lift" | "qcore.reshape" => {
            let value = operation
                .operands
                .first()
                .and_then(|operand| environment.get(operand))
                .cloned()
                .flatten();
            bind_results(operation, environment, value.as_ref());
        }
        "tuple" | "array" | "qcore.tuple" => {
            let values = operation
                .operands
                .iter()
                .map(|operand| environment.get(operand).cloned().flatten())
                .collect::<Vec<_>>();
            let value = exact_structural_value(operation.results.first(), values);
            bind_results(operation, environment, Some(&value));
        }
        "qcore.unpack" => {
            let values = operation
                .operands
                .first()
                .and_then(|operand| environment.get(operand))
                .and_then(Option::as_ref)
                .and_then(exact_structural_components)
                .filter(|values| values.len() == operation.results.len())
                .unwrap_or_else(|| vec![None; operation.results.len()]);
            bind_exact_results(operation, environment, &values)?;
        }
        _ => return Ok(false),
    }
    Ok(true)
}

fn bind_results(
    operation: &wire::OperationProto,
    environment: &mut HashMap<u64, Option<ExactValue>>,
    value: Option<&ExactValue>,
) {
    for result in &operation.results {
        environment.insert(result.id, value.cloned());
    }
}

fn bind_exact_results(
    operation: &wire::OperationProto,
    environment: &mut HashMap<u64, Option<ExactValue>>,
    values: &[Option<ExactValue>],
) -> Result<(), ExactResourceError> {
    if operation.results.len() != values.len() {
        return Err(ExactResourceError::Missing(format!(
            "result ABI for {:?}",
            operation.opcode
        )));
    }
    for (result, value) in operation.results.iter().zip(values) {
        environment.insert(result.id, value.clone());
    }
    Ok(())
}

fn bind_region(
    block: &wire::BlockProto,
    operands: &[u64],
    environment: &HashMap<u64, Option<ExactValue>>,
    facts: RegionFacts<'_>,
) -> Result<HashMap<u64, Option<ExactValue>>, ExactResourceError> {
    let values = match facts {
        RegionFacts::Captured => operands
            .iter()
            .map(|operand| environment.get(operand).cloned().flatten())
            .collect::<Vec<_>>(),
        RegionFacts::LoopBody(invariants) => std::iter::once(None)
            .chain(loop_carry_values(operands, environment, invariants))
            .collect(),
        RegionFacts::LoopExit(invariants) => {
            loop_carry_values(operands, environment, invariants).collect()
        }
    };
    bind_region_values(block, values)
}

fn bind_region_values(
    block: &wire::BlockProto,
    values: Vec<Option<ExactValue>>,
) -> Result<HashMap<u64, Option<ExactValue>>, ExactResourceError> {
    if values.len() != block.arguments.len() {
        return Err(ExactResourceError::Missing(
            "structured region argument ABI".to_owned(),
        ));
    }
    Ok(block
        .arguments
        .iter()
        .zip(values)
        .map(|(argument, value)| (argument.id, value))
        .collect())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RegionFacts<'a> {
    Captured,
    LoopBody(&'a [bool]),
    LoopExit(&'a [bool]),
}

fn loop_carry_values<'a>(
    operands: &'a [u64],
    environment: &'a HashMap<u64, Option<ExactValue>>,
    invariants: &'a [bool],
) -> impl Iterator<Item = Option<ExactValue>> + 'a {
    operands.iter().zip(invariants).map(|(operand, invariant)| {
        invariant
            .then(|| environment.get(operand).cloned().flatten())
            .flatten()
    })
}

pub(crate) fn loop_carry_invariants(block: &wire::BlockProto, carry_count: usize) -> Vec<bool> {
    if block.arguments.len() != carry_count + 1 {
        return vec![false; carry_count];
    }
    let argument_aliases = std::iter::once(None)
        .chain((0..carry_count).map(Some))
        .collect::<Vec<_>>();
    let Some(exit_aliases) = block_exit_aliases(block, &argument_aliases) else {
        return vec![false; carry_count];
    };
    exit_aliases
        .into_iter()
        .enumerate()
        .map(|(index, alias)| alias == Some(index))
        .collect()
}

pub(crate) fn while_carry_invariants(operation: &wire::OperationProto) -> Vec<bool> {
    let count = operation.operands.len();
    let aliases = (0..count).map(Some).collect::<Vec<_>>();
    let mut invariants = vec![true; count];
    for (index, region) in operation.regions.iter().take(2).enumerate() {
        let Some(block) = region.blocks.first() else {
            return vec![false; count];
        };
        let Some(exits) = block_exit_aliases(block, &aliases) else {
            return vec![false; count];
        };
        for (carry, invariant) in invariants.iter_mut().enumerate() {
            *invariant &= exits.get(carry + usize::from(index == 0)) == Some(&Some(carry));
        }
    }
    invariants
}

fn block_exit_aliases(
    block: &wire::BlockProto,
    argument_aliases: &[Option<usize>],
) -> Option<Vec<Option<usize>>> {
    if block.arguments.len() != argument_aliases.len() {
        return None;
    }
    let mut aliases = block
        .arguments
        .iter()
        .zip(argument_aliases)
        .map(|(argument, alias)| (argument.id, *alias))
        .collect::<HashMap<_, _>>();
    for operation in &block.operations {
        let results = match operation.opcode.as_str() {
            "qcore.stage_lift" => vec![
                operation
                    .operands
                    .first()
                    .and_then(|operand| aliases.get(operand))
                    .copied()
                    .flatten();
                operation.results.len()
            ],
            "qcore.if" => joined_if_aliases(operation, &aliases),
            "qcore.for" => nested_for_aliases(operation, &aliases),
            _ => vec![None; operation.results.len()],
        };
        for (result, alias) in operation.results.iter().zip(results) {
            aliases.insert(result.id, alias);
        }
    }
    let terminator = block.terminator.as_ref()?;
    if !matches!(terminator.kind.as_str(), "yield" | "continue") {
        return None;
    }
    Some(
        terminator
            .operands
            .iter()
            .map(|operand| aliases.get(operand).copied().flatten())
            .collect(),
    )
}

fn nested_for_aliases(
    operation: &wire::OperationProto,
    aliases: &HashMap<u64, Option<usize>>,
) -> Vec<Option<usize>> {
    let carries = operation
        .operands
        .get(3..)
        .unwrap_or_default()
        .iter()
        .map(|operand| aliases.get(operand).copied().flatten())
        .collect::<Vec<_>>();
    if carries.len() != operation.results.len() {
        return vec![None; operation.results.len()];
    }
    let body_inputs = std::iter::once(None)
        .chain(carries.iter().copied())
        .collect::<Vec<_>>();
    let Some(body_outputs) = operation
        .regions
        .first()
        .and_then(|region| region.blocks.first())
        .and_then(|block| block_exit_aliases(block, &body_inputs))
        .filter(|outputs| outputs.len() == carries.len())
    else {
        return vec![None; operation.results.len()];
    };
    let mut results = carries
        .iter()
        .zip(body_outputs)
        .map(|(input, output)| (*input == output).then_some(*input).flatten())
        .collect::<Vec<_>>();
    if let Some(otherwise) = operation
        .regions
        .get(1)
        .and_then(|region| region.blocks.first())
    {
        let Some(outputs) = block_exit_aliases(otherwise, &results)
            .filter(|outputs| outputs.len() == results.len())
        else {
            return vec![None; operation.results.len()];
        };
        for (result, output) in results.iter_mut().zip(outputs) {
            if *result != output {
                *result = None;
            }
        }
    }
    results
}

fn joined_if_aliases(
    operation: &wire::OperationProto,
    aliases: &HashMap<u64, Option<usize>>,
) -> Vec<Option<usize>> {
    let capture_aliases = operation
        .operands
        .iter()
        .skip(1)
        .map(|operand| aliases.get(operand).copied().flatten())
        .collect::<Vec<_>>();
    let branches = operation
        .regions
        .iter()
        .map(|region| {
            region
                .blocks
                .first()
                .and_then(|block| block_exit_aliases(block, &capture_aliases))
        })
        .collect::<Option<Vec<_>>>();
    let Some(branches) = branches else {
        return vec![None; operation.results.len()];
    };
    (0..operation.results.len())
        .map(|index| {
            let first = branches.first()?.get(index).copied().flatten();
            branches
                .iter()
                .all(|branch| branch.get(index).copied().flatten() == first)
                .then_some(first)
                .flatten()
        })
        .collect()
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

fn mapping_string_tuple_attribute<'a>(
    operation: &'a wire::OperationProto,
    attribute_name: &str,
    key: &str,
) -> Option<Vec<&'a str>> {
    use wire::literal_proto::Value;

    let Value::MappingValue(mapping) = attribute(operation, attribute_name)?.value.as_ref()? else {
        return None;
    };
    let Value::TupleValue(sequence) = mapping
        .entries
        .iter()
        .find(|entry| entry.key == key)?
        .value
        .as_ref()?
        .value
        .as_ref()?
    else {
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

fn mapping_string_attribute<'a>(
    operation: &'a wire::OperationProto,
    attribute_name: &str,
    key: &str,
) -> Option<&'a str> {
    use wire::literal_proto::Value;

    let Value::MappingValue(mapping) = attribute(operation, attribute_name)?.value.as_ref()? else {
        return None;
    };
    match mapping
        .entries
        .iter()
        .find(|entry| entry.key == key)?
        .value
        .as_ref()?
        .value
        .as_ref()?
    {
        Value::StringValue(value) => Some(value),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn operation(opcode: &str, regions: Vec<wire::RegionProto>) -> wire::OperationProto {
        wire::OperationProto {
            opcode: opcode.to_owned(),
            regions,
            ..Default::default()
        }
    }

    fn block(operations: Vec<wire::OperationProto>) -> wire::BlockProto {
        wire::BlockProto {
            operations,
            ..Default::default()
        }
    }

    #[test]
    fn inventories_nested_sites_without_unrolling_them() {
        let nested = wire::RegionProto {
            blocks: vec![block(vec![operation("quantum.x", Vec::new())])],
        };
        let package = wire::QCorePackageProto {
            entrypoint: "native.main".to_owned(),
            functions: vec![wire::FunctionProto {
                symbol: "native.main".to_owned(),
                body: Some(block(vec![
                    operation("qcore.call", Vec::new()),
                    operation("qcore.for", vec![nested]),
                ])),
                ..Default::default()
            }],
            ..Default::default()
        };

        let result = census(&package);

        assert_eq!(result.entrypoint(), "native.main");
        assert_eq!(result.block_count(), 2);
        assert_eq!(result.region_count(), 1);
        assert_eq!(result.operation_count(), 3);
        assert_eq!(result.call_site_count(), 1);
        assert_eq!(result.opcodes()["qcore.for"], 1);
        assert_eq!(result.opcodes()["quantum.x"], 1);
    }

    #[test]
    fn retains_external_functions_as_empty_inventory_rows() {
        let package = wire::QCorePackageProto {
            entrypoint: "native.external".to_owned(),
            functions: vec![wire::FunctionProto {
                symbol: "native.external".to_owned(),
                external: true,
                ..Default::default()
            }],
            ..Default::default()
        };

        let result = census(&package);

        assert_eq!(result.functions().len(), 1);
        assert_eq!(result.functions()[0].symbol(), "native.external");
        assert_eq!(result.operation_count(), 0);
        assert!(result.opcodes().is_empty());
    }

    #[test]
    fn computes_python_range_cardinality_without_iteration() {
        assert_eq!(static_range_len(0, 10, 3), Some(4));
        assert_eq!(static_range_len(10, 0, -3), Some(4));
        assert_eq!(static_range_len(10, 0, 3), Some(0));
        assert_eq!(static_range_len(0, 10, -3), Some(0));
    }

    #[test]
    fn exact_counts_fail_instead_of_wrapping() {
        let counts = Counts {
            work: u128::MAX,
            opcodes: BTreeMap::from([("quantum.x".to_owned(), u128::MAX)]),
            families: Some(BTreeMap::from([("gates".to_owned(), u128::MAX)])),
            measurements: Some(0),
            resets: Some(0),
            cleanup_work: 0,
        };

        assert_eq!(counts.repeat(2), Err(ExactResourceError::Overflow));
    }

    #[test]
    fn nested_for_retains_only_unchanged_outer_carries() {
        let value = |id| wire::ValueProto {
            id,
            ..Default::default()
        };
        let terminator = |operands| wire::TerminatorProto {
            kind: "yield".to_owned(),
            operands,
            ..Default::default()
        };
        let changed = wire::OperationProto {
            opcode: "quantum.x".to_owned(),
            operands: vec![23],
            results: vec![value(24)],
            ..Default::default()
        };
        let nested_body = wire::BlockProto {
            arguments: vec![value(20), value(21), value(22), value(23)],
            operations: vec![changed],
            terminator: Some(terminator(vec![21, 22, 24])),
        };
        let nested_else = wire::BlockProto {
            arguments: vec![value(30), value(31), value(32)],
            terminator: Some(terminator(vec![30, 31, 32])),
            ..Default::default()
        };
        let nested = wire::OperationProto {
            opcode: "qcore.for".to_owned(),
            operands: vec![100, 101, 102, 1, 2, 3],
            results: vec![value(10), value(11), value(12)],
            regions: vec![
                wire::RegionProto {
                    blocks: vec![nested_body],
                },
                wire::RegionProto {
                    blocks: vec![nested_else],
                },
            ],
            ..Default::default()
        };
        let outer = wire::BlockProto {
            arguments: vec![value(1), value(2), value(3)],
            operations: vec![nested],
            terminator: Some(terminator(vec![11, 12])),
        };

        assert_eq!(loop_carry_invariants(&outer, 2), vec![true, false]);
    }
}
