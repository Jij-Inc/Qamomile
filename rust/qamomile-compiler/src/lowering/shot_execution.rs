//! Compiler-owned closed quantum frontiers and conditional shot execution.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use prost::Message;
use thiserror::Error;

use crate::circuit::{StaticCircuitValue, compact_circuit_scalars};
use crate::literals::{
    bytes_literal, bytes_value, int_literal, int_value, string_literal, string_value,
    tuple_literal, tuple_values,
};
use crate::result_codec::checked_result_abi;
use crate::simulation::{CircuitScalarEvaluator, evaluate_retained_result};
use crate::wire;
use crate::{
    CircuitLaunchBinding, CircuitParameter, CircuitScalarNode, StaticCircuit, StaticInstruction,
};

/// Maximum distinct joint continuation rows retained by one invocation.
pub const MAX_SHOT_COHORT_ROWS: usize = 4096;
/// Maximum physical child submissions made by one invocation.
pub const MAX_SHOT_SUBMISSIONS: u64 = 4096;
/// Maximum measurement cells retained by one invocation.
pub const MAX_SHOT_COHORT_CELLS: usize = 1_000_000;
/// Maximum physical segment-shot demand before further submissions fail.
pub const MAX_SHOT_PHYSICAL_SHOTS: u64 = 1_000_000;
/// Maximum cumulative scalar/instruction planning cells per invocation.
pub const MAX_SHOT_EXECUTION_WORK: usize = 16_000_000;

/// Rejection of invalid, unsupported, or excessive segmented execution.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[error("Q4470: shot execution plan: {0}")]
pub struct ShotExecutionError(pub(crate) String);

#[allow(clippy::needless_pass_by_value)] // Error adapters consume heterogeneous errors through map_err.
fn invalid(message: impl ToString) -> ShotExecutionError {
    ShotExecutionError(message.to_string())
}

/// Empty quantum frontier proved for one compiler-owned static projection.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ClosedQuantumFrontier {
    pub(crate) instruction_index: usize,
    pub(crate) measurement_prefix: u64,
}

impl ClosedQuantumFrontier {
    /// Return the first instruction after this source-order frontier.
    #[must_use]
    pub const fn instruction_index(self) -> usize {
        self.instruction_index
    }
    /// Return the number of preceding source measurement slots.
    #[must_use]
    pub const fn measurement_prefix(self) -> u64 {
        self.measurement_prefix
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CircuitScalarCheck {
    pub(crate) instruction_index: usize,
    pub(crate) root: crate::CircuitScalarId,
}

/// Source recipe that crossed the native schema, entry and cut-proof boundary.
#[derive(Clone, Debug)]
pub struct ShotExecutionRecipe {
    target: String,
    program: crate::VerifiedProgram,
}

impl ShotExecutionRecipe {
    /// Borrow the target child-format identity.
    #[must_use]
    pub fn target(&self) -> &str {
        &self.target
    }
    /// Borrow the canonical verified source bytes.
    #[must_use]
    pub fn source(&self) -> &[u8] {
        self.program.canonical_bytes()
    }
}

/// Encode a source-bound plan recipe without live target or provider handles.
///
/// # Errors
/// Rejects invalid sources, targets, oversized or unsupported plans.
pub fn seal_shot_execution_recipe(
    target: &str,
    source: &[u8],
) -> Result<Vec<u8>, ShotExecutionError> {
    if target.is_empty() || target.len() > 4096 || source.len() > crate::MAX_QCORE_BYTES {
        return Err(invalid("recipe component size"));
    }
    let payload = tuple_literal(vec![
        string_literal("qamomile.shot-execution"),
        int_literal(1),
        string_literal(target),
        bytes_literal(source),
    ])
    .encode_to_vec();
    inspect_shot_execution_recipe(&payload)?;
    Ok(payload)
}

/// Decode a bounded canonical recipe and rederive its cut proofs from source.
///
/// # Errors
/// Rejects malformed, incompatible, noncanonical or non-segmentable recipes.
pub fn inspect_shot_execution_recipe(
    payload: &[u8],
) -> Result<ShotExecutionRecipe, ShotExecutionError> {
    if payload.len() > crate::MAX_QCORE_BYTES + 8192 {
        return Err(invalid("recipe byte budget"));
    }
    crate::budget::preflight_literal(payload).map_err(|e| invalid(format!("{e:?}")))?;
    let literal = wire::LiteralProto::decode(payload).map_err(invalid)?;
    if literal.encode_to_vec() != payload {
        return Err(invalid("noncanonical recipe"));
    }
    let [kind, version, target, source] =
        tuple_values(&literal).ok_or_else(|| invalid("recipe tuple"))?
    else {
        return Err(invalid("recipe arity"));
    };
    if string_value(kind) != Some("qamomile.shot-execution") || int_value(version) != Some(1) {
        return Err(invalid("recipe version/kind"));
    }
    let target = string_value(target)
        .filter(|s| !s.is_empty() && s.len() <= 4096)
        .ok_or_else(|| invalid("recipe target"))?;
    let source = bytes_value(source).ok_or_else(|| invalid("recipe source"))?;
    let program = crate::VerifiedProgram::from_bytes(source).map_err(invalid)?;
    program.validate_execution_entry().map_err(invalid)?;
    let required = program
        .callable_contract(&program.header().entrypoint)
        .ok_or_else(|| invalid("missing verified entry contract"))?
        .requirements();
    let circuit = program.static_circuit().map_err(invalid)?;
    if !circuit.shot_placement(required).1 {
        return Err(invalid("recipe has no supported closed shot decomposition"));
    }
    circuit.validate_shot_checks()?;
    Ok(ShotExecutionRecipe {
        target: target.to_owned(),
        program,
    })
}

/// Find source-order cuts with no live quantum value crossing the boundary.
pub(crate) fn closed_entry_cuts(package: &wire::QCorePackageProto) -> BTreeSet<usize> {
    // Runtime structured continuations are kept fused in the initial release,
    // even when a finite projection happens to expand their instructions.
    for function in &package.functions {
        let mut blocks = function.body.iter().collect::<Vec<_>>();
        while let Some(block) = blocks.pop() {
            for operation in &block.operations {
                if matches!(operation.opcode.as_str(), "qcore.for" | "qcore.while") {
                    return BTreeSet::new();
                }
                for region in &operation.regions {
                    blocks.extend(&region.blocks);
                }
            }
        }
    }
    let Some(entry) = package
        .functions
        .iter()
        .find(|f| f.symbol == package.entrypoint)
    else {
        return BTreeSet::new();
    };
    let Some(body) = &entry.body else {
        return BTreeSet::new();
    };
    let mut last_use = HashMap::new();
    for (index, operation) in body.operations.iter().enumerate() {
        let mut pending = vec![operation];
        while let Some(operation) = pending.pop() {
            for id in &operation.operands {
                last_use.insert(*id, index);
            }
            for region in &operation.regions {
                for block in &region.blocks {
                    pending.extend(&block.operations);
                    if let Some(term) = &block.terminator {
                        for id in &term.operands {
                            last_use.insert(*id, index);
                        }
                    }
                }
            }
        }
    }
    if let Some(term) = &body.terminator {
        for id in &term.operands {
            last_use.insert(*id, body.operations.len());
        }
    }
    let mut expirations = BTreeMap::<usize, usize>::new();
    let mut live = 0_usize;
    let mut cuts = BTreeSet::new();
    for (index, op) in body.operations.iter().enumerate() {
        for value in &op.results {
            if value.r#type.as_ref().is_some_and(quantum_type)
                && let Some(end) = last_use.get(&value.id).filter(|end| **end > index)
            {
                live += 1;
                *expirations.entry(*end).or_default() += 1;
            }
        }
        live -= expirations.remove(&index).unwrap_or_default();
        if live == 0 {
            cuts.insert(index);
        }
    }
    cuts
}

fn quantum_type(ty: &wire::TypeProto) -> bool {
    if ty.kind == wire::type_proto::Kind::Function as i32 {
        return false;
    }
    ty.kind == wire::type_proto::Kind::Quantum as i32 || ty.arguments.iter().any(quantum_type)
}

/// One checked child circuit and its conditional logical population.
#[derive(Clone, Debug)]
pub struct ShotExecutionRequest {
    circuit: StaticCircuit,
    shots: u64,
    seed: u64,
    measurement_range: (u32, u32),
}

impl ShotExecutionRequest {
    /// Borrow the compiler-materialized closed circuit.
    #[must_use]
    pub const fn circuit(&self) -> &StaticCircuit {
        &self.circuit
    }
    /// Return the number of logical trajectories in this request.
    #[must_use]
    pub const fn shots(&self) -> u64 {
        self.shots
    }
    /// Return a fresh deterministic child stream identity.
    #[must_use]
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// Return the complete source-slot transport width, including inactive slots.
    #[must_use]
    pub const fn result_width(&self) -> u64 {
        (self.measurement_range.1 - self.measurement_range.0) as u64
    }

    /// Export the checked raw measurement transport interface.
    #[must_use]
    pub fn interface(&self) -> Vec<u8> {
        tuple_literal(vec![
            int_literal(1),
            int_literal(i64::from(self.measurement_range.0)),
            int_literal(i64::from(self.measurement_range.1)),
        ])
        .encode_to_vec()
    }
}

/// Endpoint seed contract for conditional quantum occurrences.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShotSeedPolicy {
    /// Defer independent sampling to a provider without explicit seed support.
    ProviderDefault,
    /// Derive child simulator streams from the original invocation seed.
    Seeded(u64),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PendingShotRequest {
    Ready,
    Physical,
    Virtual,
}

/// Native joint-histogram continuation state; samples are never memoized.
#[derive(Clone, Debug)]
pub struct ShotExecutionState {
    circuit: StaticCircuit,
    bindings: Vec<CircuitLaunchBinding>,
    result_abi: Vec<u8>,
    cuts: Vec<ClosedQuantumFrontier>,
    stage: usize,
    rows: BTreeMap<Vec<u8>, u64>,
    pending: Vec<(Vec<u8>, u64)>,
    next_rows: BTreeMap<Vec<u8>, u64>,
    request_status: PendingShotRequest,
    seed: ShotSeedPolicy,
    submissions: u64,
    physical_shots: u64,
    logical_shots: u64,
    remaining_work: usize,
}

impl ShotExecutionState {
    /// Construct bounded execution from a compiler projection and checked ABI.
    ///
    /// # Errors
    /// Rejects unsupported runtime frames, absent cuts, invalid ABI or shots.
    pub fn new(
        circuit: StaticCircuit,
        bindings: Vec<CircuitLaunchBinding>,
        result_abi: Vec<u8>,
        shots: u64,
        seed: ShotSeedPolicy,
    ) -> Result<Self, ShotExecutionError> {
        if shots == 0 {
            return Err(invalid("shots must be positive"));
        }
        checked_result_abi(&result_abi).map_err(invalid)?;
        if !circuit.portable_continuations() {
            return Err(invalid(
                "runtime call/loop state is not a shareable closed continuation",
            ));
        }
        let mut cuts = vec![ClosedQuantumFrontier {
            instruction_index: 0,
            measurement_prefix: 0,
        }];
        cuts.extend(circuit.execution_cuts());
        cuts.push(ClosedQuantumFrontier {
            instruction_index: circuit.instructions().len(),
            measurement_prefix: circuit.measurement_count(),
        });
        if cuts.len() < 3 {
            return Err(invalid(
                "no closed measured quantum frontier separates executions",
            ));
        }
        circuit.validate_shot_checks()?;
        if !circuit.sampled_children(&instruction_paths(circuit.instructions())) {
            return Err(invalid(
                "a conditional quantum child needs completion-only execution; keep the circuit fused",
            ));
        }
        let rows = BTreeMap::from([(Vec::new(), shots)]);
        Ok(Self {
            circuit,
            bindings,
            result_abi,
            cuts,
            stage: 0,
            pending: rows.iter().map(|(r, n)| (r.clone(), *n)).collect(),
            rows,
            next_rows: BTreeMap::new(),
            request_status: PendingShotRequest::Ready,
            seed,
            submissions: 0,
            physical_shots: 0,
            logical_shots: shots,
            remaining_work: MAX_SHOT_EXECUTION_WORK,
        })
    }

    /// Return the next closed child without consuming or duplicating samples.
    ///
    /// # Errors
    /// Rejects repeated outstanding requests, limits, or illegal scalar motion.
    pub fn next_request(&mut self) -> Result<Option<ShotExecutionRequest>, ShotExecutionError> {
        loop {
            if self.request_status != PendingShotRequest::Ready {
                return Err(invalid("accept the outstanding request first"));
            }
            if self.pending.is_empty() {
                if self.stage + 2 >= self.cuts.len() {
                    return Ok(None);
                }
                self.rows = std::mem::take(&mut self.next_rows);
                self.stage += 1;
                self.pending = self.rows.iter().map(|(r, n)| (r.clone(), *n)).collect();
            }
            let (prefix, shots) = self
                .pending
                .last()
                .ok_or_else(|| invalid("empty continuation population"))?;
            if self.submissions >= MAX_SHOT_SUBMISSIONS {
                return Err(invalid("physical submission budget exceeded"));
            }
            let start = self.cuts[self.stage];
            let end = self.cuts[self.stage + 1];
            let work = self
                .circuit
                .scalars()
                .len()
                .checked_add(self.circuit.operation_count())
                .ok_or_else(|| invalid("planning work overflow"))?;
            self.remaining_work = self
                .remaining_work
                .checked_sub(work)
                .ok_or_else(|| invalid("cumulative planning work budget exceeded"))?;
            let circuit = self
                .circuit
                .closed_segment(start, end, prefix, &self.bindings)?;
            let seed = match self.seed {
                ShotSeedPolicy::ProviderDefault => 0,
                ShotSeedPolicy::Seeded(seed) => {
                    seed.wrapping_add(self.submissions.wrapping_mul(0x9e37_79b9_7f4a_7c15))
                }
            };
            if circuit.instructions().is_empty() {
                self.request_status = PendingShotRequest::Virtual;
                let population = *shots;
                let width = usize::try_from(end.measurement_prefix - start.measurement_prefix)
                    .map_err(invalid)?;
                self.accept(&[(vec![0; width], population)])?;
                continue;
            }
            if self
                .physical_shots
                .checked_add(*shots)
                .is_none_or(|n| n > MAX_SHOT_PHYSICAL_SHOTS)
            {
                return Err(invalid(
                    "physical segment-shot budget exceeded before submission",
                ));
            }

            self.request_status = PendingShotRequest::Physical;
            return Ok(Some(ShotExecutionRequest {
                circuit,
                shots: *shots,
                seed,
                measurement_range: (
                    u32::try_from(start.measurement_prefix).map_err(invalid)?,
                    u32::try_from(end.measurement_prefix).map_err(invalid)?,
                ),
            }));
        }
    }

    /// Accept exactly the requested population and preserve its joint prefix.
    ///
    /// # Errors
    /// Rejects malformed rows, wrong population, overflow or cohort limits.
    pub fn accept(&mut self, counts: &[(Vec<u8>, u64)]) -> Result<(), ShotExecutionError> {
        if self.request_status == PendingShotRequest::Ready {
            return Err(invalid("no outstanding quantum request"));
        }
        let (prefix, expected) = self
            .pending
            .last()
            .ok_or_else(|| invalid("missing request population"))?;
        let width = usize::try_from(
            self.cuts[self.stage + 1].measurement_prefix - self.cuts[self.stage].measurement_prefix,
        )
        .map_err(invalid)?;
        if counts.len() > MAX_SHOT_COHORT_ROWS
            || counts
                .len()
                .checked_mul(prefix.len() + width)
                .is_none_or(|n| n > MAX_SHOT_COHORT_CELLS)
        {
            return Err(invalid(
                "child joint distribution exceeds its allocation budget",
            ));
        }
        let mut total = 0_u64;
        let mut additions = BTreeMap::<Vec<u8>, u64>::new();
        for (row, count) in counts {
            if row.len() != width || row.iter().any(|bit| *bit > 1) || *count == 0 {
                return Err(invalid("invalid child measurement row"));
            }
            total = total
                .checked_add(*count)
                .ok_or_else(|| invalid("shot count overflow"))?;
            let mut joint = prefix.clone();
            joint.extend(row);
            let n = additions.entry(joint).or_default();
            *n = n
                .checked_add(*count)
                .ok_or_else(|| invalid("row count overflow"))?;
        }
        if total != *expected {
            return Err(invalid(
                "child shot count disagrees with requested population",
            ));
        }
        let new_keys = additions
            .keys()
            .filter(|row| !self.next_rows.contains_key(*row))
            .count();
        let size = self
            .next_rows
            .len()
            .checked_add(new_keys)
            .ok_or_else(|| invalid("cohort row overflow"))?;
        let cells = size
            .checked_mul(prefix.len() + width)
            .ok_or_else(|| invalid("cohort size overflow"))?;
        let old_cells = self
            .rows
            .len()
            .checked_add(self.pending.len())
            .and_then(|rows| rows.checked_mul(prefix.len()))
            .ok_or_else(|| invalid("retained cohort cell overflow"))?;
        let peak_cells = cells
            .checked_add(old_cells)
            .and_then(|n| n.checked_add(additions.len().checked_mul(prefix.len() + width)?))
            .ok_or_else(|| invalid("peak cohort cell overflow"))?;
        if size > MAX_SHOT_COHORT_ROWS || peak_cells > MAX_SHOT_COHORT_CELLS {
            return Err(invalid("joint cohort budget exceeded"));
        }
        let physical = self
            .physical_shots
            .checked_add(if self.request_status == PendingShotRequest::Physical {
                total
            } else {
                0
            })
            .ok_or_else(|| invalid("physical shot count overflow"))?;
        for (row, n) in additions {
            let count = self.next_rows.entry(row).or_default();
            *count = count
                .checked_add(n)
                .ok_or_else(|| invalid("joint row count overflow"))?;
        }
        self.physical_shots = physical;
        if self.request_status == PendingShotRequest::Physical {
            self.submissions += 1;
        }
        self.pending.pop();
        self.request_status = PendingShotRequest::Ready;
        Ok(())
    }

    /// Return logical results only after every conditional population completes.
    ///
    /// # Errors
    /// Rejects incomplete execution or undefined classical result evaluation.
    pub fn finish(&self) -> Result<Vec<(Vec<u8>, u64)>, ShotExecutionError> {
        if self.request_status != PendingShotRequest::Ready
            || !self.pending.is_empty()
            || self.stage + 2 != self.cuts.len()
        {
            return Err(invalid("execution is incomplete"));
        }
        if self
            .next_rows
            .len()
            .checked_mul(self.circuit.scalars().len())
            .is_none_or(|work| work > self.remaining_work)
        {
            return Err(invalid("final classical evaluation work budget exceeded"));
        }
        let abi = checked_result_abi(&self.result_abi).map_err(invalid)?;
        let mut logical = BTreeMap::<Vec<u8>, u64>::new();
        for (row, count) in &self.next_rows {
            let mut evaluator =
                CircuitScalarEvaluator::new(&self.circuit, &self.bindings, row).map_err(invalid)?;
            for check in self.circuit.scalar_checks() {
                if check.instruction_index == self.circuit.instructions().len() {
                    evaluator.evaluate(check.root).map_err(invalid)?;
                }
            }
            let literal =
                evaluate_retained_result(&self.circuit, &self.bindings, row).map_err(invalid)?;
            let bits = abi.encode_literal(&literal).map_err(invalid)?;
            let n = logical.entry(bits).or_default();
            *n = n
                .checked_add(*count)
                .ok_or_else(|| invalid("logical count overflow"))?;
        }
        if logical.values().sum::<u64>() != self.logical_shots {
            return Err(invalid("logical population changed"));
        }
        Ok(logical.into_iter().collect())
    }

    /// Return physical segment-shots consumed so far.
    #[must_use]
    pub const fn physical_shots(&self) -> u64 {
        self.physical_shots
    }
    /// Return completed physical submissions so far.
    #[must_use]
    pub const fn submissions(&self) -> u64 {
        self.submissions
    }
}

fn shareable_scalar(node: &CircuitScalarNode) -> bool {
    match node {
        CircuitScalarNode::Bool(_)
        | CircuitScalarNode::Int(_)
        | CircuitScalarNode::RealBits(_)
        | CircuitScalarNode::Measurement(_)
        | CircuitScalarNode::Parameter { .. } => true,
        CircuitScalarNode::Apply { opcode, .. } => {
            match crate::operations::Opcode::try_from(opcode.as_str()) {
                Ok(crate::operations::Opcode::Classical(op)) => shareable_classical(op),
                _ => false,
            }
        }
        CircuitScalarNode::CallValue { .. } | CircuitScalarNode::LoopIndex(_) => false,
    }
}

// This exhaustive positive proof forces new classical opcodes to make an
// explicit sharing decision, even if their effect implementation is incomplete.
fn shareable_classical(op: crate::operations::ClassicalOpcode) -> bool {
    use crate::operations::ClassicalOpcode as C;
    match op {
        C::Array
        | C::ObservableZero
        | C::ObservablePauli
        | C::Add
        | C::Sub
        | C::Mul
        | C::TrueDiv
        | C::FloorDiv
        | C::Mod
        | C::Pow
        | C::AndBits
        | C::OrBits
        | C::Xor
        | C::LeftShift
        | C::RightShift
        | C::And
        | C::Or
        | C::Not
        | C::Invert
        | C::Neg
        | C::Pos
        | C::CompareChain
        | C::Select
        | C::Index
        | C::Slice
        | C::Update
        | C::MapGet
        | C::Project
        | C::Cast
        | C::Len
        | C::Ceil
        | C::Log2
        | C::Sqrt => true,
    }
}

impl StaticCircuit {
    pub(crate) fn validate_shot_checks(&self) -> Result<(), ShotExecutionError> {
        let dependencies = scalar_dependencies(self.scalars());
        let mut cuts = self.execution_cuts();
        cuts.insert(
            0,
            ClosedQuantumFrontier {
                instruction_index: 0,
                measurement_prefix: 0,
            },
        );
        cuts.push(ClosedQuantumFrontier {
            instruction_index: self.instructions().len(),
            measurement_prefix: self.measurement_count(),
        });
        for check in self.scalar_checks() {
            if !cuts.iter().any(|cut| {
                cut.instruction_index == check.instruction_index
                    && known_before(check.root, &dependencies, cut.measurement_prefix)
            }) {
                return Err(invalid(
                    "classical checks inside a live quantum segment require ordered in-job execution",
                ));
            }
        }
        Ok(())
    }

    /// Derive device feedback requirements after proving closed host cuts.
    #[must_use]
    pub fn shot_placement(&self, required: crate::RequirementSet) -> (crate::RequirementSet, bool) {
        let mut portable = self.portable_continuations();
        let summaries = instruction_paths(self.instructions());
        if portable && !self.sampled_children(&summaries) {
            portable = false;
        }
        let cuts = if portable {
            self.execution_cuts()
        } else {
            Vec::new()
        };
        let segmented = !cuts.is_empty();
        let mut residual = required.difference(
            crate::RequirementSet::EMPTY
                .with(crate::Requirement::DynamicBranch)
                .with(crate::Requirement::DynamicRotation)
                .with(crate::Requirement::MidCircuitMeasurement),
        );
        if !portable {
            for requirement in [
                crate::Requirement::DynamicBranch,
                crate::Requirement::MidCircuitMeasurement,
            ] {
                if required.contains(requirement) {
                    residual = residual.with(requirement);
                }
            }
        }
        if segmented {
            residual = residual
                .difference(crate::RequirementSet::EMPTY.with(crate::Requirement::HostDispatch));
        }
        let mut measured = false;
        let mut prefix = 0;
        let dependencies = scalar_dependencies(self.scalars());
        let mut frontiers = cuts.iter().peekable();
        for (index, op) in self.instructions().iter().enumerate() {
            if segmented {
                while frontiers
                    .peek()
                    .is_some_and(|cut| cut.instruction_index <= index)
                {
                    if let Some(cut) = frontiers.next() {
                        prefix = cut.measurement_prefix;
                        measured = false;
                    }
                }
            }
            let mut pending = vec![(op, measured)];
            while let Some((op, measured_before)) = pending.pop() {
                if op.opcode() != "measure"
                    && measured_before
                    && (!op.qubits().is_empty() || !op.controls().is_empty())
                {
                    residual = residual.with(crate::Requirement::MidCircuitMeasurement);
                }
                if op
                    .condition()
                    .is_some_and(|id| !known_before(id, &dependencies, prefix))
                {
                    residual = residual.with(crate::Requirement::DynamicBranch);
                    residual = residual.with(crate::Requirement::MidCircuitMeasurement);
                }
                for parameter in op.parameters() {
                    if let CircuitParameter::Scalar(id) = parameter
                        && self.scalars()[id.index() as usize].scalar_type()
                            == crate::CircuitScalarType::Real
                        && dependencies[id.index() as usize]
                            .latest_measurement
                            .is_some()
                        && !known_before(*id, &dependencies, prefix)
                    {
                        residual = residual
                            .with(crate::Requirement::DynamicRotation)
                            .with(crate::Requirement::MidCircuitMeasurement);
                    }
                }
                for region in op.regions() {
                    let mut branch_measured = measured_before;
                    for child in region.instructions() {
                        pending.push((child, branch_measured));
                        branch_measured |= summaries[&std::ptr::from_ref(child)].may_measure();
                    }
                }
            }
            measured |= summaries[&std::ptr::from_ref(op)].may_measure();
        }
        (residual, segmented)
    }

    fn portable_continuations(&self) -> bool {
        let mut pending = vec![self.instructions()];
        let mut portable = self.scalars().iter().all(shareable_scalar);
        while let Some(block) = pending.pop() {
            for op in block {
                if matches!(op.opcode(), "for" | "call") {
                    portable = false;
                }
                for region in op.regions() {
                    if region.has_runtime_return() {
                        portable = false;
                    }
                    pending.push(region.instructions());
                }
            }
        }
        portable
    }

    fn sampled_children(
        &self,
        summaries: &HashMap<*const StaticInstruction, InstructionPaths>,
    ) -> bool {
        let mut start = 0;
        for end in self
            .execution_cuts()
            .iter()
            .map(|cut| cut.instruction_index)
            .chain(std::iter::once(self.instructions().len()))
        {
            if block_paths(&self.instructions()[start..end], summaries).has_completion_only_path() {
                return false;
            }
            start = end;
        }
        true
    }

    fn execution_cuts(&self) -> Vec<ClosedQuantumFrontier> {
        // A child transports a nonempty source measurement interval. Keep the
        // first empty frontier at each prefix; intervening fresh quantum work
        // without measurements remains in the following child.
        let mut cuts = Vec::<ClosedQuantumFrontier>::new();
        for cut in self.closed_cuts().iter().copied() {
            if cut.instruction_index > 0
                && cut.instruction_index < self.instructions().len()
                && cut.measurement_prefix > 0
                && cut.measurement_prefix < self.measurement_count()
                && cuts
                    .last()
                    .is_none_or(|last| last.measurement_prefix != cut.measurement_prefix)
            {
                cuts.push(cut);
            }
        }
        cuts
    }

    pub(crate) fn closed_segment(
        &self,
        start: ClosedQuantumFrontier,
        end: ClosedQuantumFrontier,
        prefix: &[u8],
        bindings: &[CircuitLaunchBinding],
    ) -> Result<Self, ShotExecutionError> {
        if prefix.len() != usize::try_from(start.measurement_prefix).map_err(invalid)? {
            return Err(invalid("continuation measurement prefix width"));
        }
        let mut scalars = self.scalars().to_vec();
        for node in &mut scalars {
            if let CircuitScalarNode::Measurement(index) = node {
                if *index < start.measurement_prefix {
                    *node = CircuitScalarNode::Bool(
                        prefix[usize::try_from(*index).map_err(invalid)?] != 0,
                    );
                } else if *index < end.measurement_prefix {
                    *index -= start.measurement_prefix;
                }
            }
        }
        let mut evaluator = CircuitScalarEvaluator::new(self, bindings, prefix).map_err(invalid)?;
        let dependencies = scalar_dependencies(self.scalars());
        for check in self.scalar_checks() {
            if check.instruction_index == start.instruction_index {
                evaluator.evaluate(check.root).map_err(invalid)?;
            }
        }
        let mut instructions = materialize_instructions(
            &self.instructions()[start.instruction_index..end.instruction_index],
            prefix,
            &mut evaluator,
            &mut scalars,
            &dependencies,
        )?;
        let mut wires = BTreeSet::new();
        let mut active_measurements = BTreeSet::new();
        let mut pending = vec![instructions.as_slice()];
        while let Some(block) = pending.pop() {
            for op in block {
                wires.extend(op.qubits());
                wires.extend(op.controls());
                for parameter in op.parameters() {
                    if let CircuitParameter::MeasurementIndex(index) = parameter {
                        active_measurements.insert(*index);
                    }
                }
                for region in op.regions() {
                    if region.runtime_return_value().is_some() {
                        return Err(invalid("early returns cannot cross a closed segment"));
                    }
                    pending.push(region.instructions());
                }
            }
        }
        let wire_map = wires
            .iter()
            .enumerate()
            .map(|(new, old)| (*old, new as u64))
            .collect::<HashMap<_, _>>();
        let measurement_map = active_measurements
            .iter()
            .copied()
            .enumerate()
            .map(|(new, old)| (old, new as u64))
            .collect::<HashMap<_, _>>();
        let width = end.measurement_prefix - start.measurement_prefix;
        for node in &mut scalars {
            if let CircuitScalarNode::Measurement(index) = node
                && *index < width
            {
                *node =
                    transported_measurement(&measurement_map, *index + start.measurement_prefix);
            }
        }
        remap_instructions(&mut instructions, &wire_map, &measurement_map)?;
        let mut result = StaticCircuitValue::Aggregate(
            (0..width)
                .map(|index| {
                    let id = crate::CircuitScalarId::from_index(
                        u32::try_from(scalars.len()).expect("bounded circuit scalar arena"),
                    );
                    scalars.push(transported_measurement(
                        &measurement_map,
                        index + start.measurement_prefix,
                    ));
                    StaticCircuitValue::Scalar(id)
                })
                .collect(),
        );
        let compact =
            compact_circuit_scalars(&mut instructions, Some(&mut result), &scalars, &mut [])
                .map_err(invalid)?;
        Ok(StaticCircuit::from_closed_segment(
            self.realizations().to_vec(),
            wire_map.len() as u64,
            active_measurements.len() as u64,
            instructions,
            compact,
            result,
            active_measurements.len() as u64 == width,
        )
        .inherit_optimization_stages(self))
    }
}

// Missing slots belong to branches removed by native guarded materialization.
fn transported_measurement(measurements: &HashMap<u64, u64>, source: u64) -> CircuitScalarNode {
    measurements
        .get(&source)
        .map_or(CircuitScalarNode::Bool(false), |index| {
            CircuitScalarNode::Measurement(*index)
        })
}

// Four finite path states distinguish empty work, quantum-only work, a
// measurement, and both. Alternatives are joined; source sequences compose.
#[derive(Clone, Copy)]
struct InstructionPaths(u8);

impl InstructionPaths {
    const EMPTY: Self = Self(1);
    const fn may_measure(self) -> bool {
        self.0 & 0b1100 != 0
    }
    const fn has_completion_only_path(self) -> bool {
        self.0 & 0b0010 != 0
    }
    fn then(self, next: Self) -> Self {
        let mut paths = 0;
        for before in 0..4 {
            for after in 0..4 {
                if self.0 & (1 << before) != 0 && next.0 & (1 << after) != 0 {
                    paths |= 1 << (before | after);
                }
            }
        }
        Self(paths)
    }
}

fn block_paths(
    ops: &[StaticInstruction],
    summaries: &HashMap<*const StaticInstruction, InstructionPaths>,
) -> InstructionPaths {
    ops.iter().fold(InstructionPaths::EMPTY, |paths, op| {
        paths.then(summaries[&std::ptr::from_ref(op)])
    })
}

// Compute once in postorder, then carry measurement state separately through
// each alternative. Sequentially visiting exclusive arms creates false feedback.
fn instruction_paths(
    instructions: &[StaticInstruction],
) -> HashMap<*const StaticInstruction, InstructionPaths> {
    let mut order = Vec::new();
    let mut pending = instructions.iter().collect::<Vec<_>>();
    while let Some(op) = pending.pop() {
        order.push(op);
        for region in op.regions() {
            pending.extend(region.instructions());
        }
    }
    let mut summaries = HashMap::with_capacity(order.len());
    for op in order.into_iter().rev() {
        let paths = if op.regions().is_empty() {
            InstructionPaths(if op.opcode() == "measure" {
                0b1000
            } else {
                0b0010
            })
        } else {
            InstructionPaths(op.regions().iter().fold(0, |paths, region| {
                paths | block_paths(region.instructions(), &summaries).0
            }))
        };
        summaries.insert(std::ptr::from_ref(op), paths);
    }
    summaries
}

#[derive(Clone, Copy, Default)]
struct ScalarDependency {
    latest_measurement: Option<u64>,
    opaque: bool,
}

fn scalar_dependencies(scalars: &[CircuitScalarNode]) -> Vec<ScalarDependency> {
    let mut dependencies = Vec::<ScalarDependency>::with_capacity(scalars.len());
    for node in scalars {
        let mut dependency = ScalarDependency::default();
        match node {
            CircuitScalarNode::Measurement(index) => dependency.latest_measurement = Some(*index),
            CircuitScalarNode::CallValue { .. } | CircuitScalarNode::LoopIndex(_) => {
                dependency.opaque = true;
            }
            CircuitScalarNode::Apply { operands, .. } => {
                for id in operands {
                    match dependencies.get(id.index() as usize) {
                        Some(operand) => {
                            dependency.latest_measurement = dependency
                                .latest_measurement
                                .max(operand.latest_measurement);
                            dependency.opaque |= operand.opaque;
                        }
                        None => dependency.opaque = true,
                    }
                }
            }
            _ => {}
        }
        dependencies.push(dependency);
    }
    dependencies
}

fn known_before(
    id: crate::CircuitScalarId,
    dependencies: &[ScalarDependency],
    prefix: u64,
) -> bool {
    dependencies.get(id.index() as usize).is_some_and(|node| {
        !node.opaque && node.latest_measurement.is_none_or(|index| index < prefix)
    })
}

fn materialize_instructions(
    ops: &[StaticInstruction],
    prefix: &[u8],
    evaluator: &mut CircuitScalarEvaluator<'_>,
    scalars: &mut Vec<CircuitScalarNode>,
    dependencies: &[ScalarDependency],
) -> Result<Vec<StaticInstruction>, ShotExecutionError> {
    let mut out = Vec::new();
    for op in ops {
        if op.opcode() == "if"
            && op
                .condition()
                .is_some_and(|id| known_before(id, dependencies, prefix.len() as u64))
        {
            let value = evaluator
                .evaluate(op.condition().expect("condition"))
                .map_err(invalid)?;
            let selected = match value.value {
                Some(wire::literal_proto::Value::BoolValue(value)) => usize::from(!value),
                _ => return Err(invalid("conditional value is not Boolean")),
            };
            let region = op
                .regions()
                .get(selected)
                .ok_or_else(|| invalid("conditional region ABI"))?;
            if region.runtime_return_value().is_some() {
                return Err(invalid("early return at a host cut"));
            }
            out.extend(materialize_instructions(
                region.instructions(),
                prefix,
                evaluator,
                scalars,
                dependencies,
            )?);
        } else {
            let mut op = op.clone();
            for parameter in op.parameters_mut() {
                if let CircuitParameter::Scalar(id) = parameter
                    && known_before(*id, dependencies, prefix.len() as u64)
                {
                    let literal = evaluator.evaluate(*id).map_err(invalid)?;
                    let node = match literal.value {
                        Some(wire::literal_proto::Value::BoolValue(value)) => {
                            CircuitScalarNode::Bool(value)
                        }
                        Some(wire::literal_proto::Value::IntValue(value)) => {
                            CircuitScalarNode::Int(value)
                        }
                        Some(wire::literal_proto::Value::RealValue(value)) => {
                            CircuitScalarNode::RealBits(value.to_bits())
                        }
                        _ => {
                            return Err(invalid(
                                "materialized gate scalar exceeds the circuit scalar domain",
                            ));
                        }
                    };
                    *id = crate::CircuitScalarId::from_index(
                        u32::try_from(scalars.len()).map_err(invalid)?,
                    );
                    scalars.push(node);
                }
            }
            for region in op.regions_mut() {
                let instructions = materialize_instructions(
                    region.instructions(),
                    prefix,
                    evaluator,
                    scalars,
                    dependencies,
                )?;
                region.replace_instructions(instructions);
            }
            out.push(op);
        }
    }
    Ok(out)
}

fn remap_instructions(
    ops: &mut [StaticInstruction],
    wires: &HashMap<u64, u64>,
    measurements: &HashMap<u64, u64>,
) -> Result<(), ShotExecutionError> {
    for op in ops {
        op.remap_segment(wires, measurements).map_err(invalid)?;
        for block in op.regions_mut() {
            remap_instructions(block.instructions_mut(), wires, measurements)?;
        }
    }
    Ok(())
}

/// Derive the raw child result ABI from its complete measurement row.
///
/// # Errors
/// Rejects empty or excessive measurement widths before allocating the ABI.
pub fn shot_segment_result_abi(width: u64) -> Result<Vec<u8>, ShotExecutionError> {
    if width == 0 || width > MAX_SHOT_COHORT_CELLS as u64 {
        return Err(invalid("raw segment measurement width"));
    }
    let native_width = u32::try_from(width).map_err(invalid)?;
    let ty = wire::TypeProto {
        kind: wire::type_proto::Kind::Staged as i32,
        name: "Shot".into(),
        arguments: vec![wire::TypeProto {
            kind: wire::type_proto::Kind::Array as i32,
            name: "Bits".into(),
            arguments: vec![wire::TypeProto {
                kind: wire::type_proto::Kind::Scalar as i32,
                name: "Bit".into(),
                ..Default::default()
            }],
            dimensions: vec![wire::DimensionProto {
                value: Some(wire::dimension_proto::Value::Fixed(width)),
            }],
            ..Default::default()
        }],
        ..Default::default()
    };
    Ok(wire::ResultAbiProto {
        r#type: Some(ty),
        source_order: (0..native_width).collect(),
        native_width,
    }
    .encode_to_vec())
}

/// Verify a raw child interface against source-proved measurement frontiers.
///
/// # Errors
/// Rejects forged ranges, ABIs, noncanonical input, and excessive payloads.
pub fn validate_shot_segment_interface(
    program: &crate::VerifiedProgram,
    receipt: &[u8],
    parameter_abi: &[u8],
    result_abi: &[u8],
) -> Result<(), ShotExecutionError> {
    if receipt.len() > 128 {
        return Err(invalid("segment interface size"));
    }
    let literal = wire::LiteralProto::decode(receipt).map_err(invalid)?;
    if literal.encode_to_vec() != receipt {
        return Err(invalid("noncanonical segment interface"));
    }
    let [version, start, end] =
        tuple_values(&literal).ok_or_else(|| invalid("segment interface tuple"))?
    else {
        return Err(invalid("segment interface arity"));
    };
    if int_value(version) != Some(1) {
        return Err(invalid("segment interface version"));
    }
    let start = u64::try_from(int_value(start).ok_or_else(|| invalid("segment start"))?)
        .map_err(invalid)?;
    let end =
        u64::try_from(int_value(end).ok_or_else(|| invalid("segment end"))?).map_err(invalid)?;
    let circuit = program.static_circuit().map_err(invalid)?;
    let boundary = |n| {
        n == 0
            || n == circuit.measurement_count()
            || circuit
                .closed_cuts()
                .iter()
                .any(|cut| cut.measurement_prefix == n)
    };
    if start >= end || end > circuit.measurement_count() || !boundary(start) || !boundary(end) {
        return Err(invalid("range crosses an unproved quantum frontier"));
    }
    let derived = program.artifact_abi().map_err(invalid)?;
    if parameter_abi != derived.parameter_abi()
        || result_abi != shot_segment_result_abi(end - start)?
    {
        return Err(invalid("raw child ABI disagrees with the native interface"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregate_survivor_prevents_an_apparent_measurement_cut() {
        let quantum = wire::TypeProto {
            kind: wire::type_proto::Kind::Quantum as i32,
            ..Default::default()
        };
        let aggregate = wire::TypeProto {
            kind: wire::type_proto::Kind::Product as i32,
            arguments: vec![quantum.clone()],
            ..Default::default()
        };
        let result = |id, ty| wire::ValueProto {
            id,
            r#type: Some(ty),
            ..Default::default()
        };
        let op = |operands, results| wire::OperationProto {
            operands,
            results,
            ..Default::default()
        };
        let package = wire::QCorePackageProto {
            entrypoint: "entry".to_owned(),
            functions: vec![wire::FunctionProto {
                symbol: "entry".to_owned(),
                body: Some(wire::BlockProto {
                    operations: vec![
                        op(vec![], vec![result(1, aggregate)]),
                        op(vec![], vec![result(2, quantum.clone())]),
                        op(vec![2], vec![]),
                        op(vec![], vec![result(3, quantum.clone())]),
                        op(vec![1, 3], vec![result(4, quantum)]),
                        op(vec![4], vec![]),
                    ],
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        };
        assert_eq!(closed_entry_cuts(&package), BTreeSet::from([5]));
    }

    #[test]
    fn unknown_classical_operations_have_no_sharing_proof() {
        let unknown = CircuitScalarNode::Apply {
            opcode: "classical.random".to_owned(),
            result_type: crate::CircuitScalarType::Real,
            attributes: Box::new([]),
            operands: Box::new([]),
        };
        assert!(!shareable_scalar(&unknown));
        assert!(!shareable_scalar(&CircuitScalarNode::LoopIndex(0)));
        assert!(shareable_scalar(&CircuitScalarNode::Measurement(0)));
    }
}
