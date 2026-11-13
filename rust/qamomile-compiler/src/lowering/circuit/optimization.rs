//! Exact local circuit rewriting with source-check and segment boundaries.
//!
//! Each gate looks back along its wires past earlier gates that provably
//! commute with it (both diagonal, or both in the `{I, X}` algebra, on every
//! shared wire; controls are diagonal), then cancels inverse pairs, fuses exact
//! phase gates, and merges rotation angles. Top-level SWAPs become relabeling.

use std::collections::{HashMap, HashSet};

use super::{
    CircuitDefinition, CircuitParameter, CircuitScalarId, CircuitScalarNode, CircuitScalarType,
    StaticCircuit, StaticCircuitError, StaticInstruction, instruction_count,
};

/// Maximum earlier slots inspected for one candidate rewrite.
const LOOKBACK_WINDOW: usize = 128;
/// Maximum scalar DAG depth folded while proving literal angles.
const MAX_FOLD_DEPTH: usize = 32;

impl StaticCircuit {
    /// Simplify projected gates without changing phase, wires, or measurement slots.
    ///
    /// # Errors
    /// Rejects excessive work or region depth before returning a candidate.
    pub fn simplify_exact(mut self, max_work: u64) -> Result<Self, StaticCircuitError> {
        let count =
            u64::try_from(self.operation_count).map_err(|_| StaticCircuitError::Overflow)?;
        if max_work == 0 || count > max_work {
            return Err(StaticCircuitError::Unsupported(
                "circuit optimization work budget exceeded".into(),
            ));
        }
        let barriers = self
            .closed_cuts
            .iter()
            .map(|c| c.instruction_index)
            .chain(self.scalar_checks.iter().map(|c| c.instruction_index))
            .collect::<HashSet<_>>();
        let mut work = RewriteWork::new(max_work);
        let mut arena = ScalarArena::new(std::mem::take(&mut self.scalars).into_vec());
        // Live result wires must end in their recorded positions; otherwise
        // a trailing wire permutation is unobservable and is dropped.
        let restore = self.result_value.as_ref().is_some_and(contains_wires);
        let (instructions, offsets) = simplify(
            self.instructions.into_vec(),
            &barriers,
            0,
            &mut work,
            &mut arena,
            Some(restore),
        )?;
        self.scalars = arena.into_nodes().into_boxed_slice();
        // Checks recorded while a nested region was being emitted can carry a
        // position past the top-level end; keep them past the new end so
        // consumers see the same relation to every cut and to the end.
        let remap = |index: usize| match offsets.get(index) {
            Some(offset) => *offset,
            None => offsets[offsets.len() - 1] + (index - (offsets.len() - 1)),
        };
        for cut in &mut self.closed_cuts {
            cut.instruction_index = remap(cut.instruction_index);
        }
        for check in &mut self.scalar_checks {
            check.instruction_index = remap(check.instruction_index);
        }
        self.operation_count = instruction_count(&instructions);
        self.instructions = instructions.into_boxed_slice();
        self.optimization_stages
            .push("optimization:circuit.exact-local@1".into());
        Ok(self)
    }

    /// Return late passes actually executed on this projection.
    #[must_use]
    pub fn optimization_stages(&self) -> &[String] {
        &self.optimization_stages
    }

    pub(crate) fn inherit_optimization_stages(mut self, source: &Self) -> Self {
        self.optimization_stages
            .clone_from(&source.optimization_stages);
        self
    }
}

/// Hash-consed scalar arena that can append folded angle expressions.
struct ScalarArena {
    nodes: Vec<CircuitScalarNode>,
    ids: HashMap<CircuitScalarNode, CircuitScalarId>,
}

impl ScalarArena {
    fn new(nodes: Vec<CircuitScalarNode>) -> Self {
        let mut ids = HashMap::with_capacity(nodes.len());
        for (index, node) in nodes.iter().enumerate() {
            if let Ok(index) = u32::try_from(index) {
                ids.entry(node.clone()).or_insert(CircuitScalarId(index));
            }
        }
        Self { nodes, ids }
    }

    fn into_nodes(self) -> Vec<CircuitScalarNode> {
        self.nodes
    }

    fn intern(&mut self, node: CircuitScalarNode) -> Option<CircuitScalarId> {
        if let Some(id) = self.ids.get(&node) {
            return Some(*id);
        }
        let id = CircuitScalarId(u32::try_from(self.nodes.len()).ok()?);
        self.nodes.push(node.clone());
        self.ids.insert(node, id);
        Some(id)
    }

    fn node(&self, id: CircuitScalarId) -> Option<&CircuitScalarNode> {
        self.nodes.get(usize::try_from(id.0).ok()?)
    }

    /// Fold a Real-typed literal expression built from literals and +,-,*,neg.
    fn literal(&self, id: CircuitScalarId, depth: usize) -> Option<f64> {
        if depth > MAX_FOLD_DEPTH {
            return None;
        }
        match self.node(id)? {
            CircuitScalarNode::RealBits(bits) => Some(f64::from_bits(*bits)),
            #[expect(clippy::cast_precision_loss, reason = "only exact small integers fold")]
            CircuitScalarNode::Int(value) if value.unsigned_abs() <= 1 << 53 => Some(*value as f64),
            CircuitScalarNode::Apply {
                opcode, operands, ..
            } => {
                let value = |index: usize| self.literal(*operands.get(index)?, depth + 1);
                let result = match (opcode.as_str(), operands.len()) {
                    ("classical.neg", 1) => -value(0)?,
                    ("classical.add", 2) => value(0)? + value(1)?,
                    ("classical.sub", 2) => value(0)? - value(1)?,
                    ("classical.mul", 2) => value(0)? * value(1)?,
                    _ => return None,
                };
                result.is_finite().then_some(result)
            }
            _ => None,
        }
    }

    fn is_real(&self, id: CircuitScalarId) -> bool {
        self.node(id)
            .is_some_and(|node| node.scalar_type() == CircuitScalarType::Real)
    }

    /// Return the exact sum of two angles, folding literals when possible.
    fn add(&mut self, left: CircuitScalarId, right: CircuitScalarId) -> Option<Sum> {
        if !self.is_real(left) || !self.is_real(right) {
            return None;
        }
        match (self.literal(left, 0), self.literal(right, 0)) {
            (Some(a), Some(b)) => {
                let total = a + b;
                if total == 0.0 {
                    return Some(Sum::Zero);
                }
                self.intern(CircuitScalarNode::RealBits(total.to_bits()))
                    .map(Sum::Angle)
            }
            _ => self
                .intern(CircuitScalarNode::Apply {
                    opcode: "classical.add".to_owned(),
                    result_type: CircuitScalarType::Real,
                    attributes: Box::new([]),
                    operands: vec![left, right].into_boxed_slice(),
                })
                .map(Sum::Angle),
        }
    }
}

enum Sum {
    Zero,
    Angle(CircuitScalarId),
}

struct RewriteWork {
    remaining: u64,
    // Retain both owners so pointer identities cannot be reused during rewriting.
    definitions: HashMap<usize, (CircuitDefinition, CircuitDefinition)>,
}

impl RewriteWork {
    fn new(remaining: u64) -> Self {
        Self {
            remaining,
            definitions: HashMap::new(),
        }
    }

    fn definition(
        &mut self,
        source: &CircuitDefinition,
        depth: usize,
    ) -> Result<CircuitDefinition, StaticCircuitError> {
        if let Some((_, result)) = self.definitions.get(&source.identity()) {
            return Ok(result.clone());
        }
        // Do not force symbolic arithmetic recipes to allocate gates early.
        let Some(body) = source.retained_body() else {
            return Ok(source.clone());
        };
        if body.len() as u64 > self.remaining {
            return Err(StaticCircuitError::Unsupported(
                "circuit optimization work budget exceeded".into(),
            ));
        }
        let mut arena = ScalarArena::new(source.scalars().to_vec());
        let (instructions, _) = simplify(
            body.to_vec(),
            &HashSet::new(),
            depth + 1,
            self,
            &mut arena,
            None,
        )?;
        let result = CircuitDefinition::body(
            source.width(),
            instructions,
            arena.into_nodes().into_boxed_slice(),
        );
        self.definitions
            .insert(source.identity(), (source.clone(), result.clone()));
        Ok(result)
    }
}

fn contains_wires(value: &super::StaticCircuitValue) -> bool {
    use super::StaticCircuitValue as Value;
    match value {
        Value::Wires(_) => true,
        Value::Aggregate(items) => items.iter().any(contains_wires),
        Value::Record(fields) => fields.iter().any(|(_, item)| contains_wires(item)),
        Value::Unit | Value::Portable(_) | Value::Scalar(_) => false,
    }
}

/// Wire relabeling that replaces uncontrolled, unconditioned SWAP gates.
///
/// `logical` maps each wire named by the original instruction stream to the
/// wire currently holding its state; `physical` is the inverse. Relabeling a
/// gate's operands is exact, so a SWAP becomes a bookkeeping update.
#[derive(Default)]
struct WirePermutation {
    logical: HashMap<u64, u64>,
    physical: HashMap<u64, u64>,
}

impl WirePermutation {
    fn get(&self, wire: u64) -> u64 {
        self.logical.get(&wire).copied().unwrap_or(wire)
    }

    fn holder(&self, wire: u64) -> u64 {
        self.physical.get(&wire).copied().unwrap_or(wire)
    }

    fn record(&mut self, logical: u64, physical: u64) {
        if logical == physical {
            self.logical.remove(&logical);
            self.physical.remove(&physical);
        } else {
            self.logical.insert(logical, physical);
            self.physical.insert(physical, logical);
        }
    }

    fn swap(&mut self, left: u64, right: u64) {
        let (a, b) = (self.get(left), self.get(right));
        self.record(left, b);
        self.record(right, a);
    }

    fn relabel(&self, instruction: &mut StaticInstruction) {
        if self.logical.is_empty() {
            return;
        }
        for wire in instruction
            .qubits
            .iter_mut()
            .chain(instruction.controls.iter_mut())
        {
            *wire = self.get(*wire);
        }
    }

    /// Emit SWAPs that return every relabeled state to its recorded wire.
    fn restore(&mut self) -> Vec<StaticInstruction> {
        let mut pending = self.logical.keys().copied().collect::<Vec<_>>();
        pending.sort_unstable();
        let mut swaps = Vec::new();
        for wire in pending {
            let current = self.get(wire);
            if current == wire {
                continue;
            }
            // The state recorded for `wire` sits on `current`; the state now
            // on `wire` belongs to `displaced` and moves to `current`.
            let displaced = self.holder(wire);
            swaps.push(StaticInstruction {
                opcode: "swap".to_owned(),
                qubits: Box::new([wire, current]),
                parameters: Box::new([]),
                controls: Box::new([]),
                condition: None,
                regions: Box::new([]),
            });
            self.record(wire, wire);
            self.record(displaced, current);
        }
        debug_assert!(self.logical.is_empty() && self.physical.is_empty());
        swaps
    }
}

fn elidable_swap(instruction: &StaticInstruction) -> bool {
    instruction.opcode == "swap"
        && instruction.qubits.len() == 2
        && instruction.qubits[0] != instruction.qubits[1]
        && instruction.parameters.is_empty()
        && instruction.controls.is_empty()
        && instruction.condition.is_none()
        && instruction.regions.is_empty()
}

/// Simplify one block. `relabel_swaps` is `Some(restore)` for the top level,
/// where SWAPs become relabeling and `restore` keeps live result wires.
fn simplify(
    input: Vec<StaticInstruction>,
    barriers: &HashSet<usize>,
    depth: usize,
    work: &mut RewriteWork,
    arena: &mut ScalarArena,
    relabel_swaps: Option<bool>,
) -> Result<(Vec<StaticInstruction>, Vec<usize>), StaticCircuitError> {
    if depth >= 256 {
        return Err(StaticCircuitError::Unsupported(
            "circuit optimization region depth exceeded".into(),
        ));
    }
    work.remaining = work
        .remaining
        .checked_sub(input.len() as u64)
        .ok_or_else(|| {
            StaticCircuitError::Unsupported("circuit optimization work budget exceeded".into())
        })?;
    let mut output: Vec<Option<StaticInstruction>> = Vec::with_capacity(input.len());
    let mut offsets = Vec::with_capacity(input.len() + 1);
    let mut floor = 0;
    let mut permutation = WirePermutation::default();
    for (index, mut instruction) in input.into_iter().enumerate() {
        let opaque = !instruction.regions.is_empty();
        if relabel_swaps.is_some() && (opaque || barriers.contains(&index)) {
            // Cuts and structured regions observe wires by their recorded names.
            for swap in permutation.restore() {
                place(&mut output, swap, floor, arena);
            }
        }
        if barriers.contains(&index) {
            floor = output.len();
        }
        offsets.push(output.len());
        if relabel_swaps.is_some() {
            if elidable_swap(&instruction) {
                permutation.swap(instruction.qubits[0], instruction.qubits[1]);
                continue;
            }
            permutation.relabel(&mut instruction);
        }
        for block in &mut instruction.regions {
            let (nested, _) = simplify(
                std::mem::take(&mut block.instructions).into_vec(),
                &HashSet::new(),
                depth + 1,
                work,
                arena,
                None,
            )?;
            block.instructions = nested.into_boxed_slice();
        }
        for parameter in &mut instruction.parameters {
            if let CircuitParameter::Definition(definition) = parameter {
                *definition = work.definition(definition, depth)?;
            }
        }
        place(&mut output, instruction, floor, arena);
        if opaque {
            // Structured instructions may touch wires outside their operand
            // list, so nothing is moved across them.
            floor = output.len();
        }
    }
    // A terminal cut or live result wires observe the recorded wire names.
    if relabel_swaps == Some(true) || (relabel_swaps.is_some() && barriers.contains(&offsets.len()))
    {
        for swap in permutation.restore() {
            place(&mut output, swap, floor, arena);
        }
    }
    offsets.push(output.len());
    // Map slot positions, including removed slots, onto compacted indices.
    let mut compacted = Vec::with_capacity(output.len() + 1);
    let mut survivors = 0;
    for slot in &output {
        compacted.push(survivors);
        survivors += usize::from(slot.is_some());
    }
    compacted.push(survivors);
    let offsets = offsets.into_iter().map(|slot| compacted[slot]).collect();
    Ok((output.into_iter().flatten().collect(), offsets))
}

/// Per-wire algebra an instruction acts in; equal Z or X roles commute.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Role {
    /// Diagonal in the computational basis (phases and controls).
    Z,
    /// Within the commutative algebra generated by Pauli X.
    X,
    /// Anything else, including measurement and opaque composites.
    Other,
}

fn target_roles(instruction: &StaticInstruction) -> &'static [Role] {
    match (instruction.opcode.as_str(), instruction.qubits.len()) {
        ("z" | "s" | "sdg" | "t" | "tdg" | "rz" | "p", 1) => &[Role::Z],
        ("cz" | "cp" | "rzz", 2) => &[Role::Z, Role::Z],
        ("x" | "rx", 1) => &[Role::X],
        ("cx", 2) => &[Role::Z, Role::X],
        ("ccx", 3) => &[Role::Z, Role::Z, Role::X],
        _ => &[],
    }
}

fn role(instruction: &StaticInstruction, wire: u64) -> Role {
    if instruction.condition.is_some() || !instruction.regions.is_empty() {
        return Role::Other;
    }
    if let Some(position) = instruction.qubits.iter().position(|item| *item == wire) {
        return target_roles(instruction)
            .get(position)
            .copied()
            .unwrap_or(Role::Other);
    }
    if instruction.controls.contains(&wire) {
        return Role::Z;
    }
    Role::Other
}

fn wires(instruction: &StaticInstruction) -> impl Iterator<Item = u64> + '_ {
    instruction
        .qubits
        .iter()
        .chain(instruction.controls.iter())
        .copied()
}

fn overlaps(left: &StaticInstruction, right: &StaticInstruction) -> bool {
    wires(left).any(|wire| wires(right).any(|other| other == wire))
}

fn commutes(left: &StaticInstruction, right: &StaticInstruction) -> bool {
    wires(left).all(|wire| {
        if !wires(right).any(|other| other == wire) {
            return true;
        }
        let kind = role(left, wire);
        kind != Role::Other && kind == role(right, wire)
    })
}

/// Rewrite produced by combining an earlier instruction with a new one.
enum Combined {
    /// Both instructions cancel.
    Cancel,
    /// The new instruction is redundant after the earlier one.
    DropNew,
    /// Both become this single instruction.
    Replace(StaticInstruction),
}

fn same_operands(left: &StaticInstruction, right: &StaticInstruction) -> bool {
    if left.condition.is_some()
        || right.condition.is_some()
        || !left.regions.is_empty()
        || !right.regions.is_empty()
    {
        return false;
    }
    let mut a = left.controls.to_vec();
    let mut b = right.controls.to_vec();
    a.sort_unstable();
    b.sort_unstable();
    if a != b {
        return false;
    }
    let symmetric = matches!(left.opcode.as_str(), "cz" | "cp" | "rzz" | "swap");
    left.qubits == right.qubits
        || (symmetric
            && left.qubits.len() == 2
            && right.qubits.len() == 2
            && left.qubits[0] == right.qubits[1]
            && left.qubits[1] == right.qubits[0])
}

fn combine(
    earlier: &StaticInstruction,
    later: &StaticInstruction,
    arena: &mut ScalarArena,
) -> Option<Combined> {
    if !same_operands(earlier, later) {
        return None;
    }
    let (a, b) = (earlier.opcode.as_str(), later.opcode.as_str());
    if earlier.parameters.is_empty() && later.parameters.is_empty() {
        if inverse(a) == Some(b) {
            return Some(Combined::Cancel);
        }
        if a == "reset" && b == "reset" {
            return Some(Combined::DropNew);
        }
        let fused = match (a, b) {
            ("t", "t") | ("z", "sdg") | ("sdg", "z") => "s",
            ("tdg", "tdg") | ("z", "s") | ("s", "z") => "sdg",
            ("s", "s") | ("sdg", "sdg") => "z",
            _ => return None,
        };
        let mut result = later.clone();
        fused.clone_into(&mut result.opcode);
        return Some(Combined::Replace(result));
    }
    if a != b || !is_rotation(a) {
        return None;
    }
    let ([CircuitParameter::Scalar(left)], [CircuitParameter::Scalar(right)]) =
        (&*earlier.parameters, &*later.parameters)
    else {
        return None;
    };
    match arena.add(*left, *right)? {
        Sum::Zero => Some(Combined::Cancel),
        Sum::Angle(angle) => {
            let mut result = later.clone();
            result.parameters = vec![CircuitParameter::Scalar(angle)].into_boxed_slice();
            Some(Combined::Replace(result))
        }
    }
}

/// Single-angle rotations that compose additively and vanish at zero.
fn is_rotation(opcode: &str) -> bool {
    matches!(opcode, "rz" | "p" | "rx" | "ry" | "cp" | "rzz")
}

fn is_identity(instruction: &StaticInstruction, arena: &ScalarArena) -> bool {
    is_rotation(&instruction.opcode)
        && matches!(
            &*instruction.parameters,
            [CircuitParameter::Scalar(angle)] if arena.literal(*angle, 0) == Some(0.0)
        )
}

/// Latest live slot at or above `floor` that shares a wire with `instruction`.
fn previous_on_wires(
    output: &[Option<StaticInstruction>],
    instruction: &StaticInstruction,
    below: usize,
    floor: usize,
) -> Option<usize> {
    let lowest = floor.max(below.saturating_sub(LOOKBACK_WINDOW));
    (lowest..below).rev().find(|&slot| {
        output[slot]
            .as_ref()
            .is_some_and(|earlier| overlaps(earlier, instruction))
    })
}

/// Insert one instruction, rewriting it against commuting earlier gates.
///
/// Every rewrite removes at least one live instruction, so the loop ends.
fn place(
    output: &mut Vec<Option<StaticInstruction>>,
    mut instruction: StaticInstruction,
    floor: usize,
    arena: &mut ScalarArena,
) {
    'retry: loop {
        if instruction.condition.is_some() || !instruction.regions.is_empty() {
            break;
        }
        if is_identity(&instruction, arena) {
            return;
        }
        let lowest = floor.max(output.len().saturating_sub(LOOKBACK_WINDOW));
        for slot in (lowest..output.len()).rev() {
            let Some(earlier) = output[slot].as_ref() else {
                continue;
            };
            if !overlaps(earlier, &instruction) {
                continue;
            }
            match combine(earlier, &instruction, arena) {
                Some(Combined::Cancel) => {
                    output[slot] = None;
                    return;
                }
                Some(Combined::DropNew) => return,
                Some(Combined::Replace(result)) => {
                    output[slot] = None;
                    instruction = result;
                    continue 'retry;
                }
                None => {}
            }
            if !commutes(earlier, &instruction) {
                break;
            }
        }
        // H X H = Z and H Z H = X on identical operands with no gate between.
        if instruction.opcode == "h" && instruction.parameters.is_empty() {
            let end = output.len();
            if let Some(middle) = previous_on_wires(output, &instruction, end, floor)
                && let Some(first) = previous_on_wires(output, &instruction, middle, floor)
            {
                let replacement = match (&output[middle], &output[first]) {
                    (Some(b), Some(a))
                        if a.opcode == "h"
                            && a.parameters.is_empty()
                            && b.parameters.is_empty()
                            && same_operands(a, &instruction)
                            && same_operands(b, &instruction) =>
                    {
                        match b.opcode.as_str() {
                            "x" => Some("z"),
                            "z" => Some("x"),
                            _ => None,
                        }
                    }
                    _ => None,
                };
                if let Some(opcode) = replacement {
                    output[middle] = None;
                    output[first] = None;
                    opcode.clone_into(&mut instruction.opcode);
                    continue 'retry;
                }
            }
        }
        break;
    }
    output.push(Some(instruction));
}

fn inverse(opcode: &str) -> Option<&str> {
    match opcode {
        "h" | "x" | "y" | "z" | "cx" | "cz" | "ccx" | "swap" => Some(opcode),
        "s" => Some("sdg"),
        "sdg" => Some("s"),
        "t" => Some("tdg"),
        "tdg" => Some("t"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn on(op: &str, qubits: Vec<u64>, controls: Vec<u64>) -> StaticInstruction {
        StaticInstruction::test_new(op, qubits, vec![], controls, None, vec![])
    }

    fn gate(op: &str) -> StaticInstruction {
        on(op, vec![0], vec![])
    }

    fn rotation(op: &str, qubits: Vec<u64>, angle: u32) -> StaticInstruction {
        let angle = vec![CircuitParameter::Scalar(CircuitScalarId(angle))];
        StaticInstruction::test_new(op, qubits, angle, vec![], None, vec![])
    }

    /// Simplify one block with no scalars; `relabel` enables SWAP relabeling.
    fn run(
        input: Vec<StaticInstruction>,
        barriers: &[usize],
        relabel: Option<bool>,
    ) -> (Vec<StaticInstruction>, Vec<usize>) {
        let barriers = barriers.iter().copied().collect();
        let mut arena = ScalarArena::new(Vec::new());
        simplify(
            input,
            &barriers,
            0,
            &mut RewriteWork::new(100),
            &mut arena,
            relabel,
        )
        .unwrap()
    }

    fn opcodes(instructions: &[StaticInstruction]) -> Vec<&str> {
        instructions.iter().map(StaticInstruction::opcode).collect()
    }

    #[test]
    fn local_rules_cancel_fuse_and_commute_exactly() {
        let conditioned = StaticInstruction::test_new(
            "z",
            vec![0],
            vec![],
            vec![],
            Some(CircuitScalarId(0)),
            vec![],
        );
        let region = StaticInstruction::test_new(
            "if",
            vec![],
            vec![],
            vec![],
            Some(CircuitScalarId(0)),
            vec![super::super::StaticBlock::test_new(vec![gate("x")])],
        );
        let cases: Vec<(Vec<StaticInstruction>, Vec<&str>)> = vec![
            (
                vec![gate("h"), gate("z"), gate("h"), gate("t"), gate("t")],
                vec!["x", "s"],
            ),
            (vec![gate("s"), gate("t"), gate("t"), gate("z")], vec![]),
            // CX controls commute with T; CX targets commute with X and controls with CZ.
            (
                vec![
                    on("cx", vec![0, 1], vec![]),
                    gate("t"),
                    on("x", vec![2], vec![]),
                    on("cx", vec![0, 1], vec![]),
                ],
                vec!["t", "x"],
            ),
            (
                vec![
                    on("cx", vec![0, 1], vec![]),
                    on("x", vec![1], vec![]),
                    on("cz", vec![0, 2], vec![]),
                    on("cx", vec![0, 1], vec![]),
                ],
                vec!["x", "cz"],
            ),
            // Z S = Sdg and Sdg S = I under identical controls only.
            (
                vec![
                    on("z", vec![1], vec![0]),
                    on("s", vec![1], vec![0]),
                    on("s", vec![1], vec![0]),
                    on("s", vec![1], vec![3]),
                ],
                vec!["s"],
            ),
            // Nothing moves across H, measurement, conditioned gates or regions.
            (
                vec![
                    on("cx", vec![0, 1], vec![]),
                    gate("h"),
                    on("cx", vec![0, 1], vec![]),
                ],
                vec!["cx", "h", "cx"],
            ),
            (
                vec![gate("z"), gate("measure"), gate("z")],
                vec!["z", "measure", "z"],
            ),
            (vec![gate("z"), conditioned, gate("z")], vec!["z", "z", "z"]),
            (vec![gate("x"), region, gate("x")], vec!["x", "if", "x"]),
        ];
        for (input, expected) in cases {
            assert_eq!(opcodes(&run(input, &[], None).0), expected);
        }
        // A barrier before index 2 keeps T/Tdg and remaps offsets.
        let (result, offsets) = run(
            vec![gate("t"), gate("x"), gate("tdg"), gate("x")],
            &[2],
            None,
        );
        assert_eq!(opcodes(&result), ["t", "x", "tdg", "x"]);
        assert_eq!(offsets, [0, 1, 2, 3, 4]);
    }

    #[test]
    fn rotations_merge_exactly_and_vanish_at_zero() {
        let quarter = std::f64::consts::FRAC_PI_4;
        let mut arena = ScalarArena::new(vec![
            CircuitScalarNode::RealBits(quarter.to_bits()),
            CircuitScalarNode::Apply {
                opcode: "classical.neg".to_owned(),
                result_type: CircuitScalarType::Real,
                attributes: Box::new([]),
                operands: Box::new([CircuitScalarId(0)]),
            },
            CircuitScalarNode::Parameter {
                name: "theta".to_owned(),
                path: Box::new([]),
                scalar_type: CircuitScalarType::Real,
            },
        ]);
        let input = vec![
            rotation("cp", vec![0, 1], 0),
            rotation("rz", vec![2], 2),
            rotation("cp", vec![1, 0], 1),
            rotation("rz", vec![2], 2),
            rotation("rz", vec![2], 0),
        ];
        let mut work = RewriteWork::new(100);
        let (result, _) = simplify(input, &HashSet::new(), 0, &mut work, &mut arena, None).unwrap();
        assert_eq!(opcodes(&result), ["rz"]);
        let [CircuitParameter::Scalar(angle)] = &*result[0].parameters else {
            panic!("angle");
        };
        let Some(CircuitScalarNode::Apply {
            opcode, operands, ..
        }) = arena.node(*angle)
        else {
            panic!("symbolic sum");
        };
        assert_eq!(opcode, "classical.add");
        assert_eq!(arena.literal(operands[1], 0), Some(quarter));
    }

    #[test]
    fn shared_bodies_are_rewritten_once_with_a_shared_budget() {
        let source =
            CircuitDefinition::body(1, vec![gate("h"), gate("z"), gate("h")], Box::new([]));
        let call = StaticInstruction::test_new(
            "box:helper",
            vec![0],
            vec![CircuitParameter::Definition(source.clone())],
            vec![],
            None,
            vec![],
        );
        let mut arena = ScalarArena::new(Vec::new());
        let mut work = RewriteWork::new(5);
        let calls = vec![call.clone(), call];
        let (result, _) = simplify(
            calls.clone(),
            &HashSet::new(),
            0,
            &mut work,
            &mut arena,
            None,
        )
        .unwrap();
        let [CircuitParameter::Definition(first)] = &*result[0].parameters else {
            panic!("definition");
        };
        let [CircuitParameter::Definition(second)] = &*result[1].parameters else {
            panic!("definition");
        };
        assert_eq!(first.identity(), second.identity());
        assert_eq!(first.instructions().unwrap()[0].opcode(), "x");
        assert_eq!(source.instructions().unwrap().len(), 3);
        assert_eq!(work.remaining, 0);
        let mut work = RewriteWork::new(4);
        assert!(simplify(calls, &HashSet::new(), 0, &mut work, &mut arena, None).is_err());
    }

    #[test]
    fn swaps_become_relabeling_and_restore_live_wires() {
        let measure = |wire: u64, slot: u64| {
            let slot = vec![CircuitParameter::MeasurementIndex(slot)];
            StaticInstruction::test_new("measure", vec![wire], slot, vec![], None, vec![])
        };
        let input = vec![
            on("swap", vec![0, 1], vec![]),
            gate("h"),
            on("x", vec![2], vec![0]),
            measure(0, 0),
            measure(1, 1),
        ];
        let (result, _) = run(input, &[], Some(false));
        let wires = result
            .iter()
            .map(|item| [item.qubits(), item.controls()].concat())
            .collect::<Vec<_>>();
        assert_eq!(opcodes(&result), ["h", "x", "measure", "measure"]);
        assert_eq!(wires, [vec![1], vec![2, 1], vec![1], vec![0]]);
        // Controlled swaps stay, and live result wires are restored.
        let input = vec![
            on("swap", vec![0, 1], vec![]),
            on("swap", vec![1, 2], vec![3]),
            gate("x"),
        ];
        let (result, _) = run(input, &[], Some(true));
        assert_eq!(opcodes(&result), ["swap", "x", "swap"]);
        assert_eq!(result[0].qubits(), [0, 2]);
        assert_eq!(result[1].qubits(), [1]);
        assert_eq!(result[2].qubits(), [0, 1]);
        // Any swap network plus its restoring swaps is the original permutation.
        let index = |wire: u64| usize::try_from(wire).unwrap();
        let mut seed = 0x9e37_79b9_u64;
        for _ in 0..200 {
            let mut input = Vec::new();
            let mut expected = [0_usize, 1, 2, 3, 4];
            for _ in 0..8 {
                seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                let (a, b) = ((seed >> 33) % 5, (seed >> 43) % 5);
                if a != b {
                    input.push(on("swap", vec![a, b], vec![]));
                    expected.swap(index(a), index(b));
                }
            }
            let mut actual = [0_usize, 1, 2, 3, 4];
            for swap in run(input, &[], Some(true)).0 {
                actual.swap(index(swap.qubits()[0]), index(swap.qubits()[1]));
            }
            assert_eq!(actual, expected);
        }
    }
}
