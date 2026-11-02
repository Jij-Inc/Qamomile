//! Bounded, transient display projections over verified structured `QCore`.

use std::collections::HashMap;

use thiserror::Error;

use crate::operations::LocallyCheckedOperations;
use crate::static_values::{self, StaticScalar, StaticValue};
use crate::types::{TypeArena, TypeId};
use crate::wire;

/// Maximum nodes and values in one derived display.
pub const MAX_VISUALIZATION_ITEMS: usize = 16_384;
const MAX_DEPTH: usize = 32;
const MAX_TEXT: usize = 256;
const MAX_WORK: usize = 1_000_000;

/// Presentation of a structured loop, independent of executable lowering.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LoopDisplay {
    /// Retain only the loop interface and range.
    Folded,
    /// Retain one symbolic body and normal-completion region.
    Body,
    /// Repeat a finite, statically known, normally yielding body.
    Expanded,
}

/// Closed node categories consumed by layout and renderers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DisplayKind {
    /// A quantum operation, including measurement and arithmetic.
    Quantum,
    /// A pure classical expression or literal.
    Classical,
    /// A structural reinterpretation, pack, or unpack.
    Structure,
    /// A callable interface and optional body.
    Call,
    /// A structured loop interface.
    Loop,
    /// One concrete loop iteration.
    Iteration,
    /// Mutually exclusive branch alternatives.
    Branch,
    /// One region interface and its yielded values.
    Region,
    /// An explicit return, yield, break, or continue.
    Exit,
}

/// One immutable typed display port. IDs identify SSA instances, not qubits.
#[derive(Clone, Debug)]
pub struct DisplayValue {
    /// Unique identity within this display only.
    pub id: usize,
    /// Bounded source hint or expression.
    pub label: String,
    /// Source-style logical type, including interpretation and availability.
    pub type_label: String,
    /// Whether the verified type contains coherent data.
    pub quantum: bool,
    /// Latest native availability, or Value for unstaged data.
    pub stage: String,
    /// Native coherent width, absent for classical or symbolic shapes.
    pub width: Option<u64>,
}

/// One display region or operation with explicit dataflow interfaces.
#[derive(Clone, Debug)]
pub struct DisplayNode {
    /// Closed visual category.
    pub kind: DisplayKind,
    /// Original `QCore` opcode or region/iteration marker.
    pub opcode: String,
    /// Bounded display label.
    pub label: String,
    /// IDs of ordered input ports.
    pub inputs: Vec<usize>,
    /// IDs of ordered output ports.
    pub outputs: Vec<usize>,
    /// Structured children; branch alternatives remain separate regions.
    pub children: Vec<Self>,
    /// Original source line, absent for synthetic boundaries.
    pub line: Option<u32>,
}

/// A transient view, derived from one verified source and never serialized as IR.
#[derive(Clone, Debug)]
pub struct AlgorithmDisplay {
    /// Selected callable identity.
    pub entrypoint: String,
    /// Flat port table in deterministic encounter order.
    pub values: Vec<DisplayValue>,
    /// Ordered entrypoint input IDs.
    pub inputs: Vec<usize>,
    /// Ordered root body nodes, including the explicit return.
    pub nodes: Vec<DisplayNode>,
}

/// A display request cannot reinterpret unsupported or unbounded behavior.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum VisualizationError {
    /// Output, depth, expression, or traversal allowance was exceeded.
    #[error("Q7401: algorithm visualization exceeded its bounded work or output limit")]
    Budget,
    /// A loop cannot be faithfully unrolled with the requested presentation.
    #[error("Q7402: cannot expand loop: {0}; use loops='body' or 'folded'")]
    Expansion(String),
    /// A display option lies outside the supported bounds.
    #[error("Q7403: invalid visualization option: {0}")]
    Option(String),
}

#[derive(Clone, Default)]
struct Environment {
    ids: HashMap<u64, usize>,
    constants: HashMap<u64, Option<StaticValue>>,
}

struct Projector<'a> {
    arena: &'a TypeArena,
    operations: &'a LocallyCheckedOperations,
    functions: HashMap<&'a str, &'a wire::FunctionProto>,
    loops: LoopDisplay,
    inline_depth: usize,
    limit: usize,
    work: usize,
    nodes: usize,
    values: Vec<DisplayValue>,
    type_labels: HashMap<TypeId, String>,
}

pub(crate) fn project(
    package: &wire::QCorePackageProto,
    arena: &TypeArena,
    operations: &LocallyCheckedOperations,
    loops: LoopDisplay,
    inline_depth: usize,
    limit: usize,
) -> Result<AlgorithmDisplay, VisualizationError> {
    if limit == 0 || limit > MAX_VISUALIZATION_ITEMS || inline_depth > MAX_DEPTH {
        return Err(VisualizationError::Option(format!(
            "max_nodes must be 1..={MAX_VISUALIZATION_ITEMS} and inline_depth at most {MAX_DEPTH}"
        )));
    }
    let functions = package
        .functions
        .iter()
        .map(|function| (function.symbol.as_str(), function))
        .collect();
    let mut projector = Projector {
        arena,
        operations,
        functions,
        loops,
        inline_depth,
        limit,
        work: 0,
        nodes: 0,
        values: Vec::new(),
        type_labels: HashMap::new(),
    };
    let function = projector.functions[package.entrypoint.as_str()];
    let mut environment = Environment::default();
    let mut inputs = Vec::new();
    for parameter in &function.parameters {
        let value = parameter.value.as_ref().expect("verified parameter");
        let binding = package
            .const_bindings
            .iter()
            .find(|binding| binding.name == parameter.name)
            .and_then(|binding| binding.value.as_ref())
            .and_then(static_values::value_literal);
        let id = projector.value(function, value, &parameter.name)?;
        environment.ids.insert(value.id, id);
        environment.constants.insert(value.id, binding);
        inputs.push(id);
    }
    let (nodes, _) = projector.block(
        function,
        function.body.as_ref().expect("verified body"),
        &mut environment,
        0,
        0,
    )?;
    Ok(AlgorithmDisplay {
        entrypoint: package.entrypoint.clone(),
        values: projector.values,
        inputs,
        nodes,
    })
}

impl Projector<'_> {
    fn charge(&mut self, work: usize) -> Result<(), VisualizationError> {
        self.work = self
            .work
            .checked_add(work)
            .ok_or(VisualizationError::Budget)?;
        if self.work > MAX_WORK {
            return Err(VisualizationError::Budget);
        }
        Ok(())
    }

    fn node(&mut self, depth: usize) -> Result<(), VisualizationError> {
        self.charge(1)?;
        self.nodes += 1;
        if self.nodes > self.limit || depth > MAX_DEPTH {
            return Err(VisualizationError::Budget);
        }
        Ok(())
    }

    fn value(
        &mut self,
        function: &wire::FunctionProto,
        value: &wire::ValueProto,
        label: &str,
    ) -> Result<usize, VisualizationError> {
        self.charge(1)?;
        if self.values.len() >= self.limit {
            return Err(VisualizationError::Budget);
        }
        let root = self
            .operations
            .value_types(&function.symbol)
            .expect("verified types")[&value.id];
        let quantum = self.arena.is_quantum(root);
        let stage = self
            .arena
            .latest_availability(root)
            .map_or("Value", crate::types::Stage::as_str);
        let id = self.values.len();
        let type_label = self
            .type_labels
            .entry(root)
            .or_insert_with(|| {
                let mut remaining = MAX_TEXT;
                type_label(
                    value.r#type.as_ref().expect("verified type"),
                    0,
                    &mut remaining,
                )
            })
            .clone();
        self.values.push(DisplayValue {
            id,
            label: bounded(label),
            type_label,
            quantum,
            stage: stage.to_owned(),
            width: if quantum {
                self.arena
                    .coherent_width(root)
                    .map_err(|_| VisualizationError::Budget)?
            } else {
                None
            },
        });
        Ok(id)
    }

    fn fork(&mut self, environment: &Environment) -> Result<Environment, VisualizationError> {
        self.charge(environment.ids.len() + environment.constants.len())?;
        Ok(environment.clone())
    }

    fn block(
        &mut self,
        function: &wire::FunctionProto,
        block: &wire::BlockProto,
        environment: &mut Environment,
        depth: usize,
        calls: usize,
    ) -> Result<(Vec<DisplayNode>, Vec<usize>), VisualizationError> {
        if depth > MAX_DEPTH {
            return Err(VisualizationError::Budget);
        }
        let mut nodes = Vec::new();
        for operation in &block.operations {
            nodes.push(self.operation(function, operation, environment, depth, calls)?);
        }
        let terminator = block.terminator.as_ref().expect("verified terminator");
        let outputs = terminator
            .operands
            .iter()
            .map(|id| environment.ids[id])
            .collect::<Vec<_>>();
        self.node(depth)?;
        nodes.push(DisplayNode {
            kind: DisplayKind::Exit,
            opcode: terminator.kind.clone(),
            label: terminator.kind.clone(),
            inputs: outputs.clone(),
            outputs: Vec::new(),
            children: Vec::new(),
            line: terminator.span.as_ref().map(|span| span.start_line),
        });
        Ok((nodes, outputs))
    }

    #[allow(clippy::too_many_lines)] // Exhaustive presentation dispatch keeps categories together.
    fn operation(
        &mut self,
        function: &wire::FunctionProto,
        operation: &wire::OperationProto,
        environment: &mut Environment,
        depth: usize,
        calls: usize,
    ) -> Result<DisplayNode, VisualizationError> {
        self.node(depth)?;
        self.charge(operation.operands.len() + operation.results.len())?;
        let inputs = operation
            .operands
            .iter()
            .map(|id| environment.ids[id])
            .collect::<Vec<_>>();
        let opcode = operation.opcode.as_str();
        let labels = inputs
            .iter()
            .map(|id| self.values[*id].label.as_str())
            .collect::<Vec<_>>();
        let mut label = expression(operation, &labels);
        let mut children = Vec::new();
        let kind = match opcode {
            "qcore.for" => {
                let induction = string_attr(operation, "induction_name").unwrap_or("index");
                label = bounded(&format!(
                    "for {induction} in range({}, {}, {})",
                    labels[0], labels[1], labels[2]
                ));
                children = self.loop_children(function, operation, environment, depth, calls)?;
                DisplayKind::Loop
            }
            "qcore.while" => {
                "while".clone_into(&mut label);
                for (index, region) in operation.regions.iter().enumerate() {
                    children.push(self.region(
                        function,
                        &region.blocks[0],
                        &operation.operands,
                        environment,
                        depth + 1,
                        calls,
                        ["condition", "body", "else"][index],
                    )?);
                }
                DisplayKind::Loop
            }
            "qcore.if" => {
                label = bounded(&format!("if {}", labels[0]));
                for (index, region) in operation.regions.iter().enumerate() {
                    children.push(self.region(
                        function,
                        &region.blocks[0],
                        &operation.operands[1..],
                        environment,
                        depth + 1,
                        calls,
                        if index == 0 { "true" } else { "false" },
                    )?);
                }
                DisplayKind::Branch
            }
            "qcore.call" => {
                let callee = string_attr(operation, "callee").expect("verified callee");
                label = bounded(callee.rsplit('.').next().unwrap_or(callee));
                let target = self.functions[callee];
                if calls < self.inline_depth && !target.external {
                    let mut inner = Environment::default();
                    for (parameter, operand) in target.parameters.iter().zip(&operation.operands) {
                        let value = parameter.value.as_ref().expect("verified parameter");
                        inner.ids.insert(value.id, environment.ids[operand]);
                        inner.constants.insert(
                            value.id,
                            environment.constants.get(operand).cloned().flatten(),
                        );
                    }
                    children = self
                        .block(
                            target,
                            target.body.as_ref().expect("verified body"),
                            &mut inner,
                            depth + 1,
                            calls + 1,
                        )?
                        .0;
                }
                DisplayKind::Call
            }
            _ if opcode.starts_with("quantum.") => DisplayKind::Quantum,
            _ if opcode.starts_with("classical.") || opcode == "qcore.constant" => {
                DisplayKind::Classical
            }
            _ if opcode.contains("call") => DisplayKind::Call,
            _ => DisplayKind::Structure,
        };
        let constant = if opcode == "qcore.constant" {
            literal_attr(operation, "value").and_then(static_values::value_literal)
        } else if opcode.starts_with("classical.") {
            static_values::evaluate_value(operation, &environment.constants)
        } else {
            None
        };
        let mut outputs = Vec::new();
        for result in &operation.results {
            let output_label = if matches!(kind, DisplayKind::Classical) {
                label.as_str()
            } else if result.hint.is_empty() {
                opcode.strip_prefix("quantum.").unwrap_or("state")
            } else {
                result.hint.as_str()
            };
            let id = self.value(function, result, output_label)?;
            environment.ids.insert(result.id, id);
            // Only facts proved by the shared exact evaluator are propagated.
            if opcode != "qcore.for" {
                environment.constants.insert(result.id, constant.clone());
            }
            outputs.push(id);
        }
        Ok(DisplayNode {
            kind,
            opcode: opcode.to_owned(),
            label,
            inputs,
            outputs,
            children,
            line: operation.span.as_ref().map(|span| span.start_line),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn region(
        &mut self,
        function: &wire::FunctionProto,
        block: &wire::BlockProto,
        operands: &[u64],
        environment: &Environment,
        depth: usize,
        calls: usize,
        label: &str,
    ) -> Result<DisplayNode, VisualizationError> {
        self.node(depth)?;
        let mut inner = self.fork(environment)?;
        for (argument, operand) in block.arguments.iter().zip(operands) {
            inner.ids.insert(argument.id, environment.ids[operand]);
            inner.constants.insert(
                argument.id,
                environment.constants.get(operand).cloned().flatten(),
            );
        }
        let (children, outputs) = self.block(function, block, &mut inner, depth + 1, calls)?;
        Ok(DisplayNode {
            kind: DisplayKind::Region,
            opcode: "region".to_owned(),
            label: label.to_owned(),
            inputs: operands.iter().map(|id| environment.ids[id]).collect(),
            outputs,
            children,
            line: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn symbolic_loop(
        &mut self,
        function: &wire::FunctionProto,
        operation: &wire::OperationProto,
        environment: &Environment,
        inner: &mut Environment,
        depth: usize,
        calls: usize,
    ) -> Result<Vec<DisplayNode>, VisualizationError> {
        let body = &operation.regions[0].blocks[0];
        let induction = &body.arguments[0];
        let induction_name = string_attr(operation, "induction_name").unwrap_or("index");

        let id = self.value(function, induction, induction_name)?;
        inner.ids.insert(induction.id, id);
        inner.constants.insert(induction.id, None);
        let (children, outputs) = self.block(function, body, inner, depth + 1, calls)?;
        self.node(depth + 1)?;
        let mut regions = vec![DisplayNode {
            kind: DisplayKind::Region,
            opcode: "loop.body".to_owned(),
            label: "body (once; symbolic induction)".to_owned(),
            inputs: body
                .arguments
                .iter()
                .map(|value| inner.ids[&value.id])
                .collect(),
            outputs,
            children,
            line: None,
        }];
        // The final values may originate from zero iterations or early exits;
        // do not falsely alias them to one symbolic body's yield.
        let else_block = &operation.regions[1].blocks[0];
        if !else_block.operations.is_empty() {
            let mut completion = self.fork(environment)?;
            let mut inputs = Vec::new();
            for argument in &else_block.arguments {
                let id = self.value(function, argument, "final carried value")?;
                completion.ids.insert(argument.id, id);
                completion.constants.insert(argument.id, None);
                inputs.push(id);
            }
            let (children, outputs) =
                self.block(function, else_block, &mut completion, depth + 2, calls)?;
            self.node(depth + 1)?;
            regions.push(DisplayNode {
                kind: DisplayKind::Region,
                opcode: "loop.completion".to_owned(),
                label: "normal completion / else (final carried values)".to_owned(),
                inputs,
                outputs,
                children,
                line: None,
            });
        }
        Ok(regions)
    }

    fn loop_children(
        &mut self,
        function: &wire::FunctionProto,
        operation: &wire::OperationProto,
        environment: &mut Environment,
        depth: usize,
        calls: usize,
    ) -> Result<Vec<DisplayNode>, VisualizationError> {
        for result in &operation.results {
            environment.constants.insert(result.id, None);
        }
        if self.loops == LoopDisplay::Folded {
            return Ok(Vec::new());
        }
        let body = &operation.regions[0].blocks[0];
        let induction = &body.arguments[0];
        let induction_name = string_attr(operation, "induction_name").unwrap_or("index");
        let mut inner = self.fork(environment)?;
        for (argument, operand) in body.arguments[1..].iter().zip(&operation.operands[3..]) {
            inner.ids.insert(argument.id, environment.ids[operand]);
            inner.constants.insert(
                argument.id,
                environment.constants.get(operand).cloned().flatten(),
            );
        }
        if self.loops == LoopDisplay::Body {
            return self.symbolic_loop(function, operation, environment, &mut inner, depth, calls);
        }
        if has_early_exit(body) || has_early_exit(&operation.regions[1].blocks[0]) {
            return Err(VisualizationError::Expansion(
                "body contains return, break, or continue".to_owned(),
            ));
        }
        let (bounds, count) = display_range(operation, environment)?;
        if count > (self.limit - self.nodes) as u128 {
            return Err(VisualizationError::Budget);
        }
        let mut regions = Vec::new();
        let mut carried_ids = operation.operands[3..]
            .iter()
            .map(|id| environment.ids[id])
            .collect::<Vec<_>>();
        let mut carried_constants = operation.operands[3..]
            .iter()
            .map(|id| environment.constants.get(id).cloned().flatten())
            .collect::<Vec<_>>();
        for iteration in 0..count {
            self.node(depth + 1)?;
            let index = i128::from(bounds[0])
                + i128::try_from(iteration).map_err(|_| VisualizationError::Budget)?
                    * i128::from(bounds[2]);
            let index = i64::try_from(index).map_err(|_| VisualizationError::Budget)?;
            let id = self.value(function, induction, &index.to_string())?;
            inner.ids.insert(induction.id, id);
            inner.constants.insert(
                induction.id,
                Some(StaticValue::Scalar(StaticScalar::Int(index))),
            );
            for ((argument, id), value) in body.arguments[1..]
                .iter()
                .zip(&carried_ids)
                .zip(&carried_constants)
            {
                inner.ids.insert(argument.id, *id);
                inner.constants.insert(argument.id, value.clone());
            }
            let inputs = carried_ids.clone();
            let (children, outputs) = self.block(function, body, &mut inner, depth + 2, calls)?;
            carried_ids.clone_from(&outputs);
            carried_constants = body
                .terminator
                .as_ref()
                .expect("verified terminator")
                .operands
                .iter()
                .map(|id| inner.constants.get(id).cloned().flatten())
                .collect();
            regions.push(DisplayNode {
                kind: DisplayKind::Iteration,
                opcode: "loop.iteration".to_owned(),
                label: format!("{induction_name} = {index}"),
                inputs,
                outputs,
                children,
                line: None,
            });
        }
        regions.push(self.loop_completion(
            function,
            operation,
            environment,
            depth,
            calls,
            carried_ids,
            &carried_constants,
        )?);
        Ok(regions)
    }

    #[allow(clippy::too_many_arguments)]
    fn loop_completion(
        &mut self,
        function: &wire::FunctionProto,
        operation: &wire::OperationProto,
        environment: &mut Environment,
        depth: usize,
        calls: usize,
        carried_ids: Vec<usize>,
        carried_constants: &[Option<StaticValue>],
    ) -> Result<DisplayNode, VisualizationError> {
        let else_block = &operation.regions[1].blocks[0];
        let mut completion = self.fork(environment)?;
        for ((argument, id), value) in else_block
            .arguments
            .iter()
            .zip(&carried_ids)
            .zip(carried_constants)
        {
            completion.ids.insert(argument.id, *id);
            completion.constants.insert(argument.id, value.clone());
        }
        let (children, outputs) =
            self.block(function, else_block, &mut completion, depth + 1, calls)?;
        let terminator = else_block.terminator.as_ref().expect("verified terminator");
        for (result, operand) in operation.results.iter().zip(&terminator.operands) {
            environment.constants.insert(
                result.id,
                completion.constants.get(operand).cloned().flatten(),
            );
        }
        self.node(depth + 1)?;
        Ok(DisplayNode {
            kind: DisplayKind::Region,
            opcode: "loop.completion".to_owned(),
            label: "normal completion / else".to_owned(),
            inputs: carried_ids,
            outputs,
            children,
            line: None,
        })
    }
}

fn display_range(
    operation: &wire::OperationProto,
    environment: &Environment,
) -> Result<(Vec<i64>, u128), VisualizationError> {
    let bounds = operation.operands[..3]
        .iter()
        .map(|id| {
            environment
                .constants
                .get(id)
                .and_then(Option::as_ref)
                .and_then(StaticValue::scalar)
                .and_then(|value| match value {
                    StaticScalar::Int(value) => Some(value),
                    _ => None,
                })
        })
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| {
            VisualizationError::Expansion("range is symbolic or launch/shot dependent".to_owned())
        })?;
    let count = static_values::range_len(bounds[0], bounds[1], bounds[2])
        .ok_or_else(|| VisualizationError::Expansion("range is not finite".to_owned()))?;
    Ok((bounds, count))
}

fn has_early_exit(block: &wire::BlockProto) -> bool {
    let mut pending = vec![block];
    while let Some(block) = pending.pop() {
        if block
            .terminator
            .as_ref()
            .is_some_and(|exit| exit.kind != "yield")
        {
            return true;
        }
        for operation in &block.operations {
            for region in &operation.regions {
                pending.extend(&region.blocks);
            }
        }
    }
    false
}

fn literal_attr<'a>(
    operation: &'a wire::OperationProto,
    name: &str,
) -> Option<&'a wire::LiteralProto> {
    operation
        .attributes
        .iter()
        .find(|attribute| attribute.name == name)?
        .value
        .as_ref()
}

fn string_attr<'a>(operation: &'a wire::OperationProto, name: &str) -> Option<&'a str> {
    match literal_attr(operation, name)?.value.as_ref()? {
        wire::literal_proto::Value::StringValue(value) => Some(value),
        _ => None,
    }
}

fn bounded(source: &str) -> String {
    let mut text = source.chars().take(MAX_TEXT).collect::<String>();
    if source.chars().nth(MAX_TEXT).is_some() {
        text.push('…');
    }
    text
}

fn literal_label(literal: &wire::LiteralProto) -> String {
    use wire::literal_proto::Value;
    bounded(&match literal.value.as_ref() {
        Some(Value::IntValue(value)) => value.to_string(),
        Some(Value::RealValue(value)) => value.to_string(),
        Some(Value::BoolValue(value)) => value.to_string(),
        Some(Value::StringValue(value)) => bounded(value),
        Some(Value::TupleValue(value)) => format!("aggregate[{}]", value.values.len()),
        Some(Value::UnitValue(_)) => "Unit".to_owned(),
        _ => "literal".to_owned(),
    })
}

fn expression(operation: &wire::OperationProto, operands: &[&str]) -> String {
    let opcode = operation.opcode.as_str();
    if opcode == "qcore.constant" {
        return literal_attr(operation, "value")
            .map_or_else(|| "constant".to_owned(), literal_label);
    }
    let suffix = opcode.split('.').next_back().unwrap_or(opcode);
    let binary = match suffix {
        "add" => Some("+"),
        "sub" => Some("-"),
        "mul" => Some("*"),
        "truediv" => Some("/"),
        "floordiv" => Some("//"),
        "mod" => Some("%"),
        _ => None,
    };
    if opcode.starts_with("classical.") {
        if let Some(symbol) = binary
            && operands.len() == 2
        {
            return bounded(&format!("({} {symbol} {})", operands[0], operands[1]));
        }
        if suffix == "index" && operands.len() == 2 {
            return bounded(&format!("{}[{}]", operands[0], operands[1]));
        }
        return bounded(&format!("{suffix}({})", operands.join(", ")));
    }
    if opcode == "qcore.reshape" {
        return "reinterpret / split / concatenate".to_owned();
    }
    if opcode == "qcore.unpack" {
        return "unpack".to_owned();
    }
    if opcode == "qcore.tuple" {
        return "pack".to_owned();
    }
    if opcode.starts_with("quantum.") {
        return bounded(&format!("{suffix}({})", operands.join(", ")));
    }
    let callee = string_attr(operation, "callee").unwrap_or("");
    bounded(&format!("{suffix} {callee}"))
}

fn type_label(value: &wire::TypeProto, depth: usize, remaining: &mut usize) -> String {
    if depth > MAX_DEPTH || *remaining == 0 {
        return "…".to_owned();
    }
    *remaining -= 1;
    let name = match value.kind {
        3 => "tuple",
        5 => "Q",
        7 => "UFixed",
        _ if value.name == "Array" => match value.dimensions.len() {
            1 => "Vector",
            2 => "Matrix",
            _ => "Tensor",
        },
        _ => &value.name,
    };
    let mut arguments = value
        .arguments
        .iter()
        .skip(usize::from(matches!(value.name.as_str(), "Bits" | "UInt")))
        .take(16)
        .map(|argument| type_label(argument, depth + 1, remaining))
        .collect::<Vec<_>>();
    arguments.extend(value.dimensions.iter().take(16).map(
        |dimension| match dimension.value.as_ref() {
            Some(wire::dimension_proto::Value::Fixed(value)) => value.to_string(),
            Some(wire::dimension_proto::Value::Symbol(value)) => bounded(value),
            None => "?".to_owned(),
        },
    ));
    if value.arguments.len() > 16 || value.dimensions.len() > 16 {
        arguments.push("…".to_owned());
    }
    if arguments.is_empty() {
        bounded(name)
    } else {
        bounded(&format!("{name}[{}]", arguments.join(", ")))
    }
}
