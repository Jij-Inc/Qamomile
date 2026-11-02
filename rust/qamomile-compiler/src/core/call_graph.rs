//! Closed, normalized static callable graph for persistent `QCore` programs.
//!
//! Every callable operation is decoded through one exact attribute schema.
//! The resulting graph is retained by the checked program so later effect and
//! capability passes never need to reinterpret producer-controlled literals.

#![allow(dead_code)]

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use thiserror::Error;

use crate::control_state::ControlState;
use crate::verification::QCoreVerificationWork;
use crate::wire;

enum TraversalWork<'a> {
    Block(&'a wire::BlockProto),
    Operation(&'a wire::OperationProto),
}

/// The semantic transformation requested at one static call site.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum CallFlavor {
    /// Direct application of the declared callable.
    Ordinary,
    /// Application of the callable's exact adjoint.
    Adjoint,
    /// Replay-backed adjoint paired with one exact forward call result.
    HistoryAdjoint,
    /// Coherently controlled application.
    Control,
    /// Total coherent selection among several callables.
    Select,
    /// Exact coherent lifting of one pure Boolean callable.
    BooleanLift,
}

/// One normalized static site in deterministic structural traversal order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct StaticCallSite {
    caller: usize,
    callees: Box<[usize]>,
    flavor: CallFlavor,
    operands: Box<[u64]>,
}

impl StaticCallSite {
    /// Return the caller's stable function index.
    #[must_use]
    pub(crate) const fn caller(&self) -> usize {
        self.caller
    }

    /// Return all resolved callees in semantic selection order.
    #[must_use]
    pub(crate) fn callees(&self) -> &[usize] {
        &self.callees
    }

    /// Return the callable transformation carried by this site.
    #[must_use]
    pub(crate) const fn flavor(&self) -> CallFlavor {
        self.flavor
    }

    /// Return the exact SSA operands attached to the verified call operation.
    #[must_use]
    pub(crate) fn operands(&self) -> &[u64] {
        &self.operands
    }
}

/// Closed static call graph retained by a checked program, including recursion.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct StaticCallGraph {
    symbols: Box<[Arc<str>]>,
    indices: HashMap<Arc<str>, usize>,
    sites: Box<[StaticCallSite]>,
    site_offsets: Box<[usize]>,
    adjacency: Box<[Box<[usize]>]>,
    callee_first: Box<[usize]>,
    components: Box<[CallComponent]>,
}

/// A checked strongly connected group in callee-before-caller component order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CallComponent {
    members: Box<[usize]>,
    recursive: bool,
}

impl CallComponent {
    /// Return stable function indices belonging to this component.
    pub(crate) fn members(&self) -> &[usize] {
        &self.members
    }

    /// Return whether this component contains a self or mutual recursive edge.
    pub(crate) const fn is_recursive(&self) -> bool {
        self.recursive
    }
}

impl StaticCallGraph {
    /// Return one stable symbol by its graph index.
    #[must_use]
    pub(crate) fn symbol(&self, index: usize) -> Option<&str> {
        self.symbols.get(index).map(AsRef::as_ref)
    }

    /// Resolve one normalized symbol to its stable function index.
    #[must_use]
    pub(crate) fn index(&self, symbol: &str) -> Option<usize> {
        self.indices.get(symbol).copied()
    }

    /// Return normalized static sites in deterministic structural order.
    #[must_use]
    pub(crate) fn sites(&self) -> &[StaticCallSite] {
        &self.sites
    }

    /// Return one caller's sites in their original structural order.
    #[must_use]
    pub(crate) fn sites_for_caller(&self, index: usize) -> Option<&[StaticCallSite]> {
        let start = *self.site_offsets.get(index)?;
        let end = *self.site_offsets.get(index.checked_add(1)?)?;
        self.sites.get(start..end)
    }

    /// Return unique outgoing callees for one function.
    #[must_use]
    pub(crate) fn callees(&self, index: usize) -> Option<&[usize]> {
        self.adjacency.get(index).map(AsRef::as_ref)
    }

    /// Flatten the callee-first component order with stable member indices.
    /// Consumers closing recursive facts must iterate components to convergence.
    #[must_use]
    pub(crate) fn callee_first_order(&self) -> &[usize] {
        &self.callee_first
    }

    /// Return strongly connected components in callee-before-caller order.
    pub(crate) fn components(&self) -> &[CallComponent] {
        &self.components
    }

    /// Return whether the closed graph contains recursive calls.
    pub(crate) fn has_recursion(&self) -> bool {
        self.components.iter().any(CallComponent::is_recursive)
    }

    /// Return the number of functions represented by this closed graph.
    #[must_use]
    pub(crate) const fn function_count(&self) -> usize {
        self.symbols.len()
    }
}

/// Failure to establish a closed normalized static call graph.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub(crate) enum CallGraphError {
    /// A callable operation contains duplicate producer-controlled attributes.
    #[error("Q1025: duplicate attributes on {opcode:?}")]
    DuplicateAttributes { opcode: String },
    /// An ordinary call does not carry exactly one resolved callee.
    #[error("Q1026: unresolved static callable {callee:?}")]
    InvalidOrdinaryCall { callee: Option<String> },
    /// An adjoint call does not carry a closed callable identity and proof row.
    #[error("Q1029: invalid adjoint callable {callee:?}")]
    InvalidAdjointCall { callee: Option<String> },
    /// A controlled call does not carry its exact static attribute row.
    #[error("Q1051: invalid controlled callable {callee:?}")]
    InvalidControlCall { callee: Option<String> },
    /// A selected call does not carry a nonempty closed callable row.
    #[error("Q1051: invalid selected callable list")]
    InvalidSelectCall,
    /// A Boolean lift does not carry one resolved predicate identity.
    #[error("Q7680: unresolved Boolean lift predicate {predicate:?}")]
    InvalidBooleanPredicate { predicate: Option<String> },
    /// An internal graph count cannot be represented safely.
    #[error("Q4007: QCore call graph exceeds its supported capacity")]
    GraphCapacityExceeded,
}

/// Decode and retain the complete closed static call graph.
///
/// # Errors
///
/// Returns [`CallGraphError`] when a call attribute row is ambiguous, a
/// referenced symbol is absent, or the graph exceeds supported capacity.
pub(crate) fn build(package: &wire::QCorePackageProto) -> Result<StaticCallGraph, CallGraphError> {
    let mut work = QCoreVerificationWork::default();
    build_with_work(package, &mut work)
}

pub(crate) fn build_with_work(
    package: &wire::QCorePackageProto,
    verification_work: &mut QCoreVerificationWork,
) -> Result<StaticCallGraph, CallGraphError> {
    verification_work.charge_call_graph(package.functions.len().saturating_mul(2));
    let symbols = package
        .functions
        .iter()
        .map(|function| Arc::<str>::from(function.symbol.as_str()))
        .collect::<Box<_>>();
    let indices = symbols
        .iter()
        .enumerate()
        .map(|(index, symbol)| (Arc::clone(symbol), index))
        .collect::<HashMap<_, _>>();
    let mut sites = Vec::new();
    let mut site_offsets = Vec::with_capacity(symbols.len().saturating_add(1));
    site_offsets.push(0);
    let mut outgoing = (0..symbols.len())
        .map(|_| HashSet::new())
        .collect::<Vec<_>>();

    for (caller, function) in package.functions.iter().enumerate() {
        verification_work.charge_call_graph(1);
        let mut traversal = function
            .body
            .iter()
            .map(TraversalWork::Block)
            .collect::<Vec<_>>();
        while let Some(item) = traversal.pop() {
            match item {
                TraversalWork::Block(block) => {
                    verification_work.charge_call_graph(1);
                    traversal.extend(block.operations.iter().rev().map(TraversalWork::Operation));
                }
                TraversalWork::Operation(operation) => {
                    verification_work.charge_call_graph(1);
                    if let Some((flavor, names)) =
                        decode_call(operation, &indices, verification_work)?
                    {
                        verification_work.charge_call_graph(names.len());
                        let callees = names
                            .iter()
                            .map(|name| {
                                indices
                                    .get(name.as_str())
                                    .copied()
                                    .ok_or_else(|| unresolved(flavor, names.first().cloned()))
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        let row = outgoing
                            .get_mut(caller)
                            .ok_or(CallGraphError::GraphCapacityExceeded)?;
                        verification_work.charge_call_graph(callees.len());
                        row.extend(callees.iter().copied());
                        sites.push(StaticCallSite {
                            caller,
                            callees: callees.into_boxed_slice(),
                            flavor,
                            operands: operation.operands.clone().into_boxed_slice(),
                        });
                    }
                    for region in operation.regions.iter().rev() {
                        traversal.extend(region.blocks.iter().rev().map(TraversalWork::Block));
                    }
                }
            }
        }
        site_offsets.push(sites.len());
    }

    let adjacency = outgoing
        .into_iter()
        .map(|row| {
            verification_work.charge_call_graph(1_usize.saturating_add(row.len()));
            let mut row = row.into_iter().collect::<Vec<_>>();
            row.sort_unstable();
            row.into_boxed_slice()
        })
        .collect::<Box<_>>();
    let components = call_components(&adjacency, verification_work)?;
    let callee_first = components
        .iter()
        .flat_map(|group| group.members.iter().copied())
        .collect();
    Ok(StaticCallGraph {
        symbols,
        indices,
        sites: sites.into_boxed_slice(),
        site_offsets: site_offsets.into_boxed_slice(),
        adjacency,
        callee_first,
        components,
    })
}

fn decode_call(
    operation: &wire::OperationProto,
    functions: &HashMap<Arc<str>, usize>,
    work: &mut QCoreVerificationWork,
) -> Result<Option<(CallFlavor, Vec<String>)>, CallGraphError> {
    work.charge_call_graph(operation.attributes.len());
    require_unique_attributes(operation)?;
    let decoded = match operation.opcode.as_str() {
        "qcore.call" => {
            let callee = exact_string(operation, &["callee"], "callee");
            if !callee
                .as_ref()
                .is_some_and(|name| functions.contains_key(name.as_str()))
            {
                return Err(CallGraphError::InvalidOrdinaryCall { callee });
            }
            let callee = callee.expect("ordinary callee was checked above");
            Some((CallFlavor::Ordinary, vec![callee]))
        }
        "qcore.adjoint_call" => {
            let names = attribute_names(operation);
            let history = names == ["callee", "history_origin", "proof"]
                && integer_attribute(operation, "history_origin")
                && string_attribute(operation, "proof") == Some("modular-adjoint");
            let schema_ok = history
                || names == ["callee"]
                || (names == ["callee", "proof"]
                    && string_attribute(operation, "proof").is_some_and(|proof| {
                        matches!(proof, "explicit-adjoint" | "modular-adjoint")
                    }));
            let callee = string_attribute(operation, "callee").map(str::to_owned);
            if !callee
                .as_ref()
                .is_some_and(|name| schema_ok && functions.contains_key(name.as_str()))
            {
                return Err(CallGraphError::InvalidAdjointCall { callee });
            }
            // Proof spellings are normalized and then discarded. They are not
            // capability evidence; a later pass must replay the actual proof.
            let callee = callee.expect("adjoint callee was checked above");
            Some((
                if history {
                    CallFlavor::HistoryAdjoint
                } else {
                    CallFlavor::Adjoint
                },
                vec![callee],
            ))
        }
        "qcore.control_call" => {
            let schema_ok = attribute_names(operation) == ["callee", "control_state"]
                && operation
                    .attributes
                    .iter()
                    .find(|attribute| attribute.name == "control_state")
                    .and_then(|attribute| attribute.value.as_ref())
                    .and_then(ControlState::decode)
                    .is_some();
            let callee = string_attribute(operation, "callee").map(str::to_owned);
            if !callee
                .as_ref()
                .is_some_and(|name| schema_ok && functions.contains_key(name.as_str()))
            {
                return Err(CallGraphError::InvalidControlCall { callee });
            }
            let callee = callee.expect("controlled callee was checked above");
            Some((CallFlavor::Control, vec![callee]))
        }
        "qcore.select_call" => {
            if attribute_names(operation) != ["callees"] {
                return Err(CallGraphError::InvalidSelectCall);
            }
            let Some(callees) = string_tuple_attribute(operation, "callees") else {
                return Err(CallGraphError::InvalidSelectCall);
            };
            if callees.is_empty()
                || callees
                    .iter()
                    .any(|callee| !functions.contains_key(callee.as_str()))
            {
                return Err(CallGraphError::InvalidSelectCall);
            }
            Some((CallFlavor::Select, callees))
        }
        "qcore.phase_by" | "qcore.xor_by" | "qcore.permute_by" => {
            let predicate = exact_string(operation, &["predicate"], "predicate");
            if !predicate
                .as_ref()
                .is_some_and(|name| functions.contains_key(name.as_str()))
            {
                return Err(CallGraphError::InvalidBooleanPredicate { predicate });
            }
            Some((
                CallFlavor::BooleanLift,
                vec![predicate.expect("Boolean predicate was checked above")],
            ))
        }
        _ => None,
    };
    Ok(decoded)
}

fn unresolved(flavor: CallFlavor, callee: Option<String>) -> CallGraphError {
    match flavor {
        CallFlavor::Ordinary => CallGraphError::InvalidOrdinaryCall { callee },
        CallFlavor::Adjoint | CallFlavor::HistoryAdjoint => {
            CallGraphError::InvalidAdjointCall { callee }
        }
        CallFlavor::Control => CallGraphError::InvalidControlCall { callee },
        CallFlavor::Select => CallGraphError::InvalidSelectCall,
        CallFlavor::BooleanLift => CallGraphError::InvalidBooleanPredicate { predicate: callee },
    }
}

fn require_unique_attributes(operation: &wire::OperationProto) -> Result<(), CallGraphError> {
    let mut names = HashSet::with_capacity(operation.attributes.len());
    if operation
        .attributes
        .iter()
        .any(|attribute| !names.insert(attribute.name.as_str()))
    {
        return Err(CallGraphError::DuplicateAttributes {
            opcode: operation.opcode.clone(),
        });
    }
    Ok(())
}

fn attribute_names(operation: &wire::OperationProto) -> Vec<&str> {
    operation
        .attributes
        .iter()
        .map(|attribute| attribute.name.as_str())
        .collect()
}

fn exact_string(
    operation: &wire::OperationProto,
    names: &[&str],
    value_name: &str,
) -> Option<String> {
    (attribute_names(operation) == names)
        .then(|| string_attribute(operation, value_name).map(str::to_owned))
        .flatten()
}

fn string_attribute<'a>(operation: &'a wire::OperationProto, name: &str) -> Option<&'a str> {
    use wire::literal_proto::Value;

    operation
        .attributes
        .iter()
        .find(|attribute| attribute.name == name)
        .and_then(|attribute| attribute.value.as_ref())
        .and_then(|literal| match literal.value.as_ref() {
            Some(Value::StringValue(value)) => Some(value.as_str()),
            _ => None,
        })
}

fn integer_attribute(operation: &wire::OperationProto, name: &str) -> bool {
    use wire::literal_proto::Value;

    operation
        .attributes
        .iter()
        .find(|attribute| attribute.name == name)
        .and_then(|attribute| attribute.value.as_ref())
        .is_some_and(|literal| {
            matches!(
                literal.value,
                Some(Value::IntValue(_) | Value::BigintValue(_))
            )
        })
}

fn string_tuple_attribute(operation: &wire::OperationProto, name: &str) -> Option<Vec<String>> {
    use wire::literal_proto::Value;

    let literal = operation
        .attributes
        .iter()
        .find(|attribute| attribute.name == name)?
        .value
        .as_ref()?;
    let Value::TupleValue(sequence) = literal.value.as_ref()? else {
        return None;
    };
    sequence
        .values
        .iter()
        .map(|item| match item.value.as_ref() {
            Some(Value::StringValue(value)) if !value.is_empty() => Some(value.clone()),
            _ => None,
        })
        .collect()
}

/// Compute SCCs without using the host stack for user-sized call graphs.
fn call_components(
    adjacency: &[Box<[usize]>],
    work: &mut QCoreVerificationWork,
) -> Result<Box<[CallComponent]>, CallGraphError> {
    let mut callers = vec![Vec::new(); adjacency.len()];
    for (caller, row) in adjacency.iter().enumerate() {
        work.charge_call_graph(1_usize.saturating_add(row.len()));
        for &callee in row {
            callers
                .get_mut(callee)
                .ok_or(CallGraphError::GraphCapacityExceeded)?
                .push(caller);
        }
    }
    let mut seen = vec![false; adjacency.len()];
    let mut finished = Vec::with_capacity(adjacency.len());
    for root in 0..adjacency.len() {
        let mut pending = vec![(root, false)];
        while let Some((node, exiting)) = pending.pop() {
            work.charge_call_graph(1);
            if exiting {
                finished.push(node);
            } else if !seen[node] {
                seen[node] = true;
                pending.push((node, true));
                pending.extend(adjacency[node].iter().rev().map(|&child| (child, false)));
            }
        }
    }
    let mut owner = vec![usize::MAX; adjacency.len()];
    let mut groups = Vec::new();
    for root in finished.into_iter().rev() {
        if owner[root] != usize::MAX {
            continue;
        }
        let index = groups.len();
        let mut pending = vec![root];
        let mut members = Vec::new();
        while let Some(node) = pending.pop() {
            work.charge_call_graph(1);
            if owner[node] != usize::MAX {
                continue;
            }
            owner[node] = index;
            members.push(node);
            pending.extend(callers[node].iter().copied());
        }
        members.sort_unstable();
        let recursive = members.len() > 1 || adjacency[root].contains(&root);
        groups.push(CallComponent {
            members: members.into_boxed_slice(),
            recursive,
        });
    }
    let mut outgoing = vec![BTreeSet::new(); groups.len()];
    let mut incoming = vec![BTreeSet::new(); groups.len()];
    for (caller, row) in adjacency.iter().enumerate() {
        work.charge_call_graph(1_usize.saturating_add(row.len()));
        for &callee in row {
            if owner[caller] != owner[callee] {
                outgoing[owner[caller]].insert(owner[callee]);
                incoming[owner[callee]].insert(owner[caller]);
            }
        }
    }
    let mut remaining = outgoing.iter().map(BTreeSet::len).collect::<Vec<_>>();
    let mut ready = groups
        .iter()
        .enumerate()
        .filter_map(|(index, group)| (remaining[index] == 0).then_some((group.members[0], index)))
        .collect::<BTreeSet<_>>();
    let mut ordered = Vec::with_capacity(groups.len());
    while let Some((_, next)) = ready.pop_first() {
        work.charge_call_graph(1);
        ordered.push(groups[next].clone());
        for &caller in &incoming[next] {
            work.charge_call_graph(1);
            remaining[caller] = remaining[caller]
                .checked_sub(1)
                .ok_or(CallGraphError::GraphCapacityExceeded)?;
            if remaining[caller] == 0 {
                ready.insert((groups[caller].members[0], caller));
            }
        }
    }
    if ordered.len() != groups.len() {
        return Err(CallGraphError::GraphCapacityExceeded);
    }
    Ok(ordered.into_boxed_slice())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn string(value: &str) -> wire::LiteralProto {
        wire::LiteralProto {
            value: Some(wire::literal_proto::Value::StringValue(value.to_owned())),
        }
    }

    fn integer(value: i64) -> wire::LiteralProto {
        wire::LiteralProto {
            value: Some(wire::literal_proto::Value::IntValue(value)),
        }
    }

    fn string_tuple(values: &[&str]) -> wire::LiteralProto {
        wire::LiteralProto {
            value: Some(wire::literal_proto::Value::TupleValue(
                wire::literal_proto::Sequence {
                    values: values.iter().map(|value| string(value)).collect(),
                },
            )),
        }
    }

    fn attribute(name: &str, value: wire::LiteralProto) -> wire::AttributeProto {
        wire::AttributeProto {
            name: name.to_owned(),
            value: Some(value),
        }
    }

    fn function(symbol: &str, operations: Vec<wire::OperationProto>) -> wire::FunctionProto {
        wire::FunctionProto {
            symbol: symbol.to_owned(),
            body: Some(wire::BlockProto {
                operations,
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn normalizes_every_static_call_flavor() {
        let package = wire::QCorePackageProto {
            functions: vec![
                function("test.a", vec![]),
                function("test.b", vec![]),
                function(
                    "test.main",
                    vec![
                        wire::OperationProto {
                            opcode: "qcore.call".to_owned(),
                            attributes: vec![attribute("callee", string("test.a"))],
                            ..Default::default()
                        },
                        wire::OperationProto {
                            opcode: "qcore.adjoint_call".to_owned(),
                            attributes: vec![
                                attribute("callee", string("test.a")),
                                attribute("proof", string("explicit-adjoint")),
                            ],
                            ..Default::default()
                        },
                        wire::OperationProto {
                            opcode: "qcore.control_call".to_owned(),
                            attributes: vec![
                                attribute("callee", string("test.b")),
                                attribute("control_state", integer(1)),
                            ],
                            ..Default::default()
                        },
                        wire::OperationProto {
                            opcode: "qcore.select_call".to_owned(),
                            attributes: vec![attribute(
                                "callees",
                                string_tuple(&["test.a", "test.b"]),
                            )],
                            ..Default::default()
                        },
                    ],
                ),
            ],
            ..Default::default()
        };

        let graph = build(&package).expect("closed acyclic graph");
        assert_eq!(graph.sites().len(), 4);
        assert_eq!(graph.sites()[0].flavor(), CallFlavor::Ordinary);
        assert_eq!(graph.sites()[1].flavor(), CallFlavor::Adjoint);
        assert_eq!(graph.sites()[2].flavor(), CallFlavor::Control);
        assert_eq!(graph.sites()[3].flavor(), CallFlavor::Select);
        assert_eq!(graph.callees(2), Some([0, 1].as_slice()));
    }

    #[test]
    fn rejects_extra_ordinary_call_attributes() {
        let package = wire::QCorePackageProto {
            functions: vec![
                function("test.callee", vec![]),
                function(
                    "test.main",
                    vec![wire::OperationProto {
                        opcode: "qcore.call".to_owned(),
                        attributes: vec![
                            attribute("callee", string("test.callee")),
                            attribute("proof", string("forged")),
                        ],
                        ..Default::default()
                    }],
                ),
            ],
            ..Default::default()
        };

        assert_eq!(
            build(&package),
            Err(CallGraphError::InvalidOrdinaryCall { callee: None })
        );
    }

    #[test]
    fn groups_cycles_across_distinct_call_flavors() {
        let package = wire::QCorePackageProto {
            functions: vec![
                function(
                    "test.a",
                    vec![wire::OperationProto {
                        opcode: "qcore.call".to_owned(),
                        attributes: vec![attribute("callee", string("test.b"))],
                        ..Default::default()
                    }],
                ),
                function(
                    "test.b",
                    vec![wire::OperationProto {
                        opcode: "qcore.adjoint_call".to_owned(),
                        attributes: vec![attribute("callee", string("test.a"))],
                        ..Default::default()
                    }],
                ),
            ],
            ..Default::default()
        };

        let graph = build(&package).expect("recursive graph remains closed");
        assert!(graph.has_recursion());
        assert_eq!(graph.components().len(), 1);
        assert_eq!(
            graph.components()[0].members().len(),
            package.functions.len()
        );
    }

    #[test]
    fn retains_self_recursion_hidden_in_a_region() {
        let recursive = wire::OperationProto {
            opcode: "qcore.call".to_owned(),
            attributes: vec![attribute("callee", string("test.main"))],
            ..Default::default()
        };
        let package = wire::QCorePackageProto {
            functions: vec![function(
                "test.main",
                vec![wire::OperationProto {
                    opcode: "qcore.if".to_owned(),
                    regions: vec![wire::RegionProto {
                        blocks: vec![wire::BlockProto {
                            operations: vec![recursive],
                            ..Default::default()
                        }],
                    }],
                    ..Default::default()
                }],
            )],
            ..Default::default()
        };

        let graph = build(&package).expect("recursive graph remains closed");
        assert!(graph.has_recursion());
        assert_eq!(graph.components().len(), 1);
        assert_eq!(
            graph.components()[0].members().len(),
            package.functions.len()
        );
    }

    #[test]
    fn retains_true_lexical_preorder_and_call_operands() {
        let call = |callee: &str, operand| wire::OperationProto {
            opcode: "qcore.call".to_owned(),
            operands: vec![operand],
            attributes: vec![attribute("callee", string(callee))],
            ..Default::default()
        };
        let nested = wire::OperationProto {
            opcode: "qcore.if".to_owned(),
            regions: vec![wire::RegionProto {
                blocks: vec![wire::BlockProto {
                    operations: vec![call("test.b", 12)],
                    ..Default::default()
                }],
            }],
            ..Default::default()
        };
        let package = wire::QCorePackageProto {
            functions: vec![
                function("test.a", vec![]),
                function("test.b", vec![]),
                function("test.c", vec![]),
                function(
                    "test.main",
                    vec![call("test.a", 11), nested, call("test.c", 13)],
                ),
            ],
            ..Default::default()
        };

        let graph = build(&package).expect("closed graph");
        assert_eq!(graph.sites_for_caller(0), Some([].as_slice()));
        assert_eq!(graph.sites_for_caller(1), Some([].as_slice()));
        assert_eq!(graph.sites_for_caller(2), Some([].as_slice()));
        assert_eq!(graph.sites_for_caller(4), None);
        assert_eq!(graph.sites_for_caller(3), Some(graph.sites()));
        let sites = graph
            .sites()
            .iter()
            .map(|site| (graph.symbol(site.callees()[0]), site.operands()))
            .collect::<Vec<_>>();
        assert_eq!(
            sites,
            vec![
                (Some("test.a"), [11].as_slice()),
                (Some("test.b"), [12].as_slice()),
                (Some("test.c"), [13].as_slice()),
            ]
        );
    }

    #[test]
    fn computes_a_deterministic_callee_first_diamond_order() {
        let call = |callee: &str| wire::OperationProto {
            opcode: "qcore.call".to_owned(),
            attributes: vec![attribute("callee", string(callee))],
            ..Default::default()
        };
        let package = wire::QCorePackageProto {
            functions: vec![
                function("test.leaf", vec![]),
                function("test.left", vec![call("test.leaf")]),
                function("test.right", vec![call("test.leaf")]),
                function("test.root", vec![call("test.left"), call("test.right")]),
            ],
            ..Default::default()
        };

        let graph = build(&package).expect("closed graph");
        assert_eq!(graph.callee_first_order(), [0, 1, 2, 3]);
    }
}
