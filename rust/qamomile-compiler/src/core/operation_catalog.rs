//! Read-only projection of the compiler's closed operation vocabulary.
//!
//! Source elaboration and inspection consume this catalog rather than carrying
//! a second Python copy of gate signatures, effects, or target requirements.

use crate::effects::{Effect, Requirement, operation_effects, operation_requirements};
use crate::operations::{Opcode, PrimitiveGateSpec, TerminatorKind, primitive_gate_spec};

/// One operation's compiler-owned semantic classification.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OperationCatalogEntry {
    opcode: &'static str,
    dialect: &'static str,
    structured: bool,
    effects: Vec<Effect>,
    requirements: Vec<Requirement>,
    gate: Option<PrimitiveGateSpec>,
}

impl OperationCatalogEntry {
    /// Return the canonical wire opcode.
    #[must_use]
    pub const fn opcode(&self) -> &'static str {
        self.opcode
    }

    /// Return `qcore`, `classical`, or `quantum`.
    #[must_use]
    pub const fn dialect(&self) -> &'static str {
        self.dialect
    }

    /// Return whether the operation may own structured regions.
    #[must_use]
    pub const fn is_structured(&self) -> bool {
        self.structured
    }

    /// Return direct denotational effects in stable semantic order.
    #[must_use]
    pub fn effects(&self) -> &[Effect] {
        &self.effects
    }

    /// Return direct target requirements in stable contract order.
    #[must_use]
    pub fn requirements(&self) -> &[Requirement] {
        &self.requirements
    }

    /// Return source signature metadata for a primitive gate.
    #[must_use]
    pub const fn gate(&self) -> Option<PrimitiveGateSpec> {
        self.gate
    }
}

/// Project the complete closed operation vocabulary from compiler authority.
#[must_use]
pub fn operation_catalog() -> Vec<OperationCatalogEntry> {
    Opcode::all()
        .map(|opcode| OperationCatalogEntry {
            opcode: opcode.as_str(),
            dialect: match opcode {
                Opcode::QCore(_) => "qcore",
                Opcode::Classical(_) => "classical",
                Opcode::Quantum(_) => "quantum",
            },
            structured: opcode.is_structured(),
            effects: operation_effects(opcode).iter().collect(),
            requirements: operation_requirements(opcode).iter().collect(),
            gate: primitive_gate_spec(opcode.as_str()),
        })
        .collect()
}

/// Return every closed terminator spelling in stable language order.
#[must_use]
pub fn terminator_catalog() -> Vec<&'static str> {
    TerminatorKind::ALL
        .iter()
        .copied()
        .map(TerminatorKind::as_str)
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::operations::primitive_gate_specs;

    #[test]
    fn catalog_is_unique_and_covers_every_primitive_gate_once() {
        let catalog = operation_catalog();
        let opcodes = catalog
            .iter()
            .map(OperationCatalogEntry::opcode)
            .collect::<HashSet<_>>();
        let gates = catalog
            .iter()
            .filter_map(OperationCatalogEntry::gate)
            .collect::<Vec<_>>();

        assert_eq!(opcodes.len(), catalog.len());
        assert_eq!(gates, primitive_gate_specs());
        assert!(catalog.iter().all(|entry| {
            entry.gate().is_none_or(|gate| {
                entry.dialect() == "quantum"
                    && gate.opcode() == entry.opcode()
                    && gate.families().contains(&"gates")
            })
        }));
        assert_eq!(
            terminator_catalog(),
            [
                "return",
                "unreachable",
                "yield",
                "break",
                "continue",
                "external",
            ]
        );
    }
}
