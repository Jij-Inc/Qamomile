//! Type system, operation algebra, verification, and compiler invariants.

pub(crate) mod attributes;
pub(crate) mod budget;
pub(crate) mod call_graph;
pub(crate) mod contracts;
pub(crate) mod control_dependence;
pub(crate) mod control_state;
pub(crate) mod declarations;
pub(crate) mod domains;
pub(crate) mod effects;
pub(crate) mod equations;
pub(crate) mod literals;
pub(crate) mod numeric_finiteness;
pub(crate) mod numeric_values;
pub(crate) mod operation_catalog;
pub(crate) mod operations;
pub(crate) mod slice_shape;
pub(crate) mod structure;
pub(crate) mod target_profile;
pub(crate) mod type_relations;
pub(crate) mod types;
pub(crate) mod verification;
pub(crate) mod visualization;
