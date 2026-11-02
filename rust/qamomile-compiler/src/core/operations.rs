//! Closed operation vocabulary and function-local classical type equations.
//!
//! This module establishes the dialect boundary, function-local `ValueId`
//! resolution, and complete local equations for the classical dialect.
//! Dominance, ownership, non-classical equations, and region signatures remain
//! the responsibility of later verification passes.
//! Structured iteration uses an explicit shape-only length projection: its
//! Const extent is independent of element availability. Ordinary source `len`
//! keeps the operand's stage.

mod selectors;

pub(crate) use selectors::{SelectorOperand, selector_operands};

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use num_bigint::BigInt;
use prost::Message;
use thiserror::Error;

use crate::static_values::normalize_aggregate_index;
use crate::types::{Dimension, ScalarType, SemanticType, Stage, TypeArena, TypeId};
use crate::verification::QCoreVerificationWork;
use crate::wire;

macro_rules! closed_string_enum {
    (
        $(#[$enum_meta:meta])*
        $visibility:vis enum $name:ident {
            $($variant:ident => $spelling:literal),+ $(,)?
        }
    ) => {
        $(#[$enum_meta])*
        #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
        $visibility enum $name {
            $($variant),+
        }

        impl $name {
            pub(crate) fn parse(value: &str) -> Option<Self> {
                match value {
                    $($spelling => Some(Self::$variant),)+
                    _ => None,
                }
            }

            pub(crate) const fn as_str(self) -> &'static str {
                match self {
                    $(Self::$variant => $spelling,)+
                }
            }

            pub(crate) const ALL: &'static [Self] = &[$(Self::$variant),+];
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(self.as_str())
            }
        }
    };
}

closed_string_enum! {
    /// Operations in the compiler's core dialect.
    pub(crate) enum QCoreOpcode {
        Constant => "qcore.constant",
        Reshape => "qcore.reshape",
        StageLift => "qcore.stage_lift",
        Tuple => "qcore.tuple",
        Unpack => "qcore.unpack",
        Adjoint => "qcore.adjoint",
        Call => "qcore.call",
        AdjointCall => "qcore.adjoint_call",
        ControlCall => "qcore.control_call",
        SelectCall => "qcore.select_call",
        PhaseBy => "qcore.phase_by",
        XorBy => "qcore.xor_by",
        PermuteBy => "qcore.permute_by",
        Forget => "qcore.forget",
        If => "qcore.if",
        For => "qcore.for",
        While => "qcore.while",
    }
}

impl QCoreOpcode {
    const fn is_structured(self) -> bool {
        matches!(self, Self::If | Self::For | Self::While)
    }

    /// Return the exact callable operation that reverses this operation.
    #[must_use]
    pub(crate) const fn exact_adjoint(self) -> Option<Self> {
        match self {
            Self::Call => Some(Self::AdjointCall),
            Self::AdjointCall => Some(Self::Call),
            _ => None,
        }
    }
}

closed_string_enum! {
    /// Operations in the classical dialect.
    pub(crate) enum ClassicalOpcode {
        Array => "array",
        ObservableZero => "classical.observable_zero",
        ObservablePauli => "classical.observable_pauli",
        Add => "classical.add",
        Sub => "classical.sub",
        Mul => "classical.mul",
        TrueDiv => "classical.truediv",
        FloorDiv => "classical.floordiv",
        Mod => "classical.mod",
        Pow => "classical.pow",
        AndBits => "classical.and_bits",
        OrBits => "classical.or_bits",
        Xor => "classical.xor",
        LeftShift => "classical.lshift",
        RightShift => "classical.rshift",
        And => "classical.and",
        Or => "classical.or",
        Not => "classical.not",
        Invert => "classical.invert",
        Neg => "classical.neg",
        Pos => "classical.pos",
        CompareChain => "classical.compare_chain",
        Select => "classical.select",
        Index => "classical.index",
        Slice => "classical.slice",
        Update => "classical.update",
        MapGet => "classical.map_get",
        Project => "classical.project",
        Cast => "classical.cast",
        Len => "classical.len",
        Ceil => "classical.ceil",
        Log2 => "classical.log2",
        Sqrt => "classical.sqrt",
    }
}

/// One primitive gate's complete source and logical-resource contract.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrimitiveGateSpec {
    name: &'static str,
    opcode: &'static str,
    selectors: &'static [&'static str],
    families: &'static [&'static str],
    has_angle: bool,
}

impl PrimitiveGateSpec {
    /// Return the Python source spelling without its dialect prefix.
    #[must_use]
    pub const fn name(self) -> &'static str {
        self.name
    }

    /// Return the canonical `QCore` opcode spelling.
    #[must_use]
    pub const fn opcode(self) -> &'static str {
        self.opcode
    }

    /// Return optional component selectors in source argument order.
    #[must_use]
    pub const fn selectors(self) -> &'static [&'static str] {
        self.selectors
    }

    /// Return the default logical lane for one selector position.
    ///
    /// Primitive gates default their ordered selectors to distinct lanes in
    /// declaration order. Returning the value through the catalog keeps
    /// source binding and both resource analyzers on the same ABI rule.
    #[must_use]
    pub const fn selector_default(self, index: usize) -> Option<usize> {
        if index < self.selectors.len() {
            Some(index)
        } else {
            None
        }
    }

    /// Return whether omission applies this gate independently to every lane.
    #[must_use]
    pub const fn broadcasts_when_unselected(self) -> bool {
        self.selectors.len() == 1
    }

    /// Return logical resource families in stable lexical order.
    #[must_use]
    pub const fn families(self) -> &'static [&'static str] {
        self.families
    }

    /// Return whether the gate requires one angle before its selectors.
    #[must_use]
    pub const fn has_angle(self) -> bool {
        self.has_angle
    }
}

const SINGLE: &[&str] = &["target"];
const CONTROLLED: &[&str] = &["control", "target"];
const PAIR: &[&str] = &["left", "right"];
const TOFFOLI_SELECTORS: &[&str] = &["control1", "control2", "target"];
const SINGLE_CLIFFORD: &[&str] = &["clifford_gates", "gates", "single_qubit_gates"];
const SINGLE_T: &[&str] = &[
    "gates",
    "non_clifford_gates",
    "single_qubit_gates",
    "t_gates",
];
const SINGLE_ROTATION: &[&str] = &[
    "gates",
    "non_clifford_gates",
    "rotation_gates",
    "single_qubit_gates",
];
const TWO_CLIFFORD: &[&str] = &["clifford_gates", "gates", "two_qubit_gates"];
const TWO_ROTATION: &[&str] = &[
    "gates",
    "non_clifford_gates",
    "rotation_gates",
    "two_qubit_gates",
];
const TOFFOLI_FAMILIES: &[&str] = &[
    "gates",
    "multi_qubit_gates",
    "non_clifford_gates",
    "toffoli_gates",
];

macro_rules! primitive_gate_catalog {
    ($($variant:ident => ($name:literal, $selectors:expr, $families:expr, $has_angle:literal)),+ $(,)?) => {
        /// Primitive gates accepted by the quantum dialect.
        #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
        pub(crate) enum GateOpcode {
            $($variant),+
        }

        impl GateOpcode {
            pub(crate) fn parse(value: &str) -> Option<Self> {
                match value {
                    $(concat!("quantum.", $name) => Some(Self::$variant),)+
                    _ => None,
                }
            }

            pub(crate) fn parse_name(value: &str) -> Option<Self> {
                match value {
                    $($name => Some(Self::$variant),)+
                    _ => None,
                }
            }

            pub(crate) const fn as_str(self) -> &'static str {
                match self {
                    $(Self::$variant => concat!("quantum.", $name),)+
                }
            }

            pub(crate) const fn name(self) -> &'static str {
                match self {
                    $(Self::$variant => $name,)+
                }
            }

            pub(crate) const fn has_angle(self) -> bool {
                self.spec().has_angle()
            }

            const fn spec(self) -> PrimitiveGateSpec {
                match self {
                    $(Self::$variant => PrimitiveGateSpec {
                        name: $name,
                        opcode: concat!("quantum.", $name),
                        selectors: $selectors,
                        families: $families,
                        has_angle: $has_angle,
                    },)+
                }
            }

            const ALL: &'static [Self] = &[$(Self::$variant),+];
        }

        const PRIMITIVE_GATE_SPECS: &[PrimitiveGateSpec] = &[
            $(GateOpcode::$variant.spec()),+
        ];
    };
}

primitive_gate_catalog! {
    H => ("h", SINGLE, SINGLE_CLIFFORD, false),
    X => ("x", SINGLE, SINGLE_CLIFFORD, false),
    Y => ("y", SINGLE, SINGLE_CLIFFORD, false),
    Z => ("z", SINGLE, SINGLE_CLIFFORD, false),
    S => ("s", SINGLE, SINGLE_CLIFFORD, false),
    Sdg => ("sdg", SINGLE, SINGLE_CLIFFORD, false),
    T => ("t", SINGLE, SINGLE_T, false),
    Tdg => ("tdg", SINGLE, SINGLE_T, false),
    P => ("p", SINGLE, SINGLE_ROTATION, true),
    Rx => ("rx", SINGLE, SINGLE_ROTATION, true),
    Ry => ("ry", SINGLE, SINGLE_ROTATION, true),
    Rz => ("rz", SINGLE, SINGLE_ROTATION, true),
    Cx => ("cx", CONTROLLED, TWO_CLIFFORD, false),
    Cz => ("cz", CONTROLLED, TWO_CLIFFORD, false),
    Ccx => ("ccx", TOFFOLI_SELECTORS, TOFFOLI_FAMILIES, false),
    Cp => ("cp", CONTROLLED, TWO_ROTATION, true),
    Rzz => ("rzz", PAIR, TWO_ROTATION, true),
    Swap => ("swap", PAIR, TWO_CLIFFORD, false),
}

impl GateOpcode {
    /// Return the exact primitive gate used by this gate's adjoint.
    #[must_use]
    pub(crate) const fn adjoint(self) -> Self {
        match self {
            Self::S => Self::Sdg,
            Self::Sdg => Self::S,
            Self::T => Self::Tdg,
            Self::Tdg => Self::T,
            Self::H
            | Self::X
            | Self::Y
            | Self::Z
            | Self::P
            | Self::Rx
            | Self::Ry
            | Self::Rz
            | Self::Cx
            | Self::Cz
            | Self::Ccx
            | Self::Cp
            | Self::Rzz
            | Self::Swap => self,
        }
    }
}

/// Return every primitive gate specification in stable language order.
#[must_use]
pub const fn primitive_gate_specs() -> &'static [PrimitiveGateSpec] {
    PRIMITIVE_GATE_SPECS
}

pub(crate) fn primitive_gate_spec(opcode: &str) -> Option<PrimitiveGateSpec> {
    GateOpcode::parse(opcode).map(GateOpcode::spec)
}

/// Return the Qamomile-compatible logical families for one primitive gate.
pub(crate) fn logical_gate_families(opcode: &str) -> &'static [&'static str] {
    primitive_gate_spec(opcode).map_or(&[], PrimitiveGateSpec::families)
}

/// Abstract unitary work whose elementary gate costs depend on realization.
pub(crate) fn defers_gate_families(opcode: &str) -> bool {
    matches!(
        opcode,
        "quantum.add_into"
            | "quantum.subtract_into"
            | "quantum.pauli_evolution"
            | "quantum.add_constant"
            | "quantum.subtract_constant"
            | "quantum.mod_add"
            | "quantum.mod_sub"
            | "quantum.mod_add_constant"
            | "quantum.mod_sub_constant"
            | "quantum.mul_add"
            | "quantum.mul_sub"
            | "quantum.mod_mul_add"
            | "quantum.mod_mul_sub"
            | "quantum.mod_mul"
            | "quantum.mod_div"
            | "quantum.mod_pow"
            | "quantum.mod_pow_inverse"
            | "quantum.mod_multi_pow"
            | "quantum.mod_multi_pow_inverse"
            | "quantum.less_than_xor"
    )
}

impl fmt::Display for GateOpcode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Operations in the quantum dialect.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum QuantumOpcode {
    Alloc,
    Prepare,
    Measure,
    MeasureReset,
    MeasureX,
    MeasureY,
    MeasureZ,
    Discard,
    Reset,
    Release,
    Unprepare,
    GlobalPhase,
    PauliEvolution,
    AddInto,
    SubtractInto,
    AddConstant,
    SubtractConstant,
    ModAdd,
    ModSub,
    ModAddConstant,
    ModSubConstant,
    MulAdd,
    MulSub,
    ModMulAdd,
    ModMulSub,
    ModMul,
    ModDiv,
    ModPow,
    ModPowInverse,
    ModMultiPow,
    ModMultiPowInverse,
    LessThanXor,
    Gate(GateOpcode),
}

impl QuantumOpcode {
    pub(crate) fn parse(value: &str) -> Option<Self> {
        let operation = match value {
            "quantum.alloc" => Self::Alloc,
            "quantum.prepare" => Self::Prepare,
            "quantum.measure" => Self::Measure,
            "quantum.measure_reset" => Self::MeasureReset,
            "quantum.measure_x" => Self::MeasureX,
            "quantum.measure_y" => Self::MeasureY,
            "quantum.measure_z" => Self::MeasureZ,
            "quantum.discard" => Self::Discard,
            "quantum.reset" => Self::Reset,
            "quantum.release" => Self::Release,
            "quantum.unprepare" => Self::Unprepare,
            "quantum.global_phase" => Self::GlobalPhase,
            "quantum.pauli_evolution" => Self::PauliEvolution,
            "quantum.add_into" => Self::AddInto,
            "quantum.subtract_into" => Self::SubtractInto,
            "quantum.add_constant" => Self::AddConstant,
            "quantum.subtract_constant" => Self::SubtractConstant,
            "quantum.mod_add" => Self::ModAdd,
            "quantum.mod_sub" => Self::ModSub,
            "quantum.mod_add_constant" => Self::ModAddConstant,
            "quantum.mod_sub_constant" => Self::ModSubConstant,
            "quantum.mul_add" => Self::MulAdd,
            "quantum.mul_sub" => Self::MulSub,
            "quantum.mod_mul_add" => Self::ModMulAdd,
            "quantum.mod_mul_sub" => Self::ModMulSub,
            "quantum.mod_mul" => Self::ModMul,
            "quantum.mod_div" => Self::ModDiv,
            "quantum.mod_pow" => Self::ModPow,
            "quantum.mod_pow_inverse" => Self::ModPowInverse,
            "quantum.mod_multi_pow" => Self::ModMultiPow,
            "quantum.mod_multi_pow_inverse" => Self::ModMultiPowInverse,
            "quantum.less_than_xor" => Self::LessThanXor,
            _ => Self::Gate(GateOpcode::parse(value)?),
        };
        Some(operation)
    }

    pub(crate) fn parse_name(value: &str) -> Option<Self> {
        let operation = match value {
            "alloc" => Self::Alloc,
            "prepare" => Self::Prepare,
            "measure" => Self::Measure,
            "measure_reset" => Self::MeasureReset,
            "measure_x" => Self::MeasureX,
            "measure_y" => Self::MeasureY,
            "measure_z" => Self::MeasureZ,
            "discard" => Self::Discard,
            "reset" => Self::Reset,
            "release" => Self::Release,
            "unprepare" => Self::Unprepare,
            "global_phase" => Self::GlobalPhase,
            "pauli_evolution" => Self::PauliEvolution,
            "add_into" => Self::AddInto,
            "subtract_into" => Self::SubtractInto,
            "add_constant" => Self::AddConstant,
            "subtract_constant" => Self::SubtractConstant,
            "mod_add" => Self::ModAdd,
            "mod_sub" => Self::ModSub,
            "mod_add_constant" => Self::ModAddConstant,
            "mod_sub_constant" => Self::ModSubConstant,
            "mul_add" => Self::MulAdd,
            "mul_sub" => Self::MulSub,
            "mod_mul_add" => Self::ModMulAdd,
            "mod_mul_sub" => Self::ModMulSub,
            "mod_mul" => Self::ModMul,
            "mod_div" => Self::ModDiv,
            "mod_pow" => Self::ModPow,
            "mod_pow_inverse" => Self::ModPowInverse,
            "mod_multi_pow" => Self::ModMultiPow,
            "mod_multi_pow_inverse" => Self::ModMultiPowInverse,
            "less_than_xor" => Self::LessThanXor,
            _ => Self::Gate(GateOpcode::parse_name(value)?),
        };
        Some(operation)
    }

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Alloc => "quantum.alloc",
            Self::Prepare => "quantum.prepare",
            Self::Measure => "quantum.measure",
            Self::MeasureReset => "quantum.measure_reset",
            Self::MeasureX => "quantum.measure_x",
            Self::MeasureY => "quantum.measure_y",
            Self::MeasureZ => "quantum.measure_z",
            Self::Discard => "quantum.discard",
            Self::Reset => "quantum.reset",
            Self::Release => "quantum.release",
            Self::Unprepare => "quantum.unprepare",
            Self::GlobalPhase => "quantum.global_phase",
            Self::PauliEvolution => "quantum.pauli_evolution",
            Self::AddInto => "quantum.add_into",
            Self::SubtractInto => "quantum.subtract_into",
            Self::AddConstant => "quantum.add_constant",
            Self::SubtractConstant => "quantum.subtract_constant",
            Self::ModAdd => "quantum.mod_add",
            Self::ModSub => "quantum.mod_sub",
            Self::ModAddConstant => "quantum.mod_add_constant",
            Self::ModSubConstant => "quantum.mod_sub_constant",
            Self::MulAdd => "quantum.mul_add",
            Self::MulSub => "quantum.mul_sub",
            Self::ModMulAdd => "quantum.mod_mul_add",
            Self::ModMulSub => "quantum.mod_mul_sub",
            Self::ModMul => "quantum.mod_mul",
            Self::ModDiv => "quantum.mod_div",
            Self::ModPow => "quantum.mod_pow",
            Self::ModPowInverse => "quantum.mod_pow_inverse",
            Self::ModMultiPow => "quantum.mod_multi_pow",
            Self::ModMultiPowInverse => "quantum.mod_multi_pow_inverse",
            Self::LessThanXor => "quantum.less_than_xor",
            Self::Gate(gate) => gate.as_str(),
        }
    }

    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Alloc => "alloc",
            Self::Prepare => "prepare",
            Self::Measure => "measure",
            Self::MeasureReset => "measure_reset",
            Self::MeasureX => "measure_x",
            Self::MeasureY => "measure_y",
            Self::MeasureZ => "measure_z",
            Self::Discard => "discard",
            Self::Reset => "reset",
            Self::Release => "release",
            Self::Unprepare => "unprepare",
            Self::GlobalPhase => "global_phase",
            Self::PauliEvolution => "pauli_evolution",
            Self::AddInto => "add_into",
            Self::SubtractInto => "subtract_into",
            Self::AddConstant => "add_constant",
            Self::SubtractConstant => "subtract_constant",
            Self::ModAdd => "mod_add",
            Self::ModSub => "mod_sub",
            Self::ModAddConstant => "mod_add_constant",
            Self::ModSubConstant => "mod_sub_constant",
            Self::MulAdd => "mul_add",
            Self::MulSub => "mul_sub",
            Self::ModMulAdd => "mod_mul_add",
            Self::ModMulSub => "mod_mul_sub",
            Self::ModMul => "mod_mul",
            Self::ModDiv => "mod_div",
            Self::ModPow => "mod_pow",
            Self::ModPowInverse => "mod_pow_inverse",
            Self::ModMultiPow => "mod_multi_pow",
            Self::ModMultiPowInverse => "mod_multi_pow_inverse",
            Self::LessThanXor => "less_than_xor",
            Self::Gate(gate) => gate.name(),
        }
    }

    /// Return the exact quantum operation used by this operation's adjoint.
    #[must_use]
    pub(crate) const fn exact_adjoint(self) -> Option<Self> {
        match self {
            Self::GlobalPhase | Self::PauliEvolution => Some(self),
            Self::AddInto => Some(Self::SubtractInto),
            Self::SubtractInto => Some(Self::AddInto),
            Self::AddConstant => Some(Self::SubtractConstant),
            Self::SubtractConstant => Some(Self::AddConstant),
            Self::ModAdd => Some(Self::ModSub),
            Self::ModSub => Some(Self::ModAdd),
            Self::ModAddConstant => Some(Self::ModSubConstant),
            Self::ModSubConstant => Some(Self::ModAddConstant),
            Self::MulAdd => Some(Self::MulSub),
            Self::MulSub => Some(Self::MulAdd),
            Self::ModMulAdd => Some(Self::ModMulSub),
            Self::ModMulSub => Some(Self::ModMulAdd),
            Self::ModMul => Some(Self::ModDiv),
            Self::ModDiv => Some(Self::ModMul),
            Self::ModPow => Some(Self::ModPowInverse),
            Self::ModPowInverse => Some(Self::ModPow),
            Self::ModMultiPow => Some(Self::ModMultiPowInverse),
            Self::ModMultiPowInverse => Some(Self::ModMultiPow),
            Self::LessThanXor => Some(Self::LessThanXor),
            Self::Gate(gate) => Some(Self::Gate(gate.adjoint())),
            Self::Alloc
            | Self::Prepare
            | Self::Measure
            | Self::MeasureReset
            | Self::MeasureX
            | Self::MeasureY
            | Self::MeasureZ
            | Self::Discard
            | Self::Reset
            | Self::Release
            | Self::Unprepare => None,
        }
    }

    /// Return whether adjoint lowering negates this operation's scalar angle.
    #[must_use]
    pub(crate) const fn adjoint_negates_parameter(self) -> bool {
        matches!(self, Self::GlobalPhase | Self::PauliEvolution)
            || matches!(self, Self::Gate(gate) if gate.has_angle())
    }

    fn all() -> impl Iterator<Item = Self> {
        const OPERATIONS: &[QuantumOpcode] = &[
            QuantumOpcode::Alloc,
            QuantumOpcode::Prepare,
            QuantumOpcode::Measure,
            QuantumOpcode::MeasureReset,
            QuantumOpcode::MeasureX,
            QuantumOpcode::MeasureY,
            QuantumOpcode::MeasureZ,
            QuantumOpcode::Discard,
            QuantumOpcode::Reset,
            QuantumOpcode::Release,
            QuantumOpcode::Unprepare,
            QuantumOpcode::GlobalPhase,
            QuantumOpcode::PauliEvolution,
            QuantumOpcode::AddInto,
            QuantumOpcode::SubtractInto,
            QuantumOpcode::AddConstant,
            QuantumOpcode::SubtractConstant,
            QuantumOpcode::ModAdd,
            QuantumOpcode::ModSub,
            QuantumOpcode::ModAddConstant,
            QuantumOpcode::ModSubConstant,
            QuantumOpcode::MulAdd,
            QuantumOpcode::MulSub,
            QuantumOpcode::ModMul,
            QuantumOpcode::ModDiv,
            QuantumOpcode::ModPow,
            QuantumOpcode::ModPowInverse,
            QuantumOpcode::ModMultiPow,
            QuantumOpcode::ModMultiPowInverse,
            QuantumOpcode::LessThanXor,
        ];
        OPERATIONS
            .iter()
            .copied()
            .chain(GateOpcode::ALL.iter().copied().map(Self::Gate))
    }
}

impl fmt::Display for QuantumOpcode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// One operation in the closed native `QCore` vocabulary.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum Opcode {
    QCore(QCoreOpcode),
    Classical(ClassicalOpcode),
    Quantum(QuantumOpcode),
}

impl Opcode {
    /// Return whether this operation is allowed to own regions.
    #[must_use]
    pub(crate) const fn is_structured(self) -> bool {
        matches!(self, Self::QCore(operation) if operation.is_structured())
    }

    /// Return the canonical wire spelling.
    #[must_use]
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::QCore(operation) => operation.as_str(),
            Self::Classical(operation) => operation.as_str(),
            Self::Quantum(operation) => operation.as_str(),
        }
    }

    pub(crate) fn all() -> impl Iterator<Item = Self> {
        QCoreOpcode::ALL
            .iter()
            .copied()
            .map(Self::QCore)
            .chain(ClassicalOpcode::ALL.iter().copied().map(Self::Classical))
            .chain(QuantumOpcode::all().map(Self::Quantum))
    }
}

impl TryFrom<&str> for Opcode {
    type Error = OperationKernelError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        QCoreOpcode::parse(value)
            .map(Self::QCore)
            .or_else(|| ClassicalOpcode::parse(value).map(Self::Classical))
            .or_else(|| QuantumOpcode::parse(value).map(Self::Quantum))
            .ok_or_else(|| OperationKernelError::UnknownOpcode {
                opcode: value.to_owned(),
            })
    }
}

impl fmt::Display for Opcode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Return whether an opcode denotes one primitive reversible quantum step.
#[must_use]
pub(crate) fn is_primitive_adjointable(opcode: &str) -> bool {
    matches!(
        Opcode::try_from(opcode),
        Ok(
            Opcode::QCore(QCoreOpcode::PhaseBy | QCoreOpcode::XorBy | QCoreOpcode::PermuteBy)
                | Opcode::Quantum(
                    QuantumOpcode::Gate(_)
                        | QuantumOpcode::GlobalPhase
                        | QuantumOpcode::PauliEvolution
                        | QuantumOpcode::AddInto
                        | QuantumOpcode::SubtractInto
                        | QuantumOpcode::AddConstant
                        | QuantumOpcode::SubtractConstant
                        | QuantumOpcode::ModAdd
                        | QuantumOpcode::ModSub
                        | QuantumOpcode::ModAddConstant
                        | QuantumOpcode::ModSubConstant
                        | QuantumOpcode::MulAdd
                        | QuantumOpcode::MulSub
                        | QuantumOpcode::ModMul
                        | QuantumOpcode::ModDiv
                        | QuantumOpcode::ModPow
                        | QuantumOpcode::ModPowInverse
                        | QuantumOpcode::ModMultiPow
                        | QuantumOpcode::ModMultiPowInverse
                        | QuantumOpcode::LessThanXor
                )
        )
    )
}

closed_string_enum! {
    /// One terminator in the closed native `QCore` vocabulary.
    pub(crate) enum TerminatorKind {
        Return => "return",
        Unreachable => "unreachable",
        Yield => "yield",
        Break => "break",
        Continue => "continue",
        External => "external",
    }
}

impl TryFrom<&str> for TerminatorKind {
    type Error = OperationKernelError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::parse(value).ok_or_else(|| OperationKernelError::UnknownTerminator {
            kind: value.to_owned(),
        })
    }
}

/// Failures established by the native operation vocabulary pass.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub(crate) enum OperationKernelError {
    /// A block omitted its required terminator.
    #[error("Q1031: QCore block is missing a terminator")]
    MissingTerminator,
    /// A terminator lies outside the closed vocabulary.
    #[error("Q1031: unknown QCore terminator {kind:?}")]
    UnknownTerminator { kind: String },
    /// An operation lies outside the closed vocabulary.
    #[error("Q1032: unknown QCore opcode {opcode:?}")]
    UnknownOpcode { opcode: String },
    /// A leaf operation contains one or more regions.
    #[error("Q1033: {opcode} cannot contain nested regions")]
    RegionsOnNonStructuredOperation { opcode: Opcode },
    /// One function contains two definitions for the same SSA identity.
    #[error("Q1011: SSA value %{value_id} is defined more than once")]
    DuplicateValueId { value_id: u64 },
    /// An operand does not name any definition in its containing function.
    #[error("Q1002: SSA value %{value_id} is not defined in function {function:?}")]
    UndefinedValueId { function: String, value_id: u64 },
    /// A value definition could not be resolved through the shared type arena.
    #[error("Q4002: invalid semantic type for SSA value %{value_id}: {reason}")]
    InvalidValueType { value_id: u64, reason: String },
    /// A parameter omitted its required SSA value definition.
    #[error("Q4002: parameter {parameter:?} in function {function:?} has no SSA value")]
    MissingParameterValue { function: String, parameter: String },
    /// A classical operation violates its unique local type equation.
    #[error("Q1034: malformed {opcode} operation")]
    MalformedClassicalOperation { opcode: ClassicalOpcode },
    /// A locally derived semantic type could not be interned.
    #[error("Q4002: invalid derived semantic type: {0}")]
    InvalidDerivedType(String),
    /// The operation total cannot be represented on this platform.
    #[error("Q4007: QCore package exceeds the native operation-count capacity")]
    OperationCountOverflow,
}

/// Failure to derive one source-level classical result type.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum ClassicalTypeInferenceError {
    /// The requested opcode is outside the classical dialect.
    #[error("Q4502: unsupported classical operation {0:?}")]
    UnsupportedOpcode(String),
    /// An encoded operand type or attribute is malformed or exceeds a budget.
    #[error("Q4503: invalid classical type-inference input: {0}")]
    InvalidInput(String),
    /// The operation has no result for the supplied local type equation.
    #[error("Q4504: {opcode}: {contract}")]
    MalformedOperation {
        /// Canonical operation spelling used for machine inspection.
        opcode: String,
        /// Concise source-facing equation contract.
        contract: &'static str,
    },
}

const MAX_CLASSICAL_TYPE_OPERANDS: usize = 1_000_000;
const MAX_CLASSICAL_TYPE_ATTRIBUTES: usize = 16;

/// Derive one classical result type through the verifier's local equation.
///
/// Operand types, attributes, and the result use their canonical standalone
/// Protobuf encodings. The boundary is intentionally independent of Python so
/// every source adapter can ask the compiler for exactly the equation later
/// enforced on persistent `QCore`.
///
/// # Errors
///
/// Returns an error for an unknown opcode, malformed wire values, an exceeded
/// structural budget, or operands and attributes outside the opcode equation.
pub fn infer_classical_type(
    opcode: &str,
    operand_types: &[Vec<u8>],
    attributes: &[(String, Vec<u8>)],
) -> Result<Vec<u8>, ClassicalTypeInferenceError> {
    if operand_types.len() > MAX_CLASSICAL_TYPE_OPERANDS {
        return Err(ClassicalTypeInferenceError::InvalidInput(
            "operand count exceeds the local equation budget".to_owned(),
        ));
    }
    if attributes.len() > MAX_CLASSICAL_TYPE_ATTRIBUTES {
        return Err(ClassicalTypeInferenceError::InvalidInput(
            "attribute count exceeds the local equation budget".to_owned(),
        ));
    }
    let parsed_opcode = ClassicalOpcode::parse(opcode)
        .ok_or_else(|| ClassicalTypeInferenceError::UnsupportedOpcode(opcode.to_owned()))?;
    let mut arena = TypeArena::default();
    let scope = arena.intern_scope("source");
    let operands = operand_types
        .iter()
        .map(|payload| {
            arena
                .decode_canonical_wire_type(payload, &scope)
                .map_err(ClassicalTypeInferenceError::InvalidInput)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let attributes = attributes
        .iter()
        .map(|(name, payload)| {
            crate::budget::preflight_literal(payload).map_err(|error| {
                ClassicalTypeInferenceError::InvalidInput(format!("attribute {name:?}: {error:?}"))
            })?;
            let value = wire::LiteralProto::decode(payload.as_slice()).map_err(|error| {
                ClassicalTypeInferenceError::InvalidInput(format!("attribute {name:?}: {error}"))
            })?;
            if value.encode_to_vec() != *payload {
                return Err(ClassicalTypeInferenceError::InvalidInput(format!(
                    "attribute {name:?} is not canonical"
                )));
            }
            Ok(wire::AttributeProto {
                name: name.clone(),
                value: Some(value),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let operation = wire::OperationProto {
        opcode: opcode.to_owned(),
        attributes,
        ..wire::OperationProto::default()
    };
    let result =
        derive_classical_result_type(&operation, parsed_opcode, &operands, &mut arena, &scope)
            .map_err(|error| match error {
                EquationError::Malformed => ClassicalTypeInferenceError::MalformedOperation {
                    opcode: opcode.to_owned(),
                    contract: classical_equation_contract(parsed_opcode),
                },
                EquationError::InvalidDerivedType(reason) => {
                    ClassicalTypeInferenceError::InvalidInput(reason)
                }
            })?;
    arena
        .encode_wire_type(result)
        .map(|result| result.encode_to_vec())
        .map_err(ClassicalTypeInferenceError::InvalidInput)
}

const fn classical_equation_contract(opcode: ClassicalOpcode) -> &'static str {
    use ClassicalOpcode::{
        Add, And, AndBits, Array, Cast, Ceil, CompareChain, FloorDiv, Index, Invert, LeftShift,
        Len, Log2, MapGet, Mod, Mul, Neg, Not, ObservablePauli, ObservableZero, Or, OrBits, Pos,
        Pow, Project, RightShift, Select, Slice, Sqrt, Sub, TrueDiv, Update, Xor,
    };

    match opcode {
        Array => "array elements must be one or more local classical peers",
        ObservableZero => "classical.observable_zero expects one Value or Const Int logical width",
        ObservablePauli => {
            "classical.observable_pauli expects one Value or Const Int index and an X, Y, or Z axis"
        }
        Add | Sub | Mul | TrueDiv | Pow => {
            "binary arithmetic expects two compatible numeric scalar operands or a Const Observable algebra operation"
        }
        FloorDiv | Mod => {
            "floor division and modulo expect two compatible non-Complex numeric scalars"
        }
        AndBits | OrBits | Xor => "bitwise operations expect two Bool, Bit, or Int operands",
        LeftShift | RightShift => "shift operations expect two Int operands",
        And | Or => "classical boolean folds expect one or more Bool or Bit operands",
        Not => "classical.not expects one Bool or Bit operand",
        Invert => "classical.invert expects one Bool, Bit, or Int operand",
        Neg | Pos => {
            "classical sign operations expect one numeric scalar operand, or negation of a Const Observable"
        }
        Ceil | Log2 => "classical.ceil and classical.log2 expect one Int or Real operand",
        Sqrt => "classical.sqrt expects one Int or Real operand",
        CompareChain => {
            "classical.compare_chain operators and peer operands must form one valid chain"
        }
        Select => "classical.select alternatives must be classical peers selected by Bool or Bit",
        Index => "classical.index expects a homogeneous aggregate and one Int index",
        Slice => "classical.slice expects one array and literal or Const Int boundaries",
        Update => {
            "classical.update expects one classical array/word, an Int index, and a peer element"
        }
        MapGet => {
            "classical.map_get expects one Const fixed mapping entry array and one matching key"
        }
        Project => "classical.project expects one aggregate and an in-range static position",
        Cast => "classical.cast expects one operand and a supported Bool, Int, or Real target",
        Len => "classical.len expects one array, word, product, or record operand",
    }
}

/// Facts established while traversing the closed operation vocabulary.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct OperationSummary {
    operation_count: usize,
}

impl OperationSummary {
    /// Return the number of operations across all function and region blocks.
    #[must_use]
    pub(crate) const fn operation_count(self) -> usize {
        self.operation_count
    }
}

/// Function-scoped type roots retained by the locally checked program graph.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct FunctionValueTypes {
    function: Arc<str>,
    result_type: TypeId,
    values: HashMap<u64, TypeId>,
}

/// Operation facts and type roots established by the complete local pass.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct LocallyCheckedOperations {
    summary: OperationSummary,
    function_values: Box<[FunctionValueTypes]>,
    function_indices: HashMap<Arc<str>, usize>,
}

impl LocallyCheckedOperations {
    /// Return the number of checked operations in the retained graph.
    #[must_use]
    pub(crate) const fn operation_count(&self) -> usize {
        self.summary.operation_count()
    }

    /// Return the retained value/type roots for one function.
    #[must_use]
    pub(crate) fn value_types(&self, function: &str) -> Option<&HashMap<u64, TypeId>> {
        self.function_indices
            .get(function)
            .and_then(|index| self.function_values.get(*index))
            .map(|facts| &facts.values)
    }

    /// Return the retained callable result-type root for one function.
    #[must_use]
    pub(crate) fn result_type(&self, function: &str) -> Option<TypeId> {
        self.function_indices
            .get(function)
            .and_then(|index| self.function_values.get(*index))
            .map(|facts| facts.result_type)
    }
}

/// Validate operation and terminator names plus structural region ownership.
///
/// The work list follows the Python verifier's pre-order encounter order while
/// avoiding recursion through untrusted nested regions.
///
/// # Errors
///
/// Returns [`OperationKernelError`] for a missing or unknown terminator, an
/// unknown opcode, a region attached to a leaf operation, or an unrepresentable
/// operation total.
#[cfg(test)]
pub(crate) fn validate_program_operations(
    package: &wire::QCorePackageProto,
) -> Result<OperationSummary, OperationKernelError> {
    let mut work = QCoreVerificationWork::default();
    validate_program_operations_with_work(package, &mut work)
}

fn validate_program_operations_with_work(
    package: &wire::QCorePackageProto,
    verification_work: &mut QCoreVerificationWork,
) -> Result<OperationSummary, OperationKernelError> {
    enum Work<'a> {
        Block(&'a wire::BlockProto),
        Operation(&'a wire::OperationProto),
    }

    let mut operation_count = 0_usize;
    let mut traversal = Vec::new();

    for function in package.functions.iter().rev() {
        verification_work.charge_operation_kernel(1);
        if let Some(body) = &function.body {
            traversal.push(Work::Block(body));
        }
    }

    while let Some(item) = traversal.pop() {
        match item {
            Work::Block(block) => {
                verification_work.charge_operation_kernel(2);
                let terminator = block
                    .terminator
                    .as_ref()
                    .ok_or(OperationKernelError::MissingTerminator)?;
                TerminatorKind::try_from(terminator.kind.as_str())?;

                traversal.extend(block.operations.iter().rev().map(Work::Operation));
            }
            Work::Operation(operation) => {
                verification_work.charge_operation_kernel(1);
                operation_count = operation_count
                    .checked_add(1)
                    .ok_or(OperationKernelError::OperationCountOverflow)?;
                let opcode = Opcode::try_from(operation.opcode.as_str())?;
                if !opcode.is_structured() && !operation.regions.is_empty() {
                    return Err(OperationKernelError::RegionsOnNonStructuredOperation { opcode });
                }

                if opcode.is_structured() {
                    for region in operation.regions.iter().rev() {
                        traversal.extend(region.blocks.iter().rev().map(Work::Block));
                    }
                }
            }
        }
    }

    Ok(OperationSummary { operation_count })
}

/// Validate the closed dialect, function-local identities, and classical equations.
///
/// Operand lookup intentionally uses the complete function-local definition
/// index. This establishes referential integrity without claiming lexical
/// visibility or dominance; a later SSA pass must still reject forward,
/// sibling-region, and non-dominating uses.
///
/// # Errors
///
/// Returns [`OperationKernelError`] for structural dialect failures, duplicate
/// or unresolved `ValueId`s, malformed value types, or a classical operation
/// whose claimed result disagrees with its unique local equation.
#[cfg(test)]
pub(crate) fn validate_program_operations_with_types(
    package: &wire::QCorePackageProto,
    arena: &mut TypeArena,
) -> Result<LocallyCheckedOperations, OperationKernelError> {
    let mut work = QCoreVerificationWork::default();
    validate_program_operations_with_types_and_work(package, arena, &mut work)
}

pub(crate) fn validate_program_operations_with_types_and_work(
    package: &wire::QCorePackageProto,
    arena: &mut TypeArena,
    work: &mut QCoreVerificationWork,
) -> Result<LocallyCheckedOperations, OperationKernelError> {
    let summary = validate_program_operations_with_work(package, work)?;
    let mut function_values = Vec::with_capacity(package.functions.len());
    let mut function_indices = HashMap::with_capacity(package.functions.len());
    for function in &package.functions {
        work.charge_operation_kernel(2);
        let scope = arena.intern_scope(&function.symbol);
        let result_type = function
            .result_type
            .as_ref()
            .ok_or_else(|| OperationKernelError::InvalidValueType {
                value_id: u64::MAX,
                reason: format!("function {:?} has no result type", function.symbol),
            })
            .and_then(|source| {
                arena.resolve_wire_type(source, &scope).map_err(|reason| {
                    OperationKernelError::InvalidValueType {
                        value_id: u64::MAX,
                        reason,
                    }
                })
            })?;
        let values = index_function_values(function, &scope, arena, work)?;
        if let Some(body) = &function.body {
            validate_function_uses(function, body, &values, arena, work)?;
        }
        let index = function_values.len();
        function_indices.insert(Arc::clone(&scope), index);
        function_values.push(FunctionValueTypes {
            function: scope,
            result_type,
            values,
        });
    }
    Ok(LocallyCheckedOperations {
        summary,
        function_values: function_values.into_boxed_slice(),
        function_indices,
    })
}

fn index_function_values(
    function: &wire::FunctionProto,
    scope: &Arc<str>,
    arena: &mut TypeArena,
    work: &mut QCoreVerificationWork,
) -> Result<HashMap<u64, TypeId>, OperationKernelError> {
    let mut values = HashMap::new();
    for parameter in &function.parameters {
        let value = parameter.value.as_ref().ok_or_else(|| {
            OperationKernelError::MissingParameterValue {
                function: function.symbol.clone(),
                parameter: parameter.name.clone(),
            }
        })?;
        define_value(value, scope, arena, &mut values, work)?;
    }

    let mut blocks = Vec::new();
    if let Some(body) = &function.body {
        blocks.push(body);
    }
    while let Some(block) = blocks.pop() {
        work.charge_operation_kernel(1);
        for argument in &block.arguments {
            define_value(argument, scope, arena, &mut values, work)?;
        }
        for operation in &block.operations {
            work.charge_operation_kernel(1);
            for result in &operation.results {
                define_value(result, scope, arena, &mut values, work)?;
            }
            for region in &operation.regions {
                blocks.extend(&region.blocks);
            }
        }
    }
    Ok(values)
}

fn define_value(
    value: &wire::ValueProto,
    scope: &Arc<str>,
    arena: &mut TypeArena,
    values: &mut HashMap<u64, TypeId>,
    work: &mut QCoreVerificationWork,
) -> Result<(), OperationKernelError> {
    work.charge_operation_kernel(1);
    if values.contains_key(&value.id) {
        return Err(OperationKernelError::DuplicateValueId { value_id: value.id });
    }
    let source = value
        .r#type
        .as_ref()
        .ok_or_else(|| OperationKernelError::InvalidValueType {
            value_id: value.id,
            reason: "value has no semantic type".to_owned(),
        })?;
    let type_id = arena.resolve_wire_type(source, scope).map_err(|reason| {
        OperationKernelError::InvalidValueType {
            value_id: value.id,
            reason,
        }
    })?;
    values.insert(value.id, type_id);
    Ok(())
}

fn validate_function_uses(
    function: &wire::FunctionProto,
    body: &wire::BlockProto,
    values: &HashMap<u64, TypeId>,
    arena: &mut TypeArena,
    verification_work: &mut QCoreVerificationWork,
) -> Result<(), OperationKernelError> {
    enum Work<'a> {
        Block(&'a wire::BlockProto),
        Operation(&'a wire::OperationProto),
        Terminator(&'a wire::TerminatorProto),
    }

    let mut traversal = vec![Work::Block(body)];
    while let Some(item) = traversal.pop() {
        match item {
            Work::Block(block) => {
                verification_work.charge_operation_kernel(1);
                if let Some(terminator) = &block.terminator {
                    traversal.push(Work::Terminator(terminator));
                }
                traversal.extend(block.operations.iter().rev().map(Work::Operation));
            }
            Work::Operation(operation) => {
                verification_work.charge_operation_kernel(1);
                let operands =
                    resolve_operands(&operation.operands, function, values, verification_work)?;
                if let Opcode::Classical(opcode) = Opcode::try_from(operation.opcode.as_str())? {
                    verify_classical_operation(
                        operation,
                        opcode,
                        &operands,
                        values,
                        arena,
                        &std::sync::Arc::from(function.symbol.as_str()),
                    )?;
                }
                for region in operation.regions.iter().rev() {
                    traversal.extend(region.blocks.iter().rev().map(Work::Block));
                }
            }
            Work::Terminator(terminator) => {
                verification_work.charge_operation_kernel(1);
                resolve_operands(&terminator.operands, function, values, verification_work)?;
            }
        }
    }
    Ok(())
}

fn resolve_operands(
    operands: &[u64],
    function: &wire::FunctionProto,
    values: &HashMap<u64, TypeId>,
    work: &mut QCoreVerificationWork,
) -> Result<Vec<TypeId>, OperationKernelError> {
    work.charge_operation_kernel(operands.len());
    operands
        .iter()
        .map(|value_id| {
            values
                .get(value_id)
                .copied()
                .ok_or_else(|| OperationKernelError::UndefinedValueId {
                    function: function.symbol.clone(),
                    value_id: *value_id,
                })
        })
        .collect()
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum EquationError {
    Malformed,
    InvalidDerivedType(String),
}

type EquationResult<T> = Result<T, EquationError>;

fn verify_classical_operation(
    operation: &wire::OperationProto,
    opcode: ClassicalOpcode,
    operands: &[TypeId],
    values: &HashMap<u64, TypeId>,
    arena: &mut TypeArena,
    scope: &std::sync::Arc<str>,
) -> Result<(), OperationKernelError> {
    let malformed = || OperationKernelError::MalformedClassicalOperation { opcode };
    if operands.iter().any(|operand| arena.is_quantum(*operand))
        || operation.results.len() != 1
        || operation.results[0].has_region
        || operation.results[0].region.is_some()
    {
        return Err(malformed());
    }
    let result = values
        .get(&operation.results[0].id)
        .copied()
        .ok_or_else(malformed)?;
    let expected = match derive_classical_result_type(operation, opcode, operands, arena, scope) {
        Ok(expected) => expected,
        Err(EquationError::Malformed) => return Err(malformed()),
        Err(EquationError::InvalidDerivedType(reason)) => {
            return Err(OperationKernelError::InvalidDerivedType(reason));
        }
    };
    if result != expected {
        return Err(malformed());
    }
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn derive_classical_result_type(
    operation: &wire::OperationProto,
    opcode: ClassicalOpcode,
    operands: &[TypeId],
    arena: &mut TypeArena,
    scope: &std::sync::Arc<str>,
) -> EquationResult<TypeId> {
    use ClassicalOpcode::{
        Add, And, AndBits, Array, Cast, Ceil, CompareChain, FloorDiv, Index, Invert, LeftShift,
        Len, Log2, MapGet, Mod, Mul, Neg, Not, ObservablePauli, ObservableZero, Or, OrBits, Pos,
        Pow, Project, RightShift, Select, Slice, Sqrt, Sub, TrueDiv, Update, Xor,
    };

    match opcode {
        Array => infer_array(operation, operands, arena),
        ObservableZero => infer_observable_zero(operation, operands, arena),
        ObservablePauli => infer_observable_pauli(operation, operands, arena),
        Add | Sub | Mul | TrueDiv | FloorDiv | Mod | Pow | AndBits | OrBits | Xor | LeftShift
        | RightShift => infer_binary(operation, opcode, operands, arena),
        And | Or => {
            require_no_attributes(operation)?;
            if operands.is_empty()
                || operands.iter().any(|operand| {
                    !matches!(
                        base_scalar(arena, *operand),
                        Some(ScalarType::Bool | ScalarType::Bit)
                    )
                })
            {
                return Err(EquationError::Malformed);
            }
            scalar_at_stage(arena, ScalarType::Bool, common_stage(arena, operands))
        }
        Not | Invert | Neg | Pos | Ceil | Log2 | Sqrt => {
            require_no_attributes(operation)?;
            let [operand] = operands else {
                return Err(EquationError::Malformed);
            };
            let Some(base) = base_scalar(arena, *operand) else {
                return Err(EquationError::Malformed);
            };
            match opcode {
                Not if matches!(base, ScalarType::Bool | ScalarType::Bit) => {
                    scalar_at_stage(arena, ScalarType::Bool, common_stage(arena, operands))
                }
                Invert if matches!(base, ScalarType::Bool | ScalarType::Bit | ScalarType::Int) => {
                    Ok(*operand)
                }
                Neg | Pos
                    if matches!(
                        base,
                        ScalarType::Int | ScalarType::Real | ScalarType::Complex
                    ) =>
                {
                    Ok(*operand)
                }
                Neg if base == ScalarType::Observable && const_available(arena, *operand) => {
                    scalar_at_stage(arena, ScalarType::Observable, Some(Stage::Const))
                }
                Ceil if matches!(base, ScalarType::Int | ScalarType::Real) => {
                    scalar_at_stage(arena, ScalarType::Int, common_stage(arena, operands))
                }
                Log2 | Sqrt if matches!(base, ScalarType::Int | ScalarType::Real) => {
                    scalar_at_stage(arena, ScalarType::Real, common_stage(arena, operands))
                }
                _ => Err(EquationError::Malformed),
            }
        }
        CompareChain => infer_compare_chain(operation, operands, arena),
        Select => infer_select(operation, operands, arena),
        Index => infer_index(operation, operands, arena),
        Slice => infer_slice(operation, operands, arena, scope),
        Update => infer_update(operation, operands, arena),
        MapGet => infer_map_get(operation, operands, arena),
        Project => infer_project(operation, operands, arena),
        Cast => infer_cast(operation, operands, arena),
        Len => infer_len(operation, operands, arena),
    }
}

fn infer_binary(
    operation: &wire::OperationProto,
    opcode: ClassicalOpcode,
    operands: &[TypeId],
    arena: &mut TypeArena,
) -> EquationResult<TypeId> {
    use ClassicalOpcode::{
        Add, AndBits, FloorDiv, LeftShift, Mod, Mul, OrBits, RightShift, Sub, TrueDiv, Xor,
    };

    require_no_attributes(operation)?;
    let [left, right] = operands else {
        return Err(EquationError::Malformed);
    };
    let (Some(left_base), Some(right_base)) =
        (base_scalar(arena, *left), base_scalar(arena, *right))
    else {
        return Err(EquationError::Malformed);
    };
    if left_base == ScalarType::Observable || right_base == ScalarType::Observable {
        let numeric = |value| {
            matches!(
                value,
                ScalarType::Int | ScalarType::Real | ScalarType::Complex
            )
        };
        let valid = match opcode {
            Add | Sub | Mul => {
                (left_base == ScalarType::Observable
                    && (right_base == ScalarType::Observable || numeric(right_base)))
                    || (right_base == ScalarType::Observable && numeric(left_base))
            }
            _ => false,
        };
        if !valid
            || operands
                .iter()
                .any(|operand| !const_available(arena, *operand))
        {
            return Err(EquationError::Malformed);
        }
        return scalar_at_stage(arena, ScalarType::Observable, Some(Stage::Const));
    }
    let result = match opcode {
        LeftShift | RightShift if left_base == ScalarType::Int && right_base == ScalarType::Int => {
            ScalarType::Int
        }
        AndBits | OrBits | Xor
            if matches!(
                left_base,
                ScalarType::Bool | ScalarType::Bit | ScalarType::Int
            ) && matches!(
                right_base,
                ScalarType::Bool | ScalarType::Bit | ScalarType::Int
            ) =>
        {
            if matches!(left_base, ScalarType::Bool | ScalarType::Bit)
                && matches!(right_base, ScalarType::Bool | ScalarType::Bit)
            {
                ScalarType::Bool
            } else {
                ScalarType::Int
            }
        }
        LeftShift | RightShift | AndBits | OrBits | Xor => {
            return Err(EquationError::Malformed);
        }
        _ => {
            let left_rank = numeric_rank(left_base).ok_or(EquationError::Malformed)?;
            let right_rank = numeric_rank(right_base).ok_or(EquationError::Malformed)?;
            let mut numeric = scalar_for_numeric_rank(left_rank.max(right_rank));
            if opcode == TrueDiv && numeric == ScalarType::Int {
                numeric = ScalarType::Real;
            }
            if matches!(opcode, FloorDiv | Mod) && numeric == ScalarType::Complex {
                return Err(EquationError::Malformed);
            }
            numeric
        }
    };
    scalar_at_stage(arena, result, common_stage(arena, operands))
}

fn infer_observable_zero(
    operation: &wire::OperationProto,
    operands: &[TypeId],
    arena: &mut TypeArena,
) -> EquationResult<TypeId> {
    require_no_attributes(operation)?;
    let [width] = operands else {
        return Err(EquationError::Malformed);
    };
    if base_scalar(arena, *width) != Some(ScalarType::Int) || !const_available(arena, *width) {
        return Err(EquationError::Malformed);
    }
    scalar_at_stage(arena, ScalarType::Observable, Some(Stage::Const))
}

fn infer_observable_pauli(
    operation: &wire::OperationProto,
    operands: &[TypeId],
    arena: &mut TypeArena,
) -> EquationResult<TypeId> {
    let [index] = operands else {
        return Err(EquationError::Malformed);
    };
    if base_scalar(arena, *index) != Some(ScalarType::Int)
        || !const_available(arena, *index)
        || !matches!(string_attribute(operation, "axis"), Ok("X" | "Y" | "Z"))
    {
        return Err(EquationError::Malformed);
    }
    scalar_at_stage(arena, ScalarType::Observable, Some(Stage::Const))
}

fn const_available(arena: &TypeArena, type_id: TypeId) -> bool {
    arena
        .latest_availability(type_id)
        .is_none_or(|stage| stage == Stage::Const)
}

fn infer_array(
    operation: &wire::OperationProto,
    operands: &[TypeId],
    arena: &mut TypeArena,
) -> EquationResult<TypeId> {
    require_no_attributes(operation)?;
    let first = *operands.first().ok_or(EquationError::Malformed)?;
    if operands
        .iter()
        .skip(1)
        .any(|operand| !arena.local_peers(first, *operand))
    {
        return Err(EquationError::Malformed);
    }
    let extent = u64::try_from(operands.len()).map_err(|_| EquationError::Malformed)?;
    let base = arena.strip_stage(first);
    let derived = match arena.get(base).cloned() {
        Some(SemanticType::Array { element, mut shape }) => {
            shape.insert(0, crate::types::Dimension::Fixed(extent));
            SemanticType::Array { element, shape }
        }
        Some(_) => SemanticType::Array {
            element: base,
            shape: vec![crate::types::Dimension::Fixed(extent)],
        },
        None => return Err(EquationError::Malformed),
    };
    let array = intern(arena, derived)?;
    at_stage(arena, array, common_stage(arena, operands))
}

fn infer_select(
    operation: &wire::OperationProto,
    operands: &[TypeId],
    arena: &mut TypeArena,
) -> EquationResult<TypeId> {
    require_no_attributes(operation)?;
    let [condition, then_value, else_value] = operands else {
        return Err(EquationError::Malformed);
    };
    if !matches!(
        base_scalar(arena, *condition),
        Some(ScalarType::Bool | ScalarType::Bit)
    ) || arena.is_quantum(*then_value)
        || !arena.local_peers(*then_value, *else_value)
    {
        return Err(EquationError::Malformed);
    }
    at_stage(
        arena,
        arena.strip_stage(*then_value),
        common_stage(arena, operands),
    )
}

fn infer_index(
    operation: &wire::OperationProto,
    operands: &[TypeId],
    arena: &mut TypeArena,
) -> EquationResult<TypeId> {
    require_no_attributes(operation)?;
    let [collection, index] = operands else {
        return Err(EquationError::Malformed);
    };
    if base_scalar(arena, *index) != Some(ScalarType::Int) {
        return Err(EquationError::Malformed);
    }
    let element = if matches!(
        arena.get(arena.strip_stage(*collection)),
        Some(SemanticType::Array { .. } | SemanticType::Word { .. })
    ) {
        sequence_element_type(arena, *collection)?
    } else {
        arena
            .homogeneous_aggregate_base(*collection)
            .ok_or(EquationError::Malformed)?
    };
    at_stage(
        arena,
        arena.strip_stage(element),
        common_stage(arena, operands),
    )
}

fn infer_slice(
    operation: &wire::OperationProto,
    operands: &[TypeId],
    arena: &mut TypeArena,
    scope: &std::sync::Arc<str>,
) -> EquationResult<TypeId> {
    use crate::core::slice_shape::{SliceBound, SliceShape, operation_bounds};
    let Some((collection, bound_operands)) = operands.split_first() else {
        return Err(EquationError::Malformed);
    };
    let bounds = operation_bounds(operation).map_err(EquationError::InvalidDerivedType)?;
    if bounds.iter().filter(|b| **b != SliceBound::Omitted).count() != bound_operands.len()
        || bound_operands.iter().any(|operand| {
            base_scalar(arena, *operand) != Some(ScalarType::Int)
                || !const_available(arena, *operand)
        })
    {
        return Err(EquationError::Malformed);
    }
    let Some(SemanticType::Array { element, shape }) =
        arena.get(arena.strip_stage(*collection)).cloned()
    else {
        return Err(EquationError::Malformed);
    };
    let extent = match &shape[0] {
        Dimension::Fixed(n) => n.to_string(),
        Dimension::Symbol { name, .. } => name.to_string(),
    };
    let extent = SliceShape { extent, bounds }
        .dimension(scope)
        .map_err(EquationError::InvalidDerivedType)?;
    let mut shape = shape;
    shape[0] = extent;
    let result = arena
        .intern_derived(SemanticType::Array { element, shape })
        .map_err(EquationError::InvalidDerivedType)?;
    at_stage(arena, result, common_stage(arena, operands))
}

fn infer_update(
    operation: &wire::OperationProto,
    operands: &[TypeId],
    arena: &mut TypeArena,
) -> EquationResult<TypeId> {
    require_no_attributes(operation)?;
    let [collection, index, replacement] = operands else {
        return Err(EquationError::Malformed);
    };
    if base_scalar(arena, *index) != Some(ScalarType::Int)
        || !matches!(
            arena.get(arena.strip_stage(*collection)),
            Some(SemanticType::Array { .. } | SemanticType::Word { .. })
        )
        || arena.is_quantum(*collection)
        || arena.is_quantum(*replacement)
    {
        return Err(EquationError::Malformed);
    }
    let element = sequence_element_type(arena, *collection)?;
    if !(arena.local_peers(element, *replacement)
        || (base_scalar(arena, element) == Some(ScalarType::Bit)
            && base_scalar(arena, *replacement) == Some(ScalarType::Bool)))
    {
        return Err(EquationError::Malformed);
    }
    at_stage(
        arena,
        arena.strip_stage(*collection),
        common_stage(arena, operands),
    )
}

fn infer_map_get(
    operation: &wire::OperationProto,
    operands: &[TypeId],
    arena: &mut TypeArena,
) -> EquationResult<TypeId> {
    require_no_attributes(operation)?;
    let [mapping, key] = operands else {
        return Err(EquationError::Malformed);
    };
    if arena.latest_availability(*mapping) != Some(Stage::Const) {
        return Err(EquationError::Malformed);
    }
    let mapping_base = arena.strip_stage(*mapping);
    let Some(SemanticType::Array { element, shape }) = arena.get(mapping_base) else {
        return Err(EquationError::Malformed);
    };
    if shape.len() != 1 {
        return Err(EquationError::Malformed);
    }
    let Some(SemanticType::Record { name, fields }) = arena.get(*element) else {
        return Err(EquationError::Malformed);
    };
    if name != "__qamomile_mapping_entry__"
        || fields.len() != 2
        || fields[0].0 != "key"
        || fields[1].0 != "value"
        || !arena.local_peers(fields[0].1, *key)
    {
        return Err(EquationError::Malformed);
    }
    at_stage(
        arena,
        arena.strip_stage(fields[1].1),
        common_stage(arena, operands),
    )
}

fn infer_project(
    operation: &wire::OperationProto,
    operands: &[TypeId],
    arena: &mut TypeArena,
) -> EquationResult<TypeId> {
    let [operand] = operands else {
        return Err(EquationError::Malformed);
    };
    let position = int_attribute(operation, "position")?;
    let length = arena
        .aggregate_len(*operand)
        .ok_or(EquationError::Malformed)?;
    let index = normalize_aggregate_index(length, &BigInt::from(position))
        .ok_or(EquationError::Malformed)?;
    arena
        .aggregate_element_at(*operand, index)
        .map_err(EquationError::InvalidDerivedType)?
        .ok_or(EquationError::Malformed)
}

fn infer_cast(
    operation: &wire::OperationProto,
    operands: &[TypeId],
    arena: &mut TypeArena,
) -> EquationResult<TypeId> {
    let [operand] = operands else {
        return Err(EquationError::Malformed);
    };
    let target = string_attribute(operation, "target")?;
    let target = match target {
        "Bool" => ScalarType::Bool,
        "Int" => ScalarType::Int,
        "Real" => ScalarType::Real,
        _ => return Err(EquationError::Malformed),
    };
    let source = base_scalar(arena, *operand).ok_or(EquationError::Malformed)?;
    let allowed = if target == ScalarType::Bool {
        matches!(
            source,
            ScalarType::Bool | ScalarType::Bit | ScalarType::Int | ScalarType::Real
        )
    } else {
        matches!(source, ScalarType::Int | ScalarType::Real)
    };
    if !allowed {
        return Err(EquationError::Malformed);
    }
    scalar_at_stage(arena, target, common_stage(arena, operands))
}

fn infer_len(
    operation: &wire::OperationProto,
    operands: &[TypeId],
    arena: &mut TypeArena,
) -> EquationResult<TypeId> {
    let shape_only = if operation.attributes.is_empty() {
        false
    } else if matches!(
        single_attribute(operation, "shape_only")?.value,
        Some(wire::literal_proto::Value::BoolValue(true))
    ) {
        true
    } else {
        return Err(EquationError::Malformed);
    };
    let [operand] = operands else {
        return Err(EquationError::Malformed);
    };
    if !matches!(
        arena.get(arena.strip_stage(*operand)),
        Some(
            SemanticType::Array { .. }
                | SemanticType::Word { .. }
                | SemanticType::Product(_)
                | SemanticType::Record { .. }
        )
    ) {
        return Err(EquationError::Malformed);
    }
    // A type's extent is available before its elements. Structured iteration
    // requests this projection explicitly, preserving ordinary len staging.
    scalar_at_stage(
        arena,
        ScalarType::Int,
        if shape_only {
            Some(Stage::Const)
        } else {
            common_stage(arena, operands)
        },
    )
}

fn infer_compare_chain(
    operation: &wire::OperationProto,
    operands: &[TypeId],
    arena: &mut TypeArena,
) -> EquationResult<TypeId> {
    let operators = string_tuple_attribute(operation, "operators")?;
    if operands.len() < 2 || operators.len() != operands.len() - 1 {
        return Err(EquationError::Malformed);
    }
    for (operator, pair) in operators.iter().zip(operands.windows(2)) {
        verify_comparison(operator, pair[0], pair[1], arena)?;
    }
    scalar_at_stage(
        arena,
        ScalarType::Bool,
        operands.iter().fold(None, |stage, operand| {
            Stage::join(stage, arena.latest_availability(*operand))
        }),
    )
}

fn verify_comparison(
    comparison: &str,
    left: TypeId,
    right: TypeId,
    arena: &mut TypeArena,
) -> EquationResult<()> {
    let left_base = base_scalar(arena, left);
    let right_base = base_scalar(arena, right);
    match comparison {
        "lt" | "le" | "gt" | "ge"
            if matches!(left_base, Some(ScalarType::Int | ScalarType::Real))
                && matches!(right_base, Some(ScalarType::Int | ScalarType::Real)) =>
        {
            Ok(())
        }
        "in" | "not_in"
            if matches!(
                arena.get(arena.strip_stage(right)),
                Some(SemanticType::Array { .. } | SemanticType::Word { .. })
            ) =>
        {
            let element = sequence_element_type(arena, right)?;
            if arena.local_peers(element, left) {
                Ok(())
            } else {
                Err(EquationError::Malformed)
            }
        }
        "eq" | "ne" if left_base.is_some_and(is_numeric) && right_base.is_some_and(is_numeric) => {
            Ok(())
        }
        "eq" | "ne" if arena.local_peers(left, right) => Ok(()),
        _ => Err(EquationError::Malformed),
    }
}

fn sequence_element_type(arena: &mut TypeArena, sequence: TypeId) -> EquationResult<TypeId> {
    arena
        .sequence_element(sequence)
        .map_err(EquationError::InvalidDerivedType)?
        .ok_or(EquationError::Malformed)
}

fn base_scalar(arena: &TypeArena, type_id: TypeId) -> Option<ScalarType> {
    match arena.get(arena.strip_stage(type_id)) {
        Some(SemanticType::Scalar(scalar)) => Some(*scalar),
        _ => None,
    }
}

fn numeric_rank(scalar: ScalarType) -> Option<u8> {
    match scalar {
        ScalarType::Int => Some(0),
        ScalarType::Real => Some(1),
        ScalarType::Complex => Some(2),
        ScalarType::Unit
        | ScalarType::Bit
        | ScalarType::Bool
        | ScalarType::Probability
        | ScalarType::Observable => None,
    }
}

const fn scalar_for_numeric_rank(rank: u8) -> ScalarType {
    match rank {
        0 => ScalarType::Int,
        1 => ScalarType::Real,
        _ => ScalarType::Complex,
    }
}

fn is_numeric(scalar: ScalarType) -> bool {
    numeric_rank(scalar).is_some()
}

fn common_stage(arena: &TypeArena, operands: &[TypeId]) -> Option<Stage> {
    operands.iter().fold(None, |stage, operand| {
        Stage::join(stage, arena.stage_of(*operand))
    })
}

fn scalar_at_stage(
    arena: &mut TypeArena,
    scalar: ScalarType,
    stage: Option<Stage>,
) -> EquationResult<TypeId> {
    let base = intern(arena, SemanticType::Scalar(scalar))?;
    at_stage(arena, base, stage)
}

fn at_stage(
    arena: &mut TypeArena,
    type_id: TypeId,
    stage: Option<Stage>,
) -> EquationResult<TypeId> {
    let inner = arena.strip_stage(type_id);
    match stage {
        Some(stage) => intern(arena, SemanticType::Staged { stage, inner }),
        None => Ok(inner),
    }
}

fn intern(arena: &mut TypeArena, type_spec: SemanticType) -> EquationResult<TypeId> {
    arena
        .intern_derived(type_spec)
        .map_err(EquationError::InvalidDerivedType)
}

fn require_no_attributes(operation: &wire::OperationProto) -> EquationResult<()> {
    if operation.attributes.is_empty() {
        Ok(())
    } else {
        Err(EquationError::Malformed)
    }
}

fn single_attribute<'a>(
    operation: &'a wire::OperationProto,
    expected: &str,
) -> EquationResult<&'a wire::LiteralProto> {
    let [attribute] = operation.attributes.as_slice() else {
        return Err(EquationError::Malformed);
    };
    if attribute.name != expected {
        return Err(EquationError::Malformed);
    }
    attribute.value.as_ref().ok_or(EquationError::Malformed)
}

fn int_attribute(operation: &wire::OperationProto, expected: &str) -> EquationResult<i64> {
    match single_attribute(operation, expected)?.value {
        Some(wire::literal_proto::Value::IntValue(value)) => Ok(value),
        _ => Err(EquationError::Malformed),
    }
}

fn string_attribute<'a>(
    operation: &'a wire::OperationProto,
    expected: &str,
) -> EquationResult<&'a str> {
    match &single_attribute(operation, expected)?.value {
        Some(wire::literal_proto::Value::StringValue(value)) => Ok(value),
        _ => Err(EquationError::Malformed),
    }
}

fn string_tuple_attribute<'a>(
    operation: &'a wire::OperationProto,
    expected: &str,
) -> EquationResult<Vec<&'a str>> {
    let Some(wire::literal_proto::Value::TupleValue(sequence)) =
        &single_attribute(operation, expected)?.value
    else {
        return Err(EquationError::Malformed);
    };
    sequence
        .values
        .iter()
        .map(|value| match &value.value {
            Some(wire::literal_proto::Value::StringValue(value)) => Ok(value.as_str()),
            _ => Err(EquationError::Malformed),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn array_type(element: wire::TypeProto, dimensions: &[u64]) -> wire::TypeProto {
        wire::TypeProto {
            kind: wire::type_proto::Kind::Array.into(),
            name: "Array".to_owned(),
            arguments: vec![element],
            dimensions: dimensions
                .iter()
                .map(|dimension| wire::DimensionProto {
                    value: Some(wire::dimension_proto::Value::Fixed(*dimension)),
                })
                .collect(),
            ..Default::default()
        }
    }

    fn word_type(kind: &str, width: u64) -> wire::TypeProto {
        wire::TypeProto {
            kind: wire::type_proto::Kind::Array.into(),
            name: kind.to_owned(),
            arguments: vec![scalar("Bit")],
            dimensions: vec![wire::DimensionProto {
                value: Some(wire::dimension_proto::Value::Fixed(width)),
            }],
            ..Default::default()
        }
    }

    fn symbolic_word_type(kind: &str, width: &str) -> wire::TypeProto {
        wire::TypeProto {
            kind: wire::type_proto::Kind::Array.into(),
            name: kind.to_owned(),
            arguments: vec![scalar("Bit")],
            dimensions: vec![wire::DimensionProto {
                value: Some(wire::dimension_proto::Value::Symbol(width.to_owned())),
            }],
            ..Default::default()
        }
    }

    fn product_type(elements: Vec<wire::TypeProto>) -> wire::TypeProto {
        wire::TypeProto {
            kind: wire::type_proto::Kind::Product.into(),
            arguments: elements,
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

    fn attribute(name: &str, value: wire::literal_proto::Value) -> wire::AttributeProto {
        wire::AttributeProto {
            name: name.to_owned(),
            value: Some(wire::LiteralProto { value: Some(value) }),
        }
    }

    fn typed_package(
        operand_types: Vec<wire::TypeProto>,
        opcode: &str,
        result_type: wire::TypeProto,
        attributes: Vec<wire::AttributeProto>,
    ) -> wire::QCorePackageProto {
        let result_id = u64::try_from(operand_types.len()).expect("small test arity") + 1;
        let parameters = operand_types
            .into_iter()
            .enumerate()
            .map(|(index, r#type)| wire::ParameterProto {
                name: format!("operand_{index}"),
                value: Some(value(
                    u64::try_from(index).expect("small test index") + 1,
                    r#type,
                )),
                ..Default::default()
            })
            .collect::<Vec<_>>();
        let operands = parameters
            .iter()
            .map(|parameter| parameter.value.as_ref().expect("test value").id)
            .collect();
        wire::QCorePackageProto {
            functions: vec![wire::FunctionProto {
                symbol: "native_test.main".to_owned(),
                parameters,
                result_type: Some(scalar("Unit")),
                body: Some(wire::BlockProto {
                    operations: vec![wire::OperationProto {
                        opcode: opcode.to_owned(),
                        operands,
                        results: vec![value(result_id, result_type)],
                        attributes,
                        ..Default::default()
                    }],
                    terminator: Some(wire::TerminatorProto {
                        kind: "return".to_owned(),
                        operands: vec![result_id],
                        ..Default::default()
                    }),
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn validate_typed(
        package: &wire::QCorePackageProto,
    ) -> Result<LocallyCheckedOperations, OperationKernelError> {
        let mut arena = crate::types::validate_program_types(package).expect("valid test types");
        validate_program_operations_with_types(package, &mut arena)
    }

    fn operation(opcode: &str) -> wire::OperationProto {
        wire::OperationProto {
            opcode: opcode.to_owned(),
            ..Default::default()
        }
    }

    fn terminator(kind: &str) -> wire::TerminatorProto {
        wire::TerminatorProto {
            kind: kind.to_owned(),
            ..Default::default()
        }
    }

    fn block(operations: Vec<wire::OperationProto>, terminator_kind: &str) -> wire::BlockProto {
        wire::BlockProto {
            operations,
            terminator: Some(terminator(terminator_kind)),
            ..Default::default()
        }
    }

    fn package(body: wire::BlockProto) -> wire::QCorePackageProto {
        wire::QCorePackageProto {
            functions: vec![wire::FunctionProto {
                body: Some(body),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn every_closed_opcode_round_trips() {
        let operations = Opcode::all().collect::<Vec<_>>();

        assert_eq!(operations.len(), 98);
        for operation in operations {
            assert_eq!(Opcode::try_from(operation.as_str()), Ok(operation));
        }
        assert!(matches!(
            Opcode::try_from("qcore.until"),
            Err(OperationKernelError::UnknownOpcode { .. })
        ));
    }

    #[test]
    fn every_closed_terminator_round_trips() {
        assert_eq!(TerminatorKind::ALL.len(), 6);
        for terminator in TerminatorKind::ALL {
            assert_eq!(
                TerminatorKind::try_from(terminator.as_str()),
                Ok(*terminator)
            );
        }
        assert!(matches!(
            TerminatorKind::try_from("condition"),
            Err(OperationKernelError::UnknownTerminator { .. })
        ));
    }

    #[test]
    fn counts_operations_in_structured_regions() {
        let nested_for = wire::OperationProto {
            opcode: "qcore.for".to_owned(),
            regions: vec![wire::RegionProto {
                blocks: vec![block(vec![operation("classical.add")], "yield")],
            }],
            ..Default::default()
        };
        let nested_if = wire::OperationProto {
            opcode: "qcore.if".to_owned(),
            regions: vec![wire::RegionProto {
                blocks: vec![block(vec![operation("quantum.h"), nested_for], "yield")],
            }],
            ..Default::default()
        };
        let image = package(block(
            vec![operation("qcore.constant"), nested_if],
            "return",
        ));

        let summary = validate_program_operations(&image).expect("valid operation catalog");

        assert_eq!(summary.operation_count(), 5);
    }

    #[test]
    fn rejects_an_unknown_opcode_in_a_nested_region() {
        let structured = wire::OperationProto {
            opcode: "qcore.if".to_owned(),
            regions: vec![wire::RegionProto {
                blocks: vec![block(vec![operation("quantum.clone")], "yield")],
            }],
            ..Default::default()
        };
        let image = package(block(vec![structured], "return"));

        assert_eq!(
            validate_program_operations(&image),
            Err(OperationKernelError::UnknownOpcode {
                opcode: "quantum.clone".to_owned(),
            })
        );
    }

    #[test]
    fn rejects_an_unknown_or_missing_terminator() {
        let unknown = package(block(Vec::new(), "branch"));
        assert_eq!(
            validate_program_operations(&unknown),
            Err(OperationKernelError::UnknownTerminator {
                kind: "branch".to_owned(),
            })
        );

        let missing = package(wire::BlockProto::default());
        assert_eq!(
            validate_program_operations(&missing),
            Err(OperationKernelError::MissingTerminator)
        );
    }

    #[test]
    fn rejects_regions_on_a_non_structured_operation() {
        let leaf = wire::OperationProto {
            opcode: "quantum.h".to_owned(),
            regions: vec![wire::RegionProto {
                blocks: vec![block(Vec::new(), "yield")],
            }],
            ..Default::default()
        };
        let image = package(block(vec![leaf], "return"));

        assert_eq!(
            validate_program_operations(&image),
            Err(OperationKernelError::RegionsOnNonStructuredOperation {
                opcode: Opcode::Quantum(QuantumOpcode::Gate(GateOpcode::H)),
            })
        );
    }

    #[test]
    fn rejects_duplicate_function_local_value_ids() {
        let image = wire::QCorePackageProto {
            functions: vec![wire::FunctionProto {
                symbol: "native_test.main".to_owned(),
                parameters: vec![wire::ParameterProto {
                    name: "input".to_owned(),
                    value: Some(value(1, scalar("Int"))),
                    ..Default::default()
                }],
                result_type: Some(scalar("Unit")),
                body: Some(wire::BlockProto {
                    arguments: vec![value(1, scalar("Int"))],
                    terminator: Some(terminator("unreachable")),
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        };

        assert_eq!(
            validate_typed(&image),
            Err(OperationKernelError::DuplicateValueId { value_id: 1 })
        );
    }

    #[test]
    fn retains_function_local_value_type_roots() {
        let image = typed_package(
            vec![scalar("Int"), scalar("Real")],
            "classical.add",
            scalar("Real"),
            Vec::new(),
        );

        let checked = validate_typed(&image).expect("valid typed operation graph");

        assert_eq!(checked.function_values.len(), 1);
        assert_eq!(checked.function_indices.get("native_test.main"), Some(&0));
        assert_eq!(
            checked.value_types("native_test.main").map(HashMap::len),
            Some(3)
        );
    }

    #[test]
    fn rejects_unresolved_operation_and_terminator_operands() {
        let operation_use = wire::QCorePackageProto {
            functions: vec![wire::FunctionProto {
                symbol: "native_test.main".to_owned(),
                result_type: Some(scalar("Unit")),
                body: Some(wire::BlockProto {
                    operations: vec![wire::OperationProto {
                        opcode: "qcore.forget".to_owned(),
                        operands: vec![99],
                        ..Default::default()
                    }],
                    terminator: Some(terminator("unreachable")),
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        };
        assert_eq!(
            validate_typed(&operation_use),
            Err(OperationKernelError::UndefinedValueId {
                function: "native_test.main".to_owned(),
                value_id: 99,
            })
        );

        let terminator_use = wire::QCorePackageProto {
            functions: vec![wire::FunctionProto {
                symbol: "native_test.main".to_owned(),
                result_type: Some(scalar("Unit")),
                body: Some(wire::BlockProto {
                    terminator: Some(wire::TerminatorProto {
                        kind: "return".to_owned(),
                        operands: vec![99],
                        ..Default::default()
                    }),
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        };
        assert_eq!(
            validate_typed(&terminator_use),
            Err(OperationKernelError::UndefinedValueId {
                function: "native_test.main".to_owned(),
                value_id: 99,
            })
        );
    }

    #[test]
    fn function_local_resolution_deliberately_defers_dominance() {
        let image = wire::QCorePackageProto {
            functions: vec![wire::FunctionProto {
                symbol: "native_test.main".to_owned(),
                result_type: Some(scalar("Unit")),
                body: Some(block(
                    vec![
                        wire::OperationProto {
                            opcode: "classical.add".to_owned(),
                            operands: vec![2, 3],
                            results: vec![value(1, scalar("Int"))],
                            ..Default::default()
                        },
                        wire::OperationProto {
                            opcode: "qcore.constant".to_owned(),
                            results: vec![value(2, scalar("Int"))],
                            ..Default::default()
                        },
                        wire::OperationProto {
                            opcode: "qcore.constant".to_owned(),
                            results: vec![value(3, scalar("Int"))],
                            ..Default::default()
                        },
                    ],
                    "unreachable",
                )),
                ..Default::default()
            }],
            ..Default::default()
        };

        assert!(validate_typed(&image).is_ok());
    }

    #[test]
    fn derives_every_attribute_free_scalar_binary_equation() {
        let cases = [
            ("classical.add", "Int", "Real", "Real"),
            ("classical.sub", "Real", "Int", "Real"),
            ("classical.mul", "Complex", "Real", "Complex"),
            ("classical.truediv", "Int", "Int", "Real"),
            ("classical.floordiv", "Real", "Int", "Real"),
            ("classical.mod", "Int", "Int", "Int"),
            ("classical.pow", "Int", "Complex", "Complex"),
            ("classical.and_bits", "Bool", "Bit", "Bool"),
            ("classical.or_bits", "Bool", "Int", "Int"),
            ("classical.xor", "Int", "Bit", "Int"),
            ("classical.lshift", "Int", "Int", "Int"),
            ("classical.rshift", "Int", "Int", "Int"),
        ];
        for (opcode, left, right, result) in cases {
            let image = typed_package(
                vec![scalar(left), scalar(right)],
                opcode,
                scalar(result),
                Vec::new(),
            );
            assert!(validate_typed(&image).is_ok(), "failed {opcode}");
        }

        let staged_add = typed_package(
            vec![
                staged("Param", scalar("Int")),
                staged("Shot", scalar("Real")),
            ],
            "classical.add",
            staged("Shot", scalar("Real")),
            Vec::new(),
        );
        assert!(validate_typed(&staged_add).is_ok());
    }

    #[test]
    fn observable_algebra_is_const_only_and_compiler_typed() {
        let observable = staged("Const", scalar("Observable"));
        let integer = staged("Const", scalar("Int"));
        let real = staged("Const", scalar("Real"));
        assert!(
            validate_typed(&typed_package(
                vec![integer.clone()],
                "classical.observable_zero",
                observable.clone(),
                Vec::new(),
            ))
            .is_ok()
        );
        assert!(
            validate_typed(&typed_package(
                vec![integer],
                "classical.observable_pauli",
                observable.clone(),
                vec![attribute(
                    "axis",
                    wire::literal_proto::Value::StringValue("X".to_owned()),
                )],
            ))
            .is_ok()
        );
        for (opcode, operands) in [
            (
                "classical.add",
                vec![observable.clone(), observable.clone()],
            ),
            ("classical.sub", vec![observable.clone(), real.clone()]),
            ("classical.mul", vec![real, observable.clone()]),
            ("classical.neg", vec![observable.clone()]),
        ] {
            assert!(
                validate_typed(&typed_package(
                    operands,
                    opcode,
                    observable.clone(),
                    Vec::new(),
                ))
                .is_ok(),
                "failed {opcode}",
            );
        }

        let parametric = typed_package(
            vec![staged("Param", scalar("Real")), observable.clone()],
            "classical.mul",
            observable,
            Vec::new(),
        );
        assert!(matches!(
            validate_typed(&parametric),
            Err(OperationKernelError::MalformedClassicalOperation {
                opcode: ClassicalOpcode::Mul,
            })
        ));
    }

    #[test]
    fn derives_attribute_free_scalar_unary_and_boolean_equations() {
        let cases = [
            ("classical.not", vec!["Bit"], "Bool"),
            ("classical.invert", vec!["Bool"], "Bool"),
            ("classical.neg", vec!["Real"], "Real"),
            ("classical.pos", vec!["Complex"], "Complex"),
            ("classical.ceil", vec!["Real"], "Int"),
            ("classical.log2", vec!["Int"], "Real"),
            ("classical.sqrt", vec!["Int"], "Real"),
            ("classical.and", vec!["Bool", "Bit"], "Bool"),
            ("classical.or", vec!["Bit", "Bit", "Bool"], "Bool"),
        ];
        for (opcode, operands, result) in cases {
            let image = typed_package(
                operands.into_iter().map(scalar).collect(),
                opcode,
                scalar(result),
                Vec::new(),
            );
            assert!(validate_typed(&image).is_ok(), "failed {opcode}");
        }
    }

    #[test]
    fn rejects_forged_and_undefined_scalar_equations() {
        let forged = typed_package(
            vec![scalar("Int"), scalar("Int")],
            "classical.add",
            scalar("Bool"),
            Vec::new(),
        );
        assert_eq!(
            validate_typed(&forged),
            Err(OperationKernelError::MalformedClassicalOperation {
                opcode: ClassicalOpcode::Add,
            })
        );

        let complex_mod = typed_package(
            vec![scalar("Complex"), scalar("Int")],
            "classical.mod",
            scalar("Complex"),
            Vec::new(),
        );
        assert_eq!(
            validate_typed(&complex_mod),
            Err(OperationKernelError::MalformedClassicalOperation {
                opcode: ClassicalOpcode::Mod,
            })
        );
    }

    #[test]
    fn derives_aggregate_and_attribute_driven_classical_equations() {
        let operators = wire::literal_proto::Sequence {
            values: vec![wire::LiteralProto {
                value: Some(wire::literal_proto::Value::StringValue("lt".to_owned())),
            }],
        };
        let cases = vec![
            typed_package(
                vec![scalar("Int"), staged("Param", scalar("Int"))],
                "array",
                staged("Param", array_type(scalar("Int"), &[2])),
                Vec::new(),
            ),
            typed_package(
                vec![
                    staged("Shot", scalar("Bool")),
                    staged("Param", scalar("Int")),
                    scalar("Int"),
                ],
                "classical.select",
                staged("Shot", scalar("Int")),
                Vec::new(),
            ),
            typed_package(
                vec![array_type(scalar("Real"), &[2, 3]), scalar("Int")],
                "classical.index",
                array_type(scalar("Real"), &[3]),
                Vec::new(),
            ),
            typed_package(
                vec![
                    product_type(vec![scalar("Int"), scalar("Int")]),
                    scalar("Int"),
                ],
                "classical.index",
                scalar("Int"),
                Vec::new(),
            ),
            typed_package(
                vec![staged(
                    "Param",
                    product_type(vec![scalar("Int"), scalar("Real")]),
                )],
                "classical.project",
                staged("Param", scalar("Real")),
                vec![attribute(
                    "position",
                    wire::literal_proto::Value::IntValue(-1),
                )],
            ),
            typed_package(
                vec![staged("Param", scalar("Int"))],
                "classical.cast",
                staged("Param", scalar("Real")),
                vec![attribute(
                    "target",
                    wire::literal_proto::Value::StringValue("Real".to_owned()),
                )],
            ),
            typed_package(
                vec![staged("Shot", word_type("Bits", 4))],
                "classical.len",
                staged("Shot", scalar("Int")),
                Vec::new(),
            ),
            typed_package(
                vec![staged("Param", scalar("Int")), scalar("Real")],
                "classical.compare_chain",
                staged("Param", scalar("Bool")),
                vec![attribute(
                    "operators",
                    wire::literal_proto::Value::TupleValue(operators),
                )],
            ),
        ];

        for image in cases {
            assert!(validate_typed(&image).is_ok());
        }
    }

    #[test]
    fn shape_only_length_is_const_independently_of_element_availability() {
        let mut sliced = array_type(scalar("Real"), &[2]);
        sliced.dimensions[0].value = Some(wire::dimension_proto::Value::Symbol(
            "slice(2,start,~,1)".to_owned(),
        ));
        let collections = [
            staged("Param", array_type(scalar("Real"), &[2, 3])),
            staged("Param", sliced),
            staged("Shot", word_type("Bits", 4)),
            product_type(vec![staged("Param", scalar("Int")), scalar("Int")]),
        ];
        for collection in collections {
            let image = typed_package(
                vec![collection],
                "classical.len",
                staged("Const", scalar("Int")),
                vec![attribute(
                    "shape_only",
                    wire::literal_proto::Value::BoolValue(true),
                )],
            );
            assert!(validate_typed(&image).is_ok());
        }
    }

    #[test]
    fn length_modes_preserve_exact_result_stages() {
        for (attributes, result_type) in [
            (Vec::new(), staged("Const", scalar("Int"))),
            (
                vec![attribute(
                    "shape_only",
                    wire::literal_proto::Value::BoolValue(true),
                )],
                staged("Param", scalar("Int")),
            ),
        ] {
            let image = typed_package(
                vec![staged("Param", array_type(scalar("Real"), &[2]))],
                "classical.len",
                result_type,
                attributes,
            );
            assert_eq!(
                validate_typed(&image),
                Err(OperationKernelError::MalformedClassicalOperation {
                    opcode: ClassicalOpcode::Len,
                }),
            );
        }
    }

    #[test]
    fn rejects_noncanonical_shape_only_length_attributes() {
        let canonical = attribute("shape_only", wire::literal_proto::Value::BoolValue(true));
        for attributes in [
            vec![attribute(
                "shape_only",
                wire::literal_proto::Value::BoolValue(false),
            )],
            vec![attribute(
                "shape_only",
                wire::literal_proto::Value::IntValue(1),
            )],
            vec![attribute(
                "other",
                wire::literal_proto::Value::BoolValue(true),
            )],
            vec![canonical.clone(), canonical],
        ] {
            let image = typed_package(
                vec![staged("Param", array_type(scalar("Real"), &[2]))],
                "classical.len",
                staged("Const", scalar("Int")),
                attributes,
            );
            assert_eq!(
                validate_typed(&image),
                Err(OperationKernelError::MalformedClassicalOperation {
                    opcode: ClassicalOpcode::Len,
                }),
            );
        }
    }

    #[test]
    fn local_peer_equations_require_rigid_dimensions_and_nested_stages() {
        let mismatched_dimensions = typed_package(
            vec![
                scalar("Bool"),
                symbolic_word_type("Bits", "n"),
                symbolic_word_type("Bits", "m"),
            ],
            "classical.select",
            symbolic_word_type("Bits", "n"),
            Vec::new(),
        );
        assert_eq!(
            validate_typed(&mismatched_dimensions),
            Err(OperationKernelError::MalformedClassicalOperation {
                opcode: ClassicalOpcode::Select,
            })
        );

        let mismatched_nested_stages = typed_package(
            vec![
                scalar("Bool"),
                product_type(vec![staged("Param", scalar("Int"))]),
                product_type(vec![staged("Shot", scalar("Int"))]),
            ],
            "classical.select",
            product_type(vec![staged("Param", scalar("Int"))]),
            Vec::new(),
        );
        assert_eq!(
            validate_typed(&mismatched_nested_stages),
            Err(OperationKernelError::MalformedClassicalOperation {
                opcode: ClassicalOpcode::Select,
            })
        );
    }

    #[test]
    fn local_peer_equations_join_only_outer_stages() {
        let same_dimension = typed_package(
            vec![
                scalar("Bool"),
                staged("Param", symbolic_word_type("Bits", "n")),
                staged("Shot", symbolic_word_type("Bits", "n")),
            ],
            "classical.select",
            staged("Shot", symbolic_word_type("Bits", "n")),
            Vec::new(),
        );

        assert!(validate_typed(&same_dimension).is_ok());
    }

    #[test]
    fn aggregate_comparison_joins_nested_availability() {
        let operators = wire::literal_proto::Sequence {
            values: vec![wire::LiteralProto {
                value: Some(wire::literal_proto::Value::StringValue("eq".to_owned())),
            }],
        };
        let nested = product_type(vec![
            staged("Param", word_type("UInt", 3)),
            product_type(vec![staged("Shot", scalar("Bool"))]),
        ]);
        let image = typed_package(
            vec![nested.clone(), nested],
            "classical.compare_chain",
            staged("Shot", scalar("Bool")),
            vec![attribute(
                "operators",
                wire::literal_proto::Value::TupleValue(operators),
            )],
        );

        assert!(validate_typed(&image).is_ok());
    }

    #[test]
    fn classifies_every_primitive_gate_into_logical_families() {
        for gate in GateOpcode::ALL {
            let spec = gate.spec();
            let families = logical_gate_families(gate.as_str());
            assert!(
                families.contains(&"gates"),
                "missing gate family for {gate}"
            );
            assert!(
                families.iter().any(|family| matches!(
                    *family,
                    "single_qubit_gates" | "two_qubit_gates" | "multi_qubit_gates"
                )),
                "missing arity family for {gate}"
            );
            assert_eq!(
                (0..spec.selectors().len()).collect::<Vec<_>>(),
                (0..spec.selectors().len())
                    .map(|index| spec.selector_default(index).expect("declared selector"))
                    .collect::<Vec<_>>(),
                "invalid selector defaults for {gate}"
            );
            assert_eq!(spec.selector_default(spec.selectors().len()), None);
        }
        assert!(logical_gate_families("quantum.measure").is_empty());
    }

    #[test]
    fn primitive_adjoint_catalog_is_closed_and_involutive() {
        for gate in GateOpcode::ALL {
            assert_eq!(GateOpcode::parse(gate.as_str()), Some(*gate));
            assert_eq!(GateOpcode::parse_name(gate.name()), Some(*gate));
            assert_eq!(gate.adjoint().adjoint(), *gate);
            assert_eq!(gate.has_angle(), gate.spec().has_angle());
        }
    }

    #[test]
    fn exact_adjoint_catalog_is_closed_and_involutive() {
        for operation in QuantumOpcode::all() {
            assert_eq!(QuantumOpcode::parse(operation.as_str()), Some(operation));
            assert_eq!(QuantumOpcode::parse_name(operation.name()), Some(operation));
            if let Some(adjoint) = operation.exact_adjoint() {
                assert_eq!(adjoint.exact_adjoint(), Some(operation));
            }
        }
        for operation in QCoreOpcode::ALL {
            if let Some(adjoint) = operation.exact_adjoint() {
                assert_eq!(adjoint.exact_adjoint(), Some(*operation));
            }
        }
    }

    #[test]
    fn source_type_inference_reuses_the_closed_local_equations() {
        let operands = vec![
            staged("Const", scalar("Int")).encode_to_vec(),
            staged("Param", scalar("Real")).encode_to_vec(),
        ];
        let result =
            infer_classical_type("classical.add", &operands, &[]).expect("valid promoted addition");
        assert_eq!(
            wire::TypeProto::decode(result.as_slice()).expect("result type"),
            staged("Param", scalar("Real"))
        );

        let array_operands = vec![
            staged("Const", scalar("Int")).encode_to_vec(),
            staged("Param", scalar("Int")).encode_to_vec(),
        ];
        let result = infer_classical_type("array", &array_operands, &[])
            .expect("valid staged array construction");
        assert_eq!(
            wire::TypeProto::decode(result.as_slice()).expect("array result type"),
            staged("Param", array_type(scalar("Int"), &[2]))
        );

        assert!(matches!(
            infer_classical_type("classical.add", &[scalar("Bool").encode_to_vec()], &[]),
            Err(ClassicalTypeInferenceError::MalformedOperation { opcode, .. })
                if opcode == "classical.add"
        ));
        assert_eq!(
            infer_classical_type("quantum.h", &[], &[]),
            Err(ClassicalTypeInferenceError::UnsupportedOpcode(
                "quantum.h".to_owned()
            ))
        );

        let mut noncanonical = scalar("Int").encode_to_vec();
        noncanonical.extend_from_slice(&[0xf8, 0x07, 0x01]);
        assert!(matches!(
            infer_classical_type("classical.pos", &[noncanonical], &[]),
            Err(ClassicalTypeInferenceError::InvalidInput(reason))
                if reason == "type payload is not canonical"
        ));
        assert!(matches!(
            infer_classical_type("classical.pos", &[vec![0x1a, 0x01, 0xff]], &[]),
            Err(ClassicalTypeInferenceError::InvalidInput(_))
        ));
    }
}
