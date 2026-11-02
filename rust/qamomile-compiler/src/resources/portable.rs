//! Small exact, ancilla-free portable gate recipes.
//!
//! The ordered recipe is the authority for both symbolic counting and bounded
//! materialization. Angles are affine in one input angle, including its sign
//! under adjoint. Arbitrary rotations are deliberately not priced as T gates.

use crate::operations::primitive_gate_spec;

use super::symbolic_resources::GateBasis;

/// One gate in a local recipe, with controls preceding original target lanes.
#[derive(Clone, Debug)]
pub struct PortableGateStep {
    pub(crate) opcode: &'static str,
    pub(crate) lanes: Vec<u8>,
    pub(crate) angle: RecipeAngle,
}

/// Exact angle description, avoiding floating point in resource derivations.
#[derive(Clone, Copy, Debug)]
pub enum RecipeAngle {
    /// No angle operand.
    None,
    /// The input angle multiplied by this rational.
    Input(i32, u32),
    /// Pi multiplied by this rational.
    Pi(i32, u32),
}

impl PortableGateStep {
    /// Return the semantic primitive opcode.
    #[must_use]
    pub const fn opcode(&self) -> &'static str {
        self.opcode
    }

    /// Return the ordered local lane selectors.
    #[must_use]
    pub fn lanes(&self) -> &[u8] {
        &self.lanes
    }

    /// Return the exact symbolic angle.
    #[must_use]
    pub const fn angle(&self) -> RecipeAngle {
        self.angle
    }
}

fn step(opcode: &'static str, lanes: &[u8], angle: RecipeAngle) -> PortableGateStep {
    PortableGateStep {
        opcode,
        lanes: lanes.to_vec(),
        angle,
    }
}

/// Construct an exact phase-preserving recipe, or leave the operation unresolved.
#[must_use]
pub fn portable_gate_recipe(
    opcode: &str,
    controls: u32,
    basis: GateBasis,
) -> Option<Vec<PortableGateStep>> {
    use RecipeAngle::{Input, None as NoAngle, Pi};
    let plain = |name, lanes: &[u8]| step(name, lanes, NoAngle);
    let toffoli = || {
        if basis == GateBasis::CliffordToffoliRotations {
            vec![plain("quantum.ccx", &[0, 1, 2])]
        } else {
            vec![
                plain("quantum.h", &[2]),
                plain("quantum.cx", &[1, 2]),
                plain("quantum.tdg", &[2]),
                plain("quantum.cx", &[0, 2]),
                plain("quantum.t", &[2]),
                plain("quantum.cx", &[1, 2]),
                plain("quantum.tdg", &[2]),
                plain("quantum.cx", &[0, 2]),
                plain("quantum.t", &[1]),
                plain("quantum.t", &[2]),
                plain("quantum.h", &[2]),
                plain("quantum.cx", &[0, 1]),
                plain("quantum.t", &[0]),
                plain("quantum.tdg", &[1]),
                plain("quantum.cx", &[0, 1]),
            ]
        }
    };
    Some(match (opcode, controls) {
        ("quantum.x", 2) | ("quantum.cx", 1) | ("quantum.ccx", 0) => toffoli(),
        ("quantum.x", 1) => vec![plain("quantum.cx", &[0, 1])],
        ("quantum.z", 1) => vec![plain("quantum.cz", &[0, 1])],
        ("quantum.y", 1) => vec![
            plain("quantum.sdg", &[1]),
            plain("quantum.cx", &[0, 1]),
            plain("quantum.s", &[1]),
        ],
        ("quantum.h", 1) => vec![
            step("quantum.ry", &[1], Pi(-1, 4)),
            plain("quantum.cz", &[0, 1]),
            step("quantum.ry", &[1], Pi(1, 4)),
        ],
        ("quantum.ry" | "quantum.rz", 1) => {
            let rotation = if opcode == "quantum.ry" {
                "quantum.ry"
            } else {
                "quantum.rz"
            };
            vec![
                step(rotation, &[1], Input(1, 2)),
                plain("quantum.cx", &[0, 1]),
                step(rotation, &[1], Input(-1, 2)),
                plain("quantum.cx", &[0, 1]),
            ]
        }
        ("quantum.p", 1) | ("quantum.cp", 0) => vec![
            step("quantum.p", &[0], Input(1, 2)),
            step("quantum.p", &[1], Input(1, 2)),
            plain("quantum.cx", &[0, 1]),
            step("quantum.p", &[1], Input(-1, 2)),
            plain("quantum.cx", &[0, 1]),
        ],
        ("quantum.swap", 0) => vec![
            plain("quantum.cx", &[0, 1]),
            plain("quantum.cx", &[1, 0]),
            plain("quantum.cx", &[0, 1]),
        ],
        ("quantum.rzz", 0) => vec![
            plain("quantum.cx", &[0, 1]),
            step("quantum.rz", &[1], Input(1, 1)),
            plain("quantum.cx", &[0, 1]),
        ],
        (_, 0) => {
            let spec = primitive_gate_spec(opcode)?;
            // CP, SWAP, RZZ and CCX were handled above. Every remaining
            // primitive belongs to the explicitly declared alphabet.
            let lanes = (0..spec.selectors().len())
                .map(u8::try_from)
                .collect::<Result<Vec<_>, _>>()
                .ok()?;
            vec![step(
                spec.opcode(),
                &lanes,
                if spec.has_angle() {
                    Input(1, 1)
                } else {
                    NoAngle
                },
            )]
        }
        _ => return None,
    })
}
