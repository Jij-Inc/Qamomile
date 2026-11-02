//! Exact bounded reversible synthesis for first-class unsigned arithmetic.

use crate::lowering::arithmetic::{
    ArithmeticPlan, ArithmeticStep, MAX_ARITHMETIC_SELECTORS, MAX_ARITHMETIC_STEPS,
};
use crate::lowering::realization::{Realization, RealizationError};
use crate::number_theory::CheckedModulus;

/// Maximum total coherent width accepted by the generic exact realization.
pub const MAX_PERMUTATION_WIDTH: u64 = 9;

/// One semantic operation accepted by exact arithmetic permutation planning.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ArithmeticOperation {
    /// Add a compile-time integer modulo the register width.
    AddConstant,
    /// Subtract a compile-time integer modulo the register width.
    SubtractConstant,
    /// Add one retained register modulo a retained modulus register.
    ModAdd,
    /// Subtract one retained register modulo a retained modulus register.
    ModSub,
    /// Add a compile-time integer modulo a compile-time modulus.
    ModAddConstant,
    /// Subtract a compile-time integer modulo a compile-time modulus.
    ModSubConstant,
    /// Multiply retained factors and add into a double-width accumulator.
    MulAdd,
    /// Multiply retained factors and subtract from a double-width accumulator.
    MulSub,
    /// Multiply by a compile-time unit modulo a compile-time modulus.
    ModMul,
    /// Divide by a compile-time unit modulo a compile-time modulus.
    ModDiv,
    /// Toggle a result bit when the left retained register is smaller.
    LessThanXor,
}

impl ArithmeticOperation {
    /// Complete stable semantic operation set accepted by this planner.
    pub const ALL: [Self; 11] = [
        Self::AddConstant,
        Self::SubtractConstant,
        Self::ModAdd,
        Self::ModSub,
        Self::ModAddConstant,
        Self::ModSubConstant,
        Self::MulAdd,
        Self::MulSub,
        Self::ModMul,
        Self::ModDiv,
        Self::LessThanXor,
    ];

    /// Return the canonical `QCore` opcode spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AddConstant => "quantum.add_constant",
            Self::SubtractConstant => "quantum.subtract_constant",
            Self::ModAdd => "quantum.mod_add",
            Self::ModSub => "quantum.mod_sub",
            Self::ModAddConstant => "quantum.mod_add_constant",
            Self::ModSubConstant => "quantum.mod_sub_constant",
            Self::MulAdd => "quantum.mul_add",
            Self::MulSub => "quantum.mul_sub",
            Self::ModMul => "quantum.mod_mul",
            Self::ModDiv => "quantum.mod_div",
            Self::LessThanXor => "quantum.less_than_xor",
        }
    }

    const fn adjoint(self) -> Self {
        match self {
            Self::AddConstant => Self::SubtractConstant,
            Self::SubtractConstant => Self::AddConstant,
            Self::ModAdd => Self::ModSub,
            Self::ModSub => Self::ModAdd,
            Self::ModAddConstant => Self::ModSubConstant,
            Self::ModSubConstant => Self::ModAddConstant,
            Self::MulAdd => Self::MulSub,
            Self::MulSub => Self::MulAdd,
            Self::ModMul => Self::ModDiv,
            Self::ModDiv => Self::ModMul,
            Self::LessThanXor => Self::LessThanXor,
        }
    }
}

impl TryFrom<&str> for ArithmeticOperation {
    type Error = RealizationError;

    fn try_from(opcode: &str) -> Result<Self, Self::Error> {
        let operation = match opcode {
            "quantum.add_constant" => Self::AddConstant,
            "quantum.subtract_constant" => Self::SubtractConstant,
            "quantum.mod_add" => Self::ModAdd,
            "quantum.mod_sub" => Self::ModSub,
            "quantum.mod_add_constant" => Self::ModAddConstant,
            "quantum.mod_sub_constant" => Self::ModSubConstant,
            "quantum.mul_add" => Self::MulAdd,
            "quantum.mul_sub" => Self::MulSub,
            "quantum.mod_mul" => Self::ModMul,
            "quantum.mod_div" => Self::ModDiv,
            "quantum.less_than_xor" => Self::LessThanXor,
            _ => return Err(RealizationError::ArithmeticParameters),
        };
        Ok(operation)
    }
}

/// Validated arithmetic inputs whose gate schedule is materialized on demand.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct CheckedArithmeticPermutation {
    operation: ArithmeticOperation,
    width: u64,
    constants: Box<[i64]>,
}

impl CheckedArithmeticPermutation {
    /// Check dimensions, constants, and conservative schedule bounds without gates.
    ///
    /// # Errors
    /// Rejects the same invalid domains and bounded generation limits as planning.
    pub fn new(
        operation: ArithmeticOperation,
        width: u64,
        constants: &[i64],
    ) -> Result<Self, RealizationError> {
        if !matches!(
            operation,
            ArithmeticOperation::AddConstant
                | ArithmeticOperation::SubtractConstant
                | ArithmeticOperation::MulAdd
                | ArithmeticOperation::MulSub
        ) && !(1..=MAX_PERMUTATION_WIDTH).contains(&width)
        {
            return Err(RealizationError::ArithmeticWidth);
        }
        let layout = operation.layout(width, constants)?;
        let n = layout.n as u128;
        let (steps, selectors) = match operation {
            ArithmeticOperation::AddConstant | ArithmeticOperation::SubtractConstant => {
                if width > 1024 {
                    return Err(RealizationError::ScheduleBudget);
                }
                let magnitude = i128::from(constants[0]).unsigned_abs();
                let mut counts = (0_u128, 0_u128);
                for bit in 0..width.min(64) {
                    if magnitude & (1_u128 << bit) != 0 {
                        let span = u128::from(width - bit);
                        counts.0 += span;
                        counts.1 += span * (span + 1) / 2;
                    }
                }
                counts
            }
            ArithmeticOperation::MulAdd | ArithmeticOperation::MulSub => {
                let steps = n
                    .checked_mul(n)
                    .and_then(|v| v.checked_mul(n + 1))
                    .filter(|v| *v <= u128::from(MAX_ARITHMETIC_STEPS))
                    .ok_or(RealizationError::ScheduleBudget)?;
                let mut selectors = 0_u128;
                for left in 0..n {
                    for right in 0..n {
                        let span = 2 * n - left - right;
                        selectors += span * (span + 5) / 2;
                    }
                }
                (steps, selectors)
            }
            _ => {
                if !(1..=MAX_PERMUTATION_WIDTH).contains(&width) {
                    return Err(RealizationError::ArithmeticWidth);
                }
                // At most 2^w - 1 transpositions, each crossing <= w edges.
                let edges = ((1_u128 << width) - 1) * (2 * u128::from(width) - 1);
                (
                    edges * (2 * u128::from(width) - 1),
                    edges * (3 * u128::from(width) - 2),
                )
            }
        };
        if steps > u128::from(MAX_ARITHMETIC_STEPS)
            || selectors > u128::from(MAX_ARITHMETIC_SELECTORS)
        {
            // The generic upper bound can be loose; preserve accepted inputs.
            if width <= MAX_PERMUTATION_WIDTH {
                plan_arithmetic_permutation(operation, width, constants, false)?;
            } else {
                return Err(RealizationError::ScheduleBudget);
            }
        }
        Ok(Self {
            operation,
            width,
            constants: constants.into(),
        })
    }

    /// Generate the checked operation through the single existing implementation.
    ///
    /// # Errors
    /// Returns a realization error if a generation invariant is violated.
    pub fn materialize(&self) -> Result<ArithmeticPlan, RealizationError> {
        plan_arithmetic_permutation(self.operation, self.width, &self.constants, false)
    }
}

/// Plan one complete-basis arithmetic permutation without ancillary qubits.
///
/// The result contains only reversible X operations. Every inactive modulus
/// domain state is fixed, so coherent control and adjoint preserve the full
/// unitary, including phase. Width and output budgets are checked before any
/// unbounded allocation.
///
/// # Errors
/// Rejects invalid widths, modulus/factor parameters, and output budgets.
pub fn plan_arithmetic_permutation(
    operation: ArithmeticOperation,
    width: u64,
    constants: &[i64],
    adjoint: bool,
) -> Result<ArithmeticPlan, RealizationError> {
    let mut kind = operation;
    if adjoint {
        kind = kind.adjoint();
    }
    if !matches!(
        kind,
        ArithmeticOperation::MulAdd
            | ArithmeticOperation::MulSub
            | ArithmeticOperation::AddConstant
            | ArithmeticOperation::SubtractConstant
    ) && !(1..=MAX_PERMUTATION_WIDTH).contains(&width)
    {
        return Err(RealizationError::ArithmeticWidth);
    }
    let layout = kind.layout(width, constants)?;
    if matches!(
        kind,
        ArithmeticOperation::AddConstant | ArithmeticOperation::SubtractConstant
    ) {
        return plan_constant_addition(
            layout.n,
            constants[0],
            matches!(kind, ArithmeticOperation::SubtractConstant),
        );
    }
    if matches!(
        kind,
        ArithmeticOperation::MulAdd | ArithmeticOperation::MulSub
    ) {
        return plan_multiply_accumulate(layout.n, matches!(kind, ArithmeticOperation::MulSub));
    }
    let states = 1_usize
        .checked_shl(u32::try_from(width).map_err(|_| RealizationError::ArithmeticWidth)?)
        .ok_or(RealizationError::ArithmeticWidth)?;
    let map = (0..states)
        .map(|state| layout.map(state))
        .collect::<Vec<_>>();
    let mut images = vec![false; states];
    for &image in &map {
        let occupied = images
            .get_mut(image)
            .ok_or(RealizationError::ArithmeticParameters)?;
        if *occupied {
            return Err(RealizationError::ArithmeticParameters);
        }
        *occupied = true;
    }
    let mut visited = vec![false; states];
    let mut steps = Vec::new();
    let mut selectors = 0_u64;
    for start in 0..states {
        if visited[start] {
            continue;
        }
        let mut cycle = Vec::new();
        let mut current = start;
        while !visited[current] {
            visited[current] = true;
            cycle.push(current);
            current = map[current];
        }
        if current != start {
            return Err(RealizationError::ArithmeticParameters);
        }
        for &next in cycle.iter().skip(1) {
            append_transposition(start, next, width, &mut steps, &mut selectors)?;
        }
    }
    Ok(ArithmeticPlan::new(Realization::ArithmeticPortable, steps))
}

fn plan_constant_addition(
    width: usize,
    value: i64,
    inverse: bool,
) -> Result<ArithmeticPlan, RealizationError> {
    if width == 0 || width > 1_024 {
        return Err(RealizationError::ScheduleBudget);
    }
    let magnitude = i128::from(value).unsigned_abs();
    let mut count = 0_u128;
    for bit in 0..width.min(64) {
        if magnitude & (1_u128 << bit) != 0 {
            count += (width - bit) as u128;
        }
    }
    if count > u128::from(MAX_ARITHMETIC_STEPS) {
        return Err(RealizationError::ScheduleBudget);
    }
    let mut steps =
        Vec::with_capacity(usize::try_from(count).map_err(|_| RealizationError::ScheduleBudget)?);
    let mut selectors = 0_u64;
    for bit in 0..width.min(64) {
        if magnitude & (1_u128 << bit) == 0 {
            continue;
        }
        for target in (bit..width).rev() {
            let controls = (bit..target).map(|index| index as u64).collect::<Vec<_>>();
            push_x(target as u64, &controls, &mut steps, &mut selectors)?;
        }
    }
    if inverse ^ value.is_negative() {
        steps.reverse();
    }
    Ok(ArithmeticPlan::new(Realization::ArithmeticPortable, steps))
}

fn plan_multiply_accumulate(
    factor_width: usize,
    inverse: bool,
) -> Result<ArithmeticPlan, RealizationError> {
    let n = factor_width as u128;
    let count = n
        .checked_mul(n)
        .and_then(|square| square.checked_mul(n + 1))
        .filter(|count| *count <= u128::from(MAX_ARITHMETIC_STEPS))
        .ok_or(RealizationError::ScheduleBudget)?;
    let mut steps =
        Vec::with_capacity(usize::try_from(count).map_err(|_| RealizationError::ScheduleBudget)?);
    let mut selectors = 0_u64;
    let product_width = 2 * factor_width;
    for left in 0..factor_width {
        for right in 0..factor_width {
            let offset = left + right;
            for target in (offset..product_width).rev() {
                let mut controls = vec![
                    (product_width + left) as u64,
                    (product_width + factor_width + right) as u64,
                ];
                controls.extend((offset..target).map(|bit| bit as u64));
                push_x(target as u64, &controls, &mut steps, &mut selectors)?;
            }
        }
    }
    if inverse {
        steps.reverse();
    }
    Ok(ArithmeticPlan::new(Realization::ArithmeticPortable, steps))
}

impl ArithmeticOperation {
    fn layout(self, width: u64, constants: &[i64]) -> Result<Layout, RealizationError> {
        let valid = match self {
            Self::AddConstant | Self::SubtractConstant => constants.len() == 1,
            Self::ModAddConstant | Self::ModSubConstant | Self::ModMul | Self::ModDiv => {
                constants.len() == 2
            }
            _ => constants.is_empty(),
        };
        if !valid {
            return Err(RealizationError::ArithmeticParameters);
        }
        let n = match self {
            Self::ModAdd | Self::ModSub if width.is_multiple_of(3) => width / 3,
            Self::MulAdd | Self::MulSub if width.is_multiple_of(4) => width / 4,
            Self::LessThanXor if width >= 3 && width % 2 == 1 => (width - 1) / 2,
            Self::AddConstant
            | Self::SubtractConstant
            | Self::ModAddConstant
            | Self::ModSubConstant
            | Self::ModMul
            | Self::ModDiv => width,
            _ => return Err(RealizationError::ArithmeticWidth),
        };
        if n == 0 {
            return Err(RealizationError::ArithmeticWidth);
        }
        let mut parameters = [0_i128; 2];
        for (slot, value) in parameters.iter_mut().zip(constants) {
            *slot = i128::from(*value);
        }
        if matches!(
            self,
            Self::ModAddConstant | Self::ModSubConstant | Self::ModMul | Self::ModDiv
        ) {
            let domain = CheckedModulus::new(n, constants[1])
                .ok_or(RealizationError::ArithmeticParameters)?;
            parameters[0] = i128::from(domain.normalize(constants[0]));
            if matches!(self, Self::ModMul | Self::ModDiv) {
                let unit = domain
                    .unit(constants[0])
                    .ok_or(RealizationError::ArithmeticParameters)?;
                parameters[0] = i128::from(
                    if matches!(self, Self::ModDiv) {
                        unit.inverse()
                    } else {
                        unit
                    }
                    .residue(),
                );
            }
        }
        Ok(Layout {
            kind: self,
            n: usize::try_from(n).map_err(|_| RealizationError::ArithmeticWidth)?,
            parameters,
        })
    }
}

struct Layout {
    kind: ArithmeticOperation,
    n: usize,
    parameters: [i128; 2],
}

impl Layout {
    fn map(&self, state: usize) -> usize {
        let n = self.n;
        let mask = (1_usize << n) - 1;
        match self.kind {
            ArithmeticOperation::AddConstant | ArithmeticOperation::SubtractConstant => {
                let sign = if matches!(self.kind, ArithmeticOperation::AddConstant) {
                    1
                } else {
                    -1
                };
                (i128::try_from(state).unwrap() + sign * self.parameters[0]).rem_euclid(1_i128 << n)
                    as usize
            }
            ArithmeticOperation::ModAdd | ArithmeticOperation::ModSub => {
                let target = state & mask;
                let addend = (state >> n) & mask;
                let modulus = (state >> (2 * n)) & mask;
                if modulus == 0 || target >= modulus || addend >= modulus {
                    return state;
                }
                let updated = if matches!(self.kind, ArithmeticOperation::ModAdd) {
                    (target + addend) % modulus
                } else {
                    (target + modulus - addend) % modulus
                };
                (state & !mask) | updated
            }
            ArithmeticOperation::ModAddConstant | ArithmeticOperation::ModSubConstant => {
                let modulus = usize::try_from(self.parameters[1]).expect("bounded modulus");
                if state >= modulus {
                    return state;
                }
                let addend = usize::try_from(self.parameters[0]).expect("normalized addend");
                if matches!(self.kind, ArithmeticOperation::ModAddConstant) {
                    (state + addend) % modulus
                } else {
                    (state + modulus - addend) % modulus
                }
            }
            ArithmeticOperation::MulAdd | ArithmeticOperation::MulSub => {
                let product_width = 2 * n;
                let product_mask = (1_usize << product_width) - 1;
                let product = state & product_mask;
                let left = (state >> product_width) & mask;
                let right = (state >> (product_width + n)) & mask;
                let term = left * right;
                let updated = if matches!(self.kind, ArithmeticOperation::MulAdd) {
                    product.wrapping_add(term)
                } else {
                    product.wrapping_sub(term)
                } & product_mask;
                (state & !product_mask) | updated
            }
            ArithmeticOperation::ModMul | ArithmeticOperation::ModDiv => {
                let modulus = usize::try_from(self.parameters[1]).expect("bounded modulus");
                if state >= modulus {
                    state
                } else {
                    state * usize::try_from(self.parameters[0]).expect("normalized factor")
                        % modulus
                }
            }
            ArithmeticOperation::LessThanXor => {
                let left = (state >> 1) & mask;
                let right = (state >> (n + 1)) & mask;
                state ^ usize::from(left < right)
            }
        }
    }
}

fn append_transposition(
    left: usize,
    right: usize,
    width: u64,
    steps: &mut Vec<ArithmeticStep>,
    selectors: &mut u64,
) -> Result<(), RealizationError> {
    let mut path = vec![left];
    let mut current = left;
    for bit in 0..width {
        let mask = 1_usize << bit;
        if (left ^ right) & mask != 0 {
            current ^= mask;
            path.push(current);
        }
    }
    for pair in path.windows(2) {
        append_adjacent_swap(pair[0], pair[1], width, steps, selectors)?;
    }
    for pair in path.windows(2).rev().skip(1) {
        append_adjacent_swap(pair[0], pair[1], width, steps, selectors)?;
    }
    Ok(())
}

fn append_adjacent_swap(
    left: usize,
    right: usize,
    width: u64,
    steps: &mut Vec<ArithmeticStep>,
    selectors: &mut u64,
) -> Result<(), RealizationError> {
    let target = u64::from((left ^ right).trailing_zeros());
    let mut negative = Vec::new();
    let mut positive = Vec::new();
    for bit in 0..width {
        if bit != target {
            if left & (1_usize << bit) == 0 {
                negative.push(bit);
            } else {
                positive.push(bit);
            }
        }
    }
    for &bit in &negative {
        push_x(bit, &[], steps, selectors)?;
    }
    positive.extend(negative.iter().copied());
    push_x(target, &positive, steps, selectors)?;
    for &bit in negative.iter().rev() {
        push_x(bit, &[], steps, selectors)?;
    }
    Ok(())
}

fn push_x(
    target: u64,
    controls: &[u64],
    steps: &mut Vec<ArithmeticStep>,
    selectors: &mut u64,
) -> Result<(), RealizationError> {
    *selectors = selectors
        .checked_add(1 + controls.len() as u64)
        .filter(|value| *value <= MAX_ARITHMETIC_SELECTORS)
        .ok_or(RealizationError::ScheduleBudget)?;
    if steps.len() as u64 >= MAX_ARITHMETIC_STEPS {
        return Err(RealizationError::ScheduleBudget);
    }
    steps.push(ArithmeticStep::pauli_x(target, controls.to_vec()));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apply(plan: &ArithmeticPlan, mut state: usize) -> usize {
        for step in plan.steps() {
            if step.controls().iter().all(|bit| state & (1 << bit) != 0) {
                state ^= 1 << step.indices()[0];
            }
        }
        state
    }

    #[test]
    fn closed_operations_round_trip_at_the_wire_boundary() {
        for operation in ArithmeticOperation::ALL {
            assert_eq!(
                ArithmeticOperation::try_from(operation.as_str()),
                Ok(operation)
            );
        }
        assert_eq!(
            ArithmeticOperation::try_from("quantum.unknown_arithmetic"),
            Err(RealizationError::ArithmeticParameters)
        );
    }

    #[test]
    fn complete_basis_matches_each_semantic_arithmetic_map_and_adjoint() {
        for (opcode, width, constants) in [
            ("quantum.add_constant", 3, vec![-2]),
            ("quantum.mod_add", 6, vec![]),
            ("quantum.mod_add_constant", 3, vec![2, 5]),
            ("quantum.mul_add", 8, vec![]),
            ("quantum.mod_mul", 3, vec![2, 5]),
            ("quantum.less_than_xor", 5, vec![]),
        ] {
            let operation = ArithmeticOperation::try_from(opcode).unwrap();
            let layout = operation.layout(width, &constants).unwrap();
            let plan = plan_arithmetic_permutation(operation, width, &constants, false).unwrap();
            let inverse = plan_arithmetic_permutation(operation, width, &constants, true).unwrap();
            for state in 0..(1 << width) {
                let observed = apply(&plan, state);
                assert_eq!(observed, layout.map(state), "{opcode} on {state}");
                assert_eq!(apply(&inverse, observed), state, "inverse of {opcode}");
            }
        }
    }

    #[test]
    fn invalid_parameters_and_width_fail_before_synthesis() {
        assert_eq!(
            plan_arithmetic_permutation(ArithmeticOperation::ModMul, 3, &[2, 6], false),
            Err(RealizationError::ArithmeticParameters)
        );
        assert_eq!(
            plan_arithmetic_permutation(ArithmeticOperation::ModAdd, 12, &[], false),
            Err(RealizationError::ArithmeticWidth)
        );
        assert_eq!(
            plan_arithmetic_permutation(ArithmeticOperation::ModAddConstant, 2, &[1, 5], false,),
            Err(RealizationError::ArithmeticParameters)
        );
        assert_eq!(
            plan_arithmetic_permutation(ArithmeticOperation::MulAdd, 12, &[], false)
                .unwrap()
                .steps()
                .len(),
            36
        );
        assert_eq!(
            plan_arithmetic_permutation(ArithmeticOperation::AddConstant, 12, &[5], false)
                .unwrap()
                .steps()
                .len(),
            22
        );
        assert_eq!(
            plan_arithmetic_permutation(ArithmeticOperation::MulAdd, 400, &[], false),
            Err(RealizationError::ScheduleBudget)
        );
    }
}
