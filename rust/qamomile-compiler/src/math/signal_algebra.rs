//! Checked projected-map layouts and conservative reconstruction arithmetic.

use super::binary64::{SUBNORMAL_EXPONENT, float_to_subnormal_units, scaled_integer_to_f64};
use num_complex::Complex64;
use num_traits::{Signed, Zero};
use thiserror::Error;

use super::lcu::{LcuCoefficientPlan, plan_lcu_coefficients};
use super::signals::{MAX_SIGNAL_REGION_WIDTH, SignalRegion};

/// Maximum distinct branches in a coherent signal sum.
pub const MAX_SIGNAL_BRANCHES: usize = 256;
/// Maximum expanded recipe size before building source closures.
pub const MAX_SIGNAL_NODES: usize = 4096;
/// Maximum nested composition depth accepted by source factories.
pub const MAX_SIGNAL_DEPTH: usize = 32;

/// Invalid projected-map request.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum SignalError {
    /// Logical bit-coordinate spaces must agree at composition boundaries.
    #[error("Q7740: incompatible signal input/output spaces")]
    Space,
    /// Finite positive scales and nonnegative errors are required.
    #[error("Q7741: invalid or unrepresentable signal scale/error")]
    Scalar,
    /// Source construction must remain bounded before allocation.
    #[error("Q7742: signal construction exceeds width, branch, node, or depth budget")]
    Budget,
    /// Existing checked producer rejected an input.
    #[error("Q7743: {0}")]
    Producer(String),
    /// A requested theorem lacks its structural premise.
    #[error("Q7744: eigenvalue response requires a structurally Hermitian signal")]
    Hermitian,
}

/// Closed target evidence carried by library constructions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TargetKind {
    General,
    Hermitian,
}

/// Immutable selected-map contract; no independent executable representation.
#[derive(Clone, Debug)]
pub struct SignalContract {
    input: SignalRegion,
    output: SignalRegion,
    scale: f64,
    error: f64,
    nodes: usize,
    depth: usize,
    kind: TargetKind,
}

impl SignalContract {
    /// Check one imported declaration (not a proof about an external matrix).
    ///
    /// # Errors
    /// Rejects invalid regions, mismatched ambient widths, or scalar domains.
    pub fn new(input: &str, output: &str, scale: f64, error: f64) -> Result<Self, SignalError> {
        let input = SignalRegion::new(input).map_err(|e| SignalError::Producer(e.to_string()))?;
        let output = SignalRegion::new(output).map_err(|e| SignalError::Producer(e.to_string()))?;
        if input.width() != output.width() {
            return Err(SignalError::Space);
        }
        if !scale.is_finite() || scale <= 0.0 || !error.is_finite() || error < 0.0 {
            return Err(SignalError::Scalar);
        }
        Ok(Self {
            input,
            output,
            scale,
            error,
            nodes: 1,
            depth: 1,
            kind: TargetKind::General,
        })
    }

    /// Return the input embedding.
    #[must_use]
    pub const fn input(&self) -> &SignalRegion {
        &self.input
    }
    /// Return the output embedding.
    #[must_use]
    pub const fn output(&self) -> &SignalRegion {
        &self.output
    }
    /// Return the reconstruction scale.
    #[must_use]
    pub const fn scale(&self) -> f64 {
        self.scale
    }
    /// Return the reconstruction error bound, conditional on imported premises.
    #[must_use]
    pub const fn error(&self) -> f64 {
        self.error
    }
    /// Return whether the target has structural Hermitian evidence.
    #[must_use]
    pub fn is_hermitian(&self) -> bool {
        self.kind == TargetKind::Hermitian
    }
    /// Return expanded recipe size.
    #[must_use]
    pub const fn nodes(&self) -> usize {
        self.nodes
    }
    /// Return composition depth.
    #[must_use]
    pub const fn depth(&self) -> usize {
        self.depth
    }

    /// Change reconstruction scale while conservatively retaining the same target.
    ///
    /// # Errors
    /// Rejects invalid scales or an error exceeding the explicit tolerance.
    pub fn rescale(&self, scale: f64, max_error: f64) -> Result<Self, SignalError> {
        if !scale.is_finite() || scale <= 0.0 || !max_error.is_finite() || max_error <= 0.0 {
            return Err(SignalError::Scalar);
        }
        #[expect(
            clippy::float_cmp,
            reason = "Only an exactly unchanged scale contributes zero reconstruction error"
        )]
        let difference = if scale == self.scale {
            0.0
        } else {
            (scale - self.scale).abs().next_up()
        };
        let error = add_up(self.error, difference)?;
        if error > max_error {
            return Err(SignalError::Producer(format!(
                "total response error {error} exceeds requested {max_error}"
            )));
        }
        Ok(Self {
            scale,
            error,
            ..self.clone()
        })
    }

    /// Derive an adjoint contract without changing its error units.
    ///
    /// # Errors
    /// Rejects excessive nested source construction.
    pub fn adjoint(&self) -> Result<Self, SignalError> {
        Self {
            input: self.output.clone(),
            output: self.input.clone(),
            ..self.clone()
        }
        .budget(&[self])
    }

    /// Define a fresh prepared-state target without inheriting old matrix claims.
    ///
    /// # Errors
    /// Rejects excessive nested source construction.
    pub fn prepared(&self) -> Result<Self, SignalError> {
        Self::new(
            &"0".repeat(self.input.width()),
            self.output.pattern(),
            1.0,
            0.0,
        )?
        .budget(&[self])
    }

    /// Derive evidence for the explicit `(A + A†)/2` construction.
    ///
    /// # Errors
    /// The two logical spaces must be the same canonical bit space.
    pub fn hermitian_part(&self) -> Result<Self, SignalError> {
        if self.input.payload_indices().len() != self.output.payload_indices().len() {
            return Err(SignalError::Space);
        }
        Ok(Self {
            kind: TargetKind::Hermitian,
            ..self.clone()
        })
    }

    pub(super) fn derived(self, operands: &[&Self]) -> Result<Self, SignalError> {
        self.budget(operands)
    }

    fn budget(mut self, operands: &[&Self]) -> Result<Self, SignalError> {
        self.nodes = 1 + operands.iter().map(|s| s.nodes).sum::<usize>();
        self.depth = 1 + operands.iter().map(|s| s.depth).max().unwrap_or(0);
        if self.nodes > MAX_SIGNAL_NODES || self.depth > MAX_SIGNAL_DEPTH {
            return Err(SignalError::Budget);
        }
        Ok(self)
    }
}

/// Routing witness for `left @ right`, with independent success workspaces.
#[derive(Clone, Debug)]
pub struct SignalProductPlan {
    contract: SignalContract,
    swaps: Vec<(usize, usize)>,
}

impl SignalProductPlan {
    /// Return the resulting checked contract.
    #[must_use]
    pub const fn contract(&self) -> &SignalContract {
        &self.contract
    }
    /// Return exchanges from the right output to the left input coordinates.
    #[must_use]
    pub fn swaps(&self) -> &[(usize, usize)] {
        &self.swaps
    }
}

/// Plan a selected-map product, not a product of complete processes.
///
/// # Errors
/// Rejects incompatible intermediate spaces, excessive work, or scalar overflow.
pub fn signal_product(
    left: &SignalContract,
    right: &SignalContract,
) -> Result<SignalProductPlan, SignalError> {
    if left.input.payload_indices().len() != right.output.payload_indices().len() {
        return Err(SignalError::Space);
    }
    let offset = right.input.width();
    if offset + left.input.width() > MAX_SIGNAL_REGION_WIDTH {
        return Err(SignalError::Budget);
    }
    let input = format!(
        "{}{}",
        right.input.pattern(),
        left.input.pattern().replace('*', "0")
    );
    let output = format!(
        "{}{}",
        right.output.pattern().replace('*', "0"),
        left.output.pattern()
    );
    let scale = right.scale * left.scale;
    // One rounded multiplication of the reconstruction scale is itself bounded.
    let scale_rounding = product_rounding(right.scale, left.scale, scale)?;
    let error = add_up(
        add_up(
            mul_up(left.scale, right.error)?,
            mul_up(right.scale, left.error)?,
        )?,
        add_up(mul_up(left.error, right.error)?, scale_rounding)?,
    )?;
    let contract = SignalContract::new(&input, &output, scale, error)?.budget(&[left, right])?;
    let swaps = right
        .output
        .payload_indices()
        .iter()
        .zip(left.input.payload_indices())
        .map(|(&r, &l)| (r, offset + l))
        .collect();
    Ok(SignalProductPlan { contract, swaps })
}

/// Source operations mapping the canonical zero-prefix embedding to a region.
#[derive(Clone, Debug)]
pub struct SignalEmbeddingPlan {
    swaps: Vec<(usize, usize)>,
    flips: Vec<usize>,
}

impl SignalEmbeddingPlan {
    /// Return swaps performed before the fixed-one flips.
    #[must_use]
    pub fn swaps(&self) -> &[(usize, usize)] {
        &self.swaps
    }
    /// Return original coordinates fixed to one.
    #[must_use]
    pub fn flips(&self) -> &[usize] {
        &self.flips
    }
}

/// Derive a complete, source-order-preserving embedding basis change.
#[must_use]
pub fn signal_embedding(region: &SignalRegion) -> SignalEmbeddingPlan {
    let fixed: Vec<_> = region
        .pattern()
        .bytes()
        .enumerate()
        .filter_map(|(i, c)| (c != b'*').then_some(i))
        .collect();
    let desired: Vec<_> = fixed
        .into_iter()
        .chain(region.payload_indices().iter().copied())
        .collect();
    let mut current: Vec<_> = (0..region.width()).collect();
    let mut positions = current.clone();
    let mut pack = Vec::new();
    for (destination, original) in desired.into_iter().enumerate() {
        let source = positions[original];
        if source != destination {
            pack.push((destination, source));
            positions[current[destination]] = source;
            positions[original] = destination;
            current.swap(destination, source);
        }
    }
    pack.reverse();
    let flips = region
        .pattern()
        .bytes()
        .enumerate()
        .filter_map(|(i, c)| (c == b'1').then_some(i))
        .collect();
    SignalEmbeddingPlan { swaps: pack, flips }
}

/// A common-layout coherent sum with scale-aware PREPARE coefficients.
#[derive(Clone, Debug)]
pub struct SignalSumPlan {
    contract: SignalContract,
    workspace: usize,
    coefficients: LcuCoefficientPlan,
}

impl SignalSumPlan {
    /// Return the complete selected-map contract, including selector success.
    #[must_use]
    pub const fn contract(&self) -> &SignalContract {
        &self.contract
    }
    /// Return the shared ambient width before prepending the selector.
    #[must_use]
    pub const fn workspace(&self) -> usize {
        self.workspace
    }
    /// Return checked preparation, phase, and branch coordinates.
    #[must_use]
    pub const fn coefficients(&self) -> &LcuCoefficientPlan {
        &self.coefficients
    }
}

/// Plan a weighted sum. Groups must identify identical complete witnesses.
///
/// # Errors
/// Rejects shape disagreement, invalid coefficients, scalar overflow, or budgets.
pub fn signal_sum(
    operands: &[SignalContract],
    rows: &[(usize, Complex64)],
) -> Result<SignalSumPlan, SignalError> {
    if operands.is_empty()
        || operands.len() > MAX_SIGNAL_BRANCHES
        || rows.len() > MAX_SIGNAL_BRANCHES
    {
        return Err(SignalError::Budget);
    }
    let first = &operands[0];
    let input_size = first.input.payload_indices().len();
    let output_size = first.output.payload_indices().len();
    if operands.iter().any(|s| {
        s.input.payload_indices().len() != input_size
            || s.output.payload_indices().len() != output_size
    }) {
        return Err(SignalError::Space);
    }
    let raw = plan_lcu_coefficients(operands.len(), rows)
        .map_err(|e| SignalError::Producer(e.to_string()))?;
    let mut weighted = Vec::new();
    let mut error = 0.0;
    // Retain all group ids for the existing canonicalizer, including cancelled groups.
    let mut effective = vec![Complex64::default(); operands.len()];
    for term in raw.terms() {
        effective[term.group()] = term.coefficient();
    }
    for (i, (operand, c)) in operands.iter().zip(effective).enumerate() {
        let weight = c * operand.scale;
        if !weight.re.is_finite() || !weight.im.is_finite() {
            return Err(SignalError::Scalar);
        }
        let rounding = add_up(
            product_rounding(c.re, operand.scale, weight.re)?,
            product_rounding(c.im, operand.scale, weight.im)?,
        )?;
        error = add_up(
            error,
            add_up(mul_up(c.norm().next_up(), operand.error)?, rounding)?,
        )?;
        weighted.push((i, weight));
    }
    let coefficients = plan_lcu_coefficients(operands.len(), &weighted)
        .map_err(|e| SignalError::Producer(e.to_string()))?;
    let workspace = operands
        .iter()
        .map(|s| s.input.width())
        .max()
        .ok_or(SignalError::Budget)?;
    let width = workspace + coefficients.signal_width();
    if width > MAX_SIGNAL_REGION_WIDTH {
        return Err(SignalError::Budget);
    }
    let input = canonical_pattern(width, input_size);
    let output = canonical_pattern(width, output_size);
    let contract = SignalContract::new(&input, &output, coefficients.normalization(), error)?
        .budget(&operands.iter().collect::<Vec<_>>())?;
    Ok(SignalSumPlan {
        contract,
        workspace,
        coefficients,
    })
}

/// Return a zero-prefix, source-ordered logical bit-space embedding.
#[must_use]
pub fn canonical_pattern(width: usize, payload: usize) -> String {
    format!("{}{}", "0".repeat(width - payload), "*".repeat(payload))
}

fn product_rounding(a: f64, b: f64, product: f64) -> Result<f64, SignalError> {
    if !product.is_finite() || (a != 0.0 && b != 0.0 && product == 0.0) {
        return Err(SignalError::Scalar);
    }
    let residual = (float_to_subnormal_units(a) * float_to_subnormal_units(b)
        - (float_to_subnormal_units(product) << SUBNORMAL_EXPONENT))
        .abs();
    if residual.is_zero() {
        return Ok(0.0);
    }
    let rounded =
        scaled_integer_to_f64(&residual, SUBNORMAL_EXPONENT * 2).ok_or(SignalError::Scalar)?;
    Ok(rounded.next_up())
}

pub(super) fn add_up(a: f64, b: f64) -> Result<f64, SignalError> {
    if a == 0.0 {
        return Ok(b);
    }
    if b == 0.0 {
        return Ok(a);
    }
    let result = (a + b).next_up();
    if result.is_finite() {
        Ok(result)
    } else {
        Err(SignalError::Scalar)
    }
}

pub(super) fn mul_up(a: f64, b: f64) -> Result<f64, SignalError> {
    if a == 0.0 || b == 0.0 {
        return Ok(0.0);
    }
    let result = (a * b).next_up();
    if result.is_finite() {
        Ok(result)
    } else {
        Err(SignalError::Scalar)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn product_retains_all_success_conditions_and_rectangular_coordinates() {
        let a = SignalContract::new("0**", "10*", 2.0, 0.1).unwrap();
        let b = SignalContract::new("1*", "**", 3.0, 0.2).unwrap();
        let p = signal_product(&b, &a).unwrap();
        assert_eq!(p.contract.input.pattern(), "0**10");
        assert_eq!(p.contract.output.pattern(), "100**");
        assert_eq!(p.swaps, [(2, 4)]);
        assert!(p.contract.error >= 0.72);
    }
    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "These exact sums and cancellations must retain their representable canonical scales"
    )]
    fn sums_keep_scale_and_cancel_only_identical_witness_groups() {
        let a = SignalContract::new("0*", "1*", 2.0, 0.0).unwrap();
        let b = SignalContract::new("00*", "00*", 3.0, 0.0).unwrap();
        let p = signal_sum(
            &[a.clone(), b],
            &[(0, Complex64::new(1.0, 0.0)), (1, Complex64::new(0.0, 1.0))],
        )
        .unwrap();
        assert_eq!(p.contract.scale, 5.0);
        assert_eq!(p.contract.input.pattern(), "000*");
        let z = signal_sum(
            &[a],
            &[
                (0, Complex64::new(1.0, 0.0)),
                (0, Complex64::new(-1.0, 0.0)),
            ],
        )
        .unwrap();
        assert!(z.coefficients.terms().is_empty());
        assert_eq!(z.contract.scale, 1.0);
    }
}
