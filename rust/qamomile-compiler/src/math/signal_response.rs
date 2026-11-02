//! General projected-unitary polynomial responses and explicit error premises.

use super::binary64::exact_sum;
use super::qsvt::{MAX_CERTIFIED_QSVT_DEGREE, synthesize_qsvt_polynomial};
use super::signal_algebra::{MAX_SIGNAL_NODES, SignalContract, SignalError, add_up, mul_up};
use super::signals::MAX_SIGNAL_REGION_WIDTH;

/// Checked real definite-parity Chebyshev target and explicit response scale.
#[derive(Clone, Debug)]
pub struct SignalResponse {
    coefficients: Vec<f64>,
    scale: f64,
}

/// Bounded per-term realization plan for explicit Chebyshev interference.
#[derive(Clone, Debug)]
pub struct SignalLcuResponsePlan {
    terms: Vec<(usize, f64)>,
    margin: f64,
    tolerance: f64,
}

impl SignalLcuResponsePlan {
    /// Return nonzero degree/coefficient coordinates in ascending order.
    #[must_use]
    pub fn terms(&self) -> &[(usize, f64)] {
        &self.terms
    }
    /// Return the positive-scale zero-block padding weight, possibly zero.
    #[must_use]
    pub const fn margin(&self) -> f64 {
        self.margin
    }
    /// Return the per-term reconstruction tolerance in target units.
    #[must_use]
    pub const fn tolerance(&self) -> f64 {
        self.tolerance
    }
}

impl SignalResponse {
    /// Validate a bounded target without solving a phase problem yet.
    ///
    /// # Errors
    /// Rejects nonfinite, mixed-parity, oversized, or insufficiently scaled data.
    pub fn new(coefficients: &[f64], scale: f64) -> Result<Self, SignalError> {
        if coefficients.is_empty() || coefficients.len() > MAX_CERTIFIED_QSVT_DEGREE + 1 {
            return Err(SignalError::Budget);
        }
        if !scale.is_finite() || scale <= 0.0 || coefficients.iter().any(|v| !v.is_finite()) {
            return Err(SignalError::Scalar);
        }
        let degree = coefficients.iter().rposition(|v| *v != 0.0).unwrap_or(0);
        if coefficients
            .iter()
            .enumerate()
            .any(|(i, c)| *c != 0.0 && i % 2 != degree % 2)
        {
            return Err(SignalError::Producer(
                "real definite parity is required; mixed parity needs an explicit construction"
                    .into(),
            ));
        }
        Ok(Self {
            coefficients: coefficients[..=degree].to_vec(),
            scale,
        })
    }
    /// Return the user's ascending first-kind Chebyshev coordinates.
    #[must_use]
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }
    /// Return the explicit realization scale.
    #[must_use]
    pub const fn scale(&self) -> f64 {
        self.scale
    }
    /// Return the canonical degree.
    #[must_use]
    pub fn degree(&self) -> usize {
        self.coefficients.len() - 1
    }

    /// Plan an explicit Chebyshev-LCU alternative outside the Newton solver region.
    ///
    /// # Errors
    /// Requires scale at least the coefficient one-norm and bounded expanded work.
    pub fn lcu_terms(
        &self,
        source: &SignalContract,
        max_error: f64,
    ) -> Result<SignalLcuResponsePlan, SignalError> {
        let terms: Vec<_> = self
            .coefficients
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, c)| *c != 0.0)
            .collect();
        let l1 = exact_sum(terms.iter().map(|(_, c)| c.abs())).ok_or(SignalError::Scalar)?;
        if l1 > self.scale {
            return Err(SignalError::Producer(format!(
                "Chebyshev LCU requires scale at least {l1}; scale is not changed automatically"
            )));
        }
        let margin = self.scale - l1;
        let branches = terms.len() + usize::from(margin > 0.0);
        let selector = usize::try_from(branches.max(2).next_power_of_two().ilog2())
            .map_err(|_| SignalError::Budget)?;
        if source.input().width() + 1 + usize::from(margin > 0.0) + selector
            > MAX_SIGNAL_REGION_WIDTH
            || branches * (source.nodes() * 2 + 2) > MAX_SIGNAL_NODES
        {
            return Err(SignalError::Budget);
        }
        let count = f64::from(u32::try_from(branches + 1).map_err(|_| SignalError::Budget)?);
        let tolerance = max_error / count / 4.0;
        if !tolerance.is_finite() || tolerance <= 0.0 {
            return Err(SignalError::Scalar);
        }
        Ok(SignalLcuResponsePlan {
            terms,
            margin,
            tolerance,
        })
    }
    /// Evaluate the requested target (not an observed or certified realization).
    ///
    /// # Errors
    /// Rejects nonfinite coordinates or points outside the normalized interval.
    pub fn evaluate(&self, x: f64) -> Result<f64, SignalError> {
        if !x.is_finite() || !(-1.0..=1.0).contains(&x) {
            return Err(SignalError::Scalar);
        }
        let mut t0 = 1.0;
        let mut t1 = x;
        let mut value = self.coefficients[0];
        for (i, c) in self.coefficients.iter().enumerate().skip(1) {
            if i > 1 {
                let next = 2.0 * x * t1 - t0;
                t0 = t1;
                t1 = next;
            }
            value += c * t1;
        }
        if value.is_finite() {
            Ok(value)
        } else {
            Err(SignalError::Scalar)
        }
    }
}

/// Certified two-branch phase witness preserving the same singular orientation.
#[derive(Clone, Debug)]
pub struct SignalResponsePlan {
    contract: SignalContract,
    phases: Vec<(f64, f64)>,
    reverse: bool,
    synthesis_error: f64,
    input_error: f64,
}

impl SignalResponsePlan {
    /// Return the complete response contract including its extraction selector.
    #[must_use]
    pub const fn contract(&self) -> &SignalContract {
        &self.contract
    }
    /// Return `(selected phase, scalar phase)` rows, in application order.
    #[must_use]
    pub fn phases(&self) -> &[(f64, f64)] {
        &self.phases
    }
    /// Return whether the even output-space response uses the adjoint source.
    #[must_use]
    pub const fn reverse(&self) -> bool {
        self.reverse
    }
    /// Return synthesis plus coefficient-conversion error in target units.
    #[must_use]
    pub const fn synthesis_error(&self) -> f64 {
        self.synthesis_error
    }
    /// Return the conditional noncommutative input-perturbation enclosure.
    #[must_use]
    pub const fn input_error(&self) -> f64 {
        self.input_error
    }
}

/// Synthesize a response to `A / alpha`, preserving explicit input/output spaces.
///
/// # Errors
/// Rejects absent Hermitian premises, invalid parity/side, exhausted synthesis,
/// or a total error larger than the user's requested reconstruction tolerance.
pub fn signal_response_plan(
    spec: &SignalResponse,
    source: &SignalContract,
    kind: &str,
    side: &str,
    max_error: f64,
    max_iterations: usize,
) -> Result<SignalResponsePlan, SignalError> {
    if !max_error.is_finite() || max_error <= 0.0 {
        return Err(SignalError::Scalar);
    }
    if kind != "singular" && kind != "eigenvalue" {
        return Err(SignalError::Producer(
            "kind must be singular or eigenvalue".into(),
        ));
    }
    if kind == "eigenvalue" && !source.is_hermitian() {
        return Err(SignalError::Hermitian);
    }
    let odd = spec.degree() % 2 == 1;
    if (odd && !side.is_empty()) || (!odd && side != "input" && side != "output") {
        return Err(SignalError::Producer(
            "odd responses have no side; even responses require input or output".into(),
        ));
    }
    let reverse = !odd && side == "output";
    let oriented = if reverse {
        source.adjoint()?
    } else {
        source.clone()
    };
    let normalized: Vec<_> = spec.coefficients.iter().map(|c| c / spec.scale).collect();
    let tolerance = max_error / spec.scale / 2.0;
    let certificate = synthesize_qsvt_polynomial(&normalized, tolerance, max_iterations)
        .map_err(|e| SignalError::Producer(format!("{e}; scale is not changed automatically")))?;
    let mut conversion = 0.0;
    #[expect(
        clippy::float_cmp,
        reason = "Only division by exactly one can omit the coefficient-conversion error bound"
    )]
    if spec.scale != 1.0 {
        for value in &normalized {
            if *value != 0.0 {
                conversion = add_up(
                    conversion,
                    mul_up(
                        spec.scale,
                        add_up(mul_up(value.abs(), f64::EPSILON)?, f64::from_bits(1))?,
                    )?,
                )?;
            }
        }
    }
    let synthesis_error = add_up(
        mul_up(spec.scale, certificate.uniform_error_bound())?,
        conversion,
    )?;
    let input_error = perturbation(spec, source)?;
    let total_error = add_up(synthesis_error, input_error)?;
    if total_error > max_error {
        return Err(SignalError::Producer(format!(
            "total response error {total_error} exceeds requested {max_error}"
        )));
    }
    let input = format!("0{}", oriented.input().pattern());
    let output = format!(
        "0{}",
        if odd {
            oriented.output().pattern()
        } else {
            oriented.input().pattern()
        }
    );
    let contract = SignalContract::new(&input, &output, spec.scale, total_error)?
        .derived(&[source, source])?;
    let contract = if !odd || source.is_hermitian() {
        contract.hermitian_part()?
    } else {
        contract
    };
    let phases = if spec.degree() == 0 {
        certificate.qsp_phases()
    } else {
        certificate.projector_phases()
    }
    .iter()
    .map(|p| (2.0 * p, -p))
    .collect();
    Ok(SignalResponsePlan {
        contract,
        phases,
        reverse,
        synthesis_error,
        input_error,
    })
}

fn perturbation(spec: &SignalResponse, source: &SignalContract) -> Result<f64, SignalError> {
    if source.error() == 0.0 {
        return Ok(0.0);
    }
    let delta = (source.error() / source.scale()).next_up();
    let radius = add_up(1.0, delta)?;
    let mut power = 1.0;
    let mut previous_l1 = 1.0;
    let mut l1 = 1.0;
    let mut result = 0.0;
    for (j, c) in spec.coefficients.iter().enumerate().skip(1) {
        if j > 1 {
            let next = add_up(mul_up(2.0, l1)?, previous_l1)?;
            previous_l1 = l1;
            l1 = next;
        }
        let degree = f64::from(u32::try_from(j).map_err(|_| SignalError::Budget)?);
        let bound = mul_up(mul_up(mul_up(degree, power)?, delta)?, l1)?;
        result = add_up(result, mul_up(c.abs(), bound)?)?;
        power = mul_up(power, radius)?;
    }
    Ok(result)
}
