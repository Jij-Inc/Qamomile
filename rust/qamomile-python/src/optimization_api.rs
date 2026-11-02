//! Optimization projections and entry points for the native Python module.

use super::conversion::{package_digest_hex, parse_signed_decimal, parse_unsigned_decimal};
use super::observables_api::observable_projection;
use super::results_api::SampleCountSummary;
use num_bigint::{BigInt, BigUint};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use qamomile_compiler::{
    CanonicalPolynomial, CorrelationPauli,
    DecodedPolynomialSample as NativeDecodedPolynomialSample,
    DecodedPolynomialSamples as NativeDecodedPolynomialSamples, ExactBinaryEncoding, FermionicMode,
    FqaoaDecodedSample, FqaoaPlan as NativeFqaoaPlan, ImprovementMethod, LocalSearchResult,
    MAX_FQAOA_DECODE_CELLS, MAX_FQAOA_DECODE_ROWS, MAX_QRAC_DECODE_CELLS, MAX_QRAC_DECODE_ROWS,
    PauliCorrelationPlan as NativePauliCorrelationPlan, PolynomialDomain, QracDecodedSample,
    QracPlan as NativeQracPlan, QracVariant as NativeQracVariant,
    convert_polynomial as convert_native_polynomial,
    decode_optimization_problem as decode_native_optimization_problem,
    encode_optimization_problem as encode_native_optimization_problem,
    minimize_polynomial as minimize_native_polynomial,
    minimum_pce_qubits as native_minimum_pce_qubits,
    normalize_polynomial as normalize_native_polynomial,
    optimization_problem_digest as native_optimization_problem_digest,
    validate_decoded_polynomial_samples as validate_native_decoded_polynomial_samples,
};

type DecodedPolynomialInputRow = (Vec<String>, Vec<String>, f64, String);
type DecodedPolynomialOutputRow = (Vec<String>, Vec<String>, f64, String);
type DecodedPolynomialProjection = (Vec<DecodedPolynomialOutputRow>, String, f64, usize);

/// One immutable native sparse-polynomial normal form.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native")]
struct SparsePolynomial {
    polynomial: CanonicalPolynomial,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pymethods]
impl SparsePolynomial {
    /// Normalize raw arbitrary-precision variable rows.
    #[new]
    #[pyo3(signature = (domain, terms, coefficients, constant, variables=None))]
    #[allow(clippy::needless_pass_by_value)]
    fn new(
        domain: &str,
        terms: Vec<Vec<String>>,
        coefficients: Vec<f64>,
        constant: f64,
        variables: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let domain = parse_polynomial_domain(domain)?;
        let terms = parse_polynomial_term_ids(terms)?;
        let variables = variables.map(parse_polynomial_ids).transpose()?;
        let polynomial = normalize_native_polynomial(
            domain,
            &terms,
            &coefficients,
            constant,
            variables.as_deref(),
        )
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self { polynomial })
    }

    /// Reconstruct one polynomial from its canonical integrity-sealed artifact.
    #[staticmethod]
    fn from_bytes(
        py: Python<'_>,
        #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
    ) -> PyResult<Self> {
        let verified = py
            .detach(|| decode_native_optimization_problem(payload))
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self {
            polynomial: verified.polynomial().clone(),
        })
    }

    /// Serialize this polynomial as one canonical integrity-sealed artifact.
    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let payload = py
            .detach(|| encode_native_optimization_problem(&self.polynomial))
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(PyBytes::new(py, &payload))
    }

    /// Return the Rust-owned semantic identity of this canonical problem.
    #[getter]
    fn semantic_digest(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| native_optimization_problem_digest(&self.polynomial))
            .map(|digest| package_digest_hex(&digest))
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Return the selected source value algebra.
    #[getter]
    fn domain(&self) -> &'static str {
        polynomial_domain_name(self.polynomial.domain())
    }

    /// Return lexicographically ordered canonical terms.
    #[getter]
    fn terms(&self) -> Vec<(Vec<String>, f64)> {
        self.polynomial
            .terms()
            .iter()
            .map(|term| {
                (
                    term.variables()
                        .iter()
                        .map(|variable| variable.to_str_radix(16))
                        .collect(),
                    term.coefficient(),
                )
            })
            .collect()
    }

    /// Return the exactly accumulated constant.
    #[getter]
    fn constant(&self) -> f64 {
        self.polynomial.constant()
    }

    /// Return the complete canonical variable universe.
    #[getter]
    fn variables(&self) -> Vec<String> {
        self.polynomial
            .variables()
            .iter()
            .map(|variable| variable.to_str_radix(16))
            .collect()
    }

    /// Return the equivalent bounded polynomial in another value algebra.
    fn convert(&self, target: &str) -> PyResult<Self> {
        let target = parse_polynomial_domain(target)?;
        let polynomial = convert_native_polynomial(&self.polynomial, target)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self { polynomial })
    }

    /// Return a model normalized by its largest absolute nonconstant coefficient.
    fn normalized_by_abs_max(&self, py: Python<'_>) -> PyResult<Self> {
        let polynomial = py
            .detach(|| self.polynomial.normalized_by_abs_max())
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self { polynomial })
    }

    /// Return a model normalized by its degree-wise root-mean-square scale.
    fn normalized_by_rms(&self, py: Python<'_>) -> PyResult<Self> {
        let polynomial = py
            .detach(|| self.polynomial.normalized_by_rms())
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self { polynomial })
    }

    /// Return an exact integer model, scale, strict threshold, and safe width.
    fn exact_integer_encoding(
        &self,
        py: Python<'_>,
        threshold: f64,
    ) -> PyResult<(SparsePolynomial, f64, String, usize)> {
        let encoding = py
            .detach(|| ExactBinaryEncoding::new(&self.polynomial, threshold))
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok((
            Self {
                polynomial: encoding.polynomial().clone(),
            },
            encoding.scale(),
            encoding.threshold().to_string(),
            encoding.output_bits(),
        ))
    }

    /// Return the dense commuting Pauli-Z Hamiltonian for this objective.
    fn hamiltonian<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyBytes>, bool)> {
        let observable = py
            .detach(|| self.polynomial.hamiltonian())
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(observable_projection(py, &observable))
    }

    /// Evaluate one dense assignment with native exact accumulation.
    #[allow(clippy::needless_pass_by_value)]
    fn energy(&self, py: Python<'_>, values: Vec<i8>) -> PyResult<f64> {
        py.detach(|| self.polynomial.energy(&values))
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Normalize one dense or original-ID assignment in canonical variable order.
    #[pyo3(signature = (values, variables=None))]
    #[allow(clippy::needless_pass_by_value)]
    fn normalize_assignment(
        &self,
        py: Python<'_>,
        values: Vec<String>,
        variables: Option<Vec<String>>,
    ) -> PyResult<Vec<i8>> {
        let values = parse_polynomial_assignment_values(values)?;
        let variables = variables.map(parse_polynomial_ids).transpose()?;
        py.detach(|| {
            self.polynomial
                .normalize_assignment(variables.as_deref(), &values)
        })
        .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Evaluate one dense or original-ID assignment in a single native pass.
    #[pyo3(signature = (values, variables=None))]
    #[allow(clippy::needless_pass_by_value)]
    fn energy_assignment(
        &self,
        py: Python<'_>,
        values: Vec<String>,
        variables: Option<Vec<String>>,
    ) -> PyResult<f64> {
        let values = parse_polynomial_assignment_values(values)?;
        let variables = variables.map(parse_polynomial_ids).transpose()?;
        py.detach(|| {
            self.polynomial
                .energy_assignment(variables.as_deref(), &values)
        })
        .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Decode one canonical logical histogram with the polynomial semantics.
    #[allow(clippy::needless_pass_by_value)] // PyO3 supplies the native summary by value.
    fn decode_samples(
        &self,
        py: Python<'_>,
        summary: PyRef<'_, SampleCountSummary>,
    ) -> PyResult<DecodedPolynomialProjection> {
        let summary = summary.summary().clone();
        let decoded = py
            .detach(|| self.polynomial.decode_samples(&summary))
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(decoded_polynomial_projection(&decoded))
    }

    /// Minimize from one dense assignment using the same canonical model.
    #[allow(clippy::needless_pass_by_value)]
    fn minimize(
        &self,
        py: Python<'_>,
        initial_values: Vec<i8>,
        method: &str,
        max_steps: Option<usize>,
    ) -> PyResult<(Vec<i8>, f64, usize)> {
        let method = parse_improvement_method(method)?;
        let result = py
            .detach(|| {
                minimize_native_polynomial(&self.polynomial, &initial_values, method, max_steps)
            })
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(local_search_projection(&result))
    }

    /// Minimize from one dense or original-ID assignment in a single native pass.
    #[pyo3(signature = (values, method, max_steps, variables=None))]
    #[allow(clippy::needless_pass_by_value)]
    fn minimize_assignment(
        &self,
        py: Python<'_>,
        values: Vec<String>,
        method: &str,
        max_steps: Option<usize>,
        variables: Option<Vec<String>>,
    ) -> PyResult<(Vec<i8>, f64, usize)> {
        let values = parse_polynomial_assignment_values(values)?;
        let variables = variables.map(parse_polynomial_ids).transpose()?;
        let method = parse_improvement_method(method)?;
        let initial_values = py
            .detach(|| {
                self.polynomial
                    .normalize_assignment(variables.as_deref(), &values)
            })
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let result = py
            .detach(|| {
                minimize_native_polynomial(&self.polynomial, &initial_values, method, max_steps)
            })
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(local_search_projection(&result))
    }

    /// Derive one bounded Pauli-correlation plan from this normal form.
    fn pauli_correlation_plan(
        &self,
        py: Python<'_>,
        correlator_order: usize,
        width: Option<usize>,
    ) -> PyResult<PauliCorrelationPlan> {
        let plan = py
            .detach(|| NativePauliCorrelationPlan::new(&self.polynomial, correlator_order, width))
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(PauliCorrelationPlan { plan })
    }

    /// Derive one bounded QRAC plan from this normal form.
    fn qrac_plan(&self, py: Python<'_>, variant: &str) -> PyResult<QracPlan> {
        let variant = parse_qrac_variant(variant)?;
        let plan = py
            .detach(|| NativeQracPlan::new(&self.polynomial, variant))
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(QracPlan { plan })
    }

    /// Derive one bounded FQAOA geometry, preparation, cost, and decode plan.
    #[allow(clippy::needless_pass_by_value)]
    fn fqaoa_plan(
        &self,
        py: Python<'_>,
        mode_ids: Vec<String>,
        positions: Vec<(usize, usize)>,
        particles: usize,
    ) -> PyResult<FqaoaPlan> {
        if mode_ids.len() != positions.len() {
            return Err(PyValueError::new_err(
                "Q7593: FQAOA mode IDs and positions must have equal length",
            ));
        }
        let variables = parse_polynomial_ids(mode_ids)?;
        let modes = variables
            .into_iter()
            .zip(positions)
            .map(|(variable, (site, orbital))| FermionicMode::new(variable, site, orbital))
            .collect::<Vec<_>>();
        let plan = py
            .detach(|| NativeFqaoaPlan::new(&self.polynomial, &modes, particles))
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(FqaoaPlan { plan })
    }
}

/// One immutable native Pauli-correlation coordinate plan.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native")]
struct PauliCorrelationPlan {
    plan: NativePauliCorrelationPlan,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pymethods]
impl PauliCorrelationPlan {
    /// Return the selected physical width.
    #[getter]
    fn width(&self) -> usize {
        self.plan.width()
    }

    /// Return one lexicographic Pauli-factor row per source variable.
    #[getter]
    fn words(&self) -> Vec<Vec<(usize, &'static str)>> {
        self.plan
            .words()
            .iter()
            .map(|word| {
                word.factors()
                    .iter()
                    .map(|(index, pauli)| {
                        let name = match pauli {
                            CorrelationPauli::X => "X",
                            CorrelationPauli::Y => "Y",
                            CorrelationPauli::Z => "Z",
                        };
                        (*index, name)
                    })
                    .collect()
            })
            .collect()
    }

    /// Evaluate one bounded continuous expectation coordinate.
    #[allow(clippy::needless_pass_by_value)]
    fn relaxed_cost(&self, py: Python<'_>, values: Vec<f64>) -> PyResult<f64> {
        py.detach(|| self.plan.relaxed_cost(&values))
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Sign-round expectations into the source polynomial domain.
    #[allow(clippy::needless_pass_by_value)]
    fn decode(&self, py: Python<'_>, values: Vec<f64>) -> PyResult<(Vec<i8>, f64)> {
        py.detach(|| self.plan.decode(&values))
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }
}

/// Return the bounded exact PCE capacity minimum.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn minimum_pce_qubits(
    py: Python<'_>,
    num_variables: usize,
    correlator_order: usize,
) -> PyResult<usize> {
    py.detach(|| native_minimum_pce_qubits(num_variables, correlator_order))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

type QracDecodedProjection = (Vec<i8>, f64, usize);

/// One immutable native QRAC grouping and observable plan.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native")]
struct QracPlan {
    plan: NativeQracPlan,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pymethods]
impl QracPlan {
    /// Return the selected physical width.
    #[getter]
    fn width(&self) -> usize {
        self.plan.width()
    }

    /// Return the maximum rows accepted by one decode call.
    #[getter]
    #[allow(clippy::unused_self)]
    const fn max_decode_rows(&self) -> usize {
        MAX_QRAC_DECODE_ROWS
    }

    /// Return the maximum cells accepted by one decode call.
    #[getter]
    #[allow(clippy::unused_self)]
    const fn max_decode_cells(&self) -> usize {
        MAX_QRAC_DECODE_CELLS
    }

    /// Return variable-position groups in deterministic color order.
    #[getter]
    fn groups(&self) -> Vec<Vec<usize>> {
        self.plan.groups().to_vec()
    }

    /// Return one real Pauli observable per source variable.
    #[getter]
    fn observables<'py>(&self, py: Python<'py>) -> Vec<(Bound<'py, PyBytes>, bool)> {
        self.plan
            .observables()
            .iter()
            .map(|observable| observable_projection(py, observable))
            .collect()
    }

    /// Return the real relaxed Hamiltonian.
    #[getter]
    fn hamiltonian<'py>(&self, py: Python<'py>) -> (Bound<'py, PyBytes>, bool) {
        observable_projection(py, self.plan.hamiltonian())
    }

    /// Sign-round one expectation row in Rust.
    #[allow(clippy::needless_pass_by_value)]
    fn sign_decode(
        &self,
        py: Python<'_>,
        values: Vec<f64>,
    ) -> PyResult<Vec<QracDecodedProjection>> {
        py.detach(|| self.plan.sign_decode(&values))
            .map(|samples| qrac_decoded_projection(&samples))
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Decode and aggregate rounded spin rows in Rust.
    #[allow(clippy::needless_pass_by_value)]
    fn decode(&self, py: Python<'_>, rows: Vec<Vec<i8>>) -> PyResult<Vec<QracDecodedProjection>> {
        py.detach(|| self.plan.decode(&rows))
            .map(|samples| qrac_decoded_projection(&samples))
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }
}

fn qrac_decoded_projection(samples: &[QracDecodedSample]) -> Vec<QracDecodedProjection> {
    samples
        .iter()
        .map(|sample| {
            (
                sample.values().to_vec(),
                sample.energy(),
                sample.occurrences(),
            )
        })
        .collect()
}

type FqaoaDecodedProjection = (Vec<i8>, f64, usize);

/// One immutable native FQAOA host plan.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native")]
struct FqaoaPlan {
    plan: NativeFqaoaPlan,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pymethods]
impl FqaoaPlan {
    /// Return the dense site count.
    #[getter]
    fn num_sites(&self) -> usize {
        self.plan.num_sites()
    }

    /// Return the dense orbital count.
    #[getter]
    fn num_orbitals(&self) -> usize {
        self.plan.num_orbitals()
    }

    /// Return source-variable positions in physical periodic order.
    #[getter]
    fn variable_order(&self) -> Vec<usize> {
        self.plan.variable_order().to_vec()
    }

    /// Return adjacent mode pairs for inverse preparation.
    #[getter]
    fn givens_pairs(&self) -> Vec<(usize, usize)> {
        self.plan.givens_pairs().to_vec()
    }

    /// Return finite inverse-preparation angles.
    #[getter]
    fn givens_angles(&self) -> Vec<f64> {
        self.plan.givens_angles().to_vec()
    }

    /// Return the physical-mode cost in the common Pauli wire form.
    #[getter]
    fn hamiltonian<'py>(&self, py: Python<'py>) -> (Bound<'py, PyBytes>, bool) {
        observable_projection(py, self.plan.hamiltonian())
    }

    /// Return the maximum rows accepted by one decode call.
    #[getter]
    #[allow(clippy::unused_self)]
    const fn max_decode_rows(&self) -> usize {
        MAX_FQAOA_DECODE_ROWS
    }

    /// Return the maximum cells accepted by one decode call.
    #[getter]
    #[allow(clippy::unused_self)]
    const fn max_decode_cells(&self) -> usize {
        MAX_FQAOA_DECODE_CELLS
    }

    /// Decode physical bit rows into canonical source assignments.
    #[allow(clippy::needless_pass_by_value)]
    fn decode(
        &self,
        py: Python<'_>,
        rows: Vec<(Vec<u8>, usize)>,
    ) -> PyResult<Vec<FqaoaDecodedProjection>> {
        py.detach(|| self.plan.decode(&rows))
            .map(|samples| fqaoa_decoded_projection(&samples))
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }
}

fn fqaoa_decoded_projection(samples: &[FqaoaDecodedSample]) -> Vec<FqaoaDecodedProjection> {
    samples
        .iter()
        .map(|sample| {
            (
                sample.values().to_vec(),
                sample.energy(),
                sample.occurrences(),
            )
        })
        .collect()
}

fn parse_qrac_variant(variant: &str) -> PyResult<NativeQracVariant> {
    match variant {
        "2,1" => Ok(NativeQracVariant::Qrac21),
        "3,1" => Ok(NativeQracVariant::Qrac31),
        "3,2" => Ok(NativeQracVariant::Qrac32),
        "space-efficient" => Ok(NativeQracVariant::SpaceEfficient),
        _ => Err(PyValueError::new_err("Q7580: unknown QRAC variant")),
    }
}

fn parse_polynomial_domain(domain: &str) -> PyResult<PolynomialDomain> {
    match domain {
        "BINARY" => Ok(PolynomialDomain::Binary),
        "SPIN" => Ok(PolynomialDomain::Spin),
        _ => Err(PyValueError::new_err("Q7551: unknown polynomial domain")),
    }
}

fn polynomial_domain_name(domain: PolynomialDomain) -> &'static str {
    match domain {
        PolynomialDomain::Binary => "BINARY",
        PolynomialDomain::Spin => "SPIN",
    }
}

fn parse_decoded_polynomial_rows(
    rows: Vec<DecodedPolynomialInputRow>,
) -> PyResult<Vec<NativeDecodedPolynomialSample>> {
    rows.into_iter()
        .map(|(variables, values, energy, occurrences)| {
            let variables = parse_polynomial_ids(variables)?;
            let values = values
                .into_iter()
                .map(|value| {
                    parse_signed_decimal(&value, "Q7569: invalid decoded assignment value")
                })
                .collect::<PyResult<Vec<_>>>()?;
            let occurrences = parse_unsigned_decimal(
                &occurrences,
                "Q7569: invalid decoded sample occurrence count",
            )?;
            Ok(NativeDecodedPolynomialSample::new(
                variables,
                values,
                energy,
                occurrences,
            ))
        })
        .collect()
}

fn decoded_polynomial_projection(
    decoded: &NativeDecodedPolynomialSamples,
) -> DecodedPolynomialProjection {
    let rows = decoded
        .samples()
        .iter()
        .map(|sample| {
            (
                sample
                    .variables()
                    .iter()
                    .map(|variable| variable.to_str_radix(16))
                    .collect(),
                sample
                    .values()
                    .iter()
                    .map(|value| value.to_str_radix(10))
                    .collect(),
                sample.energy(),
                sample.occurrences().to_str_radix(10),
            )
        })
        .collect();
    (
        rows,
        decoded.total_occurrences().to_str_radix(10),
        decoded.mean_energy(),
        decoded.lowest_index(),
    )
}

/// Validate and summarize one decoded optimization-sample distribution.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[allow(clippy::needless_pass_by_value)] // PyO3 extracts transient rows into owned values.
fn validate_decoded_samples(
    py: Python<'_>,
    domain: Option<&str>,
    rows: Vec<DecodedPolynomialInputRow>,
) -> PyResult<DecodedPolynomialProjection> {
    let domain = domain.map(parse_polynomial_domain).transpose()?;
    let rows = parse_decoded_polynomial_rows(rows)?;
    let decoded = py
        .detach(|| validate_native_decoded_polynomial_samples(domain, rows))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(decoded_polynomial_projection(&decoded))
}

fn parse_polynomial_term_ids(terms: Vec<Vec<String>>) -> PyResult<Vec<Vec<BigUint>>> {
    terms.into_iter().map(parse_polynomial_ids).collect()
}

fn parse_polynomial_ids(variables: Vec<String>) -> PyResult<Vec<BigUint>> {
    variables
        .into_iter()
        .map(|variable| {
            BigUint::parse_bytes(variable.as_bytes(), 16)
                .ok_or_else(|| PyValueError::new_err("Q7563: invalid polynomial variable ID"))
        })
        .collect()
}

fn parse_polynomial_assignment_values(values: Vec<String>) -> PyResult<Vec<BigInt>> {
    values
        .into_iter()
        .map(|value| parse_signed_decimal(&value, "Q7567: invalid polynomial assignment value"))
        .collect()
}

fn parse_improvement_method(method: &str) -> PyResult<ImprovementMethod> {
    Ok(match method {
        "first" => ImprovementMethod::First,
        "best" => ImprovementMethod::Best,
        _ => return Err(PyValueError::new_err("Q7552: unknown local-search method")),
    })
}

fn local_search_projection(result: &LocalSearchResult) -> (Vec<i8>, f64, usize) {
    (result.values().to_vec(), result.energy(), result.steps())
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<SparsePolynomial>()?;
    module.add_class::<PauliCorrelationPlan>()?;
    module.add_class::<QracPlan>()?;
    module.add_class::<FqaoaPlan>()?;
    module.add_function(wrap_pyfunction!(minimum_pce_qubits, module)?)?;
    module.add_function(wrap_pyfunction!(validate_decoded_samples, module)?)?;
    Ok(())
}
