"""Per-SDK end-to-end verification of the Transpiler tab snippets.

Articles under ``docs/{en,ja}/{tutorial,algorithm}/`` that ship a
3-SDK Transpiler tab block advertise that swapping the Qiskit
default with the QURI Parts or CUDA-Q tab snippet produces a
runnable program. The Qiskit default is already executed by
``tests/docs/test_tutorials.py``; this file pins the other two SDKs.

For each (article, sdk) case the test:

1. Reads the .py source.
2. Replaces (a) the article-top Transpiler import + instantiation
   with the matching SDK-specific two-liner, and (b) the SDK-specific
   body cell (seeded executor / estimator / statevector helper) with
   the matching tab snippet.
3. Writes the rewritten source to a temp file and runs it via
   ``runpy.run_path`` with ``QAMOMILE_DOCS_TEST=1`` so optimisation
   loops short-circuit.
4. Considers any successful execution a pass.

Markers:

- ``@pytest.mark.docs`` so the file participates in the docs job.
- ``@pytest.mark.quri_parts`` on QURI Parts cases — only the
  ``quri-parts-test`` CI job (``--extra quri_parts``) selects them.
- ``@pytest.mark.cudaq`` on CUDA-Q cases — only the ``cudaq-test-*``
  CI jobs select them.

Two SDK gaps that map to article-level decisions:

- **qec / steane on QURI Parts**: qulacs (the QURI Parts default
  simulator) does not expose mid-circuit measurement at the public
  API level, so syndrome → conditional-correction flows cannot run
  on it regardless of Qamomile's emit pass. Those two articles
  intentionally drop the QURI Parts tab (the article-top tab still
  has a QURI Parts entry but with an explicit "this article does
  not run on QURI Parts" warning), and we therefore exclude both
  from ``QURI_PARTS_CASES`` rather than mark them ``xfail``.
- **qec / steane on CUDA-Q**: covered after the
  ``CudaqEmitPass._emit_runtime_classical_expr`` lowering landed
  alongside this PR; included in ``CUDAQ_CASES``.
"""

from __future__ import annotations

import runpy
import tempfile
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Article-top Transpiler import substitutions (same for every article).
# ---------------------------------------------------------------------------

TRANSPILER_QURI_PARTS = (
    "from qamomile.qiskit import QiskitTranspiler\n\ntranspiler = QiskitTranspiler()",
    "from qamomile.quri_parts import QuriPartsTranspiler\n\ntranspiler = QuriPartsTranspiler()",
)
TRANSPILER_CUDAQ = (
    "from qamomile.qiskit import QiskitTranspiler\n\ntranspiler = QiskitTranspiler()",
    "from qamomile.cudaq import CudaqTranspiler\n\ntranspiler = CudaqTranspiler()",
)


# ---------------------------------------------------------------------------
# Pattern A — seeded sampler executor (qaoa_*, qec, steane).
# ---------------------------------------------------------------------------

QAOA_MAXCUT_QISKIT_BODY = """from qiskit_aer import AerSimulator


def make_executor():
    \"\"\"Fresh executor with deterministic sampling for this tutorial.\"\"\"
    return transpiler.executor(
        backend=AerSimulator(seed_simulator=SEED, max_parallel_threads=1)
    )


executor = make_executor()"""

QAOA_MAXCUT_QURI_BODY = """def make_executor():
    return transpiler.executor()


executor = make_executor()"""

QAOA_MAXCUT_CUDAQ_BODY = """import cudaq


def make_executor():
    cudaq.set_random_seed(SEED)
    return transpiler.executor()


executor = make_executor()"""

QAOA_GRAPH_QISKIT_BODY = """from qiskit_aer import AerSimulator

executor = transpiler.executor(
    backend=AerSimulator(seed_simulator=901, max_parallel_threads=1)
)"""

QAOA_GRAPH_QURI_BODY = "executor = transpiler.executor()"

QAOA_GRAPH_CUDAQ_BODY = """import cudaq

cudaq.set_random_seed(901)
executor = transpiler.executor()"""

QEC_STEANE_QISKIT_BODY = """from qiskit_aer import AerSimulator

_seeded_executor = transpiler.executor(
    backend=AerSimulator(seed_simulator=42, max_parallel_threads=1)
)"""

QEC_STEANE_CUDAQ_BODY = """import cudaq

cudaq.set_random_seed(42)
_seeded_executor = transpiler.executor()"""


# ---------------------------------------------------------------------------
# Pattern B — estimator-style executor (vqe).
# ---------------------------------------------------------------------------

VQE_QISKIT_BODY = """from qiskit_aer.primitives import EstimatorV2

from qamomile.qiskit.transpiler import QiskitExecutor

executor = QiskitExecutor(estimator=EstimatorV2())"""

VQE_QURI_BODY = """from qamomile.quri_parts import QuriPartsExecutor

executor = QuriPartsExecutor()"""

VQE_CUDAQ_BODY = """from qamomile.cudaq import CudaqExecutor

executor = CudaqExecutor()"""


# ---------------------------------------------------------------------------
# Pattern A (seeded) — executor for QeMCMC (qe_mcmc).
# ---------------------------------------------------------------------------

QE_MCMC_QISKIT_BODY = """from qiskit_aer import AerSimulator

executor = transpiler.executor(backend=AerSimulator(seed_simulator=7))"""

QE_MCMC_QURI_BODY = """executor = transpiler.executor()"""

QE_MCMC_CUDAQ_BODY = """import cudaq

cudaq.set_random_seed(7)
executor = transpiler.executor()"""


# ---------------------------------------------------------------------------
# Pattern C — statevector helpers (hamiltonian, 07).
# ---------------------------------------------------------------------------

HAM_QISKIT_BODY = '''from qiskit import QuantumCircuit, transpile as qk_transpile
from qiskit_aer import AerSimulator


def statevector(circuit) -> np.ndarray:
    """Strip measurements, lower PauliEvolutionGate, and read out the state.

    The default ``pauli_evolve`` emitter produces a ``PauliEvolutionGate``, which
    is not in AerSimulator\'s native basis.  We run a shallow Qiskit transpile
    pass to expand it into elementary rotations before simulating.
    """
    stripped = QuantumCircuit(*circuit.qregs)
    for instr in circuit.data:
        if instr.operation.name not in ("measure", "save_statevector"):
            stripped.append(instr)
    stripped = qk_transpile(
        stripped,
        basis_gates=["u", "cx", "rx", "ry", "rz", "h", "p", "sx", "x", "y", "z"],
    )
    stripped.save_statevector()
    sim = AerSimulator(method="statevector")
    return np.asarray(sim.run(stripped).result().get_statevector())'''

HAM_QURI_BODY = """from quri_parts.core.state import GeneralCircuitQuantumState
from quri_parts.qulacs.simulator import evaluate_state_to_vector


def statevector(circuit) -> np.ndarray:
    state = GeneralCircuitQuantumState(circuit.qubit_count, circuit)
    return np.asarray(evaluate_state_to_vector(state).vector)"""

HAM_CUDAQ_BODY = """import cudaq


def statevector(circuit) -> np.ndarray:
    state = cudaq.get_state(circuit.kernel_func)
    return np.asarray(state)"""

H07_QISKIT_BODY = """import warnings

from qiskit.quantum_info import Statevector
from scipy.sparse import SparseEfficiencyWarning

unitary_circuit = circuit.remove_final_measurements(inplace=False)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=SparseEfficiencyWarning)
    psi_qm = np.array(Statevector.from_instruction(unitary_circuit).data)

fidelity = abs(np.vdot(psi_exact, psi_qm))
print(f"fidelity (|<exact|qamomile>|): {fidelity:.12f}")
assert abs(fidelity - 1.0) < 1e-8"""

H07_QURI_BODY = """from quri_parts.core.state import GeneralCircuitQuantumState
from quri_parts.qulacs.simulator import evaluate_state_to_vector

state = GeneralCircuitQuantumState(circuit.qubit_count, circuit)
psi_qm = np.asarray(evaluate_state_to_vector(state).vector)

fidelity = abs(np.vdot(psi_exact, psi_qm))
print(f"fidelity (|<exact|qamomile>|): {fidelity:.12f}")
assert fidelity > 0.85"""

H07_CUDAQ_BODY = """import cudaq

state = cudaq.get_state(circuit.kernel_func)
psi_qm = np.asarray(state)

fidelity = abs(np.vdot(psi_exact, psi_qm))
print(f"fidelity (|<exact|qamomile>|): {fidelity:.12f}")
assert fidelity > 0.85"""


# ---------------------------------------------------------------------------
# Pattern D — reusable statevector-helper definition (mottonen).
# ---------------------------------------------------------------------------

MOTTONEN_QISKIT_BODY = """from qiskit.quantum_info import Statevector


def statevector_of(kernel: qmc.QKernel, **bindings) -> np.ndarray:
    \"\"\"Run *kernel* through Qiskit's statevector simulator and return the data.\"\"\"
    qc = transpiler.to_circuit(kernel, bindings=bindings or None)
    return Statevector.from_instruction(
        qc.remove_final_measurements(inplace=False)
    ).data"""

MOTTONEN_QURI_BODY = """from quri_parts.core.state import GeneralCircuitQuantumState
from quri_parts.qulacs.simulator import evaluate_state_to_vector


def statevector_of(kernel: qmc.QKernel, **bindings) -> np.ndarray:
    \"\"\"Run *kernel* through QURI Parts' statevector simulator and return the data.\"\"\"
    circuit = transpiler.to_circuit(kernel, bindings=bindings or None)
    state = GeneralCircuitQuantumState(circuit.qubit_count, circuit)
    return np.asarray(evaluate_state_to_vector(state).vector)"""

MOTTONEN_CUDAQ_BODY = """import cudaq


def statevector_of(kernel: qmc.QKernel, **bindings) -> np.ndarray:
    \"\"\"Run *kernel* through CUDA-Q's statevector simulator and return the data.\"\"\"
    artifact = transpiler.to_circuit(kernel, bindings=bindings or None)
    return np.asarray(cudaq.get_state(artifact.kernel_func))"""


# ---------------------------------------------------------------------------
# Per-(article, sdk) substitution tables.
# ---------------------------------------------------------------------------

ARTICLE_BODIES: dict[str, tuple[str, str | None, str | None]] = {
    # (article, qiskit_body, quri_parts_body, cudaq_body)
    "docs/en/algorithm/qaoa_maxcut.py": (
        QAOA_MAXCUT_QISKIT_BODY,
        QAOA_MAXCUT_QURI_BODY,
        QAOA_MAXCUT_CUDAQ_BODY,
    ),
    "docs/en/algorithm/qaoa_graph_partition.py": (
        QAOA_GRAPH_QISKIT_BODY,
        QAOA_GRAPH_QURI_BODY,
        QAOA_GRAPH_CUDAQ_BODY,
    ),
    "docs/en/algorithm/quantum_error_correction.py": (
        QEC_STEANE_QISKIT_BODY,
        None,  # qulacs has no mid-circuit measurement support.
        QEC_STEANE_CUDAQ_BODY,
    ),
    "docs/en/algorithm/steane_code.py": (
        QEC_STEANE_QISKIT_BODY,
        None,
        QEC_STEANE_CUDAQ_BODY,
    ),
    "docs/en/algorithm/vqe_for_hydrogen.py": (
        VQE_QISKIT_BODY,
        VQE_QURI_BODY,
        VQE_CUDAQ_BODY,
    ),
    "docs/en/algorithm/hamiltonian_simulation.py": (
        HAM_QISKIT_BODY,
        HAM_QURI_BODY,
        HAM_CUDAQ_BODY,
    ),
    "docs/en/tutorial/07_hermitian_decomposition.py": (
        H07_QISKIT_BODY,
        H07_QURI_BODY,
        H07_CUDAQ_BODY,
    ),
    "docs/en/algorithm/mottonen_amplitude_encoding.py": (
        MOTTONEN_QISKIT_BODY,
        MOTTONEN_QURI_BODY,
        MOTTONEN_CUDAQ_BODY,
    ),
    "docs/en/algorithm/qe_mcmc.py": (
        QE_MCMC_QISKIT_BODY,
        QE_MCMC_QURI_BODY,
        QE_MCMC_CUDAQ_BODY,
    ),
}


QURI_PARTS_CASES = [
    article
    for article, (_, quri_body, _) in ARTICLE_BODIES.items()
    if quri_body is not None
]
CUDAQ_CASES = [
    article
    for article, (_, _, cudaq_body) in ARTICLE_BODIES.items()
    if cudaq_body is not None
]


# ---------------------------------------------------------------------------
# Helper: apply substitutions, write to temp .py, run via runpy.
# ---------------------------------------------------------------------------


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _apply_substitutions(
    article_path: Path,
    sdk: str,
) -> str:
    """Return the article source with the ``sdk`` tab snippets applied.

    Raises:
        AssertionError: if any expected substring is not found verbatim
            in the article. The substitutions in this module are tied
            to the article structure produced by the SDK-tab pattern;
            a failure here means the article and the test data have
            drifted apart and one of them needs to be updated.
    """
    rel = article_path.relative_to(_REPO_ROOT).as_posix()
    qiskit_body, quri_body, cudaq_body = ARTICLE_BODIES[rel]

    if sdk == "quri_parts":
        assert quri_body is not None, f"No QURI Parts body for {rel}"
        transpiler_sub = TRANSPILER_QURI_PARTS
        body_sub = (qiskit_body, quri_body)
    elif sdk == "cudaq":
        assert cudaq_body is not None, f"No CUDA-Q body for {rel}"
        transpiler_sub = TRANSPILER_CUDAQ
        body_sub = (qiskit_body, cudaq_body)
    else:
        raise ValueError(f"Unknown sdk: {sdk!r}")

    text = article_path.read_text(encoding="utf-8")
    for find, replace in (transpiler_sub, body_sub):
        assert find in text, (
            f"Tab substitution drift: expected substring not found in {rel} "
            f"(sdk={sdk}). The first 80 chars: {find[:80]!r}"
        )
        text = text.replace(find, replace, 1)
    return text


def _run_substituted(article_path: Path, sdk: str, monkeypatch, tmp_path: Path) -> None:
    """Run the substituted article via ``runpy`` in an isolated cwd."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("QAMOMILE_DOCS_TEST", "1")
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)

    rewritten = _apply_substitutions(article_path, sdk)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        dir=tmp_path,
        delete=False,
        encoding="utf-8",
    ) as f:
        f.write(rewritten)
        script_path = f.name

    try:
        runpy.run_path(script_path, run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (0, None):
            pytest.fail(f"Substituted article exited with code {exc.code}")


# ---------------------------------------------------------------------------
# Test cases.
# ---------------------------------------------------------------------------


@pytest.mark.docs
@pytest.mark.quri_parts
@pytest.mark.parametrize("article_rel", QURI_PARTS_CASES)
def test_quri_parts_tab_runs(article_rel: str, monkeypatch, tmp_path: Path) -> None:
    """Each non-default QURI Parts tab snippet runs end-to-end."""
    pytest.importorskip("quri_parts")
    pytest.importorskip("qamomile.quri_parts")
    article_path = _REPO_ROOT / article_rel
    assert article_path.exists(), f"Article not found: {article_path}"
    _run_substituted(article_path, "quri_parts", monkeypatch, tmp_path)


@pytest.mark.docs
@pytest.mark.cudaq
@pytest.mark.parametrize("article_rel", CUDAQ_CASES)
def test_cudaq_tab_runs(article_rel: str, monkeypatch, tmp_path: Path) -> None:
    """Each non-default CUDA-Q tab snippet runs end-to-end."""
    pytest.importorskip("cudaq")
    pytest.importorskip("qamomile.cudaq")
    article_path = _REPO_ROOT / article_rel
    assert article_path.exists(), f"Article not found: {article_path}"
    _run_substituted(article_path, "cudaq", monkeypatch, tmp_path)
