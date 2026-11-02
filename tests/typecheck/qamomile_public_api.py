"""Static contracts for the typed Qamomile program lifecycle."""

from __future__ import annotations

from typing import assert_type

import numpy as np
import numpy.typing as npt
import ommx.v1

import qamomile as qm
from qamomile.compilation import ExecutionEndpoint
from qamomile.optimization import (
    BinaryDomain,
    BinaryModel,
    CompiledFQAOAExperiment,
    CompiledQAOAExperiment,
    DecodedSamples,
    FQAOAExperiment,
    OMMXEvaluation,
    OMMXProblem,
    OptimizationRun,
    PauliCorrelationEncoding,
    QAOAExperiment,
    QRACEncoding,
    QRACVariant,
    SpinDomain,
)
from qamomile.stdlib import grover_iteration_count

_as_uint = qm.as_uint


@qm.qkernel
def _add(left: int, right: int) -> int:
    """Return a scalar sum inside Qamomile source."""

    return left + right


@qm.qkernel(kind="classical")
def _choose(value: float, *, enabled: bool = True) -> float:
    """Exercise configured decorators and keyword-only parameters."""

    if enabled:
        return value
    return 0.0


@qm.qkernel
def _increment_twice(value: int) -> int:
    """Exercise typed calls between Qamomile source declarations."""

    return _add(_add(value, 1), 1)


def _check_qkernel_signature() -> None:
    """Verify that decoration preserves argument and result types."""

    add_ref: qm.QKernel[..., int] = _add
    assert_type(add_ref, qm.QKernel[..., int])
    assert_type(_add(1, 2), int)
    assert_type(_choose(1.5), float)
    assert_type(_choose(value=1.5, enabled=False), float)
    assert_type(_increment_twice(1), int)
    assert_type(grover_iteration_count(4), int)
    assert_type(qm.linalg.HermitianMatrix(np.eye(2)).to_hamiltonian(), qm.Hamiltonian)
    assert_type(
        qm.linalg.decompose_pauli_matrix(np.eye(2)),
        qm.linalg.PauliDecomposition,
    )
    assert_type(
        qm.linalg.generalized_subspace_matrices(
            ((0,),),
            (qm.Pauli.X,),
            qm.Z(0),
        ),
        tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]],
    )


def _check_optimization_domain_types(result: qm.SampleResult) -> None:
    """Verify model construction and decoding retain the value domain."""

    binary = BinaryModel.from_hubo({(0, 1, 2): 1.0})
    spin = binary.to_ising()
    assert_type(binary, BinaryModel[BinaryDomain])
    assert_type(spin, BinaryModel[SpinDomain])
    assert_type(binary.decode(result), DecodedSamples[BinaryDomain])
    assert_type(spin.decode(result), DecodedSamples[SpinDomain])
    encoding = PauliCorrelationEncoding(binary, correlator_order=2)
    assert_type(encoding, PauliCorrelationEncoding[BinaryDomain])
    assert_type(encoding.decode((0.2, -0.1, 0.5)), DecodedSamples[BinaryDomain])
    qrac = QRACEncoding(binary, QRACVariant.QRAC31)
    assert_type(qrac, QRACEncoding[BinaryDomain])
    assert_type(qrac.decode((1, -1, 1)), DecodedSamples[BinaryDomain])
    experiment = QAOAExperiment(binary, layers=2)
    assert_type(experiment, QAOAExperiment[BinaryDomain])
    compiled = experiment.compile(to=qm.Reference())
    assert_type(compiled, CompiledQAOAExperiment[BinaryDomain])
    assert_type(
        compiled.run(
            on=qm.Reference(),
            gammas=(0.1, 0.2),
            betas=(0.3, 0.4),
            shots=10,
        ),
        OptimizationRun[BinaryDomain],
    )


def _check_ommx_problem_types(
    instance: ommx.v1.Instance,
    result: qm.SampleResult,
) -> None:
    """Verify the OMMX adapter always produces a BINARY penalty model."""

    problem = OMMXProblem.from_instance(instance)
    assert_type(problem.model, BinaryModel[BinaryDomain])
    assert_type(problem.evaluate(problem.model.decode(result)), OMMXEvaluation)


def _check_fqaoa_types(
    experiment: FQAOAExperiment,
    endpoint: ExecutionEndpoint,
) -> None:
    """FQAOA execution retains the BINARY optimization domain."""

    assert_type(
        experiment.compile(to=endpoint.target),
        CompiledFQAOAExperiment,
    )
    assert_type(
        experiment.compile(to=endpoint.target).run(
            on=endpoint,
            gammas=(0.1,),
            betas=(0.2,),
            shots=10,
        ),
        OptimizationRun[BinaryDomain],
    )


def _check_program_lifecycle(source: str) -> None:
    """Verify the supported source-to-execution value transitions statically."""

    module = qm.build_source(source)
    assert_type(module, qm.Module)

    family = module["entry"]
    assert_type(family, qm.ProgramFamily)

    program = family.specialize()
    assert_type(program, qm.Program)

    study = qm.resources.logical(program)
    assert_type(study, qm.resources.ResourceStudy)

    endpoint = qm.Reference()
    artifact = qm.compile(study, to=endpoint)
    assert_type(artifact, qm.CompiledProgram)

    record = qm.run(program, on=endpoint)
    assert_type(record, qm.RunRecord)
    assert_type(record.invocation, qm.Invocation)
    assert_type(record.result, qm.ValueResult | qm.SampleResult)
    assert_type(record.resources, qm.resources.ResourceStudy)
