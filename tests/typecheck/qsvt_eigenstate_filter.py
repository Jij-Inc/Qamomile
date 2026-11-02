"""QSVT filtering retains the original model domain in its public lifecycle."""

from __future__ import annotations

from typing import assert_type

import qamomile as qm
from qamomile.optimization import (
    BinaryDomain,
    BinaryModel,
    CompiledQSVTEigenstateFilterExperiment,
    DecodedSamples,
    QSVTEigenstateFilterExperiment,
    QSVTFilterRun,
    SpinDomain,
)
from qamomile.stdlib import QSVTPhases, signals


def _check_filter_domain_types(result: qm.SampleResult) -> None:
    """Compilation, success readout, and postselection preserve domain types."""

    model = BinaryModel.from_hubo({(7,): 2.0}, constant=-0.5)
    phases = QSVTPhases.from_qsp_wx((0.0, 0.0))
    experiment = QSVTEigenstateFilterExperiment(model, 0.75, phases=phases)
    assert_type(experiment, QSVTEigenstateFilterExperiment[BinaryDomain])
    assert_type(experiment.decode(result), DecodedSamples[BinaryDomain] | None)
    assert_type(experiment.success_probability(result), float)
    compiled = experiment.compile(to=qm.Reference().target)
    assert_type(compiled, CompiledQSVTEigenstateFilterExperiment[BinaryDomain])
    run = compiled.run(on=qm.Reference(), shots=8)
    assert_type(run, QSVTFilterRun[BinaryDomain])
    assert_type(run.samples, DecodedSamples[BinaryDomain] | None)
    assert_type(run.success, signals.SuccessEstimate)
    assert_type(run.success_probability, float)
    spin = QSVTEigenstateFilterExperiment(model.to_ising(), 0.75, phases=phases)
    assert_type(spin, QSVTEigenstateFilterExperiment[SpinDomain])
    assert_type(spin.run(on=qm.Reference(), shots=8), QSVTFilterRun[SpinDomain])
