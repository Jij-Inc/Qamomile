#!/usr/bin/env python3
"""Exercise the installed Qamomile distribution without repository imports."""

from __future__ import annotations

import importlib
import importlib.util
import os
from importlib import metadata
from pathlib import Path
from xml.etree import ElementTree

import qamomile as qm

PUBLIC_PACKAGES = (
    "qamomile",
    "qamomile.braket",
    "qamomile.cirq",
    "qamomile.cudaq",
    "qamomile.hugr",
    "qamomile.qbraid",
    "qamomile.qir",
    "qamomile.qiskit",
    "qamomile.quri_parts",
    "qamomile.quration",
    "qamomile.visualization",
)
REMOVED_PACKAGES = tuple(
    f"qamomile_{backend}"
    for backend in ("braket", "cudaq", "hugr", "qbraid", "qiskit", "quri_parts")
)

SOURCE = """
import qamomile as qm

@qm.qkernel
def one() -> qm.Bits[1]:
    state = qm.qubits(1)
    state = qm.x(state)
    return qm.measure(state)
"""


def main() -> None:
    """Verify package identity, layout, native execution, and visualization."""

    expected_version = os.environ["EXPECTED_VERSION"]
    assert metadata.version("qamomile") == expected_version
    try:
        metadata.distribution("qamomile-native")
    except metadata.PackageNotFoundError:
        pass
    else:
        raise AssertionError("obsolete qamomile-native distribution is installed")
    from qamomile import _qamomile_native as native

    package_root = Path(qm.__file__).resolve().parent
    assert Path(native.__file__).resolve().parent == package_root
    assert (package_root / "_qamomile_native.pyi").is_file()
    assert native.VerifiedProgramHandle.__module__ == "qamomile._qamomile_native"
    for package in PUBLIC_PACKAGES:
        module = importlib.import_module(package)
        assert module.__file__ is not None
    for package in REMOVED_PACKAGES:
        assert importlib.util.find_spec(package) is None

    program = qm.build_source(SOURCE).specialize("one")
    before = program.to_bytes()
    diagram = qm.visualization.draw(program)
    svg = diagram.to_svg()
    assert ElementTree.fromstring(svg).tag == "{http://www.w3.org/2000/svg}svg"
    assert 'data-opcode="quantum.x"' in svg
    assert "Open body" in diagram.to_html()
    assert program.to_bytes() == before
    study = qm.resources.logical(program)
    assert study.metric(qm.resources.LOGICAL_PEAK_QUBITS).numeric_enclosure() == (
        1,
        1,
    )
    result = qm.run(program, on=qm.Reference(), shots=4, seed=7).result
    assert result.counts == {(True,): 4}


if __name__ == "__main__":
    main()
