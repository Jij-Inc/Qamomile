"""Public backend namespace tests."""

import importlib
import importlib.util

import qamomile as qm


def test_qamomile_backend_namespaces_are_canonical() -> None:
    """Nested namespaces expose every backend implementation."""

    from qamomile.braket import Braket
    from qamomile.cudaq import Cudaq
    from qamomile.hugr import Hugr
    from qamomile.qbraid import QBraidDevice
    from qamomile.qir import QIR
    from qamomile.qiskit import Aer
    from qamomile.quration import QurationLocal
    from qamomile.quri_parts import Qulacs

    assert qm.qiskit.Aer is Aer
    assert all((Aer, Qulacs, Braket, Cudaq, Hugr, QBraidDevice, QurationLocal))
    assert qm.quration.QurationLocal is QurationLocal
    assert qm.qir.QIR is QIR
    assert qm.cirq.CirqSimulator is not None


def test_legacy_backend_packages_do_not_exist() -> None:
    """The distribution does not expose former top-level backend packages."""

    for backend in ("braket", "cudaq", "hugr", "qbraid", "qiskit", "quri_parts"):
        assert importlib.util.find_spec(f"qamomile_{backend}") is None


def test_backend_facades_reexport_the_declared_integration_api() -> None:
    """Thin backend packages preserve every declared public integration name."""

    for backend in (
        "braket",
        "cudaq",
        "qbraid",
        "qir",
        "qiskit",
        "quri_parts",
        "quration",
    ):
        facade = importlib.import_module(f"qamomile.{backend}")
        integration = importlib.import_module(f"qamomile.{backend}._integration")

        assert facade.__all__ == integration.__all__
        assert all(
            getattr(facade, name) is getattr(integration, name)
            for name in integration.__all__
        )


def test_backend_facades_preserve_public_symbol_module_names() -> None:
    """Moved implementations retain their stable public import locations."""

    symbols = {
        "qir": ("QIR", "load_ir", "load_bitcode"),
        "quration": (
            "Quration",
            "QurationLocal",
            "QurationResourceResult",
            "compile_resources",
            "load_circuit",
        ),
        "braket": ("Braket", "BraketDevice", "BraketLocal", "load_openqasm"),
        "cudaq": ("Cudaq", "CudaqLocal", "load_kernel"),
        "qbraid": ("QBraidDevice",),
        "qiskit": (
            "Aer",
            "Qiskit",
            "QiskitBackend",
            "bind_circuit",
            "decode_counts",
            "load_circuit",
        ),
        "quri_parts": (
            "Qulacs",
            "QuriParts",
            "QuriPartsSampler",
            "load_circuit",
            "load_logical_circuit",
        ),
    }

    for backend, names in symbols.items():
        facade = importlib.import_module(f"qamomile.{backend}")
        assert all(
            getattr(facade, name).__module__ == facade.__name__ for name in names
        )
