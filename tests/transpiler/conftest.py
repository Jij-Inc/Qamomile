"""Pytest configuration for transpiler tests.

This module provides markers and hooks for organizing transpiler tests.
"""

import pytest


def pytest_configure(config):
    """Register custom markers for transpiler tests."""
    config.addinivalue_line(
        "markers", "single_qubit: mark test as single-qubit gate test"
    )
    config.addinivalue_line(
        "markers", "rotation: mark test as rotation gate test"
    )
    config.addinivalue_line(
        "markers", "two_qubit: mark test as two-qubit gate test"
    )
    config.addinivalue_line(
        "markers", "three_qubit: mark test as three-qubit gate test"
    )
    config.addinivalue_line(
        "markers", "controlled: mark test as controlled gate test"
    )
    config.addinivalue_line(
        "markers", "measurement: mark test as measurement test"
    )
    config.addinivalue_line(
        "markers", "hamiltonian: mark test as Hamiltonian transpilation test"
    )
    config.addinivalue_line(
        "markers", "backend(name): mark test for a specific backend"
    )


def pytest_collection_modifyitems(config, items):
    """Add markers to tests based on test names."""
    for item in items:
        # Add markers based on test name patterns
        if "single_qubit" in item.name:
            item.add_marker(pytest.mark.single_qubit)
        if "rotation" in item.name:
            item.add_marker(pytest.mark.rotation)
        if "two_qubit" in item.name:
            item.add_marker(pytest.mark.two_qubit)
        if "toffoli" in item.name.lower() or "three_qubit" in item.name:
            item.add_marker(pytest.mark.three_qubit)
        if "controlled" in item.name:
            item.add_marker(pytest.mark.controlled)
        if "measure" in item.name.lower():
            item.add_marker(pytest.mark.measurement)
        if "hamiltonian" in item.name.lower():
            item.add_marker(pytest.mark.hamiltonian)

        # Add backend markers based on parent class name
        parent = item.parent
        if parent and hasattr(parent, "cls"):
            cls = parent.cls
            if cls and hasattr(cls, "backend_name"):
                item.add_marker(pytest.mark.backend(cls.backend_name))
