"""Integrity tests for the shared qkernel catalog test asset.

The catalog's ``min_params`` metadata is what downstream consumers
(visualization regression, resource estimation with concrete bindings)
feed into ``QKernel.build``. This module guards that contract directly:
symbolic-only consumers such as the resource-estimation tests never
exercise concrete bindings, which previously let invalid ``min_params``
(wrong argument names, affine-coverage violations that only trigger on
concrete shapes) ship unnoticed.
"""

import inspect

import pytest

from tests.circuit.qkernel_catalog import QKERNEL_CATALOG


@pytest.mark.parametrize(
    "entry", QKERNEL_CATALOG, ids=[entry.id for entry in QKERNEL_CATALOG]
)
def test_entry_builds_with_min_params(entry):
    """Every catalog entry must build successfully with its declared min_params."""
    block = entry.qkernel.build(**entry.min_params)
    assert block is not None


@pytest.mark.parametrize(
    "entry", QKERNEL_CATALOG, ids=[entry.id for entry in QKERNEL_CATALOG]
)
def test_metadata_names_are_kernel_arguments(entry):
    """param_names and min_params keys must name real kernel arguments."""
    kernel_args = set(inspect.signature(entry.qkernel.func).parameters)
    assert set(entry.param_names) <= kernel_args
    assert set(entry.min_params) <= kernel_args
