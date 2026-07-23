"""Integrity tests for the shared qkernel catalog test asset.

The catalog's minimum and fixed input metadata is what downstream consumers
(visualization regression, resource estimation with concrete bindings) feed
into ``QKernel.build``. This module guards that contract directly:
symbolic-only consumers such as the resource-estimation tests never exercise
concrete bindings, which previously let invalid metadata (wrong argument
names, affine-coverage violations that only trigger on concrete shapes) ship
unnoticed.
"""

import pytest

from tests.circuit.qkernel_catalog import QKERNEL_CATALOG


@pytest.mark.parametrize(
    "entry", QKERNEL_CATALOG, ids=[entry.id for entry in QKERNEL_CATALOG]
)
def test_entry_builds_with_minimum_inputs(entry):
    """Every catalog entry builds with its declared minimum input set."""
    block = entry.qkernel.build(**entry.minimum_inputs())
    assert block is not None


@pytest.mark.parametrize(
    "entry", QKERNEL_CATALOG, ids=[entry.id for entry in QKERNEL_CATALOG]
)
def test_metadata_names_are_kernel_arguments(entry):
    """All catalog input metadata must name real kernel arguments."""
    kernel_args = set(entry.qkernel.signature.parameters.keys())
    assert set(entry.param_names) <= kernel_args
    assert set(entry.min_params.keys()) <= kernel_args
    assert set(entry.fixed_inputs.keys()) <= kernel_args
    assert set(entry.min_params).isdisjoint(entry.fixed_inputs)
