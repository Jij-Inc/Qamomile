"""Expose the exact-distribution version policy for qkernel serialization.

The protobuf payload stores Qamomile's complete installed distribution version
rather than an independent integer schema counter. Development and local build
metadata remain significant so different wire implementations cannot claim the
same compatibility marker. The format provides no cross-version migration
layer.
"""

from importlib.metadata import version

QAMOMILE_VERSION: str = version("qamomile")

__all__ = ["QAMOMILE_VERSION"]
