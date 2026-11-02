# Installation

Qamomile requires Python 3.11 or newer. Install the public distribution with pip:

```bash
pip install qamomile
```

The distribution is named `qamomile`, while the Qamomile language module is imported as follows:

```python
import qamomile as qm
```

The distribution includes the compatible Rust compiler. The built-in `qm.Reference()` runtime, compiler, and symbolic resource analysis need no optional quantum SDK.

## Optional SDK integrations

Install only the target integrations that you need. A Qamomile program keeps the same source semantics and typed result when its endpoint changes.

::::{tab-set}

:::{tab-item} Qiskit
:sync: qiskit

```bash
pip install "qamomile[qiskit]"
```

```python
from qamomile.qiskit import Aer, Qiskit
```

`Qiskit()` authors a QPY artifact. `Aer()` authors and executes it locally.

:::

:::{tab-item} QURI Parts
:sync: quri-parts

```bash
pip install "qamomile[quri_parts]"
```

```python
from qamomile.quri_parts import Qulacs, QuriParts
```

`QuriParts()` authors a portable logical-circuit artifact. `Qulacs()` executes that artifact locally.

:::

:::{tab-item} Amazon Braket
:sync: braket

```bash
pip install "qamomile[braket]"
```

```python
from qamomile.braket import Braket, BraketDevice, BraketLocal
```

`Braket()` authors OpenQASM 3. `BraketLocal()` uses the local simulator, and `BraketDevice` is the managed-device endpoint.

:::

:::{tab-item} CUDA-Q
:sync: cuda-q

CUDA-Q is available on Linux and macOS ARM64. Install the extra matching your platform and CUDA version:

```bash
pip install "qamomile[cudaq-cu12]"  # CUDA 12.x on Linux
pip install "qamomile[cudaq-cu13]"  # CUDA 13.x on Linux or macOS ARM64
```

```python
from qamomile.cudaq import Cudaq, CudaqLocal
```

:::

:::{tab-item} HUGR, Selene, and Helios
:sync: hugr

```bash
pip install "qamomile[hugr]"
```

```python
from qamomile.hugr import Helios, Hugr, Selene
```

This integration lowers structured QCore directly to HUGR. `Selene()` builds
and executes the package locally with Quest, while `Helios` submits the same
result-instrumented HUGR profile through Nexus.

:::

:::{tab-item} Cirq and Google QVM
:sync: cirq

```bash
pip install "qamomile[cirq]"
```

```python
from qamomile.cirq import Cirq, CirqSimulator, DepolarizingNoise, google_qvm
```

`Cirq()` authors portable Cirq JSON. `CirqSimulator()` executes it in the
calling process, optionally with `DepolarizingNoise(p)`. `google_qvm()` adds
automatic placement and routing for a local Google virtual processor.

:::

::::

## Source checkout

Source builds require Git, Python 3.11 or newer, `rustup`, and a native linker. Use Xcode Command Line Tools on macOS, C/C++ build tools on Linux, or MSVC C++ build tools on Windows. `rustup` selects the Rust version pinned in `rust-toolchain.toml`.

For repository development, install `uv` and run this command from the repository root. It builds Qamomile and its bundled Rust compiler as part of the locked environment:

```bash
uv sync --frozen --group dev
```

Alternatively, install the source checkout with pip from the repository root. The same build prerequisites apply. This command installs Qamomile and its bundled Rust compiler together:

```bash
pip install -e .
```
