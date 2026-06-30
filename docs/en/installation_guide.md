# Installation

Qamomile can be installed via standard package managers, such as pip.

```bash
pip install qamomile
```

Qiskit is included as the default quantum SDK integration.

```python
from qamomile.qiskit import QiskitTranspiler, QiskitExecutor
```

## Supported quantum SDKs

Qamomile supports multiple quantum SDK integrations. To use an additional quantum SDK, install its optional dependencies.

::::{tab-set}

:::{tab-item} CUDA-Q
:sync: cuda-q

CUDA-Q is available on Linux and macOS ARM64 (Apple Silicon). Install the extra that matches your CUDA version:

```bash
pip install "qamomile[cudaq-cu12]"   # CUDA 12.x, Linux
pip install "qamomile[cudaq-cu13]"   # CUDA 13.x, Linux or macOS ARM64
```

```python
from qamomile.cudaq import CudaqTranspiler, CudaqExecutor
```

:::

:::{tab-item} QURI Parts
:sync: quri-parts

```bash
pip install "qamomile[quri_parts]"
```

```python
from qamomile.quri_parts import QuriPartsTranspiler, QuriPartsExecutor
```

:::

:::{tab-item} qBraid
:sync: qbraid

Run Qiskit circuits on qBraid-supported devices and simulators.

```bash
pip install "qamomile[qbraid]"
```

```python
from qamomile.qbraid import QBraidExecutor
```

:::

::::
