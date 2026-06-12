# インストール

Qamomileは、pipなどの標準的なパッケージマネージャでインストールできます。

```bash
pip install qamomile
```

Qiskitはデフォルトの実行バックエンドとして含まれています。

```python
from qamomile.qiskit import QiskitTranspiler, QiskitExecutor
```

## 対応する量子SDK

Qamomileは、複数の量子SDKを実行バックエンドとしてサポートしています。追加のバックエンドを使う場合は、必要な依存関係をオプションとしてインストールします。

::::{tab-set}

:::{tab-item} CUDA-Q
:sync: cuda-q

CUDA-QはLinuxとmacOS ARM64（Apple Silicon）で利用できます。利用するCUDAバージョンに合わせて、次のいずれかをインストールしてください：

```bash
pip install "qamomile[cudaq-cu12]"   # CUDA 12.x、Linux向け
pip install "qamomile[cudaq-cu13]"   # CUDA 13.x、LinuxまたはmacOS ARM64向け
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

qBraid対応のデバイスやシミュレータ上で、Qiskitの量子回路を実行できます。

```bash
pip install "qamomile[qbraid]"
```

```python
from qamomile.qbraid import QBraidExecutor
```

:::

::::
