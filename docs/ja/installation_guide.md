# インストール

QamomileにはPython 3.11以降が必要です。公開ディストリビューションをpipでインストールします。

```bash
pip install qamomile
```

ディストリビューション名は`qamomile`ですが、Qamomileの言語モジュールは次のようにimportします。

```python
import qamomile as qm
```

ディストリビューションには互換性のあるRustコンパイラが含まれています。組み込みの`qm.Reference()` runtime、コンパイラ、Symbolic Resource Estimationには、オプションの量子SDKは必要ありません。

## オプションの量子SDK連携

必要なtarget連携だけをインストールします。endpointを変更しても、Qamomileプログラムのソース意味論と型付き結果は変わりません。

::::{tab-set}

:::{tab-item} Qiskit
:sync: qiskit

```bash
pip install "qamomile[qiskit]"
```

```python
from qamomile.qiskit import Aer, Qiskit
```

`Qiskit()`はQPY artifactを作成します。`Aer()`はartifactを作成してローカルで実行します。

:::

:::{tab-item} QURI Parts
:sync: quri-parts

```bash
pip install "qamomile[quri_parts]"
```

```python
from qamomile.quri_parts import Qulacs, QuriParts
```

`QuriParts()`はportableな論理回路artifactを作成します。`Qulacs()`はそのartifactをローカルで実行します。

:::

:::{tab-item} Amazon Braket
:sync: braket

```bash
pip install "qamomile[braket]"
```

```python
from qamomile.braket import Braket, BraketDevice, BraketLocal
```

`Braket()`はOpenQASM 3を作成します。`BraketLocal()`はローカルシミュレータを使い、`BraketDevice`はmanaged device用のendpointです。

:::

:::{tab-item} CUDA-Q
:sync: cuda-q

CUDA-QはLinuxとmacOS ARM64で利用できます。platformとCUDAバージョンに合うextraをインストールします。

```bash
pip install "qamomile[cudaq-cu12]"  # CUDA 12.x on Linux
pip install "qamomile[cudaq-cu13]"  # CUDA 13.x on Linux or macOS ARM64
```

```python
from qamomile.cudaq import Cudaq, CudaqLocal
```

:::

:::{tab-item} HUGR、Selene、Helios
:sync: hugr

```bash
pip install "qamomile[hugr]"
```

```python
from qamomile.hugr import Helios, Hugr, Selene
```

この連携はstructured QCoreをHUGRへ直接loweringします。`Selene()`はQuestを
使ってpackageをローカルでbuild・実行し、`Helios`は同じresult記録付きHUGR
profileをNexus経由でsubmitします。

:::

:::{tab-item} CirqとGoogle QVM
:sync: cirq

```bash
pip install "qamomile[cirq]"
```

```python
from qamomile.cirq import Cirq, CirqSimulator, DepolarizingNoise, google_qvm
```

`Cirq()`はportableなCirq JSONを作成します。`CirqSimulator()`は呼び出し元のprocessで
実行し、必要なら`DepolarizingNoise(p)`を指定できます。`google_qvm()`はlocal
Google virtual processor向けの自動配置とroutingを追加します。

:::

::::

## ソースcheckout

ソースからビルドするには、Git、Python 3.11以降、`rustup`、ネイティブリンカーが必要です。macOSではXcode Command Line Tools、LinuxではC/C++ビルドツール、WindowsではMSVC C++ビルドツールを使用します。`rustup`は`rust-toolchain.toml`で固定されたRustバージョンを選択します。

リポジトリ開発では、`uv`をインストールし、リポジトリのルートディレクトリで次のコマンドを実行します。固定された依存関係で環境を構築し、Qamomileと同梱のRustコンパイラをビルドします。

```bash
uv sync --frozen --group dev
```

pipを使う場合も、同じビルド要件を満たしたうえでリポジトリのルートディレクトリからインストールします。次のコマンドでQamomileと同梱のRustコンパイラをまとめてインストールできます。

```bash
pip install -e .
```
