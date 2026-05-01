# Qamomileドキュメントへようこそ

**Qamomile**（カモミール、/ˈkæməˌmiːl/）は、カモミールの花にちなんで名付けられました。カモミールは穏やかさと明瞭さの象徴として知られるハーブです。

Qamomileは量子プログラミングSDKです。型付きPython関数で量子回路を記述し、Qiskit・CUDA-Q・QURI Parts・qBraidなどのQuantum SDKで実行できます。また、シンボリックな代数的リソース推定やブラックボックス（オラクル）を含むような実行そのものができない回路のリソース推定も可能です。

> **注意** Qamomileは現在もアクティブに開発中であり、リリース間で破壊的変更が加わる可能性があります。不具合を見つけた場合は、[GitHub Issues](https://github.com/Jij-Inc/Qamomile/issues/new)でお知らせいただければ幸いです。

1. [はじめての量子カーネル](tutorial/01_your_first_quantum_kernel) — カーネルの定義・可視化・実行、アフィンルール
2. [パラメータ付きカーネル](tutorial/02_parameterized_kernels) — 構造パラメータとランタイムパラメータ、バインド/スイープパターン
3. [リソース推定](tutorial/03_resource_estimation) — シンボリックなコスト分析、ゲート内訳、スケーリング分析
4. [実行モデル](tutorial/04_execution_models) — `sample()`と`run()`、オブザーバブル、ビット順序
5. [古典フローパターン](tutorial/05_classical_flow_patterns) — ループ、スパースデータ、条件分岐
6. [再利用パターン](tutorial/06_reuse_patterns) — ヘルパーカーネル、コンポジットゲート、スタブ

## VQA

- [QAOAでMaxCutを解く](vqa/qaoa_maxcut) — QAOA回路をゼロから構築してMaxCutを解き、組み込みの`qaoa_state`と比較する
- [PCEでMaxCutを解く](vqa/pce_maxcut) — `PCEConverter(k=2)`とtanh緩和した目的関数で、20変数のMaxCutをわずか3量子ビットで解く
- [水素分子のためのVQE](vqa/vqe_for_hydrogen) — OpenFermionで分子ハミルトニアンを構築し、VQEで基底状態エネルギーを求める

## 最適化

- [QAOAによるグラフ分割](optimization/qaoa_graph_partition) — QAOAでグラフ分割問題をエンドツーエンドで解く

## 対応Quantum SDK

Qamomileは複数のQuantum SDKを実行バックエンドとしてサポートしています。Qiskitはデフォルトで含まれており、その他はオプションの追加パッケージです。

### Qiskit（デフォルト）

`pip install qamomile` に含まれています。追加のフラグは不要です。

```python
from qamomile.qiskit import QiskitTranspiler, QiskitExecutor
```

### CUDA-Q（オプション）

CUDA-QはLinuxおよびmacOS ARM64（Apple Silicon）をサポートしています。使用するCUDAバージョンに合わせてインストールしてください：

```bash
pip install "qamomile[cudaq-cu12]"   # CUDA 12.x、Linux向け
pip install "qamomile[cudaq-cu13]"   # CUDA 13.x、LinuxまたはmacOS ARM64向け
```

```python
from qamomile.cudaq import CudaqTranspiler, CudaqExecutor
```

### QURI Parts（オプション）

```bash
pip install "qamomile[quri_parts]"
```

```python
from qamomile.quri_parts import QuriPartsTranspiler, QuriPartsExecutor
```

### qBraid（オプション）

QiskitのQuantum回路をqBraid対応のデバイスやシミュレータで実行できます。

```bash
pip install "qamomile[qbraid]"
```

```python
from qamomile.qbraid import QBraidExecutor
```

## インストール

```bash
pip install qamomile
```

## クイック例

```python
import qamomile.circuit as qmc
from qamomile.qiskit import QiskitTranspiler

@qmc.qkernel
def bell_state() -> tuple[qmc.Bit, qmc.Bit]:
    q0 = qmc.qubit(name="q0")
    q1 = qmc.qubit(name="q1")
    q0 = qmc.h(q0)
    q0, q1 = qmc.cx(q0, q1)
    return qmc.measure(q0), qmc.measure(q1)

transpiler = QiskitTranspiler()
exe = transpiler.transpile(bell_state)
result = exe.sample(transpiler.executor(), shots=1000).result()

for outcome, count in result.results:
    print(f"  {outcome}: {count}")
```

## リンク

- [GitHubリポジトリ](https://github.com/Jij-Inc/Qamomile)
- [APIリファレンス](api/index.md)
