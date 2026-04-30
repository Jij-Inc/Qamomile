# Qamomileドキュメントへようこそ

**Qamomile**（カモミール、/ˈkæməˌmiːl/）は、カモミールの花にちなんで名付けられました。カモミールは穏やかさと明瞭さの象徴として知られるハーブです。

Qamomileは量子プログラミングSDKです。型付きPython関数で量子回路を記述し、Qiskit・CUDA-Q・QURI Parts・qBraidなどのQuantum SDKで実行できます。また、シンボリックな代数的リソース推定やブラックボックス（オラクル）を含むような実行そのものができない回路のリソース推定も可能です。

> **注意** Qamomileは現在もアクティブに開発中であり、リリース間で破壊的変更が加わる可能性があります。不具合を見つけた場合は、[GitHub Issues](https://github.com/Jij-Inc/Qamomile/issues/new)でお知らせいただければ幸いです。

## インストール

```bash
pip install qamomile
```

## Quick Example

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

## セクション

- [アルゴリズム](algorithm/index.md) — 変分・量子アルゴリズム(QAOA, VQEなど)をQamomileでend-to-endに動かす具体例集。
- [使い方](usage/index.md) — 個別モジュール(`BinaryModel`など)のHow-toガイド。
- [コラボレーション](collaboration/index.md) — 外部量子プラットフォームやサービス(qBraidなど)との連携。
- [リリースノート](release_notes/index.md) — バージョン別の変更履歴・主な機能追加・破壊的変更まとめ。

SDK自体の基礎(カーネル、パラメータ、実行、トランスパイル、QEC基礎)については[チュートリアル](tutorial/index.md)を参照してください。

## リンク

- [GitHubリポジトリ](https://github.com/Jij-Inc/Qamomile)
- [APIリファレンス](api/index.md)
