# Qamomileドキュメントへようこそ

**Qamomile**（カモミール、/ˈkæməˌmiːl/）は、カモミールの花にちなんで名付けられました。カモミールは穏やかさと明瞭さの象徴として知られるハーブです。

Qamomileは量子プログラミングSDKです。型付きPython関数で量子回路を記述し、Qiskit・CUDA-Q・QURI Parts・qBraidなどのQuantum SDKで実行できます。また、シンボリックな代数的リソース推定やブラックボックス（オラクル）を含むような実行そのものができない回路のリソース推定も可能です。

> **注意** Qamomileは現在もアクティブに開発中であり、リリース間で破壊的変更が加わる可能性があります。不具合を見つけた場合は、[GitHub Issues](https://github.com/Jij-Inc/Qamomile/issues/new)でお知らせいただければ幸いです。

## インストール

```bash
pip install qamomile
```

## 対応Quantum SDK

Qamomileは複数のQuantum SDKを実行バックエンドとしてサポートしています。Qiskitはデフォルトで含まれており、その他はオプションの追加パッケージです。Qiskit・CUDA-Q・QURI Partsの3つはそれぞれ`Transpiler` + `Executor`の対を提供しているので、下のimportを差し替えるだけで切り替えられます。qBraidだけは構造が異なり、`QiskitTranspiler`で生成したQiskit回路をqBraidの提供するデバイス/シミュレータ上で実行するexecutor-only adapterなので、別セクションで紹介します。

::::{tab-set}
:::{tab-item} Qiskit（デフォルト）
:sync: qiskit

`pip install qamomile` に含まれています。追加のフラグは不要です。

```python
from qamomile.qiskit import QiskitTranspiler, QiskitExecutor
```
:::

:::{tab-item} CUDA-Q
:sync: cudaq

CUDA-QはLinuxおよびmacOS ARM64（Apple Silicon）をサポートしています。使用するCUDAバージョンに合わせてインストールしてください：

```bash
pip install "qamomile[cudaq-cu12]"   # CUDA 12.x、Linux向け
pip install "qamomile[cudaq-cu13]"   # CUDA 13.x、LinuxまたはmacOS ARM64向け
```

```python
from qamomile.cudaq import CudaqTranspiler, CudaqExecutor
```
:::

:::{tab-item} QURI Parts
:sync: quri_parts

```bash
pip install "qamomile[quri_parts]"
```

```python
from qamomile.quri_parts import QuriPartsTranspiler, QuriPartsExecutor
```
:::
::::

### qBraid（オプション、executor-only）

qBraidはexecutor-only adapterで、`QBraidTranspiler`は存在しません。まず`QiskitTranspiler`でQiskit回路にトランスパイルしてから、`QBraidExecutor`経由でqBraid対応のデバイス/シミュレータに投げる、という流れになります：

```bash
pip install "qamomile[qbraid]"
```

```python
from qamomile.qiskit import QiskitTranspiler
from qamomile.qbraid import QBraidExecutor
```

詳しい流れ（qBraid APIキーの渡し方含む）は[integration/qbraid_executor](integration/qbraid_executor.ipynb)を参照してください。

## セクション

- [アルゴリズム](algorithm/index.md) — 変分・量子アルゴリズム(QAOA, VQEなど)をQamomileでend-to-endに動かす具体例集。
- [使い方](usage/index.md) — 個別モジュール(`BinaryModel`など)のHow-toガイド。
- [インテグレーション](integration/index.md) — Qamomileと外部ライブラリ・量子プラットフォーム(qBraidなど)を組み合わせて使うときのノート。
- [リリースノート](release_notes/index.md) — バージョン別の変更履歴・主な機能追加・破壊的変更まとめ。

SDK自体の基礎(カーネル、パラメータ、実行、トランスパイル、QEC基礎)については[チュートリアル](tutorial/index.md)を参照してください。

## リンク

- [GitHubリポジトリ](https://github.com/Jij-Inc/Qamomile)
- [APIリファレンス](api/index.md)
