# Qamomileドキュメントへようこそ

**Qamomile**（カモミール、/ˈkæməˌmiːl/）は、カモミールの花にちなんで名付けられました。カモミールは穏やかさと明瞭さの象徴として知られるハーブです。

Qamomileは量子プログラミングSDKです。型付きPython関数で量子回路を記述し、Qiskit・CUDA-Q・QURI Parts・qBraidなどのQuantum SDKで実行できます。また、シンボリックな代数的リソース推定やブラックボックス（オラクル）を含むような実行そのものができない回路のリソース推定も可能です。

:::{note}
Qamomileは現在も活発に開発中であり、リリース間で破壊的変更が加わる可能性があります。
不具合を見つけた場合は、[GitHub Issues](https://github.com/Jij-Inc/Qamomile/issues/new)でお知らせいただけると助かります。
:::

---

## インストール

Qamomileは、pipなどの標準的なパッケージマネージャでインストールできます。

```bash
pip install qamomile
```

Qiskitはデフォルトの実行バックエンドとして含まれています。

```python
from qamomile.qiskit import QiskitTranspiler, QiskitExecutor
```

### 対応Quantum SDK

Qamomileは複数のQuantum SDKを実行バックエンドとしてサポートしています。追加のバックエンドはoptional extrasとしてインストールします。

::::{tab-set}

:::{tab-item} CUDA-Q
:sync: cuda-q

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

QiskitのQuantum回路をqBraid対応のデバイスやシミュレータで実行できます。

```bash
pip install "qamomile[qbraid]"
```

```python
from qamomile.qbraid import QBraidExecutor
```

:::

::::

---

## セクション

::::{grid} 1 2 2 2

:::{card}
:header: **チュートリアル**
:link: tutorial/index.md
SDK自体の基礎(量子カーネル、パラメータ、実行、トランスパイル、QEC基礎)を扱います。
:::

:::{card}
:header: **アルゴリズム**
:link: algorithm/index.md
変分・量子アルゴリズム(QAOA、VQEなど)をQamomileでend-to-endに動かす具体例集です。
:::

:::{card}
:header: **使い方**
:link: usage/index.md
個別モジュール(`BinaryModel`など)のHow-toガイドです。
:::

:::{card}
:header: **インテグレーション**
:link: integration/index.md
Qamomileと外部ライブラリ・量子プラットフォーム(qBraidなど)を組み合わせて使うときのノートです。
:::

:::{card}
:header: **リリースノート**
:link: release_notes/index.md
バージョン別の変更履歴・主な機能追加・破壊的変更をまとめています。
:::

:::{card}
:header: **APIリファレンス**
:link: api/index.md
APIリファレンスです。
:::

::::

---

## リンク

- [GitHubリポジトリ](https://github.com/Jij-Inc/Qamomile)
