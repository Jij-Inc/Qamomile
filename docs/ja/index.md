# Qamomileドキュメントへようこそ

**Qamomile**（カモミール、/ˈkæməˌmiːl/）は、カモミールの花にちなんで名付けられました。カモミールは穏やかさと明瞭さの象徴として知られるハーブです。

:::{note}
Qamomileは現在も活発に開発中であり、リリース間で破壊的変更が加わる可能性があります。
不具合を見つけた場合は、[GitHub Issues](https://github.com/Jij-Inc/Qamomile/issues/new)でお知らせいただけると助かります。
:::

## Qamomileとは？

Qamomileは、量子回路を型付きのPython関数として記述できる量子プログラミングSDKです。作成した量子回路は、Qiskit、CUDA-Q、QURI Parts、qBraidなどの量子SDKで実行できます。また、シンボリックな代数的リソース推定やブラックボックス（オラクル）を含むような直接実行できない回路のリソース推定も可能です。

::::{grid} 1 1 3 3

:::{card}
**Pythonによる量子回路記述**

型付きのPython関数で量子回路を表現し、通常のコードのように可視化・確認・再利用できます。
:::

:::{card}
**多様な量子SDK対応**

Qiskit、CUDA-Q、QURI Parts、qBraidなどへ、同じ量子回路をそのまま展開できます。
:::

:::{card}
**数理最適化との連携**

QUBOやIsingモデルを量子アルゴリズムへつなぎ、最適化問題の実装と実行を一貫して扱えます。
:::

::::

---

## クイックスタート

Qamomileはpipからインストールすることができます。

```bash
pip install "qamomile[qiskit,visualization]"
```

以下は、Qamomileによる量子アルゴリズムの実行例です。詳細は、[はじめての量子カーネル](tutorial/01_your_first_quantum_kernel.ipynb)を参照してください。

```python
import math

import qamomile.circuit as qmc
from qamomile.qiskit import QiskitTranspiler


@qmc.qkernel
def biased_coin(theta: qmc.Float) -> qmc.Bit:
    q = qmc.qubit(name="q")
    q = qmc.ry(q, theta)
    return qmc.measure(q)


# 実行前に量子カーネルを可視化し、リソースを見積もる
biased_coin.draw(theta=0.6)
est = biased_coin.estimate_resources()
print("qubits:", est.qubits)
print("total gates:", est.gates.total)

# thetaを実行時に指定できるパラメータとして残してトランスパイル
transpiler = QiskitTranspiler()
exe = transpiler.transpile(biased_coin, parameters=["theta"])

# thetaの値を指定して実行
result = exe.sample(
    transpiler.executor(),
    shots=256,
    bindings={"theta": math.pi / 4},
).result()

print(result.results)
```

---

## はじめに

::::{grid} 1 2 2 2

:::{card}
:header: **インストール**
:link: installation_guide.md
Qamomile本体に加えて、CUDA-Q、QURI Parts、qBraidなどの量子SDK連携を必要に応じてインストールします。
:::

:::{card}
:header: **チュートリアル**
:link: tutorial/index.md
量子カーネル、パラメータ、実行、トランスパイルなど、Qamomileの基本的な使い方を順に学べます。
:::

::::

---

## Qamomileを使う

::::{grid} 1 2 2 2

:::{card}
:header: **アルゴリズム**
:link: algorithm/index.md
QAOAやVQEなどの量子アルゴリズムを、Qamomileで実装して実行するための実践的なガイドです。
:::

:::{card}
:header: **使い方**
:link: usage/index.md
`BinaryModel`など、個別モジュールの使い方を目的別に確認できます。
:::

:::{card}
:header: **インテグレーション**
:link: integration/index.md
qBraidなどの量子プラットフォームや外部ライブラリと、Qamomileを組み合わせて使う方法を紹介します。
:::

::::

---

## リファレンス

::::{grid} 1 2 2 2

:::{card}
:header: **リリースノート**
:link: release_notes/index.md
バージョン別の変更履歴・主な機能追加・破壊的変更をまとめています。
:::

:::{card}
:header: **APIリファレンス**
:link: api/index.md
Qamomileが提供するAPIの使い方、引数、戻り値を確認できます。
:::

::::

---

## 引用

研究でQamomileを使用する場合は、以下を引用してください。

```bibtex
@INPROCEEDINGS{11249901,
  author={Huang, Wei-Hao and Matsuyama, Hiromichi and Tam, Wai-Hong and Sato, Keisuke and Yamashiro, Yu},
  booktitle={2025 IEEE International Conference on Quantum Computing and Engineering (QCE)},
  title={Qamomile: A Cross-SDK Bridge for Quantum Optimization},
  year={2025},
  volume={02},
  pages={516-517},
  doi={10.1109/QCE65121.2025.10423}
}
```

---

## リンク

- [GitHubリポジトリ](https://github.com/Jij-Inc/Qamomile)
