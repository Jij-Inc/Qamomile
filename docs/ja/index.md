# Qamomileドキュメントへようこそ

**Qamomile**（カモミール、/ˈkæməˌmiːl/）は、カモミールの花にちなんで名付けられました。カモミールは穏やかさと明瞭さの象徴として知られるハーブです。

:::{note}
Qamomileは現在も活発に開発中であり、リリース間で破壊的変更が加わる可能性があります。
不具合を見つけた場合は、[GitHub Issues](https://github.com/Jij-Inc/Qamomile/issues/new)でお知らせいただけると助かります。
:::

## Qamomileとは？

Qamomileは、量子プログラムを型付きのPython関数として記述できる量子プログラミングSDK（開発キット）です。同じソースコードでリソース推定と、対応する量子SDKでの実行を扱えます。

## 開発の動機

量子ハードウェアの進歩に伴い、実行できる問題の規模や、ある規模の問題に対してアルゴリズムがいつ実用的になるかを評価することが重要になっています。そのため、必要な量子ビット数やゲート数を見積もるリソース推定も、実機での実行と同じくらい大切です。

また、一度書いたプログラムをさまざまな環境で動かす「write once, run anywhere」を目指しています。量子SDKごとの強みを活かしながら、移植のための書き直しを減らします。さらに、高水準の言語機能や再利用できるアルゴリズムを通じて、低水準の回路操作にとどまらない量子プログラムの記述を可能にし、アプリケーション開発に必要な専門知識の負担を減らします。

既存のSDKもこうした課題の一部に取り組んでいます。Qamomileは、リソース推定、移植性、高水準のプログラミングをひとつのSDKで扱うために生まれました。

## 主な機能

- **型による安全性。** affine type system（アフィン型システム）により、各量子リソースを表す値の使用を高々1回に制限します。同じ値の不正な再利用を実行前に検出します。
- **組み込みモジュール。** よく使う操作パターンや代表的なアルゴリズムを提供し、複雑な処理や定型コードを一から書く手間を減らします。
- **アルゴリズム単位のリソース推定。** 量子ビット数やゲート数を記号式で見積もり、実装の評価やボトルネックの特定に使えます。必要なリソースを指定したブラックボックス操作（オラクル）により、すべての操作を実装する前でもプログラム全体を見積もれます。実行にはオラクルの実装が必要です。
- **量子プログラムの移植性。** トランスパイラ（プログラムの変換器）がQiskit、CUDA-Q、HUGRなどの対応形式に変換します。変換したプログラムは、対応するエグゼキュータ（実行を担う仕組み）を通じてQamomileから実行できます。
- **推定と実行に共通のソースコード。** 同じQamomileプログラムをリソース推定と実行の両方に使えるため、別々の実装を管理する必要がありません。
- **最適化モジュール。** ソルバー（最適化問題を解くソフトウェア）に依存しないデータ形式OMMXを使い、量子最適化アルゴリズムを実行できます。OMMXが提供する古典・量子ソルバー向けアダプタ（接続機能）を使うことで、Qamomileのアルゴリズムとの結果比較もできます。

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
