# Qamomileドキュメントへようこそ

**Qamomile**（カモミール、/ˈkæməˌmiːl/）は、カモミールの花にちなんで名付けられました。カモミールは穏やかさと明瞭さの象徴として知られるハーブです。

:::{note}
Qamomileは現在も活発に開発中であり、リリース間で破壊的変更が加わる可能性があります。
不具合を見つけた場合は、[GitHub Issues](https://github.com/Jij-Inc/Qamomile/issues/new)でお知らせいただけると助かります。
:::

## Qamomileとは？

Qamomileは、Pythonのソースコードからコンパイルする、型安全な関数型量子プログラミング言語です。Pythonに埋め込まれたドメイン固有言語（embedded domain-specific language、eDSL）として、同じソースコードでリソース推定と、対応する量子SDKでの実行を扱えます。

## 開発の動機

::::{grid} 1 1 3 3

:::{card}
:header: リソース推定

量子ハードウェアの進歩に伴い、実行できる問題の規模や、ある規模の問題に対してアルゴリズムがいつ実用的になるかを評価することが重要になっています。そのため、必要な量子ビット数やゲート数を見積もるリソース推定も、実機での実行と同じくらい大切です。
:::

:::{card}
:header: プログラムの移植性

一度書いたプログラムをさまざまな環境で動かす「write once, run anywhere」を目指しています。量子SDKごとの強みを活かしながら、移植のための書き直しを減らします。
:::

:::{card}
:header: 高水準のプログラミング

高水準の言語機能や再利用できるアルゴリズムを通じて、低水準の回路操作にとどまらない量子プログラムの記述を可能にし、アプリケーション開発に必要な専門知識の負担を減らします。
:::

::::

## 特徴

::::{grid} 1 1 3 3
:class: qamomile-feature-cards

:::{card}
:header: 型による安全性

線形型システム（各量子状態の値をちょうど1回使う規則）により、同じ値の不正な再利用や未使用の量子状態を実行前に検出します。
:::

:::{card}
:header: 組み込みモジュール

よく使う操作パターンや代表的なアルゴリズムを提供し、複雑な処理や定型コードを一から書く手間を減らします。
:::

:::{card}
:header: アルゴリズム単位のリソース推定

量子ビット数やゲート数を記号式で見積もり、実装の評価やボトルネックの特定に使えます。必要なリソースを指定したブラックボックス操作（オラクル）により、すべての操作を実装する前でもプログラム全体を見積もれます。実行にはオラクルの実装が必要です。
:::

:::{card}
:header: 量子プログラムの移植性

`qm.compile`がQiskit、CUDA-Q、HUGRなどの対応形式に変換します。`qm.run`を使うと、元のプログラムや変換済みプログラムを、対応する実行先で実行できます。
:::

:::{card}
:header: 推定と実行に共通のソースコード

同じQamomileプログラムをリソース推定と実行の両方に使えるため、別々の実装を管理する必要がありません。
:::

:::{card}
:header: 最適化モジュール

ソルバー（最適化問題を解くソフトウェア）に依存しないデータ形式OMMXを使い、量子最適化アルゴリズムを実行できます。OMMXが提供する古典・量子ソルバー向けアダプタ（接続機能）を使うことで、Qamomileのアルゴリズムとの結果比較もできます。
:::

::::

---

## クイックスタート

Qamomileはpipからインストールできます。ディストリビューション名は`qamomile`で、新しい言語モジュールは`qamomile`です。

```bash
pip install qamomile
```

以下では、同じBellプログラムをEstimateして実行します。詳細は[はじめてのQamomile関数](tutorial/01_your_first_quantum_kernel.ipynb)を参照してください。

```python
import qamomile as qm


@qm.qkernel
def bell() -> qm.Bits[2]:
    state = qm.qubits(2)
    state = qm.h(state, target=0)
    state = qm.cx(state, control=0, target=1)
    return qm.measure(state)


study = qm.resources.logical(bell)
print(study.metric(qm.resources.LOGICAL_GATES))

record = qm.run(bell, on=qm.Reference(), shots=16, seed=7)
print(record.result.counts)
```

---

## はじめに

::::{grid} 1 2 2 2

:::{card}
:header: **インストール**
:link: installation_guide.md
Qamomileをインストールし、Qiskit、Cirq、QURI Parts、CUDA-Q、Braket、HUGRは必要な場合だけ追加します。
:::

:::{card}
:header: **チュートリアル**
:link: tutorial/index.md
型付き関数、関数型の量子値、パラメータ、リソース、実行、target向けトランスパイルを順に学べます。
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
Qiskit、Cirq、QURI Parts、CUDA-Q、Amazon Braket、HUGR向けのガイドは同じQamomileライフサイクルを利用します。
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
公開`qamomile`パッケージのAPIリファレンスです。
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
