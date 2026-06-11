---
slug: tutorial
---

# チュートリアル

Qamomileを基礎から学ぶステップバイステップガイドです。

::::{grid} 1 1 1 1

:::{card}
:header: **1. はじめての量子カーネル**
:link: 01_your_first_quantum_kernel
量子カーネルの定義・可視化・実行、アフィンルールを扱います。
:::

:::{card}
:header: **2. パラメータ付き量子カーネル**
:link: 02_parameterized_kernels
構造パラメータとランタイムパラメータ、バインド/スイープパターンを扱います。
:::

:::{card}
:header: **3. Vectorのスライシング**
:link: 03_vector_slicing
`VectorView`、スライス代入、ネストしたスライス、ヘルパー量子カーネルへの引き渡しを扱います。
:::

:::{card}
:header: **4. 制御ゲート**
:link: 04_controlled_gates
`qmc.control`によるビルトインゲートや量子カーネルの制御、concrete/symbolicの制御数、エラーカタログを扱います。
:::

:::{card}
:header: **5. リソース推定**
:link: 05_resource_estimation
シンボリックなコスト分析、ゲート内訳、スケーリング分析を扱います。
:::

:::{card}
:header: **6. 実行モデル**
:link: 06_execution_models
`sample()`と`run()`、オブザーバブル、ビット順序を扱います。
:::

:::{card}
:header: **7. 古典フローパターン**
:link: 07_classical_flow_patterns
ループ、スパースデータ、条件分岐を扱います。
:::

:::{card}
:header: **8. 再利用パターン**
:link: 08_reuse_patterns
ヘルパー量子カーネル、コンポジットゲート、スタブを扱います。
:::

:::{card}
:header: **9. エルミート行列の分解**
:link: 09_hermitian_decomposition
密なエルミート行列からPauli和、そして時間発展回路までを扱います。
:::

:::{card}
:header: **10. コンパイルとトランスパイル**
:link: 10_compilation_and_transpilation
パイプライン段階ごとの解説、IRの語彙、バックエンドemissionを扱います。
:::

::::

具体的なアルゴリズムに進みたいときは[アルゴリズム](../algorithm/index.md)セクションへ。タグからQAOAやハミルトニアンシミュレーションなどへ直接飛べます。
