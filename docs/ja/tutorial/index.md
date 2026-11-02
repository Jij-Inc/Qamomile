---
slug: tutorial
---

# チュートリアル

量子関数の定義から実行、再利用、リソース推定までを学ぶ9本のチュートリアルです。初めて使う場合は第1章から順に進めてください。

::::{grid} 1 1 1 1

:::{card}
:header: **1. はじめての量子カーネル**
:link: 01_your_first_quantum_kernel
量子関数を定義して実行し、測定結果の出現回数を読み取ります。
:::

:::{card}
:header: **2. パラメータと特殊化**
:link: 02_parameterized_kernels
`Const`で回路のサイズを決め、`Param`でコンパイル済みプログラムを再利用します。
:::

:::{card}
:header: **3. 制御ゲートと制御関数**
:link: 03_controlled_gates
量子状態を制御条件として、ゲートや再利用可能な関数を適用します。
:::

:::{card}
:header: **4. リソース推定**
:link: 04_resource_estimation
実行前に量子ビット数とゲート数を推定し、リソースモデルを比較します。
:::

:::{card}
:header: **5. 量子プログラムの実行と期待値の推定**
:link: 05_execution_models
ショットごとの結果を読み取り、サンプリングの統計とオブザーバブルの推定値を求めます。
:::

:::{card}
:header: **6. 古典制御フロー**
:link: 06_classical_flow_patterns
量子プログラム中でのループ・分岐・測定結果に基づく制御を学びます。
:::

:::{card}
:header: **7. 量子カーネルの再利用**
:link: 07_reuse_patterns
QKernelを再利用してGHZ状態を作り、カーネルを引数として渡す方法を学びます。
:::

:::{card}
:header: **8. エルミート演算子**
:link: 08_hermitian_decomposition
小さなエルミート行列をパウリ演算子の和に分解し、時間発展をシミュレーションします。
:::

:::{card}
:header: **9. コンパイルとトランスパイル**
:link: 09_compilation_and_transpilation
ターゲットに合わせてコンパイルし、異なる環境で実行する方法を学びます。
:::

::::

次は[アルゴリズム](../algorithm/index.md)で具体的な応用例を確認できます。
