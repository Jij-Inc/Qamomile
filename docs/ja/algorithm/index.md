---
slug: algorithm
---

# アルゴリズム

Qamomileで実装し、結果を確認する量子アルゴリズムの例です。

::::{grid} 1 1 1 1

:::{card}
:header: **Grover適応探索**
:link: grover_adaptive_search
`BinaryModel`とGroverオラクルを使い、古典側でしきい値を更新しながらポートフォリオの二値目的関数を最小化します。
:::

:::{card}
:header: **ハミルトニアンシミュレーション**
:link: hamiltonian_simulation
Suzuki積公式でRabi振動をシミュレーションし、解析解と比較します。
:::

:::{card}
:header: **ハイブリッド量子ニューラルネットワーク**
:link: hybrid_qnn
量子回路をPyTorchと組み合わせ、パラメータシフト法で求めた勾配を使って学習します。
:::

:::{card}
:header: **Möttönen振幅エンコーディング**
:link: mottonen_amplitude_encoding
古典的な振幅データから量子状態を準備し、測定確率を確認します。
:::

:::{card}
:header: **量子振幅推定によるオプション価格計算**
:link: option_pricing
資産価格の分布とペイオフを量子状態に符号化し、期待ペイオフを推定して割引後のオプション価格を求めます。
:::

:::{card}
:header: **多次元量子フーリエ変換**
:link: multidimensional_qft
量子レジスタを論理的な軸に分け、各軸に同じQFT関数を適用します。
:::

:::{card}
:header: **Pauli相関エンコーディング**
:link: pce_maxcut
4変数のMaxCut問題を2量子ビットに符号化し、推定したPauli相関から分割を復元します。
:::

:::{card}
:header: **QAOAによる制約付きグラフ分割**
:link: qaoa_graph_partition
JijModelingとOMMXで制約付き問題を定式化し、QAOAの結果から制約の充足と目的関数値を確認します。
:::

:::{card}
:header: **QAOAによるMaxCut**
:link: qaoa_maxcut
再利用可能な関数からQAOAを構成し、4頂点の閉路グラフの分割を求めます。
:::

:::{card}
:header: **QSVT固有状態フィルタリングによるグラフ分割**
:link: qsvt_eigenstate_filter_graph_partitioning
実数多項式による固有状態フィルタを構成し、成功確率を推定して、頂点数の等しいグラフ分割を求めます。
:::

:::{card}
:header: **量子回路を用いたMarkov連鎖Monte Carlo法**
:link: qe_mcmc
量子時間発展回路で古典側の候補を生成し、詳細釣り合いを確認します。
:::

:::{card}
:header: **量子選択配置間相互作用（QSCI）**
:link: qsci
試行状態をサンプリングし、得られた部分空間のハミルトニアンから変分エネルギーを求めます。
:::

:::{card}
:header: **量子特異値変換（QSVT）**
:link: qsvt
非エルミート行列をブロックエンコーディングし、2次の特異値多項式を適用して、射影後の行列と論理リソースを検証します。
:::

:::{card}
:header: **量子誤り訂正入門**
:link: quantum_error_correction
論理状態を符号化し、単一のビット反転を検出して、論理振幅を保ちながら訂正します。
:::

:::{card}
:header: **量子位相推定（QPE）**
:link: qpe
個々の操作と`qamomile.stdlib`の`phase_estimation`の両方で量子位相推定を実装し、推定精度と論理ゲート数を比較します。
:::

:::{card}
:header: **量子カーネル分類**
:link: quantum_kernel_classification
量子状態の重なりからカーネル行列を推定し、サポートベクトル分類器に使います。
:::

:::{card}
:header: **スタビライザとSteane符号**
:link: steane_code
Steane符号を構成し、シンドロームからPauli誤りを訂正します。
:::

:::{card}
:header: **水素分子のVQE**
:link: vqe_for_hydrogen
分子ハミルトニアンを構成し、2量子ビットの試行状態を最適化して基底状態エネルギーを推定します。
:::

::::
