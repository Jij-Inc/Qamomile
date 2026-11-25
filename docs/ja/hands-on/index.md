# ハンズオン

量子テレポーテーションから始め、リソース推定、回路の実行、位相推定、グラフ彩色を順に学びます。各ノートブックは単独で実行できます。

::::{grid} 1 1 1 1

:::{card}
:header: **量子テレポーテーション**
:link: 01_quantum_teleportation
ベル対を準備し、Aliceの測定結果に応じてBobの量子ビットを補正します。Bobが入力状態を復元できることを確認します。
:::

:::{card}
:header: **リソース推定**
:link: 02_resource_estimation
GHZ状態を準備する2つの方法を比較し、オラクルを呼び出す回路のコストを見積もります。
:::

:::{card}
:header: **実行の流れ**
:link: 03_execution_workflow
パラメータ付き回路を実行し、古典最適化ソルバーでエネルギーの最小値を探す間に回路を再利用します。
:::

:::{card}
:header: **量子位相推定とリソース推定**
:link: 04_qpe_and_resource_estimation
量子位相推定（QPE）を手作業で組み立て、標準ライブラリ版と比較して必要なリソースを数えます。
:::

:::{card}
:header: **グラフ彩色による量子最適化**
:link: 05_quantum_optimization_graph_coloring
グラフ彩色問題をモデル化し、QAOAを実行して、2つの古典ソルバーと結果を比較します。
:::

::::
