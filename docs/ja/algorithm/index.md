---
slug: algorithm
---

# アルゴリズム

Qamomileで実装した具体的な量子アルゴリズム例です。

::::{grid} 1 1 1 1

:::{card}
:header: **Suzuki–Trotter分解によるハミルトニアンシミュレーション (Rabi振動)**
:link: hamiltonian_simulation
RabiモデルでのTrotter–Suzuki積公式と収束次数の実験です。
:::

:::{card}
:header: **ハイブリッド量子ニューラルネットワーク (HQNN)**
:link: hybrid_qnn
CNN＋量子変分回路をFashion-MNISTでend-to-end学習し、パラメータシフトルールで勾配を計算します。
:::

:::{card}
:header: **Möttönen振幅エンコーディング**
:link: mottonen_amplitude_encoding
Gray符号Ry/Rz多重制御回転で任意の実数・複素振幅ベクトルを準備します。3つの入力モード(具体値、バインドされた`Vector[Float]`、ランタイムパラメトリックな角度)を扱います。
:::

:::{card}
:header: **多次元量子フーリエ変換**
:link: multidimensional_qft
任意のグリッドサイズの入力に対する多次元QFTと、2のべき乗でないグリッドに対応する古典前処理を実装します。
:::

:::{card}
:header: **Pauli Correlation Encoding (PCE)**
:link: pce_maxcut
Pauli Correlation Encodingを使い、20変数MaxCutを3量子ビットで解きます。ハードウェア効率的なアンザッツをtanh緩和された代理損失で学習し、符号丸めで離散解を復元します。
:::

:::{card}
:header: **QAOA によるグラフ分割**
:link: qaoa_graph_partition
OMMX・JijModeling・`QAOAConverter`を使ったend-to-endの最適化例です。
:::

:::{card}
:header: **QAOAでMaxCutを解く: 回路をゼロから構築する**
:link: qaoa_maxcut
QAOA回路をゼロから構築してMaxCutを解き、組み込みの`qaoa_state`と比較します。
:::

:::{card}
:header: **Quantum-enhanced Markov chain Monte Carlo**
:link: qe_mcmc
トロッター分解による時間発展を利用して、Quantum-enhanced MCMCを実装します。
:::

:::{card}
:header: **Quantum Selected Configuration Interaction (QSCI)**
:link: qsci
量子状態からビット列をサンプリングして有効ハミルトニアンを構築し、変分原理の保証付きで古典的に対角化します。
:::

:::{card}
:header: **量子誤り訂正入門**
:link: quantum_error_correction
3量子ビットbit-flip/phase-flip符号からShor 9量子ビット符号、スタビライザー形式までを扱います。
:::

:::{card}
:header: **量子位相推定（QPE）入門**
:link: qpe
4x4ユニタリに組み込みの`qpe`ヘルパーを適用し、復号された位相をサンプリングして、カウント用レジスタを大きくしたときの精度を比較します。
:::

:::{card}
:header: **量子カーネル分類**
:link: quantum_kernel_classification
量子特徴マップとカーネル法を使ってmake_circlesデータセットの分類を行います。
:::

:::{card}
:header: **スタビライザ形式論と Steane 符号**
:link: steane_code
Hamming [7,4,3]符号からのCSS構成、6スタビライザー、横断的Hadamardを扱います。
:::

:::{card}
:header: **水素分子のための変分量子固有値ソルバー（VQE）**
:link: vqe_for_hydrogen
OpenFermionで分子ハミルトニアンを構築し、VQEで基底状態エネルギーを求めます。
:::

::::
