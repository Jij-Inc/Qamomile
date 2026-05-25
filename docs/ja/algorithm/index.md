---
slug: algorithm
---

# アルゴリズム

Qamomileで実装した具体的な量子アルゴリズム例です。

## すべての記事

- [QAOAでMaxCutを解く](qaoa_maxcut) — QAOA回路をゼロから構築してMaxCutを解き、組み込みの`qaoa_state`と比較する
- [QAOAによるグラフ分割](qaoa_graph_partition) — OMMX・JijModeling・`QAOAConverter`を使ったend-to-endの最適化例
- [水素分子のためのVQE](vqe_for_hydrogen) — OpenFermionで分子ハミルトニアンを構築し、VQEで基底状態エネルギーを求める
- [Suzuki–Trotterによるハミルトニアンシミュレーション](hamiltonian_simulation) — RabiモデルでのTrotter–Suzuki積公式と収束次数の実験
- [量子誤り訂正入門](quantum_error_correction) — 3量子ビットbit-flip/phase-flip符号からShor 9量子ビット符号、スタビライザー形式まで
- [Steane [[7,1,3]] 符号](steane_code) — Hamming [7,4,3] 符号からのCSS構成、6スタビライザー、横断的Hadamard
- [Quantum Selected Configuration Interaction (QSCI)](qsci) — 量子状態からビット列をサンプリングして有効ハミルトニアンを構築し、変分原理の保証付きで古典的に対角化する
- [ハイブリッド量子ニューラルネットワーク (HQNN)](hybrid_qnn) — CNN＋量子変分回路をFashion-MNISTでend-to-end学習、パラメータシフトルールによる勾配計算
- [量子カーネル法](quantum_kernel_classification) — 量子特徴マップとカーネル法を使ってmake_circlesデータセットの分類を行う
- [Quantum-enhanced MCMCを実装する](qe_mcmc) — トロッター分解による時間発展を利用して、Quantum-enhanced MCMCを実装
- [Möttönen振幅エンコーディング](mottonen_amplitude_encoding) — Gray符号Ry/Rz多重制御回転で任意の実数・複素振幅ベクトルを準備する。3つの入力モード（具体値、バインドされた`Vector[Float]`、ランタイムパラメトリックな角度）を扱う
