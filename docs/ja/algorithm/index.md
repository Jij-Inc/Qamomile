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
- [Möttönen振幅エンコーディング](mottonen_amplitude_encoding) — Gray符号Ry/Rz多重制御回転で任意の実数・複素振幅ベクトルを準備する。3つの入力モード（具体値、バインドされた`Vector[Float]`、ランタイムパラメトリックな角度）を扱う
- [多次元量子フーリエ変換](multidimensional_qft) — 多次元量子フーリエ変換の実装例の紹介、さらにグリッド数が2のべき乗でない場合に対応するための古典前処理手法を示した
