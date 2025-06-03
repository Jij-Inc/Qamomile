# ライブラリの基本的な使い方

Qamomile チュートリアルへようこそ！  
このガイドでは、量子最適化アルゴリズム向けに設計された強力な SDK である Qamomile の使い方を紹介します。  
Qamomile は数理モデルを量子回路に変換することに特化しており、古典的な最適化問題と量子計算による解法との橋渡しをします。

## 対応している量子最適化エンコーディングとアルゴリズム

- **QAOA**：量子近似最適化アルゴリズム（Quantum Approximate Optimization Algorithm）
- **QRAO**：量子ランダム近似最適化（Quantum Random Approximation Optimization）

## 対応している量子回路 SDK

- **Qiskit**
- **Quri-parts**
- **PennyLane**
- **Qutip**

## チュートリアル一覧

- [量子回路の構築](building_quantum_circuits.ipynb)：Qamomile を使って量子回路を構築する方法を学びます。
- [ハミルトニアンを代数的に記述する](algebraic_operator.ipynb)：代数モデラー `jijmodeling` を使って Qamomile のハミルトニアンを作成する方法を学びます。
- [QiskitTranspiler を使う](Using_the_QiskitTranspiler_in_Qamomile.ipynb)：Qamomile から Qiskit へトランスパイルする方法を学びます。
- [QuriPartsTranspiler を使う](Using_the_QuriPartsTranspiler_in_Qamomile.ipynb)：Qamomile から Quri-Parts へトランスパイルする方法を学びます。
- [PennyLaneTranspiler を使う](Using_the_PennyLaneTranspiler_in_Qamomile.ipynb)：Qamomile から PennyLane へトランスパイルする方法を学びます。
- [QuTiPTranspiler を使う](quantum_annealing.ipynb)：QuTiP の組み込み機能を用いて量子アニーリングを実行する方法を学びます。
