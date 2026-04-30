---
slug: tutorial
---

# チュートリアル

Qamomileを基礎から学ぶステップバイステップガイドです。

## タグで探す

<!-- BEGIN browse-by-tag -->
**このセクション:** <a class="tag-chip" href="../tags/algorithm.md">algorithm</a> <a class="tag-chip" href="../tags/hamiltonian-simulation.md">hamiltonian-simulation</a> <a class="tag-chip" href="../tags/pauli-decomposition.md">pauli-decomposition</a> <a class="tag-chip" href="../tags/qec.md">qec</a> <a class="tag-chip" href="../tags/resource-estimation.md">resource-estimation</a> <a class="tag-chip" href="../tags/trotterization.md">trotterization</a> <a class="tag-chip" href="../tags/tutorial.md">tutorial</a>

**他のセクション:** <a class="tag-chip" href="../tags/binary-model.md">binary-model</a> <a class="tag-chip" href="../tags/collaboration.md">collaboration</a> <a class="tag-chip" href="../tags/optimization.md">optimization</a> <a class="tag-chip" href="../tags/qamomile-optimization.md">qamomile-optimization</a> <a class="tag-chip" href="../tags/qaoa.md">qaoa</a> <a class="tag-chip" href="../tags/qbraid.md">qbraid</a> <a class="tag-chip" href="../tags/usage.md">usage</a> <a class="tag-chip" href="../tags/variational.md">variational</a> <a class="tag-chip" href="../tags/vqe.md">vqe</a>
<!-- END browse-by-tag -->

## すべてのチュートリアル

1. [はじめての量子カーネル](01_your_first_quantum_kernel) — カーネルの定義・可視化・実行、アフィンルール
2. [パラメータ付きカーネル](02_parameterized_kernels) — 構造パラメータとランタイムパラメータ、バインド/スイープパターン
3. [リソース推定](03_resource_estimation) — シンボリックなコスト分析、ゲート内訳、スケーリング分析
4. [実行モデル](04_execution_models) — `sample()`と`run()`、オブザーバブル、ビット順序
5. [古典フローパターン](05_classical_flow_patterns) — ループ、スパースデータ、条件分岐
6. [再利用パターン](06_reuse_patterns) — ヘルパーカーネル、コンポジットゲート、スタブ
7. [ハミルトニアンシミュレーション](07_hamiltonian_simulation) — RabiモデルでのSuzuki–Trotter、収束次数
8. [エルミート行列の分解](08_hermitian_decomposition) — 密なエルミート行列からPauli和、そして時間発展回路まで
9. [コンパイルとトランスパイル](09_compilation_and_transpilation) — パイプライン段階ごとの解説、IRの語彙、バックエンドemission
10. [量子誤り訂正入門](10_quantum_error_correction) — 3量子ビットbit-flip/phase-flip符号からShor 9量子ビット符号、スタビライザー形式まで
11. [量子誤り訂正(2): Steane 符号と CSS 構成](11_steane_code) — Hamming [7,4,3] 符号からの CSS 構成、6 スタビライザー、横断的 Hadamard

具体的なアルゴリズムに進みたいときは[アルゴリズム](../algorithm/index.md)セクションへ。タグからQAOAやハミルトニアンシミュレーションなどへ直接飛べます。
