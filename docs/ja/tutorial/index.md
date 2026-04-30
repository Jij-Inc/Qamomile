---
slug: tutorial
---

# チュートリアル

Qamomileを基礎から学ぶステップバイステップガイドです。

## タグで探す

<!-- BEGIN browse-by-tag -->
**このセクション:** <a class="tag-chip" href="../tags/encoding.md">encoding</a> <a class="tag-chip" href="../tags/primitives.md">primitives</a> <a class="tag-chip" href="../tags/resource-estimation.md">resource-estimation</a> <a class="tag-chip" href="../tags/simulation.md">simulation</a>

**他のセクション:** <a class="tag-chip" href="../tags/chemistry.md">chemistry</a> <a class="tag-chip" href="../tags/error-correction.md">error-correction</a> <a class="tag-chip" href="../tags/integration.md">integration</a> <a class="tag-chip" href="../tags/optimization.md">optimization</a> <a class="tag-chip" href="../tags/variational.md">variational</a>
<!-- END browse-by-tag -->

## すべてのチュートリアル

1. [はじめての量子カーネル](01_your_first_quantum_kernel) — カーネルの定義・可視化・実行、アフィンルール
2. [パラメータ付きカーネル](02_parameterized_kernels) — 構造パラメータとランタイムパラメータ、バインド/スイープパターン
3. [リソース推定](03_resource_estimation) — シンボリックなコスト分析、ゲート内訳、スケーリング分析
4. [実行モデル](04_execution_models) — `sample()`と`run()`、オブザーバブル、ビット順序
5. [古典フローパターン](05_classical_flow_patterns) — ループ、スパースデータ、条件分岐
6. [再利用パターン](06_reuse_patterns) — ヘルパーカーネル、コンポジットゲート、スタブ
7. [エルミート行列の分解](07_hermitian_decomposition) — 密なエルミート行列からPauli和、そして時間発展回路まで
8. [コンパイルとトランスパイル](08_compilation_and_transpilation) — パイプライン段階ごとの解説、IRの語彙、バックエンドemission

具体的なアルゴリズムに進みたいときは[アルゴリズム](../algorithm/index.md)セクションへ。タグからQAOAやハミルトニアンシミュレーションなどへ直接飛べます。
