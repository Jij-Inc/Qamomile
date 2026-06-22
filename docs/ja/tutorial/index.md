---
slug: tutorial
---

# チュートリアル

Qamomileを基礎から学ぶステップバイステップガイドです。

## すべてのチュートリアル

1. [はじめての量子カーネル](01_your_first_quantum_kernel) — カーネルの定義・可視化・実行、アフィンルール
2. [パラメータ付きカーネル](02_parameterized_kernels) — 構造パラメータとランタイムパラメータ、バインド/スイープパターン
3. [Vectorのスライシング](03_vector_slicing) — `VectorView`、スライス代入、ネストしたスライス、ヘルパーカーネルへの引き渡し
4. [制御ゲート](04_controlled_gates) — `qmc.control`によるビルトインゲートやサブカーネルの制御、concrete/symbolicの制御数、エラーカタログ
5. [リソース推定](05_resource_estimation) — シンボリックなコスト分析、ゲート内訳、スケーリング分析
6. [実行モデル](06_execution_models) — `sample()`と`run()`、オブザーバブル、ビット順序
7. [古典フローパターン](07_classical_flow_patterns) — ループ、スパースデータ、条件分岐
8. [再利用パターン](08_reuse_patterns) — ヘルパーカーネル、コンポジットゲート、スタブ
9. [エルミート行列の分解](09_hermitian_decomposition) — 密なエルミート行列からPauli和、そして時間発展回路まで
10. [コンパイルとトランスパイル](10_compilation_and_transpilation) — パイプライン段階ごとの解説、IRの語彙、バックエンドemission

具体的なアルゴリズムに進みたいときは[アルゴリズム](../algorithm/index.md)セクションへ。タグからQAOAやハミルトニアンシミュレーションなどへ直接飛べます。
