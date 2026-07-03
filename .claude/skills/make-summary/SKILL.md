---
name: make-summary
description: Create a summary markdown file for the current branch that explains its work relative to the main branch. This skill uses a separate subagent or AI model to validate the summary. Keep an eye on the usage limit.
model: opus
---

# Branch Summary

main と現在の作業 worktree branch の差分を、コードを読んで要約する。レビューワーやジュニアエンジニアが、この会話履歴を知らない状態で読んでも内容を理解できるレベルに仕上げる。

## When To Use

- 作業 branch でひと段落ついたタイミング。
- PR を書く前、レビューに出す前。
- ユーザーが「サマリを作って」「ブランチサマリを書いて」「branch-summary」「make-summary」と依頼した時。

## Output Rules

- 出力先は現在の作業 worktree root にする。ユーザーから別指定がある場合だけ従う。
- ファイル名は `<branch-name>-summary.md` にする。必要なら branch 名をファイル名として安全な形に sanitize する。
- 同名ファイルが既にあれば確認なく上書きする。
- summary file は commit しない。untracked のまま残す。
- `/tmp` など消えやすい場所を自分で出力先に選ばない。current worktree root 自体が一時領域にある場合は、worktree root を優先する。
- 完了時は絶対パスと、commit していないことだけを伝える。

## Writing Rules

- 箇条書き中心にしない。添付例のように、各セクションを説明文の段落で書く。
- 段落内に手動 soft line break を入れない。1 段落 = 1 行、段落区切りは blank line にする。
- 会話・レビュー履歴を書かない。summary は branch の現在差分の説明に限定する。
- 初出の Qamomile 固有用語や専門用語は、短く説明してから使う。
- ジュニアエンジニアに口頭で説明するつもりで、なぜ問題なのか、なぜ対処が必要なのかを省略しない。
- コード参照は `path/to/file.py:123` や `path/to/file.py:123-145` のように具体的に書く。
- テスト一覧、diff stats、commit history、検証ログ、作業経緯は独立セクションにしない。必要な事実だけを該当セクションの説明に吸収する。

## Summary Structure

次の 5 セクションだけを、この順で書く。`0. Glossary`、検証結果、TODO、会話履歴などの追加セクションは作らない。

```md
1. 問題の概要

2. フロントエンド(ユーザーが書くコードレベル)での変更

3. バックエンド(IR 等全体的)での変更

4. 採用しなかった代替案と、今回の方法を選んだ理由

5. 既知の限界
```

`1. 問題の概要` では、bugfix の場合は main で何が起きていたか、なぜそうなるか、branch でどう変わるかを区別して書く。機能追加の場合は、何ができるようになり、なぜ必要なのかを書く。

`2. フロントエンド(ユーザーが書くコードレベル)での変更` では、ユーザーが書く qkernel、受け取るエラーメッセージ、触る API の振る舞いなど、利用者視点の変更を書く。コード例を 1 つ以上載せる。内部実装詳細はここに書かない。

`3. バックエンド(IR 等全体的)での変更` では、compiler / IR / transpiler / backend 側の変更を書く。新しい IR op、dataclass、pass、helper がある場合は、役割と `file:line` を示す。Qamomile 用語が初めて出る場合は、このセクション内でも短く説明する。

`4. 採用しなかった代替案と、今回の方法を選んだ理由` では、設計レベルで複数案があったものについて、trade-off と採用/不採用理由を書く。該当しなければ「該当なし」と書く。このセクションだけは、設計判断の根拠として外部レビューや会話で出た代替案を最終形に整理して含めてよいが、作業経緯の物語にはしない。

`5. 既知の限界` では、merge 後にも残るギャップを書く。false negative / false positive、未対応 AST 形、backend 差など、実コードで踏み得るものを `When:` / `Why:` / `Future fix:` の形で説明する。複数ある場合も箇条書きにせず、限界ごとに段落を分ける。無ければ「該当なし」と書く。

## Workflow

### Step 1. Grasp The Context

worktree root を確定し、差分の規模を把握する。

```bash
pwd
git status -sb
git log origin/main..HEAD --oneline
git diff origin/main...HEAD --stat
```

### Step 2. Read The Diff

変更ファイルを実際に読む。大きく変わったファイル、新規ファイル、新規クラス/関数、削除されたシンボルを把握する。テスト追加も読むが、summary にテスト一覧を作らない。期待される振る舞いとして各セクションに吸収する。bugfix なら、可能な範囲で main の不具合と branch の改善を具体例で説明できるようにする。

### Step 3. Write The Draft

worktree root に `<branch-name>-summary.md` を作成または上書きする。セクション 2 と 3 を混ぜない。セクション 4 は作業経緯ではなく最終的な設計選択として書く。セクション 5 は抽象的な TODO ではなく、残る条件と将来対応を書く。

### Step 4. Check Discrepancies With Another AI

原則として `claude` skill を使い、summary と実コードの食い違いを敵対的に確認してもらう。使用量には注意する。`claude` が利用できない場合や、外部送信の承認待ちで止まる場合は、記憶を共有しない別 subagent / another available AI に同じ確認を依頼する。どちらも使えない場合だけ、draft の絶対パスと blocked reason をユーザーに伝えて止まる。

プロンプトには、summary の絶対パス、主な実装ファイルの絶対パス、summary が参照している主要シンボルや行範囲、「実コードと一致しない記述があれば箇条書きで指摘し、無ければ `齟齬なし` と答えて」という依頼、コード編集・ファイル作成/削除・コマンド実行を禁止する制約を含める。

Claude の指摘が出た場合は、指摘を実コードで再確認し、summary 側を修正し、行番号修正で他の参照がずれていないか確認してから、再度 Claude に確認を依頼する。指摘がゼロになるまで反復する。ただし、指摘が設計判断の見直しを要求する内容なら、summary だけを直さずユーザーに相談する。乖離チェックの反復ログを毎回ユーザーに中継しない。

### Step 5. Report Completion

Step 4 が `齟齬なし` になったら、summary の絶対パスと commit していないことだけを伝えて完了する。

## Do Not

- summary を commit しない。
- `/tmp` に置かない。
- 5 セクション以外を追加しない。
- 段落内に手動 soft line break を入れない。
- 会話履歴や作業物語を書かない。
- セクション 4 以外で代替案を議論しない。
- 乖離チェックの反復ログを毎回ユーザーに中継しない。
