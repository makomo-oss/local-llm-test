# 完全ブラウザ完結型 ローカル LLM 推論・ベンチマークスタジオ (`local-llm-test`)

[![Deploy to GitHub Pages](https://github.com/makomo-oss/local-llm-test/actions/workflows/deploy.yml/badge.svg)](https://github.com/makomo-oss/local-llm-test/actions/workflows/deploy.yml)

**WebGPU / WebAssembly (WASM)** を活用し、ブラウザ上だけでオープンソース LLM の推論・対話・リアルタイムテレメトリ計測（応答速度・負荷/メモリ・推定電力消費）およびモデル間ベンチマーク比較を行える静的 Web アプリケーションです。

- **Live Demo (GitHub Pages)**: <https://makomo-oss.github.io/local-llm-test/>

---

## 特徴と完全ブラウザ完結型アーキテクチャ

本アプリケーションは **Zero Server Inference（サーバー推論ゼロ）** で設計されており、バックエンド API や推論サーバーを一切持ちません。

- **ランタイム基盤**: [`@huggingface/transformers@4.3.2`](https://www.npmjs.com/package/@huggingface/transformers) + **ONNX Runtime Web**（WebGPU / WASM バックエンド）
- **専用 Web Worker 推論 (`src/engine/llm.worker.ts`)**: モデル取得・トークナイズ・ONNX 推論セッション生成・`TextStreamer` によるトークン生成をすべて Dedicated Web Worker（ES Module Worker）内で実行し、UI メインスレッドのブロッキングを防止します。
- **モデルキャッシュ (`CacheStorage`)**: 初回ロード時のみ Hugging Face (`onnx-community`) の公開 CDN から ONNX モデル・トークナイザーを取得し、ブラウザの `CacheStorage`（`transformers-cache`）へ永続キャッシュします。入力プロンプトや生成結果が外部サーバーへ送信されることはありません。
- **軽量フロントエンド**: Vite + TypeScript（Strict Mode）+ Vanilla TypeScript + Modern CSS（ビルド済み静的ファイルを GitHub Pages へ配信）。
- **対話・ベンチマーク機能**:
  - `InterruptableStoppingCriteria` による生成の即時中断
  - Thinking Mode（`<think>` / `<|channel>thought` トレースの折りたたみ分離表示）
  - `marked` + `DOMPurify` によるサニタイズ済み Markdown レンダリングとコードコピー
  - IME 変換確定誤送信対策 (`compositionstart` / `compositionend` ガード)
  - 1クリックベンチマーク、インライン SVG 比較チャート、`localStorage` 履歴保存、JSON / CSV エクスポート

---

## 対応モデル (6モデル) と量子化 / WASM Split-Dtype 仕様

すべて認証不要の公開 ONNX リポジトリ（`onnx-community`）を対象とし、テキスト推論パス（`AutoTokenizer` + `AutoModelForCausalLM`）で動作します。

| モデル名 | Hugging Face Repo ID | パラメータ | コンテキスト長 | Thinking 対応 | ライセンス | 目安サイズ (`q4f16` / `q4` テキスト推論時) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Gemma 4 E4B** | `onnx-community/gemma-4-E4B-it-ONNX` | E4B | 128K (131,072) | 対応 (デフォルト OFF) | Apache-2.0 | 約 4.59 GiB / 5.25 GiB |
| **Gemma 4 E2B** | `onnx-community/gemma-4-E2B-it-ONNX` | E2B | 128K (131,072) | 対応 (デフォルト OFF) | Apache-2.0 | 約 2.92 GiB / 3.40 GiB |
| **Qwen 3.5 4B** | `onnx-community/Qwen3.5-4B-ONNX` | 4B | 256K (262,144) | 対応 (デフォルト ON) | Apache-2.0 | 約 2.63 GiB / 2.91 GiB |
| **Qwen 3.5 2B** | `onnx-community/Qwen3.5-2B-ONNX` | 2B | 256K (262,144) | 対応 (デフォルト OFF) | Apache-2.0 | 約 1.31 GiB / 1.45 GiB |
| **Qwen 3.5 0.8B** | `onnx-community/Qwen3.5-0.8B-ONNX` | 0.8B | 256K (262,144) | 対応 (デフォルト OFF) | Apache-2.0 | 約 576.0 MiB / 637.5 MiB (※WASM時 967.1 MiB) |
| **Llama 3.2 3B Instruct** | `onnx-community/Llama-3.2-3B-Instruct-ONNX` | 3B | 128K (131,072) | 非対応 | llama3.2 | 約 2.25 GiB / 3.18 GiB |

### 量子化解決ルール (`resolveEffectiveDtype`)

選択されたモデル・計算デバイス・希望量子化（`auto` / `q4f16` / `q4` / `q8` / `fp16`）に応じて安全な実効量子化を自動解決します。

1. **WebGPU `shader-f16` 判定**:
   - `shader-f16` 対応アダプターでは `auto` 時に `q4f16` を選択します。
   - 非対応アダプターで `q4f16` / `fp16` を指定した場合は自動的に `q4` へフォールバックします。
2. **`Llama 3.2 3B Instruct` の `q8` 制御**:
   - 当該リポジトリの `config.json` は `model_quantized.onnx` の `use_external_data_format` 指定を省略しているため、UI 上で `use_external_data_format: true` を明示的にオプトインしない限り安全なデフォルト量子化へフォールバックします。
3. **CPU (`WASM`) の `embed_tokens` Split-Dtype 仕様**:
   - `Qwen 3.5` 系および `Gemma 4` 系の量子化済み `embed_tokens`（`embed_tokens_q4.onnx` / `embed_tokens_quantized.onnx`）には ONNX Runtime WASM CPU バックエンドが未対応の `GatherBlockQuantized(1)` オペレータが含まれます。
   - これを回避するため、CPU/WASM 実行時はセッション単位の `modelDtype` マップを自動適用します:
     - **Qwen 3.5 (0.8B / 2B / 4B)**: `{ embed_tokens: 'fp16', decoder_model_merged: 'q4' | 'q8' }`
     - **Gemma 4 (E2B / E4B)**: `{ embed_tokens: 'fp32', decoder_model_merged: 'q4' | 'q8' }`
   - UI（モデルカード・実効量子化プレビュー・ダウンロード進捗）は `getResolvedDownloadBytes()` により split-dtype 適用後の正確な合計サイズ（例: `Qwen 3.5 0.8B` + `WASM` + `q4` で **967.1 MiB**）を表示します。

---

## GPU アダプター選択と CPU (`WASM`) 実行

### WebGPU アダプター探索と明示的バインディング (`src/engine/deviceProbe.ts`)

1. `navigator.gpu.requestAdapter()` に対し以下の 4 プロファイルを順にプローブし、アダプター指紋 (`vendor | architecture | device | description | fallback`) および主要 `limits` で重複排除して一覧化します:
   - `high-performance` (`{ powerPreference: 'high-performance' }`)
   - `low-power` (`{ powerPreference: 'low-power' }`)
   - `default` (`{}`)
   - `fallback` (`{ forceFallbackAdapter: true }`)
2. 選択されたプロファイルを Web Worker へ渡し、Worker 内で `adapter.requestDevice({ requiredFeatures, requiredLimits })` を呼び出して `GPUDevice` を取得します（`shader-f16` / `subgroups` / `subgroup-size-control` / `timestamp-query` およびアダプター上限値を転送し、ドライバー非対応時は段階的に安全な記述子へフォールバック）。
3. 取得した `GPUDevice` を `AutoModelForCausalLM.from_pretrained()` の `session_options` へ直接バインドします:

```ts
session_options: {
  executionProviders: [{ name: 'webgpu', device: acquiredGpu.device }],
}
```

### CPU (`WASM`) 実行と非 COI 環境互換性

- `env.backends.onnx.wasm` にて **WASM SIMD (`simd = true`)** を有効化して実行します。
- GitHub Pages のように COOP/COEP ヘッダーが付与されず `crossOriginIsolated === false`（`SharedArrayBuffer` 利用不可）となる静的ホスティング環境では、ランタイム制約に合わせて自動的に **1 スレッド** (`numThreads = 1`) で安全に動作します（`crossOriginIsolated` 有効時は論理コア数に応じて最大 4 スレッドまで利用）。

---

## テレメトリ・計測メソドロジー (`src/telemetry/`)

推論実行ごとに 3 本柱のメトリクスをリアルタイム（HUD）および確定値（ベンチマーク記録）として計測します。

### 1. 応答速度 (Speed)
- Web Worker 内の `TextStreamer` トークンコールバックと `performance.now()` により計測:
  - **TTFT (Time To First Token)** / **Prefill 時間**: 生成開始から最初のトークン到着までの遅延
  - **Decode tok/s**: 第 1 トークン到着後から生成完了までのデコードスループット（`(generatedTokens - 1) / decodeSec`）
  - **Overall tok/s**: 総生成時間全体のスループット（瞬間 tok/s・1000ms 移動平均 tok/s もリアルタイム表示）
  - **ITL (Inter-Token Latency) 平均 / P95**: トークン間到着遅延の平均値と 95 パーセンタイル

### 2. 負荷・メモリ (Load)
- **推定モデル + KV キャッシュメモリ (`estimatedVramBytes`)**: テキスト推論重みサイズ + モデルごとの KV キャッシュ要素数 (`2 * layers * kvHeads * headDim` × コンテキストトークン数 × FP16/FP32 バイト幅) + アクティベーション・ランタイムオーバーヘッド + トークナイザーサイズから算出
- **JS Heap (`performance.memory`)**: Chromium 系など対応ブラウザにおいて Worker / Main の JS ヒープ使用量を表示
- **UA Memory (`performance.measureUserAgentSpecificMemory()`)**: `crossOriginIsolated` が有効な環境でのみ計測
- **メインスレッド応答性**: `requestAnimationFrame` ループによる平均フレーム時間・ドロップフレーム数 (`> 25 ms`)・CPU ビジー率、および `PerformanceObserver('longtask')` による Long Task 回数を計測

### 3. 電力消費 (Power)
- **実測バッテリー情報 + 負荷推定 (`battery-api+utilization`)**: Web Battery Status API (`navigator.getBattery()`) に対応し、バッテリー放電中 (`charging === false`) と判定された場合、バッテリー残量変化 (`level` 差分) や推定残放電時間 (`dischargingTime`) とデバイス負荷モデルを組み合わせて消費電力を算出します。
- **負荷ベース推定モデル (`utilization-model`)**: Battery Status API 非対応ブラウザ（Safari / Firefox 等）や AC 給電時は、デバイス種別（`discrete-gpu` / `integrated-gpu` / `software-gpu` / `cpu-wasm`）、モデルパラメータ規模、量子化倍率、メインスレッド/スレッド稼働率から推定電力 (W)・消費電力量 (Wh / mWh)・電力効率 (`tokens/Wh`) を算出します。

---

## ローカル開発・ビルド・デプロイ

### 必要環境
- Node.js 24 以上推奨（CI と同一バージョン）

### コマンド一覧

```bash
# 依存パッケージのインストール
npm install

# 開発サーバーの起動 (Vite)
npm run dev

# TypeScript 型チェック (tsc --noEmit)
npm run typecheck

# ユニット・UI統合テストの実行 (Vitest)
npm test

# プロダクションビルド (dist/ へ出力)
npm run build
```

> **Note**: `vite.config.ts` では `build.target: 'esnext'`、`worker.format: 'es'` が設定されており、ベースパスは環境変数 `VITE_BASE_PATH`（デフォルト `/local-llm-test/`）で切り替え可能です。

### GitHub Pages デプロイ (`.github/workflows/deploy.yml`)

`main` ブランチへのプッシュ（または GitHub Actions の手動実行 `workflow_dispatch`）をトリガーに、以下のパイプラインが自動実行されます:

1. `npm ci`
2. `npm run typecheck`
3. `npm test`
4. `VITE_BASE_PATH=/local-llm-test/ npm run build`
5. `actions/upload-pages-artifact@v3` → `actions/deploy-pages@v4` による GitHub Pages 公開
