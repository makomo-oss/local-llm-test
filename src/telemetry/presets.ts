import type { GenerationConfig, ModelSpec } from '../types.ts';

export type PresetCategory =
  | 'speed'
  | 'summarization'
  | 'reasoning'
  | 'coding'
  | 'comparison'
  | 'structured';

export type PresetLanguage = 'ja' | 'en' | 'ja/en';

/**
 * Curated prompt template for interactive testing and reproducible cross-device benchmarking.
 */
export interface BenchmarkPromptPreset {
  id: string;
  label: string;
  labelEn: string;
  category: PresetCategory;
  language: PresetLanguage;
  description: string;
  prompt: string;
  systemPrompt?: string;
  recommendedMaxNewTokens: number;
  recommendedEnableThinking: boolean;
}

/**
 * Curated Japanese and English prompt presets covering latency (TTFT), long-context
 * summarization, logical reasoning, TypeScript coding, and standardized comparison.
 */
export const BENCHMARK_PRESETS: readonly BenchmarkPromptPreset[] = [
  {
    id: 'quick-qa-ja',
    label: '簡潔QA・応答速度計測',
    labelEn: 'Quick QA & TTFT Probe',
    category: 'speed',
    language: 'ja',
    description:
      '短い日本語QAで初回トークン遅延 (TTFT) と立ち上がり速度を素早く測定します。',
    prompt:
      'WebGPU と WebAssembly (WASM) の違いを、ブラウザ上でローカル LLM を動かす観点で箇条書き3点・合計150文字以内で簡潔に説明してください。',
    recommendedMaxNewTokens: 256,
    recommendedEnableThinking: false,
  },
  {
    id: 'ja-summarization',
    label: '日本語長文要約',
    labelEn: 'Japanese Long-Text Summarization',
    category: 'summarization',
    language: 'ja',
    description:
      '長めの日本語技術文を入力し、プリフィル負荷と安定した日本語要約品質を確認します。',
    prompt: `次の文章を読み、「背景・課題」「技術的アプローチ」「期待される効果」の3項目に分けて日本語で要約してください。

---
近年、生成AIモデルをクラウドAPIに送信せず、ユーザーの端末内（オンデバイス）で直接実行する需要が高まっている。最大の理由はプライバシー保護とデータガバナンスであり、機密文書や社内の技術メモを外部サーバーへ送信せずに要約・検索・コード補助を行いたいという要件が企業や個人開発者の双方で増加している。また、サーバー推論コストの削減やオフライン環境・低遅延応答への期待も大きい。

しかし、ブラウザ単体で大規模言語モデル（LLM）を動かすには複数の技術的制約がある。第一にメモリ制約であり、FP16のままでは数GB〜十数GBに達する重みを4bit量子化（q4 / q4f16）へ圧縮しつつ精度劣化を抑える必要がある。第二に演算バックエンドの差異であり、WebGPUが利用できる環境ではシェーダー並列処理（特に shader-f16）により高速なトークン生成が可能だが、非対応ブラウザやGPUメモリ不足環境ではCPU（WebAssembly SIMD）へのフォールバックが不可欠となる。第三にUI応答性の維持であり、重い推論処理やモデル初期化をメインスレッドで実行すると画面の描画が停止するため、Web Workerへ推論を分離し、トークン単位のストリーミングと進捗可視化を行う設計が求められる。
---`,
    recommendedMaxNewTokens: 512,
    recommendedEnableThinking: false,
  },
  {
    id: 'logic-reasoning',
    label: '論理推論チェック',
    labelEn: 'Logical Reasoning Check',
    category: 'reasoning',
    language: 'ja',
    description:
      '制約充足パズルで論理的整合性と思考モード（Thinking）の挙動を検証します。',
    prompt:
      'A・B・Cの3人がそれぞれ異なる飲み物（コーヒー、紅茶、緑茶）を注文しました。条件は次の通りです。\n1. Aはコーヒーを注文していない。\n2. 紅茶を注文したのはBではない。\n3. Cは緑茶もコーヒーも注文していない。\nそれぞれが何を注文したか、根拠を段階的に示して答えてください。',
    recommendedMaxNewTokens: 768,
    recommendedEnableThinking: true,
  },
  {
    id: 'ts-codegen',
    label: 'TypeScriptコード生成',
    labelEn: 'TypeScript Code Generation',
    category: 'coding',
    language: 'ja/en',
    description:
      '型安全なTypeScriptユーティリティとエッジケース処理を実装させ、コード生成速度と品質を評価します。',
    prompt:
      'TypeScript で、非同期タスクを最大同時実行数 `concurrency` で制限しながら実行し、元の配列順序を保ったまま結果配列を返す汎用関数 `mapWithConcurrency<T, R>(items: readonly T[], concurrency: number, worker: (item: T, index: number) => Promise<R>): Promise<R[]>` を実装してください。異常系（`concurrency <= 0`、空配列、エラー発生時の挙動）への配慮と JSDoc コメントも含めてください。',
    recommendedMaxNewTokens: 768,
    recommendedEnableThinking: false,
  },
  {
    id: 'standard-comparison',
    label: '比較ベンチマーク標準プロンプト',
    labelEn: 'Standard Cross-Model Benchmark',
    category: 'comparison',
    language: 'en',
    description:
      'モデル・GPU/CPU・量子化方式 (q4f16 / q4 / q8) 間で tok/s・TTFT・tokens/Wh を公平に比較するための英語標準プロンプトです。',
    prompt:
      'Write a clear, structured technical explanation of how quantized KV caches reduce memory bandwidth bottlenecks during autoregressive LLM decoding. Include exactly 4 numbered sections: (1) Problem Statement, (2) KV Cache Growth, (3) FP16 vs Int8/Int4 Trade-offs, and (4) Practical Takeaways for Browser Inference. Keep the explanation concise and informative (~200 words).',
    recommendedMaxNewTokens: 512,
    recommendedEnableThinking: false,
  },
  {
    id: 'structured-json-en',
    label: '構造化JSON抽出 (EN)',
    labelEn: 'Structured JSON Extraction',
    category: 'structured',
    language: 'en',
    description:
      '指示追従性とJSONフォーマット厳守率を確認するための英語データ抽出タスクです。',
    prompt:
      'Extract the benchmark metadata from the text below and return ONLY valid JSON with keys `model`, `backend`, `dtype`, `tokensPerSec`, `ttftMs`, and `notes` (string array).\n\nText: "Tested Qwen 3.5 2B using q4f16 on WebGPU (discrete high-performance GPU). First token arrived in 412 ms and steady decode averaged 38.4 tokens/sec with zero dropped UI frames."',
    recommendedMaxNewTokens: 320,
    recommendedEnableThinking: false,
  },
] as const;

const PRESET_BY_ID = new Map<string, BenchmarkPromptPreset>(
  BENCHMARK_PRESETS.map((preset) => [preset.id, preset])
);

/**
 * Look up a benchmark preset by ID.
 */
export function getPresetById(id: string): BenchmarkPromptPreset | undefined {
  return PRESET_BY_ID.get(id);
}

/**
 * Create sensible default generation parameters for a given `ModelSpec`,
 * aligning `enableThinking` with the model's capabilities and catalog default.
 */
export function createDefaultGenerationConfig(
  model: ModelSpec,
  overrides?: Partial<GenerationConfig>
): GenerationConfig {
  const defaultThinking = model.supportsThinking ? model.defaultThinkingEnabled : false;
  const enableThinking = model.supportsThinking
    ? (overrides?.enableThinking ?? defaultThinking)
    : false;

  const defaultMaxNewTokens = enableThinking ? 1024 : 512;

  return {
    maxNewTokens: overrides?.maxNewTokens ?? defaultMaxNewTokens,
    temperature: overrides?.temperature ?? 0.7,
    topP: overrides?.topP ?? 0.9,
    topK: overrides?.topK ?? 40,
    doSample: overrides?.doSample ?? true,
    enableThinking,
  };
}
