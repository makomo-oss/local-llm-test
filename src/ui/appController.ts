import {
  getBrowserStorageEstimate,
  inspectModelCaches,
  type BrowserStorageEstimate,
} from '../engine/cacheManager.ts';
import {
  pickDefaultDevice,
  probeComputeDevices,
  toWorkerDeviceTarget,
} from '../engine/deviceProbe.ts';
import { LlmRuntimeClient, type GenerateResult } from '../engine/runtimeClient.ts';
import {
  MODEL_CATALOG,
  formatBytes,
  getModelById,
  getResolvedDownloadBytes,
  parseThinkingOutput,
  resolveEffectiveDtype,
} from '../models/catalog.ts';
import { BenchmarkStore, type StorageLike } from '../telemetry/benchmarkStore.ts';
import {
  LiveTelemetrySession,
  type LiveTelemetrySnapshot,
  type TelemetryEnvironment,
} from '../telemetry/metricsTracker.ts';
import {
  BENCHMARK_PRESETS,
  createDefaultGenerationConfig,
  getPresetById,
} from '../telemetry/presets.ts';
import {
  assessModelMemoryRisk,
  detectMobileDeviceContext,
  observeBatteryChanges,
  sampleBatteryStatus,
  type BatterySnapshot,
  type MemoryRiskAssessment,
  type MemoryRiskEnvironment,
} from '../telemetry/systemProbe.ts';
import type {
  BenchmarkRecord,
  ChatMessage,
  ChatPromptMessage,
  DeviceOption,
  GenerationConfig,
  InferenceMetrics,
  ModelCacheInfo,
  ModelId,
  ModelLoadProgress,
  QuantDtype,
} from '../types.ts';
import { StudioDialogHost } from './dialogs.ts';
import { attachImeSafeEnterSubmit } from './imeSubmit.ts';
import { escapeHtml } from './markdown.ts';
import {
  formatBatterySummary,
  formatLoadStageLabel,
  renderAssistantBodyHtml,
  renderChatMessageElement,
  renderComparisonViewHtml,
  renderDeviceDetailsHtml,
  renderEmptyGuideHtml,
  renderLoadProgressHtml,
  renderModelCardsHtml,
  renderTelemetryHudHtml,
} from './viewTemplates.ts';

export type ThemeMode = 'system' | 'dark' | 'light';
export type WorkspaceTab = 'chat' | 'compare';

const THEME_STORAGE_KEY = 'local-llm-test:theme';
const ACTIVE_LOAD_RECOVERY_KEY = 'local-llm-test:active-load';

interface ActiveLoadRecoveryMarker {
  modelId: ModelId;
  modelName: string;
  dtype: QuantDtype;
  backend: 'webgpu' | 'wasm';
  startedAt: number;
}

/**
 * Centralized debounced `aria-live` announcer so streaming tokens and progress
 * bursts never spam assistive technologies.
 */
export class DebouncedLiveAnnouncer {
  private readonly politeEl: HTMLElement;
  private readonly assertiveEl: HTMLElement;
  private readonly minIntervalMs: number;
  private readonly nowFn: () => number;
  private lastPoliteAt = Number.NEGATIVE_INFINITY;
  private pendingTimer: ReturnType<typeof setTimeout> | null = null;
  private pendingMessage: string | null = null;

  constructor(options: {
    politeEl: HTMLElement;
    assertiveEl: HTMLElement;
    minIntervalMs?: number;
    nowFn?: () => number;
  }) {
    this.politeEl = options.politeEl;
    this.assertiveEl = options.assertiveEl;
    this.minIntervalMs = options.minIntervalMs ?? 1500;
    this.nowFn = options.nowFn ?? (() => Date.now());
  }

  announcePolite(message: string, opts?: { immediate?: boolean }): void {
    const trimmed = message.trim();
    if (!trimmed) {
      return;
    }

    if (opts?.immediate) {
      if (this.pendingTimer !== null) {
        clearTimeout(this.pendingTimer);
        this.pendingTimer = null;
      }
      this.pendingMessage = null;
      this.lastPoliteAt = this.nowFn();
      this.politeEl.textContent = trimmed;
      return;
    }

    const elapsed = this.nowFn() - this.lastPoliteAt;
    if (elapsed >= this.minIntervalMs && this.pendingTimer === null) {
      this.lastPoliteAt = this.nowFn();
      this.politeEl.textContent = trimmed;
      return;
    }

    this.pendingMessage = trimmed;
    if (this.pendingTimer === null) {
      const waitMs = Math.max(50, this.minIntervalMs - Math.max(0, elapsed));
      this.pendingTimer = setTimeout(() => {
        this.pendingTimer = null;
        if (this.pendingMessage) {
          const msg = this.pendingMessage;
          this.pendingMessage = null;
          this.lastPoliteAt = this.nowFn();
          this.politeEl.textContent = msg;
        }
      }, waitMs);
    }
  }

  announceAssertive(message: string): void {
    const trimmed = message.trim();
    if (!trimmed) {
      return;
    }
    this.assertiveEl.textContent = trimmed;
  }

  dispose(): void {
    if (this.pendingTimer !== null) {
      clearTimeout(this.pendingTimer);
      this.pendingTimer = null;
    }
    this.pendingMessage = null;
  }
}

export interface AppControllerDeps {
  client?: LlmRuntimeClient;
  benchmarkStore?: BenchmarkStore;
  probeDevicesFn?: () => Promise<DeviceOption[]>;
  inspectCacheFn?: () => Promise<ModelCacheInfo[]>;
  clearCacheFn?: (
    modelId?: ModelId
  ) => Promise<{ clearedBytes: number; deletedEntries: number }>;
  getStorageEstimateFn?: () => Promise<BrowserStorageEstimate>;
  sampleBatteryFn?: () => Promise<BatterySnapshot>;
  observeBatteryFn?: (onChange: (snap: BatterySnapshot) => void) => Promise<() => void>;
  telemetryEnv?: TelemetryEnvironment;
  requestAnimationFrameFn?: (cb: FrameRequestCallback) => number;
  cancelAnimationFrameFn?: (id: number) => void;
  nowFn?: () => number;
  themeStorage?: StorageLike | null;
  sessionStorage?: StorageLike | null;
  memoryRiskEnv?: MemoryRiskEnvironment;
  downloadFileFn?: (filename: string, content: string, mimeType: string) => void;
  copyTextFn?: (text: string) => Promise<void>;
}

interface LoadedRuntimeState {
  modelId: ModelId;
  device: DeviceOption;
  effectiveDtype: QuantDtype;
  loadTimeMs: number;
  fallbackNotice?: string;
}

interface ActiveGenerationContext {
  assistantMessageId: string;
  modelId: ModelId;
  device: DeviceOption;
  dtype: QuantDtype;
  thinkingExpectedInPrompt: boolean;
  session: LiveTelemetrySession;
  rawFullText: string;
  tokenCount: number;
  dirty: boolean;
  rafId: number | null;
  liveSnapshot: LiveTelemetrySnapshot | null;
  promptId?: string;
  promptLabel: string;
  promptText: string;
  generationConfig: GenerationConfig;
  forceBenchmarkRecord: boolean;
}

function resolveDefaultThemeStorage(): StorageLike | undefined {
  try {
    const storage = (globalThis as { localStorage?: StorageLike }).localStorage;
    if (storage && typeof storage.getItem === 'function') {
      return storage;
    }
  } catch {
    // Ignore restricted localStorage.
  }
  return undefined;
}

function resolveDefaultSessionStorage(): StorageLike | undefined {
  try {
    const storage = (globalThis as { sessionStorage?: StorageLike }).sessionStorage;
    if (storage && typeof storage.getItem === 'function') {
      return storage;
    }
  } catch {
    // Ignore restricted sessionStorage.
  }
  return undefined;
}

function defaultDownloadFile(filename: string, content: string, mimeType: string): void {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.style.display = 'none';
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function defaultCopyText(text: string): Promise<void> {
  if (typeof navigator !== 'undefined' && navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const helper = document.createElement('textarea');
  helper.value = text;
  helper.setAttribute('readonly', '');
  helper.style.position = 'fixed';
  helper.style.opacity = '0';
  document.body.appendChild(helper);
  helper.select();
  document.execCommand?.('copy');
  helper.remove();
}

export class AppController {
  private readonly root: HTMLElement;
  private readonly client: LlmRuntimeClient;
  private readonly ownsClient: boolean;
  private readonly benchmarkStore: BenchmarkStore;
  private readonly deps: AppControllerDeps;
  private readonly dialogHost: StudioDialogHost;
  private readonly themeStorage: StorageLike | undefined;
  private readonly loadSessionStorage: StorageLike | undefined;
  private readonly memoryRiskEnv: MemoryRiskEnvironment | undefined;

  private announcer: DebouncedLiveAnnouncer | null = null;
  private cleanupImeSubmit: (() => void) | null = null;
  private cleanupBatteryObserver: (() => void) | null = null;
  private cleanupPageHideListener: (() => void) | null = null;
  private disposed = false;

  // Core UI state
  private themeMode: ThemeMode = 'system';
  private activeTab: WorkspaceTab = 'chat';
  private devices: DeviceOption[] = [];
  private selectedDeviceId = '';
  private selectedModelId: ModelId = 'qwen-3.5-0.8b';
  private requestedDtype: QuantDtype | 'auto' = 'auto';
  private useExternalDataFormat = false;
  private systemPrompt = '';
  private selectedPresetId = BENCHMARK_PRESETS[0]?.id ?? 'quick-qa-ja';
  private generationConfig: GenerationConfig = createDefaultGenerationConfig(
    getModelById('qwen-3.5-0.8b')
  );
  private autoRecordBenchmarks = true;

  private cacheMap = new Map<ModelId, ModelCacheInfo>();
  private storageEstimate: BrowserStorageEstimate = {};
  private liveBattery: BatterySnapshot | undefined;

  private loadedState: LoadedRuntimeState | null = null;
  private isLoadingModel = false;
  private loadProgress: ModelLoadProgress | null = null;
  private statusBanner: { kind: 'info' | 'success' | 'warn' | 'danger'; text: string } | null = null;

  private messages: ChatMessage[] = [];
  private recordedMessageIds = new Set<string>();
  private lastCompletedMetrics: InferenceMetrics | null = null;
  private activeGen: ActiveGenerationContext | null = null;
  private userToggledThinkingIds = new Set<string>();
  private messageCounter = 0;

  constructor(root: HTMLElement, deps: AppControllerDeps = {}) {
    this.root = root;
    this.deps = deps;
    this.ownsClient = !deps.client;
    this.client =
      deps.client ??
      new LlmRuntimeClient({
        onUnexpectedError: (err) => {
          this.handleUnexpectedRuntimeError(err.message);
        },
      });
    this.benchmarkStore = deps.benchmarkStore ?? new BenchmarkStore();
    this.themeStorage =
      deps.themeStorage === null ? undefined : (deps.themeStorage ?? resolveDefaultThemeStorage());
    this.loadSessionStorage =
      deps.sessionStorage === null
        ? undefined
        : (deps.sessionStorage ?? resolveDefaultSessionStorage());
    this.memoryRiskEnv = deps.memoryRiskEnv;
    this.dialogHost = new StudioDialogHost(this.root);

    this.restoreThemeMode();
  }

  async init(): Promise<void> {
    this.renderShell();
    this.bindEvents();
    this.bindPageLifecycleCleanup();
    this.consumeInterruptedLoadRecovery();
    this.applyThemeMode(this.themeMode, false);
    this.syncModelControls();
    this.renderChatMessages();
    this.renderComparison();
    this.renderTelemetryHud();

    await Promise.all([
      this.refreshDevices(),
      this.refreshCachesAndStorage(),
      this.refreshBattery(),
    ]);

    if (this.disposed) {
      return;
    }

    const observeFn = this.deps.observeBatteryFn ?? observeBatteryChanges;
    try {
      this.cleanupBatteryObserver = await observeFn((snap) => {
        if (this.disposed) return;
        this.liveBattery = snap;
        this.renderHeaderPills();
        if (!this.activeGen) {
          this.renderTelemetryHud();
        }
      });
    } catch {
      this.cleanupBatteryObserver = null;
    }
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.cancelStreamingRaf();
    this.cleanupImeSubmit?.();
    this.cleanupImeSubmit = null;
    this.cleanupBatteryObserver?.();
    this.cleanupBatteryObserver = null;
    this.cleanupPageHideListener?.();
    this.cleanupPageHideListener = null;
    this.clearActiveLoadRecoveryMarker();
    this.announcer?.dispose();
    this.announcer = null;
    this.dialogHost.closeActive();
    if (this.ownsClient) {
      this.client.dispose();
    }
  }

  private bindPageLifecycleCleanup(): void {
    if (typeof window === 'undefined' || typeof window.addEventListener !== 'function') {
      return;
    }
    const handleCleanExit = (): void => {
      this.clearActiveLoadRecoveryMarker();
    };
    window.addEventListener('pagehide', handleCleanExit);
    window.addEventListener('beforeunload', handleCleanExit);
    this.cleanupPageHideListener = () => {
      window.removeEventListener('pagehide', handleCleanExit);
      window.removeEventListener('beforeunload', handleCleanExit);
    };
  }

  private saveActiveLoadRecoveryMarker(marker: ActiveLoadRecoveryMarker): void {
    try {
      this.loadSessionStorage?.setItem(ACTIVE_LOAD_RECOVERY_KEY, JSON.stringify(marker));
    } catch {
      // Ignore restricted storage.
    }
  }

  private clearActiveLoadRecoveryMarker(): void {
    try {
      this.loadSessionStorage?.removeItem(ACTIVE_LOAD_RECOVERY_KEY);
    } catch {
      // Ignore restricted storage.
    }
  }

  private consumeInterruptedLoadRecovery(): void {
    try {
      const raw = this.loadSessionStorage?.getItem(ACTIVE_LOAD_RECOVERY_KEY);
      if (!raw) {
        return;
      }
      this.loadSessionStorage?.removeItem(ACTIVE_LOAD_RECOVERY_KEY);
      const parsed = JSON.parse(raw) as Partial<ActiveLoadRecoveryMarker>;
      const label =
        typeof parsed.modelName === 'string' && parsed.modelName.trim()
          ? parsed.modelName.trim()
          : 'モデル';
      this.statusBanner = {
        kind: 'warn',
        text: `⚠ 前回の ${label} ロード中にブラウザのメモリ上限などでタブが再起動されました。ダウンロード済みファイルはキャッシュに残っています。スマホ環境では「Qwen 3.5 0.8B (WebGPU)」の使用や他タブの終了をおすすめします。`,
      };
    } catch {
      this.clearActiveLoadRecoveryMarker();
    }
  }

  private getSelectedMemoryRisk(): MemoryRiskAssessment {
    const model = this.getSelectedModel();
    const resolved = this.getEffectiveDtypeResolution();
    const device = this.getSelectedDevice();
    return assessModelMemoryRisk(model, resolved, device, this.memoryRiskEnv);
  }

  getStateSnapshot() {
    return {
      themeMode: this.themeMode,
      activeTab: this.activeTab,
      selectedModelId: this.selectedModelId,
      selectedDeviceId: this.selectedDeviceId,
      requestedDtype: this.requestedDtype,
      loadedState: this.loadedState,
      isLoadingModel: this.isLoadingModel,
      isGenerating: Boolean(this.activeGen),
      messages: [...this.messages],
      benchmarkCount: this.benchmarkStore.list().length,
    };
  }

  private restoreThemeMode(): void {
    try {
      const saved = this.themeStorage?.getItem(THEME_STORAGE_KEY);
      if (saved === 'light' || saved === 'dark' || saved === 'system') {
        this.themeMode = saved;
      }
    } catch {
      this.themeMode = 'system';
    }
  }

  private applyThemeMode(mode: ThemeMode, announce = true): void {
    this.themeMode = mode;
    const docEl = this.root.ownerDocument?.documentElement;
    if (docEl) {
      if (mode === 'system') {
        delete docEl.dataset.theme;
      } else {
        docEl.dataset.theme = mode;
      }
    }

    try {
      if (mode === 'system') {
        this.themeStorage?.removeItem(THEME_STORAGE_KEY);
      } else {
        this.themeStorage?.setItem(THEME_STORAGE_KEY, mode);
      }
    } catch {
      // Ignore storage failures.
    }

    const buttons = this.root.querySelectorAll<HTMLButtonElement>('button[data-theme-mode]');
    buttons.forEach((btn) => {
      btn.setAttribute('aria-pressed', btn.dataset.themeMode === mode ? 'true' : 'false');
    });

    if (announce) {
      const label = mode === 'system' ? 'システム設定' : mode === 'dark' ? 'ダーク' : 'ライト';
      this.announcer?.announcePolite(`テーマを ${label} に切り替えました。`, { immediate: true });
    }
  }

  private renderShell(): void {
    const presetOptionsHtml = BENCHMARK_PRESETS.map(
      (p) => `<option value="${escapeHtml(p.id)}">${escapeHtml(p.label)}</option>`
    ).join('');

    this.root.innerHTML = `
      <div class="studio-shell">
        <div id="studio-live-polite" class="sr-only" aria-live="polite" aria-atomic="true"></div>
        <div id="studio-live-assertive" class="sr-only" aria-live="assertive" aria-atomic="true"></div>

        <header class="app-header">
          <div class="brand-group">
            <div class="brand-mark" aria-hidden="true">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <rect x="4" y="4" width="16" height="16" rx="2"/>
                <rect x="9" y="9" width="6" height="6"/>
                <path d="M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 14h3M1 9h3M1 14h3"/>
              </svg>
            </div>
            <div class="brand-titles">
              <h1 class="app-title">完全ブラウザ完結型ローカルLLM推論・ベンチマークスタジオ</h1>
              <div class="brand-badges">
                <span class="status-pill status-pill--privacy">🔒 完全ブラウザ実行・サーバー送信なし</span>
                <span id="header-battery-pill" class="status-pill">バッテリー確認中...</span>
                <span id="header-storage-pill" class="status-pill">ストレージ確認中...</span>
              </div>
            </div>
          </div>

          <div class="header-controls">
            <div class="theme-switcher" role="group" aria-label="テーマ切替">
              <button type="button" class="theme-btn" data-theme-mode="system" aria-pressed="true">システム</button>
              <button type="button" class="theme-btn" data-theme-mode="dark" aria-pressed="false">ダーク</button>
              <button type="button" class="theme-btn" data-theme-mode="light" aria-pressed="false">ライト</button>
            </div>
            <button type="button" class="btn btn--sm" id="btn-open-methodology">技術仕様 / 計測方法</button>
          </div>
        </header>

        <div class="studio-body">
          <aside class="control-sidebar" aria-label="モデル・デバイス・推論設定">
            <section class="panel-section" aria-labelledby="sec-model-heading">
              <div class="section-header">
                <h2 id="sec-model-heading" class="section-title">1. モデル選択 (6モデル)</h2>
                <span id="loaded-model-badge" class="chip">未ロード</span>
              </div>
              <div id="model-list" class="model-list" role="group" aria-label="利用可能なモデル一覧"></div>
              <p id="selected-model-desc" class="model-desc-box"></p>
            </section>

            <section class="panel-section" aria-labelledby="sec-device-heading">
              <div class="section-header">
                <h2 id="sec-device-heading" class="section-title">2. 使用デバイス選択</h2>
                <div class="btn-group">
                  <button type="button" class="btn btn--sm" id="btn-reprobe-devices">デバイス再検出</button>
                  <button type="button" class="btn btn--sm" id="btn-device-specs">詳細スペック</button>
                </div>
              </div>
              <div class="field">
                <label class="field-label" for="device-select">WebGPU GPU / CPU (WASM)</label>
                <select id="device-select" class="control-select"></select>
              </div>
              <div id="device-details-slot"></div>
            </section>

            <section class="panel-section" aria-labelledby="sec-quant-heading">
              <div class="section-header">
                <h2 id="sec-quant-heading" class="section-title">3. 量子化 &amp; キャッシュ管理</h2>
              </div>

              <div class="field">
                <label class="field-label" for="dtype-select">量子化方式 (dtype)</label>
                <select id="dtype-select" class="control-select">
                  <option value="auto">auto (デバイス最適を自動選択)</option>
                  <option value="q4f16">q4f16 (4bit重み + FP16 KV / 最速推奨)</option>
                  <option value="q4">q4 (4bit重み + FP32 / 汎用・CPU推奨)</option>
                  <option value="q8">q8 (8bit量子化)</option>
                  <option value="fp16">fp16 (半精度 / 大容量VRAM向け)</option>
                </select>
              </div>

              <div id="llama-q8-optin-row" hidden>
                <label class="checkbox-row">
                  <input type="checkbox" id="chk-external-data-q8" />
                  <span>Llama q8 外部データ形式を明示許可 (<code>use_external_data_format: true</code>)</span>
                </label>
              </div>

              <div id="dtype-preview-box" class="notice-box"></div>
              <div id="runtime-status-banner" hidden></div>

              <div class="btn-group">
                <button type="button" class="btn btn--primary" id="btn-load-model">モデルをロード / ダウンロード</button>
                <button type="button" class="btn" id="btn-unload-model" disabled>モデル解放 (Unload)</button>
              </div>

              <div id="load-progress-slot"></div>

              <div id="storage-quota-box" class="notice-box"></div>

              <div class="btn-group">
                <button type="button" class="btn btn--sm btn--danger" id="btn-clear-model-cache">選択モデルのキャッシュ削除</button>
                <button type="button" class="btn btn--sm btn--danger" id="btn-clear-all-cache">全キャッシュ削除</button>
              </div>
            </section>

            <section class="panel-section" aria-labelledby="sec-gen-heading">
              <div class="section-header">
                <h2 id="sec-gen-heading" class="section-title">4. 推論・ベンチマーク設定</h2>
              </div>

              <div class="field">
                <label class="field-label" for="preset-select">プリセットプロンプト</label>
                <div class="btn-group">
                  <select id="preset-select" class="control-select" style="flex: 1 1 180px;">
                    ${presetOptionsHtml}
                  </select>
                  <button type="button" class="btn btn--sm" id="btn-apply-preset">入力欄に反映</button>
                </div>
              </div>

              <div class="field">
                <label class="field-label" for="system-prompt-input">System Prompt (任意)</label>
                <textarea id="system-prompt-input" class="control-textarea" rows="2" placeholder="例: あなたは正確で簡潔な技術アシスタントです。"></textarea>
              </div>

              <label class="checkbox-row" id="thinking-toggle-row">
                <input type="checkbox" id="chk-enable-thinking" />
                <span id="thinking-toggle-label">Thinking Mode (思考トレース生成)</span>
              </label>

              <label class="checkbox-row">
                <input type="checkbox" id="chk-do-sample" checked />
                <span>サンプリング有効 (do_sample)</span>
              </label>

              <div class="field-row">
                <div class="field">
                  <label class="field-label" for="input-max-tokens">maxNewTokens</label>
                  <input id="input-max-tokens" class="control-input" type="number" min="16" max="4096" step="16" value="512" />
                </div>
                <div class="field">
                  <label class="field-label" for="input-temperature">temperature</label>
                  <input id="input-temperature" class="control-input" type="number" min="0" max="2" step="0.05" value="0.7" />
                </div>
                <div class="field">
                  <label class="field-label" for="input-top-p">topP</label>
                  <input id="input-top-p" class="control-input" type="number" min="0.05" max="1" step="0.05" value="0.9" />
                </div>
                <div class="field">
                  <label class="field-label" for="input-top-k">topK</label>
                  <input id="input-top-k" class="control-input" type="number" min="1" max="200" step="1" value="40" />
                </div>
              </div>
            </section>
          </aside>

          <main class="workspace-main">
            <nav class="workspace-tabs" aria-label="ワークスペース表示切替">
              <div class="tablist" role="tablist" aria-label="スタジオ機能タブ">
                <button
                  type="button"
                  role="tab"
                  id="tab-chat"
                  class="tab-btn"
                  aria-selected="true"
                  aria-controls="panel-chat"
                >
                  チャット &amp; ライブ計測
                </button>
                <button
                  type="button"
                  role="tab"
                  id="tab-compare"
                  class="tab-btn"
                  aria-selected="false"
                  aria-controls="panel-compare"
                  tabindex="-1"
                >
                  比較ベンチマーク
                  <span id="benchmark-count-badge" class="tab-count-badge">0</span>
                </button>
              </div>
              <div id="active-run-summary" class="section-subtitle">モデルをロードすると推論・計測が可能になります。</div>
            </nav>

            <section id="panel-chat" class="workspace-panel" role="tabpanel" aria-labelledby="tab-chat">
              <div id="telemetry-hud" class="telemetry-hud" aria-label="リアルタイム推論テレメトリー"></div>

              <div class="chat-card">
                <div id="chat-messages" class="chat-messages" aria-label="チャット履歴"></div>

                <form id="chat-composer-form" class="composer-form">
                  <div class="field">
                    <label class="field-label" for="chat-prompt-input">プロンプト入力</label>
                    <textarea
                      id="chat-prompt-input"
                      class="composer-textarea"
                      rows="2"
                      placeholder="質問やベンチマーク用プロンプトを入力... (Enter で送信 / Shift+Enter で改行 / IME変換中は誤送信しません)"
                    ></textarea>
                  </div>
                  <div class="composer-footer">
                    <div class="btn-group">
                      <label class="checkbox-row">
                        <input type="checkbox" id="chk-auto-record" checked />
                        <span>完了時に比較ベンチへ自動記録</span>
                      </label>
                    </div>
                    <div class="btn-group">
                      <button type="button" class="btn btn--sm" id="btn-clear-chat">会話クリア</button>
                      <button type="button" class="btn" id="btn-run-benchmark">ベンチマーク実行 (1クリック計測)</button>
                      <button type="button" class="btn btn--danger" id="btn-stop-generation" disabled>停止</button>
                      <button type="submit" class="btn btn--primary" id="btn-send-chat" aria-disabled="true">送信</button>
                    </div>
                  </div>
                </form>
              </div>
            </section>

            <section id="panel-compare" class="workspace-panel" role="tabpanel" aria-labelledby="tab-compare" hidden>
              <div id="comparison-container"></div>
            </section>
          </main>
        </div>
      </div>
    `;

    const politeEl = this.root.querySelector<HTMLElement>('#studio-live-polite')!;
    const assertiveEl = this.root.querySelector<HTMLElement>('#studio-live-assertive')!;
    this.announcer = new DebouncedLiveAnnouncer({
      politeEl,
      assertiveEl,
      nowFn: this.deps.nowFn,
    });
  }

  private bindEvents(): void {
    // Header theme switcher & methodology modal
    this.root.addEventListener('click', (event) => {
      const target = event.target as HTMLElement | null;
      if (!target) return;

      const themeBtn = target.closest<HTMLButtonElement>('button[data-theme-mode]');
      if (themeBtn?.dataset.themeMode) {
        this.applyThemeMode(themeBtn.dataset.themeMode as ThemeMode);
        return;
      }

      if (target.closest('#btn-open-methodology')) {
        this.dialogHost.showMethodologyDialog();
        return;
      }

      const modelBtn = target.closest<HTMLButtonElement>('button[data-select-model]');
      if (modelBtn?.dataset.selectModel) {
        this.selectModel(modelBtn.dataset.selectModel as ModelId);
        return;
      }

      if (target.closest('#btn-reprobe-devices')) {
        void this.refreshDevices();
        return;
      }

      if (target.closest('#btn-device-specs')) {
        const dev = this.getSelectedDevice();
        if (dev) {
          this.dialogHost.showDeviceSpecsDialog(dev);
        }
        return;
      }

      if (target.closest('#btn-load-model')) {
        void this.handleLoadModel();
        return;
      }

      if (target.closest('#btn-unload-model')) {
        void this.handleUnloadModel();
        return;
      }

      if (target.closest('#btn-clear-model-cache')) {
        void this.handleClearCache(this.selectedModelId);
        return;
      }

      if (target.closest('#btn-clear-all-cache')) {
        void this.handleClearCache(undefined);
        return;
      }

      if (target.closest('#btn-apply-preset')) {
        this.applySelectedPresetToComposer(true);
        return;
      }

      const presetChip = target.closest<HTMLButtonElement>('button[data-apply-preset]');
      if (presetChip?.dataset.applyPreset) {
        this.selectedPresetId = presetChip.dataset.applyPreset;
        const select = this.root.querySelector<HTMLSelectElement>('#preset-select');
        if (select) select.value = this.selectedPresetId;
        this.applySelectedPresetToComposer(true);
        return;
      }

      if (target.closest('#btn-clear-chat')) {
        this.clearChatHistory();
        return;
      }

      if (target.closest('#btn-run-benchmark')) {
        void this.handleRunBenchmark();
        return;
      }

      if (target.closest('#btn-stop-generation')) {
        this.handleAbortGeneration();
        return;
      }

      const copyCodeBtn = target.closest<HTMLButtonElement>('button.code-copy-btn');
      if (copyCodeBtn) {
        const codeText =
          copyCodeBtn.closest('.md-code-block')?.querySelector('pre > code')?.textContent ?? '';
        void this.copyWithFeedback(copyCodeBtn, codeText);
        return;
      }

      const copyMsgBtn = target.closest<HTMLButtonElement>('button[data-copy-message-id]');
      if (copyMsgBtn?.dataset.copyMessageId) {
        const msg = this.messages.find((m) => m.id === copyMsgBtn.dataset.copyMessageId);
        if (msg) {
          void this.copyWithFeedback(copyMsgBtn, msg.content);
        }
        return;
      }

      const recordMsgBtn = target.closest<HTMLButtonElement>('button[data-record-message-id]');
      if (recordMsgBtn?.dataset.recordMessageId) {
        this.recordExistingMessageToBenchmark(recordMsgBtn.dataset.recordMessageId);
        return;
      }

      const delBenchBtn = target.closest<HTMLButtonElement>('button[data-delete-benchmark]');
      if (delBenchBtn?.dataset.deleteBenchmark) {
        this.benchmarkStore.remove(delBenchBtn.dataset.deleteBenchmark);
        this.renderComparison();
        this.announcer?.announcePolite('ベンチマーク記録を1件削除しました。', { immediate: true });
        return;
      }

      if (target.closest('button[data-export-json]')) {
        const downloadFn = this.deps.downloadFileFn ?? defaultDownloadFile;
        downloadFn(
          `local-llm-benchmarks-${Date.now()}.json`,
          this.benchmarkStore.exportJson(),
          'application/json;charset=utf-8'
        );
        return;
      }

      if (target.closest('button[data-export-csv]')) {
        const downloadFn = this.deps.downloadFileFn ?? defaultDownloadFile;
        downloadFn(
          `local-llm-benchmarks-${Date.now()}.csv`,
          this.benchmarkStore.exportCsv(),
          'text/csv;charset=utf-8'
        );
        return;
      }

      if (target.closest('button[data-clear-benchmarks]')) {
        void this.handleClearAllBenchmarks();
        return;
      }

      const tabBtn = target.closest<HTMLButtonElement>('button[role="tab"]');
      if (tabBtn) {
        const nextTab: WorkspaceTab = tabBtn.id === 'tab-compare' ? 'compare' : 'chat';
        this.switchTab(nextTab);
      }
    });

    // Track user toggling thinking <details> during streaming
    this.root.addEventListener(
      'toggle',
      (event) => {
        const details = event.target as HTMLElement | null;
        if (!details || !details.classList.contains('thinking-trace')) return;
        const msgEl = details.closest<HTMLElement>('[data-message-id]');
        if (msgEl?.dataset.messageId) {
          this.userToggledThinkingIds.add(msgEl.dataset.messageId);
        }
      },
      true
    );

    // Tablist keyboard navigation
    const tablist = this.root.querySelector<HTMLElement>('[role="tablist"]');
    tablist?.addEventListener('keydown', (event) => {
      if (
        event.key === 'ArrowRight' ||
        event.key === 'ArrowLeft' ||
        event.key === 'Home' ||
        event.key === 'End'
      ) {
        event.preventDefault();
        const nextTab: WorkspaceTab =
          event.key === 'Home'
            ? 'chat'
            : event.key === 'End'
              ? 'compare'
              : this.activeTab === 'chat'
                ? 'compare'
                : 'chat';
        this.switchTab(nextTab, true);
      }
    });

    // Select & input bindings
    const deviceSelect = this.root.querySelector<HTMLSelectElement>('#device-select');
    deviceSelect?.addEventListener('change', () => {
      this.selectedDeviceId = deviceSelect.value;
      this.syncDeviceAndDtypeUi();
    });

    const dtypeSelect = this.root.querySelector<HTMLSelectElement>('#dtype-select');
    dtypeSelect?.addEventListener('change', () => {
      this.requestedDtype = dtypeSelect.value as QuantDtype | 'auto';
      this.syncModelControls();
    });

    const extQ8Checkbox = this.root.querySelector<HTMLInputElement>('#chk-external-data-q8');
    extQ8Checkbox?.addEventListener('change', () => {
      this.useExternalDataFormat = extQ8Checkbox.checked;
      this.syncModelControls();
    });

    const presetSelect = this.root.querySelector<HTMLSelectElement>('#preset-select');
    presetSelect?.addEventListener('change', () => {
      this.selectedPresetId = presetSelect.value;
    });

    const systemPromptInput = this.root.querySelector<HTMLTextAreaElement>('#system-prompt-input');
    systemPromptInput?.addEventListener('input', () => {
      this.systemPrompt = systemPromptInput.value;
    });

    const chkThinking = this.root.querySelector<HTMLInputElement>('#chk-enable-thinking');
    chkThinking?.addEventListener('change', () => {
      this.generationConfig = {
        ...this.generationConfig,
        enableThinking: chkThinking.checked,
      };
    });

    const chkDoSample = this.root.querySelector<HTMLInputElement>('#chk-do-sample');
    chkDoSample?.addEventListener('change', () => {
      this.generationConfig = {
        ...this.generationConfig,
        doSample: chkDoSample.checked,
      };
    });

    const chkAutoRecord = this.root.querySelector<HTMLInputElement>('#chk-auto-record');
    chkAutoRecord?.addEventListener('change', () => {
      this.autoRecordBenchmarks = chkAutoRecord.checked;
    });

    this.bindNumericInput('#input-max-tokens', 'maxNewTokens', 16, 4096, true);
    this.bindNumericInput('#input-temperature', 'temperature', 0, 2, false);
    this.bindNumericInput('#input-top-p', 'topP', 0.05, 1, false);
    this.bindNumericInput('#input-top-k', 'topK', 1, 200, true);

    // Composer form + IME-safe Enter submit
    const form = this.root.querySelector<HTMLFormElement>('#chat-composer-form');
    const textarea = this.root.querySelector<HTMLTextAreaElement>('#chat-prompt-input');
    if (form && textarea) {
      this.cleanupImeSubmit = attachImeSafeEnterSubmit({
        textarea,
        form,
      });

      form.addEventListener('submit', (event) => {
        event.preventDefault();
        void this.handleChatSubmit();
      });
    }
  }

  private bindNumericInput(
    selector: string,
    key: 'maxNewTokens' | 'temperature' | 'topP' | 'topK',
    min: number,
    max: number,
    integer: boolean
  ): void {
    const input = this.root.querySelector<HTMLInputElement>(selector);
    input?.addEventListener('change', () => {
      const raw = Number(input.value);
      if (!Number.isFinite(raw)) {
        input.value = String(this.generationConfig[key]);
        return;
      }
      const clamped = Math.min(max, Math.max(min, integer ? Math.round(raw) : raw));
      input.value = String(clamped);
      this.generationConfig = {
        ...this.generationConfig,
        [key]: clamped,
      };
    });
  }

  private switchTab(tab: WorkspaceTab, focusTab = false): void {
    this.activeTab = tab;
    const chatTab = this.root.querySelector<HTMLButtonElement>('#tab-chat');
    const compareTab = this.root.querySelector<HTMLButtonElement>('#tab-compare');
    const chatPanel = this.root.querySelector<HTMLElement>('#panel-chat');
    const comparePanel = this.root.querySelector<HTMLElement>('#panel-compare');

    const isChat = tab === 'chat';
    if (chatTab) {
      chatTab.setAttribute('aria-selected', isChat ? 'true' : 'false');
      chatTab.tabIndex = isChat ? 0 : -1;
    }
    if (compareTab) {
      compareTab.setAttribute('aria-selected', isChat ? 'false' : 'true');
      compareTab.tabIndex = isChat ? -1 : 0;
    }
    if (chatPanel) {
      chatPanel.hidden = !isChat;
    }
    if (comparePanel) {
      comparePanel.hidden = isChat;
    }

    if (focusTab) {
      (isChat ? chatTab : compareTab)?.focus();
    }
  }

  private getSelectedModel() {
    return getModelById(this.selectedModelId);
  }

  private getSelectedDevice(): DeviceOption | undefined {
    return (
      this.devices.find((d) => d.id === this.selectedDeviceId) ??
      pickDefaultDevice(this.devices) ??
      this.devices[0]
    );
  }

  private getEffectiveDtypeResolution() {
    const model = this.getSelectedModel();
    const device = this.getSelectedDevice() ?? {
      backend: 'webgpu' as const,
      supportsFp16: true,
      features: ['shader-f16'],
    };
    return resolveEffectiveDtype(model, device, this.requestedDtype, {
      useExternalDataFormat: this.useExternalDataFormat,
    });
  }

  private selectModel(modelId: ModelId): void {
    if (this.selectedModelId === modelId) return;
    this.selectedModelId = modelId;
    const model = getModelById(modelId);

    // Sync Thinking Mode to selected model capability & default
    const nextConfig = createDefaultGenerationConfig(model, {
      maxNewTokens: this.generationConfig.maxNewTokens,
      temperature: this.generationConfig.temperature,
      topP: this.generationConfig.topP,
      topK: this.generationConfig.topK,
      doSample: this.generationConfig.doSample,
    });
    this.generationConfig = nextConfig;
    if (this.loadedState && this.loadedState.modelId !== modelId) {
      const loadedModel = getModelById(this.loadedState.modelId);
      this.statusBanner = {
        kind: 'info',
        text: `現在 ${loadedModel.name} がロード済みです。${model.name} に切り替えるには「モデルをロード / ダウンロード」を実行してください。`,
      };
    }
    this.syncModelControls();
  }

  private syncModelControls(): void {
    const model = this.getSelectedModel();

    // Update model cards
    const modelListEl = this.root.querySelector<HTMLElement>('#model-list');
    if (modelListEl) {
      modelListEl.innerHTML = renderModelCardsHtml({
        models: MODEL_CATALOG,
        selectedModelId: this.selectedModelId,
        loadedModelId: this.loadedState?.modelId ?? null,
        selectedDevice: this.getSelectedDevice(),
        requestedDtype: this.requestedDtype,
        useExternalDataFormat: this.useExternalDataFormat,
        cacheMap: this.cacheMap,
        resolveDtypeFn: resolveEffectiveDtype,
      });
    }

    const descEl = this.root.querySelector<HTMLElement>('#selected-model-desc');
    if (descEl) {
      descEl.textContent = `${model.descriptionJa} (${model.repoId} / ${model.license})`;
    }

    // Llama q8 explicit opt-in checkbox
    const llamaOptRow = this.root.querySelector<HTMLElement>('#llama-q8-optin-row');
    if (llamaOptRow) {
      llamaOptRow.hidden = !(
        model.requiresExplicitExternalDataForQ8 && this.requestedDtype === 'q8'
      );
    }

    // Thinking toggle synced with model
    const chkThinking = this.root.querySelector<HTMLInputElement>('#chk-enable-thinking');
    const thinkingLabel = this.root.querySelector<HTMLElement>('#thinking-toggle-label');
    if (chkThinking) {
      chkThinking.checked = model.supportsThinking && this.generationConfig.enableThinking;
      chkThinking.disabled = !model.supportsThinking;
    }
    if (thinkingLabel) {
      thinkingLabel.textContent = model.supportsThinking
        ? 'Thinking Mode (思考トレース生成)'
        : 'Thinking Mode (このモデルは非思考型)';
    }

    const chkDoSample = this.root.querySelector<HTMLInputElement>('#chk-do-sample');
    if (chkDoSample) chkDoSample.checked = this.generationConfig.doSample;
    const maxTokensInput = this.root.querySelector<HTMLInputElement>('#input-max-tokens');
    if (maxTokensInput) maxTokensInput.value = String(this.generationConfig.maxNewTokens);
    const tempInput = this.root.querySelector<HTMLInputElement>('#input-temperature');
    if (tempInput) tempInput.value = String(this.generationConfig.temperature);
    const topPInput = this.root.querySelector<HTMLInputElement>('#input-top-p');
    if (topPInput) topPInput.value = String(this.generationConfig.topP);
    const topKInput = this.root.querySelector<HTMLInputElement>('#input-top-k');
    if (topKInput) topKInput.value = String(this.generationConfig.topK);

    // Sync dtype & device-memory risk preview box
    const resolved = this.getEffectiveDtypeResolution();
    const expectedBytes = getResolvedDownloadBytes(model, resolved, true);
    const risk = this.getSelectedMemoryRisk();
    const previewEl = this.root.querySelector<HTMLElement>('#dtype-preview-box');
    if (previewEl) {
      previewEl.className =
        risk.level === 'high'
          ? 'notice-box notice-box--danger'
          : resolved.adjusted || risk.level === 'caution'
            ? 'notice-box notice-box--warn'
            : 'notice-box';
      const warningLine =
        resolved.adjusted && resolved.reason
          ? `<span>⚠ ${escapeHtml(resolved.reason)}</span>`
          : '';
      const memoryLine =
        risk.level !== 'ok'
          ? `<span>${risk.level === 'high' ? '🛑' : '⚠'} ${escapeHtml(risk.headline)}${risk.detail ? ` ${escapeHtml(risk.detail)}` : ''}</span>`
          : '';
      previewEl.innerHTML = `
        <strong>実効量子化: ${escapeHtml(resolved.dtype)} (予想取得サイズ: ${escapeHtml(formatBytes(expectedBytes))})</strong>
        ${warningLine}
        ${memoryLine}`;
    }

    this.syncActionButtons();
  }

  private syncDeviceAndDtypeUi(): void {
    const device = this.getSelectedDevice();
    const detailsEl = this.root.querySelector<HTMLElement>('#device-details-slot');
    if (detailsEl) {
      detailsEl.innerHTML = renderDeviceDetailsHtml(device);
    }
    this.syncModelControls();
    if (!this.activeGen) {
      this.renderTelemetryHud();
    }
  }

  private syncActionButtons(): void {
    const loadedBadge = this.root.querySelector<HTMLElement>('#loaded-model-badge');
    const activeSummary = this.root.querySelector<HTMLElement>('#active-run-summary');
    const btnLoad = this.root.querySelector<HTMLButtonElement>('#btn-load-model');
    const btnUnload = this.root.querySelector<HTMLButtonElement>('#btn-unload-model');
    const btnSend = this.root.querySelector<HTMLButtonElement>('#btn-send-chat');
    const btnBench = this.root.querySelector<HTMLButtonElement>('#btn-run-benchmark');
    const btnStop = this.root.querySelector<HTMLButtonElement>('#btn-stop-generation');
    const btnClearModelCache = this.root.querySelector<HTMLButtonElement>('#btn-clear-model-cache');
    const btnClearAllCache = this.root.querySelector<HTMLButtonElement>('#btn-clear-all-cache');

    const busy = this.isLoadingModel || Boolean(this.activeGen);
    const hasLoaded = Boolean(this.loadedState);
    const selectedDevice = this.getSelectedDevice();
    const canLoad = !busy && Boolean(selectedDevice?.available);

    if (loadedBadge) {
      if (this.isLoadingModel) {
        loadedBadge.className = 'chip chip--accent';
        loadedBadge.textContent = 'ロード中...';
      } else if (this.loadedState) {
        const m = getModelById(this.loadedState.modelId);
        loadedBadge.className = 'chip chip--loaded';
        loadedBadge.textContent = `${m.name} (${this.loadedState.effectiveDtype})`;
      } else {
        loadedBadge.className = 'chip';
        loadedBadge.textContent = '未ロード';
      }
    }

    if (activeSummary) {
      if (this.activeGen) {
        activeSummary.textContent = `推論実行中: ${getModelById(this.activeGen.modelId).name} (${this.activeGen.tokenCount} tokens)`;
      } else if (this.loadedState) {
        const m = getModelById(this.loadedState.modelId);
        activeSummary.textContent = `準備完了: ${m.name} / ${this.loadedState.device.label} / ${this.loadedState.effectiveDtype}`;
      } else {
        activeSummary.textContent = 'モデルをロードすると推論・計測が可能になります。';
      }
    }

    if (btnLoad) {
      btnLoad.disabled = !canLoad;
    }
    if (btnUnload) {
      btnUnload.disabled = busy || !hasLoaded;
    }
    if (btnStop) {
      btnStop.disabled = !this.activeGen;
    }
    if (btnSend) {
      btnSend.setAttribute('aria-disabled', busy || !hasLoaded ? 'true' : 'false');
    }
    if (btnBench) {
      btnBench.setAttribute('aria-disabled', busy || !hasLoaded ? 'true' : 'false');
    }
    if (btnClearModelCache) {
      btnClearModelCache.disabled = this.isLoadingModel;
    }
    if (btnClearAllCache) {
      btnClearAllCache.disabled = this.isLoadingModel;
    }

    const bannerEl = this.root.querySelector<HTMLElement>('#runtime-status-banner');
    if (bannerEl) {
      if (!this.statusBanner) {
        bannerEl.hidden = true;
        bannerEl.innerHTML = '';
      } else {
        bannerEl.hidden = false;
        const cls =
          this.statusBanner.kind === 'danger'
            ? 'notice-box notice-box--danger'
            : this.statusBanner.kind === 'warn'
              ? 'notice-box notice-box--warn'
              : this.statusBanner.kind === 'success'
                ? 'notice-box notice-box--success'
                : 'notice-box';
        bannerEl.className = cls;
        bannerEl.textContent = this.statusBanner.text;
      }
    }
  }

  private renderHeaderPills(): void {
    const batPill = this.root.querySelector<HTMLElement>('#header-battery-pill');
    if (batPill) {
      batPill.textContent = formatBatterySummary(this.liveBattery);
    }

    const storagePill = this.root.querySelector<HTMLElement>('#header-storage-pill');
    const quotaBox = this.root.querySelector<HTMLElement>('#storage-quota-box');

    const { usageBytes, quotaBytes, availableBytes } = this.storageEstimate;
    if (storagePill) {
      storagePill.textContent =
        usageBytes !== undefined && quotaBytes !== undefined
          ? `ストレージ: ${formatBytes(usageBytes)} / ${formatBytes(quotaBytes)}`
          : 'ストレージ: 計測不可';
    }

    if (quotaBox) {
      if (usageBytes !== undefined && quotaBytes !== undefined && quotaBytes > 0) {
        const usedPct = Math.min(100, Math.round((usageBytes / quotaBytes) * 1000) / 10);
        quotaBox.innerHTML = `
          <div class="progress-header">
            <span>ブラウザストレージ使用量</span>
            <span>${usedPct.toFixed(1)}%</span>
          </div>
          <progress class="studio-progress" max="100" value="${usedPct}">${usedPct}%</progress>
          <span class="hud-note">使用: ${escapeHtml(formatBytes(usageBytes))} / 空き: ${escapeHtml(
            formatBytes(availableBytes ?? 0)
          )} / 上限: ${escapeHtml(formatBytes(quotaBytes))}</span>`;
      } else {
        quotaBox.innerHTML =
          '<span class="hud-note">navigator.storage.estimate() の利用上限情報を取得できませんでした。</span>';
      }
    }
  }

  private renderLoadProgress(): void {
    const slot = this.root.querySelector<HTMLElement>('#load-progress-slot');
    if (slot) {
      slot.innerHTML = renderLoadProgressHtml(this.loadProgress);
    }
  }

  private renderTelemetryHud(): void {
    const hud = this.root.querySelector<HTMLElement>('#telemetry-hud');
    if (!hud) return;
    hud.innerHTML = renderTelemetryHudHtml({
      liveSnapshot: this.activeGen?.liveSnapshot ?? null,
      finalMetrics: this.lastCompletedMetrics,
      liveBattery: this.liveBattery,
      isGenerating: Boolean(this.activeGen),
      selectedDevice: this.loadedState?.device ?? this.getSelectedDevice(),
    });
  }

  private renderChatMessages(): void {
    const container = this.root.querySelector<HTMLElement>('#chat-messages');
    if (!container) return;

    if (this.messages.length === 0) {
      container.innerHTML = renderEmptyGuideHtml(BENCHMARK_PRESETS);
      return;
    }

    container.innerHTML = '';
    const fragment = document.createDocumentFragment();
    for (const msg of this.messages) {
      fragment.appendChild(renderChatMessageElement(msg, this.recordedMessageIds));
    }
    container.appendChild(fragment);
    container.scrollTop = container.scrollHeight;
  }

  private renderComparison(): void {
    const records = this.benchmarkStore.list();
    const badge = this.root.querySelector<HTMLElement>('#benchmark-count-badge');
    if (badge) {
      badge.textContent = String(records.length);
    }

    const container = this.root.querySelector<HTMLElement>('#comparison-container');
    if (!container) return;

    container.innerHTML = renderComparisonViewHtml({
      records,
      summary: this.benchmarkStore.getSummaryStats(records),
      comparison: this.benchmarkStore.computeComparison(records),
    });
  }

  async refreshDevices(): Promise<void> {
    const probeFn =
      this.deps.probeDevicesFn ??
      (async () => {
        try {
          return await this.client.probeDevices();
        } catch {
          return await probeComputeDevices();
        }
      });

    try {
      const devices = await probeFn();
      if (this.disposed) return;
      this.devices = devices;
      if (!devices.some((d) => d.id === this.selectedDeviceId && d.available)) {
        this.selectedDeviceId = pickDefaultDevice(devices)?.id ?? devices[0]?.id ?? '';
      }

      const select = this.root.querySelector<HTMLSelectElement>('#device-select');
      if (select) {
        select.innerHTML = devices
          .map(
            (d) =>
              `<option value="${escapeHtml(d.id)}"${!d.available ? ' disabled' : ''}${
                d.id === this.selectedDeviceId ? ' selected' : ''
              }>${escapeHtml(d.label)}${!d.available ? ' [利用不可]' : ''}</option>`
          )
          .join('');
        select.value = this.selectedDeviceId;
      }

      this.syncDeviceAndDtypeUi();
      this.announcer?.announcePolite(`${devices.filter((d) => d.available).length} 個の実行デバイスを検出しました。`);
    } catch (err) {
      this.statusBanner = {
        kind: 'danger',
        text: `デバイス検出エラー: ${err instanceof Error ? err.message : String(err)}`,
      };
      this.syncActionButtons();
    }
  }

  async refreshCachesAndStorage(): Promise<void> {
    const inspectFn =
      this.deps.inspectCacheFn ??
      (async () => {
        try {
          return await this.client.inspectCache();
        } catch {
          return await inspectModelCaches();
        }
      });
    const estimateFn = this.deps.getStorageEstimateFn ?? getBrowserStorageEstimate;

    const [caches, estimate] = await Promise.all([
      inspectFn().catch(() => [] as ModelCacheInfo[]),
      estimateFn().catch(() => ({} as BrowserStorageEstimate)),
    ]);

    if (this.disposed) return;
    this.cacheMap = new Map(caches.map((c) => [c.modelId, c]));
    this.storageEstimate = estimate;
    this.renderHeaderPills();
    this.syncModelControls();
  }

  async refreshBattery(): Promise<void> {
    const sampleFn = this.deps.sampleBatteryFn ?? sampleBatteryStatus;
    try {
      const snap = await sampleFn();
      if (this.disposed) return;
      this.liveBattery = snap;
      this.renderHeaderPills();
      if (!this.activeGen) {
        this.renderTelemetryHud();
      }
    } catch {
      // Ignore battery sampling error.
    }
  }

  async handleLoadModel(): Promise<void> {
    if (this.isLoadingModel || this.activeGen) return;

    const device = this.getSelectedDevice();
    if (!device || !device.available) {
      this.statusBanner = {
        kind: 'warn',
        text: '利用可能な計算デバイスを選択してください。WebGPU が利用不可の場合は CPU (WASM) を選択できます。',
      };
      this.syncActionButtons();
      this.announcer?.announceAssertive(this.statusBanner.text);
      return;
    }

    const model = this.getSelectedModel();
    const resolved = this.getEffectiveDtypeResolution();
    const memoryRisk = this.getSelectedMemoryRisk();

    if (memoryRisk.level === 'high') {
      const confirmed = await this.dialogHost.showConfirmDialog({
        title: '端末メモリ上限によるクラッシュ・再読込の警告',
        message: memoryRisk.headline,
        detail:
          memoryRisk.detail ??
          'スマホや低メモリ環境では Qwen 3.5 0.8B (WebGPU) を選択するか、他タブを閉じて実行してください。',
        confirmLabel: 'このまま試行する',
        cancelLabel: 'キャンセル',
        danger: true,
      });
      if (!confirmed || this.disposed) {
        return;
      }
    }

    this.isLoadingModel = true;
    this.loadProgress = {
      modelId: model.id,
      stage: 'probing-device',
      message: `${model.name} (${resolved.dtype}) のロードを開始します...`,
      overallProgressPct: 0,
      loadedBytes: 0,
      expectedTotalBytes: getResolvedDownloadBytes(model, resolved, true),
      files: [],
    };
    this.statusBanner = null;
    this.renderLoadProgress();
    this.syncActionButtons();
    this.announcer?.announcePolite(`${model.name} のロードを開始します。`, { immediate: true });
    this.saveActiveLoadRecoveryMarker({
      modelId: model.id,
      modelName: model.name,
      dtype: resolved.dtype,
      backend: device.backend,
      startedAt: Date.now(),
    });

    let lastProgressRenderAt = 0;
    let lastRenderedStage: ModelLoadProgress['stage'] | null = null;

    try {
      const loaded = await this.client.loadModel(
        {
          modelId: model.id,
          deviceTarget: toWorkerDeviceTarget(device),
          dtype: resolved.dtype,
          ...(resolved.useExternalDataFormat ? { useExternalDataFormat: true } : {}),
          mobileContext: detectMobileDeviceContext(this.memoryRiskEnv),
        },
        (progress) => {
          if (this.disposed) return;
          this.loadProgress = progress;
          const now = Date.now();
          if (
            progress.stage !== lastRenderedStage ||
            progress.stage !== 'downloading' ||
            now - lastProgressRenderAt >= 80
          ) {
            lastRenderedStage = progress.stage;
            lastProgressRenderAt = now;
            this.renderLoadProgress();
          }
          this.announcer?.announcePolite(
            `${formatLoadStageLabel(progress.stage)}: ${Math.round(progress.overallProgressPct)}%`
          );
        }
      );

      if (this.disposed) return;
      this.loadedState = {
        modelId: loaded.modelId,
        device: loaded.device,
        effectiveDtype: loaded.effectiveDtype,
        loadTimeMs: loaded.loadTimeMs,
        ...(loaded.fallbackNotice ? { fallbackNotice: loaded.fallbackNotice } : {}),
      };
      this.loadProgress = {
        modelId: loaded.modelId,
        stage: 'ready',
        message: `${model.name} のロードが完了しました (${(loaded.loadTimeMs / 1000).toFixed(1)} 秒)。`,
        overallProgressPct: 100,
        loadedBytes: this.loadProgress?.expectedTotalBytes ?? 0,
        expectedTotalBytes: this.loadProgress?.expectedTotalBytes ?? 0,
        files: this.loadProgress?.files ?? [],
      };
      this.statusBanner = loaded.fallbackNotice
        ? { kind: 'warn', text: loaded.fallbackNotice }
        : {
            kind: 'success',
            text: `✓ ${model.name} (${loaded.effectiveDtype}) を ${loaded.device.label} へロードしました (${(loaded.loadTimeMs / 1000).toFixed(1)} 秒)。`,
          };
      this.announcer?.announcePolite(`${model.name} の準備が完了しました。`, { immediate: true });
      await this.refreshCachesAndStorage();
    } catch (err) {
      if (this.disposed) return;
      const msg = err instanceof Error ? err.message : String(err);
      this.loadProgress = null;
      this.statusBanner = {
        kind: 'danger',
        text: `✕ モデルロード失敗: ${msg}`,
      };
      this.announcer?.announceAssertive(`モデルのロードに失敗しました。${msg}`);
    } finally {
      this.clearActiveLoadRecoveryMarker();
      this.isLoadingModel = false;
      this.renderLoadProgress();
      this.syncModelControls();
    }
  }

  async handleUnloadModel(): Promise<void> {
    if (this.isLoadingModel || this.activeGen || !this.loadedState) return;
    try {
      await this.client.unloadModel();
      this.loadedState = null;
      this.loadProgress = null;
      this.statusBanner = {
        kind: 'info',
        text: 'モデルを解放し、VRAM / メモリをクリアしました。',
      };
      this.renderLoadProgress();
      this.syncModelControls();
      this.announcer?.announcePolite('モデルを解放しました。', { immediate: true });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      this.statusBanner = { kind: 'danger', text: `モデル解放エラー: ${msg}` };
      this.syncActionButtons();
      this.announcer?.announceAssertive(`モデル解放エラー: ${msg}`);
    }
  }

  async handleClearCache(modelId?: ModelId): Promise<void> {
    if (this.isLoadingModel) return;
    const model = modelId ? getModelById(modelId) : undefined;
    const confirmed = await this.dialogHost.showConfirmDialog({
      title: model ? `${model.name} のキャッシュを削除` : 'すべてのブラウザキャッシュを削除',
      message: model
        ? `${model.name} の保存済み重みファイル・トークナイザーをブラウザキャッシュ (CacheStorage) から削除しますか？`
        : 'すべてのモデルキャッシュおよび ONNX Runtime WASM キャッシュを削除しますか？',
      detail: '削除後、次回ロード時に再ダウンロードが必要になります。',
      confirmLabel: '削除する',
      cancelLabel: 'キャンセル',
      danger: true,
    });

    if (!confirmed || this.disposed) return;

    try {
      const clearFn = this.deps.clearCacheFn ?? ((id) => this.client.clearCache(id));
      const res = await clearFn(modelId);
      this.statusBanner = {
        kind: 'success',
        text: `キャッシュを削除しました (${res.deletedEntries} エントリ / ${formatBytes(res.clearedBytes)})。`,
      };
      await this.refreshCachesAndStorage();
      this.announcer?.announcePolite(this.statusBanner.text, { immediate: true });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      this.statusBanner = { kind: 'danger', text: `キャッシュ削除失敗: ${msg}` };
      this.syncActionButtons();
      this.announcer?.announceAssertive(this.statusBanner.text);
    }
  }

  private async handleClearAllBenchmarks(): Promise<void> {
    const confirmed = await this.dialogHost.showConfirmDialog({
      title: 'ベンチマーク履歴の削除',
      message: '保存されているすべての比較ベンチマーク記録を削除しますか？',
      detail: '削除前に必要に応じて JSON / CSV エクスポートをご利用ください。',
      confirmLabel: 'すべて削除',
      cancelLabel: 'キャンセル',
      danger: true,
    });
    if (!confirmed || this.disposed) return;

    this.benchmarkStore.clear();
    this.recordedMessageIds.clear();
    this.renderComparison();
    this.renderChatMessages();
    this.announcer?.announcePolite('すべてのベンチマーク履歴を削除しました。', { immediate: true });
  }

  private applySelectedPresetToComposer(focusInput = false): void {
    const preset = getPresetById(this.selectedPresetId);
    if (!preset) return;

    const textarea = this.root.querySelector<HTMLTextAreaElement>('#chat-prompt-input');
    if (textarea) {
      textarea.value = preset.prompt;
      if (focusInput) textarea.focus();
    }

    if (preset.systemPrompt !== undefined) {
      this.systemPrompt = preset.systemPrompt;
      const sysEl = this.root.querySelector<HTMLTextAreaElement>('#system-prompt-input');
      if (sysEl) sysEl.value = preset.systemPrompt;
    }

    const model = this.getSelectedModel();
    const enableThinking = model.supportsThinking ? preset.recommendedEnableThinking : false;
    this.generationConfig = {
      ...this.generationConfig,
      maxNewTokens: preset.recommendedMaxNewTokens,
      enableThinking,
    };

    const maxTokensInput = this.root.querySelector<HTMLInputElement>('#input-max-tokens');
    if (maxTokensInput) maxTokensInput.value = String(preset.recommendedMaxNewTokens);

    const chkThinking = this.root.querySelector<HTMLInputElement>('#chk-enable-thinking');
    if (chkThinking && model.supportsThinking) chkThinking.checked = enableThinking;

    this.announcer?.announcePolite(`プリセット「${preset.label}」を適用しました。`, {
      immediate: true,
    });
  }

  private clearChatHistory(): void {
    if (this.activeGen) return;
    this.messages = [];
    this.recordedMessageIds.clear();
    this.userToggledThinkingIds.clear();
    this.renderChatMessages();
    this.announcer?.announcePolite('会話履歴をクリアしました。', { immediate: true });
  }

  async handleChatSubmit(): Promise<void> {
    if (this.activeGen || this.isLoadingModel) return;

    if (!this.loadedState) {
      this.statusBanner = {
        kind: 'warn',
        text: 'モデルが未ロードです。左パネルの「モデルをロード / ダウンロード」を先に実行してください。',
      };
      this.syncActionButtons();
      this.announcer?.announceAssertive(this.statusBanner.text);
      this.root.querySelector<HTMLButtonElement>('#btn-load-model')?.focus();
      return;
    }

    const textarea = this.root.querySelector<HTMLTextAreaElement>('#chat-prompt-input');
    const promptText = textarea?.value.trim() ?? '';
    if (!promptText) {
      this.announcer?.announcePolite('プロンプトを入力してください。', { immediate: true });
      textarea?.focus();
      return;
    }

    if (textarea) {
      textarea.value = '';
    }

    const matchingPreset = BENCHMARK_PRESETS.find((p) => p.prompt.trim() === promptText);
    await this.executeGeneration({
      promptText,
      promptId: matchingPreset?.id,
      promptLabel: matchingPreset?.label ?? 'チャット対話',
      singleTurn: false,
      forceBenchmarkRecord: this.autoRecordBenchmarks,
    });
  }

  async handleRunBenchmark(): Promise<void> {
    if (this.activeGen || this.isLoadingModel) return;

    if (!this.loadedState) {
      this.statusBanner = {
        kind: 'warn',
        text: 'モデルが未ロードです。左パネルでモデルをロードしてからベンチマークを実行してください。',
      };
      this.syncActionButtons();
      this.announcer?.announceAssertive(this.statusBanner.text);
      this.root.querySelector<HTMLButtonElement>('#btn-load-model')?.focus();
      return;
    }

    const textarea = this.root.querySelector<HTMLTextAreaElement>('#chat-prompt-input');
    let promptText = textarea?.value.trim() ?? '';
    const preset = getPresetById(this.selectedPresetId) ?? BENCHMARK_PRESETS[0];

    let promptId: string | undefined;
    let promptLabel = 'カスタム計測';

    if (!promptText && preset) {
      promptText = preset.prompt;
      promptId = preset.id;
      promptLabel = preset.label;
      if (textarea) {
        textarea.value = preset.prompt;
      }
    } else if (preset && promptText === preset.prompt.trim()) {
      promptId = preset.id;
      promptLabel = preset.label;
    }

    if (!promptText) return;

    await this.executeGeneration({
      promptText,
      promptId,
      promptLabel,
      singleTurn: true,
      forceBenchmarkRecord: true,
    });
  }

  private async executeGeneration(params: {
    promptText: string;
    promptId?: string;
    promptLabel: string;
    singleTurn: boolean;
    forceBenchmarkRecord: boolean;
  }): Promise<void> {
    const loaded = this.loadedState;
    if (!loaded) return;

    const model = getModelById(loaded.modelId);
    const config: GenerationConfig = {
      ...this.generationConfig,
      enableThinking: model.supportsThinking ? this.generationConfig.enableThinking : false,
    };

    const promptMessages: ChatPromptMessage[] = [];
    if (this.systemPrompt.trim().length > 0) {
      promptMessages.push({ role: 'system', content: this.systemPrompt.trim() });
    }
    if (!params.singleTurn) {
      for (const m of this.messages) {
        if ((m.role === 'user' || m.role === 'assistant') && !m.error && m.content.trim().length > 0) {
          promptMessages.push({ role: m.role, content: m.content });
        }
      }
    }
    promptMessages.push({ role: 'user', content: params.promptText });

    const userMsg: ChatMessage = {
      id: this.nextMessageId('user'),
      role: 'user',
      content: params.promptText,
      createdAt: Date.now(),
    };
    const assistantMsg: ChatMessage = {
      id: this.nextMessageId('assistant'),
      role: 'assistant',
      content: '',
      thinking: '',
      createdAt: Date.now(),
      modelId: model.id,
      deviceId: loaded.device.id,
      dtype: loaded.effectiveDtype,
      isStreaming: true,
      isThinkingInProgress: false,
    };

    this.messages.push(userMsg, assistantMsg);
    this.renderChatMessages();

    const session = new LiveTelemetrySession({
      model,
      device: loaded.device,
      dtype: loaded.effectiveDtype,
      activeThreads: loaded.device.runtimeDetails.numThreads,
      env: this.deps.telemetryEnv,
    });

    const initialSnap = await session.start();

    const thinkingExpectedInPrompt = model.family === 'qwen3_5' && config.enableThinking;
    const genCtx: ActiveGenerationContext = {
      assistantMessageId: assistantMsg.id,
      modelId: model.id,
      device: loaded.device,
      dtype: loaded.effectiveDtype,
      thinkingExpectedInPrompt,
      session,
      rawFullText: '',
      tokenCount: 0,
      dirty: false,
      rafId: null,
      liveSnapshot: initialSnap,
      promptId: params.promptId,
      promptLabel: params.promptLabel,
      promptText: params.promptText,
      generationConfig: config,
      forceBenchmarkRecord: params.forceBenchmarkRecord,
    };

    this.activeGen = genCtx;
    this.statusBanner = null;
    this.syncActionButtons();
    this.renderTelemetryHud();
    this.announcer?.announcePolite(`${model.name} で推論を開始しました。`, { immediate: true });

    let result: GenerateResult | null = null;
    let errorMessage: string | undefined;

    try {
      result = await this.client.generate(promptMessages, config, {
        onStart: ({ promptTokens }) => {
          session.onGenerationStarted(promptTokens);
          genCtx.liveSnapshot = session.getSnapshot();
          this.scheduleStreamingFlush();
        },
        onToken: (ev) => {
          genCtx.rawFullText = ev.fullText;
          genCtx.tokenCount = ev.tokenCount;
          genCtx.liveSnapshot = session.onToken(ev.tokenCount);
          genCtx.dirty = true;
          this.scheduleStreamingFlush();
        },
      });
    } catch (err) {
      errorMessage = err instanceof Error ? err.message : String(err);
    } finally {
      this.cancelStreamingRaf();
    }

    if (this.disposed) return;

    this.activeGen = null;

    if (result) {
      const finalMetrics = await session.finish({
        speed: result.speed,
        workerJsHeapUsedBytes: result.workerJsHeapUsedBytes,
        workerJsHeapTotalBytes: result.workerJsHeapTotalBytes,
        activeThreads: result.activeThreads,
        backend: result.backend,
        dtype: result.dtype,
      });

      this.lastCompletedMetrics = finalMetrics;
      assistantMsg.content = result.content;
      assistantMsg.thinking = result.thinking;
      assistantMsg.isStreaming = false;
      assistantMsg.isThinkingInProgress = false;
      assistantMsg.aborted = result.aborted;
      assistantMsg.metrics = finalMetrics;

      if (
        (genCtx.forceBenchmarkRecord || this.autoRecordBenchmarks) &&
        !result.aborted &&
        result.speed.generatedTokens > 0
      ) {
        this.saveBenchmarkRecord({
          assistantMsg,
          promptId: genCtx.promptId,
          promptLabel: genCtx.promptLabel,
          promptText: genCtx.promptText,
          device: genCtx.device,
          dtype: genCtx.dtype,
          generationConfig: genCtx.generationConfig,
          status: 'completed',
          metrics: finalMetrics,
        });
      }

      const speedVal = (
        finalMetrics.speed.decodeTokensPerSec > 0
          ? finalMetrics.speed.decodeTokensPerSec
          : finalMetrics.speed.tokensPerSec
      ).toFixed(1);
      this.statusBanner = result.aborted
        ? { kind: 'warn', text: `推論を中断しました (${finalMetrics.speed.generatedTokens} tokens 生成)。` }
        : {
            kind: 'success',
            text: `✓ 完了: ${finalMetrics.speed.generatedTokens} tokens / ${speedVal} tok/s / TTFT ${Math.round(finalMetrics.speed.ttftMs)} ms`,
          };
      this.announcer?.announcePolite(
        result.aborted
          ? `生成を中断しました。${finalMetrics.speed.generatedTokens} トークン生成。`
          : `生成完了。${finalMetrics.speed.generatedTokens} トークン、${speedVal} トークン毎秒。`,
        { immediate: true }
      );
    } else {
      session.abort();
      assistantMsg.isStreaming = false;
      assistantMsg.isThinkingInProgress = false;
      assistantMsg.error = errorMessage ?? '不明な推論エラーが発生しました。';
      this.statusBanner = {
        kind: 'danger',
        text: `✕ 推論エラー: ${assistantMsg.error}`,
      };
      this.announcer?.announceAssertive(`推論エラー: ${assistantMsg.error}`);
    }

    this.renderChatMessages();
    this.renderTelemetryHud();
    this.renderComparison();
    this.syncActionButtons();
  }

  private scheduleStreamingFlush(): void {
    const genCtx = this.activeGen;
    if (!genCtx || genCtx.rafId !== null) {
      return;
    }

    const raf =
      this.deps.requestAnimationFrameFn ??
      ((cb: FrameRequestCallback) =>
        typeof requestAnimationFrame === 'function'
          ? requestAnimationFrame(cb)
          : (setTimeout(() => cb(Date.now()), 16) as unknown as number));

    genCtx.rafId = raf(() => {
      this.flushStreamingUpdate();
    });
  }

  private cancelStreamingRaf(): void {
    const genCtx = this.activeGen;
    if (!genCtx || genCtx.rafId === null) {
      return;
    }
    const cancelRaf =
      this.deps.cancelAnimationFrameFn ??
      ((id: number) => {
        if (typeof cancelAnimationFrame === 'function') {
          cancelAnimationFrame(id);
        } else {
          clearTimeout(id);
        }
      });
    cancelRaf(genCtx.rafId);
    genCtx.rafId = null;
  }

  /**
   * Flush buffered streaming tokens and live telemetry to the DOM once per animation frame.
   */
  flushStreamingUpdate(): void {
    const genCtx = this.activeGen;
    if (!genCtx) return;
    genCtx.rafId = null;
    if (!genCtx.dirty) {
      this.renderTelemetryHud();
      return;
    }
    genCtx.dirty = false;

    const msg = this.messages.find((m) => m.id === genCtx.assistantMessageId);
    if (!msg) return;

    const model = getModelById(genCtx.modelId);
    const parsed = parseThinkingOutput(genCtx.rawFullText, model.family, {
      thinkingExpectedInPrompt: genCtx.thinkingExpectedInPrompt,
    });
    msg.thinking = parsed.thinking;
    msg.content = parsed.content;
    msg.isThinkingInProgress = parsed.isThinkingInProgress;

    const msgEl = this.root.querySelector<HTMLElement>(
      `[data-message-id="${genCtx.assistantMessageId}"]`
    );
    const slot = msgEl?.querySelector<HTMLElement>('.assistant-content-slot');
    if (slot) {
      const existingDetails = slot.querySelector<HTMLDetailsElement>('details.thinking-trace');
      const preserveOpen =
        existingDetails && this.userToggledThinkingIds.has(genCtx.assistantMessageId)
          ? existingDetails.open
          : undefined;

      slot.innerHTML = renderAssistantBodyHtml(msg);

      if (preserveOpen !== undefined) {
        const nextDetails = slot.querySelector<HTMLDetailsElement>('details.thinking-trace');
        if (nextDetails) {
          nextDetails.open = preserveOpen;
        }
      }
    }

    const chatContainer = this.root.querySelector<HTMLElement>('#chat-messages');
    if (chatContainer) {
      chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    this.renderTelemetryHud();
    this.syncActionButtons();
    this.announcer?.announcePolite(`生成中: ${genCtx.tokenCount} トークン`);
  }

  handleAbortGeneration(): void {
    if (!this.activeGen) return;
    this.client.abort();
    this.announcer?.announcePolite('推論の中断をリクエストしました。', { immediate: true });
  }

  private recordExistingMessageToBenchmark(messageId: string): void {
    const idx = this.messages.findIndex((m) => m.id === messageId);
    if (idx === -1) return;
    const assistantMsg = this.messages[idx]!;
    if (!assistantMsg.metrics || this.recordedMessageIds.has(assistantMsg.id)) return;

    const prevUser = [...this.messages.slice(0, idx)].reverse().find((m) => m.role === 'user');
    const promptText = prevUser?.content ?? '';
    const matchingPreset = BENCHMARK_PRESETS.find((p) => p.prompt.trim() === promptText.trim());
    const device =
      this.devices.find((d) => d.id === assistantMsg.deviceId) ??
      this.loadedState?.device ??
      this.getSelectedDevice();

    if (!device) return;

    this.saveBenchmarkRecord({
      assistantMsg,
      promptId: matchingPreset?.id,
      promptLabel: matchingPreset?.label ?? '手動記録',
      promptText,
      device,
      dtype: assistantMsg.dtype ?? assistantMsg.metrics.load.dtype,
      generationConfig: this.generationConfig,
      status: assistantMsg.aborted ? 'aborted' : 'completed',
      metrics: assistantMsg.metrics,
    });

    this.renderChatMessages();
    this.renderComparison();
    this.announcer?.announcePolite('比較ベンチマークに記録しました。', { immediate: true });
  }

  private saveBenchmarkRecord(input: {
    assistantMsg: ChatMessage;
    promptId?: string;
    promptLabel: string;
    promptText: string;
    device: DeviceOption;
    dtype: QuantDtype;
    generationConfig: GenerationConfig;
    status: 'completed' | 'aborted' | 'error';
    metrics: InferenceMetrics;
  }): void {
    const model = getModelById(input.assistantMsg.modelId ?? this.selectedModelId);
    const record: BenchmarkRecord = {
      id: `bench-${input.assistantMsg.id}`,
      timestamp: Date.now(),
      ...(input.promptId ? { promptId: input.promptId } : {}),
      promptLabel: input.promptLabel,
      promptText: input.promptText,
      modelId: model.id,
      modelName: model.name,
      deviceId: input.device.id,
      deviceLabel: input.device.label,
      backend: input.device.backend,
      ...(input.device.gpuPreference ? { gpuPreference: input.device.gpuPreference } : {}),
      dtype: input.dtype,
      generationConfig: input.generationConfig,
      status: input.status,
      outputPreview: input.assistantMsg.content.slice(0, 240),
      ...(input.assistantMsg.thinking
        ? { thinkingPreview: input.assistantMsg.thinking.slice(0, 240) }
        : {}),
      metrics: input.metrics,
    };

    this.benchmarkStore.add(record);
    this.recordedMessageIds.add(input.assistantMsg.id);
  }

  private handleUnexpectedRuntimeError(message: string): void {
    if (this.disposed) return;
    this.loadedState = null;
    this.statusBanner = {
      kind: 'danger',
      text: `✕ ランタイム通知: ${message}`,
    };
    this.syncModelControls();
    this.announcer?.announceAssertive(message);
  }

  private async copyWithFeedback(button: HTMLButtonElement, text: string): Promise<void> {
    const copyFn = this.deps.copyTextFn ?? defaultCopyText;
    const orig = button.textContent ?? 'コピー';
    try {
      await copyFn(text);
      button.textContent = '✓ コピー済';
      setTimeout(() => {
        if (!this.disposed) {
          button.textContent = orig;
        }
      }, 1500);
    } catch {
      button.textContent = 'コピー失敗';
    }
  }

  private nextMessageId(prefix: string): string {
    this.messageCounter += 1;
    return `${prefix}-${Date.now()}-${this.messageCounter}`;
  }
}
