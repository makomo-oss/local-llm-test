import { formatBytes, getResolvedDownloadBytes } from '../models/catalog.ts';
import type {
  BenchmarkComparisonResult,
  BenchmarkSummaryStats,
} from '../telemetry/benchmarkStore.ts';
import type { LiveTelemetrySnapshot } from '../telemetry/metricsTracker.ts';
import type { BenchmarkPromptPreset } from '../telemetry/presets.ts';
import type { BatterySnapshot } from '../telemetry/systemProbe.ts';
import type {
  BenchmarkRecord,
  ChatMessage,
  DeviceOption,
  InferenceMetrics,
  ModelCacheInfo,
  ModelId,
  ModelLoadProgress,
  ModelLoadStage,
  ModelSpec,
  QuantDtype,
  ResolvedDtype,
} from '../types.ts';
import { buildBenchmarkComparisonCharts } from './charts.ts';
import { escapeHtml, renderSafeMarkdown } from './markdown.ts';

export function formatContextLength(tokens: number): string {
  if (tokens >= 1024) {
    return `${Math.round(tokens / 1024)}K`;
  }
  return String(tokens);
}

export function formatDurationMs(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) {
    return '0 ms';
  }
  if (ms >= 1000) {
    return `${(ms / 1000).toFixed(2)} s`;
  }
  return `${Math.round(ms)} ms`;
}

export function formatSecondsDuration(seconds: number | undefined): string {
  if (seconds === undefined || !Number.isFinite(seconds) || seconds < 0) {
    return '-';
  }
  const mins = Math.floor(seconds / 60);
  const hours = Math.floor(mins / 60);
  const remMins = mins % 60;
  if (hours > 0) {
    return `${hours}時間${remMins}分`;
  }
  if (mins > 0) {
    return `${mins}分`;
  }
  return `${Math.round(seconds)}秒`;
}

export function formatLoadStageLabel(stage: ModelLoadStage): string {
  switch (stage) {
    case 'probing-device':
      return 'デバイス確認';
    case 'downloading':
      return 'ダウンロード中';
    case 'compiling-shaders':
    case 'initializing-session':
      return 'シェーダー準備・セッション初期化';
    case 'ready':
      return '準備完了';
  }
}

export function formatPowerMethodLabel(
  method: 'battery-api+utilization' | 'utilization-model'
): string {
  return method === 'battery-api+utilization'
    ? '実測バッテリー情報 + 負荷推定'
    : '負荷ベース推定モデル';
}

export function formatBatterySummary(battery: BatterySnapshot | undefined): string {
  if (!battery || !battery.supported) {
    return 'バッテリー: 非対応 / 取得不可';
  }
  const levelStr = battery.level !== undefined ? `${battery.level.toFixed(0)}%` : '-';
  const stateStr =
    battery.charging === true ? '充電中' : battery.charging === false ? '放電中' : '状態不明';
  return `バッテリー: ${levelStr} (${stateStr})`;
}

export interface ModelCardsViewInput {
  models: readonly ModelSpec[];
  selectedModelId: ModelId;
  loadedModelId: ModelId | null;
  selectedDevice: DeviceOption | undefined;
  requestedDtype: QuantDtype | 'auto';
  useExternalDataFormat: boolean;
  cacheMap: ReadonlyMap<ModelId, ModelCacheInfo>;
  resolveDtypeFn: (
    model: ModelSpec,
    device: Pick<DeviceOption, 'backend' | 'supportsFp16' | 'features'>,
    requestedDtype?: QuantDtype | 'auto',
    options?: { useExternalDataFormat?: boolean }
  ) => ResolvedDtype;
}

export function renderModelCardsHtml(input: ModelCardsViewInput): string {
  const fallbackDevice: Pick<DeviceOption, 'backend' | 'supportsFp16' | 'features'> =
    input.selectedDevice ?? {
      backend: 'webgpu',
      supportsFp16: true,
      features: ['shader-f16'],
    };

  return input.models
    .map((model) => {
      const selected = model.id === input.selectedModelId;
      const loaded = model.id === input.loadedModelId;
      const resolved = input.resolveDtypeFn(model, fallbackDevice, input.requestedDtype, {
        useExternalDataFormat: input.useExternalDataFormat,
      });
      const downloadBytes = getResolvedDownloadBytes(model, resolved, true);
      const cache = input.cacheMap.get(model.id);
      const cacheBadge = cache?.cached
        ? `<span class="chip chip--cached">✓ キャッシュ済 (${escapeHtml(formatBytes(cache.cachedBytes))})</span>`
        : '<span class="chip">未ダウンロード</span>';

      return `
        <button
          type="button"
          class="model-card"
          data-select-model="${escapeHtml(model.id)}"
          aria-pressed="${selected ? 'true' : 'false'}"
        >
          <div class="model-card-top">
            <span class="model-card-name">${escapeHtml(model.name)}</span>
            <span class="model-card-size">${escapeHtml(formatBytes(downloadBytes))}</span>
          </div>
          <div class="model-card-meta">
            <span class="chip chip--accent">${escapeHtml(model.parameterLabel)}</span>
            <span class="chip">${escapeHtml(formatContextLength(model.contextLength))} ctx</span>
            <span class="chip">${model.supportsThinking ? 'Thinking対応' : '非思考型'}</span>
            ${cacheBadge}
            ${loaded ? '<span class="chip chip--loaded">ロード済</span>' : ''}
          </div>
        </button>`;
    })
    .join('');
}

export function renderDeviceDetailsHtml(device: DeviceOption | undefined): string {
  if (!device) {
    return '<p class="notice-box notice-box--warn">利用可能なデバイスを検出中、または未検出です。</p>';
  }

  if (!device.available) {
    return `
      <div class="notice-box notice-box--warn">
        <strong>${escapeHtml(device.label)}</strong>
        <span>${escapeHtml(device.unavailableReason ?? '利用できません。')}</span>
      </div>`;
  }

  if (device.backend === 'webgpu') {
    const info = device.adapterInfo;
    const maxBuffer = device.limits.maxBufferSize
      ? formatBytes(device.limits.maxBufferSize)
      : '-';
    const maxStorage = device.limits.maxStorageBufferBindingSize
      ? formatBytes(device.limits.maxStorageBufferBindingSize)
      : '-';
    return `
      <dl class="spec-grid">
        <dt>Vendor / Arch</dt>
        <dd>${escapeHtml([info?.vendor || '-', info?.architecture || '-'].join(' / '))}</dd>
        <dt>Device / Desc</dt>
        <dd>${escapeHtml(info?.description || info?.device || '-')}</dd>
        <dt>shader-f16</dt>
        <dd>${device.supportsFp16 ? '✓ 対応 (q4f16推奨)' : '⚠ 非対応 (q4使用)'}</dd>
        <dt>maxBufferSize</dt>
        <dd>${escapeHtml(maxBuffer)}</dd>
        <dt>maxStorageBinding</dt>
        <dd>${escapeHtml(maxStorage)}</dd>
      </dl>`;
  }

  return `
    <dl class="spec-grid">
      <dt>論理コア数</dt>
      <dd>${device.runtimeDetails.hardwareConcurrency}</dd>
      <dt>WASM SIMD</dt>
      <dd>${device.runtimeDetails.simdSupported ? '✓ 対応' : '非対応'}</dd>
      <dt>実行スレッド</dt>
      <dd>${device.runtimeDetails.numThreads} スレッド</dd>
      <dt>COI状態</dt>
      <dd>${device.runtimeDetails.crossOriginIsolated ? 'COI有効' : '非COI (単一スレッド)'}</dd>
    </dl>`;
}

export function renderLoadProgressHtml(progress: ModelLoadProgress | null): string {
  if (!progress) {
    return '';
  }

  const stageLabel = formatLoadStageLabel(progress.stage);
  const pct = Math.min(100, Math.max(0, Math.round(progress.overallProgressPct * 10) / 10));
  const bytesText =
    progress.expectedTotalBytes > 0
      ? `${formatBytes(progress.loadedBytes)} / ${formatBytes(progress.expectedTotalBytes)}`
      : `${formatBytes(progress.loadedBytes)}`;

  const filesHtml = progress.files
    .slice(0, 8)
    .map((file) => {
      const fileBytes =
        file.totalBytes > 0
          ? `${formatBytes(file.loadedBytes)} / ${formatBytes(file.totalBytes)}`
          : `${Math.round(file.progressPct)}%`;
      return `
        <li class="file-progress-item">
          <span class="file-progress-name" title="${escapeHtml(file.file)}">${escapeHtml(file.file)}</span>
          <span>${escapeHtml(fileBytes)}</span>
        </li>`;
    })
    .join('');

  return `
    <div class="progress-panel" role="status" aria-label="モデルロード進捗">
      <div class="progress-header">
        <span class="chip chip--accent">${escapeHtml(stageLabel)}</span>
        <span>${pct.toFixed(1)}% (${escapeHtml(bytesText)})</span>
      </div>
      <progress class="studio-progress" max="100" value="${pct}">${pct}%</progress>
      <div class="section-subtitle">${escapeHtml(progress.message)}</div>
      ${filesHtml ? `<ul class="file-progress-list">${filesHtml}</ul>` : ''}
    </div>`;
}

export function renderTelemetryHudHtml(input: {
  liveSnapshot: LiveTelemetrySnapshot | null;
  finalMetrics: InferenceMetrics | null;
  liveBattery: BatterySnapshot | undefined;
  isGenerating: boolean;
  selectedDevice: DeviceOption | undefined;
}): string {
  const { liveSnapshot, finalMetrics, liveBattery, isGenerating, selectedDevice } = input;

  // Pillar 1: Speed
  let heroSpeed = '-';
  let overallTps = '-';
  let instantRollingTps = '-';
  let ttft = '-';
  let itl = '-';
  let breakdown = '-';
  let tokenCounts = '-';

  if (isGenerating && liveSnapshot) {
    const s = liveSnapshot.speed;
    heroSpeed = s.decodeTokensPerSec > 0 ? s.decodeTokensPerSec.toFixed(1) : '0.0';
    overallTps = `${s.overallTokensPerSec.toFixed(1)} tok/s`;
    instantRollingTps = `${s.instantTokensPerSec.toFixed(1)} / ${s.rollingTokensPerSec.toFixed(1)}`;
    ttft = s.ttftMs !== undefined ? formatDurationMs(s.ttftMs) : '待機中...';
    itl = `${s.interTokenLatencyAvgMs.toFixed(1)} / ${s.interTokenLatencyP95Ms.toFixed(1)} ms`;
    breakdown = `${formatDurationMs(s.elapsedMs)} 経過`;
    tokenCounts = `${s.promptTokens} / ${s.generatedTokens}`;
  } else if (finalMetrics) {
    const s = finalMetrics.speed;
    heroSpeed = (s.decodeTokensPerSec > 0 ? s.decodeTokensPerSec : s.tokensPerSec).toFixed(1);
    overallTps = `${s.tokensPerSec.toFixed(1)} tok/s`;
    instantRollingTps = '完了 (確定値)';
    ttft = formatDurationMs(s.ttftMs);
    itl = `${s.interTokenLatencyAvgMs.toFixed(1)} / ${s.interTokenLatencyP95Ms.toFixed(1)} ms`;
    breakdown = `${Math.round(s.prefillMs)} / ${Math.round(s.decodeMs)} / ${Math.round(s.totalMs)} ms`;
    tokenCounts = `${s.promptTokens} / ${s.generatedTokens}`;
  }

  // Pillar 2: Load
  let heroVram = '-';
  let jsHeapText = '未計測';
  let uaMemText = '-';
  let frameMsText = '-';
  let cpuBusyText = '-';
  let dropsLongText = '-';
  let backendText = selectedDevice
    ? `${selectedDevice.backend} / ${selectedDevice.recommendedDtype} / ${selectedDevice.runtimeDetails.numThreads}T`
    : '-';
  let heapUnavailable = true;

  if (isGenerating && liveSnapshot) {
    heroVram = formatBytes(liveSnapshot.estimatedVramBytes);
    if (liveSnapshot.memory.jsHeapUsedBytes !== undefined) {
      heapUnavailable = false;
      jsHeapText = formatBytes(liveSnapshot.memory.jsHeapUsedBytes);
    } else {
      jsHeapText = '非公開 (非対応)';
    }
    uaMemText =
      liveSnapshot.memory.uaMemoryBytes !== undefined
        ? formatBytes(liveSnapshot.memory.uaMemoryBytes)
        : 'COI時のみ';
    frameMsText = `${liveSnapshot.load.mainThreadAvgFrameMs.toFixed(1)} ms`;
    cpuBusyText = `${(liveSnapshot.load.mainThreadCpuBusyRatio * 100).toFixed(1)}%`;
    dropsLongText = `${liveSnapshot.load.droppedFrames} / ${liveSnapshot.load.longTaskCount}`;
  } else if (finalMetrics) {
    const l = finalMetrics.load;
    heroVram = formatBytes(l.estimatedVramBytes);
    if (l.jsHeapUsedBytes !== undefined) {
      heapUnavailable = false;
      jsHeapText = formatBytes(l.jsHeapUsedBytes);
    } else {
      jsHeapText = '非公開 (非対応)';
    }
    uaMemText = l.uaMemoryBytes !== undefined ? formatBytes(l.uaMemoryBytes) : 'COI時のみ';
    frameMsText = `${l.mainThreadAvgFrameMs.toFixed(1)} ms`;
    cpuBusyText = `${(l.mainThreadCpuBusyRatio * 100).toFixed(1)}%`;
    dropsLongText = `${l.droppedFrames} / ${l.longTaskCount}`;
    backendText = `${l.backend} / ${l.dtype} / ${l.activeThreads}T`;
  }

  // Pillar 3: Power
  const activePower = isGenerating ? liveSnapshot?.power : finalMetrics?.power;
  const heroPower = activePower ? activePower.estimatedPowerWatts.toFixed(1) : '-';
  const energyText = activePower
    ? `${(activePower.estimatedEnergyWh * 1000).toFixed(2)} mWh (${activePower.estimatedEnergyWh.toFixed(5)} Wh)`
    : '-';
  const efficiencyText =
    activePower?.tokensPerWattHour !== undefined
      ? `${Math.round(activePower.tokensPerWattHour)} tokens/Wh`
      : '-';
  const methodLabel = activePower
    ? formatPowerMethodLabel(activePower.estimationMethod)
    : '負荷ベース推定モデル';

  const batSupported = activePower ? activePower.batterySupported : Boolean(liveBattery?.supported);
  const batLevel = activePower?.batteryLevelEnd ?? activePower?.batteryLevelStart ?? liveBattery?.level;
  const batCharging = activePower?.batteryCharging ?? liveBattery?.charging;
  const batDischargeSec = activePower?.batteryDischargingTimeSec ?? liveBattery?.dischargingTimeSec;
  const batDelta = activePower?.batteryDeltaPct;

  const batteryStateText = !batSupported
    ? '非対応 (AC/推定のみ)'
    : `${batLevel !== undefined ? `${batLevel.toFixed(1)}%` : '-'} (${
        batCharging === true ? '充電中' : batCharging === false ? '放電中' : '不明'
      })`;
  const batteryDetailText = !batSupported
    ? '-'
    : `${formatSecondsDuration(batDischargeSec)} / ${
        batDelta !== undefined ? `${batDelta > 0 ? '+' : ''}${batDelta.toFixed(2)}%` : '-'
      }`;

  return `
    <div class="hud-pillars">
      <section class="hud-pillar" aria-labelledby="hud-speed-title">
        <div class="hud-pillar-header">
          <h3 id="hud-speed-title" class="hud-pillar-title">⚡ 応答速度 (Speed)</h3>
          <span class="chip">${isGenerating ? 'リアルタイム計測中' : finalMetrics ? '計測完了' : '待機中'}</span>
        </div>
        <div class="hud-hero">
          <span class="hud-hero-value">${escapeHtml(heroSpeed)}</span>
          <span class="hud-hero-unit">Decode tok/s</span>
        </div>
        <dl class="hud-metrics-grid">
          <dt>Overall tok/s</dt><dd>${escapeHtml(overallTps)}</dd>
          <dt>瞬間 / 移動平均</dt><dd>${escapeHtml(instantRollingTps)}</dd>
          <dt>TTFT (初回遅延)</dt><dd>${escapeHtml(ttft)}</dd>
          <dt>ITL 平均 / P95</dt><dd>${escapeHtml(itl)}</dd>
          <dt>Prefill / Decode / 計</dt><dd>${escapeHtml(breakdown)}</dd>
          <dt>トークン (入 / 出)</dt><dd>${escapeHtml(tokenCounts)}</dd>
        </dl>
      </section>

      <section class="hud-pillar" aria-labelledby="hud-load-title">
        <div class="hud-pillar-header">
          <h3 id="hud-load-title" class="hud-pillar-title">🧠 負荷・メモリ (Load)</h3>
          <span class="chip">${escapeHtml(backendText)}</span>
        </div>
        <div class="hud-hero">
          <span class="hud-hero-value">${escapeHtml(heroVram)}</span>
          <span class="hud-hero-unit">推定モデル+KV</span>
        </div>
        <dl class="hud-metrics-grid">
          <dt>JS Heap (Worker/Main)</dt><dd>${escapeHtml(jsHeapText)}</dd>
          <dt>UA Memory</dt><dd>${escapeHtml(uaMemText)}</dd>
          <dt>平均フレーム時間</dt><dd>${escapeHtml(frameMsText)}</dd>
          <dt>メインスレッド負荷</dt><dd>${escapeHtml(cpuBusyText)}</dd>
          <dt>ドロップ / LongTask</dt><dd>${escapeHtml(dropsLongText)}</dd>
        </dl>
        ${
          heapUnavailable
            ? '<p class="hud-note">※ JS Heap は Chrome / Edge など <code>performance.memory</code> 対応ブラウザで表示されます。</p>'
            : ''
        }
      </section>

      <section class="hud-pillar" aria-labelledby="hud-power-title">
        <div class="hud-pillar-header">
          <h3 id="hud-power-title" class="hud-pillar-title">🔋 電力消費 (Power)</h3>
          <span class="chip ${activePower?.estimationMethod === 'battery-api+utilization' ? 'chip--cached' : ''}">${escapeHtml(methodLabel)}</span>
        </div>
        <div class="hud-hero">
          <span class="hud-hero-value">${escapeHtml(heroPower)}</span>
          <span class="hud-hero-unit">推定 W</span>
        </div>
        <dl class="hud-metrics-grid">
          <dt>バッテリー状態</dt><dd>${escapeHtml(batteryStateText)}</dd>
          <dt>残放電時間 / 変化</dt><dd>${escapeHtml(batteryDetailText)}</dd>
          <dt>消費電力量</dt><dd>${escapeHtml(energyText)}</dd>
          <dt>電力効率</dt><dd>${escapeHtml(efficiencyText)}</dd>
        </dl>
      </section>
    </div>`;
}

export function renderEmptyGuideHtml(presets: readonly BenchmarkPromptPreset[]): string {
  const chipsHtml = presets
    .map(
      (preset) => `
        <button type="button" class="preset-chip" data-apply-preset="${escapeHtml(preset.id)}">
          ${escapeHtml(preset.label)}
        </button>`
    )
    .join('');

  return `
    <div class="empty-guide">
      <h2>完全ブラウザ完結型 ローカル LLM 推論スタジオへようこそ</h2>
      <ol class="guide-steps">
        <li>設定パネル（PCは左欄・スマホは下部「⚙ 設定」タブ）で <strong>モデル (6種類)</strong> と <strong>実行デバイス (WebGPU GPU / CPU WASM)</strong> を選択します。</li>
        <li><strong>「モデルをロード / ダウンロード」</strong> をクリックしてブラウザへ展開します（初回後はキャッシュから高速ロード）。</li>
        <li>下の入力欄から対話するか、<strong>「ベンチマーク実行 (1クリック計測)」</strong> で速度・負荷・電力効率を比較できます。</li>
      </ol>
      <div>
        <div class="field-label">クイックスタート・ベンチマークプリセット</div>
        <div class="preset-chips">${chipsHtml}</div>
      </div>
    </div>`;
}

export function renderAssistantBodyHtml(msg: ChatMessage): string {
  const thinkingSection =
    (msg.thinking && msg.thinking.length > 0) || msg.isThinkingInProgress
      ? `
        <details class="thinking-trace"${msg.isThinkingInProgress ? ' open' : ''}>
          <summary>${msg.isThinkingInProgress ? '⏳ 思考プロセス (生成中...)' : '💭 思考プロセス (Thinking Trace)'}</summary>
          <div class="thinking-trace-body">${escapeHtml(msg.thinking ?? '')}</div>
        </details>`
      : '';

  const contentHtml = msg.content
    ? `<div class="markdown-body">${renderSafeMarkdown(msg.content)}</div>`
    : msg.isStreaming
      ? '<div class="markdown-body"><p class="hud-note">トークン生成待機中...</p></div>'
      : '';

  const errorHtml = msg.error
    ? `<div class="notice-box notice-box--danger">⚠ ${escapeHtml(msg.error)}</div>`
    : '';

  const badgesHtml = msg.metrics
    ? `
      <div class="msg-telemetry-badges">
        <span class="chip chip--accent">⚡ ${
          (msg.metrics.speed.decodeTokensPerSec > 0
            ? msg.metrics.speed.decodeTokensPerSec
            : msg.metrics.speed.tokensPerSec
          ).toFixed(1)
        } tok/s・TTFT ${Math.round(msg.metrics.speed.ttftMs)} ms</span>
        <span class="chip">🧠 推定 ${escapeHtml(formatBytes(msg.metrics.load.estimatedVramBytes))}・Drop ${msg.metrics.load.droppedFrames}</span>
        <span class="chip">🔋 ${msg.metrics.power.estimatedPowerWatts.toFixed(1)} W${
          msg.metrics.power.tokensPerWattHour
            ? `・${Math.round(msg.metrics.power.tokensPerWattHour)} tok/Wh`
            : ''
        }</span>
      </div>`
    : '';

  return `${thinkingSection}${contentHtml}${errorHtml}${badgesHtml}`;
}

export function renderChatMessageElement(
  msg: ChatMessage,
  recordedMessageIds: ReadonlySet<string>
): HTMLElement {
  const article = document.createElement('article');
  article.className = `chat-message chat-message--${msg.role === 'user' ? 'user' : 'assistant'}`;
  article.dataset.messageId = msg.id;

  if (msg.role === 'user') {
    article.innerHTML = `
      <div class="chat-msg-header">
        <span class="chat-msg-role">👤 ユーザー</span>
      </div>
      <div class="markdown-body" style="white-space: pre-wrap;">${escapeHtml(msg.content)}</div>`;
    return article;
  }

  const metaTag = [msg.modelId, msg.dtype, msg.deviceId].filter(Boolean).join(' / ');
  const statusTag = msg.isStreaming
    ? '<span class="chip chip--accent">生成中...</span>'
    : msg.aborted
      ? '<span class="chip">中断済</span>'
      : msg.error
        ? '<span class="chip">エラー</span>'
        : '<span class="chip chip--cached">完了</span>';

  const canRecord = Boolean(msg.metrics && !msg.isStreaming && !recordedMessageIds.has(msg.id));

  article.innerHTML = `
    <div class="chat-msg-header">
      <span class="chat-msg-role">🤖 アシスタント ${metaTag ? `<span class="chip">${escapeHtml(metaTag)}</span>` : ''} ${statusTag}</span>
      <div class="chat-msg-actions">
        ${
          canRecord
            ? `<button type="button" class="btn btn--sm" data-record-message-id="${escapeHtml(msg.id)}">ベンチ記録</button>`
            : ''
        }
        <button type="button" class="btn btn--sm" data-copy-message-id="${escapeHtml(msg.id)}">コピー</button>
      </div>
    </div>
    <div class="assistant-content-slot">${renderAssistantBodyHtml(msg)}</div>`;

  return article;
}

export function renderComparisonViewHtml(input: {
  records: readonly BenchmarkRecord[];
  summary: BenchmarkSummaryStats;
  comparison: BenchmarkComparisonResult;
}): string {
  const { records, summary, comparison } = input;
  const { winners } = comparison;

  const formatWinnerSub = (
    entry: BenchmarkComparisonResult['winners']['fastestThroughput']
  ): string => {
    if (!entry) return '記録なし';
    return `${entry.record.modelName} (${entry.record.backend} / ${entry.record.dtype})`;
  };

  const rowsHtml =
    records.length === 0
      ? `<tr><td colspan="12" class="table-empty-cell">まだベンチマーク記録がありません。</td></tr>`
      : records
          .map((r) => {
            const { speed, load, power } = r.metrics;
            const statusText =
              r.status === 'completed'
                ? '✓ 完了'
                : r.status === 'aborted'
                  ? '■ 中断'
                  : '✕ エラー';
            return `
              <tr>
                <th scope="row">${escapeHtml(r.modelName)}<br><span class="hud-note">${escapeHtml(r.promptLabel)}</span></th>
                <td data-label="デバイス">${escapeHtml(r.deviceLabel)}</td>
                <td data-label="量子化">${escapeHtml(r.dtype)}</td>
                <td data-label="状態">${escapeHtml(statusText)}</td>
                <td data-label="Overall tok/s" class="num">${speed.tokensPerSec.toFixed(1)}</td>
                <td data-label="Decode tok/s" class="num">${speed.decodeTokensPerSec.toFixed(1)}</td>
                <td data-label="TTFT" class="num">${Math.round(speed.ttftMs)} ms</td>
                <td data-label="ITL 平均/P95" class="num">${speed.interTokenLatencyAvgMs.toFixed(1)} / ${speed.interTokenLatencyP95Ms.toFixed(1)} ms</td>
                <td data-label="推定メモリ" class="num">${escapeHtml(formatBytes(load.estimatedVramBytes))}</td>
                <td data-label="推定電力" class="num">${power.estimatedPowerWatts.toFixed(1)} W</td>
                <td data-label="電力効率" class="num">${power.tokensPerWattHour ? `${Math.round(power.tokensPerWattHour)} t/Wh` : '-'}</td>
                <td data-label="操作">
                  <button type="button" class="btn btn--sm btn--danger" data-delete-benchmark="${escapeHtml(r.id)}">削除</button>
                </td>
              </tr>`;
          })
          .join('');

  return `
    <div class="comparison-view">
      <section class="summary-cards" aria-label="ベンチマーク統計サマリー">
        <div class="summary-card">
          <span class="summary-card-label">総計測回数 (完了)</span>
          <span class="summary-card-value">${summary.totalRuns} (${summary.completedRuns})</span>
          <span class="summary-card-sub">${summary.uniqueModels} モデル / ${summary.uniqueDevices} デバイス</span>
        </div>
        <div class="summary-card winner-card">
          <span class="summary-card-label">🏆 最速スループット</span>
          <span class="summary-card-value">${winners.fastestThroughput ? `${winners.fastestThroughput.value.toFixed(1)} tok/s` : '-'}</span>
          <span class="summary-card-sub">${escapeHtml(formatWinnerSub(winners.fastestThroughput))}</span>
        </div>
        <div class="summary-card winner-card">
          <span class="summary-card-label">⚡ 最短 TTFT</span>
          <span class="summary-card-value">${winners.lowestTtft ? `${Math.round(winners.lowestTtft.value)} ms` : '-'}</span>
          <span class="summary-card-sub">${escapeHtml(formatWinnerSub(winners.lowestTtft))}</span>
        </div>
        <div class="summary-card winner-card">
          <span class="summary-card-label">🔋 最高電力効率</span>
          <span class="summary-card-value">${winners.bestEfficiency ? `${Math.round(winners.bestEfficiency.value)} tok/Wh` : '-'}</span>
          <span class="summary-card-sub">${escapeHtml(formatWinnerSub(winners.bestEfficiency))}</span>
        </div>
        <div class="summary-card winner-card">
          <span class="summary-card-label">💾 最小推定メモリ</span>
          <span class="summary-card-value">${winners.lowestMemory ? escapeHtml(formatBytes(winners.lowestMemory.value)) : '-'}</span>
          <span class="summary-card-sub">${escapeHtml(formatWinnerSub(winners.lowestMemory))}</span>
        </div>
      </section>

      ${buildBenchmarkComparisonCharts(records)}

      <section class="table-card" aria-labelledby="benchmark-table-heading">
        <div class="table-toolbar">
          <h2 id="benchmark-table-heading" class="section-title">計測履歴・比較テーブル (${records.length}件)</h2>
          <div class="btn-group">
            <button type="button" class="btn btn--sm" data-export-json ${records.length === 0 ? 'disabled' : ''}>JSON エクスポート</button>
            <button type="button" class="btn btn--sm" data-export-csv ${records.length === 0 ? 'disabled' : ''}>CSV エクスポート</button>
            <button type="button" class="btn btn--sm btn--danger" data-clear-benchmarks ${records.length === 0 ? 'disabled' : ''}>履歴全削除</button>
          </div>
        </div>
        <div tabindex="0" role="region" aria-labelledby="benchmark-table-heading" class="table-scroll-region">
          <table class="benchmark-table">
            <thead>
              <tr>
                <th scope="col">モデル / プロンプト</th>
                <th scope="col">デバイス</th>
                <th scope="col">量子化</th>
                <th scope="col">状態</th>
                <th scope="col">Overall tok/s</th>
                <th scope="col">Decode tok/s</th>
                <th scope="col">TTFT</th>
                <th scope="col">ITL 平均/P95</th>
                <th scope="col">推定メモリ</th>
                <th scope="col">推定電力</th>
                <th scope="col">電力効率</th>
                <th scope="col">操作</th>
              </tr>
            </thead>
            <tbody>
              ${rowsHtml}
            </tbody>
          </table>
        </div>
      </section>
    </div>`;
}
