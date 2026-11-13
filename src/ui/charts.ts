import type { BenchmarkRecord } from '../types.ts';
import { escapeHtml } from './markdown.ts';

export interface HorizontalBarItem {
  id: string;
  label: string;
  subLabel: string;
  value: number;
  formattedValue: string;
  isWinner: boolean;
}

export interface HorizontalBarChartSpec {
  id: string;
  title: string;
  description: string;
  unitLabel: string;
  barColorVar: string;
  direction: 'higher-is-better' | 'lower-is-better';
  items: HorizontalBarItem[];
}

function truncateLabel(text: string, maxChars: number): string {
  if (text.length <= maxChars) {
    return text;
  }
  return `${text.slice(0, Math.max(1, maxChars - 1))}…`;
}

/**
 * Render an accessible inline SVG horizontal bar chart with crisp labels,
 * value callouts, winner tags, and forced-colors compatibility (`data-bar`).
 */
export function renderHorizontalBarChartSvg(spec: HorizontalBarChartSpec): string {
  const titleId = `${spec.id}-title`;
  const descId = `${spec.id}-desc`;

  if (spec.items.length === 0) {
    return `<p class="hud-note">比較可能な完了データがありません。</p>`;
  }

  const maxValue = spec.items.reduce((max, item) => (item.value > max ? item.value : max), 0);
  const safeMax = maxValue > 0 ? maxValue : 1;

  const svgWidth = 560;
  const labelColWidth = 176;
  const valueColWidth = 98;
  const barMaxWidth = svgWidth - labelColWidth - valueColWidth - 20;
  const rowHeight = 42;
  const topPad = 14;
  const bottomPad = 14;
  const svgHeight = topPad + spec.items.length * rowHeight + bottomPad;

  const rowsMarkup = spec.items
    .map((item, idx) => {
      const y = topPad + idx * rowHeight;
      const ratio = Math.max(0.03, Math.min(1, item.value / safeMax));
      const barWidth = Math.round(ratio * barMaxWidth);
      const shortLabel = truncateLabel(item.label, 22);
      const shortSub = truncateLabel(item.subLabel, 26);
      const winnerPrefix = item.isWinner ? '★ ' : '';
      const barOpacity = item.isWinner ? '1' : '0.78';

      return `
        <g transform="translate(0, ${y})">
          <title>${escapeHtml(`${item.label} (${item.subLabel}): ${item.formattedValue}`)}</title>
          <text x="6" y="14" font-size="11.5" font-weight="${item.isWinner ? '700' : '600'}" fill="currentColor">${escapeHtml(`${winnerPrefix}${shortLabel}`)}</text>
          <text x="6" y="29" font-size="10" fill="var(--text-muted, #64748b)">${escapeHtml(shortSub)}</text>
          <rect x="${labelColWidth}" y="6" width="${barMaxWidth}" height="22" rx="4" fill="var(--bg-subtle, #e2e8f0)" />
          <rect data-bar="true" x="${labelColWidth}" y="6" width="${barWidth}" height="22" rx="4" fill="${spec.barColorVar}" opacity="${barOpacity}" />
          <text x="${labelColWidth + barMaxWidth + 10}" y="21" font-size="11.5" font-family="var(--font-mono, monospace)" font-weight="700" fill="currentColor">${escapeHtml(item.formattedValue)}</text>
        </g>`;
    })
    .join('');

  return `
    <svg
      class="chart-svg"
      viewBox="0 0 ${svgWidth} ${svgHeight}"
      role="img"
      aria-labelledby="${escapeHtml(titleId)} ${escapeHtml(descId)}"
    >
      <title id="${escapeHtml(titleId)}">${escapeHtml(spec.title)}</title>
      <desc id="${escapeHtml(descId)}">${escapeHtml(spec.description)}</desc>
      ${rowsMarkup}
    </svg>`;
}

/**
 * Build the 3 core benchmark comparison charts (Speed tok/s, TTFT ms, Efficiency tokens/Wh)
 * from completed benchmark records.
 */
export function buildBenchmarkComparisonCharts(records: readonly BenchmarkRecord[]): string {
  const completed = records
    .filter((r) => r.status === 'completed' && r.metrics.speed.generatedTokens > 0)
    .slice(0, 10);

  if (completed.length === 0) {
    return `
      <div class="notice-box">
        <span>完了したベンチマーク記録がまだありません。「チャット &amp; ライブ計測」から実行するとここに比較チャートが表示されます。</span>
      </div>`;
  }

  const fastestSpeed = Math.max(...completed.map((r) => r.metrics.speed.tokensPerSec));
  const lowestTtft = Math.min(...completed.map((r) => r.metrics.speed.ttftMs));
  const effCandidates = completed
    .map((r) => r.metrics.power.tokensPerWattHour)
    .filter((v): v is number => typeof v === 'number' && Number.isFinite(v) && v > 0);
  const bestEff = effCandidates.length > 0 ? Math.max(...effCandidates) : 0;

  const makeMeta = (r: BenchmarkRecord): string =>
    `${r.backend === 'webgpu' ? 'WebGPU' : 'CPU WASM'} / ${r.dtype} / ${r.promptLabel}`;

  const speedChart = renderHorizontalBarChartSvg({
    id: 'chart-speed-tps',
    title: '応答速度比較 (Overall tok/s)',
    description: '各実行の総合スループット (tok/s、高いほど高速)',
    unitLabel: 'tok/s',
    barColorVar: 'var(--chart-bar-speed)',
    direction: 'higher-is-better',
    items: completed.map((r) => ({
      id: r.id,
      label: r.modelName,
      subLabel: makeMeta(r),
      value: r.metrics.speed.tokensPerSec,
      formattedValue: `${r.metrics.speed.tokensPerSec.toFixed(1)} t/s`,
      isWinner: r.metrics.speed.tokensPerSec === fastestSpeed && fastestSpeed > 0,
    })),
  });

  const ttftChart = renderHorizontalBarChartSvg({
    id: 'chart-ttft-ms',
    title: '初回トークン遅延比較 (TTFT ms)',
    description: '初回トークンまでの待ち時間 (ms、低いほど高速)',
    unitLabel: 'ms',
    barColorVar: 'var(--chart-bar-ttft)',
    direction: 'lower-is-better',
    items: completed.map((r) => ({
      id: r.id,
      label: r.modelName,
      subLabel: makeMeta(r),
      value: r.metrics.speed.ttftMs,
      formattedValue: `${Math.round(r.metrics.speed.ttftMs)} ms`,
      isWinner: r.metrics.speed.ttftMs === lowestTtft,
    })),
  });

  const effItems: HorizontalBarItem[] = completed
    .filter((r) => typeof r.metrics.power.tokensPerWattHour === 'number' && r.metrics.power.tokensPerWattHour > 0)
    .map((r) => {
      const val = r.metrics.power.tokensPerWattHour ?? 0;
      return {
        id: r.id,
        label: r.modelName,
        subLabel: makeMeta(r),
        value: val,
        formattedValue: `${Math.round(val)} t/Wh`,
        isWinner: val === bestEff && bestEff > 0,
      };
    });

  const effChart = renderHorizontalBarChartSvg({
    id: 'chart-efficiency-twh',
    title: '電力効率比較 (tokens/Wh)',
    description: '1Whあたりの生成トークン数 (高いほど省電力)',
    unitLabel: 'tokens/Wh',
    barColorVar: 'var(--chart-bar-eff)',
    direction: 'higher-is-better',
    items: effItems,
  });

  return `
    <div class="charts-grid">
      <section class="chart-card" aria-labelledby="chart-heading-speed">
        <h3 id="chart-heading-speed">応答速度比較 (Overall tok/s・高いほど高速)</h3>
        ${speedChart}
      </section>
      <section class="chart-card" aria-labelledby="chart-heading-ttft">
        <h3 id="chart-heading-ttft">初回トークン遅延 TTFT (ms・低いほど低遅延)</h3>
        ${ttftChart}
      </section>
      <section class="chart-card" aria-labelledby="chart-heading-eff">
        <h3 id="chart-heading-eff">電力効率比較 (tokens/Wh・高いほど省電力)</h3>
        ${effChart}
      </section>
    </div>`;
}
