import type { BenchmarkRecord } from '../types.ts';

export const BENCHMARK_STORAGE_KEY = 'local-llm-test:benchmarks:v1';

export interface StorageLike {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

export interface BenchmarkStoreOptions {
  storage?: StorageLike | null;
  storageKey?: string;
  maxRecords?: number;
}

export interface BenchmarkSummaryStats {
  totalRuns: number;
  completedRuns: number;
  abortedRuns: number;
  errorRuns: number;
  uniqueModels: number;
  uniqueDevices: number;
  avgTokensPerSec: number;
  maxTokensPerSec: number;
  avgDecodeTokensPerSec: number;
  avgTtftMs: number;
  minTtftMs: number;
  avgEstimatedPowerWatts: number;
  totalGeneratedTokens: number;
  totalEnergyWh: number;
}

export interface BenchmarkRankEntry {
  rank: number;
  record: BenchmarkRecord;
  value: number;
  /** Ratio vs best (1.0 = best). >1 for "higher is better" means worse? Always expressed relative to best (`value / best` or `best / value` via `ratioToBest` in (0..1] where 1.0 = best). */
  ratioToBest: number;
  deltaFromBestPct: number;
}

export interface BenchmarkComparisonResult {
  totalRecords: number;
  comparableCount: number;
  excludedCount: number;
  fastestThroughput: BenchmarkRankEntry[];
  lowestTtft: BenchmarkRankEntry[];
  lowestMemory: BenchmarkRankEntry[];
  bestEfficiency: BenchmarkRankEntry[];
  winners: {
    fastestThroughput?: BenchmarkRankEntry;
    lowestTtft?: BenchmarkRankEntry;
    lowestMemory?: BenchmarkRankEntry;
    bestEfficiency?: BenchmarkRankEntry;
  };
}

function roundTo(value: number, decimals: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  const factor = 10 ** decimals;
  return Math.round(value * factor) / factor;
}

function resolveDefaultStorage(): StorageLike | undefined {
  try {
    const storage = (globalThis as { localStorage?: StorageLike }).localStorage;
    if (
      storage &&
      typeof storage.getItem === 'function' &&
      typeof storage.setItem === 'function' &&
      typeof storage.removeItem === 'function'
    ) {
      return storage;
    }
  } catch {
    // Accessing localStorage can throw SecurityError in sandboxed contexts.
  }
  return undefined;
}

function isValidBenchmarkRecord(candidate: unknown): candidate is BenchmarkRecord {
  if (!candidate || typeof candidate !== 'object') {
    return false;
  }
  const r = candidate as Partial<BenchmarkRecord>;
  return (
    typeof r.id === 'string' &&
    r.id.length > 0 &&
    typeof r.timestamp === 'number' &&
    Number.isFinite(r.timestamp) &&
    typeof r.modelId === 'string' &&
    typeof r.deviceId === 'string' &&
    typeof r.dtype === 'string' &&
    typeof r.status === 'string' &&
    Boolean(r.metrics && typeof r.metrics === 'object' && r.metrics.speed && r.metrics.load && r.metrics.power)
  );
}

/**
 * RFC 4180 CSV field escaping: wraps values containing commas, quotes, CR, or LF
 * in double quotes and doubles internal double quotes.
 */
export function escapeCsvCell(value: string | number | boolean | undefined | null): string {
  if (value === undefined || value === null) {
    return '';
  }
  const str = String(value);
  if (str.includes('"') || str.includes(',') || str.includes('\n') || str.includes('\r')) {
    return `"${str.replace(/"/g, '""')}"`;
  }
  return str;
}

const CSV_HEADERS = [
  'id',
  'timestamp',
  'isoTime',
  'status',
  'promptId',
  'promptLabel',
  'modelId',
  'modelName',
  'deviceId',
  'deviceLabel',
  'backend',
  'gpuPreference',
  'dtype',
  'ttftMs',
  'prefillMs',
  'decodeMs',
  'totalMs',
  'promptTokens',
  'generatedTokens',
  'tokensPerSec',
  'decodeTokensPerSec',
  'interTokenLatencyAvgMs',
  'interTokenLatencyP95Ms',
  'estimatedVramBytes',
  'jsHeapUsedBytes',
  'uaMemoryBytes',
  'mainThreadAvgFrameMs',
  'mainThreadCpuBusyRatio',
  'droppedFrames',
  'longTaskCount',
  'activeThreads',
  'batterySupported',
  'batteryLevelStart',
  'batteryLevelEnd',
  'batteryCharging',
  'batteryDeltaPct',
  'estimatedPowerWatts',
  'estimatedEnergyWh',
  'tokensPerWattHour',
  'estimationMethod',
  'outputPreview',
  'error',
] as const;

function buildRankings(
  records: readonly BenchmarkRecord[],
  extractValue: (record: BenchmarkRecord) => number | undefined,
  direction: 'higher-is-better' | 'lower-is-better'
): BenchmarkRankEntry[] {
  const valid: Array<{ record: BenchmarkRecord; value: number }> = [];
  for (const record of records) {
    const val = extractValue(record);
    if (typeof val === 'number' && Number.isFinite(val)) {
      valid.push({ record, value: val });
    }
  }

  if (valid.length === 0) {
    return [];
  }

  valid.sort((a, b) => {
    const diff = direction === 'higher-is-better' ? b.value - a.value : a.value - b.value;
    if (diff !== 0) {
      return diff;
    }
    return b.record.timestamp - a.record.timestamp;
  });

  const bestValue = valid[0]!.value;

  return valid.map((item, idx) => {
    let ratioToBest = 1;
    let deltaFromBestPct = 0;

    if (bestValue > 0 && item.value > 0) {
      ratioToBest =
        direction === 'higher-is-better'
          ? roundTo(item.value / bestValue, 4)
          : roundTo(bestValue / item.value, 4);
      deltaFromBestPct = roundTo(((item.value - bestValue) / bestValue) * 100, 2);
    } else if (bestValue === 0 && item.value === 0) {
      ratioToBest = 1;
      deltaFromBestPct = 0;
    } else {
      ratioToBest = 0;
      deltaFromBestPct = 0;
    }

    return {
      rank: idx + 1,
      record: item.record,
      value: item.value,
      ratioToBest,
      deltaFromBestPct,
    };
  });
}

/**
 * Compute side-by-side performance, latency, memory, and energy efficiency rankings
 * across completed benchmark runs.
 */
export function computeBenchmarkComparison(
  records: readonly BenchmarkRecord[]
): BenchmarkComparisonResult {
  const comparable = records.filter(
    (record) => record.status === 'completed' && record.metrics.speed.generatedTokens > 0
  );

  const fastestThroughput = buildRankings(
    comparable,
    (r) => (r.metrics.speed.tokensPerSec > 0 ? r.metrics.speed.tokensPerSec : undefined),
    'higher-is-better'
  );
  const lowestTtft = buildRankings(
    comparable,
    (r) => (r.metrics.speed.ttftMs >= 0 ? r.metrics.speed.ttftMs : undefined),
    'lower-is-better'
  );
  const lowestMemory = buildRankings(
    comparable,
    (r) => (r.metrics.load.estimatedVramBytes > 0 ? r.metrics.load.estimatedVramBytes : undefined),
    'lower-is-better'
  );
  const bestEfficiency = buildRankings(
    comparable,
    (r) =>
      r.metrics.power.tokensPerWattHour !== undefined && r.metrics.power.tokensPerWattHour > 0
        ? r.metrics.power.tokensPerWattHour
        : undefined,
    'higher-is-better'
  );

  return {
    totalRecords: records.length,
    comparableCount: comparable.length,
    excludedCount: records.length - comparable.length,
    fastestThroughput,
    lowestTtft,
    lowestMemory,
    bestEfficiency,
    winners: {
      ...(fastestThroughput[0] ? { fastestThroughput: fastestThroughput[0] } : {}),
      ...(lowestTtft[0] ? { lowestTtft: lowestTtft[0] } : {}),
      ...(lowestMemory[0] ? { lowestMemory: lowestMemory[0] } : {}),
      ...(bestEfficiency[0] ? { bestEfficiency: bestEfficiency[0] } : {}),
    },
  };
}

/**
 * Compute aggregate summary statistics over a collection of benchmark runs.
 */
export function summarizeBenchmarks(records: readonly BenchmarkRecord[]): BenchmarkSummaryStats {
  let completedRuns = 0;
  let abortedRuns = 0;
  let errorRuns = 0;
  let totalGeneratedTokens = 0;
  let totalEnergyWh = 0;

  let tpsSum = 0;
  let decodeTpsSum = 0;
  let maxTokensPerSec = 0;
  let ttftSum = 0;
  let minTtftMs = Number.POSITIVE_INFINITY;
  let powerSum = 0;
  let speedSampleCount = 0;

  const uniqueModels = new Set<string>();
  const uniqueDevices = new Set<string>();

  for (const record of records) {
    uniqueModels.add(record.modelId);
    uniqueDevices.add(record.deviceId);

    if (record.status === 'completed') {
      completedRuns += 1;
    } else if (record.status === 'aborted') {
      abortedRuns += 1;
    } else {
      errorRuns += 1;
    }

    const { speed, power } = record.metrics;
    if (Number.isFinite(speed.generatedTokens) && speed.generatedTokens > 0) {
      totalGeneratedTokens += speed.generatedTokens;
    }
    if (Number.isFinite(power.estimatedEnergyWh) && power.estimatedEnergyWh > 0) {
      totalEnergyWh += power.estimatedEnergyWh;
    }

    if (record.status === 'completed' && speed.generatedTokens > 0) {
      speedSampleCount += 1;
      tpsSum += speed.tokensPerSec;
      decodeTpsSum += speed.decodeTokensPerSec;
      if (speed.tokensPerSec > maxTokensPerSec) {
        maxTokensPerSec = speed.tokensPerSec;
      }
      ttftSum += speed.ttftMs;
      if (speed.ttftMs < minTtftMs) {
        minTtftMs = speed.ttftMs;
      }
      powerSum += power.estimatedPowerWatts;
    }
  }

  return {
    totalRuns: records.length,
    completedRuns,
    abortedRuns,
    errorRuns,
    uniqueModels: uniqueModels.size,
    uniqueDevices: uniqueDevices.size,
    avgTokensPerSec: speedSampleCount > 0 ? roundTo(tpsSum / speedSampleCount, 2) : 0,
    maxTokensPerSec: roundTo(maxTokensPerSec, 2),
    avgDecodeTokensPerSec:
      speedSampleCount > 0 ? roundTo(decodeTpsSum / speedSampleCount, 2) : 0,
    avgTtftMs: speedSampleCount > 0 ? roundTo(ttftSum / speedSampleCount, 2) : 0,
    minTtftMs: Number.isFinite(minTtftMs) ? roundTo(minTtftMs, 2) : 0,
    avgEstimatedPowerWatts: speedSampleCount > 0 ? roundTo(powerSum / speedSampleCount, 2) : 0,
    totalGeneratedTokens,
    totalEnergyWh: roundTo(totalEnergyWh, 6),
  };
}

/**
 * Serialize benchmark records to RFC 4180 CSV with header row.
 */
export function serializeBenchmarksToCsv(records: readonly BenchmarkRecord[]): string {
  const lines: string[] = [CSV_HEADERS.join(',')];

  for (const r of records) {
    const { speed, load, power } = r.metrics;
    const row = [
      r.id,
      r.timestamp,
      Number.isFinite(r.timestamp) ? new Date(r.timestamp).toISOString() : '',
      r.status,
      r.promptId ?? '',
      r.promptLabel,
      r.modelId,
      r.modelName,
      r.deviceId,
      r.deviceLabel,
      r.backend,
      r.gpuPreference ?? '',
      r.dtype,
      speed.ttftMs,
      speed.prefillMs,
      speed.decodeMs,
      speed.totalMs,
      speed.promptTokens,
      speed.generatedTokens,
      speed.tokensPerSec,
      speed.decodeTokensPerSec,
      speed.interTokenLatencyAvgMs,
      speed.interTokenLatencyP95Ms,
      load.estimatedVramBytes,
      load.jsHeapUsedBytes ?? '',
      load.uaMemoryBytes ?? '',
      load.mainThreadAvgFrameMs,
      load.mainThreadCpuBusyRatio,
      load.droppedFrames,
      load.longTaskCount,
      load.activeThreads,
      power.batterySupported,
      power.batteryLevelStart ?? '',
      power.batteryLevelEnd ?? '',
      power.batteryCharging ?? '',
      power.batteryDeltaPct ?? '',
      power.estimatedPowerWatts,
      power.estimatedEnergyWh,
      power.tokensPerWattHour ?? '',
      power.estimationMethod,
      r.outputPreview,
      r.error ?? '',
    ];
    lines.push(row.map((cell) => escapeCsvCell(cell)).join(','));
  }

  return lines.join('\r\n');
}

/**
 * Persistent store for local LLM benchmark records backed by `localStorage`
 * (`'local-llm-test:benchmarks:v1'`) with safe error handling when storage is
 * full, corrupted, or unavailable.
 */
export class BenchmarkStore {
  private readonly storage: StorageLike | undefined;
  private readonly storageKey: string;
  private readonly maxRecords: number;
  private records: BenchmarkRecord[] = [];
  private lastStorageError: string | undefined;

  constructor(options: BenchmarkStoreOptions = {}) {
    this.storage =
      options.storage === null ? undefined : (options.storage ?? resolveDefaultStorage());
    this.storageKey = options.storageKey ?? BENCHMARK_STORAGE_KEY;
    this.maxRecords =
      typeof options.maxRecords === 'number' && options.maxRecords >= 1
        ? Math.round(options.maxRecords)
        : 200;

    this.loadFromStorage();
  }

  getStorageError(): string | undefined {
    return this.lastStorageError;
  }

  private loadFromStorage(): void {
    if (!this.storage) {
      return;
    }
    try {
      const raw = this.storage.getItem(this.storageKey);
      if (!raw) {
        return;
      }
      const parsed: unknown = JSON.parse(raw);
      if (!Array.isArray(parsed)) {
        return;
      }
      this.records = parsed
        .filter(isValidBenchmarkRecord)
        .sort((a, b) => b.timestamp - a.timestamp)
        .slice(0, this.maxRecords);
      this.lastStorageError = undefined;
    } catch (err) {
      this.records = [];
      this.lastStorageError = err instanceof Error ? err.message : 'Failed to read benchmark storage';
    }
  }

  private persist(): boolean {
    if (!this.storage) {
      return false;
    }
    try {
      this.storage.setItem(this.storageKey, JSON.stringify(this.records));
      this.lastStorageError = undefined;
      return true;
    } catch (err) {
      this.lastStorageError =
        err instanceof Error ? err.message : 'Failed to persist benchmark storage';
      return false;
    }
  }

  list(): BenchmarkRecord[] {
    return [...this.records];
  }

  add(record: BenchmarkRecord): BenchmarkRecord {
    const filtered = this.records.filter((item) => item.id !== record.id);
    filtered.unshift(record);
    filtered.sort((a, b) => b.timestamp - a.timestamp);
    if (filtered.length > this.maxRecords) {
      filtered.length = this.maxRecords;
    }
    this.records = filtered;
    this.persist();
    return record;
  }

  remove(id: string): boolean {
    const before = this.records.length;
    this.records = this.records.filter((item) => item.id !== id);
    if (this.records.length === before) {
      return false;
    }
    this.persist();
    return true;
  }

  clear(): void {
    this.records = [];
    if (!this.storage) {
      return;
    }
    try {
      this.storage.removeItem(this.storageKey);
      this.lastStorageError = undefined;
    } catch (err) {
      this.lastStorageError =
        err instanceof Error ? err.message : 'Failed to clear benchmark storage';
    }
  }

  getSummaryStats(records: readonly BenchmarkRecord[] = this.records): BenchmarkSummaryStats {
    return summarizeBenchmarks(records);
  }

  exportJson(records: readonly BenchmarkRecord[] = this.records): string {
    return JSON.stringify(
      {
        version: 1,
        storageKey: this.storageKey,
        exportedAt: new Date().toISOString(),
        summary: summarizeBenchmarks(records),
        records,
      },
      null,
      2
    );
  }

  exportCsv(records: readonly BenchmarkRecord[] = this.records): string {
    return serializeBenchmarksToCsv(records);
  }

  computeComparison(
    records: readonly BenchmarkRecord[] = this.records
  ): BenchmarkComparisonResult {
    return computeBenchmarkComparison(records);
  }
}
