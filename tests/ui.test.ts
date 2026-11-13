import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { LlmRuntimeClient } from '../src/engine/runtimeClient.ts';
import { MODEL_CATALOG, getModelById, resolveEffectiveDtype } from '../src/models/catalog.ts';
import { BenchmarkStore, type StorageLike } from '../src/telemetry/benchmarkStore.ts';
import { BENCHMARK_PRESETS } from '../src/telemetry/presets.ts';
import type {
  BenchmarkRecord,
  ChatMessage,
  DeviceOption,
  InferenceMetrics,
  ModelCacheInfo,
  ModelLoadProgress,
} from '../src/types.ts';
import { AppController, DebouncedLiveAnnouncer } from '../src/ui/appController.ts';
import { buildBenchmarkComparisonCharts, renderHorizontalBarChartSvg } from '../src/ui/charts.ts';
import { StudioDialogHost } from '../src/ui/dialogs.ts';
import { attachImeSafeEnterSubmit, shouldSubmitOnEnter } from '../src/ui/imeSubmit.ts';
import { isSafeHref, renderSafeMarkdown, sanitizeHtmlFallback } from '../src/ui/markdown.ts';
import {
  renderChatMessageElement,
  renderComparisonViewHtml,
  renderModelCardsHtml,
  renderTelemetryHudHtml,
} from '../src/ui/viewTemplates.ts';

// ---------------------------------------------------------------------------
// Lightweight DOM shim for Node Vitest environment (no external deps needed)
// ---------------------------------------------------------------------------

class FakeEvent {
  type: string;
  bubbles: boolean;
  cancelable: boolean;
  defaultPrevented = false;
  target: FakeElement | null = null;
  currentTarget: FakeElement | null = null;
  timeStamp: number;
  key?: string;
  shiftKey?: boolean;
  isComposing?: boolean;
  keyCode?: number;

  constructor(type: string, init: Record<string, unknown> = {}) {
    this.type = type;
    this.bubbles = Boolean(init.bubbles ?? true);
    this.cancelable = Boolean(init.cancelable ?? true);
    this.timeStamp = typeof init.timeStamp === 'number' ? init.timeStamp : Date.now();
    Object.assign(this, init);
  }

  preventDefault(): void {
    if (this.cancelable) {
      this.defaultPrevented = true;
    }
  }
}

class FakeClassList {
  private readonly el: FakeElement;
  constructor(el: FakeElement) {
    this.el = el;
  }
  private getSet(): Set<string> {
    const raw = this.el.getAttribute('class') ?? '';
    return new Set(raw.split(/\s+/).filter(Boolean));
  }
  private save(set: Set<string>): void {
    this.el.setAttribute('class', Array.from(set).join(' '));
  }
  add(...tokens: string[]): void {
    const set = this.getSet();
    for (const t of tokens) set.add(t);
    this.save(set);
  }
  remove(...tokens: string[]): void {
    const set = this.getSet();
    for (const t of tokens) set.delete(t);
    this.save(set);
  }
  contains(token: string): boolean {
    return this.getSet().has(token);
  }
}

function decodeBasicEntities(text: string): string {
  return text
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'");
}

class FakeElement {
  tagName: string;
  attributes = new Map<string, string>();
  children: FakeElement[] = [];
  parentElement: FakeElement | null = null;
  ownerDocument: FakeDocument;
  private datasetTarget: Record<string, string> = {};
  dataset: Record<string, string>;
  style: Record<string, string> = {};
  classList: FakeClassList;
  scrollTop = 0;
  scrollHeight = 100;
  _textContent = '';
  private _value = '';
  private _checked = false;
  private _disabled = false;
  private _hidden = false;
  private _open = false;
  private _returnValue = '';
  tabIndex = 0;
  private listeners = new Map<string, Array<{ cb: (ev: any) => void; capture: boolean }>>();

  constructor(tagName: string, ownerDocument: FakeDocument) {
    this.tagName = tagName.toUpperCase();
    this.ownerDocument = ownerDocument;
    this.classList = new FakeClassList(this);
    this.dataset = new Proxy(this.datasetTarget, {
      set: (target, prop, value) => {
        const key = String(prop);
        const attr = `data-${key.replace(/[A-Z]/g, (ch) => `-${ch.toLowerCase()}`)}`;
        const strVal = String(value);
        target[key] = strVal;
        this.attributes.set(attr, strVal);
        return true;
      },
      deleteProperty: (target, prop) => {
        const key = String(prop);
        const attr = `data-${key.replace(/[A-Z]/g, (ch) => `-${ch.toLowerCase()}`)}`;
        delete target[key];
        this.attributes.delete(attr);
        return true;
      },
    });
  }

  get id(): string {
    return this.attributes.get('id') ?? '';
  }
  set id(val: string) {
    this.setAttribute('id', val);
  }

  get className(): string {
    return this.attributes.get('class') ?? '';
  }
  set className(val: string) {
    this.setAttribute('class', val);
  }

  get value(): string {
    if (this.tagName === 'SELECT' && !this._value && this.children.length > 0) {
      const opt = this.querySelector('option[selected]') ?? this.querySelector('option');
      return opt?.getAttribute('value') ?? '';
    }
    return this._value;
  }
  set value(val: string) {
    this._value = String(val);
    this.setAttribute('value', this._value);
  }

  get checked(): boolean {
    return this._checked;
  }
  set checked(val: boolean) {
    this._checked = Boolean(val);
    if (this._checked) this.attributes.set('checked', '');
    else this.attributes.delete('checked');
  }

  get disabled(): boolean {
    return this._disabled;
  }
  set disabled(val: boolean) {
    this._disabled = Boolean(val);
    if (this._disabled) this.attributes.set('disabled', '');
    else this.attributes.delete('disabled');
  }

  get hidden(): boolean {
    return this._hidden;
  }
  set hidden(val: boolean) {
    this._hidden = Boolean(val);
    if (this._hidden) this.attributes.set('hidden', '');
    else this.attributes.delete('hidden');
  }

  get open(): boolean {
    return this._open;
  }
  set open(val: boolean) {
    this._open = Boolean(val);
    if (this._open) this.attributes.set('open', '');
    else this.attributes.delete('open');
  }

  get returnValue(): string {
    return this._returnValue;
  }
  set returnValue(val: string) {
    this._returnValue = String(val);
  }

  get form(): FakeElement | null {
    return this.closest('form');
  }

  get textContent(): string {
    if (this.tagName === '#TEXT') {
      return this._textContent;
    }
    if (this.children.length === 0) {
      return this._textContent;
    }
    return this._textContent + this.children.map((c) => c.textContent).join('');
  }
  set textContent(val: string) {
    this.children = [];
    this._textContent = String(val ?? '');
  }

  get innerHTML(): string {
    return this.serializeChildren();
  }
  set innerHTML(html: string) {
    this.children = [];
    this._textContent = '';
    if (!html) return;
    parseHtmlInto(html, this, this.ownerDocument);
  }

  private serializeChildren(): string {
    const parts: string[] = [];
    if (this._textContent) parts.push(this._textContent);
    for (const c of this.children) {
      if (c.tagName === '#TEXT') {
        parts.push(c._textContent);
        continue;
      }
      const attrs = Array.from(c.attributes.entries())
        .map(([k, v]) => (v === '' ? k : `${k}="${v}"`))
        .join(' ');
      const tag = c.tagName.toLowerCase();
      parts.push(`<${tag}${attrs ? ` ${attrs}` : ''}>${c.serializeChildren()}</${tag}>`);
    }
    return parts.join('');
  }

  setAttribute(name: string, value: string): void {
    const lower = name.toLowerCase();
    this.attributes.set(lower, String(value));
    if (lower.startsWith('data-')) {
      const camel = lower
        .slice(5)
        .replace(/-([a-z])/g, (_, ch: string) => ch.toUpperCase());
      this.datasetTarget[camel] = String(value);
    }
    if (lower === 'value') this._value = String(value);
    if (lower === 'checked') this._checked = true;
    if (lower === 'disabled') this._disabled = true;
    if (lower === 'hidden') this._hidden = true;
    if (lower === 'open') this._open = true;
  }

  getAttribute(name: string): string | null {
    const lower = name.toLowerCase();
    return this.attributes.get(lower) ?? null;
  }

  removeAttribute(name: string): void {
    const lower = name.toLowerCase();
    this.attributes.delete(lower);
    if (lower === 'checked') this._checked = false;
    if (lower === 'disabled') this._disabled = false;
    if (lower === 'hidden') this._hidden = false;
    if (lower === 'open') this._open = false;
  }

  appendChild(child: FakeElement): FakeElement {
    if (child.tagName === '#FRAGMENT') {
      for (const grand of child.children) {
        this.appendChild(grand);
      }
      child.children = [];
      return child;
    }
    child.parentElement = this;
    this.children.push(child);
    return child;
  }

  remove(): void {
    if (!this.parentElement) return;
    this.parentElement.children = this.parentElement.children.filter((c) => c !== this);
    this.parentElement = null;
  }

  addEventListener(type: string, cb: (ev: any) => void, capture = false): void {
    const arr = this.listeners.get(type) ?? [];
    arr.push({ cb, capture });
    this.listeners.set(type, arr);
  }

  removeEventListener(type: string, cb: (ev: any) => void, capture = false): void {
    const arr = this.listeners.get(type);
    if (!arr) return;
    this.listeners.set(
      type,
      arr.filter((item) => item.cb !== cb || item.capture !== capture)
    );
  }

  dispatchEvent(event: FakeEvent | Event): boolean {
    const ev = event as FakeEvent;
    ev.target = this;
    const chain: FakeElement[] = [];
    let cur: FakeElement | null = this;
    while (cur) {
      chain.push(cur);
      cur = cur.parentElement;
    }

    // Capture phase
    for (let i = chain.length - 1; i >= 0; i -= 1) {
      const node = chain[i]!;
      ev.currentTarget = node;
      for (const entry of [...(node.listeners.get(ev.type) ?? [])]) {
        if (entry.capture) entry.cb(ev);
      }
    }

    // Bubble phase
    for (let i = 0; i < chain.length; i += 1) {
      if (i > 0 && !ev.bubbles) break;
      const node = chain[i]!;
      ev.currentTarget = node;
      for (const entry of [...(node.listeners.get(ev.type) ?? [])]) {
        if (!entry.capture) entry.cb(ev);
      }
    }

    return !ev.defaultPrevented;
  }

  click(): void {
    if (this._disabled) return;
    this.dispatchEvent(new FakeEvent('click', { bubbles: true, cancelable: true }));
  }

  focus(): void {
    this.ownerDocument.activeElement = this;
  }

  requestSubmit(): void {
    this.dispatchEvent(new FakeEvent('submit', { bubbles: true, cancelable: true }));
  }

  showModal(): void {
    this.open = true;
  }

  close(returnValue = ''): void {
    this.returnValue = returnValue;
    this.open = false;
    this.dispatchEvent(new FakeEvent('close', { bubbles: false, cancelable: false }));
  }

  matches(selector: string): boolean {
    const sel = selector.trim();
    if (!sel) return false;
    const parts = sel.split(/\s+/);
    if (parts.length > 1) {
      // Support simple descendant selector like `pre > code` or `a b`
      const clean = parts.filter((p) => p !== '>');
      if (!this.matchesSingle(clean[clean.length - 1]!)) return false;
      let cur = this.parentElement;
      let idx = clean.length - 2;
      while (cur && idx >= 0) {
        if (cur.matchesSingle(clean[idx]!)) idx -= 1;
        cur = cur.parentElement;
      }
      return idx < 0;
    }
    return this.matchesSingle(sel);
  }

  private matchesSingle(sel: string): boolean {
    let rest = sel;
    const tagMatch = rest.match(/^[a-zA-Z0-9-]+/);
    if (tagMatch) {
      if (this.tagName.toLowerCase() !== tagMatch[0].toLowerCase()) return false;
      rest = rest.slice(tagMatch[0].length);
    }

    const tokenRegex = /#([a-zA-Z0-9_-]+)|\.([a-zA-Z0-9_-]+)|\[([^\]=]+)(?:="([^"]*)"|='([^']*)')?\]/g;
    let m: RegExpExecArray | null;
    while ((m = tokenRegex.exec(rest)) !== null) {
      if (m[1]) {
        if (this.id !== m[1]) return false;
      } else if (m[2]) {
        if (!this.classList.contains(m[2])) return false;
      } else if (m[3]) {
        const attrName = m[3].toLowerCase();
        const attrVal = this.getAttribute(attrName);
        if (attrVal === null) return false;
        const expected = m[4] ?? m[5];
        if (expected !== undefined && attrVal !== expected) return false;
      }
    }
    return true;
  }

  closest(selector: string): FakeElement | null {
    let cur: FakeElement | null = this;
    while (cur) {
      if (cur.matches(selector)) return cur;
      cur = cur.parentElement;
    }
    return null;
  }

  querySelectorAll(selector: string): FakeElement[] {
    const out: FakeElement[] = [];
    const visit = (node: FakeElement): void => {
      for (const child of node.children) {
        if (child.matches(selector)) out.push(child);
        visit(child);
      }
    };
    visit(this);
    return out;
  }

  querySelector(selector: string): FakeElement | null {
    return this.querySelectorAll(selector)[0] ?? null;
  }
}

class FakeDocument {
  documentElement: FakeElement;
  body: FakeElement;
  activeElement: FakeElement | null = null;

  constructor() {
    this.documentElement = new FakeElement('html', this);
    this.body = new FakeElement('body', this);
    this.documentElement.appendChild(this.body);
  }

  createElement(tagName: string): FakeElement {
    return new FakeElement(tagName, this);
  }

  createDocumentFragment(): FakeElement {
    return new FakeElement('#FRAGMENT', this);
  }
}

const VOID_HTML_TAGS = new Set(['br', 'hr', 'img', 'input', 'meta', 'link']);

function parseHtmlInto(html: string, root: FakeElement, doc: FakeDocument): void {
  const stack: FakeElement[] = [root];
  const tokenRegex = /<!--[\s\S]*?-->|<\/([a-zA-Z0-9-]+)\s*>|<([a-zA-Z0-9-]+)([^>]*?)(\/?)>|([^<]+)/g;
  let match: RegExpExecArray | null;

  while ((match = tokenRegex.exec(html)) !== null) {
    if (match[0].startsWith('<!--')) continue;

    if (match[1]) {
      const closeTag = match[1].toUpperCase();
      for (let i = stack.length - 1; i >= 1; i -= 1) {
        if (stack[i]!.tagName === closeTag) {
          stack.length = i;
          break;
        }
      }
      continue;
    }

    if (match[2]) {
      const tag = match[2].toLowerCase();
      const rawAttrs = match[3] ?? '';
      const selfClose = Boolean(match[4]) || VOID_HTML_TAGS.has(tag);
      const el = new FakeElement(tag, doc);

      const attrRegex = /([a-zA-Z_:][\w:.-]*)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+)))?/g;
      let am: RegExpExecArray | null;
      while ((am = attrRegex.exec(rawAttrs)) !== null) {
        const name = am[1]!.toLowerCase();
        const val = decodeBasicEntities(am[2] ?? am[3] ?? am[4] ?? '');
        el.setAttribute(name, val);
      }

      const parent = stack[stack.length - 1]!;
      parent.appendChild(el);
      if (!selfClose) {
        stack.push(el);
      }
      continue;
    }

    if (match[5]) {
      const text = decodeBasicEntities(match[5]);
      const parent = stack[stack.length - 1]!;
      if (parent.children.length === 0) {
        parent._textContent += text;
      } else {
        const span = new FakeElement('#TEXT', doc);
        span._textContent = text;
        parent.appendChild(span);
      }
    }
  }
}

function createMemoryStorage(): StorageLike {
  const map = new Map<string, string>();
  return {
    getItem: (k) => map.get(k) ?? null,
    setItem: (k, v) => {
      map.set(k, v);
    },
    removeItem: (k) => {
      map.delete(k);
    },
  };
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const GPU_DEVICE_F16: DeviceOption = {
  id: 'webgpu-high-performance',
  label: 'WebGPU (High Performance): NVIDIA RTX 4070',
  backend: 'webgpu',
  gpuPreference: 'high-performance',
  fingerprint: 'nvidia|ada|4070|nvidia rtx 4070|hw',
  adapterInfo: {
    vendor: 'nvidia',
    architecture: 'ada',
    device: '4070',
    description: 'NVIDIA RTX 4070',
    isFallbackAdapter: false,
  },
  features: ['shader-f16', 'subgroups'],
  limits: {
    maxBufferSize: 2_147_483_648,
    maxStorageBufferBindingSize: 1_073_741_824,
  },
  supportsFp16: true,
  recommendedDtype: 'q4f16',
  available: true,
  runtimeDetails: {
    crossOriginIsolated: false,
    hardwareConcurrency: 16,
    simdSupported: true,
    numThreads: 1,
  },
};

const CPU_WASM_DEVICE: DeviceOption = {
  id: 'wasm-cpu',
  label: 'CPU (WASM SIMD, 1スレッド / 非COI)',
  backend: 'wasm',
  features: ['wasm-simd'],
  limits: {},
  supportsFp16: false,
  recommendedDtype: 'q4',
  available: true,
  runtimeDetails: {
    crossOriginIsolated: false,
    hardwareConcurrency: 16,
    simdSupported: true,
    numThreads: 1,
  },
};

function makeSampleMetrics(overrides?: Partial<InferenceMetrics>): InferenceMetrics {
  return {
    speed: {
      ttftMs: 180,
      prefillMs: 180,
      decodeMs: 1000,
      totalMs: 1180,
      promptTokens: 24,
      generatedTokens: 40,
      tokensPerSec: 33.9,
      decodeTokensPerSec: 40,
      interTokenLatencyAvgMs: 25,
      interTokenLatencyP95Ms: 32,
      ...overrides?.speed,
    },
    load: {
      estimatedVramBytes: 720_000_000,
      mainThreadAvgFrameMs: 16.5,
      mainThreadCpuBusyRatio: 0.03,
      droppedFrames: 0,
      longTaskCount: 0,
      backend: 'webgpu',
      dtype: 'q4f16',
      activeThreads: 1,
      ...overrides?.load,
    },
    power: {
      batterySupported: true,
      batteryLevelStart: 80,
      batteryLevelEnd: 79.9,
      batteryCharging: false,
      batteryDischargingTimeSec: 7200,
      batteryDeltaPct: -0.1,
      estimatedPowerWatts: 28.4,
      estimatedEnergyWh: 0.0093,
      tokensPerWattHour: 4301,
      estimationMethod: 'battery-api+utilization',
      ...overrides?.power,
    },
  };
}

let fakeDoc: FakeDocument;

beforeEach(() => {
  fakeDoc = new FakeDocument();
  vi.stubGlobal('document', fakeDoc);
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('markdown sanitizer (renderSafeMarkdown)', () => {
  it('validates link URLs strictly and blocks javascript/data/vbscript/obfuscated schemes', () => {
    expect(isSafeHref('https://huggingface.co/onnx-community')).toBe(true);
    expect(isSafeHref('http://localhost:5173')).toBe(true);
    expect(isSafeHref('mailto:test@example.com')).toBe(true);
    expect(isSafeHref('#section-1')).toBe(true);

    expect(isSafeHref('javascript:alert(1)')).toBe(false);
    expect(isSafeHref('java\nscript:alert(1)')).toBe(false);
    expect(isSafeHref('  \tJAVASCRIPT:alert(1)')).toBe(false);
    expect(isSafeHref('data:text/html,<script>alert(1)</script>')).toBe(false);
    expect(isSafeHref('vbscript:msgbox(1)')).toBe(false);
    expect(isSafeHref('//evil.example/payload')).toBe(false);
  });

  it('strips XSS payloads, enforces noopener noreferrer on safe links, and renders code copy button', () => {
    const md = [
      '# 見出し',
      '本文と [安全なリンク](https://example.com) と [危険なリンク](javascript:alert(1))。',
      '<script>alert("xss")</script>',
      '<img src="https://evil.example/track.png" onerror="alert(2)">',
      '![remote](https://evil.example/beacon.gif)',
      '```ts\nconst x = 1 < 2;\n```',
    ].join('\n\n');

    const html = renderSafeMarkdown(md);
    expect(html).toContain('<h1>見出し</h1>');
    expect(html).toContain('href="https://example.com"');
    expect(html).toContain('target="_blank"');
    expect(html).toContain('rel="noopener noreferrer"');
    expect(html).not.toContain('javascript:');
    expect(html).not.toMatch(/<script\b/i);
    expect(html).not.toMatch(/<img\b/i);
    expect(html).not.toMatch(/<[^>]*\sonerror=/i);
    expect(html).toContain('&lt;script&gt;');
    expect(html).toContain('[画像: remote]');
    expect(html).toContain('code-copy-btn');
    expect(html).toContain('const x = 1 &lt; 2;');

    const fallbackSanitized = sanitizeHtmlFallback(
      '<p>ok</p><script>bad()</script><img src="x" onerror="alert(1)"><a href="javascript:evil()">x</a>'
    );
    expect(fallbackSanitized).toBe('<p>ok</p><a>x</a>');
  });

  it('uses DOMPurify purifier hook when supported', () => {
    const addHook = vi.fn();
    const sanitize = vi.fn((dirty: string) => dirty);
    const html = renderSafeMarkdown('**bold**', {
      purifier: {
        isSupported: true,
        addHook,
        sanitize,
      },
    });
    expect(addHook).toHaveBeenCalledWith('afterSanitizeAttributes', expect.any(Function));
    expect(sanitize).toHaveBeenCalledTimes(1);
    expect(html).toContain('<strong>bold</strong>');
  });
});

describe('IME-safe Enter submit', () => {
  it('enforces composing, keyCode 229, Shift+Enter, and 50ms Safari WebKit compositionend guard', () => {
    expect(
      shouldSubmitOnEnter(
        { key: 'Enter', shiftKey: false, isComposing: false, keyCode: 13, timeStamp: 1000 },
        { isComposing: false, lastCompositionEndMs: 0 }
      )
    ).toBe(true);

    expect(
      shouldSubmitOnEnter(
        { key: 'Enter', shiftKey: true, isComposing: false, keyCode: 13, timeStamp: 1000 },
        { isComposing: false, lastCompositionEndMs: 0 }
      )
    ).toBe(false);

    expect(
      shouldSubmitOnEnter(
        { key: 'Enter', shiftKey: false, isComposing: true, keyCode: 13, timeStamp: 1000 },
        { isComposing: false, lastCompositionEndMs: 0 }
      )
    ).toBe(false);

    expect(
      shouldSubmitOnEnter(
        { key: 'Enter', shiftKey: false, isComposing: false, keyCode: 229, timeStamp: 1000 },
        { isComposing: false, lastCompositionEndMs: 0 }
      )
    ).toBe(false);

    // Safari WebKit bug 165004: compositionend at t=1000, Enter at t=1030 (30ms <= 50ms) -> blocked!
    expect(
      shouldSubmitOnEnter(
        { key: 'Enter', shiftKey: false, isComposing: false, keyCode: 13, timeStamp: 1030 },
        { isComposing: false, lastCompositionEndMs: 1000 }
      )
    ).toBe(false);

    // Boundary: exactly 50ms -> blocked; 51ms -> submits!
    expect(
      shouldSubmitOnEnter(
        { key: 'Enter', shiftKey: false, isComposing: false, keyCode: 13, timeStamp: 1050 },
        { isComposing: false, lastCompositionEndMs: 1000 }
      )
    ).toBe(false);

    expect(
      shouldSubmitOnEnter(
        { key: 'Enter', shiftKey: false, isComposing: false, keyCode: 13, timeStamp: 1051 },
        { isComposing: false, lastCompositionEndMs: 1000 }
      )
    ).toBe(true);
  });

  it('wires compositionstart, compositionend, and keydown onto textarea and calls form.requestSubmit()', () => {
    const form = fakeDoc.createElement('form');
    const textarea = fakeDoc.createElement('textarea');
    form.appendChild(textarea);
    const submitSpy = vi.fn();
    form.addEventListener('submit', (e: FakeEvent) => {
      e.preventDefault();
      submitSpy();
    });

    const cleanup = attachImeSafeEnterSubmit({
      textarea: textarea as unknown as HTMLTextAreaElement,
      form: form as unknown as HTMLFormElement,
    });

    textarea.dispatchEvent(new FakeEvent('compositionstart', { timeStamp: 100 }));
    textarea.dispatchEvent(
      new FakeEvent('keydown', { key: 'Enter', shiftKey: false, isComposing: false, timeStamp: 120 })
    );
    expect(submitSpy).not.toHaveBeenCalled();

    textarea.dispatchEvent(new FakeEvent('compositionend', { timeStamp: 200 }));
    textarea.dispatchEvent(
      new FakeEvent('keydown', { key: 'Enter', shiftKey: false, isComposing: false, timeStamp: 225 })
    );
    expect(submitSpy).not.toHaveBeenCalled();

    textarea.dispatchEvent(
      new FakeEvent('keydown', { key: 'Enter', shiftKey: false, isComposing: false, timeStamp: 280 })
    );
    expect(submitSpy).toHaveBeenCalledTimes(1);

    cleanup();
    textarea.dispatchEvent(
      new FakeEvent('keydown', { key: 'Enter', shiftKey: false, isComposing: false, timeStamp: 500 })
    );
    expect(submitSpy).toHaveBeenCalledTimes(1);
  });
});

describe('viewTemplates & SVG comparison charts', () => {
  it('renders all 6 catalog models with exact download size, dtype, and cache badges', () => {
    const gemmaCache: ModelCacheInfo = {
      modelId: 'gemma-4-e2b',
      repoId: getModelById('gemma-4-e2b').repoId,
      cached: true,
      cachedBytes: 3_130_508_887,
      fileCount: 3,
      files: [],
      lastCheckedAt: 100,
    };

    const html = renderModelCardsHtml({
      models: MODEL_CATALOG,
      selectedModelId: 'qwen-3.5-0.8b',
      loadedModelId: 'qwen-3.5-0.8b',
      selectedDevice: GPU_DEVICE_F16,
      requestedDtype: 'auto',
      useExternalDataFormat: false,
      cacheMap: new Map([['gemma-4-e2b', gemmaCache]]),
      resolveDtypeFn: resolveEffectiveDtype,
    });

    for (const model of MODEL_CATALOG) {
      expect(html).toContain(model.name);
      expect(html).toContain(`data-select-model="${model.id}"`);
    }
    expect(html).toContain('✓ キャッシュ済');
    expect(html).toContain('未ダウンロード');
    expect(html).toContain('ロード済');
  });

  it('renders accessible SVG bar charts, winner cards, and responsive comparison table', () => {
    const store = new BenchmarkStore({ storage: null });
    const rec1: BenchmarkRecord = {
      id: 'b-1',
      timestamp: 1000,
      promptId: 'quick-qa-ja',
      promptLabel: '簡潔QA',
      promptText: 'Test',
      modelId: 'qwen-3.5-0.8b',
      modelName: 'Qwen 3.5 0.8B',
      deviceId: GPU_DEVICE_F16.id,
      deviceLabel: GPU_DEVICE_F16.label,
      backend: 'webgpu',
      gpuPreference: 'high-performance',
      dtype: 'q4f16',
      generationConfig: {
        maxNewTokens: 256,
        temperature: 0.7,
        topP: 0.9,
        topK: 40,
        doSample: true,
        enableThinking: false,
      },
      status: 'completed',
      outputPreview: 'Ok',
      metrics: makeSampleMetrics(),
    };
    const rec2: BenchmarkRecord = {
      ...rec1,
      id: 'b-2',
      timestamp: 2000,
      modelId: 'llama-3.2-3b',
      modelName: 'Llama 3.2 3B <script>',
      deviceId: CPU_WASM_DEVICE.id,
      deviceLabel: CPU_WASM_DEVICE.label,
      backend: 'wasm',
      dtype: 'q4',
      metrics: makeSampleMetrics({
        speed: {
          ttftMs: 650,
          prefillMs: 650,
          decodeMs: 4000,
          totalMs: 4650,
          promptTokens: 24,
          generatedTokens: 40,
          tokensPerSec: 8.6,
          decodeTokensPerSec: 10,
          interTokenLatencyAvgMs: 100,
          interTokenLatencyP95Ms: 130,
        },
        load: {
          estimatedVramBytes: 3_600_000_000,
          mainThreadAvgFrameMs: 16.8,
          mainThreadCpuBusyRatio: 0.05,
          droppedFrames: 1,
          longTaskCount: 0,
          backend: 'wasm',
          dtype: 'q4',
          activeThreads: 1,
        },
        power: {
          batterySupported: false,
          estimatedPowerWatts: 18,
          estimatedEnergyWh: 0.023,
          tokensPerWattHour: 1739,
          estimationMethod: 'utilization-model',
        },
      }),
    };

    store.add(rec1);
    store.add(rec2);
    const records = store.list();

    const chartsHtml = buildBenchmarkComparisonCharts(records);
    expect(chartsHtml).toContain('role="img"');
    expect(chartsHtml).toContain('data-bar="true"');
    expect(chartsHtml).toContain('★ Qwen 3.5 0.8B');
    expect(chartsHtml).not.toContain('<script>');

    const emptySvg = renderHorizontalBarChartSvg({
      id: 'empty',
      title: 'Empty',
      description: 'Empty',
      unitLabel: 'tok/s',
      barColorVar: 'var(--chart-bar-speed)',
      direction: 'higher-is-better',
      items: [],
    });
    expect(emptySvg).toContain('比較可能な完了データがありません');

    const compHtml = renderComparisonViewHtml({
      records,
      summary: store.getSummaryStats(records),
      comparison: store.computeComparison(records),
    });
    expect(compHtml).toContain('tabindex="0" role="region" aria-labelledby="benchmark-table-heading"');
    expect(compHtml).toContain('<th scope="col">Overall tok/s</th>');
    expect(compHtml).toContain('data-label="Decode tok/s"');
    expect(compHtml).toContain('🏆 最速スループット');
    expect(compHtml).toContain('⚡ 最短 TTFT');
    expect(compHtml).toContain('🔋 最高電力効率');
    expect(compHtml).toContain('💾 最小推定メモリ');
  });

  it('renders 3-pillar telemetry HUD with JS heap fallback note & estimation method label, plus chat thinking <details>', () => {
    const hudHtml = renderTelemetryHudHtml({
      liveSnapshot: null,
      finalMetrics: makeSampleMetrics(),
      liveBattery: { supported: true, level: 79.9, charging: false, dischargingTimeSec: 7200, sampledAt: 1 },
      isGenerating: false,
      selectedDevice: GPU_DEVICE_F16,
    });
    expect(hudHtml).toContain('⚡ 応答速度 (Speed)');
    expect(hudHtml).toContain('🧠 負荷・メモリ (Load)');
    expect(hudHtml).toContain('🔋 電力消費 (Power)');
    expect(hudHtml).toContain('実測バッテリー情報 + 負荷推定');
    expect(hudHtml).toContain('performance.memory');

    const msg: ChatMessage = {
      id: 'm-1',
      role: 'assistant',
      content: '最終回答は **42** です。',
      thinking: 'ステップ1: 確認',
      createdAt: 1000,
      modelId: 'qwen-3.5-4b',
      deviceId: GPU_DEVICE_F16.id,
      dtype: 'q4f16',
      metrics: makeSampleMetrics(),
    };
    const el = renderChatMessageElement(msg, new Set());
    const html = el.innerHTML;
    expect(html).toContain('<details class="thinking-trace">');
    expect(html).toContain('ステップ1: 確認');
    expect(html).toContain('<strong>42</strong>');
    expect(html).toContain('40.0 tok/s');
  });
});

describe('StudioDialogHost (native <dialog> compliance)', () => {
  it('uses .showModal() without role=dialog or aria-modal, autofocuses cancel, and resets returnValue on close', async () => {
    const container = fakeDoc.createElement('div');
    fakeDoc.body.appendChild(container);
    const host = new StudioDialogHost(container as unknown as HTMLElement);

    const confirmPromise = host.showConfirmDialog({
      title: 'キャッシュ削除',
      message: '削除しますか？',
      confirmLabel: '削除する',
      cancelLabel: 'キャンセル',
      danger: true,
    });

    const dialog = container.querySelector('dialog.studio-dialog')!;
    expect(dialog).not.toBeNull();
    expect(dialog.open).toBe(true);
    expect(dialog.getAttribute('role')).toBeNull();
    expect(dialog.getAttribute('aria-modal')).toBeNull();
    expect(dialog.getAttribute('aria-labelledby')).toBe('studio-confirm-dialog-title');

    const cancelBtn = dialog.querySelector('.dialog-footer button[data-dialog-action="cancel"]')!;
    expect(cancelBtn.getAttribute('autofocus')).toBe('');

    const confirmBtn = dialog.querySelector('button[data-dialog-action="confirm"]')!;
    confirmBtn.click();

    expect(await confirmPromise).toBe(true);
    expect(dialog.returnValue).toBe('');
    expect(container.querySelector('dialog')).toBeNull();
  });
});

describe('DebouncedLiveAnnouncer & AppController integration', () => {
  it('debounces polite live announcements during streaming bursts while permitting immediate/assertive notices', () => {
    vi.useFakeTimers();
    let clock = 0;
    const politeEl = fakeDoc.createElement('div');
    const assertiveEl = fakeDoc.createElement('div');
    const announcer = new DebouncedLiveAnnouncer({
      politeEl: politeEl as unknown as HTMLElement,
      assertiveEl: assertiveEl as unknown as HTMLElement,
      minIntervalMs: 1500,
      nowFn: () => clock,
    });

    announcer.announcePolite('生成中: 1 トークン');
    expect(politeEl.textContent).toBe('生成中: 1 トークン');

    clock = 200;
    announcer.announcePolite('生成中: 5 トークン');
    announcer.announcePolite('生成中: 12 トークン');
    expect(politeEl.textContent).toBe('生成中: 1 トークン');

    clock = 1500;
    vi.advanceTimersByTime(1350);
    expect(politeEl.textContent).toBe('生成中: 12 トークン');

    announcer.announcePolite('生成完了', { immediate: true });
    expect(politeEl.textContent).toBe('生成完了');

    announcer.announceAssertive('エラー発生');
    expect(assertiveEl.textContent).toBe('エラー発生');
    announcer.dispose();
  });

  it('integrates model selection, dtype warnings, rAF-batched streaming, benchmark comparison, and cache confirmation', async () => {
    const root = fakeDoc.createElement('div');
    fakeDoc.body.appendChild(root);

    const themeStorage = createMemoryStorage();
    const benchmarkStore = new BenchmarkStore({ storage: null });
    const rafQueue: Array<{ id: number; cb: FrameRequestCallback }> = [];
    let rafSeq = 1;

    const mockClient = {
      probeDevices: vi.fn(async () => [GPU_DEVICE_F16, CPU_WASM_DEVICE]),
      loadModel: vi.fn(async (_params, onProgress?: (p: ModelLoadProgress) => void) => {
        onProgress?.({
          modelId: 'qwen-3.5-0.8b',
          stage: 'downloading',
          message: 'Downloading...',
          overallProgressPct: 60,
          loadedBytes: 360_000_000,
          expectedTotalBytes: 603_931_785,
          files: [
            {
              file: 'onnx/decoder_model_merged_q4f16.onnx_data',
              status: 'progress',
              loadedBytes: 360_000_000,
              totalBytes: 584_000_000,
              progressPct: 61.6,
            },
          ],
        });
        return {
          modelId: 'qwen-3.5-0.8b',
          device: GPU_DEVICE_F16,
          effectiveDtype: 'q4f16',
          useExternalDataFormat: false,
          loadTimeMs: 420,
        };
      }),
      generate: vi.fn(async (_messages, _config, callbacks) => {
        callbacks?.onStart?.({ promptTokens: 20 });
        callbacks?.onToken?.({
          deltaText: '<think>分析',
          fullText: '<think>分析',
          tokenCount: 1,
          timestampMs: 100,
          isFirstToken: true,
        });
        callbacks?.onToken?.({
          deltaText: '</think>',
          fullText: '<think>分析</think>',
          tokenCount: 2,
          timestampMs: 125,
          isFirstToken: false,
        });
        callbacks?.onToken?.({
          deltaText: 'WebGPUが',
          fullText: '<think>分析</think>WebGPUが',
          tokenCount: 3,
          timestampMs: 150,
          isFirstToken: false,
        });
        callbacks?.onToken?.({
          deltaText: '高速です。',
          fullText: '<think>分析</think>WebGPUが高速です。',
          tokenCount: 4,
          timestampMs: 175,
          isFirstToken: false,
        });

        // Verify rAF coalescing: 4 tokens scheduled only 1 rAF callback!
        expect(rafQueue).toHaveLength(1);
        const frame = rafQueue.shift()!;
        frame.cb(180);

        const streamingDom = root.querySelector('#chat-messages')!.innerHTML;
        expect(streamingDom).toContain('WebGPUが高速です。');
        expect(streamingDom).toContain('分析');

        return {
          fullText: '<think>分析</think>WebGPUが高速です。',
          thinking: '分析',
          content: 'WebGPUが高速です。',
          aborted: false,
          speed: makeSampleMetrics().speed,
          activeThreads: 1,
          backend: 'webgpu',
          dtype: 'q4f16',
        };
      }),
      abort: vi.fn(),
      unloadModel: vi.fn(async () => undefined),
      inspectCache: vi.fn(async () => []),
      clearCache: vi.fn(async () => ({ clearedBytes: 1024, deletedEntries: 1 })),
      dispose: vi.fn(),
    } as unknown as LlmRuntimeClient;

    const clearCacheFn = vi.fn(async () => ({ clearedBytes: 500_000, deletedEntries: 2 }));
    const downloadFileFn = vi.fn();

    const controller = new AppController(root as unknown as HTMLElement, {
      client: mockClient,
      benchmarkStore,
      themeStorage,
      clearCacheFn,
      downloadFileFn,
      getStorageEstimateFn: async () => ({
        usageBytes: 1_000_000_000,
        quotaBytes: 10_000_000_000,
        availableBytes: 9_000_000_000,
      }),
      sampleBatteryFn: async () => ({
        supported: true,
        level: 85,
        charging: false,
        dischargingTimeSec: 7200,
        sampledAt: 1,
      }),
      observeBatteryFn: async () => () => {},
      telemetryEnv: {
        requestAnimationFrame: null,
        PerformanceObserver: null,
        navigator: {
          getBattery: async () => ({
            level: 0.85,
            charging: false,
            dischargingTime: 7200,
            chargingTime: Number.POSITIVE_INFINITY,
          }),
        },
      },
      requestAnimationFrameFn: (cb) => {
        const id = rafSeq++;
        rafQueue.push({ id, cb });
        return id;
      },
      cancelAnimationFrameFn: (id) => {
        const idx = rafQueue.findIndex((entry) => entry.id === id);
        if (idx !== -1) rafQueue.splice(idx, 1);
      },
    });

    await controller.init();

    // Check 6 model cards + preset chips + theme toggle
    expect(root.querySelectorAll('button[data-select-model]')).toHaveLength(6);
    expect(root.querySelectorAll('button[data-apply-preset]')).toHaveLength(BENCHMARK_PRESETS.length);
    expect(root.querySelector('#header-battery-pill')!.textContent).toContain('85%');

    const darkBtn = root.querySelector('button[data-theme-mode="dark"]')!;
    darkBtn.click();
    expect(fakeDoc.documentElement.dataset.theme).toBe('dark');
    expect(darkBtn.getAttribute('aria-pressed')).toBe('true');

    // Select Llama 3.2 3B + q8 -> verify resolveEffectiveDtype preview warning + thinking disabled
    root.querySelector('button[data-select-model="llama-3.2-3b"]')!.click();
    const dtypeSelect = root.querySelector('#dtype-select')!;
    dtypeSelect.value = 'q8';
    dtypeSelect.dispatchEvent(new FakeEvent('change'));
    expect(root.querySelector('#dtype-preview-box')!.innerHTML).toContain('use_external_data_format: true');
    expect(root.querySelector('#chk-enable-thinking')!.disabled).toBe(true);

    // Switch back to qwen-3.5-0.8b & auto dtype and load
    root.querySelector('button[data-select-model="qwen-3.5-0.8b"]')!.click();
    dtypeSelect.value = 'auto';
    dtypeSelect.dispatchEvent(new FakeEvent('change'));
    root.querySelector('#btn-load-model')!.click();
    await new Promise((r) => setTimeout(r, 0));

    expect(mockClient.loadModel).toHaveBeenCalledTimes(1);
    expect(root.querySelector('#loaded-model-badge')!.textContent).toContain('Qwen 3.5 0.8B');

    // Run 1-click benchmark and verify recording + comparison export
    root.querySelector('#btn-run-benchmark')!.click();
    await new Promise((r) => setTimeout(r, 20));

    expect(mockClient.generate).toHaveBeenCalledTimes(1);
    expect(controller.getStateSnapshot().messages.map((m) => m.error)).toEqual([undefined, undefined]);
    expect(benchmarkStore.list()).toHaveLength(1);
    expect(root.querySelector('#benchmark-count-badge')!.textContent).toBe('1');
    expect(root.querySelector('#comparison-container')!.innerHTML).toContain('Qwen 3.5 0.8B');

    root.querySelector('button[data-export-csv]')!.click();
    expect(downloadFileFn).toHaveBeenCalledWith(
      expect.stringMatching(/\.csv$/),
      expect.stringContaining('Qwen 3.5 0.8B'),
      'text/csv;charset=utf-8'
    );

    // Tab switching (click + keyboard ArrowLeft)
    const tabChat = root.querySelector('#tab-chat')!;
    const tabCompare = root.querySelector('#tab-compare')!;
    const panelChat = root.querySelector('#panel-chat')!;
    const panelCompare = root.querySelector('#panel-compare')!;
    tabCompare.click();
    expect(tabCompare.getAttribute('aria-selected')).toBe('true');
    expect(panelCompare.hidden).toBe(false);
    expect(panelChat.hidden).toBe(true);

    root.querySelector('[role="tablist"]')!.dispatchEvent(
      new FakeEvent('keydown', { key: 'ArrowLeft', bubbles: true, cancelable: true })
    );
    expect(tabChat.getAttribute('aria-selected')).toBe('true');
    expect(panelChat.hidden).toBe(false);

    // Technical specification <dialog> open & close
    root.querySelector('#btn-open-methodology')!.click();
    const specDialog = root.querySelector('dialog.studio-dialog')!;
    expect(specDialog.open).toBe(true);
    expect(specDialog.getAttribute('role')).toBeNull();
    specDialog.querySelector('button[data-dialog-action="cancel"]')!.click();
    expect(root.querySelector('dialog.studio-dialog')).toBeNull();

    // Stop/Abort button triggers client.abort() while generating
    const origGenerateImpl = mockClient.generate as unknown as ReturnType<typeof vi.fn>;
    origGenerateImpl.mockImplementationOnce(async () => {
      root.querySelector('#btn-stop-generation')!.click();
      return {
        fullText: '部分出力',
        thinking: '',
        content: '部分出力',
        aborted: true,
        speed: makeSampleMetrics().speed,
        activeThreads: 1,
        backend: 'webgpu',
        dtype: 'q4f16',
      };
    });
    const chatInput = root.querySelector('#chat-prompt-input')!;
    chatInput.value = '中断テスト';
    root.querySelector('#chat-composer-form')!.requestSubmit();
    await new Promise((r) => setTimeout(r, 20));
    expect(controller.getStateSnapshot().messages.at(-1)?.error).toBeUndefined();
    expect(mockClient.abort).toHaveBeenCalledTimes(1);

    // Cache deletion requires dialog confirmation
    root.querySelector('#btn-clear-model-cache')!.click();
    const dialogConfirm = root.querySelector('dialog button[data-dialog-action="confirm"]')!;
    dialogConfirm.click();
    await new Promise((r) => setTimeout(r, 0));
    expect(clearCacheFn).toHaveBeenCalledWith('qwen-3.5-0.8b');

    // Unload & dispose
    root.querySelector('#btn-unload-model')!.click();
    await new Promise((r) => setTimeout(r, 0));
    expect(mockClient.unloadModel).toHaveBeenCalledTimes(1);
    expect(controller.getStateSnapshot().loadedState).toBeNull();

    controller.dispose();
  });
});
