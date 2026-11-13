import { formatBytes } from '../models/catalog.ts';
import type { DeviceOption } from '../types.ts';
import { escapeHtml } from './markdown.ts';

export interface ConfirmDialogOptions {
  title: string;
  message: string;
  detail?: string;
  confirmLabel?: string;
  cancelLabel?: string;
  danger?: boolean;
}

/**
 * Native `<dialog>` controller adhering strictly to modern-web-guidance:
 * - Uses `.showModal()` only (no `role="dialog"`, no `aria-modal`, no manual focus traps,
 *   no `body { overflow: hidden }`).
 * - Places `autofocus` on the safe cancel/close button.
 * - Reads and resets `dialog.returnValue = ''` inside the `'close'` listener.
 */
export class StudioDialogHost {
  private readonly container: HTMLElement;
  private activeDialog: HTMLDialogElement | null = null;

  constructor(container: HTMLElement) {
    this.container = container;
  }

  private openModalDialog(
    titleId: string,
    innerHtml: string,
    onClose?: (returnValue: string) => void
  ): HTMLDialogElement {
    this.closeActive();

    const dialog = document.createElement('dialog');
    dialog.className = 'studio-dialog';
    dialog.setAttribute('aria-labelledby', titleId);
    dialog.innerHTML = innerHtml;

    const handleClose = (): void => {
      const result = dialog.returnValue;
      dialog.returnValue = '';
      dialog.removeEventListener('close', handleClose);
      if (this.activeDialog === dialog) {
        this.activeDialog = null;
      }
      dialog.remove();
      onClose?.(result);
    };

    dialog.addEventListener('close', handleClose);

    // Handle action buttons with data-dialog-action="confirm" | "cancel"
    dialog.addEventListener('click', (event) => {
      const target = event.target as HTMLElement | null;
      const actionBtn = target?.closest<HTMLButtonElement>('button[data-dialog-action]');
      if (!actionBtn) {
        return;
      }
      const action = actionBtn.dataset.dialogAction ?? 'cancel';
      dialog.close(action);
    });

    this.container.appendChild(dialog);
    this.activeDialog = dialog;

    if (typeof dialog.showModal === 'function') {
      dialog.showModal();
    } else {
      dialog.setAttribute('open', '');
    }

    const autoFocusEl = dialog.querySelector<HTMLElement>('[autofocus]');
    autoFocusEl?.focus?.();

    return dialog;
  }

  closeActive(): void {
    if (!this.activeDialog) {
      return;
    }
    const dialog = this.activeDialog;
    this.activeDialog = null;
    if (dialog.open && typeof dialog.close === 'function') {
      dialog.close('cancel');
    } else {
      dialog.returnValue = '';
      dialog.remove();
    }
  }

  showConfirmDialog(options: ConfirmDialogOptions): Promise<boolean> {
    const titleId = 'studio-confirm-dialog-title';
    const confirmLabel = options.confirmLabel ?? '実行する';
    const cancelLabel = options.cancelLabel ?? 'キャンセル';
    const confirmClass = options.danger ? 'btn btn--danger-solid' : 'btn btn--primary';
    const detailHtml = options.detail
      ? `<p class="notice-box">${escapeHtml(options.detail)}</p>`
      : '';

    return new Promise<boolean>((resolve) => {
      this.openModalDialog(
        titleId,
        `
          <div class="dialog-header">
            <h2 id="${titleId}" class="dialog-title">${escapeHtml(options.title)}</h2>
            <button type="button" class="btn btn--sm" data-dialog-action="cancel" aria-label="閉じる">✕</button>
          </div>
          <div class="dialog-body">
            <p>${escapeHtml(options.message)}</p>
            ${detailHtml}
          </div>
          <div class="dialog-footer">
            <button type="button" class="btn" data-dialog-action="cancel" autofocus>${escapeHtml(cancelLabel)}</button>
            <button type="button" class="${confirmClass}" data-dialog-action="confirm">${escapeHtml(confirmLabel)}</button>
          </div>
        `,
        (returnValue) => {
          resolve(returnValue === 'confirm');
        }
      );
    });
  }

  showDeviceSpecsDialog(device: DeviceOption): void {
    const titleId = 'studio-device-specs-title';
    const info = device.adapterInfo;
    const featureBadges =
      device.features.length > 0
        ? device.features.map((f) => `<span class="chip">${escapeHtml(f)}</span>`).join('')
        : '<span class="hud-note">機能フラグ情報なし</span>';

    const limitRows = Object.entries(device.limits)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([key, val]) => {
        const display =
          key.toLowerCase().includes('size') && val >= 1024 ? `${formatBytes(val)} (${val})` : String(val);
        return `<dt>${escapeHtml(key)}</dt><dd>${escapeHtml(display)}</dd>`;
      })
      .join('');

    this.openModalDialog(
      titleId,
      `
        <div class="dialog-header">
          <h2 id="${titleId}" class="dialog-title">計算デバイス詳細スペック</h2>
          <button type="button" class="btn btn--sm" data-dialog-action="cancel" aria-label="閉じる" autofocus>閉じる</button>
        </div>
        <div class="dialog-body">
          <p><strong>${escapeHtml(device.label)}</strong></p>
          <dl class="spec-grid">
            <dt>バックエンド</dt><dd>${escapeHtml(device.backend)}</dd>
            <dt>推奨量子化</dt><dd>${escapeHtml(device.recommendedDtype)}</dd>
            <dt>shader-f16</dt><dd>${device.supportsFp16 ? '対応 (q4f16可)' : '非対応 (q4推奨)'}</dd>
            <dt>利用可否</dt><dd>${device.available ? '利用可能' : '利用不可'}</dd>
            ${info ? `<dt>Vendor</dt><dd>${escapeHtml(info.vendor || '-')}</dd>` : ''}
            ${info ? `<dt>Architecture</dt><dd>${escapeHtml(info.architecture || '-')}</dd>` : ''}
            ${info ? `<dt>Device</dt><dd>${escapeHtml(info.device || '-')}</dd>` : ''}
            ${info ? `<dt>Description</dt><dd>${escapeHtml(info.description || '-')}</dd>` : ''}
            <dt>CPU コア数</dt><dd>${device.runtimeDetails.hardwareConcurrency}</dd>
            <dt>WASM SIMD</dt><dd>${device.runtimeDetails.simdSupported ? '対応' : '非対応'}</dd>
            <dt>COI / スレッド数</dt><dd>${device.runtimeDetails.crossOriginIsolated ? 'COI有効' : '非COI (GitHub Pages互換)'} / ${device.runtimeDetails.numThreads}スレッド</dd>
          </dl>
          ${device.unavailableReason ? `<p class="notice-box notice-box--warn">${escapeHtml(device.unavailableReason)}</p>` : ''}
          <div>
            <h3>サポート機能 (${device.features.length})</h3>
            <div class="model-card-meta">${featureBadges}</div>
          </div>
          ${
            limitRows
              ? `<div><h3>WebGPU Limits</h3><dl class="spec-grid">${limitRows}</dl></div>`
              : ''
          }
        </div>
        <div class="dialog-footer">
          <button type="button" class="btn btn--primary" data-dialog-action="cancel">OK</button>
        </div>
      `
    );
  }

  showMethodologyDialog(): void {
    const titleId = 'studio-methodology-title';
    this.openModalDialog(
      titleId,
      `
        <div class="dialog-header">
          <h2 id="${titleId}" class="dialog-title">技術仕様・計測方法について</h2>
          <button type="button" class="btn btn--sm" data-dialog-action="cancel" aria-label="閉じる" autofocus>閉じる</button>
        </div>
        <div class="dialog-body">
          <section>
            <h3>1. 完全クライアントサイド実行とプライバシー</h3>
            <p>本アプリは GitHub Pages 上で完結する静的 SPA です。入力プロンプトや推論結果は外部サーバーへ送信されず、モデルファイルのみ Hugging Face (<code>onnx-community</code>) の公開 CDN から取得してブラウザの <code>CacheStorage (transformers-cache)</code> に保存します。</p>
          </section>
          <section>
            <h3>2. デバイス制御 (WebGPU / CPU WASM)</h3>
            <ul>
              <li><strong>WebGPU</strong>: 選択した GPU Preference に応じて Web Worker 内で <code>adapter.requestDevice()</code> を呼び出し、ONNX Runtime Web (<code>executionProviders: [{ name: 'webgpu', device }]</code>) に明示的にバインドします。<code>shader-f16</code> 対応時は <code>q4f16</code>、非対応時は <code>q4</code> へ自動調整します。</li>
              <li><strong>CPU (WASM)</strong>: <code>crossOriginIsolated</code> が無効な静的ホスティング環境では <code>SharedArrayBuffer</code> 制約のため 1 スレッド + WASM SIMD で安全に実行します。</li>
            </ul>
          </section>
          <section>
            <h3>3. 計測メトリクス (応答速度・負荷・電力)</h3>
            <ul>
              <li><strong>応答速度</strong>: Web Worker 内の <code>TextStreamer</code> トークンコールバックと <code>performance.now()</code> により、TTFT、プリフィル時間、デコード tok/s、平均・P95 トークン間遅延 (ITL) を実測します。</li>
              <li><strong>負荷</strong>: メインスレッドの <code>requestAnimationFrame</code> フレーム時間・ドロップフレーム数・Long Task、推定モデル+KV キャッシュメモリを表示します。<code>performance.memory</code> は Chromium 系など対応ブラウザのみ表示されます。</li>
              <li><strong>電力消費</strong>: Web Battery Status API (<code>navigator.getBattery()</code>) に対応しバッテリー放電中の場合は実測レベル変化とデバイス負荷モデルを併用（<code>実測バッテリー情報 + 負荷推定</code>）、非対応または AC 給電時はハードウェア種別・量子化・稼働率に基づく推定モデル（<code>負荷ベース推定モデル</code>）で算出します。</li>
            </ul>
          </section>
        </div>
        <div class="dialog-footer">
          <button type="button" class="btn btn--primary" data-dialog-action="cancel">閉じる</button>
        </div>
      `
    );
  }
}
