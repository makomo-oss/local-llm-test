export interface ImeCompositionState {
  isComposing: boolean;
  lastCompositionEndMs: number;
}

export interface KeyDownEventLike {
  key: string;
  shiftKey: boolean;
  isComposing?: boolean;
  keyCode?: number;
  timeStamp: number;
}

export const DEFAULT_COMPOSITION_GUARD_MS = 50;

/**
 * Determine whether an Enter keydown inside a composer textarea should submit the form.
 * Blocks submission during active IME composition, legacy `keyCode === 229`,
 * `Shift+Enter` newline input, and within 50ms after `compositionend` to mitigate
 * Safari / WebKit bug 165004 where `compositionend` fires right before the confirming Enter.
 */
export function shouldSubmitOnEnter(
  event: KeyDownEventLike,
  state: ImeCompositionState,
  guardWindowMs = DEFAULT_COMPOSITION_GUARD_MS
): boolean {
  if (event.key !== 'Enter' || event.shiftKey) {
    return false;
  }

  if (event.isComposing || state.isComposing || event.keyCode === 229) {
    return false;
  }

  const elapsedSinceCompositionEnd = event.timeStamp - state.lastCompositionEndMs;
  if (
    Number.isFinite(elapsedSinceCompositionEnd) &&
    elapsedSinceCompositionEnd >= 0 &&
    elapsedSinceCompositionEnd <= guardWindowMs
  ) {
    return false;
  }

  return true;
}

export interface AttachImeSafeEnterSubmitOptions {
  textarea: HTMLTextAreaElement;
  form?: HTMLFormElement | null;
  onSubmit?: () => void;
  guardWindowMs?: number;
}

/**
 * Attach IME-safe Enter-to-submit listeners to a `<textarea>` and return a cleanup function.
 */
export function attachImeSafeEnterSubmit(
  options: AttachImeSafeEnterSubmitOptions
): () => void {
  const { textarea, form, onSubmit, guardWindowMs = DEFAULT_COMPOSITION_GUARD_MS } = options;

  const state: ImeCompositionState = {
    isComposing: false,
    lastCompositionEndMs: Number.NEGATIVE_INFINITY,
  };

  const handleCompositionStart = (): void => {
    state.isComposing = true;
  };

  const handleCompositionEnd = (event: CompositionEvent): void => {
    state.isComposing = false;
    state.lastCompositionEndMs =
      typeof event.timeStamp === 'number' && Number.isFinite(event.timeStamp)
        ? event.timeStamp
        : performance.now();
  };

  const handleKeyDown = (event: KeyboardEvent): void => {
    if (!shouldSubmitOnEnter(event, state, guardWindowMs)) {
      return;
    }

    event.preventDefault();
    if (onSubmit) {
      onSubmit();
      return;
    }

    const targetForm = form ?? textarea.form;
    if (targetForm && typeof targetForm.requestSubmit === 'function') {
      targetForm.requestSubmit();
    } else if (targetForm) {
      targetForm.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    }
  };

  textarea.addEventListener('compositionstart', handleCompositionStart);
  textarea.addEventListener('compositionend', handleCompositionEnd);
  textarea.addEventListener('keydown', handleKeyDown);

  return () => {
    textarea.removeEventListener('compositionstart', handleCompositionStart);
    textarea.removeEventListener('compositionend', handleCompositionEnd);
    textarea.removeEventListener('keydown', handleKeyDown);
  };
}
