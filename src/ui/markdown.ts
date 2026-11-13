import DOMPurify from 'dompurify';
import { Marked, Renderer, type Tokens } from 'marked';

export interface PurifyLike {
  isSupported?: boolean;
  sanitize(dirty: string, config?: Record<string, unknown>): string;
  addHook?(
    hookName: 'afterSanitizeAttributes',
    cb: (node: Element) => void
  ): void;
  removeHook?(hookName: 'afterSanitizeAttributes'): void;
}

export interface RenderMarkdownOptions {
  purifier?: PurifyLike;
}

const ALLOWED_TAGS = [
  'p',
  'br',
  'strong',
  'b',
  'em',
  'i',
  'del',
  's',
  'code',
  'pre',
  'blockquote',
  'ul',
  'ol',
  'li',
  'h1',
  'h2',
  'h3',
  'h4',
  'h5',
  'h6',
  'hr',
  'table',
  'thead',
  'tbody',
  'tr',
  'th',
  'td',
  'a',
  'span',
  'div',
  'button',
] as const;

const ALLOWED_TAG_SET = new Set<string>(ALLOWED_TAGS);
const VOID_TAG_SET = new Set<string>(['br', 'hr']);

const ALLOWED_ATTRS = ['href', 'title', 'target', 'rel', 'class', 'type'] as const;

/**
 * Escape HTML special characters for safe literal rendering.
 */
export function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/**
 * Validate a link URL strictly, permitting only `https:`, `http:`, `mailto:`,
 * and local `#fragment` anchors while blocking `javascript:`, `data:`, `vbscript:`, etc.
 */
export function isSafeHref(rawHref: string | null | undefined): boolean {
  if (typeof rawHref !== 'string') {
    return false;
  }
  // Remove ASCII control characters and whitespace that browsers strip before scheme parsing.
  const stripped = rawHref.replace(/[\u0000-\u0020\u007f]+/g, '').trim();
  if (stripped.length === 0) {
    return false;
  }
  if (stripped.startsWith('#')) {
    return true;
  }
  if (stripped.startsWith('//')) {
    return false;
  }
  try {
    const parsed = new URL(stripped);
    return (
      parsed.protocol === 'https:' ||
      parsed.protocol === 'http:' ||
      parsed.protocol === 'mailto:'
    );
  } catch {
    return false;
  }
}

function createConfiguredMarked(): Marked {
  const renderer = new Renderer();

  renderer.html = ({ text }: Tokens.HTML | Tokens.Tag): string => {
    return escapeHtml(text);
  };

  renderer.image = ({ href, text }: Tokens.Image): string => {
    const label = text?.trim() || href?.trim() || 'image';
    return `<span class="md-blocked-image">[画像: ${escapeHtml(label)}]</span>`;
  };

  renderer.link = function (this: Renderer, { href, title, tokens }: Tokens.Link): string {
    const innerHtml = this.parser.parseInline(tokens);
    if (!isSafeHref(href)) {
      return `<span class="md-unsafe-link">${innerHtml}</span>`;
    }
    const safeHref = escapeHtml(href.trim());
    const titleAttr = title ? ` title="${escapeHtml(title)}"` : '';
    return `<a href="${safeHref}"${titleAttr} target="_blank" rel="noopener noreferrer">${innerHtml}</a>`;
  };

  renderer.code = ({ text, lang }: Tokens.Code): string => {
    const safeLang = (lang ?? '').trim().replace(/[^a-zA-Z0-9_+-]/g, '');
    const langClass = safeLang ? ` class="language-${safeLang}"` : '';
    const langLabel = safeLang || 'text';
    return [
      '<div class="md-code-block">',
      `<div class="md-code-header"><span>${langLabel}</span><button type="button" class="btn btn--sm code-copy-btn">コピー</button></div>`,
      `<pre><code${langClass}>${escapeHtml(text)}</code></pre>`,
      '</div>',
    ].join('');
  };

  return new Marked({
    gfm: true,
    breaks: true,
    renderer,
  });
}

const markedInstance = createConfiguredMarked();
let hookRegisteredOnDefault = false;

function enforceSafeElementAttributes(node: Element): void {
  const tagName = node.tagName.toLowerCase();
  if (tagName === 'a') {
    const href = node.getAttribute('href');
    if (!isSafeHref(href)) {
      node.removeAttribute('href');
      node.removeAttribute('target');
      node.removeAttribute('rel');
    } else {
      node.setAttribute('target', '_blank');
      node.setAttribute('rel', 'noopener noreferrer');
    }
  } else if (tagName === 'button') {
    node.setAttribute('type', 'button');
  }
}

function sanitizeAttributeList(tagName: string, rawAttrs: string): string {
  const attrs: string[] = [];
  const attrRegex = /([a-zA-Z_:][\w:.-]*)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+)))?/g;
  let match: RegExpExecArray | null;
  let hasHref = false;
  let safeHrefValue = '';
  let safeTitleValue = '';
  let safeClassValue = '';

  while ((match = attrRegex.exec(rawAttrs)) !== null) {
    const name = match[1]?.toLowerCase() ?? '';
    const value = match[2] ?? match[3] ?? match[4] ?? '';

    if (name.startsWith('on') || name === 'style' || name.startsWith('data-')) {
      continue;
    }
    if (name === 'href' && tagName === 'a') {
      if (isSafeHref(value)) {
        hasHref = true;
        safeHrefValue = value.trim();
      }
    } else if (name === 'title' && (tagName === 'a' || tagName === 'span')) {
      safeTitleValue = value;
    } else if (name === 'class') {
      const cleanedClasses = value
        .split(/\s+/)
        .filter((cls) => /^[a-zA-Z0-9_-]+$/.test(cls))
        .join(' ');
      if (cleanedClasses) {
        safeClassValue = cleanedClasses;
      }
    }
  }

  if (safeClassValue) {
    attrs.push(`class="${escapeHtml(safeClassValue)}"`);
  }
  if (safeTitleValue) {
    attrs.push(`title="${escapeHtml(safeTitleValue)}"`);
  }
  if (tagName === 'a' && hasHref) {
    attrs.push(`href="${escapeHtml(safeHrefValue)}"`);
    attrs.push('target="_blank"');
    attrs.push('rel="noopener noreferrer"');
  }
  if (tagName === 'button') {
    attrs.push('type="button"');
  }

  return attrs.length > 0 ? ` ${attrs.join(' ')}` : '';
}

/**
 * Deterministic tag & attribute allowlist sanitizer used when DOMPurify cannot
 * initialize (e.g. Node unit tests without a browser DOM).
 */
export function sanitizeHtmlFallback(dirtyHtml: string): string {
  const withoutDangerousBlocks = dirtyHtml
    .replace(/<\s*(script|style|iframe|object|embed|svg|math|form|noscript)\b[^>]*>[\s\S]*?<\s*\/\s*\1\s*>/gi, '')
    .replace(/<!--[\s\S]*?-->/g, '');

  return withoutDangerousBlocks.replace(
    /<\s*(\/?)\s*([a-zA-Z0-9-]+)([^>]*)>/g,
    (_full, slash: string, rawTag: string, rawAttrs: string) => {
      const tag = rawTag.toLowerCase();
      if (!ALLOWED_TAG_SET.has(tag)) {
        return '';
      }
      if (slash === '/') {
        return VOID_TAG_SET.has(tag) ? '' : `</${tag}>`;
      }
      const attrs = sanitizeAttributeList(tag, rawAttrs);
      return `<${tag}${attrs}>`;
    }
  );
}

/**
 * Convert Markdown output from local LLM inference into sanitized, XSS-safe HTML.
 * Uses `marked` + `DOMPurify` in browsers and falls back to strict allowlist
 * sanitization in DOM-less environments.
 */
export function renderSafeMarkdown(
  markdown: string,
  options?: RenderMarkdownOptions
): string {
  if (!markdown || markdown.trim().length === 0) {
    return '';
  }

  const parsed = markedInstance.parse(markdown);
  const rawHtml = typeof parsed === 'string' ? parsed : '';

  const purifier = options?.purifier ?? (DOMPurify as unknown as PurifyLike);
  if (purifier && purifier.isSupported && typeof purifier.sanitize === 'function') {
    if (!options?.purifier && !hookRegisteredOnDefault && typeof purifier.addHook === 'function') {
      purifier.addHook('afterSanitizeAttributes', enforceSafeElementAttributes);
      hookRegisteredOnDefault = true;
    } else if (options?.purifier && typeof purifier.addHook === 'function') {
      purifier.addHook('afterSanitizeAttributes', enforceSafeElementAttributes);
    }

    return purifier.sanitize(rawHtml, {
      ALLOWED_TAGS: [...ALLOWED_TAGS],
      ALLOWED_ATTR: [...ALLOWED_ATTRS],
      ALLOW_DATA_ATTR: false,
      FORBID_TAGS: ['script', 'style', 'iframe', 'object', 'embed', 'svg', 'math', 'form'],
      FORBID_ATTR: ['style', 'onerror', 'onload', 'onclick', 'onmouseover'],
    });
  }

  return sanitizeHtmlFallback(rawHtml);
}
