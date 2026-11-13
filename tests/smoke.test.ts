import { describe, expect, it } from 'vitest';
import type { AutoTokenizer, env } from '@huggingface/transformers';
import config from '../vite.config.ts';

describe('project scaffolding smoke test', () => {
  it('configures Vite for GitHub Pages and ES module workers', () => {
    const expectedBase = process.env.VITE_BASE_PATH || '/local-llm-test/';
    expect(config.base).toBe(expectedBase);
    expect(config.build?.target).toBe('esnext');
    expect(config.worker?.format).toBe('es');
  });

  it('can import core runtime dependencies', async () => {
    const [{ marked }, DOMPurify] = await Promise.all([
      import('marked'),
      import('dompurify'),
    ]);

    expect(typeof marked.parse).toBe('function');
    expect(typeof DOMPurify.default).toBe('function');

    const typeCheck: {
      env?: typeof env;
      tokenizer?: typeof AutoTokenizer;
    } = {};
    expect(typeCheck).toEqual({});
  });
});
