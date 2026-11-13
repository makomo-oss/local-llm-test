import { defineConfig } from 'vite';

const basePath = process.env.VITE_BASE_PATH || '/local-llm-test/';

export default defineConfig({
  base: basePath,
  build: {
    target: 'esnext',
  },
  worker: {
    format: 'es',
  },
});
