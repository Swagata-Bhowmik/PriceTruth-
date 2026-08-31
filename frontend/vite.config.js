/// <reference types="vitest/config" />
import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

// Vite + Vitest configuration for the Price Truth frontend.
//
// The dev server proxies API traffic to the FastAPI backend so the browser
// can call same-origin relative paths (e.g. `/api/v1/search`, `/health`)
// during local development. In production, VITE_API_BASE_URL points at the
// deployed backend instead (see src/config.js).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/health': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  // Vitest configuration block.
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: './src/test/setup.js',
    css: true,
    include: ['src/**/*.{test,spec}.{js,jsx}'],
  },
})
