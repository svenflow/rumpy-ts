/**
 * Main test file - runs all tests against all backends
 *
 * This mirrors the Rust test structure in crates/rumpy-tests/
 * but runs via JavaScript/TypeScript against multiple backends.
 *
 * Usage:
 *   bun test              # Run all tests
 *   bun test --watch      # Watch mode
 */

import { describe, beforeAll, it } from 'vitest';
import { Backend } from './test-utils';
import { creationTests } from './creation.test';
import { mathTests } from './math.test';
import { linalgTests } from './linalg.test';
import { statsTests } from './stats.test';
import { manipulationTests } from './manipulation.test';
import { phase2Tests } from './phase2.test';

// Import backends
import { createJsBackend } from './js-backend';
import { initWasmBackend, createWasmBackend } from './wasm-backend';
import { initWebGPUBackend, createWebGPUBackend } from './webgpu-backend';

// ============ Test Suites ============

describe('rumpy-ts', () => {
  // Run tests against pure JS backend (always works)
  describe('js backend', () => {
    let backend: Backend;

    beforeAll(() => {
      backend = createJsBackend();
    });

    const getBackend = () => backend;

    creationTests(getBackend);
    mathTests(getBackend);
    linalgTests(getBackend);
    statsTests(getBackend);
    manipulationTests(getBackend);
    phase2Tests(getBackend);
  });

  // Run tests against WASM backend (requires wasm-pack build)
  describe('wasm backend', () => {
    let backend: Backend;

    beforeAll(async () => {
      try {
        await initWasmBackend();
        backend = createWasmBackend();
      } catch (e) {
        // Skip if WASM not available
        console.warn('WASM backend not available:', e);
      }
    });

    // Skip WASM tests if backend not initialized
    it.skipIf(() => !backend)('wasm backend available', () => {});

    const getBackend = () => backend;

    if (typeof process !== 'undefined') {
      // Skip in Node/Bun where WASM SIMD may not be supported
      it.skip('wasm tests skipped in Node/Bun', () => {});
    } else {
      creationTests(getBackend);
      mathTests(getBackend);
      linalgTests(getBackend);
      statsTests(getBackend);
      manipulationTests(getBackend);
      phase2Tests(getBackend);
    }
  });

  // Run tests against WebGPU backend (requires browser)
  describe('webgpu backend', () => {
    let backend: Backend;

    beforeAll(async () => {
      try {
        await initWebGPUBackend();
        backend = createWebGPUBackend();
      } catch (e) {
        // Skip if WebGPU not available
        console.warn('WebGPU backend not available:', e);
      }
    });

    // Skip WebGPU tests if backend not initialized
    it.skipIf(() => !backend)('webgpu backend available', () => {});

    const getBackend = () => backend;

    if (typeof navigator === 'undefined' || !navigator.gpu) {
      // Skip in Node/Bun where WebGPU is not available
      it.skip('webgpu tests skipped (no browser)', () => {});
    } else {
      creationTests(getBackend);
      mathTests(getBackend);
      linalgTests(getBackend);
      statsTests(getBackend);
      manipulationTests(getBackend);
      phase2Tests(getBackend);
    }
  });
});
