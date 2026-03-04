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

import { describe, beforeAll } from 'vitest';
import { Backend } from './test-utils';
import { creationTests } from './creation.test';
import { mathTests } from './math.test';
import { linalgTests } from './linalg.test';
import { statsTests } from './stats.test';
import { manipulationTests } from './manipulation.test';

// Import backends
import { initWasmBackend, createWasmBackend } from './wasm-backend';
import { initWebGPUBackend, createWebGPUBackend } from './webgpu-backend';

// ============ Test Suites ============

describe('rumpy-ts', () => {
  // Run tests against WASM backend
  describe('wasm backend', () => {
    let backend: Backend;

    beforeAll(async () => {
      await initWasmBackend();
      backend = createWasmBackend();
    });

    const getBackend = () => backend;

    creationTests(getBackend);
    mathTests(getBackend);
    linalgTests(getBackend);
    statsTests(getBackend);
    manipulationTests(getBackend);
  });

  // Run tests against WebGPU backend
  describe('webgpu backend', () => {
    let backend: Backend;

    beforeAll(async () => {
      await initWebGPUBackend();
      backend = createWebGPUBackend();
    });

    const getBackend = () => backend;

    creationTests(getBackend);
    mathTests(getBackend);
    linalgTests(getBackend);
    statsTests(getBackend);
    manipulationTests(getBackend);
  });
});
