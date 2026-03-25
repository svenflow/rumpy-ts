/**
 * numpyjs - NumPy-compatible array library for JavaScript
 *
 * Backends:
 * - JS (CPU): Pure JavaScript, works everywhere
 * - WebGPU (GPU): GPU-accelerated via WebGPU compute shaders
 */

// Re-export types
export type { NDArray, Backend } from './types.js';

// Import Backend type for the createBackend return type
import type { Backend } from './types.js';

/**
 * Create a backend instance.
 *
 * @param type - 'js' for pure-JS CPU backend, 'webgpu' for GPU-accelerated backend
 * @returns A Backend instance ready to use
 */
export async function createBackend(type: 'js' | 'webgpu' = 'js'): Promise<Backend> {
  if (type === 'js') {
    const { createJsBackend } = await import('./js-backend.js');
    return createJsBackend();
  }
  if (type === 'webgpu') {
    const { initWebGPUBackend, createWebGPUBackend } = await import('./webgpu-backend.js');
    await initWebGPUBackend();
    return createWebGPUBackend();
  }
  throw new Error(`Unknown backend type: ${type}. Use 'js' or 'webgpu'.`);
}

// Version
export const version = '1.0.0';
