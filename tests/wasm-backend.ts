/**
 * WASM Backend adapter for tests
 *
 * This wraps the rumpy-wasm module to implement the Backend interface.
 */

import { Backend, NDArray as IFaceNDArray } from './test-utils';

// These will be populated by initWasmBackend
let wasmModule: any = null;

class WasmNDArray implements IFaceNDArray {
  private _inner: any;

  constructor(inner: any) {
    this._inner = inner;
  }

  get shape(): number[] {
    return Array.from(this._inner.shape);
  }

  get data(): Float64Array {
    return this._inner.toTypedArray();
  }

  toArray(): number[] {
    return Array.from(this._inner.toTypedArray());
  }

  get inner(): any {
    return this._inner;
  }
}

export class WasmBackend implements Backend {
  name = 'wasm';
  private wasm: any;

  constructor(wasm: any) {
    this.wasm = wasm;
  }

  private wrap(inner: any): WasmNDArray {
    return new WasmNDArray(inner);
  }

  private unwrap(arr: IFaceNDArray): any {
    return (arr as WasmNDArray).inner;
  }

  // ============ Creation ============

  zeros(shape: number[]): IFaceNDArray {
    return this.wrap(this.wasm.zeros(new Uint32Array(shape)));
  }

  ones(shape: number[]): IFaceNDArray {
    return this.wrap(this.wasm.ones(new Uint32Array(shape)));
  }

  full(shape: number[], value: number): IFaceNDArray {
    return this.wrap(this.wasm.full(new Uint32Array(shape), value));
  }

  arange(start: number, stop: number, step: number): IFaceNDArray {
    if (step === 0) {
      throw new Error('step cannot be zero');
    }
    return this.wrap(this.wasm.arange(start, stop, step));
  }

  linspace(start: number, stop: number, num: number): IFaceNDArray {
    return this.wrap(this.wasm.linspace(start, stop, num));
  }

  eye(n: number): IFaceNDArray {
    return this.wrap(this.wasm.eye(n));
  }

  diag(arr: IFaceNDArray, k: number = 0): IFaceNDArray {
    return this.wrap(this.unwrap(arr).diag(k));
  }

  array(data: number[], shape?: number[]): IFaceNDArray {
    const s = shape || [data.length];
    return this.wrap(this.wasm.arrayFromTyped(new Float64Array(data), new Uint32Array(s)));
  }

  // ============ Math - Unary ============

  sin(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.sinArr(this.unwrap(arr)));
  }

  cos(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.cosArr(this.unwrap(arr)));
  }

  tan(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.tanArr(this.unwrap(arr)));
  }

  arcsin(arr: IFaceNDArray): IFaceNDArray {
    // Fallback - not directly available in WASM
    const result = Array.from(arr.data).map(Math.asin);
    return this.array(result, arr.shape);
  }

  arccos(arr: IFaceNDArray): IFaceNDArray {
    const result = Array.from(arr.data).map(Math.acos);
    return this.array(result, arr.shape);
  }

  arctan(arr: IFaceNDArray): IFaceNDArray {
    const result = Array.from(arr.data).map(Math.atan);
    return this.array(result, arr.shape);
  }

  sinh(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.sinhArr(this.unwrap(arr)));
  }

  cosh(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.coshArr(this.unwrap(arr)));
  }

  tanh(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.tanhArr(this.unwrap(arr)));
  }

  exp(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.expArr(this.unwrap(arr)));
  }

  log(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.logArr(this.unwrap(arr)));
  }

  log2(arr: IFaceNDArray): IFaceNDArray {
    const result = Array.from(arr.data).map(Math.log2);
    return this.array(result, arr.shape);
  }

  log10(arr: IFaceNDArray): IFaceNDArray {
    const result = Array.from(arr.data).map(Math.log10);
    return this.array(result, arr.shape);
  }

  sqrt(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.sqrtArr(this.unwrap(arr)));
  }

  cbrt(arr: IFaceNDArray): IFaceNDArray {
    const result = Array.from(arr.data).map(Math.cbrt);
    return this.array(result, arr.shape);
  }

  abs(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.absArr(this.unwrap(arr)));
  }

  sign(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.signArr(this.unwrap(arr)));
  }

  floor(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.floorArr(this.unwrap(arr)));
  }

  ceil(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.ceilArr(this.unwrap(arr)));
  }

  round(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.roundArr(this.unwrap(arr)));
  }

  neg(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.negArr(this.unwrap(arr)));
  }

  reciprocal(arr: IFaceNDArray): IFaceNDArray {
    const result = Array.from(arr.data).map(x => 1 / x);
    return this.array(result, arr.shape);
  }

  square(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.squareArr(this.unwrap(arr)));
  }

  // ============ Math - Binary ============

  add(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.unwrap(a).add(this.unwrap(b)));
  }

  sub(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.unwrap(a).sub(this.unwrap(b)));
  }

  mul(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.unwrap(a).mul(this.unwrap(b)));
  }

  div(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.unwrap(a).div(this.unwrap(b)));
  }

  pow(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    // Element-wise power - use scalar power for each element
    // This is a fallback since element-wise pow isn't directly available
    const aData = a.data;
    const bData = b.data;
    const result = new Float64Array(aData.length);
    for (let i = 0; i < aData.length; i++) {
      result[i] = Math.pow(aData[i], bData[i]);
    }
    return this.array(Array.from(result), a.shape);
  }

  maximum(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.maximum(this.unwrap(a), this.unwrap(b)));
  }

  minimum(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.minimum(this.unwrap(a), this.unwrap(b)));
  }

  // ============ Math - Scalar ============

  addScalar(arr: IFaceNDArray, scalar: number): IFaceNDArray {
    return this.wrap(this.unwrap(arr).addScalar(scalar));
  }

  subScalar(arr: IFaceNDArray, scalar: number): IFaceNDArray {
    return this.wrap(this.unwrap(arr).subScalar(scalar));
  }

  mulScalar(arr: IFaceNDArray, scalar: number): IFaceNDArray {
    return this.wrap(this.unwrap(arr).mulScalar(scalar));
  }

  divScalar(arr: IFaceNDArray, scalar: number): IFaceNDArray {
    return this.wrap(this.unwrap(arr).divScalar(scalar));
  }

  powScalar(arr: IFaceNDArray, scalar: number): IFaceNDArray {
    return this.wrap(this.unwrap(arr).powScalar(scalar));
  }

  clip(arr: IFaceNDArray, min: number, max: number): IFaceNDArray {
    return this.wrap(this.unwrap(arr).clip(min, max));
  }

  // ============ Stats ============

  sum(arr: IFaceNDArray): number {
    return this.unwrap(arr).sum();
  }

  prod(arr: IFaceNDArray): number {
    return this.unwrap(arr).prod();
  }

  mean(arr: IFaceNDArray): number {
    return this.unwrap(arr).mean();
  }

  var(arr: IFaceNDArray, ddof: number = 0): number {
    if (ddof === 0) {
      return this.unwrap(arr).var();
    }
    return this.unwrap(arr).varDdof(ddof);
  }

  std(arr: IFaceNDArray, ddof: number = 0): number {
    return this.unwrap(arr).std(ddof);
  }

  min(arr: IFaceNDArray): number {
    if (arr.data.length === 0) throw new Error('zero-size array');
    return this.unwrap(arr).min();
  }

  max(arr: IFaceNDArray): number {
    if (arr.data.length === 0) throw new Error('zero-size array');
    return this.unwrap(arr).max();
  }

  argmin(arr: IFaceNDArray): number {
    if (arr.data.length === 0) throw new Error('zero-size array');
    return this.unwrap(arr).argmin();
  }

  argmax(arr: IFaceNDArray): number {
    if (arr.data.length === 0) throw new Error('zero-size array');
    return this.unwrap(arr).argmax();
  }

  cumsum(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.unwrap(arr).cumsum(0));
  }

  cumprod(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.unwrap(arr).cumprod(0));
  }

  all(arr: IFaceNDArray): boolean {
    return this.unwrap(arr).all() !== 0;
  }

  any(arr: IFaceNDArray): boolean {
    return this.unwrap(arr).any() !== 0;
  }

  sumAxis(arr: IFaceNDArray, axis: number): IFaceNDArray {
    return this.wrap(this.unwrap(arr).sumAxis(axis, false));
  }

  meanAxis(arr: IFaceNDArray, axis: number): IFaceNDArray {
    return this.wrap(this.unwrap(arr).meanAxis(axis, false));
  }

  // ============ Linalg ============

  matmul(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    // Use optimized f32 parallel matmul for better performance
    // The generic wasm.matmul goes through f64 which is much slower
    const aShape = a.shape;
    const bShape = b.shape;

    // Handle different cases
    if (aShape.length === 2 && bShape.length === 2) {
      // Validate dimensions: a is [m, k], b is [k2, n] - k must equal k2
      const m = aShape[0];
      const k = aShape[1];
      const k2 = bShape[0];
      const n = bShape[1];

      if (k !== k2) {
        throw new Error(`dimension mismatch for matmul: a.shape[1] (${k}) != b.shape[0] (${k2})`);
      }

      // Convert to f32 for optimized kernel
      const aData = new Float32Array(a.data);
      const bData = new Float32Array(b.data);

      // Use the optimized parallel f32 matmul (XNNPACK-style 6x8 kernel with packing)
      const cData = this.wasm.matmulF32OptimizedParallelV3(aData, bData, m, n, k);

      // Convert result back and wrap
      return this.array(Array.from(cData), [m, n]);
    }

    // Fallback to generic matmul for non-2D cases
    return this.wrap(this.wasm.matmul(this.unwrap(a), this.unwrap(b)));
  }

  dot(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.dot(this.unwrap(a), this.unwrap(b)));
  }

  inner(a: IFaceNDArray, b: IFaceNDArray): number {
    // Inner product: sum of element-wise multiplication
    const aData = a.data;
    const bData = b.data;
    let sum = 0;
    for (let i = 0; i < aData.length; i++) {
      sum += aData[i] * bData[i];
    }
    return sum;
  }

  outer(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    // Outer product: a[i] * b[j] for all i, j
    const aData = a.data;
    const bData = b.data;
    const m = aData.length;
    const n = bData.length;
    const result = new Float64Array(m * n);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        result[i * n + j] = aData[i] * bData[j];
      }
    }
    return this.array(Array.from(result), [m, n]);
  }

  transpose(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.unwrap(arr).transpose());
  }

  trace(arr: IFaceNDArray): number {
    // Trace: sum of diagonal elements
    const shape = arr.shape;
    if (shape.length !== 2) throw new Error('Matrix must be 2D');
    const data = arr.data;
    const n = Math.min(shape[0], shape[1]);
    let sum = 0;
    for (let i = 0; i < n; i++) {
      sum += data[i * shape[1] + i];
    }
    return sum;
  }

  det(arr: IFaceNDArray): number {
    return this.wasm.det(this.unwrap(arr));
  }

  inv(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.inv(this.unwrap(arr)));
  }

  solve(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.solve(this.unwrap(a), this.unwrap(b)));
  }

  norm(arr: IFaceNDArray, ord: number = 2): number {
    // WASM norm might have different signature - adjust as needed
    if (ord === Infinity) {
      return Math.max(...Array.from(arr.data).map(Math.abs));
    }
    if (ord === 1) {
      return Array.from(arr.data).reduce((acc, x) => acc + Math.abs(x), 0);
    }
    // Default L2
    return Math.sqrt(Array.from(arr.data).reduce((acc, x) => acc + x * x, 0));
  }

  qr(_arr: IFaceNDArray): { q: IFaceNDArray; r: IFaceNDArray } {
    // QR decomposition not yet implemented in WASM
    throw new Error('QR decomposition not yet implemented in WASM backend');
  }

  svd(_arr: IFaceNDArray): { u: IFaceNDArray; s: IFaceNDArray; vt: IFaceNDArray } {
    // SVD not yet implemented in WASM
    throw new Error('SVD not yet implemented in WASM backend');
  }
}

/**
 * Initialize the WASM backend
 *
 * Must be called before createWasmBackend()
 */
export async function initWasmBackend(): Promise<void> {
  // Import the WASM module
  const module = await import('./wasm-pkg/rumpy_wasm.js');

  // In browser context, fetch the WASM file
  const wasmUrl = new URL('./wasm-pkg/rumpy_wasm_bg.wasm', import.meta.url);
  const wasmResponse = await fetch(wasmUrl);
  const wasmBytes = await wasmResponse.arrayBuffer();

  // Initialize the module with wasm bytes
  await module.default(wasmBytes);

  // Initialize thread pool (wasm-bindgen-rayon)
  // In browser, this sets up Web Workers
  await module.initThreadPool(navigator.hardwareConcurrency || 4);

  wasmModule = module;
}

/**
 * Create a WASM backend instance
 *
 * Requires initWasmBackend() to have been called first
 */
export function createWasmBackend(): Backend {
  if (!wasmModule) {
    throw new Error('WASM module not initialized. Call initWasmBackend() first.');
  }
  return new WasmBackend(wasmModule);
}
