/**
 * Pure JavaScript reference backend for testing
 *
 * This implements all Backend operations in pure JS, serving as:
 * 1. A reference implementation for testing
 * 2. A fallback when WASM is not available
 * 3. A baseline for performance comparisons
 */

import { Backend, NDArray } from './test-utils';

class JsNDArray implements NDArray {
  data: Float64Array;
  shape: number[];

  constructor(data: Float64Array | number[], shape: number[]) {
    this.data = data instanceof Float64Array ? data : new Float64Array(data);
    this.shape = shape;
  }

  toArray(): number[] {
    return Array.from(this.data);
  }

  get size(): number {
    return this.shape.reduce((a, b) => a * b, 1);
  }
}

export class JsBackend implements Backend {
  name = 'js';

  // ============ Creation ============

  zeros(shape: number[]): NDArray {
    const size = shape.reduce((a, b) => a * b, 1);
    return new JsNDArray(new Float64Array(size), shape);
  }

  ones(shape: number[]): NDArray {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float64Array(size).fill(1.0);
    return new JsNDArray(data, shape);
  }

  full(shape: number[], value: number): NDArray {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float64Array(size).fill(value);
    return new JsNDArray(data, shape);
  }

  arange(start: number, stop: number, step: number): NDArray {
    if (step === 0) {
      throw new Error('step cannot be zero');
    }
    const data: number[] = [];
    if (step > 0) {
      for (let x = start; x < stop; x += step) {
        data.push(x);
      }
    } else {
      for (let x = start; x > stop; x += step) {
        data.push(x);
      }
    }
    return new JsNDArray(data, [data.length]);
  }

  linspace(start: number, stop: number, num: number): NDArray {
    if (num === 0) return new JsNDArray([], [0]);
    if (num === 1) return new JsNDArray([start], [1]);
    const step = (stop - start) / (num - 1);
    const data: number[] = [];
    for (let i = 0; i < num; i++) {
      data.push(start + i * step);
    }
    return new JsNDArray(data, [num]);
  }

  eye(n: number): NDArray {
    const data = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      data[i * n + i] = 1.0;
    }
    return new JsNDArray(data, [n, n]);
  }

  diag(arr: NDArray, k: number = 0): NDArray {
    if (arr.shape.length === 1) {
      // Create diagonal matrix from vector
      const n = arr.shape[0] + Math.abs(k);
      const data = new Float64Array(n * n);
      for (let i = 0; i < arr.shape[0]; i++) {
        const row = k >= 0 ? i : i - k;
        const col = k >= 0 ? i + k : i;
        data[row * n + col] = arr.data[i];
      }
      return new JsNDArray(data, [n, n]);
    } else if (arr.shape.length === 2) {
      // Extract diagonal from matrix
      const [rows, cols] = arr.shape;
      const startRow = k >= 0 ? 0 : -k;
      const startCol = k >= 0 ? k : 0;
      const diagLen = Math.min(rows - startRow, cols - startCol);
      const data: number[] = [];
      for (let i = 0; i < diagLen; i++) {
        data.push(arr.data[(startRow + i) * cols + (startCol + i)]);
      }
      return new JsNDArray(data, [data.length]);
    }
    throw new Error('diag requires 1D or 2D array');
  }

  array(data: number[], shape?: number[]): NDArray {
    const s = shape || [data.length];
    return new JsNDArray(data, s);
  }

  // ============ Math - Unary ============

  sin(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map(Math.sin), arr.shape);
  }

  cos(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map(Math.cos), arr.shape);
  }

  tan(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map(Math.tan), arr.shape);
  }

  arcsin(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map(Math.asin), arr.shape);
  }

  arccos(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map(Math.acos), arr.shape);
  }

  arctan(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map(Math.atan), arr.shape);
  }

  sinh(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map(Math.sinh), arr.shape);
  }

  cosh(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map(Math.cosh), arr.shape);
  }

  tanh(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map(Math.tanh), arr.shape);
  }

  exp(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map(Math.exp), arr.shape);
  }

  log(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map(Math.log), arr.shape);
  }

  log2(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map(Math.log2), arr.shape);
  }

  log10(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map(Math.log10), arr.shape);
  }

  sqrt(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map(Math.sqrt), arr.shape);
  }

  cbrt(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map(Math.cbrt), arr.shape);
  }

  abs(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map(Math.abs), arr.shape);
  }

  sign(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map(Math.sign), arr.shape);
  }

  floor(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map(Math.floor), arr.shape);
  }

  ceil(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map(Math.ceil), arr.shape);
  }

  round(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map(Math.round), arr.shape);
  }

  neg(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map((x) => -x), arr.shape);
  }

  reciprocal(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map((x) => 1 / x), arr.shape);
  }

  square(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.map((x) => x * x), arr.shape);
  }

  // ============ Math - Binary ============

  add(a: NDArray, b: NDArray): NDArray {
    this._checkSameShape(a, b);
    const data = new Float64Array(a.data.length);
    for (let i = 0; i < a.data.length; i++) {
      data[i] = a.data[i] + b.data[i];
    }
    return new JsNDArray(data, a.shape);
  }

  sub(a: NDArray, b: NDArray): NDArray {
    this._checkSameShape(a, b);
    const data = new Float64Array(a.data.length);
    for (let i = 0; i < a.data.length; i++) {
      data[i] = a.data[i] - b.data[i];
    }
    return new JsNDArray(data, a.shape);
  }

  mul(a: NDArray, b: NDArray): NDArray {
    this._checkSameShape(a, b);
    const data = new Float64Array(a.data.length);
    for (let i = 0; i < a.data.length; i++) {
      data[i] = a.data[i] * b.data[i];
    }
    return new JsNDArray(data, a.shape);
  }

  div(a: NDArray, b: NDArray): NDArray {
    this._checkSameShape(a, b);
    const data = new Float64Array(a.data.length);
    for (let i = 0; i < a.data.length; i++) {
      data[i] = a.data[i] / b.data[i];
    }
    return new JsNDArray(data, a.shape);
  }

  pow(a: NDArray, b: NDArray): NDArray {
    this._checkSameShape(a, b);
    const data = new Float64Array(a.data.length);
    for (let i = 0; i < a.data.length; i++) {
      data[i] = Math.pow(a.data[i], b.data[i]);
    }
    return new JsNDArray(data, a.shape);
  }

  maximum(a: NDArray, b: NDArray): NDArray {
    this._checkSameShape(a, b);
    const data = new Float64Array(a.data.length);
    for (let i = 0; i < a.data.length; i++) {
      data[i] = Math.max(a.data[i], b.data[i]);
    }
    return new JsNDArray(data, a.shape);
  }

  minimum(a: NDArray, b: NDArray): NDArray {
    this._checkSameShape(a, b);
    const data = new Float64Array(a.data.length);
    for (let i = 0; i < a.data.length; i++) {
      data[i] = Math.min(a.data[i], b.data[i]);
    }
    return new JsNDArray(data, a.shape);
  }

  // ============ Math - Scalar ============

  addScalar(arr: NDArray, scalar: number): NDArray {
    return new JsNDArray(arr.data.map((x) => x + scalar), arr.shape);
  }

  subScalar(arr: NDArray, scalar: number): NDArray {
    return new JsNDArray(arr.data.map((x) => x - scalar), arr.shape);
  }

  mulScalar(arr: NDArray, scalar: number): NDArray {
    return new JsNDArray(arr.data.map((x) => x * scalar), arr.shape);
  }

  divScalar(arr: NDArray, scalar: number): NDArray {
    return new JsNDArray(arr.data.map((x) => x / scalar), arr.shape);
  }

  powScalar(arr: NDArray, scalar: number): NDArray {
    return new JsNDArray(arr.data.map((x) => Math.pow(x, scalar)), arr.shape);
  }

  clip(arr: NDArray, min: number, max: number): NDArray {
    return new JsNDArray(
      arr.data.map((x) => Math.min(Math.max(x, min), max)),
      arr.shape
    );
  }

  // ============ Stats ============

  sum(arr: NDArray): number {
    return arr.data.reduce((a, b) => a + b, 0);
  }

  prod(arr: NDArray): number {
    return arr.data.reduce((a, b) => a * b, 1);
  }

  mean(arr: NDArray): number {
    if (arr.data.length === 0) return NaN;
    return this.sum(arr) / arr.data.length;
  }

  var(arr: NDArray, ddof: number = 0): number {
    if (arr.data.length === 0) return NaN;
    const m = this.mean(arr);
    const sumSq = arr.data.reduce((acc, x) => acc + (x - m) ** 2, 0);
    return sumSq / (arr.data.length - ddof);
  }

  std(arr: NDArray, ddof: number = 0): number {
    return Math.sqrt(this.var(arr, ddof));
  }

  min(arr: NDArray): number {
    if (arr.data.length === 0) throw new Error('zero-size array');
    return Math.min(...arr.data);
  }

  max(arr: NDArray): number {
    if (arr.data.length === 0) throw new Error('zero-size array');
    return Math.max(...arr.data);
  }

  argmin(arr: NDArray): number {
    if (arr.data.length === 0) throw new Error('zero-size array');
    let minIdx = 0;
    for (let i = 1; i < arr.data.length; i++) {
      if (arr.data[i] < arr.data[minIdx]) minIdx = i;
    }
    return minIdx;
  }

  argmax(arr: NDArray): number {
    if (arr.data.length === 0) throw new Error('zero-size array');
    let maxIdx = 0;
    for (let i = 1; i < arr.data.length; i++) {
      if (arr.data[i] > arr.data[maxIdx]) maxIdx = i;
    }
    return maxIdx;
  }

  cumsum(arr: NDArray): NDArray {
    const data = new Float64Array(arr.data.length);
    let sum = 0;
    for (let i = 0; i < arr.data.length; i++) {
      sum += arr.data[i];
      data[i] = sum;
    }
    return new JsNDArray(data, arr.shape);
  }

  cumprod(arr: NDArray): NDArray {
    const data = new Float64Array(arr.data.length);
    let prod = 1;
    for (let i = 0; i < arr.data.length; i++) {
      prod *= arr.data[i];
      data[i] = prod;
    }
    return new JsNDArray(data, arr.shape);
  }

  all(arr: NDArray): boolean {
    return arr.data.every((x) => x !== 0);
  }

  any(arr: NDArray): boolean {
    return arr.data.some((x) => x !== 0);
  }

  sumAxis(arr: NDArray, axis: number): NDArray {
    if (arr.shape.length !== 2) throw new Error('sumAxis only supports 2D');
    if (axis < 0 || axis >= arr.shape.length) throw new Error('invalid axis');
    const [rows, cols] = arr.shape;
    if (axis === 0) {
      // Sum along rows, result is [cols]
      const data = new Float64Array(cols);
      for (let j = 0; j < cols; j++) {
        for (let i = 0; i < rows; i++) {
          data[j] += arr.data[i * cols + j];
        }
      }
      return new JsNDArray(data, [cols]);
    } else {
      // Sum along cols, result is [rows]
      const data = new Float64Array(rows);
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          data[i] += arr.data[i * cols + j];
        }
      }
      return new JsNDArray(data, [rows]);
    }
  }

  meanAxis(arr: NDArray, axis: number): NDArray {
    if (arr.shape.length !== 2) throw new Error('meanAxis only supports 2D');
    const sumResult = this.sumAxis(arr, axis);
    const divisor = arr.shape[axis];
    return new JsNDArray(
      sumResult.data.map((x) => x / divisor),
      sumResult.shape
    );
  }

  // ============ Linalg ============

  matmul(a: NDArray, b: NDArray): NDArray {
    if (a.shape.length !== 2 || b.shape.length !== 2) {
      throw new Error('matmul requires 2D arrays');
    }
    const [m, k1] = a.shape;
    const [k2, n] = b.shape;
    if (k1 !== k2) throw new Error('matmul dimension mismatch');

    const data = new Float64Array(m * n);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        let sum = 0;
        for (let k = 0; k < k1; k++) {
          sum += a.data[i * k1 + k] * b.data[k * n + j];
        }
        data[i * n + j] = sum;
      }
    }
    return new JsNDArray(data, [m, n]);
  }

  dot(a: NDArray, b: NDArray): NDArray {
    if (a.shape.length === 1 && b.shape.length === 1) {
      // Vector dot product
      if (a.shape[0] !== b.shape[0]) throw new Error('dot dimension mismatch');
      let sum = 0;
      for (let i = 0; i < a.data.length; i++) {
        sum += a.data[i] * b.data[i];
      }
      return new JsNDArray([sum], [1]);
    }
    // For 2D, same as matmul
    return this.matmul(a, b);
  }

  inner(a: NDArray, b: NDArray): number {
    if (a.shape[0] !== b.shape[0]) throw new Error('inner dimension mismatch');
    let sum = 0;
    for (let i = 0; i < a.data.length; i++) {
      sum += a.data[i] * b.data[i];
    }
    return sum;
  }

  outer(a: NDArray, b: NDArray): NDArray {
    const m = a.data.length;
    const n = b.data.length;
    const data = new Float64Array(m * n);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        data[i * n + j] = a.data[i] * b.data[j];
      }
    }
    return new JsNDArray(data, [m, n]);
  }

  transpose(arr: NDArray): NDArray {
    if (arr.shape.length === 1) {
      return new JsNDArray(arr.data.slice(), arr.shape);
    }
    if (arr.shape.length !== 2) throw new Error('transpose requires 1D or 2D');
    const [rows, cols] = arr.shape;
    const data = new Float64Array(rows * cols);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        data[j * rows + i] = arr.data[i * cols + j];
      }
    }
    return new JsNDArray(data, [cols, rows]);
  }

  trace(arr: NDArray): number {
    if (arr.shape.length !== 2) throw new Error('trace requires 2D');
    const [rows, cols] = arr.shape;
    const n = Math.min(rows, cols);
    let sum = 0;
    for (let i = 0; i < n; i++) {
      sum += arr.data[i * cols + i];
    }
    return sum;
  }

  det(arr: NDArray): number {
    if (arr.shape.length !== 2) throw new Error('det requires 2D');
    const [rows, cols] = arr.shape;
    if (rows !== cols) throw new Error('det requires square matrix');

    // Simple 2x2 and 3x3 determinants
    if (rows === 2) {
      return arr.data[0] * arr.data[3] - arr.data[1] * arr.data[2];
    }
    if (rows === 3) {
      const [a, b, c, d, e, f, g, h, i] = arr.data;
      return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    }

    // LU decomposition for larger matrices
    const lu = this._luDecompose(arr);
    let det = lu.sign;
    for (let i = 0; i < rows; i++) {
      det *= lu.u.data[i * cols + i];
    }
    return det;
  }

  inv(arr: NDArray): NDArray {
    if (arr.shape.length !== 2) throw new Error('inv requires 2D');
    const [rows, cols] = arr.shape;
    if (rows !== cols) throw new Error('inv requires square matrix');

    const n = rows;
    // Gauss-Jordan elimination
    const aug = new Float64Array(n * 2 * n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        aug[i * 2 * n + j] = arr.data[i * n + j];
      }
      aug[i * 2 * n + n + i] = 1;
    }

    for (let i = 0; i < n; i++) {
      // Find pivot
      let maxRow = i;
      for (let k = i + 1; k < n; k++) {
        if (Math.abs(aug[k * 2 * n + i]) > Math.abs(aug[maxRow * 2 * n + i])) {
          maxRow = k;
        }
      }
      // Swap rows
      for (let k = 0; k < 2 * n; k++) {
        const tmp = aug[i * 2 * n + k];
        aug[i * 2 * n + k] = aug[maxRow * 2 * n + k];
        aug[maxRow * 2 * n + k] = tmp;
      }

      const pivot = aug[i * 2 * n + i];
      if (Math.abs(pivot) < 1e-10) throw new Error('singular matrix');

      // Scale row
      for (let k = 0; k < 2 * n; k++) {
        aug[i * 2 * n + k] /= pivot;
      }

      // Eliminate
      for (let k = 0; k < n; k++) {
        if (k !== i) {
          const factor = aug[k * 2 * n + i];
          for (let j = 0; j < 2 * n; j++) {
            aug[k * 2 * n + j] -= factor * aug[i * 2 * n + j];
          }
        }
      }
    }

    // Extract inverse
    const result = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        result[i * n + j] = aug[i * 2 * n + n + j];
      }
    }
    return new JsNDArray(result, [n, n]);
  }

  solve(a: NDArray, b: NDArray): NDArray {
    // Solve Ax = b using LU decomposition
    const aInv = this.inv(a);
    return this.matmul(aInv, b);
  }

  norm(arr: NDArray, ord: number = 2): number {
    if (ord === 1) {
      return arr.data.reduce((acc, x) => acc + Math.abs(x), 0);
    }
    if (ord === Infinity) {
      return Math.max(...arr.data.map(Math.abs));
    }
    if (ord === -Infinity) {
      return Math.min(...arr.data.map(Math.abs));
    }
    // Default L2 norm
    return Math.sqrt(arr.data.reduce((acc, x) => acc + x * x, 0));
  }

  qr(arr: NDArray): { q: NDArray; r: NDArray } {
    if (arr.shape.length !== 2) throw new Error('qr requires 2D');
    const [m, n] = arr.shape;

    // Modified Gram-Schmidt
    const q = new Float64Array(m * n);
    const r = new Float64Array(n * n);

    // Copy A to Q
    for (let i = 0; i < m * n; i++) q[i] = arr.data[i];

    for (let j = 0; j < n; j++) {
      // Compute norm of column j
      let norm = 0;
      for (let i = 0; i < m; i++) {
        norm += q[i * n + j] ** 2;
      }
      norm = Math.sqrt(norm);
      r[j * n + j] = norm;

      // Normalize column j
      for (let i = 0; i < m; i++) {
        q[i * n + j] /= norm;
      }

      // Orthogonalize remaining columns
      for (let k = j + 1; k < n; k++) {
        let dot = 0;
        for (let i = 0; i < m; i++) {
          dot += q[i * n + j] * q[i * n + k];
        }
        r[j * n + k] = dot;
        for (let i = 0; i < m; i++) {
          q[i * n + k] -= dot * q[i * n + j];
        }
      }
    }

    return {
      q: new JsNDArray(q, [m, n]),
      r: new JsNDArray(r, [n, n]),
    };
  }

  svd(arr: NDArray): { u: NDArray; s: NDArray; vt: NDArray } {
    // Simplified SVD using power iteration (not production quality)
    // For a real implementation, use a proper library
    if (arr.shape.length !== 2) throw new Error('svd requires 2D');
    const [m, n] = arr.shape;
    const k = Math.min(m, n);

    // For now, return placeholder shapes
    // A real implementation would compute actual SVD
    const u = this.eye(m);
    const s = this.zeros([k]);
    const vt = this.eye(n);

    // Compute singular values as sqrt of eigenvalues of A^T A
    const at = this.transpose(arr);
    const ata = this.matmul(at, arr);

    // Simple approximation: diagonal elements
    for (let i = 0; i < k; i++) {
      s.data[i] = Math.sqrt(Math.abs(ata.data[i * n + i]));
    }

    // Sort descending
    const indices = Array.from({ length: k }, (_, i) => i);
    indices.sort((a, b) => s.data[b] - s.data[a]);
    const sortedS = new Float64Array(k);
    for (let i = 0; i < k; i++) {
      sortedS[i] = s.data[indices[i]];
    }

    return {
      u: new JsNDArray(u.data.slice(0, m * k), [m, k]),
      s: new JsNDArray(sortedS, [k]),
      vt: new JsNDArray(vt.data.slice(0, k * n), [k, n]),
    };
  }

  // ============ Helpers ============

  private _checkSameShape(a: NDArray, b: NDArray): void {
    if (a.shape.length !== b.shape.length) {
      throw new Error('shape mismatch');
    }
    for (let i = 0; i < a.shape.length; i++) {
      if (a.shape[i] !== b.shape[i]) {
        throw new Error('shape mismatch');
      }
    }
  }

  private _luDecompose(arr: NDArray): { l: NDArray; u: NDArray; sign: number } {
    const n = arr.shape[0];
    const l = new Float64Array(n * n);
    const u = new Float64Array(n * n);
    let sign = 1;

    // Copy to U
    for (let i = 0; i < n * n; i++) u[i] = arr.data[i];

    // Initialize L as identity
    for (let i = 0; i < n; i++) l[i * n + i] = 1;

    for (let i = 0; i < n; i++) {
      // Find pivot
      let maxRow = i;
      for (let k = i + 1; k < n; k++) {
        if (Math.abs(u[k * n + i]) > Math.abs(u[maxRow * n + i])) {
          maxRow = k;
        }
      }
      if (maxRow !== i) {
        sign *= -1;
        for (let k = 0; k < n; k++) {
          const tmp = u[i * n + k];
          u[i * n + k] = u[maxRow * n + k];
          u[maxRow * n + k] = tmp;
        }
      }

      for (let k = i + 1; k < n; k++) {
        const factor = u[k * n + i] / u[i * n + i];
        l[k * n + i] = factor;
        for (let j = i; j < n; j++) {
          u[k * n + j] -= factor * u[i * n + j];
        }
      }
    }

    return {
      l: new JsNDArray(l, [n, n]),
      u: new JsNDArray(u, [n, n]),
      sign,
    };
  }

  // ============ Creation - Like Functions ============

  zerosLike(arr: NDArray): NDArray {
    return this.zeros(arr.shape);
  }

  onesLike(arr: NDArray): NDArray {
    return this.ones(arr.shape);
  }

  emptyLike(arr: NDArray): NDArray {
    // In JS, we can't have uninitialized memory, so same as zeros
    return this.zeros(arr.shape);
  }

  fullLike(arr: NDArray, value: number): NDArray {
    return this.full(arr.shape, value);
  }

  // ============ Broadcasting ============

  broadcastTo(arr: NDArray, shape: number[]): NDArray {
    // Validate shapes are compatible
    const arrShape = arr.shape;
    if (arrShape.length > shape.length) {
      throw new Error('Cannot broadcast to smaller number of dimensions');
    }

    // Pad arr shape with 1s on the left
    const paddedShape = new Array(shape.length - arrShape.length).fill(1).concat(arrShape);

    // Check compatibility
    for (let i = 0; i < shape.length; i++) {
      if (paddedShape[i] !== 1 && paddedShape[i] !== shape[i]) {
        throw new Error(`Cannot broadcast shape [${arrShape}] to [${shape}]`);
      }
    }

    const size = shape.reduce((a, b) => a * b, 1);
    const result = new Float64Array(size);
    const strides = this._computeStrides(shape);
    const srcStrides = this._computeBroadcastStrides(paddedShape, shape);

    for (let i = 0; i < size; i++) {
      let srcIdx = 0;
      let remaining = i;
      for (let d = 0; d < shape.length; d++) {
        const coord = Math.floor(remaining / strides[d]);
        remaining = remaining % strides[d];
        srcIdx += coord * srcStrides[d];
      }
      result[i] = arr.data[srcIdx];
    }

    return new JsNDArray(result, shape);
  }

  broadcastArrays(...arrays: NDArray[]): NDArray[] {
    if (arrays.length === 0) return [];
    if (arrays.length === 1) return [new JsNDArray(arrays[0].data.slice(), arrays[0].shape)];

    // Compute broadcast shape
    const shapes = arrays.map(a => a.shape);
    const maxDims = Math.max(...shapes.map(s => s.length));

    // Pad all shapes with 1s on the left
    const paddedShapes = shapes.map(s => {
      const padded = new Array(maxDims - s.length).fill(1);
      return padded.concat(s);
    });

    // Compute output shape
    const outShape: number[] = [];
    for (let i = 0; i < maxDims; i++) {
      const dims = paddedShapes.map(s => s[i]);
      const maxDim = Math.max(...dims);
      for (const d of dims) {
        if (d !== 1 && d !== maxDim) {
          throw new Error('Shapes are not broadcastable');
        }
      }
      outShape.push(maxDim);
    }

    // Broadcast each array
    return arrays.map(arr => this.broadcastTo(arr, outShape));
  }

  private _computeStrides(shape: number[]): number[] {
    const strides = new Array(shape.length);
    let stride = 1;
    for (let i = shape.length - 1; i >= 0; i--) {
      strides[i] = stride;
      stride *= shape[i];
    }
    return strides;
  }

  private _computeBroadcastStrides(srcShape: number[], dstShape: number[]): number[] {
    const strides = new Array(dstShape.length);
    let srcStride = 1;
    for (let i = srcShape.length - 1; i >= 0; i--) {
      // If dimension is 1, stride is 0 (broadcast)
      strides[i] = srcShape[i] === 1 ? 0 : srcStride;
      srcStride *= srcShape[i];
    }
    return strides;
  }

  // ============ Shape Manipulation ============

  private _normalizeAxis(axis: number, ndim: number): number {
    if (axis < 0) axis += ndim;
    if (axis < 0 || axis >= ndim) {
      throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
    }
    return axis;
  }

  swapaxes(arr: NDArray, axis1: number, axis2: number): NDArray {
    const ndim = arr.shape.length;
    axis1 = this._normalizeAxis(axis1, ndim);
    axis2 = this._normalizeAxis(axis2, ndim);

    if (axis1 === axis2) {
      return new JsNDArray(arr.data.slice(), arr.shape);
    }

    // Create new shape with swapped axes
    const newShape = [...arr.shape];
    [newShape[axis1], newShape[axis2]] = [newShape[axis2], newShape[axis1]];

    // Create permutation array
    const perm = Array.from({ length: ndim }, (_, i) => i);
    [perm[axis1], perm[axis2]] = [perm[axis2], perm[axis1]];

    return this._transposeGeneral(arr, perm, newShape);
  }

  moveaxis(arr: NDArray, source: number, destination: number): NDArray {
    const ndim = arr.shape.length;
    source = this._normalizeAxis(source, ndim);
    destination = this._normalizeAxis(destination, ndim);

    if (source === destination) {
      return new JsNDArray(arr.data.slice(), arr.shape);
    }

    // Build permutation
    const perm: number[] = [];
    for (let i = 0; i < ndim; i++) {
      if (i !== source) perm.push(i);
    }
    perm.splice(destination, 0, source);

    const newShape = perm.map(i => arr.shape[i]);
    return this._transposeGeneral(arr, perm, newShape);
  }

  private _transposeGeneral(arr: NDArray, perm: number[], newShape: number[]): NDArray {
    const size = arr.data.length;
    const result = new Float64Array(size);

    const oldStrides = this._computeStrides(arr.shape);
    const newStrides = this._computeStrides(newShape);

    for (let i = 0; i < size; i++) {
      // Convert flat index to coordinates in new array
      const coords = new Array(newShape.length);
      let remaining = i;
      for (let d = 0; d < newShape.length; d++) {
        coords[d] = Math.floor(remaining / newStrides[d]);
        remaining = remaining % newStrides[d];
      }

      // Map to old coordinates using inverse permutation
      let oldIdx = 0;
      for (let d = 0; d < perm.length; d++) {
        oldIdx += coords[d] * oldStrides[perm[d]];
      }

      result[i] = arr.data[oldIdx];
    }

    return new JsNDArray(result, newShape);
  }

  squeeze(arr: NDArray, axis?: number): NDArray {
    if (axis !== undefined) {
      const normalizedAxis = this._normalizeAxis(axis, arr.shape.length);
      if (arr.shape[normalizedAxis] !== 1) {
        throw new Error(`cannot squeeze axis ${axis} with size ${arr.shape[normalizedAxis]}`);
      }
      const newShape = arr.shape.filter((_, i) => i !== normalizedAxis);
      return new JsNDArray(arr.data.slice(), newShape.length === 0 ? [1] : newShape);
    }

    // Remove all dimensions of size 1
    const newShape = arr.shape.filter(d => d !== 1);
    return new JsNDArray(arr.data.slice(), newShape.length === 0 ? [1] : newShape);
  }

  expandDims(arr: NDArray, axis: number): NDArray {
    const ndim = arr.shape.length + 1;
    if (axis < 0) axis += ndim;
    if (axis < 0 || axis >= ndim) {
      throw new Error(`axis ${axis} is out of bounds`);
    }

    const newShape = [...arr.shape];
    newShape.splice(axis, 0, 1);
    return new JsNDArray(arr.data.slice(), newShape);
  }

  reshape(arr: NDArray, shape: number[]): NDArray {
    // Handle -1 in shape (infer dimension)
    let inferIdx = -1;
    let knownSize = 1;
    for (let i = 0; i < shape.length; i++) {
      if (shape[i] === -1) {
        if (inferIdx !== -1) throw new Error('can only specify one unknown dimension');
        inferIdx = i;
      } else {
        knownSize *= shape[i];
      }
    }

    const newShape = [...shape];
    if (inferIdx !== -1) {
      newShape[inferIdx] = arr.data.length / knownSize;
    }

    const newSize = newShape.reduce((a, b) => a * b, 1);
    if (newSize !== arr.data.length) {
      throw new Error(`cannot reshape array of size ${arr.data.length} into shape [${newShape}]`);
    }

    return new JsNDArray(arr.data.slice(), newShape);
  }

  flatten(arr: NDArray): NDArray {
    return new JsNDArray(arr.data.slice(), [arr.data.length]);
  }

  concatenate(arrays: NDArray[], axis: number = 0): NDArray {
    if (arrays.length === 0) throw new Error('need at least one array to concatenate');
    if (arrays.length === 1) return new JsNDArray(arrays[0].data.slice(), arrays[0].shape);

    const ndim = arrays[0].shape.length;
    axis = this._normalizeAxis(axis, ndim);

    // Verify shapes match except along concat axis
    for (let i = 1; i < arrays.length; i++) {
      if (arrays[i].shape.length !== ndim) {
        throw new Error('all input arrays must have same number of dimensions');
      }
      for (let d = 0; d < ndim; d++) {
        if (d !== axis && arrays[i].shape[d] !== arrays[0].shape[d]) {
          throw new Error('all input array dimensions except concat axis must match');
        }
      }
    }

    // Compute output shape
    const outShape = [...arrays[0].shape];
    outShape[axis] = arrays.reduce((sum, arr) => sum + arr.shape[axis], 0);

    const outSize = outShape.reduce((a, b) => a * b, 1);
    const result = new Float64Array(outSize);

    // For 1D, simple concatenation
    if (ndim === 1) {
      let offset = 0;
      for (const arr of arrays) {
        result.set(arr.data, offset);
        offset += arr.data.length;
      }
      return new JsNDArray(result, outShape);
    }

    // For nD, use stride-based copy
    const outStrides = this._computeStrides(outShape);

    let axisOffset = 0;
    for (const arr of arrays) {
      const srcStrides = this._computeStrides(arr.shape);
      const srcSize = arr.data.length;

      for (let srcIdx = 0; srcIdx < srcSize; srcIdx++) {
        // Convert to coordinates
        const coords = new Array(ndim);
        let remaining = srcIdx;
        for (let d = 0; d < ndim; d++) {
          coords[d] = Math.floor(remaining / srcStrides[d]);
          remaining = remaining % srcStrides[d];
        }

        // Add offset along concat axis
        coords[axis] += axisOffset;

        // Convert to dest index
        let dstIdx = 0;
        for (let d = 0; d < ndim; d++) {
          dstIdx += coords[d] * outStrides[d];
        }

        result[dstIdx] = arr.data[srcIdx];
      }

      axisOffset += arr.shape[axis];
    }

    return new JsNDArray(result, outShape);
  }

  stack(arrays: NDArray[], axis: number = 0): NDArray {
    if (arrays.length === 0) throw new Error('need at least one array to stack');

    // Verify all shapes are the same
    const shape = arrays[0].shape;
    for (let i = 1; i < arrays.length; i++) {
      if (arrays[i].shape.length !== shape.length) {
        throw new Error('all input arrays must have the same shape');
      }
      for (let d = 0; d < shape.length; d++) {
        if (arrays[i].shape[d] !== shape[d]) {
          throw new Error('all input arrays must have the same shape');
        }
      }
    }

    // Expand dims on each array, then concatenate
    const expanded = arrays.map(arr => this.expandDims(arr, axis));
    return this.concatenate(expanded, axis);
  }

  split(arr: NDArray, indices: number | number[], axis: number = 0): NDArray[] {
    const ndim = arr.shape.length;
    axis = this._normalizeAxis(axis, ndim);
    const axisSize = arr.shape[axis];

    let splitIndices: number[];
    if (typeof indices === 'number') {
      // Split into n equal parts
      if (axisSize % indices !== 0) {
        throw new Error(`array of size ${axisSize} cannot be split into ${indices} equal parts`);
      }
      const partSize = axisSize / indices;
      splitIndices = [];
      for (let i = partSize; i < axisSize; i += partSize) {
        splitIndices.push(i);
      }
    } else {
      splitIndices = indices;
    }

    const results: NDArray[] = [];
    let start = 0;

    const getSlice = (startIdx: number, endIdx: number): NDArray => {
      const sliceShape = [...arr.shape];
      sliceShape[axis] = endIdx - startIdx;
      const sliceSize = sliceShape.reduce((a, b) => a * b, 1);
      const sliceData = new Float64Array(sliceSize);

      const srcStrides = this._computeStrides(arr.shape);
      const dstStrides = this._computeStrides(sliceShape);

      for (let dstIdx = 0; dstIdx < sliceSize; dstIdx++) {
        const coords = new Array(ndim);
        let remaining = dstIdx;
        for (let d = 0; d < ndim; d++) {
          coords[d] = Math.floor(remaining / dstStrides[d]);
          remaining = remaining % dstStrides[d];
        }

        coords[axis] += startIdx;

        let srcIdx = 0;
        for (let d = 0; d < ndim; d++) {
          srcIdx += coords[d] * srcStrides[d];
        }

        sliceData[dstIdx] = arr.data[srcIdx];
      }

      return new JsNDArray(sliceData, sliceShape);
    };

    for (const idx of splitIndices) {
      results.push(getSlice(start, idx));
      start = idx;
    }
    results.push(getSlice(start, axisSize));

    return results;
  }

  // ============ Conditional ============

  where(condition: NDArray, x: NDArray, y: NDArray): NDArray {
    // Broadcast all arrays to the same shape
    const [condBcast, xBcast, yBcast] = this.broadcastArrays(condition, x, y);
    const size = condBcast.data.length;
    const result = new Float64Array(size);

    for (let i = 0; i < size; i++) {
      result[i] = condBcast.data[i] !== 0 ? xBcast.data[i] : yBcast.data[i];
    }

    return new JsNDArray(result, condBcast.shape);
  }

  // ============ Advanced Indexing ============

  take(arr: NDArray, indices: NDArray | number[], axis?: number): NDArray {
    const indexArray = Array.isArray(indices) ? indices : Array.from(indices.data);

    if (axis === undefined) {
      // Take from flattened array
      const result = new Float64Array(indexArray.length);
      for (let i = 0; i < indexArray.length; i++) {
        let idx = indexArray[i];
        if (idx < 0) idx += arr.data.length;
        result[i] = arr.data[idx];
      }
      return new JsNDArray(result, [indexArray.length]);
    }

    const ndim = arr.shape.length;
    axis = this._normalizeAxis(axis, ndim);

    // Output shape: replace axis dimension with indices length
    const outShape = [...arr.shape];
    outShape[axis] = indexArray.length;

    const outSize = outShape.reduce((a, b) => a * b, 1);
    const result = new Float64Array(outSize);

    const srcStrides = this._computeStrides(arr.shape);
    const dstStrides = this._computeStrides(outShape);

    for (let dstIdx = 0; dstIdx < outSize; dstIdx++) {
      const coords = new Array(ndim);
      let remaining = dstIdx;
      for (let d = 0; d < ndim; d++) {
        coords[d] = Math.floor(remaining / dstStrides[d]);
        remaining = remaining % dstStrides[d];
      }

      // Map the axis coordinate through indices
      let srcAxisCoord = indexArray[coords[axis]];
      if (srcAxisCoord < 0) srcAxisCoord += arr.shape[axis];
      coords[axis] = srcAxisCoord;

      let srcIdx = 0;
      for (let d = 0; d < ndim; d++) {
        srcIdx += coords[d] * srcStrides[d];
      }

      result[dstIdx] = arr.data[srcIdx];
    }

    return new JsNDArray(result, outShape);
  }

  // ============ Batched Operations ============

  batchedMatmul(a: NDArray, b: NDArray): NDArray {
    // Supports shapes like (batch, M, K) @ (batch, K, N) -> (batch, M, N)
    // Or (batch, M, K) @ (K, N) -> (batch, M, N) with broadcasting
    if (a.shape.length < 2 || b.shape.length < 2) {
      throw new Error('batchedMatmul requires at least 2D arrays');
    }

    // Get matrix dimensions (last two axes)
    const aM = a.shape[a.shape.length - 2];
    const aK = a.shape[a.shape.length - 1];
    const bK = b.shape[b.shape.length - 2];
    const bN = b.shape[b.shape.length - 1];

    if (aK !== bK) throw new Error('matmul inner dimensions must match');

    // Compute batch dimensions
    const aBatchShape = a.shape.slice(0, -2);
    const bBatchShape = b.shape.slice(0, -2);

    // Broadcast batch dimensions
    const maxBatchDims = Math.max(aBatchShape.length, bBatchShape.length);
    const paddedABatch = new Array(maxBatchDims - aBatchShape.length).fill(1).concat(aBatchShape);
    const paddedBBatch = new Array(maxBatchDims - bBatchShape.length).fill(1).concat(bBatchShape);

    const outBatchShape: number[] = [];
    for (let i = 0; i < maxBatchDims; i++) {
      const ad = paddedABatch[i];
      const bd = paddedBBatch[i];
      if (ad !== 1 && bd !== 1 && ad !== bd) {
        throw new Error('batch dimensions are not broadcastable');
      }
      outBatchShape.push(Math.max(ad, bd));
    }

    const outShape = [...outBatchShape, aM, bN];
    const batchSize = outBatchShape.reduce((a, b) => a * b, 1);
    const matSize = aM * bN;
    const result = new Float64Array(batchSize * matSize);

    // Compute strides for batch indexing
    const aBatchStrides = this._computeStrides(paddedABatch);
    const bBatchStrides = this._computeStrides(paddedBBatch);
    const outBatchStrides = this._computeStrides(outBatchShape);

    const aMatStride = aM * aK;
    const bMatStride = bK * bN;

    for (let batch = 0; batch < batchSize; batch++) {
      // Convert batch index to coordinates
      const coords = new Array(maxBatchDims);
      let remaining = batch;
      for (let d = 0; d < maxBatchDims; d++) {
        coords[d] = Math.floor(remaining / outBatchStrides[d]);
        remaining = remaining % outBatchStrides[d];
      }

      // Map to source batch indices with broadcasting
      let aOffset = 0;
      let bOffset = 0;
      for (let d = 0; d < maxBatchDims; d++) {
        const aCoord = paddedABatch[d] === 1 ? 0 : coords[d];
        const bCoord = paddedBBatch[d] === 1 ? 0 : coords[d];
        aOffset += aCoord * aBatchStrides[d];
        bOffset += bCoord * bBatchStrides[d];
      }
      aOffset *= aMatStride;
      bOffset *= bMatStride;

      // Perform matmul for this batch
      const outOffset = batch * matSize;
      for (let i = 0; i < aM; i++) {
        for (let j = 0; j < bN; j++) {
          let sum = 0;
          for (let k = 0; k < aK; k++) {
            sum += a.data[aOffset + i * aK + k] * b.data[bOffset + k * bN + j];
          }
          result[outOffset + i * bN + j] = sum;
        }
      }
    }

    return new JsNDArray(result, outShape);
  }

  // ============ Einstein Summation ============

  einsum(subscripts: string, ...operands: NDArray[]): NDArray {
    // Parse einsum string
    const [inputStr, outputStr] = subscripts.split('->').map(s => s.trim());
    const inputSubscripts = inputStr.split(',').map(s => s.trim());

    if (inputSubscripts.length !== operands.length) {
      throw new Error(`einsum: expected ${inputSubscripts.length} operands, got ${operands.length}`);
    }

    // Map each label to its dimension size
    const labelSizes: Map<string, number> = new Map();
    const inputLabels: string[][] = [];

    for (let i = 0; i < operands.length; i++) {
      const labels = inputSubscripts[i].split('');
      inputLabels.push(labels);
      if (labels.length !== operands[i].shape.length) {
        throw new Error(`einsum: operand ${i} has ${operands[i].shape.length} dimensions but subscripts specify ${labels.length}`);
      }
      for (let j = 0; j < labels.length; j++) {
        const label = labels[j];
        const size = operands[i].shape[j];
        if (labelSizes.has(label)) {
          if (labelSizes.get(label) !== size) {
            throw new Error(`einsum: inconsistent size for label '${label}'`);
          }
        } else {
          labelSizes.set(label, size);
        }
      }
    }

    // Determine output labels
    let outputLabels: string[];
    if (outputStr !== undefined) {
      outputLabels = outputStr.split('');
    } else {
      // Implicit mode: output labels are those that appear exactly once
      const labelCounts: Map<string, number> = new Map();
      for (const labels of inputLabels) {
        for (const label of labels) {
          labelCounts.set(label, (labelCounts.get(label) || 0) + 1);
        }
      }
      outputLabels = [];
      const allLabels = Array.from(labelSizes.keys()).sort();
      for (const label of allLabels) {
        if (labelCounts.get(label) === 1) {
          outputLabels.push(label);
        }
      }
    }

    // Compute output shape
    const outputShape = outputLabels.map(l => labelSizes.get(l)!);
    const outputSize = outputShape.length === 0 ? 1 : outputShape.reduce((a, b) => a * b, 1);

    // Find contracted (summed) labels
    const outputSet = new Set(outputLabels);
    const allLabels = Array.from(labelSizes.keys());
    const contractedLabels = allLabels.filter(l => !outputSet.has(l));

    // Compute contracted dimensions size
    const contractedSizes = contractedLabels.map(l => labelSizes.get(l)!);
    const contractedTotal = contractedSizes.length === 0 ? 1 : contractedSizes.reduce((a, b) => a * b, 1);

    const result = new Float64Array(outputSize);

    // Compute input strides
    const inputStrides = operands.map(op => this._computeStrides(op.shape));

    // For each output position
    const outputStrides = outputShape.length === 0 ? [] : this._computeStrides(outputShape);

    for (let outIdx = 0; outIdx < outputSize; outIdx++) {
      // Convert to output coordinates
      const outCoords: Map<string, number> = new Map();
      let remaining = outIdx;
      for (let d = 0; d < outputLabels.length; d++) {
        const coord = Math.floor(remaining / outputStrides[d]);
        remaining = remaining % outputStrides[d];
        outCoords.set(outputLabels[d], coord);
      }

      // Sum over contracted indices
      let sum = 0;
      for (let contrIdx = 0; contrIdx < contractedTotal; contrIdx++) {
        // Convert to contracted coordinates
        const contrCoords: Map<string, number> = new Map();
        let contrRemaining = contrIdx;
        for (let d = 0; d < contractedLabels.length; d++) {
          const size = contractedSizes[d];
          const stride = d < contractedSizes.length - 1
            ? contractedSizes.slice(d + 1).reduce((a, b) => a * b, 1)
            : 1;
          const coord = Math.floor(contrRemaining / stride);
          contrRemaining = contrRemaining % stride;
          contrCoords.set(contractedLabels[d], coord);
        }

        // Merge coordinates
        const allCoords = new Map([...outCoords, ...contrCoords]);

        // Compute product of all operands at these coordinates
        let prod = 1;
        for (let i = 0; i < operands.length; i++) {
          const labels = inputLabels[i];
          const strides = inputStrides[i];
          let idx = 0;
          for (let d = 0; d < labels.length; d++) {
            idx += allCoords.get(labels[d])! * strides[d];
          }
          prod *= operands[i].data[idx];
        }
        sum += prod;
      }

      result[outIdx] = sum;
    }

    return new JsNDArray(result, outputShape.length === 0 ? [1] : outputShape);
  }
}

export function createJsBackend(): Backend {
  return new JsBackend();
}
