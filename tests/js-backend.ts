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
}

export function createJsBackend(): Backend {
  return new JsBackend();
}
