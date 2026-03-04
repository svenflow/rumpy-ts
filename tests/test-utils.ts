/**
 * Test utilities - matching rumpy-tests/src/lib.rs
 */

/** Check if two f64 values are approximately equal */
export function approxEq(a: number, b: number, tol: number): boolean {
  if (Number.isNaN(a) && Number.isNaN(b)) return true;
  if (!Number.isFinite(a) && !Number.isFinite(b)) {
    return Math.sign(a) === Math.sign(b);
  }
  return Math.abs(a - b) < tol;
}

/** Check if two arrays are approximately equal */
export function arraysApproxEq(
  a: Float64Array | number[],
  b: Float64Array | number[],
  tol: number
): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (!approxEq(a[i], b[i], tol)) return false;
  }
  return true;
}

/** Default tolerance for floating point comparisons */
export const DEFAULT_TOL = 1e-10;

/** Relaxed tolerance for operations with accumulated error */
export const RELAXED_TOL = 1e-6;

/** NDArray interface */
export interface NDArray {
  shape: number[];
  data: Float64Array;
  toArray(): number[];
}

/** Backend interface that all backends must implement */
export interface Backend {
  name: string;

  // ============ Creation ============
  zeros(shape: number[]): NDArray;
  ones(shape: number[]): NDArray;
  full(shape: number[], value: number): NDArray;
  arange(start: number, stop: number, step: number): NDArray;
  linspace(start: number, stop: number, num: number): NDArray;
  eye(n: number): NDArray;
  diag(arr: NDArray, k?: number): NDArray;
  array(data: number[], shape?: number[]): NDArray;

  // ============ Math - Unary ============
  sin(arr: NDArray): NDArray;
  cos(arr: NDArray): NDArray;
  tan(arr: NDArray): NDArray;
  arcsin(arr: NDArray): NDArray;
  arccos(arr: NDArray): NDArray;
  arctan(arr: NDArray): NDArray;
  sinh(arr: NDArray): NDArray;
  cosh(arr: NDArray): NDArray;
  tanh(arr: NDArray): NDArray;
  exp(arr: NDArray): NDArray;
  log(arr: NDArray): NDArray;
  log2(arr: NDArray): NDArray;
  log10(arr: NDArray): NDArray;
  sqrt(arr: NDArray): NDArray;
  cbrt(arr: NDArray): NDArray;
  abs(arr: NDArray): NDArray;
  sign(arr: NDArray): NDArray;
  floor(arr: NDArray): NDArray;
  ceil(arr: NDArray): NDArray;
  round(arr: NDArray): NDArray;
  neg(arr: NDArray): NDArray;
  reciprocal(arr: NDArray): NDArray;
  square(arr: NDArray): NDArray;

  // ============ Math - Binary ============
  add(a: NDArray, b: NDArray): NDArray;
  sub(a: NDArray, b: NDArray): NDArray;
  mul(a: NDArray, b: NDArray): NDArray;
  div(a: NDArray, b: NDArray): NDArray;
  pow(a: NDArray, b: NDArray): NDArray;
  maximum(a: NDArray, b: NDArray): NDArray;
  minimum(a: NDArray, b: NDArray): NDArray;

  // ============ Math - Scalar ============
  addScalar(arr: NDArray, scalar: number): NDArray;
  subScalar(arr: NDArray, scalar: number): NDArray;
  mulScalar(arr: NDArray, scalar: number): NDArray;
  divScalar(arr: NDArray, scalar: number): NDArray;
  powScalar(arr: NDArray, scalar: number): NDArray;
  clip(arr: NDArray, min: number, max: number): NDArray;

  // ============ Stats ============
  sum(arr: NDArray): number;
  prod(arr: NDArray): number;
  mean(arr: NDArray): number;
  var(arr: NDArray, ddof?: number): number;
  std(arr: NDArray, ddof?: number): number;
  min(arr: NDArray): number;
  max(arr: NDArray): number;
  argmin(arr: NDArray): number;
  argmax(arr: NDArray): number;
  cumsum(arr: NDArray): NDArray;
  cumprod(arr: NDArray): NDArray;
  all(arr: NDArray): boolean;
  any(arr: NDArray): boolean;
  sumAxis(arr: NDArray, axis: number): NDArray;
  meanAxis(arr: NDArray, axis: number): NDArray;

  // ============ Linalg ============
  matmul(a: NDArray, b: NDArray): NDArray;
  dot(a: NDArray, b: NDArray): NDArray;
  inner(a: NDArray, b: NDArray): number;
  outer(a: NDArray, b: NDArray): NDArray;
  transpose(arr: NDArray): NDArray;
  trace(arr: NDArray): number;
  det(arr: NDArray): number;
  inv(arr: NDArray): NDArray;
  solve(a: NDArray, b: NDArray): NDArray;
  norm(arr: NDArray, ord?: number): number;
  qr(arr: NDArray): { q: NDArray; r: NDArray };
  svd(arr: NDArray): { u: NDArray; s: NDArray; vt: NDArray };
}
