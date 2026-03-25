/* tslint:disable */
/* eslint-disable */

export function argsort_f64(data: Float64Array): Uint32Array;

export function binary_add(a_data: Float64Array, a_shape: Uint32Array, b_data: Float64Array, b_shape: Uint32Array): Float64Array;

export function binary_divide(a_data: Float64Array, a_shape: Uint32Array, b_data: Float64Array, b_shape: Uint32Array): Float64Array;

export function binary_maximum(a_data: Float64Array, a_shape: Uint32Array, b_data: Float64Array, b_shape: Uint32Array): Float64Array;

export function binary_minimum(a_data: Float64Array, a_shape: Uint32Array, b_data: Float64Array, b_shape: Uint32Array): Float64Array;

export function binary_mod(a_data: Float64Array, a_shape: Uint32Array, b_data: Float64Array, b_shape: Uint32Array): Float64Array;

export function binary_multiply(a_data: Float64Array, a_shape: Uint32Array, b_data: Float64Array, b_shape: Uint32Array): Float64Array;

export function binary_power(a_data: Float64Array, a_shape: Uint32Array, b_data: Float64Array, b_shape: Uint32Array): Float64Array;

export function binary_subtract(a_data: Float64Array, a_shape: Uint32Array, b_data: Float64Array, b_shape: Uint32Array): Float64Array;

/**
 * Returns the broadcast output shape as a Vec<u32>.
 * Used by the TypeScript side to know the result shape.
 */
export function broadcast_shape(a_shape: Uint32Array, b_shape: Uint32Array): Uint32Array;

export function matmul(a: Float64Array, m: number, k: number, b: Float64Array, n: number): Float64Array;

export function reduce_max(data: Float64Array): number;

export function reduce_max_axis(data: Float64Array, shape: Uint32Array, axis: number): Float64Array;

export function reduce_mean(data: Float64Array): number;

export function reduce_mean_axis(data: Float64Array, shape: Uint32Array, axis: number): Float64Array;

export function reduce_min(data: Float64Array): number;

export function reduce_min_axis(data: Float64Array, shape: Uint32Array, axis: number): Float64Array;

export function reduce_prod(data: Float64Array): number;

export function reduce_prod_axis(data: Float64Array, shape: Uint32Array, axis: number): Float64Array;

export function reduce_sum(data: Float64Array): number;

export function reduce_sum_axis(data: Float64Array, shape: Uint32Array, axis: number): Float64Array;

export function sort_f64(data: Float64Array): Float64Array;

export function unary_abs(data: Float64Array): Float64Array;

export function unary_acos(data: Float64Array): Float64Array;

export function unary_acosh(data: Float64Array): Float64Array;

export function unary_asin(data: Float64Array): Float64Array;

export function unary_asinh(data: Float64Array): Float64Array;

export function unary_atan(data: Float64Array): Float64Array;

export function unary_atanh(data: Float64Array): Float64Array;

export function unary_cbrt(data: Float64Array): Float64Array;

export function unary_ceil(data: Float64Array): Float64Array;

export function unary_cos(data: Float64Array): Float64Array;

export function unary_cosh(data: Float64Array): Float64Array;

export function unary_exp(data: Float64Array): Float64Array;

export function unary_expm1(data: Float64Array): Float64Array;

export function unary_floor(data: Float64Array): Float64Array;

export function unary_log(data: Float64Array): Float64Array;

export function unary_log10(data: Float64Array): Float64Array;

export function unary_log1p(data: Float64Array): Float64Array;

export function unary_log2(data: Float64Array): Float64Array;

export function unary_negative(data: Float64Array): Float64Array;

export function unary_reciprocal(data: Float64Array): Float64Array;

export function unary_round(data: Float64Array): Float64Array;

export function unary_sign(data: Float64Array): Float64Array;

export function unary_sin(data: Float64Array): Float64Array;

export function unary_sinh(data: Float64Array): Float64Array;

export function unary_sqrt(data: Float64Array): Float64Array;

export function unary_square(data: Float64Array): Float64Array;

export function unary_tan(data: Float64Array): Float64Array;

export function unary_tanh(data: Float64Array): Float64Array;

export function unary_trunc(data: Float64Array): Float64Array;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly argsort_f64: (a: number, b: number) => [number, number];
    readonly binary_add: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => [number, number];
    readonly binary_divide: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => [number, number];
    readonly binary_maximum: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => [number, number];
    readonly binary_minimum: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => [number, number];
    readonly binary_mod: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => [number, number];
    readonly binary_multiply: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => [number, number];
    readonly binary_power: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => [number, number];
    readonly binary_subtract: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => [number, number];
    readonly broadcast_shape: (a: number, b: number, c: number, d: number) => [number, number];
    readonly matmul: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number];
    readonly reduce_max: (a: number, b: number) => number;
    readonly reduce_max_axis: (a: number, b: number, c: number, d: number, e: number) => [number, number];
    readonly reduce_mean: (a: number, b: number) => number;
    readonly reduce_mean_axis: (a: number, b: number, c: number, d: number, e: number) => [number, number];
    readonly reduce_min: (a: number, b: number) => number;
    readonly reduce_min_axis: (a: number, b: number, c: number, d: number, e: number) => [number, number];
    readonly reduce_prod: (a: number, b: number) => number;
    readonly reduce_prod_axis: (a: number, b: number, c: number, d: number, e: number) => [number, number];
    readonly reduce_sum: (a: number, b: number) => number;
    readonly reduce_sum_axis: (a: number, b: number, c: number, d: number, e: number) => [number, number];
    readonly sort_f64: (a: number, b: number) => [number, number];
    readonly unary_abs: (a: number, b: number) => [number, number];
    readonly unary_acos: (a: number, b: number) => [number, number];
    readonly unary_acosh: (a: number, b: number) => [number, number];
    readonly unary_asin: (a: number, b: number) => [number, number];
    readonly unary_asinh: (a: number, b: number) => [number, number];
    readonly unary_atan: (a: number, b: number) => [number, number];
    readonly unary_atanh: (a: number, b: number) => [number, number];
    readonly unary_cbrt: (a: number, b: number) => [number, number];
    readonly unary_ceil: (a: number, b: number) => [number, number];
    readonly unary_cos: (a: number, b: number) => [number, number];
    readonly unary_cosh: (a: number, b: number) => [number, number];
    readonly unary_exp: (a: number, b: number) => [number, number];
    readonly unary_expm1: (a: number, b: number) => [number, number];
    readonly unary_floor: (a: number, b: number) => [number, number];
    readonly unary_log: (a: number, b: number) => [number, number];
    readonly unary_log10: (a: number, b: number) => [number, number];
    readonly unary_log1p: (a: number, b: number) => [number, number];
    readonly unary_log2: (a: number, b: number) => [number, number];
    readonly unary_negative: (a: number, b: number) => [number, number];
    readonly unary_reciprocal: (a: number, b: number) => [number, number];
    readonly unary_round: (a: number, b: number) => [number, number];
    readonly unary_sign: (a: number, b: number) => [number, number];
    readonly unary_sin: (a: number, b: number) => [number, number];
    readonly unary_sinh: (a: number, b: number) => [number, number];
    readonly unary_sqrt: (a: number, b: number) => [number, number];
    readonly unary_square: (a: number, b: number) => [number, number];
    readonly unary_tan: (a: number, b: number) => [number, number];
    readonly unary_tanh: (a: number, b: number) => [number, number];
    readonly unary_trunc: (a: number, b: number) => [number, number];
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
