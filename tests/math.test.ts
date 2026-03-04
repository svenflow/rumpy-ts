/**
 * Math function tests - NumPy compatible
 * Mirrors: crates/rumpy-tests/src/math.rs
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { Backend, DEFAULT_TOL, RELAXED_TOL, approxEq } from './test-utils';

export function mathTests(getBackend: () => Backend) {
  describe('math', () => {
    let B: Backend;
    beforeAll(() => {
      B = getBackend();
    });

    const arr = (data: number[]) => B.array(data, [data.length]);

    // ============ Trigonometric ============

    describe('trigonometric', () => {
      it('computes sin', () => {
        const a = arr([0.0, Math.PI / 6, Math.PI / 4, Math.PI / 3, Math.PI / 2, Math.PI]);
        const result = B.sin(a);
        const data = result.toArray();

        expect(approxEq(data[0], 0.0, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[1], 0.5, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[2], Math.sqrt(2) / 2, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[3], Math.sqrt(3) / 2, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[4], 1.0, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[5], 0.0, DEFAULT_TOL)).toBe(true);
      });

      it('computes cos', () => {
        const a = arr([0.0, Math.PI / 3, Math.PI / 2, Math.PI]);
        const result = B.cos(a);
        const data = result.toArray();

        expect(approxEq(data[0], 1.0, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[1], 0.5, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[2], 0.0, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[3], -1.0, DEFAULT_TOL)).toBe(true);
      });

      it('computes tan', () => {
        const a = arr([0.0, Math.PI / 4]);
        const result = B.tan(a);
        const data = result.toArray();

        expect(approxEq(data[0], 0.0, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[1], 1.0, DEFAULT_TOL)).toBe(true);
      });

      it('computes arcsin', () => {
        const a = arr([0.0, 0.5, 1.0]);
        const result = B.arcsin(a);
        const data = result.toArray();

        expect(approxEq(data[0], 0.0, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[1], Math.PI / 6, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[2], Math.PI / 2, DEFAULT_TOL)).toBe(true);
      });
    });

    // ============ Hyperbolic ============

    describe('hyperbolic', () => {
      it('computes sinh and cosh', () => {
        const a = arr([0.0, 1.0, 2.0]);
        const sinh = B.sinh(a);
        const cosh = B.cosh(a);

        // sinh(0) = 0, cosh(0) = 1
        expect(approxEq(sinh.toArray()[0], 0.0, DEFAULT_TOL)).toBe(true);
        expect(approxEq(cosh.toArray()[0], 1.0, DEFAULT_TOL)).toBe(true);

        // Identity: cosh^2 - sinh^2 = 1
        for (let i = 0; i < 3; i++) {
          const s = sinh.toArray()[i];
          const c = cosh.toArray()[i];
          expect(approxEq(c * c - s * s, 1.0, DEFAULT_TOL)).toBe(true);
        }
      });

      it('computes tanh', () => {
        const a = arr([0.0, 1.0, -1.0, 10.0, -10.0]);
        const result = B.tanh(a);
        const data = result.toArray();

        expect(approxEq(data[0], 0.0, DEFAULT_TOL)).toBe(true);
        expect(data[1] > 0.0 && data[1] < 1.0).toBe(true);
        expect(data[2] < 0.0 && data[2] > -1.0).toBe(true);
        expect(approxEq(data[3], 1.0, RELAXED_TOL)).toBe(true); // tanh saturates
        expect(approxEq(data[4], -1.0, RELAXED_TOL)).toBe(true);
      });
    });

    // ============ Exponential and Logarithmic ============

    describe('exponential and logarithmic', () => {
      it('computes exp', () => {
        const a = arr([0.0, 1.0, 2.0, -1.0]);
        const result = B.exp(a);
        const data = result.toArray();

        expect(approxEq(data[0], 1.0, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[1], Math.E, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[2], Math.E * Math.E, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[3], 1.0 / Math.E, DEFAULT_TOL)).toBe(true);
      });

      it('computes log', () => {
        const a = arr([1.0, Math.E, Math.E * Math.E]);
        const result = B.log(a);
        const data = result.toArray();

        expect(approxEq(data[0], 0.0, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[1], 1.0, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[2], 2.0, DEFAULT_TOL)).toBe(true);
      });

      it('returns NaN for log of negative', () => {
        const a = arr([-1.0]);
        const result = B.log(a);
        expect(Number.isNaN(result.toArray()[0])).toBe(true);
      });

      it('exp and log are inverse operations', () => {
        const a = arr([0.5, 1.0, 2.0, 5.0, 10.0]);
        const expA = B.exp(a);
        const logExpA = B.log(expA);

        const aData = a.toArray();
        const resultData = logExpA.toArray();
        for (let i = 0; i < aData.length; i++) {
          expect(approxEq(aData[i], resultData[i], DEFAULT_TOL)).toBe(true);
        }
      });

      it('computes log2', () => {
        const a = arr([1.0, 2.0, 4.0, 8.0]);
        const result = B.log2(a);
        const data = result.toArray();

        expect(approxEq(data[0], 0.0, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[1], 1.0, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[2], 2.0, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[3], 3.0, DEFAULT_TOL)).toBe(true);
      });

      it('computes log10', () => {
        const a = arr([1.0, 10.0, 100.0, 1000.0]);
        const result = B.log10(a);
        const data = result.toArray();

        expect(approxEq(data[0], 0.0, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[1], 1.0, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[2], 2.0, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[3], 3.0, DEFAULT_TOL)).toBe(true);
      });

      it('computes sqrt', () => {
        const a = arr([0.0, 1.0, 4.0, 9.0, 16.0, 25.0]);
        const result = B.sqrt(a);
        expect(result.toArray()).toEqual([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
      });

      it('returns NaN for sqrt of negative', () => {
        const a = arr([-1.0]);
        const result = B.sqrt(a);
        expect(Number.isNaN(result.toArray()[0])).toBe(true);
      });

      it('computes cbrt', () => {
        const a = arr([0.0, 1.0, 8.0, 27.0, -8.0]);
        const result = B.cbrt(a);
        const data = result.toArray();

        expect(approxEq(data[0], 0.0, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[1], 1.0, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[2], 2.0, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[3], 3.0, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[4], -2.0, DEFAULT_TOL)).toBe(true);
      });

      it('computes square', () => {
        const a = arr([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]);
        const result = B.square(a);
        expect(result.toArray()).toEqual([9.0, 4.0, 1.0, 0.0, 1.0, 4.0, 9.0]);
      });
    });

    // ============ Rounding ============

    describe('rounding', () => {
      it('computes floor', () => {
        const a = arr([-2.7, -0.5, 0.0, 0.5, 2.7]);
        const result = B.floor(a);
        expect(result.toArray()).toEqual([-3.0, -1.0, 0.0, 0.0, 2.0]);
      });

      it('computes ceil', () => {
        const a = arr([-2.7, -0.5, 0.0, 0.5, 2.7]);
        const result = B.ceil(a);
        const data = result.toArray();
        expect(approxEq(data[0], -2.0, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[1], 0.0, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[2], 0.0, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[3], 1.0, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[4], 3.0, DEFAULT_TOL)).toBe(true);
      });

      it('computes round', () => {
        const a = arr([-2.7, -0.5, 0.0, 0.5, 2.7]);
        const result = B.round(a);
        const data = result.toArray();
        expect(data[0]).toBe(-3.0);
        expect(data[2]).toBe(0.0);
        expect(data[4]).toBe(3.0);
      });
    });

    // ============ Other Unary ============

    describe('other unary', () => {
      it('computes abs', () => {
        const a = arr([-5.0, -2.5, 0.0, 2.5, 5.0]);
        const result = B.abs(a);
        expect(result.toArray()).toEqual([5.0, 2.5, 0.0, 2.5, 5.0]);
      });

      it('computes sign', () => {
        const a = arr([-5.0, -0.5, 0.0, 0.5, 5.0]);
        const result = B.sign(a);
        expect(result.toArray()).toEqual([-1.0, -1.0, 0.0, 1.0, 1.0]);
      });

      it('computes neg', () => {
        const a = arr([-2.0, -1.0, 0.0, 1.0, 2.0]);
        const result = B.neg(a);
        const data = result.toArray();
        expect(approxEq(data[0], 2.0, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[1], 1.0, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[2], 0.0, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[3], -1.0, DEFAULT_TOL)).toBe(true);
        expect(approxEq(data[4], -2.0, DEFAULT_TOL)).toBe(true);
      });

      it('computes reciprocal', () => {
        const a = arr([1.0, 2.0, 4.0, 0.5]);
        const result = B.reciprocal(a);
        expect(result.toArray()).toEqual([1.0, 0.5, 0.25, 2.0]);
      });
    });

    // ============ Binary Operations ============

    describe('binary operations', () => {
      it('adds arrays', () => {
        const a = arr([1.0, 2.0, 3.0]);
        const b = arr([4.0, 5.0, 6.0]);
        const result = B.add(a, b);
        expect(result.toArray()).toEqual([5.0, 7.0, 9.0]);
      });

      it('subtracts arrays', () => {
        const a = arr([5.0, 7.0, 9.0]);
        const b = arr([1.0, 2.0, 3.0]);
        const result = B.sub(a, b);
        expect(result.toArray()).toEqual([4.0, 5.0, 6.0]);
      });

      it('multiplies arrays element-wise', () => {
        const a = arr([1.0, 2.0, 3.0]);
        const b = arr([2.0, 3.0, 4.0]);
        const result = B.mul(a, b);
        expect(result.toArray()).toEqual([2.0, 6.0, 12.0]);
      });

      it('divides arrays element-wise', () => {
        const a = arr([4.0, 9.0, 16.0]);
        const b = arr([2.0, 3.0, 4.0]);
        const result = B.div(a, b);
        expect(result.toArray()).toEqual([2.0, 3.0, 4.0]);
      });

      it('raises to power', () => {
        const a = arr([2.0, 3.0, 4.0]);
        const b = arr([2.0, 2.0, 2.0]);
        const result = B.pow(a, b);
        expect(result.toArray()).toEqual([4.0, 9.0, 16.0]);
      });

      it('computes maximum', () => {
        const a = arr([1.0, 5.0, 3.0]);
        const b = arr([2.0, 3.0, 4.0]);
        const result = B.maximum(a, b);
        expect(result.toArray()).toEqual([2.0, 5.0, 4.0]);
      });

      it('computes minimum', () => {
        const a = arr([1.0, 5.0, 3.0]);
        const b = arr([2.0, 3.0, 4.0]);
        const result = B.minimum(a, b);
        expect(result.toArray()).toEqual([1.0, 3.0, 3.0]);
      });

      it('throws on shape mismatch', () => {
        const a = arr([1.0, 2.0, 3.0]);
        const b = arr([1.0, 2.0]);
        expect(() => B.add(a, b)).toThrow();
        expect(() => B.sub(a, b)).toThrow();
        expect(() => B.mul(a, b)).toThrow();
        expect(() => B.div(a, b)).toThrow();
      });
    });

    // ============ Scalar Operations ============

    describe('scalar operations', () => {
      it('adds scalar', () => {
        const a = arr([1.0, 2.0, 3.0]);
        const result = B.addScalar(a, 10.0);
        expect(result.toArray()).toEqual([11.0, 12.0, 13.0]);
      });

      it('multiplies by scalar', () => {
        const a = arr([1.0, 2.0, 3.0]);
        const result = B.mulScalar(a, 2.0);
        expect(result.toArray()).toEqual([2.0, 4.0, 6.0]);
      });

      it('raises to scalar power', () => {
        const a = arr([1.0, 2.0, 3.0, 4.0]);
        const result = B.powScalar(a, 2.0);
        expect(result.toArray()).toEqual([1.0, 4.0, 9.0, 16.0]);
      });

      it('clips values to range', () => {
        const a = arr([-5.0, 0.0, 5.0, 10.0, 15.0]);
        const result = B.clip(a, 0.0, 10.0);
        expect(result.toArray()).toEqual([0.0, 0.0, 5.0, 10.0, 10.0]);
      });
    });
  });
}
