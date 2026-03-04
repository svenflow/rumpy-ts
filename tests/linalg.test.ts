/**
 * Linear algebra tests - NumPy compatible
 * Mirrors: crates/rumpy-tests/src/linalg.rs
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { Backend, DEFAULT_TOL, RELAXED_TOL, approxEq } from './test-utils';

export function linalgTests(getBackend: () => Backend) {
  describe('linalg', () => {
    let B: Backend;
    beforeAll(() => {
      B = getBackend();
    });

    const mat = (data: number[], rows: number, cols: number) =>
      B.array(data, [rows, cols]);
    const vec1d = (data: number[]) => B.array(data, [data.length]);

    // ============ matmul ============

    describe('matmul', () => {
      it('multiplies 2x2 matrices', () => {
        const a = mat([1.0, 2.0, 3.0, 4.0], 2, 2);
        const b = mat([5.0, 6.0, 7.0, 8.0], 2, 2);
        const c = B.matmul(a, b);

        // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
        expect(c.toArray()).toEqual([19.0, 22.0, 43.0, 50.0]);
      });

      it('multiplies 2x3 and 3x2 matrices', () => {
        const a = mat([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        const b = mat([7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 3, 2);
        const c = B.matmul(a, b);

        expect(c.shape).toEqual([2, 2]);
        expect(c.toArray()).toEqual([58.0, 64.0, 139.0, 154.0]);
      });

      it('throws on dimension mismatch', () => {
        const a = mat([1.0, 2.0, 3.0, 4.0], 2, 2);
        const b = mat([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        expect(() => B.matmul(a, b)).toThrow();
      });
    });

    // ============ dot ============

    describe('dot', () => {
      it('computes dot product of vectors', () => {
        const a = vec1d([1.0, 2.0, 3.0]);
        const b = vec1d([4.0, 5.0, 6.0]);
        const c = B.dot(a, b);
        // 1*4 + 2*5 + 3*6 = 32
        expect(c.toArray()[0]).toBe(32.0);
      });

      it('computes matmul for 2D arrays', () => {
        const a = mat([1.0, 2.0, 3.0, 4.0], 2, 2);
        const b = mat([5.0, 6.0, 7.0, 8.0], 2, 2);
        const c = B.dot(a, b);
        expect(c.toArray()).toEqual([19.0, 22.0, 43.0, 50.0]);
      });
    });

    // ============ inner ============

    describe('inner', () => {
      it('computes inner product', () => {
        const a = vec1d([1.0, 2.0, 3.0]);
        const b = vec1d([4.0, 5.0, 6.0]);
        const result = B.inner(a, b);
        expect(result).toBe(32.0);
      });
    });

    // ============ outer ============

    describe('outer', () => {
      it('computes outer product', () => {
        const a = vec1d([1.0, 2.0]);
        const b = vec1d([3.0, 4.0, 5.0]);
        const c = B.outer(a, b);

        expect(c.shape).toEqual([2, 3]);
        expect(c.toArray()).toEqual([3.0, 4.0, 5.0, 6.0, 8.0, 10.0]);
      });
    });

    // ============ inv ============

    describe('inv', () => {
      it('computes inverse of 2x2 matrix', () => {
        const a = mat([4.0, 7.0, 2.0, 6.0], 2, 2);
        const aInv = B.inv(a);

        // A @ A^-1 should be identity
        const identity = B.matmul(a, aInv);
        const data = identity.toArray();
        expect(approxEq(data[0], 1.0, RELAXED_TOL)).toBe(true);
        expect(approxEq(data[1], 0.0, RELAXED_TOL)).toBe(true);
        expect(approxEq(data[2], 0.0, RELAXED_TOL)).toBe(true);
        expect(approxEq(data[3], 1.0, RELAXED_TOL)).toBe(true);
      });

      it('computes inverse of 3x3 matrix', () => {
        const a = mat([1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0], 3, 3);
        const aInv = B.inv(a);

        // A @ A^-1 should be identity
        const identity = B.matmul(a, aInv);
        const data = identity.toArray();
        expect(approxEq(data[0], 1.0, RELAXED_TOL)).toBe(true);
        expect(approxEq(data[4], 1.0, RELAXED_TOL)).toBe(true);
        expect(approxEq(data[8], 1.0, RELAXED_TOL)).toBe(true);
      });

      it('throws for non-square matrix', () => {
        const a = mat([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        expect(() => B.inv(a)).toThrow();
      });
    });

    // ============ det ============

    describe('det', () => {
      it('computes determinant of 2x2 matrix', () => {
        const a = mat([1.0, 2.0, 3.0, 4.0], 2, 2);
        const det = B.det(a);
        // det([[1,2],[3,4]]) = 1*4 - 2*3 = -2
        expect(approxEq(det, -2.0, RELAXED_TOL)).toBe(true);
      });

      it('computes determinant of 3x3 singular matrix', () => {
        const a = mat([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3, 3);
        const det = B.det(a);
        // This matrix is singular, det = 0
        expect(approxEq(det, 0.0, RELAXED_TOL)).toBe(true);
      });

      it('computes determinant of identity matrix', () => {
        const a = B.eye(3);
        const det = B.det(a);
        expect(approxEq(det, 1.0, RELAXED_TOL)).toBe(true);
      });
    });

    // ============ trace ============

    describe('trace', () => {
      it('computes trace of matrix', () => {
        const a = mat([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3, 3);
        const tr = B.trace(a);
        expect(tr).toBe(15.0); // 1 + 5 + 9
      });

      it('computes trace of identity', () => {
        const a = B.eye(5);
        const tr = B.trace(a);
        expect(tr).toBe(5.0);
      });
    });

    // ============ norm ============

    describe('norm', () => {
      it('computes L2 norm', () => {
        const a = vec1d([3.0, 4.0]);
        const n = B.norm(a, 2);
        expect(approxEq(n, 5.0, DEFAULT_TOL)).toBe(true);
      });

      it('computes L1 norm', () => {
        const a = vec1d([-3.0, 4.0]);
        const n = B.norm(a, 1);
        expect(approxEq(n, 7.0, DEFAULT_TOL)).toBe(true);
      });

      it('computes L-infinity norm', () => {
        const a = vec1d([-3.0, 4.0, 2.0]);
        const n = B.norm(a, Infinity);
        expect(approxEq(n, 4.0, DEFAULT_TOL)).toBe(true);
      });
    });

    // ============ solve ============

    describe('solve', () => {
      it('solves linear system', () => {
        // Solve Ax = b where A = [[3,1],[1,2]], b = [[9],[8]]
        // Solution: x = [[2],[3]]
        const a = mat([3.0, 1.0, 1.0, 2.0], 2, 2);
        const b = mat([9.0, 8.0], 2, 1);
        const x = B.solve(a, b);

        expect(approxEq(x.toArray()[0], 2.0, RELAXED_TOL)).toBe(true);
        expect(approxEq(x.toArray()[1], 3.0, RELAXED_TOL)).toBe(true);
      });

      it('verifies A @ x = b', () => {
        const a = mat([3.0, 1.0, 1.0, 2.0], 2, 2);
        const b = mat([9.0, 8.0], 2, 1);
        const x = B.solve(a, b);

        // Reshape x for matmul (solve may return 1D)
        const xData = x.toArray();
        const x2d = B.array(xData, [xData.length, 1]);

        // Verify A @ x = b
        const ax = B.matmul(a, x2d);
        expect(approxEq(ax.toArray()[0], 9.0, RELAXED_TOL)).toBe(true);
        expect(approxEq(ax.toArray()[1], 8.0, RELAXED_TOL)).toBe(true);
      });
    });

    // ============ qr ============

    describe('qr', () => {
      it.skip('computes QR decomposition', () => {
        const a = mat([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        const { q, r } = B.qr(a);

        // Q @ R should equal A (approximately)
        const reconstructed = B.matmul(q, r);
        const aData = a.toArray();
        const recData = reconstructed.toArray();
        for (let i = 0; i < aData.length; i++) {
          expect(approxEq(aData[i], recData[i], RELAXED_TOL)).toBe(true);
        }
      });

      it.skip('Q is orthogonal (Q @ Q^T = I)', () => {
        const a = mat([1.0, 2.0, 3.0, 4.0], 2, 2);
        const { q } = B.qr(a);

        // Q @ Q^T should be identity
        const qt = B.transpose(q);
        const qqt = B.matmul(q, qt);
        const data = qqt.toArray();
        expect(approxEq(data[0], 1.0, RELAXED_TOL)).toBe(true);
        expect(approxEq(data[1], 0.0, RELAXED_TOL)).toBe(true);
        expect(approxEq(data[2], 0.0, RELAXED_TOL)).toBe(true);
        expect(approxEq(data[3], 1.0, RELAXED_TOL)).toBe(true);
      });
    });

    // ============ svd ============

    describe('svd', () => {
      it.skip('computes SVD with correct shapes', () => {
        const a = mat([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        const { u, s, vt } = B.svd(a);

        // Verify shapes
        expect(u.shape[0]).toBe(2);
        expect(s.shape[0]).toBe(2);
        expect(vt.shape[1]).toBe(3);

        // Singular values should be non-negative
        const sData = s.toArray();
        expect(sData.every((x) => x >= 0)).toBe(true);
      });
    });

    // ============ transpose ============

    describe('transpose', () => {
      it('transposes 2x3 matrix', () => {
        const a = mat([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        const at = B.transpose(a);

        expect(at.shape).toEqual([3, 2]);
        expect(at.toArray()).toEqual([1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
      });

      it('transposes square matrix', () => {
        const a = mat([1.0, 2.0, 3.0, 4.0], 2, 2);
        const at = B.transpose(a);
        expect(at.toArray()).toEqual([1.0, 3.0, 2.0, 4.0]);
      });

      it('transpose is no-op for 1D', () => {
        const a = vec1d([1.0, 2.0, 3.0]);
        const at = B.transpose(a);
        expect(at.shape).toEqual(a.shape);
        expect(at.toArray()).toEqual(a.toArray());
      });

      it('double transpose returns original', () => {
        const a = mat([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        const att = B.transpose(B.transpose(a));
        expect(att.shape).toEqual(a.shape);
        expect(att.toArray()).toEqual(a.toArray());
      });
    });
  });
}
