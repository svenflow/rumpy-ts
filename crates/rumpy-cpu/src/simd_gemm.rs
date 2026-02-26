//! WASM SIMD-optimized GEMM kernel
//!
//! Implements XNNPACK-style matrix multiplication using WASM simd128 intrinsics.
//! Uses f32 for 4 elements per v128 vector (matching XNNPACK).
//!
//! Key optimizations:
//! - 6x8 micro-kernel (like XNNPACK)
//! - Vectorized A load: load 4 A values at once, shuffle to broadcast each
//! - Matrix packing for B (XNNPACK-style: interleaved by K blocks of 4)
//! - SIMD vectorized inner loop
//! - FMA (fused multiply-add) via relaxed-simd for better throughput

#[cfg(target_arch = "wasm32")]
use std::arch::wasm32::*;

/// XNNPACK-style 6x8 kernel with vectorized A loading
/// Loads 4 A values at once per row, then shuffles to broadcast each lane.
/// This processes 4 K iterations per loop, reducing memory operations.
///
/// The weights (B) are expected to be packed: for each K block of 4,
/// 8 columns are stored contiguously: [k0:col0-7][k1:col0-7][k2:col0-7][k3:col0-7]
#[cfg(target_arch = "wasm32")]
pub unsafe fn matmul_simd_f32_xnnpack_style(
    a: &[f32],
    packed_b: &[f32],  // Pre-packed B in XNNPACK format
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize
) {
    const MR: usize = 6;
    const NR: usize = 8;

    let m_main = m / MR * MR;
    let n_panels = n / NR;
    let k_main = k / 4 * 4;  // Round down to multiple of 4

    for i in (0..m_main).step_by(MR) {
        // Pointers to A rows
        let a0_ptr = a.as_ptr().add(i * k);
        let a1_ptr = a.as_ptr().add((i + 1) * k);
        let a2_ptr = a.as_ptr().add((i + 2) * k);
        let a3_ptr = a.as_ptr().add((i + 3) * k);
        let a4_ptr = a.as_ptr().add((i + 4) * k);
        let a5_ptr = a.as_ptr().add((i + 5) * k);

        for panel in 0..n_panels {
            let j = panel * NR;
            let mut w_ptr = packed_b.as_ptr().add(panel * k * NR);

            // 6 rows × 8 cols = 12 accumulators
            let mut acc00 = f32x4_splat(0.0);
            let mut acc01 = f32x4_splat(0.0);
            let mut acc10 = f32x4_splat(0.0);
            let mut acc11 = f32x4_splat(0.0);
            let mut acc20 = f32x4_splat(0.0);
            let mut acc21 = f32x4_splat(0.0);
            let mut acc30 = f32x4_splat(0.0);
            let mut acc31 = f32x4_splat(0.0);
            let mut acc40 = f32x4_splat(0.0);
            let mut acc41 = f32x4_splat(0.0);
            let mut acc50 = f32x4_splat(0.0);
            let mut acc51 = f32x4_splat(0.0);

            let mut kk = 0;
            while kk < k_main {
                // Load 4 A values at once for each row
                let va0 = v128_load(a0_ptr.add(kk) as *const v128);
                let va1 = v128_load(a1_ptr.add(kk) as *const v128);
                let va2 = v128_load(a2_ptr.add(kk) as *const v128);
                let va3 = v128_load(a3_ptr.add(kk) as *const v128);
                let va4 = v128_load(a4_ptr.add(kk) as *const v128);
                let va5 = v128_load(a5_ptr.add(kk) as *const v128);

                // K iteration 0: broadcast lane 0
                let va0c0 = i32x4_shuffle::<0, 0, 0, 0>(va0, va0);
                let va1c0 = i32x4_shuffle::<0, 0, 0, 0>(va1, va1);
                let va2c0 = i32x4_shuffle::<0, 0, 0, 0>(va2, va2);
                let va3c0 = i32x4_shuffle::<0, 0, 0, 0>(va3, va3);
                let va4c0 = i32x4_shuffle::<0, 0, 0, 0>(va4, va4);
                let va5c0 = i32x4_shuffle::<0, 0, 0, 0>(va5, va5);

                let vb0123c0 = v128_load(w_ptr as *const v128);
                let vb4567c0 = v128_load(w_ptr.add(4) as *const v128);

                acc00 = f32x4_relaxed_madd(va0c0, vb0123c0, acc00);
                acc01 = f32x4_relaxed_madd(va0c0, vb4567c0, acc01);
                acc10 = f32x4_relaxed_madd(va1c0, vb0123c0, acc10);
                acc11 = f32x4_relaxed_madd(va1c0, vb4567c0, acc11);
                acc20 = f32x4_relaxed_madd(va2c0, vb0123c0, acc20);
                acc21 = f32x4_relaxed_madd(va2c0, vb4567c0, acc21);
                acc30 = f32x4_relaxed_madd(va3c0, vb0123c0, acc30);
                acc31 = f32x4_relaxed_madd(va3c0, vb4567c0, acc31);
                acc40 = f32x4_relaxed_madd(va4c0, vb0123c0, acc40);
                acc41 = f32x4_relaxed_madd(va4c0, vb4567c0, acc41);
                acc50 = f32x4_relaxed_madd(va5c0, vb0123c0, acc50);
                acc51 = f32x4_relaxed_madd(va5c0, vb4567c0, acc51);

                // K iteration 1: broadcast lane 1
                let va0c1 = i32x4_shuffle::<1, 1, 1, 1>(va0, va0);
                let va1c1 = i32x4_shuffle::<1, 1, 1, 1>(va1, va1);
                let va2c1 = i32x4_shuffle::<1, 1, 1, 1>(va2, va2);
                let va3c1 = i32x4_shuffle::<1, 1, 1, 1>(va3, va3);
                let va4c1 = i32x4_shuffle::<1, 1, 1, 1>(va4, va4);
                let va5c1 = i32x4_shuffle::<1, 1, 1, 1>(va5, va5);

                let vb0123c1 = v128_load(w_ptr.add(8) as *const v128);
                let vb4567c1 = v128_load(w_ptr.add(12) as *const v128);

                acc00 = f32x4_relaxed_madd(va0c1, vb0123c1, acc00);
                acc01 = f32x4_relaxed_madd(va0c1, vb4567c1, acc01);
                acc10 = f32x4_relaxed_madd(va1c1, vb0123c1, acc10);
                acc11 = f32x4_relaxed_madd(va1c1, vb4567c1, acc11);
                acc20 = f32x4_relaxed_madd(va2c1, vb0123c1, acc20);
                acc21 = f32x4_relaxed_madd(va2c1, vb4567c1, acc21);
                acc30 = f32x4_relaxed_madd(va3c1, vb0123c1, acc30);
                acc31 = f32x4_relaxed_madd(va3c1, vb4567c1, acc31);
                acc40 = f32x4_relaxed_madd(va4c1, vb0123c1, acc40);
                acc41 = f32x4_relaxed_madd(va4c1, vb4567c1, acc41);
                acc50 = f32x4_relaxed_madd(va5c1, vb0123c1, acc50);
                acc51 = f32x4_relaxed_madd(va5c1, vb4567c1, acc51);

                // K iteration 2: broadcast lane 2
                let va0c2 = i32x4_shuffle::<2, 2, 2, 2>(va0, va0);
                let va1c2 = i32x4_shuffle::<2, 2, 2, 2>(va1, va1);
                let va2c2 = i32x4_shuffle::<2, 2, 2, 2>(va2, va2);
                let va3c2 = i32x4_shuffle::<2, 2, 2, 2>(va3, va3);
                let va4c2 = i32x4_shuffle::<2, 2, 2, 2>(va4, va4);
                let va5c2 = i32x4_shuffle::<2, 2, 2, 2>(va5, va5);

                let vb0123c2 = v128_load(w_ptr.add(16) as *const v128);
                let vb4567c2 = v128_load(w_ptr.add(20) as *const v128);

                acc00 = f32x4_relaxed_madd(va0c2, vb0123c2, acc00);
                acc01 = f32x4_relaxed_madd(va0c2, vb4567c2, acc01);
                acc10 = f32x4_relaxed_madd(va1c2, vb0123c2, acc10);
                acc11 = f32x4_relaxed_madd(va1c2, vb4567c2, acc11);
                acc20 = f32x4_relaxed_madd(va2c2, vb0123c2, acc20);
                acc21 = f32x4_relaxed_madd(va2c2, vb4567c2, acc21);
                acc30 = f32x4_relaxed_madd(va3c2, vb0123c2, acc30);
                acc31 = f32x4_relaxed_madd(va3c2, vb4567c2, acc31);
                acc40 = f32x4_relaxed_madd(va4c2, vb0123c2, acc40);
                acc41 = f32x4_relaxed_madd(va4c2, vb4567c2, acc41);
                acc50 = f32x4_relaxed_madd(va5c2, vb0123c2, acc50);
                acc51 = f32x4_relaxed_madd(va5c2, vb4567c2, acc51);

                // K iteration 3: broadcast lane 3
                let va0c3 = i32x4_shuffle::<3, 3, 3, 3>(va0, va0);
                let va1c3 = i32x4_shuffle::<3, 3, 3, 3>(va1, va1);
                let va2c3 = i32x4_shuffle::<3, 3, 3, 3>(va2, va2);
                let va3c3 = i32x4_shuffle::<3, 3, 3, 3>(va3, va3);
                let va4c3 = i32x4_shuffle::<3, 3, 3, 3>(va4, va4);
                let va5c3 = i32x4_shuffle::<3, 3, 3, 3>(va5, va5);

                let vb0123c3 = v128_load(w_ptr.add(24) as *const v128);
                let vb4567c3 = v128_load(w_ptr.add(28) as *const v128);

                acc00 = f32x4_relaxed_madd(va0c3, vb0123c3, acc00);
                acc01 = f32x4_relaxed_madd(va0c3, vb4567c3, acc01);
                acc10 = f32x4_relaxed_madd(va1c3, vb0123c3, acc10);
                acc11 = f32x4_relaxed_madd(va1c3, vb4567c3, acc11);
                acc20 = f32x4_relaxed_madd(va2c3, vb0123c3, acc20);
                acc21 = f32x4_relaxed_madd(va2c3, vb4567c3, acc21);
                acc30 = f32x4_relaxed_madd(va3c3, vb0123c3, acc30);
                acc31 = f32x4_relaxed_madd(va3c3, vb4567c3, acc31);
                acc40 = f32x4_relaxed_madd(va4c3, vb0123c3, acc40);
                acc41 = f32x4_relaxed_madd(va4c3, vb4567c3, acc41);
                acc50 = f32x4_relaxed_madd(va5c3, vb0123c3, acc50);
                acc51 = f32x4_relaxed_madd(va5c3, vb4567c3, acc51);

                w_ptr = w_ptr.add(32);  // 4 k values × 8 cols
                kk += 4;
            }

            // Handle remaining K iterations (0-3)
            while kk < k {
                let a0 = f32x4_splat(*a0_ptr.add(kk));
                let a1 = f32x4_splat(*a1_ptr.add(kk));
                let a2 = f32x4_splat(*a2_ptr.add(kk));
                let a3 = f32x4_splat(*a3_ptr.add(kk));
                let a4 = f32x4_splat(*a4_ptr.add(kk));
                let a5 = f32x4_splat(*a5_ptr.add(kk));
                let b0 = v128_load(w_ptr as *const v128);
                let b1 = v128_load(w_ptr.add(4) as *const v128);
                acc00 = f32x4_relaxed_madd(a0, b0, acc00);
                acc01 = f32x4_relaxed_madd(a0, b1, acc01);
                acc10 = f32x4_relaxed_madd(a1, b0, acc10);
                acc11 = f32x4_relaxed_madd(a1, b1, acc11);
                acc20 = f32x4_relaxed_madd(a2, b0, acc20);
                acc21 = f32x4_relaxed_madd(a2, b1, acc21);
                acc30 = f32x4_relaxed_madd(a3, b0, acc30);
                acc31 = f32x4_relaxed_madd(a3, b1, acc31);
                acc40 = f32x4_relaxed_madd(a4, b0, acc40);
                acc41 = f32x4_relaxed_madd(a4, b1, acc41);
                acc50 = f32x4_relaxed_madd(a5, b0, acc50);
                acc51 = f32x4_relaxed_madd(a5, b1, acc51);
                w_ptr = w_ptr.add(8);
                kk += 1;
            }

            // Store results
            v128_store(c.as_mut_ptr().add((i + 0) * n + j + 0) as *mut v128, acc00);
            v128_store(c.as_mut_ptr().add((i + 0) * n + j + 4) as *mut v128, acc01);
            v128_store(c.as_mut_ptr().add((i + 1) * n + j + 0) as *mut v128, acc10);
            v128_store(c.as_mut_ptr().add((i + 1) * n + j + 4) as *mut v128, acc11);
            v128_store(c.as_mut_ptr().add((i + 2) * n + j + 0) as *mut v128, acc20);
            v128_store(c.as_mut_ptr().add((i + 2) * n + j + 4) as *mut v128, acc21);
            v128_store(c.as_mut_ptr().add((i + 3) * n + j + 0) as *mut v128, acc30);
            v128_store(c.as_mut_ptr().add((i + 3) * n + j + 4) as *mut v128, acc31);
            v128_store(c.as_mut_ptr().add((i + 4) * n + j + 0) as *mut v128, acc40);
            v128_store(c.as_mut_ptr().add((i + 4) * n + j + 4) as *mut v128, acc41);
            v128_store(c.as_mut_ptr().add((i + 5) * n + j + 0) as *mut v128, acc50);
            v128_store(c.as_mut_ptr().add((i + 5) * n + j + 4) as *mut v128, acc51);
        }
    }

    // NOTE: This kernel only handles the case where M % 6 == 0 and N % 8 == 0.
    // For arbitrary matrix dimensions, use matmul_simd_f32_xnnpack_style_full instead.
    // Remaining rows/columns should be computed by caller using original B matrix.
}

/// XNNPACK-style 6x8 kernel that handles arbitrary N (not just multiples of 8)
/// Takes both original B (for remaining columns) and packed_b (for SIMD panels)
#[cfg(target_arch = "wasm32")]
pub unsafe fn matmul_simd_f32_xnnpack_style_full(
    a: &[f32],
    b: &[f32],       // Original B for remaining columns
    packed_b: &[f32], // Pre-packed B for SIMD panels
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize
) {
    const MR: usize = 6;
    const NR: usize = 8;

    let m_main = m / MR * MR;
    let n_main = n / NR * NR;
    let n_panels = n / NR;
    let k_main = k / 4 * 4;

    // Main SIMD loop for full 6x8 tiles
    for i in (0..m_main).step_by(MR) {
        let a0_ptr = a.as_ptr().add(i * k);
        let a1_ptr = a.as_ptr().add((i + 1) * k);
        let a2_ptr = a.as_ptr().add((i + 2) * k);
        let a3_ptr = a.as_ptr().add((i + 3) * k);
        let a4_ptr = a.as_ptr().add((i + 4) * k);
        let a5_ptr = a.as_ptr().add((i + 5) * k);

        for panel in 0..n_panels {
            let j = panel * NR;
            let mut w_ptr = packed_b.as_ptr().add(panel * k * NR);

            let mut acc00 = f32x4_splat(0.0);
            let mut acc01 = f32x4_splat(0.0);
            let mut acc10 = f32x4_splat(0.0);
            let mut acc11 = f32x4_splat(0.0);
            let mut acc20 = f32x4_splat(0.0);
            let mut acc21 = f32x4_splat(0.0);
            let mut acc30 = f32x4_splat(0.0);
            let mut acc31 = f32x4_splat(0.0);
            let mut acc40 = f32x4_splat(0.0);
            let mut acc41 = f32x4_splat(0.0);
            let mut acc50 = f32x4_splat(0.0);
            let mut acc51 = f32x4_splat(0.0);

            let mut kk = 0;
            while kk < k_main {
                let va0 = v128_load(a0_ptr.add(kk) as *const v128);
                let va1 = v128_load(a1_ptr.add(kk) as *const v128);
                let va2 = v128_load(a2_ptr.add(kk) as *const v128);
                let va3 = v128_load(a3_ptr.add(kk) as *const v128);
                let va4 = v128_load(a4_ptr.add(kk) as *const v128);
                let va5 = v128_load(a5_ptr.add(kk) as *const v128);

                // 4 k iterations
                for lane in 0..4 {
                    let va0c = match lane {
                        0 => i32x4_shuffle::<0, 0, 0, 0>(va0, va0),
                        1 => i32x4_shuffle::<1, 1, 1, 1>(va0, va0),
                        2 => i32x4_shuffle::<2, 2, 2, 2>(va0, va0),
                        _ => i32x4_shuffle::<3, 3, 3, 3>(va0, va0),
                    };
                    let va1c = match lane {
                        0 => i32x4_shuffle::<0, 0, 0, 0>(va1, va1),
                        1 => i32x4_shuffle::<1, 1, 1, 1>(va1, va1),
                        2 => i32x4_shuffle::<2, 2, 2, 2>(va1, va1),
                        _ => i32x4_shuffle::<3, 3, 3, 3>(va1, va1),
                    };
                    let va2c = match lane {
                        0 => i32x4_shuffle::<0, 0, 0, 0>(va2, va2),
                        1 => i32x4_shuffle::<1, 1, 1, 1>(va2, va2),
                        2 => i32x4_shuffle::<2, 2, 2, 2>(va2, va2),
                        _ => i32x4_shuffle::<3, 3, 3, 3>(va2, va2),
                    };
                    let va3c = match lane {
                        0 => i32x4_shuffle::<0, 0, 0, 0>(va3, va3),
                        1 => i32x4_shuffle::<1, 1, 1, 1>(va3, va3),
                        2 => i32x4_shuffle::<2, 2, 2, 2>(va3, va3),
                        _ => i32x4_shuffle::<3, 3, 3, 3>(va3, va3),
                    };
                    let va4c = match lane {
                        0 => i32x4_shuffle::<0, 0, 0, 0>(va4, va4),
                        1 => i32x4_shuffle::<1, 1, 1, 1>(va4, va4),
                        2 => i32x4_shuffle::<2, 2, 2, 2>(va4, va4),
                        _ => i32x4_shuffle::<3, 3, 3, 3>(va4, va4),
                    };
                    let va5c = match lane {
                        0 => i32x4_shuffle::<0, 0, 0, 0>(va5, va5),
                        1 => i32x4_shuffle::<1, 1, 1, 1>(va5, va5),
                        2 => i32x4_shuffle::<2, 2, 2, 2>(va5, va5),
                        _ => i32x4_shuffle::<3, 3, 3, 3>(va5, va5),
                    };

                    let vb0 = v128_load(w_ptr as *const v128);
                    let vb1 = v128_load(w_ptr.add(4) as *const v128);

                    acc00 = f32x4_relaxed_madd(va0c, vb0, acc00);
                    acc01 = f32x4_relaxed_madd(va0c, vb1, acc01);
                    acc10 = f32x4_relaxed_madd(va1c, vb0, acc10);
                    acc11 = f32x4_relaxed_madd(va1c, vb1, acc11);
                    acc20 = f32x4_relaxed_madd(va2c, vb0, acc20);
                    acc21 = f32x4_relaxed_madd(va2c, vb1, acc21);
                    acc30 = f32x4_relaxed_madd(va3c, vb0, acc30);
                    acc31 = f32x4_relaxed_madd(va3c, vb1, acc31);
                    acc40 = f32x4_relaxed_madd(va4c, vb0, acc40);
                    acc41 = f32x4_relaxed_madd(va4c, vb1, acc41);
                    acc50 = f32x4_relaxed_madd(va5c, vb0, acc50);
                    acc51 = f32x4_relaxed_madd(va5c, vb1, acc51);

                    w_ptr = w_ptr.add(8);
                }
                kk += 4;
            }

            // Handle remaining k values
            while kk < k {
                let a0 = f32x4_splat(*a.get_unchecked((i + 0) * k + kk));
                let a1 = f32x4_splat(*a.get_unchecked((i + 1) * k + kk));
                let a2 = f32x4_splat(*a.get_unchecked((i + 2) * k + kk));
                let a3 = f32x4_splat(*a.get_unchecked((i + 3) * k + kk));
                let a4 = f32x4_splat(*a.get_unchecked((i + 4) * k + kk));
                let a5 = f32x4_splat(*a.get_unchecked((i + 5) * k + kk));
                let b0 = v128_load(w_ptr as *const v128);
                let b1 = v128_load(w_ptr.add(4) as *const v128);
                acc00 = f32x4_relaxed_madd(a0, b0, acc00);
                acc01 = f32x4_relaxed_madd(a0, b1, acc01);
                acc10 = f32x4_relaxed_madd(a1, b0, acc10);
                acc11 = f32x4_relaxed_madd(a1, b1, acc11);
                acc20 = f32x4_relaxed_madd(a2, b0, acc20);
                acc21 = f32x4_relaxed_madd(a2, b1, acc21);
                acc30 = f32x4_relaxed_madd(a3, b0, acc30);
                acc31 = f32x4_relaxed_madd(a3, b1, acc31);
                acc40 = f32x4_relaxed_madd(a4, b0, acc40);
                acc41 = f32x4_relaxed_madd(a4, b1, acc41);
                acc50 = f32x4_relaxed_madd(a5, b0, acc50);
                acc51 = f32x4_relaxed_madd(a5, b1, acc51);
                w_ptr = w_ptr.add(8);
                kk += 1;
            }

            // Store results
            v128_store(c.as_mut_ptr().add((i + 0) * n + j + 0) as *mut v128, acc00);
            v128_store(c.as_mut_ptr().add((i + 0) * n + j + 4) as *mut v128, acc01);
            v128_store(c.as_mut_ptr().add((i + 1) * n + j + 0) as *mut v128, acc10);
            v128_store(c.as_mut_ptr().add((i + 1) * n + j + 4) as *mut v128, acc11);
            v128_store(c.as_mut_ptr().add((i + 2) * n + j + 0) as *mut v128, acc20);
            v128_store(c.as_mut_ptr().add((i + 2) * n + j + 4) as *mut v128, acc21);
            v128_store(c.as_mut_ptr().add((i + 3) * n + j + 0) as *mut v128, acc30);
            v128_store(c.as_mut_ptr().add((i + 3) * n + j + 4) as *mut v128, acc31);
            v128_store(c.as_mut_ptr().add((i + 4) * n + j + 0) as *mut v128, acc40);
            v128_store(c.as_mut_ptr().add((i + 4) * n + j + 4) as *mut v128, acc41);
            v128_store(c.as_mut_ptr().add((i + 5) * n + j + 0) as *mut v128, acc50);
            v128_store(c.as_mut_ptr().add((i + 5) * n + j + 4) as *mut v128, acc51);
        }

        // Handle remaining columns (n_main..n) using original B
        for j in n_main..n {
            for di in 0..MR {
                let ii = i + di;
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += a[ii * k + kk] * b[kk * n + j];
                }
                c[ii * n + j] = sum;
            }
        }
    }

    // Handle remaining rows (m_main..m) using original B
    for i in m_main..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// Pack B in XNNPACK format: for each panel of 8 cols, store K values interleaved
/// Layout: panel0[k0:col0-7, k1:col0-7, ...], panel1[...], ...
pub fn pack_b_xnnpack(b: &[f32], packed: &mut [f32], k: usize, n: usize) {
    const NR: usize = 8;
    let n_panels = n / NR;

    for panel in 0..n_panels {
        let j = panel * NR;
        let panel_offset = panel * k * NR;

        for kk in 0..k {
            let b_row = kk * n + j;
            let pack_offset = panel_offset + kk * NR;
            packed[pack_offset..pack_offset + NR].copy_from_slice(&b[b_row..b_row + NR]);
        }
    }
}

// ============ Cache Blocking Constants ============
// These are tuned for typical L1/L2 cache sizes
// L1 data cache is typically 32KB, L2 is typically 256KB
// We want the working set (A panel + B panel + C panel) to fit in L2

/// Block size for K dimension (should fit in L1 with micro-panel)
const KC: usize = 256;

/// Block size for M dimension (should fit in L2 with packed B)
const MC: usize = 128;

/// Block size for N dimension
const NC: usize = 256;

/// Cache-blocked 6x8 GEMM with mul+add (not FMA)
///
/// Uses GOTO-style blocking:
/// - Outer loop tiles by NC (N dimension)
/// - Middle loop tiles by KC (K dimension)
/// - Inner loop tiles by MC (M dimension)
///
/// This ensures that:
/// - A micro-panel (MC x KC) fits in L2 cache
/// - B micro-panel (KC x NC) is reused across MC rows
/// - Better cache efficiency for large matrices
#[cfg(target_arch = "wasm32")]
pub unsafe fn matmul_simd_f32_6x8_blocked(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize
) {
    const MR: usize = 6;  // micro-kernel rows
    const NR: usize = 8;  // micro-kernel cols

    // Initialize C to zero
    c.iter_mut().for_each(|x| *x = 0.0);

    // Outer loop over N in blocks of NC
    for jc in (0..n).step_by(NC) {
        let nc = (n - jc).min(NC);
        let nc_main = nc / NR * NR;

        // Middle loop over K in blocks of KC
        for pc in (0..k).step_by(KC) {
            let kc = (k - pc).min(KC);

            // Inner loop over M in blocks of MC
            for ic in (0..m).step_by(MC) {
                let mc = (m - ic).min(MC);
                let mc_main = mc / MR * MR;

                // Micro-kernel: process MR x NR tiles
                for i in (0..mc_main).step_by(MR) {
                    let ii = ic + i;

                    for j in (0..nc_main).step_by(NR) {
                        let jj = jc + j;

                        // 6 rows × 8 cols = 12 accumulators
                        let mut acc00 = f32x4_splat(0.0);
                        let mut acc01 = f32x4_splat(0.0);
                        let mut acc10 = f32x4_splat(0.0);
                        let mut acc11 = f32x4_splat(0.0);
                        let mut acc20 = f32x4_splat(0.0);
                        let mut acc21 = f32x4_splat(0.0);
                        let mut acc30 = f32x4_splat(0.0);
                        let mut acc31 = f32x4_splat(0.0);
                        let mut acc40 = f32x4_splat(0.0);
                        let mut acc41 = f32x4_splat(0.0);
                        let mut acc50 = f32x4_splat(0.0);
                        let mut acc51 = f32x4_splat(0.0);

                        // K loop within the block
                        for kk in 0..kc {
                            let pk = pc + kk;

                            let a0 = f32x4_splat(*a.get_unchecked((ii + 0) * k + pk));
                            let a1 = f32x4_splat(*a.get_unchecked((ii + 1) * k + pk));
                            let a2 = f32x4_splat(*a.get_unchecked((ii + 2) * k + pk));
                            let a3 = f32x4_splat(*a.get_unchecked((ii + 3) * k + pk));
                            let a4 = f32x4_splat(*a.get_unchecked((ii + 4) * k + pk));
                            let a5 = f32x4_splat(*a.get_unchecked((ii + 5) * k + pk));

                            let b_base = pk * n + jj;
                            let b0 = v128_load(b.as_ptr().add(b_base + 0) as *const v128);
                            let b1 = v128_load(b.as_ptr().add(b_base + 4) as *const v128);

                            // Use mul+add (matches XNNPACK)
                            acc00 = f32x4_add(f32x4_mul(a0, b0), acc00);
                            acc01 = f32x4_add(f32x4_mul(a0, b1), acc01);
                            acc10 = f32x4_add(f32x4_mul(a1, b0), acc10);
                            acc11 = f32x4_add(f32x4_mul(a1, b1), acc11);
                            acc20 = f32x4_add(f32x4_mul(a2, b0), acc20);
                            acc21 = f32x4_add(f32x4_mul(a2, b1), acc21);
                            acc30 = f32x4_add(f32x4_mul(a3, b0), acc30);
                            acc31 = f32x4_add(f32x4_mul(a3, b1), acc31);
                            acc40 = f32x4_add(f32x4_mul(a4, b0), acc40);
                            acc41 = f32x4_add(f32x4_mul(a4, b1), acc41);
                            acc50 = f32x4_add(f32x4_mul(a5, b0), acc50);
                            acc51 = f32x4_add(f32x4_mul(a5, b1), acc51);
                        }

                        // Accumulate into C (not overwrite - we're tiling K)
                        let c00 = v128_load(c.as_ptr().add((ii + 0) * n + jj + 0) as *const v128);
                        let c01 = v128_load(c.as_ptr().add((ii + 0) * n + jj + 4) as *const v128);
                        let c10 = v128_load(c.as_ptr().add((ii + 1) * n + jj + 0) as *const v128);
                        let c11 = v128_load(c.as_ptr().add((ii + 1) * n + jj + 4) as *const v128);
                        let c20 = v128_load(c.as_ptr().add((ii + 2) * n + jj + 0) as *const v128);
                        let c21 = v128_load(c.as_ptr().add((ii + 2) * n + jj + 4) as *const v128);
                        let c30 = v128_load(c.as_ptr().add((ii + 3) * n + jj + 0) as *const v128);
                        let c31 = v128_load(c.as_ptr().add((ii + 3) * n + jj + 4) as *const v128);
                        let c40 = v128_load(c.as_ptr().add((ii + 4) * n + jj + 0) as *const v128);
                        let c41 = v128_load(c.as_ptr().add((ii + 4) * n + jj + 4) as *const v128);
                        let c50 = v128_load(c.as_ptr().add((ii + 5) * n + jj + 0) as *const v128);
                        let c51 = v128_load(c.as_ptr().add((ii + 5) * n + jj + 4) as *const v128);

                        v128_store(c.as_mut_ptr().add((ii + 0) * n + jj + 0) as *mut v128, f32x4_add(c00, acc00));
                        v128_store(c.as_mut_ptr().add((ii + 0) * n + jj + 4) as *mut v128, f32x4_add(c01, acc01));
                        v128_store(c.as_mut_ptr().add((ii + 1) * n + jj + 0) as *mut v128, f32x4_add(c10, acc10));
                        v128_store(c.as_mut_ptr().add((ii + 1) * n + jj + 4) as *mut v128, f32x4_add(c11, acc11));
                        v128_store(c.as_mut_ptr().add((ii + 2) * n + jj + 0) as *mut v128, f32x4_add(c20, acc20));
                        v128_store(c.as_mut_ptr().add((ii + 2) * n + jj + 4) as *mut v128, f32x4_add(c21, acc21));
                        v128_store(c.as_mut_ptr().add((ii + 3) * n + jj + 0) as *mut v128, f32x4_add(c30, acc30));
                        v128_store(c.as_mut_ptr().add((ii + 3) * n + jj + 4) as *mut v128, f32x4_add(c31, acc31));
                        v128_store(c.as_mut_ptr().add((ii + 4) * n + jj + 0) as *mut v128, f32x4_add(c40, acc40));
                        v128_store(c.as_mut_ptr().add((ii + 4) * n + jj + 4) as *mut v128, f32x4_add(c41, acc41));
                        v128_store(c.as_mut_ptr().add((ii + 5) * n + jj + 0) as *mut v128, f32x4_add(c50, acc50));
                        v128_store(c.as_mut_ptr().add((ii + 5) * n + jj + 4) as *mut v128, f32x4_add(c51, acc51));
                    }

                    // Handle remaining columns in nc block
                    for j in nc_main..nc {
                        let jj = jc + j;
                        for di in 0..MR {
                            let iii = ii + di;
                            let mut sum = 0.0f32;
                            for kk in 0..kc {
                                sum += a[iii * k + pc + kk] * b[(pc + kk) * n + jj];
                            }
                            c[iii * n + jj] += sum;
                        }
                    }
                }

                // Handle remaining rows in mc block
                for i in mc_main..mc {
                    let ii = ic + i;
                    for j in 0..nc {
                        let jj = jc + j;
                        let mut sum = 0.0f32;
                        for kk in 0..kc {
                            sum += a[ii * k + pc + kk] * b[(pc + kk) * n + jj];
                        }
                        c[ii * n + jj] += sum;
                    }
                }
            }
        }
    }
}

/// Cache-blocked XNNPACK-style GEMM with pre-packed B
///
/// This combines cache blocking with B-matrix packing for optimal performance.
/// The B matrix is packed into KC x NC panels on-the-fly as we tile through K and N.
#[cfg(target_arch = "wasm32")]
pub unsafe fn matmul_simd_f32_xnnpack_blocked(
    a: &[f32],
    b: &[f32],
    packed_b: &[f32],  // Pre-packed B in XNNPACK format
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize
) {
    const MR: usize = 6;
    const NR: usize = 8;

    let m_main = m / MR * MR;
    let n_main = n / NR * NR;
    let n_panels = n / NR;

    // Initialize C to zero
    c.iter_mut().for_each(|x| *x = 0.0);

    // Block over K for better cache reuse
    for kc_start in (0..k).step_by(KC) {
        let kc_end = (kc_start + KC).min(k);
        let kc = kc_end - kc_start;
        let kc_main = kc / 4 * 4;

        // Main SIMD loop for full 6x8 tiles
        for i in (0..m_main).step_by(MR) {
            let a0_ptr = a.as_ptr().add(i * k + kc_start);
            let a1_ptr = a.as_ptr().add((i + 1) * k + kc_start);
            let a2_ptr = a.as_ptr().add((i + 2) * k + kc_start);
            let a3_ptr = a.as_ptr().add((i + 3) * k + kc_start);
            let a4_ptr = a.as_ptr().add((i + 4) * k + kc_start);
            let a5_ptr = a.as_ptr().add((i + 5) * k + kc_start);

            for panel in 0..n_panels {
                let j = panel * NR;
                // Offset into packed_b for this K block
                let mut w_ptr = packed_b.as_ptr().add(panel * k * NR + kc_start * NR);

                let mut acc00 = f32x4_splat(0.0);
                let mut acc01 = f32x4_splat(0.0);
                let mut acc10 = f32x4_splat(0.0);
                let mut acc11 = f32x4_splat(0.0);
                let mut acc20 = f32x4_splat(0.0);
                let mut acc21 = f32x4_splat(0.0);
                let mut acc30 = f32x4_splat(0.0);
                let mut acc31 = f32x4_splat(0.0);
                let mut acc40 = f32x4_splat(0.0);
                let mut acc41 = f32x4_splat(0.0);
                let mut acc50 = f32x4_splat(0.0);
                let mut acc51 = f32x4_splat(0.0);

                let mut kk = 0;
                while kk < kc_main {
                    // Load 4 A values at once for each row
                    let va0 = v128_load(a0_ptr.add(kk) as *const v128);
                    let va1 = v128_load(a1_ptr.add(kk) as *const v128);
                    let va2 = v128_load(a2_ptr.add(kk) as *const v128);
                    let va3 = v128_load(a3_ptr.add(kk) as *const v128);
                    let va4 = v128_load(a4_ptr.add(kk) as *const v128);
                    let va5 = v128_load(a5_ptr.add(kk) as *const v128);

                    // K iteration 0: broadcast lane 0
                    let va0c0 = i32x4_shuffle::<0, 0, 0, 0>(va0, va0);
                    let va1c0 = i32x4_shuffle::<0, 0, 0, 0>(va1, va1);
                    let va2c0 = i32x4_shuffle::<0, 0, 0, 0>(va2, va2);
                    let va3c0 = i32x4_shuffle::<0, 0, 0, 0>(va3, va3);
                    let va4c0 = i32x4_shuffle::<0, 0, 0, 0>(va4, va4);
                    let va5c0 = i32x4_shuffle::<0, 0, 0, 0>(va5, va5);

                    let vb0123c0 = v128_load(w_ptr as *const v128);
                    let vb4567c0 = v128_load(w_ptr.add(4) as *const v128);

                    acc00 = f32x4_add(f32x4_mul(va0c0, vb0123c0), acc00);
                    acc01 = f32x4_add(f32x4_mul(va0c0, vb4567c0), acc01);
                    acc10 = f32x4_add(f32x4_mul(va1c0, vb0123c0), acc10);
                    acc11 = f32x4_add(f32x4_mul(va1c0, vb4567c0), acc11);
                    acc20 = f32x4_add(f32x4_mul(va2c0, vb0123c0), acc20);
                    acc21 = f32x4_add(f32x4_mul(va2c0, vb4567c0), acc21);
                    acc30 = f32x4_add(f32x4_mul(va3c0, vb0123c0), acc30);
                    acc31 = f32x4_add(f32x4_mul(va3c0, vb4567c0), acc31);
                    acc40 = f32x4_add(f32x4_mul(va4c0, vb0123c0), acc40);
                    acc41 = f32x4_add(f32x4_mul(va4c0, vb4567c0), acc41);
                    acc50 = f32x4_add(f32x4_mul(va5c0, vb0123c0), acc50);
                    acc51 = f32x4_add(f32x4_mul(va5c0, vb4567c0), acc51);

                    // K iteration 1
                    let va0c1 = i32x4_shuffle::<1, 1, 1, 1>(va0, va0);
                    let va1c1 = i32x4_shuffle::<1, 1, 1, 1>(va1, va1);
                    let va2c1 = i32x4_shuffle::<1, 1, 1, 1>(va2, va2);
                    let va3c1 = i32x4_shuffle::<1, 1, 1, 1>(va3, va3);
                    let va4c1 = i32x4_shuffle::<1, 1, 1, 1>(va4, va4);
                    let va5c1 = i32x4_shuffle::<1, 1, 1, 1>(va5, va5);

                    let vb0123c1 = v128_load(w_ptr.add(8) as *const v128);
                    let vb4567c1 = v128_load(w_ptr.add(12) as *const v128);

                    acc00 = f32x4_add(f32x4_mul(va0c1, vb0123c1), acc00);
                    acc01 = f32x4_add(f32x4_mul(va0c1, vb4567c1), acc01);
                    acc10 = f32x4_add(f32x4_mul(va1c1, vb0123c1), acc10);
                    acc11 = f32x4_add(f32x4_mul(va1c1, vb4567c1), acc11);
                    acc20 = f32x4_add(f32x4_mul(va2c1, vb0123c1), acc20);
                    acc21 = f32x4_add(f32x4_mul(va2c1, vb4567c1), acc21);
                    acc30 = f32x4_add(f32x4_mul(va3c1, vb0123c1), acc30);
                    acc31 = f32x4_add(f32x4_mul(va3c1, vb4567c1), acc31);
                    acc40 = f32x4_add(f32x4_mul(va4c1, vb0123c1), acc40);
                    acc41 = f32x4_add(f32x4_mul(va4c1, vb4567c1), acc41);
                    acc50 = f32x4_add(f32x4_mul(va5c1, vb0123c1), acc50);
                    acc51 = f32x4_add(f32x4_mul(va5c1, vb4567c1), acc51);

                    // K iteration 2
                    let va0c2 = i32x4_shuffle::<2, 2, 2, 2>(va0, va0);
                    let va1c2 = i32x4_shuffle::<2, 2, 2, 2>(va1, va1);
                    let va2c2 = i32x4_shuffle::<2, 2, 2, 2>(va2, va2);
                    let va3c2 = i32x4_shuffle::<2, 2, 2, 2>(va3, va3);
                    let va4c2 = i32x4_shuffle::<2, 2, 2, 2>(va4, va4);
                    let va5c2 = i32x4_shuffle::<2, 2, 2, 2>(va5, va5);

                    let vb0123c2 = v128_load(w_ptr.add(16) as *const v128);
                    let vb4567c2 = v128_load(w_ptr.add(20) as *const v128);

                    acc00 = f32x4_add(f32x4_mul(va0c2, vb0123c2), acc00);
                    acc01 = f32x4_add(f32x4_mul(va0c2, vb4567c2), acc01);
                    acc10 = f32x4_add(f32x4_mul(va1c2, vb0123c2), acc10);
                    acc11 = f32x4_add(f32x4_mul(va1c2, vb4567c2), acc11);
                    acc20 = f32x4_add(f32x4_mul(va2c2, vb0123c2), acc20);
                    acc21 = f32x4_add(f32x4_mul(va2c2, vb4567c2), acc21);
                    acc30 = f32x4_add(f32x4_mul(va3c2, vb0123c2), acc30);
                    acc31 = f32x4_add(f32x4_mul(va3c2, vb4567c2), acc31);
                    acc40 = f32x4_add(f32x4_mul(va4c2, vb0123c2), acc40);
                    acc41 = f32x4_add(f32x4_mul(va4c2, vb4567c2), acc41);
                    acc50 = f32x4_add(f32x4_mul(va5c2, vb0123c2), acc50);
                    acc51 = f32x4_add(f32x4_mul(va5c2, vb4567c2), acc51);

                    // K iteration 3
                    let va0c3 = i32x4_shuffle::<3, 3, 3, 3>(va0, va0);
                    let va1c3 = i32x4_shuffle::<3, 3, 3, 3>(va1, va1);
                    let va2c3 = i32x4_shuffle::<3, 3, 3, 3>(va2, va2);
                    let va3c3 = i32x4_shuffle::<3, 3, 3, 3>(va3, va3);
                    let va4c3 = i32x4_shuffle::<3, 3, 3, 3>(va4, va4);
                    let va5c3 = i32x4_shuffle::<3, 3, 3, 3>(va5, va5);

                    let vb0123c3 = v128_load(w_ptr.add(24) as *const v128);
                    let vb4567c3 = v128_load(w_ptr.add(28) as *const v128);

                    acc00 = f32x4_add(f32x4_mul(va0c3, vb0123c3), acc00);
                    acc01 = f32x4_add(f32x4_mul(va0c3, vb4567c3), acc01);
                    acc10 = f32x4_add(f32x4_mul(va1c3, vb0123c3), acc10);
                    acc11 = f32x4_add(f32x4_mul(va1c3, vb4567c3), acc11);
                    acc20 = f32x4_add(f32x4_mul(va2c3, vb0123c3), acc20);
                    acc21 = f32x4_add(f32x4_mul(va2c3, vb4567c3), acc21);
                    acc30 = f32x4_add(f32x4_mul(va3c3, vb0123c3), acc30);
                    acc31 = f32x4_add(f32x4_mul(va3c3, vb4567c3), acc31);
                    acc40 = f32x4_add(f32x4_mul(va4c3, vb0123c3), acc40);
                    acc41 = f32x4_add(f32x4_mul(va4c3, vb4567c3), acc41);
                    acc50 = f32x4_add(f32x4_mul(va5c3, vb0123c3), acc50);
                    acc51 = f32x4_add(f32x4_mul(va5c3, vb4567c3), acc51);

                    w_ptr = w_ptr.add(32);
                    kk += 4;
                }

                // Handle remaining K iterations
                while kk < kc {
                    let a0 = f32x4_splat(*a.get_unchecked((i + 0) * k + kc_start + kk));
                    let a1 = f32x4_splat(*a.get_unchecked((i + 1) * k + kc_start + kk));
                    let a2 = f32x4_splat(*a.get_unchecked((i + 2) * k + kc_start + kk));
                    let a3 = f32x4_splat(*a.get_unchecked((i + 3) * k + kc_start + kk));
                    let a4 = f32x4_splat(*a.get_unchecked((i + 4) * k + kc_start + kk));
                    let a5 = f32x4_splat(*a.get_unchecked((i + 5) * k + kc_start + kk));
                    let b0 = v128_load(w_ptr as *const v128);
                    let b1 = v128_load(w_ptr.add(4) as *const v128);
                    acc00 = f32x4_add(f32x4_mul(a0, b0), acc00);
                    acc01 = f32x4_add(f32x4_mul(a0, b1), acc01);
                    acc10 = f32x4_add(f32x4_mul(a1, b0), acc10);
                    acc11 = f32x4_add(f32x4_mul(a1, b1), acc11);
                    acc20 = f32x4_add(f32x4_mul(a2, b0), acc20);
                    acc21 = f32x4_add(f32x4_mul(a2, b1), acc21);
                    acc30 = f32x4_add(f32x4_mul(a3, b0), acc30);
                    acc31 = f32x4_add(f32x4_mul(a3, b1), acc31);
                    acc40 = f32x4_add(f32x4_mul(a4, b0), acc40);
                    acc41 = f32x4_add(f32x4_mul(a4, b1), acc41);
                    acc50 = f32x4_add(f32x4_mul(a5, b0), acc50);
                    acc51 = f32x4_add(f32x4_mul(a5, b1), acc51);
                    w_ptr = w_ptr.add(8);
                    kk += 1;
                }

                // Accumulate into C
                let c00 = v128_load(c.as_ptr().add((i + 0) * n + j + 0) as *const v128);
                let c01 = v128_load(c.as_ptr().add((i + 0) * n + j + 4) as *const v128);
                let c10 = v128_load(c.as_ptr().add((i + 1) * n + j + 0) as *const v128);
                let c11 = v128_load(c.as_ptr().add((i + 1) * n + j + 4) as *const v128);
                let c20 = v128_load(c.as_ptr().add((i + 2) * n + j + 0) as *const v128);
                let c21 = v128_load(c.as_ptr().add((i + 2) * n + j + 4) as *const v128);
                let c30 = v128_load(c.as_ptr().add((i + 3) * n + j + 0) as *const v128);
                let c31 = v128_load(c.as_ptr().add((i + 3) * n + j + 4) as *const v128);
                let c40 = v128_load(c.as_ptr().add((i + 4) * n + j + 0) as *const v128);
                let c41 = v128_load(c.as_ptr().add((i + 4) * n + j + 4) as *const v128);
                let c50 = v128_load(c.as_ptr().add((i + 5) * n + j + 0) as *const v128);
                let c51 = v128_load(c.as_ptr().add((i + 5) * n + j + 4) as *const v128);

                v128_store(c.as_mut_ptr().add((i + 0) * n + j + 0) as *mut v128, f32x4_add(c00, acc00));
                v128_store(c.as_mut_ptr().add((i + 0) * n + j + 4) as *mut v128, f32x4_add(c01, acc01));
                v128_store(c.as_mut_ptr().add((i + 1) * n + j + 0) as *mut v128, f32x4_add(c10, acc10));
                v128_store(c.as_mut_ptr().add((i + 1) * n + j + 4) as *mut v128, f32x4_add(c11, acc11));
                v128_store(c.as_mut_ptr().add((i + 2) * n + j + 0) as *mut v128, f32x4_add(c20, acc20));
                v128_store(c.as_mut_ptr().add((i + 2) * n + j + 4) as *mut v128, f32x4_add(c21, acc21));
                v128_store(c.as_mut_ptr().add((i + 3) * n + j + 0) as *mut v128, f32x4_add(c30, acc30));
                v128_store(c.as_mut_ptr().add((i + 3) * n + j + 4) as *mut v128, f32x4_add(c31, acc31));
                v128_store(c.as_mut_ptr().add((i + 4) * n + j + 0) as *mut v128, f32x4_add(c40, acc40));
                v128_store(c.as_mut_ptr().add((i + 4) * n + j + 4) as *mut v128, f32x4_add(c41, acc41));
                v128_store(c.as_mut_ptr().add((i + 5) * n + j + 0) as *mut v128, f32x4_add(c50, acc50));
                v128_store(c.as_mut_ptr().add((i + 5) * n + j + 4) as *mut v128, f32x4_add(c51, acc51));
            }

            // Handle remaining columns using original B
            for j in n_main..n {
                for di in 0..MR {
                    let ii = i + di;
                    let mut sum = 0.0f32;
                    for kk in kc_start..kc_end {
                        sum += a[ii * k + kk] * b[kk * n + j];
                    }
                    c[ii * n + j] += sum;
                }
            }
        }

        // Handle remaining rows using original B
        for i in m_main..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for kk in kc_start..kc_end {
                    sum += a[i * k + kk] * b[kk * n + j];
                }
                c[i * n + j] += sum;
            }
        }
    }
}

/// Pack B matrix into column panels of NR=8 columns
/// Layout: for each panel j, store all k rows contiguously
/// packed_b[panel_idx * k * NR + kk * NR + col_in_panel] = b[kk * n + j + col_in_panel]
#[inline]
fn pack_b_f32(b: &[f32], packed: &mut [f32], k: usize, n: usize) {
    const NR: usize = 8;
    let n_panels = n / NR;

    for panel in 0..n_panels {
        let j = panel * NR;
        let panel_offset = panel * k * NR;

        for kk in 0..k {
            let b_row = kk * n + j;
            let pack_offset = panel_offset + kk * NR;

            // Copy 8 consecutive elements
            packed[pack_offset..pack_offset + NR].copy_from_slice(&b[b_row..b_row + NR]);
        }
    }
}

/// XNNPACK-style SIMD matrix multiplication for f32 with matrix packing
/// Uses 4x8 micro-kernel (4 rows of A, 8 cols of B = 4 elements per v128 * 2 vectors)
///
/// Layout: A is MxK, B is KxN, C is MxN (all row-major)
///
/// # Safety
/// - a must have at least m*k elements
/// - b must have at least k*n elements
/// - c must have at least m*n elements
#[cfg(target_arch = "wasm32")]
pub unsafe fn matmul_simd_f32_packed(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    const MR: usize = 4;  // rows per micro-kernel (matches XNNPACK)
    const NR: usize = 8;  // cols per micro-kernel (2 v128s of 4 f32s each)

    let m_main = m / MR * MR;
    let n_main = n / NR * NR;
    let n_panels = n / NR;

    // Pack B matrix for better cache locality
    let mut packed_b = vec![0.0f32; n_panels * k * NR];
    pack_b_f32(b, &mut packed_b, k, n);

    // Main loop: process MR rows × NR cols at a time
    for i in (0..m_main).step_by(MR) {
        for panel in 0..n_panels {
            let j = panel * NR;
            let panel_offset = panel * k * NR;

            // 4 rows × 8 cols = 8 v128 accumulators (2 per row)
            let mut acc00 = f32x4_splat(0.0);
            let mut acc01 = f32x4_splat(0.0);
            let mut acc10 = f32x4_splat(0.0);
            let mut acc11 = f32x4_splat(0.0);
            let mut acc20 = f32x4_splat(0.0);
            let mut acc21 = f32x4_splat(0.0);
            let mut acc30 = f32x4_splat(0.0);
            let mut acc31 = f32x4_splat(0.0);

            // K loop - accumulate products using packed B
            for kk in 0..k {
                // Load 4 A values and splat each to a v128
                let a0 = f32x4_splat(a[(i + 0) * k + kk]);
                let a1 = f32x4_splat(a[(i + 1) * k + kk]);
                let a2 = f32x4_splat(a[(i + 2) * k + kk]);
                let a3 = f32x4_splat(a[(i + 3) * k + kk]);

                // Load 8 B values from packed buffer (contiguous!)
                let pack_offset = panel_offset + kk * NR;
                let b0 = v128_load(packed_b.as_ptr().add(pack_offset + 0) as *const v128);
                let b1 = v128_load(packed_b.as_ptr().add(pack_offset + 4) as *const v128);

                // Accumulate: C[i,j] += A[i,k] * B[k,j]
                acc00 = f32x4_add(acc00, f32x4_mul(a0, b0));
                acc01 = f32x4_add(acc01, f32x4_mul(a0, b1));
                acc10 = f32x4_add(acc10, f32x4_mul(a1, b0));
                acc11 = f32x4_add(acc11, f32x4_mul(a1, b1));
                acc20 = f32x4_add(acc20, f32x4_mul(a2, b0));
                acc21 = f32x4_add(acc21, f32x4_mul(a2, b1));
                acc30 = f32x4_add(acc30, f32x4_mul(a3, b0));
                acc31 = f32x4_add(acc31, f32x4_mul(a3, b1));
            }

            // Store results
            let c0_base = (i + 0) * n + j;
            let c1_base = (i + 1) * n + j;
            let c2_base = (i + 2) * n + j;
            let c3_base = (i + 3) * n + j;
            v128_store(c.as_mut_ptr().add(c0_base + 0) as *mut v128, acc00);
            v128_store(c.as_mut_ptr().add(c0_base + 4) as *mut v128, acc01);
            v128_store(c.as_mut_ptr().add(c1_base + 0) as *mut v128, acc10);
            v128_store(c.as_mut_ptr().add(c1_base + 4) as *mut v128, acc11);
            v128_store(c.as_mut_ptr().add(c2_base + 0) as *mut v128, acc20);
            v128_store(c.as_mut_ptr().add(c2_base + 4) as *mut v128, acc21);
            v128_store(c.as_mut_ptr().add(c3_base + 0) as *mut v128, acc30);
            v128_store(c.as_mut_ptr().add(c3_base + 4) as *mut v128, acc31);
        }

        // Handle remaining columns (n_main..n) with scalar
        for j in n_main..n {
            for di in 0..MR {
                let ii = i + di;
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += a[ii * k + kk] * b[kk * n + j];
                }
                c[ii * n + j] = sum;
            }
        }
    }

    // Handle remaining rows (m_main..m)
    for i in m_main..m {
        // Use packed B for columns that fit
        for panel in 0..n_panels {
            let j = panel * NR;
            let panel_offset = panel * k * NR;

            let mut acc0 = f32x4_splat(0.0);
            let mut acc1 = f32x4_splat(0.0);

            for kk in 0..k {
                let a_val = f32x4_splat(a[i * k + kk]);
                let pack_offset = panel_offset + kk * NR;
                let b0 = v128_load(packed_b.as_ptr().add(pack_offset + 0) as *const v128);
                let b1 = v128_load(packed_b.as_ptr().add(pack_offset + 4) as *const v128);
                acc0 = f32x4_add(acc0, f32x4_mul(a_val, b0));
                acc1 = f32x4_add(acc1, f32x4_mul(a_val, b1));
            }

            let c_base = i * n + j;
            v128_store(c.as_mut_ptr().add(c_base + 0) as *mut v128, acc0);
            v128_store(c.as_mut_ptr().add(c_base + 4) as *mut v128, acc1);
        }

        // Remaining columns
        for j in n_main..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// Original SIMD matrix multiplication for f32 (no packing)
/// Uses 4x8 micro-kernel with 4x unrolled K loop
///
/// Layout: A is MxK, B is KxN, C is MxN (all row-major)
///
/// # Safety
/// - a must have at least m*k elements
/// - b must have at least k*n elements
/// - c must have at least m*n elements
#[cfg(target_arch = "wasm32")]
pub unsafe fn matmul_simd_f32(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    const MR: usize = 4;  // rows per micro-kernel (matches XNNPACK)
    const NR: usize = 8;  // cols per micro-kernel (2 v128s of 4 f32s each)
    const KU: usize = 4;  // K unroll factor

    let m_main = m / MR * MR;
    let n_main = n / NR * NR;
    let k_main = k / KU * KU;

    // Main loop: process MR rows × NR cols at a time
    for i in (0..m_main).step_by(MR) {
        for j in (0..n_main).step_by(NR) {
            // 4 rows × 8 cols = 8 v128 accumulators (2 per row)
            let mut acc00 = f32x4_splat(0.0);
            let mut acc01 = f32x4_splat(0.0);
            let mut acc10 = f32x4_splat(0.0);
            let mut acc11 = f32x4_splat(0.0);
            let mut acc20 = f32x4_splat(0.0);
            let mut acc21 = f32x4_splat(0.0);
            let mut acc30 = f32x4_splat(0.0);
            let mut acc31 = f32x4_splat(0.0);

            // K loop - 4x unrolled for better instruction-level parallelism
            let mut kk = 0;
            while kk < k_main {
                // Unroll 0
                {
                    let a0 = f32x4_splat(*a.get_unchecked((i + 0) * k + kk));
                    let a1 = f32x4_splat(*a.get_unchecked((i + 1) * k + kk));
                    let a2 = f32x4_splat(*a.get_unchecked((i + 2) * k + kk));
                    let a3 = f32x4_splat(*a.get_unchecked((i + 3) * k + kk));
                    let b_base = kk * n + j;
                    let b0 = v128_load(b.as_ptr().add(b_base + 0) as *const v128);
                    let b1 = v128_load(b.as_ptr().add(b_base + 4) as *const v128);
                    acc00 = f32x4_add(acc00, f32x4_mul(a0, b0));
                    acc01 = f32x4_add(acc01, f32x4_mul(a0, b1));
                    acc10 = f32x4_add(acc10, f32x4_mul(a1, b0));
                    acc11 = f32x4_add(acc11, f32x4_mul(a1, b1));
                    acc20 = f32x4_add(acc20, f32x4_mul(a2, b0));
                    acc21 = f32x4_add(acc21, f32x4_mul(a2, b1));
                    acc30 = f32x4_add(acc30, f32x4_mul(a3, b0));
                    acc31 = f32x4_add(acc31, f32x4_mul(a3, b1));
                }
                // Unroll 1
                {
                    let a0 = f32x4_splat(*a.get_unchecked((i + 0) * k + kk + 1));
                    let a1 = f32x4_splat(*a.get_unchecked((i + 1) * k + kk + 1));
                    let a2 = f32x4_splat(*a.get_unchecked((i + 2) * k + kk + 1));
                    let a3 = f32x4_splat(*a.get_unchecked((i + 3) * k + kk + 1));
                    let b_base = (kk + 1) * n + j;
                    let b0 = v128_load(b.as_ptr().add(b_base + 0) as *const v128);
                    let b1 = v128_load(b.as_ptr().add(b_base + 4) as *const v128);
                    acc00 = f32x4_add(acc00, f32x4_mul(a0, b0));
                    acc01 = f32x4_add(acc01, f32x4_mul(a0, b1));
                    acc10 = f32x4_add(acc10, f32x4_mul(a1, b0));
                    acc11 = f32x4_add(acc11, f32x4_mul(a1, b1));
                    acc20 = f32x4_add(acc20, f32x4_mul(a2, b0));
                    acc21 = f32x4_add(acc21, f32x4_mul(a2, b1));
                    acc30 = f32x4_add(acc30, f32x4_mul(a3, b0));
                    acc31 = f32x4_add(acc31, f32x4_mul(a3, b1));
                }
                // Unroll 2
                {
                    let a0 = f32x4_splat(*a.get_unchecked((i + 0) * k + kk + 2));
                    let a1 = f32x4_splat(*a.get_unchecked((i + 1) * k + kk + 2));
                    let a2 = f32x4_splat(*a.get_unchecked((i + 2) * k + kk + 2));
                    let a3 = f32x4_splat(*a.get_unchecked((i + 3) * k + kk + 2));
                    let b_base = (kk + 2) * n + j;
                    let b0 = v128_load(b.as_ptr().add(b_base + 0) as *const v128);
                    let b1 = v128_load(b.as_ptr().add(b_base + 4) as *const v128);
                    acc00 = f32x4_add(acc00, f32x4_mul(a0, b0));
                    acc01 = f32x4_add(acc01, f32x4_mul(a0, b1));
                    acc10 = f32x4_add(acc10, f32x4_mul(a1, b0));
                    acc11 = f32x4_add(acc11, f32x4_mul(a1, b1));
                    acc20 = f32x4_add(acc20, f32x4_mul(a2, b0));
                    acc21 = f32x4_add(acc21, f32x4_mul(a2, b1));
                    acc30 = f32x4_add(acc30, f32x4_mul(a3, b0));
                    acc31 = f32x4_add(acc31, f32x4_mul(a3, b1));
                }
                // Unroll 3
                {
                    let a0 = f32x4_splat(*a.get_unchecked((i + 0) * k + kk + 3));
                    let a1 = f32x4_splat(*a.get_unchecked((i + 1) * k + kk + 3));
                    let a2 = f32x4_splat(*a.get_unchecked((i + 2) * k + kk + 3));
                    let a3 = f32x4_splat(*a.get_unchecked((i + 3) * k + kk + 3));
                    let b_base = (kk + 3) * n + j;
                    let b0 = v128_load(b.as_ptr().add(b_base + 0) as *const v128);
                    let b1 = v128_load(b.as_ptr().add(b_base + 4) as *const v128);
                    acc00 = f32x4_add(acc00, f32x4_mul(a0, b0));
                    acc01 = f32x4_add(acc01, f32x4_mul(a0, b1));
                    acc10 = f32x4_add(acc10, f32x4_mul(a1, b0));
                    acc11 = f32x4_add(acc11, f32x4_mul(a1, b1));
                    acc20 = f32x4_add(acc20, f32x4_mul(a2, b0));
                    acc21 = f32x4_add(acc21, f32x4_mul(a2, b1));
                    acc30 = f32x4_add(acc30, f32x4_mul(a3, b0));
                    acc31 = f32x4_add(acc31, f32x4_mul(a3, b1));
                }
                kk += KU;
            }

            // Handle remaining K iterations
            while kk < k {
                let a0 = f32x4_splat(*a.get_unchecked((i + 0) * k + kk));
                let a1 = f32x4_splat(*a.get_unchecked((i + 1) * k + kk));
                let a2 = f32x4_splat(*a.get_unchecked((i + 2) * k + kk));
                let a3 = f32x4_splat(*a.get_unchecked((i + 3) * k + kk));
                let b_base = kk * n + j;
                let b0 = v128_load(b.as_ptr().add(b_base + 0) as *const v128);
                let b1 = v128_load(b.as_ptr().add(b_base + 4) as *const v128);
                acc00 = f32x4_add(acc00, f32x4_mul(a0, b0));
                acc01 = f32x4_add(acc01, f32x4_mul(a0, b1));
                acc10 = f32x4_add(acc10, f32x4_mul(a1, b0));
                acc11 = f32x4_add(acc11, f32x4_mul(a1, b1));
                acc20 = f32x4_add(acc20, f32x4_mul(a2, b0));
                acc21 = f32x4_add(acc21, f32x4_mul(a2, b1));
                acc30 = f32x4_add(acc30, f32x4_mul(a3, b0));
                acc31 = f32x4_add(acc31, f32x4_mul(a3, b1));
                kk += 1;
            }

            // Store results
            let c0_base = (i + 0) * n + j;
            let c1_base = (i + 1) * n + j;
            let c2_base = (i + 2) * n + j;
            let c3_base = (i + 3) * n + j;
            v128_store(c.as_mut_ptr().add(c0_base + 0) as *mut v128, acc00);
            v128_store(c.as_mut_ptr().add(c0_base + 4) as *mut v128, acc01);
            v128_store(c.as_mut_ptr().add(c1_base + 0) as *mut v128, acc10);
            v128_store(c.as_mut_ptr().add(c1_base + 4) as *mut v128, acc11);
            v128_store(c.as_mut_ptr().add(c2_base + 0) as *mut v128, acc20);
            v128_store(c.as_mut_ptr().add(c2_base + 4) as *mut v128, acc21);
            v128_store(c.as_mut_ptr().add(c3_base + 0) as *mut v128, acc30);
            v128_store(c.as_mut_ptr().add(c3_base + 4) as *mut v128, acc31);
        }

        // Handle remaining columns (n_main..n) with scalar
        for j in n_main..n {
            for di in 0..MR {
                let ii = i + di;
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += a[ii * k + kk] * b[kk * n + j];
                }
                c[ii * n + j] = sum;
            }
        }
    }

    // Handle remaining rows (m_main..m)
    for i in m_main..m {
        // Use SIMD for columns if n >= 8
        for j in (0..n_main).step_by(8) {
            let mut acc0 = f32x4_splat(0.0);
            let mut acc1 = f32x4_splat(0.0);

            for kk in 0..k {
                let a_val = f32x4_splat(a[i * k + kk]);
                let b_base = kk * n + j;
                let b0 = v128_load(b.as_ptr().add(b_base + 0) as *const v128);
                let b1 = v128_load(b.as_ptr().add(b_base + 4) as *const v128);
                acc0 = f32x4_add(acc0, f32x4_mul(a_val, b0));
                acc1 = f32x4_add(acc1, f32x4_mul(a_val, b1));
            }

            let c_base = i * n + j;
            v128_store(c.as_mut_ptr().add(c_base + 0) as *mut v128, acc0);
            v128_store(c.as_mut_ptr().add(c_base + 4) as *mut v128, acc1);
        }

        // Remaining columns
        for j in n_main..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// FMA-optimized SIMD matrix multiplication for f32
/// Uses relaxed-simd f32x4_relaxed_madd for fused multiply-add
/// This reduces 2 instructions (mul + add) to 1 instruction (fmadd)
///
/// # Safety
/// - a must have at least m*k elements
/// - b must have at least k*n elements
/// - c must have at least m*n elements
#[cfg(target_arch = "wasm32")]
pub unsafe fn matmul_simd_f32_fma(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    const MR: usize = 4;
    const NR: usize = 8;
    const KU: usize = 4;  // K unroll factor

    let m_main = m / MR * MR;
    let n_main = n / NR * NR;
    let k_main = k / KU * KU;

    // Main loop: process MR rows × NR cols at a time
    for i in (0..m_main).step_by(MR) {
        for j in (0..n_main).step_by(NR) {
            // 4 rows × 8 cols = 8 v128 accumulators (2 per row)
            let mut acc00 = f32x4_splat(0.0);
            let mut acc01 = f32x4_splat(0.0);
            let mut acc10 = f32x4_splat(0.0);
            let mut acc11 = f32x4_splat(0.0);
            let mut acc20 = f32x4_splat(0.0);
            let mut acc21 = f32x4_splat(0.0);
            let mut acc30 = f32x4_splat(0.0);
            let mut acc31 = f32x4_splat(0.0);

            // K loop - 4x unrolled with FMA
            let mut kk = 0;
            while kk < k_main {
                // Unroll 0 - using FMA: acc = a * b + acc
                {
                    let a0 = f32x4_splat(*a.get_unchecked((i + 0) * k + kk));
                    let a1 = f32x4_splat(*a.get_unchecked((i + 1) * k + kk));
                    let a2 = f32x4_splat(*a.get_unchecked((i + 2) * k + kk));
                    let a3 = f32x4_splat(*a.get_unchecked((i + 3) * k + kk));
                    let b_base = kk * n + j;
                    let b0 = v128_load(b.as_ptr().add(b_base + 0) as *const v128);
                    let b1 = v128_load(b.as_ptr().add(b_base + 4) as *const v128);
                    acc00 = f32x4_relaxed_madd(a0, b0, acc00);
                    acc01 = f32x4_relaxed_madd(a0, b1, acc01);
                    acc10 = f32x4_relaxed_madd(a1, b0, acc10);
                    acc11 = f32x4_relaxed_madd(a1, b1, acc11);
                    acc20 = f32x4_relaxed_madd(a2, b0, acc20);
                    acc21 = f32x4_relaxed_madd(a2, b1, acc21);
                    acc30 = f32x4_relaxed_madd(a3, b0, acc30);
                    acc31 = f32x4_relaxed_madd(a3, b1, acc31);
                }
                // Unroll 1
                {
                    let a0 = f32x4_splat(*a.get_unchecked((i + 0) * k + kk + 1));
                    let a1 = f32x4_splat(*a.get_unchecked((i + 1) * k + kk + 1));
                    let a2 = f32x4_splat(*a.get_unchecked((i + 2) * k + kk + 1));
                    let a3 = f32x4_splat(*a.get_unchecked((i + 3) * k + kk + 1));
                    let b_base = (kk + 1) * n + j;
                    let b0 = v128_load(b.as_ptr().add(b_base + 0) as *const v128);
                    let b1 = v128_load(b.as_ptr().add(b_base + 4) as *const v128);
                    acc00 = f32x4_relaxed_madd(a0, b0, acc00);
                    acc01 = f32x4_relaxed_madd(a0, b1, acc01);
                    acc10 = f32x4_relaxed_madd(a1, b0, acc10);
                    acc11 = f32x4_relaxed_madd(a1, b1, acc11);
                    acc20 = f32x4_relaxed_madd(a2, b0, acc20);
                    acc21 = f32x4_relaxed_madd(a2, b1, acc21);
                    acc30 = f32x4_relaxed_madd(a3, b0, acc30);
                    acc31 = f32x4_relaxed_madd(a3, b1, acc31);
                }
                // Unroll 2
                {
                    let a0 = f32x4_splat(*a.get_unchecked((i + 0) * k + kk + 2));
                    let a1 = f32x4_splat(*a.get_unchecked((i + 1) * k + kk + 2));
                    let a2 = f32x4_splat(*a.get_unchecked((i + 2) * k + kk + 2));
                    let a3 = f32x4_splat(*a.get_unchecked((i + 3) * k + kk + 2));
                    let b_base = (kk + 2) * n + j;
                    let b0 = v128_load(b.as_ptr().add(b_base + 0) as *const v128);
                    let b1 = v128_load(b.as_ptr().add(b_base + 4) as *const v128);
                    acc00 = f32x4_relaxed_madd(a0, b0, acc00);
                    acc01 = f32x4_relaxed_madd(a0, b1, acc01);
                    acc10 = f32x4_relaxed_madd(a1, b0, acc10);
                    acc11 = f32x4_relaxed_madd(a1, b1, acc11);
                    acc20 = f32x4_relaxed_madd(a2, b0, acc20);
                    acc21 = f32x4_relaxed_madd(a2, b1, acc21);
                    acc30 = f32x4_relaxed_madd(a3, b0, acc30);
                    acc31 = f32x4_relaxed_madd(a3, b1, acc31);
                }
                // Unroll 3
                {
                    let a0 = f32x4_splat(*a.get_unchecked((i + 0) * k + kk + 3));
                    let a1 = f32x4_splat(*a.get_unchecked((i + 1) * k + kk + 3));
                    let a2 = f32x4_splat(*a.get_unchecked((i + 2) * k + kk + 3));
                    let a3 = f32x4_splat(*a.get_unchecked((i + 3) * k + kk + 3));
                    let b_base = (kk + 3) * n + j;
                    let b0 = v128_load(b.as_ptr().add(b_base + 0) as *const v128);
                    let b1 = v128_load(b.as_ptr().add(b_base + 4) as *const v128);
                    acc00 = f32x4_relaxed_madd(a0, b0, acc00);
                    acc01 = f32x4_relaxed_madd(a0, b1, acc01);
                    acc10 = f32x4_relaxed_madd(a1, b0, acc10);
                    acc11 = f32x4_relaxed_madd(a1, b1, acc11);
                    acc20 = f32x4_relaxed_madd(a2, b0, acc20);
                    acc21 = f32x4_relaxed_madd(a2, b1, acc21);
                    acc30 = f32x4_relaxed_madd(a3, b0, acc30);
                    acc31 = f32x4_relaxed_madd(a3, b1, acc31);
                }
                kk += KU;
            }

            // Handle remaining K iterations
            while kk < k {
                let a0 = f32x4_splat(*a.get_unchecked((i + 0) * k + kk));
                let a1 = f32x4_splat(*a.get_unchecked((i + 1) * k + kk));
                let a2 = f32x4_splat(*a.get_unchecked((i + 2) * k + kk));
                let a3 = f32x4_splat(*a.get_unchecked((i + 3) * k + kk));
                let b_base = kk * n + j;
                let b0 = v128_load(b.as_ptr().add(b_base + 0) as *const v128);
                let b1 = v128_load(b.as_ptr().add(b_base + 4) as *const v128);
                acc00 = f32x4_relaxed_madd(a0, b0, acc00);
                acc01 = f32x4_relaxed_madd(a0, b1, acc01);
                acc10 = f32x4_relaxed_madd(a1, b0, acc10);
                acc11 = f32x4_relaxed_madd(a1, b1, acc11);
                acc20 = f32x4_relaxed_madd(a2, b0, acc20);
                acc21 = f32x4_relaxed_madd(a2, b1, acc21);
                acc30 = f32x4_relaxed_madd(a3, b0, acc30);
                acc31 = f32x4_relaxed_madd(a3, b1, acc31);
                kk += 1;
            }

            // Store results
            let c0_base = (i + 0) * n + j;
            let c1_base = (i + 1) * n + j;
            let c2_base = (i + 2) * n + j;
            let c3_base = (i + 3) * n + j;
            v128_store(c.as_mut_ptr().add(c0_base + 0) as *mut v128, acc00);
            v128_store(c.as_mut_ptr().add(c0_base + 4) as *mut v128, acc01);
            v128_store(c.as_mut_ptr().add(c1_base + 0) as *mut v128, acc10);
            v128_store(c.as_mut_ptr().add(c1_base + 4) as *mut v128, acc11);
            v128_store(c.as_mut_ptr().add(c2_base + 0) as *mut v128, acc20);
            v128_store(c.as_mut_ptr().add(c2_base + 4) as *mut v128, acc21);
            v128_store(c.as_mut_ptr().add(c3_base + 0) as *mut v128, acc30);
            v128_store(c.as_mut_ptr().add(c3_base + 4) as *mut v128, acc31);
        }

        // Handle remaining columns with scalar
        for j in n_main..n {
            for di in 0..MR {
                let ii = i + di;
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += a[ii * k + kk] * b[kk * n + j];
                }
                c[ii * n + j] = sum;
            }
        }
    }

    // Handle remaining rows
    for i in m_main..m {
        for j in (0..n_main).step_by(8) {
            let mut acc0 = f32x4_splat(0.0);
            let mut acc1 = f32x4_splat(0.0);

            for kk in 0..k {
                let a_val = f32x4_splat(a[i * k + kk]);
                let b_base = kk * n + j;
                let b0 = v128_load(b.as_ptr().add(b_base + 0) as *const v128);
                let b1 = v128_load(b.as_ptr().add(b_base + 4) as *const v128);
                acc0 = f32x4_relaxed_madd(a_val, b0, acc0);
                acc1 = f32x4_relaxed_madd(a_val, b1, acc1);
            }

            let c_base = i * n + j;
            v128_store(c.as_mut_ptr().add(c_base + 0) as *mut v128, acc0);
            v128_store(c.as_mut_ptr().add(c_base + 4) as *mut v128, acc1);
        }

        for j in n_main..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// Direct SIMD matrix multiplication for f64
/// Uses 2x8 micro-kernel (2 rows, 8 cols = 4 v128 accumulators per row)
#[cfg(target_arch = "wasm32")]
pub unsafe fn matmul_simd_f64(a: &[f64], b: &[f64], c: &mut [f64], m: usize, n: usize, k: usize) {
    const MR: usize = 2;
    const NR: usize = 8;

    let m_main = m / MR * MR;
    let n_main = n / NR * NR;

    // Main loop: process MR rows × NR cols at a time
    for i in (0..m_main).step_by(MR) {
        for j in (0..n_main).step_by(NR) {
            // 2 rows × 8 cols = 8 v128 accumulators
            let mut acc0_0 = f64x2_splat(0.0);
            let mut acc0_1 = f64x2_splat(0.0);
            let mut acc0_2 = f64x2_splat(0.0);
            let mut acc0_3 = f64x2_splat(0.0);
            let mut acc1_0 = f64x2_splat(0.0);
            let mut acc1_1 = f64x2_splat(0.0);
            let mut acc1_2 = f64x2_splat(0.0);
            let mut acc1_3 = f64x2_splat(0.0);

            // K loop
            for kk in 0..k {
                let a0 = f64x2_splat(a[(i + 0) * k + kk]);
                let a1 = f64x2_splat(a[(i + 1) * k + kk]);

                let b_base = kk * n + j;
                let b0 = v128_load(b.as_ptr().add(b_base + 0) as *const v128);
                let b1 = v128_load(b.as_ptr().add(b_base + 2) as *const v128);
                let b2 = v128_load(b.as_ptr().add(b_base + 4) as *const v128);
                let b3 = v128_load(b.as_ptr().add(b_base + 6) as *const v128);

                acc0_0 = f64x2_add(acc0_0, f64x2_mul(a0, b0));
                acc0_1 = f64x2_add(acc0_1, f64x2_mul(a0, b1));
                acc0_2 = f64x2_add(acc0_2, f64x2_mul(a0, b2));
                acc0_3 = f64x2_add(acc0_3, f64x2_mul(a0, b3));
                acc1_0 = f64x2_add(acc1_0, f64x2_mul(a1, b0));
                acc1_1 = f64x2_add(acc1_1, f64x2_mul(a1, b1));
                acc1_2 = f64x2_add(acc1_2, f64x2_mul(a1, b2));
                acc1_3 = f64x2_add(acc1_3, f64x2_mul(a1, b3));
            }

            let c0_base = (i + 0) * n + j;
            let c1_base = (i + 1) * n + j;
            v128_store(c.as_mut_ptr().add(c0_base + 0) as *mut v128, acc0_0);
            v128_store(c.as_mut_ptr().add(c0_base + 2) as *mut v128, acc0_1);
            v128_store(c.as_mut_ptr().add(c0_base + 4) as *mut v128, acc0_2);
            v128_store(c.as_mut_ptr().add(c0_base + 6) as *mut v128, acc0_3);
            v128_store(c.as_mut_ptr().add(c1_base + 0) as *mut v128, acc1_0);
            v128_store(c.as_mut_ptr().add(c1_base + 2) as *mut v128, acc1_1);
            v128_store(c.as_mut_ptr().add(c1_base + 4) as *mut v128, acc1_2);
            v128_store(c.as_mut_ptr().add(c1_base + 6) as *mut v128, acc1_3);
        }

        for j in n_main..n {
            for di in 0..MR {
                let ii = i + di;
                let mut sum = 0.0;
                for kk in 0..k {
                    sum += a[ii * k + kk] * b[kk * n + j];
                }
                c[ii * n + j] = sum;
            }
        }
    }

    for i in m_main..m {
        for j in (0..n_main).step_by(8) {
            let mut acc0 = f64x2_splat(0.0);
            let mut acc1 = f64x2_splat(0.0);
            let mut acc2 = f64x2_splat(0.0);
            let mut acc3 = f64x2_splat(0.0);

            for kk in 0..k {
                let a_val = f64x2_splat(a[i * k + kk]);
                let b_base = kk * n + j;
                let b0 = v128_load(b.as_ptr().add(b_base + 0) as *const v128);
                let b1 = v128_load(b.as_ptr().add(b_base + 2) as *const v128);
                let b2 = v128_load(b.as_ptr().add(b_base + 4) as *const v128);
                let b3 = v128_load(b.as_ptr().add(b_base + 6) as *const v128);
                acc0 = f64x2_add(acc0, f64x2_mul(a_val, b0));
                acc1 = f64x2_add(acc1, f64x2_mul(a_val, b1));
                acc2 = f64x2_add(acc2, f64x2_mul(a_val, b2));
                acc3 = f64x2_add(acc3, f64x2_mul(a_val, b3));
            }

            let c_base = i * n + j;
            v128_store(c.as_mut_ptr().add(c_base + 0) as *mut v128, acc0);
            v128_store(c.as_mut_ptr().add(c_base + 2) as *mut v128, acc1);
            v128_store(c.as_mut_ptr().add(c_base + 4) as *mut v128, acc2);
            v128_store(c.as_mut_ptr().add(c_base + 6) as *mut v128, acc3);
        }

        for j in n_main..n {
            let mut sum = 0.0;
            for kk in 0..k {
                sum += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// Check if WASM SIMD is available at compile time
#[cfg(target_arch = "wasm32")]
pub fn simd_available() -> bool {
    true // If we're on wasm32 with this build, SIMD is enabled
}

#[cfg(not(target_arch = "wasm32"))]
pub fn simd_available() -> bool {
    false
}

/// FMA + Packed B: combines both optimizations for best performance
/// Uses relaxed-simd FMA with pre-packed B matrix
#[cfg(target_arch = "wasm32")]
pub unsafe fn matmul_simd_f32_fma_packed(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    const MR: usize = 4;
    const NR: usize = 8;
    const KU: usize = 4;

    let m_main = m / MR * MR;
    let n_main = n / NR * NR;
    let n_panels = n / NR;
    let k_main = k / KU * KU;

    // Pack B matrix for better cache locality
    let mut packed_b = vec![0.0f32; n_panels * k * NR];
    pack_b_f32(b, &mut packed_b, k, n);

    // Main loop: process MR rows × NR cols at a time
    for i in (0..m_main).step_by(MR) {
        for panel in 0..n_panels {
            let j = panel * NR;
            let panel_offset = panel * k * NR;

            // 4 rows × 8 cols = 8 v128 accumulators (2 per row)
            let mut acc00 = f32x4_splat(0.0);
            let mut acc01 = f32x4_splat(0.0);
            let mut acc10 = f32x4_splat(0.0);
            let mut acc11 = f32x4_splat(0.0);
            let mut acc20 = f32x4_splat(0.0);
            let mut acc21 = f32x4_splat(0.0);
            let mut acc30 = f32x4_splat(0.0);
            let mut acc31 = f32x4_splat(0.0);

            // K loop - 4x unrolled with FMA using packed B
            let mut kk = 0;
            while kk < k_main {
                // Unroll 0
                {
                    let a0 = f32x4_splat(*a.get_unchecked((i + 0) * k + kk));
                    let a1 = f32x4_splat(*a.get_unchecked((i + 1) * k + kk));
                    let a2 = f32x4_splat(*a.get_unchecked((i + 2) * k + kk));
                    let a3 = f32x4_splat(*a.get_unchecked((i + 3) * k + kk));
                    let pack_offset = panel_offset + kk * NR;
                    let b0 = v128_load(packed_b.as_ptr().add(pack_offset + 0) as *const v128);
                    let b1 = v128_load(packed_b.as_ptr().add(pack_offset + 4) as *const v128);
                    acc00 = f32x4_relaxed_madd(a0, b0, acc00);
                    acc01 = f32x4_relaxed_madd(a0, b1, acc01);
                    acc10 = f32x4_relaxed_madd(a1, b0, acc10);
                    acc11 = f32x4_relaxed_madd(a1, b1, acc11);
                    acc20 = f32x4_relaxed_madd(a2, b0, acc20);
                    acc21 = f32x4_relaxed_madd(a2, b1, acc21);
                    acc30 = f32x4_relaxed_madd(a3, b0, acc30);
                    acc31 = f32x4_relaxed_madd(a3, b1, acc31);
                }
                // Unroll 1
                {
                    let a0 = f32x4_splat(*a.get_unchecked((i + 0) * k + kk + 1));
                    let a1 = f32x4_splat(*a.get_unchecked((i + 1) * k + kk + 1));
                    let a2 = f32x4_splat(*a.get_unchecked((i + 2) * k + kk + 1));
                    let a3 = f32x4_splat(*a.get_unchecked((i + 3) * k + kk + 1));
                    let pack_offset = panel_offset + (kk + 1) * NR;
                    let b0 = v128_load(packed_b.as_ptr().add(pack_offset + 0) as *const v128);
                    let b1 = v128_load(packed_b.as_ptr().add(pack_offset + 4) as *const v128);
                    acc00 = f32x4_relaxed_madd(a0, b0, acc00);
                    acc01 = f32x4_relaxed_madd(a0, b1, acc01);
                    acc10 = f32x4_relaxed_madd(a1, b0, acc10);
                    acc11 = f32x4_relaxed_madd(a1, b1, acc11);
                    acc20 = f32x4_relaxed_madd(a2, b0, acc20);
                    acc21 = f32x4_relaxed_madd(a2, b1, acc21);
                    acc30 = f32x4_relaxed_madd(a3, b0, acc30);
                    acc31 = f32x4_relaxed_madd(a3, b1, acc31);
                }
                // Unroll 2
                {
                    let a0 = f32x4_splat(*a.get_unchecked((i + 0) * k + kk + 2));
                    let a1 = f32x4_splat(*a.get_unchecked((i + 1) * k + kk + 2));
                    let a2 = f32x4_splat(*a.get_unchecked((i + 2) * k + kk + 2));
                    let a3 = f32x4_splat(*a.get_unchecked((i + 3) * k + kk + 2));
                    let pack_offset = panel_offset + (kk + 2) * NR;
                    let b0 = v128_load(packed_b.as_ptr().add(pack_offset + 0) as *const v128);
                    let b1 = v128_load(packed_b.as_ptr().add(pack_offset + 4) as *const v128);
                    acc00 = f32x4_relaxed_madd(a0, b0, acc00);
                    acc01 = f32x4_relaxed_madd(a0, b1, acc01);
                    acc10 = f32x4_relaxed_madd(a1, b0, acc10);
                    acc11 = f32x4_relaxed_madd(a1, b1, acc11);
                    acc20 = f32x4_relaxed_madd(a2, b0, acc20);
                    acc21 = f32x4_relaxed_madd(a2, b1, acc21);
                    acc30 = f32x4_relaxed_madd(a3, b0, acc30);
                    acc31 = f32x4_relaxed_madd(a3, b1, acc31);
                }
                // Unroll 3
                {
                    let a0 = f32x4_splat(*a.get_unchecked((i + 0) * k + kk + 3));
                    let a1 = f32x4_splat(*a.get_unchecked((i + 1) * k + kk + 3));
                    let a2 = f32x4_splat(*a.get_unchecked((i + 2) * k + kk + 3));
                    let a3 = f32x4_splat(*a.get_unchecked((i + 3) * k + kk + 3));
                    let pack_offset = panel_offset + (kk + 3) * NR;
                    let b0 = v128_load(packed_b.as_ptr().add(pack_offset + 0) as *const v128);
                    let b1 = v128_load(packed_b.as_ptr().add(pack_offset + 4) as *const v128);
                    acc00 = f32x4_relaxed_madd(a0, b0, acc00);
                    acc01 = f32x4_relaxed_madd(a0, b1, acc01);
                    acc10 = f32x4_relaxed_madd(a1, b0, acc10);
                    acc11 = f32x4_relaxed_madd(a1, b1, acc11);
                    acc20 = f32x4_relaxed_madd(a2, b0, acc20);
                    acc21 = f32x4_relaxed_madd(a2, b1, acc21);
                    acc30 = f32x4_relaxed_madd(a3, b0, acc30);
                    acc31 = f32x4_relaxed_madd(a3, b1, acc31);
                }
                kk += KU;
            }

            // Handle remaining K iterations
            while kk < k {
                let a0 = f32x4_splat(*a.get_unchecked((i + 0) * k + kk));
                let a1 = f32x4_splat(*a.get_unchecked((i + 1) * k + kk));
                let a2 = f32x4_splat(*a.get_unchecked((i + 2) * k + kk));
                let a3 = f32x4_splat(*a.get_unchecked((i + 3) * k + kk));
                let pack_offset = panel_offset + kk * NR;
                let b0 = v128_load(packed_b.as_ptr().add(pack_offset + 0) as *const v128);
                let b1 = v128_load(packed_b.as_ptr().add(pack_offset + 4) as *const v128);
                acc00 = f32x4_relaxed_madd(a0, b0, acc00);
                acc01 = f32x4_relaxed_madd(a0, b1, acc01);
                acc10 = f32x4_relaxed_madd(a1, b0, acc10);
                acc11 = f32x4_relaxed_madd(a1, b1, acc11);
                acc20 = f32x4_relaxed_madd(a2, b0, acc20);
                acc21 = f32x4_relaxed_madd(a2, b1, acc21);
                acc30 = f32x4_relaxed_madd(a3, b0, acc30);
                acc31 = f32x4_relaxed_madd(a3, b1, acc31);
                kk += 1;
            }

            // Store results
            let c0_base = (i + 0) * n + j;
            let c1_base = (i + 1) * n + j;
            let c2_base = (i + 2) * n + j;
            let c3_base = (i + 3) * n + j;
            v128_store(c.as_mut_ptr().add(c0_base + 0) as *mut v128, acc00);
            v128_store(c.as_mut_ptr().add(c0_base + 4) as *mut v128, acc01);
            v128_store(c.as_mut_ptr().add(c1_base + 0) as *mut v128, acc10);
            v128_store(c.as_mut_ptr().add(c1_base + 4) as *mut v128, acc11);
            v128_store(c.as_mut_ptr().add(c2_base + 0) as *mut v128, acc20);
            v128_store(c.as_mut_ptr().add(c2_base + 4) as *mut v128, acc21);
            v128_store(c.as_mut_ptr().add(c3_base + 0) as *mut v128, acc30);
            v128_store(c.as_mut_ptr().add(c3_base + 4) as *mut v128, acc31);
        }

        // Handle remaining columns (n_main..n) with scalar
        for j in n_main..n {
            for di in 0..MR {
                let ii = i + di;
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += a[ii * k + kk] * b[kk * n + j];
                }
                c[ii * n + j] = sum;
            }
        }
    }

    // Handle remaining rows (m_main..m)
    for i in m_main..m {
        // Use packed B for columns that fit
        for panel in 0..n_panels {
            let j = panel * NR;
            let panel_offset = panel * k * NR;

            let mut acc0 = f32x4_splat(0.0);
            let mut acc1 = f32x4_splat(0.0);

            for kk in 0..k {
                let a_val = f32x4_splat(a[i * k + kk]);
                let pack_offset = panel_offset + kk * NR;
                let b0 = v128_load(packed_b.as_ptr().add(pack_offset + 0) as *const v128);
                let b1 = v128_load(packed_b.as_ptr().add(pack_offset + 4) as *const v128);
                acc0 = f32x4_relaxed_madd(a_val, b0, acc0);
                acc1 = f32x4_relaxed_madd(a_val, b1, acc1);
            }

            let c_base = i * n + j;
            v128_store(c.as_mut_ptr().add(c_base + 0) as *mut v128, acc0);
            v128_store(c.as_mut_ptr().add(c_base + 4) as *mut v128, acc1);
        }

        // Remaining columns
        for j in n_main..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// 2x8 micro-kernel for small matrices (M < 4)
/// More efficient when there are few rows
#[cfg(target_arch = "wasm32")]
pub unsafe fn matmul_simd_f32_2x8(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    const MR: usize = 2;
    const NR: usize = 8;

    let m_main = m / MR * MR;
    let n_main = n / NR * NR;

    for i in (0..m_main).step_by(MR) {
        for j in (0..n_main).step_by(NR) {
            let mut acc00 = f32x4_splat(0.0);
            let mut acc01 = f32x4_splat(0.0);
            let mut acc10 = f32x4_splat(0.0);
            let mut acc11 = f32x4_splat(0.0);

            for kk in 0..k {
                let a0 = f32x4_splat(*a.get_unchecked((i + 0) * k + kk));
                let a1 = f32x4_splat(*a.get_unchecked((i + 1) * k + kk));
                let b_base = kk * n + j;
                let b0 = v128_load(b.as_ptr().add(b_base + 0) as *const v128);
                let b1 = v128_load(b.as_ptr().add(b_base + 4) as *const v128);
                acc00 = f32x4_relaxed_madd(a0, b0, acc00);
                acc01 = f32x4_relaxed_madd(a0, b1, acc01);
                acc10 = f32x4_relaxed_madd(a1, b0, acc10);
                acc11 = f32x4_relaxed_madd(a1, b1, acc11);
            }

            let c0_base = (i + 0) * n + j;
            let c1_base = (i + 1) * n + j;
            v128_store(c.as_mut_ptr().add(c0_base + 0) as *mut v128, acc00);
            v128_store(c.as_mut_ptr().add(c0_base + 4) as *mut v128, acc01);
            v128_store(c.as_mut_ptr().add(c1_base + 0) as *mut v128, acc10);
            v128_store(c.as_mut_ptr().add(c1_base + 4) as *mut v128, acc11);
        }

        // Remaining columns
        for j in n_main..n {
            for di in 0..MR {
                let ii = i + di;
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += a[ii * k + kk] * b[kk * n + j];
                }
                c[ii * n + j] = sum;
            }
        }
    }

    // Remaining rows
    for i in m_main..m {
        for j in (0..n_main).step_by(8) {
            let mut acc0 = f32x4_splat(0.0);
            let mut acc1 = f32x4_splat(0.0);

            for kk in 0..k {
                let a_val = f32x4_splat(a[i * k + kk]);
                let b_base = kk * n + j;
                let b0 = v128_load(b.as_ptr().add(b_base + 0) as *const v128);
                let b1 = v128_load(b.as_ptr().add(b_base + 4) as *const v128);
                acc0 = f32x4_relaxed_madd(a_val, b0, acc0);
                acc1 = f32x4_relaxed_madd(a_val, b1, acc1);
            }

            let c_base = i * n + j;
            v128_store(c.as_mut_ptr().add(c_base + 0) as *mut v128, acc0);
            v128_store(c.as_mut_ptr().add(c_base + 4) as *mut v128, acc1);
        }

        for j in n_main..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// 6x8 micro-kernel matching XNNPACK exactly - using MUL+ADD not FMA
/// XNNPACK uses wasm_f32x4_add(wasm_f32x4_mul(a, b), c), NOT FMA
#[cfg(target_arch = "wasm32")]
pub unsafe fn matmul_simd_f32_6x8_muladd(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    const MR: usize = 6;
    const NR: usize = 8;

    let m_main = m / MR * MR;
    let n_main = n / NR * NR;

    for i in (0..m_main).step_by(MR) {
        for j in (0..n_main).step_by(NR) {
            // 6 rows × 8 cols = 12 accumulators
            let mut acc00 = f32x4_splat(0.0);
            let mut acc01 = f32x4_splat(0.0);
            let mut acc10 = f32x4_splat(0.0);
            let mut acc11 = f32x4_splat(0.0);
            let mut acc20 = f32x4_splat(0.0);
            let mut acc21 = f32x4_splat(0.0);
            let mut acc30 = f32x4_splat(0.0);
            let mut acc31 = f32x4_splat(0.0);
            let mut acc40 = f32x4_splat(0.0);
            let mut acc41 = f32x4_splat(0.0);
            let mut acc50 = f32x4_splat(0.0);
            let mut acc51 = f32x4_splat(0.0);

            for kk in 0..k {
                let a0 = f32x4_splat(*a.get_unchecked((i + 0) * k + kk));
                let a1 = f32x4_splat(*a.get_unchecked((i + 1) * k + kk));
                let a2 = f32x4_splat(*a.get_unchecked((i + 2) * k + kk));
                let a3 = f32x4_splat(*a.get_unchecked((i + 3) * k + kk));
                let a4 = f32x4_splat(*a.get_unchecked((i + 4) * k + kk));
                let a5 = f32x4_splat(*a.get_unchecked((i + 5) * k + kk));
                let b_base = kk * n + j;
                let b0 = v128_load(b.as_ptr().add(b_base + 0) as *const v128);
                let b1 = v128_load(b.as_ptr().add(b_base + 4) as *const v128);
                // XNNPACK style: mul + add, not FMA
                acc00 = f32x4_add(f32x4_mul(a0, b0), acc00);
                acc01 = f32x4_add(f32x4_mul(a0, b1), acc01);
                acc10 = f32x4_add(f32x4_mul(a1, b0), acc10);
                acc11 = f32x4_add(f32x4_mul(a1, b1), acc11);
                acc20 = f32x4_add(f32x4_mul(a2, b0), acc20);
                acc21 = f32x4_add(f32x4_mul(a2, b1), acc21);
                acc30 = f32x4_add(f32x4_mul(a3, b0), acc30);
                acc31 = f32x4_add(f32x4_mul(a3, b1), acc31);
                acc40 = f32x4_add(f32x4_mul(a4, b0), acc40);
                acc41 = f32x4_add(f32x4_mul(a4, b1), acc41);
                acc50 = f32x4_add(f32x4_mul(a5, b0), acc50);
                acc51 = f32x4_add(f32x4_mul(a5, b1), acc51);
            }

            v128_store(c.as_mut_ptr().add((i + 0) * n + j + 0) as *mut v128, acc00);
            v128_store(c.as_mut_ptr().add((i + 0) * n + j + 4) as *mut v128, acc01);
            v128_store(c.as_mut_ptr().add((i + 1) * n + j + 0) as *mut v128, acc10);
            v128_store(c.as_mut_ptr().add((i + 1) * n + j + 4) as *mut v128, acc11);
            v128_store(c.as_mut_ptr().add((i + 2) * n + j + 0) as *mut v128, acc20);
            v128_store(c.as_mut_ptr().add((i + 2) * n + j + 4) as *mut v128, acc21);
            v128_store(c.as_mut_ptr().add((i + 3) * n + j + 0) as *mut v128, acc30);
            v128_store(c.as_mut_ptr().add((i + 3) * n + j + 4) as *mut v128, acc31);
            v128_store(c.as_mut_ptr().add((i + 4) * n + j + 0) as *mut v128, acc40);
            v128_store(c.as_mut_ptr().add((i + 4) * n + j + 4) as *mut v128, acc41);
            v128_store(c.as_mut_ptr().add((i + 5) * n + j + 0) as *mut v128, acc50);
            v128_store(c.as_mut_ptr().add((i + 5) * n + j + 4) as *mut v128, acc51);
        }

        // Remaining columns
        for j in n_main..n {
            for di in 0..MR {
                let ii = i + di;
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += a[ii * k + kk] * b[kk * n + j];
                }
                c[ii * n + j] = sum;
            }
        }
    }

    // Remaining rows - use scalar
    for i in m_main..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// 6x8 micro-kernel matching XNNPACK's tile size
/// Uses 6 rows x 8 cols = 12 accumulators (like XNNPACK)
#[cfg(target_arch = "wasm32")]
pub unsafe fn matmul_simd_f32_6x8(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    const MR: usize = 6;
    const NR: usize = 8;

    let m_main = m / MR * MR;
    let n_main = n / NR * NR;

    for i in (0..m_main).step_by(MR) {
        for j in (0..n_main).step_by(NR) {
            // 6 rows × 8 cols = 12 accumulators
            let mut acc00 = f32x4_splat(0.0);
            let mut acc01 = f32x4_splat(0.0);
            let mut acc10 = f32x4_splat(0.0);
            let mut acc11 = f32x4_splat(0.0);
            let mut acc20 = f32x4_splat(0.0);
            let mut acc21 = f32x4_splat(0.0);
            let mut acc30 = f32x4_splat(0.0);
            let mut acc31 = f32x4_splat(0.0);
            let mut acc40 = f32x4_splat(0.0);
            let mut acc41 = f32x4_splat(0.0);
            let mut acc50 = f32x4_splat(0.0);
            let mut acc51 = f32x4_splat(0.0);

            for kk in 0..k {
                let a0 = f32x4_splat(*a.get_unchecked((i + 0) * k + kk));
                let a1 = f32x4_splat(*a.get_unchecked((i + 1) * k + kk));
                let a2 = f32x4_splat(*a.get_unchecked((i + 2) * k + kk));
                let a3 = f32x4_splat(*a.get_unchecked((i + 3) * k + kk));
                let a4 = f32x4_splat(*a.get_unchecked((i + 4) * k + kk));
                let a5 = f32x4_splat(*a.get_unchecked((i + 5) * k + kk));
                let b_base = kk * n + j;
                let b0 = v128_load(b.as_ptr().add(b_base + 0) as *const v128);
                let b1 = v128_load(b.as_ptr().add(b_base + 4) as *const v128);
                acc00 = f32x4_relaxed_madd(a0, b0, acc00);
                acc01 = f32x4_relaxed_madd(a0, b1, acc01);
                acc10 = f32x4_relaxed_madd(a1, b0, acc10);
                acc11 = f32x4_relaxed_madd(a1, b1, acc11);
                acc20 = f32x4_relaxed_madd(a2, b0, acc20);
                acc21 = f32x4_relaxed_madd(a2, b1, acc21);
                acc30 = f32x4_relaxed_madd(a3, b0, acc30);
                acc31 = f32x4_relaxed_madd(a3, b1, acc31);
                acc40 = f32x4_relaxed_madd(a4, b0, acc40);
                acc41 = f32x4_relaxed_madd(a4, b1, acc41);
                acc50 = f32x4_relaxed_madd(a5, b0, acc50);
                acc51 = f32x4_relaxed_madd(a5, b1, acc51);
            }

            v128_store(c.as_mut_ptr().add((i + 0) * n + j + 0) as *mut v128, acc00);
            v128_store(c.as_mut_ptr().add((i + 0) * n + j + 4) as *mut v128, acc01);
            v128_store(c.as_mut_ptr().add((i + 1) * n + j + 0) as *mut v128, acc10);
            v128_store(c.as_mut_ptr().add((i + 1) * n + j + 4) as *mut v128, acc11);
            v128_store(c.as_mut_ptr().add((i + 2) * n + j + 0) as *mut v128, acc20);
            v128_store(c.as_mut_ptr().add((i + 2) * n + j + 4) as *mut v128, acc21);
            v128_store(c.as_mut_ptr().add((i + 3) * n + j + 0) as *mut v128, acc30);
            v128_store(c.as_mut_ptr().add((i + 3) * n + j + 4) as *mut v128, acc31);
            v128_store(c.as_mut_ptr().add((i + 4) * n + j + 0) as *mut v128, acc40);
            v128_store(c.as_mut_ptr().add((i + 4) * n + j + 4) as *mut v128, acc41);
            v128_store(c.as_mut_ptr().add((i + 5) * n + j + 0) as *mut v128, acc50);
            v128_store(c.as_mut_ptr().add((i + 5) * n + j + 4) as *mut v128, acc51);
        }

        // Remaining columns
        for j in n_main..n {
            for di in 0..MR {
                let ii = i + di;
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += a[ii * k + kk] * b[kk * n + j];
                }
                c[ii * n + j] = sum;
            }
        }
    }

    // Remaining rows - use scalar
    for i in m_main..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// 5x8 micro-kernel for matrices where M is divisible by 5 (like 100)
/// Uses 5 rows x 8 cols = 10 accumulators
#[cfg(target_arch = "wasm32")]
pub unsafe fn matmul_simd_f32_5x8(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    const MR: usize = 5;
    const NR: usize = 8;

    let m_main = m / MR * MR;
    let n_main = n / NR * NR;

    for i in (0..m_main).step_by(MR) {
        for j in (0..n_main).step_by(NR) {
            let mut acc00 = f32x4_splat(0.0);
            let mut acc01 = f32x4_splat(0.0);
            let mut acc10 = f32x4_splat(0.0);
            let mut acc11 = f32x4_splat(0.0);
            let mut acc20 = f32x4_splat(0.0);
            let mut acc21 = f32x4_splat(0.0);
            let mut acc30 = f32x4_splat(0.0);
            let mut acc31 = f32x4_splat(0.0);
            let mut acc40 = f32x4_splat(0.0);
            let mut acc41 = f32x4_splat(0.0);

            for kk in 0..k {
                let a0 = f32x4_splat(*a.get_unchecked((i + 0) * k + kk));
                let a1 = f32x4_splat(*a.get_unchecked((i + 1) * k + kk));
                let a2 = f32x4_splat(*a.get_unchecked((i + 2) * k + kk));
                let a3 = f32x4_splat(*a.get_unchecked((i + 3) * k + kk));
                let a4 = f32x4_splat(*a.get_unchecked((i + 4) * k + kk));
                let b_base = kk * n + j;
                let b0 = v128_load(b.as_ptr().add(b_base + 0) as *const v128);
                let b1 = v128_load(b.as_ptr().add(b_base + 4) as *const v128);
                acc00 = f32x4_relaxed_madd(a0, b0, acc00);
                acc01 = f32x4_relaxed_madd(a0, b1, acc01);
                acc10 = f32x4_relaxed_madd(a1, b0, acc10);
                acc11 = f32x4_relaxed_madd(a1, b1, acc11);
                acc20 = f32x4_relaxed_madd(a2, b0, acc20);
                acc21 = f32x4_relaxed_madd(a2, b1, acc21);
                acc30 = f32x4_relaxed_madd(a3, b0, acc30);
                acc31 = f32x4_relaxed_madd(a3, b1, acc31);
                acc40 = f32x4_relaxed_madd(a4, b0, acc40);
                acc41 = f32x4_relaxed_madd(a4, b1, acc41);
            }

            v128_store(c.as_mut_ptr().add((i + 0) * n + j + 0) as *mut v128, acc00);
            v128_store(c.as_mut_ptr().add((i + 0) * n + j + 4) as *mut v128, acc01);
            v128_store(c.as_mut_ptr().add((i + 1) * n + j + 0) as *mut v128, acc10);
            v128_store(c.as_mut_ptr().add((i + 1) * n + j + 4) as *mut v128, acc11);
            v128_store(c.as_mut_ptr().add((i + 2) * n + j + 0) as *mut v128, acc20);
            v128_store(c.as_mut_ptr().add((i + 2) * n + j + 4) as *mut v128, acc21);
            v128_store(c.as_mut_ptr().add((i + 3) * n + j + 0) as *mut v128, acc30);
            v128_store(c.as_mut_ptr().add((i + 3) * n + j + 4) as *mut v128, acc31);
            v128_store(c.as_mut_ptr().add((i + 4) * n + j + 0) as *mut v128, acc40);
            v128_store(c.as_mut_ptr().add((i + 4) * n + j + 4) as *mut v128, acc41);
        }

        // Remaining columns
        for j in n_main..n {
            for di in 0..MR {
                let ii = i + di;
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += a[ii * k + kk] * b[kk * n + j];
                }
                c[ii * n + j] = sum;
            }
        }
    }

    // Remaining rows - use scalar
    for i in m_main..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// Auto-tuned dispatch that picks the best kernel for each size
/// Heuristics based on matrix dimensions and cache considerations
#[cfg(target_arch = "wasm32")]
pub unsafe fn matmul_simd_f32_auto(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    // For very small matrices, avoid packing overhead
    if m < 4 || n < 8 {
        matmul_scalar_f32(a, b, c, m, n, k);
        return;
    }

    // For matrices where M % 6 == 0, use 6x8 kernel (like XNNPACK)
    if m % 6 == 0 && n >= 8 {
        matmul_simd_f32_6x8(a, b, c, m, n, k);
        return;
    }

    // For matrices where M % 5 == 0, use 5x8 kernel
    if m % 5 == 0 && n >= 8 {
        matmul_simd_f32_5x8(a, b, c, m, n, k);
        return;
    }

    // For small M (2-3 rows), use 2x8 kernel
    if m < 4 && m >= 2 && n >= 8 {
        matmul_simd_f32_2x8(a, b, c, m, n, k);
        return;
    }

    // For medium matrices, FMA without packing (packing overhead not amortized)
    if m < 64 || n < 64 || k < 64 {
        matmul_simd_f32_fma(a, b, c, m, n, k);
        return;
    }

    // Default: 6x8 kernel works for most sizes
    matmul_simd_f32_6x8(a, b, c, m, n, k);
}

/// Fallback scalar GEMM for f32
pub fn matmul_scalar_f32(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// Fallback scalar GEMM for f64
pub fn matmul_scalar_f64(a: &[f64], b: &[f64], c: &mut [f64], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for kk in 0..k {
                sum += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// High-level f32 matmul that dispatches to SIMD or scalar
/// Uses packed version for larger matrices where packing overhead is amortized
pub fn matmul_dispatch_f32(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];

    #[cfg(target_arch = "wasm32")]
    {
        if m >= 4 && n >= 8 {
            // Use packed version for matrices >= 64x64 (packing overhead is worth it)
            // For smaller matrices, the overhead of packing isn't amortized
            if m >= 64 && n >= 64 && k >= 64 {
                unsafe {
                    matmul_simd_f32_packed(a, b, &mut c, m, n, k);
                }
            } else {
                unsafe {
                    matmul_simd_f32(a, b, &mut c, m, n, k);
                }
            }
        } else {
            matmul_scalar_f32(a, b, &mut c, m, n, k);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        matmul_scalar_f32(a, b, &mut c, m, n, k);
    }

    c
}

/// High-level f64 matmul that dispatches to SIMD or scalar
pub fn matmul_dispatch_f64(a: &[f64], b: &[f64], m: usize, n: usize, k: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * n];

    #[cfg(target_arch = "wasm32")]
    {
        if m >= 2 && n >= 8 {
            unsafe {
                matmul_simd_f64(a, b, &mut c, m, n, k);
            }
        } else {
            matmul_scalar_f64(a, b, &mut c, m, n, k);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        matmul_scalar_f64(a, b, &mut c, m, n, k);
    }

    c
}

/// Use the gemm crate for highly optimized GEMM
/// The gemm crate uses BLIS-style optimizations including:
/// - Cache-blocking at L1/L2/L3 levels
/// - Micro-kernel tiling
/// - Packing for better memory access patterns
/// This should be competitive with or better than our hand-written kernels
pub fn matmul_gemm_f32(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];

    unsafe {
        gemm::gemm(
            m, n, k,
            c.as_mut_ptr(),
            n as isize, 1,  // C strides: row-major (row stride = n, col stride = 1)
            false,          // don't read C (we're computing C = A*B, not C += A*B)
            a.as_ptr(),
            k as isize, 1,  // A strides: row-major
            b.as_ptr(),
            n as isize, 1,  // B strides: row-major
            1.0,            // alpha
            0.0,            // beta (0 = overwrite, 1 = accumulate)
            false, false, false,  // no conjugation
            gemm::Parallelism::None,  // single-threaded (no rayon in WASM)
        );
    }

    c
}

/// gemm crate version for f64
pub fn matmul_gemm_f64(a: &[f64], b: &[f64], m: usize, n: usize, k: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; m * n];

    unsafe {
        gemm::gemm(
            m, n, k,
            c.as_mut_ptr(),
            n as isize, 1,
            false,
            a.as_ptr(),
            k as isize, 1,
            b.as_ptr(),
            n as isize, 1,
            1.0,
            0.0,
            false, false, false,
            gemm::Parallelism::None,
        );
    }

    c
}

/// Parallel f32 GEMM using rayon
///
/// Splits the M dimension across threads.
/// Each thread computes a block of rows using our SIMD kernel.
pub fn matmul_parallel_f32(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    use rayon::prelude::*;

    // For small matrices, single-threaded is faster (no thread overhead)
    if m * n * k < 64 * 64 * 64 {
        return matmul_dispatch_f32(a, b, m, n, k);
    }

    // Split by rows - each thread gets a chunk of rows
    let num_threads = rayon::current_num_threads();
    let rows_per_thread = (m + num_threads - 1) / num_threads;

    // Each thread produces its portion of C
    let results: Vec<Vec<f32>> = (0..num_threads)
        .into_par_iter()
        .filter_map(|tid| {
            let start_row = tid * rows_per_thread;
            if start_row >= m {
                return None;
            }
            let end_row = (start_row + rows_per_thread).min(m);
            let local_m = end_row - start_row;

            // Extract the portion of A for this thread
            let a_slice = &a[start_row * k..end_row * k];

            // Compute this thread's portion
            let c_local = matmul_dispatch_f32(a_slice, b, local_m, n, k);

            Some(c_local)
        })
        .collect();

    // Combine results
    let mut c = Vec::with_capacity(m * n);
    for chunk in results {
        c.extend(chunk);
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_matmul_f64() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let c = matmul_dispatch_f64(&a, &b, 2, 2, 3);
        assert!((c[0] - 58.0).abs() < 1e-10);
        assert!((c[1] - 64.0).abs() < 1e-10);
        assert!((c[2] - 139.0).abs() < 1e-10);
        assert!((c[3] - 154.0).abs() < 1e-10);
    }

    #[test]
    fn test_scalar_matmul_f32() {
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b: Vec<f32> = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let c = matmul_dispatch_f32(&a, &b, 2, 2, 3);
        assert!((c[0] - 58.0).abs() < 1e-5);
        assert!((c[1] - 64.0).abs() < 1e-5);
        assert!((c[2] - 139.0).abs() < 1e-5);
        assert!((c[3] - 154.0).abs() < 1e-5);
    }
}
