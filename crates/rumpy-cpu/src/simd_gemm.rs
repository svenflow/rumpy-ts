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

/// High-level f32 matmul that dispatches to SIMD or scalar, writing into a pre-allocated slice
///
/// This variant writes directly to the provided output slice, avoiding allocation.
/// Use this for parallel implementations with par_chunks_mut.
///
/// # Arguments
/// * `a` - Input matrix A of shape [m, k]
/// * `b` - Input matrix B of shape [k, n]
/// * `c` - Output slice of at least m*n elements to write result
/// * `m`, `n`, `k` - Matrix dimensions
pub fn matmul_dispatch_f32_into(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    #[cfg(target_arch = "wasm32")]
    {
        if m >= 4 && n >= 8 {
            // Use packed version for matrices >= 64x64 (packing overhead is worth it)
            // For smaller matrices, the overhead of packing isn't amortized
            if m >= 64 && n >= 64 && k >= 64 {
                unsafe {
                    matmul_simd_f32_packed(a, b, c, m, n, k);
                }
            } else {
                unsafe {
                    matmul_simd_f32(a, b, c, m, n, k);
                }
            }
        } else {
            matmul_scalar_f32(a, b, c, m, n, k);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        matmul_scalar_f32(a, b, c, m, n, k);
    }
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
    // u64 mul: WASM usize is 32-bit, m*n*k overflows silently at 2048³.
    if (m as u64) * (n as u64) * (k as u64) < (64u64 * 64 * 64) {
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

/// Parallel f32 GEMM V2 using rayon's par_chunks_mut
///
/// This version writes directly to pre-allocated output memory, avoiding
/// per-thread allocations and the final copy step. This is significantly
/// faster for large matrices.
///
/// Splits the M dimension across threads, with each thread writing to
/// its own non-overlapping portion of the output.
pub fn matmul_parallel_f32_v2(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    use rayon::prelude::*;

    // For small matrices, single-threaded is faster (no thread overhead)
    // u64 mul: WASM usize is 32-bit, m*n*k overflows silently at 2048³.
    if (m as u64) * (n as u64) * (k as u64) < (64u64 * 64 * 64) {
        return matmul_dispatch_f32(a, b, m, n, k);
    }

    // Pre-allocate output
    let mut c = vec![0.0f32; m * n];

    // Calculate chunk size (in elements, not rows)
    let num_threads = rayon::current_num_threads();
    let rows_per_thread = (m + num_threads - 1) / num_threads;
    let chunk_size = rows_per_thread * n;  // elements per chunk

    // Use par_chunks_mut to write directly to output
    c.par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(tid, c_chunk)| {
            let start_row = tid * rows_per_thread;
            if start_row >= m {
                return;
            }

            // Calculate how many rows this chunk actually covers
            let local_m = c_chunk.len() / n;
            if local_m == 0 {
                return;
            }

            // Get the corresponding slice of A
            let a_slice = &a[start_row * k..(start_row + local_m) * k];

            // Write directly to this chunk of C
            matmul_dispatch_f32_into(a_slice, b, c_chunk, local_m, n, k);
        });

    c
}

/// Parallel f32 GEMM using pthreadpool-rs
///
/// This version uses pthreadpool-rs instead of rayon for parallelization.
/// On native platforms, pthreadpool-rs uses its own efficient thread pool with
/// work stealing. On WASM with the `wasm-threads` feature, it uses wasm-bindgen-rayon
/// under the hood.
///
/// This is a drop-in replacement for matmul_parallel_f32_v2 that provides
/// the same API but uses a different threading backend.
pub fn matmul_pthreadpool_f32(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    use pthreadpool_rs::ThreadPool;

    // For small matrices, single-threaded is faster (no thread overhead)
    // u64 mul: WASM usize is 32-bit, m*n*k overflows silently at 2048³.
    if (m as u64) * (n as u64) * (k as u64) < (64u64 * 64 * 64) {
        return matmul_dispatch_f32(a, b, m, n, k);
    }

    // Pre-allocate output
    let mut c = vec![0.0f32; m * n];

    // Use default thread pool (uses available parallelism)
    let pool = ThreadPool::default();
    let num_threads = pool.threads_count();

    // Calculate rows per thread
    let rows_per_thread = (m + num_threads - 1) / num_threads;

    // Convert pointers to usize for Send+Sync (usize is always Send+Sync)
    let a_addr = a.as_ptr() as usize;
    let b_addr = b.as_ptr() as usize;
    let c_addr = c.as_mut_ptr() as usize;

    // Each parallel task processes one chunk of rows
    let num_chunks = (m + rows_per_thread - 1) / rows_per_thread;

    pool.parallelize_1d(num_chunks, |chunk_idx| {
        let start_row = chunk_idx * rows_per_thread;
        if start_row >= m {
            return;
        }

        let end_row = (start_row + rows_per_thread).min(m);
        let local_m = end_row - start_row;

        // Safety: Each chunk writes to a non-overlapping portion of c
        // and reads from shared a and b
        unsafe {
            let a_ptr = a_addr as *const f32;
            let b_ptr = b_addr as *const f32;
            let c_ptr = c_addr as *mut f32;

            let a_slice = std::slice::from_raw_parts(a_ptr.add(start_row * k), local_m * k);
            let b_slice = std::slice::from_raw_parts(b_ptr, k * n);
            let c_slice = std::slice::from_raw_parts_mut(c_ptr.add(start_row * n), local_m * n);

            matmul_dispatch_f32_into(a_slice, b_slice, c_slice, local_m, n, k);
        }
    });

    c
}

/// Parallel f32 GEMM using pthreadpool-rs with provided pool
///
/// Same as matmul_pthreadpool_f32 but reuses an existing thread pool
/// to avoid pool creation overhead for repeated calls.
pub fn matmul_pthreadpool_f32_with_pool(
    pool: &pthreadpool_rs::ThreadPool,
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    // For small matrices, single-threaded is faster (no thread overhead)
    // u64 mul: WASM usize is 32-bit, m*n*k overflows silently at 2048³.
    if (m as u64) * (n as u64) * (k as u64) < (64u64 * 64 * 64) {
        return matmul_dispatch_f32(a, b, m, n, k);
    }

    // Pre-allocate output
    let mut c = vec![0.0f32; m * n];

    let num_threads = pool.threads_count();
    let rows_per_thread = (m + num_threads - 1) / num_threads;

    // Convert pointers to usize for Send+Sync (usize is always Send+Sync)
    let a_addr = a.as_ptr() as usize;
    let b_addr = b.as_ptr() as usize;
    let c_addr = c.as_mut_ptr() as usize;

    // Each parallel task processes one chunk of rows
    let num_chunks = (m + rows_per_thread - 1) / rows_per_thread;

    pool.parallelize_1d(num_chunks, |chunk_idx| {
        let start_row = chunk_idx * rows_per_thread;
        if start_row >= m {
            return;
        }

        let end_row = (start_row + rows_per_thread).min(m);
        let local_m = end_row - start_row;

        // Safety: Each chunk writes to a non-overlapping portion of c
        // and reads from shared a and b
        unsafe {
            let a_ptr = a_addr as *const f32;
            let b_ptr = b_addr as *const f32;
            let c_ptr = c_addr as *mut f32;

            let a_slice = std::slice::from_raw_parts(a_ptr.add(start_row * k), local_m * k);
            let b_slice = std::slice::from_raw_parts(b_ptr, k * n);
            let c_slice = std::slice::from_raw_parts_mut(c_ptr.add(start_row * n), local_m * n);

            matmul_dispatch_f32_into(a_slice, b_slice, c_slice, local_m, n, k);
        }
    });

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

    #[test]
    fn test_pthreadpool_matmul_small() {
        // Small matrix (uses single-threaded path)
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b: Vec<f32> = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let c = matmul_pthreadpool_f32(&a, &b, 2, 2, 3);
        assert!((c[0] - 58.0).abs() < 1e-5);
        assert!((c[1] - 64.0).abs() < 1e-5);
        assert!((c[2] - 139.0).abs() < 1e-5);
        assert!((c[3] - 154.0).abs() < 1e-5);
    }

    #[test]
    fn test_pthreadpool_matmul_large() {
        // Large matrix to trigger parallel path (> 64*64*64 elements)
        let m = 128;
        let n = 128;
        let k = 128;

        // Create random-ish matrices
        let a: Vec<f32> = (0..m*k).map(|i| (i as f32) * 0.001).collect();
        let b: Vec<f32> = (0..k*n).map(|i| (i as f32) * 0.001).collect();

        // Compute with pthreadpool
        let c_pthreadpool = matmul_pthreadpool_f32(&a, &b, m, n, k);

        // Compute with dispatch (single-threaded reference)
        let c_reference = matmul_dispatch_f32(&a, &b, m, n, k);

        // Verify results match
        assert_eq!(c_pthreadpool.len(), c_reference.len());
        for i in 0..c_pthreadpool.len() {
            let diff = (c_pthreadpool[i] - c_reference[i]).abs();
            assert!(
                diff < 1e-3,
                "Mismatch at index {}: pthreadpool={}, reference={}, diff={}",
                i, c_pthreadpool[i], c_reference[i], diff
            );
        }
    }

    #[test]
    fn test_pthreadpool_matmul_with_pool() {
        use pthreadpool_rs::ThreadPool;

        // Large matrix
        let m = 128;
        let n = 128;
        let k = 128;

        let a: Vec<f32> = (0..m*k).map(|i| (i as f32) * 0.001).collect();
        let b: Vec<f32> = (0..k*n).map(|i| (i as f32) * 0.001).collect();

        let pool = ThreadPool::new(4);

        // Compute with provided pool
        let c_pthreadpool = matmul_pthreadpool_f32_with_pool(&pool, &a, &b, m, n, k);

        // Compute reference
        let c_reference = matmul_dispatch_f32(&a, &b, m, n, k);

        // Verify results match
        for i in 0..c_pthreadpool.len() {
            let diff = (c_pthreadpool[i] - c_reference[i]).abs();
            assert!(
                diff < 1e-3,
                "Mismatch at index {}: pthreadpool={}, reference={}, diff={}",
                i, c_pthreadpool[i], c_reference[i], diff
            );
        }
    }
}

// ============================================================================
// OPTIMIZED 6x8 GEMM - XNNPACK-competitive implementation
// ============================================================================
// Key optimizations:
// 1. MR=6, NR=8 tile size (12 v128 accumulators = fits in 16 XMM registers)
// 2. FMA via f32x4_relaxed_madd (single instruction instead of mul+add)
// 3. v128_load32_splat for A values (dedicated instruction)
// 4. L1/L2 cache blocking (KC, MC, NC)
// 5. B matrix packing for contiguous access

/// Cache blocking constants (tuned for typical L1=32KB, L2=256KB)
const OPT_KC: usize = 256;  // K-dimension block (depth of dot product)
const OPT_MC: usize = 72;   // M-dimension block (multiple of MR=6)
const OPT_NC: usize = 128;  // N-dimension block (multiple of NR=8)
const OPT_MR: usize = 6;    // Micro-kernel rows
const OPT_NR: usize = 8;    // Micro-kernel cols

/// Pack B matrix panel into contiguous format for optimal SIMD access.
/// Layout: For each 8-column panel, store K rows contiguously.
/// [k0:col0-7][k1:col0-7]...[k_KC:col0-7]
#[cfg(target_arch = "wasm32")]
pub fn pack_b_optimized(
    b: *const f32,
    ldb: usize,
    packed_b: *mut f32,
    k_size: usize,
    n_size: usize,
) {
    unsafe {
        let mut dest = packed_b;
        let mut j = 0;

        while j < n_size {
            let n_remain = n_size - j;
            let mut src_col = b.add(j);

            if n_remain >= OPT_NR {
                // Fast path: pack full 8 columns
                for _k in 0..k_size {
                    // Load 8 floats from B[k, j..j+8] (contiguous in row-major B)
                    let v0 = v128_load(src_col as *const v128);
                    let v1 = v128_load(src_col.add(4) as *const v128);

                    v128_store(dest as *mut v128, v0);
                    v128_store(dest.add(4) as *mut v128, v1);

                    dest = dest.add(OPT_NR);
                    src_col = src_col.add(ldb);
                }
            } else {
                // Edge case: pad with zeros
                for _k in 0..k_size {
                    for x in 0..n_remain {
                        *dest.add(x) = *src_col.add(x);
                    }
                    for x in n_remain..OPT_NR {
                        *dest.add(x) = 0.0;
                    }
                    dest = dest.add(OPT_NR);
                    src_col = src_col.add(ldb);
                }
            }
            j += OPT_NR;
        }
    }
}

/// Pack a 6×KC block of A into contiguous column-major layout.
///
/// Source: A[0..6, 0..KC] with row stride `lda` (typically = full K).
/// Dest layout: [a0[0],a1[0],a2[0],a3[0],a4[0],a5[0], a0[1],a1[1],...]
/// i.e. KC groups of MR=6 consecutive floats.
///
/// WHY PACK A: at power-of-2 strides (k=2048 → lda×4=8192=2^13), the 6 row
/// pointers a0..a5 all map to the same L1 cache set (32 KiB / 8-way → 4 KiB
/// set span; 8192 mod 4096 = 0). Every A load evicts the previous row's
/// line. Single-threaded this is a ~20% hit; with 8 parallel threads it's
/// 48 accesses competing for one 8-way set → parallel scaling collapses to
/// 1.0× at k=2048, 4096, etc. Packing A makes rows 24 bytes apart
/// (contiguous), breaking the aliasing regardless of the original stride.
///
/// Also: packed A is read sequentially in the K loop, so the prefetcher
/// can stream it. Unpacked A is 6 gather-ish pointers — harder to
/// prefetch.
///
/// Pack cost: MR × p_block = 6 × 256 = 1536 f32 reads + writes = 6 KiB.
/// Amortised over p_block × NC/NR = 256 × 16 = 4096 micro-kernel inner
/// iterations, i.e. < 1 f32-move per FMA. Negligible.
///
/// Safety: dest must have space for MR × p_block floats (or MR × p_block
/// + 2 floats if m_size < MR, due to the 2-float tail overwrite below —
/// callers allocate MR × OPT_KC which always suffices).
#[cfg(target_arch = "wasm32")]
#[inline]
unsafe fn pack_a_6xkc(
    a: *const f32,
    lda: usize,
    m_size: usize,   // how many of the 6 rows are real (≤ 6; rest zero-padded)
    p_block: usize,
    dest: *mut f32,
) {
    // CRITICAL: read A ROW-WISE (one full row at a time), not column-wise.
    //
    // The naïve column-wise pack (for kk: d[6k+r] = a_r[kk]) reads 6
    // elements per K-step at stride lda. When lda is a power of 2 (2048,
    // 4096…), those 6 addresses map to ONE L1 cache set. With 8 parallel
    // threads each doing this, it's 48 concurrent aliasing reads → total
    // serialisation. Benchmarked: K=2048 gave 1.00× parallel scaling.
    //
    // Row-wise pack: read row 0's p_block elements (contiguous!), scatter
    // to dest[r, r+MR, r+2×MR, …]. Then row 1. The READS are now
    // sequential (hardware prefetcher streams them); only the WRITES are
    // strided, and the write stride is MR=6 floats = 24 B — way below any
    // cache set span. No aliasing at any lda.
    //
    // Trade-off: writes now stride-6 instead of stride-1. But the write
    // target is the 6 KiB packed-A scratch, which stays in L1 for the
    // whole (ii, p) tile. First touch brings the line in for write;
    // subsequent writes to the same line hit. Effectively streaming.

    if m_size >= OPT_MR {
        // Full 6 rows. Copy each row's p_block elements with SIMD reads,
        // scatter-store to stride-MR layout.
        //
        // For each row r, dest addresses are r, r+6, r+12, … (stride 6).
        // Can't vectorise the store (no v128.scatter in WASM), but the
        // v128.load on A is the one that matters — it's the one that
        // would alias at pow-of-2 lda, and it's contiguous now.
        for r in 0..OPT_MR {
            let src = a.add(r * lda);
            let mut d = dest.add(r);

            // Read 4-at-a-time (sequential, cacheline-friendly), scatter.
            let k4 = p_block & !3;
            let mut kk = 0;
            while kk < k4 {
                let v = v128_load(src.add(kk) as *const v128);
                // Extract 4 lanes, store at stride MR. f32x4_extract_lane
                // is cheap (it's a shuffle + scalar store).
                *d                = f32x4_extract_lane::<0>(v);
                *d.add(OPT_MR)    = f32x4_extract_lane::<1>(v);
                *d.add(OPT_MR*2)  = f32x4_extract_lane::<2>(v);
                *d.add(OPT_MR*3)  = f32x4_extract_lane::<3>(v);
                d = d.add(OPT_MR * 4);
                kk += 4;
            }
            // Tail.
            while kk < p_block {
                *d = *src.add(kk);
                d = d.add(OPT_MR);
                kk += 1;
            }
        }
    } else {
        // Tail case (< 6 real rows). Zero the whole pack region first
        // (phantom rows contribute 0 in the micro-kernel, results for
        // those rows are discarded by the caller's partial store).
        let zero = f32x4_splat(0.0);
        let total = OPT_MR * p_block;
        let mut i = 0;
        while i + 4 <= total {
            v128_store(dest.add(i) as *mut v128, zero);
            i += 4;
        }
        while i < total {
            *dest.add(i) = 0.0;
            i += 1;
        }
        // Fill real rows, row-wise.
        for r in 0..m_size {
            let src = a.add(r * lda);
            let mut d = dest.add(r);
            let mut kk = 0;
            while kk < p_block {
                *d = *src.add(kk);
                d = d.add(OPT_MR);
                kk += 1;
            }
        }
    }
}

/// Optimized 6x8 micro-kernel using FMA and load32_splat, **packed A**.
///
/// Computes C[6x8] (+)= A_packed[6xK] · B_packed[Kx8]
///
/// A is pre-packed column-major (MR consecutive floats per K-step). This
/// is the cache-aliasing-safe version: A reads are sequential, stride=6,
/// immune to the power-of-2 pathology that wrecked the unpacked kernel at
/// k=2048/4096.
///
/// Register budget: 12 accumulators + 2 B + 1 A-splat = 15 v128 = fits
/// WASM's 16-register file.
#[cfg(target_arch = "wasm32")]
#[inline(always)]
unsafe fn micro_kernel_6x8_fma_pa(
    k_size: usize,
    a_packed: *const f32,   // [a0[k],a1[k],a2[k],a3[k],a4[k],a5[k]] × k_size
    b_packed: *const f32,   // [b[k,0..8]] × k_size
    c_ptr: *mut f32,
    ldc: usize,
    beta: f32,
) {
    let mut c00 = f32x4_splat(0.0); let mut c01 = f32x4_splat(0.0);
    let mut c10 = f32x4_splat(0.0); let mut c11 = f32x4_splat(0.0);
    let mut c20 = f32x4_splat(0.0); let mut c21 = f32x4_splat(0.0);
    let mut c30 = f32x4_splat(0.0); let mut c31 = f32x4_splat(0.0);
    let mut c40 = f32x4_splat(0.0); let mut c41 = f32x4_splat(0.0);
    let mut c50 = f32x4_splat(0.0); let mut c51 = f32x4_splat(0.0);

    let mut a_run = a_packed;
    let mut b_run = b_packed;

    // K loop. A and B are both contiguous now — two streaming pointers,
    // no strided gather.
    for _ in 0..k_size {
        let vb0 = v128_load(b_run as *const v128);
        let vb1 = v128_load(b_run.add(4) as *const v128);
        b_run = b_run.add(8);

        // load32_splat is a single instruction (v128.load32_splat) that
        // reads 4 bytes and broadcasts. Six of these, sequential addresses
        // a_run..a_run+6 → likely same cache line, definitely same page.
        let va0 = v128_load32_splat(a_run as *const u32);
        c00 = f32x4_relaxed_madd(va0, vb0, c00);
        c01 = f32x4_relaxed_madd(va0, vb1, c01);

        let va1 = v128_load32_splat(a_run.add(1) as *const u32);
        c10 = f32x4_relaxed_madd(va1, vb0, c10);
        c11 = f32x4_relaxed_madd(va1, vb1, c11);

        let va2 = v128_load32_splat(a_run.add(2) as *const u32);
        c20 = f32x4_relaxed_madd(va2, vb0, c20);
        c21 = f32x4_relaxed_madd(va2, vb1, c21);

        let va3 = v128_load32_splat(a_run.add(3) as *const u32);
        c30 = f32x4_relaxed_madd(va3, vb0, c30);
        c31 = f32x4_relaxed_madd(va3, vb1, c31);

        let va4 = v128_load32_splat(a_run.add(4) as *const u32);
        c40 = f32x4_relaxed_madd(va4, vb0, c40);
        c41 = f32x4_relaxed_madd(va4, vb1, c41);

        let va5 = v128_load32_splat(a_run.add(5) as *const u32);
        c50 = f32x4_relaxed_madd(va5, vb0, c50);
        c51 = f32x4_relaxed_madd(va5, vb1, c51);

        a_run = a_run.add(6);
    }

    let c0 = c_ptr;
    let c1 = c_ptr.add(ldc);
    let c2 = c_ptr.add(ldc * 2);
    let c3 = c_ptr.add(ldc * 3);
    let c4 = c_ptr.add(ldc * 4);
    let c5 = c_ptr.add(ldc * 5);

    if beta == 0.0 {
        v128_store(c0 as *mut v128, c00); v128_store(c0.add(4) as *mut v128, c01);
        v128_store(c1 as *mut v128, c10); v128_store(c1.add(4) as *mut v128, c11);
        v128_store(c2 as *mut v128, c20); v128_store(c2.add(4) as *mut v128, c21);
        v128_store(c3 as *mut v128, c30); v128_store(c3.add(4) as *mut v128, c31);
        v128_store(c4 as *mut v128, c40); v128_store(c4.add(4) as *mut v128, c41);
        v128_store(c5 as *mut v128, c50); v128_store(c5.add(4) as *mut v128, c51);
    } else {
        v128_store(c0 as *mut v128, f32x4_add(v128_load(c0 as *const v128), c00));
        v128_store(c0.add(4) as *mut v128, f32x4_add(v128_load(c0.add(4) as *const v128), c01));
        v128_store(c1 as *mut v128, f32x4_add(v128_load(c1 as *const v128), c10));
        v128_store(c1.add(4) as *mut v128, f32x4_add(v128_load(c1.add(4) as *const v128), c11));
        v128_store(c2 as *mut v128, f32x4_add(v128_load(c2 as *const v128), c20));
        v128_store(c2.add(4) as *mut v128, f32x4_add(v128_load(c2.add(4) as *const v128), c21));
        v128_store(c3 as *mut v128, f32x4_add(v128_load(c3 as *const v128), c30));
        v128_store(c3.add(4) as *mut v128, f32x4_add(v128_load(c3.add(4) as *const v128), c31));
        v128_store(c4 as *mut v128, f32x4_add(v128_load(c4 as *const v128), c40));
        v128_store(c4.add(4) as *mut v128, f32x4_add(v128_load(c4.add(4) as *const v128), c41));
        v128_store(c5 as *mut v128, f32x4_add(v128_load(c5 as *const v128), c50));
        v128_store(c5.add(4) as *mut v128, f32x4_add(v128_load(c5.add(4) as *const v128), c51));
    }
}

/// Optimized 6x8 micro-kernel using FMA and loadsplat
///
/// Computes C[6x8] += A[6xK] * B_packed[Kx8]
///
/// Uses 12 accumulator registers (fits in 16 XMM), 2 for B, 1 for A splat
#[cfg(target_arch = "wasm32")]
#[inline(always)]
pub unsafe fn micro_kernel_6x8_fma(
    k_size: usize,
    a_ptr: *const f32,
    lda: usize,
    b_packed: *const f32,
    c_ptr: *mut f32,
    ldc: usize,
    beta: f32,  // 0.0 = overwrite, 1.0 = accumulate
) {
    // Setup A row pointers
    let a0 = a_ptr;
    let a1 = a_ptr.add(lda);
    let a2 = a_ptr.add(lda * 2);
    let a3 = a_ptr.add(lda * 3);
    let a4 = a_ptr.add(lda * 4);
    let a5 = a_ptr.add(lda * 5);

    // 12 accumulators: 6 rows × 2 vectors (8 cols)
    let mut c00 = f32x4_splat(0.0);
    let mut c01 = f32x4_splat(0.0);
    let mut c10 = f32x4_splat(0.0);
    let mut c11 = f32x4_splat(0.0);
    let mut c20 = f32x4_splat(0.0);
    let mut c21 = f32x4_splat(0.0);
    let mut c30 = f32x4_splat(0.0);
    let mut c31 = f32x4_splat(0.0);
    let mut c40 = f32x4_splat(0.0);
    let mut c41 = f32x4_splat(0.0);
    let mut c50 = f32x4_splat(0.0);
    let mut c51 = f32x4_splat(0.0);

    let mut b_run = b_packed;

    // K loop - single iteration per K value
    for kk in 0..k_size {
        // Load B: 8 columns = 2 vectors (contiguous in packed format)
        let vb0 = v128_load(b_run as *const v128);
        let vb1 = v128_load(b_run.add(4) as *const v128);
        b_run = b_run.add(8);

        // Row 0: loadsplat + FMA
        let va0 = v128_load32_splat(a0.add(kk) as *const u32);
        c00 = f32x4_relaxed_madd(va0, vb0, c00);
        c01 = f32x4_relaxed_madd(va0, vb1, c01);

        // Row 1
        let va1 = v128_load32_splat(a1.add(kk) as *const u32);
        c10 = f32x4_relaxed_madd(va1, vb0, c10);
        c11 = f32x4_relaxed_madd(va1, vb1, c11);

        // Row 2
        let va2 = v128_load32_splat(a2.add(kk) as *const u32);
        c20 = f32x4_relaxed_madd(va2, vb0, c20);
        c21 = f32x4_relaxed_madd(va2, vb1, c21);

        // Row 3
        let va3 = v128_load32_splat(a3.add(kk) as *const u32);
        c30 = f32x4_relaxed_madd(va3, vb0, c30);
        c31 = f32x4_relaxed_madd(va3, vb1, c31);

        // Row 4
        let va4 = v128_load32_splat(a4.add(kk) as *const u32);
        c40 = f32x4_relaxed_madd(va4, vb0, c40);
        c41 = f32x4_relaxed_madd(va4, vb1, c41);

        // Row 5
        let va5 = v128_load32_splat(a5.add(kk) as *const u32);
        c50 = f32x4_relaxed_madd(va5, vb0, c50);
        c51 = f32x4_relaxed_madd(va5, vb1, c51);
    }

    // Store results
    let c0 = c_ptr;
    let c1 = c_ptr.add(ldc);
    let c2 = c_ptr.add(ldc * 2);
    let c3 = c_ptr.add(ldc * 3);
    let c4 = c_ptr.add(ldc * 4);
    let c5 = c_ptr.add(ldc * 5);

    if beta == 0.0 {
        // Overwrite
        v128_store(c0 as *mut v128, c00);
        v128_store(c0.add(4) as *mut v128, c01);
        v128_store(c1 as *mut v128, c10);
        v128_store(c1.add(4) as *mut v128, c11);
        v128_store(c2 as *mut v128, c20);
        v128_store(c2.add(4) as *mut v128, c21);
        v128_store(c3 as *mut v128, c30);
        v128_store(c3.add(4) as *mut v128, c31);
        v128_store(c4 as *mut v128, c40);
        v128_store(c4.add(4) as *mut v128, c41);
        v128_store(c5 as *mut v128, c50);
        v128_store(c5.add(4) as *mut v128, c51);
    } else {
        // Accumulate (beta = 1.0)
        v128_store(c0 as *mut v128, f32x4_add(v128_load(c0 as *const v128), c00));
        v128_store(c0.add(4) as *mut v128, f32x4_add(v128_load(c0.add(4) as *const v128), c01));
        v128_store(c1 as *mut v128, f32x4_add(v128_load(c1 as *const v128), c10));
        v128_store(c1.add(4) as *mut v128, f32x4_add(v128_load(c1.add(4) as *const v128), c11));
        v128_store(c2 as *mut v128, f32x4_add(v128_load(c2 as *const v128), c20));
        v128_store(c2.add(4) as *mut v128, f32x4_add(v128_load(c2.add(4) as *const v128), c21));
        v128_store(c3 as *mut v128, f32x4_add(v128_load(c3 as *const v128), c30));
        v128_store(c3.add(4) as *mut v128, f32x4_add(v128_load(c3.add(4) as *const v128), c31));
        v128_store(c4 as *mut v128, f32x4_add(v128_load(c4 as *const v128), c40));
        v128_store(c4.add(4) as *mut v128, f32x4_add(v128_load(c4.add(4) as *const v128), c41));
        v128_store(c5 as *mut v128, f32x4_add(v128_load(c5 as *const v128), c50));
        v128_store(c5.add(4) as *mut v128, f32x4_add(v128_load(c5.add(4) as *const v128), c51));
    }
}

/// Handle edge cases where M < 6 or N < 8
#[cfg(target_arch = "wasm32")]
unsafe fn micro_kernel_edge(
    m_rem: usize,
    n_rem: usize,
    k: usize,
    a: *const f32,
    lda: usize,
    b_packed: *const f32,
    c: *mut f32,
    ldc: usize,
    beta: f32,
) {
    // Use a temp buffer for the full 6x8 tile
    let mut tmp_c = [0.0f32; OPT_MR * OPT_NR];
    let tmp_ldc = OPT_NR;

    // Load existing C if accumulating
    if beta != 0.0 {
        for r in 0..m_rem {
            for col in 0..n_rem.min(OPT_NR) {
                tmp_c[r * tmp_ldc + col] = *c.add(r * ldc + col);
            }
        }
    }

    // Run full kernel on temp buffer
    micro_kernel_6x8_fma(k, a, lda, b_packed, tmp_c.as_mut_ptr(), tmp_ldc, 0.0);

    // Copy valid results back
    for r in 0..m_rem {
        for col in 0..n_rem.min(OPT_NR) {
            if beta == 0.0 {
                *c.add(r * ldc + col) = tmp_c[r * tmp_ldc + col];
            } else {
                *c.add(r * ldc + col) += tmp_c[r * tmp_ldc + col];
            }
        }
    }
}

/// Cache-blocked GEMM dispatcher using optimized 6x8 micro-kernel
///
/// C = A * B where A is [m, k], B is [k, n], C is [m, n]
#[cfg(target_arch = "wasm32")]
pub fn matmul_optimized_f32(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];

    // Allocate packing buffer for B (KC x NC)
    let pack_size = OPT_KC * ((OPT_NC + OPT_NR - 1) / OPT_NR) * OPT_NR;
    let mut packed_b = vec![0.0f32; pack_size];

    unsafe {
        // Loop over N in blocks of NC
        let mut j = 0;
        while j < n {
            let j_block = (n - j).min(OPT_NC);
            let j_panels = (j_block + OPT_NR - 1) / OPT_NR;

            // Loop over K in blocks of KC
            let mut p = 0;
            while p < k {
                let p_block = (k - p).min(OPT_KC);

                // Pack B panel: B[p..p+p_block, j..j+j_block]
                pack_b_optimized(
                    b.as_ptr().add(p * n + j),
                    n,
                    packed_b.as_mut_ptr(),
                    p_block,
                    j_block,
                );

                // Beta = 0.0 for first K block, 1.0 for subsequent (accumulate)
                let beta = if p == 0 { 0.0 } else { 1.0 };

                // Loop over M in blocks of MC
                let mut i = 0;
                while i < m {
                    let i_block = (m - i).min(OPT_MC);
                    let i_main = i_block / OPT_MR * OPT_MR;

                    // Process full MR×NR tiles
                    let mut ii = 0;
                    while ii < i_main {
                        let mut jj = 0;
                        while jj < j_panels * OPT_NR && jj < j_block {
                            let panel_idx = jj / OPT_NR;
                            let b_panel_ptr = packed_b.as_ptr().add(panel_idx * p_block * OPT_NR);
                            let n_rem = j_block - jj;

                            if n_rem >= OPT_NR {
                                micro_kernel_6x8_fma(
                                    p_block,
                                    a.as_ptr().add((i + ii) * k + p),
                                    k,
                                    b_panel_ptr,
                                    c.as_mut_ptr().add((i + ii) * n + j + jj),
                                    n,
                                    beta,
                                );
                            } else {
                                micro_kernel_edge(
                                    OPT_MR,
                                    n_rem,
                                    p_block,
                                    a.as_ptr().add((i + ii) * k + p),
                                    k,
                                    b_panel_ptr,
                                    c.as_mut_ptr().add((i + ii) * n + j + jj),
                                    n,
                                    beta,
                                );
                            }
                            jj += OPT_NR;
                        }
                        ii += OPT_MR;
                    }

                    // Handle remaining rows (i_main..i_block)
                    if ii < i_block {
                        let m_rem = i_block - ii;
                        let mut jj = 0;
                        while jj < j_panels * OPT_NR && jj < j_block {
                            let panel_idx = jj / OPT_NR;
                            let b_panel_ptr = packed_b.as_ptr().add(panel_idx * p_block * OPT_NR);
                            let n_rem = (j_block - jj).min(OPT_NR);

                            micro_kernel_edge(
                                m_rem,
                                n_rem,
                                p_block,
                                a.as_ptr().add((i + ii) * k + p),
                                k,
                                b_panel_ptr,
                                c.as_mut_ptr().add((i + ii) * n + j + jj),
                                n,
                                beta,
                            );
                            jj += OPT_NR;
                        }
                    }

                    i += OPT_MC;
                }

                p += OPT_KC;
            }

            j += OPT_NC;
        }
    }

    c
}

/// Parallel version of optimized GEMM using rayon (LEGACY - slow)
///
/// Known issues (see matmul_optimized_f32_parallel_v3 for the fix):
///   - Each thread re-packs B independently (N× redundant work)
///   - Each thread allocates its own Vec<f32> for C and packed_b
///     (WASM's dlmalloc is globally-locked, so this serialises threads)
///   - Extra copy_from_slice at the end
///   - 1D partitioning only; no cache reuse of packed B across threads
#[cfg(target_arch = "wasm32")]
pub fn matmul_optimized_f32_parallel(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    use rayon::prelude::*;

    // For small matrices, single-threaded is faster
    // u64 mul: WASM usize is 32-bit, m*n*k overflows silently at 2048³.
    if (m as u64) * (n as u64) * (k as u64) < (64u64 * 64 * 64) {
        return matmul_optimized_f32(a, b, m, n, k);
    }

    let mut c = vec![0.0f32; m * n];

    // Split by rows - each thread gets a chunk of rows
    let num_threads = rayon::current_num_threads();
    let rows_per_thread = (m + num_threads - 1) / num_threads;

    c.par_chunks_mut(rows_per_thread * n)
        .enumerate()
        .for_each(|(chunk_idx, c_chunk)| {
            let start_row = chunk_idx * rows_per_thread;
            let local_m = c_chunk.len() / n;
            if local_m == 0 { return; }

            let a_slice = &a[start_row * k..(start_row + local_m) * k];

            // Use optimized kernel for this chunk
            let local_c = matmul_optimized_f32(a_slice, b, local_m, n, k);
            c_chunk.copy_from_slice(&local_c);
        });

    c
}

// ============================================================================
// Parallel GEMM v3: single-dispatch, per-thread packing buffers
// ============================================================================
//
// The legacy `matmul_optimized_f32_parallel` is 1.5–5× slower than tf.js
// multi-threaded. The root causes are NOT architectural — they're
// embarrassingly mechanical:
//
//   1. Each thread calls `matmul_optimized_f32`, which allocates 2 fresh
//      Vec<f32>s (one for C, one for packed-B). WASM's dlmalloc is globally
//      locked, so N parallel allocs serialise.
//
//   2. Each thread's result gets `copy_from_slice`'d back into the real C.
//      Redundant M×N/N_threads stores per thread.
//
//   3. `par_chunks_mut` gives 1 big slab per thread. Zero load balancing;
//      on Apple Silicon an efficiency core stalls the whole join.
//
// The fix here is deliberately unsexy: give each thread a **pre-allocated**
// packing scratch buffer, have it run the full BLIS (j, p, i) loop over its
// assigned M-row slab, and write directly to C. That's it. No cross-thread
// barriers, no shared packed-B (the packing cost per thread is identical
// to single-threaded: O(K×N), the same reads/writes, just split across
// threads in time), and — critically — **ONE rayon dispatch per matmul**.
//
// Why not share packed-B across threads?  Because doing so requires a
// barrier per (j, p) block (main packs, workers wait, workers compute,
// main waits, repeat). For 1024² that's 32 barrier round-trips. On WASM
// each barrier is a park/unpark (or spin) cycle of ~10-100μs → 0.3–3 ms
// of pure sync overhead. The "redundant" per-thread packing, by contrast,
// is ~K×N/N_threads ≈ 75K stores per thread for 1024²/14, i.e. ~75 μs.
// Packing wins by a mile.
//
// The one architectural upgrade vs legacy: tiles not slabs. We hand out
// MC-row tiles via an atomic counter so heterogeneous cores self-balance.

#[cfg(target_arch = "wasm32")]
use std::sync::atomic::{AtomicI32, AtomicU32, AtomicUsize, Ordering};

/// Pack ALL of B into XNNPACK panel-major layout.
///
/// Output layout: for each NR-column panel (0..ceil(N/NR)), emit K rows
/// of NR floats contiguously. Panel p occupies [p × K × NR, (p+1) × K × NR).
///
/// The last panel is zero-padded to NR width if N % NR != 0. This lets
/// the micro-kernel always do full-width v128 stores; the caller just
/// discards the padded columns from C.
///
/// This is the "pre-pack once, matmul many" layout that tf.js/XNNPACK
/// uses for weight matrices. We do it per-matmul here (B varies), but
/// ONCE per call instead of once per slab — with 8 parallel slabs at
/// 256² that's 8× less packing work.
///
/// Safety: dest must have space for ceil(N/NR) × K × NR floats.
#[cfg(target_arch = "wasm32")]
#[inline(never)]  // big serial loop, inlining bloats callers
pub unsafe fn pack_b_full_xnnpack(
    b: *const f32,
    ldb: usize,     // = N (B's row stride)
    dest: *mut f32,
    k: usize,
    n: usize,
) {
    let n_full_panels = n / OPT_NR;
    let n_tail = n % OPT_NR;

    let mut d = dest;

    // Full NR-wide panels: pure SIMD copy.
    for panel in 0..n_full_panels {
        let j0 = panel * OPT_NR;
        let mut src = b.add(j0);
        // For each K-row, copy NR=8 contiguous floats.
        // This is 2 v128 loads + 2 v128 stores per row — the read side
        // is sequential within each row (stride 1), write side is
        // fully streaming. Hardware prefetchers handle both.
        for _ in 0..k {
            let v0 = v128_load(src as *const v128);
            let v1 = v128_load(src.add(4) as *const v128);
            v128_store(d as *mut v128, v0);
            v128_store(d.add(4) as *mut v128, v1);
            src = src.add(ldb);
            d = d.add(OPT_NR);
        }
    }

    // Tail panel: scalar copy for valid cols, zero-pad the rest.
    // Rare (only when N % 8 != 0) so not worth SIMD-ing.
    if n_tail > 0 {
        let j0 = n_full_panels * OPT_NR;
        let mut src = b.add(j0);
        let zero = f32x4_splat(0.0);
        for _ in 0..k {
            // Zero all 8 slots first, then overwrite real cols.
            v128_store(d as *mut v128, zero);
            v128_store(d.add(4) as *mut v128, zero);
            for c in 0..n_tail {
                *d.add(c) = *src.add(c);
            }
            src = src.add(ldb);
            d = d.add(OPT_NR);
        }
    }
}

/// GEMM over an M-slab using pre-packed B.
///
/// B is already in panel-major layout (from pack_b_full_xnnpack), so
/// there's NO B-packing inside this loop — just a pointer offset per
/// (j, p) block. The entire per-slab cost is micro-kernel calls.
///
/// Still does KC-blocking on the K dimension: reading K floats of A per
/// micro-kernel call would blow L1, so we process K in chunks of KC=256.
/// Between KC-blocks, C is accumulated (beta=1 after the first block).
///
/// lda = k (A's natural stride), ldc = n (C's natural stride). No
/// pow-of-2 padding here — if you hit that, the pow2 path in v3 handles
/// it separately.
///
/// Safety: packed_b must cover ceil(n/NR) × k × NR floats in the layout
/// produced by pack_b_full_xnnpack; c must have space for (m_start+m_size)
/// rows × n cols.
///
/// PUBLIC: also used by matmulF32Prepacked — the persistent-B API where
/// JS calls packBFull() once and reuses the result for many matmuls
/// (NN inference: B = constant weights).
#[cfg(target_arch = "wasm32")]
pub unsafe fn matmul_slab_prepackedb(
    a: *const f32,
    packed_b: *const f32,
    c: *mut f32,
    m_start: usize,
    m_size: usize,
    n: usize,
    k: usize,
) {
    let a = a.add(m_start * k);
    let c = c.add(m_start * n);

    let n_panels = (n + OPT_NR - 1) / OPT_NR;
    let i_main = m_size / OPT_MR * OPT_MR;

    // K-loop outermost so each KC-block's worth of B stays hot in L1
    // across all (ii, jj) tiles before advancing.
    let mut p = 0;
    while p < k {
        let p_block = (k - p).min(OPT_KC);
        let beta = if p == 0 { 0.0 } else { 1.0 };

        // jj outer, ii inner: processing one NR-panel's K-block across
        // all M-rows before moving to the next panel keeps THAT panel's
        // floats in L1 for the whole M sweep. Same iteration order as
        // matmul_optimized_f32's (j, p, i) — just without the pack.
        for panel in 0..n_panels {
            // Packed B for this panel, this K-block, is at:
            //   packed_b + panel × k × NR + p × NR
            // (panel stride = k × NR, within-panel stride per K = NR)
            let pb = packed_b.add(panel * k * OPT_NR + p * OPT_NR);
            let jj = panel * OPT_NR;
            let n_rem = n - jj;

            let mut ii = 0;
            while ii < i_main {
                if n_rem >= OPT_NR {
                    micro_kernel_6x8_fma(
                        p_block,
                        a.add(ii * k + p), k,
                        pb,
                        c.add(ii * n + jj), n,
                        beta,
                    );
                } else {
                    micro_kernel_edge(
                        OPT_MR, n_rem, p_block,
                        a.add(ii * k + p), k,
                        pb,
                        c.add(ii * n + jj), n,
                        beta,
                    );
                }
                ii += OPT_MR;
            }
            // M tail
            if ii < m_size {
                let m_rem = m_size - ii;
                micro_kernel_edge(
                    m_rem, n_rem.min(OPT_NR), p_block,
                    a.add(ii * k + p), k,
                    pb,
                    c.add(ii * n + jj), n,
                    beta,
                );
            }
        }

        p += OPT_KC;
    }
}

/// Run the BLIS GEMM over a 2D rectangle of C, packing BOTH A and B.
///
/// Computes `C[m_start..m_start+m_size, n_start..n_start+n_size] +=
/// A[m_start..][..k] · B[..k][n_start..n_start+n_size]`.
///
/// Scratch layout (caller provides one contiguous buffer per thread):
///   [0 .. KC × NC)          → packed B panel (one (NC, KC) block at a time)
///   [KC × NC .. KC × NC + MR × KC)  → packed A panel (one MR × KC strip)
/// The A panel is repacked per (ii, p) step — tiny (6 × 256 = 1536 f32 =
/// 6 KiB), amortised over the full NR-loop.
///
/// WHY PACK BOTH:
/// * B-packing (existing): sequential inner-loop access, NR-panel format.
/// * A-packing (NEW): fixes the power-of-2 cache-aliasing catastrophe.
///   At k=2048 (stride 8192 B), unpacked A's 6 row pointers all map to the
///   same L1 set → with 8 parallel threads, 48 competing accesses → scaling
///   collapses to 1.0×.  Packed A has rows 24 B apart (contiguous) — no
///   aliasing at any k.  Measured: 2048² went 0.99× → expected ~5.8×.
///
/// WHY 2D SLABS: each thread reads only K × n_size of B, not K × N.
/// At large sizes this is a second-order effect (the aliasing fix is the
/// dominant term), but it does cut DRAM pressure by the N-tile factor.
#[cfg(target_arch = "wasm32")]
unsafe fn matmul_optimized_f32_slab(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m_start: usize,
    m_size: usize,
    n_start: usize,
    n_size: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
    k: usize,
    pack_a: bool,      // enable A-packing (for pow-of-2 K strides)
    scratch: *mut f32, // ≥ KC × NC (+ MR × KC if pack_a) floats
) {
    let a = a.add(m_start * lda);
    let c = c.add(m_start * ldc + n_start);
    let b = b.add(n_start);

    let packed_b = scratch;
    // packed_a only used when pack_a is true; sits after the B panel.
    let packed_a = scratch.add(OPT_KC * ((OPT_NC + OPT_NR - 1) / OPT_NR) * OPT_NR);

    let mut j = 0;
    while j < n_size {
        let j_block = (n_size - j).min(OPT_NC);
        let j_panels = (j_block + OPT_NR - 1) / OPT_NR;

        let mut p = 0;
        while p < k {
            let p_block = (k - p).min(OPT_KC);

            pack_b_optimized(
                b.add(p * ldb + j),
                ldb,
                packed_b,
                p_block,
                j_block,
            );

            let beta = if p == 0 { 0.0 } else { 1.0 };
            let i_main = m_size / OPT_MR * OPT_MR;

            // Two code paths: packed-A (for pow-of-2 K — the micro-kernel
            // reads contiguous A, no stride aliasing) vs direct-A (fast
            // path for everything else — skips the ~1.5K-float A-pack per
            // (ii, p) step, which adds up to ~7 ms at 1024²).
            //
            // pack_a decided ONCE by the caller based on K's pow-of-2-ness.
            // We duplicate the ii/jj loops so the branch is hoisted out of
            // the hot path and each arm inlines its micro-kernel cleanly.
            if pack_a {
                let mut ii = 0;
                while ii < i_main {
                    pack_a_6xkc(a.add(ii * lda + p), lda, OPT_MR, p_block, packed_a);
                    let mut jj = 0;
                    while jj < j_panels * OPT_NR && jj < j_block {
                        let panel = jj / OPT_NR;
                        let pb = packed_b.add(panel * p_block * OPT_NR);
                        let n_rem = j_block - jj;
                        if n_rem >= OPT_NR {
                            micro_kernel_6x8_fma_pa(
                                p_block, packed_a, pb,
                                c.add(ii * ldc + j + jj), ldc, beta,
                            );
                        } else {
                            let mut tmp = [0.0f32; OPT_MR * OPT_NR];
                            if beta != 0.0 {
                                for r in 0..OPT_MR {
                                    for cc in 0..n_rem {
                                        tmp[r * OPT_NR + cc] = *c.add((ii + r) * ldc + j + jj + cc);
                                    }
                                }
                            }
                            micro_kernel_6x8_fma_pa(p_block, packed_a, pb, tmp.as_mut_ptr(), OPT_NR, beta);
                            for r in 0..OPT_MR {
                                for cc in 0..n_rem {
                                    *c.add((ii + r) * ldc + j + jj + cc) = tmp[r * OPT_NR + cc];
                                }
                            }
                        }
                        jj += OPT_NR;
                    }
                    ii += OPT_MR;
                }
                if ii < m_size {
                    let m_rem = m_size - ii;
                    pack_a_6xkc(a.add(ii * lda + p), lda, m_rem, p_block, packed_a);
                    let mut jj = 0;
                    while jj < j_panels * OPT_NR && jj < j_block {
                        let panel = jj / OPT_NR;
                        let pb = packed_b.add(panel * p_block * OPT_NR);
                        let n_rem = (j_block - jj).min(OPT_NR);
                        let mut tmp = [0.0f32; OPT_MR * OPT_NR];
                        if beta != 0.0 {
                            for r in 0..m_rem {
                                for cc in 0..n_rem {
                                    tmp[r * OPT_NR + cc] = *c.add((ii + r) * ldc + j + jj + cc);
                                }
                            }
                        }
                        micro_kernel_6x8_fma_pa(p_block, packed_a, pb, tmp.as_mut_ptr(), OPT_NR, beta);
                        for r in 0..m_rem {
                            for cc in 0..n_rem {
                                *c.add((ii + r) * ldc + j + jj + cc) = tmp[r * OPT_NR + cc];
                            }
                        }
                        jj += OPT_NR;
                    }
                }
            } else {
                // Fast path: micro-kernel reads A directly (strided). Safe
                // when lda × 4 is not a pathological power-of-2 multiple.
                // Identical to the single-threaded matmul_optimized_f32's
                // inner loop — we just skip the allocation.
                let mut ii = 0;
                while ii < i_main {
                    let mut jj = 0;
                    while jj < j_panels * OPT_NR && jj < j_block {
                        let panel = jj / OPT_NR;
                        let pb = packed_b.add(panel * p_block * OPT_NR);
                        let n_rem = j_block - jj;
                        if n_rem >= OPT_NR {
                            micro_kernel_6x8_fma(
                                p_block,
                                a.add(ii * lda + p), lda,
                                pb,
                                c.add(ii * ldc + j + jj), ldc,
                                beta,
                            );
                        } else {
                            micro_kernel_edge(
                                OPT_MR, n_rem, p_block,
                                a.add(ii * lda + p), lda,
                                pb,
                                c.add(ii * ldc + j + jj), ldc,
                                beta,
                            );
                        }
                        jj += OPT_NR;
                    }
                    ii += OPT_MR;
                }
                if ii < m_size {
                    let m_rem = m_size - ii;
                    let mut jj = 0;
                    while jj < j_panels * OPT_NR && jj < j_block {
                        let panel = jj / OPT_NR;
                        let pb = packed_b.add(panel * p_block * OPT_NR);
                        let n_rem = (j_block - jj).min(OPT_NR);
                        micro_kernel_edge(
                            m_rem, n_rem, p_block,
                            a.add(ii * lda + p), lda,
                            pb,
                            c.add(ii * ldc + j + jj), ldc,
                            beta,
                        );
                        jj += OPT_NR;
                    }
                }
            }

            p += OPT_KC;
        }
        j += OPT_NC;
    }
}

/// Parallel optimised GEMM, v3.
///
/// Single dispatch, per-thread scratch, 2D (m_slab, n_slab) tiles.
///
/// EVOLUTION:
///   rev-1: MC-sized tiles → each tile re-packs B → 15 packings at 1024²
///          vs v1's 8 → LOST to v1. Dumb.
///   rev-2: thread-count-sized M-slabs → matches v1's packing count, wins
///          marginally on alloc-contention fix. BUT hits the memory
///          bandwidth wall at 2048²: every thread reads ALL of B (16 MiB),
///          8 threads × 16 MiB = 128 MiB aggregate → threads serialise on
///          DRAM. 0× scaling at 2048².
///   rev-3 (this): 2D tiles. Each thread works on (m_slab, n_slab), reads
///          only K × n_slab of B. Aggregate B bandwidth = N_threads ×
///          K × (N / n_tiles) ≈ K × N = one pass through B regardless of
///          thread count. This is why XNNPACK uses parallelize_2d_tile_2d.
///
/// Tile-grid heuristic: we want ~2 tiles per thread for stealing, and we
/// want N-tiles >> 1 when B is large (to cut bandwidth). Square-ish grid
/// works: pick m_tiles × n_tiles ≈ 2 × n_participants with m_tiles ≥ n_tiles
/// (M-parallelism is cheaper: each M-tile reads the SAME N-slice of B,
/// so it may hit in L2/L3 from a sibling; N-tiles read disjoint B).
#[cfg(target_arch = "wasm32")]
pub fn matmul_optimized_f32_parallel_v3(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    // WARNING: WASM usize is 32-bit. `m * n * k` at 2048³ = 8.6e9 overflows
    // to 0 and silently routed to the single-threaded fallback — looked
    // EXACTLY like a cache-conflict collapse in benchmarks (1.0× scaling,
    // workers confirmed running on distinct threads via probe, any other
    // dimension combination fine). Took ~6 hours of phantom-chasing to
    // find. ALWAYS use u64 for flop estimates in WASM.
    //
    // Threshold: 192³ ≈ 7M flops. Below this, rayon dispatch overhead
    // (~100-500 μs depending on worker warmth) dominates the ~100 μs
    // of parallel compute. 128³ was just at the break-even and lost.
    let flops = (m as u64) * (n as u64) * (k as u64);
    if flops < (192u64 * 192 * 192) {
        return matmul_optimized_f32(a, b, m, n, k);
    }

    // Participants = rayon workers + caller. Caller always computes.
    let n_workers = rayon::current_num_threads().max(1);
    let n_parts = n_workers + 1;
    if n_workers <= 0 {
        return matmul_optimized_f32(a, b, m, n, k);
    }

    // Padding decisions come FIRST so the fast-path short-circuit can
    // branch on them.
    //
    // HISTORICAL NOTE: A-packing and C-stride padding were added while
    // chasing a "pow-of-2 collapse" that turned out to be the 32-bit
    // `m * n * k` overflow at 2048³ (= 8.6e9 wraps to 0 → silent fallback
    // to single-threaded). Once fixed with u64, the pow-of-2 sizes scale
    // fine WITHOUT these mitigations (v1 gets 6.3× at 2048³).
    //
    // We keep A-packing and C-padding gated but with a HIGH threshold
    // (only K or N ≥ 4096 AND divisible by 4096 triggers) — there's a
    // theoretical cache-set-aliasing risk at very large pow2 strides that
    // we haven't yet hit in practice but is cheap to defend against.
    // At the typical ML sizes (512-4096) the fast path is always taken.
    const PAD_ZEROS_THRESHOLD: u32 = 14;  // K or N divisible by 4096
    let pack_a_enabled = (k * 4).trailing_zeros() >= PAD_ZEROS_THRESHOLD;
    let n_stride = if (n * 4).trailing_zeros() >= PAD_ZEROS_THRESHOLD {
        n + OPT_NR
    } else {
        n
    };
    let c_padded = n_stride != n;

    // 1D M-only slabs. Exactly n_workers slabs (one per worker) minimises
    // B-packing overhead (each slab packs B once for its whole row range);
    // the caller still participates via the atomic counter if a worker is
    // slow to start, but typically just steals the last slab.
    //
    // WHY NOT MORE SLABS (2× over-partition): each slab packs B over the
    // full (k/KC × n/NC) block schedule. 2× slabs → 2× B-packing. At small
    // sizes this dominates; at large sizes the load-balancing benefit
    // doesn't justify it on uniform x86 cores.
    //
    // WHY NOT 2D TILES: each 2D (m_tile, n_tile) re-packs B for its n_size
    // — with 20 tiles you get 5× the B-packing. Measured 18% slower at
    // 1024². The "2D reduces bandwidth" hypothesis was wrong; A-packing
    // (inside the slab) is what actually fixes pow-of-2 K.
    let slab_rows = {
        let base = (m + n_workers - 1) / n_workers;
        ((base + OPT_MR - 1) / OPT_MR * OPT_MR).max(OPT_MR)
    };
    let total_tiles = (m + slab_rows - 1) / slab_rows;

    if total_tiles < 2 {
        return matmul_optimized_f32(a, b, m, n, k);
    }

    // ========================================================================
    // FAST PATH (non-pow2 K and N) — shared packed-B, single dispatch
    // ========================================================================
    //
    // At small sizes (256-512) the per-slab B-packing was the dominant
    // cost: 8 slabs × full-K×full-N pack = 8× redundant work. For 256²
    // that's ~0.4 ms of packing for ~0.05 ms of useful compute.
    //
    // Fix: pack B ONCE here (serial, full K × N in panel-major layout),
    // then workers read from the shared packed buffer.  Each worker's
    // inner loop is then pure micro-kernel calls — no B-copy at all.
    //
    // Pack layout: XNNPACK-style, one NR-column panel at a time, K rows
    // contiguous within each panel.  Panel p (cols p×NR .. (p+1)×NR)
    // occupies packed_b[p × K × NR .. (p+1) × K × NR].
    //
    // The K-loop still does KC-blocking for L1-residency but now
    // B is already contiguous, so "packing" inside the K-block is just
    // a pointer offset into the pre-packed buffer.
    if !pack_a_enabled && !c_padded {
        use rayon::prelude::*;

        // Two strategies for B, picked by whether packed-B fits in L2:
        //
        //   SHARED PACKED-B (small K×N): pack B once serially, workers
        //   read from the shared panel-major buffer. Each worker's B
        //   reads hit L2 (everyone reads the same cache lines). At 256²
        //   this beat tf.js by 37% — the 8× packing redundancy of
        //   per-slab B was the entire gap.
        //
        //   PER-SLAB PACKED-B (large K×N): each worker packs its own
        //   KC×NC sub-panel into L1-resident scratch. The full packed-B
        //   wouldn't fit in L2 anyway, so sharing doesn't help — and the
        //   8× redundant packing is O(K×N) which is o(K×N×M_slab) compute.
        //   Measured: per-slab wins by ~20% at 1024²+.
        //
        // Crossover at packed-B ≈ 512 KiB (half a typical L2 — leave room
        // for A and C lines).  512 KiB = 128 K floats.  k × n ≤ 131072.
        // For square problems that's n ≈ 360.
        const L2_BUDGET_FLOATS: usize = 128 * 1024;
        let b_fits_l2 = k * n <= L2_BUDGET_FLOATS;

        if b_fits_l2 {
            // SHARED PACKED-B path.
            let n_panels = (n + OPT_NR - 1) / OPT_NR;
            let pb_size = n_panels * k * OPT_NR;
            // Uninitialised: pack_b_full_xnnpack overwrites everything.
            let mut packed_b: Vec<f32> = Vec::with_capacity(pb_size);
            unsafe { packed_b.set_len(pb_size); }

            // Pack serially. ONE pass through B.  At 256² this is 64 K
            // f32 = 256 KiB ≈ 0.05 ms. Once, not 8×.
            unsafe {
                pack_b_full_xnnpack(b.as_ptr(), n, packed_b.as_mut_ptr(), k, n);
            }

            let a_addr = a.as_ptr() as usize;
            let pb_addr = packed_b.as_ptr() as usize;
            let mut c = vec![0.0f32; m * n];
            let c_addr = c.as_mut_ptr() as usize;

            c.par_chunks_mut(slab_rows * n)
                .enumerate()
                .for_each(|(slab_idx, _chunk)| {
                    let m_start = slab_idx * slab_rows;
                    let m_size = (m - m_start).min(slab_rows);
                    if m_size == 0 { return; }
                    unsafe {
                        matmul_slab_prepackedb(
                            a_addr as *const f32,
                            pb_addr as *const f32,
                            c_addr as *mut f32,
                            m_start, m_size,
                            n, k,
                        );
                    }
                });

            drop(packed_b);
            return c;
        }

        // PER-SLAB PACKED-B path: each worker packs its own KC×NC panel
        // into L1-resident scratch.  matmul_optimized_f32_slab does
        // exactly this internally (pack_a=false → only B-packing).
        //
        // Uninitialised scratch (pack_b_optimized writes before reading).
        // `mut` + as_mut_ptr is REQUIRED for provenance — LLVM elided
        // writes through (as_ptr() as usize as *mut) giving ~8× slowdown.
        let pb_region = OPT_KC * ((OPT_NC + OPT_NR - 1) / OPT_NR) * OPT_NR;
        let scratch_stride_fast = pb_region + 17;
        let mut scratch_arena: Vec<f32> = Vec::with_capacity(scratch_stride_fast * total_tiles);
        unsafe { scratch_arena.set_len(scratch_stride_fast * total_tiles); }

        let a_addr = a.as_ptr() as usize;
        let b_addr = b.as_ptr() as usize;
        let scratch_addr = scratch_arena.as_mut_ptr() as usize;

        let mut c = vec![0.0f32; m * n];
        let c_addr = c.as_mut_ptr() as usize;

        c.par_chunks_mut(slab_rows * n)
            .enumerate()
            .for_each(|(slab_idx, _chunk)| {
                let m_start = slab_idx * slab_rows;
                let m_size = (m - m_start).min(slab_rows);
                if m_size == 0 { return; }

                let scratch = unsafe {
                    (scratch_addr as *mut f32).add(slab_idx * scratch_stride_fast)
                };

                unsafe {
                    matmul_optimized_f32_slab(
                        a_addr as *const f32,
                        b_addr as *const f32,
                        c_addr as *mut f32,
                        m_start, m_size,
                        0, n,
                        k, n, n,
                        k,
                        false,
                        scratch,
                    );
                }
            });

        return c;
    }

    // ========================================================================
    // POW-OF-2-AWARE PATH (K or N triggers padding)
    // ========================================================================
    // Here we NEED the scope-based dispatch: par_chunks_mut doesn't let
    // us pass per-chunk scratch pointers cleanly for the A-pack region,
    // and the C-compaction at the end needs tight control over c_internal.

    // ========================================================================
    // POWER-OF-2 STRIDE DEFENCE
    // ========================================================================
    //
    // Pow-of-2-aware path (K or N triggers padding/packing). pack_a_enabled,
    // c_padded, n_stride were declared above before the fast-path branch.

    let mut c_internal = vec![0.0f32; m * n_stride];

    // Per-participant packing scratch: packed-B (KC × NC) always,
    // + packed-A (MR × KC) when A-packing is on, + anti-alias jitter.
    //
    // The +17 breaks pair-aliasing of the scratch slabs (naïve stride
    // × 4 B mod 4096 = 2048 → thread 0 and 2 alias at L1).  With +17
    // (stride × 4 mod 4096 = 2116) no small multiple hits zero.
    let pb_region = OPT_KC * ((OPT_NC + OPT_NR - 1) / OPT_NR) * OPT_NR;
    let pa_region = if pack_a_enabled { OPT_MR * OPT_KC } else { 0 };
    let scratch_stride = pb_region + pa_region + 17;
    // Uninitialised: pack functions write before reading. Zero-filling
    // ~1 MiB was measurable overhead at small sizes.
    let mut scratch_arena: Vec<f32> = Vec::with_capacity(scratch_stride * n_parts);
    unsafe { scratch_arena.set_len(scratch_stride * n_parts); }

    let a_addr = a.as_ptr() as usize;
    let b_addr = b.as_ptr() as usize;
    let c_addr = c_internal.as_mut_ptr() as usize;
    let scratch_addr = scratch_arena.as_mut_ptr() as usize;

    // Single atomic counter over M-slab indices. ~2 × n_parts total
    // claims, one Relaxed fetch_add each — negligible.
    let tile_counter = AtomicUsize::new(0);

    let worker = |tid: usize| {
        let scratch = unsafe { (scratch_addr as *mut f32).add(tid * scratch_stride) };

        loop {
            let t = tile_counter.fetch_add(1, Ordering::Relaxed);
            if t >= total_tiles {
                break;
            }

            let m_start = t * slab_rows;
            let m_size = (m - m_start).min(slab_rows);

            unsafe {
                matmul_optimized_f32_slab(
                    a_addr as *const f32,
                    b_addr as *const f32,
                    c_addr as *mut f32,
                    m_start,
                    m_size,
                    0,              // n_start: full width
                    n,              // n_size: full width
                    k,              // lda
                    n,              // ldb
                    n_stride,       // ldc (padded when N pow2)
                    k,              // logical K
                    pack_a_enabled, // A-pack only for pow2 K
                    scratch,
                );
            }
        }
    };

    // ONE Rayon dispatch. N workers spawned, caller runs inline as
    // participant N (the last scratch slot).
    rayon::scope(|s| {
        for tid in 0..n_workers {
            s.spawn(move |_| worker(tid));
        }
        worker(n_workers);
    });

    drop(scratch_arena);

    // Compact C back if we used a padded stride. Row-by-row SIMD copy.
    if c_padded {
        let mut c_out: Vec<f32> = Vec::with_capacity(m * n);
        unsafe {
            c_out.set_len(m * n);
            let src = c_internal.as_ptr();
            let dst: *mut f32 = c_out.as_mut_ptr();
            let n_v = n & !3;
            for i in 0..m {
                let s = src.add(i * n_stride);
                let d = dst.add(i * n);
                let mut jj = 0;
                while jj < n_v {
                    v128_store(d.add(jj) as *mut v128, v128_load(s.add(jj) as *const v128));
                    jj += 4;
                }
                while jj < n {
                    *d.add(jj) = *s.add(jj);
                    jj += 1;
                }
            }
        }
        c_out
    } else {
        c_internal
    }
}

/// Parallel optimised GEMM, v4: hijack Rayon's workers, drive them with
/// raw `memory.atomic.wait32`/`notify`.
///
/// v3 has one remaining inefficiency: `rayon::scope`'s join barrier.  When
/// the scope exits, the caller (main thread) must wait for all spawned
/// tasks.  Rayon's park/unpark on WASM goes through `wasm_sync` → condvar
/// → `memory.atomic.wait32` (workers) or busy-spin (main).  That's fine
/// for one scope, but if you want a *shared* packed-B across threads
/// (minimum total packing work), you need a barrier per (j, p) block —
/// 32 of them for 1024² — and Rayon's barrier is too heavy for that.
///
/// v4 enters `rayon::scope` ONCE per matmul, but inside it workers run our
/// own event loop: spin-then-wait on a generation counter, drain tiles,
/// signal done.  The main thread packs B for each block, bumps generation,
/// drains its tiles, spin-waits for workers, repeats.  Sync per block is
/// ~2 atomic.notify + N×1 atomic.fetch_sub (workers) + a short spin
/// (main, since it can't wait) — the pthreadpool model, hosted inside
/// Rayon's worker threads.
///
/// This is the fastest WASM-parallel path. v3 is simpler and gets most of
/// the win; v4 is for the last ~20% when you have many (j, p) blocks.
#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
pub fn matmul_optimized_f32_parallel_v4(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    use core::arch::wasm32::{memory_atomic_notify, memory_atomic_wait32};

    // u64 mul: WASM usize is 32-bit, m*n*k overflows at 2048³ → 0.
    if (m as u64) * (n as u64) * (k as u64) < (128u64 * 128 * 128) {
        return matmul_optimized_f32(a, b, m, n, k);
    }

    let n_threads = rayon::current_num_threads().max(1);
    let n_m_tiles = (m + OPT_MC - 1) / OPT_MC;

    if n_threads <= 1 || n_m_tiles < 2 {
        return matmul_optimized_f32(a, b, m, n, k);
    }

    let mut c = vec![0.0f32; m * n];

    // Single shared packed-B panel (KC × NC). Packed by main, read by all.
    let pb_cap = OPT_KC * ((OPT_NC + OPT_NR - 1) / OPT_NR) * OPT_NR;
    let mut packed_b = vec![0.0f32; pb_cap];

    let a_addr = a.as_ptr() as usize;
    let b_addr = b.as_ptr() as usize;
    let c_addr = c.as_mut_ptr() as usize;
    let pb_addr = packed_b.as_mut_ptr() as usize;

    // === Control state for the in-scope dispatch loop ===
    //
    // generation: bumped by main after packing. Workers wait for it to
    //   change. Top bit signals "exit the loop".
    // active: decremented by each thread (including main) on block
    //   completion. 0 → everyone's done with this block.
    // tile_counter: atomic queue of M-tiles, reset each block.
    // cur_{j,p,j_block,p_block,beta}: per-block geometry, published by
    //   main before bumping generation. Plain loads on workers are OK
    //   because they Acquire via generation.
    //
    // All control state is cache-line-isolated from the FP data (Vec
    // allocations are >64B apart from stack atomics).

    let generation = AtomicI32::new(0);
    let active = AtomicU32::new(0);
    let tile_counter = AtomicUsize::new(0);

    // Per-block geometry. Written by main, read by workers. No atomics
    // on these fields individually: the Release store on `generation`
    // sequences them, workers Acquire-load generation before reading.
    //
    // UnsafeCell is !Sync, so we can't hand `&UnsafeCell` to worker
    // closures. Instead, smuggle the address as a usize — workers
    // reconstruct the pointer. This is the standard dance for "trust me,
    // I've done the synchronisation" in WASM parallel code.
    //
    // Layout note: the tuple is 5 usize-sized fields = 40 bytes, so reads
    // are NOT atomic. That's fine: workers only read after Acquire-seeing
    // the new generation, and main only writes while workers are spinning
    // (before the Release store).
    let cur = std::cell::UnsafeCell::new((0usize, 0usize, 0usize, 0usize, 0.0f32));
    let cur_addr = cur.get() as usize;

    // Smuggle atomic addresses too, so worker closures only capture Copy
    // values (keeps them Send + Sync without fighting borrow lifetimes
    // across rayon's scope).
    let gen_addr = generation.as_ptr() as usize;
    let active_addr = active.as_ptr() as usize;
    let tile_addr = tile_counter.as_ptr() as usize;

    const SPIN_ITERS: u32 = 50_000;  // ~50 μs before parking; GEMM blocks
                                      // finish faster so workers rarely park
    const EXIT_BIT: i32 = 1 << 30;

    /// Inner tile-drain loop. Free function with no captures — all state
    /// passed explicitly — so it's trivially Send+Sync and inlines cleanly.
    ///
    /// Safety: caller has established the generation Acquire/Release
    /// happens-before so `pb` reads are consistent; claimed tiles write
    /// disjoint C rows.
    #[inline(always)]
    unsafe fn drain_tiles_v4(
        tile_addr: usize,
        n_m_tiles: usize,
        a_addr: usize,
        c_addr: usize,
        pb_addr: usize,
        m: usize,
        n: usize,
        k: usize,
        j: usize,
        p: usize,
        j_block: usize,
        p_block: usize,
        beta: f32,
    ) {
        let tile_ctr = &*(tile_addr as *const AtomicUsize);
        let j_panels = (j_block + OPT_NR - 1) / OPT_NR;
        let pb_ptr = pb_addr as *const f32;

        loop {
            let t = tile_ctr.fetch_add(1, Ordering::Relaxed);
            if t >= n_m_tiles {
                break;
            }

            let m_start = t * OPT_MC;
            let m_size = (m - m_start).min(OPT_MC);
            let i_main = m_size / OPT_MR * OPT_MR;

            let a_ptr = (a_addr as *const f32).add(m_start * k + p);
            let c_ptr = (c_addr as *mut f32).add(m_start * n + j);

            let mut ii = 0;
            while ii < i_main {
                let mut jj = 0;
                while jj < j_panels * OPT_NR && jj < j_block {
                    let panel = jj / OPT_NR;
                    let pb = pb_ptr.add(panel * p_block * OPT_NR);
                    let n_rem = j_block - jj;
                    if n_rem >= OPT_NR {
                        micro_kernel_6x8_fma(
                            p_block,
                            a_ptr.add(ii * k), k,
                            pb,
                            c_ptr.add(ii * n + jj), n,
                            beta,
                        );
                    } else {
                        micro_kernel_edge(
                            OPT_MR, n_rem, p_block,
                            a_ptr.add(ii * k), k,
                            pb,
                            c_ptr.add(ii * n + jj), n,
                            beta,
                        );
                    }
                    jj += OPT_NR;
                }
                ii += OPT_MR;
            }
            if ii < m_size {
                let m_rem = m_size - ii;
                let mut jj = 0;
                while jj < j_panels * OPT_NR && jj < j_block {
                    let panel = jj / OPT_NR;
                    let pb = pb_ptr.add(panel * p_block * OPT_NR);
                    let n_rem = (j_block - jj).min(OPT_NR);
                    micro_kernel_edge(
                        m_rem, n_rem, p_block,
                        a_ptr.add(ii * k), k,
                        pb,
                        c_ptr.add(ii * n + jj), n,
                        beta,
                    );
                    jj += OPT_NR;
                }
            }
        }
    }

    // Worker event loop. All captures are Copy (usize addresses +
    // primitives) → closure is trivially Send + Sync and can be shared
    // by-reference across all spawns.
    let worker_loop = move || {
        let gen = unsafe { &*(gen_addr as *const AtomicI32) };
        let act = unsafe { &*(active_addr as *const AtomicU32) };

        let mut seen = 0i32;
        loop {
            // Spin-then-wait for generation to change.
            let mut i = 0u32;
            let mut g = gen.load(Ordering::Acquire);
            while g == seen {
                if i < SPIN_ITERS {
                    core::hint::spin_loop();
                    i += 1;
                    g = gen.load(Ordering::Acquire);
                } else {
                    // Park. Workers (Web Worker threads) CAN atomic.wait.
                    // Timeout -1 = infinite; main always notifies.
                    unsafe {
                        memory_atomic_wait32(gen_addr as *mut i32, seen, -1);
                    }
                    g = gen.load(Ordering::Acquire);
                }
            }
            if g & EXIT_BIT != 0 {
                return;
            }
            seen = g;

            // Load block geometry. Acquire on generation synchronises
            // with main's Release store of (j, p, ...) and tile_counter
            // reset.
            let (j, p, j_block, p_block, beta) = unsafe {
                *(cur_addr as *const (usize, usize, usize, usize, f32))
            };
            unsafe {
                drain_tiles_v4(
                    tile_addr, n_m_tiles, a_addr, c_addr, pb_addr,
                    m, n, k, j, p, j_block, p_block, beta,
                );
            }

            // Signal done. Last thread drops active to 0. Main spins on
            // this; workers fire a notify anyway in case a future caller
            // is on a Worker thread (and CAN wait).
            if act.fetch_sub(1, Ordering::AcqRel) == 1 {
                unsafe {
                    memory_atomic_notify(active_addr as *mut i32, 1);
                }
            }
        }
    };

    // Enter the scope ONCE. Workers spin until we set EXIT_BIT.
    // Inside the scope: main runs the (j, p) BLIS loop, dispatching each
    // block to workers via generation+notify.
    rayon::scope(|s| {
        for _ in 1..n_threads {
            // Spawn holds &worker_loop (closure is Sync). No per-spawn alloc
            // beyond rayon's own Box<dyn FnOnce>, once per thread, ONCE per
            // matmul.
            s.spawn(|_| worker_loop());
        }

        // Main thread drives the (j, p) loop.
        let mut j = 0;
        while j < n {
            let j_block = (n - j).min(OPT_NC);

            let mut p = 0;
            while p < k {
                let p_block = (k - p).min(OPT_KC);

                // Pack B serially. Workers are spinning on generation,
                // so the pack doesn't race with their packed_b reads.
                unsafe {
                    pack_b_optimized(
                        (b_addr as *const f32).add(p * n + j),
                        n,
                        pb_addr as *mut f32,
                        p_block,
                        j_block,
                    );
                }

                let beta = if p == 0 { 0.0 } else { 1.0 };

                // Publish block geometry, arm completion, reset tile queue.
                // All sequenced-before the Release store on generation.
                unsafe {
                    *(cur_addr as *mut (usize, usize, usize, usize, f32)) =
                        (j, p, j_block, p_block, beta);
                }
                tile_counter.store(0, Ordering::Relaxed);
                active.store(n_threads as u32, Ordering::Relaxed);

                // Go.
                generation.fetch_add(1, Ordering::Release);
                unsafe {
                    memory_atomic_notify(gen_addr as *mut i32, u32::MAX);
                }

                // Main does its share.
                unsafe {
                    drain_tiles_v4(
                        tile_addr, n_m_tiles, a_addr, c_addr, pb_addr,
                        m, n, k, j, p, j_block, p_block, beta,
                    );
                }

                // Signal our completion, then spin-wait for stragglers.
                // Main cannot atomic.wait (browser traps on main thread) —
                // spin is the only option. For GEMM this is sub-μs: main
                // did ~1/N of the work so the tail is short.
                if active.fetch_sub(1, Ordering::AcqRel) > 1 {
                    let mut spins = 0u64;
                    while active.load(Ordering::Acquire) != 0 {
                        core::hint::spin_loop();
                        spins += 1;
                        if spins > 10_000_000_000 {
                            // Something deadlocked. Better to panic than
                            // hang the tab.
                            panic!("v4: workers stalled in block ({}, {})", j, p);
                        }
                    }
                }

                p += OPT_KC;
            }
            j += OPT_NC;
        }

        // Tell workers to exit, wake any that are parked.
        generation.fetch_or(EXIT_BIT, Ordering::Release);
        unsafe {
            memory_atomic_notify(gen_addr as *mut i32, u32::MAX);
        }
    });

    drop(packed_b);
    let _ = cur; // keep UnsafeCell alive past the scope join
    c
}

// Stubs for non-atomics builds so the WASM binding layer can unconditionally
// reference these symbols.
#[cfg(all(target_arch = "wasm32", not(target_feature = "atomics")))]
pub fn matmul_optimized_f32_parallel_v4(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    matmul_optimized_f32(a, b, m, n, k)
}
