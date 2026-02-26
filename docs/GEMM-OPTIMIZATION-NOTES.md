# GEMM Optimization Notes

## Current Status (2026-02-25 - Updated)

### What's Working

**All kernels produce correct outputs** (verified against TensorFlow.js XNNPACK):
- `matmulF32` - Basic f32 SIMD with mul+add (6x8 micro-kernel)
- `matmulF32Packed` - With B matrix packing (4x8 micro-kernel)
- `matmulF32FMA` - Using relaxed_madd FMA instructions
- `matmulF32Parallel` - Multi-threaded via Rayon (14 threads)
- `matmulXnnpack` - XNNPACK-style with pre-packed B (fastest single-threaded)
- `matmulF32Blocked` - Cache-blocked 6x8 kernel (experimental)
- `matmulXnnpackBlocked` - Cache-blocked XNNPACK-style (experimental)

### Performance vs XNNPACK (single-threaded, 2026-02-25 benchmarks)

With tfjs `setThreadsCount(1)` for fair comparison:

| Size | rumpy XNNPACK | tfjs 1-thread | Ratio | Notes |
|------|---------------|---------------|-------|-------|
| 32x32 | 0.013ms | 0.016ms | **0.81x FASTER** | Small overhead wins |
| 64x64 | 0.022ms | 0.017ms | 1.29x slower | Overhead dominates |
| 100x100 | 0.067ms | 0.044ms | 1.53x slower | N%8 handled correctly |
| 128x128 | 0.081ms | 0.070ms | 1.16x slower | |
| 256x256 | 0.672ms | 0.531ms | 1.27x slower | Main gap area |
| 384x384 | 1.70ms | 1.77ms | **0.96x FASTER** | |
| 512x512 | 4.30ms | 4.21ms | 1.02x slower | Nearly matched |
| 768x768 | 13.4ms | 16.0ms | **0.84x FASTER** | |
| 1024x1024 | 37.0ms | 33.7ms | 1.10x slower | Close |

**Summary: We match or beat XNNPACK at 32, 384, 768!**

### Multi-threaded Performance (14 threads)

| Size | rumpy parallel | tfjs | Ratio | Notes |
|------|----------------|------|-------|-------|
| 256x256 | 0.34ms | 0.53ms | **1.56x FASTER** | |
| 384x384 | 0.56ms | 1.77ms | **3.18x FASTER** | |
| 512x512 | 1.23ms | 4.21ms | **3.41x FASTER** | |
| 768x768 | 3.41ms | 15.96ms | **4.68x FASTER** | |
| 1024x1024 | 8.95ms | 33.74ms | **3.77x FASTER** | |

**Our parallel implementation is 1.5-4.7x FASTER than single-threaded tfjs!**

## Key Learnings

### 1. Pre-packing B is Critical
The single biggest win was pre-packing B matrix once and reusing it. Without pre-packing, we were 6-8x slower because we packed B on every matmul call.

With pre-packing API (`packB` + `matmulXnnpack`), we match XNNPACK at most sizes.

### 2. FMA is Not Always Faster
Counterintuitively, `f32x4_relaxed_madd` (FMA) is slightly SLOWER than `f32x4_mul` + `f32x4_add` in WASM:
- 1024x1024: FMA 53ms vs mul+add 47ms

This might be because:
- WASM JIT may not have optimal FMA codegen
- Instruction scheduling may be worse with FMA
- XNNPACK itself uses mul+add, not FMA

### 3. Cache Blocking Adds Overhead in WASM
Implemented cache blocking (KC=256, MC=128, NC=256) but it's SLOWER than non-blocked:
- Overhead of zeroing C and load-add-store per tile
- WASM memory model may be different from native
- XNNPACK likely has more sophisticated streaming/prefetch that WASM can't express

The blocked kernels (`matmulF32Blocked`, `matmulXnnpackBlocked`) are available but not recommended.

### 4. XNNPACK's Multi-threading is Built-in
TensorFlow.js WASM backend uses XNNPACK which has built-in multi-threading. When comparing, we need to ensure fair thread counts. With `tf.wasm.setThreadsCount(1)`, we get fair single-threaded comparison.

### 5. Tile Size Matters
We use 6x8 tiles (6 rows, 8 cols = 2 v128s per row). XNNPACK also uses 6x8 or similar. The tile size affects:
- Register pressure (12 accumulators = 12 v128 registers)
- Cache efficiency
- Alignment (n must be divisible by 8 for SIMD panels)

### 6. Memory Layout
XNNPACK packs B in a specific format: for each panel of 8 columns, K values are interleaved:
```
panel0[k0:col0-7, k1:col0-7, k2:col0-7, ...]
panel1[k0:col8-15, ...]
```

This gives sequential memory access in the inner loop.

## Resolved Issues

### N % 8 != 0 Bug (FIXED)
The XNNPACK-style kernel now handles arbitrary matrix dimensions:
- `matmul_simd_f32_xnnpack_style_full` takes both original B and packed_b
- SIMD panels use packed_b, remaining columns use original B
- 100x100 matrices now work correctly

## Outstanding Issues

### 1. 256x256 Performance Gap
At 256x256, we're 1.27x slower than tfjs. This is the main area for improvement.

Possible causes:
- Not enough work to amortize packing overhead
- Sub-optimal K loop unrolling at this size
- XNNPACK may have special-cased this size

### 2. Small Matrix Overhead (64-128)
At 64x64 and 128x128, we're 1.16-1.29x slower. Function call and packing overhead dominates.

Potential fixes:
- Inline small matrices
- Skip packing for small sizes
- Use simpler 4x8 kernel for sizes < 128

### 3. Parallel Overhead at Small Sizes
At small sizes (64x64), parallel is much slower than single-threaded due to thread spawn/join overhead. Current heuristic requires m*n*k >= 64^3 to use parallel.

## Architecture

```
matmulF32 (default)
  -> matmul_dispatch_f32
    -> if m >= 4, n >= 8: matmul_simd_f32 (mul+add kernel)
    -> else: matmul_scalar_f32

matmulF32Parallel
  -> rayon parallel_iter over row chunks
    -> each thread calls matmul_dispatch_f32

matmulXnnpack (fastest single-threaded)
  -> packB (pre-pack B once)
  -> matmul_simd_f32_xnnpack_style_full (6x8 kernel with mul+add)
    -> SIMD for panels 0..(n/8)
    -> scalar fallback for remaining columns

matmulF32Blocked (experimental, not recommended)
  -> Zero C
  -> Triple-nested blocking (NC, KC, MC)
  -> Load-add-store per tile
```

## Recommendations

**For production use:**
1. **Single matmul**: Use `matmulF32` for simplicity, or `matmulXnnpack` with pre-packed B if weights are reused
2. **Neural network inference**: Pre-pack weights with `packB`, use `matmulXnnpack` for forward passes
3. **Large matrices**: Use `matmulF32Parallel` for matrices >= 256 (3-4x faster than tfjs single-threaded)
4. **Small matrices**: Use `matmulF32` (parallel overhead not worth it)

## Build Commands

```bash
# Build with threads+SIMD
rustup default nightly
wasm-pack build crates/rumpy-wasm --target web --out-dir ../../benchmarks/pkg-web --release

# Run benchmarks
cd benchmarks
npm run dev  # or: npx vite
node run-f32-benchmark.js
```

## Future Work

1. **Tune small matrix path** - Special-case 64-256 to reduce overhead
2. **Investigate 256 gap** - Profile why this specific size is slower
3. **Parallel XNNPACK** - Combine pre-packing with multi-threading for best of both
4. **WebGPU GEMM** - For matrices > 1024, GPU should be faster
