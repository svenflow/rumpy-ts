# rumpy-ts

High-performance NumPy-like library for TypeScript, powered by Rust and WebAssembly.

## Rename Decision

**New name: `tsnp`** (TypeScript NumPy)
- npm package name `tsnp` is available (verified 2026-03-04)
- Short, memorable, hints at TS + NumPy
- GitHub: no competing repos with this name for numeric/array libs

Rename checklist when ready:
- [ ] Update package.json name to "tsnp"
- [ ] Update README title and references
- [ ] Create new GitHub repo or rename existing
- [ ] Publish to npm as "tsnp"

## Development

**CRITICAL: Always use `bun`, never use raw `npm`.**

```bash
# CORRECT
bun install
bun run build
bun test

# WRONG - never do this
npm install
npm run build
npm test
```

## Build Commands

```bash
# Build Rust crates
cargo build --all

# Run Rust tests
cargo test --all

# Build WASM
wasm-pack build crates/rumpy-wasm --target web

# Build and test TypeScript
cd tests && bun install && bun test
```

## Architecture

```
rumpy-ts/
├── crates/
│   ├── rumpy-core/      # Backend traits and common types
│   ├── rumpy-cpu/       # CPU backend (ndarray/faer)
│   ├── rumpy-wasm/      # WASM bindings
│   ├── rumpy-webgpu/    # WebGPU backend (future - will be TypeScript)
│   ├── rumpy-tests/     # Shared Rust test suite
│   └── pthreadpool-rs/  # Thread pool for parallel computation
├── tests/               # JavaScript test suite (mirrors Rust tests)
│   ├── test-utils.ts    # Backend interface + helpers
│   ├── creation.test.ts # zeros, ones, arange, linspace, eye, diag
│   ├── math.test.ts     # trig, exp, log, binary ops
│   ├── linalg.test.ts   # matmul, dot, inv, det, svd, qr
│   ├── stats.test.ts    # sum, mean, std, min, max, cumsum
│   └── index.test.ts    # Main entry - runs all tests against backends
└── benchmarks/          # Performance benchmarks
```

## Backend System

All backends implement the same trait interface (Rust) or TypeScript interface (JS):

- `CreationOps` - zeros, ones, arange, linspace, eye, diag
- `MathOps` - sin, cos, exp, log, sqrt, add, mul, etc.
- `StatsOps` - sum, mean, std, var, min, max, cumsum, cumprod
- `LinalgOps` - matmul, inv, det, svd, qr, solve
- `ManipulationOps` - reshape, transpose, concatenate
- `RandomOps` - rand, randn, uniform, normal
- `CompareOps` - eq, lt, gt, isnan, isinf

## Testing Strategy

Tests are parameterized to run against multiple backends:

1. **Rust tests** (`crates/rumpy-tests/`) - Run via `cargo test`
2. **JavaScript tests** (`tests/`) - Run via `bun test`, test WASM and WebGPU backends

Both test suites should have identical test coverage.

## WebGPU Performance (2026-03-04)

**🏆 rumpy-webgpu CRUSHES tfjs-webgpu at ALL matrix sizes! 🏆**

### Latest Benchmark Results (Native Chrome, M4 Pro)

| Size | rumpy GFLOPS | tfjs GFLOPS | Winner |
|------|-------------|-------------|--------|
| 512x512 | 224 | 206 | **rumpy 1.08x** 🎉 |
| 1024x1024 | 1023 | 467 | **rumpy 2.19x** 🎉🎉 |
| 2048x2048 | 2070 | 1273 | **rumpy 1.63x** 🎉🎉 |
| 4096x4096 | 5799 | 2264 | **rumpy 2.56x** 🎉🎉🎉 |

### Key Breakthroughs

1. **Store A as vec4 along K dimension (NOT M)** - THE critical insight from reverse-engineering tfjs
2. **B-value register caching** - Load 4 B vec4s into registers BEFORE iterating over A rows (maximizes ILP)
3. **Autotune selects optimal shader per size**:
   - small-16 (16x16 tiles): Best for 512 and 4096
   - tfjs-32 (32x32 tiles): Best for 1024 and 2048

### The Winning Algorithm

```wgsl
// tfjs exact pattern - the key is 8 groups of 4 K values
for (var k = 0; k < 8; k++) {
  // Cache 4 consecutive B values in registers
  let BCached0 = mm_Bsub[k * 4 + 0][tileCol];
  let BCached1 = mm_Bsub[k * 4 + 1][tileCol];
  let BCached2 = mm_Bsub[k * 4 + 2][tileCol];
  let BCached3 = mm_Bsub[k * 4 + 3][tileCol];

  // Iterate over 4 A rows, using ACached[0-3] for 4 K values each
  for (var i = 0; i < 4; i++) {
    let ACached = mm_Asub[tileRow + i][k];  // 4 K values as vec4
    acc[i] = fma(BCached0, vec4<f32>(ACached[0]), acc[i]);
    acc[i] = fma(BCached1, vec4<f32>(ACached[1]), acc[i]);
    acc[i] = fma(BCached2, vec4<f32>(ACached[2]), acc[i]);
    acc[i] = fma(BCached3, vec4<f32>(ACached[3]), acc[i]);
  }
}
```

### Benchmark Files

Run `node tests/serve.mjs` then open in Chrome:
- http://localhost:8089/final-benchmark.html - Full autotune + all sizes
- http://localhost:8089/tfjs-exact-v2-benchmark.html - tfjs-32 shader only
- http://localhost:8089/tfjs-shader-dump.html - Dump actual tfjs shader code

**CRITICAL: ALWAYS test WebGPU in native Chrome, NOT Playwright/headless (adds ~15-20% overhead)**
