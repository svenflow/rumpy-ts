# rumpy.ts

High-performance NumPy-like library for TypeScript, powered by Rust and WebAssembly.

[![CI](https://github.com/svenflow/rumpy-ts/actions/workflows/ci.yml/badge.svg)](https://github.com/svenflow/rumpy-ts/actions/workflows/ci.yml)

## Features

- **NumPy-compatible API** - Familiar interface for Python developers
- **Rust performance** - Compiled to native code or WebAssembly
- **Pluggable backends** - CPU (ndarray/faer), WASM, WebGPU (planned)
- **TypeScript-first** - Full type definitions included
- **Zero dependencies** - Self-contained WASM bundle

## Installation

```bash
npm install rumpy-ts
```

## Quick Start

```typescript
import { initNumpy, zeros, ones, arange, linspace, eye } from 'rumpy-ts';
import { sin, cos, exp, log, sqrt } from 'rumpy-ts';
import { linalg, random } from 'rumpy-ts';

// Initialize WASM module
await initNumpy();

// Array creation
const a = zeros([3, 4]);           // 3x4 array of zeros
const b = ones([2, 3]);            // 2x3 array of ones
const c = arange(0, 10, 1);        // [0, 1, 2, ..., 9]
const d = linspace(0, 1, 5);       // [0, 0.25, 0.5, 0.75, 1]
const I = eye(3);                  // 3x3 identity matrix

// Math operations
const angles = linspace(0, Math.PI, 100);
const sines = sin(angles);
const cosines = cos(angles);

// Element-wise arithmetic
const x = arange(1, 6, 1);
const y = x.add(10);               // [11, 12, 13, 14, 15]
const z = x.mul(2);                // [2, 4, 6, 8, 10]

// Aggregations
console.log(x.sum());              // 15
console.log(x.mean());             // 3
console.log(x.std());              // 1.414...

// Linear algebra
const A = NDArray.fromArray([[1, 2], [3, 4]]);
const B = NDArray.fromArray([[5, 6], [7, 8]]);
const C = linalg.matmul(A, B);     // Matrix multiplication
const Ainv = linalg.inv(A);        // Matrix inverse
const det = linalg.det(A);         // Determinant

// Random numbers
random.seed(42);
const r = random.rand([3, 3]);     // Uniform [0, 1)
const n = random.randn([3, 3]);    // Standard normal
```

## Architecture

rumpy.ts uses a pluggable backend architecture:

```
┌─────────────────────────────────────────┐
│           TypeScript API                │
│    (NDArray, linalg, random, etc.)      │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────┴───────────────────────┐
│           rumpy-core                    │
│    (Backend traits, Array interface)    │
└─────────────────┬───────────────────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
┌───┴───┐   ┌─────┴─────┐   ┌───┴───┐
│  CPU  │   │   WASM    │   │WebGPU │
│ndarray│   │wasm-bindgen│  │(future)│
│ faer  │   │           │   │       │
└───────┘   └───────────┘   └───────┘
```

### Backend Traits

All backends implement the same operation traits:

- `CreationOps` - zeros, ones, arange, linspace, eye
- `MathOps` - sin, cos, exp, log, sqrt, add, mul, etc.
- `StatsOps` - sum, mean, std, var, min, max
- `LinalgOps` - matmul, inv, det, svd, qr, solve
- `ManipulationOps` - reshape, transpose, concatenate
- `RandomOps` - rand, randn, uniform, normal
- `CompareOps` - eq, lt, gt, isnan, isinf

## Performance

### Matrix Multiplication (GEMM)

rumpy.ts **beats TensorFlow.js WASM backend** on matrix multiplication, especially at large sizes:

| Size | rumpy.ts | TF.js WASM | NumPy | vs TF.js | vs NumPy |
|------|----------|------------|-------|----------|----------|
| 512 | 0.8ms | 0.7ms | 1.1ms | 0.97x | **1.37x** ⭐ |
| 1024 | 4.4ms | 5.4ms | 1.9ms | **1.23x** ⭐ | 0.43x |
| 2048 | 31ms | 41ms | 9.5ms | **1.32x** ⭐ | 0.31x |
| 4096 | 199ms | 322ms | 74ms | **1.62x** ⭐ | 0.37x |
| 8192 | 1.5s | 2.6s | 596ms | **1.74x** ⭐ | 0.40x |

*Benchmarked on Mac mini M4 Pro (14 cores, 64GB RAM). rumpy.ts and TF.js use WASM SIMD + threads. NumPy uses Apple Accelerate (native ARM BLAS with AMX coprocessor).*

**Key takeaways:**
- rumpy.ts is **1.2-1.7x faster than TensorFlow.js** at sizes ≥1024
- NumPy with native BLAS is 2.5-3x faster than any WASM implementation (hardware advantage)
- At 512×512, rumpy.ts actually beats NumPy due to lower overhead

### Auto-Pack API (tfjs-like)

rumpy.ts automatically caches packed weight matrices for fast repeated matmuls — same pattern as TensorFlow.js:

```typescript
import { matmul, initFutexPool, futexWorkersReady } from 'rumpy';

// Initialize thread pool (one-time setup)
initFutexPool(navigator.hardwareConcurrency);
while (!futexWorkersReady()) await new Promise(r => setTimeout(r, 10));

// Create weight matrix once
const weights = new Float32Array(K * N);
// ... fill weights ...

// First call: packs B and caches (slightly slower)
const out1 = matmul(input1, weights, M, N, K);

// Subsequent calls: uses cached pack (fast!)
const out2 = matmul(input2, weights, M, N, K);
const out3 = matmul(input3, weights, M, N, K);
```

The caching is automatic — no API changes needed. The packed format is cached in a WeakMap keyed by the Float32Array, so it's automatically garbage collected when the weights are no longer referenced.

### Zero-Copy API for Maximum Performance

For NN inference workloads, use the zero-copy API to eliminate JS↔WASM data transfer overhead:

```typescript
import { initNumpy, allocF32, packBInPlace, matmulF32ZeroCopy } from 'rumpy-ts';

await initNumpy();
await initThreadPool(8);

// Allocate buffers in WASM memory (one-time setup)
const bufA = allocF32(M * K);
const bufB = allocF32(K * N);
const bufC = allocF32(M * N);

// Fill via zero-copy Float32Array views
const viewA = new Float32Array(wasmMemory().buffer, bufA.ptr(), M * K);
const viewB = new Float32Array(wasmMemory().buffer, bufB.ptr(), K * N);
viewA.set(yourInputData);
viewB.set(yourWeights);

// Matmul operates entirely in WASM memory — no copies per call!
matmulF32ZeroCopy(bufA, bufB, bufC, M, K, N);

// Read result via zero-copy view
const viewC = new Float32Array(wasmMemory().buffer, bufC.ptr(), M * N);
```

### Other Operations

| Operation | Pure JS | rumpy.ts | NumPy |
|-----------|---------|----------|-------|
| Sum 1M elements | ~50ms | ~0.5ms | ~0.4ms |
| Element-wise sin | ~200ms | ~3ms | ~2.5ms |

rumpy.ts achieves near-NumPy performance through:
- SIMD-optimized 6x8 GEMM micro-kernel with FMA instructions
- Futex-based thread pool with ~10μs dispatch overhead
- Auto-packed weight matrices (cached on first use)
- Cache-blocked tiling (MC=384, KC=256, NC=4096)
- Zero-copy API for WASM-resident tensor workflows
- WebAssembly SIMD and threading (SharedArrayBuffer)

## Development

```bash
# Build Rust crates
cargo build --all

# Run tests
cargo test --all

# Build WASM
wasm-pack build crates/rumpy-wasm --target web

# Build TypeScript
npm run build
```

## License

MIT
