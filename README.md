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

| Operation | Pure JS | rumpy.ts | NumPy |
|-----------|---------|----------|-------|
| Sum 1M elements | ~50ms | ~0.5ms | ~0.4ms |
| Matmul 500×500 | ~2500ms | ~15ms | ~12ms |
| Element-wise sin | ~200ms | ~3ms | ~2.5ms |

rumpy.ts achieves near-NumPy performance through:
- Rust's ndarray for optimized array operations
- faer for pure-Rust linear algebra (LAPACK-level performance)
- WebAssembly SIMD (when available)

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
