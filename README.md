# Tensor Engine

A C++ tensor library with cache-blocked GEMM, numerically stable softmax, and O(1) transpose via stride manipulation.

---

## What This Repo Contains

- `Tensor` class: N-dimensional array with stride-based indexing, 32-byte aligned allocation, and `std::shared_ptr<float[]>` reference counting.
- `matmul_naive`: Triple-loop GEMM with stride-aware indexing. Baseline for benchmarking.
- `matmul_tiled`: Cache-blocked GEMM with 6 loops (outer 3 for tiles, inner 3 for multiply). TILE size chosen to fit in L1 cache. Uses `i,k,j` inner loop order with pointer arithmetic for contiguous access.
- `matmul`: Stride-aware GEMM with dynamic branching. Fast path uses contiguous pointer arithmetic when `Bstride1 == 1 && Cstride1 == 1` (compiler auto-vectorization). Fallback uses stride-aware indexing for transposed or strided tensors.
- `softmax` / `softmax_naive`: Numerically stable vs unstable softmax. Naive overflows on large inputs; stable version subtracts the max before exponentiation.
- `transpose()`: O(1) operation that reverses `shape` and `strideVector`. No data copied; both tensors share the same underlying buffer via `shared_ptr`.
- Rule of Five: Copy constructor/assignment perform deep copy. Move constructor/assignment transfer ownership and zero out the source.

---

## Build and Run

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/tensor_run
```

### Build Profiles

| Profile | Flags | Use Case |
|---------|-------|----------|
| `Release` | `-O3 -march=native -ffast-math` | Production / max speed |
| `Debug` | `-O0 -g` | Valgrind-compatible, no arch-specific instructions |
| `Benchmark` | `-O2 -g` | Realistic baseline, no auto-vectorization tricks |
| `Asan` | `-O1 -g -fsanitize=address` | Memory error detection |

### Memory Safety Check

```bash
cmake -B build_dbg -DCMAKE_BUILD_TYPE=Debug
cmake --build build_dbg
valgrind --leak-check=full --show-leak-kinds=all ./build_dbg/tensor_run
```

```bash
==197371== HEAP SUMMARY:
==197371==     in use at exit: 0 bytes in 0 blocks
==197371==   total heap usage: 131,092 allocs, 131,092 frees, 1,908,928 bytes allocated
==197371== 
==197371== All heap blocks were freed -- no leaks are possible
==197371== 
==197371== For lists of detected and suppressed errors, rerun with: -s
==197371== ERROR SUMMARY: 0 errors from 0 contexts (suppressed: 0 from 0)
```
---

## Benchmarks

### GEMM: Naive vs Tiled

Profiled with `perf stat` on an Intel Core i7-1165G7 (single-thread). Tile size = 32.

| N | Implementation | Time (ms) | GFLOPS | L1 Misses | LLC Misses |
|---|---------------|-----------|--------|-----------|------------|
| 256 | naive | 17.24 | 1.95 | 17,320,216 | 5,406 |
| 512 | naive | 180.37 | 1.49 | 135,612,995 | 11,311 |
| 1024 | naive | 2370.11 | 0.91 | 1,156,057,014 | 1,884,210 |
| 2048 | naive | 39582.3 | 0.43 | 9,440,082,246 | 3,064,563,441 |
| 256 | tiled | 5.84 | 5.74 | 250,527 | 4,775 |
| 512 | tiled | 51.37 | 5.23 | 15,993,300 | 8,908 |
| 1024 | tiled | 412.83 | 5.20 | 132,443,665 | 26,622 |
| 2048 | tiled | 3603.1 | 4.77 | 1,074,751,775 | 3,795,964 |

### Roofline Model

![Roofline Plot](benchmarks/roofline.png)

*CPU: Intel Core i7-1165G7. Single-thread peak: 150.4 GFLOPS (FP32). Memory bandwidth: 51.2 GB/s (DDR4-3200 dual channel).*

### Methodology

I profiled naive and tiled matrix multiplication at N = 256, 512, 1024, and 2048 using `perf stat` to count L1 and LLC cache misses. The naive version accesses matrix B with stride-N, jumping 4096 bytes per inner-loop iteration and loading a new cache line every time. Tiling loads a 32×32 block into cache and reuses it across multiple iterations, reducing L1 misses by 8.5× at N=512. At N=512, naive took 180.37 ms with 135,612,995 L1 misses; tiled took 51.37 ms with 15,993,300 misses. The Roofline plot shows that by the LLC-miss proxy for memory traffic, most points fall to the right of the 2.94 ridge point, placing them in the compute-bound region by operational intensity. However, both naive and tiled perform far below the 150 GFLOPS compute roof due to missing SIMD intrinsics, no register blocking, and cache associativity conflicts within the tile. Only N=2048 naive sits on the memory roof, where the working set exceeds cache capacity and causes massive LLC thrashing.
### Known Limitations

The tiled kernel reaches 5 GFLOPS, which is 3.5% of the single-core peak. Further speedup would require SIMD vectorization, multithreading, and transposed B storage. Transposed inputs are not handled in the tiled kernel; the fast path assumes contiguous inner dimension.
## Implementation Notes

- **Flat Data Layout:** Memory is a single 1D `float*` array, 32-byte aligned via `std::aligned_alloc` (Linux) or `_aligned_malloc` (Windows), wrapped in `std::shared_ptr<float[]>` with a custom deleter.
- **Stride Vector:** Computed at construction for row-major layout. `transpose()` reverses both `shape` and `strideVector`, making arbitrary-dimension transpose a metadata-only operation.
- **Bounds Checking:** `operator()` validates coordinate rank and bounds, throwing `std::out_of_range` on violation.
- **SIMD Fast Lane:** `matmul` checks if the inner dimension is contiguous (`Bstride1 == 1`). If true, it uses raw pointer arithmetic (`B_row[j]`, `C_row[j]`) that the compiler can auto-vectorize with AVX2. If false, it falls back to stride-aware indexing.
- **Tiled GEMM:** Outer 3 loops step by TILE. Inner 3 loops use `i,k,j` order with `std::min` boundary handling. `A(i,k)` is hoisted to a register and reused across the inner `j` loop. Pointer arithmetic is used for B and C rows when contiguous.
- **Softmax Stability:** Computes the maximum value first, then subtracts it from every element before calling `std::exp()`. This keeps the largest exponent at `e^0 = 1`, preventing overflow on inputs like `[1000, 1001, 999]`.
- **Aligned Allocation:** `data_initialization` rounds up to the nearest 32-byte boundary and zero-initializes with `std::memset`.

---

## Tests

The test suite in `main.cpp` covers:

- **Allocation & Indexing:** Basic element access, bounds checking, rank mismatch.
- **Transpose:** Zero-copy aliasing, shape reversal, stride correctness.
- **Matmul:** All 4 stride combinations (normal×normal, normal×transposed, transposed×normal, transposed×transposed), identity check, outer product.
- **Softmax:** Numerical stability on extreme values, sum-to-one verification, uniform input handling.
- **Rule of Five:** Deep copy independence, move ownership transfer.
- **Tiled GEMM Correctness:** Square (N=256, TILE=64), non-divisible (N=100, TILE=32), small exact (N=4, TILE=2).
- **Benchmark:** 500×500 timing across all stride cases, correctness verification on small exact integer matrices.
