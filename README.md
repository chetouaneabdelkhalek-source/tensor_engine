# C++ Tensor Engine

A lightweight, zero-dependency C++ Tensor engine built entirely from scratch. This project is an exercise in low-level systems engineering, focusing on contiguous memory management, stride manipulation, hardware-aware computation, and numerical stability.

## 🚀 Engineering Features

### 1. O(1) Zero-Copy Transpose
Matrix transpositions do not copy any underlying data.
* **Implementation:** The `transpose()` method simply reverses the `shape` and `strideVector` arrays, returning a new `Tensor` that points at the same underlying buffer with different metadata.
* **Memory Aliasing:** The data buffer is held in a `std::shared_ptr<float[]>`. The transposed `Tensor` is constructed by copying this `shared_ptr`, which atomically increments its reference count. No bytes are copied — both views share one allocation, and the buffer is only freed once the last owning `Tensor` is destroyed.

### 2. Hardware-Aware Matmul (GEMM)
The `matmul` friend function includes a dynamic branching optimization to maximize CPU cache locality and compiler auto-vectorization.
* **SIMD Fast Lane:** It checks if the inner dimensions are contiguous in memory (`Bstride1 == 1 && Cstride1 == 1`). If so, it executes a contiguous loop that the compiler can easily auto-vectorize (e.g., using AVX instructions).
* **Safe Fallback:** If the tensors are strided or transposed views, it falls back to a stride-aware access pattern.

### 3. Numerically Stable Softmax
A naive $e^x$ implementation fails with `NaN` on large inputs due to floating-point overflow.
* **Max-Subtraction Trick:** The `softmax()` method computes the maximum value in the tensor first, then computes $e^{x_i - max}$. This keeps the maximum exponent at $e^0 = 1$, guaranteeing absolute numerical stability even with extreme logits (e.g., `1000.0` or `-1000.0`).

### 4. Memory Ownership & The Rule of Five
The engine implements the full **Rule of Five** (Destructor, Copy Constructor, Copy Assignment, Move Constructor, Move Assignment) on top of a `std::shared_ptr<float[]>` data buffer.
* **Leak-Free:** Copies perform a deep copy of the underlying buffer; moves transfer ownership of the `shared_ptr` and zero out the source's `dim`/`size`.
* **Aligned Allocation:** Data buffers are allocated 32-byte aligned (via `aligned_alloc` on Linux/macOS, `_aligned_malloc` on MSVC) to support SIMD vectorization, with a custom deleter wired into the `shared_ptr` to call the matching free function.
* **Move Semantics:** Utilizes `noexcept` move constructors to efficiently steal the `shared_ptr` and metadata from temporary r-values, avoiding expensive reallocations.

---

## 🛠️ Build and Execution

This project requires a C++20 compiler (CMake `CMAKE_CXX_STANDARD 20`).

**Build profiles** (set via `-DCMAKE_BUILD_TYPE=...`):
* `Release` — `-O3 -march=native -ffast-math` (default if unset)
* `Debug` — `-O0 -g`, no architecture-specific instructions (safe for Valgrind)
* `Asan` — `-O1 -g -fsanitize=address`

### Compile (Release)
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Run
```bash
./build/tensor_run
```

### Validate Memory Safety
Build in Debug mode (avoids `-march=native`, which Valgrind's instruction emulator may not support) and run Valgrind:
```bash
cmake -B build_dbg -DCMAKE_BUILD_TYPE=Debug
cmake --build build_dbg
valgrind --leak-check=full --show-leak-kinds=all ./build_dbg/tensor_run
```
*(Expected output: `0 bytes in 0 blocks` lost)*

---

## 💻 Usage Examples

### 1. Initialization and Flat Indexing
Tensors support N-dimensional shapes. Coordinates are translated to 1D flat memory via pre-computed stride vectors.
```cpp
Tensor A({2, 3}); // Creates a 2x3 matrix
A({0, 0}) = 1.0f; // Multi-dimensional indexing with bounds checking
A({1, 2}) = 6.0f;
```

### 2. Zero-Copy Operations
Because `transpose()` shares memory via `shared_ptr`, modifying the original tensor modifies the transposed view.
```cpp
Tensor A({1024, 1024});
Tensor A_T = A.transpose(); // O(1) time complexity, 0 bytes copied

A({5, 5}) = 42.0f;
// A_T({5, 5}) is now also 42.0f
```

### 3. Stable Softmax on Extreme Values
```cpp
Tensor logits({3});
logits({0}) = 1000.0f;
logits({1}) = 1001.0f;
logits({2}) = 999.0f;

// Will correctly output valid probabilities without NaN overflow
Tensor probs = logits.softmax();
```

### 4. Benchmark (The Metric Law)
To measure the performance of the hardware-aware matrix multiplication, benchmark a 500x500 GEMM operation using `<chrono>`.

```cpp
#include <chrono>

Tensor t1({500, 500});
Tensor t2({500, 500});
// (Fill t1 and t2 with 1.0f or random numbers here)

auto start = std::chrono::high_resolution_clock::now();
Tensor result = matmul(t1, t2);
auto end = std::chrono::high_resolution_clock::now();

auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
std::cout << "[BASELINE] Naive MatMul 500x500: " << duration.count() << " milliseconds.\n";
```

## 🏗️ Internal Architecture Notes

* **Flat Data Layout:** Memory is allocated as a single 1D `float*` array (32-byte aligned), wrapped in `std::shared_ptr<float[]>`. Size is calculated dynamically based on the product of the shape dimensions.
* **Stride Vector:** Computed once at construction for row-major layout; `transpose()` reverses both `shape` and `strideVector`, which is what makes arbitrary-dimension transpose a metadata-only operation.
* **Reference Counting:** Buffer lifetime is managed by `shared_ptr`'s atomic refcount (thread-safe by default), removing the need for a hand-rolled refcount.

## 📊 Benchmarks
*(Coming in Block 04 — Lab 1.6: naïve vs. tiled GEMM, Roofline Model, `perf stat` cache-miss data)*