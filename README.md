# C++ Tensor Engine

A lightweight, zero-dependency C++ Tensor engine built entirely from scratch. This project is an exercise in low-level systems engineering, focusing on contiguous memory management, stride manipulation, hardware-aware computation, and numerical stability.

## 🚀 Engineering Features

### 1. O(1) Zero-Copy Transpose
Matrix transpositions do not copy any underlying data. 
* **Implementation:** The `transpose()` method simply reverses the `shape` and `strideVector` arrays. 
* **Memory Aliasing:** A custom non-atomic reference counter (`alias_num`) ensures the new transposed `Tensor` safely shares the exact same `float* data` buffer as the original, preventing deep copies and saving memory bandwidth.

### 2. Hardware-Aware Matmul (GEMM)
The `matmul` friend function includes a dynamic branching optimization to maximize CPU cache locality and compiler auto-vectorization.
* **SIMD Fast Lane:** It checks if the inner dimensions are contiguous in memory (`Bstride1 == 1 && Cstride1 == 1`). If so, it executes a contiguous loop that the compiler can easily auto-vectorize (e.g., using AVX instructions).
* **Safe Fallback:** If the tensors are strided or transposed views, it falls back to a stride-aware access pattern.

### 3. Numerically Stable Softmax
A naive $e^x$ implementation fails with `NaN` on large inputs due to floating-point overflow. 
* **Max-Subtraction Trick:** The `softmax()` method computes the maximum value in the tensor first, then computes $e^{x_i - max}$. This keeps the maximum exponent at $e^0 = 1$, guaranteeing absolute numerical stability even with extreme logits (e.g., `1000.0` or `-1000.0`).

### 4. Memory Ownership & The Rule of Five
The engine eschews `std::vector` for its data buffers, opting for raw `float*` pointers to maintain absolute control over heap allocations.
* **Leak-Free:** Fully implements the **Rule of Five** (Destructor, Copy Constructor, Copy Assignment, Move Constructor, Move Assignment). 
* **Move Semantics:** Utilizes `noexcept` move constructors to efficiently steal pointers from temporary r-values, avoiding expensive reallocations.

---

## 🛠️ Build and Execution

This project requires a standard C++11 (or higher) compiler. 

**Optimization Flag:** It is highly recommended to compile with `-O2` or `-O3`. This allows the compiler to unroll loops and apply vectorization to the `matmul` fast-lane.

### Compile
```bash
makdir build
cd build 
make
```

### Run
```bash
./tensor_engine
```

### Validate Memory Safety
The custom reference counting engine is designed to be strictly leak-free. You can verify this using Valgrind:
```bash
valgrind --leak-check=full ./tensor_engine
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
Because `transpose()` shares memory, modifying the original tensor modifies the transposed view.
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

## 🏗️ Internal Architecture Notes

* **Flat Data Layout:** Memory is allocated as a single 1D `float*` array. Size is calculated dynamically based on the product of the shape dimensions.
* **Single-Threaded Context:** As noted in `tensor.h`, the current `alias_num` is a standard `int*`. It is optimized for single-threaded execution. For a multi-threaded context, this would easily be swapped to `std::atomic<int>`.