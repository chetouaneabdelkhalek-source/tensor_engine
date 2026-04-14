# 🏛️ TensorCore: High-Performance N-Dimensional Infrastructure

**Candidate:** Systems Architect (AI Infrastructure / DL Systems)  
**Status:** Block 2 Active (Zero-Copy Mechanics Validated)  
**Engine:** C++17  

## 🎯 Overview
TensorCore is a custom N-Dimensional tensor engine built from the ground up in C++. Unlike standard high-level libraries, TensorCore focuses on the **Silicon-Software interface**, utilizing contiguous memory layouts and manual pointer management to mimic the internals of production-grade frameworks like PyTorch (ATen) and GGML.

## 🛠️ Architectural Decisions

### 1. Contiguous Memory Layout (Cache Locality)
Instead of using nested vectors (e.g., `vector<vector<float>>`), which causes "pointer chasing" and cache misses, TensorCore utilizes a **flat 1D array on the heap**. 
*   **Impact:** This ensures that data is stored in contiguous memory blocks, respecting CPU cache lines and enabling future SIMD (Single Instruction, Multiple Data) vectorization.

### 2. N-Dimensional Stride Math
The engine supports arbitrary dimensions (Rank-N). Mapping multi-dimensional coordinates to the underlying 1D memory is handled via **Stride Logic**:
$$Index = \sum_{i=0}^{n-1} (coords_i \times stride_i)$$
Where strides are pre-calculated during construction to minimize runtime overhead.

### 3. Memory Safety & Reference Counting
To ensure system stability under heavy algorithmic loads, memory ownership is strictly managed:
*   **Rule of Five:** Deep copies are explicitly handled, and move semantics utilize pointer-stealing (`noexcept`) for maximum efficiency.
*   **Manual Reference Counting:** The engine utilizes a custom `alias_num` tracker. This allows multiple Tensor objects to safely alias the same underlying `float* data` array without triggering double-free crashes during destruction.

### 4. Zero-Copy Operations
*   **$O(1)$ Transpose:** Transposition is achieved via metadata manipulation (swapping shape and stride vectors) rather than migrating physical floats in RAM. A private constructor bypasses `new` allocations, preventing OS-level heap fragmentation.

## 🚀 Performance & Stability
*   **Memory Integrity:** Validated via **Valgrind**. All tests confirm 0 memory leaks and complete mitigation of uninitialized values.
*   **Bounds Checking:** Strict `std::out_of_range` validation for coordinate dimensions and tensor rank.

## 📂 Project Structure
```text
.
├── include/
│   └── tensor.h      # Class blueprint, Private Constructors & API
├── src/
│   ├── tensor.cpp    # Implementation, Stride Logic, Ref Counting
│   └── main.cpp      # Stress tests & Benchmarks
├── build/            # CMake artifacts
└── CMakeLists.txt    # Build system
```

## ⚙️ Building and Testing
**Requirements:** `cmake`, `g++`, `valgrind`

```bash
mkdir build && cd build
cmake ..
make
valgrind --leak-check=full --track-origins=yes ./tensor_run
```

## 🗺️ Roadmap
- [x] **Block 1:** N-Dimensional Stride Math & Rule of Five.
- [x] **Block 2:** $O(1)$ Zero-Copy Transpose & Reference Counting.
- [ ] **Block 3:** Circular Buffers & Numerically Stable Softmax.
- [ ] **Block 4:** Cache-friendly GEMM (General Matrix Multiply).
```
