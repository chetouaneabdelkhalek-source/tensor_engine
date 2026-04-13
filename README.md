# 🏛️ TensorCore: High-Performance N-Dimensional Infrastructure

**Candidate:** Systems Architect (AI Infrastructure / DL Systems)  
**Status:** Phase 1 Bedrock Complete  
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

### 3. Rule of Five (Memory Safety)
To ensure system stability and high-performance data movement, I implemented the full **C++ Rule of Five**:
*   **Deep Copy:** Copy Constructor & Assignment prevent double-free errors.
*   **Move Semantics:** Move Constructor & Assignment utilize pointer stealing (`noexcept`) to transfer ownership of massive data blocks with zero-copy overhead.

## 🚀 Performance & Stability
*   **Memory Safety:** Validated via **Valgrind**. All heap blocks are freed, and 0 memory leaks are possible.
*   **Bounds Checking:** Strict `std::out_of_range` validation for coordinate dimensions and tensor rank.
*   **Zero-Fill Initialization:** All tensors are value-initialized to zero to prevent garbage-value computation.

## 📂 Project Structure
```text
.
├── include/
│   └── tensor.h      # Class blueprint & API
├── src/
│   ├── tensor.cpp    # Implementation & Stride Logic
│   └── main.cpp      # Stress tests & Benchmarks
├── build/            # CMake artifacts
└── CMakeLists.txt    # Build system
```

## ⚙️ Building and Testing

**Requirements:** `cmake`, `g++`, `valgrind`

1. **Configure and Build:**
   ```bash
   mkdir build && cd build
   cmake ..
   make
   ```

2. **Run Stress Test:**
   ```bash
   ./tensor_run
   ```

3. **Verify Memory Integrity:**
   ```bash
   valgrind --leak-check=full --track-origins=yes ./tensor_run
   ```

## 🗺️ Roadmap
- [x] **Block 1:** N-Dimensional Stride Math & Rule of Five.
- [x] **Block 2:** 'O(1)' Zero-Copy Transpose.
- [ ] **Block 3:** Circular Buffers & Numerically Stable Softmax.
- [ ] **Block 4:** Cache-friendly GEMM (General Matrix Multiply).

***