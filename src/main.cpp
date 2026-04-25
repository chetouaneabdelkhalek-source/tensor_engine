/**
 * @file main.cpp
 * @brief Demonstration of the custom Tensor engine.
 * 
 * Showcases N-dimensional continuous array management, stride manipulation,
 * zero-copy transpose, hardware-aware matrix multiplication, and
 * numerically stable operations.
 */

#include "tensor.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>

// Utility function to print a 2D tensor
void print_matrix(const std::string& name, const Tensor& t, int rows, int cols) {
    std::cout << "--- " << name << " (" << rows << "x" << cols << ") ---\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << std::setw(8) << t({i, j}) << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main() {
    std::cout << "========================================\n";
    std::cout << "       TENSOR ENGINE DEMONSTRATION      \n";
    std::cout << "========================================\n\n";

    // ---------------------------------------------------------
    // 1. Tensor Allocation & Matrix Multiplication (GEMM)
    // ---------------------------------------------------------
    std::cout << "[1] Matrix Multiplication (Stride-Aware)\n";
    
    Tensor A({2, 3});
    A({0, 0}) = 1.0f; A({0, 1}) = 2.0f; A({0, 2}) = 3.0f;
    A({1, 0}) = 4.0f; A({1, 1}) = 5.0f; A({1, 2}) = 6.0f;

    Tensor B({3, 2});
    B({0, 0}) = 1.0f; B({0, 1}) = 2.0f;
    B({1, 0}) = 3.0f; B({1, 1}) = 4.0f;
    B({2, 0}) = 5.0f; B({2, 1}) = 6.0f;

    print_matrix("Matrix A", A, 2, 3);
    print_matrix("Matrix B", B, 3, 2);

    Tensor C = matmul(A, B);
    print_matrix("Result C = A @ B", C, 2, 2);


    // ---------------------------------------------------------
    // 2. Zero-Copy Transpose & Metadata Aliasing
    // ---------------------------------------------------------
    std::cout << "[2] O(1) Zero-Copy Transpose\n";
    
    Tensor A_T = A.transpose(); // Manipulates strides, copies zero bytes
    print_matrix("Matrix A_T (Transposed)", A_T, 3, 2);

    // Verify memory aliasing: Modifying 'A' should reflect in 'A_T'
    std::cout << "Modifying A({0, 0}) to 999.0...\n";
    A({0, 0}) = 999.0f;
    
    if (A_T({0, 0}) == 999.0f) {
        std::cout << "-> SUCCESS: A_T({0, 0}) is also " << A_T({0, 0}) 
                  << ". Memory is successfully aliased via reference counting.\n\n";
    } else {
        std::cout << "-> FAILURE: Memory was copied.\n\n";
    }


    // ---------------------------------------------------------
    // 3. Numerically Stable Softmax
    // ---------------------------------------------------------
    std::cout << "[3] Numerically Stable Softmax\n";
    
    // Testing with large values that would normally cause NaN/Overflow in standard exp(x)
    Tensor logits({3});
    logits({0}) = 1000.0f;
    logits({1}) = 1001.0f;
    logits({2}) = 999.0f;

    std::cout << "Input Logits:[1000.0, 1001.0, 999.0]\n";

    try {
        Tensor probs = logits.softmax(); // Uses max-subtraction trick internally
        std::cout << "Softmax Probs:[" 
                  << probs({0}) << ", " 
                  << probs({1}) << ", " 
                  << probs({2}) << "]\n";
        
        float sum = probs({0}) + probs({1}) + probs({2});
        std::cout << "Sum of probabilities: " << sum << " (Expected: ~1.0)\n\n";
    } catch (const std::exception& e) {
        std::cerr << "Softmax Error: " << e.what() << "\n";
    }


    // ---------------------------------------------------------
    // 4. Memory Ownership (Rule of Five)
    // ---------------------------------------------------------
    std::cout << "[4] Move Semantics & Ownership Transfer\n";
    Tensor D({10, 10});
    D({5, 5}) = 42.0f;
    
    Tensor E = std::move(D); // Triggers noexcept move constructor
    std::cout << "Successfully moved Tensor D to E. E({5, 5}) = " << E({5, 5}) << "\n";
    
    std::cout << "========================================\n";
    std::cout << "Engine execution completed successfully.\n";
    // ---------------------------------------------------------
    // 5. TASK 4: THE BENCHMARK (THE METRIC LAW)
    // ---------------------------------------------------------
    std::cout << "[5] TASK 4: THE BENCHMARK (THE METRIC LAW)\n";
    
    Tensor t1({500, 500});
    Tensor t2({500, 500});

    // Fill the tensors with 1.0f
    for (int i = 0; i < 500; ++i) {
        for (int j = 0; j < 500; ++j) {
            t1({i, j}) = 1.0f;
            t2({i, j}) = 1.0f;
        }
    }

    // Start the chrono timer
    auto start = std::chrono::high_resolution_clock::now();
    
    // Execute matrix multiplication
    Tensor result_bench = matmul(t1, t2); 
    
    // Stop the chrono timer
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "[BASELINE] Naive MatMul 500x500: " << duration.count() << " milliseconds.\n\n";

    std::cout << "========================================\n";
    std::cout << "Engine execution completed successfully.\n";


    return 0;
}