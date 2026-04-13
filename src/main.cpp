#include <iostream>
#include <vector>
#include "tensor.h"

int main() {
    try {
        std::cout << "--- Testing Tensor Construction ---" << std::endl;
        // Create a 2x3 Tensor: [ [1, 2, 3], [4, 5, 6] ]
        Tensor t1({2, 3});
        t1({0, 0}) = 1.0f; t1({0, 1}) = 2.0f; t1({0, 2}) = 3.0f;
        t1({1, 0}) = 4.0f; t1({1, 1}) = 5.0f; t1({1, 2}) = 6.0f;

        std::cout << "Original Tensor (0,1): " << t1({0, 1}) << " (Expected 2.0)" << std::endl;

        std::cout << "\n--- Testing Transpose (The Alias Test) ---" << std::endl;
        Tensor t_trans = t1.transpose();
        
        // In transposed 3x2, (1, 0) should be original (0, 1)
        std::cout << "Transposed (1,0): " << t_trans({1, 0}) << " (Expected 2.0)" << std::endl;

        // MODIFY THE TRANSPOSE - THE AUDIT TEST
        t_trans({1, 0}) = 99.0f; 
        if (t1({0, 1}) == 99.0f) {
            std::cout << "PASS: Modifying transpose modified original (Alias working!)" << std::endl;
        } else {
            std::cout << "FAIL: Modifying transpose did NOT modify original (Data copied!)" << std::endl;
        }

        std::cout << "\n--- Testing Deep Copy ---" << std::endl;
        Tensor t_copy = t1; // Calls Copy Constructor
        t_copy({0, 0}) = -1.0f;
        if (t1({0, 0}) == 1.0f) {
            std::cout << "PASS: Copy is independent (Deep copy working!)" << std::endl;
        } else {
            std::cout << "FAIL: Copy is sharing memory!" << std::endl;
        }

        std::cout << "\n--- Testing N-Dimensional (3D) Transpose ---" << std::endl;
        // 2x3x4 -> 4x3x2
        Tensor t3d({2, 3, 4});
        Tensor t3d_trans = t3d.transpose();
        std::cout << "PASS: 3D Transpose created successfully." << std::endl;

        std::cout << "\n--- Testing Move Semantics ---" << std::endl;
        Tensor t_mover = std::move(t1);
        std::cout << "PASS: Move completed without crash." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "CRASH/ERROR: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\n--- Cleanup Test ---" << std::endl;
    std::cout << "If the program finishes now without 'double free' errors, ownership logic is correct." << std::endl;

    return 0;
}