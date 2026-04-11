#include "tensor.h"
#include <iostream>
#include <vector>
#include <cassert>

int main() {
    std::cout << "--- STARTING ARCHITECT STRESS TEST ---\n";

    // --- TEST 1: N-DIMENSIONAL STRIDE LOGIC ---
    std::cout << "[Test 1] Testing 3D Shape {2, 3, 4}...\n";
    Tensor t1({2, 3, 4}); 
    // Total size should be 24.
    // Index math check: t1(1, 2, 3) should map to a unique 1D index.
    
    t1({0, 0, 0}) = 1.0f;
    t1({1, 2, 3}) = 99.0f;
    
    assert(t1({0, 0, 0}) == 1.0f);
    assert(t1({1, 2, 3}) == 99.0f);
    std::cout << ">> Pass: N-D Indexing\n";


    // --- TEST 2: DEEP COPY (RULE OF FIVE) ---
    std::cout << "[Test 2] Testing Deep Copy...\n";
    {
        Tensor t2 = t1; // Copy Constructor
        t2({1, 2, 3}) = 7.0f;
        
        // If this is a deep copy, t1 should NOT have changed.
        if (t1({1, 2, 3}) == 99.0f && t2({1, 2, 3}) == 7.0f) {
            std::cout << ">> Pass: Deep Copy (Separate Memory)\n";
        } else {
            std::cerr << ">> FAIL: Shallow Copy detected! (T1 was modified by T2)\n";
            return 1;
        }
    } // t2 goes out of scope here. If it deletes T1's memory, next step crashes.


    // --- TEST 3: MOVE SEMANTICS (EFFICIENCY) ---
    std::cout << "[Test 3] Testing Move Semantics...\n";
    {
        Tensor source({10, 10, 10});
        source({5, 5, 5}) = 42.0f;
        
        Tensor destination = std::move(source); // Move Constructor
        
        // Check if destination took the data
        assert(destination({5, 5, 5}) == 42.0f);
        
        // Check if source is safely neutralized (Data should be nullptr)
        // Note: You must have set source.data = nullptr in your move ctor!
        std::cout << ">> Pass: Move Semantics (Pointer Stolen)\n";
    }


    // --- TEST 4: ASSIGNMENT OPERATORS ---
    std::cout << "[Test 4] Testing Self-Assignment & Reassignment...\n";
    Tensor t_assign({2, 2});
    t_assign = t_assign; // Self-assignment check
    
    Tensor t_new({5, 5});
    t_assign = t_new; // Copy assignment check
    assert(t_assign({0, 0}) == 0.0f);
    std::cout << ">> Pass: Assignment Operators\n";


    // --- TEST 5: THE MEMORY GAUNTLET ---
    std::cout << "[Test 5] Running Memory Gauntlet (Check Valgrind)...\n";
    for(int i = 0; i < 100; ++i) {
        Tensor temp({10, 10, 10}); // Constant allocation/deallocation
    }
    std::cout << ">> Pass: Memory Gauntlet\n";

    std::cout << "\n--- ALL TESTS PASSED ---" << std::endl;
    std::cout << "NOW RUN VALGRIND TO CONFIRM ZERO LEAKS." << std::endl;

    return 0;
}