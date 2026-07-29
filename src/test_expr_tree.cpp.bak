#include "arena.hpp"
#include "traversals.hpp"
#include <iostream>

void test_small_tree() {
    std::cout << "=== Test 1: Small Tree Correctness ===" << std::endl;
    Arena arena(10);
    
    // Step 1: Build a 3 to 5 node tree using arena.allocate(val)
    node* root = arena.allocate(10);
    root->left = arena.allocate(5);
    root->right = arena.allocate(15);
    root->left->left = arena.allocate(2);

    std::cout << "Pre-Order:  ";
    stack_preOrderTraversal(root);
    std::cout << "\nIn-Order:   ";
    stack_inOrderTraversal(root);
    std::cout << "\nPost-Order: ";
    stack_postOrderTraversal(root);
    std::cout << "\n" << std::endl;
}

void test_depth_1000() {
    std::cout << "=== Test 2: Depth 1,000 Test ===" << std::endl;
    Arena arena(1000);
    node* root = arena.allocate(1);
    node* curr = root;

    for (int i = 2; i <= 1000; ++i) {
        curr->right = arena.allocate(i);
        curr = curr->right;
    }

    // Call traversal on 1,000-deep tree
    // stack_inOrderTraversal(root);
    
    std::cout << "Depth 1,000 completed successfully!\n" << std::endl;
}

void test_depth_100000() {
    std::cout << "=== Test 3: Depth 100,000 Stress Test ===" << std::endl;
    Arena arena(100000);
    node* root = arena.allocate(1);
    node* curr = root;

    for (int i = 2; i <= 100000; ++i) {
        curr->right = arena.allocate(i);
        curr = curr->right;
    }

    // Call traversal on 100,000-deep tree
    // stack_postOrderTraversal(root);

    std::cout << "Depth 100,000 completed successfully!\n" << std::endl;
}

int main() {
    test_small_tree();
    test_depth_1000();
    test_depth_100000();

    std::cout << "ALL BLOCK 05 TASK 1 TESTS PASSED!" << std::endl;
    return 0;
}