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
#include <cassert>
#include <cmath>

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────

void print_matrix(const std::string &name, const Tensor &t, int rows, int cols)
{
    std::cout << "--- " << name << " (" << rows << "x" << cols << ") ---\n";
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
            std::cout << std::setw(10) << std::fixed << std::setprecision(4) << t({i, j}) << " ";
        std::cout << "\n";
    }
    std::cout << "\n";
}

void print_vector(const std::string &name, const Tensor &t, int n)
{
    std::cout << "--- " << name << " ---\n[";
    for (int i = 0; i < n; ++i)
    {
        std::cout << std::fixed << std::setprecision(6) << t({i});
        if (i < n - 1)
            std::cout << ", ";
    }
    std::cout << "]\n\n";
}

void section(const std::string &title)
{
    std::cout << "\n========================================\n";
    std::cout << "  " << title << "\n";
    std::cout << "========================================\n";
}

bool nearly_equal(float a, float b, float eps = 1e-4f)
{
    return std::fabs(a - b) < eps;
}

// ─────────────────────────────────────────────
// Test 1 — Basic allocation & element access
// ─────────────────────────────────────────────
void test_allocation()
{
    section("1. Allocation & Element Access");

    Tensor A({3, 3});
    // Fill with identity
    for (int i = 0; i < 3; ++i)
        A({i, i}) = 1.0f;

    print_matrix("3x3 Identity", A, 3, 3);

    assert(A({0, 0}) == 1.0f);
    assert(A({1, 1}) == 1.0f);
    assert(A({2, 2}) == 1.0f);
    assert(A({0, 1}) == 0.0f);
    std::cout << "[PASS] Element access correct.\n";
}

// ─────────────────────────────────────────────
// Test 2 — Transpose & memory aliasing
// ─────────────────────────────────────────────
void test_transpose()
{
    section("2. Zero-Copy Transpose & Memory Aliasing");

    Tensor A({2, 3});
    A({0, 0}) = 1;
    A({0, 1}) = 2;
    A({0, 2}) = 3;
    A({1, 0}) = 4;
    A({1, 1}) = 5;
    A({1, 2}) = 6;

    Tensor A_T = A.transpose();
    print_matrix("A   (2x3)", A, 2, 3);
    print_matrix("A^T (3x2)", A_T, 3, 2);

    assert(nearly_equal(A_T({0, 0}), 1.0f));
    assert(nearly_equal(A_T({1, 0}), 2.0f));
    assert(nearly_equal(A_T({2, 0}), 3.0f));
    assert(nearly_equal(A_T({0, 1}), 4.0f));

    A({0, 0}) = 999.0f;
    if (nearly_equal(A_T({0, 0}), 999.0f))
        std::cout << "[PASS] Memory aliased correctly — A_T({0,0}) = " << A_T({0, 0}) << "\n";
    else
        std::cout << "[FAIL] Memory was copied instead of aliased.\n";
}

// ─────────────────────────────────────────────
// Test 3 — Matmul: all cases
// ─────────────────────────────────────────────
void test_matmul()
{
    section("3. Matrix Multiplication — All Cases");

    // ── Case A: normal × normal (contiguous × contiguous) ──────────────
    std::cout << "\n[Case A] Normal x Normal  (contiguous B → SIMD fast path)\n";
    {
        Tensor A({2, 3});
        A({0, 0}) = 1;
        A({0, 1}) = 2;
        A({0, 2}) = 3;
        A({1, 0}) = 4;
        A({1, 1}) = 5;
        A({1, 2}) = 6;

        Tensor B({3, 2});
        B({0, 0}) = 1;
        B({0, 1}) = 2;
        B({1, 0}) = 3;
        B({1, 1}) = 4;
        B({2, 0}) = 5;
        B({2, 1}) = 6;

        print_matrix("A (2x3)", A, 2, 3);
        print_matrix("B (3x2)", B, 3, 2);
        Tensor C = matmul(A, B);
        print_matrix("C = A @ B", C, 2, 2);

        // Expected [[22,28],[49,64]]
        assert(nearly_equal(C({0, 0}), 22.0f));
        assert(nearly_equal(C({0, 1}), 28.0f));
        assert(nearly_equal(C({1, 0}), 49.0f));
        assert(nearly_equal(C({1, 1}), 64.0f));
        std::cout << "[PASS] Normal x Normal correct.\n";
    }

    // ── Case B: normal × transposed  (non-contiguous B → safe fallback) ──
    std::cout << "\n[Case B] Normal x Transposed  (non-contiguous B → safe fallback path)\n";
    {
        Tensor A({2, 3});
        A({0, 0}) = 1;
        A({0, 1}) = 2;
        A({0, 2}) = 3;
        A({1, 0}) = 4;
        A({1, 1}) = 5;
        A({1, 2}) = 6;

        // B = A^T  shape (3×2), but we want to multiply A(2×3) × BT(3×2)
        // so we need a (3×2) matrix transposed to (2×3) ...
        // simpler: build C(3×2) then transpose to get non-contiguous (2×3) won't match dims
        // Correct: A(2×3) @ AT(3×2) where AT = A^T
        Tensor AT = A.transpose(); // shape (3×2), non-contiguous strides
        print_matrix("A  (2x3)", A, 2, 3);
        print_matrix("AT (3x2) [transposed, non-contiguous]", AT, 3, 2);
        Tensor C = matmul(A, AT);
        print_matrix("C = A @ A^T (2x2)", C, 2, 2);

        // Expected [[14,32],[32,77]]
        assert(nearly_equal(C({0, 0}), 14.0f));
        assert(nearly_equal(C({0, 1}), 32.0f));
        assert(nearly_equal(C({1, 0}), 32.0f));
        assert(nearly_equal(C({1, 1}), 77.0f));
        std::cout << "[PASS] Normal x Transposed correct.\n";
    }

    // ── Case C: transposed × normal ────────────────────────────────────
    std::cout << "\n[Case C] Transposed x Normal  (non-contiguous A, contiguous B)\n";
    {
        // A^T (3×2) @ B (2×4)
        Tensor A({2, 3});
        A({0, 0}) = 1;
        A({0, 1}) = 2;
        A({0, 2}) = 3;
        A({1, 0}) = 4;
        A({1, 1}) = 5;
        A({1, 2}) = 6;
        Tensor AT = A.transpose(); // (3×2)

        Tensor B({2, 4});
        B({0, 0}) = 1;
        B({0, 1}) = 0;
        B({0, 2}) = 2;
        B({0, 3}) = 1;
        B({1, 0}) = 0;
        B({1, 1}) = 1;
        B({1, 2}) = 1;
        B({1, 3}) = 3;

        print_matrix("AT (3x2) [non-contiguous]", AT, 3, 2);
        print_matrix("B  (2x4)", B, 2, 4);
        Tensor C = matmul(AT, B);
        print_matrix("C = AT @ B (3x4)", C, 3, 4);

        // Verify with naive
        Tensor C_ref = matmul_naive(AT, B);
        bool ok = true;
        for (int i = 0; i < 3 && ok; ++i)
            for (int j = 0; j < 4 && ok; ++j)
                ok = nearly_equal(C({i, j}), C_ref({i, j}));
        std::cout << (ok ? "[PASS]" : "[FAIL]") << " Transposed x Normal correct.\n";
    }

    // ── Case D: transposed × transposed ────────────────────────────────
    std::cout << "\n[Case D] Transposed x Transposed  (both non-contiguous)\n";
    {
        // AT(3×2) @ BT(2×3)  →  result (3×3)
        Tensor A({2, 3});
        A({0, 0}) = 1;
        A({0, 1}) = 2;
        A({0, 2}) = 3;
        A({1, 0}) = 4;
        A({1, 1}) = 5;
        A({1, 2}) = 6;

        Tensor B({3, 2});
        B({0, 0}) = 7;
        B({0, 1}) = 8;
        B({1, 0}) = 9;
        B({1, 1}) = 10;
        B({2, 0}) = 11;
        B({2, 1}) = 12;

        Tensor AT = A.transpose(); // (3×2)
        Tensor BT = B.transpose(); // (2×3)

        print_matrix("AT (3x2) [non-contiguous]", AT, 3, 2);
        print_matrix("BT (2x3) [non-contiguous]", BT, 2, 3);
        Tensor C = matmul(AT, BT);
        print_matrix("C = AT @ BT (3x3)", C, 3, 3);

        Tensor C_ref = matmul_naive(AT, BT);
        bool ok = true;
        for (int i = 0; i < 3 && ok; ++i)
            for (int j = 0; j < 3 && ok; ++j)
                ok = nearly_equal(C({i, j}), C_ref({i, j}));
        std::cout << (ok ? "[PASS]" : "[FAIL]") << " Transposed x Transposed correct.\n";
    }

    // ── Case E: square identity ─────────────────────────────────────────
    std::cout << "\n[Case E] A @ I = A  (identity check)\n";
    {
        Tensor A({3, 3});
        A({0, 0}) = 2;
        A({0, 1}) = 3;
        A({0, 2}) = 1;
        A({1, 0}) = 0;
        A({1, 1}) = 5;
        A({1, 2}) = 4;
        A({2, 0}) = 7;
        A({2, 1}) = 1;
        A({2, 2}) = 9;

        Tensor I({3, 3});
        for (int i = 0; i < 3; ++i)
            I({i, i}) = 1.0f;

        Tensor C = matmul(A, I);
        bool ok = true;
        for (int i = 0; i < 3 && ok; ++i)
            for (int j = 0; j < 3 && ok; ++j)
                ok = nearly_equal(C({i, j}), A({i, j}));
        std::cout << (ok ? "[PASS]" : "[FAIL]") << " A @ I = A.\n";
    }

    // ── Case F: non-square tall × wide ──────────────────────────────────
    std::cout << "\n[Case F] Tall x Wide  (4x1) @ (1x4) → outer product (4x4)\n";
    {
        Tensor col({4, 1});
        col({0, 0}) = 1;
        col({1, 0}) = 2;
        col({2, 0}) = 3;
        col({3, 0}) = 4;

        Tensor row({1, 4});
        row({0, 0}) = 1;
        row({0, 1}) = 2;
        row({0, 2}) = 3;
        row({0, 3}) = 4;

        Tensor C = matmul(col, row);
        print_matrix("outer product (4x4)", C, 4, 4);

        // C[i][j] = (i+1)*(j+1)
        bool ok = true;
        for (int i = 0; i < 4 && ok; ++i)
            for (int j = 0; j < 4 && ok; ++j)
                ok = nearly_equal(C({i, j}), float((i + 1) * (j + 1)));
        std::cout << (ok ? "[PASS]" : "[FAIL]") << " Outer product correct.\n";
    }
}

// ─────────────────────────────────────────────
// Test 5 — Numerically stable softmax
// ─────────────────────────────────────────────
void test_softmax()
{
    section("5. Numerically Stable Softmax");

    // Large values → naive exp() would overflow to inf
    Tensor logits({3});
    logits({0}) = 1000.0f;
    logits({1}) = 1001.0f;
    logits({2}) = 999.0f;

    print_vector("Logits [1000, 1001, 999]", logits, 3);

    Tensor probs = logits.softmax();
    print_vector("Softmax output", probs, 3);

    float sum = probs({0}) + probs({1}) + probs({2});
    std::cout << "Sum of probabilities: " << std::fixed << std::setprecision(8) << sum << " (expected ~1.0)\n";

    assert(nearly_equal(sum, 1.0f, 1e-5f));
    assert(probs({1}) > probs({0})); // highest logit → highest prob
    assert(probs({0}) > probs({2}));
    std::cout << "[PASS] Softmax is numerically stable and sums to 1.\n";

    // Edge: uniform logits → uniform probs
    Tensor uniform({4});
    for (int i = 0; i < 4; ++i)
        uniform({i}) = 2.0f;
    Tensor uprobs = uniform.softmax();
    for (int i = 0; i < 4; ++i)
        assert(nearly_equal(uprobs({i}), 0.25f));
    std::cout << "[PASS] Uniform logits → uniform probabilities (0.25 each).\n";
}

// ─────────────────────────────────────────────
// Test 6 — Rule of Five
// ─────────────────────────────────────────────
void test_rule_of_five()
{
    section("6. Rule of Five — Copy, Move, Assign");

    Tensor original({2, 2});
    original({0, 0}) = 1;
    original({0, 1}) = 2;
    original({1, 0}) = 3;
    original({1, 1}) = 4;

    // Copy constructor — deep copy
    Tensor copied(original);
    original({0, 0}) = 99.0f;
    assert(nearly_equal(copied({0, 0}), 1.0f)); // must be independent
    std::cout << "[PASS] Copy constructor: deep copy confirmed.\n";

    // Copy assignment
    Tensor assigned({1, 1});
    assigned = copied;
    copied({0, 0}) = 77.0f;
    assert(nearly_equal(assigned({0, 0}), 1.0f));
    std::cout << "[PASS] Copy assignment: deep copy confirmed.\n";

    // Move constructor
    Tensor source({3, 3});
    source({1, 1}) = 42.0f;
    Tensor moved(std::move(source));
    assert(nearly_equal(moved({1, 1}), 42.0f));
    std::cout << "[PASS] Move constructor: ownership transferred.\n";

    // Move assignment
    Tensor target({1, 1});
    Tensor temp({2, 2});
    temp({0, 1}) = 7.0f;
    target = std::move(temp);
    assert(nearly_equal(target({0, 1}), 7.0f));
    std::cout << "[PASS] Move assignment: ownership transferred.\n";
}

// ─────────────────────────────────────────────
// Test 7 — Bounds checking
// ─────────────────────────────────────────────
void test_bounds()
{
    section("7. Bounds & Rank Checking");

    Tensor A({3, 3});

    bool caught_oob = false;
    try
    {
        A({5, 0});
    }
    catch (const std::out_of_range &)
    {
        caught_oob = true;
    }
    assert(caught_oob);
    std::cout << "[PASS] Out-of-bounds index throws std::out_of_range.\n";

    bool caught_rank = false;
    try
    {
        A({0, 0, 0});
    }
    catch (const std::out_of_range &)
    {
        caught_rank = true;
    }
    assert(caught_rank);
    std::cout << "[PASS] Wrong rank throws std::out_of_range.\n";

    bool caught_dim = false;
    try
    {
        matmul(Tensor({2, 3}), Tensor({2, 3}));
    }
    catch (const std::invalid_argument &)
    {
        caught_dim = true;
    }
    assert(caught_dim);
    std::cout << "[PASS] Dimension mismatch in matmul throws std::invalid_argument.\n";
}

// ─────────────────────────────────────────────
// Benchmark helper
// ─────────────────────────────────────────────
// ── Correctness: compare matmul against hand-computed expected values ──
// Small inputs → exact integer results → immune to -ffast-math reordering.
bool verify_exact(const Tensor &A, const Tensor &B,
                  const std::vector<std::vector<float>> &expected)
{
    Tensor C = matmul(A, B);
    for (int i = 0; i < (int)expected.size(); ++i)
        for (int j = 0; j < (int)expected[0].size(); ++j)
            if (!nearly_equal(C({i, j}), expected[i][j], 1e-2f))
                return false;
    return true;
}

struct BenchResult
{
    long fast_ms;
    long naive_ms;
};

BenchResult run_bench(const std::string &label, const Tensor &A, const Tensor &B)
{
    std::cout << "\n  [" << label << "]\n";

    auto start = std::chrono::high_resolution_clock::now();
    Tensor r1 = matmul(A, B);
    auto end = std::chrono::high_resolution_clock::now();
    long fast_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "    stride-aware : " << std::setw(6) << fast_ms << " ms\n";

    start = std::chrono::high_resolution_clock::now();
    Tensor r2 = matmul_naive(A, B);
    end = std::chrono::high_resolution_clock::now();
    long naive_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "    naive        : " << std::setw(6) << naive_ms << " ms\n";

    if (fast_ms < naive_ms)
        std::cout << "    speedup      : " << naive_ms - fast_ms << " ms faster\n";

    return {fast_ms, naive_ms};
}

// ─────────────────────────────────────────────
// Test 8 — Overall benchmark (all stride cases)
// ─────────────────────────────────────────────
void test_benchmark()
{
    section("8. Overall Benchmark — All Stride Cases (500x500)");

    const int N = 500;

    Tensor A({N, N}), B({N, N});
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
        {
            A({i, j}) = static_cast<float>(i + j + 1);
            B({i, j}) = static_cast<float>(i - j + 1);
        }
    Tensor AT = A.transpose();
    Tensor BT = B.transpose();

    // ── Correctness on small exact inputs ──────────────────────────────
    // (small integer values → no fp accumulation error regardless of path)
    std::cout << "\n  Correctness check (small 3x3, exact integer results):\n";
    {
        Tensor a({2, 3}), b({3, 2}), at({2, 3}), bt({3, 2});
        a({0, 0}) = 1;
        a({0, 1}) = 2;
        a({0, 2}) = 3;
        a({1, 0}) = 4;
        a({1, 1}) = 5;
        a({1, 2}) = 6;
        b({0, 0}) = 1;
        b({0, 1}) = 2;
        b({1, 0}) = 3;
        b({1, 1}) = 4;
        b({2, 0}) = 5;
        b({2, 1}) = 6;
        Tensor aT = a.transpose(); // (3x2)
        Tensor bT = b.transpose(); // (2x3)

        // Normal x Normal  → [[22,28],[49,64]]
        bool c1 = verify_exact(a, b, {{22, 28}, {49, 64}});
        // Normal x Transposed  a(2x3) @ bT(3x2) wait bT is (2x3), wrong dims
        // Use aT(3x2) for second arg: a(2x3) @ aT... aT is (3x2) ✓ → [[14,32],[32,77]]
        bool c2 = verify_exact(a, aT, {{14, 32}, {32, 77}});
        // Transposed x Normal  aT(3x2) @ b(3x2) — dim mismatch, use bT(2x3)
        // aT(3x2) @ b(3x2) doesn't work. Need aT(3x2) @ something(2,N).
        // Use: aT(3x2) @ bT(2x3) → (3x3), expected [[39,49,59],[54,68,82],[69,87,105]]
        bool c3 = verify_exact(aT, bT, {{9, 19, 29}, {12, 26, 40}, {15, 33, 51}});
        // Transposed x Normal: aT(3x2) @ b_small(2x2)
        Tensor b2({2, 2});
        b2({0, 0}) = 1;
        b2({0, 1}) = 0;
        b2({1, 0}) = 0;
        b2({1, 1}) = 1;
        bool c4 = verify_exact(aT, b2, {{1, 4}, {2, 5}, {3, 6}}); // aT @ I = aT

        std::cout << "    Normal   x Normal      : " << (c1 ? "PASS" : "FAIL") << "\n";
        std::cout << "    Normal   x Transposed  : " << (c2 ? "PASS" : "FAIL") << "\n";
        std::cout << "    Transposed x Transposed: " << (c3 ? "PASS" : "FAIL") << "\n";
        std::cout << "    Transposed x Normal    : " << (c4 ? "PASS" : "FAIL") << "\n";
    }

    // ── Timing on 500x500 ─────────────────────────────────────────────
    std::cout << "\n  Timing (500x500, Release build):\n";
    std::cout << std::left;

    auto r1 = run_bench("Normal   x Normal    [SIMD path]", A, B);
    auto r2 = run_bench("Normal   x Transposed [safe path]", A, BT);
    auto r3 = run_bench("Transposed x Normal  [SIMD path]", AT, B);
    auto r4 = run_bench("Transposed x Transposed [safe path]", AT, BT);

    // ── Summary table ─────────────────────────────────────────────────
    std::cout << "\n  ┌─────────────────────────────────┬──────────┬──────────┐\n";
    std::cout << "  │ Case                            │ Fast(ms) │Naive(ms) │\n";
    std::cout << "  ├─────────────────────────────────┼──────────┼──────────┤\n";

    auto row = [](const std::string &name, const BenchResult &r)
    {
        std::cout << "  │ " << std::setw(31) << std::left << name
                  << " │ " << std::setw(8) << r.fast_ms
                  << " │ " << std::setw(8) << r.naive_ms << " │\n";
    };

    row("Normal x Normal", r1);
    row("Normal x Transposed", r2);
    row("Transposed x Normal", r3);
    row("Transposed x Transposed", r4);

    std::cout << "  └─────────────────────────────────┴──────────┴──────────┘\n";
}
void test_softmax_compare()
{
    section("SOFTMAX: Stable vs Naive (NaN Demo)");

    Tensor logits({3});
    logits({0}) = 1000.0f;
    logits({1}) = 1001.0f;
    logits({2}) = 999.0f;

    std::cout << "Input: [1000.0, 1001.0, 999.0]\n\n";

    // Naive — will produce NaN/inf
    Tensor naive = logits.softmax_naive();
    std::cout << "[NAIVE]  exp(1000) = " << std::exp(1000.0f) << " → overflows to inf\n";
    std::cout << "[NAIVE]  result: ["
              << naive({0}) << ", "
              << naive({1}) << ", "
              << naive({2}) << "]\n";
    std::cout << "[NAIVE]  sum: " << naive({0}) + naive({1}) + naive({2}) << "\n\n";

    // Stable
    Tensor stable = logits.softmax();
    std::cout << "[STABLE] subtracts max (1001) before exp → no overflow\n";
    std::cout << "[STABLE] result: ["
              << stable({0}) << ", "
              << stable({1}) << ", "
              << stable({2}) << "]\n";
    std::cout << "[STABLE] sum: " << stable({0}) + stable({1}) + stable({2}) << "\n";
}
// ─────────────────────────────────────────────
// Test 9 — Tiled GEMM correctness
// ─────────────────────────────────────────────
// ─────────────────────────────────────────────
// Test 9 — Tiled GEMM correctness
// ─────────────────────────────────────────────
void test_tiled_matmul()
{
    section("9. Tiled GEMM Correctness");

    // ── Case A: Square N=256, TILE=64 ────────────────────────────────
    std::cout << "\n[Case A] Square 256x256, TILE=64\n";
    {
        const int N = 256;
        Tensor A({N, N}), B({N, N});
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
            {
                A({i, j}) = 1.0f;
                B({i, j}) = 1.0f;
            }

        Tensor C_naive = matmul_naive(A, B);
        Tensor C_tiled = matmul_tiled(A, B, 64);

        bool match = true;
        for (int i = 0; i < N && match; ++i)
            for (int j = 0; j < N && match; ++j)
                if (!nearly_equal(C_naive({i, j}), C_tiled({i, j}), 1e-4f))
                {
                    match = false;
                    std::cout << "Mismatch at (" << i << "," << j << "): naive=" << C_naive({i, j}) << " tiled=" << C_tiled({i, j}) << "\n";
                }

        if (match)
        {
            float expected = static_cast<float>(N);
            std::cout << "[PASS] Tiled matches naive. C[0,0] = " << C_tiled({0, 0}) << " (expected " << expected << ")\n";
        }
        else
        {
            std::cout << "[FAIL] Tiled output does not match naive.\n";
        }
    }

    // ── Case B: Non-divisible N=100, TILE=32 ─────────────────────────
    std::cout << "\n[Case B] Non-divisible 100x100, TILE=32\n";
    {
        const int N = 100;
        Tensor A({N, N}), B({N, N});
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
            {
                A({i, j}) = static_cast<float>((i * N + j) % 7 + 1);
                B({i, j}) = static_cast<float>((i * N + j) % 5 + 1);
            }

        Tensor C_naive = matmul_naive(A, B);
        Tensor C_tiled = matmul_tiled(A, B, 32);

        bool match = true;
        for (int i = 0; i < N && match; ++i)
            for (int j = 0; j < N && match; ++j)
                if (!nearly_equal(C_naive({i, j}), C_tiled({i, j}), 1e-4f))
                {
                    match = false;
                    std::cout << "Mismatch at (" << i << "," << j << "): naive=" << C_naive({i, j}) << " tiled=" << C_tiled({i, j}) << "\n";
                }

        std::cout << (match ? "[PASS]" : "[FAIL]") << " Tiled matches naive for non-divisible dimensions.\n";
    }

    // ── Case C: Small exact N=4, TILE=2 ──────────────────────────────
    std::cout << "\n[Case C] Small exact 4x4, TILE=2\n";
    {
        Tensor A({4, 4}), B({4, 4});
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
            {
                A({i, j}) = static_cast<float>(i * 4 + j + 1);
                B({i, j}) = static_cast<float>(j * 4 + i + 1);
            }

        Tensor C_naive = matmul_naive(A, B);
        Tensor C_tiled = matmul_tiled(A, B, 2);

        bool match = true;
        for (int i = 0; i < 4 && match; ++i)
            for (int j = 0; j < 4 && match; ++j)
                if (!nearly_equal(C_naive({i, j}), C_tiled({i, j}), 1e-4f))
                {
                    match = false;
                    std::cout << "Mismatch at (" << i << "," << j << "): naive=" << C_naive({i, j}) << " tiled=" << C_tiled({i, j}) << "\n";
                }

        std::cout << (match ? "[PASS]" : "[FAIL]") << " Tiled matches naive for 4x4 TILE=2.\n";
    }
}
void benchmark_naive_only()
{
    const int N = 2048;
    Tensor A({N, N}), B({N, N});
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
        {
            A({i, j}) = 1.0f;
            B({i, j}) = 1.0f;
        }

    auto start = std::chrono::high_resolution_clock::now();
    Tensor C = matmul_naive(A, B);
    auto end = std::chrono::high_resolution_clock::now();

    float time_ms = std::chrono::duration<float, std::milli>(end - start).count();
    float gflops = (2.0f * N * N * N) / (time_ms * 1e6f);

    std::cout << "N=" << N << " naive: " << time_ms << " ms, " << gflops << " GFLOPS\n";
}
void benchmark_matmul_only()
{
    const int N = 2048;
    Tensor A({N, N}), B({N, N});
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
        {
            A({i, j}) = 1.0f;
            B({i, j}) = 1.0f;
        }

    auto start = std::chrono::high_resolution_clock::now();
    Tensor C = matmul(A, B);
    auto end = std::chrono::high_resolution_clock::now();

    float time_ms = std::chrono::duration<float, std::milli>(end - start).count();
    float gflops = (2.0f * N * N * N) / (time_ms * 1e6f);

    std::cout << "N=" << N << " matmul: " << time_ms << " ms, " << gflops << " GFLOPS\n";
}
void benchmark_matmul_tiled_only()
{
    const int N =256;
    Tensor A({N, N}), B({N, N});
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
        {
            A({i, j}) = 1.0f;
            B({i, j}) = 1.0f;
        }

    auto start = std::chrono::high_resolution_clock::now();
    Tensor C = matmul_tiled(A, B,32);
    auto end = std::chrono::high_resolution_clock::now();

    float time_ms = std::chrono::duration<float, std::milli>(end - start).count();
    float gflops = (2.0f * N * N * N) / (time_ms * 1e6f);

    std::cout << "N=" << N << " matmul_tiled: " << time_ms << " ms, " << gflops << " GFLOPS\n";
}
// ─────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────
int main()
{
    std::cout << "╔══════════════════════════════════════╗\n";
    std::cout << "║      TENSOR ENGINE TEST SUITE        ║\n";
    std::cout << "╚══════════════════════════════════════╝\n";
    //benchmark_naive_only();
   //benchmark_matmul_only();
    benchmark_matmul_tiled_only();
    std::cout << "\n========================================\n";
    std::cout << "  All tests completed.\n";
    std::cout << "========================================\n";

    return 0;
}