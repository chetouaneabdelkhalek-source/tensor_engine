#pragma once
#include <vector>

class Tensor
{
public:
    Tensor(std::vector<int> shape);
    ~Tensor();
    Tensor(const Tensor &);                  // copy constructor
    Tensor &operator=(const Tensor &);       // copy assignment
    Tensor(Tensor &&other) noexcept;         // move constructor
    Tensor &operator=(Tensor &&other) noexcept; // move assignment

    Tensor transpose() const;
    float &operator()(std::vector<int> coords);
    const float &operator()(std::vector<int> coords) const;
    Tensor softmax() const;

    friend Tensor matmul(const Tensor &A, const Tensor &B);

private:
    Tensor(std::vector<int> shape, std::vector<int> strideVector,
           float *data, int dim, int size, int *alias_num);
    void cleanup();

    float *data;
    std::vector<int> strideVector;
    std::vector<int> shape;
    // NOTE: alias_num is a non-atomic reference count.
    // Single-threaded only. For multi-threaded use: std::atomic<int>.
    int *alias_num;
    int dim;
    int size;
};