#pragma once
#include <vector>
#include <memory>
class Tensor
{
public:
    Tensor(std::vector<int> shape);
    ~Tensor();
    Tensor(const Tensor &);                     // copy constructor
    Tensor &operator=(const Tensor &);          // copy assignment
    Tensor(Tensor &&other) noexcept;            // move constructor
    Tensor &operator=(Tensor &&other) noexcept; // move assignment

    Tensor transpose() const;
    float &operator()(const std::vector<int> &coords);
    const float &operator()(const std::vector<int> &coords) const;
    Tensor softmax() const;

    friend Tensor matmul(const Tensor &A, const Tensor &B);

private:
    Tensor(std::vector<int> shape, std::vector<int> strideVector,
           std::shared_ptr<float[]> data, int dim, int size);

    std::shared_ptr<float[]> data;
    std::vector<int> strideVector;
    std::vector<int> shape;

    int dim;
    int size;
};