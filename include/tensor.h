#pragma once
#include <vector>

class Tensor
{
public:
    Tensor(std::vector<int> shape); // Constructor
    ~Tensor();                      // Desturctor
    Tensor(Tensor &);               // coppy assignment
    Tensor &operator=(Tensor &);
    Tensor(Tensor &&other) noexcept;
    Tensor &operator=(Tensor &&other) noexcept;
    Tensor transpose();
    float &operator()(std::vector<int> coords);

private:
    Tensor(std::vector<int> shape, std::vector<int> strideVector, float *data, int dim, int size ,int *alias_num);
    void cleanup();
    float *data;
    std::vector<int> strideVector;
    std::vector<int> shape;
    int *alias_num;
    int dim;
    int size;
};
