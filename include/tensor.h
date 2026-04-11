#pragma once
#include <vector>

class Tensor
{
public:
    Tensor(std::vector<int> shape); // Constructor
    ~Tensor(); // Desturctor
    Tensor(Tensor &); // coppy assignment
    Tensor& operator=(Tensor&); 
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    
    float &operator()(std::vector<int> coords);

private:
    float *data;

    std::vector<int> strideVector;
    std::vector<int> shape;
    int dim;
    int size; 
};

