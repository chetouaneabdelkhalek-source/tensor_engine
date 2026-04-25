#include "tensor.h"
#include <iostream>
#include <cmath>

Tensor::Tensor(std::vector<int> shape) : shape(shape),
                                         dim(shape.size())
{
    int size = 1;
    // alocating the tensor
    for (int i = 0; i < dim; i++)
    {
        size *= shape[i];
    }
    this->data = new float[size]();

    // creating the stride vector
    std::vector<int> strideVector(dim);
    strideVector[dim - 1] = 1;
    for (int i = dim - 2; i >= 0; i--)
    {
        strideVector[i] = strideVector[i + 1] * shape[i + 1];
    }
    this->alias_num = new int(1);
    this->size = size;
    this->strideVector = strideVector;
}

Tensor::~Tensor()
{
    this->cleanup();
}

Tensor::Tensor(const Tensor &other) : shape(other.shape),
                                      dim(other.dim),
                                      size(other.size),
                                      strideVector(other.strideVector)
{
    data = new float[size]();
    this->alias_num = new int(1);
    for (int i = 0; i < size; i++)
    {
        data[i] = other.data[i];
    }
}
Tensor &Tensor::operator=(const Tensor &other) 
{
    if (this == &other)
        return *this;
    else
    {
        cleanup();

        this->strideVector = other.strideVector;
        this->dim = other.dim;
        this->size = other.size;
        this->shape = other.shape;

        this->alias_num = new int(1);
        data = new float[size]();
        for (int i = 0; i < size; i++)
        {
            data[i] = other.data[i];
        }
        return *this;
    }
}
Tensor::Tensor(Tensor &&other) noexcept : shape(std::move(other.shape)),
                                          dim(other.dim),
                                          size(other.size),
                                          strideVector(std::move(other.strideVector)),
                                          data(other.data),
                                          alias_num(other.alias_num)
{
    other.data = nullptr;
    other.alias_num = nullptr;
    other.dim = 0;
    other.size = 0;
}

Tensor &Tensor::operator=(Tensor &&other)  noexcept
{
    if (this == &other)
        return *this;
    else
    {
        this->cleanup();

        this->strideVector = std::move(other.strideVector);
        this->dim = other.dim;
        this->size = other.size;
        this->shape = std::move(other.shape);
        this->alias_num = other.alias_num;
        this->data = other.data;

        other.data = nullptr;
        other.alias_num = nullptr;
        other.dim = 0;
        other.size = 0;

        return *this;
    }
}

Tensor Tensor::transpose() const 
{
    std::vector<int> shape = this->shape;
    std::vector<int> strideVector = this->strideVector;
    int dim = this->dim;
    for (int i = 0; i < dim / 2; i++)
    {

        std::swap(shape[dim - i - 1], shape[i]);
        std::swap(strideVector[dim - i - 1], strideVector[i]);
    }

    return Tensor(shape, strideVector, this->data, this->dim, this->size, this->alias_num);
};

float &Tensor::operator()(std::vector<int> coords)
{

    if (coords.size() != dim)
    {
        throw std::out_of_range("Coordinate dimensions do not match tensor rank");
    }
    // from multi-dim to flat indexing
    int index = 0;
    for (int i = 0; i < dim; i++)
    {
        if (coords[i] >= shape[i] || coords[i] < 0)
        {
            throw std::out_of_range("Index out of bounds for dimension");
        }
        index += coords[i] * strideVector[i];
    }
    return data[index];
}
const float &Tensor::operator()(std::vector<int> coords) const
{
    if (coords.size() != dim)
        throw std::out_of_range("Coordinate dimensions do not match tensor rank");

    int index = 0;
    for (int i = 0; i < dim; i++)
    {
        if (coords[i] >= shape[i] || coords[i] < 0)
            throw std::out_of_range("Index out of bounds for dimension");
        index += coords[i] * strideVector[i];
    }
    return data[index];
}
Tensor Tensor::softmax() const 
{   
    if (size == 0)
        return Tensor(this->shape);
    // TODO: generalize to axis parameter
    if (this->dim != 1)
        throw std::invalid_argument("Can't calc softMax for tensor, we can only for vector. ");
    float max = this->data[0];
    int size = this->size;
    float sum = 1;
    Tensor softmaxVec(this->shape);

    for (int i = 1; i < size; i++)
    {
        float entry = this->data[i*strideVector[0]];
        if (entry > max)
        {
            sum = 1 + sum * std::exp(max - entry);
            max = entry;
        }
        else
        {
            sum += std::exp(entry - max);
        }
    }

    for (int i = 0; i < size; i++)
    {
        softmaxVec.data[i] = std::exp(this->data[i*strideVector[0]] - max) / sum;
    }
    return softmaxVec;
}
Tensor matmul(const Tensor &A, const Tensor &B) {
    
    if (A.dim != 2 || B.dim != 2) throw std::invalid_argument("Matmul requires 2D tensors.");
    if (A.shape[1] != B.shape[0]) throw std::invalid_argument("Dimension mismatch.");

    // THESE MUST REMAIN ACTIVE
    int M = A.shape[0], K = A.shape[1], N = B.shape[1];
    Tensor C({M, N});

    int Astride0 = A.strideVector[0], Astride1 = A.strideVector[1];
    int Bstride0 = B.strideVector[0], Bstride1 = B.strideVector[1];
    int Cstride0 = C.strideVector[0], Cstride1 = C.strideVector[1];

    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float a_val = A.data[i * Astride0 + k * Astride1];
            

            float* B_row = &B.data[k * Bstride0];
            float* C_row = &C.data[i * Cstride0];

            // --- THE BRANCHING OPTIMIZATION ---
            if (Bstride1 == 1 && Cstride1 == 1) {
                for (int j = 0; j < N; j++) {
                    C_row[j] += a_val * B_row[j]; // SIMD Fast Lane
                }
            } 
            else {
                for (int j = 0; j < N; j++) {
                    C_row[j * Cstride1] += a_val * B_row[j * Bstride1]; // Safe Fallback
                }
            }
        }
    }
    return C;
}
void Tensor::cleanup()
{
    if (alias_num != nullptr)
    {
        (*alias_num)--;
        if (*alias_num == 0)

        {
            delete[] data;
            delete alias_num;
        }
        alias_num = nullptr;
        data = nullptr;
    }
}

Tensor::Tensor(std::vector<int> shape, std::vector<int> strideVector, float *data, int dim, int size, int *alias_num) : shape(shape), strideVector(strideVector), data(data), alias_num(alias_num), dim(dim),
                                                                                                                        size(size)
{
    if (this->alias_num)
    {
        (*this->alias_num)++;
    }
}