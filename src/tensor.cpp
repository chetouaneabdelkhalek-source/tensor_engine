#include "tensor.h"
#include <iostream>

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

Tensor::Tensor(Tensor &other) : shape(other.shape),
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
Tensor &Tensor::operator=(Tensor &other)
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

Tensor &Tensor::operator=(Tensor &&other) noexcept
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

Tensor Tensor::transpose()
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