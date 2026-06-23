#include "tensor.h"
#include <iostream>
#include <cmath>
#include <memory>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <utility>

Tensor::Tensor(std::vector<int> shape) : shape(shape),
                                         dim(shape.size())
{
    int size = 1;
    // alocating the tensor
    for (int i = 0; i < dim; i++)
    {
        size *= shape[i];
    }
    this->data = data_initialization(size);

    // creating the stride vector
    std::vector<int> strideVector(dim);
    if (dim > 0)
    {
        strideVector[dim - 1] = 1;
        for (int i = dim - 2; i >= 0; i--)
        {
            strideVector[i] = strideVector[i + 1] * shape[i + 1];
        }
    }
    this->size = size;
    this->strideVector = strideVector;
}

Tensor::~Tensor()
{
    // shared_ptr clean automaticly
}

Tensor::Tensor(const Tensor &other) : shape(other.shape),
                                      dim(other.dim),
                                      size(other.size),
                                      strideVector(other.strideVector)
{
    data = data_initialization(size);

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

        this->strideVector = other.strideVector;
        this->dim = other.dim;
        this->size = other.size;
        this->shape = other.shape;

        data = data_initialization(size);
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
                                          data(std::move(other.data))

{

    other.dim = 0;
    other.size = 0;
}

Tensor &Tensor::operator=(Tensor &&other) noexcept
{
    if (this == &other)
        return *this;
    else
    {

        this->strideVector = std::move(other.strideVector);
        this->dim = other.dim;
        this->size = other.size;
        this->shape = std::move(other.shape);
        this->data = std::move(other.data);

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

    return Tensor(shape, strideVector, this->data, this->dim, this->size);
};

float &Tensor::operator()(const std::vector<int> &coords)
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
const float &Tensor::operator()(const std::vector<int> &coords) const
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
        float entry = this->data[i * strideVector[0]];
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
        softmaxVec.data[i] = std::exp(this->data[i * strideVector[0]] - max) / sum;
    }
    return softmaxVec;
}
Tensor Tensor::softmax_naive() const
{
    Tensor out(this->shape);
    float sum = 0;
    for (int i = 0; i < size; i++)
        sum += std::exp(data[i * strideVector[0]]);
    for (int i = 0; i < size; i++)
        out.data[i] = std::exp(data[i * strideVector[0]]) / sum;
    return out;
}
Tensor matmul(const Tensor &A, const Tensor &B)
{

    if (A.dim != 2 || B.dim != 2)
        throw std::invalid_argument("Matmul requires 2D tensors.");
    if (A.shape[1] != B.shape[0])
        throw std::invalid_argument("Dimension mismatch.");

    int M = A.shape[0], K = A.shape[1], N = B.shape[1];
    Tensor C({M, N});

    int Astride0 = A.strideVector[0], Astride1 = A.strideVector[1];
    int Bstride0 = B.strideVector[0], Bstride1 = B.strideVector[1];
    int Cstride0 = C.strideVector[0], Cstride1 = C.strideVector[1];
    if (Bstride1 == 1)
    {
        for (int i = 0; i < M; i++)
        {
            for (int k = 0; k < K; k++)
            {
                float a_val = A.data[i * Astride0 + k * Astride1];

                float *B_row = &B.data[k * Bstride0];
                float *C_row = &C.data[i * Cstride0];

                {
                    for (int j = 0; j < N; j++)
                    {
                        C_row[j] += a_val * B_row[j]; // SIMD Fast Lane
                    }
                }
            }
        }
    }
    else
    {

        for (int i = 0; i < M; i++)
        {
            float *A_row = &A.data[i * Astride0];
            for (int j = 0; j < N; j++)
            {
                float sum = 0.0f;

                for (int k = 0; k < K; k++)
                {
                    sum += A_row[k * Astride1] * B.data[k * Bstride0 + j * Bstride1];
                }
                C.data[i * Cstride0 + j * Cstride1] = sum;
            }
        }
    }
    return C;
}

Tensor matmul_naive(const Tensor &A, const Tensor &B)
{
    int M = A.shape[0], K = A.shape[1], N = B.shape[1];
    Tensor C({M, N});
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            for (int k = 0; k < K; k++)
            {
                // Bypass the std::vector creation overhead
                C.data[i * C.strideVector[0] + j * C.strideVector[1]] +=
                    A.data[i * A.strideVector[0] + k * A.strideVector[1]] * B.data[k * B.strideVector[0] + j * B.strideVector[1]];
            }
        }
    }
    return C;
}
Tensor matmul_tiled(const Tensor &A, const Tensor &B, int TILE)
{

    // This is not build yet to hendle Transposed
    if (A.dim != 2 || B.dim != 2)
        throw std::invalid_argument("Matmul requires 2D tensors.");
    if (A.shape[1] != B.shape[0])
        throw std::invalid_argument("Dimension mismatch.");

    int M = A.shape[0], K = A.shape[1], N = B.shape[1];
    Tensor C({M, N});

    int Astride0 = A.strideVector[0], Astride1 = A.strideVector[1];
    int Bstride0 = B.strideVector[0], Bstride1 = B.strideVector[1];
    int Cstride0 = C.strideVector[0], Cstride1 = C.strideVector[1];

    for (int ii = 0; ii < M; ii += TILE)
    {
        for (int jj = 0; jj < N; jj += TILE)
        {
            for (int kk = 0; kk < K; kk += TILE)
            {
                for (int i = ii; i < std::min(M, ii + TILE); i++)
                {
                    
                    for (int k = kk; k < std::min(K, kk + TILE); k++)

                    {
                        float a_val = A.data[i * Astride0 + k * Astride1];

                        float *B_row = &B.data[k * Bstride0];
                        float *C_row = &C.data[i * Cstride0];
                        for (int j = jj; j < std::min(N, jj + TILE); j++)
                        {

                            C_row[j] += a_val * B_row[j]; // SIMD Fast Lane
                        }
                    }
                }
            }
        }
    }
    return C;
}

Tensor::Tensor(std::vector<int> shape, std::vector<int> strideVector, std::shared_ptr<float[]> data, int dim, int size) : shape(shape), strideVector(strideVector), data(data), dim(dim),
                                                                                                                          size(size)
{
}
std::shared_ptr<float[]> Tensor::data_initialization(int size)
{
    if (size <= 0)
        return nullptr;
    size_t aligned_bytes = ((size * sizeof(float) + 31) / 32) * 32;
    float *raw_ptr = nullptr;
#ifdef _MSC_VER
    // Windows (MSVC)
    raw_ptr = static_cast<float *>(_aligned_malloc(aligned_bytes, 32));
    if (!raw_ptr)
        throw std::bad_alloc();
    std::memset(raw_ptr, 0, aligned_bytes);
    return std::shared_ptr<float[]>(raw_ptr, [](float *p)
                                    { _aligned_free(p); });
#else
    // Linux / macOS (GCC / Clang)
    raw_ptr = static_cast<float *>(std::aligned_alloc(32, aligned_bytes));
    if (!raw_ptr)
        throw std::bad_alloc();
    std::memset(raw_ptr, 0, aligned_bytes);
    return std::shared_ptr<float[]>(raw_ptr, std::free);
#endif
}