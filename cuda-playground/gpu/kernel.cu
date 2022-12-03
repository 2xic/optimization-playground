#include <stdlib.h>
#include "kernel.h"
#include <stdio.h>

void print_array(float *ptr, int length)
{
    printf("[");
    for (int i = 0; i < length; i++)
    {
        if (i > 0)
        {
            printf(", ");
        }
        printf("%f", ptr[i]);
    }
    printf("]");
}

Matrix *createMatrixGpu(int rows, int columns)
{
    float *a_device;
    int SIZE = rows * columns;
    Matrix *a = (Matrix *)malloc(sizeof(Matrix));
    cudaMalloc(&a_device, SIZE * sizeof(float));
    a->data = a_device;
    a->rows = rows;
    a->columns = columns;
    a->device = 1;

    return a;
}

extern "C" void sendToHost(Matrix *m)
{
    float *c_host;
    int SIZE = m->rows * m->columns;
    c_host = (float *)malloc(SIZE * sizeof(float));

    cudaMemcpy(c_host, m->data, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(m->data);

    m->data = c_host;
    m->device = 0;
}

extern "C" void sendToGpu(Matrix *m)
{
    int SIZE = m->rows * m->columns;
    float *c_device;
    cudaMalloc(&c_device, SIZE * sizeof(float));

    cudaMemcpy(c_device, m->data, SIZE * sizeof(float), cudaMemcpyHostToDevice);
    free(m->data);

    m->data = c_device;
    m->device = 1;
}

extern "C" Matrix *GpuMatrixMatMul(Matrix *a, Matrix *b)
{
    Matrix *c = createMatrixGpu(a->rows, b->columns);
    dim3 dimBlock(a->rows, a->columns);
    dim3 dimGrid(1, 1);

    MatMul<<<dimGrid, dimBlock>>>(a->data, b->data, c->data, a->rows, b->columns);

    return c;
}

__global__ void setElement(float *data, int columns, int row, int col, float value)
{
    int rowIndex = columns * row;
    data[rowIndex + col] = value;
}

__device__ void getElement(float *data, int row, int colsize, int col, float *value)
{
    int row_idx = row * colsize;
    *value = data[row_idx + col];
}

__global__ void AdvancedMatrixOperator(float *matrix_a, float *results_matrix, int cols, int rows, DirectOperator operator_val)
{
    float a_item;

    auto getDataPointer = [](float *data, int columns, int row, int col)
    {
        int rowIndex = columns * row;
        float *pointer = &data[rowIndex + col];
        return pointer;
    };

    auto get = [&](int cols, int row, int col, float *M, float C, float *results)
    {
        if (M != NULL)
        {
            *results = *getDataPointer(M, cols, row, col);
        }
        else
        {
            *results = C;
        }
    };

    auto set = [&](float *data, int columns, int row, int col, float value)
    {
        float *results = getDataPointer(data, columns, row, col);
        *results = value;
    };

    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (operator_val == EXP)
    {
        get(cols, row, col, matrix_a, -1, &a_item);
        set(
            results_matrix,
            cols,
            row,
            col,
            a_item);
    }
    else if (operator_val == TRANSPOSE)
    {
        // TODO: something seems wrong here
        get(rows, col, row, matrix_a, 0xdeadbeef, &a_item);
        set(
            results_matrix,
            rows,
            row,
            col,
            a_item);
    }
    else
    {
        // Unknown operator
        return;
    }
}

__global__ void SimpleMatrixOperator(float *matrix_a, float *matrix_b, float constant_value, float *results_matrix, int cols, SimplePairWiseOperator operator_val, int value_direction)
{
    float a_item;
    float b_item;

    auto getDataPointer = [](float *data, int columns, int row, int col)
    {
        int rowIndex = columns * row;
        float *pointer = &data[rowIndex + col];
        return pointer;
    };

    auto get = [&](int cols, int row, int col, float *M, float C, float *res)
    {
        if (M != NULL)
        {
            *res = *getDataPointer(M, cols, row, col);
        }
        else
        {
            *res = C;
        }
    };

    auto set = [&](float *data, int columns, int row, int col, float value)
    {
        float *results = getDataPointer(data, columns, row, col);
        *results = value;
    };

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float value = 0;

    if (value_direction == 0)
    {
        get(cols, row, col, matrix_a, constant_value, &a_item);
        get(cols, row, col, matrix_b, constant_value, &b_item);
    }
    else
    {
        get(cols, row, col, matrix_b, constant_value, &a_item);
        get(cols, row, col, matrix_a, constant_value, &b_item);
    }

    if (operator_val == ADD)
    {
        value = a_item + b_item;
    }
    else if (operator_val == SUB)
    {
        value = a_item - b_item;
    }
    else if (operator_val == MUL)
    {
        value = a_item * b_item;
    }
    else if (operator_val == DIV)
    {
        value = a_item / b_item;
    }
    else
    {
        // Unknown opcode throw an error
        return;
    }

    set(
        results_matrix,
        cols,
        row,
        col,
        value);
}

extern "C" Matrix *GpuAdd(Matrix *a, Matrix *b)
{
    Matrix *c = createMatrixGpu(a->rows, b->columns);

    dim3 dimBlock(a->rows, b->columns);
    dim3 dimGrid(1, 1);

    SimpleMatrixOperator<<<dimGrid, dimBlock>>>(a->data, b->data, -1, c->data, b->columns, ADD, 0);

    return c;
}

extern "C" Matrix *GpuAddConstant(Matrix *a, float b, int direction)
{
    Matrix *c = createMatrixGpu(a->rows, a->columns);
    dim3 dimBlock(a->rows, a->columns);
    dim3 dimGrid(1, 1);

    SimpleMatrixOperator<<<dimGrid, dimBlock>>>(a->data, nullptr, b, c->data, a->columns, ADD, direction);

    return c;
}

extern "C" Matrix *GpuMul(Matrix *a, Matrix *b)
{
    Matrix *c = createMatrixGpu(a->rows, b->columns);
    dim3 dimBlock(a->rows, a->columns);
    dim3 dimGrid(1, 1);

    SimpleMatrixOperator<<<dimGrid, dimBlock>>>(a->data, b->data, -1, c->data, b->columns, MUL, 0);

    return c;
}

extern "C" Matrix *GpuMulConstant(Matrix *a, float b, int direction)
{
    Matrix *c = createMatrixGpu(a->rows, a->columns);
    dim3 dimBlock(a->rows, a->columns);
    dim3 dimGrid(1, 1);

    SimpleMatrixOperator<<<dimGrid, dimBlock>>>(a->data, nullptr, b, c->data, a->columns, MUL, direction);

    return c;
}

extern "C" Matrix *GpuDivideConstant(Matrix *a, float b, int direction)
{
    Matrix *c = createMatrixGpu(a->rows, a->columns);
    dim3 dimBlock(a->rows, a->columns);
    dim3 dimGrid(1, 1);

    SimpleMatrixOperator<<<dimGrid, dimBlock>>>(a->data, nullptr, b, c->data, a->columns, DIV, direction);

    return c;
}

extern "C" Matrix *GpuSubtract(Matrix *a, Matrix *b)
{
    Matrix *c = createMatrixGpu(a->rows, b->columns);
    dim3 dimBlock(a->rows, a->columns);
    dim3 dimGrid(1, 1);

    SimpleMatrixOperator<<<dimGrid, dimBlock>>>(a->data, b->data, -1, c->data, b->columns, SUB, 0);

    return c;
}

extern "C" Matrix *GpuSubtractConstant(Matrix *a, float b, int direction)
{
    Matrix *c = createMatrixGpu(a->rows, a->columns);
    dim3 dimBlock(a->rows, a->columns);
    dim3 dimGrid(1, 1);

    SimpleMatrixOperator<<<dimGrid, dimBlock>>>(a->data, nullptr, b, c->data, a->columns, SUB, direction);

    return c;
}


"""
Time usage :

EXP

Cpu
Time used : 0.01662468910217285
Gpu
Time used : 0.11232900619506836

Transpose

Cpu
Time used : 0.004513978958129883
Gpu
Time used : 0.06209087371826172

Matmul
Cpu
Time used : 0.006204366683959961
Gpu
Time used : 0.06139945983886719
"""


extern "C" Matrix *GpuTranspose(Matrix *a)
{
    Matrix *c = createMatrixGpu(a->columns, a->rows);
    dim3 dimBlock(a->rows, a->columns);
    dim3 dimGrid(1, 1);

    AdvancedMatrixOperator<<<dimGrid, dimBlock>>>(a->data, c->data, a->columns, a->rows, TRANSPOSE);

    return c;
}

extern "C" Matrix *GpuExp(Matrix *a)
{
    Matrix *c = createMatrixGpu(a->rows, a->columns);
    dim3 dimBlock(a->rows, a->columns);
    dim3 dimGrid(1, 1);

    AdvancedMatrixOperator<<<dimGrid, dimBlock>>>(a->data, c->data, a->columns, a->rows, EXP);

    return c;
}

// https://developer.nvidia.com/blog/cuda-dynamic-parallelism-api-principles/
// https://stackoverflow.com/questions/49687130/pass-by-reference-in-device-function-cuda
//  -> Looks like there is some improved synchronization I can do
__global__ void MatMul(float *a, float *b, float *c, int columns, int rows)
{
    float a_item;
    float b_item;

    auto getDataPointer = [](float *data, int columns, int row, int col)
    {
        int rowIndex = columns * row;
        float *pointer = &data[rowIndex + col];
        return pointer;
    };

    auto get = [&](int cols, int row, int col, float *M, float C, float *res)
    {
        if (M != NULL)
        {
            *res = *getDataPointer(M, cols, row, col);
        }
        else
        {
            *res = C;
        }
    };

    auto set = [&](float *data, int columns, int row, int col, float value)
    {
        float *results = getDataPointer(data, columns, row, col);
        *results = value;
    };

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float accumulator = 0;
    for (int current_colum = 0; current_colum < columns; current_colum++)
    {
        getElement(a, columns, row, current_colum, &a_item);
        getElement(b, columns, current_colum, col, &b_item);

        accumulator += a_item * b_item;
    }

    set(
        c,
        columns,
        row,
        col,
        accumulator);
}
