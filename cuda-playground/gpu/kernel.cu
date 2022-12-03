#include <stdlib.h>
#include "kernel.h"
#include <stdio.h>

__device__ float a_item;
__device__ float b_item;

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
    // printf("Sending it to host \n");
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
    //  printf("Sending it to host \n");
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
    MatMul<<<1, 1>>>(a->data, b->data, c->data, a->rows, b->columns);

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

__global__ void SimpleMatrixOperator(float *a, float *b, float constant, float *c, int rows, int cols, int operator_val)
{
    auto get = [](int cols, int i, int j, float *M, float C, float *res)
    {
        if (M != NULL)
        {
            getElement(M, cols, i, j, res);
        }
        else
        {
            *res = C;
        }
    };

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            float value = 0;

            if (operator_val == ADD)
            {
                get(cols, i, j, a, constant, &a_item);
                get(cols, i, j, b, constant, &b_item);
                value = a_item + b_item;
            }
            else if (operator_val == SUB)
            {
                get(cols, i, j, a, constant, &a_item);
                get(cols, i, j, b, constant, &b_item);
                value = a_item - b_item;
            }
            else if (operator_val == MUL)
            {
                get(cols, i, j, a, constant, &a_item);
                get(cols, i, j, b, constant, &b_item);
                value = a_item * b_item;
            }
            else if (operator_val == DIV)
            {
                get(cols, i, j, a, constant, &a_item);
                get(cols, i, j, b, constant, &b_item);
                value = a_item / b_item;
            }

            setElement<<<1, 1>>>(
                c,
                cols,
                i,
                j,
                value);
        }
    }
}

__global__ void FastSimpleMatrixAddOperator(float *a, float *b, float constant, float *c, int rows, int cols, int operator_val)
{
    auto get = [](int cols, int i, int j, float *M, float C, float *res)
    {
        if (M != NULL)
        {
            getElement(M, cols, i, j, res);
        }
        else
        {
            *res = C;
        }
    };
    auto set = [](float *data, int columns, int row, int col, float value)
    {
        int rowIndex = columns * row;
        data[rowIndex + col] = value;
    };

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    float value = 0;

    if (operator_val == ADD)
    {
        get(cols, i, j, a, constant, &a_item);
        get(cols, i, j, b, constant, &b_item);
        value = a_item + b_item;
    }

    set(
        c,
        cols,
        i,
        j,
        value);
}

extern "C" Matrix *GpuAdd(Matrix *a, Matrix *b)
{
    Matrix *c = createMatrixGpu(a->rows, b->columns);

    dim3 dimBlock(a->rows, b->columns);
    dim3 dimGrid(1, 1);

    FastSimpleMatrixAddOperator<<<dimGrid, dimBlock>>>(a->data, b->data, -1, c->data, a->rows, b->columns, ADD);
    //    SimpleMatrixOperator<<<1, 1>>>(a->data, b->data, -1, c->data, a->rows, b->columns, ADD);

    return c;
}

extern "C" Matrix *GpuAddConstant(Matrix *a, float b, int direction)
{
    Matrix *c = createMatrixGpu(a->rows, a->columns);
    SimpleMatrixOperator<<<1, 1>>>(a->data, nullptr, b, c->data, a->rows, a->columns, ADD);

    return c;
}

extern "C" Matrix *GpuMul(Matrix *a, Matrix *b)
{
    Matrix *c = createMatrixGpu(a->rows, b->columns);
    SimpleMatrixOperator<<<1, 1>>>(a->data, b->data, -1, c->data, a->rows, b->columns, MUL);

    return c;
}

extern "C" Matrix *GpuMulConstant(Matrix *a, float b, int direction)
{
    Matrix *c = createMatrixGpu(a->rows, a->columns);
    SimpleMatrixOperator<<<1, 1>>>(a->data, nullptr, b, c->data, a->rows, a->columns, MUL);

    return c;
}

extern "C" Matrix *GpuDivideConstant(Matrix *a, float b, int direction)
{
    Matrix *c = createMatrixGpu(a->rows, a->columns);
    SimpleMatrixOperator<<<1, 1>>>(a->data, nullptr, b, c->data, a->rows, a->columns, DIV);

    return c;
}

extern "C" Matrix *GpuSubtract(Matrix *a, Matrix *b)
{
    Matrix *c = createMatrixGpu(a->rows, b->columns);
    SimpleMatrixOperator<<<1, 1>>>(a->data, b->data, -1, c->data, a->rows, b->columns, SUB);

    return c;
}

extern "C" Matrix *GpuSubtractConstant(Matrix *a, float b, int direction)
{
    Matrix *c = createMatrixGpu(a->rows, a->columns);
    SimpleMatrixOperator<<<1, 1>>>(a->data, nullptr, b, c->data, a->rows, a->columns, SUB);

    return c;
}

__global__ void _transpose(float *target, float *source, int columns, int rows)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            getElement(source, columns, i, j, &a_item);
            //   printf("Hei! %f\n", a_item);
            setElement<<<1, 1>>>(target, columns, j, i, a_item);
        }
    }
}

extern "C" Matrix *GpuTranspose(Matrix *a)
{
    // printf("Test! (%i, %i)\n", a->columns, a->rows);

    Matrix *c = createMatrixGpu(a->columns, a->rows);
    _transpose<<<1, 1>>>(c->data, a->data, a->columns, a->rows);

    return c;
}

__global__ void _Exp(float *target, float *source, int columns, int rows)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            getElement(source, columns, i, j, &a_item);
            //   printf("Hei! %f\n", a_item);
            setElement<<<1, 1>>>(target, columns, j, i, exp(a_item));
        }
    }
}

extern "C" Matrix *GpuExp(Matrix *a)
{
    // printf("Test! (%i, %i)\n", a->columns, a->rows);

    Matrix *c = createMatrixGpu(a->columns, a->rows);
    _Exp<<<1, 1>>>(c->data, a->data, a->columns, a->rows);

    return c;
}

// Add the remaning operators = Victory :)

// https://developer.nvidia.com/blog/cuda-dynamic-parallelism-api-principles/
// https://stackoverflow.com/questions/49687130/pass-by-reference-in-device-function-cuda
//  -> Looks like there is some improved synchronization I can do
__device__ float accumulator;
__global__ void MatMul(float *a, float *b, float *c, int columns, int rows)
{
    //    printf("hello world \n");
    for (int row = 0; row < rows; row++)
    {
        for (int column = 0; column < columns; column++)
        {
            accumulator = 0;
            // float accumulator = 0;
            for (int current_colum = 0; current_colum < columns; current_colum++)
            {
                getElement(a, columns, row, current_colum, &a_item);
                getElement(b, columns, current_colum, column, &b_item);

                accumulator += a_item * b_item;
            }
            setElement<<<1, 1>>>(
                c,
                columns,
                row,
                column,
                accumulator);
        }
    }
}
