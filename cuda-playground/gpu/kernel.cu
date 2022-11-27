#include <stdlib.h>
#include "kernel.h"
#include <stdio.h>


__global__ void MatMul(float *a, float *b, float *c, int columns, int rows);

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

Matrix *createMatrixGpu(int rows, int columns) {
    // -> 
    printf("Sending it to GPU\n");
    float *a_device;
    int SIZE = rows * columns;
    Matrix *a = (Matrix*)malloc(sizeof(Matrix));
    cudaMalloc(&a_device, SIZE * sizeof(float));
    a->data = a_device;
    a->rows = rows;
    a->columns = columns;
    a->device = 1;

    return a;
}

extern "C" void sendToHost(Matrix *m) {
    printf("Sending it to host \n");
    float *c_host;
    int SIZE = m->rows * m->columns;
    c_host = (float*)malloc(SIZE * sizeof(float));

    cudaMemcpy(c_host, m->data, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(m->data);

    m->data = c_host;
    m->device = 0;
}

extern "C" void sendToGpu(Matrix *m) {
    printf("Sending it to host \n");
    int SIZE = m->rows * m->columns;
    float *c_device;
    cudaMalloc(&c_device, SIZE * sizeof(float));

    cudaMemcpy(c_device, m->data, SIZE * sizeof(float), cudaMemcpyHostToDevice);
    free(m->data);

    m->data = c_device;
    m->device = 1;
}

extern "C" Matrix* MatrixMatMul(Matrix*a, Matrix*b) {
    // Results
    Matrix *c = createMatrixGpu(a->rows, b->columns);
    MatMul<<<1, 1>>>(a->data, b->data, c->data, a->rows, b->columns);

    return c;
}

__global__ void setElement(float *data, int columns, int row, int col, float value){
    int rowIndex = columns * row;
    data[rowIndex + col] = value;
}


__global__ void MatMul(float *a, float *b, float *c, int columns, int rows)
{
    printf("hello world \n");
    for (int row = 0; row < rows; row++)
    {
        for (int column = 0; column < columns; column++)
        {
            float accumulator = 0;
            for (int current_colum = 0; current_colum < columns; current_colum++)
            {
                // current row + current column
                int a_rowIndex = columns * row;
                float a_item = a[a_rowIndex + current_colum];

                // current column + column
                int b_rowIndex = columns * current_colum;
                float b_item = b[b_rowIndex + column];

                accumulator += a_item * b_item;
            }
            setElement<<<1,1>>>(
                c,
                columns,
                row,
                column,
                accumulator
            );
        }
    }
}
