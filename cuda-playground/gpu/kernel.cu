#include <stdlib.h>
#include "kernel.h"
#include <stdio.h>
/*




void freeMatrix( Matrix *a) {
    free(a->data);
    free(a);
}
*/

void print_array(int *ptr, int length)
{
    printf("[");
    for (int i = 0; i < length; i++)
    {
        if (i > 0)
        {
            printf(", ");
        }
        printf("%i", ptr[i]);
    }
    printf("]");
}

__global__ void setElement(Matrix *a, int row, int col, int value)
{
    int rowIndex = a->columns * row;
    a->data[rowIndex + col] = value;
}

__global__ void getElement(Matrix *a, int row, int col, int *value)
{
    int rowIndex = a->columns * row;
    *value = a->data[rowIndex + col];
}

// TODO: this should support any dimension.
__global__ void createMatrix(void **device, int rows, int columns)
{
    cudaMalloc((void **)&device, sizeof(Matrix));
    Matrix *matrix = (Matrix *)&device;
    if (device == NULL)
    {
        return;
    }
    matrix->rows = rows;
    matrix->columns = columns;

    int size = rows * columns * sizeof(int *);
    cudaMalloc((int **)matrix->data, size);
}

__global__ void MatMul(Matrix *a, Matrix *b, Matrix *c)
{
    printf("hello world \n");
    for (int row = 0; row < c->rows; row++)
    {
        for (int column = 0; column < c->columns; column++)
        {
            int accumulator = 0;
            for (int column_j = 0; column_j < c->columns; column_j++)
            {
                int *res_a;
                int *res_b;
                cudaMalloc((void**)&res_a, sizeof(int));
                getElement<<<1,1>>>(a, row, column_j, res_a);

                cudaMalloc((void**)&res_b, sizeof(int));
                getElement<<<1,1>>>(b, column_j, column, res_b);

/*               int res_b; 
                getElement<<<1,1>>>(b, column_j, column, &res_b);
*/
                printf("%d", *res_a);
                accumulator += *res_a * *res_b;
            }
/*
            setElement<<<1,1>>>(
                c,
                row,
                column,
                accumulator);
                */
        }
    }
}
