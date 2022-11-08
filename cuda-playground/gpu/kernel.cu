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

__global__ void setElement(int *data, int columns, int row, int col, int value){
    int rowIndex = columns * row;
    data[rowIndex + col] = value;
}

// TODO: this should support any dimension.
/*
__global__ void createMatrix(void **device, int rows, int columns)
{
    cudaMalloc(device, sizeof(Matrix));
    void *deviceMatrix = *device;
    if (deviceMatrix == NULL)
    {
        return;
    }
    Matrix *matrix = (Matrix*)deviceMatrix;
    matrix->rows = 2;
    matrix->columns = 2;
    /*
    int size = rows * columns * sizeof(int *);
    int **datapointer = &matrix->data;
    int results = cudaMalloc(datapointer, size);
    if (results != cudaSuccess) {
        matrix->rows = -1;
    }
   // cudaMemset(datapointer, 0, size);
}
*/

__global__ void MatMul(int *a, int *b, int *c, int columns, int rows)
{
    printf("hello world \n");
    for (int row = 0; row < rows; row++)
    {
        for (int column = 0; column < columns; column++)
        {
            int accumulator = 0;
            for (int current_colum = 0; current_colum < columns; current_colum++)
            {
                // current row + current column
                int a_rowIndex = columns * row;
                int a_item = a[a_rowIndex + current_colum];

                // current column + column
                int b_rowIndex = columns * current_colum;
                int b_item = b[b_rowIndex + column];

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
